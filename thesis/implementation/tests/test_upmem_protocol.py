from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import scripts.research_benchmark_pack as report_pack_module
import quantum_bench.bench.upmem_hardware_taskgraph_resident as resident_runner
from quantum_bench.bench.generic_task_bridge import run_generic_task_bridge
from quantum_bench.bench.upmem_hardware_taskgraph_resident import (
    prepare_upmem_hardware_taskgraph_resident,
)
from quantum_bench.core.records import TensorSpec, TensorValue
from quantum_bench.formats import FixedPointSpec
from quantum_bench.routing import (
    GenericTaskPreparationCaps,
    GenericTaskPreparationInput,
    generic_loop_reference_float32,
    prepare_generic_task,
)
from quantum_bench.routing.generic_numeric_contract import classify_numeric
from quantum_bench.routing.generic_prepare import generic_structural_feasibility
from quantum_bench.targets.upmem.generic_boundary import (
    GENERIC_BOUNDARY_CASE_ID,
    build_generic_boundary_workload,
)
from quantum_bench.targets.upmem.generic_bridge import (
    execute_generic_bridge,
    read_generic_bridge_output_manifest,
    write_generic_bridge_input_manifest,
)
from quantum_bench.targets.upmem.hardware_session import (
    HardwareSessionBuild,
    ResidentGraphSessionExecution,
    hardware_environment_metadata,
    sanitised_hardware_environment,
    _resident_response_valid,
)
from quantum_bench.targets.upmem.runtime_checks import (
    strict_upmem_runtime_assertions,
    upmem_sdk_simulator_preflight_payload,
)
from quantum_bench.targets.upmem.runtime_evidence import transfer_accounting
from quantum_bench.targets.upmem.taskgraph_runtime import execute_upmem_taskgraph_runtime
from quantum_bench.tn import execute_task_sequence_np_einsum

from .support import (
    contraction_task,
    record_with_updates,
    resident_package_fixture,
    valid_resident_response,
)


def test_rank_selector_absent_preserves_default_profile_behavior() -> None:
    environment = {
        "DPU_PROFILE": "simulator",
        "SIMPLEPIM_BACKEND": "simulator",
        "UPMEM_BACKEND": "simulator",
        "UPMEM_MODE": "simulator",
        "UPMEM_TARGET": "simulator",
        "UPMEM_EXECUTION_MODE": "simulator",
        "UPMEM_PROFILE": "backend=simulator",
        "UPMEM_PROFILE_BASE": "simulator",
        "DPU_BACKEND": "simulator",
    }

    child = sanitised_hardware_environment(environment)

    assert "UPMEM_PROFILE" not in child
    assert "UPMEM_PROFILE_BASE" not in child
    assert "DPU_BACKEND" not in child
    for name in (
        "DPU_PROFILE",
        "SIMPLEPIM_BACKEND",
        "UPMEM_BACKEND",
        "UPMEM_MODE",
        "UPMEM_TARGET",
        "UPMEM_EXECUTION_MODE",
    ):
        assert name not in child
    assert hardware_environment_metadata(environment) == {
        "upmem_rank_path_requested": None,
        "upmem_sdk_profile_effective": None,
    }


def test_rank_selector_sets_only_the_project_controlled_sdk_profile() -> None:
    environment = {
        "UPMEM_HW_RANK_PATH": "/dev/dpu_rank20",
        "DPU_PROFILE": "simulator",
        "SIMPLEPIM_BACKEND": "simulator",
        "UPMEM_BACKEND": "simulator",
        "UPMEM_MODE": "simulator",
        "UPMEM_TARGET": "simulator",
        "UPMEM_EXECUTION_MODE": "simulator",
        "UPMEM_PROFILE": "backend=simulator",
        "UPMEM_PROFILE_BASE": "simulator",
        "DPU_BACKEND": "simulator",
    }

    child = sanitised_hardware_environment(environment)

    assert child["UPMEM_PROFILE"] == (
        "backend=hw,rankPath=/dev/dpu_rank20,ignoreVpd=true"
    )
    assert "UPMEM_HW_RANK_PATH" not in child
    assert "UPMEM_PROFILE_BASE" not in child
    assert "DPU_BACKEND" not in child
    for name in (
        "DPU_PROFILE",
        "SIMPLEPIM_BACKEND",
        "UPMEM_BACKEND",
        "UPMEM_MODE",
        "UPMEM_TARGET",
        "UPMEM_EXECUTION_MODE",
    ):
        assert name not in child
    assert hardware_environment_metadata(environment) == {
        "upmem_rank_path_requested": "/dev/dpu_rank20",
        "upmem_sdk_profile_effective": child["UPMEM_PROFILE"],
    }


@pytest.mark.parametrize(
    "rank_path",
    ("", "/dev/dpu_rank", "/dev/dpu_rank-1", "/dev/dpu_rank1/", "/tmp/dpu_rank1"),
)
def test_rank_selector_rejects_invalid_values_before_native_execution(
    rank_path: str,
) -> None:
    with pytest.raises(ValueError, match="hardware_profile_violation: UPMEM_HW_RANK_PATH"):
        sanitised_hardware_environment({"UPMEM_HW_RANK_PATH": rank_path})


def _prepared_input(*, quantization_mode: str = "per_task_input_quantize") -> GenericTaskPreparationInput:
    task = contraction_task("generic", shape=(2, 3, 2))
    left = np.array([[0.1, -0.2, 0.3], [0.4, -0.5, 0.6]], dtype=np.float64)
    right = np.array([[0.2, 0.3], [-0.4, 0.5], [0.6, -0.7]], dtype=np.float64)
    return GenericTaskPreparationInput(
        task=task,
        left_tensor=TensorValue(TensorSpec("generic_left", task.left_labels, left.shape, "dense", dtype="float64"), left),
        right_tensor=TensorValue(TensorSpec("generic_right", task.right_labels, right.shape, "dense", dtype="float64"), right),
        quantization_mode=quantization_mode,
    )


@pytest.mark.parametrize(
    ("value", "kind"),
    [
        (np.array([1, 2], dtype=np.int64), "real"),
        (np.array([1.0 + 0.0j], dtype=np.complex128), "complex_zero_imag"),
        (np.array([1.0 + 2.0j], dtype=np.complex128), "complex_nonzero"),
        (np.array([np.nan], dtype=np.float64), "nonfinite"),
    ],
)
def test_numeric_contract_classifies_without_discarding_complexity(value: np.ndarray, kind: str) -> None:
    result = classify_numeric(value)

    assert result.kind == kind
    assert result.has_nonfinite is (kind == "nonfinite")
    assert result.is_complex is (kind in {"complex_zero_imag", "complex_nonzero"})


def test_transfer_accounting_requires_directional_total_and_marks_bus_unobserved() -> None:
    result = transfer_accounting(
        64,
        24,
        declared_total_bytes=88,
        prepared_h2d_bytes=48,
        prepared_d2h_bytes=16,
        control_bytes=16,
        alignment_padding_bytes=8,
    )

    assert result["actual_transfer_bytes"] == result["actual_h2d_bytes"] + result["actual_d2h_bytes"]
    assert result["actual_transfer_bytes_invariant"] == "passed"
    assert result["physical_bus_bytes_available"] is False
    assert result["transfer_components"]["control_structure_bytes"] == 16

    with pytest.raises(ValueError, match="invariant failed"):
        transfer_accounting(64, 24, declared_total_bytes=87)


@pytest.mark.parametrize(
    ("task", "caps", "reason"),
    [
        (replace(contraction_task("rank"), input_shapes=((1,) * 17, (1, 1)), output_shape=(1,)), GenericTaskPreparationCaps(), "rank_cap_exceeded"),
        (replace(contraction_task("elements"), input_shapes=((257, 257), (257, 1)), output_shape=(257, 1)), GenericTaskPreparationCaps(), "element_count_cap_exceeded"),
        (contraction_task("contracted", shape=(1, 5000, 1)), GenericTaskPreparationCaps(max_contracted_combinations=4096), "contracted_combination_cap_exceeded"),
        (contraction_task("overflow", shape=(1, 150000, 1)), GenericTaskPreparationCaps(max_tensor_elements=200000, max_contracted_combinations=200000), "int32_accumulation_overflow_risk"),
        (replace(contraction_task("labels"), contracted_labels=(9,)), GenericTaskPreparationCaps(), "label_mapping_invalid"),
    ],
)
def test_generic_structural_feasibility_has_stable_rejection_boundaries(task, caps, reason: str) -> None:
    result = generic_structural_feasibility(task, caps)

    assert result.feasible is False
    assert result.reason == reason


def test_generic_preparation_is_json_safe_and_matches_float32_loop() -> None:
    result = prepare_generic_task(_prepared_input(quantization_mode="none"))

    assert result.status == "prepared"
    assert result.metadata["input_dtype_on_dpu"] == "float32"
    assert result.metadata["accumulator_dtype_on_dpu"] == "float32"
    assert result.metadata["scaling_applied"] is False
    assert result.validation_metrics["max_abs_error"] == 0.0
    assert json.dumps(result.to_json_dict())

    operands = result.prepared_operands
    assert operands is not None
    expected = generic_loop_reference_float32(
        operands.left_operand,
        operands.right_operand,
        output_shape=result.output_shape,
        left_strides=result.left_strides,
        right_strides=result.right_strides,
        output_strides=result.output_strides,
        output_to_left_axes=result.output_to_left_axes,
        output_to_right_axes=result.output_to_right_axes,
        contracted_to_left_axes=result.contracted_to_left_axes,
        contracted_to_right_axes=result.contracted_to_right_axes,
        contracted_dims=result.contracted_dims,
    )
    np.testing.assert_array_equal(operands.expected_reference_output, expected)


def test_generic_preparation_rejects_nonfinite_complex_and_wrong_dtype() -> None:
    prepared = _prepared_input()
    complex_left = np.ones(prepared.left_tensor.array.shape, dtype=np.complex128)
    complex_left[0, 0] += 1.0j
    complex_input = replace(
        prepared,
        left_tensor=TensorValue(prepared.left_tensor.spec, complex_left),
    )
    assert prepare_generic_task(complex_input).reason == "complex_generic_loop_not_implemented"

    nonfinite_input = replace(
        prepared,
        left_tensor=TensorValue(prepared.left_tensor.spec, np.full(prepared.left_tensor.array.shape, np.nan)),
    )
    assert prepare_generic_task(nonfinite_input).reason == "nonfinite_values_not_supported"

    bad_dtype = replace(prepared, fixed_point_spec=FixedPointSpec(route_dtype="int16"))
    assert prepare_generic_task(bad_dtype).reason == "unsupported_dtype"


def test_generic_boundary_reference_is_non_gemm_and_exact() -> None:
    workload = build_generic_boundary_workload()
    reference = execute_task_sequence_np_einsum(workload.graph, workload.network)[0]

    assert workload.case_id == GENERIC_BOUNDARY_CASE_ID
    assert workload.graph.tasks[0].structure == "generic_boundary"
    np.testing.assert_allclose(
        reference,
        np.einsum("abc,cde->abde", workload.network.tensors[0].array, workload.network.tensors[1].array),
    )
    assert workload.manifest["input_ranks"] == (3, 3)


def test_generic_bridge_rejects_path_escape_and_never_claims_disabled_execution(tmp_path: Path) -> None:
    preparation = prepare_generic_task(_prepared_input())
    write_generic_bridge_input_manifest(preparation, tmp_path)
    payload = json.loads((tmp_path / "input_manifest.json").read_text(encoding="utf-8"))
    payload["operands"]["left"]["relative_path"] = "../escape.npy"
    (tmp_path / "input_manifest.json").write_text(json.dumps(payload), encoding="utf-8")

    rejected = execute_generic_bridge(tmp_path / "input_manifest.json", execute_external=False)
    assert rejected.execution_status == "failed"
    assert rejected.reason == "input_manifest_invalid"
    assert rejected.external_command_executed is False
    assert read_generic_bridge_output_manifest(tmp_path / "output_manifest.json").status == "failed"

    preparation_dir = tmp_path / "disabled"
    preparation_dir.mkdir()
    write_generic_bridge_input_manifest(preparation, preparation_dir)
    disabled = execute_generic_bridge(preparation_dir / "input_manifest.json", execute_external=False)
    assert disabled.execution_status == "not_implemented"
    assert disabled.execution_implemented is False
    assert disabled.external_command_executed is False


def test_generic_task_bridge_public_harness_is_skipped_without_external_execution(tmp_path: Path) -> None:
    result = run_generic_task_bridge(tmp_path, case="bell_2q", execute_external=False)

    assert result.status == "skipped"
    assert result.reason == "generic_external_execution_disabled"
    assert result.external_command_executed is False
    assert result.summary["cpu_fallback_used"] is False
    assert result.summary["dpu_program_invocations"] == 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"policy": "unsupported-policy"},
        {"quantization_mode": "unsupported-mode"},
        {"schedule_mode": "frontier"},
        {"dpu_group_count": 0},
    ],
)
def test_taskgraph_runtime_rejects_unsupported_modes_without_cpu_fallback(minimal_graph, tmp_path: Path, kwargs: dict[str, object]) -> None:
    result = execute_upmem_taskgraph_runtime(
        graph=minimal_graph.graph,
        network=minimal_graph.network,
        case_id="fixture",
        policy=kwargs.pop("policy", "generic-only"),
        quantization_mode=kwargs.pop("quantization_mode", "per_task_input_quantize"),
        bridge_root=tmp_path,
        execute_external=False,
        **kwargs,
    )

    assert result.status == "unsupported"
    assert result.summary["cpu_fallback_used"] is False
    assert result.output is None


def test_taskgraph_runtime_requires_external_sdk_execution(minimal_graph, tmp_path: Path) -> None:
    result = execute_upmem_taskgraph_runtime(
        graph=minimal_graph.graph,
        network=minimal_graph.network,
        case_id="fixture",
        policy="generic-only",
        quantization_mode="per_task_input_quantize",
        bridge_root=tmp_path,
        execute_external=False,
    )

    assert result.reason == "external_execution_required"
    assert result.summary["upmem_execution_mode"] == "sdk_simulator"
    assert result.summary["hardware_speedup_applicable"] is False


def test_strict_runtime_assertions_and_preflight_expose_failure_stage() -> None:
    passing = strict_upmem_runtime_assertions(
        {
            "total_tasks": 3,
            "dpu_program_executed_task_count": 3,
            "dpu_program_executed_all_tasks": True,
            "cpu_fallback_used": False,
            "native_sdk_control_path": True,
            "simplepim_api_used": False,
        }
    )
    failing = strict_upmem_runtime_assertions({"total_tasks": 3, "dpu_program_executed_task_count": 2, "cpu_fallback_used": True})

    assert passing["status"] == "passed"
    assert failing["status"] == "failed"
    assert "cpu_fallback_task_count_zero" in failing["reason"]
    assert upmem_sdk_simulator_preflight_payload("skipped", "sdk_missing")["required_conditions"]["upmem_sdk_present"] is False


@pytest.mark.parametrize(
    "updates",
    [
        {"simulator_kernel_executed": True},
        {"cpu_fallback_used": True},
        {"release_confirmed": False},
        {"hardware_release_verified": False},
        {"actual_transfer_bytes": 1},
        {"failure_stage": "release_failed"},
        {"tasklets": 2},
    ],
)
def test_resident_response_validator_rejects_unsafe_evidence(minimal_graph, resident_hardware_suite, tmp_path: Path, updates: dict[str, object]) -> None:
    _, manifest = resident_package_fixture(minimal_graph, tmp_path)
    response = valid_resident_response(manifest)
    assert _resident_response_valid(response, manifest, resident_hardware_suite.profile)
    assert not _resident_response_valid(record_with_updates(response, **updates), manifest, resident_hardware_suite.profile)


def test_resident_variant_fake_native_session_enforces_opt_in_and_projects_contract(
    minimal_graph, resident_hardware_suite, monkeypatch, tmp_path: Path
) -> None:
    reference, _ = execute_task_sequence_np_einsum(minimal_graph.graph, minimal_graph.network)
    session_root = tmp_path / "native_session"
    session_root.mkdir()
    dpu_binary = session_root / "dpu_resident"
    dpu_binary.write_bytes(b"fake-dpu")
    native_build = HardwareSessionBuild(
        session_root=session_root,
        source_snapshot=session_root,
        build_dir=session_root,
        host_binary=session_root / "host",
        dpu_binary=dpu_binary,
        source_tree_hash="source",
        host_binary_hash="host",
        dpu_binary_hash="dpu",
        build_time_s=0.0,
        build_command=("fake-build",),
        sdk_tools={"fake": "fixture"},
    )
    mismatch = False
    captured_manifests: list[dict[str, object]] = []

    def fake_native_session(build, *, manifest_path, response_path, profile, environment):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        captured_manifests.append(manifest)
        output = np.asarray(reference).copy()
        if mismatch:
            output.flat[0] += 1.0
        for item in manifest["final_outputs"]:
            component = str(item["component"])
            values = output.imag if component == "imag" else output.real
            np.asarray(values, dtype="<f4").ravel().tofile(build.session_root / str(item["output_path"]))
        response = valid_resident_response(
            manifest, steady_state_graph_execution_s=0.001
        )
        response_path.write_text(json.dumps(response), encoding="utf-8")
        return ResidentGraphSessionExecution(
            status="completed",
            failure_stage=None,
            response_path=response_path,
            response=response,
            process_time_s=0.0,
            command=("fake-native",),
            stdout_snippet="",
            stderr_snippet="",
        )

    monkeypatch.setattr(resident_runner, "execute_resident_graph_session", fake_native_session)
    kwargs = {
        "root_dir": tmp_path,
        "run_dir": tmp_path / "run",
        "native_build": native_build,
        "profile": resident_hardware_suite.profile,
        "suite_id": "fixture_resident",
        "case_id": "fixture",
        "variant_id": "fixture_variant",
        "graph": minimal_graph.graph,
        "network": minimal_graph.network,
        "reference_output": reference,
        "quantization_mode": "none",
        "environment": {"UPMEM_ALLOW_PHYSICAL_HARDWARE": "1"},
    }

    with pytest.raises(ValueError, match="UPMEM_ALLOW_PHYSICAL_HARDWARE=1"):
        resident_runner.execute_resident_variant(**{**kwargs, "request_id": "missing-opt-in", "environment": {}})
    assert captured_manifests == []

    execution = resident_runner.execute_resident_variant(**{**kwargs, "request_id": "valid-request"})

    assert execution.status == "completed"
    assert execution.summary["policy_reference_validation"]["passed"] is True
    assert execution.summary["full_precision_accuracy"]["passed"] is True
    assert execution.summary["release_confirmed"] is True
    assert execution.summary["physical_dependency_chain_verified"] is True
    assert execution.summary["execution_plan_hash"] == execution.summary["execution_plan"]["execution_plan_hash"]
    assert execution.summary["provider_id"] == "upmem_resident_hardware"
    assert execution.summary["provider_metadata"]["backend_id"] == resident_hardware_suite.profile.backend_id
    assert execution.summary["execution_plan_provenance"] == "host_declared"
    assert execution.summary["execution_plan_native_package_binding"] == "not_native_package_bound"
    assert execution.summary["resource_context"]["allocation_status"] == "verified"
    assert execution.summary["resource_context"]["allocated_dpu_count"] == 1
    assert execution.summary["resource_context"]["allocated_tasklets_per_dpu"] == 1
    assert execution.summary["execution_contract_status"] == "passed"
    assert execution.summary["policy_reference_status"] == "passed"
    assert execution.summary["full_precision_accuracy_status"] == "passed"
    assert execution.summary["scientific_validation_status"] == "passed"
    assert execution.summary["execution_plan"]["validation"]["scientific_validation_status"] == "passed"
    manifest = captured_manifests[0]
    expected_h2d = (
        int(manifest["initial_h2d_bytes"])
        + int(manifest["descriptor_h2d_bytes"])
        + int(manifest["control_h2d_bytes"])
    )
    expected_d2h = int(manifest["final_d2h_bytes"])
    assert execution.summary["actual_h2d_bytes"] == expected_h2d
    assert execution.summary["actual_d2h_bytes"] == expected_d2h
    assert execution.summary["actual_transfer_bytes"] == expected_h2d + expected_d2h
    assert execution.summary["actual_transfer_bytes_invariant"] == "passed"

    quantized_plan, quantized_resources = resident_runner._resident_execution_plan(
        minimal_graph.graph,
        resident_hardware_suite.profile,
        quantization_mode="per_task_resident_requantize",
    )
    assert quantized_plan.numeric.input_dtype == "int8"
    assert quantized_plan.numeric.accumulator_dtype == "int32"
    assert quantized_resources.allocation_status == "not_run"

    normalized = resident_runner._normalized_record(
        {"case_id": "fixture"},
        resident_hardware_suite,
        "fixture_variant",
        "none",
        0,
        0,
        execution,
        tmp_path / "run",
    )
    assert normalized["suite_id"] == resident_hardware_suite.suite["suite_id"]
    assert normalized["persistent_session_reused"] is False
    assert normalized["execution_plan_provenance"] == "host_declared"
    assert normalized["execution_plan_native_package_binding"] == "not_native_package_bound"
    assert report_pack_module._is_valid_one_dpu_record(normalized)
    run_resources = resident_runner._resident_run_resource_context(
        resident_hardware_suite.profile, [normalized]
    )
    assert run_resources.allocation_status == "verified"
    assert run_resources.requested_dpu_count == 1
    assert run_resources.allocated_dpu_count == 1
    assert run_resources.requested_tasklets_per_dpu == 1
    assert run_resources.allocated_tasklets_per_dpu == 1
    failed_record = resident_runner._failure_record(
        resident_hardware_suite,
        "native_build_failed",
        {"case_id": "fixture"},
        environment={"UPMEM_HW_RANK_PATH": "/dev/dpu_rank20"},
    )
    assert failed_record["suite_id"] == resident_hardware_suite.suite["suite_id"]
    assert failed_record["upmem_rank_path_requested"] == "/dev/dpu_rank20"
    assert failed_record["upmem_sdk_profile_effective"] == (
        "backend=hw,rankPath=/dev/dpu_rank20,ignoreVpd=true"
    )

    inaccurate_reference = np.asarray(reference).copy()
    inaccurate_reference.flat[0] += 1.0
    accuracy_failed = resident_runner.execute_resident_variant(
        **{
            **kwargs,
            "request_id": "full-precision-mismatch-request",
            "reference_output": inaccurate_reference,
        }
    )
    assert accuracy_failed.status == "completed"
    assert accuracy_failed.summary["policy_reference_status"] == "passed"
    assert accuracy_failed.summary["full_precision_accuracy_status"] == "failed"
    assert accuracy_failed.summary["scientific_validation_status"] == "failed"
    assert accuracy_failed.summary["execution_plan"]["validation"]["scientific_validation_status"] == "failed"
    # This field deliberately retains its historical policy-reference/transfer meaning.
    assert accuracy_failed.summary["validation_status"] == "passed"

    mismatch = True
    failed = resident_runner.execute_resident_variant(**{**kwargs, "request_id": "mismatch-request"})
    assert failed.status == "failed"
    assert failed.summary["policy_reference_validation"]["passed"] is False


def test_resident_prepare_only_has_one_dpu_profile_and_no_allocation(tmp_path: Path) -> None:
    from .support import RESIDENT_SUITE_PATH

    result = prepare_upmem_hardware_taskgraph_resident(
        tmp_path,
        suite_path=RESIDENT_SUITE_PATH,
        build=False,
        environment={},
    )
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))

    assert result.status == "prepared"
    assert summary["profile"]["requested_dpu_count"] == 1
    assert summary["profile"]["tasklets_per_dpu"] == 1
    assert summary["dpu_allocation_attempted"] is False
    assert summary["dpu_launch_attempted"] is False
    assert summary["provider_id"] == "upmem_resident_hardware"
    assert summary["prepared_cases"][0]["path_variants"][0]["execution_plan_hash"]
    assert summary["prepared_cases"][0]["path_variants"][0]["execution_plan_provenance"] == "host_declared"
    assert (
        summary["prepared_cases"][0]["path_variants"][0][
            "execution_plan_native_package_binding"
        ]
        == "not_native_package_bound"
    )
    assert summary["resource_context"]["allocation_status"] == "not_run"
    assert summary["resource_context"]["allocated_dpu_count"] is None
    assert summary["resource_context"]["allocated_tasklets_per_dpu"] is None
    resource_context = summary["prepared_cases"][0]["path_variants"][0]["resource_context"]
    assert resource_context["allocation_status"] == "not_run"
    assert resource_context["requested_dpu_count"] == 1
    assert resource_context["requested_tasklets_per_dpu"] == 1
    assert resource_context["allocated_dpu_count"] is None
    assert resource_context["allocated_tasklets_per_dpu"] is None


def test_completed_resident_suite_summary_uses_verified_row_allocation(
    resident_hardware_suite, monkeypatch, tmp_path: Path
) -> None:
    run_dir = tmp_path / "run"
    (run_dir / "config").mkdir(parents=True)
    session_root = tmp_path / "native_session"
    session_root.mkdir()
    native_build = HardwareSessionBuild(
        session_root=session_root,
        source_snapshot=session_root,
        build_dir=session_root,
        host_binary=session_root / "host",
        dpu_binary=session_root / "dpu_resident",
        source_tree_hash="source",
        host_binary_hash="host",
        dpu_binary_hash="dpu",
        build_time_s=0.0,
        build_command=("fake-build",),
        sdk_tools={"fake": "fixture"},
    )
    suite = replace(
        resident_hardware_suite,
        suite={
            **resident_hardware_suite.suite,
            "cases": [{"case_id": "fixture"}],
        },
    )
    record = {
        "status": "completed",
        "execution_plan_hash": "plan",
        "resource_context": {
            "requested_dpu_count": 1,
            "allocated_dpu_count": 1,
            "requested_tasklets_per_dpu": 1,
            "allocated_tasklets_per_dpu": 1,
            "allocation_status": "verified",
        },
    }

    monkeypatch.setattr(
        resident_runner, "build_resident_hardware_session", lambda *args, **kwargs: native_build
    )
    monkeypatch.setattr(
        resident_runner, "prepare_resident_case", lambda *args, **kwargs: {}
    )
    monkeypatch.setattr(
        resident_runner,
        "_run_resident_case",
        lambda **kwargs: (
            [record],
            [],
            {"case_id": "fixture", "status": "passed"},
        ),
    )
    monkeypatch.setattr(
        resident_runner, "write_normalized_records", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(resident_runner, "capture_environment", lambda *args: {})

    result = resident_runner.run_resident_suite(
        tmp_path,
        run_dir,
        suite,
        environment={"UPMEM_HW_RANK_PATH": "/dev/dpu_rank20"},
    )
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    environment_artifact = json.loads(
        (run_dir / "environment.json").read_text(encoding="utf-8")
    )
    manifest = json.loads(
        (run_dir / "run_manifest.json").read_text(encoding="utf-8")
    )

    assert result.status == "completed"
    assert summary["resource_context"] == record["resource_context"]
    expected_environment_metadata = {
        "upmem_rank_path_requested": "/dev/dpu_rank20",
        "upmem_sdk_profile_effective": (
            "backend=hw,rankPath=/dev/dpu_rank20,ignoreVpd=true"
        ),
    }
    assert environment_artifact["hardware_environment_metadata"] == expected_environment_metadata
    assert manifest["hardware_environment_metadata"] == expected_environment_metadata
