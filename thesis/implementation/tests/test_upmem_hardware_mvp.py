from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from quantum_bench.bench.upmem_hardware_mvp import (
    _normalized_record,
    _prepare_case_bridge,
    prepare_upmem_hardware_mvp,
    validate_hardware_execution_request,
)
from quantum_bench.targets.upmem.dense_bridge import (
    DENSE_BRIDGE_ID,
    DENSE_BRIDGE_SCHEMA_VERSION,
    DenseBridgeBackendIdentity,
    DenseBridgeBlob,
    DenseBridgeExecutionResult,
    DenseBridgeOutputManifest,
    execute_dense_bridge,
)
import quantum_bench.targets.upmem.dense_bridge as dense_bridge_module
from quantum_bench.targets.upmem.hardware_mvp import (
    load_hardware_mvp_suite,
    validate_hardware_mvp_manifest,
)


ROOT = Path(__file__).resolve().parents[1]
SUITE_PATH = ROOT / "configs" / "suites" / "upmem_hardware_mvp.yml"


def test_hardware_mvp_suite_is_fixed_and_deterministic() -> None:
    suite = load_hardware_mvp_suite(SUITE_PATH)

    assert suite.suite_id == "upmem_hardware_mvp"
    assert [case.case_id for case in suite.cases] == ["dense_l1_2x2", "dense_l1_4x4"]
    assert suite.profile.requested_dpu_count == 1
    assert suite.profile.tasklets_per_dpu == 1
    assert suite.profile.repetitions == 5
    assert suite.cases[0].expected_accumulator.tolist() == [[13, -5], [19, 11]]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("hardware_profile_version", "hardware_mvp_l1_v1"),
        ("requested_dpu_count", 2),
        ("tasklets_per_dpu", 2),
        ("max_dim", 5),
        ("repetitions", 6),
    ],
)
def test_hardware_mvp_suite_rejects_profile_expansion(
    tmp_path: Path, field: str, value: object
) -> None:
    payload = yaml.safe_load(SUITE_PATH.read_text(encoding="utf-8"))
    payload["profile"][field] = value
    suite_path = tmp_path / "expanded.yml"
    suite_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="hardware_profile_violation"):
        load_hardware_mvp_suite(suite_path)


def test_hardware_mvp_profile_rejects_expanded_or_nonreal_manifests(
    tmp_path: Path,
) -> None:
    suite = load_hardware_mvp_suite(SUITE_PATH)
    manifest, _ = _prepare_case_bridge(
        suite.cases[0], tmp_path / "bridge", suite.profile
    )
    payload = json.loads(
        (tmp_path / "bridge" / "input_manifest.json").read_text(encoding="utf-8")
    )
    validate_hardware_mvp_manifest(payload)

    too_many_dimensions = dict(payload)
    too_many_dimensions["gemm_m"] = 5
    with pytest.raises(ValueError, match="hardware_profile_violation"):
        validate_hardware_mvp_manifest(too_many_dimensions)

    complex_policy = dict(payload)
    complex_policy["fixed_point_spec"] = {
        **payload["fixed_point_spec"],
        "complex_policy": "split_real_imag_last_axis",
    }
    with pytest.raises(ValueError, match="complex policy"):
        validate_hardware_mvp_manifest(complex_policy)

    wrong_allocation_profile = dict(payload)
    wrong_allocation_profile["metadata"] = {
        **payload["metadata"],
        "sdk_allocation_profile": "backend=simulator",
    }
    with pytest.raises(ValueError, match="sdk_allocation_profile"):
        validate_hardware_mvp_manifest(wrong_allocation_profile)


def test_hardware_execution_request_fails_closed_without_opt_in() -> None:
    with pytest.raises(ValueError, match="UPMEM_ALLOW_PHYSICAL_HARDWARE=1"):
        validate_hardware_execution_request(execute=True, environment={})
    with pytest.raises(ValueError, match="DPU_BACKEND"):
        validate_hardware_execution_request(
            execute=True,
            environment={
                "UPMEM_ALLOW_PHYSICAL_HARDWARE": "1",
                "DPU_BACKEND": "simulator",
            },
        )
    validate_hardware_execution_request(
        execute=True,
        environment={
            "UPMEM_ALLOW_PHYSICAL_HARDWARE": "1",
            "UPMEM_PROFILE": "backend=simulator",
            "UPMEM_PROFILE_BASE": "backend=hw",
        },
    )


def test_hardware_bridge_returns_explicit_opt_in_failure_without_external_process(
    tmp_path: Path,
) -> None:
    suite = load_hardware_mvp_suite(SUITE_PATH)
    _prepare_case_bridge(suite.cases[0], tmp_path / "bridge", suite.profile)

    result = execute_dense_bridge(
        tmp_path / "bridge" / "input_manifest.json",
        backend="upmem_sdk_hardware_dense",
        execute_external=True,
        env={},
    )

    assert result.execution_status == "failed"
    assert result.reason == "hardware_opt_in_missing"
    assert result.external_command_executed is False
    assert result.output_manifest is not None
    assert result.output_manifest.metadata["hardware_kernel_executed"] is False


def test_hardware_bridge_requires_status_proof_and_never_injects_simulator(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    suite = load_hardware_mvp_suite(SUITE_PATH)
    bridge = tmp_path / "bridge"
    _prepare_case_bridge(suite.cases[0], bridge, suite.profile)
    output = bridge / "outputs" / "upmem_sdk_hardware_output.npy"
    accumulator = bridge / "outputs" / "hardware_accumulator_crop_i32.npy"
    output.parent.mkdir(exist_ok=True)
    expected = suite.cases[0].expected_accumulator.astype("<i4")
    np.save(output, expected.astype(np.float32), allow_pickle=False)
    np.save(accumulator, expected, allow_pickle=False)
    captured: dict[str, object] = {}

    class Completed:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(command: object, **kwargs: object) -> Completed:
        captured["command"] = command
        captured["env"] = kwargs.get("env")
        valid = {
            "schema_version": DENSE_BRIDGE_SCHEMA_VERSION,
            "bridge_id": DENSE_BRIDGE_ID,
            "manifest_kind": "dense_bridge_output",
            "backend": "upmem_sdk_hardware_dense",
            "status": "upmem_sdk_hardware_executed",
            "input_manifest": "input_manifest.json",
            "route_id": suite.profile.route_id,
            "task_id": "task",
            "output_blob": {
                "relative_path": "outputs/upmem_sdk_hardware_output.npy",
                "dtype": "float32",
                "shape": [2, 2],
                "representation": "dequantized_output",
                "nbytes": 16,
                "labels": [0, 2],
                "role": "hardware_output",
            },
            "accumulator_blob": {
                "relative_path": "outputs/hardware_accumulator_crop_i32.npy",
                "dtype": "<i4",
                "shape": [2, 2],
                "representation": "int32_accumulator_crop",
                "nbytes": 16,
                "role": "hardware_accumulator_crop",
            },
            "validation_metrics": {"exact_integer_passed": True, "passed": True},
            "compute_time_s": 0.0,
            "write_time_s": 0.0,
            "total_time_s": 0.0,
            "external_command_executed": True,
            "execution_implemented": True,
            "error": None,
            "metadata": {
                "hardware_status_json": {
                    "success": True,
                    "failure_stage": None,
                    "allocation_profile": "backend=hw",
                    "requested_dpus": 1,
                    "allocated_dpus": 1,
                    "tasklets": 1,
                },
                "sdk_allocation_profile": "backend=hw",
                "sdk_allocation_profile_source": "compiled_native_literal",
                "raw_accumulator_crop": True,
                "cpu_reference": "int8_x_int8_to_int32_exact",
                "hashes": {
                    "left": "a",
                    "right": "b",
                    "accumulator": "c",
                    "output": "d",
                },
                "application_visible_transfer_bytes": {
                    "h2d": 72,
                    "d2h": 64,
                    "total": 136,
                },
                "timing_labels": "hardware_bringup_functionality_only",
                "speedup_claims": False,
                "hardware_kernel_executed": True,
                "simulator_kernel_executed": False,
                "cpu_fallback_used": False,
            },
        }
        (bridge / "output_manifest.json").write_text(
            json.dumps(valid), encoding="utf-8"
        )
        return Completed()

    monkeypatch.setattr(dense_bridge_module.subprocess, "run", fake_run)
    result = execute_dense_bridge(
        bridge / "input_manifest.json",
        backend="upmem_sdk_hardware_dense",
        execute_external=True,
        env={
            "UPMEM_ALLOW_PHYSICAL_HARDWARE": "1",
            "UPMEM_PROFILE": "backend=simulator",
            "UPMEM_PROFILE_BASE": "backend=hw",
        },
    )

    assert result.execution_status == "upmem_sdk_hardware_executed"
    assert isinstance(captured["env"], dict)
    assert "DPU_BACKEND" not in captured["env"]
    assert "UPMEM_PROFILE" not in captured["env"]
    assert "UPMEM_PROFILE_BASE" not in captured["env"]

    invalid = json.loads((bridge / "output_manifest.json").read_text(encoding="utf-8"))
    invalid["metadata"]["hardware_status_json"]["allocated_dpus"] = 2
    (bridge / "output_manifest.json").write_text(json.dumps(invalid), encoding="utf-8")

    def fake_invalid_run(command: object, **kwargs: object) -> Completed:
        return Completed()

    monkeypatch.setattr(dense_bridge_module.subprocess, "run", fake_invalid_run)
    invalid_result = execute_dense_bridge(
        bridge / "input_manifest.json",
        backend="upmem_sdk_hardware_dense",
        execute_external=True,
        env={
            "UPMEM_ALLOW_PHYSICAL_HARDWARE": "1",
            "UPMEM_PROFILE": "backend=simulator",
            "UPMEM_PROFILE_BASE": "backend=hw",
        },
    )

    assert invalid_result.execution_status == "failed"
    assert invalid_result.reason == "output_validation_failed"
    assert invalid_result.output_manifest is not None
    assert invalid_result.output_manifest.status == "failed"

    missing_profile = json.loads(
        (bridge / "output_manifest.json").read_text(encoding="utf-8")
    )
    missing_profile["metadata"]["hardware_status_json"]["allocated_dpus"] = 1
    missing_profile["metadata"].pop("sdk_allocation_profile")
    (bridge / "output_manifest.json").write_text(
        json.dumps(missing_profile), encoding="utf-8"
    )
    missing_profile_result = execute_dense_bridge(
        bridge / "input_manifest.json",
        backend="upmem_sdk_hardware_dense",
        execute_external=True,
        env={"UPMEM_ALLOW_PHYSICAL_HARDWARE": "1"},
    )

    assert missing_profile_result.execution_status == "failed"
    assert missing_profile_result.output_manifest is not None
    assert missing_profile_result.output_manifest.status == "failed"


def test_prepare_only_writes_inputs_without_build_or_dpu_allocation(
    tmp_path: Path,
) -> None:
    plan = prepare_upmem_hardware_mvp(
        tmp_path, suite_path=SUITE_PATH, build=False, environment={}
    )
    payload = json.loads(plan.summary_path.read_text(encoding="utf-8"))

    assert plan.status == "prepared"
    assert payload["dpu_allocation_attempted"] is False
    assert payload["dpu_launch_attempted"] is False
    assert (
        plan.plan_dir / "cases" / "dense_l1_2x2" / "bridge" / "input_manifest.json"
    ).exists()
    assert (
        plan.plan_dir
        / "cases"
        / "dense_l1_4x4"
        / "bridge"
        / "references"
        / "expected_accumulator_i32.npy"
    ).exists()


def test_hardware_normalized_record_requires_single_dpu_exact_flags(
    tmp_path: Path,
) -> None:
    suite = load_hardware_mvp_suite(SUITE_PATH)
    case = suite.cases[0]
    output = DenseBridgeOutputManifest(
        schema_version=DENSE_BRIDGE_SCHEMA_VERSION,
        bridge_id=DENSE_BRIDGE_ID,
        manifest_kind="dense_bridge_output",
        backend="upmem_sdk_hardware_dense",
        status="upmem_sdk_hardware_executed",
        input_manifest="input_manifest.json",
        route_id=suite.profile.route_id,
        task_id="task",
        output_blob=DenseBridgeBlob(
            "outputs/result.npy", "float32", (2, 2), "dequantized", 16
        ),
        accumulator_blob={
            "relative_path": "outputs/acc.npy",
            "dtype": "<i4",
            "shape": [2, 2],
            "nbytes": 16,
        },
        validation_metrics={
            "exact_integer_passed": True,
            "passed": True,
            "max_abs_error": 0.0,
            "l2_error": 0.0,
        },
        compute_time_s=0.0,
        write_time_s=0.0,
        total_time_s=0.1,
        external_command_executed=True,
        execution_implemented=True,
        metadata={
            "hardware_kernel_executed": True,
            "hardware_status_json": {
                "success": True,
                "failure_stage": None,
                "allocation_profile": "backend=hw",
                "requested_dpus": 1,
                "allocated_dpus": 1,
                "tasklets": 1,
            },
            "sdk_allocation_profile": "backend=hw",
            "application_visible_transfer_bytes": {"h2d": 72, "d2h": 64, "total": 136},
            "hashes": {},
        },
    )
    result = DenseBridgeExecutionResult(
        schema_version=DENSE_BRIDGE_SCHEMA_VERSION,
        bridge_id=DENSE_BRIDGE_ID,
        execution_status="upmem_sdk_hardware_executed",
        backend_id="upmem_sdk_hardware_dense",
        backend_identity=DenseBridgeBackendIdentity(
            "upmem_sdk_hardware_dense",
            "hardware",
            "upmem_sdk",
            "external_process",
            True,
            True,
            "test",
        ),
        reason=None,
        error=None,
        error_type=None,
        input_manifest_path="input_manifest.json",
        output_manifest_path="output_manifest.json",
        output_blob_path="outputs/result.npy",
        output_manifest=output,
        invocation_metadata={},
        external_command_executed=True,
        execution_implemented=True,
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    input_manifest = (
        run_dir
        / "cases"
        / "dense_l1_2x2"
        / "repeat_00"
        / "bridge"
        / "input_manifest.json"
    )
    input_manifest.parent.mkdir(parents=True)
    input_manifest.write_text("{}", encoding="utf-8")
    reference = input_manifest.parent / "references" / "expected.npy"
    reference.parent.mkdir()
    reference.write_bytes(b"reference")
    record = _normalized_record(
        run_dir=run_dir,
        suite=suite,
        case=case,
        repeat_id=0,
        result=result,
        input_manifest_path=input_manifest,
        expected_accumulator_path=reference,
        elapsed_s=0.1,
        source_commit="abc",
    )

    assert record["status"] == "completed"
    assert record["allocated_dpu_count"] == 1
    assert record["hardware_allocation_verified"] is True
    assert record["sdk_allocation_profile_verified"] is True
    assert record["simulator_kernel_executed"] is False
    assert record["cpu_fallback_used"] is False
    assert (
        record["actual_transfer_bytes"]
        == record["actual_h2d_bytes"] + record["actual_d2h_bytes"]
    )
    assert record["hardware_speedup_applicable"] is False


def test_native_hardware_mvp_sources_enforce_one_tasklet_and_checked_status() -> None:
    source = ROOT / "native" / "upmem" / "simplepim" / "upmem_sdk_dense"
    makefile = (source / "Makefile").read_text(encoding="utf-8")
    dpu = (source / "dpu.c").read_text(encoding="utf-8")
    host = (source / "host.c").read_text(encoding="utf-8")
    runner = (
        ROOT
        / "native"
        / "upmem"
        / "simplepim"
        / "upmem_sdk_dense_hardware_mvp_runner.py"
    ).read_text(encoding="utf-8")

    assert "NR_TASKLETS ?= 1" in makefile
    assert "UPMEM_DENSE_HARDWARE_MVP ?= 0" in makefile
    assert "UPMEM_DENSE_HARDWARE_MVP requires exactly one DPU tasklet" in dpu
    assert "if (me() != 0)" in dpu
    assert "dpu_get_nr_dpus" in host
    assert 'UPMEM_DENSE_ALLOCATION_PROFILE "backend=hw"' in host
    assert "dpu_alloc(requested_dpus, UPMEM_DENSE_ALLOCATION_PROFILE" in host
    assert "DPU_ERR_INVALID_PROFILE" in host
    assert "DPU_SYNCHRONOUS" in host
    assert "UPMEM_DENSE_STATUS_JSON" in host
    assert 'write_status("result_transfer_failed"' in host
    assert "native_build_stdout_snippet" in runner
    assert "host_stderr_snippet" in runner
    assert '"sdk_metadata"' in runner
    assert '"compiler_metadata"' in runner
    assert 'PROFILE_VERSION = "hardware_mvp_l1_v2"' in runner
    assert 'SDK_ALLOCATION_PROFILE = "backend=hw"' in runner
