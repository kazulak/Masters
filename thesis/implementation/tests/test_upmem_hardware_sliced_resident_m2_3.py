from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

import quantum_bench.bench.upmem_hardware_sliced_resident_mvp as mvp
import quantum_bench.targets.upmem.hardware_sliced_resident_session as adapter
from quantum_bench.targets.upmem.hardware_session import HardwareSessionBuild
from quantum_bench.targets.upmem.hardware_taskgraph_sliced_resident import (
    build_two_slice_resident_graph_packages,
    reconstruct_host_slice_outputs,
)
from quantum_bench.targets.upmem.hardware_taskgraph_resident import (
    build_resident_policy_reference,
)


ROOT = Path(__file__).resolve().parents[1]
M2_3_SUITE = (
    ROOT / "configs" / "suites" / "upmem_hardware_sliced_resident_m2_3.yml"
)


def _prepared_cases():
    loaded = mvp.load_m2_suite(M2_3_SUITE)
    prepared = [
        (case, mvp._prepare_case(ROOT, case, loaded))
        for case in loaded.suite["cases"]
    ]
    return loaded, prepared


def test_m2_3_loader_requires_the_exact_path_variant_pair() -> None:
    loaded = mvp.load_m2_suite(M2_3_SUITE)

    assert loaded.is_m2_3
    assert loaded.fixture_version == mvp.M2_3_SCHEMA_VERSION
    assert loaded.experiment_profile_version == (
        "hardware_sliced_resident_two_dpu_m2_3_v1"
    )
    assert loaded.profile.version == "hardware_sliced_resident_two_dpu_m2_v1"
    assert loaded.suite["warmups"] == 1
    assert loaded.suite["repeats"] == 5
    assert loaded.numeric_modes == (
        "none",
        "per_task_resident_requantize",
    )
    assert set(loaded.path_variants or {}) == {
        "opt_einsum_greedy",
        "custom_upmem_v2_balanced",
    }
    assert {
        path_id: variant["label"]
        for path_id, variant in (loaded.path_variants or {}).items()
    } == {
        "opt_einsum_greedy": "opt_einsum greedy",
        "custom_upmem_v2_balanced": "custom UPMEM v2 balanced",
    }
    assert loaded.raw["metadata"]["purpose"] == mvp.M2_3_PURPOSE
    assert (
        loaded.raw["metadata"]["claim_boundary"]
        == mvp.M2_3_SUITE_CLAIM_BOUNDARY
    )


def test_m2_3_loader_rejects_duplicate_path_variant(tmp_path, monkeypatch) -> None:
    payload = yaml.safe_load(M2_3_SUITE.read_text(encoding="utf-8"))
    payload["path_variants"][1] = dict(payload["path_variants"][0])
    copied = tmp_path / M2_3_SUITE.name
    copied.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    monkeypatch.setattr(mvp, "CANONICAL_M2_3_SUITE_PATH", copied)

    with pytest.raises(ValueError, match="path variants"):
        mvp.load_m2_suite(copied)


@pytest.mark.parametrize(
    "mutation",
    (
        "purpose",
        "claim_boundary",
        "manual_invocation",
        "deterministic",
        "fixture_scope",
        "native_profile",
        "numeric_coverage",
        "label",
    ),
)
def test_m2_3_loader_fails_closed_on_claim_and_fixture_metadata(
    tmp_path, monkeypatch, mutation
) -> None:
    payload = yaml.safe_load(M2_3_SUITE.read_text(encoding="utf-8"))
    if mutation == "purpose":
        payload["metadata"]["purpose"] = "changed"
    elif mutation == "claim_boundary":
        payload["metadata"]["claim_boundary"] = "changed"
    elif mutation == "manual_invocation":
        payload["metadata"]["manual_invocation_required"] = False
    elif mutation == "deterministic":
        payload["metadata"]["deterministic_unitary_only"] = False
    elif mutation == "fixture_scope":
        payload["metadata"]["fixture_scope"] = "changed"
    elif mutation == "native_profile":
        payload["metadata"].pop("native_hardware_profile_version")
    elif mutation == "numeric_coverage":
        payload["workloads"][0]["hardware_numeric_coverage"] = "complex"
    else:
        payload["path_variants"][0]["label"] = "changed"
    copied = tmp_path / M2_3_SUITE.name
    copied.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    monkeypatch.setattr(mvp, "CANONICAL_M2_3_SUITE_PATH", copied)

    with pytest.raises(ValueError, match="hardware_profile_violation"):
        mvp.load_m2_suite(copied)


def test_m2_3_loader_requires_exact_fixture_path_binding_set(
    tmp_path, monkeypatch
) -> None:
    payload = yaml.safe_load(M2_3_SUITE.read_text(encoding="utf-8"))
    duplicate = payload["workloads"][1]
    canonical = payload["workloads"][3]
    duplicate["fixture_id"] = canonical["fixture_id"]
    duplicate["circuit"] = dict(canonical["circuit"])
    duplicate["expected_output"] = list(canonical["expected_output"])
    copied = tmp_path / M2_3_SUITE.name
    copied.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    monkeypatch.setattr(mvp, "CANONICAL_M2_3_SUITE_PATH", copied)

    with pytest.raises(ValueError, match="fixture/path binding set"):
        mvp.load_m2_suite(copied)


def test_m2_3_planners_produce_required_paths_and_identity_contract() -> None:
    loaded, prepared_cases = _prepared_cases()
    expected_paths = {
        "opt_einsum_greedy": ((0, 1), (0, 1), (0, 1)),
        "custom_upmem_v2_balanced": ((0, 1), (0, 2), (0, 1)),
    }
    by_fixture: dict[str, list[dict]] = {}
    for case, prepared in prepared_cases:
        path_id = str(case["path_variant_id"])
        fixture_id = str(case["fixture_id"])
        assert prepared["planner_path"] == expected_paths[path_id]
        assert prepared["planner_path_matches_expected"] is True
        assert len(prepared["graph"].tasks) == 3
        np.testing.assert_allclose(
            prepared["reference"],
            np.asarray(case["expected_output"]),
            atol=1.0e-12,
            rtol=1.0e-12,
        )
        assert prepared["planner_config"] == loaded.path_variants[path_id]["planner"]
        assert prepared["path_variant_label"] == loaded.path_variants[path_id]["label"]
        assert prepared["planner_candidate_evidence_type"] == "modeled"
        assert prepared["execution_route_policy"] == (
            "hardware_sliced_resident_two_dpu_m2_3_v1"
        )
        assert prepared["planner_policy_matches_execution_route"] is False
        assert prepared["planner_engine"] in {"opt_einsum", "custom_upmem"}
        by_fixture.setdefault(fixture_id, []).append(prepared)

    for variants in by_fixture.values():
        assert len(variants) == 2
        assert len({item["graph"].circuit_semantics_hash for item in variants}) == 1
        assert len({item["graph"].tensor_network_hash for item in variants}) == 1
        assert len({item["graph"].contraction_plan_hash for item in variants}) == 2
        for mode in loaded.numeric_modes:
            executor_hashes = {
                mvp._executor_config_hash(
                    mode,
                    profile_version=item["plan"].hardware_profile_version,
                    operation_count=item["source_task_count"],
                )
                for item in variants
            }
            assert len(executor_hashes) == 1
        assert (
            mvp._executor_config_hash(
                "none",
                profile_version=variants[0]["plan"].hardware_profile_version,
                operation_count=3,
            )
            != mvp._executor_config_hash(
                "per_task_resident_requantize",
                profile_version=variants[0]["plan"].hardware_profile_version,
                operation_count=3,
            )
        )


def test_m2_3_policy_reference_has_discriminating_numeric_modes() -> None:
    _, prepared_cases = _prepared_cases()
    for _, prepared in prepared_cases:
        packages = build_two_slice_resident_graph_packages(
            prepared["plan"],
            case_id=prepared["case_id"],
            suite_id="m2_3_test",
            quantization_mode="none",
        )
        outputs: dict[str, np.ndarray] = {}
        for mode in ("none", "per_task_resident_requantize"):
            partials = {
                item.slice_id: build_resident_policy_reference(
                    item.package.graph,
                    item.network,
                    quantization_mode=mode,
                )["output"]
                for item in packages
            }
            outputs[mode] = reconstruct_host_slice_outputs(
                prepared["plan"], partials
            )

        none_error = float(
            np.max(np.abs(outputs["none"] - prepared["reference"]))
        )
        requant_error = float(
            np.max(
                np.abs(
                    outputs["per_task_resident_requantize"] - prepared["reference"]
                )
            )
        )
        assert none_error <= mvp.M2_3_NONE_VALIDATION_TOLERANCE
        assert mvp.M2_3_MIN_REQUANTIZED_ERROR < requant_error < 1.0e-2
        policy_metrics = mvp._accuracy_metrics(
            outputs["per_task_resident_requantize"],
            outputs["per_task_resident_requantize"],
            tolerance=1.0e-5,
            reference_kind="dpu_mirroring_policy_reference",
        )
        assert policy_metrics["max_abs_error"] <= 1.0e-5


def _fake_build(root: Path, session_root: Path, **_: object) -> HardwareSessionBuild:
    binary_dir = session_root / "bin"
    binary_dir.mkdir(parents=True, exist_ok=True)
    host = binary_dir / "host_two_dpu"
    dpu = binary_dir / "dpu_resident_two_dpu"
    host.write_bytes(b"host")
    dpu.write_bytes(b"dpu")
    return HardwareSessionBuild(
        session_root=session_root,
        source_snapshot=session_root / "src",
        build_dir=binary_dir,
        host_binary=host,
        dpu_binary=dpu,
        source_tree_hash="source-hash",
        host_binary_hash="host-hash",
        dpu_binary_hash="dpu-hash",
        build_time_s=0.1,
        build_command=("make", "all"),
        sdk_tools={"make": "fake"},
    )


def test_m2_3_prepare_only_preflights_all_48_exact_native_shapes(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(mvp, "build_sliced_resident_hardware_session", _fake_build)
    calls: list[tuple[str, ...]] = []
    real_run = mvp.subprocess.run

    def parse_only(command, **_kwargs):
        command = tuple(str(item) for item in command)
        if "--validate-slice-packages" not in command:
            return real_run(command, **_kwargs)
        calls.append(command)
        assert command[1] == "--validate-slice-packages"
        assert "--slice-package-0" not in command
        assert len(command) == 4
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps({"status": "valid", "reason": None}),
            stderr="",
        )

    monkeypatch.setattr(mvp.subprocess, "run", parse_only)
    result = mvp.prepare_upmem_hardware_sliced_resident_mvp(
        tmp_path, suite_path=M2_3_SUITE, build=True, environment={}
    )
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))

    assert result.status == "prepared"
    assert len(summary["prepared_operations"]) == 48
    assert len(summary["native_manifest_validation"]["entries"]) == 48
    assert len(calls) == 48
    assert summary["dpu_allocation_attempted"] is False
    assert summary["dpu_launch_attempted"] is False
    manifests = sorted(result.plan_dir.rglob("slice_0_manifest.json"))
    assert len(manifests) == 48
    session_ids: set[str] = set()
    for manifest_path in manifests:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        session_ids.add(manifest["session_id"])
        assert manifest["component_operation_count"] == 3
        assert (
            manifest["slice_execution"]["outer_execution_identity"][
                "hardware_profile_version"
            ]
            == "hardware_sliced_resident_two_dpu_m2_3_v1"
        )
    assert len(session_ids) == 48
    assert len({command[2] for command in calls}) == 48
    assert len({command[3] for command in calls}) == 48


def test_m2_3_runner_dispatches_eight_warmups_and_forty_measurements(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(mvp, "build_sliced_resident_hardware_session", _fake_build)
    calls: list[tuple[str, str, str, str, int]] = []

    def fake_run_operation(
        _run_dir,
        _native,
        _m2,
        case,
        _prepared,
        phase,
        repeat_id,
        _environment,
        *,
        numeric_mode,
    ):
        calls.append(
            (
                str(case["fixture_id"]),
                str(case["path_variant_id"]),
                numeric_mode,
                phase,
                repeat_id,
            )
        )
        return {
            "status": "completed",
            "validation_status": "passed",
            "fixture_id": str(case["fixture_id"]),
            "path_variant_id": str(case["path_variant_id"]),
            "phase": phase,
            "repeat_id": repeat_id,
            "numeric_mode": numeric_mode,
        }

    monkeypatch.setattr(mvp, "_run_operation", fake_run_operation)
    result = mvp.run_upmem_hardware_sliced_resident_mvp(
        tmp_path,
        suite_path=M2_3_SUITE,
        environment={"UPMEM_ALLOW_PHYSICAL_HARDWARE": "1"},
    )
    summary = json.loads(
        (result.run_dir / "upmem_hardware_sliced_resident_mvp_summary.json").read_text(
            encoding="utf-8"
        )
    )

    assert result.status == "completed"
    assert result.row_count == 40
    assert len(calls) == 48
    expected = {
        (fixture_id, path_variant_id, numeric_mode, phase, repeat_id)
        for fixture_id in ("ry_h_ry_a", "ry_h_ry_b")
        for path_variant_id in (
            "opt_einsum_greedy",
            "custom_upmem_v2_balanced",
        )
        for numeric_mode in ("none", "per_task_resident_requantize")
        for phase, repeat_count in (("warmup", 1), ("measured", 5))
        for repeat_id in range(repeat_count)
    }
    assert set(calls) == expected
    assert summary["expected_warmup_row_count"] == 8
    assert summary["expected_measured_row_count"] == 40
    assert summary["dispatch_matrix_status"] == "passed"
    assert summary["warmup_dispatch_matrix"]["expected_key_count"] == 8
    assert summary["measured_dispatch_matrix"]["expected_key_count"] == 40
    assert summary["numeric_modes"] == ["none", "per_task_resident_requantize"]


def test_m2_3_record_reports_six_completed_physical_task_instances(
    tmp_path,
) -> None:
    loaded, prepared_cases = _prepared_cases()
    case, prepared = next(
        item
        for item in prepared_cases
        if item[0]["path_variant_id"] == "custom_upmem_v2_balanced"
    )
    dpu_binary = tmp_path / "dpu"
    dpu_binary.write_bytes(b"dpu")
    artifacts = mvp._write_packages(
        prepared,
        loaded,
        dpu_binary,
        tmp_path / "packages",
        prefix="test",
        numeric_mode="none",
    )
    h2d, d2h = mvp._transfer_bytes(artifacts["packages"])
    response = {
        "backend_id": adapter.BACKEND_ID,
        "backend_family": "upmem_sdk",
        "target_requested": "hardware",
        "target_observed": "hardware",
        "hardware_profile_version": adapter.PROFILE_VERSION,
        "quantization_mode": "none",
        "hardware_execution": True,
        "hardware_kernel_executed": True,
        "simulator_kernel_executed": False,
        "cpu_fallback_used": False,
        "operation_count": 3,
        "observed_operation_completion_counts": [3, 3],
        "completion_sentinel_read_counts": [3, 3],
        "native_execution_sentinel_available": True,
        "completion_evidence": "dpu_written_completion_sentinel_read_after_each_sync",
        "device_completion_confirmed": True,
        "actual_h2d_bytes": h2d,
        "actual_d2h_bytes": d2h,
        "actual_transfer_bytes": h2d + d2h,
        "allocation": {"allocated_dpus": 2, "verified": True},
        "launch": {"completed": True, "synchronize_count": 3},
        "release": {"confirmed": True},
        "slices": [
            {
                "completion_confirmed": True,
                "dpu_completion_sentinel": {"verified": True},
                "completion_sentinel_read_count": 3,
            }
            for _ in range(2)
        ],
    }
    session = SimpleNamespace(
        response=response,
        response_path=tmp_path / "response.json",
        command=("host_two_dpu",),
        stdout_snippet="",
        stderr_snippet="",
        failure_stage=None,
        timed_out=False,
        cleanup_confirmed=True,
        process_time_s=0.1,
    )
    native = SimpleNamespace(
        source_tree_hash="source-tree",
        host_binary_hash="host-binary",
        dpu_binary_hash="dpu-binary",
        build_time_s=0.1,
    )
    output = np.asarray(prepared["reference"].real, dtype=np.float32)
    metrics = mvp._accuracy_metrics(
        output,
        output,
        tolerance=1.0e-5,
        reference_kind="dpu_mirroring_policy_reference",
    )
    full_precision = mvp._accuracy_metrics(
        prepared["reference"],
        output,
        tolerance=mvp.M2_3_NONE_VALIDATION_TOLERANCE,
        reference_kind="cpu_exact_taskgraph_full_precision",
    )
    record = mvp._record(
        loaded,
        case,
        prepared,
        "measured",
        0,
        run_dir=tmp_path,
        status="completed",
        failure_stage=None,
        reason=None,
        native=native,
        session=session,
        artifacts=artifacts,
        reconstruction={
            "partial_outputs": {},
            "per_slice_output_validation_status": "passed",
            "slice_useful_work": {"status": "passed"},
        },
        output=output,
        cpu_ok=True,
        expected_ok=True,
        reconstruction_time_s=0.01,
        total_time_s=0.2,
        numeric_mode="none",
        policy_reference=metrics,
        full_precision_accuracy=full_precision,
    )

    assert record["status"] == "completed"
    assert record["task_count"] == 3
    assert record["operation_count"] == 3
    assert record["source_task_count"] == 3
    assert record["source_task_completion_count"] == 3
    assert record["expanded_task_count"] == 6
    assert record["expanded_task_completion_count"] == 6
    assert record["expanded_task_count_scope"] == (
        "physical_source_operation_instances_across_two_slices"
    )
    assert record["slice_descriptor_count"] == 2
    assert record["slice_descriptor_completion_count"] == 2
    assert record["completed_operation_count_per_slice"] == [3, 3]
    assert record["completed_physical_task_instance_count"] == 6
    assert record["expected_physical_task_instance_count"] == 6
    assert record["expanded_physical_operation_count"] == 6
    assert record["expanded_physical_operation_completion_count"] == 6
    assert record["slice_model_task_count"] == 2
    assert record["slice_model_executed_task_count"] == 2
    assert record["slice_model_task_count_scope"] == "slice_descriptors"
    assert record["executed_task_count"] == 6
    assert record["executed_task_count_scope"] == (
        "compatibility_alias_for_expanded_task_completion_count"
    )
    assert record["planner_path_matches_expected"] is True
    assert record["path_variant_label"] == "custom UPMEM v2 balanced"
    assert record["planner_candidate_evidence_type"] == "modeled"
    assert record["planner_execution_policy"] == (
        "generic_single_dpu_split_complex_v2"
    )
    assert record["execution_route_policy"] == (
        "hardware_sliced_resident_two_dpu_m2_3_v1"
    )
    assert record["planner_policy_matches_execution_route"] is False
    assert record["planner_route_relation"] == (
        "fixed_modeled_candidate_path_executed_on_different_two_dpu_sliced_resident_route"
    )
    assert record["experiment_schema_version"] == mvp.M2_3_SCHEMA_VERSION
    assert record["hardware_profile_version"] == (
        "hardware_sliced_resident_two_dpu_m2_3_v1"
    )


@pytest.mark.parametrize(
    "stage",
    (
        "binary_load_failed",
        "hardware_allocation_failed",
        "hardware_release_failed",
        "kernel_completion_sentinel_failed",
        "kernel_launch_failed",
        "kernel_synchronize_failed",
        "operation_control_transfer_failed",
        "partial_output_read_failed",
        "partial_output_write_failed",
        "slice_execution_parse_failed",
        "slice_input_allocation_failed",
        "slice_input_load_failed",
        "slice_manifest_hash_failed",
        "slice_manifest_parse_failed",
        "slice_package_transfer_failed",
        "slice_partial_output_allocation_failed",
    ),
)
def test_m2_3_preserves_known_native_failure_stages(stage) -> None:
    session = SimpleNamespace(failure_stage=stage)

    assert mvp._operation_failure_stage("native execution failed", session) == stage


def test_m2_3_unknown_native_failure_stage_fails_safe() -> None:
    session = SimpleNamespace(failure_stage="unrecognized_native_stage")

    assert (
        mvp._operation_failure_stage("unrecognized_native_stage", session)
        == "operation_failed"
    )


def test_m2_3_malformed_native_response_becomes_explicit_failure(
    tmp_path, monkeypatch
) -> None:
    loaded, prepared_cases = _prepared_cases()
    case, prepared = prepared_cases[0]
    native_root = tmp_path / "native_session"
    native_root.mkdir()
    dpu_binary = native_root / "dpu"
    dpu_binary.write_bytes(b"dpu")
    native = SimpleNamespace(session_root=native_root, dpu_binary=dpu_binary)

    def malformed(_native, *, response_path, **_kwargs):
        return SimpleNamespace(
            status="failed",
            failure_stage="response_evidence_invalid",
            response_path=response_path,
            response={"status": "completed"},
            command=("fake-host",),
            stdout_snippet="",
            stderr_snippet="malformed response",
            timed_out=False,
            cleanup_confirmed=True,
            process_time_s=0.01,
        )

    monkeypatch.setattr(mvp, "execute_sliced_resident_hardware_session", malformed)
    record = mvp._run_operation(
        tmp_path,
        native,
        loaded,
        case,
        prepared,
        "measured",
        0,
        {"UPMEM_ALLOW_PHYSICAL_HARDWARE": "1"},
        numeric_mode="none",
    )

    assert record["status"] == "failed"
    assert record["failure_stage"] == "response_evidence_invalid"
    assert record["native_failure_stage"] == "response_evidence_invalid"
    assert record["fixture_id"] == "ry_h_ry_a"
    assert record["path_variant_label"] == "opt_einsum greedy"
    assert record["contraction_plan_hash"] == prepared["graph"].contraction_plan_hash


def test_m2_3_preparation_failure_keeps_context_without_identity_hashes() -> None:
    loaded = mvp.load_m2_suite(M2_3_SUITE)
    case = loaded.suite["cases"][0]

    record = mvp._failure_record(
        loaded,
        "hardware_profile_violation: preparation failed",
        case,
        "prepare",
        None,
        "hardware_profile_violation",
    )

    assert record["experiment_schema_version"] == mvp.M2_3_SCHEMA_VERSION
    assert record["hardware_profile_version"] == (
        "hardware_sliced_resident_two_dpu_m2_3_v1"
    )
    assert record["fixture_id"] == "ry_h_ry_a"
    assert record["path_variant_id"] == "opt_einsum_greedy"
    assert record["path_variant_label"] == "opt_einsum greedy"
    assert record["planner_engine"] == "opt_einsum"
    assert record["planner_candidate_evidence_type"] == "modeled"
    assert "circuit_semantics_hash" not in record
    assert "tensor_network_hash" not in record
    assert "contraction_plan_hash" not in record
    assert "executor_config_hash" not in record


def test_m2_3_prepare_only_failure_rows_keep_canonical_context(
    tmp_path, monkeypatch
) -> None:
    def fail_prepare(*_args, **_kwargs):
        raise ValueError("hardware_profile_violation: synthetic preparation failure")

    monkeypatch.setattr(mvp, "_prepare_case", fail_prepare)
    result = mvp.prepare_upmem_hardware_sliced_resident_mvp(
        tmp_path, suite_path=M2_3_SUITE, build=False, environment={}
    )
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))

    assert result.status == "failed"
    assert len(summary["prepared_operations"]) == 4
    for row in summary["prepared_operations"]:
        assert row["experiment_schema_version"] == mvp.M2_3_SCHEMA_VERSION
        assert row["hardware_profile_version"] == (
            "hardware_sliced_resident_two_dpu_m2_3_v1"
        )
        assert row["fixture_id"] in {"ry_h_ry_a", "ry_h_ry_b"}
        assert row["path_variant_label"] in {
            "opt_einsum greedy",
            "custom UPMEM v2 balanced",
        }
        assert row["planner_candidate_evidence_type"] == "modeled"
        assert "circuit_semantics_hash" not in row
        assert "tensor_network_hash" not in row
        assert "contraction_plan_hash" not in row
        assert "executor_config_hash" not in row


def test_m2_3_response_paths_are_unique_for_the_exact_dispatch_matrix(tmp_path) -> None:
    loaded = mvp.load_m2_suite(M2_3_SUITE)
    native = SimpleNamespace(session_root=tmp_path)
    paths = {
        mvp._response_path(
            native,
            str(case["case_id"]),
            phase,
            repeat_id,
            m2_2=True,
            numeric_mode=numeric_mode,
        )
        for case in loaded.suite["cases"]
        for numeric_mode in loaded.numeric_modes
        for phase, repeat_id in mvp._phase_ids(loaded.suite)
    }

    assert len(paths) == 48


def _dispatch_matrix_rows(loaded, phase: str) -> list[dict]:
    repeat_count = (
        loaded.suite["warmups"]
        if phase == "warmup"
        else loaded.suite["repeats"]
    )
    return [
        {
            "status": "completed",
            "validation_status": "passed",
            "fixture_id": case["fixture_id"],
            "path_variant_id": case["path_variant_id"],
            "numeric_mode": numeric_mode,
            "phase": phase,
            "repeat_id": repeat_id,
        }
        for case in loaded.suite["cases"]
        for numeric_mode in loaded.numeric_modes
        for repeat_id in range(repeat_count)
    ]


def test_m2_3_finalizer_requires_unique_exact_dispatch_matrix(tmp_path) -> None:
    loaded = mvp.load_m2_suite(M2_3_SUITE)
    warmups = _dispatch_matrix_rows(loaded, "warmup")
    measured = _dispatch_matrix_rows(loaded, "measured")

    failed_key_rows = [dict(row) for row in measured]
    failed_key_rows[0].update(status="failed", validation_status="failed")
    failed_dir = tmp_path / "failed_key"
    failed_dir.mkdir()
    failed_result = mvp._finish_run(
        failed_dir, {}, loaded, failed_key_rows, warmups, native=None
    )
    failed_summary = json.loads(failed_result.summary_path.read_text(encoding="utf-8"))

    assert failed_result.status == "failed"
    assert failed_summary["measured_dispatch_matrix"]["status"] == "passed"
    assert failed_summary["measured_failed_count"] == 1

    duplicate_rows = [dict(row) for row in measured]
    duplicate_rows[-1] = dict(duplicate_rows[0])
    duplicate_dir = tmp_path / "duplicate_key"
    duplicate_dir.mkdir()
    duplicate_result = mvp._finish_run(
        duplicate_dir, {}, loaded, duplicate_rows, warmups, native=None
    )
    duplicate_summary = json.loads(
        duplicate_result.summary_path.read_text(encoding="utf-8")
    )

    assert duplicate_result.status == "failed"
    assert duplicate_summary["measured_passed_count"] == 40
    assert duplicate_summary["measured_dispatch_matrix"]["status"] == "failed"
    assert len(duplicate_summary["measured_dispatch_matrix"]["missing_keys"]) == 1
    assert len(duplicate_summary["measured_dispatch_matrix"]["duplicate_keys"]) == 1
