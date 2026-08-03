from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

import quantum_bench.bench.upmem_hardware_sliced_resident_mvp as mvp
import quantum_bench.targets.upmem.hardware_sliced_resident_session as adapter
from quantum_bench.core.jsonio import read_jsonl, write_json
from quantum_bench.targets.upmem.hardware_session import HardwareSessionBuild
from quantum_bench.targets.upmem.hardware_sliced_resident_session import (
    SlicedResidentSessionExecution,
)
from quantum_bench.targets.upmem.hardware_taskgraph_sliced_resident import (
    _file_fingerprints,
)


ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT / "configs" / "suites" / "upmem_hardware_sliced_resident_mvp.yml"
M2_1_SUITE = ROOT / "configs" / "suites" / "upmem_hardware_sliced_resident_m2_1.yml"


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
        build_time_s=0.125,
        build_command=("make", "all"),
        sdk_tools={"make": "fake"},
    )


def _fake_execute(build, *, manifest_paths, response_path, **_: object):
    payload = _write_completed_native_response(manifest_paths, response_path)
    return SlicedResidentSessionExecution(
        status="completed",
        failure_stage=None,
        response_path=response_path,
        response=payload,
        process_time_s=0.25,
        command=("fake-host",),
        stdout_snippet="",
        stderr_snippet="",
        timed_out=False,
        cleanup_confirmed=True,
    )


def _write_completed_native_response(manifest_paths, response_path: Path) -> dict:
    outputs = {
        "one_qubit_x_m2": np.asarray([0.0, 1.0], dtype=np.float32),
        "one_qubit_h_m2": np.asarray([2**-0.5, 2**-0.5], dtype=np.float32),
        "one_qubit_z_m2": np.asarray([1.0, 0.0], dtype=np.float32),
    }
    entries = []
    for slice_id, manifest_path in enumerate(manifest_paths):
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        case_id = next(key for key in outputs if key in manifest_path.name)
        output_path = Path(payload["final_outputs"][0]["output_path"])
        if not output_path.is_absolute():
            output_path = manifest_path.parent / output_path
        (outputs[case_id] / 2).tofile(output_path)
        entries.append(
            {
                "slice_id": slice_id,
                "dpu_index": slice_id,
                "allocated": True,
                "release_confirmed": True,
                "package_transferred": True,
                "input_count": 2,
                "inputs_transferred": True,
                "partial_output_elements": 2,
                "partial_output_bytes": 8,
                "partial_output_raw_bytes": 8,
                "partial_output_transfer_bytes": 8,
                "partial_output_path": str(output_path),
                "partial_output_read": True,
                "partial_output_written": True,
                "completion_confirmed": True,
                "operation_count": 1,
                "completed_operation_count": 1,
                "observed_operation_completion_count": 1,
                "operation_completion_confirmed": True,
                "manifest_path": str(manifest_path),
                "manifest_fnv1a64": _file_fingerprints(manifest_path)["fnv1a64"],
            }
        )
    payload = {
        "schema_version": "generic_loop_resident_two_dpu_contraction_slice_v1",
        "manifest_kind": "resident_two_slice_response",
        "status": "completed",
        "failure_stage": None,
        "backend_id": adapter.BACKEND_ID,
        "backend_family": "upmem_sdk",
        "target_requested": "hardware",
        "target_observed": "hardware",
        "hardware_profile_version": adapter.PROFILE_VERSION,
        "cpu_fallback_used": False,
        "topology": "two_dpu_allocation",
        "hardware_execution": True,
        "hardware_kernel_executed": True,
        "hardware_functionality_evidence": True,
        "simulator_kernel_executed": False,
        "tasklets_per_dpu": 1,
        "operation_count": 1,
        "async_launch_count": 1,
        "synchronize_count": 1,
        "observed_operation_completion_counts": [1, 1],
        "device_launch_mode": "asynchronous_dpu_set",
        "host_completion_mode": "blocking_sync",
        "completion_evidence": "host_observed_dpu_set_sync_and_final_output_read",
        "timing_scope": "host_observed_sdk_stage_boundaries",
        "timing_is_bringup_only": True,
        "timing": {
            "clock": "clock_monotonic",
            "sync_wait_is_not_pure_kernel_time": True,
            "kernel_time_s": None,
            **{
                name: 0.001
                for name in (
                    "package_parse_time_s",
                    "allocation_time_s",
                    "binary_load_time_s",
                    "initial_h2d_time_s",
                    "operation_control_h2d_time_s",
                    "launch_enqueue_time_s",
                    "sync_wait_time_s",
                    "final_d2h_time_s",
                    "output_write_time_s",
                    "release_time_s",
                    "total_route_time_s",
                )
            },
        },
        "native_reconstruction_performed": False,
        "reconstruction_contract": "python_sum_partials",
        "allocation": {
            "requested_dpus": 2,
            "allocated_dpus": 2,
            "profile": "backend=hw",
            "verified": True,
        },
        "launch": {
            "mode": "asynchronous",
            "device_launch_mode": "asynchronous_dpu_set",
            "host_completion_mode": "blocking_sync",
            "operation_count": 1,
            "async_launch_count": 1,
            "synchronize_count": 1,
            "completed": True,
        },
        "release": {"attempted": True, "confirmed": True},
        "slices": entries,
    }
    write_json(response_path, payload)
    return payload


def test_prepare_only_writes_all_plans_without_adapter_execution(
    tmp_path, monkeypatch
) -> None:
    calls: list[object] = []
    monkeypatch.setattr(
        mvp,
        "execute_sliced_resident_hardware_session",
        lambda *args, **kwargs: calls.append(args),
    )

    result = mvp.prepare_upmem_hardware_sliced_resident_mvp(
        tmp_path, suite_path=SUITE, environment={}
    )

    assert result.status == "prepared"
    assert calls == []
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert len(summary["prepared_operations"]) == 12
    assert summary["dpu_allocation_attempted"] is False
    assert (
        result.plan_dir
        / "cases"
        / "one_qubit_x_m2"
        / "measured_02"
        / "slice_0_manifest.json"
    ).is_file()


@pytest.mark.parametrize(
    ("returncode", "parser_status", "expected_status"),
    ((0, "valid", "prepared"), (1, "invalid", "failed")),
)
def test_prepare_build_runs_native_parse_only_without_allocation_or_launch(
    tmp_path, monkeypatch, returncode, parser_status, expected_status
) -> None:
    calls: list[tuple[object, dict[str, object]]] = []
    monkeypatch.setattr(mvp, "build_sliced_resident_hardware_session", _fake_build)

    def parse_only(command, **kwargs):
        if "--validate-slice-packages" not in command:
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        calls.append((command, kwargs))
        assert command[1] == "--validate-slice-packages"
        assert "--slice-package-0" not in command
        return SimpleNamespace(
            returncode=returncode,
            stdout=json.dumps({"status": parser_status, "reason": None}),
            stderr="",
        )

    monkeypatch.setattr(mvp.subprocess, "run", parse_only)
    result = mvp.prepare_upmem_hardware_sliced_resident_mvp(
        tmp_path,
        suite_path=M2_1_SUITE,
        build=True,
        environment={},
    )

    assert result.status == expected_status
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    validation = summary["native_manifest_validation"]
    assert validation["status"] == (
        "passed" if expected_status == "prepared" else "failed"
    )
    assert len(validation["entries"]) == 4
    assert len(calls) == 4
    assert summary["dpu_allocation_attempted"] is False
    assert summary["dpu_launch_attempted"] is False
    assert all(
        entry["dpu_allocation_attempted"] is False
        and entry["dpu_launch_attempted"] is False
        for entry in validation["entries"]
    )
    rows = summary["prepared_operations"]
    assert all(
        row["native_manifest_validation"]["status"]
        == ("passed" if expected_status == "prepared" else "failed")
        for row in rows
    )
    if expected_status == "prepared":
        assert validation["reason"] is None


def test_prepare_without_build_keeps_native_validation_explicitly_unrun(tmp_path) -> None:
    result = mvp.prepare_upmem_hardware_sliced_resident_mvp(
        tmp_path,
        suite_path=M2_1_SUITE,
        build=False,
        environment={},
    )
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))

    assert summary["native_manifest_validation"] == {
        "status": "not_run",
        "reason": "native host build was not requested",
        "entries": [],
        "dpu_allocation_attempted": False,
        "dpu_launch_attempted": False,
    }


def test_failure_record_preserves_observed_native_identity_without_success(
    tmp_path,
) -> None:
    m2 = mvp.load_m2_suite(M2_1_SUITE)
    response = {
        "backend_family": "upmem_sdk",
        "target_observed": "hardware",
        "hardware_execution": True,
        "hardware_kernel_executed": True,
        "simulator_kernel_executed": False,
        "cpu_fallback_used": False,
    }

    record = mvp._failure_record(
        m2,
        "output_validation_failed",
        {"case_id": "fixture", "workload_id": "fixture"},
        "execute",
        1,
        "output_validation_failed",
        observed_response=response,
    )

    assert record["status"] == "failed"
    assert record["backend_family"] == "upmem_sdk"
    assert record["target_observed"] == "hardware"
    assert record["hardware_execution"] is True
    assert record["native_kernel_executed"] is True
    assert record["hardware_kernel_executed"] is True
    assert record["simulator_kernel_executed"] is False
    assert record["cpu_fallback_used"] is False
    assert record["validation_status"] == "not_run"

    pre_execution = mvp._failure_record(
        m2,
        "native_build_failed",
        None,
        "execute",
        None,
        "native_build_failed",
    )
    assert pre_execution["backend_family"] is None
    assert pre_execution["target_observed"] == "not_observed"
    assert pre_execution["hardware_execution"] is False
    assert pre_execution["native_kernel_executed"] is False
    assert pre_execution["simulator_kernel_executed"] is False
    assert pre_execution["cpu_fallback_used"] is False


def test_binary_manifest_binding_rejects_malformed_final_bytes(tmp_path) -> None:
    m2 = mvp.load_m2_suite(M2_1_SUITE)
    prepared = mvp._prepare_case(ROOT, m2.suite["cases"][0], m2)
    dpu_binary = tmp_path / "dpu_resident"
    dpu_binary.write_bytes(b"fixture")
    artifacts = mvp._write_packages(
        prepared,
        m2,
        dpu_binary,
        tmp_path / "artifacts",
        prefix="test",
    )
    manifest_path = artifacts["packages"][0].package.manifest_path
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["final_outputs"][0]["raw_bytes"] += 4
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="binary_mismatch"):
        mvp._validate_manifest_bindings_against_binary(artifacts["packages"])


def test_execution_uses_one_fake_session_and_preserves_m2_evidence(
    tmp_path, monkeypatch
) -> None:
    builds: list[object] = []

    def build(*args, **kwargs):
        value = _fake_build(*args, **kwargs)
        builds.append(value)
        return value

    monkeypatch.setattr(mvp, "build_sliced_resident_hardware_session", build)
    monkeypatch.setattr(mvp, "execute_sliced_resident_hardware_session", _fake_execute)

    result = mvp.run_upmem_hardware_sliced_resident_mvp(
        tmp_path, suite_path=SUITE, environment={"UPMEM_ALLOW_PHYSICAL_HARDWARE": "1"}
    )

    assert result.status == "completed"
    assert len(builds) == 1
    measured = read_jsonl(result.run_dir / "normalized_records.jsonl")
    warmups = read_jsonl(result.run_dir / "warmups.jsonl")
    assert len(measured) == 9
    assert len(warmups) == 3
    assert {row["phase"] for row in measured} == {"measured"}
    assert all(row["validation_status"] == "passed" for row in measured)
    assert all(row["allocated_dpu_count"] == 2 for row in measured)
    assert all(
        row["slice_count"] == 2 and row["tasklets_per_dpu"] == 1 for row in measured
    )
    for row in measured:
        assert row["application_visible_transfer_bytes"] == (
            row["application_visible_h2d_bytes"] + row["application_visible_d2h_bytes"]
        )
        assert row["actual_transfer_bytes"] == (
            row["actual_h2d_bytes"] + row["actual_d2h_bytes"]
        )
        assert row["actual_transfer_bytes"] == row["application_visible_transfer_bytes"]
    x_row = next(row for row in measured if row["case_id"] == "one_qubit_x_m2")
    expected_hash = hashlib.sha256(
        (ROOT / "configs/circuits/upmem_m2/one_qubit_x.qasm").read_bytes()
    ).hexdigest()
    assert x_row["qasm_source_sha256"] == expected_hash
    assert (
        x_row["source_hashes"]["circuit_semantics_hash"]
        == x_row["circuit_semantics_hash"]
    )
    assert (
        result.run_dir
        / "cases"
        / "one_qubit_h_m2"
        / "measured_01"
        / "reconstructed_output.npy"
    ).is_file()


def test_execute_requires_opt_in_and_retains_failure_row(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        mvp,
        "build_sliced_resident_hardware_session",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must not build")),
    )

    result = mvp.run_upmem_hardware_sliced_resident_mvp(
        tmp_path, suite_path=SUITE, environment={}
    )

    assert result.status == "failed"
    rows = read_jsonl(result.run_dir / "normalized_records.jsonl")
    assert len(rows) == 1
    assert rows[0]["failure_stage"] == "hardware_opt_in_missing"
    assert rows[0]["application_visible_transfer_bytes"] == 0
    assert rows[0]["actual_transfer_bytes"] == 0


def test_native_failure_is_preserved_for_each_measured_operation(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(mvp, "build_sliced_resident_hardware_session", _fake_build)

    def failed(build, *, response_path, **kwargs):
        return SlicedResidentSessionExecution(
            status="failed",
            failure_stage="kernel_launch_failed",
            response_path=response_path,
            response={"status": "failed", "failure_stage": "kernel_launch_failed"},
            process_time_s=0.125,
            command=("fake-host", "--slice-package-0"),
            stdout_snippet="native stdout",
            stderr_snippet="native stderr",
            timed_out=False,
            cleanup_confirmed=True,
        )

    monkeypatch.setattr(mvp, "execute_sliced_resident_hardware_session", failed)
    result = mvp.run_upmem_hardware_sliced_resident_mvp(
        tmp_path, suite_path=SUITE, environment={"UPMEM_ALLOW_PHYSICAL_HARDWARE": "1"}
    )

    rows = read_jsonl(result.run_dir / "normalized_records.jsonl")
    assert result.status == "failed"
    assert len(rows) == 9
    assert {row["failure_stage"] for row in rows} == {"kernel_launch_failed"}
    assert all(
        row["native_response_artifact"].startswith("native_session/") for row in rows
    )
    assert all(len(row["package_manifest_artifacts"]) == 2 for row in rows)
    assert all(
        row["native_session_command"] == ["fake-host", "--slice-package-0"]
        for row in rows
    )
    assert all(row["native_stdout_snippet"] == "native stdout" for row in rows)
    assert all(row["native_stderr_snippet"] == "native stderr" for row in rows)
    assert all(row["native_failure_stage"] == "kernel_launch_failed" for row in rows)
    assert all(
        row["application_visible_transfer_bytes"]
        == row["application_visible_h2d_bytes"] + row["application_visible_d2h_bytes"]
        == row["actual_transfer_bytes"]
        for row in rows
    )


def test_runner_rejects_copied_or_noncanonical_suite_paths(tmp_path) -> None:
    copied = tmp_path / SUITE.name
    copied.write_text(SUITE.read_text(encoding="utf-8"), encoding="utf-8")

    with pytest.raises(ValueError, match="canonical|committed"):
        mvp.load_m2_suite(copied)


def test_m2_1_fixture_is_canonical_and_requires_the_new_contract() -> None:
    loaded = mvp.load_m2_suite(M2_1_SUITE)

    assert loaded.fixture_version == "upmem_hardware_sliced_resident_m2_1_v1"
    assert loaded.fixture_scope == (
        "two_operation_h_then_x_full_graph_replicated_prefix"
    )
    assert loaded.require_nonzero_slice_partials is True
    assert loaded.suite["warmups"] == 1
    assert loaded.suite["repeats"] == 3
    assert [case["case_id"] for case in loaded.suite["cases"]] == [
        "one_qubit_hx_m2_1"
    ]


def test_m2_1_core_integration_materializes_graph_wide_slices() -> None:
    loaded = mvp.load_m2_suite(M2_1_SUITE)
    prepared = mvp._prepare_case(ROOT, loaded.suite["cases"][0], loaded)

    assert len(prepared["graph"].tasks) == 2
    assert set(prepared["reference_partials"]) == {0, 1}
    assert all(
        np.linalg.norm(partial) > mvp.SLICE_NONZERO_THRESHOLD
        for partial in prepared["reference_partials"].values()
    )
    np.testing.assert_allclose(
        sum(prepared["reference_partials"].values()),
        prepared["reference"],
        atol=1.0e-6,
        rtol=1.0e-6,
    )


def test_m2_1_record_contract_preserves_useful_slice_and_claim_metadata(tmp_path) -> None:
    loaded = mvp.load_m2_suite(M2_1_SUITE)
    case = loaded.suite["cases"][0]
    circuit = mvp.load_circuit(case, ROOT)
    network = mvp.build_tensor_network(circuit)
    graph = mvp.with_execution_identity(
        mvp.plan_task_graph_with_config(network, loaded.suite["planner"])
    )
    prepared = {
        "circuit": circuit,
        "graph": graph,
        "fixture_version": loaded.fixture_version,
        "fixture_scope": loaded.fixture_scope,
        "source_task_count": 2,
        "tensor_count": 3,
        "selected_task_id": graph.tasks[-1].id,
        "qasm_source_sha256": "qasm-hash",
    }
    manifest_paths = []
    packages = []
    for slice_id in (0, 1):
        manifest_path = tmp_path / f"slice_{slice_id}.json"
        write_json(
            manifest_path,
            {
                "initial_h2d_bytes": 16,
                "descriptor_h2d_bytes": 16,
                "control_h2d_bytes": 0,
                "final_d2h_bytes": 8,
            },
        )
        manifest_paths.append(manifest_path)
        packages.append(
            SimpleNamespace(
                slice_id=slice_id,
                package=SimpleNamespace(
                    descriptor_sha256=f"descriptor-{slice_id}",
                    manifest_path=manifest_path,
                ),
            )
        )
    artifacts = {
        "packages": packages,
        "manifest_paths": tuple(manifest_paths),
        "validation": {
            "source_hashes": {
                "circuit_semantics_hash": "circuit-hash",
                "tensor_network_hash": "network-hash",
                "contraction_plan_hash": "plan-hash",
            },
            "validated": True,
        },
    }
    session = SimpleNamespace(
        response={
            "backend_id": adapter.BACKEND_ID,
            "hardware_execution": True,
            "target_requested": "hardware",
            "target_observed": "hardware",
            "backend_family": "upmem_sdk",
            "hardware_kernel_executed": True,
            "simulator_kernel_executed": False,
            "cpu_fallback_used": False,
            "operation_count": 2,
            "observed_operation_completion_counts": [2, 2],
            "completion_sentinel_read_counts": [2, 2],
            "native_execution_sentinel_available": True,
            "completion_evidence": "dpu_written_completion_sentinel_read_after_each_sync",
            "device_completion_confirmed": True,
            "actual_h2d_bytes": 64,
            "actual_d2h_bytes": 16,
            "actual_transfer_bytes": 80,
            "allocation": {"allocated_dpus": 2, "verified": True},
            "launch": {
                "completed": True,
                "operation_count": 2,
                "async_launch_count": 2,
                "synchronize_count": 2,
            },
            "release": {"confirmed": True},
            "slices": [
                {
                    "completion_confirmed": True,
                    "dpu_completion_sentinel": {
                        "verified": True,
                        "active_operation_index": 1,
                        "completion_status": 1,
                        "completed_operation_count": 2,
                        "output_elements_processed": 2,
                    },
                    "completion_sentinel_read_count": 2,
                },
                {
                    "completion_confirmed": True,
                    "dpu_completion_sentinel": {
                        "verified": True,
                        "active_operation_index": 1,
                        "completion_status": 1,
                        "completed_operation_count": 2,
                        "output_elements_processed": 2,
                    },
                    "completion_sentinel_read_count": 2,
                },
            ],
            "timing": {"clock": "monotonic", "status": "aggregate"},
        },
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
    output = np.asarray(case["expected_output"], dtype=np.float32)
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
            "partial_outputs": {
                "0": [0.0, 0.70710678],
                "1": [0.70710678, 0.0],
            },
            "per_slice_output_validation_status": "passed",
            "slice_useful_work": {"status": "passed"},
        },
        output=output,
        cpu_ok=True,
        expected_ok=True,
        reconstruction_time_s=0.01,
        total_time_s=0.2,
    )

    assert record["execution_scope"] == (
        "physical_two_dpu_two_slice_full_replicated_prefix_taskgraph"
    )
    assert record["gate_count"] == 2
    assert record["task_count"] == 2
    assert record["source_task_count"] == 2
    assert record["expanded_task_count"] == 4
    assert record["executed_task_count"] == 4
    assert record["slice_model_task_count"] == 2
    assert record["slice_model_executed_task_count"] == 4
    assert record["slice_parallel_execution"] is False
    assert record["slice_overlap_measured"] is False
    assert record["device_launch_mode"] == "asynchronous_dpu_set"
    assert record["host_completion_mode"] == "blocking_sync"
    assert record["target_requested"] == "hardware"
    assert record["target_observed"] == "hardware"
    assert record["backend_family"] == "upmem_sdk"
    assert record["operation_count"] == 2
    assert record["native_kernel_executed"] is True
    assert record["hardware_kernel_executed"] is True
    assert record["simulator_kernel_executed"] is False
    assert record["validation_status"] == "passed"
    assert record["hardware_functionality_evidence"] is True
    assert record["hardware_speedup_applicable"] is False
    assert record["energy_measurement_available"] is False

    # A self-consistent native byte total is still inadmissible when it does
    # not match the planned manifest scope.
    session.response["actual_h2d_bytes"] = 65
    session.response["actual_transfer_bytes"] = 81
    mismatched = mvp._record(
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
            "partial_outputs": {
                "0": [0.0, 0.70710678],
                "1": [0.70710678, 0.0],
            },
            "per_slice_output_validation_status": "passed",
            "slice_useful_work": {"status": "passed"},
        },
        output=output,
        cpu_ok=True,
        expected_ok=True,
        reconstruction_time_s=0.01,
        total_time_s=0.2,
    )
    assert mismatched["status"] == "failed"
    assert mismatched["transfer_accounting_status"] == "failed"
    assert mismatched["transfer_accounting_invariant"] is True
    assert mismatched["transfer_matches_manifest_plan"] is False
    assert mismatched["validation_status"] == "failed"


def test_m2_1_scientific_validation_failure_controls_record_status(tmp_path) -> None:
    loaded = mvp.load_m2_suite(M2_1_SUITE)
    case = loaded.suite["cases"][0]
    circuit = mvp.load_circuit(case, ROOT)
    network = mvp.build_tensor_network(circuit)
    graph = mvp.with_execution_identity(
        mvp.plan_task_graph_with_config(network, loaded.suite["planner"])
    )
    prepared = {
        "circuit": circuit,
        "graph": graph,
        "fixture_version": loaded.fixture_version,
        "fixture_scope": loaded.fixture_scope,
        "source_task_count": 2,
        "tensor_count": 3,
        "selected_task_id": graph.tasks[-1].id,
        "qasm_source_sha256": "qasm-hash",
    }
    manifest_paths = []
    packages = []
    for slice_id in (0, 1):
        manifest_path = tmp_path / f"invalid_slice_{slice_id}.json"
        write_json(
            manifest_path,
            {
                "initial_h2d_bytes": 16,
                "descriptor_h2d_bytes": 16,
                "control_h2d_bytes": 0,
                "final_d2h_bytes": 8,
            },
        )
        manifest_paths.append(manifest_path)
        packages.append(
            SimpleNamespace(
                slice_id=slice_id,
                package=SimpleNamespace(
                    descriptor_sha256=f"descriptor-{slice_id}",
                    manifest_path=manifest_path,
                ),
            )
        )
    session = SimpleNamespace(
        response={
            "hardware_execution": True,
            "cpu_fallback_used": False,
            "allocation": {"allocated_dpus": 2, "verified": True},
            "launch": {"completed": True, "synchronize_count": 1},
            "release": {"confirmed": True},
        },
        response_path=tmp_path / "invalid_response.json",
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
        artifacts={
            "packages": packages,
            "manifest_paths": tuple(manifest_paths),
            "validation": {"source_hashes": {}},
        },
        reconstruction={
            "partial_outputs": {"0": [0.0, 0.0], "1": [0.0, 0.0]},
            "per_slice_output_validation_status": "failed",
            "slice_useful_work": {"status": "failed"},
        },
        output=np.asarray(case["expected_output"], dtype=np.float32),
        cpu_ok=True,
        expected_ok=True,
        reconstruction_time_s=0.01,
        total_time_s=0.2,
    )

    assert record["scientific_validation_status"] == "failed"
    assert record["validation_status"] == "failed"
    assert record["status"] == "failed"


def test_runner_uses_real_adapter_session_path_and_command_contract(
    tmp_path, monkeypatch
) -> None:
    calls: list[tuple[tuple[str, ...], Path, dict[str, str], float]] = []
    monkeypatch.setattr(mvp, "build_sliced_resident_hardware_session", _fake_build)

    def physical(command, *, cwd, env, timeout_s):
        command = tuple(command)
        calls.append((command, cwd, dict(env), timeout_s))
        _write_completed_native_response(
            (Path(command[2]), Path(command[4])), Path(command[6])
        )
        return {
            "returncode": 0,
            "elapsed_s": 0.25,
            "timed_out": False,
            "cleanup_confirmed": True,
            "stdout_snippet": "low-level stdout",
            "stderr_snippet": "",
        }

    monkeypatch.setattr(adapter, "_run_physical_command", physical)
    result = mvp.run_upmem_hardware_sliced_resident_mvp(
        tmp_path, suite_path=SUITE, environment={"UPMEM_ALLOW_PHYSICAL_HARDWARE": "1"}
    )

    assert result.status == "completed"
    assert len(calls) == 12
    for command, cwd, environment, timeout_s in calls:
        session_root = cwd.parent
        assert command[1] == "--slice-package-0"
        assert command[3] == "--slice-package-1"
        assert command[5] == "--resident-response"
        assert Path(command[2]).is_relative_to(session_root)
        assert Path(command[4]).is_relative_to(session_root)
        assert Path(command[6]).is_relative_to(session_root)
        assert environment == {"UPMEM_ALLOW_PHYSICAL_HARDWARE": "1"}
        assert timeout_s == 30.0


def test_cli_and_make_targets_expose_the_internal_research_mvp() -> None:
    command = subprocess.run(
        [sys.executable, "-m", "quantum_bench.bench", "--help"],
        cwd=ROOT,
        env={"PYTHONPATH": "src"},
        capture_output=True,
        text=True,
        check=False,
    )
    plan = subprocess.run(
        ["make", "-n", "upmem-hw-sliced-resident-plan"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    execute = subprocess.run(
        ["make", "-n", "upmem-hw-sliced-resident"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert command.returncode == 0
    assert "upmem-hardware-sliced-resident-mvp" in command.stdout
    assert plan.returncode == execute.returncode == 0
    assert "--prepare-only --build" in plan.stdout
    assert "UPMEM_ALLOW_PHYSICAL_HARDWARE" in execute.stdout
    assert "--execute" in execute.stdout
