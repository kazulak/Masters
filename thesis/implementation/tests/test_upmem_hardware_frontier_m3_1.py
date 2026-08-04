from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import json

import numpy as np
import pytest

from quantum_bench.bench import upmem_hardware_frontier_m3_1 as m31
from quantum_bench.circuits import load_circuit
from quantum_bench.tn import build_tensor_network, plan_task_graph_with_config
from quantum_bench.targets.upmem.hardware_taskgraph_frontier import (
    BACKEND_ID,
    NATIVE_SCHEMA,
    PROFILE_ID,
    REQUEST_SCHEMA,
    ROUTE_ID,
    TIMING_ALIAS_FIELDS,
    TIMING_COMPONENT_FIELDS,
)


ROOT = Path(__file__).parents[1]
SUITE_PATH = ROOT / "configs" / "suites" / "upmem_hardware_frontier_m3_1.yml"


def _suite() -> m31.M31Suite:
    return m31.load_upmem_hardware_frontier_m3_1_suite(SUITE_PATH)


def _fake_build(root: Path) -> SimpleNamespace:
    session = root / "native_session"
    build_dir = session / "native" / "build"
    build_dir.mkdir(parents=True)
    (build_dir / "host_frontier_two_dpu").write_bytes(b"host")
    (build_dir / "dpu_frontier_two_dpu").write_bytes(b"dpu")
    return SimpleNamespace(
        session_root=session,
        build_dir=build_dir,
        host_binary=build_dir / "host_frontier_two_dpu",
        dpu_binary=build_dir / "dpu_frontier_two_dpu",
        source_tree_hash="source",
        host_binary_hash="host",
        dpu_binary_hash="dpu",
        build_time_s=0.01,
        build_command=("make", "all"),
        sdk_tools={},
    )


def _fnv1a64(value: bytes) -> str:
    result = 14695981039346656037
    for byte in value:
        result ^= byte
        result = (result * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return f"{result:016x}"


def _native_response(
    manifest: dict[str, object], *, output_root: Path, bad_transfer: bool = False
) -> dict[str, object]:
    operations = manifest["frontier_task_operations"]
    assert isinstance(operations, list)
    expected_transfer = manifest["expected_frontier_transfer"]
    assert isinstance(expected_transfer, dict)
    transfer = dict(expected_transfer)
    if bad_transfer:
        transfer["total_bytes"] += 1
    final_binding = manifest["final_output_binding"]
    assert isinstance(final_binding, dict)
    output = output_root / final_binding["output_path"]
    output.parent.mkdir(parents=True, exist_ok=True)
    raw_output = np.asarray([0.7986355, 0.6018150], dtype="<f4").tobytes()
    output.write_bytes(raw_output)
    output_path = final_binding["output_path"]
    output_hash = _fnv1a64(raw_output)
    timing = {
        key: 0.001 for key in (*TIMING_COMPONENT_FIELDS, *TIMING_ALIAS_FIELDS)
    }
    timing["total_route_time_s"] = sum(timing[key] for key in TIMING_COMPONENT_FIELDS)
    return {
        "schema_version": NATIVE_SCHEMA,
        "native_schema_version": NATIVE_SCHEMA,
        "manifest_kind": "frontier_two_dpu_response",
        "error": None,
        "failure_context": None,
        "sdk_error_code": 0,
        "route_id": ROUTE_ID,
        "profile_id": PROFILE_ID,
        "backend_id": BACKEND_ID,
        "hardware_profile_version": PROFILE_ID,
        "target_requested": "hardware",
        "target_observed": "hardware",
        "numeric_mode": "none",
        "tasklets_per_dpu": 1,
        "requested_dpus": 2,
        "allocated_dpus": 2,
        "status": "completed",
        "failure_stage": None,
        "hardware_execution": True,
        "hardware_kernel_executed": True,
        "hardware_functionality_evidence": True,
        "cpu_fallback_used": False,
        "simulator_kernel_executed": False,
        "no_cpu_fallback": True,
        "no_simulator_fallback": True,
        "native_failure_fallback_used": False,
        "hardware_no_fallback": True,
        "performance_claim_applicable": False,
        "physical_dpu_count": 2,
        "operation_count": 3,
        "numeric_contract": "float32_real",
        "complex_combine_used": False,
        "quantization_mode": "none",
        "timing_scope": "two_dpu_frontier_resident_full_taskgraph_v1",
        "timing": {"clock": "clock_monotonic", "overlap_measured": False, **timing, "kernel_time_s": None},
        "allocation": {"attempted": True, "requested_dpus": 2, "allocated_dpus": 2, "profile": "backend=hw", "verified": True},
        "load": {"attempted": True, "succeeded": True, "confirmed": True, "hardware": True},
        "launch": {
            "wave0_attempted": True, "wave0_synchronized": True,
            "wave1_attempted": True, "wave1_synchronized": True,
            "async_launch_count": 2, "completed": True, "task_count": 3, "barrier_count": 2,
        },
        "release": {"attempted": True, "confirmed": True},
        "co_dispatch_observed": True,
        "co_dispatch_confirmed": True,
        "overlap_measurement": "unmeasured",
        "overlap_measured": False,
        "overlap_claim": "unmeasured",
        "overlap_evidence": "co_dispatch_without_overlap_measurement",
        "wave0_complete_before_wave1": True,
        "completed_task_ids": ["task_0", "task_1", "task_2"],
        "completed_task_ids_scope": "wave_dependency_order_not_intra_wave_finish_order",
        "wave_plan": [
            {"wave": 0, "assignments": [{"dpu": 0, "operation": 0}, {"dpu": 1, "operation": 1}], "launch": "dpu_set_async", "synchronize": "dpu_sync_set", "barrier": True},
            {"wave": 1, "assignments": [{"dpu": 0, "operation": 2}], "launch": "dpu0_async", "synchronize": "dpu_sync_dpu0", "barrier": True},
        ],
        "wave_barrier_count": 2,
        "physical_task_instances": [
            {"instance": 0, "dpu": 0, "operation": 0},
            {"instance": 1, "dpu": 1, "operation": 1},
            {"instance": 2, "dpu": 0, "operation": 2},
        ],
        "per_dpu_completed_operations": [2, 1],
        "completion_sentinels": [
            {"dpu": 0, "operation": 0, "read": True, "verified": True, "magic": 1381194576, "version": 1, "status": 1, "completed_operation_count": 1, "output_elements": 2},
            {"dpu": 1, "operation": 1, "read": True, "verified": True, "magic": 1381194576, "version": 1, "status": 1, "completed_operation_count": 2, "output_elements": 2},
            {"dpu": 0, "operation": 2, "read": True, "verified": True, "magic": 1381194576, "version": 1, "status": 1, "completed_operation_count": 3, "output_elements": 2},
        ],
        "barrier_count": 2,
        "barriers": [
            {"barrier_index": 0, "wave_index": 0, "completed": True},
            {"barrier_index": 1, "wave_index": 1, "completed": True},
        ],
        "observed_dpu_task_counts": [2, 1],
        "actual_h2d_bytes": expected_transfer["h2d_bytes"],
        "actual_d2h_bytes": expected_transfer["d2h_bytes"],
        "actual_transfer_bytes": transfer["total_bytes"],
        "transfer_accounting_scope": "native_sdk_observed_application_visible",
        "stage_flags": {"inter_wave_d2h": True, "inter_wave_h2d": True, "final_d2h": True, "output_written": True},
        "transfer": {
            **transfer,
            "transfer_invariant": not bad_transfer,
            "accounting_scope": "sdk_argument_byte_counts",
        },
        "tasks": [{**item, "completed": True, "completion_confirmed": True} for item in operations],
        "final_output": {
            **final_binding,
            "output_path": output_path,
            "hash_fnv1a64": output_hash,
            "path": output_path,
            "written": True,
        },
        "hashes": {
            "manifest_fnv1a64": "0" * 16,
            "package_fnv1a64": "0" * 16,
            "dpu_binary_fnv1a64": "0" * 16,
            "host_source_fnv1a64": "0" * 16,
            "final_output_file_fnv1a64": output_hash,
        },
    }


def test_strict_suite_shape_and_actual_qasm_graph() -> None:
    suite = _suite()
    case = suite.suite["cases"][0]
    circuit = load_circuit(case, ROOT)
    network = build_tensor_network(circuit)
    graph = plan_task_graph_with_config(network, suite.suite["planner"])
    assert case["circuit"]["path"] == "configs/circuits/upmem_m2/one_qubit_ry_h_ry_a.qasm"
    assert circuit.source["kind"] == "qasm_file"
    assert graph.path == ((0, 1), (0, 1), (0, 1))
    assert len(graph.tasks) == 3
    assert suite.suite["warmups"] == 1
    assert suite.suite["repeats"] == 5

    with pytest.raises(ValueError, match="committed suite"):
        m31.load_upmem_hardware_frontier_m3_1_suite(ROOT / "configs" / "suites" / "upmem_hardware_sliced_resident_m2_3.yml")


def test_prepare_materializes_frontier_without_dpu_allocation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(m31, "_unique_dir", lambda _parent: tmp_path / "plan")
    monkeypatch.setattr(m31, "build_hardware_frontier_session", pytest.fail)
    result = m31.prepare_upmem_hardware_frontier_m3_1(ROOT, suite_path=SUITE_PATH)
    payload = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert result.status == "prepared"
    assert payload["dpu_allocation_attempted"] is False
    assert payload["dpu_launch_attempted"] is False
    assert payload["prepared_case"]["task_count"] == 3
    assert payload["prepared_case"]["frontier_plan"]["expected_dpu_task_counts"] == [2, 1]


def test_prepare_build_path_never_executes_or_allocates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(m31, "_unique_dir", lambda _parent: tmp_path / "plan")
    monkeypatch.setattr(
        m31,
        "build_hardware_frontier_session",
        lambda *_args, **_kwargs: _fake_build(tmp_path / "plan"),
    )
    monkeypatch.setattr(
        m31,
        "_native_validate_only",
        lambda *_args, **_kwargs: {"status": "passed", "allocation_attempted": False, "launch_attempted": False},
    )
    monkeypatch.setattr(m31, "execute_hardware_frontier_session", pytest.fail)
    result = m31.prepare_upmem_hardware_frontier_m3_1(
        ROOT,
        suite_path=SUITE_PATH,
        build=True,
        environment={"UPMEM_ALLOW_PHYSICAL_HARDWARE": "1"},
    )
    payload = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert result.status == "prepared"
    assert payload["native_validation"] == "native_session/prepare_validation.json"
    assert payload["dpu_allocation_attempted"] is False
    assert payload["dpu_launch_attempted"] is False


def test_fake_run_has_one_warmup_and_exact_five_measured_requests(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    suite = _suite()
    run_dir = tmp_path / "run"
    (run_dir / "config").mkdir(parents=True)
    (run_dir / "cases").mkdir()
    build = _fake_build(run_dir)
    calls: list[str] = []

    def fake_build(*_args, **_kwargs):
        return build

    def fake_execute(_build, *, manifest_path, response_path, **_kwargs):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        request_id = manifest["session_id"]
        calls.append(request_id)
        output = build.session_root / manifest["final_output_binding"]["output_path"]
        output.parent.mkdir(parents=True, exist_ok=True)
        prepared = json.loads((run_dir / "cases" / suite.suite["cases"][0]["case_id"] / "task_graph.json").read_text(encoding="utf-8"))
        del prepared
        np.asarray([0.7986355, 0.6018150], dtype="<f4").tofile(output)
        response = _native_response(manifest, output_root=build.session_root)
        response_path.write_text(json.dumps(response), encoding="utf-8")
        return SimpleNamespace(
            status="completed", failure_stage=None, response_path=response_path, response=response
        )

    monkeypatch.setattr(m31, "create_run_dir", lambda *_args, **_kwargs: run_dir)
    monkeypatch.setattr(m31, "build_hardware_frontier_session", fake_build)
    monkeypatch.setattr(m31, "execute_hardware_frontier_session", fake_execute)
    result = m31.run_upmem_hardware_frontier_m3_1(
        ROOT,
        suite_path=SUITE_PATH,
        environment={"UPMEM_ALLOW_PHYSICAL_HARDWARE": "1"},
    )
    rows = [json.loads(line) for line in (run_dir / "normalized_records.jsonl").read_text().splitlines()]
    warmups = [json.loads(line) for line in (run_dir / "warmups.jsonl").read_text().splitlines()]
    assert result.status == "completed"
    assert result.row_count == 5
    assert len(rows) == 5
    assert len(warmups) == 1
    assert calls == ["warmup-00", "measured-00", "measured-01", "measured-02", "measured-03", "measured-04"]
    assert all(row["validation_status"] == "passed" for row in rows)
    for row in rows:
        assert row["hardware_functionality_evidence"] is True
        assert row["hardware_kernel_executed"] is True
        assert row["requested_dpu_count"] == 2
        assert row["allocated_dpu_count"] == 2
        assert row["tasklets_per_dpu"] == 1
        assert row["co_dispatch_observed"] is True
        assert row["co_dispatch_confirmed"] is True
        assert row["overlap_measured"] is False
        assert row["frontier_parallel_execution"] is False
        assert row["timing_is_bringup_only"] is True
        assert row["execution_plan_kind"] == "taskgraph_frontier_scheduler"
        assert row["circuit_semantics_hash"]
        assert row["tensor_network_hash"]
        assert row["contraction_plan_hash"]
        assert m31.M31_EXECUTOR_CONFIG["session_protocol"] == REQUEST_SCHEMA
        assert m31.M31_EXECUTOR_CONFIG["native_schema"] == NATIVE_SCHEMA
        assert row["executor_config_hash"] == m31.executor_config_hash(
            ROUTE_ID, m31.M31_EXECUTOR_CONFIG
        )
        assert row["execution_bundle_artifact"]["relative_path"].endswith(
            "/execution_bundle.json"
        )
        assert row["native_response_artifact"]["role"] == "native_response"
        assert row["native_response_artifact"]["relative_path"] == (
            f"native_session/{row['request_id']}_response.json"
        )
        assert row["native_response_artifact"]["retained"] is True
        assert row["validation_max_abs_error"] <= row["validation_tolerance_abs"]
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["upmem_sdk_available"] == "verified_by_execution"


def test_fake_response_mismatch_is_a_failed_measured_row(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    suite = _suite()
    prepared = m31._prepare_case(ROOT, tmp_path / "case", suite, suite.suite["cases"][0])
    build = _fake_build(tmp_path)

    def fake_execute(_build, *, manifest_path, **_kwargs):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return SimpleNamespace(status="completed", failure_stage=None, response=_native_response(manifest, output_root=build.session_root, bad_transfer=True))

    monkeypatch.setattr(m31, "execute_hardware_frontier_session", fake_execute)
    row = m31._execute_request(build, prepared, suite, {"UPMEM_ALLOW_PHYSICAL_HARDWARE": "1"}, "measured-00", warmup=False)
    assert row["status"] == "failed"
    assert row["failure_stage"] == "response_evidence_invalid"
    assert str(row["reason"]).startswith("response_validation_error:")

    def failed_execute(*_args, **_kwargs):
        return SimpleNamespace(
            status="failed",
            failure_stage="native_new_stage",
            response={
                "status": "failed",
                "failure_stage": "native_new_stage",
                "failure_context": {"wave": 1, "dpu": 0},
                "error": "native detail",
            },
        )

    monkeypatch.setattr(m31, "execute_hardware_frontier_session", failed_execute)
    failed = m31._execute_request(build, prepared, suite, {"UPMEM_ALLOW_PHYSICAL_HARDWARE": "1"}, "measured-01", warmup=False)
    assert failed["failure_stage"] == "native_new_stage"
    assert failed["native_response"]["failure_context"] == {"wave": 1, "dpu": 0}
    assert failed["parallelism_evidence_type"] == "not_observed"
    assert failed["source_task_count"] is None
    assert failed["hardware_functionality_evidence"] is False
    assert failed["hardware_kernel_executed"] is False
    assert failed["requested_dpu_count"] is None
    assert failed["co_dispatch_observed"] is False

    def output_mismatch_execute(_build, *, manifest_path, **_kwargs):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        response = _native_response(manifest, output_root=build.session_root)
        output = build.session_root / manifest["final_output_binding"]["output_path"]
        np.asarray([0.0, 0.0], dtype="<f4").tofile(output)
        return SimpleNamespace(status="completed", failure_stage=None, response=response)

    monkeypatch.setattr(m31, "execute_hardware_frontier_session", output_mismatch_execute)
    output_failed = m31._execute_request(
        build, prepared, suite, {"UPMEM_ALLOW_PHYSICAL_HARDWARE": "1"}, "measured-02", warmup=False
    )
    assert output_failed["status"] == "failed"
    assert output_failed["hardware_functionality_evidence"] is False
    assert output_failed["hardware_kernel_executed"] is False
    assert output_failed["source_task_count"] is None
    assert output_failed["validation_max_abs_error"] is None


def test_fail_fast_stops_after_failed_warmup_and_measured_repeat(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def run_with_failure(failing_request: str, run_dir: Path) -> tuple[list[str], list[dict[str, object]], list[dict[str, object]]]:
        (run_dir / "config").mkdir(parents=True)
        (run_dir / "cases").mkdir()
        build = _fake_build(run_dir)
        calls: list[str] = []

        def fake_execute(_build, *, manifest_path, response_path, **_kwargs):
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            request_id = manifest["session_id"]
            calls.append(request_id)
            if request_id == failing_request:
                return SimpleNamespace(
                    status="failed",
                    failure_stage="native_new_stage",
                    response={
                        "status": "failed",
                        "failure_stage": "native_new_stage",
                        "failure_context": {"request_id": request_id},
                    },
                )
            output = build.session_root / manifest["final_output_binding"]["output_path"]
            output.parent.mkdir(parents=True, exist_ok=True)
            np.asarray([0.7986355, 0.6018150], dtype="<f4").tofile(output)
            return SimpleNamespace(status="completed", failure_stage=None, response=_native_response(manifest, output_root=build.session_root))

        monkeypatch.setattr(m31, "create_run_dir", lambda *_args, **_kwargs: run_dir)
        monkeypatch.setattr(m31, "build_hardware_frontier_session", lambda *_args, **_kwargs: build)
        monkeypatch.setattr(m31, "execute_hardware_frontier_session", fake_execute)
        m31.run_upmem_hardware_frontier_m3_1(
            ROOT, suite_path=SUITE_PATH, environment={"UPMEM_ALLOW_PHYSICAL_HARDWARE": "1"}
        )
        def read(name: str) -> list[dict[str, object]]:
            return [json.loads(line) for line in (run_dir / name).read_text().splitlines()]

        return calls, read("warmups.jsonl"), read("normalized_records.jsonl")

    warmup_calls, warmup_rows, warmup_records = run_with_failure("warmup-00", tmp_path / "warmup")
    assert warmup_calls == ["warmup-00"]
    assert len(warmup_rows) == 1
    assert warmup_records == []
    assert warmup_rows[0]["parallelism_evidence_type"] == "not_observed"

    measured_calls, measured_rows, measured_records = run_with_failure("measured-00", tmp_path / "measured")
    assert measured_calls == ["warmup-00", "measured-00"]
    assert len(measured_rows) == 1
    assert len(measured_records) == 1
    assert measured_records[0]["failure_stage"] == "native_new_stage"
