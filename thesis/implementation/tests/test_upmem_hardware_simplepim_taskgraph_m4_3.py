from __future__ import annotations

import json
from pathlib import Path

import pytest

from quantum_bench.bench import upmem_hardware_simplepim_taskgraph_m4_3 as m43
from quantum_bench.targets.upmem.simplepim_rank1_task import build_rank1_taskgraph_workload


ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT / "configs" / "suites" / "upmem_hardware_simplepim_taskgraph_m4_3.yml"


def _workload():
    return build_rank1_taskgraph_workload()


def _identities() -> dict[str, object]:
    return m43._identity_payload(_workload().graph, case_id="m4_3_rank1_taskgraph_256")


def _response(input_sha256: str, reference: int, identities: dict[str, object]) -> dict[str, object]:
    repetitions = []
    for repeat_id, warmup in [(0, True), *[(index, False) for index in range(5)]]:
        repetitions.append({
            "repeat_id": repeat_id,
            "warmup": warmup,
            "input_hash": "b" * 16,
            "reference_int64": reference,
            "result_int64": reference,
            "output_hash": m43._int64_output_hash(reference),
            "exact_integer_match": True,
            "scatter_time_s": 1.0,
            "virtual_zip_time_s": 1.0,
            "map_time_s": 1.0,
            "reduction_time_s": 1.0,
            "total_time_s": 4.0,
        })
    return {
        "schema_version": m43.NATIVE_SCHEMA_VERSION,
        "profile_id": m43.NATIVE_PROFILE_ID,
        "backend_id": m43.NATIVE_BACKEND_ID,
        "route_id": m43.NATIVE_ROUTE_ID,
        "target_requested": "physical_hardware",
        "target_observed": "physical_hardware",
        "requested_dpu_count": 2,
        "allocated_dpu_count": 2,
        "allocation_profile": "backend=hw",
        "cpu_fallback_used": False,
        "simulator_kernel_executed": False,
        "hardware_speedup_applicable": False,
        "warmup_count": 1,
        "repeat_count": 5,
        "external_operand_transport": True,
        "operand_input_length_bytes": 512,
        "operand_input_hash": "b" * 16,
        "provider_initialized": True,
        "simplepim_operator_api_used": True,
        "operator_validations_passed": True,
        "operator_metadata_checks_passed": True,
        "all_tasks_completed": True,
        "exact_integer_match": True,
        "hardware_kernel_executed": True,
        "release_attempted": True,
        "release_confirmed": True,
        "hardware_functionality_evidence": True,
        "persistent_allocation_observed": True,
        "simplepim_managed_allocation": True,
        "status": "completed",
        "validation_status": "passed",
        "logical_payload_h2d_bytes_per_iteration": 2048,
        "logical_payload_d2h_bytes_per_iteration": 16,
        "logical_payload_transfer_bytes_per_iteration": 2064,
        "logical_payload_h2d_bytes_total_session": 12288,
        "logical_payload_d2h_bytes_total_session": 96,
        "logical_payload_transfer_bytes_total_session": 12384,
        "source_commit": "1d639c53532555f01e9f71d872e7712b166d6cba",
        "simplepim_source_commit": "1d639c53532555f01e9f71d872e7712b166d6cba",
        "host_binary_hash": "host",
        "initialization_binary_hash": "init",
        "map_binary_hash": "map",
        "genred_binary_hash": "genred",
        "genred_reduce_shared_object_hash": "reduce",
        "repetitions": repetitions,
    }


def test_suite_and_plan_profile_are_fixed() -> None:
    suite = m43.load_suite(SUITE)
    assert suite["profile"]["requested_dpu_count"] == 2
    assert suite["profile"]["tasklets_per_dpu"] == 12
    assert suite["workload"]["task_graph"]["task_count"] == 1


def test_response_requires_hardware_allocation_profile() -> None:
    payload = _response("a" * 64, _workload().reference_int64, _identities())
    payload["allocation_profile"] = "backend=simulator"
    with pytest.raises(ValueError, match="allocation_profile mismatch"):
        m43._require_response(payload, identities=_identities(), input_sha256="a" * 64, input_hash="b" * 16, reference=_workload().reference_int64)


def test_response_rejects_native_failure_without_admission() -> None:
    payload = _response("a" * 64, _workload().reference_int64, _identities())
    payload["status"] = "failed"
    with pytest.raises(ValueError, match="not completed"):
        m43._require_response(payload, identities=_identities(), input_sha256="a" * 64, input_hash="b" * 16, reference=_workload().reference_int64)


def test_happy_response_preserves_task_identity_and_operand_hash() -> None:
    identities = _identities()
    payload = _response("a" * 64, _workload().reference_int64, identities)
    m43._require_response(payload, identities=identities, input_sha256="a" * 64, input_hash="b" * 16, reference=_workload().reference_int64)
    rows = m43._records(payload, identities, _workload().graph.tasks[0], "native_execute_response.json", input_sha256="a" * 64, thesis_source_commit="thesis")
    assert len(rows) == 5
    assert rows[0]["task_graph_integrated"] is False
    assert rows[0]["taskgraph_derived_operand_adapter"] is True
    assert rows[0]["native_taskgraph_protocol"] is False
    assert rows[0]["thesis_source_commit"] == "thesis"
    assert rows[0]["simplepim_source_commit"] == payload["simplepim_source_commit"]
    assert rows[0]["per_iteration_operator_time_s"] == 4.0
    assert rows[0]["input_tensor_ids"] == ["rank1_left", "rank1_right"]
    assert rows[0]["scientific_input_file_bytes"] == 512
    assert rows[0]["application_visible_h2d_bytes"] == 2048
    assert rows[0]["map_binary_hash"] == "map"
    assert "total_route_time_s" not in rows[0]


def test_output_hash_is_little_endian_int64_and_reference_rejects_float() -> None:
    workload = _workload()
    payload = _response("a" * 64, workload.reference_int64, _identities())
    m43._require_response(payload, identities=_identities(), input_sha256="a" * 64, input_hash="b" * 16, reference=workload.reference_int64)
    with pytest.raises(ValueError, match="reference_int64 is not an integer"):
        m43._require_response(payload, identities=_identities(), input_sha256="a" * 64, input_hash="b" * 16, reference=1.5)


def test_production_fixture_binding_contains_plan_task_and_operand_sha256(tmp_path: Path) -> None:
    workload = _workload()
    result = m43.prepare(tmp_path, suite_path=SUITE, build=False, environment={})
    plan = Path(result["plan_dir"])
    manifest = json.loads((plan / "input_manifest.json").read_text())
    bundle = json.loads((plan / "execution_bundle.json").read_text())
    assert manifest["operand_binding"]["binding_hash"] == bundle["operand_binding"]["binding_hash"]
    assert manifest["operand_binding"]["contraction_plan_hash"] == workload.graph.contraction_plan_hash
    assert manifest["operand_binding"]["task"]["id"] == workload.graph.tasks[0].id
    assert manifest["operand_binding"]["input_file_sha256"] == manifest["input_file_sha256"]


def test_prepare_uses_workload_operands_and_writes_identity(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    workload = _workload()
    monkeypatch.setattr(m43, "_load_workload", lambda: workload)
    result = m43.prepare(tmp_path, suite_path=SUITE, build=False, environment={})
    plan = Path(result["plan_dir"])
    assert (plan / "operands.bin").stat().st_size == 512
    manifest = json.loads((plan / "input_manifest.json").read_text())
    assert manifest["input_file_sha256"] == result["input_file_sha256"]
    assert manifest["contraction_plan_hash"] == workload.graph.contraction_plan_hash


def test_report_rejects_incomplete_run(tmp_path: Path) -> None:
    from scripts import upmem_m4_3_report

    with pytest.raises(ValueError, match="must contain"):
        upmem_m4_3_report.inspect(tmp_path)
