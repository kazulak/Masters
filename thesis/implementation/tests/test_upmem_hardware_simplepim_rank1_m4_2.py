from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from quantum_bench.bench import upmem_hardware_simplepim_rank1_m4_2 as m42


ROOT = Path(__file__).parents[1]
SUITE = ROOT / "configs/suites/upmem_hardware_simplepim_rank1_m4_2.yml"


def _response(*, parser: bool = False) -> dict[str, object]:
    repetitions = [
        {"repeat_id": 0, "warmup": True, "input_hash": "0123456789abcdef", "output_hash": "fedcba9876543210", "exact_integer_match": True, "scatter_time_s": 0.01, "virtual_zip_time_s": 0.02, "map_time_s": 0.03, "reduction_time_s": 0.04, "total_time_s": 0.1},
        *[
            {"repeat_id": index, "warmup": False, "input_hash": "0123456789abcdef", "output_hash": f"{index:016x}", "reference_int64": 9, "result_int64": 9, "exact_integer_match": True, "scatter_time_s": 0.01, "virtual_zip_time_s": 0.02, "map_time_s": 0.03, "reduction_time_s": 0.04, "total_time_s": 0.1}
            for index in range(5)
        ],
    ]
    return {
        "schema_version": m42.NATIVE_SCHEMA_VERSION,
        "profile_id": m42.PROFILE_ID,
        "backend_id": m42.BACKEND_ID,
        "route_id": m42.ROUTE_ID,
        "provider_id": "simplepim",
        "target_requested": "physical_hardware",
        "target_observed": "not_executed" if parser else "physical_hardware",
        "requested_dpu_count": 2,
        "allocated_dpu_count": None if parser else 2,
        "initialization_tasklets_per_dpu": 1,
        "operator_tasklets_per_dpu": 12,
        "warmup_count": 1,
        "repeat_count": 5,
        "operator_sequence": m42.OPERATOR_SEQUENCE,
        "cpu_fallback_used": False,
        "simulator_kernel_executed": False,
        "hardware_speedup_applicable": False,
        "timing_is_bringup_only": True,
        "hardware_target_observation_method": "default_sdk_allocation_and_observed_dpu_count_after_simulator_selector_rejection",
        "allocation_attempted": not parser,
        "provider_initialized": not parser,
        "simplepim_operator_api_used": not parser,
        "virtual_zip": not parser,
        "map_kernel_executed": not parser,
        "genred_kernel_executed": not parser,
        "host_mediated_reduction": not parser,
        "all_tasks_completed": not parser,
        "exact_integer_match": not parser,
        "hardware_kernel_executed": not parser,
        "release_attempted": not parser,
        "release_confirmed": not parser,
        "hardware_functionality_evidence": not parser,
        "status": "prepared" if parser else "completed",
        "validation_status": "not_run" if parser else "passed",
        "parser_mode": parser,
        "thesis_direct_raw_sdk_allocation_used": False,
        "simplepim_managed_allocation": not parser,
        "persistent_allocation_requested": not parser,
        "persistent_allocation_observed": not parser,
        "operator_validations_passed": not parser,
        "operator_metadata_checks_passed": not parser,
        "logical_payload_h2d_bytes_per_iteration": 2048,
        "logical_payload_d2h_bytes_per_iteration": 16,
        "logical_payload_transfer_bytes_per_iteration": 2064,
        "logical_payload_h2d_bytes_total_session": 12288,
        "logical_payload_d2h_bytes_total_session": 96,
        "logical_payload_transfer_bytes_total_session": 12384,
        "expected_table_count_session": 30,
        "observed_table_count": None if parser else 30,
        "mram_layout_bound_bytes_per_dpu": 18624,
        "mram_high_water_bytes_per_dpu": 18000 if not parser else None,
        "mram_capacity_verified": False,
        "allocation_time_s": 0.1 if not parser else 0.0,
        "handle_compile_time_s": 0.1 if not parser else 0.0,
        "release_time_s": 0.1 if not parser else 0.0,
        "total_route_time_s": 0.9 if not parser else 0.0,
        "host_binary_hash": "1" * 16 if not parser else None,
        "initialization_binary_hash": "2" * 16 if not parser else None,
        "map_binary_hash": "3" * 16 if not parser else None,
        "genred_binary_hash": "4" * 16 if not parser else None,
        "genred_reduce_shared_object_hash": "5" * 16 if not parser else None,
        "timing_scope": "qualification",
        "source_commit": "1d639c53532555f01e9f71d872e7712b166d6cba",
        "staged_source_tree_sha256": "b" * 64,
        "staged_overlay_tree_sha256": "c" * 64,
        "patch_sha256": "d" * 64,
        "stage_manifest_hash": "e" * 16,
        "repetitions": [] if parser else repetitions,
    }


def test_m42_suite_is_fixed_one_rank1_task() -> None:
    suite = m42.load_suite(SUITE)
    assert suite["profile"]["requested_dpu_count"] == 2
    assert suite["profile"]["operator_tasklets_per_dpu"] == 12
    assert suite["workload"]["qualification_fixture"] == {
        "task_count": 1,
        "task_kind": "rank1_contraction",
        "rank": 1,
        "vector_length": 256,
        "operand_transport": "native_fixed_deterministic_operands",
    }


def test_m42_rejects_noncanonical_or_non_rank1_suite(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="committed suite"):
        m42.load_suite(tmp_path / "other.yml")
    payload = yaml_load(SUITE)
    payload["workloads"][0]["qualification_fixture"]["rank"] = 2
    altered = tmp_path / "m42.yml"
    altered.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="committed suite"):
        m42.load_suite(altered)


def test_m42_parser_response_is_allocation_free() -> None:
    payload = _response(parser=True)
    m42._require_response(payload, parser=True)
    assert payload["allocation_attempted"] is False
    assert payload["provider_initialized"] is False


def test_m42_execution_validation_requires_explicit_native_fallback_field() -> None:
    payload = _response()
    m42._require_response(payload, parser=False)

    incomplete = copy.deepcopy(payload)
    incomplete.pop("thesis_direct_raw_sdk_allocation_used")
    with pytest.raises(ValueError, match="native_schema_correction_required"):
        m42._require_response(incomplete, parser=False)


def test_m42_records_preserve_task_identity_and_claim_boundary() -> None:
    payload = _response()
    suite = m42.load_suite(SUITE)
    rows = m42._records(payload, suite, "native_execute_response.json")
    assert len(rows) == 5
    assert {row["repeat_id"] for row in rows} == set(range(5))
    assert all(row["qualification_task_count"] == 1 for row in rows)
    assert all(row["task_graph_integrated"] is False for row in rows)
    assert all(row["contraction_plan_identity"] is None for row in rows)
    assert all("task_graph_task_count" not in row for row in rows)
    assert all("native_schema_correction_required" not in row for row in rows)
    assert all(row["simplepim_operator_api_used"] is True for row in rows)
    assert all(row["communication_provider"] == "host_mediated" for row in rows)
    assert all(row["pid_comm_invoked"] is False for row in rows)
    assert all(row["atim_integrated"] is False for row in rows)
    assert all(row["hardware_speedup_applicable"] is False for row in rows)


def test_m42_execute_does_not_admit_stale_response_after_failed_host(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    native = tmp_path / m42.NATIVE_REL / "build" / "simplepim_rank1_dot_m4_2"
    native.mkdir(parents=True)
    stale = native / "execute_response.json"
    stale.write_text(json.dumps(_response()), encoding="utf-8")

    monkeypatch.setattr(
        m42,
        "_run",
        lambda *args, **kwargs: {"command": ["make", "execute"], "returncode": 2, "timed_out": False, "elapsed_s": 0.01, "stdout": "", "stderr": "native failure"},
    )
    result = m42.execute(tmp_path, suite_path=SUITE, environment={"UPMEM_ALLOW_PHYSICAL_HARDWARE": "1"})
    assert result["status"] == "failed"
    assert not stale.exists()
    assert not (Path(result["run_dir"]) / "normalized_records.jsonl").exists()


def test_m42_prepare_returns_failed_plan_after_parser_failure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    native = tmp_path / m42.NATIVE_REL / "build" / "simplepim_rank1_dot_m4_2"
    native.mkdir(parents=True)
    stale = native / "parser_response.json"
    stale.write_text(json.dumps(_response(parser=True)), encoding="utf-8")

    monkeypatch.setattr(
        m42,
        "_run",
        lambda *args, **kwargs: {"command": ["make", "parser"], "returncode": 2, "timed_out": False, "elapsed_s": 0.01, "stdout": "", "stderr": "parser failure"},
    )
    result = m42.prepare(tmp_path, suite_path=SUITE, build=True, environment={})
    assert result["status"] == "failed"
    assert result["native_build"]["returncode"] == 2
    assert not stale.exists()
    assert not (Path(result["plan_dir"]) / "parser_response.json").exists()


@pytest.mark.parametrize("selector", sorted(m42.SIMULATOR_ENV_KEYS))
def test_m42_rejects_every_simulator_selector(selector: str, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="simulator selector keys are forbidden"):
        m42.execute(tmp_path, suite_path=SUITE, environment={"UPMEM_ALLOW_PHYSICAL_HARDWARE": "1", selector: ""})


def yaml_load(path: Path) -> dict[str, object]:
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8"))
