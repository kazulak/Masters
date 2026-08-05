"""Inspect an M4.1 differential run without creating performance claims."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path


EXPECTED_ROW_COUNT = 10
EXPECTED_PROVIDER_COUNTS = {"raw_sdk": 5, "simplepim_management": 5}
EXPECTED_REPEAT_IDS = set(range(5))
EXPECTED_PAIR_PROVIDERS = {"raw_sdk": 1, "simplepim_management": 1}
CPU_REFERENCE_ATOL = 1.0e-6
SCIENTIFIC_IDENTITY_FIELDS = (
    "circuit_semantics_hash",
    "tensor_network_hash",
    "contraction_plan_hash",
    "graph_serialized_sha256",
    "input_tensor_hash",
    "numeric_mode",
    "source_task_count",
    "frontier_wave_count",
    "task_assignment_fingerprint",
    "package_file_sha256",
)


def inspect_m4_1_run(run: Path) -> dict[str, object]:
    run = run.resolve()
    summary_path = run / "upmem_hardware_taskgraph_m4_1_summary.json"
    records_path = run / "normalized_records.jsonl"
    if not summary_path.is_file() or not records_path.is_file():
        raise ValueError("M4.1 run must contain summary and normalized_records.jsonl")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    records = [
        json.loads(line)
        for line in records_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    provider_counts = Counter(row.get("provider_id") for row in records)
    pair_rows: dict[tuple[str, int], list[dict[str, object]]] = {}
    pair_keys_valid = True
    for row in records:
        case_id = row.get("case_id")
        repeat_id = row.get("repeat_id")
        if (
            not isinstance(case_id, str)
            or not case_id
            or not isinstance(repeat_id, int)
            or isinstance(repeat_id, bool)
        ):
            pair_keys_valid = False
            continue
        pair_rows.setdefault((case_id, repeat_id), []).append(row)
    case_ids = {case_id for case_id, _repeat_id in pair_rows}
    repeat_ids = {repeat_id for _case_id, repeat_id in pair_rows}
    measured_pair_structure = (
        pair_keys_valid
        and len(case_ids) == 1
        and repeat_ids == EXPECTED_REPEAT_IDS
        and len(pair_rows) == 5
        and all(
            dict(Counter(row.get("provider_id") for row in pair))
            == EXPECTED_PAIR_PROVIDERS
            for pair in pair_rows.values()
        )
    )
    pair_identity = measured_pair_structure and all(
        _scientific_identity_equal(pair) for pair in pair_rows.values()
    )
    cpu_errors_valid = bool(records) and all(_cpu_error_valid(row) for row in records)
    exact_pair_outputs = measured_pair_structure and all(
        _exact_pair_output_valid(pair) for pair in pair_rows.values()
    )
    common_truth = all(
        row.get("status") == "completed"
        and row.get("warmup") is False
        and row.get("raw_vs_simplepim_output_equal") is True
        and row.get("scientific_identity_equal") is True
        and row.get("hardware_execution") is True
        and row.get("target_observed") == "hardware"
        and row.get("hardware_allocation_verified") is True
        and row.get("allocated_dpu_count") == 2
        and row.get("native_kernel_executed") is True
        and row.get("cpu_fallback_used") is False
        and row.get("simulator_kernel_executed") is False
        and row.get("no_cpu_fallback") is True
        and row.get("no_simulator_fallback") is True
        and row.get("native_failure_fallback_used") is False
        and row.get("hardware_no_fallback") is True
        and row.get("release_attempted") is True
        and row.get("release_confirmed") is True
        and row.get("allocation_still_owned") is False
        and row.get("all_tasks_completed") is True
        and row.get("complete_taskgraph_executed") is True
        and row.get("raw_sdk_load_used") is True
        and row.get("raw_sdk_transfer_used") is True
        and row.get("raw_sdk_launch_used") is True
        and row.get("raw_sdk_sync_used") is True
        and row.get("raw_sdk_control_calls_used") is True
        and row.get("thesis_resident_kernel_executed") is True
        and row.get("cpu_reference_validation_status") == "passed"
        for row in records
    )
    raw_truth = all(
        row.get("native_provider_id") == "raw_sdk"
        and row.get("control_provider") == "raw_sdk"
        and row.get("kernel_provider") == "thesis_resident_generic_contract"
        and row.get("allocation_source") == "raw_sdk"
        and row.get("allocation_profile") == "backend=hw"
        and row.get("raw_sdk_direct_allocation_used") is True
        for row in records
        if row.get("provider_id") == "raw_sdk"
    )
    simplepim_truth = all(
        row.get("native_provider_id") == "simplepim_management"
        and row.get("control_provider") == "simplepim_management"
        and row.get("kernel_provider") == "thesis_resident_generic_contract"
        and row.get("simplepim_management_allocation_used") is True
        and row.get("simplepim_management_object_created") is True
        and row.get("allocation_source") == "simplepim_management"
        and row.get("allocation_profile") == "backend=hw"
        and row.get("raw_sdk_direct_allocation_used") is False
        and row.get("simplepim_operator_api_used") is False
        and row.get("simplepim_operator_names") == []
        and row.get("simplepim_kernel_executed") is False
        and row.get("provider_release_attempted") is True
        and row.get("provider_release_succeeded") is True
        and row.get("provider_release_error") == 0
        for row in records
        if row.get("provider_id") == "simplepim_management"
    )
    checks = {
        "summary_completed": summary.get("status") == "completed",
        "exact_measured_row_count": len(records) == EXPECTED_ROW_COUNT
        and summary.get("row_count") == EXPECTED_ROW_COUNT,
        "provider_counts": dict(provider_counts) == EXPECTED_PROVIDER_COUNTS,
        "five_distinct_measured_pairs": measured_pair_structure,
        "pair_scientific_identity": pair_identity,
        "cpu_reference_errors_within_tolerance": cpu_errors_valid,
        "exact_same_binary_pair_outputs": exact_pair_outputs,
        "summary_claim_boundary": summary.get("hardware_speedup_applicable") is False
        and summary.get("timing_is_bringup_only") is True
        and summary.get("allocation_scope") == "per_request"
        and summary.get("persistent_allocation") is False,
        "summary_validation": summary.get("validation_status") == "passed"
        and summary.get("scientific_validation_status") == "passed"
        and summary.get("cross_route_output_equality") == "passed",
        "completed_identity_and_hardware_truth": bool(records) and common_truth,
        "raw_provider_truth": provider_counts.get("raw_sdk") == 5 and raw_truth,
        "simplepim_provider_truth": provider_counts.get("simplepim_management") == 5
        and simplepim_truth,
    }
    valid = all(checks.values())
    return {
        "run_dir": str(run),
        "status": "valid_functionality_evidence" if valid else "invalid_or_incomplete",
        "summary_status": summary.get("status"),
        "row_count": len(records),
        "provider_counts": dict(provider_counts),
        "checks": checks,
        "hardware_speedup_applicable": summary.get("hardware_speedup_applicable"),
        "claim_boundary": summary.get("claim_boundary"),
        "next_m4_items": [
            "persistent allocation",
            "SimplePIM operator or kernel execution",
            "multi-DPU scaling and concurrency measurement",
        ],
    }


def _scientific_identity_equal(pair: list[dict[str, object]]) -> bool:
    for field in SCIENTIFIC_IDENTITY_FIELDS:
        values = [row.get(field) for row in pair]
        if any(value is None or value == "" for value in values):
            return False
        if values[0] != values[1]:
            return False
    return True


def _cpu_error_valid(row: dict[str, object]) -> bool:
    error = row.get("cpu_reference_max_abs_error")
    tolerance = row.get("validation_tolerance_abs", CPU_REFERENCE_ATOL)
    return (
        _finite_number(error)
        and _finite_number(tolerance)
        and float(error) >= 0.0
        and float(tolerance) > 0.0
        and float(error) <= float(tolerance)
    )


def _exact_pair_output_valid(pair: list[dict[str, object]]) -> bool:
    errors = [row.get("raw_vs_simplepim_max_abs_error") for row in pair]
    return all(_finite_number(error) and float(error) == 0.0 for error in errors)


def _finite_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    args = parser.parse_args()
    try:
        payload = inspect_m4_1_run(args.input)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(json.dumps(payload, indent=2))
    return 0 if payload["status"] == "valid_functionality_evidence" else 2


if __name__ == "__main__":
    raise SystemExit(main())
