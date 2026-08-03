#!/usr/bin/env python3
"""KISS report for the physical two-DPU M2.2 numeric-mode study.

This reporter intentionally has a narrower contract than the general research
pack. It only emits derived results after the complete M2.2 admission contract
has passed.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Iterable

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / "build" / "matplotlib")
)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    ROOT
    / "runs"
    / "evidence"
    / "upmem_hardware_sliced_resident_m2_2"
    / "upmem_hw_sliced_resident"
    / "latest"
)
COMPARISON_ROOT = ROOT / "runs" / "comparisons" / "upmem_m2_2"
EXPECTED_MODES = ("none", "per_task_resident_requantize")
MODE_LABELS = {"none": "Float32", "per_task_resident_requantize": "Requantized int8"}
EXPECTED_ROUTE_ID = "upmem_tn_hardware_sliced_resident_two_dpu"
EXPECTED_BACKEND_ID = "upmem_sdk_hardware_sliced_resident_two_dpu"
EXPECTED_FIXTURE_VERSION = "upmem_hardware_sliced_resident_m2_2_v1"
EXPECTED_FIXTURE_SCOPE = "two_operation_h_then_x_full_graph_replicated_prefix"
EXPECTED_EXECUTION_SCOPE = "physical_two_dpu_two_slice_full_replicated_prefix_taskgraph"
EXPECTED_TIMING_SCOPE = "host_observed_sdk_process_wall_and_blocking_sync"
EXPECTED_HARDWARE_PROFILE_VERSION = "hardware_sliced_resident_two_dpu_m2_v1"
REQUIRED_MEASURED_PER_MODE = 5
EXPECTED_MEASURED_REPEAT_IDS = frozenset(range(REQUIRED_MEASURED_PER_MODE))
EXPECTED_WARMUP_REPEAT_IDS = frozenset({0})
REQUIRED_WARMUPS = len(EXPECTED_MODES)
IQR_METHOD = "numpy.percentile(method='linear'); IQR=P75-P25"
VALIDATION_FIELDS = (
    "phase",
    "case_id",
    "workload_id",
    "repeat_id",
    "numeric_mode",
    "status",
    "admission_status",
    "admission_errors",
    "validation_status",
    "route_id",
    "backend_id",
    "backend_family",
    "target_requested",
    "target_observed",
    "requested_dpu_count",
    "allocated_dpu_count",
    "tasklets_per_dpu",
    "hardware_execution",
    "hardware_functionality_evidence",
    "hardware_kernel_executed",
    "native_kernel_executed",
    "simulator_kernel_executed",
    "cpu_fallback_used",
    "device_completion_confirmed",
    "native_cleanup_confirmed",
    "release_confirmed",
    "response_numeric_mode",
    "fixture_version",
    "fixture_scope",
    "execution_scope",
    "operation_count",
    "slice_count",
    "source_slice_count",
    "slice_ids",
    "expanded_task_count",
    "executed_task_count",
    "completed_task_count",
    "executed_slice_count",
    "completed_slice_count",
    "source_task_count",
    "source_task_completion_count",
    "source_task_completion_scope",
    "slice_model_task_count",
    "slice_model_executed_task_count",
    "observed_operation_completion_counts",
    "duplicate_contraction_check",
    "missing_dependency_check",
    "dependency_violation_detected",
    "execution_contract_status",
    "policy_reference_status",
    "full_precision_accuracy_status",
    "scientific_validation_status",
    "reconstruction_validation_status",
    "per_slice_output_validation_status",
    "strict_cpu_reference_validation",
    "transfer_accounting_invariant",
    "transfer_matches_manifest_plan",
    "slice_package_validation_status",
    "actual_h2d_bytes",
    "actual_d2h_bytes",
    "actual_transfer_bytes",
    "planned_h2d_bytes",
    "planned_d2h_bytes",
    "planned_transfer_bytes",
    "observed_h2d_bytes",
    "observed_d2h_bytes",
    "observed_transfer_bytes",
    "application_visible_h2d_bytes",
    "application_visible_d2h_bytes",
    "application_visible_transfer_bytes",
    "application_visible_total_bytes",
    "circuit_semantics_hash",
    "tensor_network_hash",
    "contraction_plan_hash",
    "executor_config_hash",
    "host_binary_hash",
    "dpu_binary_hash",
    "binary_source_tree_hash",
    "output_hash",
)


class ReportError(ValueError):
    """Raised when the source run is not admissible for M2.2 reporting."""


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ReportError(f"expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ReportError(f"expected JSON object at {path}:{line_number}")
            rows.append(value)
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_true(value: Any) -> bool:
    return value is True


def _finite_number(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _positive_finite_number(value: Any) -> bool:
    return _finite_number(value) and float(value) > 0.0


def _nonnegative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _optional_byte_group_errors(
    row: dict[str, Any], prefix: str, fields: tuple[str, str, str]
) -> list[str]:
    present = [field in row and row[field] is not None for field in fields]
    if not any(present):
        return []
    errors: list[str] = []
    for field, is_present in zip(fields, present):
        if is_present and not _nonnegative_int(row[field]):
            errors.append(f"nonnegative_integer_{field}")
    if not all(present):
        errors.append(f"complete_{prefix}_byte_group")
    if not all(present) or any(
        not _nonnegative_int(row[field])
        for field in fields
        if field in row and row[field] is not None
    ):
        return errors
    first, second, total = fields
    if row[total] != row[first] + row[second]:
        errors.append(f"{prefix}_parts_sum")
    return errors


def _byte_errors(row: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    actual = ("actual_h2d_bytes", "actual_d2h_bytes", "actual_transfer_bytes")
    if not all(_nonnegative_int(row.get(field)) for field in actual):
        errors.append("actual_bytes_nonnegative_integers")
    elif row[actual[2]] != row[actual[0]] + row[actual[1]]:
        errors.append("actual_transfer_parts_sum")
    for prefix, fields in (
        (
            "planned",
            ("planned_h2d_bytes", "planned_d2h_bytes", "planned_transfer_bytes"),
        ),
        (
            "observed",
            ("observed_h2d_bytes", "observed_d2h_bytes", "observed_transfer_bytes"),
        ),
        (
            "application_visible",
            (
                "application_visible_h2d_bytes",
                "application_visible_d2h_bytes",
                "application_visible_transfer_bytes",
            ),
        ),
    ):
        errors.extend(_optional_byte_group_errors(row, prefix, fields))
        if all(field in row and row[field] is not None for field in fields) and all(
            _nonnegative_int(row[field]) for field in fields
        ):
            for actual_field, comparison_field in zip(actual, fields):
                if row[actual_field] != row[comparison_field]:
                    errors.append(
                        f"actual_{actual_field.removeprefix('actual_')}_differs_from_{prefix}"
                    )
    if (
        "application_visible_total_bytes" in row
        and row["application_visible_total_bytes"] is not None
    ):
        if not _nonnegative_int(row["application_visible_total_bytes"]):
            errors.append("application_visible_total_bytes_nonnegative_integer")
        elif row["application_visible_total_bytes"] != row.get("actual_transfer_bytes"):
            errors.append("actual_transfer_differs_from_application_visible_total")
        if (
            "application_visible_transfer_bytes" in row
            and row["application_visible_total_bytes"]
            != row["application_visible_transfer_bytes"]
        ):
            errors.append("application_visible_total_differs_from_transfer")
    return errors


def _field(row: dict[str, Any], path: str, default: Any = None) -> Any:
    value: Any = row
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            return default
        value = value[part]
    return value


def _required_exact_checks(row: dict[str, Any], expected: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for field, expected_value in expected.items():
        if field not in row:
            errors.append(f"missing_{field}")
        elif row[field] != expected_value:
            errors.append(f"{field}_expected_{expected_value!r}")
    return errors


def _task_shape_errors(row: dict[str, Any], expected_phase: str) -> list[str]:
    errors: list[str] = []
    expected = {
        "operation_count": 2,
        "slice_count": 2,
        "source_slice_count": 2,
        "slice_ids": [0, 1],
        "expanded_task_count": 4,
        "executed_task_count": 4,
        "completed_task_count": 4,
        "executed_slice_count": 2,
        "completed_slice_count": 2,
        "source_task_count": 2,
        "source_task_completion_count": 4,
        "source_task_completion_scope": "replicated_slice_operations",
        "slice_model_task_count": 2,
        "slice_model_executed_task_count": 4,
        "observed_operation_completion_counts": [2, 2],
    }
    for field, expected_value in expected.items():
        if field not in row:
            errors.append(f"missing_{field}")
        elif row[field] != expected_value:
            errors.append(f"{field}_expected_{expected_value!r}")
    if expected_phase == "measured":
        for field in ("duplicate_contraction_check", "missing_dependency_check"):
            if field not in row:
                errors.append(f"missing_{field}")
            elif row[field] not in (None, "passed", True):
                errors.append(f"{field}_invalid")
        if "dependency_violation_detected" not in row:
            errors.append("missing_dependency_violation_detected")
        elif row["dependency_violation_detected"] is not False:
            errors.append("dependency_violation_detected")
    return errors


def _timing_errors(row: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    checks = {
        "total_time_s": row.get("total_time_s"),
        "process_time_s": row.get("process_time_s"),
        "reconstruction_time_s": row.get("reconstruction_time_s"),
        "stage_timings.total_route_time_s": _field(
            row, "stage_timings.total_route_time_s"
        ),
        "stage_timings.sync_wait_time_s": _field(row, "stage_timings.sync_wait_time_s"),
    }
    for path, value in checks.items():
        if not _positive_finite_number(value):
            errors.append(f"positive_finite_{path}")
    timing_checks = (
        ("timing_is_bringup_only", row.get("timing_is_bringup_only") is True),
        ("timing_scope", row.get("timing_scope") == EXPECTED_TIMING_SCOPE),
        (
            "timing_breakdown_status",
            row.get("timing_breakdown_status") == "host_stage_boundaries",
        ),
        ("stage_timings_object", isinstance(row.get("stage_timings"), dict)),
        (
            "stage_timings_status",
            _field(row, "stage_timings.status") == "host_stage_boundaries",
        ),
        ("kernel_time_unavailable", _field(row, "stage_timings.kernel_time_s") is None),
        (
            "sync_wait_not_pure_kernel_time",
            _field(row, "stage_timings.sync_wait_is_not_pure_kernel_time") is True,
        ),
    )
    errors.extend(name for name, passed in timing_checks if not passed)
    return errors


def _stable_row_metadata(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Return stable provenance values without inventing missing metadata."""

    fields = (
        "host_binary_hash",
        "dpu_binary_hash",
        "native_source_tree_hash",
        "binary_source_tree_hash",
        "qasm_source_sha256",
    )
    result: dict[str, Any] = {}
    for field in fields:
        values = {row.get(field) for row in rows if row.get(field) is not None}
        if len(values) == 1:
            result[field] = next(iter(values))
        elif len(values) > 1:
            result[field] = None
    return result


def _stable_execution_metadata(rows: list[dict[str, Any]]) -> dict[str, Any]:
    fields = (
        "target_requested",
        "target_observed",
        "requested_dpu_count",
        "allocated_dpu_count",
        "tasklets_per_dpu",
        "device_launch_mode",
        "host_completion_mode",
    )
    result: dict[str, Any] = {}
    for field in fields:
        values = {row.get(field) for row in rows if row.get(field) is not None}
        if len(values) == 1:
            result[field] = next(iter(values))
    return result


def _manifest_contract_errors(manifest: dict[str, Any]) -> list[str]:
    required = {
        "fixture_version": EXPECTED_FIXTURE_VERSION,
        "fixture_scope": EXPECTED_FIXTURE_SCOPE,
        "execution_scope": EXPECTED_EXECUTION_SCOPE,
        "operation_count": 2,
        "source_task_count": 2,
    }
    errors: list[str] = []
    for field, expected_value in required.items():
        if field not in manifest:
            errors.append(f"run_manifest_missing_{field}")
        elif manifest[field] != expected_value:
            errors.append(f"run_manifest_{field}_expected_{expected_value!r}")
    if "slice_count" in manifest and manifest["slice_count"] != 2:
        errors.append("run_manifest_slice_count_expected_2")
    return errors


def _provenance_errors(
    manifest: dict[str, Any],
    environment: dict[str, Any],
    summary: dict[str, Any],
    hardware_profile: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    source_commit = manifest.get("benchmark_source_commit") or manifest.get(
        "git_commit"
    )
    if not isinstance(source_commit, str) or not source_commit:
        errors.append("missing_source_commit")
    if not isinstance(manifest.get("benchmark_source_worktree_dirty"), bool):
        errors.append("missing_benchmark_source_worktree_dirty")
    environment_commit = environment.get("git_commit")
    if environment_commit is not None and environment_commit != source_commit:
        errors.append("environment_source_commit_mismatch")
    native_build = summary.get("native_build")
    sdk_tools = (
        native_build.get("sdk_tools") if isinstance(native_build, dict) else None
    )
    if not isinstance(sdk_tools, dict) or not sdk_tools:
        errors.append("missing_sdk_metadata")
    expected_profile = {
        "hardware_profile_version": EXPECTED_HARDWARE_PROFILE_VERSION,
        "target": "hardware",
        "backend_id": EXPECTED_BACKEND_ID,
        "route_id": EXPECTED_ROUTE_ID,
        "requested_dpu_count": 2,
        "slices": 2,
        "tasklets_per_dpu": 1,
        "numeric_modes": list(EXPECTED_MODES),
        "performance_claim_applicable": False,
    }
    for field, expected_value in expected_profile.items():
        if field not in hardware_profile:
            errors.append(f"hardware_profile_missing_{field}")
        elif hardware_profile[field] != expected_value:
            errors.append(f"hardware_profile_{field}_expected_{expected_value!r}")
    return errors


def _admission_errors(row: dict[str, Any], expected_phase: str) -> list[str]:
    errors: list[str] = []
    checks: tuple[tuple[str, bool], ...] = (
        ("status_completed", row.get("status") == "completed"),
        (
            "repeat_id_integer",
            isinstance(row.get("repeat_id"), int)
            and not isinstance(row.get("repeat_id"), bool),
        ),
        ("phase_matches_source", row.get("phase") == expected_phase),
        (
            "case_id_nonempty",
            isinstance(row.get("case_id"), str) and bool(row.get("case_id")),
        ),
        (
            "workload_id_nonempty",
            isinstance(row.get("workload_id"), str) and bool(row.get("workload_id")),
        ),
        ("route_id_exact", row.get("route_id") == EXPECTED_ROUTE_ID),
        ("backend_id_exact", row.get("backend_id") == EXPECTED_BACKEND_ID),
        ("backend_family_upmem_sdk", row.get("backend_family") == "upmem_sdk"),
        ("target_requested_hardware", row.get("target_requested") == "hardware"),
        ("target_observed_hardware", row.get("target_observed") == "hardware"),
        ("requested_dpu_count_2", row.get("requested_dpu_count") == 2),
        ("allocated_dpu_count_2", row.get("allocated_dpu_count") == 2),
        ("tasklets_per_dpu_1", row.get("tasklets_per_dpu") == 1),
        ("hardware_execution", _is_true(row.get("hardware_execution"))),
        ("hardware_kernel_executed", _is_true(row.get("hardware_kernel_executed"))),
        ("native_kernel_executed", _is_true(row.get("native_kernel_executed"))),
        (
            "hardware_functionality_evidence",
            _is_true(row.get("hardware_functionality_evidence")),
        ),
        (
            "device_completion_confirmed",
            _is_true(row.get("device_completion_confirmed")),
        ),
        ("native_cleanup_confirmed", _is_true(row.get("native_cleanup_confirmed"))),
        ("release_confirmed", _is_true(_field(row, "release_evidence.confirmed"))),
        (
            "simulator_kernel_not_executed",
            row.get("simulator_kernel_executed") is False,
        ),
        ("cpu_fallback_not_used", row.get("cpu_fallback_used") is False),
        ("execution_contract_passed", row.get("execution_contract_status") == "passed"),
        ("policy_reference_passed", row.get("policy_reference_status") == "passed"),
        (
            "full_precision_passed",
            row.get("full_precision_accuracy_status") == "passed",
        ),
        (
            "scientific_validation_passed",
            row.get("scientific_validation_status") == "passed",
        ),
        ("top_level_validation_passed", row.get("validation_status") == "passed"),
        (
            "reconstruction_passed",
            row.get("reconstruction_validation_status") == "passed",
        ),
        (
            "slice_output_validation_passed",
            row.get("per_slice_output_validation_status") == "passed",
        ),
        (
            "expected_output_validation_passed",
            _is_true(row.get("expected_output_validation")),
        ),
        (
            "strict_cpu_reference_passed",
            _is_true(row.get("strict_cpu_reference_validation")),
        ),
        (
            "transfer_invariant_passed",
            _is_true(row.get("transfer_accounting_invariant")),
        ),
        ("transfer_matches_plan", _is_true(row.get("transfer_matches_manifest_plan"))),
        ("source_hashes_preserved", _is_true(row.get("source_hashes_preserved"))),
        (
            "slice_package_validation_passed",
            row.get("slice_package_validation_status") == "passed",
        ),
        (
            "hardware_speedup_not_applicable",
            row.get("hardware_speedup_applicable") is False,
        ),
        ("energy_not_available", row.get("energy_measurement_available") is False),
        (
            "response_mode_matches",
            row.get("response_numeric_mode") == row.get("numeric_mode"),
        ),
    )
    errors.extend(name for name, passed in checks if not passed)
    errors.extend(
        _required_exact_checks(
            row,
            {
                "fixture_version": EXPECTED_FIXTURE_VERSION,
                "fixture_scope": EXPECTED_FIXTURE_SCOPE,
                "execution_scope": EXPECTED_EXECUTION_SCOPE,
            },
        )
    )
    errors.extend(_task_shape_errors(row, expected_phase))
    errors.extend(_timing_errors(row))
    for path in (
        "policy_reference_max_abs_error",
        "policy_reference_l2_error",
        "full_precision_max_abs_error",
        "full_precision_l2_error",
    ):
        if not _finite_number(row.get(path)) or float(row[path]) < 0:
            errors.append(f"finite_nonnegative_{path}")
    errors.extend(_byte_errors(row))
    if row.get("numeric_mode") not in EXPECTED_MODES:
        errors.append("recognized_numeric_mode")
    return errors


def _validation_row(row: dict[str, Any], phase: str) -> dict[str, Any]:
    errors = _admission_errors(row, phase)
    result = {field: row.get(field) for field in VALIDATION_FIELDS}
    result["phase"] = phase
    result["release_confirmed"] = _field(row, "release_evidence.confirmed")
    result["admission_status"] = "passed" if not errors else "rejected"
    result["admission_errors"] = ";".join(errors)
    return result


def validate_source(
    warmups: list[dict[str, Any]],
    measured: list[dict[str, Any]],
    summary: dict[str, Any] | None = None,
    manifest: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Validate the complete M2.2 admission contract and return row audit data."""

    validation_rows = [_validation_row(row, "warmup") for row in warmups]
    validation_rows.extend(_validation_row(row, "measured") for row in measured)
    errors: list[str] = []
    if len(warmups) != REQUIRED_WARMUPS:
        errors.append(f"warmup_count={len(warmups)} expected {REQUIRED_WARMUPS}")
    if len(measured) != len(EXPECTED_MODES) * REQUIRED_MEASURED_PER_MODE:
        errors.append(
            f"measured_count={len(measured)} expected {len(EXPECTED_MODES) * REQUIRED_MEASURED_PER_MODE}"
        )
    if any(item["admission_status"] != "passed" for item in validation_rows):
        errors.append("one_or_more_rows_failed_admission")
        for item in validation_rows:
            if item["admission_status"] == "rejected" and item["admission_errors"]:
                errors.extend(
                    error
                    for error in item["admission_errors"].split(";")
                    if error not in errors
                )

    all_rows = warmups + measured
    if not all_rows:
        errors.append("no_source_rows")
    modes = {row.get("numeric_mode") for row in all_rows}
    if modes != set(EXPECTED_MODES):
        errors.append(
            f"numeric_modes={sorted(modes, key=str)} expected {list(EXPECTED_MODES)}"
        )
    for mode in EXPECTED_MODES:
        warmup_mode = [row for row in warmups if row.get("numeric_mode") == mode]
        measured_mode = [row for row in measured if row.get("numeric_mode") == mode]
        if len(warmup_mode) != 1:
            errors.append(f"warmups_{mode}={len(warmup_mode)} expected 1")
        if len(measured_mode) != REQUIRED_MEASURED_PER_MODE:
            errors.append(
                f"measured_{mode}={len(measured_mode)} expected {REQUIRED_MEASURED_PER_MODE}"
            )
        warmup_ids = [row.get("repeat_id") for row in warmup_mode]
        measured_ids = [row.get("repeat_id") for row in measured_mode]
        if len(warmup_ids) != len(set(warmup_ids)):
            errors.append(f"duplicate_warmup_repeat_ids_{mode}")
        if len(measured_ids) != len(set(measured_ids)):
            errors.append(f"duplicate_measured_repeat_ids_{mode}")
        if set(warmup_ids) != set(EXPECTED_WARMUP_REPEAT_IDS):
            errors.append(
                f"warmup_repeat_ids_{mode}={sorted(warmup_ids, key=str)} expected [0]"
            )
        if set(measured_ids) != set(EXPECTED_MEASURED_REPEAT_IDS):
            errors.append(
                f"measured_repeat_ids_{mode}={sorted(measured_ids, key=str)} expected [0, 1, 2, 3, 4]"
            )

    measured_pairs = {
        mode: [row for row in measured if row.get("numeric_mode") == mode]
        for mode in EXPECTED_MODES
    }
    if all(
        len(measured_pairs[mode]) == REQUIRED_MEASURED_PER_MODE
        for mode in EXPECTED_MODES
    ):
        measured_id_sets = {
            mode: {row.get("repeat_id") for row in measured_pairs[mode]}
            for mode in EXPECTED_MODES
        }
        if any(
            measured_id_sets[mode] != EXPECTED_MEASURED_REPEAT_IDS
            for mode in EXPECTED_MODES
        ):
            errors.append("paired_measured_repeat_ids_not_exact")
        elif measured_id_sets[EXPECTED_MODES[0]] != measured_id_sets[EXPECTED_MODES[1]]:
            errors.append("paired_measured_repeat_ids_mismatch")

    identity_fields = (
        "circuit_semantics_hash",
        "tensor_network_hash",
        "contraction_plan_hash",
    )
    for field in identity_fields:
        values = {row.get(field) for row in all_rows}
        if len(values) != 1 or None in values:
            errors.append(f"single_{field}")
    for field in ("case_id", "workload_id"):
        values = {row.get(field) for row in all_rows}
        if len(values) != 1 or None in values:
            errors.append(f"single_{field}")
    executor_by_mode = {
        mode: {
            row.get("executor_config_hash")
            for row in all_rows
            if row.get("numeric_mode") == mode
        }
        for mode in EXPECTED_MODES
    }
    if any(len(values) != 1 or None in values for values in executor_by_mode.values()):
        errors.append("one_executor_hash_per_mode")
    if (
        executor_by_mode[EXPECTED_MODES[0]]
        and executor_by_mode[EXPECTED_MODES[0]] == executor_by_mode[EXPECTED_MODES[1]]
    ):
        errors.append("executor_hashes_must_differ_by_mode")

    if summary is not None:
        if summary.get("measured_passed_count") != len(measured):
            errors.append("summary_measured_passed_count")
        if summary.get("warmup_passed_count") != len(warmups):
            errors.append("summary_warmup_passed_count")
        if summary.get("status") != "completed":
            errors.append("summary_status")
    if manifest is not None:
        errors.extend(_manifest_contract_errors(manifest))

    for field in (
        "host_binary_hash",
        "dpu_binary_hash",
        "native_source_tree_hash",
        "binary_source_tree_hash",
    ):
        values = {row.get(field) for row in all_rows if row.get(field) is not None}
        if len(values) > 1:
            errors.append(f"stable_{field}")

    optional_contract_fields = (
        "fixture_version",
        "fixture_scope",
        "execution_scope",
    )
    for field in optional_contract_fields:
        values = {row.get(field) for row in all_rows if field in row}
        if values and (len(values) != 1 or None in values):
            errors.append(f"single_{field}")

    context = {
        "errors": errors,
        "identity": {
            field: next(iter({row.get(field) for row in all_rows}))
            for field in (*identity_fields, "case_id", "workload_id")
            if all_rows
        },
        "executor_hashes": {
            mode: next(iter(values)) if len(values) == 1 else None
            for mode, values in executor_by_mode.items()
        },
        "measured_count": len(measured),
        "warmup_count": len(warmups),
        "iqr_method": IQR_METHOD,
        "row_provenance": _stable_row_metadata(all_rows),
        "execution_metadata": _stable_execution_metadata(all_rows),
        "fixture": {
            field: next((row.get(field) for row in all_rows if field in row), None)
            for field in optional_contract_fields
        },
    }
    return validation_rows, context


def _values(rows: Iterable[dict[str, Any]], path: str) -> np.ndarray:
    values: list[float] = []
    for row in rows:
        value = _field(row, path)
        try:
            converted = float(value)
        except (TypeError, ValueError) as exc:
            raise ReportError(f"invalid metric: {path}") from exc
        if not np.isfinite(converted):
            raise ReportError(f"non-finite metric: {path}")
        values.append(converted)
    if not values:
        raise ReportError(f"missing metric: {path}")
    return np.asarray(values, dtype=float)


def _stats(values: np.ndarray) -> tuple[float, float, float, float, float]:
    q1, median, q3 = np.percentile(values, [25, 50, 75], method="linear")
    return (
        float(median),
        float(q3 - q1),
        float(values.min()),
        float(values.max()),
        float(q1),
    )


def _mode_statistics(rows: list[dict[str, Any]], mode: str) -> dict[str, Any]:
    selected = [row for row in rows if row.get("numeric_mode") == mode]
    total = _values(selected, "total_time_s")
    process = _values(selected, "process_time_s")
    native = _values(selected, "stage_timings.total_route_time_s")
    reconstruction = _values(selected, "reconstruction_time_s")
    sync_wait = _values(selected, "stage_timings.sync_wait_time_s")
    fields: dict[str, Any] = {
        "numeric_mode": mode,
        "numeric_mode_label": MODE_LABELS.get(mode, mode),
        "case_id": selected[0].get("case_id"),
        "workload_id": selected[0].get("workload_id"),
        "record_count": len(selected),
        "warmup_count": 0,
        "measured_count": len(selected),
        "median_total_time_s": None,
        "iqr_total_time_s": None,
        "min_total_time_s": None,
        "max_total_time_s": None,
        "median_process_time_s": None,
        "iqr_process_time_s": None,
        "min_process_time_s": None,
        "max_process_time_s": None,
        "median_native_route_time_s": None,
        "iqr_native_route_time_s": None,
        "min_native_route_time_s": None,
        "max_native_route_time_s": None,
        "median_reconstruction_time_s": float(np.median(reconstruction)),
        "median_sync_wait_time_s": float(np.median(sync_wait)),
        "median_h2d_bytes": float(np.median(_values(selected, "actual_h2d_bytes"))),
        "median_d2h_bytes": float(np.median(_values(selected, "actual_d2h_bytes"))),
        "median_transfer_bytes": float(
            np.median(_values(selected, "actual_transfer_bytes"))
        ),
        "policy_reference_max_abs_error": float(
            np.max(_values(selected, "policy_reference_max_abs_error"))
        ),
        "policy_reference_l2_error": float(
            np.max(_values(selected, "policy_reference_l2_error"))
        ),
        "full_precision_max_abs_error": float(
            np.max(_values(selected, "full_precision_max_abs_error"))
        ),
        "full_precision_l2_error": float(
            np.max(_values(selected, "full_precision_l2_error"))
        ),
        "validation_passed_count": len(selected),
        "validation_failed_count": 0,
    }
    for prefix, values in (
        ("total", total),
        ("process", process),
        ("native_route", native),
    ):
        median, iqr, minimum, maximum, _ = _stats(values)
        fields[f"median_{prefix}_time_s"] = median
        fields[f"iqr_{prefix}_time_s"] = iqr
        fields[f"min_{prefix}_time_s"] = minimum
        fields[f"max_{prefix}_time_s"] = maximum
    return fields


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _plot_runtime(path: Path, stats: list[dict[str, Any]]) -> None:
    labels = [
        MODE_LABELS.get(str(item["numeric_mode"]), str(item["numeric_mode"]))
        for item in stats
    ]
    medians = [float(item["median_total_time_s"]) * 1000 for item in stats]
    iqr = [float(item["iqr_total_time_s"]) * 1000 for item in stats]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(
        labels,
        medians,
        yerr=np.asarray(iqr) / 2,
        capsize=5,
        color=["#3b82f6", "#f97316"],
    )
    ax.set_title(
        "Physical UPMEM two-DPU host-observed bring-up timing\n(no speedup claim)"
    )
    ax.set_ylabel("Total route time (ms), median +/- half IQR")
    ax.set_xlabel("Numeric mode")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_accuracy(path: Path, stats: list[dict[str, Any]]) -> None:
    labels = [
        MODE_LABELS.get(str(item["numeric_mode"]), str(item["numeric_mode"]))
        for item in stats
    ]
    policy = [float(item["policy_reference_max_abs_error"]) for item in stats]
    full = [float(item["full_precision_max_abs_error"]) for item in stats]
    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8, 5))
    floor = 1e-12
    plot_policy = [value if value > 0 else floor for value in policy]
    plot_full = [value if value > 0 else floor for value in full]
    ax.bar(x - width / 2, plot_policy, width, label="Policy-reference max abs error")
    ax.bar(x + width / 2, plot_full, width, label="Full-precision max abs error")
    maximum = max(plot_policy + plot_full + [floor])
    ax.set_yscale("log")
    ax.set_ylim(floor / 2, maximum * 8)
    for positions, values in ((x - width / 2, policy), (x + width / 2, full)):
        for position, value in zip(positions, values):
            label = "0 (exact)" if value == 0 else f"{value:.3g}"
            ax.text(
                position,
                value if value > 0 else floor * 2,
                label,
                ha="center",
                va="bottom",
                fontsize=8,
            )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, ha="center")
    ax.set_title("Physical UPMEM M2.2 validation error by numeric mode")
    ax.set_ylabel("Maximum absolute error (log scale; zero plotted at floor)")
    ax.set_xlabel("Numeric mode")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout(pad=1.2)
    fig.subplots_adjust(bottom=0.22)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _markdown(
    input_dir: Path,
    source_hashes: dict[str, str],
    source_commit: str | None,
    stats: list[dict[str, Any]],
    paired: list[dict[str, Any]],
    context: dict[str, Any],
) -> str:
    by_mode = {item["numeric_mode"]: item for item in stats}
    none = by_mode[EXPECTED_MODES[0]]
    req = by_mode[EXPECTED_MODES[1]]
    ratio_of_medians = req["median_total_time_s"] / none["median_total_time_s"]
    native_ratio_of_medians = (
        req["median_native_route_time_s"] / none["median_native_route_time_s"]
    )
    paired_total = [item["requantized_over_none_total_time_ratio"] for item in paired]
    paired_native = [
        item["requantized_over_none_native_route_ratio"] for item in paired
    ]
    return f"""# M2.2 Physical Numeric-Mode Report

Source run: `{input_dir}`  
Source commit: `{source_commit or "unknown"}`

## Scope

This report admits exactly **2 passed warmups** and **10 passed measured rows**: five measured repeats for each numeric mode. The route allocated **2 physical UPMEM DPUs**, with **1 tasklet per DPU**, for one deterministic H-X case and one contraction plan.

Timing statistics use NumPy's linear percentile definition: **IQR = P75 - P25**, with `numpy.percentile(..., method="linear")`. The plots and tables report host-observed bring-up timing, not kernel-only timing.

| Numeric mode | Measured rows | Median total route (ms) | IQR (ms) | Median native route (ms) | Median reconstruction (ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Float32 | {none["measured_count"]} | {none["median_total_time_s"] * 1000:.3f} | {none["iqr_total_time_s"] * 1000:.3f} | {none["median_native_route_time_s"] * 1000:.3f} | {none["median_reconstruction_time_s"] * 1000:.3f} |
| Requantized int8 | {req["measured_count"]} | {req["median_total_time_s"] * 1000:.3f} | {req["iqr_total_time_s"] * 1000:.3f} | {req["median_native_route_time_s"] * 1000:.3f} | {req["median_reconstruction_time_s"] * 1000:.3f} |

## Ratio definitions

For matched repeat `i`, `requantized_over_none_total_time_ratio = T_requantized_i / T_none_i`; values above 1 mean the requantized route took longer in that matched bring-up run. The corresponding native ratio uses `stage_timings.total_route_time_s`. These are **timing ratios**, not speedups.

- `ratio_of_medians` (total route): **{ratio_of_medians:.6f}x**.
- `median_of_paired_ratios` (total route): **{float(np.median(paired_total)):.6f}x**.
- `ratio_of_medians` (native route): **{native_ratio_of_medians:.6f}x**.
- `median_of_paired_ratios` (native route): **{float(np.median(paired_native)):.6f}x**.

## Validation and identity

All 12 rows passed execution-contract, policy-reference, full-precision, scientific, reconstruction, per-slice, transfer, and strict CPU-reference validation. Application-visible transfer was stable at **{none["median_h2d_bytes"]:.0f} H2D + {none["median_d2h_bytes"]:.0f} D2H = {none["median_transfer_bytes"]:.0f} bytes** per row. These are SDK-visible application bytes, not physical bus counters.

Circuit, tensor-network, and contraction-plan hashes were constant across both modes. Executor configuration hashes were constant within each mode and distinct between modes.

## Caveats

- This is one H-X case and one plan, not a general TN benchmark.
- `slice_parallel_execution=false`; asynchronous set launch was recorded, but overlap was not measured.
- `kernel_time_s` is unavailable; timings include host allocation, binary load, transfers, synchronization, release, and reconstruction according to their stated scopes.
- The experiment has one tasklet per DPU and does not establish multi-DPU scaling or concurrency speedup.
- No energy measurement or physical energy model is present.

## Claims

Allowed: the bounded two-DPU physical route executed and validated the same sliced resident TaskGraph in both numeric modes, with reproducible hashes and transfer accounting. The mode ratios above may be reported as host-observed bring-up timing ratios.

Not allowed: no speedup claim, no quantization speedup, no scaling claim, no energy-efficiency claim, and no general tensor-network quantum-circuit performance claim.

## Artifacts

- `mode_statistics.csv`
- `paired_mode_ratios.csv`
- `validation_rows.csv`
- `runtime_by_mode.png`
- `accuracy_by_mode.png`
- `report_manifest.json`
"""


def _output_directory(output_dir: Path | None, comparison_root: Path | None) -> Path:
    root = (comparison_root or COMPARISON_ROOT).resolve()
    output = (
        output_dir.resolve()
        if output_dir is not None
        else root / datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S_%f")
    )
    try:
        relative = output.relative_to(root)
    except ValueError as exc:
        raise ReportError(
            f"report output must be under comparison root: {root}"
        ) from exc
    if not relative.parts:
        raise ReportError(f"report output must be a child of comparison root: {root}")
    output.mkdir(parents=True, exist_ok=False)
    return output


def _artifact_manifest(output: Path) -> dict[str, dict[str, Any]]:
    artifacts: dict[str, dict[str, Any]] = {}
    for path in sorted(output.iterdir()):
        if path.name == "report_manifest.json" or not path.is_file():
            continue
        entry: dict[str, Any] = {"sha256": _sha256(path), "bytes": path.stat().st_size}
        if path.name in {"runtime_by_mode.png", "accuracy_by_mode.png"}:
            entry.update({"kind": "plot", "source_csv": "mode_statistics.csv"})
        elif path.suffix == ".csv":
            entry["kind"] = "source_table"
        elif path.name == "benchmark_summary.md":
            entry["kind"] = "summary"
        artifacts[path.name] = entry
    return artifacts


def generate_report(
    input_dir: Path,
    output_dir: Path | None = None,
    *,
    comparison_root: Path | None = None,
) -> Path:
    input_dir = input_dir.resolve()
    required = {
        "normalized_records.jsonl": input_dir / "normalized_records.jsonl",
        "warmups.jsonl": input_dir / "warmups.jsonl",
        "run_manifest.json": input_dir / "run_manifest.json",
        "environment.json": input_dir / "environment.json",
        "config/resolved_suite.yml": input_dir / "config" / "resolved_suite.yml",
        "config/hardware_profile.json": input_dir / "config" / "hardware_profile.json",
        "upmem_hardware_sliced_resident_mvp_summary.json": input_dir
        / "upmem_hardware_sliced_resident_mvp_summary.json",
    }
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        raise ReportError(f"missing M2.2 source artifacts: {', '.join(missing)}")
    measured = _read_jsonl(required["normalized_records.jsonl"])
    warmups = _read_jsonl(required["warmups.jsonl"])
    summary = _read_json(required["upmem_hardware_sliced_resident_mvp_summary.json"])
    manifest = _read_json(required["run_manifest.json"])
    environment = _read_json(required["environment.json"])
    hardware_profile = _read_json(required["config/hardware_profile.json"])
    validation_rows, context = validate_source(warmups, measured, summary, manifest)
    context["errors"].extend(
        _provenance_errors(manifest, environment, summary, hardware_profile)
    )
    output = _output_directory(output_dir, comparison_root)
    validation_fields = list(VALIDATION_FIELDS)
    _write_csv(output / "validation_rows.csv", validation_rows, validation_fields)
    source_paths = {name: path for name, path in required.items()}
    source_hashes = {name: _sha256(path) for name, path in source_paths.items()}
    build_metadata = summary.get("native_build")
    context["environment"] = environment
    context["sdk_metadata"] = (
        build_metadata.get("sdk_tools")
        if isinstance(build_metadata, dict)
        else environment.get("upmem")
    )
    if context["errors"]:
        (output / "benchmark_summary.md").write_text(
            "# M2.2 Report Rejected\n\n"
            + "\n".join(f"- {item}" for item in context["errors"])
            + "\n",
            encoding="utf-8",
        )
        (output / "report_manifest.json").write_text(
            json.dumps(
                {
                    "status": "rejected",
                    "source": str(input_dir),
                    "source_hashes": source_hashes,
                    "validation_errors": context["errors"],
                    "outputs": ["validation_rows.csv", "benchmark_summary.md"],
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        raise ReportError(
            f"M2.2 source rejected; validation artifact: {output / 'validation_rows.csv'}"
        )
    stats = [_mode_statistics(measured, mode) for mode in EXPECTED_MODES]
    pairs: list[dict[str, Any]] = []
    expected_case = context["identity"].get("case_id")
    expected_workload = context["identity"].get("workload_id")
    none_rows = {
        (row.get("case_id"), row.get("workload_id"), row["repeat_id"]): row
        for row in measured
        if row["numeric_mode"] == EXPECTED_MODES[0]
    }
    req_rows = {
        (row.get("case_id"), row.get("workload_id"), row["repeat_id"]): row
        for row in measured
        if row["numeric_mode"] == EXPECTED_MODES[1]
    }
    expected_keys = {
        (expected_case, expected_workload, repeat_id)
        for repeat_id in EXPECTED_MEASURED_REPEAT_IDS
    }
    if set(none_rows) != expected_keys or set(req_rows) != expected_keys:
        raise ReportError(
            "measured case/workload/repeat pairing must be exactly one M2.2 identity with repeat IDs {0,1,2,3,4}"
        )
    for repeat_id in sorted(EXPECTED_MEASURED_REPEAT_IDS):
        key = (expected_case, expected_workload, repeat_id)
        none_row, req_row = none_rows[key], req_rows[key]
        pairs.append(
            {
                "case_id": expected_case,
                "workload_id": expected_workload,
                "repeat_id": repeat_id,
                "none_total_time_s": none_row["total_time_s"],
                "requantized_total_time_s": req_row["total_time_s"],
                "requantized_over_none_total_time_ratio": req_row["total_time_s"]
                / none_row["total_time_s"],
                "none_native_route_time_s": none_row["stage_timings"][
                    "total_route_time_s"
                ],
                "requantized_native_route_time_s": req_row["stage_timings"][
                    "total_route_time_s"
                ],
                "requantized_over_none_native_route_ratio": req_row["stage_timings"][
                    "total_route_time_s"
                ]
                / none_row["stage_timings"]["total_route_time_s"],
            }
        )
    if len(pairs) != REQUIRED_MEASURED_PER_MODE:
        raise ReportError(
            f"paired repeat count={len(pairs)} expected {REQUIRED_MEASURED_PER_MODE}"
        )
    _write_csv(output / "mode_statistics.csv", stats, list(stats[0]))
    _write_csv(output / "paired_mode_ratios.csv", pairs, list(pairs[0]))
    _plot_runtime(output / "runtime_by_mode.png", stats)
    _plot_accuracy(output / "accuracy_by_mode.png", stats)
    summary_text = _markdown(
        input_dir,
        source_hashes,
        manifest.get("benchmark_source_commit"),
        stats,
        pairs,
        context,
    )
    (output / "benchmark_summary.md").write_text(summary_text, encoding="utf-8")
    output_files = [
        "mode_statistics.csv",
        "paired_mode_ratios.csv",
        "validation_rows.csv",
        "runtime_by_mode.png",
        "accuracy_by_mode.png",
        "benchmark_summary.md",
        "report_manifest.json",
    ]
    artifacts = _artifact_manifest(output)
    report_manifest = {
        "schema_version": "upmem_m2_2_report_v1",
        "status": "valid",
        "source_run": str(input_dir),
        "source_paths": {name: str(path) for name, path in source_paths.items()},
        "source_hashes": source_hashes,
        "source_commit": manifest.get("benchmark_source_commit")
        or manifest.get("git_commit"),
        "provenance": {
            "source_commit": manifest.get("benchmark_source_commit")
            or manifest.get("git_commit"),
            "worktree_state": {
                key: manifest.get(key)
                for key in (
                    "benchmark_source_worktree_dirty",
                    "repository_worktree_dirty",
                    "dirty_tree",
                    "dirty_worktree",
                )
                if key in manifest
            },
            "hostname": context.get("environment", {}).get("hostname"),
            "host_metadata": {
                key: context.get("environment", {}).get(key)
                for key in (
                    "machine",
                    "platform",
                    "processor",
                    "cpu_count",
                    "physical_cpu_count",
                )
                if key in context.get("environment", {})
            },
            "device_metadata": {
                **(
                    context.get("environment", {}).get("upmem")
                    if isinstance(context.get("environment", {}).get("upmem"), dict)
                    else {}
                ),
                **context.get("execution_metadata", {}),
            },
            "sdk_metadata": context.get("sdk_metadata"),
            "row_hashes": context["row_provenance"],
            "hardware_profile": hardware_profile,
            "environment_sha256": source_hashes.get("environment.json"),
            "resolved_suite_sha256": source_hashes.get("config/resolved_suite.yml"),
            "hardware_profile_sha256": source_hashes.get(
                "config/hardware_profile.json"
            ),
        },
        "counts": {
            "warmups": len(warmups),
            "measured": len(measured),
            "measured_per_mode": REQUIRED_MEASURED_PER_MODE,
        },
        "numeric_modes": list(EXPECTED_MODES),
        "numeric_mode_labels": MODE_LABELS,
        "statistics": {
            "iqr_definition": IQR_METHOD,
            "repeat_ids": sorted(EXPECTED_MEASURED_REPEAT_IDS),
        },
        "identity": context["identity"],
        "executor_hashes": context["executor_hashes"],
        "claims": {
            "physical_functionality": True,
            "bringup_timing_ratio": True,
            "speedup": False,
            "scaling": False,
            "energy": False,
            "general_tn_performance": False,
        },
        "outputs": output_files,
        "artifacts": artifacts,
        "plots": [
            {
                "path": "runtime_by_mode.png",
                "source_csv": "mode_statistics.csv",
                "metric": "median_total_time_s",
                "status": "generated",
            },
            {
                "path": "accuracy_by_mode.png",
                "source_csv": "mode_statistics.csv",
                "metric": "policy_reference_max_abs_error,full_precision_max_abs_error",
                "status": "generated",
            },
        ],
    }
    (output / "report_manifest.json").write_text(
        json.dumps(report_manifest, indent=2) + "\n", encoding="utf-8"
    )
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=DEFAULT_INPUT, help="M2.2 evidence run directory"
    )
    args = parser.parse_args(argv)
    try:
        output = generate_report(args.input)
    except (OSError, ReportError, json.JSONDecodeError) as exc:
        print(f"M2.2 report failed: {exc}", file=sys.stderr)
        return 2
    print(f"comparison_dir={output}")
    print(f"summary={output / 'benchmark_summary.md'}")
    print(f"runtime_plot={output / 'runtime_by_mode.png'}")
    print(f"accuracy_plot={output / 'accuracy_by_mode.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
