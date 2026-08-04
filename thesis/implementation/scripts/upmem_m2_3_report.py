#!/usr/bin/env python3
"""Strict derived report for the physical two-DPU M2.3 study."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / "build" / "matplotlib")
)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    ROOT
    / "runs"
    / "evidence"
    / "upmem_hardware_sliced_resident_m2_3"
    / "upmem_hw_sliced_resident"
    / "latest"
)
COMPARISON_ROOT = ROOT / "runs" / "comparisons" / "upmem_m2_3"
EXPECTED_SCHEMA = "upmem_hardware_sliced_resident_m2_3_v1"
EXPECTED_PROFILE = "hardware_sliced_resident_two_dpu_m2_3_v1"
EXPECTED_ROUTE = "upmem_tn_hardware_sliced_resident_two_dpu"
EXPECTED_BACKEND = "upmem_sdk_hardware_sliced_resident_two_dpu"
EXPECTED_BACKEND_FAMILY = "upmem_sdk"
EXPECTED_FIXTURE_SCOPE = "three_operation_ry_h_ry_full_graph_replicated_prefix"
EXPECTED_EXECUTION_SCOPE = "physical_two_dpu_two_slice_full_replicated_prefix_taskgraph"
EXPECTED_TIMING_SCOPE = "host_observed_sdk_process_wall_and_blocking_sync"
EXPECTED_SOURCE_COMPLETION_SCOPE = "unique_source_tasks_completed_on_every_slice"
EXPECTED_NATIVE_PROFILE = "hardware_sliced_resident_two_dpu_m2_v1"
EXPECTED_PLANNER_EVIDENCE = "modeled"
EXPECTED_EXECUTION_ROUTE_POLICY = EXPECTED_PROFILE
EXPECTED_PLANNER_ROUTE_RELATION = (
    "fixed_modeled_candidate_path_executed_on_different_two_dpu_sliced_resident_route"
)
EXPECTED_MODES = ("none", "per_task_resident_requantize")
MODE_LABELS = {
    "none": "Float32",
    "per_task_resident_requantize": "Resident requantized int8",
}
EXPECTED_PATHS = ("opt_einsum_greedy", "custom_upmem_v2_balanced")
PATH_LABELS = {
    "opt_einsum_greedy": "opt_einsum greedy",
    "custom_upmem_v2_balanced": "custom UPMEM v2 balanced",
}
EXPECTED_FIXTURES = ("ry_h_ry_a", "ry_h_ry_b")
CANONICAL_WORKLOAD_IDS = {
    (fixture, path): f"m2_3_{fixture}_{path}"
    for fixture in EXPECTED_FIXTURES
    for path in EXPECTED_PATHS
}
EXPECTED_MEASURED_REPEAT_IDS = frozenset(range(5))
EXPECTED_WARMUP_REPEAT_IDS = frozenset({0})
IQR_METHOD = "numpy.percentile(method='linear'); IQR=P75-P25"
REQUIRED_SOURCE_FILES = (
    "normalized_records.jsonl",
    "warmups.jsonl",
    "run_manifest.json",
    "environment.json",
    "config/resolved_suite.yml",
    "config/hardware_profile.json",
    "upmem_hardware_sliced_resident_mvp_summary.json",
)
VALIDATION_FIELDS = (
    "record_type",
    "phase",
    "fixture_id",
    "case_id",
    "workload_id",
    "path_variant_id",
    "numeric_mode",
    "repeat_id",
    "status",
    "admission_status",
    "admission_errors",
    "validation_status",
    "route_id",
    "backend_id",
    "target_observed",
    "allocated_dpu_count",
    "tasklets_per_dpu",
    "allocation_verified",
    "launch_completed",
    "release_confirmed",
    "native_session_profile_version",
    "planner_candidate_evidence_type",
    "execution_route_policy",
    "planner_policy_matches_execution_route",
    "planner_route_relation",
    "planner_execution_policy",
    "operation_count",
    "source_task_count",
    "source_task_completion_count",
    "source_task_completion_scope",
    "expanded_task_count",
    "expanded_task_completion_count",
    "completed_task_count",
    "slice_model_task_count",
    "slice_model_executed_task_count",
    "slice_model_task_count_scope",
    "slice_model_executed_task_count_scope",
    "slice_count",
    "actual_transfer_bytes",
    "native_sdk_stage_time_s",
    "python_end_to_end_time_s",
    "full_precision_max_abs_error",
    "policy_reference_max_abs_error",
)


class ReportError(ValueError):
    """Raised when source evidence cannot be admitted."""


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ReportError(f"expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ReportError(f"expected JSON object at {path}:{number}")
            rows.append(value)
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _field(row: Mapping[str, Any], path: str, default: Any = None) -> Any:
    value: Any = row
    for part in path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return default
        value = value[part]
    return value


def _finite(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _positive(value: Any) -> bool:
    return _finite(value) and float(value) > 0.0


def _nonnegative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _exact(row: Mapping[str, Any], field: str, expected: Any) -> list[str]:
    if field not in row:
        return [f"missing_{field}"]
    if row[field] != expected:
        return [f"{field}_expected_{expected!r}"]
    return []


def _byte_errors(row: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    groups = {
        "actual": ("actual_h2d_bytes", "actual_d2h_bytes", "actual_transfer_bytes"),
        "planned": ("planned_h2d_bytes", "planned_d2h_bytes", "planned_transfer_bytes"),
        "application": (
            "application_visible_h2d_bytes",
            "application_visible_d2h_bytes",
            "application_visible_transfer_bytes",
        ),
    }
    values: dict[str, tuple[int, int, int]] = {}
    for name, fields in groups.items():
        if not all(_nonnegative_int(row.get(field)) for field in fields):
            errors.append(f"{name}_bytes_complete_nonnegative")
            continue
        first, second, total = (int(row[field]) for field in fields)
        if total != first + second:
            errors.append(f"{name}_parts_sum")
        values[name] = (first, second, total)
    if len(values) == len(groups):
        if values["actual"] != values["planned"]:
            errors.append("actual_planned_transfer_mismatch")
        if values["actual"] != values["application"]:
            errors.append("actual_application_transfer_mismatch")
    observed = ("observed_h2d_bytes", "observed_d2h_bytes", "observed_transfer_bytes")
    if any(row.get(field) is not None for field in observed):
        if not all(_nonnegative_int(row.get(field)) for field in observed):
            errors.append("observed_bytes_complete_nonnegative")
        else:
            observed_values = tuple(int(row[field]) for field in observed)
            if observed_values[2] != observed_values[0] + observed_values[1]:
                errors.append("observed_parts_sum")
            elif len(values) == len(groups) and observed_values != values["actual"]:
                errors.append("actual_observed_transfer_mismatch")
    total = row.get("application_visible_total_bytes")
    if not _nonnegative_int(total):
        errors.append("application_visible_total_bytes_nonnegative")
    elif len(values) == len(groups) and total != values["actual"][2]:
        errors.append("application_visible_total_mismatch")
    return errors


def _timing_errors(row: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in ("total_time_s", "process_time_s", "reconstruction_time_s"):
        if not _positive(row.get(field)):
            errors.append(f"positive_finite_{field}")
    for field in ("stage_timings.total_route_time_s", "stage_timings.sync_wait_time_s"):
        if not _positive(_field(row, field)):
            errors.append(f"positive_finite_{field}")
    for field, expected in (
        ("timing_is_bringup_only", True),
        ("timing_scope", EXPECTED_TIMING_SCOPE),
        ("timing_breakdown_status", "host_stage_boundaries"),
    ):
        errors.extend(_exact(row, field, expected))
    stage = row.get("stage_timings")
    if not isinstance(stage, Mapping):
        return errors + ["missing_stage_timings_object"]
    errors.extend(_exact(stage, "status", "host_stage_boundaries"))
    errors.extend(_exact(stage, "kernel_time_s", None))
    errors.extend(_exact(stage, "sync_wait_is_not_pure_kernel_time", True))
    return errors


def _task_errors(row: Mapping[str, Any]) -> list[str]:
    expected = {
        "operation_count": 3,
        "source_task_count": 3,
        "source_task_completion_count": 3,
        "source_task_completion_scope": EXPECTED_SOURCE_COMPLETION_SCOPE,
        "expanded_task_count": 6,
        "expanded_task_completion_count": 6,
        "expanded_task_count_scope": (
            "physical_source_operation_instances_across_two_slices"
        ),
        "executed_task_count": 6,
        "completed_task_count": 6,
        "executed_task_count_scope": (
            "compatibility_alias_for_expanded_task_completion_count"
        ),
        "completed_task_count_scope": (
            "compatibility_alias_for_expanded_task_completion_count"
        ),
        "slice_model_task_count": 2,
        "slice_model_operation_count": 3,
        "slice_model_executed_task_count": 2,
        "slice_descriptor_count": 2,
        "slice_descriptor_completion_count": 2,
        "slice_model_task_count_scope": "slice_descriptors",
        "slice_model_executed_task_count_scope": "completed_slice_descriptors",
        "slice_model_operation_count_scope": ("source_operations_replicated_per_slice"),
        "expanded_physical_operation_count": 6,
        "expanded_physical_operation_completion_count": 6,
        "operations_per_slice": 3,
        "slice_count": 2,
        "source_slice_count": 2,
        "executed_slice_count": 2,
        "completed_slice_count": 2,
        "slice_ids": [0, 1],
        "observed_operation_completion_counts": [3, 3],
        "completed_operation_count_per_slice": [3, 3],
        "completed_physical_task_instance_count": 6,
        "expected_physical_task_instance_count": 6,
        "physical_task_instances_per_slice": [3, 3],
    }
    return [
        error
        for field, expected in expected.items()
        for error in _exact(row, field, expected)
    ]


def _planner_errors(row: Mapping[str, Any]) -> list[str]:
    path = row.get("path_variant_id")
    expected_paths = {
        "opt_einsum_greedy": [[0, 1], [0, 1], [0, 1]],
        "custom_upmem_v2_balanced": [[0, 1], [0, 2], [0, 1]],
    }
    errors: list[str] = []
    for field, expected in {
        "planner_candidate_evidence_type": EXPECTED_PLANNER_EVIDENCE,
        "execution_route_policy": EXPECTED_EXECUTION_ROUTE_POLICY,
        "planner_policy_matches_execution_route": False,
        "planner_route_relation": EXPECTED_PLANNER_ROUTE_RELATION,
    }.items():
        errors.extend(_exact(row, field, expected))
    if path in expected_paths:
        errors.extend(_exact(row, "planner_path", expected_paths[path]))
        errors.extend(_exact(row, "expected_path", expected_paths[path]))
    if path == "opt_einsum_greedy":
        for field, expected in {
            "planner_engine": "opt_einsum",
            "planner_config": {"engine": "opt_einsum", "optimize": "greedy"},
            "planner_objective_version": None,
            "planner_weight_profile": None,
            "planner_profile": None,
            "planner_selection_scope": None,
            "planner_normalization": None,
            "planner_execution_policy": None,
        }.items():
            errors.extend(_exact(row, field, expected))
    elif path == "custom_upmem_v2_balanced":
        for field, expected in {
            "planner_engine": "custom_upmem",
            "planner_config": {
                "engine": "custom_upmem",
                "algorithm": "greedy",
                "objective_version": "upmem_path_cost_v2",
                "selection_scope": "projected_prefix",
                "weight_profile": "balanced_literature_informed",
                "normalization": "fixed_log1p_generic_budgets_v2",
                "execution_policy": "generic_single_dpu_split_complex_v2",
            },
            "planner_objective_version": "upmem_path_cost_v2",
            "planner_selection_scope": "projected_prefix",
            "planner_weight_profile": "balanced_literature_informed",
            "planner_profile": "balanced_literature_informed",
            "planner_normalization": "fixed_log1p_generic_budgets_v2",
            "planner_execution_policy": "generic_single_dpu_split_complex_v2",
        }.items():
            errors.extend(_exact(row, field, expected))
    return errors


def _admission_errors(row: Mapping[str, Any], phase: str) -> list[str]:
    errors: list[str] = []
    exact = {
        "status": "completed",
        "phase": phase,
        "route_id": EXPECTED_ROUTE,
        "backend_id": EXPECTED_BACKEND,
        "backend_family": EXPECTED_BACKEND_FAMILY,
        "target_requested": "hardware",
        "target_observed": "hardware",
        "requested_dpu_count": 2,
        "allocated_dpu_count": 2,
        "tasklets_per_dpu": 1,
        "hardware_execution": True,
        "hardware_functionality_evidence": True,
        "hardware_kernel_executed": True,
        "native_kernel_executed": True,
        "simulator_kernel_executed": False,
        "cpu_fallback_used": False,
        "hardware_speedup_applicable": False,
        "performance_claim_applicable": False,
        "energy_measurement_available": False,
        "fixture_version": EXPECTED_SCHEMA,
        "fixture_scope": EXPECTED_FIXTURE_SCOPE,
        "execution_scope": EXPECTED_EXECUTION_SCOPE,
        "experiment_schema_version": EXPECTED_SCHEMA,
        "experiment_profile_version": EXPECTED_PROFILE,
        "hardware_profile_version": EXPECTED_PROFILE,
        "native_session_profile_version": EXPECTED_NATIVE_PROFILE,
        "execution_contract_status": "passed",
        "slice_package_validation_status": "passed",
        "per_slice_output_validation_status": "passed",
        "reconstruction_validation_status": "passed",
        "final_output_validation_status": "passed",
        "policy_reference_status": "passed",
        "full_precision_accuracy_status": "passed",
        "scientific_validation_status": "passed",
        "validation_status": "passed",
        "expected_output_validation": True,
        "transfer_accounting_invariant": True,
        "transfer_matches_manifest_plan": True,
        "timing_is_bringup_only": True,
    }
    for field, expected in exact.items():
        errors.extend(_exact(row, field, expected))
    for field in ("case_id", "workload_id", "fixture_id", "path_variant_id"):
        if not isinstance(row.get(field), str) or not row[field]:
            errors.append(f"{field}_invalid")
    if not isinstance(row.get("repeat_id"), int) or isinstance(
        row.get("repeat_id"), bool
    ):
        errors.append("repeat_id_invalid")
    if row.get("fixture_id") not in EXPECTED_FIXTURES:
        errors.append("fixture_id_not_in_m2_3_matrix")
    if row.get("path_variant_id") not in EXPECTED_PATHS:
        errors.append("path_variant_id_not_in_m2_3_matrix")
    if row.get("numeric_mode") not in EXPECTED_MODES:
        errors.append("numeric_mode_not_in_m2_3_matrix")
    expected_repeats = (
        EXPECTED_MEASURED_REPEAT_IDS
        if phase == "measured"
        else EXPECTED_WARMUP_REPEAT_IDS
    )
    if row.get("repeat_id") not in expected_repeats:
        errors.append(f"{phase}_repeat_id_invalid")
    if row.get("response_numeric_mode") != row.get("numeric_mode"):
        errors.append("response_numeric_mode_mismatch")
    if row.get("strict_cpu_reference_validation") is not (
        row.get("numeric_mode") == "none"
    ):
        errors.append("strict_cpu_reference_validation_semantics")
    fixture = row.get("fixture_id")
    path = row.get("path_variant_id")
    canonical_id = CANONICAL_WORKLOAD_IDS.get((fixture, path))
    if canonical_id is not None:
        errors.extend(_exact(row, "case_id", canonical_id))
        errors.extend(_exact(row, "workload_id", canonical_id))
    for field in (
        "case_id",
        "workload_id",
        "fixture_id",
        "path_variant_id",
        "numeric_mode",
        "repeat_id",
    ):
        if field not in row:
            errors.append(f"missing_{field}")
    errors.extend(_task_errors(row))
    errors.extend(_timing_errors(row))
    errors.extend(_byte_errors(row))
    if (
        not isinstance(row.get("allocation_evidence"), Mapping)
        or row["allocation_evidence"].get("verified") is not True
        or row["allocation_evidence"].get("requested_dpus") != 2
        or row["allocation_evidence"].get("allocated_dpus") != 2
    ):
        errors.append("allocation_evidence_not_verified")
    if (
        not isinstance(row.get("launch_evidence"), Mapping)
        or row["launch_evidence"].get("completed") is not True
    ):
        errors.append("launch_evidence_not_completed")
    if (
        not isinstance(row.get("release_evidence"), Mapping)
        or row["release_evidence"].get("confirmed") is not True
    ):
        errors.append("release_not_confirmed")
    if row.get("native_cleanup_confirmed") is not True:
        errors.append("native_cleanup_not_confirmed")
    if row.get("device_completion_confirmed") is not True:
        errors.append("device_completion_not_confirmed")
    if row.get("native_execution_sentinel_available") is not True:
        errors.append("native_execution_sentinel_unavailable")
    if row.get("completion_sentinel_read_counts") != [3, 3]:
        errors.append("completion_sentinel_counts")
    if row.get("validation_errors") not in ([], None):
        errors.append("validation_errors_present")
    errors.extend(_planner_errors(row))
    return list(dict.fromkeys(errors))


def _validation_row(
    row: Mapping[str, Any],
    phase: str,
    errors: Iterable[str],
    *,
    record_type: str = "row",
) -> dict[str, Any]:
    values = list(dict.fromkeys(errors))
    return {
        "record_type": record_type,
        "phase": phase,
        "fixture_id": row.get("fixture_id"),
        "case_id": row.get("case_id"),
        "workload_id": row.get("workload_id"),
        "path_variant_id": row.get("path_variant_id"),
        "numeric_mode": row.get("numeric_mode"),
        "repeat_id": row.get("repeat_id"),
        "status": row.get("status"),
        "admission_status": "passed" if not values else "rejected",
        "admission_errors": ";".join(values),
        "validation_status": row.get("validation_status"),
        "route_id": row.get("route_id"),
        "backend_id": row.get("backend_id"),
        "target_observed": row.get("target_observed"),
        "allocated_dpu_count": row.get("allocated_dpu_count"),
        "tasklets_per_dpu": row.get("tasklets_per_dpu"),
        "allocation_verified": _field(row, "allocation_evidence.verified"),
        "launch_completed": _field(row, "launch_evidence.completed"),
        "release_confirmed": _field(row, "release_evidence.confirmed"),
        "native_session_profile_version": row.get("native_session_profile_version"),
        "planner_candidate_evidence_type": row.get("planner_candidate_evidence_type"),
        "execution_route_policy": row.get("execution_route_policy"),
        "planner_policy_matches_execution_route": row.get(
            "planner_policy_matches_execution_route"
        ),
        "planner_route_relation": row.get("planner_route_relation"),
        "planner_execution_policy": row.get("planner_execution_policy"),
        "operation_count": row.get("operation_count"),
        "source_task_count": row.get("source_task_count"),
        "source_task_completion_count": row.get("source_task_completion_count"),
        "expanded_task_count": row.get("expanded_task_count"),
        "source_task_completion_scope": row.get("source_task_completion_scope"),
        "expanded_task_completion_count": row.get("expanded_task_completion_count"),
        "completed_task_count": row.get("completed_task_count"),
        "slice_model_task_count": row.get("slice_model_task_count"),
        "slice_model_executed_task_count": row.get("slice_model_executed_task_count"),
        "slice_model_task_count_scope": row.get("slice_model_task_count_scope"),
        "slice_model_executed_task_count_scope": row.get(
            "slice_model_executed_task_count_scope"
        ),
        "slice_count": row.get("slice_count"),
        "actual_transfer_bytes": row.get("actual_transfer_bytes"),
        "native_sdk_stage_time_s": _field(row, "stage_timings.total_route_time_s"),
        "python_end_to_end_time_s": row.get("total_time_s"),
        "full_precision_max_abs_error": row.get("full_precision_max_abs_error"),
        "policy_reference_max_abs_error": row.get("policy_reference_max_abs_error"),
    }


def _key(row: Mapping[str, Any]) -> tuple[Any, Any, Any, Any]:
    return (
        row.get("fixture_id"),
        row.get("path_variant_id"),
        row.get("numeric_mode"),
        row.get("repeat_id"),
    )


def _matrix_errors(
    warmups: list[dict[str, Any]], measured: list[dict[str, Any]]
) -> list[str]:
    expected_warmups = {
        (fixture, path, mode, 0)
        for fixture in EXPECTED_FIXTURES
        for path in EXPECTED_PATHS
        for mode in EXPECTED_MODES
    }
    expected_measured = {
        (fixture, path, mode, repeat)
        for fixture in EXPECTED_FIXTURES
        for path in EXPECTED_PATHS
        for mode in EXPECTED_MODES
        for repeat in EXPECTED_MEASURED_REPEAT_IDS
    }
    errors: list[str] = []
    for label, rows, expected in (
        ("warmup", warmups, expected_warmups),
        ("measured", measured, expected_measured),
    ):
        keys = [_key(row) for row in rows]
        if len(keys) != len(set(keys)):
            errors.append(f"{label}_duplicate_matrix_key")
        if set(keys) != expected:
            errors.append(f"{label}_matrix_not_exact")
    all_rows = [*warmups, *measured]
    for fixture in EXPECTED_FIXTURES:
        rows = [row for row in all_rows if row.get("fixture_id") == fixture]
        for field in ("circuit_semantics_hash", "tensor_network_hash"):
            values = {row.get(field) for row in rows}
            if len(values) != 1 or None in values:
                errors.append(f"{fixture}_stable_{field}")
        path_plans = {
            path: {
                row.get("contraction_plan_hash")
                for row in rows
                if row.get("path_variant_id") == path
            }
            for path in EXPECTED_PATHS
        }
        if any(len(values) != 1 or None in values for values in path_plans.values()):
            errors.append(f"{fixture}_plan_hash_not_stable")
        elif next(iter(path_plans[EXPECTED_PATHS[0]])) == next(
            iter(path_plans[EXPECTED_PATHS[1]])
        ):
            errors.append(f"{fixture}_path_plan_hashes_not_distinct")
        executors = {
            mode: {
                row.get("executor_config_hash")
                for row in rows
                if row.get("numeric_mode") == mode
            }
            for mode in EXPECTED_MODES
        }
        if any(len(values) != 1 or None in values for values in executors.values()):
            errors.append(f"{fixture}_executor_hash_not_stable")
        elif executors[EXPECTED_MODES[0]] == executors[EXPECTED_MODES[1]]:
            errors.append(f"{fixture}_executor_hashes_not_distinct")
        for path in EXPECTED_PATHS:
            ids = {
                row.get("workload_id")
                for row in rows
                if row.get("path_variant_id") == path
            }
            if len(ids) != 1 or None in ids:
                errors.append(f"{fixture}_{path}_workload_id_not_stable")
    return errors


def _manifest_errors(
    manifest: Mapping[str, Any],
    summary: Mapping[str, Any],
    profile: Mapping[str, Any],
) -> list[str]:
    errors: list[str] = []
    for field, expected in {
        "suite_id": "upmem_hardware_sliced_resident_m2_3",
        "fixture_version": EXPECTED_SCHEMA,
        "fixture_scope": EXPECTED_FIXTURE_SCOPE,
        "execution_scope": EXPECTED_EXECUTION_SCOPE,
        "route_id": EXPECTED_ROUTE,
        "backend_id": EXPECTED_BACKEND,
        "numeric_modes": list(EXPECTED_MODES),
        "operation_count": 3,
        "source_task_count": 3,
    }.items():
        errors.extend(_exact(manifest, field, expected))
    for field, expected in {
        "status": "completed",
        "measured_row_count": 40,
        "warmup_count": 8,
        "measured_passed_count": 40,
        "warmup_passed_count": 8,
        "expected_measured_row_count": 40,
        "expected_warmup_row_count": 8,
        "all_required_records_validated": True,
        "route_id": EXPECTED_ROUTE,
        "backend_id": EXPECTED_BACKEND,
        "fixture_version": EXPECTED_SCHEMA,
        "experiment_profile_version": EXPECTED_PROFILE,
    }.items():
        errors.extend(_exact(summary, field, expected))
    for field, expected in {
        "hardware_profile_version": EXPECTED_PROFILE,
        "native_session_profile_version": EXPECTED_NATIVE_PROFILE,
        "target": "hardware",
        "backend_id": EXPECTED_BACKEND,
        "route_id": EXPECTED_ROUTE,
        "requested_dpu_count": 2,
        "slices": 2,
        "tasklets_per_dpu": 1,
        "numeric_modes": list(EXPECTED_MODES),
        "performance_claim_applicable": False,
    }.items():
        errors.extend(_exact(profile, field, expected))
    commit = manifest.get("benchmark_source_commit") or manifest.get("git_commit")
    if not isinstance(commit, str) or not commit:
        errors.append("missing_source_commit")
    if not isinstance(manifest.get("benchmark_source_worktree_dirty"), bool):
        errors.append("missing_source_worktree_dirty")
    return errors


def _nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _provenance_errors(
    manifest: Mapping[str, Any],
    environment: Mapping[str, Any],
    summary: Mapping[str, Any],
    rows: list[dict[str, Any]],
) -> list[str]:
    errors: list[str] = []
    source_commit = manifest.get("benchmark_source_commit") or manifest.get(
        "git_commit"
    )
    if not _nonempty_string(source_commit):
        errors.append("missing_source_commit")
    if not isinstance(manifest.get("benchmark_source_worktree_dirty"), bool):
        errors.append("missing_source_worktree_dirty")
    for field in ("platform", "machine"):
        if not _nonempty_string(environment.get(field)):
            errors.append(f"missing_environment_{field}")
    environment_commit = environment.get("git_commit")
    if not _nonempty_string(environment_commit):
        errors.append("missing_environment_git_commit")
    elif source_commit != environment_commit:
        errors.append("environment_source_commit_mismatch")
    upmem = environment.get("upmem")
    if not isinstance(upmem, Mapping) or not _nonempty_string(
        upmem.get("dpu_compiler")
    ):
        errors.append("missing_environment_upmem_compiler")

    native_build = summary.get("native_build")
    if not isinstance(native_build, Mapping):
        return errors + ["missing_native_build_provenance"]
    for field, expected in {"attempted": True, "status": "passed"}.items():
        errors.extend(_exact(native_build, field, expected))
    for field in ("source_tree_hash", "host_binary_hash", "dpu_binary_hash"):
        if not _nonempty_string(native_build.get(field)):
            errors.append(f"missing_native_build_{field}")
    sdk_tools = native_build.get("sdk_tools")
    if not isinstance(sdk_tools, Mapping) or not _nonempty_string(
        sdk_tools.get("dpu-upmem-dpurte-clang")
    ):
        errors.append("missing_native_build_upmem_compiler")

    expected_hashes = {
        "host_binary_hash": native_build.get("host_binary_hash"),
        "dpu_binary_hash": native_build.get("dpu_binary_hash"),
        "native_source_tree_hash": native_build.get("source_tree_hash"),
        "binary_source_tree_hash": native_build.get("source_tree_hash"),
    }
    for field, expected in expected_hashes.items():
        values = {row.get(field) for row in rows}
        if len(values) != 1 or not all(_nonempty_string(value) for value in values):
            errors.append(f"unstable_or_missing_{field}")
        elif next(iter(values)) != expected:
            errors.append(f"{field}_native_build_mismatch")
    return errors


def _provenance(
    manifest: Mapping[str, Any],
    environment: Mapping[str, Any],
    summary: Mapping[str, Any],
    profile: Mapping[str, Any],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    native_build = summary.get("native_build")
    hostname = environment.get("hostname") or manifest.get("hostname")
    caveats = (
        []
        if _nonempty_string(hostname)
        else ["hostname_unavailable_in_source_environment"]
    )
    return {
        "source_commit": manifest.get("benchmark_source_commit")
        or manifest.get("git_commit")
        or environment.get("git_commit"),
        "source_worktree_dirty": manifest.get("benchmark_source_worktree_dirty"),
        "hostname": hostname,
        "missing_provenance_caveats": caveats,
        "platform": environment.get("platform"),
        "machine": environment.get("machine"),
        "upmem_compiler": _field(environment, "upmem.dpu_compiler"),
        "sdk_metadata": (
            native_build.get("sdk_tools")
            if isinstance(native_build, Mapping)
            else environment.get("upmem")
        ),
        "device_metadata": environment.get("device"),
        "allocation_metadata": next(
            (
                row.get("allocation_evidence")
                for row in rows
                if isinstance(row.get("allocation_evidence"), Mapping)
            ),
            None,
        ),
        "hardware_profile": profile,
        "binary_hashes": {
            field: next(
                (row.get(field) for row in rows if row.get(field) is not None), None
            )
            for field in (
                "host_binary_hash",
                "dpu_binary_hash",
                "native_source_tree_hash",
                "binary_source_tree_hash",
            )
        },
    }


def validate_source(
    warmups: list[dict[str, Any]],
    measured: list[dict[str, Any]],
    summary: Mapping[str, Any] | None = None,
    manifest: Mapping[str, Any] | None = None,
    environment: Mapping[str, Any] | None = None,
    hardware_profile: Mapping[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    validation_rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for phase, rows in (("warmup", warmups), ("measured", measured)):
        for row in rows:
            row_errors = _admission_errors(row, phase)
            validation_rows.append(_validation_row(row, phase, row_errors))
            errors.extend(f"{phase}:{_key(row)}:{error}" for error in row_errors)
    matrix_errors = _matrix_errors(warmups, measured)
    errors.extend(matrix_errors)
    global_errors = list(matrix_errors)
    if summary is not None and manifest is not None and hardware_profile is not None:
        metadata_errors = _manifest_errors(manifest, summary, hardware_profile)
        global_errors.extend(metadata_errors)
        if environment is None:
            global_errors.append("missing_environment_metadata")
        else:
            global_errors.extend(
                _provenance_errors(
                    manifest, environment, summary, [*warmups, *measured]
                )
            )
    errors.extend(error for error in global_errors if error not in matrix_errors)
    validation_rows.extend(
        _validation_row({}, "global", [error], record_type="global")
        for error in global_errors
    )
    return validation_rows, {
        "errors": list(dict.fromkeys(errors)),
        "valid": not errors,
        "measured_count": len(measured),
        "warmup_count": len(warmups),
    }


def _metric_values(rows: Iterable[Mapping[str, Any]], path: str) -> np.ndarray:
    values = [float(_field(row, path)) for row in rows]
    if not values or not all(np.isfinite(value) for value in values):
        raise ReportError(f"invalid metric: {path}")
    return np.asarray(values, dtype=float)


def _stats(values: np.ndarray) -> tuple[float, float, float, float, float]:
    q1, median, q3 = np.percentile(values, [25, 50, 75], method="linear")
    return (
        float(median),
        float(values.mean()),
        float(values.min()),
        float(values.max()),
        float(q3 - q1),
    )


def _combination_statistics(measured: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for fixture in EXPECTED_FIXTURES:
        for path in EXPECTED_PATHS:
            for mode in EXPECTED_MODES:
                rows = [
                    row
                    for row in measured
                    if row.get("fixture_id") == fixture
                    and row.get("path_variant_id") == path
                    and row.get("numeric_mode") == mode
                ]
                python_end_to_end = _stats(_metric_values(rows, "total_time_s"))
                native_sdk_stage = _stats(
                    _metric_values(rows, "stage_timings.total_route_time_s")
                )
                result.append(
                    {
                        "fixture_id": fixture,
                        "path_variant_id": path,
                        "path_label": PATH_LABELS[path],
                        "numeric_mode": mode,
                        "numeric_mode_label": MODE_LABELS[mode],
                        "repeat_count": len(rows),
                        "median_native_sdk_stage_time_s": native_sdk_stage[0],
                        "mean_native_sdk_stage_time_s": native_sdk_stage[1],
                        "min_native_sdk_stage_time_s": native_sdk_stage[2],
                        "max_native_sdk_stage_time_s": native_sdk_stage[3],
                        "iqr_native_sdk_stage_time_s": native_sdk_stage[4],
                        "median_python_end_to_end_time_s": python_end_to_end[0],
                        "mean_python_end_to_end_time_s": python_end_to_end[1],
                        "min_python_end_to_end_time_s": python_end_to_end[2],
                        "max_python_end_to_end_time_s": python_end_to_end[3],
                        "iqr_python_end_to_end_time_s": python_end_to_end[4],
                        "validation_passed_count": sum(
                            row.get("validation_status") == "passed" for row in rows
                        ),
                        "validation_failed_count": sum(
                            row.get("validation_status") != "passed" for row in rows
                        ),
                        "policy_reference_max_abs_error": max(
                            float(row["policy_reference_max_abs_error"]) for row in rows
                        ),
                        "full_precision_max_abs_error": max(
                            float(row["full_precision_max_abs_error"]) for row in rows
                        ),
                        "median_h2d_bytes": float(
                            np.median(_metric_values(rows, "actual_h2d_bytes"))
                        ),
                        "median_d2h_bytes": float(
                            np.median(_metric_values(rows, "actual_d2h_bytes"))
                        ),
                        "median_transfer_bytes": float(
                            np.median(_metric_values(rows, "actual_transfer_bytes"))
                        ),
                    }
                )
    return result


def _pair_rows(measured: list[dict[str, Any]], kind: str) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    if kind == "numeric":
        for fixture in EXPECTED_FIXTURES:
            for path in EXPECTED_PATHS:
                for repeat in sorted(EXPECTED_MEASURED_REPEAT_IDS):
                    rows = {
                        row["numeric_mode"]: row
                        for row in measured
                        if row.get("fixture_id") == fixture
                        and row.get("path_variant_id") == path
                        and row.get("repeat_id") == repeat
                    }
                    none, requant = rows["none"], rows["per_task_resident_requantize"]
                    result.append(
                        {
                            "fixture_id": fixture,
                            "path_variant_id": path,
                            "repeat_id": repeat,
                            "ratio_name": "requantized_over_none",
                            "none_native_sdk_stage_time_s": _field(
                                none, "stage_timings.total_route_time_s"
                            ),
                            "requantized_native_sdk_stage_time_s": _field(
                                requant, "stage_timings.total_route_time_s"
                            ),
                            "native_sdk_stage_timing_ratio": _field(
                                requant, "stage_timings.total_route_time_s"
                            )
                            / _field(none, "stage_timings.total_route_time_s"),
                            "none_python_end_to_end_time_s": none["total_time_s"],
                            "requantized_python_end_to_end_time_s": requant[
                                "total_time_s"
                            ],
                            "python_end_to_end_timing_ratio": requant["total_time_s"]
                            / none["total_time_s"],
                        }
                    )
    else:
        for fixture in EXPECTED_FIXTURES:
            for mode in EXPECTED_MODES:
                for repeat in sorted(EXPECTED_MEASURED_REPEAT_IDS):
                    rows = {
                        row["path_variant_id"]: row
                        for row in measured
                        if row.get("fixture_id") == fixture
                        and row.get("numeric_mode") == mode
                        and row.get("repeat_id") == repeat
                    }
                    greedy, custom = (
                        rows["opt_einsum_greedy"],
                        rows["custom_upmem_v2_balanced"],
                    )
                    result.append(
                        {
                            "fixture_id": fixture,
                            "numeric_mode": mode,
                            "repeat_id": repeat,
                            "ratio_name": "custom_over_greedy",
                            "greedy_native_sdk_stage_time_s": _field(
                                greedy, "stage_timings.total_route_time_s"
                            ),
                            "custom_native_sdk_stage_time_s": _field(
                                custom, "stage_timings.total_route_time_s"
                            ),
                            "native_sdk_stage_timing_ratio": _field(
                                custom, "stage_timings.total_route_time_s"
                            )
                            / _field(greedy, "stage_timings.total_route_time_s"),
                            "greedy_python_end_to_end_time_s": greedy["total_time_s"],
                            "custom_python_end_to_end_time_s": custom["total_time_s"],
                            "python_end_to_end_timing_ratio": custom["total_time_s"]
                            / greedy["total_time_s"],
                        }
                    )
    return result


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = (
        list(VALIDATION_FIELDS)
        if path.name == "validation_rows.csv"
        else list(dict.fromkeys(field for row in rows for field in row))
    )
    if not fields:
        fields = ["status"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _plot_runtime(path: Path, stats: list[dict[str, Any]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for axis, fixture in zip(axes, EXPECTED_FIXTURES, strict=True):
        selected = [row for row in stats if row["fixture_id"] == fixture]
        labels = [
            f"{PATH_LABELS[row['path_variant_id']]}\n{MODE_LABELS[row['numeric_mode']]}"
            for row in selected
        ]
        positions = np.arange(len(selected))
        medians = [
            float(row["median_native_sdk_stage_time_s"]) * 1000 for row in selected
        ]
        errors = [float(row["iqr_native_sdk_stage_time_s"]) * 500 for row in selected]
        python_medians = [
            float(row["median_python_end_to_end_time_s"]) * 1000 for row in selected
        ]
        axis.bar(
            positions,
            medians,
            yerr=errors,
            capsize=4,
            label="Native SDK stage median +/- half IQR",
        )
        axis.scatter(
            positions,
            python_medians,
            marker="D",
            color="black",
            facecolors="none",
            label="Python end-to-end median (secondary)",
            zorder=3,
        )
        axis.set_xticks(np.arange(len(selected)), labels, rotation=25, ha="right")
        axis.set_title(f"RY-H-RY fixture {fixture[-1].upper()}")
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Host-observed time (ms)")
    axes[1].legend(fontsize=8, loc="best")
    fig.suptitle(
        "M2.3 host-observed native SDK stage time by circuit, path and mode\n"
        "Python orchestration is secondary; kernel-only time is unavailable"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_error(path: Path, stats: list[dict[str, Any]]) -> None:
    selected = [row for row in stats if row["numeric_mode"] == EXPECTED_MODES[1]]
    labels = [
        f"{row['fixture_id'][-1].upper()} / {PATH_LABELS[row['path_variant_id']]}"
        for row in selected
    ]
    x = np.arange(len(selected))
    policy = np.asarray(
        [float(row["policy_reference_max_abs_error"]) for row in selected]
    )
    full = np.asarray([float(row["full_precision_max_abs_error"]) for row in selected])
    floor = 1.0e-12
    fig, axis = plt.subplots(figsize=(11, 5))
    width = 0.36
    axis.bar(
        x - width / 2,
        np.maximum(policy, floor),
        width,
        label="Policy-reference max abs error",
    )
    axis.bar(
        x + width / 2,
        np.maximum(full, floor),
        width,
        label="Full-precision max abs error",
    )
    axis.set_yscale("log")
    axis.set_xticks(x, labels, rotation=20, ha="right")
    axis.set_ylabel("Maximum absolute error (log scale; exact zero at floor)")
    axis.set_title(
        "M2.3 requantized validation error by circuit and path\nPolicy and full-precision validation are separate"
    )
    for positions, values in ((x - width / 2, policy), (x + width / 2, full)):
        for position, value in zip(positions, values):
            axis.text(
                position,
                max(value, floor) * 1.2,
                "0 (exact)" if value == 0 else f"{value:.3g}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    axis.legend(fontsize=8)
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_ratios(
    path: Path, numeric: list[dict[str, Any]], paths: list[dict[str, Any]]
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=False)

    def grouped_panel(
        axis: Any,
        rows: list[dict[str, Any]],
        groups: list[tuple[str, str]],
        labels: list[str],
        group_fields: tuple[str, str],
        title: str,
    ) -> None:
        for position, group in enumerate(groups):
            selected = [
                row
                for row in rows
                if tuple(row[field] for field in group_fields) == group
            ]
            selected.sort(key=lambda row: int(row["repeat_id"]))
            offsets = np.linspace(-0.14, 0.14, len(selected))
            native = np.asarray(
                [float(row["native_sdk_stage_timing_ratio"]) for row in selected]
            )
            python = np.asarray(
                [float(row["python_end_to_end_timing_ratio"]) for row in selected]
            )
            axis.scatter(
                position + offsets,
                native,
                color="tab:blue",
                s=28,
                label="Native SDK stage repeats" if position == 0 else None,
            )
            axis.scatter(
                position + offsets,
                python,
                color="tab:orange",
                marker="x",
                s=30,
                label="Python end-to-end repeats (secondary)"
                if position == 0
                else None,
            )
            axis.hlines(
                float(np.median(native)),
                position - 0.24,
                position + 0.24,
                color="tab:blue",
                linewidth=2.5,
            )
        axis.axhline(1.0, color="black", linestyle="--", linewidth=1)
        axis.set_xticks(np.arange(len(groups)), labels, rotation=22, ha="right")
        axis.set_ylabel("Matched timing ratio")
        axis.set_title(title + "\nValues above 1.0 mean numerator is slower")
        axis.grid(axis="y", alpha=0.25)
        axis.legend(fontsize=8, loc="best")

    numeric_groups = [
        (fixture, path) for fixture in EXPECTED_FIXTURES for path in EXPECTED_PATHS
    ]
    numeric_labels = [
        f"Fixture {fixture[-1].upper()}\n{PATH_LABELS[path]}"
        for fixture, path in numeric_groups
    ]
    grouped_panel(
        axes[0],
        numeric,
        numeric_groups,
        numeric_labels,
        ("fixture_id", "path_variant_id"),
        "Requantized / float32 by fixture and path",
    )
    path_groups = [
        (fixture, mode) for fixture in EXPECTED_FIXTURES for mode in EXPECTED_MODES
    ]
    path_labels = [
        f"Fixture {fixture[-1].upper()}\n{MODE_LABELS[mode]}"
        for fixture, mode in path_groups
    ]
    grouped_panel(
        axes[1],
        paths,
        path_groups,
        path_labels,
        ("fixture_id", "numeric_mode"),
        "Custom / greedy by fixture and numeric mode",
    )
    fig.suptitle(
        "M2.3 matched host-observed timing ratios\n"
        "Native SDK stage is primary; ratios are not speedups"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


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
        raise ReportError("report output must be a child of comparison root")
    output.mkdir(parents=True, exist_ok=False)
    return output


def _write_rejection(
    output: Path,
    source: Path,
    hashes: Mapping[str, str],
    rows: list[dict[str, Any]],
    errors: list[str],
) -> None:
    _write_csv(output / "validation_rows.csv", rows)
    (output / "benchmark_summary.md").write_text(
        "# M2.3 Report Rejected\n\n"
        f"Source: {source}\n\n"
        "The source run was not admitted. No plots or statistics were generated.\n\n"
        + "\n".join(f"- {error}" for error in errors)
        + "\n",
        encoding="utf-8",
    )
    (output / "report_manifest.json").write_text(
        json.dumps(
            {
                "status": "rejected",
                "study": "M2.3 physical two-DPU path and numeric-mode study",
                "source": str(source),
                "source_hashes": dict(hashes),
                "validation_errors": errors,
                "outputs": ["validation_rows.csv", "benchmark_summary.md"],
                "plots": [],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def generate_report(
    input_dir: Path,
    output_dir: Path | None = None,
    *,
    comparison_root: Path | None = None,
) -> Path:
    input_dir = input_dir.resolve()
    output = _output_directory(output_dir, comparison_root)
    paths = {name: input_dir / name for name in REQUIRED_SOURCE_FILES}
    hashes = {name: _sha256(path) for name, path in paths.items() if path.is_file()}
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        rows = [
            _validation_row(
                {}, "source", [f"missing_source_artifact:{name}"], record_type="source"
            )
            for name in missing
        ]
        errors = [f"missing_source_artifact:{name}" for name in missing]
        _write_rejection(output, input_dir, hashes, rows, errors)
        raise ReportError(f"M2.3 source rejected; missing: {', '.join(missing)}")
    try:
        warmups = _read_jsonl(paths["warmups.jsonl"])
        measured = _read_jsonl(paths["normalized_records.jsonl"])
        summary = _read_json(paths["upmem_hardware_sliced_resident_mvp_summary.json"])
        manifest = _read_json(paths["run_manifest.json"])
        environment = _read_json(paths["environment.json"])
        profile = _read_json(paths["config/hardware_profile.json"])
        resolved_suite = yaml.safe_load(
            paths["config/resolved_suite.yml"].read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError, yaml.YAMLError, ReportError) as exc:
        row = _validation_row(
            {}, "source", [f"malformed_source:{exc}"], record_type="source"
        )
        _write_rejection(output, input_dir, hashes, [row], [f"malformed_source:{exc}"])
        raise ReportError(f"M2.3 source rejected: {exc}") from exc
    rows, context = validate_source(
        warmups, measured, summary, manifest, environment, profile
    )
    if (
        not isinstance(resolved_suite, Mapping)
        or resolved_suite.get("suite_id") != "upmem_hardware_sliced_resident_m2_3"
    ):
        context["errors"].append("resolved_suite_id")
    if context["errors"]:
        errors = list(dict.fromkeys(context["errors"]))
        _write_rejection(output, input_dir, hashes, rows, errors)
        raise ReportError(
            f"M2.3 source rejected; validation artifact: {output / 'validation_rows.csv'}"
        )
    _write_csv(output / "validation_rows.csv", rows)
    stats = _combination_statistics(measured)
    numeric_pairs = _pair_rows(measured, "numeric")
    path_pairs = _pair_rows(measured, "path")
    _write_csv(output / "combination_statistics.csv", stats)
    _write_csv(output / "paired_numeric_mode_ratios.csv", numeric_pairs)
    _write_csv(output / "paired_path_ratios.csv", path_pairs)
    _plot_runtime(output / "runtime_by_circuit_path_mode.png", stats)
    _plot_error(output / "quantization_error_by_circuit_path.png", stats)
    _plot_ratios(output / "timing_ratios.png", numeric_pairs, path_pairs)
    provenance = _provenance(manifest, environment, summary, profile, measured)
    (output / "benchmark_summary.md").write_text(
        _markdown(input_dir, stats, numeric_pairs, path_pairs, hashes, provenance),
        encoding="utf-8",
    )
    artifact_names = sorted(path.name for path in output.iterdir() if path.is_file())
    artifacts = {
        name: {
            "sha256": _sha256(output / name),
            "bytes": (output / name).stat().st_size,
        }
        for name in artifact_names
        if name != "report_manifest.json"
    }
    (output / "report_manifest.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "study": "M2.3 physical two-DPU path and numeric-mode study",
                "source": str(input_dir),
                "source_hashes": hashes,
                "provenance": provenance,
                "counts": {"warmups": 8, "measured": 40, "measured_per_combination": 5},
                "statistics": {"iqr_definition": IQR_METHOD},
                "timing_interpretation": {
                    "primary_metric": "stage_timings.total_route_time_s",
                    "primary_label": "host-observed native SDK stage time",
                    "secondary_metric": "total_time_s",
                    "secondary_label": "Python end-to-end orchestration time",
                    "kernel_time_available": False,
                },
                "identity_contract": {
                    "same_circuit_and_network_per_fixture": True,
                    "distinct_plan_hashes_per_path": True,
                    "stable_plan_hash_per_path": True,
                    "stable_executor_hash_per_mode": True,
                    "distinct_executor_hashes_per_mode": True,
                },
                "planner_interpretation": {
                    "custom_candidate": "modeled",
                    "custom_execution_policy": "generic_single_dpu_split_complex_v2",
                    "physical_execution_route": "two-DPU sliced resident",
                    "calibrated_for_physical_route": False,
                    "planner_optimality_claim": False,
                },
                "claims": {
                    "physical_functionality": True,
                    "bringup_timing_ratios": True,
                    "speedup": False,
                    "scaling": False,
                    "energy": False,
                    "planner_optimality": False,
                },
                "plots": [
                    {
                        "path": "runtime_by_circuit_path_mode.png",
                        "source_csv": "combination_statistics.csv",
                        "status": "generated_valid",
                    },
                    {
                        "path": "quantization_error_by_circuit_path.png",
                        "source_csv": "combination_statistics.csv",
                        "status": "generated_valid",
                    },
                    {
                        "path": "timing_ratios.png",
                        "source_csv": "paired_numeric_mode_ratios.csv,paired_path_ratios.csv",
                        "status": "generated_valid",
                    },
                ],
                "artifacts": artifacts,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return output


def _markdown(
    input_dir: Path,
    stats: list[dict[str, Any]],
    numeric_pairs: list[dict[str, Any]],
    path_pairs: list[dict[str, Any]],
    hashes: Mapping[str, str],
    provenance: Mapping[str, Any],
) -> str:
    numeric_ratio = float(
        np.median([row["native_sdk_stage_timing_ratio"] for row in numeric_pairs])
    )
    path_ratio = float(
        np.median([row["native_sdk_stage_timing_ratio"] for row in path_pairs])
    )
    python_numeric_ratio = float(
        np.median([row["python_end_to_end_timing_ratio"] for row in numeric_pairs])
    )
    python_path_ratio = float(
        np.median([row["python_end_to_end_timing_ratio"] for row in path_pairs])
    )
    del stats
    hash_lines = "\n".join(f"- {name}: {value}" for name, value in hashes.items())
    return f"""# M2.3 Physical Two-DPU Path and Numeric-Mode Report

Source run: {input_dir}

## Scope

This report admits exactly 8 passed warmups and 40 passed measured rows:
two deterministic RY-H-RY fixtures, two fixed path variants, two numeric modes,
and five measured repeats per combination. The route used two physical DPUs,
one tasklet per DPU, two slices, and three source contraction operations.

Statistics use {IQR_METHOD}. The primary timing is
`stage_timings.total_route_time_s`, labelled host-observed native SDK stage
time. Python `total_time_s` is secondary end-to-end orchestration timing.
Neither field is kernel-only timing.

## Results

The median matched native SDK stage timing ratio for requantized int8 over
float32 is {numeric_ratio:.6f}x. The corresponding secondary Python
end-to-end ratio is {python_numeric_ratio:.6f}x. The median matched native SDK
stage timing ratio for the custom path over opt_einsum greedy is
{path_ratio:.6f}x; the secondary Python end-to-end ratio is
{python_path_ratio:.6f}x. These are timing ratios, not speedups; values above
1.0 mean the numerator took longer.

Policy-reference validation and full-precision accuracy are reported separately.

## Planner interpretation

The custom path is a modeled candidate using
generic_single_dpu_split_complex_v2. It was executed by the two-DPU sliced
resident route. This mismatch is recorded deliberately: the experiment is not
calibrated planner evidence and does not prove planner optimality.

## Provenance

- Source commit: {provenance.get("source_commit")}
- Source worktree dirty: {provenance.get("source_worktree_dirty")}
- Hostname: {provenance.get("hostname")}
- Platform: {provenance.get("platform")}
- Machine: {provenance.get("machine")}
- UPMEM compiler: {provenance.get("upmem_compiler")}
- Missing-provenance caveats: {provenance.get("missing_provenance_caveats")}
- SDK, device and binary metadata are preserved when present.
- Required source artifact hashes are in report_manifest.json.

## Claims allowed

- The bounded three-operation TaskGraph executed and validated on two physical
  DPUs in both numeric modes and for both selected path variants.
- The timing values are reproducible host-observed bring-up ratios.
- Circuit, TN, contraction-plan and executor identities were checked.
- Policy-reference and full-precision validation results may be reported.

## Claims not allowed

- No speedup or quantization performance claim.
- No scaling, concurrency or energy claim.
- No kernel-only timing claim.
- No claim that the custom planner is optimal or calibrated for this route.
- No general TN quantum-circuit performance claim.

## Figures

- runtime_by_circuit_path_mode.png: primary host-observed native SDK stage
  median and IQR, with secondary Python end-to-end medians, by fixture, fixed
  path, and numeric mode. Source:
  combination_statistics.csv.
- quantization_error_by_circuit_path.png: requantized policy-reference and
  full-precision maximum absolute error by fixture and fixed path. Source:
  combination_statistics.csv.
- timing_ratios.png: per-fixture/path requantized/float32 and per-fixture/mode
  custom/greedy native SDK stage ratios, with secondary Python end-to-end
  ratios and a 1.0 reference. Sources:
  paired_numeric_mode_ratios.csv and paired_path_ratios.csv.

## Artifacts

- combination_statistics.csv
- paired_numeric_mode_ratios.csv
- paired_path_ratios.csv
- validation_rows.csv
- runtime_by_circuit_path_mode.png
- quantization_error_by_circuit_path.png
- timing_ratios.png
- report_manifest.json

## Source hashes

{hash_lines}
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    try:
        output = generate_report(args.input, args.output)
    except (OSError, ReportError, json.JSONDecodeError) as exc:
        print(f"M2.3 report failed: {exc}", file=sys.stderr)
        return 2
    print(f"comparison_dir={output}")
    print(f"summary={output / 'benchmark_summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
