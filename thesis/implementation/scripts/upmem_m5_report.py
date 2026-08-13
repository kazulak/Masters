"""KISS report for an M5 UPMEM evidence run.

The command consumes only ``normalized_records.jsonl``.  It deliberately keeps
the evidence contract small: rows are normalized for grouping, successful
runtime rows are summarized, and every source row (including failures and
unsupported cases) is copied to a report table.  No file is written below the
input evidence directory.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import textwrap
from typing import Any, Iterable, Mapping


COMPATIBILITY_FIELDS = (
    "case_id",
    "route_id",
    "numeric_mode",
    "partition_mode",
    "tasklets_per_dpu",
    "timing_scope",
    "workload_kind",
    "scaling_kind",
)
DIMENSION_FIELDS = (*COMPATIBILITY_FIELDS, "dpu_count")
SUCCESS_STATUSES = {"completed", "passed", "success", "succeeded", "ok", "validated"}
H2D_FIELDS = ("actual_h2d_bytes", "h2d_bytes", "application_visible_h2d_bytes")
D2H_FIELDS = ("actual_d2h_bytes", "d2h_bytes", "application_visible_d2h_bytes")
REDUCTION_FIELDS = (
    "host_mediated_reduction_bytes",
    "host_reduction_bytes",
    "reduction_bytes",
    "reduction_d2h_bytes",
)
TIMESTAMP_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
SHA256_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")
NESTED_EVIDENCE_KEYS = (
    "timing",
    "per_repeat_timing",
    "transfers",
    "transfer",
    "load_balance",
    "load_balance_metrics",
    "validation",
    "accuracy",
    "quantization",
    "metadata",
    "native_response",
)
STAT_FIELDS = [
    *DIMENSION_FIELDS,
    "status",
    "statuses",
    "median",
    "iqr",
    "min",
    "max",
    "repeat_count",
    "measured_repeat_count",
    "unsupported_or_failure_count",
]
RATIO_FIELDS = [
    *DIMENSION_FIELDS[:-1],
    "baseline_dpu_count",
    "dpu_count",
    "baseline_runtime_median_s",
    "runtime_median_s",
    "speedup",
    "efficiency",
    "baseline_repeat_count",
    "repeat_count",
    "pairing_rule",
]
NUMERIC_RATIO_FIELDS = [
    "case_id",
    "route_id",
    "partition_mode",
    "tasklets_per_dpu",
    "timing_scope",
    "workload_kind",
    "scaling_kind",
    "dpu_count",
    "float32_runtime_median_s",
    "int8_runtime_median_s",
    "runtime_ratio_float32_over_int8",
    "float32_repeat_count",
    "int8_repeat_count",
    "pairing_rule",
]
PARTITION_RATIO_FIELDS = [
    "case_id",
    "route_id",
    "numeric_mode",
    "tasklets_per_dpu",
    "timing_scope",
    "workload_kind",
    "scaling_kind",
    "dpu_count",
    "output_runtime_median_s",
    "contracted_runtime_median_s",
    "runtime_ratio_output_over_contracted",
    "output_repeat_count",
    "contracted_repeat_count",
    "pairing_rule",
]
M5_RECORD_FIELDS = (
    *DIMENSION_FIELDS,
    "requested_dpu_count",
    "allocated_dpu_count",
    "repeat_id",
    "status",
    "reason",
    "runtime_s",
    "accuracy_max_abs_error",
    "h2d_bytes",
    "d2h_bytes",
    "host_mediated_reduction_bytes",
    "run_h2d_bytes_provenance",
    "run_d2h_bytes_provenance",
    "run_reduction_bytes_provenance",
    "load_balance_ratio",
    "one_rank",
    "physical_one_rank_valid",
    "validation_status",
    "quantization_evidence",
)
CANONICAL_FALLBACK_FIELDS = (
    "cpu_fallback_used",
    "simulator_kernel_executed",
    "fallback_used",
)
CANONICAL_QUANTIZATION_FIELDS = (
    "quantization_mode",
    "numeric_arithmetic",
    "numeric_transport",
    "requantization_scope",
    "packed_int8_transfer",
)
PAIR_IDENTITY_ALIASES = {
    "task": ("task_id", "task_hash", "selected_task_hash", "task_identity", "task_semantics_hash"),
    "circuit": (
        "circuit_id",
        "circuit_hash",
        "circuit_semantics_hash",
        "circuit_identity",
        "package_circuit_semantics_hash",
    ),
    "network": (
        "network_id",
        "network_hash",
        "network_identity",
        "tensor_network_hash",
        "package_tensor_network_hash",
    ),
    "plan": ("contraction_plan_hash", "task_graph_hash", "package_contraction_plan_hash"),
    "path": (
        "contraction_path_structure_hash",
        "contraction_path_hash",
        "path_hash",
    ),
    "operation": (
        "operation_id",
        "operation_identity",
        "logical_operation_id",
        "operation_hash",
        "operation_semantics_hash",
        "operation_sha256",
        "package_operation_sha256",
    ),
}
BINARY_HASH_ALIASES = {
    "host": ("host_binary_hash", "host_binary_sha256", "host_hash"),
    "dpu": ("dpu_binary_hash", "dpu_binary_sha256", "dpu_hash"),
}


class ReportError(ValueError):
    """Raised for malformed report input."""


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_file(input_path: Path) -> tuple[Path, Path]:
    """Return ``(normalized_records.jsonl, source_run_directory)``."""

    resolved = input_path.expanduser().resolve()
    if resolved.is_dir():
        return resolved / "normalized_records.jsonl", resolved
    return resolved, resolved.parent


def _containers(row: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Return the row and its bounded producer evidence objects."""

    containers: list[Mapping[str, Any]] = [row]
    seen = {id(row)}
    index = 0
    while index < len(containers):
        container = containers[index]
        index += 1
        for key in NESTED_EVIDENCE_KEYS:
            value = container.get(key)
            if isinstance(value, Mapping) and id(value) not in seen:
                containers.append(value)
                seen.add(id(value))
    return containers


def _pick(row: Mapping[str, Any], *names: str) -> Any:
    """Pick a value from a producer row or its nested evidence objects."""

    for name in names:
        for container in _containers(row):
            value = container.get(name)
            if value is not None:
                return value
    return None


def _values(row: Mapping[str, Any], *names: str) -> list[Any]:
    """Return all matching evidence values, including bounded nested evidence."""

    result: list[Any] = []
    for name in names:
        for container in _containers(row):
            value = container.get(name)
            if value is not None:
                result.append(value)
    return result


def _top_pick(row: Mapping[str, Any], *names: str) -> Any:
    """Pick status-like fields without treating validation.status as row status."""

    containers = [row]
    for key in ("metadata", "native_response"):
        value = row.get(key)
        if isinstance(value, Mapping):
            containers.append(value)
    for name in names:
        for container in containers:
            value = container.get(name)
            if value is not None:
                return value
    return None


def _text(value: Any, default: str = "unknown") -> str:
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


def _float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _positive_float(value: Any) -> float | None:
    number = _float(value)
    return number if number is not None and number >= 0 else None


def _integer(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number >= 0 else None


def _requested_dpu_count(row: Mapping[str, Any]) -> int | None:
    return _integer(_pick(row, "requested_dpu_count", "requested_dpus", "dpu_count"))


def _allocated_dpu_count(row: Mapping[str, Any]) -> int | None:
    return _integer(_pick(row, "allocated_dpu_count", "observed_dpu_count", "dpu_count"))


def _dpu_count(row: Mapping[str, Any]) -> int | None:
    """Use the request to identify failed attempts, allocation for measurements."""

    requested = _requested_dpu_count(row)
    allocated = _allocated_dpu_count(row)
    if _status(row) not in SUCCESS_STATUSES and requested is not None:
        return requested
    return allocated


def _is_one_rank(row: Mapping[str, Any]) -> bool | None:
    """Recognize the two accepted rank forms and reject contradictions."""

    direct = _top_pick(row, "one_rank")
    rank_count = _integer(_top_pick(row, "observed_rank_count", "rank_count"))
    if isinstance(direct, bool) and rank_count is not None and direct != (rank_count == 1):
        return None
    if direct is True:
        return True
    if rank_count is not None:
        return rank_count == 1
    if direct is False:
        return False
    return None


def _scaling_kind(row: Mapping[str, Any]) -> str:
    value = _pick(row, "scaling_kind", "scaling_mode", "scaling_type", "scaling")
    if value is None and _pick(row, "weak_scaling") is True:
        return "weak_scaling"
    if value is None and _pick(row, "strong_scaling") is True:
        return "strong_scaling"
    if value is None:
        return "unknown"
    normalized = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in {"strong", "strong_scaling", "same_route_dpu_scaling", "same_route_dpu_strong"} or normalized.endswith("_strong_scaling"):
        return "strong_scaling"
    if normalized in {"weak", "weak_scaling", "same_route_dpu_weak"} or normalized.endswith("_weak_scaling"):
        return "weak_scaling"
    return "unknown"


def _is_weak(row: Mapping[str, Any]) -> bool:
    return _scaling_kind(row) == "weak_scaling"


def _status(row: Mapping[str, Any]) -> str:
    return _text(_top_pick(row, "status", "execution_status", "result_status"), "unknown").lower()


def _timing_value(row: Mapping[str, Any]) -> Any:
    for container, name in (
        (row, "timing_s"),
        (row.get("timing"), "total_time_s"),
        (row.get("per_repeat_timing"), "total_time_s"),
    ):
        if isinstance(container, Mapping) and container.get(name) is not None:
            return container[name]
    return None


def _runtime(row: Mapping[str, Any], *, valid_figure: bool) -> float | None:
    if not valid_figure or _status(row) not in SUCCESS_STATUSES:
        return None
    runtime = _float(_timing_value(row))
    return runtime if runtime is not None and runtime > 0.0 else None


def _metric(row: Mapping[str, Any], fields: Iterable[str], *, valid_figure: bool) -> float | None:
    if not valid_figure or _status(row) not in SUCCESS_STATUSES:
        return None
    return _positive_float(_pick(row, *fields))


def _repeat_containers(row: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Return only evidence objects describing the current repeat."""

    roots: list[Mapping[str, Any]] = []
    for key in ("per_repeat_timing", "repeat_timing", "repeat_metrics", "per_repeat", "repeat", "timing"):
        value = row.get(key)
        if isinstance(value, Mapping):
            roots.append(value)
        elif isinstance(value, list):
            repeat_id = row.get("repeat_id")
            matches = [item for item in value if isinstance(item, Mapping) and item.get("repeat_id") == repeat_id]
            roots.extend(matches)
    containers: list[Mapping[str, Any]] = []
    for root in roots:
        containers.extend(_containers(root))
    return containers


def _repeat_metric(row: Mapping[str, Any], fields: Iterable[str], *, valid_figure: bool) -> float | None:
    if not valid_figure or _status(row) not in SUCCESS_STATUSES:
        return None
    for field in fields:
        for container in _repeat_containers(row):
            value = container.get(field)
            if value is not None:
                return _positive_float(value)
    return None


def _run_metric(row: Mapping[str, Any], fields: Iterable[str]) -> float | None:
    """Read only explicit run-level transfer provenance, in precedence order."""

    roots: list[Mapping[str, Any]] = []
    global_transfers = row.get("run_global_transfers")
    if isinstance(global_transfers, Mapping):
        roots.append(global_transfers)
    run_metadata = row.get("run_metadata")
    if isinstance(run_metadata, Mapping):
        for key in ("transfers", "run_totals"):
            value = run_metadata.get(key)
            if isinstance(value, Mapping):
                roots.append(value)
    for field in fields:
        for root in roots:
            value = root.get(field)
            if value is not None:
                return _positive_float(value)
    return None


def _first_identity_value(row: Mapping[str, Any], aliases: Iterable[str]) -> Any:
    for value in _values(row, *tuple(aliases)):
        if value is not None and str(value).strip() not in {"", "unknown"}:
            return value
    return None


def _pairing_identities(row: Mapping[str, Any]) -> dict[str, Any]:
    return {name: _first_identity_value(row, aliases) for name, aliases in PAIR_IDENTITY_ALIASES.items()}


def _binary_hashes(row: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, aliases in BINARY_HASH_ALIASES.items():
        value = _first_identity_value(row, aliases)
        for container in _containers(row):
            hashes = container.get("binary_hashes")
            if isinstance(hashes, Mapping):
                value = value or _first_identity_value(hashes, aliases)
        if value is not None:
            result[name] = value
    return result


def _identities_and_hashes_valid(row: Mapping[str, Any]) -> bool:
    identities = _pairing_identities(row)
    hashes = _binary_hashes(row)
    return (
        all(value is not None and str(value).strip() not in {"", "unknown"} for value in identities.values())
        and all(isinstance(hashes.get(name), str) and SHA256_PATTERN.fullmatch(hashes[name]) for name in BINARY_HASH_ALIASES)
    )


def _pair_compatible(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    *,
    ignored_identities: frozenset[str] = frozenset(),
) -> bool:
    left_identities = left.get("pairing_identities", {})
    right_identities = right.get("pairing_identities", {})
    if not isinstance(left_identities, Mapping) or not isinstance(right_identities, Mapping):
        return False
    if any(
        left_identities.get(name) is None
        or right_identities.get(name) is None
        or left_identities.get(name) != right_identities.get(name)
        for name in PAIR_IDENTITY_ALIASES
        if name not in ignored_identities
    ):
        return False
    left_hashes = left.get("binary_hashes", {})
    right_hashes = right.get("binary_hashes", {})
    if not isinstance(left_hashes, Mapping) or not isinstance(right_hashes, Mapping):
        return False
    return all(
        left_hashes.get(name) == right_hashes.get(name)
        for name in BINARY_HASH_ALIASES
        if left_hashes.get(name) is not None or right_hashes.get(name) is not None
    ) and all(
        left_hashes.get(name) is not None and right_hashes.get(name) is not None
        for name in BINARY_HASH_ALIASES
        if left_hashes.get(name) is not None or right_hashes.get(name) is not None
    )


def _load_balance(row: Mapping[str, Any], *, valid_figure: bool) -> float | None:
    if not valid_figure or _status(row) not in SUCCESS_STATUSES:
        return None
    direct = _positive_float(
        _pick(
            row,
            "load_balance_ratio",
            "ratio",
            "max_min_ratio",
            "max_min_runtime_ratio",
            "imbalance_ratio",
            "work_balance_ratio",
        )
    )
    if direct is not None:
        return direct
    values = _pick(
        row,
        "dpu_runtime_s",
        "per_dpu_runtime_s",
        "runtime_by_dpu",
        "dpu_times_s",
        "per_dpu",
    )
    if isinstance(values, Mapping):
        values = list(values.values())
    if isinstance(values, (list, tuple)):
        numbers = []
        for value in values:
            if isinstance(value, Mapping):
                value = _pick(value, "runtime_s", "runtime_cycles", "work_elements", "work")
            number = _positive_float(value)
            if number is not None:
                numbers.append(number)
        if numbers and min(numbers) > 0:
            return max(numbers) / min(numbers)
    return None


def _no_fallback(row: Mapping[str, Any]) -> bool:
    if any(row.get(field) is not False for field in CANONICAL_FALLBACK_FIELDS):
        return False
    return _all_false(row, "cpu_fallback", "simulator", "cpu_fallback_used", "simulator_kernel_executed", "fallback_used")


def _all_false(row: Mapping[str, Any], *names: str) -> bool:
    values = _values(row, *names)
    return bool(values) and all(value is False for value in values)


def _version_is_v3(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return value == 3
    normalized = str(value).strip().lower().replace("-", "_")
    return normalized == "3" or re.search(r"(?:^|[_./])v?3(?:$|[_./])", normalized) is not None


def _schema_and_route_v3(row: Mapping[str, Any]) -> bool:
    schema_values = _values(
        row,
        "schema_version",
        "schema",
        "record_schema_version",
        "normalized_record_schema_version",
        "native_response_schema",
        "execution_plan_schema_version",
    )
    nested_schema_values = [
        container.get("schema_version")
        for container in _containers(row)[1:]
        if container.get("schema_version") is not None
    ]
    route_values = _values(
        row,
        "route_version",
        "route_schema",
        "route_schema_version",
        "route_abi_version",
        "execution_plan_kind",
        "upmem_execution_mode",
        "native_plan_kind",
        "native_response_schema",
        "route_id",
        "route",
        "route_label",
    )
    return any(_version_is_v3(value) for value in schema_values) and any(
        _version_is_v3(value) for value in (*route_values, *nested_schema_values)
    )


def _verified(row: Mapping[str, Any], boolean_names: tuple[str, ...], status_names: tuple[str, ...] = ()) -> bool:
    boolean_values = _values(row, *boolean_names)
    if boolean_values and (any(value is False for value in boolean_values) or not all(value is True for value in boolean_values)):
        return False
    status_values = _values(row, *status_names)
    valid_statuses = {"verified", "passed", "success", "succeeded", "validated", "released", "confirmed", "exact"}
    if status_values and any(str(value).strip().lower() not in valid_statuses for value in status_values):
        return False
    return bool(boolean_values or status_values) and not any(value is False for value in boolean_values)


def _policy_reference_passed(row: Mapping[str, Any]) -> bool:
    values = _values(
        row,
        "policy_reference_validation",
        "policy_reference_validation_status",
        "policy_reference_status",
    )
    if not values:
        return False
    for value in values:
        if isinstance(value, Mapping):
            if value.get("passed") is True:
                continue
            status = value.get("status")
            if isinstance(status, str) and status.strip().lower() in {"passed", "verified", "success", "validated"}:
                continue
            return False
        if value is True or (isinstance(value, str) and value.strip().lower() in {"passed", "verified", "success", "validated"}):
            continue
        return False
    return True


def _observed_rank_count(row: Mapping[str, Any]) -> int | None:
    return _integer(_pick(row, "observed_rank_count", "rank_count"))


def _physical_one_rank_valid(row: Mapping[str, Any], allocated_dpu_count: int | None) -> bool:
    requested_evidence = _integer(_pick(row, "requested_dpu_count", "requested_dpus"))
    allocated_evidence = _integer(_pick(row, "allocated_dpu_count", "observed_dpu_count"))
    return (
        _schema_and_route_v3(row)
        and row.get("route_id") == "upmem_tn_hardware_distributed_m5"
        and row.get("backend_id") == "upmem_sdk_hardware_distributed_m5"
        and row.get("native_provider_kind") == "default_native"
        and _top_pick(row, "target_observed") == "physical_hardware"
        and requested_evidence is not None
        and allocated_evidence is not None
        and requested_evidence == allocated_evidence
        and allocated_evidence == allocated_dpu_count
        and _is_one_rank(row) is True
        and _identities_and_hashes_valid(row)
        and _no_fallback(row)
        and _verified(
            row,
            ("hardware_allocation_verified", "allocation_verified"),
            ("allocation_status", "allocation_verification_status"),
        )
        and _verified(
            row,
            ("native_execution", "native_execution_verified", "native_kernel_executed", "native_verified"),
            ("native_execution_status", "native_verification_status"),
        )
        and _verified(
            row,
            ("hardware_execution", "hardware_execution_verified", "hardware_kernel_executed", "hardware_verified"),
            ("hardware_execution_status", "hardware_verification_status"),
        )
        and _verified(row, ("hardware_functionality_evidence",))
        and _verified(
            row,
            ("hardware_release_verified", "release_verified", "release_confirmed"),
            ("release_status", "release_verification_status"),
        )
        and _policy_reference_passed(row)
        and allocated_dpu_count is not None
        and allocated_dpu_count > 0
    )


def _quantization_evidence(row: Mapping[str, Any]) -> bool:
    if any(field not in row for field in CANONICAL_QUANTIZATION_FIELDS):
        return False
    mode = str(row["quantization_mode"]).lower()
    arithmetic = str(row["numeric_arithmetic"]).lower()
    transport = str(row["numeric_transport"]).lower()
    scope = str(row["requantization_scope"]).lower()
    packed = row["packed_int8_transfer"]
    int8_mode = "int8" in mode or "requant" in mode
    per_task = "per_task" in mode or scope in {"dpu", "on_dpu", "per_task", "per_task_on_dpu"}
    return (
        int8_mode
        and per_task
        and arithmetic in {"int8", "int8_requantized", "int8_requantization"}
        and transport in {"float32_mram", "float32", "float32_mram_transport"}
        and packed is False
    )


def _partition_reduction_evidence(row: Mapping[str, Any]) -> bool:
    """Require the canonical contracted-route provider declarations."""

    return (
        row.get("collective_provider") == "host_mediated_sum_v1"
        and row.get("reconstruction_provider") == "host_float64_reduction_v1"
    )


def _accuracy_value(value: Any) -> float | None:
    if isinstance(value, Mapping):
        for field in ("max_abs_error", "full_precision_max_abs_error"):
            if value.get(field) is not None:
                return _positive_float(value[field])
        return None
    return _positive_float(value)


def _quantization_accuracy(row: Mapping[str, Any]) -> float | None:
    """Read quantization/full-precision evidence, never policy-reference error."""

    mode = str(_pick(row, "quantization_mode", "numeric_mode", "precision_mode") or "").lower()
    if "int8" in mode or "requant" in mode:
        quantization_error = _pick(row, "quantization_error_vs_float32")
        if quantization_error is not None:
            return _accuracy_value(quantization_error)
    return _accuracy_value(_pick(row, "full_precision_accuracy"))


def _view(row: Mapping[str, Any], source_index: int) -> dict[str, Any]:
    status = _status(row)
    requested_dpu_count = _requested_dpu_count(row)
    allocated_dpu_count = _allocated_dpu_count(row)
    dpu_count = _dpu_count(row)
    one_rank = _is_one_rank(row)
    physical_one_rank_valid = _physical_one_rank_valid(row, allocated_dpu_count)
    return {
        "source_index": source_index,
        "case_id": _text(_pick(row, "case_id", "workload_case_id", "benchmark_case_id")),
        "route_id": _text(_pick(row, "route_id", "route", "route_label")),
        "numeric_mode": _text(_pick(row, "numeric_mode", "quantization_mode", "precision_mode")),
        "partition_mode": _text(_pick(row, "partition_mode", "partition", "partition_strategy")),
        "tasklets_per_dpu": _integer(_pick(row, "tasklets_per_dpu", "tasklets", "tasklet_count")) or "unknown",
        "timing_scope": _text(_pick(row, "timing_scope")),
        "workload_kind": _text(_pick(row, "workload_kind", "workload_type", "quantum_case")),
        "scaling_kind": _scaling_kind(row),
        "dpu_count": dpu_count if dpu_count is not None else "unknown",
        "requested_dpu_count": requested_dpu_count if requested_dpu_count is not None else "unknown",
        "allocated_dpu_count": allocated_dpu_count if allocated_dpu_count is not None else "unknown",
        "status": status,
        "reason": _text(_top_pick(row, "reason", "failure_reason", "error", "failure_stage"), ""),
        "repeat_id": _pick(row, "repeat_id", "repetition", "rep"),
        "runtime_s": _runtime(row, valid_figure=physical_one_rank_valid),
        "accuracy": _quantization_accuracy(row) if physical_one_rank_valid and status in SUCCESS_STATUSES else None,
        "h2d_bytes": _repeat_metric(row, H2D_FIELDS, valid_figure=physical_one_rank_valid),
        "d2h_bytes": _repeat_metric(row, D2H_FIELDS, valid_figure=physical_one_rank_valid),
        "reduction_bytes": _repeat_metric(row, REDUCTION_FIELDS, valid_figure=physical_one_rank_valid),
        "run_h2d_bytes": _run_metric(row, H2D_FIELDS),
        "run_d2h_bytes": _run_metric(row, D2H_FIELDS),
        "run_reduction_bytes": _run_metric(row, REDUCTION_FIELDS),
        "load_balance": _load_balance(row, valid_figure=physical_one_rank_valid),
        "validation_status": _text(_pick(row, "validation_status"), "unknown"),
        "physical_one_rank_valid": physical_one_rank_valid,
        "quantization_evidence": _quantization_evidence(row),
        "partition_reduction_evidence": _partition_reduction_evidence(row),
        "pairing_identities": _pairing_identities(row),
        "binary_hashes": _binary_hashes(row),
        "one_rank": one_rank,
        "raw": row,
    }


def _sort_key(view: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(str(view.get(field, "")) for field in DIMENSION_FIELDS) + (
        str(view.get("repeat_id", "")),
        str(view.get("source_index", "")),
    )


def load_records(input_path: Path) -> tuple[list[dict[str, Any]], Path, str | None]:
    """Load and deterministically order normalized records.

    A missing or empty source is valid report input: it produces a report with
    explicit TODO placeholders rather than invented measurements.
    """

    source_file, source_run = _source_file(Path(input_path))
    if not source_file.is_file():
        return [], source_run, None
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(source_file.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ReportError(f"invalid JSON at {source_file}:{line_number}: {exc}") from exc
        if not isinstance(row, dict):
            raise ReportError(f"normalized record at {source_file}:{line_number} is not an object")
        rows.append(row)
    views = [_view(row, index) for index, row in enumerate(rows)]
    views.sort(key=_sort_key)
    return views, source_run, _sha256(source_file)


def _percentile(values: list[float], fraction: float) -> float:
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def _stats(values: Iterable[float]) -> dict[str, float | int | None]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {"median": None, "iqr": None, "min": None, "max": None, "repeat_count": 0}
    return {
        "median": _percentile(ordered, 0.5),
        "iqr": _percentile(ordered, 0.75) - _percentile(ordered, 0.25),
        "min": ordered[0],
        "max": ordered[-1],
        "repeat_count": len(ordered),
    }


def _group_key(view: Mapping[str, Any]) -> tuple[Any, ...]:
    return tuple(view.get(field) for field in DIMENSION_FIELDS)


def _group_views(views: Iterable[Mapping[str, Any]]) -> dict[tuple[Any, ...], list[Mapping[str, Any]]]:
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for view in views:
        groups.setdefault(_group_key(view), []).append(view)
    return groups


def _dimension_row(group: list[Mapping[str, Any]]) -> dict[str, Any]:
    first = group[0]
    return {field: first.get(field) for field in DIMENSION_FIELDS}


def metric_statistics(views: Iterable[Mapping[str, Any]], metric: str) -> list[dict[str, Any]]:
    """Return one deterministic statistics row per M5 dimension tuple."""

    result: list[dict[str, Any]] = []
    for key, group in sorted(_group_views(views).items(), key=lambda item: tuple(map(str, item[0]))):
        values = [float(view[metric]) for view in group if view.get(metric) is not None]
        stats = _stats(values)
        statuses = sorted({str(view.get("status", "unknown")) for view in group})
        measured_count = len(values)
        row = _dimension_row(group)
        row.update(
            {
                "status": "measured" if measured_count and measured_count == len(group) else (
                    "mixed" if measured_count else "unsupported_or_failed"
                ),
                "statuses": ";".join(statuses),
                "median": stats["median"],
                "iqr": stats["iqr"],
                "min": stats["min"],
                "max": stats["max"],
                "repeat_count": len(group),
                "measured_repeat_count": measured_count,
                "unsupported_or_failure_count": len(group) - measured_count,
            }
        )
        result.append(row)
    return result


def runtime_statistics(views: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return metric_statistics(views, "runtime_s")


def _strong_views(views: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [
        view
        for view in views
        if view.get("physical_one_rank_valid") is True
        and view.get("scaling_kind") == "strong_scaling"
        and view.get("runtime_s") is not None
        and float(view["runtime_s"]) > 0.0
        and isinstance(view.get("dpu_count"), int)
    ]


def _complete_compatibility_key(view: Mapping[str, Any]) -> bool:
    return all(
        view.get(field) not in {None, "", "unknown"}
        for field in COMPATIBILITY_FIELDS
    )


def strong_scaling_ratios(views: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Compute diagnostic T1/TN only for compatible physical evidence pairs."""

    eligible = [view for view in _strong_views(views) if _complete_compatibility_key(view)]
    stats = runtime_statistics(eligible)
    stats_by_key = {
        tuple(row[field] for field in DIMENSION_FIELDS): row
        for row in stats
        if row["median"] is not None
    }
    raw_groups: dict[tuple[Any, ...], dict[int, list[Mapping[str, Any]]]] = {}
    for view in eligible:
        base = tuple(view[field] for field in DIMENSION_FIELDS[:-1])
        count = view.get("dpu_count")
        if isinstance(count, int):
            raw_groups.setdefault(base, {}).setdefault(count, []).append(view)
    result: list[dict[str, Any]] = []
    for base, groups in sorted(raw_groups.items(), key=lambda item: tuple(map(str, item[0]))):
        baseline_views = groups.get(1, [])
        baseline = stats_by_key.get((*base, 1))
        if not baseline_views or baseline is None:
            continue
        for count in sorted(groups):
            if not isinstance(count, int) or count <= 1 or count == 0:
                continue
            target = stats_by_key.get((*base, count))
            target_views = groups[count]
            if target is None or not all(
                _pair_compatible(left, right) for left in baseline_views for right in target_views
            ):
                continue
            if float(baseline["median"]) <= 0.0 or float(target["median"]) <= 0.0:
                continue
            speedup = float(baseline["median"]) / float(target["median"])
            result.append(
                {
                    **{field: baseline[field] for field in DIMENSION_FIELDS[:-1]},
                    "baseline_dpu_count": 1,
                    "dpu_count": count,
                    "baseline_runtime_median_s": baseline["median"],
                    "runtime_median_s": target["median"],
                    "speedup": speedup,
                    "efficiency": speedup / count,
                    "baseline_repeat_count": baseline["measured_repeat_count"],
                    "repeat_count": target["measured_repeat_count"],
                    "pairing_rule": (
                        "same-route measured one-rank diagnostic; matching task/circuit/network/operation identities "
                        "and host/DPU binary hashes when present; execution-plan hash may differ by DPU count"
                    ),
                }
            )
    return result


def _numeric_family(value: Any) -> str | None:
    normalized = str(value).strip().lower().replace("-", "_")
    if normalized in {"float32", "float32_real", "real_float32", "none"}:
        return "float32"
    if "int8" in normalized or "requant" in normalized:
        return "int8_requantized"
    return None


def _partition_family(value: Any) -> str | None:
    normalized = str(value).strip().lower().replace("-", "_")
    if normalized in {"output", "output_tile", "output_partition"}:
        return "output"
    if normalized in {
        "contracted",
        "contracted_axis",
        "contracted_partial_sum",
        "contracted_partition",
    }:
        return "contracted"
    return None


def _comparison_views(views: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [
        view
        for view in views
        if view.get("physical_one_rank_valid") is True
        and view.get("status") in SUCCESS_STATUSES
        and isinstance(view.get("dpu_count"), int)
        and _complete_compatibility_key(view)
        and _float(view.get("runtime_s")) is not None
        and float(view["runtime_s"]) > 0.0
    ]


def numeric_mode_ratios(views: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return paired median ``T_float32 / T_int8`` comparisons."""

    base_fields = (
        "case_id",
        "route_id",
        "partition_mode",
        "tasklets_per_dpu",
        "timing_scope",
        "workload_kind",
        "scaling_kind",
        "dpu_count",
    )
    groups: dict[tuple[Any, ...], dict[str, list[Mapping[str, Any]]]] = {}
    for view in _comparison_views(views):
        family = _numeric_family(view.get("numeric_mode"))
        if family is None:
            continue
        key = tuple(view[field] for field in base_fields)
        groups.setdefault(key, {}).setdefault(family, []).append(view)

    result: list[dict[str, Any]] = []
    for key, variants in sorted(groups.items(), key=lambda item: tuple(map(str, item[0]))):
        float_views = variants.get("float32", [])
        int8_views = [
            view for view in variants.get("int8_requantized", [])
            if view.get("quantization_evidence") is True
        ]
        if not float_views or not int8_views or not all(
            _pair_compatible(left, right, ignored_identities=frozenset({"operation"}))
            for left in float_views
            for right in int8_views
        ):
            continue
        float_stats = _stats(float(view["runtime_s"]) for view in float_views)
        int8_stats = _stats(float(view["runtime_s"]) for view in int8_views)
        float_median = float(float_stats["median"])  # type: ignore[arg-type]
        int8_median = float(int8_stats["median"])  # type: ignore[arg-type]
        if int8_median <= 0.0:
            continue
        result.append(
            {
                **dict(zip(base_fields, key)),
                "float32_runtime_median_s": float_median,
                "int8_runtime_median_s": int8_median,
                "runtime_ratio_float32_over_int8": float_median / int8_median,
                "float32_repeat_count": float_stats["repeat_count"],
                "int8_repeat_count": int8_stats["repeat_count"],
                "pairing_rule": (
                    "same physical one-rank route/task/plan/path, partition, DPU count, tasklets, timing scope, "
                    "workload, scaling kind, and host/DPU binaries; operation bytes may differ only because the "
                    "numeric policy descriptor differs"
                ),
            }
        )
    return result


def partition_runtime_ratios(views: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return paired median ``T_output / T_contracted`` comparisons."""

    base_fields = (
        "case_id",
        "route_id",
        "numeric_mode",
        "tasklets_per_dpu",
        "timing_scope",
        "workload_kind",
        "scaling_kind",
        "dpu_count",
    )
    groups: dict[tuple[Any, ...], dict[str, list[Mapping[str, Any]]]] = {}
    for view in _comparison_views(views):
        family = _partition_family(view.get("partition_mode"))
        if family is None:
            continue
        key = tuple(view[field] for field in base_fields)
        groups.setdefault(key, {}).setdefault(family, []).append(view)

    result: list[dict[str, Any]] = []
    for key, variants in sorted(groups.items(), key=lambda item: tuple(map(str, item[0]))):
        output_views = variants.get("output", [])
        contracted_views = variants.get("contracted", [])
        if not output_views or not contracted_views or not all(
            view.get("partition_reduction_evidence") is True for view in contracted_views
        ) or not all(
            _pair_compatible(left, right)
            for left in output_views
            for right in contracted_views
        ):
            continue
        output_stats = _stats(float(view["runtime_s"]) for view in output_views)
        contracted_stats = _stats(float(view["runtime_s"]) for view in contracted_views)
        output_median = float(output_stats["median"])  # type: ignore[arg-type]
        contracted_median = float(contracted_stats["median"])  # type: ignore[arg-type]
        if contracted_median <= 0.0:
            continue
        result.append(
            {
                **dict(zip(base_fields, key)),
                "output_runtime_median_s": output_median,
                "contracted_runtime_median_s": contracted_median,
                "runtime_ratio_output_over_contracted": output_median / contracted_median,
                "output_repeat_count": output_stats["repeat_count"],
                "contracted_repeat_count": contracted_stats["repeat_count"],
                "pairing_rule": (
                    "same physical one-rank route/task/plan/path/operation, numeric mode, DPU count, tasklets, "
                    "timing scope, workload, scaling kind, and host/DPU binaries; only partition strategy differs"
                ),
            }
        )
    return result


def _write_csv(path: Path, rows: list[Mapping[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _record_rows(views: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for view in views:
        row = {
            **{field: view.get(field) for field in DIMENSION_FIELDS},
            "requested_dpu_count": view.get("requested_dpu_count"),
            "allocated_dpu_count": view.get("allocated_dpu_count"),
            "repeat_id": view.get("repeat_id"),
            "status": view.get("status"),
            "reason": view.get("reason"),
            "runtime_s": view.get("runtime_s"),
            "accuracy_max_abs_error": view.get("accuracy"),
            "h2d_bytes": view.get("h2d_bytes"),
            "d2h_bytes": view.get("d2h_bytes"),
            "host_mediated_reduction_bytes": view.get("reduction_bytes"),
            "run_h2d_bytes_provenance": view.get("run_h2d_bytes"),
            "run_d2h_bytes_provenance": view.get("run_d2h_bytes"),
            "run_reduction_bytes_provenance": view.get("run_reduction_bytes"),
            "load_balance_ratio": view.get("load_balance"),
            "one_rank": view.get("one_rank"),
            "physical_one_rank_valid": view.get("physical_one_rank_valid"),
            "validation_status": view.get("validation_status"),
            "quantization_evidence": view.get("quantization_evidence"),
        }
        rows.append({field: row.get(field) for field in M5_RECORD_FIELDS})
    return rows


_LABEL_FIELDS = ("numeric_mode", "partition_mode", "tasklets_per_dpu")


def _label_value(field: str, value: Any) -> str:
    text = str(value)
    if field == "case_id":
        return textwrap.shorten(text.replace("_", " "), width=28, placeholder="...")
    if field == "numeric_mode":
        return {"per_task_resident_requantize": "int8", "none": "float32"}.get(text, text)
    if field == "partition_mode":
        return {
            "output_tile": "output",
            "contracted_partial_sum": "contracted",
        }.get(text, text)
    if field == "tasklets_per_dpu":
        return f"{text} tasklets"
    if field == "dpu_count":
        return f"{text} DPU" if text == "1" else f"{text} DPUs"
    return text


def _varied_label_fields(records: Iterable[Mapping[str, Any]], fields: Iterable[str]) -> tuple[str, ...]:
    rows = list(records)
    return tuple(
        field
        for field in fields
        if len({str(row.get(field, "")) for row in rows}) > 1
    )


def _concise_label(row: Mapping[str, Any], varied_fields: Iterable[str]) -> str:
    parts = [_label_value("case_id", row.get("case_id", "unknown case"))]
    for field in varied_fields:
        parts.append(_label_value(field, row.get(field, "unknown")))
    return textwrap.shorten(" / ".join(parts), width=48, placeholder="...")


def _key_record(key: tuple[Any, ...]) -> dict[str, Any]:
    return dict(zip(DIMENSION_FIELDS, key))


def _labels_for_keys(keys: Iterable[tuple[Any, ...]], *, include_dpu: bool = False) -> dict[tuple[Any, ...], str]:
    key_list = list(keys)
    fields = (*_LABEL_FIELDS, "dpu_count") if include_dpu else _LABEL_FIELDS
    varied = _varied_label_fields((_key_record(key) for key in key_list), fields)
    return {key: _concise_label(_key_record(key), varied) for key in key_list}


def _plot_points(views: Iterable[Mapping[str, Any]], metric: str, *, weak: bool = False) -> dict[str, list[tuple[int, float]]]:
    selected = [
        view
        for view in views
        if view.get("physical_one_rank_valid") is True
        and bool(view.get("runtime_s") is not None)
        and bool(view.get(metric) is not None)
        and view.get("scaling_kind") == ("weak_scaling" if weak else "strong_scaling")
        and isinstance(view.get("dpu_count"), int)
    ]
    grouped = _group_views(selected)
    labels = _labels_for_keys(grouped)
    groups: dict[str, list[tuple[int, float]]] = {}
    for key, group in grouped.items():
        values = [float(view[metric]) for view in group]
        dpu_count = key[-1]
        label = labels[key]
        groups.setdefault(label, []).append((int(dpu_count), _stats(values)["median"]))  # type: ignore[arg-type]
    for values in groups.values():
        values.sort(key=lambda item: item[0])
    return groups


def _plot(
    path: Path,
    title: str,
    caption: str,
    groups: Mapping[str, list[tuple[int, float]]],
    ylabel: str,
    *,
    empty_status: str = "todo_missing_data",
    reference_line: float | None = None,
    y_scale: str = "linear",
) -> str:
    if y_scale not in {"linear", "log"}:
        raise ValueError(f"unsupported y scale: {y_scale}")
    plotted_groups = {
        label: [point for point in points if y_scale != "log" or point[1] > 0.0]
        for label, points in groups.items()
    }
    plotted_groups = {label: points for label, points in plotted_groups.items() if points}
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return _fallback_plot(
            path,
            title,
            caption,
            plotted_groups,
            ylabel,
            bars=False,
            empty_status=empty_status,
            reference_line=reference_line,
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(14.0, 6.0))
    if plotted_groups:
        for label in sorted(plotted_groups):
            points = plotted_groups[label]
            axis.plot([point[0] for point in points], [point[1] for point in points], marker="o", label=label)
        if reference_line is not None:
            axis.axhline(reference_line, color="black", linestyle="--", linewidth=1.0)
        axis.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
        axis.set_xlabel("DPU count")
        axis.set_ylabel(ylabel)
        axis.set_yscale(y_scale)
        axis.grid(True, alpha=0.25)
    else:
        axis.text(0.5, 0.5, "TODO: no measured data available", ha="center", va="center", transform=axis.transAxes)
        axis.set_axis_off()
    axis.set_title(title)
    figure.text(
        0.01,
        0.02,
        textwrap.fill(caption, width=140),
        ha="left",
        va="bottom",
        fontsize=8,
    )
    figure.subplots_adjust(left=0.08, right=0.58, bottom=0.24, top=0.84)
    figure.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(figure)
    return "generated" if plotted_groups else empty_status


def _bar_plot(
    path: Path,
    title: str,
    caption: str,
    groups: Mapping[str, Mapping[str, float]],
    ylabel: str,
    *,
    empty_status: str = "todo_missing_data",
) -> str:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return _fallback_plot(
            path,
            title,
            caption,
            groups,
            ylabel,
            bars=True,
            empty_status=empty_status,
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(12.0, 8.0))
    labels = sorted(groups)
    categories = sorted({category for values in groups.values() for category in values})
    if labels and categories:
        width = 0.8 / len(categories)
        positions = list(range(len(labels)))
        for index, category in enumerate(categories):
            axis.bar(
                [position + index * width for position in positions],
                [groups[label].get(category, 0.0) for label in labels],
                width=width,
                label=category,
            )
        axis.set_xticks(
            [position + width * (len(categories) - 1) / 2 for position in positions],
            labels,
            rotation=45,
            ha="right",
            fontsize=8,
        )
        axis.set_ylabel(ylabel)
        axis.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
        axis.grid(True, axis="y", alpha=0.25)
    else:
        axis.text(0.5, 0.5, "TODO: no measured data available", ha="center", va="center", transform=axis.transAxes)
        axis.set_axis_off()
    axis.set_title(title)
    figure.text(
        0.01,
        0.01,
        textwrap.fill(caption, width=145),
        ha="left",
        va="bottom",
        fontsize=8,
    )
    figure.subplots_adjust(left=0.09, right=0.72, bottom=0.44, top=0.84)
    figure.savefig(path, dpi=120)
    plt.close(figure)
    return "generated" if labels and categories else empty_status


def _heatmap_series(
    views: Iterable[Mapping[str, Any]],
    metrics: Mapping[str, str],
) -> dict[str, dict[str, dict[int, float]]]:
    """Group measured values as panel -> row label -> DPU count -> median."""

    selected = [
        view
        for view in views
        if view.get("physical_one_rank_valid") is True
        and view.get("status") in SUCCESS_STATUSES
        and isinstance(view.get("dpu_count"), int)
    ]
    grouped = _group_views(selected)
    labels = _labels_for_keys(grouped)
    panels: dict[str, dict[str, dict[int, float]]] = {name: {} for name in metrics}
    for key, group in grouped.items():
        dpu_count = int(key[-1])
        row_label = labels[key]
        for panel, metric in metrics.items():
            values = [float(view[metric]) for view in group if view.get(metric) is not None]
            if values:
                panels[panel].setdefault(row_label, {})[dpu_count] = float(_stats(values)["median"])
    return {panel: rows for panel, rows in panels.items() if rows}


def _heatmap_plot(
    path: Path,
    title: str,
    caption: str,
    panels: Mapping[str, Mapping[str, Mapping[int, float]]],
    *,
    empty_status: str = "todo_missing_data",
) -> str:
    """Render bounded case rows against DPU-count columns."""

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ModuleNotFoundError:
        return _fallback_plot(
            path,
            title,
            caption,
            {},
            "DPU count",
            bars=True,
            empty_status=empty_status,
        )

    if not panels:
        return _fallback_plot(
            path,
            title,
            caption,
            {},
            "DPU count",
            bars=True,
            empty_status=empty_status,
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    row_labels = sorted({label for rows in panels.values() for label in rows})
    dpu_counts = sorted({count for rows in panels.values() for values in rows.values() for count in values})
    figure_height = max(5.5, 0.34 * len(row_labels) + 2.4)
    figure_width = max(10.0, 5.8 * len(panels))
    figure, axes = plt.subplots(
        1,
        len(panels),
        figsize=(figure_width, figure_height),
        squeeze=False,
        sharey=True,
    )
    axes_list = list(axes[0])
    for index, (panel, rows) in enumerate(sorted(panels.items())):
        matrix = np.full((len(row_labels), len(dpu_counts)), np.nan, dtype=float)
        for row_index, row_label in enumerate(row_labels):
            for column_index, dpu_count in enumerate(dpu_counts):
                value = rows.get(row_label, {}).get(dpu_count)
                if value is not None:
                    matrix[row_index, column_index] = value
        axis = axes_list[index]
        image = axis.imshow(matrix, aspect="auto", interpolation="nearest")
        axis.set_title(panel)
        axis.set_xlabel("DPU count")
        axis.set_xticks(range(len(dpu_counts)), [str(count) for count in dpu_counts])
        axis.tick_params(axis="x", labelrotation=0)
        axis.set_yticks(range(len(row_labels)))
        if index == 0:
            axis.set_yticklabels(row_labels, fontsize=7)
            axis.set_ylabel("Case / comparison series")
        else:
            axis.tick_params(axis="y", labelleft=False)
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.suptitle(title)
    figure.text(0.01, 0.01, textwrap.fill(caption, width=120), ha="left", va="bottom", fontsize=8)
    figure.subplots_adjust(left=0.30, right=0.88, bottom=0.20, top=0.88, wspace=0.34)
    figure.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(figure)
    return "generated"


def _fallback_plot(
    path: Path,
    title: str,
    caption: str,
    groups: Mapping[str, Any],
    ylabel: str,
    *,
    bars: bool,
    empty_status: str,
    reference_line: float | None = None,
) -> str:
    """Small Pillow renderer used when optional matplotlib is unavailable."""

    from PIL import Image, ImageDraw

    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (900, 540), "white")
    draw = ImageDraw.Draw(image)
    draw.text((24, 18), title, fill="black")
    draw.text((24, 505), caption[:150], fill="black")
    if not groups:
        draw.text((330, 260), "TODO: no measured data available", fill="black")
        image.save(path, format="PNG")
        return empty_status
    left, top, right, bottom = 80, 80, 860, 455
    draw.line((left, bottom, right, bottom), fill="black", width=2)
    draw.line((left, top, left, bottom), fill="black", width=2)
    draw.text((left, top - 20), ylabel[:80], fill="black")
    if bars:
        labels = sorted(groups)
        categories = sorted({category for values in groups.values() for category in values})
        all_values = [float(value) for values in groups.values() for value in values.values()]
        maximum = max(all_values) if all_values else 1.0
        width = max(2, (right - left) // max(1, len(labels) * max(1, len(categories))))
        for label_index, label in enumerate(labels):
            for category_index, category in enumerate(categories):
                value = float(groups[label].get(category, 0.0))
                x = left + (label_index * len(categories) + category_index) * width
                height = int((bottom - top - 10) * value / maximum) if maximum else 0
                draw.rectangle((x, bottom - height, x + width - 2, bottom), fill=(40, 100, 170))
    else:
        points = [point for values in groups.values() for point in values]
        x_values = [float(point[0]) for point in points]
        y_values = [float(point[1]) for point in points]
        if reference_line is not None:
            y_values.append(reference_line)
        x_min, x_max = min(x_values), max(x_values)
        y_min, y_max = min(y_values), max(y_values)
        if reference_line is not None:
            reference_y = bottom if y_max == y_min else bottom - int(
                (reference_line - y_min) * (bottom - top) / (y_max - y_min)
            )
            draw.line((left, reference_y, right, reference_y), fill="black", width=1)
        for values in groups.values():
            previous: tuple[int, int] | None = None
            for x_value, y_value in values:
                x = left if x_max == x_min else left + int((x_value - x_min) * (right - left) / (x_max - x_min))
                y = bottom if y_max == y_min else bottom - int((y_value - y_min) * (bottom - top) / (y_max - y_min))
                if previous is not None:
                    draw.line((*previous, x, y), fill=(40, 100, 170), width=3)
                draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=(40, 100, 170))
                previous = (x, y)
    image.save(path, format="PNG")
    return "generated"


def _table_groups(views: Iterable[Mapping[str, Any]], metric: str) -> dict[str, dict[str, float]]:
    selected = [view for view in views if view.get("physical_one_rank_valid") is True and view.get(metric) is not None]
    result: dict[str, dict[str, float]] = {}
    grouped = _group_views(selected)
    labels = _labels_for_keys(grouped, include_dpu=True)
    for key, group in grouped.items():
        label = labels[key]
        result[label] = {metric: float(_stats(float(view[metric]) for view in group)["median"])}  # type: ignore[arg-type]
    return result


def _artifact_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _summary(
    *,
    source_run: Path,
    source_hash: str | None,
    views: list[Mapping[str, Any]],
    plot_manifest: Mapping[str, Any],
    output: Path,
) -> str:
    supported = sorted(
        {
            int(view["dpu_count"])
            for view in views
            if view.get("physical_one_rank_valid") is True
            and view.get("status") in SUCCESS_STATUSES
            and isinstance(view.get("dpu_count"), int)
        }
    )
    failed = sorted({int(view["dpu_count"]) for view in views if view.get("status") not in SUCCESS_STATUSES and isinstance(view.get("dpu_count"), int)})
    measured_count = sum(1 for view in views if view.get("runtime_s") is not None)
    todo = measured_count == 0
    supported_text = ", ".join(map(str, supported)) if supported else "TODO: none observed"
    failed_text = ", ".join(map(str, failed)) if failed else "none observed"
    plot_lines = "\n".join(f"- `{entry['path']}`: {entry['status']}" for entry in plot_manifest["plots"])
    return (
        "# M5 UPMEM report\n\n"
        "This report is a deterministic descriptive view of normalized M5 evidence.\n\n"
        f"- Source run: `{source_run}`\n"
        f"- Source `normalized_records.jsonl` SHA-256: `{source_hash or 'TODO: source absent'}`\n"
        f"- Source rows: **{len(views)}**; measured runtime rows: **{measured_count}**\n"
        f"- Supported DPU counts: **{supported_text}**\n"
        f"- Failed or unsupported DPU counts: **{failed_text}**\n"
        f"- Report directory: `{output}`\n\n"
        + ("**TODO: no measured data was available; numeric fields and ratios are intentionally absent.**\n\n" if todo else "")
        + "## Plot status\n\n"
        + plot_lines
        + "\n\n## Allowed claims\n\n"
        "- Per-case, route, numeric-mode, partition, and DPU-count measured summaries from the supplied rows.\n"
        "- Within-key strong-scaling ratios using `T1/TN` and efficiency `speedup/N` when a measured one-rank DPU-count-1 baseline exists.\n"
        "- Dedicated same-route `T_float32/T_int8` and `T_output/T_contracted` ratios when every non-varied execution and identity field is compatible.\n"
        "- Host-observed transfer, load-balance, and accuracy observations where those fields are present.\n\n"
        "## Not allowed\n\n"
        "- Cross-route or otherwise incompatible pairing; numeric mode and partition may differ only in their dedicated controlled comparisons.\n"
        "- PID-Comm, multi-rank, packed-int8, distributed TaskGraph, energy, or general hardware-performance claims.\n"
        "- Filling missing, unsupported, or failed measurements with estimates or fake values.\n\n"
        "The plot captions identify physical one-rank measured evidence. Host-mediated reduction is stated where applicable; quantization plots identify on-DPU int8 requantization with float32 MRAM transport.\n"
    )


def generate_report(input_path: Path, output_root: Path, *, timestamp: str | None = None) -> Path:
    """Generate an M5 comparison directory and return its path."""

    views, source_run, source_hash = load_records(Path(input_path))
    root = Path(output_root).expanduser().resolve()
    comparison_root = root / "runs" / "comparisons" / "upmem_m5"
    if comparison_root.is_relative_to(source_run.resolve()):
        raise ReportError("report output must not be inside the source evidence run")
    comparison_root.mkdir(parents=True, exist_ok=True)
    stamp = timestamp if timestamp is not None else datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S_%f")
    if not isinstance(stamp, str) or TIMESTAMP_PATTERN.fullmatch(stamp) is None:
        raise ReportError("timestamp must be a single safe path component")
    output = comparison_root / stamp
    suffix = 1
    while output.exists():
        output = comparison_root / f"{stamp}_{suffix}"
        suffix += 1
    output.mkdir()
    tables = output / "tables"
    plots = output / "plots"

    records = _record_rows(views)
    runtime_rows = runtime_statistics(views)
    accuracy_rows = metric_statistics(views, "accuracy")
    transfer_rows = metric_statistics(views, "h2d_bytes")
    ratios = strong_scaling_ratios(views)
    numeric_ratios = numeric_mode_ratios(views)
    partition_ratios = partition_runtime_ratios(views)
    _write_csv(tables / "m5_records.csv", records, list(M5_RECORD_FIELDS))
    _write_csv(tables / "m5_runtime_statistics.csv", runtime_rows, STAT_FIELDS)
    _write_csv(tables / "m5_accuracy_statistics.csv", accuracy_rows, STAT_FIELDS)
    _write_csv(tables / "m5_transfer_statistics.csv", transfer_rows, STAT_FIELDS)
    _write_csv(tables / "m5_strong_scaling.csv", ratios, RATIO_FIELDS)
    _write_csv(
        tables / "m5_numeric_mode_ratios.csv",
        numeric_ratios,
        NUMERIC_RATIO_FIELDS,
    )
    _write_csv(
        tables / "m5_partition_ratios.csv",
        partition_ratios,
        PARTITION_RATIO_FIELDS,
    )
    table_paths = [
        f"tables/{name}"
        for name in (
            "m5_records.csv",
            "m5_runtime_statistics.csv",
            "m5_accuracy_statistics.csv",
            "m5_transfer_statistics.csv",
            "m5_strong_scaling.csv",
            "m5_numeric_mode_ratios.csv",
            "m5_partition_ratios.csv",
        )
    ]
    table_sha256 = {
        relative_path: _artifact_hash(output / relative_path)
        for relative_path in table_paths
    }

    common = (
        "Same-route measured one-rank physical diagnostics only; no CPU/GPU speedup or multi-rank claim. "
        "Host-mediated reduction where applicable."
    )
    has_int8_evidence = any(
        view.get("physical_one_rank_valid") is True
        and view.get("accuracy") is not None
        and view.get("quantization_evidence") is True
        for view in views
    )
    quant = (
        "Same-route measured one-rank physical diagnostics only; no CPU/GPU speedup or multi-rank claim. "
        "Quantization error versus float32 from canonical quantization/full-precision evidence; "
        "policy-reference validation is separate. on-DPU int8 requantization with float32 MRAM transport."
        if has_int8_evidence
        else common + " Quantization error versus float32 requires canonical quantization/full-precision evidence; policy-reference validation is separate."
    )
    plot_specs: list[tuple[str, str, str, Mapping[str, list[tuple[int, float]]], str, str]] = [
        ("m5_strong_scaling_runtime.png", "M5 strong-scaling runtime (physical one-rank, measured)", common, _plot_points(views, "runtime_s"), "Runtime (s), median", "log"),
        ("m5_strong_scaling_speedup.png", "M5 strong-scaling speedup T1/TN (physical one-rank, measured)", common, _ratio_points(ratios, "speedup"), "Speedup T1/TN", "linear"),
        ("m5_strong_scaling_efficiency.png", "M5 strong-scaling efficiency speedup/N (physical one-rank, measured)", common, _ratio_points(ratios, "efficiency"), "Efficiency speedup/N", "linear"),
        ("m5_weak_scaling_runtime.png", "M5 weak-scaling runtime (physical one-rank, measured)", common, _plot_points(views, "runtime_s", weak=True), "Runtime (s), median", "log"),
    ]
    plot_entries: list[dict[str, Any]] = []
    for name, title, caption, groups, ylabel, y_scale in plot_specs:
        status = _plot(plots / name, title, caption, groups, ylabel, y_scale=y_scale)
        plot_entries.append({"path": f"plots/{name}", "status": status, "title": title, "caption": caption, "metric": ylabel})

    numeric_ratio_caption = (
        "Same-route measured one-rank physical comparison. Ratio = T_float32/T_int8; values above 1 favour "
        "per-task on-DPU int8 requantized arithmetic. Both modes use float32 MRAM transport; this is not a "
        "packed-int8 transfer comparison."
        if numeric_ratios
        else "TODO: canonical int8 quantization and float32 MRAM transport evidence are required; no numeric ratio is formed."
    )
    numeric_ratio_title = "M5 float32/int8 runtime ratio (physical one-rank, measured)"
    status = _plot(
        plots / "m5_numeric_mode_runtime_ratio.png",
        numeric_ratio_title,
        numeric_ratio_caption,
        _comparison_ratio_points(
            numeric_ratios,
            "runtime_ratio_float32_over_int8",
            (
                "case_id",
                "route_id",
                "partition_mode",
                "tasklets_per_dpu",
                "timing_scope",
                "workload_kind",
                "scaling_kind",
            ),
        ),
        "Runtime ratio T_float32/T_int8",
        reference_line=1.0,
    )
    plot_entries.append(
        {
            "path": "plots/m5_numeric_mode_runtime_ratio.png",
            "status": status,
            "title": numeric_ratio_title,
            "caption": numeric_ratio_caption,
            "metric": "runtime_ratio_float32_over_int8",
        }
    )

    partition_ratio_caption = (
        "Same-route measured one-rank physical comparison. Ratio = T_output/T_contracted; values above 1 "
        "favour contracted-axis partitioning. The contracted route includes host-mediated float64 reduction. "
        "Partition strategy is not a contraction path."
        if partition_ratios
        else "TODO: canonical host-mediated provider and float64 reduction evidence are required; no partition ratio is formed."
    )
    partition_ratio_title = "M5 output/contracted partition runtime ratio (physical one-rank, measured)"
    status = _plot(
        plots / "m5_partition_runtime_ratio.png",
        partition_ratio_title,
        partition_ratio_caption,
        _comparison_ratio_points(
            partition_ratios,
            "runtime_ratio_output_over_contracted",
            (
                "case_id",
                "route_id",
                "numeric_mode",
                "tasklets_per_dpu",
                "timing_scope",
                "workload_kind",
                "scaling_kind",
            ),
        ),
        "Runtime ratio T_output/T_contracted",
        reference_line=1.0,
    )
    plot_entries.append(
        {
            "path": "plots/m5_partition_runtime_ratio.png",
            "status": status,
            "title": partition_ratio_title,
            "caption": partition_ratio_caption,
            "metric": "runtime_ratio_output_over_contracted",
        }
    )

    transfer_groups = _heatmap_series(
        views,
        {
            "H2D bytes": "h2d_bytes",
            "D2H bytes": "d2h_bytes",
            "Host reduction bytes": "reduction_bytes",
        },
    )
    transfer_caption = common
    status = _heatmap_plot(plots / "m5_transfer_breakdown.png", "M5 transfer breakdown (physical one-rank, measured)", transfer_caption, transfer_groups)
    plot_entries.append({"path": "plots/m5_transfer_breakdown.png", "status": status, "title": "M5 transfer breakdown (physical one-rank, measured)", "caption": transfer_caption, "metric": "H2D/D2H/host reduction bytes"})

    balance_groups = _heatmap_series(views, {"Load-balance ratio": "load_balance"})
    status = _heatmap_plot(plots / "m5_load_balance.png", "M5 load balance (physical one-rank, measured)", common, balance_groups)
    plot_entries.append({"path": "plots/m5_load_balance.png", "status": status, "title": "M5 load balance (physical one-rank, measured)", "caption": common, "metric": "load_balance"})

    accuracy_groups = _heatmap_series(views, {"Maximum absolute error": "accuracy"})
    accuracy_title = "M5 quantization error versus float32 (physical one-rank, measured)"
    status = _heatmap_plot(plots / "m5_quantization_accuracy.png", accuracy_title, quant, accuracy_groups)
    plot_entries.append({"path": "plots/m5_quantization_accuracy.png", "status": status, "title": accuracy_title, "caption": quant, "metric": "accuracy"})

    for entry in plot_entries:
        entry["sha256"] = _artifact_hash(output / entry["path"])
    plot_statuses = {entry["status"] for entry in plot_entries}
    manifest = {
        "schema_version": "upmem_m5_report_v1",
        "status": "todo_missing_data" if not views or not any(entry["status"] == "generated" for entry in plot_entries) else ("partial_missing_data" if "todo_missing_data" in plot_statuses else "complete"),
        "source_run": str(source_run),
        "source_normalized_records": str(_source_file(Path(input_path))[0]),
        "source_sha256": source_hash,
        "supported_dpu_counts": sorted(
            {
                view["dpu_count"]
                for view in views
                if view.get("physical_one_rank_valid") is True
                and view.get("status") in SUCCESS_STATUSES
                and isinstance(view.get("dpu_count"), int)
            }
        ),
        "failed_or_unsupported_dpu_counts": sorted({view["dpu_count"] for view in views if view.get("status") not in SUCCESS_STATUSES and isinstance(view.get("dpu_count"), int)}),
        "plots": plot_entries,
        "tables": table_paths,
        "table_sha256": table_sha256,
        "claims": {
            "physical_one_rank_measured": any(
                view.get("physical_one_rank_valid") is True and view.get("runtime_s") is not None
                for view in views
            ),
            "host_mediated_reduction": "where recorded",
            "speedup": bool(ratios),
            "numeric_mode_ratio": bool(numeric_ratios),
            "partition_runtime_ratio": bool(partition_ratios),
            "cross_route_pairing": False,
        },
    }
    (output / "plot_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "m5_summary.md").write_text(_summary(source_run=source_run, source_hash=source_hash, views=views, plot_manifest=manifest, output=output), encoding="utf-8")
    return output


def _ratio_points(rows: Iterable[Mapping[str, Any]], metric: str) -> dict[str, list[tuple[int, float]]]:
    rows = list(rows)
    varied = _varied_label_fields(rows, _LABEL_FIELDS)
    groups: dict[str, list[tuple[int, float]]] = {}
    for row in rows:
        value = _float(row.get(metric))
        count = _integer(row.get("dpu_count"))
        if value is None or count is None:
            continue
        label = _concise_label(row, varied)
        groups.setdefault(label, []).append((count, value))
    for values in groups.values():
        values.sort(key=lambda item: item[0])
    return groups


def _comparison_ratio_points(
    rows: Iterable[Mapping[str, Any]],
    metric: str,
    label_fields: tuple[str, ...],
) -> dict[str, list[tuple[int, float]]]:
    rows = list(rows)
    allowed_fields = tuple(field for field in label_fields if field in _LABEL_FIELDS)
    varied = _varied_label_fields(rows, allowed_fields)
    groups: dict[str, list[tuple[int, float]]] = {}
    for row in rows:
        value = _float(row.get(metric))
        count = _integer(row.get("dpu_count"))
        if value is None or count is None:
            continue
        label = _concise_label(row, varied)
        groups.setdefault(label, []).append((count, value))
    for values in groups.values():
        values.sort(key=lambda item: item[0])
    return groups


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="normalized_records.jsonl or its evidence run directory")
    parser.add_argument("--output-root", type=Path, default=Path(__file__).resolve().parents[1], help="repository root for runs/comparisons/upmem_m5")
    args = parser.parse_args(argv)
    try:
        output = generate_report(args.input, args.output_root)
    except (OSError, ReportError, json.JSONDecodeError) as exc:
        print(f"M5 report failed: {exc}", file=sys.stderr)
        return 2
    print(f"comparison_dir={output}")
    print(f"summary={output / 'm5_summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
