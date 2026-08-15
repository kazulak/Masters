"""Focused whole-circuit M5.5 report generation.

This module deliberately has no benchmark execution dependencies.  It consumes
normalized records and emits compact tables and figures for a single report
directory.  Cross-algorithm rows are retained for context but never enter
same-plan CPU/UPMEM ratios.
"""

from __future__ import annotations

import csv
import json
import math
import re
import statistics
import textwrap
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from quantum_bench.evidence import Claim, ClaimDecision, ClaimPolicy


JsonDict = dict[str, Any]

_CLAIM_POLICY = ClaimPolicy()

_MAX_TIMING_ROUTE_PANELS = 8
_TIMING_STAGE_ORDER = (
    "planning",
    "session_open",
    "host_quantization",
    "h2d",
    "kernel",
    "d2h",
    "assembly",
    "host_dequantization",
    "validation",
    "session_close",
)
_TIMING_STAGE_LABELS = {
    "planning": "Planning",
    "session_open": "Session open",
    "host_quantization": "Host quantization",
    "h2d": "H2D",
    "kernel": "Kernel",
    "d2h": "D2H",
    "assembly": "Assembly",
    "host_dequantization": "Host dequantization",
    "validation": "Validation",
    "session_close": "Session close",
}
_PLOT_MARKERS = ("o", "s", "^", "D", "P", "X", "v", "<", ">", "h")


@dataclass(frozen=True)
class PlotEntry:
    name: str
    path: str
    source_csv: str
    status: str
    title: str
    reason: str | None = None


@dataclass(frozen=True)
class PlotManifest:
    schema_version: str
    plots: tuple[PlotEntry, ...]

    def to_dict(self) -> JsonDict:
        return {
            "schema_version": self.schema_version,
            "plots": [asdict(entry) for entry in self.plots],
            "generated_valid": [
                entry.path for entry in self.plots if entry.status == "generated_valid"
            ],
            "generated_todo_missing_data": [
                entry.path
                for entry in self.plots
                if entry.status == "generated_todo_missing_data"
            ],
        }


@dataclass(frozen=True)
class ReportResult:
    output_dir: Path
    manifest: PlotManifest


def load_records(source: Path | str | Iterable[Mapping[str, Any]]) -> list[JsonDict]:
    """Load JSONL records, or normalize an iterable of mapping records."""
    if isinstance(source, (str, Path)):
        path = Path(source)
        rows: list[JsonDict] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError("normalized JSONL records must be objects")
                rows.append(dict(value))
        return rows
    return [dict(row) for row in source]


def _first(row: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        value = row.get(key)
        if value is not None and value != "":
            return value
    return None


def _nested(row: Mapping[str, Any], *paths: tuple[str, ...]) -> Any:
    for path in paths:
        value: Any = row
        for key in path:
            if not isinstance(value, Mapping):
                value = None
                break
            value = value.get(key)
        if value is not None:
            return value
    return None


def _float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _runtime(row: Mapping[str, Any]) -> float | None:
    value = _first(
        row,
        "total_route_time_s",
        "total_wall_time_s",
        "timing_s",
        "runtime_s",
        "execution_time_s",
    )
    if value is None:
        value = _first(row, "steady_state_graph_execution_s", "route_time_s")
    if value is None:
        value = _nested(
            row,
            ("timing", "total_time_s"),
            ("timing", "total_s"),
            ("timings", "total_time_s"),
        )
    result = _float(value)
    return result if result is not None and result > 0 else None


def _family(row: Mapping[str, Any]) -> str:
    return str(
        _first(
            row,
            "circuit_family",
            "quantum_case",
            "family",
            "workload_kind",
            "workload_id",
            "case_id",
        )
        or "unknown"
    )


def _case(row: Mapping[str, Any]) -> str:
    return str(_first(row, "case_id", "workload_id", "quantum_case") or "")


def _engine(row: Mapping[str, Any]) -> str:
    return str(
        _first(
            row,
            "engine_id",
            "execution_engine",
            "engine",
            "backend_family",
            "backend_id",
            "route_id",
        )
        or "unknown"
    )


def _engine_class(row: Mapping[str, Any]) -> str:
    value = _engine(row).lower()
    if "quimb" in value or "quest" in value:
        return "cross_algorithm"
    if "upmem" in value or "dpu" in value:
        return "upmem"
    if "cpu" in value or "numpy" in value or _first(row, "execution_target") == "cpu":
        return "cpu"
    return "other"


def _path(row: Mapping[str, Any]) -> str:
    return str(
        _first(
            row,
            "path_variant_id",
            "planner_id",
            "path_strategy",
            "path_id",
            "contraction_path",
        )
        or "unspecified"
    )


def _numeric(row: Mapping[str, Any]) -> str:
    return str(
        _first(
            row,
            "numeric_policy",
            "numeric_mode",
            "quantization_mode",
            "numeric_arithmetic",
        )
        or "unspecified"
    )


def _timing_scope(row: Mapping[str, Any]) -> str:
    return str(_first(row, "timing_scope", "timing_contract") or "")


def _repeat(row: Mapping[str, Any]) -> str:
    return str(_first(row, "repeat_id", "repetition", "measurement_id") or "0")


def _qubits(row: Mapping[str, Any]) -> int | None:
    return _int(_first(row, "qubits", "num_qubits", "n_qubits", "size"))


def _rank_count(row: Mapping[str, Any]) -> int | None:
    value = _int(
        _first(row, "rank_count", "observed_rank_count", "requested_rank_count")
    )
    return value if value is not None and value > 0 else None


def _engine_metadata(row: Mapping[str, Any]) -> Mapping[str, Any]:
    value = row.get("engine_metadata")
    return value if isinstance(value, Mapping) else {}


def _active_id_count(metadata: Mapping[str, Any], *keys: str) -> int | None:
    for key in keys:
        value = metadata.get(key)
        if isinstance(value, (list, tuple, set, frozenset)):
            return len({str(item) for item in value})
        count = _int(value)
        if count is not None and count >= 0:
            return count
    return None


def _active_dpu_count(row: Mapping[str, Any]) -> int | None:
    metadata = _engine_metadata(row)
    return _active_id_count(
        metadata,
        "active_dpu_ids",
        "active_dpu_count",
        "observed_dpu_count",
    )


def _active_rank_count(row: Mapping[str, Any]) -> int | None:
    metadata = _engine_metadata(row)
    return _active_id_count(
        metadata,
        "active_rank_indices",
        "active_rank_ids",
        "active_ranks",
        "active_rank_count",
        "observed_rank_count",
    )


def _local_dpu_count(row: Mapping[str, Any]) -> int | None:
    explicit = _int(
        _first(
            row,
            "local_dpu_count",
            "dpus_per_rank",
            "requested_dpus_per_rank",
            "allocated_dpus_per_rank",
        )
    )
    if explicit is not None and explicit > 0:
        return explicit
    ranks = _rank_count(row)
    provisioned_total = _int(_first(row, "allocated_dpu_count", "requested_dpu_count"))
    if ranks and provisioned_total and provisioned_total > 0:
        if provisioned_total % ranks == 0:
            return provisioned_total // ranks
    if ranks in {None, 1} and provisioned_total is not None and provisioned_total > 0:
        return provisioned_total
    total = _int(_first(row, "total_dpu_count", "allocated_total_dpu_count"))
    if ranks and total and total % ranks == 0:
        return total // ranks
    return None


def _total_dpu_count(row: Mapping[str, Any]) -> int | None:
    explicit = _int(
        _first(
            row,
            "total_dpu_count",
            "allocated_total_dpu_count",
            "allocated_dpu_count",
            "requested_dpu_count",
        )
    )
    if explicit is not None and explicit > 0:
        return explicit
    local = _local_dpu_count(row)
    ranks = _rank_count(row)
    if local is not None and ranks is not None:
        return local * ranks
    return local


def _topology_label(row: Mapping[str, Any]) -> str:
    local = _local_dpu_count(row)
    ranks = _rank_count(row)
    total = _total_dpu_count(row)
    return f"{local or '?'}DPU x {ranks or '?'} rank(s) = {total or '?'} total"


def _activity_label(row: Mapping[str, Any]) -> str | None:
    """Return a compact label that distinguishes active from provisioned devices."""
    provisioned_dpus = _total_dpu_count(row)
    provisioned_ranks = _rank_count(row)
    active_dpus = _active_dpu_count(row)
    active_ranks = _active_rank_count(row)
    if (
        provisioned_dpus is None
        or provisioned_ranks is None
        or active_dpus is None
        or active_ranks is None
    ):
        return None
    if active_dpus == provisioned_dpus and active_ranks == provisioned_ranks:
        return f"{provisioned_dpus}DPU/{provisioned_ranks}R"
    return f"{active_dpus}/{provisioned_dpus}DPU + {active_ranks}/{provisioned_ranks}R"


def _fully_active_topology(row: Mapping[str, Any]) -> bool:
    provisioned_ranks = _rank_count(row)
    provisioned_dpus = _total_dpu_count(row)
    active_ranks = _active_rank_count(row)
    active_dpus = _active_dpu_count(row)
    return (
        provisioned_ranks is not None
        and provisioned_dpus is not None
        and active_ranks is not None
        and active_dpus is not None
        and active_ranks == provisioned_ranks
        and active_dpus == provisioned_dpus
    )


def _executor_config_hash(row: Mapping[str, Any]) -> str | None:
    value = str(row.get("executor_config_hash") or "").strip()
    return value or None


def _release_succeeded(row: Mapping[str, Any]) -> bool:
    if any(
        row.get(key) is True
        for key in (
            "release_succeeded",
            "hardware_release_verified",
            "release_verified",
            "release_confirmed",
            "session_release_verified",
        )
    ):
        return True
    return str(row.get("resource_release_status") or "").lower() in {
        "released",
        "passed",
        "verified",
        "clean",
    }


def _explicitly_false(row: Mapping[str, Any], *keys: str) -> bool:
    values = [row[key] for key in keys if key in row]
    return bool(values) and all(value is False for value in values)


def _valid(row: Mapping[str, Any]) -> bool:
    return _CLAIM_POLICY.evaluate_row(Claim.FUNCTIONAL_CORRECTNESS, row).allowed


def _performance_valid(row: Mapping[str, Any]) -> bool:
    """Return whether a row may support a measured performance ratio.

    A physically executed row can be useful functionality evidence while its
    timing is still explicitly bring-up-only.  Such rows remain in raw,
    runtime, validation, and transfer tables, but cannot support speedup or
    comparative runtime claims.
    """
    return _CLAIM_POLICY.evaluate_row(Claim.TIMING, row).allowed


def _hashes(row: Mapping[str, Any]) -> tuple[str, str, str] | None:
    values = tuple(
        str(row.get(key) or "")
        for key in (
            "circuit_semantics_hash",
            "tensor_network_hash",
            "contraction_plan_hash",
        )
    )
    return values if all(values) else None


def _same_plan_key(row: Mapping[str, Any]) -> tuple[str, ...] | None:
    hashes = _hashes(row)
    scope = _timing_scope(row)
    if hashes is None or not scope or not _case(row):
        return None
    return (*hashes, _case(row), _repeat(row), scope, _path(row), _numeric(row))


def _record_row(row: Mapping[str, Any]) -> JsonDict:
    local_dpus = _local_dpu_count(row)
    ranks = _rank_count(row)
    total_dpus = _total_dpu_count(row)
    active_dpus = _active_dpu_count(row)
    active_ranks = _active_rank_count(row)
    activity = _activity_label(row)
    topology = (
        f" | {local_dpus} local / {ranks} ranks / {total_dpus} total"
        if local_dpus
        else ""
    )
    return {
        "case_id": _case(row),
        "family": _family(row),
        "qubits": _qubits(row),
        "engine": _engine(row),
        "route_id": str(row.get("route_id") or ""),
        "route_config_hash": str(row.get("route_config_hash") or ""),
        "executor_config_hash": _executor_config_hash(row),
        "engine_class": _engine_class(row),
        "path": _path(row),
        "numeric_policy": _numeric(row),
        "repeat_id": _repeat(row),
        "timing_scope": _timing_scope(row),
        "runtime_s": _runtime(row),
        "status": row.get("status", "completed"),
        "scientific_admitted": _valid(row),
        "circuit_semantics_hash": row.get("circuit_semantics_hash"),
        "tensor_network_hash": row.get("tensor_network_hash"),
        "contraction_plan_hash": row.get("contraction_plan_hash"),
        "admission_identity": _admission_identity(row),
        "cross_algorithm": _engine_class(row) == "cross_algorithm",
        "local_dpu_count": local_dpus,
        "rank_count": ranks,
        "total_dpu_count": total_dpus,
        "provisioned_dpu_count": total_dpus,
        "provisioned_rank_count": ranks,
        "active_dpu_count": active_dpus,
        "active_rank_count": active_ranks,
        "fully_active_topology": _fully_active_topology(row),
        "activity_label": activity,
        "series": (
            f"{_family(row)} | {_engine(row)} | {_path(row)} | {_numeric(row)}"
            f"{(' | ' + activity) if activity else topology}"
        ),
    }


def _group_median(rows: Iterable[Mapping[str, Any]]) -> float | None:
    values = [value for row in rows if (value := _runtime(row)) is not None]
    return statistics.median(values) if values else None


def _quartiles(values: list[float]) -> tuple[float, float]:
    """Return inclusive quartiles while keeping a one-repeat row explicit."""
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0], ordered[0]
    q1, _median, q3 = statistics.quantiles(ordered, n=4, method="inclusive")
    return q1, q3


def _aggregate_matched_rows(
    rows: list[JsonDict],
    *,
    group_fields: tuple[str, ...],
    value_fields: tuple[str, ...],
) -> list[JsonDict]:
    """Summarize matched-repeat comparisons without mixing evidence identities."""
    groups: dict[tuple[Any, ...], list[JsonDict]] = {}
    for row in rows:
        groups.setdefault(tuple(row.get(field) for field in group_fields), []).append(
            row
        )
    result: list[JsonDict] = []
    for group in groups.values():
        summary = {key: value for key, value in group[0].items() if key != "repeat_id"}
        summary["aggregation"] = "median_matched_repetitions"
        summary["repeat_count"] = len(group)
        for field in value_fields:
            values = [
                value for row in group if (value := _float(row.get(field))) is not None
            ]
            if not values:
                continue
            q1, q3 = _quartiles(values)
            summary[field] = statistics.median(values)
            summary[f"{field}_q1"] = q1
            summary[f"{field}_q3"] = q3
        result.append(summary)
    return result


def _aggregate_medians(
    rows: list[JsonDict],
    *,
    group_fields: tuple[str, ...],
    value_fields: tuple[str, ...],
    all_true_fields: tuple[tuple[str, str], ...] = (),
    drop_fields: tuple[str, ...] = ("repeat_id",),
) -> list[JsonDict]:
    """Summarize repeated records with median-only numeric fields."""
    groups: dict[tuple[Any, ...], list[JsonDict]] = {}
    for row in rows:
        groups.setdefault(tuple(row.get(field) for field in group_fields), []).append(
            row
        )
    result: list[JsonDict] = []
    for group in groups.values():
        summary = {
            key: value for key, value in group[0].items() if key not in drop_fields
        }
        summary["aggregation"] = "median_matched_repetitions"
        summary["repeat_count"] = len(group)
        for field in value_fields:
            values = [
                value for row in group if (value := _float(row.get(field))) is not None
            ]
            if not values:
                continue
            summary[field] = statistics.median(values)
        for source_field, summary_field in all_true_fields:
            summary[summary_field] = all(row.get(source_field) is True for row in group)
        result.append(summary)
    return result


def _runtime_summaries(rows: list[JsonDict]) -> list[JsonDict]:
    groups: dict[tuple[Any, ...], list[JsonDict]] = {}
    for row in rows:
        if not _valid(row):
            continue
        normalized = _record_row(row)
        key = (
            normalized["case_id"],
            normalized["family"],
            normalized["qubits"],
            normalized["engine"],
            normalized["engine_class"],
            normalized["path"],
            normalized["numeric_policy"],
            normalized["timing_scope"],
            normalized["circuit_semantics_hash"],
            normalized["tensor_network_hash"],
            normalized["contraction_plan_hash"],
            normalized["route_id"],
            normalized["route_config_hash"],
            normalized["executor_config_hash"],
            normalized["local_dpu_count"],
            normalized["rank_count"],
            normalized["total_dpu_count"],
            normalized["provisioned_dpu_count"],
            normalized["provisioned_rank_count"],
            normalized["active_dpu_count"],
            normalized["active_rank_count"],
            normalized["activity_label"],
            normalized["admission_identity"],
        )
        groups.setdefault(key, []).append(row)
    summaries = []
    for group in groups.values():
        normalized = _record_row(group[0])
        runtimes = sorted(
            value for row in group if (value := _runtime(row)) is not None
        )
        summaries.append(
            {
                key: normalized[key]
                for key in (
                    "case_id",
                    "family",
                    "qubits",
                    "engine",
                    "engine_class",
                    "path",
                    "numeric_policy",
                    "timing_scope",
                    "circuit_semantics_hash",
                    "tensor_network_hash",
                    "contraction_plan_hash",
                    "route_id",
                    "route_config_hash",
                    "executor_config_hash",
                    "admission_identity",
                    "cross_algorithm",
                    "local_dpu_count",
                    "rank_count",
                    "total_dpu_count",
                    "provisioned_dpu_count",
                    "provisioned_rank_count",
                    "active_dpu_count",
                    "active_rank_count",
                    "fully_active_topology",
                    "activity_label",
                    "series",
                )
            }
            | {
                "median_runtime_s": statistics.median(runtimes),
                "min_runtime_s": runtimes[0],
                "max_runtime_s": runtimes[-1],
                "repeat_count": len(runtimes),
            }
        )
    return summaries


def _pair_rows(
    rows: list[JsonDict],
    left_class: str,
    right_class: str,
    claim_rejections: list[JsonDict] | None = None,
) -> list[JsonDict]:
    groups: dict[tuple[Any, ...], dict[str, list[JsonDict]]] = {}
    for row in rows:
        if not _performance_valid(row) or _engine_class(row) not in {
            left_class,
            right_class,
        }:
            continue
        key = _same_plan_key(row)
        if key is None:
            continue
        groups.setdefault(key, {}).setdefault(_engine_class(row), []).append(row)
    pairs: list[JsonDict] = []
    for group in groups.values():
        left_rows = group.get(left_class, [])
        right_rows = group.get(right_class, [])
        if not left_rows or not right_rows:
            continue
        left_identities = {
            json.dumps(_source_identity(row), sort_keys=True) for row in left_rows
        }
        if len(left_identities) != 1:
            # A comparison must not choose an arbitrary executor profile when
            # more than one baseline satisfies the scientific-plan key.
            continue
        # A CPU measurement is a reusable baseline for every matching UPMEM
        # topology.  Never collapse the right-hand rows into one dictionary.
        baseline = left_rows[0]
        for right in sorted(
            right_rows,
            key=lambda row: (
                _rank_count(row) or 0,
                _local_dpu_count(row) or 0,
                _total_dpu_count(row) or 0,
                _engine(row),
            ),
        ):
            decision = _CLAIM_POLICY.evaluate_pair(
                Claim.SPEEDUP,
                baseline,
                right,
                require_repeat_context=True,
            )
            if not decision.allowed:
                _append_claim_rejection(
                    claim_rejections, Claim.SPEEDUP, baseline, right, decision
                )
                continue
            pairs.append({left_class: baseline, right_class: right})
    return pairs


def _append_claim_rejection(
    rejections: list[JsonDict] | None,
    claim: Claim,
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    decision: ClaimDecision,
) -> None:
    if rejections is None or decision.allowed:
        return
    rejections.append(
        {
            "claim": claim.value,
            "case_id": _case(baseline) or _case(candidate),
            "baseline_route": str(
                baseline.get("route_id") or baseline.get("engine_id") or ""
            ),
            "candidate_route": str(
                candidate.get("route_id") or candidate.get("engine_id") or ""
            ),
            "repeat_id": _repeat(baseline),
            "reasons": "; ".join(decision.reasons),
        }
    )


def _unambiguous_variant(rows: list[JsonDict]) -> JsonDict | None:
    if not rows:
        return None
    first = rows[0]
    return first if all(row == first for row in rows[1:]) else None


def _has_matching_engine_rows(
    rows: list[JsonDict], left_class: str, right_class: str
) -> bool:
    """Return whether both engines share an identity/timing-scope key.

    This deliberately ignores performance admission.  It is used only to
    explain an empty performance table without changing which rows may enter
    measured ratios.
    """
    engines_by_key: dict[tuple[Any, ...], set[str]] = {}
    for row in rows:
        engine_class = _engine_class(row)
        if engine_class not in {left_class, right_class}:
            continue
        key = _same_plan_key(row)
        if key is not None:
            engines_by_key.setdefault(key, set()).add(engine_class)
    return any(
        {left_class, right_class}.issubset(engine_classes)
        for engine_classes in engines_by_key.values()
    )


def _write_csv(path: Path, rows: list[JsonDict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _short_series_label(value: Any, family: str | None = None) -> str:
    """Return a compact, deterministic label for a shared figure legend."""
    label = str(value or "unknown")
    if family and label.startswith(f"{family} | "):
        label = label[len(family) + 3 :]
    parts = [part.strip() for part in label.split("|") if part.strip()]

    def topology_token(part: str) -> tuple[int, int] | None:
        match = re.search(
            r"(\d+)DPU\s*x\s*(\d+)\s*rank\(s\)\s*=\s*\d+\s*total",
            part,
            flags=re.IGNORECASE,
        )
        if match is None:
            match = re.search(
                r"(\d+)\s*local\s*/\s*(\d+)\s*ranks?\s*/\s*\d+\s*total",
                part,
                flags=re.IGNORECASE,
            )
        if match is None:
            match = re.search(r"(\d+)DPU\s*/\s*(\d+)R", part)
        if match is None:
            return None
        return int(match.group(1)), int(match.group(2))

    def engine_token(part: str) -> tuple[str | None, tuple[int, int] | None]:
        lowered = part.lower()
        match = re.fullmatch(r"upmem_physical_(\d+)rank_(\d+)dpu", lowered)
        if match:
            return "UPMEM physical", (int(match.group(2)), int(match.group(1)))
        if lowered in {"numpy_cpu", "cpu_numpy"}:
            return "CPU", None
        if "upmem" in lowered or re.fullmatch(r"[a-z0-9_]*dpu[a-z0-9_]*", lowered):
            return "UPMEM", None
        if lowered == "quest_cpu_full_state":
            return "QuEST CPU", None
        if lowered == "quimb_tn":
            return "Quimb TN", None
        return None, None

    def compact_token(part: str) -> str:
        lowered = part.lower()
        if re.fullmatch(r"upmem_physical_(\d+)rank_(\d+)dpu", lowered):
            return "UPMEM physical"
        if lowered in {"numpy_cpu", "cpu_numpy"}:
            return "CPU"
        if lowered == "quest_cpu_full_state":
            return "QuEST CPU"
        if lowered == "quimb_tn":
            return "Quimb TN"
        replacements = (
            ("opt_einsum_greedy", "greedy"),
            ("cotengra_flops_seed0", "cotengra FLOPs"),
            ("host_packed_int8", "host-packed Int8"),
            ("float32_real", "Float32"),
            ("real_float32", "Float32"),
            ("f32_real", "Float32"),
            ("float32", "Float32"),
            ("upmem_m5", "UPMEM"),
            ("whole_route_including_session_lifecycle", "whole route"),
            ("whole_circuit_steady_state_v1", "steady state"),
        )
        result = part
        for source, target in replacements:
            result = result.replace(source, target)
        return result

    tokens: list[str] = []
    topology: tuple[int, int] | None = None
    activity: str | None = None
    engine, embedded_topology = engine_token(parts[0]) if parts else (None, None)
    if engine is not None:
        tokens.append(engine)
        topology = embedded_topology
        parts = parts[1:]
    for part in parts:
        nested_engine, nested_topology = engine_token(part)
        if nested_engine is not None:
            tokens.append(nested_engine)
            if nested_topology is not None:
                topology = nested_topology
            continue
        partial_match = re.fullmatch(
            r"(\d+)\s*/\s*(\d+)DPU\s*\+\s*(\d+)\s*/\s*(\d+)R",
            part,
            flags=re.IGNORECASE,
        )
        if partial_match:
            activity = (
                f"{partial_match.group(1)}/{partial_match.group(2)}DPU + "
                f"{partial_match.group(3)}/{partial_match.group(4)}R"
            )
            continue
        parsed_topology = topology_token(part)
        looks_like_topology = (
            ("dpu" in part.lower() and "rank" in part.lower())
            or ("local" in part.lower() and "total" in part.lower())
            or re.search(r"\d+DPU\s*/\s*\d+R", part, flags=re.IGNORECASE) is not None
        )
        if parsed_topology is not None or looks_like_topology:
            if parsed_topology is not None:
                topology = parsed_topology
        else:
            tokens.append(compact_token(part))
    if activity is not None:
        tokens.append(activity)
    elif engine != "CPU" and topology is not None:
        tokens.append(f"{topology[0]}DPU/{topology[1]}R")
    return " / ".join(tokens) or "unknown"


def _identity_series(
    row: Mapping[str, Any], *, include_stage: str | None = None
) -> str:
    """Build a stable series identity from record fields already present in rows."""
    scope = _timing_scope(row)
    topology = _activity_label(row) or _topology_label(row) or "host"
    parts = [
        _case(row) or "unknown_case",
        _engine(row),
        _path(row),
        _numeric(row),
        scope or "unspecified",
        topology,
    ]
    if include_stage is not None:
        parts.append(include_stage)
    return " | ".join(parts)


def _admission_identity(row: Mapping[str, Any]) -> str:
    """Serialize the raw fields that determine scientific admission."""
    return json.dumps(
        {
            "status": row.get("status", "completed"),
            "validation_status": row.get("validation_status"),
            "scientific_validation_status": row.get("scientific_validation_status"),
            "exact_once": row.get("exact_once"),
            "no_fallback_used": row.get("no_fallback_used"),
            "target_observed": row.get("target_observed"),
            "hardware_allocation_verified": row.get("hardware_allocation_verified"),
            "native_kernel_executed": row.get("native_kernel_executed"),
            "hardware_kernel_executed": row.get("hardware_kernel_executed"),
            "simulator": row.get("simulator"),
            "simulator_kernel_executed": row.get("simulator_kernel_executed"),
            "cpu_fallback": row.get("cpu_fallback"),
            "cpu_fallback_used": row.get("cpu_fallback_used"),
            "release_succeeded": _release_succeeded(row),
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _plot_provisioned_topology(row: Mapping[str, Any]) -> tuple[int, int] | None:
    dpus = _int(row.get("provisioned_dpu_count"))
    if dpus is None:
        dpus = _int(row.get("total_dpu_count"))
    if dpus is None:
        dpus = _int(row.get("allocated_dpu_count"))
    if dpus is None:
        dpus = _int(row.get("requested_dpu_count"))
    ranks = _int(row.get("provisioned_rank_count"))
    if ranks is None:
        ranks = _int(row.get("rank_count"))
    if dpus is None or ranks is None or dpus <= 0 or ranks <= 0:
        return None
    return dpus, ranks


def _plot_topology_fragment(part: str) -> bool:
    """Identify old or compact topology/activity tokens in a series label."""
    if re.fullmatch(
        r"\d+\s*/\s*\d+DPU\s*\+\s*\d+\s*/\s*\d+R",
        part,
        flags=re.IGNORECASE,
    ):
        return True
    if re.search(r"\d+DPU\s*/\s*\d+R", part, flags=re.IGNORECASE):
        return True
    if "dpu" in part.lower() and "rank" in part.lower():
        return True
    return "local" in part.lower() and "total" in part.lower()


def _plot_series_key(
    row: Mapping[str, Any], group: str, *, collapse_activity: bool
) -> tuple[str, tuple[int, int] | None]:
    raw = str(row.get(group, "unknown"))
    if not collapse_activity:
        return raw, None
    parts = [part.strip() for part in raw.split("|") if part.strip()]
    parts = [part for part in parts if not _plot_topology_fragment(part)]
    return " | ".join(parts) or "unknown", _plot_provisioned_topology(row)


def _plot_activity_range(
    rows: list[Mapping[str, Any]], topology: tuple[int, int] | None
) -> str | None:
    if topology is None:
        return None
    active_dpus = sorted(
        {
            value
            for row in rows
            if (value := _int(row.get("active_dpu_count"))) is not None
        }
    )
    active_ranks = sorted(
        {
            value
            for row in rows
            if (value := _int(row.get("active_rank_count"))) is not None
        }
    )
    if not active_dpus or not active_ranks:
        return None
    provisioned_dpus, provisioned_ranks = topology
    if len(active_dpus) == 1 and len(active_ranks) == 1:
        if active_dpus[0] == provisioned_dpus and active_ranks[0] == provisioned_ranks:
            return f"{provisioned_dpus}DPU/{provisioned_ranks}R"
        dpu_text = f"{active_dpus[0]} active of {provisioned_dpus} provisioned DPU"
        rank_text = f"{active_ranks[0]} of {provisioned_ranks} rank"
        return f"{dpu_text} / {rank_text}"
    dpu_text = (
        f"{active_dpus[0]}-{active_dpus[-1]} active of "
        f"{provisioned_dpus} provisioned DPU"
        if len(active_dpus) > 1
        else f"{active_dpus[0]} active of {provisioned_dpus} provisioned DPU"
    )
    rank_text = (
        f"{active_ranks[0]}-{active_ranks[-1]} of {provisioned_ranks} ranks"
        if len(active_ranks) > 1
        else f"{active_ranks[0]} of {provisioned_ranks} rank"
    )
    return f"{dpu_text} / {rank_text}"


def _plot_display_label(
    family: str,
    base_series: str,
    topology: tuple[int, int] | None,
    rows: list[Mapping[str, Any]],
) -> str:
    label = _short_series_label(base_series, family)
    activity = _plot_activity_range(rows, topology)
    return f"{label} / {activity}" if activity else label


def _plot_exact_identity(row: Mapping[str, Any], group: str) -> tuple[Any, ...]:
    """Return the evidence fields that figures must never collapse."""
    return (
        str(row.get(group) or ""),
        str(row.get("engine") or ""),
        str(row.get("route_id") or ""),
        str(row.get("route_config_hash") or ""),
        str(row.get("executor_config_hash") or ""),
        str(row.get("timing_scope") or ""),
        _int(row.get("provisioned_dpu_count")),
        _int(row.get("provisioned_rank_count")),
        _int(row.get("active_dpu_count")),
        _int(row.get("active_rank_count")),
        str(row.get("admission_identity") or ""),
        str(row.get("status") or ""),
        str(row.get("validation_status") or ""),
        str(row.get("scientific_validation_status") or ""),
        row.get("scientific_admitted"),
        str(row.get("comparison_identity") or ""),
    )


def _plot_semantic_label(row: Mapping[str, Any], group: str, family: str) -> str:
    """Return the compact human comparison label, independent of topology."""
    return _short_series_label(str(row.get(group) or "unknown"), family)


def _visual_engine_label(row: Mapping[str, Any]) -> str:
    """Return an engine label without embedding one topology in the legend."""
    engine_class = _engine_class(row)
    if engine_class == "upmem":
        return "UPMEM physical"
    if engine_class == "cpu":
        return "CPU"
    return _short_series_label(_engine(row))


def _plot_topology_key(row: Mapping[str, Any]) -> tuple[int, int, int, int] | None:
    provisioned = _plot_provisioned_topology(row)
    active_dpus = _int(row.get("active_dpu_count"))
    active_ranks = _int(row.get("active_rank_count"))
    if provisioned is None or active_dpus is None or active_ranks is None:
        return None
    return provisioned[0], provisioned[1], active_dpus, active_ranks


def _plot_topology_label(topology: tuple[int, int, int, int] | None) -> str:
    if topology is None:
        return "host"
    provisioned_dpus, provisioned_ranks, active_dpus, active_ranks = topology
    return f"{active_dpus}/{provisioned_dpus} DPU, {active_ranks}/{provisioned_ranks} R"


def _timing_bar_label(qubits: int, topology: str) -> str:
    """Keep timing x labels compact while retaining the active DPU count."""
    partial = re.fullmatch(
        r"(\d+)/(\d+)\s*DPU(?:\s*\+|,)\s*(\d+)/(\d+)\s*R",
        topology,
    )
    full = re.fullmatch(r"(\d+)\s*DPU/(\d+)\s*R", topology)
    if partial is not None:
        active_dpus, provisioned_dpus, active_ranks, provisioned_ranks = (
            partial.groups()
        )
    elif full is not None:
        provisioned_dpus, provisioned_ranks = full.groups()
        active_dpus, active_ranks = provisioned_dpus, provisioned_ranks
    else:
        return f"{qubits}q\n{topology}"
    ranks = (
        ""
        if active_ranks == provisioned_ranks == "1"
        else f" {active_ranks}/{provisioned_ranks}R"
    )
    return f"{qubits}q\n{active_dpus}/{provisioned_dpus} DPU{ranks}"


def _is_upmem_plot_row(row: Mapping[str, Any]) -> bool:
    return _engine_class(row) == "upmem"


def _source_identity(row: Mapping[str, Any]) -> dict[str, Any]:
    """Return the source identity retained by a derived comparison row."""
    return {
        "engine": _engine(row),
        "route_id": str(row.get("route_id") or ""),
        "route_config_hash": str(row.get("route_config_hash") or ""),
        "executor_config_hash": _executor_config_hash(row),
        "timing_scope": _timing_scope(row),
        "provisioned_dpu_count": _total_dpu_count(row),
        "provisioned_rank_count": _rank_count(row),
        "active_dpu_count": _active_dpu_count(row),
        "active_rank_count": _active_rank_count(row),
        "admission_identity": _admission_identity(row),
    }


def _comparison_identity(*rows: Mapping[str, Any]) -> str:
    return json.dumps([_source_identity(row) for row in rows], sort_keys=True)


def _faceted_plot(
    path: Path,
    title: str,
    rows: list[JsonDict],
    x: str,
    y: str,
    group: str,
    *,
    reference_y: float | None = None,
    log_y: bool = False,
    panel_title: str | None = None,
    semantic_group: str | None = None,
    collapse_activity: bool = False,
) -> bool:
    """Plot exact rows with compact semantic legends and explicit topology.

    Tables retain one row for every evidence identity.  The figure deliberately
    uses a shorter semantic label (engine/planner/numeric mode) for colour, and
    uses marker shape plus a dedicated legend for active/provisioned topology.
    UPMEM points are only connected when one exact execution identity spans the
    complete series; changing topology or configuration is rendered as points.
    """
    if not rows:
        return False
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    semantic_group = semantic_group or group
    by_family: dict[
        str, dict[str, dict[tuple[Any, ...], list[tuple[float, float, JsonDict]]]]
    ] = {}
    plot_rows: list[tuple[JsonDict, float, float, str, str, tuple[Any, ...]]] = []
    for row in rows:
        xv, yv = _float(row.get(x)), _float(row.get(y))
        if xv is None or yv is None:
            continue
        family = str(row.get("family") or "unknown")
        semantic = _plot_semantic_label(row, semantic_group, family)
        exact = _plot_exact_identity(row, group)
        plot_rows.append((row, xv, yv, family, semantic, exact))
        by_family.setdefault(family, {}).setdefault(semantic, {}).setdefault(
            exact, []
        ).append((xv, yv, row))
    if not by_family:
        return False
    families = sorted(by_family)
    ncols = min(3, len(families))
    nrows = math.ceil(len(families) / ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.6 * ncols, 3.3 * nrows),
        squeeze=False,
        sharex=False,
        sharey=False,
    )
    short_labels = sorted(
        {semantic for values in by_family.values() for semantic in values}
    )
    colors = {
        label: plt.get_cmap("tab10")(index % 10)
        for index, label in enumerate(short_labels)
    }
    from matplotlib.lines import Line2D

    semantic_handles = {
        label: Line2D([0], [0], color=colors[label], marker="o", label=label)
        for label in short_labels
    }
    topologies = sorted(
        {
            topology
            for row, _xv, _yv, _family, _semantic, _exact in plot_rows
            if _is_upmem_plot_row(row)
            if (topology := _plot_topology_key(row)) is not None
        }
    )
    topology_markers = {
        topology: _PLOT_MARKERS[index % len(_PLOT_MARKERS)]
        for index, topology in enumerate(topologies)
    }
    topology_handles = {
        topology: Line2D(
            [0],
            [0],
            color="black",
            linestyle="None",
            marker=topology_markers[topology],
            label=_plot_topology_label(topology),
        )
        for topology in topologies
    }
    offset_index: dict[tuple[str, float, tuple[Any, ...]], int] = {}
    offset_width: dict[tuple[str, float], int] = {}
    by_x: dict[tuple[str, float], set[tuple[Any, ...]]] = {}
    for row, xv, _yv, family, _semantic, exact in plot_rows:
        if _is_upmem_plot_row(row):
            by_x.setdefault((family, xv), set()).add(exact)
    for key, identities in by_x.items():
        ordered = sorted(identities, key=repr)
        offset_width[key] = len(ordered)
        for position, identity in enumerate(ordered):
            offset_index[(key[0], key[1], identity)] = position

    metric_title = panel_title or y.replace("_", " ").title()
    for index, family in enumerate(families):
        axis = axes[index // ncols][index % ncols]
        for semantic, exact_series in sorted(by_family[family].items()):
            for exact, points in sorted(
                exact_series.items(), key=lambda item: repr(item[0])
            ):
                points.sort(key=lambda point: point[0])
                first_row = points[0][2]
                is_upmem = _is_upmem_plot_row(first_row)
                topology = _plot_topology_key(first_row)
                marker = topology_markers.get(topology, "o")
                active = topology[2] if topology is not None else 1
                marker_size = 28 + 10 * math.sqrt(max(active, 1))
                x_values = [point[0] for point in points]
                y_values = [point[1] for point in points]
                # Any UPMEM series with a second identity under this semantic
                # label is intentionally point-only. It prevents visual lines
                # from implying a fixed topology/configuration experiment.
                connect = not is_upmem and len(exact_series) == 1
                if connect:
                    axis.plot(
                        x_values,
                        y_values,
                        marker="o",
                        color=colors[semantic],
                        linewidth=1.6,
                    )
                else:
                    shifted = []
                    for xv in x_values:
                        width = offset_width.get((family, xv), 1)
                        position = offset_index.get((family, xv, exact), 0)
                        shifted.append(xv + (position - (width - 1) / 2.0) * 0.12)
                    axis.scatter(
                        shifted,
                        y_values,
                        marker=marker,
                        s=marker_size,
                        color=colors[semantic],
                        edgecolors="black" if is_upmem else colors[semantic],
                        linewidths=0.35 if is_upmem else 0.0,
                        zorder=3,
                    )
        if reference_y is not None:
            axis.axhline(
                reference_y,
                color="black",
                linestyle="--",
                linewidth=1,
                alpha=0.7,
            )
        if log_y:
            axis.set_yscale("log")
        axis.set_title(f"{family} | {metric_title}", fontsize="medium")
        axis.set_xlabel(x.replace("_", " ").title())
        axis.set_ylabel(y.replace("_", " ").title())
        axis.grid(True, alpha=0.25)
    for index in range(len(families), nrows * ncols):
        axes[index // ncols][index % ncols].set_visible(False)
    fig.suptitle(title, y=0.995, fontsize="large")
    legend_rows = 0
    if semantic_handles:
        ordered = sorted(semantic_handles)
        legend_columns = min(4, len(ordered))
        legend_rows += math.ceil(len(ordered) / legend_columns)
        semantic_legend = fig.legend(
            [semantic_handles[label] for label in ordered],
            ordered,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.965),
            ncol=legend_columns,
            fontsize="small",
            frameon=False,
            title="Comparison",
        )
        fig.add_artist(semantic_legend)
    if topology_handles:
        ordered = sorted(topology_handles)
        legend_columns = min(4, len(ordered))
        legend_rows += math.ceil(len(ordered) / legend_columns)
        fig.legend(
            [topology_handles[topology] for topology in ordered],
            [_plot_topology_label(topology) for topology in ordered],
            loc="upper center",
            bbox_to_anchor=(0.5, 0.875),
            ncol=legend_columns,
            fontsize="x-small",
            frameon=False,
            title="UPMEM active/provisioned topology",
        )
    fig.text(
        0.5,
        0.015,
        "Exact route identities remain separate in source CSVs. UPMEM points are unconnected when topology/configuration differs.",
        ha="center",
        fontsize="x-small",
    )
    # Explicit margins keep the external legend and panel titles clear without
    # relying on tight_layout, which is unstable for dynamic facet grids.
    fig.subplots_adjust(
        left=0.08,
        right=0.98,
        bottom=0.16,
        top=(max(0.24, min(0.80, 0.89 - 0.085 * legend_rows)))
        if semantic_handles
        else 0.90,
        wspace=0.32,
        hspace=0.42,
    )
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return True


def _plot(
    path: Path,
    title: str,
    rows: list[JsonDict],
    x: str,
    y: str,
    group: str,
    *,
    reference_y: float | None = None,
    log_y: bool = False,
    panel_title: str | None = None,
    semantic_group: str | None = None,
    collapse_activity: bool = False,
) -> bool:
    """Compatibility wrapper for the family-faceted renderer."""
    return _faceted_plot(
        path,
        title,
        rows,
        x,
        y,
        group,
        reference_y=reference_y,
        log_y=log_y,
        panel_title=panel_title,
        semantic_group=semantic_group,
        collapse_activity=collapse_activity,
    )


def _timing_execution_identity(row: Mapping[str, Any]) -> tuple[Any, ...]:
    """Return every execution field that must remain distinct in timing plots."""
    return (
        str(row.get("path") or "unspecified"),
        str(row.get("numeric_policy") or "unspecified"),
        str(row.get("engine") or "unknown"),
        str(row.get("route_id") or ""),
        str(row.get("route_config_hash") or ""),
        str(row.get("executor_config_hash") or ""),
        str(row.get("timing_scope") or ""),
        _int(row.get("provisioned_dpu_count")),
        _int(row.get("provisioned_rank_count")),
        _int(row.get("active_dpu_count")),
        _int(row.get("active_rank_count")),
        str(row.get("admission_identity") or ""),
    )


def _timing_topology_label(row: Mapping[str, Any]) -> str:
    """Format the explicit topology fields retained by timing summaries."""
    provisioned_dpus = _int(row.get("provisioned_dpu_count"))
    provisioned_ranks = _int(row.get("provisioned_rank_count"))
    active_dpus = _int(row.get("active_dpu_count"))
    active_ranks = _int(row.get("active_rank_count"))
    if provisioned_dpus is None or provisioned_ranks is None:
        embedded = re.fullmatch(
            r"upmem_physical_(\d+)rank_(\d+)dpu",
            str(row.get("engine") or ""),
            flags=re.IGNORECASE,
        )
        if embedded is not None:
            return f"{embedded.group(2)}DPU/{embedded.group(1)}R"
        return _activity_label(row) or "topology unspecified"
    if active_dpus is None or active_ranks is None:
        return f"{provisioned_dpus}DPU/{provisioned_ranks}R provisioned"
    if active_dpus == provisioned_dpus and active_ranks == provisioned_ranks:
        return f"{provisioned_dpus}DPU/{provisioned_ranks}R"
    return f"{active_dpus}/{provisioned_dpus}DPU + {active_ranks}/{provisioned_ranks}R"


def _timing_route_label(row: Mapping[str, Any]) -> str:
    """Return a readable, unambiguous label for one physical timing identity."""
    planner_numeric = _short_series_label(
        f"{row.get('path', 'unspecified')} | {row.get('numeric_policy', 'unspecified')}"
    )
    topology = _timing_topology_label(row)
    scope_value = str(row.get("timing_scope") or "").strip()
    scope = _short_series_label(scope_value) if scope_value else ""
    engine_value = str(row.get("engine") or "unknown")
    engine = (
        "UPMEM physical"
        if re.fullmatch(
            r"upmem_physical_\d+rank_\d+dpu", engine_value, flags=re.IGNORECASE
        )
        else _short_series_label(engine_value)
    )
    identifiers = []
    route_id = str(row.get("route_id") or "").strip()
    route_hash = str(row.get("route_config_hash") or "").strip()
    executor_hash = str(row.get("executor_config_hash") or "").strip()
    if route_id:
        identifiers.append(f"route {route_id}")
    if route_hash:
        identifiers.append(f"route cfg {route_hash[:8]}")
    if executor_hash:
        identifiers.append(f"executor {executor_hash[:8]}")
    suffix = "; ".join([engine, topology, *([scope] if scope else []), *identifiers])
    return f"{planner_numeric}\n{suffix}"


def _timing_plot_summary(rows: list[JsonDict]) -> list[JsonDict]:
    """Median physical timing leaves across equally weighted circuit families."""
    by_case: dict[tuple[tuple[Any, ...], str, str, int, str], list[float]] = {}
    for row in rows:
        if (
            _engine_class(row) != "upmem"
            or row.get("timing_coverage") != "execution_stage_leaves"
        ):
            continue
        qubits = _int(row.get("qubits"))
        value = _float(row.get("time_s"))
        stage = str(row.get("stage") or "")
        if (
            qubits is None
            or value is None
            or value < 0
            or stage not in _TIMING_STAGE_ORDER
        ):
            continue
        key = (
            _timing_execution_identity(row),
            str(row.get("case_id") or "unknown_case"),
            str(row.get("family") or "unknown"),
            qubits,
            stage,
        )
        by_case.setdefault(key, []).append(value)

    by_family: dict[tuple[tuple[Any, ...], str, int, str], list[float]] = {}
    for (identity, _case_id, family, qubits, stage), values in by_case.items():
        by_family.setdefault((identity, family, qubits, stage), []).append(
            statistics.median(values)
        )

    across_families: dict[tuple[tuple[Any, ...], int, str], list[float]] = {}
    for (identity, _family, qubits, stage), values in by_family.items():
        across_families.setdefault((identity, qubits, stage), []).append(
            statistics.median(values)
        )

    result = []
    for (identity, qubits, stage), values in sorted(
        across_families.items(), key=lambda item: repr(item[0])
    ):
        (
            path_id,
            numeric_policy,
            engine,
            route_id,
            route_config_hash,
            executor_config_hash,
            timing_scope,
            provisioned_dpus,
            provisioned_ranks,
            active_dpus,
            active_ranks,
            admission_identity,
        ) = identity
        result.append(
            {
                "path": path_id,
                "numeric_policy": numeric_policy,
                "engine": engine,
                "route_id": route_id,
                "route_config_hash": route_config_hash,
                "executor_config_hash": executor_config_hash,
                "timing_scope": timing_scope,
                "provisioned_dpu_count": provisioned_dpus,
                "provisioned_rank_count": provisioned_ranks,
                "active_dpu_count": active_dpus,
                "active_rank_count": active_ranks,
                "admission_identity": admission_identity,
                "qubits": qubits,
                "stage": stage,
                "median_time_s": statistics.median(values),
                "family_count": len(values),
            }
        )
    return result


def _timing_breakdown_plot(
    path: Path, title: str, rows: list[JsonDict]
) -> tuple[bool, str]:
    """Render bounded planner/numeric panels with stacked physical timing leaves."""
    plot_rows = _timing_plot_summary(rows)
    if not plot_rows:
        reason = "no recorded physical execution-stage timing leaves"
        _todo_plot(path, title, reason)
        return False, reason

    panels = sorted(
        {(str(row["path"]), str(row["numeric_policy"])) for row in plot_rows}
    )
    if len(panels) > _MAX_TIMING_ROUTE_PANELS:
        reason = (
            f"{len(panels)} planner/numeric timing panels exceed the "
            f"{_MAX_TIMING_ROUTE_PANELS}-panel readability limit"
        )
        _todo_plot(path, title, reason)
        return False, reason

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
    except ImportError:
        return False, "matplotlib is unavailable"

    stages = [
        stage
        for stage in _TIMING_STAGE_ORDER
        if any(row["stage"] == stage for row in plot_rows)
    ]
    colors = {
        stage: plt.get_cmap("tab10")(index % 10)
        for index, stage in enumerate(_TIMING_STAGE_ORDER)
    }
    ncols = 1 if len(panels) == 1 else 2
    nrows = math.ceil(len(panels) / ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(8.0 * ncols, 4.2 * nrows + 1.6),
        squeeze=False,
        sharex=False,
        sharey=False,
    )

    for index, panel in enumerate(panels):
        axis = axes[index // ncols][index % ncols]
        panel_rows = [
            row
            for row in plot_rows
            if (str(row["path"]), str(row["numeric_policy"])) == panel
        ]
        bars: dict[tuple[Any, ...], list[JsonDict]] = {}
        for row in panel_rows:
            topology = _timing_topology_label(row)
            # A bar deliberately remains one exact execution identity. If two
            # identities would need the same human x-label, a figure would
            # conceal a configuration difference and is therefore invalid.
            bar_key = (int(row["qubits"]), topology)
            bars.setdefault(bar_key, []).append(row)
        collisions = []
        for bar_key, candidates in bars.items():
            identities = {
                _timing_execution_identity(candidate) for candidate in candidates
            }
            if len(identities) > 1:
                collisions.append(f"{bar_key[0]}q {bar_key[1]}")
        if collisions:
            reason = (
                "incompatible physical timing identities share intended bar(s): "
                + ", ".join(collisions)
            )
            _todo_plot(path, title, reason)
            return False, reason
        ordered_bars = sorted(
            bars,
            key=lambda key: (
                key[0],
                _int(bars[key][0].get("active_dpu_count")) or 0,
                _int(bars[key][0].get("provisioned_dpu_count")) or 0,
                key[1],
            ),
        )
        values_by_key = {
            (bar_key, str(row["stage"])): float(row["median_time_s"])
            for bar_key in ordered_bars
            for row in bars[bar_key]
        }
        positions = list(range(len(ordered_bars)))
        bottoms = [0.0] * len(ordered_bars)
        for stage in stages:
            values = [
                values_by_key.get((bar_key, stage), 0.0) for bar_key in ordered_bars
            ]
            axis.bar(
                positions,
                values,
                bottom=bottoms,
                width=0.72,
                color=colors[stage],
                label=_TIMING_STAGE_LABELS[stage],
            )
            bottoms = [bottom + value for bottom, value in zip(bottoms, values)]
        axis.set_title(
            _short_series_label(f"{panel[0]} | {panel[1]}"), fontsize="small"
        )
        axis.set_xlabel("Qubits and active/provisioned topology")
        axis.set_ylabel("Seconds")
        axis.set_xticks(positions)
        axis.set_xticklabels(
            [_timing_bar_label(qubits, topology) for qubits, topology in ordered_bars],
            fontsize=8,
        )
        axis.tick_params(axis="x", pad=8)
        axis.grid(axis="y", alpha=0.25)

    for index in range(len(panels), nrows * ncols):
        axes[index // ncols][index % ncols].set_visible(False)

    legend_handles = [
        Patch(facecolor=colors[stage], label=_TIMING_STAGE_LABELS[stage])
        for stage in stages
    ]
    fig.suptitle(title, y=0.995, fontsize="large")
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.91),
        ncol=min(5, len(legend_handles)),
        fontsize="small",
        frameon=False,
    )
    fig.text(
        0.5,
        0.015,
        "Absolute seconds; each bar is one exact topology/configuration identity, with medians across circuit families only.",
        ha="center",
        fontsize="small",
    )
    fig.subplots_adjust(
        left=0.08,
        right=0.98,
        bottom=0.15,
        top=0.76,
        wspace=0.24,
        hspace=0.38,
    )
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return True, ""


def _todo_plot(path: Path, title: str, reason: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(8, 4.5))
    axis.text(
        0.5,
        0.5,
        f"TODO\n{textwrap.fill(reason, width=68)}",
        ha="center",
        va="center",
        fontsize=14,
        transform=axis.transAxes,
    )
    axis.set_title(title)
    axis.set_xticks([])
    axis.set_yticks([])
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.1, top=0.88)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _entry(
    name: str, plot: Path, csv_path: Path, title: str, valid: bool, reason: str = ""
) -> PlotEntry:
    if not valid and not plot.is_file():
        _todo_plot(plot, title, reason or "no valid plot records")
    return PlotEntry(
        name,
        str(plot.name),
        str(csv_path.relative_to(csv_path.parents[1])),
        "generated_valid" if valid else "generated_todo_missing_data",
        title,
        None if valid else reason,
    )


def _one_qubit_multiple_topologies(rows: list[JsonDict]) -> bool:
    qubits = {value for row in rows if (value := _qubits(row)) is not None}
    topologies = {
        topology
        for row in rows
        if (topology := _plot_provisioned_topology(row)) is not None
    }
    return len(qubits) == 1 and len(topologies) > 1


def generate_report(
    source: Path | str | Iterable[Mapping[str, Any]], output_dir: Path | str
) -> ReportResult:
    """Write M5.5 CSV/PNG evidence and a TODO-aware plot manifest."""
    rows = load_records(source)
    output = Path(output_dir)
    tables = output / "tables"
    plots = output / "plots"
    tables.mkdir(parents=True, exist_ok=True)
    plots.mkdir(parents=True, exist_ok=True)
    normalized = [_record_row(row) for row in rows]
    runtime_summaries = _runtime_summaries(rows)
    admitted_rows = [row for row in rows if _valid(row)]
    entries: list[PlotEntry] = []
    claim_rejections: list[JsonDict] = []
    topology_dimension_reason = (
        "selected evidence has one qubit size and multiple topologies; use the "
        "dedicated strong-scaling figure and source CSV"
    )
    topology_dimension_ambiguous = _one_qubit_multiple_topologies(rows)

    runtime_csv = tables / "runtime_by_qubits.csv"
    _write_csv(
        runtime_csv,
        normalized,
        list(normalized[0])
        if normalized
        else [
            "case_id",
            "family",
            "qubits",
            "engine",
            "engine_class",
            "path",
            "numeric_policy",
            "repeat_id",
            "timing_scope",
            "runtime_s",
            "status",
            "local_dpu_count",
            "rank_count",
            "total_dpu_count",
            "provisioned_dpu_count",
            "provisioned_rank_count",
            "active_dpu_count",
            "active_rank_count",
            "fully_active_topology",
            "activity_label",
            "series",
            "circuit_semantics_hash",
            "tensor_network_hash",
            "contraction_plan_hash",
            "cross_algorithm",
        ],
    )
    runtime_summary_csv = tables / "runtime_by_case_median.csv"
    _write_csv(
        runtime_summary_csv,
        runtime_summaries,
        list(runtime_summaries[0])
        if runtime_summaries
        else [
            "case_id",
            "family",
            "qubits",
            "engine",
            "engine_class",
            "path",
            "numeric_policy",
            "timing_scope",
            "local_dpu_count",
            "rank_count",
            "total_dpu_count",
            "provisioned_dpu_count",
            "provisioned_rank_count",
            "active_dpu_count",
            "active_rank_count",
            "fully_active_topology",
            "activity_label",
            "series",
            "median_runtime_s",
            "repeat_count",
        ],
    )
    runtime_plot = plots / "runtime_by_qubits.png"
    execution_model = (
        "sequential TaskGraph task scheduling; intra-task DPU tile/rank parallelism"
    )
    runtime_title = f"M5.5 whole-circuit median runtime ({execution_model})"
    if topology_dimension_ambiguous:
        _todo_plot(runtime_plot, runtime_title, topology_dimension_reason)
        valid = False
        runtime_reason = topology_dimension_reason
    else:
        valid = _plot(
            runtime_plot,
            runtime_title,
            [
                dict(
                    row,
                    _visual_series=(
                        f"{_visual_engine_label(row)} | {row['path']} | {row['numeric_policy']}"
                    ),
                )
                for row in runtime_summaries
            ],
            "qubits",
            "median_runtime_s",
            "series",
            log_y=True,
            panel_title="median runtime (s)",
            semantic_group="_visual_series",
        )
        runtime_reason = "no valid runtime records"
    entries.append(
        _entry(
            "runtime_by_qubits",
            runtime_plot,
            runtime_summary_csv,
            runtime_title,
            valid,
            runtime_reason,
        )
    )

    pairs = _pair_rows(rows, "cpu", "upmem", claim_rejections)
    speedups = []
    for pair in pairs:
        cpu, upmem = pair["cpu"], pair["upmem"]
        cpu_time, upmem_time = _runtime(cpu), _runtime(upmem)
        if cpu_time and upmem_time:
            speedups.append(
                {
                    "case_id": _case(cpu),
                    "family": _family(cpu),
                    "qubits": _qubits(cpu),
                    "path": _path(cpu),
                    "numeric_policy": _numeric(cpu),
                    "timing_scope": _timing_scope(cpu),
                    "repeat_id": _repeat(upmem),
                    "local_dpu_count": _local_dpu_count(upmem),
                    "rank_count": _rank_count(upmem),
                    "total_dpu_count": _total_dpu_count(upmem),
                    "provisioned_dpu_count": _total_dpu_count(upmem),
                    "provisioned_rank_count": _rank_count(upmem),
                    "active_dpu_count": _active_dpu_count(upmem),
                    "active_rank_count": _active_rank_count(upmem),
                    "activity_label": _activity_label(upmem),
                    "topology": _topology_label(upmem),
                    "route_id": str(upmem.get("route_id") or ""),
                    "route_config_hash": str(upmem.get("route_config_hash") or ""),
                    "admission_identity": _admission_identity(upmem),
                    "comparison_identity": _comparison_identity(cpu, upmem),
                    "series": (
                        f"{_path(upmem)} | {_numeric(upmem)} | "
                        f"{_activity_label(upmem) or _topology_label(upmem)}"
                    ),
                    "cpu_runtime_s": cpu_time,
                    "upmem_runtime_s": upmem_time,
                    "speedup_cpu_over_upmem": cpu_time / upmem_time,
                    "circuit_semantics_hash": cpu.get("circuit_semantics_hash"),
                    "tensor_network_hash": cpu.get("tensor_network_hash"),
                    "contraction_plan_hash": cpu.get("contraction_plan_hash"),
                    "executor_config_hash": _executor_config_hash(upmem),
                }
            )
    speedups = _aggregate_matched_rows(
        speedups,
        group_fields=(
            "case_id",
            "family",
            "qubits",
            "path",
            "numeric_policy",
            "timing_scope",
            "local_dpu_count",
            "rank_count",
            "total_dpu_count",
            "provisioned_dpu_count",
            "provisioned_rank_count",
            "active_dpu_count",
            "active_rank_count",
            "activity_label",
            "topology",
            "route_id",
            "route_config_hash",
            "admission_identity",
            "comparison_identity",
            "series",
            "circuit_semantics_hash",
            "tensor_network_hash",
            "contraction_plan_hash",
            "executor_config_hash",
        ),
        value_fields=(
            "cpu_runtime_s",
            "upmem_runtime_s",
            "speedup_cpu_over_upmem",
        ),
    )
    speed_csv = tables / "same_plan_cpu_upmem_speedup.csv"
    _write_csv(
        speed_csv,
        speedups,
        list(speedups[0])
        if speedups
        else [
            "case_id",
            "family",
            "qubits",
            "path",
            "numeric_policy",
            "timing_scope",
            "repeat_count",
            "local_dpu_count",
            "rank_count",
            "total_dpu_count",
            "provisioned_dpu_count",
            "provisioned_rank_count",
            "active_dpu_count",
            "active_rank_count",
            "activity_label",
            "topology",
            "series",
            "cpu_runtime_s",
            "upmem_runtime_s",
            "speedup_cpu_over_upmem",
            "circuit_semantics_hash",
            "tensor_network_hash",
            "contraction_plan_hash",
            "executor_config_hash",
        ],
    )
    speed_plot = plots / "same_plan_cpu_upmem_speedup.png"
    speed_title = (
        "Measured same-plan CPU/UPMEM median speedup "
        f"(CPU time / UPMEM time; >1 favors UPMEM; {execution_model})"
    )
    valid = _plot(
        speed_plot,
        speed_title,
        [
            dict(row, _visual_series=f"{row['path']} | {row['numeric_policy']}")
            for row in speedups
        ],
        "qubits",
        "speedup_cpu_over_upmem",
        "series",
        reference_y=1.0,
        panel_title="CPU / UPMEM speedup",
        semantic_group="_visual_series",
    )
    entries.append(
        _entry(
            "same_plan_cpu_upmem_speedup",
            speed_plot,
            speed_csv,
            speed_title,
            valid,
            (
                "matching CPU/UPMEM rows exist, but no CPU/UPMEM pairs are "
                "performance-eligible/repeated"
                if _has_matching_engine_rows(rows, "cpu", "upmem")
                else "no matching CPU/UPMEM rows with all hashes and timing_scope"
            ),
        )
    )

    scaling: list[JsonDict] = []
    by_group: dict[tuple[Any, ...], list[JsonDict]] = {}
    for row in rows:
        if not _performance_valid(row) or _engine_class(row) != "upmem":
            continue
        hashes = _hashes(row)
        executor_hash = _executor_config_hash(row)
        if hashes is None or executor_hash is None:
            continue
        key = (
            *hashes,
            executor_hash,
            _case(row),
            _path(row),
            _numeric(row),
            _timing_scope(row),
            _repeat(row),
        )
        by_group.setdefault(key, []).append(row)
    for group in by_group.values():
        dpu_baseline = next(
            (
                row
                for row in group
                if _rank_count(row) == 1
                and _local_dpu_count(row) == 1
                and _total_dpu_count(row) == 1
                and _fully_active_topology(row)
            ),
            None,
        )
        dpu_base_time = _runtime(dpu_baseline) if dpu_baseline else None
        rank_baselines: dict[int, tuple[float, JsonDict] | None] = {}
        local_counts = sorted(
            {
                count
                for row in group
                if _rank_count(row) == 1
                if (count := _local_dpu_count(row)) is not None
            }
        )
        for local_dpus in local_counts:
            rank_baseline = next(
                (
                    row
                    for row in group
                    if _rank_count(row) == 1
                    and _local_dpu_count(row) == local_dpus
                    and _total_dpu_count(row) == local_dpus
                    and _fully_active_topology(row)
                ),
                None,
            )
            rank_baselines[local_dpus] = (
                (_runtime(rank_baseline), rank_baseline)
                if rank_baseline and _runtime(rank_baseline) is not None
                else None
            )
        for row in group:
            local_dpus = _local_dpu_count(row)
            rank_count = _rank_count(row)
            total_dpus = _total_dpu_count(row)
            active_dpus = _active_dpu_count(row)
            active_ranks = _active_rank_count(row)
            runtime = _runtime(row)
            fully_active = _fully_active_topology(row)
            comparison_baseline: JsonDict | None = None
            scaling_allowed = False
            if rank_count == 1:
                scale_dimension = "dpu"
                baseline_local = 1
                baseline_ranks = 1
                baseline_total = 1
                baseline_time = dpu_base_time
                scale_value = active_dpus
                comparison_baseline = dpu_baseline
                scaling_allowed = row is dpu_baseline
                if comparison_baseline is not None and row is not comparison_baseline:
                    decision = _CLAIM_POLICY.evaluate_pair(
                        Claim.SCALING, comparison_baseline, row
                    )
                    scaling_allowed = decision.allowed
                    if not decision.allowed:
                        _append_claim_rejection(
                            claim_rejections,
                            Claim.SCALING,
                            comparison_baseline,
                            row,
                            decision,
                        )
                speedup = (
                    baseline_time / runtime
                    if scaling_allowed
                    and fully_active
                    and baseline_time is not None
                    and runtime is not None
                    and scale_value is not None
                    else None
                )
                efficiency = speedup / scale_value if speedup and scale_value else None
            elif rank_count is not None and rank_count > 1 and local_dpus is not None:
                scale_dimension = "rank"
                baseline_local = local_dpus
                baseline_ranks = 1
                baseline_total = local_dpus
                baseline = rank_baselines.get(local_dpus)
                baseline_time = baseline[0] if baseline else None
                comparison_baseline = baseline[1] if baseline else None
                scale_value = active_ranks
                if comparison_baseline is not None:
                    decision = _CLAIM_POLICY.evaluate_pair(
                        Claim.SCALING, comparison_baseline, row
                    )
                    scaling_allowed = decision.allowed
                    if not decision.allowed:
                        _append_claim_rejection(
                            claim_rejections,
                            Claim.SCALING,
                            comparison_baseline,
                            row,
                            decision,
                        )
                speedup = (
                    baseline_time / runtime
                    if scaling_allowed
                    and fully_active
                    and baseline_time is not None
                    and runtime is not None
                    and scale_value is not None
                    else None
                )
                efficiency = speedup / scale_value if speedup and scale_value else None
            else:
                continue
            topology = _topology_label(row)
            scaling.append(
                {
                    "case_id": _case(row),
                    "family": _family(row),
                    "qubits": _qubits(row),
                    "path": _path(row),
                    "numeric_policy": _numeric(row),
                    "timing_scope": _timing_scope(row),
                    "executor_config_hash": _executor_config_hash(row),
                    "circuit_semantics_hash": row.get("circuit_semantics_hash"),
                    "tensor_network_hash": row.get("tensor_network_hash"),
                    "contraction_plan_hash": row.get("contraction_plan_hash"),
                    "scale_dimension": scale_dimension,
                    "local_dpu_count": local_dpus,
                    "rank_count": rank_count,
                    "total_dpu_count": total_dpus,
                    "active_dpu_count": active_dpus,
                    "active_rank_count": active_ranks,
                    "fully_active_topology": fully_active,
                    "topology": topology,
                    "series": f"{_path(row)} | {_numeric(row)} | {topology}",
                    "baseline_local_dpu_count": baseline_local,
                    "baseline_rank_count": baseline_ranks,
                    "baseline_total_dpu_count": baseline_total,
                    "baseline_runtime_s": baseline_time,
                    "runtime_s": runtime,
                    "speedup": speedup,
                    "efficiency": efficiency,
                }
            )
    scaling = _aggregate_matched_rows(
        scaling,
        group_fields=(
            "case_id",
            "family",
            "qubits",
            "path",
            "numeric_policy",
            "timing_scope",
            "executor_config_hash",
            "circuit_semantics_hash",
            "tensor_network_hash",
            "contraction_plan_hash",
            "scale_dimension",
            "local_dpu_count",
            "rank_count",
            "total_dpu_count",
            "active_dpu_count",
            "active_rank_count",
            "fully_active_topology",
            "topology",
            "series",
            "baseline_local_dpu_count",
            "baseline_rank_count",
            "baseline_total_dpu_count",
        ),
        value_fields=("baseline_runtime_s", "runtime_s", "speedup", "efficiency"),
    )
    scale_csv = tables / "upmem_strong_scaling.csv"
    _write_csv(
        scale_csv,
        scaling,
        list(scaling[0])
        if scaling
        else [
            "case_id",
            "family",
            "qubits",
            "path",
            "numeric_policy",
            "timing_scope",
            "scale_dimension",
            "local_dpu_count",
            "rank_count",
            "total_dpu_count",
            "active_dpu_count",
            "active_rank_count",
            "fully_active_topology",
            "topology",
            "series",
            "baseline_local_dpu_count",
            "baseline_rank_count",
            "baseline_total_dpu_count",
            "baseline_runtime_s",
            "runtime_s",
            "speedup",
            "efficiency",
            "repeat_count",
        ],
    )
    scale_plot = plots / "upmem_strong_scaling.png"
    dpu_scaling = [
        row
        for row in scaling
        if row["scale_dimension"] == "dpu"
        and row["fully_active_topology"] is True
        and row["active_rank_count"] == 1
    ]
    dpu_plot_rows = [
        dict(row, _plot_series=f"{row['path']} | {row['numeric_policy']}")
        for row in dpu_scaling
    ]
    dpu_title = (
        "Measured same-plan UPMEM DPU median scaling "
        f"(baseline time / runtime; >1 is faster; {execution_model})"
    )
    valid = _plot(
        scale_plot,
        dpu_title,
        dpu_plot_rows,
        "active_dpu_count",
        "speedup",
        "_plot_series",
        reference_y=1.0,
        panel_title="DPU scaling",
        semantic_group="_plot_series",
    )
    entries.append(
        _entry(
            "upmem_strong_scaling",
            scale_plot,
            scale_csv,
            dpu_title,
            valid,
            "no fully active one-rank rows with measured active DPU counts",
        )
    )
    rank_plot = plots / "upmem_rank_scaling.png"
    rank_scaling = [
        row
        for row in scaling
        if row["scale_dimension"] == "rank" and row["fully_active_topology"] is True
    ]
    rank_plot_rows = [
        dict(row, _plot_series=f"{row['path']} | {row['numeric_policy']}")
        for row in rank_scaling
    ]
    rank_title = (
        "Measured same-plan UPMEM rank median scaling "
        f"(baseline time / runtime; >1 is faster; {execution_model})"
    )
    valid = _plot(
        rank_plot,
        rank_title,
        rank_plot_rows,
        "rank_count",
        "speedup",
        "_plot_series",
        reference_y=1.0,
        panel_title="rank scaling",
        semantic_group="_plot_series",
    )
    entries.append(
        _entry(
            "upmem_rank_scaling",
            rank_plot,
            scale_csv,
            rank_title,
            valid,
            "no fully active multi-rank rows: active rank/DPU counts are below provisioned topology",
        )
    )

    path_rows = _variant_ratios(
        rows,
        "opt_einsum_greedy",
        "cotengra_flops_seed0",
        claim_rejections,
    )
    path_csv = tables / "path_runtime_ratio.csv"
    _write_csv(
        path_csv,
        path_rows,
        list(path_rows[0])
        if path_rows
        else [
            "case_id",
            "family",
            "engine",
            "numeric_policy",
            "timing_scope",
            "local_dpu_count",
            "rank_count",
            "total_dpu_count",
            "provisioned_dpu_count",
            "provisioned_rank_count",
            "active_dpu_count",
            "active_rank_count",
            "activity_label",
            "topology",
            "path_a",
            "path_b",
            "runtime_a_s",
            "runtime_b_s",
            "runtime_ratio_a_over_b",
            "repeat_count",
        ],
    )
    path_plot = plots / "path_runtime_ratio.png"
    path_title = (
        "Measured same-circuit/TN standard-path median ratio "
        "(greedy time / cotengra FLOPs time; >1 favors cotengra FLOPs; "
        f"{execution_model})"
    )
    valid = _plot(
        path_plot,
        path_title,
        [
            dict(
                row,
                _plot_series=f"{_visual_engine_label(row)} | {row['numeric_policy']}",
            )
            for row in path_rows
        ],
        "qubits",
        "runtime_ratio_a_over_b",
        "_plot_series",
        reference_y=1.0,
        panel_title="greedy / cotengra ratio",
        semantic_group="_plot_series",
    )
    entries.append(
        _entry(
            "path_runtime_ratio",
            path_plot,
            path_csv,
            path_title,
            valid,
            "no matched greedy and cotengra FLOPs path records",
        )
    )

    numeric_rows = _numeric_ratios(rows, claim_rejections)
    numeric_csv = tables / "float32_int8_ratios.csv"
    _write_csv(
        numeric_csv,
        numeric_rows,
        list(numeric_rows[0])
        if numeric_rows
        else [
            "case_id",
            "family",
            "engine",
            "path",
            "timing_scope",
            "local_dpu_count",
            "rank_count",
            "total_dpu_count",
            "provisioned_dpu_count",
            "provisioned_rank_count",
            "active_dpu_count",
            "active_rank_count",
            "activity_label",
            "topology",
            "series",
            "float32_runtime_s",
            "int8_runtime_s",
            "runtime_ratio_float32_over_int8",
        ],
    )
    numeric_plot = plots / "float32_int8_ratio.png"
    numeric_title = (
        "Measured same-plan Float32 / host-packed Int8 median ratio "
        f"(Float32 time / host-packed Int8 time; >1 favors host-packed Int8; {execution_model})"
    )
    if topology_dimension_ambiguous:
        _todo_plot(numeric_plot, numeric_title, topology_dimension_reason)
        valid = False
        numeric_reason = topology_dimension_reason
    else:
        valid = _plot(
            numeric_plot,
            numeric_title,
            [
                dict(
                    row,
                    _plot_series=f"{_visual_engine_label(row)} | {row['path']}",
                )
                for row in numeric_rows
            ],
            "qubits",
            "runtime_ratio_float32_over_int8",
            "_plot_series",
            reference_y=1.0,
            panel_title="Float32 / host-packed Int8 ratio",
            semantic_group="_plot_series",
        )
        numeric_reason = "no matched Float32 and host-packed Int8 records"
    entries.append(
        _entry(
            "float32_int8_ratio",
            numeric_plot,
            numeric_csv,
            numeric_title,
            valid,
            numeric_reason,
        )
    )

    validation_rows = _aggregate_medians(
        [_validation_row(row) for row in rows],
        group_fields=(
            "case_id",
            "family",
            "qubits",
            "engine",
            "route_id",
            "route_config_hash",
            "executor_config_hash",
            "path",
            "numeric_policy",
            "timing_scope",
            "provisioned_dpu_count",
            "provisioned_rank_count",
            "active_dpu_count",
            "active_rank_count",
            "activity_label",
            "status",
            "validation_status",
            "scientific_validation_status",
            "admission_identity",
            "scientific_admitted",
            "series",
        ),
        value_fields=("max_abs_error", "l2_error", "fidelity", "normalization_drift"),
        drop_fields=("repeat_id",),
    )
    validation_csv = tables / "validation_accuracy.csv"
    _write_csv(
        validation_csv,
        validation_rows,
        list(validation_rows[0])
        if validation_rows
        else [
            "case_id",
            "family",
            "engine",
            "route_id",
            "route_config_hash",
            "executor_config_hash",
            "path",
            "numeric_policy",
            "timing_scope",
            "provisioned_dpu_count",
            "provisioned_rank_count",
            "active_dpu_count",
            "active_rank_count",
            "activity_label",
            "status",
            "validation_status",
            "scientific_admitted",
            "max_abs_error",
            "l2_error",
            "fidelity",
            "normalization_drift",
            "repeat_count",
            "series",
        ],
    )
    validation_plot = plots / "validation_accuracy.png"
    valid = _plot(
        validation_plot,
        "M5.5 maximum absolute error against the full-precision reference",
        [
            dict(
                row,
                _visual_series=(
                    f"{_visual_engine_label(row)} | {row['path']} | {row['numeric_policy']}"
                ),
            )
            for row in validation_rows
            if row["scientific_admitted"]
        ],
        "qubits",
        "max_abs_error",
        "series",
        semantic_group="_visual_series",
    )
    entries.append(
        _entry(
            "validation_accuracy",
            validation_plot,
            validation_csv,
            "M5.5 maximum absolute error against the full-precision reference",
            valid,
            "no finite validation errors",
        )
    )

    timing_rows = _aggregate_medians(
        _timing_rows(admitted_rows),
        group_fields=(
            "case_id",
            "family",
            "qubits",
            "engine",
            "route_id",
            "route_config_hash",
            "executor_config_hash",
            "path",
            "numeric_policy",
            "timing_scope",
            "provisioned_dpu_count",
            "provisioned_rank_count",
            "active_dpu_count",
            "active_rank_count",
            "activity_label",
            "timing_coverage",
            "admission_identity",
            "stage",
            "series",
        ),
        value_fields=("time_s",),
        drop_fields=("repeat_id",),
    )
    timing_csv = tables / "timing_breakdown.csv"
    _write_csv(
        timing_csv,
        timing_rows,
        list(timing_rows[0])
        if timing_rows
        else [
            "case_id",
            "family",
            "engine",
            "route_id",
            "route_config_hash",
            "executor_config_hash",
            "path",
            "numeric_policy",
            "timing_scope",
            "active_dpu_count",
            "active_rank_count",
            "provisioned_dpu_count",
            "provisioned_rank_count",
            "activity_label",
            "timing_coverage",
            "stage",
            "time_s",
            "repeat_count",
            "series",
        ],
    )
    timing_plot = plots / "timing_breakdown.png"
    timing_title = (
        "M5.5 measured non-overlapping physical execution stages\n"
        "Medians across circuit families"
    )
    valid, timing_reason = _timing_breakdown_plot(
        timing_plot, timing_title, timing_rows
    )
    entries.append(
        _entry(
            "timing_breakdown",
            timing_plot,
            timing_csv,
            timing_title,
            valid,
            timing_reason,
        )
    )

    transfer_rows = _aggregate_medians(
        [_transfer_row(row) for row in rows],
        group_fields=(
            "case_id",
            "family",
            "qubits",
            "engine",
            "route_id",
            "route_config_hash",
            "executor_config_hash",
            "path",
            "numeric_policy",
            "timing_scope",
            "provisioned_dpu_count",
            "provisioned_rank_count",
            "active_dpu_count",
            "active_rank_count",
            "activity_label",
            "status",
            "validation_status",
            "scientific_validation_status",
            "admission_identity",
            "scientific_admitted",
            "series",
        ),
        value_fields=("h2d_bytes", "d2h_bytes", "transfer_bytes"),
        all_true_fields=(("invariant_passed", "raw_invariants_all_passed"),),
        drop_fields=("repeat_id",),
    )
    for row in transfer_rows:
        row["invariant_passed"] = row["raw_invariants_all_passed"]
        h2d = _float(row.get("h2d_bytes"))
        d2h = _float(row.get("d2h_bytes"))
        total = _float(row.get("transfer_bytes"))
        row["aggregate_component_medians_additive"] = (
            h2d is not None
            and d2h is not None
            and total is not None
            and math.isclose(h2d + d2h, total, rel_tol=0, abs_tol=1e-9)
        )
    transfer_csv = tables / "transfer_bytes.csv"
    _write_csv(
        transfer_csv,
        transfer_rows,
        list(transfer_rows[0])
        if transfer_rows
        else [
            "case_id",
            "engine",
            "route_id",
            "route_config_hash",
            "executor_config_hash",
            "path",
            "numeric_policy",
            "timing_scope",
            "active_dpu_count",
            "active_rank_count",
            "provisioned_dpu_count",
            "provisioned_rank_count",
            "scientific_admitted",
            "activity_label",
            "h2d_bytes",
            "d2h_bytes",
            "transfer_bytes",
            "invariant_passed",
            "raw_invariants_all_passed",
            "aggregate_component_medians_additive",
            "repeat_count",
            "series",
        ],
    )
    transfer_plot = plots / "transfer_bytes.png"
    transfer_title = "M5.5 application-visible software-recorded transfer bytes"
    if topology_dimension_ambiguous:
        _todo_plot(transfer_plot, transfer_title, topology_dimension_reason)
        valid = False
        transfer_reason = topology_dimension_reason
    else:
        valid = _plot(
            transfer_plot,
            transfer_title,
            [
                dict(
                    row,
                    _visual_series=(
                        f"{_visual_engine_label(row)} | {row['path']} | {row['numeric_policy']}"
                    ),
                )
                for row in transfer_rows
                if row["scientific_admitted"] and row["raw_invariants_all_passed"]
            ],
            "qubits",
            "transfer_bytes",
            "series",
            semantic_group="_visual_series",
        )
        transfer_reason = "no finite transfer-byte records"
    entries.append(
        _entry(
            "transfer_bytes",
            transfer_plot,
            transfer_csv,
            transfer_title,
            valid,
            transfer_reason,
        )
    )

    boundary_rows = [
        {
            "case_id": _case(row),
            "family": _family(row),
            "qubits": _qubits(row),
            "engine": _engine(row),
            "status": row.get("status", "completed"),
            "supported": _valid(row),
        }
        for row in rows
    ]
    boundary_csv = tables / "supported_boundary.csv"
    _write_csv(
        boundary_csv,
        boundary_rows,
        ["case_id", "family", "qubits", "engine", "status", "supported"],
    )
    boundary_plot = plots / "supported_boundary.png"
    valid = _plot(
        boundary_plot,
        "M5.5 supported boundary",
        [row for row in boundary_rows if row["qubits"] is not None],
        "qubits",
        "supported",
        "engine",
        collapse_activity=False,
    )
    entries.append(
        _entry(
            "supported_boundary",
            boundary_plot,
            boundary_csv,
            "Supported boundary",
            valid,
            "no qubit-size boundary records",
        )
    )

    energy_csv = tables / "energy.csv"
    _write_csv(energy_csv, [], ["case_id", "engine", "energy_joules", "energy_status"])
    energy_plot = plots / "energy_todo.png"
    _todo_plot(energy_plot, "Energy efficiency", "TODO: measured energy is unavailable")
    entries.append(
        _entry(
            "energy",
            energy_plot,
            energy_csv,
            "Energy efficiency",
            False,
            "measured energy is unavailable",
        )
    )

    rejection_fields = (
        "claim",
        "case_id",
        "baseline_route",
        "candidate_route",
        "repeat_id",
        "reasons",
    )
    unique_rejections = [
        dict(zip(rejection_fields, identity, strict=True))
        for identity in sorted(
            {
                tuple(str(row.get(field, "")) for field in rejection_fields)
                for row in claim_rejections
            }
        )
    ]
    _write_csv(
        tables / "claim_rejections.csv",
        unique_rejections,
        list(rejection_fields),
    )

    manifest = PlotManifest("m5_circuit_report_v1", tuple(entries))
    (output / "plot_manifest.json").write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return ReportResult(output, manifest)


def _variant_ratios(
    rows: list[JsonDict],
    path_a: str,
    path_b: str,
    claim_rejections: list[JsonDict] | None = None,
) -> list[JsonDict]:
    groups: dict[tuple[Any, ...], dict[str, list[JsonDict]]] = {}
    for row in rows:
        if (
            not _performance_valid(row)
            or _engine_class(row) == "cross_algorithm"
            or _path(row) not in {path_a, path_b}
        ):
            continue
        circuit_hash = str(row.get("circuit_semantics_hash") or "")
        network_hash = str(row.get("tensor_network_hash") or "")
        executor_hash = _executor_config_hash(row)
        if not circuit_hash or not network_hash or executor_hash is None:
            continue
        key = (
            circuit_hash,
            network_hash,
            executor_hash,
            _case(row),
            _engine(row),
            _numeric(row),
            _timing_scope(row),
            _repeat(row),
            _local_dpu_count(row),
            _rank_count(row),
            _total_dpu_count(row),
            _active_dpu_count(row),
            _active_rank_count(row),
        )
        groups.setdefault(key, {}).setdefault(_path(row), []).append(row)
    result: list[JsonDict] = []
    for group in groups.values():
        if path_a not in group or path_b not in group:
            continue
        a_rows, b_rows = group[path_a], group[path_b]
        a_row = _unambiguous_variant(a_rows)
        b_row = _unambiguous_variant(b_rows)
        if a_row is None or b_row is None:
            baseline = a_rows[0]
            candidate = b_rows[0]
            _append_claim_rejection(
                claim_rejections,
                Claim.PATH_ABLATION,
                baseline,
                candidate,
                ClaimDecision(
                    False,
                    (
                        "ambiguous duplicate path variants: "
                        f"{path_a}={len(a_rows)}, {path_b}={len(b_rows)}",
                    ),
                ),
            )
            continue
        decision = _CLAIM_POLICY.evaluate_pair(Claim.PATH_ABLATION, a_row, b_row)
        if not decision.allowed:
            _append_claim_rejection(
                claim_rejections, Claim.PATH_ABLATION, a_row, b_row, decision
            )
            continue
        a, b = _runtime(a_row), _runtime(b_row)
        a_hashes, b_hashes = _hashes(a_row), _hashes(b_row)
        if (
            a
            and b
            and a_hashes is not None
            and b_hashes is not None
            and a_hashes[:2] == b_hashes[:2]
            and a_hashes[2] != b_hashes[2]
        ):
            result.append(
                {
                    "case_id": _case(a_row),
                    "family": _family(a_row),
                    "qubits": _qubits(a_row),
                    "engine": _engine(a_row),
                    "numeric_policy": _numeric(a_row),
                    "timing_scope": _timing_scope(a_row),
                    "repeat_id": _repeat(a_row),
                    "local_dpu_count": _local_dpu_count(a_row),
                    "rank_count": _rank_count(a_row),
                    "total_dpu_count": _total_dpu_count(a_row),
                    "provisioned_dpu_count": _total_dpu_count(a_row),
                    "provisioned_rank_count": _rank_count(a_row),
                    "active_dpu_count": _active_dpu_count(a_row),
                    "active_rank_count": _active_rank_count(a_row),
                    "activity_label": _activity_label(a_row),
                    "topology": _topology_label(a_row),
                    "executor_config_hash": _executor_config_hash(a_row),
                    "route_id_a": str(a_row.get("route_id") or ""),
                    "route_config_hash_a": str(a_row.get("route_config_hash") or ""),
                    "executor_config_hash_a": _executor_config_hash(a_row),
                    "admission_identity_a": _admission_identity(a_row),
                    "route_id_b": str(b_row.get("route_id") or ""),
                    "route_config_hash_b": str(b_row.get("route_config_hash") or ""),
                    "executor_config_hash_b": _executor_config_hash(b_row),
                    "admission_identity_b": _admission_identity(b_row),
                    "comparison_identity": _comparison_identity(a_row, b_row),
                    "circuit_semantics_hash": a_hashes[0],
                    "tensor_network_hash": a_hashes[1],
                    "contraction_plan_hash_a": a_hashes[2],
                    "contraction_plan_hash_b": b_hashes[2],
                    "path_a": path_a,
                    "path_b": path_b,
                    "series": (
                        f"{path_a} / {path_b} | "
                        f"{_activity_label(a_row) or _topology_label(a_row)}"
                    ),
                    "direction": "runtime_a_over_b; >1 favors path_b",
                    "runtime_a_s": a,
                    "runtime_b_s": b,
                    "runtime_ratio_a_over_b": a / b,
                }
            )
    return _aggregate_matched_rows(
        result,
        group_fields=(
            "case_id",
            "family",
            "qubits",
            "engine",
            "numeric_policy",
            "timing_scope",
            "local_dpu_count",
            "rank_count",
            "total_dpu_count",
            "provisioned_dpu_count",
            "provisioned_rank_count",
            "active_dpu_count",
            "active_rank_count",
            "activity_label",
            "topology",
            "executor_config_hash",
            "route_id_a",
            "route_config_hash_a",
            "executor_config_hash_a",
            "admission_identity_a",
            "route_id_b",
            "route_config_hash_b",
            "executor_config_hash_b",
            "admission_identity_b",
            "comparison_identity",
            "circuit_semantics_hash",
            "tensor_network_hash",
            "contraction_plan_hash_a",
            "contraction_plan_hash_b",
            "path_a",
            "path_b",
            "series",
            "direction",
        ),
        value_fields=("runtime_a_s", "runtime_b_s", "runtime_ratio_a_over_b"),
    )


def _numeric_ratios(
    rows: list[JsonDict], claim_rejections: list[JsonDict] | None = None
) -> list[JsonDict]:
    groups: dict[tuple[Any, ...], dict[str, list[JsonDict]]] = {}
    for row in rows:
        if not _performance_valid(row) or _engine_class(row) == "cross_algorithm":
            continue
        numeric = _numeric(row).lower()
        kind = (
            "int8"
            if "int8" in numeric or "quant" in numeric
            else "float32"
            if "float32" in numeric
            else "other"
        )
        if kind == "other":
            continue
        hashes = _hashes(row)
        executor_hash = _executor_config_hash(row)
        if hashes is None or executor_hash is None:
            continue
        key = (
            *hashes,
            executor_hash,
            _case(row),
            _engine(row),
            _path(row),
            _timing_scope(row),
            _repeat(row),
            _local_dpu_count(row),
            _rank_count(row),
            _total_dpu_count(row),
            _active_dpu_count(row),
            _active_rank_count(row),
        )
        groups.setdefault(key, {}).setdefault(kind, []).append(row)
    result: list[JsonDict] = []
    for group in groups.values():
        if "float32" not in group or "int8" not in group:
            continue
        float_rows, int8_rows = group["float32"], group["int8"]
        float_row = _unambiguous_variant(float_rows)
        int8_row = _unambiguous_variant(int8_rows)
        if float_row is None or int8_row is None:
            _append_claim_rejection(
                claim_rejections,
                Claim.NUMERIC_ABLATION,
                float_rows[0],
                int8_rows[0],
                ClaimDecision(
                    False,
                    (
                        "ambiguous duplicate numeric variants: "
                        f"float32={len(float_rows)}, int8={len(int8_rows)}",
                    ),
                ),
            )
            continue
        decision = _CLAIM_POLICY.evaluate_pair(
            Claim.NUMERIC_ABLATION, float_row, int8_row
        )
        if not decision.allowed:
            _append_claim_rejection(
                claim_rejections,
                Claim.NUMERIC_ABLATION,
                float_row,
                int8_row,
                decision,
            )
            continue
        a, b = _runtime(float_row), _runtime(int8_row)
        if a and b:
            result.append(
                {
                    "case_id": _case(float_row),
                    "family": _family(float_row),
                    "qubits": _qubits(float_row),
                    "engine": _engine(float_row),
                    "path": _path(float_row),
                    "timing_scope": _timing_scope(float_row),
                    "repeat_id": _repeat(float_row),
                    "local_dpu_count": _local_dpu_count(float_row),
                    "rank_count": _rank_count(float_row),
                    "total_dpu_count": _total_dpu_count(float_row),
                    "provisioned_dpu_count": _total_dpu_count(float_row),
                    "provisioned_rank_count": _rank_count(float_row),
                    "active_dpu_count": _active_dpu_count(float_row),
                    "active_rank_count": _active_rank_count(float_row),
                    "activity_label": _activity_label(float_row),
                    "topology": _topology_label(float_row),
                    "executor_config_hash": _executor_config_hash(float_row),
                    "route_id_float32": str(float_row.get("route_id") or ""),
                    "route_config_hash_float32": str(
                        float_row.get("route_config_hash") or ""
                    ),
                    "executor_config_hash_float32": _executor_config_hash(float_row),
                    "admission_identity_float32": _admission_identity(float_row),
                    "route_id_int8": str(int8_row.get("route_id") or ""),
                    "route_config_hash_int8": str(
                        int8_row.get("route_config_hash") or ""
                    ),
                    "executor_config_hash_int8": _executor_config_hash(int8_row),
                    "admission_identity_int8": _admission_identity(int8_row),
                    "comparison_identity": _comparison_identity(float_row, int8_row),
                    "circuit_semantics_hash": float_row.get("circuit_semantics_hash"),
                    "tensor_network_hash": float_row.get("tensor_network_hash"),
                    "contraction_plan_hash": float_row.get("contraction_plan_hash"),
                    "series": (
                        f"{_path(float_row)} | "
                        f"{_activity_label(float_row) or _topology_label(float_row)}"
                    ),
                    "float32_runtime_s": a,
                    "int8_runtime_s": b,
                    "runtime_ratio_float32_over_int8": a / b,
                }
            )
    return _aggregate_matched_rows(
        result,
        group_fields=(
            "case_id",
            "family",
            "qubits",
            "engine",
            "path",
            "timing_scope",
            "local_dpu_count",
            "rank_count",
            "total_dpu_count",
            "provisioned_dpu_count",
            "provisioned_rank_count",
            "active_dpu_count",
            "active_rank_count",
            "activity_label",
            "topology",
            "executor_config_hash",
            "route_id_float32",
            "route_config_hash_float32",
            "executor_config_hash_float32",
            "admission_identity_float32",
            "route_id_int8",
            "route_config_hash_int8",
            "executor_config_hash_int8",
            "admission_identity_int8",
            "comparison_identity",
            "circuit_semantics_hash",
            "tensor_network_hash",
            "contraction_plan_hash",
        ),
        value_fields=(
            "float32_runtime_s",
            "int8_runtime_s",
            "runtime_ratio_float32_over_int8",
        ),
    )


def _validation_row(row: Mapping[str, Any]) -> JsonDict:
    accuracy = (
        row.get("full_precision_accuracy")
        if isinstance(row.get("full_precision_accuracy"), Mapping)
        else {}
    )
    max_abs = _float(_first(row, "max_abs_error", "validation_max_abs_error"))
    l2 = _float(_first(row, "l2_error", "validation_l2_error"))
    return {
        "case_id": _case(row),
        "family": _family(row),
        "qubits": _qubits(row),
        "engine": _engine(row),
        "route_id": str(row.get("route_id") or ""),
        "route_config_hash": str(row.get("route_config_hash") or ""),
        "executor_config_hash": _executor_config_hash(row),
        "path": _path(row),
        "numeric_policy": _numeric(row),
        "timing_scope": _timing_scope(row),
        "activity_label": _activity_label(row),
        "active_dpu_count": _active_dpu_count(row),
        "active_rank_count": _active_rank_count(row),
        "provisioned_dpu_count": _total_dpu_count(row),
        "provisioned_rank_count": _rank_count(row),
        "status": row.get("status"),
        "validation_status": row.get("validation_status"),
        "scientific_validation_status": row.get("scientific_validation_status"),
        "admission_identity": _admission_identity(row),
        "scientific_admitted": _valid(row),
        "max_abs_error": max_abs
        if max_abs is not None
        else _float(accuracy.get("max_abs_error")),
        "l2_error": l2 if l2 is not None else _float(accuracy.get("l2_error")),
        "fidelity": _float(_first(row, "fidelity", "validation_fidelity")),
        "normalization_drift": _float(
            _first(row, "normalization_drift", "validation_normalization_drift")
        ),
        "series": _identity_series(row),
    }


def _timing_rows(rows: list[JsonDict]) -> list[JsonDict]:
    result = []
    leaf_stages = (
        ("planning", ("planning_time_s", "planning_s", "planning")),
        ("session_open", ("session_open_s", "session_open_time_s")),
        (
            "host_quantization",
            ("host_quantization_time_s", "host_quantization_s"),
        ),
        ("h2d", ("h2d_time_s", "h2d_s")),
        ("d2h", ("d2h_time_s", "d2h_s")),
        ("assembly", ("assembly_time_s", "assembly_s")),
        (
            "host_dequantization",
            ("host_dequantization_time_s", "host_dequantization_s"),
        ),
        ("validation", ("validation_time_s", "validation_s")),
        ("session_close", ("session_close_s", "session_close_time_s")),
    )
    kernel_aliases = (
        "dpu_kernel_time_s",
        "dpu_kernel_s",
        "kernel_time_s",
        "kernel_s",
        "launch_time_s",
    )
    execution_stage_aliases = (
        ("host_quantization_time_s", "host_quantization_s"),
        ("h2d_time_s", "h2d_s"),
        kernel_aliases,
        ("d2h_time_s", "d2h_s"),
        ("assembly_time_s", "assembly_s"),
        ("host_dequantization_time_s", "host_dequantization_s"),
    )

    def stage_value(
        breakdown: Mapping[str, Any], row: Mapping[str, Any], aliases: tuple[str, ...]
    ) -> Any:
        value = _first(breakdown, *aliases)
        return value if value is not None else _first(row, *aliases)

    for row in rows:
        breakdown = (
            row.get("timing_breakdown") or row.get("timings") or row.get("timing")
        )
        if not isinstance(breakdown, Mapping):
            breakdown = {}
        timing_coverage = (
            "execution_stage_leaves"
            if any(
                _float(stage_value(breakdown, row, aliases)) is not None
                for aliases in execution_stage_aliases
            )
            else "lifecycle_only"
        )
        for stage, aliases in leaf_stages:
            value = stage_value(breakdown, row, aliases)
            number = _float(value)
            if number is not None:
                result.append(
                    {
                        "case_id": _case(row),
                        "family": _family(row),
                        "qubits": _qubits(row),
                        "engine": _engine(row),
                        "route_id": str(row.get("route_id") or ""),
                        "route_config_hash": str(row.get("route_config_hash") or ""),
                        "executor_config_hash": _executor_config_hash(row),
                        "path": _path(row),
                        "numeric_policy": _numeric(row),
                        "timing_scope": _timing_scope(row),
                        "timing_coverage": timing_coverage,
                        "activity_label": _activity_label(row),
                        "active_dpu_count": _active_dpu_count(row),
                        "active_rank_count": _active_rank_count(row),
                        "provisioned_dpu_count": _total_dpu_count(row),
                        "provisioned_rank_count": _rank_count(row),
                        "stage": stage,
                        "time_s": number,
                        "admission_identity": _admission_identity(row),
                        "series": _identity_series(row, include_stage=stage),
                    }
                )
    for row in rows:
        breakdown = (
            row.get("timing_breakdown") or row.get("timings") or row.get("timing")
        )
        if not isinstance(breakdown, Mapping):
            breakdown = {}
        timing_coverage = (
            "execution_stage_leaves"
            if any(
                _float(stage_value(breakdown, row, aliases)) is not None
                for aliases in execution_stage_aliases
            )
            else "lifecycle_only"
        )
        kernel = _float(stage_value(breakdown, row, kernel_aliases))
        if kernel is not None:
            result.append(
                {
                    "case_id": _case(row),
                    "family": _family(row),
                    "qubits": _qubits(row),
                    "engine": _engine(row),
                    "route_id": str(row.get("route_id") or ""),
                    "route_config_hash": str(row.get("route_config_hash") or ""),
                    "executor_config_hash": _executor_config_hash(row),
                    "path": _path(row),
                    "numeric_policy": _numeric(row),
                    "timing_scope": _timing_scope(row),
                    "timing_coverage": timing_coverage,
                    "activity_label": _activity_label(row),
                    "active_dpu_count": _active_dpu_count(row),
                    "active_rank_count": _active_rank_count(row),
                    "provisioned_dpu_count": _total_dpu_count(row),
                    "provisioned_rank_count": _rank_count(row),
                    "stage": "kernel",
                    "time_s": kernel,
                    "admission_identity": _admission_identity(row),
                    "series": _identity_series(row, include_stage="kernel"),
                }
            )
    return result


def _transfer_row(row: Mapping[str, Any]) -> JsonDict:
    raw_transfers = row.get("transfer", row.get("transfers"))
    transfers = raw_transfers if isinstance(raw_transfers, Mapping) else {}
    h2d = _float(_first(row, "actual_h2d_bytes", "application_visible_h2d_bytes"))
    d2h = _float(_first(row, "actual_d2h_bytes", "application_visible_d2h_bytes"))
    total = _float(
        _first(row, "actual_transfer_bytes", "application_visible_transfer_bytes")
    )
    if h2d is None:
        h2d = _float(_first(transfers, "application_visible_h2d_bytes", "h2d_bytes"))
    if d2h is None:
        d2h = _float(_first(transfers, "application_visible_d2h_bytes", "d2h_bytes"))
    if total is None:
        total = _float(
            _first(
                transfers,
                "application_visible_transfer_bytes",
                "transfer_bytes",
                "total_bytes",
            )
        )
    arithmetic_invariant = (
        h2d is not None
        and d2h is not None
        and total is not None
        and math.isclose(total, h2d + d2h, rel_tol=0, abs_tol=1e-9)
    )
    recorded_verification = row.get("transfer_accounting_verified")
    invariant = arithmetic_invariant and (
        recorded_verification is True if recorded_verification is not None else True
    )
    return {
        "case_id": _case(row),
        "family": _family(row),
        "qubits": _qubits(row),
        "engine": _engine(row),
        "route_id": str(row.get("route_id") or ""),
        "route_config_hash": str(row.get("route_config_hash") or ""),
        "executor_config_hash": _executor_config_hash(row),
        "path": _path(row),
        "numeric_policy": _numeric(row),
        "timing_scope": _timing_scope(row),
        "activity_label": _activity_label(row),
        "active_dpu_count": _active_dpu_count(row),
        "active_rank_count": _active_rank_count(row),
        "provisioned_dpu_count": _total_dpu_count(row),
        "provisioned_rank_count": _rank_count(row),
        "scientific_admitted": _valid(row),
        "status": row.get("status"),
        "validation_status": row.get("validation_status"),
        "scientific_validation_status": row.get("scientific_validation_status"),
        "admission_identity": _admission_identity(row),
        "h2d_bytes": h2d,
        "d2h_bytes": d2h,
        "transfer_bytes": total,
        "invariant_passed": invariant,
        "series": _identity_series(row),
    }
