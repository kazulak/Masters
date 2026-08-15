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


JsonDict = dict[str, Any]


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
    return (
        f"{active_dpus}/{provisioned_dpus}DPU + "
        f"{active_ranks}/{provisioned_ranks}R"
    )


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
    status = str(row.get("status", "completed")).lower()
    if (
        status not in {"completed", "passed", "success", "verified"}
        or _runtime(row) is None
    ):
        return False
    engine_class = _engine_class(row)
    if engine_class == "cross_algorithm":
        return str(row.get("validation_status") or "").lower() in {
            "passed",
            "passed_native_status",
            "passed_runtime_only",
        }
    if engine_class not in {"cpu", "upmem"}:
        return False
    if (
        str(row.get("scientific_validation_status") or "").lower() != "passed"
        or row.get("exact_once") is not True
        or row.get("no_fallback_used") is not True
    ):
        return False
    if engine_class == "cpu":
        return True
    return (
        row.get("target_observed") == "physical_hardware"
        and row.get("hardware_allocation_verified") is True
        and row.get("native_kernel_executed") is True
        and row.get("hardware_kernel_executed") is True
        and _explicitly_false(row, "simulator", "simulator_kernel_executed")
        and _explicitly_false(row, "cpu_fallback", "cpu_fallback_used")
        and _release_succeeded(row)
    )


def _performance_valid(row: Mapping[str, Any]) -> bool:
    """Return whether a row may support a measured performance ratio.

    A physically executed row can be useful functionality evidence while its
    timing is still explicitly bring-up-only.  Such rows remain in raw,
    runtime, validation, and transfer tables, but cannot support speedup or
    comparative runtime claims.
    """
    if not _valid(row):
        return False
    if _engine_class(row) != "upmem":
        return True
    return (
        row.get("hardware_speedup_applicable") is True
        and row.get("timing_is_bringup_only") is False
    )


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
        "executor_config_hash": _executor_config_hash(row),
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
            normalized["executor_config_hash"],
            normalized["local_dpu_count"],
            normalized["rank_count"],
            normalized["total_dpu_count"],
            normalized["provisioned_dpu_count"],
            normalized["provisioned_rank_count"],
            normalized["active_dpu_count"],
            normalized["active_rank_count"],
            normalized["activity_label"],
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
                    "executor_config_hash",
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
    rows: list[JsonDict], left_class: str, right_class: str
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
            pairs.append({left_class: baseline, right_class: right})
    return pairs


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
            return "UPMEM", (int(match.group(2)), int(match.group(1)))
        if lowered in {"numpy_cpu", "cpu_numpy"}:
            return "CPU", None
        if "upmem" in lowered or "dpu" in lowered:
            return "UPMEM", None
        if lowered == "quest_cpu_full_state":
            return "QuEST CPU", None
        if lowered == "quimb_tn":
            return "Quimb TN", None
        return None, None

    def compact_token(part: str) -> str:
        replacements = (
            ("opt_einsum_greedy", "greedy"),
            ("cotengra_flops_seed0", "cotengra"),
            ("host_packed_int8", "int8"),
            ("float32_real", "f32_real"),
            ("real_float32", "f32_real"),
            ("float32", "f32"),
            ("cpu_numpy", "CPU"),
            ("numpy_cpu", "CPU"),
            ("upmem_m5", "UPMEM"),
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
        dpu_text = (
            f"{active_dpus[0]} active of {provisioned_dpus} provisioned DPU"
        )
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
    collapse_activity: bool = True,
) -> bool:
    if not rows:
        return False
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    by_family: dict[
        str, dict[tuple[str, tuple[int, int] | None], list[tuple[float, float, JsonDict]]]
    ] = {}
    for row in rows:
        xv, yv = _float(row.get(x)), _float(row.get(y))
        if xv is None or yv is None:
            continue
        family = str(row.get("family") or "unknown")
        series_key = _plot_series_key(
            row, group, collapse_activity=collapse_activity
        )
        by_family.setdefault(family, {}).setdefault(series_key, []).append(
            (xv, yv, row)
        )
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
    display_labels = {}
    for family in families:
        for (base_series, topology), points in by_family[family].items():
            display_labels[(family, (base_series, topology))] = _plot_display_label(
                family,
                base_series,
                topology,
                [point[2] for point in points],
            )
    # Resolve collisions only within a family. The same semantic series in
    # different facets must retain one shared label and color.
    raw_by_family_label: dict[tuple[str, str], set[tuple[str, tuple[int, int] | None]]] = {}
    for (family, series), label in display_labels.items():
        raw_by_family_label.setdefault((family, label), set()).add(series)
    for key, label in list(display_labels.items()):
        family, series = key
        raw_values = sorted(raw_by_family_label[(family, label)], key=str)
        if len(raw_values) > 1:
            display_labels[key] = f"{label} ({raw_values.index(series) + 1})"
    short_labels = sorted(set(display_labels.values()))
    colors = {
        label: plt.get_cmap("tab10")(index % 10)
        for index, label in enumerate(short_labels)
    }
    handles: dict[str, Any] = {}
    metric_title = panel_title or y.replace("_", " ").title()
    for index, family in enumerate(families):
        axis = axes[index // ncols][index % ncols]
        for series, points in sorted(by_family[family].items(), key=lambda item: str(item[0])):
            points.sort(key=lambda point: point[0])
            label = display_labels[(family, series)]
            (line,) = axis.plot(
                [point[0] for point in points],
                [point[1] for point in points],
                marker="o",
                color=colors[label],
                label=label,
            )
            handles.setdefault(label, line)
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
    if handles:
        ordered = sorted(handles)
        legend_columns = min(4, len(ordered))
        legend_rows = math.ceil(len(ordered) / legend_columns)
        fig.legend(
            [handles[series] for series in ordered],
            ordered,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.945),
            ncol=min(4, len(ordered)),
            fontsize="small",
            frameon=False,
        )
    # Explicit margins keep the external legend and panel titles clear without
    # relying on tight_layout, which is unstable for dynamic facet grids.
    fig.subplots_adjust(
        left=0.08,
        right=0.98,
        bottom=0.12,
        top=(max(0.20, min(0.86, 0.88 - 0.055 * legend_rows)))
        if handles
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
    collapse_activity: bool = True,
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
        collapse_activity=collapse_activity,
    )


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
    qubits = {
        value
        for row in rows
        if (value := _qubits(row)) is not None
    }
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
            runtime_summaries,
            "qubits",
            "median_runtime_s",
            "series",
            log_y=True,
            panel_title="median runtime (s)",
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

    pairs = _pair_rows(rows, "cpu", "upmem")
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
        speedups,
        "qubits",
        "speedup_cpu_over_upmem",
        "series",
        reference_y=1.0,
        panel_title="CPU / UPMEM speedup",
    )
    entries.append(
        _entry(
            "same_plan_cpu_upmem_speedup",
            speed_plot,
            speed_csv,
            speed_title,
            valid,
            "no matching CPU/UPMEM rows with all hashes and timing_scope",
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
            if rank_count == 1:
                scale_dimension = "dpu"
                baseline_local = 1
                baseline_ranks = 1
                baseline_total = 1
                baseline_time = dpu_base_time
                scale_value = active_dpus
                speedup = (
                    baseline_time / runtime
                    if fully_active
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
                scale_value = active_ranks
                speedup = (
                    baseline_time / runtime
                    if fully_active
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
        collapse_activity=False,
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
        collapse_activity=False,
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

    path_rows = _variant_ratios(rows, "opt_einsum_greedy", "cotengra_flops_seed0")
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
        "(opt_einsum_greedy time / cotengra_flops_seed0 time; >1 favors cotengra_flops_seed0; "
        f"{execution_model})"
    )
    valid = _plot(
        path_plot,
        path_title,
        [
            dict(
                row,
                _plot_series=(
                    f"{row['engine']} | {row['numeric_policy']} | "
                    f"{row['activity_label'] or row['topology']}"
                ),
            )
            for row in path_rows
        ],
        "qubits",
        "runtime_ratio_a_over_b",
        "_plot_series",
        reference_y=1.0,
        panel_title="greedy / cotengra ratio",
    )
    entries.append(
        _entry(
            "path_runtime_ratio",
            path_plot,
            path_csv,
            path_title,
            valid,
            "no matched opt_einsum_greedy and cotengra_flops_seed0 path records",
        )
    )

    numeric_rows = _numeric_ratios(rows)
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
        "Measured same-plan float32/int8 median ratio "
        f"(float32 time / int8 time; >1 favors int8; {execution_model})"
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
                    _plot_series=(
                        f"{row['engine']} | {row['path']} | "
                        f"{row['activity_label'] or row['topology']}"
                    ),
                )
                for row in numeric_rows
            ],
            "qubits",
            "runtime_ratio_float32_over_int8",
            "_plot_series",
            reference_y=1.0,
            panel_title="float32 / int8 ratio",
        )
        numeric_reason = "no matched float32 and host-packed-int8 records"
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

    validation_rows = [_validation_row(row) for row in rows]
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
            "numeric_policy",
            "status",
            "validation_status",
            "max_abs_error",
            "l2_error",
            "fidelity",
            "normalization_drift",
        ],
    )
    validation_plot = plots / "validation_accuracy.png"
    valid = _plot(
        validation_plot,
        "M5.5 validation maximum absolute error",
        [row for row in validation_rows if row["scientific_admitted"]],
        "qubits",
        "max_abs_error",
        "engine",
    )
    entries.append(
        _entry(
            "validation_accuracy",
            validation_plot,
            validation_csv,
            "Validation accuracy",
            valid,
            "no finite validation errors",
        )
    )

    timing_rows = _timing_rows(admitted_rows)
    timing_csv = tables / "timing_breakdown.csv"
    _write_csv(
        timing_csv,
        timing_rows,
        list(timing_rows[0])
        if timing_rows
        else [
            "case_id",
            "engine",
            "active_dpu_count",
            "active_rank_count",
            "provisioned_dpu_count",
            "provisioned_rank_count",
            "activity_label",
            "stage",
            "time_s",
            "series",
        ],
    )
    timing_plot = plots / "timing_breakdown.png"
    timing_title = "M5.5 timing breakdown"
    if topology_dimension_ambiguous:
        _todo_plot(timing_plot, timing_title, topology_dimension_reason)
        valid = False
        timing_reason = topology_dimension_reason
    else:
        valid = _plot(
            timing_plot,
            timing_title,
            timing_rows,
            "qubits",
            "time_s",
            "series",
        )
        timing_reason = "no timing-stage fields"
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

    transfer_rows = [_transfer_row(row) for row in rows]
    transfer_csv = tables / "transfer_bytes.csv"
    _write_csv(
        transfer_csv,
        transfer_rows,
        list(transfer_rows[0])
        if transfer_rows
        else [
            "case_id",
            "engine",
            "active_dpu_count",
            "active_rank_count",
            "provisioned_dpu_count",
            "provisioned_rank_count",
            "activity_label",
            "h2d_bytes",
            "d2h_bytes",
            "transfer_bytes",
            "invariant_passed",
            "series",
        ],
    )
    transfer_plot = plots / "transfer_bytes.png"
    transfer_title = "Application-visible H2D/D2H transfer bytes"
    if topology_dimension_ambiguous:
        _todo_plot(transfer_plot, transfer_title, topology_dimension_reason)
        valid = False
        transfer_reason = topology_dimension_reason
    else:
        valid = _plot(
            transfer_plot,
            transfer_title,
            [row for row in transfer_rows if row["scientific_admitted"]],
            "qubits",
            "transfer_bytes",
            "series",
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

    manifest = PlotManifest("m5_circuit_report_v1", tuple(entries))
    (output / "plot_manifest.json").write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return ReportResult(output, manifest)


def _variant_ratios(rows: list[JsonDict], path_a: str, path_b: str) -> list[JsonDict]:
    groups: dict[tuple[Any, ...], dict[str, JsonDict]] = {}
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
        groups.setdefault(key, {})[_path(row)] = row
    result: list[JsonDict] = []
    for group in groups.values():
        if path_a not in group or path_b not in group:
            continue
        a_row, b_row = group[path_a], group[path_b]
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


def _numeric_ratios(rows: list[JsonDict]) -> list[JsonDict]:
    groups: dict[tuple[Any, ...], dict[str, JsonDict]] = {}
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
        groups.setdefault(key, {})[kind] = row
    result: list[JsonDict] = []
    for group in groups.values():
        if "float32" not in group or "int8" not in group:
            continue
        a, b = _runtime(group["float32"]), _runtime(group["int8"])
        if a and b:
            result.append(
                {
                    "case_id": _case(group["float32"]),
                    "family": _family(group["float32"]),
                    "qubits": _qubits(group["float32"]),
                    "engine": _engine(group["float32"]),
                    "path": _path(group["float32"]),
                    "timing_scope": _timing_scope(group["float32"]),
                    "repeat_id": _repeat(group["float32"]),
                    "local_dpu_count": _local_dpu_count(group["float32"]),
                    "rank_count": _rank_count(group["float32"]),
                    "total_dpu_count": _total_dpu_count(group["float32"]),
                    "provisioned_dpu_count": _total_dpu_count(group["float32"]),
                    "provisioned_rank_count": _rank_count(group["float32"]),
                    "active_dpu_count": _active_dpu_count(group["float32"]),
                    "active_rank_count": _active_rank_count(group["float32"]),
                    "activity_label": _activity_label(group["float32"]),
                    "topology": _topology_label(group["float32"]),
                    "executor_config_hash": _executor_config_hash(group["float32"]),
                    "circuit_semantics_hash": group["float32"].get(
                        "circuit_semantics_hash"
                    ),
                    "tensor_network_hash": group["float32"].get("tensor_network_hash"),
                    "contraction_plan_hash": group["float32"].get(
                        "contraction_plan_hash"
                    ),
                    "series": (
                        f"{_path(group['float32'])} | "
                        f"{_activity_label(group['float32']) or _topology_label(group['float32'])}"
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
        "numeric_policy": _numeric(row),
        "status": row.get("status"),
        "validation_status": row.get("validation_status"),
        "scientific_validation_status": row.get("scientific_validation_status"),
        "scientific_admitted": _valid(row),
        "max_abs_error": max_abs
        if max_abs is not None
        else _float(accuracy.get("max_abs_error")),
        "l2_error": l2 if l2 is not None else _float(accuracy.get("l2_error")),
        "fidelity": _float(_first(row, "fidelity", "validation_fidelity")),
        "normalization_drift": _float(
            _first(row, "normalization_drift", "validation_normalization_drift")
        ),
    }


def _timing_rows(rows: list[JsonDict]) -> list[JsonDict]:
    result = []
    known = {
        "planning": ("planning_time_s", "planning_s", "planning"),
        "session_open": ("session_open_s",),
        "host_quantization": ("host_quantization_time_s", "host_quantization_s"),
        "h2d": ("h2d_time_s", "h2d_s"),
        "kernel": ("kernel_time_s", "kernel_s", "launch_time_s"),
        "dpu_kernel": ("dpu_kernel_time_s", "dpu_kernel_s"),
        "d2h": ("d2h_time_s", "d2h_s"),
        "assembly": ("assembly_time_s", "assembly_s"),
        "host_dequantization": (
            "host_dequantization_time_s",
            "host_dequantization_s",
        ),
        "graph_execution": ("graph_execution_s",),
        "validation": ("validation_time_s", "validation_s"),
        "session_close": ("session_close_s",),
        "total": (
            "total_route_time_s",
            "total_time_s",
            "total_s",
            "total",
            "timing_s",
        ),
    }
    for row in rows:
        breakdown = (
            row.get("timing_breakdown") or row.get("timings") or row.get("timing")
        )
        if not isinstance(breakdown, Mapping):
            breakdown = {}
        for stage, aliases in known.items():
            value = _first(breakdown, *aliases)
            if value is None:
                value = _first(row, *aliases)
            number = _float(value)
            if number is not None:
                result.append(
                    {
                        "case_id": _case(row),
                        "family": _family(row),
                        "qubits": _qubits(row),
                        "engine": _engine(row),
                        "activity_label": _activity_label(row),
                        "active_dpu_count": _active_dpu_count(row),
                        "active_rank_count": _active_rank_count(row),
                        "provisioned_dpu_count": _total_dpu_count(row),
                        "provisioned_rank_count": _rank_count(row),
                        "stage": stage,
                        "time_s": number,
                        "series": (
                            f"{_engine(row)} | "
                            f"{_activity_label(row) or 'host'} | {stage}"
                        ),
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
        "activity_label": _activity_label(row),
        "active_dpu_count": _active_dpu_count(row),
        "active_rank_count": _active_rank_count(row),
        "provisioned_dpu_count": _total_dpu_count(row),
        "provisioned_rank_count": _rank_count(row),
        "scientific_admitted": _valid(row),
        "h2d_bytes": h2d,
        "d2h_bytes": d2h,
        "transfer_bytes": total,
        "invariant_passed": invariant,
        "series": f"{_engine(row)} | {_activity_label(row) or 'host'}",
    }
