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
import statistics
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
    requested = _int(
        _first(row, "requested_dpu_count", "allocated_dpu_count", "dpu_count")
    )
    if ranks in {None, 1} and requested is not None and requested > 0:
        return requested
    total = _int(_first(row, "total_dpu_count", "allocated_total_dpu_count"))
    if ranks and total and total % ranks == 0:
        return total // ranks
    return None


def _total_dpu_count(row: Mapping[str, Any]) -> int | None:
    explicit = _int(_first(row, "total_dpu_count", "allocated_total_dpu_count"))
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
        "series": f"{_family(row)} | {_engine(row)} | {_path(row)} | {_numeric(row)}{topology}",
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


def _plot(
    path: Path,
    title: str,
    rows: list[JsonDict],
    x: str,
    y: str,
    group: str,
    *,
    reference_y: float | None = None,
) -> bool:
    if not rows:
        return False
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    grouped: dict[str, list[tuple[float, float]]] = {}
    for row in rows:
        xv, yv = _float(row.get(x)), _float(row.get(y))
        if xv is None or yv is None:
            continue
        grouped.setdefault(str(row.get(group, "unknown")), []).append((xv, yv))
    if not grouped:
        return False
    fig, axis = plt.subplots(figsize=(8, 4.5))
    for label, points in sorted(grouped.items()):
        points.sort()
        axis.plot(
            [point[0] for point in points],
            [point[1] for point in points],
            marker="o",
            label=label,
        )
    if reference_y is not None:
        axis.axhline(reference_y, color="black", linestyle="--", linewidth=1, alpha=0.7)
    axis.set_title(title)
    axis.set_xlabel(x.replace("_", " ").title())
    axis.set_ylabel(y.replace("_", " ").title())
    axis.grid(True, alpha=0.25)
    if len(grouped) > 1:
        axis.legend(fontsize="small")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return True


def _todo_plot(path: Path, title: str, reason: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(8, 4.5))
    axis.text(
        0.5,
        0.5,
        f"TODO\n{reason}",
        ha="center",
        va="center",
        fontsize=14,
        transform=axis.transAxes,
    )
    axis.set_title(title)
    axis.set_xticks([])
    axis.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _entry(
    name: str, plot: Path, csv_path: Path, title: str, valid: bool, reason: str = ""
) -> PlotEntry:
    return PlotEntry(
        name,
        str(plot.name),
        str(csv_path.relative_to(csv_path.parents[1])),
        "generated_valid" if valid else "generated_todo_missing_data",
        title,
        None if valid else reason,
    )


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
            "median_runtime_s",
            "repeat_count",
        ],
    )
    runtime_plot = plots / "runtime_by_qubits.png"
    execution_model = (
        "sequential TaskGraph task scheduling; intra-task DPU tile/rank parallelism"
    )
    runtime_title = f"M5.5 whole-circuit median runtime ({execution_model})"
    valid = _plot(
        runtime_plot,
        runtime_title,
        runtime_summaries,
        "qubits",
        "median_runtime_s",
        "series",
    )
    entries.append(
        _entry(
            "runtime_by_qubits",
            runtime_plot,
            runtime_summary_csv,
            runtime_title,
            valid,
            "no valid runtime records",
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
                    "topology": _topology_label(upmem),
                    "series": f"{_path(upmem)} | {_numeric(upmem)} | {_topology_label(upmem)}",
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
            ),
            None,
        )
        dpu_base_time = _runtime(dpu_baseline) if dpu_baseline else None
        for row in group:
            local_dpus = _local_dpu_count(row)
            rank_count = _rank_count(row)
            total_dpus = _total_dpu_count(row)
            runtime = _runtime(row)
            if (
                dpu_base_time is None
                or rank_count != 1
                or local_dpus is None
                or local_dpus <= 1
                or total_dpus != local_dpus
                or runtime is None
            ):
                continue
            speedup = dpu_base_time / runtime
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
                    "scale_dimension": "dpu",
                    "local_dpu_count": local_dpus,
                    "rank_count": rank_count,
                    "total_dpu_count": total_dpus,
                    "topology": _topology_label(row),
                    "series": f"{_path(row)} | {_numeric(row)} | {_topology_label(row)}",
                    "baseline_local_dpu_count": 1,
                    "baseline_rank_count": 1,
                    "baseline_total_dpu_count": 1,
                    "baseline_runtime_s": dpu_base_time,
                    "runtime_s": runtime,
                    "speedup": speedup,
                    "efficiency": speedup / local_dpus,
                }
            )
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
                ),
                None,
            )
            rank_base_time = _runtime(rank_baseline) if rank_baseline else None
            if rank_base_time is None:
                continue
            for row in group:
                rank_count = _rank_count(row)
                total_dpus = _total_dpu_count(row)
                runtime = _runtime(row)
                if (
                    rank_count is None
                    or rank_count <= 1
                    or _local_dpu_count(row) != local_dpus
                    or total_dpus != local_dpus * rank_count
                    or runtime is None
                ):
                    continue
                speedup = rank_base_time / runtime
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
                        "scale_dimension": "rank",
                        "local_dpu_count": local_dpus,
                        "rank_count": rank_count,
                        "total_dpu_count": total_dpus,
                        "topology": _topology_label(row),
                        "series": f"{_path(row)} | {_numeric(row)} | {_topology_label(row)}",
                        "baseline_local_dpu_count": local_dpus,
                        "baseline_rank_count": 1,
                        "baseline_total_dpu_count": local_dpus,
                        "baseline_runtime_s": rank_base_time,
                        "runtime_s": runtime,
                        "speedup": speedup,
                        "efficiency": speedup / rank_count,
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
    dpu_scaling = [row for row in scaling if row["scale_dimension"] == "dpu"]
    dpu_title = (
        "Measured same-plan UPMEM DPU median scaling "
        f"(baseline time / runtime; >1 is faster; {execution_model})"
    )
    valid = _plot(
        scale_plot,
        dpu_title,
        dpu_scaling,
        "local_dpu_count",
        "speedup",
        "series",
        reference_y=1.0,
    )
    entries.append(
        _entry(
            "upmem_strong_scaling",
            scale_plot,
            scale_csv,
            dpu_title,
            valid,
            "no completed one-DPU baseline with larger-DPU rows",
        )
    )
    rank_plot = plots / "upmem_rank_scaling.png"
    rank_scaling = [row for row in scaling if row["scale_dimension"] == "rank"]
    rank_title = (
        "Measured same-plan UPMEM rank median scaling "
        f"(baseline time / runtime; >1 is faster; {execution_model})"
    )
    valid = _plot(
        rank_plot,
        rank_title,
        rank_scaling,
        "rank_count",
        "speedup",
        "series",
        reference_y=1.0,
    )
    entries.append(
        _entry(
            "upmem_rank_scaling",
            rank_plot,
            scale_csv,
            rank_title,
            valid,
            "no rank_count=1 baseline with equal local DPU count",
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
        path_rows,
        "qubits",
        "runtime_ratio_a_over_b",
        "series",
        reference_y=1.0,
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
    valid = _plot(
        numeric_plot,
        numeric_title,
        numeric_rows,
        "qubits",
        "runtime_ratio_float32_over_int8",
        "engine",
        reference_y=1.0,
    )
    entries.append(
        _entry(
            "float32_int8_ratio",
            numeric_plot,
            numeric_csv,
            numeric_title,
            valid,
            "no matched float32 and host-packed-int8 records",
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
        else ["case_id", "engine", "stage", "time_s"],
    )
    timing_plot = plots / "timing_breakdown.png"
    valid = _plot(
        timing_plot, "M5.5 timing breakdown", timing_rows, "qubits", "time_s", "stage"
    )
    entries.append(
        _entry(
            "timing_breakdown",
            timing_plot,
            timing_csv,
            "Timing breakdown",
            valid,
            "no timing-stage fields",
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
            "h2d_bytes",
            "d2h_bytes",
            "transfer_bytes",
            "invariant_passed",
        ],
    )
    transfer_plot = plots / "transfer_bytes.png"
    valid = _plot(
        transfer_plot,
        "Application-visible H2D/D2H transfer bytes",
        [row for row in transfer_rows if row["scientific_admitted"]],
        "qubits",
        "transfer_bytes",
        "engine",
    )
    entries.append(
        _entry(
            "transfer_bytes",
            transfer_plot,
            transfer_csv,
            "Transfer bytes",
            valid,
            "no finite transfer-byte records",
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
                    "topology": _topology_label(a_row),
                    "executor_config_hash": _executor_config_hash(a_row),
                    "circuit_semantics_hash": a_hashes[0],
                    "tensor_network_hash": a_hashes[1],
                    "contraction_plan_hash_a": a_hashes[2],
                    "contraction_plan_hash_b": b_hashes[2],
                    "path_a": path_a,
                    "path_b": path_b,
                    "series": f"{path_a} / {path_b}",
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
                    "topology": _topology_label(group["float32"]),
                    "executor_config_hash": _executor_config_hash(group["float32"]),
                    "circuit_semantics_hash": group["float32"].get(
                        "circuit_semantics_hash"
                    ),
                    "tensor_network_hash": group["float32"].get("tensor_network_hash"),
                    "contraction_plan_hash": group["float32"].get(
                        "contraction_plan_hash"
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
                        "stage": stage,
                        "time_s": number,
                    }
                )
    return result


def _transfer_row(row: Mapping[str, Any]) -> JsonDict:
    transfers = (
        row.get("transfers") if isinstance(row.get("transfers"), Mapping) else {}
    )
    h2d = _float(_first(row, "actual_h2d_bytes", "application_visible_h2d_bytes"))
    d2h = _float(_first(row, "actual_d2h_bytes", "application_visible_d2h_bytes"))
    total = _float(
        _first(row, "actual_transfer_bytes", "application_visible_transfer_bytes")
    )
    if h2d is None:
        h2d = _float(transfers.get("h2d_bytes"))
    if d2h is None:
        d2h = _float(transfers.get("d2h_bytes"))
    if total is None:
        total = _float(transfers.get("transfer_bytes"))
    invariant = (
        h2d is not None
        and d2h is not None
        and total is not None
        and math.isclose(total, h2d + d2h, rel_tol=0, abs_tol=1e-9)
    )
    return {
        "case_id": _case(row),
        "family": _family(row),
        "qubits": _qubits(row),
        "engine": _engine(row),
        "scientific_admitted": _valid(row),
        "h2d_bytes": h2d,
        "d2h_bytes": d2h,
        "transfer_bytes": total,
        "invariant_passed": invariant,
    }
