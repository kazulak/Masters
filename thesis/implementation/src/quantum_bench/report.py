"""Pure reporting for finalized canonical experiment evidence."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
import csv
import os
from pathlib import Path
from statistics import median
import tempfile
from typing import Any

from quantum_bench.evidence import canonical_json, load_artifacts


_COMPONENT_FIELDS = (
    "lowering_s",
    "planning_s",
    "slicing_s",
    "mapping_s",
    "session_open_s",
    "encode_s",
    "preparation_s",
    "h2d_s",
    "kernel_s",
    "host_reduce_s",
    "d2h_s",
    "decode_s",
    "rank_work_s",
    "energy_j",
)
_RESOURCE_FACTS = (
    "requested_dpus",
    "allocated_dpus",
    "active_dpus",
    "rank_count",
    "tasklets_per_dpu",
)
_TERMINAL_AUTHORITY_FIELDS = frozenset(
    {
        "target_observed",
        "physical_target_verified",
        "hardware_kernel_executed",
        "simulator_kernel_executed",
        "cpu_fallback_used",
        "requested_dpu_count",
        "allocated_dpu_count",
        "tasklets_per_dpu",
    }
)
_AGGREGATE_COLUMNS = (
    "case_id",
    "plan_id",
    "route_id",
    "scope_id",
    "problem_id",
    "tensor_network_structure_id",
    "logical_plan_id",
    "physical_plan_id",
    "numeric_policy",
    "sample_count",
    "median_total_wall_s",
    "min_total_wall_s",
    "max_total_wall_s",
    *tuple(f"median_{field}" for field in _COMPONENT_FIELDS),
    "median_h2d_bytes",
    "median_d2h_bytes",
    "median_max_abs_error",
    "median_relative_l2_error",
    *tuple(f"median_{field}" for field in _RESOURCE_FACTS),
)
_SPEEDUP_COLUMNS = (
    "case_id",
    "plan_id",
    "baseline_route_id",
    "candidate_route_id",
    "scope_id",
    "problem_id",
    "tensor_network_structure_id",
    "logical_plan_id",
    "numeric_policy",
    "baseline_median_total_wall_s",
    "candidate_median_total_wall_s",
    "speedup",
)


def verify_artifacts(input_dir: str | os.PathLike[str]) -> dict[str, object]:
    """Return a narrow verification summary for one finalized evidence directory."""

    manifest, samples, sessions = load_artifacts(input_dir)
    return _verification_summary(manifest, samples, sessions)


def report_artifacts(
    input_dir: str | os.PathLike[str], output_dir: str | os.PathLike[str]
) -> dict[str, object]:
    """Write deterministic tables and plots from finalized evidence only."""

    manifest, samples, sessions = load_artifacts(input_dir)
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError("report output directory must be absent or empty")
    output.mkdir(parents=True, exist_ok=True)

    aggregates = _aggregate_measurements(samples, sessions)
    if manifest["status"] != "completed":
        speedups, rejections = [], Counter({"artifact_not_completed": 1})
    elif manifest["source_worktree_dirty"] is True:
        speedups, rejections = [], Counter({"source_worktree_dirty": 1})
    else:
        speedups, rejections = _admit_speedups(aggregates)
    verification = _verification_summary(manifest, samples, sessions)
    simulator_present = any(
        _all_fact_values(aggregate, "target_observed", "sdk_simulator")
        or _any_fact_true(aggregate, "simulator_kernel_executed")
        for aggregate in aggregates
    )
    report = {
        "schema_version": "evidence_report_v1",
        "status": "completed",
        "run_id": manifest["run_id"],
        "experiment_id": manifest["experiment_id"],
        "artifact_status": manifest["status"],
        "verification": verification,
        "aggregate_count": len(aggregates),
        "speedup_count": len(speedups),
        "speedup_rejections": dict(sorted(rejections.items())),
        "failed_count": verification["failed_count"],
        "unsupported_count": verification["unsupported_count"],
        "session_count": len(sessions),
        "simulator_timing": {
            "present": simulator_present,
            "diagnostic_only": simulator_present,
            "prohibited_claims": (
                ["timing", "scaling", "speedup", "energy"]
                if simulator_present
                else []
            ),
        },
        "energy": {
            "measurement_count": sum(
                1
                for aggregate in aggregates
                if aggregate.get("median_energy_j") is not None
            ),
            "energy_efficiency_claim_generated": False,
        },
    }
    _write_json(output / "report.json", report)
    _write_csv(output / "aggregate.csv", _AGGREGATE_COLUMNS, aggregates)
    _write_csv(output / "speedups.csv", _SPEEDUP_COLUMNS, speedups)
    _write_plots(output / "plots", aggregates, speedups)
    return report


def _aggregate_measurements(
    samples: Iterable[Mapping[str, Any]],
    sessions: Iterable[Mapping[str, Any]],
) -> list[dict[str, object]]:
    terminal_facts_by_session = {
        str(session["session_instance_id"]): session["terminal_backend_facts"]
        for session in sessions
        if isinstance(session["terminal_backend_facts"], Mapping)
    }
    grouped: dict[
        tuple[object, ...], list[tuple[Mapping[str, Any], Mapping[str, Any]]]
    ] = {}
    for sample in samples:
        if sample["status"] != "success" or sample["sample_kind"] != "measurement":
            continue
        measurement = sample["measurement"]
        if not isinstance(measurement, Mapping):  # validated by load_artifacts
            raise ValueError("successful measurement sample lacks a measurement")
        identities = sample["identities"]
        if not isinstance(identities, Mapping):  # validated by load_artifacts
            raise ValueError("sample lacks identity mapping")
        key = (
            sample["case_id"],
            sample["plan_id"],
            sample["route_id"],
            measurement["scope_id"],
            identities["problem_id"],
            identities["tensor_network_structure_id"],
            identities["logical_plan_id"],
            identities["physical_plan_id"],
            _numeric_policy(sample),
        )
        grouped.setdefault(key, []).append(
            (sample, _joined_backend_facts(sample, terminal_facts_by_session))
        )

    aggregates = [_make_aggregate(key, rows) for key, rows in grouped.items()]
    return sorted(aggregates, key=_aggregate_sort_key)


def _verification_summary(
    manifest: Mapping[str, Any],
    samples: Iterable[Mapping[str, Any]],
    sessions: Iterable[Mapping[str, Any]],
) -> dict[str, object]:
    sample_rows = tuple(samples)
    session_rows = tuple(sessions)
    statuses = Counter(str(sample["status"]) for sample in sample_rows)
    scopes = sorted(
        {
            str(sample["measurement"]["scope_id"])
            for sample in sample_rows
            if sample["status"] == "success" and sample["measurement"] is not None
        }
    )
    validation_pass_count = sum(
        1
        for sample in sample_rows
        if isinstance(sample["validation"], Mapping)
        and sample["validation"].get("scientific_validation_passed") is True
    )
    return {
        "status": manifest["status"],
        "run_id": manifest["run_id"],
        "experiment_id": manifest["experiment_id"],
        "sample_count": len(sample_rows),
        "session_count": len(session_rows),
        "success_count": statuses["success"],
        "failed_count": statuses["failed"],
        "unsupported_count": statuses["unsupported"],
        "timing_scopes": scopes,
        "validation_pass_count": validation_pass_count,
    }


def _make_aggregate(
    key: tuple[object, ...],
    rows_with_facts: list[tuple[Mapping[str, Any], Mapping[str, Any]]],
) -> dict[str, object]:
    (
        case_id,
        plan_id,
        route_id,
        scope_id,
        problem_id,
        tensor_network_structure_id,
        logical_plan_id,
        physical_plan_id,
        numeric_policy,
    ) = key
    rows = [row for row, _ in rows_with_facts]
    measurements = [row["measurement"] for row in rows]
    if not all(isinstance(measurement, Mapping) for measurement in measurements):
        raise ValueError("aggregate rows must have measurements")
    typed_measurements = [
        measurement for measurement in measurements if isinstance(measurement, Mapping)
    ]
    totals = [float(measurement["total_wall_s"]) for measurement in typed_measurements]
    aggregate: dict[str, object] = {
        "case_id": case_id,
        "plan_id": plan_id,
        "route_id": route_id,
        "scope_id": scope_id,
        "problem_id": problem_id,
        "tensor_network_structure_id": tensor_network_structure_id,
        "logical_plan_id": logical_plan_id,
        "physical_plan_id": physical_plan_id,
        "numeric_policy": numeric_policy,
        "sample_count": len(rows),
        "median_total_wall_s": median(totals),
        "min_total_wall_s": min(totals),
        "max_total_wall_s": max(totals),
        "_samples": rows,
        "_joined_backend_facts": [facts for _, facts in rows_with_facts],
    }
    for field in _COMPONENT_FIELDS:
        values = _non_null_measurements(typed_measurements, field)
        if values:
            aggregate[f"median_{field}"] = median(values)
        else:
            aggregate[f"median_{field}"] = None
    for field in ("h2d_bytes", "d2h_bytes"):
        values = _non_null_measurements(typed_measurements, field)
        aggregate[f"median_{field}"] = median(values) if values else None
    for field in ("max_abs_error", "relative_l2_error"):
        values = _validation_values(rows, field)
        aggregate[f"median_{field}"] = median(values) if values else None
    for field in _RESOURCE_FACTS:
        values = _resource_values(rows_with_facts, field)
        aggregate[f"median_{field}"] = median(values) if values else None
    return aggregate


def _non_null_measurements(
    measurements: Iterable[Mapping[str, Any]], field: str
) -> list[int | float]:
    values: list[int | float] = []
    for measurement in measurements:
        value = measurement[field]
        if value is not None:
            values.append(value)
    return values


def _validation_values(rows: Iterable[Mapping[str, Any]], field: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        validation = row["validation"]
        if isinstance(validation, Mapping) and validation[field] is not None:
            values.append(float(validation[field]))
    return values


def _resource_values(
    rows: Iterable[tuple[Mapping[str, Any], Mapping[str, Any]]], field: str
) -> list[int]:
    values: list[int] = []
    for _, facts in rows:
        value = facts.get(field)
        if value is None:
            allocation = facts.get("allocation")
            if isinstance(allocation, Mapping):
                value = allocation.get(field)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            values.append(value)
    return values


def _joined_backend_facts(
    sample: Mapping[str, Any],
    terminal_facts_by_session: Mapping[str, Mapping[str, Any]],
) -> Mapping[str, Any]:
    """Use terminal facts only to fill facts absent from a linked sample."""

    facts = sample["backend_facts"]
    if not isinstance(facts, Mapping):
        raise ValueError("sample backend_facts must be a mapping")
    joined = dict(facts)
    session_instance_id = sample["session_instance_id"]
    if isinstance(session_instance_id, str):
        terminal = terminal_facts_by_session.get(session_instance_id)
        if terminal is not None:
            conflicts: list[str] = []
            for key, value in terminal.items():
                if (
                    key in _TERMINAL_AUTHORITY_FIELDS
                    and key in joined
                    and joined[key] != value
                ):
                    conflicts.append(key)
                joined.setdefault(key, value)
            if conflicts:
                joined["terminal_fact_conflicts"] = sorted(conflicts)
    return joined


def _numeric_policy(sample: Mapping[str, Any]) -> str | None:
    facts = sample["numeric_facts"]
    if not isinstance(facts, Mapping):
        return None
    value = facts.get("numeric_policy")
    return value if isinstance(value, str) and value else None


def _aggregate_sort_key(aggregate: Mapping[str, object]) -> tuple[str, ...]:
    return tuple(
        "" if aggregate[field] is None else str(aggregate[field])
        for field in (
            "case_id",
            "plan_id",
            "route_id",
            "scope_id",
            "problem_id",
            "tensor_network_structure_id",
            "logical_plan_id",
            "physical_plan_id",
            "numeric_policy",
        )
    )


def _admit_speedups(
    aggregates: Iterable[Mapping[str, object]],
) -> tuple[list[dict[str, object]], Counter[str]]:
    rows = list(aggregates)
    baselines = [
        row for row in rows if _all_fact_values(row, "backend_id", "numpy_cpu_v1")
    ]
    speedups: list[dict[str, object]] = []
    rejections: Counter[str] = Counter()
    for candidate in rows:
        if _all_fact_values(candidate, "backend_id", "numpy_cpu_v1"):
            continue
        if not _is_upmem_candidate(candidate):
            continue
        reason = _speedup_rejection(candidate, baselines)
        if reason is not None:
            rejections[reason] += 1
            continue
        baseline = _matching_baseline(candidate, baselines)
        if baseline is None:
            rejections[_missing_baseline_reason(candidate, baselines)] += 1
            continue
        baseline_time = float(baseline["median_total_wall_s"])
        candidate_time = float(candidate["median_total_wall_s"])
        speedups.append(
            {
                "case_id": candidate["case_id"],
                "plan_id": candidate["plan_id"],
                "baseline_route_id": baseline["route_id"],
                "candidate_route_id": candidate["route_id"],
                "scope_id": candidate["scope_id"],
                "problem_id": candidate["problem_id"],
                "tensor_network_structure_id": candidate["tensor_network_structure_id"],
                "logical_plan_id": candidate["logical_plan_id"],
                "numeric_policy": candidate["numeric_policy"],
                "baseline_median_total_wall_s": baseline_time,
                "candidate_median_total_wall_s": candidate_time,
                "speedup": baseline_time / candidate_time,
            }
        )
    return sorted(
        speedups, key=lambda row: tuple(str(row[key]) for key in _SPEEDUP_COLUMNS)
    ), rejections


def _is_upmem_candidate(aggregate: Mapping[str, object]) -> bool:
    facts = aggregate["_joined_backend_facts"]
    return isinstance(facts, list) and bool(facts) and all(
        isinstance(sample_facts, Mapping)
        and (
            str(sample_facts.get("backend_id") or "").startswith("upmem_")
            or sample_facts.get("execution_class")
            in {"sdk_simulator", "physical_hardware"}
            or sample_facts.get("target_observed")
            in {"sdk_simulator", "physical_hardware"}
        )
        for sample_facts in facts
    )


def _speedup_rejection(
    candidate: Mapping[str, object], baselines: Iterable[Mapping[str, object]]
) -> str | None:
    if _all_fact_values(
        candidate, "target_observed", "sdk_simulator"
    ) or _any_fact_true(candidate, "simulator_kernel_executed"):
        return "simulator_execution"
    if candidate["sample_count"] < 2:
        return "candidate_has_fewer_than_two_measurements"
    if not _all_validation_passed(candidate):
        return "candidate_validation_failed"
    if not _all_full_precision_thresholds_passed(candidate):
        return "full_precision_threshold_not_passed"
    if _any_terminal_fact_conflict(candidate):
        return "terminal_fact_conflict"
    if not _all_samples_linked_to_sessions(candidate):
        return "candidate_session_not_linked"
    if not _all_fact_values(candidate, "target_observed", "physical_hardware"):
        return "candidate_not_physical_hardware"
    if not _all_fact_values(candidate, "physical_target_verified", True):
        return "physical_target_not_verified"
    if not _all_fact_values(candidate, "cpu_fallback_used", False):
        return "cpu_fallback_used"
    if not _all_fact_values(candidate, "simulator_kernel_executed", False):
        return "simulator_kernel_executed"
    if not any(
        _same_speedup_dimensions(candidate, baseline)
        and _baseline_is_eligible(baseline)
        for baseline in baselines
    ):
        return _missing_baseline_reason(candidate, baselines)
    return None


def _missing_baseline_reason(
    candidate: Mapping[str, object], baselines: Iterable[Mapping[str, object]]
) -> str:
    baseline_rows = tuple(baselines)
    if any(_same_speedup_dimensions(candidate, baseline) for baseline in baseline_rows):
        return "baseline_not_admissible"
    dimensions_without_scope = (
        "case_id",
        "plan_id",
        "problem_id",
        "tensor_network_structure_id",
        "logical_plan_id",
        "numeric_policy",
    )
    if any(
        all(candidate[field] == baseline[field] for field in dimensions_without_scope)
        for baseline in baseline_rows
    ):
        return "timing_scope_mismatch"
    return "no_matching_numpy_baseline"


def _matching_baseline(
    candidate: Mapping[str, object], baselines: Iterable[Mapping[str, object]]
) -> Mapping[str, object] | None:
    for baseline in baselines:
        if _same_speedup_dimensions(candidate, baseline) and _baseline_is_eligible(
            baseline
        ):
            return baseline
    return None


def _baseline_is_eligible(aggregate: Mapping[str, object]) -> bool:
    return (
        aggregate["sample_count"] >= 2
        and _all_validation_passed(aggregate)
        and _all_full_precision_thresholds_passed(aggregate)
    )


def _same_speedup_dimensions(
    left: Mapping[str, object], right: Mapping[str, object]
) -> bool:
    return all(
        left[field] == right[field]
        for field in (
            "case_id",
            "plan_id",
            "scope_id",
            "problem_id",
            "tensor_network_structure_id",
            "logical_plan_id",
            "numeric_policy",
        )
    )


def _all_validation_passed(aggregate: Mapping[str, object]) -> bool:
    samples = aggregate["_samples"]
    return (
        isinstance(samples, list)
        and bool(samples)
        and all(
            isinstance(sample["validation"], Mapping)
            and sample["validation"].get("scientific_validation_passed") is True
            for sample in samples
        )
    )


def _all_full_precision_thresholds_passed(
    aggregate: Mapping[str, object],
) -> bool:
    samples = aggregate["_samples"]
    return (
        isinstance(samples, list)
        and bool(samples)
        and all(
            isinstance(sample["validation"], Mapping)
            and sample["validation"].get("full_precision_threshold_applicable") is True
            and sample["validation"].get("full_precision_passed") is True
            for sample in samples
        )
    )


def _all_samples_linked_to_sessions(aggregate: Mapping[str, object]) -> bool:
    samples = aggregate["_samples"]
    return (
        isinstance(samples, list)
        and bool(samples)
        and all(
            isinstance(sample.get("session_instance_id"), str) for sample in samples
        )
    )


def _any_terminal_fact_conflict(aggregate: Mapping[str, object]) -> bool:
    facts = aggregate["_joined_backend_facts"]
    return isinstance(facts, list) and any(
        isinstance(sample_facts, Mapping)
        and bool(sample_facts.get("terminal_fact_conflicts"))
        for sample_facts in facts
    )


def _all_fact_values(
    aggregate: Mapping[str, object], field: str, expected: object
) -> bool:
    facts = aggregate["_joined_backend_facts"]
    return (
        isinstance(facts, list)
        and bool(facts)
        and all(
            isinstance(sample_facts, Mapping) and sample_facts.get(field) == expected
            for sample_facts in facts
        )
    )


def _any_fact_true(aggregate: Mapping[str, object], field: str) -> bool:
    facts = aggregate["_joined_backend_facts"]
    return isinstance(facts, list) and any(
        isinstance(sample_facts, Mapping) and sample_facts.get(field) is True
        for sample_facts in facts
    )


def _write_plots(
    directory: Path,
    aggregates: Iterable[Mapping[str, object]],
    speedups: Iterable[Mapping[str, object]],
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    rows = list(aggregates)
    for scope_id in sorted({str(row["scope_id"]) for row in rows}):
        points = [
            _point(
                figure_id=f"runtime_{scope_id}",
                facet_id=scope_id,
                row=row,
                value=float(row["median_total_wall_s"]),
            )
            for row in rows
            if row["scope_id"] == scope_id
        ]
        _plot_grouped_bars(
            directory / f"runtime_{scope_id}.png",
            points,
            title=f"Median Runtime ({_humanize(scope_id)})",
            ylabel="Median total wall time (s)",
        )

    error_points = [
        _point(
            figure_id="numeric_error_by_case",
            facet_id="all",
            row=row,
            value=float(row["median_max_abs_error"]),
        )
        for row in rows
        if row["median_max_abs_error"] is not None
    ]
    if error_points:
        _plot_grouped_bars(
            directory / "numeric_error_by_case.png",
            error_points,
            title="Median Numeric Error by Case",
            ylabel="Median maximum absolute error",
        )

    transfer_points = []
    for row in rows:
        h2d = row["median_h2d_bytes"]
        d2h = row["median_d2h_bytes"]
        if h2d is None and d2h is None:
            continue
        transfer_points.append(
            _point(
                figure_id="transfer_bytes_by_case",
                facet_id="all",
                row=row,
                value=float(h2d or 0) + float(d2h or 0),
            )
        )
    if transfer_points:
        _plot_grouped_bars(
            directory / "transfer_bytes_by_case.png",
            transfer_points,
            title="Median Host-DPU Transfer by Case",
            ylabel="Median H2D + D2H bytes",
        )

    speedup_rows = list(speedups)
    if speedup_rows:
        points = [
            {
                "figure_id": "physical_speedup_by_case",
                "facet_id": str(row["scope_id"]),
                "series_id": "|".join(
                    str(row[field])
                    for field in (
                        "plan_id",
                        "candidate_route_id",
                        "numeric_policy",
                    )
                ),
                "series_label": _plan_label(row["plan_id"])
                + " | "
                + _humanize(str(row["candidate_route_id"]))
                + " | "
                + _numeric_label(row["numeric_policy"]),
                "x_value": str(row["case_id"]),
                "x_label": _humanize(str(row["case_id"])),
                "value": float(row["speedup"]),
            }
            for row in speedup_rows
        ]
        _plot_grouped_bars(
            directory / "physical_speedup_by_case.png",
            points,
            title="Physical UPMEM Speedup by Case",
            ylabel="NumPy median time / UPMEM median time",
            reference_line=1.0,
        )


def _point(
    *, figure_id: str, facet_id: str, row: Mapping[str, object], value: float
) -> dict[str, object]:
    # A series identifies the route intervention. Plan hashes vary by case and
    # must not manufacture a separate legend entry for every x-axis value.
    series_id = "|".join(
        "" if row[field] is None else str(row[field])
        for field in ("plan_id", "route_id", "numeric_policy")
    )
    return {
        "figure_id": figure_id,
        "facet_id": facet_id,
        "series_id": series_id,
        "series_label": _series_label(row),
        "x_value": str(row["case_id"]),
        "x_label": _humanize(str(row["case_id"])),
        "value": value,
    }


def _plot_grouped_bars(
    path: Path,
    points: list[Mapping[str, object]],
    *,
    title: str,
    ylabel: str,
    reference_line: float | None = None,
) -> None:
    _assert_unique_plot_points(points)
    import matplotlib

    matplotlib.use("Agg", force=True)
    from matplotlib import pyplot as plt

    x_values = sorted({str(point["x_value"]) for point in points})
    series_ids = sorted({str(point["series_id"]) for point in points})
    x_labels = {str(point["x_value"]): str(point["x_label"]) for point in points}
    series_labels = {
        str(point["series_id"]): str(point["series_label"]) for point in points
    }
    values = {
        (str(point["x_value"]), str(point["series_id"])): float(point["value"])
        for point in points
    }
    figure, axis = plt.subplots(figsize=(max(6.0, 1.4 * len(x_values)), 4.8))
    width = 0.8 / max(1, len(series_ids))
    centers = list(range(len(x_values)))
    for offset, series_id in enumerate(series_ids):
        positions = [center - 0.4 + width / 2 + offset * width for center in centers]
        heights = [
            values.get((x_value, series_id), float("nan")) for x_value in x_values
        ]
        axis.bar(positions, heights, width=width, label=series_labels[series_id])
    axis.set_xticks(centers, [x_labels[value] for value in x_values])
    axis.set_title(title)
    axis.set_ylabel(ylabel)
    if reference_line is not None:
        axis.axhline(reference_line, color="black", linewidth=1.0)
    if len(series_ids) > 1:
        longest_label = max(len(series_labels[series_id]) for series_id in series_ids)
        columns = 1 if longest_label > 45 else min(2, len(series_ids))
        rows = (len(series_ids) + columns - 1) // columns
        handles, labels = axis.get_legend_handles_labels()
        figure.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.01),
            ncol=columns,
            fontsize=8,
        )
        figure.tight_layout(rect=(0.0, min(0.22, 0.08 + 0.04 * rows), 1.0, 1.0))
    else:
        figure.tight_layout()
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def _assert_unique_plot_points(points: Iterable[Mapping[str, object]]) -> None:
    seen: set[tuple[str, str, str, str]] = set()
    for point in points:
        key = tuple(
            str(point[field])
            for field in ("figure_id", "facet_id", "series_id", "x_value")
        )
        if key in seen:
            raise ValueError(f"duplicate plot point: {key}")
        seen.add(key)


def _humanize(value: str) -> str:
    known = {
        "numpy_cpu_v1": "NumPy CPU",
        "split_complex_float32_v1": "Complex float32",
        "split_complex_int8_shared_scale_v1": "Complex int8 shared-scale",
        "simulation_end_to_end_v1": "Simulation end-to-end",
        "steady_execution_v1": "Steady execution",
    }
    return known.get(value, value.replace("_", " ").replace("-", " ").title())


def _humanize_nullable(value: object) -> str:
    return _humanize(value) if isinstance(value, str) else "Unspecified numeric policy"


def _series_label(row: Mapping[str, object]) -> str:
    parts = [
        _plan_label(row["plan_id"]),
        _humanize(str(row["route_id"])),
        _numeric_label(row["numeric_policy"]),
    ]
    active_dpus = row.get("median_active_dpus")
    tasklets = row.get("median_tasklets_per_dpu")
    if isinstance(active_dpus, (int, float)):
        count = int(active_dpus)
        parts.append(f"{count} active DPU" + ("s" if count != 1 else ""))
    if isinstance(tasklets, (int, float)):
        count = int(tasklets)
        parts.append(f"{count} tasklet" + ("s" if count != 1 else "") + "/DPU")
    return " | ".join(parts)


def _plan_label(value: object) -> str:
    return _humanize(value) if isinstance(value, str) else "Route-owned plan"


def _numeric_label(value: object) -> str:
    known = {
        "split_complex_float32_v1": "float32",
        "split_complex_int8_shared_scale_v1": "int8 shared-scale",
    }
    return known.get(value, _humanize_nullable(value))


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    _atomic_write(path, (canonical_json(value) + "\n").encode("utf-8"))


def _write_csv(
    path: Path, fields: tuple[str, ...], rows: Iterable[Mapping[str, object]]
) -> None:
    lines: list[str] = []
    import io

    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row.get(field) for field in fields})
    lines.append(stream.getvalue())
    _atomic_write(path, "".join(lines).encode("utf-8"))


def _atomic_write(path: Path, payload: bytes) -> None:
    file_descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    try:
        with os.fdopen(file_descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


__all__ = ["report_artifacts", "verify_artifacts"]
