from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_BASELINE_ROUTE = "quest_cpu_full_state_benchmark"

ROUTE_LABELS = {
    "cpu_tn_einsum_exact": "CPU exact TN",
    "quest_cpu_full_state_benchmark": "QuEST CPU full-state",
    "upmem_dense_int8_placeholder": "UPMEM dense placeholder",
}


def plot_run(run_dir: Path, baseline_route: str | None = None) -> list[Path]:
    mpl_config = run_dir / "plots" / ".matplotlib"
    mpl_config.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config))
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for plotting; run with thesis/.venv Python") from exc

    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    rows = [row for row in summary["rows"] if row["status"] == "passed"]
    plot_dir = run_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []
    notes: list[str] = []

    if rows:
        created.append(_plot_case_bars(plt, rows, plot_dir / "time_by_case.png", "time_median_s", "Median execution time (s)", log_y=True))
        created.extend(_plot_scaling_by_family(plt, rows, plot_dir, "time_median_s", "Median execution time (s)", "time", log_y=True))

    energy_rows = [row for row in rows if _positive(row.get("energy_median_j"))]
    if energy_rows:
        created.append(_plot_case_bars(plt, energy_rows, plot_dir / "energy_by_case.png", "energy_median_j", "Median measured energy (J)", log_y=False))
        created.extend(_plot_scaling_by_family(plt, energy_rows, plot_dir, "energy_median_j", "Median measured energy (J)", "energy", log_y=False))
    else:
        notes.append("No positive measured energy values; energy plots were skipped.")

    selected_baseline = baseline_route or _default_baseline(rows)
    if selected_baseline:
        speedup_rows = _speedup_rows(rows, selected_baseline)
        if speedup_rows:
            created.append(_plot_case_bars(plt, speedup_rows, plot_dir / "speedup_by_case.png", "speedup", f"Speedup vs {_route_label(selected_baseline)}", log_y=False))
        else:
            notes.append(f"No matching passed rows for speedup baseline route: {selected_baseline}")
    else:
        notes.append(f"Speedup plot skipped because {DEFAULT_BASELINE_ROUTE} is not present in this run.")

    if notes:
        (plot_dir / "plot_notes.json").write_text(json.dumps({"notes": notes}, indent=2) + "\n", encoding="utf-8")
    return created


def _plot_case_bars(plt: Any, rows: list[dict[str, Any]], path: Path, metric: str, ylabel: str, log_y: bool) -> Path:
    cases = sorted({row["case_id"] for row in rows})
    routes = sorted({row["route"] for row in rows})
    values = {(row["case_id"], row["route"]): row.get(metric) for row in rows}
    width = 0.8 / max(len(routes), 1)
    x_positions = list(range(len(cases)))
    fig_width = max(9, 1.5 * len(cases) + 1.2 * len(routes))
    fig, ax = plt.subplots(figsize=(fig_width, 6), constrained_layout=True)

    for route_index, route in enumerate(routes):
        offset = (route_index - (len(routes) - 1) / 2.0) * width
        xs = [x + offset for x in x_positions]
        ys = [values.get((case, route)) for case in cases]
        ax.bar(xs, [float(value) if value is not None else float("nan") for value in ys], width=width, label=_route_label(route))

    ax.set_xticks(x_positions)
    ax.set_xticklabels(cases, rotation=25, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(path.stem.replace("_", " ").title())
    ax.grid(True, axis="y", alpha=0.3)
    if log_y:
        ax.set_yscale("log")
    ax.legend()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _plot_scaling_by_family(plt: Any, rows: list[dict[str, Any]], plot_dir: Path, metric: str, ylabel: str, prefix: str, log_y: bool) -> list[Path]:
    rows_by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get(metric) is not None:
            rows_by_family[str(row["circuit_family"])].append(row)

    created: list[Path] = []
    for family, family_rows in sorted(rows_by_family.items()):
        groups = {
            route: sorted([row for row in family_rows if row["route"] == route], key=lambda item: int(item["n_qubits"]))
            for route in sorted({row["route"] for row in family_rows})
        }
        if not any(len(route_rows) >= 2 for route_rows in groups.values()):
            continue
        path = plot_dir / f"{prefix}_scaling_{_slug(family)}.png"
        fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
        for route, route_rows in groups.items():
            if len(route_rows) < 2:
                continue
            ax.plot(
                [int(row["n_qubits"]) for row in route_rows],
                [float(row[metric]) for row in route_rows],
                marker="o",
                linewidth=2,
                label=_route_label(route),
            )
        ax.set_xlabel("Allocated qubits")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{family} {prefix} scaling")
        ax.grid(True, which="both", alpha=0.3)
        if log_y:
            ax.set_yscale("log")
        ax.legend()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        created.append(path)
    return created


def _speedup_rows(rows: list[dict[str, Any]], baseline_route: str) -> list[dict[str, Any]]:
    baseline_by_case = {
        row["case_id"]: row
        for row in rows
        if row["route"] == baseline_route and row.get("time_median_s") not in {None, 0}
    }
    speedups = []
    for row in rows:
        if row["route"] == baseline_route or row.get("time_median_s") in {None, 0}:
            continue
        base = baseline_by_case.get(row["case_id"])
        if not base:
            continue
        speedups.append({**row, "speedup": float(base["time_median_s"]) / float(row["time_median_s"])})
    return speedups


def _default_baseline(rows: list[dict[str, Any]]) -> str | None:
    return DEFAULT_BASELINE_ROUTE if any(row["route"] == DEFAULT_BASELINE_ROUTE for row in rows) else None


def _route_label(route: str) -> str:
    return ROUTE_LABELS.get(route, route)


def _positive(value: object) -> bool:
    try:
        return value is not None and float(value) > 0
    except (TypeError, ValueError):
        return False


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_") or "unknown"
