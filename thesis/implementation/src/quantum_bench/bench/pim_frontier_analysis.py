from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any

import yaml

from quantum_bench.bench.config import DEFAULTS, load_suite
from quantum_bench.bench.run_dirs import create_run_dir, sanitize
from quantum_bench.circuits import builtin_circuit, load_circuit, manifest
from quantum_bench.core.jsonio import write_json, write_jsonl
from quantum_bench.core.records import JsonDict, to_jsonable
from quantum_bench.environment import capture_environment
from quantum_bench.targets.upmem import (
    PIM_FRONTIER_ANALYSIS_SCHEMA_VERSION,
    UpmemResourceModel,
    analyze_task_graph,
    build_synthetic_pressure_task_graph,
    is_synthetic_pressure_case,
    synthetic_pressure_manifest,
)
from quantum_bench.tn import build_tensor_network, plan_task_graph_with_config, with_path_cost_summary
from quantum_bench.targets.upmem import annotate_task_graph_with_upmem_estimates


_PARAMETERIZED_CASES = {
    "ghz_chain",
    "qrng",
    "bv",
    "bernstein_vazirani",
    "xor",
    "parity",
    "bb84",
    "bb_n",
    "edc",
    "dense_coding",
    "hs",
    "hidden_shift",
}

TASK_FIELDS = [
    "case_id",
    "workload_id",
    "circuit_family",
    "n_qubits",
    "planner_engine",
    "planner_id",
    "optimize_mode",
    "task_index",
    "task_id",
    "input_tensor_ids",
    "output_tensor_id",
    "dependencies",
    "frontier_wave_index",
    "gemm_m",
    "gemm_k",
    "gemm_n",
    "structure",
    "dense_lowerable",
    "memory_level",
    "memory_reason",
    "backend_supported",
    "current_backend_executable",
    "current_backend_reason",
    "estimate_reject_reason",
    "requires_tiling",
    "requires_host_aggregation",
    "tiling_implemented",
    "a_bytes",
    "b_bytes",
    "c_output_bytes",
    "c_accumulator_bytes",
    "full_task_bytes",
    "working_set_bytes",
    "estimated_flops",
    "estimated_bytes",
    "estimated_host_to_dpu_bytes",
    "estimated_dpu_to_host_bytes",
    "estimated_mram_to_wram_bytes",
    "estimated_output_tiles",
    "estimated_k_tiles",
    "estimated_total_tiles",
    "optional_parallel_tile_count",
    "memory_capacity_min_dpus",
    "estimated_dpus_required",
    "estimated_parallel_dpus",
    "blocker_reason",
    "dominant_source",
]

CASE_FIELDS = [
    "case_id",
    "workload_id",
    "circuit_family",
    "n_qubits",
    "task_count",
    "wave_count",
    "critical_path_length_tasks",
    "max_frontier_width",
    "mean_frontier_width",
    "mean_estimated_dpu_occupancy",
    "memory_level_counts",
    "dominant_source_counts",
    "potential_parallelism_source",
    "total_estimated_host_to_dpu_bytes",
    "total_estimated_dpu_to_host_bytes",
    "total_estimated_mram_to_wram_bytes",
    "total_estimated_transfer_bytes",
    "total_estimated_flops",
    "max_full_task_bytes",
    "max_working_set_bytes",
    "max_modeled_dpu_group_size",
    "l1_task_count",
    "l2_task_count",
    "l3_task_count",
    "l4_task_count",
    "unclassified_dense_task_count",
    "unresolved_dependency_count",
]

WAVE_FIELDS = [
    "case_id",
    "workload_id",
    "circuit_family",
    "n_qubits",
    "frontier_wave_index",
    "task_indices",
    "task_ids",
    "ready_task_count",
    "memory_level_counts",
    "dominant_source",
    "inter_task_parallelism_potential",
    "intra_task_parallelism_potential",
    "hybrid_parallelism_potential",
    "schedulable_task_count",
    "scheduling_rounds",
    "assigned_dpu_slots",
    "max_group_dpus",
    "estimated_dpu_occupancy",
    "idle_dpu_fraction",
]


def run_pim_frontier_analysis(
    root_dir: Path,
    *,
    suite_path: Path | None = None,
    case: str | None = None,
    n_qubits: int | None = None,
    resource_model: UpmemResourceModel | None = None,
    output_plots: bool = True,
) -> Path:
    model = resource_model or UpmemResourceModel()
    validate_cli_options(suite_path=suite_path, case=case, n_qubits=n_qubits, resource_model=model)

    suite: dict[str, Any] | None
    cases: list[dict[str, Any]]
    suite_id: str
    planner_config: dict[str, Any]
    source: str
    if suite_path is not None:
        suite = load_suite(suite_path)
        cases = [dict(item) for item in suite["cases"]]
        suite_id = str(suite["suite_id"])
        planner_config = dict(suite["planner"])
        source = "suite"
    else:
        suite = None
        circuit = _single_builtin_circuit(str(case), n_qubits)
        suite_id = circuit.name
        planner_config = dict(DEFAULTS["planner"])
        circuit_payload: JsonDict = {"kind": "builtin", "name": str(case)}
        if n_qubits is not None:
            circuit_payload["n_qubits"] = int(n_qubits)
        cases = [
            {
                "case_id": circuit.name,
                "workload_id": circuit.name,
                "circuit": circuit_payload,
                "_preloaded_circuit": circuit,
            }
        ]
        source = "case"

    run_dir = create_run_dir(root_dir, _run_id(suite_id))
    write_json(run_dir / "environment.json", capture_environment(root_dir))
    if suite is not None:
        (run_dir / "config" / "resolved_suite.yml").write_text(yaml.safe_dump(suite, sort_keys=True), encoding="utf-8")
    write_json(
        run_dir / "config" / "pim_frontier_analysis_input.json",
        {
            "schema_version": PIM_FRONTIER_ANALYSIS_SCHEMA_VERSION,
            "source": source,
            "suite_path": _display_path(suite_path, root_dir) if suite_path is not None else None,
            "case": case,
            "n_qubits": n_qubits,
            "planner": planner_config,
            "resource_model": model.as_dict(),
            "output_plots": output_plots,
        },
    )

    task_rows: list[JsonDict] = []
    wave_rows: list[JsonDict] = []
    case_summaries: list[JsonDict] = []
    for case_payload in cases:
        case_task_rows, case_wave_rows, case_summary = _analyze_case(
            root_dir=root_dir,
            run_dir=run_dir,
            case_payload=dict(case_payload),
            planner_config=planner_config,
            resource_model=model,
        )
        task_rows.extend(case_task_rows)
        wave_rows.extend(case_wave_rows)
        case_summaries.append(case_summary)

    summary = _run_summary(case_summaries)
    payload = {
        "schema_version": PIM_FRONTIER_ANALYSIS_SCHEMA_VERSION,
        "run_id": run_dir.name,
        "suite_id": suite_id,
        "source": source,
        "suite_path": _display_path(suite_path, root_dir) if suite_path is not None else None,
        "resource_model": model.as_dict(),
        "summary": summary,
        "case_summaries": case_summaries,
        "task_rows": task_rows,
        "wave_rows": wave_rows,
        "metadata": {
            "developer_only": True,
            "analysis_only": True,
            "upmem_kernels_executed": False,
            "simplepim_executed": False,
            "normal_suite_routes_executed": False,
            "suite_routes_ignored": True,
            "parallelism_mode": "modeled_only",
            "parallelism_evidence_type": "modeled",
            "execution_plan_kind": "frontier_wave_analysis",
            "execution_plan_executed": False,
            "slicing_enabled": False,
            "frontier_scheduler_enabled": False,
            "intra_contraction_parallelism_source": "modeled_tile_estimate",
            "modeled_parallelism_available": bool(summary.get("max_frontier_width", 0) or summary.get("max_modeled_dpu_group_size", 0)),
            "frontier_width_one_is_expected_for_serialized_paths": True,
        },
    }
    write_json(run_dir / "pim_frontier_analysis.json", payload)
    _write_csv(run_dir / "pim_frontier_analysis_tasks.csv", task_rows, TASK_FIELDS)
    _write_csv(run_dir / "pim_frontier_analysis_cases.csv", case_summaries, CASE_FIELDS)
    _write_csv(run_dir / "pim_frontier_analysis_waves.csv", wave_rows, WAVE_FIELDS)
    (run_dir / "pim_frontier_analysis_summary.md").write_text(_summary_markdown(summary, case_summaries), encoding="utf-8")
    if output_plots:
        _write_plots(run_dir, case_summaries)
    return run_dir


def validate_cli_options(
    *,
    suite_path: Path | None,
    case: str | None,
    n_qubits: int | None,
    resource_model: UpmemResourceModel,
) -> None:
    if (suite_path is None) == (case is None):
        raise ValueError("pim-frontier-analysis requires exactly one of --suite or --case")
    _ = resource_model.as_dict()
    if case is not None:
        _validate_case_n_qubits(str(case), n_qubits)


def _analyze_case(
    *,
    root_dir: Path,
    run_dir: Path,
    case_payload: dict[str, Any],
    planner_config: dict[str, Any],
    resource_model: UpmemResourceModel,
) -> tuple[list[JsonDict], list[JsonDict], JsonDict]:
    is_synthetic = is_synthetic_pressure_case(case_payload)
    circuit = case_payload.pop("_preloaded_circuit", None)
    if is_synthetic:
        graph = build_synthetic_pressure_task_graph(case_payload)
        circuit = graph.network.circuit
    else:
        if circuit is None:
            circuit = load_circuit(case_payload, root_dir)
        network = build_tensor_network(circuit)
        graph = plan_task_graph_with_config(network, planner_config)
        graph, _ = annotate_task_graph_with_upmem_estimates(graph)
        graph = with_path_cost_summary(graph)
    analysis = analyze_task_graph(graph, resource_model)
    case_id = str(case_payload["case_id"])
    workload_id = str(case_payload.get("workload_id", case_id))
    circuit_family = str(case_payload.get("circuit", {}).get("name", circuit.name))
    case_prefix = {
        "case_id": case_id,
        "workload_id": workload_id,
        "circuit_family": circuit_family,
        "n_qubits": int(circuit.n_qubits),
        "planner_engine": graph.path_summary.planner_engine,
        "planner_id": graph.path_summary.planner_id,
        "optimize_mode": graph.path_summary.optimize_mode,
    }
    task_rows = [
        {
            **case_prefix,
            **task.as_row(),
        }
        for task in analysis.tasks
    ]
    wave_rows = [
        {
            "case_id": case_id,
            "workload_id": workload_id,
            "circuit_family": circuit_family,
            "n_qubits": int(circuit.n_qubits),
            **wave.as_row(),
        }
        for wave in analysis.waves
    ]
    summary = {
        "case_id": case_id,
        "workload_id": workload_id,
        "circuit_family": circuit_family,
        "n_qubits": int(circuit.n_qubits),
        "circuit": synthetic_pressure_manifest(graph) if is_synthetic else manifest(circuit),
        "planner_engine": graph.path_summary.planner_engine,
        "planner_id": graph.path_summary.planner_id,
        "optimize_mode": graph.path_summary.optimize_mode,
        **analysis.summary(),
    }
    write_jsonl(run_dir / "cases" / sanitize(case_id) / "pim_frontier_analysis_tasks.jsonl", task_rows)
    return task_rows, wave_rows, summary


def _run_summary(case_summaries: list[JsonDict]) -> JsonDict:
    memory_level_counts: dict[str, int] = {}
    dominant_source_counts: dict[str, int] = {}
    for case in case_summaries:
        for level, count in dict(case.get("memory_level_counts") or {}).items():
            memory_level_counts[str(level)] = memory_level_counts.get(str(level), 0) + int(count)
        for source, count in dict(case.get("dominant_source_counts") or {}).items():
            dominant_source_counts[str(source)] = dominant_source_counts.get(str(source), 0) + int(count)
    return {
        "case_count": len(case_summaries),
        "task_count": sum(int(case.get("task_count", 0) or 0) for case in case_summaries),
        "wave_count": sum(int(case.get("wave_count", 0) or 0) for case in case_summaries),
        "memory_level_counts": dict(sorted(memory_level_counts.items())),
        "dominant_source_counts": dict(sorted(dominant_source_counts.items())),
        "max_frontier_width": max((int(case.get("max_frontier_width", 0) or 0) for case in case_summaries), default=0),
        "max_modeled_dpu_group_size": max((int(case.get("max_modeled_dpu_group_size", 0) or 0) for case in case_summaries), default=0),
        "total_estimated_host_to_dpu_bytes": sum(int(case.get("total_estimated_host_to_dpu_bytes", 0) or 0) for case in case_summaries),
        "total_estimated_dpu_to_host_bytes": sum(int(case.get("total_estimated_dpu_to_host_bytes", 0) or 0) for case in case_summaries),
        "total_estimated_mram_to_wram_bytes": sum(int(case.get("total_estimated_mram_to_wram_bytes", 0) or 0) for case in case_summaries),
        "total_estimated_transfer_bytes": sum(int(case.get("total_estimated_transfer_bytes", 0) or 0) for case in case_summaries),
        "serialized_case_count": sum(
            1
            for case in case_summaries
            if case.get("potential_parallelism_source") == "task_graph_serialized_by_planner"
        ),
    }


def _summary_markdown(summary: JsonDict, case_summaries: list[JsonDict]) -> str:
    lines = [
        "# PIM Frontier Analysis Summary",
        "",
        "This is a modeled memory-level and parallelism-frontier analysis. It does not execute UPMEM, SimplePIM, or normal provider routes.",
        "",
        "## Run Summary",
        "",
        f"- Cases: {summary['case_count']}",
        f"- Tasks: {summary['task_count']}",
        f"- Waves: {summary['wave_count']}",
        f"- Max frontier width: {summary['max_frontier_width']}",
        f"- Serialized-by-planner cases: {summary['serialized_case_count']}",
        f"- Max modeled DPU group size: {summary['max_modeled_dpu_group_size']}",
        "",
        "## Memory Levels",
        "",
    ]
    for level, count in dict(summary["memory_level_counts"]).items():
        lines.append(f"- {level}: {count}")
    if not summary["memory_level_counts"]:
        lines.append("- none: 0")
    lines.extend(["", "## Dominant Parallelism Sources", ""])
    for source, count in dict(summary["dominant_source_counts"]).items():
        lines.append(f"- {source}: {count}")
    if not summary["dominant_source_counts"]:
        lines.append("- none: 0")
    lines.extend(
        [
            "",
            "## Cases",
            "",
            "| Case | Family | Qubits | Tasks | Waves | Max Frontier | Memory Levels | Dominant Sources | Parallelism Source |",
            "|---|---|---:|---:|---:|---:|---|---|---|",
        ]
    )
    for case in case_summaries:
        levels = _counts_text(case.get("memory_level_counts"))
        sources = _counts_text(case.get("dominant_source_counts"))
        lines.append(
            "| {case_id} | {family} | {n_qubits} | {tasks} | {waves} | {frontier} | {levels} | {sources} | {source} |".format(
                case_id=case["case_id"],
                family=case["circuit_family"],
                n_qubits=case["n_qubits"],
                tasks=case["task_count"],
                waves=case["wave_count"],
                frontier=case["max_frontier_width"],
                levels=levels,
                sources=sources,
                source=case["potential_parallelism_source"],
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Frontier width 1 is expected when the current pairwise planner serializes the contraction path; it is useful evidence for future path-frontier optimization rather than a command failure.",
            "L1/L2/L3/L4 are modeled memory-capacity levels. They are separate from whether the current simulator backend can execute a task.",
            "Next implementation priorities should be chosen from the memory-level counts and dominant-source counts: L2 tiling, L3 distributed GEMM, hardware execution, SimplePIM GEMM, SparseP, or route-aware path selection.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_csv(path: Path, rows: list[JsonDict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in fields})


def _write_plots(run_dir: Path, case_summaries: list[JsonDict]) -> None:
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(plots_dir / ".matplotlib"))
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    labels = [str(case["case_id"]) for case in case_summaries]
    if not labels:
        return
    _stacked_count_plot(
        plt,
        plots_dir / "memory_levels_by_case.png",
        case_summaries,
        "memory_level_counts",
        "Memory Levels By Case",
    )
    _bar_plot(
        plt,
        plots_dir / "frontier_width_by_case.png",
        labels,
        [int(case["max_frontier_width"]) for case in case_summaries],
        "Max Frontier Width By Case",
        "Tasks",
    )
    _bar_plot(
        plt,
        plots_dir / "estimated_dpu_occupancy_by_case.png",
        labels,
        [_case_average_occupancy(case) for case in case_summaries],
        "Estimated DPU Occupancy By Case",
        "Modeled Occupancy",
    )
    _stacked_count_plot(
        plt,
        plots_dir / "inter_vs_intra_parallelism_by_case.png",
        case_summaries,
        "dominant_source_counts",
        "Dominant Parallelism Source By Case",
    )
    _l3_line_plot(plt, plots_dir / "l3_tasks_by_qubits.png", case_summaries)


def _bar_plot(plt: Any, path: Path, labels: list[str], values: list[float], title: str, ylabel: str) -> None:
    figure, axis = plt.subplots(figsize=(max(6.0, len(labels) * 0.55), 4.0))
    axis.bar(labels, values, color="#3b82f6")
    axis.set_title(title)
    axis.set_ylabel(ylabel)
    axis.tick_params(axis="x", rotation=45)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _stacked_count_plot(plt: Any, path: Path, case_summaries: list[JsonDict], key: str, title: str) -> None:
    labels = [str(case["case_id"]) for case in case_summaries]
    categories = sorted({category for case in case_summaries for category in dict(case.get(key) or {}).keys()})
    figure, axis = plt.subplots(figsize=(max(6.0, len(labels) * 0.65), 4.5))
    bottoms = [0] * len(labels)
    for category in categories:
        values = [int(dict(case.get(key) or {}).get(category, 0)) for case in case_summaries]
        axis.bar(labels, values, bottom=bottoms, label=category)
        bottoms = [bottom + value for bottom, value in zip(bottoms, values)]
    axis.set_title(title)
    axis.set_ylabel("Count")
    axis.tick_params(axis="x", rotation=45)
    if categories:
        axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _l3_line_plot(plt: Any, path: Path, case_summaries: list[JsonDict]) -> None:
    by_family: dict[str, list[JsonDict]] = {}
    for case in case_summaries:
        by_family.setdefault(str(case["circuit_family"]), []).append(case)
    figure, axis = plt.subplots(figsize=(6.5, 4.0))
    for family, cases in sorted(by_family.items()):
        ordered = sorted(cases, key=lambda item: (int(item["n_qubits"]), str(item["case_id"])))
        axis.plot(
            [int(case["n_qubits"]) for case in ordered],
            [int(case.get("l3_task_count", 0) or 0) for case in ordered],
            marker="o",
            label=family,
        )
    axis.set_title("L3 Tasks By Qubits")
    axis.set_xlabel("Qubits")
    axis.set_ylabel("L3 Modeled Tasks")
    if by_family:
        axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _case_average_occupancy(case: JsonDict) -> float:
    try:
        return float(case.get("mean_estimated_dpu_occupancy", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _counts_text(value: Any) -> str:
    counts = dict(value or {})
    if not counts:
        return "none"
    return ", ".join(f"{key}: {counts[key]}" for key in sorted(counts))


def _csv_value(value: Any) -> Any:
    value = to_jsonable(value)
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    if value is None:
        return ""
    return value


def _single_builtin_circuit(case: str, n_qubits: int | None):
    _validate_case_n_qubits(case, n_qubits)
    params: JsonDict = {"name": case}
    if n_qubits is not None:
        params["n_qubits"] = int(n_qubits)
    return builtin_circuit(case, params)


def _validate_case_n_qubits(case: str, n_qubits: int | None) -> None:
    lowered = case.lower()
    if lowered == "bell_2q":
        if n_qubits not in {None, 2}:
            raise ValueError("bell_2q only supports --n-qubits 2")
        return
    if lowered in _PARAMETERIZED_CASES and n_qubits is None:
        raise ValueError(f"{case} requires --n-qubits")


def _run_id(suite_id: str) -> str:
    sanitized = sanitize(suite_id)
    if sanitized == "pim_frontier_analysis" or sanitized.endswith("_pim_frontier_analysis"):
        return sanitized
    return f"{sanitized}_pim_frontier_analysis"


def _display_path(path: Path | None, root_dir: Path) -> str | None:
    if path is None:
        return None
    try:
        return path.resolve().relative_to(root_dir.resolve()).as_posix()
    except ValueError:
        return path.as_posix()
