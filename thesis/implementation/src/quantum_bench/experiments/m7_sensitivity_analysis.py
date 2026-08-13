"""Data science experiment harness for Milestone M7: PIM-aware path optimizer sensitivity analysis.

This script executes parameter sweeps across quantum circuit benchmark topologies,
analyzing the trade-offs between FLOPs, transfer payload bytes, peak intermediate memory,
and WRAM pressure, and generates publication-grade plots.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np

from quantum_bench.circuits import load_circuit
from quantum_bench.tn.network import build_tensor_network
from quantum_bench.tn.task_graph import plan_task_graph_with_config
from quantum_bench.tn.upmem_path_optimizer import (
    PathSearchState,
    PIMCostParameters,
    PIMPathCostOptimizer,
    eval_pair_step,
    make_sim_contraction_task,
)

# Styling configuration for publication-grade matplotlib figures
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "axes.edgecolor": "#333333",
        "axes.linewidth": 1.2,
        "grid.color": "#e0e0e0",
        "grid.linestyle": "--",
        "grid.alpha": 0.7,
    }
)


def run_single_plan(
    circuit_path: str,
    root_dir: Path,
    config_override: Mapping[str, Any],
) -> dict[str, Any]:
    """Execute task graph planning for a given circuit and planner config."""
    circuit_spec = {"circuit": {"kind": "qasm_file", "path": circuit_path}}
    circuit = load_circuit(circuit_spec, root_dir)
    network = build_tensor_network(circuit)

    base_config = {"engine": "custom_upmem", "optimize": "upmem_pim_cost_greedy"}
    full_config = {**base_config, **config_override}

    graph = plan_task_graph_with_config(network, full_config)

    total_flops = sum(task.estimated_flops for task in graph.tasks)
    total_bytes = sum(task.estimated_bytes for task in graph.tasks)
    max_intermediate_bytes = max((task.estimated_bytes for task in graph.tasks), default=0)
    task_count = len(graph.tasks)

    return {
        "task_count": task_count,
        "total_flops": total_flops,
        "total_bytes": total_bytes,
        "max_intermediate_bytes": max_intermediate_bytes,
        "planning_time_s": graph.planning_time_s,
    }


def experiment_1_flop_vs_transfer_tradeoff(
    circuit_path: str,
    root_dir: Path,
    output_dir: Path,
    artifact_dir: Path,
) -> None:
    """Sweep weight ratio alpha = w_h2d / (w_flops + w_h2d) and plot FLOPs vs Payload Bytes."""
    alphas = np.linspace(0.01, 0.99, 25)
    flops_list = []
    bytes_list = []
    max_mem_list = []

    for alpha in alphas:
        w_h2d = float(alpha) * 10.0
        w_flops = float(1.0 - alpha) * 10.0
        res = run_single_plan(
            circuit_path,
            root_dir,
            {"w_h2d": w_h2d, "w_d2h": w_h2d, "w_flops": w_flops},
        )
        flops_list.append(res["total_flops"])
        bytes_list.append(res["total_bytes"])
        max_mem_list.append(res["max_intermediate_bytes"])

    fig, ax1 = plt.subplots(figsize=(8, 5), dpi=300)

    color1 = "#2b5c8f"
    ax1.set_xlabel("PIM Transfer Weight Ratio $\\alpha = w_{\\text{h2d}} / (w_{\\text{flops}} + w_{\\text{h2d}})$", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Total Estimated FLOPs", color=color1, fontsize=11, fontweight="bold")
    line1 = ax1.plot(alphas, flops_list, color=color1, linewidth=2.5, marker="o", label="FLOP Count")
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.grid(True)

    ax2 = ax1.twinx()
    color2 = "#d95f02"
    ax2.set_ylabel("Total Transfer Payload Bytes (H2D + D2H)", color=color2, fontsize=11, fontweight="bold")
    line2 = ax2.plot(alphas, bytes_list, color=color2, linewidth=2.5, linestyle="--", marker="s", label="Payload Bytes")
    ax2.tick_params(axis="y", labelcolor=color2)

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper center", frameon=True, facecolor="#ffffff", edgecolor="#cccccc")

    plt.title("M7 PIM-Aware Optimizer: FLOPs vs Memory Movement Trade-off", fontsize=13, fontweight="bold", pad=15)
    fig.tight_layout()

    out_file = output_dir / "m7_flop_vs_transfer_tradeoff.png"
    art_file = artifact_dir / "m7_flop_vs_transfer_tradeoff.png"
    plt.savefig(out_file, bbox_inches="tight")
    plt.savefig(art_file, bbox_inches="tight")
    plt.close()
    print(f"Saved Tradeoff plot to {out_file} and {art_file}")


def experiment_2_wram_pressure_heatmap(
    circuit_path: str,
    root_dir: Path,
    output_dir: Path,
    artifact_dir: Path,
) -> None:
    """2D Grid sweep over w_wram and w_h2d to generate peak intermediate memory heatmap."""
    w_wram_range = np.linspace(0.1, 10.0, 10)
    w_h2d_range = np.linspace(0.1, 10.0, 10)

    grid = np.zeros((len(w_wram_range), len(w_h2d_range)))

    for i, w_wram in enumerate(w_wram_range):
        for j, w_h2d in enumerate(w_h2d_range):
            res = run_single_plan(
                circuit_path,
                root_dir,
                {"w_wram": float(w_wram), "w_h2d": float(w_h2d), "w_d2h": float(w_h2d)},
            )
            grid[i, j] = res["max_intermediate_bytes"]

    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
    cax = ax.matshow(grid, cmap="YlGnBu", origin="lower")
    fig.colorbar(cax, label="Peak Intermediate Tensor Bytes")

    ax.set_xticks(range(len(w_h2d_range)))
    ax.set_yticks(range(len(w_wram_range)))
    ax.set_xticklabels([f"{v:.1f}" for v in w_h2d_range], fontsize=9)
    ax.set_yticklabels([f"{v:.1f}" for v in w_wram_range], fontsize=9)

    ax.set_xlabel("Transfer Weight $w_{\\text{h2d}}$", fontsize=11, fontweight="bold")
    ax.set_ylabel("WRAM Pressure Weight $w_{\\text{wram}}$", fontsize=11, fontweight="bold")
    ax.xaxis.set_ticks_position("bottom")

    plt.title("M7 Peak Intermediate Memory Footprint ($w_{\\text{wram}} \\times w_{\\text{h2d}}$)", fontsize=12, fontweight="bold", pad=15)
    fig.tight_layout()

    out_file = output_dir / "m7_wram_pressure_heatmap.png"
    art_file = artifact_dir / "m7_wram_pressure_heatmap.png"
    plt.savefig(out_file, bbox_inches="tight")
    plt.savefig(art_file, bbox_inches="tight")
    plt.close()
    print(f"Saved Heatmap plot to {out_file} and {art_file}")


def experiment_3_planner_strategy_comparison(
    circuit_path: str,
    root_dir: Path,
    output_dir: Path,
    artifact_dir: Path,
) -> None:
    """Compare 4 planner configurations across FLOPs, Payload Bytes, and Peak Memory."""
    strategies = {
        "Compute-Centric\n(w_flops=10, w_h2d=1)": {"w_flops": 10.0, "w_h2d": 1.0, "w_d2h": 1.0},
        "Balanced PIM\n(w_flops=1, w_h2d=1)": {"w_flops": 1.0, "w_h2d": 1.0, "w_d2h": 1.0},
        "Transfer-Heavy\n(w_flops=1, w_h2d=10)": {"w_flops": 1.0, "w_h2d": 10.0, "w_d2h": 10.0},
        "WRAM-Guard PIM\n(w_wram=20, w_h2d=5)": {"w_flops": 1.0, "w_h2d": 5.0, "w_wram": 20.0},
    }

    labels = list(strategies.keys())
    flops_vals = []
    bytes_vals = []
    peak_mem_vals = []

    for name, cfg in strategies.items():
        res = run_single_plan(circuit_path, root_dir, cfg)
        flops_vals.append(res["total_flops"])
        bytes_vals.append(res["total_bytes"])
        peak_mem_vals.append(res["max_intermediate_bytes"])

    x = np.arange(len(labels))
    width = 0.25

    fig, ax1 = plt.subplots(figsize=(9, 5.5), dpi=300)

    rects1 = ax1.bar(x - width, flops_vals, width, label="Total FLOPs", color="#7570b3")
    rects2 = ax1.bar(x, bytes_vals, width, label="Payload Bytes", color="#1b9e77")
    rects3 = ax1.bar(x + width, peak_mem_vals, width, label="Peak Intermediate Bytes", color="#e7298a")

    ax1.set_ylabel("Metric Magnitude (Log Scale)", fontsize=11, fontweight="bold")
    ax1.set_yscale("log")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=10, fontweight="bold")
    ax1.legend(frameon=True, facecolor="#ffffff", edgecolor="#cccccc")
    ax1.grid(True, which="both", axis="y")

    plt.title("M7 Planner Strategy Comparison Across Hardware Metrics", fontsize=13, fontweight="bold", pad=15)
    fig.tight_layout()

    out_file = output_dir / "m7_planner_strategy_comparison.png"
    art_file = artifact_dir / "m7_planner_strategy_comparison.png"
    plt.savefig(out_file, bbox_inches="tight")
    plt.savefig(art_file, bbox_inches="tight")
    plt.close()
    print(f"Saved Strategy Comparison plot to {out_file} and {art_file}")


def experiment_4_topology_sensitivity(
    circuit_paths: list[str],
    root_dir: Path,
    output_dir: Path,
    artifact_dir: Path,
) -> None:
    """Compare PIM path optimization impact across multiple quantum circuit topologies."""
    topology_names = [Path(p).stem for p in circuit_paths]

    flop_overhead = []
    byte_reduction = []

    for path in circuit_paths:
        res_compute = run_single_plan(path, root_dir, {"w_flops": 10.0, "w_h2d": 1.0})
        res_transfer = run_single_plan(path, root_dir, {"w_flops": 1.0, "w_h2d": 10.0})

        byte_diff_pct = ((res_compute["total_bytes"] - res_transfer["total_bytes"]) / max(1, res_compute["total_bytes"])) * 100.0
        flop_diff_pct = ((res_transfer["total_flops"] - res_compute["total_flops"]) / max(1, res_compute["total_flops"])) * 100.0

        byte_reduction.append(max(0.0, byte_diff_pct))
        flop_overhead.append(max(0.0, flop_diff_pct))

    x = np.arange(len(topology_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)

    ax.bar(x - width/2, byte_reduction, width, label="Payload Byte Reduction (%)", color="#2b5c8f")
    ax.bar(x + width/2, flop_overhead, width, label="FLOP Overhead (%)", color="#d95f02")

    ax.set_ylabel("Percentage (%)", fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(topology_names, rotation=15, fontsize=10)
    ax.legend(frameon=True, facecolor="#ffffff", edgecolor="#cccccc")
    ax.grid(True, axis="y")

    plt.title("M7 Impact by Circuit Topology: Memory Transfer Reduction vs FLOP Overhead", fontsize=12, fontweight="bold", pad=15)
    fig.tight_layout()

    out_file = output_dir / "m7_topology_sensitivity.png"
    art_file = artifact_dir / "m7_topology_sensitivity.png"
    plt.savefig(out_file, bbox_inches="tight")
    plt.savefig(art_file, bbox_inches="tight")
    plt.close()
    print(f"Saved Topology Sensitivity plot to {out_file} and {art_file}")


def main() -> None:
    root_dir = Path(__file__).resolve().parents[3]
    output_dir = root_dir / "docs" / "figures" / "m7_sensitivity_analysis"
    artifact_dir = Path("/home/tom/.gemini/antigravity-cli/brain/28847fec-1a06-4436-a5b2-8cf45e419154")

    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    sample_circuit = "configs/circuits/upmem_m2/one_qubit_ry_h_ry_a.qasm"
    topology_circuits = [
        "configs/circuits/upmem_m2/one_qubit_h.qasm",
        "configs/circuits/upmem_m2/one_qubit_x.qasm",
        "configs/circuits/upmem_m2/one_qubit_ry_h_ry_a.qasm",
        "configs/circuits/upmem_m2/one_qubit_ry_h_ry_b.qasm",
    ]

    print("Running Experiment 1: FLOPs vs Transfer Bytes Tradeoff Sweep...")
    experiment_1_flop_vs_transfer_tradeoff(sample_circuit, root_dir, output_dir, artifact_dir)

    print("Running Experiment 2: WRAM Pressure Heatmap...")
    experiment_2_wram_pressure_heatmap(sample_circuit, root_dir, output_dir, artifact_dir)

    print("Running Experiment 3: Planner Strategy Comparison...")
    experiment_3_planner_strategy_comparison(sample_circuit, root_dir, output_dir, artifact_dir)

    print("Running Experiment 4: Circuit Topology Sensitivity Breakdown...")
    experiment_4_topology_sensitivity(topology_circuits, root_dir, output_dir, artifact_dir)

    print("All M7 Data Science Experiments Completed Successfully!")


if __name__ == "__main__":
    main()
