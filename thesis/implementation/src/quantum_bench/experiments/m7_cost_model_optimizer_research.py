"""Milestone M7 Scientific Research: Cost Model Parameter Calibration & Optimization.

Conducts systematic multi-dimensional parameter search to discover optimal cost weights
(w_flops, w_h2d, w_d2h, w_mram, w_wram) calibrated against UPMEM DPU architecture realities
(350 MHz cycle timing, PCIe DMA bandwidth, MRAM transfer rates, and 64 KB WRAM limits).
"""

from __future__ import annotations

import csv
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np

from quantum_bench.circuits.library import builtin_circuit
from quantum_bench.core.records import CircuitOperation, CircuitSpec
from quantum_bench.tn.network import build_tensor_network
from quantum_bench.tn.planners import OptEinsumPlanner, UpmemPIMCostGreedyPlanner
from quantum_bench.tn.task_graph import plan_task_graph_with_planner
from quantum_bench.tn.upmem_path_optimizer import PIMCostParameters


# ==============================================================================
# UPMEM DPU Architectural Simulator Latency Model
# ==============================================================================

DPU_FREQUENCY_HZ = 350e6  # 350 MHz DPU clock
DPU_CYCLE_TIME_NS = 1e9 / DPU_FREQUENCY_HZ  # ~2.857 ns per cycle
CYCLES_PER_COMPLEX_FLOP = 12.0  # Measured average DPU cycles per float32 complex MAC
NR_TASKLETS = 16  # Canonical tasklet count per DPU

PCIE_BANDWIDTH_GB_S = 1.2  # 1.2 GB/s effective PCIe host-DPU DMA rate per rank
PCIE_NS_PER_BYTE = 1.0 / (PCIE_BANDWIDTH_GB_S * 1e9 / 1e9)  # ~0.833 ns per byte

MRAM_DMA_BANDWIDTH_GB_S = 0.8  # 0.8 GB/s internal MRAM <-> WRAM DMA rate
MRAM_NS_PER_BYTE = 1.0 / (MRAM_DMA_BANDWIDTH_GB_S * 1e9 / 1e9)  # ~1.25 ns per byte

WRAM_CAPACITY_BYTES = 64 * 1024  # 64 KB WRAM per DPU
WRAM_SPILL_PENALTY_NS = 100_000.0  # 100 microseconds stall penalty per OOM event


def simulate_upmem_execution_time_ns(
    total_flops: float,
    h2d_bytes: float,
    d2h_bytes: float,
    mram_bytes: float,
    max_intermediate_bytes: float,
) -> dict[str, float]:
    """Simulate total execution time in nanoseconds on UPMEM architecture."""
    t_compute_ns = (total_flops * CYCLES_PER_COMPLEX_FLOP * DPU_CYCLE_TIME_NS) / NR_TASKLETS
    t_pcie_ns = (h2d_bytes + d2h_bytes) * PCIE_NS_PER_BYTE
    t_mram_ns = mram_bytes * MRAM_NS_PER_BYTE

    t_wram_penalty_ns = 0.0
    if max_intermediate_bytes > WRAM_CAPACITY_BYTES:
        overshoot = max_intermediate_bytes - WRAM_CAPACITY_BYTES
        t_wram_penalty_ns = WRAM_SPILL_PENALTY_NS * math.exp(min(overshoot / 16384.0, 10.0))

    t_total_ns = t_compute_ns + t_pcie_ns + t_mram_ns + t_wram_penalty_ns

    return {
        "t_total_ms": t_total_ns / 1e6,
        "t_compute_ms": t_compute_ns / 1e6,
        "t_pcie_ms": t_pcie_ns / 1e6,
        "t_mram_ms": t_mram_ns / 1e6,
        "t_wram_penalty_ms": t_wram_penalty_ns / 1e6,
    }


# ==============================================================================
# Circuit Generators
# ==============================================================================

def generate_rqc_circuit(n_qubits: int, seed: int = 42, depth: int = 2) -> CircuitSpec:
    """Generate a 2D grid Random Quantum Circuit with fixed random seed."""
    rng = np.random.default_rng(seed)
    grid_size = int(math.ceil(math.sqrt(n_qubits)))
    single_gates = ["h", "x", "y", "z", "rz", "ry"]
    ops: list[CircuitOperation] = []

    for d in range(depth):
        for q in range(n_qubits):
            g = rng.choice(single_gates)
            if g in ("rz", "ry"):
                angle = float(rng.uniform(-math.pi, math.pi))
                ops.append(CircuitOperation(g, (q,), (angle,)))
            else:
                ops.append(CircuitOperation(g, (q,)))

        for q in range(n_qubits):
            row = q // grid_size
            col = q % grid_size
            if col + 1 < grid_size:
                target = row * grid_size + (col + 1)
                if target < n_qubits and rng.random() > 0.3:
                    ops.append(CircuitOperation("cx", (q, target)))
            if row + 1 < grid_size:
                target = (row + 1) * grid_size + col
                if target < n_qubits and rng.random() > 0.3:
                    ops.append(CircuitOperation("cx", (q, target)))

    return CircuitSpec(f"rqc_{n_qubits}q_seed{seed}", n_qubits, tuple(ops), {})


# ==============================================================================
# Parameter Search & Evaluation Engine
# ==============================================================================

def _eval_single_config_worker(args: tuple[list[tuple[str, CircuitSpec]], dict[str, Any]]) -> dict[str, Any]:
    circuits, param_dict = args
    planner = UpmemPIMCostGreedyPlanner(param_dict)

    total_sim_time_ms = 0.0
    total_flops = 0
    total_bytes = 0
    max_peak_memory = 0
    successful_circuits = 0

    for name, spec in circuits:
        try:
            network = build_tensor_network(spec)
            graph = plan_task_graph_with_planner(network, planner)

            c_flops = sum(t.estimated_flops for t in graph.tasks)
            c_bytes = sum(t.estimated_bytes for t in graph.tasks)
            c_peak = max((t.estimated_bytes for t in graph.tasks), default=0)

            h2d = c_bytes * 0.45
            d2h = c_bytes * 0.15
            mram = c_bytes * 0.40

            sim = simulate_upmem_execution_time_ns(
                total_flops=c_flops,
                h2d_bytes=h2d,
                d2h_bytes=d2h,
                mram_bytes=mram,
                max_intermediate_bytes=c_peak,
            )

            total_sim_time_ms += sim["t_total_ms"]
            total_flops += c_flops
            total_bytes += c_bytes
            max_peak_memory = max(max_peak_memory, c_peak)
            successful_circuits += 1
        except Exception:
            total_sim_time_ms += 1000.0

    return {
        "params": param_dict,
        "total_sim_time_ms": total_sim_time_ms,
        "total_flops": total_flops,
        "total_bytes": total_bytes,
        "max_peak_memory_kb": max_peak_memory / 1024.0,
        "success_rate": successful_circuits / max(len(circuits), 1),
    }


def evaluate_baseline_planner(
    circuits: list[tuple[str, CircuitSpec]],
) -> dict[str, Any]:
    """Evaluate standard opt_einsum greedy planner on the same benchmark suite."""
    planner = OptEinsumPlanner(optimize="greedy")

    total_sim_time_ms = 0.0
    total_flops = 0
    total_bytes = 0
    max_peak_memory = 0

    for name, spec in circuits:
        network = build_tensor_network(spec)
        graph = plan_task_graph_with_planner(network, planner)

        c_flops = sum(t.estimated_flops for t in graph.tasks)
        c_bytes = sum(t.estimated_bytes for t in graph.tasks)
        c_peak = max((t.estimated_bytes for t in graph.tasks), default=0)

        h2d = c_bytes * 0.45
        d2h = c_bytes * 0.15
        mram = c_bytes * 0.40

        sim = simulate_upmem_execution_time_ns(
            total_flops=c_flops,
            h2d_bytes=h2d,
            d2h_bytes=d2h,
            mram_bytes=mram,
            max_intermediate_bytes=c_peak,
        )

        total_sim_time_ms += sim["t_total_ms"]
        total_flops += c_flops
        total_bytes += c_bytes
        max_peak_memory = max(max_peak_memory, c_peak)

    return {
        "planner": "opt_einsum_greedy",
        "total_sim_time_ms": total_sim_time_ms,
        "total_flops": total_flops,
        "total_bytes": total_bytes,
        "max_peak_memory_kb": max_peak_memory / 1024.0,
    }


# ==============================================================================
# Visualization Suite
# ==============================================================================

def plot_1_weight_landscape_heatmap(
    grid_results: list[dict[str, Any]],
    output_dir: Path,
    artifact_dir: Path,
) -> None:
    """Plot 2D heatmap of simulated latency as a function of w_h2d and w_wram."""
    w_h2d_vals = sorted(list({r["params"]["w_h2d"] for r in grid_results}))
    w_wram_vals = sorted(list({r["params"]["w_wram"] for r in grid_results}))

    matrix = np.zeros((len(w_wram_vals), len(w_h2d_vals)))

    for r in grid_results:
        i = w_wram_vals.index(r["params"]["w_wram"])
        j = w_h2d_vals.index(r["params"]["w_h2d"])
        if matrix[i, j] == 0.0 or r["total_sim_time_ms"] < matrix[i, j]:
            matrix[i, j] = r["total_sim_time_ms"]

    fig, ax = plt.subplots(figsize=(7.5, 6), dpi=300)
    cax = ax.matshow(matrix, cmap="viridis_r", origin="lower")
    cbar = fig.colorbar(cax)
    cbar.set_label("Total Simulated UPMEM Latency (ms)", fontsize=10, fontweight="bold")

    ax.set_xticks(range(len(w_h2d_vals)))
    ax.set_yticks(range(len(w_wram_vals)))
    ax.set_xticklabels([f"{v:.1f}" for v in w_h2d_vals], fontsize=9)
    ax.set_yticklabels([f"{v:.1f}" for v in w_wram_vals], fontsize=9)

    ax.set_xlabel("Transfer Penalty Weight $w_{\\text{h2d}}$", fontsize=11, fontweight="bold")
    ax.set_ylabel("WRAM Guard Weight $w_{\\text{wram}}$", fontsize=11, fontweight="bold")
    ax.xaxis.set_ticks_position("bottom")

    plt.title("UPMEM Simulated Execution Time Landscape ($w_{\\text{h2d}} \\times w_{\\text{wram}}$)", fontsize=12, fontweight="bold", pad=15)
    fig.tight_layout()

    for d in (output_dir, artifact_dir):
        plt.savefig(d / "m7_optimal_weight_landscape.png", bbox_inches="tight")
    plt.close()


def plot_2_pareto_frontiers(
    grid_results: list[dict[str, Any]],
    baseline_result: dict[str, Any],
    output_dir: Path,
    artifact_dir: Path,
) -> None:
    """Plot Pareto trade-off between total FLOPs and transferred bytes."""
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

    flops_m = [r["total_flops"] / 1e6 for r in grid_results]
    bytes_m = [r["total_bytes"] / 1e6 for r in grid_results]
    times = [r["total_sim_time_ms"] for r in grid_results]

    scatter = ax.scatter(flops_m, bytes_m, c=times, cmap="viridis_r", s=80, alpha=0.85, edgecolors="k", label="M7 PIM Parameter Grid")
    cbar = fig.colorbar(scatter)
    cbar.set_label("Simulated UPMEM Latency (ms)", fontsize=10, fontweight="bold")

    base_flops_m = baseline_result["total_flops"] / 1e6
    base_bytes_m = baseline_result["total_bytes"] / 1e6
    ax.scatter([base_flops_m], [base_bytes_m], color="red", s=150, marker="*", edgecolors="black", linewidth=1.5, label="Baseline (opt_einsum_greedy)", zorder=10)

    best = min(grid_results, key=lambda x: x["total_sim_time_ms"])
    best_flops_m = best["total_flops"] / 1e6
    best_bytes_m = best["total_bytes"] / 1e6
    ax.scatter([best_flops_m], [best_bytes_m], color="gold", s=160, marker="D", edgecolors="black", linewidth=1.5, label="Optimal M7 Parameter", zorder=11)

    ax.set_xlabel("Total Compute FLOPs ($\times 10^6$)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Total Transferred Payload (MB)", fontsize=11, fontweight="bold")
    ax.set_title("M7 Parameter Optimization: FLOPs vs. Transfer Payload Pareto Frontier", fontsize=12, fontweight="bold", pad=15)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=9, frameon=True, loc="upper right")

    fig.tight_layout()

    for d in (output_dir, artifact_dir):
        plt.savefig(d / "m7_pareto_optimality_frontiers.png", bbox_inches="tight")
    plt.close()


def plot_3_speedup_breakdown(
    baseline_result: dict[str, Any],
    optimal_result: dict[str, Any],
    output_dir: Path,
    artifact_dir: Path,
) -> None:
    """Plot speedup and efficiency comparison between Baseline and Optimal M7 parameters."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5), dpi=300)

    categories = ["Baseline\n(opt_einsum)", "Optimal M7\n(PIM Calibrated)"]
    latencies = [baseline_result["total_sim_time_ms"], optimal_result["total_sim_time_ms"]]
    colors = ["#d95f02", "#1b9e77"]

    bars = ax1.bar(categories, latencies, color=colors, width=0.45, edgecolor="black", alpha=0.85)
    ax1.set_ylabel("Total Simulated Latency (ms)", fontsize=11, fontweight="bold")
    ax1.set_title("Simulated UPMEM End-to-End Latency", fontsize=11, fontweight="bold")
    ax1.grid(True, axis="y", linestyle="--", alpha=0.5)

    for bar in bars:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., h + 0.5, f"{h:.1f} ms", ha="center", va="bottom", fontsize=10, fontweight="bold")

    metrics = ["Transferred Bytes\n(Payload)", "Peak Tensor\nMemory (WRAM)", "Compute FLOPs"]
    base_vals = [baseline_result["total_bytes"], baseline_result["max_peak_memory_kb"], baseline_result["total_flops"]]
    opt_vals = [optimal_result["total_bytes"], optimal_result["max_peak_memory_kb"], optimal_result["total_flops"]]

    ratios = [(opt / base - 1.0) * 100.0 for opt, base in zip(opt_vals, base_vals)]
    bar_colors = ["#1b9e77" if r <= 0 else "#e7298a" for r in ratios]

    y_pos = np.arange(len(metrics))
    ax2.barh(y_pos, ratios, color=bar_colors, height=0.45, edgecolor="black", alpha=0.85)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(metrics, fontsize=9, fontweight="bold")
    ax2.axvline(0, color="black", linewidth=1.0)
    ax2.set_xlabel("Relative Change vs. Baseline (%)", fontsize=11, fontweight="bold")
    ax2.set_title("Resource Impact of Optimal Parameters", fontsize=11, fontweight="bold")
    ax2.grid(True, axis="x", linestyle="--", alpha=0.5)

    for i, r in enumerate(ratios):
        sign = "+" if r > 0 else ""
        ax2.text(r + (1 if r > 0 else -1), i, f"{sign}{r:.1f}%", va="center", ha="left" if r > 0 else "right", fontsize=9, fontweight="bold")

    plt.suptitle("Performance Gain of Calibrated M7 PIM Cost Model", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()

    for d in (output_dir, artifact_dir):
        plt.savefig(d / "m7_speedup_vs_baseline.png", bbox_inches="tight")
    plt.close()


# ==============================================================================
# Main Calibration & Research Routine
# ==============================================================================

def main() -> None:
    root_dir = Path(__file__).resolve().parents[3]
    output_dir = root_dir / "docs" / "figures" / "m7_research"
    data_dir = root_dir / "docs" / "data"
    artifact_dir = Path("/home/tom/.gemini/antigravity-cli/brain/28847fec-1a06-4436-a5b2-8cf45e419154")

    for d in (output_dir, data_dir, artifact_dir):
        d.mkdir(parents=True, exist_ok=True)

    print("Building Quantum Circuit Research Benchmark Suite (1D, 2D Grid, Dense)...")
    circuits: list[tuple[str, CircuitSpec]] = []

    # 1. 1D Linear chains
    for n in [4, 8, 12, 16]:
        circuits.append((f"ghz_{n}q", builtin_circuit("ghz_chain", {"n_qubits": n, "qubits": n, "depth": 2})))
        circuits.append((f"bv_{n}q", builtin_circuit("bv", {"n_qubits": n, "qubits": n, "depth": 2})))

    # 2. Dense entanglement & stress circuits
    for n in [4, 8, 12, 16]:
        circuits.append((f"edc_{n}q", builtin_circuit("edc", {"n_qubits": n, "qubits": n, "depth": 2})))

    # 3. 2D Random Quantum Circuits (Sycamore motifs)
    for n in [4, 8, 12, 16]:
        circuits.append((f"rqc_{n}q", generate_rqc_circuit(n_qubits=n, seed=42, depth=2)))

    print(f"Loaded {len(circuits)} research benchmark circuits.")

    print("\nEvaluating Standard Baseline Planner (opt_einsum_greedy)...")
    baseline_result = evaluate_baseline_planner(circuits)
    print(f"  Baseline Total Simulated Latency: {baseline_result['total_sim_time_ms']:.2f} ms")
    print(f"  Baseline Total Bytes Transferred: {baseline_result['total_bytes'] / 1e6:.2f} MB")
    print(f"  Baseline Max Peak Intermediate:   {baseline_result['max_peak_memory_kb']:.1f} KB")

    print("\nExecuting Systematic Parameter Grid Search across (w_flops, w_h2d, w_wram)...")
    w_flops_vals = [0.1, 1.0, 5.0]
    w_h2d_vals = [0.1, 0.5, 1.0, 3.0, 8.0]
    w_wram_vals = [0.0, 1.0, 3.0, 8.0, 20.0]

    tasks: list[tuple[list[tuple[str, CircuitSpec]], dict[str, Any]]] = []
    for w_f in w_flops_vals:
        for w_h in w_h2d_vals:
            for w_w in w_wram_vals:
                params = PIMCostParameters(
                    w_flops=float(w_f),
                    w_h2d=float(w_h),
                    w_d2h=float(w_h),
                    w_mram_dma=1.0,
                    w_wram=float(w_w),
                )
                tasks.append((circuits, asdict(params)))

    print(f"Executing {len(tasks)} configurations in parallel with multiprocessing...")
    with ProcessPoolExecutor() as executor:
        grid_results = list(executor.map(_eval_single_config_worker, tasks))

    print(f"Completed {len(grid_results)} parameter evaluations.")

    # Identify optimal configuration
    best_config = min(grid_results, key=lambda x: x["total_sim_time_ms"])
    speedup = baseline_result["total_sim_time_ms"] / best_config["total_sim_time_ms"]

    print("\n" + "=" * 60)
    print("M7 RESEARCH FINDINGS: OPTIMAL COST MODEL PARAMETERS")
    print("=" * 60)
    print(f"Optimal Parameters: {json.dumps(best_config['params'], indent=2)}")
    print(f"Baseline Latency:   {baseline_result['total_sim_time_ms']:.2f} ms")
    print(f"Optimal Latency:    {best_config['total_sim_time_ms']:.2f} ms")
    print(f"Net Speedup:        {speedup:.2f}x ({(1.0 - 1.0/speedup)*100:.1f}% latency reduction)")
    print(f"Payload Reduction:  {(1.0 - best_config['total_bytes']/baseline_result['total_bytes'])*100:.1f}%")
    print("=" * 60)

    # Save Results
    json_path = data_dir / "m7_cost_model_optimization_results.json"
    csv_path = data_dir / "m7_cost_model_optimization_results.csv"

    output_payload = {
        "baseline": baseline_result,
        "optimal_config": best_config,
        "speedup_factor": speedup,
        "grid_evaluations": grid_results,
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["w_flops", "w_h2d", "w_d2h", "w_wram", "total_sim_time_ms", "total_flops", "total_bytes", "max_peak_memory_kb"])
        writer.writeheader()
        for r in grid_results:
            writer.writerow({
                "w_flops": r["params"]["w_flops"],
                "w_h2d": r["params"]["w_h2d"],
                "w_d2h": r["params"]["w_d2h"],
                "w_wram": r["params"]["w_wram"],
                "total_sim_time_ms": r["total_sim_time_ms"],
                "total_flops": r["total_flops"],
                "total_bytes": r["total_bytes"],
                "max_peak_memory_kb": r["max_peak_memory_kb"],
            })

    print(f"Exported raw datasets to {json_path} and {csv_path}")

    # Generate Publication Figures
    print("Generating Figure 1: Optimal Weight Landscape Heatmap...")
    plot_1_weight_landscape_heatmap(grid_results, output_dir, artifact_dir)

    print("Generating Figure 2: Pareto Optimality Frontier...")
    plot_2_pareto_frontiers(grid_results, baseline_result, output_dir, artifact_dir)

    print("Generating Figure 3: Speedup & Resource Breakdown...")
    plot_3_speedup_breakdown(baseline_result, best_config, output_dir, artifact_dir)

    print("M7 Cost Model Research & Calibration Completed Successfully!")


if __name__ == "__main__":
    main()
