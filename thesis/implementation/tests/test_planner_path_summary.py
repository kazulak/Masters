from __future__ import annotations

from dataclasses import replace

import pytest

from quantum_bench.circuits import builtin_circuit
from quantum_bench.targets.upmem import UPMEM_DENSE_ESTIMATE_KEY, annotate_task_graph_with_upmem_estimates
from quantum_bench.tn import build_tensor_network, derive_path_costs, plan_task_graph, plan_task_graph_with_config, planner_from_config, with_path_cost_summary
from quantum_bench.tn.planners import OptEinsumPlanner


def _annotated_graph():
    circuit = builtin_circuit("bell_2q")
    network = build_tensor_network(circuit)
    graph = plan_task_graph(network)
    graph, _ = annotate_task_graph_with_upmem_estimates(graph)
    return with_path_cost_summary(graph)


def test_planner_from_config_returns_opt_einsum_planner() -> None:
    planner = planner_from_config({"engine": "opt_einsum", "optimize": "greedy"})

    assert isinstance(planner, OptEinsumPlanner)
    assert planner.identity.planner_engine == "opt_einsum"
    assert planner.identity.planner_id == "opt_einsum.greedy"
    assert planner.identity.planner_kind == "external_path_optimizer"
    assert planner.identity.optimize_mode == "greedy"
    assert planner.identity.objective == "opt_einsum_contract_path"
    assert planner.identity.cost_basis == "opt_einsum_internal"
    assert planner.identity.target_estimate_key is None
    assert planner.identity.options == {"engine": "opt_einsum", "optimize": "greedy"}


def test_unknown_planner_engine_fails_explicitly() -> None:
    with pytest.raises(ValueError, match="Unsupported planner engine"):
        planner_from_config({"engine": "upmem_aware"})


def test_plan_task_graph_preserves_current_opt_einsum_behavior() -> None:
    circuit = builtin_circuit("bell_2q")
    network = build_tensor_network(circuit)
    graph = plan_task_graph_with_config(network, {"engine": "opt_einsum", "optimize": "greedy"})

    assert graph.tasks
    assert graph.path_summary.planner == "opt_einsum"
    assert graph.path_summary.planner_engine == "opt_einsum"
    assert graph.path_summary.planner_id == "opt_einsum.greedy"
    assert graph.path_summary.optimize == "greedy"
    assert graph.path_summary.optimize_mode == "greedy"
    assert graph.path_summary.task_count == len(graph.tasks)


def test_path_summary_costs_are_derived_from_tasks_and_upmem_estimates() -> None:
    graph = _annotated_graph()
    summary = graph.path_summary

    assert summary.task_count == len(graph.tasks)
    assert summary.total_estimated_flops == sum(task.estimated_flops for task in graph.tasks)
    assert summary.max_intermediate_bytes == max(_output_bytes(task) for task in graph.tasks)
    assert summary.peak_intermediate_bytes >= summary.max_intermediate_bytes
    assert summary.missing_target_estimate_count == 0
    assert summary.total_host_to_dpu_bytes == sum(
        task.target_estimates[UPMEM_DENSE_ESTIMATE_KEY]["host_to_dpu_bytes"] for task in graph.tasks
    )
    assert summary.total_dpu_to_host_bytes == sum(
        task.target_estimates[UPMEM_DENSE_ESTIMATE_KEY]["dpu_to_host_bytes"] for task in graph.tasks
    )
    assert summary.total_mram_to_wram_bytes == sum(
        task.target_estimates[UPMEM_DENSE_ESTIMATE_KEY]["mram_to_wram_bytes"] for task in graph.tasks
    )
    assert summary.unsupported_task_count == sum(
        1 for task in graph.tasks if not task.target_estimates[UPMEM_DENSE_ESTIMATE_KEY]["supported"]
    )
    assert summary.tiling_required_task_count == sum(
        1 for task in graph.tasks if task.target_estimates[UPMEM_DENSE_ESTIMATE_KEY]["requires_tiling"]
    )
    assert summary.estimated_total_tile_count == sum(
        task.target_estimates[UPMEM_DENSE_ESTIMATE_KEY]["estimated_tile_count"] for task in graph.tasks
    )
    assert summary.estimated_max_parallel_tiles == max(
        task.target_estimates[UPMEM_DENSE_ESTIMATE_KEY]["estimated_parallel_tiles"] for task in graph.tasks
    )


def test_missing_target_estimate_count_prevents_silent_zero_costs() -> None:
    graph = _annotated_graph()
    removed_task = replace(graph.tasks[0], target_estimates={})
    graph_with_missing = replace(graph, tasks=(removed_task, *graph.tasks[1:]))
    costs = derive_path_costs(graph_with_missing)

    assert costs["missing_target_estimate_count"] == 1
    assert costs["total_host_to_dpu_bytes"] == sum(
        task.target_estimates[UPMEM_DENSE_ESTIMATE_KEY]["host_to_dpu_bytes"] for task in graph.tasks[1:]
    )
    assert costs["total_dpu_to_host_bytes"] == sum(
        task.target_estimates[UPMEM_DENSE_ESTIMATE_KEY]["dpu_to_host_bytes"] for task in graph.tasks[1:]
    )
    assert costs["total_mram_to_wram_bytes"] == sum(
        task.target_estimates[UPMEM_DENSE_ESTIMATE_KEY]["mram_to_wram_bytes"] for task in graph.tasks[1:]
    )


def _output_bytes(task) -> int:
    total = 1
    for dim in task.output_shape:
        total *= dim
    return total * 16
