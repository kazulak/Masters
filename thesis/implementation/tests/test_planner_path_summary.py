from __future__ import annotations

from dataclasses import replace

import pytest
import numpy as np

from quantum_bench.circuits import builtin_circuit, quest_compatible_circuit
from quantum_bench.core.indices import LABEL_LIST_EINSUM_SENTINEL, is_label_list_einsum_expression
from quantum_bench.core.records import ContractionTask
from quantum_bench.targets.upmem import UPMEM_DENSE_ESTIMATE_KEY, annotate_task_graph_with_upmem_estimates
from quantum_bench.tn import build_tensor_network, derive_path_costs, plan_task_graph, plan_task_graph_with_config, planner_from_config, with_path_cost_summary
from quantum_bench.tn.contract import contract_binary_task
from quantum_bench.tn.planner_motifs import build_planner_motif_workload
from quantum_bench.tn.planners import CotengraPlanner, OptEinsumPlanner
from quantum_bench.tn.upmem_planner import (
    PlannerInfeasibleError,
    UpmemAwareGreedyPlanner,
    UpmemAwareProjectedPrefixPlanner,
)


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
    assert planner.identity.planner_config_hash
    assert planner.identity.planner_config["engine"] == "opt_einsum"
    assert planner.identity.options["planner_config_hash"] == planner.identity.planner_config_hash


def test_random_greedy_records_seed_policy_and_version_metadata() -> None:
    planner = planner_from_config({"engine": "opt_einsum", "optimize": "random-greedy"})

    assert planner.identity.planner_id == "opt_einsum.random-greedy"
    assert planner.identity.planner_config["random_seed_policy"] == "opt_einsum_random_optimizer_run_index_v1"
    assert planner.identity.planner_config["random_seed_version"] == "opt_einsum.path_random.RandomOptimizer"
    assert planner.identity.planner_config["opt_einsum_version"]


def test_unknown_planner_engine_fails_explicitly() -> None:
    with pytest.raises(ValueError, match="Unsupported planner engine"):
        planner_from_config({"engine": "upmem_aware"})


def test_cotengra_baseline_adapter_returns_pairwise_path() -> None:
    planner = planner_from_config(
        {"engine": "cotengra", "objective": "flops", "methods": "greedy", "max_repeats": 1}
    )
    assert isinstance(planner, CotengraPlanner)
    assert planner.identity.planner_id == "cotengra.flops"
    assert planner.identity.planner_config["optlib"] == "random"
    assert planner.identity.planner_config["seed"] == 0
    assert planner.identity.planner_config["parallel"] is False

    graph = plan_task_graph_with_config(
        build_tensor_network(builtin_circuit("bell_2q")),
        {"engine": "cotengra", "objective": "flops", "methods": "greedy", "max_repeats": 1},
    )

    assert graph.tasks
    assert graph.path_summary.planner_engine == "cotengra"
    assert graph.path_summary.objective == "cotengra_flops"
    assert all(len(step) == 2 for step in graph.path)


def test_cotengra_seeded_planning_is_repeatable() -> None:
    config = {"engine": "cotengra", "objective": "flops", "methods": "greedy", "max_repeats": 1}
    network = build_tensor_network(builtin_circuit("ghz_chain", {"n_qubits": 4}))

    first = plan_task_graph_with_config(network, config)
    second = plan_task_graph_with_config(network, config)

    assert first.path == second.path
    assert first.path_summary.planner_id == second.path_summary.planner_id == "cotengra.flops"
    assert first.path_summary.options["planner_config_hash"] == second.path_summary.options["planner_config_hash"]
    assert first.path_summary.planner_metadata["cotengra_seed"] == 0


def test_custom_upmem_planner_is_deterministic_and_uses_shared_taskgraph_lowering() -> None:
    config = {
        "engine": "custom_upmem",
        "algorithm": "greedy",
        "weight_profile": "balanced_literature_informed",
    }
    first = planner_from_config(config)
    second = planner_from_config(config)
    assert isinstance(first, UpmemAwareGreedyPlanner)
    assert isinstance(second, UpmemAwareGreedyPlanner)
    assert first.identity == second.identity

    network = build_planner_motif_workload(
        {
            "case_id": "planner_motif_chain",
            "circuit": {"kind": "planner_motif", "name": "chain"},
            "metadata": {
                "workload_type": "synthetic_planner_motif",
                "execution_scope": "model_only",
                "not_real_quantum_circuit": True,
            },
        }
    ).network
    first_graph = plan_task_graph_with_config(network, config)
    second_graph = plan_task_graph_with_config(network, config)

    assert first_graph.path == second_graph.path
    assert first_graph.path_summary.planner_engine == "custom_upmem"
    assert first_graph.path_summary.planner_kind == "native_target_greedy"
    assert first_graph.path_summary.objective == "upmem_path_cost_v1"
    assert first_graph.path_summary.planner_metadata["execution_plan_executed"] is False
    assert len(first_graph.tasks) == len(network.tensors) - 1


def test_custom_upmem_planner_rejects_complex_generic_inputs_explicitly() -> None:
    network = build_tensor_network(builtin_circuit("quantization_stress", {"n_qubits": 2}))
    planner = planner_from_config({"engine": "custom_upmem"})

    with pytest.raises(PlannerInfeasibleError, match="complex") as error:
        planner.plan(network)

    assert error.value.rejection_reasons == ("complex_generic_loop_not_implemented",)


def test_custom_upmem_v2_config_is_explicit_and_distinct_from_v1_identity() -> None:
    v1 = planner_from_config({"engine": "custom_upmem", "weight_profile": "balanced_literature_informed"})
    v2 = planner_from_config(
        {
            "engine": "custom_upmem",
            "objective_version": "upmem_path_cost_v2",
            "selection_scope": "projected_prefix",
            "weight_profile": "balanced_literature_informed",
            "normalization": "fixed_log1p_generic_budgets_v2",
            "execution_policy": "generic_single_dpu_split_complex_v2",
        }
    )

    assert isinstance(v2, UpmemAwareProjectedPrefixPlanner)
    assert isinstance(v1, UpmemAwareGreedyPlanner)
    assert v1.identity.planner_config_hash != v2.identity.planner_config_hash
    assert v2.identity.planner_config["objective_version"] == "upmem_path_cost_v2"
    assert v2.identity.planner_config["selection_scope"] == "projected_prefix"


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


def test_plan_task_graph_handles_more_labels_than_numpy_einsum_symbols() -> None:
    circuit = quest_compatible_circuit("EDC", {"n_qubits": 10})
    network = build_tensor_network(circuit)

    assert is_label_list_einsum_expression(network.spec.einsum_expression)

    graph = plan_task_graph_with_config(network, {"engine": "opt_einsum", "optimize": "greedy"})

    assert graph.tasks
    assert graph.path_summary.planner == "opt_einsum"
    assert all("," in task.index_expression and "->" in task.index_expression for task in graph.tasks)


def test_plan_task_graph_handles_large_binary_tasks_beyond_numpy_einsum_symbols() -> None:
    circuit = quest_compatible_circuit("BV", {"n_qubits": 18})
    network = build_tensor_network(circuit)

    assert is_label_list_einsum_expression(network.spec.einsum_expression)

    graph = plan_task_graph_with_config(network, {"engine": "opt_einsum", "optimize": "greedy"})

    assert graph.tasks
    assert not any(is_label_list_einsum_expression(task.index_expression) for task in graph.tasks)
    assert max(len(task.output_labels) for task in graph.tasks) == circuit.n_qubits
    assert graph.path_summary.planner == "opt_einsum"


def test_large_label_binary_task_contracts_without_numpy_einsum_symbols() -> None:
    left_labels = tuple(range(32))
    right_labels = tuple(range(32, 63))
    output_labels = left_labels + right_labels
    left = np.ones((1,) * len(left_labels), dtype=np.complex128)
    right = np.ones((1,) * len(right_labels), dtype=np.complex128)
    task = ContractionTask(
        id="task_large_label",
        input_tensor_ids=("left", "right"),
        output_tensor_id="out",
        dependencies=(),
        index_expression=f"{LABEL_LIST_EINSUM_SENTINEL}:task_labels=63",
        input_shapes=(left.shape, right.shape),
        output_shape=(1,) * len(output_labels),
        left_labels=left_labels,
        right_labels=right_labels,
        contracted_labels=(),
        output_labels=output_labels,
        gemm_m=1,
        gemm_k=1,
        gemm_n=1,
        structure="dense",
        estimated_flops=1,
        estimated_bytes=1,
    )

    output = contract_binary_task(task, left, right)

    assert output.shape == task.output_shape
    assert output.item() == 1.0 + 0.0j


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
