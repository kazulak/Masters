from __future__ import annotations

from pathlib import Path

import pytest

from quantum_bench.circuits import load_circuit
from quantum_bench.tn.network import build_tensor_network
from quantum_bench.tn.planners import OptEinsumPlanner, UpmemPIMCostGreedyPlanner, planner_from_config
from quantum_bench.tn.task_graph import plan_task_graph_with_config
from quantum_bench.tn.upmem_path_cost_v2 import model_upmem_task_cost_v2, upmem_path_cost_policy_v2
from quantum_bench.tn.upmem_path_optimizer import (
    PathSearchState,
    PIMCostParameters,
    PIMPathCostOptimizer,
    _greedy_search_pure,
    calculate_pim_step_cost,
    eval_pair_step,
    make_sim_contraction_task,
    pim_path_finder_functional,
)

ROOT = Path(__file__).resolve().parents[1]
QASM_SAMPLE = "configs/circuits/upmem_m2/one_qubit_ry_h_ry_a.qasm"


def test_pim_cost_parameters_inspection() -> None:
    # Test default parameters
    default_params = PIMCostParameters.from_config(None)
    assert default_params.w_flops == 1.0
    assert default_params.w_h2d == 1.0

    # Test parameter creation from dictionary mapping with extra unknown keys ignored
    custom_params = PIMCostParameters.from_config(
        {
            "w_flops": 5.0,
            "w_h2d": 10.0,
            "scale_h2d": 2.5,
            "memory_limit": 1024,
            "unknown_extra_key": 999.0,
        }
    )
    assert custom_params.w_flops == 5.0
    assert custom_params.w_h2d == 10.0
    assert custom_params.scale_h2d == 2.5
    assert custom_params.memory_limit == 1024
    assert not hasattr(custom_params, "unknown_extra_key")


def test_eval_pair_step_pure_state_transition() -> None:
    params = PIMCostParameters(w_flops=2.0)
    size_dict = {"a": 16, "b": 16, "c": 16}
    initial_state = PathSearchState(
        active_tensors=(("a", "b"), ("b", "c")),
        size_dict=size_dict,
        history=(),
        total_cost=0.0,
        parameters=params,
        output_labels=("a", "c"),
    )

    next_state, step_cost = eval_pair_step(initial_state, (0, 1))

    # Assert immutability: initial_state remains unchanged
    assert initial_state.history == ()
    assert initial_state.total_cost == 0.0
    assert len(initial_state.active_tensors) == 2

    # Assert next_state receives transformed state
    assert next_state.history == ((0, 1),)
    assert next_state.total_cost == step_cost
    assert len(next_state.active_tensors) == 1
    assert step_cost > 0.0


def test_batched_gemm_flop_calculation() -> None:
    # Contraction with batched index 'batch' (size 10), contracted index 'k' (size 4), left 'm' (size 2), right 'n' (size 3)
    size_dict = {"batch": 10, "k": 4, "m": 2, "n": 3}
    task = make_sim_contraction_task(
        task_id="batched_test",
        left_labels=("batch", "m", "k"),
        right_labels=("batch", "k", "n"),
        output_labels=("batch", "m", "n"),
        size_dict=size_dict,
    )

    # B = 10, M = 2, N = 3, K = 4
    # Expected FLOPs = 8 * B * M * N * K = 8 * 10 * 2 * 3 * 4 = 1920
    assert task.estimated_flops == 1920


def test_memory_limit_enforcement_and_exception() -> None:
    # Set a tiny memory limit of 1 byte so that any intermediate payload exceeds it
    params = PIMCostParameters(memory_limit=1)
    size_dict = {"a": 16, "b": 16, "c": 16}

    # Step cost evaluates to inf
    cost = calculate_pim_step_cost(
        left_labels=("a", "b"),
        right_labels=("b", "c"),
        out_labels=("a", "c"),
        size_dict=size_dict,
        params=params,
    )
    assert cost == float("inf")

    # Search raises ValueError instead of silently ignoring feasibility or returning a poisoned path
    initial_state = PathSearchState(
        active_tensors=(("a", "b"), ("b", "c")),
        size_dict=size_dict,
        history=(),
        total_cost=0.0,
        parameters=params,
        output_labels=("a", "c"),
    )
    with pytest.raises(ValueError, match="Contraction path search failed"):
        _greedy_search_pure(initial_state)


def test_weight_sensitivity() -> None:
    params_default = PIMCostParameters()
    params_h2d_heavy = PIMCostParameters(w_h2d=100.0)

    size_dict = {"a": 16, "b": 16, "c": 16}
    cost_default = calculate_pim_step_cost(
        left_labels=("a", "b"),
        right_labels=("b", "c"),
        out_labels=("a", "c"),
        size_dict=size_dict,
        params=params_default,
    )
    cost_h2d_heavy = calculate_pim_step_cost(
        left_labels=("a", "b"),
        right_labels=("b", "c"),
        out_labels=("a", "c"),
        size_dict=size_dict,
        params=params_h2d_heavy,
    )

    assert cost_default > 0.0
    assert cost_h2d_heavy > cost_default


def test_iterative_greedy_search_no_recursion_limit() -> None:
    # Construct a chain of 150 tensors to verify iterative execution without RecursionError
    active_tensors = tuple((f"idx_{i}", f"idx_{i+1}") for i in range(150))
    size_dict = {f"idx_{i}": 2 for i in range(151)}
    initial_state = PathSearchState(
        active_tensors=active_tensors,
        size_dict=size_dict,
        history=(),
        total_cost=0.0,
        parameters=PIMCostParameters(),
        output_labels=("idx_0", "idx_150"),
    )

    # Perform search using iterative loop
    final_state = _greedy_search_pure(initial_state)
    assert len(final_state.history) == 149
    assert len(final_state.active_tensors) == 1


def test_dedicated_upmem_pim_cost_greedy_planner() -> None:
    circuit = load_circuit({"circuit": {"kind": "qasm_file", "path": QASM_SAMPLE}}, ROOT)
    network = build_tensor_network(circuit)

    planner = planner_from_config({"engine": "custom_upmem", "optimize": "upmem_pim_cost_greedy"})
    assert isinstance(planner, UpmemPIMCostGreedyPlanner)
    assert planner.identity.planner_id == "upmem_pim_cost_greedy"

    graph = plan_task_graph_with_config(
        network, {"engine": "custom_upmem", "optimize": "upmem_pim_cost_greedy", "w_h2d": 2.0}
    )

    assert graph is not None
    assert len(graph.tasks) > 0
    assert graph.path_summary.optimize == "upmem_pim_cost_greedy"
