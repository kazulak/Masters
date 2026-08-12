from __future__ import annotations

from pathlib import Path

import pytest

from quantum_bench.circuits import load_circuit
from quantum_bench.tn.network import build_tensor_network
from quantum_bench.tn.planners import OptEinsumPlanner, planner_from_config
from quantum_bench.tn.task_graph import plan_task_graph_with_config
from quantum_bench.tn.upmem_path_cost_v2 import model_upmem_task_cost_v2, upmem_path_cost_policy_v2
from quantum_bench.tn.upmem_path_optimizer import (
    PathSearchState,
    PIMCostParameters,
    PIMPathCostOptimizer,
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
            "unknown_extra_key": 999.0,
        }
    )
    assert custom_params.w_flops == 5.0
    assert custom_params.w_h2d == 10.0
    assert custom_params.scale_h2d == 2.5
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


def test_task_graph_integration_functional() -> None:
    circuit = load_circuit({"circuit": {"kind": "qasm_file", "path": QASM_SAMPLE}}, ROOT)
    network = build_tensor_network(circuit)

    planner = planner_from_config({"engine": "custom_upmem", "optimize": "upmem_pim_cost_greedy"})
    graph = plan_task_graph_with_config(
        network, {"engine": "custom_upmem", "optimize": "upmem_pim_cost_greedy", "w_h2d": 2.0}
    )

    assert graph is not None
    assert len(graph.tasks) > 0
    assert graph.path_summary.optimize == "upmem_pim_cost_greedy"
