from __future__ import annotations

from quantum_bench.circuits import builtin_circuit
from quantum_bench.tn.network import build_tensor_network
from quantum_bench.tn.planners import OptEinsumPlanner, PlannerResult
from quantum_bench.tn.task_graph import (
    materialize_task_graph_from_planner_result,
    plan_task_graph,
    plan_task_graph_with_planner,
)


class FixedPlanner:
    def __init__(self, result: PlannerResult) -> None:
        self.identity = result.identity
        self.result = result
        self.calls = 0

    def plan(self, network: object) -> PlannerResult:
        self.calls += 1
        return self.result


def test_materialize_uses_supplied_result_without_planner_invocation() -> None:
    network = build_tensor_network(builtin_circuit("bell_2q"))
    planner_result = OptEinsumPlanner().plan(network)

    graph = materialize_task_graph_from_planner_result(network, planner_result)

    assert graph.path == planner_result.path
    assert graph.path_summary.planner_id == planner_result.identity.planner_id


def test_plan_task_graph_with_planner_calls_plan_once() -> None:
    network = build_tensor_network(builtin_circuit("bell_2q"))
    planner_result = OptEinsumPlanner().plan(network)
    planner = FixedPlanner(planner_result)

    graph = plan_task_graph_with_planner(network, planner)

    assert planner.calls == 1
    assert graph.path == planner_result.path


def test_planner_wrapper_matches_direct_materialization() -> None:
    network = build_tensor_network(builtin_circuit("bell_2q"))
    planner_result = OptEinsumPlanner().plan(network)

    expected = materialize_task_graph_from_planner_result(network, planner_result)
    actual = plan_task_graph_with_planner(network, FixedPlanner(planner_result))
    default = plan_task_graph(network)

    assert actual == expected
    assert default.tasks == expected.tasks
    assert default.path == expected.path
    assert default.path_summary == expected.path_summary
    assert default.contraction_plan_hash == expected.contraction_plan_hash
