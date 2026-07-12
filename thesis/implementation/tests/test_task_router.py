from __future__ import annotations

import json
from dataclasses import replace

from quantum_bench.circuits import builtin_circuit
from quantum_bench.core.records import CircuitSpec, ContractionTask, PathSummary, TaskGraph, TensorNetworkSpec, to_jsonable
from quantum_bench.routing import TASK_ROUTE_DECISION_SCHEMA_VERSION, TASK_ROUTE_SUMMARY_SCHEMA_VERSION, TaskRouteContext, route_task_graph
from quantum_bench.targets.upmem import SIMPLEPIM_PROBE_KEY, UPMEM_DENSE_ESTIMATE_KEY, annotate_task_graph_with_upmem_estimates
from quantum_bench.tn import build_tensor_network, plan_task_graph, with_path_cost_summary


def _context(case_id: str = "unit_case") -> TaskRouteContext:
    return TaskRouteContext(
        suite_id="unit_suite",
        case_id=case_id,
        run_dir=None,
        decisions_artifact=f"cases/{case_id}/task_route_decisions.jsonl",
        backend_probes={
            SIMPLEPIM_PROBE_KEY: {
                "simplepim_available": False,
                "simplepim_probe_status": "unavailable",
                "simplepim_version": None,
                "simplepim_home": None,
                "simplepim_bin": None,
                "simplepim_library_path": None,
                "simplepim_command_path": None,
                "skip_reason": "SimplePIM is not configured in unit tests",
                "metadata": {"external_command_executed": False},
            }
        },
    )


def _annotated_graph() -> TaskGraph:
    circuit = builtin_circuit("bell_2q")
    network = build_tensor_network(circuit)
    graph = plan_task_graph(network)
    graph, _ = annotate_task_graph_with_upmem_estimates(graph)
    return with_path_cost_summary(graph)


def _dense_task(task_id: str, gemm_m: int, gemm_k: int, gemm_n: int, structure: str = "dense") -> ContractionTask:
    return ContractionTask(
        id=task_id,
        input_tensor_ids=(f"{task_id}_left", f"{task_id}_right"),
        output_tensor_id=f"{task_id}_out",
        dependencies=(),
        index_expression="ab,bc->ac",
        input_shapes=((gemm_m, gemm_k), (gemm_k, gemm_n)),
        output_shape=(gemm_m, gemm_n),
        left_labels=(0, 1),
        right_labels=(1, 2),
        contracted_labels=(1,),
        output_labels=(0, 2),
        gemm_m=gemm_m,
        gemm_k=gemm_k,
        gemm_n=gemm_n,
        structure=structure,
        estimated_flops=2 * gemm_m * gemm_k * gemm_n,
        estimated_bytes=gemm_m * gemm_k + gemm_k * gemm_n + gemm_m * gemm_n,
    )


def _task_graph(*tasks: ContractionTask) -> TaskGraph:
    circuit = CircuitSpec("synthetic", 2, (), {})
    network = TensorNetworkSpec(circuit, (), (), "")
    path_summary = PathSummary("unit_test", "manual", len(tasks), None, None, None, "")
    graph = TaskGraph(network, tasks, (), path_summary, 0.0)
    graph, _ = annotate_task_graph_with_upmem_estimates(graph)
    return with_path_cost_summary(graph)


def test_task_router_is_deterministic_and_analysis_only() -> None:
    graph = _annotated_graph()
    analysis = route_task_graph(graph, _context("bell_2q"))
    route_ids = ["dense_gemm", "sparse", "heuristic_bypass", "transpim_support", "cpu_fallback"]
    first_task_route_ids = [decision.route_id for decision in analysis.decisions[: len(route_ids)]]

    assert first_task_route_ids == route_ids
    assert len(analysis.decisions) == len(graph.tasks) * len(route_ids)
    assert analysis.summary["schema_version"] == TASK_ROUTE_SUMMARY_SCHEMA_VERSION
    assert analysis.summary["router_id"] == "static_task_router_v1"
    assert analysis.summary["case_id"] == "bell_2q"
    assert analysis.summary["route_ids"] == route_ids
    assert analysis.summary["decision_count"] == len(analysis.decisions)
    assert analysis.summary["selected_task_count"] == len(graph.tasks)
    assert analysis.summary["fallback_task_count"] == len(graph.tasks)
    assert analysis.summary["non_fallback_selected_task_count"] == 0
    assert analysis.summary["missing_dense_estimate_count"] == 0
    assert analysis.summary["decisions_artifact"] == "cases/bell_2q/task_route_decisions.jsonl"
    assert analysis.summary["policy"] == "analysis_only_cpu_fallback"
    assert analysis.summary["status_counts"]["fallback"] == len(graph.tasks)
    assert analysis.summary["status_counts"]["unavailable"] == len(graph.tasks) * 3
    assert analysis.summary["route_status_counts"]["cpu_fallback"]["fallback"] == len(graph.tasks)

    for decision in analysis.decisions:
        assert decision.schema_version == TASK_ROUTE_DECISION_SCHEMA_VERSION
        assert decision.case_id == "bell_2q"
        assert decision.task_id
        assert decision.input_tensor_ids
        assert decision.output_tensor_id
        assert decision.maturity_level >= 1
        assert decision.reason
        assert decision.execution_status.execution_implemented is False

    fallback_decisions = [decision for decision in analysis.decisions if decision.route_id == "cpu_fallback"]
    assert all(decision.status == "fallback" and decision.is_selected for decision in fallback_decisions)
    assert all(decision.execution_status.state == "fallback_available" for decision in fallback_decisions)

    json.dumps(to_jsonable(analysis))


def test_dense_route_reuses_upmem_task_estimates() -> None:
    graph = _annotated_graph()
    analysis = route_task_graph(graph, _context())
    dense_decisions = [decision for decision in analysis.decisions if decision.route_id == "dense_gemm"]

    assert len(dense_decisions) == len(graph.tasks)
    for decision, task in zip(dense_decisions, graph.tasks):
        source = task.target_estimates[UPMEM_DENSE_ESTIMATE_KEY]
        assert decision.is_selected is False
        assert decision.execution_status.state == "estimate_only"
        assert decision.estimate.metadata["target_estimate_key"] == UPMEM_DENSE_ESTIMATE_KEY
        assert decision.estimate.metadata["target_estimate_model"] == source["model"]
        assert decision.estimate.metadata["conversion_required"] is True
        assert decision.estimate.metadata["intended_route_dtype"] == "int8"
        assert decision.estimate.metadata["conversion_format"] == "fixed_point_symmetric"
        assert decision.estimate.metadata["complex_policy"] == "split_real_imag_last_axis"
        assert decision.estimate.metadata["conversion_artifact"] is None
        assert decision.estimate.metadata["tile_plan_available"] is True
        assert decision.estimate.metadata["tile_plan_artifact"] is None
        assert decision.estimate.metadata["tile_count"] == source["total_tile_count"]
        assert decision.estimate.metadata["working_set_bytes"] == source["max_working_set_bytes"]
        assert decision.estimate.metadata["double_buffer_possible"] == source["double_buffer_possible"]
        assert decision.estimate.metadata["requires_host_aggregation"] == source["requires_host_aggregation"]
        assert decision.estimate.metadata["backend"] == "simplepim_unavailable"
        assert decision.estimate.metadata["simplepim_available"] is False
        assert decision.estimate.metadata["simplepim_probe_status"] == "unavailable"
        assert decision.estimate.metadata["simplepim_version"] is None
        assert decision.estimate.metadata["simplepim_command_path"] is None
        assert decision.estimate.metadata["simplepim_library_path"] is None
        assert decision.estimate.metadata["simplepim_skip_reason"] == "SimplePIM is not configured in unit tests"
        assert decision.estimate.supported == source["supported"]
        assert decision.estimate.wram_fit == source["wram_fit"]
        assert decision.estimate.requires_tiling == source["requires_tiling"]
        assert decision.estimate.tiling_implemented == source["tiling_implemented"]
        assert decision.estimate.host_to_dpu_bytes == source["host_to_dpu_bytes"]
        assert decision.estimate.dpu_to_host_bytes == source["dpu_to_host_bytes"]
        assert decision.estimate.mram_to_wram_bytes == source["mram_to_wram_bytes"]
        assert decision.estimate.estimated_tile_count == source["estimated_tile_count"]
        assert decision.estimate.estimated_parallel_tiles == source["estimated_parallel_tiles"]


def test_missing_dense_estimate_is_explicitly_skipped() -> None:
    graph = _annotated_graph()
    task_without_estimate = replace(graph.tasks[0], target_estimates={})
    graph = replace(graph, tasks=(task_without_estimate, *graph.tasks[1:]))
    analysis = route_task_graph(graph, _context())
    dense_decision = next(decision for decision in analysis.decisions if decision.route_id == "dense_gemm")

    assert dense_decision.status == "skipped"
    assert dense_decision.reason == "missing_target_estimate"
    assert dense_decision.estimate.supported is False
    assert dense_decision.estimate.reason == "missing_target_estimate"
    assert analysis.summary["missing_dense_estimate_count"] == 1


def test_large_dense_task_is_rejected_as_tiling_not_implemented() -> None:
    graph = _task_graph(_dense_task("large", 256, 256, 256))
    analysis = route_task_graph(graph, _context("large_case"))
    dense_decision = next(decision for decision in analysis.decisions if decision.route_id == "dense_gemm")

    assert dense_decision.status == "rejected"
    assert dense_decision.is_selected is False
    assert dense_decision.reason == "requires_tiling_not_implemented"
    assert dense_decision.estimate.wram_fit is True
    assert dense_decision.estimate.requires_tiling is True
    assert dense_decision.estimate.tiling_implemented is False
    assert dense_decision.estimate.estimated_tile_count > 1
    assert dense_decision.estimate.metadata["tile_plan_available"] is True
    assert dense_decision.estimate.metadata["tile_count"] == dense_decision.estimate.estimated_tile_count
    assert dense_decision.estimate.metadata["working_set_bytes"] <= 64 * 1024


def test_future_route_slots_are_unavailable_with_reasons() -> None:
    graph = _task_graph(_dense_task("small", 8, 8, 8))
    analysis = route_task_graph(graph, _context())
    future_decisions = [
        decision
        for decision in analysis.decisions
        if decision.route_id in {"sparse", "heuristic_bypass", "transpim_support"}
    ]

    assert len(future_decisions) == 3
    for decision in future_decisions:
        assert decision.status == "unavailable"
        assert decision.is_selected is False
        assert decision.reason
        assert "not implemented" in decision.reason
        assert decision.execution_status.state == "future_backend"
