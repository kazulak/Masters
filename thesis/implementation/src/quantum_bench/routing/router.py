from __future__ import annotations

from collections import Counter, defaultdict

from quantum_bench.core.records import JsonDict, TaskGraph
from quantum_bench.routing.records import (
    STATIC_TASK_ROUTER_ID,
    TASK_ROUTE_STATUSES,
    TASK_ROUTE_SUMMARY_SCHEMA_VERSION,
    TaskRouteContext,
    TaskRouteDecision,
    TaskRoutingAnalysis,
)
from quantum_bench.routing.task_routes import TaskRoute, default_task_routes


def route_task_graph(
    graph: TaskGraph,
    context: TaskRouteContext,
    routes: tuple[TaskRoute, ...] | None = None,
) -> TaskRoutingAnalysis:
    task_routes = routes or default_task_routes()
    decisions: list[TaskRouteDecision] = []
    for task_index, task in enumerate(graph.tasks):
        for route in task_routes:
            decisions.append(route.evaluate(task, task_index, context))
    return TaskRoutingAnalysis(
        router_id=STATIC_TASK_ROUTER_ID,
        case_id=context.case_id,
        decisions=tuple(decisions),
        summary=_summary(graph, context, tuple(decisions), task_routes),
    )


def _summary(
    graph: TaskGraph,
    context: TaskRouteContext,
    decisions: tuple[TaskRouteDecision, ...],
    routes: tuple[TaskRoute, ...],
) -> JsonDict:
    status_counts = {status: 0 for status in TASK_ROUTE_STATUSES}
    status_counts.update(Counter(decision.status for decision in decisions))

    route_status_counts: dict[str, dict[str, int]] = {
        route.identity.route_id: {status: 0 for status in TASK_ROUTE_STATUSES} for route in routes
    }
    per_route_counter: dict[str, Counter[str]] = defaultdict(Counter)
    for decision in decisions:
        per_route_counter[decision.route_id][decision.status] += 1
    for route_id, counts in per_route_counter.items():
        route_status_counts.setdefault(route_id, {status: 0 for status in TASK_ROUTE_STATUSES})
        route_status_counts[route_id].update(counts)

    selected_task_ids = {decision.task_id for decision in decisions if decision.is_selected}
    non_fallback_selected_task_ids = {
        decision.task_id
        for decision in decisions
        if decision.is_selected and decision.route_id != "cpu_fallback"
    }
    fallback_task_count = sum(1 for decision in decisions if decision.route_id == "cpu_fallback" and decision.status == "fallback")
    missing_dense_estimate_count = sum(
        1
        for decision in decisions
        if decision.route_id == "dense_gemm" and decision.reason == "missing_target_estimate"
    )

    return {
        "schema_version": TASK_ROUTE_SUMMARY_SCHEMA_VERSION,
        "router_id": STATIC_TASK_ROUTER_ID,
        "case_id": context.case_id,
        "task_count": len(graph.tasks),
        "route_ids": [route.identity.route_id for route in routes],
        "decision_count": len(decisions),
        "status_counts": status_counts,
        "route_status_counts": route_status_counts,
        "selected_task_count": len(selected_task_ids),
        "fallback_task_count": fallback_task_count,
        "non_fallback_selected_task_count": len(non_fallback_selected_task_ids),
        "missing_dense_estimate_count": missing_dense_estimate_count,
        "decisions_artifact": context.decisions_artifact,
        "policy": context.policy,
    }
