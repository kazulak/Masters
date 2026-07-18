from __future__ import annotations

from typing import Protocol

from quantum_bench.core.records import ContractionTask, JsonDict
from quantum_bench.routing.records import (
    STATIC_TASK_ROUTER_ID,
    TASK_ROUTE_DECISION_SCHEMA_VERSION,
    TaskRouteCapabilities,
    TaskRouteContext,
    TaskRouteDecision,
    TaskRouteDecisionStatus,
    TaskRouteEstimate,
    TaskRouteExecutionStatus,
    TaskRouteIdentity,
)


class TaskRoute(Protocol):
    identity: TaskRouteIdentity

    def capabilities(self, context: TaskRouteContext) -> TaskRouteCapabilities:
        ...

    def evaluate(self, task: ContractionTask, task_index: int, context: TaskRouteContext) -> TaskRouteDecision:
        ...


class SparseTaskRoute:
    identity = TaskRouteIdentity(
        route_id="sparse",
        display_name="Sparse task route",
        route_family="sparse",
        kernel_family="sparse_spmm",
        hardware_target="upmem_dpu",
        execution_mode="future_upmem_or_native",
        maturity_level=1,
    )

    def capabilities(self, context: TaskRouteContext) -> TaskRouteCapabilities:
        return _future_capabilities(self.identity, "sparse detection, conversion, and kernels are not implemented")

    def evaluate(self, task: ContractionTask, task_index: int, context: TaskRouteContext) -> TaskRouteDecision:
        reason = "sparse task route unavailable: sparse detection, conversion, and kernels are not implemented"
        return _future_unavailable_decision(self.identity, task, task_index, context, reason)


class HeuristicBypassTaskRoute:
    identity = TaskRouteIdentity(
        route_id="heuristic_bypass",
        display_name="Heuristic bypass task route",
        route_family="heuristic_bypass",
        kernel_family="heuristic_permutation",
        hardware_target="host_or_upmem_dpu",
        execution_mode="future_analysis_or_native",
        maturity_level=1,
    )

    def capabilities(self, context: TaskRouteContext) -> TaskRouteCapabilities:
        return _future_capabilities(self.identity, "heuristic/permutation bypass detection and execution are not implemented")

    def evaluate(self, task: ContractionTask, task_index: int, context: TaskRouteContext) -> TaskRouteDecision:
        reason = "heuristic_bypass route unavailable: bypass detection and execution are not implemented"
        return _future_unavailable_decision(self.identity, task, task_index, context, reason)


class TransPimSupportTaskRoute:
    identity = TaskRouteIdentity(
        route_id="transpim_support",
        display_name="TransPimLib/math-support task route",
        route_family="transpim_support",
        kernel_family="math_support",
        hardware_target="upmem_dpu",
        execution_mode="future_external_library",
        maturity_level=1,
    )

    def capabilities(self, context: TaskRouteContext) -> TaskRouteCapabilities:
        return _future_capabilities(self.identity, "TransPimLib/math-support integration is not implemented")

    def evaluate(self, task: ContractionTask, task_index: int, context: TaskRouteContext) -> TaskRouteDecision:
        reason = "transpim_support route unavailable: TransPimLib/math-support integration is not implemented"
        return _future_unavailable_decision(self.identity, task, task_index, context, reason)


class CpuFallbackTaskRoute:
    identity = TaskRouteIdentity(
        route_id="cpu_fallback",
        display_name="CPU fallback task route",
        route_family="cpu_fallback",
        kernel_family="einsum_contraction",
        hardware_target="cpu",
        execution_mode="in_process_python",
        maturity_level=1,
    )

    def capabilities(self, context: TaskRouteContext) -> TaskRouteCapabilities:
        return TaskRouteCapabilities(
            identity=self.identity,
            status="fallback_available",
            supported_task_structures=("dense", "sparse", "unknown"),
            can_estimate=True,
            can_prepare=False,
            can_execute=False,
            can_return_output=False,
            reason="analysis fallback maps to the existing cpu_tn_einsum_exact graph-level provider",
            metadata={"maps_to_provider": "cpu_tn_einsum_exact"},
        )

    def evaluate(self, task: ContractionTask, task_index: int, context: TaskRouteContext) -> TaskRouteDecision:
        estimate = TaskRouteEstimate(
            supported=True,
            estimated_flops=task.estimated_flops,
            estimated_bytes=task.estimated_bytes,
            estimated_peak_memory=_output_bytes(task),
            reason=None,
            metadata={"maps_to_provider": "cpu_tn_einsum_exact"},
        )
        return _decision(
            self.identity,
            task,
            task_index,
            context,
            "fallback",
            True,
            TaskRouteExecutionStatus(
                state="fallback_available",
                execution_implemented=False,
                can_prepare=False,
                can_execute=False,
                can_validate=False,
                reason="current cpu_tn_einsum_exact provider remains the execution-preserving fallback",
            ),
            "current cpu_tn_einsum_exact provider remains the execution-preserving fallback",
            estimate,
            metadata={"maps_to_provider": "cpu_tn_einsum_exact"},
        )


def default_task_routes() -> tuple[TaskRoute, ...]:
    return (
        SparseTaskRoute(),
        HeuristicBypassTaskRoute(),
        TransPimSupportTaskRoute(),
        CpuFallbackTaskRoute(),
    )


def _decision(
    identity: TaskRouteIdentity,
    task: ContractionTask,
    task_index: int,
    context: TaskRouteContext,
    status: TaskRouteDecisionStatus,
    is_selected: bool,
    execution_status: TaskRouteExecutionStatus,
    reason: str | None,
    estimate: TaskRouteEstimate,
    metadata: JsonDict | None = None,
) -> TaskRouteDecision:
    return TaskRouteDecision(
        schema_version=TASK_ROUTE_DECISION_SCHEMA_VERSION,
        router_id=STATIC_TASK_ROUTER_ID,
        case_id=context.case_id,
        task_id=task.id,
        task_index=task_index,
        input_tensor_ids=task.input_tensor_ids,
        output_tensor_id=task.output_tensor_id,
        route_id=identity.route_id,
        route_family=identity.route_family,
        kernel_family=identity.kernel_family,
        hardware_target=identity.hardware_target,
        execution_mode=identity.execution_mode,
        maturity_level=identity.maturity_level,
        status=status,
        is_selected=is_selected,
        execution_status=execution_status,
        reason=reason,
        estimate=estimate,
        metadata=metadata or {},
    )


def _future_capabilities(identity: TaskRouteIdentity, reason: str) -> TaskRouteCapabilities:
    return TaskRouteCapabilities(
        identity=identity,
        status="unavailable",
        can_estimate=False,
        can_prepare=False,
        can_execute=False,
        can_return_output=False,
        reason=reason,
    )


def _future_unavailable_decision(
    identity: TaskRouteIdentity,
    task: ContractionTask,
    task_index: int,
    context: TaskRouteContext,
    reason: str,
) -> TaskRouteDecision:
    estimate = TaskRouteEstimate(
        supported=False,
        estimated_flops=0,
        estimated_bytes=0,
        estimated_peak_memory=None,
        reason=reason,
    )
    return _decision(
        identity,
        task,
        task_index,
        context,
        "unavailable",
        False,
        TaskRouteExecutionStatus(
            state="future_backend",
            execution_implemented=False,
            can_prepare=False,
            can_execute=False,
            can_validate=False,
            reason=reason,
        ),
        reason,
        estimate,
    )


def _output_bytes(task: ContractionTask) -> int:
    total = 1
    for dim in task.output_shape:
        total *= dim
    return total * 16
