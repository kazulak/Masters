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
from quantum_bench.targets.upmem import UPMEM_DENSE_ESTIMATE_KEY, UPMEM_DENSE_TILE_PLAN_ARTIFACT_KEY
from quantum_bench.targets.upmem.simplepim import SIMPLEPIM_PROBE_KEY, probe_simplepim, simplepim_probe_metadata


class TaskRoute(Protocol):
    identity: TaskRouteIdentity

    def capabilities(self, context: TaskRouteContext) -> TaskRouteCapabilities:
        ...

    def evaluate(self, task: ContractionTask, task_index: int, context: TaskRouteContext) -> TaskRouteDecision:
        ...


class DenseGemmTaskRoute:
    identity = TaskRouteIdentity(
        route_id="dense_gemm",
        display_name="Dense GEMM task route",
        route_family="dense_gemm",
        kernel_family="dense_gemm",
        hardware_target="upmem_dpu",
        execution_mode="future_simplepim_or_native",
        maturity_level=2,
    )

    def capabilities(self, context: TaskRouteContext) -> TaskRouteCapabilities:
        return TaskRouteCapabilities(
            identity=self.identity,
            status="estimate_only",
            supported_task_structures=("dense",),
            can_estimate=True,
            can_prepare=False,
            can_execute=False,
            can_return_output=False,
            reason="dense GEMM task route is estimate-only; SimplePIM/native execution is not implemented",
            metadata={
                "target_estimate_key": UPMEM_DENSE_ESTIMATE_KEY,
                **_fixed_point_conversion_metadata(),
                **_simplepim_metadata(context),
            },
        )

    def evaluate(self, task: ContractionTask, task_index: int, context: TaskRouteContext) -> TaskRouteDecision:
        target_estimate = task.target_estimates.get(UPMEM_DENSE_ESTIMATE_KEY)
        if target_estimate is None:
            estimate = TaskRouteEstimate(
                supported=False,
                estimated_flops=task.estimated_flops,
                estimated_bytes=task.estimated_bytes,
                estimated_peak_memory=None,
                reason="missing_target_estimate",
                metadata={
                    "target_estimate_key": UPMEM_DENSE_ESTIMATE_KEY,
                    **_fixed_point_conversion_metadata(),
                    **_tile_plan_metadata(None, context),
                    **_simplepim_metadata(context),
                },
            )
            return _decision(
                self.identity,
                task,
                task_index,
                context,
                "skipped",
                False,
                _estimate_only_status("missing UPMEM dense target estimate"),
                "missing_target_estimate",
                estimate,
            )

        reason = _string_or_none(target_estimate.get("reject_reason"))
        if not bool(target_estimate.get("supported", False)):
            status: TaskRouteDecisionStatus = "rejected"
            reason = reason or "unsupported_dense_gemm_shape"
        elif bool(target_estimate.get("requires_tiling", False)):
            status = "rejected"
            reason = reason or "requires_tiling_not_implemented"
        else:
            status = "skipped"
            reason = "dense_gemm execution not implemented; estimate only"

        estimate = TaskRouteEstimate(
            supported=bool(target_estimate.get("supported", False)),
            estimated_flops=task.estimated_flops,
            estimated_bytes=_nonnegative_int(target_estimate.get("host_to_dpu_bytes"))
            + _nonnegative_int(target_estimate.get("dpu_to_host_bytes"))
            + _nonnegative_int(target_estimate.get("mram_to_wram_bytes")),
            estimated_peak_memory=_optional_nonnegative_int(target_estimate.get("max_working_set_bytes")),
            wram_fit=_optional_bool(target_estimate.get("wram_fit")),
            requires_tiling=_optional_bool(target_estimate.get("requires_tiling")),
            tiling_implemented=_optional_bool(target_estimate.get("tiling_implemented")),
            host_to_dpu_bytes=_optional_nonnegative_int(target_estimate.get("host_to_dpu_bytes")),
            dpu_to_host_bytes=_optional_nonnegative_int(target_estimate.get("dpu_to_host_bytes")),
            mram_to_wram_bytes=_optional_nonnegative_int(target_estimate.get("mram_to_wram_bytes")),
            estimated_tile_count=_optional_nonnegative_int(target_estimate.get("estimated_tile_count")),
            estimated_parallel_tiles=_optional_nonnegative_int(target_estimate.get("estimated_parallel_tiles")),
            reason=reason,
            metadata={
                "target_estimate_key": UPMEM_DENSE_ESTIMATE_KEY,
                "target_estimate_model": target_estimate.get("model"),
                "target": target_estimate.get("target"),
                **_fixed_point_conversion_metadata(),
                **_tile_plan_metadata(target_estimate, context),
                **_simplepim_metadata(context),
            },
        )
        return _decision(
            self.identity,
            task,
            task_index,
            context,
            status,
            False,
            _estimate_only_status("dense GEMM execution is not implemented"),
            reason,
            estimate,
        )


class SparseTaskRoute:
    identity = TaskRouteIdentity(
        route_id="sparse",
        display_name="Sparse task route",
        route_family="sparse",
        kernel_family="sparse_spmm",
        hardware_target="upmem_dpu",
        execution_mode="future_simplepim_or_native",
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
        DenseGemmTaskRoute(),
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


def _estimate_only_status(reason: str) -> TaskRouteExecutionStatus:
    return TaskRouteExecutionStatus(
        state="estimate_only",
        execution_implemented=False,
        can_prepare=False,
        can_execute=False,
        can_validate=False,
        reason=reason,
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


def _nonnegative_int(value: object) -> int:
    converted = _optional_nonnegative_int(value)
    return 0 if converted is None else converted


def _optional_nonnegative_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return None


def _optional_bool(value: object) -> bool | None:
    return None if value is None else bool(value)


def _string_or_none(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


def _fixed_point_conversion_metadata() -> JsonDict:
    return {
        "conversion_required": True,
        "intended_route_dtype": "int8",
        "conversion_format": "fixed_point_symmetric",
        "complex_policy": "split_real_imag_last_axis",
        "conversion_artifact": None,
    }


def _tile_plan_metadata(target_estimate: JsonDict | None, context: TaskRouteContext) -> JsonDict:
    if target_estimate is None:
        return {
            "tile_plan_available": False,
            "tile_plan_artifact": context.target_artifacts.get(UPMEM_DENSE_TILE_PLAN_ARTIFACT_KEY),
            "tile_count": None,
            "working_set_bytes": None,
            "double_buffer_possible": None,
            "requires_host_aggregation": None,
        }
    return {
        "tile_plan_available": bool(target_estimate.get("tile_plan_available", False)),
        "tile_plan_artifact": context.target_artifacts.get(UPMEM_DENSE_TILE_PLAN_ARTIFACT_KEY),
        "tile_count": target_estimate.get("total_tile_count", target_estimate.get("estimated_tile_count")),
        "working_set_bytes": target_estimate.get("max_working_set_bytes"),
        "double_buffer_possible": target_estimate.get("double_buffer_possible"),
        "requires_host_aggregation": target_estimate.get("requires_host_aggregation"),
    }


def _simplepim_metadata(context: TaskRouteContext) -> JsonDict:
    probe = context.backend_probes.get(SIMPLEPIM_PROBE_KEY)
    if probe is None:
        probe = probe_simplepim().to_json_dict()
    return simplepim_probe_metadata(probe)
