from __future__ import annotations

from dataclasses import dataclass, replace

from quantum_bench.core.target_estimates import TargetEstimateSet, TargetMetricSpec
from quantum_bench.core.records import ContractionTask, JsonDict, TaskGraph
from quantum_bench.targets.upmem.tile_plan import (
    DENSE_INT8_FORMAT,
    REQUIRES_TILING_NOT_IMPLEMENTED as REQUIRES_TILING_NOT_IMPLEMENTED,
    UNSUPPORTED_DENSE_GEMM_SHAPE as UNSUPPORTED_DENSE_GEMM_SHAPE,
    UPMEM_DENSE_ESTIMATE_KEY,
    UPMEM_DENSE_MODEL,
    UPMEM_DENSE_TILE_PLAN_ARTIFACT_KEY as UPMEM_DENSE_TILE_PLAN_ARTIFACT_KEY,
    UPMEM_DENSE_TILE_PLAN_MODEL,
    UPMEM_PROFILE,
    UpmemDataFormat,
    UpmemDenseTilePlan,
    UpmemHardwareProfile,
    plan_dense_task,
)


@dataclass(frozen=True)
class UpmemTaskEstimate:
    task_id: str
    gemm_m: int
    gemm_k: int
    gemm_n: int
    supported: bool
    left_bytes: int
    right_bytes: int
    output_bytes: int
    accumulator_bytes: int
    working_set_bytes: int
    host_to_dpu_bytes: int
    dpu_to_host_bytes: int
    mram_to_wram_bytes: int
    wram_fit: bool
    requires_tiling: bool
    tiling_implemented: bool
    estimated_tile_count: int
    estimated_parallel_tiles: int
    reject_reason: str | None
    tile_plan: UpmemDenseTilePlan

    @property
    def fits_wram_without_tiling(self) -> bool:
        return self.wram_fit and not self.requires_tiling

    def as_task_estimate(self) -> JsonDict:
        tile_plan = self.tile_plan.as_summary()
        return {
            "target": "upmem",
            "estimate_key": UPMEM_DENSE_ESTIMATE_KEY,
            "model": UPMEM_DENSE_MODEL,
            "supported": self.supported,
            "wram_fit": self.wram_fit,
            "requires_tiling": self.requires_tiling,
            "tiling_implemented": self.tiling_implemented,
            "gemm_m": self.gemm_m,
            "gemm_k": self.gemm_k,
            "gemm_n": self.gemm_n,
            "max_working_set_bytes": self.working_set_bytes,
            "estimated_tile_count": self.estimated_tile_count,
            "estimated_parallel_tiles": self.estimated_parallel_tiles,
            "host_to_dpu_bytes": self.host_to_dpu_bytes,
            "dpu_to_host_bytes": self.dpu_to_host_bytes,
            "mram_to_wram_bytes": self.mram_to_wram_bytes,
            "reject_reason": self.reject_reason,
            "tile_plan_model": UPMEM_DENSE_TILE_PLAN_MODEL,
            "tile_plan_available": self.tile_plan.supported,
            "element_dtype": tile_plan["element_dtype"],
            "element_bytes": tile_plan["element_bytes"],
            "tile_m": tile_plan["tile_m"],
            "tile_k": tile_plan["tile_k"],
            "tile_n": tile_plan["tile_n"],
            "tile_count_m": tile_plan["tile_count_m"],
            "tile_count_k": tile_plan["tile_count_k"],
            "tile_count_n": tile_plan["tile_count_n"],
            "total_tile_count": tile_plan["total_tile_count"],
            "double_buffer_possible": tile_plan["double_buffer_possible"],
            "requires_host_aggregation": tile_plan["requires_host_aggregation"],
            "fits_wram": tile_plan["fits_wram"],
            "tile_plan": tile_plan,
        }

    def as_artifact_row(self, task: ContractionTask) -> JsonDict:
        return {
            "task_id": task.id,
            "input_tensor_ids": task.input_tensor_ids,
            "output_tensor_id": task.output_tensor_id,
            **self.as_task_estimate(),
        }


@dataclass(frozen=True)
class UpmemScheduleEstimate:
    route_family: str
    hardware: UpmemHardwareProfile
    data_format: UpmemDataFormat
    tasks: tuple[UpmemTaskEstimate, ...]
    total_host_to_dpu_bytes: int
    total_dpu_to_host_bytes: int
    total_mram_to_wram_bytes: int
    max_working_set_bytes: int
    tasks_fit_without_tiling: int
    tasks_requiring_tiling: int
    unsupported_tasks: int
    total_estimated_tile_count: int
    max_estimated_parallel_tiles: int

    @property
    def target_id(self) -> str:
        return "upmem_dense_gemm"

    @property
    def model_id(self) -> str:
        return UPMEM_DENSE_MODEL

    def compatibility_metadata(self) -> JsonDict:
        return {
            "target_id": self.target_id,
            "model_id": self.model_id,
            "route_family": self.route_family,
            "hardware": {
                "name": self.hardware.name,
                "wram_bytes": self.hardware.wram_bytes,
                "dpu_count": self.hardware.dpu_count,
            },
            "data_format": {
                "name": self.data_format.name,
                "input_element_bytes": self.data_format.input_element_bytes,
                "output_element_bytes": self.data_format.output_element_bytes,
                "accumulator_element_bytes": self.data_format.accumulator_element_bytes,
            },
            "tasks": [
                {
                    "task_id": task.task_id,
                    "gemm_m": task.gemm_m,
                    "gemm_k": task.gemm_k,
                    "gemm_n": task.gemm_n,
                }
                for task in self.tasks
            ],
        }

    @property
    def all_tasks_fit_without_tiling(self) -> bool:
        return all(task.fits_wram_without_tiling for task in self.tasks)

    def first_reject_reason(self) -> str | None:
        for task in self.tasks:
            if task.reject_reason:
                return task.reject_reason
        return None

    def notes(self) -> tuple[str, ...]:
        return (
            f"UPMEM target layer: {self.hardware.name}, {self.hardware.wram_bytes} B WRAM",
            f"dense schedule estimate: {self.tasks_fit_without_tiling}/{len(self.tasks)} tasks fit without tiling",
            f"estimated H2D={self.total_host_to_dpu_bytes} B, D2H={self.total_dpu_to_host_bytes} B",
            "WRAM model is conservative: output tile storage and accumulator workspace are counted separately",
        )

    def metadata(self) -> JsonDict:
        return {
            "target": "upmem",
            "estimate_key": UPMEM_DENSE_ESTIMATE_KEY,
            "route_family": self.route_family,
            "model": self.model_id,
            "tile_plan_model": UPMEM_DENSE_TILE_PLAN_MODEL,
            "tiling_implemented": False,
            "hardware": {
                "name": self.hardware.name,
                "wram_bytes": self.hardware.wram_bytes,
                "dpu_count": self.hardware.dpu_count,
            },
            "data_format": {
                "name": self.data_format.name,
                "input_element_bytes": self.data_format.input_element_bytes,
                "output_element_bytes": self.data_format.output_element_bytes,
                "accumulator_element_bytes": self.data_format.accumulator_element_bytes,
            },
            "task_count": len(self.tasks),
            "tasks_fit_without_tiling": self.tasks_fit_without_tiling,
            "tasks_requiring_tiling": self.tasks_requiring_tiling,
            "unsupported_tasks": self.unsupported_tasks,
            "total_estimated_tile_count": self.total_estimated_tile_count,
            "max_estimated_parallel_tiles": self.max_estimated_parallel_tiles,
            "total_host_to_dpu_bytes": self.total_host_to_dpu_bytes,
            "total_dpu_to_host_bytes": self.total_dpu_to_host_bytes,
            "total_mram_to_wram_bytes": self.total_mram_to_wram_bytes,
            "max_working_set_bytes": self.max_working_set_bytes,
            "first_reject_reason": self.first_reject_reason(),
            "memory_model_note": (
                "conservative: output tile storage and accumulator workspace are "
                "counted as separate WRAM reservations"
            ),
        }


def estimate_dense_task(
    task: ContractionTask,
    hardware: UpmemHardwareProfile = UPMEM_PROFILE,
    data_format: UpmemDataFormat = DENSE_INT8_FORMAT,
) -> UpmemTaskEstimate:
    tile_plan = plan_dense_task(task, hardware, data_format)
    estimated_tile_count = tile_plan.tile_counts.total_tile_count
    host_to_dpu_bytes = estimated_tile_count * (
        tile_plan.input_a_tile_bytes + tile_plan.input_b_tile_bytes
    )
    dpu_to_host_bytes = estimated_tile_count * tile_plan.output_tile_bytes
    mram_to_wram_bytes = estimated_tile_count * (
        tile_plan.input_a_tile_bytes
        + tile_plan.input_b_tile_bytes
        + tile_plan.output_tile_bytes
    )
    return UpmemTaskEstimate(
        task_id=task.id,
        gemm_m=task.gemm_m,
        gemm_k=task.gemm_k,
        gemm_n=task.gemm_n,
        supported=tile_plan.supported,
        left_bytes=tile_plan.input_a_tile_bytes,
        right_bytes=tile_plan.input_b_tile_bytes,
        output_bytes=tile_plan.output_tile_bytes,
        accumulator_bytes=tile_plan.accumulator_tile_bytes,
        working_set_bytes=tile_plan.working_set_bytes,
        host_to_dpu_bytes=host_to_dpu_bytes,
        dpu_to_host_bytes=dpu_to_host_bytes,
        mram_to_wram_bytes=mram_to_wram_bytes,
        wram_fit=tile_plan.fits_wram,
        requires_tiling=tile_plan.requires_tiling,
        tiling_implemented=False,
        estimated_tile_count=estimated_tile_count,
        estimated_parallel_tiles=tile_plan.estimated_parallel_tiles,
        reject_reason=tile_plan.reject_reason,
        tile_plan=tile_plan,
    )


def estimate_dense_task_graph(
    graph: TaskGraph,
    hardware: UpmemHardwareProfile = UPMEM_PROFILE,
    data_format: UpmemDataFormat = DENSE_INT8_FORMAT,
) -> UpmemScheduleEstimate:
    tasks = tuple(
        estimate_dense_task(task, hardware, data_format) for task in graph.tasks
    )
    return _schedule_from_task_estimates(tasks, hardware, data_format)


def estimate_dense_task_graph_sidecar(
    graph: TaskGraph,
    hardware: UpmemHardwareProfile = UPMEM_PROFILE,
    data_format: UpmemDataFormat = DENSE_INT8_FORMAT,
) -> tuple[TargetEstimateSet, UpmemScheduleEstimate]:
    """Estimate a graph without mutating its scientific tasks or summary."""

    estimates = tuple(
        estimate_dense_task(task, hardware, data_format) for task in graph.tasks
    )
    rows = [
        estimate.as_artifact_row(task) for task, estimate in zip(graph.tasks, estimates)
    ]
    schedule = _schedule_from_task_estimates(estimates, hardware, data_format)
    sidecar = TargetEstimateSet.from_rows(
        scientific_plan_hash=graph.contraction_plan_hash,
        target_id=schedule.target_id,
        model_id=schedule.model_id,
        rows=rows,
        metric_specs=_UPMEM_METRIC_SPECS,
        metadata={"schedule_compatibility": schedule.compatibility_metadata()},
    )
    return sidecar, schedule


def upmem_target_path_summary(
    estimates: TargetEstimateSet,
    schedule: UpmemScheduleEstimate,
) -> JsonDict:
    """Return explicit target decoration separate from scientific ``PathSummary``."""

    _validate_sidecar_schedule(estimates, schedule)
    return {
        "scientific_plan_hash": estimates.scientific_plan_hash,
        "target_id": estimates.target_id,
        "model_id": estimates.model_id,
        "metric_provenance": [spec.to_json_dict() for spec in estimates.metric_specs],
        **schedule.metadata(),
    }


def annotate_task_graph_with_upmem_estimates(
    graph: TaskGraph,
    hardware: UpmemHardwareProfile = UPMEM_PROFILE,
    data_format: UpmemDataFormat = DENSE_INT8_FORMAT,
    estimates: TargetEstimateSet | None = None,
    schedule: UpmemScheduleEstimate | None = None,
) -> tuple[TaskGraph, UpmemScheduleEstimate]:
    if estimates is None and schedule is not None:
        raise ValueError(
            "A precomputed schedule requires its matching target estimate sidecar"
        )
    if estimates is None:
        sidecar, resolved_schedule = estimate_dense_task_graph_sidecar(
            graph, hardware, data_format
        )
    else:
        if schedule is None:
            raise ValueError(
                "A precomputed target estimate sidecar requires its matching schedule"
            )
        sidecar = estimates
        resolved_schedule = schedule
        _validate_sidecar_schedule(sidecar, resolved_schedule)
    sidecar.validate_graph(
        graph,
        expected_target_id="upmem_dense_gemm",
        expected_model_id=UPMEM_DENSE_MODEL,
    )
    annotated_tasks = tuple(
        replace(
            task,
            target_estimates={
                **task.target_estimates,
                UPMEM_DENSE_ESTIMATE_KEY: sidecar.values_for(task.id) or {},
            },
        )
        for task in graph.tasks
    )
    annotated_graph = replace(graph, tasks=annotated_tasks)
    return annotated_graph, resolved_schedule


def upmem_task_estimate_rows(
    graph: TaskGraph,
    estimates: TargetEstimateSet | None = None,
) -> list[JsonDict]:
    """Serialize sidecar rows first; fall back to historical inline fields."""

    if estimates is not None:
        estimates.validate_graph(
            graph,
            expected_target_id="upmem_dense_gemm",
            expected_model_id=UPMEM_DENSE_MODEL,
        )
        return estimates.jsonl_rows()
    return [
        {
            "task_id": task.id,
            "input_tensor_ids": task.input_tensor_ids,
            "output_tensor_id": task.output_tensor_id,
            **task.target_estimates[UPMEM_DENSE_ESTIMATE_KEY],
        }
        for task in graph.tasks
        if UPMEM_DENSE_ESTIMATE_KEY in task.target_estimates
    ]


_UPMEM_METRIC_SPECS = (
    TargetMetricSpec("target", "identifier", "analytic_model", "task_estimate"),
    TargetMetricSpec("estimate_key", "identifier", "analytic_model", "task_estimate"),
    TargetMetricSpec("model", "identifier", "analytic_model", "task_estimate"),
    TargetMetricSpec("supported", "boolean", "analytic_model", "task"),
    TargetMetricSpec("wram_fit", "boolean", "analytic_model", "task"),
    TargetMetricSpec("requires_tiling", "boolean", "analytic_model", "task"),
    TargetMetricSpec("tiling_implemented", "boolean", "analytic_model", "task"),
    TargetMetricSpec("tile_plan_available", "boolean", "analytic_model", "task"),
    TargetMetricSpec("double_buffer_possible", "boolean", "analytic_model", "task"),
    TargetMetricSpec("requires_host_aggregation", "boolean", "analytic_model", "task"),
    TargetMetricSpec("fits_wram", "boolean", "analytic_model", "task"),
    TargetMetricSpec("gemm_m", "elements", "scientific_plan", "task"),
    TargetMetricSpec("gemm_k", "elements", "scientific_plan", "task"),
    TargetMetricSpec("gemm_n", "elements", "scientific_plan", "task"),
    TargetMetricSpec("max_working_set_bytes", "bytes", "analytic_model", "task"),
    TargetMetricSpec("estimated_tile_count", "tiles", "analytic_model", "task"),
    TargetMetricSpec("estimated_parallel_tiles", "tiles", "analytic_model", "task"),
    TargetMetricSpec(
        "host_to_dpu_bytes",
        "bytes",
        "analytic_model",
        "modeled_application_visible_transfer_not_measured_bus_traffic",
    ),
    TargetMetricSpec(
        "dpu_to_host_bytes",
        "bytes",
        "analytic_model",
        "modeled_application_visible_transfer_not_measured_bus_traffic",
    ),
    TargetMetricSpec(
        "mram_to_wram_bytes",
        "bytes",
        "analytic_model",
        "modeled_mram_to_wram_payload_not_measured_traffic",
    ),
    TargetMetricSpec("reject_reason", "reason_code", "analytic_model", "task"),
    TargetMetricSpec("tile_plan_model", "identifier", "analytic_model", "task"),
    TargetMetricSpec("element_dtype", "identifier", "analytic_model", "task"),
    TargetMetricSpec("element_bytes", "bytes_per_element", "analytic_model", "task"),
    TargetMetricSpec("tile_m", "elements", "analytic_model", "task"),
    TargetMetricSpec("tile_k", "elements", "analytic_model", "task"),
    TargetMetricSpec("tile_n", "elements", "analytic_model", "task"),
    TargetMetricSpec("tile_count_m", "tiles", "analytic_model", "task"),
    TargetMetricSpec("tile_count_k", "tiles", "analytic_model", "task"),
    TargetMetricSpec("tile_count_n", "tiles", "analytic_model", "task"),
    TargetMetricSpec("total_tile_count", "tiles", "analytic_model", "task"),
    TargetMetricSpec("tile_plan", "json_record", "analytic_model", "task"),
)


def _validate_sidecar_schedule(
    sidecar: TargetEstimateSet,
    schedule: UpmemScheduleEstimate,
) -> None:
    if sidecar.target_id != schedule.target_id or sidecar.model_id != schedule.model_id:
        if sidecar.target_id != schedule.target_id:
            raise ValueError(
                "Target estimate sidecar target ID and schedule target ID disagree"
            )
        raise ValueError(
            "Target estimate sidecar model ID and schedule model ID disagree"
        )
    expected = sidecar.metadata_dict().get("schedule_compatibility")
    if expected is None:
        raise ValueError(
            "Target estimate sidecar is missing schedule compatibility metadata"
        )
    if expected != schedule.compatibility_metadata():
        raise ValueError(
            "Target estimate sidecar and schedule profile/task metadata disagree"
        )


def _schedule_from_task_estimates(
    tasks: tuple[UpmemTaskEstimate, ...],
    hardware: UpmemHardwareProfile,
    data_format: UpmemDataFormat,
) -> UpmemScheduleEstimate:
    return UpmemScheduleEstimate(
        route_family="dense_gemm",
        hardware=hardware,
        data_format=data_format,
        tasks=tasks,
        total_host_to_dpu_bytes=sum(task.host_to_dpu_bytes for task in tasks),
        total_dpu_to_host_bytes=sum(task.dpu_to_host_bytes for task in tasks),
        total_mram_to_wram_bytes=sum(task.mram_to_wram_bytes for task in tasks),
        max_working_set_bytes=max(
            (task.working_set_bytes for task in tasks), default=0
        ),
        tasks_fit_without_tiling=sum(
            1 for task in tasks if task.fits_wram_without_tiling
        ),
        tasks_requiring_tiling=sum(1 for task in tasks if task.requires_tiling),
        unsupported_tasks=sum(1 for task in tasks if not task.supported),
        total_estimated_tile_count=sum(task.estimated_tile_count for task in tasks),
        max_estimated_parallel_tiles=max(
            (task.estimated_parallel_tiles for task in tasks), default=0
        ),
    )
