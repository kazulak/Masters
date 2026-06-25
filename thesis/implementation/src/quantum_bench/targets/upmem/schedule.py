from __future__ import annotations

import math
from dataclasses import dataclass, replace

from quantum_bench.core.records import ContractionTask, JsonDict, TaskGraph


UPMEM_DENSE_ESTIMATE_KEY = "upmem_dense_int8"
UPMEM_DENSE_MODEL = "dense_int8_single_dpu_feasibility"
REQUIRES_TILING_NOT_IMPLEMENTED = "requires_tiling_not_implemented"
UNSUPPORTED_DENSE_GEMM_SHAPE = "unsupported_dense_gemm_shape"


@dataclass(frozen=True)
class UpmemHardwareProfile:
    name: str
    wram_bytes: int
    dpu_count: int | None = None


@dataclass(frozen=True)
class UpmemDataFormat:
    name: str
    input_element_bytes: int
    output_element_bytes: int
    accumulator_element_bytes: int


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

    @property
    def fits_wram_without_tiling(self) -> bool:
        return self.wram_fit

    def as_task_estimate(self) -> JsonDict:
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
    def all_tasks_fit_without_tiling(self) -> bool:
        return all(task.wram_fit for task in self.tasks)

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
        )

    def metadata(self) -> JsonDict:
        return {
            "target": "upmem",
            "estimate_key": UPMEM_DENSE_ESTIMATE_KEY,
            "route_family": self.route_family,
            "model": UPMEM_DENSE_MODEL,
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
        }


UPMEM_PROFILE = UpmemHardwareProfile(name="UPMEM DPU", wram_bytes=64 * 1024)
DENSE_INT8_FORMAT = UpmemDataFormat(
    name="int8_inputs_int32_accumulator",
    input_element_bytes=1,
    output_element_bytes=4,
    accumulator_element_bytes=4,
)


def estimate_dense_task(
    task: ContractionTask,
    hardware: UpmemHardwareProfile = UPMEM_PROFILE,
    data_format: UpmemDataFormat = DENSE_INT8_FORMAT,
) -> UpmemTaskEstimate:
    if not _is_supported_dense_task(task):
        return UpmemTaskEstimate(
            task_id=task.id,
            gemm_m=task.gemm_m,
            gemm_k=task.gemm_k,
            gemm_n=task.gemm_n,
            supported=False,
            left_bytes=0,
            right_bytes=0,
            output_bytes=0,
            accumulator_bytes=0,
            working_set_bytes=0,
            host_to_dpu_bytes=0,
            dpu_to_host_bytes=0,
            mram_to_wram_bytes=0,
            wram_fit=False,
            requires_tiling=False,
            tiling_implemented=False,
            estimated_tile_count=0,
            estimated_parallel_tiles=0,
            reject_reason=UNSUPPORTED_DENSE_GEMM_SHAPE,
        )

    left_elements = task.gemm_m * task.gemm_k
    right_elements = task.gemm_k * task.gemm_n
    output_elements = task.gemm_m * task.gemm_n
    left_bytes = left_elements * data_format.input_element_bytes
    right_bytes = right_elements * data_format.input_element_bytes
    output_bytes = output_elements * data_format.output_element_bytes
    accumulator_bytes = output_elements * data_format.accumulator_element_bytes
    working_set_bytes = left_bytes + right_bytes + output_bytes + accumulator_bytes
    wram_fit = working_set_bytes <= hardware.wram_bytes
    estimated_tile_count = max(1, math.ceil(working_set_bytes / hardware.wram_bytes))
    estimated_parallel_tiles = (
        min(estimated_tile_count, hardware.dpu_count) if hardware.dpu_count is not None else estimated_tile_count
    )
    requires_tiling = not wram_fit
    reject_reason = REQUIRES_TILING_NOT_IMPLEMENTED if requires_tiling else None
    return UpmemTaskEstimate(
        task_id=task.id,
        gemm_m=task.gemm_m,
        gemm_k=task.gemm_k,
        gemm_n=task.gemm_n,
        supported=True,
        left_bytes=left_bytes,
        right_bytes=right_bytes,
        output_bytes=output_bytes,
        accumulator_bytes=accumulator_bytes,
        working_set_bytes=working_set_bytes,
        host_to_dpu_bytes=left_bytes + right_bytes,
        dpu_to_host_bytes=output_bytes,
        mram_to_wram_bytes=left_bytes + right_bytes + output_bytes,
        wram_fit=wram_fit,
        requires_tiling=requires_tiling,
        tiling_implemented=False,
        estimated_tile_count=estimated_tile_count,
        estimated_parallel_tiles=estimated_parallel_tiles,
        reject_reason=reject_reason,
    )


def estimate_dense_task_graph(
    graph: TaskGraph,
    hardware: UpmemHardwareProfile = UPMEM_PROFILE,
    data_format: UpmemDataFormat = DENSE_INT8_FORMAT,
) -> UpmemScheduleEstimate:
    tasks = tuple(estimate_dense_task(task, hardware, data_format) for task in graph.tasks)
    return _schedule_from_task_estimates(tasks, hardware, data_format)


def annotate_task_graph_with_upmem_estimates(
    graph: TaskGraph,
    hardware: UpmemHardwareProfile = UPMEM_PROFILE,
    data_format: UpmemDataFormat = DENSE_INT8_FORMAT,
) -> tuple[TaskGraph, UpmemScheduleEstimate]:
    estimates = tuple(estimate_dense_task(task, hardware, data_format) for task in graph.tasks)
    annotated_tasks = tuple(
        replace(
            task,
            target_estimates={
                **task.target_estimates,
                UPMEM_DENSE_ESTIMATE_KEY: estimate.as_task_estimate(),
            },
        )
        for task, estimate in zip(graph.tasks, estimates)
    )
    annotated_graph = replace(graph, tasks=annotated_tasks)
    return annotated_graph, _schedule_from_task_estimates(estimates, hardware, data_format)


def upmem_task_estimate_rows(graph: TaskGraph) -> list[JsonDict]:
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
        max_working_set_bytes=max((task.working_set_bytes for task in tasks), default=0),
        tasks_fit_without_tiling=sum(1 for task in tasks if task.wram_fit),
        tasks_requiring_tiling=sum(1 for task in tasks if task.requires_tiling),
        unsupported_tasks=sum(1 for task in tasks if not task.supported),
        total_estimated_tile_count=sum(task.estimated_tile_count for task in tasks),
        max_estimated_parallel_tiles=max((task.estimated_parallel_tiles for task in tasks), default=0),
    )


def _is_supported_dense_task(task: ContractionTask) -> bool:
    return task.structure == "dense" and task.gemm_m > 0 and task.gemm_k > 0 and task.gemm_n > 0
