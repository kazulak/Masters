from __future__ import annotations

import math
from dataclasses import dataclass

from quantum_bench.core.records import ContractionTask, JsonDict, TaskGraph


UPMEM_DENSE_ESTIMATE_KEY = "upmem_dense_int8"
UPMEM_DENSE_MODEL = "dense_int8_single_dpu_feasibility"
UPMEM_DENSE_TILE_PLAN_ARTIFACT_KEY = "upmem_dense_tile_plan"
UPMEM_DENSE_TILE_PLAN_MODEL = "dense_int8_wram_tile_plan_v1"
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
class UpmemTileShape:
    tile_m: int
    tile_k: int
    tile_n: int


@dataclass(frozen=True)
class UpmemTileCounts:
    tile_count_m: int
    tile_count_k: int
    tile_count_n: int
    total_tile_count: int


@dataclass(frozen=True)
class UpmemDenseTilePlan:
    target: str
    estimate_key: str
    model: str
    task_id: str
    gemm_m: int
    gemm_k: int
    gemm_n: int
    supported: bool
    element_dtype: str
    element_bytes: int
    input_element_bytes: int
    output_element_bytes: int
    accumulator_element_bytes: int
    tile_shape: UpmemTileShape
    tile_counts: UpmemTileCounts
    input_a_tile_bytes: int
    input_b_tile_bytes: int
    output_tile_bytes: int
    accumulator_tile_bytes: int
    working_set_bytes: int
    full_task_working_set_bytes: int
    wram_capacity_bytes: int
    fits_wram: bool
    requires_tiling: bool
    tiling_implemented: bool
    double_buffer_possible: bool
    requires_host_aggregation: bool
    estimated_parallel_tiles: int
    reject_reason: str | None

    def as_summary(self) -> JsonDict:
        return {
            "target": self.target,
            "estimate_key": self.estimate_key,
            "model": self.model,
            "task_id": self.task_id,
            "gemm_m": self.gemm_m,
            "gemm_k": self.gemm_k,
            "gemm_n": self.gemm_n,
            "supported": self.supported,
            "element_dtype": self.element_dtype,
            "element_bytes": self.element_bytes,
            "input_element_bytes": self.input_element_bytes,
            "output_element_bytes": self.output_element_bytes,
            "accumulator_element_bytes": self.accumulator_element_bytes,
            "tile_m": self.tile_shape.tile_m,
            "tile_k": self.tile_shape.tile_k,
            "tile_n": self.tile_shape.tile_n,
            "tile_count_m": self.tile_counts.tile_count_m,
            "tile_count_k": self.tile_counts.tile_count_k,
            "tile_count_n": self.tile_counts.tile_count_n,
            "total_tile_count": self.tile_counts.total_tile_count,
            "estimated_parallel_tiles": self.estimated_parallel_tiles,
            "input_a_tile_bytes": self.input_a_tile_bytes,
            "input_b_tile_bytes": self.input_b_tile_bytes,
            "output_tile_bytes": self.output_tile_bytes,
            "accumulator_tile_bytes": self.accumulator_tile_bytes,
            "working_set_bytes": self.working_set_bytes,
            "full_task_working_set_bytes": self.full_task_working_set_bytes,
            "wram_capacity_bytes": self.wram_capacity_bytes,
            "fits_wram": self.fits_wram,
            "requires_tiling": self.requires_tiling,
            "tiling_implemented": self.tiling_implemented,
            "double_buffer_possible": self.double_buffer_possible,
            "requires_host_aggregation": self.requires_host_aggregation,
            "reject_reason": self.reject_reason,
            "memory_model_note": (
                "conservative: output tile storage and accumulator workspace are "
                "counted as separate WRAM reservations"
            ),
        }

    def as_artifact_row(self, task: ContractionTask) -> JsonDict:
        return {
            "task_id": task.id,
            "input_tensor_ids": task.input_tensor_ids,
            "output_tensor_id": task.output_tensor_id,
            **self.as_summary(),
        }


UPMEM_PROFILE = UpmemHardwareProfile(name="UPMEM DPU", wram_bytes=64 * 1024)
DENSE_INT8_FORMAT = UpmemDataFormat(
    name="int8_inputs_int32_accumulator",
    input_element_bytes=1,
    output_element_bytes=4,
    accumulator_element_bytes=4,
)


def plan_dense_task(
    task: ContractionTask,
    hardware: UpmemHardwareProfile = UPMEM_PROFILE,
    data_format: UpmemDataFormat = DENSE_INT8_FORMAT,
) -> UpmemDenseTilePlan:
    if not _is_supported_dense_task(task):
        return _unsupported_plan(task, hardware, data_format)

    full_shape = UpmemTileShape(task.gemm_m, task.gemm_k, task.gemm_n)
    full_working_set = _working_set_bytes(full_shape, data_format)
    if full_working_set <= hardware.wram_bytes:
        return _plan_from_shape(task, hardware, data_format, full_shape, False)

    tile_shape = _choose_tile_shape(task, hardware, data_format)
    if tile_shape is None:
        return _unsupported_plan(task, hardware, data_format, REQUIRES_TILING_NOT_IMPLEMENTED)
    return _plan_from_shape(task, hardware, data_format, tile_shape, True)


def plan_dense_task_graph(
    graph: TaskGraph,
    hardware: UpmemHardwareProfile = UPMEM_PROFILE,
    data_format: UpmemDataFormat = DENSE_INT8_FORMAT,
) -> tuple[UpmemDenseTilePlan, ...]:
    return tuple(plan_dense_task(task, hardware, data_format) for task in graph.tasks)


def upmem_dense_tile_plan_rows(graph: TaskGraph) -> list[JsonDict]:
    return [plan_dense_task(task).as_artifact_row(task) for task in graph.tasks]


def _plan_from_shape(
    task: ContractionTask,
    hardware: UpmemHardwareProfile,
    data_format: UpmemDataFormat,
    shape: UpmemTileShape,
    requires_tiling: bool,
) -> UpmemDenseTilePlan:
    counts = UpmemTileCounts(
        tile_count_m=math.ceil(task.gemm_m / shape.tile_m),
        tile_count_k=math.ceil(task.gemm_k / shape.tile_k),
        tile_count_n=math.ceil(task.gemm_n / shape.tile_n),
        total_tile_count=(
            math.ceil(task.gemm_m / shape.tile_m)
            * math.ceil(task.gemm_k / shape.tile_k)
            * math.ceil(task.gemm_n / shape.tile_n)
        ),
    )
    input_a_tile_bytes = shape.tile_m * shape.tile_k * data_format.input_element_bytes
    input_b_tile_bytes = shape.tile_k * shape.tile_n * data_format.input_element_bytes
    output_tile_bytes = shape.tile_m * shape.tile_n * data_format.output_element_bytes
    accumulator_tile_bytes = shape.tile_m * shape.tile_n * data_format.accumulator_element_bytes
    working_set_bytes = input_a_tile_bytes + input_b_tile_bytes + output_tile_bytes + accumulator_tile_bytes
    requires_host_aggregation = counts.tile_count_k > 1
    return UpmemDenseTilePlan(
        target="upmem",
        estimate_key=UPMEM_DENSE_ESTIMATE_KEY,
        model=UPMEM_DENSE_TILE_PLAN_MODEL,
        task_id=task.id,
        gemm_m=task.gemm_m,
        gemm_k=task.gemm_k,
        gemm_n=task.gemm_n,
        supported=True,
        element_dtype=_input_dtype_name(data_format),
        element_bytes=data_format.input_element_bytes,
        input_element_bytes=data_format.input_element_bytes,
        output_element_bytes=data_format.output_element_bytes,
        accumulator_element_bytes=data_format.accumulator_element_bytes,
        tile_shape=shape,
        tile_counts=counts,
        input_a_tile_bytes=input_a_tile_bytes,
        input_b_tile_bytes=input_b_tile_bytes,
        output_tile_bytes=output_tile_bytes,
        accumulator_tile_bytes=accumulator_tile_bytes,
        working_set_bytes=working_set_bytes,
        full_task_working_set_bytes=_working_set_bytes(
            UpmemTileShape(task.gemm_m, task.gemm_k, task.gemm_n),
            data_format,
        ),
        wram_capacity_bytes=hardware.wram_bytes,
        fits_wram=working_set_bytes <= hardware.wram_bytes,
        requires_tiling=requires_tiling,
        tiling_implemented=False,
        double_buffer_possible=(2 * working_set_bytes) <= hardware.wram_bytes,
        requires_host_aggregation=requires_host_aggregation,
        estimated_parallel_tiles=(
            min(counts.total_tile_count, hardware.dpu_count)
            if hardware.dpu_count is not None
            else counts.total_tile_count
        ),
        reject_reason=REQUIRES_TILING_NOT_IMPLEMENTED if requires_tiling else None,
    )


def _unsupported_plan(
    task: ContractionTask,
    hardware: UpmemHardwareProfile,
    data_format: UpmemDataFormat,
    reason: str = UNSUPPORTED_DENSE_GEMM_SHAPE,
) -> UpmemDenseTilePlan:
    shape = UpmemTileShape(0, 0, 0)
    counts = UpmemTileCounts(0, 0, 0, 0)
    return UpmemDenseTilePlan(
        target="upmem",
        estimate_key=UPMEM_DENSE_ESTIMATE_KEY,
        model=UPMEM_DENSE_TILE_PLAN_MODEL,
        task_id=task.id,
        gemm_m=task.gemm_m,
        gemm_k=task.gemm_k,
        gemm_n=task.gemm_n,
        supported=False,
        element_dtype=_input_dtype_name(data_format),
        element_bytes=data_format.input_element_bytes,
        input_element_bytes=data_format.input_element_bytes,
        output_element_bytes=data_format.output_element_bytes,
        accumulator_element_bytes=data_format.accumulator_element_bytes,
        tile_shape=shape,
        tile_counts=counts,
        input_a_tile_bytes=0,
        input_b_tile_bytes=0,
        output_tile_bytes=0,
        accumulator_tile_bytes=0,
        working_set_bytes=0,
        full_task_working_set_bytes=0,
        wram_capacity_bytes=hardware.wram_bytes,
        fits_wram=False,
        requires_tiling=False,
        tiling_implemented=False,
        double_buffer_possible=False,
        requires_host_aggregation=False,
        estimated_parallel_tiles=0,
        reject_reason=reason,
    )


def _choose_tile_shape(
    task: ContractionTask,
    hardware: UpmemHardwareProfile,
    data_format: UpmemDataFormat,
) -> UpmemTileShape | None:
    for tile_k in _dimension_candidates(task.gemm_k):
        best: UpmemTileShape | None = None
        best_score: tuple[int, int, int] | None = None
        for tile_m in _dimension_candidates(task.gemm_m):
            for tile_n in _dimension_candidates(task.gemm_n):
                candidate = UpmemTileShape(tile_m, tile_k, tile_n)
                if _working_set_bytes(candidate, data_format) > hardware.wram_bytes:
                    continue
                score = (tile_m * tile_n, tile_m, tile_n)
                if best_score is None or score > best_score:
                    best = candidate
                    best_score = score
        if best is not None:
            return best
    return None


def _dimension_candidates(dim: int) -> tuple[int, ...]:
    if dim <= 0:
        return ()
    values = {dim, 1}
    power = 1
    while power < dim:
        values.add(power)
        power *= 2
    if power == dim:
        values.add(power)
    return tuple(sorted(values, reverse=True))


def _working_set_bytes(shape: UpmemTileShape, data_format: UpmemDataFormat) -> int:
    input_a_tile_bytes = shape.tile_m * shape.tile_k * data_format.input_element_bytes
    input_b_tile_bytes = shape.tile_k * shape.tile_n * data_format.input_element_bytes
    output_tile_bytes = shape.tile_m * shape.tile_n * data_format.output_element_bytes
    accumulator_tile_bytes = shape.tile_m * shape.tile_n * data_format.accumulator_element_bytes
    return input_a_tile_bytes + input_b_tile_bytes + output_tile_bytes + accumulator_tile_bytes


def _is_supported_dense_task(task: ContractionTask) -> bool:
    return task.structure == "dense" and task.gemm_m > 0 and task.gemm_k > 0 and task.gemm_n > 0


def _input_dtype_name(data_format: UpmemDataFormat) -> str:
    if data_format.input_element_bytes == 1:
        return "int8"
    if data_format.input_element_bytes == 2:
        return "int16"
    return f"{data_format.input_element_bytes}_byte_input"
