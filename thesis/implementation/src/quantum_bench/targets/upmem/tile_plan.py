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
UPMEM_EXECUTION_CLASS_L1_WRAM = "L1_WRAM"
UPMEM_EXECUTION_CLASS_L2_SINGLE_DPU_MRAM = "L2_SINGLE_DPU_MRAM"
UPMEM_L1_KERNEL_STRATEGY = "l1_padded_direct_v1"
UPMEM_L2_KERNEL_STRATEGY = "l2_single_dpu_mram_wram_tiled_v1"
UPMEM_L2_EFFECTIVE_WRAM_BYTES = 60 * 1024
UPMEM_L2_PER_DPU_MRAM_BYTES = 64 * 1024 * 1024
UPMEM_L2_MAX_HOST_BLOB_BYTES = 16 * 1024 * 1024
UPMEM_L2_NATIVE_MAX_DIM = 512
UPMEM_L2_ALIGNMENT_RESERVE_BYTES = 2048
UPMEM_L2_TILE_CANDIDATES = (
    (64, 64, 64),
    (64, 32, 64),
    (32, 64, 64),
    (32, 32, 64),
    (32, 32, 32),
    (16, 32, 64),
    (32, 16, 64),
    (16, 16, 64),
    (16, 16, 32),
    (8, 16, 32),
    (16, 8, 32),
    (8, 8, 32),
)


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


@dataclass(frozen=True)
class UpmemL2TiledExecutionPlan:
    supported: bool
    reason: str | None
    execution_class: str
    kernel_strategy: str
    gemm_m: int
    gemm_k: int
    gemm_n: int
    tile_m: int
    tile_n: int
    tile_k: int
    output_tile_count: int
    k_tile_count: int
    total_tile_steps: int
    input_a_tile_bytes: int
    input_b_tile_bytes: int
    accumulator_tile_bytes: int
    local_output_scratch_bytes: int
    alignment_padding_bytes: int
    estimated_wram_bytes_per_tile: int
    effective_wram_bytes: int
    mram_bytes_a: int
    mram_bytes_b: int
    mram_bytes_c: int
    total_mram_bytes: int
    conservative_full_task_bytes: int
    max_l2_host_blob_bytes: int
    host_blob_bytes: int
    native_max_dim: int
    mram_resident_operands: bool
    wram_tiled: bool

    def as_dict(self) -> JsonDict:
        return {
            "supported": self.supported,
            "reason": self.reason,
            "execution_class": self.execution_class,
            "kernel_strategy": self.kernel_strategy,
            "gemm_m": self.gemm_m,
            "gemm_k": self.gemm_k,
            "gemm_n": self.gemm_n,
            "tile_m": self.tile_m,
            "tile_n": self.tile_n,
            "tile_k": self.tile_k,
            "output_tile_count": self.output_tile_count,
            "k_tile_count": self.k_tile_count,
            "total_tile_steps": self.total_tile_steps,
            "input_a_tile_bytes": self.input_a_tile_bytes,
            "input_b_tile_bytes": self.input_b_tile_bytes,
            "accumulator_tile_bytes": self.accumulator_tile_bytes,
            "local_output_scratch_bytes": self.local_output_scratch_bytes,
            "alignment_padding_bytes": self.alignment_padding_bytes,
            "estimated_wram_bytes_per_tile": self.estimated_wram_bytes_per_tile,
            "effective_wram_bytes": self.effective_wram_bytes,
            "mram_bytes_a": self.mram_bytes_a,
            "mram_bytes_b": self.mram_bytes_b,
            "mram_bytes_c": self.mram_bytes_c,
            "total_mram_bytes": self.total_mram_bytes,
            "conservative_full_task_bytes": self.conservative_full_task_bytes,
            "max_l2_host_blob_bytes": self.max_l2_host_blob_bytes,
            "host_blob_bytes": self.host_blob_bytes,
            "native_max_dim": self.native_max_dim,
            "mram_resident_operands": self.mram_resident_operands,
            "wram_tiled": self.wram_tiled,
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


def plan_l2_tiled_execution(
    gemm_m: int,
    gemm_k: int,
    gemm_n: int,
    *,
    effective_wram_bytes: int = UPMEM_L2_EFFECTIVE_WRAM_BYTES,
    per_dpu_mram_bytes: int = UPMEM_L2_PER_DPU_MRAM_BYTES,
    max_l2_host_blob_bytes: int = UPMEM_L2_MAX_HOST_BLOB_BYTES,
    native_max_dim: int = UPMEM_L2_NATIVE_MAX_DIM,
    alignment_padding_bytes: int = UPMEM_L2_ALIGNMENT_RESERVE_BYTES,
) -> UpmemL2TiledExecutionPlan:
    gemm_m = int(gemm_m)
    gemm_k = int(gemm_k)
    gemm_n = int(gemm_n)
    mram_bytes_a = gemm_m * gemm_k * DENSE_INT8_FORMAT.input_element_bytes
    mram_bytes_b = gemm_k * gemm_n * DENSE_INT8_FORMAT.input_element_bytes
    mram_bytes_c = gemm_m * gemm_n * DENSE_INT8_FORMAT.output_element_bytes
    c_accumulator_bytes = gemm_m * gemm_n * DENSE_INT8_FORMAT.accumulator_element_bytes
    total_mram_bytes = mram_bytes_a + mram_bytes_b + mram_bytes_c
    conservative_full_task_bytes = total_mram_bytes + c_accumulator_bytes
    host_blob_bytes = mram_bytes_a + mram_bytes_b + (gemm_m * gemm_n * 8)

    if min(gemm_m, gemm_k, gemm_n) <= 0:
        return _unsupported_l2_plan(
            "unsupported_l2_native_shape_limit",
            gemm_m,
            gemm_k,
            gemm_n,
            effective_wram_bytes,
            max_l2_host_blob_bytes,
            native_max_dim,
            mram_bytes_a,
            mram_bytes_b,
            mram_bytes_c,
            total_mram_bytes,
            conservative_full_task_bytes,
            host_blob_bytes,
            alignment_padding_bytes,
        )
    if conservative_full_task_bytes <= effective_wram_bytes:
        return _unsupported_l2_plan(
            "not_l2_wram_resident",
            gemm_m,
            gemm_k,
            gemm_n,
            effective_wram_bytes,
            max_l2_host_blob_bytes,
            native_max_dim,
            mram_bytes_a,
            mram_bytes_b,
            mram_bytes_c,
            total_mram_bytes,
            conservative_full_task_bytes,
            host_blob_bytes,
            alignment_padding_bytes,
        )
    if total_mram_bytes > per_dpu_mram_bytes:
        return _unsupported_l2_plan(
            "unsupported_l2_mram_capacity",
            gemm_m,
            gemm_k,
            gemm_n,
            effective_wram_bytes,
            max_l2_host_blob_bytes,
            native_max_dim,
            mram_bytes_a,
            mram_bytes_b,
            mram_bytes_c,
            total_mram_bytes,
            conservative_full_task_bytes,
            host_blob_bytes,
            alignment_padding_bytes,
        )
    if max(gemm_m, gemm_k, gemm_n) > native_max_dim:
        return _unsupported_l2_plan(
            "unsupported_l2_native_shape_limit",
            gemm_m,
            gemm_k,
            gemm_n,
            effective_wram_bytes,
            max_l2_host_blob_bytes,
            native_max_dim,
            mram_bytes_a,
            mram_bytes_b,
            mram_bytes_c,
            total_mram_bytes,
            conservative_full_task_bytes,
            host_blob_bytes,
            alignment_padding_bytes,
        )
    if any(dim % 8 != 0 for dim in (gemm_m, gemm_k, gemm_n)):
        return _unsupported_l2_plan(
            "unsupported_l2_native_shape_limit",
            gemm_m,
            gemm_k,
            gemm_n,
            effective_wram_bytes,
            max_l2_host_blob_bytes,
            native_max_dim,
            mram_bytes_a,
            mram_bytes_b,
            mram_bytes_c,
            total_mram_bytes,
            conservative_full_task_bytes,
            host_blob_bytes,
            alignment_padding_bytes,
        )
    if host_blob_bytes > max_l2_host_blob_bytes:
        return _unsupported_l2_plan(
            "unsupported_l2_blob_size",
            gemm_m,
            gemm_k,
            gemm_n,
            effective_wram_bytes,
            max_l2_host_blob_bytes,
            native_max_dim,
            mram_bytes_a,
            mram_bytes_b,
            mram_bytes_c,
            total_mram_bytes,
            conservative_full_task_bytes,
            host_blob_bytes,
            alignment_padding_bytes,
        )

    for candidate_m, candidate_k, candidate_n in UPMEM_L2_TILE_CANDIDATES:
        tile_m = min(candidate_m, gemm_m)
        tile_k = min(candidate_k, gemm_k)
        tile_n = min(candidate_n, gemm_n)
        input_a_tile_bytes = tile_m * tile_k * DENSE_INT8_FORMAT.input_element_bytes
        input_b_tile_bytes = tile_k * tile_n * DENSE_INT8_FORMAT.input_element_bytes
        accumulator_tile_bytes = tile_m * tile_n * DENSE_INT8_FORMAT.accumulator_element_bytes
        local_output_scratch_bytes = 0
        estimated_wram_bytes = (
            input_a_tile_bytes
            + input_b_tile_bytes
            + accumulator_tile_bytes
            + local_output_scratch_bytes
            + alignment_padding_bytes
        )
        if estimated_wram_bytes > effective_wram_bytes:
            continue
        output_tile_count = math.ceil(gemm_m / tile_m) * math.ceil(gemm_n / tile_n)
        k_tile_count = math.ceil(gemm_k / tile_k)
        return UpmemL2TiledExecutionPlan(
            supported=True,
            reason=None,
            execution_class=UPMEM_EXECUTION_CLASS_L2_SINGLE_DPU_MRAM,
            kernel_strategy=UPMEM_L2_KERNEL_STRATEGY,
            gemm_m=gemm_m,
            gemm_k=gemm_k,
            gemm_n=gemm_n,
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
            output_tile_count=output_tile_count,
            k_tile_count=k_tile_count,
            total_tile_steps=output_tile_count * k_tile_count,
            input_a_tile_bytes=input_a_tile_bytes,
            input_b_tile_bytes=input_b_tile_bytes,
            accumulator_tile_bytes=accumulator_tile_bytes,
            local_output_scratch_bytes=local_output_scratch_bytes,
            alignment_padding_bytes=alignment_padding_bytes,
            estimated_wram_bytes_per_tile=estimated_wram_bytes,
            effective_wram_bytes=effective_wram_bytes,
            mram_bytes_a=mram_bytes_a,
            mram_bytes_b=mram_bytes_b,
            mram_bytes_c=mram_bytes_c,
            total_mram_bytes=total_mram_bytes,
            conservative_full_task_bytes=conservative_full_task_bytes,
            max_l2_host_blob_bytes=max_l2_host_blob_bytes,
            host_blob_bytes=host_blob_bytes,
            native_max_dim=native_max_dim,
            mram_resident_operands=True,
            wram_tiled=True,
        )

    return _unsupported_l2_plan(
        "unsupported_l2_tile_plan",
        gemm_m,
        gemm_k,
        gemm_n,
        effective_wram_bytes,
        max_l2_host_blob_bytes,
        native_max_dim,
        mram_bytes_a,
        mram_bytes_b,
        mram_bytes_c,
        total_mram_bytes,
        conservative_full_task_bytes,
        host_blob_bytes,
        alignment_padding_bytes,
    )


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


def _unsupported_l2_plan(
    reason: str,
    gemm_m: int,
    gemm_k: int,
    gemm_n: int,
    effective_wram_bytes: int,
    max_l2_host_blob_bytes: int,
    native_max_dim: int,
    mram_bytes_a: int,
    mram_bytes_b: int,
    mram_bytes_c: int,
    total_mram_bytes: int,
    conservative_full_task_bytes: int,
    host_blob_bytes: int,
    alignment_padding_bytes: int,
) -> UpmemL2TiledExecutionPlan:
    return UpmemL2TiledExecutionPlan(
        supported=False,
        reason=reason,
        execution_class=UPMEM_EXECUTION_CLASS_L2_SINGLE_DPU_MRAM,
        kernel_strategy=UPMEM_L2_KERNEL_STRATEGY,
        gemm_m=gemm_m,
        gemm_k=gemm_k,
        gemm_n=gemm_n,
        tile_m=0,
        tile_n=0,
        tile_k=0,
        output_tile_count=0,
        k_tile_count=0,
        total_tile_steps=0,
        input_a_tile_bytes=0,
        input_b_tile_bytes=0,
        accumulator_tile_bytes=0,
        local_output_scratch_bytes=0,
        alignment_padding_bytes=alignment_padding_bytes,
        estimated_wram_bytes_per_tile=0,
        effective_wram_bytes=effective_wram_bytes,
        mram_bytes_a=mram_bytes_a,
        mram_bytes_b=mram_bytes_b,
        mram_bytes_c=mram_bytes_c,
        total_mram_bytes=total_mram_bytes,
        conservative_full_task_bytes=conservative_full_task_bytes,
        max_l2_host_blob_bytes=max_l2_host_blob_bytes,
        host_blob_bytes=host_blob_bytes,
        native_max_dim=native_max_dim,
        mram_resident_operands=True,
        wram_tiled=True,
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
