from __future__ import annotations

from quantum_bench.core.records import ContractionTask
from quantum_bench.targets.upmem.tile_plan import (
    REQUIRES_TILING_NOT_IMPLEMENTED,
    UNSUPPORTED_DENSE_GEMM_SHAPE,
    UPMEM_DENSE_ESTIMATE_KEY,
    UPMEM_DENSE_TILE_PLAN_MODEL,
    UPMEM_EXECUTION_CLASS_L2_SINGLE_DPU_MRAM,
    UPMEM_L2_EFFECTIVE_WRAM_BYTES,
    UPMEM_L2_KERNEL_STRATEGY,
    UPMEM_L2_MAX_HOST_BLOB_BYTES,
    UPMEM_L2_NATIVE_MAX_DIM,
    UPMEM_PROFILE,
    plan_l2_tiled_execution,
    plan_dense_task,
)


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


def test_small_dense_task_tile_plan_is_single_full_task_tile() -> None:
    task = _dense_task("small", 8, 8, 8)
    plan = plan_dense_task(task)
    row = plan.as_artifact_row(task)

    assert plan.target == "upmem"
    assert plan.estimate_key == UPMEM_DENSE_ESTIMATE_KEY
    assert plan.model == UPMEM_DENSE_TILE_PLAN_MODEL
    assert plan.supported is True
    assert plan.tile_shape.tile_m == 8
    assert plan.tile_shape.tile_k == 8
    assert plan.tile_shape.tile_n == 8
    assert plan.tile_counts.tile_count_m == 1
    assert plan.tile_counts.tile_count_k == 1
    assert plan.tile_counts.tile_count_n == 1
    assert plan.tile_counts.total_tile_count == 1
    assert plan.fits_wram is True
    assert plan.requires_tiling is False
    assert plan.tiling_implemented is False
    assert plan.double_buffer_possible is True
    assert plan.requires_host_aggregation is False
    assert plan.working_set_bytes == 640
    assert plan.working_set_bytes <= UPMEM_PROFILE.wram_bytes
    assert row["task_id"] == "small"
    assert row["model"] == UPMEM_DENSE_TILE_PLAN_MODEL
    assert row["memory_model_note"].startswith("conservative")


def test_large_dense_task_tile_plan_fits_tile_but_requires_tiling() -> None:
    plan = plan_dense_task(_dense_task("large", 256, 256, 256))

    assert plan.supported is True
    assert plan.fits_wram is True
    assert plan.requires_tiling is True
    assert plan.tiling_implemented is False
    assert plan.reject_reason == REQUIRES_TILING_NOT_IMPLEMENTED
    assert plan.tile_shape.tile_m == 64
    assert plan.tile_shape.tile_k == 256
    assert plan.tile_shape.tile_n == 64
    assert plan.tile_counts.tile_count_m == 4
    assert plan.tile_counts.tile_count_k == 1
    assert plan.tile_counts.tile_count_n == 4
    assert plan.tile_counts.total_tile_count == 16
    assert plan.requires_host_aggregation is False
    assert plan.double_buffer_possible is False
    assert plan.working_set_bytes == UPMEM_PROFILE.wram_bytes
    assert plan.full_task_working_set_bytes > UPMEM_PROFILE.wram_bytes


def test_split_k_tile_plan_requires_host_aggregation() -> None:
    plan = plan_dense_task(_dense_task("split_k", 8, 65536, 8))

    assert plan.supported is True
    assert plan.fits_wram is True
    assert plan.requires_tiling is True
    assert plan.tiling_implemented is False
    assert plan.requires_host_aggregation is True
    assert plan.tile_shape.tile_k < plan.gemm_k
    assert plan.tile_counts.tile_count_k > 1
    assert plan.tile_counts.total_tile_count > 1
    assert plan.working_set_bytes <= UPMEM_PROFILE.wram_bytes


def test_unsupported_tile_plan_is_explicit() -> None:
    plan = plan_dense_task(_dense_task("sparse", 8, 8, 8, structure="sparse"))

    assert plan.supported is False
    assert plan.fits_wram is False
    assert plan.requires_tiling is False
    assert plan.tiling_implemented is False
    assert plan.tile_counts.total_tile_count == 0
    assert plan.estimated_parallel_tiles == 0
    assert plan.reject_reason == UNSUPPORTED_DENSE_GEMM_SHAPE


def test_l2_tiled_execution_plan_supports_starter_shapes() -> None:
    shapes = {
        "synthetic_l2_square": (96, 96, 96),
        "synthetic_l2_rect": (128, 128, 64),
        "synthetic_l2_kheavy": (72, 512, 32),
    }

    for m, k, n in shapes.values():
        plan = plan_l2_tiled_execution(m, k, n)

        assert plan.supported is True
        assert plan.reason is None
        assert plan.execution_class == UPMEM_EXECUTION_CLASS_L2_SINGLE_DPU_MRAM
        assert plan.kernel_strategy == UPMEM_L2_KERNEL_STRATEGY
        assert plan.conservative_full_task_bytes > UPMEM_L2_EFFECTIVE_WRAM_BYTES
        assert plan.estimated_wram_bytes_per_tile <= UPMEM_L2_EFFECTIVE_WRAM_BYTES
        assert plan.host_blob_bytes <= UPMEM_L2_MAX_HOST_BLOB_BYTES
        assert max(plan.gemm_m, plan.gemm_k, plan.gemm_n) <= UPMEM_L2_NATIVE_MAX_DIM
        assert plan.tile_m > 0
        assert plan.tile_k > 0
        assert plan.tile_n > 0
        assert plan.total_tile_steps == plan.output_tile_count * plan.k_tile_count
        assert plan.mram_resident_operands is True
        assert plan.wram_tiled is True


def test_l2_tiled_execution_plan_is_distinct_from_l1_wram_fit() -> None:
    small = plan_l2_tiled_execution(16, 16, 16)

    assert small.supported is False
    assert small.reason == "not_l2_wram_resident"
    assert small.conservative_full_task_bytes <= UPMEM_L2_EFFECTIVE_WRAM_BYTES


def test_l2_tiled_execution_plan_rejects_host_blob_over_cap() -> None:
    plan = plan_l2_tiled_execution(96, 96, 96, max_l2_host_blob_bytes=1024)

    assert plan.supported is False
    assert plan.reason == "unsupported_l2_blob_size"
