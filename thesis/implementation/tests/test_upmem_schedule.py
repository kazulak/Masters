from __future__ import annotations

from quantum_bench.core.records import (
    CircuitSpec,
    ContractionTask,
    PathSummary,
    TaskGraph,
    TensorNetworkSpec,
)
from quantum_bench.targets.upmem import UPMEM_PROFILE, estimate_dense_task, estimate_dense_task_graph


def _dense_task(task_id: str, gemm_m: int, gemm_k: int, gemm_n: int) -> ContractionTask:
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
        structure="dense",
        estimated_flops=2 * gemm_m * gemm_k * gemm_n,
        estimated_bytes=gemm_m * gemm_k + gemm_k * gemm_n + gemm_m * gemm_n,
    )


def _task_graph(*tasks: ContractionTask) -> TaskGraph:
    circuit = CircuitSpec("synthetic", 2, (), {})
    network = TensorNetworkSpec(circuit, (), (), "")
    path_summary = PathSummary("unit_test", "manual", len(tasks), None, None, None, "")
    return TaskGraph(network, tasks, (), path_summary, 0.0)


def _assert_nonnegative_task_bytes(estimate: object) -> None:
    for field in (
        "host_to_dpu_bytes",
        "dpu_to_host_bytes",
        "mram_to_wram_bytes",
        "working_set_bytes",
    ):
        assert getattr(estimate, field) >= 0


def _assert_nonnegative_metadata_bytes(metadata: dict[str, object]) -> None:
    for field in (
        "total_host_to_dpu_bytes",
        "total_dpu_to_host_bytes",
        "total_mram_to_wram_bytes",
        "max_working_set_bytes",
    ):
        assert field in metadata
        assert isinstance(metadata[field], int)
        assert metadata[field] >= 0


def test_small_dense_task_estimate_fits_wram() -> None:
    task = _dense_task("small", 8, 8, 8)
    estimate = estimate_dense_task(task)

    assert estimate.fits_wram_without_tiling is True
    assert estimate.reject_reason is None
    assert estimate.host_to_dpu_bytes == 128
    assert estimate.dpu_to_host_bytes == 256
    assert estimate.mram_to_wram_bytes == 384
    assert estimate.working_set_bytes == 640
    assert estimate.working_set_bytes <= UPMEM_PROFILE.wram_bytes
    _assert_nonnegative_task_bytes(estimate)


def test_large_dense_task_estimate_requires_tiling_or_rejects() -> None:
    task = _dense_task("large", 256, 256, 256)
    estimate = estimate_dense_task(task)

    assert estimate.fits_wram_without_tiling is False
    assert estimate.working_set_bytes > UPMEM_PROFILE.wram_bytes
    assert estimate.reject_reason is not None
    assert "WRAM" in estimate.reject_reason
    assert "tiling" in estimate.reject_reason
    _assert_nonnegative_task_bytes(estimate)


def test_dense_schedule_metadata_reports_transfer_and_wram_estimates() -> None:
    small = _dense_task("small", 8, 8, 8)
    large = _dense_task("large", 256, 256, 256)
    schedule = estimate_dense_task_graph(_task_graph(small, large))
    metadata = schedule.metadata()

    assert len(schedule.tasks) == 2
    assert schedule.tasks_fit_without_tiling == 1
    assert schedule.tasks_requiring_tiling == 1
    assert schedule.all_tasks_fit_without_tiling is False
    assert schedule.max_working_set_bytes == max(task.working_set_bytes for task in schedule.tasks)
    assert metadata["target"] == "upmem"
    assert metadata["route_family"] == "dense_gemm"
    assert metadata["task_count"] == 2
    assert metadata["first_reject_reason"] == schedule.first_reject_reason()
    _assert_nonnegative_metadata_bytes(metadata)
