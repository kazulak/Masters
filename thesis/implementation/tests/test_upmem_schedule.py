from __future__ import annotations

from quantum_bench.core.records import (
    CircuitSpec,
    ContractionTask,
    PathSummary,
    TaskGraph,
    TensorNetworkSpec,
)
from quantum_bench.targets.upmem import (
    REQUIRES_TILING_NOT_IMPLEMENTED,
    UNSUPPORTED_DENSE_GEMM_SHAPE,
    UPMEM_DENSE_ESTIMATE_KEY,
    UPMEM_PROFILE,
    annotate_task_graph_with_upmem_estimates,
    estimate_dense_task,
    estimate_dense_task_graph,
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
    task_estimate = estimate.as_task_estimate()

    assert estimate.fits_wram_without_tiling is True
    assert estimate.wram_fit is True
    assert estimate.requires_tiling is False
    assert estimate.tiling_implemented is False
    assert estimate.estimated_tile_count == 1
    assert estimate.estimated_parallel_tiles == 1
    assert estimate.reject_reason is None
    assert estimate.host_to_dpu_bytes == 128
    assert estimate.dpu_to_host_bytes == 256
    assert estimate.mram_to_wram_bytes == 384
    assert estimate.working_set_bytes == 640
    assert estimate.working_set_bytes <= UPMEM_PROFILE.wram_bytes
    assert task_estimate["estimate_key"] == UPMEM_DENSE_ESTIMATE_KEY
    assert task_estimate["gemm_m"] == 8
    assert task_estimate["gemm_k"] == 8
    assert task_estimate["gemm_n"] == 8
    assert task_estimate["tiling_implemented"] is False
    _assert_nonnegative_task_bytes(estimate)


def test_large_dense_task_estimate_requires_tiling_or_rejects() -> None:
    task = _dense_task("large", 256, 256, 256)
    estimate = estimate_dense_task(task)

    assert estimate.fits_wram_without_tiling is False
    assert estimate.supported is True
    assert estimate.wram_fit is False
    assert estimate.requires_tiling is True
    assert estimate.tiling_implemented is False
    assert estimate.estimated_tile_count > 1
    assert estimate.estimated_parallel_tiles >= 1
    assert estimate.working_set_bytes > UPMEM_PROFILE.wram_bytes
    assert estimate.reject_reason == REQUIRES_TILING_NOT_IMPLEMENTED
    _assert_nonnegative_task_bytes(estimate)


def test_unsupported_dense_task_estimate_is_explicitly_rejected() -> None:
    task = _dense_task("unsupported", 8, 8, 8, structure="sparse")
    estimate = estimate_dense_task(task)
    task_estimate = estimate.as_task_estimate()

    assert estimate.supported is False
    assert estimate.wram_fit is False
    assert estimate.requires_tiling is False
    assert estimate.tiling_implemented is False
    assert estimate.estimated_tile_count == 0
    assert estimate.estimated_parallel_tiles == 0
    assert estimate.host_to_dpu_bytes == 0
    assert estimate.dpu_to_host_bytes == 0
    assert estimate.mram_to_wram_bytes == 0
    assert estimate.working_set_bytes == 0
    assert estimate.reject_reason == UNSUPPORTED_DENSE_GEMM_SHAPE
    assert task_estimate["reject_reason"] == UNSUPPORTED_DENSE_GEMM_SHAPE


def test_dense_schedule_metadata_reports_transfer_and_wram_estimates() -> None:
    small = _dense_task("small", 8, 8, 8)
    large = _dense_task("large", 256, 256, 256)
    schedule = estimate_dense_task_graph(_task_graph(small, large))
    metadata = schedule.metadata()

    assert len(schedule.tasks) == 2
    assert schedule.tasks_fit_without_tiling == 1
    assert schedule.tasks_requiring_tiling == 1
    assert schedule.all_tasks_fit_without_tiling is False
    assert schedule.total_estimated_tile_count > 1
    assert schedule.max_estimated_parallel_tiles >= 1
    assert schedule.unsupported_tasks == 0
    assert schedule.max_working_set_bytes == max(task.working_set_bytes for task in schedule.tasks)
    assert metadata["target"] == "upmem"
    assert metadata["estimate_key"] == UPMEM_DENSE_ESTIMATE_KEY
    assert metadata["route_family"] == "dense_gemm"
    assert metadata["tiling_implemented"] is False
    assert metadata["task_count"] == 2
    assert metadata["first_reject_reason"] == schedule.first_reject_reason()
    _assert_nonnegative_metadata_bytes(metadata)


def test_annotated_task_graph_has_upmem_estimates_on_each_task() -> None:
    graph = _task_graph(_dense_task("small", 8, 8, 8), _dense_task("large", 256, 256, 256))
    annotated, schedule = annotate_task_graph_with_upmem_estimates(graph)

    assert len(annotated.tasks) == len(graph.tasks)
    assert len(schedule.tasks) == len(graph.tasks)
    for task in annotated.tasks:
        estimate = task.target_estimates[UPMEM_DENSE_ESTIMATE_KEY]
        assert estimate["estimate_key"] == UPMEM_DENSE_ESTIMATE_KEY
        assert "gemm_m" in estimate
        assert "gemm_k" in estimate
        assert "gemm_n" in estimate
        assert estimate["tiling_implemented"] is False
        assert "host_to_dpu_bytes" in estimate
        assert "dpu_to_host_bytes" in estimate
        assert "mram_to_wram_bytes" in estimate
