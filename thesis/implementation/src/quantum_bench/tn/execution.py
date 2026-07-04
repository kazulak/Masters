from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from quantum_bench.core.records import ContractionTask, TaskExecutionMetric, TaskGraph
from quantum_bench.tn.network import TensorNetworkValue


def execute_task_sequence_np_einsum(graph: TaskGraph, network: TensorNetworkValue) -> tuple[np.ndarray, dict]:
    tensors = {tensor.spec.id: np.asarray(tensor.array, dtype=np.complex128) for tensor in network.tensors}
    labels = {tensor.spec.id: tensor.spec.labels for tensor in network.tensors}
    live_ids = set(tensors)
    task_metrics: list[TaskExecutionMetric] = []
    peak_live_bytes = _live_tensor_bytes(tensors, live_ids)
    max_intermediate_bytes = 0

    if not graph.tasks:
        output, final_id, final_labels, transposed = execute_empty_task_graph(graph, network)
        return output, {
            "execution_engine": "task_sequence_np_einsum",
            "task_count": 0,
            "task_metrics": task_metrics,
            "peak_intermediate_bytes": int(output.nbytes),
            "max_intermediate_tensor_bytes": 0,
            "final_tensor_id": final_id,
            "final_tensor_labels": final_labels,
            "output_labels": graph.network.output_labels,
            "final_transpose_applied": transposed,
        }

    remaining_uses = remaining_input_uses(graph)
    final_tensor_id = graph.tasks[-1].output_tensor_id
    final_labels: tuple[int, ...] | None = None

    for task in graph.tasks:
        left_id, right_id = task.input_tensor_ids
        if left_id not in tensors or right_id not in tensors:
            missing = [tensor_id for tensor_id in task.input_tensor_ids if tensor_id not in tensors]
            raise ValueError(f"Task {task.id} references unavailable tensor(s): {', '.join(missing)}")

        task_start = time.perf_counter()
        intermediate = np.einsum(task.index_expression, tensors[left_id], tensors[right_id], optimize=False)
        task_time_s = time.perf_counter() - task_start
        intermediate = np.asarray(intermediate, dtype=np.complex128)

        tensors[task.output_tensor_id] = intermediate
        labels[task.output_tensor_id] = task.output_labels
        live_ids.add(task.output_tensor_id)
        intermediate_bytes = int(intermediate.nbytes)
        max_intermediate_bytes = max(max_intermediate_bytes, intermediate_bytes)
        peak_live_bytes = max(peak_live_bytes, _live_tensor_bytes(tensors, live_ids))
        final_labels = task.output_labels

        task_metrics.append(
            TaskExecutionMetric(
                task_id=task.id,
                input_tensor_ids=task.input_tensor_ids,
                output_tensor_id=task.output_tensor_id,
                input_shapes=task.input_shapes,
                output_shape=task.output_shape,
                contracted_labels=task.contracted_labels,
                estimated_flops=task.estimated_flops,
                estimated_bytes=task.estimated_bytes,
                execution_time_s=task_time_s,
                intermediate_tensor_bytes=intermediate_bytes,
                target_estimates=task.target_estimates,
            )
        )

        release_dead_inputs(task.input_tensor_ids, task.output_tensor_id, final_tensor_id, tensors, labels, live_ids, remaining_uses)

    if final_tensor_id not in tensors or final_labels is None:
        raise ValueError(f"Task sequence did not produce final tensor {final_tensor_id}")

    output, transposed = order_final_tensor(tensors[final_tensor_id], final_labels, graph.network.output_labels)
    return output, {
        "execution_engine": "task_sequence_np_einsum",
        "task_count": len(task_metrics),
        "task_metrics": task_metrics,
        "peak_intermediate_bytes": peak_live_bytes,
        "max_intermediate_tensor_bytes": max_intermediate_bytes,
        "final_tensor_id": final_tensor_id,
        "final_tensor_labels": final_labels,
        "output_labels": graph.network.output_labels,
        "final_transpose_applied": transposed,
    }


def execute_task_frontier_np_einsum(
    graph: TaskGraph,
    network: TensorNetworkValue,
    *,
    frontier_worker_count: int = 1,
) -> tuple[np.ndarray, dict]:
    if frontier_worker_count < 1:
        raise ValueError("frontier_worker_count must be >= 1")
    tensors = {tensor.spec.id: np.asarray(tensor.array, dtype=np.complex128) for tensor in network.tensors}
    labels = {tensor.spec.id: tensor.spec.labels for tensor in network.tensors}
    live_ids = set(tensors)
    task_metrics: list[TaskExecutionMetric] = []
    task_metrics_by_id: dict[str, TaskExecutionMetric] = {}
    executed_task_ids: list[str] = []
    executed_task_set: set[str] = set()
    peak_live_bytes = _live_tensor_bytes(tensors, live_ids)
    max_intermediate_bytes = 0
    scheduler_overhead_s = 0.0

    if not graph.tasks:
        output, final_id, final_labels, transposed = execute_empty_task_graph(graph, network)
        return output, {
            "execution_engine": "task_frontier_np_einsum",
            "task_count": 0,
            "task_metrics": task_metrics,
            "peak_intermediate_bytes": int(output.nbytes),
            "max_intermediate_tensor_bytes": 0,
            "final_tensor_id": final_id,
            "final_tensor_labels": final_labels,
            "output_labels": graph.network.output_labels,
            "final_transpose_applied": transposed,
            "parallelism_mode": "frontier",
            "parallelism_evidence_type": "executed",
            "execution_plan_kind": "taskgraph_frontier_scheduler",
            "execution_plan_executed": True,
            "frontier_scheduler_enabled": True,
            "frontier_parallel_execution": False,
            "frontier_worker_count": int(frontier_worker_count),
            "frontier_wave_count": 0,
            "frontier_widths": (),
            "max_frontier_width": 0,
            "mean_frontier_width": 0.0,
            "frontier_executed_task_count": 0,
            "frontier_executed_parallel_task_count": 0,
            "executed_parallel_task_count": 0,
            "scheduler_overhead_s": 0.0,
            "duplicate_contraction_check": "passed",
            "missing_dependency_check": "passed",
        }

    waves = frontier_waves(graph)
    remaining_uses = remaining_input_uses(graph)
    final_tensor_id = graph.tasks[-1].output_tensor_id
    final_labels: tuple[int, ...] | None = None
    frontier_parallel_execution = frontier_worker_count > 1 and any(len(wave) > 1 for wave in waves)
    executed_parallel_task_count = 0

    for wave in waves:
        scheduler_start = time.perf_counter()
        for task in wave:
            if task.id in executed_task_set:
                raise ValueError(f"Task {task.id} scheduled more than once")
            missing = [tensor_id for tensor_id in task.input_tensor_ids if tensor_id not in tensors]
            if missing:
                raise ValueError(f"Task {task.id} scheduled before input tensor(s) available: {', '.join(missing)}")
        task_inputs = {
            task.id: (
                np.asarray(tensors[task.input_tensor_ids[0]], dtype=np.complex128),
                np.asarray(tensors[task.input_tensor_ids[1]], dtype=np.complex128),
            )
            for task in wave
        }
        scheduler_overhead_s += time.perf_counter() - scheduler_start

        if frontier_worker_count > 1 and len(wave) > 1:
            results = _execute_wave_threaded(wave, task_inputs, frontier_worker_count)
            executed_parallel_task_count += len(wave)
        else:
            results = [_execute_frontier_task(task, task_inputs[task.id]) for task in wave]

        scheduler_start = time.perf_counter()
        for task, intermediate, task_time_s in sorted(results, key=lambda item: item[0].id):
            if task.id in executed_task_set:
                raise ValueError(f"Task {task.id} completed more than once")
            intermediate = np.asarray(intermediate, dtype=np.complex128)
            tensors[task.output_tensor_id] = intermediate
            labels[task.output_tensor_id] = task.output_labels
            live_ids.add(task.output_tensor_id)
            intermediate_bytes = int(intermediate.nbytes)
            max_intermediate_bytes = max(max_intermediate_bytes, intermediate_bytes)
            peak_live_bytes = max(peak_live_bytes, _live_tensor_bytes(tensors, live_ids))
            final_labels = task.output_labels
            metric = TaskExecutionMetric(
                task_id=task.id,
                input_tensor_ids=task.input_tensor_ids,
                output_tensor_id=task.output_tensor_id,
                input_shapes=task.input_shapes,
                output_shape=task.output_shape,
                contracted_labels=task.contracted_labels,
                estimated_flops=task.estimated_flops,
                estimated_bytes=task.estimated_bytes,
                execution_time_s=task_time_s,
                intermediate_tensor_bytes=intermediate_bytes,
                target_estimates=task.target_estimates,
            )
            task_metrics_by_id[task.id] = metric
            executed_task_ids.append(task.id)
            executed_task_set.add(task.id)

        for task in wave:
            release_dead_inputs(task.input_tensor_ids, task.output_tensor_id, final_tensor_id, tensors, labels, live_ids, remaining_uses)
        scheduler_overhead_s += time.perf_counter() - scheduler_start

    expected_task_ids = tuple(task.id for task in graph.tasks)
    if set(executed_task_ids) != set(expected_task_ids):
        missing = sorted(set(expected_task_ids) - executed_task_set)
        extras = sorted(executed_task_set - set(expected_task_ids))
        raise ValueError(f"Frontier execution task set mismatch; missing={missing}, extras={extras}")
    if len(executed_task_ids) != len(executed_task_set):
        raise ValueError("Frontier execution detected duplicate task execution")
    if final_tensor_id not in tensors or final_labels is None:
        raise ValueError(f"Frontier execution did not produce final tensor {final_tensor_id}")

    task_metrics = [task_metrics_by_id[task.id] for task in graph.tasks]
    output, transposed = order_final_tensor(tensors[final_tensor_id], final_labels, graph.network.output_labels)
    frontier_widths = tuple(len(wave) for wave in waves)
    max_frontier_width = max(frontier_widths, default=0)
    mean_frontier_width = (sum(frontier_widths) / len(frontier_widths)) if frontier_widths else 0.0
    return output, {
        "execution_engine": "task_frontier_np_einsum",
        "task_count": len(task_metrics),
        "task_metrics": task_metrics,
        "peak_intermediate_bytes": peak_live_bytes,
        "max_intermediate_tensor_bytes": max_intermediate_bytes,
        "final_tensor_id": final_tensor_id,
        "final_tensor_labels": final_labels,
        "output_labels": graph.network.output_labels,
        "final_transpose_applied": transposed,
        "parallelism_mode": "frontier",
        "parallelism_evidence_type": "executed",
        "execution_plan_kind": "taskgraph_frontier_scheduler",
        "execution_plan_executed": True,
        "frontier_scheduler_enabled": True,
        "frontier_parallel_execution": bool(frontier_parallel_execution),
        "frontier_worker_count": int(frontier_worker_count),
        "frontier_wave_count": len(waves),
        "frontier_widths": frontier_widths,
        "max_frontier_width": max_frontier_width,
        "mean_frontier_width": mean_frontier_width,
        "frontier_executed_task_count": len(task_metrics),
        "frontier_executed_parallel_task_count": int(executed_parallel_task_count),
        "executed_parallel_task_count": int(executed_parallel_task_count),
        "scheduler_overhead_s": float(scheduler_overhead_s),
        "duplicate_contraction_check": "passed",
        "missing_dependency_check": "passed",
    }


def frontier_waves(graph: TaskGraph) -> tuple[tuple[ContractionTask, ...], ...]:
    task_by_id = {task.id: task for task in graph.tasks}
    remaining = set(task_by_id)
    completed: set[str] = set()
    waves: list[tuple[ContractionTask, ...]] = []
    while remaining:
        ready_ids = sorted(
            task_id
            for task_id in remaining
            if all(dependency in completed for dependency in task_by_id[task_id].dependencies)
        )
        if not ready_ids:
            unresolved = {task_id: task_by_id[task_id].dependencies for task_id in sorted(remaining)}
            raise ValueError(f"TaskGraph dependencies are cyclic or unresolved: {unresolved}")
        waves.append(tuple(task_by_id[task_id] for task_id in ready_ids))
        completed.update(ready_ids)
        remaining.difference_update(ready_ids)
    return tuple(waves)


def _execute_wave_threaded(
    wave: tuple[ContractionTask, ...],
    task_inputs: dict[str, tuple[np.ndarray, np.ndarray]],
    frontier_worker_count: int,
) -> list[tuple[ContractionTask, np.ndarray, float]]:
    results: list[tuple[ContractionTask, np.ndarray, float]] = []
    with ThreadPoolExecutor(max_workers=min(frontier_worker_count, len(wave))) as executor:
        futures = {executor.submit(_execute_frontier_task, task, task_inputs[task.id]): task for task in wave}
        for future in as_completed(futures):
            results.append(future.result())
    return results


def _execute_frontier_task(
    task: ContractionTask,
    task_inputs: tuple[np.ndarray, np.ndarray],
) -> tuple[ContractionTask, np.ndarray, float]:
    task_start = time.perf_counter()
    intermediate = np.einsum(task.index_expression, task_inputs[0], task_inputs[1], optimize=False)
    task_time_s = time.perf_counter() - task_start
    return task, np.asarray(intermediate, dtype=np.complex128), task_time_s


def execute_empty_task_graph(graph: TaskGraph, network: TensorNetworkValue) -> tuple[np.ndarray, str, tuple[int, ...], bool]:
    if len(network.tensors) != 1:
        raise ValueError(
            f"Cannot execute empty TaskGraph with {len(network.tensors)} original tensors; "
            "expected exactly one tensor"
        )
    tensor = network.tensors[0]
    output, transposed = order_final_tensor(
        np.asarray(tensor.array, dtype=np.complex128),
        tensor.spec.labels,
        graph.network.output_labels,
    )
    return output, tensor.spec.id, tensor.spec.labels, transposed


def order_final_tensor(array: np.ndarray, actual_labels: tuple[int, ...], output_labels: tuple[int, ...]) -> tuple[np.ndarray, bool]:
    if actual_labels == output_labels:
        return np.asarray(array, dtype=np.complex128), False
    if len(actual_labels) != len(output_labels) or set(actual_labels) != set(output_labels):
        raise ValueError(f"Final tensor labels {actual_labels} do not match requested output labels {output_labels}")
    axes = tuple(actual_labels.index(label) for label in output_labels)
    return np.asarray(np.transpose(array, axes), dtype=np.complex128), True


def remaining_input_uses(graph: TaskGraph) -> dict[str, int]:
    counts: dict[str, int] = {}
    for task in graph.tasks:
        for tensor_id in task.input_tensor_ids:
            counts[tensor_id] = counts.get(tensor_id, 0) + 1
    return counts


def release_dead_inputs(
    input_tensor_ids: tuple[str, ...],
    output_tensor_id: str,
    final_tensor_id: str | None,
    tensors: dict[str, np.ndarray],
    labels: dict[str, tuple[int, ...]],
    live_ids: set[str],
    remaining_uses: dict[str, int],
) -> None:
    for input_id in input_tensor_ids:
        remaining_uses[input_id] = remaining_uses.get(input_id, 0) - 1
        if remaining_uses[input_id] <= 0 and input_id != final_tensor_id and input_id != output_tensor_id:
            live_ids.discard(input_id)
            tensors.pop(input_id, None)
            labels.pop(input_id, None)


def live_tensor_bytes(tensors: dict[str, np.ndarray], live_ids: set[str]) -> int:
    return _live_tensor_bytes(tensors, live_ids)


def _live_tensor_bytes(tensors: dict[str, np.ndarray], live_ids: set[str]) -> int:
    return int(sum(tensors[tensor_id].nbytes for tensor_id in live_ids if tensor_id in tensors))
