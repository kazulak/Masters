from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import numpy as np

from quantum_bench.core.records import ContractionTask, TaskExecutionMetric, TaskGraph
from quantum_bench.tn.contract import contract_binary_task
from quantum_bench.tn.execution import live_tensor_bytes, order_final_tensor
from quantum_bench.tn.network import TensorNetworkValue
from quantum_bench.tn.slicing import SliceAwareTaskGraphModel, SliceModelTask, build_slice_aware_taskgraph_model, validate_slice_aware_taskgraph_model


@dataclass(frozen=True)
class _ExecutionNode:
    id: str
    kind: str
    dependencies: tuple[str, ...]
    source_task: ContractionTask
    slice_task: SliceModelTask | None = None
    reconstruction_input_tensor_ids: tuple[str, ...] = ()


def execute_task_sliced_sequence_np_einsum(
    graph: TaskGraph,
    network: TensorNetworkValue,
    *,
    max_slice_count: int = 2,
) -> tuple[np.ndarray, dict]:
    model = build_slice_aware_taskgraph_model(graph, max_slice_count=max_slice_count)
    _require_valid_model(model)
    start = time.perf_counter()
    output, metadata = _execute_slice_aware_graph(
        graph,
        network,
        model,
        frontier_worker_count=1,
        use_frontier_scheduler=False,
    )
    total_s = time.perf_counter() - start
    metadata.update(
        {
            "execution_engine": "task_sliced_sequence_np_einsum",
            "parallelism_mode": "slicing",
            "parallelism_evidence_type": "executed",
            "execution_plan_kind": "internal_slice_aware_taskgraph_sequence",
            "execution_plan_executed": True,
            "frontier_scheduler_enabled": False,
            "frontier_parallel_execution": False,
            "frontier_worker_count": 1,
            "frontier_wave_count": None,
            "max_frontier_width": None,
            "mean_frontier_width": None,
            "frontier_executed_task_count": None,
            "frontier_executed_parallel_task_count": 0,
            "executed_parallel_task_count": 0,
            "scheduler_overhead_s": 0.0,
            "slice_task_execution_mode": "sequential",
            "slice_parallel_execution": False,
            "slice_worker_count": 1,
            "hybrid_ready": False,
            "total_slice_execution_time_s": total_s,
        }
    )
    return output, metadata


def execute_task_hybrid_slice_frontier_np_einsum(
    graph: TaskGraph,
    network: TensorNetworkValue,
    *,
    frontier_worker_count: int = 1,
    max_slice_count: int = 2,
) -> tuple[np.ndarray, dict]:
    if frontier_worker_count < 1:
        raise ValueError("frontier_worker_count must be >= 1")
    model = build_slice_aware_taskgraph_model(graph, max_slice_count=max_slice_count)
    _require_valid_model(model)
    start = time.perf_counter()
    output, metadata = _execute_slice_aware_graph(
        graph,
        network,
        model,
        frontier_worker_count=frontier_worker_count,
        use_frontier_scheduler=True,
    )
    total_s = time.perf_counter() - start
    metadata.update(
        {
            "execution_engine": "task_hybrid_slice_frontier_np_einsum",
            "parallelism_mode": "hybrid",
            "parallelism_evidence_type": "executed",
            "hybrid_components": ["slicing", "frontier"],
            "execution_plan_kind": "internal_slice_aware_taskgraph_frontier_scheduler",
            "execution_plan_executed": True,
            "frontier_scheduler_enabled": True,
            "slice_task_execution_mode": "frontier_scheduled",
            "hybrid_ready": True,
            "total_slice_execution_time_s": total_s,
        }
    )
    return output, metadata


def _execute_slice_aware_graph(
    graph: TaskGraph,
    network: TensorNetworkValue,
    model: SliceAwareTaskGraphModel,
    *,
    frontier_worker_count: int,
    use_frontier_scheduler: bool,
) -> tuple[np.ndarray, dict]:
    tensors = {tensor.spec.id: np.asarray(tensor.array, dtype=np.complex128) for tensor in network.tensors}
    labels = {tensor.spec.id: tensor.spec.labels for tensor in network.tensors}
    live_ids = set(tensors)
    task_by_id = {task.id: task for task in graph.tasks}
    nodes = _expanded_execution_nodes(graph, model)
    waves = _node_frontier_waves(nodes) if use_frontier_scheduler else tuple((node,) for node in nodes)
    task_metrics_by_id: dict[str, TaskExecutionMetric] = {}
    executed_node_ids: list[str] = []
    executed_node_set: set[str] = set()
    completed_source_ids: list[str] = []
    completed_source_set: set[str] = set()
    executed_slice_task_ids: list[str] = []
    peak_live_bytes = live_tensor_bytes(tensors, live_ids)
    max_intermediate_bytes = 0
    scheduler_overhead_s = 0.0
    frontier_parallel_execution = use_frontier_scheduler and frontier_worker_count > 1 and any(len(wave) > 1 for wave in waves)
    executed_parallel_node_count = 0
    slice_parallel_wave_count = 0
    slice_parallel_execution = False
    reconstruction_validation_status = "not_run"
    reconstruction_max_abs_error: float | None = None
    dependency_violation_detected = False

    for wave in waves:
        scheduler_start = time.perf_counter()
        for node in wave:
            if node.id in executed_node_set:
                raise ValueError(f"Node {node.id} scheduled more than once")
            if not all(dependency in executed_node_set for dependency in node.dependencies):
                dependency_violation_detected = True
                missing = [dependency for dependency in node.dependencies if dependency not in executed_node_set]
                raise ValueError(f"Node {node.id} scheduled before dependency node(s) complete: {', '.join(missing)}")
            missing_inputs = _missing_inputs(node, tensors)
            if missing_inputs:
                dependency_violation_detected = True
                raise ValueError(f"Node {node.id} scheduled before tensor(s) available: {', '.join(missing_inputs)}")
        scheduler_overhead_s += time.perf_counter() - scheduler_start

        if use_frontier_scheduler and frontier_worker_count > 1 and len(wave) > 1:
            results = _execute_wave_threaded(wave, tensors, frontier_worker_count)
            executed_parallel_node_count += len(wave)
            if any(node.kind == "slice" for node in wave):
                slice_parallel_wave_count += 1
                slice_parallel_execution = True
        else:
            results = [_execute_node(node, tensors) for node in wave]

        scheduler_start = time.perf_counter()
        for node, output, elapsed_s in sorted(results, key=lambda item: item[0].id):
            if node.id in executed_node_set:
                raise ValueError(f"Node {node.id} completed more than once")
            output = np.asarray(output, dtype=np.complex128)
            tensors[_node_output_tensor_id(node)] = output
            labels[_node_output_tensor_id(node)] = node.source_task.output_labels
            live_ids.add(_node_output_tensor_id(node))
            max_intermediate_bytes = max(max_intermediate_bytes, int(output.nbytes))
            peak_live_bytes = max(peak_live_bytes, live_tensor_bytes(tensors, live_ids))
            executed_node_ids.append(node.id)
            executed_node_set.add(node.id)

            if node.kind == "slice":
                executed_slice_task_ids.append(node.id)
            elif node.kind == "reconstruct":
                direct = _execute_direct_task(node.source_task, tensors)
                error = float(np.max(np.abs(output - direct))) if output.size else 0.0
                reconstruction_max_abs_error = error
                reconstruction_validation_status = "passed" if np.allclose(output, direct, atol=1.0e-12, rtol=1.0e-12) else "failed"
                _record_source_completion(node.source_task.id, completed_source_ids, completed_source_set)
                task_metrics_by_id[node.source_task.id] = _metric(node.source_task, elapsed_s, output)
            else:
                _record_source_completion(node.source_task.id, completed_source_ids, completed_source_set)
                task_metrics_by_id[node.source_task.id] = _metric(node.source_task, elapsed_s, output)
        scheduler_overhead_s += time.perf_counter() - scheduler_start

    expected_source_ids = tuple(task.id for task in graph.tasks)
    if set(completed_source_ids) != set(expected_source_ids):
        missing = sorted(set(expected_source_ids) - completed_source_set)
        extras = sorted(completed_source_set - set(expected_source_ids))
        raise ValueError(f"Slice-aware execution source task mismatch; missing={missing}, extras={extras}")
    if len(completed_source_ids) != len(completed_source_set):
        raise ValueError("Slice-aware execution detected duplicate source task completion")
    if set(executed_slice_task_ids) != {task.id for task in model.slice_tasks}:
        raise ValueError("Slice-aware execution did not execute every slice task exactly once")
    if reconstruction_validation_status != "passed":
        raise ValueError("Slice-aware reconstruction validation failed")

    final_tensor_id = graph.tasks[-1].output_tensor_id
    if final_tensor_id not in tensors:
        raise ValueError(f"Slice-aware execution did not produce final tensor {final_tensor_id}")
    final_labels = task_by_id[graph.tasks[-1].id].output_labels
    output, transposed = order_final_tensor(tensors[final_tensor_id], final_labels, graph.network.output_labels)
    task_metrics = [task_metrics_by_id[task.id] for task in graph.tasks]
    frontier_widths = tuple(len(wave) for wave in waves)
    max_frontier_width = max(frontier_widths, default=0)
    mean_frontier_width = (sum(frontier_widths) / len(frontier_widths)) if frontier_widths else 0.0
    metadata = {
        **model.to_metadata(),
        "slicing_backend": "internal_taskgraph",
        "slicing_enabled": True,
        "slicing_strategy": "single_task_single_index",
        "slice_count": model.slice_model_slice_count,
        "sliced_index_sizes": {str(label): model.slice_model_slice_count for label in model.sliced_indices},
        "slice_model_execution_status": "executed",
        "slice_model_executed_task_count": len(executed_slice_task_ids),
        "slice_reconstruction_status": "completed",
        "slicing_reconstruction_status": "completed",
        "slice_parallel_execution": bool(slice_parallel_execution),
        "slice_worker_count": int(frontier_worker_count),
        "hybrid_reconstruction_validation_status": reconstruction_validation_status,
        "hybrid_reconstruction_max_abs_error": reconstruction_max_abs_error,
        "hybrid_execution_node_count": len(nodes),
        "frontier_parallel_execution": bool(frontier_parallel_execution),
        "frontier_worker_count": int(frontier_worker_count),
        "frontier_wave_count": len(waves) if use_frontier_scheduler else None,
        "frontier_widths": frontier_widths if use_frontier_scheduler else (),
        "max_frontier_width": max_frontier_width if use_frontier_scheduler else None,
        "mean_frontier_width": mean_frontier_width if use_frontier_scheduler else None,
        "frontier_executed_task_count": len(executed_node_ids) if use_frontier_scheduler else None,
        "source_frontier_completed_task_count": len(task_metrics) if use_frontier_scheduler else None,
        "frontier_executed_parallel_task_count": int(executed_parallel_node_count),
        "executed_parallel_task_count": int(executed_parallel_node_count),
        "slice_parallel_wave_count": int(slice_parallel_wave_count),
        "scheduler_overhead_s": float(scheduler_overhead_s if use_frontier_scheduler else 0.0),
        "duplicate_contraction_check": "passed",
        "missing_dependency_check": "passed",
        "dependency_violation_detected": bool(dependency_violation_detected),
        "task_count": len(task_metrics),
        "task_metrics": task_metrics,
        "source_task_count": len(graph.tasks),
        "source_task_completion_count": len(completed_source_ids),
        "executed_source_task_ids": tuple(completed_source_ids),
        "executed_slice_task_ids": tuple(executed_slice_task_ids),
        "peak_intermediate_bytes": int(peak_live_bytes),
        "max_intermediate_tensor_bytes": int(max_intermediate_bytes),
        "final_tensor_id": final_tensor_id,
        "final_tensor_labels": final_labels,
        "output_labels": graph.network.output_labels,
        "final_transpose_applied": transposed,
    }
    return output, metadata


def _expanded_execution_nodes(graph: TaskGraph, model: SliceAwareTaskGraphModel) -> tuple[_ExecutionNode, ...]:
    task_by_id = {task.id: task for task in graph.tasks}
    if model.sliced_task_id is None or model.reconstruction_step is None:
        raise ValueError("Slice-aware model has no selected task or reconstruction step")
    sliced_task = task_by_id[model.sliced_task_id]
    nodes: list[_ExecutionNode] = []
    for task in graph.tasks:
        if task.id == model.sliced_task_id:
            nodes.extend(
                _ExecutionNode(id=slice_task.id, kind="slice", dependencies=slice_task.dependencies, source_task=sliced_task, slice_task=slice_task)
                for slice_task in model.slice_tasks
            )
            nodes.append(
                _ExecutionNode(
                    id=model.reconstruction_step.id,
                    kind="reconstruct",
                    dependencies=model.reconstruction_step.dependencies,
                    source_task=sliced_task,
                    reconstruction_input_tensor_ids=model.reconstruction_step.input_tensor_ids,
                )
            )
        else:
            dependencies = tuple(model.reconstruction_step.id if dep == model.sliced_task_id else dep for dep in task.dependencies)
            nodes.append(_ExecutionNode(id=task.id, kind="normal", dependencies=dependencies, source_task=task))
    return tuple(nodes)


def _node_frontier_waves(nodes: tuple[_ExecutionNode, ...]) -> tuple[tuple[_ExecutionNode, ...], ...]:
    node_by_id = {node.id: node for node in nodes}
    remaining = set(node_by_id)
    completed: set[str] = set()
    waves: list[tuple[_ExecutionNode, ...]] = []
    while remaining:
        ready_ids = sorted(node_id for node_id in remaining if all(dep in completed for dep in node_by_id[node_id].dependencies))
        if not ready_ids:
            unresolved = {node_id: node_by_id[node_id].dependencies for node_id in sorted(remaining)}
            raise ValueError(f"Slice-aware dependencies are cyclic or unresolved: {unresolved}")
        waves.append(tuple(node_by_id[node_id] for node_id in ready_ids))
        completed.update(ready_ids)
        remaining.difference_update(ready_ids)
    return tuple(waves)


def _execute_wave_threaded(
    wave: tuple[_ExecutionNode, ...],
    tensors: dict[str, np.ndarray],
    frontier_worker_count: int,
) -> list[tuple[_ExecutionNode, np.ndarray, float]]:
    snapshot = {tensor_id: np.asarray(tensor, dtype=np.complex128) for tensor_id, tensor in tensors.items()}
    results: list[tuple[_ExecutionNode, np.ndarray, float]] = []
    with ThreadPoolExecutor(max_workers=min(frontier_worker_count, len(wave))) as executor:
        futures = {executor.submit(_execute_node, node, snapshot): node for node in wave}
        for future in as_completed(futures):
            results.append(future.result())
    return results


def _execute_node(node: _ExecutionNode, tensors: dict[str, np.ndarray]) -> tuple[_ExecutionNode, np.ndarray, float]:
    start = time.perf_counter()
    if node.kind == "slice":
        if node.slice_task is None:
            raise ValueError(f"Slice node {node.id} has no slice task")
        output = _execute_slice_node(node.source_task, node.slice_task, tensors)
    elif node.kind == "reconstruct":
        output = _execute_reconstruction(node, tensors)
    else:
        output = _execute_direct_task(node.source_task, tensors)
    return node, np.asarray(output, dtype=np.complex128), time.perf_counter() - start


def _execute_slice_node(task: ContractionTask, slice_task: SliceModelTask, tensors: dict[str, np.ndarray]) -> np.ndarray:
    left_id, right_id = task.input_tensor_ids
    left = _restricted_tensor(tensors[left_id], slice_task, left_id)
    right = _restricted_tensor(tensors[right_id], slice_task, right_id)
    return contract_binary_task(task, left, right)


def _execute_reconstruction(node: _ExecutionNode, tensors: dict[str, np.ndarray]) -> np.ndarray:
    partials = [tensors[tensor_id] for tensor_id in _reconstruction_input_ids(node)]
    if not partials:
        raise ValueError(f"Reconstruction node {node.id} has no partial inputs")
    return np.sum(np.stack(partials, axis=0), axis=0)


def _reconstruction_input_ids(node: _ExecutionNode) -> tuple[str, ...]:
    return node.reconstruction_input_tensor_ids


def _restricted_tensor(tensor: np.ndarray, slice_task: SliceModelTask, tensor_id: str) -> np.ndarray:
    restricted = tensor
    for restriction in slice_task.input_restrictions:
        if restriction.tensor_id != tensor_id:
            continue
        index = [slice(None)] * restricted.ndim
        index[restriction.axis] = slice(restriction.value, restriction.value + 1)
        restricted = restricted[tuple(index)]
    return np.asarray(restricted, dtype=np.complex128)


def _execute_direct_task(task: ContractionTask, tensors: dict[str, np.ndarray]) -> np.ndarray:
    left_id, right_id = task.input_tensor_ids
    return contract_binary_task(task, tensors[left_id], tensors[right_id])


def _node_output_tensor_id(node: _ExecutionNode) -> str:
    if node.kind == "slice":
        if node.slice_task is None:
            raise ValueError(f"Slice node {node.id} has no slice task")
        return node.slice_task.partial_output_tensor_id
    return node.source_task.output_tensor_id


def _missing_inputs(node: _ExecutionNode, tensors: dict[str, np.ndarray]) -> list[str]:
    if node.kind == "reconstruct":
        return [tensor_id for tensor_id in _reconstruction_input_ids(node) if tensor_id not in tensors]
    return [tensor_id for tensor_id in node.source_task.input_tensor_ids if tensor_id not in tensors]


def _metric(task: ContractionTask, execution_time_s: float, output: np.ndarray) -> TaskExecutionMetric:
    return TaskExecutionMetric(
        task_id=task.id,
        input_tensor_ids=task.input_tensor_ids,
        output_tensor_id=task.output_tensor_id,
        input_shapes=task.input_shapes,
        output_shape=task.output_shape,
        contracted_labels=task.contracted_labels,
        estimated_flops=task.estimated_flops,
        estimated_bytes=task.estimated_bytes,
        execution_time_s=float(execution_time_s),
        intermediate_tensor_bytes=int(output.nbytes),
        target_estimates=task.target_estimates,
    )


def _record_source_completion(task_id: str, completed_source_ids: list[str], completed_source_set: set[str]) -> None:
    if task_id in completed_source_set:
        raise ValueError(f"Source task {task_id} completed more than once")
    completed_source_ids.append(task_id)
    completed_source_set.add(task_id)


def _require_valid_model(model: SliceAwareTaskGraphModel) -> None:
    valid, reason = validate_slice_aware_taskgraph_model(model)
    if not valid:
        raise ValueError(reason or "slice_aware_model_unavailable")
