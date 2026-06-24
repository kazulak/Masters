from __future__ import annotations

import time

import numpy as np
import opt_einsum as oe

from quantum_bench.core.indices import index_symbols, shape_product
from quantum_bench.core.records import ContractionTask, PathSummary, TaskGraph, TensorSpec
from quantum_bench.tn.network import TensorNetworkValue


def plan_task_graph(network: TensorNetworkValue, optimize: str = "greedy") -> TaskGraph:
    start = time.perf_counter()
    arrays = [tensor.array for tensor in network.tensors]
    path, path_info = oe.contract_path(network.spec.einsum_expression, *arrays, optimize=optimize)
    active = list(network.spec.tensors)
    produced_by: dict[str, str | None] = {tensor.id: tensor.produced_by for tensor in active}
    symbols = index_symbols([tensor.labels for tensor in active], network.spec.output_labels)
    tasks: list[ContractionTask] = []

    for step_index, contraction in enumerate(path):
        if len(contraction) != 2:
            raise ValueError(f"Only pairwise contraction paths are supported; got {contraction}")
        i, j = sorted(contraction)
        left = active[i]
        right = active[j]
        contracted = tuple(label for label in left.labels if label in set(right.labels) and label not in network.spec.output_labels)
        free_left = tuple(label for label in left.labels if label not in contracted)
        free_right = tuple(label for label in right.labels if label not in contracted)
        output_labels = free_left + free_right
        output_shape = tuple(_label_dim(label, left, right) for label in output_labels)
        expression = (
            "".join(symbols[label] for label in left.labels)
            + ","
            + "".join(symbols[label] for label in right.labels)
            + "->"
            + "".join(symbols[label] for label in output_labels)
        )
        m = shape_product(tuple(_label_dim(label, left, right) for label in free_left))
        k = shape_product(tuple(_label_dim(label, left, right) for label in contracted))
        n = shape_product(tuple(_label_dim(label, left, right) for label in free_right))
        output_bytes = int(np.prod(output_shape, dtype=np.int64) * np.dtype(np.complex128).itemsize)
        task_id = f"task_{step_index}"
        output_id = f"result_{step_index}"
        dependencies = tuple(
            dep for dep in (produced_by.get(left.id), produced_by.get(right.id)) if dep and dep.startswith("task_")
        )
        tasks.append(
            ContractionTask(
                id=task_id,
                input_tensor_ids=(left.id, right.id),
                output_tensor_id=output_id,
                dependencies=dependencies,
                index_expression=expression,
                input_shapes=(left.shape, right.shape),
                output_shape=output_shape,
                left_labels=left.labels,
                right_labels=right.labels,
                contracted_labels=contracted,
                output_labels=output_labels,
                gemm_m=m,
                gemm_k=k,
                gemm_n=n,
                structure="dense",
                estimated_flops=int(8 * m * k * n),
                estimated_bytes=int(np.prod(left.shape) * 16 + np.prod(right.shape) * 16 + output_bytes),
            )
        )
        output_tensor = TensorSpec(output_id, output_labels, output_shape, "dense", produced_by=task_id)
        produced_by[output_id] = task_id
        active.pop(j)
        active.pop(i)
        active.insert(i, output_tensor)

    summary = PathSummary(
        planner="opt_einsum",
        optimize=optimize,
        path_length=len(path),
        largest_intermediate=_safe_int(getattr(path_info, "largest_intermediate", None)),
        naive_flops=_safe_float(getattr(path_info, "naive_cost", None)),
        optimized_flops=_safe_float(getattr(path_info, "opt_cost", None)),
        text=str(path_info),
    )
    return TaskGraph(
        network=network.spec,
        tasks=tuple(tasks),
        path=tuple(tuple(int(item) for item in step) for step in path),
        path_summary=summary,
        planning_time_s=time.perf_counter() - start,
    )


def _label_dim(label: int, left: TensorSpec, right: TensorSpec) -> int:
    if label in left.labels:
        return left.shape[left.labels.index(label)]
    return right.shape[right.labels.index(label)]


def _safe_int(value: object) -> int | None:
    try:
        return None if value is None else int(value)
    except (TypeError, ValueError):
        return None


def _safe_float(value: object) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None
