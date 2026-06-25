from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from quantum_bench.core.indices import index_symbols, shape_product
from quantum_bench.core.records import ContractionTask, PathSummary, TaskGraph, TensorSpec
from quantum_bench.tn.network import TensorNetworkValue
from quantum_bench.tn.planners import OptEinsumPlanner, PathPlanner, PlannerResult, planner_from_config


DEFAULT_TARGET_ESTIMATE_KEY = "upmem_dense_int8"


def plan_task_graph(network: TensorNetworkValue, optimize: str = "greedy") -> TaskGraph:
    return plan_task_graph_with_planner(network, OptEinsumPlanner(optimize=optimize))


def plan_task_graph_with_config(network: TensorNetworkValue, planner_config: dict[str, Any] | None) -> TaskGraph:
    return plan_task_graph_with_planner(network, planner_from_config(planner_config))


def plan_task_graph_with_planner(network: TensorNetworkValue, planner: PathPlanner) -> TaskGraph:
    planner_result = planner.plan(network)
    active = list(network.spec.tensors)
    produced_by: dict[str, str | None] = {tensor.id: tensor.produced_by for tensor in active}
    symbols = index_symbols([tensor.labels for tensor in active], network.spec.output_labels)
    tasks: list[ContractionTask] = []

    for step_index, contraction in enumerate(planner_result.path):
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

    summary = _base_path_summary(planner_result)
    graph = TaskGraph(
        network=network.spec,
        tasks=tuple(tasks),
        path=planner_result.path,
        path_summary=summary,
        planning_time_s=planner_result.planning_time_s,
    )
    return with_path_cost_summary(graph)


def with_path_cost_summary(graph: TaskGraph, target_estimate_key: str = DEFAULT_TARGET_ESTIMATE_KEY) -> TaskGraph:
    costs = derive_path_costs(graph, target_estimate_key)
    return replace(graph, path_summary=replace(graph.path_summary, **costs))


def derive_path_costs(graph: TaskGraph, target_estimate_key: str = DEFAULT_TARGET_ESTIMATE_KEY) -> dict[str, int]:
    max_intermediate_bytes = max((_output_bytes(task) for task in graph.tasks), default=0)
    opt_einsum_peak_bytes = (
        graph.path_summary.largest_intermediate * np.dtype(np.complex128).itemsize
        if graph.path_summary.largest_intermediate is not None
        else max_intermediate_bytes
    )
    peak_intermediate_bytes = max(int(opt_einsum_peak_bytes), max_intermediate_bytes)
    total_host_to_dpu_bytes = 0
    total_dpu_to_host_bytes = 0
    total_mram_to_wram_bytes = 0
    unsupported_task_count = 0
    tiling_required_task_count = 0
    missing_target_estimate_count = 0
    estimated_total_tile_count = 0
    estimated_max_parallel_tiles = 0

    for task in graph.tasks:
        estimate = task.target_estimates.get(target_estimate_key)
        if estimate is None:
            missing_target_estimate_count += 1
            continue
        total_host_to_dpu_bytes += int(estimate.get("host_to_dpu_bytes", 0) or 0)
        total_dpu_to_host_bytes += int(estimate.get("dpu_to_host_bytes", 0) or 0)
        total_mram_to_wram_bytes += int(estimate.get("mram_to_wram_bytes", 0) or 0)
        unsupported_task_count += 0 if estimate.get("supported", False) else 1
        tiling_required_task_count += 1 if estimate.get("requires_tiling", False) else 0
        estimated_total_tile_count += int(estimate.get("estimated_tile_count", 0) or 0)
        estimated_max_parallel_tiles = max(estimated_max_parallel_tiles, int(estimate.get("estimated_parallel_tiles", 0) or 0))

    return {
        "task_count": len(graph.tasks),
        "total_estimated_flops": sum(task.estimated_flops for task in graph.tasks),
        "peak_intermediate_bytes": int(peak_intermediate_bytes),
        "max_intermediate_bytes": int(max_intermediate_bytes),
        "total_host_to_dpu_bytes": total_host_to_dpu_bytes,
        "total_dpu_to_host_bytes": total_dpu_to_host_bytes,
        "total_mram_to_wram_bytes": total_mram_to_wram_bytes,
        "unsupported_task_count": unsupported_task_count,
        "tiling_required_task_count": tiling_required_task_count,
        "missing_target_estimate_count": missing_target_estimate_count,
        "estimated_total_tile_count": estimated_total_tile_count,
        "estimated_max_parallel_tiles": estimated_max_parallel_tiles,
    }


def _base_path_summary(planner_result: PlannerResult) -> PathSummary:
    identity = planner_result.identity
    return PathSummary(
        planner=identity.planner_engine,
        optimize=identity.optimize_mode,
        path_length=len(planner_result.path),
        largest_intermediate=planner_result.largest_intermediate,
        naive_flops=planner_result.naive_flops,
        optimized_flops=planner_result.optimized_flops,
        text=planner_result.path_info_text,
        planner_engine=identity.planner_engine,
        planner_id=identity.planner_id,
        planner_kind=identity.planner_kind,
        optimize_mode=identity.optimize_mode,
        objective=identity.objective,
        cost_basis=identity.cost_basis,
        target_estimate_key=identity.target_estimate_key,
        options=identity.options,
    )


def _label_dim(label: int, left: TensorSpec, right: TensorSpec) -> int:
    if label in left.labels:
        return left.shape[left.labels.index(label)]
    return right.shape[right.labels.index(label)]


def _output_bytes(task: ContractionTask) -> int:
    return int(np.prod(task.output_shape, dtype=np.int64) * np.dtype(np.complex128).itemsize)
