from __future__ import annotations

import time
from typing import Any

import numpy as np

from quantum_bench.core.records import (
    BenchmarkContext,
    ExecutionProfile,
    RouteCapabilities,
    RouteEstimate,
    RouteIdentity,
    RouteOutput,
    RouteProbe,
    RouteResult,
    TaskExecutionMetric,
    TaskGraph,
)
from quantum_bench.environment import read_rapl_uj
from quantum_bench.formats.fixed_point import FixedPointSpec, conversion_error_metrics, dequantize_fixed_point, quantize_fixed_point
from quantum_bench.tn import plan_task_graph_with_config, with_path_cost_summary
from quantum_bench.tn.execution import execute_empty_task_graph, order_final_tensor, release_dead_inputs, remaining_input_uses
from quantum_bench.tn.network import TensorNetworkValue


class CpuTnPathReplayFloat64Route:
    name = "cpu_tn_path_replay_float64"
    backend_family = "cpu"
    identity = RouteIdentity(
        route_id=name,
        display_name="CPU TN path replay float64/complex128",
        role="diagnostic_path_replay_baseline",
        simulation_method="exact_tensor_network",
        kernel_family="einsum_contraction",
        hardware_target="cpu",
        execution_mode="in_process_python_path_replay",
        output_contract="final_tensor",
        validation_mode="compare_output",
    )

    def probe(self) -> RouteProbe:
        return RouteProbe(self.name, True, metadata={"numpy": np.__version__})

    def capabilities(self) -> RouteCapabilities:
        return RouteCapabilities(
            identity=self.identity,
            supported_workload_families=("builtin", "qasm_file", "quest_compatible"),
            can_return_output=True,
            can_measure_energy=True,
            metadata={
                "numpy": np.__version__,
                "path_replay_execution": True,
                "per_contraction_quantization": False,
                "exact_output_comparable": True,
            },
        )

    def can_execute(self, graph: TaskGraph, context: BenchmarkContext) -> tuple[bool, str | None]:
        return True, None

    def estimate(self, graph: TaskGraph, context: BenchmarkContext) -> RouteEstimate:
        return RouteEstimate(
            self.name,
            sum(task.estimated_flops for task in graph.tasks),
            sum(task.estimated_bytes for task in graph.tasks),
            graph.path_summary.largest_intermediate * 16 if graph.path_summary.largest_intermediate else None,
            metadata={"path_replay_execution": True, "quantization_mode": "none"},
        )

    def prepare(self, graph: TaskGraph, network: TensorNetworkValue, context: BenchmarkContext) -> dict:
        replay_graph = _route_graph(graph, network, context)
        return {"graph": replay_graph, "network": network, "prepare_s": 0.0, "planner": _route_planner(context, graph)}

    def execute(self, prepared: object, context: BenchmarkContext) -> RouteResult:
        payload = dict(prepared)  # type: ignore[arg-type]
        graph: TaskGraph = payload["graph"]
        network: TensorNetworkValue = payload["network"]
        planner = dict(payload.get("planner") or {})
        energy_start = read_rapl_uj()
        start = time.perf_counter()
        output, metadata = _execute_path_replay(graph, network, quantized=False)
        kernel_s = time.perf_counter() - start
        energy_end = read_rapl_uj()
        energy_joules, energy_source = _energy(energy_start, energy_end)
        array = np.asarray(output, dtype=np.complex128)
        return RouteResult(
            route=self.name,
            backend_family=self.backend_family,
            status="passed",
            output=RouteOutput(
                contract=self.identity.output_contract,
                array=array,
                shape=tuple(int(dim) for dim in array.shape),
                dtype=str(array.dtype),
            ),
            profile=ExecutionProfile(prepare_s=float(payload.get("prepare_s", 0.0)), kernel_s=kernel_s, total_s=kernel_s),
            energy_joules=energy_joules,
            energy_source=energy_source,
            metadata={
                **metadata,
                **_path_metadata(graph, planner),
                "quantization_mode": "none",
                "per_contraction_quantization": False,
                "input_dtype": "complex128",
                "accumulator_dtype": "complex128",
                "scaling_applied": False,
                "quantized_replay_numeric_contract": "not_applicable",
                "total_quantization_time_s": 0.0,
                "total_dequantization_time_s": 0.0,
            },
        )


class CpuTnPathReplayInt8QuantizedRoute(CpuTnPathReplayFloat64Route):
    name = "cpu_tn_path_replay_int8_quantized"
    identity = RouteIdentity(
        route_id=name,
        display_name="CPU TN per-contraction int8 quantized replay",
        role="diagnostic_quantized_path_replay",
        simulation_method="exact_tensor_network",
        kernel_family="einsum_contraction",
        hardware_target="cpu",
        execution_mode="in_process_python_quantized_path_replay",
        output_contract="final_tensor",
        validation_mode="compare_output",
    )

    def capabilities(self) -> RouteCapabilities:
        return RouteCapabilities(
            identity=self.identity,
            supported_workload_families=("builtin", "qasm_file", "quest_compatible"),
            can_return_output=True,
            can_measure_energy=True,
            metadata={
                "numpy": np.__version__,
                "path_replay_execution": True,
                "per_contraction_quantization": True,
                "quantization_mode": "per_contraction_input_quantize",
                "exact_output_comparable": True,
            },
        )

    def estimate(self, graph: TaskGraph, context: BenchmarkContext) -> RouteEstimate:
        return RouteEstimate(
            self.name,
            sum(task.estimated_flops for task in graph.tasks),
            sum(task.estimated_bytes for task in graph.tasks),
            graph.path_summary.largest_intermediate * 16 if graph.path_summary.largest_intermediate else None,
            metadata={"path_replay_execution": True, "quantization_mode": "per_contraction_input_quantize"},
        )

    def execute(self, prepared: object, context: BenchmarkContext) -> RouteResult:
        payload = dict(prepared)  # type: ignore[arg-type]
        graph: TaskGraph = payload["graph"]
        network: TensorNetworkValue = payload["network"]
        planner = dict(payload.get("planner") or {})
        energy_start = read_rapl_uj()
        start = time.perf_counter()
        output, metadata = _execute_path_replay(graph, network, quantized=True)
        kernel_s = time.perf_counter() - start
        energy_end = read_rapl_uj()
        energy_joules, energy_source = _energy(energy_start, energy_end)
        array = np.asarray(output, dtype=np.complex128)
        return RouteResult(
            route=self.name,
            backend_family=self.backend_family,
            status="passed",
            output=RouteOutput(
                contract=self.identity.output_contract,
                array=array,
                shape=tuple(int(dim) for dim in array.shape),
                dtype=str(array.dtype),
            ),
            profile=ExecutionProfile(
                prepare_s=float(payload.get("prepare_s", 0.0)),
                kernel_s=kernel_s,
                total_s=kernel_s,
            ),
            energy_joules=energy_joules,
            energy_source=energy_source,
            metadata={
                **metadata,
                **_path_metadata(graph, planner),
                "quantization_mode": "per_contraction_input_quantize",
                "per_contraction_quantization": True,
                "input_dtype": "int8_split_real_imag",
                "accumulator_dtype": "complex128",
                "scaling_applied": True,
                "quantized_replay_numeric_contract": "int8_operand_quantize_dequantize_then_complex128_einsum",
            },
        )


def _route_graph(graph: TaskGraph, network: TensorNetworkValue, context: BenchmarkContext) -> TaskGraph:
    planner = _route_planner(context, graph)
    if not planner:
        return graph
    return with_path_cost_summary(plan_task_graph_with_config(network, planner))


def _route_planner(context: BenchmarkContext, graph: TaskGraph) -> dict[str, Any]:
    options = dict(context.route_config.get("options") or {})
    planner = options.get("planner")
    if isinstance(planner, dict):
        return dict(planner)
    path_strategy = options.get("path_strategy") or options.get("optimize")
    if path_strategy:
        return {"engine": "opt_einsum", "optimize": str(path_strategy)}
    return {
        "engine": graph.path_summary.planner_engine or "opt_einsum",
        "optimize": graph.path_summary.optimize_mode or graph.path_summary.optimize or "greedy",
    }


def _execute_path_replay(graph: TaskGraph, network: TensorNetworkValue, *, quantized: bool) -> tuple[np.ndarray, dict]:
    tensors = {tensor.spec.id: np.asarray(tensor.array, dtype=np.complex128) for tensor in network.tensors}
    labels = {tensor.spec.id: tensor.spec.labels for tensor in network.tensors}
    live_ids = set(tensors)
    task_metrics: list[TaskExecutionMetric] = []
    peak_live_bytes = _live_tensor_bytes(tensors, live_ids)
    max_intermediate_bytes = 0
    total_quantization_time_s = 0.0
    total_dequantization_time_s = 0.0
    source_bytes = 0
    converted_bytes = 0
    clipping_count = 0
    saturation_count = 0
    quantization_max_abs_error = 0.0
    quantization_l2_error = 0.0

    if not graph.tasks:
        output, final_id, final_labels, transposed = execute_empty_task_graph(graph, network)
        return output, {
            **_base_metadata("cpu_tn_path_replay_np_einsum", graph, (), int(output.nbytes), 0),
            "final_tensor_id": final_id,
            "final_tensor_labels": final_labels,
            "output_labels": graph.network.output_labels,
            "final_transpose_applied": transposed,
            "total_quantization_time_s": 0.0,
            "total_dequantization_time_s": 0.0,
        }

    remaining_uses = remaining_input_uses(graph)
    final_tensor_id = graph.tasks[-1].output_tensor_id
    final_labels: tuple[int, ...] | None = None

    for task in graph.tasks:
        left_id, right_id = task.input_tensor_ids
        if left_id not in tensors or right_id not in tensors:
            missing = [tensor_id for tensor_id in task.input_tensor_ids if tensor_id not in tensors]
            raise ValueError(f"Task {task.id} references unavailable tensor(s): {', '.join(missing)}")

        left = np.asarray(tensors[left_id], dtype=np.complex128)
        right = np.asarray(tensors[right_id], dtype=np.complex128)
        if quantized:
            left, right, qmeta = _quantized_operands(left, right)
            total_quantization_time_s += float(qmeta["quantization_time_s"])
            total_dequantization_time_s += float(qmeta["dequantization_time_s"])
            source_bytes += int(qmeta["source_bytes"])
            converted_bytes += int(qmeta["converted_bytes"])
            clipping_count += int(qmeta["clipping_count"])
            saturation_count += int(qmeta["saturation_count"])
            quantization_max_abs_error = max(quantization_max_abs_error, float(qmeta["max_abs_error"]))
            quantization_l2_error += float(qmeta["l2_error"])

        task_start = time.perf_counter()
        intermediate = np.einsum(task.index_expression, left, right, optimize=False)
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
        raise ValueError(f"Path replay did not produce final tensor {final_tensor_id}")

    output, transposed = order_final_tensor(tensors[final_tensor_id], final_labels, graph.network.output_labels)
    metadata = {
        **_base_metadata(
            "cpu_tn_path_replay_quantized_np_einsum" if quantized else "cpu_tn_path_replay_np_einsum",
            graph,
            task_metrics,
            peak_live_bytes,
            max_intermediate_bytes,
        ),
        "final_tensor_id": final_tensor_id,
        "final_tensor_labels": final_labels,
        "output_labels": graph.network.output_labels,
        "final_transpose_applied": transposed,
        "total_quantization_time_s": float(total_quantization_time_s),
        "total_dequantization_time_s": float(total_dequantization_time_s),
        "quantization_source_bytes": int(source_bytes),
        "quantization_converted_bytes": int(converted_bytes),
        "quantization_transfer_reduction_ratio": float(source_bytes / converted_bytes) if converted_bytes else None,
        "quantization_clipping_count": int(clipping_count),
        "quantization_saturation_count": int(saturation_count),
        "quantization_max_abs_error": float(quantization_max_abs_error),
        "quantization_l2_error": float(quantization_l2_error),
    }
    return output, metadata


def _quantized_operands(left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, float | int]]:
    spec = FixedPointSpec(route_dtype="int8", complex_policy="split_real_imag_last_axis")
    q_start = time.perf_counter()
    left_quantized = quantize_fixed_point(left, spec)
    right_quantized = quantize_fixed_point(right, spec)
    quantization_time_s = time.perf_counter() - q_start

    dq_start = time.perf_counter()
    left_dequantized = dequantize_fixed_point(left_quantized, dtype=np.complex128)
    right_dequantized = dequantize_fixed_point(right_quantized, dtype=np.complex128)
    dequantization_time_s = time.perf_counter() - dq_start

    left_error = conversion_error_metrics(left, left_dequantized)
    right_error = conversion_error_metrics(right, right_dequantized)
    return left_dequantized, right_dequantized, {
        "quantization_time_s": float(quantization_time_s),
        "dequantization_time_s": float(dequantization_time_s),
        "source_bytes": int(left_quantized.record.source_bytes + right_quantized.record.source_bytes),
        "converted_bytes": int(left_quantized.record.converted_bytes + right_quantized.record.converted_bytes),
        "clipping_count": int(left_quantized.record.clipping_count + right_quantized.record.clipping_count),
        "saturation_count": int(left_quantized.record.saturation_count + right_quantized.record.saturation_count),
        "max_abs_error": max(float(left_error.max_abs_error), float(right_error.max_abs_error)),
        "l2_error": float(left_error.l2_error + right_error.l2_error),
    }


def _base_metadata(
    execution_engine: str,
    graph: TaskGraph,
    task_metrics: tuple[TaskExecutionMetric, ...] | list[TaskExecutionMetric],
    peak_live_bytes: int,
    max_intermediate_bytes: int,
) -> dict[str, Any]:
    return {
        "execution_engine": execution_engine,
        "task_count": len(task_metrics),
        "task_metrics": task_metrics,
        "peak_intermediate_bytes": int(peak_live_bytes),
        "max_intermediate_tensor_bytes": int(max_intermediate_bytes),
        "parallelism_mode": "sequential",
        "parallelism_evidence_type": "executed",
        "execution_plan_kind": "sequential_taskgraph_path_replay",
        "execution_plan_executed": True,
        "path_replay_execution": True,
        "path_replay_task_count": len(graph.tasks),
        "frontier_scheduler_enabled": False,
        "slicing_enabled": False,
        "modeled_parallelism_available": False,
        "duplicate_contraction_check": "passed",
        "missing_dependency_check": "passed",
    }


def _path_metadata(graph: TaskGraph, planner: dict[str, Any]) -> dict[str, Any]:
    return {
        "path_strategy": str(planner.get("optimize") or graph.path_summary.optimize or "unknown"),
        "path_planner_engine": str(planner.get("engine") or graph.path_summary.planner_engine or "unknown"),
        "path_replay_planner": planner,
        "path_length": graph.path_summary.path_length,
        "path_largest_intermediate": graph.path_summary.largest_intermediate,
        "path_total_estimated_flops": graph.path_summary.total_estimated_flops,
        "path_max_intermediate_bytes": graph.path_summary.max_intermediate_bytes,
    }


def _energy(start: int | None, end: int | None) -> tuple[float | None, str]:
    if start is not None and end is not None and end >= start:
        energy = (end - start) / 1_000_000.0
        return energy, "rapl_measured" if energy > 0 else "rapl_zero_or_too_short"
    return None, "unavailable"


def _live_tensor_bytes(tensors: dict[str, np.ndarray], live_ids: set[str]) -> int:
    return int(sum(tensors[tensor_id].nbytes for tensor_id in live_ids if tensor_id in tensors))
