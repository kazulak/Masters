from __future__ import annotations

import time

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
from quantum_bench.tn.network import TensorNetworkValue


class CpuTnEinsumExactRoute:
    name = "cpu_tn_einsum_exact"
    backend_family = "cpu"
    identity = RouteIdentity(
        route_id=name,
        display_name="CPU exact tensor network (NumPy einsum)",
        role="reference",
        simulation_method="exact_tensor_network",
        kernel_family="einsum_contraction",
        hardware_target="cpu",
        execution_mode="in_process_python",
        output_contract="final_tensor",
        validation_mode="compare_output",
    )

    def probe(self) -> RouteProbe:
        return RouteProbe(self.name, True, metadata={"numpy": np.__version__})

    def capabilities(self) -> RouteCapabilities:
        return RouteCapabilities(
            identity=self.identity,
            supported_workload_families=("builtin", "qasm_file"),
            can_return_output=True,
            can_measure_energy=True,
            metadata={"numpy": np.__version__},
        )

    def can_execute(self, graph: TaskGraph, context: BenchmarkContext) -> tuple[bool, str | None]:
        return True, None

    def estimate(self, graph: TaskGraph, context: BenchmarkContext) -> RouteEstimate:
        return RouteEstimate(
            self.name,
            sum(task.estimated_flops for task in graph.tasks),
            sum(task.estimated_bytes for task in graph.tasks),
            graph.path_summary.largest_intermediate * 16 if graph.path_summary.largest_intermediate else None,
        )

    def prepare(self, graph: TaskGraph, network: TensorNetworkValue, context: BenchmarkContext) -> dict:
        return {"graph": graph, "network": network, "prepare_s": 0.0}

    def execute(self, prepared: object, context: BenchmarkContext) -> RouteResult:
        payload = dict(prepared)  # type: ignore[arg-type]
        graph: TaskGraph = payload["graph"]
        network: TensorNetworkValue = payload["network"]
        energy_start = read_rapl_uj()
        start = time.perf_counter()
        output, metadata = _execute_task_sequence(graph, network)
        kernel_s = time.perf_counter() - start
        energy_end = read_rapl_uj()
        energy_joules = None
        energy_source = "unavailable"
        if energy_start is not None and energy_end is not None and energy_end >= energy_start:
            energy_joules = (energy_end - energy_start) / 1_000_000.0
            energy_source = "rapl_measured" if energy_joules > 0 else "rapl_zero_or_too_short"
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
            metadata=metadata,
        )


def _execute_task_sequence(graph: TaskGraph, network: TensorNetworkValue) -> tuple[np.ndarray, dict]:
    tensors = {tensor.spec.id: np.asarray(tensor.array, dtype=np.complex128) for tensor in network.tensors}
    labels = {tensor.spec.id: tensor.spec.labels for tensor in network.tensors}
    live_ids = set(tensors)
    task_metrics: list[TaskExecutionMetric] = []
    peak_live_bytes = _live_tensor_bytes(tensors, live_ids)
    max_intermediate_bytes = 0

    if not graph.tasks:
        output, final_id, final_labels, transposed = _execute_empty_graph(graph, network)
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

    remaining_uses = _remaining_input_uses(graph)
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
            )
        )

        for input_id in task.input_tensor_ids:
            remaining_uses[input_id] = remaining_uses.get(input_id, 0) - 1
            if remaining_uses[input_id] <= 0 and input_id != final_tensor_id:
                live_ids.discard(input_id)
                tensors.pop(input_id, None)
                labels.pop(input_id, None)

    if final_tensor_id not in tensors or final_labels is None:
        raise ValueError(f"Task sequence did not produce final tensor {final_tensor_id}")

    output, transposed = _order_final_tensor(tensors[final_tensor_id], final_labels, graph.network.output_labels)
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


def _execute_empty_graph(graph: TaskGraph, network: TensorNetworkValue) -> tuple[np.ndarray, str, tuple[int, ...], bool]:
    if len(network.tensors) != 1:
        raise ValueError(
            f"Cannot execute empty TaskGraph with {len(network.tensors)} original tensors; "
            "expected exactly one tensor"
        )
    tensor = network.tensors[0]
    output, transposed = _order_final_tensor(
        np.asarray(tensor.array, dtype=np.complex128),
        tensor.spec.labels,
        graph.network.output_labels,
    )
    return output, tensor.spec.id, tensor.spec.labels, transposed


def _order_final_tensor(array: np.ndarray, actual_labels: tuple[int, ...], output_labels: tuple[int, ...]) -> tuple[np.ndarray, bool]:
    if actual_labels == output_labels:
        return np.asarray(array, dtype=np.complex128), False
    if len(actual_labels) != len(output_labels) or set(actual_labels) != set(output_labels):
        raise ValueError(f"Final tensor labels {actual_labels} do not match requested output labels {output_labels}")
    axes = tuple(actual_labels.index(label) for label in output_labels)
    return np.asarray(np.transpose(array, axes), dtype=np.complex128), True


def _remaining_input_uses(graph: TaskGraph) -> dict[str, int]:
    counts: dict[str, int] = {}
    for task in graph.tasks:
        for tensor_id in task.input_tensor_ids:
            counts[tensor_id] = counts.get(tensor_id, 0) + 1
    return counts


def _live_tensor_bytes(tensors: dict[str, np.ndarray], live_ids: set[str]) -> int:
    return int(sum(tensors[tensor_id].nbytes for tensor_id in live_ids if tensor_id in tensors))
