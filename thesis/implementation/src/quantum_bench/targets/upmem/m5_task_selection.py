"""Deterministic M5 selection and CPU prefix materialization helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from quantum_bench.core.records import ContractionTask, TaskGraph, TensorValue, to_jsonable
from quantum_bench.tn.execution_bundle import canonical_hash, contraction_path_structure_hash, with_execution_identity
from quantum_bench.tn.materialize import TaskInputMaterializationRequest, materialize_task_inputs
from quantum_bench.tn.network import TensorNetworkValue


@dataclass(frozen=True)
class M5TaskSelection:
    """A graph task and its CPU-prepared operands, with reproducible identity."""

    circuit_id: str
    circuit_semantics_hash: str
    tensor_network_hash: str
    contraction_plan_hash: str
    contraction_path_structure_hash: str
    task_id: str
    input_tensor_ids: tuple[str, str]
    output_tensor_id: str
    task_index: int
    estimated_flops: int
    task_hash: str
    output_elements: int
    contracted_elements: int
    materialization_status: str
    replayed_task_ids: tuple[str, ...]
    materialization_metadata: dict[str, Any]
    left_operand: np.ndarray
    right_operand: np.ndarray
    selected_task: ContractionTask = field(repr=False, compare=False)
    identified_graph: TaskGraph = field(repr=False, compare=False)

    @property
    def path_hash(self) -> str:
        return self.contraction_path_structure_hash

    @property
    def task(self) -> ContractionTask:
        """Compatibility alias for adapters consuming the selected task."""

        return self.selected_task

    @property
    def graph(self) -> TaskGraph:
        """Compatibility alias for adapters consuming the identified graph."""

        return self.identified_graph

    def to_json_dict(self) -> dict[str, Any]:
        return to_jsonable({
            "circuit_id": self.circuit_id,
            "circuit_semantics_hash": self.circuit_semantics_hash,
            "tensor_network_hash": self.tensor_network_hash,
            "contraction_plan_hash": self.contraction_plan_hash,
            "contraction_path_structure_hash": self.contraction_path_structure_hash,
            "task_id": self.task_id,
            "task_hash": self.task_hash,
            "input_tensor_ids": self.input_tensor_ids,
            "output_tensor_id": self.output_tensor_id,
            "task_index": self.task_index,
            "estimated_flops": self.estimated_flops,
            "output_elements": self.output_elements,
            "contracted_elements": self.contracted_elements,
            "materialization_status": self.materialization_status,
            "replayed_task_ids": self.replayed_task_ids,
            "materialization_metadata": self.materialization_metadata,
        })


def select_highest_work_supported_task(
    graph: TaskGraph,
    network_or_tensors: TensorNetworkValue | Mapping[str, TensorValue] | Sequence[TensorValue] | None = None,
    *,
    initial_tensors: Mapping[str, TensorValue] | None = None,
) -> M5TaskSelection:
    """Select the highest-work supported task and replay its graph prefix.

    ``network_or_tensors`` may be the real ``TensorNetworkValue`` or its tensor
    collection.  Prefix replay is deliberately delegated to the repository's
    canonical materializer and must happen before benchmark timing begins.
    """

    identified = with_execution_identity(graph)
    tensor_map = _tensor_map(identified, network_or_tensors, initial_tensors)
    candidates = sorted(
        enumerate(identified.tasks),
        key=lambda item: (-_task_work(item[1]), _task_hash(item[1]), item[1].id, item[0]),
    )
    if not candidates:
        raise ValueError("TaskGraph has no contraction tasks")
    failures: list[str] = []
    for task_index, task in candidates:
        if not _task_shape_supported(task, identified):
            failures.append(f"{task.id}:unsupported_task_shape")
            continue
        materialized = materialize_task_inputs(TaskInputMaterializationRequest(
            identified,
            tensor_map,
            target_task_index=task_index,
        ))
        if materialized.status not in {"initial_inputs_available", "materialized"} or materialized.left_tensor is None or materialized.right_tensor is None:
            failures.append(f"{task.id}:{materialized.reason or 'materialization_failed'}")
            continue
        left = _real_float32_operand(materialized.left_tensor.array, task, "left")
        right = _real_float32_operand(materialized.right_tensor.array, task, "right")
        return M5TaskSelection(
            circuit_id=identified.network.circuit.name,
            circuit_semantics_hash=identified.circuit_semantics_hash,
            tensor_network_hash=identified.tensor_network_hash,
            contraction_plan_hash=identified.contraction_plan_hash,
            contraction_path_structure_hash=contraction_path_structure_hash(identified),
            task_id=task.id,
            input_tensor_ids=task.input_tensor_ids,
            output_tensor_id=task.output_tensor_id,
            task_index=task_index,
            estimated_flops=int(task.estimated_flops),
            task_hash=_task_hash(task),
            output_elements=int(np.prod(task.output_shape, dtype=np.int64)),
            contracted_elements=int(task.gemm_k),
            materialization_status=materialized.status,
            replayed_task_ids=materialized.replayed_task_ids,
            materialization_metadata=materialized.to_json_dict(),
            left_operand=left,
            right_operand=right,
            selected_task=task,
            identified_graph=identified,
        )
    detail = "; ".join(failures) if failures else "no supported task"
    raise ValueError(f"no supported real-valued contraction task: {detail}")


select_and_materialize_highest_work_task = select_highest_work_supported_task


def _tensor_map(
    graph: TaskGraph,
    network_or_tensors: TensorNetworkValue | Mapping[str, TensorValue] | Sequence[TensorValue] | None,
    initial_tensors: Mapping[str, TensorValue] | None,
) -> dict[str, TensorValue]:
    if initial_tensors is not None:
        tensors = dict(initial_tensors)
    elif isinstance(network_or_tensors, TensorNetworkValue):
        _validate_network_spec(graph, network_or_tensors)
        tensors = {tensor.spec.id: tensor for tensor in network_or_tensors.tensors}
    elif isinstance(network_or_tensors, Mapping):
        tensors = dict(network_or_tensors)
    elif network_or_tensors is not None:
        tensors = {tensor.spec.id: tensor for tensor in network_or_tensors}
    else:
        raise ValueError("a TensorNetworkValue or initial tensors are required")

    graph_specs = {spec.id: spec for spec in graph.network.tensors}
    for tensor_id, tensor in tensors.items():
        if tensor_id not in graph_specs:
            raise ValueError(f"tensor id {tensor_id!r} is not present in TaskGraph")
        if not isinstance(tensor, TensorValue) or tensor.spec.id != tensor_id:
            raise ValueError(f"tensor id {tensor_id!r} does not match TaskGraph")
        expected_shape = tuple(graph_specs[tensor_id].shape)
        if tuple(tensor.spec.shape) != expected_shape or tuple(np.asarray(tensor.array).shape) != expected_shape:
            raise ValueError(f"tensor {tensor_id!r} shape does not match TaskGraph")
    return tensors


def _validate_network_spec(graph: TaskGraph, network: TensorNetworkValue) -> None:
    expected = tuple((spec.id, tuple(spec.shape)) for spec in graph.network.tensors)
    supplied = tuple((spec.id, tuple(spec.shape)) for spec in network.spec.tensors)
    if supplied != expected:
        raise ValueError("supplied network tensor IDs/shapes do not match TaskGraph")


def _task_shape_supported(task: ContractionTask, graph: TaskGraph) -> bool:
    graph_specs = {spec.id: spec for spec in graph.network.tensors}
    input_specs = [graph_specs.get(tensor_id) for tensor_id in task.input_tensor_ids]
    return (
        len(task.input_tensor_ids) == 2
        and all(spec is not None for spec in input_specs)
        and all(tuple(spec.shape) == tuple(shape) for spec, shape in zip(input_specs, task.input_shapes))
        and all(int(dim) > 0 for shape in task.input_shapes for dim in shape)
        and all(int(dim) > 0 for dim in task.output_shape)
        and int(task.gemm_m) > 0
        and int(task.gemm_k) > 0
        and int(task.gemm_n) > 0
    )


def _task_work(task: ContractionTask) -> int:
    return int(task.gemm_m) * int(task.gemm_k) * int(task.gemm_n)


def _task_hash(task: ContractionTask) -> str:
    return canonical_hash(task)


def _real_float32_operand(value: Any, task: ContractionTask, side: str) -> np.ndarray:
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"M5 real-valued task {task.id} has non-numeric values in {side} operand")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"M5 real-valued task {task.id} has nonfinite values in {side} operand")
    if np.iscomplexobj(array) and np.any(np.asarray(array.imag) != 0):
        raise ValueError(f"M5 real-valued task {task.id} has nonzero imaginary values in {side} operand")
    source = array.real if np.iscomplexobj(array) else array
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.asarray(source, dtype=np.float32)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"M5 real-valued task {task.id} overflows float32 in {side} operand")
    expected_shape = task.input_shapes[0 if side == "left" else 1]
    if tuple(result.shape) != tuple(expected_shape):
        raise ValueError(f"M5 {side} operand shape does not match TaskGraph")
    return np.ascontiguousarray(result)
