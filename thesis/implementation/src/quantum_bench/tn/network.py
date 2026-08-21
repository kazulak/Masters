from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from quantum_bench.circuits.library import gate_structure, gate_tensor
from quantum_bench.core.indices import LABEL_LIST_EINSUM_SENTINEL, index_symbols, label_count, supports_string_einsum
from quantum_bench.core.records import CircuitSpec, TensorNetworkSpec, TensorSpec, TensorValue

if TYPE_CHECKING:
    from quantum_bench.tn.graph import ContractionDAG


@dataclass
class TensorNetworkValue:
    spec: TensorNetworkSpec
    tensors: list[TensorValue]


@dataclass(frozen=True, slots=True, kw_only=True)
class TensorInput:
    """One numerical tensor payload, kept outside network structure."""

    tensor_id: str
    array: np.ndarray


@dataclass(frozen=True, slots=True, kw_only=True)
class TensorInputs:
    """Immutable collection boundary for execution inputs.

    NumPy arrays are mutable objects, so executors must treat them as read-only.
    The tuple prevents the pipeline itself from adding or replacing inputs.
    """

    values: tuple[TensorInput, ...]


def build_tensor_network(circuit: CircuitSpec) -> TensorNetworkValue:
    """Build the legacy combined structure/value view.

    New execution code should use :func:`build_tensor_network_data`.  This
    adapter remains while historical routes and evidence readers are migrated.
    """

    spec, inputs = build_tensor_network_data(circuit)
    specs_by_id = {tensor.id: tensor for tensor in spec.tensors}
    return TensorNetworkValue(
        spec,
        [
            TensorValue(
                specs_by_id[value.tensor_id],
                value.array,
            )
            for value in inputs.values
        ],
    )


def build_tensor_network_data(
    circuit: CircuitSpec,
) -> tuple[TensorNetworkSpec, TensorInputs]:
    """Build immutable tensor-network structure and separate numerical inputs."""

    tensors: list[TensorValue] = []
    counter = 0
    wire_label: dict[int, int] = {}
    zero = np.array([1.0, 0.0], dtype=np.complex128)

    for wire in range(circuit.n_qubits):
        label = counter
        counter += 1
        wire_label[wire] = label
        spec = TensorSpec(f"tensor_{len(tensors)}", (label,), zero.shape, "dense")
        tensors.append(TensorValue(spec, zero.copy()))

    for op_index, op in enumerate(circuit.operations):
        _validate_wires(op.wires, circuit.n_qubits)
        input_labels = tuple(wire_label[wire] for wire in op.wires)
        output_labels = tuple(range(counter, counter + len(op.wires)))
        counter += len(op.wires)
        for wire, label in zip(op.wires, output_labels):
            wire_label[wire] = label
        array = gate_tensor(op)
        spec = TensorSpec(
            id=f"tensor_{len(tensors)}",
            labels=input_labels + output_labels,
            shape=array.shape,
            structure=gate_structure(op.gate),
            produced_by=f"circuit_op_{op_index}",
        )
        tensors.append(TensorValue(spec, array))

    output_labels = tuple(wire_label[wire] for wire in range(circuit.n_qubits))
    expression = build_full_einsum_expression([tensor.spec for tensor in tensors], output_labels)
    network_spec = TensorNetworkSpec(
        circuit=circuit,
        tensors=tuple(tensor.spec for tensor in tensors),
        output_labels=output_labels,
        einsum_expression=expression,
    )
    inputs = TensorInputs(
        values=tuple(
            TensorInput(tensor_id=tensor.spec.id, array=np.asarray(tensor.array))
            for tensor in tensors
        )
    )
    validate_tensor_inputs(network_spec, inputs)
    return network_spec, inputs


def validate_tensor_inputs(network: TensorNetworkSpec, inputs: TensorInputs) -> None:
    """Validate that execution inputs match network descriptors exactly."""

    specs = {tensor.id: tensor for tensor in network.tensors}
    if len(specs) != len(network.tensors):
        raise ValueError("Tensor network contains duplicate tensor ids")
    values = {value.tensor_id: value for value in inputs.values}
    if len(values) != len(inputs.values):
        raise ValueError("Tensor inputs contain duplicate tensor ids")
    if set(values) != set(specs):
        missing = sorted(set(specs) - set(values))
        extra = sorted(set(values) - set(specs))
        raise ValueError(f"Tensor input ids do not match network: missing={missing} extra={extra}")
    for tensor_id, value in values.items():
        array = np.asarray(value.array)
        if tuple(array.shape) != specs[tensor_id].shape:
            raise ValueError(
                f"Tensor input {tensor_id} shape {array.shape} "
                f"does not match descriptor {specs[tensor_id].shape}"
            )
        if array.dtype != np.dtype(specs[tensor_id].dtype):
            raise ValueError(
                f"Tensor input {tensor_id} dtype {array.dtype} "
                f"does not match descriptor {specs[tensor_id].dtype}"
            )


def validate_dag_inputs(
    dag: ContractionDAG,
    inputs: Mapping[str, np.ndarray],
) -> None:
    """Validate input arrays against the tensor descriptors in a DAG."""

    specs = {tensor.id: tensor for tensor in dag.tensors}
    if len(specs) != len(dag.tensors):
        raise ValueError("ContractionDAG contains duplicate input tensor IDs")
    actual_ids = set(inputs)
    expected_ids = set(specs)
    if actual_ids != expected_ids:
        missing = sorted(expected_ids - actual_ids)
        extra = sorted(actual_ids - expected_ids)
        raise ValueError(f"Tensor input ids do not match DAG: missing={missing} extra={extra}")
    for tensor_id, value in inputs.items():
        array = np.asarray(value)
        descriptor = specs[tensor_id]
        if tuple(array.shape) != descriptor.shape:
            raise ValueError(
                f"Tensor input {tensor_id} shape {array.shape} "
                f"does not match descriptor {descriptor.shape}"
            )
        if array.dtype != np.dtype(descriptor.dtype):
            raise ValueError(
                f"Tensor input {tensor_id} dtype {array.dtype} "
                f"does not match descriptor {descriptor.dtype}"
            )


def tensor_input_map(inputs: TensorInputs) -> dict[str, np.ndarray]:
    """Return a fresh lookup table without copying or mutating tensor arrays."""

    return {value.tensor_id: np.asarray(value.array) for value in inputs.values}


def build_full_einsum_expression(tensors: list[TensorSpec], output_labels: tuple[int, ...]) -> str:
    label_sets = [tensor.labels for tensor in tensors]
    if not supports_string_einsum(label_sets, output_labels):
        return f"{LABEL_LIST_EINSUM_SENTINEL}:labels={label_count(label_sets, output_labels)}"
    symbols = index_symbols([tensor.labels for tensor in tensors], output_labels)
    operands = ["".join(symbols[label] for label in tensor.labels) for tensor in tensors]
    output = "".join(symbols[label] for label in output_labels)
    return ",".join(operands) + "->" + output


def interleaved_einsum_args(network: TensorNetworkValue) -> list[object]:
    args: list[object] = []
    for tensor in network.tensors:
        args.extend([tensor.array, list(tensor.spec.labels)])
    args.append(list(network.spec.output_labels))
    return args


def _validate_wires(wires: tuple[int, ...], n_qubits: int) -> None:
    if not wires:
        raise ValueError("Gate operation has no wires")
    seen: set[int] = set()
    for wire in wires:
        if wire < 0 or wire >= n_qubits:
            raise ValueError(f"Wire {wire} outside qreg size {n_qubits}")
        if wire in seen:
            raise ValueError(f"Duplicate wire {wire} in gate operation")
        seen.add(wire)
