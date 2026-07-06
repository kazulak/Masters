from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from quantum_bench.circuits.library import gate_structure, gate_tensor
from quantum_bench.core.indices import LABEL_LIST_EINSUM_SENTINEL, index_symbols, label_count, supports_string_einsum
from quantum_bench.core.records import CircuitSpec, TensorNetworkSpec, TensorSpec, TensorValue


@dataclass
class TensorNetworkValue:
    spec: TensorNetworkSpec
    tensors: list[TensorValue]


def build_tensor_network(circuit: CircuitSpec) -> TensorNetworkValue:
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
    return TensorNetworkValue(network_spec, tensors)


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
    for wire in wires:
        if wire < 0 or wire >= n_qubits:
            raise ValueError(f"Wire {wire} outside qreg size {n_qubits}")
