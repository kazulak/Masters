from __future__ import annotations

from pathlib import Path

import numpy as np

from tnsim.circuits import gate_tensor, load_circuit
from tnsim.core.model import TensorNetwork, TensorValue
from tnsim.core.utils import index_symbols


def build_tensor_network(config: dict, root_dir: Path) -> TensorNetwork:
    circuit = load_circuit(config["workload"], root_dir)
    tensors: list[TensorValue] = []
    counter = 0
    wire_label: dict[int, int] = {}

    zero = np.array([1.0, 0.0], dtype=np.complex128)
    for wire in range(circuit.n_qubits):
        label = counter
        counter += 1
        wire_label[wire] = label
        tensors.append(TensorValue(f"tensor_{len(tensors)}", (label,), zero.copy(), "dense"))

    for op_index, op in enumerate(circuit.operations):
        _validate_wires(op.wires, circuit.n_qubits)
        input_labels = tuple(wire_label[wire] for wire in op.wires)
        output_labels = tuple(range(counter, counter + len(op.wires)))
        counter += len(op.wires)
        for wire, label in zip(op.wires, output_labels):
            wire_label[wire] = label

        tensor_id = f"tensor_{len(tensors)}"
        tensors.append(
            TensorValue(
                id=tensor_id,
                labels=input_labels + output_labels,
                array=gate_tensor(op),
                structure=_gate_structure(op.gate),
                produced_by=f"circuit_op_{op_index}",
            )
        )

    output_labels = tuple(wire_label[wire] for wire in range(circuit.n_qubits))
    expression = build_full_einsum_expression(tensors, output_labels)
    return TensorNetwork(circuit, tensors, output_labels, expression)


def build_full_einsum_expression(tensors: list[TensorValue], output_labels: tuple[int, ...]) -> str:
    symbols = index_symbols([tensor.labels for tensor in tensors], output_labels)
    operands = ["".join(symbols[label] for label in tensor.labels) for tensor in tensors]
    output = "".join(symbols[label] for label in output_labels)
    return ",".join(operands) + "->" + output


def _validate_wires(wires: tuple[int, ...], n_qubits: int) -> None:
    if not wires:
        raise ValueError("Gate operation has no wires")
    for wire in wires:
        if wire < 0 or wire >= n_qubits:
            raise ValueError(f"Wire {wire} outside qreg size {n_qubits}")


def _gate_structure(gate: str) -> str:
    if gate.lower() in {"z", "s", "t", "rz", "cz"}:
        return "diagonal"
    if gate.lower() in {"x", "cx", "cnot", "swap"}:
        return "permutation"
    return "dense"

