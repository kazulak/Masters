from __future__ import annotations

import numpy as np

from quantum_bench.circuits.library import builtin_circuit, gate_matrix


def _apply_operation(state: np.ndarray, n_qubits: int, gate: str, wires: tuple[int, ...], params: tuple[float, ...]) -> None:
    matrix = gate_matrix(gate, params)
    if len(wires) == 1:
        wire = wires[0]
        for index in range(1 << n_qubits):
            if (index >> wire) & 1 == 0:
                pair = index | (1 << wire)
                values = matrix @ state[[index, pair]]
                state[index], state[pair] = values
        return
    control, target = wires
    for index in range(1 << n_qubits):
        if ((index >> control) & 1) == 0 and ((index >> target) & 1) == 0:
            indices = [index, index | (1 << target), index | (1 << control), index | (1 << control) | (1 << target)]
            state[indices] = matrix @ state[indices]


def test_quantization_stress_is_deterministic_unitary_and_nonuniform() -> None:
    for n_qubits in (4, 6, 8):
        first = builtin_circuit("quantization_stress", {"n_qubits": n_qubits, "repeat_layers": 2})
        second = builtin_circuit("quantization-stress", {"n_qubits": n_qubits, "repeat_layers": 2})

        assert first.name == second.name
        assert first.n_qubits == second.n_qubits
        assert first.operations == second.operations
        assert first.source["deterministic_unitary"] is True
        assert {operation.gate for operation in first.operations} == {"h", "rz", "cx"}
        assert all(operation.params and operation.params[0] != 0.0 for operation in first.operations if operation.gate == "rz")

        state = np.zeros(1 << n_qubits, dtype=np.complex128)
        state[0] = 1.0
        for operation in first.operations:
            _apply_operation(state, n_qubits, operation.gate, operation.wires, operation.params)

        assert np.isclose(np.linalg.norm(state), 1.0)
        assert np.unique(np.round(np.abs(state), 8)).size > 2
        assert np.unique(np.round(np.angle(state[np.abs(state) > 1e-9]), 8)).size > 2
