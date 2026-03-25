/*
 * BENCHMARK: BB84 Protocol (BB_n)
 *
 * WORKLOAD SPECIFICATION:
 * - Qubits: n
 * - 1Q Gates: 2n
 * - 2Q Gates: 0
 *
 * TO DO:
 * 1. Accept a pre-allocated Qureg and integer 'n'.
 * 2. Apply n Hadamard gates (simulating state preparation).
 * 3. Apply n Pauli-X gates (simulating basis encoding).
 */

#include <quest.h>

void build_bb84(Qureg qubits, int n) {
// Apply 2n single-qubit gates (n Hadamards + n Pauli-X)
    for (int i = 0; i < n; i++) {
        applyHadamard(qubits, i);
        applyPauliX(qubits, i);
    }
}