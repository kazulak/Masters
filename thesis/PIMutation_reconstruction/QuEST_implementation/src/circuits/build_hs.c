/*
 * BENCHMARK: Hidden Subgroup Problem (HS_2n)
 *
 * WORKLOAD SPECIFICATION:
 * - Qubits: 2n (Note: The runner will pass 'n', so the runner must allocate 2n qubits).
 * - 1Q Gates: 6n
 * - 2Q Gates: 2n
 *
 * TO DO:
 * 1. Accept a pre-allocated Qureg (size 2n) and integer 'n'.
 * 2. Implement the QFT and inverse-QFT components to hit exactly 6n 1Q gates 
 * and 2n 2Q gates.
 */


#include <quest.h>

void build_hs(Qureg qubits, int n) {
    int total_qubits = 2 * n;

    // 1. Apply 6n 1Q gates
    // We apply 3 gates to each of the 2n qubits (3 * 2n = 6n)
    for (int i = 0; i < total_qubits; i++) {
        applyHadamard(qubits, i);
        applyPauliX(qubits, i);
        applyHadamard(qubits, i);
    }

    // 2. Apply 2n 2Q gates
    // We apply exactly one CNOT originating from each of the 2n qubits
    for (int i = 0; i < total_qubits; i++) {
        int target = (i + 1) % total_qubits; // Wrap around at the end
        applyControlledPauliX(qubits, i, target);
    }
}