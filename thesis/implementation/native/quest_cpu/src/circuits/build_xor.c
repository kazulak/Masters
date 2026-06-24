/*
 * BENCHMARK: Exclusive-OR (XOR_n)
 *
 * WORKLOAD SPECIFICATION:
 * - Qubits: n
 * - 1Q Gates: 0
 * - 2Q Gates: n-1
 *
 * TO DO:
 * 1. Accept a pre-allocated Qureg and integer 'n'.
 * 2. Apply a cascade of n-1 CNOT gates to simulate the parity calculation.
 */


#include "circuit_manifest.h"

GateCounts build_xor(Qureg qubits, int n) {
    GateCounts counts = {0, 0};

    // Apply n-1 CNOT gates to simulate an XOR parity chain
    for (int i = 0; i < n - 1; i++) {
        applyControlledPauliX(qubits, i, i + 1);
        counts.two_qubit++;
    }

    return counts;
}
