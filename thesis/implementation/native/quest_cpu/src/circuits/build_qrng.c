/*
 * BENCHMARK: Quantum Random Number Generator (QRNG_n)
 *
 * WORKLOAD SPECIFICATION:
 * - Qubits: n
 * - 1Q Gates: n
 * - 2Q Gates: 0
 *
 * TO DO:
 * 1. Accept a pre-allocated Qureg and integer 'n'.
 * 2. Apply n Hadamard gates to push all qubits into a superposition.
 */

#include "circuit_manifest.h"

/*
 * BENCHMARK: Quantum Random Number Generator (QRNG_n)
 *
 * WORKLOAD SPECIFICATION:
 * - Qubits: n
 * - 1Q Gates: n
 * - 2Q Gates: 0
 */
GateCounts build_qrng(Qureg qubits, int n) {
    GateCounts counts = {0, 0};

    // Apply a Hadamard gate to every qubit in the register
    for (int i = 0; i < n; i++) {
        applyHadamard(qubits, i);
        counts.one_qubit++;
    }

    return counts;
}
