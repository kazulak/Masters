/*
 * BENCHMARK: Error Detection Code (EDC_n)
 *
 * WORKLOAD SPECIFICATION:
 * - Qubits: n
 * - 1Q Gates: 2n
 * - 2Q Gates: 2n-2
 *
 * TO DO:
 * 1. Accept a pre-allocated Qureg and integer 'n'.
 * 2. Construct a syndrome measurement loop that exactly hits 2n 1Q gates 
 * and 2n-2 2Q (CNOT) gates.
 */

#include <quest.h>

void build_edc(Qureg qubits, int n) {
    // 1. First set of 1Q gates (n Hadamards)
    for (int i = 0; i < n; i++) {
        applyHadamard(qubits, i);
    }

    // 2. The 2Q gates (2n - 2 CNOTs total)
    // Forward cascade (n - 1 gates)
    for (int i = 0; i < n - 1; i++) {
        applyControlledPauliX(qubits, i, i + 1);
    }
    // Backward cascade (n - 1 gates)
    for (int i = n - 1; i > 0; i--) {
        applyControlledPauliX(qubits, i, i - 1);
    }

    // 3. Second set of 1Q gates (n Pauli-X gates)
    for (int i = 0; i < n; i++) {
        applyPauliX(qubits, i);
    }
}