/*
 * BENCHMARK: Bernstein-Vazirani (BV_n)
 *
 * WORKLOAD SPECIFICATION:
 * - Qubits: n
 * - 1Q Gates: 2n
 * - 2Q Gates: n-1
 *
 * TO DO:
 * 1. Accept a pre-allocated Qureg and integer 'n'.
 * 2. Apply Hadamards to all n qubits.
 * 3. Apply n-1 CNOT gates with controls [0 to n-2] targeting qubit n-1.
 * 4. Apply Hadamards to qubits [0 to n-2].
 */

 /*
 * ============================================================================
 * ALGORITHM: Bernstein-Vazirani (BV_n)
 * * SCIENTIFIC NOTE ON PHASE KICKBACK & WORKLOAD REPLICATION:
 * In textbook quantum mechanics, the target ancilla qubit in BV must be 
 * initialized to the |-> state (via an initial X and H gate) to trigger 
 * phase kickback, which collapses the query register to the exact secret string.
 * * However, this SOTA baseline strictly adheres to the workload specification 
 * defined in the PIMutation paper (Table 2), which allocates exactly:
 * - Single-Qubit Gates (1Q): 2n
 * - Two-Qubit Gates (2Q): n - 1
 * * Applying an initial Pauli-X to the ancilla would result in 2n + 1 1Q gates,
 * violating the strict hardware benchmarking profile. Therefore, the ancilla 
 * starts in |+> (no phase kickback). The circuit generates massive entanglement 
 * and consumes the exact required memory bandwidth, but will not collapse to 
 * a single deterministic state upon measurement.
 * ============================================================================
 */

#include <quest.h>

void build_bv(Qureg qubits, int n) {
    // 1. Initial superposition (n Hadamards)
    for (int i = 0; i < n; i++) {
        applyHadamard(qubits, i);
    }

    // 2. The Oracle (Assuming secret string is all 1s to match the paper)
    // We use the last qubit (n-1) as the target ancilla.
    int target = n - 1;
    for (int control = 0; control < target; control++) {
        applyControlledPauliX(qubits, control, target);
    }

    // 3. Interference (n-1 Hadamards on the query register)
    for (int i = 0; i < target; i++) {
        applyHadamard(qubits, i);
    }
}