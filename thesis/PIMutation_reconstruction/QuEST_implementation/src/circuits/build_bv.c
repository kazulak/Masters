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
 * An additional Pauli-X on the |+> target ancilla is included because it is an
 * identity on that state while bringing the implementation to the paper's exact
 * 2n single-qubit gate count. The circuit remains a workload-shape reproduction,
 * not textbook BV phase kickback.
 * ============================================================================
 */

#include "circuit_manifest.h"

GateCounts build_bv(Qureg qubits, int n) {
    GateCounts counts = {0, 0};

    // 1. Initial superposition (n Hadamards)
    for (int i = 0; i < n; i++) {
        applyHadamard(qubits, i);
        counts.one_qubit++;
    }

    // 2. The Oracle (Assuming secret string is all 1s to match the paper)
    // We use the last qubit (n-1) as the target ancilla.
    int target = n - 1;
    for (int control = 0; control < target; control++) {
        applyControlledPauliX(qubits, control, target);
        counts.two_qubit++;
    }

    applyPauliX(qubits, target);
    counts.one_qubit++;

    // 3. Interference (n-1 Hadamards on the query register)
    for (int i = 0; i < target; i++) {
        applyHadamard(qubits, i);
        counts.one_qubit++;
    }

    return counts;
}
