/*
 * MODULE: Randomized Circuit Generator
 *
 * SCIENTIFIC OBJECTIVE:
 * To provide a configurable stress-test for average-case execution limits. 
 * This will be crucial later for validating Tensor Network bond-dimension growth.
 *
 * TO DO:
 * 1. Accept a pre-allocated Qureg, integer 'n' (qubits), and integer 'd' (depth).
 * 2. Seed a random number generator.
 * 3. Loop 'd' times:
 * a. Randomly select a gate type (H, X, Y, Z, RX, RY, CNOT, SWAP).
 * b. Randomly select target (and control) qubits.
 * c. Apply the gate to the Qureg.
 */


#include <quest.h>
#include <stdlib.h>

void build_random(Qureg qubits, int n, int depth) {
    // Set a fixed seed so your benchmarks are reproducible!
    srand(42); 

    for (int d = 0; d < depth; d++) {
        for (int i = 0; i < n; i++) {
            int gate_type = rand() % 3;
            
            if (gate_type == 0) {
                applyHadamard(qubits, i);
            } 
            else if (gate_type == 1) {
                applyPauliX(qubits, i);
            } 
            else if (gate_type == 2 && i < n - 1) {
                // Apply a CNOT with the neighbor, if it's not the last qubit
                applyControlledPauliX(qubits, i, i + 1);
            }
        }
    }
}