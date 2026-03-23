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

#include <quest.h>
#include <stdio.h>

void build_bv(Qureg qubits, int n) {
    printf("[WIP] Bernstein-Vazirani benchmark will go here.\n");
}