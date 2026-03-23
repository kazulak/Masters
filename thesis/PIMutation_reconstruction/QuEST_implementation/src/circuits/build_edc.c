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
#include <stdio.h>

void build_edc(Qureg qubits, int n) {
    printf("EDC benchmark will go here.\n");
}