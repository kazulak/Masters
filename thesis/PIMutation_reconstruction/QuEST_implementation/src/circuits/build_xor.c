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


#include <quest.h>
#include <stdio.h>

void build_xor(Qureg qubits, int n) {
    printf("XOR benchmark will go here.\n");
}