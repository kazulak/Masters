/*
 * BENCHMARK: BB84 Protocol (BB_n)
 *
 * WORKLOAD SPECIFICATION:
 * - Qubits: n
 * - 1Q Gates: 2n
 * - 2Q Gates: 0
 *
 * TO DO:
 * 1. Accept a pre-allocated Qureg and integer 'n'.
 * 2. Apply n Hadamard gates (simulating state preparation).
 * 3. Apply n Pauli-X gates (simulating basis encoding).
 */

#include <quest.h>
#include <stdio.h>

void build_bb84(Qureg qubits, int n) {
    printf("BB84 benchmark will go here.\n");
}