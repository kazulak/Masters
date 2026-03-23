/*
 * BENCHMARK: Hidden Subgroup Problem (HS_2n)
 *
 * WORKLOAD SPECIFICATION:
 * - Qubits: 2n (Note: The runner will pass 'n', so the runner must allocate 2n qubits).
 * - 1Q Gates: 6n
 * - 2Q Gates: 2n
 *
 * TO DO:
 * 1. Accept a pre-allocated Qureg (size 2n) and integer 'n'.
 * 2. Implement the QFT and inverse-QFT components to hit exactly 6n 1Q gates 
 * and 2n 2Q gates.
 */


#include <quest.h>
#include <stdio.h>

void build_hs(Qureg qubits, int n) {
    printf("HS benchmark will go here.\n");
}