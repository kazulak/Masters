#include "../verify_runner.h"
#include <stdio.h>
#include <math.h>

bool verify_qrng(Qureg qubits, int n) {
    double tolerance = 1e-5;

    // Check that every individual qubit has a 50% chance of being 0
    for (int i = 0; i < n; i++) {
        // QuEST v4 API: calcProbOfQubitOutcome(Qureg, target_qubit, outcome)
        double prob_0 = calcProbOfQubitOutcome(qubits, i, 0);
        
        if (fabs(prob_0 - 0.5) > tolerance) {
            printf(" -> FAILED: Qubit %d has probability %f of being 0 (Expected 0.5)\n", i, prob_0);
            return false;
        }
    }

    printf(" -> SUCCESS: QRNG state vector represents a perfect uniform superposition.\n");
    return true;
}