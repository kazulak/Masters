#include "../verify_runner.h"
#include <stdio.h>
#include <math.h>

bool verify_bb84(Qureg qubits, int n) {
    double tolerance = 1e-5;
    for (int i = 0; i < n; i++) {
        double prob_0 = calcProbOfQubitOutcome(qubits, i, 0);
        if (fabs(prob_0 - 0.5) > tolerance) {
            printf(" -> FAILED: BB84 Qubit %d has probability %f (Expected 0.5)\n", i, prob_0);
            return false;
        }
    }
    printf(" -> SUCCESS: BB84 maintained the |+> state across all qubits.\n");
    return true;
}