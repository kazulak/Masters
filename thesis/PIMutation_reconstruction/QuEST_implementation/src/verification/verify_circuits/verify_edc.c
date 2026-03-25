#include "../verify_runner.h"
#include <stdio.h>
#include <math.h>

bool verify_edc(Qureg qubits, int n) {
    double tolerance = 1e-5;
    for (int i = 0; i < n; i++) {
        double prob_0 = calcProbOfQubitOutcome(qubits, i, 0);
        if (fabs(prob_0 - 0.5) > tolerance) {
            printf(" -> FAILED: EDC Qubit %d failed un-computation (Prob: %f)\n", i, prob_0);
            return false;
        }
    }
    printf(" -> SUCCESS: EDC forward/backward cascades perfectly cancelled out.\n");
    return true;
}