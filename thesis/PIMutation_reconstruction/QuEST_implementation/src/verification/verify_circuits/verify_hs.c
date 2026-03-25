#include "../verify_runner.h"
#include <stdio.h>
#include <math.h>

bool verify_hs(Qureg qubits, int n) {
    int total_qubits = 2 * n; // HS uses 2n qubits
    double tolerance = 1e-5;
    
    for (int i = 0; i < total_qubits; i++) {
        double prob_0 = calcProbOfQubitOutcome(qubits, i, 0);
        if (fabs(prob_0 - 1.0) > tolerance) {
            printf(" -> FAILED: HS Qubit %d escaped the H-X-H identity cascade.\n", i);
            return false;
        }
    }
    printf(" -> SUCCESS: HS identity matrix (HXH) mathematically verified.\n");
    return true;
}