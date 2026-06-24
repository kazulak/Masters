#include "../verify_runner.h"
#include <stdio.h>
#include <math.h>

bool verify_xor(Qureg qubits, int n) {
    double tolerance = 1e-5;
    for (int i = 0; i < n; i++) {
        double prob_0 = calcProbOfQubitOutcome(qubits, i, 0);
        if (fabs(prob_0 - 1.0) > tolerance) {
            printf(" -> FAILED: XOR Qubit %d was flipped to 1 unexpectedly.\n", i);
            return false;
        }
    }
    printf(" -> SUCCESS: XOR parity checks maintained the ground state.\n");
    return true;
}