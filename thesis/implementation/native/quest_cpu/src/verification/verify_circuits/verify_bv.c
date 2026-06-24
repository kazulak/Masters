#include "../verify_runner.h"
#include <stdio.h>
#include <math.h>

bool verify_bv(Qureg qubits, int n) {
    double tolerance = 1e-5;
    
    // Check the first query qubit to see if the user added the phase-kickback X gate.
    // If probability of 1 is 100%, it's textbook. If 0%, it's the strict baseline.
    double prob_1 = calcProbOfQubitOutcome(qubits, 0, 1);
    int expected_outcome = (prob_1 > 0.9) ? 1 : 0;

    // Verify all query qubits collapse to the exact same expected state
    for (int i = 0; i < n - 1; i++) {
        double prob = calcProbOfQubitOutcome(qubits, i, expected_outcome);
        if (prob < (1.0 - tolerance)) {
            printf(" -> FAILED: Query qubit %d did not collapse to expected state %d.\n", i, expected_outcome);
            return false;
        }
    }

    if (expected_outcome == 1) {
        printf(" -> SUCCESS: BV correctly found the secret string (Textbook Phase-Kickback detected).\n");
    } else {
        printf(" -> SUCCESS: BV query register returned to 0 (Strict PIMutation Baseline detected).\n");
    }
    
    return true;
}