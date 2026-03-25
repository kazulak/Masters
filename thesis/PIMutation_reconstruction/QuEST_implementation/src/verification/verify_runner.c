#include "verify_runner.h"
#include <string.h>
#include <stdio.h>
#include <math.h> // Required for fabs()

// 1. Forward declarations so the compiler knows these exist!
extern void build_qrng(Qureg qubits, int n);
extern void build_bv(Qureg qubits, int n);
extern void build_bb84(Qureg qubits, int n);
extern void build_edc(Qureg qubits, int n);
extern void build_hs(Qureg qubits, int n);
extern void build_xor(Qureg qubits, int n);

extern bool verify_qrng(Qureg qubits, int n);
extern bool verify_bv(Qureg qubits, int n);
extern bool verify_bb84(Qureg qubits, int n);
extern bool verify_edc(Qureg qubits, int n);
extern bool verify_hs(Qureg qubits, int n);
extern bool verify_xor(Qureg qubits, int n);

// Helper to check if the state vector is physically possible
bool is_state_valid(Qureg q) {
    if (fabs(calcTotalProb(q) - 1.0) > 1e-5) {
        printf(" -> FAILED: State vector normalization lost (Total Prob != 1.0)!\n");
        return false;
    }
    return true;
}

void run_test_suite(const char* mode) {
    printf("==========================================\n");
    printf("Starting Verification Suite: Mode [%s]\n", mode);
    printf("==========================================\n");

    int test_qubits = 4; 
    Qureg q = createQureg(test_qubits); 
    int passed = 0;
    int total = 0;

    // We use strstr to see if the mode string CONTAINS the algorithm name.
    // This allows passing "--verify FULL" or a list like "--verify BB84,BV,QRNG"
    
    bool run_all = (strcmp(mode, "BASE") == 0 || strcmp(mode, "FULL") == 0);

    // Test QRNG
    if (run_all || strstr(mode, "QRNG") != NULL) {
        initZeroState(q);
        build_qrng(q, test_qubits);
        
        // If is_state_valid fails, verify_qrng is skipped entirely!
        if (is_state_valid(q) && verify_qrng(q, test_qubits)) passed++;
        total++;
    }

    // Test BV
    if (run_all || strstr(mode, "BV") != NULL) {
        initZeroState(q);
        build_bv(q, test_qubits);
        
        if (is_state_valid(q) && verify_bv(q, test_qubits)) passed++;
        total++;
    }
    
    // Test BB84
    if (run_all || strstr(mode, "BB84") != NULL) {
        initZeroState(q);
        build_bb84(q, test_qubits);
        if (is_state_valid(q) && verify_bb84(q, test_qubits)) passed++;
        total++;
    }

    // Test EDC
    if (run_all || strstr(mode, "EDC") != NULL) {
        initZeroState(q);
        build_edc(q, test_qubits);
        if (is_state_valid(q) && verify_edc(q, test_qubits)) passed++;
        total++;
    }

    // Test HS
    if (run_all || strstr(mode, "HS") != NULL) {
        // Remember HS allocates 2n qubits. 
        // We must destroy the 4-qubit 'q' and make an 8-qubit one temporarily
        destroyQureg(q);
        q = createQureg(test_qubits * 2);
        initZeroState(q);
        
        build_hs(q, test_qubits);
        if (is_state_valid(q) && verify_hs(q, test_qubits)) passed++;
        total++;
        
        // Revert back to the 4-qubit register for the next tests
        destroyQureg(q);
        q = createQureg(test_qubits);
    }

    // Test XOR
    if (run_all || strstr(mode, "XOR") != NULL) {
        initZeroState(q);
        build_xor(q, test_qubits);
        if (is_state_valid(q) && verify_xor(q, test_qubits)) passed++;
        total++;
    }
    
    printf("==========================================\n");
    printf("Verification Results: %d/%d Passed\n", passed, total);
    printf("==========================================\n");

    destroyQureg(q); 
}