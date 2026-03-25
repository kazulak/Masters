/*
 * SCIENTIFIC OBJECTIVE: The SOTA CPU Baseline Profiler
 * * DESCRIPTION:
 * This is the primary entry point. It handles environment initialization, 
 * command-line argument parsing, and the strict profiling boundaries 
 * (Time and Energy) for the selected quantum circuit.
 *
 * TO DO:
 * 1. Parse CLI arguments: `--algo <NAME>`, `--qubits <N>`, `--depth <D>`.
 * 2. Initialize the QuESTEnv and allocate the main Qureg of size N.
 * 3. Call `init_energy_profiler()` to dynamically detect AMD/Intel RAPL support.
 * 4. Call `start_energy_profiling()`.
 * 5. Start OpenMP wall-clock timer (omp_get_wtime).
 * 6. ROUTING LOGIC: Switch statement to build and run the circuit.
 * 7. Stop timers and energy trackers.
 * 8. Output the 'Comp.' execution time and Joule consumption to stdout in a clean format.
 * 9. Free the Qureg and QuESTEnv to prevent memory leaks.
 */

 /*
 * SCIENTIFIC OBJECTIVE: Hardware-Agnostic CPU Energy Telemetry
 * Works for Intel (Cluster) and AMD (Local, via zenpower/amd_energy modules).
 */

/*
 * SCIENTIFIC OBJECTIVE: The SOTA CPU Baseline Profiler
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <quest.h>
#include "profiling/rapl_energy.h"
#include "verification/verify_runner.h" // <-- Add this include

// Forward declarations for the circuit builders (we will write these next)
extern void build_bb84(Qureg qubits, int n);
extern void build_bv(Qureg qubits, int n);
extern void build_edc(Qureg qubits, int n);
extern void build_hs(Qureg qubits, int n);
extern void build_qrng(Qureg qubits, int n);
extern void build_xor(Qureg qubits, int n);
extern void build_random(Qureg qubits, int n, int depth);

void print_usage(const char* prog_name) {
    printf("Usage for Profiling:  %s --algo <NAME> --qubits <N> [--depth <D>]\n", prog_name);
    printf("Usage for Testing:    %s --verify <MODE>\n", prog_name);
    printf("Algorithms:           BB84, BV, EDC, HS, QRNG, XOR, RANDOM\n");
    printf("Verify Modes:         FULL, BASE, or comma-separated (e.g., BV,BB84)\n");
    printf("Example (Profile):    sudo %s --algo BV --qubits 26\n", prog_name);
    printf("Example (Test):       %s --verify FULL\n", prog_name);
}

int main(int argc, char** argv) {
    char* algo = NULL;
    int n_qubits = 0;
    int depth = 10; // Default for random
    char* verify_mode = NULL; // Replaces 'bool run_verification = false;'

    // 1. Parse CLI Arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--algo") == 0 && i + 1 < argc) {
            algo = argv[++i];
        } else if (strcmp(argv[i], "--qubits") == 0 && i + 1 < argc) {
            n_qubits = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--depth") == 0 && i + 1 < argc) {
            depth = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--verify") == 0 && i + 1 < argc) {
            verify_mode = argv[++i]; 
        }
    }

    // 2. THE INTERCEPT (Your isolated, elegant design)
    if (verify_mode != NULL) {
        initQuESTEnv();              // Boot up just for testing
        run_test_suite(verify_mode); // Run tests
        finalizeQuESTEnv();          // Shut down
        return 0;                    // Exit immediately
    }

    if (!algo || n_qubits <= 0) {
        print_usage(argv[0]);
        return 1;
    }

    // Adjust allocation for Hidden Subgroup (requires 2n qubits)
    int alloc_qubits = (strcmp(algo, "HS") == 0) ? 2 * n_qubits : n_qubits;

    if (alloc_qubits > 34) {
        printf("WARNING: %d total qubits requires massive RAM (>256GB). May OOM.\n", alloc_qubits);
    }

    // 2. Initialize QuEST Environment
    initQuESTEnv(); 
    Qureg qubits = createQureg(alloc_qubits); // No longer requires passing 'env'
    initZeroState(qubits);

    printf("==========================================\n");
    printf("QuEST SOTA Baseline Runner\n");
    printf("Algorithm: %s\n", algo);
    printf("Input Qubits (n): %d\n", n_qubits);
    printf("Allocated Qubits: %d\n", alloc_qubits);
    printf("==========================================\n");

    // 4. START PROFILING BOUNDARY
    RaplState energy_state = start_energy_profiling();
    double start_time = omp_get_wtime();

    // 5. Routing Logic
    if (strcmp(algo, "BB84") == 0)   build_bb84(qubits, n_qubits);
    else if (strcmp(algo, "BV") == 0) build_bv(qubits, n_qubits);
    else if (strcmp(algo, "EDC") == 0) build_edc(qubits, n_qubits);
    else if (strcmp(algo, "HS") == 0) build_hs(qubits, n_qubits);
    else if (strcmp(algo, "QRNG") == 0) build_qrng(qubits, n_qubits);
    else if (strcmp(algo, "XOR") == 0) build_xor(qubits, n_qubits);
    else if (strcmp(algo, "RANDOM") == 0) build_random(qubits, n_qubits, depth);
    else {
        printf("Error: Unknown algorithm '%s'\n", algo);
        // UPDATED: v4 Cleanup for error branch
        destroyQureg(qubits);
        finalizeQuESTEnv(); 
        return 1;
    }

    // 6. STOP PROFILING BOUNDARY
    double end_time = omp_get_wtime();
    double joules = stop_energy_profiling(energy_state);

    // 7. Output Clean Metrics
    printf("-> Execution Time (Comp.): %f seconds\n", end_time - start_time);
    printf("-> Energy Consumed:        %f Joules\n", joules);
    printf("==========================================\n");

    // 8. UPDATED: v4 Cleanup for successful execution
    destroyQureg(qubits);
    finalizeQuESTEnv();

    return 0;
}