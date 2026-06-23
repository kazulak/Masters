#include "verify_runner.h"
#include "../circuits/circuit_manifest.h"

#include <string.h>
#include <stdio.h>
#include <math.h> // Required for fabs()
#include <strings.h>

extern bool verify_qrng(Qureg qubits, int n);
extern bool verify_bv(Qureg qubits, int n);
extern bool verify_bb84(Qureg qubits, int n);
extern bool verify_edc(Qureg qubits, int n);
extern bool verify_hs(Qureg qubits, int n);
extern bool verify_xor(Qureg qubits, int n);

typedef bool (*VerifierFn)(Qureg qubits, int n);
typedef GateCounts (*BuilderFn)(Qureg qubits, int n);

typedef struct {
    BenchmarkAlgo algo;
    const char* name;
    BuilderFn build;
    VerifierFn verify;
} VerificationCase;

static const VerificationCase CASES[] = {
    {BENCHMARK_ALGO_QRNG, "QRNG", build_qrng, verify_qrng},
    {BENCHMARK_ALGO_BV, "BV", build_bv, verify_bv},
    {BENCHMARK_ALGO_BB84, "BB84", build_bb84, verify_bb84},
    {BENCHMARK_ALGO_EDC, "EDC", build_edc, verify_edc},
    {BENCHMARK_ALGO_HS, "HS", build_hs, verify_hs},
    {BENCHMARK_ALGO_XOR, "XOR", build_xor, verify_xor}
};

static const int CASE_COUNT = sizeof(CASES) / sizeof(CASES[0]);

// Helper to check if the state vector is physically possible
bool is_state_valid(Qureg q) {
    if (fabs(calcTotalProb(q) - 1.0) > 1e-5) {
        printf(" -> FAILED: State vector normalization lost (Total Prob != 1.0)!\n");
        return false;
    }
    return true;
}

static bool token_selected(const char* mode, const char* name) {
    if (strcasecmp(mode, "FULL") == 0 || strcasecmp(mode, "BASE") == 0) {
        return true;
    }

    char copy[256];
    snprintf(copy, sizeof(copy), "%s", mode);

    char* token = strtok(copy, ",");
    while (token != NULL) {
        while (*token == ' ') {
            token++;
        }
        if (strcasecmp(token, name) == 0) {
            return true;
        }
        if (strcasecmp(token, "BB") == 0 && strcmp(name, "BB84") == 0) {
            return true;
        }
        token = strtok(NULL, ",");
    }

    return false;
}

static bool validate_mode(const char* mode) {
    if (mode == NULL || strlen(mode) == 0) {
        printf(" -> FAILED: Empty verification mode.\n");
        return false;
    }

    if (strcasecmp(mode, "FULL") == 0 || strcasecmp(mode, "BASE") == 0) {
        return true;
    }

    char copy[256];
    snprintf(copy, sizeof(copy), "%s", mode);

    char* token = strtok(copy, ",");
    while (token != NULL) {
        while (*token == ' ') {
            token++;
        }
        if (strlen(token) == 0) {
            printf(" -> FAILED: Empty token in verification mode '%s'.\n", mode);
            return false;
        }
        if (parse_benchmark_algo(token) == BENCHMARK_ALGO_UNKNOWN ||
            parse_benchmark_algo(token) == BENCHMARK_ALGO_RANDOM) {
            printf(" -> FAILED: Unknown verification selection '%s'.\n", token);
            return false;
        }
        token = strtok(NULL, ",");
    }

    return true;
}

static bool verify_counts(
    BenchmarkAlgo algo,
    GateCounts actual,
    int input_qubits,
    int allocated_qubits
) {
    GateCounts expected = expected_gate_counts(algo, input_qubits, allocated_qubits, 0);
    if (!gate_counts_match(actual, expected)) {
        printf(
            " -> FAILED: Gate counts for %s were 1Q=%ld, 2Q=%ld; expected 1Q=%ld, 2Q=%ld.\n",
            benchmark_algo_name(algo),
            actual.one_qubit,
            actual.two_qubit,
            expected.one_qubit,
            expected.two_qubit
        );
        return false;
    }

    printf(
        " -> COUNT OK: %s applied 1Q=%ld, 2Q=%ld.\n",
        benchmark_algo_name(algo),
        actual.one_qubit,
        actual.two_qubit
    );
    return true;
}

int run_test_suite(const char* mode) {
    printf("==========================================\n");
    printf("Starting Verification Suite: Mode [%s]\n", mode);
    printf("==========================================\n");

    if (!validate_mode(mode)) {
        printf("==========================================\n");
        printf("Verification Results: 0/0 Passed\n");
        printf("==========================================\n");
        return 1;
    }

    int test_qubits = 4;
    Qureg q = createQureg(test_qubits);
    int passed = 0;
    int total = 0;

    for (int i = 0; i < CASE_COUNT; i++) {
        const VerificationCase* test_case = &CASES[i];
        if (!token_selected(mode, test_case->name)) {
            continue;
        }

        int allocated_qubits = test_qubits;
        int input_qubits = (test_case->algo == BENCHMARK_ALGO_HS) ? test_qubits / 2 : test_qubits;

        initZeroState(q);
        GateCounts counts = test_case->build(q, allocated_qubits);
        if (
            verify_counts(test_case->algo, counts, input_qubits, allocated_qubits) &&
            is_state_valid(q) &&
            test_case->verify(q, allocated_qubits)
        ) {
            passed++;
        }
        total++;
    }
    
    printf("==========================================\n");
    printf("Verification Results: %d/%d Passed\n", passed, total);
    printf("==========================================\n");

    destroyQureg(q); 

    if (total == 0 || passed != total) {
        return 1;
    }
    return 0;
}
