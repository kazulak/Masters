#include "circuit_manifest.h"

#include <stdio.h>
#include <string.h>
#include <strings.h>

static const CircuitManifestEntry MANIFEST[] = {
    {
        BENCHMARK_ALGO_BB84,
        "BB84",
        "BB_n",
        "BB84/BB_n",
        "n allocated qubits",
        "2n",
        "0",
        "2n",
        "0",
        "PIMutation workload-shape reproduction"
    },
    {
        BENCHMARK_ALGO_BV,
        "BV",
        "BV_n",
        "BV/BV_n",
        "n allocated qubits",
        "2n",
        "n-1",
        "2n",
        "n-1",
        "PIMutation workload-shape reproduction, not textbook BV phase-kickback"
    },
    {
        BENCHMARK_ALGO_EDC,
        "EDC",
        "EDC_n",
        "EDC/EDC_n",
        "n allocated qubits",
        "2n",
        "2n-2",
        "2n",
        "2n-2",
        "PIMutation workload-shape reproduction"
    },
    {
        BENCHMARK_ALGO_HS,
        "HS",
        "HS_2n",
        "HS/HS_2n",
        "logical n, 2n allocated qubits",
        "6n",
        "2n",
        "6n",
        "2n",
        "PIMutation workload-shape reproduction, identity-preserving HS shape"
    },
    {
        BENCHMARK_ALGO_QRNG,
        "QRNG",
        "QRNG_n",
        "QRNG/QRNG_n",
        "n allocated qubits",
        "n",
        "0",
        "n",
        "0",
        "Textbook QRNG"
    },
    {
        BENCHMARK_ALGO_XOR,
        "XOR",
        "XOR_n",
        "XOR/XOR_n",
        "n allocated qubits",
        "0",
        "n-1",
        "0",
        "n-1",
        "PIMutation workload-shape reproduction"
    },
    {
        BENCHMARK_ALGO_RANDOM,
        "RANDOM",
        "RANDOM",
        "RANDOM",
        "n allocated qubits",
        "runtime dependent",
        "runtime dependent",
        "runtime dependent",
        "runtime dependent",
        "Configurable stress test, not a PIMutation paper circuit"
    }
};

static const size_t MANIFEST_LEN = sizeof(MANIFEST) / sizeof(MANIFEST[0]);

BenchmarkAlgo parse_benchmark_algo(const char* name) {
    if (name == NULL) {
        return BENCHMARK_ALGO_UNKNOWN;
    }

    for (size_t i = 0; i < MANIFEST_LEN; i++) {
        if (strcasecmp(name, MANIFEST[i].cli_name) == 0) {
            return MANIFEST[i].id;
        }
    }

    return BENCHMARK_ALGO_UNKNOWN;
}

const CircuitManifestEntry* get_circuit_manifest(BenchmarkAlgo algo) {
    for (size_t i = 0; i < MANIFEST_LEN; i++) {
        if (MANIFEST[i].id == algo) {
            return &MANIFEST[i];
        }
    }
    return NULL;
}

const char* benchmark_algo_name(BenchmarkAlgo algo) {
    const CircuitManifestEntry* entry = get_circuit_manifest(algo);
    return entry ? entry->cli_name : "UNKNOWN";
}

int compute_qubit_layout(
    BenchmarkAlgo algo,
    int allocated_qubits_arg,
    int logical_qubits_arg,
    int* input_qubits,
    int* allocated_qubits,
    char* error,
    size_t error_len
) {
    if (input_qubits == NULL || allocated_qubits == NULL) {
        return 0;
    }

    if (allocated_qubits_arg > 0 && logical_qubits_arg > 0) {
        snprintf(error, error_len, "Use either --qubits or --logical-qubits, not both.");
        return 0;
    }

    if (algo == BENCHMARK_ALGO_UNKNOWN) {
        snprintf(error, error_len, "Unknown algorithm.");
        return 0;
    }

    if (logical_qubits_arg > 0) {
        *input_qubits = logical_qubits_arg;
        *allocated_qubits = (algo == BENCHMARK_ALGO_HS) ? logical_qubits_arg * 2 : logical_qubits_arg;
    } else {
        if (allocated_qubits_arg <= 0) {
            snprintf(error, error_len, "A positive --qubits value is required.");
            return 0;
        }

        *allocated_qubits = allocated_qubits_arg;
        *input_qubits = allocated_qubits_arg;

        if (algo == BENCHMARK_ALGO_HS) {
            if (allocated_qubits_arg % 2 != 0) {
                snprintf(error, error_len, "HS requires an even allocated --qubits value.");
                return 0;
            }
            *input_qubits = allocated_qubits_arg / 2;
        }
    }

    if (*input_qubits <= 0 || *allocated_qubits <= 0) {
        snprintf(error, error_len, "Qubit counts must be positive.");
        return 0;
    }

    return 1;
}

GateCounts expected_gate_counts(
    BenchmarkAlgo algo,
    int input_qubits,
    int allocated_qubits,
    int depth
) {
    GateCounts counts = {-1, -1};

    switch (algo) {
        case BENCHMARK_ALGO_BB84:
            counts.one_qubit = 2L * allocated_qubits;
            counts.two_qubit = 0;
            break;
        case BENCHMARK_ALGO_BV:
            counts.one_qubit = 2L * allocated_qubits;
            counts.two_qubit = allocated_qubits - 1;
            break;
        case BENCHMARK_ALGO_EDC:
            counts.one_qubit = 2L * allocated_qubits;
            counts.two_qubit = 2L * allocated_qubits - 2;
            break;
        case BENCHMARK_ALGO_HS:
            counts.one_qubit = 6L * input_qubits;
            counts.two_qubit = 2L * input_qubits;
            break;
        case BENCHMARK_ALGO_QRNG:
            counts.one_qubit = allocated_qubits;
            counts.two_qubit = 0;
            break;
        case BENCHMARK_ALGO_XOR:
            counts.one_qubit = 0;
            counts.two_qubit = allocated_qubits - 1;
            break;
        case BENCHMARK_ALGO_RANDOM:
            (void) depth;
            counts.one_qubit = -1;
            counts.two_qubit = -1;
            break;
        case BENCHMARK_ALGO_UNKNOWN:
            break;
    }

    return counts;
}

bool gate_counts_match(GateCounts actual, GateCounts expected) {
    return (expected.one_qubit < 0 || actual.one_qubit == expected.one_qubit) &&
           (expected.two_qubit < 0 || actual.two_qubit == expected.two_qubit);
}
