#ifndef CIRCUIT_MANIFEST_H
#define CIRCUIT_MANIFEST_H

#include <stdbool.h>
#include <stddef.h>
#include <quest.h>

typedef struct {
    long one_qubit;
    long two_qubit;
} GateCounts;

typedef enum {
    BENCHMARK_ALGO_UNKNOWN = 0,
    BENCHMARK_ALGO_BB84,
    BENCHMARK_ALGO_BV,
    BENCHMARK_ALGO_EDC,
    BENCHMARK_ALGO_HS,
    BENCHMARK_ALGO_QRNG,
    BENCHMARK_ALGO_XOR,
    BENCHMARK_ALGO_RANDOM
} BenchmarkAlgo;

typedef struct {
    BenchmarkAlgo id;
    const char* cli_name;
    const char* paper_name;
    const char* result_label;
    const char* qubit_formula;
    const char* paper_oneq_formula;
    const char* paper_twoq_formula;
    const char* implementation_oneq_formula;
    const char* implementation_twoq_formula;
    const char* workload_kind;
} CircuitManifestEntry;

BenchmarkAlgo parse_benchmark_algo(const char* name);
const CircuitManifestEntry* get_circuit_manifest(BenchmarkAlgo algo);
const char* benchmark_algo_name(BenchmarkAlgo algo);
int compute_qubit_layout(
    BenchmarkAlgo algo,
    int allocated_qubits_arg,
    int logical_qubits_arg,
    int* input_qubits,
    int* allocated_qubits,
    char* error,
    size_t error_len
);
GateCounts expected_gate_counts(
    BenchmarkAlgo algo,
    int input_qubits,
    int allocated_qubits,
    int depth
);
bool gate_counts_match(GateCounts actual, GateCounts expected);

GateCounts build_bb84(Qureg qubits, int n);
GateCounts build_bv(Qureg qubits, int n);
GateCounts build_edc(Qureg qubits, int n);
GateCounts build_hs(Qureg qubits, int n);
GateCounts build_qrng(Qureg qubits, int n);
GateCounts build_xor(Qureg qubits, int n);
GateCounts build_random(Qureg qubits, int n, int depth);

#endif
