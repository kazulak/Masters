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
#include <stdbool.h>
#include <complex.h>
#include <omp.h>
#include <quest.h>
#include "circuits/circuit_manifest.h"
#include "profiling/rapl_energy.h"
#include "verification/verify_runner.h"

void print_usage(const char* prog_name) {
    printf("Usage for Profiling:  %s --algo <NAME> --qubits <ALLOCATED_N> [--depth <D>] [--json]\n", prog_name);
    printf("                    or %s --algo HS --logical-qubits <N> [--json]\n", prog_name);
    printf("                    optional comparable-output flags: --dump-state-json <PATH> --max-output-amplitudes <N>\n");
    printf("Usage for Testing:    %s --verify <MODE>\n", prog_name);
    printf("Algorithms:           BB84, BV, EDC, HS, QRNG, XOR, RANDOM\n");
    printf("Verify Modes:         FULL, BASE, or comma-separated (e.g., BV,BB84)\n");
    printf("Example (Profile):    sudo %s --algo BV --qubits 26\n", prog_name);
    printf("Example (HS):         sudo %s --algo HS --logical-qubits 13\n", prog_name);
    printf("Example (Test):       %s --verify FULL\n", prog_name);
}

static void print_json_escaped(const char* value) {
    if (value == NULL) {
        printf("null");
        return;
    }

    putchar('"');
    for (const char* p = value; *p != '\0'; p++) {
        switch (*p) {
            case '\\':
                printf("\\\\");
                break;
            case '"':
                printf("\\\"");
                break;
            case '\n':
                printf("\\n");
                break;
            case '\r':
                printf("\\r");
                break;
            case '\t':
                printf("\\t");
                break;
            default:
                putchar(*p);
                break;
        }
    }
    putchar('"');
}

static void fprint_json_escaped(FILE* handle, const char* value) {
    if (value == NULL) {
        fprintf(handle, "null");
        return;
    }

    fputc('"', handle);
    for (const char* p = value; *p != '\0'; p++) {
        switch (*p) {
            case '\\':
                fprintf(handle, "\\\\");
                break;
            case '"':
                fprintf(handle, "\\\"");
                break;
            case '\n':
                fprintf(handle, "\\n");
                break;
            case '\r':
                fprintf(handle, "\\r");
                break;
            case '\t':
                fprintf(handle, "\\t");
                break;
            default:
                fputc(*p, handle);
                break;
        }
    }
    fputc('"', handle);
}

static void print_json_result(
    const char* algo_name,
    BenchmarkAlgo algo_id,
    int input_qubits,
    int allocated_qubits,
    int depth,
    int threads,
    double time_s,
    bool has_energy,
    double energy_joules,
    const char* energy_source,
    const char* status,
    const char* error,
    GateCounts counts
) {
    const CircuitManifestEntry* manifest = get_circuit_manifest(algo_id);

    printf("{");
    printf("\"algo\":");
    print_json_escaped(algo_name);
    printf(",\"paper_algo\":");
    print_json_escaped(manifest ? manifest->paper_name : NULL);
    printf(",\"result_label\":");
    print_json_escaped(manifest ? manifest->result_label : NULL);
    printf(",\"input_qubits\":%d", input_qubits);
    printf(",\"allocated_qubits\":%d", allocated_qubits);
    printf(",\"depth\":%d", depth);
    printf(",\"threads\":%d", threads);
    printf(",\"time_s\":%.9f", time_s);
    printf(",\"energy_joules\":");
    if (has_energy) {
        printf("%.9f", energy_joules);
    } else {
        printf("null");
    }
    printf(",\"energy_source\":");
    print_json_escaped(energy_source);
    printf(",\"status\":");
    print_json_escaped(status);
    printf(",\"error\":");
    print_json_escaped(error);
    printf(",\"one_qubit_gates\":%ld", counts.one_qubit);
    printf(",\"two_qubit_gates\":%ld", counts.two_qubit);
    printf(",\"quest_version\":");
    print_json_escaped(QUEST_VERSION_STRING);
    printf("}\n");
}

static GateCounts run_algorithm(BenchmarkAlgo algo, Qureg qubits, int allocated_qubits, int depth) {
    switch (algo) {
        case BENCHMARK_ALGO_BB84:
            return build_bb84(qubits, allocated_qubits);
        case BENCHMARK_ALGO_BV:
            return build_bv(qubits, allocated_qubits);
        case BENCHMARK_ALGO_EDC:
            return build_edc(qubits, allocated_qubits);
        case BENCHMARK_ALGO_HS:
            return build_hs(qubits, allocated_qubits);
        case BENCHMARK_ALGO_QRNG:
            return build_qrng(qubits, allocated_qubits);
        case BENCHMARK_ALGO_XOR:
            return build_xor(qubits, allocated_qubits);
        case BENCHMARK_ALGO_RANDOM:
            return build_random(qubits, allocated_qubits, depth);
        case BENCHMARK_ALGO_UNKNOWN:
            break;
    }

    GateCounts invalid = {-1, -1};
    return invalid;
}

static int dump_state_json(const char* path, Qureg qubits, int allocated_qubits, long long max_output_amplitudes, char* error, size_t error_len) {
    if (path == NULL) {
        return 1;
    }
    if (allocated_qubits < 0 || allocated_qubits >= 62) {
        snprintf(error, error_len, "Allocated qubit count is outside state dump range.");
        return 0;
    }
    long long amplitude_count = 1LL << allocated_qubits;
    if (max_output_amplitudes <= 0 || amplitude_count > max_output_amplitudes) {
        snprintf(error, error_len, "State dump amplitude cap exceeded.");
        return 0;
    }

    FILE* handle = fopen(path, "w");
    if (handle == NULL) {
        snprintf(error, error_len, "Could not open state dump path.");
        return 0;
    }

    fprintf(handle, "{");
    fprintf(handle, "\"schema_version\":\"quest_state_dump_v1\"");
    fprintf(handle, ",\"basis_order\":\"quest_little_endian_integer_index\"");
    fprintf(handle, ",\"allocated_qubits\":%d", allocated_qubits);
    fprintf(handle, ",\"amplitude_count\":%lld", amplitude_count);
    fprintf(handle, ",\"quest_version\":");
    fprint_json_escaped(handle, QUEST_VERSION_STRING);
    fprintf(handle, ",\"real\":[");
    for (long long index = 0; index < amplitude_count; index++) {
        qcomp amp = getQuregAmp(qubits, (qindex) index);
        fprintf(handle, "%s%.17g", index == 0 ? "" : ",", (double) creal(amp));
    }
    fprintf(handle, "],\"imag\":[");
    for (long long index = 0; index < amplitude_count; index++) {
        qcomp amp = getQuregAmp(qubits, (qindex) index);
        fprintf(handle, "%s%.17g", index == 0 ? "" : ",", (double) cimag(amp));
    }
    fprintf(handle, "]}");
    fclose(handle);
    return 1;
}

int main(int argc, char** argv) {
    char* algo_arg = NULL;
    int allocated_qubits_arg = 0;
    int logical_qubits_arg = 0;
    int depth_arg = 0;
    char* verify_mode = NULL;
    char* state_dump_path = NULL;
    long long max_output_amplitudes = 4096;
    bool json_output = false;
    char cli_error[256] = {0};

    // 1. Parse CLI Arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--algo") == 0 && i + 1 < argc) {
            algo_arg = argv[++i];
        } else if (strcmp(argv[i], "--qubits") == 0 && i + 1 < argc) {
            allocated_qubits_arg = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--logical-qubits") == 0 && i + 1 < argc) {
            logical_qubits_arg = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--depth") == 0 && i + 1 < argc) {
            depth_arg = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--verify") == 0 && i + 1 < argc) {
            verify_mode = argv[++i]; 
        } else if (strcmp(argv[i], "--dump-state-json") == 0 && i + 1 < argc) {
            state_dump_path = argv[++i];
        } else if (strcmp(argv[i], "--max-output-amplitudes") == 0 && i + 1 < argc) {
            max_output_amplitudes = atoll(argv[++i]);
        } else if (strcmp(argv[i], "--json") == 0) {
            json_output = true;
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        } else {
            snprintf(cli_error, sizeof(cli_error), "Unknown or incomplete argument '%s'.", argv[i]);
            break;
        }
    }

    // 2. THE INTERCEPT (Your isolated, elegant design)
    if (verify_mode != NULL) {
        initQuESTEnv();              // Boot up just for testing
        int verify_status = run_test_suite(verify_mode); // Run tests
        finalizeQuESTEnv();          // Shut down
        return verify_status;        // Exit immediately
    }

    BenchmarkAlgo algo_id = parse_benchmark_algo(algo_arg);
    const char* normalized_algo = benchmark_algo_name(algo_id);
    int input_qubits = 0;
    int allocated_qubits = 0;
    int depth = (algo_id == BENCHMARK_ALGO_RANDOM) ? (depth_arg > 0 ? depth_arg : 10) : 0;
    int threads = omp_get_max_threads();
    GateCounts no_counts = {-1, -1};

    if (cli_error[0] != '\0') {
        if (json_output) {
            print_json_result(
                algo_arg ? algo_arg : "UNKNOWN",
                algo_id,
                0,
                0,
                depth,
                threads,
                0.0,
                false,
                0.0,
                "unavailable",
                "failed",
                cli_error,
                no_counts
            );
        } else {
            fprintf(stderr, "Error: %s\n", cli_error);
            print_usage(argv[0]);
        }
        return 1;
    }

    if (algo_id == BENCHMARK_ALGO_UNKNOWN) {
        snprintf(cli_error, sizeof(cli_error), "Unknown algorithm '%s'.", algo_arg ? algo_arg : "");
        if (json_output) {
            print_json_result(
                algo_arg ? algo_arg : "UNKNOWN",
                algo_id,
                0,
                0,
                depth,
                threads,
                0.0,
                false,
                0.0,
                "unavailable",
                "failed",
                cli_error,
                no_counts
            );
        } else {
            fprintf(stderr, "Error: %s\n", cli_error);
            print_usage(argv[0]);
        }
        return 1;
    }

    if (!compute_qubit_layout(
            algo_id,
            allocated_qubits_arg,
            logical_qubits_arg,
            &input_qubits,
            &allocated_qubits,
            cli_error,
            sizeof(cli_error)
        )) {
        if (json_output) {
            print_json_result(
                normalized_algo,
                algo_id,
                input_qubits,
                allocated_qubits,
                depth,
                threads,
                0.0,
                false,
                0.0,
                "unavailable",
                "failed",
                cli_error,
                no_counts
            );
        } else {
            fprintf(stderr, "Error: %s\n", cli_error);
            print_usage(argv[0]);
        }
        return 1;
    }

    if (allocated_qubits > 34 && !json_output) {
        printf("WARNING: %d total qubits requires massive RAM (>256GB). May OOM.\n", allocated_qubits);
    }

    // 2. Initialize QuEST Environment
    initQuESTEnv(); 
    Qureg qubits = createQureg(allocated_qubits); // No longer requires passing 'env'
    initZeroState(qubits);

    const CircuitManifestEntry* manifest = get_circuit_manifest(algo_id);
    if (!json_output) {
        printf("==========================================\n");
        printf("QuEST SOTA Baseline Runner\n");
        printf("Algorithm: %s\n", normalized_algo);
        printf("Paper Label: %s\n", manifest ? manifest->paper_name : "UNKNOWN");
        printf("Result Label: %s\n", manifest ? manifest->result_label : "UNKNOWN");
        printf("Input Qubits (logical n): %d\n", input_qubits);
        printf("Allocated Qubits: %d\n", allocated_qubits);
        printf("QuEST Version: %s\n", QUEST_VERSION_STRING);
        printf("==========================================\n");
    }

    // 4. START PROFILING BOUNDARY
    RaplState energy_state = start_energy_profiling();
    double start_time = omp_get_wtime();

    // 5. Routing Logic
    GateCounts counts = run_algorithm(algo_id, qubits, allocated_qubits, depth);

    // 6. STOP PROFILING BOUNDARY
    double end_time = omp_get_wtime();
    double joules = stop_energy_profiling(energy_state);
    double elapsed = end_time - start_time;

    GateCounts expected = expected_gate_counts(algo_id, input_qubits, allocated_qubits, depth);
    bool counts_ok = gate_counts_match(counts, expected);
    bool dump_ok = dump_state_json(state_dump_path, qubits, allocated_qubits, max_output_amplitudes, cli_error, sizeof(cli_error));
    const char* status = (counts_ok && dump_ok) ? "ok" : "failed";
    const char* error = counts_ok ? (dump_ok ? NULL : cli_error) : "Gate counts did not match circuit manifest.";

    // 7. Output Clean Metrics
    if (json_output) {
        print_json_result(
            normalized_algo,
            algo_id,
            input_qubits,
            allocated_qubits,
            depth,
            threads,
            elapsed,
            energy_state.is_available,
            joules,
            energy_source_name(energy_state),
            status,
            error,
            counts
        );
    } else {
        printf("-> Execution Time (Comp.): %f seconds\n", elapsed);
        if (energy_state.is_available) {
            printf("-> Energy Consumed:        %f Joules\n", joules);
        } else {
            printf("-> Energy Consumed:        unavailable\n");
        }
        printf("-> Energy Source:          %s\n", energy_source_name(energy_state));
        printf("-> Gate Counts:            1Q=%ld, 2Q=%ld\n", counts.one_qubit, counts.two_qubit);
        if (!counts_ok) {
            printf("-> ERROR: %s\n", error);
        }
        printf("==========================================\n");
    }

    // 8. UPDATED: v4 Cleanup for successful execution
    destroyQureg(qubits);
    finalizeQuESTEnv();

    return (counts_ok && dump_ok) ? 0 : 1;
}
