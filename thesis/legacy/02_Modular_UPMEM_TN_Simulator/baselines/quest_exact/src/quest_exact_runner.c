#include <complex.h>
#include <ctype.h>
#include <errno.h>
#include <math.h>
#include <omp.h>
#include <quest.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define RAPL_PATH "/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj"

typedef struct {
    uint64_t start_uj;
    bool is_available;
} RaplState;

static RaplState start_energy_profiling(void) {
    RaplState state = {0, false};
    FILE* file = fopen(RAPL_PATH, "r");
    if (file) {
        state.is_available = true;
        if (fscanf(file, "%lu", &state.start_uj) != 1) {
            state.start_uj = 0;
        }
        fclose(file);
    }
    return state;
}

static double stop_energy_profiling(RaplState state) {
    if (!state.is_available) return 0.0;
    FILE* file = fopen(RAPL_PATH, "r");
    if (!file) return 0.0;

    uint64_t end_uj = 0;
    if (fscanf(file, "%lu", &end_uj) != 1) {
        end_uj = 0;
    }
    fclose(file);
    return (double)(end_uj - state.start_uj) / 1000000.0;
}

static void usage(const char* prog) {
    fprintf(stderr,
            "Usage: %s (--circuit NAME [--qubits N] | --qasm PATH) --output PATH\n"
            "Circuits: bell_2q, ghz_4q, ghz_chain\n",
            prog);
}

static int parse_wire(const char* text) {
    const char* open = strchr(text, '[');
    const char* close = strchr(text, ']');
    if (!open || !close || close <= open + 1) return -1;
    return atoi(open + 1);
}

static void trim(char* s) {
    char* p = s;
    while (isspace((unsigned char)*p)) p++;
    if (p != s) memmove(s, p, strlen(p) + 1);

    size_t len = strlen(s);
    while (len > 0 && isspace((unsigned char)s[len - 1])) {
        s[len - 1] = '\0';
        len--;
    }
}

static int parse_qasm_qubits(const char* path) {
    FILE* file = fopen(path, "r");
    if (!file) return -1;
    char line[512];
    int qubits = -1;
    while (fgets(line, sizeof(line), file)) {
        trim(line);
        if (strncmp(line, "qreg", 4) == 0) {
            char* open = strchr(line, '[');
            qubits = open ? atoi(open + 1) : -1;
            break;
        }
    }
    fclose(file);
    return qubits;
}

static int apply_qasm(Qureg qureg, const char* path) {
    FILE* file = fopen(path, "r");
    if (!file) return 1;

    char line[512];
    while (fgets(line, sizeof(line), file)) {
        char* comment = strstr(line, "//");
        if (comment) *comment = '\0';
        trim(line);
        if (line[0] == '\0' || strncmp(line, "OPENQASM", 8) == 0 ||
            strncmp(line, "include", 7) == 0 || strncmp(line, "qreg", 4) == 0) {
            continue;
        }

        if (strncmp(line, "h ", 2) == 0) {
            int q = parse_wire(line);
            if (q < 0) return 2;
            applyHadamard(qureg, q);
        } else if (strncmp(line, "x ", 2) == 0) {
            int q = parse_wire(line);
            if (q < 0) return 2;
            applyPauliX(qureg, q);
        } else if (strncmp(line, "z ", 2) == 0) {
            int q = parse_wire(line);
            if (q < 0) return 2;
            applyPauliZ(qureg, q);
        } else if (strncmp(line, "s ", 2) == 0) {
            int q = parse_wire(line);
            if (q < 0) return 2;
            applyS(qureg, q);
        } else if (strncmp(line, "t ", 2) == 0) {
            int q = parse_wire(line);
            if (q < 0) return 2;
            applyT(qureg, q);
        } else if (strncmp(line, "cx ", 3) == 0) {
            char* comma = strchr(line, ',');
            if (!comma) return 2;
            int ctrl = parse_wire(line);
            int targ = parse_wire(comma + 1);
            if (ctrl < 0 || targ < 0) return 2;
            applyControlledPauliX(qureg, ctrl, targ);
        } else {
            fclose(file);
            fprintf(stderr, "Unsupported QASM line: %s\n", line);
            return 3;
        }
    }

    fclose(file);
    return 0;
}

static void apply_builtin(Qureg qureg, const char* circuit, int n_qubits) {
    if (strcmp(circuit, "bell_2q") == 0) {
        applyHadamard(qureg, 0);
        applyControlledPauliX(qureg, 0, 1);
    } else if (strcmp(circuit, "ghz_4q") == 0 || strcmp(circuit, "ghz_chain") == 0) {
        applyHadamard(qureg, 0);
        for (int i = 0; i < n_qubits - 1; i++) {
            applyControlledPauliX(qureg, i, i + 1);
        }
    }
}

static int write_amplitudes(Qureg qureg, int n_qubits, const char* output_path) {
    qindex n_amps = ((qindex)1) << n_qubits;
    qcomp* amps = malloc((size_t)n_amps * sizeof(qcomp));
    if (!amps) return 1;

    getQuregAmps(amps, qureg, 0, n_amps);
    FILE* out = fopen(output_path, "wb");
    if (!out) {
        free(amps);
        return 2;
    }

    for (qindex i = 0; i < n_amps; i++) {
        double re = creal(amps[i]);
        double im = cimag(amps[i]);
        fwrite(&re, sizeof(double), 1, out);
        fwrite(&im, sizeof(double), 1, out);
    }

    fclose(out);
    free(amps);
    return 0;
}

int main(int argc, char** argv) {
    const char* circuit = NULL;
    const char* qasm = NULL;
    const char* output = NULL;
    int n_qubits = 0;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--circuit") == 0 && i + 1 < argc) {
            circuit = argv[++i];
        } else if (strcmp(argv[i], "--qasm") == 0 && i + 1 < argc) {
            qasm = argv[++i];
        } else if (strcmp(argv[i], "--qubits") == 0 && i + 1 < argc) {
            n_qubits = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
            output = argv[++i];
        }
    }

    if (!output || ((circuit == NULL) == (qasm == NULL))) {
        usage(argv[0]);
        return 1;
    }

    if (qasm) {
        n_qubits = parse_qasm_qubits(qasm);
    } else if (strcmp(circuit, "bell_2q") == 0) {
        n_qubits = 2;
    } else if (strcmp(circuit, "ghz_4q") == 0) {
        n_qubits = 4;
    }

    if (n_qubits <= 0 || n_qubits > 30) {
        fprintf(stderr, "Unsupported qubit count: %d\n", n_qubits);
        return 2;
    }

    initQuESTEnv();
    Qureg qureg = createQureg(n_qubits);
    initZeroState(qureg);

    RaplState energy = start_energy_profiling();
    double start = omp_get_wtime();

    int apply_status = 0;
    if (qasm) {
        apply_status = apply_qasm(qureg, qasm);
    } else {
        apply_builtin(qureg, circuit, n_qubits);
    }

    double end = omp_get_wtime();
    double joules = stop_energy_profiling(energy);

    int write_status = apply_status == 0 ? write_amplitudes(qureg, n_qubits, output) : apply_status;

    destroyQureg(qureg);
    finalizeQuESTEnv();

    if (write_status != 0) {
        fprintf(stderr, "QuEST baseline failed with status %d\n", write_status);
        return write_status;
    }

    printf("QUEST_RESULT {\"status\":\"ok\",\"execution_seconds\":%.12f,"
           "\"energy_joules\":%.12f,\"energy_source\":\"%s\",\"n_qubits\":%d}\n",
           end - start,
           joules,
           energy.is_available ? "rapl" : "rapl_unavailable",
           n_qubits);
    return 0;
}

