/*
 * SCIENTIFIC OBJECTIVE: Hardware-Agnostic CPU Energy Telemetry
 *
 * DESCRIPTION:
 * Measures Joules consumed during compute phases. Designed to work on 
 * both Intel (Cluster) and AMD (Local) CPUs by hooking into the Linux 
 * powercap framework. Includes fail-safes for missing kernel modules.
 *
 * TO DO:
 * 1. Implement `init_energy_profiler()`: Probe for the existence of 
 * `/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj`. Set a global 
 * flag `rapl_available`. If not found, print a warning for local testing.
 * 2. Implement `read_energy_uj()`: Safely read the microjoule value if 
 * `rapl_available` is true; otherwise, return 0.
 * 3. Implement `start_energy_profiling()`: Capture the baseline energy state.
 * 4. Implement `stop_energy_profiling()`: Capture final state, handle potential 
 * counter overflow (MSR wrap-around), and return delta in Joules.
 */

 /*
 * SCIENTIFIC OBJECTIVE: The SOTA CPU Baseline Profiler
 */

/*
 * SCIENTIFIC OBJECTIVE: Hardware-Agnostic CPU Energy Telemetry
 * Works for Intel (Cluster) and AMD (Local, via zenpower/amd_energy modules).
 */

#ifndef RAPL_ENERGY_H
#define RAPL_ENERGY_H

#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <inttypes.h>

#define RAPL_PATH "/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj"
#define RAPL_MAX_RANGE_PATH "/sys/class/powercap/intel-rapl/intel-rapl:0/max_energy_range_uj"

// 1. Encapsulate the state
typedef struct {
    uint64_t start_uj;
    uint64_t max_range_uj;
    bool is_available;
} RaplState;

static inline uint64_t read_rapl_uj(const char* path, bool* ok) {
    FILE *file = fopen(path, "r");
    if (!file) {
        if (ok) *ok = false;
        return 0;
    }

    uint64_t value = 0;
    if (fscanf(file, "%" SCNu64, &value) != 1) {
        if (ok) *ok = false;
        fclose(file);
        return 0;
    }

    fclose(file);
    if (ok) *ok = true;
    return value;
}

// 2. Pure function to initialize and capture the start state
static inline RaplState start_energy_profiling() {
    RaplState state = {0, 0, false};

    bool ok = false;
    state.start_uj = read_rapl_uj(RAPL_PATH, &ok);
    state.is_available = ok;
    if (ok) {
        bool max_ok = false;
        state.max_range_uj = read_rapl_uj(RAPL_MAX_RANGE_PATH, &max_ok);
        if (!max_ok) {
            state.max_range_uj = 0;
        }
    }
    
    return state;
}

// 3. Pure function to calculate the delta based on the passed state
static inline double stop_energy_profiling(RaplState state) {
    if (!state.is_available) return 0.0;

    bool ok = false;
    uint64_t end_energy_uj = read_rapl_uj(RAPL_PATH, &ok);
    if (!ok) return 0.0;

    uint64_t delta_uj = 0;
    if (end_energy_uj >= state.start_uj) {
        delta_uj = end_energy_uj - state.start_uj;
    } else if (state.max_range_uj > 0) {
        delta_uj = (state.max_range_uj - state.start_uj) + end_energy_uj;
    }

    return (double)delta_uj / 1000000.0;
}

static inline const char* energy_source_name(RaplState state) {
    return state.is_available ? "rapl_measured" : "unavailable";
}

#endif // RAPL_ENERGY_H
