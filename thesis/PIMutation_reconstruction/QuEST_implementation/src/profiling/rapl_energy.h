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

#define RAPL_PATH "/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj"

// 1. Encapsulate the state
typedef struct {
    uint64_t start_uj;
    bool is_available;
} RaplState;

// 2. Pure function to initialize and capture the start state
static inline RaplState start_energy_profiling() {
    RaplState state = {0, false};
    
    FILE *file = fopen(RAPL_PATH, "r");
    if (file) {
        state.is_available = true;
        if (fscanf(file, "%lu", &state.start_uj) != 1) {
            state.start_uj = 0;
        }
        fclose(file);
    } else {
        printf("[Telemetry] WARNING: RAPL interface not found at %s.\n", RAPL_PATH);
    }
    
    return state;
}

// 3. Pure function to calculate the delta based on the passed state
static inline double stop_energy_profiling(RaplState state) {
    if (!state.is_available) return 0.0;
    
    FILE *file = fopen(RAPL_PATH, "r");
    if (!file) return 0.0;
    
    uint64_t end_energy_uj = 0;
    if (fscanf(file, "%lu", &end_energy_uj) != 1) {
        end_energy_uj = 0;
    }
    fclose(file);
    
    uint64_t delta_uj = end_energy_uj - state.start_uj;
    return (double)delta_uj / 1000000.0;
}

#endif // RAPL_ENERGY_H