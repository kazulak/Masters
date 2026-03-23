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

static bool rapl_available = false;
static uint64_t start_energy_uj = 0;

static inline void init_energy_profiler() {
    FILE *file = fopen(RAPL_PATH, "r");
    if (file) {
        rapl_available = true;
        fclose(file);
        printf("[Telemetry] RAPL Energy interface detected.\n");
    } else {
        rapl_available = false;
        printf("[Telemetry] WARNING: RAPL interface not found at %s.\n", RAPL_PATH);
        printf("[Telemetry] Energy readings will report 0.0 Joules.\n");
        printf("[Telemetry] Hint: You may need 'sudo' or to load the appropriate kernel module.\n");
    }
}

static inline uint64_t read_energy_uj() {
    if (!rapl_available) return 0;
    
    FILE *file = fopen(RAPL_PATH, "r");
    if (!file) return 0;
    
    uint64_t energy;
    if (fscanf(file, "%lu", &energy) != 1) {
        energy = 0;
    }
    fclose(file);
    return energy;
}

static inline void start_energy_profiling() {
    start_energy_uj = read_energy_uj();
}

static inline double stop_energy_profiling() {
    uint64_t end_energy_uj = read_energy_uj();
    
    // Handle unsigned integer wrap-around (rare but possible with MSRs)
    uint64_t delta_uj = end_energy_uj - start_energy_uj;
    
    // Convert microjoules to Joules
    return (double)delta_uj / 1000000.0;
}

#endif // RAPL_ENERGY_H