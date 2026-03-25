import os
import subprocess
import re
import csv
import sys
import statistics

# --- Configuration ---
# 1. Get the absolute path of the directory where this script lives
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. Anchor our paths to that directory
EXEC_PATH = os.path.join(SCRIPT_DIR, "../../bin/quest_runner") 
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "results.csv")

# 3. Experiment Parameters
ALGORITHMS = ["BB84", "BV", "EDC", "XOR"]
QUBITS = list(range(10, 27, 2))  # [10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]

MAX_ALLOCATED_QUBITS = 30
NUM_RUNS = 10

# --- Regex Parsers ---
TIME_REGEX = re.compile(r"Execution Time \(Comp\.\):\s+([0-9\.]+)\s+seconds")
ENERGY_REGEX = re.compile(r"Energy Consumed:\s+([0-9\.]+)\s+Joules")

def run_experiments():
    if not os.path.exists(EXEC_PATH):
        print(f"[ERROR] Executable not found at {EXEC_PATH}")
        print("Please compile the project using 'make' in the root directory first.")
        sys.exit(1)

    # Open CSV in write mode
    with open(OUTPUT_FILE, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Algorithm", "Input_Qubits", "Allocated_Qubits", "Execution_Time_s", "Energy_Joules"])

        for algo in ALGORITHMS:
            print(f"\n--- Starting Algorithm: {algo} ---")
            for n in QUBITS:
                # 1. Calculate actual memory footprint
                allocated = n * 2 if algo == "HS" else n
                
                # 2. Memory Guard
                if allocated > MAX_ALLOCATED_QUBITS:
                    print(f"[{algo} n={n}] SKIPPED: Requires {allocated} qubits (Exceeds {MAX_ALLOCATED_QUBITS} limit).")
                    continue

                print(f"Running {algo} (n={n}, runs={NUM_RUNS})... ", end="", flush=True)

                times = []
                energies = []

                # 3. Execute the C engine multiple times for statistical smoothing
                for run_idx in range(NUM_RUNS):
                    cmd = [EXEC_PATH, "--algo", algo, "--qubits", str(n)]
                    
                    try:
                        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                        output = result.stdout

                        # Extract metrics using Regex
                        time_match = TIME_REGEX.search(output)
                        energy_match = ENERGY_REGEX.search(output)

                        if time_match and energy_match:
                            times.append(float(time_match.group(1)))
                            energies.append(float(energy_match.group(1)))
                            
                    except subprocess.CalledProcessError:
                        # Silently skip crashed individual runs to keep the loop going
                        pass 
                    except Exception:
                        pass
                
                # 4. Calculate Medians and write to CSV
                if len(times) > 0:
                    med_time = statistics.median(times)
                    med_energy = statistics.median(energies)
                    
                    # Write immediately to disk, formatting to 6 decimal places
                    writer.writerow([algo, n, allocated, f"{med_time:.6f}", f"{med_energy:.6f}"])
                    print(f"Done. Median: {med_time:.6f}s | {med_energy:.6f}J")
                else:
                    print("FAILED completely across all runs.")

if __name__ == "__main__":
    print("==========================================")
    print(" QuEST CPU Baseline Profiler (Median Smoothed)")
    print("==========================================")
    run_experiments()
    print(f"\nData successfully saved to {OUTPUT_FILE}")