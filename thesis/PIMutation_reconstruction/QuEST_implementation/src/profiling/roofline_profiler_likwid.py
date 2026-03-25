#!/usr/bin/env python3
import os
import subprocess
import csv
import re
import sys

# ====================== CONFIG ======================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXEC_PATH = os.path.join(SCRIPT_DIR, "../../bin/quest_runner")
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "roofline_data_perf.csv")

ALGORITHMS = ["BB84", "BV", "EDC", "HS", "QRNG", "XOR"]
QUBITS = 27          # 2 GiB state vector → forces DRAM traffic (same regime as paper)
CORES = "0-5"        # Ryzen 5 5600 has 6 cores

# === THESE ARE YOUR MACHINE LIMITS (update after running STREAM below) ===
PEAK_GFLOPS_DP = 422.0      # theoretical for Ryzen 5 5600 (6c * 16 DP FLOP/cycle * ~4.4 GHz)
PEAK_BW_GB_S = 52.0         # ←←← REPLACE WITH YOUR STREAM Triad result (see instructions)
RIDGE_FLOP_PER_BYTE = PEAK_GFLOPS_DP / PEAK_BW_GB_S

TIME_REGEX = re.compile(r"Execution Time \(Comp\.\):\s+([0-9\.]+)\s+seconds")

def run_perf(algo):
    print(f"Profiling {algo} (QUBITS={QUBITS})...")

    perf_cmd = [
        "sudo", "taskset", "-c", CORES,
        "perf", "stat", "-x", ",", "-e",
        "instructions,fp_ret_sse_avx_ops.all,l3_misses",
        EXEC_PATH, "--algo", algo, "--qubits", str(QUBITS)
    ]

    try:
        result = subprocess.run(perf_cmd, capture_output=True, text=True, check=True)

        # 1. Parse QuEST compute time
        exec_time = 0.0
        time_match = TIME_REGEX.search(result.stdout)
        if time_match:
            exec_time = float(time_match.group(1))

        # 2. Parse perf counters from stderr
        lines = result.stderr.strip().split('\n')
        instructions = 0
        flops = 0
        l3_misses = 0

        for line in lines:
            parts = line.split(',')
            if len(parts) < 3 or not parts[0].replace('.', '', 1).replace('e+', '', 1).replace('-', '', 1).isdigit():
                continue
            val = float(parts[0])
            event = parts[2]

            if "fp_ret_sse_avx_ops.all" in event:
                flops = int(val)
            elif "l3_misses" in event:
                l3_misses = int(val)

        if flops == 0 or l3_misses == 0:
            print(f" -> WARNING: Zero counters for {algo}. Check 'perf list | grep -E \"fp|l3\"'")

        bytes_from_ram = l3_misses * 64
        ai = flops / bytes_from_ram if bytes_from_ram > 0 else 0.0
        gfops = (flops / exec_time / 1e9) if exec_time > 0 else 0.0

        print(f" -> AI: {ai:.3f} FLOP/Byte | Perf: {gfops:.1f} GFLOPS | Time: {exec_time:.3f}s | L3 misses: {l3_misses:,}")

        return {
            "Algorithm": algo,
            "Time_s": exec_time,
            "L3_Misses": l3_misses,
            "DRAM_Bytes": bytes_from_ram,
            "FLOPS": flops,
            "Arithmetic_Intensity": ai,
            "GFLOPS": gfops,
            "Ridge_FLOP_Byte": RIDGE_FLOP_PER_BYTE
        }

    except subprocess.CalledProcessError as e:
        print(f" -> ERROR: {e.stderr.strip()}")
        print("   Try: sudo sysctl -w kernel.perf_event_paranoid=-1")
        return None

def main():
    with open(OUTPUT_FILE, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["Algorithm","Time_s","L3_Misses","DRAM_Bytes","FLOPS","Arithmetic_Intensity","GFLOPS","Ridge_FLOP_Byte"])
        writer.writeheader()
        for algo in ALGORITHMS:
            data = run_perf(algo)
            if data:
                writer.writerow(data)

    print(f"\n✅ Done! Data saved to {OUTPUT_FILE}")
    print(f"   Realistic ridge point (before STREAM measurement): {RIDGE_FLOP_PER_BYTE:.2f} FLOP/Byte")
    print("   Update PEAK_BW_GB_S after running STREAM (see below) and re-plot.")

if __name__ == "__main__":
    main()