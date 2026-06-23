import os
import subprocess
import csv
import sys
import re
from datetime import datetime
from pathlib import Path

# --- Configuration ---
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parents[1]
EXEC_PATH = ROOT_DIR / "bin" / "quest_runner"
RUN_DIR = ROOT_DIR / "runs" / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_roofline_perf"
OUTPUT_FILE = RUN_DIR / "raw" / "roofline_data.csv"

ALGORITHMS = ["BB84", "BV", "EDC", "HS", "QRNG", "XOR"]
QUBITS = 18 # 2GB State Vector - definitely hits RAM

TIME_REGEX = re.compile(r"Execution Time \(Comp\.\):\s+([0-9\.]+)\s+seconds")

def collect_roofline_data():
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Algorithm", "Time_s", "Cache_Misses", "Instructions", "Bytes_from_RAM", "Arithmetic_Intensity", "GIPS"])

        for algo in ALGORITHMS:
            print(f"Profiling {algo} (Generic Hardware Counters)...")
            
            # Check if we are on AMD or Intel to pick the right FLOP counter
            cpu_vendor = subprocess.check_output("grep vendor_id /proc/cpuinfo | head -n 1", shell=True).decode()
            
            if "AuthenticAMD" in cpu_vendor:
                # AMD Zen FP counter (may require sudo)
                fp_event = "fp_ret_sse_avx_ops.all" 
            else:
                # Intel FP counter
                fp_event = "fp_arith_inst_retired.scalar_double"

            perf_cmd = [
                "sudo", "perf", "stat", "-x", ",", "-e", f"instructions,{fp_event},cache-misses",
                str(EXEC_PATH), "--algo", algo, "--qubits", str(QUBITS), "--json"
            ]
            
            try:
                result = subprocess.run(perf_cmd, capture_output=True, text=True, check=True)
                
                # 1. Parse Compute Time from QuEST Output
                exec_time = 0.0
                try:
                    import json
                    exec_time = float(json.loads(result.stdout.strip())["time_s"])
                except Exception:
                    time_match = TIME_REGEX.search(result.stdout)
                    if time_match:
                        exec_time = float(time_match.group(1))
                
                # 2. Parse Perf Metrics from Stderr
                lines = result.stderr.strip().split('\n')
                instructions = 0
                flops = 0
                cache_misses = 0
                
                for line in lines:
                    parts = line.split(',')
                    if len(parts) < 3: continue
                    val = parts[0]
                    event = parts[2]
                    
                    if val.replace('.', '', 1).isdigit():
                        if fp_event in event:
                            flops = int(float(val))
                        elif 'cache-misses' in event:
                            cache_misses = int(float(val))

                if cache_misses == 0:
                    print(f" -> WARNING: 0 cache misses detected for {algo}. Check 'perf list'.")

                # REAL Arithmetic Intensity (FLOP/Byte)
                bytes_from_ram = cache_misses * 64 
                ai = flops / bytes_from_ram if bytes_from_ram > 0 else 0
                gips = (flops / exec_time) / 1e9 if exec_time > 0 else 0
                

                writer.writerow([algo, exec_time, cache_misses, instructions, bytes_from_ram, ai, gips])
                print(f" -> AI: {ai:.4f} Ops/Byte | Perf: {gips:.2f} Giga-Ops/s")
                
            except subprocess.CalledProcessError as e:
                print(f" -> ERROR: Perf failed. Try running: sudo sysctl -w kernel.perf_event_paranoiac=-1")

if __name__ == "__main__":
    collect_roofline_data()
    print(f"Roofline data saved to {OUTPUT_FILE}")
