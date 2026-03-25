import os
import subprocess
import csv
import sys
import re

# --- Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXEC_PATH = os.path.join(SCRIPT_DIR, "../../bin/quest_runner")
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "roofline_data.csv")

ALGORITHMS = ["BB84", "BV", "EDC", "XOR"]
QUBITS = 26 # 2GB State Vector - definitely hits RAM

TIME_REGEX = re.compile(r"Execution Time \(Comp\.\):\s+([0-9\.]+)\s+seconds")

def collect_roofline_data():
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
                EXEC_PATH, "--algo", algo, "--qubits", str(QUBITS)
            ]
            
            try:
                result = subprocess.run(perf_cmd, capture_output=True, text=True, check=True)
                
                # 1. Parse Compute Time from QuEST Output
                exec_time = 0.0
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