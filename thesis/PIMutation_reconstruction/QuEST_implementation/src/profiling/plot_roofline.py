import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import subprocess

# --- Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILE = os.path.join(SCRIPT_DIR, "roofline_data.csv")

def get_cpu_limits():
    # Detect Cores
    cores = int(subprocess.check_output("nproc", shell=True))
    # Detect Max Freq (fallback to 3.5 if detection fails)
    try:
        with open("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq", "r") as f:
            ghz = int(f.read().strip()) / 1000000.0
    except:
        ghz = 3.5 
    
    # 32 DP FLOPs/cycle for AVX2/AMD Zen
    peak_perf = cores * ghz * 32
    # Standard sustainable bandwidth estimate for dual-channel RAM
    peak_bw = 40.0 
    
    return peak_perf, peak_bw

PEAK_PERF, PEAK_BANDWIDTH = get_cpu_limits()

def plot_roofline():
    if not os.path.exists(CSV_FILE):
        print("CSV not found.")
        return
        
    df = pd.read_csv(CSV_FILE)
    plt.figure(figsize=(12, 8))
    
    # 1. Generate the Roofline
    # We expand the range to 10^3 so we can see the Ridge Point and the Flat Roof
    x_vals = np.logspace(-2, 3, 2000)
    bw_roof = PEAK_BANDWIDTH * x_vals
    y_roof = np.minimum(bw_roof, PEAK_PERF)
    
    plt.plot(x_vals, y_roof, color='black', linewidth=3, label='Hardware Limit', zorder=1)
    
    # 2. The Ridge Point (The "Turn" in the roof)
    ridge_x = PEAK_PERF / PEAK_BANDWIDTH
    plt.axvline(x=ridge_x, color='red', linestyle='--', alpha=0.6, label=f'Ridge Point ({ridge_x:.2f} FLOP/Byte)')

    # 3. Plot Algorithms
    markers = ['o', 's', '^', 'D']
    for idx, row in df.iterrows():
        plt.scatter(row['Arithmetic_Intensity'], row['GIPS'], 
                    s=250, marker=markers[idx % 4], edgecolor='black', 
                    linewidth=1.5, zorder=10, label=row['Algorithm'])

    # 4. Corrected Text Annotations
    # Placed relative to the Ridge Point
    plt.text(ridge_x * 0.04, PEAK_PERF * 0.02, "MEMORY-BOUND", fontsize=14, fontweight='bold', color='gray', alpha=0.7)
    plt.text(ridge_x * 2, PEAK_PERF * 0.1, "COMPUTE-BOUND", fontsize=14, fontweight='bold', color='gray', alpha=0.7)

    # 5. Formatting
    plt.xscale('log')
    plt.yscale('log')
    
    # Range: Show from 0.01 to 1000 to capture the whole architecture
    plt.xlim(0.01, 1000) 
    plt.ylim(0.01, PEAK_PERF * 5) 
    
    plt.xlabel("Arithmetic Intensity (FLOP/Byte)", fontsize=14)
    plt.ylabel("Performance (GFLOPS)", fontsize=14)
    plt.title("Roofline Analysis: The Quantum Memory Wall", fontsize=18, fontweight='bold', pad=20)
    
    plt.grid(True, which="both", ls="--", alpha=0.3)
    
    # Move legend to Upper Left so it doesn't block the data points
    plt.legend(loc='upper left', fontsize=11, frameon=True, shadow=True, borderpad=1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(SCRIPT_DIR, "roofline_plot.png"), dpi=300)
    print(f"Final Roofline generated. Ridge point is at {ridge_x:.2f}")

if __name__ == "__main__":
    plot_roofline()