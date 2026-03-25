import os
import pandas as pd
import matplotlib.pyplot as plt

# --- Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILE = os.path.join(SCRIPT_DIR, "results.csv")
OUTPUT_IMAGE = os.path.join(SCRIPT_DIR, "scaling_results.png")

def generate_plots():
    if not os.path.exists(CSV_FILE):
        print(f"[ERROR] Could not find {CSV_FILE}. Run run_experiments.py first.")
        return

    # 1. Read the data
    df = pd.read_csv(CSV_FILE)

    # 2. Create a high-quality scientific figure with two side-by-side plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Standard shapes for different lines to make it colorblind-friendly
    markers = {'BB84': 'o', 'BV': 's', 'EDC': '^', 'HS': 'D', 'XOR': 'v'}
    
    for algo in df['Algorithm'].unique():
        subset = df[df['Algorithm'] == algo]
        
        # Plot 1: Execution Time
        ax1.plot(subset['Input_Qubits'], subset['Execution_Time_s'], 
                 marker=markers.get(algo, 'o'), label=algo, linewidth=2, markersize=8)
        
        # Plot 2: Energy Consumption
        # Note: We filter out exactly 0.0 Joule readings (like your 12-qubit XOR/BV) 
        # because log(0) is mathematically undefined and will break the chart.
        subset_energy = subset[subset['Energy_Joules'] > 0]
        if not subset_energy.empty:
            ax2.plot(subset_energy['Input_Qubits'], subset_energy['Energy_Joules'], 
                     marker=markers.get(algo, 'o'), label=algo, linewidth=2, markersize=8)

    # --- Formatting Plot 1: Time ---
    ax1.set_title("CPU Memory Wall: Execution Time vs. Qubits", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Input Qubits ($n$)", fontsize=12)
    ax1.set_ylabel("Execution Time (Seconds)", fontsize=12)
    ax1.set_yscale('log') # Logarithmic Y-axis reveals exponential scaling
    ax1.grid(True, which="both", ls="--", alpha=0.6)
    ax1.legend(title="Algorithm", fontsize=10)

    # --- Formatting Plot 2: Energy ---
    ax2.set_title("Power Bottleneck: Energy Consumed vs. Qubits", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Input Qubits ($n$)", fontsize=12)
    ax2.set_ylabel("Energy Consumed (Joules)", fontsize=12)
    ax2.set_yscale('log') # Logarithmic Y-axis here as well
    ax2.grid(True, which="both", ls="--", alpha=0.6)
    ax2.legend(title="Algorithm", fontsize=10)

    # 3. Final layout adjustments and save
    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE, dpi=300, bbox_inches='tight')
    print(f"\nSuccess! High-resolution plots saved to:\n-> {OUTPUT_IMAGE}")
    
    # Optionally display it on your screen if your Linux environment supports GUI
    # plt.show() 

if __name__ == "__main__":
    generate_plots()