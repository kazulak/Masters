import numpy as np
import matplotlib.pyplot as plt
import matplotlib
print("DEBUG: Imports successful")
matplotlib.use('Agg')

def simulate_doppler_effect():
    print("DEBUG: Entered simulate_doppler_effect function")
    """
    Simulates and plots the relativistic Doppler effect on heartbeat intervals
    from a receding Raft leader.
    """
    # --- Simulation Constants ---
    # Proper time interval between heartbeats in the leader's reference frame (e.g., ms)
    PROPER_HEARTBEAT_INTERVAL = 50.0  
    # Speed of light for the simulation. Using a small number makes effects visible.
    C_SIM = 100.0  

    # --- Velocity Range ---
    # Generate a range of velocities from 0 to 99.9% of the speed of light.
    velocities = np.linspace(0, 0.999 * C_SIM, 500)

    # --- Relativistic Calculation ---
    # Calculate the observed heartbeat interval at the follower's frame.
    # Formula derived from special relativity for a receding source:
    # T_observed = T_proper * sqrt((1 + v/c) / (1 - v/c))
    beta = velocities / C_SIM
    observed_intervals = PROPER_HEARTBEAT_INTERVAL * np.sqrt((1 + beta) / (1 - beta))
    print("DEBUG: Relativistic calculation complete")

    # --- Plotting ---
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot the main curve
    ax.plot(velocities / C_SIM, observed_intervals, label="Observed Interval at Follower", color="red", linewidth=2)

    # Add threshold line for 'false election'
    raft_timeout_threshold = PROPER_HEARTBEAT_INTERVAL * 2.3 # From your example (115ms / 50ms)
    ax.axhline(y=raft_timeout_threshold, linestyle='--', color='gray', label=f"Example Raft Timeout ({raft_timeout_threshold:.0f}ms)")

    # Find the velocity where the timeout is crossed
    v_cross_index = np.where(observed_intervals >= raft_timeout_threshold)[0][0]
    v_cross_value = velocities[v_cross_index] / C_SIM
    ax.axvline(x=v_cross_value, linestyle=':', color='gray')
    
    # Annotations and Labels
    ax.set_title("Figure 1: Relativistic Doppler Effect on Raft Heartbeats", fontsize=16, fontweight='bold')
    ax.set_xlabel("Leader Velocity (as fraction of c)", fontsize=12)
    ax.set_ylabel("Observed Heartbeat Interval (ms)", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True)
    
    # Set plot limits to emphasize the asymptote
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, PROPER_HEARTBEAT_INTERVAL * 20) # Limit y-axis to show detail

    # Add annotation for the crossover point
    ax.annotate(f'Election triggered at v > {v_cross_value:.2f}c',
                xy=(v_cross_value, raft_timeout_threshold),
                xytext=(v_cross_value - 0.25, raft_timeout_threshold + 50),
                arrowprops=dict(facecolor='black', shrink=0.05),
                fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", fc="yellow", ec="black", lw=1, alpha=0.8))

    print("DEBUG: Plotting complete, about to save")
    # Save the figure
    plt.savefig("raft_doppler_effect.png", dpi=300)
    print("DEBUG: Savefig complete")
    print("Simulation complete. Plot saved to raft_doppler_effect.png")

if __name__ == "__main__":
    print("DEBUG: Script start")
    simulate_doppler_effect()
    print("DEBUG: Script end")
