# State-Vector Simulation Baseline: CPU SOTA

## 1. Scientific Objective
This repository establishes the State-of-the-Art (SOTA) CPU baseline for full state-vector quantum circuit simulation. This suite is inspired by the benchmarking methodology presented in the "PIMutation" paper (Lee et al., ASPDAC 2025), but is upgraded to utilize the QuEST v4 simulator for true multi-threaded SOTA performance metrics. 

This baseline serves as the control group to evaluate future hardware-accelerated approaches (such as Processing-In-Memory and Tensor Network Contractions) by explicitly demonstrating the exponential "Memory Wall" and power bottlenecks of state-vector scaling.

## 2. Methodology & The Universal Runner
We utilize the Quantum Exact Simulation Toolkit (QuEST) compiled with maximum hardware optimizations (`-O3`, `-march=native`, `-fopenmp`). 

To facilitate both strict hardware benchmarking and general stress-testing, this suite is built as a **Universal Runner**. A single executable parses command-line flags to construct, evaluate, and profile specific quantum circuits.

### Supported Algorithms:
* **The PIMutation Suite:** `BB84`, `BV`, `EDC`, `HS`, `QRNG`, `XOR`
* **Randomized Benchmarking:** `RANDOM` (Generates arbitrary circuits to test average-case limits).
* *Note: The Hidden Subgroup (HS) algorithm requires $2n$ allocated qubits for an $n$-qubit input.*

### Compilation & Manual Execution:
```bash
# Clean previous builds
make clean

# Compile the suite
make

# Run the strict Bernstein-Vazirani benchmark for 26 qubits
# (Requires sudo for RAPL energy telemetry)
sudo ./bin/quest_runner --algo BV --qubits 26

# Run an arbitrary randomized circuit with a specific gate depth
sudo ./bin/quest_runner --algo RANDOM --qubits 20 --depth 50
```

## 3. Mathematical Verification Suite
To ensure the baseline circuits are mathematically sound before profiling, the suite includes an isolated pure-C verification framework. It bypasses standard measurement and directly queries the QuEST state-vector memory to assert deterministic quantum states and physical normalization.
```bash
# Run all mathematical sanity checks
./bin/quest_runner --verify FULL

# Verify specific algorithms using a comma-separated list
./bin/quest_runner --verify BB84,BV
```

## 4. Hardware Profiling & Telemetry
To capture the true bottleneck of state-vector simulation, each benchmark strictly profiles:

- Execution Time (Compute): Measured using OpenMP wall-timers (omp_get_wtime()).

- Energy Consumption: CPU energy footprint measured via Intel RAPL (Running Average Power Limit) registers during the compute phase.

⚠️ Important Hardware Note: Because accessing RAPL registers requires hardware-level permissions, all profiling executions must be run with sudo. If run without root privileges, energy metrics will report as 0.0 Joules.

## 5. Automated Data Harvesting & Visualization
A Python automation suite is provided in src/profiling/ to sweep across qubit counts, extract metrics, filter OS-scheduling noise via median smoothing, and generate exponential scaling plots.

Requirements
```bash
pip install pandas matplotlib
```

Running the SweepsThe run_experiments.py script automatically tests the primary scaling circuits (BB84, BV, EDC, XOR, HS) from $n=10$ to $n=30$. It includes a strict memory guard that prevents the OS from crashing by skipping any allocation requiring more than 30 total qubits (~16GB RAM).
```bash
# Execute the smoothed median benchmark suite 
# (This will take time for high qubit counts)
sudo python3 ./src/profiling/run_experiments.py

# Generate high-resolution logarithmic scaling charts
python3 ./src/profiling/plot_results.py
```