# State-Vector Simulation Baseline: CPU SOTA

## 1. Scientific Objective
This repository establishes the State-of-the-Art (SOTA) CPU baseline for full state-vector quantum circuit simulation. This suite is inspired by the benchmarking methodology presented in the "PIMutation" paper (Lee et al., ASPDAC 2025), but is upgraded to utilize the latest QuEST simulator versions for true SOTA performance metrics. 

This baseline serves as the control group to evaluate future hardware-accelerated approaches (such as Processing-In-Memory and Tensor Network Contractions).

## 2. Methodology & The Universal Runner
We utilize the Quantum Exact Simulation Toolkit (QuEST) compiled with maximum hardware optimizations (`-O3`, `-march=native`, `-fopenmp`). 

To facilitate both strict hardware benchmarking and general stress-testing, this suite is built as a **Universal Runner**. A single executable parses command-line flags to construct and evaluate specific quantum circuits on the fly.

### Supported Algorithms:
* **The PIMutation Suite:** `BB84`, `BV`, `EDC`, `HS`, `QRNG`, `XOR`
* **Randomized Benchmarking:** `RANDOM` (Generates arbitrary circuits to test average-case tensor contraction/state-vector limits).

### Compilation & Execution:

To build the Universal Runner with SOTA hardware optimizations:
```bash
# Clean previous builds
make clean

# Compile the suite
make

# Run the strict Bernstein-Vazirani benchmark for 32 qubits
./bin/quest_runner --algo BV --qubits 32

# Run the Quantum Random Number Generator for 16 qubits
./bin/quest_runner --algo QRNG --qubits 16

# Run an arbitrary randomized circuit with a specific gate depth
./bin/quest_runner --algo RANDOM --qubits 20 --depth 50
```

## 3. Profiling Metrics
To capture the true bottleneck of state-vector simulation, each benchmark strictly profiles:
* **Execution Time (Compute):** Measured using `omp_get_wtime()`.
* **Energy Consumption:** CPU energy footprint measured via RAPL (Running Average Power Limit) registers during the compute phase.
