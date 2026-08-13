# Scientific Benchmark & Testing Plan: Milestone M7
## PIM-Aware Path Cost Model & Contraction Path Optimizer Engine

### 1. Benchmark Suite Dataset
We will expand beyond trivial single-qubit tests to evaluate on rigorous quantum circuit topologies. The dataset encompasses varied entanglement structures, depths, and computational complexities:

*   **GHZ State Preparation:** 4, 8, 12, 16 qubits.
*   **Quantum Fourier Transform (QFT):** 4, 8, 12 qubits.
*   **Random Quantum Circuits (RQC) / Sycamore 2D Grid:** 4, 8, 12, 16 qubits.
*   **VQE / QAOA Ansatz Circuits:** 4, 8, 12 qubits, evaluated at variable depths ($D = 2, 4, 8, 16$).

### 2. Comprehensive Empirical Metric Spectrum
The benchmarking framework will collect and analyze the following metrics across all tests:

*   **Contraction Tree Metrics:**
    *   Path Depth
    *   Tree Width
    *   Maximum Intermediate Tensor Rank (Open Index Count)
*   **Compute Metrics:**
    *   Exact Complex GEMM FLOPs (Calculated as $8 \times B \times M \times N \times K$)
*   **Memory Movement Metrics:**
    *   Host-to-Device (H2D) Payload Bytes
    *   Device-to-Host (D2H) Payload Bytes
    *   MRAM DMA Window Volume
    *   Peak Intermediate Memory Footprint (Bytes)
*   **Hardware Pressure Metrics:**
    *   Maximum WRAM Pressure Ratio

### 3. Multi-Planner Baseline Comparison
To rigorously validate the M7 PIM-Aware Greedy planner, we will execute a comparative analysis against established tensor network contraction optimizers across all circuits:

*   `opt_einsum` Greedy
*   `opt_einsum` DP (Dynamic Programming) / Optimal
*   `cotengra` Greedy
*   `cotengra` Max-Repeats
*   **M7 PIM-Aware Greedy** (Our proposed model)

### 4. Scientific Artifacts & Statistical Exports
The benchmark pipeline will produce standardized data exports and publication-grade visualizations for empirical analysis.

**Data Exports:**
*   Raw data matrix exported to `m7_benchmark_results_v1.json`
*   Tabular format exported to `m7_benchmark_results_v1.csv`

**High-DPI Publication Plots:**
1.  `m7_pareto_flop_vs_transfer.png`: Pareto frontier visualizing FLOPs vs. Transfer Bytes across varying circuit sizes.
2.  `m7_planner_benchmark_matrix.png`: Multi-metric bar comparison illustrating the performance of M7 against `opt_einsum` and `cotengra` baselines.
3.  `m7_wram_pressure_heatmap.png`: 2D parameter grid heatmap ($w\_wram \times w\_h2d$) demonstrating memory footprint control.
4.  `m7_topology_scaling.png`: Scaling behavior analysis (qubit count $N=4..16$) vs. contraction path cost.
