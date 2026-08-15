# UPMEM Tensor-Network Quantum Simulation

This directory is the active implementation for the thesis. It is a research
system, not a production simulator. Its purpose is to execute and compare
explicit tensor-network contraction routes while keeping circuit semantics,
contraction plans, executors, measurements, and claim boundaries separate.

```text
quantum circuit
  -> tensor network
  -> contraction planner
  -> immutable TaskGraph
  -> selected route modules
  -> CPU / GPU / UPMEM executor
  -> normalized records
  -> tables and plots
```

## What Is Active

The current whole-circuit lane is **M5.5**. It lowers a circuit to a hashed
TaskGraph, runs a NumPy same-plan CPU reference or the bounded physical UPMEM
v4 engine, and writes normalized records. The physical route uses explicit
ranks, bulk request launches, and host-packed int8 or float32 numeric modes.
It keeps the plan fixed while route dimensions change.

M5.5 has passed bounded ETH development contracts for canonical, scaling, and
large-boundary profiles. It supports same-plan functionality and timing
evidence only within its recorded admission rules. It does **not** establish
general UPMEM acceleration over CPU/GPU, energy efficiency, fully active
scaling beyond the recorded active-DPU limits, graph-wide DPU residency,
PID-Comm, ATiM, or a complete multi-DIMM architecture.

Supported execution families are:

| Family | Purpose | Comparison boundary |
| --- | --- | --- |
| QuEST CPU/GPU full state | Serious full-state baseline | Same algorithm family; GPU requires a verified physical backend. |
| Quimb/cotengra CPU TN | Serious external TN baseline | Cross-implementation and generally cross-plan context. |
| Internal NumPy TaskGraph | Same-plan CPU reference | Direct reference for internal UPMEM routes. |
| UPMEM SDK simulator | Contract and boundary validation | Never hardware-performance evidence. |
| Physical UPMEM M5.5 | Bounded same-plan whole-circuit route | Requires explicit hardware admission; no fallback. |

A route is selected before preparation. Its tensor-network and planner roles
produce the immutable TaskGraph; M5 then executes that graph through one
verified CPU or UPMEM executor profile. Numeric and topology choices are
selected route dimensions. Kernel, partitioning, scheduling, and communication
are fixed profile declarations today, verified against observed native metadata
for physical rows; future engines may make them selectable. A numeric policy
must not silently change the contraction plan.

## Start Here

- [Architecture](ARCHITECTURE.md): active layers, ownership, and claim
  boundaries.
- [Pipeline contract](docs/PIPELINE_CONTRACT.md): concrete symbols, inputs,
  outputs, parameters, hashes, mutable state, and adapter limits.
- [Benchmark matrix](THESIS_BENCHMARK_MATRIX.md): thesis comparisons and
  required measurements.
- [M5.5 runbook](docs/upmem_m5_5_whole_circuit_runbook.md): current ETH
  whole-circuit procedure and physical acceptance rules.
- [Documentation index](docs/README.md): active references and historical
  compatibility material.

## Setup

From this directory, with `uv` available:

```bash
make setup
make doctor
make test
```

`make setup` creates or reuses the repository-managed parent environment
`../.venv`, installs constrained dependencies, initializes submodules, builds
the CPU QuEST runner, and runs the doctor. Do not substitute system Python for
the documented workflow.

## Primary Commands

```bash
# Local baseline matrix. Use the physical-core count printed by make doctor.
BENCH_CPU_THREADS=<physical-cores> make thesis-run

# Verify or regenerate the tracked snapshot without rerunning benchmarks.
make thesis-verify
make thesis-report

# Prepare M5.5 without allocating or launching a DPU.
make m5-circuit-plan

# Execute one CPU-only route locally; no hardware environment is required.
M5_CIRCUIT_ROUTES=opt_einsum_greedy__float32_real__numpy_cpu \
make m5-circuit-smoke

# Run the small physical M5.5 acceptance profile.
UPMEM_ALLOW_PHYSICAL_HARDWARE=1 \
UPMEM_HW_RANK_PATH=/dev/dpu_rankN \
make m5-circuit-smoke

# Run a selected M5.5 suite on explicit physical ranks.
UPMEM_ALLOW_PHYSICAL_HARDWARE=1 \
UPMEM_HW_RANK_PATH=/dev/dpu_rankN \
M5_CIRCUIT_SUITE=configs/suites/m5_circuit_canonical.yml \
make m5-circuit-study

# Report an existing M5.5 run without repeating hardware execution.
M5_CIRCUIT_RUN=runs/evidence/<study>/m5_circuit_study/<timestamp> \
make m5-circuit-report

# Create the ignored local inbox used for copied ETH archives.
make evidence-inbox
```

The Makefile retains historical qualification targets for replay and
compatibility. They are intentionally not the primary workflow.

## Results Layout

Runs are written automatically; no manual output path is required.

```text
runs/                                      # ignored generated data
  evidence/<suite>/<route>/<timestamp>/    # raw execution records
  comparisons/<report>/<timestamp>/        # tables, plots, manifest
  inbox/eth/<experiment>/                  # copied ETH archives

thesis_results/                            # tracked selected snapshots
  current/                                 # reviewed current snapshot
  releases/<name>/                         # named immutable snapshots
```

Each execution directory contains its resolved suite, normalized records,
manifests, and bounded logs. Reports are generated from records, not edited by
hand. Development runs stay in ignored `runs/`; only a reviewed snapshot is
promoted into `thesis_results/`.

## Scientific Boundaries

Only compare timings when the records show compatible circuit, tensor-network,
contraction-plan, numeric-policy, topology, timing-scope, validation, and
hardware-admission identities. QuEST and Quimb remain serious baselines but
are not automatically same-plan comparisons with the internal TaskGraph.
Simulator, modeled-planner, and unsupported rows are retained as their own
evidence categories and must not be presented as physical speedup.
