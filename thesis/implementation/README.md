# UPMEM Tensor-Network Quantum Simulation

This directory is the active implementation for the thesis. It is a research
system, not a production simulator. Its purpose is to execute and compare
explicit tensor-network contraction routes while keeping circuit semantics,
contraction plans, executors, measurements, and claim boundaries separate.

```text
CircuitSpec
  -> build_tensor_network_data
  -> TensorNetworkSpec + TensorInputs
  -> plan_contractions / PlannerResult
  -> build_contraction_dag / ContractionDAG
  -> compile_execution / ExecutionPlan
  -> execute / ExecutionResult
  -> normalized records
  -> tables and plots
```

## What Is Active

The current whole-circuit lane is **M5.5**. It lowers a circuit to a hashed
`ContractionDAG`, runs a NumPy same-plan CPU reference or the bounded physical
UPMEM v4 engine, and writes normalized records. The physical route uses
explicit ranks, bulk request launches, and host-packed int8 or float32 numeric
modes. It keeps the contraction graph fixed while execution-plan dimensions
change.

M5.5 is the active implementation lane. It has code and documented ETH
development observations for canonical, scaling, and large-boundary profiles,
but those ignored runs are not accepted or verified evidence. The only
tracked verified physical capsule is M4.5, as recorded in
[docs/MILESTONES.md](docs/MILESTONES.md). M5.5 does **not** establish general
UPMEM acceleration over CPU/GPU, energy efficiency, fully active scaling,
graph-wide DPU residency, PID-Comm, ATiM, or a complete multi-DIMM
architecture.

Supported execution families are:

| Family | Purpose | Comparison boundary |
| --- | --- | --- |
| QuEST CPU/GPU full state | Serious full-state baseline | Same algorithm family; GPU requires a verified physical backend. |
| Quimb/cotengra CPU TN | Serious external TN baseline | Cross-implementation and generally cross-plan context. |
| Functional CPU TN | Same-DAG CPU reference | Direct reference for internal UPMEM routes. |
| UPMEM SDK simulator | Contract and boundary validation | Never hardware-performance evidence. |
| Physical UPMEM M5.5 | Active bounded same-plan whole-circuit implementation lane | Development observations only; no tracked evidence claim and no fallback. |

A route is selected before preparation. The circuit and planner produce the
immutable `ContractionDAG`; M5 then compiles and executes that graph through
one verified CPU or UPMEM executor profile. Numeric and topology choices are
execution-plan dimensions. Kernel, partitioning, scheduling, and communication
are fixed profile declarations today, verified against observed native metadata
for physical rows; future engines may make them selectable. A numeric policy
must not silently change the contraction DAG.

For M5 v4, the native host emits backend, profile, ABI, session, dispatch,
kernel, and execution-class identity in both `READY` and `RESPONSE`. The Python
adapter admits a physical row only when those observations agree across ranks
and `native_identity_verified=true`. `graph_intermediate_placement=host_managed`
is separately recorded with origin `m5_host_coordinator_v1` because it is a
host-runtime fact, not a native-kernel identity. M5 `exact_once` remains a
legacy compatibility field for completed host DAG nodes; it is not a
native-kernel exactly-once claim.

The old `core.records.TaskGraph` remains only as a compatibility materialization
for legacy identity and evidence records. It is not the active execution
contract. QuEST full-state CPU/GPU execution remains a separate baseline family;
CPU TN and UPMEM TN routes share the tensor-network, planner, DAG, compilation,
validation, and evidence boundaries.

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
- [Milestone ledger](docs/MILESTONES.md): sole authority for implementation
  and evidence status.

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

# Run the small physical M5.5 development profile.
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
compatibility. They are development and historical workflows, not evidence
status declarations.

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
are not automatically same-DAG comparisons with the functional internal TN
routes.
Simulator, modeled-planner, and unsupported rows are retained as their own
evidence categories and must not be presented as physical speedup.
