# UPMEM Tensor-Network Quantum Simulation

This repository is the thesis implementation for benchmarking tensor-network
(TN) quantum-circuit simulation across direct CPU/TN baselines and UPMEM.
It is a research prototype, not a general-purpose quantum simulator.

## Active Flow

```text
SimulationJob
  -> target-neutral TensorNetwork
  -> planner path
  -> ContractionDAG
  -> direct NumPy / Quimb / cotengra / QuEST execution
     or UpmemPlan -> ABI-v4 UPMEM runtime
  -> canonical evidence
  -> report
```

`TensorNetwork` is semantic and non-executable: it describes tensors,
connectivity, inputs, and the requested output. `ContractionDAG` is the sole
logical execution IR: it records the selected contraction order, explicit
dependencies, local slicing branches, and reductions. It contains no DPU
placement, tiles, kernels, scales, binary paths, or machine settings.

`UpmemPlan` is a target-specific mapping of a DAG. It selects the numeric
policy, output/K tiles, topology, stages, and kernel policy without changing
the logical plan. ABI-v4 executes real-valued tiles; complex contractions use
four real products under the selected split-complex policy.

## Commands

All commands run from `thesis/implementation`.

```bash
# Create or update ../.venv, build the CPU QuEST runner, then inspect tools.
make setup
make doctor
make test

# Default software benchmark configuration.
make plan CONFIG=configs/tn_benchmark_reset.yml OUTPUT=runs/reset-plan
make run CONFIG=configs/tn_benchmark_reset.yml OUTPUT=runs/reset-run
make verify INPUT=runs/reset-run
make report INPUT=runs/reset-run REPORT_OUTPUT=runs/reset-report

# Build the active UPMEM ABI-v4 host and DPU binaries.
make build-upmem-runtime UPMEM_TASKLETS=1

# Physical execution is opt-in and uses the dedicated physical configuration.
UPMEM_ALLOW_PHYSICAL_HARDWARE=1 \
  make qualify PHYSICAL_CONFIG=configs/tn_benchmark_physical.yml \
  OUTPUT=runs/physical-run

# Standalone future PID-Comm compatibility harness.
make pidcomm-check
```

`configs/tn_benchmark_reset.yml` is the default CPU/TN software smoke suite.
`configs/tn_benchmark_physical.yml` is the one-DPU physical qualification
suite. Before physical execution, set its `rank_paths` and binary paths for
the target machine and build the matching tasklet-count binaries.

`plan` writes a deterministic experiment plan without execution. `run` writes
canonical evidence. `verify` checks evidence identities and integrity.
`report` only reads existing evidence and produces tables and plots.

## Evidence and Comparison Rules

Each attempted warmup or measurement creates a canonical sample record.
Manifests record problem, logical-plan, physical-plan, executable,
environment, experiment, session, and sample identities. Failed and
unsupported attempts remain visible.

Timing scopes are explicit. `steady_execution_v1` excludes planning and
session lifecycle; `simulation_end_to_end_v1` includes route-specific
preparation through decoded output. Reference calculation, validation,
hashing, and report writing are outside both scopes.

Only evidence with compatible problem, logical-plan relationship, timing
scope, validation, and physical provenance can support a performance claim.
SDK-simulator runs are correctness evidence only.

## Current Capability Boundary

The reset architecture is software- and SDK-simulator-validated. The reset
physical UPMEM route has **not** yet been qualified on ETH hardware. The
repository therefore makes no reset-route claim of speedup, energy efficiency,
multi-rank scaling, broad graph residency, or hardware-calibrated planning.

The active native runtime uses pinned SimplePIM management types and its
initialization kernel around raw-SDK allocation and dispatch. It is not yet a
qualified high-level SimplePIM scheduler or compute route. The PID-Comm
qualification harness is retained, but PID-Comm is not an active communication
provider. ATiM is not integrated.

## Repository Layout

```text
src/quantum_bench/
  model.py, circuits.py, lowering.py, planning.py, numerics.py, results.py
  cpu.py, baselines.py, experiment.py, evidence.py, report.py, cli.py
  upmem/plan.py, tiling.py, protocol.py, native_session.py, runtime.py

native/upmem/runtime/       active ABI-v4 host/DPU build
native/upmem/pidcomm_qualification/  standalone compatibility harness
configs/tn_benchmark_reset.yml       software benchmark suite
configs/tn_benchmark_physical.yml    physical qualification suite
```

Generated evidence is written below `runs/` and is ignored by Git. Reviewed
historical snapshots remain in `thesis_results/`; reports never edit them.

See [ARCHITECTURE.md](ARCHITECTURE.md) for ownership boundaries and
[STATUS.md](STATUS.md) for implemented versus pending capabilities.
