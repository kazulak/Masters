# UPMEM Tensor-Network Quantum Simulation

This is the active implementation for the thesis. It tests whether hierarchical
parallel execution on UPMEM can improve end-to-end tensor-network (TN) quantum
circuit simulation while meeting an explicit accuracy requirement.

The software is a research prototype, not a general quantum simulator.

## Pipeline

```text
quantum circuit + simulation query
  -> tensor network + immutable input arrays
  -> contraction path and optional slicing
  -> target-neutral ContractionDAG
  -> CPU execution or UPMEM mapping
  -> validated quantum result
  -> one evidence row per repetition
  -> tables and plots
```

`ContractionDAG` states which mathematical contractions and reductions must run.
It is not GEMM lowering and contains no DPU IDs, kernels, tiles, scales, or
machine paths. UPMEM mapping adds those physical decisions without changing the
DAG identity.

## Research Scope

The primary contribution under development is a Host-UPMEM execution model with
parallel work at two levels:

- tasklets divide local work inside a DPU;
- DPUs or DPU groups execute independent output tiles or slices.

Supporting mechanisms are explicit but subordinate to that goal:

- contraction path and slicing selection;
- host-side numerical encoding and final decoding;
- WRAM-aware tiling and layout;
- DPU/rank placement and intermediate residency;
- transfer scheduling and host reduction.

The final study compares:

- CPU and GPU full-state QuEST baselines;
- Quimb/cotengra CPU TN baselines;
- NumPy same-DAG TN replay;
- UPMEM simulator correctness runs;
- repeated physical UPMEM runs.

These comparison families are not automatically equivalent. Evidence records
must state whether the circuit, DAG, numeric policy, timing scope, executable,
and hardware are compatible.

## Current Reset Branch

`refactor/thesis-runtime-simplification` is replacing historical milestone
pipelines with one active path:

```text
ContractionDAG -> UpmemPlan -> UPMEM runtime
```

The pre-reset implementation and evidence remain available at the tag
`pre-thesis-runtime-simplification`. The temporary
[migration ledger](MIGRATION_LEDGER.md) records remaining adapters and concrete
complexity reductions.

No accepted result is rewritten by this reset. Physical behavior must be rerun
after the active native contract changes.

## Setup

Use the repository-managed environment; do not install project packages into
the system Python:

```bash
make setup
make doctor
make test
```

During the reset, existing benchmark commands remain available until the stable
CLI replaces them. Use `make help` for the current public workflow.

Physical execution always requires explicit opt-in and rank selection:

```bash
UPMEM_ALLOW_PHYSICAL_HARDWARE=1 \
UPMEM_HW_RANK_PATH=/dev/dpu_rankN \
make m5-circuit-smoke
```

There is no automatic CPU or simulator fallback from a physical route.

## Results

Generated runs are ignored by Git:

```text
runs/evidence/       raw execution rows and manifests
runs/comparisons/    derived tables and plots
runs/inbox/eth/      copied ETH run archives
```

Selected reviewed snapshots live in `thesis_results/`. Reports consume
normalized evidence; they do not execute benchmarks or edit results manually.

## Claim Boundary

The current repository supports reproducible circuit-to-TN lowering, explicit
contraction DAGs, same-DAG CPU execution, and bounded physical UPMEM execution
with strict provenance checks. It does not yet establish general UPMEM speedup,
energy efficiency, full multi-DIMM scaling, general graph-wide residency,
PID-Comm or ATiM production use, or a hardware-calibrated planner.

Simulator timings are never physical performance evidence. Unsupported and
failed rows remain in the dataset.

## Documents

- [Architecture](ARCHITECTURE.md)
- [Benchmark methodology](THESIS_BENCHMARK_MATRIX.md)
- [Current result audit](THESIS_RESULTS_AUDIT.md)
- [Simplification audit and migration guide](ARCHITECTURE_SIMPLIFICATION_AUDIT_AND_AGENT_GUIDE.md)
