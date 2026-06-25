# Quantum Bench Architecture

This is the active implementation directory for the Master's thesis prototype.
Old experiments and generated sudo-owned runs are under `../legacy/`.

## Current Structure

```text
implementation/
  configs/suites/          benchmark suite YAML files
  external/QuEST/          local QuEST dependency for the CPU full-state baseline
  native/quest_cpu/        small C runner used by the QuEST provider
  native/upmem/            future native UPMEM code
  scripts/                 helper commands
  src/quantum_bench/
    bench/                 one benchmark CLI and runner
    circuits/              workload/circuit construction
    core/                  typed records and JSON helpers
    environment/           environment and RAPL discovery
    plots/                 plot generation from summary artifacts
    providers/             executable routes
    targets/upmem/         UPMEM WRAM, traffic, tiling, and schedule groundwork
    tn/                    exact tensor-network construction and planning
    validation/            numerical validation metrics
  tests/                   Python runtime tests
```

There is one benchmark pipeline:

```bash
PYTHONPATH=src ../.venv/bin/python -m quantum_bench.bench run --suite configs/suites/smoke.yml
```

All suite files use schema v2 with separate `workloads:` and `routes:` sections.
Route IDs are canonical only; old route names are not accepted.

## Current Routes

- `cpu_tn_einsum_exact`
  - exact tensor-network reference route
  - CPU, in-process Python, NumPy einsum
  - returns a final tensor and is validated
- `quest_cpu_full_state_benchmark`
  - CPU full-state-vector benchmark baseline
  - external C/QuEST process
  - metrics-only, benchmark-only validation mode
- `upmem_dense_int8_placeholder`
  - UPMEM dense candidate placeholder
  - consumes `targets/upmem/` to estimate WRAM fit and transfer volume
  - probes availability and records skip reasons
  - no native execution yet

## UPMEM Boundary

The future UPMEM implementation has three separate layers:

```text
tn/
  shared tensor network and contraction task graph
targets/upmem/
  host-side UPMEM model: WRAM fit, data format, traffic estimate, tiling and
  DPU schedule groundwork
providers/exact_tn/
  benchmark routes that decide whether to use the UPMEM target layer and, later,
  call native kernels
native/upmem/
  C/UPMEM SDK host and DPU kernels only
```

This keeps tensor-network creation and path finding shared across CPU TN, future
GPU TN, and UPMEM TN. UPMEM-specific constraints do not belong in `tn/`; they
belong in `targets/upmem/` and are consumed by UPMEM providers.

## Current Benchmark Artifacts

Each run writes a timestamped directory under `runs/`:

```text
runs/<timestamp>_<suite_id>/
  config/resolved_suite.yml
  environment.json
  cases/<case_id>/circuit.json
  cases/<case_id>/task_graph.json
  cases/<case_id>/route_decisions.jsonl
  raw/<case_id>.jsonl
  validation/*.json
  metrics/metrics.csv
  metrics/metrics.json
  summary.json
  summary.md
  plots/*.png
```

Plots follow one rule: bars compare different cases/routes, and lines are used
only for scaling within a single circuit family across qubit counts.

## Target Pipeline

The thesis implementation should stay simple and explicit:

```text
Quantum circuit
  -> tensor network
  -> UPMEM-aware contraction path search
       cost = FLOPs
            + peak intermediate size
            + WRAM feasibility
            + estimated host-DPU traffic
            + available task/tile parallelism
            + DPU load balance
            + synchronization/aggregation cost
  -> executable contraction schedule
       tasks/tiles mapped to DPUs
  -> UPMEM execution
  -> CPU/QuEST validation and benchmark comparison
```

The host CPU owns global planning, path search, tiling, route decisions,
validation, and aggregation. UPMEM providers should execute only bounded local
tile tasks.

## Next Implementation Priorities

1. Make `cpu_tn_einsum_exact` execute the planned `ContractionTask` sequence
   instead of one whole-network `np.einsum`.
2. Add WRAM/tile estimates to contraction tasks:
   tile shape, tile bytes, host-DPU bytes, MRAM-WRAM bytes, and fit/reject reason.
3. Extend the current `targets/upmem/` schedule estimate from untiled dense
   tasks to real tiling choices.
4. Extend route decisions from case-level to task-level so the runner records
   selected/rejected providers per contraction task.
5. Port the useful dense UPMEM kernel only after the task schedule and WRAM model
   exist.
6. Add sparse and heuristic/permutation providers after the dense path is
   structurally stable.

## Design Rules

- Keep one benchmark runner.
- Keep providers behind the provider interface.
- Do not add legacy route names during this design stage.
- Do not add separate plotting or benchmark scripts for individual baselines.
- Optional hardware or external tools must skip with explicit reasons.
- Do not claim UPMEM speedup until transfer, conversion, validation, and
  aggregation costs are visible in the artifacts.
