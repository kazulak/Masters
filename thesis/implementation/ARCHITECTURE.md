# Quantum Bench Architecture

This is the active implementation directory for the Master's thesis prototype.
Old experiments and generated sudo-owned runs are under `../legacy/`.

For the full Host-CPU + UPMEM/SimplePIM thesis runtime map, route maturity
model, and near-term roadmap, see `docs/runtime_architecture_map.md`. Future
architecture, route, planner, target, or native-code changes should read that
map before proposing changes.

## Current Structure

```text
implementation/
  configs/suites/          benchmark suite YAML files
  external/QuEST/          local QuEST dependency for the CPU full-state baseline
  native/quest_cpu/        small C runner used by the QuEST provider
  native/upmem/            future SimplePIM bridge and raw UPMEM code
  scripts/                 helper commands
  src/quantum_bench/
    bench/                 one benchmark CLI and runner
    circuits/              workload/circuit construction
    core/                  typed records and JSON helpers
    environment/           environment and RAPL discovery
    formats/               shared host-side tensor format conversion utilities
    plots/                 plot generation from summary artifacts
    providers/             executable routes
    routing/               task-level route contract, analysis router, and dense preparation boundary
    targets/upmem/         UPMEM WRAM tile-plan, traffic, schedule, probe, bridge, and microbench groundwork
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
  host-side UPMEM model: dense WRAM tile plans, data format, traffic estimate,
  tiling, DPU schedule groundwork, SimplePIM availability probe metadata, and
  explicit dense bridge and dry-run SimplePIM dense GEMM microbenchmark records
formats/
  shared host-side conversion records and deterministic fixed-point utilities
routing/
  task-level route contract, route-slot decisions, CPU fallback policy, and
  one-task dense preparation for future UPMEM/SimplePIM execution
providers/exact_tn/
  benchmark routes that decide whether to use the UPMEM target layer and, later,
  call native kernels
native/upmem/
  future SimplePIM bridge code and raw C/UPMEM SDK host/DPU kernels only
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
  cases/<case_id>/target_estimates/upmem_dense_tile_plan.jsonl
  cases/<case_id>/task_route_decisions.jsonl
  cases/<case_id>/task_route_summary.json
  cases/<case_id>/route_decisions.jsonl
  raw/<case_id>.jsonl
  validation/*.json
  metrics/metrics.csv
  metrics/metrics.json
  summary.json
  summary.md
  plots/*.png
```

The standalone SimplePIM dry-run microbenchmark is not a normal benchmark
suite. It writes only:

```text
runs/<timestamp>_simplepim_microbench/
  environment.json
  simplepim_microbench.json
```

`ready` in that artifact means ready for a future bridge attempt, not validated
SimplePIM execution.

The developer-only one-task dense bridge harness is also outside normal suite
execution:

```bash
PYTHONPATH=src ../.venv/bin/python -m quantum_bench.bench dense-task-bridge --case bell_2q --task-index 0 --backend mock_numpy_dequantized
```

For an explicitly selected later task, it can materialize inputs by replaying
earlier CPU contractions:

```bash
PYTHONPATH=src ../.venv/bin/python -m quantum_bench.bench dense-task-bridge --case bell_2q --task-index 1 --materialization cpu-replay --backend mock_numpy_dequantized
```

It writes:

```text
runs/<timestamp>_dense_task_bridge/
  environment.json
  dense_task_bridge_summary.json
  bridge/input_manifest.json
  bridge/operands/*.npy
  bridge/references/*.npy
  bridge/output_manifest.json
  bridge/outputs/*.npy       only for the mock backend
```

This command is a file-boundary and preparation check for one real
`ContractionTask`. It is mock-by-default, does not select `dense_gemm` in normal
routing, and keeps `external_command_executed=false` and
`execution_implemented=false`. CPU replay is developer-only input
materialization for the harness; it is not the normal CPU execution path and not
full routed TaskGraph execution.

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

1. Connect `formats/` fixed-point records to future route preparation artifacts
   once task routes begin preparing tensors.
2. Extend the current `targets/upmem/` tile-plan model from deterministic host
   records to executable tiling choices.
3. Connect `routing/` from analysis-only route decisions to a preparation and
   execution-aware task router.
4. Port the useful dense UPMEM kernel only after the task schedule and WRAM model
   exist.
5. Add sparse and heuristic/permutation providers after the dense path is
   structurally stable.

## Design Rules

- Keep one benchmark runner.
- Keep providers behind the provider interface.
- Do not add legacy route names during this design stage.
- Do not add separate plotting or benchmark scripts for individual baselines.
- Optional hardware or external tools must skip with explicit reasons.
- Do not claim UPMEM speedup until transfer, conversion, validation, and
  aggregation costs are visible in the artifacts.
