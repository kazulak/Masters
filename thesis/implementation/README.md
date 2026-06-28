# Quantum Bench Implementation

Active implementation for the Master's thesis runtime described in
`../CODEX_IMPLEMENTATION_DIRECTION.md` and
`../CODEX_UPMEM_ARCHITECTURE_DIRECTION.md`.

The runtime is route-aware: it builds exact tensor networks from quantum
circuits, plans contractions, asks each provider route whether it can execute,
records skip reasons, validates numerical output, and writes benchmark artifacts
under timestamped `runs/` directories.

The detailed Host-CPU + UPMEM/SimplePIM architecture scaffold is documented in
`docs/runtime_architecture_map.md`. Read it before planning route, target,
planner, or native-code changes.

## Quick Start

```bash
cd thesis/implementation
PYTHONPATH=src ../.venv/bin/python -m pytest -q
PYTHONPATH=src ../.venv/bin/python -m quantum_bench.bench run --suite configs/suites/smoke.yml
PYTHONPATH=src ../.venv/bin/python -m quantum_bench.bench upmem-env-check
PYTHONPATH=src ../.venv/bin/python -m quantum_bench.bench upmem-env-check --run-sample --target simulator
PYTHONPATH=src ../.venv/bin/python -m quantum_bench.bench simplepim-microbench --dry-run --m 8 --k 8 --n 8
PYTHONPATH=src ../.venv/bin/python -m quantum_bench.bench dense-task-bridge --case bell_2q --task-index 0 --backend mock_numpy_dequantized
PYTHONPATH=src ../.venv/bin/python -m quantum_bench.bench dense-task-bridge --case bell_2q --task-index 1 --materialization cpu-replay --backend mock_numpy_dequantized
SIMPLEPIM_STUB_BIN=native/upmem/simplepim/simplepim_dense_stub.py PYTHONPATH=src ../.venv/bin/python -m quantum_bench.bench dense-task-bridge --case bell_2q --task-index 0 --backend simplepim_external_stub --execute-external
PYTHONPATH=src ../.venv/bin/python -m quantum_bench.bench dense-task-bridge --case bell_2q --task-index 0 --backend upmem_sdk_simulator_dense --execute-external
PYTHONPATH=src ../.venv/bin/python -m quantum_bench.bench pim-bridge-eval --case bell_2q --n-qubits 2 --dry-run
PYTHONPATH=src ../.venv/bin/python -m quantum_bench.bench pim-bridge-eval --suite configs/suites/pim_bridge_eval_quick.yml --dry-run
PYTHONPATH=src ../.venv/bin/python -m quantum_bench.bench pim-bridge-eval --suite configs/suites/pim_bridge_eval_quick.yml --backend upmem_sdk_simulator_dense --execute-external --max-executed-tasks-per-case 2
PYTHONPATH=src ../.venv/bin/python -m quantum_bench.bench pim-frontier-analysis --suite configs/suites/pim_bridge_eval_quick.yml
PYTHONPATH=src ../.venv/bin/python -m quantum_bench.bench pim-frontier-analysis --suite configs/suites/pim_frontier_pressure_quick.yml
PYTHONPATH=src ../.venv/bin/python -m quantum_bench.bench benchmark-matrix-report --matrix configs/benchmark_matrix.yml
PYTHONPATH=src ../.venv/bin/python -m quantum_bench.bench dense-route-coverage --case bell_2q
PYTHONPATH=src ../.venv/bin/python -m quantum_bench.bench dense-route-coverage --suite configs/suites/planner_compare.yml
PYTHONPATH=src ../.venv/bin/python -m quantum_bench.bench shadow-routed-runtime --case bell_2q
PYTHONPATH=src ../.venv/bin/python -m quantum_bench.bench shadow-routed-runtime --case bell_2q --shadow-route-policy dense-if-estimate-supported
PYTHONPATH=src ../.venv/bin/python -m quantum_bench.bench shadow-routed-runtime --case ghz_chain --n-qubits 3 --shadow-route-policy dense-if-no-tiling
PYTHONPATH=src ../.venv/bin/python -m quantum_bench.bench plot runs/latest
```

For RAPL energy measurement, run the suite with the helper so sudo still uses
the thesis virtual environment:

```bash
scripts/run_energy_suite.sh configs/suites/local_energy.yml
```

## Active Boundaries

- `src/quantum_bench/` contains the Python implementation.
- `configs/suites/` contains the reproducible benchmark suite definitions.
- `native/quest_cpu/` contains only the C QuEST runner used by the
  `quest_cpu_full_state_benchmark` provider.
- `external/QuEST/` contains the local QuEST dependency used by that runner.
- `src/quantum_bench/formats/` contains shared host-side data-format conversion
  records and deterministic fixed-point utilities.
- `src/quantum_bench/targets/upmem/` contains host-side UPMEM WRAM, traffic,
  tile-plan, schedule, SimplePIM probe, dense bridge manifests, and explicit
  SimplePIM dry-run microbenchmark and environment bring-up groundwork shared
  by future UPMEM providers.
- `src/quantum_bench/routing/` contains the task-level route contract,
  analysis-only dynamic router skeleton, shadow route policy records, and
  one-task dense preparation boundary for future UPMEM/SimplePIM execution.
- `src/quantum_bench/tn/materialize.py` contains developer-facing TaskGraph
  CPU replay for materializing one selected task's inputs. It is not the normal
  CPU execution path.
- `src/quantum_bench/bench/dense_task_bridge.py` is a developer-only one-task
  harness that connects a real `ContractionTask` to dense preparation and the
  mock bridge. It can opt into CPU replay for explicit task selection, but it
  does not change normal benchmark routing.
- `src/quantum_bench/bench/dense_route_coverage.py` is a developer-only
  readiness analyzer over every task in selected circuits. It reports dense
  route materialization, preparation, tile-plan, bridge-manifest, and optional
  stub-contract coverage; it is not routed execution.
- `src/quantum_bench/bench/shadow_routed_runtime.py` is a developer-only full
  TaskGraph harness. It executes every task with CPU fallback as the
  authoritative numeric route while recording dense route preparation and
  optional capped bridge/stub evidence as shadow metadata. Its shadow route
  policy fields are what-if decisions only; they never replace CPU fallback.
- `src/quantum_bench/bench/pim_bridge_eval.py` is a developer-only thesis
  evaluation harness. It runs growing workloads through TaskGraph,
  materialization, dense preparation, dense bridge manifests, and capped
  `upmem_sdk_simulator_dense` task execution where eligible. It evaluates
  per-task backend evidence only and ignores normal suite `routes:`.
- `src/quantum_bench/bench/pim_frontier_analysis.py` is a developer-only
  memory-level and parallelism-frontier analyzer. It classifies TaskGraph GEMM
  contractions as modeled L1/L2/L3/L4 UPMEM memory cases and estimates
  inter-task, intra-task, and hybrid DPU parallelism. It does not execute UPMEM
  kernels or normal provider routes.
- `src/quantum_bench/targets/upmem/synthetic_pressure.py` contains
  analysis-only synthetic pressure TaskGraphs used to expose L2/L3/L4 memory
  boundaries without allocating tensors or pretending they are real quantum
  circuits. Normal circuit loading and normal suite execution reject these
  workloads explicitly.
- `src/quantum_bench/bench/benchmark_matrix_report.py` produces the thesis
  benchmark matrix and PIM pressure report. It keeps UPMEM as one top-level
  runtime category, `upmem_tn_runtime`, while reporting L1/L2/L3/L4 only as
  internal UPMEM execution classes.
- `native/upmem/simplepim/` is reserved for future SimplePIM bridge code and
  currently contains the non-executing external contract stub plus the first
  UPMEM SDK simulator dense bridge runner. The runner is in the
  SimplePIM/UPMEM bridge lane but does not use SimplePIM APIs.
- `native/upmem/raw_dense/` is reserved for future raw UPMEM SDK dense kernels.
- `../legacy/` contains old prototypes and generated sudo-owned run folders kept
  out of the active implementation.

## UPMEM/SimplePIM Environment Check

`upmem-env-check` verifies local UPMEM/SimplePIM bring-up before real dense
backend work starts. It records UPMEM SDK tools, SimplePIM source discovery,
safe bounded command probes, and optional simulator sample build/run results
under `runs/<timestamp>_upmem_env_check/`.

SimplePIM discovery checks `--simplepim-home`, then `SIMPLEPIM_HOME`, then the
repo fallback `../legacy/extern/SimplePIM` from this directory. Without
`--run-sample`, simulator and hardware execution are recorded as not verified.
Sample build or simulator-run failure is written as JSON status `failed`, but
the CLI still exits normally after writing the artifact so the failure is
auditable.

## UPMEM SDK Simulator Dense Backend

`upmem_sdk_simulator_dense` is the first real PIM-backed dense bridge backend.
It is explicit and simulator-only. The backend ID is a developer bridge backend,
not a normal benchmark route; the final UPMEM route category remains the unified
`upmem_tn_runtime`.

```bash
PYTHONPATH=src ../.venv/bin/python -m quantum_bench.bench dense-task-bridge --case bell_2q --task-index 0 --backend upmem_sdk_simulator_dense --execute-external
PYTHONPATH=src ../.venv/bin/python -m quantum_bench.bench dense-task-bridge --case synthetic_l2_square --task-index 0 --backend upmem_sdk_simulator_dense --execute-external
```

The backend consumes the existing dense bridge manifest, runs a minimal UPMEM
SDK DPU GEMM program through the simulator, writes
`outputs/upmem_sdk_simulator_output.npy`, and validates against
`references/expected_dequantized_output.npy`. It currently supports:

- `L1_WRAM`: task-level simulator dense bridge subset with padded direct GEMM;
- `L2_SINGLE_DPU_MRAM`: task-level simulator real-valued dense bridge subset
  with one-DPU MRAM operands and WRAM-resident output tiles.

It does not run hardware, does not implement L3 multi-DPU execution, and does
not make dense output authoritative in the shadow runtime. Any timings it
records are bring-up timings, not performance evidence.

## PIM Bridge Evaluation

`pim-bridge-eval` evaluates task-level readiness and optional simulator
execution over larger workload suites:

```bash
PYTHONPATH=src ../.venv/bin/python -m quantum_bench.bench pim-bridge-eval --suite configs/suites/pim_bridge_eval_quick.yml --dry-run
PYTHONPATH=src ../.venv/bin/python -m quantum_bench.bench pim-bridge-eval --suite configs/suites/pim_bridge_eval_quick.yml --backend upmem_sdk_simulator_dense --execute-external --max-executed-tasks-per-case 2
PYTHONPATH=src ../.venv/bin/python -m quantum_bench.bench pim-bridge-eval --suite configs/suites/pim_bridge_eval_quick.yml --backend upmem_sdk_simulator_dense --execute-external --max-executed-tasks-per-case 2 --debug-failures --compare-mock-on-failure
PYTHONPATH=src ../.venv/bin/python -m quantum_bench.bench pim-bridge-eval --suite configs/suites/pim_l2_tiled_quick.yml --dry-run
```

Run external simulator execution on the quick suite first. The extended
`configs/suites/pim_bridge_eval.yml` suite should be run as dry-run before
attempting simulator execution. `configs/suites/pim_l2_tiled_quick.yml` is a
developer-only synthetic pressure suite for L2 bring-up; normal benchmark runs
must not execute it as a real quantum circuit suite. The command writes JSON,
CSV, Markdown, and
optional matplotlib plot artifacts under `runs/`. It reports support, blockers,
validation metrics, and bring-up timings per task; it does not make the dense
backend authoritative for the full circuit.

For simulator correctness triage, add `--debug-failures`. Failed attempts write
`validation_diagnostics.json` beside the dense bridge artifacts, comparing the
simulator output, direct Python int8/int32 reconstruction, and optionally the
mock bridge output when `--compare-mock-on-failure` is also set. These
diagnostics are for backend hardening; failed simulator timings must not be
presented as performance evidence.

## PIM Frontier Analysis

`pim-frontier-analysis` models where each TaskGraph contraction sits in the
UPMEM memory hierarchy and whether useful parallelism is exposed by the current
contraction path:

```bash
PYTHONPATH=src ../.venv/bin/python -m quantum_bench.bench pim-frontier-analysis --case bell_2q --n-qubits 2
PYTHONPATH=src ../.venv/bin/python -m quantum_bench.bench pim-frontier-analysis --suite configs/suites/pim_bridge_eval_quick.yml
```

The command writes JSON, CSV, Markdown, and optional matplotlib plots under
`runs/`. It reports counts by memory level and by dominant parallelism source.
If the current pairwise contraction path has frontier width 1, that is recorded
as `task_graph_serialized_by_planner`; it is evidence for future path-frontier
optimization, not a command failure. These artifacts are modeled analysis, not
measured hardware speedup.

Pressure suites add explicitly labeled model-only cases for L2/L3/L4 boundaries:

```bash
PYTHONPATH=src ../.venv/bin/python -m quantum_bench.bench pim-frontier-analysis --suite configs/suites/pim_frontier_pressure_quick.yml
PYTHONPATH=src ../.venv/bin/python -m quantum_bench.bench pim-frontier-analysis --suite configs/suites/pim_frontier_pressure.yml
```

Synthetic pressure workloads must declare `workload_type: synthetic_pressure`,
`execution_scope: model_only`, and `not_real_quantum_circuit: true`. They are
accepted only by analysis commands. If they reach normal circuit loading or
normal benchmark execution, the run fails with an explicit analysis-only error.

## Benchmark Matrix And Pressure Report

`benchmark-matrix-report` prepares the final thesis comparison scaffold without
executing planned or unavailable baselines:

```bash
PYTHONPATH=src ../.venv/bin/python -m quantum_bench.bench benchmark-matrix-report --matrix configs/benchmark_matrix.yml
```

The report writes `benchmark_matrix.json`, `benchmark_matrix.csv`,
`pim_pressure_tasks.csv`, `pim_pressure_cases.csv`,
`pim_pressure_resource_models.csv`, Markdown, and optional matplotlib plots. It
distinguishes route categories from implementation route IDs. For example,
`cpu_tn_exact` is the benchmark category while `cpu_tn_einsum_exact` is the
current implemented route ID.

The UPMEM row is intentionally one top-level runtime: `upmem_tn_runtime`.
`L1_WRAM`, `L2_SINGLE_DPU_MRAM`, `L3_MULTI_DPU`, and `L4_OUT_OF_SCOPE` are
internal UPMEM execution classes under that runtime, not separate final
competitors. Current UPMEM evidence is scoped as task-level simulator dense
bridge evidence for an L1 subset plus L2/L3/L4 model-only pressure evidence.
The report must not be read as a full-circuit UPMEM speedup comparison.
