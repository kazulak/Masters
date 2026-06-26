# Runtime Architecture Map

This document is the implementation scaffold for the Master's thesis runtime.
It maps the intended Host-CPU + UPMEM/SimplePIM architecture to the current
`thesis/implementation` codebase and records where future modules belong.

The thesis contribution is a modular Host-CPU + UPMEM/SimplePIM-oriented runtime
architecture for exact tensor-network quantum-circuit simulation. The host keeps
global orchestration, planning, routing, validation, and reporting. Bounded local
execution is delegated to route providers.

Path optimization is important, but it is only one host-side module. The runtime
must not drift into a path-planner-only prototype or a dense-only throwaway
vertical slice.

## End-To-End Architecture

```text
Quantum circuit input
  -> Host CPU orchestrator
       circuit analyzer
       path optimizer
       WRAM slicer
       data format conversion
       dynamic heuristic router
  -> UPMEM/SimplePIM execution layer
       dense GEMM route
       sparse route
       heuristic/bypass route
       optional TransPimLib/math-support route
  -> host aggregation
  -> amplitude/validation output
  -> benchmark artifacts
```

The host is the control plane. It owns the circuit, tensor network, contraction
plan, route eligibility decisions, fallback policy, validation reference, and
benchmark artifacts. UPMEM/SimplePIM code is the data plane and should execute
only bounded local tasks that fit an explicit route contract.

## Current Code Mapping

| Architecture block | Current location | Current status | Intended direction |
|---|---|---|---|
| Benchmark spine | `src/quantum_bench/bench/` | Implemented | Remains the single reproducible pipeline |
| Circuit analyzer | `src/quantum_bench/circuits/`, `src/quantum_bench/tn/network.py` | Basic circuit loading and tensor-network construction | Add structural analysis for dense, sparse, permutation, and diagonal route eligibility |
| Path optimizer | `src/quantum_bench/tn/planners.py`, `src/quantum_bench/tn/task_graph.py` | opt_einsum planner interface | Later add route-aware costs, not route-aware execution yet |
| Planner comparison | `src/quantum_bench/bench/planner_compare.py`, `src/quantum_bench/bench/planner_scoring.py` | Implemented analysis layer | Host path optimizer, cost-model, and target-aware planning analysis |
| WRAM slicer | `src/quantum_bench/targets/upmem/tile_plan.py`, `src/quantum_bench/targets/upmem/schedule.py` | Deterministic dense WRAM tile-plan records | Turn tile plans into executable preparation only after route execution is introduced |
| Data format conversion | `src/quantum_bench/formats/` | Deterministic host-side fixed-point records and utilities used by explicit dense preparation | Connect conversion records to route execution artifacts after routing gains execution |
| Dynamic heuristic router | `src/quantum_bench/routing/` | Analysis-only task-level router skeleton | Add preparation/execution-aware routing after data conversion and tiling mature |
| Dense GEMM route | `src/quantum_bench/providers/exact_tn/upmem_dense_placeholder.py`, `src/quantum_bench/routing/dense_prepare.py`, `src/quantum_bench/targets/upmem/schedule.py`, future `native/upmem/simplepim/` | Estimate-only placeholder plus one-task host preparation; no execution | SimplePIM-backed microkernel, then integrated TaskGraph route |
| SimplePIM bridge/probe | `src/quantum_bench/targets/upmem/simplepim.py`, `src/quantum_bench/targets/upmem/simplepim_microbench.py`, `src/quantum_bench/targets/upmem/dense_bridge.py`, future `native/upmem/simplepim/` | Availability probe, dry-run dense GEMM microbenchmark metadata, one-task dense preparation metadata, and file-based bridge manifests | SimplePIM-backed microkernel for one prepared task |
| Sparse route | Architecture slot only | Not implemented | Add SparseP feasibility plan after dense route skeleton |
| Heuristic/bypass route | Architecture slot only | Not implemented | Host-side row-swap/permutation prototype first |
| Optional TransPimLib/math support | Architecture slot only | Not implemented | Optional support slot, not first priority |
| Host aggregation | CPU route implicit, UPMEM future | Not separated | Make aggregation cost visible before claiming route speedups |
| Validation/reporting | `src/quantum_bench/validation/`, `src/quantum_bench/bench/summary.py`, `src/quantum_bench/plots/` | Implemented for current routes | Extend to task-routed outputs later |
| Benchmark artifacts | `runs/`, JSONL, summaries, path/task artifacts | Implemented | Add conversion/aggregation metrics as route execution matures |

## Current Route Interpretation

- `cpu_tn_einsum_exact`
  - Exact tensor-network CPU reference.
  - Executes `TaskGraph.tasks` task-by-task with NumPy.
  - Current role: correctness and CPU baseline.
- `quest_cpu_full_state_benchmark`
  - CPU full-state-vector baseline through the native QuEST runner.
  - Metrics-only and benchmark-only.
  - Current role: external baseline, not thesis novelty.
- `upmem_dense_int8_placeholder`
  - Estimate-only UPMEM dense candidate.
  - Uses `targets/upmem/` to record dense int8 feasibility, WRAM fit, transfer
    estimates, tile counts, and skip reasons.
  - Current role: dense route architecture slot and estimate layer, not native
    execution.

## Current Task Router Interpretation

`src/quantum_bench/routing/` is the task-level contract and dynamic-router
skeleton. It is analysis-only in this wave. It inspects each `ContractionTask`
and records route-slot decisions for:

- `dense_gemm`, which reuses `target_estimates["upmem_dense_int8"]`;
- `sparse`, a typed future slot;
- `heuristic_bypass`, a typed future slot;
- `transpim_support`, a typed future slot;
- `cpu_fallback`, the selected analysis fallback that maps to the existing
  graph-level `cpu_tn_einsum_exact` provider.

`src/quantum_bench/routing/dense_prepare.py` is a separate developer-facing
preparation boundary for one real `ContractionTask`. It consumes actual task
input tensors, lowers the contraction to GEMM operands by labels, applies
fixed-point conversion, attaches the UPMEM dense tile plan and SimplePIM probe
metadata, and validates the dequantized-input GEMM output against the CPU/NumPy
task result. It does not run during normal benchmark suites, does not write
normal benchmark artifacts, and does not execute SimplePIM or native UPMEM.

Normal benchmark runs write:

```text
cases/<case_id>/task_route_decisions.jsonl
cases/<case_id>/task_route_summary.json
```

These artifacts do not change execution behavior. They make selected, rejected,
skipped, unavailable, and fallback decisions auditable before routed execution
exists.

The `dense_gemm` task-route decisions also advertise the intended future
conversion requirement:

```text
conversion_required: true
intended_route_dtype: int8
conversion_format: fixed_point_symmetric
complex_policy: split_real_imag_last_axis
conversion_artifact: null
tile_plan_available: true
tile_plan_artifact: cases/<case_id>/target_estimates/upmem_dense_tile_plan.jsonl
backend: simplepim_unavailable
simplepim_available: false
simplepim_probe_status: unavailable
simplepim_command_path: null
simplepim_library_path: null
simplepim_skip_reason: SimplePIM is not configured; set SIMPLEPIM_BIN or put a SimplePIM command on PATH
```

The router does not perform conversion yet.

## Current Data Format Conversion Interpretation

`src/quantum_bench/formats/` owns shared host-side conversion records and
deterministic fixed-point utilities. It currently supports signed symmetric
`int8` and `int16` conversion for real tensors, plus explicit complex splitting
into a final real/imag axis. It records source dtype, route dtype, scale,
zero point, shape, byte counts, conversion time, clipping/saturation counts,
and quantization/dequantization error metrics.

This layer is shared by future UPMEM/SimplePIM dense, sparse, heuristic, and
simulator routes. It is not UPMEM-only. `targets/upmem/UpmemDataFormat` remains
the UPMEM schedule byte model; future route preparation should connect that
model to the shared fixed-point conversion records when tensors are actually
converted.

## Current WRAM Tile-Plan Interpretation

`src/quantum_bench/targets/upmem/tile_plan.py` is the explicit dense WRAM
slicer model. It produces deterministic tile-plan records for GEMM-like
`ContractionTask`s and writes them during normal benchmark runs:

```text
cases/<case_id>/target_estimates/upmem_dense_tile_plan.jsonl
```

`src/quantum_bench/targets/upmem/schedule.py` remains the compact estimate
facade. It derives `target_estimates["upmem_dense_int8"]` from the tile plan so
existing path summaries and route decisions keep one compact estimate key.

The semantics are intentionally separate:

- `fits_wram` in a tile plan means the selected modeled tile fits WRAM.
- `requires_tiling` means the full GEMM task does not fit as one full-task tile
  and needs decomposition.
- `tiling_implemented` remains `false`; this is not executable tiling.

For large tasks it is valid to see:

```text
fits_wram: true
requires_tiling: true
tiling_implemented: false
```

The memory model is conservative: output tile storage and accumulator workspace
are counted as separate WRAM reservations. This avoids hiding workspace costs
before real SimplePIM/UPMEM kernels exist.

## Route Maturity Model

| Level | Meaning |
|---:|---|
| 0 | Documented architecture slot |
| 1 | Typed interface/contract |
| 2 | Estimate-only implementation |
| 3 | Host-side prototype |
| 4 | SimplePIM/UPMEM-backed microkernel |
| 5 | Integrated TaskGraph route |
| 6 | Thesis-grade benchmarked route |

| Module or route | Current level | Next intended level |
|---|---:|---:|
| CPU exact TN route | 5 | 6 |
| QuEST CPU full-state baseline | 5 | 6 |
| UPMEM dense GEMM route | 3 | 4 |
| SimplePIM bridge/probe | 3 | 4 |
| UPMEM WRAM slicer/tile planner | 3 | 4 |
| Data format conversion layer | 3 | 4 |
| Dynamic task router | 1 | 2 |
| Heuristic/bypass route | 1 | 3 |
| Sparse route | 1 | 2 |
| Optional TransPimLib/math support | 1 | 2 |
| Host aggregation layer | 0/implicit | 1 |
| Planner comparison/cost model | 3 | 4 only if tied into route-aware planning later |
| UPMEM-aware path selection | 0/analysis only | 1 after route estimates mature |

Level 6 requires more than implementation. It requires reproducible suite
coverage, validation where output is produced, explicit skipped/rejected reasons,
environment capture, route metrics, and thesis-grade reporting.

## Future Route Contract

The current `ExecutionRoute` protocol is case/graph-level. The task-level
contract now lives in `src/quantum_bench/routing/` as an analysis-only skeleton.
Future UPMEM work should extend this contract toward:

```text
can_execute(task, context) -> eligibility decision and reason
estimate(task, context) -> cost, memory, transfer, and precision estimate
prepare(task, context) -> prepared task, converted data, and preparation metrics
execute(prepared_task, context) -> result and execution metrics
validate(result, reference) -> validation record when output is produced
```

Every route should report explicit failures, not hidden fallback. A rejected or
skipped route is useful thesis data if the reason is recorded.

## Dynamic Router Concept

The dynamic router is a host-side module. In the current skeleton it records
analysis decisions and preserves CPU fallback. As it matures it should:

- inspect each `ContractionTask`;
- enumerate candidate routes;
- evaluate route eligibility and cost estimates;
- record selected, rejected, skipped, and failed route decisions;
- preserve fallback behavior, typically CPU exact TN for correctness;
- keep optional hardware or external libraries skippable;
- write auditable task-level route-decision artifacts.

The router must not make UPMEM providers own global pathfinding. It receives an
existing `TaskGraph`, asks which local route can handle each bounded task, and
records the reasoning.

## SimplePIM Position

SimplePIM is the preferred first implementation path for UPMEM execution if it
is practical in the local environment. It should be treated as a provider layer
for bounded UPMEM tasks, not as the architecture itself.

The current SimplePIM boundary is probe-only. `src/quantum_bench/targets/upmem/simplepim.py`
checks environment and command discovery without executing external commands by
default. Missing SimplePIM configuration is normal and should produce
unavailable/skipped metadata, not failed benchmark runs. `SIMPLEPIM_HOME` alone
is only configured-but-unverified; it does not prove execution is available.

Raw UPMEM SDK code remains an escape hatch for dense hot paths, for experiments
where SimplePIM hides too much detail, or for comparison against SimplePIM. Raw
SDK code belongs under `native/upmem/raw_dense/`; future SimplePIM bridge code
belongs under `native/upmem/simplepim/`; Python route/provider code belongs
under `src/quantum_bench/providers/`; shared host-side UPMEM estimates, tiling,
and SimplePIM probe metadata belong under `src/quantum_bench/targets/upmem/`;
shared tensor conversion records and utilities belong under
`src/quantum_bench/formats/`.

## SimplePIM Dense Microbenchmark Artifact Schema

The explicit dry-run SimplePIM dense GEMM microbenchmark is invoked outside
normal benchmark suites:

```bash
PYTHONPATH=src ../.venv/bin/python -m quantum_bench.bench simplepim-microbench --dry-run --m 8 --k 8 --n 8
```

It writes only:

```text
runs/<timestamp>_simplepim_microbench/
  environment.json
  simplepim_microbench.json
```

Artifact fields:

- `schema_version`
- `microbench_id`
- `input`
- `status`
- `skip_reason`
- `error`
- `simplepim_probe`
- `fixed_point_spec`
- `tile_plan`
- `upmem_task_estimate`
- `input_shapes`
- `output_shape`
- `conversion_records`
- `conversion_time_s`
- `reference_time_s`
- `dequantization_time_s`
- `kernel_time_s`
- `host_aggregation_time_s`
- `total_time_s`
- `host_to_dpu_bytes`
- `dpu_to_host_bytes`
- `mram_to_wram_bytes`
- `validation_metrics`
- `external_command_executed`
- `execution_implemented`

`execution_implemented` is always `false` in this wave. `ready` means ready for
a future SimplePIM bridge attempt, not validated SimplePIM execution.

Status priority is deterministic:

1. invalid input or host preparation exception -> `failed`
2. `execute=True` or future execution requested -> `not_implemented`
3. executable tiling/aggregation required -> `not_implemented`
4. no SimplePIM executable/configuration -> `skipped`
5. `SIMPLEPIM_HOME` or `SIMPLEPIM_LIB` only -> `configured_but_unverified`
6. `SIMPLEPIM_BIN` or discovered command plus dry-run preparation -> `ready`

Host-side fixed-point conversion is performed only inside this explicit
microbenchmark dry-run command. Normal benchmark runs and task-route artifacts
do not invoke conversion.

The one-task dense preparation layer now lowers a real `ContractionTask` into
the future dense route preparation pipeline before full routed TaskGraph
execution. It connects:

- `simplepim_probe`
- `fixed_point_spec`
- `tile_plan`
- `input_shapes`
- conversion records
- validation metrics

`prepared` in this context means host-side preparation succeeded. It does not
mean SimplePIM execution happened. `external_command_executed` and
`execution_implemented` remain `false`.

## Dense Bridge Contract

`src/quantum_bench/targets/upmem/dense_bridge.py` defines the Python-to-native
boundary for one prepared dense task. It is not part of normal benchmark suite
execution and does not execute SimplePIM or native UPMEM.

The bridge input layout is:

```text
<bridge_dir>/
  input_manifest.json
  operands/
    left_quantized.npy
    right_quantized.npy
  references/
    expected_dequantized_output.npy
```

The input manifest records task identity, label/order metadata, GEMM dimensions,
fixed-point scale/dequantization metadata, tile-plan metadata, and relative blob
paths. It never embeds raw arrays in JSON. `.npy` is the Wave 2C.7 blob format
because it preserves dtype and shape metadata for inspection; raw native buffers
are deferred until a real bridge exists.

The mock bridge writes:

```text
<bridge_dir>/
  outputs/
    mock_dequantized_output.npy
  output_manifest.json
```

The mock backend is `mock_numpy_dequantized`. It reads the manifest and operand
blobs, dequantizes them, performs NumPy GEMM, restores output label order, and
validates against `references/expected_dequantized_output.npy`. This proves the
file boundary only. `external_command_executed` and `execution_implemented`
remain `false`.

Wave 2C.8 adds a backend adapter interface on top of this file boundary:

- `mock_numpy_dequantized` is implemented and executes only local NumPy.
- `simplepim_external` is a future external-process backend and records planned
  invocation metadata only.
- `simplepim_external_stub` is a non-executing external-process contract stub.
  It may call a Python stub process only when `execute_external=true` and
  `SIMPLEPIM_STUB_BIN` is configured. It validates that the cross-process file
  contract can consume `input_manifest.json` and write `output_manifest.json`;
  it does not write an output blob and does not run a SimplePIM/native kernel.

`execute_dense_bridge(...)` is the generic entrypoint. For `simplepim_external`
it validates the input manifest and blob metadata first, then writes
`output_manifest.json` even when skipped or not implemented. Exact non-execution
reasons are:

- `simplepim_unavailable` when no SimplePIM command/configuration is present;
- `simplepim_external_execution_disabled` when a SimplePIM command/config exists
  but `execute_external=false`;
- `simplepim_external_execution_not_implemented` when `execute_external=true`;
- `invalid_bridge_input_manifest` when manifest/blob validation fails before any
  backend decision.

For `simplepim_external_stub`, the only subprocess exception in this runtime is:

- backend must be `simplepim_external_stub`;
- `execute_external` must be `true`;
- `SIMPLEPIM_STUB_BIN` must point to the stub script. Relative values are
  resolved from the current working directory, so validation commands run from
  `thesis/implementation`;
- the adapter invokes the script with `sys.executable`, so executable
  permissions and a shebang are not required.

The stub writes the same `dense_bridge_output` manifest kind as the mock backend,
with `backend="simplepim_external_stub"`, `status="stub_executed"`,
`output_blob=null`, `validation_metrics.status="not_applicable"`,
`external_command_executed=true`, `execution_implemented=false`, and
`metadata.native_kernel_executed=false`. A valid stub output manifest must have
`status="stub_executed"` for the adapter to return `stub_executed`. If the stub
exits nonzero, fails to write a valid output manifest, or writes
`status="failed"`, the adapter returns `failed` rather than `stub_executed`.

Invocation metadata uses relative bridge paths such as `input_manifest.json`,
`output_manifest.json`, and `outputs/simplepim_output.npy`. It may include a
SimplePIM command path from the environment/probe, but it must not include the
absolute bridge directory or raw arrays.

## Developer One-Task Dense Bridge Harness

`src/quantum_bench/bench/dense_task_bridge.py` connects one real benchmark
`ContractionTask` to the dense preparation and bridge boundary. It is a
developer-only harness, not a provider route and not a normal benchmark suite
mode. It is mock-by-default and does not change task routing; `dense_gemm`
remains non-selected in normal `task_route_decisions.jsonl`.

Example:

```bash
PYTHONPATH=src ../.venv/bin/python -m quantum_bench.bench dense-task-bridge --case bell_2q --task-index 0 --backend mock_numpy_dequantized
```

For an explicitly selected later task, the harness can opt into CPU replay input
materialization:

```bash
PYTHONPATH=src ../.venv/bin/python -m quantum_bench.bench dense-task-bridge --case bell_2q --task-index 1 --materialization cpu-replay --backend mock_numpy_dequantized
```

The command writes:

```text
runs/<timestamp>_dense_task_bridge/
  environment.json
  dense_task_bridge_summary.json
  bridge/
    input_manifest.json
    operands/
    references/
    output_manifest.json
    outputs/
```

The harness:

- builds a builtin circuit workload;
- constructs the `TensorNetwork` and `TaskGraph`;
- selects one directly materializable task, or the requested `--task-index`;
- optionally materializes selected task inputs with `--materialization cpu-replay`;
- calls dense preparation with actual task tensors;
- writes bridge input blobs and metadata;
- executes the selected bridge backend through `execute_dense_bridge(...)`;
- writes `dense_task_bridge_summary.json`.

By default, only initial-input tasks are materialized. Later tasks that require
intermediate tensors return `unsupported` with
`intermediate_tensor_inputs_not_materialized`. With an explicit
`--task-index` and `--materialization cpu-replay`, the harness calls
`src/quantum_bench/tn/materialize.py` to replay predecessor CPU contractions
until the selected task inputs are available. Omitted `--task-index` still uses
the existing first initial-input auto-selection path and does not call the
materializer.

CPU replay is developer-only input materialization. It is not the normal CPU
execution path, does not select `dense_gemm` in normal routing, and does not
implement full routed TaskGraph execution. `peak_materialized_bytes` in the
summary means retained runtime tensor-map bytes during replay, not process RSS.
Dead-tensor release is intentionally not implemented in this wave and is
reported as `dead_tensor_release_implemented=false`.

The summary contains a JSON-safe `materialization` object with `mode`, `status`,
`reason`, selected input IDs, input sources, replayed task IDs, replay timing,
peak materialized bytes, dead-tensor-release status, and per-step replay metrics.
It never embeds raw tensors.

Status mapping:

- `completed` only when the bridge backend reports a successful local/mock
  status such as `mock_executed`, or the non-kernel external contract status
  `stub_executed`;
- `skipped` for non-executing SimplePIM external backend outcomes;
- `unsupported` for task selection, materialization, or shape-preparation
  issues;
- `failed` for unexpected harness or malformed bridge failures.

`simplepim_external` may write `input_manifest.json` and a non-executing
`output_manifest.json`, but it does not write an output blob and does not call a
subprocess. `simplepim_external_stub` may write `input_manifest.json` and a
non-executing `output_manifest.json` after invoking the configured stub process,
but it also does not write an output blob. Summary artifact paths are relative
to the run directory. CLI output may print absolute `run_dir` and `summary_path`
for convenience.

`execution_implemented` remains `false` in this wave. For
`simplepim_external_stub`, `external_command_executed=true` means only that the
external stub process was invoked; `metadata.native_kernel_executed=false`
records that no SimplePIM/native kernel ran.

## Dense Route Coverage Analyzer

`src/quantum_bench/bench/dense_route_coverage.py` is the developer-only
readiness analyzer between the one-task bridge harness and future routed dense
execution. It iterates every `ContractionTask` in selected cases, materializes
task inputs with CPU replay, calls dense preparation, reads fixed-point
conversion and UPMEM tile-plan metadata, and reports whether each task is
eligible for bridge-manifest creation. It can optionally write a capped number
of bridge artifacts and invoke the non-executing external stub contract.

Example:

```bash
PYTHONPATH=src ../.venv/bin/python -m quantum_bench.bench dense-route-coverage --case bell_2q
```

Suite mode iterates workloads/cases only and uses the suite's normalized default
planner. It does not multiply by routes or planner-comparison variants.

The command writes:

```text
runs/<timestamp>_<case_or_suite>_dense_route_coverage/
  environment.json
  dense_route_coverage.json
  dense_route_coverage.csv
  dense_route_coverage_summary.md
  cases/<case_id>/dense_route_coverage.jsonl
```

By default, it writes no bridge blobs and calls no subprocess. Optional stub
checking requires `--bridge-backend simplepim_external_stub`,
`--execute-external`, `--max-bridge-artifacts > 0`, and `SIMPLEPIM_STUB_BIN`.

Readiness levels are analysis labels, not route selection:

- `not_materializable`
- `materialized_only`
- `blocked_unsupported_shape`
- `blocked_requires_tiling`
- `dense_prepare_failed`
- `dense_prepare_ready`
- `bridge_manifest_ready`
- `external_stub_ready`

This artifact family is thesis evidence for dense route feasibility and
bottlenecks: how many tasks can be materialized, prepared, represented as dense
GEMM, fit the current WRAM model without executable tiling, and satisfy the
future bridge file contract. It does not select `dense_gemm`, does not execute
SimplePIM/native UPMEM, and does not replace normal provider execution.

## Planner Comparison Position

The planner comparison work belongs to:

- host path optimizer analysis;
- cost-model analysis;
- target-aware planning analysis.

It is not an execution route and it does not replace the dynamic router. It is
evidence that FLOP-oriented paths can differ from modeled UPMEM-pressure paths,
which motivates route-aware costs later.

## Near-Term Roadmap

- `2C.5` SimplePIM dense GEMM dry-run microbenchmark scaffold
- `2C.6` lower one real `ContractionTask` into dense route preparation
- `2C.7` dense native/SimplePIM bridge contract for prepared payloads
- `2C.8` executable dense bridge adapter interface, execution disabled by default
- `2C.9` developer-only one-task dense bridge harness
- `2C.10` CPU replay materialization for later one-task bridge inputs
- `2C.11` non-executing external SimplePIM dense bridge stub
- `2C.12` dense route readiness coverage analyzer over full `TaskGraph`
- `2D.1` heuristic/bypass route prototype
- `2D.2` sparse route feasibility and SparseP integration plan
- `2D.3` host aggregation/PID-Comm-inspired reporting
- `2D.4` optional TransPimLib support slot
- `2E.1` revisit UPMEM-aware path selection using route-aware costs

The next implementation wave should build on the coverage evidence and either
start a real dense execution microkernel path or introduce the route-selection
analysis layer that consumes coverage/readiness results.

## Future Codex Instruction

Before proposing architecture, route, planner, target, or native-code changes,
read:

- `README.md`
- `ARCHITECTURE.md`
- `docs/runtime_architecture_map.md`
- `../CODEX_IMPLEMENTATION_DIRECTION.md`
- `../CODEX_UPMEM_ARCHITECTURE_DIRECTION.md`

Future plans should preserve this map unless the user explicitly approves an
architecture change.

## Non-Goals For This Scaffold

- Do not implement SimplePIM execution.
- Do not implement native UPMEM.
- Do not implement dense, sparse, heuristic, or math-support kernels.
- Do not change benchmark execution behavior.
- Do not change route IDs.
- Do not change suite schema requirements.
- Do not restructure packages.
- Do not remove or weaken planner comparison work.
- Do not add public APIs unless a later implementation wave requires them.
