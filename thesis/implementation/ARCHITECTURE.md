# Quantum Bench Architecture

This directory contains the active thesis implementation. Older prototypes and
historical runs live outside this package under `../legacy/`. The active code
should be understandable and movable as a standalone implementation rooted at
`thesis/implementation`.

The project is a route-aware exact tensor-network quantum-circuit simulation
runtime. The CPU owns global orchestration, planning, dispatch, validation, and
reporting. UPMEM-style PIM executes bounded local contraction work only when an
explicit route/backend contract supports it.

## Source Layout

```text
implementation/
  configs/                 benchmark suites and thesis matrix inputs
  external/                implementation-local external Git submodules
  native/quest_cpu/        QuEST C runner used by the CPU full-state baseline
  native/upmem/            UPMEM SDK code and future SimplePIM/native bridge code
  scripts/                 helper commands
  src/quantum_bench/
    bench/                 one CLI, runner, summaries, evaluation harnesses
    circuits/              builtin workload construction
    core/                  JSON and typed record helpers
    environment/           environment capture
    formats/               fixed-point and tensor-format conversion utilities
    plots/                 matplotlib plots from generated summaries
    providers/             benchmark-executable graph-level routes
    routing/               task-level routing, policy, and dense preparation
    targets/upmem/         UPMEM estimates, tile plans, bridge, probes, analysis
    tn/                    tensor-network construction, planning, materialization
    validation/            numerical validation metrics
  tests/                   pytest suite
```

There is one benchmark CLI:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src ../.venv/bin/python -m quantum_bench.bench run --suite configs/suites/smoke.yml
```

## Terminology

| Term | Meaning | Example |
|---|---|---|
| `route_id` | Graph-level benchmark provider identity | `cpu_tn_einsum_exact` |
| `backend_id` | Developer bridge/backend implementation identity | `upmem_sdk_simulator_dense` |
| `route_category` | Thesis benchmark-matrix category | `upmem_tn_runtime` |
| `execution_class` | Internal UPMEM scheduler/memory class | `L2_SINGLE_DPU_MRAM` |
| `kernel_family` | Kind of local work delegated to a backend | `dense_gemm`, `generic_loop_fallback` |
| `evidence_type` | How the result was obtained | `measured`, `simulated`, `modeled`, `planned` |
| `execution_scope` | What the evidence covers | `full_circuit`, `task_level`, `model_only` |

Keep these distinct. In particular, `upmem_sdk_simulator_dense` is not a final
benchmark route. It is a developer bridge backend used to produce task-level
UPMEM SDK simulator evidence. The final UPMEM matrix category remains one
runtime: `upmem_tn_runtime`.

## End-To-End Direction

```text
Quantum circuit
  -> tensor network
  -> contraction path and TaskGraph
  -> host-side UPMEM-aware cost and readiness analysis
  -> task scheduling / route decisions
  -> UPMEM execution for supported bounded local tasks
  -> host aggregation
  -> CPU/QuEST validation and benchmark artifacts
```

The intended UPMEM path is not CPU fallback hidden behind route metadata. CPU
fallback remains essential for validation and diagnostics, but UPMEM evaluation
must report which tensor-network contraction work actually executed on UPMEM,
which work was modeled only, which work fell back, and why.

## Current Routes

| Route ID | Role | Output authority |
|---|---|---|
| `cpu_tn_einsum_exact` | Exact tensor-network CPU reference | Authoritative full tensor output |
| `quimb_tn_exact` | External exact tensor-network CPU backend | Comparable final tensor output with Quimb/cotengra provenance |
| `quest_cpu_full_state_benchmark` | QuEST CPU full-state baseline | Benchmark-only metrics |
| `quest_cpu_full_state_exact` | QuEST CPU full-state comparable baseline | Statevector output for small deterministic unitary circuits |
| `upmem_dense_int8_placeholder` | UPMEM dense candidate placeholder | Estimate/skip metadata only |

The metrics-only QuEST route is useful for timing comparison, but it is not an
output-comparable route. `quest_cpu_full_state_exact` is additive and compares
QuEST full-state output against CPU TN output on shared deterministic circuit
semantics.
`quimb_tn_exact` is additive as the first external tensor-network execution
backend. It is exact CPU execution, not GPU execution, and does not replace the
simple internal `cpu_tn_einsum_exact` baseline.

## UPMEM Runtime Status

`upmem_tn_runtime` is the final UPMEM benchmark category. Its internal execution
classes currently stand here:

| Class | Current status | Complex support | Notes |
|---|---|---|---|
| `L1_WRAM` | Implemented for task-level UPMEM SDK simulator dense bridge subset | Split-complex supported when manifest layout is explicit | Padded direct GEMM, simulator only |
| `L2_SINGLE_DPU_MRAM` | Implemented for task-level UPMEM SDK simulator real-valued tiled subset | Complex rejected as `complex_l2_not_implemented` | One-DPU MRAM operands, WRAM output tiles |
| `L3_MULTI_DPU` | Model-only/planned | Not implemented | Needs distributed scheduling and communication |
| `L4_OUT_OF_SCOPE` | Model-only pressure boundary | Not applicable | Exceeds aggregate modeled UPMEM memory |

Current UPMEM evidence is task-level simulator evidence. It must not be reported
as full-circuit acceleration.

## Core Subsystems

| Subsystem | Location | Status |
|---|---|---|
| Benchmark runner and suite loading | `src/quantum_bench/bench/runner.py` | Implemented |
| Circuit and tensor-network construction | `circuits/`, `tn/` | Implemented for current builtins |
| CPU exact task execution | `providers/exact_tn/cpu_einsum.py` | Implemented |
| External exact TN execution | `providers/exact_tn/quimb_tn.py` | Implemented initial Quimb backend |
| TaskGraph materialization replay | `tn/materialize.py` | Developer helper |
| Task-level routing and policy | `routing/` | Analysis/shadow only |
| Fixed-point conversion | `formats/fixed_point.py` | Implemented host-side utilities |
| UPMEM schedule and tile model | `targets/upmem/schedule.py`, `tile_plan.py` | Implemented model and L1/L2 bridge metadata |
| Dense bridge manifests | `targets/upmem/dense_bridge.py` | Implemented |
| Generic fallback preparation and bridge | `routing/generic_prepare.py`, `targets/upmem/generic_bridge.py` | Implemented MVP for small real binary contractions |
| UPMEM SDK simulator dense runner | `native/upmem/simplepim/upmem_sdk_dense*` | Implemented L1/L2 subsets |
| UPMEM SDK simulator generic loop runner | `native/upmem/simplepim/upmem_sdk_generic_loop*` | Implemented correctness/coverage MVP |
| Strict UPMEM TaskGraph runtime | `bench/upmem_taskgraph_runtime.py`, `targets/upmem/taskgraph_runtime.py` | Implemented MVP for small sequential TaskGraphs in SDK simulator mode |
| Suite-level UPMEM MVP benchmark | `bench/upmem_mvp_benchmark.py` | Implemented report regeneration pipeline over strict runtime variants |
| Benchmark reporting and retention | `bench/reporting.py`, `bench/result_artifacts.py` | Implemented canonical normalized records, compact retention, report regeneration, and run comparison |
| PIM bridge evaluation | `bench/pim_bridge_eval.py` | Developer task-level evidence harness |
| PIM frontier analysis | `bench/pim_frontier_analysis.py`, `targets/upmem/frontier.py` | Model-only analysis |
| Benchmark matrix report | `bench/benchmark_matrix_report.py` | Thesis scaffold |
| External library scan | `targets/upmem/external_libs.py` | Evidence-only feasibility scan |

## External Source Trees

`implementation/external` is canonical for active implementation dependencies
and is populated by Git submodules. `../legacy/extern` is historical fallback
only and should not be required by a standalone checkout.

| Source | Current role |
|---|---|
| `external/QuEST` | Submodule dependency for the CPU full-state C runner |
| `external/SimplePIM` | Submodule candidate for UPMEM compute/runtime abstraction for L1/L2 and local tile compute inside L3 |
| `external/PID-Comm` | Submodule candidate for communication/orchestration across L1/L2/L3, strongest for L3 distributed contraction |
| ATiM | SLR-derived tensor-kernel autotuning candidate; no submodule until the authoritative URL is confirmed |
| SparseP, PRISM, PyGim | Sparse or irregular PIM references, not integrated |
| PIM-LLM GEMM | Optimized GEMM design reference, not integrated |
| TransPimLib | Optional special-math support candidate, not integrated |
| Native UPMEM SDK | Current control/fallback implementation |

Capability scans must not turn text markers into proven support. For example,
`gemm`, `int8`, or `allreduce` evidence in source files is recorded as evidence
only until a bounded build/run/API check proves the capability.

## Developer Harnesses

These commands are intentionally outside normal benchmark provider execution:

| Command | Purpose | Evidence boundary |
|---|---|---|
| `dense-task-bridge` | One real task through dense preparation and bridge backend | Task-level only |
| `generic-task-bridge` | One real task through the unoptimized generic fallback bridge | Task-level simulator evidence only |
| `upmem-taskgraph-runtime` | Sequential full TaskGraph through UPMEM SDK DPU programs using SDK simulator mode | Small strict UPMEM code-path evidence, not hardware timing |
| `upmem-mvp-benchmark` | Suite-level CPU reference plus strict UPMEM runtime variants | Reproducible MVP report pipeline, not optimized performance evidence |
| `simulation-backend-compare` | QuEST full-state, internal CPU TN, and external TN comparison on shared deterministic circuits | CPU backend comparison scaffold |
| `simulation-backend-probe` | Optional simulation library and GPU feasibility probe | Feasibility metadata only; no fake GPU records |
| `report-run` | Regenerate derived CSV/Markdown/metrics/validation/plots from retained artifacts | Reporting only; non-destructive for execution artifacts |
| `prune-run` | Explicit compact artifact pruning for new MVP run layouts | Destructive only when explicitly requested; idempotent |
| `dense-route-coverage` | All-task readiness and bridge eligibility | Analysis only |
| `shadow-routed-runtime` | Full graph with CPU fallback authoritative and shadow route evidence | Diagnostic only |
| `pim-bridge-eval` | Capped task-level simulator execution across workloads | Task-level simulator evidence |
| `pim-frontier-analysis` | Memory-level and parallelism-frontier model | Model-only |
| `benchmark-matrix-report` | Final thesis matrix scaffold | Report scaffold |
| `compare-results` | Artifact-driven comparison of generated result files | Reporting only |
| `compare-runs` | Baseline/candidate run comparison from normalized records | Reporting only |
| `upmem-env-check` | UPMEM SDK / SimplePIM environment bring-up | Environment evidence |
| `upmem-external-libs-check` | SimplePIM/PID-Comm/native SDK candidate report | Feasibility evidence |

`shadow-routed-runtime` always keeps CPU fallback authoritative. By contrast,
`upmem-taskgraph-runtime` is strict: every contraction task must execute through
the UPMEM SDK DPU program path, and only gathered UPMEM task outputs may feed
later runtime tensors. CPU exact output is used for final validation only.
`upmem-mvp-benchmark` preserves that boundary while running the strict runtime
across a suite and reporting CPU reference timing separately from UPMEM SDK
simulator timing.

New MVP benchmark runs write `run_manifest.json` and root
`normalized_records.jsonl`. The normalized records are canonical for
`compare-results`; recursive child-summary discovery remains for standalone
task/runtime outputs. Compact retention is the default MVP mode: it removes
debug-heavy bridge blobs and native `runner_work/**`, keeps audit summaries and
task metrics, and records intentional pruning so reports can distinguish pruned
artifacts from missing artifacts.

Simulation backend comparison has four suite tiers. `simulation_backend_compare_quick.yml`
is a validation suite. `simulation_backend_compare_thesis_small.yml` and
`simulation_backend_compare_scaling.yml` are bounded local suites for readable
trend plots. `simulation_backend_compare_compute_medium.yml` is a CPU-only,
GPU-independent compute-focused suite with warmups and repeats. GPU execution
belongs in `simulation_backend_compare_gpu_medium.yml` only after a route proves
real GPU computation; otherwise GPU candidates stay in probe metadata.
Scaling plots group by circuit family; relative runtime plots use only rows
with a valid measured QuEST anchor and are labeled as relative backend timing,
not speedup.

GPU candidate reporting is SOTA-oriented but evidence-bound. QuEST HIP/CUDA,
Qiskit Aer GPU, CUDA Quantum, cuQuantum, Quimb/cotengra GPU execution, and
generic tensor GPU paths can appear as candidates. Source support such as
`ENABLE_HIP` or `ENABLE_CUDA` is not a benchmarkable route until build, run, and
minimal GPU execution are verified. Generated GPU/native build artifacts must
stay out of submodule source trees or be ignored/cleaned.

## Synthetic Pressure Workloads

Synthetic pressure workloads live in `targets/upmem/synthetic_pressure.py` and
pressure suites. They are not quantum circuits. They exist to expose L2/L3/L4
memory and parallelism boundaries without allocating real tensors.

They must declare:

```text
workload_type: synthetic_pressure
execution_scope: model_only
not_real_quantum_circuit: true
```

Normal circuit loading and normal benchmark execution must reject
`circuit.kind: synthetic_pressure` with an explicit analysis-only error.

## Artifact Policy

Runs write timestamped directories under `runs/`. Those artifacts are generated
and should not be treated as source. The repo should keep code, configs, tests,
docs, and implementation-local external source manifests. Native build outputs,
Python caches, and local benchmark runs are generated.

Plots follow a simple rule: bars compare cases/routes/categories; lines show
scaling within one comparable circuit family. Do not connect unrelated circuit
families with one line.

## Design Rules

- Keep one benchmark runner and one comparable artifact schema.
- Use kernel-family work-share reporting before interpreting benchmark timings.
- Keep route IDs, backend IDs, and internal execution classes distinct.
- Keep UPMEM as one final benchmark category.
- Keep optional tools skippable with explicit reasons.
- Do not claim speedup until execution scope, validation, transfer, conversion,
  aggregation, and fallback work share are visible.
- Do not let synthetic pressure workloads enter normal benchmark execution.
- Do not add compatibility aliases during this design stage unless explicitly
  required by a migration.
