# Runtime Architecture Map And Roadmap

This document records the current thesis runtime status and the next
implementation direction. It is intentionally concise: `README.md` is the
quickstart, `ARCHITECTURE.md` is the stable boundary document, and this file is
the roadmap/status map.

## Thesis Runtime Direction

The thesis implementation is an exact tensor-network quantum-circuit simulation
runtime for UPMEM-style PIM. The CPU is the control plane:

- circuit analyzer;
- tensor-network builder;
- contraction planner;
- WRAM/MRAM/resource modeler;
- route dispatcher;
- validator;
- reporter.

UPMEM-style PIM is the data plane. It should execute bounded tensor-network
contraction work only when the route/backend contract proves the task is
eligible. CPU fallback remains a correctness and diagnostic tool, but it should
not be mistaken for UPMEM evaluation.

```text
Quantum circuit
  -> tensor network
  -> TaskGraph and contraction path
  -> UPMEM-aware task analysis
  -> route/backend choice
  -> UPMEM local task execution where supported
  -> host aggregation
  -> validation and artifacts
```

## Current Implementation Status

| Area | Status | Evidence boundary |
|---|---|---|
| Benchmark runner | Implemented | One CLI and one timestamped artifact layout |
| CPU exact TN | Implemented | Full output, validation authority |
| QuEST CPU full-state | Implemented | Metrics-only baseline |
| External exact TN | Implemented initial backend | Quimb exact CPU TN route with dependency provenance |
| Planner comparison | Implemented | Analysis of FLOP vs modeled UPMEM pressure |
| Fixed-point conversion | Implemented | Host-side deterministic utilities |
| Dense preparation | Implemented | One-task preparation and validation |
| Dense bridge manifests | Implemented | `.npy` blobs plus JSON manifests |
| UPMEM SDK simulator L1 | Implemented subset | Task-level simulator dense execution |
| UPMEM SDK simulator L2 | Implemented subset | Task-level real-valued tiled simulator execution |
| Strict UPMEM TaskGraph runtime | Implemented MVP | Small sequential TaskGraphs execute through UPMEM SDK DPU programs using SDK simulator mode |
| Suite-level UPMEM MVP benchmark | Implemented MVP | Runs strict runtime variants across suites and writes canonical normalized records |
| Benchmark reporting and artifact hygiene | Implemented MVP | Report regeneration, compact retention, run manifests, and run-to-run comparison |
| PIM bridge eval | Implemented | Task-level evidence, not full-circuit speedup |
| Frontier analysis | Implemented | Model-only memory/parallelism analysis |
| Benchmark matrix report | Implemented | Thesis scaffold, not final result table |
| SimplePIM GEMM | Planned/candidate | Capability not proven |
| PID-Comm execution | Planned/candidate | Capability not integrated |
| L3 distributed UPMEM | Model-only/planned | No execution |
| Hardware UPMEM | Not implemented | No hardware evidence |
| GPU TN/full-state | Feasibility only | ROCm/GPU probes only; no records without real execution |
| Sparse/irregular routes | Planned | Not implemented |

## UPMEM Internal Execution Classes

UPMEM remains one final runtime category: `upmem_tn_runtime`.

| Internal class | Current maturity | Complex status | Next step |
|---|---|---|---|
| `L1_WRAM` | Task-level UPMEM SDK simulator subset | Split-complex supported when manifest layout is explicit | Broaden supported shapes and compare against SimplePIM feasibility |
| `L2_SINGLE_DPU_MRAM` | Task-level UPMEM SDK simulator real-valued subset | Complex rejected as `complex_l2_not_implemented` | Add complex L2 or keep rejection explicit; improve tiling coverage |
| `L3_MULTI_DPU` | Model-only | Not implemented | Design distributed contraction and communication layer |
| `L4_OUT_OF_SCOPE` | Model-only boundary | Not applicable | Use as pressure boundary, not execution target |

Current L1/L2 evidence is task-level simulator evidence. It does not prove
full-circuit UPMEM acceleration.

## Route, Backend, And Evidence Rules

- `route_id` identifies graph-level benchmark providers.
- `backend_id` identifies developer bridge implementations.
- `execution_class` identifies an internal UPMEM memory/scheduler class.
- `route_category` identifies a thesis benchmark matrix category.

Examples:

| Concept | Example |
|---|---|
| Route ID | `cpu_tn_einsum_exact` |
| External TN route ID | `quimb_tn_exact` |
| Comparable full-state route ID | `quest_cpu_full_state_exact` |
| Backend ID | `upmem_sdk_simulator_dense` |
| Execution class | `L2_SINGLE_DPU_MRAM` |
| Route category | `upmem_tn_runtime` |

Do not add `upmem_l1`, `upmem_l2`, or `upmem_l3` as final benchmark routes.
L1/L2/L3 are internal scheduler choices inside `upmem_tn_runtime`.

## Complex-Valued Correctness

Complex-valued quantum correctness is mandatory for the final thesis runtime.
Current status must be reported honestly:

| Path | Current complex support |
|---|---|
| CPU exact TN | Supported by NumPy complex tensors |
| Quimb exact TN | Supported by Quimb complex tensors |
| QuEST CPU full-state | Metrics-only route plus output-comparable exact route for small deterministic circuits |
| Dense preparation | Supports explicit split real/imag representation where metadata proves layout |
| UPMEM L1 simulator | Split-complex four-GEMM path supported for explicit manifest layout |
| UPMEM L2 simulator | Real-valued only; complex rejected as `complex_l2_not_implemented` |
| L3 distributed UPMEM | Not implemented |

Do not hide complex limitations by falling back silently to CPU. Record fallback
or rejection reasons in artifacts.

## External Candidate Registry

`implementation/external` is canonical and populated by Git submodules.
`../legacy/extern` is historical only.

The SLR-derived candidate registry should stay extensible. Add future papers or
libraries by adding one row with role, applicable execution classes, kernel
family, status, evidence level, blocker, and next action.

| Candidate | Role | Classes | Kernel family | Status | Evidence level | Blocker | Next action |
|---|---|---|---|---|---|---|---|
| Native UPMEM SDK | Control/fallback implementation | L1, L2, future L3 | dense GEMM, low-level kernels | Partially implemented | Task-level simulator | Not final productivity abstraction | Keep as correctness/control baseline |
| SimplePIM | Target UPMEM compute/runtime abstraction for L1/L2 and local tile compute inside L3 | L1, L2, L3 local tile compute | dense and structured local kernels | Submodule present, not integrated for GEMM | Source evidence only | Ready GEMM primitive not proven | Decide whether to implement SimplePIM dense bridge or keep native SDK fallback |
| PID-Comm | Target communication/orchestration substrate across L1/L2/L3, strongest for L3 distributed contraction | L1/L2 orchestration, L3 communication | collectives, scatter/gather, reductions | Submodule present, not integrated | Source evidence only | Build/run and simulator/hardware fit unproven | Evaluate for L3 distributed contraction protocol |
| ATiM | Tensor-kernel autotuning candidate | L1, L2, L3 local kernels | autotuned tensor kernels | Planned only; authoritative URL not confirmed | SLR reference | Needs source/provenance and feasibility scan | Add a submodule only after the URL is confirmed |
| SparseP | Sparse PIM reference | Sparse route | sparse/zero-heavy kernels | Not integrated | SLR reference | No route contract yet | Plan sparse route after dense path stabilizes |
| PRISM | Irregular/sparse reference | Sparse/heuristic route | irregular kernels | Not integrated | SLR reference | No route contract yet | Record useful kernel ideas only |
| PyGim | Sparse/irregular reference | Sparse/heuristic route | irregular kernels | Not integrated | SLR reference | No route contract yet | Record useful kernel ideas only |
| PIM-LLM GEMM | Optimized GEMM design reference | L1, L2, L3 local tiles | GEMM | Not integrated | SLR reference | Not quantum-specific | Use as kernel design reference |
| TransPimLib | Optional special-math support | Optional route | special math | Not integrated | SLR reference | Not on critical path | Keep optional support slot |

Capability scans must separate source evidence from proven capability. Text
markers such as `gemm`, `int8`, `reduce`, or `allreduce` are not proof that a
usable API or kernel exists.

## Future Kernel Work-Share Reporting

This is a future report-design requirement, not new schema work in this cleanup
wave. Final reports should explain what fraction of tensor-network work falls
into each family:

- dense GEMM share;
- quantum-structured kernel share;
- sparse or zero-heavy share;
- communication/collective share;
- generic fallback share;
- unsupported or CPU-fallback share.

The goal is to avoid claiming that the thesis runtime is only GEMM on UPMEM.
GEMM is the current backbone, not the complete design.

## Canonical Workflows

| Workflow | Command | Interpretation |
|---|---|---|
| Normal smoke benchmark | `quantum_bench.bench run --suite configs/suites/smoke.yml` | Current runner/provider health |
| Environment check | `quantum_bench.bench upmem-env-check` | Local UPMEM/SimplePIM setup evidence |
| External library check | `quantum_bench.bench upmem-external-libs-check` | Candidate feasibility evidence |
| Simulation backend probe | `quantum_bench.bench simulation-backend-probe` | Optional Quimb/cotengra/GPU feasibility metadata |
| PIM bridge eval | `quantum_bench.bench pim-bridge-eval ...` | Task-level simulator evidence |
| Strict UPMEM runtime | `quantum_bench.bench upmem-taskgraph-runtime ...` | Small full-TaskGraph SDK simulator code-path evidence |
| Suite MVP benchmark | `quantum_bench.bench upmem-mvp-benchmark ...` | Suite-level CPU reference, UPMEM runtime, and report regeneration |
| Simulation backend comparison | `quantum_bench.bench simulation-backend-compare ...` | QuEST full-state, internal CPU TN, and external TN output comparison |
| Report regeneration | `quantum_bench.bench report-run ...` | Non-destructive derived report refresh |
| Compact pruning | `quantum_bench.bench prune-run ...` | Explicit artifact pruning for new MVP run layouts |
| Frontier analysis | `quantum_bench.bench pim-frontier-analysis ...` | Model-only memory/parallelism evidence |
| Matrix report | `quantum_bench.bench benchmark-matrix-report ...` | Thesis benchmark scaffold |
| Generic fallback bridge | `quantum_bench.bench generic-task-bridge ...` | Task-level coverage/correctness evidence |
| Result comparison | `quantum_bench.bench compare-results ...` | Artifact-driven report only |
| Run comparison | `quantum_bench.bench compare-runs ...` | Baseline/candidate comparison from normalized records |

Developer diagnostics such as `dense-task-bridge`, `dense-route-coverage`,
`shadow-routed-runtime`, `simplepim-microbench`, and `compare-planners` are
useful for implementation, but they are not final benchmark claims.

## Roadmap

Near-term priorities:

1. Keep native UPMEM SDK L1/L2 as the control implementation and fallback.
2. Harden complex support reporting: L1 split-complex, L2 real-only, L3 absent.
3. Use the generic loop fallback only for correctness coverage and route
   validation of small binary contractions. It is not a performance kernel.
4. Use the strict UPMEM TaskGraph runtime as the baseline pipeline for replacing
   individual kernels without CPU contraction fallback.
5. Use the suite-level MVP benchmark plus `report-run`/`compare-runs` to
   regenerate CPU reference, strict UPMEM, kernel-family, quantization,
   unsupported-reason, and run-delta reports from artifacts.
6. Decide whether SimplePIM can implement the dense local tile compute path
   cleanly enough to replace or complement native SDK L1/L2.
7. Evaluate PID-Comm as the communication/orchestration substrate across
   L1/L2/L3, with strongest importance for L3 distributed contraction.
8. Design L3 multi-DPU contraction around explicit communication, aggregation,
   and load-balance artifacts.
9. Add sparse/irregular kernel planning only after the dense and communication
   boundaries remain stable.
10. Revisit route-aware path planning once route execution evidence includes
   memory, transfer, tiling, communication, fallback, and validation costs.

Do not add performance or speedup claims until the report distinguishes
full-circuit execution from task-level simulator execution and model-only
analysis.

## Cleanup Rules

- Keep `README.md`, `ARCHITECTURE.md`, and this file as the canonical docs.
- Keep `external/EXTERNAL_SOURCES.md` as submodule provenance, not architecture prose.
- Treat `runs/`, `.pytest_cache/`, `__pycache__/`, `*.pyc`, and native build
  outputs as generated.
- Do not delete historical runs without a separate result-curation decision.
- Use compact retention for normal MVP sweeps; use full retention only for
  debugging native bridge artifacts.
- Serious simulation backend reports must come from `normalized_records.jsonl`.
  Report regeneration may overwrite derived CSV/Markdown/plot files, but must
  not prune execution artifacts. Plot source CSVs and skipped-plot reasons are
  part of the report contract.
- Do not rename stable route IDs during documentation cleanup.
- Do not let synthetic pressure workloads enter normal benchmark execution.
