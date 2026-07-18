# Research Benchmark Methodology

This document describes the benchmark methodology and claim boundaries.
Generated executions remain under ignored `runs/`; the selected compact result,
source CSVs, and plots are promoted to tracked `thesis_results/current/`.

## Implemented Methodology

Already in place:

- fixed suite files and canonical evidence shortcuts;
- normalized records as the source for report and comparison generation;
- warmups, repeats, median, quartiles/IQR, mean, min, max, and standard deviation fields;
- route role metadata for serious baselines and diagnostic routes;
- CPU/GPU performance-tier metadata with `state_output_mode=none`;
- UPMEM SDK simulator fields that distinguish SDK simulator mode from hardware;
- quantization attribution that preserves recorded probability errors and
  clipping/saturation counts when the underlying validation/runtime evidence
  provides them;
- unsupported/skipped rows and resource guard reasons;
- evidence/comparison artifact split;
- content hashes for circuit semantics, internal TN structure, and contraction
  plan identity;
- tracked compact snapshots with checksums and report regeneration;
- explicit parallelism claim boundaries for slicing, frontier, hybrid,
  full-state GPU, GPU TN feasibility, and UPMEM SDK simulator evidence.

Remaining limitations:

- long research runs are manual and hardware-dependent;
- energy is not reported unless real measured sensor metadata exists;
- UPMEM SDK simulator timing is code-path evidence, not hardware timing;
- UPMEM generic-boundary cases are limited by the current bounded generic runtime.
- GPU tensor-network support is feasibility-only until a real GPU TN candidate
  executes tensor-network work on the GPU with no CPU fallback.

## Research Suites

| Suite | Role |
|---|---|
| `configs/suites/manual/thesis_full_state_cpu_gpu.yml` | QuEST CPU vs verified QuEST GPU 8--20q performance tier. |
| `configs/suites/manual/thesis_full_state_correctness.yml` | Smaller full-statevector correctness tier for the same QuEST CPU/GPU routes. |
| `configs/suites/manual/thesis_cpu_tn_quimb.yml` | QuEST anchor, Quimb unsliced, and Quimb sliced 8--20q CPU TN evidence. |
| `configs/suites/manual/thesis_tn_paths_quantization.yml` | Same-path float64/int8 internal replay diagnostic. |
| `configs/suites/manual/thesis_planner_compare.yml` | Standard opt_einsum/cotengra baselines plus deterministic custom UPMEM-greedy modeled path comparison over the canonical grid. |
| `configs/suites/manual/thesis_planner_sensitivity.yml` | Modeled-only scenario sensitivity for named custom UPMEM objective weight profiles. |
| `configs/suites/manual/thesis_planner_semantic_v2.yml` | V2 projected-prefix modeled planner semantics on small quantum cases plus controlled synthetic motifs, with standard planning baselines. |
| `configs/suites/manual/thesis_planner_sensitivity_v2.yml` | V2 projected-prefix sensitivity across named profiles, retaining standard opt_einsum/cotengra baselines. |
| Historical `research_internal_parallelism` artifacts | Retained diagnostic schema and report compatibility; the runnable frontier/hybrid suite was retired in Phase A. |
| `configs/suites/manual/thesis_upmem_quantization_boundary.yml` | Strict generic-only UPMEM SDK simulator boundary and same-route float32/int8 attribution evidence. |

## Commands

```bash
BENCH_CPU_THREADS=<physical-core-count> make thesis-run
make thesis-promote
make thesis-verify
make thesis-report
```

`make research-plan` remains available to print each underlying command.

## Allowed Claims

- QuEST CPU/GPU speedup only for matched CPU/GPU full-state rows with the same
  circuit, qubit count, repeat, validation method, output mode, and timing
  scope.
- Performance-tier CPU/GPU speedup uses compute timing and is metrics-only; it
  is paired with a separate correctness tier.
- Quimb sliced and unsliced rows are CPU TN implementation evidence.
  `slicing_flop_ratio` means sliced cotengra reported FLOPs divided by unsliced
  cotengra reported FLOPs.
- `quimb_tn_sliced_exact` is executed slicing evidence, but it currently uses
  single-worker slice reconstruction unless a row explicitly records otherwise.
- Historical `cpu_tn_frontier_exact` and
  `cpu_tn_hybrid_sliced_frontier_exact` rows remain readable for old reports;
  their runnable diagnostic routes were retired in Phase A.
- `quest_gpu_full_state_exact` is full-state GPU evidence. It is not GPU
  tensor-network evidence.
- GPU tensor-network rows are not thesis evidence until a separate GPU TN route
  proves real tensor-network execution on a GPU with no CPU fallback.
- UPMEM SDK simulator rows prove strict SDK simulator code-path execution and
  current bounded generic behavior when `policy=generic-only`,
  `generic_only_all_tasks_used_generic_backend=true`, and
  `cpu_fallback_used=false`.
- The UPMEM research group runs `upmem-mvp-benchmark`, not the dense-capable
  route-comparison suite. It records `quantization_mode=none` and
  `per_task_input_quantize` separately for same-route attribution.
- A CPU/UPMEM row is labeled same-plan only when its
  `contraction_plan_hash` matches. Planner candidates are modeled evidence and
  remain separate from executor timing.
- The existing v1 `custom_upmem` suites are historical evidence under the
  recorded `upmem_path_cost_v1` and `generic_single_dpu_float32_v1` contract.
  Their costs are planner-estimated/modelled, not a hardware runtime
  predictor.
- The additive v2 suites use `upmem_path_cost_v2` with deterministic
  `projected_prefix` greedy selection. Zero-imaginary complex inputs are
  accepted as real-valued work; nonzero complex inputs are modeled as split
  real/imaginary components where the bounded policy supports them. The v2
  contract does not imply unrestricted complex execution.
- Both v1 and v2 planner suites retain standard opt_einsum/cotengra baselines.
  Planner rows are modeled path evidence only and never support a UPMEM
  hardware performance claim.
- Controlled chain/tree/star/cycle/grid/trade-off planner motifs are marked
  `not_real_quantum_circuit=true`; they validate planner behavior and never
  support circuit-runtime or hardware claims.

## Claims Not Allowed

- No hardware speedup claim from UPMEM SDK simulator timing.
- No GPU benchmark row unless `gpu_backend_verified=true` and
  `gpu_program_executed=true`.
- Cached GPU verification artifacts must be interpreted with their
  `gpu_verification_source` and current-process preflight fields. A cached
  artifact proves an earlier GPU run; it does not by itself prove that the
  current process can see GPU devices for a new benchmark.
- No GPU TN claim from QuEST full-state GPU rows or from feasibility metadata
  alone.
- No energy-efficiency claim without real measured energy metadata.
- No speedup across incompatible route families, such as Quimb versus internal
  TaskGraph diagnostics.
- Historical frontier or hybrid rows do not support a parallel speedup claim;
  current planning primitives remain modeled-only unless a future active route
  records execution evidence.
- No full-output exactness claim for `state_output_mode=none` rows.
- No physical bus-traffic claim from `application_visible_sdk_recorded`
  transfer bytes. When directional fields exist, reports require
  `actual_transfer_bytes = actual_h2d_bytes + actual_d2h_bytes`.
- The native generic-loop sidecar separately records prepared operand/result
  payload, control arguments, and modeled 8-byte payload alignment. Those are
  application-visible SDK-call lengths; unknown SDK and physical-DIMM traffic
  remain explicitly unavailable.

## Outputs

Research packs write derived artifacts under:

```text
runs/comparisons/research_pack/<timestamp>/
```

Expected files include:

- `benchmark_manifest.json`
- `per_case_route_stats.csv`
- `paired_speedups.csv`
- `cpu_gpu_performance_summary.csv`
- `upmem_quantization_attribution.csv`
- `same_plan_execution.csv`
- `planner_comparison.csv`
- `unsupported_cases.csv`
- `validation_summary.csv`
- `route_capability_matrix.csv`
- `plot_manifest.json`
- `benchmark_summary.md`

Every generated figure names one of these source CSVs in `plot_manifest.json`.
Figures with missing/zero-variance/unimplemented data remain as visible TODO
PNGs with `generated_todo_*` status and an exact reason; they are separate from
valid figures in the benchmark summary.
The summary ends with `Next UPMEM Implementation Readiness`, which lists the
current blockers and recommends one concrete next implementation target.
