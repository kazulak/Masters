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
| `configs/suites/manual/thesis_planner_compare.yml` | Greedy/auto path costs and modeled UPMEM pressure over the canonical grid. |
| `configs/suites/manual/research_internal_parallelism.yml` | Diagnostic internal TaskGraph sequential/frontier/hybrid evidence. |
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
- `cpu_tn_frontier_exact` and `cpu_tn_hybrid_sliced_frontier_exact` are
  diagnostic internal TaskGraph routes. They support architecture evidence, not
  serious TN baseline claims.
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
- No parallel speedup claim from diagnostic frontier or hybrid rows without a
  separate matched performance/scaling methodology.
- No full-output exactness claim for `state_output_mode=none` rows.

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
The summary ends with `Next UPMEM Implementation Readiness`, which lists the
current blockers and recommends one concrete next implementation target.
