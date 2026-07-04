# Research Benchmark Methodology

This document describes the current thesis-grade benchmark workflow. Generated
evidence and comparisons remain under ignored `runs/` directories; this file
records the methodology and claim boundaries.

## Stage A Audit Summary

Already in place:

- fixed suite files and canonical evidence shortcuts;
- normalized records as the source for report and comparison generation;
- warmups, repeats, median, mean, min, max, and standard deviation fields;
- route role metadata for serious baselines and diagnostic routes;
- CPU/GPU performance-tier metadata with `state_output_mode=none`;
- UPMEM SDK simulator fields that distinguish SDK simulator mode from hardware;
- unsupported/skipped rows and resource guard reasons;
- evidence/comparison artifact split.

Added in Wave 2E.58:

- manual research suites for CPU/GPU, CPU TN, internal diagnostics, and UPMEM
  boundary evidence;
- `scripts/research_benchmark_pack.py` for reproducible pack generation;
- Make targets for plan, lightweight pack creation, full opt-in research runs,
  and report regeneration;
- derived research CSVs, plots, manifest, and summary under
  `runs/comparisons/research_pack/...`.

Remaining limitations:

- long research runs are manual and hardware-dependent;
- energy is not reported unless real measured sensor metadata exists;
- UPMEM SDK simulator timing is code-path evidence, not hardware timing;
- UPMEM boundary cases are limited by the current bounded generic/dense runtime.

## Research Suites

| Suite | Role |
|---|---|
| `configs/suites/manual/research_cpu_gpu.yml` | QuEST CPU vs verified QuEST GPU performance tier. |
| `configs/suites/manual/research_cpu_gpu_correctness.yml` | Smaller full-statevector correctness tier for the same QuEST CPU/GPU routes. |
| `configs/suites/manual/research_cpu_tn.yml` | QuEST anchor, Quimb unsliced, and Quimb sliced CPU TN evidence. |
| `configs/suites/manual/research_internal_parallelism.yml` | Diagnostic internal TaskGraph sequential/frontier/hybrid evidence. |
| `configs/suites/manual/research_upmem_boundary.yml` | Strict UPMEM SDK simulator supported/unsupported boundary evidence. |

## Commands

```bash
make research-plan
make research-benchmarks
RUN_RESEARCH=1 make research-benchmarks
make research-report
```

`make research-benchmarks` is lightweight by default. Full benchmark execution
requires `RUN_RESEARCH=1` so long GPU, TN, and UPMEM simulator runs are never
started accidentally.

## Allowed Claims

- QuEST CPU/GPU speedup only for matched CPU/GPU full-state rows with the same
  circuit, qubit count, repeat, validation method, output mode, and timing
  scope.
- Performance-tier CPU/GPU speedup uses compute timing and is metrics-only; it
  is paired with a separate correctness tier.
- Quimb sliced and unsliced rows are CPU TN implementation evidence.
  `slicing_flop_ratio` means sliced cotengra reported FLOPs divided by unsliced
  cotengra reported FLOPs.
- UPMEM SDK simulator rows prove strict SDK simulator code-path execution and
  current boundary behavior when `cpu_fallback_used=false`.

## Claims Not Allowed

- No hardware speedup claim from UPMEM SDK simulator timing.
- No GPU benchmark row unless `gpu_backend_verified=true` and
  `gpu_program_executed=true`.
- No energy-efficiency claim without real measured energy metadata.
- No speedup across incompatible route families, such as Quimb versus internal
  TaskGraph diagnostics.
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
- `unsupported_cases.csv`
- `validation_summary.csv`
- `route_capability_matrix.csv`
- `plot_manifest.json`
- `benchmark_summary.md`

The summary ends with `Next UPMEM Implementation Readiness`, which lists the
current UPMEM blockers visible in the loaded evidence and recommends one next
UPMEM implementation target.
