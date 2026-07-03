# Thesis Evidence Snapshot - Wave 2E.45C

Snapshot date: 2026-07-03
Git commit: `bee4b42`
Snapshot scope: fresh canonical CPU, GPU, and UPMEM evidence runs listed below.
Older ignored runs under `runs/` may reflect earlier artifact behavior and are
not part of this tracked snapshot.

This note records local generated evidence. The generated run directories are
intentionally ignored by git; this Markdown file is the tracked pointer and
interpretation record.

## Commands Run

From `thesis/implementation`:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src ../.venv/bin/python scripts/evidence_shortcuts.py check-gpu runs/evidence/gpu_evidence/simulation_backend_compare/20260703_193655
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src ../.venv/bin/python -m quantum_bench.bench report-run --input runs/evidence/gpu_evidence/simulation_backend_compare/20260703_193655 --out runs/comparisons/gpu_evidence/report_run/20260703_193655_gpu_verified
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src ../.venv/bin/python -m quantum_bench.bench compare-results --inputs runs/evidence/cpu_evidence/simulation_backend_compare/20260703_151903 runs/evidence/gpu_evidence/simulation_backend_compare/20260703_193655 --comparison-type cpu_vs_gpu --out runs/comparisons/gpu_evidence/cpu_vs_gpu/20260703_193655
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src ../.venv/bin/python -m quantum_bench.bench compare-results --inputs runs/evidence/cpu_evidence/simulation_backend_compare/20260703_151903 runs/evidence/gpu_evidence/simulation_backend_compare/20260703_193655 runs/evidence/upmem_sim_evidence/simulation_backend_compare/20260703_151910 --comparison-type thesis_benchmark_cpu_gpu_upmem --out runs/comparisons/thesis_benchmark/cpu_gpu_upmem/20260703_193655
```

## Snapshot Runs

| Artifact | Path | Status | Notes |
|---|---|---|---|
| CPU evidence | `runs/evidence/cpu_evidence/simulation_backend_compare/20260703_151903` | completed | QuEST CPU + Quimb TN only. |
| GPU evidence | `runs/evidence/gpu_evidence/simulation_backend_compare/20260703_193655` | completed with verified GPU rows | QuEST CPU anchor + QuEST HIP GPU rows. |
| UPMEM simulator evidence | `runs/evidence/upmem_sim_evidence/simulation_backend_compare/20260703_151910` | completed | Includes strict SDK simulator UPMEM rows. |
| GPU verification artifact | `build/gpu_verification/quest_gpu_full_state_exact.json` | verified | QuEST HIP on AMD Radeon RX 6600 / gfx1032. |
| GPU report | `runs/comparisons/gpu_evidence/report_run/20260703_193655_gpu_verified` | completed | Derived report from GPU evidence. |
| CPU vs GPU comparison | `runs/comparisons/gpu_evidence/cpu_vs_gpu/20260703_193655` | completed | Derived comparison artifact. |
| CPU vs GPU vs UPMEM comparison | `runs/comparisons/thesis_benchmark/cpu_gpu_upmem/20260703_193655` | completed | Derived comparison artifact. |
| Generic feasibility sweep | `runs/20260703_151947_upmem_generic_sweep_upmem_generic_feasibility` | completed | Scanner-only boundary artifact. |
| Generic strict UPMEM sweep | `runs/evidence/upmem_generic_sweep/upmem_generic_both_modes/20260703_151951` | completed with unsupported boundary rows | Runs generic-only, float32 and int8 modes. |
| Generic sweep report | `runs/comparisons/upmem_generic_sweep/report_run/20260703_152032` | completed | Derived report from generic sweep. |
| Generic sweep comparison | `runs/comparisons/upmem_generic_sweep/latest_single_run/20260703_152032` | completed | Derived comparison from generic sweep. |

## Evidence Checks

| Check | Result |
|---|---|
| CPU rows are serious baselines only | passed: `quest_cpu_full_state_exact`, `quimb_tn_exact`. |
| CPU validation status | passed for 36 normalized records. |
| GPU verified rows | passed: 3 `quest_gpu_full_state_exact` rows. |
| GPU execution target | passed: `contraction_execution_target=gpu`. |
| GPU verification flags | passed: `gpu_backend_verified=true`, `gpu_program_executed=true`. |
| GPU device | passed: `AMD Radeon RX 6600 (gfx1032)`. |
| GPU validation status | passed for all GPU rows. |
| GPU CPU fallback | passed: no GPU row has CPU fallback enabled. |
| UPMEM simulator route present | passed: `upmem_tn_sdk_simulator_quantized`. |
| UPMEM execution target/mode | passed: `contraction_execution_target=upmem`, `upmem_execution_mode=sdk_simulator`. |
| UPMEM SDK backend semantics | passed: `execution_backend=upmem_sdk`, `upmem_program_executed=true`. |
| UPMEM CPU fallback | passed: `cpu_fallback_used=false` for UPMEM rows. |
| UPMEM hardware claims | passed: `hardware_speedup_applicable=false`, `hardware_timing_available=false`. |
| Evidence figures | passed for listed fresh evidence: no `.png`, `.svg`, or `.pdf` files. |
| Evidence tables/reports boundary | passed for listed fresh evidence: known derived files are absent. |
| Derived reports/plots | passed: report and comparison artifacts are under `runs/comparisons`. |

Known derived files checked as absent from fresh evidence:

- `comparison_summary.md`
- `simulation_backend_compare_results.csv`
- `simulation_backend_compare_pairs.csv`
- `upmem_mvp_benchmark_results.csv`
- `kernel_family_summary.csv`
- `quantization_accuracy_summary.csv`
- `unsupported_reasons.csv`
- `quantization_comparison.csv`

## GPU Verification Evidence

The GPU verification artifact proves a real QuEST HIP execution path:

- `/dev/kfd`: visible.
- `/dev/dri`: visible.
- render node: `/dev/dri/renderD128` visible.
- `rocminfo`: found `gfx1032`.
- HIP smoke: built and ran.
- QuEST HIP runner: built.
- Minimal QuEST GPU run: succeeded.
- Device: `AMD Radeon RX 6600 (gfx1032)`.

The GPU evidence run contains 6 normalized records total:

- 3 `quest_cpu_full_state_exact` CPU anchor rows.
- 3 `quest_gpu_full_state_exact` GPU rows.

## Generic UPMEM Boundary

The generic sweep confirms bounded generic UPMEM coverage, not full general TN
coverage.

Strict generic UPMEM run:

- 33 normalized records total.
- 11 CPU reference records.
- 14 UPMEM records completed and validated.
- 8 UPMEM records unsupported with `generic_feasibility_rank_cap_exceeded`.
- Supported modes:
  - `quantization_mode=none`, `input_dtype_on_dpu=float32`, `accumulator_dtype_on_dpu=float32`.
  - `quantization_mode=per_task_input_quantize`, `input_dtype_on_dpu=int8`, `accumulator_dtype_on_dpu=int32`.

Dominant blocker: `generic_feasibility_rank_cap_exceeded`.

## Interpretation Limits

- QuEST CPU full-state and Quimb TN are the serious CPU baselines.
- QuEST HIP GPU is a verified full-state GPU baseline.
- The GPU evidence is not tensor-network execution and not PIM execution.
- UPMEM SDK simulator rows are real UPMEM SDK DPU-program simulator execution.
- SDK simulator timing is development evidence only; it is not hardware timing
  and must not be reported as hardware speedup.
- Bounded generic UPMEM contraction exists, but fully general UPMEM TN
  contraction does not yet exist.

## Verdict

Ready to use as the current thesis benchmark baseline for CPU, verified QuEST
HIP GPU, and UPMEM SDK simulator evidence.

The next UPMEM implementation target should still address the generic runtime
`rank_cap_exceeded` boundary. That is the first blocker preventing the bounded
generic UPMEM path from covering the next larger circuit-family cases in the
current sweep.
