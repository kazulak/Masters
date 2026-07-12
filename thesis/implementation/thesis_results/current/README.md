# Research Benchmark Summary

This is a derived research pack generated from normalized benchmark records. Evidence inputs remain read-only.

## Benchmark Matrix

| Route | Role | Target | Records | Unsupported |
| --- | --- | --- | ---: | ---: |
| cpu_tn_einsum_exact | internal_debug_baseline | cpu | 27 | 0 |
| cpu_tn_frontier_exact | internal_frontier_diagnostic | cpu | 6 | 0 |
| cpu_tn_hybrid_sliced_frontier_exact | internal_hybrid_diagnostic | cpu | 6 | 0 |
| cpu_tn_path_replay_float64 | diagnostic_path_replay_baseline | cpu | 42 | 0 |
| cpu_tn_path_replay_int8_quantized | diagnostic_quantized_path_replay | cpu | 42 | 0 |
| planner_candidate_model | contraction_path_candidate | modeled | 84 | 0 |
| quest_cpu_full_state_exact | serious_full_state_baseline | cpu | 408 | 0 |
| quest_gpu_full_state_exact | serious_gpu_full_state_baseline | gpu | 234 | 0 |
| quimb_tn_exact | serious_external_tn_baseline | cpu | 126 | 0 |
| quimb_tn_sliced_exact | explicit_slicing_evidence | cpu | 126 | 0 |
| upmem_tn_runtime | strict_upmem_sdk_simulator_generic | upmem | 42 | 2 |

## Suite Paths

- `cpu_gpu_correctness`: `configs/suites/manual/thesis_full_state_correctness.yml`
- `cpu_gpu`: `configs/suites/manual/thesis_full_state_cpu_gpu.yml`
- `cpu_tn`: `configs/suites/manual/thesis_cpu_tn_quimb.yml`
- `tn_path_quantization`: `configs/suites/manual/thesis_tn_paths_quantization.yml`
- `planner_paths`: `configs/suites/manual/thesis_planner_compare.yml`
- `upmem_boundary`: `configs/suites/manual/thesis_upmem_quantization_boundary.yml`
- `upmem_quantization_stress`: `configs/suites/manual/thesis_upmem_quantization_stress.yml`
- `internal_parallelism`: `configs/suites/manual/research_internal_parallelism.yml`

## Exact Commands

Run `make research-plan` to print the underlying commands.
Run `BENCH_CPU_THREADS=<physical-core-count> make thesis-run` for the complete local benchmark matrix.

## Hardware And Software Manifest

- Git commit: `29fcd17b18521d08c3c86c1c092eb6eca67bb48f`
- Dirty worktree: `False`
- Host: `kazulak`
- Python: `3.10.12 (main, Jun 22 2026, 18:55:27) [GCC 11.4.0]`
- Packages: `{"cotengra": "0.7.5", "matplotlib": "3.10.8", "numpy": "2.2.6", "opt_einsum": "3.4.0", "quimb": "1.11.2"}`

## Benchmark Group Status

- `report existing evidence`: returncode `0`.

## Repeats, Warmups, And Timing

Statistics are computed from normalized records. Median and spread fields are reported per case/route.
CPU/GPU performance-tier speedup uses `simulation_compute_time_s`; UPMEM route timing prefers `total_host_residual_time_s` when present and retains execution validation separately.
Qubit-scaling tables and plots use `benchmark_n_qubits` / `actual_n_qubits`; suite caps and output caps are not used as circuit size.

## Validation Methods

- `cpu_tn_einsum_exact` `passed`: 6 records
- `cpu_tn_einsum_exact` `reference`: 21 records
- `cpu_tn_frontier_exact` `passed`: 6 records
- `cpu_tn_hybrid_sliced_frontier_exact` `passed`: 6 records
- `cpu_tn_path_replay_float64` `passed`: 42 records
- `cpu_tn_path_replay_int8_quantized` `passed`: 42 records
- `planner_candidate_model` `not_applicable`: 84 records
- `quest_cpu_full_state_exact` `passed`: 198 records
- `quest_cpu_full_state_exact` `passed_native_status`: 210 records
- `quest_gpu_full_state_exact` `passed`: 24 records
- `quest_gpu_full_state_exact` `passed_native_status`: 210 records
- `quimb_tn_exact` `passed`: 126 records
- `quimb_tn_sliced_exact` `passed`: 126 records
- `upmem_tn_runtime` `passed`: 40 records
- `upmem_tn_runtime` `skipped`: 2 records

## Plots

- `cpu_gpu_runtime_by_qubits.png`: generated (ok). Performance-tier median QuEST CPU and verified QuEST GPU compute time by circuit size.
- `cpu_gpu_speedup_by_qubits.png`: generated (ok). Performance-tier CPU/GPU compute speedup; values above 1 mean GPU faster.
- `cpu_tn_runtime_by_qubits.png`: generated (ok). Quimb CPU tensor-network timing; sliced and unsliced routes are separate execution modes.
- `full_state_vs_tn_runtime_by_qubits.png`: generated (ok). QuEST CPU full-state and Quimb CPU TN timing on the same shallow circuits; this is an algorithm/backend comparison, not same-plan speedup.
- `tn_planning_vs_contraction.png`: generated (ok). Planning and contraction timing are reported separately for external CPU TN routes.
- `tn_path_flops_by_family_size.png`: generated (ok). Reported contraction-plan FLOP estimates by circuit family and size.
- `tn_path_peak_memory_by_family_size.png`: generated (ok). Reported peak intermediate tensor bytes by circuit family and size.
- `cpu_tn_slicing_flop_ratio.png`: generated (ok). slicing_flop_ratio = sliced cotengra reported FLOPs / unsliced cotengra reported FLOPs.
- `upmem_supported_boundary.png`: generated (ok). Supported versus unsupported strict generic-only UPMEM SDK simulator rows.
- `upmem_accuracy_error.png`: generated (ok). Strict generic UPMEM SDK simulator max absolute error where validation data exists.
- `upmem_quantization_attribution.png`: generated (ok). Same-route float32 versus int8 ratios for strict generic UPMEM SDK simulator execution; this is not hardware speedup.
- `quantization_runtime_by_executor.png`: generated (ok). Same-plan SDK simulator float32/int8 host-side residual-time ratio; this is not hardware speedup.
- `quantization_transfer_bytes.png`: generated (ok). Same-plan float32/int8 host-DPU transfer-volume ratio.
- `quantization_error_by_family_size.png`: generated (ok). Int8 maximum absolute error against the full-precision TaskGraph reference.
- `same_plan_cpu_upmem_runtime.png`: generated (ok). CPU replay and UPMEM SDK simulator rows share an identical contraction-plan hash; timing is not hardware speedup.
- `planner_flops_vs_upmem_pressure.png`: generated (ok). Planner FLOP estimates versus modeled UPMEM pressure when multiple plan candidates are available.
- `internal_parallelism_metadata_by_qubits.png`: generated (ok). Diagnostic internal TaskGraph frontier metadata, not serious baseline performance.

## Key Findings

- Normalized records loaded: 1143.
- Per-case route statistic rows: 513.
- Valid CPU/GPU paired speedup rows: 234.
- Matched strict generic UPMEM float32/int8 attribution rows: 20.
- Modeled contraction-path candidate rows: 84.
- Unsupported/skipped rows preserved: 2.

## Observed Result Snapshot

- Verified QuEST GPU compute ratio (CPU/GPU): median `0.283x`, range `0.164x` to `5.4x`; GPU was faster in `59/210` matched repeats.
- Shallow exact CPU comparison (Quimb TN time / QuEST full-state time): median `2.53x` across `42` cases. This is an algorithm/backend runtime ratio, not same-plan speedup.
- Executed Quimb slicing time / unsliced Quimb time: median `2.07x`, range `1.28x` to `43.5x`; slice reconstruction used one worker.
- Strict generic UPMEM SDK-simulator float32/int8 attribution: median host-residual-time ratio `0.999x`, median transfer ratio `1.01x`, maximum observed int8 absolute error `0.00642`. These are simulator-route measurements, not hardware speedup.
- Planner evidence covers `42` cases and `2` candidates with plan hashes, FLOP/peak-memory estimates, and modeled UPMEM pressure.
- Explicit boundary rows: `generic_feasibility_rank_cap_exceeded` = 2.

## Unsupported Cases

- `qrng_17q_thesis_upmem_feasibility_boundary` / `upmem_tn_runtime`: generic_feasibility_rank_cap_exceeded
- `qrng_17q_thesis_upmem_feasibility_boundary` / `upmem_tn_runtime`: generic_feasibility_rank_cap_exceeded

## Claims Allowed

- QuEST CPU/GPU full-state speedup only for matched CPU/GPU rows with the same timing scope.
- Quimb unsliced vs sliced comparisons as CPU TN implementation evidence, with slicing metrics labeled as `slicing_flop_ratio`.
- Strict generic-only UPMEM SDK simulator rows as bounded generic code-path and boundary evidence.
- Same-route float32 versus int8 generic UPMEM ratios as SDK-simulator route attribution, not hardware speedup.

## Claims Not Allowed

- No hardware speedup claim from UPMEM SDK simulator timing.
- No fake GPU rows without verified GPU execution.
- No energy-efficiency claim without real measured energy metadata.
- No speedup across incompatible route families such as Quimb versus internal TaskGraph diagnostics.

## Artifact Boundary Checks

- Evidence boundary status: `passed`.
- Guard issues: 0.

## Next UPMEM Implementation Readiness

- UPMEM SDK simulator records loaded: 42; SDK simulator rows: 42.
- Strict generic-only UPMEM rows: 42.
- CPU fallback flagged in UPMEM rows: 0.
- Unsupported/boundary rows: 2.
- Top blocker reasons: generic_feasibility_rank_cap_exceeded=2.
- Tiling support derived from records: 40/40 rows report executable or observed tiling metadata.
- Highest supported UPMEM case in these records: `qrng_13q_thesis_upmem_supported` at `13` qubits.
- First unsupported case by recorded qubit count: `qrng_17q_thesis_upmem_feasibility_boundary` at `17` qubits; reason: `generic_feasibility_rank_cap_exceeded`.
- Next target derived from the boundary: investigate `qrng_17q_thesis_upmem_feasibility_boundary` and address `generic_feasibility_rank_cap_exceeded` without CPU fallback.

## Missing Evidence

- No mandatory evidence class is obviously absent from loaded records.
