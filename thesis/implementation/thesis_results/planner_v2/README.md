# Research Benchmark Summary

This is a derived research pack generated from normalized benchmark records. Evidence inputs remain read-only.

## Benchmark Matrix

| Route | Role | Target | Records | Unsupported |
| --- | --- | --- | ---: | ---: |
| planner_candidate_model | contraction_path_candidate | modeled | 150 | 0 |

## Suite Paths

- `cpu_gpu_correctness`: `configs/suites/manual/thesis_full_state_correctness.yml`
- `cpu_gpu`: `configs/suites/manual/thesis_full_state_cpu_gpu.yml`
- `cpu_tn`: `configs/suites/manual/thesis_cpu_tn_quimb.yml`
- `tn_path_quantization`: `configs/suites/manual/thesis_tn_paths_quantization.yml`
- `planner_paths`: `configs/suites/manual/thesis_planner_semantic_v2.yml`
- `planner_sensitivity`: `configs/suites/manual/thesis_planner_sensitivity_v2.yml`
- `upmem_boundary`: `configs/suites/manual/thesis_upmem_quantization_boundary.yml`
- `upmem_quantization_stress`: `configs/suites/manual/thesis_upmem_quantization_stress.yml`
- `internal_parallelism`: `configs/suites/manual/research_internal_parallelism.yml`

## Exact Commands

Run `make research-plan` to print the underlying commands.
Run `BENCH_CPU_THREADS=<physical-core-count> make thesis-run` for the complete local benchmark matrix.

## Hardware And Software Manifest

- Git commit: `a453331d7bc240f64c3b6391f4ea2ac916de798c`
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

- `planner_candidate_model` `not_applicable`: 150 records

## Plots

### Completed Scientific Figures

- `planner_flops_vs_upmem_pressure.png`: Planner-estimated FLOPs versus normalized modeled PIM objective for feasible candidates.
- `planner_component_scores.png`: Modeled PIM objective components for feasible planner candidates.
- `planner_selection.png`: Selected feasible planner candidate per modeled PIM objective profile.
- `planner_pareto_frontier.png`: Feasible planner candidates colored by modeled Pareto status.
- `planner_sensitivity.png`: Selected planner candidates across modeled PIM weight profiles.
- `planner_component_diagnostics.png`: Versioned planner component diagnostics where v2 records provide numeric execution decomposition fields.

### TODO Figures

- `cpu_gpu_runtime_by_qubits.png`: generated_todo_missing_data (source fields contain no numeric data: benchmark_n_qubits, cpu_simulation_compute_time_s_median, gpu_simulation_compute_time_s_median). Measured QuEST CPU and verified QuEST GPU compute time by circuit size.
- `cpu_gpu_speedup_by_qubits.png`: generated_todo_missing_data (source fields contain no numeric data: benchmark_n_qubits, compute_speedup_cpu_over_gpu_median). Measured CPU/GPU compute ratio; values above 1 mean GPU compute was faster.
- `cpu_gpu_energy_efficiency_by_qubits.png`: generated_todo_not_implemented (energy measurements are unavailable in the research evidence contract). TODO: energy metadata is unavailable or not a validated paired measurement.
- `cpu_tn_runtime_by_qubits.png`: generated_todo_missing_data (source fields contain no numeric data: benchmark_n_qubits, route_id, simulation_compute_time_s_median). Measured Quimb unsliced and sliced tensor-network compute time by circuit size.
- `full_state_vs_tn_runtime_by_qubits.png`: generated_todo_missing_data (source fields contain no numeric data: benchmark_n_qubits, route_id, simulation_compute_time_s_median). Cross-algorithm/backend measured compute time on the same shallow circuits.
- `tn_planning_vs_contraction.png`: generated_todo_missing_data (no_quimb_timing_rows). Measured Quimb planning and contraction compute time; phases are not combined.
- `tn_path_flops_by_family_size.png`: generated_todo_missing_data (no_tn_estimated_flops_rows). Planner-estimated contraction FLOPs by circuit family and size.
- `tn_path_peak_memory_by_family_size.png`: generated_todo_missing_data (no_tn_max_intermediate_bytes_rows). Planner-estimated largest intermediate tensor bytes by circuit family and size.
- `cpu_tn_slicing_flop_ratio.png`: generated_todo_missing_data (source fields contain no numeric data: benchmark_n_qubits, slicing_flop_ratio, slicing_flop_change_kind). Slicing FLOP ratio = sliced cotengra plan reported FLOPs / unsliced cotengra plan reported FLOPs.
- `cpu_tn_slicing_tradeoff.png`: generated_todo_missing_data (source fields contain no numeric data: benchmark_n_qubits, runtime_ratio_sliced_over_unsliced, slicing_flop_ratio, largest_intermediate_ratio_sliced_over_unsliced). Matched Quimb slicing trade-off ratios: runtime, planner-estimated FLOPs, and largest intermediate size (sliced / unsliced).
- `upmem_supported_boundary.png`: generated_todo_no_variance (source data has zero variance across the available records). Supported versus unsupported strict generic-only UPMEM SDK simulator rows.
- `upmem_accuracy_error.png`: generated_todo_missing_data (source fields contain no numeric data: benchmark_n_qubits, max_abs_error, validation_method, upmem_execution_mode). Strict generic UPMEM SDK simulator maximum absolute error where validation data exists.
- `upmem_quantization_attribution.png`: generated_todo_missing_data (source fields contain no numeric data: benchmark_n_qubits, route_runtime_ratio_none_over_quantized, transfer_ratio_none_over_quantized). Same-route float32 versus int8 attribution for strict generic UPMEM SDK simulator execution.
- `quantization_runtime_by_executor.png`: generated_todo_missing_data (source fields contain no numeric data: benchmark_n_qubits, route_runtime_ratio_none_over_quantized, unquantized_host_residual_time_s, quantized_host_residual_time_s). Same-plan SDK simulator float32/int8 software-recorded host/control residual-time ratio.
- `quantization_transfer_bytes.png`: generated_todo_missing_data (source fields contain no numeric data: benchmark_n_qubits, unquantized_h2d_bytes, quantized_h2d_bytes, unquantized_d2h_bytes, quantized_d2h_bytes, transfer_ratio_none_over_quantized). Absolute application-visible H2D/D2H bytes plus the same-plan float32/int8 total-byte ratio.
- `quantization_error_by_family_size.png`: generated_todo_missing_data (source fields contain no numeric data: benchmark_n_qubits, quantized_max_abs_error_vs_full_precision, quantized_execution_max_abs_error). UPMEM SDK simulator int8 maximum absolute error against the full-precision reference.
- `quantization_probability_error_by_family_size.png`: generated_todo_missing_data (source fields contain no numeric data: benchmark_n_qubits, quantized_probability_max_abs_error, quantized_probability_l1_error). Matched UPMEM SDK-simulator quantized probability error against the recorded validation reference.
- `same_plan_cpu_upmem_runtime.png`: generated_todo_missing_data (source fields contain no numeric data: benchmark_n_qubits, contraction_plan_hash, cpu_time_s, upmem_simulator_time_s). Same-plan CPU and UPMEM SDK simulator execution timing.
- `internal_parallelism_metadata_by_qubits.png`: generated_todo_missing_data (source fields contain no numeric data: benchmark_n_qubits, parallelism_mode, parallelism_evidence_type, frontier_worker_count, frontier_wave_count, max_frontier_width, frontier_executed_parallel_task_count). Diagnostic internal TaskGraph frontier metadata, not serious baseline performance.

### Failed Figures


## Key Findings

- Normalized records loaded: 150.
- Per-case route statistic rows: 22.
- Valid CPU/GPU paired speedup rows: 0.
- Matched strict generic UPMEM float32/int8 attribution rows: 0.
- Modeled contraction-path candidate rows: 150.
- Unsupported/skipped rows preserved: 0.

## Observed Result Snapshot

- No matched verified CPU/GPU performance repeats were available.
- Planner evidence covers `22` cases and `10` candidates with plan hashes, FLOP/peak-memory estimates, and modeled UPMEM pressure.

## Unsupported Cases

- None in loaded records.

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

- No UPMEM SDK simulator records were loaded in this pack.
- Next target: run the selected strict generic-only UPMEM suite and regenerate this pack before making stronger UPMEM claims.

## Missing Evidence

- QuEST CPU full-state baseline records are absent.
- Verified QuEST GPU records are absent.
- Quimb TN baseline records are absent.
- UPMEM SDK simulator quantized records are absent.
