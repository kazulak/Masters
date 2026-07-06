# Thesis Results Audit

Date: 2026-07-06

This audit is based on the current generated thesis evidence and the derived report:

- Full-state CPU/GPU evidence: `runs/evidence/thesis_full_state_cpu_gpu/simulation_backend_compare/20260706_161354`
- CPU TN path/quantization evidence: `runs/evidence/thesis_tn_paths_quantization/simulation_backend_compare/20260706_161628`
- UPMEM SDK simulator boundary evidence: `runs/evidence/thesis_upmem_quantization_boundary/upmem_generic_both_modes/20260706_162159`
- Derived report: `runs/comparisons/thesis/audit_20260706_current`

The report is derived from `1095` normalized records. It does not rerun benchmarks.

## Verdict

The benchmark system is now useful enough to expose real behavior, but the results are not yet thesis-final. The most important current finding is that several apparent performance surprises are caused by route semantics, not by quantum simulation theory.

The CPU quantized TN replay is slower than unquantized CPU replay because it is a diagnostic CPU route:

- `contraction_execution_target=cpu`
- `accelerator_kind=none`
- `input_dtype=int8_split_real_imag`
- `accumulator_dtype=complex128`
- `quantized_replay_numeric_contract=int8_operand_quantize_dequantize_then_complex128_einsum`

So the current quantized CPU replay measures per-contraction quantize/dequantize overhead plus normal CPU complex128 contraction. It is not a native int8 GEMM/TN kernel benchmark and not evidence that quantization would be slow on UPMEM.

Post-cleanup status:

- thesis report CSVs now include execution target, accelerator, dtype, and numeric-contract fields for CPU diagnostic replay rows.
- quantization runtime plots are labeled as CPU diagnostic TN replay, not native int8 or UPMEM performance.
- UPMEM report rows now include thesis-facing route labels, task counts, fallback counts, DPU dtype/byte fields where available, strict SDK simulator semantics, and flattened validation errors.
- existing route IDs are preserved for compatibility; thesis-facing labels are added in derived reports.

## Result Findings

### Full-State CPU/GPU

Evidence rows:

- routes: `quest_cpu_full_state_exact`, `quest_gpu_full_state_exact`
- rows: `420`
- repeats: `5`
- validation status: `passed_native_status`
- output mode: performance tier / metrics-only
- GPU device: `AMD Radeon RX 6600 (gfx1032)`

Compute-time GPU speedup is only clear at the largest tested sizes. Across all CPU/GPU rows:

- median compute speedup CPU/GPU: `0.95x`
- compute-speedup rows where GPU is faster: `12 / 42`
- wall-time rows where GPU is faster: `4 / 42`
- 20q compute speedup range: about `2.63x` to `3.55x`

Interpretation:

- The GPU route works and is verified.
- Current 8q-18q rows are mostly overhead-dominated.
- 20q rows begin to show real GPU compute benefit.
- These are QuEST full-state GPU results, not GPU tensor-network results.
- Because the performance tier uses `state_output_mode=none`, these rows are runtime evidence, not full-statevector exact-output evidence.

### CPU Tensor Network Routes

Evidence rows:

- routes: `quest_cpu_full_state_exact`, `quimb_tn_exact`, `quimb_tn_sliced_exact`, `cpu_tn_path_replay_float64`, `cpu_tn_path_replay_int8_quantized`
- rows: `630`
- cases: `42`
- validation status: all `passed`
- max TaskGraph intermediate after the fix: `16 MiB`

Median route runtime over the per-circuit-size summaries:

- `quimb_tn_exact`: `0.00063 s`
- `quimb_tn_sliced_exact`: `0.00176 s`
- `cpu_tn_path_replay_float64`: `0.00080 s`
- `cpu_tn_path_replay_int8_quantized`: `0.01149 s`

Interpretation:

- `quimb_tn_exact` is the serious CPU TN baseline.
- `quimb_tn_sliced_exact` is slicing evidence, but slices are not parallelized, so it should not be presented as a speedup route.
- `cpu_tn_path_replay_*` routes are diagnostic attribution routes, not serious TN baselines.
- The previous OOM-like TN behavior was an implementation bug in TaskGraph replay semantics, not a valid TN limitation.

### CPU Quantized Replay

From `tn_quantization_speedup_by_circuit_size.csv`:

- matched quantization rows: `42`
- median `unquantized / quantized` compute ratio: `0.0669`
- median slowdown of quantized replay: about `15x`
- family median slowdown range: about `7.7x` to `19.8x`

Interpretation:

- This slowdown is expected for CPU diagnostic replay.
- The route quantizes operands, dequantizes them, and then uses CPU complex128 einsum.
- It is useful for attribution and accuracy plumbing, but not for CPU speed claims.
- The plot title/legend should explicitly say "CPU diagnostic replay" so it cannot be mistaken for a UPMEM or native int8 result.

Current accuracy reporting note:

- `tn_quantization_error_by_circuit_size.csv` reports near-zero validation error versus reference for many rows.
- The route-level `quantization_max_abs_error` and `quantization_l2_error` are often blank or zero because the current diagnostic contract dequantizes before complex128 contraction.
- The report now labels this as CPU diagnostic replay accuracy, not native hardware quantized accuracy.

### UPMEM SDK Simulator

Evidence rows:

- UPMEM rows: `30`
- CPU reference rows: `15`
- UPMEM completed/passed rows: `28`
- UPMEM unsupported rows: `2`
- unsupported case: `qrng_7q_thesis_upmem_boundary`, both `none` and `per_task_input_quantize`
- unsupported reason: `generic_feasibility_rank_cap_exceeded`

UPMEM row semantics:

- `contraction_execution_target=upmem`
- `upmem_execution_mode=sdk_simulator`
- `execution_backend=upmem_sdk`
- `hardware_execution=false`
- `hardware_timing_available=false`
- `hardware_speedup_applicable=false`
- `cpu_fallback_used=false`
- `upmem_program_executed=true` for supported rows

Interpretation:

- The UPMEM SDK simulator route is real SDK/DPU simulator code-path evidence.
- It is not hardware speedup evidence.
- Current UPMEM cases are still small boundary cases.
- The first visible boundary is a generic-kernel rank cap, not performance.

Current UPMEM reporting status:

- route id is still emitted as `upmem_tn_runtime` for compatibility, while derived thesis reports add labels for float32/no-quantization and int8 per-task quantized SDK simulator rows.
- derived reports fill `policy`, `task_count`, `upmem_task_count`, `cpu_fallback_task_count`, `backend_family`, and `accelerator_kind` where possible.
- nested validation errors are flattened into UPMEM report CSVs.
- `total_simulator_time_s` is reported from simulator compute timing when a dedicated simulator-total field is absent.

## Implementation State

Already in good shape:

- evidence and comparison outputs are separated.
- canonical thesis evidence runs exist for full-state CPU/GPU, CPU TN, and UPMEM SDK simulator.
- QuEST GPU execution is verified on AMD ROCm.
- Quimb is the serious CPU TN baseline.
- internal path replay can compare float64/complex128 replay against int8 operand quantization.
- TaskGraph replay now follows opt_einsum operand-list semantics.
- generated report includes CSVs and plots from normalized records.

Not thesis-final yet:

- generated evidence can still be large because final tensors/state dumps are retained for many repeats.
- GPU full-state benchmark is still small and overhead-dominated except at 20q.
- serious UPMEM evidence is still boundary/code-path evidence, not hardware timing.

## Before More UPMEM Architecture Work

Do these first.

1. Make artifact retention practical
   - For repeated thesis runs, avoid retaining every full statevector/final tensor unless explicitly requested.
   - Keep checksums/shapes/dtypes and validation records by default.

2. Strengthen benchmark methodology
   - Keep full-state CPU/GPU as a separate route family.
   - Add deeper or more compute-heavy full-state workloads if GPU speedup is a thesis claim.
   - Add TN path variants only where the route semantics are explicit.
   - Keep diagnostic internal routes out of serious baseline plots unless clearly marked.

3. Define the next UPMEM implementation target from evidence
   - Current blocker: bounded generic kernel rank/shape caps.
   - Current missing architecture: tiling/slicing, layout/transpose handling, multi-DPU scheduling, and hardware timing.
   - Recommended next target: implement the smallest UPMEM generic-kernel extension that moves the boundary case (`qrng_7q_thesis_upmem_boundary`) from unsupported to supported without CPU contraction fallback.

## Recommended Next Waves

1. Thesis report semantics cleanup.
   Fix labels, route metadata, flattened error fields, and plot captions. No kernels.

2. Evidence retention cleanup for thesis runs.
   Reduce retained repeated tensors while preserving checksums and validation records.

3. UPMEM boundary-target implementation.
   Use the current boundary evidence to implement the next minimal generic-kernel capability, then rerun only the UPMEM boundary suite.

4. Larger UPMEM architecture work.
   Start only after the reporting layer can clearly show current UPMEM boundary, next supported case, strict code path, and no hardware-speedup overclaim.
