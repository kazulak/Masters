# Hierarchical UPMEM Parallel Diagnostic v1

This document freezes the physically validated diagnostic at
`thesis-upmem-hierarchical-parallel-diagnostic-v1`. It is a descriptive
diagnostic, not a `physical_performance_v1` campaign.

## Scope

The experiment fixes the Stress18 (`quantization_stress`, 18 qubits, two
repeat layers) circuit, its target-neutral `opt_einsum` greedy plan, one
logical `ContractionDAG`, split-complex float32, host-roundtrip intermediates,
and `steady_execution_v1`. It varies tasklets on one DPU and DPUs at T8 on
one rank. The run used `diagnostic_v1`, one warmup block, five measured blocks
per route, CPU affinity `[0]`, `/dev/dpu_rank1`, and the `powersave` governor.

The execution source is
`7e3ca432a3b109da15710b32dcb1edec9e4771fb`. The recovered experiment
`d9cdd98ed6aeb44a1812f743e7a953abcfe2f7d43d46054f51bc70b30d7592f6` and run
`608a1f8f-de5d-44c9-9076-de34835ce35c` contain 36 successful samples and 36
successful physical sessions. No sample used a simulator or CPU fallback.

## Execution and validation

The active path is:

```text
Stress18 circuit -> TensorNetwork -> greedy path -> ContractionDAG
  -> UpmemPlan -> ABI-v4 host/DPU runtime -> reconstruction
  -> CPU replay, policy validation, float32 accuracy, canonical evidence
```

Every route passed physical-target verification, startup and execution
resource admission, CPU physical-plan replay, full-precision float32
validation, and output hashing. Requested and observed resources matched.
The six routes are:

| Route | DPUs | Tasklets/DPU | Measurements |
|---|---:|---:|---:|
| `upmem_float32_1dpu_t1` | 1 | 1 | 5 |
| `upmem_float32_1dpu_t2` | 1 | 2 | 5 |
| `upmem_float32_1dpu_t4` | 1 | 4 | 5 |
| `upmem_float32_1dpu_t8` | 1 | 8 | 5 |
| `upmem_float32_2dpu_t8` | 2 | 8 | 5 |
| `upmem_float32_4dpu_t8` | 4 | 8 | 5 |

## Runtime results

Medians below use measurement samples only. Speedups are descriptive ratios of
route medians; they are not claim-eligible performance estimates.

| Route | Median total wall (s) | Raw MAD (s) | Median kernel (s) | Raw MAD (s) |
|---|---:|---:|---:|---:|
| 1 DPU x T1 | 30.0796 | 0.0354 | 24.3208 | 0.0118 |
| 1 DPU x T2 | 18.0566 | 0.0351 | 12.3420 | 0.0015 |
| 1 DPU x T4 | 12.0096 | 0.0268 | 6.3413 | 0.0049 |
| 1 DPU x T8 | 9.1998 | 0.0765 | 3.5947 | 0.0040 |
| 2 DPUs x T8 | 6.6853 | 0.0231 | 1.8005 | 0.0009 |
| 4 DPUs x T8 | 6.1062 | 0.0254 | 0.9783 | 0.0008 |

| Comparison | Kernel speedup | Total-wall speedup | Kernel efficiency | Total-wall efficiency |
|---|---:|---:|---:|---:|
| T1 -> T2 | 1.971 | 1.666 | 98.5% | 83.3% |
| T1 -> T4 | 3.835 | 2.505 | 95.9% | 62.6% |
| T1 -> T8 | 6.766 | 3.270 | 84.6% | 40.9% |
| 1 -> 2 DPUs at T8 | 1.996 | 1.376 | 99.8% | 68.8% |
| 1 -> 4 DPUs at T8 | 3.674 | 1.507 | 91.9% | 37.7% |

## Utilization and movement

| Route | Arithmetic tasklet utilization | Arithmetic DPU-slot utilization | Dominant-wave utilization |
|---|---:|---:|---:|
| 1 DPU x T1 | 100.0% | 100.0% | 100.0% |
| 1 DPU x T2 | 99.99% | 100.0% | 100.0% |
| 1 DPU x T4 | 99.93% | 100.0% | 100.0% |
| 1 DPU x T8 | 99.86% | 100.0% | 100.0% |
| 2 DPUs x T8 | 99.86% | 99.05% | 100.0% |
| 4 DPUs x T8 | 99.86% | 98.57% | 100.0% |

At 4 DPU x T8, the median transfer volume was 1,711,584 H2D bytes and
5,726,496 D2H bytes. These are recorded transfer facts, not a claim that all
host/device movement time or device counters have been measured by these
fields.

## Interpretation

Kernel work scales strongly along both measured axes. Total steady-execution
time scales less strongly because host-side and other non-kernel work does not
decrease proportionally with DPU computation. The high dominant-wave and
weighted utilization values show that the result is not primarily an
insufficient-work or empty-slot artifact. This experiment does not identify a
single optimization cause, and it does not claim general UPMEM acceleration.

Rank 1 was successfully qualified for this diagnostic. Rank 0 previously
exhibited a timeout at three or more DPUs; no claim is made that every rank is
healthy.

Two earlier runs are preserved as excluded incident evidence because their
physical accesses overlapped. They are not part of this 36-sample result. A
private host lock and rank-ownership preflight were used for the recovered run;
future physical runs must remain serial.

## Claims and limits

Allowed claims are physical correctness for this route matrix, descriptive
tasklet and one-rank DPU scaling on Stress18, and the measured steady-execution
timing composition. The result is powersave-conditioned, rank-specific,
single-workload evidence with five measurements per route.

This tag does not establish final `physical_performance_v1` estimates,
optimized-host performance, broad tensor-network scalability, multi-rank
scaling, slice parallelism, energy efficiency, int8 competitiveness, resident
intermediates, or UPMEM-aware path-search benefit. The sequential baseline
and later performance experiments must recollect contemporaneous comparison
routes rather than using this diagnostic as a permanent denominator.

The canonical evidence is under
`runs/eth/safari-baguette1/7e3ca432a3b109da15710b32dcb1edec9e4771fb/parallel-scaling/recovery-20260829T142657Z/raw/`.
The dedicated summary and figures are generated without changing canonical
evidence:

```bash
PYTHONPATH=src ../.venv/bin/python scripts/inspect_parallel_scaling.py \
  --input <canonical-evidence> \
  --summary-output <report>/parallel_scaling_summary.json \
  --output-dir <report> \
  --expected-source-commit 7e3ca432a3b109da15710b32dcb1edec9e4771fb
```
