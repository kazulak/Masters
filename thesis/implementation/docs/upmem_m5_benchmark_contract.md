# M5 UPMEM Benchmark Contract

This document defines the input and claim boundary for the additive M5
execution-plan-v3 route and `scripts/upmem_m5_report.py`.
The command is a standalone report generator. It reads an evidence run's
`normalized_records.jsonl` and writes a derived comparison below
`<output-root>/runs/comparisons/upmem_m5/<timestamp>/`. It never writes plots or
other report artifacts into the evidence run.

## Input

`--input` may name either an evidence-run directory or its
`normalized_records.jsonl`. Each non-empty line must be a JSON object. Records
are ordered deterministically before processing. A missing or empty source is
reported as missing data, with visible TODO plot placeholders and null numeric
fields; the reporter does not synthesize values.

The normalized fields are intentionally tolerant of existing evidence naming:

| Contract dimension | Accepted primary aliases |
| --- | --- |
| Case | `case_id`, `workload_case_id`, `benchmark_case_id` |
| Route | `route_id`, `route`, `route_label` |
| Numeric mode | `numeric_mode`, `quantization_mode`, `precision_mode` |
| Partition | `partition_mode`, `partition`, `partition_strategy` |
| Target admission | `target_observed` must be `physical_hardware` for scientific rows |
| Rank admission | `one_rank: true` or an explicit rank count of `1` |
| Fallback admission | `cpu_fallback_used`, `simulator_kernel_executed`, and `fallback_used` must all be `false` |
| DPU count | `requested_dpu_count`, `allocated_dpu_count`, `observed_dpu_count`, `dpu_count` |
| Tasklets | `tasklets_per_dpu`, `tasklets`, `tasklet_count` |
| Timing scope | `timing_scope` |
| Status | `status`, `execution_status`, `result_status` |
| Runtime | `timing_s`, `timing.total_time_s`, `per_repeat_timing.total_time_s` |
| Scaling | `scaling_kind`, `scaling_mode`, `scaling_type`, `weak_scaling`, `strong_scaling` |

Only rows with the physical, one-rank, no-fallback admission evidence above and
a successful status contribute numeric scientific data. Missing admission
evidence is fail-closed. Failed or unsupported rows remain in the source
record table; their report DPU dimension uses the requested count, while
`allocated_dpu_count` remains a separate field. Successful measured rows use
the allocated count.

Nested `timing`, `transfers`/`transfer`, `load_balance`/`load_balance_metrics`,
`validation`, and `native_response` objects are traversed for accepted metric
fields.

Successful statuses are `completed`, `passed`, `success`, `succeeded`, `ok`,
and `validated`. All other statuses are retained as unsupported, failed, or
unknown rows and do not contribute measured statistics.

## Execution-plan-v3 lane

The v3 lane is an active one-rank multi-DPU single-contraction route, not full
distributed TaskGraph execution. It supports output-tile or contracted-axis
partitioning. Numeric modes are float32 and per-task resident int8
requantization; both use float32 MRAM transport. The suite contains real
highest-work contractions plus explicit synthetic strong- and weak-scaling
diagnostics. Partition mode is an execution-layout variable, not a
contraction-path comparison; paired rows retain the same contraction plan.

Local hardware-free validation is complete. The exact preparation command is:

```bash
UPMEM_HW_M5_DPU_COUNTS=3 UPMEM_HW_M5_TASKLETS=3 make upmem-hw-m5-plan
```

It prepares the configured plan set, preserves unsupported cases, reports
failures explicitly, and performs no DPU allocation or launch. Physical ETH execution is pending;
the future command is:

```bash
UPMEM_HW_RANK_PATH=/dev/dpu_rank1 \
UPMEM_ALLOW_PHYSICAL_HARDWARE=1 make upmem-hw-m5
```

The command is an execution route, not physical acceptance. Until it is run
and reviewed, no performance or scaling claim is allowed. SimplePIM's v3 role
is `initialization_binary_and_management_state_only`; allocation, transfer,
and launch use raw synchronous UPMEM SDK calls. The thesis-owned C kernel, SDK
transfers, and host `float64` reduction are not SimplePIM compute operators.

## Tables and statistics

The report writes CSV tables for source rows, runtime, accuracy, transfer, and
strong-scaling ratios, plus `m5_numeric_mode_ratios.csv` and
`m5_partition_ratios.csv`. Runtime and auxiliary metric tables group by the exact
tuple:

`case_id + route_id + numeric_mode + partition_mode + tasklets_per_dpu + timing_scope + workload_kind + scaling_kind + dpu_count`

For each group, statistics are median, linear-percentile IQR (`P75 - P25`),
minimum, maximum, and repeat count. The tables also expose measured repeat
count and unsupported/failure count. Failed and unsupported source rows remain
visible in `tables/m5_records.csv` and in groups with null statistics.

## Scaling and pairing

Strong-scaling runtime rows are measured physical one-rank rows explicitly
marked `strong`. A ratio is emitted only when the exact same case, route,
numeric mode, partition, tasklet count, timing scope, workload kind, and
scaling kind has a measured DPU-count-1 baseline. Pairing is within-route and
one-rank only; the resulting speedup is a diagnostic, not a general hardware
scaling claim. For a target count `N`:

```text
speedup    = T1 / TN
efficiency = speedup / N
```

`T1` and `TN` are the corresponding group medians. There is no route fallback,
numeric-mode fallback, partition fallback, tasklet/timing-scope fallback, or
cross-case pairing. Weak-scaling plots use only rows explicitly marked
`weak`/`weak_scaling` or with `weak_scaling=true`; otherwise the plot is a TODO
rather than an inferred experiment.

## Plot wording and evidence boundary

The nine fixed plot names are:

`m5_strong_scaling_runtime.png`, `m5_strong_scaling_speedup.png`,
`m5_strong_scaling_efficiency.png`, `m5_weak_scaling_runtime.png`,
`m5_numeric_mode_runtime_ratio.png`, `m5_partition_runtime_ratio.png`,
`m5_transfer_breakdown.png`, `m5_load_balance.png`, and
`m5_quantization_accuracy.png`.

The ratio CSVs and plots use same-route, same-plan paired medians. Numeric-mode
ratio is `T_float32/T_int8`; values greater than 1 mean int8 is faster under
the measured timing scope, while values below 1 mean float32 is faster.
Partition ratio is `T_output/T_contracted`; values greater than 1 mean the
contracted-axis partition is faster, while values below 1 mean output
partitioning is faster. These are diagnostic within-study ratios, not broad
hardware speedup claims.

Captions identify the results as **physical one-rank, measured** evidence.
Where transfer or partition fields support it, captions say **host-mediated
reduction where applicable**. The accuracy caption says **on-DPU int8
requantization with float32 MRAM transport** only when the corresponding
arithmetic, transport, scope, and unpacked-transfer evidence is present.
These labels describe the contracted evidence boundary; they do not promote a
bounded probe into a general architecture result. The timestamp component is
restricted to a single safe path component, so report generation cannot
traverse outside the comparison directory.

`plot_manifest.json` records per-plot generated/TODO status, source hash,
artifact hashes, and the supported and failed DPU counts. `m5_summary.md`
records the source run path and SHA-256, allowed claims, and not-allowed
claims.

## Allowed claims

- Descriptive measured timing, transfer, load-balance, and accuracy summaries
  for the exact recorded dimensions.
- Within-key, within-route one-rank diagnostic `T1/TN` and `speedup/N` ratios
  when the required baseline and target rows are present.
- Retention and enumeration of unsupported or failed DPU-count attempts.

## Not allowed claims

- Cross-route or otherwise incompatible speedup, efficiency, or accuracy
  pairing.
- PID-Comm communication, multi-rank execution, packed-int8 transport, or
  general distributed TaskGraph support.
- Energy, CPU/GPU speedup, general UPMEM performance, broad hardware speedup, or extrapolated
  scaling claims.

## Deferred M5 items

The following remain explicitly deferred and are not represented as completed
by this report:

- **PID-Comm:** qualify the communication provider and a physical collective
  or relocation path under a compatible SDK.
- **Multi-rank:** execute and validate multiple ranks or DIMMs rather than the
  physical one-rank report boundary.
- **Packed-int8:** measure packed-int8 transport and its memory/transfer
  accounting; float32 MRAM transport must not be relabeled as packed-int8.
- **Distributed TaskGraph:** implement general distributed TaskGraph
  scheduling, ownership, dependency movement, and validation beyond the
  bounded single-contraction v3 lane and historical M5.1/M5.2 probes.
- **Energy and CPU/GPU speedup:** defer both until separately instrumented,
  matched experiments with an explicit measurement contract.
