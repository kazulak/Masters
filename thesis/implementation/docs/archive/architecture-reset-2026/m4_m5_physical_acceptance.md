# M4/M5 Physical Development Acceptance

Status: **development acceptance record**, not a final evidence capsule and not
a promoted thesis result.

The historical M4/M5.1/M5.2 observations were made on 2026-08-11 from branch
`feature/m4-m5-recovery`, source commit `2c09984`; their current-source
provenance checks were rerun from commit `7175ccb` after the evidence-contract
patch. The separate M5 v3 observation was made on 2026-08-13 from clean source
commit `5401597fdc2458087e112f5bd2e1869a5a0a5ab0`. All runs were copied and
audited as ignored development evidence; they are intentionally not included
in `thesis_results/current`.

The M5.1/M5.2 entries below are historical bounded physical probes. The
additive M5 execution-plan-v3 lane is documented separately below and has now
passed its bounded physical development-acceptance gate.

## Rank and Environment

Commands selected the requested physical rank explicitly:

```bash
UPMEM_HW_RANK_PATH=/dev/dpu_rank1 \
UPMEM_ALLOW_PHYSICAL_HARDWARE=1 make upmem-hw-m4-6-tasklet-scaling
UPMEM_HW_RANK_PATH=/dev/dpu_rank1 \
UPMEM_ALLOW_PHYSICAL_HARDWARE=1 make upmem-hw-m5-1
UPMEM_HW_RANK_PATH=/dev/dpu_rank1 \
UPMEM_ALLOW_PHYSICAL_HARDWARE=1 make upmem-hw-m5-2
```

The runner records the requested rank path and effective SDK profile. Those
fields establish requested/effective selection, not an independent observation
of the physical rank. On this ETH host, vendor diagnostics failed for the
default `/dev/dpu_rank0` and passed for ranks 1, 20, and 39. This is an
environment condition, not a benchmark result.

## M4.6 Tasklet Sweep

The suite covered 12 BV/BB84/EDC/XOR cases at 3/4/5 qubits, two path variants,
two numeric modes, and seven repeats: 336 rows per tasklet count and 1680 rows
in total. All rows passed physical execution and validation flags, with no
simulator or CPU fallback.

Observed run directories:

```text
runs/evidence/upmem_hardware_taskgraph_resident_m4_6_tasklet_scaling/
  upmem_hw_taskgraph_resident/2026-08-11_15-45-20  # tasklets=1
  upmem_hw_taskgraph_resident/2026-08-11_15-54-19  # tasklets=2
  upmem_hw_taskgraph_resident/2026-08-11_15-55-30  # tasklets=4
  upmem_hw_taskgraph_resident/2026-08-11_15-56-16  # tasklets=8
  upmem_hw_taskgraph_resident/2026-08-11_15-57-02  # tasklets=16
```

The shared-operation fix kept the maximum DPU stack at `256/1024` in the
accepted multi-tasklet build. Before the fix, the per-tasklet operation copy
exceeded the 1024-byte stack budget.

Development diagnostics showed a small-workload optimum near eight tasklets:
median DPU-cycle speedups versus one tasklet were `1.720x`, `2.340x`, `2.387x`,
and `2.192x` for 2/4/8/16 tasklets. The corresponding paired host-observed
steady-state ratios were `1.186x`, `1.300x`, `1.304x`, and `1.223x`. These are
development-run observations, not final scaling or speedup claims.

## M5.1 Output Partition

Run:

```text
runs/evidence/upmem_hardware_distributed_m5_1/
  upmem_hw_m5_1/2026-08-11_16-28-05
```

The bounded real-float32 contraction passed on 1/2/4 physical DPUs using
exclusive output-tile ownership. SimplePIM supplied management/allocation and
the thesis-owned kernel performed the contraction. The probe used zero warmups
and one repetition. It proves functionality only,
not distributed TaskGraph execution, scaling, communication performance, or
speedup.

## M5.2 Contracted-Axis Partition

Run:

```text
runs/evidence/upmem_hardware_distributed_m5_2/
  upmem_hw_m5_2/2026-08-11_16-28-15
```

The same bounded real-float32 contraction passed on 1/2/4 physical DPUs using
contracted-axis partials and deterministic ascending-DPU
`host_mediated_sum_v1` reduction. Maximum absolute error was `2.98e-08`. The
probe used zero warmups and one repetition. It proves host-mediated reduction
functionality only; it does not prove PID-Comm, DPU-to-DPU communication,
general distributed TaskGraph execution, scaling, or speedup.

Across all six M5 rows, requested and allocated DPU counts matched exactly,
`target_observed=physical_hardware`, `tasklets_per_dpu=1`, the expected
SimplePIM/kernel/communication provider identities were recorded,
`hardware_kernel_executed=true`, simulator execution and CPU fallback were
false, release was confirmed, scientific validation and the transfer invariant
passed, and rank 1 was explicitly requested; the M5.2 maximum absolute error
remains `2.98e-08`.

## Additive M5 execution-plan-v3 Lane

The v3 route is a one-rank single-contraction design accepting configured DPU
counts from `1..64` and tasklet counts from `1..24`. The accepted run used the
default DPU counts `1/2/4/8/16/32/64` and `8` tasklets, output or
contracted-axis partitioning, float32 or per-task int8, five workloads, and
synthetic strong/weak diagnostics.

Local hardware-free validation is complete. The exact command:

```bash
UPMEM_HW_M5_DPU_COUNTS=3 UPMEM_HW_M5_TASKLETS=3 make upmem-hw-m5-plan
```

prepares the configured plan set, preserves unsupported cases, reports failures
explicitly, and performs no DPU allocation or launch. The canonical physical
development command was:

```bash
UPMEM_HW_RANK_PATH=/dev/dpu_rank1 \
UPMEM_ALLOW_PHYSICAL_HARDWARE=1 make upmem-hw-m5
```

The audited source commit was
`5401597fdc2458087e112f5bd2e1869a5a0a5ab0` with a clean worktree. The ETH run
was `runs/evidence/upmem_hardware_distributed_m5/upmem_hw_m5/2026-08-13_15-01-11`;
the local ignored copy is
`runs/inbox/eth/m5_v3/canonical-2026-08-13_15-01-11`; and the fixed report is
`runs/comparisons/upmem_m5/2026-08-13_16-29-36_450221`. The normalized-record
hash is `1a7714b8dce25b0b0959ed08cae73aaf47e6d7084b90200d4895bf4c521202a0` and
the suite hash is `e71ec4518a99a8c7f463926da845b1c67bef7242c72233c0c0cfdc107177e26c`.

The 140 plan cells produced 92 prepared/executed cells, 644 completed
measured rows, 48 partition-incompatible unsupported rows, and 0 failures.
All physical/provider/rank/allocation/kernel/release/no-fallback/transfer/
validation checks passed. The report is complete; all nine plots and all
table/plot hashes are valid. This is bounded development acceptance only. The
route is one-rank multi-DPU execution of one contraction, not full distributed
TaskGraph execution. For this route SimplePIM is
`initialization_binary_and_management_state_only`; allocation, transfer, and
launch use raw synchronous UPMEM SDK calls. The thesis-owned C kernel performs
the contraction and the host performs the `float64` reduction. Both float32
and per-task resident int8 use float32 MRAM transport.

Output-versus-contracted-axis partitioning compares execution layout under a
fixed contraction plan. It is not a contraction-path comparison.

## PID-Comm Qualification

The allocation-free qualification commands were:

```bash
make upmem-pidcomm-plan
make upmem-pidcomm-compatibility
```

Qualification was blocked before DPU allocation under the installed ETH SDK
2023.1. The pinned PID-Comm source expects missing `dpu_alloc_comm`,
`DPU_FOREACH_ENTANGLED_GROUP`, and older PID-Comm API/source macros. There was
no simulator fallback, CPU fallback, allocation, or physical PID-Comm launch.

## Claims Allowed From This Record

This record supports bounded M5 v3 physical development execution and
descriptive same-route measurements for the admitted configurations, in
addition to the historical M4.6/M5.1/M5.2 functionality observations. The
float/int8 median runtime ratio is `0.165` (range `0.109--0.799`), the
output/contracted ratio is `0.980` (range `0.492--1.092`), and same-route
`T1/TN` is `0.634` (range `0.073--0.999`); none of these is a broad hardware
speedup claim. Float32 maximum error is `7.15e-06` within `1e-05`; int8 error
`0.0303011` is descriptive. The record does not support claims of a complete
UPMEM TN simulator, final architecture completion, general TaskGraph,
ATiM/SparseP execution, PID-Comm integration, multi-rank/DIMM execution,
energy efficiency, CPU/GPU speedup, planner superiority, or final scaling.
