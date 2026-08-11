# M4/M5 Physical Development Acceptance

Status: **development acceptance record**, not a final evidence capsule and not
a promoted thesis result.

This record covers the ETH UPMEM observations made on 2026-08-11 from branch
`feature/m4-m5-recovery`, source commit `2c09984`. The runs were copied/audited
as development evidence; they are intentionally not included in
`thesis_results/current`.
Current-head provenance validation was rerun from commit `7175ccb` after the
evidence-contract patch.

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

This record supports only that the bounded M4.6 tasklet route and M5.1/M5.2
single-contraction physical probes executed and validated on a selected ETH
UPMEM rank under their declared profiles. It does not support claims of a
complete UPMEM TN simulator, final architecture completion, ATiM or SparseP
execution, PID-Comm integration, multi-rank/DIMM execution, performance
advantage, energy efficiency, or final benchmark scaling.
