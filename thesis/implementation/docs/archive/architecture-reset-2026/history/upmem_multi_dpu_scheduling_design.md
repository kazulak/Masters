# UPMEM Multi-DPU Scheduling Design

Status: M2 foundation/MVP and M2.1 useful-slice physical acceptance complete.
M3.1 also passed its bounded two-wave frontier qualification. This document
still describes the bounded contract, not a general scheduler.

This document separates the committed M2 execution contract from the target
multi-DPU scheduler. M2 proves one fixed form of coarse parallelism. It does
not implement general TaskGraph scheduling or the full M2 architecture.

## Implemented M2 Contract

The active route is
`upmem_tn_hardware_sliced_resident_two_dpu`, with backend
`upmem_sdk_hardware_sliced_resident_two_dpu` and profile
`hardware_sliced_resident_two_dpu_m2_v1`.

The route accepts only the terminal boundary of a one-qubit, one-operation,
real-valued X/H/Z circuit. Its greedy TaskGraph has one operation and a
dimension-two contracted index. That index is split into exactly two independent
slices:

```text
slice 0 -> physical DPU 0 -> one resident package -> one tasklet
slice 1 -> physical DPU 1 -> one resident package -> one tasklet
```

The native host launches the DPU set asynchronously and synchronizes it once.
Each DPU returns a float32 partial output. Python reconstructs the logical
output by summing those two partial outputs. There is no native reduction, retry,
CPU fallback, simulator fallback, or alternate route.

The committed X/H/Z suite and its normalized evidence workflow are the source of
truth. Use the [M2 runbook](upmem_hardware_sliced_resident_mvp_runbook.md) for
the exact ETH commands, artifact layout, and acceptance fields; do not duplicate
those details here.

## Evidence Boundary

The implementation records route/backend/profile identity, source hashes, slice
and DPU ownership, allocation, asynchronous launch, synchronization, release,
partial-output reconstruction, validation, transfer accounting, timings, and
failure evidence in normalized records. Physical acceptance passed for the
declared bounded M2/M2.1 contract. The result is functionality evidence only;
it is not a general scheduler or a scaling result.

The M2 MVP permits only a functionality and diagnostic timing statement. It
does not support a speedup, energy, scaling, communication-performance, or
general-TaskGraph claim. A successful local test or package build is not physical
acceptance.

## What Is Not Implemented

- scheduling dependency-ready contractions or independent subtrees;
- extending terminal slicing to larger multi-operation graphs;
- splitting one large contraction beyond this fixed two-slice boundary;
- multi-tasklet execution or dynamic DPU-group sizing;
- PID-Comm collectives or distributed intermediate relocation;
- operation classification, provider dispatch, and specialized kernels; and
- hardware-calibrated scheduling, scaling, or energy measurement.

## Subsequent Milestones

The original M2 control fixture used a `|0>` case whose second slice was zero;
the separate M2.1 fixture corrected this and passed with two nonzero useful
partials. M3.1 then passed dependency-safe three-task/two-wave dispatch on two
DPUs, without an overlap or scaling claim. The next architecture target is the
descriptor-driven M4.5 shared runtime, which must use explicit provider and
execution-plan terminology rather than treating these fixed lanes as a general
TaskGraph executor. See the [ETH evidence audit](upmem_m2_eth_evidence_analysis.md).

SimplePIM, PID-Comm, ATiM, and SparseP are central components of the subsequent
architecture: SimplePIM is physically qualified only for bounded
management/operator lanes, PID-Comm is for relocation and collectives, ATiM is
for generated dense local kernels, and SparseP is for sparse formats, kernels,
and load balancing. They are provider/kernel/communication components behind
thesis-owned planning and adapter interfaces, not interchangeable runtimes or
current M2 substitutes.
