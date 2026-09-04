# Quantized UPMEM Execution Diagnostic v1

This document freezes the physical diagnostic realization of the software
policy `complex_int8_shared_scale_v1`. It is a descriptive, accuracy-
unqualified result. It does not make a universal speedup, accuracy, or
resource-topology claim.

## Provenance

The authoritative software policy is commit
`6c9ca849a5ccc246dc645b63598ee391da75c599`. The physical execution source is
commit `c0ec6c76439e418e537a953a6b768ce2e1ea0dc6`. The reporting correction is
a descendant of the physical source and is recorded separately in the
release metadata.

The accepted physical diagnostic is run
`49339c75-dac1-438d-9307-dc77ebe5805d`, experiment
`7797fa21b27957e070633ed69219ce774ddd3861881477d610461f8b037f6ff2`. It has
180 successful samples and 180 fresh sessions: 30 warmups and 150 measured
samples across three circuits, two policies, and five resource routes. The
accepted pilot has 4 samples and 4 sessions. An interrupted 18-sample/19-
session incident is preserved separately and is excluded from every result.

Execution used one rank, `/dev/dpu_rank1`, CPU affinity 0, and the recorded
`powersave` governor. SDK, compiler, binary, source, and worktree identities
are retained in the evidence capsule.

## Frozen physical numerical boundary

For each contraction operand, the host discovers one float64 shared scale,
encodes contiguous real and imaginary int8 planes with deterministic
round-to-nearest-even, and restricts values to `[-127, 127]`. The four integer
lanes are `rr`, `ii`, `ri`, and `ir`; their signs are `rr - ii` and `ri + ir`.
Packed descriptors are little-endian and raw integer lanes are recorded as
`<i4`.

Each DPU product lane uses an int32 accumulator with per-lane bound
`K_local * 127^2`. The stronger combined-component bound is
`2 * K_local * 127^2`; host combination across lanes and K chunks uses int64.
The scale product is evaluated as float64, the reconstructed intermediate is
materialized as complex64, and a later contraction quantizes that intermediate
again with newly discovered scales.

The physical route is therefore hybrid:

```text
host complex intermediate
  -> shared-scale int8 encoding
  -> compact int8 H2D operands
  -> integer DPU contraction
  -> host complex64 reconstruction/dequantization
  -> later host requantization
```

It is not a fully integer-resident or DPU-resident tensor-network execution.
Logical storage is approximately `2n + 8` bytes for `n` complex int8 values,
versus `8n` bytes for complex float32. This is logical encoded size; it is not
a claim about measured H2D, MRAM, or WRAM traffic.

## Timing interpretation

`total_wall_s` is the steady sample wall time. Session-inclusive time is the
fresh session open time plus sample wall time plus session close time. H2D,
kernel, and D2H are reported only where the one-rank timing contract provides
global values.

Request-wave, request-build, rank-submit, response-wait, and native-route
timers are inclusive envelopes with different boundaries. They are retained
in `quantized_upmem_timing_envelopes.csv` and are not additive peer phases.
Disjoint host/native attribution is calculated per sample before summary:

```text
native_request_overhead = native_route - H2D - kernel - D2H
host_request_overhead = request_wave - native_route
operation_other = operation_total
                  - preparation - encode - request_wave - assembly - decode
coordinator_other = sample_wall - sum(operation_total) - host_reduce
```

Residuals below `1e-6` seconds are numerical zero; materially negative
residuals fail validation. Medians and raw MADs are calculated from paired
per-sample components. No parent-minus-sum-of-medians calculation is used.

## Results and comparison scopes

Fixed-route ratios compare float32 and int8 at the same circuit, path,
resource topology, and timing scope. Across the tested cells, int8 improved
kernel, steady-wall, and session-inclusive time. The corrected tables retain
the complete per-cell medians and raw MADs.

Best-route rows select the minimum median steady wall only within the five
tested routes. They are best observed routes in a finite diagnostic grid.
GHZ18 selected 4 DPUs × T8 for both policies. HS18 and Stress18 selected
4 DPUs × T8 for float32 and 2 DPUs × T8 for int8. These are not global
resource optima.

The int8 route reduced measured H2D bytes, but by less than its approximately
3.3–4.0x logical operand compression; D2H remains a complex64 reconstruction
boundary. The complete route table reports wall time, session-inclusive time,
H2D, D2H, kernel, host attribution, logical sizes, and MADs separately.

Numerically, GHZ18 and HS18 reproduced their float32 same-DAG results for the
tested paths, while Stress18 showed approximately 8.3% relative L2 error
against that reference. Every physical int8 sample matched the CPU same-policy
replay, including the recorded exact integer lanes. The observed approximation
error is therefore a policy/circuit/path result, not evidence of a physical
implementation mismatch. No post-hoc accuracy threshold is applied.

## Claim boundary

Supported statements are limited to physical same-policy correctness,
descriptive fixed-route performance and movement, numerical error on the
tested circuits, and best-observed policy-conditioned route selection.

This diagnostic does not establish universal int8 accuracy or speed, CPU/GPU
competitiveness, a 4x physical transfer reduction, a fully quantized
end-to-end pipeline, a globally best topology, path/quantization co-optimization,
or final thesis performance. Numerical approximation error remains separate
from the future `E_num` execution-overhead term and should be handled later as
an admissibility constraint or Pareto dimension.

No further int8 optimization or physical A/B experiment is included in this
freeze.
