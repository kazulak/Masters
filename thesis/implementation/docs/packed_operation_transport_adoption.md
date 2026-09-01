# Packed Operation Transport Adoption v1

This document records the adoption of the packed operation transport for the
sequential, one-rank UPMEM tensor-network execution path.

## Scope

The execution contract is unchanged:

- deterministic full pre-measurement statevector simulation;
- exact, untruncated tensor-network contraction;
- one `ContractionDAG` and one `UpmemPlan` per sample;
- split-complex float32 transport;
- ABI-v4 and the WRAM-panel DPU kernel;
- host-roundtrip intermediates;
- one physical rank, CPU affinity `[0]`, and one-tasklet or multi-tasklet
  routes within that rank;
- `steady_execution_v1` timing and diagnostic claim policy.

The active host transport is `packed_operation_v1`: one variable-length
operation envelope is submitted to the existing persistent C host, which
executes the unchanged embedded ABI-v4 requests in deterministic order.
Directory request transport is retired from the active runtime path.

## Provenance

The physical qualification source is immutable:

```text
032b3ab5ba774fed0e61fc10eb02f60814c7a190
```

The reporting and qualification tools are on a clean descendant and are
recorded as `reporting_tool_source_commit` in the release bundle. The
execution tag is:

```text
thesis-upmem-packed-operation-transport-v1
```

The selected physical host was `safari-baguette1`, rank `/dev/dpu_rank1`,
CPU 0, with the powersave governor. The recorded SDK compiler is UPMEM SDK
0.29.1 with clang 12.0.0. The remote Python environment used Python 3.10.20,
NumPy 2.2.6, and one thread for OpenMP, OpenBLAS, MKL, and NumExpr.

## Physical qualification

The generalized-resource correctness qualification used experiment
`219686cdad5641e0b0e5f0e9a794899914c882c71be5270d91a1ea379bfb1384`, run
`7798a90d-377e-4f66-b753-f5b4123ec02c`, with five successful samples and five
released physical sessions:

| Route | Result | Resource fact |
| --- | --- | --- |
| 1 DPU x T3 | pass | odd tasklet count |
| 1 DPU x T7 | pass | odd tasklet count |
| 1 DPU x T12 | pass | non-power tasklet count |
| 1 DPU x T24 | pass | 8 idle tasklets in the relevant local work |
| 3 DPUs x T8 | pass | non-power DPU count |

All routes passed physical provenance, resource admission, replay, float32
validation, output hashing, and packed-transport checks. The T24 route is a
correctness result; its idle-tasklet condition is not a performance claim.

The packed six-route diagnostic used experiment
`a809d28bf3242440a6e6e249fb1836c78f86ac9a24c081030c3accea3638dba3`, run
`ca087652-bc19-470d-914c-bd2dae67e793`. It contains one warmup and five
measurements per route, 36 successful samples, 36 successful sessions, and no
failed or unsupported attempts.

## Optimized diagnostic results

The following are medians over the five measured samples. Raw MAD is the
median absolute deviation of those five observations. Times are seconds;
transfer values are application-visible bytes.

| Route | Total wall median | Total wall MAD | Kernel median | Kernel MAD | H2D bytes | D2H bytes | Tasklet util. | DPU util. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 DPU x T1 | 27.475147 | 0.028630 | 24.332646 | 0.004394 | 1,592,352 | 5,660,256 | 1.000000 | 1.000000 |
| 1 DPU x T2 | 15.413002 | 0.006156 | 12.335475 | 0.004307 | 1,592,352 | 5,660,256 | 0.999877 | 1.000000 |
| 1 DPU x T4 | 9.407978 | 0.023211 | 6.340718 | 0.004567 | 1,592,352 | 5,660,256 | 0.999341 | 1.000000 |
| 1 DPU x T8 | 6.595403 | 0.009741 | 3.587937 | 0.002360 | 1,592,352 | 5,660,256 | 0.998620 | 1.000000 |
| 2 DPUs x T8 | 4.832369 | 0.013462 | 1.800743 | 0.002944 | 1,632,096 | 5,682,336 | 0.998620 | 0.990454 |
| 4 DPUs x T8 | 4.275410 | 0.011157 | 0.977901 | 0.001107 | 1,711,584 | 5,726,496 | 0.998620 | 0.985680 |

The descriptive comparisons are:

| Comparison | Kernel speedup | Kernel efficiency | Total-wall speedup | Total-wall efficiency |
| --- | ---: | ---: | ---: | ---: |
| T1 -> T2 | 1.973x | 98.63% | 1.783x | 89.13% |
| T1 -> T4 | 3.838x | 95.94% | 2.920x | 73.01% |
| T1 -> T8 | 6.782x | 84.77% | 4.166x | 52.07% |
| 1 -> 2 DPUs at T8 | 1.992x | 99.62% | 1.365x | 68.24% |
| 1 -> 4 DPUs at T8 | 3.669x | 91.73% | 1.543x | 38.57% |

Kernel and total-wall scaling are reported separately. The reduced total-wall
efficiency is consistent with a remaining host-side floor; it is not evidence
of a kernel or numerical change.

## Transport A/B evidence

The earlier contemporaneous A/B experiment used source
`b04c2cfcf4a62603edbfea5dc0320147c2c518ad`, experiment
`9e5ff1f81e918bdcea2bb8c972720f9d6cdfc116a7907f3b9a31061b25e6daea`, and run
`fc093eb8-75aa-4db3-b54b-182c7047719a`. It compared the accepted directory
transport with the packed transport for Stress18, HS18, and GHZ18 at one and
four DPUs with five measured blocks per arm. All 72 samples and sessions
passed. The generic report retained zero claim-eligible speedup rows.

| Circuit and route | Steady-wall reduction | Session-inclusive reduction | Request-build reduction | Payload-staging reduction | Kernel change |
| --- | ---: | ---: | ---: | ---: | ---: |
| Stress18, 1 DPU x T8 | 25.13% | 24.03% | 61.87% | 68.57% | +0.58% |
| Stress18, 4 DPUs x T8 | 15.22% | 14.23% | 42.42% | 53.44% | -0.04% |
| HS18, 1 DPU x T8 | 6.77% | 6.18% | 30.38% | 46.46% | -0.14% |
| HS18, 4 DPUs x T8 | 8.48% | 7.67% | 29.40% | 42.89% | -0.06% |
| GHZ18, 1 DPU x T8 | 31.08% | 30.65% | 64.15% | 63.95% | +1.04% |
| GHZ18, 4 DPUs x T8 | 6.10% | 5.90% | 49.06% | 55.19% | -1.28% |

The packed path improved both steady and session-inclusive time in every A/B
cell while leaving kernel timing effectively stable. This supports boundary
count reduction as the causal interpretation. It does not turn the diagnostic
into a final performance campaign.

## Failure semantics and memory

Packed transport is not an atomic transaction. If an embedded request fails,
the response preserves the failed request index and completed-request count;
the sample fails, no completed request is silently rerun, and session cleanup
and evidence preservation remain required.

The envelope is scoped to one contraction operation. Descriptor count, envelope
bytes, payload bytes, and packed request counts are recorded in backend facts.
The six-route diagnostic observed at most 64 descriptors and 329,852 envelope
bytes for the one-DPU routes, and at most 16 descriptors and 309,980 bytes for
the four-DPU route. The binary inventory and hashes are included in the
bundle.

## Claims

This milestone supports the following narrow statement:

> Packed operation transport was physically qualified for the selected
> Stress18 workload and for the tested generalized-resource routes. In a
> contemporaneous diagnostic A/B across Stress18, HS18, and GHZ18, it reduced
> repeated host request-boundary work and improved both steady and
> session-inclusive execution while preserving ABI-v4 requests, numerical
> results, resource facts, and DPU kernel behavior.

The result remains diagnostic and powersave-conditioned. It does not claim
final `physical_performance_v1` estimates, arbitrary-circuit performance,
arbitrary resource-count performance, CPU/GPU competitiveness, multi-rank or
sliced performance, energy efficiency, or whole circuit-to-result acceleration.

The next formal campaign must recollect contemporaneous controls under its own
pre-registered protocol. The historical directory-arm result is context and
the packed six-route diagnostic is not a substitute for the final 2+30
campaign.
