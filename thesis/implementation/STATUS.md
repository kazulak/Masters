# Final Implementation Status

This file describes the final thesis implementation and accepted research state. It
supersedes earlier status text that described physical qualification, DAG scheduling, or
hardware-calibrated path optimization as future work.

## Frozen scientific identities

| Role | Identity |
| --- | --- |
| Final UPMEM executor | `459935f586fdd16c82013838e6d27a12604c3093` |
| Executor tag | `thesis-upmem-kernel-schedule-system-v1` |
| P6 qualified software | `2beea27411c16e90ed76988613ddb00bcc09f942` |
| P6 software tag | `thesis-upmem-cost-guided-software-v1` |
| P6 accepted result package | `8df2ebac61bacd08309ea490309be5a8dcb943b2` |
| P6 results tag | `thesis-upmem-cost-guided-results-v1` |

Later documentation/publication commits do not replace these experiment identities.

## Final capability matrix

| Capability | Final status | Evidence/claim boundary |
| --- | --- | --- |
| Circuit -> target-neutral TN lowering | Complete | Supported circuit/query scope only |
| Complete path -> `ContractionDAG` lowering | Complete | Exact, untruncated contraction path |
| NumPy same-DAG replay | Complete | Correctness/reference route |
| Quimb/cotengra CPU TN adapters | Complete | External CPU TN baselines |
| QuEST CPU/GPU adapters | Complete | GPU claims only where real GPU execution was verified |
| UPMEM physical mapping | Complete for retained one-rank profile | Bounded output/K tiling and declared resource admission |
| Persistent packed-wave transport | Retained | `packed_wave_v1` |
| WRAM-panel kernel | Retained | `panel_only_v1`, KC=64, NC=32 |
| Tasklet parallelism | Retained and physically studied | T1-T24 build qualification; measured subsets reported explicitly |
| Multi-DPU contraction | Retained and physically studied | One-rank resource scaling only |
| Independent DAG-wave execution | Retained and physically studied | Dependency-ready disjoint DPU groups, synchronous cohorts |
| Four-product complex fusion | Retained when admitted | Generic UPMEM fallback when fused layout is not admitted |
| Outer-K1 specialization | Completed negative experiment; not retained | No general claim that all shape specialization is unhelpful |
| Exact slicing/concurrency | Bounded experiment | Workload-dependent; not automatic production selection |
| DPU-resident intermediate pair | Completed negative bounded experiment; not retained | Does not rule out other residency designs |
| Shared-scale complex int8 | Complete diagnostic policy | Same-policy correctness; approximation error reported separately |
| Hardware-aware cost model | Complete | Ranking surrogate, not seconds predictor or physical constants |
| UPMEM-aware reranking | Complete and physically evaluated | Improves conventional selection in the tested P6 domain |
| UPMEM-guided adaptive generation | Complete and physically evaluated | No resolved benefit over reranking under the tested budget |
| Multi-rank execution | Out of final thesis scope | No claim |
| Async host/DPU overlap | Out of final thesis scope | No claim |
| Energy measurement | Not implemented | No energy-efficiency claim |
| Automatic CPU/GPU/UPMEM placement | Out of final thesis scope | No claim |

## Accepted execution-system findings

The final composed Stress16 scaling diagnostic reported:

```text
D1T1 session-inclusive median: 7.104246 s
D1T16 session-inclusive median: 1.589183 s
paired geometric speedup:       4.501040x

D1T8 session-inclusive median:  1.834585 s
D4T8 session-inclusive median:  1.199998 s
paired geometric speedup:       1.530679x
```

Fresh fusion confirmation reported 1.9034x session-inclusive speedup for the confirmed
development cell.

The six-cell serial/static-DAG development A/B reported a 1.2142126437x
session-inclusive equal-cell geometric speedup. A separately selected Stress16 D4/T8
development confirmation reported 1.431726322x.

Outer-K1 specialization failed its adoption gate. The bounded Stress16 resident-pair
probe and Stress16 slicing cells were valid negative results; EDC14 D4 slicing produced a
positive development confirmation. These outcomes are retained as research findings
rather than reopened optimization tasks.

## Accepted P6 state

P6 executed:

```text
initial calibration: 192
feedback round 1:    144
feedback round 2:    144
evaluation:          198
total:               678
ceiling:              768
retries:                0
replacements:           0
```

The evaluation maximum was 288 attempts, but coincident method selections were
deduplicated before physical execution. The frozen evaluation required 33 distinct
cell/path executions per block, therefore `33 * 6 = 198` physical attempts.

Final cost-model integer weights:

```text
[1, 2, 1, 1, 5]
```

The final model diagnostics report 1,001 coefficient tuples, 47 distinct measured-pool
selection vectors, and 16 tuples tied at the best rounded training objective. Therefore
the fitted weights must not be interpreted as unique physical coefficients or measured
runtime shares.

## P6 primary result

```text
R/U session-inclusive:
ratio: 1.0013425605931643
descriptive paired-block 95% interval:
[0.9948355448727421, 1.0085087497299605]
same path: 8/12 cells
```

No additional physical advantage of adaptive UPMEM-guided generation over UPMEM-aware
reranking is resolved under the tested 128-proposal protocol.

UPMEM-aware selection remains useful:

```text
F/R session-inclusive overall: 1.0390296080428785x
F/U session-inclusive overall: 1.040424568249768x
F/U session-inclusive, 4 DPU:  1.0818412974163845x
F/U steady wall, 4 DPU:        1.1153755638648002x
G/U session-inclusive overall: 1.2665079983332086x
```

## Generalization boundary

The P6 test set contains six circuit families represented in both development and test
with distinct instances/sizes. The result therefore tests **instance/size transfer within
represented families**, not family-held-out generalization.

The paired bootstrap intervals resample five complete timing blocks under one paired
search-seed schedule. They are descriptive and are not equivalence tests or
optimizer-population confidence intervals.

## Research status

The implementation research phase is closed. Future work belongs in a new study rather
than being added to the accepted thesis campaign because a result is neutral or because
an excluded feature could be interesting.
