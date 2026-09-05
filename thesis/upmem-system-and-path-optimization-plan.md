# UPMEM Execution and Contraction-Path Study

Date: 5 September 2026. Status: bounded research roadmap. This document does
not change executable code, frozen configurations, candidate pools or evidence.
It supersedes the earlier heterogeneous-placement and runtime-expansion roadmap.

## 1. Research Objective

Study exact tensor-network contraction for quantum-circuit simulation using
UPMEM. Characterize how contraction geometry, contraction paths, and resource
counts affect kernel performance, host overhead, data movement, and numerical
correctness.

The bounded research question is:

> How do contraction-path choice and tasklet/DPU parallelism affect the execution
> time and data movement of exact tensor-network quantum-circuit simulation on
> UPMEM?

The objective is not to build the fastest heterogeneous simulator or eliminate
every performance bottleneck. Exact means an untruncated tensor network and full
pre-measurement statevector; finite-precision error must still be measured.

### Execution Boundary

Python/host code performs circuit construction, TN lowering, contraction-path
planning, request preparation, and reconstruction required by the established
UPMEM execution policy.

Supported contraction arithmetic executes on UPMEM. The system does not choose
CPU execution instead of UPMEM based on tensor size or predicted performance.
Unsupported cases are explicitly rejected, not silently redirected.

CPU execution remains available for reference validation and separately
identified comparisons. Necessary host reconstruction, including split-K
assembly and complex decoding, is not a heterogeneous placement policy.

### Explicitly Out of Scope

- CPU/UPMEM contraction placement and placement thresholds.
- Concurrent CPU and DPU contraction scheduling.
- Automatic selection between CPU and UPMEM backends.
- Broad optimization toward a competitive general-purpose simulator.

These are removed deliverables, not deferred phases of this roadmap.

Small contractions performing poorly on UPMEM are a valid research finding.
Host overhead limiting application-level speedup is a result to quantify, not
an obligation to remove through CPU offloading.

## 2. Baseline and Pending Gate

Recorded source identities below are checkpoints, not a claim that remote
branches cannot advance. Verify exact identities before physical execution.

| Role | Branch or source | Full SHA |
| --- | --- | --- |
| Accepted main checkpoint | main | fa0dedf628a3612371daa4f6502da4d5465bbaff |
| Quantization reporting input | feature/upmem-quantized-physical-v1 | 62505ae637bdd3cf963b70f61754bf90a658b527 |
| Path-infrastructure input | feature/upmem-path-heuristic-generalization-v1 | e225947a84f937629ed46003dad7d8edff160a8f |
| Published integration execution, not yet adopted | feature/upmem-execution-integration-v1 | b921b8804e324da75222354ee2f4df41e770b75c |
| Software-preparation branch base | feature/upmem-execution-preparation-v1 | 5b93f87c1a034944859348c99e2fe263961a2114 |

The integration checkpoint has 1088 passing software tests, Ruff and exact-head
CI qualification, and a corrected 14/14 SDK sample/session gate with two verified
evidence copies. Its seven-session physical gate is still pending.
See [integration provenance](implementation/docs/upmem_execution_integration_v1.md#qualified-execution-checkpoint).

The pending physical gate remains unchanged: Bell2 float32/int8 at 1D/T1;
Stress14 float32/int8 at 1D/T8, float32 at 4D/T8, and int8 at 3D/T8 and 4D/T8.
Seven one-shot attempts, no warmups, no timing claims. Preserve
`configs/tn_benchmark_upmem_execution_integration_physical_v1.yml`.
Documentation or isolated analysis changes do not require repeating that
checkpoint's completed software/SDK gate.

The recorded local SDK is 2025.1.0; ETH is 2023.1.0. Record actual SDK provenance,
not the pkg-config utility version. Do not assume cross-environment equivalence.

Freeze one ContractionDAG and one UpmemPlan, packed_operation_v1, ABI-v4,
the WRAM-panel kernel, deterministic work mapping, host-roundtrip intermediate
policy, one rank, and declared numerical semantics. No dispatch framework or
new runtime is required. The sequential release already uses the WRAM kernel;
do not relabel it a naive scalar-MRAM implementation.

### Existing Diagnostic Context

| Stress18 route | Total-wall median (s) | Kernel median (s) |
| --- | ---: | ---: |
| 1 DPU x T1 | 27.475147 | 24.332646 |
| 4 DPUs x T8 | 4.275410 | 0.977901 |

These accepted packed-transport diagnostic medians describe 6.43x total and
24.88x kernel endpoint ratios. The second endpoint has a 77.1% non-kernel share,
calculated from component medians. This is not a Python-CPU measurement, paired
speedup estimate, removable fraction or end-to-end simulator claim. Do not mix
it with the proposed 4.9x/84% figures from a different comparison.

At b921b88, runtime.py submits RR, II, RI and IR lane envelopes, each containing
its waves. The persistent C host launches once per embedded request: four
launches per active wave, not four process starts. It also performs serial
per-DPU transfers. Preserve these facts when interpreting the host floor.

## 3. Finite Roadmap

| Phase | Deliverable | Completion Gate |
| --- | --- | --- |
| A. Qualify and freeze | Adopt the existing integrated executor | Unchanged physical gate passes; exact source, binaries and evidence retained |
| B. Characterize | Fixed-path geometry, movement, host/kernel and feasibility census | Explicit counts, bounds, exclusions and numerical-policy distinctions; no placement rule |
| C. Resource scaling | Bounded one-rank tasklet/DPU curves | Correctness and admission, matched fixed paths, paired diagnostic analysis |
| D. Path study | UPMEM-aware selector versus conventional paths | Frozen candidates, finite calibration, offline fitting, frozen profile and held-out evaluation |
| E. Report and stop | Benefits, limitations and negative results | Complete raw evidence, checksums, claim boundaries and thesis tables |

A stable platform with well-explained limitations is sufficient. Additional
runtime improvements are not prerequisites for phases C or D. Do not assume
the physical gate passes or collect downstream physical evidence against an
unadopted execution source. A genuine correctness repair requires a new source,
affected requalification and explicit disposition of earlier evidence.

After A, freeze the existing system with a semantic execution-system tag and
an exact source/binary manifest. The tag does not promise extra kernels,
maximum resource utilization or final performance claims.

## 4. Software Work While Hardware Is Occupied

The isolated preparation branch may complete the census and one optional
host-only complex-envelope batching experiment. Allowed changes are analysis
scripts, focused tests and documentation. No production/native/ABI/transport,
candidate-pool, configuration or fitted-profile change.

Use retained development paths for Stress16 (two repeat layers), HS20 (depth
one), EDC14 and already observed BV18. Select greedy, minimum conventional
FLOPs and minimum peak intermediate; break ties by candidate ID and deduplicate.
Never regenerate candidates or replace a rejected path. Exclude GHZ16, GHZ14
and XOR18 from this work.

Lower at 1D/T8 and 4D/T8 under float32 and shared-scale int8. Preserve the
512 MiB conservative host estimate, 400-work-unit limit and 60-second isolated
lowering timeout. At most 48 census cells. Record logical/physical identities,
B/M/N/K, tiles, K chunks, waves, active/idle slots, aligned-byte estimates,
memory facts, and explicit exclusions. Legal underfilled waves remain useful
host-only census cases; scaling admission is a separate reported fact.

Use the census for explanation and feasibility, never CPU placement. Distinguish
measurements from static estimates and unmeasured quantities. Existing heuristic
`packed_operation_count` aliases waves; report actual lane submissions separately
without silently redefining frozen features.

### Optional Batching Experiment

Compare four lane envelopes against one envelope containing identical ordered
ABI-v4 requests using existing builders and templates. Preserve embedded bytes,
hashes, relative output paths, lane/wave order and numerical encoding. There
remain four real-product launches per wave. This is neither a fused complex
kernel nor proof of reduced DPU execution or host-DPU transfer volume.

Use real planned geometry with deterministic synthetic tensors, both policies
and both topologies, all 16 eligible greedy cells, one warmup and seven measured
paired blocks, seed 20260905. Keep the same filesystem/durability, fresh roots,
one CPU thread and per-arm process memory accounting. Preserve all raw results,
failures and exclusions; no replacement. Report construction, packing, writes,
validation, cleanup, counts and memory with explicit scopes.

Complete this single probe within the existing three-day preparation cap.
Positive, neutral or negative results close it. Do not add another optimization
in response. Production batching adoption would require a separately authorized
physical experiment; it is not a prerequisite for this research roadmap.
See [preparation protocol](implementation/docs/upmem_execution_preparation_v1.md).

## 5. One-Rank Parallelism Study

Keep the contraction path, tensor values and numerical policy fixed while
varying tasklets or DPUs. Reuse the existing six-route diagnostic where suitable:
T1, T2, T4, T8 at one DPU, and 1, 2, 4 DPUs at T8. The repeated 1D/T8 route is one
cell: six routes, one warmup plus five measurements, 36 attempts.

Retain existing T1-T24 build and representative non-power resource correctness
evidence. Additional T16 or larger one-rank DPU counts are optional *measurement
points*, only after real-plan feasibility, usable hardware and a finite attempt
budget are declared. Neither T16 nor 64 DPUs is a required result. Do not create
new tiles/kernels merely to populate more DPUs.

Report kernel and total scaling, parallel efficiency, host/request fractions,
H2D/D2H, wave utilization, partial waves and numerical correctness. Use paired
blocks for uncertainty. Account for session opening and closing as well as
steady execution; report planning separately.

Qubit count alone does not determine matrix geometry or useful DPU occupancy.
Derive work from the actual plan. A rank, DIMM and channel are distinct; do not
infer independent bandwidth or identify the host transfer path as PCIe without
evidence. Small contractions and idle resources are explanatory results.

## 6. Frozen-Candidate Path Study

Use opt_einsum greedy as the fixed reference and the existing deterministic
cotengra candidate generator. Do not implement search or tune candidate
generation against new timing. Freeze circuit definitions, split, seed, budget,
candidate pools and features before physical calibration.

Every candidate is lowered through the existing ContractionDAG and UpmemPlan.
Reject unsupported or infeasible paths explicitly. Score actual physical-plan
consequences using the documented SLR terms and greedy-relative normalization.
Compare only the six-term model and its grouped movement/compute/coordination
projection. Avoid redundant fitted terms and unsupported runtime predictions.

Physically measure a predeclared feature-diverse subset once. Freeze the raw
runtime table, then fit weights offline to geometric-mean cross-cell speedup
with worst-cell behavior as a secondary criterion. Do not execute hardware per
weight vector, add adaptive hardware-search rounds, or enlarge the pool after
seeing timing.

Freeze the selected model, weights and comparison paths before held-out timing.
Use unique greedy, FLOP-best and UPMEM-selected paths at 1D/T8 and 4D/T8, with
one warmup and five measured attempts. Both topologies of a circuit belong to
the same split. Declare exact attempt and elapsed-time ceilings before launch.
This document does not launch or silently replace a frozen campaign.

Stress16, HS20 and EDC14 are proposed development instances; BV18 and EDC16 are
already observed. GHZ16 may be validation. GHZ14 and XOR18 are only eligible
untouched tests after an exposure audit: GHZ14 is size-held-out if GHZ16 was
used, and XOR18 is family-held-out only if no XOR development evidence informed
selection. Do not relabel observed instances as untouched.

Use a new study identity when the accepted integrated execution or protocol
differs from earlier frozen calibration. Preserve earlier candidate/config
files and incident records; do not silently execute the superseded 192-attempt
campaign or combine incompatible physical data.

Report each circuit/topology separately: runtime and uncertainty, movement,
feature contributions, selected-path rank, oracle regret and measurable
candidate-pool headroom. Report grouped-model agreement and path stability.
Do not pool raw runtimes across circuits or equate coefficient precision with
physical precision. Negative generalization completes the study.

Float32 is the primary path profile. Int8 comparisons are separately identified
numerical trade-offs, not equal-quality substitutions: the reported Stress18
8.316% relative L2 error requires explicit qualification. Do not transfer
float32 weights to int8. No additional int8 calibration campaign is required
here. Separate approximation error from the SLR numeric-execution cost term.

## 7. Evidence and Claims

Before each physical stage acquire the private experiment lock; verify exact
clean source, configuration/candidate/profile hashes, host/DPU/init binaries,
rank ownership, CPU affinity, governor, SDK and writable evidence storage.
Only one hardware controller may execute. Check occupancy between blocks.
Do not disturb another user's processes; wait at most 15 minutes, then leave
a ready-to-run handoff. A past occupancy check is not current admission.

Stop at the first failure, unsupported attempt or fallback. Preserve and
retrieve the complete partial stage. No individual retry, replacement or
splicing. A proven infrastructure incident may justify only the separately
authorized complete rerun under a new experiment ID.

Accept a stage only after sorted relative SHA256SUMS, immediate durable
retrieval, checksum verification, canonical evidence inspection, exact
sample/session/schedule checks, an archive with an outer SHA-256, and two
verified copies. Temporary storage and recorded digests alone are not archives.

Keep lost-raw historical path results labeled pilot evidence. Do not reconstruct
observations from summaries or use them as canonical held-out uncertainty.

Report steady execution, session-inclusive execution and planning/preparation
costs distinctly. Kernel scaling is not application scaling; neither establishes
circuit-to-result time-to-solution without that boundary being measured.
Compute planning break-even only when a positive measured saving exists.

Claims concern the tested workloads, paths, policies, resources and diagnostic
environment. Do not claim universal improvement, globally optimal paths,
CPU/GPU competitiveness, full-system acceleration or elimination of host costs.

## 8. Future Research, Not Current Deliverables

The following require a new explicit research decision and are not prerequisites,
optional integration slots or automatic successors within this roadmap:

- Multi-rank execution and concurrent independent DAG branches.
- MRAM intermediate residency.
- Fused complex kernels or a changed DPU ABI.
- Additional kernels, skinny/GEMV dispatch or ATiM integration.
- Parallel slicing and asynchronous SDK overlap.
- New numerical formats, joint path/resource/policy optimization and energy.
- A final 2+30 campaign or cross-platform performance matrix.

Do not remove a single-rank guard or add thread pools as a shortcut to a new
execution contract. CPU/UPMEM placement is excluded, not listed as future work.

## 9. Immediate Actions and Stop

1. Finish the software-only census and optional bounded probe while waiting.
2. Run the unchanged seven-session physical integration gate when admitted.
3. Freeze the qualified existing system; no further optimization is required.
4. Complete bounded one-rank scaling and the frozen-candidate path study.
5. Archive benefits, limitations and negative results; stop.

The former ten-day runtime-expansion allowance is a ceiling, not work to fill.
Preparation remains capped at three focused days. Removed placement/kernel/
residency/multi-rank tasks receive no replacement engineering allocation.
Freeze the remaining physical attempt/time budgets before execution.

Use at most two implementation workers with disjoint files, a bounded read-only
auditor and one exclusive physical controller. Keep implementation branches
isolated until their own gates pass. Do not merge speculative runtime work,
assume hardware success or continue tuning because one workload remains slow.
