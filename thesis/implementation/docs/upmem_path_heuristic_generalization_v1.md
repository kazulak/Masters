# UPMEM Path Heuristic Generalization v1

## Scope

This milestone replaces the physically promising but incompletely retained
UPMEM path-heuristic v1 pilot with a new canonical study. The two original raw
pilot archives were lost before durable archival and no observations were
reconstructed. The incident and surviving pilot material are recorded in
`thesis_results/upmem_path_heuristic_generalization_v1/pilot_evidence_loss.json`.
Pilot medians and fitted weights may motivate this study, but they are not
included in its statistical fit or final held-out evidence.

The frozen scientific contracts remain unchanged: `TensorNetwork`,
`ContractionDAG`, `UpmemPlan`, ABI-v4, the WRAM-panel kernel, packed-operation
transport, one-rank execution, and `split_complex_float32_v1`.

## Frozen Workload

The complete declaration is
`configs/upmem_path_thesis_workload_v1.json`; its materialized audit manifest
is under `thesis_results/upmem_path_heuristic_generalization_v1/workload/`.
The new physically eligible instances are:

| Split | Circuit instances |
| --- | --- |
| training | `quantization_stress_16q_l2`, `hs_20q_d1`, `edc_14q` |
| validation | `ghz_chain_16q` |
| untouched test | `ghz_chain_14q`, `xor_18q` |

EDC16 and BV18 were observed in the pilot and are therefore not reused as
untouched tests. Bell2, QRNG18, and BB8418 remain correctness-only workloads.

Both physical topologies are fixed at T8 on one rank: one DPU and four DPUs.
The calibration schedule is one warmup plus three measured blocks. Final
untouched evaluation uses one warmup plus five measured blocks.

## Frozen Candidates

Candidate generation uses the same deterministic opt_einsum greedy reference
and cotengra strategy as v1. Each new instance contains 65 complete paths
(greedy plus 64 cotengra candidates) before physical feasibility filtering.
The feasible counts at both topologies are 8, 65, 7, 19, 5, and 27 in the
workload order above.

The calibration set contains 48 candidate/topology cells: at most six paths
for each of four development circuits and two topologies. Its roles are
greedy, minimum FLOPs, minimum peak intermediate, minimum writes, the path
selected by the frozen pilot profile, and one feature-diverse path. Coincident
roles are deduplicated without substituting timing-selected candidates.

Candidate-generation source:
`de27ced63d601b58c5905d8f95739a626afcea42`.

Key canonical hashes:

```text
candidate_paths.json:
  d95150ddf89f6aafa861000b0db2d8447d64456a035c5404463a878c3a319049
calibration_candidate_set.json:
  9be4f054764508a771ad77310de0d8184da694181fb6037297a2ef109d4b9c90
candidate_pool_hashes.json:
  8ce9b7957c2b6ea7895c146cf22a7fbc56c5dd11e6c01ebbb9f093bfb5d3f0d0
frozen pilot selection profile:
  cc1e3deb6b5a227b4efe9c84e43679d385cb9b65da76e293ad0d074889cb868a
```

## Model Comparison

The analysis compares only two nonnegative simplex models over the same
candidate and timing tables:

1. An identifiable subset of the six SLR terms.
2. The grouped movement, compute, and coordination projection.

Zero-range terms are excluded from fitting. Strongly correlated or
rank-redundant terms are removed deterministically rather than receiving
arbitrary fitted weight. Model selection uses leave-one-family-out results,
paired-block bootstrap refits, worst-cell behavior, and path-selection
agreement. The grouped form wins ties.

The analyzer validates all source, split, candidate, logical-plan,
physical-plan, topology, resource, transport, and output identities. It
calculates medians, raw MAD, extrema, bootstrap intervals, score/runtime rank
correlations, selected-path rank, oracle regret, and captured headroom. The
pretest profile is emitted only from training and validation observations and
records that final-test timing was not used.

## Qualification State

CPU same-DAG replay has passed for 20 representative candidates. The strict
SDK simulator matrix passed 15/15 samples and sessions under corrected source
`08803d0473c6a46152619298a4c656f58c58c5b4`; its evidence is checksummed and
retained in the ignored durable runs tree. An earlier 15-case SDK attempt is
preserved as an excluded simulator-allocation incident.

The focused heuristic, qualifier, and generalization analysis suite passes.
The local full suite reports 923 passed and 18 failures solely because Quimb
and cotengra are absent from that local environment. Exact-head hosted CI in
the complete pinned environment is therefore a mandatory pre-physical gate.

No physical calibration for this generalization study has started. The
planned calibration is 192 attempts: 48 candidate/topology cells times one
warmup and three measured blocks. Hardware was occupied by another user for
the bounded 15-minute admission window, so no partial campaign was launched.

## Evidence Retention

A physical stage is accepted only after remote relative checksums exist, the
complete stage is retrieved into the durable ignored runs tree, local checksum
verification and canonical evidence verification pass, and a second verified
copy exists. Temporary remote evidence is not deleted before these gates.

## Claim Boundary

Before new physical calibration and untouched testing, this branch supports
only deterministic software and SDK-correctness claims. It does not yet
support a new runtime, generalization, fitted-weight, held-out, or production
path-selection claim. A neutral or negative final result completes the study.
