# UPMEM-Aware Contraction-Path Heuristic v1

> **Evidence-retention notice:** the two original raw physical archives were
> lost before durable archival. The implementation and compact derived
> artifacts remain available, but the physical results below are historical
> pilot results rather than canonical final thesis evidence. No raw
> observations were reconstructed. The incident is recorded in
> `thesis_results/upmem_path_heuristic_generalization_v1/pilot_evidence_loss.json`.

## Purpose

This milestone tests whether an SLR-derived UPMEM-aware candidate-path score
can select physically faster contraction paths than the conventional
`opt_einsum` greedy path. Cotengra generates complete candidate paths; it is
not replaced or modified. Every candidate is lowered through the canonical
`ContractionDAG` and `UpmemPlan` before it is scored.

The experiment is preregistered in
`configs/upmem_path_heuristic_v1.json`. Circuit splits, candidate-generation
seeds, resource topologies, calibration size, fitting objective, and physical
collection sizes are frozen before candidate timing is observed.

## Cost Model

The score identifier is `upmem_slr_cost_v1`:

```text
C = alpha * B_host_dpu
  + beta  * B_mram_wram
  + gamma * I_dpu
  + delta * N_sync
  + eta   * E_num
  + theta * P_wram
```

Each term is normalized relative to the greedy path for the same circuit and
topology:

```text
z_i = log((x_i + 1) / (x_i_greedy + 1))
```

Weights are non-negative and sum to one. Lower scores are preferred.

Candidate physical lowering has a 60-second generation guard. Exceeding that
guard aborts the complete generation run; elapsed machine load therefore
cannot silently change candidate-pool membership. Only deterministic resource
admission failures are retained as explicit infeasibility facts.

Before materializing tiles, the generator derives their exact count from the
canonical B/M/N/K geometry and fixed tile limits. Candidates requiring more
than 400 planned work units are retained as explicitly infeasible. This keeps
the 371-unit GHZ18 greedy anchor feasible and is approximately 1.8 times the
222-unit accepted Stress18 greedy plan while
bounding the finite calibration
campaign. The time bound is therefore a failure guard rather than the primary admission
rule.

The existing canonical DAG identity recursively embeds escaped semantic
subtrees. Candidate generation therefore rejects paths whose deterministic
semantic-identity expansion estimate exceeds 1,000,000 units before hashing.
This is a software representability bound, not a UPMEM hardware limit.

- `B_host_dpu` is planned H2D plus D2H traffic. The components are retained.
- `B_mram_wram` is the existing aligned-transfer estimate derived from tile
  geometry. It is not a hardware counter.
- `I_dpu` is exact real-MAC work for the four real products used by the
  split-complex float32 route.
- `N_sync` exposes waves, launches, host reductions, and modeled barrier
  events. One packed operation is currently an alias for one wave and is not
  counted a second time.
- `E_num` is inactive in v1 when float32 representation overhead is already
  represented by movement and arithmetic terms.
- WRAM admission is a hard feasibility condition. `P_wram` records only
  existing modeled buffer and tiling facts and is inactive when it does not
  discriminate candidates.

Transfer savings are not also credited to `E_num`; four-real-product work is
not counted twice; and movement caused by WRAM tiling is not automatically
penalized again through `P_wram`.

Conventional intermediate metrics exclude the final output. Contraction-pair
indices are canonicalized because their left/right ordering is not semantic;
the sequence of contraction steps remains identity-bearing.

## Data Separation

Training uses Stress18, HS18, GHZ18, and EDC18. EDC16 is validation-only and
BV18 is held out for the final test. Both resource topologies for one circuit
remain in the same split.

For each training circuit and topology, no more than six paths are physically
calibrated: greedy, minimum FLOPs, minimum peak intermediate, minimum writes,
equal-weight score best, and deterministic feature-diverse candidates needed
to fill coincident selections. The calibration set is selected without
runtime observations.

Weights are fitted offline only against measured calibration paths. Validation
does not authorize retuning. The held-out test is evaluated only after the
weight profile and all model decisions are frozen. The frozen-profile selector
rejects training instances and records that no timing was used for held-out
selection.

## Physical Scope

Physical v1 uses one rank, packed-operation transport, split-complex float32,
and two explicit resource topologies: one DPU with eight tasklets and four
DPUs with eight tasklets each. Simulator timings are never used for fitting.

Calibration used one warmup and three measurements. Validation and test used
one warmup and five measurements for each unique greedy, minimum-FLOP, and
UPMEM-selected path. Compact calibration observations survive, but the
original raw calibration and validation/test archives do not. Paths were
never replaced or automatically retried after a failed physical attempt.

## Claim Boundary

The milestone can report a physically calibrated, interpretable
UPMEM-aware candidate-path heuristic and diagnostic held-out results for the
frozen circuit instances. It cannot claim global path optimality, universal
circuit generality, optimal resource selection, final thesis performance, or
CPU/GPU competitiveness. A negative held-out result is a valid outcome.

## Frozen Dataset And Qualification

The frozen dataset contains 390 complete candidate paths: 65 candidates for
each of Stress18, HS18, GHZ18, EDC18, EDC16, and BV18. After exact DAG and
physical-plan lowering, the numbers with at least one feasible topology were
20, 65, 28, 29, 32, and 26 respectively. The calibration set contains 48
candidate/topology cells representing 25 distinct paths.

The accepted calibration campaign used execution source
`d452106e697890439cfc94fdabfb6a80fb0fccc2` and produced 192/192 successful
samples and physical sessions. Experiment
`424886cb45ecf2f43b304da76c7333b4720fb882066e04025593821f1eba1e04`, run
`8b6ebfbd-1c06-4fb1-b4bf-4d072212375d`, had no failed, unsupported, fallback,
resource-admission, or numerical-validation result. An earlier complete
campaign at `f2e21a0...` is excluded because 24 BV16/4-DPU attempts had an
underfilled dominant wave and failed collection resource admission. No sample
was spliced or retried.

The strict SDK gate then executed four unique validation candidates and two
unique test candidates successfully. The physical validation and test source
was `16704f8221a4aec3fe694668b6bd1383b68fa4fc`. Validation produced 30/30
successful samples and sessions; held-out test produced 24/24. Both used one
warmup and five measured blocks, rank1, CPU affinity 0, powersave, SDK 0.29.1,
packed-operation transport, and diagnostic claim policy.

## Fitted Profile

The six conceptual features were not independently identifiable: `E_num` and
`P_wram` had zero discriminating range, and movement terms were correlated.
The preregistered grouped projection was therefore used. The fitted underlying
weights are:

| Feature | Weight |
| --- | ---: |
| `B_host_dpu` | 0.2707370926 |
| `B_mram_wram` | 0.2707370926 |
| `I_dpu` | 0.4523235890 |
| `N_sync` | 0.0062022258 |
| `E_num` | 0 |
| `P_wram` | 0 |

Across eight training cells, the geometric-mean descriptive speedup was
2.2314x, the worst-cell speedup was 1.0x, seven cells improved, and one was
unchanged. These are fitted calibration results, not held-out evidence.

## Validation And Held-Out Result

Primary comparisons use median `steady_execution_v1` total wall time within
the same circuit and topology. They do not pool circuit runtimes.

| Split | Circuit | Topology | Greedy (s) | FLOP-best (s) | UPMEM-selected (s) | vs greedy | vs FLOP-best |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| validation | EDC16 | 1 DPU/T8 | 1.857156 | 0.662355 | 0.662355 | 2.8039x | 1.0000x |
| validation | EDC16 | 4 DPU/T8 | 1.231882 | 0.663494 | 0.654318 | 1.8827x | 1.0140x |
| test | BV18 | 1 DPU/T8 | 1.343369 | 1.350928 | 1.350928 | 0.9944x | 1.0000x |
| test | BV18 | 4 DPU/T8 | 1.276361 | 1.278622 | 1.278622 | 0.9982x | 1.0000x |

Validation geometric-mean speedup was 2.2976x. The fully held-out BV18
geometric mean was 0.9963x: both cells regressed slightly in steady execution.
The selected BV18 path was also the minimum-FLOP candidate and differed only
minutely from greedy in the normalized features. This is a negative held-out
result and the weights were not changed after observing it.

EDC16 selected different paths at one and four DPUs. In training, HS18 also
selected topology-dependent paths; the other three training circuits did not.
BV18 selected the same path at both topologies.

## Planning Cost

Deterministic single-circuit replay measured EDC16 candidate generation at
2.0424 s and feature extraction at 3.8445 s. Frozen scoring/selection,
including artifact loading and serialization, had a 0.0913 s median. The
steady-time break-even is approximately 5 executions at one DPU and 11 at four
DPUs. BV18 required 2.0483 s generation, 5.8416 s extraction, and 0.0915 s
scoring, but has no finite primary break-even because its held-out steady time
did not improve.

## Interpretation

The heuristic demonstrates that physical-plan movement and work descriptors
can identify substantially better paths for calibration and validation
structures, and that topology can change the selected path. It does not
generalize positively to the held-out BV18 instance. The result supports the
methodology and exposes a generalization limitation; it does not support a
universal UPMEM path-speedup claim or production replacement of greedy.

## Independent Audit

An independent read-only audit of source `fcfb9d22688c71234e78b8f5db4be72dad37e6f2`
found no scientific/correctness blocker, calibration leakage, or physical
integration blocker. It verified training-only measured-candidate fitting,
the geometric-mean then worst-cell objective ordering, timing-independent
hash-bound validation/test selection, and consistency between the reported
positive validation and negative held-out result. The compact audit record is
stored with the milestone results.
