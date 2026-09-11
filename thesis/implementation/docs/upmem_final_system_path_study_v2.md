> **SUPERSEDED RESEARCH RECORD.** This earlier final-system path-study design was not
> the final P6 protocol. It is retained for provenance only. The authoritative completed
> study is `upmem_cost_guided_path_v1.md` plus
> `../thesis_results/upmem_cost_guided_path_v1/`. Do not resume this protocol.
>
# UPMEM Final System Path Study v2

## Superseded Final-Workload Role

The user reclassified this six-instance allocation as development-only on
2026-09-08. Its archived candidate pools, source identities and 92 physical
development attempts remain unchanged. No weights were fitted. The final
benchmark must cover all six PIMutation families; see
[the workload reconciliation](upmem_pimutation_workload_reconciliation_v1.md).
The historical preparation status and intended split labels below are not a
current final-workload freeze or authorization to resume fitting or hardware.

## Historical Preparation Record

Status: `draft_not_frozen`.

This is a bounded preregistration and exposure decision for the final path study. Host-only preparation and admission checks are recorded below; no P6 SDK or physical execution has occurred. The draft remains non-final until lead review, exact source/profile qualification, candidate and physical-plan hashes, and the pretest freeze are recorded.

## Governing contract

The controlling plan is section 11 of `thesis/upmem-system-and-path-optimization-plan-v2.md`. Section 11.3 permits at most three adaptive rounds. That live limit is used here; the earlier 192-attempt configuration is historical and superseded, not an active frozen campaign.

The explicit execution profile is `kernel_schedule_system_v1`, with the accepted executor source `459935f586fdd16c82013838e6d27a12604c3093` and tag `thesis-upmem-kernel-schedule-system-v1`. The frozen contract is:

- `static_dag_waves_v1` scheduling over dependency-ready disjoint DPU groups;
- `packed_wave_v1` request transport;
- `fuse_complex=true`;
- `panel_only_v1` geometry;
- `split_complex_float32_v1` as the primary numeric policy;
- one rank, with `1dpu_t8` and `4dpu_t8` cells;
- host-roundtrip intermediates, the frozen executor default of 512 MiB declared host admission, and zero reserve; this is not an RSS bound and does not add route options;
- `upmem_slr_wave_cost_v1` as the score and feature context.

The six-term feature form is primary and the grouped movement/compute/coordination form is the preregistered simpler comparison. `E_num` and `P_wram` are inactive. No old serial score, pilot weight, profile, or lost raw calibration is imported.

## Workload and exposure

The fixed workload is three training instances, one validation instance, and two proposed test instances:

| Split | Instances | Use |
| --- | --- | --- |
| training | `quantization_stress_16q_l2`, `hs_20q_d1`, `edc_14q` | fitting and adaptive proposal decisions |
| validation | `ghz_chain_16q` | development evidence only |
| test | `ghz_chain_15q`, `xor_17q` | untouched instance-held-out evaluation candidate |

The test choices use the existing built-in generators with new size parameters. The bounded local audit found the old `ghz_chain_14q` and `xor_18q` in completed SDK-simulator correctness packets. It found no matching path-study physical timing packet in the checked local primary/active/P2 metadata roots, but correctness-observed is not proof of optimization-unseen. Therefore those old sizes are not accepted as untouched tests. A subsequent read-only ETH scan checked 42 unpacked manifests under `/home/tkazulak/evidence` and `/home/tkazulak/work`, matching circuit definitions as well as case aliases, and found no GHZ15 or XOR17 records. No timing values were emitted and no DPUs were allocated. Deleted, unavailable, and archive-only material remains outside this bounded audit; it is not a global certification.

`bv_18q` is not silently treated as an untouched replacement: the fixed prior development allocation already assigns it to pilot development. The new finite allocation has no slot for it and does not refill after that allocation decision. This is a budget/exposure decision, not a claim that `bv_18q` is training-ineligible or an arbitrary post-timing exclusion.

No family-held-out claim is made. The holdout unit is the instance. A family-held-out interpretation is allowed only when that family has never influenced development, including candidate or profile decisions.

## Diagnostic admission correction

The first host-only generation attempt at preparation source
`95c4a70e56c8192ae1b4f4e77ece87e1b6677e61` stopped before writing candidate
artifacts because the EDC14 greedy path failed tasklet-scaling eligibility at
four DPUs: its dominant wave had output-row counts `(8, 8, 8, 4)` for T8.
Declared memory admission passed. This was not an observed numerical failure.
No P6 physical observations existed, and no candidate pool had been frozen.

The private P6 preparation layer incorrectly treated collection/scaling
eligibility as execution feasibility even though its collection policy is
`diagnostic_v1`. The frozen executor distinguishes them: idle tasklets are
permitted for correctness, and the CLI enforces dominant-wave row/occupancy
requirements only for `physical_performance_v1`. Filtering those paths would
remove path-dependent underutilization from the fixed-topology study.

Retain EDC14 and the existing workload, generator, seed, and budget. The bounded
repair is confined to private preparation, qualification, and analysis code:
retain collection eligibility and utilization as explicit diagnostic facts,
including false values, rather than requiring all paths to qualify for scaling
claims. Missing, malformed, or inconsistent facts are errors. Actual plan,
memory, allocation, active-resource, binary, execution, replay, and numerical
checks remain mandatory. The frozen runtime, feature definitions, public
evidence schemas, and physical-performance admission rules are unchanged.
Diagnostic results remain ineligible for generic performance claims.

In particular, preparation must predict the frozen execution-resource check:
the union of non-idle planned DPU slots over all contraction waves must cover
the requested topology. An underfilled dominant wave is not the same as a DPU
that never executes any work. Plans failing all-wave coverage are explicitly
infeasible before collection; qualification recomputes that coverage from the
selected plan. The tiny Bell four-DPU fixture has only two active slots across
all waves and remains inadmissible for this reason, not for tasklet occupancy.

Two prospective geometry-only checks (EDC13 and EDC16) also failed four-DPU
tasklet-scaling eligibility. Neither replaces EDC14. Their diagnostic records
are retained for traceability, not imported as candidate pools or timing data.
There will be no further size search to force collection eligibility.

Retained preparation artifacts under `runs/p6-preparation/`:

| Artifact | SHA-256 |
| --- | --- |
| `candidate-generation-95c4a70-v1.log` | `d2a36ef94ba2be8f032550e572a57acfdf456f3b58a21de14e51ffcf836e56b7` |
| `edc14-greedy-admission-diagnostic.json` | `b7d4609dc4369bc55319be248663e8c040ec6eafd450c86a16eb3c32c133c5e9` |
| `edc13-edc16-greedy-admission-diagnostic.json` | `0b437278d883711c8c5c3fefb3657afe15ae3d840c702f428f4fb521e8d79ca7` |

The corrected preparation source requires new software qualification and new
artifact hashes before generation is resumed. This does not invalidate any P6
physical data: none has been collected.

## Candidate pool and proposal strategy

Candidate generation reuses the existing field pattern and cotengra strategy exactly: `cotengra_method=greedy`, `cotengra_objective=flops`, `master_seed=20260902`, `one_trial_searches=64`, `maximum_planned_work_units=400`, `maximum_semantic_identity_expansion_units=1000000`, `opt_einsum_reference=greedy`, and the existing 60-second physical-lowering admission timeout. The failed preparation attempt above did not produce a frozen pool.

The candidate pool is finite and fixed before the first timing observation in each training cell: one opt_einsum greedy reference plus 64 cotengra one-trial candidates, followed by feasibility checks and physical-choice deduplication. The initial six-role calibration set is selected without timing. Its diversity role is one feasible candidate farthest from greedy in normalized eligible feature space, with candidate path ID breaking ties. Roles can collapse to the same physical choice; the resulting unused slots stay unused, without an iterative refill to six candidates.

The adaptive proposal strategy is separate from pool construction. Across at most three training-only rounds, each cell has up to three unique roles: the greedy control, the current training incumbent selected by the primary aggregate model with its worst-cell tie-break, and one as-yet-unmeasured feasible member selected from the already fixed pool by the lowest current training-fitted score, with that score and ordering frozen before the batch and candidate path ID as the tie-break. The initial feature-model score is computed and frozen before the initial timing batch. Previous training measurements may determine which fixed-pool member fills the `new_fixed_pool_candidate` role, but no new surrogate model is introduced. The process may not create a new candidate set, use validation/test timing, choose the fastest raw observation, or refill a deduplicated slot. Same-round controls are retained.

There are two distinct freeze points. Before any physical execution, the candidate generator, fixed pool, execution/config contract, workload splits, finite budgets, and source/environment inputs are frozen. This initial candidate/config freeze is not a pretest freeze. Only after initial and adaptive training are physically complete is the pretest profile, fitted weights, model form, and heldout path-role mapping frozen. Development confirmation then evaluates that frozen pretest selection on the training instances without retuning, followed by validation and test sessions. No pretest profile is frozen before the first physical execution.

## Timing and fitting

The primary optimization quantity is `session_inclusive_s`, derived from the existing evidence scope `scope_id=steady_execution_v1` by the preregistered formula `session_inclusive_s = session_open_s + steady_execution_s + session_close_s`. The steady interval is enclosed by fresh session open/allocation and session release; runtime integrity checks and hashing performed inside the session remain included. Candidate generation, lowering, external reference computation, external correctness comparison, and external manifest/report/evidence hashing and writing are outside execution timing. A steady execution side view is reported separately. A full-job side view may be reported only if independently measured; session-inclusive execution is not full circuit-to-result time. No new public timing scope or evidence schema is introduced.

The initial-stage wave path-study adapter now derives the primary quantity
from the existing open, steady, and close evidence fields while preserving raw
`scope_id=steady_execution_v1`. It accepts the explicit frozen wave contract
and produces a profile validated by the evaluation adapter. The historical
serial fitting path remains separate. Training-only instance cross-validation
has focused software coverage; adaptive multi-round integration remains under
review. These software checks do not establish a physical fitting result or
authorize a campaign before the remaining gates.

The host memory value is the frozen executor's declared admission budget, not a process RSS limit. The draft does not introduce an alternate runtime memory policy or route option; the lead must check physical-plan identities against the actual wave lowerer.

The primary score uses equal cell weights, `pi_j = 1 / number_of_preregistered_cells`, and a geometric mean of preregistered repeated-block speedups. The worst-cell repeated-block speedup is the tie-break. Raw times are not pooled across cells. Weight vectors are explored offline; hardware is not run once per weight vector. Correlation rejection, normalization, six-term versus grouped comparison, and numerical eligibility are fixed before test. There is no positive-chasing rule.

## Model choice before confirmation

Compare the two model forms using training-only leave-one-circuit-instance-out
folds, keeping both topologies together. Fit each fold on the other training
instances and select only among measured candidates in the omitted instance.
Retain every omitted-cell prediction before comparing forms. Bootstrap complete
paired blocks within each physical round, with 2,000 resamples, seed 20260904,
and percentile 95% intervals; do not independently resample candidates and
their greedy controls. Use the same block draw for all cells sharing an
experimental round, preserving their common collection conditions. Raw
per-candidate medians and MADs remain descriptive. For adaptive-round ranking,
regret, and headroom, use greedy-normalized paired execution ratios so a path
measured only in a faster or slower round is not ranked by that drift alone.
The measured-pool oracle is the lowest paired normalized time, not a global
path optimum. Captured headroom is left null when the oracle's log speedup
does not exceed its raw paired-log MAD or the numerical safeguard of `1e-12`
in log-speedup units.
This is a descriptive denominator guard, not a statistical significance or
equivalence test; the paired bootstrap intervals are reported separately.

The cross-validation comparison bootstrap conditions on the original fitted
fold profiles and selected paths. It describes measurement uncertainty for
those decisions, not the uncertainty of repeating the entire fitting and
adaptive-selection procedure. A separately identified, budgeted refit stability
analysis reports its resample and weight-search budgets explicitly; it does not
replace the preregistered model-choice rule.
This auxiliary analysis uses 200 resamples, 1,000 sampled weight vectors per
fold/model fit, bootstrap seed 20260905, and the existing fitting seed. Its
reduced search budget is reported alongside the main 100,000-vector fits.

Adaptive proposal decisions use the training data before these folds are fit.
The observed candidate pools therefore are not independently selected within
each omitted-instance fold. Cross-validation here is a conditional development
diagnostic and model-complexity decision, not an unbiased generalization
estimate for the adaptive search procedure. Only the separately frozen,
untouched instance evaluation supplies the final holdout comparison.

Choose the six-term form only when the lower interval bound of its
cross-validated log geometric speedup relative to the grouped form is strictly
positive and its observed worst-cell speedup is no lower. Otherwise choose the
grouped form. Report path-selection agreement and the uncertainty regardless
of the decision. An interval containing zero is not proof of equivalence:
choosing grouped in that case is a conservative complexity decision. Refit the
chosen form on all training observations, then freeze it before confirmation,
validation, or test timing. Neither heldout split may influence this decision.

## Finite budget

All attempt counts include the stated warmup and measurement blocks. A failed or incomplete attempt consumes its slot. The total ceiling is 540:

Physical stages also have a cumulative monotonic elapsed-time ceiling of
86,400 seconds (24 hours). This includes collection and runtime overhead while
a stage runs, but excludes occupancy waits, offline analysis, and archival
verification. The single hardware controller records stage durations and
enforces the remaining budget. On expiry, collection stops, partial evidence
is preserved and retrieved, and resources are released. The cap is a research
budget, not a performance threshold or permission to replace missing samples.

| Stage | Formula | Attempts |
| --- | --- | ---: |
| initial training | `3 * 2 * 6 * (1 + 3)` | 144 |
| adaptive training | `3 rounds * 3 * 2 * 3 * (1 + 3)` | 216 |
| development confirmation | `3 * 2 * 2 * (1 + 5)` | 72 |
| validation | `1 * 2 * 3 * (1 + 5)` | 36 |
| untouched test | `2 * 2 * 3 * (1 + 5)` | 72 |

Initial and adaptive training use the three training instances and both topologies. Development confirmation compares `greedy` with the pretest selection frozen after training. Validation and test use `greedy`, `minimum_flops`, and `pretest_selected`. Candidate and config hashes are frozen before initial physical execution; the pretest selection, model form, and weights are frozen after training and before development confirmation or any heldout session. Validation is not a proposal source; it cannot change candidates or the model under this draft.

Deduplication can reduce the number of runnable unique paths. Unused attempts are not reassigned, and no timing result causes budget refill. The maximum number of adaptive rounds is three, even if an earlier round is informative or a later round appears promising.

The qualifier's `prepare --mode confirmation --split training --profile ...`
prepares the independent development-confirmation packet. It recomputes the
selected path over the profile's measured training pool and includes only
greedy and that selection, deduplicating coincident paths. Physical collection
is one warmup plus five measurements; `--execution-target sdk` prepares a
zero-warmup, one-measurement correctness packet instead. Both topologies are
retained. Preparation is not admission: the pretest profile and its evidence
must be frozen and verified before this packet is executed.

## Qualification and evidence obligations

Before any physical execution or claim, lead review must accept this draft; the exact executor, runtime, SDK, and environment must be qualified; CPU/software and strict SDK correctness checks must pass; candidates must be lowered and hashed; and the frozen candidate/configuration manifest must retain verified primary and secondary copies. The fitted pretest manifest is frozen after training and before development confirmation or heldout execution, also with two verified copies. Runtime identity must include the accepted executor source, runtime file hashes, the SDK binary-manifest hash and SDK raw-archive hash as separate identities, environment hashes, candidate and physical-plan hashes, experiment/run/session/sample/manifest identities, and attempt accounting. Runtime integrity hashing performed inside a session remains part of the primary execution boundary; only external evidence hashing is excluded.

The audit records metadata only. Old SDK records are historical correctness/exposure evidence and are not physical path optimization evidence. No old pilot weight/profile or lost raw calibration is mixed into the new fit. Before the gates are complete, the only permitted statements are about this preregistration, source lineage, and the bounded exposure audit.

The runner's manifest contains the normalized experiment configuration, not
the private path-study sidecar. Each physical packet therefore freezes
`preregistration/physical.yml` and `physical.yml.provenance.json` with the
candidate, calibration, profile (when applicable), and preregistration hashes
before launch. The sidecar binds the written YAML bytes and the complete
normalized `load_experiment_config` mapping. Extraction compares that normalized
mapping with the raw manifest and checks the full stage tuple and artifact
identities against the archived packet. No stage field is injected into the
public experiment or evidence schema. Checksums bind content; the controller's
prelaunch freeze establishes the temporal gate, not the sidecar alone.

Wave preparation freezes absolute binary and session paths before hashing.
For physical packets, `prepare --execution-root` must identify the absolute
`thesis/implementation` directory of the accepted `459935f...` executor, not
the reporting checkout. The default is the local implementation directory for
software preparation. Deployment paths, binary hashes, and source identity
are checked together before launch. Relocating the evidence archive must not
change the normalized configuration; no path fields are discarded to make
hash comparisons pass.

## Remaining decisions

The bounded independent read-only review of the fitter and analyzer found no
mathematical blockers in cell/round/block pairing, shared-round resampling,
instance folds, or the declared model-choice rule; its focused run passed 36
tests. This software review does not qualify physical evidence. The separate
integration review identified private-packet identity and split-admission
defects, whose repairs still require the final integrated software gate.

Lead review accepts GHZ15 and XOR17 as the proposed instance holdouts on the available local and remote metadata evidence. Remaining gates are qualification of the session-inclusive implementation, candidate hashes and eligibility, and the post-training pretest freeze before development confirmation and heldout execution. The bounded audit does not certify deleted, unavailable, or archive-only evidence. Its local and remote records are retained under `runs/p6-preparation/`.
