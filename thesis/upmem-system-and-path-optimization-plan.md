# UPMEM thesis: execution-system development and path optimization plan

Date: 4 September 2026. Status: proposed plan; no branches merged, source code changed, or physical experiments launched in this planning task.

## Decision

Complete and freeze the execution system before the final contraction-path optimization. The execution system includes kernel dispatch, packing/layout decisions, numerical policies, tasklet/DPU configuration rules, and host work. A path must be scored after lowering through that system, because the same contraction can have very different execution costs under different kernels and placements.

Recommended order:

1. Integrate the released quantization branch with the qualified path infrastructure.
2. Characterize contraction shapes and their actual costs using fixed development paths.
3. Add minimal kernel dispatch and one measured specialization; run a bounded ATiM experiment in parallel with software development.
4. Add a second specialization or explicit CPU/UPMEM placement only if measurements justify it.
5. Freeze the best qualified execution profiles and their cost features.
6. Run bounded, adaptive path optimization with physical ETH feedback.
7. Freeze the resulting path policy, perform independent confirmation and untouched testing, and release the evidence.

“Best” means best observed under a declared workload, resource set, timing scope, and numerical-quality contract. It does not imply maximum DPU/tasklet counts or int8 for every circuit.

The immediate engineering priority is integration, shape attribution, and kernel dispatch. The previous 192-attempt calibration should be postponed and superseded by a new study for the final system. Preserve its frozen files unchanged; record the supersession separately.

## 1. Verified starting point

Live branch heads were checked during this planning task. The local checkout remains clean.

| Role | Branch | Commit |
| --- | --- | --- |
| Accepted packed-transport baseline | `main` | `fa0dedf628a3612371daa4f6502da4d5465bbaff` |
| Released quantization reporting source, unmerged | `feature/upmem-quantized-physical-v1` | `62505ae637bdd3cf963b70f61754bf90a658b527` |
| Qualified path infrastructure, unrun calibration | `feature/upmem-path-heuristic-generalization-v1` | `e225947a84f937629ed46003dad7d8edff160a8f` |

Both feature branches descend from the accepted main. Their common merge base is that main commit. Physical-source and reporting-source distinctions in existing releases remain intact.

The implementation already canonicalizes binary contractions to batched `(B,M,K) @ (B,K,N)` in `src/quantum_bench/upmem/tiling.py`. The active DPU implementation already performs blocked matrix contraction using a shared WRAM B panel, private tasklet buffers, and multi-DPU tile waves. Therefore, this milestone adds better implementations and dispatch to existing matrix lowering.

The current sequential release uses the WRAM kernel. It must not be labeled the old scalar-MRAM naive implementation.

Source inspection: [tiling](https://github.com/kazulak/Masters/blob/e225947a84f937629ed46003dad7d8edff160a8f/thesis/implementation/src/quantum_bench/upmem/tiling.py), [DPU kernel](https://github.com/kazulak/Masters/blob/e225947a84f937629ed46003dad7d8edff160a8f/thesis/implementation/native/upmem/runtime/dpu.c), [physical plan](https://github.com/kazulak/Masters/blob/e225947a84f937629ed46003dad7d8edff160a8f/thesis/implementation/src/quantum_bench/upmem/plan.py).

## 2. Branch and merge sequence

The following names are proposed; they have not been created.

| Phase | Proposed branch | Contents and completion gate |
| --- | --- | --- |
| A. Integrate | `feature/upmem-execution-integration-v1` | Start from verified main; merge quantization reporting head, then path-infrastructure head. Preserve both histories. Complete software, SDK, and a small physical integration matrix before adoption. |
| B. Kernel system | `feature/upmem-kernel-dispatch-v1` | Start from the qualified integration commit. Add shape attribution, concrete dispatch records, one specialization and matched ablations. |
| B, optional spike | `feature/upmem-atim-kernel-probe-v1` | Isolated, timeboxed feasibility experiment. Import only a qualified generated kernel or useful schedule; do not make this branch a prerequisite for B. |
| C. Final path study | `feature/upmem-final-system-path-search-v1` | Start only from the frozen kernel-system tag. New study identity, physical features, profiles and adaptive-search protocol. |

Merge each accepted milestone into main before starting the next production phase. Keep rejected probes separate with a short decision and evidence. Preserve released source commits/tags; do not rebase or rewrite them. Do not develop the standalone repository in parallel.

The four overlapping files between the two existing branches are `native/upmem/runtime/simplepim_provider.c`, `src/quantum_bench/cli.py`, `tests/test_cli_report.py`, and `tests/test_upmem_protocol.py`. Important merge semantics:

- Preserve explicit simulator-backend selection and simulator virtual-rank handling, while physical admission still enforces the requested one-rank topology.
- Preserve persisted failure records and immediate physical collection termination.
- Preserve canonical `complex_int8_shared_scale_v1`, prequantization float64 handling, and the repaired complex accumulation bound `2*K*127^2` where required by the contract.
- Preserve path-specific validation and workload hashing.
- Retain the original float32 study as historical preregistration. Its old physical-plan identities and weights do not qualify the integrated int8 system.

Qualification should cover representative dense, skinny, complex, zero-input and boundary cases; float32/int8; T1 and T8; one and four DPUs; and a non-power-of-two correctness route. Run the pinned full suite and exact-head CI once after integration, then targeted SDK coverage and a small fail-fast physical smoke. Do not repeat the old 180-sample quantization campaign merely because branches were merged.

## 3. Thesis comparison ladder

Use a cumulative development story plus controlled ablations. Historical releases provide chronology; matched experiments establish the cause of speedups.

| Variant | Path | Precision | Resources | Purpose |
| --- | --- | --- | --- | --- |
| Scalar naive contraction | Fixed reference | Float32 | 1 DPU × T1 | Establish a simple arithmetic/memory baseline. |
| Existing WRAM-panel kernel | Same | Float32 | 1 DPU × T1 | Isolate memory staging/blocking. |
| WRAM kernel with tasklets | Same | Float32 | 1 DPU × T8 | Isolate intra-DPU parallelism. |
| WRAM kernel across DPUs | Same | Float32 | 4 DPUs × T8 | Isolate the second parallelism level. |
| Kernel-dispatch system | Same | Float32 | Same frozen route | Isolate geometry/structure specialization. |
| Explicit CPU/UPMEM placement, if justified | Same | Same declared policy | Same allowed resources | Isolate placement. |
| Quantized execution | Same | Shared-scale int8 | Same route | Measure representation cost and error together. |
| Tuned contraction path | Selected path | Same final profile | Same topology or frozen selection rule | Isolate path optimization. |

For the naive comparison, use a small scalar kernel variant under the contemporary packed transport and measurement boundary. Keep it confined to ablation builds or an isolated tag. Do not restore an obsolete runtime or protocol. Where a naive full-circuit run is impractically slow, report a matched smaller instance or kernel microbenchmark; do not invent missing full-circuit timings.

For every kernel comparison, hold DAG/path, tensor values, precision, topology and timing scope fixed. After that comparison, report the cumulative best system separately. Kernel speedup alone does not establish simulation speedup.

## 4. Minimal kernel infrastructure

Extend the current flat plans and direct functions. Avoid a kernel registry, plugin framework, general scheduler or new executor hierarchy.

The core flow is:

`classify contraction → enumerate the few supported implementations → choose using frozen rules → record the physical decision → execute → validate`.

A per-operation decision needs only fields actually used: semantic operation kind, B/M/N/K geometry, layout/permutation requirements, kernel ID, numeric policy, tile parameters, and placement when CPU contractions are implemented. Reuse existing topology and identity records.

| Existing area | Necessary change |
| --- | --- |
| `upmem/tiling.py` | Reuse canonical geometry; expose packing/layout facts and precise specialization predicates. |
| `upmem/plan.py` | Represent per-operation decisions and validate deterministic selection. Include decisions in physical identity. |
| Native protocol, `dpu.c`, runtime | Support a small validated kernel selector and matching completion identity. Keep packed transport and persistent host lifecycle. |
| `upmem/runtime.py` | Emit actual kernel/movement/launch facts; remove assumptions that every operation uses the one WRAM mechanism. |
| `cpu.py` and validation | Replay the chosen arithmetic and reduction policy, including specialized and quantized cases. |
| `upmem/path_heuristic.py` | Extract features from the resulting executed plan, including selected kernels and host work. |
| Experiment/reporting code | Record per-kernel time, placement, bytes, packing, launches, numerical error and complete execution time. |

Prefer one DPU binary containing the small dispatch set if instruction and WRAM capacity allow. Existing sessions assume one DPU binary; loading different binaries for individual contractions introduces costs that must be justified and measured. Make an explicit protocol-version change if the wire semantics require one; keep a single active reader rather than adding compatibility machinery.

Keep `logical_plan_id` stable when the circuit and contraction DAG are unchanged. Kernel, numeric representation, layout, tiling or placement changes must affect physical/executable identity. A future transformation that changes the tensor network or DAG requires a new logical identity too.

The current feature extractor assumes the panel kernel, four real products and particular launch/movement counts. Updating those facts is a correctness requirement for kernel-aware path ranking, not optional reporting polish.

## 5. Which kernels to develop

First obtain a development-workload census: B/M/N/K, batch count, boundary tiles, contiguity, packing/permutation time, real/imaginary product count, launches, reduction cost, WRAM demand and numerical mode. Rank shapes by wall-time contribution and removable cost. Frequent tiny contractions and expensive large contractions are different optimization targets.

Use a fixed greedy path plus a small, predeclared set of alternative development paths for this census so dispatch is not tailored to one narrow geometry distribution. Exclude final test instances from kernel/routing tuning.

| Candidate | Eligibility | Why investigate | Priority |
| --- | --- | --- | --- |
| Generic WRAM GEMM | Any currently supported dense contraction | Required coverage and stable comparison route | Keep |
| GEMV/DOT or skinny GEMM | Degenerate or narrow M/N dimensions | Avoid poorly amortized panels, idle row tasklets and unnecessary setup | First candidate if attribution supports it |
| Outer product | K=1 | Remove generic reduction machinery | Alternative first candidate if common and costly |
| Tuned dense GEMM | Dense, repeatedly costly shapes | Better tile sizes, traversal or WRAM retention of partials | Evaluate against the existing kernel, not against naive alone |
| Permutation/diagonal/sign operation | Structure proven from tensor semantics | Replace dense arithmetic with indexing or elementwise work | Add only if metadata survives lowering cheaply |
| Fused complex contraction | Four real-product boundaries dominate | Potentially reduce repeated packing, transfers or launches | Optional second experiment; new arithmetic/replay checks required |

Develop one new specialization first. Add a second only when the census identifies a separate material bottleneck. Keep the entire retained set small—typically the generic kernel and one or two alternatives.

Do not classify approximate zeros as exact structure. Establish permutation/diagonal predicates from circuit/tensor semantics and preserve label order. A quantized fast path must reproduce the declared quantization/reconstruction boundaries, or receive a distinct numerical policy and validation.

Tune a small grid of tile parameters and T1/T4/T8/T12 or a smaller justified subset on representative development shapes. Confirm winners in full contraction/circuit execution before selecting them. Account for static buffers, tasklet stacks, barriers, DMA alignment/limits, output ownership, and int32 bounds. Retain only variants whose measured benefit justifies their code and WRAM footprint; a useful region-specific win can justify dispatch even if one variant is not fastest everywhere.

## 6. External work: ATiM and PIMutation

ATiM is a public search-based tensor compiler with a modified TVM stack. Its documented artifact environment uses Ubuntu 20.04 and SDK 2021.3.0; compatibility with the reported ETH SDK 0.29.1 is unverified. Its root repository uses Apache-2.0. [Official artifact](https://github.com/SNU-CODElab/atim), [archived artifact](https://zenodo.org/records/15379924).

The paper studies joint host/DPU scheduling and data movement for tensor programs, including matrix-vector and tensor-vector workloads. It does not establish a ready-made complex GEMM implementation for this thesis. [ATiM paper](https://arxiv.org/html/2412.19630v2).

The supplied tuning driver uses int32, while the evaluation support can record generated UPMEM C and can exclude transfer terms in some benchmark settings. Therefore, generated code must be tested through the thesis runtime and timing contract. Shared-scale int8 inputs with int32 accumulation require explicit verification. [Tuning driver](https://github.com/SNU-CODElab/atim/blob/artifact/evaluation/atim_autotune.py), [runtime/evaluation support](https://github.com/SNU-CODElab/atim/blob/artifact/evaluation/base.py).

Proposed ATiM spike: at most one engineering day and 32 physical candidate configurations for one representative GEMV or related supported tensor program. Pin the external source; build in an isolated environment without changing ETH's system SDK. Verify generated C, arithmetic, one-rank admission and compatibility with the existing host boundary. Account for repeated timing attempts separately from the 32 configurations. Stop at the cap and keep the existing/manual specialization if integration fails or full-route benefit is absent. Do not use the artifact's default large tuning campaign.

PIMutation uses direct statevector simulation with gate merging, row swapping and partitioning of separable state representations. Useful inspiration is to exploit exact structure and locality. Its direct gate-evolution implementation is not interchangeable with a TN contraction backend. Public implementation code and a code-reuse license were not verified in this review. [PIMutation paper](https://arxiv.org/html/2503.00668v1).

A tensor permutation/diagonal specialization is a narrow way to test the relevant idea while retaining the TN algorithm. Global gate fusion, statevector partitioning or a different integer representation would change the scientific comparison and should be separate later work. Preserve attribution, exact upstream versions and applicable notices for any incorporated code.

## 7. Meaning of hybrid execution and parallelism

The required hybrid route is **dispatch among kernels**: generic contraction, GEMM/GEMV and selected semantic specializations. It can remain entirely on UPMEM for contractions.

An optional second meaning is **planned CPU/UPMEM placement**. Small contractions may be candidates for a host BLAS path, but include permutation/packing, dispatch, transfers, reconstruction and synchronization when measuring the crossover. Generic CPU BLAS code cannot simply execute on a DPU.

If placement is implemented, add a concrete host-contract stage and a fixed selection rule. Record host contraction count/time and DPU contraction count/time. This is planned heterogeneous execution and must be distinguished from an unsupported-operation fallback. Keep a device-only control.

Ordinary complex64 BLAS does not implement shared-scale int8 arithmetic. Either preserve the same integer policy on CPU, or define an explicit mixed policy and validate it separately. Do not label mixed execution uniformly int8.

The two existing parallelism levels are tasklets within a DPU and tiles distributed across DPUs. Both belong in the final system. Select their useful configurations from measurements; four DPUs or 24 tasklets are not automatically best. Concurrency between independent DAG branches, transfer/compute overlap, and multi-rank execution are additional projects, not necessary consequences of “double parallelism.”

## 8. Freeze the system before path training

Proposed tag: `thesis-upmem-kernel-system-v1`.

Freeze source, dependencies/SDK facts, host/DPU/init binaries, supported kernel set, dispatch thresholds, layouts, numerical rules, resource choices, CPU-thread settings when relevant, cost-feature extraction and measurement scopes. Qualify complete statevector reconstruction and path-specific arithmetic replay.

A deterministic dispatch policy may choose different kernels for different shapes after the freeze. That is expected. Freeze the selection rule and its calibration data; do not retune it during path training.

Keep two explicit profiles:

- Optimized float32, the primary equal-quality path study.
- Quantized int8, a separate policy/profile with accuracy gates or a reported runtime/error frontier.

The reported Stress18 int8 relative L2 error of 8.316% prevents assuming it is an equal-quality replacement. Path changes can change quantization error. Check the output of every proposed path against the declared quality contract before including it as an admissible speed winner. If no defensible error threshold exists, retain float32 as the primary comparison and report quantized trade-offs separately.

A single frozen implementation may support both profiles; they need separate calibrated weights. If both profiles use the same final test instances, freeze both profiles before observing either profile's final test timings. A later int8 study must otherwise declare the earlier exposure and use new untouched instances for a new generalization claim. Any mixed-precision placement rule is a third explicit policy, not an implicit correction.

## 9. Cost model for the final dispatcher

Continue using the SLR terms, but calculate them from the selected physical implementation. Include kernel-dependent compute, actual or defensible estimated H2D/D2H and MRAM/WRAM movement, packing/permutation, launch/wave/barrier counts, host reductions and numerical execution overhead. Include CPU operation/transition costs if placement is enabled.

A small piecewise lookup or regression over kernel and geometry is a reasonable candidate when the original aggregate linear terms cannot distinguish kernels. Start with the grouped movement/compute/coordination model and a small number of justified kernel distinctions. Compare additional complexity using training robustness and validation; do not create a large learned model before data show the need.

Execution cost `C(c,p,q,r)` and numerical error `E(c,p,q)` remain separate. The SLR numerical term represents execution overhead, not approximation error. Changing the dispatcher invalidates old fitted coefficients, even when some logical candidate paths remain reusable.

Measure complete execution to correct approximations. A sum of kernel microbenchmark times alone misses host boundaries and interactions. If data residency is unchanged, retain host-roundtrip execution for this milestone. Introduce persistent intermediate residency only after a measured need and before the final system freeze.

## 10. Bounded physical-feedback path optimization

The intended feedback loop is appropriate: propose paths, run selected proposals on ETH, compare with baseline, and update the model. Use a fixed budget instead of stopping when results happen to look satisfactory.

There are two different optimization variables:

1. Contraction trees/paths, explored through cotengra's search, subtree changes or simulated annealing.
2. Hardware-cost parameters that rank those paths, fit or perturbed offline using accumulated physical observations.

Cotengra already provides simulated annealing and subtree reconfiguration. Reuse its path machinery; qualify the installed version's scoring interface before connecting a whole-plan hardware score. Where direct custom scoring is unsuitable, use it to propose complete paths and rerank them after physical lowering. [Cotengra documentation](https://cotengra.readthedocs.io/en/latest/advanced.html).

Proposed process:

1. Freeze training/validation/test membership, baseline policies, numerical eligibility, seeds, annealing schedule, proposal limits, hardware budget and stopping rules.
2. Measure a small seed set containing greedy, FLOP-best and diverse feasible development paths on the frozen executor.
3. Fit ranking weights offline. Anneal or otherwise search many proposals using the cost model; deduplicate identical resulting executable paths.
4. Select a small batch of promising or informative unmeasured paths. Freeze that round's manifest before its hardware run.
5. Run randomized complete blocks with nearby greedy controls, one warmup and three measured attempts. Validate, retrieve and checksum the entire round.
6. Append observations without replacing older samples. Refit and repeat up to the declared cap.
7. Select the provisional profile using training evidence, check validation once using a declared retain/fallback rule, then freeze the pretest profile.
8. Run the separately budgeted fresh development-confirmation blocks and untouched tests. Confirmation measures the already frozen selection and does not reopen tuning. Do not modify weights, candidates, dispatcher or numerical policy after viewing final test timing.

Hardware measures previously unmeasured executable choices, not every weight vector. Cache observations by source/system profile, workload, path, kernel decisions, precision, topology and timing scope. Fresh drift controls and independent confirmation are still necessary. A model score that selects an unmeasured path remains a prediction until that path is measured.

For actual simulated annealing, freeze a decreasing temperature schedule and an acceptance rule such as accepting an uphill objective change Δ with probability `exp(-Δ/temperature)`. Use this cheaply for offline exploration; keep a separately confirmed physical incumbent. Do not mistake stochastic acceptance under a surrogate for a measured speedup.

### Objective and timing

For circuit/topology cell j, use repeated timings:

`S_j = median_b(T_greedy,j,b / T_selected,j,b)`, where b identifies paired measurement blocks.

Use greedy controls from the same round and paired blocks as each selected path. Do not pool absolute timings across adaptive rounds to infer a gain. Use paired-block resampling for uncertainty, and fresh final confirmation to reassess the selected development paths under a common environment.

Optimize geometric-mean speedup, equivalently the mean log speedup, with worst-cell behavior and numerical eligibility as additional criteria. Declare circuit/family weights so adding many tiny instances does not change the objective accidentally.

Report runtime reduction as `100 * (1 - T_selected/T_greedy)%`, plus speedup. “Best run” means the best configuration selected from repeated development measurements and reevaluated independently; it must not mean the minimum timing sample.

Use the existing session-inclusive execution scope as the primary one-simulation comparison, with a precisely frozen boundary; retain steady execution and its decomposition as secondary metrics. Report circuit construction, path selection, preparation and offline/physical tuning costs separately, and report complete job time when claiming end-to-end simulation improvement. This makes it possible to assess whether preprocessing/search cost is amortized in the intended usage.

Keep both a fixed greedy reference and a small equal-budget target-neutral path-search comparator. If claiming that the UPMEM-aware search method outperforms another search method, match candidate/compute or physical-query budgets; comparison with greedy alone establishes improvement over greedy, not the superiority of the optimizer.

### Concrete starter budget

This budget uses the existing representative split and is a proposed ceiling per numerical profile, not a launch instruction.

| Stage | Design | Maximum attempts |
| --- | --- | ---: |
| Initial training | Stress16, HS20, EDC14 × 2 topologies × 6 paths × 4 attempts | 144 |
| Adaptive training | At most 3 rounds × 3 circuits × 2 topologies × (2 new paths + greedy control) × 4 attempts | 216 |
| Development confirmation | 3 training circuits × 2 topologies × greedy/selected × 6 attempts | 72 |
| Validation | GHZ16 × 2 topologies × 3 roles × 6 attempts | 36 |
| Untouched test | GHZ14 and XOR18 × 2 topologies × 3 roles × 6 attempts | 72 |
| Total ceiling | Deduplicate coincident roles; hardware-time cap also applies | 540 |

Each six-attempt evaluation has one warmup and five measurements; validation/test roles are greedy, FLOP-best and selected. Stop earlier if the prescribed proposal procedure cannot produce new feasible executable choices. Set an elapsed hardware-time cap from integration measurements before this study starts. Do not silently double the budget for int8; complete float32 training first, then justify a separate quantized allocation and freeze both profiles before any shared final test.

For a larger development suite, the same initial-plus-adaptive design costs at most `60 × N_training_circuits × N_topologies` attempts, before validation/test. Freeze the actual named set and budget, rather than extending it in response to disappointing results.

Run the complete supported basic-circuit suite as software correctness coverage and a bounded physical regression/reporting suite. This includes Bell, GHZ, QRNG, BB84, BV, HS, EDC, XOR and stress instances where supported. Tiny/product-state circuits are useful overhead examples; they need not dominate expensive path tuning. If the goal is tuning on every known basic family, designate those timings as development data and reserve new instances. Claim cross-family generalization only when an entire family remains outside all kernel, routing and path tuning.

GHZ14 and XOR18 remain eligible as untouched instances only if their timings have not influenced system development. GHZ14 is same-family size transfer when GHZ16 is validation. XOR18 is a family-held-out test only if no other XOR instance was used for tuning. BV18 and EDC16 are already observed development material.

## 11. ETH execution and evidence

Only one worker controls physical hardware. Before each round, verify the remote source and clean worktree, binaries, private flock, rank ownership/availability, CPU affinity, governor, SDK, writable evidence destination and retrieval route. Wait at most approximately 15 minutes for admission and do not disturb another user. A shell session ending must release resources or leave a clearly owned, managed process; no overlapping campaign controllers.

Simulator runs qualify correctness and geometry only. They cannot supply timing-model coefficients or physical ranks of candidate performance.

Stop on the first failed/unsupported/fallback attempt, preserve and retrieve partial evidence, and classify the cause. No cell replacement or sample splicing. A new adaptive round is intentionally a new experiment. A replacement for a failed round needs an independently established infrastructure cause and a new complete run identity.

Accept each stage only after sorted relative checksums, immediate complete retrieval, checksum verification, canonical evidence verification, exact schedule/sample/session checks, a portable archive with outer SHA-256, and two verified copies. Retain raw tuning observations as well as final medians. Powersave-conditioned results remain diagnostic under that environment.

Report tuning progress, number of distinct measured choices, wall-time/search cost, numerical failures, worst-cell regressions, candidate-pool headroom, selected-path physical rank/regret and final held-out results. Oracle/headroom and physical rank apply to the physically measured candidate subset; they do not describe every generated path unless all were timed. Neutral and negative results complete the bounded study.

ETH access is not configured in this chat's execution environment. This does not block the plan or software work; physical stages require the existing authorized ETH execution environment.

## 12. Effort limits and immediate backlog

Effort estimates exclude hardware queues and access setup.

| Work package | Proposed effort cap | Deliverable |
| --- | --- | --- |
| Branch integration and qualification | 0.5–1 focused day | Integrated checkpoint, merged semantics and small physical smoke when available |
| Shape attribution and first specialization | 1–2 focused days | Cost census, one kernel, dispatcher, fixed-path A/B evidence |
| ATiM feasibility spike | At most 1 day, parallel software work | Go/no-go with emitted-code and full-route evidence |
| Optional second kernel/CPU placement | At most 1 additional day | Only if a distinct measured bottleneck warrants it |
| System qualification and freeze | One bounded hardware session plus reporting | Exact system tag and numerical/resource profiles |
| Physical-feedback path study | Fixed attempt/time budget | Verified training rounds, pretest profile, untouched evaluation |
| Closure | 0.5–1 focused day | Thesis tables, retained raw evidence, report, tag and release |

Do not make the thesis depend on every optional package. A successful minimum result is: qualified integrated numerics, generic kernel plus one useful specialization, both existing parallelism levels, matched ablations, one frozen float32 path study, and honest quantized error/performance results. ATiM can conclude with a documented no-go. CPU placement and a second kernel can be deferred without breaking the thesis narrative.

The first implementation task should be phase A, followed immediately by the shape census. Do not begin physical path annealing before phase B's final system freeze. Export the standalone repository after the accepted integration/kernel/path milestones, with source and evidence provenance.
