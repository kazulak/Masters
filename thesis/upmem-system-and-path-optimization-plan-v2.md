# UPMEM thesis: kernel and hierarchical-parallel execution research plan

> **Active phase, 9 September 2026:** the execution-system freeze is accepted and complete. Sections 1-10 retain historical system-development context; they are not a new execution queue. The replacement Section 11 controls the remaining cost-guided path study and supersedes earlier fixed-pool, score, normalization, fitting and budget instructions. This amendment is planning only.

**Revision:** v2 final-phase cost-guided amendment, 9 September 2026.
**Purpose:** replacement roadmap for the stalled execution-system/path-optimization goal.  
**Project:** deterministic, full pre-measurement statevector simulation through exact, untruncated tensor-network contraction on physical UPMEM.  
**Execution status:** planning only. This document does not merge branches, modify the remote repository, or authorize an unbounded physical campaign.

## 1. Decision and change from the previous plan

**Develop and evaluate a small UPMEM kernel portfolio and genuine contraction-DAG parallelism before freezing the executor and calibrating the final path selector.** Preserve the existing tasklet and multi-DPU tile parallelism. Investigate resident subgraphs and exact slicing as a separate, bounded extension; do not assume slicing is always superior.

The attached 4 September plan already proposed kernel dispatch, one specialization, and an ATiM probe. The subsequently quoted narrowed roadmap instead excludes additional kernels and DAG concurrency. This revision explicitly restores kernel work, adds a static DAG-frontier execution mechanism, and removes CPU/GPU contraction placement from the proposed scope. It is not a claim that the current implementation is inadequate or that prior milestones must be repeated.

The final research sequence is:

`reconcile existing integration → shape/frontier attribution → kernel portfolio + static DAG waves → fixed-path physical comparisons → bounded residency/slicing investigation → freeze selected executor → bounded path search → untouched evaluation → release`

**Required implementation deliverables:** a qualified integrated executor; minimal explicit kernel dispatch; at least one new kernel implementation tested against the existing kernel; a static dependency-ready DAG-wave mechanism; schedule-aware cost extraction; and matched physical evaluations. A retained optimization must earn its place through measurements. A qualified prototype with a neutral or negative result is a valid research outcome, not permission to tune indefinitely.

**Separate extension deliverable:** a bounded feasibility decision for resident subgraphs and slice concurrency, with a small implementation/physical test when the stated gates permit. This is planned research, not an assertion that an autonomous resident slice engine already exists.

## 2. Evidence basis and verification limits

### 2.1 What the provided material establishes

The attachment and handoff report a functioning circuit → TN → ContractionDAG → UpmemPlan route; packed transport; WRAM-panel contractions; tasklet and one-rank tile parallelism; released float32/int8 work; and qualified path infrastructure. They distinguish numerical qualification from execution success and physical diagnostics from broad performance claims. These are the starting research assets, not work to recreate. [S1]

Historical reference points supplied by the user:

| Role | Reported source |
|---|---|
| Accepted packed-transport main | `fa0dedf628a3612371daa4f6502da4d5465bbaff` |
| Quantization reporting head | `62505ae637bdd3cf963b70f61754bf90a658b527` |
| Qualified path-generalization head | `e225947a84f937629ed46003dad7d8edff160a8f` |

Later supplied messages report an integrated executor, a software census, a pending seven-session physical gate, and a host-only complex-envelope prototype. Their exact current source commits are not established by the attachment. Do not rerun completed integration or regenerate an existing census without checking its source and coverage first.

### 2.2 What was independently accessible in this review

The public implementation README was readable and describes the logical/physical split and four-real-product execution. The live branch list, proposed integration branch, and several individual source pages could not be retrieved reliably. A local `git ls-remote` attempt failed because the execution environment could not resolve GitHub. Therefore this document does **not** certify current remote SHAs, worktree cleanliness, integration status, CI, release digests, ETH occupancy, or binary qualification. Those are P0 prerequisites. [S2]

The separate Julia study's raw observations, timing code, and claimed 8,157-sample results were not independently inspected. Its CPU overhead percentages are treated as user-reported observations, not as established UPMEM measurements or universal TN properties. No peer-review status is inferred from a repository document title.

The original SLR itself is not attached to this request. Its cost abstraction is used as represented in the supplied handoff and plan; this review does not claim to re-audit the full SLR.

## 3. Research corrections that affect implementation

**Distributed contractions need not use Cannon's algorithm.** PrIM distributes GEMV rows across DPUs and replicates the input vector; each DPU owns an output segment. This is an explicit counterexample to the blanket assertion that intra-contraction parallelism requires continuous inter-DPU exchanges. [S3]

For a dense contraction, output ownership can similarly use

\[
C[I_d,J_d]=\sum_K A[I_d,K]B[K,J_d].
\]

Each owner receives the necessary operand panels and computes its assigned output. This has input-distribution and gathering costs, but no mathematical requirement for DPU-to-DPU rotation. Splitting the reduction dimension is a different choice and requires a declared partial-sum reduction. Preserve and measure the current mapping rather than replacing it on the basis of a supposed impossibility.

**CPU allocation/permutation overhead is not a DPU roofline result.** The PrIM characterization finds that instruction throughput can constrain DPU execution even at low operational intensity. A CPU-side percentage cannot identify the bottleneck of a different executor, arithmetic format, or memory boundary. [S3] Profile actual allocation, packing, transfer, DPU instructions, local memory movement, and coordination separately. Do not classify by output element count alone: B/M/N/K, layout, numerical policy, and available concurrency matter.

**Slicing preserves open outputs unless they are explicitly sliced too.** Cotengra distinguishes summing internally sliced indices from stacking output slices. It also documents the extra work that slicing can introduce. [S6, S7] For the thesis query,

\[
\psi_{\mathbf{x}}=\sum_{\mathbf{s}}\psi^{(\mathbf{s})}_{\mathbf{x}},
\]

not, in general, a sum of scalar amplitudes. Output slicing produces chunks to assemble; internal slicing produces contributions to sum. Quantum amplitudes are generally complex. For example, a complex64 statevector at 18 qubits occupies `8 × 2^18 = 2 MiB`; returning one such vector for each of 64 internal slices would produce 128 MiB of output traffic before further optimizations. Local accumulation or output chunking can change that traffic, and must be represented explicitly.

**Memory feasibility is a liveness calculation.** A bound on the largest individual tensor is not a bound on simultaneously live inputs, outputs, temporaries, packed copies, and resident metadata. Test peak live MRAM and separate WRAM/IRAM requirements. Do not promise residency merely because a slicing API reaches a target tensor size.

**Q1.31 is not a free single-cycle replacement.** UPMEM Unleashed distinguishes native byte multiplication from full-width integer multiplication and documents compiler-generated slow paths for some int8 code. [S4] Normalization of the final state does not establish a safe accumulator bound for every contraction. Keep the released shared-scale policy; do not introduce a new numerical format in this milestone.

**UPMEM-focused does not mean host-free.** The host still manages execution, transfers, and declared reconstruction/reductions. ATiM explicitly includes this host work in its model of UPMEM execution. [S5] Excluding CPU/GPU contraction placement does not require hiding host costs or inventing a native inter-DPU reduction tree.

**Do not assume free overlap.** An asynchronous host launch and a tasklet's MRAM access are different mechanisms. Verify the installed SDK and memory-ownership restrictions before attempting overlap. Shared tasklet buffers and barriers do not disappear when different contractions are assigned to tasklets. No latency-hiding or aggregate-bandwidth claim is a prerequisite or promised outcome here.

## 4. Scope and research questions

The primary execution route remains UPMEM contraction work with host planning, orchestration, and explicitly accounted reconstruction/reduction. No CPU/GPU placement threshold, automatic backend selection, shots, noise, MPS/PEPS truncation, multi-rank campaign, or energy study is added.

The execution study asks:

1. **Kernels:** Which contraction geometries or repeated execution boundaries justify a specialized UPMEM implementation?
2. **Parallelism:** When does combining independent DAG nodes on disjoint DPU groups improve execution compared with assigning the same total resources to one contraction at a time?
3. **Locality and slicing:** When can a bounded resident segment or exact sliced decomposition reduce host interaction enough to offset added work, storage, and reconstruction?
4. **Planning:** Can a cost model based on the selected kernels and physical schedule choose faster paths on the frozen executor?

These supplement the existing feasibility, hierarchical-scaling, transport, and numerical-policy results. Do not discard previously released negative results or rewrite their original interpretation.

## 5. Planned execution mechanisms

| Mechanism | Status in this revision | First implementation boundary |
|---|---|---|
| Tasklets cooperate on one contraction | Retain and tune selectively | Existing WRAM-panel route |
| Multiple DPUs own tiles of one contraction | Retain | Existing one-rank mapping and reductions |
| Multiple independent DAG contractions run concurrently | **Required new prototype** | Static dependency-ready waves; disjoint DPU groups; one native controller |
| Small independent tile/task lists per DPU | Conditional refinement | Only when one-item waves leave measured avoidable launch overhead |
| Concurrent exact slice branches | **Planned bounded extension** | Reuse wave machinery; preserve full-statevector output and reductions |
| MRAM-resident consecutive contractions/subtrees | **Planned bounded extension** | One statically planned local segment, no general graph runtime |
| Different DAG nodes on tasklets within one DPU | Defer from core | Requires separate WRAM ownership and synchronization design |
| Host transfer/kernel overlap | Defer from core | A later SDK-qualified experiment, not a consequence of static waves |
| Producer/consumer DMA tasklets or double buffering | Conditional kernel probe only | Must beat ordinary tasklet execution under WRAM constraints |
| More DPUs within the same rank | Measurement extension | After 1/2/4-DPU correctness; no new claim of multi-rank support |
| Multi-rank execution | Outside this revision | Separate resource and communication milestone |

### 5.1 Static DAG-frontier waves

Introduce a small deterministic scheduling function over the existing DAG and physical work units, not a generic scheduling framework.

A frontier contains nodes whose inputs are already available. Build a wave from independent ready nodes, assign disjoint DPU groups or slots, and publish a node's output only after all its tiles and declared reductions complete. At four DPUs, a wave might allocate two DPUs to contraction A and two to independent contraction B. At another frontier, one large contraction can use all four DPUs. Tasklets still cooperate inside each participating DPU.

Use a single coordinated launch over the admitted DPU set, with a validated per-DPU descriptor identifying its operation and geometry. A synchronous host launch can still execute different nodes concurrently on different DPUs. Host Python threads and overlapping independent benchmark processes are not required and must not be used as a shortcut.

The conservative first version groups compatible numerical modes and kernel families, uses deterministic tie-breaking, and keeps current host-roundtrip boundaries between waves. Account for common transfer-size requirements and padding in the installed SDK. Inactive DPUs receive an explicit validated no-work descriptor when required by the launch protocol.

This changes **when** independent nodes run, not which tensors are contracted or the within-node numerical reduction order. The logical DAG identity therefore remains stable; the physical schedule identity changes. Batch formation can increase the live tensor set and padding; those costs belong in the comparison.

A long dependency chain has little frontier parallelism. Splitting resources between ready nodes can also lose to running each node with the full DPU set. Both are expected scientific cases. Keep a serial-node policy as a controlled scheduling mode within the same executor.

### 5.2 Exact slicing and resident subgraphs

Slicing is a separate graph decomposition, not a synonym for scheduling independent nodes of the original DAG. It may expose additional parallelism when the original tree has a narrow critical path. Cotengra supplies the decomposition machinery; it does not supply this project's native DPU-resident executor. [S6–S8]

Two implementation levels must remain distinct:

- **Host-mediated slice concurrency:** lower supported slice branches into the existing DAG/wave route. Intermediates still return to the host where that route requires it. This is a concurrency experiment, not a residency claim.
- **Resident segment/slice execution:** upload a bounded instruction list and its inputs; keep eligible intermediate tensors in one DPU's MRAM; return only segment boundaries or final chunk/contribution outputs. This requires explicit memory planning and compatible producer/consumer layouts.

First attempt residency for one short, statically known segment, preferably float32. A flat arena, offsets, use counts, and compile-time/direct operation dispatch are sufficient. Do not build an allocator service, graph interpreter with arbitrary callbacks, or distributed object store.

Resident int8 execution is not automatic. The existing policy derives a shared scale from the complete logical complex operand. Per-DPU or per-slice scaling may change that policy. Either the required operand and scale computation are local and validated, the prescribed host/global boundary is retained, or the resident int8 case is declared unsupported by that experimental mode. Do not silently rename per-slice quantization as the old shared-scale policy.

Test three roles on the same supported circuit: unsliced serial-node execution, sliced serial execution, and the same sliced plan executed concurrently. This separates slicing's work inflation from the benefit of concurrency. Compare resident and host-roundtrip variants using the same sliced/segment DAG where possible. Any added slicing changes the logical identity; residency alone can preserve it.

## 6. Small kernel portfolio

The existing lowering already exposes batched matrix geometry. The contribution is not discovering that contractions can become GEMM, but choosing and implementing suitable UPMEM kernels for that geometry and for proven exact tensor structure. [S1]

| Kernel or improvement | Decision | Main qualification issue |
|---|---|---|
| Existing generic real WRAM-panel kernel | Keep as coverage/control | No regression in admitted shapes or policies |
| One-launch four-product complex execution | Default boundary-focused candidate | Fewer physical launches, not merely fewer host envelopes; preserve arithmetic/reconstruction |
| Skinny GEMM/GEMV/DOT specialization | Default geometry-focused candidate if costly in census | Tails, orientation, tasklet utilization, long reductions |
| Outer product (`K=1`) | Alternative to skinny kernel when more valuable | Remove reduction machinery without changing output semantics |
| Native signed-int8 multiply/load-loop refinement | Inspect generated code first | Correct instruction selection, sign extension, int32 accumulation, actual ETH build |
| Dense tile/traversal tuning | Small search inside existing kernel | WRAM/IRAM use, DMA legality, register/stack pressure |
| Diagonal/permutation/sign specialization | Conditional second geometry/semantic kernel | Exact predicate, label ordering, same declared numerical policy |
| DPU packing/permutation or layout propagation | Conditional locality work | Real removable host cost; no hidden host equivalent or extra full copies |
| ATiM-generated implementation | Bounded external probe | SDK compatibility and complete-route benefit |
| New integer/quantized format | Excluded | Preserve the current numerical experiment |

Retain generic coverage plus at most two new principal kernel variants in the first system freeze. An instruction-level refinement is a version of a kernel, not a reason to build a registry.

### 6.1 Complex launch fusion versus envelope batching

The reported host-only complex-envelope prototype reduces packaging boundaries, not the number of DPU launches. Do not present it as arithmetic or kernel fusion.

The first complex kernel candidate should execute all four real products within one launch. Start with separate real-product accumulators and the original reconstruction convention; returning four products initially is acceptable. This isolates launch/control savings without silently changing the arithmetic. Combining the final complex output on the DPU, reducing return traffic, or changing accumulation order is a separately checked refinement.

For float32, preserve the existing per-product summation and reconstruction sequence where practical. Where rounding semantics change, assign the appropriate numerical execution identity and use explicit same-policy replay rather than claiming bitwise equivalence.

For int8, retain whole-operand scales, nearest-even rounding and the original reconstruction/requantization boundaries. Check real-product bounds `K*127^2`; if a signed accumulator combines two products, check the relevant `2*K*127^2` bound and actual reduction schedule. Never rely on signed-overflow behavior. The proposed launch fusion does not require a three-real-product algebraic reformulation.

### 6.2 External reuse

**PrIM:** use its GEMV design and microbenchmarks as reference material or reusable code after inspecting the actual pinned files and license. Its repository provides MIT-licensed benchmark implementations. [S9]

**ATiM:** use an isolated, one-operation probe. Its artifact documents Ubuntu 20.04 and SDK 2021.3.0; compatibility with the project's reported ETH SDK must be established. Its root license is Apache-2.0. [S10] Import a useful generated kernel or schedule only after correctness and timing through this runtime. Do not adopt its whole compiler/runtime as a prerequisite.

**PIMutation:** use its exact-structure ideas, especially replacing suitable dense operations by permutations. It is a direct-statevector system, not a TN executor. [S11] A paper license is not evidence of an implementation's code-reuse license. Do not import unverified code, gate-fusion assumptions, or separable-state behavior into the TN comparison.

**UPMEM Unleashed:** inspect the actual emitted int8 multiplication and loading loops before changing them. Its compiler observations motivate this audit, not an assumption that this project's binary necessarily has the same slow path. [S4]

## 7. Minimal architecture and ownership

Use the existing functional core and imperative shell. The additional conceptual records are small:

`KernelDecision(operation_id, kernel_id, geometry, layout, tile_parameters, numeric_policy)`

`ExecutionWave(wave_id, ready_node_ids, dpu_assignments, transfer_groups)`

A resident extension can add a bounded `ResidentSegment` with tensor offsets, instruction records and boundary tensors. Do not add it until that experiment is admitted.

| Area | Planned change |
|---|---|
| `upmem/tiling.py` | Geometry predicates, layout/packing facts, kernel eligibility |
| `upmem/plan.py` | Kernel decisions, static waves, admission, liveness and physical identity |
| Native wire protocol/host/DPU | Per-slot operation descriptor, validated selector, shared launch, completion facts |
| `upmem/runtime.py` | Execute waves, reconstruct outputs by operation, deterministic dependency publication |
| CPU replay/numerics | Replay chosen kernel arithmetic and declared reductions; keep high-precision input handling |
| Path feature extraction | Features from final kernel choices and schedule, not old fixed four-launch assumptions |
| Evidence/reporting | Per-operation and wave facts, padding, useful work, waiting, movement and timing scopes |

The native protocol must associate every result with its operation, work unit, logical/physical plan, and numerical mode. A protocol change requires one active validated reader/writer; do not maintain parallel legacy production protocols. Preserve existing source tags for historical reproduction.

One DPU binary containing the small dispatch set is preferred only when its IRAM and WRAM use qualify. Otherwise freeze separate explicit executable profiles and measure load/session costs. Do not silently swap binaries mid-contraction.

## 8. Implementation work packages and branch sequence

Branch names are proposals, not assertions that branches exist. Reuse an existing correct branch rather than creating a duplicate.

### P0 — Reconcile and qualify integration

**Branch:** existing integration branch, or `feature/upmem-execution-integration-v2` from verified main.

Read current remote heads, ancestry, local worktrees, merged files, current plan, evidence, and qualification results. Determine whether quantization and path infrastructure have already been integrated. Preserve the semantics identified by the original plan: simulator selection, physical one-rank admission, fail-fast records, path-specific validation, workload hashing and repaired numerical handling. [S1]

Run only missing or invalidated qualification. Use the existing seven-session physical gate if its exact manifest applies to the current source; otherwise preregister a small replacement with a new identity. Historical released campaigns need not be repeated merely because histories were merged.

**Exit:** accepted integration source, clean lineage, exact-head software/SDK checks and physical smoke, or an explicit physical-access block. Software work may continue on a checkpoint while hardware is unavailable, but no physical acceptance is inferred.

### P1 — Cost and parallel-headroom census

**Branch:** `feature/upmem-execution-census-v2`, or reuse the reported census.

Read existing census outputs first. Extend them only for missing questions: time-weighted geometry, four-product boundaries, ready frontier width, critical path, work per wave, DPU-slot filling, live host/MRAM memory, transfer padding, and candidate resident boundaries. Use fixed greedy and a small predeclared set of alternate development paths; do not use final test timings.

A source-only census identifies opportunity, not physical speedup. Collect a narrow attribution run only for unresolved ranking of high-impact targets.

**Exit:** selected boundary kernel, selected geometry kernel or alternative, static-wave test cases, and residency/slicing go/no-go criteria. Record this choice once; do not keep changing targets after every noisy sample.

### P2 — Kernel implementations and direct dispatch

**Branch:** `feature/upmem-kernel-portfolio-v1`.

Implement one-launch complex execution and one census-selected geometry kernel in bounded sequence; begin with whichever has the stronger measured removable-cost case. Inspect native int8 arithmetic concurrently with kernel work. Qualify each variant in isolation before composing it with the scheduler.

The generic implementation remains the explicit coverage route. Unsupported fast-path shapes are rejected during planning or routed to that generic implementation by the declared policy—not to a CPU contraction fallback.

**Exit:** complete kernel correctness, realistic shape coverage, fixed-path physical A/B, retained/rejected decision, and source/evidence checkpoint.

### P3 — Static dependency-ready DAG-wave execution

**Branch:** `feature/upmem-static-dag-waves-v1`.

Develop scheduling as a pure function while P2 proceeds. A single owner integrates native descriptor changes to avoid conflicting ABI work. First compare serial-node and frontier-wave execution with the same generic kernel; then test accepted kernels with the same scheduler.

Prove no dependency violation, no shared writable output, deterministic within-node reductions and resource identity. Use a synthetic fork-join DAG for targeted validation and supported quantum-circuit DAGs for performance relevance. Do not infer useful quantum-workload parallelism from a synthetic graph alone.

**Exit:** physically demonstrated concurrent independent nodes on disjoint DPUs, fixed-resource comparisons, supported-circuit correctness and a decision about production selection. A narrow frontier or a measured loss is a reportable result.

### P4 — Bounded resident-subgraph and slicing study

**Branch:** `feature/upmem-resident-slice-probe-v1`.

Begin only after P1/P3 identify candidates. First qualify host-mediated slice scheduling where existing semantics permit; then attempt one local resident segment with a fixed buffer plan. Full autonomous arbitrary slice execution is not an implicit requirement.

Use at most two development instances and a few slices, with all slices included. Derive the slice count from actual admitted DPUs, working-set feasibility and work inflation—not a hypothetical 2,560-DPU system. Prefer one-rank 2/4-DPU trials before larger counts.

**Exit:** an implemented/qualified bounded extension with evidence, or a documented no-go at the effort cap. Report separately whether concurrency, residency, and full-output assembly were actually implemented. Only accepted extensions enter the final executor.

### P5 — Composition, ablation and system freeze

**Branch:** accepted milestone descendants; **tag proposal:** `thesis-upmem-kernel-schedule-system-v1`.

Run composition tests across retained kernels, schedules and numerical modes. Freeze source, binaries, dependencies, layouts, dispatch and scheduling rules, memory policy, numerical boundaries, resource profiles, timing scopes and feature extraction.

The frozen policy may select different kernels or assign different DPU groups for different shapes/frontiers. Its rules are fixed; decisions need not be constant. No final path training before this gate.

### P6 — Final-system path optimization and untouched evaluation

**Branch:** `feature/upmem-final-system-path-search-v2`.

Use the procedure in Section 11. Preserve the old 192-attempt configuration unchanged as superseded preregistration; it is not calibrated evidence for this executor. Create new study identities and physical-plan hashes.

### P7 — Closure

Release the accepted system and the final path profile with source lineage and raw archives. Keep rejection reports short. Export the standalone repository only after the accepted implementation/profile checkpoint; do not develop two active codebases.

## 9. Qualification and adoption gates

### 9.1 Correctness and scientific blockers

Kernel coverage must include zeros, signed complex inputs, unequal/skinny dimensions, `K=1`, boundary tiles, DMA alignment, long reductions, int8 extrema and near-overflow geometry. Verify the full statevector and ordering, not only kernel outputs. Keep physical execution success, policy-replay agreement, and accuracy qualification separate.

Scheduler tests must cover a chain, fork-join, uneven branches, more ready work than DPUs, fewer items than DPUs, a one-DPU degeneration, multiple tiles per node, reduction nodes, mixed geometry, idle descriptors, deterministic repeated planning and failure propagation. Every consumer waits for all required outputs, including reconstruction/reduction. Include one non-power-of-two physical correctness route when affected by changes.

Resident/sliced tests must cover complete slice enumeration, no duplicates/missing contributions, open-output assembly, tensor lifetimes, layout compatibility, arena bounds, and the actual quantization/reduction semantics. No local-scale substitution is permitted under an unchanged policy name.

### 9.2 Physical adoption rule

Before each A/B packet, declare the target workload/geometry region, primary timing scope, permitted regression, and practical improvement threshold. Use repeated paired blocks and a fresh confirmation of the chosen version. A candidate qualifies for deployment in a region only when its complete-route benefit justifies its maintenance and resource cost. Do not accept a microkernel win that becomes a full-route regression.

Suggested default policy—not a scientific constant—is a 5% practical improvement target for the preregistered region, with no unexplained regression beyond 5% outside that region under the dispatcher. Final thresholds should be fixed from timing variability and use case before the candidate comparison, not selected after seeing the winner. Statistical uncertainty and effect size must both be reported.

Negative results remain in the thesis even when the corresponding implementation is not retained. Existing controls may remain as minimal ablation modes; do not preserve an obsolete execution stack merely for historical comparison.

### 9.3 Work classification

**Fix now:** wrong output or policy replay; missing dependencies; overflow; invalid memory access; wrong resource/provenance identity; timing-scope mismatch; incompatible live evidence.

**Address when it impedes the next phase:** duplicated four-product assumptions; missing operation IDs; monolithic per-operation submission; feature extraction that cannot describe a wave; ambiguous tensor ownership.

**Defer:** cosmetic renaming, a general scheduler, registries, cross-platform compatibility work, new plotting frameworks, and unmeasured broad refactors.

## 10. Effort and physical-budget boundaries

These are planning estimates, not promises. Core integration/census/kernel/scheduler work is approximately **4–7 focused engineering days**, depending on existing integration and native-protocol work. P4 adds a **2–3-day capped probe**, not a guaranteed complete resident TN engine. ETH queueing and evidence retrieval are outside those estimates. The thesis writing track continues independently.

| Packet | Proposed ceiling | Notes |
|---|---:|---|
| Missing integration smoke | Existing valid seven-session gate, or small replacement | Never repeat a gate already valid for this source |
| Kernel micro-exploration | 24 configurations × 4 attempts = 96 | Across selected knobs; not an unrestricted Cartesian product |
| One kernel full-route A/B | 3 circuits × 2 topologies × 2 variants × 6 attempts = 72 | A second principal kernel needs a separately accounted packet |
| DAG-wave full-route A/B | 3 circuits × 2 topologies × 2 modes × 6 attempts = 72 | Prefer 2 and 4 DPUs, with fixed total resources |
| Resident/slice exploration | 2 instances × 2 topologies × 3 roles × 4 attempts = 48 | Only after admission; all output/reduction costs included |
| Resident/slice confirmation | At most 48 additional attempts | Only for an adopted candidate |
| ATiM probe | One engineering day; at most 16 generated schedules initially | Optional, separately budgeted physical use |
| Final path study | At most 540 attempts for the primary profile | Section 11; no silent doubling for int8 |

Four attempts mean one warmup and three measurements; six mean one warmup and five measurements. These counts support a bounded diagnostic study, not guaranteed statistical power. Stop or report inconclusive results rather than silently extending the campaign. Freeze a wall-time cap per packet from P0/P1 observations before launch. Count smoke, warmups, failed attempts, repeated controls and confirmation in the hardware ledger.

For each implementation packet: one implementation, focused tests, checkpoint, independent audit, physical A/B, and at most one evidence-justified repair or optimization cycle. A correctness defect may require repair for qualification, but it does not reopen an unlimited performance search.

## 11. UPMEM cost-guided path optimization: final-phase contract

### 11.1 Scope, verified starting point, and supersession

This section is the controlling implementation plan for the remaining phase. Treat P0-P5 as complete; do not reopen branch integration, tasklet/DPU allocation, DAG scheduling, quantization, kernel dispatch, fusion, slicing, residency, transport, or resource admission. The research contribution is a whole-plan cost function consulted during adaptive path generation, not only after conventional candidates have been generated.

Initial read-only binding on 2026-09-09 (historical preparation checkpoint;
see 11.10 for the subsequent software checkpoint and remaining work):

| Item | Bound value / disposition |
|---|---|
| Active implementation worktree | /home/tom/repos/Masters/.agent-work-final-system-path-search-v2 |
| Branch and preparation source | feature/upmem-final-system-path-search-v2; 29ebaa2d1b20a1f6469435efe7309f89e161f263 |
| Accepted execution source | 459935f586fdd16c82013838e6d27a12604c3093 |
| Execution tag | thesis-upmem-kernel-schedule-system-v1, locally verified to peel to the execution source |
| Frozen execution | packed_wave_v1; static_dag_waves_v1; fuse_complex=true; panel_only_v1; host-roundtrip reconstruction; one rank |
| Primary policy and resources | split_complex_float32_v1; 1 DPU/T8 and 4 DPUs/T8 |
| Workload source | implementation/configs/upmem_family_aligned_workload_v1.json |
| Workload raw SHA-256 | f2c85f27508d9b9fb5f55d13b89334939aa424330e27555dfafe3216452382d7 |
| Split | Six development/training instances and six test instances; all six families in each split |
| Existing preparation qualification | 1,973 tests and Ruff passed; exact-head CI run 34340679355 succeeded, as recorded before this planning amendment |
| Installed search library | cotengra 0.7.5, Python backend available; Optuna not installed |
| Historical material | Old 92-attempt development evidence and interrupted 64-trial FLOP-pool generation remain separate; neither supplies this study's fit |

Preserve all twelve QASM bytes, hashes, canonical problem IDs, parameters and split assignments. The six families are BB84, BV, EDC, HS, QRNG and XOR, using our declared family-aligned instances, not claims of reproducing unavailable PIMutation code. EDC has one logical qubit, encoded data wires and separate syndrome ancillas. Preserve full pre-measurement statevector outputs including ancillas.

The old family-aligned study configuration specifies fixed-pool reranking, a six-term score, per-cell log normalization, one adaptive round and an older comparison layout. It is historical preregistration, not the controller for this amendment. At checkpoint 1 create a new private study configuration, upmem_cost_guided_path_study_v1, bound to the unchanged workload and executor. Do not rewrite old observations or silently reuse their model/profile identities. The interrupted old generator must not resume automatically.

Planning provenance: this amendment was initially prepared without implementation, installation, searches or experiments. The subsequently approved goal authorizes implementation; the software, freeze and evidence gates below still control execution.

### 11.2 One production-plan boundary and five facts

Define the reusable function

\[
(c,p,e,\theta)\longmapsto C_\theta(\operatorname{Lower}_e(c,p)).
\]

The fixed executor chooses kernels, assignments and schedule through its existing rules. The scorer observes those choices; it must not invent a different schedule or choose CPU arithmetic. P means host array-pass traffic, never the historical P_wram feature; M means local transfer traffic, never the MRAM storage-span field.

Reuse build_contraction_dag, plan_upmem and extract_execution_features. The last function already derives emitted control waves from build_cohort_controls and reports slot-level work, movement and launch facts. Verify that each scored launch corresponds to one native physical launch. Do not substitute a logical ready frontier for its emitted launch sequence.

| Fact | Exact scope |
|---|---|
| H | Total planned H2D plus D2H bytes, including operands, outputs, replication, alignment/padding and control/completion transfers |
| P | Bytes read plus bytes written by explicitly modeled host array passes in preparation and reconstruction; a data-volume proxy, not measured CPU time or peak memory |
| N | Native request-envelope count plus physical launch count; no extra barrier, DAG bonus or critical-path penalty |
| M[l,d] | Source-derived estimate of MRAM-WRAM transfer bytes assigned to DPU d in physical launch l |
| W[l,d] | Real MAC pairs assigned to that DPU in that physical launch |

For P, freeze a small pass inventory tied to actual runtime function locations: operand materialization/packing and real/imaginary conversion, payload copies that actually occur, output decoding/assembly, complex reconstruction and required host reductions. Count each modeled read/write once at its actual dtype and extent. Zero-copy views count zero movement. Explicitly list excluded hashing, filesystem and allocator work; do not label P a complete host-cost measurement. Do not substitute the current retained-memory estimate for cumulative host-pass traffic. Binding this inventory is a checkpoint-2 gate, not permission for another profiling/optimization campaign.

Fused complex still performs four real products. Use actual tile extents and kernel behavior. Never divide already assigned W or M by DPU/tasklet counts. Record idle-slot facts and fixed transfer/launch overhead even when arithmetic work is zero.

Extract facts without SDK calls, contraction execution, or allocating intermediate/statevector-sized dummy arrays. Shape/stride metadata and existing small input tensors are sufficient. If a helper currently materializes an intermediate merely for planning, provide the smallest metadata-only adaptation, not a second planner. Preserve production feasibility checks for live storage, WRAM, MRAM, alignment, accumulators and ownership. Numerical correctness is a separate predicate, never an error penalty in the score.

### 11.3 Freeze global normalization and the launch-aware score

For each development cell j, obtain greedy facts with the frozen lowering:

\[
X(g_j)=(H,P,N,\sum_{l,d}M[l,d],\sum_{l,d}W[l,d]),\qquad
s_k=\max(1,\operatorname{median}_{j\in D}X_k(g_j)).
\]

Freeze these five scales before any calibration timing. Test instances do not contribute. Do not renormalize by candidate, cell, round or newly observed timings.

Use exactly

\[
C_\theta=
\theta_H H/s_H+\theta_P P/s_P+\theta_N N/s_N+
\sum_l\max_{d\in D_l}
\left(\theta_M M[l,d]/s_M+\theta_W W[l,d]/s_W\right).
\]

Movement and computation are combined for each DPU BEFORE taking the maximum. The old sum of independent maxima and old per-cell log-ratio score are not this model and their weights must not migrate. Record the new private model identity upmem_launch_cost_v1 and extractor/source hashes.

This is a dimensionless ranking score, not a calibrated prediction in seconds. Report feature correlations and constant features, but retain the prescribed five-dimensional model and grid; do not introduce nonlinear terms, a six-vs-three model-selection branch, additional penalties, or an int8 fit. Int8 may later use the same software only with its own profile, calibration and separately authorized budget.

### 11.4 Deterministic coefficient fitting

Enumerate every nonnegative integer tuple k=(kH,kP,kN,kM,kW) with sum(k)=10, exactly C(14,4)=1001 tuples; theta=k/10. Initial theta is (0.2,0.2,0.2,0.2,0.2). Store integers as the authoritative representation.

The primary timing T is session_open_s + steady_execution_v1 total_wall_s + session_close_s. Keep steady execution as a secondary view; do not change this choice after timing. Runtime work performed inside those boundaries remains included; external reference validation/reporting is excluded.

Reject evaluation-role observations from fitting. For each measured, numerically eligible development candidate p in cell j, pair its non-warmup timing with greedy from the SAME cell, round and block:

\[
\ell_{jp}=\operatorname{median}_{(r,b)\in O_{jp}}
\log(T_{j,g,r,b}/T_{j,p,r,b}).
\]

Missing or duplicate controls, nonpositive timings, source/policy mismatches and incomplete accepted rounds fail validation. Never pair unrelated medians or impute observations. Deduplicated greedy rows have log-speedup zero.

For every grid tuple select among measured eligible paths only, minimizing (cost, path_id). Maximize J=sum_j a_j*ell[j,selected], with a_j=1/(F*n_family(j)). Freeze the complete development membership; no silent reweighting after dropping an inconvenient cell. Report ineligible cells explicitly and stop/declare an incomplete campaign when required controls cannot be admitted.

Tie-break in this exact order: greatest round(J,12); greatest worst-cell log-speedup; least sum_i(k_i-2)^2; lexicographically smallest integer tuple. Fixed sorted cell/path order and float64 arithmetic make reductions reproducible. No Monte Carlo coefficient search, optimizer registry or hardware per tuple. exp(J) is the family-balanced geometric training score, not unbiased generalization.

### 11.5 One adaptive search route

Use a serial Optuna ask/tell TPE loop around cotengra, not custom local annealing, subtree objectives or HyperOptimizer random sampling. The installed cotengra annealing hook scores local FLOPs/sizes, whereas this objective requires the complete schedule. A fresh random-greedy object avoids an internal best-so-far FLOP filter.

Checkpoint 1 pins cotengra==0.7.5 and Optuna==4.5.0 in the research dependency environment, without changing the frozen executor environment. Installation and dependency-resolution checks are recorded in the software checkpoint in 11.10; bind the complete resolved research lock before experimental traces. Do not silently use a newer release. [Optuna release](https://pypi.org/project/optuna/4.5.0/)

Each call makes exactly 128 proposals, counting duplicates and known-infeasible proposals. Use TPESampler with explicit seed, n_startup_trials=16, serial independent-parameter sampling and no pruning/distributed service. Ask for costmod in [0.1,4.0] linearly and temperature in [0.001,1.0] logarithmically.

For EACH ask: construct a fresh RandomGreedyOptimizer(max_repeats=1, costmod=float(...), temperature=float(...), seed=proposal_seed, accel=False, parallel=False, simplify=True); call search exactly once; convert its complete tree path with the existing canonical adapter; lower/admit; evaluate the selected objective; tell before the next ask. No extra annealing, reconfiguration, slicing, inner repetitions or reuse of optimizer best-path state.

Use master_seed=20260909. Derive sampler and proposal seeds from SHA-256 of sorted compact UTF-8 JSON containing a seed-domain label, master seed, workload/cell identity, stage/round and proposal ordinal where applicable; take the first four digest bytes as an unsigned big-endian integer. Pair F and U by the same sampler seed and ordinal proposal-seed schedule so differing scores, not differing seeds, can alter later parameters. Model/profile hashes belong in trace identity, not in the paired seed. The first startup proposals may coincide; later proposal dependence must be demonstrated.

Known deterministic infeasibility consumes its proposal and is reported to the minimizer as positive infinity, with a JSON-safe rejected status/reason and null finite score. Duplicates consume proposals, receive the correct score and remain in the trace. Unexpected exceptions, nonfinite eligible scores, or wall-time interruption abort the trace; do not turn bugs/timeouts into conveniently excluded paths or resample until 128 successful paths exist. Qualify pinned Optuna's rejected-trial behavior before use. An interrupted trace is not a completed search.

Record proposal number, parameters, seeds, path/plan IDs, feasibility, raw facts, score, objective/profile identity and tell order. Two fresh-process runs must match parameters, paths, decisions and scores exactly, excluding timestamps. Persist every attempted proposal. Optuna's ask/tell interface provides the required feedback order; sequential execution and a deterministic objective remain necessary. [Ask/tell](https://optuna.readthedocs.io/en/v4.5.0/tutorial/20_recipes/009_ask_and_tell.html), [reproducibility guidance](https://optuna.readthedocs.io/en/v4.5.0/faq.html)

Avoid a cache framework. A bounded per-search map may memoize deterministic facts by circuit/path/executor/extractor/numeric identity and scores by those keys plus model/scales/weights. No cross-profile/global mutable cache. Changed identity must force recomputation. A duplicate must still be told to TPE and counted.

### 11.6 Physical feedback and the four evaluation methods

All search calls use the same 128-proposal engine. FLOP-guided calls tell the conventional complete-tree FLOP cost; UPMEM-guided calls tell C_theta. Apply the same deterministic feasibility and proposal rules to both.

Round 0, development only: run a FLOP-guided trace per cell; include G, its FLOP-best path F and the uniform-score reranked path R. Deduplicate, then fill only up to the effective initial-path cap in 11.7 by farthest-first L1 distance on z=X/s. At each step maximize minimum distance to the selected set, tie by path_id. Do not use timing for membership. Qualify selected paths, freeze the round manifest, collect one warmup and three randomized complete measured blocks. Archive and accept the entire round before fitting theta1.

Rounds 1 and 2: use the current frozen theta to guide a NEW 128-proposal trace per development cell. From eligible previously unmeasured paths choose the lowest-cost path, then one diverse path using the same distance rule relative to the chosen path and G. Add a fresh G control, deduplicate to at most three paths. If there is no new eligible path, skip that cell without another search. Collect 1+3 blocks, archive/accept the complete declared round, append observations, enumerate the grid and freeze the next theta. Stop adaptation after round 2, including rounds with no new choices.

Before evaluation, freeze theta, scales, source/binaries, extractor, search settings/seeds, timing and accuracy policy, and all evaluation membership. Generate and freeze ALL evaluation traces and method selections before any evaluation timing:

| Method | Definition |
|---|---|
| G | Existing deterministic greedy control |
| F | Lowest-FLOP eligible path from its 128-proposal FLOP-guided trace plus G |
| R | Lowest-C_theta eligible path from EXACTLY F's recorded trace plus G |
| U | Lowest-C_theta eligible path from a separate 128-proposal UPMEM-guided trace plus G |

U must not receive F's trace as extra candidates. Physical method coincidences are measured once per cell/block, retaining every method label. Evaluation uses one warmup and five randomized complete blocks, including G in each block. After any evaluation timing, no changes to features, scales, weights, search settings, candidates or model selection are permitted.

Compare U versus R to isolate feedback-guided generation beyond reranking, and U versus F/G for conventional controls. Report method identities, coincident paths, per-cell medians/MADs/paired intervals, session-inclusive and steady execution, kernel/H2D/D2H/host costs, planning/search time, geometric summaries, worst cells and regressions. Equal proposal counts are not equal search wall time. Include complete job time before claiming end-to-end acceleration; report planning break-even only for positive savings. This is instance/size transfer within six represented families, not family-held-out generalization.

### 11.7 Mechanically bounded execution

Read development/evaluation cells from the unchanged manifest, including topology. For this packet nD=nE=12. The new contract's unconstrained ceiling is 48*nD+24*nE=864 attempts, but the already declared 792-attempt cap takes precedence.

Keep two feedback rounds and all four final methods. Derive a uniform initial-path cap before any timing:

\[
m_0=\min\left(6,\left\lfloor
\frac{B-24n_D-24n_E}{4n_D}
\right\rfloor\right).
\]

Here B=792 yields m0=4. A cap below the mandatory distinct initial roles is a preflight budget failure, not permission to omit roles/families. The effective maximum is:

| Stage | Maximum attempts |
|---|---:|
| Initial: 12 cells * 4 paths * (1+3) | 192 |
| Feedback 1: 12 * 3 * (1+3) | 144 |
| Feedback 2: 12 * 3 * (1+3) | 144 |
| Final G/F/R/U: 12 * 4 * (1+5) | 288 |
| Total | 768 |

The remaining 24 attempts are unallocated, not a retry/refill reserve. Actual counts come from frozen deduplicated route sets; failed attempts count. No former training-confirmation stage or separate validation campaign is added. Old 92 observations and abandoned preparation have separate ledgers and never enter this fit.

Keep the current 86,400-second cumulative physical-stage ceiling and 120-second per-attempt runner timeout. Preserve the existing 300-second parent guard for each proposal and 60-second lowering guard, with a 7,200-second whole-search ceiling. Pass only supported parameters to RandomGreedyOptimizer: it has no max_time argument in 0.7.5. Hitting a wall-time guard aborts/incompletes the trace rather than changing membership. No unbounded retry, automatic budget increase or additional feedback round.

Use the existing private flock, one physical controller on safari-baguette1, known-good admitted rank, CPU affinity/governor/SDK records, exact binaries/configs and writable evidence storage. Availability is not reservation. Stop on the first failed/unsupported/fallback execution; retrieve the complete partial stage, release owned resources and diagnose without splicing replacements.

A stage is accepted only after remote sorted relative SHA256SUMS, durable local retrieval, checksum and canonical verification, exact sample/session/route/block identities, portable archive plus outer hash, and two independently verified copies. Do not delete volatile data first. Do not re-execute an accepted round. Persistent state binds study/round hashes and records frozen, running, accepted or failed using the existing controller conventions; no workflow framework or new public evidence schema.

### 11.8 Five implementation checkpoints and file ownership

One bounded implementer, one independent read-only reviewer, and the lead as sole physical controller suffice. No recursive delegation. Reviews target mathematical correctness, dependency/ownership safety, feedback validity and evidence leakage, not generic refactors.

| Checkpoint | Implementation boundary | Acceptance and stop condition |
|---|---|---|
| 1. Bind | Existing family manifest/QASM; new private study config; implementation/pyproject.toml research pins; private freeze record | Exact sources/binaries/workload/scales-input membership and effective 768 budget bound; library signatures/smoke tests qualified; old fixed-pool study explicitly superseded. Stop on any mismatch. |
| 2. Score | execution_features.py metadata facts and path_heuristic.py pure scale/score functions | Host-pass inventory, physical-launch mapping, joint maximum, four-product counts and metadata-only lowering tests pass. No executor behavior changes. |
| 3. Search | A small private search adapter/CLI, preferably scripts/upmem_cost_guided_path.py, reusing existing canonical path and plan helpers | Fresh single-candidate cotengra, ask/evaluate/tell order, no hidden execution, two-process full-trace reproducibility and objective-responsive proposals pass. |
| 4. Control | Pure grid/batch functions and thin campaign commands; reuse qualify_upmem_path_candidates.py and existing collection/evidence helpers | All 1001 tuples and tie-breaks verified; fake-runner counts, failure, dedup, accepted-round idempotency and leakage/freeze tests pass. No real hardware yet. |
| 5. Execute and close | Existing CPU/SDK/physical runner, extraction, archive and report path | Full suite/Ruff/diff/CI; small strict SDK candidate check; bounded rounds durably accepted; frozen evaluation; final report/profile/raw evidence retained. Negative performance completes the study. |

These are four responsibilities (facts/score, search, fitting/batches, campaign CLI), not four class hierarchies. Extend existing pure helpers where appropriate. The historical score/extractor identities remain interpretable for old artifacts; do not silently change their mathematics. Do not add compatibility readers, registries, plugins, generic scheduling/cache systems, or another physical runner.

Required tests:
- Compute-only work (4,4): one two-DPU launch scores 4; two serial launches score 8.
- Joint maximum: slots (M,W)=(10,0),(0,10), half weights, unit scales score 5, not 10.
- Generic four launches versus fused one: identical total real MAC count; only applicable launches/movement/packing differ.
- Extracted kernels, slots, products, extents, resource facts and launches equal production-lowered controls.
- Spy/fail SDK/contraction calls and large allocation attempts during scoring/search; metadata-only preparation remains valid.
- Exactly 1001 integer tuples; synthetic paired timing tables exercise every tie-break, missing controls, warmup exclusion and family weighting.
- Fake ask/tell verifies tell precedes next ask; a nonconstant toy objective changes post-startup proposal traces when scores change.
- Fresh-process deterministic 128-entry traces; no hidden inner repeats, automatic backend or score-smudge/compression.
- Duplicate and known-infeasible proposals count once each; unexpected errors abort; cache identities cover all semantic inputs.
- Fake physical runner enforces frozen exact counts, role deduplication, first-failure stop, two-copy acceptance and no accepted-round rerun.
- Fitting rejects evaluation data and missing development controls; pretest freeze forbids further adaptation.
- Existing complete-path/CPU numerical checks plus representative strict SDK paths; never assert a required speedup.

### 11.9 Artifacts, closure, and claims

Retain a compact private bundle: binding.json; normalization.json; per-call search_trace.jsonl and path/fact tables; round manifests; raw sample/session evidence and runtime tables; 1001-row fit tables and integer profiles; final_freeze.json; evaluation method mappings/results; report.md; relative SHA256SUMS and outer archive digests. Each binds source, executor/binaries, workload/QASM, path/plan, model/extractor/scales/profile, seeds, split, timing/numeric policy and environment as applicable.

Report the new model's explicit relationship to SLR section 9.3: operationalized host traffic, host array work, coordination, local movement and arithmetic; constraints remain separate. Do not call coefficients exact architectural constants or the unique physical model. Do not claim literal cotengra annealing or global path optimality: this is UPMEM-cost-guided hyperparameter search over cotengra-generated complete paths.

The decisive completion evidence is a trace showing full-plan cost feedback changes subsequent search proposals, followed by a frozen G/F/R/U evaluation through the unchanged executor. A favorable reranking result alone does not complete this milestone. No further optimization follows the declared evaluation; publish only after the complete evidence has two verified copies.

### 11.10 Current checkpoint and ordered continuation

The committed software checkpoint is
939a0161c22aa2fb6f315b1c2a9116654a32fe1e on
feature/upmem-final-system-path-search-v2. Its recorded qualification is
2,023 passing tests without skips, Ruff and diff checks, with the detailed
record in implementation/docs/upmem_cost_guided_path_v1.md. It includes the
launch-aware observer/score, isolated pinned research environment, study
configuration and adaptive search primitives. This is software evidence, not
physical calibration or completion of the campaign controller.

At this planning update, fitting, deterministic batch selection and explicit
path-replay changes are present as uncommitted work in the active worktree.
Preserve and review those edits; do not treat the dirty checkout as an accepted
execution source or restart the implementation from scratch. No new physical
observations or fitted profile are established by this status update.

Resume in this dependency order:

1. **Close the software controller.** Review the existing fitting and batch
   changes against 11.4 and 11.6. Finish only the missing campaign commands and
   guards in the small private CLI. Retain per-proposal traces, complete round
   membership, hashes, budgets and explicit accepted/failed state. Use synthetic
   observations and the existing runner interface for fail-fast, resume,
   two-copy acceptance and evaluation-leakage tests.
   The read-only fitter review identified two guards to close before acceptance:
   enforce the declared float32 policy and valid source identities even when
   supplied identity dictionaries agree, and reject failed execution/startup
   admission facts. Add focused regression cases; matching metadata alone does
   not prove a valid physical observation.
2. **Bind exact selected-path replay.** The historical planner entry regenerates
   paths through its optimizer configuration; that is not an identity guarantee
   for a new RandomGreedy proposal. Use the smallest explicit complete-path
   input through planning.py, experiment.py and cli.py. Validate integer pairs,
   completeness, tensor-network identity and resulting DAG identity before
   executor allocation. Never substitute a regenerated HyperOptimizer path.
   Reuse production lowering and execution unchanged; this adapter is a path
   input, not a new planner, schedule or physical runner. Record preparation
   source separately from frozen execution source and binary identities.
3. **Qualify and freeze preparation.** Run focused mathematics, replay,
   fresh-process search and fake-campaign tests, then the full pinned suite,
   Ruff, diff checks and exact-head CI. Qualify representative selected paths
   with CPU reference and strict SDK correctness using the existing runner.
   Freeze development-only greedy scales, research/source bindings and the
   complete initial candidate manifest before physical admission. Stop on a
   genuine correctness or binding defect; never assume the physical gate passes.
4. **Collect and fit the bounded development rounds.** Admit hardware through
   the existing lock/preflight. Initial collection has at most 192 attempts;
   each of two feedback rounds has at most 144. Accept and durably archive each
   whole declared round before fitting or advancing. No hardware per tuple,
   retries, replacement observations or additional searches to fill duplicates.
5. **Freeze and evaluate.** Freeze the final integer coefficients and every
   pretest identity, then generate and freeze all final G/F/R/U selections.
   Execute at most 288 evaluation attempts without subsequent adaptation.
   Report U versus R explicitly, alongside G/F, session-inclusive and steady
   timings, search cost, numerical validity and regressions.
6. **Close once.** Verify exact evidence sets and two independent copies,
   publish the profiles/traces/raw results and bounded interpretation, then
   stop. Do not reopen execution-system optimization because a family is
   neutral or U does not outperform R.

One implementer owns the bounded code task, an independent reviewer checks the
mathematics, replay identity and leakage/failure gates, and the lead is the only
hardware controller. An updated plan is not itself permission to launch a
campaign before these gates.

## 12. Thesis ablations and reporting

Use the following controlled comparisons, then separately report cumulative selected-system results:

| Comparison | Held fixed | Isolated question |
|---|---|---|
| Scalar-MRAM versus WRAM-panel | Contemporary transport, small feasible workload, float32, T1/DPU1 | Memory staging/blocking |
| T1 versus selected tasklet count | Same path/kernel/one DPU | Intra-DPU parallelism |
| One versus multiple DPUs | Same path/kernel/tasklet count | Existing intra-contraction distribution |
| Four-launch versus one-launch complex execution | Same arithmetic/path/resources | Launch/control fusion |
| Generic versus shape/semantic kernel | Same path/numerical policy/resources | Kernel specialization |
| Serial-node versus DAG-frontier waves | Same DAG/kernel policy/total resources | New inter-contraction concurrency |
| Unsliced versus sliced serial | Same circuit/query/precision, declared changed DAG | Decomposition cost |
| Sliced serial versus sliced concurrent | Same sliced DAG/total resources | Slice concurrency |
| Roundtrip versus resident segment | Same eligible DAG/precision/resources | Intermediate locality |
| Float32 versus shared-scale int8 | Fixed path and declared numerical contracts | Runtime/error trade-off |
| Greedy versus selected path | Frozen final executor/profile | Path selection |

The frozen sequential release already uses the WRAM kernel; do not relabel it as scalar naive. [S1] Reconstruct a small ablation-only scalar variant under the current boundary rather than resurrecting an obsolete runtime. An infeasibly slow full-circuit naive run may be replaced by an explicitly labeled smaller-instance or microkernel comparison.

Report sample counts, raw timing availability, uncertainty, kernel and host decomposition, H2D/D2H, estimated/measured local movement, useful work, padding, resource occupancy, numerical error and per-family results. Never add speedups multiplicatively across changing controls as though they were a single matched experiment.

## 13. Agent operation and ETH evidence protocol

Use one coordinator, a kernel implementer, and a scheduling implementer where disjoint software work is possible. An independent reviewer inspects correctness and scientific comparability. The coordinator owns shared protocol integration and scope decisions. These are instructions for the authorized execution workspace, not a claim that agents were launched during this planning review.

There is exactly one physical-hardware controller. All requests for ETH measurements enter its queue. Before every packet it verifies remote source/worktree, binaries, SDK, private lock, admitted rank/resources, other-user occupancy, CPU affinity/governor, writable durable evidence storage and a functioning retrieval route. Wait at most about 15 minutes for occupancy; do not interfere with other users or change system-wide settings.

Stop on a failed/unsupported/fallback execution; retain the partial packet and establish the cause. Numerical inaccuracy is separately recorded, not silently converted into an infrastructure failure. No cell replacement or evidence splicing. An independently established infrastructure incident may justify one complete new packet with a new identity.

A completed stage is accepted only after sorted relative checksums, complete immediate retrieval, local checksum and canonical verification, exact schedule/sample/session checks, portable archive and outer digest, and two verified copies. Never keep the only copy under temporary storage. Preserve physical-source versus reporting-source lineage. Simulator timings never determine physical speedups or fitting coefficients.

Use deep research only for unresolved consequential choices: source/toolchain capability, numerical equivalence, scheduling legality, memory/communication behavior, or a library-reuse decision. Routine coding, naming, and style do not need another literature review. Escalation produces a short supported decision, not new speculative research documents or broad scope changes.

## 14. Remaining goal text

> Complete only the cost-guided path-optimization phase against the accepted frozen UPMEM executor. Preserve the twelve family-aligned circuit instances, complete statevector query, numerical/resource contracts and existing evidence runner. Implement one metadata-only five-term launch-aware cost with fixed development-greedy scales, one serial Optuna TPE ask/tell route using exactly one fresh cotengra candidate per proposal, and the deterministic 1001-tuple fitting/batch procedure in Section 11.
>
> Prove that the whole-plan cost guides subsequent proposals, not merely reranking. Use one implementer, one independent reviewer and one hardware controller. Complete at most two feedback rounds within the existing 792 cap using the derived 768-attempt ceiling, then freeze all final G/F/R/U paths before timing. Archive, verify and report positive, neutral or negative outcomes without retuning. Do not reopen executor integration, parallelization, kernels, quantization, slicing, residency, transport or multi-rank work. Planning updates do not themselves launch execution.

## Sources

**S1 — Provided project material.** Uploaded `upmem-system-and-path-optimization-plan.md`, dated 4 September 2026; user-provided handoff and subsequent execution/roadmap excerpts. These establish reported project history and intended scope, not independently verified live state.

**S2 — Project README.** kazulak, *Masters / thesis / implementation*, public repository, accessed 5 September 2026. [Implementation README](https://github.com/kazulak/Masters/tree/main/thesis/implementation). Some status paragraphs are internally historical; no current branch qualification is inferred from them.

**S3 — Hardware characterization and distributed GEMV.** Juan Gómez-Luna et al., *Benchmarking a New Paradigm: An Experimental Analysis of a Real Processing-in-Memory Architecture*, arXiv:2105.03814; related IEEE Access publication, 2022. Sections 3.1, 3.3 and 4.2. [Paper](https://arxiv.org/pdf/2105.03814). Text inspected; GEMV page visually checked. Figures not used to extrapolate current ETH performance.

**S4 — Integer kernel code generation.** Krystian Chmielewski, Jarosław Ławnicki and Uladzislau Lukyanau, *UPMEM Unleashed: Software Secrets for Speed*, arXiv:2510.15927v1, 2025. Sections III-B–III-D. [Paper](https://arxiv.org/html/2510.15927v1). Compiler behavior is motivation for inspection, not confirmation about this project's binaries.

**S5 — UPMEM tensor-program scheduling.** *ATiM: Autotuning Tensor Programs for Processing-in-DRAM*, arXiv:2412.19630v2, 2025. Sections 2, 5 and 8. [Paper](https://arxiv.org/html/2412.19630v2). Supports joint attention to host distribution, DPU tiling and tasklets; does not validate this proposed TN scheduler.

**S6 — Slicing and work inflation.** Cotengra maintainers, *Tree Surgery*, live documentation accessed 5 September 2026. [Documentation](https://cotengra.readthedocs.io/en/main/trees.html).

**S7 — Sliced output semantics.** Cotengra maintainers, *Contraction* and `ContractionTree.gather_slices`, live documentation accessed 5 September 2026. [Contraction documentation](https://cotengra.readthedocs.io/en/main/contraction.html); [API reference](https://cotengra.readthedocs.io/en/main/autoapi/cotengra/core/index.html).

**S8 — Offline path proposals and annealing.** Cotengra maintainers, *Advanced Config*, live documentation accessed 5 September 2026. [Documentation](https://cotengra.readthedocs.io/en/latest/advanced.html). Verify the repository's pinned version before implementing API calls.

**S9 — Reusable benchmark sources.** CMU-SAFARI, *PrIM benchmarks*, repository README and license declaration, accessed 5 September 2026. [Repository](https://github.com/CMU-SAFARI/prim-benchmarks). Individual imported files still require review.

**S10 — ATiM environment and license.** SNU-CODElab, *ATiM*, artifact repository README and root license declaration, accessed 5 September 2026. [Repository](https://github.com/SNU-CODElab/atim). No compatibility with the ETH installation is certified here.

**S11 — Exact-structure inspiration.** Dongin Lee et al., *PIMutation: Exploring the Potential of PIM Architecture for Quantum Circuit Simulation*, ASP-DAC 2025; arXiv:2503.00668v1. Sections 4.2–4.4. [Paper](https://arxiv.org/html/2503.00668v1). Direct-statevector execution, not a ready-made TN kernel library.
