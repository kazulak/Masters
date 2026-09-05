# UPMEM thesis: kernel and hierarchical-parallel execution research plan

**Revision:** proposed v2, 5 September 2026.  
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

## 11. Schedule-aware path optimization

### 11.1 Update the cost representation

The six SLR-motivated features remain useful explanatory totals, but a DAG-parallel executor cannot be modeled adequately by summing independent kernel durations without representing overlap and waiting. Under the first synchronous-wave design, a proposed surrogate is

\[
\widehat T(p,q,r)=\sum_{w\in\mathcal W(p,q,r)}\left[
\widehat H_w+
\max_{d\in D_w}\left(\sum_{u\in U_{w,d}}\widehat K_u\right)+
\widehat G_w\right],
\]

where `H` includes wave preparation/input transfer, `K` the assigned DPU work, and `G` gathering/reconstruction/reduction. This is a model proposal, not a calibrated runtime law. Later real overlap or multi-launch subwaves require the model to reflect the actual schedule.

Extract actual or explicitly estimated movement, instruction/work, coordination and representation overhead from the chosen physical plan. Record live memory, padding and idle/wait time. Compare a grouped movement/compute/coordination model with the six-term version, but neither is excused from representing the scheduler. Use the simpler one when decisions and held-out behavior are materially equivalent.

Normalize feature scales from training data before interpreting nonnegative simplex weights. Correlated features do not yield uniquely identifiable physical constants. Keep fitted ranking coefficients distinct from measured hardware penalties.

### 11.2 Freeze first, then learn paths

The final executor includes its deterministic kernel, resource-allocation and scheduling rules. Changes in path can expose different kernels or frontiers without changing these rules. That is precisely why calibration should happen after the system freeze.

Keep optimized float32 as the primary equal-quality study. Int8 is a separate profile with independently calibrated cost data and declared accuracy eligibility or a runtime/error frontier. The handoff's reported Stress18 error prevents presuming int8 is equally accurate everywhere. [S1] No post-hoc tolerance or normalization is used to rescue a speed winner.

### 11.3 Bounded adaptive search

Use cotengra to generate or alter paths through its documented search/reconfiguration/annealing facilities. Qualify the installed version rather than assuming current online APIs match it. [S8] Lower candidates through the frozen scheduler, deduplicate physical choices, and run only a small batch of informative/promising candidates on ETH.

Separate offline exploration of cost weights from physical measurements of selected executable plans. Many weights choose the same plan; do not invoke UPMEM for every weight vector. Pin proposal seeds, temperature/acceptance schedule when using actual annealing, candidate count, three adaptive rounds at most, hardware attempts, and elapsed-time cap.

The existing plan's 540-attempt float32 ceiling can remain a template: 144 initial-training, 216 adaptive-training, 72 independent development-confirmation, 36 validation and 72 untouched-test attempts. Freeze the actual named workloads and schedule after P5. Do not launch this packet directly from the old names or presume that prior test instances are still untouched.

Use paired same-round controls and the objective

\[
J=\exp\!\left(\sum_j\pi_j\,\log S_j\right),\qquad
\sum_j\pi_j=1,
\]

where `S_j` is the preregistered repeated-block speedup and `pi_j` fixes circuit/family weighting. Track worst-cell regression, numerical eligibility, physical selected-path rank, regret, measured-pool headroom and search cost. Do not optimize the fastest individual observation or stop when a pleasing result appears.

Use session-inclusive execution as the primary one-simulation execution comparison, with its boundary stated exactly. Also report steady execution and full job time including construction/path selection/preparation where an end-to-end claim is made. Validation, reference calculation, hashing and reports remain outside the timed execution. Comparing against greedy establishes a gain over greedy; claims about optimizer superiority require an equal-budget target-neutral search comparator.

### 11.4 Holdout discipline

All kernel, scheduling, residency, slicing and path decisions consume development data. Repeatedly inspected validation is development, not a final holdout. A test is family-held-out only when the family has not influenced any of these stages; size-held-out is a different claim.

GHZ14/XOR18 from the old proposal remain untouched only after an exposure audit. If either has already influenced development, choose and freeze new instances before their timings are observed. If both numerical profiles share tests, freeze both before observing either profile's test performance. No weights, kernels, dispatch, schedules, candidates or acceptance criteria change after the final test. A neutral result completes the study.

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

## 14. Replacement goal text

> **Objective:** Complete the UPMEM execution-system development milestone before final contraction-path optimization. Reconcile the actual repository and reuse already integrated/qualified work. Implement and independently audit minimal kernel dispatch, one-launch four-product complex execution and a census-selected geometry specialization, plus genuine static dependency-ready DAG-wave execution across disjoint DPU groups. Preserve existing tasklet/tile parallelism, exact untruncated full-statevector semantics, packed transport, numerical policies and provenance. Evaluate resident subgraphs and exact slice concurrency through the bounded probe in this plan; retain only supported, qualified improvements. Do not add CPU/GPU contraction placement or a generic scheduling framework.
>
> Use bounded software agents for independent work and review, with one owner for protocol integration and exactly one ETH hardware controller. Complete fixed-path, fixed-resource physical comparisons and durable evidence retrieval before accepting each mechanism. Distinguish implementation correctness, numerical quality, and measured benefit; a negative result is acceptable. Freeze kernels, dispatch, scheduling, numerical rules, memory policy, resource profiles and cost extraction before the final bounded physical-feedback path search. Then freeze the pretest profile, run untouched evaluation without retuning, and release source/evidence. Use deep research for unresolved consequential technical questions only. Do not silently move the named kernel and DAG-parallelism deliverables back to future work; document any scope change and its evidence.

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
