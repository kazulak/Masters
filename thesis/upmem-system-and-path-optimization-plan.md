# UPMEM thesis: execution-system development and path optimization plan

Date: 5 September 2026. Status: gated roadmap. This revision changes documentation only, not executable code, configurations, candidate pools, numerical policy, or physical evidence. Proposed mechanisms below are not implemented or qualified by this document.

## Decision

Complete and freeze the execution system before the final contraction-path optimization. The execution system includes kernel dispatch, packing/layout decisions, numerical policies, tasklet/DPU configuration rules, planned CPU placement, and host work. A path must be scored after lowering through that system, because the same contraction can have very different execution costs under different kernels and placements.

Use a gated roadmap and one shared execution system. Quantization is selected on or off through the existing `numeric_policy`; it is not a second runtime. Every mechanism retained after a gate must support both float32 and the existing shared-scale int8 policy in that executor. A mechanism that only works for one policy is a rejected or deferred probe. Different widths, arithmetic helpers and admission bounds are necessary policy choices, not two execution architectures.

1. A, integration adoption: keep the existing integration gate unchanged; pass the pending seven-route physical gate, then adopt the published integration checkpoint.
2. B0, census and policy: use fixed development paths to measure shape, cost, resource, and policy facts.
3. B1, complex batching: evaluate one complex operation envelope carrying both input planes. One DPU launch per wave is a separately versioned kernel/ABI change, not a property of packing ABI-v4 requests together. Stop this probe if that change exceeds its bounded correctness/engineering gate.
4. B2, planned CPU placement: add deterministic first-class CPU placement through the existing policy-correct CPU route, with BLAS only when qualified.
5. B3, one further improvement: choose exactly one distinct measured improvement from the B0 census, such as skinny/GEMV/outer work or synchronous bulk SDK copies.
6. B+, resource scaling: measure real WRAM/IRAM use and actual tile, work, and admission facts for the declared one-rank resource matrix.
7. System qualification and freeze: freeze only mechanisms that pass their individual gates and then begin the path study.
8. C, path study: run the bounded adaptive physical-feedback study under the frozen executor and policy contracts.

“Best” means best observed under a declared workload, resource set, timing scope, and numerical-quality contract. It does not imply maximum DPU/tasklet counts or int8 for every circuit.

Residency and multi-rank execution are separate, gated optional extensions and are not required for the path freeze. Slice concurrency and concurrency between independent semantic DAG branches are outside this phase. The previous 192-attempt calibration is superseded by the new declared path study; preserve its frozen files unchanged and record the supersession separately.

## 1. Verified starting point

The durable working location for this roadmap is `/home/tom/repos/Masters/.agent-work-execution-integration-v1`.

| Role | Branch | Full source SHA |
| --- | --- | --- |
| Current accepted main | `main` | `fa0dedf628a3612371daa4f6502da4d5465bbaff` |
| Quantization reporting input | `feature/upmem-quantized-physical-v1` | `62505ae637bdd3cf963b70f61754bf90a658b527` |
| Path-infrastructure input | `feature/upmem-path-heuristic-generalization-v1` | `e225947a84f937629ed46003dad7d8edff160a8f` |
| Published integration, not yet adopted | `feature/upmem-execution-integration-v1` | `b921b8804e324da75222354ee2f4df41e770b75c` |

Both input branches descend from the accepted main; the integration preserves their histories. Historical execution/reporting identities remain distinct. A documentation-only descendant does not replace the qualified `b921b88` execution source for the pending gate.

| Item | Current status |
| --- | --- |
| Published integration checkpoint | `b921b8804e324da75222354ee2f4df41e770b75c` |
| Software and CI | 1088 tests passed; Ruff green; exact-head CI green |
| SDK qualification | One corrected 14/14 sample/session run; raw evidence retained in two independently verified copies |
| Physical integration gate | Seven-route gate pending |

The published checkpoint is the input to gate A, not evidence that `main` already contains the integration. The [integration checkpoint](implementation/docs/upmem_execution_integration_v1.md#qualified-execution-checkpoint) records exact experiment/run identities, CI and archive hashes. SDK policy replay passed for both policies; float32 accuracy passed, while the int8 observations are not qualified under the float32 full-precision accuracy criterion. Do not conflate those gates.

The local SDK is 2025.1.0. The ETH environment is recorded as SDK 2023.1.0. `pkg-config` 0.29.1 is a `pkg-config` tool version, not an SDK version, and must not be used as ETH SDK provenance. Cross-environment compatibility is a qualification question, not an assumption.

The implementation already canonicalizes binary contractions to batched `(B,M,K) @ (B,K,N)` in `src/quantum_bench/upmem/tiling.py`. The active DPU implementation already performs blocked matrix contraction using a shared WRAM B panel, private tasklet buffers, and multi-DPU tile waves. Future gated work extends this lowering; this documentation task does not implement it.

The current sequential release uses the WRAM kernel. It must not be labeled the old scalar-MRAM naive implementation.

The adopted local contract is documented in the [packed-operation adoption document](implementation/docs/packed_operation_transport_adoption.md): deterministic full pre-measurement statevector simulation, exact untruncated contraction, one `ContractionDAG` and one `UpmemPlan` per sample, `packed_operation_v1`, embedded ABI-v4 requests, the WRAM-panel DPU kernel, host-roundtrip intermediates, and one physical rank for the current route.

The accepted packed-transport Stress18 diagnostic baseline is:

| Route | Total-wall median (s) | Kernel median (s) |
| --- | ---: | ---: |
| 1 DPU x T1 | 27.475147 | 24.332646 |
| 4 DPUs x T8 | 4.275410 | 0.977901 |

Thus the endpoint comparison is 6.43x total-wall speedup and 24.88x kernel speedup. At the 4-DPU x T8 endpoint, the non-kernel share is `1 - 0.977901 / 4.275410 = 77.1%` of total wall time. This ratio of component medians is descriptive, not a paired speedup estimate or a Python-CPU measurement. It includes transfer and other non-kernel work and is not necessarily removable. The suggested 4.9x/84% figures are not this table's comparison; do not mix sources or promise their elimination.

DPU count in this plan means DPU count within the selected one-rank route. A DIMM, a rank, and a channel are distinct topology facts and must be recorded and reported separately rather than used interchangeably.

Source audit at `b921b88`: `implementation/src/quantum_bench/upmem/runtime.py` iterates `rr`, `ii`, `ri`, `ir` in `_execute_complex_core_unlocked`, submitting all waves separately for each lane. `implementation/native/upmem/runtime/host.c` launches once per embedded request. The C host is already persistent. Today there are four lane-envelope submissions per complex contraction and four launches per active wave, not four process starts. The v4 control in `native/upmem/runtime/protocol.h` has one A/B/C transfer description. Packing those requests without a kernel change reduces submissions but does not reduce launches.

The same source audit confirms serial per-DPU copies in `host.c`, host split-K assembly and complex decoding in `runtime.py`, session non-reentrancy guards, and explicit single-rank checks. These are concrete boundaries for the experiments below. Source links: [runtime](https://github.com/kazulak/Masters/blob/b921b8804e324da75222354ee2f4df41e770b75c/thesis/implementation/src/quantum_bench/upmem/runtime.py), [host](https://github.com/kazulak/Masters/blob/b921b8804e324da75222354ee2f4df41e770b75c/thesis/implementation/native/upmem/runtime/host.c), [DPU kernel](https://github.com/kazulak/Masters/blob/b921b8804e324da75222354ee2f4df41e770b75c/thesis/implementation/native/upmem/runtime/dpu.c), [numerics](https://github.com/kazulak/Masters/blob/b921b8804e324da75222354ee2f4df41e770b75c/thesis/implementation/src/quantum_bench/numerics.py).

## 2. Gated roadmap and merge sequence

The integration branch exists; the remaining branch names are proposed. A failed bounded probe is a valid completed gate, not a reason to keep extending the phase.

| Phase | Proposed branch | Contents and completion gate |
| --- | --- | --- |
| A. Integrate | `feature/upmem-execution-integration-v1` | Histories integrated; software/CI/SDK passed at `b921b88`. Complete the unchanged seven-cell physical gate before adoption. |
| B0-B+. Shared execution system | `feature/upmem-kernel-dispatch-v1` | Start from adopted integration. Census, separately gated batching and CPU placement, one further measured mechanism, useful one-rank scaling, matched ablations. |
| Optional ATiM spike | `feature/upmem-atim-kernel-probe-v1` | Isolated one-day ceiling within the shared optional budget; import only a qualified kernel or schedule. |
| Optional residency or multi-rank probe | Separate branch only after its GO decision | One bounded mechanism, both-policy parity, no generic scheduler. Neither is required for system freeze. |
| C. Final path study | `feature/upmem-final-system-path-search-v1` | Start only from the frozen kernel-system tag. New study identity, physical features, profiles and adaptive-search protocol. |

Merge each accepted milestone into main before starting the next production phase. Keep rejected probes separate with a short decision and evidence. Preserve released source commits/tags; do not rebase or rewrite them. Do not develop the standalone repository in parallel.

The four overlapping files in the completed input merges were `native/upmem/runtime/simplepim_provider.c`, `src/quantum_bench/cli.py`, `tests/test_cli_report.py`, and `tests/test_upmem_protocol.py`. Preserve the reviewed merge semantics:

- Preserve explicit simulator-backend selection and simulator virtual-rank handling, while physical admission still enforces the requested one-rank topology.
- Preserve persisted failure records and immediate physical collection termination.
- Preserve canonical `complex_int8_shared_scale_v1`, prequantization float64 handling, and the repaired complex accumulation bound `2*K*127^2` where required by the contract.
- Preserve path-specific validation and workload hashing.
- Retain the original float32 study as historical preregistration. Its old physical-plan identities and weights do not qualify the integrated int8 system.

The corrected SDK gate covered Bell2 and Stress14 across seven routes, 14 one-shot samples/sessions. The pending physical gate is Bell2 float32/int8 at 1D/T1; Stress14 float32/int8 at 1D/T8, float32 at 4D/T8, and int8 at 3D/T8 and 4D/T8. It has seven attempts, no warmups and no timing claim. Keep `configs/tn_benchmark_upmem_execution_integration_physical_v1.yml` unchanged. Do not repeat completed software/SDK qualification for this documentation revision or the old 180-sample quantization campaign merely because branches were merged.

## 3. Thesis comparison ladder

Use a cumulative development story plus controlled ablations. Historical releases provide chronology; matched experiments establish the cause of speedups.

| Variant | Path | Precision | Resources | Purpose |
| --- | --- | --- | --- | --- |
| Scalar naive contraction | Fixed reference | Float32 | 1 DPU × T1 | Establish a simple arithmetic/memory baseline. |
| Existing WRAM-panel kernel | Same | Float32 | 1 DPU × T1 | Isolate memory staging/blocking. |
| WRAM kernel with tasklets | Same | Float32 | 1 DPU × T8 | Isolate intra-DPU parallelism. |
| WRAM kernel across DPUs | Same | Float32 | 4 DPUs × T8 | Isolate the second parallelism level. |
| Batched complex execution, if accepted | Same | Each declared policy separately | Same route | Isolate submission/launch reduction, without also changing placement. |
| Explicit CPU/UPMEM placement, if accepted | Same | Each declared policy separately | Same allowed resources | Isolate placement, including CPU threads and policy replay. |
| One further measured improvement | Same | Each declared policy separately | Same route | Isolate a geometry specialization or synchronous bulk transfers. |
| Useful larger one-rank configuration | Same | Same declared policy | Named tasklet/DPU counts | Establish the actual kernel and total-time scaling curves. |
| Optional residency or multi-rank | Same | Same declared policy | Explicitly qualified topology | Separate experiments, not assumed cumulative gains. |
| Quantized execution | Same | Shared-scale int8 | Same route | Measure representation cost and error together. |
| Tuned contraction path | Selected path | Same final profile | Same topology or frozen selection rule | Isolate path optimization. |

For the naive comparison, use a small scalar kernel variant under the contemporary packed transport and measurement boundary. Keep it confined to ablation builds or an isolated tag. Do not restore an obsolete runtime or protocol. Where a naive full-circuit run is impractically slow, report a matched smaller instance or kernel microbenchmark; do not invent missing full-circuit timings.

For every mechanism comparison, hold DAG/path, tensor values, precision, topology and timing scope fixed, except the factor under test. After that comparison, report the cumulative accepted system separately. Kernel speedup alone does not establish simulation speedup. A scalar comparison is historical or a bounded ablation, not a requirement to create another production kernel before the host work.

## 4. One shared execution system

Extend the current flat plans and direct functions. Avoid a kernel registry, plugin framework, general scheduler or new executor hierarchy.

The core flow is:

`classify contraction → enumerate the few supported implementations → choose using frozen rules → record the physical decision → execute → validate`.

A per-operation decision needs only fields actually used: semantic operation kind, B/M/N/K geometry, layout/permutation requirements, kernel ID, numeric policy, tile parameters, and placement. Extend the existing physical-plan model, not the semantic DAG or a second planner IR. Reuse existing topology and identity records.

| Existing area | Necessary change |
| --- | --- |
| `upmem/tiling.py` | Reuse canonical geometry; expose packing/layout facts and precise specialization predicates. |
| `upmem/plan.py` | Represent per-operation decisions and validate deterministic selection. Include decisions in physical identity. |
| Native protocol, `dpu.c`, runtime | Only after a mechanism's GO: explicit selector/representation and matching completion identity, retaining the persistent host and packed-operation architecture. |
| `upmem/runtime.py` | Emit actual kernel/movement/launch facts; remove assumptions that every operation uses the one WRAM mechanism. |
| `numerics.py`, `cpu.py` and validation | Reuse pure policy-specific encoding, product accumulation and decoding helpers; replay the declared arithmetic and reduction policy. |
| `upmem/path_heuristic.py` | Extract features from the resulting executed plan, including selected kernels and host work. |
| Experiment/reporting code | Record per-kernel time, placement, bytes, packing, launches, numerical error and complete execution time. |

Prefer one DPU binary containing the small dispatch set if instruction and WRAM capacity allow. Existing sessions assume one DPU binary; loading different binaries for individual contractions introduces costs that must be justified and measured. Both numerical policies use that execution architecture; retain only necessary width/scale/accumulation branches.

Phase A freezes ABI-v4 and the existing kernel. A later one-launch complex kernel, multi-plane layout or resident-state protocol is explicitly a new execution contract. Require a separate design/qualification gate and version any changed wire semantics; never reuse reserved v4 fields silently. If the effort or exact numerical/evidence contract cannot be met, keep the v4 baseline and record NO-GO. Keep one active reader after adoption; historical commits supply A/B controls, not permanent duplicate runtimes. Do not introduce a new public evidence schema as an incidental optimization; stop for an explicit contract review if existing facts cannot represent the change honestly.

Keep `logical_plan_id` stable when the circuit and contraction DAG are unchanged. Kernel, numeric representation, layout, tiling or placement changes must affect physical/executable identity. A future transformation that changes the tensor network or DAG requires a new logical identity too.

The current feature extractor assumes the panel kernel, four real products and particular launch/movement counts. Updating those facts is a correctness requirement for kernel-aware path ranking, not optional reporting polish.

## 5. Host-cost reduction before more kernels

### B0: fixed-path census and removable-cost bound

Use Stress16, HS20, EDC14 and the already observed BV18, with fixed greedy paths and a small predeclared alternative-path set. Freeze definitions before timing and exclude final test instances. Both policies must be represented. Record B/M/N/K, operand/intermediate sizes, K chunks, output tiles, active/idle work units, waves, contiguous/aligned bytes, WRAM/IRAM demand and host peak memory.

Separate Python/native CPU work from waits: materialization/copying, encoding, hashing/serialization, envelope/filesystem staging, submissions, launches, H2D/D2H, output files, host reduction and reconstruction. The persistent native process count is not the command count. Use existing counters/timestamps and optional external profiling before adding timers. Do not sum nested timers or label all non-kernel time Python overhead.

Produce a per-operation cost/count table and a source-level dependency map. For a measured affected fraction `p` and equivalent prototype speedup `s`, compute `S_max = 1 / ((1-p) + p/s)`. Include setup and all moved work. This bounds the complexity budget; it is not a physical performance prediction without an A/B.

### B1: complex batching, with explicit contract boundaries

The candidate input is one operation envelope carrying `A_real`, `A_imag`, `B_real`, `B_imag` plus the unchanged logical work ordering. There are two distinct effects:

- Transport batching can reduce four lane submissions to one while still executing four ABI-v4 requests per wave. It does not itself fuse launches or arithmetic.
- One-launch execution requires a versioned multi-plane request/control and a kernel that performs the four real products within one launch per wave. Qualify it as a kernel/ABI change. Preserve per-product reduction order and numeric boundaries initially, returning the same four product planes for host assembly.

Select and preregister one bounded intervention from the census; do not count submission batching and device output combination as one unexamined optimization. Fewer launches/submissions are count targets, not a guaranteed 4x host or wall-time improvement.

Device combination `R = RR - II`, `I = RI + IR` is a further gated subcase, not required for B1 acceptance. A full-K output tile with matching arithmetic may permit two output planes instead of four, halving that product readback only. For split-K, combining before host accumulation changes float32 association; do not assume equivalence. Int8 must preserve safe product/complex bounds, shared scales and final decode. Retain four planes or reject the subcase where these conditions cannot be proved. Do not introduce a third numeric policy to force adoption.

Recalculate aligned MRAM layout, WRAM buffers, tasklet stacks and IRAM for both policies. The current 512 KiB protocol pool and real A/B/C layout do not automatically fit four operands plus multiple outputs. A 256x256 output tile is not a 256x256 WRAM panel. If the versioned design cannot fit the B1 timebox, record NO-GO and continue with the accepted executor.

### B2: first-class, policy-correct CPU placement

Measure the complete crossover between CPU contraction and UPMEM execution, including packing, encoding, transfer, launch, reduction and reconstruction. Do not choose an arbitrary 2x2 or 8x8 size threshold. Derive one transparent deterministic placement rule from development geometry, numeric policy and configured topology; freeze it before held-out timing.

Use the existing CPU policy replay helpers as the starting point. Ordinary complex64 BLAS is not shared-scale int8 execution. CPU int8 work must reproduce float64 prequantization, scale selection over the complete operand planes, integer accumulation bounds and scale-aware decoding. Float32 BLAS requires declared replay/accuracy qualification and must not be claimed bitwise-equivalent merely because the mathematical contraction is identical. Unsupported policy-correct CPU geometry remains planned UPMEM work, not silent fallback.

Record placement in the physical plan and execution facts, including CPU/DPU contraction count and time. Pin one CPU thread for matched initial comparisons and count it as part of the heterogeneous system. Preserve device-only historical controls. Use direct functions and a flat per-operation decision, no scheduler hierarchy, dynamic load balancer or mixed-policy escape route.

### B3: choose exactly one further measured improvement

After B1/B2, reassess the remaining cost on the same development paths. Choose one of:

| Candidate | Admission and measurement gate |
| --- | --- |
| Skinny GEMM/GEMV/DOT, or outer product for K=1 | One exact geometry predicate, both-policy coverage, generic kernel retained; demonstrate a material full-route contribution, not only a microkernel win. |
| Synchronous bulk SDK transfers | Current host copies operands and results DPU by DPU. Check SDK-compatible `dpu_prepare_xfer`/`dpu_push_xfer` grouping, alignment, sizes, ownership and completion. Compare with the serial-copy baseline; do not also add asynchronous overlap. |

Choose the higher measured benefit per implementation cost, or neither. Do not combine a new kernel, serializer change and output-transport replacement in this A/B. Census findings about output-file overhead remain documented if not selected. Exact structural predicates must not infer zeros or permutations approximately. Avoid a general dispatcher for variants that never survive qualification.

### Common gate for every retained mechanism

Require deterministic software correctness, full pinned tests/Ruff/CI on the changed source, strict SDK correctness, a sparse physical pilot, then one preregistered matched A/B. Both policies need coverage, but int8 error/eligibility is reported separately from float32. Freeze total attempt and hardware-time caps per experiment before launch; do not silently duplicate the Phase C budget or rerun until positive.

Report session open, steady execution, session close, `attempt = open + steady + close`, plan/preparation cost, kernel, transfers, reconstruction, counters and peak memory. Use paired-block differences/ratios and MADs, not unrelated medians to infer propagation. Preregister the acceptance threshold from prior variability, removable work and the prototype bound before observing optimized physical timings. A component-only improvement or relocated work is insufficient; report persistent-session break-even `extra_setup / per_run_saving` only when reuse is real.

Mandatory fixtures cover complex values, zeros, odd dimensions/alignment, split-K, boundary tiles, idle tasklets, partial waves, non-divisible work, T3/T7/T12/T24 and 3D/T8. Check numeric bounds including `2*K*127^2`, deterministic output ordering, tasklet-specific binary identity, no omitted/duplicated work, and no fallback. Preserve session non-reentrancy, exact failure/partial-completion facts, timeout handling and cleanup. Device code changes also require all T1-T24 builds and actual WRAM/IRAM admission. No exact-time unit tests.

## 6. External work: ATiM and PIMutation

ATiM is a public search-based tensor compiler with a modified TVM stack. Its documented artifact environment uses Ubuntu 20.04 and SDK 2021.3.0; compatibility with ETH SDK 2023.1.0 is unverified. Its root repository uses Apache-2.0. [Official artifact](https://github.com/SNU-CODElab/atim), [archived artifact](https://zenodo.org/records/15379924).

The paper studies joint host/DPU scheduling and data movement for tensor programs, including matrix-vector and tensor-vector workloads. It does not establish a ready-made complex GEMM implementation for this thesis. [ATiM paper](https://arxiv.org/html/2412.19630v2).

The supplied tuning driver uses int32, while the evaluation support can record generated UPMEM C and can exclude transfer terms in some benchmark settings. Therefore, generated code must be tested through the thesis runtime and timing contract. Shared-scale int8 inputs with int32 accumulation require explicit verification. [Tuning driver](https://github.com/SNU-CODElab/atim/blob/artifact/evaluation/atim_autotune.py), [runtime/evaluation support](https://github.com/SNU-CODElab/atim/blob/artifact/evaluation/base.py).

Proposed ATiM spike: at most one engineering day and 32 physical candidate configurations for one representative GEMV or related supported tensor program, within the shared optional budget in section 12, not an extra day. Select it only when the B3 census supports that operation. Pin the external source; build in an isolated environment without changing ETH's system SDK. Verify generated C, both-policy arithmetic, one-rank admission and compatibility with the existing host boundary. Account for repeated timing attempts separately from the 32 configurations. Stop at the cap and keep the existing/manual specialization if integration fails or full-route benefit is absent. Do not use the artifact's default large tuning campaign.

PIMutation uses direct statevector simulation with gate merging, row swapping and partitioning of separable state representations. Useful inspiration is to exploit exact structure and locality. Its direct gate-evolution implementation is not interchangeable with a TN contraction backend. Public implementation code and a code-reuse license were not verified in this review. [PIMutation paper](https://arxiv.org/html/2503.00668v1).

A tensor permutation/diagonal specialization is a narrow way to test the relevant idea while retaining the TN algorithm. Global gate fusion, statevector partitioning or a different integer representation would change the scientific comparison and should be separate later work. Preserve attribution, exact upstream versions and applicable notices for any incorporated code.

## 7. Useful scaling and optional locality extensions

### B+: one-rank scaling first

The implemented parallelism partitions one contraction's output work across tasklets and DPUs. Semantic DAG stages still execute in dependency order. Increasing resource counts does not add independent DAG-branch or slice concurrency.

| Tier | Bounded progression | Required evidence |
| --- | --- | --- |
| Intra-DPU | T8, T12, T16, T24, retaining a T1 reference | Actual compiled WRAM/IRAM plus stacks, row ownership, panel-loader activity, barrier cost, idle-tasklet fraction, kernel and attempt time. T16 is a candidate, not an assumed optimum. |
| Inter-DPU, one rank | D1, D2, D4, then D8, D16, D32, D64 when admitted | Physical inventory, eligible tile/work count, partial-wave utilization, operand duplication, transfer time, host memory and full-route scaling for fixed work. |
| Multi-rank, optional | First two ranks, then at most four only after a separate GO | Rank-local mapping/ownership, complete output coverage, actual channel topology, per-rank and total timings, coordinated failure/cleanup. |

Predeclare a sparse fixed-work ladder from software plan census and safe memory limits; do not time every tasklet/DPU/policy combination by default. Stop a resource extension when insufficient work, admission failure or the effort cap makes it uninformative. Record infeasibility rather than manufacturing a larger circuit after timing. A weak scaling experiment with a growing problem is a separate comparison, not a fixed-work speedup.

Qubit count alone does not establish a 2048x2048 or 4096x4096 contraction. For an actual output plane, `ceil(M/tile_M) * ceil(N/tile_N)` counts output tiles before batch and split-K factors; it does not prove admission or useful occupancy. For example, 2048x2048 with 256x256 output tiles gives 64 tiles arithmetically, but full-K operands may exceed the declared MRAM pool and require additional work units/waves. Derive geometry from the real plan, not an illustrative diagram.

Do not equate one rank with one DIMM/stick or an independent memory channel, and do not call the host-DPU memory path PCIe without evidence. Record detected usable DPUs per rank; 64 is a target upper configuration, not guaranteed available hardware. Measure transfer contention and NUMA placement before promising linear multi-rank bandwidth. There is no universal pipeline argument that makes T16 or a quoted 35 KiB footprint optimal for all layouts/policies.

### Optional R: bounded two-step MRAM residency

This is not supported by the current host-roundtrip/v4 route. It needs explicit resident-buffer ownership and execution identity, not removal of a host copy alone. Consider one producer and one immediate, single-consumer contraction on the same DPU only: complete local output, compatible layout, no cross-DPU or split-K reduction dependency, bounded MRAM lifetime and deterministic release after the consumer.

Retaining an int8 product buffer is not equivalent to retaining the next operand. The accepted policy decodes intermediate products and chooses a new shared scale over complete real/imaginary operands before reuse. Prove identical scale selection, rounding, bounds and replay facts without an unavailable global reduction, or reject that geometry/probe. Float32 accumulation/complex combination must likewise preserve the declared boundary. No implicit tensor-residency graph, changed numeric policy, global cache or skipped evidence checks.

GO only if both-policy semantics and retained-memory bounds are demonstrable within the optional timebox and B0 shows meaningful eliminable movement. Measure actual H2D/D2H saved, extra setup and session-inclusive benefit. Test stale handles, operation/session isolation and failure cleanup. Otherwise retain host roundtrips and freeze without residency.

### Optional M: independent ranks, not a shared-session thread patch

First test static disjoint output-tile/row partitions of one contraction. Each rank owns its persistent native process, local DPU map, request sequence, buffers, completion records and cleanup. Define deterministic global tile identities and ordered host reconstruction; duplicate operand movement and host reductions remain costs. Prefer complete-K ownership where admitted; preserve existing split-K semantics otherwise.

Removing `len(rank_paths) != 1` is insufficient: packed submission, the SimplePIM provider, plan mapping, identities and resource validation all enforce rank-local contracts. A bounded thread pool may submit concurrently to distinct rank sessions only after those contracts are implemented; never re-enter one session or share its pipes/sequence state. Acquire the private campaign lock and all requested ranks before work, roll back partial allocation, and retain partial-completion facts when any rank fails. Fail the whole sample without retrying completed work.

Require both-policy SDK and physical correctness, including non-power counts and partially filled ranks, before multi-rank timing. No inference that independent ranks have independent DDR bandwidth. Keep this probe outside the required one-rank freeze; starting it requires a separate GO and attempt budget.

### Deferred concurrency

Independent DAG branches, parallel slices, asynchronous SDK overlap and dynamic work stealing remain separate later milestones. They require dependency/lifetime/reduction semantics beyond output-tile partitioning. No generic scheduler or recursive thread-pool orchestration is introduced here.

## 8. Freeze the system before path training

Proposed tag: `thesis-upmem-kernel-system-v1`.

Freeze source, dependencies/SDK facts, host/DPU/init binaries, supported kernel set, dispatch thresholds, layouts, numerical rules, resource choices, CPU-thread settings when relevant, cost-feature extraction and measurement scopes. Qualify complete statevector reconstruction and path-specific arithmetic replay.

A deterministic dispatch policy may choose different kernels for different shapes after the freeze. That is expected. Freeze the selection rule and its calibration data; do not retune it during path training.

Keep two explicit profiles:

- Optimized float32, the primary equal-quality path study.
- Quantized int8, a separate policy/profile with accuracy gates or a reported runtime/error frontier.

The reported Stress18 int8 relative L2 error of 8.316% prevents assuming it is an equal-quality replacement. Path changes can change quantization error. Check the output of every proposed path against the declared quality contract before including it as an admissible speed winner. If no defensible error threshold exists, retain float32 as the primary comparison and report quantized trade-offs separately.

A single frozen implementation must support both policies; performance profiles need separate calibrated weights and quality eligibility. If both profiles use the same final test instances, freeze both profiles before observing either profile's final test timings. A later int8 study must otherwise declare the earlier exposure and use new untouched instances for a new generalization claim. A mixed-precision placement policy is outside this roadmap; it must not be introduced as an implicit correction.

## 9. Cost model for the final dispatcher

Continue using the SLR terms, but calculate them from the selected physical implementation. Include kernel-dependent compute, actual or defensible estimated H2D/D2H and MRAM/WRAM movement, packing/permutation, launch/wave/barrier counts, host reductions and numerical execution overhead. Include CPU operation/transition costs if placement is enabled.

A small piecewise lookup or regression over kernel and geometry is a reasonable candidate when the original aggregate linear terms cannot distinguish kernels. Start with the grouped movement/compute/coordination model and a small number of justified kernel distinctions. Compare additional complexity using training robustness and validation; do not create a large learned model before data show the need.

Execution cost `C(c,p,q,r)` and numerical error `E(c,p,q)` remain separate. The SLR numerical term represents execution overhead, not approximation error. Changing the dispatcher invalidates old fitted coefficients, even when some logical candidate paths remain reusable.

Measure complete execution to correct approximations. A sum of kernel microbenchmark times alone misses host boundaries and interactions. Count four-real-product work even if launches are fused; count host transfers only where they actually occur. CPU-placed work is not DPU arithmetic or transfer. If residency is rejected, retain the host-roundtrip model. Any accepted residency/multi-rank mechanism must be qualified and represented in features before system freeze, never inserted during path training.

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

ETH access was checked read-only on 5 September 2026. The exact clean `b921b88` remote checkout was available, but all ranks were owned while another user's workload ran. The physical integration gate remains pending. Fresh occupancy/lock/provenance checks are required before execution; neither this plan revision nor a read-only availability check authorizes bypassing admission or launching a later phase.

## 12. Effort limits and immediate backlog

The execution-development cap is ten focused engineering days, excluding hardware queues and access setup. These are ceilings, not a promise that all probes pass. The path study retains its separate finite attempt/time budget. Finish or reject each mechanism within its allowance; do not spend qualification/closure time on another optimization.

| Work package | Proposed effort cap | Deliverable |
| --- | --- | --- |
| A: remaining integration closure | 0.5 day | Unchanged seven-cell physical gate, retained evidence and adoption |
| B0: census and shared-policy audit | 1 day | Fixed paths, boundary/count table, policy and memory bounds |
| B1: complex batching gate | 2 days | Versioned design only if justified; exact semantics and independent A/B, or NO-GO |
| B2: CPU placement gate | 1 day | Deterministic policy-correct crossover, or NO-GO |
| B3: one further mechanism | 1 day | One specialization or synchronous bulk transfers, or neither |
| B+: useful one-rank scaling | 1 day | Sparse admitted tasklet/DPU curves, utilization and host-floor data |
| Shared optional allowance | 1 day total | At most one ATiM, residency or multi-rank feasibility probe; no entitlement to production integration |
| System qualification and freeze | 1.5 days | Both-policy regression, exact source/binaries, ablations and retained evidence |
| Execution total | 10 days | Freeze accepted mechanisms; discard unjustified complexity |
| C: physical-feedback path study | Separate fixed attempt/time budget | Verified rounds, pretest profile, independent confirmation and untouched evaluation |
| Final reporting/archival closure | 0.5-1 day after C | Thesis tables, verified raw evidence, tags and release |

Do not make the thesis depend on every proposed mechanism. A successful bounded result is qualified integrated numerics in one executor, documented GO/NO-GO decisions, useful measured tasklet/DPU configurations, matched evidence for retained improvements, one frozen float32 path study and honest int8 error/performance results. Neither T16/64-DPU wins, fused output, residency, multi-rank nor a new specialized kernel is required to be positive. Optional integration that exceeds the remaining cap is deferred, not silently added to the critical path.

The next execution task is the pending Phase A physical gate on clean `b921b88`, not another source merge or qualification rerun. Then proceed to B0 from adopted integration. Do not begin physical path annealing before the system freeze. Export the standalone repository after accepted milestones, with source and evidence provenance.

Use at most two implementation workers with disjoint file ownership, one bounded read-only auditor, and one exclusive hardware controller. Runtime/protocol/kernel changes are not independent when they share a wire contract; define that contract before dispatching workers. Stop at the milestone caps, including on neutral/negative results. No autonomous expansion to another optimizer, numeric format, scheduler, multi-rank mode or final 2+30 campaign.
