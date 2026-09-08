# UPMEM Kernel and DAG Execution System v1

## Controlling Objective

The [v2 research plan](../../upmem-system-and-path-optimization-plan-v2.md)
supersedes the narrowed roadmap in the historical integration/preparation
reports. Complete kernel and static DAG-wave experiments before freezing the
executor and calibrating final paths. A neutral physical result is valid;
omitting the named kernel or DAG prototypes is not completion.

Required mechanisms are direct kernel dispatch, one-launch RR/II/RI/IR with
separate product outputs, one census-selected geometry specialization, and
dependency-ready nodes running on disjoint DPU groups in one native launch.
Retain tasklet/tile parallelism, packed transport, exact untruncated full
statevectors, declared float32/int8 policies and deterministic reconstruction.
CPU/GPU placement, multi-rank, async overlap and new numeric formats are excluded.

Float32 is the primary performance profile. Int8 requires policy replay,
resource correctness and honest accuracy reporting, not a duplicate performance
campaign. Review unfinished core work at engineering day seven; do not silently
remove deliverables. The resident/slice probe has a separate three-day cap.

On 2026-09-07 the user explicitly reaffirmed this expanded v2 scope after P0
acceptance. The narrower scheduled-continuation text is not the downstream
roadmap. Kernel/DAG mechanisms remain required bounded experiments, not assumed
speedups or prerequisites for indefinite runtime optimization.

The user-approved autonomous execution model supersedes the older Phase A-only
authorization. Continue between passed gates within the declared scope and
budgets without phase-by-phase approval. Make bounded repairs and accept
negative results; freeze identities for reproducibility, not as a user-input
pause. Keep exclusive hardware admission, no replacement samples, untouched
holdouts and two-copy archival. Tool-enforced permissions and safety limits
still apply; do not bypass a rejection or expand budgets implicitly.

## Reconciled Sources

Remote heads were checked on 2026-09-05 before implementation:

| Role | Exact source |
| --- | --- |
| Published main | `fa0dedf628a3612371daa4f6502da4d5465bbaff` |
| Integrated execution; physical gate accepted 2026-09-07 | `b921b8804e324da75222354ee2f4df41e770b75c` |
| Integration reporting head | `5b93f87c1a034944859348c99e2fe263961a2114` |
| Host-only preparation execution | `56b159dc7e8cd945265a6e02dfb5e7c74edf381a` |
| Clean v2 branch predecessor | `18556c3c9b6fb7c5db13c93fb0e253f22eeb3337` |

The implementation branch is `feature/upmem-kernel-schedule-system-v1`, based
on the preparation reporting head, not the parent hardening checkout. The
integrated execution source is its ancestor. Leave other worktrees untouched.
Historical software/SDK qualification remains associated with its exact source.
Do not rerun the unchanged 14-cell SDK gate simply because this roadmap changed.

At 2026-09-05T20:12:37Z a fresh read-only ETH check found all 40 ranks owned,
including rank1, with another user's `gwfa_host` PID 5663 active. No lock,
allocation, transfer or physical attempt was made. That historical occupancy
block was resolved by the separately recorded 2026-09-07 gate below.

### Accepted P0 Physical Gate

The unchanged seven-cell correctness gate executed once on safari-baguette1 at
clean source `b921b8804e324da75222354ee2f4df41e770b75c`, SDK 2023.1.0,
CPU 0, powersave governor and `/dev/dpu_rank1`. All six T1/T8 host, DPU and
initialization binary hashes matched the previously qualified checkpoint. The
frozen template SHA-256 remained
`8c03fe6e8cefd67838bfb5ddbd199676b1933719fe21a0a3b7a0e7a09ba62a62`.
The existing preparation helper resolved only execution/session paths into a
new, identity-bound configuration; it did not change the matrix or policies.

- Experiment: `e3d94c8fde216894fcff96c34c129f58b3f244aaa5e4f31ae14248848b01821e`.
- Run: `abb9fe81-996e-44ad-8214-191a446cf14d`.
- Results: 7 successful samples, 7 successful released sessions, zero failed
  or unsupported attempts, no fallback, no retry or replacement.
- All 7 passed exact numeric-policy replay. All 3 float32 attempts passed
  full-precision accuracy qualification. The 4 int8 attempts report error and
  remain full-precision accuracy-unqualified; this is not float32 equivalence.
- The exact Bell2/Stress14, route, block, source, executable and physical-plan
  identities passed both canonical and strict gate verification on both hosts.
- Archive: `phase-a-physical-b921b88-20260907-v1.tar.gz`.
- Archive SHA-256:
  `bbe6c5c18247c167e5bb5cb40a6e84ca87f88438197ae95af14a9bcdf9b787a8`.
- All 23 internal file checksums passed. Remote stage/archive and the safely
  extracted local archive were retained and independently verified. No remote
  original was deleted.

Remote evidence is under `/home/tkazulak/evidence/`; local evidence is under
`thesis/implementation/runs/eth/safari-baguette1/` followed by the full execution
SHA and `execution-integration-v1/physical-20260907-v1/`. `acceptance.json`
records verifier outputs and retention status. Historical SDK acceptance and
blocked preflights remain unchanged. This is physical correctness evidence,
not a timing campaign, production merge, final-system freeze or qualification
of the experimental v5 mechanisms.

All subsequent controllers use the agreed nonblocking exclusive flock at
`/home/tkazulak/evidence/upmem-experiment.lock`. The lock was held through the
gate and archive finalization, then released; rank1 was observed unowned after
completion. The persistent file is not itself a reservation. Fresh ownership,
competing-process, identity and environment admission is still mandatory.

## State and Dependencies

| Phase | Current state | Required exit evidence |
| --- | --- | --- |
| P0 reconcile | Seven-session physical correctness accepted at exact `b921b88`; two verified copies | Complete; does not adopt experimental v5 execution |
| P1 census | Source-only frontier extension implemented; physical weighting pending | Frozen targets, ready-width/critical-path/liveness facts and benchmark cells |
| P2 kernels | Fusion physically confirmed; K1 correctness passed but the fixed-path A/B rejected adoption at `3056090` | Preserve K1 negative evidence; retain panel geometry and admitted fusion for later composition |
| P3 DAG waves | Seven-session physical correctness independently audited; 72-attempt fixed-resource A/B raw accepted after offline verifier correction; separate 12-attempt Stress16 D4/T8 development confirmation now accepted | Use the confirmation for later composition only; no global production adoption |
| P4 resident/slice | Physical campaigns complete and independently audited: resident and both Stress16 slice cells NO_GO; fresh EDC14 D4 confirmation PASS; cumulative 324 attempts | Preserve negative results and unused confirmation slots; EDC result is development-only |
| P5 composition | Composition SDK tests and schedule/kernel-aware features exist; final-policy admission and path-cost integration remain open | After P4 audit closure, jointly qualify and freeze executor/source/binaries/policies/features; no automatic production adoption |
| P6 paths | Not started | New bounded physical data, offline profile, untouched test and raw evidence |
| P7 release | Not started | Source lineage, checksummed portable bundle and two verified copies |

The initial v2 documentation/budget checkpoint `f07e7d98fb81b533e8b669016cc9b7913c63aa37`
passed 1,114 local tests with strict SDK requirements enabled, Ruff, and hosted
CI run `33989762666`. This qualification does not cover subsequent uncommitted
kernel/scheduler work or establish physical acceptance.

Software work may proceed before P0 hardware access; no downstream physical
campaign may bypass that gate. Do not launch the superseded 192-attempt path
calibration. No old pilot medians or lost raw archives become final-system data.

## Implementation Contracts

At the initial implementation checkpoint, the runtime split multi-node stages
into sequential nodes. Its v4
requests describe one canonical geometry and one real product. A versioned native
contract is required, not reuse of reserved fields or a Python-thread shortcut.
One protocol owner coordinates operation/work/wave IDs, selectors, bounds,
completion identities and Python/native validation. Keep one active transport
implementation; historical sources provide old-runtime reproduction.

Use a pure deterministic frontier scheduler: ready requires all producer and
host-reduction completion; descending remaining critical-path work prioritizes
nodes, with node-ID ties. Give selected compatible nodes one DPU each, then
distribute spare DPUs by remaining work per assigned DPU, capped by useful work.
Freeze disjoint groups for each cohort's subwaves. Preserve tile/partial-sum
order and publish reconstructed outputs only after complete validated results.
One-DPU and chain cases degenerate to serial UPMEM, never CPU fallback.

Fusion preserves all four products and existing reconstruction. Account for two
complex operands plus four output planes in the actual 512-KiB arena. Retain
current tile/reduction geometry: non-fitting fusion cases take the explicit
generic UPMEM policy. Do not silently retile or combine complex outputs in core.
Resident segments additionally require complete local reductions, compatible
layouts, bounded simultaneous storage and exact policy preservation.

### Initial Native Boundary Work

An experimental private v5 control codec and C validation helper are under test in
`wave_protocol.py` and `wave_protocol.h`. Controls are 144 explicitly
little-endian bytes with operation/launch/tile identity and eight bounded plane
descriptors; the declared completion layout is 72 bytes. Python and C reject
unknown selectors, invalid resources, corrupt geometry, reserved fields,
unaligned/overlapping regions and overflow before MRAM access. The C header also
passes DPU-compiler syntax checks. The prepared-cohort integration below adds
operation tables, envelope digests and native host dispatch. Python whole-DAG
execution and evidence integration are still required before production use.

The 59 focused control tests include Python/C layout, native corruption
rejection, idle controls, explicit non-contiguous spans, and the existing int8
component accumulation bound. Kernels must dereference validated spans rather
than assume canonical offsets. The standalone control codec alone does not prove
operation-table binding, payload hashing or whole-runtime completion correlation.

The independent source audit confirmed that fusion needs `2A + 2B + 4C`
aligned MRAM bytes. A current legal float32 tile `(M,N,K)=(128,256,256)`
uses 512 KiB for one product but 1,310,720 bytes for all four products. That
case must take the declared generic route in the initial fixed-tiling study.
Control/completion symbols are outside the MRAM arena; do not invent additional
MRAM reservations unless the implementation actually places metadata there.
Preserve the distinction between KC panels accumulated within one request and
separate K-chunk work units reduced on the host. Fusion does not add K residency.

Review also identified two census/scheduling safeguards before integration:
memory-only resident candidates are not residency-qualified; and a DAG cohort
must not coalesce a node's existing K-wave boundaries and thereby change the
serial control even when the frontier has width one. These are software review
corrections, not failed physical experiments.

The pure `schedule_dag_waves` implementation has 15 focused tests. It preserves
each node's original wave boundaries, splitting but never merging them when a
node receives fewer DPUs. Useful group size is capped by the original maximum
wave width, not by the total number of work units across sequential K chunks.
It preserves within-node unit ordering, uses deterministic critical-path/group
assignment and emits host-reduction stages. No public route selects this
scheduler yet, and it is not evidence of physical DAG concurrency.

The census extension reuses all 40 frozen cells and rejects the four missing-
identity selections before reconstruction. Each eligible cell runs under the
existing 60-second subprocess timeout. It reports critical-path MACs, frontier
width, original wave occupancy, fused tile admission and geometry categories.
All measured timing fields remain null. Liveness is explicitly partial logical
tensor payload accounting, not RSS or whole-host admission: raw lane arrays,
encoded operands, transport copies and object overhead are not included, and
alias storage is not deduplicated. Resident pairs are memory candidates only;
all remain `admitted=false` pending locality/layout/reconstruction/scale proofs.

Reproduce from a clean committed head with:

```bash
PYTHONPATH=src /home/tom/repos/Masters/thesis/.venv/bin/python \
  scripts/characterize_upmem_frontiers.py --output-dir <new-ignored-run-directory>
```

The clean checkpoint `de783052e6f2b5bf2008da2ba229bbeae44a1b87` regenerated
all 40 cells: 36 eligible, four excluded, no measured timings. Relative checksums
pass. Its full pinned suite passed 1,201 tests with zero failures/skips and strict
SDK requirements, Ruff passed, and exact-head hosted CI `33991586159` succeeded.
The source-only census reports 164 non-fitting fused tiles across 20 operation
entries; these retain generic UPMEM geometry. Its local portable archive is
`runs/kernel-schedule-system-v1/kernel-schedule-census-de78305.tar.gz`, SHA-256
`7fe44a5d7f4252ac2b0e62bf855087e82fd145c065aeb00218101d83ae2ba41d`.
Second independent-copy upload is awaiting explicit export approval; do not
describe this archive as satisfying the two-copy completion gate yet.

### Experimental One-Launch Kernel

`panel_compute.h` contains the accepted real-product panel body, extracted
without arithmetic or panel-barrier changes. Both `dpu.c` and experimental
`dpu_wave.c` call this one implementation. The latter dispatches one real product
or RR, II, RI, IR in one SDK launch and retains four distinct output planes.
It validates controls before MRAM access and dereferences explicit plane spans.
Tasklet zero alone updates completion facts; all tasklets follow the same panel
barriers, including idle tasklets. Idle/invalid controls skip arithmetic.

The test-only native probe always requests `backend=simulator`, loads once and
can issue repeated launches. It is not a second production host. Tests compare
individual products byte for byte with four real launches and deterministic
sequential float32/int32 replay, including odd rows, K panels, inactive tasklets,
noncontiguous spans, untouched padding, invalid controls and session-local reset.
No simulator time is performance evidence. Compile qualification covers T1-T24.
The focused kernel gate passed 29 tests; the control/completion codec gate passed
87 tests, including native/Python byte-layout checks. Pending prefix progress is
retained only as diagnostic information. Even a pending record with all four bits
set cannot satisfy successful terminal correlation.

The new kernel is not selected by public runtime configuration. Operation-table
identity, input digest/scale binding and native completion checks are implemented
in the experimental prepared-cohort path. Whole-runtime integration remains
mandatory. Host reconstruction must
retain original lane-major and K-chunk accumulation order when fused outputs
arrive wave-major. Do not interleave lane reductions simply because products
arrive together. A hard DPU/SDK fault can still prevent completion retrieval;
the caller must fail closed and never interpret a missing response as success.
The independent native review found no arithmetic, barrier or addressing blocker
in this experimental target. Physical DPU-index ownership is deliberately a host
gate: the kernel checks range and echoes the supplied index but cannot establish
the host's enumeration mapping. Production dispatch must call admission with the
actual selected DPU index and correlate every completion before publishing data.

### Census-Selected Outer-Product Prototype

The frozen `de78305` census selects the plan's `K=1` alternative rather than a
GEMV kernel. Across its repeated circuit/path/topology/numeric entries, outer
products account for 136 of 2,304 operation entries, 42,102,784 of 54,284,992 real
MACs, and 1,325,264 of 1,500,592 operand-read helper calls. These are source-derived
counts, not pooled physical timings. GEMV entries are numerous but mostly tiny;
counting operations alone would select a different target. Freeze this target
for one bounded experiment; do not switch kernels after noisy physical results.

`outer_compute.h` implements a real `K=1` product using existing WRAM storage.
Tasklets cooperatively read the two padded operand vectors once into disjoint
shared regions, synchronize, and own cyclic contiguous 32-element output blocks.
The arithmetic keeps the existing positive-zero accumulator followed by the one
multiply/add; no K reduction, requantization or complex-output fusion is added.
Output block boundaries are 8-byte aligned, and only the last block may need a
short unaligned write. No two tasklets own the same output word. A final barrier
protects shared-buffer reuse for the next product.

The private v5 selectors `REAL_OUTER=3` and `FOUR_PRODUCT_OUTER=4` have exactly the
existing real/four-product plane layout and completion semantics. Python and C
reject these selectors for `K != 1`. The accepted ABI-v4 kernel and panel body are
unchanged. Native executable hashes change because v5 dispatch and validation
admit new selectors; old binaries are not equivalent evidence for this source.

`geometry_policy="outer_k1_v1"` opts in through the prepared-wave runtime.
`panel_only_v1` remains the control/default. The rule considers each existing
work unit's local K, including a one-element tail of a larger reduction; it never
changes tile geometry or the host's K-chunk order. Fusion admission is independent:
a fused tile that exceeds MRAM uses four real launches with the same geometry.
Geometry policy enters the execution-strategy hash and sample/operation facts;
the same scientific physical tiling retains its plan identity. Per-operation
`outer_product_tile_count` counts unique work units, not four repeated products.
`real_product_tile_launch_count` counts unfused active tile launches, independently
of panel/outer geometry. It replaces the ambiguous experimental
`generic_tile_count` label; `fused_tile_count` counts fused active tile launches.

This is a prototype, not an accepted kernel optimization. Compare panel and outer
policies at fixed paths, resources, launch policy and schedule. Then perform
composition tests. Do not reuse old panel-only path-cost calibration for this
executor; schedule/kernel-aware cost extraction remains a P5 gate. No physical
benefit or change in numerical accuracy is inferred from source counts or SDK
execution. No second geometry kernel is authorized in this core milestone.

The independent T8 disassembly audit under local SDK 2025.1.0 found native
`mul_sl_sl` with sign-extending byte loads in both the accepted panel and new
outer int8 product loops. This multiplies the low signed bytes into a 32-bit
result, not arbitrary full-width 32-bit operands. Neither product loop calls
`__mulsi3`; surrounding address arithmetic can. The outer float path retains
`__mulsf3` followed by `__addsf3`. One division remains per output block, not per
MAC. No arithmetic-helper replacement is justified by this inspection.
The bounded review found no remaining numerical, ownership, dispatch or codegen
blocker; it did not inspect all tasklet instruction streams or physical timing.

Local T1-T24 builds include host, accepted v4 DPU, experimental v5 DPU and init
binaries (96 hashes). In the inspected T24 v5 link, `.text` is 13,480 bytes and
the WRAM data/stack/cache end is 50,096 bytes. These are linked image facts, not
peak host-memory measurements or ETH SDK 2023.1.0 qualification. Isolated SDK
tests cover signed zero, int8 extrema, padding, maximal admitted geometry,
repeated mixed kernels and corrupt K. Full-DAG tests also cover an outer K1
tail of K257, preserving original reduction order. Physical gates remain open.

### Prepared-Cohort Native Dispatch

The existing persistent host accepts experimental `--wave-v5` and
`SUBMIT_PACKED_WAVES <session-root-basename> <sha256>`. Each process selects exactly
one protocol; the accepted default remains v4 during the bounded integration/A-B
period. Retire the superseded active path after parity and adoption, not before
physical qualification. No second native host or Python thread scheduler was added.

The private executable envelope contains a 136-byte little-endian header,
112-byte operation records, 160-byte dense wave-major/DPU-major tile records and
one input blob. It binds physical-plan and DPU-binary hashes, node/contract hashes,
canonical geometry, shared numeric scales, output offsets, invocation identities
and all four input-plane bytes through a submitted whole-file SHA-256. Controls
remain the existing 144-byte v5 records. This is a lowering of `UpmemPlan`, not a
new planning model. Native dispatch does not perform DAG dependency analysis.

Python/C reject geometry overflow, overlapping outputs, duplicate identities,
invalid padding/nonfinite float data/asymmetric int8 values, and DPU group changes
within a cohort. Wave/request IDs increase within each envelope; envelope and
request sequences increase across the session. Logical wave/tile IDs may recur in
a later execution with new request IDs. A global tile-ID exclusion cache would
incorrectly prohibit legitimate prepared-plan reuse and is intentionally absent.

All request data is validated before its SDK transfers or launches. The existing
session allocation and executable load happen at startup, before dynamic inputs
exist. Required symbols and `WAVE_TASKLETS` are checked before READY; wrong
executables/tasklet builds release the allocation and fail startup. Do not claim
that request validation precedes persistent session allocation.

The native reader uses a nonblocking, no-symlink regular-file open and one owned
snapshot, with a 512-MiB per-envelope admission limit. This is a parser/host
allocation policy, not a DPU geometry or complete-host-liveness guarantee. An
oversized cohort is rejected, not silently retiled or routed to CPU. The snapshot
is hashed before use, preventing file truncation/mutation from changing validated
bytes during execution. One reusable 256-KiB output buffer bounds collection
scratch. Reported snapshot/payload/control/output-buffer byte counts expose the
copy cost; do not claim it disappeared. Future full-route memory admission must
include Python inputs, encoded planes, the packed blob and this native snapshot.

One synchronous set launch executes each subwave. Disjoint DPU groups can carry
different operations/geometries, with explicit idle controls in tail waves.
The host validates every completion and writes one deterministic result stream:
72-byte completion followed by that slot's logical product bytes, without MRAM
padding. Responses retain file hash, completed-wave/result counts and failing
wave/DPU/operation/product facts. A failure stops the session, preserves available
prefix evidence, ignores later queued submissions and releases once. The stream
is not an atomic transaction and no completed work is retried automatically.

The focused gates cover 39 Python/native codec checks and 22 persistent-host SDK
checks: T3/T7/T8/T12/T24, float32/int8, two independent operations on three DPUs,
partial waves, repeated invocations, exact product/completion bytes, transfer
counts, malformed/replayed envelopes, wrong binaries, FIFO/symlink/oversize input,
and injected second-wave failure. Independent review found no remaining native
correctness blocker within this scope. The whole `session.run_once()` DAG route
still needs exact scheduled-unit coverage, tensor ownership, completion/reduction
and result publication integration. These tests are not a physical DAG speedup.

Local SDK development qualification uses **2025.1.0**, not ETH's frozen
**2023.1.0** toolchain. The previous clean kernel checkpoint `412fc3c5fac70fdb0e30688b643ab6caf38502d6`
passed 1,258 local strict tests and hosted CI `33992817366`. Its portable local
qualification archive has SHA-256
`e960b14fe20a8cd59bdff20f6abe7d75e5f0002a012470e9e2dedd9d5a2af276`.
ETH-toolchain qualification and physical acceptance are separate pending gates.

### Python Prepared-Cohort Client

The existing `V4Profile`/`V4Session` lifecycle now admits an explicit experimental
`packed_wave_v1` profile and selects the native `--wave-v5` mode. READY must match
the profile, ABI, kernel, resource allocation and target; binary digests are
required. There is still one process manager, non-reentrant operation lock,
bounded output pump, timeout path and release implementation. The default public
whole-TN route remains the qualified packed-operation executor during integration.

`submit_waves` binds each envelope to the opened DPU binary, exact profile
resources/numeric mode, a session-fixed plan digest and increasing invocation
sequences. It packs once, writes one exclusive session-owned file and validates
native launch/result counts, transfer bytes, timing facts and failure fields.
The result must be the exact sequence-named regular file with its expected byte
length and SHA-256; symlinks and FIFOs are rejected without blocking. Result
snapshots have a separate 512-MiB admission cap checked before submission. This
is not complete host-memory admission, and oversized work is never silently
retiled or sent to CPU.

`decode_wave_results` checks every completion against its control, including idle
slots, and exposes RR/II/RI/IR as read-only views of one immutable result snapshot.
Absent products remain empty slots; generic real execution has only RR populated.
Float outputs must be finite; arbitrary int32 patterns are not incorrectly
restricted to the int8 operand range. Failure poisons the session and retains
the request identity, native partial-progress fields and available result file.
No retry or atomicity is implied. Successful artifact cleanup remains the
whole-runtime caller's responsibility after reconstruction/evidence capture.
An adversarial simulator regression delays Python consumption until a faulted
native process has exited. The wave event queue must retain RESPONSE, RELEASE and
EOF together, and cleanup must consume a queued release even if the process is
already dead. Otherwise valid failure/provenance records can be lost to a host
timing race. The queue remains bounded to these three protocol events.
The deadline is cooperative between preparation, submission, response reading,
hashing and decoding; individual Python/native-array operations and regular-file
system calls are not forcibly preempted. Existing terminate/kill cleanup can add
up to two seconds of process-wait grace. This is not a hard real-time timeout.
Whole-attempt/session-close accounting must include actual elapsed cleanup time.
The independent client audit retained these timeout limitations explicitly and
found no further identity, reentry, file-admission or partial-failure blocker.

The client-only checkpoint did not establish whole-DAG correctness. The following
integration supplies that caller; physical acceptance remains a separate gate.

## Whole-DAG Prepared-Wave Integration

`plan_upmem(..., schedule_policy="static_dag_waves_v1")` lowers the existing
deterministic scheduler into the existing `UpmemPlan.stages`. Validation
recomputes that exact schedule. Logical DAG identity stays unchanged; scheduled
placement and policy enter physical-plan identity. Default serial-plan hashes
remain unchanged. Runtime never reschedules or retile-fixes an admitted plan.

Select `UpmemResources(request_transport="packed_wave_v1")` to execute prepared
waves. The default remains the accepted packed-operation route while prototypes
await physical decisions. A static-wave plan cannot use the old transport.
`fuse_complex=True` enables one-launch four-product tiles only when their existing
geometry fits the fused MRAM layout; otherwise the same tile uses four real
launches. This is kernel dispatch, never CPU fallback. Transport, schedule,
native ABI/binary and complex-launch policy are recorded in execution identities.

`wave_work.py` checks exact tile/work-unit coverage, extents, resource slots,
exclusive DPU-group ownership and canonical encoded operands. Whole-operand int8
scales are established before slicing into tiles. Generic and fused results are
reconstructed lane-major with the existing K-chunk order and CPU policy replay.
All cohort results and reconstructions must finish before any cohort output is
published. Host reductions then follow the existing deterministic order.

Native kernel/transfer/route counters describe a cohort, not an individual node.
They are carried on its first operation only, with explicit
`cohort_counters_on_first_node_v1` scope; sums therefore count each launch once.
Per-operation preparation, arithmetic products and bytes remain separately
attributed. Idle control/completion overhead is carried only on the first node
under `cohort_idle_overhead_on_first_node_v1`, with explicit
`cohort_idle_h2d_bytes`/`cohort_idle_d2h_bytes`; subtract these to obtain that
node's active traffic. This is an accounting allocation, not an operation-local
measurement. Shared preparation extraction includes operand copies in the inner
operation timer; whole `steady_execution_v1` already included those copies.
No timing comparison should silently treat that inner boundary as unchanged.
The old serial SLR feature extractor rejects static-wave plans until P5 supplies
qualified schedule-aware costs. SDK timing is not calibration evidence.

SDK tests cover full fork/join DAGs, repeated runs, split-K, sliced host reductions,
Bell/GHZ/Stress full statevectors, float32 and shared-scale int8, T3/T7/T8/T12/T24,
one/three/four DPUs, partial waves, and both launch policies. Injected partial
failure preserves the cohort/operation context and prevents dependent submission
or session reuse. Lower-level parser/native-failure tests remain required.
Successful cohort files are removed after reconstruction and evidence capture;
failed artifacts are retained deliberately for incident retrieval, not deleted
as successful-work cleanup. A poisoned session cannot submit another cohort;
the failed cohort has at most one envelope and one result file, each under its
512-MiB cap. The campaign owner must archive these before removal. Encoded inputs,
envelope buffers and result views
can coexist in host memory. The existing per-envelope/result size caps are not
a complete peak-live-host-memory bound; that remains a composition admission gate.

## Bounded Locality Preparation

`scripts/prepare_upmem_locality_probe.py` reads the checksummed P1 frontier census
(`b26a20e821c1510c6975c4990b2224c42d3f656cf98c0e14c958d7cfe19c3095`).
It reconstructs only retained greedy paths for Stress16 and EDC14, verifies
their logical and physical identities, and never generates candidate paths or
launches a simulator/device. Its JSON is a preparation artifact, not physical
evidence or execution authorization. By default it requires clean source and
verified ancestry from the census source. `--allow-dirty-preview` is an explicit
development-only opt-in; its outputs are labelled previews and cannot freeze a packet.

The static residency checker admits only consecutive single-node operations,
one full unbatched tile each on the same DPU, exactly one use of the intermediate,
no fanout, no fixed slices, no unary host reduction, and identical native label
orders at the resident boundary. Shared-scale int8 is explicitly unsupported.
Both product sets, both reconstructed float32 planes and the external operand
remain live in one conservative joint MRAM layout; no recycling is assumed.
Static admission alone does not qualify native reconstruction or WRAM/IRAM use;
the test-only qualification below supplies that separate software check.

The retained two-circuit census gives 40 statically eligible pairs. The maximum
padded intermediate-traffic candidate is Stress16 `contract_121 -> contract_122`
on one DPU: local `(M,N,K)` geometries `(16,64,4)` and `(16,256,4)`, 1,024
intermediate complex elements, 93,184 bytes of joint live MRAM. Relative to two
fused four-product launches, retaining it could eliminate 24,576 padded payload
bytes (four product readbacks plus two component uploads). It adds an estimated
24,576 local payload bytes for reconstruction reads/writes. Neither estimate is
a hardware counter or a runtime improvement. Control/completion traffic is
excluded. Production residency remains unimplemented and unaccepted; the
test-only native qualification below does not change that execution policy.

The slice probe keeps every Cartesian partial with its complete output indices
and an explicit host sum. Stress16 selects `contract_124`, labels `(57,117)`;
EDC14 selects `contract_27`, label `(19,)`. Selection uses original arithmetic
work, then slice count and stable IDs, with actual disjoint sibling cohorts
required at both two/four DPUs. No timing enters the choice. The controls are
unsliced serial, sliced serial, and the same sliced DAG with static waves,
using unfused panel execution to isolate decomposition and scheduling.
Static waves may also overlap other ready original nodes: this comparison
measures whole-DAG scheduling of the sliced graph, not slice-only concurrency.

Planned launch counts (two/four DPUs) are respectively 532/512, 628/560,
392/208 for Stress16 and 144/124, 176/140, 136/72 for EDC14. Arithmetic MAC
counts remain equal for these single-node decompositions, but full partial
output traffic and host reduction increase. Lower launch counts do not establish
faster execution. The preparation records padded payload, idle-slot, control,
completion and host-reduction counts separately.

SDK fixtures at four qubits qualify complete partial coverage, float32 policy
replay, full-statevector shape/order, serial/static equivalence, disjoint sibling
ownership, partial waves, repeated sessions and host reduction before dependent
consumers. Simulator timing remains claim-ineligible. Development-sized physical
slice timing and resident integration remain separate gates.

CPU replay of all three development-sized arms at both topologies preserves
65,536 Stress16 and 16,384 EDC14 amplitudes. Sliced serial/static outputs agree
exactly; maximum absolute errors against the unsliced complex128 reference are
`4.692546e-7` and `1.210162e-8`, respectively, within the `2e-6` absolute/relative
qualification tolerance. This is numerical qualification, not execution timing.

### Test-Only Native Resident Pair

`tests/native/upmem_resident_probe_{host,dpu}.c` and its private header execute
one fixed pair through the SDK simulator. The host hardcodes `backend=simulator`;
there is no physical option, production command, general graph interpreter, or
new public plan type. Existing production host/DPU sources and ABI-v4 are unchanged.
The 320-byte little-endian test descriptor contains two existing v5 controls,
two retained-plane spans, a version, operand side, and monotonically increasing
pair ID. It is not a second semantic physical plan. Exact label compatibility
remains the Python admission proof; equal element counts alone are insufficient.

Both arms use **two launches** and the identical panel helper. Launch one creates
four separate products. In the host-roundtrip control, host-decoded intermediate
planes are uploaded before launch two. In the resident arm, launch two reads
the products still in MRAM, applies positive-zero lane assembly, then float32
`RR-II` and `RI+IR`, and consumes the retained planes. No launch-fusion benefit
is attributed to residency. Every tasklet owns disjoint 16-element blocks;
only the final block can write a four-byte tail. Existing per-tasklet A/output
buffers are reused, with no additional numerical WRAM arena.

The producer saves an immutable descriptor. A changed pair, stale pair ID,
out-of-order command, bad bounds/layout or numerical reconstruction failure
poisons the probe session. No consumer arithmetic follows a failed reconstruction.
The failure uses the strict v5 execution-failure record with product index zero
and an empty completed prefix. A reconstruction failure may have written partial
retained data; there is no atomicity/rollback claim. Kernel completion still means
that products executed, not numerical acceptance. As in the ordinary wave route,
final readbacks must pass the host finite-value decoder and policy replay before
qualification. An explicit finite-input/final-product-overflow fixture verifies
that these two gates cannot be conflated.

SDK cases exercise left/right resident operands, odd tails, idle tasklets,
T1/T3/T7/T8/T12/T24, repeated pairs, corruption, stale/changed identities and
failure poisoning. All T1-T24 probe binaries build. At T24, linked WRAM end is
50,680 bytes, IRAM text is 15,056 bytes, and the main stack frame is 200 bytes.
T8 disassembly retains the four positive-zero additions followed by subtraction
and addition; native reconstruction also preserves signed-zero/subnormal cases
and rejects nonfinite lanes or overflowed reconstructed components.

The frozen Stress16 corpus test uses the exact greedy candidate and physical-plan
IDs above, captures the accepted CPU replay's actual encoded operands, and checks
resident/host arms against the same first/second product bytes. Both arms have
identical final MRAM, including padding. Injecting the native consumer lanes into
the unchanged reference DAG reproduces all 65,536 final amplitudes exactly and
passes complex128 validation. This is a bounded native-pair plus reference-DAG
proof, **not** a complete physical or SDK-native resident Stress16 simulation.

The test harness reads back the full arena after every command for diagnostics.
Its timing and traffic are therefore not production transport measurements.
No physical gain, SDK timing speedup, production adoption, or two-copy physical
evidence acceptance is claimed. A genuine integration experiment must remove the
diagnostic readbacks and preserve the same-pair/two-launch control, whole-attempt
accounting, finite-value gates and bounded memory before physical comparison.
Production adoption would also need an explicit resident memory/execution-policy
identity. It must not silently claim `host_roundtrip_v1` while omitting that
roundtrip. The current corpus plan ID identifies the admission/reference plan,
not an already qualified production resident route.

## Prepared-Wave Execution Facts

`execution_features.py` supplies a separate, deterministic description of the
implemented prepared-wave executor. It is not the historical serial SLR profile
and does not enable calibration or accept any kernel/scheduling policy physically.
Its inputs are a validated DAG/physical plan plus explicit fusion and geometry
policies. It requires no tensor payloads, native process, or timing observations.

The description distinguishes logical work waves from physical micro-wave
launches. A mixed fused/generic wave executes all admitted slots in its first
launch, then only generic slots for the remaining three products. Completed
fused slots still incur idle controls/completions and kernel entry barriers.
Serial scheduling submits each node separately, including grouped slice stages;
static scheduling submits a ready cohort on disjoint DPU groups.

Count padded operand and product payloads separately from control/completion
traffic. These are application-visible host/DPU bytes, not PCIe bus counters,
filesystem traffic, or total host copies. Keep useful four-product MACs separate
from `wave_critical_real_mac_sum`, the sum of each physical wave's maximum DPU
MAC count. The latter describes arithmetic imbalance and available overlap; it
is not elapsed time, an instruction counter, or a calibrated kernel predictor.
Host preparation, SDK waiting, transfers, and reconstruction do not disappear
when this arithmetic quantity falls.

Local traffic counts follow `panel_compute.h` and `outer_compute.h`. Aligned
spans are estimates, not a model of every transaction inside the SDK's unaligned
helpers. In particular, they do not count hidden read/modify/write traffic or
virtual-lock contention. All product-plane bases are eight-byte aligned, so
their absolute placement does not change these span estimates. Barrier events
count three wrapper barriers per allocated DPU/launch, plus two per panel/product
or two per outer product. Tasklet call counts multiply those events by the
compiled tasklet count. These are DPU-local barriers, not cross-DPU barriers;
the final wrapper barrier is outside the kernel's cycle-counter interval.

Known WRAM buffer bytes exclude stacks, globals, SDK runtime storage, and linked
IRAM admission. The MRAM fact is the peak occupied span within one tile arena,
not a complete host-memory or resident-segment admission result. Numerical
representation overhead is explicitly not estimated, not presumed zero or
non-discriminating.

Composition tests compare the planned launch/transfer counts with actual
persistent-host SDK facts across serial/static scheduling, fusion on/off,
panel/outer dispatch, both numeric policies, and sliced reductions. They also
require unchanged CPU policy replay and repeated-session results. A separate
large/small mixed-wave fixture checks the real prepared control sequence without
executing hardware. SDK timings are never fitting data.

The full system freeze remains open. In particular, these facts do not prove a
complete peak-live host-memory bound, qualify production residency, fit a new
cost profile, replace the P0 physical gate, or authorize final path search.

## Prepared Snapshot Admission

Prepared-wave sessions derive input-envelope and logical result-snapshot sizes
from the validated plan's control tables before constructing operand payloads or
opening the native session. Both snapshots must satisfy the existing 512-MiB
limit. The codec repeats admission before copying bytes-like payloads, and checks
their buffer byte lengths before conversion. This preserves accepted wire bytes;
it changes when oversized requests are rejected, not the transport format.

Result sizes include completion records and logical, unpadded product planes.
They must not be confused with padded DPU-to-host transfer spans. Admission
covers serial and static DAG schedules, including per-node serial slice stages,
both numerical policies, and the selected fusion/geometry policy. Control-table
construction itself still allocates Python objects and adds session-opening work;
no timing improvement is claimed.

This is not whole-process host-memory admission. Whole-DAG input/intermediate
arrays, encoded operands, raw response views retained for evidence, current-wave
payloads, Python envelope assembly, and the native snapshot can overlap in
lifetime. Caller-owned backing allocations and allocator overhead are also not
bounded by tensor `nbytes`. The complete peak-live bound remains a composition
gate; neither the snapshot limit nor known WRAM/MRAM spans close it.

The execution-facts report now exposes `host_buffers`, an inventory of the bulk
allocations deliberately retained through the end of steady execution:

- Every graph output, including sliced host reductions, as a complex64 array.
- Two encoded planes per logical canonical operand, once per contraction.
- The full immutable result snapshot for every cohort, not merely its useful
  output bytes. Raw lane views pin completion records and idle-slot bytes too.
- The final C-contiguous complex64 output copy.

Caller input descriptor bytes are reported separately without alias-storage
deduplication. Input envelopes have a cohort lifetime; their maximum is reported
separately and is not added to the retained sum. Neither a largest intermediate
nor the largest response snapshot describes cumulative retained storage. Split-K
partials and all slice branches count before their reconstruction/reduction.

The inventory is checked against actual SDK response-buffer objects and encoded
NumPy arrays for both numeric modes, schedules, fusion settings and sliced
execution. It is not measured peak RSS, a complete workspace upper bound, or a
performance result. Canonicalization, quantization, reconstruction, hashing,
packing copies, native storage and Python/allocator/SDK overhead remain separate.
No new memory-budget threshold, execution-policy change or fitted score term is
introduced by this reporting-only addition.

The native allocation audit found one envelope snapshot plus a fixed 262,144-byte
output scratch allocation on the successful prepared path, at most 512.25 MiB
of those explicit heap allocations. Per-wave control/completion arrays are
bounded stack storage; result bytes stream to the output file. This excludes
SDK/provider allocations, libc and allocator overhead and is not a native RSS
bound. The output scratch bound depends on the existing validated 256-by-256
maximum tile geometry.

Applying the inventory to the existing frozen census (36 eligible cells, eight
schedule/fusion/geometry combinations each) gives 288 rows. Retained executor
bulk allocations range from 548,684 to 33,784,832 bytes. The largest input
envelope is 289,016 bytes; the largest result snapshot is 16,781,824 bytes.
Maximum cumulative retained response storage is 16,926,848 bytes. These values
describe the frozen paths only, not arbitrary future candidates or measured
RSS. No candidates were generated and no physical execution was used.

## Canonical Experiment Integration

The existing `plan` and `run` commands accept four optional UPMEM route options
for both SDK simulator and explicitly authorized physical execution:

```yaml
request_transport: packed_wave_v1
schedule_policy: static_dag_waves_v1
fuse_complex: true
geometry_policy: outer_k1_v1
```

These entries supplement the existing required route options; they are not a
complete experiment configuration. Omitted options retain the accepted
`packed_operation_v1`, `serial_nodes_v1`, unfused, `panel_only_v1` behavior.
Defaults are not inserted into normalized legacy configurations, preserving
their experiment identities. Non-default schedule, fusion or geometry requires
explicit prepared-wave transport, one rank and at most 64 DPUs. Unsupported
values and incompatible combinations are rejected before execution.

Scheduling remains part of physical-plan identity. Prepared executable identity
binds ABI-v5 transport and fusion/geometry dispatch in addition to binary hashes;
legacy executable identity is unchanged. Prepared sessions declare
`upmem_prepared_wave_abi_v5`. Existing sample/session schemas and canonical
verification are reused, with selected policies and actual cohort/kernel facts
in the existing backend-facts mapping. This is experimental route exposure,
not default adoption or physical qualification.

The integration tests exercise the normal CLI, planner, SDK session and canonical
verifier in 16 sessions: float32/int8, serial/static DAG schedules, unfused panel
versus fused outer dispatch, and unsliced/sliced Stress4 fixtures at 3 DPUs/T8.
They check identities, actual multi-node cohorts, dispatch counts, numerical
policy replay and absence of CPU fallback. Simulator timings are not performance
evidence. Runtime, kernels, transport codecs and candidate pools are unchanged.

## Budget and Preregistration

### Accepted Fusion Correctness Packet

After P0 acceptance, the frozen fusion packet ran once on 2026-09-07 from clean
execution source `30560900354b523b2a3a44971f81860a04888640`. No executable source,
binary, candidate or configuration changed. It selects `packed_wave_v1`,
`serial_nodes_v1`, `fuse_complex=true`, and `panel_only_v1`; no concurrent DAG
nodes, outer-product specialization, slicing or residency is selected.

The seven samples and seven released sessions passed canonical verification and
the packet-specific v5 identity/dispatch inspector both remotely and locally.
All seven passed policy replay; all three float32 cells passed full-precision
accuracy. Four int8 cells remain accuracy-unqualified with descriptive errors.
There were zero failures, unsupported attempts, retries or fallback launches.

| Cell | Fused tiles | Native launches |
| --- | ---: | ---: |
| Bell2 float32, 1D/T1 | 3 | 3 |
| Bell2 int8, 1D/T1 | 3 | 3 |
| Stress14 float32, 1D/T8 | 112 | 112 |
| Stress14 int8, 1D/T8 | 112 | 112 |
| Stress14 float32, 4D/T8 | 112 | 109 |
| Stress14 int8, 3D/T8 | 112 | 110 |
| Stress14 int8, 4D/T8 | 112 | 109 |

Every per-node count matched the frozen work-unit and wave tables. Outer-product
tiles and separate real-product tile launches were zero. This confirms the
tested physical fused path, not a timing improvement or general resource claim.

- Experiment: `691d896fe623e97c83063fefd4f04ccd1b2022d671cc7e487a5e1e3974e00858`.
- Run: `2d39313e-f8a4-4e16-97e2-7e2a2653d94b`.
- Configuration SHA-256:
  `ad6a77b6d9968657f6d069ee0ca3da926ac13a457e3e954e7612aa099b15dff9`.
- Archive: `kernel-schedule-fusion-correctness-3056090-v1.tar.gz`.
- Archive SHA-256:
  `4bab79ce6eceac60b623b669ad66eaf6f08eabe3330f8e7826f45bb02de9a036`.
- All 29 internal file checksums passed; two independently verified copies
  remain on ETH and locally under `runs/eth/safari-baguette1/`, the full source
  SHA, and `fusion-correctness-v1/`. No remote original was deleted.

The original preregistration retains its historical prepared-only status;
`acceptance.json` in the new result directory records the executed outcome.
The shared lock was held through archive finalization. There is no production
adoption or session-inclusive speedup claim. Matched fusion A/B, geometry and
DAG qualification are still separate gates.

At fusion-correctness closure, physical budget used was 14 attempts, consisting
of 7 P0 attempts and 7 of the 28 changed-checkpoint correctness slots. No
performance slots had yet been used; the later A/B consumption is recorded below.

### Fusion A/B Preparation

The next packet compares unfused and fused policies on the retained greedy
Stress16 (two layers), HS20 (depth one), and EDC14 paths, at one/four DPUs with
eight tasklets. Both arms use the same prepared-wave transport, binaries,
serial-node scheduling, panel kernel, float32 policy and physical work mapping.
Only fusion and its execution-policy identity differ. Execution remains at
`30560900354b523b2a3a44971f81860a04888640`; this reporting/analysis change does
not modify execution code or require another unchanged SDK qualification.

The packet has 72 attempts: twelve arm-cells, one warmup and five measured
complete blocks. The executed outcome is recorded below. Configuration SHA-256 is
`860d331cd43aff90a7fc08cd0c22a4e5e782f714f45555bb2eaa395e57513006` and experiment
identity is `73c70f5df054fc8335c581cbf78ef0c20eaf5df56bd4492b4a3342d9161579be`.
Frozen tables and admission controls live in ignored
`runs/kernel-schedule-system-v1/fusion-ab-preregistration-3056090/`.

HS20 intentionally retains 64 generic real-product tile launches in either
fused-policy topology because those tiles fail fusion-arena admission. That is
UPMEM execution, not CPU/simulator fallback. No retile or exclusion is permitted.

The primary estimand is the equal-cell geometric mean of unfused/fused arm
median session-inclusive times. Session open, steady execution and close are
summed per sample before aggregation. Ten thousand deterministic paired-block
bootstrap resamples share block indices across all six cells. The separate
median of paired speedup ratios is descriptive, not the primary estimand.
Raw observations, MADs and setup differences are retained. Missing legacy
request-build/request-wave timers remain null; measured cohort wall time is
not labelled Python CPU time.

The preregistered decision uses the existing 5% aggregate session-inclusive
reduction, lower paired-bootstrap speedup bound above one, and no cell median
regression beyond 5%. A passing result still requires the separately reserved
Stress16/four-DPU fresh confirmation. The 7,200-second command cap, no-retry
rule and two-copy evidence gate apply independently of the observed outcome.
The pure analyzer, synthetic corruption tests and controller deadline tests
passed together: 43 tests, zero failures/errors/skips. These tests are software
evidence only; they do not manufacture or replace physical observations.

### Fusion Physical A/B Result (2026-09-07)

The unchanged packet executed once at `30560900354b523b2a3a44971f81860a04888640`:
72/72 successful samples and sessions, including 12 warmups and 60 measured
attempts. All float32 accuracy and policy-replay gates passed, with no failures,
unsupported attempts, CPU/simulator fallback, retries or replacements. Both
arms retained the same paths, prepared-wave transport, serial scheduling,
panel-only geometry and T8 host/DPU/init binaries. Per-node launch counts and
output hashes matched the frozen controls. HS20 retained the planned generic
UPMEM tiles; no post-timing retile or candidate change occurred.

All times below are seconds and are medians of the five measured observations.
Session-inclusive time is computed per sample before taking its median.

| Circuit | DPUs | Unfused steady | Fused steady | Unfused inclusive | Fused inclusive | Inclusive speedup | Inclusive reduction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Stress16, two layers | 1 | 2.138216 | 1.354812 | 2.472895 | 1.675666 | 1.4758x | 32.24% |
| Stress16, two layers | 4 | 1.963459 | 0.887273 | 2.325208 | 1.201514 | 1.9352x | 48.33% |
| HS20, depth one | 1 | 2.823442 | 2.588956 | 2.970032 | 2.740228 | 1.0839x | 7.74% |
| HS20, depth one | 4 | 1.772681 | 1.302855 | 1.943463 | 1.459210 | 1.3319x | 24.92% |
| EDC14 | 1 | 0.475021 | 0.253601 | 0.586225 | 0.359565 | 1.6304x | 38.66% |
| EDC14 | 4 | 0.453967 | 0.192750 | 0.576156 | 0.303242 | 1.9000x | 47.37% |

The equal-cell geometric-mean session-inclusive speedup is **1.528869x**
(34.59% reduction), with the frozen paired-block bootstrap interval
**[1.504233, 1.559857]**. Steady speedup is 1.684688x
([1.645091, 1.718739]); reported kernel speedup is 1.221193x
([1.200894, 1.228775]). These are diagnostic intervals from five blocks,
not final-performance inference. All six inclusive medians improved. The
predeclared 5% aggregate reduction, lower bound above one, and 5% maximum
cell-regression gates pass. Decision: **A/B pass, pending fresh confirmation**.
No production default or adoption changed.

Kernel time is not held constant in this intervention: the same four products
execute with fewer physical launches and synchronization events. H2D, D2H,
preparation and measured cohort wall medians also decreased in each cell.
Session opening did not absorb the steady saving. The result supports this
admitted fusion policy on the tested paths, not universal fusion eligibility
or a claim that output byte volume halved. Whole-command peak RSS was 178,836
KiB, including planning, validation and all arms; it is not a per-arm memory
comparison. The command finished in 3:41.24 under its 7,200-second cap.

The sole controller acquired the shared lock, verified all rank ownership
checks and clean source, and ran on CPU 0 under powersave with SDK 2023.1.0.
Rank1 was unowned at termination. The availability timer remained paused.

- Run: `d6e1b1cc-db84-41dd-aadf-d4f0a3511a75`.
- Raw archive: `kernel-schedule-fusion-ab-3056090-v1.tar.gz`.
- Raw archive SHA-256:
  `cb8631eb0ac010ef60ec1adfc0a78c9c07a7a36fc3a1c7f6499af03db241fc9d`.
- Preregistration archive SHA-256:
  `b8794f826f38f0a31ef5c0355131e8ae87ff3e7913a5dc2b6998b6f8be32000a`.
- Analysis source: `e4655c849e9f6a1770594e0734ef074525c9c647`,
  [exact-head CI passed](https://github.com/kazulak/Masters/actions/runs/34148456660).
- The independent pre-run audit found three verifier gaps. Expected waves
  are now recomputed/cross-checked, terminal binary hashes checked separately,
  and integer counters distinguished from booleans. All 50 focused tests
  passed after these packet-only repairs, before timing. Runtime was unchanged.

All 38 internal raw-stage checksums, canonical verification and strict packet
verification passed on both hosts. The complete archive remains on ETH under
`/home/tkazulak/evidence/` and locally under
`runs/eth/safari-baguette1/30560900354b523b2a3a44971f81860a04888640/fusion-ab-v1/`.
No remote original was removed. Local `analysis.json`, `cell_summary.csv`,
`normalized_rows.json`, `decision.json` and `ANALYSIS_SHA256SUMS` retain raw
paired observations, medians, MADs, ranges, intervals, unavailable timer fields
and the unchanged decision rules. The raw stage embeds the frozen analyzer
and preregistration, so derived results can be regenerated.

Total physical budget used after that A/B was **86 attempts**: seven P0, seven fusion
correctness and 72 fusion A/B. The preregistered Stress16/four-DPU confirmation
was reserved as a separate twelve-attempt packet, completed below. Geometry, DAG,
locality, composition, executor freeze and final path work remain open.

### Fresh Fusion Confirmation Preparation

The separately frozen packet implements the confirmation selected before A/B timing:
Stress16/two layers, four DPUs/T8, unfused and fused, one warmup plus five
measurements per arm. The expected path/work/wave table is selected unchanged
from the earlier packet and cross-checked against the configured physical plan.
There is no new candidate search or execution-source change.

The new configuration hash is
`bba45438b4ec1b66e3cf41e6370c135583ba5100d38179188425a52832803e3d`, and experiment
identity is `f2ec0d416a406e4d72853b2a817cd4a3d14555dd5b0ae2e3c7b8bde4bb8e1a19`.
Block and bootstrap seed is 20260908. The same 5% practical benefit and paired
uncertainty gate applies, using only the twelve fresh observations. The
physical command cap is 1,800 seconds; native requests remain capped at 120
seconds. No failed or inconclusive result authorizes a retry or another tuning
round. It used twelve reserved confirmation slots; its completed result follows.

The combined A/B-math regression, confirmation analysis, strict packet and
controller suite passed 64 tests, with no failures/errors/skips; Ruff passed.
The confirmation analyzer reuses existing paired-bootstrap and row-validation
helpers. Independent static review confirmed the declared cell, no old-row
pooling and unchanged execution policies. Its preparation-state and deadline
metadata findings are resolved by the explicit freeze gate and documented
1,800-second cleanup policy. This is software preparation, not another
unchanged execution/SDK qualification or evidence of confirmation speedup.

### Fresh Fusion Confirmation Result

The frozen packet ran exactly once on 2026-09-07 at clean execution source
`30560900354b523b2a3a44971f81860a04888640`. All **12 samples and sessions** passed
physical, canonical and strict packet verification, including float32 accuracy,
policy replay, output hashes, resource admission and binary identities. There
were no failed, unsupported, fallback, retried or replacement observations.
One warmup per arm is retained but excluded from these five-measurement medians.

| Metric | Unfused median (s) | Fused median (s) | Speedup | Paired bootstrap 95% interval |
| --- | ---: | ---: | ---: | --- |
| Steady execution | 1.949019391 | 0.882959795 | 2.2074x | [1.8589, 2.2209] |
| Session-inclusive execution | 2.318010724 | 1.217846132 | 1.9034x | [1.6617, 1.9361] |
| Kernel | 0.362073661 | 0.292509492 | 1.2378x | [1.1532, 1.2474] |

Session-inclusive time is computed per sample before summarization. Its raw
MAD is 0.012155718 s unfused and 0.011244486 s fused; respective ranges are
[2.005035647, 2.349175930] s and [1.199979374, 1.293748946] s. All observations,
including the faster fifth unfused measurement, remain included. Bootstrap
resamples common measured blocks 10,000 times with frozen seed 20260908; it does
not pool observations from the preceding A/B. The **47.46% session-inclusive
reduction** passes the preregistered 5% practical threshold, lower interval
bound above one, and maximum 5% regression gate. Fusion therefore passes its
bounded fresh confirmation; this is not whole-system production adoption.
Kernel time is allowed to change because fusion changes launch/synchronization
cost, while retaining four-product arithmetic and output equivalence.

- Run ID: `528fa6ac-355b-45e2-b255-147ae85bd984`.
- Analysis source: `fc4ba55a85b4b233751a46320b99e37fdab61ecd` (exact-head CI
  [34153902908](https://github.com/kazulak/Masters/actions/runs/34153902908) passed).
- Raw archive SHA-256:
  `889d5a5c9372aa08831efd944f4c15840cfaaccb58a58be94bb7d569628fb22f`.
- Remote stage: `/home/tkazulak/evidence/kernel-schedule-fusion-confirmation-3056090-v1`.
- Local archive, extracted raw records, acceptance, normalized observations,
  analysis, CSV summary and decision:
  `runs/eth/safari-baguette1/30560900354b523b2a3a44971f81860a04888640/fusion-confirmation-v1/`.

All 42 internal raw-stage checksums passed on both hosts. Two verified copies
are retained; no remote original was deleted. The source remained clean and
the rank was released. No unchanged execution/SDK qualification was rerun.
Cumulative physical use is **98 attempts**. Next is the K=1 geometry correctness
gate, to be separately frozen, then its fixed-policy performance comparison;
DAG, locality, composition, executor freeze and path study remain open.

### K1 Geometry Physical Correctness

The seven-session geometry gate completed exactly once on 2026-09-07 at clean
execution source `30560900354b523b2a3a44971f81860a04888640`. It uses the same
qualified ETH SDK 2023.1.0 binaries, one rank, CPU0 and observed powersave governor.
No runtime, kernel, physical mapper or numerical-policy source changed.

The original Bell2 and Stress14 gate plans contain zero K1 units, so a selector
change alone would not test the specialization. Before execution the new packet
retained their seven resource/numeric routes and budget, replacing the two T1
cases with the existing SDK-qualified sliced Stress4 fixture (`contract_24`,
minimum four slices), and the five T8 cases with the already observed greedy
HS20/depth1 path. No candidate search or held-out instance was used. Every route
holds fusion enabled and `serial_nodes_v1` fixed, selecting `outer_k1_v1` through
the existing `packed_wave_v1` transport. This is a correctness gate, not an A/B.

| Circuit | Policy | DPUs | Tasklets | Unique K1 units | Fused tile launches | Real-product tile launches | Native launches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Sliced Stress4 | float32 | 1 | 1 | 4 | 32 | 0 | 32 |
| Sliced Stress4 | shared-scale int8 | 1 | 1 | 4 | 32 | 0 | 32 |
| HS20 | float32 | 1 | 8 | 24 | 58 | 64 | 122 |
| HS20 | shared-scale int8 | 1 | 8 | 24 | 58 | 64 | 122 |
| HS20 | float32 | 4 | 8 | 24 | 58 | 64 | 74 |
| HS20 | shared-scale int8 | 3 | 8 | 24 | 58 | 64 | 82 |
| HS20 | shared-scale int8 | 4 | 8 | 24 | 58 | 64 | 74 |

All **7/7 samples and sessions** passed physical, canonical and strict packet
verification, with zero failures, unsupported attempts, CPU/simulator fallback,
retries or replacements. Per-node kernel counts, complete work-unit sets,
resource admission, executable identities and release facts match the frozen
physical plans. All requested DPUs receive work; three/four-DPU HS20 includes
partial waves. HS20's 24 outer units comprise eight fused units plus sixteen
non-fitting units executed through 64 real-product launches, not 72 unique
units. Its other fifty units retain the panel kernel.

All three float32 samples pass the existing full-precision policy. Sliced
Stress4's maximum absolute error is approximately `1.9592e-7`; HS20's is
`5.8208e-10`. All four int8 samples pass exact policy replay, with maximum
absolute errors `0.00638418` for sliced Stress4 and `5.8208e-10` for HS20.
Int8 remains accuracy-unqualified under the existing reporting policy; these
values do not establish general full-precision int8 accuracy. Complex inputs
and host slice reduction are exercised by Stress4; this is not a slice
concurrency or residency qualification.

Packet preparation passed 39 focused tests and Ruff, with independent read-only
review. Pre-freeze repairs corrected the remote interpreter path, the prior
decision's `result` key and configuration-derived 3/4 accuracy counts. The
unchanged 414-test strict SDK qualification and T1-T24 binaries were reused,
not rebuilt or rerun. CPU fixture records are explicitly separate from physical
evidence. The freeze gate checks named tests because out-of-tree JUnit records
have empty `classname` attributes; it does not infer coverage from that field.

- Experiment: `7c0aaa809682ea65401905c36ac27ef1025b4366b94da404ec8cb4bf0b2b6dbf`.
- Run: `36bd228c-c634-4874-a803-0ccdcdcbf172`.
- Configuration SHA-256:
  `8da3b3d60a24b660c7da24d35bff788621fe126b26401148549cc00fad0433b7`.
- Frozen packet archive SHA-256:
  `7bec60990efa591005b8b3c91fca697c1a6fb5cc4a852cbd017e3f1025db74da`.
- Raw archive SHA-256:
  `90361cfbdbfa728175018a0c877bf6d2d27ea124700fc0338eb158232f4ee084`.
- Preparation/reporting predecessor: `3002f78df68325fbc7c91c0f2a50b50dd22188d6`.
- Remote: `/home/tkazulak/evidence/kernel-schedule-geometry-correctness-3056090-v1`.
- Local: `runs/eth/safari-baguette1/30560900354b523b2a3a44971f81860a04888640/geometry-correctness-v1/`.

All 43 raw-stage file checksums and both verifiers passed on both hosts. Two
verified copies are retained; no original was deleted. The source stayed clean
and rank release was verified. Cumulative physical use is **105 attempts**.
The K1 implementation is now physically correctness-qualified for these routes,
not adopted as a performance optimization. Next: the separately preregistered
fixed-path panel-versus-outer A/B; DAG, locality, composition, final executor
freeze and path optimization remain open.

### K1 Geometry A/B Preparation

The next packet keeps the previously retained Stress16/two-layer, HS20/depth-one
and EDC14 greedy paths at one/four DPUs and T8. Both arms keep admitted complex
fusion enabled, serial-node scheduling, float32 and `packed_wave_v1` at execution
source `30560900354b523b2a3a44971f81860a04888640`. The sole intervention is
`panel_only_v1` versus `outer_k1_v1`. There is no new candidate generation,
retile, runtime edit, SDK rerun or held-out exposure.

HS20 contains 24 K1 units per topology; the other two circuits contain none.
The preregistered benefit region is therefore the two HS20 resource cells.
Stress16 and EDC14 remain unchanged-work regression controls, not discarded
observations. Plan checks preserve physical/logical identity, work ordering,
DPU mapping, launch counts and planned host-DPU bytes. Geometry policy is
explicit in executable identity. DPU-local traffic and synchronization may
change with the specialized kernel.

One warmup and five measurements for each arm/circuit/topology produces 72
attempts. Complete blocks use seed 20260909. Session-inclusive time is computed
per sample before medians. The HS20 equal-cell geometric-mean speedup must show
at least 5% reduction and a lower paired-bootstrap 95% bound above one; every
one of the six cells must satisfy the 5% median-regression bound. Report the
all-six aggregate separately. These region and control rules follow the v2
plan and are fixed before geometry candidate timing. The reserved fresh
confirmation is HS20/four-DPU/T8, twelve new attempts, only after an A/B pass.

The analysis uses explicit `panel`/`outer` labels and shares the existing
paired-block statistics without relabeling rows as fusion observations.
Historical fusion and correctness timings do not enter the new analysis.
Missing component timers remain null. Packet software tests and read-only
audit are separate from unchanged executor/SDK qualification.

- Experiment: `09f40f0c445e14323aaae42e7613ecec758059929b3a333f1b514f1136e74a89`.
- Configuration SHA-256:
  `c2290b762dacf277a56c77c02279aa6d2355408f8085de58f2a8c976a5fe4bba`.
- Local preparation: `runs/kernel-schedule-system-v1/geometry-ab-preregistration-3056090/`.
- Physical budget: 72 reserved geometry A/B attempts, prior use 105 and maximum
  cumulative use 177 after this packet. Preparation has used zero new attempts.

The packet requires exact-head analysis CI, complete focused tests, independent
audit and immutable checksums before fresh locked admission. No geometry
performance result, confirmation, production adoption or full-system freeze
is asserted here. Failure requires preservation of the complete partial
artifact, retrieval and stop without retry or replacement.

### K1 Geometry A/B Result: No-Go

The frozen packet executed exactly once on 2026-09-07 at clean source
`30560900354b523b2a3a44971f81860a04888640`, SDK2023.1.0, CPU0, observed
powersave and rank1. All **72 samples and sessions** passed physical, canonical
and strict verification: 12 warmups and 60 measurements, float32 accuracy and
policy replay throughout, no failed/unsupported/fallback attempts and no
retries, replacements or splicing. Warmups are retained but excluded below.

| Circuit | DPUs | Panel steady (s) | Outer steady (s) | Panel inclusive (s) | Outer inclusive (s) | Inclusive speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Stress16, two layers | 1 | 1.366523 | 1.364052 | 1.686927 | 1.680240 | 1.00398x |
| Stress16, two layers | 4 | 0.858342 | 0.878090 | 1.178273 | 1.205914 | 0.97708x |
| HS20, depth one | 1 | 2.589612 | 2.572220 | 2.734322 | 2.721429 | 1.00474x |
| HS20, depth one | 4 | 1.269760 | 1.317346 | 1.418998 | 1.467628 | 0.96687x |
| EDC14 | 1 | 0.254276 | 0.254443 | 0.361162 | 0.362920 | 0.99516x |
| EDC14 | 4 | 0.189480 | 0.192558 | 0.301718 | 0.299348 | 1.00792x |

The preregistered HS20 target-region session-inclusive geometric-mean speedup
is **0.985619x**, paired-bootstrap 95% interval **[0.942911, 1.033391]**.
Its point estimate corresponds to 1.459% longer inclusive execution, not an
established slowdown across arbitrary workloads. It fails both the 5% practical
benefit gate and the lower-bound-above-one gate. All six cells stay within the
5% median-regression bound; the largest observed regression is HS20/four-DPU
at 3.427%. The all-six-cell inclusive aggregate is 0.992502x
([0.971326, 1.013285]); it does not replace the predefined target region.

HS20 kernel medians decrease from 1.785528 to 1.762418 seconds at one DPU and
from 0.458615 to 0.453479 seconds at four DPUs. Its target-region kernel speedup
is 1.012218x ([1.011490, 1.012531]), approximately a 1.207% time reduction.
The small kernel saving does not propagate to a reliable complete-route gain.
All output hashes, planned host-DPU bytes, work ordering and launch counts
remain equivalent. The 24 unique HS20 K1 units dispatch as intended, including
the four-real-product non-fitting cases; the zero-K1 controls remain in the
analysis. These are diagnostic intervals from five blocks, not final-performance
inference or proof that every possible K1 geometry is unhelpful.

**Decision: do not adopt `outer_k1_v1`.** Keep the generic panel geometry and
preserve the tested specialization as negative experimental evidence. No
geometry confirmation, new target region, threshold adjustment or retuning run
is authorized by this result. Its twelve reserved confirmation slots remain
unused. The next physical work is the separately frozen DAG correctness and
fixed-resource scheduling comparison, not another kernel optimization.

- Run: `bc301c5f-0c10-4898-9f7c-50431ac9a76f`.
- Analysis source: `d92e28767bd68a86d8598d90df09afd6b700ba1c`, exact-head
  [CI passed](https://github.com/kazulak/Masters/actions/runs/34159279239).
- Frozen packet SHA-256:
  `6c38f9e1100cd985ff4813ca32c6f761fa0d236c232587a21c69bdd759bc113b`.
- Raw archive SHA-256:
  `1d19c4891493e452432bed598d996a0f19684bf0e6a9db230355d5e80bf499a8`.
- Remote: `/home/tkazulak/evidence/kernel-schedule-geometry-ab-3056090-v1`.
- Local: `runs/eth/safari-baguette1/30560900354b523b2a3a44971f81860a04888640/geometry-ab-v1/`.

All 48 internal raw checksums and both verifiers passed on both hosts. Two
verified copies remain retained; none was deleted. Normalized observations,
per-cell medians/MADs/ranges, paired intervals, decision and analysis checksums
are retained separately. Session-inclusive values are computed per sample;
no old fusion or correctness timings enter this analysis. The whole collection
command took 3:22.55 with peak RSS177,960KiB, including all arms, preparation
and validation, not a per-arm memory comparison. Rank and private-lock release
were independently checked. Total milestone use is now **177 attempts**.

### Remaining Budget

The approved ceiling is **1,051 physical attempts**, not a target to exhaust.
The adjacent [budget manifest](../configs/upmem_kernel_schedule_budget_v1.json)
separates all packets. Warmups, failed attempts, controls and confirmations count.
No implicit additional int8 or ATiM campaign is authorized.

The 96-attempt reserve is fixed: 28 one-shot correctness attempts (seven each
after fusion, geometry, DAG and composition), 36 fresh confirmation attempts
(one preregistered circuit/topology cell, two arms, 1+5, per principal mechanism),
24 final scaling attempts (six routes, 1+3), and eight scalar-MRAM/WRAM ablation
attempts (one small shape, two arms, 1+3). Final scaling uses one DPU at T1/T4/T8/
T16 and two/four DPUs at T8. Scalar ablation is test-only, not another production
kernel family. Correctness manifests must exercise each changed mechanism and
both numeric policies; DAG coverage must include a real concurrent fork-join.

Kernel A/B is three frozen development circuits, one/four DPUs at T8, two arms,
1+5. DAG A/B uses two/four DPUs, equal total resources, generic kernels first.
The P4 48+48 ceiling covers slice exploration and any admitted resident/slice
confirmation; it is not 96 per extension. Record its exact split before runs.
Path budget is 144 initial, 216 adaptive, 72 development confirmation,
36 validation and 72 untouched test attempts. Deduplication leaves slots unused.

Freeze each packet's named cells, source, identities, block order, wall-time cap,
practical threshold and regression bound before candidate timing. Default is a
5% session-inclusive benefit with paired uncertainty supporting improvement and
no unexplained regression beyond 5%; preregister adjustments from baseline
variability, never the observed winner. Inconclusive results stop at the cap.
Correctness, policy replay and full-precision accuracy are distinct gates.

### Static DAG Correctness: Seven Physical Sessions

The frozen generic-panel, unfused static-DAG correctness packet completed on
the unchanged execution source `30560900354b523b2a3a44971f81860a04888640`.
The earlier permission rejection launched no process and consumed no attempt;
execution followed the user's explicit autonomous-execution approval.

- Experiment: `902877a1241df71e1bcb8924faf2677f18ff7c8a755acbcfde7da60c40cd8c90`.
- Run: `265166af-2373-4226-86f9-59bf626c598f`.
- Configuration SHA-256: `a688d00fcd715b2f3d46cb03a266b3fd3e007ba399da2244cd5806d9d278eaab`.
- Packet archive SHA-256: `eec8cb53024441ab0437076c2978baf30a8e972bfe9920deba0b3114fd8c1ea0`.
- Raw archive SHA-256: `b05607cc3106b50104b312bf8645495bc2e7237e7477d30a388e11ae37f540dc`.
- Seven measurements, zero warmups, seven fresh successful sessions; zero
  failed/unsupported attempts, fallback, retries or replacements.
- All seven policy replays pass. All three float32 accuracy checks pass;
  four int8 observations remain separately error-reported, not full-precision
  qualified.
- Rank1, CPU0, powersave governor, SDK2023.1.0; preflight recorded exact clean
  source, unchanged T1/T8 binaries, private lock and free rank ownership.
- Remote and local canonical/strict verification pass; 51 relative raw-file
  checksums and the outer archive checksum verified, with both copies retained.

| Circuit | Numeric policy | DPUs / tasklets | Concurrent native waves |
| --- | --- | --- | ---: |
| Bell2 | float32 | 1 / T1 | 0 |
| Bell2 | shared-scale int8 | 1 / T1 | 0 |
| Stress14 | float32 | 1 / T8 | 0 |
| Stress14 | shared-scale int8 | 1 / T8 | 0 |
| Stress14 | float32 | 4 / T8 | 108 |
| Stress14 | shared-scale int8 | 3 / T8 | 140 |
| Stress14 | shared-scale int8 | 4 / T8 | 108 |

Multi-node cohort summaries bind distinct DPU assignments, shared native
request/response identities, owner-only launch accounting and later dependent
joins. The source-audited synchronous submit/assembly/publication path and
numerical replay support dependency correctness; there is no independent
timestamp-level tensor-publication trace. One-DPU routes test degeneration.
These one-shot observations do not establish a DAG speedup.

Packet qualification passed 72 tests and Ruff. Independent preparation review
caught a raw-array versus dtype/shape-framed output-hash mismatch in the new
verifier before physical execution. A fresh CPU-only supplement reproduces the
original raw hashes and supplies canonical hashes; the original observations
remain preserved. Runtime, kernel, numeric policies and source were unchanged.

Raw evidence is retained locally under
`runs/eth/safari-baguette1/30560900354b523b2a3a44971f81860a04888640/dag-correctness-v1/`
and remotely at
`/home/tkazulak/evidence/kernel-schedule-dag-correctness-3056090-v1`.
Cumulative milestone attempts at this correctness checkpoint were 184.
Geometry confirmation remains unused after its no-go. The subsequent 72-attempt,
equal-resource DAG A/B at two/four DPUs using generic kernels is closed below.
The initial independent reader stopped on a service usage limit; the completed
`dag-correctness-v1/independent_postrun_audit.json` now records
`pass_with_bounded_limits` and no blockers.

The source-only A/B draft retained all three development circuits and 12 arm/cell
combinations (72 scheduled attempts). Four structural tests initially passed. EDC14's
static-DAG routes at two/four DPUs fail the formal dominant-wave tasklet-row
scaling criterion; the unchanged CLI permits these under `diagnostic_v1`.
They remain included with their failed eligibility facts, not filtered out or
promoted to formal scaling claims. At that draft checkpoint it had not yet been
qualified, frozen, admitted or physically executed.
CPU preparation subsequently passed all 12 arm/cell combinations, reproducing
exactly equal outputs across serial/static schedules and two/four DPUs for
each circuit. Canonical output hashes, physical/executable identities and
float32 reference checks are retained separately from physical evidence; five
draft tests pass. No SDK or physical attempt was used for this preparation.

`scripts/analyze_upmem_dag_ab.py` reuses the established paired-block statistics
with explicit serial/DAG labels and the two/four-DPU matrix. The shared
analyzer retains its original one/four-DPU defaults. All 53 focused A/B analysis
tests pass, and complete output equality against the frozen prior analyzer was
checked on the retained 72 fusion observations, including the default 10,000
bootstrap resamples. These are analysis qualification checks, not new timing
evidence. Subsequent controller/verifier qualification, independent review,
freeze and physical execution are reflected in the closure below.

### Static DAG A/B Closure: Accepted Raw and Timing GO

The generic-panel, unfused serial/static-DAG comparison completed at execution
source `30560900354b523b2a3a44971f81860a04888640`: **72 samples / 72 sessions**,
12 warmups and 60 measurements across 12 arm/cell combinations. Each arm/cell
has one warmup and five paired measurement blocks. All 72 samples are accuracy
qualified, with zero failed/unsupported/fallback observations, retries or
replacements. Cumulative milestone attempts are **256**.

The authoritative `dag-ab-v1/acceptance.json` records
`accepted_physical_dag_ab_raw_after_offline_verifier_correction`, with local and
ETH verified copies and released rank1/private lock. The original controller
results remain **physical=0, canonical=0, dag_ab=1**. Its strict verifier failed
with `terminal/sample: backend_id`: plan-level and native-session `backend_id`
and `execution_class` occupy distinct namespaces in source 305. The separately
identified offline correction checks each layer against its exact expected
identity and retains all other shared-field and strict checks. It changed no
raw observations, runtime or frozen packet and required no physical rerun.
The original verifier and failure remain preserved. Twelve correction
regressions passed; local and remote corrected strict verification passed and
produced identical 72-row normalized output.

Ratios below are serial median / DAG median over the five measurement blocks;
values above one favor DAG. Warmups are excluded. Session-inclusive time is
session open + steady execution + session close.

| Circuit | DPUs | Steady ratio | Session-inclusive ratio | Kernel ratio |
| --- | --- | --- | --- | --- |
| Stress16 | 2 | 1.163529 | 1.068863 | 0.954027 |
| Stress16 | 4 | 1.610240 | 1.395117 | 0.902584 |
| HS20 | 2 | 1.141947 | 1.110758 | 1.024015 |
| HS20 | 4 | 1.345870 | 1.295389 | 1.076382 |
| EDC14 | 2 | 1.123528 | 1.087409 | 0.994849 |
| EDC14 | 4 | 1.547584 | 1.373483 | 1.255278 |

The frozen source-203437 analyzer gives an equal-six-cell session-inclusive
geometric speedup of **1.2142126437156118**, paired-bootstrap 95% CI
**[1.1845729381785106, 1.2413408868724962]**, or **17.6421%** time reduction.
Seed `20260909`, 10,000 resamples, medians/raw MAD and common paired-block
resampling are unchanged. The fixed 5% practical-reduction, lower-bound >1,
and maximum 5% per-cell median-regression gates all pass. Both EDC14 cells
remain included despite their failed formal tasklet-row/scaling eligibility;
this is a diagnostic fixed-resource scheduling comparison, not a formal
scaling-eligibility claim. Kernel-only gains are not uniform, as the table shows.

The independent correction audit passed for offline correction adoption, and
`analysis-v1/independent_analysis_audit.json` records GO for timing interpretation:
complete analysis/decision reproduction, 168 median/MAD checks and consistent
summary/CSV timing fields. These results do not authorize production adoption.
At the time of this historical draft, the fresh **12-attempt confirmation was
pending**; its later, separate result is recorded below. No historical A/B
thresholds or timing targets were retuned.

Evidence root:
`runs/eth/safari-baguette1/30560900354b523b2a3a44971f81860a04888640/dag-ab-v1/`.
Authoritative records are `acceptance.json`,
`verifier-repair-v1/correction_report.json` and `analysis-v1/analysis.json`,
`decision.json`, `summary.json` and `cell_summary.csv`.

- Frozen preregistration archive SHA-256: `93e5f95499751e8719b241a73a9368b4a50c89e072afa4418f83d99ed521f074`.
- Original raw archive SHA-256: `000fbc8b2d4e514d7d81e1b935f7d88bef7df1f7acb28f199b21be3a5d3fcd46`.
- Original verifier SHA-256: `ac8b39da44f8e132c534ee8ca078e7cb243034df698bc6604115126dcb78800b`.
- Corrected verifier SHA-256: `9eee8256846486a3505e9148a500513070155eee617f50a8422fba3bf0108eed`.
- Correction archive SHA-256: `e3970c52d452dd0fc15964a5230b972d10a43f31047b5224c58fd1eae5557056`.
- Identical local/remote normalized rows SHA-256: `a4168da6fd94b6e836054f76e8e85e59545fb96cdf3b9e6d9986dcead6f86976`.
- Analysis and independent-audit archive SHA-256: `f07f7f8d933aac6e56b5c3157e1bb6f323ee5acfa7a76f38fa5bde5dcd4bda0c`; six internal files and the outer digest verified locally and on ETH.

The confirmation draft selects the established Stress16/four-DPU/T8 sentinel
after observing this A/B. No DAG-specific confirmation cell was preregistered
before these timings. The proposed twelve fresh attempts are therefore
development confirmation, not an untouched test or a pre-A/B-selected cell.
Their results must remain separate from this dataset. Qualification, immutable
packet freezing and fresh locked hardware admission remain required.

Focused confirmation preparation passed 55 verifier/analysis tests and ten
preparation tests. The actual prepared records pass the unmocked preparation
validator. CPU references are explicitly reused from hash-verified, unchanged
DAG/numeric/physical-plan records; no new CPU replay, SDK execution or physical
attempt is claimed. At that preparation checkpoint, controller qualification
and independent review remained open; the actual acceptance, analysis, audit
and retention closure is recorded below.

### DAG Confirmation Closure: Separate 12-Attempt Development Result

The separate RAW confirmation completed at execution source
`30560900354b523b2a3a44971f81860a04888640` for the post-A/B selected
Stress16 `quantization_stress_16q_l2` sentinel at D4/T8, using the serial and
DAG routes. It contains **12 successful, accuracy-qualified, released
sessions**: 2 warmups and 10 measured observations, with zero failed,
unsupported or fallback observations, retries or replacements. Cumulative
milestone attempts are **268**. The run is one recorded physical run; retrieval
and audit created no new execution and raw observations were not modified.

Warmups are excluded from the summaries. The primary metric is
session-inclusive time (session open + steady + close per observation before
arm medians); ratios are serial median / DAG median, so values above one favor
DAG.

| Metric | Serial median (s) | DAG median (s) | Ratio |
| --- | ---: | ---: | ---: |
| Session-inclusive | 2.367904067 | 1.653880376 | 1.431726322 |
| Steady | 1.996177945 | 1.199680734 | 1.663924318 |
| Kernel | 0.362022523 | 0.400041616 | 0.904962155 |

The inclusive result is **1.4317263**, paired bootstrap 95% CI
**[1.3620203, 1.4483858]**, a **30.154249%** reduction, and passes the
development analysis gates. Kernel-only ratio **0.9049622** means the DAG
kernel median is approximately 10.50% longer; it is diagnostic, not a stable
kernel-level improvement or the inclusive acceptance gate.

The result is a **post-A/B selected development sentinel**, not held-out
evidence and not pooled with the prior completed 72-attempt raw A/B. The prior
raw A/B remains a separate dataset. The raw acceptance is correctness,
provenance and two-copy acceptance only; the derived analysis records
`confirmation_pass`, and the bounded audit records pass with no blocking
findings. The audit is not independent of raw retrieval, so this closure does
not claim independent retrieval.

The execution source is `30560900354b523b2a3a44971f81860a04888640`; the
analysis source is `203437e88f5d8a192bd8b1f1e232f8fcb8c8f699`; and the
qualified reporting source is `879249f52902b861099daaf14161a12ade6148a4`.
The retained raw archive `kernel-schedule-dag-confirmation-3056090-v1.tar.gz`
has SHA-256
`00faa30b7b09706a900c3e638863acef0c600eeadca14de578bfda52bbf08db6`.
The retained derived archive `dag-confirmation-analysis-3056090-v1.tar.gz`
has SHA-256
`1147d1097eda649ecaa580970eb7de084f058fed61ad5d671b7887952b1d5838`.
Both retention records report local and remote verification with no raw
observation modification.

This closes the P3 confirmation gate for later composition only; it does not
establish global production adoption. At that checkpoint P4 was an unfrozen
**44+48** proposal; its subsequent physical results are recorded below.

## P4 Physical Closure

All three packets executed exactly once at clean source
`ce20a6cde88924b30418d5026eabeec2c807c246`. Slice execution reused accepted T8
production binaries built at `30560900354b523b2a3a44971f81860a04888640`, with
exact `src/native` tree identity proof; resident execution used its separately
SDK-qualified test-only ce20 pair binaries. Rank1, SDK2023.1.0, CPU0/powersave
and float32 policies remained fixed. Local/remote raw copies were verified.

| Packet / cell | Warmups / measured | Inclusive speedup | Decision |
| --- | ---: | ---: | --- |
| Resident8: Stress16 D1, host roundtrip vs resident pair | 2 / 6 | 0.959854641 | NO_GO; resident is 4.1824% longer |
| Slice36: Stress16 D2 | 3 / 9 | 0.608675995 | NO_GO |
| Slice36: Stress16 D4 | 3 / 9 | 0.780553338 | NO_GO |
| Slice36: EDC14 D4 | 3 / 9 | 1.124898715 | GO to fresh confirmation; 11.1031% reduction |
| Fresh EDC14 D4 confirmation12 | 2 / 10 | 1.125243871 | CONFIRMATION_PASS; 11.1304% reduction; independently audited |

Slice36 included unsliced serial, sliced serial and sliced static; the table
reports the whole-route unsliced-serial/sliced-static contrast, without pooling.
Confirmation retained only those two EDC14 D4 arms and the same sliced candidate.
Its exact speedup is **1.1252438705643877**, inclusive reduction
**11.130375720382213%**, paired 95% CI
**[1.0597364159578495, 1.3228109896830418]**. Frozen local analysis matches remote.
The unchanged gate is at least 5% inclusive reduction and lower CI greater than
1, using seed20260910 and 10,000 paired resamples; confirmation uses five fresh
measured blocks. Resident primary is subprocess wall through reap; slice primary
is session open + steady + close. These timing boundaries are not interchangeable.

P4 used **56 attempts = 8 + 36 + 12**, all successful observations, advancing
cumulative use from **268 to 324**. No retries or replacements occurred. Stress
confirmation24 and resident confirmation12 remain unused, without reallocation.
Resident's valid negative experiment is accepted. Its controller hit a post-run
`PermissionError` during proc-fd archival; archive-only recovery retained the
original pending record after exclusive-lock/release proof, with no physical rerun.
Slice36's independent postrun audit is **PASS**. Both slice packets have all four
stage exits0, clean terminal source and rank1 `is_owned=0`; the final EDC12
independent postrun audit is **PASS**, with no blockers. Audit SHA-256:
`67cc28b0a3aa39d4144612aef0182acc1932884426eebb879dc6d57396fb7ad8`.

Evidence root: `runs/eth/safari-baguette1/ce20a6cde88924b30418d5026eabeec2c807c246/`.
Use the respective `resident-exploration-v1`, `slice36-exploration-v1` and
`edc14-d4-confirmation-v1` directories; retained analysis, audit/retention records
and raw terminal records are authoritative.

| Raw archive | SHA-256 |
| --- | --- |
| `resident-pair-exploration-ce20a6c-v1.tar.gz` | `74d0afddfc43dba4f68e97e05c8a84eb1861c899c90bacbd393afdbfd6d7865e` |
| `p4-slice36-exploration-ce20a6c-v1.tar.gz` | `bb8d91303440a2f1ff2b1dd76028c206b500e70964a6ccc8b783e2570714e7ad` |
| `p4-edc14-d4-confirmation-ce20a6c-v1.tar.gz` | `374e7fad28ab7a921161db962af84e2f80cddcc6327c42ab764f987319a3f58e` |

This is development-only fixed-route confirmation selected after exploration,
not untouched generalization, a whole-DAG resident result or production adoption.
The negative Stress/resident outcomes are not displaced by the EDC result.
Correctness rests on the qualified runtime's CPU/complex128 validation, exact
canonical output hashes and bound slice/reduction evidence; no offline recovery
of unretained physical tensor bytes is claimed. P4 is closed; next is P5
composition qualification, not additional tuning of these experiments.

## P5 Composition Policy Under Qualification

The proposed retained route combines `packed_wave_v1`, `static_dag_waves_v1`,
`fuse_complex=true` with its existing deterministic non-fit fallback, and
`panel_only_v1`. Float32 is the primary performance policy; shared-scale int8
receives separate correctness/accuracy qualification, not float32 fitted weights.
Slicing remains an explicitly declared exact DAG transformation, not automatic
family-specific selection. Resident execution and outer-K1 dispatch are excluded
from the retained policy. No transfer overlap or multi-rank work is introduced.

Reuse the seven-route changed-checkpoint correctness design with the retained
composition, then the budgeted six-route Stress16 scaling diagnostic: one DPU at
T1/T4/T8/T16 and two/four DPUs at T8, one warmup plus three measurements each.
Reuse the existing 32-combination SDK composition tests. Admission and the
schedule-aware cost adapter must pass software/review gates before freezing the
physical configuration. Separate mechanism speedups must not be multiplied into
an invented combined result. The executor/profile is not yet frozen.

## Qualification and Archival

Changed executable checkpoints need full pinned pytest/Ruff/diff checks,
exact-head CI, T1-T24 builds, strict SDK correctness and independent audit.
Test arithmetic tails/alignment/split-K, int8 extrema, parser corruption/overflow,
operation identity, dependency readiness, memory ownership, partial failure,
timeouts, cleanup and session re-entry. SDK timing is never calibration data.

The sole hardware controller checks occupancy for at most 15 minutes, takes
the private lock, verifies clean exact source/binaries/configuration, rank,
SDK, CPU affinity/governor and writable storage, and never interferes with
another user's process. On failure stop and retrieve the complete partial stage;
no sample retries, replacements or splicing. An infrastructure rerun needs a
proven incident and a new identity without silently exceeding the total ceiling.

Acceptance requires remote sorted relative SHA256SUMS, immediate durable local
retrieval, all checksums, canonical verification, exact sample/session/cell sets,
portable archive plus outer digest, and at least two verified retained copies.
Do not delete volatile originals before this gate. Keep physical and reporting
source identities separate. Never reconstruct missing raw observations.

## Worker Ownership and Next Action

At most two disjoint implementation workers; one lead owns shared protocol and
runtime integration, one independent reader audits, and one controller owns ETH.
Prepared-cohort encoding, native host dispatch, session lifecycle and whole-DAG
execution are connected with SDK correctness coverage. The outer-product
prototype completes the named geometry experiment, with correctness qualified
and performance adoption rejected. The bounded resident pair and exact slice
concurrency have SDK correctness coverage; resident production integration is
not enabled. Fusion is physically confirmed, and outer dispatch has passed its
seven-session correctness gate. Static DAG correctness has now passed seven
physical sessions, both local/remote verifiers and independent post-run review.
The fixed-resource DAG A/B raw is accepted after offline verifier correction,
and its frozen timing gates pass. The separate 12-attempt Stress16 D4/T8
development confirmation is also accepted for later composition only; it does
not establish global production adoption. P4 physical collection is complete as
recorded above; final EDC confirmation postrun audit passed. Next is P5:
reuse existing composition SDK tests and `execution_features.py`, close final-policy
and host-memory admission, and adapt the existing path-cost pipeline to the frozen
schedule/kernel policy. Do not repeat rejected resident/Stress confirmations or
reallocate their unused slots. Joint composition qualification remains open.
The geometry A/B is now complete with a no-go: do not run its reserved
confirmation or silently retune the specialization. The separate DAG
confirmation gate is complete; joint composition qualification is still required.
Neither the A/B results nor this confirmation establish production adoption.
SDK concurrency does not establish physical speedup. No final path fitting starts
before the retained executor and its schedule-aware feature extraction freeze.
