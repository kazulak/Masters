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

## Reconciled Sources

Remote heads were checked on 2026-09-05 before implementation:

| Role | Exact source |
| --- | --- |
| Published main | `fa0dedf628a3612371daa4f6502da4d5465bbaff` |
| Integrated execution and pending physical gate | `b921b8804e324da75222354ee2f4df41e770b75c` |
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
allocation, transfer or physical attempt was made. This blocks P0 physical
acceptance, not source-only census or speculative kernel/scheduler development.

## State and Dependencies

| Phase | Current state | Required exit evidence |
| --- | --- | --- |
| P0 reconcile | Source lineage checked; physical gate pending | Existing seven-session gate at exact `b921b88`, verified and retrieved |
| P1 census | Source-only frontier extension implemented; physical weighting pending | Frozen targets, ready-width/critical-path/liveness facts and benchmark cells |
| P2 kernels | Experimental fusion and K=1 outer-product dispatch connected; software/SDK checkpoint below, physical qualification pending | Separate correctness, native audit, A/B and confirmation for fusion and specialization |
| P3 DAG waves | Static physical plans connected to whole-TN execution and SDK correctness; physical concurrency qualification pending | One launch with independent operation IDs/disjoint DPUs; fixed-resource A/B |
| P4 resident/slice | Not started | Bounded exact slice and local segment decision, qualified or explicit no-go |
| P5 composition | Software accounting and composition qualification in progress | Joint qualification and frozen executor/source/binaries/policies/features |
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

The current runtime splits multi-node stages into sequential nodes. Its v4
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

## Budget and Preregistration

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
prototype completes the named geometry implementation, subject to its
qualification and physical decision. The bounded resident pair and exact slice
concurrency have SDK correctness coverage; resident production integration is
not enabled. Next: physical fusion/outer/DAG gates after P0 access and the
budgeted locality decision. Composition admission,
schedule-aware cost extraction and all physical acceptance gates remain open.
SDK concurrency does not establish physical speedup. No final path fitting starts
before the retained executor and its schedule-aware feature extraction freeze.
