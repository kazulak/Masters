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
| P2 kernels | Experimental one-launch four-product DPU target; host wiring and specialization pending | Separate correctness, native audit, A/B and confirmation for fusion and specialization |
| P3 DAG waves | Pure scheduler implemented/tested; native execution not wired | One launch with independent operation IDs/disjoint DPUs; fixed-resource A/B |
| P4 resident/slice | Not started | Bounded exact slice and local segment decision, qualified or explicit no-go |
| P5 composition | Not started | Joint qualification and frozen executor/source/binaries/policies/features |
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
passes DPU-compiler syntax checks. This is not yet a production v5 runtime:
operation tables, envelope digests and native host integration remain required.

The 59 focused control tests include Python/C layout, native corruption
rejection, idle controls, explicit non-contiguous spans, and the existing int8
component accumulation bound. Kernels must dereference validated spans rather
than assume canonical offsets. Whole-wave operation tables, payload hashes and
runtime completion correlation remain integration gates, not claims made by the
standalone codec.

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
identity, input digest/scale binding, completion correlation in the native host,
and packed transport integration remain mandatory. Host reconstruction must
retain original lane-major and K-chunk accumulation order when fused outputs
arrive wave-major. Do not interleave lane reductions simply because products
arrive together. A hard DPU/SDK fault can still prevent completion retrieval;
the caller must fail closed and never interpret a missing response as success.
The independent native review found no arithmetic, barrier or addressing blocker
in this experimental target. Physical DPU-index ownership is deliberately a host
gate: the kernel checks range and echoes the supplied index but cannot establish
the host's enumeration mapping. Production dispatch must call admission with the
actual selected DPU index and correlate every completion before publishing data.

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
Current bounded work is the experimental one-launch kernel and completion codec.
Next: wire the admitted operation/control/response contract into the existing
native host, then qualify the geometry specialization and genuine DAG launches.
The pure scheduler alone is not runtime DAG concurrency. No final path fitting starts
before the retained executor and its schedule-aware feature extraction freeze.
