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
| P1 census | Retained preparation reused; frontier extension in progress | Frozen targets, ready-width/critical-path/liveness facts and benchmark cells |
| P2 kernels | Not implemented | Separate correctness, native audit, A/B and confirmation for fusion and specialization |
| P3 DAG waves | Not implemented | One launch with independent operation IDs/disjoint DPUs; fixed-resource A/B |
| P4 resident/slice | Not started | Bounded exact slice and local segment decision, qualified or explicit no-go |
| P5 composition | Not started | Joint qualification and frozen executor/source/binaries/policies/features |
| P6 paths | Not started | New bounded physical data, offline profile, untouched test and raw evidence |
| P7 release | Not started | Source lineage, checksummed portable bundle and two verified copies |

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
Current bounded work is P1 frontier/liveness analysis and the native-boundary
audit. Next: finish and test P1, freeze opportunity/eligibility facts, then build
the required protocol/kernel/scheduler prototypes. No final path fitting starts
before the retained executor and its schedule-aware feature extraction freeze.
