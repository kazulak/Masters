# Execution-System Integration v1

## Objective and Sequence

The governing plan is `thesis/upmem-system-and-path-optimization-plan.md`.
The bounded objective is to qualify the existing integrated UPMEM executor,
complete the software census, measure bounded one-rank tasklet/DPU scaling, and
evaluate frozen candidates with offline fitting and held-out timing. The
sequence ends with reporting and stop. CPU/DPU placement is removed entirely
from this roadmap, not deferred; new kernels, fused kernels, residency,
multi-rank execution and ATiM integration are not deliverables. Preparation
remains capped at three focused days, excluding hardware queues. This document
does not redefine completion as software integration alone.

| Phase | Current state | Acceptance evidence still required |
| --- | --- | --- |
| A: qualify existing executor | `b921b88`: published; 1,088 strict-SDK tests, Ruff, exact-head CI and corrected 14-cell ETH SDK gate passed | Unchanged seven-cell physical correctness matrix, verified evidence retention and adoption |
| B: software characterization | Required census and optional probe finished at `56b159dc7e8cd945265a6e02dfb5e7c74edf381a`; not physical adoption | Host-only stage complete; raw evidence and two verified copies retained |
| C: bounded one-rank scaling | Not started | Correctness and admission, matched fixed paths, paired measurements and resource-scaling analysis |
| D: frozen-candidate path study | Not started | Frozen candidates, finite physical calibration, offline fitting, frozen profile and held-out evaluation |
| E: report and stop | Not started | Complete raw evidence, checksums, claim boundaries and thesis tables |

The census completes the software characterization task. The separately
completed batching probe at `56b159dc7e8cd945265a6e02dfb5e7c74edf381a` remains
optional preparation evidence. Neither establishes physical adoption nor
replaces the pending seven-cell gate. No additional optimization is required.

## Integrated Lineage

- Main baseline: `fa0dedf628a3612371daa4f6502da4d5465bbaff`.
- Released quantization reporting head: `62505ae637bdd3cf963b70f61754bf90a658b527`.
- Quantization's historical physical execution source remains
  `c0ec6c76439e418e537a953a6b768ce2e1ea0dc6`, distinct from that reporting head.
- Qualified path-infrastructure head: `e225947a84f937629ed46003dad7d8edff160a8f`.
- The two feature heads share the main baseline as their merge base.
- Integration preserves both histories; released commits and tags are not rewritten.
- The separate parallel-runtime-hardening branch is not silently included in
  these two merges. Newly exposed correctness defects require explicit review.

## Qualified Execution Checkpoint

The immutable Phase A execution source is
`b921b8804e324da75222354ee2f4df41e770b75c`. Its hosted CI run is
[33951186529](https://github.com/kazulak/Masters/actions/runs/33951186529).
The local strict-SDK suite passed 1,088 tests with zero failures, errors or
skips. The corrected ETH SDK experiment is
`9c365d1d4126e16cf4f832bfc54cbd2ce1574cb49f81bd69adf5f8c2eb57eacb`,
run `12b8f8fd-ea98-4ba9-9de4-c28dca5a0bd9`: 14 successful samples and
sessions, all policy replays passed, no fallback. Six float32 samples passed
full-precision accuracy; eight int8 samples remain explicitly unqualified
under that accuracy criterion. Stress14 used every planned DPU; Bell2's
intentional idle slots matched its exact plan.

Retained archives and outer SHA-256 values:

- `software-b921b88.tar.gz`:
  `f17ddbd4e8986a889e0fbb7741d58e6497b3a29d16b014f0d8b8636a4a0be107`.
- `sdk-correctness-b921b88.tar.gz`:
  `a52b7baaee95fb9397d3a5f7c35b365943ab65df5fcfa01d51ca70a285652bfe`.

Both were independently copied, safely extracted and checksum-verified
locally and on ETH. Canonical and strict SDK inspection also passed on the
extracted copies. Raw evidence remains under the ignored local
`runs/eth/safari-baguette1/b921b8804e324da75222354ee2f4df41e770b75c/execution-integration-v1/`
and remote `/home/tkazulak/evidence/upmem-execution-integration-b921b88/`.
The earlier `09e19e0` coverage probe remains preserved separately.

The missing temporary integration worktree was restored from its committed
branch at `/home/tom/repos/Masters/.agent-work-execution-integration-v1`.
Roadmap-only descendants do not replace the qualified execution source.
Keep the remote execution checkout at the exact clean `b921b88` source for
the pending, unchanged seven-cell gate; do not rerun software or SDK simply
because the roadmap changed. A production or experiment-definition change
requires a new execution identity and affected qualification.

At the fresh read-only preflight on 5 September 2026, 09:31 UTC, the remote
source was clean and exact but all ranks were owned while another user's
`gwfa_host` ran. No physical attempt was launched and no rank was taken.
The next action remains a fresh bounded hardware admission check, not kernel
development or the superseded 192-attempt calibration. No main merge, tag
or release is implied by this checkpoint.

The automatic overlap resolution preserves explicit simulator selection and
simulator virtual-rank handling while hardware admission still requires one
physical rank. It also preserves physical fail-fast collection, path-specific
output validation, complete workload-manifest hashing, float64 inputs to the
shared-scale quantizer, and the complex accumulation bound `2*K*127^2`.

## Superseded Calibration

The unexecuted 192-attempt generalization calibration is superseded by the
final-system study. Its configurations, candidate pools, split and checksum
files remain byte-for-byte historical preregistration. The separate record is
`thesis_results/upmem_execution_integration_v1/prior_calibration_supersession.json`.
Do not launch it from this branch or relabel its old profiles as calibrated
for integrated int8 execution or the frozen-candidate path study.

## Numerical and Experimental Boundaries

Float32 is the primary equal-quality profile. Shared-scale int8 remains a
distinct numerical policy with its own replay and accuracy facts, not a claim
of float32-equivalent quality. The frozen existing execution policy is the basis
for the path-study fit; any later execution-policy change would invalidate
existing execution-cost fits. Approximation error is not an execution-cost
feature.

Phase A qualification is correctness-only, with no speedup inference. Partial
waves and idle resources are valid correctness cases, not collection-occupancy
failures. Exact allocation, execution admission, provenance, output validation,
session cleanup and no-fallback requirements still apply.

## Bounded Qualification and Audit

The SDK gate contains 14 one-shot cells: Bell2 and Stress14 at float32/int8 T1,
float32/int8 T8 with one and four DPUs, and int8 T8 with three DPUs. The physical
gate contains seven one-shot cells: Bell2 at float32/int8 T1 and Stress14 at the
five T8 routes. Both use one rank, fresh sessions, zero warmups, one measured
block, and observed CPU-0 affinity. These measurements establish correctness,
not comparative performance. No final-test circuit is used for tuning here.

`scripts/qualify_upmem_execution_integration.py` prepares explicit physical
paths and inspects canonical evidence against this exact matrix. It does not
execute hardware. The inspector checks source cleanliness, sample/session
sets, numerical replay, transport, allocation/admission, release and execution
identities. Int8 full-precision error remains descriptive, not silently
admitted under the float32 accuracy threshold.

Independent read-only review found an executable-binding gap in the initial
inspector. It now recomputes each executable ID from the recorded host, DPU and
initializer SHA-256 values, executor and packed transport policy. Mutation
tests reject either an altered executable ID or an altered binary digest.

The first ETH SDK stage at `09e19e0713517a40c2f74e46641f6bd50ad928f4`
completed 14 successful samples and sessions, but exposed a qualification
coverage defect: both Bell2 and Stress4 placed all useful work on DPU 0. The
inspector also conflated allocated resources with active work. This stage is
preserved as a successful SDK execution with insufficient integration coverage,
not accepted as the full multi-DPU qualification. No physical run occurred.

Before any physical timing, software-only lowering selected Stress14: its
largest stage has four work units and uses all four DPUs under both policies.
Bell2 remains as the deliberate partial-wave/idle-DPU case. The corrected
configuration labels end in `sdk-v2` and `physical-v2`, with new experiment
identities. No old observations are replaced or spliced.

The corrected inspector reconstructs each expected physical plan and verifies
its identity and active placements. The existing
`execution_resource_admission_passed` fact tests full allocation utilization,
so it is false for the deliberately idle Bell2 multi-DPU routes. Its value and
reason must match the reconstructed plan; this is not treated as permission
to omit planned work or accept wrong allocation. Stress14 must activate every
requested DPU. Startup allocation, same-policy replay and release remain strict.
See `thesis_results/upmem_execution_integration_v1/sdk_coverage_incident.json`.

That stage's explicit preflight records SDK 2023.1.0, while the old CLI
manifest labels pkg-config's 0.29.1 tool version as the SDK version. The CLI now
queries `dpu-pkg-config --modversion dpu` in both environment and preflight
facts. Historical manifests are not rewritten. This corrects provenance only;
the existing evidence schema, numerical policy and execution code are retained.

The integration audit exposed narrow correctness gaps in the shared boundary:

- Physical mapping rejection must terminate collection after persisting the
  unsupported attempt. A CPU policy-replay exception before opening hardware
  must retain one failed attempt without inventing a session.
- An externally constructed encoded int8 plane must reject -128 because the
  declared arithmetic bound assumes the symmetric range [-127, 127]. Normal
  quantization, rounding and output semantics are unchanged.
- Mutable session requests/pipes require fail-fast single-caller ownership.
  Independent sessions must remain independent; normal cleanup remains valid.
- The pinned SimplePIM initializer resets a shared allocator unconditionally.
  A local wrapper invokes it only on tasklet zero. Other tasklets consume no
  initialized state, and the existing synchronous launch joins them; an extra
  DPU barrier is unnecessary. The submodule and contraction kernel stay intact.

The initializer binary identity changes and therefore requires fresh builds
and qualification. This is not an import of the separate hardening branch's
provider replacement, validation optimization or kernel barrier changes.

Candidate-sidecar admission belongs to the frozen-candidate executable-choice
preflight in Phase D. Phase A uses a fixed greedy matrix without external
candidate handoff; no legacy 192-attempt path calibration is launched here.

This software integration alone does not establish the pending seven-cell
physical gate, bounded one-rank scaling, frozen-candidate calibration, offline
fitting, held-out timing, main merge or release. New kernels, fused kernels,
residency, multi-rank execution and ATiM integration are future research outside
this milestone, not replacement deliverables. CPU/DPU placement remains excluded.

## Research Boundary

The retained ATiM review and primary-source ledger under
`thesis_results/upmem_execution_integration_v1/research/` are background only,
not a current deliverable. They do not establish SDK compatibility or an
optimization win, and no follow-up, runtime dependency or system SDK
installation follows from that review.
