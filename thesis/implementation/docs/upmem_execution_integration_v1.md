# Execution-System Integration v1

## Objective and Sequence

The governing plan is `thesis/upmem-system-and-path-optimization-plan.md`.
The full objective remains integration, shape attribution, one useful kernel
specialization with dispatch and matched ablations, execution-profile freeze,
bounded physical-feedback path search, independent confirmation, untouched
testing, and archived release. This document does not redefine completion as
software integration alone.

| Phase | Current state | Acceptance evidence still required |
| --- | --- | --- |
| A: integrate numerics and path infrastructure | Histories integrated; qualification in progress | Full pinned tests, exact-head CI, SDK matrix, small physical correctness matrix and retained evidence |
| B: shape census and kernel system | Not started | Fixed development paths, shape/cost attribution, generic plus one useful specialization, deterministic physical dispatch, matched ablations and system freeze |
| Optional ATiM probe | Primary-source boundary review complete; no code executed | At most one engineering day and 32 configurations; emitted-code and full-route qualification, or a documented no-go |
| Optional second kernel or CPU placement | Deferred pending measured need | A distinct bottleneck and bounded independent evidence |
| C: final-system path study | Not started | Frozen system, named split, round manifests, fixed query/time budget, paired controls, fitting and profile freeze |
| Confirmation and untouched tests | Not started | Separate confirmation and final-test raw evidence without retuning |
| Closure | Not started | Main adoption after gates, tags, two verified evidence copies, release verification and standalone export |

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
for integrated int8 execution or future kernel dispatch.

## Numerical and Experimental Boundaries

Float32 is the primary equal-quality profile. Shared-scale int8 remains a
distinct numerical policy with its own replay and accuracy facts, not a claim
of float32-equivalent quality. Future dispatch changes invalidate existing
execution-cost fits; approximation error is not an execution-cost feature.

Phase A qualification is correctness-only, with no speedup inference. Partial
waves and idle resources are valid correctness cases, not collection-occupancy
failures. Exact allocation, execution admission, provenance, output validation,
session cleanup and no-fallback requirements still apply.

## Bounded Qualification and Audit

The SDK gate contains 14 one-shot cells: Bell2 and Stress4 at float32/int8 T1,
float32/int8 T8 with one and four DPUs, and int8 T8 with three DPUs. The physical
gate contains seven one-shot cells: Bell2 at float32/int8 T1 and Stress4 at the
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

Candidate-sidecar admission belongs to the final-system executable-choice
preflight in Phase C. Phase A uses a fixed greedy matrix without external
candidate handoff; no legacy 192-attempt path calibration is launched here.

No Phase B kernel choice, ATiM import, physical path-training round, held-out
timing, main merge or release is accepted by this software integration alone.

## External Research

The bounded ATiM review and primary-source ledger are retained under
`thesis_results/upmem_execution_integration_v1/research/`. They establish a
possible generated-C boundary, not SDK compatibility or an optimization win.
No new runtime dependency or system SDK installation follows from that review.
