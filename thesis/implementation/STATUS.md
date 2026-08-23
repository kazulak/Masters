# Implementation Status

This file separates the reset target from the current reset-branch
implementation. Exact tested checkpoints are recorded in
`MIGRATION_LEDGER.md`.

## Status Vocabulary

- **Implemented at base**: code exists on the reset base.
- **Planned**: part of the reset charter; not implemented at base.
- **Simulator-qualified**: tested through the SDK simulator or software tests;
  this is not physical-hardware evidence.
- **Physically-qualified**: supported by a retained physical run for the exact
  route and contract named.
- **Claimable**: permitted by evidence and claim policy for the stated scope.

## Current State

T1D activation: `native/upmem/runtime/` is the active, self-contained ABI-v4
native tree and `src/quantum_bench/upmem/runtime.py` is the active Python
runtime. The active provider preserves raw UPMEM SDK
allocation on an explicit rank path, allocation verification, manual
SimplePIM management metadata construction, initialization-binary launch,
and release. It does not call `table_management_init_with_profile`; the copied
management-profile patch is provenance only because its profile syntax is
incompatible with explicit rank selection. The v4 compute kernel is raw SDK
code using SimplePIM metadata and initialization types, not a SimplePIM
operator.

| Area | Base state | Qualification and claim status |
|---|---|---|
| Circuit and TN code | Reset model, circuit, lowering, and planner boundaries are implemented. | Software-tested; physical claims do not apply to this pure boundary. |
| Logical execution IR | `ContractionDAG` is the sole target-neutral logical execution IR. | Software-tested, including bounded local multi-label slicing. |
| UPMEM plan | Final schema-v1 contract-batch/reduction stages and deterministic output/K work units are implemented. Compatible direct slice branches are grouped under their parent reduction. | Software-tested; branches execute sequentially, so this is not slice concurrency, tasklet scheduling, residency, or physical evidence. |
| UPMEM numeric route | Split-complex float32 and shared-scale host-packed int8 policies are implemented. | CPU policy replay is software-tested; physical numerical qualification is pending. |
| Complex UPMEM policy | Four sequential real ABI-v4 passes consume one final contract stage. | Fake-session differential tests pass; SDK-simulator and physical qualification are pending. |
| Logical slicing | Deterministic local-contraction slicing selects enough contracted labels, creates their Cartesian partials in one pass, and inserts one explicit reduction. | Software-tested through logical planning, physical mapping, CPU replay, and controlled UPMEM sessions; this is bounded local slicing with sequential branch execution, not global slicing or physical parallel evidence. |
| Session and timing API | Final `open_upmem` and persistent `UpmemSession.run_once` execute one DAG sample, preserve native failure stages, and admit close only with positive allocation, identity, execution, and release facts. | Software-qualified with controlled sessions; simulator and physical qualification remain pending. |
| Canonical execution boundary | Direct CPU and final UPMEM APIs consume canonical DAG/plan records; generic execution records survive only inside the historical adapter pending T12 deletion. | Software-tested; no target fallback is admitted. |
| Evidence | Canonical `manifest.json`, `samples.jsonl`, and `sessions.jsonl` schemas plus experiment-owned repetition/session lifecycle are implemented. | Software-tested for strict serialization, failure rows, timing scopes, release admission, and terminal manifest transitions; no evidence is promoted and no physical claim follows. |
| CPU TN reference | Direct same-DAG execution, complex128 validation, and same-physical-plan replay are implemented. | Software-tested; replay is a policy oracle, not a performance baseline. |
| Quimb/cotengra | Existing provider code exists. | Direct baseline adapter is planned. |
| QuEST CPU/GPU | Existing full-state providers exist. | Direct baseline and GPU runtime verification are planned. |
| SimplePIM | Pinned external sources and historical management-assisted routes exist. | SimplePIM compute integration is not active in the reset baseline. |
| PID-Comm, ATiM, SparseP | Repository/history references exist. | Not active in the reset baseline. |
| Physical UPMEM | Historical bounded physical capsules exist. | Reset architecture is pending physical qualification. |

T1C/T1D source-string tests are drift tripwires for the ABI/source
contract; they are not runtime or hardware tests. Clean local SDK builds of
the staged tree were performed at `NR_TASKLETS=1` and `24`. Physical behavior
remains unqualified until ETH qualification.

## Claim Boundary

The repository may describe the reset branch's software implementation of
split-complex final-stage execution and its fake-session differential tests. It
may not claim complex physical execution, general slicing, physical speedup,
multi-rank scaling, energy efficiency, or a hardware-calibrated planner.

## Qualification State

- Software merge state: `pending` until T13 passes.
- Physical qualification state: `pending`; owner `tkazulak`.
- Hardware reservation/date: `pending`.
- Hardware rank path: `pending`.
- UPMEM SDK version: `pending`.
