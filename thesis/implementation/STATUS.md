# Implementation Status

This file separates the reset target from the implementation at commit
`869b19c0a2581463b04a35288e9c59352fc6f3b9`.

## Status Vocabulary

- **Implemented at base**: code exists on the reset base.
- **Planned**: part of the reset charter; not implemented at base.
- **Simulator-qualified**: tested through the SDK simulator or software tests;
  this is not physical-hardware evidence.
- **Physically-qualified**: supported by a retained physical run for the exact
  route and contract named.
- **Claimable**: permitted by evidence and claim policy for the stated scope.

## Current State

| Area | Base state | Qualification and claim status |
|---|---|---|
| Circuit and TN code | Existing circuit, TN, and DAG modules exist. | Existing behavior only; the reset model/lowering boundary is planned. |
| Logical execution IR | `ContractionDAG` is the active target-neutral contraction IR. | Simulator-qualified through existing tests; no new reset qualification. |
| UPMEM plan | Bounded v4 compiler, tiling, work units, and identity exist. | Historical route only; not a general slicing, tasklet, or residency plan. |
| UPMEM numeric route | Real float32 and real host-packed int8 paths exist. | Historical simulator/physical capsules may be cited only within their own scope. |
| Complex UPMEM policy | Not implemented. | Planned; not simulator-qualified, physically-qualified, or claimable. |
| Logical slicing | Existing historical slicing code exists, but the reset one-pass contract is absent. | Planned for reset; no reset claim. |
| Session and timing API | Existing versioned runtime/session code exists. | New single-run and timing contracts are planned. |
| Evidence | Existing writers and historical capsules exist. | New manifest/sample/session schema is planned; no evidence is promoted by T0. |
| CPU TN reference | Existing CPU routes exist. | Same-physical-plan replay is planned. |
| Quimb/cotengra | Existing provider code exists. | Direct baseline adapter is planned. |
| QuEST CPU/GPU | Existing full-state providers exist. | Direct baseline and GPU runtime verification are planned. |
| SimplePIM | Pinned external sources and historical management-assisted routes exist. | SimplePIM compute integration is not active in the reset baseline. |
| PID-Comm, ATiM, SparseP | Repository/history references exist. | Not active in the reset baseline. |
| Physical UPMEM | Historical bounded physical capsules exist. | Reset architecture is pending physical qualification. |

## Claim Boundary

At T0, the repository may describe the existing bounded implementation and
historical evidence using their recorded route and evidence identities. It may
not claim that the reset architecture supports complex TN execution, general
slicing, physical speedup, multi-rank scaling, energy efficiency, or a
hardware-calibrated planner.

## Qualification State

- Software merge state: `pending` until T13 passes.
- Physical qualification state: `pending`; owner `tkazulak`.
- Hardware reservation/date: `pending`.
- Hardware rank path: `pending`.
- UPMEM SDK version: `pending`.
