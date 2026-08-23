# Architecture Reset Migration Ledger

This ledger records the architecture reset. Planned contracts are not
evidence that the corresponding capability is implemented.

## Baseline

- Branch: `refactor/thesis-runtime-simplification`
- Commit: `869b19c0a2581463b04a35288e9c59352fc6f3b9`
- Python: `3.10.12`
- Dependency constraints SHA-256: `b4652b0d4de4bf0a5ee0c429cb89b3f5ecaa4ed4df96cf5e7a058d5823d6e002`
- Baseline test: `1330 passed in 186.28s` with `../.venv/bin/python -m pytest -q`
- Ruff: clean with `../.venv/bin/python -m ruff check src tests scripts`
- Worktree: clean at baseline capture
- Accepted evidence: unchanged; historical evidence remains reachable from the tag

## Non-Negotiable Invariants

- `TensorNetwork` is the target-neutral semantic network and contains no
  execution order. `ContractionDAG` is the only logical execution IR and
  contains the selected order, slicing branches, reductions, and dependencies.
- Numeric, tiling, placement, and kernel choices do not change the DAG hash.
- Physical UPMEM execution never falls back to simulator or CPU.
- Unsupported and failed runs remain explicit evidence rows.
- Hashing and validation remain outside kernel timing.
- Historical evidence is not rewritten by this migration.
- `execution`, `tn`, and `targets/upmem` package initializers are inert;
  callers import symbols from their owning modules so dependencies remain
  visible.

## Base Capability Status

| Capability | Base state | Reset interpretation |
|---|---|---|
| Target-neutral DAG | Implemented at base | Retain as the sole logical execution IR. |
| Bounded UPMEM v4 mapping | Implemented at base | Starting physical mapper; not general slicing, tasklet, or residency scheduling. |
| Real float32 and real host-packed int8 routes | Implemented at base | Preserve while split-complex policies are added. |
| Complex UPMEM execution | Planned | Not implemented or claimable by this reset baseline. |
| Logical multi-label slicing | Planned | Not implemented by this reset baseline. |
| UPMEM slice stages | Planned | Not implemented by this reset baseline. |
| One-row-per-sample evidence | Planned | Existing evidence is not silently migrated by T0. |
| SDK-simulator correctness | Simulator-qualified | Existing simulator tests and historical routes only. |
| Physical UPMEM execution | Historical physical capsules only | Reset architecture is pending physical qualification. |
| Speedup, scaling, energy, general TN claims | Not claimable from T0 | Require later matched physical evidence and claim admission. |

## Work Packages

| Package | State | Exit condition |
|---|---|---|
| WP0 baseline | complete | reset commit, environment facts, inventory, and test result recorded |
| WP1 research contract | complete | concise README and architecture agree on scope and claims |
| WP2 semantic model | complete | direct DAG input validation; no active reverse TaskGraph adapter |
| WP3 planning | complete | active planners consume tensor metadata and emit one planner result/provenance record without TaskGraph lowering |
| WP4 numerics | complete | shared pure encode/contract/decode boundary; conversion, kernel, reduction, and decode timings are non-overlapping |
| WP5 mapping | complete (bounded v4 ownership) | `targets/upmem/compiler.py` owns v4 lowering, identity, geometry, tiling and work assignment; this does not claim slice/tasklet/residency scheduling |
| WP6 runtime | complete | active runtime uses `UpmemV4Executor`/`UpmemV4Session`, `NumericMode`, `UpmemTopology`, and tuple node results; obsolete whole-circuit package removed |
| WP7 baselines | in progress | same-DAG CPU timing is symmetric; external baseline adapters remain to simplify |
| WP8 evidence | pending | one timing/evidence schema and compatibility policy |
| WP9 interface | pending | stable CLI, one experiment schema, and bounded public command set |
| WP10 cleanup | pending | historical/versioned active source is deleted |
| WP11 qualification | pending | software gates pass; physical rerun requirements are explicit |

## Temporary Adapter Expiry

| Adapter | Current location | Must be removed by |
|---|---|---|
| DAG node to `ContractionTask` | removed in `7d497a2` | complete |
| M5 engine/session wrapper | removed; active implementation is `targets/upmem/v4_executor.py` | complete |
| fake `TensorNetworkSpec(None, ...)` input validation | removed in `4907013` | complete |
| `TensorInputs` wrapper | removed in `e66e2a3` | complete |
| one-implementation M5 strategy registry | removed in current WP6 batch | complete |
| projected-prefix planner to `ContractionTask` | removed in current WP3 batch | complete |
| eager legacy imports from `tn/__init__.py` | removed; callers use owning modules | complete |
| eager legacy imports from `targets/upmem/__init__.py` | removed; callers use owning modules | complete |
| M5/v4 defaults in generic contracts | removed in `execution/contracts.py` | complete |
| milestone CLI and Make targets | `bench/__main__.py`, `Makefile` | WP9 |

## Complexity Delta

Update this table after each integration batch.

| Metric | Baseline | Current | Target |
|---|---:|---:|---:|
| Active Python modules | 138 | 135 | 12-16 |
| Class declarations | 307 | 279 | only stable boundary types |
| Test modules | 78 | 76 | about 10 |
| Config files | 63 | 63 | 2 principal experiments |
| Public Make targets | 78 | 78 | 10 or fewer |
| Active contraction IRs | 2 | 1 | 1 |
| Active UPMEM plan schemas | multiple | multiple | 1 |
| Active native ABIs | multiple | multiple | 1 |

## Corrected Ordered Tasks

1. T0: freeze contracts, semantics, identities, baseline, and dependency rules.
2. T1A: move UPMEM plan and tiling ownership.
3. T1B: split Python ABI protocol from native-session lifecycle.
4. T1C: create a self-contained native UPMEM runtime tree.
5. T1D: move the active UPMEM runtime coordinator.
6. T2: create the core model, circuit model, and lowering modules.
7. T3: isolate opt_einsum and cotengra planning adapters.
8. T4A: add results contracts and the CPU single-run API.
9. T4B1: add the UPMEM session and single-run API.
10. T4B2: remove generic execution wrappers and migrate callers.
11. T4C: freeze the final `UpmemStage` and `UpmemPlan` schema.
12. T5A: add evidence schemas and identity serialization.
13. T5B: move repetition, warmup, and session lifecycle to experiments.
14. T5C: normalize timing scopes and remove old active emitters.
15. T6A: implement pure split-complex float32 and shared-scale int8 numerics.
16. T6B: implement CPU replay of the physical UPMEM plan.
17. T7: execute complex policies through the unchanged real-tile ABI v4.
18. T8: implement one-pass logical multi-label slicing.
19. T9: add deterministic slice batches and host reduction stages.
20. T10A: add the Quimb/cotengra direct baseline.
21. T10B: add the QuEST CPU direct baseline.
22. T10C: add QuEST GPU capability and runtime verification.
23. T11A: add the configuration schema and public CLI.
24. T11B: add evidence verification and reporting.
25. T12A: remove providers and routing replaced by direct routes.
26. T12B: remove replaced milestone workflows and configurations.
27. T12C: remove the old TaskGraph and UPMEM plan generations.
28. T13: run software qualification and mark the branch software-ready.
29. T14: perform later ETH physical qualification and create a qualification tag.

## External Build Inputs

- SimplePIM commit: `1d639c53532555f01e9f71d872e7712b166d6cba`
- SimplePIM management patch SHA-256: `5ac09fd1c0a25c234e44615540f2e1585ce162a27a2d4215e5992ddbdf549a0d`
- T0 active v4 tree: `native/upmem/simplepim/upmem_sdk_execution_plan/`
- T0 active v4 build command: `make -C native/upmem/simplepim/upmem_sdk_execution_plan v4 NR_TASKLETS=1`
- T1C target build command, not active at T0: `make -C native/upmem/runtime NR_TASKLETS=<1..24> all`

These are build inputs, not evidence that SimplePIM compute is active in the
reset baseline.
