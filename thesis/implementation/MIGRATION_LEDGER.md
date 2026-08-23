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

- `SimulationQuery` is the public alias `Literal["pre_measurement_statevector"]`.
- `SimulationJob` is frozen and slotted, with fields `circuit`, `query`,
  `parameters`, and `seed` in that order. `make_simulation_job(...)` accepts
  iterable scalar parameters, rejects empty or duplicate keys, sorts by key,
  and constructs the job. Direct construction rejects unsorted or duplicate
  parameters, unsupported queries, and invalid seeds.
- `TensorNetwork` is frozen and slotted, with exactly `circuit`, `tensors`,
  `output_labels`, and `einsum_expression`. It contains no arrays, path,
  slicing, dependencies, target estimates, executor data, or timing.
- `ContractionDAG` is the only logical execution IR and contains the selected
  order, slicing branches, reductions, and dependencies.
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
| WP3 planning | complete (T3) | root opt_einsum/cotengra function adapters return validated pairwise paths and the frozen 14-key provenance mapping; the historical projected-prefix planner is not canonical |
| WP4 numerics | complete (T6A pure policies) | split-complex float32 and shared-scale int8 encode/contract/decode are implemented and verified; complex UPMEM runtime execution and physical validation remain pending |
| WP5 mapping | complete (T4C final staged mapper) | `plan_upmem`, `validate_upmem_plan`, and `physical_plan_id` implement the singleton contract-stage/host-reduce plan; this does not claim slice grouping, tasklet scheduling, residency, or physical execution |
| WP6 runtime | complete (old bounded base ownership only) | the old bounded runtime ownership is retained as migration input; the final reset `UpmemResources`/`UpmemSession` single-run boundary remains implementation-pending in T4B1/T4B2 |
| WP7 baselines | in progress (T4A) | direct same-DAG CPU single-run execution is implemented; physical-plan replay and external baseline adapters remain |
| WP8 evidence | pending | one timing/evidence schema and compatibility policy |
| WP9 interface | pending | stable CLI, one experiment schema, and bounded public command set |
| WP10 cleanup | pending | historical/versioned active source is deleted |
| WP11 qualification | pending | software gates pass; physical rerun requirements are explicit |

## Temporary Adapter Expiry

| Adapter | Current location | Must be removed by |
|---|---|---|
| DAG node to `ContractionTask` | removed in `7d497a2` | complete |
| M5 engine/session wrapper | removed; active implementation is `upmem/runtime.py` | complete |
| fake `TensorNetworkSpec(None, ...)` input validation | removed in `4907013` | complete |
| `TensorInputs` wrapper | removed in `e66e2a3` | complete |
| one-implementation M5 strategy registry | removed in current WP6 batch | complete |
| projected-prefix planner to `ContractionTask` | historical implementation retained for old configurations, tests, and evidence | remove by T12; it is not part of the canonical planner dispatcher |
| eager legacy imports from `tn/__init__.py` | removed; callers use owning modules | complete |
| eager legacy imports from `targets/upmem/__init__.py` | removed; callers use owning modules | complete |
| M5/v4 defaults in generic contracts | removed in `execution/contracts.py` | complete |
| milestone CLI and Make targets | `bench/__main__.py`, `Makefile` | WP9 |

## T2-0 Contract Corrections

These documentation-only entries freeze the T2 boundary at `HEAD 1426226`.
They do not claim that the production implementation already satisfies them.

| Decision | Required migration | Expiry or gate |
|---|---|---|
| Exact `SimulationQuery`, `SimulationJob`, and `make_simulation_job` forms | Implement the public alias, frozen/slotted job, strict direct-construction validation, and functional constructor in `model.py` | T2; numeric dtype is excluded from `SimulationJob` |
| Exact semantic `TensorNetwork` form | Keep only `circuit`, `tensors`, `output_labels`, and `einsum_expression`; keep arrays and execution data outside it | T2; `ContractionDAG` remains the sole logical execution IR |
| Flatten circuit ownership | Retain `quantum_bench/circuits.py`; do not retain both a package and module | T2 |
| Canonical migration route | Check `circuits -> lowering -> planning -> cpu/upmem -> experiment -> evidence/report` | T2 import checks; historical providers are excluded |
| Temporary re-exports | Permit only the `core.records`, narrowed `tn.graph`, and `tn.network` re-exports listed in `docs/reset_contract.md`; `tn.graph` exports no planner, executor, or TaskGraph types | Remove by T12; no duplicate definitions or generic compatibility framework |

T2 implementation order is: model ownership/re-exports, circuits flattening,
lowering ownership and canonical consumer migration, then the full suite.
The temporary `tn.graph` re-export set is exactly `TensorView`, `SliceSpec`,
`ContractNode`, `ReduceNode`, `GraphNode`, and `ContractionDAG`; it contains no
planner, executor, or `TaskGraph` types.
`ARCHITECTURE.md` was reconciled during T2-0 so its ownership, capability, and
migration-order statements match this contract.
T2B circuit ownership flattening is complete in the working tree: the sole
owner is `src/quantum_bench/circuits.py`, direct package-submodule imports are
gone, and the full-suite checkpoint remains deferred to the lead T2 batch.

T2C lowering ownership is complete in the working tree: pure circuit lowering,
input validation, DAG construction, single-label slicing, DAG validation, and
DAG hashing are owned by `src/quantum_bench/lowering.py`; `tn.graph` contains
only the six authorized model type re-exports; `tn.network` contains only the
historical value adapter; canonical execution consumers no longer import either
temporary module; and `test_tensor_network_data.py` was replaced by
`test_lowering.py`. The T2C focused checkpoint passed 181 tests in 1.68s,
Ruff was clean for all changed Python files, and `git diff --check` passed.

The full T2 branch checkpoint passed `1349 passed in 185.50s` with
`../.venv/bin/python -m pytest -q`; Ruff was clean with
`../.venv/bin/python -m ruff check src tests scripts`; and
`git diff --check` was clean.

## T3-0 Planner Contract

T3-0 is documentation-only and frozen at `HEAD e6e97bfb`. It defines the root
`planning.py` API, the exact path/provenance mapping, dependency direction,
canonical adapter set, historical status of the projected-prefix PIM-aware
planner, and explicit unsupported behavior for non-canonical planner engines.
The active implementation status is recorded in the entry below.

## T3 Canonical Planner Migration

T3 is implemented on the reset base `d868151` under the corrected planner
contract. `src/quantum_bench/planning.py` is the only active planner module and
exports only `plan_opt_einsum` and `plan_cotengra`. The M5 circuit study uses a
private engine-to-function helper and stores path/provenance as plain values.
`src/quantum_bench/tn/planning.py` was deleted; historical planner records and
projected-prefix modules remain until T12.

The T3 full-suite checkpoint is green: `1342 passed in 186.46s`. The earlier
focused checkpoint passed 83 tests across the functional planner, planner
models, M5 architecture/study/profile, and lowering tests. Ruff and
`git diff --check` passed for the changed files. Structural counts at this
checkpoint are 120 active Python modules, 278 class declarations, 79 test
modules, 63 configuration files, and 81 public Make targets using the
commands below.

## T4-0/T6A Dependency Correction

T4-0 is contract-frozen and T6A is complete. T4-0 corrected only the
implementation dependency order; it did not redesign the final architecture.
T6A implements the pure split-complex numeric contract needed by the T4A
`run_cpu_once` API: float32 and shared-scale int8 policies, immutable real and
imaginary planes, four-product arithmetic (`rr`, `ii`, `ri`, `ir`), unilateral
contractions, and fail-closed finiteness and overflow checks. The int8 policy
uses one shared scale per complex operand, nearest-even rounding, and bounded
int8 payloads. Focused verification passed 20 tests and Ruff. This completes
the pure numeric module only; CPU/UPMEM execution migration and physical
validation are not claimed.
T4-0 also freezes the recursively immutable result facts, the exact
`Measurement` fields including `preparation_s`, the exact five public UPMEM
plan/resource records, and the `UpmemSession` boundary. The authoritative field
and validation definitions remain in `docs/reset_contract.md` and
`docs/timing.md`; no production implementation is claimed by this entry.

The corrected implementation order is:

```text
T6A -> T4A -> T4C -> T6B -> T7 -> T4B1 -> T4B2 -> T5 -> T8+
```

`T4C` implements the already-frozen final `UpmemStage`/`UpmemPlan` schema.
Runtime constants, ABI identifiers, and executable hashes remain provenance,
not plan fields.
Current WP5 and WP6 `complete` wording above refers only to the old bounded
base; it is not completion of the final reset stage, session, or evidence
implementation.

## T4A Results and CPU Single-Run Boundary

T4A is complete on the reset base. The new results boundary defines immutable
`Measurement` and `ExecutionSample` values, recursively immutable JSON facts,
and explicit `UnsupportedExecution` and `ExecutionFailed` classifications.
The direct `run_cpu_once` route executes one DAG run and returns its output with
measurements; repetition, warmup, evidence writing, and UPMEM session policy
remain outside this route. The independent `run_complex128_reference` path
replays the DAG without timing, hashing, or reuse of the policy executor.

The CPU route validates int8 admission from the requested output dataflow,
preserves deterministic producer-ID reduction order, applies explicit phase
timing, and classifies runtime failures by stage. It also preserves immutable
outputs, finite-result checks, input immutability, and the frozen null-versus-
zero timing semantics. Focused verification passed 43 tests and Ruff, with the
forbidden legacy-import scan and `git diff --check` clean.
The full-suite checkpoint passed 1385 tests in 185.58s, with Ruff clean across `src` and `tests`.

This entry does not claim UPMEM execution migration, the evidence schema, or
physical validation. T4C is recorded below as complete.

## T4C Final Staged UPMEM Mapper

T4C is complete. The final pure UPMEM mapper implements `plan_upmem`,
`validate_upmem_plan`, and `physical_plan_id` with `PLAN_SCHEMA_VERSION = 1`.
The final public plan/resource records temporarily coexist with privately
aliased legacy compiler records until T4B2 removes the old route.

The current generation emits one singleton `contract_batch` per
`ContractNode` and one `host_reduce` stage per `ReduceNode`; T9 slice grouping
is not implemented. Static work and byte fields describe one real ABI-v4 tile
invocation. Runtime-measured transfers remain pending and will be recorded by
the execution/evidence layers.

Mapping fails closed for unsupported topology, numeric policy, ABI/geometry,
int8 bounds, and no-contract work. Internal implementation defects are not
normalized as unsupported results. T4C does not claim complex UPMEM runtime
execution, slicing groups, residency, tasklet scheduling, physical validation,
speedup, scaling, or energy.

The focused final UPMEM set passed 145 tests; an independent reviewer accepted
the implementation with no P0/P1 findings. The corrected full-suite checkpoint
passed 1419 tests in 185.46s. Ruff was clean across `src` and `tests`, and
`git diff --check` passed. The next task is T6B: CPU replay of the physical
UPMEM plan.

## Complexity Delta

Update this table after each integration batch.

| Metric | Baseline | Current | Target |
|---|---:|---:|---:|
| Active Python modules | 138 | 123 | 12-16 |
| Class declarations | 307 | 288 | only stable boundary types |
| Test modules | 78 | 80 | about 10 |
| Config files | 63 | 63 | 2 principal experiments |
| Public Make targets | 78 | 81 | 10 or fewer |
| Active contraction IRs | 2 | 1 | 1 |
| Active UPMEM plan schemas | multiple | multiple | 1 |
| Active native ABIs | multiple | multiple | 1 |

The current counts use the documented commands: `find src/quantum_bench -name
'*.py' ! -name '__init__.py' | wc -l`, the analogous `find tests` command,
`rg '^class ' src/quantum_bench --glob '*.py' | wc -l`, `find configs -type f |
wc -l`, and `rg '^[A-Za-z0-9_.-]+:' Makefile | wc -l`.

## Corrected Ordered Tasks

1. T0: freeze contracts, semantics, identities, baseline, and dependency rules.
2. T1A: move UPMEM plan and tiling ownership.
3. T1B: split Python ABI protocol from native-session lifecycle.
4. T1C: create a self-contained native UPMEM runtime tree.
5. T1D: move the active UPMEM runtime coordinator.
6. T2: create the core model, circuit model, and lowering modules.
7. T3: isolate opt_einsum and cotengra planning adapters.
8. T4-0: freeze the dependency correction; implementation remains pending.
9. T6A: complete; pure split-complex float32 and shared-scale int8 numerics
   verified by 20 focused tests and Ruff.
10. T4A: complete; add immutable results/failure contracts and the direct CPU
    single-run API, independently verified by 43 focused tests and Ruff.
11. T4C: complete; final staged `UpmemStage`/`UpmemPlan` mapper verified by
    145 focused tests and the corrected full-suite checkpoint.
12. T6B: next; implement CPU replay of the physical UPMEM plan.
13. T7: execute complex policies through the unchanged real-tile ABI v4.
14. T4B1: add the UPMEM session and single-run API.
15. T4B2: remove generic execution wrappers and migrate callers.
16. T5A: add evidence schemas and identity serialization.
17. T5B: move repetition, warmup, and session lifecycle to experiments.
18. T5C: normalize timing scopes and remove old active emitters.
19. T8: implement one-pass logical multi-label slicing.
20. T9: add deterministic slice batches and host reduction stages.
21. T10A: add the Quimb/cotengra direct baseline.
22. T10B: add the QuEST CPU direct baseline.
23. T10C: add QuEST GPU capability and runtime verification.
24. T11A: add the configuration schema and public CLI.
25. T11B: add evidence verification and reporting.
26. T12A: remove providers and routing replaced by direct routes.
27. T12B: remove replaced milestone workflows and configurations.
28. T12C: remove the old TaskGraph and UPMEM plan generations.
29. T13: run software qualification and mark the branch software-ready.
30. T14: perform later ETH physical qualification and create a qualification tag.

## External Build Inputs

- SimplePIM commit: `1d639c53532555f01e9f71d872e7712b166d6cba`
- SimplePIM management patch SHA-256: `5ac09fd1c0a25c234e44615540f2e1585ce162a27a2d4215e5992ddbdf549a0d`
- T1D active/self-contained v4 tree: `native/upmem/runtime/`
- T1D active Python runtime: `src/quantum_bench/upmem/runtime.py`
- T1D build command: `make -C native/upmem/runtime NR_TASKLETS=<1..24> all`

The T1C correction is intentional: the staged provider uses raw SDK allocation
with an explicit `backend=hw,rankPath=...` profile, verifies the allocation,
manually constructs SimplePIM management metadata, successfully calls SDK
`dpu_load` and synchronous `dpu_launch` for the initialization binary, and
releases the set. There is no separate initialization terminal record. It does
not call `table_management_init_with_profile`; the copied management-profile
patch is provenance only because its profile syntax is incompatible with
explicit rank selection. The v4 compute kernel is raw SDK code, not a
SimplePIM operator. These inputs are not evidence of SimplePIM compute
integration.

T1D is now the activation gate: Python discovery and the coordinator use
`native/upmem/runtime/`, while the old v4-specific sources and Make v4 target
are deleted. The T1C/T1D source-string tests are drift tripwires; clean local
SDK builds at tasklets 1 and 24 passed, but physical behavior remains
unqualified.
