# Architecture Simplification Migration

This temporary ledger tracks the reset defined by
`ARCHITECTURE_SIMPLIFICATION_AUDIT_AND_AGENT_GUIDE.md`. It is not part of the
final documentation set.

## Baseline

- Tag: `pre-thesis-runtime-simplification`
- Commit: `c20e634`
- Branch: `refactor/thesis-runtime-simplification`
- Active Python modules: 138
- Python class declarations: 307
- Test modules: 78
- Configuration files: 63
- Public Make targets: 78
- Baseline test: `1323 passed in 176.76s` with `../.venv/bin/python -m pytest -q`
- Accepted evidence: unchanged; historical evidence remains reachable from the tag

## Non-Negotiable Invariants

- `ContractionDAG` remains target-neutral and is the only active contraction IR.
- Numeric, tiling, placement, and kernel choices do not change the DAG hash.
- Physical UPMEM execution never falls back to simulator or CPU.
- Unsupported and failed runs remain explicit evidence rows.
- Hashing and validation remain outside kernel timing.
- Historical evidence is not rewritten by this migration.
- Package-level convenience re-exports are not preserved; active callers import
  symbols from their owning modules so dependencies remain visible.

## Work Packages

| Package | State | Exit condition |
|---|---|---|
| WP0 baseline | complete | baseline tag, branch, inventory, and test result recorded |
| WP1 research contract | complete | concise README and architecture agree on scope and claims |
| WP2 semantic model | complete | direct DAG input validation; no active reverse TaskGraph adapter |
| WP3 planning | complete | active planners consume tensor metadata and emit one planner result/provenance record without TaskGraph lowering |
| WP4 numerics | complete | shared pure encode/contract/decode boundary; conversion, kernel, reduction, and decode timings are non-overlapping |
| WP5 mapping | in progress | deterministic `UpmemPlan` owns work and assignment; active compiler still imports v4 helpers |
| WP6 runtime | complete | active runtime uses `UpmemV4Executor`/`UpmemV4Session`, `NumericMode`, `UpmemTopology`, and tuple node results; obsolete whole-circuit package removed |
| WP7 baselines | in progress | same-DAG CPU timing is symmetric; external baseline adapters remain to simplify |
| WP8 evidence | pending | one timing/evidence schema and compatibility policy |
| WP9 interface | pending | stable CLI, two experiment files, at most ten Make targets |
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
| M5/v4 defaults in generic contracts | `execution/contracts.py` | WP5/WP6 |
| milestone CLI and Make targets | `bench/__main__.py`, `Makefile` | WP9 |

## Complexity Delta

Update this table after each integration batch.

| Metric | Baseline | Current | Target |
|---|---:|---:|---:|
| Active Python modules | 138 | 134 | 12-16 |
| Class declarations | 307 | 279 | only stable boundary types |
| Test modules | 78 | 76 | about 10 |
| Config files | 63 | 63 | 2 principal experiments |
| Public Make targets | 78 | 78 | 10 or fewer |
| Active contraction IRs | 2 | 1 | 1 |
| Active UPMEM plan schemas | multiple | multiple | 1 |
| Active native ABIs | multiple | multiple | 1 |
