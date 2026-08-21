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

## Work Packages

| Package | State | Exit condition |
|---|---|---|
| WP0 baseline | complete | baseline tag, branch, inventory, and test result recorded |
| WP1 research contract | complete | concise README and architecture agree on scope and claims |
| WP2 semantic model | complete | direct DAG input validation; no active reverse TaskGraph adapter |
| WP3 planning | complete | active planners consume tensor metadata and emit one planner result/provenance record without TaskGraph lowering |
| WP4 numerics | pending | host encoding/decoding and accuracy semantics are explicit |
| WP5 mapping | in progress | deterministic `UpmemPlan` owns work and assignment; active compiler still imports M5/v4 helpers |
| WP6 runtime | in progress | runtime requires compiled node plans and has no strategy registry; M5/v4 engine/session shell remains |
| WP7 baselines | in progress | same-DAG CPU timing is symmetric; external baseline adapters remain to simplify |
| WP8 evidence | pending | one timing/evidence schema and compatibility policy |
| WP9 interface | pending | stable CLI, two experiment files, at most ten Make targets |
| WP10 cleanup | pending | historical/versioned active source is deleted |
| WP11 qualification | pending | software gates pass; physical rerun requirements are explicit |

## Temporary Adapter Expiry

| Adapter | Current location | Must be removed by |
|---|---|---|
| DAG node to `ContractionTask` | removed in `7d497a2` | complete |
| M5 engine/session wrapper | `targets/upmem/m5_whole_circuit_engine.py` | WP6 |
| fake `TensorNetworkSpec(None, ...)` input validation | removed in `4907013` | complete |
| `TensorInputs` wrapper | removed in `e66e2a3` | complete |
| one-implementation M5 strategy registry | removed in current WP6 batch | complete |
| projected-prefix planner to `ContractionTask` | removed in current WP3 batch | complete |
| eager legacy imports from `tn/__init__.py` | `tn/__init__.py` | WP10 |
| M5/v4 defaults in generic contracts | `execution/contracts.py` | WP5/WP6 |
| milestone CLI and Make targets | `bench/__main__.py`, `Makefile` | WP9 |

## Complexity Delta

Update this table after each integration batch.

| Metric | Baseline | Current | Target |
|---|---:|---:|---:|
| Active Python modules | 138 | 138 | 12-16 |
| Class declarations | 307 | 301 | only stable boundary types |
| Test modules | 78 | 77 | about 10 |
| Config files | 63 | 63 | 2 principal experiments |
| Public Make targets | 78 | 78 | 10 or fewer |
| Active contraction IRs | 2 | 1 | 1 |
| Active UPMEM plan schemas | multiple | multiple | 1 |
| Active native ABIs | multiple | multiple | 1 |
