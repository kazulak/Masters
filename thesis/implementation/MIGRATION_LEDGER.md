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
| WP2 semantic model | pending | direct DAG input validation; no active reverse TaskGraph adapter |
| WP3 planning | pending | one target-neutral planner/provenance boundary |
| WP4 numerics | pending | host encoding/decoding and accuracy semantics are explicit |
| WP5 mapping | pending | one deterministic `UpmemPlan` owns physical work and assignment |
| WP6 runtime | pending | runtime consumes `UpmemPlan` without `ContractionTask` |
| WP7 baselines | pending | same-DAG CPU timing is symmetric and external baselines are direct |
| WP8 evidence | pending | one timing/evidence schema and compatibility policy |
| WP9 interface | pending | stable CLI, two experiment files, at most ten Make targets |
| WP10 cleanup | pending | historical/versioned active source is deleted |
| WP11 qualification | pending | software gates pass; physical rerun requirements are explicit |

## Temporary Adapter Expiry

| Adapter | Current location | Must be removed by |
|---|---|---|
| DAG node to `ContractionTask` | `execution/upmem.py` | WP6 |
| M5 engine/session wrapper | `targets/upmem/m5_whole_circuit_engine.py` | WP6 |
| fake `TensorNetworkSpec(None, ...)` input validation | `execution/cpu.py` | WP2 |
| M5/v4 defaults in generic contracts | `execution/contracts.py` | WP5 |
| milestone CLI and Make targets | `bench/__main__.py`, `Makefile` | WP9 |

## Complexity Delta

Update this table after each integration batch.

| Metric | Baseline | Current | Target |
|---|---:|---:|---:|
| Active Python modules | 138 | 138 | 12-16 |
| Class declarations | 307 | 307 | only stable boundary types |
| Test modules | 78 | 78 | about 10 |
| Config files | 63 | 63 | 2 principal experiments |
| Public Make targets | 78 | 78 | 10 or fewer |
| Active contraction IRs | 2 | 2 | 1 |
| Active UPMEM plan schemas | multiple | multiple | 1 |
| Active native ABIs | multiple | multiple | 1 |
