# Tracked Thesis Results

These directories are tracked evidence snapshots. They have different roles
and must not be treated as one experiment.

| Snapshot | Role | Model/profile/schema | Allowed claims | Not allowed |
| --- | --- | --- | --- | --- |
| [`current/`](current/) | Selected compact mixed research evidence for thesis writing. | Report schema `research_benchmark_pack_v1`; use the recorded normalized result schema, route, and timing metadata. | Claims supported by the normalized rows, validation status, and matching rules in its report. | Claims that combine incompatible routes, unverified GPU rows, or simulator timing with physical hardware performance. |
| [`planner_v2/`](planner_v2/) | Modeled contraction-path planner hypothesis evidence. | Report schema `research_benchmark_pack_v1`; objective `upmem_path_cost_v2`, legacy component model `upmem_pressure_v1`, profiles and normalization recorded in the manifest. | Modeled candidate, objective-component, feasibility, sensitivity, and path-structure comparisons. | Hardware performance, hardware speedup, measured runtime, energy, or executed UPMEM claims. |
| [`physical_hardware_mvp_v1/`](physical_hardware_mvp_v1/) | Physical UPMEM bring-up functionality evidence. | Report schema `research_benchmark_pack_v1`; hardware profile `hardware_mvp_l1_v2`; fixed one-DPU/one-tasklet dense MVP route and recorded validation schema. | Exact CPU-reference validation, allocation, kernel-execution, and directional transfer-accounting functionality. | Performance, speedup, energy, scaling, multi-DPU, generic tensor-network, or quantum-circuit claims. |

`current/` is the default thesis-writing snapshot. `planner_v2/` is a model-only
hypothesis surface, not a hardware benchmark. `physical_hardware_mvp_v1/` is a
small physical functionality surface, not a performance result.

Generated report packs remain under `runs/comparisons/` and are ignored. The
default location is `runs/comparisons/research_pack/<timestamp>/`; a labeled
pack such as `--label planner_v2` is written under
`runs/comparisons/planner_v2/<timestamp>/`. Each namespace retains its own
`latest` link. Raw tensor dumps and native build output are intentionally not
tracked here.
