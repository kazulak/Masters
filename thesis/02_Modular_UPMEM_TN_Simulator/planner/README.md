# Planner

Future home for host-side tensor-network planning.

Responsibilities:

- build tensor networks from circuits or benchmark generators;
- choose contraction paths on the host CPU;
- slice tensors and contractions to respect UPMEM memory limits;
- classify operations before dispatch;
- emit candidate paths for later route-aware planning;
- estimate tensor lifetimes and peak host memory;
- emit task graph v2;
- preserve deterministic planning for reproducible experiments.

Initial implementation should preserve the current MVP behavior by representing
the existing `opt_einsum` pairwise path in task graph v2 before adding new planner
features.

The planner should start as "plan first, route later." Route-aware planning is a
later stage and must use `CostOracle` estimates instead of hardcoding UPMEM
preferences into the pathfinder.
