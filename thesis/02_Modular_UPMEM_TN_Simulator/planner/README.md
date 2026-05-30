# Planner

Future home for host-side tensor-network planning.

Responsibilities:

- build tensor networks from circuits or benchmark generators;
- choose contraction paths on the host CPU;
- classify operations before dispatch;
- slice tensors and contractions to respect UPMEM memory limits;
- emit task graph v2;
- preserve deterministic planning for reproducible experiments.

Initial implementation should preserve the current MVP behavior by representing
the existing `opt_einsum` pairwise path in task graph v2 before adding new planner
features.
