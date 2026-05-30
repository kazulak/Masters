# Kernels

Future home for DPU kernels grouped by execution route.

Expected route families:

- `simplepim_default`: default UPMEM provider for legal map/reduce, elementwise,
  layout, and simple dense tasks;
- `raw_upmem_dense`: frozen MVP dense GEMM wrapper and performance-control path;
- `custom_dense`: later dense GEMM path for K-tiling, multi-DPU tiling, tasklet
  sweeps, fixed-point, and buffering experiments;
- `heuristic_ops`: diagonal, permutation, scalar, reshape, identity, and trivial
  operations;
- `sparsep`: sparse route if density and conversion cost justify it;
- `collectives`: host collective and PID-Comm-style providers.

Each kernel family should document:

- supported task graph operation kinds;
- supported data formats;
- route eligibility rules;
- WRAM use;
- MRAM transfer layout;
- tasklet strategy;
- preparation steps;
- known limitations.
