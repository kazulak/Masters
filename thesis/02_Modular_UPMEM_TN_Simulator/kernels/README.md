# Kernels

Future home for DPU kernels grouped by execution route.

Expected route families:

- `dense_gemm`: dense integer GEMM tile kernels;
- `heuristic_ops`: simple memory movement or elementwise kernels where worthwhile;
- `sparse`: sparse tensor kernels if density and conversion cost justify them;
- `prototype_simplepim`: productivity-oriented prototypes kept separate from
  performance-critical kernels.

Each kernel family should document:

- supported task graph operation kinds;
- supported data formats;
- WRAM use;
- MRAM transfer layout;
- tasklet strategy;
- known limitations.
