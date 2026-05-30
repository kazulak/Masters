# Thesis Code Map

This directory contains the thesis implementation work and supporting baselines.

## Current Structure

```text
thesis/
+-- PIMutation_reconstruction/
|   +-- QuEST_implementation/      # CPU state-vector baseline inspired by PIMutation
|   +-- simple_python/             # Small reference experiments
+-- 01_MVP_DenseGEMM/              # Current UPMEM dense-GEMM MVP baseline
+-- 02_Modular_UPMEM_TN_Simulator/ # Planning scaffold for the next thesis stage
+-- SimplePIM_*                    # SimplePIM experiments and prototypes
+-- extern/                        # External projects used for comparison/prototyping
+-- SLR/                           # Literature scoping review material
```

## Working Baseline

`01_MVP_DenseGEMM` should remain the reproducible baseline for the current thesis
state:

1. Python builds a small tensor network.
2. `opt_einsum` creates a pairwise contraction path.
3. The host converts each pairwise contraction into tiled GEMM.
4. Tiles are quantized to int8 on the host.
5. A UPMEM DPU executes integer GEMM tiles.
6. The host dequantizes, accumulates, and validates against NumPy.

The next stage should not erase this baseline. New architectural work belongs in
`02_Modular_UPMEM_TN_Simulator` until it is mature enough to replace or extend the
MVP.

## Planning Entry Point

Start with:

- `02_Modular_UPMEM_TN_Simulator/README.md`
- `02_Modular_UPMEM_TN_Simulator/docs/literature_guidelines_mapping.md`
- `02_Modular_UPMEM_TN_Simulator/docs/implementation_plan.md`

The planning scaffold is intentionally documentation-only at this point. It records
the architecture, steps, decisions, experiments, and unresolved ambiguities before
the next implementation pass.
