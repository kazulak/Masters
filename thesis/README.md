# Thesis Start Here

The active thesis implementation is:

```text
thesis/implementation/
```

Older prototypes, planning scaffolds, and one-off experiments are under
`thesis/legacy/`. They are useful for provenance, but they are not the active
code path.

## Current Direction

The implementation is a benchmarkable quantum-circuit simulation runtime built
around tensor-network contraction and reproducible result artifacts.

Current serious baselines:

- QuEST CPU full-state simulation.
- Quimb/cotengra CPU tensor-network simulation.
- Optional QuEST HIP GPU execution when the local ROCm path verifies real GPU
  computation.
- Strict UPMEM SDK simulator execution for bounded TaskGraph contractions.

UPMEM SDK simulator results execute real SDK DPU programs in simulator mode.
They are not hardware timings and must not be reported as hardware speedup.

Current UPMEM limitation:

> Bounded generic UPMEM contraction exists, but fully general UPMEM TN
> contraction does not yet exist.

## Run The Implementation

Start here:

```bash
cd thesis/implementation
```

Then use the implementation README:

```text
thesis/implementation/README.md
```

That file is the only active command page. The current architecture is in:

```text
thesis/implementation/ARCHITECTURE.md
```

## Legacy Material

Historical direction and audit documents were moved to:

```text
thesis/legacy/direction/
thesis/legacy/audits/
```

The old `01_MVP_DenseGEMM`, `02_Modular_UPMEM_TN_Simulator`, SimplePIM
experiments, and earlier PIMutation reconstruction folders remain legacy
material only.
