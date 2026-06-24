# Roadmap

## Current Baseline

The project currently has three comparable routes in the same pipeline:

1. `cpu_reference`: tensor-network contraction on CPU.
2. `quest_exact_statevector`: exact full-state CPU simulation through QuEST.
3. `raw_upmem_dense`: replay of the `01_MVP_DenseGEMM` UPMEM simulator path for
   Bell 2q and GHZ 4q.

These routes share benchmark YAML, TaskGraph generation, validation records,
execution logs, metrics, and result charts.

## Next Implementation Order

1. Keep expanding CPU and QuEST workloads until local-PC limits are understood.
2. Add GPU only when the local environment has a stable dependency path.
3. Add heuristic route support for diagonal, permutation, scalar, and trivial
   contractions before optimizing dense kernels.
4. Add SimplePIM as the default UPMEM provider once the raw MVP replay remains
   reproducible through this pipeline.
5. Add custom dense UPMEM kernels only after the profiler shows which DenseGEMM
   pieces dominate.
6. Add PID-Comm-style collectives only for explicit slice/reduction tasks.
7. Add SparseP only as a conditional route with conversion cost measured.
8. Move from route-after-planning to route-aware planning when enough profiles
   exist to make the cost model empirical.

## Non-Goals Right Now

- Do not turn providers into microservices.
- Do not rewrite the frozen MVP before it is fully replayable and compared.
- Do not add a backend-specific benchmark runner.
- Do not hide preparation, conversion, or validation cost inside kernel timings.
- Do not treat simulator energy as real hardware energy.

## Done Criteria For A New Route

A new route is thesis-grade only when it:

- is selectable from YAML;
- writes route decisions and rejected-route reasons;
- validates against the CPU reference or an explicitly accepted baseline;
- reports runtime, energy, data movement, and route-specific profiles;
- appears in suite summaries and grouped comparison charts;
- can be disabled without changing source code.
