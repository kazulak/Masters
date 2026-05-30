# Literature Guidelines To Architecture Mapping

This document translates the scoping-review recommendations into concrete
planning constraints for the next UPMEM tensor-network runtime.

The cited systems are used carefully:

- when code is integrated and measured, they are providers;
- when code is not integrated, they are design influences;
- in both cases, the thesis must distinguish evidence from intention.

## External Evidence Notes

- UPMEM architecture constraints motivate host orchestration, explicit DMA,
  WRAM-aware tasks, and no dynamic peer-DPU dependency.
- SimplePIM motivates the default UPMEM programming substrate because it provides
  higher-level processing and communication abstractions for UPMEM.
- SparseP motivates a conditional sparse route because it is an SpMV-oriented PIM
  library, not a general tensor-network runtime.
- PID-Comm motivates an optimized collective provider because its contribution is
  collective communication across PIM processing elements.

Source links used while revising this plan:

- UPMEM technology overview: https://www.upmem.com/technology/
- UPMEM architecture description with WRAM/MRAM and no peer-DPU access:
  https://pmc.ncbi.nlm.nih.gov/articles/PMC10159653/
- SimplePIM repository: https://github.com/CMU-SAFARI/SimplePIM
- SparseP repository and paper links: https://github.com/CMU-SAFARI/SparseP
- SparseP arXiv abstract: https://arxiv.org/abs/2201.05072
- PID-Comm overview: https://www.upmem.com/wp-content/uploads/2024/09/08-Siung_ABUMPIMP24_PIDComm_20min.pdf

## Summary Table

| Review guidance | Committed architecture consequence | Planned structure | Evidence needed |
| --- | --- | --- | --- |
| Avoid a monolithic, single-paradigm engine. | Use TaskGraphV2 plus dispatcher-selected providers. | `docs/task_graph_v2.md`, `dispatcher/`, `runtime/`, provider modules. | Ablation results with routes enabled and disabled. |
| Use SimplePIM-style productivity where possible. | Make SimplePIM the default UPMEM provider, but keep it behind a route interface. | `UPMEMBackend/SimplePIMProvider_default`. | SimplePIM vs raw UPMEM timing and error on equivalent tasks. |
| Keep raw UPMEM control. | Wrap the dense MVP as a frozen baseline and escape hatch. | `RawUPMEMProvider_baseline` around `../01_MVP_DenseGEMM`. | MVP replay and raw-vs-SimplePIM comparison. |
| Use dense autotuning ideas. | Dense route owns tile shapes, tasklet count, K-tiling, buffering, and cost metadata. | `CustomDenseProvider` after profiling exists. | Runtime and DMA sensitivity across tile variants. |
| Use SparseP-style sparse handling for sparse workloads. | SparseP is a conditional route selected only when density and conversion cost justify it. | `SparsePProvider` plus density and conversion metadata. | Density threshold curves including conversion and densification cost. |
| Do not brute-force simple operations through GEMM. | Dispatcher must classify operations before contraction. | `HeuristicProvider` for diagonal, permutation, scalar, reshape, identity, trivial operations. | Correctness and speedup of heuristic route versus dense fallback. |
| Host CPU must perform pathfinding. | Planner owns contraction path, slicing, and shape legality. | `planner/`, `Slicer`, `CostOracle`. | Planner time, peak host memory, path quality, route-aware path examples. |
| Respect WRAM and MRAM locality. | Every route declares tile memory and transfer estimates before execution. | TaskGraphV2 cost/slicing fields and runtime WRAM checks. | Per-route WRAM budget checks and DMA byte counts. |
| Avoid default DPU FP32/FP64. | Data format is a first-class route parameter. | `data_formats/`, `DataFormatProviderPort`. | Error/performance tradeoff for int8, fixed-point, block-floating-point. |
| Avoid dynamic inter-DPU communication. | Reductions and reshuffles are explicit collective tasks. | `NaiveHostCollectiveProvider`, then `PIDCommCollectiveProvider`. | Host-reduction overhead and PID-Comm comparison if integrated. |
| Preserve CPU baselines. | CPU remains correctness reference, fallback, pathfinding host, and state-vector comparison. | `CPUBackend`, NumPy/opt_einsum, QuEST/PIMutation baseline. | Correctness metrics and CPU-vs-UPMEM comparisons. |

## Interpretation Of The Cited Systems

### SimplePIM

Architecture decision:

```text
SimplePIMProvider_default is the default UPMEM provider.
```

Use SimplePIM first for elementwise operations, diagonal apply, map/reduce-like
tasks, layout transforms, and new UPMEM kernels when the abstraction exposes
enough cost/profile information.

Do not let SimplePIM replace TaskGraphV2 or the dispatcher. If a hot dense kernel
is faster or clearer in raw/custom UPMEM, the dispatcher can select that route.

### SparseP

Architecture decision:

```text
SparsePProvider is conditional.
```

SparseP should be tested where tensor-network tasks reduce to sparse matrix or
sparse-vector style kernels. The thesis must include conversion cost and must not
claim SparseP helps dense intermediates.

### PID-Comm

Architecture decision:

```text
PIDCommCollectiveProvider belongs to collectives.
```

PID-Comm-style methods should be compared against naive host collectives for
broadcast, gather, scatter, reduce, and sliced aggregation. It should not be used
as the ordinary pairwise contraction route.

### PIMutation / QuEST

Architecture decision:

```text
PIMutation-style work remains a CPU state-vector baseline and heuristic
inspiration.
```

The next-stage runtime is tensor-network centered. State-vector results are still
valuable as a baseline and as evidence that operation-specific heuristics matter.

### ATiM, PRISM, TransPimLib, Alpha-PIM, And Similar Systems

Architecture decision:

```text
Use as motivation unless integration is independently proven useful.
```

These systems can motivate dense autotuning, block formats, quantization, and
library-backed math experiments. They should not enter the critical path before
TaskGraphV2, SimplePIM, raw dense replay, and profiling are stable.

## Required Thesis Argument

The thesis must not claim that modularity is better only by design preference. It
needs ablations where each major route is removed, replaced, or forced.

| Variant | Purpose |
| --- | --- |
| CPU reference only | Validates TaskGraphV2 independent of UPMEM. |
| Raw dense MVP replay | Shows that V2 preserves the existing proof of correctness. |
| SimplePIM default vs raw UPMEM | Measures productivity-route overhead or benefit. |
| Dense-only route | Shows the cost of MVP-style generalized contraction. |
| Dense plus heuristic route | Measures benefit of bypassing GEMM for simple gates. |
| Dense plus SparseP route | Measures sparse benefit after conversion cost. |
| Naive collectives vs PID-Comm collectives | Measures reduction and slicing overhead. |
| Int8 vs fixed-point or block-floating-point | Measures accuracy/performance tradeoff. |
| FLOP path vs route-aware path | Measures whether UPMEM constraints change planning. |

If a route cannot outperform the dense baseline on any well-defined workload, it
should remain a negative result rather than being hidden.

## Integration Acceptance Checklist

Before the thesis says that an external provider is integrated:

- license is compatible with the thesis repository and publication plan;
- build is reproducible on the target machine;
- SDK version compatibility is recorded;
- route interface adapter exists;
- `prepare` and `execute` times are measured separately;
- correctness is validated against CPU reference;
- route can be enabled, disabled, and forced;
- failure modes are logged with concrete reasons.
