# Literature Guidelines To Architecture Mapping

This document translates the scoping-review recommendations into concrete planning
constraints for the next UPMEM tensor-network simulator.

## Summary Table

| Review guidance | Architecture consequence | Planned structure | Evidence needed |
| --- | --- | --- | --- |
| Avoid a monolithic, single-paradigm engine. | Introduce a dispatcher and route-specific modules. | `dispatcher/`, `kernels/`, `runtime/`, route metadata in task graph v2. | Ablation results with routes enabled and disabled. |
| Use SimplePIM-style productivity only where appropriate. | Keep a prototype route separate from performance-critical dense contraction. | `kernels/prototype_simplepim` later, not the dense default. | Compare prototype route overhead against direct UPMEM SDK route. |
| Use ATiM-style dense autotuning ideas. | Dense route owns tile shapes, tasklet count, buffering, and cost metadata. | Dense route under `kernels/dense_gemm`; autotune records in task graph v2. | Runtime and DMA sensitivity across tile variants. |
| Use SparseP-style sparse handling for sparse workloads. | Sparse route is separate from dense GEMM and selected only when density justifies it. | Future `kernels/sparse` route plus density metadata. | Density threshold curves, sparse format conversion cost. |
| Do not brute-force algebraically simple operations through GEMM. | Dispatcher must classify operations before contraction. | `dispatcher/` route rules for permutation, diagonal, controlled, and mergeable gates. | Correctness and speedup of heuristic route versus dense fallback. |
| Host CPU must perform pathfinding. | Planner owns contraction path, slicing, and shape legality. | `planner/` and `docs/task_graph_v2.md`. | Planner time, peak host memory, and path quality metrics. |
| Respect 64 KiB WRAM and MRAM-WRAM locality. | Runtime must make tile memory explicit and reject illegal plans. | `runtime/` plus tile-budget fields in task graph v2. | Per-route WRAM budget checks and DMA byte counts. |
| Avoid standard DPU FP32/FP64 as the default. | Data format is a route parameter, not an afterthought. | `data_formats/` and format fields in task graph v2. | Error/performance tradeoff for int8, fixed-point, block floating point, and any library route. |
| Avoid dynamic inter-DPU communication. | Reductions and data reshuffles are host-mediated. | `runtime/host_collectives` later; reduction tasks in task graph v2. | Host reduction cost and rank-allocation sensitivity. |

## Interpretation Of The Cited Systems

The plan treats the cited systems as architectural patterns unless direct code
integration is later proven practical.

- `SimplePIM @chen_simplepim_2023`: useful for rapid streaming-kernel prototypes,
  but not assumed to be sufficient for high-reuse tensor contraction.
- `ATiM @shin_atim_2024`: used as motivation for dense-route autotuning and
  tile/buffer search, not as a committed dependency.
- `SparseP @giannoula_sparsep_2022`: used as motivation for a sparse route with
  density-aware dispatch.
- `PIMutation @lee_pimutation_2025`: used as motivation for operation-specific
  state-vector heuristics such as row-swapping and gate merging.
- `Alpha-PIM @barkhordar_alpha-pim_2025`, `PRISM @pacheco_prism_2025`, and
  `TransPimLib @item_transpimlib_2023`: used as motivation to compare integer,
  quantized, block-floating-point, and library-backed math formats.
- `PID-Comm @noh_pid-comm_2024`: used as motivation for host-mediated collective
  scheduling and rank allocation.

## Required Thesis Argument

The thesis should not claim that modularity is better only by design preference.
It needs an ablation table where each major route is removed or replaced:

| Variant | Purpose |
| --- | --- |
| Dense-only route | Shows the cost of the MVP-style generalized path. |
| Dense plus heuristic route | Measures benefit of bypassing GEMM for simple gates. |
| Dense plus sparse route | Measures benefit on sparse circuits/tensors. |
| Dense with and without autotuning | Measures benefit of hardware-aware tiling. |
| Int8 versus alternate formats | Measures accuracy/performance tradeoff. |
| Host collective strategies | Measures reduction and slicing overhead. |

If a route cannot outperform the dense baseline on any well-defined workload, it
should remain a negative result rather than being hidden.
