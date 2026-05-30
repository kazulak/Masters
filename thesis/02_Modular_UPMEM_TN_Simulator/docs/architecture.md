# Architecture

The next-stage simulator should be a host-orchestrated, route-dispatched tensor
network contraction system. DPUs execute small numeric kernels; the host owns graph
construction, pathfinding, slicing, routing, data movement, and reductions.

## Data Flow

```text
Circuit / QASM / generated benchmark
        |
        v
Tensor-network builder
        |
        v
Host contraction planner
  - pathfinding
  - slicing
  - tensor lifetime planning
  - candidate operation classification
        |
        v
Task graph v2
        |
        v
Dispatcher
  - route selection
  - data-format selection
  - ablation switches
  - expected cost logging
        |
        +--> heuristic route
        +--> dense GEMM route
        +--> sparse route
        +--> prototype/SimplePIM route
        +--> host collective route
        |
        v
UPMEM runtime and host aggregation
        |
        v
Output amplitudes / observables / validation metrics
```

## Components

### Planner

The planner is a host-side module. It should:

- Build tensor networks from circuits.
- Select contraction paths using established host libraries.
- Slice contractions so every DPU task respects WRAM and MRAM transfer limits.
- Emit a task graph with explicit tensor shapes, labels, routes, dependencies, and
  memory estimates.
- Preserve a deterministic mode for reproducible thesis measurements.

The current MVP already uses `opt_einsum` for pathfinding. The next stage can keep
that initially and later evaluate a tensor-network-focused planner if needed.

### Dispatcher

The dispatcher receives task graph nodes and selects execution routes. It should
not execute kernels directly. It should produce a route decision record containing:

- selected route;
- rejected candidate routes;
- reason for the decision;
- estimated bytes moved;
- estimated integer operations or host operations;
- selected data format;
- ablation flags active during the run.

The dispatcher is the main mechanism for thesis ablation studies.

### Execution Routes

The route modules should be isolated behind a small contract:

```text
can_execute(task, tensor_metadata, hardware_state) -> yes/no plus reason
estimate(task, format, hardware_state) -> cost record
execute(task, tensors, runtime_context) -> output tensor handle and profile record
```

Planned routes:

| Route | Use case | Notes |
| --- | --- | --- |
| `heuristic_ops` | Row swaps, permutations, diagonal gates, trivial contractions, gate merging. | Host-controlled and may use DPU only if data movement is justified. |
| `dense_gemm` | General dense tensor contractions. | Starts from `01_MVP_DenseGEMM`, then adds K-tiling, multi-DPU dispatch, and double buffering. |
| `sparse` | Tensors whose density is low enough to justify sparse representation. | Requires explicit conversion-cost accounting. |
| `prototype_simplepim` | Fast experiments for simple streaming kernels. | Kept separate from performance-critical dense kernels. |
| `host_collective` | Reductions, reshapes, gather/scatter, sliced result aggregation. | Required because DPUs cannot communicate directly. |

### Runtime

The runtime owns interaction with UPMEM:

- DPU allocation and rank selection.
- DPU binary loading.
- DMA transfers.
- MRAM/WRAM buffer layout.
- Double-buffering when supported by a route.
- Kernel launch timing.
- Host-mediated reductions and result assembly.

Runtime code must expose timing counters for host-to-DPU DMA, DPU compute,
DPU-to-host DMA, host packing, host unpacking, and host reductions.

### Data Formats

Complex-valued tensors should not imply FP64 execution on DPUs. The data-format
layer should describe:

- storage format;
- scale metadata;
- accumulation type;
- dequantization path;
- error model;
- supported routes.

Initial required formats:

| Format | Purpose |
| --- | --- |
| `complex_f64_host` | Reference and validation format on CPU. |
| `complex_i8_tile_scaled` | Current MVP-style baseline format. |
| `fixed_point` | Candidate for integer-native DPU execution. |
| `block_floating_point` | Candidate for larger dynamic range with shared exponent. |

Library-backed formats can be added later if the dependency and license situation
is clear.

## Core Invariants

1. DPUs never search contraction paths.
2. DPUs never request data from other DPUs.
3. Every DPU task declares its WRAM requirement before execution.
4. Every route can be disabled for ablation.
5. Every non-reference data format reports an error metric against a host
   reference.
6. Host orchestration is measured separately from DPU execution.
7. The current dense-GEMM MVP remains a baseline, not hidden history.

## Expected Evolution From The MVP

| MVP behavior | Next-stage behavior |
| --- | --- |
| One direct dense int8 GEMM route. | Multiple explicit routes selected by dispatcher. |
| One DPU allocation. | Runtime can allocate multiple DPUs/ranks with host reduction. |
| K-tiling rejected. | K-tiling is supported and measured. |
| Hardcoded circuits. | Benchmarks can include QASM/generator-based circuits. |
| JSON task graph optimized for MVP only. | Task graph v2 records routing, format, slicing, and profiling metadata. |
| Correctness-first timing report. | Reproducible experiment records for thesis tables. |
