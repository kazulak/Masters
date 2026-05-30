# Experiment Plan

The implementation should be evaluated as a set of architectural choices, not as
a single final runtime number. Every experiment must report route configuration,
data format, correctness, and timing breakdown.

## Research Questions

1. Can TaskGraphV2 replay the dense MVP without losing correctness or
   reproducibility?
2. What overhead does the generalized dispatcher/runtime add compared with the
   frozen MVP?
3. When does SimplePIM match, beat, or lose to raw UPMEM for equivalent tasks?
4. Which operation classes benefit from heuristic routing?
5. How sensitive is dense contraction to tile shape, tasklet count, K-tiling,
   DPU count, and buffering?
6. Which data format provides the best accuracy/performance tradeoff on UPMEM?
7. How much overhead comes from collective reductions in multi-DPU execution?
8. When do PID-Comm-style collectives beat naive host collectives?
9. When does SparseP beat dense execution after conversion and densification cost?
10. Can route-aware planning choose a better contraction path than pure
    FLOP-minimizing planning under UPMEM constraints?
11. How does the UPMEM tensor-network approach compare to CPU tensor-network and
    CPU state-vector baselines?

## Baselines

| Baseline | Purpose |
| --- | --- |
| NumPy or opt_einsum CPU tensor contraction | Correctness and tensor-network CPU reference. |
| QuEST/PIMutation CPU state-vector | State-vector control group and heuristic inspiration. |
| `01_MVP_DenseGEMM` | Frozen UPMEM dense int8 MVP. |
| V2 CPU reference route | Debug reference for TaskGraphV2. |
| V2 raw dense route | MVP logic behind the route interface. |
| V2 SimplePIM default route | Default UPMEM provider comparison. |
| Optional GPU dense baseline | High-performance dense reference if cheaply available. |

## Workloads

Start small and grow only when instrumentation is stable.

| Workload group | Examples | Purpose |
| --- | --- | --- |
| Sanity circuits | Bell 2q, GHZ 4q | Regression and quick validation. |
| PIMutation-style circuits | BB84, BV, EDC, HS, QRNG, XOR | Compare against existing CPU baseline. |
| Random circuits | Fixed seed, depth sweep, qubit sweep | Average-case stress testing. |
| Entanglement-heavy circuits | Random two-qubit layers, QFT-like circuits if added | Stress dense contraction and slicing. |
| Diagonal-heavy circuits | Phase gates, QAOA-like diagonal layers | Test heuristic and sparse alternatives. |
| Permutation-heavy circuits | X/CNOT/SWAP-heavy patterns | Test heuristic routing and layout transforms. |
| Sparse synthetic tensors | Controlled density sweep | Find SparseP threshold. |
| Multi-DPU sliced dense tasks | Fixed contraction with varying DPU count | Measure collective overhead. |

## Metrics

Correctness:

- max absolute error;
- max relative error;
- state norm drift;
- fidelity when a full reference state is available;
- observable error if full state comparison becomes too expensive;
- pass/fail status under a recorded tolerance.

Performance:

- total wall time;
- host planning time;
- dispatcher time;
- route preparation time;
- host packing/unpacking time;
- format conversion or quantization time;
- host-to-DPU DMA time and bytes;
- DPU kernel time;
- DPU-to-host DMA time and bytes;
- host reduction/collective time;
- validation time;
- peak host tensor memory;
- estimated and measured WRAM use;
- DPU count, rank count, and tasklet count;
- energy when a reliable meter is available.

Route behavior:

- selected route counts;
- rejected route counts and reasons;
- forced-route failures;
- fallback counts;
- tile shapes selected;
- data format selected;
- sparse density estimates;
- collective provider selected;
- route-aware planner path score.

## Ablation Matrix

| Variant | What it isolates |
| --- | --- |
| CPU reference only | TaskGraphV2 correctness independent of UPMEM. |
| Frozen MVP | Current baseline behavior and timing. |
| V2 raw dense replay | Overhead of V2 route interface and logging. |
| SimplePIM forced | SimplePIM overhead and suitability. |
| Raw UPMEM forced | Raw dense control path. |
| Default dispatch | Dispatcher behavior under normal policy. |
| Dense-only V2 | Cost of generalized dense route. |
| Dense plus heuristic | Benefit of bypassing GEMM for simple operations. |
| Dense plus SparseP | Benefit of sparse representation after conversion cost. |
| Naive collectives only | Baseline multi-DPU reduction overhead. |
| PID-Comm collectives | Benefit of optimized collective provider. |
| Fixed tile vs tile sweep | Benefit of hardware-aware dense tuning. |
| Single DPU vs multi-DPU | Scaling and collective overhead. |
| Int8 vs fixed-point | First accuracy/performance data-format comparison. |
| Fixed-point vs block-floating-point | Later dynamic-range comparison if implemented. |
| Double buffering off vs on | DMA/compute overlap benefit if implemented. |
| FLOP-min path vs route-aware path | Planning benefit under UPMEM costs. |

## Experiment Families

### E1: MVP Replay

Purpose: prove that V2 preserves existing behavior.

Compare:

- frozen MVP;
- V2 CPU reference;
- V2 raw dense replay.

Report:

- correctness against NumPy;
- overhead of V2 logging/dispatch;
- exact route decision log.

### E2: SimplePIM Default Provider

Purpose: decide where SimplePIM should remain default.

Compare:

- SimplePIM forced;
- raw UPMEM forced;
- CPU reference;
- default dispatch.

Report:

- preparation time;
- kernel time;
- DMA time;
- total time;
- error;
- route rejection/fallback reasons.

### E3: Dense Route Scaling

Purpose: make dense contraction thesis-grade.

Sweep:

- tile rows;
- tile columns;
- `tile_k`;
- tasklets;
- DPU count;
- K-tiling on/off;
- double buffering on/off if available.

Report:

- throughput;
- DMA bytes;
- WRAM feasibility;
- reduction cost;
- error by data format.

### E4: Heuristic Route

Purpose: test whether circuit structure matters.

Compare:

- heuristics off;
- heuristics on;
- forced dense fallback where legal.

Report:

- operations bypassed;
- runtime saved or lost;
- tensor size changes from fusion;
- correctness.

### E5: SparseP Threshold

Purpose: find where sparse PIM kernels help.

Sweep:

- density;
- sparsity pattern;
- conversion format;
- downstream dense vs sparse consumer.

Report:

- conversion time;
- SparseP execution time;
- densification time when needed;
- dense fallback time;
- threshold where sparse wins or loses.

### E6: Collectives

Purpose: quantify multi-DPU aggregation costs.

Compare:

- naive host gather/reduce;
- PID-Comm-style provider if integrated;
- single-DPU fallback.

Report:

- collective size;
- DPU count;
- host reduction time;
- total runtime;
- correctness after reduction.

### E7: Route-Aware Planning

Purpose: show mature research contribution.

Compare:

- pure FLOP-minimizing path;
- route-aware path using transfer, format, slicing, and collective penalties.

Report:

- selected path;
- estimated score components;
- measured runtime;
- correctness;
- reason for path difference.

## Reporting Rules

- Always report hardware and SDK details.
- Always report route configuration and ablation flags.
- Always include correctness metrics next to runtime metrics.
- Always include preparation and conversion cost.
- Do not compare a quantized UPMEM route against a CPU FP64 baseline without
  reporting the induced error.
- Do not hide fallback routes.
- Treat negative results as useful results when the workload and measurement are
  controlled.

## Minimum Final Figure Set

The final thesis should have at least:

1. MVP vs V2 replay correctness and overhead.
2. SimplePIM vs raw UPMEM comparison.
3. Dense route tile/DPU scaling.
4. Data-format speed/error tradeoff.
5. Heuristic route ablation.
6. Sparse density threshold or negative result.
7. Collective provider comparison or measured naive baseline with integration
   limitation.
8. Route-aware path example.
