# Experiment Plan

The implementation should be evaluated as a set of architectural choices, not as a
single final runtime number.

## Research Questions

1. Does modular dispatch outperform a dense-only generalized path?
2. Which operation classes benefit from heuristic routing?
3. When does sparse execution beat dense int8 GEMM after conversion overhead?
4. How sensitive is dense contraction to tile shape, tasklet count, K-tiling, and
   buffering?
5. Which data format provides the best accuracy/performance tradeoff on UPMEM?
6. How much overhead comes from host-mediated reductions in multi-DPU execution?
7. How does the UPMEM tensor-network approach compare to the CPU state-vector
   baseline from the PIMutation reconstruction?

## Baselines

| Baseline | Purpose |
| --- | --- |
| QuEST/PIMutation CPU state-vector | SOTA-style CPU control group for state-vector scaling. |
| NumPy or opt_einsum CPU tensor contraction | Correctness and tensor-network CPU reference. |
| `01_MVP_DenseGEMM` | Current UPMEM dense int8 MVP. |
| V2 host reference route | Debug reference for the new task graph. |
| V2 dense-only route | Main comparison point for dispatch ablations. |

## Workloads

Start small and grow only when instrumentation is stable.

| Workload group | Examples | Purpose |
| --- | --- | --- |
| Sanity circuits | Bell 2q, GHZ 4q | Regression and quick validation. |
| PIMutation-style circuits | BB84, BV, EDC, HS, QRNG, XOR | Compare against existing CPU baseline. |
| Random circuits | Fixed seed, depth sweep, qubit sweep | Average-case stress testing. |
| Entanglement-heavy circuits | Random two-qubit layers, QFT-like circuits if added | Stress dense contraction and slicing. |
| Sparse/structured circuits | Diagonal-heavy, permutation-heavy, low-density tensors | Test heuristic and sparse routes. |

## Metrics

Correctness:

- max absolute error;
- max relative error;
- state norm drift;
- fidelity when a full reference state is available;
- observable error if full state comparison becomes too expensive.

Performance:

- total wall time;
- host planning time;
- dispatcher time;
- host packing/unpacking time;
- host-to-DPU DMA time and bytes;
- DPU compute time;
- DPU-to-host DMA time and bytes;
- host reduction time;
- peak host tensor memory;
- DPU count and rank count;
- energy when RAPL or another reliable meter is available.

Route behavior:

- selected route counts;
- forced-route failures;
- fallback counts;
- tile shapes selected;
- data format selected;
- sparse density estimates.

## Ablation Matrix

| Variant | What it isolates |
| --- | --- |
| Host reference only | Planner and task graph correctness. |
| Dense-only V2 | Cost of generalized dense route. |
| Dense plus heuristic | Benefit of bypassing GEMM for simple operations. |
| Dense plus sparse | Benefit of sparse representation after conversion cost. |
| Dense fixed tile versus autotuned tile | Benefit of hardware-aware dense tuning. |
| Single DPU versus multi-DPU | Scaling and host collective overhead. |
| Int8 versus alternate data format | Accuracy/performance tradeoff. |
| Double buffering off versus on | DMA/compute overlap benefit if implemented. |

## Reporting Rules

- Always report hardware and SDK details.
- Always report route configuration and ablation flags.
- Always include correctness metrics next to runtime metrics.
- Do not compare a quantized UPMEM route against a CPU FP64 baseline without
  reporting the induced error.
- Treat negative results as useful results when the workload and measurement are
  controlled.
