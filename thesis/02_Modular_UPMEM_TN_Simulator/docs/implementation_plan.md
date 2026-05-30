# Implementation Plan

This plan intentionally starts with structure and observability before adding new
kernels. The goal is a research platform with credible ablations, not a pile of
uncontrolled optimizations.

## Phase 0: Freeze And Document Baselines

Goal: make the existing baselines reproducible.

Deliverables:

- Keep `../01_MVP_DenseGEMM` as the dense int8 UPMEM MVP.
- Keep `../PIMutation_reconstruction/QuEST_implementation` as the CPU
  state-vector baseline.
- Record compiler, UPMEM SDK, CPU, DPU/rank count, and dataset details for every
  experiment.

Done criteria:

- Bell and GHZ pass through the MVP.
- QuEST/PIMutation benchmark commands are documented for the target machine.
- Baseline metrics have wall time and, when available, energy.

## Phase 1: Task Graph V2 And Host Reference Route

Goal: introduce the new IR without changing numerical behavior.

Deliverables:

- V2 task graph emitter.
- V2 task graph parser.
- Host reference route for every V2 operation.
- Compatibility mode that represents the current MVP dense path.
- Execution log with route decisions.

Done criteria:

- V2 host reference matches NumPy/QuEST for small circuits.
- V2 dense compatibility mode reproduces MVP outputs within current int8 tolerance.
- Logs explain route selection even when only one route is enabled.

## Phase 2: Dispatcher Skeleton And Ablation Controls

Goal: make route selection explicit and experimentally controllable.

Deliverables:

- `can_execute`, `estimate`, and `execute` route interface.
- Route registry.
- CLI/config switches to enable or disable routes.
- Route-decision JSON output.

Done criteria:

- Dense-only, host-only, and forced-route modes can be run from the same task graph.
- Invalid forced routes fail with a clear reason.

## Phase 3: Heuristic Operation Route

Goal: avoid dense GEMM for operations with cheap algebraic behavior.

Initial target operations:

- row swaps/permutations;
- diagonal gates;
- identity/no-op tensors;
- gate merging where it reduces transfers and does not increase tensor order
  excessively.

Deliverables:

- Operation classifier in the planner or dispatcher.
- Host implementation first, then optional DPU-assisted implementation only where
  transfer cost makes sense.
- Correctness comparison against dense route.

Done criteria:

- Heuristic route can be enabled and disabled.
- The execution log states which operations bypassed GEMM and why.
- At least one benchmark shows measured benefit or a documented negative result.

## Phase 4: Dense GEMM Route V2

Goal: turn the MVP dense path into a robust dense route.

Deliverables:

- K-tiling support.
- Multi-DPU tile distribution.
- Host-mediated accumulation of K-slices and output slices.
- Manual WRAM budget checks for all tile shapes.
- Optional double-buffering design if the UPMEM SDK path supports it cleanly.
- Tile/tasklet parameter sweep inspired by ATiM-style dense autotuning.

Done criteria:

- No task aborts only because `k > TILE_K`.
- Dense route reports DMA and DPU timing per slice.
- Tile variants can be benchmarked under a fixed task graph.
- The old MVP tile shape remains available as a baseline configuration.

## Phase 5: Data Format Experiments

Goal: quantify the cost and accuracy of integer-native representations.

Deliverables:

- Common format interface for quantize, transfer, accumulate, dequantize, and
  error reporting.
- Current int8 tile scaling as the first format.
- Fixed-point or block-floating-point candidate.
- Accuracy metrics beyond a single tolerance: max absolute error, relative error,
  norm drift, and state fidelity when applicable.

Done criteria:

- Format can be selected independently from route where legal.
- Thesis tables can show performance/error tradeoffs.
- Unsupported route-format pairs fail before execution.

## Phase 6: Sparse Route

Goal: evaluate whether sparse tensor execution helps sparse workloads.

Deliverables:

- Density estimator and sparse eligibility rule.
- Sparse storage format chosen explicitly.
- Host conversion-cost accounting.
- Sparse route implementation or controlled prototype.

Done criteria:

- Sparse route wins on at least one sparse workload or is documented as a negative
  result with conversion overhead.
- Dense fallback remains available for all sparse-eligible operations.

## Phase 7: Host-Mediated Collectives

Goal: handle sliced and multi-DPU results without pretending DPUs communicate.

Deliverables:

- Host reduction tasks in task graph v2.
- Rank/DPU allocation policy.
- Reduction timing and memory accounting.
- Optional PID-Comm-inspired scheduling strategy if it can be implemented without
  changing UPMEM's communication model.

Done criteria:

- Multi-DPU dense route produces identical results to single-DPU dense route within
  selected format tolerance.
- Reduction cost is visible in experiment output.

## Phase 8: Thesis Evaluation Harness

Goal: produce reproducible final plots and tables.

Deliverables:

- Benchmark definitions.
- Run scripts with fixed seeds.
- Result schema.
- Plotting notebooks or scripts.
- Ablation matrix.

Done criteria:

- Every figure can be regenerated from checked-in scripts and recorded raw data.
- Every architectural claim has a matching measurement or a clearly labeled
  limitation.

## Recommended Near-Term Order

1. Implement task graph v2 in parallel with the MVP graph, not by modifying the MVP
   in place.
2. Add a host reference executor for V2.
3. Port the current dense route behind the route interface.
4. Add route logging and ablation switches.
5. Only then add heuristic routing, K-tiling, multi-DPU execution, and format
   experiments.
