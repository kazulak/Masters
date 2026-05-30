# Modular UPMEM Tensor-Network Simulator

Status: planning scaffold only. No implementation is added here yet.

This directory defines the planned second-stage architecture after
`../01_MVP_DenseGEMM`. The goal is a modular UPMEM-backed tensor-network
contraction system for quantum circuit simulation, with explicit routing between
heuristic, dense, sparse, prototype, and host-collective execution paths.

The current MVP proves correctness for a direct dense-GEMM path. This stage turns
that proof into a research platform that can answer: which architectural segments
actually improve correctness, scaling, runtime, DMA cost, energy, and memory use?

## Design Principles From the Literature Review

1. Avoid a monolithic execution engine.
   Use a modular dispatcher with separable execution routes.
2. Do not brute-force every operation through GEMM.
   Detect row swaps, diagonal gates, permutation-like updates, gate fusions, and
   other memory-cheap operations before selecting dense tensor contraction.
3. Keep pathfinding and slicing on the host CPU.
   DPUs execute precisely scoped numeric kernels; they do not search contraction
   paths or dimension trees.
4. Optimize for MRAM-WRAM locality.
   Productivity layers are allowed only when the generated route still exposes
   tiling, buffering, and transfer behavior clearly.
5. Treat data format as a first-class execution choice.
   Standard FP32/FP64 on DPU is not the default route. Int8, fixed-point,
   block-floating-point, and library-backed formats must be compared explicitly.
6. Never require dynamic DPU-to-DPU communication.
   Reductions, reshuffles, and aggregation are host-mediated.

## Directory Layout

```text
02_Modular_UPMEM_TN_Simulator/
+-- README.md
+-- docs/
|   +-- architecture.md
|   +-- design_decisions.md
|   +-- experiment_plan.md
|   +-- implementation_plan.md
|   +-- literature_guidelines_mapping.md
|   +-- open_questions.md
|   +-- task_graph_v2.md
+-- planner/       # Future host pathfinding, slicing, and task-graph generation
+-- dispatcher/    # Future route selection and ablation controls
+-- runtime/       # Future UPMEM allocation, DMA, rank mapping, profiling
+-- kernels/       # Future DPU kernels grouped by execution route
+-- data_formats/  # Future quantization and block-format experiments
+-- benchmarks/    # Future benchmark suites and run definitions
+-- validation/    # Future correctness and regression tests
```

## Reading Order

1. `docs/literature_guidelines_mapping.md`: how each review recommendation maps to
   an engineering choice.
2. `docs/architecture.md`: proposed components and data flow.
3. `docs/task_graph_v2.md`: planned host-to-runtime intermediate representation.
4. `docs/implementation_plan.md`: milestones and done criteria.
5. `docs/design_decisions.md`: decisions that should remain stable unless evidence
   changes them.
6. `docs/experiment_plan.md`: ablation and evaluation plan.
7. `docs/open_questions.md`: ambiguous points that need clarification.

## Non-Goals For This Planning Pass

- No new DPU kernels.
- No refactor of `01_MVP_DenseGEMM`.
- No claim that SimplePIM, ATiM, SparseP, TransPimLib, PRISM, or PID-Comm are
  already integrated.
- No replacement of the CPU baseline in `PIMutation_reconstruction`.

## Success Criteria For The Next Implementation Stage

The next implementation should be considered useful only when it can:

1. Reproduce the current dense-GEMM MVP result through the new task graph.
2. Explain every route decision in a machine-readable execution log.
3. Disable each route independently for ablation.
4. Report correctness, total wall time, host planning time, DMA time, DPU time,
   host reduction time, memory footprint, and quantization error.
5. Compare against the QuEST/PIMutation CPU state-vector baseline and the
   `01_MVP_DenseGEMM` UPMEM baseline.
