# Modular UPMEM Tensor-Network Simulator

Status: planning scaffold only. No implementation is added here yet.

This directory defines the planned second-stage architecture after
`../01_MVP_DenseGEMM`. The goal is a TaskGraph-centered modular tensor-network
runtime for quantum circuit simulation. The host builds and slices the tensor
network, the dispatcher selects execution routes, and the UPMEM backend is
SimplePIM-first while still allowing raw/custom dense kernels, sparse kernels,
heuristic transformations, and collective communication providers.

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
   SimplePIM is the default UPMEM programming substrate, but the route must still
   expose tiling, buffering, transfer, and preparation costs.
5. Treat data format as a first-class execution choice.
   Standard FP32/FP64 on DPU is not the default route. Int8, fixed-point,
   block-floating-point, and library-backed formats must be compared explicitly.
6. Never require dynamic DPU-to-DPU communication.
   Reductions, reshuffles, and aggregation are explicit collective tasks.

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
|   +-- thesis_structure.md
+-- planner/       # Future host pathfinding, slicing, and task-graph generation
+-- dispatcher/    # Future route selection and ablation controls
+-- runtime/       # Future UPMEM allocation, DMA, rank mapping, profiling
+-- kernels/       # Future DPU kernels grouped by execution route
+-- data_formats/  # Future quantization and block-format experiments
+-- benchmarks/    # Future benchmark suites and run definitions
+-- validation/    # Future correctness and regression tests
```

## Reading Order

1. `docs/thesis_structure.md`: research claim, contribution structure, and thesis
   chapter plan.
2. `docs/architecture.md`: committed architecture, layers, ports, providers, and
   data flow.
3. `docs/task_graph_v2.md`: planned host-to-runtime intermediate representation.
4. `docs/implementation_plan.md`: milestones, sequencing, and done criteria.
5. `docs/literature_guidelines_mapping.md`: how each review recommendation maps to
   an engineering choice.
6. `docs/design_decisions.md`: decisions that should remain stable unless evidence
   changes them.
7. `docs/experiment_plan.md`: ablation and evaluation plan.
8. `docs/open_questions.md`: resolved and remaining ambiguities.

## Non-Goals For This Planning Pass

- No new DPU kernels.
- No refactor of `01_MVP_DenseGEMM`.
- No claim that SimplePIM, SparseP, PID-Comm, or any future provider is already
  integrated into the V2 runtime.
- No replacement of the CPU baseline in `PIMutation_reconstruction`.
- No conversion of providers into network microservices. These are plugin-style
  modules inside one host-orchestrated runtime.

## Success Criteria For The Next Implementation Stage

The next implementation should be considered useful only when it can:

1. Reproduce the current dense-GEMM MVP result through the new task graph.
2. Explain every route decision in a machine-readable execution log.
3. Disable each route independently for ablation.
4. Report correctness, total wall time, host planning time, DMA time, DPU time,
   host reduction time, memory footprint, and quantization error.
5. Compare against the QuEST/PIMutation CPU state-vector baseline and the
   `01_MVP_DenseGEMM` UPMEM baseline.
6. Produce `execution_log.json` records that explain selected and rejected routes
   well enough for another developer to reproduce the decision.
