# Thesis Structure

This document is the research-level plan for the next stage. It describes what
the thesis is actually claiming, what the implementation must prove, and how the
written thesis should be structured.

## Working Thesis Claim

The project is a TaskGraph-centered modular tensor-network runtime for quantum
circuit simulation on UPMEM. The scientific contribution is not a new UPMEM
simulator, not a SimplePIM wrapper, and not a SparseP integration. The
contribution is a planner, task graph, dispatcher, cost model, validation path,
and ablation framework that decide when CPU, optional GPU, and UPMEM execution
routes are legal and worthwhile.

One-sentence version:

```text
Build a TaskGraph-centered tensor-network runtime where the host owns planning,
slicing, routing, validation, and reductions; CPU/GPU/UPMEM are replaceable
backends; and the UPMEM backend is SimplePIM-first but can dispatch individual
tasks to raw/custom dense kernels, SparseP sparse kernels, heuristic operations,
or PID-Comm-style collectives, with every decision logged and measured.
```

## Explicit Scope

In scope:

- tensor-network simulation of quantum circuits;
- host-side circuit-to-tensor-network conversion;
- host-side contraction planning and slicing;
- TaskGraphV2 as the execution contract;
- route dispatch across CPU, optional GPU, and UPMEM providers;
- UPMEM execution through SimplePIM by default;
- raw UPMEM dense GEMM as a baseline and escape hatch;
- custom dense kernels only after the framework can compare them fairly;
- sparse execution only when density and conversion cost justify it;
- collective communication as explicit runtime tasks;
- validation against CPU FP64/reference execution.

Out of scope:

- simulating the UPMEM hardware itself in software;
- replacing the whole project with a state-vector simulator;
- treating SimplePIM, SparseP, or PID-Comm as the thesis contribution by itself;
- hiding quantization, format conversion, or host reduction cost inside a kernel;
- building distributed services. Modules are local adapters, not microservices.

## Research Contributions

The thesis should be written around these contributions.

| ID | Contribution | Evidence required |
| --- | --- | --- |
| C1 | TaskGraphV2 IR for tensor-network operations, formats, routing, costs, and profiles. | MVP replay, schema examples, route logs, validation records. |
| C2 | Modular dispatcher that selects legal execution routes under hardware and numerical constraints. | Forced-route tests, rejection reasons, ablation switches. |
| C3 | UPMEM backend design that is SimplePIM-first but not SimplePIM-only. | SimplePIM vs raw UPMEM comparison on equivalent tasks. |
| C4 | Cost and profiling framework separating planning, preparation, DMA, DPU kernel, reduction, and validation. | Per-task timing/byte/error records. |
| C5 | Empirical routing results for dense, heuristic, sparse, collective, and format choices. | Ablation matrix and threshold plots. |
| C6 | Route-aware planning extension where UPMEM costs can change the contraction path. | At least one workload where the selected path differs from pure FLOP-minimizing planning. |

## Research Questions

1. Can a TaskGraph-centered runtime reproduce the current dense UPMEM MVP while
   making route decisions observable?
2. When is SimplePIM competitive with raw UPMEM for tensor-network sub-tasks, and
   when does the raw/custom dense route remain necessary?
3. How much work can be avoided by recognizing diagonal, permutation, scalar,
   reshape, and trivial operations before dense contraction?
4. Which dense tiling and data formats give the best speed/error tradeoff on
   UPMEM?
5. When does SparseP-style sparse execution beat dense execution after conversion
   cost is included?
6. When do PID-Comm-style collectives beat naive host-mediated collectives for
   sliced or multi-DPU outputs?
7. Can route-aware planning choose better tensor-network paths than a pure
   FLOP-minimizing planner under UPMEM constraints?

## Written Thesis Chapter Plan

### Chapter 1: Introduction

State the problem: tensor-network simulation exposes irregular contractions,
large intermediate tensors, and data movement pressure; UPMEM offers high memory
parallelism but has strict WRAM, MRAM, integer arithmetic, and communication
constraints. Present the main claim that route selection and measurement are the
central research problem.

### Chapter 2: Background And Related Work

Cover:

- tensor-network simulation and contraction planning;
- UPMEM programming constraints: host orchestration, MRAM, WRAM, tasklets, DMA,
  integer-native execution, and no dynamic DPU-to-DPU dependency;
- SimplePIM as the default productivity substrate;
- SparseP as evidence for sparse PIM kernels;
- PID-Comm as evidence for collective communication optimization;
- PIMutation/QuEST as a state-vector baseline and heuristic inspiration.

The chapter should end by explaining why a monolithic UPMEM tensor-network
engine is the wrong abstraction.

### Chapter 3: Architecture

Describe:

- clean architecture layers;
- Domain Core and TaskGraphV2;
- planner, slicer, cost oracle, dispatcher, experiment runner, validator;
- ports and adapters;
- CPU, optional GPU, and UPMEM backends;
- UPMEM provider structure;
- route contract with `can_execute`, `estimate`, `prepare`, and `execute`;
- data-format contract;
- validation and profiling records.

This chapter should use the architecture in `docs/architecture.md` as the source
of truth.

### Chapter 4: Implementation Methodology

Follow the staged implementation plan:

1. freeze the MVP;
2. replay the MVP through TaskGraphV2;
3. introduce SimplePIM as the default UPMEM provider;
4. add profiling and empirical cost records;
5. add heuristic operations;
6. strengthen dense UPMEM execution;
7. add collectives and PID-Comm comparison;
8. add SparseP comparison;
9. add route-aware planning.

The chapter should explain why each stage is gated by validation before the next
optimization is introduced.

### Chapter 5: Evaluation

Organize results by research question, not by implementation chronology:

- MVP replay and overhead of generalization;
- SimplePIM versus raw/custom UPMEM;
- dense route scaling and tiling;
- data-format accuracy/performance tradeoffs;
- heuristic route benefits and failures;
- sparse threshold analysis;
- collective communication analysis;
- route-aware planning examples;
- comparison against CPU tensor-network and CPU state-vector baselines.

Every runtime plot must have a matching correctness metric.

### Chapter 6: Discussion

Discuss negative results and boundaries:

- workloads where UPMEM loses because transfer or reduction dominates;
- workloads where sparse conversion is not justified;
- workloads where SimplePIM hides too much for a hot dense kernel;
- numerical limits of int8, fixed-point, and block-floating-point formats;
- portability limits caused by SDK and hardware profile.

### Chapter 7: Conclusion

Summarize what the architecture proves: route-aware, measurable PIM execution is
the defensible path for tensor-network workloads, while provider-specific speedups
are conditional.

## Implementation Handoff Rules

These rules are for junior developers or agents implementing the plan later.

1. Do not edit `../01_MVP_DenseGEMM` while building V2 replay unless explicitly
   tasked to preserve or document the baseline.
2. Any new operation must first be representable in TaskGraphV2.
3. Any new route must implement `can_execute`, `estimate`, `prepare`, and
   `execute`.
4. Any route rejection must be logged with a concrete reason.
5. Any non-reference data format must emit an error record against CPU reference
   data.
6. Any UPMEM route must expose host packing, DMA, DPU kernel, DPU-to-host DMA,
   host unpacking, and host reduction timing.
7. Any claim that a provider is faster must include conversion and preparation
   cost.
8. Any external framework must pass license, build, SDK, and reproducibility
   checks before the thesis calls it integrated.

## Minimum Successful Thesis

The minimum credible thesis does not require every provider to win. It requires:

- TaskGraphV2 replay of the dense MVP;
- CPU reference route;
- raw UPMEM baseline route;
- SimplePIM default route for at least one real operation family;
- measured route decisions with ablation switches;
- one dense-route scaling result;
- one heuristic-route result;
- one data-format accuracy/performance result;
- one sparse or collective study, even if negative;
- a written explanation of where UPMEM helps and where host/data movement costs
  dominate.
