# Architecture

## Commitment

The second-stage project is a TaskGraph-centered modular PIM tensor-network
runtime.

It is not:

- a UPMEM hardware simulator;
- a UPMEM tensor-network simulator written as one monolithic DPU program;
- a SimplePIM-only project;
- a SparseP-only project;
- a PID-Comm-only project.

The host CPU owns planning, slicing, routing, validation, profiling, and all
global coordination. DPUs execute bounded numeric or data-movement tasks selected
by the dispatcher. TaskGraphV2 is the central artifact that makes those decisions
inspectable and reproducible.

## Clean Architecture Layers

```text
02_Modular_UPMEM_TN_Simulator/

Domain Core
+-- CircuitIR
+-- TensorNetworkIR
+-- TensorMetadata
+-- TaskGraphV2
+-- TaskNode
+-- DataFormat
+-- CostRecord
+-- ProfileRecord
+-- ValidationRecord

Application Layer
+-- TensorNetworkBuilder
+-- ContractionPlanner
+-- Slicer
+-- CostOracle
+-- Dispatcher
+-- ExperimentRunner
+-- Validator

Ports / Interfaces
+-- PlannerEnginePort
+-- ExecutionRoutePort
+-- CollectiveProviderPort
+-- DataFormatProviderPort
+-- ProfilerPort
+-- BaselineBackendPort

Infrastructure Adapters
+-- CPUBackend
|   +-- NumPy / opt_einsum reference
|   +-- QuEST / PIMutation-style state-vector baseline
|
+-- GPUBackend_optional
|   +-- cuTensorNet / CuPy / future dense baseline
|
+-- UPMEMBackend
    +-- SimplePIMProvider_default
    +-- RawUPMEMProvider_baseline
    +-- CustomDenseProvider
    +-- SparsePProvider
    +-- HeuristicProvider
    +-- NaiveHostCollectiveProvider
    +-- PIDCommCollectiveProvider
```

Dependency rule:

```text
Domain Core knows nothing about SimplePIM, SparseP, PID-Comm, UPMEM SDK, NumPy,
QuEST, opt_einsum, cuTensorNet, or CuPy.

Application Layer depends only on Domain Core and abstract ports.

Infrastructure Adapters depend on external libraries and hardware APIs.
```

This rule is important because the thesis claim is about route-aware execution,
not about one specific external framework.

## Main Data Flow

```text
Circuit / QASM / generated benchmark
        |
        v
TensorNetworkBuilder
        |
        v
TensorNetworkIR
        |
        v
ContractionPlanner
  - opt_einsum initially
  - cotengra or route-aware planner later if justified
        |
        v
Slicer
  - WRAM-aware
  - DPU-count-aware
  - reduction-aware
        |
        v
TaskGraphV2
        |
        v
Dispatcher + CostOracle
        |
        v
Selected execution route per TaskNode
        |
        +--> CPU route
        +--> optional GPU route
        +--> UPMEM SimplePIM route
        +--> UPMEM raw/custom dense route
        +--> UPMEM SparseP route
        +--> heuristic route
        +--> collective route
        |
        v
Validation + profiling + experiment record
```

## Source Layout Rule

The implementation follows the same pipeline boundaries as the architecture.
Each step has a folder so multiple implementations can coexist without turning
one file into a framework dump:

```text
src/tnsim/
+-- core/        # shared dataclasses, file IO, small utilities
+-- config/      # YAML config loaders and future config schemas
+-- circuits/    # built-in workloads and QASM parsers
+-- network/     # circuit-to-tensor-network builders
+-- task_graph/  # TaskGraphV2 planners/lowerers
+-- dispatch/    # route eligibility and selection policies
+-- execution/   # CPU TN, QuEST exact, and future GPU/PIM executors
+-- validation/  # reference outputs and correctness metrics
+-- records/     # execution logs and per-run records
+-- results/     # metric extraction, summaries, charts
+-- suite/       # multi-run orchestration
+-- runner.py    # single-run orchestrator
```

Adding a new implementation should usually mean adding a file inside the
corresponding step folder, for example `execution/gpu_cupy.py`,
`task_graph/cotengra_v1.py`, or `results/energy_rapl.py`. The orchestrators
should remain small.

## Domain Core

Domain objects must be serializable, deterministic, and free of hardware calls.

| Object | Purpose |
| --- | --- |
| `CircuitIR` | Input circuit or generated benchmark in a normalized form. |
| `TensorNetworkIR` | Tensors, indices, labels, and contraction intent before execution. |
| `TensorMetadata` | Shape, labels, logical dtype, structure, density, storage, lifetime. |
| `TaskGraphV2` | Ordered dependency graph of operations and data movement decisions. |
| `TaskNode` | One contraction, transformation, reduction, collective, or host-only action. |
| `DataFormat` | Logical and physical tensor representation, including scale metadata. |
| `CostRecord` | Estimated transfer, compute, conversion, reduction, and error cost. |
| `ProfileRecord` | Measured timing, bytes, status, and counters for a task. |
| `ValidationRecord` | Error metrics against a CPU reference or accepted baseline. |

TaskGraphV2 must remain expressive enough to represent the current MVP dense path
and future sparse, collective, heuristic, multi-DPU, and route-aware-planning
cases.

## Application Components

### TensorNetworkBuilder

Responsibilities:

- parse QASM or generated benchmark definitions;
- construct tensors with labels and structure hints;
- preserve deterministic seeds and benchmark metadata;
- emit `TensorNetworkIR`.

It should not choose UPMEM routes.

### ContractionPlanner

Responsibilities:

- select a pairwise contraction path on the host;
- initially use `opt_einsum`;
- expose candidate paths later for route-aware planning;
- record planner name, version/config, seed, estimated FLOPs, and peak tensor size.

It should not execute kernels or call UPMEM.

### Slicer

Responsibilities:

- split contractions so candidate route tasks respect WRAM, MRAM, and host memory
  limits;
- create explicit slice and reduction tasks;
- record why slicing was required;
- fail early when no legal tile shape exists.

### CostOracle

Responsibilities:

- estimate transfer, computation, preparation, conversion, reduction, and
  numerical-error costs;
- start with rules and static estimates;
- later use empirical profiles from previous runs;
- expose enough detail for the dispatcher to explain route choices.

### Dispatcher

Responsibilities:

- enumerate candidate routes;
- ask each route whether it can execute a task;
- request estimates for legal route/format pairs;
- select one route under ablation/configuration constraints;
- log selected and rejected routes with concrete reasons;
- never hide fallback decisions.

Initial dispatch is rule-based. Mature dispatch minimizes:

```text
transfer_cost
+ compute_cost
+ preparation_cost
+ conversion_cost
+ reduction_cost
+ numerical_error_penalty
+ fallback_penalty
```

### ExperimentRunner

Responsibilities:

- execute a fixed benchmark definition;
- persist task graph, route decisions, profiles, validation, environment, and
  ablation flags;
- keep raw records sufficient to regenerate thesis plots.

### Validator

Responsibilities:

- compare route outputs against CPU reference outputs;
- compute max absolute error, max relative error, norm drift, fidelity when
  applicable, and observable error when full state comparison is too expensive;
- attach validation records to task and run outputs.

## Route Interface

Every execution route implements the same contract:

```text
can_execute(task, tensor_metadata, hardware_state) -> RouteEligibility
estimate(task, tensor_metadata, data_format, hardware_state) -> CostRecord
prepare(task, selected_format, runtime_context) -> PreparedTask + ProfileRecord
execute(prepared_task, runtime_context) -> TensorHandle + ProfileRecord
```

`prepare` is mandatory. It separates layout conversion, quantization, sparse
format conversion, tile allocation, DPU selection, and buffer planning from kernel
execution. Without this boundary, the thesis cannot fairly compare SimplePIM,
raw UPMEM, SparseP, and CPU fallback routes.

The route interface must return explicit failures:

```text
unsupported_op_kind
unsupported_structure
unsupported_data_format
tile_exceeds_wram
requires_unavailable_dependency
requires_unavailable_hardware
estimated_error_too_high
conversion_cost_not_justified
disabled_by_ablation
```

## Backend Decisions

### CPUBackend

Role:

- correctness reference;
- host pathfinding;
- fallback execution;
- small contractions where DPU transfer is unjustified;
- final reductions if collective routes are not beneficial;
- state-vector comparison through QuEST/PIMutation-style baseline.

CPU is not just a baseline. It is the control path and the source of reference
truth.

### GPUBackend Optional

Role:

- optional high-performance dense baseline;
- useful for final comparison if available cheaply.

Decision: postpone until the UPMEM architecture, profiling, and CPU references
are stable.

### UPMEMBackend

The UPMEM backend is internally modular.

```text
UPMEMBackend
+-- SimplePIMProvider_default
+-- RawUPMEMProvider_baseline
+-- CustomDenseProvider
+-- SparsePProvider
+-- HeuristicProvider
+-- NaiveHostCollectiveProvider
+-- PIDCommCollectiveProvider
```

#### SimplePIMProvider_default

Decision:

```text
SimplePIM is the default UPMEM programming substrate.
```

Use it first for:

- elementwise kernels;
- diagonal apply;
- map/reduce style operations;
- layout transforms;
- simple dense kernels if expressible;
- new UPMEM kernels unless evidence says raw/custom is better.

Constraint:

```text
SimplePIM is a provider, not the architecture.
```

If SimplePIM is slower or less transparent for a hot dense path, the dispatcher
can select raw/custom dense kernels while still preserving SimplePIM as the
default for suitable tasks.

#### RawUPMEMProvider_baseline

Decision:

```text
Keep 01_MVP_DenseGEMM as a frozen baseline and wrap it as a V2 route.
```

Use it for:

- reproducibility;
- proof that V2 can replay current results;
- performance control;
- escape hatch when SimplePIM hides too much;
- hot kernels if measured faster.

#### CustomDenseProvider

Decision:

```text
Build only after TaskGraphV2 can compare SimplePIM and raw UPMEM fairly.
```

Use it for:

- dense GEMM;
- K-tiling;
- multi-DPU tiling;
- double buffering when supported;
- fixed-point and block-floating-point experiments.

Dense contraction is the trunk numerical route, but it must be optimized after
the measurement framework exists.

#### SparsePProvider

Decision:

```text
SparseP is a conditional sparse-linear-algebra route, not a general
tensor-network backend.
```

Use it when:

- task structure is sparse, diagonal-as-sparse, or graph-local sparse;
- measured or estimated density is below a route-specific threshold;
- conversion cost is included;
- the next operation can consume sparse output or densification is cheap.

Do not use it when:

- the intermediate tensor is dense;
- CSR/COO conversion dominates;
- the next task immediately requires dense GEMM;
- a direct diagonal/permutation/heuristic kernel is simpler.

Main experiment:

```text
Find the density threshold where SparseP beats dense SimplePIM/raw UPMEM after
conversion cost is included.
```

#### HeuristicProvider

Decision:

```text
Add heuristic operations before SparseP.
```

Use it for:

- diagonal gates;
- permutation gates;
- row swaps;
- trivial contractions;
- scalar contractions;
- identity elimination;
- reshape-only operations;
- gate fusion when it reduces transfers without making tensors too large.

The initial implementation can be host-only. Later it can call SimplePIM if the
data is already resident on DPUs and transfer cost is favorable.

#### Collective Providers

Decision:

```text
Collectives are explicit providers, not hidden inside dense kernels.
```

Always implement first:

```text
NaiveHostCollectiveProvider
```

Then compare with:

```text
PIDCommCollectiveProvider
```

Use collectives for:

- broadcast;
- gather;
- scatter;
- reduce;
- all-reduce-like sliced result aggregation.

Do not use collectives for:

- ordinary pairwise contraction;
- dynamic DPU-to-DPU dependency;
- small reductions where naive CPU aggregation is faster.

## Data Formats

Data format is a first-class module.

```text
DataFormatProvider
+-- complex_f64_host
+-- complex_i8_tile_scaled
+-- fixed_point
+-- block_floating_point
```

Rule:

```text
No DPU route may hide its numerical format.
Every non-reference format must report error against CPU reference.
```

Initial path:

| Format | Role |
| --- | --- |
| `complex_f64_host` | CPU reference and validation. |
| `complex_i8_tile_scaled` | Current MVP compatibility. |
| `fixed_point` | First serious integer-native alternative. |
| `block_floating_point` | Later dynamic-range experiment. |

## Initial Dispatcher Rules

The dispatcher starts with rules before a learned or empirical cost model.

```text
if op is pathfinding, slicing, symbolic analysis, or benchmark setup:
    CPUBackend

elif op is diagonal, permutation, reshape-only, scalar, identity, or trivial:
    HeuristicProvider

elif op is dense contraction:
    SimplePIMProvider_default
    compare with RawUPMEMProvider_baseline when ablation is enabled
    fall back to CPU/GPU if precision, WRAM, or transfer estimate is bad

elif op is sparse and conversion is justified:
    SparsePProvider

elif op is collective:
    NaiveHostCollectiveProvider first
    PIDCommCollectiveProvider when size/repetition justifies it

else:
    CPUBackend fallback
```

Mature dispatch must select routes using measured and estimated cost records,
not only static rules.

## Core Invariants

1. TaskGraphV2 is the only execution contract between planning and runtime.
2. DPUs never search contraction paths.
3. DPUs never request data from other DPUs.
4. Every DPU task declares WRAM, MRAM, transfer, and tasklet requirements before
   execution.
5. Every route can be disabled for ablation.
6. Every route decision records selected and rejected routes.
7. Every non-reference data format reports an error metric against host reference.
8. Host orchestration is measured separately from DPU execution.
9. Preparation cost is measured separately from kernel execution.
10. The current dense-GEMM MVP remains a baseline, not hidden history.

## Expected Evolution From The MVP

| MVP behavior | Next-stage behavior |
| --- | --- |
| One direct dense int8 GEMM route. | Multiple explicit providers selected by dispatcher. |
| Ad hoc task graph. | TaskGraphV2 records routing, format, slicing, cost, and profiles. |
| Host directly lowers to tiled GEMM. | Planner emits TaskNodes; dispatcher selects route. |
| K-tiling rejected. | K-tiling is represented, implemented, and measured. |
| One DPU allocation path. | Runtime can allocate multiple DPUs/ranks with explicit collective tasks. |
| Quantization hidden in dense flow. | DataFormat records expose scale, accumulator, and error. |
| Timing is correctness-first. | Per-task profile records support thesis tables and ablations. |

## Acceptance Gate For Architecture Implementation

The architecture is implemented enough to move beyond scaffolding only when:

- the current MVP dense result can be reproduced through TaskGraphV2;
- each TaskNode has selected and rejected route records;
- CPU reference, raw UPMEM baseline, and SimplePIM provider can be compared on at
  least one equivalent task family;
- `execution_log.json` contains cost, profile, format, and validation records;
- a new developer can disable any provider and understand the resulting fallback
  or failure from the log alone.
