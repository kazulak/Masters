# Architecture

The implementation is a modular monolith with a functional core and a small
stateful shell. Its purpose is one end-to-end experiment, not a general plugin
platform.

```text
Circuit + query
  -> Tensor network + inputs
  -> Planner
  -> ContractionDAG
  -> CPU executor ---------------------> result
  -> UPMEM mapper -> UpmemPlan -> runtime -> result
  -> validation -> evidence -> report
```

## Boundaries

| Boundary | Input | Output | Owns |
|---|---|---|---|
| Circuit lowering | circuit and query | tensor descriptors and arrays | gate semantics and output ordering |
| Planning | tensor descriptors | path, slicing, provenance | contraction order only |
| Semantic graph | descriptors and path | `ContractionDAG` | contractions, reductions, dependencies, logical identity |
| Numerics | arrays and numeric mode | encoded arrays and scale metadata | packing, rounding, saturation, decode, error utilities |
| Parallel mapping | DAG, topology, numeric mode | `UpmemPlan` | tiling, work units, assignment, residency, transfers, reductions |
| CPU execution | DAG and arrays | output and timing | same-DAG NumPy replay |
| UPMEM runtime | `UpmemPlan` and arrays | output, timing, backend facts | sessions, transfers, launches, collection, no-fallback checks |
| Experiment | cases and route dimensions | one raw row per repetition | warmups, repetitions, references, validation |
| Evidence | raw execution facts | normalized rows and claim decisions | hashes, compatibility, provenance |
| Reporting | normalized rows | tables and plots | aggregation and presentation only |

Each boundary is a module with plain functions. A structured immutable type is
used only when it crosses boundaries and owns validation or identity semantics.
Local return values remain local variables, tuples, or mappings.

## Canonical Graph

`ContractionDAG` is the only active contraction intermediate representation.
It contains:

- input tensor descriptors;
- binary contraction nodes;
- explicit sliced-result reductions;
- tensor views and fixed semantic slices;
- dependencies and final output view.

It does not contain arrays, planner timing, target estimates, DPU placement,
numeric scales, kernel IDs, or machine paths.

The DAG is the lowering of a selected contraction path into executable
mathematical dependencies. It is not the lowering of a contraction into GEMM.
GEMM-like canonicalization, tiles, and kernels are UPMEM mapping decisions.

Slicing changes the mathematical graph and therefore changes its hash. Tiling
only divides one graph node for a target and therefore changes the physical plan
without changing the DAG hash.

## Planning

Planning consumes labels and shapes, not tensor values or hardware sessions.
The active adapters are opt_einsum, cotengra, and the experimental PIM-aware
greedy heuristic. Planner provenance is separate from DAG identity: different
planners can select the same graph.

The PIM-aware score is an uncalibrated candidate estimator until physical
measurements establish prediction quality. It is not described as globally
optimal or as measured hardware performance.

## UPMEM Mapping

The mapper converts one DAG into one target-specific physical plan. The plan
records:

- numeric representation;
- tasklet, output-tile, and slice work units;
- DPU and rank assignments;
- memory and intermediate residency decisions;
- transfer and host-reduction steps;
- kernel and native protocol identity.

Machine-local rank paths, binary paths, working directories, SDK installation,
and timeouts are runtime configuration. They do not affect DAG or physical-plan
identity.

Parallelism is hierarchical:

1. tasklets divide local output work inside one DPU;
2. DPUs divide output tiles or independent slices;
3. independent DAG nodes may run concurrently only after the first two levels
   are correct and measurable.

The minimum final parallel claim requires measured tasklet and multi-DPU or
multi-group execution. Allocation alone is not parallel execution.

## Numerics

UPMEM numerical representation is a physical policy, not a planner or DAG
choice. Initial quantization occurs once on the host. A practical integer route
uses packed real/imaginary int8 operands, int32 local accumulation, explicit
scale metadata, and host decode. Resident intermediates remain encoded when the
selected mapping permits it.

The evidence row records encode, transfer, kernel, reduction, download, and
decode time separately. End-to-end speedup includes required conversion work.

## State and Mutation

The functional core returns new values. Mutable state is confined to:

- NumPy buffers owned by one execution call;
- UPMEM sessions and native subprocesses;
- run-directory and report writers.

Inputs are treated as read-only. Planning and mapping never mutate the circuit,
tensor descriptors, arrays, or DAG. The runtime executes an existing plan and
does not silently replan it.

## Timing and Evidence

Stable timing components are:

```text
planning, mapping, encode, prepare, h2d, kernel, host_reduce,
d2h, decode, validation, session_open, session_close,
steady_state, end_to_end
```

Hashing and validation are outside kernel time. Null is used when a component
cannot be measured honestly. CPU and UPMEM comparisons declare the timing scope
they pair.

Physical execution fails closed. A physical row is admitted only when observed
native identity, allocation, kernel execution, release, and validation agree.
Simulator rows cannot support physical speedup. Unsupported and failed rows are
retained.

## Dependency Rules

- the semantic graph imports no executor, filesystem, subprocess, or UPMEM ID;
- planning imports no native runtime;
- mapping opens no hardware session;
- the runtime performs no path search;
- reporting never executes a benchmark;
- baselines do not use a provider registry;
- historical milestone code is not imported by the active path.

## Migration Status

The active physical adapter is
`src/quantum_bench/upmem/runtime.py`, backed by the self-contained native tree
at `native/upmem/runtime/`. Historical M5/v4 Python and native modules are not
imported by the active path. The completed T1A-D ownership migration is followed
by these remaining reset steps:

1. collapse milestone commands, configs, reports, and compatibility tests;
2. create the canonical model, circuit, and lowering boundaries;
3. add direct baselines, evidence schemas, and the public verification flow;
4. delete historical active source once parity tests pass.

Progress and temporary adapter expiry are recorded in
[MIGRATION_LEDGER.md](MIGRATION_LEDGER.md). Historical behavior remains at the
baseline tag rather than as permanent active architecture.
