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
| Semantic network | tensor descriptors and requested output | `TensorNetwork` | non-executable metadata, descriptors, connectivity, and output request |
| Logical graph | `TensorNetwork` and selected path | `ContractionDAG` | contractions, reductions, dependencies, and logical identity |
| Numerics | arrays and numeric mode | encoded arrays and scale metadata | packing, rounding, saturation, decode, error utilities |
| Parallel mapping | DAG, topology, numeric mode | `UpmemPlan` | bounded output-tile assignment and host-roundtrip transfer accounting |
| CPU execution | DAG and arrays | output and timing | same-DAG NumPy replay |
| UPMEM runtime | `UpmemPlan` and arrays | output, timing, backend facts | sessions, transfers, launches, collection, no-fallback checks |
| Experiment | cases and route dimensions | one raw row per repetition | warmups, repetitions, references, validation |
| Evidence | raw execution facts | normalized rows and claim decisions | hashes, compatibility, provenance |
| Reporting | normalized rows | tables and plots | aggregation and presentation only |

Each boundary is a module with plain functions. A structured immutable type is
used only when it crosses boundaries and owns validation or identity semantics.
Local return values remain local variables, tuples, or mappings.

## Canonical Graph

`TensorNetwork` is non-executable semantic metadata. It contains tensor
descriptors, connectivity, and the requested output, but no contraction order,
slicing, dependencies, target estimates, executor data, arrays, or timing.

`ContractionDAG` is the sole logical execution intermediate representation. It
contains:

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
The canonical adapters are opt_einsum and cotengra. The root functions in
`src/quantum_bench/planning.py` return a validated binary active-list path and
a JSON-compatible provenance mapping. Planner provenance is separate from DAG
identity: different planners can select the same graph.

The PIM-aware projected-prefix greedy heuristic is historical and exploratory,
not a canonical adapter. It remains available only for old configurations,
tests, and evidence until T12. It is an uncalibrated candidate estimator and
cannot be described as globally optimal or as measured hardware performance.
Hardware-calibrated
target-aware planning is separate future work.

T3-0 froze this boundary and T3 implemented it. The active route has no
generic planner dispatcher; the M5 coordinator selects exactly opt_einsum or
cotengra through a private configuration helper. The projected-prefix planner
remains historical.

## UPMEM Mapping

The mapper converts one DAG into one target-specific physical plan. The current
bounded v4 mapping records:

- numeric representation;
- output-tile work units and their assignment;
- requested topology, including DPU and rank assignments;
- host-roundtrip transfer accounting;
- kernel policy.

Tasklet scheduling, slice-stage scheduling, and intermediate residency are
planned extensions. They are not implemented or claimable by the current v4
mapping.
The current bounded mapper does not make memory and intermediate residency
decisions; those remain planned extensions.

The only public physical-plan records are `UpmemTopology`, `UpmemWorkUnit`,
`UpmemStage`, and `UpmemPlan`; `UpmemResources` is runtime configuration. Their
exact frozen field contracts are defined in `docs/reset_contract.md`.

Machine-local rank paths, binary paths, working directories, SDK installation,
timeouts, ABI identifiers, and executable hashes are runtime or executable
provenance. They are not additional `UpmemPlan` fields and do not affect DAG or
physical-plan identity.

The target architecture is hierarchical, but only bounded output-tile mapping
is currently implemented:

1. tasklets may divide local output work inside one DPU;
2. DPUs divide output tiles;
3. slice groups and independent DAG nodes may run concurrently after their
   scheduling stages are implemented and measured.

Tasklet scheduling, slice-stage scheduling, and DPU-resident intermediate
execution are planned work. The current mapping does not support claims for
those mechanisms. Allocation alone is not parallel execution.

## Numerics

UPMEM numerical representation is a physical policy, not a planner or DAG
choice. Initial quantization occurs once on the host. A practical integer route
uses packed real/imaginary int8 operands, int32 local accumulation, explicit
scale metadata, and host decode. Intermediate residency is a planned policy and
is not implemented by the current host-roundtrip mapping.

The evidence row records encode, transfer, kernel, reduction, download, and
decode time separately. End-to-end speedup includes required conversion work.

The reset implementation order is dependency-driven: pure split-complex
numerics are implemented before the CPU single-run result contract because the
CPU boundary consumes the final `NumericPolicy` type. This is an implementation
ordering correction only; it does not change the final architecture or the
ownership of the numeric, CPU, UPMEM, experiment, or evidence boundaries.

```text
T6A pure numerics
  -> T4A results and CPU single-run API
  -> T4C implement final UpmemStage/UpmemPlan schema
  -> T6B physical-plan CPU replay
  -> T7 four real-product ABI execution
  -> T4B1 UPMEM session API
  -> T4B2 wrapper removal
  -> T5 evidence and experiment lifecycle
  -> T8+ unchanged later work
```

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
planning, mapping, encode, preparation, h2d, kernel, host_reduce,
d2h, decode, validation, session_open,
steady_state, end_to_end
```

Hashing and validation are outside kernel time. Null is used when a component
cannot be measured honestly. CPU and UPMEM comparisons declare the timing scope
they pair. Session close exists only in session evidence; it is not a
per-sample measurement or part of either total scope.

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
imported by the active path. The completed T1A-D ownership migration is
followed by the corrected task order above. T4-0 is contract-frozen and
implementation-pending. The current WP4, WP5, and WP6 completion labels refer
only to the old bounded base ownership and runtime behavior; they do not
certify the final reset numeric, session, stage, or evidence contracts. The
next implementation work is therefore T6A, followed by T4A, T4C, T6B, T7,
T4B1, T4B2, and T5. Configuration, reporting, cleanup, software qualification,
and later ETH qualification remain subsequent work.

Progress and temporary adapter expiry are recorded in
[MIGRATION_LEDGER.md](MIGRATION_LEDGER.md). Historical behavior remains at the
baseline tag rather than as permanent active architecture.
