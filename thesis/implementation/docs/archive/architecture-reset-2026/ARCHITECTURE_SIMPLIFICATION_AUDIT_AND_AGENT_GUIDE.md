# Architecture Simplification Audit and Agentic Migration Guide

**Thesis:** *Accelerating Tensor Network Contraction for Quantum Circuit Simulation using Processing-in-Memory Architectures*  
**Repository area:** `thesis/implementation/`  
**Target integration branch:** `refactor/thesis-runtime-simplification`  
**Status:** Proposed normative guide for the architecture-reset branch  
**Audience:** Thesis author, coding agents, reviewers, and hardware experiment operators

---

## 1. Purpose of this document

This document combines two things that must stay connected:

1. an audit of the current implementation and its scientific strengths and structural weaknesses;
2. an executable migration guide for agentic workers performing a strong simplification on a separate branch.

The intended change is not a cosmetic package rearrangement. It is an architecture reset that completes migrations already started in the repository, removes historical execution paths from the active import graph, and aligns the implementation with the actual thesis objective.

The reset must preserve the project's strongest scientific controls while reducing the number of active files, types, abstractions, commands, configurations, and compatibility layers.

The central rule is:

> **Simplify the software architecture without simplifying away the scientific problem.**

The thesis is not about running one isolated tensor contraction on one DPU. It is about end-to-end tensor-network quantum-circuit simulation on UPMEM, with hierarchical parallel execution as the most direct technical contribution. Numerical representation, tiling, layout, placement, residency, transfer scheduling, slicing, and plan selection remain necessary modular mechanisms, but they must not be implemented as a general-purpose research platform with speculative extension points.

---

## 2. Executive architecture decision

The new branch shall implement and document the following thesis story:

```text
A quantum circuit and simulation query are lowered to a validated tensor network.
A planner selects a contraction order and slicing strategy.
The result is represented as one target-neutral ContractionDAG.
A UPMEM mapper chooses numerical representation, tiling, parallel work units,
placement, residency, transfers, and reductions.
CPU and UPMEM executors run comparable logical work.
The quantum result is validated.
Every repetition is recorded with timing, accuracy, plan identity, and provenance.
```

The primary technical contribution is:

> **A hierarchical parallel Host–UPMEM execution model for end-to-end tensor-network quantum-circuit simulation.**

The supporting mechanisms are:

- contraction-path and slicing selection;
- host-side numerical encoding or quantization;
- WRAM-aware tiling and tensor layout;
- work decomposition across tasklets, DPUs, ranks, and independent slices;
- placement and intermediate residency;
- host–DPU transfer scheduling;
- host-mediated reduction;
- optional UPMEM-aware candidate-plan estimation.

These mechanisms are not equal claims and must not each grow into an independent framework. They are explicit policies feeding one physical UPMEM plan.

A UPMEM-aware path or execution-plan cost model is a supported research direction, not a predeclared central algorithmic contribution. Its usefulness must be established empirically after physical measurements exist.

---

## 3. Source-derived scientific framing

The Scoping Literature Review establishes the following basis for the implementation:

- The direct intersection—tensor-network quantum-circuit simulation on UPMEM-style digital PIM—was not represented by an included direct study in the executed search. The implementation therefore addresses a genuine prototype gap.
- The relevant evidence comes from three adjacent domains: quantum simulation on PIM, tensor-network contraction on conventional architectures, and UPMEM proxy workloads.
- Existing CPU/GPU tensor-network methods provide foundations for contraction ordering, slicing, layout, and memory-aware scheduling, but their hardware assumptions do not transfer directly to commodity UPMEM.
- UPMEM introduces first-order constraints: host–DPU transfer, small WRAM, explicit MRAM–WRAM movement, integer-oriented execution, no native floating point, isolated DPUs, and host-mediated communication.
- The literature motivates a Host–PIM model in which global planning remains on the host and DPUs execute local tiled kernels.
- The literature-derived cost expression is a candidate-plan scoring abstraction. It is not a validated performance predictor.
- Conventional large-scale tensor-network simulators show that slicing, hierarchical parallelism, memory-aware execution, and low-precision communication can be decisive, but those results cannot be assumed to transfer unchanged to UPMEM.

Therefore, the implementation branch must test an architecture hypothesis rather than encode the literature as already-proven truth.

### 3.1 Correct research focus

The implementation research focus is:

> **To design, implement, and evaluate an end-to-end tensor-network quantum-circuit simulator that uses hierarchical parallel execution on commodity UPMEM hardware, and to determine under which circuit, numerical, memory, and scheduling conditions it can improve time-to-solution or energy-to-solution relative to sequential UPMEM and credible CPU/GPU baselines.**

UPMEM is the concrete infrastructure used to instantiate and test the model. The scientific object remains tensor-network contraction within quantum-circuit simulation.

### 3.2 Proposed implementation research questions

#### Main research question

> To what extent can hierarchical parallel execution on commodity UPMEM Processing-in-Memory hardware accelerate end-to-end tensor-network quantum-circuit simulation, in terms of time-to-solution and, where reproducibly measurable, energy-to-solution, relative to sequential UPMEM and established CPU/GPU baselines while satisfying a defined numerical-accuracy requirement?

#### RQ1 — Whole-simulation mapping

> How should a quantum circuit and simulation query be transformed into a contraction, slicing, and Host–DPU execution plan that respects UPMEM memory, arithmetic, and communication constraints?

#### RQ2 — Parallel execution

> How do slice-level, contraction-level, DPU-level, rank-level, and tasklet-level parallelism affect scalability, load balance, utilization, time-to-solution, and energy-to-solution?

This is the principal technical research question.

#### RQ3 — Hardware-adaptation policies

> How do numerical representation, host-side encoding or quantization, WRAM-aware tiling, tensor layout, intermediate residency, transfer batching, and host-mediated reduction affect performance, memory use, energy, and numerical accuracy?

#### RQ4 — Workload dependence and competitiveness

> For which circuit families, circuit sizes, depths, entanglement structures, simulation queries, and contraction characteristics is the UPMEM implementation competitive with same-plan and optimized CPU/GPU tensor-network baselines?

#### Optional secondary question — plan estimation

> Does a UPMEM-aware candidate-plan estimator select better execution plans than conventional FLOP-count and peak-intermediate-size objectives?

This question is optional until enough physical data exist to calibrate and evaluate the estimator.

### 3.3 Contribution hierarchy

The branch must use this hierarchy when naming modules, experiments, reports, and thesis claims:

1. **Primary systems contribution:** hierarchical parallel Host–UPMEM execution of whole tensor-network quantum-circuit simulations.
2. **Enabling implementation contribution:** modular physical mapping of logical contractions to UPMEM numerical, memory, communication, and scheduling constraints.
3. **Empirical contribution:** a compatibility envelope showing where the approach is correct, scalable, competitive, limited, or unsupported.
4. **Optional algorithmic contribution:** measured UPMEM-aware plan reranking or cost modeling.

No agent may silently reverse this hierarchy by making path search the organizing center of the implementation.

---

## 4. Current repository audit

### 4.1 What is already strong and must be preserved

#### A. Claim and evidence discipline

The repository explicitly distinguishes simulator observations, physical hardware execution, development observations, accepted evidence, and unsupported claims. It already states that simulator timings are not hardware speedups and that current M5.5 work does not establish general acceleration, energy efficiency, full scaling, graph-wide residency, or a complete multi-DIMM architecture.

This discipline is a scientific asset. The simplification branch must make it more central, not remove it.

#### B. Strict no-fallback behavior

The physical UPMEM route checks observed backend identity and rejects simulator execution or CPU fallback. A failed hardware execution remains a failed or unsupported hardware observation.

This is non-negotiable.

#### C. `ContractionDAG` as a target-neutral semantic graph

The current DAG correctly separates mathematical contraction semantics from arrays, target estimates, sessions, and runtime resources. It validates shapes, labels, producers, dependencies, and cycles. It also supports explicit slicing through graph transformation.

This is the correct canonical intermediate representation and should become the sole active contraction graph.

#### D. Same-plan replay and separated identities

The repository distinguishes circuit/tensor semantics, planner identity, DAG identity, execution-plan identity, and runtime provenance. It supports a same-DAG NumPy executor and external baselines.

This is essential for separating:

- algorithmic differences;
- target mapping differences;
- executor differences;
- machine and environment differences.

#### E. Credible baseline families

The repository includes serious baseline directions:

- NumPy same-DAG replay for correctness and controlled comparison;
- Quimb/cotengra as an external tensor-network baseline;
- QuEST as a full-state baseline;
- optional verified GPU execution;
- UPMEM simulator for protocol and correctness only;
- physical UPMEM with strict admission.

The simplified system must preserve the distinction between same-plan reference, external TN baseline, full-state baseline, simulator, and physical target.

#### F. Acceptance of negative results

The project already leaves room for conditional or negative findings. That is appropriate. The final thesis contribution may be a precise account of where transfers, numerical representation, synchronization, or insufficient parallel work prevent competitiveness.

### 4.2 Structural problems in the active implementation

#### A. Multiple architectural generations remain active

The current implementation has a new data-first path, but it still depends on legacy structures beneath the boundary:

```text
CircuitSpec
  -> TensorNetworkSpec / TensorInputs
  -> PlannerResult
  -> ContractionDAG
  -> ExecutionPlan
  -> UPMEM adapter
  -> legacy ContractionTask materialization
  -> M5 whole-circuit engine
  -> versioned native/runtime machinery
```

The documentation describes `TaskGraph` as compatibility material, yet the physical adapter converts each `ContractNode` back into `ContractionTask`. Historical whole-circuit classes and M5 shells remain required by the active route.

This means the migration to `ContractionDAG` is conceptually complete but mechanically incomplete.

**Required correction:** the active physical route must consume a physical plan derived directly from `ContractionDAG`. There must be no DAG-to-legacy-TaskGraph or DAG-to-`ContractionTask` conversion in the final active path.

#### B. Repository history is represented as executable architecture

The active UPMEM package contains multiple execution-plan generations and milestone-specific implementations, including files named with `v1`, `v2`, `v3`, `v4`, `m5`, and earlier physical routes.

This is history encoded as current architecture. Git should preserve previous implementations; the active package should preserve only the selected implementation.

**Required correction:** one active UPMEM physical-plan schema, one active ABI, one active host runner, and only the kernels used by the final experiments.

#### C. Package boundaries do not match scientific responsibilities

The current top-level package is divided into many technical namespaces: `bench`, `core`, `environment`, `evidence`, `execution`, `formats`, `plots`, `providers`, `routing`, `targets`, `tn`, `validation`, and `whole_circuit`.

Scientific operations are spread across several of these:

- planning is split across TN, routing, execution, and target code;
- execution is split across execution, whole-circuit, providers, targets, and native code;
- evidence and timing types exist in several places;
- reporting is split across bench modules, scripts, formats, plots, and evidence helpers.

A reader cannot infer the end-to-end thesis pipeline from the package names.

**Required correction:** organize the active code around semantic lowering, planning, numerics, parallel mapping, execution, experiments, and evidence.

#### D. Dataclass proliferation obscures the actual domain

`core/records.py` combines circuit semantics, tensors, legacy contraction tasks, route identities, capabilities, estimates, decisions, timing contracts, validation, benchmark context, and large result records. Other packages define additional execution, routing, and pipeline records.

Many structured types do not own stable invariants. They exist primarily to group local values, preserve migration compatibility, or carry `dict[str, Any]` metadata.

**Required correction:** retain structured types only for stable cross-boundary concepts. Eliminate tuple-wrapper dataclasses, compatibility-only records, duplicate timing structures, route-decision hierarchies, and giant benchmark-result constructors.

#### E. Generic abstractions are ahead of demonstrated variation

Provider registries, route identities, capability types, module-role pipelines, and generic decision objects describe a future extensible platform. Most have one meaningful active implementation or exist to support historical routes.

An abstraction with one implementation does not simplify the thesis. It moves complexity into factories, configuration, metadata, and indirection.

**Required correction:** use direct modules and small functions. Introduce a protocol only where two real executors share a stable interface.

#### F. The CLI and Makefile preserve every milestone

The current Makefile and benchmark CLI expose a large catalog of milestone-specific targets and commands. Historical qualification paths are still visible as public workflow.

This turns orchestration into a second application and makes earlier experiments look like permanent architecture.

**Required correction:** five or six stable user commands, fewer than ten public Make targets, and one experiment/report pipeline.

#### G. Timing scopes need normalization

Current CPU execution includes output hashing inside the measured repeated loop, while UPMEM route totals include session lifecycle and target-specific components. These scopes are not directly symmetric.

**Required correction:** define timing semantics once and record both end-to-end and steady-state views. Hashing and validation must be outside kernel timing. One field name must always mean one scope.

#### H. Target-neutral contracts contain target-version defaults

The generic execution contract currently contains M5/v4-specific kernel, profile, ABI, placement, and reduction identifiers. This makes the generic layer depend on the one implementation it is intended to abstract.

**Required correction:** target-neutral code may describe required work and dependencies. UPMEM identifiers and physical constraints belong only in the UPMEM mapping/runtime modules.

### 4.3 Scientific risks created by the current structure

The current structure creates risks beyond maintainability:

- two supposedly comparable results may pass through different hidden architectures;
- timing fields may include different lifecycle work;
- compatibility records may be mistaken for active semantics;
- future mechanisms may appear implemented because a type or provider exists;
- a milestone-specific identity may leak into a supposedly generic plan;
- agents may extend the wrong layer because the active boundary is unclear;
- deleting one historical route may appear unsafe because it is imported indirectly;
- the volume of configuration makes it difficult to know which variables are controlled in an experiment.

The simplification is therefore part of scientific validity, not only code hygiene.

---

## 5. Target architecture

### 5.1 Architectural principles

The target system must satisfy these principles:

1. **Whole-circuit first.** Primary experiments begin with a quantum circuit and a defined simulation query and end with a validated quantum result.
2. **One canonical semantic graph.** `ContractionDAG` is the only active contraction IR.
3. **Conceptual separation of logical and physical planning.** Contraction order and slicing are distinct from UPMEM numeric, tiling, placement, and communication decisions.
4. **No redundant wrapper by default.** The logical plan may be represented by `ContractionDAG` plus planner provenance; do not add a `LogicalPlan` dataclass if it only wraps those values.
5. **One physical UPMEM plan.** There is one active `UpmemPlan` schema.
6. **Parallelism is explicit.** Work units, assignment, load balance, and reductions are represented and measurable.
7. **Numerics are explicit.** Encoding, scales, rounding, saturation, accumulator width, and decoding are physical-plan decisions.
8. **The host is a planner and coordinator.** Global path selection, slicing, mapping, synchronization, and aggregation remain host-side.
9. **No silent fallback.** Unsupported or failed execution remains visible.
10. **Evidence is part of the architecture.** Every result has reproducible identity, timing scope, accuracy, and provenance.
11. **Historical code is not active code.** Git tags and archived documentation preserve history.
12. **No abstraction without demonstrated alternatives.** A policy interface must have at least two real implementations or a clear testing seam.

### 5.2 Target end-to-end pipeline

```text
SimulationJob
  ├── quantum circuit
  ├── simulation query
  ├── accuracy requirement
  └── deterministic seed
          │
          ▼
Circuit-to-TN lowering
          │
          ▼
TensorNetwork + immutable input arrays
          │
          ▼
Planner
  ├── candidate contraction path
  └── slicing
          │
          ▼
ContractionDAG
  ├── target-neutral semantics
  ├── explicit dependencies
  └── logical identity
          │
          ├───────────────────────────────┐
          ▼                               ▼
CPU same-DAG executor             UPMEM physical mapper
                                  ├── numeric representation
                                  ├── tiling and layout
                                  ├── parallel work units
                                  ├── DPU/rank placement
                                  ├── intermediate residency
                                  ├── transfer schedule
                                  └── host reduction
                                             │
                                             ▼
                                         UpmemPlan
                                             │
                                             ▼
                                       UPMEM runtime
                                             │
          └───────────────────────────────┬──┘
                                          ▼
                                  SimulationResult
                                  ├── output
                                  ├── accuracy
                                  ├── timing
                                  ├── transfer counts
                                  └── backend facts
                                          │
                                          ▼
                                  one evidence row per repetition
                                          │
                                          ▼
                                  tables, plots, and claims
```

### 5.3 Target source layout

Keep the existing package name unless a rename is independently justified. Renaming `quantum_bench` does not contribute to the thesis and should not be combined with the semantic migration unless done once in the first structural commit.

Recommended active layout:

```text
thesis/implementation/
├── README.md
├── ARCHITECTURE.md
├── pyproject.toml
├── uv.lock
├── Makefile
│
├── src/quantum_bench/
│   ├── __init__.py
│   ├── model.py
│   ├── circuits.py
│   ├── lowering.py
│   ├── planning.py
│   ├── numerics.py
│   ├── parallel.py
│   ├── mapping.py
│   ├── cpu.py
│   ├── upmem.py
│   ├── baselines.py
│   ├── experiment.py
│   ├── evidence.py
│   ├── report.py
│   └── cli.py
│
├── native/upmem/
│   ├── host.c
│   ├── dpu.c
│   ├── protocol.h
│   └── Makefile
│
├── native/quest/
│   └── ...
│
├── experiments/
│   ├── smoke.yml
│   └── thesis.yml
│
├── tests/
│   ├── test_model.py
│   ├── test_lowering.py
│   ├── test_planning.py
│   ├── test_numerics.py
│   ├── test_parallel.py
│   ├── test_mapping.py
│   ├── test_cpu.py
│   ├── test_upmem.py
│   ├── test_evidence.py
│   └── test_end_to_end.py
│
└── docs/
    ├── design.md
    ├── methodology.md
    ├── status.md
    └── results.md
```

Files may be merged further when small. Do not create a directory for a single file.

### 5.4 Module boundaries

| Module | Owns | Must not own |
|---|---|---|
| `model.py` | Stable semantic records, `ContractionDAG`, validation, canonical identities | NumPy execution, subprocesses, hardware paths, report formatting |
| `circuits.py` | Deterministic circuit families and query definitions | Target mapping or evidence writing |
| `lowering.py` | Circuit/query to tensor-network data and DAG inputs | UPMEM topology or cost weights |
| `planning.py` | Candidate path generation, slicing, planner provenance | Native execution or hardware allocation |
| `numerics.py` | Host encode/decode, scale metadata, rounding, saturation, numeric error utilities | DPU assignment or session management |
| `parallel.py` | Work-unit decomposition, DPU grouping, tasklet assignment, load-balance estimates | Binary paths or JSON reporting |
| `mapping.py` | Builds and validates the one `UpmemPlan`; tiling, placement, residency, transfer/reduction schedule | Session lifecycle or result aggregation |
| `cpu.py` | Same-DAG reference execution and controlled CPU timing | Route selection or UPMEM identifiers |
| `upmem.py` | Physical/simulator probing, session lifecycle, execution, no-fallback checks | Circuit lowering, path search, report generation |
| `baselines.py` | Quimb/cotengra, QuEST, and verified GPU adapters | General provider registry |
| `experiment.py` | Matrix expansion, warmups, repetitions, ablations, validation orchestration | Plot implementation details |
| `evidence.py` | One evidence schema, canonical serialization, hashes, comparison compatibility, claim policy | Hardware execution |
| `report.py` | Generic tables and plots from evidence | Running experiments |
| `cli.py` | Thin command parsing and dispatch | Milestone-specific business logic |

### 5.5 Dependency direction

```text
cli
 ├── experiment
 │    ├── circuits
 │    ├── lowering ─────── model
 │    ├── planning ─────── model
 │    ├── numerics
 │    ├── parallel
 │    ├── mapping ──────── model, numerics, parallel
 │    ├── cpu ──────────── model
 │    ├── upmem ────────── model, mapping
 │    ├── baselines
 │    └── evidence
 └── report ────────────── evidence
```

Forbidden dependency directions:

- `model.py` importing executors, evidence, CLI, filesystem, subprocess, or target constants;
- `planning.py` importing native UPMEM code;
- `cpu.py` importing UPMEM plan types;
- `report.py` executing benchmarks;
- `upmem.py` reconstructing `TaskGraph` or `ContractionTask`;
- `mapping.py` opening sessions or reading machine-local binary paths;
- baseline adapters importing route/provider registries.

### 5.6 Canonical semantic representation

`ContractionDAG` remains the canonical logical execution graph.

It must contain only:

- tensor descriptors;
- contract/reduce nodes;
- labels, shapes, dtypes, and fixed slices;
- dependencies;
- final output view.

It must not contain:

- tensor arrays;
- planner runtime;
- target costs;
- DPU IDs;
- tasklet counts;
- tile sizes;
- numeric scales;
- ABI or kernel IDs;
- physical rank paths.

Slicing that changes mathematical execution belongs in the DAG. Tiling that only decomposes one semantic contraction belongs in the physical plan.

### 5.7 Logical-plan representation

There are two conceptual planning layers, but the implementation should avoid redundant records.

Preferred representation:

```text
Planner output:
  ContractionDAG
  + PlannerProvenance stored for evidence
```

A separate `LogicalPlan` class is justified only when it replaces several existing types and owns stable invariants not already represented by the DAG. It must not be introduced merely because the architecture diagram has a logical-plan box.

Planner provenance may be a small immutable record or a validated mapping containing:

```text
planner_id
planner_version
objective
options
seed
planning_time_s
candidate_count
selected_candidate
```

### 5.8 One physical UPMEM plan

The active target-specific plan should be one type, for example:

```python
@dataclass(frozen=True, slots=True)
class UpmemPlan:
    dag_hash: str
    topology: UpmemTopology
    numeric: NumericSpec
    work_units: tuple[WorkUnit, ...]
    assignments: tuple[Assignment, ...]
    residency: tuple[ResidencyDecision, ...]
    transfers: tuple[TransferStep, ...]
    reductions: tuple[ReductionStep, ...]
    kernel_id: str
```

This example does not require every nested item to be a dataclass. Small local records may be tuples or validated mappings. The final shape should minimize public types while preserving plan serialization and invariants.

Machine-local resources are not part of the plan:

```text
rank device paths
host binary path
dpu binary path
working directory
timeout
local SDK installation
```

They belong to runtime configuration and environment identity.

### 5.9 Dataclass and tuplet rules

The target should contain roughly five to ten public structured classes, not several dozen.

A class may remain when it satisfies most of these conditions:

1. crosses at least two architectural boundaries;
2. owns meaningful validation invariants;
3. has stable identity or serialization semantics;
4. has several independent consumers;
5. names an important scientific concept.

Likely survivors:

```text
SimulationJob
TensorSpec
TensorView
ContractNode / ReduceNode
ContractionDAG
UpmemTopology
UpmemPlan
Measurement
SimulationResult
```

Some of these may be combined.

Remove or replace:

- classes used only to return two or three local values;
- wrappers around `Mapping[str, np.ndarray]`;
- route identity/capability/estimate/decision hierarchies;
- multiple timing records;
- milestone-specific plan records;
- private one-use result bundles;
- records whose main payload is `dict[str, Any]`;
- compatibility aliases preserved only for positional construction.

Rules for simple values:

- use local variables when values do not cross a boundary;
- use a tuple only when ordering is obvious and the tuple remains local;
- use a mapping for optional diagnostics;
- do not replace every dataclass with `NamedTuple`;
- do not create a new dataclass to avoid changing a function signature;
- prefer changing the boundary itself.

`TensorInputs` should become:

```python
TensorInputs = Mapping[str, np.ndarray]
```

Validation occurs once at executor boundaries.

### 5.10 Hierarchical parallelism model

Parallelism is the primary direct implementation contribution. The architecture must make the following concepts explicit and measurable.

#### Level A — Tasklet parallelism

Tasklets divide local output rows, output elements, or tiles within one DPU.

The plan records:

```text
tasklets_per_dpu
local partition rule
local work per tasklet
barrier count
```

#### Level B — Output-tile parallelism across DPUs

Independent output regions of one contraction are assigned to different DPUs or DPU groups. Prefer output partitioning when possible because it avoids cross-DPU partial-sum reductions.

The plan records:

```text
output tile bounds
assigned DPU group
input replication requirements
output ownership
```

#### Level C — Slice parallelism across DPU groups

Slicing creates independent contraction instances. This is a natural source of UPMEM parallelism because slices can execute without fine-grained inter-DPU dependencies and can be combined on the host.

The plan records:

```text
slice coordinates
DPU group
expected work
final aggregation rule
```

#### Level D — DAG-frontier parallelism

Independent ready nodes may execute concurrently when their data placement permits it. This is more complex and should follow slice/output parallelism rather than precede it.

#### Level E — Transfer/compute overlap

Double buffering or asynchronous host preparation may be evaluated only after static execution is correct and measurable.

#### Minimum final hierarchical claim

A final thesis claim of hierarchical parallel execution should require at least two real parallel levels, for example:

```text
multiple tasklets per DPU
+
multiple DPUs or DPU groups across output tiles or slices
```

DAG-frontier concurrency and overlap may remain later extensions.

#### Parallelization measurements

Every physical row should make it possible to compute or estimate:

```text
number of work units
work-unit size distribution
DPU group count
active versus allocated DPUs
tasklets per DPU
predicted and observed imbalance
critical-path time
host coordination count
reduction count
```

### 5.11 Numerical representation

UPMEM's lack of native floating-point arithmetic makes numerical representation a first-class physical policy.

The first practical design should perform encoding on the host:

```text
complex host tensor
  -> scale selection
  -> rounding and saturation check
  -> real/imaginary packing
  -> integer or block-floating payload
  -> UPMEM integer kernel
  -> encoded intermediate or result
  -> host decode or rescale when required
```

The physical plan must record:

```text
numeric mode
payload width
accumulator width
scale granularity
rounding mode
saturation behavior
complex layout
rescaling points
```

The implementation must time:

```text
encode_s
upload_s
kernel_s
host_reduce_s
download_s
decode_s
```

Host quantization is not free preprocessing and may not be excluded from end-to-end acceleration claims.

Resident intermediates should remain encoded when possible. The runtime should not repeatedly download, decode, re-encode, and upload an intermediate unless required by mapping or accuracy policy.

Start with one floating reference and one practical encoded mode. Do not create a broad quantization framework with unused modes.

Accuracy must be evaluated against the selected quantum query, not only generic array closeness. Depending on the query, record amplitude error, probability error, expectation-value error, norm drift, or fidelity.

### 5.12 Role of contraction-path and plan search

Path search remains part of the simulator but is not the organizing center of the architecture.

A practical sequence is:

1. generate candidate paths with an existing planner such as cotengra or opt_einsum;
2. generate feasible slicing choices;
3. build a `ContractionDAG` for each candidate;
4. map each candidate to an `UpmemPlan`;
5. estimate or measure the physical plan;
6. select the best candidate under the declared objective.

The literature-derived feature set may include:

```text
host_dpu_bytes
mram_wram_bytes
dpu_integer_work
synchronization_count
numeric_conversion_work
peak_wram
parallel_work_units
predicted_imbalance
```

Use a small estimator function or one compact result record. Do not create one class per cost term.

The estimator must be labeled according to evidence status:

- analytical and uncalibrated;
- microbenchmark-calibrated;
- cross-validated on physical runs;
- descriptive only.

A path with fewer FLOPs is not assumed to be better on UPMEM. Conversely, a UPMEM-aware score is not assumed to be useful until evaluated.

### 5.13 Runtime boundary

The UPMEM runtime owns:

- physical versus simulator probing;
- explicit rank allocation;
- host and DPU binary identity;
- session lifecycle;
- transfers and launches;
- execution of the physical plan;
- terminal metadata validation;
- no-fallback checks;
- result collection.

It does not own:

- contraction-path search;
- semantic graph construction;
- experiment matrix generation;
- report formatting;
- target-neutral plan types.

The final runtime must not import `TaskGraph`, `ContractionTask`, milestone-specific whole-circuit classes, or versioned execution-plan modules.

### 5.14 Evidence and timing model

Write one raw evidence row per repetition. Aggregates and plots are derived artifacts.

Minimum row contents:

```text
schema_version
repository_commit
worktree_dirty
problem_id
circuit_family
circuit_parameters
simulation_query
seed
dtype
tensor_network_hash
dag_hash
planner_id
planner_options
physical_plan_hash
executor_id
environment_id
execution_mode
warmup_or_measured
repetition_index
status
unsupported_reason
timing_scope
timing_components
host_dpu_bytes
mram_wram_bytes_or_estimate
allocated_dpus
active_dpus
tasklets_per_dpu
output_hash
accuracy_metrics
validation_status
backend_facts
```

Timing components should have stable meanings:

```text
planning_s
mapping_s
encode_s
prepare_s
h2d_s
kernel_s
host_reduce_s
d2h_s
decode_s
validation_s
session_open_s
session_close_s
steady_state_s
end_to_end_s
```

Rules:

- output hashing is outside `kernel_s`;
- validation is outside `kernel_s`;
- `steady_state_s` excludes one-time setup according to a documented rule;
- `end_to_end_s` includes every operation required for the user-visible result;
- CPU and UPMEM timing scopes are defined symmetrically where possible;
- `null` is better than assigning a misleading value;
- simulator and physical rows cannot be paired as hardware speedups;
- unsupported and failed rows remain in the dataset.

---

## 6. Separate-branch strategy

The architecture reset must not be developed directly on `main`.

### 6.1 Baseline preservation

Before starting:

```bash
git switch main
git pull --ff-only

git tag -a pre-thesis-runtime-simplification \
  -m "Baseline before strong thesis architecture simplification"
git push origin pre-thesis-runtime-simplification
```

The tag preserves the exact implementation and accepted evidence state before the reset.

Do not rewrite or delete accepted evidence while creating the branch. Existing evidence may be copied into test fixtures or migrated, but its original form remains reachable through the tag.

### 6.2 Integration branch

Create one long-lived integration branch:

```bash
git switch -c refactor/thesis-runtime-simplification
git push -u origin refactor/thesis-runtime-simplification
```

No architecture-reset work is merged directly to `main` until all final gates pass.

### 6.3 Agent branches

Each agent works from the integration branch:

```text
refactor/thesis-runtime-simplification
├── agent/wp01-architecture-contract
├── agent/wp02-canonical-model
├── agent/wp03-planning
├── agent/wp04-numerics
├── agent/wp05-parallel-mapping
├── agent/wp06-upmem-runtime
├── agent/wp07-cpu-baselines
├── agent/wp08-evidence
├── agent/wp09-experiment-cli
└── agent/wp10-tests-cleanup
```

Every pull request targets `refactor/thesis-runtime-simplification`, never `main`.

### 6.4 Branch governance

The integration branch should be treated as protected even when repository settings do not enforce it:

- no force pushes;
- no direct commits except by the integration coordinator;
- required test pass before merge;
- required architecture-guide compliance review;
- one work package per PR;
- no hidden hardware claims;
- no merge with unresolved compatibility code marked “temporary” unless a dated removal issue exists in the same branch.

### 6.5 Commit policy

A work-package PR should contain small, reviewable commits. Recommended pattern:

```text
contract: add new boundary and tests
migrate: move one active path to the boundary
delete: remove the replaced compatibility path
docs: update architecture and migration ledger
```

Do not combine broad renaming, semantic changes, and deletion in one opaque commit.

### 6.6 Integration coordinator responsibilities

One coordinator must own:

- the branch-level architecture decisions;
- conflict resolution;
- the migration ledger;
- temporary adapter expiry;
- evidence compatibility decisions;
- final branch qualification;
- deciding when old paths are deleted.

Agents may propose changes but may not independently add a new architectural layer.

---

## 7. Agent operating contract

Every agent must read this document before editing code.

### 7.1 Mandatory rules

An agent MUST:

- state the work package and base commit;
- list files it intends to modify before implementation;
- preserve all non-negotiable scientific invariants;
- add or update tests for every moved invariant;
- update the migration ledger in its PR;
- distinguish physical, simulator, and CPU execution explicitly;
- leave unsupported behavior explicit;
- remove replaced compatibility code within the same work package when safe;
- report any unresolved ambiguity rather than inventing behavior.

An agent MUST NOT:

- add a provider registry, service container, plugin framework, or route hierarchy;
- add a milestone- or version-numbered active module;
- add a dataclass solely to group local return values;
- make the generic model import UPMEM identifiers;
- change the scientific DAG when applying a numeric mode or target-local tile;
- silently change timing scope;
- silently fall back from physical UPMEM to simulator or CPU;
- report simulator timing as hardware performance;
- delete raw evidence or rewrite accepted historical records;
- add placeholder strategies that are not executed by a test or experiment;
- retain a compatibility adapter merely because deletion is inconvenient.

### 7.2 Complexity budget

Every PR should either reduce active complexity or justify a temporary increase with an explicit removal step.

The PR description must report:

```text
active Python files added/removed
public structured types added/removed
active CLI commands added/removed
compatibility imports added/removed
```

A net increase is not automatically prohibited, but it must be necessary for the target architecture rather than migration convenience.

### 7.3 Decision escalation

Stop and ask for an architecture decision when:

- current tests and documentation disagree about semantics;
- a compatibility field appears in accepted evidence but has no clear meaning;
- a physical runtime assumption cannot be verified without hardware;
- two work packages need to own the same data type;
- preserving an old schema would force a new permanent abstraction;
- a requested optimization would alter the simulation query or accuracy contract.

Do not guess.

---

## 8. Work packages

### WP0 — Freeze baseline and establish branch controls

**Owner:** integration coordinator  
**Dependencies:** none

#### Scope

- tag the current baseline;
- create the integration branch;
- record current tests, commands, evidence snapshots, and known failures;
- add this guide to the branch;
- add a migration ledger.

#### Deliverables

```text
docs/architecture_simplification.md
MIGRATION_LEDGER.md
baseline test output
baseline file/type/command inventory
```

#### Acceptance

- baseline tag exists remotely;
- integration branch exists remotely;
- current accepted evidence remains unchanged;
- all known current tests are recorded as pass/fail rather than assumed;
- no implementation behavior has changed.

---

### WP1 — Freeze research contract and active claim boundary

**Owner:** architecture/documentation agent  
**Dependencies:** WP0

#### Scope

- update root README and architecture documentation with the corrected research focus;
- state contribution hierarchy and research questions;
- define primary simulation query or bounded set of queries;
- define exact versus approximate scope;
- define current and intended claim boundaries;
- define non-goals.

#### Required decisions

```text
primary circuit input format
supported gate set
primary simulation query
accuracy metrics
energy measurement availability
final baseline families
```

#### Acceptance

- no document describes path search as the central contribution;
- whole-circuit circuit-to-result semantics are explicit;
- parallelization is the primary direct contribution;
- supporting policies are named but not claimed as implemented without evidence;
- energy claims are conditional on reproducible whole-system measurement.

---

### WP2 — Extract the canonical semantic model

**Owner:** model agent  
**Dependencies:** WP1

#### Scope

- move or consolidate `TensorSpec`, `TensorView`, node types, `ContractionDAG`, validation, and hashing into the canonical model boundary;
- replace `TensorInputs` wrappers with `Mapping[str, np.ndarray]`;
- make DAG validation independent of `TensorNetworkSpec(None, ...)` reconstruction;
- preserve semantic hash behavior or document an intentional schema change;
- create a temporary one-way `TaskGraph -> ContractionDAG` reader only if needed for tests or old records.

#### Forbidden

- no reverse `ContractionDAG -> TaskGraph` adapter;
- no target fields in the DAG;
- no new generic graph hierarchy.

#### Acceptance

- CPU model tests pass using only the canonical DAG;
- input validation operates directly on DAG descriptors;
- active code no longer needs a fake circuit value to validate inputs;
- semantic identities are covered by deterministic tests;
- any temporary legacy reader is isolated outside the active execution path.

---

### WP3 — Consolidate planning and slicing

**Owner:** planning agent  
**Dependencies:** WP2

#### Scope

- consolidate planner adapters and planner result/provenance;
- keep candidate path generation and slicing target-neutral;
- ensure slicing that changes semantics rewrites the DAG;
- remove route/provider selection from the planner;
- support candidate enumeration for later UPMEM reranking without making reranking mandatory.

#### Deliverables

```text
planning.py
planner determinism tests
candidate-plan API
planner provenance schema
```

#### Acceptance

- same inputs, seed, and options produce deterministic selected output where the underlying planner permits it;
- planner output contains no DPU IDs, ABI IDs, or binary paths;
- path and slicing choices are reproducibly identified;
- conventional planner operation remains available for final experiments;
- no physical hardware is required to test planning.

---

### WP4 — Extract numerical representation

**Owner:** numerics agent  
**Dependencies:** WP2; interface coordination with WP5

#### Scope

- create one explicit host numeric encoding boundary;
- implement the floating reference and one practical encoded mode;
- define scale, rounding, saturation, packing, accumulator, and complex-layout semantics;
- expose measured encode/decode timing;
- create accuracy tests on analytically known contractions and small circuits;
- remove numeric policy classes that only wrap fixed behavior.

#### Acceptance

- encoding and decoding are deterministic;
- overflow and saturation are explicit;
- complex data behavior is explicit; unsupported imaginary values fail clearly rather than being silently discarded;
- encoded payload size is measurable;
- host conversion time is recorded;
- accuracy metrics are tied to the simulation query;
- no numeric choice changes the DAG hash.

---

### WP5 — Implement parallel decomposition and one physical mapping model

**Owner:** parallel/mapping agent  
**Dependencies:** WP2, WP3, WP4 interfaces

#### Scope

- define work units for tasklets, output tiles, and slices;
- define DPU groups and rank placement;
- estimate work and imbalance;
- define intermediate residency and transfer steps;
- build one `UpmemPlan` from a `ContractionDAG` and mapping options;
- validate WRAM, MRAM, topology, assignment completeness, and reduction requirements;
- keep machine-local resources out of plan identity.

#### Minimum implementation order

1. tasklet partitioning;
2. output-tile distribution across DPUs;
3. slice distribution across DPU groups;
4. optional DAG-frontier concurrency;
5. optional overlap.

#### Acceptance

- every semantic output element or slice is owned exactly once unless a declared reduction requires partial ownership;
- assignments neither overlap incorrectly nor leave gaps;
- plan serialization is deterministic;
- plan validation rejects WRAM and topology violations before hardware allocation;
- load-balance estimates are present;
- one physical plan schema replaces versioned plan families;
- mapping imports no session or binary code.

---

### WP6 — Consolidate the UPMEM runtime

**Owner:** UPMEM runtime agent  
**Dependencies:** WP5

#### Scope

- make the runtime execute `UpmemPlan` directly;
- remove `ContractionTask` conversion;
- extract the active native protocol and kernel path;
- retain strict physical/simulator and no-fallback validation;
- preserve backend facts and binary hashes;
- reduce active native code to one host runner, one shared protocol, and only measured kernels;
- remove M5/version names from active runtime symbols and files.

#### Acceptance

- active UPMEM execution imports no `TaskGraph`, `ContractionTask`, versioned plan module, or historical whole-circuit class;
- simulator execution cannot be admitted as physical;
- CPU fallback is impossible or explicitly rejected;
- terminal identity is validated;
- physical resource paths remain runtime facts, not plan identity;
- simulator smoke tests pass without physical hardware;
- physical tests are explicitly gated and do not run accidentally.

---

### WP7 — Simplify CPU execution and external baselines

**Owner:** baseline agent  
**Dependencies:** WP2, WP3

#### Scope

- implement same-DAG CPU replay with clean timing;
- move output hashing outside kernel timing;
- provide optional sequential and host-parallel CPU modes only when clearly identified;
- consolidate Quimb/cotengra and QuEST adapters into direct baseline functions;
- remove provider/route registries.

#### Acceptance

- same-DAG CPU output matches analytical fixtures;
- CPU and UPMEM share the same DAG for controlled comparisons;
- output hashing and validation are outside `kernel_s`;
- external baselines are labeled cross-plan or same-plan correctly;
- GPU results require verified physical GPU execution;
- no baseline adapter is described as active merely because an import exists.

---

### WP8 — Unify evidence, timing, and comparison policy

**Owner:** evidence agent  
**Dependencies:** interfaces from WP2, WP5, WP6, WP7

#### Scope

- define one `Measurement` or timing mapping;
- define one raw evidence row schema;
- define canonical serialization and domain-separated hashes;
- define comparison compatibility rules;
- preserve unsupported and failed rows;
- define migration of old evidence only where required.

#### Required hashes

```text
problem_id
dag_id
physical_plan_id
environment_id
```

Planner provenance may be included in `problem_id`, `dag_id`, or a separate field according to the documented identity model, but the rule must be unambiguous.

#### Acceptance

- one row per repetition;
- one canonical serializer;
- no duplicate timing schema in `core`, `execution`, and `bench`;
- same-plan pairing checks DAG and relevant physical semantics;
- simulator/physical incompatibility is enforced;
- old evidence is either migrated through one isolated reader or explicitly left historical;
- report code cannot silently pair incompatible rows.

---

### WP9 — Collapse experiment orchestration, CLI, configuration, and reports

**Owner:** orchestration agent  
**Dependencies:** WP3–WP8

#### Scope

- replace milestone-specific commands with stable commands;
- create `smoke.yml` and `thesis.yml` experiment matrices;
- implement matrix expansion, warmups, repetitions, ablations, and run directories;
- consolidate reporting into one report pipeline;
- reduce Makefile to convenience targets only.

#### Target commands

```text
quantum-bench doctor
quantum-bench plan experiments/thesis.yml
quantum-bench run experiments/thesis.yml
quantum-bench verify <run-or-snapshot>
quantum-bench report <run-or-snapshot>
```

A sixth command for explicitly gated hardware smoke execution is acceptable when it improves safety.

#### Target Makefile

```text
setup
test
lint
doctor
plan
run
report
verify
hardware-smoke
clean
```

#### Acceptance

- no public milestone-numbered command remains;
- historical workflows are reachable through the baseline tag, not active help text;
- one experiment file can express circuit, planner, numeric, topology, and ablation dimensions;
- report generation never reruns hardware;
- a new reader can execute a CPU smoke experiment from the README alone.

---

### WP10 — Reorganize tests and delete compatibility architecture

**Owner:** test/cleanup agent  
**Dependencies:** all migration work packages

#### Scope

- reorganize tests around scientific invariants;
- parameterize useful historical cases;
- delete compatibility modules from the active import path;
- delete versioned execution plans and milestone-specific runtime modules;
- delete obsolete configs, commands, and report scripts;
- move only genuinely necessary provenance material to legacy/archive documentation.

#### Target test modules

```text
test_model.py
test_lowering.py
test_planning.py
test_numerics.py
test_parallel.py
test_mapping.py
test_cpu.py
test_upmem.py
test_evidence.py
test_end_to_end.py
```

#### Acceptance

- deleting the archive does not affect imports or tests;
- active source contains no milestone-numbered module;
- active execution contains no TaskGraph compatibility path;
- old cases survive only as parameterized fixtures when scientifically useful;
- all public structured types have a documented reason to exist;
- file and type counts meet the branch complexity targets or have explicit justification.

---

### WP11 — Final scientific qualification

**Owner:** integration coordinator with experiment operator  
**Dependencies:** WP0–WP10

#### Scope

- freeze circuit families, seeds, queries, planners, numeric modes, and topology ranges;
- run correctness qualification;
- freeze selected DAG and physical-plan hashes;
- run sequential and parallel UPMEM experiments;
- run same-plan CPU and optimized external baselines;
- run energy measurements only under a documented whole-system method;
- generate all final tables and plots from raw rows;
- document negative and unsupported results.

#### Acceptance

- every thesis number traces to a raw row, repository commit, DAG hash, plan hash, and environment manifest;
- scaling speedup and absolute competitiveness are reported separately;
- end-to-end and steady-state timing are both available;
- no simulator result appears in a physical performance table;
- no unsupported combination disappears from the matrix;
- final claims are no stronger than the recorded evidence.

---

## 9. Current-to-target migration map

| Current area | Target | Action |
|---|---|---|
| `core/records.py` | `model.py`, `evidence.py`, small runtime result type | Split by responsibility; delete grab-bag |
| `tn/graph.py` | `model.py` or retained as one focused graph module | Preserve canonical DAG and tests |
| `tn/network.py` | `lowering.py` | Remove migration wrappers and array container classes |
| `tn/task_graph.py` | isolated legacy reader, then delete | No active execution dependency |
| `tn/planning.py` and planner modules | `planning.py` | One planner interface and provenance shape |
| `execution/contracts.py` | `model.py`, `mapping.py`, `evidence.py` | Remove M5/v4 defaults from generic contracts |
| `execution/compiler.py` | `mapping.py` | Physical compiler belongs with UPMEM mapping |
| `execution/cpu.py` | `cpu.py` | Clean timing and direct DAG validation |
| `execution/upmem.py` | `upmem.py` | Direct `UpmemPlan` execution; no legacy task conversion |
| `whole_circuit/*` | delete or absorb selected runtime code | Remove historical intermediate architecture |
| `targets/upmem/execution_plan*.py` | `mapping.py` / one `UpmemPlan` | One active schema |
| `targets/upmem/distributed_plan*.py` | one physical mapping implementation | Delete version families |
| `targets/upmem/m5_*` | mechanism-named modules or delete | No milestone names in active code |
| `targets/upmem/hardware_taskgraph*` | delete after parity | Historical route, not active architecture |
| `routing/*` | delete | Experiment config selects executor directly |
| `providers/*` | `baselines.py` or direct runtime adapter | No registry |
| `bench/runner.py` | `experiment.py` | Orchestration only |
| `bench/__main__.py` | `cli.py` | Five stable commands |
| milestone report scripts | `report.py` | One report pipeline |
| milestone configs | `experiments/smoke.yml`, `experiments/thesis.yml` | Matrix dimensions instead of many files |
| `formats`, `plots`, `environment`, separate evidence packages | flat focused modules | Remove one-file namespaces |
| milestone/version tests | invariant-oriented parameterized tests | Preserve cases, not architecture history |
| multiple native lanes | `native/upmem` | One active ABI and measured kernels |
| milestone documentation | Git tag plus concise archive index | Four active documents |

---

## 10. Required experimental architecture

### 10.1 Primary experimental unit

A primary benchmark case must be:

```text
quantum circuit
+ simulation query
+ planner/slicing policy
+ accuracy requirement
+ executor/mapping policy
```

It must produce:

```text
quantum result
+ correctness metrics
+ performance measurements
+ energy measurements when available
+ provenance
```

Synthetic contractions and microbenchmarks are supporting evidence only. They calibrate kernels, transfers, WRAM behavior, tasklet scaling, and estimators.

### 10.2 Circuit families

Use a controlled set that changes TN structure rather than a large arbitrary suite. Candidate categories include:

- random/grid circuits with controllable depth and entanglement;
- QAOA-like circuits;
- one structured low-entanglement family;
- one family requiring meaningful slicing.

The exact final set belongs in `docs/methodology.md` and `experiments/thesis.yml`.

### 10.3 Simulation query

Do not attempt to support every simulator output. Select one primary query and at most one secondary query, for example:

```text
primary: a batch of selected output amplitudes
secondary: an expectation value for a selected observable
```

The query must determine correctness metrics and baseline compatibility.

### 10.4 Required comparisons

#### Controlled same-plan comparison

```text
same circuit
same query
same ContractionDAG
same slicing
same accuracy target
CPU same-DAG executor versus UPMEM physical plan
```

This isolates execution and physical mapping.

#### Best-practical-backend comparison

```text
same circuit
same query
same accuracy target
best reasonable plan for each backend
```

This addresses practical competitiveness.

#### UPMEM parallel scaling comparison

```text
one DPU / one tasklet
one DPU / multiple tasklets
multiple DPUs / output tiles
multiple DPU groups / slices
full selected hierarchical mode
```

Speedup over the sequential UPMEM baseline demonstrates scalability. It does not by itself demonstrate superiority over CPU or GPU.

### 10.5 Policy ablations

Use one-factor-at-a-time or staged ablations rather than a complete Cartesian product:

```text
floating reference versus selected encoded format
re-upload versus resident intermediate
simple versus WRAM-aware tiling
single-level versus hierarchical parallelism
conventional plan versus optional UPMEM reranking
```

First identify a stable default configuration. Then vary one policy while fixing the others.

---

## 11. Test and evidence gates

### 11.1 Semantic gates

- DAG rejects cycles, missing producers, invalid labels, shape mismatches, and unsupported slicing.
- DAG hashes are deterministic and tested against documented semantic changes.
- Numeric and target-local mapping never change the DAG hash.
- Simulation query and output ordering are explicit.

### 11.2 Planning gates

- candidate paths and slicing are reproducible under a fixed seed and configuration;
- planner provenance is serialized;
- candidate reranking is separated from candidate generation;
- uncalibrated scores are labeled modeled, not measured.

### 11.3 Numerics gates

- known values encode/decode correctly;
- overflow and saturation are detected;
- complex layout is tested;
- accuracy is measured at circuit-query level;
- quantization time and payload size are recorded.

### 11.4 Parallel mapping gates

- work units cover the intended domain;
- ownership is exact;
- reductions are explicit;
- topology and WRAM constraints fail before allocation;
- plan serialization round-trips;
- load imbalance is measurable.

### 11.5 Runtime gates

- simulator and physical modes cannot be confused;
- physical execution requires explicit opt-in and rank paths;
- CPU fallback is rejected;
- terminal metadata is validated;
- failed allocation and kernel execution remain failed rows;
- session close failures are visible.

### 11.6 Timing gates

- hashing is outside kernel timing;
- validation is outside kernel timing;
- session lifecycle is recorded separately;
- end-to-end and steady-state scopes are documented;
- CPU and UPMEM fields have the same semantic definitions;
- all timings are finite and non-negative.

### 11.7 Evidence gates

- one raw row per repetition;
- canonical serialization is deterministic;
- comparison compatibility is enforced;
- unsupported rows remain present;
- every result identifies repository state and environment;
- reports are reproducible from raw rows without rerunning hardware.

---

## 12. Complexity targets

These are architectural targets, not arbitrary code-golf requirements.

| Area | Target |
|---|---:|
| Active top-level Python package layout | one flat package |
| Active Python modules | approximately 12–16 |
| Active contraction IRs | 1 |
| Active UPMEM physical-plan schemas | 1 |
| Active native ABIs | 1 |
| Public structured domain/result types | approximately 5–10 |
| Public CLI commands | 5–6 |
| Public Make targets | 10 or fewer |
| Principal experiment files | 2 |
| Active test modules | approximately 10 |
| Active design/method/result documents | 4 |
| Canonical serializers | 1 |
| Canonical timing schema | 1 |
| Compatibility conversion in active executor | 0 |
| Milestone/version names in active source filenames | 0 |

A reduction of roughly 50% or more in active Python files is plausible because many current files represent historical versions, compatibility routes, wrappers, and milestone orchestration. The exact percentage is secondary to the architectural gates.

---

## 13. Definition of done for the integration branch

The branch is ready for final review only when all of the following hold.

### Architecture

- `ContractionDAG` is the only active contraction graph.
- UPMEM execution consumes one physical plan directly.
- No active runtime reconstructs `ContractionTask` or `TaskGraph`.
- One physical-plan schema and one ABI remain.
- Numerics, parallel decomposition, physical mapping, runtime, and evidence have explicit boundaries.
- No provider/route/module-role framework remains.

### Scientific semantics

- primary cases are whole-circuit simulations;
- the simulation query is explicit;
- CPU same-DAG and UPMEM paths share semantic inputs;
- external baselines are clearly labeled;
- accuracy policy is explicit;
- parallel scaling and absolute competitiveness are separate claims.

### Execution integrity

- no CPU or simulator fallback from the physical path;
- simulator results cannot enter physical performance reports;
- machine-local resources are not part of logical plan identity;
- unsupported and failed rows remain visible.

### Measurement integrity

- hashing and validation are excluded from kernel timing;
- end-to-end and steady-state time are both available;
- host encode/decode and transfer costs are included appropriately;
- energy-to-solution includes the host and is reported only when reproducible.

### Simplification

- active source filenames contain no milestone or implementation-version suffixes;
- historical code is reachable through the baseline tag, not imported by active code;
- public types and commands meet the complexity budget or have documented exceptions;
- a new reader can understand the pipeline from README and `docs/design.md`.

### Reproducibility

- dependencies are locked;
- the repository commit and dirty state are recorded;
- seeds are deterministic;
- raw rows are retained;
- every table and plot is generated from raw evidence;
- final benchmark configuration is frozen.

---

## 14. Recommended first sequence of integration commits

```text
1. docs: freeze research focus, branch policy, and architecture decisions
2. model: make ContractionDAG the sole canonical semantic IR
3. model: replace TensorInputs wrappers and direct DAG input validation
4. planning: consolidate planner provenance and slicing
5. numerics: extract host encoding/decoding and accuracy policy
6. mapping: introduce the one UpmemPlan and work-unit model
7. parallel: add tasklet/output-tile/slice decomposition
8. runtime: execute UpmemPlan without ContractionTask conversion
9. baseline: simplify CPU same-DAG and external adapters
10. evidence: unify timing, identities, rows, and comparison rules
11. interface: replace milestone CLI, Make targets, configs, and reports
12. cleanup: delete versioned plans, whole-circuit compatibility, routing, providers
13. tests: reorganize around scientific invariants
14. reproducibility: lock environment and freeze final experiment matrix
```

Every commit must leave the branch testable. Temporary adapters require a named removal commit later in the sequence.

---

## 15. Reusable agent task template

Use the following template when assigning a work package to an agent.

```text
Task: <work-package title>
Base branch: refactor/thesis-runtime-simplification
Base commit: <sha>
Normative guide: docs/architecture_simplification.md

Goal:
<one paragraph>

Allowed files:
<paths>

Forbidden changes:
<paths or architectural rules>

Scientific invariants:
- ContractionDAG remains target-neutral.
- Numeric policy does not change semantic identity.
- No physical-to-simulator/CPU fallback.
- Unsupported behavior remains explicit.
- Timing scope is not changed silently.

Required deliverables:
- implementation
- tests
- documentation update
- migration-ledger update
- file/type/command delta

Acceptance commands:
<commands>

Expected deletion:
<old files or types that this work replaces>

Stop conditions:
<ambiguities that require coordinator decision>
```

An agent's final report must state:

```text
what changed
what was deleted
which invariants were tested
which behavior remains unsupported
whether any temporary adapter remains
exact commands run
```

---

## 16. Risks and rollback

### Risk: semantic drift during graph migration

**Mitigation:** parity fixtures compare old and new outputs for small circuits before the old path is deleted. Freeze representative DAG hashes when hash semantics are intended to remain stable.

### Risk: agents create new layers to ease migration

**Mitigation:** complexity budget, no-abstraction rule, coordinator review, and required deletion in each work package.

### Risk: accepted evidence becomes unreadable

**Mitigation:** preserve the baseline tag; add at most one isolated historical evidence reader; do not force the active runtime to carry old schemas.

### Risk: physical hardware is unavailable during refactor

**Mitigation:** separate pure mapping tests, simulator protocol tests, and explicitly gated physical smoke tests. Do not claim physical parity without a physical run.

### Risk: branch becomes too long-lived and diverges from `main`

**Mitigation:** freeze architecture-related changes on `main`, periodically merge only small non-conflicting documentation or dependency fixes, and avoid feature development on both branches.

### Risk: parallel speedup is mistaken for platform competitiveness

**Mitigation:** report sequential UPMEM scaling and CPU/GPU absolute comparisons in separate tables and claims.

### Risk: quantization improves speed but invalidates simulation accuracy

**Mitigation:** predeclare query-specific accuracy thresholds, record saturation, and keep floating reference rows.

### Risk: cost model becomes a speculative project

**Mitigation:** do not implement calibration before physical traces exist. Candidate generation and physical mapping must work without it.

### Rollback rule

If the branch cannot satisfy semantic parity or strict execution provenance, do not partially merge it into `main`. Retain it as an experimental branch and continue using the tagged baseline until the blocking issue is resolved.

---

## 17. Final target state

The finished repository should communicate one coherent scientific system:

```text
Circuit and query
  -> tensor-network lowering
  -> path and slicing
  -> ContractionDAG
  -> UPMEM numerical and parallel mapping
  -> one physical plan
  -> strict runtime
  -> validated quantum result
  -> raw evidence
  -> reproducible report
```

The project remains modular because the scientific decisions are separable and measurable. It becomes simple because historical architectures, speculative frameworks, tuple-wrapper types, milestone interfaces, and duplicate contracts are removed.

The strongest existing mechanisms—semantic identity, same-plan comparison, strict hardware admission, no fallback, explicit claim boundaries, credible baselines, and acceptance of negative results—must remain.

The architecture reset succeeds when an agent, reviewer, or thesis examiner can answer these questions without navigating historical code:

1. What quantum simulation problem is being solved?
2. What logical tensor contractions are performed?
3. How are they numerically represented and parallelized on UPMEM?
4. What data move between host, MRAM, and WRAM?
5. Which work executes on which tasklets and DPUs?
6. How is the result validated?
7. What exactly is timed and measured?
8. Which comparisons are scientifically compatible?
9. Which claims are supported, unsupported, or still hypotheses?

That is the required standard for the separate simplification branch.
