# System Architecture

This is a compact research architecture for evaluating tensor-network quantum
simulation routes. The system is deliberately split into an immutable planning
path and a stateful execution path so that one circuit and one contraction plan
can be evaluated by several engines without implying that those engines are
algorithmically identical.

```text
CircuitSpec
  -> TensorNetworkValue
  -> PipelineRoute planner role -> TaskGraph
  -> remaining PipelineRoute execution roles
  -> execution records
  -> normalized_records.jsonl
  -> report CSVs, plots, and manifests
```

## Ownership At A Glance

| Stage | Main symbols/files | Input | Output | State and side effects |
| --- | --- | --- | --- | --- |
| Circuit definition | `circuits/`, `CircuitSpec` | suite case/configuration | immutable operation sequence | Pure construction. |
| TN lowering | `tn/network.py:build_tensor_network` | `CircuitSpec` | `TensorNetworkValue` | Creates tensor arrays; source arrays are read-only in the generic pipeline. |
| Path planning | `tn/planners.py`, `tn/task_graph.py` | TN and planner config | immutable `TaskGraph` | Planner search is local computation; no hardware side effect. |
| Plan identity | `tn/execution_bundle.py` | graph and route-independent plan data | semantic/TN/plan hashes | Pure canonical serialization and hashing. |
| Route composition | `whole_circuit/pipeline.py` | module declarations | immutable `PipelineRoute` / `ComparisonSpec` | Pure JSON-safe specification and hashing only. |
| Execution | `bench/m5_circuit_study.py`, `whole_circuit/core.py`, target engines | circuit, graph, selected route | output plus task/session metadata | M5 is the single public execution pipeline; owns tensor store, sessions, subprocesses, device allocation, and transfers. |
| Evidence | `bench/*`, `core/records.py` | execution metadata | normalized records/manifests | Writes run directories. |
| Reporting | `bench/m5_circuit_report.py`, `scripts/` | normalized records | CSVs, plots, report manifest | Writes comparison artifacts only. |

The detailed symbol-level contract is in
[docs/PIPELINE_CONTRACT.md](docs/PIPELINE_CONTRACT.md).

## Immutable Planning Path

`CircuitSpec` is the semantic input. `build_tensor_network()` converts it into
gate tensors, index labels, output order, and a full einsum expression.
`plan_task_graph_with_config()` chooses a pairwise contraction path and lowers
it into a `TaskGraph` whose `ContractionTask` items carry dependencies, shapes,
index expressions, and estimates.

The identity boundary is intentionally before execution:

```text
circuit_semantics_hash
  -> tensor_network_hash
  -> contraction_plan_hash
```

`contraction_plan_hash` does not include executor-specific settings. Therefore
CPU float32, UPMEM float32, and UPMEM int8 can prove that they consumed one
selected plan while retaining distinct executor and route identities. A change
of planner or path produces a different plan identity.

## Route Modules

`PipelineRoute` is a frozen declaration of the complete composition. It is
selected before preparation: its lowering and planner roles create the graph,
then its remaining roles execute that graph. Every route contains these
required `ModuleSpec` roles:

| Role | Meaning | Current examples |
| --- | --- | --- |
| `tensor_network` | lowering implementation | `quantum_gate_tn_v1` |
| `planner` | selected path algorithm and parameters | `opt_einsum`, cotengra, custom modeled planner |
| `numeric` | numeric representation/policy | float32, host-packed int8 |
| `executor` | engine that consumes tasks | NumPy reference, UPMEM v4 |
| `topology` | target placement | CPU device or explicit UPMEM ranks/DPUs |

For M5, these are fixed verified declarations of the executor profile, not
independently dispatchable implementations. The physical admission check
compares the declared profile with observed native metadata. A later engine may
make a role selectable only when it executes that selection:

| Optional role | Intended responsibility | Current boundary |
| --- | --- | --- |
| `kernel` | contraction/permutation kernel profile | fixed by the current UPMEM executor profile. |
| `partitioner` | task-to-device partition profile | fixed by the current physical route. |
| `scheduler` | ready-work and device-wave profile | fixed by the current physical route. |
| `communication` | transfer/reduce/broadcast profile | host-managed today; no PID-Comm claim unless execution records real use. |

This makes ablations explicit: a valid `ComparisonSpec` states the two routes
and the exact changed roles. Numeric choice, topology, and kernel selection
must not modify the already selected `TaskGraph`.

The M5 route adapter is deliberately strict rather than general: it accepts
only the built-in verified executor profiles. Physical admission compares the
requested profile, ABI, session, dispatch mode, kernel, execution class, and
intermediate placement with native metadata. A mismatch is a failed row, not a
silent fallback.

## Stateful Execution Path

Execution starts only after a graph and route have been prepared. The primary
replaceable boundary is `TaskExecutionEngine.open_session(policy, topology)`.
`WholeGraphExecutor` creates a fresh `TensorStore`, consumes input lifetimes,
and invokes a session once per ready task.

| Object | Owner | Mutable state |
| --- | --- | --- |
| `InMemoryTensorStore` | one `WholeGraphExecutor` run | live arrays, remaining uses, released IDs |
| `TaskExecutionSession` | selected engine | device/session state, native process and timing state |
| physical UPMEM engine | target module | SDK allocation, MRAM slots, transfer buffers, native subprocesses |
| reporter | report command | output directories, tables, images, manifests |

No prepared graph, route specification, or hash is mutated by normal
execution. Output arrays and evidence files are new owned artifacts.

## Execution Families

| Family | Purpose | Same-plan status |
| --- | --- | --- |
| QuEST CPU/GPU full state | External serious full-state baseline | Same algorithm family, not an internal TaskGraph plan. |
| Quimb/cotengra CPU TN | External serious TN baseline | TN comparison context; path identity is distinct unless an adapter proves otherwise. |
| NumPy TaskGraph | Internal CPU reference | Same-plan reference for internal routes. |
| UPMEM SDK simulator | Layout, protocol, and boundary validation | Not physical performance evidence. |
| Physical UPMEM M5.5 | Bounded same-plan whole-circuit execution | Hardware evidence only when its admission fields pass. |

SimplePIM, PID-Comm, ATiM, and SparseP are external components behind
thesis-owned route boundaries. They are not credited as active compute,
communication, or kernel providers unless a normalized record identifies the
actual provider invocation. The raw UPMEM SDK remains a valid explicit fallback
engine where the route says so; it is not silently substituted.

## Current M5.5 Boundary

M5.5 is the current active physical whole-circuit lane. It executes a selected
TaskGraph with a NumPy CPU reference or a bounded UPMEM v4 route. The physical
route uses explicit rank paths, a persistent bounded session, bulk request
launches, float32 or host-packed int8 task inputs, and host-managed graph
intermediates. It records hardware allocation, native execution, transfer,
validation, timing scope, and no-fallback state.

It does not yet provide graph-wide DPU-resident intermediates, general DAG
scheduling, PID-Comm collectives, ATiM-generated kernels, multi-DIMM scaling,
energy data, or hardware-calibrated planning. These are future route modules,
not hidden behavior of the current executor.

## Evidence And Claims

Every execution writes structured data first. `normalized_records.jsonl` is
the reporting input; plots and tables are derived artifacts. A report can
compare only rows whose recorded identities and admission fields are compatible.
Unsupported and failed rows stay visible. Simulator, modeled, cross-algorithm,
and physical records remain labelled as different evidence classes.

`route_config_hash` identifies the requested module composition. It does not
claim which binary ran; executor, host/DPU binary, and observed hardware
provenance remain separate evidence fields.

The benchmark choices and allowed comparisons are specified in
[THESIS_BENCHMARK_MATRIX.md](THESIS_BENCHMARK_MATRIX.md). The active physical
procedure is [docs/upmem_m5_5_whole_circuit_runbook.md](docs/upmem_m5_5_whole_circuit_runbook.md).
