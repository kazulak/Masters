# Pipeline Contract

This document defines the active route as small composable stages. It names the
actual Python types so a stage can later become a separate process or container
without guessing what it consumes, owns, or mutates.

```text
CircuitSpec -> TensorNetworkValue -> TaskGraph -> M5 route executor
            -> normalized_records.jsonl -> report files
```

## Stage Contract

| Stage | Main symbol | Input | Output | Parameters | Owned mutable state / side effect | Identity |
| --- | --- | --- | --- | --- | --- | --- |
| Circuit | circuit factory -> `CircuitSpec` | suite case | immutable gate operations and source metadata | family, qubits, seed/depth settings | none | circuit semantics later hashed by the TaskGraph bundle |
| TN lowering | `build_tensor_network(circuit)` | `CircuitSpec` | `TensorNetworkValue` | no executor settings | creates `TensorValue.array` arrays | tensor-network hash includes specs and circuit identity |
| Planning | `plan_task_graph_with_config(network, planner_config)` | TN and planner config | immutable `TaskGraph` | planner engine/objective/options | planner-local search only | plan hash captures ordered path and lowered tasks |
| Route declaration | `PipelineParameters`, `ModuleSpec`, `PipelineRoute` | JSON-safe module settings | frozen route specification | module implementation and parameters | none | module config hashes and `route_config_hash` |
| Route selection | M5 study route expansion | suite planner/numeric/engine/topology choices | `PipelineRoute` | suite configuration | none | `route_config_hash` records requested composition |
| Execution | `m5_circuit_study.run_study()` | circuit, TaskGraph, selected `PipelineRoute` | executor result and normalized record | fixed verified executor profile | executor-owned tensor store, session, device state | plan identity plus requested and observed executor identities |
| Evidence | benchmark runner | execution result and environment | normalized record/manifests | suite/repeats/admission settings | writes an ignored `runs/evidence/...` directory | record stores scientific and executor identities |
| Report | `m5_circuit_report.generate_report()` | normalized records | CSVs, plots, manifests | report filters and output path | writes `runs/comparisons/...` | preserves source row identity and report manifest |

## Immutable Specifications

### `PipelineParameters`

`PipelineParameters(mapping)` accepts a JSON-safe mapping, canonicalizes it,
and stores only canonical JSON. `to_dict()` returns a fresh dictionary. It
rejects non-finite values and unsupported mutable/object types. Its `hash` is
the SHA-256 of canonical JSON.

### `ModuleSpec`

`ModuleSpec(role, implementation, parameters)` is frozen. Its `config_hash`
hashes exactly those three values. It is descriptive: an executor must consume
a module explicitly before the module has runtime effect.

### `PipelineRoute`

`PipelineRoute(route_id, label, modules)` is frozen and has one module for each
required role:

```text
tensor_network, planner, numeric, executor, topology
```

It may also contain:

```text
kernel, partitioner, scheduler, communication
```

Roles are unique and the `route_config_hash` hashes the ordered module map, not
the display name. The route is selected before preparation. Its
`tensor_network` and `planner` modules create the `TaskGraph`; changing a later
numeric, topology, kernel, partitioning, scheduling, or communication module
must not alter that chosen graph.

### `ComparisonSpec`

`ComparisonSpec(baseline_route, candidate_route, changed_roles, label)` is a
frozen declaration of a controlled ablation. It validates that `changed_roles`
exactly matches the two route specifications. It does not execute or interpret
the comparison. M5 records applicable comparison IDs as provenance; the
current report independently enforces compatible plan, timing, validation,
and hardware-admission fields.

## Current M5 Executor Profile

There is one public execution pipeline: the M5 study builds the tensor network
and TaskGraph, then runs the selected route through its CPU or UPMEM executor.
`PipelineRoute` is the immutable request and evidence contract; it is not a
second generic executor API.

Today `kernel`, `partitioner`, `scheduler`, and `communication` are fixed,
verified declarations of an executor profile. M5 admits a physical result only
when observed native metadata agrees with the declared profile. They are not
independently dispatchable modules yet. A future engine may make one selectable
only when it consumes the selection in execution and records the observation.

## Mutable State

Only execution and artifact creation need mutation.

| Mutable object | Defined in | Owner and lifetime | What changes |
| --- | --- | --- | --- |
| `TensorNetworkValue.tensors` / `TensorValue.array` | `tn/network.py`, `core/records.py` | TN lowering owns creation; M5 executor treats source values as inputs | Arrays exist as numerical source values; execution-owned stores and device buffers must not mutate the selected graph. |
| `InMemoryTensorStore` | `whole_circuit/core.py` | one `WholeGraphExecutor.execute()` call | live values, remaining-use counts, released tensor IDs. |
| `WholeGraphExecutor` local worklist | `whole_circuit/core.py` | one execution | pending and completed task IDs, task records. |
| `TaskExecutionSession` | selected engine | open-to-close session | engine-specific native/device/session state. |
| Physical UPMEM runtime | target-specific executor | one physical execution/session | SDK allocation, MRAM slots, transfer buffers, subprocess/native binary state. |
| Evidence/report directories | benchmark/report code | one command | JSONL, manifests, CSVs, PNGs, logs. |

`CircuitSpec`, `TaskGraph`, `ModuleSpec`, `PipelineRoute`, and `ComparisonSpec`
are frozen dataclasses. Frozen does not make nested third-party values
intrinsically immutable; the contract above identifies the owner of each
mutable array or runtime resource.

## Planning And Execution Identity

`tn/execution_bundle.py` provides the scientific identity chain:

```text
circuit_semantics_hash
  -> tensor_network_hash
  -> contraction_plan_hash
```

`contraction_path_structure_hash` identifies the ordered pairwise path and
task structure. `executor_config_hash()` identifies executor settings without
changing the plan hash. `route_config_hash` identifies the declared module
composition requested by the experiment; observed binary and hardware
provenance are recorded separately. A valid same-plan comparison normally needs matching circuit,
network, plan, and path identities, while executor/route identities are
expected to differ for the explicitly studied module.

## Container Or Process Boundary

The least invasive future boundary is a stage request/response with canonical
JSON metadata and binary arrays:

```text
execute request:  CircuitSpec + planner config + PipelineRoute + source tensors
execute response: TaskGraph identity + output + task/session metadata + record fields
report request:   normalized_records.jsonl path
report response:  CSV/PNG paths + report manifest
```

The current code remains in-process because it is a small research repository.
The contract isolates I/O and mutable ownership without introducing a container
or RPC framework before it is useful.
