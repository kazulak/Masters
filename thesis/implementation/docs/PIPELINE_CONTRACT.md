# Pipeline Contract

This document describes the active tensor-network path as small, explicit data
transformations. It deliberately does not introduce a ScientificPlan layer.

~~~text
CircuitSpec
  -> build_tensor_network_data
  -> TensorNetworkSpec + Mapping[str, ndarray]
  -> plan_contractions / PlannerResult
  -> build_contraction_dag / ContractionDAG
  -> compile_execution / ExecutionPlan
  -> execute / ExecutionResult
  -> normalized_records.jsonl
  -> reports
~~~

## Stage Contracts

| Stage and symbol | Input | Output | Parameters | Mutable state/side effects |
| --- | --- | --- | --- | --- |
| Circuit factories in circuits/ | Suite case/config | CircuitSpec | Family, qubits, depth/repetitions, seed, gate parameters | Local construction only; returned operations are immutable. |
| build_tensor_network_data in tn/network.py | CircuitSpec | (TensorNetworkSpec, Mapping[str, ndarray]) | Gate tensor definitions and output ordering | Allocates a plain tensor-id lookup and NumPy arrays. Callers and executors treat the arrays as read-only. |
| plan_contractions in tn/planning.py | TensorNetworkSpec, PlannerRequest or config mapping | PlannerResult | Engine, algorithm, objective, seed/repeats, UPMEM profile, normalization, representation assumption | Planner-local memory and external planner state only; no device I/O. |
| build_contraction_dag in tn/graph.py | TensorNetworkSpec, PlannerResult.path | ContractionDAG | Pairwise dynamic active-list path | None; no arrays are inspected. |
| apply_slicing in tn/graph.py | ContractionDAG, SliceSpec | New ContractionDAG | One supported global contracted-label slice | None; original DAG is not mutated. |
| compile_execution in execution/compiler.py | ContractionDAG, CpuCompileRequest or UpmemCompileRequest | ExecutionPlan or UnsupportedExecution | Target, numeric mode, logical topology, kernel/decomposition/placement/reduction and profile IDs | None; it does not allocate devices, access binaries, or transfer data. |
| execute in execution/runner.py | ExecutionPlan, DAG, Mapping[str, ndarray], RunContext | ExecutionResult or deterministic dispatch ExecutionFailure | Run ID, target, warmups/repetitions, timeout, target runtime resources | Dispatches exactly one target. Malformed inputs and native/session failures raise unchanged so experiment orchestration can retain their failure stage. |
| run_cpu in execution/cpu.py | CPU plan, DAG, inputs, context | ExecutionResult | Numeric mode, node order, repetitions | Fresh local tensor map and output buffer per call; no source mutation. |
| run_upmem in execution/upmem.py | UPMEM plan, DAG, inputs, context with UpmemRuntimeResources | ExecutionResult; malformed/native/session failures raise | Logical topology in plan; rank paths/binaries/session opener in runtime resources; timeout and repetitions | UPMEM allocation/session, MRAM buffers, subprocess/device state, and local graph values. No CPU/simulator fallback. |
| m5_circuit_study.run_study | Suite config and injected engine factories | Run directory and normalized records | Route variants, tolerances, warmups/repeats, timeout | Study worklist, reference arrays, injected sessions, and files under ignored runs/. |
| Report generator | Normalized records and manifests | CSVs, plots, report manifest | Selection and aggregation settings | Writes report artifacts only; never executes a route. |

The current execution/ package implements CPU and UPMEM dispatch. GPU is an
explicit unsupported target in this slice, not a CPU fallback.

## Data Records

TensorNetworkSpec contains circuit identity, tensor descriptors, output labels,
and the einsum expression. A plain mapping contains tensor-id-to-array values
with exact ID, shape, and dtype validation. The separation allows a planner to
operate on metadata while an executor receives numerical payloads without a
wrapper type.

PlannerResult contains the selected pairwise path, planner identity, path
summary, planning time, and planner metadata. It is planning output, not an
execution plan. PlannerResult.identity includes the planner engine and full
resolved configuration hash.

ContractionDAG contains TensorView, ContractNode, ReduceNode, and the final
output view. It contains labels, shapes, producers, and dependencies but no
arrays, planner configuration, target estimates, rank paths, or binaries.

The legacy core.records.TaskGraph contains ContractionTask records. The
compatibility function materialize_task_graph_from_planner_result creates it
from the same already-selected PlannerResult; it must not re-plan. Existing
legacy identity/evidence readers can use it, but active execution consumes the
ContractionDAG. WholeGraphExecutor and old whole-circuit classes remain
historical compatibility/native-shell code; the active M5 study does not use
WholeGraphExecutor.

## Slicing Versus Tiling

Global slicing changes the semantic calculation. apply_slicing fixes one
contracted label, creates partial DAG contract nodes, and adds an explicit
reduction node. It therefore changes the ContractionDAG and its hash.

Target tiling changes how one DAG contract is represented for a target memory
hierarchy. UPMEM tile sizes, buffers, DPU assignment, and tile counts are
execution/compiler concerns. They belong in ExecutionPlan work plans and
runtime evidence; they do not change the DAG hash unless the mathematical graph
itself is changed.

## Identity Rules

The active identities are intentionally separate:

~~~text
PlannerResult.identity.planner_config_hash  = how the path was selected
contraction_dag_hash(dag)                   = what contractions the graph means
execution_plan_hash(plan)                   = how a target will execute that graph
backend/executable facts                    = what binary and machine actually ran
run/evidence identity                       = when and where it ran
~~~

Changing a planner configuration can change the planner identity without
necessarily changing the DAG. Changing the path changes the DAG hash. Changing
numeric mode, kernel, decomposition, placement, reduction, logical topology, or
ABI/session/dispatch profile changes the execution plan. Physical rank paths,
host/DPU binary paths, and local session resources are excluded from the plan
hash and are recorded as runtime provenance instead.

Report pairing requires compatible circuit/network/plan and route evidence. When
both rows have DAG v2 fields, the DAG hash and schema must match. A legacy row
without DAG identity cannot be silently treated as same-DAG evidence.

## Runtime Ownership

The functional core does not mutate its input records. Mutable state is limited
to:

- NumPy arrays created by TN lowering and passed as read-only source payloads;
- per-call CPU tensor maps and output buffers;
- UPMEM sessions, allocations, MRAM/transfer buffers, and native subprocesses;
- M5 study worklists and reference outputs;
- JSONL, manifest, CSV, and plot files created by evidence/report stages.

RunContext.target_resources carries machine-local UPMEM paths and the session
opener. Those resources are required at runtime but are excluded from the
logical ExecutionPlan identity.

For the M5 v4 physical route, `READY` and `RESPONSE` carry the native host's
compiled backend/profile/ABI/session/dispatch/kernel/execution-class contract.
The M5 terminal record derives those fields only from agreeing native events
across ranks and requires `native_identity_verified=true`. The coordinator
records `graph_intermediate_placement=host_managed` with origin
`m5_host_coordinator_v1`: it rehydrates DAG intermediates on the host between
contractions. `executed_node_ids` records completed host DAG nodes only; it is
not native-kernel exactly-once evidence.

## Functional Process Boundary

The current implementation is in-process, but the contracts are suitable for a
later process/container boundary:

~~~text
request:  CircuitSpec/config + planner request + execution request + inputs
response: PlannerResult + ContractionDAG + ExecutionPlan + result/failure + facts
report:   normalized_records.jsonl -> tables/plots/manifest
~~~

No RPC or container framework is part of the active slice. The important rule is
that each function has explicit records in and out, while device/session state
stays in the execution shell.
