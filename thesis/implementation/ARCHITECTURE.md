# System Architecture

This is a small modular monolith for comparing tensor-network (TN) contraction
routes. The active TN path is data-first: module functions transform explicit
records, and state is confined to execution-owned maps, device sessions, and
artifact writers.

~~~text
CircuitSpec
  -> build_tensor_network_data
  -> TensorNetworkSpec + TensorInputs
  -> plan_contractions / PlannerResult
  -> build_contraction_dag / ContractionDAG
  -> compile_execution / ExecutionPlan
  -> execute / ExecutionResult
  -> normalized evidence
  -> reports
~~~

There is no ScientificPlan abstraction in the active pipeline. TensorNetworkSpec
describes the TN, PlannerResult records how a path was selected, and
ContractionDAG is the target-neutral semantic contraction graph.

## Boundaries

| Module | Input | Output | Parameters | Mutable state or side effects |
| --- | --- | --- | --- | --- |
| circuits/ | Suite case and circuit parameters | CircuitSpec | Family, qubit count, depth/repetitions, seed, gate parameters | None in the returned value; construction uses local lists. |
| tn/network.py | CircuitSpec | TensorNetworkSpec, TensorInputs | Gate tensor definitions and output wire order | Creates NumPy input arrays. They are input payloads and must be treated read-only after construction. |
| tn/planning.py | TensorNetworkSpec, PlannerRequest | PlannerResult | Engine, algorithm, objective, seed/repeats, UPMEM profile, normalization, numeric representation assumption | Planner-local search and external-library calls only; no device or file state. |
| tn/graph.py | TensorNetworkSpec, planner path | ContractionDAG | Pairwise path; optional SliceSpec | None. The DAG contains descriptors and dependencies, never tensor arrays or target estimates. |
| execution/compiler.py | ContractionDAG, CPU/UPMEM compile request | ExecutionPlan or UnsupportedExecution | Target, numeric mode, kernel/decomposition/placement/reduction IDs, logical topology, node order | None. No allocation, transfer, binary lookup, or device call. |
| execution/cpu.py | CPU ExecutionPlan, ContractionDAG, TensorInputs, RunContext | ExecutionResult | Warmups, repetitions, numeric mode | Local tensor map and output buffer for one call. Source arrays and DAG are not mutated. |
| execution/upmem.py | UPMEM ExecutionPlan, ContractionDAG, TensorInputs, RunContext with runtime resources | ExecutionResult; malformed/native/session failures raise | Rank bindings, binary paths, timeout, warmups/repetitions | UPMEM session, allocation, MRAM buffers, native subprocess/device state, and per-run tensor map. No fallback is allowed. |
| execution/runner.py | Compiled plan, DAG, inputs, context | Target-specific result/failure | Target dispatch | Only the state of the selected executor is mutable. GPU returns explicit unsupported in this slice. |
| bench/m5_circuit_study.py | Suite, circuit cases, planner/numeric/engine/topology variants | Normalized records and run artifacts | Route selection, repeats, warmups, tolerances, timeout | Study worklists, reference outputs, injected UPMEM sessions, and run-directory files. |
| bench/reporting.py and report scripts | Normalized records and manifests | CSVs, plots, report manifests | Filters, aggregation, output directory | Writes report artifacts; does not execute a route. |

The functional core is the first five rows. The stateful shell begins at
execution and includes only the resources required to run and record a route.
Dataclasses are used for explicit immutable boundaries; behavior is provided by
module-level functions rather than service objects or a plugin framework.

## What ContractionDAG Means

ContractionDAG is the semantic lowering of a selected pairwise contraction
path. ContractNode describes one tensor contraction, ReduceNode describes an
explicit reconstruction after global slicing, and TensorView describes a tensor
or fixed-index view. Dependencies are explicit and validation checks shapes,
labels, producers, ordering, and cycles.

This is not GEMM lowering. The graph says which mathematical contractions must
occur. Target compilation later decides whether a supported contraction is
implemented by a GEMM-like DPU kernel, NumPy einsum, tiling, or another target
kernel. The active DAG exposes dependency parallelism, but the current CPU and
M5 execution slice uses deterministic topological node order; it does not yet
claim concurrent independent-node scheduling.

The old core.records.TaskGraph contains legacy ContractionTask records. The
function materialize_task_graph_from_planner_result creates it from an already
computed PlannerResult for compatibility with historical identity/evidence
readers. It is not passed as the active executor contract, and the M5 study does
not use WholeGraphExecutor. The old whole-circuit classes remain as historical
compatibility/native-shell code.

## Planning, Slicing, and Identity

plan_contractions(network, request) dispatches to the selected planner adapter:

- opt_einsum and cotengra provide external path baselines;
- custom_upmem provides the deterministic modeled UPMEM greedy planner.

PlannerResult.identity records the planner engine, configuration, objective,
version, profile, normalization, and planning time. contraction_dag_hash(dag)
identifies only the resulting semantic graph. Planner identity and DAG identity
are therefore separate: two planners may produce the same DAG, and one planner
configuration may produce different DAGs for different inputs.

Global slicing is a semantic transformation. apply_slicing rewrites a
ContractionDAG into partial ContractNode values and a ReduceNode; it changes the
DAG hash because the mathematical execution graph changed. Target tiling is
different: it is a compilation/runtime decomposition of one DAG node under
device memory and kernel limits. It belongs in ExecutionPlan and runtime
metadata, not in the scientific DAG.

The execution-plan hash includes the selected target numeric mode, kernel,
decomposition, placement, reduction, logical topology, ABI/session/dispatch
profile, and node work order. Physical rank paths, host/DPU binary paths, and
machine-local session resources are excluded from that hash and recorded as
runtime/provenance facts instead. Changing a binary or hardware binding changes
run/executable evidence, not the mathematical DAG.

## Execution Families

| Family | Role | Current boundary |
| --- | --- | --- |
| QuEST CPU/GPU | Full-state baseline | Separate algorithm family; not an internal TN DAG route. |
| Quimb/cotengra CPU TN | External TN baseline | Comparison context; same-plan status requires an explicit adapter. |
| Functional CPU TN | Same-DAG reference | Implemented through compile_execution and run_cpu. |
| UPMEM SDK simulator | Protocol/layout validation | Implemented only as simulator evidence; never hardware speedup evidence. |
| Physical UPMEM TN | Bounded M5.5 route | Implemented through the UPMEM compile/execute adapter and strict native admission. |
| GPU TN target | Future/unsupported slice | compile_execution and execute return explicit unsupported results here. |

M5.5 remains the current active physical whole-circuit lane and retains its
existing status/evidence wording. It is bounded: raw SDK execution is used
behind the verified M5 v4 shell, graph-wide DPU residency and general DAG
scheduling are not silently implied, and PID-Comm/ATiM/SimplePIM provider
execution is not claimed unless a record proves that provider actually ran.
The current provider boundary is deliberately narrow. M5.5 uses a
SimplePIM-derived initialization/management binary, but allocation, transfers,
launch, and synchronization are performed through the raw UPMEM SDK and the
contraction kernel is thesis-owned. PID-Comm and ATiM are not invoked by M5.5;
they remain future provider adapters. A provider is credited only when a
normalized record identifies its actual invocation. The current route uses
host-managed graph intermediates and does not provide hardware-calibrated
planning.

## Evidence and Pairing

The M5 coordinator builds the separated network data, obtains one PlannerResult,
builds one ContractionDAG, materializes a legacy TaskGraph only where
compatibility fields require it, and sends the DAG through
compile_execution -> execute. CPU anchors and UPMEM route runs therefore share
the same TN inputs and DAG construction boundary.

Normalized evidence records retain planner configuration, planner hash, DAG
hash/schema, execution-plan hash/schema, target, numeric policy, timing scope,
validation, transfer accounting, and hardware admission fields. Report pairing
requires matching DAG identity whenever both rows provide it. A report must not
silently pair different semantic contraction graphs merely because circuit or
legacy plan fields look similar.

Only compatible evidence supports a comparison. Physical timing is not simulator
timing; modeled planner scores are not measured runtime; QuEST full-state and TN
routes are not same-plan speedups. Unsupported and failed rows remain visible,
and no executor may silently fall back to CPU or simulator execution.

## Current Claim Boundary

The active implementation supports a reproducible circuit-to-TN-to-DAG pipeline,
interchangeable planner adapters, functional CPU execution, and a bounded
physical UPMEM adapter with explicit validation and provenance. It does not yet
establish UPMEM acceleration, energy efficiency, arbitrary tensor-shape support,
general concurrent DAG scheduling, complete multi-DIMM scaling, PID-Comm
communication, ATiM-generated production kernels, or a hardware-calibrated
planner. Those are future components, not hidden capabilities of the current
route.

The benchmark matrix and claim rules remain authoritative in
[THESIS_BENCHMARK_MATRIX.md](THESIS_BENCHMARK_MATRIX.md). Milestone and evidence
status remains authoritative in [docs/MILESTONES.md](docs/MILESTONES.md).
