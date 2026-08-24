# Pipeline Contract

The active tensor-network path is a sequence of direct functions. It has one
logical execution IR, `ContractionDAG`, and one UPMEM physical-plan schema,
`UpmemPlan` schema version 1.

```text
SimulationJob
  -> lower_tensor_network
  -> TensorNetwork + Mapping[str, ndarray]
  -> plan_opt_einsum | plan_cotengra
  -> pairwise contraction path
  -> build_contraction_dag
  -> ContractionDAG
  -> run_cpu_once
       or
     plan_upmem -> UpmemPlan -> open_upmem -> UpmemSession.run_once
  -> ExecutionSample
  -> experiment evidence
  -> report
```

There is no `ScientificPlan`, provider registry, or generic target dispatcher
in this canonical path.

## Boundaries

| Function | Input | Output | Side effects |
| --- | --- | --- | --- |
| `lower_tensor_network` | `SimulationJob` | `TensorNetwork`, tensor-value mapping | Allocates NumPy arrays only |
| `plan_opt_einsum` / `plan_cotengra` | `TensorNetwork` and planner parameters | Pairwise path and planner facts | Planner-local work only |
| `build_contraction_dag` | `TensorNetwork` and path | `ContractionDAG` | None |
| `run_cpu_once` | DAG, inputs, `NumericPolicy` | `ExecutionSample` | CPU-local arrays only |
| `plan_upmem` | DAG, `NumericPolicy`, `UpmemTopology` | `UpmemPlan` or `UnsupportedExecution` | None; no device access |
| `replay_upmem_plan_once` | DAG, physical plan, inputs | `ExecutionSample` | CPU-local policy replay only |
| `open_upmem` | DAG, physical plan, resources and timeout | `UpmemSession` | Opens physical native sessions |
| `UpmemSession.run_once` | Tensor-value mapping | `ExecutionSample` | Transfers, kernels, host reconstruction |
| `UpmemSession.close` | Open session | Terminal facts | Releases physical resources |

`TensorNetwork` describes target-neutral tensors and connectivity but no
execution order. `ContractionDAG` is the sole logical execution IR: it owns the
selected contraction order, explicit slicing branches, reductions, and
dependencies. `UpmemPlan` records only physical choices such as numeric policy,
tiling, placement, topology, tasklets, stages, and the current
`host_roundtrip_v1` intermediate policy.

## Numerical Boundary

The execution policies are:

- `split_complex_float32_v1`;
- `split_complex_int8_shared_scale_v1`.

Complex contraction uses four real products in fixed `rr`, `ii`, `ri`, `ir`
order. The int8 policy uses one shared scale for the real and imaginary planes
of each operand. Complex128 is a validation reference, not an execution mode.
The same-physical-plan CPU replay is the policy oracle for UPMEM execution.

## Parallel Boundary

The current mapper assigns output and K tiles to deterministic rank/DPU waves.
The coordinator can submit different ranks concurrently. Graph intermediates
remain host-managed, reductions remain on the host, and each current contract
stage contains one logical contraction. Multi-label logical slicing and
slice-batch stages are T8/T9 work; they are not implemented merely because the
schema can represent more than one node.

## Timing Boundary

Executors perform one run. Warmups, repetitions, validation, hashing, and
evidence writing belong to the experiment layer. `Measurement.total_wall_s`
is authoritative. Independently reported rank durations are work diagnostics,
not inferred wall time. Exact timing scopes are defined in `docs/timing.md`.

## Identity Boundary

The identities are independent:

```text
problem_id
tensor_network_structure_id
logical_plan_id
physical_plan_id
executable_id
environment_id
experiment_id
run_id
analysis identity
```

Changing the contraction path changes `logical_plan_id`. Changing numeric
policy, tiling, placement, topology, or tasklets changes `physical_plan_id`.
Changing native binaries or compiler flags changes `executable_id`. Exact
definitions are in `docs/identities.md`.

## Mutable State

Mutable state is restricted to:

- NumPy arrays created for one lowering or execution;
- UPMEM subprocesses, allocations, rank sessions, transfers, and native
  buffers inside `UpmemSession`;
- experiment-owned manifest and JSONL files;
- report artifacts created from normalized evidence.

The DAG and physical-plan records are immutable. The runtime must not silently
re-plan, fall back to CPU, or accept simulator execution as physical evidence.

## Historical Compatibility

`quantum_bench.execution.compiler` and `quantum_bench.execution.runner` retain
the old `ExecutionPlan`/`RunContext` public commands until T12 deletes their
callers. They adapt old numeric, topology, and node-plan records at that
historical boundary. Canonical `quantum_bench.upmem.plan` and
`quantum_bench.upmem.runtime` neither import nor expose those legacy records.

Historical milestone runners, provider registries, TaskGraph code, and old
native lanes are not architectural extension points. They remain only to keep
the branch testable while their direct replacements are installed and are
deleted in the same ownership batches as those replacements.
