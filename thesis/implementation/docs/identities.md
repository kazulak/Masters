# Identity Contracts

These identifiers describe different objects. They must not be collapsed into
one hash or reused for a different scope.

## Identity Taxonomy

| Identity | Includes | Excludes |
|---|---|---|
| `problem_id` | Canonical circuit, canonical query parameters, gate parameters, and generator seed | Numeric representation, path, executor, machine |
| `tensor_network_structure_id` | Target-neutral tensor descriptors, labels, dimensions, connectivity, and requested output structure | Contraction order, slicing decisions, target details |
| `logical_plan_id` | Tensor-network structure, selected contraction order, slicing branches, reductions, and dependencies | Numeric policy, tiling, placement, topology, executable |
| `physical_plan_id` | Logical plan, numeric policy, stages, tiling, placement, topology, tasklets, intermediate policy, and kernel policy | Host timestamp, report formatting |
| `executable_id` | ABI, host/DPU binary hashes, compiler, flags, SDK, and required build-input hashes | Circuit data and report settings |
| `environment_id` | Host, OS, CPU, SDK, rank inventory, affinity, and environment facts | Invocation uniqueness |
| `experiment_id` | Configuration, route, warmups, repetitions, timeout, and validation policy | Individual invocation timestamp |
| `run_id` | Unique identifier for one actual invocation | Deterministic configuration identity |
| `validation_policy_id` | Reference dtype, tolerances, metrics, and fixture-specific bounds | Physical execution choice |
| `session_protocol_id` | Serialized ABI and protocol version | A particular opened session |
| `session_instance_id` | One opened runtime session for a case and route within a run | Protocol identity |
| `sample_id` | `run_id`, `case_id`, `route_id`, sample kind, and sample index | Other samples or sessions |

Reordering canonical JSON keys does not change an identity. Changing a path
changes `logical_plan_id`. Changing numeric policy, tiling, placement, or
tasklets changes `physical_plan_id`. Changing compiler flags changes
`executable_id`. Changing host affinity changes `environment_id`.

## Semantic Objects

`TensorNetwork` is a target-neutral semantic network. It owns tensor input
descriptors, indices, connectivity, and the requested result; numerical arrays
are separate execution inputs. It does not select an execution order.

`ContractionDAG` is the sole logical execution IR. It describes the selected
contraction path, explicit slicing branches, reductions, and dependencies. Its
hash is therefore a logical-plan identity, not a path-independent problem
identity.

UPMEM mapping produces `UpmemPlan` from the DAG. It must not mutate the DAG or
silently introduce slicing.

## Tensor-Network Structure Hash Payload

`tensor_network_structure_id` version 1 hashes this JSON payload and no tensor
values:

```json
{
  "schema_version": "tensor_network_structure_v1",
  "tensors": [
    {
      "id": "...",
      "labels": [0],
      "shape": [2],
      "structure": "dense",
      "dtype": "complex128",
      "produced_by": null
    }
  ],
  "output_labels": [0, 1],
  "einsum_expression": "..."
}
```

The `tensors` list is sorted by the canonical JSON representation of each
descriptor. The payload is serialized with sorted object keys, compact
separators `(',', ':')`, and `ensure_ascii=True`, then hashed with SHA-256.
Input array contents, numeric policy, contraction path, slicing, placement,
and runtime details are excluded.

## Public-Type Allowlist

The reset allows these public types only:

```text
model.py:
  GateOperation, Circuit, SimulationQuery, SimulationJob,
  TensorSpec, TensorView, TensorNetwork,
  ContractNode, ReduceNode, ContractionDAG

  results.py:
  JsonScalar, JsonValue, Measurement, ExecutionSample,
  UnsupportedExecution, ExecutionFailed

numerics.py:
  NumericPolicy, EncodedComplexTensor

  upmem/plan.py:
  UpmemTopology, UpmemResources, UpmemWorkUnit,
  UpmemStage, UpmemPlan

upmem/runtime.py:
  UpmemSession
```

Private module-local helpers are allowed. A new public type requires an
explicit contract update before implementation.
