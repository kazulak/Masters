# Identity Contract

Identities are deterministic hashes unless explicitly stated otherwise. They
are serialized from canonical JSON with sorted keys, compact separators, and
SHA-256. Reordering keys does not change an identity.

## Domains

| Identity | Includes | Excludes |
|---|---|---|
| `problem_id` | Circuit semantics, query, parameters, generator seed | Numeric mode, path, executor, machine |
| `tensor_network_structure_id` | Tensor descriptors, labels, shapes, connectivity, output structure | Path, slicing, target details, tensor values |
| `logical_plan_id` | TN structure, contraction order, explicit slice branches, reductions, dependencies | Numeric mode, tiling, placement, topology, executable |
| `physical_plan_id` | Logical plan, numeric policy, stages, tiles, placement, topology, tasklets, intermediate/kernel policy | Timestamp, report formatting |
| `executable_id` | ABI, host/DPU binary hashes, compiler, flags, SDK and build-input hashes | Circuit, report settings |
| `environment_id` | Host/OS/CPU, SDK, rank inventory, affinity and environment facts | Invocation uniqueness |
| `experiment_id` | Configuration, routes, warmups, repetitions, timeout, validation policy | One invocation timestamp |
| `run_id` | Fresh UUID4 for one invocation | Deterministic configuration identity |
| `validation_policy_id` | Reference dtype, metrics, tolerances, fixture bounds | Execution choice |
| `session_protocol_id` | ABI and serialized protocol version | Open-session instance |
| `session_instance_id` | One opened runtime session within a run | Protocol identity |
| `sample_id` | Run, case, plan, route, sample kind and index | Other samples |

`problem_id` describes the simulated problem, not its representation. Numeric
precision is an intervention and is therefore outside `problem_id`. The
structure ID hashes descriptors, not input array contents. `ContractionDAG`
hashes are logical-plan identities because the DAG contains a selected path.

## Required Invariants

Changing a path changes `logical_plan_id`. Changing numeric policy, tiling,
placement, topology, tasklet count, or kernel/intermediate policy changes
`physical_plan_id` but not the logical plan. Changing compiler flags or ABI
changes `executable_id`. Changing host affinity changes `environment_id`.
Changing report styling changes none of these execution identities.

`sample_id` is derived from `run_id`, `case_id`, `plan_id`, `route_id`,
`sample_kind`, and `sample_index`; it is not a replacement for the unique
`run_id`.

## Structure Payload

Version 1 of `tensor_network_structure_id` hashes the tensor descriptor list,
output labels, and einsum expression. Descriptors include tensor ID, labels,
shape, structure, dtype, and producer. Descriptors are canonically sorted.
Input values, path order, slicing, numeric policy, placement, and runtime facts
are excluded.

## Comparison Meaning

`same problem` means matching `problem_id`. `same TN structure` means matching
`tensor_network_structure_id`. `same logical plan` requires matching structure
and `logical_plan_id`. `same physical route` additionally requires matching
`physical_plan_id`, executable facts, environment requirements, timing scope,
and validation policy appropriate to the comparison. A report must not label
two rows “same plan” from a circuit name alone.

## Evidence Binding

The manifest stores `run_id`, `experiment_id`, `environment_id`, and
`validation_policy_id`. Each sample stores the problem, TN, logical, physical,
and executable identities as applicable. Sessions repeat run, case, plan,
route, protocol, and instance identity. Canonical verification recomputes
sample IDs and rejects mismatched or missing linked identities.
