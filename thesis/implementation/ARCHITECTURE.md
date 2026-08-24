# Architecture

The implementation is a modular monolith with a functional core and a small
stateful execution shell. It benchmarks a fixed quantum-simulation problem
through comparable TN and full-state routes.

```text
SimulationJob
  -> TensorNetwork
  -> path planner
  -> ContractionDAG
  -> NumPy / Quimb / cotengra / QuEST
     or UpmemPlan -> ABI-v4 runtime
  -> canonical evidence -> report
```

## Core Concepts

`SimulationJob` defines one requested quantum-simulation problem: circuit,
query, parameters, seed, and canonical output ordering.

`TensorNetwork` is a target-neutral semantic network produced by circuit
lowering. It describes tensor descriptors, labels, connectivity, input names,
and requested output. It is not executable and has no contraction order,
slicing, dependencies, target estimates, timing, arrays, or hardware state.

`ContractionDAG` is the sole logical execution IR. A path planner selects an
order, then lowering creates binary contraction nodes, explicit dependencies,
local slice branches, and reduction nodes. Its identity is the logical-plan
identity. The DAG is not GEMM lowering: tiles, layouts, kernels, and GEMM-like
canonicalization belong to the UPMEM mapping.

`UpmemPlan` is a physical plan derived from a DAG. It records numeric policy,
topology, ordered `contract_batch` and `host_reduce` stages, real-tile work
units, and kernel policy. The current intermediate policy is host round-trip.

## Ownership Boundaries

| Module | Input | Output | Responsibility |
|---|---|---|---|
| `model.py` | values | immutable records | circuit/TN/DAG data and invariants |
| `circuits.py` | job description | circuit | supported circuit construction |
| `lowering.py` | job, network, path | network, inputs, DAG | circuit-to-TN and path-to-DAG lowering |
| `planning.py` | network structure | path, provenance | opt_einsum/cotengra planning only |
| `numerics.py` | complex arrays, policy | encoded/decoded arrays, facts | split-complex float32/int8 policy |
| `results.py` | output and observations | immutable execution sample | target-neutral measurement/failure contract |
| `cpu.py` | DAG or UPMEM plan, inputs | execution sample | same-DAG execution and plan replay |
| `baselines.py` | job | execution sample | direct Quimb/cotengra/QuEST adapters |
| `upmem/plan.py` | DAG, policy, topology | `UpmemPlan` | deterministic physical mapping |
| `upmem/runtime.py` | plan, inputs | execution sample | session, transfer, launch, collection |
| `experiment.py` | config and routes | evidence artifacts | warmups, repetitions, references |
| `evidence.py` | raw facts | canonical records | identity, validation, integrity |
| `report.py` | canonical evidence | tables and figures | analysis only |
| `cli.py` | command line | command result | plan/run/report/verify/qualify |

The core modules do not import runtime, subprocess, filesystem, or reporting
code. Planning does not open devices. Mapping does not allocate devices. The
runtime does not search paths or silently replan. Reporting never runs a
benchmark.

## Numeric and Parallel Execution

The active numeric policies are `split_complex_float32_v1` and
`split_complex_int8_shared_scale_v1`. Complex values use separate real and
imaginary planes. Int8 uses one shared scale per complex operand, packed host
inputs, int32 tile accumulation, and deterministic host decoding/reduction.

ABI-v4 is one real-valued output-tile contraction ABI. A complex contraction
launches `rr`, `ii`, `ri`, and `ir` sequentially on the assigned DPU set, then
forms `(rr - ii) + i(ri + ir)`. The CPU physical-plan replay is a correctness
oracle for this exact tile and accumulation order, not a performance baseline.

Current physical mapping partitions output/K tiles across assigned DPUs. It
does not yet implement resident DAG intermediates, execution of slice groups
in parallel, tasklet scheduling policy, active PID-Comm communication, or a
hardware-calibrated path score.

## Identity and Evidence

The evidence layer records distinct identities:

```text
problem_id                    circuit, query, parameters, seed
tensor_network_structure_id   network structure only
logical_plan_id               path, slicing, reductions, dependencies
physical_plan_id              numeric policy, tiles, placement, topology, stages
executable_id                 ABI, binary hashes, compiler/SDK facts
environment_id                host and hardware environment
experiment_id                 configuration and repetition policy
run_id                        one invocation
session_instance_id           one opened runtime session
sample_id                     one warmup or measurement attempt
```

An execution sample contains its output, authoritative wall time, nullable
component measurements, backend facts, and numeric facts. Concurrent rank
work is recorded as work, not inferred wall time. Missing measurements are
null, not zero.

`simulation_end_to_end_v1` starts immediately before route-specific
preparation of an already-created job and stops at decoded output. TN routes
include lowering, planning, slicing, DAG construction, mapping, and execution.
QuEST routes include circuit translation, state setup, execution, and query
extraction. `steady_execution_v1` measures a reusable open execution context
from input encoding to decoded output.

Physical routes fail closed: no simulator or CPU fallback can satisfy a
physical request. Simulator evidence is never admitted as physical timing,
speedup, scaling, or energy evidence.

## Native Boundary

`native/upmem/runtime/` is the only active UPMEM native build. It owns the
ABI-v4 host program, DPU program, protocol headers, and build rules. Python
serializes plan work units through `upmem/protocol.py` and manages native
process lifecycle through `upmem/native_session.py`.

The native runtime uses pinned SimplePIM management types and its initialization
kernel around raw-SDK allocation and dispatch. This does not make SimplePIM an
active high-level scheduler or compute runtime. The retained PID-Comm harness
is a separate compatibility check, not a runtime provider. ATiM is not
integrated.

## Current Limits

The reset architecture is software- and SDK-simulator-validated. Its physical
UPMEM route is not yet ETH-qualified. Consequently, current code supports no
claim of physical speedup, energy efficiency, multi-rank scaling, or general
UPMEM TN execution. Physical qualification is a separate run using the exact
source, native binaries, topology, and configuration recorded in evidence.

The next measured upgrade should be driven by physical one-DPU qualification,
then tasklet/DPU scaling, slice scheduling, residency, communication, and
calibrated planning only where evidence justifies the extra mechanism.
