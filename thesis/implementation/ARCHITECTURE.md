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
units, and kernel policy. `contract_batch` is a deterministic logical grouping
whose work units execute sequentially today; it is not a concurrent batch or
slice-group execution mechanism. The current intermediate policy is host
round-trip.

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

The active real-tile kernel policy is `dpu_real_tile_v4_wram_panel_v1`. It
uses fixed `KC=64` and `NC=32` panel geometry, a global shared B panel, and
tasklet-indexed A/output buffers in WRAM. Full panels use aligned transfers;
tail geometry uses the ABI's bounded unaligned helper path. This is local
operand staging and reuse inside one dense real-tile operation. It is not
graph-wide residency, a general memory scheduler, a tasklet-scaling result, or
a physical performance result.

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

The scope definition is broader than current route support. Direct
Quimb/cotengra and QuEST routes emit `simulation_end_to_end_v1`. NumPy
same-DAG and UPMEM routes emit `steady_execution_v1` because their DAG or
session is prepared before repetitions. A matched end-to-end
NumPy-versus-UPMEM comparison is not yet available.

Physical routes fail closed: no simulator or CPU fallback can satisfy a
physical request. Simulator evidence is never admitted as physical timing,
speedup, scaling, or energy evidence.

## Cost-Model Feature Status

The implementation exposes deterministic features for future calibration; it
does not instantiate or validate a PIM-aware cost model. `B_host-DPU` is
represented by application-visible H2D/D2H byte fields, not a hardware DMA
counter. `B_MRAM-WRAM` is represented by exact source-level helper-call and
requested-payload formulas plus an aligned-transfer-byte estimate, not measured
MRAM traffic. `I_DPU` is an exact real-MAC formula, `N_sync` is an exact barrier
event/tasklet-call formula, `E_num` is numeric-policy, scale, and saturation
evidence without a calibrated coefficient, and `P_WRAM` is a kernel-buffer
allocation formula plus executable section facts. Directional A/B/partial-C/C
movement facts are exact source-level helper and payload formulas; aligned span
bytes are estimates, not hardware counters. No coefficient, prediction error,
or path-ranking claim is currently calibrated.

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

Controlled tests cover the reset software and SDK-simulator boundaries,
including ABI-v4 WRAM-panel complex float32/int8 checks. Tags
`thesis-m6-software-ready-v1` and
`thesis-m7a-wram-kernel-software-ready-v1`, with their release bundles,
confirm exact-head software qualification for their respective sources. The
physical UPMEM route remains pending and is not yet ETH-qualified. M7B
pre-physical software qualification is recorded by
`thesis-m7b-prephysical-software-ready-v1`; that tag is not physical evidence.
Consequently, current code supports no claim of physical speedup, energy
efficiency, multi-rank scaling, or general UPMEM TN execution. Physical
qualification is a separate run using the exact source, native binaries,
topology, and configuration recorded in evidence.

The next measured upgrade should be driven by physical one-DPU qualification,
then tasklet/DPU scaling, slice scheduling, residency, communication, and
calibrated planning only where evidence justifies the extra mechanism.
