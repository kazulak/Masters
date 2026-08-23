# Architecture Reset Contract

This is the T0 contract for the reset branch. It is a boundary document; it
does not certify planned capabilities as implemented.

## Simulation Semantics

`SimulationJob` contains `circuit`, `query`, `parameters`, and `seed`.

The supported circuit representation is either a deterministic built-in circuit
name plus parameter mapping or an OpenQASM 2 source file. Both normalize to a
circuit with `name`, `n_qubits`, and an ordered tuple of gate operations. Each
operation contains a lowercase gate name, an ordered wire tuple, and angle
parameters in radians. The supported gate set is `I`, `H`, `X`, `Y`, `Z`, `S`,
`T`, `RY`, `RZ`, `CX`/`CNOT`, `CZ`, and `SWAP`.

Version 1 supports only the `pre_measurement_statevector` query, meaning the
complete noiseless statevector immediately after the final gate.
Measurements, noise, reset, classical conditions, and approximate sampling are
unsupported. Built-in generators are deterministic and require `seed=None`.
A future random generator must require an explicit integer seed.

Tensor axes are ordered by ascending wire number. Returned statevectors use
QuEST little-endian basis indexing: amplitude `i` uses bit `wire` of `i`.
Shared comparison families are `QRNG`, `BB84`, `BV`, `EDC`, `XOR`, and `HS`.

TN output tensors retain axes in `(wire 0, wire 1, ..., wire n-1)` order.
Every adapter converting a TN result to a statevector must flatten the output
tensor in Fortran order, equivalent to the current
`tensor_to_quest_statevector` bit mapping, and must verify the resulting length
and ordering against the query contract. No adapter may silently use C-order
flattening.

The target-neutral flow is:

```text
network, inputs = lower_tensor_network(job)
dag = build_contraction_dag(network, path)
```

## Result And Failure Contracts

```text
JsonScalar = str | int | float | bool | None
JsonValue = JsonScalar | list[JsonValue] | dict[str, JsonValue]

ExecutionSample:
  output: copied, read-only NumPy array
  measurement: Measurement
  backend_facts: mapping[str, JsonValue]
  numeric_facts: mapping[str, JsonValue]
```

`Measurement` has the exact fields listed in `docs/timing.md`.

```text
UnsupportedExecution(stage, reason, capability)
  preflight rejected the request before runtime side effects

ExecutionFailed(stage, reason, backend_facts)
  a runtime attempt began, including session/process opening,
  encoding, transfer, kernel execution, decoding, or finalization
```

The experiment layer catches both exceptions and writes a sample row. Every
attempt that returns or raises inside the experiment process produces a row;
an externally killed process cannot guarantee evidence completion.

## Dependency Direction

```text
model -> standard library and NumPy typing only
circuits -> model
lowering -> model, circuits
planning -> model
numerics -> model
results -> standard library and NumPy typing only
cpu -> model, numerics, results
upmem.plan -> model, numerics
upmem.runtime -> model, numerics, results, upmem.plan, protocol, native_session
baselines -> model, results
evidence -> model, results
experiment -> planning, cpu, baselines, upmem.runtime, evidence
report -> evidence
cli -> experiment, report
```

`model.py` and `results.py` are foundational modules and import no other
`quantum_bench` module. Planning does not import UPMEM runtime. Reporting does
not execute experiments.

## Numeric Policies

Execution policies are:

```text
split_complex_float32_v1
split_complex_int8_shared_scale_v1
```

Float32 uses float32 input planes, product and K accumulation, float32 host
combination, and complex64 output. Int8 uses one shared scale per complex
operand, nearest-even rounding, range `[-127, 127]`, scale `1.0` for an
all-zero tensor, DPU int32 products, and host int64 accumulation and
combination.

Each complex contraction computes four real products:

```text
rr = Are * Bre
ii = Aim * Bim
ri = Are * Bim
ir = Aim * Bre
real = int64(rr) - int64(ii)
imag = int64(ri) + int64(ir)
```

Different slice branches may use different scales. Each branch is decoded with
`scale_a * scale_b`, then decoded partials are reduced in deterministic
node-ID order. Raw integer equality is checked per branch and product; final
decoded results are compared numerically.

For int8, each logical input `TensorView` receives one scale per slice branch.
The scale is shared by its real and imaginary planes and reused for every
output tile and K chunk in that branch. It is not recomputed per tile, chunk,
or operand access.

Plans are unsupported when any of these bounds fail:

```text
k_chunk * 127^2 <= INT32_MAX
total_k * 127^2 <= INT64_MAX
abs(rr) + abs(ii) <= INT64_MAX
abs(ri) + abs(ir) <= INT64_MAX
```

Complex128 is validation-only through `run_complex128_reference`; it is not an
execution numeric policy or part of `physical_plan_id`.

## Physical Plan Schema

`PLAN_SCHEMA_VERSION = 1`.

```text
UpmemPlan:
  logical_plan_id
  numeric_policy
  topology
  stages
  intermediate_policy = host_roundtrip_v1
  kernel_policy
```

```text
UpmemStage:
  stage_id
  kind = contract_batch | host_reduce
  node_ids
  work_units
```

Stages are ordered first by DAG topological order and then by lexicographic
`stage_id`. An unsliced node is one `contract_batch` with that node as its
only node. A sliced `contract_batch` contains exactly the direct
`ContractNode` dependencies of one `ReduceNode`, sorted lexicographically.
Compatibility requires equal operation signature, B/M/K/N geometry, numeric
policy, tile policy, tasklet count, requested topology, output dtype, and output
layout. Work units are sorted by stage, rank, DPU, wave, and tile coordinates.

A `host_reduce` stage has exactly one `ReduceNode` in `node_ids`, has an empty
`work_units` tuple, and consumes exactly the immediately preceding matching
`contract_batch` outputs. UPMEM mapping may group existing branches but never
introduces slicing.

## Target Trees

The reset target Python tree is:

```text
src/quantum_bench/
  model.py circuits.py lowering.py planning.py numerics.py results.py
  cpu.py baselines.py experiment.py evidence.py report.py cli.py
  upmem/plan.py upmem/tiling.py upmem/protocol.py
  upmem/native_session.py upmem/runtime.py
```

The T1D active, self-contained native tree is:

```text
native/upmem/runtime/
  Makefile host.c dpu.c protocol.h request.c request.h
  plan.c plan.h simplepim_provider.c simplepim_provider.h
  simplepim_management_profile.patch
```

The focused reset test tree is:

```text
tests/
  test_model.py test_lowering.py test_planning.py test_numerics.py
  test_cpu.py test_upmem_plan.py test_upmem_protocol.py
  test_upmem_native_session.py test_upmem_runtime.py
  test_experiment_evidence.py test_baselines.py test_cli_report.py
```

Old tests are deleted only after their scientific assertion is mapped to one
of these focused modules. Obsolete class-shape tests do not need replacement.

## Native And External Build Contract

### T1D Active v4 Truth

The active v4 implementation is now split by responsibility between:

```text
native/upmem/runtime/
src/quantum_bench/upmem/runtime.py
src/quantum_bench/upmem/plan.py
```

Its build command is:

```text
make -C native/upmem/runtime NR_TASKLETS=1 all
```

`NR_TASKLETS` may be selected in the active v4 range `1..24`. This is the
active reset command and tree.

The active tree is self-contained for ABI-v4. The Python coordinator discovers
its binaries below `native/upmem/runtime/`, and the old mixed-tree ABI-v4
sources and Make target have been removed. The active provider preserves raw
SDK explicit-rank allocation, allocation verification, management metadata
construction, initialization-binary launch, and release.

### SimplePIM Provider Contract

The staged `simplepim_provider` does:

1. allocate exactly the requested DPU count on an explicit rank path; it must
   not auto-select another rank or silently allocate a different count;
2. construct SimplePIM management metadata after allocation; it does not call
   `table_management_init_with_profile`;
3. require successful SDK `dpu_load` followed by synchronous SDK
   `dpu_launch` of the initialization binary before exposing management
   metadata; there is no separate initialization terminal record;
4. expose requested rank path, requested DPU count, allocated DPU count,
   observed rank/topology, initialization binary identity, and allocation
   verification;
5. after allocation succeeds, release the allocation on every terminal path,
   including load, transfer, launch, timeout, validation, and unexpected-error
   paths; build and pre-allocation failures have no allocation to release; and
6. expose release attempted, release succeeded, and release verification.

Failure to verify allocation or release is a failed run, not a successful run
with incomplete metadata. The copied management-profile patch is retained as
provenance only. Its profile initializer is not an active v4 build input:
the patch's backend syntax is incompatible with explicit `rankPath` allocation.

Required SimplePIM inputs are commit
`1d639c53532555f01e9f71d872e7712b166d6cba` and management patch SHA-256
`5ac09fd1c0a25c234e44615540f2e1585ce162a27a2d4215e5992ddbdf549a0d`.
These inputs do not mean SimplePIM compute is active. The v4 compute kernel is
raw UPMEM SDK code using SimplePIM management metadata/types and the
initialization source; it is not a SimplePIM operator. The source-string tests
are drift tripwires for this active source contract, not runtime or hardware
tests. Clean local SDK builds passed for `NR_TASKLETS=1` and `24`; physical
behavior remains unqualified until ETH qualification.

## Session Evidence

`sessions.jsonl` contains at least these fields for every opened or attempted
session:

```text
session_instance_id
session_protocol_id
open_s
close_s
status
terminal_backend_facts
release_attempted
release_succeeded
release_verified
```

`release_verified` must be true for a successful session. A failed release
remains visible in the session record and causes the associated run to fail.

## Qualification Fixture

The first reset qualification fixture is frozen as follows:

```text
circuit: builtin quantization_stress
n_qubits: 4
repeat_layers: 2
query: pre_measurement_statevector
seed: None
planner: opt_einsum.greedy
opt_einsum: 3.4.0
planner_config_hash: 2197acf1524706bd82747388500c1b3a4fe50aefeb9d40cdf554575733b0b275
tensor_network_structure_id_v1: 9f9e84cbc09eefdd8a1d93178051181db6f91c6c5b177e34d0a689e32d2d6d13
logical_plan_id_base_dag_hash: 0893aab5797f8fe01bfda4345fc0b71efbbcb10941be3b158565149d8755a95f
slice_node: contract_24
selected_labels: (12, 14)
selected_dimensions: (2, 2)
expected_slice_count: 4
```

The exact contraction path is:

```text
((0,4),(0,3),(0,2),(0,1),(0,4),(20,24),(1,3),(19,22),
 (1,2),(17,20),(1,5),(1,4),(1,3),(1,2),(1,4),(1,3),
 (1,2),(0,2),(1,11),(3,7),(4,6),(0,5),(0,4),(1,3),
 (4,5),(1,3),(0,2),(0,2),(0,1))
```

The fixture path, planner version/configuration, structure payload, and hashes
are part of the qualification input. A future planner change creates a new
fixture identity; it must not silently change this one.

## Qualification And Merge

The branch has two independent states:

```text
software-ready
  full software tests, Ruff, simulator correctness, schema checks,
  no fallback, and clean active architecture

physical-qualified
  later ETH run with the exact source, executable, environment,
  allocation, validation, and artifact records
```

T13 may mark the branch software-ready. T14 is required before physical
qualification. Hardware owner: `tkazulak`. Reservation/date, rank path, and
SDK version are pending at T0.

## Ordered Tasks

```text
T0    contracts, semantics, identities, baseline, dependency rules
T1A   UPMEM plan/tiling ownership
T1B   Python protocol/session split
T1C   self-contained native runtime
T1D   runtime coordinator move
T2    core model, circuits, lowering
T3    planner isolation
T4A   results and CPU single-run API
T4B1  UPMEM session API
T4B2  remove generic execution wrappers and migrate callers
T4C   final UpmemStage/UpmemPlan schema
T5A   evidence schemas and identities
T5B   experiment repetition/session lifecycle
T5C   timing normalization and old-emitter deletion
T6A   pure complex encoding/decoding
T6B   CPU physical-plan replay
T7    complex UPMEM execution
T8    logical multi-label slicing
T9    slice batches and host reduction
T10A  Quimb/cotengra baseline
T10B  QuEST CPU baseline
T10C  QuEST GPU verification
T11A  configuration and CLI
T11B  verification and reporting
T12A  remove providers/routing
T12B  remove milestone workflows/configurations
T12C  remove old TaskGraph and UPMEM generations
T13   software qualification
T14   later ETH physical qualification
```
