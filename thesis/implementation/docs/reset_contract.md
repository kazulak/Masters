# Active Reset Contract

This document describes the active software contract. It is not a milestone
history and it does not certify physical UPMEM qualification.

## Simulation Job

```python
SimulationQuery = Literal["pre_measurement_statevector"]

@dataclass(frozen=True, slots=True)
class SimulationJob:
    circuit: CircuitSpec
    query: SimulationQuery = "pre_measurement_statevector"
    parameters: tuple[tuple[str, str | int | float | bool | None], ...] = ()
    seed: int | None = None
```

`make_simulation_job` accepts an iterable of parameter pairs, rejects empty or
duplicate keys and non-finite floats, sorts parameters by key, and returns the
normalized job. Direct `SimulationJob` instances must already be sorted and
unique. The only query is the complete noiseless statevector immediately after
the final gate. Measurements, noise, reset, classical conditions and sampling
are unsupported. Numeric execution dtype is not part of `SimulationJob`.

The supported circuit representation is a deterministic built-in circuit or
OpenQASM 2 normalized to `CircuitSpec`: name, qubit count, ordered operations,
and source metadata. Operations contain a lowercase gate name, ordered wires,
and angles in radians. The active gate set is `I`, `H`, `X`, `Y`, `Z`, `S`, `T`,
`RY`, `RZ`, `CX`/`CNOT`, `CZ`, and `SWAP`. Built-in jobs are deterministic and
use `seed=None`; a randomized generator must use an explicit integer seed.

Statevector basis indexing is little-endian: amplitude `i` uses bit `wire` of
`i`. Tensor axes and returned statevector ordering are ascending wire order.
TN adapters must flatten the result in the documented Fortran-order mapping
and verify length and ordering. The comparison families currently used by the
benchmark are `QRNG`, `BB84`, `BV`, `EDC`, `XOR`, and `HS`.

## Tensor Network and Logical Graph

`TensorNetwork` is target-neutral semantic metadata:

```python
@dataclass(frozen=True)
class TensorNetwork:
    circuit: CircuitSpec
    tensors: tuple[TensorSpec, ...]
    output_labels: tuple[int, ...]
    einsum_expression: str
```

It contains no arrays, selected path, dependencies, slicing, target estimates,
or runtime state. `ContractionDAG` is the sole logical execution IR. It holds
`TensorSpec`, `TensorView`, `ContractNode`, `ReduceNode`, and dependencies for
the selected path and explicit slice reconstruction. The DAG is a logical plan,
not a path-independent problem description.

The target-neutral lowering flow is:

```python
network, inputs = lower_tensor_network(job)
dag = build_contraction_dag(network, path)
```

Lowering returns the network and a mapping from tensor ID to NumPy input array.
It does not import or execute UPMEM code. UPMEM mapping consumes the DAG and may
tile or place work, but must not mutate the DAG or silently introduce slicing.

## Public Types

These are the active public records and aliases. New public types require an
explicit contract update.

```text
model.py:
  CircuitOperation, CircuitSpec, SimulationQuery, SimulationJob,
  TensorSpec, TensorView, SliceSpec, ContractNode, ReduceNode,
  GraphNode, ContractionDAG, TensorNetwork

results.py:
  JsonScalar, JsonValue, Measurement, ExecutionSample,
  UnsupportedExecution, ExecutionFailed

numerics.py:
  NumericPolicy, EncodedComplexTensor

upmem/plan.py:
  UpmemTopology, UpmemResources, UpmemWorkUnit, UpmemStage, UpmemPlan

upmem/runtime.py:
  UpmemSession
```

Records are frozen. Execution outputs are detached, read-only NumPy arrays.
Fact mappings contain only JSON-compatible scalar, tuple, list-equivalent, and
mapping values; non-finite numbers and arbitrary objects are rejected.

## Numeric Policy

The executable policies are exactly:

```text
split_complex_float32_v1
split_complex_int8_shared_scale_v1
```

`EncodedComplexTensor` contains real and imaginary planes, one shared operand
scale, and separate real/imaginary saturation counts. Planes are owned,
C-contiguous, read-only arrays. Encoding is pure and rejects non-finite input.

For int8, the host computes one scale from the maximum absolute value of both
planes of an operand. A zero operand uses scale `1.0` and an all-zero payload.
Values are rounded to the selected integer representation; saturation means
the rounded value was outside `[-127, 127]` before clipping. A value that
legitimately rounds to `-127` or `127` is not saturation.

Complex products are four real contractions:

```text
rr = Are * Bre       ii = Aim * Bim
ri = Are * Bim       ir = Aim * Bre
real = rr - ii       imag = ri + ir
```

The four product arrays are combined in the declared policy precision. Int8
products use int32 tile accumulation, checked int64 host combination, and
decode using the two operand scales. Complex128 is validation reference only;
it is not an execution policy or physical-plan identity.

## UPMEM Plan

`UpmemPlan` is target-specific and contains `logical_plan_id`, numeric policy,
topology, ordered stages, intermediate policy and kernel policy. The active
intermediate policy is `host_roundtrip_v1`. The active kernel policy is
`dpu_real_tile_v4_wram_panel_v1`. The split-complex numeric policies normatively
use sequential `rr`, `ii`, `ri`, and `ir` real-product orchestration.

```python
@dataclass(frozen=True)
class UpmemStage:
    stage_id: str
    kind: Literal["contract_batch", "host_reduce"]
    node_ids: tuple[str, ...]
    work_units: tuple[UpmemWorkUnit, ...]
```

`contract_batch` contains work units and is a deterministic logical grouping
whose work units execute sequentially today. It is not a concurrent batch or
slice-group execution mechanism. `host_reduce` names one reduction node and
contains no work units. Stage and node IDs are unique; work-unit IDs are
globally unique; work units reference a declared stage node. Work-unit fields
record rank/DPU/wave, tile coordinates and sizes, aligned MRAM bytes, and
estimated arithmetic work. These estimates are plan metadata, not measured
hardware timings.

The native ABI is a real-valued output-tile contraction ABI. Complex execution
uses four real-product launches on the assigned work. The current plan is
bounded and host-roundtrip; it does not imply graph-wide DPU residency,
arbitrary slicing, multi-rank scaling, speedup, or energy efficiency.

The active kernel stages a fixed `KC=64`, `NC=32` B panel in global WRAM and
uses tasklet-indexed A/output buffers. Full panels use aligned MRAM transfers;
the ABI's bounded unaligned helper is for tails only. Its exact source-level
movement and barrier formulas are plan/runtime facts, not hardware counters or
a calibrated cost model.
The requested tasklet count must match both the host and DPU binaries'
compile-time `NR_TASKLETS`; either side rejects a mismatch before accepting
kernel results.

## Execution and Failure

`run_cpu_once`, `replay_upmem_plan_once`, and an opened UPMEM session return an
`ExecutionSample` containing output, `Measurement`, backend facts, and numeric
facts. The experiment layer owns warmup/measurement blocks, evidence rows, and
validation. The active collection policy opens one UPMEM session per attempt;
session open/close remain separate from `steady_execution_v1` timing.

`UnsupportedExecution(stage, reason, capability)` means preflight rejected the
request before runtime side effects. `ExecutionFailed(stage, reason,
backend_facts)` means an attempt began, including session/process opening,
encoding, transfer, kernel, decoding, or finalization. The experiment layer
catches both. Every attempt that returns or raises inside the process gets a
sample row; an externally killed process cannot guarantee one.

## Active Commands

```text
python -m quantum_bench.cli plan    --config CONFIG --output DIR
python -m quantum_bench.cli run     --config CONFIG --output DIR
python -m quantum_bench.cli qualify --config CONFIG --output DIR --allow-physical
python -m quantum_bench.cli verify  --input DIR
python -m quantum_bench.cli report  --input DIR --output DIR
```

Physical routes additionally require `UPMEM_ALLOW_PHYSICAL_HARDWARE=1` and a
clean worktree for `qualify`. No physical qualification is asserted by this
contract.

## Evidence and Claims

Every run has `manifest.json`, `samples.jsonl`, and `sessions.jsonl`. The
manifest binds source commit, dirty-tree state, experiment, environment,
validation policy, expected counts, file names, and one canonical identity
binding for every selected `(case_id, plan_id, route_id)`. Samples bind case,
route, plan, identities, attempt kind/index, block/order, timing, facts,
validation, and failure.
Sessions bind protocol/runtime identity, open/close state, terminal facts, and
resource release. Samples record attempt kind/index plus deterministic block
and order. Canonical validation rejects missing links, duplicate IDs, wrong
counts, routes outside or missing from the experiment matrix, invalid scopes,
failed release, or identity mismatches. The active schemas are manifest v2,
sample v3, session v1, and report v3.

Claim admission is explicit. Execution/policy correctness requires a successful
sample and an applicable, passed policy reference. Full-precision accuracy
qualification additionally requires `accuracy_qualified=true`. Physical
execution requires physical UPMEM facts; simulator and model routes are
excluded. Timing rejects model/simulator rows and requires measured timing
facts. A speedup candidate must have an applicable and passed policy reference,
`accuracy_qualified=true`, qualified physical
provenance, and matching scope and identities. Its matching CPU same-plan
baseline must have `accuracy_qualified=true` and pass its policy reference when
applicable. Complete planned measurements, clean linked artifacts, and a
non-bring-up timing scope remain required. Scaling requires a matched physical
pair. Energy
requires positive measured energy, sensor/counter identity, interval, boundary
and provenance. A rejected claim must remain visible with reasons.
