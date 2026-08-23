# Architecture Reset Contract

This is the T0 contract for the reset branch. It is a boundary document; it
does not certify planned capabilities as implemented.

## Simulation Semantics

The public query type is:

```python
SimulationQuery = Literal["pre_measurement_statevector"]
```

`SimulationJob` is a frozen, slotted record with exactly these fields, in this
order:

```python
@dataclass(frozen=True, slots=True)
class SimulationJob:
    circuit: CircuitSpec
    query: SimulationQuery = "pre_measurement_statevector"
    parameters: tuple[tuple[str, str | int | float | bool | None], ...] = ()
    seed: int | None = None
```

The functional construction API is:

```python
make_simulation_job(
    circuit,
    *,
    query="pre_measurement_statevector",
    parameters=(),
    seed=None,
) -> SimulationJob
```

Its `parameters` input is an `Iterable[tuple[str, scalar]]`, where scalar is
`str | int | float | bool | None`. The function validates nonempty string keys,
rejects duplicate keys, sorts entries lexicographically by key, and returns a
`SimulationJob`. `SimulationJob.__post_init__` rejects direct instances whose
parameters are not already strictly key-sorted and unique, whose query is
unsupported, or whose seed is not `None` or an integer. Current built-in jobs
use empty parameters and `seed=None`; the normalized `CircuitSpec` contains
gate parameters and source information. Numeric execution dtype is not a
`SimulationJob` field.

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

`TensorNetwork` is a frozen, slotted, non-executable semantic record with
exactly these fields:

```python
@dataclass(frozen=True, slots=True)
class TensorNetwork:
    circuit: CircuitSpec
    tensors: tuple[TensorSpec, ...]
    output_labels: tuple[int, ...]
    einsum_expression: str
```

It contains no arrays, path, slicing, dependencies, target estimates,
executor data, or timing. `ContractionDAG` remains the sole logical execution
IR: it contains the selected contraction order, slicing branches, reductions,
and dependencies. `TensorNetwork` only describes semantic metadata produced by
lowering.

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

## Planner Contract (T3-0)

T3-0 freezes the canonical planner boundary. The production migration is
implemented by the root `planning.py` module described below. This section
records the active contract and its focused software verification; it does not
claim physical hardware qualification.

The canonical public API contains only these adapter functions:

```python
def plan_opt_einsum(
    network: TensorNetwork,
    *,
    optimize: str = "greedy",
) -> tuple[tuple[tuple[int, int], ...], dict[str, object]]:
    ...

def plan_cotengra(
    network: TensorNetwork,
    *,
    objective: str = "flops",
    methods: str = "greedy",
    max_repeats: int = 1,
    seed: int = 0,
) -> tuple[tuple[tuple[int, int], ...], dict[str, object]]:
    ...
```

The returned path is a complete, validated binary active-list path. The
provenance value is a JSON-compatible mapping with exactly these keys:

```text
planner_engine         selected external planner engine
planner_id              deterministic adapter and mode identifier
planner_kind            external optimizer/tree adapter classification
optimize_mode           requested opt_einsum mode or cotengra method
objective               requested planner objective
cost_basis              engine-specific path-cost basis
planner_config          resolved caller and constant optimizer settings used
                        by the adapter; dependency versions are excluded
planner_config_hash     SHA-256 of sorted canonical JSON configuration
path_info_text          human-readable optimizer summary
largest_intermediate    engine-reported largest intermediate, if available
naive_flops             engine-reported naive FLOPs, if available
optimized_flops         engine-reported optimized FLOPs, if available
planning_time_s         wall time spent only inside path planning
dependency_versions     relevant adapter dependency versions
```

`planner_config_hash` is computed from the canonical sorted JSON encoding of
the resolved caller and constant optimizer settings actually used by the
selected adapter. Dependency versions and `planning_time_s` are excluded from
the hash. Unused UPMEM, numeric, kernel, and topology settings are excluded.
Planner provenance does not include target execution choices.

The root `planning.py` imports the canonical model and opt_einsum; it
imports cotengra lazily only when that adapter is selected. It imports no
UPMEM runtime or target-planning module. The standard active adapters are
opt_einsum and cotengra only. The canonical path exposes no `PlannerRequest`,
`PlannerResult`, `PlannerIdentity`, `PlannerEngine`, public planner classes,
or generic public dispatcher. A private config-to-function helper in the
experiment or M5 coordinator is permitted.

Both adapters reject an empty tensor network with
`ValueError("cannot plan an empty tensor network")`. A singleton network
returns an empty path and normal provenance without invoking the external path
optimizer. The cotengra adapter may still import cotengra for dependency
provenance in that case, and a missing dependency remains an explicit runtime
error.

The PIM-aware projected-prefix greedy planner is historical and exploratory.
It is uncalibrated, is not part of the canonical dispatcher, and is retained
through T12 only for old configurations, tests, and evidence. Hardware-
calibrated target-aware planning is separate future work.

Canonical M5 circuit-study configurations use opt_einsum or cotengra.
Unsupported planner engines fail explicitly. Historical suites are not
silently migrated.

## T4-0/T6A Dependency Correction

This documentation-only correction freezes the dependency order for the reset
implementation. T6A, T4A, T4C, and T6B are implemented at the current base;
T7 remains the next implementation task.

Pure T6A numerics must be implemented before T4A results and CPU execution,
because `run_cpu_once` consumes the final `NumericPolicy` contract. The
correct dependency order is:

```text
T6A  pure numerics
  -> T4A  results and CPU single-run API
  -> T4C  final UpmemStage/UpmemPlan schema (complete)
  -> T6B  physical-plan CPU replay (complete)
  -> T7   four real-product ABI execution (next)
  -> T4B1 UPMEM session API
  -> T4B2 removal of generic wrappers
  -> T5   evidence and experiment lifecycle
```

T8 and all later tasks retain their existing order. T4-0 is contract-frozen
and implementation-pending.

### Numeric contract

`NumericPolicy` is a public type alias, not a class:

```python
NumericPolicy = Literal[
    "split_complex_float32_v1",
    "split_complex_int8_shared_scale_v1",
]
```

The public encoded value is:

```python
@dataclass(frozen=True, slots=True)
class EncodedComplexTensor:
    real: np.ndarray
    imag: np.ndarray
    scale: float
    saturation_real: int
    saturation_imag: int
```

`real` and `imag` have the same shape, are owned C-contiguous copies, and are
read-only. Encoding is pure, never mutates the input, and rejects non-finite
values and unsupported policies. The public pure functions are:

```python
def encode_complex_tensor(
    value: np.ndarray,
    policy: NumericPolicy,
) -> EncodedComplexTensor: ...

def contract_complex_products(
    node: ContractNode,
    left: EncodedComplexTensor,
    right: EncodedComplexTensor,
    policy: NumericPolicy,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return rr, ii, ri, and ir in that order."""

def decode_complex_products(
    products: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    left_scale: float,
    right_scale: float,
    policy: NumericPolicy,
) -> np.ndarray: ...
```

`contract_complex_products` requires the left and right real/imaginary planes
to match `node.left.shape` and `node.right.shape`, respectively. Plane dtypes
must be float32 for `split_complex_float32_v1` and int8 for
`split_complex_int8_shared_scale_v1`. It returns four same-shape, owned product
arrays in `rr`, `ii`, `ri`, `ir` order. `decode_complex_products` requires four
same-shape product arrays and finite, strictly positive scales. It preserves
the product-axis order established for `node.output_labels` and returns an
owned, read-only array.

The validation-only complex128 reference is owned by `cpu.py`:

```python
def run_complex128_reference(
    dag: ContractionDAG,
    inputs: Mapping[str, np.ndarray],
) -> np.ndarray: ...
```

It returns an owned, read-only complex128 output. Complex128 is not an
execution policy and is not part of a physical-plan identity. Reference values
and executor outputs must be finite.

### Result and failure contracts

```python
JsonScalar = str | int | float | bool | None
JsonValue = (
    JsonScalar
    | tuple["JsonValue", ...]
    | Mapping[str, "JsonValue"]
)

@dataclass(frozen=True, slots=True)
class ExecutionSample:
    output: np.ndarray
    measurement: Measurement
    backend_facts: Mapping[str, JsonValue]
    numeric_facts: Mapping[str, JsonValue]
```

The sample owns a copied, read-only output array. Both fact mappings and every
nested container are recursively copied and frozen. Canonical evidence
serialization converts immutable tuples and mappings to JSON arrays and
objects. `Measurement` has the exact fields listed in `docs/timing.md`;
`total_wall_s` is the coordinator's authoritative wall observation.

```text
UnsupportedExecution(stage, reason, capability)
  preflight rejected the request before runtime side effects

ExecutionFailed(stage, reason, backend_facts)
  a runtime attempt began, including session/process opening,
  encoding, transfer, kernel execution, decoding, or finalization
```

`UnsupportedExecution` and `ExecutionFailed` are exceptions with the named
fields. The experiment layer catches either exception and writes a sample row.

Every attempt that returns or raises inside the experiment process produces a
row; an externally killed process cannot guarantee evidence completion.

## Public Reset Types

No additional public reset type may be introduced without amending this
contract. The following compatibility names are explicitly temporary and
expire at T12.

```text
model.py:
  SimulationQuery (type alias), SimulationJob, CircuitOperation, CircuitSpec,
  TensorSpec, TensorView, TensorNetwork, SliceSpec, ContractNode, ReduceNode,
  GraphNode (type alias), ContractionDAG

results.py:
  Measurement, ExecutionSample, UnsupportedExecution, ExecutionFailed

numerics.py:
  NumericPolicy, EncodedComplexTensor

upmem/plan.py:
  UpmemTopology, UpmemResources, UpmemWorkUnit, UpmemStage, UpmemPlan

upmem/runtime.py:
  UpmemSession
```

## Canonical Migration Route

Migration checks use only this active route:

```text
circuits -> lowering -> planning -> cpu/upmem -> experiment -> evidence/report
```

Historical providers and milestone code are not canonical and are excluded
from canonical-route import checks.

The following temporary re-exports are authorized until T12. They do not form
a generic compatibility framework and must not duplicate type definitions:

```text
core.records:
  CircuitOperation, CircuitSpec, TensorSpec, and TensorNetworkSpec as an
  alias of TensorNetwork for historical consumers

tn.graph:
  exactly TensorView, SliceSpec, ContractNode, ReduceNode, GraphNode, and
  ContractionDAG while historical functions migrate; no planner, executor,
  or TaskGraph types

tn.network:
  TensorNetworkValue, build_tensor_network, and interleaved_einsum_args
  wrappers for historical consumers. `interleaved_einsum_args` is a temporary
  adapter required until T12.
```

These compatibility modules must not be imported by the canonical route after
T2. The T2 order is: model ownership and re-exports, flatten the
`quantum_bench.circuits` package into `quantum_bench.circuits.py`, move
lowering ownership and migrate canonical consumers, then run the full suite.

## Dependency Direction

```text
model -> standard library and NumPy typing only
circuits -> model
lowering -> model, circuits, core.indices
planning -> model, opt_einsum; cotengra is imported lazily
numerics -> model
results -> standard library and NumPy typing only
cpu -> model, lowering, numerics, results
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

`core.indices` is a pure target-neutral helper for label allocation, einsum
symbol generation, and string-einsum capability checks. It contains no
planner, executor, filesystem, or hardware behavior.

## Numeric Policies

The execution policy is the public `NumericPolicy` alias frozen in the T4-0
section. Its only values are:

```text
split_complex_float32_v1
split_complex_int8_shared_scale_v1
```

Each complex contraction computes four real products:

```text
rr = Are * Bre
ii = Aim * Bim
ri = Are * Bim
ir = Aim * Bre
```

The float32 policy combines those products in float32 and returns complex64:

```text
real_float32 = float32(rr - ii)
imag_float32 = float32(ri + ir)
```

Only the int8 policy widens products for host combination:

```text
real_int64 = int64(rr) - int64(ii)
imag_int64 = int64(ri) + int64(ir)
```

For int8, different slice branches may use different scales. Each branch is
decoded with `scale_a * scale_b`, then decoded partials are reduced in
deterministic node-ID order. Raw integer equality is checked per branch and
product; final decoded results are compared numerically.

For int8, each logical input `TensorView` receives one scale per slice branch.
The scale is shared by its real and imaginary planes and reused for every
output tile and K chunk in that branch. It is not recomputed per tile, chunk,
or operand access.

Static int8 mapping rejects plans when either of these bounds fails:

```text
k_chunk * 127^2 <= INT32_MAX
2 * total_k * 127^2 <= INT64_MAX
```

Runtime execution additionally checks the observed complex accumulators:

```text
abs(rr) + abs(ii) <= INT64_MAX
abs(ri) + abs(ir) <= INT64_MAX
```

The float32 policy uses float32 input planes, product and K accumulation, and
host combination. The int8 policy uses host encoding, one shared scale per
complex operand, nearest-even rounding, range `[-127, 127]`, scale `1.0` for an
all-zero tensor, DPU int32 products, and host int64 accumulation and
combination.

Complex128 is validation-only through the `cpu.py` function
`run_complex128_reference`; it is not an execution numeric policy or part of
`physical_plan_id`.

## Physical Plan Schema

`PLAN_SCHEMA_VERSION = 1`.

The final public mapper API is:

```python
PLAN_SCHEMA_VERSION = 1

def plan_upmem(
    dag: ContractionDAG,
    *,
    numeric_policy: NumericPolicy,
    topology: UpmemTopology,
) -> UpmemPlan: ...

def validate_upmem_plan(
    dag: ContractionDAG,
    plan: UpmemPlan,
) -> None: ...

def physical_plan_id(plan: UpmemPlan) -> str: ...
```

`plan_upmem` performs only pure mapping and validation. Unsupported topology,
geometry, or overflow raises `results.UnsupportedExecution` at stage
`"mapping"`, before any runtime side effect.
It also rejects a valid DAG with no `ContractNode` as
`upmem_no_contract_work`; such a plan has no physical kernel work.
Only `TileLoweringError` and explicit mapper capability failures are converted
to unsupported results; generic `ValueError` and `OverflowError` propagate as defects.

```python
@dataclass(frozen=True, slots=True, kw_only=True)
class UpmemTopology:
    dpu_count: int
    tasklets_per_dpu: int
    rank_count: int = 1

@dataclass(frozen=True, slots=True, kw_only=True)
class UpmemWorkUnit:
    node_id: str
    stable_tile_id: str
    wave: int
    logical_rank: int
    logical_dpu: int
    batch_start: int
    batch_size: int
    m_start: int
    m_size: int
    n_start: int
    n_size: int
    k_start: int
    k_size: int
    estimated_input_bytes: int
    estimated_output_bytes: int
    aligned_mram_bytes: int
    estimated_arithmetic_work: int

@dataclass(frozen=True, slots=True, kw_only=True)
class UpmemStage:
    stage_id: str
    kind: Literal["contract_batch", "host_reduce"]
    node_ids: tuple[str, ...]
    work_units: tuple[UpmemWorkUnit, ...]

@dataclass(frozen=True, slots=True, kw_only=True)
class UpmemPlan:
    logical_plan_id: str
    numeric_policy: NumericPolicy
    topology: UpmemTopology
    stages: tuple[UpmemStage, ...]
    intermediate_policy: Literal["host_roundtrip_v1"] = "host_roundtrip_v1"
    kernel_policy: str = "real_tile_four_product_v1"
```

These are the only public physical-plan types. No additional public plan type
may be introduced without amending this contract.

T4C currently emits singleton `contract_batch` stages in deterministic DAG
topological order, selecting the lexicographically smallest ready `node_id`
at each step. One stage contains one unsliced `ContractNode` and its work
units. T9 will add grouped sliced
`contract_batch` stages containing exactly the direct `ContractNode`
dependencies of one `ReduceNode`, sorted lexicographically. That grouping is
future schema generation behavior, not a T4C guarantee.
Compatibility requires equal operation signature, B/M/K/N geometry, numeric
policy, tile policy, tasklet count, requested topology, output dtype, and output
layout. Within a stage, work units are sorted exactly by:

```text
(logical_rank, logical_dpu, wave, batch_start, m_start, n_start, k_start,
 stable_tile_id)
```

A `host_reduce` stage has exactly one `ReduceNode` in `node_ids`, has an empty
`work_units` tuple, and consumes the direct producer nodes declared by its
`ReduceNode`. Every such producer must occur in an earlier stage; the producer
stage need not be immediately preceding. UPMEM mapping may group existing
branches but never introduces slicing.

The byte and work fields describe one real-valued ABI-v4 tile invocation:

```text
float32: two float32 inputs and one float32 output
int8:    two packed int8 inputs and one int32 output
aligned_mram_bytes: aligned footprint for that one invocation
estimated_arithmetic_work: real MACs, m_size * n_size * k_size
```

The complex route invokes each work unit four times. The plan fields are not
silently multiplied; aggregate transfer and work are runtime evidence.

`physical_plan_id` is the SHA-256 of canonical JSON containing
`PLAN_SCHEMA_VERSION`, every `UpmemPlan` field, and every ordered nested
stage/work-unit field. It excludes `UpmemResources`, paths, callbacks,
binary/ABI hashes, scales, saturation counts, and runtime facts.

The stage and plan fields above are the complete frozen schema for this reset.
Runtime constants, ABI identifiers, executable hashes, binary paths, and
machine-local settings are recorded as executable or run provenance; they do
not become additional `UpmemPlan` fields.

During migration, these final public records temporarily coexist with privately
aliased legacy records. This is compatibility state for T4B2 only, not a
permanent architecture.

## CPU And UPMEM Single-Run Contracts

The CPU route is a single-pass function. It does not own warmups,
repetitions, hashing, reference calculation, validation, or evidence writing:

```python
def run_cpu_once(
    dag: ContractionDAG,
    inputs: Mapping[str, np.ndarray],
    numeric_policy: NumericPolicy,
    *,
    scope_id: str = "steady_execution_v1",
) -> ExecutionSample: ...
```

`run_cpu_once` is the single-run route coordinator and measures its own
authoritative `total_wall_s`. The later experiment layer owns warmup and
repetition loops. The function measures encode, kernel,
host-reduce, and decode phases when they are available.

Preflight validates `NumericPolicy`. For requested-output dataflow, every
`ContractNode` output is int8-policy-derived, a `ReduceNode` output is derived
only when every input is derived, and an original input tensor is not derived.
The `split_complex_int8_shared_scale_v1` policy is unsupported unless
`dag.output` references a derived tensor. The `split_complex_float32_v1`
policy may execute empty or reduce-only DAGs. After the single-run timer
starts, executor phase failures are `ExecutionFailed` with stage `encode`,
`kernel`, `decode`, `host_reduce`, or `finalize`.

UPMEM resources are an immutable, keyword-only boundary record:

```python
@dataclass(frozen=True, slots=True, kw_only=True)
class UpmemResources:
    session_root: str
    host_binary: str
    dpu_binary: str
    initialization_binary: str
    rank_paths: tuple[str, ...] = ()
    session_opener: Callable[..., object] | None = field(
        default=None,
        repr=False,
        compare=False,
    )
```

`session_opener` is a private test/injection seam. It is excluded from plan
identity and evidence. The exact session boundary is:

```python
def open_upmem(
    dag: ContractionDAG,
    plan: UpmemPlan,
    resources: UpmemResources,
    *,
    timeout_s: float = 120.0,
) -> UpmemSession: ...

class UpmemSession:
    def run_once(
        self,
        inputs: Mapping[str, np.ndarray],
    ) -> ExecutionSample: ...

    def close(self) -> Mapping[str, JsonValue]: ...

    def __enter__(self) -> "UpmemSession": ...
    def __exit__(self, exc_type, exc_value, traceback) -> None: ...
```

`open_upmem` validates DAG/plan compatibility before runtime side effects.
`close` is idempotent. A session opening or finalization attempt is a runtime
attempt and therefore reports `ExecutionFailed` on failure.

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
session_close_s
status
terminal_backend_facts
release_attempted
release_succeeded
release_verified
```

`release_verified` must be true for a successful session. A failed release
remains visible in the session record and causes the associated run to fail.
`session_close_s` exists only in this session record or an equivalent session
manifest; it is not part of per-sample `Measurement` or either total scope.

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

## T6B CPU Physical-Plan Replay Contract

T6B adds one CPU-only policy reference for the final physical plan. It is not
UPMEM execution, does not open a device session, and is not a performance
baseline. Its purpose is to reproduce the arithmetic, tile coverage, K-chunk
order, complex reconstruction, and host reduction that T7 must implement on
the real ABI.

The exact public function is:

```python
def replay_upmem_plan_once(
    dag: ContractionDAG,
    plan: UpmemPlan,
    inputs: Mapping[str, np.ndarray],
    *,
    scope_id: str = "steady_execution_v1",
) -> ExecutionSample: ...
```

Before timing, replay validates the DAG, inputs, physical plan, node/stage
coverage, work-unit geometry, byte estimates, MRAM footprints, and complete
non-duplicate tile/K coverage. It supports only `steady_execution_v1`, the
current singleton `contract_batch` stages, and `host_reduce` stages. Grouped
contract batches are deferred to T9. Unsupported scope or stage policy raises
`UnsupportedExecution` before timing; malformed or tampered DAG, input, or
plan data remains `ValueError`.

For every contract, replay materializes complex `TensorView`s, lowers the real
and imaginary float32 planes separately using the tile limits selected by the
numeric policy, and performs unilateral-label reductions in float32 before
host encoding. It verifies matching canonical metadata and exact
correspondence between lowered tiles and final `UpmemWorkUnit`s. Canonical
planes are combined into complex64 operands, and each canonical operand is
encoded once using its shared complex scale. This pre-reduce-before-quantize
order matches the intended physical route and may intentionally differ from
`run_cpu_once` for int8 unilateral reductions.

Each real ABI work unit executes the four products `rr`, `ii`, `ri`, and `ir`:

```text
rr = Are * Bre
ii = Aim * Bim
ri = Are * Bim
ir = Aim * Bre
```

For `split_complex_float32_v1`, products and K-chunk assembly use ascending
K order with float32 multiply/add operations. Float results are compared by
tolerance; raw float32 hashes are diagnostic only. For
`split_complex_int8_shared_scale_v1`, operands are int8, every K chunk uses
explicit int8-by-int8 to int32 products, and chunks are assembled in
ascending `k_start` order with int64 accumulation. Raw integer values are
validated exactly.

Every branch is decoded before host reduction because branch scales can differ.
An explicit DAG `ReduceNode` reduces decoded complex64 branches in
producer-node-ID order. `host_reduce_s` measures only this reduction; it does
not include tile K assembly or output reconstruction.

The replay must validate every final work unit by tile ID, B/M/K/N extents,
byte estimates, MRAM footprint, and complete non-duplicate coverage. Hashing
is outside `total_wall_s`. Raw lane facts are keyed by
`node_id/stable_tile_id/lane`, where lane is `rr`, `ii`, `ri`, or `ir`. Each
fact records dtype, shape, and a hash of canonical little-endian bytes:
`<f4` for diagnostic float data and `<i4` for exact int data. Facts also
record encoded input-plane payload hashes, shared scales, and saturation
counts.

The authoritative measurement records `total_wall_s` plus `preparation_s`,
`encode_s`, `kernel_s`, `decode_s`, and `host_reduce_s`. H2D, D2H, session,
and energy fields remain null. `decode_s` includes K-chunk assembly, output
reconstruction, and complex decoding. `host_reduce_s` includes only explicit
DAG `ReduceNode` work.

Required backend facts are:

```text
backend_id: cpu_upmem_plan_replay_v1
execution_class: cpu_physical_plan_reference
physical_plan_id: <the consumed plan identity>
physical_plan_consumed: true
topology fields: requested DPU, tasklet, and rank values
hardware_execution: false
```

Expected preparation, encoding, kernel, decode, and reduction failures raise
`ExecutionFailed` at their corresponding phase. T7 must emit matching int8
raw lane hashes so exact physical validation can compare the same products;
this requirement does not change the `UpmemPlan` schema.

## Ordered Tasks

```text
T0    contracts, semantics, identities, baseline, dependency rules
T1A   UPMEM plan/tiling ownership
T1B   Python protocol/session split
T1C   self-contained native runtime
T1D   runtime coordinator move
T2    core model, circuits, lowering
T3    planner isolation
T4-0  dependency correction; contract frozen, implementation pending
T6A   pure complex encoding/decoding
T4A   results and CPU single-run API
T4C   final UpmemStage/UpmemPlan schema (complete)
T6B   CPU physical-plan replay (complete)
T7    complex UPMEM execution (next)
T4B1  UPMEM session API
T4B2  remove generic execution wrappers and migrate callers
T5A   evidence schemas and identities
T5B   experiment repetition/session lifecycle
T5C   timing normalization and old-emitter deletion
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
