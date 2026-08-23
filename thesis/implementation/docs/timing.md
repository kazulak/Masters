# Timing Contract

Timing records observations; it does not infer overlapping phases from
independent component values.

## Measurement Fields

```python
@dataclass(frozen=True, slots=True)
class Measurement:
    scope_id: str
    total_wall_s: float
    lowering_s: float | None = None
    planning_s: float | None = None
    slicing_s: float | None = None
    mapping_s: float | None = None
    session_open_s: float | None = None
    encode_s: float | None = None
    h2d_s: float | None = None
    kernel_s: float | None = None
    host_reduce_s: float | None = None
    d2h_s: float | None = None
    decode_s: float | None = None
    rank_work_s: float | None = None
    h2d_bytes: int | None = None
    d2h_bytes: int | None = None
    energy_j: float | None = None
```

`total_wall_s` is measured once around the declared coordinator operation and
is authoritative. Components are observations and need not sum to it because
operations may overlap. `rank_work_s` is summed work, never wall time. An
unavailable value is `null`, not zero.

## Scopes

### `simulation_end_to_end_v1`

Start immediately before route-specific preparation of an already-created
`SimulationJob`. For TN routes include lowering, planning, slicing, DAG
construction, physical mapping, session opening, encoding, preparation,
transfers, kernels, host reductions, download, and decoding. For QuEST routes
include circuit translation, state allocation and initialization, circuit
execution, and query extraction. Stop when the requested decoded query result
is available.

Exclude configuration parsing, native compilation, environment setup, the
reference calculation, validation, hashing, evidence writing, and session
close. Session close is recorded separately.

### `steady_execution_v1`

Requires a reusable prepared context. The session is opened before warmups and
remains open for all warmups and measured samples. Start immediately before
input encoding and stop when the decoded result is available. Exclude planning,
mapping, session opening, session close, validation, hashing, and evidence
writing. A route that cannot provide this lifecycle returns unsupported rather
than measuring a different scope.

## Concurrent Ranks

The coordinator measures global wall time. It must not calculate a wall phase as
the maximum of independently reported rank phases unless a global phase was
explicitly measured around all ranks. Per-rank timings belong in backend facts;
`rank_work_s` is the sum of their elapsed durations.

## Comparison Rules

Only matching scope IDs may be compared directly. Ratios are defined as:

```text
float32/int8 speedup = float32 time / int8 time
unsliced/sliced speedup = unsliced time / sliced time
parallel speedup = one-active-DPU time / n-active-DPU time
tasklet speedup = one-tasklet time / n-tasklet time
```

Planning, native compilation, validation, hashing, and report generation must
not enter kernel timing. Energy remains `None` unless a declared measurement
boundary and sensor source provide it.
