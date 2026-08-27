# Timing Contract

Timing records observations. It does not infer overlapping phases from
independent component values.

## Measurement

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
    preparation_s: float | None = None
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
is authoritative. Components need not sum to it because work may overlap.
`rank_work_s` is the sum of rank durations and is never wall time. The current
runtime leaves `rank_work_s` null unless it directly measures the declared
quantity; it must never be inferred from per-rank values. Unavailable values
are `null`, not zero. Bytes are non-negative integers; times and energy are
finite non-negative values.

Native counter capture needed by the coordinator is inside the timer. Canonical
fact normalization, output hashing, validation, evidence serialization and
artifact writing are after the timer. `session_close_s` is session evidence,
not a per-sample measurement field.

## Scopes

### `simulation_end_to_end_v1`

Start immediately before route-specific preparation of an existing
`SimulationJob`. TN routes include lowering, path planning, slicing, DAG
construction, UPMEM mapping, session opening, encoding, preparation, transfers,
kernels, host reduction, download and decoding. QuEST routes include circuit
translation, state allocation/initialization, circuit execution and query
extraction. Stop when the decoded requested result is available.

Exclude configuration parsing, native compilation, environment setup, reference
calculation, validation, hashing, evidence writing and session close.

Current support is limited to direct Quimb/cotengra and QuEST routes. NumPy
same-DAG and UPMEM routes do not emit this scope because logical or physical
preparation currently happens before their repetition loops.

### `steady_execution_v1`

Requires a reusable prepared context. A session is opened before an attempt and
remains open through that attempt. Start before input encoding and stop after
decoded output. Include encode, preparation, transfers, kernels, host reduction
and decode. Exclude planning, mapping, session open/close, validation, hashing
and evidence writing. A route unable to provide this lifecycle is unsupported
for this scope.

The active collection policy is `fresh_session_per_attempt_v1`: each UPMEM
attempt opens its session before this scope, executes exactly one warmup or
measurement attempt, then closes after this scope. Warmup blocks warm the
machine and software path, not a persistent session reused by later
measurements. A persistent-session campaign is a different collection policy
and cannot be pooled with these samples.

Current NumPy same-DAG and UPMEM session routes emit this scope. Reports reject
pairing it with `simulation_end_to_end_v1`.

## Concurrency and Comparisons

The coordinator measures global wall time. It must not use the maximum of
independent rank phase timings as wall time unless a global phase was measured.
Per-rank values belong in backend facts; `rank_work_s` remains summed work.
For a one-rank UPMEM session, `h2d_s`, `kernel_s`, and `d2h_s` are the sums of
the native per-operation phase counters. For multi-rank sessions those fields
remain `null`; independently timed rank phases are not inferred to be global
wall-clock phases.

UPMEM operation backend facts retain two additional request-lifecycle values.
`request_wave_wall_sum_s` is the sum of coordinator wall-clock durations around
each sequential request wave. `rank_response_total_route_max_sum_s` is the sum
of the maximum native `total_route_time_s` reported by the ranks in each wave.
Both values are nested and inclusive: the native route value includes its H2D,
kernel, and D2H counters, while the coordinator wave value includes the native
route value. They are not `Measurement` phases and must not be added directly
to the named phase counters. Attribution analysis derives disjoint host and
native request overhead from these values.

For one-rank M7F attribution, operation facts additionally retain coarse host
request-wave boundaries. `request_build_sum_s` covers coordinator validation and
request-artifact construction, including staged payload and manifest writes.
`rank_submit_parallel_wall_sum_s` surrounds the parallel rank-client submit phase.
The `rank_submit_*_max_sum_s` fields retain the maximum per-rank client durations
for artifact validation, protocol write, response wait/JSON parse, response
validation, and their total. `coordinator_response_processing_sum_s` covers
response accounting and output-file reads after rank submits complete. These are
backend facts, not `Measurement` phases. One-rank attribution subtracts the nested
native route time from response wait before adding native H2D, kernel, and D2H
components, so the resulting components remain disjoint. Session startup remains
outside `steady_execution_v1` and is not measured by this split.

Direct ratios are defined only for matching scopes and compatible identities:

```text
float32/int8 speedup = float32 time / int8 time
unsliced/sliced speedup = unsliced time / sliced time
parallel speedup = one-active-DPU time / n-active-DPU time
tasklet speedup = one-tasklet time / n-tasklet time
```

Planning, native compilation, validation, hashing and report generation never
enter kernel timing. `energy_j` stays null unless a declared sensor source,
measurement boundary and interval provide a measured value. Timing evidence
does not by itself establish physical qualification or speedup.

For the physical campaign configuration, two deterministic warmup blocks are
excluded from statistics and thirty measured blocks are planned per route. The
report retains all attempted outcomes, uses successful measurements for the
median and raw MAD, and reports a 95% percentile-bootstrap interval. It makes
no post-hoc outlier exclusion. A speedup interval uses block-paired bootstrap
resampling only after every planned measurement and provenance gate passes.
