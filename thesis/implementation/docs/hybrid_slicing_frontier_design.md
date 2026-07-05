# Hybrid Slicing + Frontier Design

Wave 2E.55 verdict: do not claim hybrid evidence by placing Quimb slicing rows
beside internal TaskGraph frontier rows. At that point, true hybrid slicing plus
frontier execution had to be deferred until a shared slice-aware TaskGraph
representation existed.

Later waves added that diagnostic internal path:

- Wave 2E.56 added a model-only slice-aware internal TaskGraph representation.
- Wave 2E.57 added `cpu_tn_hybrid_sliced_frontier_exact`, an executed
  diagnostic route that performs internal slice reconstruction and frontier
  scheduling in one execution path on tiny deterministic cases.

This is a design note, not an implementation claim.

## Current Evidence

`quimb_tn_sliced_exact` provides executed Quimb/cotengra slicing evidence:

- `parallelism_mode=slicing`
- `parallelism_evidence_type=executed`
- `slicing_enabled=true`
- `slice_parallel_execution=false`
- `slice_worker_count=1`
- `slicing_flop_ratio` records sliced cotengra plan reported FLOPs divided by
  unsliced cotengra plan reported FLOPs.
- `slicing_reconstruction_status=completed`

`cpu_tn_frontier_exact` provides executed internal TaskGraph frontier evidence:

- `parallelism_mode=frontier`
- `parallelism_evidence_type=executed`
- `frontier_scheduler_enabled=true`
- `frontier_executed_task_count` records total completed TaskGraph
  contractions.
- `frontier_executed_parallel_task_count` records contractions dispatched in
  multi-task frontier waves.
- duplicate and missing dependency checks are recorded.

These are both useful diagnostics, but they are not one hybrid execution path.

`cpu_tn_hybrid_sliced_frontier_exact` provides executed diagnostic hybrid
evidence inside the internal TaskGraph family:

- `parallelism_mode=hybrid`
- `parallelism_evidence_type=executed`
- `hybrid_components=["slicing","frontier"]`
- `slicing_backend=internal_taskgraph`
- `slice_model_execution_status=executed`
- `slice_task_execution_mode=frontier_scheduled`
- `slice_reconstruction_status=completed`
- `hybrid_reconstruction_validation_status=passed`
- `frontier_scheduler_enabled=true`
- source task counts and expanded execution-node counts are recorded.
- duplicate, missing dependency, and dependency-violation checks are recorded.

This route remains diagnostic/internal. It is not a serious TN baseline, does
not replace `quimb_tn_exact`, and should not be used for speedup claims without
a separate same-family performance/scaling methodology.

## Feasibility Verdict

Do not implement a hybrid route by comparing `quimb_tn_sliced_exact` beside
`cpu_tn_frontier_exact`. That would be a fake hybrid: Quimb owns the slicing
tree and internal TaskGraph owns the frontier waves.

Original blockers before Wave 2E.56/2E.57:

- Internal TaskGraph has no slice-aware representation. `ContractionTask`
  records tensor IDs, labels, dependencies, and dense contraction metadata, but
  it does not carry `slice_id`, sliced index domains, per-slice reconstruction
  tasks, or slice aggregation metadata.
- The frontier executor schedules complete `ContractionTask` nodes. It cannot
  currently schedule slice subtasks or validate slice reconstruction.
- Quimb/cotengra exposes a sliced contraction tree with `nslices` and
  `sliced_inds`, but the current route does not expose TaskGraph-compatible
  frontier waves from that tree.
- Current Quimb slicing evidence is single-worker reconstruction
  (`slice_parallel_execution=false`), so it is not yet slice-worker parallel
  evidence.

## Rejected Shortcut

The following is not acceptable thesis evidence:

```text
quimb_tn_sliced_exact + cpu_tn_frontier_exact = hybrid
```

It compares two different implementation families. Reports may put the rows in
one diagnostic table, but they must not claim that one route executed both
slicing and frontier scheduling.

## Future Route Strategy

The additive diagnostic route now exists:

```text
cpu_tn_hybrid_sliced_frontier_exact
```

This route should remain diagnostic/internal until a separate benchmark
methodology proves correctness and fair timing beyond tiny diagnostic cases. It
should not replace `quimb_tn_exact` as the serious CPU TN baseline.

Required metadata for executed hybrid rows:

- `parallelism_mode=hybrid`
- `parallelism_evidence_type=executed`
- `hybrid_components=["slicing","frontier"]`
- `slicing_enabled=true`
- `frontier_scheduler_enabled=true`
- `slice_count`
- `slice_id` or equivalent per-slice task identity in task logs
- `frontier_wave_count`
- `max_frontier_width`
- `frontier_executed_task_count`
- `frontier_executed_parallel_task_count`
- `duplicate_contraction_check=passed`
- `missing_dependency_check=passed`
- `slicing_reconstruction_status=completed`
- `hybrid_reconstruction_validation_status=passed`

Required validation:

- every slice task executes exactly once;
- every frontier dependency is available before execution;
- no duplicate output reconstruction;
- final reconstructed output matches the unsliced exact baseline on small
  deterministic circuits;
- same-family timing comparisons only.

## Completed Follow-Up

Wave 2E.56 implemented the prerequisite representation:

1. Add a slice-aware TaskGraph design/prototype for internal CPU diagnostics.
2. Start with one deterministic sliced index on tiny circuits.
3. Emit slice task metadata first, with dry-run/model-only support if execution
   is risky.
4. Add sequential slice execution and reconstruction.
5. Add frontier-over-slices only after slice execution and reconstruction are
   independently validated.

Wave 2E.57 then added the diagnostic executed hybrid route. The remaining
parallelization milestone is not another fake hybrid comparison; it is GPU
tensor-network feasibility, tracked in [gpu_tn_feasibility.md](gpu_tn_feasibility.md).

## Continuing Non-Goals

- no Quimb execution changes;
- no UPMEM work;
- no GPU TN backend;
- no speedup claim.
