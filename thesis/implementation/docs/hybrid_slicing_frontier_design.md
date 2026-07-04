# Hybrid Slicing + Frontier Design

Wave 2E.55 verdict: true hybrid slicing plus frontier execution should be
deferred. The current implementation has executed slicing evidence and executed
frontier evidence, but they are in different execution families and do not share
a task representation.

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

## Feasibility Verdict

Do not implement a hybrid route by comparing `quimb_tn_sliced_exact` beside
`cpu_tn_frontier_exact`. That would be a fake hybrid: Quimb owns the slicing
tree and internal TaskGraph owns the frontier waves.

Current blockers:

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

Use an additive diagnostic route only after a shared execution representation
exists. Candidate future route ID:

```text
cpu_tn_hybrid_sliced_frontier_exact
```

This route should remain diagnostic/internal until it proves correctness and
fair timing. It should not replace `quimb_tn_exact` as the serious CPU TN
baseline.

Required metadata for a future executed hybrid route:

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

## Recommended Next Wave

Wave 2E.56 should implement the prerequisite representation, not a full hybrid
route:

1. Add a slice-aware TaskGraph design/prototype for internal CPU diagnostics.
2. Start with one deterministic sliced index on tiny circuits.
3. Emit slice task metadata first, with dry-run/model-only support if execution
   is risky.
4. Add sequential slice execution and reconstruction.
5. Add frontier-over-slices only after slice execution and reconstruction are
   independently validated.

Only after Wave 2E.56 proves slice-aware TaskGraph reconstruction should a
hybrid frontier scheduler be implemented.

## Non-Goals For 2E.55

- no hybrid route implementation;
- no Quimb execution changes;
- no frontier execution changes;
- no UPMEM work;
- no GPU TN backend;
- no speedup claim.
