# M2.3 Physical Two-DPU Study

This is a fixed functionality and bring-up study. It runs two deterministic
RY-H-RY circuits, two selected contraction paths, and two numeric modes on two
physical DPUs with one tasklet per DPU.

It does not establish speedup, scaling, concurrency, energy efficiency,
kernel-only timing, or planner optimality.

## Prepare on ETH

From thesis/implementation, after the project environment exists:

    make doctor
    make upmem-hw-m2-3-plan

The plan command builds and validates native package manifests. It must not
allocate or launch a DPU.

## Execute on ETH

    UPMEM_ALLOW_PHYSICAL_HARDWARE=1 make upmem-hw-m2-3

The command fails closed without the explicit hardware opt-in. It does not
retry with the simulator or CPU. The raw run is written below:

    runs/evidence/upmem_hardware_sliced_resident_m2_3/

Keep the complete timestamped run directory, including config/,
environment.json, normalized_records.jsonl, warmups.jsonl, run_manifest.json,
the summary JSON, and native-session artifacts.

## Copy evidence to the development machine

Copy the timestamped run directory into the ignored ETH inbox, for example:

    mkdir -p runs/inbox/eth/m2_3
    scp -r safari-baguette1:~/work/Masters/thesis/implementation/runs/evidence/upmem_hardware_sliced_resident_m2_3/upmem_hw_sliced_resident/YYYY-MM-DD_HH-MM-SS runs/inbox/eth/m2_3/

The report command does not read from runs/evidence on the development
machine unless explicitly configured. Point it at the copied inbox run:

    UPMEM_HW_M2_3_RUN=runs/inbox/eth/m2_3/YYYY-MM-DD_HH-MM-SS make upmem-hw-m2-3-report

Derived output is written only below:

    runs/comparisons/upmem_m2_3/<timestamp>/

The raw ETH inbox and generated comparisons are not thesis snapshots and are
not automatically promoted into tracked results.

## Admission contract

The strict reporter admits exactly:

- 8 passed warmups;
- 40 passed measured rows;
- 2 fixture IDs;
- 2 exact path variants;
- 2 numeric modes;
- measured repeat IDs 0 through 4.

It checks physical backend identity, two-DPU allocation, one tasklet per DPU,
completion and release evidence, package and reconstruction validation,
transfer-byte invariants, timing scope, and scientific validation. It also
checks that circuit and tensor-network identities are shared across paths,
plan hashes differ by path but remain stable across modes, and executor hashes
are shared within a mode but differ between modes.

The source completion contract is source-graph semantics:

    source_task_count = 3
    source_task_completion_count = 3
    source_task_completion_scope = unique_source_tasks_completed_on_every_slice
    expanded_task_count = 6
    expanded_task_completion_count = 6
    completed_task_count = 6  # compatibility alias only
    slice_model_task_count = 2
    slice_model_executed_task_count = 2
    slice_model_task_count_scope = slice_descriptors
    slice_model_executed_task_count_scope = completed_slice_descriptors

If a run reports the replicated physical total as source completion, the
report is rejected and the retained validation_rows.csv identifies the
inconsistent fields. Do not repair that by editing the evidence manually.

## Report outputs

Successful reports contain:

- combination_statistics.csv;
- paired_numeric_mode_ratios.csv;
- paired_path_ratios.csv;
- validation_rows.csv;
- runtime_by_circuit_path_mode.png;
- quantization_error_by_circuit_path.png;
- timing_ratios.png;
- benchmark_summary.md;
- report_manifest.json.

The primary timing is the native response's `stage_timings.total_route_time_s`,
labelled host-observed native SDK stage time. Python `total_time_s` is retained
as secondary end-to-end orchestration timing. Neither is kernel-only timing. A
ratio above 1.0 means that the numerator was slower in the matched run; these
values are not speedups.
The custom path is a modeled generic_single_dpu_split_complex_v2 candidate
executed on the two-DPU sliced-resident route. It is not calibrated planner
evidence.

## Failure handling

On missing, malformed, incomplete, or semantically inconsistent evidence, the
reporter creates a comparison directory containing validation_rows.csv,
benchmark_summary.md, and a rejected report_manifest.json. It creates no
plots or statistics. Preserve that rejected directory when diagnosing a run.
