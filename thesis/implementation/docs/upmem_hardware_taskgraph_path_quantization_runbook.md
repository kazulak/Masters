# ETH UPMEM One-DPU TaskGraph Path and Quantization Study

This runbook defines the physical one-DPU steady-state timing study. It
compares two fixed contraction paths and two numeric modes over the fixed
13-workload matrix. It is not a speedup, energy-efficiency, scaling, or
multi-DPU benchmark.

## Claim Boundary

The study measures the full TaskGraph execution in one persistent physical
DPU session per case. The session is reused across the two warmups and seven
timed repeats, so allocation, program load, and release are outside the
steady-state timing block. Host-side transfer, quantization, dequantization,
and DPU execution remain part of the recorded case timing according to the
study record contract.

The supported comparison is within this route: the two fixed paths and the
two numeric modes on the same one-DPU hardware profile. The results may
describe path or numeric-mode timing differences for this route. They must not
be converted into CPU/UPMEM or GPU/UPMEM speedup claims, energy claims, or
multi-DPU scaling claims. No energy measurement is defined by this study, and
`hardware_speedup_applicable=false` is intentional.

## Fixed Matrix

Every case runs with:

- one physical DPU and one tasklet;
- two warmups followed by seven timed repeats;
- path variants `opt_einsum_greedy` and `custom_upmem_v2_balanced`;
- numeric modes `none` and `per_task_input_quantize`;
- split real/imaginary float32 handling for complex inputs;
- 13 workloads: BV, XOR, EDC, and BB84 at 3, 4, and 5 qubits, plus
  `quantization_stress_2q_one_dpu`.

The persistent route is
`upmem_tn_hardware_taskgraph_persistent`, using
`generic_loop_interactive_session_v1`. Cases must retain their path, numeric
mode, warmup/repeat identity, session scope, and timing metadata in the
normalized records. The report only emits path or numeric-mode ratio rows
when each member has the complete timed repeat set `0` through `6`, matching
hardware/session/binary identity, and passed native validation. A partial or
failed run remains preserved in evidence but is not a completed comparison.

## ETH Procedure

Run from `thesis/implementation` on the ETH hardware host. The plan command
only resolves and builds the guarded study; it must not allocate a DPU. The
execution command requires explicit physical-hardware opt-in:

```bash
make upmem-hw-taskgraph-study-plan
UPMEM_ALLOW_PHYSICAL_HARDWARE=1 make upmem-hw-taskgraph-study
make UPMEM_HW_TASKGRAPH_STUDY_RUN=<run_dir> upmem-hw-taskgraph-study-report
```

Do not set `DPU_BACKEND` for the physical study. Do not substitute the
correctness-only `upmem-hw-taskgraph` route: it has a different session and
timing contract.

The report command consumes the exact saved run and does not rerun hardware.
`<run_dir>` is the extracted timestamped evidence directory, for example:

```text
runs/evidence/upmem_hardware_taskgraph_path_quantization/upmem_hw_taskgraph_study/<timestamp>/
```

## Evidence and Reports

Retain the complete run, including the resolved suite and plan, run and
environment manifests, hardware profile, native build/status records,
bounded stdout/stderr, per-case artifacts, and normalized records:

```text
build/upmem_hardware_taskgraph_study_plan/<timestamp>/
runs/evidence/upmem_hardware_taskgraph_path_quantization/upmem_hw_taskgraph_study/<timestamp>/
```

The report is derived under:

```text
runs/comparisons/research_pack/upmem_hw_taskgraph_study/<timestamp>/
```

Inspect `benchmark_summary.md`, `benchmark_manifest.json`,
`plot_manifest.json`, `tables/`, and `plots/`. Keep raw evidence under
`runs/inbox/eth/` or `runs/evidence/`; only a reviewed compact result belongs
under `thesis_results/`.

## Copying ETH Evidence

On ETH, archive the completed study run directory, not a source checkout:

```bash
RUN=$(readlink -f ~/work/Masters/thesis/implementation/runs/evidence/upmem_hardware_taskgraph_path_quantization/upmem_hw_taskgraph_study/latest)
tar -C "$(dirname "$RUN")" -czf ~/upmem_taskgraph_study_$(date -u +%Y-%m-%d_%H-%M-%S).tar.gz "$(basename "$RUN")"
```

On the local machine, from `thesis/implementation`, copy the archive into the
ignored inbox, inspect it, and extract it without changing its contents:

```bash
make evidence-inbox
scp safari-baguette1:~/upmem_taskgraph_study_<timestamp>.tar.gz runs/inbox/eth/
tar -tzf runs/inbox/eth/upmem_taskgraph_study_<timestamp>.tar.gz | head
mkdir -p runs/evidence/upmem_hardware_taskgraph_path_quantization/upmem_hw_taskgraph_study
tar -xzf runs/inbox/eth/upmem_taskgraph_study_<timestamp>.tar.gz \
  -C runs/evidence/upmem_hardware_taskgraph_path_quantization/upmem_hw_taskgraph_study
```

Verify that the extracted timestamped directory contains at least
`run_manifest.json`, `environment.json`, and `normalized_records.jsonl`, then
generate the report using the exact `<run_dir>`. Preserve the original archive
until the report and any reviewed snapshot have been verified.

## Multi-DPU Boundary

This study does not exercise multi-DPU allocation, task assignment,
inter-DPU communication, synchronization, reduction, or occupancy. Its
one-DPU timing cannot establish any of those properties. Before a separate
multi-DPU prototype, follow
[upmem_multi_dpu_prototype_readiness.md](upmem_multi_dpu_prototype_readiness.md)
and prove the prerequisites there, including dependency-safe assignment,
exactly-once task coverage, no CPU contraction fallback, explicit transfer and
synchronization timing, and final output validation.
