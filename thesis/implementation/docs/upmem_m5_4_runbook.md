# M5.4 Concurrent Packed-Int8 Runbook

M5.4 corrects two measured limitations of the historical M5 study:

- all selected DPUs are launched once per repetition with
  `dpu_launch(set, DPU_SYNCHRONOUS)`;
- the corrected integer route quantizes initial operands once on the host,
  transfers aligned packed `int8`, accumulates exact `int32` on the DPU, and
  dequantizes the final output on the host.

The historical `per_task_resident_requantize` suite remains replayable through
`make upmem-hw-m5`. M5.4 uses
`configs/suites/upmem_hardware_distributed_m5_4.yml` and does not silently fall
back to that route.

The accepted physical development run used source `eef42e4`, one explicitly
selected ETH rank, and DPU counts `1/2/4/8/16/32/64`. It produced 644 measured
rows, 48 explicit unsupported rows, and no failed rows. All ten criteria in
`m5_4_acceptance.json` passed. This run remains development evidence in the
ignored evidence inbox; it is not promoted automatically.

That accepted archive predates automatic suite retention and therefore relies
on source commit `eef42e4` for its committed suite. New runs retain and hash
both `config/source_suite.yml` and the effective
`config/resolved_suite.yml`, including DPU-count and tasklet overrides.

## Local or ETH preparation

From `thesis/implementation` on a clean checkout:

```bash
make doctor
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  ../.venv/bin/python -m pytest -q \
  tests/test_upmem_hardware_distributed_m5.py \
  tests/test_upmem_execution_plan_v3_adapter.py \
  tests/test_upmem_resident_native_source.py \
  tests/test_upmem_sdk_execution_plan_runner_v3.py \
  tests/test_upmem_m5_report.py
make upmem-hw-m5-4-plan
```

The plan command must report no allocation or launch. It builds and validates
both package transports through the native parser.

## Physical smoke gate

Select one physical rank explicitly and run only DPU counts `1/2/4/8`:

```bash
UPMEM_HW_RANK_PATH=/dev/dpu_rank1 \
UPMEM_ALLOW_PHYSICAL_HARDWARE=1 \
make upmem-hw-m5-4-smoke
```

Use the printed `run_dir` directly when generating the report:

```bash
UPMEM_HW_M5_4_RUN=<printed-run-dir> make upmem-hw-m5-4-report
```

Inspect `m5_4_acceptance.json` in the printed comparison directory. Do not run
the full matrix until the smoke report confirms:

- bulk set dispatch and zero explicit `dpu_sync` calls;
- exact packed-int8 `int32` agreement with zero mismatches;
- no CPU or simulator fallback;
- output partitions declare `output_slice_per_dpu`, while contracted
  partitions declare `final_reference_validation_only`;
- packed operand payload at most 30% of the float32 payload;
- maximum per-DPU cycles decrease with partition count;
- measured launch-wall scaling follows the maximum-cycle trend;
- `T8/T1 <= 0.35` for eligible strong-scaling diagnostics;
- weak-scaling launch time varies by at most `1.5x` for eligible diagnostics.

Weak-scaling admission also requires constant `mac_count / dpu_count` across
the compared rows. Changing semantic and plan hashes are expected when the
weak-scaling tensor shape grows, but changing per-DPU work is not.

Missing measurements remain `not_evaluated`; they never pass by inference.

## Full physical matrix

After the smoke gate passes:

```bash
UPMEM_HW_RANK_PATH=/dev/dpu_rank1 \
UPMEM_ALLOW_PHYSICAL_HARDWARE=1 \
make upmem-hw-m5-4

UPMEM_HW_M5_4_RUN=<printed-run-dir> make upmem-hw-m5-4-report
```

The run contains five workloads, float32 and host-packed-int8 modes,
output/contracted partitions, two warmups, and seven measured repetitions over
DPU counts `1/2/4/8/16/32/64`. Unsupported partition cells remain explicit.

For output partitioning, each DPU output slice is checksummed before host
assembly. For contracted-axis partitioning, each DPU writes a complete partial
output, so checksumming that full output on every DPU adds fixed work unrelated
to the partitioned contraction. The contracted path therefore emits a
completion acknowledgement for `final_reference_validation_only`; the host
still performs int64 or float64 reduction and mandatory exact/policy reference
validation of the reconstructed output.

If allocation fails before launch with a UFI message such as
`Timeout waiting for result to be correct`, retain that failed run and select a
known healthy rank explicitly for a new command. This is a new manual run, not
an automatic fallback; never rewrite the failed record or silently switch
ranks inside the benchmark.

## Claim boundary

M5.4 is a one-rank, single-contraction architecture experiment. It may support
same-route DPU-scaling and numeric-transport observations after its acceptance
checks pass. It is not a CPU/GPU speedup, energy, multi-rank, PID-Comm, general
distributed TaskGraph, or full tensor-network result.
