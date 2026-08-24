# Internal/Research M2 Sliced-Resident MVP

This is a bounded internal/research command for the committed M2 fixture. It
is not a public or publishable benchmark workflow and makes no speedup or
energy claim.

- route: `upmem_tn_hardware_sliced_resident_two_dpu`
- backend: `upmem_sdk_hardware_sliced_resident_two_dpu`
- profile: `hardware_sliced_resident_two_dpu_m2_v1`
- fixed shape: two slices, two physical DPUs, one tasklet per DPU, numeric mode
  `none`

The command accepts only the canonical repository file
`thesis/implementation/configs/suites/upmem_hardware_sliced_resident_mvp.yml`.
Copied, renamed, or alternate YAML files are rejected even if their contents
match. Its fixed workload IDs and QASM paths are part of the contract: three
one-qubit workloads, one warmup, and three measured repeats. Each workload must
produce a greedy TaskGraph with one terminal contraction of dimension two.

## Prepare

```text
make upmem-hw-sliced-resident-plan
```

This builds the separate native sources, writes the two restricted packages and
their preflight manifests for each warmup/repeat, and never allocates or launches
a DPU. The same command can be used without `--build` to create unbuilt package
plans:

```text
PYTHONPATH=src python -m quantum_bench.bench \
  upmem-hardware-sliced-resident-mvp \
  --suite configs/suites/upmem_hardware_sliced_resident_mvp.yml \
  --prepare-only
```

`--build` is valid only with `--prepare-only`.

## Execute

```text
UPMEM_ALLOW_PHYSICAL_HARDWARE=1 make upmem-hw-sliced-resident
```

Execution is explicit opt-in. Simulator selectors are rejected by the committed
session adapter. One native session is built for the run. For every warmup and
measured repeat, the runner materializes exactly two independent restricted
packages, preflights their bytes, invokes the adapter once, and reconstructs the
result from the two native float32 partial-output files. There is no retry, CPU
fallback, or alternate execution route.

Evidence is written below:

```text
runs/evidence/upmem_hardware_sliced_resident_mvp/upmem_hw_sliced_resident/<timestamp>/
```

The directory contains the resolved suite, environment, run manifest, measured
`normalized_records.jsonl`, `warmups.jsonl`, summary JSON, and per-operation
slice plans, copied manifests, native response, package preflight, and
reconstructed output. Warmups remain identifiable but are excluded from the
measured row count, which is exactly nine on a successful run.

Measured records retain route/backend identity, circuit and package source
hashes, topology and allocation/launch/synchronization/release evidence,
timings, CPU and expected-output validation, native binary hashes, failure
stage, and explicit `not_applicable` speedup and energy claims. Transfer
accounting includes application-visible and actual H2D/D2H/total fields, with
each total equal to H2D plus D2H. Failed or unsupported operations retain their
relative response/manifest paths and available native command, stdout, stderr,
and failure evidence; do not delete them when reviewing evidence.

## M2.1 useful-slice fixture

The next fixture is kept separate from the historical M2 control fixture. It is
the canonical:

```text
configs/suites/upmem_hardware_sliced_resident_m2_1.yml
configs/circuits/upmem_m2/one_qubit_hx.qasm
```

It contains exactly one one-qubit circuit with the explicit gate sequence
`h q[0]; x q[0];`. The expected TaskGraph has three source tensors, two
contraction tasks, and a dependency from the H task to the X task. The sliced
edge is the internal H-to-X contraction index. Both CPU slice references are
nonzero and their sum equals the unsliced CPU result. Each physical package
must execute both source tasks; an intermediate `result_0` must not be supplied
as an initial package input.

The ETH commands are:

```text
make upmem-hw-m2-1-plan

UPMEM_ALLOW_PHYSICAL_HARDWARE=1 \
make upmem-hw-m2-1
```

M2.1 uses a graph-wide internal H-to-X slice restriction. Each DPU receives
the complete two-operation graph and computes its own prefix; the host only
sums the two final partial outputs. The native host reads a DPU-written
completion sentinel after every blocking synchronization. A successful M2.1
row therefore requires two sentinel-verified operations per DPU, nonzero
per-slice output, matching native and planned transfer totals, and final
reconstruction validation.

The acceptance remains one warmup and three measured repeats, exactly two
physical DPUs, one tasklet per DPU, both nonzero per-slice references, full
two-task execution on both DPUs, successful reconstruction, and no speedup,
scaling, or energy claim.

## 2026-08-02 ETH Result

The first physical run completed all three warmups and all nine measured rows
with two allocated DPUs, one asynchronous set launch, one synchronization,
successful release, no fallback, and correct reconstruction. The run is a
physical control-path pass.

It is not yet a useful-work pass: all slice-1 partials are zero because these
single-gate circuits slice the input-state index of the fixed `|0>` state. The
normalized rows also inherit several generic default fields that contradict the
native slice evidence. Do not use those rows for a parallelism or speedup plot.
The complete findings and M2.1 acceptance plan are in the
[ETH evidence audit](upmem_m2_eth_evidence_analysis.md).
