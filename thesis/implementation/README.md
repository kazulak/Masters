# UPMEM Tensor-Network Quantum Simulation

This directory is the active thesis implementation. Its research question is:

> Can quantum-circuit tensor-network contraction be mapped to UPMEM PIM in a
> way that is correct, measurable, and eventually faster on real DPU hardware?

The current code establishes the circuit-to-TaskGraph pipeline, serious CPU and
GPU baselines, bounded UPMEM SDK-simulator execution, physically validated
M4/M5 development routes on ETH hardware, and a bounded physical M5
execution-plan-v3 development acceptance. The active M5 route is a one-rank,
multi-DPU, single-contraction study; it is not the full distributed TaskGraph
architecture.
It does **not** yet claim UPMEM hardware speedup or a fully general UPMEM tensor
contraction kernel.

The active physical evidence is a sequence of bounded qualification lanes. M2.1
useful-slice execution, M2.2 float32/requantized execution, M2.3 two-path/two-
numeric-mode execution, M3.1 two-wave frontier dispatch, and M4.2--M4.4
SimplePIM lanes have passed their declared physical functionality checks on ETH.
These are separate fixtures and adapters, not one general executor. The M2
sliced-resident foundation remains documented in the [M2 ETH runbook](docs/upmem_hardware_sliced_resident_mvp_runbook.md),
and the M4 lanes are documented in their individual runbooks. None of these
results supports a speedup, energy, scaling, or general-TaskGraph claim.

The previous one-DPU MRAM-resident route remains readable as historical context.
Dense, legacy generic/persistent, CPU frontier/hybrid, PIM bridge/frontier, and
external-library probe artifacts are historical only. Their normalized readers,
route labels, tables, plots, and snapshot compatibility remain available for
existing evidence.

The historical status applies to those old runnable experiments, not to the
target architecture. SimplePIM is physically qualified for the bounded
management/operator lanes demonstrated by M4.2--M4.4. M4.5 is physically
accepted on ETH for its bounded functionality contract: one resident package
is used with separate one-DPU and two-DPU schedules, producing 3 and 2 waves;
SimplePIM handles management/allocation, the thesis resident kernel handles
contraction, and the host performs the one two-DPU handoff. Each session
validates one final output against the CPU reference, with no simulator or CPU
fallback. The tracked M4.5 evidence capsule is
[thesis_results/physical_simplepim_taskgraph_m4_5](thesis_results/physical_simplepim_taskgraph_m4_5).
The route remains functionality evidence only: it makes no timing, speedup,
scaling, energy, or general tensor-network performance claim. PID-Comm, ATiM,
and SparseP remain subsequent provider/kernel/communication components behind
thesis-owned interfaces. The later M4.6, M5.1, and M5.2 observations below are
audited development runs copied from ETH, not promoted or tracked thesis
results. The additive M5 execution-plan-v3 lane is separate and has passed its
bounded physical development-acceptance checks on the audited ETH run. Its
evidence remains in ignored `runs/` and is not promoted or tracked as thesis
results.
For M5 v3, SimplePIM's role is
`initialization_binary_and_management_state_only`. Allocation, transfer, and
launch use the thesis-owned raw synchronous UPMEM SDK route. The thesis-owned C
kernel performs the contraction and the host performs the `float64` reduction;
none of these are SimplePIM compute operators.

Start with [ARCHITECTURE.md](ARCHITECTURE.md) for module ownership, external
provenance, thesis contributions, and the planned UPMEM architecture. The fixed
benchmark matrix is in [THESIS_BENCHMARK_MATRIX.md](THESIS_BENCHMARK_MATRIX.md).
The SLR-derived long-term implementation sequence and completion criteria are
in [docs/slr_architecture_implementation_roadmap.md](docs/slr_architecture_implementation_roadmap.md).
The exact recent ETH observations are consolidated in the
[M4/M5 physical development acceptance record](docs/m4_m5_physical_acceptance.md);
they are not promoted thesis results.

## Current Milestone Status

This is the authoritative current status table. SimplePIM remains central to
the implementation, and M4.5 is the current accepted SimplePIM-managed
baseline.

| Milestone | Current status | Boundary or next gate |
| --- | --- | --- |
| M4.1 | Physically accepted | Bounded physical qualification; no general executor or performance claim. |
| M4.2 | Physically accepted | SimplePIM rank-1 operator qualification. |
| M4.3 | Physically accepted | TaskGraph-derived SimplePIM operand adapter. |
| M4.4 | Physically accepted | Bounded persistent SimplePIM-managed operator chain. |
| M4.5 | Physically accepted; current baseline | SimplePIM-managed descriptor-driven shared runtime with bounded one- and two-DPU functionality. |
| M4.6 | Physically validated development run | One physical DPU, tasklets `1/2/4/8/16`, 12 small circuit cases, two path variants, two numeric modes, and 7 repeats per configuration. All 1680 rows passed validation; functionality and diagnostic tasklet evidence only. |
| M5.1 | Physically validated bounded probe | One bounded real `float32` contraction on 1/2/4 DPUs with exclusive output-tile ownership. Exact CPU agreement; SimplePIM management plus thesis-owned kernel; one repetition and zero warmups; functionality only. |
| M5.2 | Physically validated bounded probe | The same contraction on 1/2/4 DPUs with contracted-axis partials and deterministic `host_mediated_sum_v1` reduction. Maximum absolute error `2.98e-08`; one repetition and zero warmups; functionality only. |
| M5 execution-plan-v3 | Physically accepted bounded development study | One-rank, one selected ETH rank, DPU counts 1/2/4/8/16/32/64, tasklets 8, 5 workloads, float32/int8 modes, output/contracted partitions, 2 warmups and 7 measured repeats. The 140-cell matrix produced 644 measured rows and 48 partition-incompatible unsupported rows, with 0 failures. Same-route diagnostics only; no broad performance claim. |
| M5.3 | Blocked before physical execution | PID-Comm compile/link qualification is blocked under ETH SDK 2023.1 by missing `dpu_alloc_comm`, `DPU_FOREACH_ENTANGLED_GROUP`, and old PID-Comm API/source macros. No fallback and no physical PID-Comm execution. |

M4.1--M5.2 are bounded physical functionality milestones, not a claim of
complete M4/M5 architecture, general distributed TN execution, performance,
speedup, energy, or scaling. The M4.6 development sweep showed a small-workload
tasklet optimum near 8 tasklets with lower efficiency at 16; this observation is
not a final benchmark result. M5 execution-plan-v3 is an additive physical
development study; its measured ratios are descriptive within-route evidence
and do not establish general physical performance or acceleration.

Physical ETH runs require an explicit rank selection. Use a healthy rank chosen
on the server, for example:

```bash
UPMEM_HW_RANK_PATH=/dev/dpu_rank1 \
UPMEM_ALLOW_PHYSICAL_HARDWARE=1 make upmem-hw-m4-6-tasklet-scaling
UPMEM_HW_RANK_PATH=/dev/dpu_rank1 \
UPMEM_ALLOW_PHYSICAL_HARDWARE=1 make upmem-hw-m5-1
UPMEM_HW_RANK_PATH=/dev/dpu_rank1 \
UPMEM_ALLOW_PHYSICAL_HARDWARE=1 make upmem-hw-m5-2

# Additive M5 execution-plan-v3 physical development study.
UPMEM_HW_RANK_PATH=/dev/dpu_rank1 \
UPMEM_ALLOW_PHYSICAL_HARDWARE=1 make upmem-hw-m5
```

The exact hardware-free preparation check for the new lane is:

```bash
UPMEM_HW_M5_DPU_COUNTS=3 UPMEM_HW_M5_TASKLETS=3 make upmem-hw-m5-plan
```

It prepares the configured plan set, preserves unsupported cases, reports
failures explicitly, and performs no DPU allocation or launch. The audited
source revision is `5401597fdc2458087e112f5bd2e1869a5a0a5ab0` with a clean
worktree. The ETH run was
`runs/evidence/upmem_hardware_distributed_m5/upmem_hw_m5/2026-08-13_15-01-11`;
the local ignored copy is
`runs/inbox/eth/m5_v3/canonical-2026-08-13_15-01-11`, and the fixed report is
`runs/comparisons/upmem_m5/2026-08-13_16-29-36_450221`. The normalized-record
hash is `1a7714b8dce25b0b0959ed08cae73aaf47e6d7084b90200d4895bf4c521202a0` and
the suite hash is `e71ec4518a99a8c7f463926da845b1c67bef7242c72233c0c0cfdc107177e26c`.
All physical/provider/rank/allocation/kernel/release/no-fallback/transfer/
validation checks passed; the report is complete and all nine plots and table
hashes are valid. This is development acceptance only, not a promoted result.

The runner records the requested rank path and effective SDK profile. These
fields document selection, not an independent observation of the physical rank.
On the 2026-08-11 ETH host, `/dev/dpu_rank0` failed vendor diagnostics while
ranks 1, 20, and 39 passed; this is an environment observation, not a software
performance conclusion.

## Setup

From `thesis/implementation`:

```bash
# Requires uv. It creates/reuses ../.venv with .python-version, installs
# constrained dev dependencies, initializes submodules, then runs doctor.
make setup
```

`make setup` refuses to replace an existing incompatible environment. Its
diagnostic prints the exact environment path; remove that path explicitly only
when you intend to rebuild it.

Run tests:

```bash
make test
```

## Thesis Workflow

The Makefile is the public execution surface.

```bash
# 1. Run the complete local research matrix.
# Use the physical-core count printed by `make doctor`.
BENCH_CPU_THREADS=6 make thesis-run

# 2. Inspect generated evidence and comparisons.
make list-runs

# 3. Reserved for the post-M9 reviewed evidence freeze; do not promote
#    development evidence during M0--M8.
make thesis-promote

# 4. Verify or regenerate tables and plots without rerunning benchmarks.
make thesis-verify
make thesis-report

# 5. Preview stale generated runs, then remove them after promotion succeeds.
make thesis-clean
make thesis-clean APPLY=1

# Optional immutable named thesis milestone.
make thesis-release NAME=upmem-baseline-v1
```

`make thesis-run` requires an explicit physical-core count, fixes the OpenMP,
OpenBLAS, MKL, and NumExpr thread settings for every subprocess, runs fixed
suite files, and saves every execution automatically. No run path needs to be
copied manually.

The long run includes:

- QuEST CPU/GPU correctness evidence;
- QuEST CPU/GPU 8--20q performance evidence across seven sizes;
- QuEST CPU plus Quimb unsliced/sliced 8--20q CPU TN evidence;
- same-path float64/int8 internal TaskGraph replay for quantization attribution;
- modeled `opt_einsum` contraction-path candidates with UPMEM pressure scores;
- strict generic UPMEM SDK-simulator float32/int8 boundary evidence;
- modeled TaskGraph scheduling and multi-DPU planning primitives, clearly
  excluded from executed serious baselines.

The separate M1 provider qualification workflow is documented in the
[UPMEM provider qualification runbook](docs/upmem_provider_qualification_runbook.md).
The SimplePIM physical probe passed its bounded functionality contract. The
broader M1 gate remains incomplete because PID-Comm, ATiM, and SparseP have not
yet passed their provider-specific physical qualification lanes.

GPU execution requires a GPU-visible shell. On the local AMD machine the route
uses QuEST HIP and verifies a real HIP program before emitting GPU rows. On a
future NVIDIA cluster the GPU software/build route must be adapted and verified
there; no row is created merely because CUDA support exists in source code.

## Results Layout

Generated runs use readable local timestamps:

```text
runs/
  inbox/eth/<experiment-id>/              # copied raw ETH archives; ignored
  evidence/<suite>/<route>/2026-07-10_18-30-00/
  comparisons/research_pack/2026-07-10_19-15-00/
```

Each suite and route directory has a `latest` link. `runs/latest` points to the
latest evidence run only. These generated directories are ignored by Git.

The selected thesis result is compact and tracked:

```text
thesis_results/
  current/
    README.md                 # generated interpretation and claim limits
    snapshot_manifest.json   # selected run IDs, commits, and checksums
    evidence/                 # normalized records and compact manifests
    suites/                   # exact resolved suite files
    tables/                   # source CSVs for every figure
    plots/                    # human-readable figures
  releases/<name>/            # optional immutable milestones
```

The broad `thesis_results/current` snapshot is historical and does not contain
the M5 v3 development run.

Reports are regenerated from `normalized_records.jsonl`; benchmark execution is
not repeated. Generated plots and comparison tables never belong in
`runs/evidence`.

## Importing ETH Evidence

Raw results copied from the ETH UPMEM host belong in the ignored
`runs/inbox/eth/` staging area, not beside source code. Create it with:

```bash
make evidence-inbox
```

Keep the archive there, extract the validated evidence run under
`runs/evidence/`, then regenerate a report from that exact run. During M0--M8,
copied archives and generated development evidence remain ignored and are not
promoted. Only after M9, the source commit, normalized rows, and report have
been reviewed should a compact named snapshot be promoted into
`thesis_results/`. The complete copy, audit, and promotion procedure is in
[the evidence workflow](docs/evidence_workflow.md).

## Individual Commands

The full matrix is preferred, but focused commands remain useful:

```bash
make build-quest-cpu
make bench-cpu
make bench-gpu
make bench-upmem-sim
make upmem-hw-sliced-resident-plan
make upmem-hw-sliced-resident
make upmem-hw-m5-plan
make upmem-hw-m5
make planner-report
make research-plan
make upmem-provider-plan
```

On the ETH hardware host, use the exact preparation/execution commands and
acceptance fields in the [M2 ETH runbook](docs/upmem_hardware_sliced_resident_mvp_runbook.md).
Acceptance is manual and outside pytest/CI. The runbook is the source of truth
for the normalized evidence workflow; the M2 route has no CPU/simulator retry.

For M1 SimplePIM qualification, prepare locally with
`make upmem-provider-plan`, then on a clean ETH checkout run
`UPMEM_ALLOW_PHYSICAL_HARDWARE=1 make upmem-provider-qualify PROVIDER=simplepim`.
This is a one-DPU, 12-tasklet, 256-`uint32` virtual-array map/zip functionality
probe only. It has no simulator or fallback path and makes no performance
claim. Artifacts are stored under
`runs/evidence/provider_qualification/simplepim/`; admission requires the
passed fields and fingerprints in the [runbook](docs/upmem_provider_qualification_runbook.md).
That broader M1 catalog gate remains separate from the passed M4 SimplePIM
lanes. PID-Comm is a separate 2021.3.0/AVX512/1024-DPU lane; ATiM's official
artifact and SparseP source remain unpinned and blocked.

The retired dense and legacy TaskGraph runs remain readable as historical
evidence, but are no longer runnable through the public CLI or Makefile. The
bounded physical routes use explicit SDK/SimplePIM contracts and require
`UPMEM_ALLOW_PHYSICAL_HARDWARE=1`; it rejects simulator selectors and has no
CPU fallback. Its exact allocation, launch, synchronization, reconstruction,
validation, and normalized-record acceptance fields are defined by the
[M2 ETH runbook](docs/upmem_hardware_sliced_resident_mvp_runbook.md).

The active simulator command is strict generic-only UPMEM SDK evidence. It
records bounded TaskGraph validation and simulator timing, never hardware
timing or hardware speedup. Older dense and bridge artifacts remain readable
through the normalized report/snapshot readers only.

Modeled planner evidence has its own comparison namespace:

```bash
make planner-evidence
make planner-report
```

These commands write under `runs/comparisons/planner_v2/`; their scores are
model-based path-selection hypotheses, not hardware-calibrated timing.

For a specific research group:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src ../.venv/bin/python \
  scripts/research_benchmark_pack.py run --full --suite cpu_tn
```

Valid group names are printed by `make research-plan`.

## Evidence Rules

- `quest_cpu_full_state_exact`: serious CPU full-state baseline.
- `quest_gpu_full_state_exact`: serious full-state GPU comparison only after
  verified GPU execution; it is not a GPU TN route.
- `quimb_tn_exact`: serious external CPU TN baseline.
- `quimb_tn_sliced_exact`: explicit Quimb/cotengra slicing evidence.
- historical internal frontier/hybrid routes: readable diagnostics only; they
  are not active benchmark commands.
- strict UPMEM generic SDK-simulator rows: code-path, boundary, transfer, and
  accuracy evidence; never hardware speedup.

TaskGraph-based CPU and UPMEM records carry
`circuit_semantics_hash`, `tensor_network_hash`, and
`contraction_plan_hash`. A CPU/UPMEM execution comparison is called same-plan
only when those hashes match. Planning time is recorded separately from route
execution time.

Evidence provenance is stage-specific. `benchmark_source_*` fields identify
the code that executed the workload, `report_generation_*` fields identify the
revision that derived tables and plots, and `snapshot_promotion_*` fields
identify the clean base revision used to select tracked evidence. Repository
dirtiness outside `thesis/implementation` is recorded separately and does not
silently change benchmark-source cleanliness.

Current limitation:

> The current UPMEM evidence is bounded to the documented generic simulator,
> physical M2--M4 qualification lanes, the accepted M4.5 shared runtime, and
> the accepted M5 v3 one-rank single-contraction development study. M5 v3
> provides same-route numeric, partition, transfer, accuracy, and bounded
> DPU-count diagnostics: `T_float32/T_int8` has median `0.165` (range
> `0.109--0.799`), `T_output/T_contracted` has median `0.980` (range
> `0.492--1.092`), and same-route `T1/TN` has median `0.634` (range
> `0.073--0.999`). No broad hardware speedup, energy, general distributed
> TaskGraph, PID-Comm, ATiM, SparseP, multi-rank, or planner-superiority claim
> is supported. Float32 maximum error was `7.15e-06` against a `1e-05`
> threshold; int8 error `0.0303011` is descriptive.

## Generated Cleanup

`make thesis-clean APPLY=1` retains the evidence selected by
`thesis_results/current`, its source research pack, and removes older generated
run directories. `make clean-generated` removes build/cache output and keeps
`runs/` unless `CLEAN_RUNS=1` is explicitly supplied.
