# UPMEM Tensor-Network Quantum Simulation

This directory is the active thesis implementation. Its research question is:

> Can quantum-circuit tensor-network contraction be mapped to UPMEM PIM in a
> way that is correct, measurable, and eventually faster on real DPU hardware?

The current code establishes the circuit-to-TaskGraph pipeline, serious CPU and
GPU baselines, bounded UPMEM SDK-simulator execution, and reproducible evidence.
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
target architecture. SimplePIM is physically qualified only for the bounded
management/operator lanes demonstrated by M4.2--M4.4. The shared descriptor-
driven M4.5 runtime is the active next phase; it has not yet passed physical
acceptance. PID-Comm, ATiM, and SparseP remain subsequent provider/kernel/
communication components behind thesis-owned interfaces, each for its own task
class rather than as a universal execution route.

Start with [ARCHITECTURE.md](ARCHITECTURE.md) for module ownership, external
provenance, thesis contributions, and the planned UPMEM architecture. The fixed
benchmark matrix is in [THESIS_BENCHMARK_MATRIX.md](THESIS_BENCHMARK_MATRIX.md).
The SLR-derived long-term implementation sequence and completion criteria are
in [docs/slr_architecture_implementation_roadmap.md](docs/slr_architecture_implementation_roadmap.md).

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

> The current UPMEM evidence is bounded to the documented generic simulator and
> separate physical M2--M4 qualification lanes. The descriptor-driven M4.5
> shared runtime, fully general distributed operation-aware execution, provider
> collectives, and hardware-calibrated UPMEM TN execution do not yet exist.

## Generated Cleanup

`make thesis-clean APPLY=1` retains the evidence selected by
`thesis_results/current`, its source research pack, and removes older generated
run directories. `make clean-generated` removes build/cache output and keeps
`runs/` unless `CLEAN_RUNS=1` is explicitly supplied.
