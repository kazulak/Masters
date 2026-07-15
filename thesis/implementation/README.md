# UPMEM Tensor-Network Quantum Simulation

This directory is the active thesis implementation. Its research question is:

> Can quantum-circuit tensor-network contraction be mapped to UPMEM PIM in a
> way that is correct, measurable, and eventually faster on real DPU hardware?

The current code establishes the circuit-to-TaskGraph pipeline, serious CPU and
GPU baselines, bounded UPMEM SDK-simulator execution, and reproducible evidence.
It does **not** yet claim UPMEM hardware speedup or a fully general UPMEM tensor
contraction kernel.

The first physical one-DPU dense functionality run is documented separately in
[docs/upmem_hardware_mvp_runbook.md](docs/upmem_hardware_mvp_runbook.md). It is
hardware correctness evidence only and is not part of the default thesis matrix.

Start with [ARCHITECTURE.md](ARCHITECTURE.md) for module ownership, external
provenance, thesis contributions, and the planned UPMEM architecture. The fixed
benchmark matrix is in [THESIS_BENCHMARK_MATRIX.md](THESIS_BENCHMARK_MATRIX.md).

## Setup

From `thesis/implementation`:

```bash
python3 -m venv ../.venv
../.venv/bin/python -m pip install --upgrade pip
git submodule update --init --recursive external/QuEST external/SimplePIM external/PID-Comm
../.venv/bin/python -m pip install -c ci/constraints.txt -e ".[dev]"
make doctor
```

Run tests:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src ../.venv/bin/python -m pytest -q
```

## Thesis Workflow

The Makefile is the public execution surface.

```bash
# 1. Run the complete local research matrix.
# Use the physical-core count printed by `make doctor`.
BENCH_CPU_THREADS=6 make thesis-run

# 2. Inspect generated evidence and comparisons.
make list-runs

# 3. Promote the latest successful pack into tracked, human-readable results.
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
- internal parallelism diagnostics, clearly excluded from serious baselines.

GPU execution requires a GPU-visible shell. On the local AMD machine the route
uses QuEST HIP and verifies a real HIP program before emitting GPU rows. On a
future NVIDIA cluster the GPU software/build route must be adapted and verified
there; no row is created merely because CUDA support exists in source code.

## Results Layout

Generated runs use readable local timestamps:

```text
runs/
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

## Individual Commands

The full matrix is preferred, but focused commands remain useful:

```bash
make build-quest-cpu
make bench-cpu
make bench-gpu
make bench-upmem-sim
make upmem-hw-mvp-plan
make research-plan
```

On the ETH hardware host, after the preparation command succeeds:

```bash
make PYTHON=.venv/bin/python upmem-hw-mvp-plan
UPMEM_ALLOW_PHYSICAL_HARDWARE=1 make PYTHON=.venv/bin/python upmem-hw-mvp
```

This runs only the fixed 2x2 then 4x4 one-DPU/one-tasklet dense int8 MVP. It
does not run the generic TaskGraph route, report timing as speedup, or promote
the resulting evidence automatically.
Physical allocation uses the explicit SDK `backend=hw` contract. The runner
isolates `UPMEM_PROFILE` and `UPMEM_PROFILE_BASE` from child processes; do not
set either variable to select hardware. `UPMEM_ALLOW_PHYSICAL_HARDWARE=1`
remains mandatory, with no fallback. The repaired evidence profile is
`hardware_mvp_l1_v2`; failed v1 evidence is historical and must not be treated
as a corrected run.

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
- internal einsum/frontier/hybrid routes: diagnostics, not serious baselines.
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

> Bounded generic UPMEM contraction exists, but fully general UPMEM TN
> contraction does not yet exist.

## Generated Cleanup

`make thesis-clean APPLY=1` retains the evidence selected by
`thesis_results/current`, its source research pack, and removes older generated
run directories. `make clean-generated` removes build/cache output and keeps
`runs/` unless `CLEAN_RUNS=1` is explicitly supplied.
