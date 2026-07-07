# Quantum Bench Thesis Runtime

This is the active thesis implementation. It compares exact quantum-circuit
simulation backends and records benchmark evidence in a reproducible artifact
layout.

The active architecture story is [ARCHITECTURE.md](ARCHITECTURE.md).

## Setup

Run commands from this directory:

```bash
cd thesis/implementation
git submodule update --init --recursive external/QuEST external/SimplePIM external/PID-Comm
../.venv/bin/python -m pip install -e ".[dev]"
```

For a fresh clone, prefer:

```bash
git clone --recurse-submodules <repo-url>
```

Run the test suite:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src ../.venv/bin/python -m pytest -q
```

Run the smoke suite:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src ../.venv/bin/python -m quantum_bench.bench run --suite configs/suites/smoke.yml
```

## Thesis Evidence Shortcuts

The Makefile is the main execution surface. Evidence runs are written under
`runs/evidence/...`; derived comparisons are written under
`runs/comparisons/...`. `runs/latest` points only to the latest evidence run.

Check local prerequisites without writing benchmark artifacts:

```bash
make doctor
```

Build the native QuEST CPU runner:

```bash
make build-quest-cpu
```

Run CPU evidence:

```bash
make bench-cpu
```

The default CPU suite is:

```text
configs/suites/cpu_evidence.yml
```

This default CPU evidence path uses the serious baselines only:
`quest_cpu_full_state_exact` for full-state simulation and `quimb_tn_exact` for
tensor-network simulation. `cpu_tn_einsum_exact` remains a diagnostic/internal
route for small checks and is not part of the default CPU evidence suite.

Run GPU evidence:

```bash
make bench-gpu
```

This verifies QuEST HIP first, then runs the GPU execution-only suite. If the
ROCm/GPU path is unavailable, it fails with a blocker instead of emitting fake
GPU rows. The benchmark process must be able to see `/dev/kfd`, `/dev/dri`, and
a `/dev/dri/renderD*` node; the verifier then runs a tiny HIP kernel before it
builds and runs the QuEST HIP route.

`simulation-backend-probe` reports whether GPU status came from a fresh
`--verify-gpu` run or a cached verification artifact. Cached verification can
prove that an earlier QuEST GPU run executed, but current-process device
visibility is reported separately and should be checked before new GPU
benchmarks.

CPU/GPU full-state benchmarking has two tiers. Correctness-tier suites use
`state_output_mode=full_dump` and full-statevector validation. Performance-tier
suites use `state_output_mode=none`, `output_contract=metrics_only`, and
native status/gate-count validation; those rows are valid for CPU/GPU compute
timing comparison, not as full-output exactness evidence.

Run UPMEM SDK simulator evidence:

```bash
make bench-upmem-sim
```

This checks the UPMEM SDK simulator path, then runs the strict UPMEM SDK
simulator comparison suite. It fails if no strict UPMEM SDK simulator row is
emitted.

Run the canonical CPU + UPMEM SDK simulator evidence and comparison workflow:

```bash
make thesis-benchmark
```

This runs CPU evidence, regenerates a CPU report, runs strict UPMEM SDK
simulator evidence, regenerates a UPMEM report, and writes a derived comparison
under `runs/comparisons/thesis_benchmark/...`.

Run the full thesis benchmark workflow and generate plots:

```bash
make thesis-results
```

This runs the thesis full-state CPU/GPU suite, Quimb CPU TN suite, diagnostic
CPU TN path/quantization suite, and UPMEM SDK simulator boundary suite. Each benchmark run is saved
automatically under `runs/evidence/...`; the target captures those paths and
writes derived CSV/Markdown/plots under `runs/comparisons/thesis/<timestamp>/`.
The GPU thesis suite must be run manually in a GPU-visible shell when this
Codex process cannot see `/dev/kfd` and `/dev/dri`.

Regenerate the thesis comparison report from existing evidence paths:

```bash
make thesis-report THESIS_INPUTS="runs/evidence/<suite>/<route>/<run_id> runs/evidence/<suite>/<route>/<run_id>"
```

Use this only when the evidence already exists and you want a new report
without rerunning benchmarks.

`make thesis-results` expands to these commands:

```bash
# Full-state CPU/GPU, seven sizes from 8q to 20q.
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src ../.venv/bin/python -m quantum_bench.bench simulation-backend-probe --verify-gpu quest-hip
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src ../.venv/bin/python -m quantum_bench.bench simulation-backend-compare --suite configs/suites/manual/thesis_full_state_cpu_gpu.yml --artifact-retention compact

# Serious CPU tensor-network evidence, seven target sizes.
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src ../.venv/bin/python -m quantum_bench.bench simulation-backend-compare --suite configs/suites/manual/thesis_cpu_tn_quimb.yml --artifact-retention compact

# Diagnostic CPU tensor-network path and quantization attribution, seven target sizes.
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src ../.venv/bin/python -m quantum_bench.bench simulation-backend-compare --suite configs/suites/manual/thesis_tn_paths_quantization.yml --artifact-retention compact

# UPMEM SDK simulator boundary, no hardware speedup claim.
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src ../.venv/bin/python -m quantum_bench.bench upmem-mvp-benchmark --suite configs/suites/manual/thesis_upmem_quantization_boundary.yml --policies generic-only --quantization-modes none,per_task_input_quantize --execute-external

# Derived thesis tables and plots from the evidence paths produced above.
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src ../.venv/bin/python scripts/thesis_report.py --inputs "<captured_cpu_gpu_run>" "<captured_quimb_tn_run>" "<captured_quantization_diagnostic_run>" "<captured_upmem_run>" --out runs/comparisons/thesis/<timestamp>
```

The report produces CPU/GPU speedup plots, TN path runtime plots, TN
quantization speedup/error plots, and UPMEM boundary plots. It does not create
GPU evidence by itself; GPU evidence must come from a GPU-visible shell.

If `make thesis-results` stops after printing a full-state CPU/GPU evidence
path, resume without rerunning the GPU sweep:

```bash
make thesis-results-resume FULL_STATE_RUN=runs/evidence/thesis_full_state_cpu_gpu/simulation_backend_compare/<run_id>
```

Print the research benchmark pack plan:

```bash
make research-plan
```

Create a lightweight research pack without long benchmark execution:

```bash
make research-benchmarks
```

Run the full manual research pack only when you are ready for long runs:

```bash
RUN_RESEARCH=1 make research-benchmarks
```

Regenerate a research pack from existing evidence:

```bash
make research-report
```

Research packs write derived CSVs, plots, and `benchmark_summary.md` under
`runs/comparisons/research_pack/...`. See
[docs/research_benchmark_methodology.md](docs/research_benchmark_methodology.md)
for the thesis-safe claims and limitations.

Parallelism work is tracked separately from benchmark execution commands. See
[docs/parallelization_roadmap.md](docs/parallelization_roadmap.md) and
[docs/parallelization_implementation_strategy.md](docs/parallelization_implementation_strategy.md)
for the current slicing, frontier, hybrid, GPU TN, and UPMEM/PIM claim
boundaries. GPU tensor-network support remains feasibility-only until a real
GPU TN route executes tensor-network work on a GPU with no CPU fallback; see
[docs/gpu_tn_feasibility.md](docs/gpu_tn_feasibility.md).
UPMEM multi-DPU work currently has modeled assignment evidence plus a
frontier-scheduled SDK simulator prototype. The modeled assignment command is:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src ../.venv/bin/python -m quantum_bench.bench upmem-multi-dpu-assignment \
  --suite configs/suites/upmem_sim_evidence.yml \
  --dpu-groups 4
```

This writes assignment evidence under `runs/evidence/...`; it does not execute
multi-DPU work or create hardware speedup evidence.

The first executed SDK simulator frontier prototype is:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src ../.venv/bin/python -m quantum_bench.bench upmem-taskgraph-frontier-runtime \
  --case bell_2q \
  --policy generic-only \
  --quantization-mode per_task_input_quantize \
  --dpu-groups 2 \
  --frontier-worker-count 1 \
  --execute-external
```

This executes ready TaskGraph contractions wave-by-wave through the existing
strict UPMEM SDK simulator path. It is still simulator evidence, not hardware
multi-DPU execution or hardware speedup evidence.

Regenerate reports for the latest evidence run:

```bash
make report-latest
```

This reads `runs/latest` and writes derived tables/plots under
`runs/comparisons/...`; it does not add figures to the evidence run.

Generate a derived comparison for the latest evidence run:

```bash
make compare-latest
```

Generate the current parallelism evidence comparison:

```bash
make parallelism-report
```

This runs the existing CPU slicing/frontier diagnostic suite, the internal
hybrid diagnostic suite, and the modeled UPMEM DPU assignment report, then
writes derived comparison artifacts under `runs/comparisons/parallelism/...`.
It does not execute UPMEM multi-DPU work and does not create hardware speedup
evidence.

To include the executed SDK simulator frontier prototype row in the same
comparison, opt in explicitly:

```bash
make parallelism-report PARALLELISM_INCLUDE_UPMEM_FRONTIER=1
```

That run requires a working UPMEM SDK simulator path. The added UPMEM row is
still SDK simulator evidence, not hardware multi-DPU evidence.

Inspect normalized records:

```bash
head -n 3 runs/latest/normalized_records.jsonl
```

Clean generated build/cache files while keeping benchmark evidence:

```bash
make clean-generated
```

Remove benchmark evidence and comparisons only when explicitly requested:

```bash
make clean-generated CLEAN_RUNS=1
```

`CLEAN_RUNS=1` is intentionally destructive for generated local run artifacts:
it removes `runs/evidence`, `runs/comparisons`, `runs/latest`, and older legacy
run folders. Use it when you want to reclaim disk space or reset local evidence
state.

## Where Results Live

```text
runs/evidence/<suite_id>/<route_label>/<run_id>/
runs/comparisons/<suite_id>/<comparison_type>/<comparison_id>/
```

Every evidence run should contain `run_manifest.json` and
`normalized_records.jsonl`. Comparison reports are derived analysis and should
not mutate evidence folders. Derived tables, plot-source CSVs, and figures are
written by `report-latest`, `report-run`, or comparison commands under
`runs/comparisons/...`. Build/cache outputs are ignored and live under paths
such as `build/`, `.pytest_cache/`, and native `bin/` or `build/` directories.

The legacy smoke command, `run --suite`, writes raw JSONL, validation JSON, and
metrics CSV files as its own evidence format. Use the Makefile evidence commands
when you need `normalized_records.jsonl` for report and comparison regeneration.

## Canonical Suites

The active suite family is intentionally small:

```text
configs/suites/smoke.yml
configs/suites/cpu_evidence.yml
configs/suites/gpu_evidence.yml
configs/suites/cpu_gpu_sweep.yml
configs/suites/upmem_sim_evidence.yml
configs/suites/upmem_generic_sweep.yml
configs/suites/manual_large.yml
```

Developer diagnostics and historical bring-up suites live under
`configs/suites/diagnostics/` and are not part of the Makefile evidence
surface. Manual staged thesis helpers live under `configs/suites/manual/`; for
example, `cpu_gpu_sweep_tier1.yml` and `cpu_gpu_sweep_tier2.yml` split the
canonical CPU/GPU sweep so larger cases can be run after smaller cases pass.
`cpu_gpu_correctness_deep.yml` keeps full-output validation under a small cap,
while `cpu_gpu_performance.yml` disables state dumps for compute-focused timing.
Research-grade manual suites use the `research_*.yml` prefix in the same manual
directory and are intended for explicit thesis/paper evidence generation.
