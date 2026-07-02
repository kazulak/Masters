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

Run CPU evidence:

```bash
make bench-cpu
```

The default CPU suite is:

```text
configs/suites/simulation_backend_compare_compute_medium.yml
```

Override it when needed:

```bash
make bench-cpu CPU_SUITE=configs/suites/simulation_backend_compare_thesis_small.yml
```

Run GPU evidence:

```bash
make bench-gpu
```

This verifies QuEST HIP first, then runs the GPU execution-only suite. If the
ROCm/GPU path is unavailable, it fails with a blocker instead of emitting fake
GPU rows.

Run UPMEM SDK simulator evidence:

```bash
make bench-upmem-sim
```

This checks the UPMEM SDK simulator path, then runs the strict UPMEM SDK
simulator comparison suite. It fails if no strict UPMEM SDK simulator row is
emitted.

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
