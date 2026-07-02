# Wave 2E.31 Placeholder And Diagnostic Route Cleanup Audit

Date: 2026-07-02

This is an audit-only note. No route, suite, CLI, report schema, or runtime
behavior should change in Wave 2E.31.

Reference scan used:

```bash
rg -n "upmem_dense_int8_placeholder|quest_cpu_full_state_benchmark|sub\.add_parser\(\"summarize\"|sub\.add_parser\(\"plot\"|sub\.add_parser\(\"probe\"|args\.command == \"summarize\"|args\.command == \"plot\"|args\.command == \"probe\"" . --glob '!external/**' --glob '!runs/**' --glob '!build/**' --glob '!*.pyc'
```

## Route And CLI Findings

| Item | Current evidence | Classification | Proposed action | Required safety check before change |
|---|---|---|---|---|
| `upmem_dense_int8_placeholder` | `configs/suites/smoke.yml`; `src/quantum_bench/providers/exact_tn/upmem_dense_placeholder.py`; `tests/test_core_contract.py`; `tests/test_benchmark_smoke.py`; old plot labels. | Placeholder/diagnostic route. | Keep for now. Remove from smoke first in a later wave, then remove the provider only after no active suite/test depends on placeholder skipped rows. | `rg "upmem_dense_int8_placeholder"` must shrink to provider/registry only before provider deletion; `pytest -q`; smoke run must still validate route metadata without fake UPMEM placeholder rows. |
| `quest_cpu_full_state_benchmark` | `configs/suites/local_energy.yml`; `configs/benchmark_matrix.yml`; `src/quantum_bench/providers/full_state/quest_cpu_benchmark.py`; `src/quantum_bench/plots/plot.py`; `src/quantum_bench/bench/run_dirs.py`; matrix/core/simulation tests. | Metrics-only diagnostic/legacy baseline. | Keep for now. Later replace benchmark-matrix/local-energy usage with `quest_cpu_full_state_exact` or an explicitly named energy-only diagnostic route. | Update `local_energy.yml`, benchmark matrix config/tests, plot defaults, route labels, and registry tests; prove exact QuEST route covers required evidence or keep a renamed diagnostic. |
| `summarize` CLI | `src/quantum_bench/bench/__main__.py`; calls `bench.summary.write_summary`. | Legacy raw-run summary path. | Hide from active docs; consider removal after `report-run` fully covers active summary use. | `rg "summarize"` active refs clean; reporting tests cover replacement behavior. |
| `plot` CLI | `src/quantum_bench/bench/__main__.py`; calls `quantum_bench.plots.plot_run`; old plot path uses `quest_cpu_full_state_benchmark` as default baseline. | Legacy raw-run plotting path. | Hide from active docs; consider removal after `report-run` plot generation is the only supported path. | `rg "plot "` and parser refs clean; report-run plot tests cover required plots. |
| Bare `probe` CLI | `src/quantum_bench/bench/__main__.py`; dumps provider registry probes/capabilities. | Legacy provider-probe dump. | Keep for now or later alias behavior into `simulation-backend-probe`; do not remove until probe needs are covered. | `rg " probe"` active refs clean; simulation-backend-probe tests cover provider metadata needs. |

## Cleanup Direction

The safest cleanup order is:

1. Remove `upmem_dense_int8_placeholder` from `smoke.yml` and update smoke/core
   tests so smoke validates CPU route output and artifact schema without fake
   UPMEM skipped rows.
2. Keep `upmem_dense_int8_placeholder` registered for one transition wave.
3. Delete `upmem_dense_int8_placeholder` provider only after reference checks
   show no suites/tests need it.
4. Defer `quest_cpu_full_state_benchmark` cleanup because it crosses local
   energy, benchmark matrix, old plotting, run labels, and tests.
5. Defer `summarize`, `plot`, and bare `probe` removal until report-run and
   simulation-backend-probe are the only active reporting/probe paths.

## Recommended Next Wave

Wave 2E.32 should remove `upmem_dense_int8_placeholder` from smoke only:

- Update `configs/suites/smoke.yml` to use only `cpu_tn_einsum_exact`.
- Update `tests/test_core_contract.py` and `tests/test_benchmark_smoke.py` to
  stop asserting placeholder skipped rows.
- Keep the provider registered for compatibility.
- Run `rg "upmem_dense_int8_placeholder"` and accept provider/registry/plot-label
  references only after smoke cleanup.

Do not remove `quest_cpu_full_state_benchmark` or legacy CLI paths in the same
wave.

## Validation Commands

```bash
rg -n "upmem_dense_int8_placeholder|quest_cpu_full_state_benchmark|sub\.add_parser\(\"summarize\"|sub\.add_parser\(\"plot\"|sub\.add_parser\(\"probe\"" . --glob '!external/**' --glob '!runs/**' --glob '!build/**' --glob '!*.pyc'
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src ../.venv/bin/python -m pytest -q
make -n bench-cpu
make -n bench-gpu
make -n bench-upmem-sim
make -n report-latest
make -n compare-latest
git diff --check -- .gitmodules thesis/implementation thesis/*.md thesis/legacy
find thesis/implementation \( -path "thesis/implementation/external" -prune -o -name "__pycache__" -type d -print -o -name "*.pyc" -type f -print \)
```
