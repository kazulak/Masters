# Wave 2E.35 KISS Module Boundary Audit

Date: 2026-07-02

This is an audit-only note. No provider, route, CLI, report schema, UPMEM
kernel, or benchmark behavior changed in Wave 2E.35.

Inspection sources:

- module tree under `src/quantum_bench/`
- scripts and tests under `scripts/` and `tests/`
- active Makefile shortcut references
- targeted `rg` checks excluding `external/`, `runs/`, `build/`, and `*.pyc`

Current active model remains:

```text
quantum circuit -> tensor network -> TaskGraph -> route execution -> normalized records
```

## Candidate 1: Narrow The UPMEM Target Facade

| Field | Finding |
|---|---|
| Current evidence | `src/quantum_bench/targets/upmem/__init__.py` re-exports scheduling, dense bridge, generic bridge, diagnostics, environment checks, external-lib scans, frontier analysis, SimplePIM helpers, synthetic pressure, and tile planning. Many callers use `from quantum_bench.targets.upmem import ...`. |
| Problem | The package facade has become a mixed public API. It blurs target modeling, SDK execution, diagnostics, and analysis-only helpers. |
| Proposed boundary | Keep `targets.upmem` as a transition facade, but move new imports to explicit modules: `schedule`, `dense_bridge`, `generic_bridge`, `taskgraph_runtime`, `environment`, `frontier`, and `synthetic_pressure`. |
| First small refactor | Convert one low-risk group, such as `tests/test_upmem_schedule.py` or `bench/upmem_generic_feasibility.py`, to explicit module imports only. |
| Independent checks | `rg "from quantum_bench.targets.upmem import"` count should shrink; run the touched focused tests and then `pytest -q`. |

## Candidate 2: Centralize Normalized Record Construction

| Field | Finding |
|---|---|
| Current evidence | `bench/reporting.py` owns `write_normalized_records`, but `bench/simulation_backend_compare.py`, `bench/upmem_mvp_benchmark.py`, and `bench/upmem_taskgraph_runtime.py` each construct route/runtime records and status metadata locally. |
| Problem | Common fields such as route identity, execution target, timing, validation, artifact references, UPMEM flags, and hardware-speedup flags are repeated and can drift. |
| Proposed boundary | Keep record writing in `bench/reporting.py`, and add small shared builder helpers there for common normalized-record fragments. Producers should pass domain-specific metrics only. |
| First small refactor | Migrate `upmem_taskgraph_runtime` normalized-record construction first because it is narrower than suite-level compare code. |
| Independent checks | Existing `tests/test_upmem_taskgraph_runtime.py`, `tests/test_result_artifacts.py`, and `compare-results` fixture tests must pass. Assert generated records retain the same schema version and required fields. |

## Candidate 3: Separate UPMEM Evidence Route Checks From Bridge Diagnostics

| Field | Finding |
|---|---|
| Current evidence | `providers/exact_tn/upmem_sdk_simulator.py` performs route gating, SDK simulator preflight, strict runtime assertions, and record metadata construction. Dense/generic bridge diagnostics live under `targets/upmem`, while developer CLIs live under `bench`. |
| Problem | The provider route contains reusable strict-runtime checks that are not provider-specific, while bridge diagnostics and runtime evidence share overlapping flags and reasons. |
| Proposed boundary | Provider route should stay responsible for `can_execute`, `prepare`, and `execute`. Reusable preflight payloads, DPU-invocation assertions, and strict no-CPU-fallback checks should move to a narrow target helper such as `targets/upmem/runtime_checks.py`. |
| First small refactor | Extract only the strict assertion helper from the provider route. Do not move bridge execution or change artifact fields. |
| Independent checks | `tests/test_simulation_backend_compare.py`, `tests/test_upmem_taskgraph_runtime.py`, `make -n bench-upmem-sim`, and UPMEM route metadata assertions must pass. |

## Candidate 4: Keep Scripts As Makefile Glue Or Diagnostics

| Field | Finding |
|---|---|
| Current evidence | `scripts/evidence_shortcuts.py` is used by Makefile and `tests/test_thesis_shortcuts.py`. `scripts/run_energy_suite.sh` defaults to `configs/suites/local_energy.yml` and is referenced only as a local diagnostic helper in audits. |
| Problem | Scripts can look like public APIs even when they are implementation details behind Makefile targets or local diagnostics. |
| Proposed boundary | Keep `evidence_shortcuts.py` as Makefile-only glue with tests. Keep `run_energy_suite.sh` local/diagnostic and out of README/Makefile public evidence flow. |
| First small refactor | Add a short module-level note to `evidence_shortcuts.py` and optionally move `run_energy_suite.sh` under a diagnostics path in a later wave. |
| Independent checks | `tests/test_thesis_shortcuts.py`, `make -n bench-cpu`, `make -n bench-gpu`, `make -n bench-upmem-sim`, `make -n report-latest`, and `make -n compare-latest`. |

## Candidate 5: Unify UPMEM Status And Flag Vocabulary

| Field | Finding |
|---|---|
| Current evidence | Dense bridge, generic bridge, strict TaskGraph runtime, and UPMEM simulator provider code all emit related concepts: execution status, blocker reason, validation status, DPU program invocation, SDK simulator mode, hardware flags, SimplePIM usage, and CPU fallback flags. |
| Problem | Similar status fields are built in multiple modules. This increases the risk that reports treat simulator, hardware, CPU fallback, and SDK control-path evidence inconsistently. |
| Proposed boundary | Add a small constants/helper module for shared UPMEM evidence flags and common failure reasons. Do not rename existing artifact fields in the first pass. |
| First small refactor | Migrate constants in one path only, preferably generic bridge or strict runtime summary generation. |
| Independent checks | Existing generic bridge/runtime tests plus reference checks proving artifact field names remain unchanged. |

## Not Recommended Now

- Do not add a generic accelerator abstraction. Current explicit CPU/GPU/UPMEM
  fields are easier to audit.
- Do not merge provider routes with target execution code.
- Do not move all UPMEM modules at once. The target package is broad, but a
  staged import-boundary cleanup is safer.
- Do not remove diagnostic CLIs or scripts in this wave; route/CLI cleanup was
  handled separately in Waves 2E.31-2E.34.

## Suggested Next Implementation Wave

Start with Candidate 1 or Candidate 2.

- Candidate 1 is the least behavior-risky and reduces import coupling.
- Candidate 2 gives the largest reporting payoff but touches evidence artifacts.

Recommended first implementation:

```text
Wave 2E.36 — Narrow UPMEM Import Boundaries
```

Move one focused test/module group from the broad `targets.upmem` facade to
explicit imports, keep the facade intact for compatibility, and use reference
counts plus tests to prove behavior is unchanged.

## Validation Commands

Run from `thesis/implementation`:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src ../.venv/bin/python -m pytest -q
make -n bench-cpu
make -n bench-gpu
make -n bench-upmem-sim
make -n report-latest
make -n compare-latest
rg -n "from quantum_bench\.targets\.upmem import" src tests --glob '!external/**' --glob '!runs/**' --glob '!build/**' --glob '!*.pyc'
git diff --check -- .gitmodules thesis/implementation thesis/*.md thesis/legacy
find thesis/implementation \( -path "thesis/implementation/external" -prune -o -name "__pycache__" -type d -print -o -name "*.pyc" -type f -print \)
```
