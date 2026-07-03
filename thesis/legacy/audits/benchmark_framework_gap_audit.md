# Wave 2E.40 Thesis-Ready Benchmark Framework Gap Audit

Date: 2026-07-03

This is an audit-only note. No benchmark logic, suites, routes, report schemas,
UPMEM kernels, workflow commands, or active README/ARCHITECTURE pages changed in
this wave.

Inspection sources:

- `thesis/implementation/Makefile`
- `thesis/implementation/README.md`
- `thesis/implementation/ARCHITECTURE.md`
- `thesis/implementation/configs/suites/*.yml`
- `thesis/implementation/src/quantum_bench/bench/`
- `thesis/implementation/src/quantum_bench/providers/`
- `thesis/implementation/src/quantum_bench/targets/upmem/`
- `thesis/implementation/scripts/`
- `thesis/implementation/tests/`
- latest local evidence/comparison artifacts when present:
  - `runs/evidence/upmem_generic_medium/upmem_generic_int8/20260702_231217/`
  - `runs/comparisons/upmem_generic_medium/upmem_generic_sweep/20260702_231217/`

Validation observed during the audit:

- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src ../.venv/bin/python -m pytest -q`:
  277 passed.
- `make -n bench-cpu`: command resolves to `simulation-backend-compare` with
  `simulation_backend_compare_compute_medium.yml`.
- `make -n bench-gpu`: command resolves to GPU probe plus GPU execution-only
  suite and GPU row checker.
- `make -n bench-upmem-sim`: command resolves to UPMEM SDK sample check plus
  UPMEM simulator comparison suite and UPMEM row checker.
- `make -n report-latest` and `make -n compare-latest`: both write under
  `runs/comparisons/...`.
- `git diff --check -- .gitmodules thesis/implementation thesis/*.md thesis/legacy`:
  clean before this audit file was added.
- Cache hygiene check for `__pycache__` and `*.pyc`: clean.

## Already Thesis-Ready

| Area | Verdict | Evidence |
|---|---|---|
| CPU full-state baseline | Ready. `quest_cpu_full_state_exact` is the serious full-state baseline. | `ARCHITECTURE.md`; `tests/test_core_contract.py`; compute suites use `benchmark_role: serious_full_state_baseline`. |
| CPU TN baseline | Ready. `quimb_tn_exact` is the serious TN baseline. | `ARCHITECTURE.md`; `tests/test_core_contract.py`; compute suites use `benchmark_role: serious_external_tn_baseline`. |
| Internal einsum role | Ready enough. It is explicitly diagnostic/internal, not the serious TN baseline. | `ARCHITECTURE.md`; `tests/test_core_contract.py`; serious suites mark `cpu_tn_einsum_exact` optional/diagnostic. |
| Optional GPU evidence | Ready as optional evidence. GPU records are gated by verification and checker scripts. | `Makefile`; `simulation_backend_compare_gpu_execution_only.yml`; `tests/test_simulation_backend_compare.py`; `tests/test_thesis_shortcuts.py`. |
| UPMEM SDK simulator route | Ready for bounded simulator evidence. Records distinguish `upmem`, `sdk_simulator`, SDK control path, and no hardware speedup. | `simulation_backend_compare_upmem_sdk_simulator.yml`; `providers/exact_tn/upmem_sdk_simulator.py`; `tests/test_simulation_backend_compare.py`. |
| Strict UPMEM generic runtime evidence | Ready for bounded current claims. Latest sweep completed 7/11 cases, all completed rows used real SDK simulator path and `cpu_fallback_used=false`. | `runs/comparisons/upmem_generic_medium/upmem_generic_sweep/20260702_231217/upmem_generic_sweep_boundary_summary.json`. |
| Artifact boundary | Ready. Evidence and comparison roots are distinct; reports/plots are not written into evidence by report commands. | `README.md`; `ARCHITECTURE.md`; `tests/test_result_artifacts.py`; `tests/test_reporting.py`. |
| Daily KISS workflow | Ready. Makefile exposes the intended run/report/compare loop. | `Makefile`; `README.md`; `tests/test_thesis_shortcuts.py`; `make -n` validation. |

## Not Thesis-Ready Yet

| Area | Gap | Evidence |
|---|---|---|
| `summary-only` retention | CLI choices expose `summary-only`, but implementation rejects it as deferred. This is safe but confusing for thesis workflow polish. | `bench/__main__.py`; `reporting.validate_retention_mode`; `tests/test_reporting.py::test_summary_only_retention_is_deferred`. |
| Validation toggle | There is no first-class benchmark-level validation on/off toggle. For thesis evidence this is acceptable if validation remains intentionally always-on, but the choice should be explicit. | CLI parser lacks a validation toggle; suites carry validation metadata; route execution validates by default. |
| UPMEM boundary summary | Boundary evidence exists, but the 2E.39 family-level summary was generated manually under comparisons. It is not yet a standard report table. | `runs/comparisons/upmem_generic_medium/upmem_generic_sweep/20260702_231217/upmem_generic_sweep_boundary_summary.json`. |
| Diagnostic CLI visibility | The raw CLI parser still exposes many developer diagnostics. Active README/Makefile stay clean, but `python -m quantum_bench.bench -h` remains broad. | `bench/__main__.py`; diagnostic tests for bridge, frontier, matrix, shadow, and external-lib commands. |
| Legacy `run --suite` evidence | The smoke runner still writes raw JSONL, metrics, and validation JSON instead of normalized records. This is documented, but it is not the same artifact model as thesis evidence runs. | `README.md`; `ARCHITECTURE.md`; `bench/runner.py`; `bench/summary.py`; `tests/test_benchmark_smoke.py`. |
| Suite naming | Suite surface is usable but still verbose. Current family names are explicit, not minimal. | `configs/suites/*.yml`; previous suite surface audit. |

## Benchmark Toggle Readiness

| Toggle | Verdict | Evidence / note |
|---|---|---|
| Quantization on/off | Mostly ready for UPMEM MVP/generic benchmarking. | `upmem-mvp-benchmark --quantization-modes`; `tests/test_upmem_mvp_benchmark.py`; `targets/upmem/taskgraph_runtime.py`. |
| Policy `generic-only` / `dense-only` / `dense-then-generic` | Ready in strict runtime and MVP benchmark paths. | `bench/__main__.py`; `tests/test_upmem_taskgraph_runtime.py`; `tests/test_upmem_mvp_benchmark.py`. |
| Validation on/off | Not a public thesis toggle. | No CLI-level evidence toggle; validation is effectively always-on for evidence. |
| `execute_external` gating | Ready. External SDK paths require explicit execution and tests cover rejection. | Dense/generic/runtime tests; `providers/exact_tn/upmem_sdk_simulator.py`. |
| Artifact retention | Compact/full are usable; `summary-only` remains deferred. | `reporting.py`; `tests/test_reporting.py`; `tests/test_result_artifacts.py`. |

## Polish And Fix Candidates

1. **Retention CLI polish**
   - Current issue: `summary-only` appears as a CLI choice but fails later.
   - Preferred fix: remove it from user-facing choices until implemented, or fail with a clearer parser-level message.
   - Safety check: reporting tests and `make -n` shortcuts.

2. **Validation toggle decision**
   - Current issue: the prompt mentions validation on/off, but thesis evidence currently assumes validation-on.
   - Preferred fix: document validation-on as the default and only supported evidence mode; do not add a toggle unless a concrete performance-run need appears.
   - Safety check: README/ARCHITECTURE wording plus existing validation tests.

3. **UPMEM boundary table promotion**
   - Current issue: largest supported case / first unsupported case / blocker summaries are manual derived artifacts.
   - Preferred fix: add this as a derived report table for UPMEM generic evidence in existing `report-run` or `compare-results` output, not a new command.
   - Safety check: evidence dirs still contain no plots/derived figures; report tests cover the table.

4. **Diagnostics help surface**
   - Current issue: raw CLI help is broad even though README/Makefile are KISS.
   - Preferred fix: keep all commands, but group diagnostics in help text or add a short diagnostics note only if needed.
   - Safety check: CLI parser tests and diagnostic command tests.

5. **Legacy smoke artifact boundary**
   - Current issue: `run --suite` uses raw/metrics/validation files, not normalized records.
   - Preferred fix: leave it as smoke-only for now, or later add normalized records to `run --suite` without replacing current smoke artifacts.
   - Safety check: `tests/test_benchmark_smoke.py` and artifact boundary tests.

## Deletion Or Hiding Candidates

| Candidate | Recommendation | Why | Required check |
|---|---|---|---|
| `summary-only` CLI choice | Hide until implemented. | Avoids a valid-looking option that always rejects. | Reporting tests and CLI smoke. |
| Diagnostic commands in CLI help | Hide/group, not delete. | They remain useful and tested, but are not the thesis daily path. | All diagnostic tests still pass. |
| `local_energy.yml` / energy script | Keep diagnostic or move later. | Not part of current evidence workflow. | Reference checks and local energy docs/tests. |
| Legacy `run --suite` normalized gap | Do not delete. | Smoke suite still depends on it. | Smoke tests. |
| Manual UPMEM sweep snippets | Do not commit as repo code. | Keep 2E.39 KISS; promote only the table behavior if needed. | Report/comparison tests if promoted later. |

## Exact Next Implementation Plan

Maximum five independent subtasks:

1. **Hide Deferred `summary-only`**
   - Remove `summary-only` from `upmem-mvp-benchmark`, `simulation-backend-compare`, and `prune-run` argparse choices until fully implemented.
   - Keep `validate_retention_mode("summary-only")` test or adjust it to assert parser-level rejection.
   - Validate: `pytest -q`; `make -n` shortcuts.

2. **Document Validation-On Evidence Policy**
   - Add one concise README/ARCHITECTURE sentence: thesis evidence runs validate by default; validation-off is not a supported evidence mode yet.
   - Do not add a new toggle.
   - Validate: docs diff check and existing tests.

3. **Add UPMEM Generic Boundary Derived Table**
   - Extend existing report/comparison generation to emit a UPMEM boundary CSV/Markdown when UPMEM generic task metrics are present.
   - Include largest supported case, first unsupported case, blocker, max rank, max elements, max contracted combinations, DPU execution, and CPU fallback status.
   - Write only under `runs/comparisons/...`.
   - Validate: report tests; evidence-no-plots/no-derived-artifacts tests.

4. **Group Diagnostics In CLI Help**
   - Keep command behavior unchanged.
   - Improve help labels/grouping enough that public evidence commands are obvious and diagnostics are visibly diagnostics.
   - Validate: CLI tests and focused parser checks.

5. **Decide Legacy `run --suite` Future**
   - Either keep it documented as smoke-only legacy evidence, or add `normalized_records.jsonl` output in a later separate wave.
   - Do not migrate in the same wave as UPMEM boundary reporting.
   - Validate: smoke tests and result artifact tests.

## Current Verdict

The benchmark framework is close to thesis-ready for current CPU, optional GPU,
and bounded UPMEM SDK simulator evidence. The remaining work is polish and
consistency, not a blocker to current bounded claims. The highest-value next fix
is promoting the UPMEM generic boundary table into existing derived reporting
while preserving the evidence/comparison artifact boundary.
