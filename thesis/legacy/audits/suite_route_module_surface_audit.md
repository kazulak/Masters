# Wave 2E.29 Suite, Route, And Module Surface Audit

Date: 2026-07-02

This audit records the current executable surface after the KISS documentation
cleanup. Classifications below are inspection results from the current
repository, not desired truth baked into code. No runtime behavior, suite
behavior, routes, or report schemas were changed for this audit.

Inspection sources:

- `configs/suites/*.yml`
- `src/quantum_bench/providers/registry.py`
- provider classes under `src/quantum_bench/providers/`
- CLI registrations in `src/quantum_bench/bench/__main__.py`
- `Makefile`, `scripts/`, and `tests/`
- `rg` reference checks excluding `external/`, `runs/`, and `build/`

## Suite Surface

| Suite | Current evidence | Classification | Proposed action | Required safety check before delete/merge |
|---|---|---|---|---|
| `smoke.yml` | `configs/suites/smoke.yml`; routes `cpu_tn_einsum_exact`, `upmem_dense_int8_placeholder`; referenced by smoke tests. | Quick smoke | Keep as default smoke. | `pytest -q`; `run --suite configs/suites/smoke.yml`. |
| `smoke_v2.yml` | `configs/suites/smoke_v2.yml`; same route shape as `smoke.yml`; referenced by `tests/test_core_contract.py` and `tests/test_benchmark_smoke.py`. | Duplicate/candidate merge | Merge into `smoke.yml` later or delete after test migration. | `rg "smoke_v2"` must find no active refs; update tests that assert v2 schema behavior. |
| `simulation_backend_compare_compute_medium.yml` | `configs/suites/...compute_medium.yml`; routes QuEST CPU, Quimb TN, internal CPU TN; Makefile `CPU_SUITE` default. | Active CPU evidence | Keep as `bench-cpu` default. | None before keep; if renamed, update Makefile and `tests/test_thesis_shortcuts.py`. |
| `simulation_backend_compare_gpu_execution_only.yml` | Routes QuEST CPU and optional QuEST GPU; Makefile `GPU_SUITE` default. | Active optional GPU evidence | Keep as `bench-gpu` target. | GPU row tests and blocker behavior must still pass. |
| `simulation_backend_compare_upmem_sdk_simulator.yml` | Routes QuEST CPU, Quimb TN, `upmem_tn_sdk_simulator_quantized`; Makefile `UPMEM_SIM_SUITE` default. | Active UPMEM SDK simulator evidence | Keep as `bench-upmem-sim` target. | UPMEM row assertions must still require SDK/DPU execution and no CPU fallback. |
| `simulation_backend_compare_compute_large.yml` | Metadata `expected_runtime_class=manual_large`, `intended_use=thesis_evidence`; routes QuEST CPU, Quimb, internal CPU TN. | Manual large evidence | Keep manual-only; not default validation. | Large-suite config/load tests; no Makefile default dependency. |
| `simulation_backend_compare_quick.yml` | Routes internal CPU TN, QuEST CPU, Quimb; referenced by simulation comparison tests. | Quick backend comparison | Keep for fast correctness validation. | If merged, update tests and README-free workflow references. |
| `simulation_backend_compare_thesis_small.yml` | Metadata `expected_runtime_class=local_minutes`; routes internal CPU TN, QuEST CPU, Quimb; accepted as CPU suite override in README/tests. | Local small evidence | Candidate merge with quick or scaling family. | Update README, `test_thesis_shortcuts.py`, and suite comparison tests. |
| `simulation_backend_compare_scaling.yml` | Metadata `expected_runtime_class=local_heavier_than_thesis_small`; referenced by `tests/test_simulation_backend_compare.py`. | Local scaling evidence | Candidate merge into future `cpu_evidence` or `manual_large`. | `rg "simulation_backend_compare_scaling"` must find no active refs after migration. |
| `simulation_backend_compare_gpu_medium.yml` | Routes internal CPU TN, QuEST CPU, Quimb, optional QuEST GPU; GPU scaffold metadata. | Optional GPU medium | Keep as optional/manual GPU scaffold, but not Makefile default. | If merged with execution-only, preserve optional GPU no-fake-row tests. |
| `simulation_backend_compare_gpu_execution_only.local.yml` | Routes QuEST CPU and optional QuEST GPU; `rg` finds only the file itself in active tree. | Local duplicate/candidate delete | Delete or move to local ignored examples after reference check. | `rg "gpu_execution_only.local"` excluding generated dirs must return no active refs; suite loader tests not required. |
| `pim_bridge_eval.yml` | Route `cpu_tn_einsum_exact`; consumed by `pim-bridge-eval` developer workflow. | Diagnostic/dev | Keep out of README/Makefile. | If deleted, update `tests/test_pim_bridge_eval.py` and any manual validation docs. |
| `pim_bridge_eval_quick.yml` | Quick variant for `pim-bridge-eval`; route `cpu_tn_einsum_exact`. | Diagnostic/dev quick | Keep until PIM bridge diagnostics are retired. | `rg "pim_bridge_eval_quick"` and PIM eval tests. |
| `pim_frontier_pressure.yml` | Synthetic pressure/frontier suite; route `cpu_tn_einsum_exact`. | Diagnostic/model-only | Keep as analysis fixture, not evidence. | Frontier tests and synthetic workload loader checks. |
| `pim_frontier_pressure_quick.yml` | Quick pressure/frontier suite; used by frontier validation. | Diagnostic/model-only quick | Keep until frontier tooling is archived. | `tests/test_pim_frontier_analysis.py`. |
| `pim_l2_tiled_quick.yml` | Referenced by `tests/test_pim_bridge_eval.py`; L2 synthetic validation context. | Diagnostic/dev | Keep while L2 bridge tests exist. | Migrate tests before removal. |
| `upmem_generic_medium.yml` | Metadata `intended_use=developer_benchmark`; route `cpu_tn_einsum_exact`; UPMEM MVP/generic context. | Developer benchmark | Keep out of main Makefile unless promoted. | `rg "upmem_generic_medium"` and UPMEM MVP tests. |
| `planner_compare.yml` | Planner comparison route `cpu_tn_einsum_exact`. | Diagnostic/dev | Keep for planner tests. | `tests/test_planner_comparison.py` and planner CLI refs. |
| `planner_compare_extended.yml` | Referenced by `tests/test_planner_comparison.py`. | Diagnostic/dev extended | Candidate merge into `planner_compare.yml` later. | Update extended planner tests before merge. |
| `local_energy.yml` | Used by `scripts/run_energy_suite.sh`; routes internal CPU TN and metrics-only QuEST route. | Local diagnostic | Keep as local/energy helper or move to diagnostics later. | Update `scripts/run_energy_suite.sh`, native README refs, and energy-related tests/docs. |
| `local_plot.yml` | Route `cpu_tn_einsum_exact`; no active refs found beyond file path in current inspection. | Local/candidate delete | Delete after confirming no plot workflow depends on it. | `rg "local_plot"` excluding generated dirs must be empty; run plot/report tests. |

Suggested future suite family:

- `smoke`
- `cpu_evidence`
- `gpu_evidence`
- `upmem_sim_evidence`
- `manual_large`
- optional diagnostics only if the suite loader and tests support a diagnostics
  namespace.

## Route And Backend Surface

Graph-level route IDs come from `route_registry()` in
`src/quantum_bench/providers/registry.py`.

| Route ID | Current evidence | Classification | Proposed action | Required safety check before delete/merge |
|---|---|---|---|---|
| `quest_cpu_full_state_exact` | Provider `QuestCpuFullStateExactRoute`; role `serious_full_state_baseline`; output `statevector`; validation `compare_statevector`. | Serious evidence | Keep. | Route registry and simulation comparison tests. |
| `quimb_tn_exact` | Provider `QuimbTnExactRoute`; role `serious_external_tn_baseline`; output `final_tensor`; backend family `quimb`. | Serious CPU TN evidence | Keep as primary TN baseline. | Quimb dependency/probe and output comparison tests. |
| `quest_gpu_full_state_exact` | Provider `QuestGpuFullStateExactRoute`; role `optional_gpu_candidate`; target `gpu`; execution requires verification artifact. | Optional comparison | Keep optional/gated. | Tests must prove no fake GPU rows and CPU fallback cannot be reported as GPU. |
| `upmem_tn_sdk_simulator_quantized` | Provider `UpmemTnSdkSimulatorQuantizedRoute`; target `upmem`; mode `sdk_simulator`; quantized name avoids exactness claim. | Active UPMEM simulator evidence | Keep as strict simulator comparison route. | Tests must require DPU invocations, `cpu_fallback_used=false`, and no hardware speedup claims. |
| `cpu_tn_einsum_exact` | Provider `CpuTnEinsumExactRoute`; role `internal_debug_baseline`; many diagnostic suites use it. | Diagnostic/internal | Keep, but keep out of serious TN claims. | Before removal from serious suites, update reference routes and comparison expectations. |
| `quest_cpu_full_state_benchmark` | Provider `QuestCpuFullStateBenchmarkRoute`; output `metrics_only`; used by `local_energy.yml`, matrix tests, and old plotting defaults. | Metrics-only diagnostic/legacy candidate | Demote further or remove after exact route covers needed evidence. | Update local energy suite, matrix config/tests, plot defaults, and route registry tests. |
| `upmem_dense_int8_placeholder` | Provider `UpmemDenseInt8PlaceholderRoute`; output `none`; used by smoke suites and tests as skipped placeholder. | Placeholder/diagnostic candidate | Replace smoke placeholder with better readiness check later, then remove. | Update `smoke*.yml`, `tests/test_core_contract.py`, `tests/test_benchmark_smoke.py`, plot labels. |

Developer backend IDs observed from CLI choices and bridge code are not
graph-level routes:

- `upmem_sdk_simulator_dense`
- `upmem_sdk_simulator_generic_loop`
- `mock_numpy_dequantized`
- `simplepim_external_stub`
- `simplepim_external`

They should remain clearly separated from route IDs in docs and artifacts.

## Script And CLI Surface

Public execution surface should stay Makefile-first.

| Item | Current evidence | Classification | Proposed action | Safety check before removal/change |
|---|---|---|---|---|
| `make bench-cpu` | `Makefile`; tested by `tests/test_thesis_shortcuts.py`; default suite compute-medium. | Public evidence shortcut | Keep. | `make -n bench-cpu`; shortcut tests. |
| `make bench-gpu` | `Makefile`; runs GPU probe then GPU execution-only suite. | Public optional GPU shortcut | Keep. | `make -n bench-gpu`; no fake GPU row tests. |
| `make bench-upmem-sim` | `Makefile`; runs UPMEM env sample then UPMEM simulator comparison. | Public UPMEM simulator shortcut | Keep. | `make -n bench-upmem-sim`; UPMEM row check tests. |
| `make report-latest` | `Makefile`; calls `report-run`. | Public report shortcut | Keep. | `make -n report-latest`. |
| `make compare-latest` | `Makefile`; writes under `runs/comparisons`. | Public comparison shortcut | Keep. | `make -n compare-latest`; run boundary tests. |
| `scripts/evidence_shortcuts.py` | Used by Makefile and `tests/test_thesis_shortcuts.py`. | Public helper behind Makefile | Keep. | Shortcut tests. |
| `scripts/run_energy_suite.sh` | Defaults to `local_energy.yml`; native QuEST README references it. | Local diagnostic helper | Keep or move to diagnostics later. | Update references and energy workflow docs. |

CLI subcommands from `src/quantum_bench/bench/__main__.py`:

| Command group | Current evidence | Classification | Proposed action | Safety check |
|---|---|---|---|---|
| `run` | CLI parser and smoke README command. | Basic/smoke | Keep. | Smoke tests. |
| `simulation-backend-compare`, `simulation-backend-probe` | CLI parser; Makefile uses both. | Evidence/probe | Keep. | Simulation backend tests and Makefile dry-runs. |
| `upmem-env-check` | CLI parser; Makefile `bench-upmem-sim` uses it. | Evidence preflight | Keep. | UPMEM environment tests. |
| `report-run`, `compare-results`, `compare-runs`, `prune-run` | CLI parser; reporting tests. | Evidence/reporting | Keep. | Reporting and artifact tests. |
| `dense-task-bridge`, `generic-task-bridge`, `upmem-taskgraph-runtime`, `upmem-mvp-benchmark`, `upmem-generic-feasibility` | CLI parser; dedicated tests. | Developer/diagnostic UPMEM | Keep out of main README. | Dedicated bridge/runtime/MVP tests. |
| `pim-bridge-eval`, `pim-frontier-analysis`, `benchmark-matrix-report`, `dense-route-coverage`, `shadow-routed-runtime` | CLI parser; dedicated tests. | Diagnostic/model/report scaffold | Keep out of main README. | Dedicated tests and suite refs. |
| `compare-planners` | CLI parser; planner tests. | Diagnostic/dev | Keep until planner surface is consolidated. | Planner tests. |
| `simplepim-microbench`, `upmem-external-libs-check` | CLI parser; SimplePIM/external-lib tests. | Diagnostic/probe | Keep until external candidate strategy changes. | External probe tests. |
| `summarize`, `plot`, bare `probe` | CLI parser; older plotting/probe path; limited active references in current inspection. | Candidate legacy/unclear | Audit separately before deletion. | `rg` refs, plot tests, route probe expectations. |

## Module Surface Recommendations

These are cleanup proposals only:

- Keep the active mental model as `TaskGraph -> route execution -> normalized records`.
- Keep Makefile shortcuts as the public entry point; do not promote diagnostic CLIs to README.
- Do not add a generic accelerator abstraction. Current explicit CPU/GPU/UPMEM fields are clearer.
- Consolidate status and metadata builders only after route/suite cleanup; current repeated fields appear across reporting, UPMEM runtime, bridge code, and provider records.
- Keep `targets/upmem/` split into execution, probes, and analysis only while tests make the boundary clear. Future cleanup could separate diagnostics from strict execution modules.
- Preserve `cpu_tn_einsum_exact` as a diagnostic route until serious suites no longer need it as an optional/debug row.

## Conservative Candidate List

| Candidate | Action type | Why | Must pass first |
|---|---|---|---|
| `smoke_v2.yml` | Merge/delete | Duplicates smoke shape and is test-only. | Migrate `test_core_contract.py` and `test_benchmark_smoke.py`; `rg "smoke_v2"` clean. |
| `simulation_backend_compare_gpu_execution_only.local.yml` | Delete/move local example | Appears unreferenced outside itself. | `rg "gpu_execution_only.local"` clean; GPU suite tests pass. |
| `local_plot.yml` | Delete | No active refs found in inspection beyond file path. | `rg "local_plot"` clean; plot/report tests pass. |
| `planner_compare_extended.yml` | Merge | Extended planner fixture only. | Planner tests migrated; `rg "planner_compare_extended"` clean. |
| `quest_cpu_full_state_benchmark` | Route demotion/removal | Metrics-only route overlaps with exact QuEST route. | Update `local_energy.yml`, matrix config/tests, plot defaults, route registry tests. |
| `upmem_dense_int8_placeholder` | Route removal | Placeholder route remains only for smoke/skip metadata. | Replace smoke assertions and plot label expectations. |
| `summarize`, `plot`, bare `probe` | CLI removal | Older/unclear public value after `report-run` and `simulation-backend-probe`. | `rg` refs clean; reporting/plot tests cover replacements. |

No candidate should be removed until its checks are complete.

## Smallest Next KISS Wave

Wave 2E.30 should consolidate suites only:

1. Remove or merge `smoke_v2.yml` after tests no longer reference it.
2. Delete or move `simulation_backend_compare_gpu_execution_only.local.yml` if
   the reference check remains clean.
3. Delete `local_plot.yml` if reference checks stay clean.
4. Leave route removals for a later wave after suite consolidation is stable.

## Validation Commands

Run from `thesis/implementation`:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src ../.venv/bin/python -m pytest -q
make -n bench-cpu
make -n bench-gpu
make -n bench-upmem-sim
make -n report-latest
make -n compare-latest
rg -n "candidate_name" . --glob '!external/**' --glob '!runs/**' --glob '!build/**'
git diff --check -- .gitmodules thesis/implementation thesis/*.md thesis/legacy
```
