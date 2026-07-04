# Quantum Bench Architecture

The active implementation is a route-aware benchmark runtime for exact quantum
circuit simulation. Its current system model is intentionally simple:

```text
quantum circuit
  -> tensor network
  -> TaskGraph
  -> route execution
  -> normalized_records.jsonl
```

The CPU owns planning, orchestration, dispatch, validation, and reporting.
Accelerators execute only the work that an explicit route/backend contract says
they executed.

## Active Backend Roles

| Role | Current implementation | Notes |
|---|---|---|
| CPU full-state | `quest_cpu_full_state_exact` | Serious full-state baseline for comparable deterministic circuits. |
| CPU tensor-network | `quimb_tn_exact` | Serious TN baseline. Quimb/cotengra is the main TN evidence path. |
| Internal CPU TN | `cpu_tn_einsum_exact` | Small/debug/diagnostic route only; failures are internal engine limits, not TN limits. |
| Optional GPU | `quest_gpu_full_state_exact` | Emits benchmark rows only after verified real GPU execution. |
| UPMEM SDK simulator | `upmem_tn_sdk_simulator_quantized` and strict UPMEM runtime commands | Real UPMEM SDK DPU programs in simulator mode; not hardware timing. |

QuEST CPU and Quimb TN are the serious CPU baselines. GPU evidence is optional
and hardware-dependent. UPMEM SDK simulator evidence proves code-path execution
through SDK DPU programs, but hardware speedup is not applicable.

CPU/GPU full-state comparison has separate correctness and performance tiers:

- `state_output_mode=full_dump`, `validation_method=full_statevector`:
  full-output exactness evidence under configured statevector caps.
- `state_output_mode=none`, `validation_method=native_status_gate_counts`,
  `performance_tier=true`: metrics-only performance evidence. These rows must
  include `exact_output_comparable=false` and
  `full_statevector_validation_available=false`.

## UPMEM Status

Current UPMEM work is bounded and explicit:

- Dense GEMM bridge paths exist for selected L1/L2-style tasks.
- A generic UPMEM SDK loop fallback exists for bounded binary contractions.
- Strict generic-only TaskGraph execution exists for small supported plans.
- The SDK simulator path can run quantized int8/int32 and unquantized float32
  generic modes for attribution.
- CPU fallback is not allowed inside strict UPMEM runtime tensor production.

Current limitation:

> Bounded generic UPMEM contraction exists, but fully general UPMEM TN
> contraction does not yet exist.

The detailed historical audit evidence is archived at
[../legacy/audits/upmem_kernel_architecture_audit.md](../legacy/audits/upmem_kernel_architecture_audit.md).

Missing UPMEM capabilities include general tiling for arbitrary tensor
contractions, multi-DPU distribution, PID-Comm orchestration, native
transpose/slicing kernels, hardware execution, and full general TN coverage.

Parallelism goals and evidence requirements are tracked separately in
[docs/parallelization_roadmap.md](docs/parallelization_roadmap.md). That roadmap
distinguishes modeled opportunity from executed slicing, frontier, GPU, and
UPMEM/PIM parallelism.

## Artifact Boundary

Benchmark execution produces evidence:

```text
runs/evidence/<suite_id>/<route_label>/<run_id>/
```

Derived comparison/report generation produces analysis:

```text
runs/comparisons/<suite_id>/<comparison_type>/<comparison_id>/
```

`runs/latest` points only to the latest evidence run. `normalized_records.jsonl`
is the canonical source for report and comparison commands. Evidence runs should
contain raw execution evidence, manifests, summaries, and normalized records
only.

Derived tables, validation summaries, plot-source CSVs, and figures belong under
comparison/report output directories, not inside evidence runs. Build/cache
outputs belong under ignored locations such as `build/`, `.pytest_cache/`, and
native `bin/` or `build/` directories. Historical timestamped run folders that
pre-date the `runs/evidence` and `runs/comparisons` split are legacy generated
diagnostics, not canonical thesis evidence.

Research benchmark packs are derived comparison artifacts under
`runs/comparisons/research_pack/...`; their methodology and claim boundaries are
documented in [docs/research_benchmark_methodology.md](docs/research_benchmark_methodology.md).

The older `run --suite` smoke path still writes concrete legacy evidence files
under `raw/`, `validation/`, and `metrics/`. Those files are not inputs to
`report-run`; normalized benchmark/report workflows use `normalized_records.jsonl`
as the source artifact.

## Route And Device Rules

- Do not report simulator timing as hardware speedup.
- Do not emit GPU rows unless real GPU execution was verified.
- Do not present `state_output_mode=none` rows as full-statevector validation
  evidence.
- Do not emit UPMEM benchmark rows unless strict UPMEM execution actually ran
  SDK DPU programs and `cpu_fallback_used=false`.
- Keep route IDs, developer backend IDs, and internal UPMEM execution classes
  distinct.
- Keep SimplePIM, PID-Comm, ATiM, SparseP, PRISM, PyGim, TransPimLib, and
  PIM-LLM GEMM as candidate/future implementation references unless a real
  integrated execution path exists.

## Future Simplifications

- Keep the suite family small: `smoke`, `cpu_evidence`, `gpu_evidence`,
  `cpu_gpu_sweep`, `upmem_sim_evidence`, `upmem_generic_sweep`,
  `manual_large`, and `diagnostics`.
- Keep diagnostic suites and commands out of README and Makefile defaults.
- Defer route cleanup until suite cleanup is stable.
- Reduce repeated route/status vocabulary where it does not affect artifact
  semantics.
- Keep the Makefile as the main execution surface.
- Avoid reintroducing multiple active roadmap documents.
- Remove dead placeholder routes/configs only after reference checks and tests
  prove they are unused.
- Keep diagnostics in CLI help or tests unless one diagnostics page becomes
  clearly necessary.
