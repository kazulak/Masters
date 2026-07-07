# Parallelization Implementation Strategy

This document maps the tensor-network parallelization roadmap to small future
implementation waves. It is an implementation strategy, not an implementation
claim: the serious Quimb TN baseline and strict UPMEM SDK simulator baseline
remain sequential, while internal TaskGraph frontier and hybrid routes provide
diagnostic executed parallelism evidence only.

## Shared Metadata Primitives

Future parallelization work should extend the existing `normalized_records.jsonl`
artifact path instead of creating a new reporting framework. Add fields
incrementally, and keep missing values explicit as `not_applicable`,
`unsupported`, or `modeled` rather than implying execution.

| Primitive | Purpose | Initial values or examples |
|---|---|---|
| `parallelism_mode` | Names the selected mode. | `sequential`, `slicing`, `frontier`, `intra_contraction`, `hybrid`, `modeled_only`, `not_applicable`. |
| `parallelism_evidence_type` | Separates executed evidence from analysis. | `executed`, `modeled`, `configured_not_executed`, `unsupported`, `not_applicable`. |
| `execution_plan_*` | Identifies the plan used by a route. | plan id, plan kind, worker count, status, executed flag. |
| `slicing_*` | Captures slicing configuration and reconstruction. | backend, sliced indices, slice count, FLOP ratio/change, reconstruction status. |
| `frontier_*` | Captures graph-level scheduling evidence. | wave count, max/mean frontier width, executed parallel task count, scheduler worker count. |
| `backend/device metadata` | Keeps comparisons fair. | CPU thread and BLAS settings, GPU device/runtime, UPMEM DPU execution mode. |
| `timing breakdown` | Preserves attribution. | Reuse existing total, setup, lowering, transfer, compute, validation, output fields; add mode-specific timing only when needed. |
| `validation/output contract` | Prevents overclaiming. | Reuse `validation_method`, `output_contract`, `exact_output_comparable`, and metrics-only flags. |

Default records for existing routes should report `parallelism_mode=sequential`
or `not_applicable` and `parallelism_evidence_type=executed` only when that
route actually executed. Modeled frontier analysis must remain
`parallelism_evidence_type=modeled`.

## Route Strategy

Do not replace current serious baselines:

- `quest_cpu_full_state_exact` remains the serious CPU full-state baseline.
- `quimb_tn_exact` remains the serious unsliced CPU tensor-network baseline.
- `quest_gpu_full_state_exact` remains the verified full-state GPU baseline.
- `upmem_tn_sdk_simulator_quantized` remains the strict sequential UPMEM SDK
  simulator baseline.

The following route IDs are now implemented and should remain additive rather
than replacing existing baselines:

- `quimb_tn_sliced_exact`: explicit Quimb/cotengra
  slicing evidence.
- `cpu_tn_frontier_exact`: internal TaskGraph
  frontier scheduler prototype.
- `cpu_tn_hybrid_sliced_frontier_exact`: diagnostic internal slice-aware
  TaskGraph plus frontier scheduler prototype.

Use additive route IDs when different execution modes need to appear
side-by-side in one comparison suite. Use route options only when the mode is a
minor configuration of the same route and does not need independent baseline
identity in reports.

Route-specific guidance:

- CPU slicing should start with Quimb/cotengra because Quimb is already the
  serious CPU TN baseline.
- CPU frontier scheduling should start as an internal/diagnostic route because
  it exercises the local TaskGraph executor rather than the serious external TN
  baseline.
- Hybrid slicing plus frontier is available only as the diagnostic internal
  route `cpu_tn_hybrid_sliced_frontier_exact`; it is not a serious TN
  baseline and should not be used for speedup claims without a separate
  performance/scaling methodology.
- GPU TN must be separate from QuEST GPU full-state. Do not report QuEST
  full-state GPU speedup as GPU TN speedup.
- UPMEM multi-DPU execution must remain under the unified UPMEM runtime concept.
  Do not introduce `upmem_l1`, `upmem_l2`, or `upmem_l3` top-level benchmark
  routes.

## Artifact And Reporting Strategy

Parallelism fields should be enough to compare modes honestly:

- distinguish executed parallelism from modeled potential;
- distinguish slicing FLOP ratio/change from useful work;
- distinguish graph-level scheduler parallelism from intra-contraction tiling;
- distinguish SDK simulator timing from UPMEM hardware timing;
- distinguish exact output validation from metrics-only performance rows.

Reports should derive the following tables from normalized records:

- parallelism mode comparison table;
- slicing configuration and FLOP ratio/change table;
- frontier concurrency table;
- hybrid component table;
- timing breakdown table;
- validation/output-contract table;
- CPU/GPU/UPMEM capability matrix.

`compare-results` now emits `parallelism_capability_matrix.csv` and
`parallelism_capability_matrix.md` whenever parallelism evidence is present.
The matrix is a claim-boundary artifact: it records route role, execution
target, evidence type, same-family timing group, and whether a speedup claim is
allowed from the available evidence. Modeled UPMEM assignment and SDK simulator
rows remain marked as non-hardware-speedup evidence.

Current implemented evidence can be gathered with:

```bash
make parallelism-report
```

This target compares existing executed CPU slicing/frontier/hybrid diagnostics
with modeled UPMEM assignment evidence. It is a reporting workflow only; it
does not implement UPMEM multi-DPU execution. The comparison output includes
`parallelism_mode_summary.*` and `parallelism_capability_matrix.*` artifacts.

No fake speedup fields should be added. Any ratio must state its timing scope
and denominator route. UPMEM SDK simulator timing may be reported only as SDK
simulator timing.

## Test Strategy

Future implementation waves should add tests at the point where the behavior is
introduced:

- correctness tests comparing output to existing exact baselines;
- deterministic slice reconstruction tests;
- no duplicated or missing contractions in sliced and frontier execution;
- frontier dependency safety tests;
- CPU thread and BLAS metadata capture tests;
- GPU verification and no-CPU-fallback tests for any GPU TN route;
- UPMEM strict-path tests with `cpu_fallback_used=false`;
- metadata/reporting tests for executed versus modeled parallelism;
- metrics-only rows must never be treated as exact validation.

## Staged Implementation Waves

### 2E.51 - Passive Parallelism Metadata Foundation

Add default parallelism fields to normalized records for existing routes without
changing execution behavior. Existing rows should remain semantically identical
except for explicit `parallelism_mode` and `parallelism_evidence_type` fields.

Reuse:

- current normalized record writer;
- existing route metadata and timing fields.

Defer:

- new benchmark suites beyond metadata tests.

### 2E.52 - Explicit Quimb/Cotengra Slicing Evidence

Add an explicit slicing route for Quimb/cotengra:
`quimb_tn_sliced_exact`. Keep `quimb_tn_exact` as the unsliced serious CPU TN
baseline so both routes can appear side by side in one diagnostic suite.

Required evidence:

- slice count and sliced index metadata;
- `slicing_flop_ratio` and cost-source metadata;
- deterministic slice reconstruction;
- full output agreement for small correctness cases.

This is slicing evidence only. It does not claim slice-worker parallel speedup
unless a later wave runs slices concurrently and records worker metadata.

### 2E.53 - CPU Frontier TaskGraph Prototype

Add an internal/diagnostic TaskGraph frontier scheduler prototype:
`cpu_tn_frontier_exact`. It executes ready TaskGraph nodes only when
dependencies are satisfied and must prove no duplicated or missing
contractions.

Required evidence:

- frontier wave IDs;
- executed parallel task count;
- scheduler worker count;
- scheduler overhead;
- final output agreement.

### 2E.54 - CPU Slicing Vs Frontier Comparison

Add a focused comparison suite for sequential Quimb, sliced Quimb/cotengra, and
frontier TaskGraph execution. This wave should compare modes, not optimize them.

Required evidence:

- matched circuits and validation settings;
- controlled CPU thread/BLAS settings;
- timing, memory, FLOP inflation, and validation tables.

### 2E.55 - Hybrid Slicing Plus Frontier Experiment

The Wave 2E.55 feasibility verdict is tracked in
[hybrid_slicing_frontier_design.md](hybrid_slicing_frontier_design.md): do not
call side-by-side Quimb slicing and internal TaskGraph frontier rows a hybrid.
Later waves added the shared internal representation and diagnostic hybrid
route, but the original warning remains the key thesis-safety rule.

Required evidence:

- slice IDs and frontier wave IDs in task logs;
- duplicate contraction checks;
- reconstruction validation;
- per-component timing.

### 2E.56 - Internal Slice-Aware TaskGraph Model

Add a model-only internal slice-aware TaskGraph representation before claiming
any hybrid execution. This stage provides slice task metadata and reconstruction
requirements without executing numerical slice contractions.

Required evidence:

- slice model task count and slice count;
- `slice_model_execution_status=model_only`;
- `hybrid_ready=false`;
- no frontier metadata that implies executed hybrid behavior.

### 2E.57 - Executable Internal Hybrid Spike

Add a diagnostic internal route only after sequential slice reconstruction and
frontier scheduling run in the same execution path:
`cpu_tn_hybrid_sliced_frontier_exact`.

Required evidence:

- `parallelism_mode=hybrid`;
- `parallelism_evidence_type=executed`;
- `hybrid_components=["slicing", "frontier"]`;
- completed slice reconstruction and final validation;
- source task counts and expanded execution-node counts;
- duplicate, missing dependency, and dependency-violation checks.

This route is diagnostic. It must not be promoted to the serious TN baseline or
used for speedup claims without a separate performance/scaling methodology.

### 2E.58+ - Benchmark Evidence Readiness

Research benchmark packs and reports should keep slicing, frontier, hybrid,
GPU, and UPMEM claims thesis-safe. Parallelization evidence must stay derived
from normalized records, with diagnostic routes clearly separated from serious
baselines.

### Next Stage - GPU TN Feasibility

The feasibility plan is [gpu_tn_feasibility.md](gpu_tn_feasibility.md), and
`simulation-backend-probe` now reports a feasibility-only GPU tensor-network
candidate section. It investigates real GPU tensor-network backends separately
from QuEST full-state GPU. Candidate families include cuTensorNet/cuQuantum,
CUDA-Q `tensornet`, Qiskit Aer `tensor_network`, and Quimb/cotengra with a
real GPU array backend.

Required evidence:

- verified GPU device metadata;
- no CPU fallback;
- synchronization status for timing;
- exact or clearly labeled metrics-only validation.

Current boundary:

- no GPU TN route is registered;
- no GPU TN benchmark records are emitted;
- the next implementation step is a minimal candidate execution spike outside
  benchmark-row emission.

### 2E.59 - UPMEM Modeled Multi-DPU Assignment Report

The design contract is tracked in
[upmem_multi_dpu_scheduling_design.md](upmem_multi_dpu_scheduling_design.md).
The first modeled implementation is now available as:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src ../.venv/bin/python -m quantum_bench.bench upmem-multi-dpu-assignment \
  --suite configs/suites/upmem_sim_evidence.yml \
  --dpu-groups 4
```

Required evidence goals:

- task assignment plan;
- DPU group IDs;
- transfer and synchronization cost model;
- strict no CPU contraction fallback invariant.

This command emits `upmem_parallelism_evidence_type=modeled`, writes
`upmem_multi_dpu_assignment_plan.json`, and leaves the existing sequential SDK
simulator runtime unchanged. It is not executed multi-DPU evidence.

### Future Stage - UPMEM Multi-DPU Prototype

Implement only after the design can be tested against real SDK/hardware
constraints. Hardware timing and speedup claims require real hardware execution;
SDK simulator timing remains simulator evidence.

The readiness gate is
[upmem_multi_dpu_prototype_readiness.md](upmem_multi_dpu_prototype_readiness.md).
It records the current boundary: the generic native host allocates one DPU per
task invocation, and the next safe implementation is an SDK simulator
frontier-scheduled prototype that consumes assignment plans without claiming
hardware execution.

## Deferred Decisions

The following choices should be made during the implementation wave that first
needs them:

- GPU TN backend selection;
- candidate route ID for any verified GPU TN backend;
- UPMEM multi-DPU communication substrate and PID-Comm role.
