# Parallelization Implementation Strategy

This document maps the tensor-network parallelization roadmap to small future
implementation waves. It is an implementation strategy, not an implementation
claim: current TaskGraph execution and strict UPMEM SDK simulator execution
remain sequential until later waves add and validate parallel execution.

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

Candidate future route IDs may be useful for planning, but final names must be
confirmed during implementation based on artifact and report compatibility:

- `quimb_tn_sliced_exact`: candidate route ID for explicit Quimb/cotengra
  slicing evidence.
- `cpu_tn_frontier_exact`: candidate route ID for an internal TaskGraph
  frontier scheduler prototype.

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
- Hybrid slicing plus frontier should be deferred until both individual modes
  have independent correctness and metadata coverage.
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
[hybrid_slicing_frontier_design.md](hybrid_slicing_frontier_design.md).
True hybrid execution should be deferred until a shared slice-aware TaskGraph
representation exists. Do not call side-by-side Quimb slicing and internal
TaskGraph frontier rows a hybrid.

Required evidence:

- slice IDs and frontier wave IDs in task logs;
- duplicate contraction checks;
- reconstruction validation;
- per-component timing.

### 2E.56 - GPU TN Feasibility

Investigate a real GPU tensor-network backend separately from QuEST full-state
GPU. Candidate families include cuTensorNet/cuQuantum/CUDA-Q/Qiskit Aer GPU or
another exact GPU TN path that can prove real device execution.

Required evidence:

- verified GPU device metadata;
- no CPU fallback;
- synchronization status for timing;
- exact or clearly labeled metrics-only validation.

### 2E.57 - UPMEM Multi-DPU Scheduling Design

Design task-to-DPU assignment and reduction/synchronization metadata before
implementing multi-DPU execution. Use current frontier analysis as modeled input
only.

Required evidence goals:

- task assignment plan;
- DPU group IDs;
- transfer and synchronization cost model;
- strict no CPU contraction fallback invariant.

### 2E.58+ - UPMEM Multi-DPU Prototype

Implement only after the design can be tested against real SDK/hardware
constraints. Hardware timing and speedup claims require real hardware execution;
SDK simulator timing remains simulator evidence.

## Deferred Decisions

The following choices should be made during the implementation wave that first
needs them:

- final route IDs for sliced Quimb/cotengra and frontier TaskGraph execution;
- whether slicing is exposed as a route option, additive route, or both;
- exact cotengra slicing API and optimizer configuration;
- frontier worker implementation mechanism;
- GPU TN backend selection;
- UPMEM multi-DPU communication substrate and PID-Comm role.
