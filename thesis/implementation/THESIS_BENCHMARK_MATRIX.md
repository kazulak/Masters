# Thesis Benchmark Matrix - Wave 2E.46

This document defines the benchmark matrix for the thesis evaluation. It is a
tracked design artifact; generated evidence and comparison outputs remain under
`runs/` and are intentionally ignored by git.

Wave 2E.46 does not generate new benchmark artifacts. The first concrete output
from this matrix should be produced by Wave 2E.47: a new CPU/GPU direct
benchmark sweep.

## Current Evidence Inputs

| Evidence | Path | Role |
|---|---|---|
| CPU evidence | `runs/evidence/cpu_evidence/simulation_backend_compare/20260703_195342` | Current QuEST CPU and Quimb TN evidence. |
| GPU smoke evidence | `runs/evidence/gpu_evidence/simulation_backend_compare/20260703_195327` | Verified QuEST HIP route smoke evidence only. |
| UPMEM simulator evidence | `runs/evidence/upmem_sim_evidence/simulation_backend_compare/20260703_195349` | Current strict UPMEM SDK simulator evidence. |

The existing 4q GPU run proves that the QuEST HIP route works and can emit real
GPU rows. It is not the thesis CPU/GPU benchmark. The thesis CPU/GPU benchmark
requires a new overlapping sweep with CPU and GPU rows for the same circuit
families and qubit counts.

## First Required Output: CPU/GPU Direct Benchmark Sweep

Wave 2E.47 should create and run a new CPU/GPU sweep using only:

- `quest_cpu_full_state_exact`
- `quest_gpu_full_state_exact`

The sweep must use identical deterministic circuit semantics for CPU and GPU
rows. CPU/GPU speedup is valid only when the row pair has the same circuit
family, qubit count, validation status, and measured timing scope.

### Circuit Families

- QRNG
- BV
- XOR
- GHZ/HS
- BB84
- EDC
- random shallow only if already supported by existing generators

### Qubit Sweep

| Tier | Qubits | Use |
|---|---:|---|
| small | 4, 6, 8 | Correctness and low-cost sanity. |
| medium | 10, 12, 14, 16, 18 | Main thesis figure range. |
| stress/manual | 20, 22, 24+ | Manual only, if runtime and memory allow. |

### Measurement Rules

- Use at least 3 measured repeats for timing stability.
- Keep warmup timing separate from measured timing where supported.
- Stop a case cleanly on timeout, memory guard, validation failure, or GPU
  verification failure.
- Do not emit GPU benchmark rows unless `gpu_backend_verified=true` and
  `gpu_program_executed=true`.

### Required Metrics

- CPU runtime.
- GPU runtime.
- CPU/GPU speedup.
- Validation status and error metrics.
- GPU device name.
- Energy only if real sensor data is available later.

### Required 2E.47 Outputs

Evidence should be written under:

```text
runs/evidence/cpu_gpu_sweep/...
```

Derived comparison artifacts should be written under:

```text
runs/comparisons/cpu_gpu/...
```

Required derived artifacts:

- CPU/GPU comparison CSV.
- CPU/GPU Markdown summary.
- Runtime and speedup plot if existing report tooling supports it cleanly.

## Full Thesis Matrix

### Engines

- `quest_cpu_full_state_exact`
- `quimb_tn_exact`
- `quest_gpu_full_state_exact`
- `upmem_tn_sdk_simulator_quantized`

### Boundary Evidence

- `upmem_generic_sweep`

`upmem_tn_sdk_simulator_quantized` is the executable UPMEM SDK simulator
comparison route.

`upmem_generic_sweep` is boundary and coverage evidence, not a normal baseline
engine.

### Matrix Roles

| Route or evidence | Execution model | Target | Thesis role |
|---|---|---|---|
| `quest_cpu_full_state_exact` | full-state | CPU | Serious CPU full-state baseline. |
| `quimb_tn_exact` | tensor-network | CPU | Serious CPU tensor-network baseline. |
| `quest_gpu_full_state_exact` | full-state | GPU | Verified QuEST HIP GPU baseline. |
| `upmem_tn_sdk_simulator_quantized` | tensor-network | UPMEM SDK simulator | Executable quantized UPMEM simulator comparison route. |
| `upmem_generic_sweep` | tensor-network boundary scan | UPMEM SDK simulator | Boundary evidence for generic UPMEM coverage. |

### Sweep Levels

| Sweep | Purpose | Expected behavior |
|---|---|---|
| small correctness | Verify output agreement and route semantics. | Full validation where feasible. |
| medium thesis figure | Main CPU/GPU and CPU/TN figures. | Repeats and clean timing summaries. |
| stress/manual boundary | Find scaling and support boundaries. | Manual invocation with explicit stop reasons. |

Stop conditions for all sweeps:

- timeout
- memory guard
- validation failure
- unsupported reason
- GPU verification failure
- explicit resource skip

## Scientific Comparison Rules

- QuEST CPU full-state vs QuEST GPU full-state is a valid direct backend
  comparison.
- Quimb TN vs QuEST full-state is a valid algorithm/backend comparison, not the
  same execution model.
- UPMEM SDK simulator vs CPU/GPU is a functional/runtime-path comparison, not
  hardware speedup.
- UPMEM generic sweep is boundary evidence only and must not be presented as a
  universal TN coverage claim.
- SDK simulator timing may be reported only as SDK simulator timing.
- CPU/GPU speedup is valid only for same circuit, same size, same QuEST
  semantics, and measured CPU/GPU rows from the same sweep.

## Required Thesis Outputs

| Output | Required artifacts | Notes |
|---|---|---|
| CPU vs GPU runtime by circuit size | CSV, Markdown, plot if supported | First required 2E.47 output. |
| CPU vs GPU speedup by circuit family and size | CSV, Markdown, plot if supported | Only same-case CPU/GPU rows. |
| CPU vs Quimb vs GPU runtime by circuit family | CSV, Markdown, plot | Algorithm/backend comparison. |
| UPMEM supported/unsupported boundary table | CSV, Markdown | Include blocker reasons. |
| UPMEM quantized accuracy/error table | CSV, Markdown | Simulator path evidence only. |
| Energy table | CSV, Markdown | Optional, only with real sensor data. |

## Wave 2E.47 Implementation Plan

1. Add a CPU/GPU sweep suite or command that runs overlapping
   `quest_cpu_full_state_exact` and `quest_gpu_full_state_exact` rows across the
   defined circuit families and qubit sizes.
2. Keep the existing 4q GPU evidence as route-smoke validation only.
3. Run the new CPU/GPU sweep after GPU verification succeeds.
4. Write evidence under `runs/evidence/cpu_gpu_sweep/...`.
5. Write derived comparison outputs under `runs/comparisons/cpu_gpu/...`.
6. Verify every CPU/GPU speedup row has matching circuit family, qubit count,
   validation status, and measured CPU/GPU timing.
7. Then extend the matrix outputs to CPU + Quimb + GPU + UPMEM using the full
   thesis matrix rules.

