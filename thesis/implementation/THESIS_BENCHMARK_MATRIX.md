# Thesis Benchmark Matrix

This document fixes the benchmark questions before further UPMEM kernel work.
Suite YAML files are the executable specification; this document defines how
their evidence is interpreted.

## Canonical Local Grid

Circuit families, chosen for continuity with PIMutation:

```text
QRNG, BV, XOR, BB84, EDC, HS
```

Canonical local sizes:

```text
8, 10, 12, 14, 16, 18, 20 qubits
```

This is 42 family/size cases. The correctness tier remains a smaller 4/8/12/16q
semantic check; it is not the performance grid.

## Executable Groups

| Group | Suite | Routes/modes | Repeats | Scientific purpose |
| --- | --- | --- | ---: | --- |
| Full-state correctness | `configs/suites/manual/thesis_full_state_correctness.yml` | QuEST CPU + verified QuEST GPU, full dump | 1 | Prove CPU/GPU semantic agreement on representative 4/8/12/16q cases |
| Full-state performance | `configs/suites/manual/thesis_full_state_cpu_gpu.yml` | QuEST CPU + verified QuEST GPU, metrics only | 5 + 1 warmup | Compare compute time and process wall time over all 42 cases |
| CPU TN | `configs/suites/manual/thesis_cpu_tn_quimb.yml` | QuEST CPU anchor, Quimb unsliced, Quimb sliced | 3 | Compare full-state and external TN execution, path cost, slicing, and memory proxies over all 42 shallow cases |
| Same-path quantization | `configs/suites/manual/thesis_tn_paths_quantization.yml` | float64 and int8 internal TaskGraph replay | 1 | Attribute runtime and error to per-contraction quantization on an identical path; diagnostic, not serious TN baseline |
| Planner candidates | `configs/suites/manual/thesis_planner_compare.yml` | `opt_einsum` greedy and auto | planning once | Compare plan FLOPs, peak intermediates, modeled transfers/tiling, and UPMEM pressure |
| UPMEM boundary | `configs/suites/manual/thesis_upmem_quantization_boundary.yml` | Same internal TaskGraph, float32 and int8 strict generic UPMEM SDK simulator | 1 | Find supported/unsupported boundary and attribute transfer/error/runtime changes to quantization |
| Internal parallelism | `configs/suites/manual/research_internal_parallelism.yml` | sequential/frontier/hybrid internal TaskGraph | 1 | Diagnostic architecture evidence only |

The CPU/GPU performance suite uses deeper repeated circuits to reduce startup
dominance. The CPU TN suite uses shallow exact circuits on the same canonical
family/size grid because deep full-output TN contraction can be a different,
much larger computational problem. Depth/repeat count is always present in the
resolved suite and must be included when interpreting results.

## Required Comparisons

### Full-State CPU Versus GPU

Valid direct comparison:

```text
quest_cpu_full_state_exact vs quest_gpu_full_state_exact
```

Pair only equal case, repeat, output contract, validation method, and timing
scope. Report:

- median CPU and GPU compute time;
- CPU/GPU compute speedup (`CPU time / GPU time`);
- process wall-time ratio separately;
- matched repeat count and spread;
- verified GPU device/runtime metadata.

Performance-tier rows are metrics-only and are not full-statevector validation
evidence. Correctness comes from the separate full-dump tier.

### Full-State CPU Versus CPU TN

Valid algorithm/backend comparison on the exact same shallow circuit:

```text
quest_cpu_full_state_exact vs quimb_tn_exact vs quimb_tn_sliced_exact
```

Report planning time, contraction time, total wall time, peak intermediate
proxy, task/path metrics, output agreement, slice count, and
`slicing_flop_ratio`. This is not same-implementation speedup: QuEST and Quimb
use different simulation models.

### Contraction Path Choice

The planner suite emits one content-addressed TaskGraph per candidate. Report:

- estimated FLOPs;
- peak intermediate bytes;
- host-to-DPU, DPU-to-host, and MRAM-to-WRAM estimates;
- tiling-required task count and estimated tile parallelism;
- modeled UPMEM pressure score/rank;
- planning time and contraction-plan hash.

Planner rows are modeled path evidence. They do not claim execution speedup.
The chosen future UPMEM-aware objective should be compared with the unchanged
greedy/auto baselines using these fields.

### Same-Plan CPU Versus UPMEM

The strict UPMEM suite generates one internal TaskGraph and runs:

```text
CPU exact TaskGraph replay
UPMEM SDK simulator generic float32
UPMEM SDK simulator generic int8/int32
```

A pair is valid only when `contraction_plan_hash` matches. Report route time,
SDK-simulator kernel time, actual host/DPU bytes, quantization/dequantization
time, invocation count, error, and unsupported reason. No ratio is labeled
hardware speedup.

### Quantization Attribution

Compare float32 and int8 only within the same strict generic UPMEM route and
plan. Required outputs:

- float32/int8 route-time ratio;
- float32/int8 simulator-kernel ratio;
- transfer-volume ratio;
- max-absolute and L2 error against the full-precision TaskGraph reference;
- clipping/saturation information when available.

CPU path-replay quantization remains a diagnostic for numerical behavior and
host conversion overhead. It is not evidence that quantization is slower on a
DPU.

## PIMutation Comparison

The six shared family names provide a recognizable comparison surface. The
thesis should synthesize, per family:

| Question | This implementation | PIMutation context |
| --- | --- | --- |
| Baseline model | QuEST full state and Quimb TN | State-vector PIM simulation |
| Scaling variable | qubits, gate count/depth, TN path width | qubits/gate workload reported by PIMutation |
| PIM execution | TN contraction TaskGraph (simulator now, hardware later) | state-vector gate operations on UPMEM |
| Quantization | same-plan float32 vs int8 contraction | fixed/integer PIM motivation |
| Specialization | future permutation/layout/sparse kernels | gate-aware data operations inspire candidates |

This is a relative scientific comparison, not a claim that both systems execute
identical kernels, paths, software versions, or hardware.

## Required Derived Tables

All live under `thesis_results/current/tables/` after promotion:

- `per_case_route_stats.csv`
- `cpu_gpu_performance_summary.csv`
- `full_state_tn_comparison.csv`
- `paired_speedups.csv`
- `planner_comparison.csv`
- `same_plan_execution.csv`
- `upmem_quantization_attribution.csv`
- `unsupported_cases.csv`
- `validation_summary.csv`
- `route_capability_matrix.csv`

## Required Human-Readable Figures

All live under `thesis_results/current/plots/`, each with a source CSV in the
tables directory:

- CPU/GPU runtime and speedup by family/qubits;
- CPU TN runtime, planning/contraction split, path FLOPs, and peak memory;
- Quimb slicing FLOP ratio;
- planner FLOPs versus modeled UPMEM pressure;
- UPMEM support boundary and validation error;
- float32/int8 UPMEM runtime, transfer, and error attribution;
- same-plan CPU replay versus UPMEM SDK-simulator route timing.

If a figure lacks valid input rows, the plot manifest records a skip reason.
The report must not fabricate empty or incomparable values.

## Cluster Extension

The local matrix is rerun on the final CPU/NVIDIA/UPMEM environments using the
same suite semantics where possible. Environment/device records must identify:

- CPU model, core/thread policy, BLAS thread variables;
- GPU model, driver/toolkit, real program verification, synchronization;
- UPMEM SDK version, simulator versus hardware, DPU count, task assignment;
- Git commit, resolved suite, dependency versions, and semantic/plan hashes.

NVIDIA full-state or TN software is added only after real execution is verified.
Physical UPMEM rows use a distinct hardware execution mode and are never merged
with SDK-simulator timing.

## Stop/Boundary Rules

A case is retained as unsupported rather than hidden when it reaches a rank,
element, memory, timeout, planner, validation, or device boundary. A manual
stress run may stop before larger sizes after such a boundary is established.
The canonical local grid remains fixed so regressions and improvements are
visible across waves.
