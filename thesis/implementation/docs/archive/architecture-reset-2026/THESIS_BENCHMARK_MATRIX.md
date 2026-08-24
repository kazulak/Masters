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
| Planner candidates | `configs/suites/manual/thesis_planner_compare.yml` | `opt_einsum`, cotengra objectives, custom UPMEM greedy | planning once | Compare plan FLOPs, peak intermediates, modeled transfers/tiling, and a fixed-policy UPMEM objective |
| Planner sensitivity | `configs/suites/manual/thesis_planner_sensitivity.yml` | custom UPMEM greedy under six named scenario profiles | planning once | Show objective sensitivity without presenting scenario weights as measured hardware constants |
| Planner semantic v2 | `configs/suites/manual/thesis_planner_semantic_v2.yml` | Standard opt_einsum/cotengra baselines plus `custom_upmem` `upmem_path_cost_v2` projected-prefix planner | planning once | Exercise small quantum cases and all controlled motifs under the v2 numeric/modeling contract |
| Planner sensitivity v2 | `configs/suites/manual/thesis_planner_sensitivity_v2.yml` | Standard opt_einsum/cotengra baselines plus six v2 `custom_upmem` profiles | planning once | Measure modeled projected-prefix sensitivity; profile weights remain scenario assumptions, not hardware constants |
| UPMEM boundary | `configs/suites/manual/thesis_upmem_quantization_boundary.yml` | Same internal TaskGraph, float32 and int8 strict generic UPMEM SDK simulator | 1 | Find supported/unsupported boundary and attribute transfer/error/runtime changes to quantization |
| Internal parallelism | Historical resolved artifacts | sequential/frontier/hybrid internal TaskGraph | 1 | Historical diagnostic evidence; no longer a runnable Phase A route |
| M4.6 tasklet development sweep | `configs/suites/upmem_hardware_taskgraph_resident_m4_6_tasklet_scaling.yml` | Physical resident TaskGraph, one DPU, tasklets 1/2/4/8/16, two paths, two numeric modes | 7 | Validate tasklet ownership, DPU-cycle metadata, and correctness on small development cases; no final scaling claim |
| M5.1 output partition probe | Fixed CLI fixture: `make upmem-hw-m5-1` | One bounded real float32 contraction, 1/2/4 DPUs, exclusive output-tile ownership | 1 | Validate bounded multi-DPU output ownership and exact CPU agreement; functionality only |
| M5.2 contracted partition probe | Fixed CLI fixture: `make upmem-hw-m5-2` | One bounded real float32 contraction, 1/2/4 DPUs, host-mediated ascending-DPU reduction | 1 | Validate bounded partial-sum reconstruction; functionality only |
| M5 execution-plan-v3 lane | `make upmem-hw-m5-plan` / `make upmem-hw-m5` | One-rank configured 1/2/4/8/16/32/64 DPUs and tasklets 8; 5 workloads; real and synthetic diagnostics; output/contracted partitioning; float32/per-task int8 | 2 + 7 | Physically accepted bounded development study: 140 cells, 644 measured rows, 48 partition-incompatible unsupported rows, 0 failures; same-route diagnostics only |
| M5.4 corrected execution-plan-v3 lane | `make upmem-hw-m5-4-plan` / `make upmem-hw-m5-4-smoke` / `make upmem-hw-m5-4` | Same bounded one-rank single-contraction matrix; float32 versus host-packed int8; one bulk set launch per repetition | 2 + 7 | Physically accepted at source `eef42e4`: all 10 exact-int32, no-fallback, checksum-policy, payload, cycle, strong-scaling, and weak-scaling gates passed; 644 measured rows, 48 explicit unsupported rows, and 0 failures; bounded same-route diagnostics only |
| M5.5 whole-circuit baseline | `make m5-circuit-plan` / `make m5-circuit-smoke` / `make m5-circuit-study` | Same circuit/TN/plan through NumPy CPU and physical v4 engines; opt_einsum greedy and pinned cotengra FLOP-greedy paths; float32 and host-packed int8; host-managed graph intermediates; sequential TaskGraph tasks; output/K-tiled intra-task DPU/rank execution | smoke 1 + canonical 42 cases, warmup 1 + 3 repeats; scaling 1 + 5 repeats; large boundary 1 | Physical development contracts passed on clean source `b550c46`: canonical `1008/1008`, scaling `80/80`, and large `96` completed plus `24` explicit unsupported rows, with 0 failures. Fully active scaling is valid through 16 DPUs; 32/64/128 provisioned rows used 16 active DPUs and the two-rank row used one active rank. The large one-repeat profile is boundary evidence only. No energy, provider, calibrated-planner, frontier-concurrency, or graph-wide DPU-residency claim |
| M5.3 PID-Comm qualification | `make upmem-pidcomm-compatibility` | Pinned PID-Comm compile/link qualification under the installed ETH SDK | allocation-free | Record compatibility or an explicit blocker; current SDK 2023.1 blocker prevents physical execution |

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

The planner suite emits one content-addressed TaskGraph per candidate. It keeps
standard-library candidates as permanent baselines and adds a deterministic
`custom_upmem` greedy generator, not merely post-hoc candidate rescoring.
Report:

- estimated FLOPs;
- peak intermediate bytes;
- host-to-DPU, DPU-to-host, and MRAM-to-WRAM estimates;
- tiling-required task count and estimated tile parallelism;
- modeled UPMEM pressure score/rank;
- fixed execution-policy assumptions, objective version, normalization, and
  named weight profile;
- modeled components: FLOPs, largest intermediate, intermediate writes,
  host-to-DPU and DPU-to-host bytes, MRAM-to-WRAM bytes, local work,
  synchronization events, numerical penalty, WRAM pressure, tile count,
  feasibility, and rejection reasons;
- modeled Pareto status and selected candidate within each weight profile;
- planning time and contraction-plan hash.

Planner rows are modeled path evidence. They do not claim execution speedup.
Their fixed single-DPU policy is a literature-informed planning scenario, not a
calibrated UPMEM hardware predictor. Controlled planner motifs are separately
labeled modeled-only and not real quantum circuits.

Planner objective versions are separate evidence contracts. The existing
`upmem_path_cost_v1` suites are historical v1 evidence: their custom policy is
the real float32 generic model and complex quantum paths are retained as
standard-planner baselines, with modeled UPMEM infeasibility where applicable.
The additive v2 suites use `upmem_path_cost_v2` and the projected-prefix greedy
selection scope. V2 accepts complex-typed inputs whose imaginary values are
zero as real-valued work; genuinely nonzero complex inputs are modeled as split
real/imaginary components under the bounded policy where supported. A v2
selection remains a modeled planner result, not an execution or hardware
performance result.

The v2 semantic and sensitivity suites deliberately use small quantum cases
alongside the controlled chain, tree, star, cycle, grid, and FLOP/memory
trade-off motifs. The motifs remain `not_real_quantum_circuit=true` and
`execution_scope=model_only`; they validate planner behavior only.

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
- probability maximum-absolute and L1 error when validation records contain
  probability outputs; no amplitude error is relabeled as probability error;
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
- planner FLOPs versus normalized modeled PIM objective, component scores,
  selection, Pareto status, and scenario sensitivity;
- UPMEM support boundary and validation error;
- float32/int8 UPMEM runtime, transfer, and error attribution;
- same-plan CPU replay versus UPMEM SDK-simulator route timing.

If a figure lacks valid input rows, has zero variance, or is not yet
implemented, the report retains a visible TODO PNG at the expected path. The
plot manifest records a `generated_todo_*` status and exact reason; the figure
is listed separately from valid scientific figures. The report must not
fabricate empty or incomparable values.

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

M5.5 is the first whole-circuit benchmark surface for the architecture. Its
CPU and physical UPMEM rows share circuit, tensor-network, and contraction-plan
hashes when the same plan is selected. The CPU route is a same-plan reference;
QuEST full-state, Quimb/cotengra TN, and verified GPU routes remain contextual
or cross-algorithm baselines and are not silently converted into same-plan
speedups. The physical v4 engine uses output/K tiling, one persistent session
per selected rank, one bulk set launch per request, and unique DPU descriptors.
Its whole-graph intermediate tensors are currently held by the Python host
store and re-uploaded for downstream tasks. Development evidence remains in
ignored `runs/`; it is not promoted while M5.5 is still changing.

The clean-source physical development runs at `b550c46` completed the
canonical (`1008/1008` rows), scaling (`80/80` rows), and large-boundary (`96`
completed plus `24` explicit 30q resource-limit rows) profiles without a
failed row. The scaling suite demonstrated fully active same-plan scaling only
through 16 DPUs. Its 32/64/128-DPU provisioned rows used 16 active DPUs, and
the two-rank row used one active rank, so those rows are retained as
overprovisioning evidence rather than rank or full-DPU scaling. The large suite
used all 64 allocated DPUs for supported 22--28q cases, but its single repeat
is admitted only as support-boundary and runtime observation.

The M4.6/M5.1/M5.2 rows above are historical development acceptance probes
copied from ETH, not promoted thesis evidence. The M5 execution-plan-v3 row is
an additive physically accepted development lane, not a promoted thesis result.
Its hardware-free check remains:

```bash
UPMEM_HW_M5_DPU_COUNTS=3 UPMEM_HW_M5_TASKLETS=3 make upmem-hw-m5-plan
```

This prepares a selected plan set, preserves unsupported cases, reports failures
explicitly, and performs no DPU allocation or launch. The audited physical run
used source commit `5401597fdc2458087e112f5bd2e1869a5a0a5ab0`, clean worktree,
and normalized-record hash
`1a7714b8dce25b0b0959ed08cae73aaf47e6d7084b90200d4895bf4c521202a0`. Its
suite hash is `e71ec4518a99a8c7f463926da845b1c67bef7242c72233c0c0cfdc107177e26c`.
The ETH run and fixed report are retained in ignored `runs/` at:

```text
runs/evidence/upmem_hardware_distributed_m5/upmem_hw_m5/2026-08-13_15-01-11
runs/comparisons/upmem_m5/2026-08-13_16-29-36_450221
```

All physical/provider/rank/allocation/kernel/release/no-fallback/transfer/
validation checks pass; all nine plots and table/plot hashes are valid.

```bash
UPMEM_HW_RANK_PATH=/dev/dpu_rank1 \
UPMEM_ALLOW_PHYSICAL_HARDWARE=1 make upmem-hw-m5
```

Physical commands require
`UPMEM_HW_RANK_PATH=/dev/dpu_rankN` and
`UPMEM_ALLOW_PHYSICAL_HARDWARE=1`; requested/effective rank selection is
recorded, but is not an independent observed-rank measurement.
The broad `thesis_results/current` snapshot is historical and does not contain
the M5 v3 development run. Do not promote it during development.

## Stop/Boundary Rules

A case is retained as unsupported rather than hidden when it reaches a rank,
element, memory, timeout, planner, validation, or device boundary. A manual
stress run may stop before larger sizes after such a boundary is established.
The canonical local grid remains fixed so regressions and improvements are
visible across waves.
