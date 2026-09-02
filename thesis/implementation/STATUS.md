# Implementation Status

This status applies to the active post-reset code only. It is not a summary of
historical experiments or a performance claim.

## Software Qualification State

Sequential UPMEM baseline v1 now has software conformance plus exact-head
`prepare`, `inspect`, and `bundle` operator tooling. Its physical performance
campaign remains separate; the hierarchical result below is diagnostic-only.
See [docs/sequential_upmem_baseline.md](docs/sequential_upmem_baseline.md) and run
`make sequential-conformance` or `make sequential-baseline` for the safe entries.

Hierarchical Parallel Diagnostic v1 is physically validated at
`thesis-upmem-hierarchical-parallel-diagnostic-v1` on `safari-baguette1`, rank 1,
with 36/36 successful samples and 36/36 successful sessions. See
[docs/hierarchical_parallel_diagnostic.md](docs/hierarchical_parallel_diagnostic.md).

The authoritative M6 exact-head record is the annotated tag
`thesis-m6-software-ready-v1` and its published GitHub release bundle. M6
software qualification is complete and `software_merge_ready` is established.
Physical UPMEM qualification remains pending.

M7A activates the bounded WRAM-panel dense real-tile kernel
`dpu_real_tile_v4_wram_panel_v1`. Its exact-head software qualification is
complete at `thesis-m7a-wram-kernel-software-ready-v1`; the published release
bundle records SDK-simulator execution only. Every physical measurement remains
pending, and no M7A performance or scaling result exists.

| Capability | Implemented | Software/simulator validation | Physical evidence | Claimable now |
|---|---|---|---|---|
| Circuit to TN lowering | Yes | Yes | Not applicable | Correct lowering within supported circuit/query scope |
| Target-neutral semantic network | Yes | Yes | Not applicable | `TensorNetwork` is non-executable semantic data |
| Logical path/DAG lowering | Yes | Yes | Not applicable | One `ContractionDAG` is the sole logical execution IR |
| Direct NumPy same-DAG replay | Yes | Yes | Not applicable | CPU logical-plan correctness reference |
| Quimb/cotengra adapters | Yes | Yes | Not applicable | CPU TN baseline routes for declared scopes |
| QuEST CPU adapter | Yes | Controlled software tests | No current reset run | Adapter availability, not CPU performance |
| QuEST GPU adapter | Yes | Controlled software tests | No compatible GPU run | Capability detection and explicit unsupported result |
| UPMEM physical mapping | Yes, bounded output/K tiles | Yes | M7C rank-1 diagnostic | Diagnostic topology only |
| ABI-v4 UPMEM runtime | Yes | M6/M7A controlled SDK-simulator qualification complete | M7C physical diagnostic | Diagnostic correctness and timing only |
| WRAM-panel dense real-tile kernel | Yes, `KC=64`, `NC=32` | M7A source, ABI, CPU replay, and SDK-simulator qualification complete | No | Bounded kernel correctness and deterministic movement facts only |
| Split-complex float32 | Yes | M6/M7A controlled CPU replay and simulator qualification complete | M7C physical diagnostic | Diagnostic correctness only |
| Split-complex packed int8 | Yes | M6/M7A controlled CPU replay and simulator qualification complete | Reset route pending | Controlled-test correctness and numeric facts |
| Local contraction slicing | Yes | Yes | No M7C slice route | Logical slicing correctness only |
| Host reduction | Yes | Yes | M7C physical diagnostic | Bounded host-round-trip correctness |
| DPU-resident intermediates | No | No | No | No claim |
| Tasklet-aware row ownership | Yes, fixed kernel mapping only | One and eight tasklets in controlled tests | M7C T1/T2/T4/T8 | Diagnostic scaling only |
| Tasklet scheduling/scaling | Fixed route matrix only | Controlled simulator and physical diagnostic | M7C T1/T2/T4/T8 | Powersave diagnostic only |
| Slice-group parallel execution | No | No | No | No claim |
| Multi-rank execution/scaling | No | No | No | No claim |
| PID-Comm provider | No | Standalone harness only | No | No communication claim |
| ATiM kernel provider | No | No | No | No claim |
| Energy measurement | No | Evidence schema supports null field | No | No energy claim |
| Hardware-calibrated planner | No | No | No | No planner-performance claim |
| Matched NumPy/UPMEM end-to-end scope | No | Scope mismatch is rejected | No | No end-to-end speedup claim |

## What Is Valid Today

Controlled tests cover the active pipeline from `SimulationJob` through a
target-neutral `TensorNetwork`, selected path, `ContractionDAG`, direct CPU/TN
execution or a bounded `UpmemPlan`, canonical evidence, and reports.

M6 software qualification is complete at tag `thesis-m6-software-ready-v1` and
its release bundle. The qualification run uses the deterministic sliced complex
4-qubit `quantization_stress` suite (four partial branches, one host reduction,
and split-complex float32 plus shared-scale packed int8 routes on one rank, one
DPU and one tasklet). This produces simulator-only correctness evidence.
Simulator timing is never physical performance evidence. Sequential physical
performance qualification remains separate from the M7C diagnostic, with no
final physical speedup or energy claim.

M7C physically validated the fixed one-rank hierarchical route matrix on
Stress18. It demonstrates descriptive tasklet and DPU kernel scaling and more
limited total-wall scaling under `powersave`; it does not establish a final
`physical_performance_v1` campaign or machine-independent performance claim.

## Physical Qualification State

The post-reset sequential performance route is not `physical_performance_v1`
qualified. M7C physical diagnostic evidence is qualified only for its exact
Stress18 route matrix and rank-1/powersave environment. Before a later
performance campaign, generate an ignored machine-specific copy and run the
physical-only qualification command:

```bash
PYTHONPATH=src ../.venv/bin/python scripts/qualify_m7c_physical.py prepare \
  --template configs/tn_benchmark_physical_smoke.yml \
  --output runs/configs/eth/one-dpu-float32.yml \
  --mode float32-smoke --rank-path /dev/dpu_rank0 \
  --session-root runs/upmem_sessions/eth-one-dpu --expected-cpus 0
UPMEM_ALLOW_PHYSICAL_HARDWARE=1 make qualify \
  PHYSICAL_CONFIG=runs/configs/eth/one-dpu-float32.yml \
  OUTPUT=runs/evidence/eth-one-dpu-float32
PYTHONPATH=src ../.venv/bin/python scripts/qualify_m7c_physical.py inspect \
  --input runs/evidence/eth-one-dpu-float32 \
  --expected-samples 6 --expected-sessions 6 \
  --numeric-policy split_complex_float32_v1
```

The tracked smoke configuration is a template. Generate an ignored ETH copy
through the M7C physical preparation script, which resolves target-specific
rank, session, and executable paths before writing it. Qualification evidence
must record allocation, launch, release, observed physical backend facts,
output validation, and source/binary/environment identities.

Until a separate performance campaign succeeds, do not claim final physical
speedup, energy efficiency, multi-rank operation, graph-wide residency, or
general UPMEM TN acceleration. M7C descriptive route ratios remain limited to
the tagged diagnostic environment.

## Retained External Components

- **SimplePIM:** retained as pinned external research source only. The active
  runtime uses direct SDK allocation/dispatch and does not launch the former
  allocator-reset initialization kernel. No high-level scheduling route is
  qualified.
- **PID-Comm:** the retained `native/upmem/pidcomm_qualification/` source is a
  standalone future compatibility harness. It is not an active communication
  provider or public command.
- **ATiM:** not integrated.

M7A exact-head software qualification is complete. M7B exact-head
pre-physical qualification is complete at
`thesis-m7b-prephysical-software-ready-v1`; its release records CPU,
SDK-simulator, direct native-boundary, provenance, and evidence checks.
Sequential physical performance qualification remains pending. The recovered
M7C diagnostic is the measured one-rank scaling evidence; later performance
work must use a separate preregistered campaign and contemporaneous controls.
The tracked M7C source selection chooses stress18 for controlled primary
scaling and GHZ18 for structural confirmation; it uses no timing evidence.
Tasklet/DPU scaling, slice scheduling, and residency require measured evidence
before they become claims.
