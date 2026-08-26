# Implementation Status

This status applies to the active post-reset code only. It is not a summary of
historical experiments or a performance claim.

## Software Qualification State

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
| UPMEM physical mapping | Yes, bounded output/K tiles | Yes | Reset route pending | Plan construction only |
| ABI-v4 UPMEM runtime | Yes | M6/M7A controlled SDK-simulator qualification complete | Reset route pending | Controlled-test correctness only |
| WRAM-panel dense real-tile kernel | Yes, `KC=64`, `NC=32` | M7A source, ABI, CPU replay, and SDK-simulator qualification complete | No | Bounded kernel correctness and deterministic movement facts only |
| Split-complex float32 | Yes | M6/M7A controlled CPU replay and simulator qualification complete | Reset route pending | Controlled-test correctness only |
| Split-complex packed int8 | Yes | M6/M7A controlled CPU replay and simulator qualification complete | Reset route pending | Controlled-test correctness and numeric facts |
| Local contraction slicing | Yes | Yes | Reset route pending | Logical slicing correctness only |
| Host reduction | Yes | Yes | Reset route pending | Bounded host-round-trip correctness |
| DPU-resident intermediates | No | No | No | No claim |
| Tasklet-aware row ownership | Yes, fixed kernel mapping only | One and eight tasklets in controlled tests | No | Requested/observed tasklet correctness only; no scaling claim |
| Tasklet scheduling/scaling | No dynamic policy or measurement | No | No | No scaling claim |
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
Simulator timing is never physical performance evidence, and physical UPMEM
qualification remains pending with no physical speedup, scaling, or energy claim.

## Physical Qualification State

The post-reset physical route is **not ETH-qualified**. M7C preparation is
implemented on the active source but still requires its exact-head
software/SDK qualification before an ETH command. Before it can support even a
one-DPU physical execution claim, generate an ignored machine-specific copy and
run the physical-only qualification command:

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

Until that run succeeds, do not claim physical speedup, energy efficiency,
parallel scaling, multi-rank operation, graph-wide residency, or general
UPMEM TN acceleration.

## Retained External Components

- **SimplePIM:** pinned management types and its initialization kernel are used
  around raw-SDK allocation/dispatch. No high-level scheduling route is yet
  qualified.
- **PID-Comm:** the retained `native/upmem/pidcomm_qualification/` source is a
  standalone future compatibility harness. It is not an active communication
  provider or public command.
- **ATiM:** not integrated.

M7A exact-head software qualification is complete. M7B exact-head
pre-physical qualification is complete at
`thesis-m7b-prephysical-software-ready-v1`; its release records CPU,
SDK-simulator, direct native-boundary, provenance, and evidence checks.
Physical UPMEM qualification remains pending. The next stage is a bounded
one-DPU physical smoke followed by separately preregistered scaling work.
The tracked M7C source selection chooses stress18 for controlled primary
scaling and GHZ18 for structural confirmation; it uses no timing evidence.
Tasklet/DPU scaling, slice scheduling, and residency require measured evidence
before they become claims.
