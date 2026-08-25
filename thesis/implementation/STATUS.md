# Implementation Status

This status applies to the active post-reset code only. It is not a summary of
historical experiments or a performance claim.

## Software Qualification State

The authoritative M6 exact-head record is the annotated tag
`thesis-m6-software-ready-v1` and its published GitHub release bundle. M6
software qualification is complete and `software_merge_ready` is established.
Physical UPMEM qualification remains pending.

M7A activates a bounded WRAM-panel dense real-tile kernel in source. Its final
exact-head software qualification and every physical measurement remain
pending; no M7A performance or scaling result exists.

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
| ABI-v4 UPMEM runtime | Yes | Controlled simulator tests; M6 software qualification complete | Reset route pending | Controlled-test correctness only |
| WRAM-panel dense real-tile kernel | Yes, `KC=64`, `NC=32` | Source, ABI, CPU replay, and SDK-simulator tests | No | Bounded kernel correctness and deterministic movement facts only |
| Split-complex float32 | Yes | Controlled CPU replay/simulator tests; M6 software qualification complete | Reset route pending | Controlled-test correctness only |
| Split-complex packed int8 | Yes | Controlled CPU replay/simulator tests; M6 software qualification complete | Reset route pending | Controlled-test correctness and numeric facts |
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

The post-reset physical route is **not ETH-qualified**. Before it can support
even a one-DPU physical execution claim, run:

```bash
make build-upmem-runtime UPMEM_TASKLETS=1
UPMEM_ALLOW_PHYSICAL_HARDWARE=1 \
  make qualify PHYSICAL_CONFIG=configs/tn_benchmark_physical.yml \
  OUTPUT=runs/physical-qualification
make verify INPUT=runs/physical-qualification
make report INPUT=runs/physical-qualification \
  REPORT_OUTPUT=runs/physical-qualification-report
```

The physical configuration must contain the target's real rank path and paths
to matching ABI-v4 binaries. Qualification evidence must record allocation,
launch, release, observed physical backend facts, output validation, and the
source/binary/environment identities.

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

After M7A exact-head software qualification, the next allowed hardware work is
a bounded physical one-DPU qualification using the active WRAM-panel kernel.
Tasklet/DPU scaling, slice scheduling, and residency remain later experiments
and require measured evidence before they become claims.
