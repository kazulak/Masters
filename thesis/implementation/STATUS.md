# Implementation Status

This status applies to the active post-reset code only. It is not a summary of
historical experiments or a performance claim.

## Software Qualification State

The authoritative M6 exact-head record is the annotated tag
`thesis-m6-software-ready-v1` and its archived release bundle. Until that tag
and release bundle exist, `software_merge_ready` is not established. The
pending M6.4 qualification run must not be pre-claimed here.

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
| ABI-v4 UPMEM runtime | Yes | Controlled simulator tests; M6.4 pending | Reset route pending | Controlled-test correctness only |
| Split-complex float32 | Yes | Controlled CPU replay/simulator tests; M6.4 pending | Reset route pending | Controlled-test correctness only |
| Split-complex packed int8 | Yes | Controlled CPU replay/simulator tests; M6.4 pending | Reset route pending | Controlled-test correctness and numeric facts |
| Local contraction slicing | Yes | Yes | Reset route pending | Logical slicing correctness only |
| Host reduction | Yes | Yes | Reset route pending | Bounded host-round-trip correctness |
| DPU-resident intermediates | No | No | No | No claim |
| Tasklet scheduling/scaling | No | No | No | No claim |
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
execution or a bounded `UpmemPlan`, canonical evidence, and reports. These
coverage results do not establish exact-head M6.4 qualification.

The active M6.4 simulator configuration is deterministic sliced complex
4-qubit `quantization_stress`: four partial branches, one host reduction, and
split-complex float32 plus shared-scale packed int8 routes on one rank, one DPU
and one tasklet. When executed and verified during M6.4, it produces
simulator-only correctness evidence. It has not yet established exact-head
qualification, and its timing is never physical performance evidence.

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

The next measured upgrade should follow physical one-DPU qualification: first
tasklet/DPU scaling, then slice scheduling or residency only when transfer and
runtime evidence identifies a clear bottleneck.
