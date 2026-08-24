# Implementation Status

This status applies to the active post-reset code only. It is not a summary of
historical experiments or a performance claim.

## Software Qualification Checkpoint

Source under test: `8122c75145d00526c4ad9ad2e03c7ce49d628d0e` on Python
3.10.12 with a clean worktree. Constraint-file SHA-256:
`269ae0c52099a299743ca765e5251f24edfc1347f4dbcd9927a7a7a4530b981e`.

- `make test`: 449 passed in 23.17 s.
- `ruff check src tests`: clean.
- wheel and source distribution: built with `python -m build --no-isolation`.
- QuEST CPU runner and ABI-v4 host/DPU binaries: clean builds passed.
- default CPU/TN workflow: 12/12 samples passed verification.
- SDK-simulator workflow: 6/6 samples and 2/2 sessions passed verification.
- physical configuration preparation: 4 route entries planned without device
  allocation.

This checkpoint is software and simulator qualification. It is not physical
UPMEM evidence.

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
| ABI-v4 UPMEM runtime | Yes | SDK simulator and controlled sessions | Reset route pending | Simulator correctness only |
| Split-complex float32 | Yes | CPU replay and simulator | Reset route pending | Software/simulator correctness |
| Split-complex packed int8 | Yes | CPU replay and simulator | Reset route pending | Software/simulator correctness and numeric facts |
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

The active implementation can be described as a software-validated,
simulator-checked pipeline from `SimulationJob` through a target-neutral
`TensorNetwork`, selected path, `ContractionDAG`, direct CPU/TN execution or a
bounded `UpmemPlan`, canonical evidence, and reports.

The local SDK simulator has exercised the active one-rank, one-DPU ABI-v4
route for split-complex float32 and shared-scale packed int8 against the CPU
physical-plan replay. Simulator timing is not physical performance evidence.

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
- **PID-Comm:** `make pidcomm-check` runs a standalone compatibility harness.
  It is not integrated as a communication provider.
- **ATiM:** not integrated.

The next measured upgrade should follow physical one-DPU qualification: first
tasklet/DPU scaling, then slice scheduling or residency only when transfer and
runtime evidence identifies a clear bottleneck.
