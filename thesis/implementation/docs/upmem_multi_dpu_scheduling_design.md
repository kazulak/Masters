# UPMEM Multi-DPU Scheduling Design

This document defines the evidence contract and implementation boundary for
UPMEM multi-DPU scheduling. A modeled assignment report exists, but it is not an
execution claim. The current UPMEM execution route remains strict sequential SDK
simulator execution.

## Current Baseline

Implemented UPMEM evidence today:

- `upmem_tn_sdk_simulator_quantized` executes tensor-network TaskGraph
  contractions through the UPMEM SDK simulator path.
- The runtime walks TaskGraph tasks sequentially.
- Supported contractions require `cpu_fallback_used=false`.
- SDK simulator rows are code-path and boundary evidence, not hardware timing
  or hardware speedup evidence.

Current strict-path execution invariants:

- `task_count > 0`
- `upmem_task_count == task_count`
- `cpu_fallback_task_count == 0`
- `dpu_program_invocations > 0`
- `upmem_program_executed=true`
- `native_sdk_control_path=true`
- `simplepim_api_used=false`

## Design Goal

Add a future scheduling layer that can map ready TaskGraph work to DPU groups
without changing the top-level benchmark route identity.

The design must support three distinct evidence levels:

| Evidence level | Meaning | Allowed claim |
|---|---|---|
| `modeled_only` | Scheduler computes an assignment plan but does not execute it. | "This graph exposes this modeled DPU assignment/frontier opportunity." |
| `sdk_simulator_executed` | SDK simulator executes assigned work through DPU programs. | "The SDK simulator executed the assigned DPU code path." |
| `hardware_executed` | Real UPMEM hardware executes assigned work. | "This hardware run executed the assigned DPU schedule." |

SDK simulator timing must remain simulator timing. Hardware speedup claims are
allowed only for `hardware_executed` rows with hardware timing metadata.

## Scheduling Model

The scheduler operates on the existing TaskGraph:

1. Compute dependency-safe frontier waves from `ContractionTask.dependencies`.
2. For each wave, choose a DPU group assignment for ready tasks.
3. Execute or model each assigned task exactly once.
4. Synchronize after a wave when downstream dependencies require outputs.
5. Preserve final validation against the CPU exact or quantized reference
   appropriate for the route.

The first implementation should be modeled-only. It should not launch multiple
DPUs, change kernels, introduce PID-Comm, or alter strict runtime execution.

## Assignment Strategies

Initial strategy names should be explicit:

- `sequential_single_dpu`: current baseline, one task at a time.
- `frontier_round_robin_dpu_groups`: assign ready tasks across DPU groups by
  frontier order.
- `frontier_size_aware_dpu_groups`: assign ready tasks using estimated bytes or
  FLOPs to reduce imbalance.
- `single_task_multi_dpu_model`: future intra-contraction partitioning model;
  not execution until reduction/synchronization exists.

Only `sequential_single_dpu` is executed today. All other strategies must be
reported as modeled until execution exists.

## Metadata Contract

Future normalized rows should add fields incrementally and keep missing values
explicit.

Required modeled/executed scheduling fields:

- `upmem_parallelism_mode`: `sequential | frontier_multi_dpu | intra_task_multi_dpu | hybrid_multi_dpu`
- `upmem_parallelism_evidence_type`: `modeled | sdk_simulator_executed | hardware_executed | unsupported`
- `task_assignment_strategy`
- `dpu_group_count`
- `dpu_group_id` where row granularity is per assignment
- `frontier_wave_count`
- `max_frontier_width`
- `mean_frontier_width`
- `assigned_task_count`
- `executed_dpu_task_count`
- `unassigned_task_count`
- `dpu_assignment_plan_artifact`
- `dpu_assignment_validation_status`

Required timing/cost fields when measured or modeled:

- `host_transfer_time_s`
- `dpu_program_time_s`
- `dpu_sync_time_s`
- `dpu_reduction_time_s`
- `host_orchestration_time_s`
- `quantization_time_s`
- `dequantization_time_s`
- `assigned_h2d_bytes`
- `assigned_d2h_bytes`
- `modeled_dpu_occupancy`
- `modeled_load_imbalance_ratio`

Required evidence-boundary fields:

- `contraction_execution_target=upmem`
- `upmem_execution_mode=sdk_simulator | hardware`
- `execution_backend=upmem_sdk`
- `hardware_execution`
- `hardware_timing_available`
- `hardware_speedup_applicable`
- `cpu_fallback_used=false`
- `cpu_fallback_task_count=0`
- `native_sdk_control_path=true`
- `simplepim_api_used=false`

## Assignment Plan Artifact

A modeled or executed schedule should write one JSON artifact:

```text
upmem_multi_dpu_assignment_plan.json
```

Minimum content:

- schema version
- case ID and route ID
- scheduler strategy
- evidence type
- DPU group count
- frontier waves
- assigned task IDs per wave/group
- estimated H2D/D2H bytes per assignment
- estimated FLOPs per assignment
- dependency validation status
- duplicate assignment check
- unassigned/unsupported task reasons

The artifact is evidence metadata. It must not be interpreted as executed
parallelism unless paired with execution rows proving DPU program execution.

## Validation Invariants

Any modeled or executed multi-DPU scheduler must prove:

- no task is assigned before all dependencies are available;
- every supported task is assigned exactly once;
- unassigned tasks have explicit unsupported reasons;
- duplicate assignment check passes;
- missing dependency check passes;
- final output validation passes for executed rows;
- CPU contraction fallback remains false;
- simulator rows do not claim hardware timing or hardware speedup.

## Implemented Modeled Assignment Wave

The first safe implementation is:

**UPMEM modeled multi-DPU assignment report**

Behavior:

- read the same TaskGraph used by strict UPMEM runtime;
- compute frontier waves;
- assign ready tasks to a configurable modeled DPU group count;
- write `upmem_multi_dpu_assignment_plan.json`;
- emit normalized records with
  `upmem_parallelism_evidence_type=modeled`;
- run no DPU programs beyond the existing sequential baseline.

CLI sketch:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src ../.venv/bin/python -m quantum_bench.bench upmem-multi-dpu-assignment \
  --suite configs/suites/upmem_sim_evidence.yml \
  --dpu-groups 4 \
  --strategy frontier_round_robin_dpu_groups
```

Tests:

- assignment covers every supported task exactly once;
- dependency order is respected;
- modeled records are not marked executed;
- `hardware_speedup_applicable=false`;
- current strict sequential UPMEM route remains unchanged.

## Later Execution Waves

Only after modeled scheduling is validated:

1. Add SDK simulator multi-DPU execution if the SDK path can represent assigned
   DPU groups.
2. Add reduction/synchronization metadata.
3. Add hardware execution only when real UPMEM hardware is available.
4. Introduce PID-Comm only when communication or synchronization is the actual
   bottleneck being tested.

## Non-Goals

- no multi-DPU execution in this design wave;
- no new UPMEM kernel;
- no PID-Comm integration;
- no SimplePIM execution change;
- no top-level route renaming;
- no hardware speedup claim from SDK simulator rows;
- no CPU contraction fallback in strict UPMEM runtime.
