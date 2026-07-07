# UPMEM Multi-DPU Prototype Readiness Plan

This document is the readiness gate for the first executed UPMEM multi-DPU
prototype. It is not an implementation claim. The current implemented UPMEM
parallelism artifact is still the modeled assignment report from
`upmem-multi-dpu-assignment`.

## Current Evidence Boundary

Implemented today:

- `upmem_tn_sdk_simulator_quantized` executes TaskGraph contractions through
  strict sequential UPMEM SDK simulator programs.
- `upmem-multi-dpu-assignment` computes dependency-safe frontier waves and
  assigns ready tasks to modeled DPU groups.
- The assignment report emits `upmem_parallelism_evidence_type=modeled`,
  `execution_plan_executed=false`, `dpu_program_invocations=0`, and
  `hardware_speedup_applicable=false`.
- `upmem-taskgraph-frontier-runtime` executes the same strict per-task UPMEM
  SDK simulator path by dependency-safe frontier waves with
  `frontier_worker_count=1`. It emits
  `upmem_parallelism_evidence_type=sdk_simulator_executed`, not hardware
  evidence.

Not implemented today:

- no runtime consumes `upmem_multi_dpu_assignment_plan.json` to execute tasks;
- no UPMEM task runner uses `dpu_group_id` or `dpu_group_count` to control real
  DPU allocation;
- no native generic host distributes one contraction across multiple DPUs;
- no PID-Comm or inter-DPU communication path is integrated;
- no UPMEM hardware execution or hardware timing exists.

## Source Evidence

Current sequential runtime evidence:

- `src/quantum_bench/targets/upmem/taskgraph_runtime.py`
  `execute_upmem_taskgraph_runtime(...)` iterates over `graph.tasks` in a
  single task order and invokes `_execute_task_by_policy(...)` for each task.
- `src/quantum_bench/targets/upmem/generic_bridge.py`
  `_execute_upmem_sdk_generic_loop(...)` launches one external generic runner
  per task bridge input.
- `native/upmem/simplepim/upmem_sdk_generic_loop/host.c` calls
  `dpu_alloc(1, NULL, &set)`, so the current generic host uses one DPU for a
  task invocation.
- `native/upmem/simplepim/upmem_sdk_generic_loop_runner.py` rejects
  `--target hardware` and runs the SDK simulator path only.
- `src/quantum_bench/bench/upmem_multi_dpu_assignment.py` writes assignment
  plans and normalized modeled records, but intentionally reports zero DPU
  invocations.

## Smallest Safe Prototype

The implementation should not jump directly to hardware multi-DPU claims. The
smallest safe prototype is:

**UPMEM SDK simulator frontier-scheduled execution over assignment plans**

Status: implemented for `frontier_worker_count=1` through
`upmem-taskgraph-frontier-runtime`. This is an executed SDK simulator scheduler
prototype, not hardware multi-DPU execution.

Purpose:

- execute independent ready tasks wave-by-wave using the existing strict UPMEM
  SDK simulator task bridge;
- consume or reproduce the modeled assignment plan;
- preserve strict no-CPU-contraction-fallback semantics;
- prove task dependency, duplicate, and output validation invariants;
- report simulator execution separately from hardware execution.

Allowed claim if successful:

> The UPMEM SDK simulator executed a frontier-scheduled TaskGraph according to a
> DPU assignment plan.

Not allowed:

- hardware speedup;
- hardware timing;
- true hardware multi-DPU execution;
- intra-contraction multi-DPU distribution;
- PID-Comm communication evidence.

## Prototype Execution Semantics

The prototype should execute by frontier waves:

1. Build the same TaskGraph and tensor map as strict sequential runtime.
2. Compute frontier waves from task dependencies.
3. Assign ready tasks using the modeled assignment strategy.
4. For each wave, execute only tasks whose dependencies are available.
5. Permit same-wave task execution concurrently only when their inputs and
   outputs are independent and the bridge work directories are isolated.
6. Merge task outputs into the live tensor map after the wave completes.
7. Release dead inputs after all wave outputs are committed.
8. Validate the final output against the same reference contract as the
   sequential UPMEM route.

Worker count is explicit:

- `frontier_worker_count=1` proves sequential-frontier semantics and is
  implemented.
- `frontier_worker_count>1` remains future work. If implemented, it may execute
  multiple independent task invocations concurrently on the host, but must still
  be labeled SDK simulator execution, not hardware speedup.

## Required Normalized Metadata

Rows for a successful SDK simulator frontier prototype must include:

- `parallelism_mode=frontier`
- `parallelism_evidence_type=executed`
- `upmem_parallelism_mode=frontier_multi_dpu`
- `upmem_parallelism_evidence_type=sdk_simulator_executed`
- `execution_plan_kind=upmem_frontier_assignment_scheduler`
- `execution_plan_executed=true`
- `task_assignment_strategy`
- `dpu_group_count`
- `frontier_worker_count`
- `frontier_wave_count`
- `max_frontier_width`
- `mean_frontier_width`
- `assigned_task_count`
- `executed_dpu_task_count`
- `dpu_program_invocations`
- `dpu_assignment_plan_artifact`
- `dpu_assignment_validation_status=passed`
- `duplicate_contraction_check=passed`
- `missing_dependency_check=passed`
- `dependency_violation_detected=false`
- `cpu_fallback_used=false`
- `cpu_fallback_task_count=0`
- `hardware_execution=false`
- `hardware_timing_available=false`
- `hardware_speedup_applicable=false`

Rows must not use `hardware_executed` unless the run used real UPMEM hardware.

## Hard Gates Before Implementation

Do not start the executed prototype unless these gates are satisfied:

1. The assignment plan covers every supported TaskGraph task exactly once.
2. The scheduler can prove no task executes before dependencies are available.
3. Bridge input/output directories are unique per task invocation.
4. Runtime tensor production never uses CPU contraction fallback.
5. Failure of any task in a wave aborts the run with an explicit unsupported or
   failed reason.
6. The final output validates against the chosen UPMEM reference contract.
7. Reported timing scope says SDK simulator, not hardware timing.
8. `frontier_worker_count>1` is tested separately from `frontier_worker_count=1`.

## First Test Suite

Use tiny deterministic circuits only:

- QRNG 3q or 4q
- BV 4q if supported by the generic kernel caps
- XOR 4q if supported by the generic kernel caps

The first suite should contain only:

- sequential strict UPMEM SDK simulator reference row;
- frontier-scheduled UPMEM SDK simulator prototype row;
- CPU or QuEST validation anchor if needed for final validation.

## Validation Commands For A Future Prototype

Expected validation shape:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src ../.venv/bin/python -m pytest -q

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src ../.venv/bin/python -m quantum_bench.bench upmem-multi-dpu-assignment \
  --suite configs/suites/upmem_sim_evidence.yml \
  --dpu-groups 2

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src ../.venv/bin/python -m quantum_bench.bench upmem-taskgraph-frontier-runtime \
  --case bell_2q \
  --policy generic-only \
  --quantization-mode per_task_input_quantize \
  --dpu-groups 2 \
  --frontier-worker-count 1 \
  --execute-external
```

## Later True Multi-DPU Work

A true multi-DPU or hardware prototype is a separate step. It requires one of:

- native host support for `dpu_alloc(N, ...)` and per-DPU input/output
  partitioning;
- intra-contraction partitioning and reduction semantics;
- a communication/synchronization path if partial outputs must be exchanged or
  reduced;
- hardware availability and timing metadata.

Until that exists, simulator frontier scheduling is only SDK simulator evidence.

## Next Milestone Recommendation

Do not jump directly to hardware. The next implementation should either:

1. add a tiny suite-level harness around `upmem-taskgraph-frontier-runtime`, or
2. enable `frontier_worker_count>1` for isolated host-side SDK simulator task
   invocations, with unique bridge directories and strict dependency checks.

Only after that passes validation should true hardware multi-DPU allocation be
considered.
