# Audited & Approved Implementation Plan: M4.6 Audit Fixes

> **Status:** AGREED - PLAN APPROVED BY AUDITOR & PLANNER AGENTS  
> **Target:** M4.6 Audit Fixes (Python Pipeline, Perfcounter Race, Barrier Safety, Scaling Test & Frequency Constant)  
> **Source Document:** [docs/m4_6_audit_fix_implementation_plan.md](file:///home/tom/repos/Masters/thesis/implementation/docs/m4_6_audit_fix_implementation_plan.md)

---

## 1. Executive Summary & Design Invariants

This plan details the exact fixes required to resolve all 5 issues identified in `m4_6_audit.md`:

1. **Python Data Pipeline**: Connect missing `dpu_run_time_cycles` extraction in `hardware_session.py`, pass it into `task_metrics` in `taskgraph_runtime.py`, and surface `tasklets_per_dpu` in `executor_config`.
2. **Perfcounter Race Condition**: Move `perfcounter_config` and `start_cycles_shared` sampling inside an `if (me() == 0)` block after `barrier_wait(&tasklet_barrier)` using a static variable `static perfcounter_t start_cycles_shared;` in `dpu.c`.
3. **Barrier Safety**: Eliminate early returns (`return 3;`, `return 4;`) before barrier points in `dpu.c` so all tasklets reach `barrier_wait` unconditionally.
4. **Scaling Benchmark Test**: Expand `test_multi_tasklet_benchmark_scaling` in `test_upmem_simplepim_taskgraph_executor.py` across $NR\_TASKLETS \in \{1, 2, 4, 8, 11, 16\}$ and explicitly assert metric dictionary keys (`"dpu_run_time_cycles"`, `"dpu_compute_seconds"`).
5. **DPU Clock Frequency Constant**: Define `UPMEM_DPU_CLOCK_HZ = 350_000_000` at module scope in `runtime_evidence.py`.

---

## 2. Component Diffs & File Changes

### Component 1: C DPU Kernel (`dpu.c`)

#### [MODIFY] `native/upmem/simplepim/upmem_sdk_generic_loop_resident/dpu.c`
- Declare `static perfcounter_t start_cycles_shared;` at file/function static scope.
- Wrap `perfcounter_config(COUNT_CYCLES, true);` and `start_cycles_shared = perfcounter_get();` inside `if (me() == 0)` after `barrier_wait(&tasklet_barrier);`.
- Replace early `return 3;` and `return 4;` with unified control flow so tasklets evaluate an `status_code` variable and hit `barrier_wait(&tasklet_barrier);` unconditionally before exiting.

---

### Component 2: Python Session & TaskGraph Runtime Pipeline

#### [MODIFY] `src/quantum_bench/targets/upmem/hardware_session.py`
- Add `dpu_run_time_cycles: int` field to `ResidentGraphSessionExecution`.
- Extract `dpu_run_time_cycles=response.get("dpu_run_time_cycles", 0)` in `_resident_execute_task`.

#### [MODIFY] `src/quantum_bench/targets/upmem/taskgraph_runtime.py`
- Add `tasklets_per_dpu: int = 1` parameter to `upmem_taskgraph_executor_config` and expose it in returned config dict.
- Pass `dpu_run_time_cycles=result.dpu_run_time_cycles` into `_base_task_metric` constructor inside `_execute_generic_real_component`.

#### [MODIFY] `src/quantum_bench/targets/upmem/runtime_evidence.py`
- Define `UPMEM_DPU_CLOCK_HZ = 350_000_000` at module scope.
- Add `dpu_run_time_cycles: int = 0` parameter to `_base_task_metric` and include `"dpu_run_time_cycles": int(dpu_run_time_cycles)` in task metric dict.
- Update `_summary_payload` to compute `dpu_compute_seconds = (total_dpu_run_time_cycles / UPMEM_DPU_CLOCK_HZ) if total_dpu_run_time_cycles > 0 else None`.

---

### Component 3: Test Verification Suite

#### [MODIFY] `tests/test_upmem_simplepim_taskgraph_executor.py`
- Expand `test_multi_tasklet_benchmark_scaling` across `tasklets` in `{1, 2, 4, 8, 11, 16}`.
- Assert that metric dictionary contains `"dpu_run_time_cycles"` and `"dpu_compute_seconds"` keys.
- Verify fallback guard behavior and profile replacement.

---

## 3. Verification Plan

```bash
# 1. Run the updated multi-tasklet benchmark scaling test
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src ../.venv/bin/python -m pytest tests/test_upmem_simplepim_taskgraph_executor.py -k test_multi_tasklet_benchmark_scaling -v

# 2. Run full repository pytest regression suite
make test
```

All 641+ tests must pass cleanly.
