# Audited & Approved Implementation Plan: Milestone M4.6b

> **Status:** AGREED - PLAN APPROVED BY AUDITOR & PLANNER AGENTS  
> **Target:** Milestone M4.6b (Multi-Tasklet Work Division & Parallel Execution, $1 \le \text{NR\_TASKLETS} \le 16$)  
> **Source Document:** [docs/m4_6b_implementation_plan.md](file:///home/tom/repos/Masters/thesis/implementation/docs/m4_6b_implementation_plan.md)

---

## 1. Executive Summary & Design Invariants

Micro-Step 4.6b introduces multi-tasklet parallel contraction execution ($1 \le \text{NR\_TASKLETS} \le 16$) on UPMEM DPUs.

### Key Design Invariants:
1. **KISS Compliance**: Uses strided loop division (`tile_start = me() * TILE_ELEMS ... step += NR_TASKLETS * TILE_ELEMS`) without dynamic tasklet queues or over-engineering.
2. **Barrier Safety & Unconditional Execution**: `BARRIER_INIT(tasklet_barrier, NR_TASKLETS)` is initialized globally, and `barrier_wait(&tasklet_barrier)` runs **unconditionally** after the strided loop by all tasklets, eliminating DPU deadlocks.
3. **Deterministic Checksum**: Do NOT accumulate FNV-1a checksums in parallel during strided execution. After `barrier_wait`, Tasklet 0 executes a single sequential pass over the completed output slot in MRAM, guaranteeing exact CPU-identical order.
4. **WRAM Memory Safety**: Static allocations (`resident_output_tile[16][258]`: 16.5 KB, `resident_input_window[16][8]`: 128 B, tasklet stacks: 16 KB) total ~34.1 KB, well below the 60 KB effective WRAM budget.

---

## 2. Component Diffs & File Changes

### Component 1: Native C Header & Makefile Adjustments

#### [MODIFY] `native/upmem/simplepim/upmem_sdk_generic_loop_resident/common.h`
```c
#if NR_TASKLETS < 1 || NR_TASKLETS > 16
#error "generic_loop_resident_graph_session_v1 requires 1 <= NR_TASKLETS <= 16"
#endif
```

#### [MODIFY] `native/upmem/simplepim/upmem_sdk_generic_loop_resident/Makefile`
- Relax static check to allow `1 <= NR_TASKLETS <= 16`.

---

### Component 2: DPU Multi-Tasklet Kernel Instrumentation

#### [MODIFY] `native/upmem/simplepim/upmem_sdk_generic_loop_resident/dpu.c`
- Include `<barrier.h>` and define `BARRIER_INIT(tasklet_barrier, NR_TASKLETS);`.
- Per-tasklet WRAM buffers:
```c
__dma_aligned float resident_output_tile[NR_TASKLETS][RESIDENT_OUTPUT_TILE_ELEMS + 2u];
__dma_aligned uint8_t resident_input_window[NR_TASKLETS][8];
```
- Strided tile loop per tasklet:
```c
for (uint32_t tile_start = me() * RESIDENT_OUTPUT_TILE_ELEMS;
     tile_start < operation->output_elements;
     tile_start += NR_TASKLETS * RESIDENT_OUTPUT_TILE_ELEMS) {
    // Contraction math using resident_output_tile[me()] ...
}

barrier_wait(&tasklet_barrier);

if (me() == 0) {
    // Tasklet 0 computes deterministic FNV-1a checksum over output slot in MRAM
    // Tasklet 0 writes RESIDENT_COMPLETION
}
```

---

### Component 3: Host C & Python Infrastructure

#### [MODIFY] `native/upmem/simplepim/upmem_sdk_generic_loop_resident/host.c`
- Relax `NR_TASKLETS` validation check to accept `1 <= NR_TASKLETS <= 16`.

#### [MODIFY] `src/quantum_bench/targets/upmem/hardware_session.py`
- Pass `NR_TASKLETS={profile.tasklets_per_dpu}` in Makefile build command.
- Validate `1 <= profile.tasklets_per_dpu <= 16`.

---

## 3. Verification Plan

```bash
# 1. Run full unit & integration test suite
make test

# 2. Test multi-tasklet execution with 1, 2, 4, 8, 12, 16 tasklets
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src ../.venv/bin/python -m pytest tests/test_upmem_simplepim_taskgraph_executor.py -v
```

All 635 tests must pass without deadlocks or checksum discrepancies.
