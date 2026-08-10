# Audited & Approved Implementation Plan: Milestone M4.6a

> **Status:** AGREED - PLAN APPROVED BY AUDITOR & PLANNER AGENTS  
> **Target:** Milestone M4.6a (Hardware & Host Timing Metadata Instrumentation)  
> **Source Document:** [docs/m4_6a_implementation_plan.md](file:///home/tom/repos/Masters/thesis/implementation/docs/m4_6a_implementation_plan.md)

---

## 1. Executive Summary & Design Invariants

Micro-Step 4.6a adds high-precision host transfer timing and DPU cycle counter metadata (`perfcounter_get()`) to the physically accepted M4.5 single-tasklet runtime. 

### Key Design Invariants:
1. **KISS Compliance**: Reuses existing `host.c` / `session_protocol.c` JSON response streams. No custom binary sidecar files or Python `struct.unpack` calls are introduced.
2. **ABI Struct Alignment**: `resident_completion_t` is extended to 40 bytes (6x 32-bit `uint32_t` = 24 bytes, 2x 64-bit `uint64_t` = 16 bytes), guaranteeing identical 8-byte natural alignment without compiler padding across 32-bit DPU (`clang-dpu`) and 64-bit Host (`gcc/clang`).
3. **Simulator Safety**: `dpu_cycles` returns `0` on the software simulator. Python host calculations use explicit division-by-zero guards: `dpu_compute_seconds = (dpu_cycles / dpu_freq_hz) if (dpu_cycles > 0 and dpu_freq_hz > 0) else None`.
4. **Test Suite Integrity**: All 635 pytest tests are preserved; mock `native_response` payload dictionaries are updated to include `"dpu_run_time_cycles": 0`.

---

## 2. Component Diffs & File Changes

### Component 1: Native C ABI Header & DPU Kernel Instrumentation

#### [MODIFY] `native/upmem/simplepim/upmem_sdk_generic_loop_resident/common.h`
```c
/* Small host-visible ABI record. The host reads this only after dpu_sync(). */
/* ABI Alignment Map:
 * Offset 00: magic (uint32_t, 4 bytes)
 * Offset 04: version (uint32_t, 4 bytes)
 * Offset 08: active_operation_index (uint32_t, 4 bytes)
 * Offset 12: completion_status (uint32_t, 4 bytes)
 * Offset 16: completed_operation_count (uint32_t, 4 bytes)
 * Offset 20: output_elements_processed (uint32_t, 4 bytes)
 * Offset 24: output_checksum_fnv1a64 (uint64_t, 8 bytes)
 * Offset 32: dpu_run_time_cycles (uint64_t, 8 bytes)
 * Total Size: 40 bytes (perfectly aligned without padding on 32-bit & 64-bit)
 */
typedef struct {
    uint32_t magic;
    uint32_t version;
    uint32_t active_operation_index;
    uint32_t completion_status;
    uint32_t completed_operation_count;
    uint32_t output_elements_processed;
    uint64_t output_checksum_fnv1a64;
    uint64_t dpu_run_time_cycles;
} resident_completion_t;
```

#### [MODIFY] `native/upmem/simplepim/upmem_sdk_generic_loop_resident/dpu.c`
```c
#include <perfcounter.h>

int main(void) {
    if (NR_TASKLETS != 1) return 2;
    // ...
    RESIDENT_COMPLETION.dpu_run_time_cycles = 0;
    // ...
    perfcounter_config(COUNT_CYCLES, true);
    uint64_t start_cycles = perfcounter_get();
    
    if (operation.kind == RESIDENT_OPERATION_CONTRACT) {
        resident_contract(&operation, &checksum);
    } else if (operation.kind == RESIDENT_OPERATION_COMPLEX_COMBINE) {
        resident_complex_combine(&operation, &checksum);
    } else {
        return 4;
    }
    
    RESIDENT_COMPLETION.dpu_run_time_cycles = perfcounter_get() - start_cycles;
    RESIDENT_COMPLETION.active_operation_index = (uint32_t)RESIDENT_ACTIVE_OPERATION;
    RESIDENT_COMPLETION.completion_status = RESIDENT_COMPLETION_COMPLETED;
    RESIDENT_COMPLETION.completed_operation_count = (uint32_t)RESIDENT_ACTIVE_OPERATION + 1u;
    RESIDENT_COMPLETION.output_elements_processed = operation.output_elements;
    RESIDENT_COMPLETION.output_checksum_fnv1a64 = checksum;
    return 0;
}
```

---

### Component 2: Host C Wrapper & Session Protocol JSON Emitter

#### [MODIFY] `native/upmem/simplepim/upmem_sdk_generic_loop_resident/host.c`
- After `dpu_launch(set, DPU_SYNCHRONOUS);`, copy completion struct from DPU MRAM/WRAM:
```c
resident_completion_t completion;
dpu_copy_from(set, "RESIDENT_COMPLETION", 0, &completion, sizeof(completion));
```
- Pass `completion.dpu_run_time_cycles` to `resident_response_write()`.

#### [MODIFY] `native/upmem/simplepim/upmem_sdk_generic_loop_resident/session_protocol.c` & `session_protocol.h`
- Include `"dpu_run_time_cycles": <uint64>` in the generated JSON response file payload.

---

### Component 3: Python Host Session & TaskGraph Metrics

#### [MODIFY] `src/quantum_bench/targets/upmem/hardware_session.py`
- Instrument high-resolution `time.perf_counter()` probes for host operations:
  - `host_to_dpu_transfer_seconds`
  - `host_dpu_launch_seconds`
  - `dpu_to_host_transfer_seconds`
- Parse `dpu_run_time_cycles` from native response JSON.
- Division-by-zero guard:
  ```python
  dpu_freq_hz = 350_000_000
  dpu_compute_seconds = (dpu_cycles / dpu_freq_hz) if (dpu_cycles > 0 and dpu_freq_hz > 0) else None
  ```

#### [MODIFY] `src/quantum_bench/targets/upmem/taskgraph_runtime.py`
- Propagate `dpu_run_time_cycles`, `dpu_compute_seconds`, `host_to_dpu_transfer_seconds`, `host_dpu_launch_seconds`, and `dpu_to_host_transfer_seconds` into normalized task metrics payload.

---

### Component 4: Test Suite Verification

#### [MODIFY] `tests/test_upmem_simplepim_taskgraph_executor.py` & `tests/test_upmem_hardware_session.py`
- Add `"dpu_run_time_cycles": 0` to mock `native_response` dictionaries across unit tests.

---

## 3. Verification Plan

```bash
# 1. Run unit & integration test suite
make test

# 2. Run specific executor tests
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src ../.venv/bin/python -m pytest tests/test_upmem_simplepim_taskgraph_executor.py -v
```

All 635 tests must pass without warning or schema errors.
