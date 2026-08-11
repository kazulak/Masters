# Milestone M5 Phase 1: Native C Driver & DPU Kernel updates for Multi-DPU Execution

This document outlines the production-grade, strict KISS implementation plan for M5 Phase 1. The goal is to extend the single-DPU resident loop execution to support multi-DPU sliced execution, strictly dividing work and correctly aligning structures.

## 1. C Struct Alignment & ABI (`common.h`)

**Design Updates**:
We need to inject multi-DPU slice parameters. These must be identical in layout between the 32-bit DPU compiler and the 64-bit host compiler.
The slice variables are added to the existing `upmem_generic_args_t` structure.

**Fields to add to `upmem_generic_args_t`**:
- `uint32_t dpu_slice_offset;`
- `uint32_t dpu_slice_elements;`
- `uint32_t contracted_offset;`
- `uint32_t contracted_elements_slice;`

To guarantee matching ABI:
- Append these parameters to `upmem_generic_args_t` exactly where they will natively align (they are 32-bit unsigned integers, so placing them consecutively at the end of the struct avoids implicit padding issues).
- Alternatively, if space in `upmem_generic_args_t` is constrained or strictly for tensor dimensions, they can be placed at the end of `resident_operation_t` or inside `upmem_generic_args_t` ensuring the total struct size remains a multiple of 8 bytes for DMA alignment.
- Verify `sizeof(resident_operation_t)` remains identical on host and DPU (padding explicit if necessary).

## 2. Host Driver Multi-DPU Operations (`host.c`)

**Design Updates**:
The `host.c` currently assumes a single DPU execution. 

**Modifications**:
- **Allocation**: Replace `dpu_alloc(1, ...)` with `dpu_alloc(request.requested_dpus, RESIDENT_ALLOCATION_PROFILE, &set);` where `request.requested_dpus` is provided in the manifest. Explicitly check the return status (`if (status != DPU_OK)`). On failure, set `failure_stage = "hardware_allocation_failed"`, clean up resources, and write a structured error JSON response instead of using a `DPU_ASSERT()` hard abort.
- **DPU Specific Configuration via `dpu_push_xfer`**:
  Instead of a single `dpu_broadcast_to` for `RESIDENT_OPERATIONS`, iterate over the DPU set.
  Use `dpu_prepare_xfer` and `dpu_push_xfer` to send tailored `RESIDENT_OPERATIONS` to each specific DPU. Each DPU receives its specific `dpu_slice_offset`, `dpu_slice_elements`, etc.
- **Synchronous Launch**:
  `dpu_launch(set, DPU_SYNCHRONOUS)` remains, but now applies to the multi-DPU set.
- **Sentinel & Metric Gathering**:
  Replace `DPU_FOREACH` `break` semantics with full iteration over the set to gather `RESIDENT_COMPLETION` records from all DPUs. Aggregate metrics like `dpu_run_time_cycles`.
- **Output Gathering**:
  Iterate via `DPU_FOREACH(set, dpu)` and compute the correct host-buffer offset based on `dpu_slice_offset` to gather output slices via `dpu_copy_from` directly into contiguous host buffers.

## 3. DPU Kernel Slice Work Division (`dpu.c`)

**Design Updates**:
The kernel must constrain its execution range purely to the slice given, but retain the existing tasklet loop division to maximize utilization within that slice.

**Modifications**:
- **`resident_contract`**: 
  - Change the global iteration boundary. Instead of looping from `0` to `operation->output_elements`, calculate relative to `args->dpu_slice_elements`.
  - The loop becomes:
    ```c
    for (uint32_t tile_start = tid * RESIDENT_OUTPUT_TILE_ELEMS; tile_start < args->dpu_slice_elements; tile_start += NR_TASKLETS * RESIDENT_OUTPUT_TILE_ELEMS) { ... }
    ```
  - When decoding the linear index to N-D coordinates (`resident_decode_index`), add `args->dpu_slice_offset` to `tile_start + tile_index` so the correct memory locations in `slot_a` and `slot_b` are accessed.
  - Apply similar bounding to the inner loop for partial sum computations (`contracted_offset` and `contracted_elements_slice`).
- **`resident_complex_combine`**:
  - Similar bound logic over `dpu_slice_elements` and index offsetting by `dpu_slice_offset`.
- **Barrier Synchronization**:
  - Leave `barrier_wait(&tasklet_barrier)` completely intact and unconditional. Multi-tasklet coordination continues seamlessly within the assigned DPU slice bounds.

## 4. Backward Compatibility & Single-DPU Guard

**Design Updates**:
A strictly identical execution flow must occur when only 1 DPU is requested.

**Modifications**:
- Ensure the Python session builder / C manifest parser defaults these slice parameters:
  - `requested_dpus = 1`
  - `dpu_slice_offset = 0`
  - `dpu_slice_elements = output_elements`
  - `contracted_offset = 0`
  - `contracted_elements_slice = contracted_elems`
- When `dpu_slice_offset == 0` and `dpu_slice_elements == output_elements`, the index math directly reduces to the existing M4.6 logic, proving no performance degradation or logic drift for single-DPU deployments.
