#include <barrier.h>
#include <defs.h>
#include <mram.h>
#include <perfcounter.h>
#include <stddef.h>
#include <stdint.h>

#include "common.h"

#ifndef UPMEM_GENERIC_HARDWARE_MVP
#define UPMEM_GENERIC_HARDWARE_MVP 0
#endif

BARRIER_INIT(tasklet_barrier, NR_TASKLETS);

/* These names are resident-only symbols.  The legacy generic-loop symbols
 * GENERIC_ARGS/GENERIC_A_RAW/GENERIC_B_RAW/GENERIC_C_RAW are not reused. */
__host resident_slot_descriptor_t RESIDENT_SLOT_DESCRIPTORS[RESIDENT_MAX_SLOT_DESCRIPTORS];
__host resident_control_t RESIDENT_CONTROL;
__host uint64_t RESIDENT_ACTIVE_OPERATION;
__host resident_completion_t RESIDENT_COMPLETION;
__mram_noinit resident_operation_t RESIDENT_OPERATIONS[RESIDENT_MAX_COMPONENT_OPS];
__mram_noinit uint8_t RESIDENT_SLOT_POOL[RESIDENT_MRAM_POOL_BYTES];

__dma_aligned uint8_t resident_operation_window[sizeof(resident_operation_t)];
__dma_aligned float resident_output_tile[NR_TASKLETS][RESIDENT_OUTPUT_TILE_ELEMS + 2u];
__dma_aligned uint8_t resident_input_window[NR_TASKLETS][8];

static uint32_t resident_align8(uint32_t bytes) {
    return (bytes + 7u) & ~7u;
}

static const resident_slot_descriptor_t *resident_slot(uint32_t slot_id) {
    for (uint32_t index = 0; index < RESIDENT_CONTROL.slot_count; index++) {
        if (RESIDENT_SLOT_DESCRIPTORS[index].slot_id == slot_id) {
            return &RESIDENT_SLOT_DESCRIPTORS[index];
        }
    }
    return NULL;
}

static float resident_read_f32(uint32_t slot_id, uint32_t element) {
    const uint32_t tid = me();
    const resident_slot_descriptor_t *slot = resident_slot(slot_id);
    float value;
    if (slot == NULL || element >= slot->element_count) {
        return 0.0f;
    }
    const uint32_t byte_offset = slot->offset_bytes + element * (uint32_t)sizeof(float);
    const uint32_t aligned_offset = byte_offset & ~7u;
    mram_read(RESIDENT_SLOT_POOL + aligned_offset, resident_input_window[tid], sizeof(resident_input_window[tid]));
    __builtin_memcpy(&value, &resident_input_window[tid][byte_offset - aligned_offset], sizeof(value));
    return value;
}

static float resident_scale(uint32_t slot_id, uint32_t elements) {
    float max_abs = 0.0f;
    for (uint32_t index = 0; index < elements; index++) {
        const float value = resident_read_f32(slot_id, index);
        const float absolute = value < 0.0f ? -value : value;
        if (absolute > max_abs) max_abs = absolute;
    }
    return max_abs == 0.0f ? 1.0f : max_abs / 127.0f;
}

static float resident_floor(float value) {
    union {
        float value;
        uint32_t bits;
    } encoded = {value};
    const uint32_t exponent = (encoded.bits >> 23u) & 0xffu;
    const uint32_t fraction = encoded.bits & 0x7fffffu;
    const uint32_t sign = encoded.bits >> 31u;
    if (exponent == 0xffu || value == 0.0f) return value;
    if (exponent < 127u) return sign ? -1.0f : 0.0f;
    {
        const uint32_t integer_bits = exponent - 127u;
        if (integer_bits >= 23u) return value;
        const uint32_t fractional_mask = (1u << (23u - integer_bits)) - 1u;
        if ((fraction & fractional_mask) == 0u) return value;
        encoded.bits &= ~fractional_mask;
        return sign ? encoded.value - 1.0f : encoded.value;
    }
}

/* The rounding decision is explicit.  The only casts below convert an
 * already-selected integral value in the bounded [-128,128] range; ties do
 * not depend on the C integer-conversion rounding behavior. */
static int32_t resident_round_nearest_even(float value) {
    const float lower = resident_floor(value);
    const float fraction = value - lower;
    if (fraction < 0.5f) return (int32_t)lower;
    if (fraction > 0.5f) return (int32_t)(lower + 1.0f);
    const int32_t lower_integer = (int32_t)lower;
    return (lower_integer & 1) == 0 ? lower_integer : lower_integer + 1;
}

static int32_t resident_quantized(uint32_t slot_id, uint32_t element, float scale) {
    const float scaled = resident_read_f32(slot_id, element) / scale;
    int32_t value = resident_round_nearest_even(scaled);
    if (value < -127) value = -127;
    if (value > 127) value = 127;
    return value;
}

static void resident_decode_index(
    uint32_t linear,
    uint32_t rank,
    const uint32_t *shape,
    const uint32_t *strides,
    uint32_t *coords
) {
    for (uint32_t axis = 0; axis < rank; axis++) {
        const uint32_t stride = strides[axis];
        coords[axis] = stride == 0 ? 0 : linear / stride;
        linear -= coords[axis] * stride;
        if (shape[axis] == 0) coords[axis] = 0;
    }
}

static uint64_t resident_checksum_update(uint64_t checksum, float value) {
    union {
        float value;
        uint32_t bits;
    } encoded = {value};
    for (uint32_t shift = 0; shift < sizeof(encoded.bits) * 8u; shift += 8u) {
        checksum ^= (uint8_t)(encoded.bits >> shift);
        checksum *= 1099511628211ULL;
    }
    return checksum;
}

static void resident_contract(const resident_operation_t *operation) {
    const uint32_t tid = me();
    const upmem_generic_args_t *args = &operation->args;
    uint32_t output_coords[UPMEM_GENERIC_MAX_RANK] = {0};
    const int requantize = operation->mode == 1u;
    const float left_scale = requantize ? resident_scale(operation->slot_a, args->left_elems) : 1.0f;
    const float right_scale = requantize ? resident_scale(operation->slot_b, args->right_elems) : 1.0f;

    const uint32_t dpu_slice_elements = (args->dpu_slice_elements > 0) ? args->dpu_slice_elements : operation->output_elements;
    const uint32_t dpu_slice_offset = args->dpu_slice_offset;
    const uint32_t contracted_elements = (args->contracted_elements_slice > 0) ? args->contracted_elements_slice : args->contracted_elems;
    const uint32_t contracted_offset = args->contracted_offset;

    for (uint32_t tile_start = tid * RESIDENT_OUTPUT_TILE_ELEMS; tile_start < dpu_slice_elements; tile_start += NR_TASKLETS * RESIDENT_OUTPUT_TILE_ELEMS) {
        const uint32_t tile_elems = (dpu_slice_elements - tile_start) < RESIDENT_OUTPUT_TILE_ELEMS
            ? dpu_slice_elements - tile_start : RESIDENT_OUTPUT_TILE_ELEMS;
        for (uint32_t tile_index = 0; tile_index < tile_elems; tile_index++) {
            const uint32_t output_linear = dpu_slice_offset + tile_start + tile_index;
            resident_decode_index(output_linear, args->output_rank, args->output_shape, args->output_strides, output_coords);
            uint32_t left_base = 0;
            uint32_t right_base = 0;
            for (uint32_t output_axis = 0; output_axis < args->output_rank; output_axis++) {
                const int32_t left_axis = args->output_to_left_axes[output_axis];
                const int32_t right_axis = args->output_to_right_axes[output_axis];
                if (left_axis >= 0) left_base += output_coords[output_axis] * args->left_strides[(uint32_t)left_axis];
                if (right_axis >= 0) right_base += output_coords[output_axis] * args->right_strides[(uint32_t)right_axis];
            }
            int32_t total_i32 = 0;
            float total_f32 = 0.0f;
            for (uint32_t c_idx = 0; c_idx < contracted_elements; c_idx++) {
                const uint32_t contracted_linear = contracted_offset + c_idx;
                uint32_t left_offset = left_base;
                uint32_t right_offset = right_base;
                uint32_t remaining = contracted_linear;
                for (uint32_t contracted_axis = 0; contracted_axis < args->contracted_rank; contracted_axis++) {
                    uint32_t stride = 1;
                    for (uint32_t later = contracted_axis + 1; later < args->contracted_rank; later++) {
                        stride *= args->contracted_dims[later];
                    }
                    const uint32_t coordinate = stride == 0 ? 0 : remaining / stride;
                    remaining -= coordinate * stride;
                    left_offset += coordinate * args->left_strides[(uint32_t)args->contracted_to_left_axes[contracted_axis]];
                    right_offset += coordinate * args->right_strides[(uint32_t)args->contracted_to_right_axes[contracted_axis]];
                }
                if (requantize) {
                    total_i32 += resident_quantized(operation->slot_a, left_offset, left_scale)
                        * resident_quantized(operation->slot_b, right_offset, right_scale);
                } else {
                    total_f32 += resident_read_f32(operation->slot_a, left_offset)
                        * resident_read_f32(operation->slot_b, right_offset);
                }
            }
            resident_output_tile[tid][tile_index] = requantize
                ? (float)total_i32 * left_scale * right_scale : total_f32;
        }
        if ((tile_elems & 1u) != 0u) resident_output_tile[tid][tile_elems] = 0.0f;
        const uint32_t tile_bytes = resident_align8(tile_elems * (uint32_t)sizeof(float));
        const resident_slot_descriptor_t *output = resident_slot(operation->slot_out_real);
        if (output != NULL) {
            mram_write(
                resident_output_tile[tid],
                RESIDENT_SLOT_POOL + output->offset_bytes + (dpu_slice_offset + tile_start) * sizeof(float),
                tile_bytes
            );
        }
    }
}

static void resident_complex_combine(const resident_operation_t *operation) {
    const uint32_t tid = me();
    const upmem_generic_args_t *args = &operation->args;
    const uint32_t dpu_slice_elements = (args->dpu_slice_elements > 0) ? args->dpu_slice_elements : operation->output_elements;
    const uint32_t dpu_slice_offset = args->dpu_slice_offset;

    for (uint32_t tile_start = tid * RESIDENT_OUTPUT_TILE_ELEMS; tile_start < dpu_slice_elements; tile_start += NR_TASKLETS * RESIDENT_OUTPUT_TILE_ELEMS) {
        const uint32_t tile_elems = (dpu_slice_elements - tile_start) < RESIDENT_OUTPUT_TILE_ELEMS
            ? dpu_slice_elements - tile_start : RESIDENT_OUTPUT_TILE_ELEMS;
        for (uint32_t index = 0; index < tile_elems; index++) {
            const uint32_t element = dpu_slice_offset + tile_start + index;
            resident_output_tile[tid][index] = resident_read_f32(operation->slot_a, element)
                - resident_read_f32(operation->slot_b, element);
        }
        if ((tile_elems & 1u) != 0u) resident_output_tile[tid][tile_elems] = 0.0f;
        const uint32_t tile_bytes = resident_align8(tile_elems * (uint32_t)sizeof(float));
        const resident_slot_descriptor_t *output = resident_slot(operation->slot_out_real);
        if (output != NULL) {
            mram_write(resident_output_tile[tid], RESIDENT_SLOT_POOL + output->offset_bytes + (dpu_slice_offset + tile_start) * sizeof(float), tile_bytes);
        }
        for (uint32_t index = 0; index < tile_elems; index++) {
            const uint32_t element = dpu_slice_offset + tile_start + index;
            resident_output_tile[tid][index] = resident_read_f32(operation->slot_c, element)
                + resident_read_f32(operation->slot_d, element);
        }
        if ((tile_elems & 1u) != 0u) resident_output_tile[tid][tile_elems] = 0.0f;
        output = resident_slot(operation->slot_out_imag);
        if (output != NULL) {
            mram_write(resident_output_tile[tid], RESIDENT_SLOT_POOL + output->offset_bytes + (dpu_slice_offset + tile_start) * sizeof(float), tile_bytes);
        }
    }
}

static perfcounter_t start_cycles_shared;
static int status_code;

int main(void) {
    if (me() == 0) {
        status_code = 0;
        RESIDENT_COMPLETION.magic = RESIDENT_COMPLETION_MAGIC;
        RESIDENT_COMPLETION.version = RESIDENT_COMPLETION_VERSION;
        RESIDENT_COMPLETION.active_operation_index = (uint32_t)RESIDENT_ACTIVE_OPERATION;
        RESIDENT_COMPLETION.completion_status = RESIDENT_COMPLETION_PENDING;
        RESIDENT_COMPLETION.completed_operation_count = 0u;
        RESIDENT_COMPLETION.output_elements_processed = 0u;
        RESIDENT_COMPLETION.output_checksum_fnv1a64 = 14695981039346656037ULL;
        RESIDENT_COMPLETION.dpu_run_time_cycles = 0ULL;
        if (RESIDENT_ACTIVE_OPERATION >= RESIDENT_CONTROL.operation_count) {
            status_code = 3;
        }
    }
    barrier_wait(&tasklet_barrier);

    if (status_code == 0) {
        resident_operation_t operation;
        if (me() == 0) {
            mram_read(
                RESIDENT_OPERATIONS + RESIDENT_ACTIVE_OPERATION,
                &resident_operation_window,
                sizeof(operation)
            );
        }
        barrier_wait(&tasklet_barrier);
        __builtin_memcpy(&operation, resident_operation_window, sizeof(operation));

        if (me() == 0) {
            perfcounter_config(COUNT_CYCLES, true);
            start_cycles_shared = perfcounter_get();
        }
        barrier_wait(&tasklet_barrier);

        if (operation.kind == RESIDENT_OPERATION_CONTRACT) {
            resident_contract(&operation);
        } else if (operation.kind == RESIDENT_OPERATION_COMPLEX_COMBINE) {
            resident_complex_combine(&operation);
        } else {
            if (me() == 0) {
                status_code = 4;
            }
        }

        barrier_wait(&tasklet_barrier);

        if (status_code == 0 && me() == 0) {
            const perfcounter_t end_cycles = perfcounter_get();
            RESIDENT_COMPLETION.dpu_run_time_cycles = (uint64_t)(end_cycles - start_cycles_shared);
            RESIDENT_COMPLETION.active_operation_index = (uint32_t)RESIDENT_ACTIVE_OPERATION;
            RESIDENT_COMPLETION.completion_status = RESIDENT_COMPLETION_COMPLETED;
            RESIDENT_COMPLETION.completed_operation_count = (uint32_t)RESIDENT_ACTIVE_OPERATION + 1u;
            RESIDENT_COMPLETION.output_elements_processed = operation.output_elements;

            uint64_t checksum = 14695981039346656037ULL;
            const resident_slot_descriptor_t *out_slot = resident_slot(operation.slot_out_real);
            if (out_slot != NULL) {
                for (uint32_t idx = 0; idx < operation.output_elements; idx++) {
                    const float val = resident_read_f32(operation.slot_out_real, idx);
                    checksum = resident_checksum_update(checksum, val);
                }
            }
            if (operation.kind == RESIDENT_OPERATION_COMPLEX_COMBINE) {
                const resident_slot_descriptor_t *out_imag_slot = resident_slot(operation.slot_out_imag);
                if (out_imag_slot != NULL) {
                    for (uint32_t idx = 0; idx < operation.output_elements; idx++) {
                        const float val = resident_read_f32(operation.slot_out_imag, idx);
                        checksum = resident_checksum_update(checksum, val);
                    }
                }
            }
            RESIDENT_COMPLETION.output_checksum_fnv1a64 = checksum;
        }
        barrier_wait(&tasklet_barrier);
    }
    return status_code;
}
