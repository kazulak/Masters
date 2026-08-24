#include <barrier.h>
#include <defs.h>
#include <mram.h>
#include <perfcounter.h>
#include <stdint.h>

#include "protocol.h"

#ifndef NR_TASKLETS
#define NR_TASKLETS 1
#endif

#if NR_TASKLETS < 1 || NR_TASKLETS > EXECUTION_PLAN_V4_MAX_TASKLETS
#error "v4 requires NR_TASKLETS in [1,24]"
#endif

BARRIER_INIT(v4_barrier, NR_TASKLETS);

__mram_noinit uint8_t V4_MRAM[EXECUTION_PLAN_V4_MRAM_POOL_BYTES];
__host execution_plan_v4_control_t V4_CONTROL;
__host execution_plan_v4_completion_t V4_COMPLETION;

__dma_aligned uint8_t v4_input_window[NR_TASKLETS][8];
__dma_aligned uint8_t v4_output_window[NR_TASKLETS][8];

static volatile int v4_status = 0;
static perfcounter_t v4_start_cycles;

static float v4_read_f32(uint32_t offset, uint32_t element) {
    const uint32_t byte_offset = offset + element * (uint32_t)sizeof(float);
    const uint32_t aligned_offset = byte_offset & ~7u;
    float value = 0.0f;
    mram_read(V4_MRAM + aligned_offset, v4_input_window[me()], sizeof(v4_input_window[me()]));
    __builtin_memcpy(&value, v4_input_window[me()] + (byte_offset - aligned_offset), sizeof(value));
    return value;
}

static int32_t v4_read_i8(uint32_t offset, uint32_t element) {
    const uint32_t byte_offset = offset + element;
    const uint32_t aligned_offset = byte_offset & ~7u;
    mram_read(V4_MRAM + aligned_offset, v4_input_window[me()], sizeof(v4_input_window[me()]));
    return (int32_t)(int8_t)v4_input_window[me()][byte_offset - aligned_offset];
}

static void v4_compute_pair(uint32_t output_index, uint32_t valid_count) {
    for (uint32_t pair_index = output_index; pair_index < valid_count; pair_index += 2u) {
        const uint32_t row = pair_index / V4_CONTROL.n_elements;
        const uint32_t column = pair_index % V4_CONTROL.n_elements;
        float float_value = 0.0f;
        int32_t int_value = 0;
        for (uint32_t k = 0u; k < V4_CONTROL.k_elements; k++) {
            const uint32_t a_index = row * V4_CONTROL.k_elements + k;
            const uint32_t b_index = k * V4_CONTROL.n_elements + column;
            if (V4_CONTROL.numeric_mode == EXECUTION_PLAN_V4_NUMERIC_HOST_PACKED_INT8) {
                int_value += v4_read_i8(V4_CONTROL.a_offset_bytes, a_index) *
                    v4_read_i8(V4_CONTROL.b_offset_bytes, b_index);
            } else {
                float_value += v4_read_f32(V4_CONTROL.a_offset_bytes, a_index) *
                    v4_read_f32(V4_CONTROL.b_offset_bytes, b_index);
            }
        }
        uint32_t first_value;
        __builtin_memcpy(&first_value,
            V4_CONTROL.numeric_mode == EXECUTION_PLAN_V4_NUMERIC_HOST_PACKED_INT8
                ? (const void *)&int_value : (const void *)&float_value,
            sizeof(first_value));
        uint32_t second_value = 0u;
        if (pair_index + 1u < valid_count) {
            const uint32_t next_row = (pair_index + 1u) / V4_CONTROL.n_elements;
            const uint32_t next_column = (pair_index + 1u) % V4_CONTROL.n_elements;
            float next_float = 0.0f;
            int32_t next_int = 0;
            for (uint32_t k = 0u; k < V4_CONTROL.k_elements; k++) {
                const uint32_t a_index = next_row * V4_CONTROL.k_elements + k;
                const uint32_t b_index = k * V4_CONTROL.n_elements + next_column;
                if (V4_CONTROL.numeric_mode == EXECUTION_PLAN_V4_NUMERIC_HOST_PACKED_INT8) {
                    next_int += v4_read_i8(V4_CONTROL.a_offset_bytes, a_index) *
                        v4_read_i8(V4_CONTROL.b_offset_bytes, b_index);
                } else {
                    next_float += v4_read_f32(V4_CONTROL.a_offset_bytes, a_index) *
                        v4_read_f32(V4_CONTROL.b_offset_bytes, b_index);
                }
            }
            __builtin_memcpy(&second_value,
                V4_CONTROL.numeric_mode == EXECUTION_PLAN_V4_NUMERIC_HOST_PACKED_INT8
                    ? (const void *)&next_int : (const void *)&next_float,
                sizeof(second_value));
        }
        __builtin_memset(v4_output_window[me()], 0, sizeof(v4_output_window[me()]));
        __builtin_memcpy(v4_output_window[me()], &first_value, sizeof(first_value));
        __builtin_memcpy(v4_output_window[me()] + sizeof(first_value), &second_value, sizeof(second_value));
        mram_write(v4_output_window[me()], V4_MRAM + V4_CONTROL.c_offset_bytes +
            pair_index * sizeof(uint32_t), sizeof(v4_output_window[me()]));
        break;
    }
}

int main(void) {
    const uint32_t tid = me();
    if (tid == 0u) {
        v4_status = 0;
        V4_COMPLETION.magic = EXECUTION_PLAN_V4_COMPLETION_MAGIC;
        V4_COMPLETION.version = EXECUTION_PLAN_V4_VERSION;
        V4_COMPLETION.status = EXECUTION_PLAN_V4_STATUS_PENDING;
        V4_COMPLETION.dpu_id = V4_CONTROL.dpu_id;
        V4_COMPLETION.cycles = 0u;
        V4_COMPLETION.processed_elements = 0u;
        V4_COMPLETION.checksum_fnv1a64 = 0u;
        if (V4_CONTROL.magic != EXECUTION_PLAN_V4_CONTROL_MAGIC ||
            V4_CONTROL.version != EXECUTION_PLAN_V4_VERSION ||
            V4_CONTROL.reserved0 != (uint32_t)NR_TASKLETS ||
            (V4_CONTROL.numeric_mode != EXECUTION_PLAN_V4_NUMERIC_FLOAT32 &&
             V4_CONTROL.numeric_mode != EXECUTION_PLAN_V4_NUMERIC_HOST_PACKED_INT8) ||
            ((V4_CONTROL.flags & EXECUTION_PLAN_V4_FLAG_ZERO_WORK) == 0u &&
                (V4_CONTROL.m_elements == 0u || V4_CONTROL.n_elements == 0u ||
                 V4_CONTROL.k_elements == 0u || V4_CONTROL.k_elements > EXECUTION_PLAN_V4_MAX_CONTRACTED ||
                 (uint64_t)V4_CONTROL.k_elements * 128u * 128u > 2147483647u ||
                 (uint64_t)V4_CONTROL.m_elements * V4_CONTROL.n_elements > UINT32_MAX)) ||
            ((V4_CONTROL.flags & EXECUTION_PLAN_V4_FLAG_ZERO_WORK) != 0u &&
                (V4_CONTROL.m_elements != 0u || V4_CONTROL.n_elements != 0u || V4_CONTROL.k_elements != 0u))) {
            v4_status = 1;
        }
    }
    barrier_wait(&v4_barrier);
    if (v4_status == 0) {
        if (tid == 0u) {
            perfcounter_config(COUNT_CYCLES, true);
            v4_start_cycles = perfcounter_get();
        }
        barrier_wait(&v4_barrier);
        if ((V4_CONTROL.flags & EXECUTION_PLAN_V4_FLAG_ZERO_WORK) == 0u) {
            for (uint32_t pair_index = tid * 2u;
                 pair_index < V4_CONTROL.m_elements * V4_CONTROL.n_elements;
                 pair_index += NR_TASKLETS * 2u) {
                v4_compute_pair(pair_index, V4_CONTROL.m_elements * V4_CONTROL.n_elements);
            }
        }
        barrier_wait(&v4_barrier);
        if (tid == 0u) {
            V4_COMPLETION.status = EXECUTION_PLAN_V4_STATUS_COMPLETED;
            V4_COMPLETION.cycles = (uint64_t)(perfcounter_get() - v4_start_cycles);
            V4_COMPLETION.processed_elements = (uint64_t)V4_CONTROL.m_elements * V4_CONTROL.n_elements;
        }
    } else if (tid == 0u) {
        V4_COMPLETION.status = EXECUTION_PLAN_V4_STATUS_FAILED;
    }
    barrier_wait(&v4_barrier);
    return v4_status;
}
