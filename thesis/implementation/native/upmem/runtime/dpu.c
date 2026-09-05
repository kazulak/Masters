#include <barrier.h>
#include <defs.h>
#include <mram.h>
#include <mram_unaligned.h>
#include <perfcounter.h>
#include <stdint.h>

#include "protocol.h"

#ifndef NR_TASKLETS
#define NR_TASKLETS 1
#endif

#if NR_TASKLETS < 1 || NR_TASKLETS > EXECUTION_PLAN_V4_MAX_TASKLETS
#error "v4 requires NR_TASKLETS in [1,24]"
#endif

#include "panel_compute.h"

__mram_noinit uint8_t V4_MRAM[EXECUTION_PLAN_V4_MRAM_POOL_BYTES];
__host execution_plan_v4_control_t V4_CONTROL;
__host execution_plan_v4_completion_t V4_COMPLETION;

static volatile int v4_status = 0;
static perfcounter_t v4_start_cycles;

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
                 V4_CONTROL.m_elements > MAX_TILE_M || V4_CONTROL.n_elements > MAX_TILE_N ||
                 (uint64_t)V4_CONTROL.k_elements *
                     (uint64_t)EXECUTION_PLAN_V4_INT8_COMPONENT_PRODUCT > 2147483647u ||
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
            panel_compute(V4_MRAM, V4_CONTROL.m_elements, V4_CONTROL.n_elements,
                V4_CONTROL.k_elements,
                V4_CONTROL.numeric_mode == EXECUTION_PLAN_V4_NUMERIC_HOST_PACKED_INT8,
                V4_CONTROL.a_offset_bytes, V4_CONTROL.b_offset_bytes,
                V4_CONTROL.c_offset_bytes);
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
