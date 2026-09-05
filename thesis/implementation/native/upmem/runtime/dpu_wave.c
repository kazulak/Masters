#include <barrier.h>
#include <defs.h>
#include <mram.h>
#include <mram_unaligned.h>
#include <perfcounter.h>
#include <stdint.h>

#include "protocol.h"
#include "wave_protocol.h"

#ifndef NR_TASKLETS
#define NR_TASKLETS 1
#endif
#if NR_TASKLETS < 1 || NR_TASKLETS > UPMEM_WAVE_MAX_TASKLETS
#error "wave kernel requires NR_TASKLETS in [1,24]"
#endif

#include "panel_compute.h"
#include "outer_compute.h"

__mram_noinit uint8_t WAVE_MRAM[UPMEM_WAVE_MRAM_BYTES];
__host upmem_wave_control_t WAVE_CONTROL;
__host upmem_wave_completion_t WAVE_COMPLETION;
__host uint32_t WAVE_TASKLETS = NR_TASKLETS;
static volatile int wave_valid;
static perfcounter_t start_cycles;

int main(void) {
    const uint32_t tid = me();
    if (tid == 0u) {
        WAVE_COMPLETION = (upmem_wave_completion_t){
            .magic = UPMEM_WAVE_COMPLETION_MAGIC,
            .version = UPMEM_WAVE_VERSION,
            .status = UPMEM_WAVE_PENDING,
            .dpu_id = WAVE_CONTROL.dpu_id,
            .operation_index = WAVE_CONTROL.operation_index,
            .wave_id = WAVE_CONTROL.wave_id,
            .request_sequence = WAVE_CONTROL.request_sequence,
            .tile_id = WAVE_CONTROL.tile_id,
            .failing_product = UPMEM_WAVE_NO_PRODUCT,
        };
        /* Host dispatch owns the physical DPU-index mapping; no hardware index is read here. */
        wave_valid = upmem_wave_control_valid(&WAVE_CONTROL,
            WAVE_CONTROL.dpu_id, NR_TASKLETS);
        if (!wave_valid) {
            WAVE_COMPLETION.status = UPMEM_WAVE_FAILED;
            WAVE_COMPLETION.failure_stage = UPMEM_WAVE_FAILURE_VALIDATION;
        } else {
            perfcounter_config(COUNT_CYCLES, true);
            start_cycles = perfcounter_get();
        }
    }
    barrier_wait(&v4_barrier);
    if (wave_valid && WAVE_CONTROL.flags != UPMEM_WAVE_IDLE) {
        /* Products remain separate: the host retains its original reduction order. */
        const uint32_t a_planes[4] = {UPMEM_WAVE_A_REAL, UPMEM_WAVE_A_IMAG,
            UPMEM_WAVE_A_REAL, UPMEM_WAVE_A_IMAG};
        const uint32_t b_planes[4] = {UPMEM_WAVE_B_REAL, UPMEM_WAVE_B_IMAG,
            UPMEM_WAVE_B_IMAG, UPMEM_WAVE_B_REAL};
        const uint32_t count = upmem_wave_kernel_products(WAVE_CONTROL.kernel);
        for (uint32_t product = 0u; product < count; ++product) {
            const uint32_t a = WAVE_CONTROL.planes[a_planes[product]].offset;
            const uint32_t b = WAVE_CONTROL.planes[b_planes[product]].offset;
            const uint32_t c = WAVE_CONTROL.planes[UPMEM_WAVE_RR + product].offset;
            if (WAVE_CONTROL.kernel == UPMEM_WAVE_KERNEL_REAL_OUTER ||
                    WAVE_CONTROL.kernel == UPMEM_WAVE_KERNEL_FOUR_PRODUCT_OUTER) {
                outer_compute(WAVE_MRAM, WAVE_CONTROL.m, WAVE_CONTROL.n,
                    WAVE_CONTROL.numeric_mode == UPMEM_WAVE_INT8, a, b, c);
            } else {
                panel_compute(WAVE_MRAM, WAVE_CONTROL.m, WAVE_CONTROL.n,
                    WAVE_CONTROL.k, WAVE_CONTROL.numeric_mode == UPMEM_WAVE_INT8, a, b, c);
            }
            /* Each helper's final barrier bounds the product's shared-buffer lifetime. */
            if (tid == 0u) {
                WAVE_COMPLETION.completed_product_mask |= 1u << product;
                WAVE_COMPLETION.processed_elements +=
                    (uint64_t)WAVE_CONTROL.m * WAVE_CONTROL.n;
            }
        }
    }
    barrier_wait(&v4_barrier);
    if (tid == 0u && wave_valid) {
        WAVE_COMPLETION.cycles = (uint64_t)(perfcounter_get() - start_cycles);
        WAVE_COMPLETION.status = UPMEM_WAVE_COMPLETED;
    }
    barrier_wait(&v4_barrier);
    return wave_valid ? 0 : 1;
}
