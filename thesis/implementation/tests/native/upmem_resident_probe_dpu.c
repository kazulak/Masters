/* Test-only two-launch residency probe; production kernels/ABI are unchanged.
   Like the ordinary wave kernel, COMPLETED means the products were executed,
   not that they passed host numerical qualification. Final readbacks must be
   finite and pass the existing host decoder before a case can be accepted. */
#include <barrier.h>
#include <defs.h>
#include <mram.h>
#include <mram_unaligned.h>
#include <perfcounter.h>
#include <stdint.h>
#include <string.h>

#include "protocol.h"
#include "upmem_resident_probe.h"

#ifndef NR_TASKLETS
#define NR_TASKLETS 1
#endif
#if NR_TASKLETS < 1 || NR_TASKLETS > 24
#error "resident probe requires T1-T24"
#endif

#include "panel_compute.h"

__mram_noinit uint8_t PROBE_MRAM[UPMEM_WAVE_MRAM_BYTES];
__host resident_probe_plan_t PROBE_PLAN;
__host uint32_t PROBE_COMMAND;
__host upmem_wave_completion_t PROBE_COMPLETION;
static resident_probe_plan_t saved_plan;
static uint64_t last_pair_id;
static int phase; /* 0: empty, 1: producer complete, -1: poisoned */
static volatile int executable;
static uint32_t numeric_errors[NR_TASKLETS];
static perfcounter_t start_cycles;

static int finite_float(float value) {
    v4_output_slot_t bits;
    bits.f32 = value;
    return (bits.u32 & 0x7f800000u) != 0x7f800000u;
}

static void reconstruct_intermediate(void) {
    const uint32_t tid = me();
    const upmem_wave_control_t *first = &PROBE_PLAN.controls[0];
    const uint32_t count = first->m * first->n;
    /* Four 16-float planes fit the existing tasklet A buffer; the two
       reconstructed planes fit its output buffer. No extra WRAM arena. */
    float *lanes = (float *)tasklet_a_buffer[tid];
    float *output = (float *)tasklet_output_buffer[tid];
    for (uint32_t index = tid * 16u; index < count; index += NR_TASKLETS * 16u) {
        const uint32_t elements = count - index < 16u ? count - index : 16u;
        const uint32_t bytes = (elements * 4u + 7u) & ~7u;
        for (uint32_t lane = 0u; lane < 4u; ++lane)
            mram_read(PROBE_MRAM + first->planes[4u + lane].offset + index * 4u,
                      lanes + 16u * lane, bytes);
        for (uint32_t i = 0u; i < elements; ++i) {
            const float rr = 0.0f + lanes[i];
            const float ii = 0.0f + lanes[16u + i];
            const float ri = 0.0f + lanes[32u + i];
            const float ir = 0.0f + lanes[48u + i];
            output[i] = rr - ii;
            output[16u + i] = ri + ir;
            if (!finite_float(rr) || !finite_float(ii) || !finite_float(ri) || !finite_float(ir) ||
                    !finite_float(output[i]) || !finite_float(output[16u + i])) numeric_errors[tid] = 1u;
        }
        for (uint32_t component = 0u; component < 2u; ++component) {
            const uint32_t offset = PROBE_PLAN.retained[component].offset + index * 4u;
            if (elements % 2u == 0u) {
                mram_write(output + 16u * component, PROBE_MRAM + offset, elements * 4u);
            } else {
                /* Only the final block has a 4-byte tail. Its owner preserves
                   the padding, and no other tasklet writes that 8-byte span. */
                mram_write_unaligned(output + 16u * component, PROBE_MRAM + offset, elements * 4u);
            }
        }
    }
}

static void compute_products(const upmem_wave_control_t *control) {
    const uint32_t a_planes[4] = {0u, 1u, 0u, 1u};
    const uint32_t b_planes[4] = {2u, 3u, 3u, 2u};
    for (uint32_t product = 0u; product < 4u; ++product) {
        panel_compute(PROBE_MRAM, control->m, control->n, control->k, 0,
                      control->planes[a_planes[product]].offset,
                      control->planes[b_planes[product]].offset,
                      control->planes[4u + product].offset);
        if (me() == 0u) {
            PROBE_COMPLETION.completed_product_mask |= 1u << product;
            PROBE_COMPLETION.processed_elements += (uint64_t)control->m * control->n;
        }
    }
}

int main(void) {
    const uint32_t tid = me();
    numeric_errors[tid] = 0u;
    const uint32_t index = PROBE_COMMAND == RESIDENT_FIRST ? 0u : 1u;
    const upmem_wave_control_t *control = &PROBE_PLAN.controls[index];
    if (tid == 0u) {
        PROBE_COMPLETION = (upmem_wave_completion_t){
            .magic = UPMEM_WAVE_COMPLETION_MAGIC, .version = UPMEM_WAVE_VERSION,
            .status = UPMEM_WAVE_PENDING, .dpu_id = control->dpu_id,
            .operation_index = control->operation_index, .wave_id = control->wave_id,
            .request_sequence = control->request_sequence, .tile_id = control->tile_id,
            .failing_product = UPMEM_WAVE_NO_PRODUCT,
        };
        executable = phase != -1 && resident_probe_plan_valid(&PROBE_PLAN, NR_TASKLETS) &&
            PROBE_COMMAND >= RESIDENT_FIRST && PROBE_COMMAND <= RESIDENT_LOCAL_SECOND;
        if (executable) {
            if (PROBE_COMMAND == RESIDENT_FIRST) {
                executable = phase == 0 && PROBE_PLAN.pair_id > last_pair_id;
                if (executable) {
                    saved_plan = PROBE_PLAN;
                    last_pair_id = PROBE_PLAN.pair_id;
                }
            } else {
                executable = phase == 1 && memcmp(&PROBE_PLAN, &saved_plan, sizeof(saved_plan)) == 0;
            }
        }
        if (!executable) {
            phase = -1;
            PROBE_COMPLETION.status = UPMEM_WAVE_FAILED;
            PROBE_COMPLETION.failure_stage = UPMEM_WAVE_FAILURE_VALIDATION;
        } else {
            perfcounter_config(COUNT_CYCLES, true);
            start_cycles = perfcounter_get();
        }
    }
    barrier_wait(&v4_barrier);
    if (executable && PROBE_COMMAND == RESIDENT_LOCAL_SECOND) reconstruct_intermediate();
    barrier_wait(&v4_barrier);
    if (tid == 0u && executable) {
        for (uint32_t i = 0u; i < NR_TASKLETS; ++i) {
            if (numeric_errors[i]) {
                executable = 0;
                phase = -1;
                PROBE_COMPLETION.status = UPMEM_WAVE_FAILED;
                PROBE_COMPLETION.failure_stage = UPMEM_WAVE_FAILURE_EXECUTION;
                /* Invalid reconstruction prevents the first consumer product.
                   No consumer product is included in the completed prefix. */
                PROBE_COMPLETION.failing_product = 0u;
            }
        }
    }
    barrier_wait(&v4_barrier);
    if (executable) compute_products(control);
    barrier_wait(&v4_barrier);
    if (tid == 0u && executable) {
        PROBE_COMPLETION.cycles = (uint64_t)(perfcounter_get() - start_cycles);
        PROBE_COMPLETION.status = UPMEM_WAVE_COMPLETED;
        phase = PROBE_COMMAND == RESIDENT_FIRST ? 1 : 0;
    }
    barrier_wait(&v4_barrier);
    return PROBE_COMPLETION.status == UPMEM_WAVE_COMPLETED ? 0 : 1;
}
