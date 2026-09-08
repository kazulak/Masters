/* Test-only scalar-MRAM reference for the v5 wave contract. */
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
#error "scalar reference requires NR_TASKLETS in [1,24]"
#endif

#ifdef __FAST_MATH__
#error "scalar reference requires strict float32 arithmetic"
#endif

/* Keep the wrapper's barrier name and tasklet geometry identical to dpu_wave.c. */
BARRIER_INIT(v4_barrier, NR_TASKLETS);

__mram_noinit uint8_t WAVE_MRAM[UPMEM_WAVE_MRAM_BYTES];
__host upmem_wave_control_t WAVE_CONTROL;
__host upmem_wave_completion_t WAVE_COMPLETION;
__host uint32_t WAVE_TASKLETS = NR_TASKLETS;

/* One private eight-byte transfer window per tasklet; no shared panel is allocated. */
__dma_aligned uint8_t scalar_io[NR_TASKLETS][8];

typedef union {
    float f32;
    int32_t i32;
    uint32_t u32;
} scalar_output_slot_t;

static int8_t scalar_read_i8(__mram_ptr uint8_t *arena, uint32_t offset,
        uint32_t tid) {
    const uint32_t aligned_offset = offset & ~7u;
    mram_read(arena + aligned_offset, scalar_io[tid], 8u);
    return (int8_t)scalar_io[tid][offset - aligned_offset];
}

static float scalar_read_f32(__mram_ptr uint8_t *arena, uint32_t byte_offset,
        uint32_t tid) {
    const uint32_t aligned_offset = byte_offset & ~7u;
    float value;
    mram_read(arena + aligned_offset, scalar_io[tid], 8u);
    __builtin_memcpy(&value, scalar_io[tid] + (byte_offset - aligned_offset),
        sizeof(value));
    return value;
}

static void scalar_write_u32(__mram_ptr uint8_t *arena, uint32_t offset,
        uint32_t value, uint32_t tid) {
    const uint32_t destination_alignment = offset & 7u;
    uint8_t *source = scalar_io[tid] + destination_alignment;
    __builtin_memcpy(source, &value, sizeof(value));
    mram_write_unaligned(source, arena + offset, sizeof(value));
}

/* All tasklets call this with the same geometry and own cyclic output rows. */
static void scalar_reference_panel_compute(__mram_ptr uint8_t *arena,
        uint32_t M, uint32_t N, uint32_t K, int is_int8, uint32_t a_offset,
        uint32_t b_offset, uint32_t c_offset) {
    const uint32_t tid = me();

    for (uint32_t row = tid; row < M; row += (uint32_t)NR_TASKLETS) {
        for (uint32_t col = 0u; col < N; ++col) {
            scalar_output_slot_t output;
            if (is_int8) {
                int32_t total = 0;
                for (uint32_t k = 0u; k < K; ++k) {
                    const int32_t a = (int32_t)scalar_read_i8(arena,
                        a_offset + row * K + k, tid);
                    const int32_t b = (int32_t)scalar_read_i8(arena,
                        b_offset + k * N + col, tid);
                    total += a * b;
                }
                output.i32 = total;
            } else {
                float total = 0.0f;
                for (uint32_t k = 0u; k < K; ++k) {
                    const float a = scalar_read_f32(arena,
                        a_offset + (row * K + k) * (uint32_t)sizeof(float), tid);
                    const float b = scalar_read_f32(arena,
                        b_offset + (k * N + col) * (uint32_t)sizeof(float), tid);
                    total += a * b;
                }
                output.f32 = total;
            }
            scalar_write_u32(arena,
                c_offset + (row * N + col) * (uint32_t)sizeof(uint32_t),
                output.u32, tid);
        }
    }
    barrier_wait(&v4_barrier);
}

/* Keep the accepted outer control shape executable without adding another kernel. */
static void scalar_reference_outer_compute(__mram_ptr uint8_t *arena,
        uint32_t M, uint32_t N, int is_int8, uint32_t a_offset,
        uint32_t b_offset, uint32_t c_offset) {
    scalar_reference_panel_compute(arena, M, N, 1u, is_int8, a_offset,
        b_offset, c_offset);
}

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
        /* Host dispatch owns the physical DPU-index mapping. */
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
                scalar_reference_outer_compute(WAVE_MRAM, WAVE_CONTROL.m,
                    WAVE_CONTROL.n, WAVE_CONTROL.numeric_mode == UPMEM_WAVE_INT8,
                    a, b, c);
            } else {
                scalar_reference_panel_compute(WAVE_MRAM, WAVE_CONTROL.m,
                    WAVE_CONTROL.n, WAVE_CONTROL.k,
                    WAVE_CONTROL.numeric_mode == UPMEM_WAVE_INT8, a, b, c);
            }
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
