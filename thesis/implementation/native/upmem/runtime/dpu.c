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

#define KC EXECUTION_PLAN_V4_WRAM_PANEL_KC
#define NC EXECUTION_PLAN_V4_WRAM_PANEL_NC
#define MAX_TILE_M 256u
#define MAX_TILE_N 256u
#define B_CONTIGUOUS_CHUNK_BYTES EXECUTION_PLAN_V4_WRAM_PANEL_DMA_BYTES
#define B_PANEL_DATA_BYTES (KC * NC * sizeof(float))
#define B_PANEL_ROW_STRIDE_BYTES (NC * sizeof(float))
#define A_BUFFER_DATA_BYTES (KC * sizeof(float))
#define OUTPUT_BUFFER_DATA_BYTES (NC * sizeof(uint32_t))
#define UNALIGNED_SCRATCH_BYTES EXECUTION_PLAN_V4_WRAM_PANEL_UNALIGNED_SCRATCH_BYTES

BARRIER_INIT(v4_barrier, NR_TASKLETS);

__mram_noinit uint8_t V4_MRAM[EXECUTION_PLAN_V4_MRAM_POOL_BYTES];
__host execution_plan_v4_control_t V4_CONTROL;
__host execution_plan_v4_completion_t V4_COMPLETION;

typedef union {
    float f32;
    int32_t i32;
    uint32_t u32;
} v4_output_slot_t;

__dma_aligned uint8_t shared_b_panel[B_PANEL_DATA_BYTES];
__dma_aligned uint8_t tasklet_a_buffer[NR_TASKLETS][A_BUFFER_DATA_BYTES];
/* K-chunk partials return to the tile's MRAM output between panel iterations. */
__dma_aligned v4_output_slot_t tasklet_output_buffer[NR_TASKLETS][NC];
__dma_aligned uint8_t tasklet_unaligned_scratch[NR_TASKLETS][UNALIGNED_SCRATCH_BYTES];

_Static_assert(KC == 64u, "KC must be 64");
_Static_assert(NC == 32u, "NC must be 32");
_Static_assert(MAX_TILE_M == 256u, "MAX_TILE_M must be 256");
_Static_assert(MAX_TILE_N == 256u, "MAX_TILE_N must be 256");
_Static_assert(B_PANEL_DATA_BYTES == 8192u, "shared B panel must be 8 KiB");
_Static_assert(B_PANEL_ROW_STRIDE_BYTES == 128u, "float B rows must be 128 bytes");
_Static_assert(A_BUFFER_DATA_BYTES == 256u, "float A rows must be 256 bytes");
_Static_assert(OUTPUT_BUFFER_DATA_BYTES == 128u, "output rows must be 128 bytes");
_Static_assert(sizeof(shared_b_panel) == B_PANEL_DATA_BYTES,
    "shared_b_panel dimension mismatch");
_Static_assert(sizeof(tasklet_a_buffer) == (uint32_t)NR_TASKLETS * A_BUFFER_DATA_BYTES,
    "tasklet_a_buffer dimension mismatch");
_Static_assert(sizeof(tasklet_output_buffer) ==
    (uint32_t)NR_TASKLETS * NC * sizeof(v4_output_slot_t),
    "tasklet_output_buffer dimension mismatch");
_Static_assert(sizeof(tasklet_unaligned_scratch) ==
    (uint32_t)NR_TASKLETS * UNALIGNED_SCRATCH_BYTES,
    "tasklet_unaligned_scratch dimension mismatch");
_Static_assert(UNALIGNED_SCRATCH_BYTES >= A_BUFFER_DATA_BYTES + 16u,
    "unaligned scratch must hold max payload + 16");
_Static_assert(UNALIGNED_SCRATCH_BYTES % 8u == 0u,
    "unaligned scratch must be 8-byte aligned size");
_Static_assert(B_CONTIGUOUS_CHUNK_BYTES <= 2048u && B_CONTIGUOUS_CHUNK_BYTES % 8u == 0u,
    "B contiguous chunk must be <= 2048 and divisible by 8");
_Static_assert(sizeof(shared_b_panel) % B_CONTIGUOUS_CHUNK_BYTES == 0u,
    "shared_b_panel must be divisible by contiguous chunk size");
_Static_assert(B_PANEL_ROW_STRIDE_BYTES <= 2048u && B_PANEL_ROW_STRIDE_BYTES % 8u == 0u,
    "B row float DMA size must be <= 2048 and divisible by 8");
_Static_assert(NC * sizeof(int8_t) <= 2048u && (NC * sizeof(int8_t)) % 8u == 0u,
    "B row int8 DMA size must be <= 2048 and divisible by 8");
_Static_assert(A_BUFFER_DATA_BYTES <= 2048u && A_BUFFER_DATA_BYTES % 8u == 0u,
    "A row float DMA size must be <= 2048 and divisible by 8");
_Static_assert(KC * sizeof(int8_t) <= 2048u && (KC * sizeof(int8_t)) % 8u == 0u,
    "A row int8 DMA size must be <= 2048 and divisible by 8");
_Static_assert(OUTPUT_BUFFER_DATA_BYTES <= 2048u && OUTPUT_BUFFER_DATA_BYTES % 8u == 0u,
    "Output row DMA size must be <= 2048 and divisible by 8");

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
            const uint32_t M = V4_CONTROL.m_elements;
            const uint32_t N = V4_CONTROL.n_elements;
            const uint32_t K = V4_CONTROL.k_elements;
            const int is_int8 = (V4_CONTROL.numeric_mode == EXECUTION_PLAN_V4_NUMERIC_HOST_PACKED_INT8);
            const uint32_t elem_size = is_int8 ? 1u : (uint32_t)sizeof(float);

            for (uint32_t n_start = 0u; n_start < N; n_start += NC) {
                const uint32_t actual_nc = (n_start + NC <= N) ? NC : (N - n_start);

                for (uint32_t k_start = 0u; k_start < K; k_start += KC) {
                    const uint32_t actual_kc = (k_start + KC <= K) ? KC : (K - k_start);

                    if (actual_kc == KC && actual_nc == NC && N == NC &&
                        (V4_CONTROL.b_offset_bytes % 8u == 0u)) {
                        const uint32_t total_b_bytes = KC * NC * elem_size;
                        const uint32_t mram_b_base = V4_CONTROL.b_offset_bytes +
                            (k_start * N + n_start) * elem_size;
                        for (uint32_t offset = tid * B_CONTIGUOUS_CHUNK_BYTES;
                             offset < total_b_bytes;
                             offset += (uint32_t)NR_TASKLETS * B_CONTIGUOUS_CHUNK_BYTES) {
                            mram_read(V4_MRAM + mram_b_base + offset,
                                      shared_b_panel + offset,
                                      B_CONTIGUOUS_CHUNK_BYTES);
                        }
                    } else {
                        for (uint32_t k_idx = tid; k_idx < actual_kc; k_idx += (uint32_t)NR_TASKLETS) {
                            const uint32_t mram_row_offset = V4_CONTROL.b_offset_bytes +
                                ((k_start + k_idx) * N + n_start) * elem_size;
                            const uint32_t row_bytes = actual_nc * elem_size;
                            uint8_t *wram_dst = shared_b_panel + k_idx * (NC * elem_size);
                            if ((mram_row_offset % 8u == 0u) && (row_bytes % 8u == 0u) &&
                                (row_bytes >= 8u) && (row_bytes <= 2048u)) {
                                mram_read(V4_MRAM + mram_row_offset, wram_dst, row_bytes);
                            } else if (row_bytes > 0u) {
                                void *ptr = mram_read_unaligned(V4_MRAM + mram_row_offset,
                                                                tasklet_unaligned_scratch[tid],
                                                                row_bytes);
                                __builtin_memcpy(wram_dst, ptr, row_bytes);
                            }
                        }
                    }

                    barrier_wait(&v4_barrier);

                    for (uint32_t row = tid; row < M; row += (uint32_t)NR_TASKLETS) {
                        v4_output_slot_t *out_row = tasklet_output_buffer[tid];
                        const uint32_t mram_c_offset = V4_CONTROL.c_offset_bytes +
                            (row * N + n_start) * (uint32_t)sizeof(uint32_t);
                        const uint32_t c_row_bytes = actual_nc * (uint32_t)sizeof(uint32_t);

                        if (k_start == 0u) {
                            for (uint32_t c = 0u; c < actual_nc; c++) {
                                out_row[c].u32 = 0u;
                            }
                        } else if ((mram_c_offset % 8u == 0u) && (c_row_bytes % 8u == 0u) &&
                                   (c_row_bytes >= 8u) && (c_row_bytes <= 2048u)) {
                            mram_read(V4_MRAM + mram_c_offset, out_row, c_row_bytes);
                        } else {
                            void *ptr = mram_read_unaligned(V4_MRAM + mram_c_offset,
                                                            tasklet_unaligned_scratch[tid],
                                                            c_row_bytes);
                            __builtin_memcpy(out_row, ptr, c_row_bytes);
                        }

                        const uint32_t mram_a_offset = V4_CONTROL.a_offset_bytes +
                            (row * K + k_start) * elem_size;
                        const uint32_t a_row_bytes = actual_kc * elem_size;
                        uint8_t *wram_a_dst = tasklet_a_buffer[tid];
                        if ((mram_a_offset % 8u == 0u) && (a_row_bytes % 8u == 0u) &&
                            (a_row_bytes >= 8u) && (a_row_bytes <= 2048u)) {
                            mram_read(V4_MRAM + mram_a_offset, wram_a_dst, a_row_bytes);
                        } else if (a_row_bytes > 0u) {
                            void *ptr = mram_read_unaligned(V4_MRAM + mram_a_offset,
                                                            tasklet_unaligned_scratch[tid],
                                                            a_row_bytes);
                            __builtin_memcpy(wram_a_dst, ptr, a_row_bytes);
                        }

                        if (is_int8) {
                            const int8_t *a_row = (const int8_t *)tasklet_a_buffer[tid];
                            const int8_t *b_panel = (const int8_t *)shared_b_panel;
                            for (uint32_t k = 0u; k < actual_kc; k++) {
                                const int32_t a_val = (int32_t)a_row[k];
                                const int8_t *b_row = b_panel + k * NC;
                                for (uint32_t c = 0u; c < actual_nc; c++) {
                                    out_row[c].i32 += a_val * (int32_t)b_row[c];
                                }
                            }
                        } else {
                            const float *a_row = (const float *)tasklet_a_buffer[tid];
                            const float *b_panel = (const float *)shared_b_panel;
                            for (uint32_t k = 0u; k < actual_kc; k++) {
                                const float a_val = a_row[k];
                                const float *b_row = b_panel + k * NC;
                                for (uint32_t c = 0u; c < actual_nc; c++) {
                                    out_row[c].f32 += a_val * b_row[c];
                                }
                            }
                        }

                        if ((mram_c_offset % 8u == 0u) && (c_row_bytes % 8u == 0u) &&
                            (c_row_bytes >= 8u) && (c_row_bytes <= 2048u)) {
                            mram_write(out_row, V4_MRAM + mram_c_offset, c_row_bytes);
                        } else if (c_row_bytes > 0u) {
                            const uint32_t dst_align = mram_c_offset & 7u;
                            uint8_t *unaligned_src = tasklet_unaligned_scratch[tid] + dst_align;
                            __builtin_memcpy(unaligned_src, out_row, c_row_bytes);
                            mram_write_unaligned(
                                unaligned_src, V4_MRAM + mram_c_offset, c_row_bytes);
                        }
                    }

                    barrier_wait(&v4_barrier);
                }
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
