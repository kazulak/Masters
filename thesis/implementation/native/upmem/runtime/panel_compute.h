#ifndef UPMEM_PANEL_COMPUTE_H
#define UPMEM_PANEL_COMPUTE_H

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

/* All tasklets call this with the same geometry; each owns cyclic output rows. */
static void panel_compute(__mram_ptr uint8_t *arena, uint32_t M, uint32_t N,
        uint32_t K, int is_int8, uint32_t a_offset, uint32_t b_offset,
        uint32_t c_offset) {
    const uint32_t tid = me();
    const uint32_t elem_size = is_int8 ? 1u : (uint32_t)sizeof(float);

    for (uint32_t n_start = 0u; n_start < N; n_start += NC) {
        const uint32_t actual_nc = (n_start + NC <= N) ? NC : (N - n_start);

        for (uint32_t k_start = 0u; k_start < K; k_start += KC) {
            const uint32_t actual_kc = (k_start + KC <= K) ? KC : (K - k_start);

            if (actual_kc == KC && actual_nc == NC && N == NC &&
                (b_offset % 8u == 0u)) {
                const uint32_t total_b_bytes = KC * NC * elem_size;
                const uint32_t mram_b_base = b_offset +
                    (k_start * N + n_start) * elem_size;
                for (uint32_t offset = tid * B_CONTIGUOUS_CHUNK_BYTES;
                     offset < total_b_bytes;
                     offset += (uint32_t)NR_TASKLETS * B_CONTIGUOUS_CHUNK_BYTES) {
                    mram_read(arena + mram_b_base + offset,
                              shared_b_panel + offset,
                              B_CONTIGUOUS_CHUNK_BYTES);
                }
            } else {
                for (uint32_t k_idx = tid; k_idx < actual_kc; k_idx += (uint32_t)NR_TASKLETS) {
                    const uint32_t mram_row_offset = b_offset +
                        ((k_start + k_idx) * N + n_start) * elem_size;
                    const uint32_t row_bytes = actual_nc * elem_size;
                    uint8_t *wram_dst = shared_b_panel + k_idx * (NC * elem_size);
                    if ((mram_row_offset % 8u == 0u) && (row_bytes % 8u == 0u) &&
                        (row_bytes >= 8u) && (row_bytes <= 2048u)) {
                        mram_read(arena + mram_row_offset, wram_dst, row_bytes);
                    } else if (row_bytes > 0u) {
                        void *ptr = mram_read_unaligned(arena + mram_row_offset,
                                                        tasklet_unaligned_scratch[tid],
                                                        row_bytes);
                        __builtin_memcpy(wram_dst, ptr, row_bytes);
                    }
                }
            }

            barrier_wait(&v4_barrier);

            for (uint32_t row = tid; row < M; row += (uint32_t)NR_TASKLETS) {
                v4_output_slot_t *out_row = tasklet_output_buffer[tid];
                const uint32_t mram_c_offset = c_offset +
                    (row * N + n_start) * (uint32_t)sizeof(uint32_t);
                const uint32_t c_row_bytes = actual_nc * (uint32_t)sizeof(uint32_t);

                if (k_start == 0u) {
                    for (uint32_t c = 0u; c < actual_nc; c++) {
                        out_row[c].u32 = 0u;
                    }
                } else if ((mram_c_offset % 8u == 0u) && (c_row_bytes % 8u == 0u) &&
                           (c_row_bytes >= 8u) && (c_row_bytes <= 2048u)) {
                    mram_read(arena + mram_c_offset, out_row, c_row_bytes);
                } else {
                    void *ptr = mram_read_unaligned(arena + mram_c_offset,
                                                    tasklet_unaligned_scratch[tid],
                                                    c_row_bytes);
                    __builtin_memcpy(out_row, ptr, c_row_bytes);
                }

                const uint32_t mram_a_offset = a_offset +
                    (row * K + k_start) * elem_size;
                const uint32_t a_row_bytes = actual_kc * elem_size;
                uint8_t *wram_a_dst = tasklet_a_buffer[tid];
                if ((mram_a_offset % 8u == 0u) && (a_row_bytes % 8u == 0u) &&
                    (a_row_bytes >= 8u) && (a_row_bytes <= 2048u)) {
                    mram_read(arena + mram_a_offset, wram_a_dst, a_row_bytes);
                } else if (a_row_bytes > 0u) {
                    void *ptr = mram_read_unaligned(arena + mram_a_offset,
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
                    mram_write(out_row, arena + mram_c_offset, c_row_bytes);
                } else if (c_row_bytes > 0u) {
                    const uint32_t dst_align = mram_c_offset & 7u;
                    uint8_t *unaligned_src = tasklet_unaligned_scratch[tid] + dst_align;
                    __builtin_memcpy(unaligned_src, out_row, c_row_bytes);
                    mram_write_unaligned(
                        unaligned_src, arena + mram_c_offset, c_row_bytes);
                }
            }

            barrier_wait(&v4_barrier);
        }
    }
}

#endif

