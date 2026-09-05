#ifndef UPMEM_OUTER_COMPUTE_H
#define UPMEM_OUTER_COMPUTE_H

#define OUTER_MAX_ELEMENTS 256u
#define OUTER_A_PANEL_OFFSET_BYTES 0u
#define OUTER_B_PANEL_OFFSET_BYTES 1024u
#define OUTER_DMA_CHUNK_BYTES 256u
#define OUTER_MAX_FLOAT_BYTES (OUTER_MAX_ELEMENTS * sizeof(float))

_Static_assert(NC == 32u, "outer kernel requires NC32 output blocks");
_Static_assert(OUTER_B_PANEL_OFFSET_BYTES + OUTER_MAX_FLOAT_BYTES <=
    sizeof(shared_b_panel), "outer input staging exceeds shared B panel");
_Static_assert(OUTER_DMA_CHUNK_BYTES <= 256u &&
    OUTER_DMA_CHUNK_BYTES % 8u == 0u,
    "outer input DMA chunks must be aligned and <= 256 bytes");
_Static_assert(sizeof(tasklet_unaligned_scratch[0]) >= 8u,
    "outer tail write needs unaligned scratch");

static inline void outer_write_block(__mram_ptr uint8_t *arena,
        v4_output_slot_t *out_block, uint32_t block_count,
        uint32_t mram_c_offset, uint32_t tid) {
    const uint32_t block_bytes = block_count * (uint32_t)sizeof(uint32_t);
    const uint32_t aligned_bytes = block_bytes & ~7u;
    if (aligned_bytes != 0u) {
        mram_write(out_block, arena + mram_c_offset, aligned_bytes);
    }
    if (aligned_bytes != block_bytes) {
        uint8_t *unaligned_src = tasklet_unaligned_scratch[tid] +
            (mram_c_offset & 7u);
        __builtin_memcpy(unaligned_src,
            ((uint8_t *)out_block) + aligned_bytes,
            block_bytes - aligned_bytes);
        mram_write_unaligned(unaligned_src,
            arena + mram_c_offset + aligned_bytes,
            block_bytes - aligned_bytes);
    }
}

/* All tasklets call this helper for a single M-by-N outer product. */
static void outer_compute(__mram_ptr uint8_t *arena, uint32_t M, uint32_t N,
        int is_int8, uint32_t a_offset, uint32_t b_offset, uint32_t c_offset) {
    const uint32_t tid = me();
    const uint32_t elem_size = is_int8 ? 1u : (uint32_t)sizeof(float);
    const uint32_t a_bytes = (M * elem_size + 7u) & ~7u;
    const uint32_t b_bytes = (N * elem_size + 7u) & ~7u;

    for (uint32_t source_offset = tid * OUTER_DMA_CHUNK_BYTES;
         source_offset < a_bytes;
         source_offset += (uint32_t)NR_TASKLETS * OUTER_DMA_CHUNK_BYTES) {
        uint32_t chunk_bytes = a_bytes - source_offset;
        if (chunk_bytes > OUTER_DMA_CHUNK_BYTES) chunk_bytes = OUTER_DMA_CHUNK_BYTES;
        mram_read(arena + a_offset + source_offset,
                  shared_b_panel + OUTER_A_PANEL_OFFSET_BYTES + source_offset,
                  chunk_bytes);
    }
    for (uint32_t source_offset = tid * OUTER_DMA_CHUNK_BYTES;
         source_offset < b_bytes;
         source_offset += (uint32_t)NR_TASKLETS * OUTER_DMA_CHUNK_BYTES) {
        uint32_t chunk_bytes = b_bytes - source_offset;
        if (chunk_bytes > OUTER_DMA_CHUNK_BYTES) chunk_bytes = OUTER_DMA_CHUNK_BYTES;
        mram_read(arena + b_offset + source_offset,
                  shared_b_panel + OUTER_B_PANEL_OFFSET_BYTES + source_offset,
                  chunk_bytes);
    }

    barrier_wait(&v4_barrier);

    const uint32_t output_count = M * N;
    v4_output_slot_t *out_block = tasklet_output_buffer[tid];
    const int8_t *a_i8 = (const int8_t *)
        (shared_b_panel + OUTER_A_PANEL_OFFSET_BYTES);
    const int8_t *b_i8 = (const int8_t *)
        (shared_b_panel + OUTER_B_PANEL_OFFSET_BYTES);
    const float *a_f32 = (const float *)
        (shared_b_panel + OUTER_A_PANEL_OFFSET_BYTES);
    const float *b_f32 = (const float *)
        (shared_b_panel + OUTER_B_PANEL_OFFSET_BYTES);
    for (uint32_t block_start = tid * NC;
         block_start < output_count;
         block_start += (uint32_t)NR_TASKLETS * NC) {
        uint32_t block_count = output_count - block_start;
        if (block_count > NC) block_count = NC;
        const uint32_t first_row = block_start / N;
        uint32_t row = first_row;
        uint32_t col = block_start - first_row * N;
        if (is_int8) {
            for (uint32_t local = 0u; local < block_count; ++local) {
                out_block[local].i32 = 0;
                out_block[local].i32 +=
                    (int32_t)a_i8[row] * (int32_t)b_i8[col];
                if (++col == N) {
                    col = 0u;
                    ++row;
                }
            }
        } else {
            for (uint32_t local = 0u; local < block_count; ++local) {
                out_block[local].f32 = 0.0f;
                out_block[local].f32 += a_f32[row] * b_f32[col];
                if (++col == N) {
                    col = 0u;
                    ++row;
                }
            }
        }
        outer_write_block(arena, out_block, block_count,
            c_offset + block_start * (uint32_t)sizeof(uint32_t), tid);
    }

    barrier_wait(&v4_barrier);
}

#endif
