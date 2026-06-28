#include <defs.h>
#include <mram.h>
#include <stdint.h>

#include "common.h"

__host upmem_dense_args_t DENSE_ARGS;

__mram_noinit int8_t DENSE_A[UPMEM_DENSE_L2_MAX_A_ELEMS];
__mram_noinit int8_t DENSE_B[UPMEM_DENSE_L2_MAX_B_ELEMS];
__mram_noinit int32_t DENSE_C[UPMEM_DENSE_L2_MAX_C_ELEMS];

__dma_aligned int8_t local_l1_a[UPMEM_DENSE_MAX_ELEMS];
__dma_aligned int8_t local_l1_b[UPMEM_DENSE_MAX_ELEMS];
__dma_aligned int32_t local_l1_c[UPMEM_DENSE_MAX_ELEMS];

__dma_aligned int8_t local_tile_a[UPMEM_DENSE_L2_TILE_MAX_DIM * UPMEM_DENSE_L2_TILE_MAX_DIM];
__dma_aligned int8_t local_tile_b[UPMEM_DENSE_L2_TILE_MAX_DIM * UPMEM_DENSE_L2_TILE_MAX_DIM];
__dma_aligned int32_t local_tile_acc[UPMEM_DENSE_L2_TILE_MAX_DIM * UPMEM_DENSE_L2_TILE_MAX_DIM];

static uint32_t min_u32(uint32_t a, uint32_t b) {
    return a < b ? a : b;
}

static void run_l1_direct(void) {
    const uint32_t m = DENSE_ARGS.m;
    const uint32_t k = DENSE_ARGS.k;
    const uint32_t n = DENSE_ARGS.n;
    const uint32_t a_stride = DENSE_ARGS.a_stride;
    const uint32_t b_stride = DENSE_ARGS.b_stride;
    const uint32_t c_stride = DENSE_ARGS.c_stride;

    mram_read(DENSE_A, local_l1_a, sizeof(local_l1_a));
    mram_read(DENSE_B, local_l1_b, sizeof(local_l1_b));

    for (uint32_t i = 0; i < m; i++) {
        for (uint32_t j = 0; j < n; j++) {
            int32_t sum = 0;
            for (uint32_t p = 0; p < k; p++) {
                const int32_t a = (int32_t)local_l1_a[i * a_stride + p];
                const int32_t b = (int32_t)local_l1_b[p * b_stride + j];
                sum += a * b;
            }
            local_l1_c[i * c_stride + j] = sum;
        }
    }

    mram_write(local_l1_c, DENSE_C, sizeof(local_l1_c));
}

static void run_l2_tiled(void) {
    const uint32_t m = DENSE_ARGS.m;
    const uint32_t k = DENSE_ARGS.k;
    const uint32_t n = DENSE_ARGS.n;
    const uint32_t tile_m = DENSE_ARGS.tile_m;
    const uint32_t tile_k = DENSE_ARGS.tile_k;
    const uint32_t tile_n = DENSE_ARGS.tile_n;

    for (uint32_t row0 = 0; row0 < m; row0 += tile_m) {
        const uint32_t rows = min_u32(tile_m, m - row0);
        for (uint32_t col0 = 0; col0 < n; col0 += tile_n) {
            const uint32_t cols = min_u32(tile_n, n - col0);

            for (uint32_t i = 0; i < rows; i++) {
                for (uint32_t j = 0; j < cols; j++) {
                    local_tile_acc[i * tile_n + j] = 0;
                }
            }

            for (uint32_t k0 = 0; k0 < k; k0 += tile_k) {
                const uint32_t depth = min_u32(tile_k, k - k0);

                for (uint32_t i = 0; i < rows; i++) {
                    mram_read(
                        &DENSE_A[(row0 + i) * k + k0],
                        &local_tile_a[i * tile_k],
                        depth * sizeof(int8_t)
                    );
                }
                for (uint32_t p = 0; p < depth; p++) {
                    mram_read(
                        &DENSE_B[(k0 + p) * n + col0],
                        &local_tile_b[p * tile_n],
                        cols * sizeof(int8_t)
                    );
                }

                for (uint32_t i = 0; i < rows; i++) {
                    for (uint32_t j = 0; j < cols; j++) {
                        int32_t sum = local_tile_acc[i * tile_n + j];
                        for (uint32_t p = 0; p < depth; p++) {
                            const int32_t a = (int32_t)local_tile_a[i * tile_k + p];
                            const int32_t b = (int32_t)local_tile_b[p * tile_n + j];
                            sum += a * b;
                        }
                        local_tile_acc[i * tile_n + j] = sum;
                    }
                }
            }

            for (uint32_t i = 0; i < rows; i++) {
                mram_write(
                    &local_tile_acc[i * tile_n],
                    &DENSE_C[(row0 + i) * n + col0],
                    cols * sizeof(int32_t)
                );
            }
        }
    }
}

int main(void) {
    if (DENSE_ARGS.strategy == UPMEM_DENSE_STRATEGY_L2) {
        run_l2_tiled();
    } else {
        run_l1_direct();
    }
    return 0;
}
