#include <defs.h>
#include <mram.h>
#include <stdint.h>

#include "common.h"

__host upmem_dense_args_t DENSE_ARGS;

__mram_noinit int8_t DENSE_A[UPMEM_DENSE_MAX_ELEMS];
__mram_noinit int8_t DENSE_B[UPMEM_DENSE_MAX_ELEMS];
__mram_noinit int32_t DENSE_C[UPMEM_DENSE_MAX_ELEMS];

__dma_aligned int8_t local_a[UPMEM_DENSE_MAX_ELEMS];
__dma_aligned int8_t local_b[UPMEM_DENSE_MAX_ELEMS];
__dma_aligned int32_t local_c[UPMEM_DENSE_MAX_ELEMS];

int main(void) {
    const uint32_t m = DENSE_ARGS.m;
    const uint32_t k = DENSE_ARGS.k;
    const uint32_t n = DENSE_ARGS.n;
    const uint32_t a_stride = DENSE_ARGS.a_stride;
    const uint32_t b_stride = DENSE_ARGS.b_stride;
    const uint32_t c_stride = DENSE_ARGS.c_stride;

    mram_read(DENSE_A, local_a, sizeof(local_a));
    mram_read(DENSE_B, local_b, sizeof(local_b));

    for (uint32_t i = 0; i < m; i++) {
        for (uint32_t j = 0; j < n; j++) {
            int32_t sum = 0;
            for (uint32_t p = 0; p < k; p++) {
                const int32_t a = (int32_t)local_a[i * a_stride + p];
                const int32_t b = (int32_t)local_b[p * b_stride + j];
                sum += a * b;
            }
            local_c[i * c_stride + j] = sum;
        }
    }

    mram_write(local_c, DENSE_C, sizeof(local_c));
    return 0;
}
