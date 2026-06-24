#ifndef PARAM_H
#define PARAM_H

#include <stdint.h>

#define WRAM_SAFE_BYTES       (48 * 1024)

#define TILE_ROWS   16
#define TILE_K     256
#define TILE_N      64

#ifdef NR_TASKLETS
#undef NR_TASKLETS
#endif
#define NR_TASKLETS   8

#define INT8_SCALE_MAX  127

#define GEMM_TILE_INPUT_BYTES \
    (sizeof(int32_t) * 3 + (TILE_ROWS * TILE_K) + (TILE_K * TILE_N))
#define GEMM_TILE_OUTPUT_BYTES \
    (sizeof(int32_t) * TILE_ROWS * TILE_N)

typedef struct {
    int32_t tile_rows;
    int32_t k;
    int32_t tile_cols;
    int8_t a[TILE_ROWS * TILE_K];
    int8_t b[TILE_K * TILE_N];
} GemmTileInput;

typedef struct {
    int32_t c[TILE_ROWS * TILE_N];
} GemmTileOutput;

#if (TILE_ROWS * TILE_K + TILE_K * TILE_N + TILE_ROWS * TILE_N * 4) > WRAM_SAFE_BYTES
#error "Configured GEMM tile exceeds the safe UPMEM WRAM data budget"
#endif

#endif /* PARAM_H */
