#ifndef UPMEM_SDK_DENSE_COMMON_H
#define UPMEM_SDK_DENSE_COMMON_H

#include <stdint.h>

#ifndef UPMEM_DENSE_MAX_DIM
#define UPMEM_DENSE_MAX_DIM 16
#endif

#ifndef UPMEM_DENSE_L2_MAX_DIM
#define UPMEM_DENSE_L2_MAX_DIM 512
#endif

#ifndef UPMEM_DENSE_L2_TILE_MAX_DIM
#define UPMEM_DENSE_L2_TILE_MAX_DIM 64
#endif

#define UPMEM_DENSE_STRATEGY_L1 1
#define UPMEM_DENSE_STRATEGY_L2 2
#define UPMEM_DENSE_MAX_ELEMS (UPMEM_DENSE_MAX_DIM * UPMEM_DENSE_MAX_DIM)
#define UPMEM_DENSE_L2_MAX_A_ELEMS (UPMEM_DENSE_L2_MAX_DIM * UPMEM_DENSE_L2_MAX_DIM)
#define UPMEM_DENSE_L2_MAX_B_ELEMS (UPMEM_DENSE_L2_MAX_DIM * UPMEM_DENSE_L2_MAX_DIM)
#define UPMEM_DENSE_L2_MAX_C_ELEMS (UPMEM_DENSE_L2_MAX_DIM * UPMEM_DENSE_L2_MAX_DIM)
#define UPMEM_DENSE_L2_TILE_MAX_ELEMS (UPMEM_DENSE_L2_TILE_MAX_DIM * UPMEM_DENSE_L2_TILE_MAX_DIM)

typedef struct {
    uint32_t strategy;
    uint32_t m;
    uint32_t k;
    uint32_t n;
    uint32_t a_stride;
    uint32_t b_stride;
    uint32_t c_stride;
    uint32_t tile_m;
    uint32_t tile_k;
    uint32_t tile_n;
} upmem_dense_args_t;

#endif
