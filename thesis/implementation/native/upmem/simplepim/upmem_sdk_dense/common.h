#ifndef UPMEM_SDK_DENSE_COMMON_H
#define UPMEM_SDK_DENSE_COMMON_H

#include <stdint.h>

#ifndef UPMEM_DENSE_MAX_DIM
#define UPMEM_DENSE_MAX_DIM 16
#endif

#define UPMEM_DENSE_MAX_ELEMS (UPMEM_DENSE_MAX_DIM * UPMEM_DENSE_MAX_DIM)

typedef struct {
    uint32_t m;
    uint32_t k;
    uint32_t n;
    uint32_t a_stride;
    uint32_t b_stride;
    uint32_t c_stride;
} upmem_dense_args_t;

#endif
