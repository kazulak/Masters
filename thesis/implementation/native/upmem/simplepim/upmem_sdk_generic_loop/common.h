#ifndef UPMEM_SDK_GENERIC_LOOP_COMMON_H
#define UPMEM_SDK_GENERIC_LOOP_COMMON_H

#include <stdint.h>

#ifndef UPMEM_GENERIC_MAX_RANK
#define UPMEM_GENERIC_MAX_RANK 7
#endif

#ifndef UPMEM_GENERIC_MAX_ELEMS
#define UPMEM_GENERIC_MAX_ELEMS 4096
#endif

#define UPMEM_GENERIC_MODE_INT8_SCALED 0u
#define UPMEM_GENERIC_MODE_FLOAT32_NO_QUANT 1u

typedef struct {
    uint32_t left_rank;
    uint32_t right_rank;
    uint32_t output_rank;
    uint32_t contracted_rank;
    uint32_t left_elems;
    uint32_t right_elems;
    uint32_t output_elems;
    uint32_t contracted_elems;
    uint32_t operand_mode;
    uint32_t left_shape[UPMEM_GENERIC_MAX_RANK];
    uint32_t right_shape[UPMEM_GENERIC_MAX_RANK];
    uint32_t output_shape[UPMEM_GENERIC_MAX_RANK];
    uint32_t contracted_dims[UPMEM_GENERIC_MAX_RANK];
    uint32_t left_strides[UPMEM_GENERIC_MAX_RANK];
    uint32_t right_strides[UPMEM_GENERIC_MAX_RANK];
    uint32_t output_strides[UPMEM_GENERIC_MAX_RANK];
    int32_t output_to_left_axes[UPMEM_GENERIC_MAX_RANK];
    int32_t output_to_right_axes[UPMEM_GENERIC_MAX_RANK];
    int32_t contracted_to_left_axes[UPMEM_GENERIC_MAX_RANK];
    int32_t contracted_to_right_axes[UPMEM_GENERIC_MAX_RANK];
} upmem_generic_args_t;

#endif
