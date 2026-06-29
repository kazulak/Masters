#include <defs.h>
#include <mram.h>
#include <stdint.h>

#include "common.h"

__host upmem_generic_args_t GENERIC_ARGS;

__mram_noinit int8_t GENERIC_A[UPMEM_GENERIC_MAX_ELEMS];
__mram_noinit int8_t GENERIC_B[UPMEM_GENERIC_MAX_ELEMS];
__mram_noinit int32_t GENERIC_C[UPMEM_GENERIC_MAX_ELEMS];

__dma_aligned int8_t local_a[UPMEM_GENERIC_MAX_ELEMS];
__dma_aligned int8_t local_b[UPMEM_GENERIC_MAX_ELEMS];
__dma_aligned int32_t local_c[UPMEM_GENERIC_MAX_ELEMS];

static void decode_index(uint32_t linear, uint32_t rank, const uint32_t *shape, const uint32_t *strides, uint32_t *coords) {
    for (uint32_t axis = 0; axis < rank; axis++) {
        const uint32_t stride = strides[axis];
        coords[axis] = stride == 0 ? 0 : linear / stride;
        linear -= coords[axis] * stride;
        if (shape[axis] == 0) {
            coords[axis] = 0;
        }
    }
}

static uint32_t align8(uint32_t bytes) {
    return (bytes + 7u) & ~7u;
}

int main(void) {
    const upmem_generic_args_t args = GENERIC_ARGS;
    uint32_t out_coords[UPMEM_GENERIC_MAX_RANK] = {0};

    mram_read(GENERIC_A, local_a, align8(args.left_elems * sizeof(int8_t)));
    mram_read(GENERIC_B, local_b, align8(args.right_elems * sizeof(int8_t)));

    for (uint32_t output_linear = 0; output_linear < args.output_elems; output_linear++) {
        decode_index(output_linear, args.output_rank, args.output_shape, args.output_strides, out_coords);
        int32_t total = 0;

        for (uint32_t contracted_linear = 0; contracted_linear < args.contracted_elems; contracted_linear++) {
            uint32_t left_offset = 0;
            uint32_t right_offset = 0;

            for (uint32_t output_axis = 0; output_axis < args.output_rank; output_axis++) {
                const int32_t left_axis = args.output_to_left_axes[output_axis];
                const int32_t right_axis = args.output_to_right_axes[output_axis];
                if (left_axis >= 0) {
                    left_offset += out_coords[output_axis] * args.left_strides[(uint32_t)left_axis];
                }
                if (right_axis >= 0) {
                    right_offset += out_coords[output_axis] * args.right_strides[(uint32_t)right_axis];
                }
            }

            uint32_t remaining = contracted_linear;
            for (uint32_t contracted_axis = 0; contracted_axis < args.contracted_rank; contracted_axis++) {
                uint32_t stride = 1;
                for (uint32_t later = contracted_axis + 1; later < args.contracted_rank; later++) {
                    stride *= args.contracted_dims[later];
                }
                const uint32_t coord = stride == 0 ? 0 : remaining / stride;
                remaining -= coord * stride;
                const int32_t left_axis = args.contracted_to_left_axes[contracted_axis];
                const int32_t right_axis = args.contracted_to_right_axes[contracted_axis];
                left_offset += coord * args.left_strides[(uint32_t)left_axis];
                right_offset += coord * args.right_strides[(uint32_t)right_axis];
            }

            total += (int32_t)local_a[left_offset] * (int32_t)local_b[right_offset];
        }
        local_c[output_linear] = total;
    }

    mram_write(local_c, GENERIC_C, align8(args.output_elems * sizeof(int32_t)));
    return 0;
}
