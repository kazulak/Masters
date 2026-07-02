#include <defs.h>
#include <mram.h>
#include <stdint.h>

#include "common.h"

__host upmem_generic_args_t GENERIC_ARGS;

__mram_noinit uint32_t GENERIC_A_RAW[UPMEM_GENERIC_MAX_ELEMS];
__mram_noinit uint32_t GENERIC_B_RAW[UPMEM_GENERIC_MAX_ELEMS];
__mram_noinit uint32_t GENERIC_C_RAW[UPMEM_GENERIC_MAX_ELEMS];

typedef union {
    int8_t i8[UPMEM_GENERIC_MAX_ELEMS];
    int32_t i32[UPMEM_GENERIC_MAX_ELEMS];
    float f32[UPMEM_GENERIC_MAX_ELEMS];
} upmem_generic_local_buffer_t;

__dma_aligned upmem_generic_local_buffer_t local_a;
__dma_aligned upmem_generic_local_buffer_t local_b;
__dma_aligned upmem_generic_local_buffer_t local_c;

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

    if (args.operand_mode == UPMEM_GENERIC_MODE_FLOAT32_NO_QUANT) {
        mram_read(GENERIC_A_RAW, local_a.f32, align8(args.left_elems * sizeof(float)));
        mram_read(GENERIC_B_RAW, local_b.f32, align8(args.right_elems * sizeof(float)));
    } else {
        mram_read(GENERIC_A_RAW, local_a.i8, align8(args.left_elems * sizeof(int8_t)));
        mram_read(GENERIC_B_RAW, local_b.i8, align8(args.right_elems * sizeof(int8_t)));
    }

    for (uint32_t output_linear = 0; output_linear < args.output_elems; output_linear++) {
        decode_index(output_linear, args.output_rank, args.output_shape, args.output_strides, out_coords);
        int32_t total = 0;
        float total_f32 = 0.0f;

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

            if (args.operand_mode == UPMEM_GENERIC_MODE_FLOAT32_NO_QUANT) {
                total_f32 += local_a.f32[left_offset] * local_b.f32[right_offset];
            } else {
                total += (int32_t)local_a.i8[left_offset] * (int32_t)local_b.i8[right_offset];
            }
        }
        if (args.operand_mode == UPMEM_GENERIC_MODE_FLOAT32_NO_QUANT) {
            local_c.f32[output_linear] = total_f32;
        } else {
            local_c.i32[output_linear] = total;
        }
    }

    if (args.operand_mode == UPMEM_GENERIC_MODE_FLOAT32_NO_QUANT) {
        mram_write(local_c.f32, GENERIC_C_RAW, align8(args.output_elems * sizeof(float)));
    } else {
        mram_write(local_c.i32, GENERIC_C_RAW, align8(args.output_elems * sizeof(int32_t)));
    }
    return 0;
}
