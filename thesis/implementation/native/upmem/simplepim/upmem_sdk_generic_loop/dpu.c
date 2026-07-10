#include <defs.h>
#include <mram.h>
#include <stdint.h>

#include "common.h"

__host upmem_generic_args_t GENERIC_ARGS;

/* The host transfers only the mode-sized prefix of each full MRAM region. */
__mram_noinit uint8_t GENERIC_A_RAW[UPMEM_GENERIC_MAX_ELEMS * sizeof(uint32_t)];
__mram_noinit uint8_t GENERIC_B_RAW[UPMEM_GENERIC_MAX_ELEMS * sizeof(uint32_t)];
__mram_noinit uint8_t GENERIC_C_RAW[UPMEM_GENERIC_MAX_ELEMS * sizeof(uint32_t)];

typedef union {
    int32_t i32[UPMEM_GENERIC_OUTPUT_TILE_ELEMS];
    float f32[UPMEM_GENERIC_OUTPUT_TILE_ELEMS];
} upmem_generic_output_tile_t;

__dma_aligned uint8_t input_window[8];
__dma_aligned upmem_generic_output_tile_t output_tile;

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

static int8_t read_i8(const __mram_ptr uint8_t *base, uint32_t element) {
    const uint32_t byte_offset = element;
    const uint32_t aligned_offset = byte_offset & ~7u;
    mram_read(base + aligned_offset, input_window, sizeof(input_window));
    return (int8_t)input_window[byte_offset - aligned_offset];
}

static float read_f32(const __mram_ptr uint8_t *base, uint32_t element) {
    const uint32_t byte_offset = element * (uint32_t)sizeof(float);
    const uint32_t aligned_offset = byte_offset & ~7u;
    float value;
    mram_read(base + aligned_offset, input_window, sizeof(input_window));
    __builtin_memcpy(&value, &input_window[byte_offset - aligned_offset], sizeof(value));
    return value;
}

int main(void) {
    const upmem_generic_args_t args = GENERIC_ARGS;
    uint32_t out_coords[UPMEM_GENERIC_MAX_RANK] = {0};
    const int float32_mode = args.operand_mode == UPMEM_GENERIC_MODE_FLOAT32_NO_QUANT;

    for (uint32_t tile_start = 0; tile_start < args.output_elems; tile_start += UPMEM_GENERIC_OUTPUT_TILE_ELEMS) {
        const uint32_t tile_elems = (args.output_elems - tile_start) < UPMEM_GENERIC_OUTPUT_TILE_ELEMS
            ? args.output_elems - tile_start
            : UPMEM_GENERIC_OUTPUT_TILE_ELEMS;

        for (uint32_t tile_index = 0; tile_index < tile_elems; tile_index++) {
            const uint32_t output_linear = tile_start + tile_index;
            decode_index(output_linear, args.output_rank, args.output_shape, args.output_strides, out_coords);

            uint32_t left_base = 0;
            uint32_t right_base = 0;
            for (uint32_t output_axis = 0; output_axis < args.output_rank; output_axis++) {
                const int32_t left_axis = args.output_to_left_axes[output_axis];
                const int32_t right_axis = args.output_to_right_axes[output_axis];
                if (left_axis >= 0) {
                    left_base += out_coords[output_axis] * args.left_strides[(uint32_t)left_axis];
                }
                if (right_axis >= 0) {
                    right_base += out_coords[output_axis] * args.right_strides[(uint32_t)right_axis];
                }
            }

            int32_t total = 0;
            float total_f32 = 0.0f;
            for (uint32_t contracted_linear = 0; contracted_linear < args.contracted_elems; contracted_linear++) {
                uint32_t left_offset = left_base;
                uint32_t right_offset = right_base;
                uint32_t remaining = contracted_linear;
                for (uint32_t contracted_axis = 0; contracted_axis < args.contracted_rank; contracted_axis++) {
                    uint32_t stride = 1;
                    for (uint32_t later = contracted_axis + 1; later < args.contracted_rank; later++) {
                        stride *= args.contracted_dims[later];
                    }
                    const uint32_t coord = stride == 0 ? 0 : remaining / stride;
                    remaining -= coord * stride;
                    left_offset += coord * args.left_strides[(uint32_t)args.contracted_to_left_axes[contracted_axis]];
                    right_offset += coord * args.right_strides[(uint32_t)args.contracted_to_right_axes[contracted_axis]];
                }

                if (float32_mode) {
                    total_f32 += read_f32(GENERIC_A_RAW, left_offset) * read_f32(GENERIC_B_RAW, right_offset);
                } else {
                    total += (int32_t)read_i8(GENERIC_A_RAW, left_offset) * (int32_t)read_i8(GENERIC_B_RAW, right_offset);
                }
            }
            if (float32_mode) {
                output_tile.f32[tile_index] = total_f32;
            } else {
                output_tile.i32[tile_index] = total;
            }
        }

        if ((tile_elems & 1u) != 0u) {
            output_tile.i32[tile_elems] = 0;
        }
        const uint32_t tile_bytes = align8(tile_elems * (uint32_t)sizeof(uint32_t));
        if (float32_mode) {
            mram_write(output_tile.f32, GENERIC_C_RAW + tile_start * sizeof(float), tile_bytes);
        } else {
            mram_write(output_tile.i32, GENERIC_C_RAW + tile_start * sizeof(int32_t), tile_bytes);
        }
    }
    return 0;
}
