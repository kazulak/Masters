#include <dpu.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "common.h"

static int read_exact(const char *path, void *buffer, size_t bytes) {
    FILE *file = fopen(path, "rb");
    if (file == NULL) {
        perror(path);
        return 1;
    }
    size_t count = fread(buffer, 1, bytes, file);
    fclose(file);
    if (count != bytes) {
        fprintf(stderr, "short read from %s: expected %zu bytes, got %zu\n", path, bytes, count);
        return 1;
    }
    return 0;
}

static int write_exact(const char *path, const void *buffer, size_t bytes) {
    FILE *file = fopen(path, "wb");
    if (file == NULL) {
        perror(path);
        return 1;
    }
    size_t count = fwrite(buffer, 1, bytes, file);
    fclose(file);
    if (count != bytes) {
        fprintf(stderr, "short write to %s: expected %zu bytes, got %zu\n", path, bytes, count);
        return 1;
    }
    return 0;
}

static size_t align8(size_t bytes) {
    return (bytes + 7u) & ~((size_t)7u);
}

static int transfer_sizes(uint32_t elements, size_t element_size, size_t *bytes, size_t *transfer_bytes) {
    if ((size_t)elements > SIZE_MAX / element_size) {
        return 1;
    }
    *bytes = (size_t)elements * element_size;
    if (*bytes > SIZE_MAX - 7u) {
        return 1;
    }
    *transfer_bytes = align8(*bytes);
    return *transfer_bytes == 0 || (*transfer_bytes % 8u) != 0;
}

static int validate_row_major(const uint32_t *shape, const uint32_t *strides, uint32_t rank, uint32_t expected_elements) {
    uint64_t product = 1;
    uint64_t expected_stride = 1;
    for (uint32_t reverse_axis = 0; reverse_axis < rank; reverse_axis++) {
        const uint32_t axis = rank - reverse_axis - 1u;
        if (shape[axis] == 0 || strides[axis] != expected_stride) {
            return 1;
        }
        product *= shape[axis];
        expected_stride *= shape[axis];
        if (product > UINT32_MAX || expected_stride > UINT32_MAX) {
            return 1;
        }
    }
    return product != expected_elements;
}

static int validate_index_maps(const upmem_generic_args_t *args) {
    uint8_t left_used[UPMEM_GENERIC_MAX_RANK] = {0};
    uint8_t right_used[UPMEM_GENERIC_MAX_RANK] = {0};

    for (uint32_t output_axis = 0; output_axis < args->output_rank; output_axis++) {
        const int32_t left_axis = args->output_to_left_axes[output_axis];
        const int32_t right_axis = args->output_to_right_axes[output_axis];
        if (left_axis < -1 || right_axis < -1 ||
            left_axis >= (int32_t)args->left_rank || right_axis >= (int32_t)args->right_rank ||
            (left_axis < 0 && right_axis < 0)) {
            return 1;
        }
        if (left_axis >= 0) {
            if (left_used[left_axis] || args->left_shape[left_axis] != args->output_shape[output_axis]) {
                return 1;
            }
            left_used[left_axis] = 1;
        }
        if (right_axis >= 0) {
            if (right_used[right_axis] || args->right_shape[right_axis] != args->output_shape[output_axis]) {
                return 1;
            }
            right_used[right_axis] = 1;
        }
    }

    for (uint32_t contracted_axis = 0; contracted_axis < args->contracted_rank; contracted_axis++) {
        const int32_t left_axis = args->contracted_to_left_axes[contracted_axis];
        const int32_t right_axis = args->contracted_to_right_axes[contracted_axis];
        if (left_axis < 0 || right_axis < 0 ||
            left_axis >= (int32_t)args->left_rank || right_axis >= (int32_t)args->right_rank ||
            left_used[left_axis] || right_used[right_axis] ||
            args->contracted_dims[contracted_axis] == 0 ||
            args->left_shape[left_axis] != args->contracted_dims[contracted_axis] ||
            args->right_shape[right_axis] != args->contracted_dims[contracted_axis]) {
            return 1;
        }
        left_used[left_axis] = 1;
        right_used[right_axis] = 1;
    }

    for (uint32_t axis = 0; axis < args->left_rank; axis++) {
        if (!left_used[axis]) {
            return 1;
        }
    }
    for (uint32_t axis = 0; axis < args->right_rank; axis++) {
        if (!right_used[axis]) {
            return 1;
        }
    }
    return 0;
}

int main(int argc, char **argv) {
    if (argc != 6) {
        fprintf(stderr, "usage: %s <dpu_binary> <args.bin> <left_i8.bin> <right_i8.bin> <out_i32.bin>\n", argv[0]);
        return 2;
    }

    const char *dpu_binary = argv[1];
    const char *args_path = argv[2];
    const char *left_path = argv[3];
    const char *right_path = argv[4];
    const char *out_path = argv[5];

    upmem_generic_args_t args;
    if (read_exact(args_path, &args, sizeof(args)) != 0) {
        return 1;
    }
    if (args.left_rank > UPMEM_GENERIC_MAX_RANK || args.right_rank > UPMEM_GENERIC_MAX_RANK || args.output_rank > UPMEM_GENERIC_MAX_RANK || args.contracted_rank > UPMEM_GENERIC_MAX_RANK) {
        fprintf(stderr, "rank exceeds max rank %u\n", UPMEM_GENERIC_MAX_RANK);
        return 2;
    }
    if (args.left_elems == 0 || args.right_elems == 0 || args.output_elems == 0 || args.contracted_elems == 0) {
        fprintf(stderr, "zero element counts are unsupported\n");
        return 2;
    }
    if (args.left_elems > UPMEM_GENERIC_MAX_ELEMS || args.right_elems > UPMEM_GENERIC_MAX_ELEMS || args.output_elems > UPMEM_GENERIC_MAX_ELEMS || args.contracted_elems > UPMEM_GENERIC_MAX_ELEMS) {
        fprintf(stderr, "element counts exceed max elems %u\n", UPMEM_GENERIC_MAX_ELEMS);
        return 2;
    }
    if (args.operand_mode != UPMEM_GENERIC_MODE_INT8_SCALED && args.operand_mode != UPMEM_GENERIC_MODE_FLOAT32_NO_QUANT) {
        fprintf(stderr, "unsupported operand mode %u\n", args.operand_mode);
        return 2;
    }
    if (validate_row_major(args.left_shape, args.left_strides, args.left_rank, args.left_elems) != 0 ||
        validate_row_major(args.right_shape, args.right_strides, args.right_rank, args.right_elems) != 0 ||
        validate_row_major(args.output_shape, args.output_strides, args.output_rank, args.output_elems) != 0) {
        fprintf(stderr, "invalid row-major tensor metadata\n");
        return 2;
    }
    uint64_t contracted_product = 1;
    for (uint32_t axis = 0; axis < args.contracted_rank; axis++) {
        if (args.contracted_dims[axis] == 0 || contracted_product > UINT32_MAX / args.contracted_dims[axis]) {
            fprintf(stderr, "invalid contracted dimensions\n");
            return 2;
        }
        contracted_product *= args.contracted_dims[axis];
    }
    if (contracted_product != args.contracted_elems || validate_index_maps(&args) != 0) {
        fprintf(stderr, "invalid generic contraction index metadata\n");
        return 2;
    }

    const int float32_mode = args.operand_mode == UPMEM_GENERIC_MODE_FLOAT32_NO_QUANT;
    const size_t input_elem_size = float32_mode ? sizeof(float) : sizeof(int8_t);
    const size_t output_elem_size = float32_mode ? sizeof(float) : sizeof(int32_t);
    const char *left_symbol = "GENERIC_A_RAW";
    const char *right_symbol = "GENERIC_B_RAW";
    const char *output_symbol = "GENERIC_C_RAW";
    size_t left_bytes;
    size_t right_bytes;
    size_t output_bytes;
    size_t left_transfer_bytes;
    size_t right_transfer_bytes;
    size_t output_transfer_bytes;
    if (transfer_sizes(args.left_elems, input_elem_size, &left_bytes, &left_transfer_bytes) != 0 ||
        transfer_sizes(args.right_elems, input_elem_size, &right_bytes, &right_transfer_bytes) != 0 ||
        transfer_sizes(args.output_elems, output_elem_size, &output_bytes, &output_transfer_bytes) != 0) {
        fprintf(stderr, "unaligned or overflowing native transfer size\n");
        return 2;
    }

    unsigned char *left = (unsigned char *)calloc(left_transfer_bytes, 1);
    unsigned char *right = (unsigned char *)calloc(right_transfer_bytes, 1);
    unsigned char *output = (unsigned char *)calloc(output_transfer_bytes, 1);
    if (left == NULL || right == NULL || output == NULL) {
        fprintf(stderr, "host allocation failed\n");
        free(left);
        free(right);
        free(output);
        return 1;
    }
    if (read_exact(left_path, left, left_bytes) != 0) {
        free(left);
        free(right);
        free(output);
        return 1;
    }
    if (read_exact(right_path, right, right_bytes) != 0) {
        free(left);
        free(right);
        free(output);
        return 1;
    }

    struct dpu_set_t set;
    struct dpu_set_t dpu;
    DPU_ASSERT(dpu_alloc(1, NULL, &set));
    DPU_ASSERT(dpu_load(set, dpu_binary, NULL));
    DPU_ASSERT(dpu_broadcast_to(set, "GENERIC_ARGS", 0, &args, sizeof(args), DPU_XFER_DEFAULT));
    DPU_ASSERT(dpu_broadcast_to(set, left_symbol, 0, left, left_transfer_bytes, DPU_XFER_DEFAULT));
    DPU_ASSERT(dpu_broadcast_to(set, right_symbol, 0, right, right_transfer_bytes, DPU_XFER_DEFAULT));
    DPU_ASSERT(dpu_launch(set, DPU_SYNCHRONOUS));
    DPU_FOREACH(set, dpu) {
        DPU_ASSERT(dpu_copy_from(dpu, output_symbol, 0, output, output_transfer_bytes));
        break;
    }
    DPU_ASSERT(dpu_free(set));

    int rc = write_exact(out_path, output, output_bytes);
    free(left);
    free(right);
    free(output);
    return rc;
}
