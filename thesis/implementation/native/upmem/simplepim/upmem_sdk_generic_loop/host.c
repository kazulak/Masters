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
    if (args.left_elems > UPMEM_GENERIC_MAX_ELEMS || args.right_elems > UPMEM_GENERIC_MAX_ELEMS || args.output_elems > UPMEM_GENERIC_MAX_ELEMS) {
        fprintf(stderr, "element counts exceed max elems %u\n", UPMEM_GENERIC_MAX_ELEMS);
        return 2;
    }

    const size_t left_bytes = args.left_elems * sizeof(int8_t);
    const size_t right_bytes = args.right_elems * sizeof(int8_t);
    const size_t output_bytes = args.output_elems * sizeof(int32_t);
    const size_t left_transfer_bytes = align8(left_bytes);
    const size_t right_transfer_bytes = align8(right_bytes);
    const size_t output_transfer_bytes = align8(output_bytes);

    int8_t *left = (int8_t *)calloc(left_transfer_bytes, 1);
    int8_t *right = (int8_t *)calloc(right_transfer_bytes, 1);
    int32_t *output = (int32_t *)calloc(output_transfer_bytes, 1);
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
    DPU_ASSERT(dpu_broadcast_to(set, "GENERIC_A", 0, left, left_transfer_bytes, DPU_XFER_DEFAULT));
    DPU_ASSERT(dpu_broadcast_to(set, "GENERIC_B", 0, right, right_transfer_bytes, DPU_XFER_DEFAULT));
    DPU_ASSERT(dpu_launch(set, DPU_SYNCHRONOUS));
    DPU_FOREACH(set, dpu) {
        DPU_ASSERT(dpu_copy_from(dpu, "GENERIC_C", 0, output, output_transfer_bytes));
        break;
    }
    DPU_ASSERT(dpu_free(set));

    int rc = write_exact(out_path, output, output_bytes);
    free(left);
    free(right);
    free(output);
    return rc;
}
