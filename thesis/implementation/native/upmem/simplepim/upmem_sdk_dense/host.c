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

int main(int argc, char **argv) {
    if (argc != 8) {
        fprintf(stderr, "usage: %s <dpu_binary> <m> <k> <n> <left_i8.bin> <right_i8.bin> <out_i32.bin>\n", argv[0]);
        return 2;
    }

    const char *dpu_binary = argv[1];
    const uint32_t m = (uint32_t)strtoul(argv[2], NULL, 10);
    const uint32_t k = (uint32_t)strtoul(argv[3], NULL, 10);
    const uint32_t n = (uint32_t)strtoul(argv[4], NULL, 10);
    const char *left_path = argv[5];
    const char *right_path = argv[6];
    const char *out_path = argv[7];

    if (m == 0 || k == 0 || n == 0 || m > UPMEM_DENSE_MAX_DIM || k > UPMEM_DENSE_MAX_DIM || n > UPMEM_DENSE_MAX_DIM) {
        fprintf(stderr, "unsupported GEMM dimensions: m=%u k=%u n=%u max=%u\n", m, k, n, UPMEM_DENSE_MAX_DIM);
        return 2;
    }

    int8_t left[UPMEM_DENSE_MAX_ELEMS] = {0};
    int8_t right[UPMEM_DENSE_MAX_ELEMS] = {0};
    int32_t output[UPMEM_DENSE_MAX_ELEMS] = {0};

    if (read_exact(left_path, left, sizeof(left)) != 0) {
        return 1;
    }
    if (read_exact(right_path, right, sizeof(right)) != 0) {
        return 1;
    }

    struct dpu_set_t set;
    struct dpu_set_t dpu;
    DPU_ASSERT(dpu_alloc(1, NULL, &set));
    DPU_ASSERT(dpu_load(set, dpu_binary, NULL));

    upmem_dense_args_t args = {
        m,
        k,
        n,
        UPMEM_DENSE_MAX_DIM,
        UPMEM_DENSE_MAX_DIM,
        UPMEM_DENSE_MAX_DIM,
    };
    DPU_ASSERT(dpu_broadcast_to(set, "DENSE_ARGS", 0, &args, sizeof(args), DPU_XFER_DEFAULT));
    DPU_ASSERT(dpu_broadcast_to(set, "DENSE_A", 0, left, sizeof(left), DPU_XFER_DEFAULT));
    DPU_ASSERT(dpu_broadcast_to(set, "DENSE_B", 0, right, sizeof(right), DPU_XFER_DEFAULT));
    DPU_ASSERT(dpu_launch(set, DPU_SYNCHRONOUS));

    DPU_FOREACH(set, dpu) {
        DPU_ASSERT(dpu_copy_from(dpu, "DENSE_C", 0, output, sizeof(output)));
        break;
    }
    DPU_ASSERT(dpu_free(set));

    if (write_exact(out_path, output, sizeof(output)) != 0) {
        return 1;
    }
    return 0;
}
