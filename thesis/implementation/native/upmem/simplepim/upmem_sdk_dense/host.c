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
    if (argc != 8 && argc != 12) {
        fprintf(stderr, "usage L1: %s <dpu_binary> <m> <k> <n> <left_i8.bin> <right_i8.bin> <out_i32.bin>\n", argv[0]);
        fprintf(stderr, "usage L2: %s <dpu_binary> l2 <m> <k> <n> <tile_m> <tile_k> <tile_n> <left_i8.bin> <right_i8.bin> <out_i32.bin>\n", argv[0]);
        return 2;
    }

    const char *dpu_binary = argv[1];
    uint32_t strategy = UPMEM_DENSE_STRATEGY_L1;
    uint32_t arg_offset = 2;
    uint32_t tile_m = 0;
    uint32_t tile_k = 0;
    uint32_t tile_n = 0;
    if (argc == 12) {
        if (argv[2][0] != 'l' || argv[2][1] != '2' || argv[2][2] != '\0') {
            fprintf(stderr, "unsupported strategy: %s\n", argv[2]);
            return 2;
        }
        strategy = UPMEM_DENSE_STRATEGY_L2;
        arg_offset = 3;
    }

    const uint32_t m = (uint32_t)strtoul(argv[arg_offset], NULL, 10);
    const uint32_t k = (uint32_t)strtoul(argv[arg_offset + 1], NULL, 10);
    const uint32_t n = (uint32_t)strtoul(argv[arg_offset + 2], NULL, 10);
    const char *left_path;
    const char *right_path;
    const char *out_path;
    if (strategy == UPMEM_DENSE_STRATEGY_L2) {
        tile_m = (uint32_t)strtoul(argv[arg_offset + 3], NULL, 10);
        tile_k = (uint32_t)strtoul(argv[arg_offset + 4], NULL, 10);
        tile_n = (uint32_t)strtoul(argv[arg_offset + 5], NULL, 10);
        left_path = argv[arg_offset + 6];
        right_path = argv[arg_offset + 7];
        out_path = argv[arg_offset + 8];
    } else {
        left_path = argv[arg_offset + 3];
        right_path = argv[arg_offset + 4];
        out_path = argv[arg_offset + 5];
    }

    if (m == 0 || k == 0 || n == 0) {
        fprintf(stderr, "unsupported GEMM dimensions: m=%u k=%u n=%u\n", m, k, n);
        return 2;
    }

    uint32_t a_stride = UPMEM_DENSE_MAX_DIM;
    uint32_t b_stride = UPMEM_DENSE_MAX_DIM;
    uint32_t c_stride = UPMEM_DENSE_MAX_DIM;
    size_t left_bytes;
    size_t right_bytes;
    size_t output_bytes;
    if (strategy == UPMEM_DENSE_STRATEGY_L2) {
        if (m > UPMEM_DENSE_L2_MAX_DIM || k > UPMEM_DENSE_L2_MAX_DIM || n > UPMEM_DENSE_L2_MAX_DIM) {
            fprintf(stderr, "unsupported L2 GEMM dimensions: m=%u k=%u n=%u max=%u\n", m, k, n, UPMEM_DENSE_L2_MAX_DIM);
            return 2;
        }
        if (tile_m == 0 || tile_k == 0 || tile_n == 0 || tile_m > UPMEM_DENSE_L2_TILE_MAX_DIM || tile_k > UPMEM_DENSE_L2_TILE_MAX_DIM || tile_n > UPMEM_DENSE_L2_TILE_MAX_DIM) {
            fprintf(stderr, "unsupported L2 tile dimensions: tile_m=%u tile_k=%u tile_n=%u max=%u\n", tile_m, tile_k, tile_n, UPMEM_DENSE_L2_TILE_MAX_DIM);
            return 2;
        }
        if ((m % 8U) != 0U || (k % 8U) != 0U || (n % 8U) != 0U || (tile_m % 8U) != 0U || (tile_k % 8U) != 0U || (tile_n % 8U) != 0U) {
            fprintf(stderr, "L2 dimensions and tile dimensions must be multiples of 8 for this initial DMA-safe backend\n");
            return 2;
        }
        a_stride = k;
        b_stride = n;
        c_stride = n;
        left_bytes = (size_t)m * (size_t)k * sizeof(int8_t);
        right_bytes = (size_t)k * (size_t)n * sizeof(int8_t);
        output_bytes = (size_t)m * (size_t)n * sizeof(int32_t);
    } else {
        if (m > UPMEM_DENSE_MAX_DIM || k > UPMEM_DENSE_MAX_DIM || n > UPMEM_DENSE_MAX_DIM) {
            fprintf(stderr, "unsupported GEMM dimensions: m=%u k=%u n=%u max=%u\n", m, k, n, UPMEM_DENSE_MAX_DIM);
            return 2;
        }
        left_bytes = sizeof(int8_t) * UPMEM_DENSE_MAX_ELEMS;
        right_bytes = sizeof(int8_t) * UPMEM_DENSE_MAX_ELEMS;
        output_bytes = sizeof(int32_t) * UPMEM_DENSE_MAX_ELEMS;
    }

    int8_t *left = (int8_t *)calloc(left_bytes, 1);
    int8_t *right = (int8_t *)calloc(right_bytes, 1);
    int32_t *output = (int32_t *)calloc(output_bytes, 1);
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

    upmem_dense_args_t args = {
        strategy,
        m,
        k,
        n,
        a_stride,
        b_stride,
        c_stride,
        tile_m,
        tile_k,
        tile_n,
    };
    DPU_ASSERT(dpu_broadcast_to(set, "DENSE_ARGS", 0, &args, sizeof(args), DPU_XFER_DEFAULT));
    DPU_ASSERT(dpu_broadcast_to(set, "DENSE_A", 0, left, left_bytes, DPU_XFER_DEFAULT));
    DPU_ASSERT(dpu_broadcast_to(set, "DENSE_B", 0, right, right_bytes, DPU_XFER_DEFAULT));
    DPU_ASSERT(dpu_launch(set, DPU_SYNCHRONOUS));

    DPU_FOREACH(set, dpu) {
        DPU_ASSERT(dpu_copy_from(dpu, "DENSE_C", 0, output, output_bytes));
        break;
    }
    DPU_ASSERT(dpu_free(set));

    if (write_exact(out_path, output, output_bytes) != 0) {
        free(left);
        free(right);
        free(output);
        return 1;
    }
    free(left);
    free(right);
    free(output);
    return 0;
}
