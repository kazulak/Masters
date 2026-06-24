#ifndef GEMM_INT8_MAP_H
#define GEMM_INT8_MAP_H

#include <defs.h>
#include <stdint.h>
#include "../../Param.h"

extern GemmTileInput DPU_INPUT;
extern GemmTileOutput DPU_OUTPUT;

void map(void) {
    int tid = me();
    int rows = DPU_INPUT.tile_rows;
    int k = DPU_INPUT.k;
    int cols = DPU_INPUT.tile_cols;
    int rows_per_tasklet = (rows + NR_TASKLETS - 1) / NR_TASKLETS;
    int row_start = tid * rows_per_tasklet;
    int row_end = row_start + rows_per_tasklet;
    if (row_end > rows) row_end = rows;

    for (int i = row_start; i < row_end; i++) {
        for (int j = 0; j < cols; j++) {
            int32_t acc = 0;
            for (int l = 0; l < k; l++) {
                acc += (int32_t)DPU_INPUT.a[i * k + l] *
                       (int32_t)DPU_INPUT.b[l * cols + j];
            }
            DPU_OUTPUT.c[i * cols + j] = acc;
        }
    }
}

#endif /* GEMM_INT8_MAP_H */
