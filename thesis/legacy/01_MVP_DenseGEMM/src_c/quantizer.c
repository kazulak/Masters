#include "quantizer.h"

#include <math.h>
#include <string.h>

void quantize_f64_to_i8(const double *src, int8_t *dst,
                        size_t n_elements, double *scale_out) {
    double max_abs = 0.0;
    for (size_t i = 0; i < n_elements; i++) {
        double a = fabs(src[i]);
        if (a > max_abs) max_abs = a;
    }

    double scale = (max_abs < 1e-300) ? 1.0 : (max_abs / 127.0);
    *scale_out = scale;

    for (size_t i = 0; i < n_elements; i++) {
        double v = src[i] / scale;
        if (v > 127.0) v = 127.0;
        if (v < -127.0) v = -127.0;
        dst[i] = (int8_t)(v >= 0.0 ? v + 0.5 : v - 0.5);
    }
}

void dequantize_i32_accumulate(const int32_t *src, double *dst,
                               size_t n_elements,
                               double scale_A, double scale_B) {
    double combined_scale = scale_A * scale_B;
    for (size_t i = 0; i < n_elements; i++) {
        dst[i] += (double)src[i] * combined_scale;
    }
}

void extract_tile(const double *matrix, double *tile,
                  int total_rows, int total_cols,
                  int row_start, int col_start,
                  int tile_rows, int tile_cols) {
    memset(tile, 0, (size_t)tile_rows * (size_t)tile_cols * sizeof(double));
    for (int r = 0; r < tile_rows; r++) {
        int global_r = row_start + r;
        if (global_r >= total_rows) break;
        for (int c = 0; c < tile_cols; c++) {
            int global_c = col_start + c;
            if (global_c >= total_cols) break;
            tile[r * tile_cols + c] = matrix[global_r * total_cols + global_c];
        }
    }
}

void accumulate_tile(double *accumulator, const double *tile,
                     int total_rows, int total_cols,
                     int row_start, int col_start,
                     int tile_rows, int tile_cols) {
    for (int r = 0; r < tile_rows; r++) {
        int global_r = row_start + r;
        if (global_r >= total_rows) break;
        for (int c = 0; c < tile_cols; c++) {
            int global_c = col_start + c;
            if (global_c >= total_cols) break;
            accumulator[global_r * total_cols + global_c] += tile[r * tile_cols + c];
        }
    }
}
