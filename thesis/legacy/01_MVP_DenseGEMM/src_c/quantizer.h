#ifndef QUANTIZER_H
#define QUANTIZER_H

#include <stdint.h>
#include <stddef.h>

void quantize_f64_to_i8(const double *src, int8_t *dst,
                        size_t n_elements, double *scale_out);
void dequantize_i32_accumulate(const int32_t *src, double *dst,
                               size_t n_elements,
                               double scale_A, double scale_B);
void extract_tile(const double *matrix, double *tile,
                  int total_rows, int total_cols,
                  int row_start, int col_start,
                  int tile_rows, int tile_cols);
void accumulate_tile(double *accumulator, const double *tile,
                     int total_rows, int total_cols,
                     int row_start, int col_start,
                     int tile_rows, int tile_cols);

#endif /* QUANTIZER_H */
