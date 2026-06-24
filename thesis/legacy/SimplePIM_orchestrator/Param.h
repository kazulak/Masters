#ifndef PARAM_H
#define PARAM_H

#include <stdint.h>

// Now using floating point!
typedef float T;

// 4096 floats = 16 KiB per DPU (Fits perfectly in 64 KiB WRAM with room for double buffering later)
static const uint64_t nr_elements = 4096;
static const uint32_t dpu_number = 1;
static const int print_info = 0;

#endif