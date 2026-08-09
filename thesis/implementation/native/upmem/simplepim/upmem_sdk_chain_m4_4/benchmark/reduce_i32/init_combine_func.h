#ifndef THESIS_M44_REDUCE_I32_INIT_COMBINE_H
#define THESIS_M44_REDUCE_I32_INIT_COMBINE_H

#include <stdint.h>

void init_func(uint32_t size, void *ptr) {
    uint8_t *bytes = (uint8_t *)ptr;
    for (uint32_t i = 0; i < size; ++i) bytes[i] = 0;
}

void combine_func(void *dest, void *src) {
    *(int64_t *)dest += *(const int64_t *)src;
}

#endif
