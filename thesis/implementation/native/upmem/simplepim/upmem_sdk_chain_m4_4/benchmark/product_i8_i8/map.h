#ifndef THESIS_M44_PRODUCT_I8_I8_MAP_H
#define THESIS_M44_PRODUCT_I8_I8_MAP_H

#include <stdint.h>

void start_func(void *args) { (void)args; }

void map_func(void *input, void *result) {
    const int8_t *pair = (const int8_t *)input;
    *(int32_t *)result = (int32_t)pair[0] * (int32_t)pair[1];
}

#endif
