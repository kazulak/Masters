#ifndef THESIS_M42_DOT_MAP_H
#define THESIS_M42_DOT_MAP_H

#include <stdint.h>

void start_func(void *args) {
    (void)args;
}

void map_func(void *input, void *result) {
    const int32_t *pair = (const int32_t *)input;
    *(int64_t *)result = (int64_t)pair[0] * (int64_t)pair[1];
}

#endif
