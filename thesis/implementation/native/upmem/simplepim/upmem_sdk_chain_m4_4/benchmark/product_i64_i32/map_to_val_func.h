#ifndef THESIS_M42_DOT_MAP_TO_VAL_H
#define THESIS_M42_DOT_MAP_TO_VAL_H

#include <stdint.h>

void start_func(void *args) {
    (void)args;
}

void map_to_val_func(void *input, void *output, uint32_t *key) {
    *key = 0;
    *(int64_t *)output = *(const int64_t *)input;
}

#endif
