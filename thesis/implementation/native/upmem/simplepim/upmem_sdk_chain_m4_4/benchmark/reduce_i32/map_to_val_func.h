#ifndef THESIS_M44_REDUCE_I32_MAP_TO_VAL_H
#define THESIS_M44_REDUCE_I32_MAP_TO_VAL_H

#include <stdint.h>

void start_func(void *args) { (void)args; }

void map_to_val_func(void *input, void *output, uint32_t *key) {
    *key = 0;
    *(int64_t *)output = (int64_t)*(const int32_t *)input;
}

#endif
