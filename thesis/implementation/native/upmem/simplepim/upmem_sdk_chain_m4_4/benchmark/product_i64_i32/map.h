#ifndef THESIS_M44_PRODUCT_I64_I32_MAP_H
#define THESIS_M44_PRODUCT_I64_I32_MAP_H

#include <stdint.h>
#include <string.h>

void start_func(void *args) { (void)args; }

void map_func(void *input, void *result) {
    const unsigned char *bytes = (const unsigned char *)input;
    int64_t left;
    int32_t right;
    memcpy(&left, bytes, sizeof(left));
    memcpy(&right, bytes + sizeof(left), sizeof(right));
    *(int64_t *)result = left * (int64_t)right;
}

#endif
