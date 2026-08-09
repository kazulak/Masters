#ifndef THESIS_M44_PRODUCT_I32_I8_MAP_H
#define THESIS_M44_PRODUCT_I32_I8_MAP_H

#include <stdint.h>
#include <string.h>

void start_func(void *args) { (void)args; }

void map_func(void *input, void *result) {
    const unsigned char *bytes = (const unsigned char *)input;
    int32_t left;
    int8_t right;
    memcpy(&left, bytes, sizeof(left));
    memcpy(&right, bytes + sizeof(left), sizeof(right));
    *(int32_t *)result = left * (int32_t)right;
}

#endif
