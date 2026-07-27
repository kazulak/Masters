#ifndef SIMPLEPIM_QUALIFICATION_MAP_H
#define SIMPLEPIM_QUALIFICATION_MAP_H

/* Adapted from the pinned SimplePIM benchmarks/va/va_funcs/map.h. */
#include "Param.h"
#include "../../../lib/processing/map/MapArgs.h"

void start_func(map_arguments_t *args) {
    (void)args;
}

void map_func(void *input, void *result) {
    const T *pair = (const T *)input;
    *(T *)result = pair[0] + pair[1];
}

#endif
