#ifndef USER_H
#define USER_H

#include <stdio.h>
#include <stdlib.h>
#include "Param.h"
#include "/home/tom/repos/Masters/thesis/extern/SimplePIM/lib/processing/map/MapArgs.h"

void start_func(map_arguments_t* args){}

void map_func(void* input, void* res){
    float val = *(T*)input;
    
    // Float emulation is slow on UPMEM. Looping 50 times simulates a heavy GEMM tile.
    for(int i = 0; i < 50; i++) {
        val = val * 1.01f;
    }
    
    *(T*)res = val;
}

#endif