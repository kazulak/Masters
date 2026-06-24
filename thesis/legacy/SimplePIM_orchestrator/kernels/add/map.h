#ifndef USER_H
#define USER_H

#include <stdio.h>
#include <stdlib.h>
#include "Param.h"
#include "/home/tom/repos/Masters/thesis/extern/SimplePIM/lib/processing/map/MapArgs.h"

void start_func(map_arguments_t* args){}

void map_func(void* input, void* res){
    // Input is a zipped struct containing one element from Train A and one from Train B
    float a = ((T*)input)[0];
    float b = ((T*)input)[1];
    *(T*)res = a + b;
}

#endif