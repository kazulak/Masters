#ifndef USER_H
#define USER_H

#include <stdio.h>
#include <stdlib.h>
#include "Param.h"
#include "/home/tom/repos/Masters/thesis/extern/SimplePIM/lib/processing/map/MapArgs.h"

// Required by the framework but can be empty
void start_func(map_arguments_t* args){}

// input is the pointer to the current element
// res is the pointer where the result should be written
void map_func(void* input, void* res){
    *(T*)res = (*(T*)input) * 100;
}

#endif
