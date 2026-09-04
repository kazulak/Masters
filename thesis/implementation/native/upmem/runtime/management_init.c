#include <defs.h>

/* Keep the pinned SimplePIM initializer, but give its shared allocator one owner. */
#define main simplepim_management_init
#include "SmallTableInit_dpu.c"
#undef main

int main(void) {
    /* Other tasklets consume no initialized state; synchronous launch joins all. */
    return me() == 0 ? simplepim_management_init() : 0;
}
