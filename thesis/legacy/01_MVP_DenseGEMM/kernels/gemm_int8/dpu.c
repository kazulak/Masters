#include <barrier.h>
#include <defs.h>

#include "../../Param.h"
#include "map.h"

__host GemmTileInput DPU_INPUT;
__host GemmTileOutput DPU_OUTPUT;

BARRIER_INIT(gemm_barrier, NR_TASKLETS);

int main(void) {
    barrier_wait(&gemm_barrier);
    map();
    barrier_wait(&gemm_barrier);
    return 0;
}
