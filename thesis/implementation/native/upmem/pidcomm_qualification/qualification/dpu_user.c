#include <alloc.h>
#include <barrier.h>
#include <defs.h>
#include <mram.h>

#define NR_TASKLETS 16
BARRIER_INIT(qualification_barrier, NR_TASKLETS);

int main(void) {
    barrier_wait(&qualification_barrier);
    return 0;
}
