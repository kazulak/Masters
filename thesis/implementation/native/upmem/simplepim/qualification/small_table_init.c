#include <dpu.h>

#include "../../lib/management/SmallTableInit.h"

/*
 * Link-compatible implementation for upstream Management.c. The qualification
 * host performs this load itself so allocation and release stay in one cleanup
 * path; this function never compiles code or runs in the qualification flow.
 */
void small_table_init(struct dpu_set_t set) {
    DPU_ASSERT(dpu_load(set, "bin/dpu_init_binary", NULL));
    DPU_ASSERT(dpu_launch(set, DPU_SYNCHRONOUS));
}
