#ifndef UPMEM_EXECUTION_PLAN_PROVIDER_H
#define UPMEM_EXECUTION_PLAN_PROVIDER_H

#include <stdint.h>

#include <dpu.h>
#include <Management.h>

#include "execution_plan_common.h"

typedef struct {
    simplepim_management_t *management;
    struct dpu_set_t set;
    uint32_t requested_dpus;
    uint32_t observed_dpus;
    uint32_t observed_ranks;
    char requested_rank_path[256];
    int allocation_attempted;
    int allocation_used;
    int allocation_active;
    int initialization_completed;
    int release_attempted;
    int release_succeeded;
    dpu_error_t release_error;
} execution_plan_provider_t;

dpu_error_t execution_plan_provider_init(
    execution_plan_provider_t *provider,
    uint32_t requested_dpus,
    const char *allocation_profile,
    const char *initialization_binary
);

dpu_error_t execution_plan_provider_init_on_rank(
    execution_plan_provider_t *provider,
    uint32_t requested_dpus,
    const char *requested_rank_path,
    const char *initialization_binary
);

dpu_error_t execution_plan_provider_release(execution_plan_provider_t *provider);

#endif
