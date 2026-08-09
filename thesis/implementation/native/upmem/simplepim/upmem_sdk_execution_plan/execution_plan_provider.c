#include "execution_plan_provider.h"

#include <stdlib.h>
#include <string.h>

static void execution_plan_free_management(simplepim_management_t *management) {
    if (management == NULL) return;
    if (management->num_tables != 0u) return;
    free(management->tables);
    free(management->zip_args);
    free(management->map_args);
    free(management->red_args);
    free(management);
}

dpu_error_t execution_plan_provider_init(
    execution_plan_provider_t *provider,
    uint32_t requested_dpus,
    const char *allocation_profile,
    const char *initialization_binary
) {
    dpu_error_t error = DPU_OK;
    int allocation_used = 0;
    int release_attempted = 0;
    int release_succeeded = 0;
    dpu_error_t release_error = DPU_OK;
    if (provider == NULL || requested_dpus < 1u || requested_dpus > EXECUTION_PLAN_MAX_DPUS ||
        allocation_profile == NULL || initialization_binary == NULL ||
        strcmp(allocation_profile, "backend=hw") != 0) return DPU_ERR_INVALID_PROFILE;
    *provider = (execution_plan_provider_t){0};
    provider->requested_dpus = requested_dpus;
    provider->allocation_attempted = 1;
    provider->management = table_management_init_with_profile(
        requested_dpus, allocation_profile, initialization_binary, &provider->set,
        &provider->allocation_active, &error, &allocation_used,
        &release_attempted, &release_succeeded, &release_error);
    provider->allocation_used = allocation_used;
    provider->release_attempted = release_attempted;
    provider->release_succeeded = release_succeeded;
    provider->release_error = release_error;
    if (provider->management == NULL) {
        provider->observed_dpus = allocation_used ? requested_dpus : 0u;
        return error == DPU_OK ? DPU_ERR_ALLOCATION : error;
    }
    provider->set = provider->management->set;
    provider->allocation_active = 1;
    error = dpu_get_nr_dpus(provider->set, &provider->observed_dpus);
    if (error != DPU_OK || provider->observed_dpus != requested_dpus) {
        return error == DPU_OK ? DPU_ERR_INVALID_PROFILE : error;
    }
    provider->initialization_completed = 1;
    return DPU_OK;
}

dpu_error_t execution_plan_provider_release(execution_plan_provider_t *provider) {
    dpu_error_t error = DPU_OK;
    if (provider == NULL) return DPU_ERR_INVALID_PROFILE;
    if (provider->release_attempted && !provider->allocation_active) return provider->release_error;
    provider->release_attempted = 1;
    if (provider->allocation_active) {
        error = dpu_free(provider->set);
        provider->release_error = error;
        if (error != DPU_OK) {
            provider->release_succeeded = 0;
            return error;
        }
        provider->allocation_active = 0;
    }
    if (provider->management != NULL && provider->management->num_tables != 0u) {
        provider->release_succeeded = 0;
        provider->release_error = DPU_ERR_INVALID_PROFILE;
        return provider->release_error;
    }
    execution_plan_free_management(provider->management);
    provider->management = NULL;
    provider->initialization_completed = 0;
    provider->release_succeeded = 1;
    return error;
}
