#include "simplepim_provider.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

static void execution_plan_free_management(simplepim_management_t *management) {
    if (management == NULL) return;
    if (management->num_tables != 0u) return;
    free(management->tables);
    free(management->zip_args);
    free(management->map_args);
    free(management->red_args);
    free(management);
}

static simplepim_management_t *execution_plan_management_from_set(
    struct dpu_set_t set, uint32_t dpu_count
) {
    simplepim_management_t *management = (simplepim_management_t *)calloc(1u, sizeof(*management));
    if (management == NULL) return NULL;
    management->set = set;
    management->num_dpus = dpu_count;
    management->curr_space = 16u;
    management->tables = (table_host_t **)calloc(management->curr_space, sizeof(*management->tables));
    management->zip_args = (zip_arguments_t *)calloc(dpu_count, sizeof(*management->zip_args));
    management->map_args = (map_arguments_t *)calloc(dpu_count, sizeof(*management->map_args));
    management->red_args = (gen_red_arguments_t *)calloc(dpu_count, sizeof(*management->red_args));
    if (management->tables == NULL || management->zip_args == NULL || management->map_args == NULL ||
        management->red_args == NULL) {
        execution_plan_free_management(management);
        return NULL;
    }
    return management;
}

dpu_error_t upmem_v4_provider_init_on_rank(
    upmem_v4_provider_t *provider,
    uint32_t requested_dpus,
    const char *requested_rank_path,
    const char *initialization_binary
) {
    dpu_error_t error = DPU_OK;
    char allocation_profile[320];
    uint32_t sdk_rank_count = 0u;
    if (provider == NULL || requested_dpus < 1u || requested_dpus > EXECUTION_PLAN_V4_MAX_DPUS ||
        requested_rank_path == NULL || requested_rank_path[0] == '\0' || initialization_binary == NULL ||
        initialization_binary[0] == '\0' ||
        snprintf(allocation_profile, sizeof(allocation_profile), "backend=hw,rankPath=%s",
            requested_rank_path) >= (int)sizeof(allocation_profile)) return DPU_ERR_INVALID_PROFILE;
    *provider = (upmem_v4_provider_t){0};
    provider->requested_dpus = requested_dpus;
    (void)snprintf(provider->requested_rank_path, sizeof(provider->requested_rank_path), "%s",
        requested_rank_path);
    provider->allocation_attempted = 1;
    error = dpu_alloc(requested_dpus, allocation_profile, &provider->set);
    if (error != DPU_OK) return error;
    provider->allocation_used = 1;
    provider->allocation_active = 1;
    error = dpu_get_nr_dpus(provider->set, &provider->observed_dpus);
    if (error == DPU_OK) error = dpu_get_nr_ranks(provider->set, &sdk_rank_count);
    if (error == DPU_OK) {
        struct dpu_set_t rank;
        uint32_t rank_index;
        DPU_RANK_FOREACH(provider->set, rank, rank_index) {
            (void)rank;
            (void)rank_index;
            provider->observed_ranks++;
        }
    }
    if (error != DPU_OK || provider->observed_dpus != requested_dpus || sdk_rank_count != 1u ||
        provider->observed_ranks != sdk_rank_count) {
        return error == DPU_OK ? DPU_ERR_INVALID_PROFILE : error;
    }
    error = dpu_load(provider->set, initialization_binary, NULL);
    if (error == DPU_OK) error = dpu_launch(provider->set, DPU_SYNCHRONOUS);
    if (error != DPU_OK) return error;
    provider->management = execution_plan_management_from_set(provider->set, requested_dpus);
    if (provider->management == NULL) return DPU_ERR_ALLOCATION;
    provider->initialization_completed = 1;
    return DPU_OK;
}

dpu_error_t upmem_v4_provider_release(upmem_v4_provider_t *provider) {
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
