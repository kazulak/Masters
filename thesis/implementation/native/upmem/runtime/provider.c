#include "provider.h"

#include <stdio.h>
#include <string.h>

static dpu_error_t verify_allocation(
    upmem_v4_provider_t *provider,
    uint32_t requested_dpus
) {
    dpu_error_t error;
    uint32_t sdk_rank_count = 0u;

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
    if (error != DPU_OK || provider->observed_dpus != requested_dpus ||
        sdk_rank_count != 1u || provider->observed_ranks != sdk_rank_count) {
        return error == DPU_OK ? DPU_ERR_INVALID_PROFILE : error;
    }
    return DPU_OK;
}

static dpu_error_t finish_initialization(
    upmem_v4_provider_t *provider,
    uint32_t requested_dpus
) {
    dpu_error_t error = verify_allocation(provider, requested_dpus);
    if (error != DPU_OK) {
        (void)upmem_v4_provider_release(provider);
    }
    return error;
}

dpu_error_t upmem_v4_provider_init_on_rank(
    upmem_v4_provider_t *provider,
    uint32_t requested_dpus,
    const char *requested_rank_path
) {
    dpu_error_t error;
    char allocation_profile[320];

    if (provider == NULL || provider->allocation_active || requested_dpus < 1u ||
        requested_dpus > EXECUTION_PLAN_V4_MAX_DPUS ||
        requested_rank_path == NULL || requested_rank_path[0] == '\0' ||
        snprintf(allocation_profile, sizeof(allocation_profile),
            "backend=hw,rankPath=%s", requested_rank_path) >=
            (int)sizeof(allocation_profile)) {
        return DPU_ERR_INVALID_PROFILE;
    }
    *provider = (upmem_v4_provider_t){0};
    provider->requested_dpus = requested_dpus;
    (void)snprintf(provider->requested_rank_path,
        sizeof(provider->requested_rank_path), "%s", requested_rank_path);
    provider->allocation_attempted = 1;
    error = dpu_alloc(requested_dpus, allocation_profile, &provider->set);
    if (error != DPU_OK) return error;
    provider->allocation_used = 1;
    provider->allocation_active = 1;
    return finish_initialization(provider, requested_dpus);
}

dpu_error_t upmem_v4_provider_init_simulator(
    upmem_v4_provider_t *provider,
    uint32_t requested_dpus
) {
    dpu_error_t error;

    if (provider == NULL || provider->allocation_active || requested_dpus < 1u ||
        requested_dpus > EXECUTION_PLAN_V4_MAX_DPUS) {
        return DPU_ERR_INVALID_PROFILE;
    }
    *provider = (upmem_v4_provider_t){0};
    provider->requested_dpus = requested_dpus;
    provider->allocation_attempted = 1;
    error = dpu_alloc(requested_dpus, NULL, &provider->set);
    if (error != DPU_OK) return error;
    provider->allocation_used = 1;
    provider->allocation_active = 1;
    return finish_initialization(provider, requested_dpus);
}

dpu_error_t upmem_v4_provider_release(upmem_v4_provider_t *provider) {
    dpu_error_t error = DPU_OK;

    if (provider == NULL) return DPU_ERR_INVALID_PROFILE;
    if (provider->release_attempted && !provider->allocation_active) {
        return provider->release_error;
    }
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
    provider->release_succeeded = 1;
    return error;
}
