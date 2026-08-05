#ifndef UPMEM_SIMPLEPIM_FRONTIER_PROVIDER_H
#define UPMEM_SIMPLEPIM_FRONTIER_PROVIDER_H

#include <stdint.h>

#include <dpu.h>
#include <Management.h>

typedef struct {
    simplepim_management_t *management;
    struct dpu_set_t set;
    uint32_t observed_dpus;
    int allocation_attempted;
    int allocation_used;
    int allocation_active;
    int initialization_completed;
    int release_attempted;
    int release_succeeded;
    dpu_error_t release_error;
} simplepim_frontier_provider_t;

/* Thesis-owned, fail-closed adapter around the staged SimplePIM management
 * extension. The returned object is allocated by SimplePIM's management
 * implementation, not by this adapter. */
dpu_error_t simplepim_frontier_provider_init(
    simplepim_frontier_provider_t *provider,
    uint32_t requested_dpus,
    const char *allocation_profile,
    const char *initialization_binary
);

dpu_error_t simplepim_frontier_provider_release(
    simplepim_frontier_provider_t *provider
);

#endif
