#ifndef UPMEM_UPMEM_RUNTIME_PROVIDER_H
#define UPMEM_UPMEM_RUNTIME_PROVIDER_H

#include <stdint.h>

#include <dpu.h>

#include "protocol.h"

typedef struct {
    struct dpu_set_t set;
    uint32_t requested_dpus;
    uint32_t observed_dpus;
    uint32_t observed_ranks;
    char requested_rank_path[256];
    int allocation_attempted;
    int allocation_used;
    int allocation_active;
    int release_attempted;
    int release_succeeded;
    dpu_error_t release_error;
} upmem_v4_provider_t;

/* Provider storage must be zero-initialized before its first init call. */
dpu_error_t upmem_v4_provider_init_on_rank(
    upmem_v4_provider_t *provider,
    uint32_t requested_dpus,
    const char *requested_rank_path
);

dpu_error_t upmem_v4_provider_init_simulator(
    upmem_v4_provider_t *provider,
    uint32_t requested_dpus
);

dpu_error_t upmem_v4_provider_release(upmem_v4_provider_t *provider);

#endif
