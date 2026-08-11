#ifndef UPMEM_DISTRIBUTED_PLAN_V2_H
#define UPMEM_DISTRIBUTED_PLAN_V2_H

#include <stdint.h>

#include "execution_plan_v2_common.h"
#include "plan_schedule.h"

typedef struct {
    execution_plan_v2_header_t header;
    execution_plan_v2_work_unit_t work_units[EXECUTION_PLAN_V2_MAX_WORK_UNITS];
    uint32_t work_unit_count;
    uint64_t sidecar_file_fnv1a64_runtime;
    unsigned char package_file_sha256[32];
    char *file_sha256;
    char *file_path;
} execution_plan_distributed_v2_t;

int execution_plan_distributed_v2_load(
    const char *path,
    const unsigned char expected_package_sha256[32],
    const resident_request_t *resident,
    execution_plan_distributed_v2_t *plan,
    char **error_message
);

void execution_plan_distributed_v2_free(execution_plan_distributed_v2_t *plan);

#endif
