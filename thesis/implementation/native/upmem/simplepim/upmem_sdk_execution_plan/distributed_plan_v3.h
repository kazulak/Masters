#ifndef UPMEM_DISTRIBUTED_PLAN_V3_H
#define UPMEM_DISTRIBUTED_PLAN_V3_H

#include <stdint.h>

#include "execution_plan_v3_common.h"
#include "plan_schedule.h"

typedef struct {
    execution_plan_v3_header_t header;
    execution_plan_v3_work_unit_t *work_units;
    uint32_t work_unit_count;
    uint64_t sidecar_file_fnv1a64_runtime;
    unsigned char package_file_sha256[32];
    char *file_sha256;
    char *file_path;
} execution_plan_distributed_v3_t;

int execution_plan_distributed_v3_load(
    const char *path,
    const unsigned char expected_package_sha256[32],
    const resident_request_t *resident,
    execution_plan_distributed_v3_t *plan,
    char **error_message
);

void execution_plan_distributed_v3_free(execution_plan_distributed_v3_t *plan);

const execution_plan_v3_work_unit_t *execution_plan_distributed_v3_work_unit_for_dpu(
    const execution_plan_distributed_v3_t *plan,
    uint32_t dpu_id
);

#endif
