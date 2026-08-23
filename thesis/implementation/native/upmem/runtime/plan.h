#ifndef UPMEM_UPMEM_RUNTIME_PLAN_H
#define UPMEM_UPMEM_RUNTIME_PLAN_H

#include <stdint.h>
#include <stddef.h>

#include "protocol.h"

int execution_plan_distributed_v4_validate(
    const execution_plan_v4_header_t *header,
    const execution_plan_v4_work_unit_t *work_units,
    uint32_t expected_dpus,
    uint32_t expected_tasklets,
    char **error_message
);

const execution_plan_v4_work_unit_t *execution_plan_distributed_v4_work_unit_for_dpu(
    const execution_plan_v4_work_unit_t *work_units,
    uint32_t work_unit_count,
    uint32_t dpu_id
);

int execution_plan_sha256_file(const char *path, char output[65]);
int execution_plan_sha256_bytes(const unsigned char *data, size_t length, char output[65]);

#endif
