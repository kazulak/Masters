#ifndef UPMEM_DISTRIBUTED_PLAN_V4_H
#define UPMEM_DISTRIBUTED_PLAN_V4_H

#include <stdint.h>

#include "execution_plan_v4_common.h"

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

#endif
