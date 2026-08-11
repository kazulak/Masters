#ifndef UPMEM_EXECUTION_PLAN_REQUEST_H
#define UPMEM_EXECUTION_PLAN_REQUEST_H

#include <stdint.h>

#include "execution_plan_common.h"
#include "distributed_plan_v2.h"
#include "plan_schedule.h"

/* The native request is deliberately an argument contract, not another JSON
 * schema. Python owns the execution-plan JSON; native owns the resident
 * package manifest and the versioned binary sidecar. */
typedef struct {
    resident_request_t resident;
    execution_plan_schedule_t schedule;
    execution_plan_distributed_v2_t distributed_v2;
    int distributed_v2_mode;
    char *resident_manifest_path;
    char actual_package_file_sha256[65];
    uint64_t package_file_fnv1a64_runtime;
    uint32_t warmup_repetitions;
    uint32_t measured_repetitions;
    uint32_t record_for_package[EXECUTION_PLAN_MAX_TASKS];
    uint32_t package_for_operation[EXECUTION_PLAN_MAX_TASKS];
    int32_t producer_by_slot[RESIDENT_MAX_SLOT_DESCRIPTORS];
} execution_plan_request_t;

int execution_plan_request_load(
    const char *resident_manifest_path,
    const char *schedule_path,
    uint32_t warmup_repetitions,
    uint32_t measured_repetitions,
    execution_plan_request_t *request,
    char **error_message
);

int execution_plan_request_load_v2(
    const char *resident_manifest_path,
    const char *distributed_plan_path,
    uint32_t warmup_repetitions,
    uint32_t measured_repetitions,
    execution_plan_request_t *request,
    char **error_message
);

int execution_plan_request_validate(
    execution_plan_request_t *request,
    char **error_message
);

void execution_plan_request_free(execution_plan_request_t *request);

#endif
