#ifndef UPMEM_EXECUTION_PLAN_SCHEDULE_H
#define UPMEM_EXECUTION_PLAN_SCHEDULE_H

#include <stddef.h>
#include <stdint.h>

#include "execution_plan_common.h"

typedef struct {
    execution_plan_schedule_header_t header;
    execution_plan_schedule_record_t records[EXECUTION_PLAN_MAX_TASKS];
    uint32_t record_count;
    uint64_t schedule_file_fnv1a64_runtime;
    unsigned char package_file_sha256[32];
    char *file_sha256;
    char *file_path;
} execution_plan_schedule_t;

int execution_plan_schedule_load(
    const char *path,
    const unsigned char expected_package_sha256[32],
    execution_plan_schedule_t *schedule,
    char **error_message
);

void execution_plan_schedule_free(execution_plan_schedule_t *schedule);

uint64_t execution_plan_fnv1a64(const unsigned char *data, size_t length);
int execution_plan_hash_file(const char *path, uint64_t *hash);
int execution_plan_sha256_file(const char *path, char output[65]);
int execution_plan_sha256_bytes(const unsigned char *data, size_t length, char output[65]);

#endif
