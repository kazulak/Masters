#ifndef UPMEM_EXECUTION_PLAN_V2_COMMON_H
#define UPMEM_EXECUTION_PLAN_V2_COMMON_H

#include <stdint.h>

#include "execution_plan_common.h"

#define EXECUTION_PLAN_V2_RESPONSE_SCHEMA "upmem_execution_plan_native_v2"
#define EXECUTION_PLAN_V2_PROFILE "upmem_execution_plan_v2_distributed_partition"
#define EXECUTION_PLAN_V2_PROVIDER_COUNT 1u
#define EXECUTION_PLAN_V2_MAX_DPUS 4u
#define EXECUTION_PLAN_V2_MAX_WORK_UNITS 4u
#define EXECUTION_PLAN_V2_MAX_PATH 4096u

#define EXECUTION_PLAN_V2_MAGIC "UPXDPV2"
#define EXECUTION_PLAN_V2_VERSION 2u
#define EXECUTION_PLAN_V2_HEADER_BYTES 132u
#define EXECUTION_PLAN_V2_RECORD_BYTES 32u
#define EXECUTION_PLAN_V2_PARTITION_OUTPUT 1u
#define EXECUTION_PLAN_V2_PARTITION_CONTRACTED 2u

typedef struct __attribute__((packed)) {
    char magic[8];
    uint32_t version;
    uint32_t header_bytes;
    uint32_t work_unit_count;
    uint32_t dpu_count;
    uint32_t tasklets_per_dpu;
    uint32_t provider_count;
    uint32_t partition_mode;
    uint32_t package_operation_index;
    uint32_t operation_id;
    uint32_t output_elements;
    uint32_t contracted_elements;
    uint32_t output_slot;
    uint32_t record_bytes;
    uint32_t reserved0;
    uint32_t reserved1;
    unsigned char package_sha256[32];
    unsigned char operation_sha256[32];
} execution_plan_v2_header_t;

typedef struct __attribute__((packed)) {
    uint32_t package_operation_index;
    uint32_t operation_id;
    uint32_t partition_mode;
    uint32_t dpu_id;
    uint32_t output_offset;
    uint32_t output_elements;
    uint32_t contracted_offset;
    uint32_t contracted_elements;
} execution_plan_v2_work_unit_t;

_Static_assert(sizeof(execution_plan_v2_header_t) == EXECUTION_PLAN_V2_HEADER_BYTES, "distributed v2 header ABI drifted");
_Static_assert(sizeof(execution_plan_v2_work_unit_t) == EXECUTION_PLAN_V2_RECORD_BYTES, "distributed v2 work-unit ABI drifted");

#endif
