#ifndef UPMEM_EXECUTION_PLAN_COMMON_H
#define UPMEM_EXECUTION_PLAN_COMMON_H

#include <stdint.h>

#include "../upmem_sdk_generic_loop_resident/common.h"
#include "../upmem_sdk_generic_loop_resident/session_protocol.h"

#define EXECUTION_PLAN_RESPONSE_SCHEMA "upmem_execution_plan_native_v1"
#define EXECUTION_PLAN_PROVIDER "simplepim_management_v1"
#define EXECUTION_PLAN_KERNEL_PROVIDER "thesis_resident_generic_contract_v1"
#define EXECUTION_PLAN_COMMUNICATION_PROVIDER "host_mediated_v1"
#define EXECUTION_PLAN_NUMERIC_MODE "none"
#define EXECUTION_PLAN_PROFILE "upmem_execution_plan_v1"
#define EXECUTION_PLAN_MAX_TASKS 8u
#define EXECUTION_PLAN_MAX_WAVES 8u
#define EXECUTION_PLAN_MAX_DPUS 2u
#define EXECUTION_PLAN_MAX_REPETITIONS 16u
#define EXECUTION_PLAN_MAX_PATH 4096u

#define EXECUTION_PLAN_SCHEDULE_MAGIC "UPXPLAN1"
#define EXECUTION_PLAN_SCHEDULE_VERSION 1u
#define EXECUTION_PLAN_SCHEDULE_HEADER_BYTES 80u
#define EXECUTION_PLAN_SCHEDULE_RECORD_BYTES 32u

typedef struct __attribute__((packed)) {
    char magic[8];
    uint32_t version;
    uint32_t header_bytes;
    uint32_t operation_count;
    uint32_t wave_count;
    uint32_t dpu_count;
    uint32_t tasklets_per_dpu;
    uint32_t provider_count;
    uint32_t record_bytes;
    uint32_t reserved0;
    uint32_t reserved1;
    unsigned char package_sha256[32];
} execution_plan_schedule_header_t;

typedef struct __attribute__((packed)) {
    uint32_t package_operation_index;
    uint32_t operation_id;
    uint32_t dependency_mask;
    uint32_t wave_index;
    uint32_t dpu_id;
    uint32_t input_slot_a;
    uint32_t input_slot_b;
    uint32_t output_slot;
} execution_plan_schedule_record_t;

_Static_assert(sizeof(resident_package_header_t) == 96u, "resident package header ABI drifted");
_Static_assert(sizeof(resident_slot_descriptor_t) == 16u, "resident slot ABI drifted");
_Static_assert(sizeof(resident_operation_t) == 800u, "resident operation ABI drifted");
_Static_assert(sizeof(execution_plan_schedule_header_t) == EXECUTION_PLAN_SCHEDULE_HEADER_BYTES, "execution plan header ABI drifted");
_Static_assert(sizeof(execution_plan_schedule_record_t) == EXECUTION_PLAN_SCHEDULE_RECORD_BYTES, "execution plan record ABI drifted");

#endif
