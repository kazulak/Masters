#ifndef UPMEM_EXECUTION_PLAN_V3_COMMON_H
#define UPMEM_EXECUTION_PLAN_V3_COMMON_H

#include <stdint.h>

#include "execution_plan_v2_common.h"

#define EXECUTION_PLAN_V3_RESPONSE_SCHEMA "upmem_execution_plan_native_v3"
#define EXECUTION_PLAN_V3_PROFILE "upmem_execution_plan_v3_distributed_partition"
#define EXECUTION_PLAN_V3_PROVIDER_COUNT 1u
#define EXECUTION_PLAN_V3_MAX_DPUS 64u
#define EXECUTION_PLAN_V3_MAX_TASKLETS 24u
#define EXECUTION_PLAN_V3_MAX_REPETITIONS 16u
#define EXECUTION_PLAN_V3_MAX_ELEMS 65536u
#define EXECUTION_PLAN_V3_MRAM_POOL_BYTES (512u * 1024u)
#define EXECUTION_PLAN_V3_OUTPUT_TILE_ELEMS 2u
#define EXECUTION_PLAN_V3_MAGIC "UPXDPV3"
#define EXECUTION_PLAN_V3_VERSION 3u
#define EXECUTION_PLAN_V3_HEADER_BYTES 136u
#define EXECUTION_PLAN_V3_RECORD_BYTES 32u

#define EXECUTION_PLAN_V3_PARTITION_OUTPUT EXECUTION_PLAN_V2_PARTITION_OUTPUT
#define EXECUTION_PLAN_V3_PARTITION_CONTRACTED EXECUTION_PLAN_V2_PARTITION_CONTRACTED

/* The sidecar describes transport and arithmetic separately.  Both modes
 * move float32 slots through MRAM; mode 1 performs per-task int8 arithmetic
 * and requantizes the result on the DPU. */
#define EXECUTION_PLAN_V3_NUMERIC_FLOAT32 0u
#define EXECUTION_PLAN_V3_NUMERIC_INT8_REQUANTIZE 1u

typedef struct __attribute__((packed)) {
    char magic[8];
    uint32_t version;
    uint32_t header_bytes;
    uint32_t work_unit_count;
    uint32_t dpu_count;
    uint32_t tasklets_per_dpu;
    uint32_t provider_count;
    uint32_t partition_mode;
    uint32_t numeric_mode;
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
} execution_plan_v3_header_t;

typedef struct __attribute__((packed)) {
    uint32_t package_operation_index;
    uint32_t operation_id;
    uint32_t partition_mode;
    uint32_t dpu_id;
    uint32_t output_offset;
    uint32_t output_elements;
    uint32_t contracted_offset;
    uint32_t contracted_elements;
} execution_plan_v3_work_unit_t;

_Static_assert(sizeof(execution_plan_v3_header_t) == EXECUTION_PLAN_V3_HEADER_BYTES,
    "distributed v3 header ABI drifted");
_Static_assert(sizeof(execution_plan_v3_work_unit_t) == EXECUTION_PLAN_V3_RECORD_BYTES,
    "distributed v3 work-unit ABI drifted");
#if defined(RESIDENT_V3)
_Static_assert(UPMEM_GENERIC_MAX_ELEMS == EXECUTION_PLAN_V3_MAX_ELEMS,
    "resident and execution-plan v3 element limits disagree");
_Static_assert(RESIDENT_MRAM_POOL_BYTES == EXECUTION_PLAN_V3_MRAM_POOL_BYTES,
    "resident and execution-plan v3 MRAM limits disagree");
_Static_assert(RESIDENT_OUTPUT_TILE_ELEMS == EXECUTION_PLAN_V3_OUTPUT_TILE_ELEMS,
    "resident and execution-plan v3 tile sizes disagree");
#endif

#endif
