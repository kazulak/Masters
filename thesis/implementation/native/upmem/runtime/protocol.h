#ifndef UPMEM_UPMEM_RUNTIME_PROTOCOL_H
#define UPMEM_UPMEM_RUNTIME_PROTOCOL_H

#include <stdint.h>

#define EXECUTION_PLAN_V4_RESPONSE_SCHEMA "upmem_execution_plan_native_v4"
#define EXECUTION_PLAN_V4_PROFILE "upmem_execution_plan_v4_tile_session"
#define EXECUTION_PLAN_V4_NATIVE_BACKEND_ID "upmem_sdk_hardware_v4_tile_session"
#define EXECUTION_PLAN_V4_NATIVE_SIMULATOR_BACKEND_ID "upmem_sdk_simulator_v4_tile_session"
#define EXECUTION_PLAN_V4_NATIVE_BACKEND_FAMILY "upmem_sdk"
#define EXECUTION_PLAN_V4_NATIVE_M5_PROFILE "m5_whole_circuit_v4_v1"
#define EXECUTION_PLAN_V4_NATIVE_ABI "execution_plan_v4"
#define EXECUTION_PLAN_V4_NATIVE_SESSION "persistent_rank_session_v1"
#define EXECUTION_PLAN_V4_NATIVE_DISPATCH "bulk_set_synchronous_v1"
#define EXECUTION_PLAN_V4_NATIVE_KERNEL "dpu_real_tile_v4_wram_panel_v1"
#define EXECUTION_PLAN_V4_NATIVE_EXECUTION_CLASS "physical_v4_output_tile"
#define EXECUTION_PLAN_V4_NATIVE_SIMULATOR_EXECUTION_CLASS "sdk_simulator_v4_output_tile"
#define EXECUTION_PLAN_V4_MAGIC "UPXDPV4"
#define EXECUTION_PLAN_V4_VERSION 4u
#define EXECUTION_PLAN_V4_MAX_DPUS 64u
#define EXECUTION_PLAN_V4_MAX_TASKLETS 24u
#define EXECUTION_PLAN_V4_MAX_WORK_UNITS EXECUTION_PLAN_V4_MAX_DPUS
#define EXECUTION_PLAN_V4_MRAM_POOL_BYTES (512u * 1024u)
#define EXECUTION_PLAN_V4_MRAM_ALIGNMENT 8u
#define EXECUTION_PLAN_V4_MAX_REQUEST_BYTES (8u * 1024u * 1024u)
#define EXECUTION_PLAN_V4_MAX_CONTRACTED 65536u
#define EXECUTION_PLAN_V4_WRAM_PANEL_KC 64u
#define EXECUTION_PLAN_V4_WRAM_PANEL_NC 32u
#define EXECUTION_PLAN_V4_WRAM_PANEL_DMA_BYTES 2048u
#define EXECUTION_PLAN_V4_WRAM_PANEL_UNALIGNED_SCRATCH_BYTES 288u
#define EXECUTION_PLAN_V4_PARTITION_OUTPUT_TILE 1u
#define EXECUTION_PLAN_V4_NUMERIC_FLOAT32 0u
#define EXECUTION_PLAN_V4_NUMERIC_HOST_PACKED_INT8 1u
#define EXECUTION_PLAN_V4_INT8_MAX_ABS 127u
#define EXECUTION_PLAN_V4_FLAG_ZERO_WORK 0x00000001u
#define EXECUTION_PLAN_V4_CONTROL_MAGIC 0x34564354u
#define EXECUTION_PLAN_V4_COMPLETION_MAGIC 0x34564350u
#define EXECUTION_PLAN_V4_STATUS_PENDING 0u
#define EXECUTION_PLAN_V4_STATUS_COMPLETED 1u
#define EXECUTION_PLAN_V4_STATUS_FAILED 2u

typedef struct __attribute__((packed)) {
    char magic[8];
    uint32_t version;
    uint32_t header_bytes;
    uint32_t work_unit_count;
    uint32_t dpu_count;
    uint32_t tasklets_per_dpu;
    uint32_t numeric_mode;
    uint32_t partition_mode;
    uint32_t record_bytes;
    uint32_t reserved0;
    uint32_t reserved1;
    uint64_t canonical_batch_count;
    uint64_t canonical_m;
    uint64_t canonical_n;
    uint64_t canonical_k;
    uint64_t global_output_elements;
    uint64_t request_output_elements;
    uint64_t request_sequence;
    unsigned char task_contract_sha256[32];
    unsigned char request_sha256[32];
} execution_plan_v4_header_t;

typedef struct __attribute__((packed)) {
    uint32_t local_dpu_id;
    uint32_t flags;
    uint64_t tile_id;
    uint64_t batch_index;
    uint64_t m_offset;
    uint64_t n_offset;
    uint64_t k_offset;
    uint32_t m_elements;
    uint32_t n_elements;
    uint32_t k_elements;
    uint32_t a_transfer_bytes;
    uint32_t b_transfer_bytes;
    uint32_t c_transfer_bytes;
    uint32_t a_offset_bytes;
    uint32_t b_offset_bytes;
    uint32_t c_offset_bytes;
} execution_plan_v4_work_unit_t;

typedef struct __attribute__((packed)) {
    uint32_t magic;
    uint32_t version;
    uint32_t numeric_mode;
    uint32_t dpu_id;
    uint32_t flags;
    uint32_t batch_index;
    uint32_t m_elements;
    uint32_t n_elements;
    uint32_t k_elements;
    uint32_t a_transfer_bytes;
    uint32_t b_transfer_bytes;
    uint32_t c_transfer_bytes;
    uint32_t a_offset_bytes;
    uint32_t b_offset_bytes;
    uint32_t c_offset_bytes;
    uint32_t k_offset;
    uint32_t reserved0;
    uint32_t reserved1;
} execution_plan_v4_control_t;

typedef struct __attribute__((packed)) {
    uint32_t magic;
    uint32_t version;
    uint32_t status;
    uint32_t dpu_id;
    uint64_t cycles;
    uint64_t processed_elements;
    uint64_t checksum_fnv1a64;
} execution_plan_v4_completion_t;

_Static_assert(sizeof(execution_plan_v4_header_t) == 168u,
    "distributed v4 header ABI drifted");
_Static_assert(sizeof(execution_plan_v4_work_unit_t) == 84u,
    "distributed v4 work-unit ABI drifted");
_Static_assert(sizeof(execution_plan_v4_control_t) == 72u,
    "distributed v4 control ABI drifted");
_Static_assert(sizeof(execution_plan_v4_completion_t) == 40u,
    "distributed v4 completion ABI drifted");
_Static_assert((uint64_t)EXECUTION_PLAN_V4_MAX_CONTRACTED *
        (uint64_t)EXECUTION_PLAN_V4_INT8_MAX_ABS *
        (uint64_t)EXECUTION_PLAN_V4_INT8_MAX_ABS <= 2147483647u,
    "v4 int32 accumulation bound is unsafe");

#endif
