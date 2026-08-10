#ifndef UPMEM_SDK_GENERIC_LOOP_RESIDENT_COMMON_H
#define UPMEM_SDK_GENERIC_LOOP_RESIDENT_COMMON_H

#include <stdint.h>

#ifndef NR_TASKLETS
#define NR_TASKLETS 1
#endif

#if NR_TASKLETS < 1 || NR_TASKLETS > 16
#error "generic_loop_resident_graph_session_v1 requires 1 <= NR_TASKLETS <= 16"
#endif

#ifndef UPMEM_GENERIC_MAX_RANK
#define UPMEM_GENERIC_MAX_RANK 16
#endif
#ifndef UPMEM_GENERIC_MAX_ELEMS
#define UPMEM_GENERIC_MAX_ELEMS 256
#endif
#ifndef RESIDENT_MAX_LOGICAL_TASKS
#define RESIDENT_MAX_LOGICAL_TASKS 32
#endif
#ifndef RESIDENT_MAX_COMPONENT_OPS
#define RESIDENT_MAX_COMPONENT_OPS 128
#endif
#ifndef RESIDENT_MAX_SLOT_DESCRIPTORS
#define RESIDENT_MAX_SLOT_DESCRIPTORS 128
#endif
#ifndef RESIDENT_MRAM_POOL_BYTES
#define RESIDENT_MRAM_POOL_BYTES (512u * 1024u)
#endif
#ifndef RESIDENT_OUTPUT_TILE_ELEMS
#define RESIDENT_OUTPUT_TILE_ELEMS 256
#endif

#define UPMEM_GENERIC_MODE_INT8_SCALED 0u
#define UPMEM_GENERIC_MODE_FLOAT32_NO_QUANT 1u
#define RESIDENT_INVALID_SLOT 0xffffffffu
#define RESIDENT_OPERATION_CONTRACT 1u
#define RESIDENT_OPERATION_COMPLEX_COMBINE 2u
#define RESIDENT_PACKAGE_VERSION 1u
#define RESIDENT_PACKAGE_ENDIAN 0x01020304u
#define RESIDENT_SLOT_ID_MASK 0x3fffffffu
#define RESIDENT_SLOT_INITIAL_FLAG 0x40000000u
#define RESIDENT_SLOT_FINAL_FLAG 0x80000000u
#define RESIDENT_COMPLETION_MAGIC 0x52534350u
#define RESIDENT_COMPLETION_VERSION 1u
#define RESIDENT_COMPLETION_PENDING 0u
#define RESIDENT_COMPLETION_COMPLETED 1u

typedef struct {
    uint32_t left_rank;
    uint32_t right_rank;
    uint32_t output_rank;
    uint32_t contracted_rank;
    uint32_t left_elems;
    uint32_t right_elems;
    uint32_t output_elems;
    uint32_t contracted_elems;
    uint32_t operand_mode;
    uint32_t left_shape[UPMEM_GENERIC_MAX_RANK];
    uint32_t right_shape[UPMEM_GENERIC_MAX_RANK];
    uint32_t output_shape[UPMEM_GENERIC_MAX_RANK];
    uint32_t contracted_dims[UPMEM_GENERIC_MAX_RANK];
    uint32_t left_strides[UPMEM_GENERIC_MAX_RANK];
    uint32_t right_strides[UPMEM_GENERIC_MAX_RANK];
    uint32_t output_strides[UPMEM_GENERIC_MAX_RANK];
    int32_t output_to_left_axes[UPMEM_GENERIC_MAX_RANK];
    int32_t output_to_right_axes[UPMEM_GENERIC_MAX_RANK];
    int32_t contracted_to_left_axes[UPMEM_GENERIC_MAX_RANK];
    int32_t contracted_to_right_axes[UPMEM_GENERIC_MAX_RANK];
} upmem_generic_args_t;

typedef struct {
    uint32_t slot_id;
    uint32_t offset_bytes;
    uint32_t capacity_elements;
    uint32_t element_count;
} resident_slot_descriptor_t;

typedef struct {
    uint32_t slot_count;
    uint32_t operation_count;
    uint32_t pool_bytes;
    uint32_t reserved;
} resident_control_t;

/* Small host-visible ABI record. The host reads this only after dpu_sync(). */
/* ABI Alignment Map:
 * Offset 00: magic (uint32_t, 4 bytes)
 * Offset 04: version (uint32_t, 4 bytes)
 * Offset 08: active_operation_index (uint32_t, 4 bytes)
 * Offset 12: completion_status (uint32_t, 4 bytes)
 * Offset 16: completed_operation_count (uint32_t, 4 bytes)
 * Offset 20: output_elements_processed (uint32_t, 4 bytes)
 * Offset 24: output_checksum_fnv1a64 (uint64_t, 8 bytes)
 * Offset 32: dpu_run_time_cycles (uint64_t, 8 bytes)
 * Total Size: 40 bytes (perfectly aligned without padding on 32-bit & 64-bit)
 */
typedef struct {
    uint32_t magic;
    uint32_t version;
    uint32_t active_operation_index;
    uint32_t completion_status;
    uint32_t completed_operation_count;
    uint32_t output_elements_processed;
    uint64_t output_checksum_fnv1a64;
    uint64_t dpu_run_time_cycles;
} resident_completion_t;

typedef struct {
    uint32_t kind;
    uint32_t mode;
    uint32_t output_elements;
    uint32_t slot_a;
    uint32_t slot_b;
    uint32_t slot_c;
    uint32_t slot_d;
    uint32_t slot_out_real;
    uint32_t slot_out_imag;
    float left_scale;
    float right_scale;
    upmem_generic_args_t args;
} resident_operation_t;

typedef struct {
    char magic[8];
    uint32_t version;
    uint32_t endian;
    uint32_t header_bytes;
    uint32_t flags;
    uint64_t file_bytes;
    uint64_t slot_offset;
    uint64_t slot_bytes;
    uint64_t operation_offset;
    uint64_t operation_bytes;
    uint32_t slot_count;
    uint32_t operation_count;
    uint32_t pool_bytes;
    uint32_t graph_request_count;
    uint32_t initial_slot_count;
    uint32_t final_output_count;
    uint32_t max_rank;
    uint32_t reserved;
} resident_package_header_t;

#endif
