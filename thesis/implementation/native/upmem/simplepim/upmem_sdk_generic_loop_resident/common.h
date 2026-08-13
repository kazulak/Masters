#ifndef UPMEM_SDK_GENERIC_LOOP_RESIDENT_COMMON_H
#define UPMEM_SDK_GENERIC_LOOP_RESIDENT_COMMON_H

#include <stddef.h>
#include <stdint.h>

#ifndef NR_TASKLETS
#define NR_TASKLETS 1
#endif

#if defined(RESIDENT_V3)
#if NR_TASKLETS < 1 || NR_TASKLETS > 24
#error "resident v3 profile requires NR_TASKLETS in [1,24]"
#endif
#else
#if NR_TASKLETS != 1 && NR_TASKLETS != 2 && NR_TASKLETS != 4 && NR_TASKLETS != 8 && NR_TASKLETS != 16
#error "resident M4.6 profile requires NR_TASKLETS in {1,2,4,8,16}"
#endif
#endif

#ifndef UPMEM_GENERIC_MAX_RANK
#define UPMEM_GENERIC_MAX_RANK 16
#endif
#ifndef UPMEM_GENERIC_MAX_ELEMS
#if defined(RESIDENT_V3)
#define UPMEM_GENERIC_MAX_ELEMS 65536
#else
#define UPMEM_GENERIC_MAX_ELEMS 256
#endif
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
#if defined(RESIDENT_V3)
#define RESIDENT_OUTPUT_TILE_ELEMS 2
#else
#define RESIDENT_OUTPUT_TILE_ELEMS 256
#endif
#endif

#if defined(RESIDENT_V3)
#if UPMEM_GENERIC_MAX_ELEMS != 65536
#error "resident v3 profile requires UPMEM_GENERIC_MAX_ELEMS=65536"
#endif
#if RESIDENT_MRAM_POOL_BYTES != (512u * 1024u)
#error "resident v3 profile requires a 512 KiB MRAM pool"
#endif
#if RESIDENT_OUTPUT_TILE_ELEMS != 2
#error "resident v3 profile requires RESIDENT_OUTPUT_TILE_ELEMS=2"
#endif
#endif

#define UPMEM_GENERIC_MODE_INT8_SCALED 0u
#define UPMEM_GENERIC_MODE_FLOAT32_NO_QUANT 1u
#define UPMEM_GENERIC_MODE_HOST_PACKED_INT8 2u
#define RESIDENT_INVALID_SLOT 0xffffffffu
#define RESIDENT_OPERATION_CONTRACT 1u
#define RESIDENT_OPERATION_COMPLEX_COMBINE 2u
#define RESIDENT_MODE_FLOAT32 0u
#define RESIDENT_MODE_PER_TASK_REQUANTIZE 1u
#define RESIDENT_MODE_HOST_PACKED_INT8 2u
#define RESIDENT_OPERATION_ABI_V1 1u
#define RESIDENT_OPERATION_ABI_V2 2u
#ifndef RESIDENT_OPERATION_ABI_VERSION
#define RESIDENT_OPERATION_ABI_VERSION RESIDENT_OPERATION_ABI_V1
#endif
#if RESIDENT_OPERATION_ABI_VERSION != RESIDENT_OPERATION_ABI_V1 && RESIDENT_OPERATION_ABI_VERSION != RESIDENT_OPERATION_ABI_V2
#error "resident operation ABI must be version 1 or 2"
#endif
#define RESIDENT_PACKAGE_MAGIC_V1 "UPRGPCK1"
#define RESIDENT_PACKAGE_MAGIC_V2 "UPRGPCK2"
#define RESIDENT_PACKAGE_MAGIC_V3 "UPRGPCK3"
#define RESIDENT_PACKAGE_VERSION_V1 1u
#define RESIDENT_PACKAGE_VERSION_V2 2u
#define RESIDENT_PACKAGE_VERSION_V3 3u
#ifndef RESIDENT_PACKAGE_ABI_VERSION
#if defined(RESIDENT_V3)
#define RESIDENT_PACKAGE_ABI_VERSION RESIDENT_PACKAGE_VERSION_V3
#else
#define RESIDENT_PACKAGE_ABI_VERSION RESIDENT_OPERATION_ABI_VERSION
#endif
#endif
#if RESIDENT_PACKAGE_ABI_VERSION != RESIDENT_PACKAGE_VERSION_V1 && \
    RESIDENT_PACKAGE_ABI_VERSION != RESIDENT_PACKAGE_VERSION_V2 && \
    RESIDENT_PACKAGE_ABI_VERSION != RESIDENT_PACKAGE_VERSION_V3
#error "resident package ABI must be version 1, 2, or 3"
#endif
#if RESIDENT_PACKAGE_ABI_VERSION == RESIDENT_PACKAGE_VERSION_V3 && \
    RESIDENT_OPERATION_ABI_VERSION != RESIDENT_OPERATION_ABI_V2
#error "resident package ABI v3 requires operation ABI v2"
#endif
#if RESIDENT_PACKAGE_ABI_VERSION == RESIDENT_PACKAGE_VERSION_V1
#define RESIDENT_PACKAGE_MAGIC RESIDENT_PACKAGE_MAGIC_V1
#define RESIDENT_PACKAGE_VERSION RESIDENT_PACKAGE_VERSION_V1
#elif RESIDENT_PACKAGE_ABI_VERSION == RESIDENT_PACKAGE_VERSION_V2
#define RESIDENT_PACKAGE_MAGIC RESIDENT_PACKAGE_MAGIC_V2
#define RESIDENT_PACKAGE_VERSION RESIDENT_PACKAGE_VERSION_V2
#else
#define RESIDENT_PACKAGE_MAGIC RESIDENT_PACKAGE_MAGIC_V3
#define RESIDENT_PACKAGE_VERSION RESIDENT_PACKAGE_VERSION_V3
#endif
#if RESIDENT_OPERATION_ABI_VERSION == RESIDENT_OPERATION_ABI_V1
#define RESIDENT_OPERATION_BYTES 784u
#else
#define RESIDENT_OPERATION_BYTES 800u
#endif
#define RESIDENT_PACKAGE_ENDIAN 0x01020304u
#define RESIDENT_PACKAGE_FLAG_PACKED_INT8 0x00000001u
#define RESIDENT_SLOT_ID_MASK 0x3fffffffu
#define RESIDENT_SLOT_INITIAL_FLAG 0x40000000u
#define RESIDENT_SLOT_FINAL_FLAG 0x80000000u
#define RESIDENT_STORAGE_FLOAT32 1u
#define RESIDENT_STORAGE_PACKED_INT8 2u
#define RESIDENT_STORAGE_INT32 3u
#define RESIDENT_PACKED_INT8_MAX_ABS 127u
#define RESIDENT_PACKED_INT8_MAX_CONTRACTED 65536u
#define RESIDENT_COMPLETION_MAGIC 0x52534350u
#ifndef RESIDENT_COMPLETION_VERSION
#define RESIDENT_COMPLETION_VERSION 1u
#endif
#define RESIDENT_COMPLETION_PENDING 0u
#define RESIDENT_COMPLETION_COMPLETED 1u
#define RESIDENT_CONTROL_FLAG_CONTRACTED_FINAL_REFERENCE_VALIDATION_ONLY 0x00000001u
#define RESIDENT_CHECKSUM_FNV1A64_OFFSET_BASIS 14695981039346656037ULL

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
} upmem_generic_args_v1_t;

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
    /* v2 only: distributed execution range, never implicit zero/full. */
    uint32_t dpu_slice_offset;
    uint32_t dpu_slice_elements;
    uint32_t contracted_offset;
    uint32_t contracted_elements_slice;
} upmem_generic_args_v2_t;

typedef struct {
    uint32_t slot_id;
    uint32_t offset_bytes;
    uint32_t capacity_elements;
    uint32_t element_count;
} resident_slot_descriptor_v1_t;

typedef struct {
    uint32_t slot_id;
    uint32_t offset_bytes;
    uint32_t capacity_elements;
    uint32_t element_count;
    uint32_t element_bytes;
    uint32_t storage_kind;
    uint32_t logical_bytes;
    uint32_t transfer_bytes;
} resident_slot_descriptor_v3_t;

#if RESIDENT_PACKAGE_ABI_VERSION == RESIDENT_PACKAGE_VERSION_V3
typedef resident_slot_descriptor_v3_t resident_slot_descriptor_t;
#else
typedef resident_slot_descriptor_v1_t resident_slot_descriptor_t;
#endif

typedef struct {
    uint32_t slot_count;
    uint32_t operation_count;
    uint32_t pool_bytes;
    uint32_t reserved;
} resident_control_t;

/* Completion ABI is independent of the package/operation ABI above. */
typedef struct {
    uint32_t magic;
    uint32_t version;
    uint32_t active_operation_index;
    uint32_t completion_status;
    uint32_t completed_operation_count;
    uint32_t output_elements_processed;
    uint64_t output_checksum_fnv1a64;
    /* This field is part of both completion ABIs. */
    uint64_t dpu_run_time_cycles;
#if RESIDENT_COMPLETION_VERSION >= 2
#if RESIDENT_COMPLETION_VERSION >= 3
    uint32_t tasklet_processed_elements[24];
#else
    uint32_t tasklet_processed_elements[16];
#endif
    uint32_t active_tasklet_count;
    uint32_t tasklet_min_processed_elements;
    uint32_t tasklet_max_processed_elements;
#endif
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
    upmem_generic_args_v1_t args;
} resident_operation_v1_t;

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
    upmem_generic_args_v2_t args;
} resident_operation_v2_t;

#if RESIDENT_OPERATION_ABI_VERSION == RESIDENT_OPERATION_ABI_V1
typedef upmem_generic_args_v1_t upmem_generic_args_t;
typedef resident_operation_v1_t resident_operation_t;
#else
typedef upmem_generic_args_v2_t upmem_generic_args_t;
typedef resident_operation_v2_t resident_operation_t;
#endif

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

_Static_assert(sizeof(upmem_generic_args_v1_t) == 740u, "resident generic args v1 ABI drifted");
_Static_assert(sizeof(upmem_generic_args_v2_t) == 756u, "resident generic args v2 ABI drifted");
#ifndef __DPU__
_Static_assert(offsetof(upmem_generic_args_v1_t, left_rank) == 0u, "resident generic args v1 offset drifted");
_Static_assert(offsetof(upmem_generic_args_v1_t, contracted_to_right_axes) == 676u, "resident generic args v1 offset drifted");
_Static_assert(offsetof(upmem_generic_args_v2_t, dpu_slice_offset) == 740u, "resident generic args v2 slice offset drifted");
_Static_assert(offsetof(upmem_generic_args_v2_t, contracted_elements_slice) == 752u, "resident generic args v2 slice offset drifted");
#endif
_Static_assert(sizeof(resident_operation_v1_t) == 784u, "resident operation v1 ABI drifted");
_Static_assert(sizeof(resident_operation_v2_t) == 800u, "resident operation v2 ABI drifted");
_Static_assert(sizeof(resident_slot_descriptor_v1_t) == 16u, "resident slot v1 ABI drifted");
_Static_assert(sizeof(resident_slot_descriptor_v3_t) == 32u, "resident slot v3 ABI drifted");
_Static_assert(
    (uint64_t)RESIDENT_PACKED_INT8_MAX_CONTRACTED *
        (uint64_t)RESIDENT_PACKED_INT8_MAX_ABS *
        (uint64_t)RESIDENT_PACKED_INT8_MAX_ABS <= INT32_MAX,
    "packed int8 accumulation exceeds int32"
);
#ifndef __DPU__
_Static_assert(offsetof(resident_operation_v1_t, args) == 44u, "resident operation v1 args offset drifted");
_Static_assert(offsetof(resident_operation_v2_t, args) == 44u, "resident operation v2 args offset drifted");
_Static_assert(offsetof(resident_operation_v2_t, args) + offsetof(upmem_generic_args_v2_t, dpu_slice_offset) == 784u, "resident operation v2 slice offset drifted");
_Static_assert(offsetof(resident_operation_v2_t, args) + offsetof(upmem_generic_args_v2_t, contracted_elements_slice) == 796u, "resident operation v2 slice offset drifted");
#endif
_Static_assert(sizeof(resident_package_header_t) == 96u, "resident package header ABI drifted");
#if RESIDENT_COMPLETION_VERSION == 1
_Static_assert(sizeof(resident_completion_t) == 40u, "resident completion v1 ABI drifted");
#elif RESIDENT_COMPLETION_VERSION == 2
_Static_assert(sizeof(resident_completion_t) == 120u, "resident completion v2 ABI drifted");
#elif RESIDENT_COMPLETION_VERSION == 3
_Static_assert(sizeof(resident_completion_t) == 152u, "resident completion v3 ABI drifted");
#else
#error "unsupported resident completion ABI"
#endif

#endif
