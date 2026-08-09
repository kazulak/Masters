#ifndef UPMEM_SDK_GENERIC_LOOP_RESIDENT_SESSION_PROTOCOL_H
#define UPMEM_SDK_GENERIC_LOOP_RESIDENT_SESSION_PROTOCOL_H

#include <stddef.h>
#include <stdint.h>

#include "common.h"

#define RESIDENT_SESSION_SCHEMA "generic_loop_resident_graph_session_v1"
#define RESIDENT_REQUEST_KIND "resident_graph_request"
#define RESIDENT_RESPONSE_KIND "resident_graph_response"
#define RESIDENT_MAX_PATH 4096u
#define RESIDENT_ROUTE_ID "upmem_tn_hardware_taskgraph_resident"
#define RESIDENT_BACKEND_ID "upmem_sdk_hardware_taskgraph_resident"
#define RESIDENT_PROFILE_VERSION "hardware_taskgraph_single_dpu_mram_resident_v1"
#define RESIDENT_ALLOCATION_PROFILE "backend=hw"
#define RESIDENT_TARGET "hardware"

typedef struct {
    uint32_t slot_id;
    uint32_t elements;
    char *path;
    size_t raw_bytes;
    size_t transfer_bytes;
} resident_input_file_t;

typedef struct {
    char *component;
    uint32_t slot_id;
    uint32_t elements;
    char *path;
    size_t raw_bytes;
    size_t transfer_bytes;
    int status;
} resident_final_file_t;

typedef struct {
    char *session_id;
    char *dpu_binary_path;
    resident_package_header_t header;
    resident_slot_descriptor_t *slots;
    resident_operation_t *operations;
    resident_input_file_t *inputs;
    resident_final_file_t *final_outputs;
    size_t input_count;
    size_t final_count;
    uint32_t logical_task_count;
    uint32_t requested_dpus;
    char *manifest_root;
    uint32_t *slot_flags;
    char *package_path;
    char *route_id;
    char *backend_id;
    char *profile_version;
    char *allocation_profile;
    char *quantization_mode;
} resident_request_t;

typedef struct {
    double package_parse_time_s;
    double allocation_time_s;
    double binary_load_time_s;
    double initial_h2d_time_s;
    double descriptor_h2d_time_s;
    double control_h2d_time_s;
    double kernel_time_s;
    double final_d2h_time_s;
    double output_write_time_s;
    double release_time_s;
} resident_timing_t;

int resident_request_load(
    const char *manifest_path,
    resident_request_t *request,
    char **error_message
);

int resident_request_load_execution_plan(
    const char *manifest_path,
    resident_request_t *request,
    char **error_message
);

void resident_request_free(resident_request_t *request);

int resident_response_write(
    const char *response_path,
    const resident_request_t *request,
    const char *status,
    const char *failure_stage,
    const char *error_message,
    uint32_t allocated_dpus,
    int sdk_error_code,
    const resident_timing_t *timing,
    uint32_t native_launch_count,
    int release_confirmed,
    uint64_t initial_h2d_bytes,
    uint64_t descriptor_h2d_bytes,
    uint64_t control_h2d_bytes,
    uint64_t final_d2h_bytes
);

#endif
