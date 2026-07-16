#ifndef UPMEM_SDK_GENERIC_LOOP_SESSION_PROTOCOL_H
#define UPMEM_SDK_GENERIC_LOOP_SESSION_PROTOCOL_H

#include <stddef.h>
#include <stdint.h>

#include "common.h"

/*
 * Persistent host protocol, upmem_generic_session_v1.
 *
 * Input is a JSON object with schema_version, manifest_kind,
 * dpu_binary, requested_dpus=1, tasklets=1, and an ordered tasks array.
 * Each task names relative args_path, left_path, right_path, and output_path
 * files. Paths are resolved relative to the input manifest and cannot contain
 * an absolute path or a '..' component. The host allocates and loads the DPU
 * set once, then processes tasks in array order with synchronous launches.
 *
 * The response is a JSON object with one result per task. Session allocation,
 * binary-load, batch, and release timings are session fields; transfer,
 * launch, copy-back, and output-write timings are task fields. A failed task
 * stops the batch and later tasks are reported as not_run. There is no fallback
 * executor in this protocol.
 */

#define UPMEM_GENERIC_SESSION_SCHEMA "upmem_generic_session_v1"
#define UPMEM_GENERIC_SESSION_INPUT_KIND "upmem_generic_session_input"
#define UPMEM_GENERIC_SESSION_OUTPUT_KIND "upmem_generic_session_response"
#define UPMEM_GENERIC_SESSION_MAX_TASKS 1024u

typedef struct {
    double input_read_time_s;
    double h2d_time_s;
    double kernel_time_s;
    double d2h_time_s;
    double output_write_time_s;
    double total_time_s;
} upmem_generic_session_task_timing;

typedef struct {
    char *task_id;
    char *args_ref;
    char *left_ref;
    char *right_ref;
    char *output_ref;
    char *args_path;
    char *left_path;
    char *right_path;
    char *output_path;
    upmem_generic_args_t args;
    size_t left_bytes;
    size_t right_bytes;
    size_t output_bytes;
    size_t left_transfer_bytes;
    size_t right_transfer_bytes;
    size_t output_transfer_bytes;
    int result_status;
    char failure_stage[64];
    int sdk_error_code;
    upmem_generic_session_task_timing timing;
} upmem_generic_session_task;

typedef struct {
    char *session_id;
    char *dpu_binary_ref;
    char *dpu_binary_path;
    upmem_generic_session_task *tasks;
    size_t task_count;
    uint32_t requested_dpus;
    uint32_t tasklets;
} upmem_generic_session;

enum {
    UPMEM_GENERIC_SESSION_TASK_NOT_RUN = 0,
    UPMEM_GENERIC_SESSION_TASK_COMPLETED = 1,
    UPMEM_GENERIC_SESSION_TASK_FAILED = 2,
};

int upmem_generic_session_load(
    const char *manifest_path,
    upmem_generic_session *session,
    char **error_message
);

void upmem_generic_session_free(upmem_generic_session *session);

int upmem_generic_session_write_response(
    const char *response_path,
    const upmem_generic_session *session,
    const char *status,
    const char *failure_stage,
    const char *error_message,
    uint32_t allocated_dpus,
    int sdk_error_code,
    double allocation_time_s,
    double binary_load_time_s,
    double batch_time_s,
    double release_time_s
);

int upmem_generic_session_write_error_response(
    const char *response_path,
    const char *failure_stage,
    const char *error_message
);

#endif
