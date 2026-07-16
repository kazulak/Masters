#include <dpu.h>
#include <ctype.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#include "common.h"
#include "session_protocol.h"

#ifndef UPMEM_GENERIC_HARDWARE_MVP
#define UPMEM_GENERIC_HARDWARE_MVP 0
#endif

#if UPMEM_GENERIC_HARDWARE_MVP
#define UPMEM_GENERIC_ALLOCATION_PROFILE "backend=hw"
#else
#define UPMEM_GENERIC_ALLOCATION_PROFILE NULL
#endif

typedef struct {
    double allocation_time_s;
    double binary_load_time_s;
    double h2d_time_s;
    double kernel_time_s;
    double d2h_time_s;
    double output_write_time_s;
} upmem_generic_timing_t;

static double now_s(void) {
    struct timeval value;
    gettimeofday(&value, NULL);
    return (double)value.tv_sec + (double)value.tv_usec / 1000000.0;
}

static void write_status(
    const char *stage,
    int success,
    uint32_t requested,
    uint32_t allocated,
    int sdk_error_code,
    const upmem_generic_timing_t *timing
) {
    const char *path = getenv("UPMEM_GENERIC_STATUS_JSON");
    const char *profile_json = UPMEM_GENERIC_ALLOCATION_PROFILE == NULL
        ? "null" : "\"backend=hw\"";
    const upmem_generic_timing_t empty = {0};
    const upmem_generic_timing_t *current = timing == NULL ? &empty : timing;
    if (path == NULL || path[0] == '\0') {
        return;
    }
    FILE *file = fopen(path, "wb");
    if (file == NULL) {
        return;
    }
    if (stage == NULL) {
        fprintf(file,
            "{\"requested_dpus\":%u,\"allocated_dpus\":%u,\"tasklets\":%u,"
            "\"success\":%s,\"failure_stage\":null,\"allocation_profile\":%s,"
            "\"sdk_error_code\":%d,\"allocation_time_s\":%.9f,"
            "\"binary_load_time_s\":%.9f,\"h2d_time_s\":%.9f,"
            "\"kernel_time_s\":%.9f,\"d2h_time_s\":%.9f,"
            "\"output_write_time_s\":%.9f}\n",
            requested, allocated, (unsigned)NR_TASKLETS, success ? "true" : "false",
            profile_json, sdk_error_code, current->allocation_time_s,
            current->binary_load_time_s, current->h2d_time_s,
            current->kernel_time_s, current->d2h_time_s,
            current->output_write_time_s
        );
    } else {
        fprintf(file,
            "{\"requested_dpus\":%u,\"allocated_dpus\":%u,\"tasklets\":%u,"
            "\"success\":%s,\"failure_stage\":\"%s\",\"allocation_profile\":%s,"
            "\"sdk_error_code\":%d,\"allocation_time_s\":%.9f,"
            "\"binary_load_time_s\":%.9f,\"h2d_time_s\":%.9f,"
            "\"kernel_time_s\":%.9f,\"d2h_time_s\":%.9f,"
            "\"output_write_time_s\":%.9f}\n",
            requested, allocated, (unsigned)NR_TASKLETS, success ? "true" : "false",
            stage, profile_json, sdk_error_code, current->allocation_time_s,
            current->binary_load_time_s, current->h2d_time_s,
            current->kernel_time_s, current->d2h_time_s,
            current->output_write_time_s
        );
    }
    fclose(file);
}

static void report_sdk_error(const char *operation, dpu_error_t error) {
    fprintf(stderr, "%s failed: %s\n", operation, dpu_error_to_string(error));
}

static int read_exact(const char *path, void *buffer, size_t bytes) {
    FILE *file = fopen(path, "rb");
    if (file == NULL) {
        perror(path);
        return 1;
    }
    size_t count = fread(buffer, 1, bytes, file);
    fclose(file);
    if (count != bytes) {
        fprintf(stderr, "short read from %s: expected %zu bytes, got %zu\n", path, bytes, count);
        return 1;
    }
    return 0;
}

static int write_exact(const char *path, const void *buffer, size_t bytes) {
    FILE *file = fopen(path, "wb");
    if (file == NULL) {
        perror(path);
        return 1;
    }
    size_t count = fwrite(buffer, 1, bytes, file);
    fclose(file);
    if (count != bytes) {
        fprintf(stderr, "short write to %s: expected %zu bytes, got %zu\n", path, bytes, count);
        return 1;
    }
    return 0;
}

static int write_transfer_accounting(
    const char *path,
    size_t left_bytes,
    size_t right_bytes,
    size_t output_bytes,
    size_t left_transfer_bytes,
    size_t right_transfer_bytes,
    size_t output_transfer_bytes,
    size_t control_bytes
) {
    const size_t prepared_payload_h2d_bytes = left_bytes + right_bytes;
    const size_t prepared_payload_d2h_bytes = output_bytes;
    const size_t h2d_alignment_padding_bytes =
        (left_transfer_bytes - left_bytes) + (right_transfer_bytes - right_bytes);
    const size_t d2h_alignment_padding_bytes = output_transfer_bytes - output_bytes;
    const size_t alignment_padding_bytes =
        h2d_alignment_padding_bytes + d2h_alignment_padding_bytes;
    const size_t sdk_observed_h2d_bytes =
        control_bytes + left_transfer_bytes + right_transfer_bytes;
    const size_t sdk_observed_d2h_bytes = output_transfer_bytes;
    FILE *file = fopen(path, "w");
    if (file == NULL) {
        perror(path);
        return 1;
    }

    fprintf(file,
        "{\n"
        "  \"schema_version\": \"upmem_sdk_generic_loop_transfer_accounting_v1\",\n"
        "  \"transfer_accounting_scope\": \"application_visible_sdk_recorded\",\n"
        "  \"physical_bus_bytes_available\": false,\n"
        "  \"prepared_payload_h2d_bytes\": %zu,\n"
        "  \"prepared_payload_d2h_bytes\": %zu,\n"
        "  \"sdk_observed_h2d_bytes\": %zu,\n"
        "  \"sdk_observed_d2h_bytes\": %zu,\n"
        "  \"actual_h2d_bytes\": %zu,\n"
        "  \"actual_d2h_bytes\": %zu,\n"
        "  \"actual_transfer_bytes\": %zu,\n"
        "  \"actual_transfer_bytes_invariant\": \"passed\",\n"
        "  \"control_bytes\": %zu,\n"
        "  \"control_argument_bytes\": %zu,\n"
        "  \"alignment_padding_bytes\": %zu,\n"
        "  \"h2d_alignment_padding_bytes\": %zu,\n"
        "  \"d2h_alignment_padding_bytes\": %zu,\n"
        "  \"alignment_model\": {\n"
        "    \"boundary_bytes\": 8,\n"
        "    \"control_argument_transfer\": \"exact_sizeof_args_no_padding_claimed\",\n"
        "    \"payload_transfer\": \"each_payload_buffer_rounded_up_to_8_bytes\",\n"
        "    \"physical_bus_padding_observed\": false\n"
        "  },\n"
        "  \"transfer_components\": {\n"
        "    \"h2d_application_visible_payload_bytes\": %zu,\n"
        "    \"d2h_application_visible_payload_bytes\": %zu,\n"
        "    \"control_structure_bytes\": %zu,\n"
        "    \"alignment_padding_bytes\": %zu,\n"
        "    \"unobserved_sdk_overhead_bytes\": null\n"
        "  },\n"
        "  \"sdk_observed_bytes_definition\": \"sum_of_lengths_passed_to_dpu_broadcast_to_and_dpu_copy_from\"\n"
        "}\n",
        prepared_payload_h2d_bytes,
        prepared_payload_d2h_bytes,
        sdk_observed_h2d_bytes,
        sdk_observed_d2h_bytes,
        sdk_observed_h2d_bytes,
        sdk_observed_d2h_bytes,
        sdk_observed_h2d_bytes + sdk_observed_d2h_bytes,
        control_bytes,
        control_bytes,
        alignment_padding_bytes,
        h2d_alignment_padding_bytes,
        d2h_alignment_padding_bytes,
        prepared_payload_h2d_bytes,
        prepared_payload_d2h_bytes,
        control_bytes,
        alignment_padding_bytes
    );
    int failed = ferror(file) != 0;
    if (fclose(file) != 0) {
        failed = 1;
    }
    if (failed) {
        fprintf(stderr, "failed to write transfer accounting JSON to %s\n", path);
        return 1;
    }
    return 0;
}

static size_t align8(size_t bytes) {
    return (bytes + 7u) & ~((size_t)7u);
}

static int transfer_sizes(uint32_t elements, size_t element_size, size_t *bytes, size_t *transfer_bytes) {
    if ((size_t)elements > SIZE_MAX / element_size) {
        return 1;
    }
    *bytes = (size_t)elements * element_size;
    if (*bytes > SIZE_MAX - 7u) {
        return 1;
    }
    *transfer_bytes = align8(*bytes);
    return *transfer_bytes == 0 || (*transfer_bytes % 8u) != 0;
}

static int validate_row_major(const uint32_t *shape, const uint32_t *strides, uint32_t rank, uint32_t expected_elements) {
    uint64_t product = 1;
    uint64_t expected_stride = 1;
    for (uint32_t reverse_axis = 0; reverse_axis < rank; reverse_axis++) {
        const uint32_t axis = rank - reverse_axis - 1u;
        if (shape[axis] == 0 || strides[axis] != expected_stride) {
            return 1;
        }
        product *= shape[axis];
        expected_stride *= shape[axis];
        if (product > UINT32_MAX || expected_stride > UINT32_MAX) {
            return 1;
        }
    }
    return product != expected_elements;
}

static int validate_index_maps(const upmem_generic_args_t *args) {
    uint8_t left_used[UPMEM_GENERIC_MAX_RANK] = {0};
    uint8_t right_used[UPMEM_GENERIC_MAX_RANK] = {0};

    for (uint32_t output_axis = 0; output_axis < args->output_rank; output_axis++) {
        const int32_t left_axis = args->output_to_left_axes[output_axis];
        const int32_t right_axis = args->output_to_right_axes[output_axis];
        if (left_axis < -1 || right_axis < -1 ||
            left_axis >= (int32_t)args->left_rank || right_axis >= (int32_t)args->right_rank ||
            (left_axis < 0 && right_axis < 0)) {
            return 1;
        }
        if (left_axis >= 0) {
            if (left_used[left_axis] || args->left_shape[left_axis] != args->output_shape[output_axis]) {
                return 1;
            }
            left_used[left_axis] = 1;
        }
        if (right_axis >= 0) {
            if (right_used[right_axis] || args->right_shape[right_axis] != args->output_shape[output_axis]) {
                return 1;
            }
            right_used[right_axis] = 1;
        }
    }

    for (uint32_t contracted_axis = 0; contracted_axis < args->contracted_rank; contracted_axis++) {
        const int32_t left_axis = args->contracted_to_left_axes[contracted_axis];
        const int32_t right_axis = args->contracted_to_right_axes[contracted_axis];
        if (left_axis < 0 || right_axis < 0 ||
            left_axis >= (int32_t)args->left_rank || right_axis >= (int32_t)args->right_rank ||
            left_used[left_axis] || right_used[right_axis] ||
            args->contracted_dims[contracted_axis] == 0 ||
            args->left_shape[left_axis] != args->contracted_dims[contracted_axis] ||
            args->right_shape[right_axis] != args->contracted_dims[contracted_axis]) {
            return 1;
        }
        left_used[left_axis] = 1;
        right_used[right_axis] = 1;
    }

    for (uint32_t axis = 0; axis < args->left_rank; axis++) {
        if (!left_used[axis]) {
            return 1;
        }
    }
    for (uint32_t axis = 0; axis < args->right_rank; axis++) {
        if (!right_used[axis]) {
            return 1;
        }
    }
    return 0;
}

static int validate_session_task(upmem_generic_session_task *task) {
    const upmem_generic_args_t *args = &task->args;
    uint64_t contracted_product = 1u;
    const int float32_mode = args->operand_mode == UPMEM_GENERIC_MODE_FLOAT32_NO_QUANT;
    const size_t input_elem_size = float32_mode ? sizeof(float) : sizeof(int8_t);
    const size_t output_elem_size = float32_mode ? sizeof(float) : sizeof(int32_t);

    if (args->left_rank > UPMEM_GENERIC_MAX_RANK ||
        args->right_rank > UPMEM_GENERIC_MAX_RANK ||
        args->output_rank > UPMEM_GENERIC_MAX_RANK ||
        args->contracted_rank > UPMEM_GENERIC_MAX_RANK ||
        args->left_elems == 0 || args->right_elems == 0 ||
        args->output_elems == 0 || args->contracted_elems == 0 ||
        args->left_elems > UPMEM_GENERIC_MAX_ELEMS ||
        args->right_elems > UPMEM_GENERIC_MAX_ELEMS ||
        args->output_elems > UPMEM_GENERIC_MAX_ELEMS ||
        args->contracted_elems > UPMEM_GENERIC_MAX_ELEMS ||
        (args->operand_mode != UPMEM_GENERIC_MODE_INT8_SCALED &&
         args->operand_mode != UPMEM_GENERIC_MODE_FLOAT32_NO_QUANT) ||
        validate_row_major(args->left_shape, args->left_strides, args->left_rank, args->left_elems) != 0 ||
        validate_row_major(args->right_shape, args->right_strides, args->right_rank, args->right_elems) != 0 ||
        validate_index_maps(args) != 0) {
        return 1;
    }
    for (uint32_t axis = 0; axis < args->contracted_rank; axis++) {
        if (args->contracted_dims[axis] == 0 ||
            contracted_product > UINT32_MAX / args->contracted_dims[axis]) {
            return 1;
        }
        contracted_product *= args->contracted_dims[axis];
    }
    if (contracted_product != args->contracted_elems ||
        validate_row_major(args->output_shape, args->output_strides,
                           args->output_rank, args->output_elems) != 0) {
        return 1;
    }
    return transfer_sizes(args->left_elems, input_elem_size,
                          &task->left_bytes, &task->left_transfer_bytes) != 0 ||
        transfer_sizes(args->right_elems, input_elem_size,
                       &task->right_bytes, &task->right_transfer_bytes) != 0 ||
        transfer_sizes(args->output_elems, output_elem_size,
                       &task->output_bytes, &task->output_transfer_bytes) != 0;
}

static void mark_session_task_failure(
    upmem_generic_session_task *task,
    const char *stage,
    int sdk_error_code
) {
    task->result_status = UPMEM_GENERIC_SESSION_TASK_FAILED;
    task->sdk_error_code = sdk_error_code;
    snprintf(task->failure_stage, sizeof(task->failure_stage), "%s", stage);
}

static int run_session(const char *manifest_path, const char *response_path) {
    upmem_generic_session session;
    char *manifest_error = NULL;
    struct dpu_set_t set;
    struct dpu_set_t dpu;
    dpu_error_t error = DPU_OK;
    uint32_t allocated_dpus = 0u;
    int set_allocated = 0;
    int sdk_error_code = -1;
    const char *failure_stage = NULL;
    const char *failure_message = NULL;
    double allocation_time_s = 0.0;
    double binary_load_time_s = 0.0;
    double batch_time_s = 0.0;
    double release_time_s = 0.0;

    if (upmem_generic_session_load(manifest_path, &session, &manifest_error) != 0) {
        int response_rc = upmem_generic_session_write_error_response(
            response_path, "manifest_parse_failed",
            manifest_error ? manifest_error : "invalid session manifest"
        );
        free(manifest_error);
        return response_rc == 0 ? 2 : 1;
    }

#if NR_TASKLETS != 1
    failure_stage = "hardware_profile_violation";
    failure_message = "persistent generic session requires NR_TASKLETS=1";
    goto session_response;
#endif

    /*
     * Preflight every args.bin before allocating a DPU. This makes malformed
     * later tasks deterministic and prevents a partial batch from consuming a
     * physical DPU when no task could have run.
     */
    for (size_t index = 0; index < session.task_count; index++) {
        upmem_generic_session_task *task = &session.tasks[index];
        if (read_exact(task->args_path, &task->args, sizeof(task->args)) != 0) {
            mark_session_task_failure(task, "argument_transfer_failed", -1);
            failure_stage = "argument_transfer_failed";
            failure_message = "session task args.bin could not be read";
            goto session_response;
        }
        if (validate_session_task(task) != 0) {
            mark_session_task_failure(task, "hardware_profile_violation", -1);
            failure_stage = "hardware_profile_violation";
            failure_message = "session task generic contraction metadata is invalid";
            goto session_response;
        }
    }

    {
        const double started = now_s();
        error = dpu_alloc(session.requested_dpus, UPMEM_GENERIC_ALLOCATION_PROFILE, &set);
        allocation_time_s = now_s() - started;
    }
    if (error != DPU_OK) {
        report_sdk_error("dpu_alloc", error);
        sdk_error_code = (int)error;
        failure_stage = error == DPU_ERR_INVALID_PROFILE
            ? "hardware_profile_violation" : "hardware_allocation_failed";
        failure_message = "persistent session DPU allocation failed";
        goto session_response;
    }
    set_allocated = 1;
    error = dpu_get_nr_dpus(set, &allocated_dpus);
    if (error != DPU_OK || allocated_dpus != session.requested_dpus) {
        if (error != DPU_OK) {
            report_sdk_error("dpu_get_nr_dpus", error);
            sdk_error_code = (int)error;
        }
        failure_stage = "hardware_allocation_failed";
        failure_message = "persistent session did not receive exactly one DPU";
        goto session_release;
    }
    {
        const double started = now_s();
        error = dpu_load(set, session.dpu_binary_path, NULL);
        binary_load_time_s = now_s() - started;
    }
    if (error != DPU_OK) {
        report_sdk_error("dpu_load", error);
        sdk_error_code = (int)error;
        failure_stage = "binary_load_failed";
        failure_message = "persistent session DPU binary load failed";
        goto session_release;
    }

    {
        const double batch_started = now_s();
        for (size_t index = 0; index < session.task_count; index++) {
            upmem_generic_session_task *task = &session.tasks[index];
            unsigned char *left = NULL;
            unsigned char *right = NULL;
            unsigned char *output = NULL;
            const double task_started = now_s();
            double stage_started;

            left = (unsigned char *)calloc(task->left_transfer_bytes, 1u);
            right = (unsigned char *)calloc(task->right_transfer_bytes, 1u);
            output = (unsigned char *)calloc(task->output_transfer_bytes, 1u);
            if (left == NULL || right == NULL || output == NULL) {
                task->timing.total_time_s = now_s() - task_started;
                mark_session_task_failure(task, "hardware_allocation_failed", -1);
                failure_stage = "hardware_allocation_failed";
                failure_message = "persistent session task buffers could not be allocated";
                free(left); free(right); free(output);
                break;
            }
            stage_started = now_s();
            if (read_exact(task->left_path, left, task->left_bytes) != 0 ||
                read_exact(task->right_path, right, task->right_bytes) != 0) {
                task->timing.input_read_time_s = now_s() - stage_started;
                task->timing.total_time_s = now_s() - task_started;
                mark_session_task_failure(task, "operand_transfer_failed", -1);
                failure_stage = "operand_transfer_failed";
                failure_message = "persistent session task operand could not be read";
                free(left); free(right); free(output);
                break;
            }
            task->timing.input_read_time_s = now_s() - stage_started;

            stage_started = now_s();
            error = dpu_broadcast_to(set, "GENERIC_ARGS", 0, &task->args,
                                     sizeof(task->args), DPU_XFER_DEFAULT);
            if (error == DPU_OK) {
                error = dpu_broadcast_to(set, "GENERIC_A_RAW", 0, left,
                                         task->left_transfer_bytes, DPU_XFER_DEFAULT);
            }
            if (error == DPU_OK) {
                error = dpu_broadcast_to(set, "GENERIC_B_RAW", 0, right,
                                         task->right_transfer_bytes, DPU_XFER_DEFAULT);
            }
            task->timing.h2d_time_s = now_s() - stage_started;
            if (error != DPU_OK) {
                report_sdk_error("persistent task operand transfer", error);
                sdk_error_code = (int)error;
                task->timing.total_time_s = now_s() - task_started;
                mark_session_task_failure(task, "operand_transfer_failed", sdk_error_code);
                failure_stage = "operand_transfer_failed";
                failure_message = "persistent session task H2D transfer failed";
                free(left); free(right); free(output);
                break;
            }

            stage_started = now_s();
            error = dpu_launch(set, DPU_SYNCHRONOUS);
            task->timing.kernel_time_s = now_s() - stage_started;
            if (error != DPU_OK) {
                report_sdk_error("persistent task dpu_launch", error);
                sdk_error_code = (int)error;
                task->timing.total_time_s = now_s() - task_started;
                mark_session_task_failure(task, "kernel_launch_failed", sdk_error_code);
                failure_stage = "kernel_launch_failed";
                failure_message = "persistent session synchronous launch failed";
                free(left); free(right); free(output);
                break;
            }

            stage_started = now_s();
            DPU_FOREACH(set, dpu) {
                error = dpu_copy_from(dpu, "GENERIC_C_RAW", 0, output,
                                       task->output_transfer_bytes);
                break;
            }
            task->timing.d2h_time_s = now_s() - stage_started;
            if (error != DPU_OK) {
                report_sdk_error("persistent task result transfer", error);
                sdk_error_code = (int)error;
                task->timing.total_time_s = now_s() - task_started;
                mark_session_task_failure(task, "result_transfer_failed", sdk_error_code);
                failure_stage = "result_transfer_failed";
                failure_message = "persistent session task D2H transfer failed";
                free(left); free(right); free(output);
                break;
            }

            stage_started = now_s();
            if (write_exact(task->output_path, output, task->output_bytes) != 0) {
                task->timing.output_write_time_s = now_s() - stage_started;
                task->timing.total_time_s = now_s() - task_started;
                mark_session_task_failure(task, "output_manifest_failed", -1);
                failure_stage = "output_manifest_failed";
                failure_message = "persistent session task output could not be written";
                free(left); free(right); free(output);
                break;
            }
            task->timing.output_write_time_s = now_s() - stage_started;
            task->timing.total_time_s = now_s() - task_started;
            task->result_status = UPMEM_GENERIC_SESSION_TASK_COMPLETED;
            task->sdk_error_code = 0;
            free(left); free(right); free(output);
        }
        batch_time_s = now_s() - batch_started;
    }

session_release:
    if (set_allocated) {
        const double started = now_s();
        error = dpu_free(set);
        release_time_s = now_s() - started;
        if (error != DPU_OK) {
            report_sdk_error("dpu_free", error);
            sdk_error_code = (int)error;
            failure_stage = "hardware_release_failed";
            failure_message = "persistent session DPU release failed";
        }
    }

session_response:
    if (failure_stage == NULL) {
        sdk_error_code = 0;
        failure_message = NULL;
    } else {
        for (size_t index = 0; index < session.task_count; index++) {
            if (session.tasks[index].result_status == UPMEM_GENERIC_SESSION_TASK_NOT_RUN) {
                snprintf(session.tasks[index].failure_stage,
                         sizeof(session.tasks[index].failure_stage),
                         "%s", "not_run_after_failure");
            }
        }
    }
    {
        const int response_rc = upmem_generic_session_write_response(
            response_path, &session, failure_stage == NULL ? "completed" : "failed",
            failure_stage, failure_message, allocated_dpus, sdk_error_code,
            allocation_time_s, binary_load_time_s, batch_time_s, release_time_s
        );
        upmem_generic_session_free(&session);
        return response_rc == 0 && failure_stage == NULL ? 0 : 1;
    }
}

static int interactive_safe_relative(const char *path) {
    const char *cursor = path;
    if (path == NULL || path[0] == '\0' || path[0] == '/') return 1;
    while (*cursor != '\0') {
        const char *start = cursor;
        while (*cursor != '\0' && *cursor != '/') cursor++;
        if ((size_t)(cursor - start) == 2u && start[0] == '.' && start[1] == '.') return 1;
        if (*cursor == '/') cursor++;
    }
    return 0;
}

static char *interactive_base(const char *manifest_path) {
    const char *slash = strrchr(manifest_path, '/');
    const size_t length = slash == NULL ? 1u : (size_t)(slash - manifest_path);
    char *base = (char *)malloc(length + 1u);
    if (base == NULL) return NULL;
    if (slash == NULL) memcpy(base, ".", 2u);
    else {
        memcpy(base, manifest_path, length);
        base[length] = '\0';
    }
    return base;
}

static char *interactive_resolve(const char *base, const char *relative) {
    const size_t base_length = strlen(base);
    const size_t relative_length = strlen(relative);
    char *resolved;
    if (interactive_safe_relative(relative) != 0) return NULL;
    resolved = (char *)malloc(base_length + 1u + relative_length + 1u);
    if (resolved == NULL) return NULL;
    memcpy(resolved, base, base_length);
    resolved[base_length] = '/';
    memcpy(resolved + base_length + 1u, relative, relative_length + 1u);
    return resolved;
}

static void interactive_json_string(const char *value) {
    const unsigned char *cursor = (const unsigned char *)(value ? value : "");
    fputc('"', stdout);
    for (; *cursor != '\0'; cursor++) {
        if (*cursor == '"' || *cursor == '\\') {
            fputc('\\', stdout);
            fputc(*cursor, stdout);
        } else if (*cursor == '\n') fputs("\\n", stdout);
        else if (*cursor == '\r') fputs("\\r", stdout);
        else if (*cursor == '\t') fputs("\\t", stdout);
        else if (*cursor < 0x20u) fprintf(stdout, "\\u%04x", (unsigned)*cursor);
        else fputc(*cursor, stdout);
    }
    fputc('"', stdout);
}

static void interactive_event(
    const char *event,
    const char *status,
    const char *failure_stage,
    const char *error_message,
    const char *response_path,
    uint32_t allocated_dpus,
    double allocation_time_s,
    double binary_load_time_s,
    double release_time_s,
    int released
) {
    printf("{\"schema_version\":");
    interactive_json_string(UPMEM_GENERIC_INTERACTIVE_SCHEMA);
    printf(",\"event\":");
    interactive_json_string(event);
    printf(",\"status\":");
    interactive_json_string(status);
    printf(",\"failure_stage\":");
    if (failure_stage) interactive_json_string(failure_stage); else fputs("null", stdout);
    printf(",\"error\":");
    if (error_message) interactive_json_string(error_message); else fputs("null", stdout);
    printf(",\"response_path\":");
    if (response_path) interactive_json_string(response_path); else fputs("null", stdout);
    printf(",\"requested_dpus\":1,\"allocated_dpus\":%u,"
           "\"allocation_time_s\":%.9f,\"binary_load_time_s\":%.9f,"
           "\"released\":%s,\"release_time_s\":%.9f}\n",
           allocated_dpus, allocation_time_s, binary_load_time_s,
           released ? "true" : "false", release_time_s);
    fflush(stdout);
}

static dpu_error_t interactive_release_and_event(
    struct dpu_set_t set,
    uint32_t allocated_dpus,
    double allocation_time_s,
    double binary_load_time_s
) {
    const double started = now_s();
    const dpu_error_t error = dpu_free(set);
    const double release_time_s = now_s() - started;
    interactive_event(
        "closed", error == DPU_OK ? "closed" : "failed",
        error == DPU_OK ? NULL : "hardware_release_failed",
        error == DPU_OK ? NULL : "interactive DPU release failed",
        NULL, allocated_dpus, allocation_time_s, binary_load_time_s,
        release_time_s, error == DPU_OK
    );
    return error;
}

static int run_interactive_request_tasks(
    struct dpu_set_t set,
    upmem_generic_session *request,
    const char **failure_stage,
    const char **failure_message
) {
    for (size_t index = 0; index < request->task_count; index++) {
        upmem_generic_session_task *task = &request->tasks[index];
        unsigned char *left = NULL;
        unsigned char *right = NULL;
        unsigned char *output = NULL;
        const double task_started = now_s();
        double stage_started;
        dpu_error_t error = DPU_OK;

        left = (unsigned char *)calloc(task->left_transfer_bytes, 1u);
        right = (unsigned char *)calloc(task->right_transfer_bytes, 1u);
        output = (unsigned char *)calloc(task->output_transfer_bytes, 1u);
        if (left == NULL || right == NULL || output == NULL) {
            mark_session_task_failure(task, "hardware_allocation_failed", -1);
            *failure_stage = "hardware_allocation_failed";
            *failure_message = "interactive request task buffers could not be allocated";
            task->timing.total_time_s = now_s() - task_started;
            free(left); free(right); free(output);
            return 1;
        }
        stage_started = now_s();
        if (read_exact(task->left_path, left, task->left_bytes) != 0 ||
            read_exact(task->right_path, right, task->right_bytes) != 0) {
            task->timing.input_read_time_s = now_s() - stage_started;
            task->timing.total_time_s = now_s() - task_started;
            mark_session_task_failure(task, "operand_transfer_failed", -1);
            *failure_stage = "operand_transfer_failed";
            *failure_message = "interactive request operands could not be read";
            free(left); free(right); free(output);
            return 1;
        }
        task->timing.input_read_time_s = now_s() - stage_started;

        stage_started = now_s();
        error = dpu_broadcast_to(set, "GENERIC_ARGS", 0, &task->args,
                                 sizeof(task->args), DPU_XFER_DEFAULT);
        if (error == DPU_OK) error = dpu_broadcast_to(
            set, "GENERIC_A_RAW", 0, left, task->left_transfer_bytes, DPU_XFER_DEFAULT
        );
        if (error == DPU_OK) error = dpu_broadcast_to(
            set, "GENERIC_B_RAW", 0, right, task->right_transfer_bytes, DPU_XFER_DEFAULT
        );
        task->timing.h2d_time_s = now_s() - stage_started;
        if (error != DPU_OK) {
            report_sdk_error("interactive request H2D", error);
            task->sdk_error_code = (int)error;
            task->timing.total_time_s = now_s() - task_started;
            mark_session_task_failure(task, "operand_transfer_failed", (int)error);
            *failure_stage = "operand_transfer_failed";
            *failure_message = "interactive request H2D transfer failed";
            free(left); free(right); free(output);
            return 1;
        }

        stage_started = now_s();
        error = dpu_launch(set, DPU_SYNCHRONOUS);
        task->timing.kernel_time_s = now_s() - stage_started;
        if (error != DPU_OK) {
            report_sdk_error("interactive request dpu_launch", error);
            task->sdk_error_code = (int)error;
            task->timing.total_time_s = now_s() - task_started;
            mark_session_task_failure(task, "kernel_launch_failed", (int)error);
            *failure_stage = "kernel_launch_failed";
            *failure_message = "interactive request synchronous launch failed";
            free(left); free(right); free(output);
            return 1;
        }

        stage_started = now_s();
        {
            struct dpu_set_t dpu;
            DPU_FOREACH(set, dpu) {
                error = dpu_copy_from(dpu, "GENERIC_C_RAW", 0, output,
                                      task->output_transfer_bytes);
                break;
            }
        }
        task->timing.d2h_time_s = now_s() - stage_started;
        if (error != DPU_OK) {
            report_sdk_error("interactive request D2H", error);
            task->sdk_error_code = (int)error;
            task->timing.total_time_s = now_s() - task_started;
            mark_session_task_failure(task, "result_transfer_failed", (int)error);
            *failure_stage = "result_transfer_failed";
            *failure_message = "interactive request D2H transfer failed";
            free(left); free(right); free(output);
            return 1;
        }

        stage_started = now_s();
        if (write_exact(task->output_path, output, task->output_bytes) != 0) {
            task->timing.output_write_time_s = now_s() - stage_started;
            task->timing.total_time_s = now_s() - task_started;
            mark_session_task_failure(task, "output_manifest_failed", -1);
            *failure_stage = "output_manifest_failed";
            *failure_message = "interactive request output could not be written";
            free(left); free(right); free(output);
            return 1;
        }
        task->timing.output_write_time_s = now_s() - stage_started;
        task->timing.total_time_s = now_s() - task_started;
        task->result_status = UPMEM_GENERIC_SESSION_TASK_COMPLETED;
        task->sdk_error_code = 0;
        free(left); free(right); free(output);
    }
    return 0;
}

static int run_interactive(const char *bootstrap_path) {
    upmem_generic_interactive_bootstrap bootstrap;
    char *manifest_error = NULL;
    char *base = NULL;
    struct dpu_set_t set;
    dpu_error_t error = DPU_OK;
    uint32_t allocated_dpus = 0u;
    int set_allocated = 0;
    double allocation_time_s = 0.0;
    double binary_load_time_s = 0.0;
    char line[4096];

    if (upmem_generic_interactive_bootstrap_load(
            bootstrap_path, &bootstrap, &manifest_error) != 0) {
        interactive_event("error", "failed", "manifest_parse_failed", manifest_error,
                          NULL, 0u, 0.0, 0.0, 0.0, 0);
        free(manifest_error);
        return 2;
    }
    base = interactive_base(bootstrap_path);
    if (base == NULL) {
        interactive_event("error", "failed", "protocol_error",
                          "interactive session root unavailable", NULL, 0u, 0.0, 0.0, 0.0, 0);
        upmem_generic_interactive_bootstrap_free(&bootstrap);
        return 2;
    }
#if NR_TASKLETS != 1
    interactive_event("error", "failed", "hardware_profile_violation",
                      "interactive session requires NR_TASKLETS=1", NULL, 0u, 0.0, 0.0, 0.0, 0);
    free(base);
    upmem_generic_interactive_bootstrap_free(&bootstrap);
    return 2;
#endif
    {
        const double started = now_s();
        error = dpu_alloc(bootstrap.requested_dpus, UPMEM_GENERIC_ALLOCATION_PROFILE, &set);
        allocation_time_s = now_s() - started;
    }
    if (error != DPU_OK) {
        report_sdk_error("interactive dpu_alloc", error);
        interactive_event("error", "failed", "hardware_allocation_failed",
                          "interactive DPU allocation failed", NULL, 0u,
                          allocation_time_s, 0.0, 0.0, 0);
        free(base);
        upmem_generic_interactive_bootstrap_free(&bootstrap);
        return 1;
    }
    set_allocated = 1;
    error = dpu_get_nr_dpus(set, &allocated_dpus);
    if (error != DPU_OK || allocated_dpus != 1u) {
        if (error != DPU_OK) report_sdk_error("interactive dpu_get_nr_dpus", error);
        interactive_event("error", "failed", "hardware_allocation_failed",
                          "interactive session did not receive exactly one DPU", NULL,
                          allocated_dpus, allocation_time_s, 0.0, 0.0, 0);
        if (set_allocated) {
            interactive_release_and_event(
                set, allocated_dpus, allocation_time_s, binary_load_time_s
            );
        }
        free(base);
        upmem_generic_interactive_bootstrap_free(&bootstrap);
        return 1;
    }
    {
        const double started = now_s();
        error = dpu_load(set, bootstrap.dpu_binary_path, NULL);
        binary_load_time_s = now_s() - started;
    }
    if (error != DPU_OK) {
        report_sdk_error("interactive dpu_load", error);
        interactive_event("error", "failed", "binary_load_failed",
                          "interactive DPU binary load failed", NULL, allocated_dpus,
                          allocation_time_s, binary_load_time_s, 0.0, 0);
        interactive_release_and_event(
            set, allocated_dpus, allocation_time_s, binary_load_time_s
        );
        free(base);
        upmem_generic_interactive_bootstrap_free(&bootstrap);
        return 1;
    }
    interactive_event("ready", "ready", NULL, NULL, NULL, allocated_dpus,
                      allocation_time_s, binary_load_time_s, 0.0, 0);

    while (fgets(line, sizeof(line), stdin) != NULL) {
        char *request_ref;
        char *response_ref;
        char *cursor;
        char *request_path = NULL;
        char *response_path = NULL;
        upmem_generic_session request;
        char *request_error = NULL;
        const char *failure_stage = NULL;
        const char *failure_message = NULL;
        double request_time_s;
        double release_time_s = 0.0;
        int request_failed = 0;

        line[strcspn(line, "\r\n")] = '\0';
        if (strcmp(line, "CLOSE") == 0) {
            const double started = now_s();
            error = dpu_free(set);
            release_time_s = now_s() - started;
            set_allocated = 0;
            if (error != DPU_OK) {
                report_sdk_error("interactive dpu_free", error);
                interactive_event("closed", "failed", "hardware_release_failed",
                                  "interactive DPU release failed", NULL,
                                  allocated_dpus, allocation_time_s, binary_load_time_s,
                                  release_time_s, 0);
                free(base);
                upmem_generic_interactive_bootstrap_free(&bootstrap);
                return 1;
            }
            interactive_event("closed", "closed", NULL, NULL, NULL,
                              allocated_dpus, allocation_time_s, binary_load_time_s,
                              release_time_s, 1);
            free(base);
            upmem_generic_interactive_bootstrap_free(&bootstrap);
            return 0;
        }
        if (strncmp(line, "REQUEST ", 8) != 0) {
            interactive_event("error", "failed", "protocol_error",
                              "expected REQUEST or CLOSE", NULL,
                              allocated_dpus, allocation_time_s, binary_load_time_s, 0.0, 0);
            interactive_release_and_event(
                set, allocated_dpus, allocation_time_s, binary_load_time_s
            );
            free(base);
            upmem_generic_interactive_bootstrap_free(&bootstrap);
            return 2;
        }
        cursor = line + 8;
        request_ref = cursor;
        while (*cursor != '\0' && !isspace((unsigned char)*cursor)) cursor++;
        if (*cursor == '\0') {
            interactive_event("error", "failed", "protocol_error",
                              "REQUEST requires request and response paths", NULL,
                              allocated_dpus, allocation_time_s, binary_load_time_s, 0.0, 0);
            interactive_release_and_event(
                set, allocated_dpus, allocation_time_s, binary_load_time_s
            );
            free(base);
            upmem_generic_interactive_bootstrap_free(&bootstrap);
            return 2;
        }
        *cursor++ = '\0';
        while (isspace((unsigned char)*cursor)) cursor++;
        response_ref = cursor;
        while (*cursor != '\0' && !isspace((unsigned char)*cursor)) cursor++;
        if (*cursor != '\0') {
            interactive_event("error", "failed", "protocol_error",
                              "REQUEST paths must be whitespace-free relative paths", NULL,
                              allocated_dpus, allocation_time_s, binary_load_time_s, 0.0, 0);
            interactive_release_and_event(
                set, allocated_dpus, allocation_time_s, binary_load_time_s
            );
            free(base);
            upmem_generic_interactive_bootstrap_free(&bootstrap);
            return 2;
        }
        request_path = interactive_resolve(base, request_ref);
        response_path = interactive_resolve(base, response_ref);
        if (request_path == NULL || response_path == NULL) {
            interactive_event("error", "failed", "path_containment_failed",
                              "interactive request paths must be safe relative paths", NULL,
                              allocated_dpus, allocation_time_s, binary_load_time_s, 0.0, 0);
            free(request_path); free(response_path);
            interactive_release_and_event(
                set, allocated_dpus, allocation_time_s, binary_load_time_s
            );
            free(base);
            upmem_generic_interactive_bootstrap_free(&bootstrap);
            return 2;
        }
        memset(&request, 0, sizeof(request));
        if (upmem_generic_interactive_request_load(
                request_path, &request, &request_error) != 0) {
            upmem_generic_interactive_request_write_error_response(
                response_path, "manifest_parse_failed",
                request_error ? request_error : "invalid interactive request manifest"
            );
            interactive_event("error", "failed", "manifest_parse_failed",
                              request_error, response_ref, allocated_dpus,
                              allocation_time_s, binary_load_time_s, 0.0, 0);
            free(request_error);
            free(request_path); free(response_path);
            interactive_release_and_event(
                set, allocated_dpus, allocation_time_s, binary_load_time_s
            );
            free(base);
            upmem_generic_interactive_bootstrap_free(&bootstrap);
            return 1;
        }
        for (size_t index = 0; index < request.task_count; index++) {
            if (read_exact(request.tasks[index].args_path, &request.tasks[index].args,
                           sizeof(request.tasks[index].args)) != 0 ||
                validate_session_task(&request.tasks[index]) != 0) {
                mark_session_task_failure(&request.tasks[index],
                                          "hardware_profile_violation", -1);
                failure_stage = "hardware_profile_violation";
                failure_message = "interactive request metadata is invalid";
                request_failed = 1;
                break;
            }
        }
        request_time_s = now_s();
        if (!request_failed) {
            request_failed = run_interactive_request_tasks(
                set, &request, &failure_stage, &failure_message
            );
        }
        request_time_s = now_s() - request_time_s;
        if (request_failed) {
            for (size_t index = 0; index < request.task_count; index++) {
                if (request.tasks[index].result_status == UPMEM_GENERIC_SESSION_TASK_NOT_RUN) {
                    snprintf(request.tasks[index].failure_stage,
                             sizeof(request.tasks[index].failure_stage),
                             "%s", "not_run_after_failure");
                }
            }
        }
        if (request_failed) {
            const double started = now_s();
            error = dpu_free(set);
            release_time_s = now_s() - started;
            set_allocated = 0;
            if (error != DPU_OK) {
                failure_stage = "hardware_release_failed";
                failure_message = "interactive DPU release failed";
            }
        }
        if (upmem_generic_interactive_request_write_response(
                response_path, &request, request_failed ? "failed" : "completed",
                request_failed ? failure_stage : NULL,
                request_failed ? failure_message : NULL,
                allocated_dpus, request_failed && error != DPU_OK ? (int)error : 0,
                allocation_time_s, binary_load_time_s, request_time_s, release_time_s
            ) != 0) {
            interactive_event("error", "failed", "response_write_failed",
                              "interactive response could not be written", response_ref,
                              allocated_dpus, allocation_time_s, binary_load_time_s,
                              release_time_s, 0);
            upmem_generic_session_free(&request);
            free(request_path); free(response_path);
            if (set_allocated) {
                interactive_release_and_event(
                    set, allocated_dpus, allocation_time_s, binary_load_time_s
                );
            }
            free(base);
            upmem_generic_interactive_bootstrap_free(&bootstrap);
            return 1;
        }
        interactive_event("response", request_failed ? "failed" : "completed",
                          request_failed ? failure_stage : NULL,
                          request_failed ? failure_message : NULL, response_ref,
                          allocated_dpus, allocation_time_s, binary_load_time_s,
                          release_time_s, request_failed ? 0 : 0);
        upmem_generic_session_free(&request);
        free(request_path); free(response_path);
        if (request_failed) {
            interactive_event("closed", error == DPU_OK ? "closed" : "failed",
                              error == DPU_OK ? NULL : "hardware_release_failed",
                              error == DPU_OK ? NULL : "interactive DPU release failed",
                              NULL, allocated_dpus, allocation_time_s, binary_load_time_s,
                              release_time_s, error == DPU_OK);
            free(base);
            upmem_generic_interactive_bootstrap_free(&bootstrap);
            return 1;
        }
    }

    if (set_allocated) {
        const double started = now_s();
        error = dpu_free(set);
        {
            const double release_time_s = now_s() - started;
            interactive_event("closed", error == DPU_OK ? "closed" : "failed",
                              error == DPU_OK ? NULL : "hardware_release_failed",
                              error == DPU_OK ? NULL : "interactive DPU release failed",
                              NULL, allocated_dpus, allocation_time_s, binary_load_time_s,
                              release_time_s, error == DPU_OK);
        }
    }
    free(base);
    upmem_generic_interactive_bootstrap_free(&bootstrap);
    return error == DPU_OK ? 0 : 1;
}

int main(int argc, char **argv) {
    if (argc == 3 &&
        strcmp(argv[1], "--interactive-session") == 0 &&
        strcmp(argv[2], "--bootstrap-manifest") == 0) {
        fprintf(stderr, "usage: %s --interactive-session --bootstrap-manifest <path>\n", argv[0]);
        return 2;
    }
    if (argc == 4 &&
        strcmp(argv[1], "--interactive-session") == 0 &&
        strcmp(argv[2], "--bootstrap-manifest") == 0) {
        return run_interactive(argv[3]);
    }
    if (argc == 5 &&
        strcmp(argv[1], "--session-manifest") == 0 &&
        strcmp(argv[3], "--response-manifest") == 0) {
        return run_session(argv[2], argv[4]);
    }

    const uint32_t requested_dpus = 1;
    uint32_t allocated_dpus = 0;
    int set_allocated = 0;
    const char *failure_stage = NULL;
    int sdk_error_code = -1;
    upmem_generic_timing_t timing = {0};
    struct dpu_set_t set;
    struct dpu_set_t dpu;
    dpu_error_t error = DPU_OK;
    unsigned char *left = NULL;
    unsigned char *right = NULL;
    unsigned char *output = NULL;

    if (argc != 6) {
        fprintf(stderr, "usage: %s <dpu_binary> <args.bin> <left_i8.bin> <right_i8.bin> <out_i32.bin>\n", argv[0]);
        write_status("hardware_profile_violation", 0, requested_dpus, 0, -1, &timing);
        return 2;
    }

    const char *dpu_binary = argv[1];
    const char *args_path = argv[2];
    const char *left_path = argv[3];
    const char *right_path = argv[4];
    const char *out_path = argv[5];
    upmem_generic_args_t args;
    if (read_exact(args_path, &args, sizeof(args)) != 0) {
        write_status("argument_transfer_failed", 0, requested_dpus, 0, -1, &timing);
        return 1;
    }
    if (args.left_rank > UPMEM_GENERIC_MAX_RANK || args.right_rank > UPMEM_GENERIC_MAX_RANK || args.output_rank > UPMEM_GENERIC_MAX_RANK || args.contracted_rank > UPMEM_GENERIC_MAX_RANK ||
        args.left_elems == 0 || args.right_elems == 0 || args.output_elems == 0 || args.contracted_elems == 0 ||
        args.left_elems > UPMEM_GENERIC_MAX_ELEMS || args.right_elems > UPMEM_GENERIC_MAX_ELEMS || args.output_elems > UPMEM_GENERIC_MAX_ELEMS || args.contracted_elems > UPMEM_GENERIC_MAX_ELEMS ||
        (args.operand_mode != UPMEM_GENERIC_MODE_INT8_SCALED &&
         args.operand_mode != UPMEM_GENERIC_MODE_FLOAT32_NO_QUANT) ||
        validate_row_major(args.left_shape, args.left_strides, args.left_rank, args.left_elems) != 0 ||
        validate_row_major(args.right_shape, args.right_strides, args.right_rank, args.right_elems) != 0 ||
        validate_index_maps(&args) != 0) {
        fprintf(stderr, "invalid generic contraction metadata\n");
        write_status("hardware_profile_violation", 0, requested_dpus, 0, -1, &timing);
        return 2;
    }
    uint64_t contracted_product = 1;
    for (uint32_t axis = 0; axis < args.contracted_rank; axis++) {
        if (args.contracted_dims[axis] == 0 || contracted_product > UINT32_MAX / args.contracted_dims[axis]) {
            write_status("hardware_profile_violation", 0, requested_dpus, 0, -1, &timing);
            return 2;
        }
        contracted_product *= args.contracted_dims[axis];
    }
    if (contracted_product != args.contracted_elems || validate_row_major(args.output_shape, args.output_strides, args.output_rank, args.output_elems) != 0) {
        write_status("hardware_profile_violation", 0, requested_dpus, 0, -1, &timing);
        return 2;
    }

    const int float32_mode = args.operand_mode == UPMEM_GENERIC_MODE_FLOAT32_NO_QUANT;
    const size_t input_elem_size = float32_mode ? sizeof(float) : sizeof(int8_t);
    const size_t output_elem_size = float32_mode ? sizeof(float) : sizeof(int32_t);
    size_t left_bytes, right_bytes, output_bytes, left_transfer_bytes, right_transfer_bytes, output_transfer_bytes;
    if (transfer_sizes(args.left_elems, input_elem_size, &left_bytes, &left_transfer_bytes) != 0 ||
        transfer_sizes(args.right_elems, input_elem_size, &right_bytes, &right_transfer_bytes) != 0 ||
        transfer_sizes(args.output_elems, output_elem_size, &output_bytes, &output_transfer_bytes) != 0) {
        write_status("hardware_profile_violation", 0, requested_dpus, 0, -1, &timing);
        return 2;
    }
    left = (unsigned char *)calloc(left_transfer_bytes, 1);
    right = (unsigned char *)calloc(right_transfer_bytes, 1);
    output = (unsigned char *)calloc(output_transfer_bytes, 1);
    if (left == NULL || right == NULL || output == NULL) {
        failure_stage = "hardware_allocation_failed";
        goto release;
    }
    if (read_exact(left_path, left, left_bytes) != 0 || read_exact(right_path, right, right_bytes) != 0) {
        failure_stage = "operand_transfer_failed";
        goto release;
    }

    double stage_started = now_s();
    error = dpu_alloc(requested_dpus, UPMEM_GENERIC_ALLOCATION_PROFILE, &set);
    timing.allocation_time_s = now_s() - stage_started;
    if (error != DPU_OK) {
        report_sdk_error("dpu_alloc", error);
        sdk_error_code = (int)error;
        failure_stage = error == DPU_ERR_INVALID_PROFILE ? "hardware_profile_violation" : "hardware_allocation_failed";
        goto release;
    }
    set_allocated = 1;
    error = dpu_get_nr_dpus(set, &allocated_dpus);
    if (error != DPU_OK || allocated_dpus != requested_dpus) {
        if (error != DPU_OK) {
            report_sdk_error("dpu_get_nr_dpus", error);
            sdk_error_code = (int)error;
        }
        failure_stage = "hardware_allocation_failed";
        goto release;
    }
    stage_started = now_s();
    error = dpu_load(set, dpu_binary, NULL);
    timing.binary_load_time_s = now_s() - stage_started;
    if (error != DPU_OK) {
        report_sdk_error("dpu_load", error);
        sdk_error_code = (int)error;
        failure_stage = "binary_load_failed";
        goto release;
    }
    stage_started = now_s();
    error = dpu_broadcast_to(set, "GENERIC_ARGS", 0, &args, sizeof(args), DPU_XFER_DEFAULT);
    if (error != DPU_OK) {
        report_sdk_error("GENERIC_ARGS transfer", error);
        sdk_error_code = (int)error;
        failure_stage = "argument_transfer_failed";
        goto release;
    }
    error = dpu_broadcast_to(set, "GENERIC_A_RAW", 0, left, left_transfer_bytes, DPU_XFER_DEFAULT);
    if (error == DPU_OK) error = dpu_broadcast_to(set, "GENERIC_B_RAW", 0, right, right_transfer_bytes, DPU_XFER_DEFAULT);
    timing.h2d_time_s = now_s() - stage_started;
    if (error != DPU_OK) {
        report_sdk_error("generic operand transfer", error);
        sdk_error_code = (int)error;
        failure_stage = "operand_transfer_failed";
        goto release;
    }
    stage_started = now_s();
    error = dpu_launch(set, DPU_SYNCHRONOUS);
    timing.kernel_time_s = now_s() - stage_started;
    if (error != DPU_OK) {
        report_sdk_error("dpu_launch", error);
        sdk_error_code = (int)error;
        failure_stage = "kernel_launch_failed";
        goto release;
    }
    stage_started = now_s();
    DPU_FOREACH(set, dpu) {
        error = dpu_copy_from(dpu, "GENERIC_C_RAW", 0, output, output_transfer_bytes);
        break;
    }
    timing.d2h_time_s = now_s() - stage_started;
    if (error != DPU_OK) {
        report_sdk_error("GENERIC_C_RAW transfer", error);
        sdk_error_code = (int)error;
        failure_stage = "result_transfer_failed";
        goto release;
    }

release:
    if (set_allocated) {
        error = dpu_free(set);
        if (error != DPU_OK) {
            report_sdk_error("dpu_free", error);
            sdk_error_code = (int)error;
            failure_stage = "hardware_release_failed";
        }
    }
    if (failure_stage != NULL) {
        write_status(failure_stage, 0, requested_dpus, allocated_dpus, sdk_error_code, &timing);
        free(left); free(right); free(output);
        return 1;
    }
    stage_started = now_s();
    int rc = write_exact(out_path, output, output_bytes);
    timing.output_write_time_s = now_s() - stage_started;
    int accounting_rc = 0;
    const char *accounting_path = getenv("UPMEM_GENERIC_TRANSFER_ACCOUNTING_JSON");
    if (rc == 0 && accounting_path != NULL && accounting_path[0] != '\0') {
        accounting_rc = write_transfer_accounting(accounting_path, left_bytes, right_bytes, output_bytes, left_transfer_bytes, right_transfer_bytes, output_transfer_bytes, sizeof(args));
    }
    if (rc != 0) {
        write_status("result_transfer_failed", 0, requested_dpus, allocated_dpus, sdk_error_code, &timing);
        free(left); free(right); free(output);
        return rc;
    }
    if (accounting_rc != 0) {
        write_status("output_manifest_failed", 0, requested_dpus, allocated_dpus, sdk_error_code, &timing);
        free(left); free(right); free(output);
        return accounting_rc;
    }
    write_status(NULL, 1, requested_dpus, allocated_dpus, 0, &timing);
    free(left); free(right); free(output);
    return 0;
}
