#define _POSIX_C_SOURCE 200809L

#include <dpu.h>

#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include "plan.h"
#include "operation_envelope.h"
#include "wave_envelope.h"
#include "simplepim_provider.h"

#ifndef NR_TASKLETS
#define NR_TASKLETS 1
#endif
#include "protocol.h"
#include "request.h"

#ifndef PATH_MAX
#define PATH_MAX 4096
#endif

typedef struct {
    uint64_t tile_id;
    uint64_t cycles;
    uint64_t h2d_bytes;
    uint64_t d2h_bytes;
    uint64_t processed_elements;
    uint32_t dpu_id;
    uint32_t completion_status;
} v4_dpu_result_t;

typedef struct {
    double h2d_time_s;
    double launch_time_s;
    double d2h_time_s;
    double output_time_s;
    double total_route_time_s;
    uint64_t h2d_bytes;
    uint64_t d2h_bytes;
} v4_request_metrics_t;

static volatile sig_atomic_t v4_interrupted = 0;
static volatile sig_atomic_t v4_timeout = 0;
static upmem_v4_provider_t v4_provider;
static int v4_provider_initialized = 0;
static int v4_release_done = 0;
static int v4_release_succeeded = 0;
static const char *v4_session_rank_path = NULL;
static int v4_simulator_target = 0;
static uint64_t v4_last_request_sequence = 0u;
static int v4_have_request_sequence = 0;
static uint64_t v4_last_operation_sequence = 0u;
static int v4_have_operation_sequence = 0;
static int wave_mode = 0;
static char wave_binary_sha256[65];
static unsigned char wave_plan_digest[32];
static int wave_have_plan = 0;

static const char *v4_target_requested(void) {
    return v4_simulator_target ? "simulator" : "hardware";
}

static const char *v4_target_observed(void) {
    return v4_simulator_target ? "sdk_simulator" : "physical_hardware";
}

static const char *v4_backend_id(void) {
    return v4_simulator_target
        ? EXECUTION_PLAN_V4_NATIVE_SIMULATOR_BACKEND_ID
        : EXECUTION_PLAN_V4_NATIVE_BACKEND_ID;
}

static const char *v4_execution_class(void) {
    return v4_simulator_target
        ? EXECUTION_PLAN_V4_NATIVE_SIMULATOR_EXECUTION_CLASS
        : EXECUTION_PLAN_V4_NATIVE_EXECUTION_CLASS;
}

static void v4_signal_handler(int signal_number) {
    (void)signal_number;
    v4_interrupted = 1;
    v4_timeout = 1;
}

static double now_s(void) {
    struct timespec value;
    if (clock_gettime(CLOCK_MONOTONIC, &value) != 0) return 0.0;
    return (double)value.tv_sec + (double)value.tv_nsec / 1000000000.0;
}

static void json_string(FILE *file, const char *value) {
    if (value == NULL) {
        fputs("null", file);
        return;
    }
    fputc('"', file);
    for (const unsigned char *cursor = (const unsigned char *)value; *cursor != '\0'; cursor++) {
        if (*cursor == '"' || *cursor == '\\') fprintf(file, "\\%c", *cursor);
        else if (*cursor < 0x20u) fprintf(file, "\\u%04x", *cursor);
        else fputc(*cursor, file);
    }
    fputc('"', file);
}

static void digest_hex(const unsigned char digest[32], char output[65]) {
    for (uint32_t index = 0u; index < 32u; index++) {
        (void)snprintf(output + index * 2u, 3u, "%02x", digest[index]);
    }
    output[64] = '\0';
}

static void v4_error(char **message, const char *value) {
    if (message != NULL && *message == NULL) *message = strdup(value);
}

static int parse_u32(const char *text, uint32_t *value) {
    char *end = NULL;
    unsigned long parsed;
    errno = 0;
    parsed = strtoul(text == NULL ? "" : text, &end, 10);
    if (errno != 0 || end == text || end == NULL || *end != '\0' || parsed > UINT32_MAX) return 1;
    *value = (uint32_t)parsed;
    return 0;
}

static int path_exists(const char *path) {
    struct stat info;
    return path != NULL && stat(path, &info) == 0 && S_ISREG(info.st_mode);
}

static void v4_emit_release(void) {
    dpu_error_t error = DPU_OK;
    int dpu_free_called_once = 0;
    if (v4_release_done) return;
    v4_release_done = 1;
    if (v4_provider_initialized) {
        dpu_free_called_once = v4_provider.allocation_active;
        error = upmem_v4_provider_release(&v4_provider);
    }
    v4_release_succeeded = !v4_provider_initialized ||
        (error == DPU_OK && v4_provider.release_succeeded);
    printf("{\"event\":\"RELEASE\",\"status\":\"%s\",\"release_attempted\":true,\"release_succeeded\":%s,\"dpu_free_called_once\":%s}\n",
        v4_release_succeeded ? "released" : "failed",
        v4_release_succeeded ? "true" : "false",
        dpu_free_called_once ? "true" : "false");
    fflush(stdout);
}

static void v4_emit_startup_failure(const char *stage, const char *message) {
    printf("{\"event\":\"STARTUP\",\"status\":\"failed\",\"failure_stage\":");
    json_string(stdout, stage);
    fputs(",\"error\":", stdout);
    json_string(stdout, message);
    fputs(",\"target_observed\":\"not_executed\",\"simulator_kernel_executed\":false,\"cpu_fallback_used\":false}\n", stdout);
    fflush(stdout);
}

static void v4_emit_ready(
    const char *rank_path,
    const char *dpu_binary,
    const char *initialization_binary,
    uint32_t dpus,
    uint32_t tasklets
) {
    char dpu_hash[65] = {0};
    char init_hash[65] = {0};
    (void)execution_plan_sha256_file(dpu_binary, dpu_hash);
    (void)execution_plan_sha256_file(initialization_binary, init_hash);
    printf("{\"event\":\"READY\",\"status\":\"ready\",\"backend_id\":\"");
    fputs(v4_backend_id(), stdout);
    fputs("\",\"backend_family\":\"", stdout);
    fputs(EXECUTION_PLAN_V4_NATIVE_BACKEND_FAMILY, stdout);
    fputs("\",\"profile\":\"", stdout);
    fputs(wave_mode ? "prepared_wave_v1" : EXECUTION_PLAN_V4_NATIVE_M5_PROFILE, stdout);
    fputs("\",\"abi\":\"", stdout);
    fputs(wave_mode ? "wave_control_v5" : EXECUTION_PLAN_V4_NATIVE_ABI, stdout);
    fputs("\",\"session_protocol\":\"", stdout);
    fputs(wave_mode ? "prepared_wave_session_v1" : EXECUTION_PLAN_V4_NATIVE_SESSION, stdout);
    fputs("\",\"request_transport\":\"", stdout);
    fputs(wave_mode ? "packed_wave_v1" : "packed_operation_v1", stdout);
    fputs("\",\"dispatch_mode\":\"", stdout);
    fputs(EXECUTION_PLAN_V4_NATIVE_DISPATCH, stdout);
    fputs("\",\"kernel_identity\":\"", stdout);
    fputs(wave_mode ? "dpu_panel_dispatch_v5_v1" : EXECUTION_PLAN_V4_NATIVE_KERNEL, stdout);
    fputs("\",\"execution_class\":\"", stdout);
    fputs(v4_execution_class(), stdout);
    fputs("\",\"target_requested\":\"", stdout);
    fputs(v4_target_requested(), stdout);
    fputs("\",\"target_observed\":\"", stdout);
    fputs(v4_target_observed(), stdout);
    fputs("\",\"rank_path\":", stdout);
    json_string(stdout, rank_path);
    printf(",\"requested_dpu_count\":%u,\"allocated_dpu_count\":%u,\"tasklets_per_dpu\":%u,\"allocation_verified\":true,\"hardware_allocation_verified\":%s,\"native_kernel_executed\":false,\"hardware_kernel_executed\":false,\"simulator_kernel_executed\":false,\"cpu_fallback_used\":false,\"hardware_functionality_evidence\":%s,\"simulator_functionality_evidence\":%s%s,\"dpu_binary_sha256\":", dpus, dpus, tasklets,
        v4_simulator_target ? "false" : "true",
        v4_simulator_target ? "false" : "true",
        v4_simulator_target ? "true" : "false",
        v4_simulator_target ? ",\"timing_claim_applicable\":false,\"scaling_claim_applicable\":false,\"speedup_claim_applicable\":false,\"energy_claim_applicable\":false" : "");
    json_string(stdout, dpu_hash);
    fputs(",\"initialization_binary_sha256\":", stdout);
    json_string(stdout, init_hash);
    fputs(",\"session_release_pending\":true}\n", stdout);
    fflush(stdout);
}

static int copy_request_to_dpus(
    const execution_plan_v4_request_t *request,
    struct dpu_set_t set,
    uint32_t tasklets,
    v4_request_metrics_t *metrics,
    char **error_message
) {
    struct dpu_set_t dpu;
    uint32_t dpu_id;
    const double started = now_s();
    DPU_FOREACH(set, dpu, dpu_id) {
        const execution_plan_v4_work_unit_t *unit =
            execution_plan_distributed_v4_work_unit_for_dpu(request->work_units, request->header.work_unit_count, dpu_id);
        const execution_plan_v4_request_item_t *item = &request->items[dpu_id];
        execution_plan_v4_control_t control = {0};
        dpu_error_t error;
        if (unit == NULL) {
            v4_error(error_message, "argument_transfer_failed: no dense v4 work unit for DPU");
            return 1;
        }
        control.magic = EXECUTION_PLAN_V4_CONTROL_MAGIC;
        control.version = EXECUTION_PLAN_V4_VERSION;
        control.numeric_mode = request->header.numeric_mode;
        control.dpu_id = dpu_id;
        control.flags = unit->flags;
        control.batch_index = (uint32_t)unit->batch_index;
        control.m_elements = unit->m_elements;
        control.n_elements = unit->n_elements;
        control.k_elements = unit->k_elements;
        control.a_transfer_bytes = unit->a_transfer_bytes;
        control.b_transfer_bytes = unit->b_transfer_bytes;
        control.c_transfer_bytes = unit->c_transfer_bytes;
        control.a_offset_bytes = unit->a_offset_bytes;
        control.b_offset_bytes = unit->b_offset_bytes;
        control.c_offset_bytes = unit->c_offset_bytes;
        control.k_offset = (uint32_t)unit->k_offset;
        control.reserved0 = tasklets;
        error = dpu_copy_to(dpu, "V4_CONTROL", 0u, &control, sizeof(control));
        metrics->h2d_bytes += sizeof(control);
        if (error == DPU_OK && (unit->flags & EXECUTION_PLAN_V4_FLAG_ZERO_WORK) == 0u) {
            error = dpu_copy_to(dpu, "V4_MRAM", unit->a_offset_bytes,
                item->a_payload, unit->a_transfer_bytes);
            metrics->h2d_bytes += unit->a_transfer_bytes;
        }
        if (error == DPU_OK && (unit->flags & EXECUTION_PLAN_V4_FLAG_ZERO_WORK) == 0u) {
            error = dpu_copy_to(dpu, "V4_MRAM", unit->b_offset_bytes,
                item->b_payload, unit->b_transfer_bytes);
            metrics->h2d_bytes += unit->b_transfer_bytes;
        }
        if (error != DPU_OK) {
            v4_error(error_message, "argument_transfer_failed: per-DPU descriptor or payload transfer failed");
            return 1;
        }
    }
    metrics->h2d_time_s = now_s() - started;
    return 0;
}

static int collect_request_from_dpus(
    execution_plan_v4_request_t *request,
    struct dpu_set_t set,
    v4_request_metrics_t *metrics,
    v4_dpu_result_t results[EXECUTION_PLAN_V4_MAX_DPUS],
    char **error_message
) {
    struct dpu_set_t dpu;
    uint32_t dpu_id;
    const double started = now_s();
    DPU_FOREACH(set, dpu, dpu_id) {
        execution_plan_v4_request_item_t *item = &request->items[dpu_id];
        execution_plan_v4_completion_t completion = {0};
        dpu_error_t error = dpu_copy_from(dpu, "V4_COMPLETION", 0u, &completion, sizeof(completion));
        metrics->d2h_bytes += sizeof(completion);
        if (error != DPU_OK || completion.magic != EXECUTION_PLAN_V4_COMPLETION_MAGIC ||
            completion.version != EXECUTION_PLAN_V4_VERSION || completion.dpu_id != dpu_id ||
            completion.status != EXECUTION_PLAN_V4_STATUS_COMPLETED) {
            v4_error(error_message, "result_transfer_failed: invalid v4 DPU completion");
            return 1;
        }
        results[dpu_id].tile_id = item->work_unit.tile_id;
        results[dpu_id].cycles = completion.cycles;
        results[dpu_id].processed_elements = completion.processed_elements;
        results[dpu_id].dpu_id = dpu_id;
        results[dpu_id].completion_status = completion.status;
        results[dpu_id].h2d_bytes = item->work_unit.a_transfer_bytes + item->work_unit.b_transfer_bytes + sizeof(execution_plan_v4_control_t);
        results[dpu_id].d2h_bytes = item->work_unit.c_transfer_bytes + sizeof(completion);
        if ((item->work_unit.flags & EXECUTION_PLAN_V4_FLAG_ZERO_WORK) == 0u) {
            error = dpu_copy_from(dpu, "V4_MRAM", item->work_unit.c_offset_bytes,
                item->c_payload, item->work_unit.c_transfer_bytes);
            metrics->d2h_bytes += item->work_unit.c_transfer_bytes;
            if (error != DPU_OK) {
                v4_error(error_message, "result_transfer_failed: C output transfer failed");
                return 1;
            }
            if (execution_plan_v4_request_write_output(item, error_message) != 0) return 1;
        }
    }
    metrics->d2h_time_s = now_s() - started;
    return 0;
}

static void v4_emit_response_to(
    FILE *output,
    const execution_plan_v4_request_t *request,
    const v4_request_metrics_t *metrics,
    const v4_dpu_result_t results[EXECUTION_PLAN_V4_MAX_DPUS],
    const char *failure_stage,
    const char *error_message,
    int native_kernel_executed,
    int bulk_set_launch_verified
) {
    char task_contract_sha256[65] = {0};
    if (request != NULL) digest_hex(request->header.task_contract_sha256, task_contract_sha256);
    fprintf(output, "{\"event\":\"RESPONSE\",\"status\":\"%s\",\"failure_stage\":",
        failure_stage == NULL ? "completed" : "failed");
    json_string(output, failure_stage);
    fputs(",\"error\":", output);
    json_string(output, error_message);
    fputs(",\"backend_id\":\"", output);
    fputs(v4_backend_id(), output);
    fputs("\",\"backend_family\":\"", output);
    fputs(EXECUTION_PLAN_V4_NATIVE_BACKEND_FAMILY, output);
    fputs("\",\"profile\":\"", output);
    fputs(EXECUTION_PLAN_V4_NATIVE_M5_PROFILE, output);
    fputs("\",\"abi\":\"", output);
    fputs(EXECUTION_PLAN_V4_NATIVE_ABI, output);
    fputs("\",\"session_protocol\":\"", output);
    fputs(EXECUTION_PLAN_V4_NATIVE_SESSION, output);
    fputs("\",\"dispatch_mode\":\"", output);
    fputs(EXECUTION_PLAN_V4_NATIVE_DISPATCH, output);
    fputs("\",\"kernel_identity\":\"", output);
    fputs(EXECUTION_PLAN_V4_NATIVE_KERNEL, output);
    fputs("\",\"target_requested\":\"", output);
    fputs(v4_target_requested(), output);
    fputs("\",\"target_observed\":\"", output);
    fputs(v4_target_observed(), output);
    fputs("\",\"execution_class\":\"", output);
    fputs(v4_execution_class(), output);
    fputs("\",\"rank_path\":", output);
    json_string(output, v4_session_rank_path);
    fprintf(output, ",\"request_sequence\":%llu,\"request_output_elements\":%llu,\"global_output_elements\":%llu,\"global_completeness\":false,\"task_contract_sha256\":\"%s\",\"request_sha256\":\"%s\",\"request_manifest_sha256\":\"%s\",\"sidecar_sha256\":\"%s\",\"bulk_set_launch_verified\":%s,\"requested_dpu_count\":%u,\"allocated_dpu_count\":%u,\"tasklets_per_dpu\":%u,\"allocation_verified\":true,\"hardware_allocation_verified\":%s,\"native_kernel_executed\":%s,\"simulator_kernel_executed\":%s,\"hardware_kernel_executed\":%s,\"cpu_fallback_used\":false,\"session_release_pending\":true,\"timing_scope\":\"one_bulk_request_in_persistent_session\",\"request_timing_is_bringup_only\":true,\"request_level_speedup_applicable\":false,\"hardware_functionality_evidence\":%s,\"simulator_functionality_evidence\":%s%s,\"timing\":{\"h2d_time_s\":%.9f,\"launch_time_s\":%.9f,\"d2h_time_s\":%.9f,\"output_time_s\":%.9f,\"total_route_time_s\":%.9f},\"transfer\":{\"h2d_bytes\":%llu,\"d2h_bytes\":%llu,\"total_bytes\":%llu},\"per_dpu\":[",
        request == NULL ? 0ull : (unsigned long long)request->header.request_sequence,
        request == NULL ? 0ull : (unsigned long long)request->header.request_output_elements,
        request == NULL ? 0ull : (unsigned long long)request->header.global_output_elements,
        task_contract_sha256,
        request == NULL ? "" : request->manifest_sha256,
        request == NULL ? "" : request->manifest_sha256,
        request == NULL ? "" : request->sidecar_sha256,
        bulk_set_launch_verified ? "true" : "false",
        request == NULL ? 0u : request->header.dpu_count,
        v4_provider.observed_dpus,
        request == NULL ? 0u : request->header.tasklets_per_dpu,
        v4_simulator_target ? "false" : "true",
        native_kernel_executed ? "true" : "false",
        v4_simulator_target && native_kernel_executed ? "true" : "false",
        !v4_simulator_target && native_kernel_executed ? "true" : "false",
        !v4_simulator_target && native_kernel_executed && failure_stage == NULL ? "true" : "false",
        v4_simulator_target && native_kernel_executed && failure_stage == NULL ? "true" : "false",
        v4_simulator_target ? ",\"timing_claim_applicable\":false,\"scaling_claim_applicable\":false,\"speedup_claim_applicable\":false,\"energy_claim_applicable\":false" : "",
        metrics == NULL ? 0.0 : metrics->h2d_time_s,
        metrics == NULL ? 0.0 : metrics->launch_time_s,
        metrics == NULL ? 0.0 : metrics->d2h_time_s,
        metrics == NULL ? 0.0 : metrics->output_time_s,
        metrics == NULL ? 0.0 : metrics->total_route_time_s,
        metrics == NULL ? 0ull : (unsigned long long)metrics->h2d_bytes,
        metrics == NULL ? 0ull : (unsigned long long)metrics->d2h_bytes,
        metrics == NULL ? 0ull : (unsigned long long)(metrics->h2d_bytes + metrics->d2h_bytes));
    if (request != NULL) {
        for (uint32_t index = 0u; index < request->header.dpu_count; index++) {
            if (index != 0u) fputc(',', output);
            fprintf(output, "{\"dpu_id\":%u,\"tile_id\":%llu,\"completion_status\":%u,\"cycles\":%llu,\"processed_elements\":%llu,\"h2d_bytes\":%llu,\"d2h_bytes\":%llu}",
                results[index].dpu_id, (unsigned long long)results[index].tile_id,
                results[index].completion_status, (unsigned long long)results[index].cycles,
                (unsigned long long)results[index].processed_elements,
                (unsigned long long)results[index].h2d_bytes,
                (unsigned long long)results[index].d2h_bytes);
        }
    }
    fputs("]}\n", output);
    fflush(output);
}

static int execute_loaded_request(
    execution_plan_v4_request_t *request,
    uint32_t tasklets,
    uint32_t timeout_s,
    FILE *response_output
) {
    v4_request_metrics_t metrics = {0};
    v4_dpu_result_t results[EXECUTION_PLAN_V4_MAX_DPUS] = {0};
    char *error_message = NULL;
    const char *failure_stage = NULL;
    dpu_error_t error;
    int native_kernel_executed = 0;
    int result = 1;
    const double route_started = now_s();
    FILE *output = response_output == NULL ? stdout : response_output;
    if (request == NULL) {
        v4_emit_response_to(output, NULL, &metrics, results, "request_manifest_failed",
            "missing loaded request", 0, 0);
        return 1;
    }
    if (v4_have_request_sequence && request->header.request_sequence <= v4_last_request_sequence) {
        failure_stage = "hardware_profile_violation";
        metrics.total_route_time_s = now_s() - route_started;
        v4_emit_response_to(output, request, &metrics, results, failure_stage,
            "v4 request_sequence must increase for each SUBMIT", 0, 0);
        return 1;
    }
    v4_last_request_sequence = request->header.request_sequence;
    v4_have_request_sequence = 1;
    if (v4_interrupted) {
        failure_stage = v4_timeout ? "kernel_timeout" : "kernel_launch_failed";
        metrics.total_route_time_s = now_s() - route_started;
        v4_emit_response_to(output, request, &metrics, results, failure_stage,
            "request interrupted before launch", 0, 0);
        return 1;
    }
    if (copy_request_to_dpus(request, v4_provider.set, tasklets, &metrics, &error_message) != 0) {
        failure_stage = "argument_transfer_failed";
        metrics.total_route_time_s = now_s() - route_started;
        v4_emit_response_to(output, request, &metrics, results, failure_stage, error_message, 0, 0);
        free(error_message);
        return 1;
    }
    alarm(timeout_s);
    {
        const double started = now_s();
        error = dpu_launch(v4_provider.set, DPU_SYNCHRONOUS);
        metrics.launch_time_s = now_s() - started;
    }
    alarm(0u);
    if (error != DPU_OK || v4_timeout || v4_interrupted) {
        failure_stage = v4_timeout ? "kernel_timeout" : "kernel_launch_failed";
        metrics.total_route_time_s = now_s() - route_started;
        v4_emit_response_to(output, request, &metrics, results, failure_stage,
            "bulk synchronous v4 launch failed", 0, error == DPU_OK);
        return 1;
    }
    native_kernel_executed = 1;
    {
        const double started = now_s();
        if (collect_request_from_dpus(request, v4_provider.set, &metrics, results, &error_message) != 0) {
            failure_stage = "result_transfer_failed";
        }
        metrics.output_time_s = now_s() - started;
    }
    metrics.total_route_time_s = now_s() - route_started;
    v4_emit_response_to(output, request, &metrics, results, failure_stage, error_message,
        native_kernel_executed, 1);
    result = failure_stage == NULL ? 0 : 1;
    free(error_message);
    return result;
}

static int host_path_inside_root(const char *root, const char *candidate) {
    size_t root_length = strlen(root);
    return strncmp(root, candidate, root_length) == 0 &&
        (candidate[root_length] == '\0' || candidate[root_length] == '/');
}

static int open_operation_result_file(
    const char *session_root,
    uint64_t operation_sequence,
    char relative_path[PATH_MAX],
    char absolute_path[PATH_MAX],
    FILE **result_file,
    char **error_message
) {
    char results_directory[PATH_MAX];
    char resolved_directory[PATH_MAX];
    struct stat info;
    int descriptor;
    if (session_root == NULL || relative_path == NULL || absolute_path == NULL || result_file == NULL ||
        snprintf(relative_path, PATH_MAX, "results/operation_%016llx.jsonl",
            (unsigned long long)operation_sequence) >= PATH_MAX ||
        snprintf(results_directory, sizeof(results_directory), "%s/results", session_root) >= (int)sizeof(results_directory) ||
        (mkdir(results_directory, 0777) != 0 && errno != EEXIST) ||
        stat(results_directory, &info) != 0 || !S_ISDIR(info.st_mode) ||
        realpath(results_directory, resolved_directory) == NULL ||
        !host_path_inside_root(session_root, resolved_directory) ||
        snprintf(absolute_path, PATH_MAX, "%s/%s", session_root, relative_path) >= PATH_MAX) {
        v4_error(error_message, "output_manifest_failed: operation result directory is unsafe or unavailable");
        return 1;
    }
#ifdef O_CLOEXEC
    descriptor = open(absolute_path, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0666);
#else
    descriptor = open(absolute_path, O_WRONLY | O_CREAT | O_EXCL, 0666);
#endif
    if (descriptor < 0) {
        v4_error(error_message, "output_manifest_failed: operation result file could not be created");
        return 1;
    }
    *result_file = fdopen(descriptor, "w");
    if (*result_file == NULL) {
        close(descriptor);
        v4_error(error_message, "output_manifest_failed: operation result stream could not be opened");
        return 1;
    }
    return 0;
}

static void v4_emit_operation_response(
    uint64_t operation_sequence,
    const char *failure_stage,
    const char *error_message,
    const char *result_path,
    uint64_t response_count,
    uint64_t completed_request_count,
    int failed_request_index_present,
    uint64_t failed_request_index,
    const char *result_sha256
) {
    printf("{\"event\":\"OPERATION_RESPONSE\",\"status\":\"%s\",\"failure_stage\":",
        failure_stage == NULL ? "completed" : "failed");
    json_string(stdout, failure_stage);
    fputs(",\"error\":", stdout);
    json_string(stdout, error_message);
    printf(",\"operation_sequence\":%llu,\"response_path\":",
        (unsigned long long)operation_sequence);
    json_string(stdout, result_path);
    printf(",\"response_count\":%llu,\"completed_request_count\":%llu,\"failed_request_index\":",
        (unsigned long long)response_count,
        (unsigned long long)completed_request_count);
    if (failed_request_index_present) {
        printf("%llu", (unsigned long long)failed_request_index);
    } else {
        fputs("null", stdout);
    }
    fputs(",\"response_sha256\":", stdout);
    json_string(stdout, result_sha256);
    fputs("}\n", stdout);
    fflush(stdout);
}

static int execute_packed_operation(
    const char *session_root,
    const char *relative_path,
    const char *envelope_sha256,
    uint32_t dpus,
    uint32_t tasklets,
    uint32_t timeout_s
) {
    execution_plan_v4_operation_envelope_t operation = {0};
    FILE *result_file = NULL;
    char result_relative_path[PATH_MAX] = {0};
    char result_absolute_path[PATH_MAX] = {0};
    char *error_message = NULL;
    char result_sha256[65] = {0};
    const char *failure_stage = NULL;
    uint64_t response_count = 0u;
    uint64_t completed_request_count = 0u;
    uint64_t failed_request_index = 0u;
    int failed_request_index_present = 0;
    int result_file_close_status = 0;
    int result_hash_status = 1;
    int rc = 1;
    if (execution_plan_v4_operation_open(session_root, relative_path, envelope_sha256,
            dpus, tasklets, &operation, &error_message) != 0) {
        v4_emit_operation_response(0u, "operation_envelope_failed", error_message,
            NULL, 0u, 0u, 0, 0u, NULL);
        free(error_message);
        execution_plan_v4_operation_close(&operation);
        return 1;
    }
    if (v4_have_operation_sequence && operation.operation_sequence <= v4_last_operation_sequence) {
        v4_emit_operation_response(operation.operation_sequence, "hardware_profile_violation",
            "v4 operation_sequence must increase for each packed operation", NULL,
            0u, 0u, 0, 0u, NULL);
        execution_plan_v4_operation_close(&operation);
        return 1;
    }
    v4_last_operation_sequence = operation.operation_sequence;
    v4_have_operation_sequence = 1;
    if (open_operation_result_file(session_root, operation.operation_sequence,
            result_relative_path, result_absolute_path, &result_file, &error_message) != 0) {
        v4_emit_operation_response(operation.operation_sequence, "output_manifest_failed",
            error_message, NULL, 0u, 0u, 0, 0u, NULL);
        free(error_message);
        execution_plan_v4_operation_close(&operation);
        return 1;
    }
    for (uint32_t index = 0u; index < operation.descriptor_count; index++) {
        execution_plan_v4_embedded_request_t embedded = {0};
        execution_plan_v4_request_t request = {0};
        char *request_error = NULL;
        int request_status;
        if (execution_plan_v4_operation_descriptor(&operation, index, &embedded, &request_error) != 0 ||
            execution_plan_v4_request_load_embedded(session_root, &embedded, dpus, tasklets,
                &request, &request_error) != 0) {
            failed_request_index = index;
            failed_request_index_present = 1;
            v4_emit_response_to(result_file, &request, NULL,
                (const v4_dpu_result_t[EXECUTION_PLAN_V4_MAX_DPUS]){0},
                "request_manifest_failed", request_error, 0, 0);
            response_count++;
            if (ferror(result_file) != 0) {
                failure_stage = "output_manifest_failed";
                v4_error(&error_message, "operation result JSONL write failed");
            } else {
                failure_stage = "request_manifest_failed";
                v4_error(&error_message, request_error == NULL
                    ? "embedded request could not be loaded" : request_error);
            }
            free(request_error);
            execution_plan_v4_request_free(&request);
            break;
        }
        request_status = execute_loaded_request(&request, tasklets, timeout_s, result_file);
        response_count++;
        if (ferror(result_file) != 0) {
            failed_request_index = index;
            failed_request_index_present = 1;
            failure_stage = "output_manifest_failed";
            v4_error(&error_message, "operation result JSONL write failed");
            execution_plan_v4_request_free(&request);
            break;
        }
        execution_plan_v4_request_free(&request);
        if (request_status != 0) {
            failed_request_index = index;
            failed_request_index_present = 1;
            failure_stage = "request_execution_failed";
            v4_error(&error_message, "embedded request execution failed; operation stopped");
            break;
        }
        completed_request_count++;
    }
    result_file_close_status = fclose(result_file);
    result_file = NULL;
    if (result_file_close_status != 0 && failure_stage == NULL) {
        failure_stage = "output_manifest_failed";
        v4_error(&error_message, "operation result JSONL close failed");
    }
    if (execution_plan_sha256_file(result_absolute_path, result_sha256) == 0) {
        result_hash_status = 0;
    } else if (failure_stage == NULL) {
        failure_stage = "output_manifest_failed";
        v4_error(&error_message, "operation result JSONL hash failed");
    }
    if (failure_stage == NULL && response_count != operation.descriptor_count) {
        failure_stage = "request_execution_failed";
        v4_error(&error_message, "packed operation did not execute every descriptor");
    }
    v4_emit_operation_response(operation.operation_sequence, failure_stage, error_message,
        result_relative_path, response_count, completed_request_count,
        failed_request_index_present, failed_request_index,
        result_hash_status == 0 ? result_sha256 : NULL);
    free(error_message);
    execution_plan_v4_operation_close(&operation);
    rc = failure_stage == NULL ? 0 : 1;
    return rc;
}

static int verify_wave_binary(struct dpu_program_t *program, uint32_t tasklets) {
    const char *names[] = {"WAVE_CONTROL", "WAVE_COMPLETION", "WAVE_MRAM", "WAVE_TASKLETS"};
    const uint32_t sizes[] = {sizeof(upmem_wave_control_t), sizeof(upmem_wave_completion_t),
        UPMEM_WAVE_MRAM_BYTES, sizeof(uint32_t)};
    struct dpu_symbol_t symbol;
    for (unsigned i = 0; i < 4; ++i)
        if (dpu_get_symbol(program, names[i], &symbol) != DPU_OK || symbol.size != sizes[i]) return 1;
    struct dpu_set_t dpu;
    DPU_FOREACH(v4_provider.set, dpu) {
        uint32_t actual = 0;
        if (dpu_copy_from(dpu, "WAVE_TASKLETS", 0, &actual, sizeof(actual)) != DPU_OK ||
                actual != tasklets) return 1;
    }
    return 0;
}

static int execute_wave_envelope(const char *root, const char *name, const char *digest,
        uint32_t dpus, uint32_t tasklets, uint32_t timeout_s) {
    upmem_wave_envelope_t envelope = {.fd = -1};
    char *message = NULL;
    const char *failure = NULL;
    FILE *output = NULL;
    unsigned char *output_buffer = NULL;
    uint32_t output_buffer_bytes = 0;
    char output_name[96] = {0}, output_path[PATH_MAX] = {0}, output_sha256[65] = {0};
    uint64_t completed_waves = 0, completed_results = 0, launches = 0;
    uint32_t failed_dpu = UINT32_MAX, failed_operation = UINT32_MAX;
    upmem_wave_completion_t failed_completion = {0};
    int have_failed_completion = 0;
    int have_output = 0;
    v4_request_metrics_t metrics = {0};
    const double started = now_s();
    if (upmem_wave_envelope_open(root, name, digest, wave_binary_sha256, dpus, tasklets,
            &envelope, &message)) { failure = "wave_envelope_failed"; goto done; }
    upmem_wave_tile_t first;
    upmem_wave_envelope_tile(&envelope, 0, &first);
    if ((v4_have_operation_sequence && envelope.sequence <= v4_last_operation_sequence) ||
            (v4_have_request_sequence && first.control.request_sequence <= v4_last_request_sequence) ||
            (wave_have_plan && memcmp(wave_plan_digest, envelope.data+72, 32))) {
        failure = "wave_identity_failed";
        v4_error(&message, "wave session plan changed or sequence replay detected");
        goto done;
    }
    memcpy(wave_plan_digest, envelope.data+72, 32); wave_have_plan = 1;
    v4_last_operation_sequence = envelope.sequence; v4_have_operation_sequence = 1;
    snprintf(output_name, sizeof(output_name), "wave-result-%020llu.bin",
        (unsigned long long)envelope.sequence);
    if (snprintf(output_path, sizeof(output_path), "%s/%s", root, output_name) >= (int)sizeof(output_path)) {
        failure = "output_manifest_failed"; goto done;
    }
    int fd = open(output_path, O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0600);
    if (fd < 0) { failure = "output_manifest_failed"; goto done; }
    output = fdopen(fd, "wb");
    if (!output) { close(fd); unlink(output_path); failure = "output_manifest_failed"; goto done; }
    have_output = 1;
    output_buffer = malloc(256u * 256u * sizeof(uint32_t));
    if (!output_buffer) { failure = "host_allocation_failed"; goto done; }
    output_buffer_bytes = 256u * 256u * sizeof(uint32_t);
    size_t cursor = envelope.payload_offset;
    for (uint32_t w = 0; w < envelope.wave_count; ++w) {
        upmem_wave_tile_t tiles[UPMEM_WAVE_MAX_DPUS];
        struct dpu_set_t dpu;
        uint32_t d;
        double section = now_s();
        DPU_FOREACH(v4_provider.set, dpu, d) {
            upmem_wave_envelope_tile(&envelope, (uint64_t)w * dpus + d, &tiles[d]);
            const upmem_wave_control_t *c = &tiles[d].control;
            failed_dpu = d; failed_operation = c->operation_index;
            if (v4_interrupted || !upmem_wave_control_valid(c, d, tasklets)) {
                failure = v4_timeout ? "kernel_timeout" : "wave_identity_failed"; break;
            }
            if (dpu_copy_to(dpu, "WAVE_CONTROL", 0, c, sizeof(*c)) != DPU_OK) {
                failure = "argument_transfer_failed"; break;
            }
            metrics.h2d_bytes += sizeof(*c);
            for (unsigned plane = 0; plane < 4; ++plane) {
                const upmem_wave_span_t span = c->planes[plane];
                if (span.length && dpu_copy_to(dpu, "WAVE_MRAM", span.offset,
                        envelope.data+cursor, span.length) != DPU_OK) {
                    failure = "argument_transfer_failed"; break;
                }
                cursor += span.length; metrics.h2d_bytes += span.length;
            }
            if (failure) break;
        }
        metrics.h2d_time_s += now_s() - section;
        if (failure) break;
        failed_dpu = failed_operation = UINT32_MAX;
        v4_last_request_sequence = tiles[0].control.request_sequence;
        v4_have_request_sequence = 1;
        section = now_s();
        alarm(timeout_s);
        dpu_error_t launched = dpu_launch(v4_provider.set, DPU_SYNCHRONOUS);
        alarm(0);
        metrics.launch_time_s += now_s() - section;
        ++launches;
        if (launched != DPU_OK || v4_interrupted || v4_timeout) {
            failure = v4_timeout ? "kernel_timeout" : "kernel_launch_failed"; break;
        }
        DPU_FOREACH(v4_provider.set, dpu, d) {
            const upmem_wave_control_t *c = &tiles[d].control;
            upmem_wave_completion_t completion;
            failed_dpu = d; failed_operation = c->operation_index;
            section = now_s();
            dpu_error_t copied = dpu_copy_from(dpu, "WAVE_COMPLETION", 0,
                &completion, sizeof(completion));
            metrics.d2h_time_s += now_s() - section;
            metrics.d2h_bytes += sizeof(completion);
            if (copied != DPU_OK) { failure = "result_transfer_failed"; break; }
            if (!upmem_wave_completion_success(&completion, c)) {
                failed_completion = completion; have_failed_completion = 1;
                failure = "wave_completion_failed"; break;
            }
            section = now_s();
            if (fwrite(&completion, 1, sizeof(completion), output) != sizeof(completion))
                failure = "output_manifest_failed";
            metrics.output_time_s += now_s() - section;
            if (failure) break;
            for (unsigned plane = 4; plane < UPMEM_WAVE_PLANE_COUNT; ++plane) {
                const upmem_wave_span_t span = c->planes[plane];
                if (!span.length) continue;
                section = now_s();
                copied = dpu_copy_from(dpu, "WAVE_MRAM", span.offset, output_buffer, span.length);
                metrics.d2h_time_s += now_s() - section;
                metrics.d2h_bytes += span.length;
                if (copied != DPU_OK) failure = "result_transfer_failed";
                else {
                    const size_t live = (size_t)c->m * c->n * sizeof(uint32_t);
                    section = now_s();
                    if (fwrite(output_buffer, 1, live, output) != live) failure = "output_manifest_failed";
                    metrics.output_time_s += now_s() - section;
                }
                if (failure) break;
            }
            if (failure) break;
            ++completed_results;
        }
        if (failure) break;
        ++completed_waves;
        failed_dpu = failed_operation = UINT32_MAX;
    }
done:
    free(output_buffer);
    if (v4_interrupted && !failure) failure = v4_timeout ? "kernel_timeout" : "kernel_launch_failed";
    if (output) {
        if (fclose(output) && !failure) failure = "output_manifest_failed";
        output = NULL;
        if (execution_plan_sha256_file(output_path, output_sha256) && !failure)
            failure = "output_manifest_failed";
    }
    metrics.total_route_time_s = now_s() - started;
    printf("{\"event\":\"WAVE_RESPONSE\",\"status\":\"%s\",\"failure_stage\":",
        failure ? "failed" : "completed");
    json_string(stdout, failure);
    fputs(",\"error\":", stdout); json_string(stdout, message ? message : failure);
    printf(",\"sequence\":%llu,\"completed_wave_count\":%llu,\"completed_result_count\":%llu,"
        "\"launch_count\":%llu,\"failed_wave_index\":",
        (unsigned long long)envelope.sequence, (unsigned long long)completed_waves,
        (unsigned long long)completed_results, (unsigned long long)launches);
    if (failure && envelope.data && completed_waves < envelope.wave_count)
        printf("%llu", (unsigned long long)completed_waves);
    else fputs("null", stdout);
    fputs(",\"failed_dpu_id\":", stdout);
    if (failed_dpu != UINT32_MAX) printf("%u", failed_dpu); else fputs("null", stdout);
    fputs(",\"failed_operation_index\":", stdout);
    if (failed_operation != UINT32_MAX) printf("%u", failed_operation); else fputs("null", stdout);
    fputs(",\"failed_completion_mask\":", stdout);
    if (have_failed_completion) printf("%u", failed_completion.completed_product_mask); else fputs("null", stdout);
    fputs(",\"failed_completion_status\":", stdout);
    if (have_failed_completion) printf("%u", failed_completion.status); else fputs("null", stdout);
    fputs(",\"failed_completion_stage\":", stdout);
    if (have_failed_completion) printf("%u", failed_completion.failure_stage); else fputs("null", stdout);
    fputs(",\"failed_product\":", stdout);
    if (have_failed_completion && failed_completion.failing_product != UPMEM_WAVE_NO_PRODUCT)
        printf("%u", failed_completion.failing_product); else fputs("null", stdout);
    fputs(",\"response_path\":", stdout); json_string(stdout, have_output ? output_name : NULL);
    fputs(",\"response_sha256\":", stdout); json_string(stdout, *output_sha256 ? output_sha256 : NULL);
    printf(",\"h2d_bytes\":%llu,\"d2h_bytes\":%llu,\"h2d_time_s\":%.9f,"
        "\"kernel_time_s\":%.9f,\"d2h_time_s\":%.9f,\"output_time_s\":%.9f,"
        "\"total_route_time_s\":%.9f,\"allocated_dpu_count\":%u,\"tasklets_per_dpu\":%u,"
        "\"cpu_fallback_used\":false,\"target_observed\":",
        (unsigned long long)metrics.h2d_bytes, (unsigned long long)metrics.d2h_bytes,
        metrics.h2d_time_s, metrics.launch_time_s, metrics.d2h_time_s,
        metrics.output_time_s, metrics.total_route_time_s, dpus, tasklets);
    json_string(stdout, v4_target_observed());
    printf(",\"envelope_bytes\":%llu,\"native_snapshot_bytes\":%llu,"
        "\"input_payload_bytes\":%llu,\"operation_count\":%u,\"control_count\":%llu,"
        "\"native_output_buffer_bytes\":%u",
        (unsigned long long)envelope.size, (unsigned long long)envelope.size,
        (unsigned long long)(envelope.size - envelope.payload_offset), envelope.operation_count,
        (unsigned long long)envelope.control_count, output_buffer_bytes);
    fputs("}\n", stdout); fflush(stdout);
    upmem_wave_envelope_close(&envelope); free(message);
    return failure ? 1 : 0;
}

static void v4_usage(const char *program) {
    fprintf(stderr, "usage: %s --target hardware|simulator --session-root DIR [--rank-path /dev/dpu_rankN] --dpus N --tasklets N --initialization-binary PATH --dpu-binary PATH [--timeout-s N] [--wave-v5]\n", program);
}

int main(int argc, char **argv) {
    const char *session_root = NULL;
    const char *rank_path = NULL;
    const char *target = NULL;
    const char *initialization_binary = NULL;
    const char *dpu_binary = NULL;
    uint32_t dpus = 0u, tasklets = 0u, timeout_s = 60u;
    char root_real[PATH_MAX];
    struct stat root_stat;
    dpu_error_t error;
    int rc = 1;
    for (int index = 1; index < argc; index++) {
        if (strcmp(argv[index], "--target") == 0 && index + 1 < argc) target = argv[++index];
        else if (strcmp(argv[index], "--session-root") == 0 && index + 1 < argc) session_root = argv[++index];
        else if (strcmp(argv[index], "--rank-path") == 0 && index + 1 < argc) rank_path = argv[++index];
        else if (strcmp(argv[index], "--dpus") == 0 && index + 1 < argc) { if (parse_u32(argv[++index], &dpus) != 0) return 2; }
        else if (strcmp(argv[index], "--tasklets") == 0 && index + 1 < argc) { if (parse_u32(argv[++index], &tasklets) != 0) return 2; }
        else if (strcmp(argv[index], "--initialization-binary") == 0 && index + 1 < argc) initialization_binary = argv[++index];
        else if (strcmp(argv[index], "--dpu-binary") == 0 && index + 1 < argc) dpu_binary = argv[++index];
        else if (strcmp(argv[index], "--timeout-s") == 0 && index + 1 < argc) { if (parse_u32(argv[++index], &timeout_s) != 0) return 2; }
        else if (strcmp(argv[index], "--wave-v5") == 0) wave_mode = 1;
        else { v4_usage(argv[0]); return 2; }
    }
    (void)signal(SIGINT, v4_signal_handler);
    (void)signal(SIGTERM, v4_signal_handler);
    (void)signal(SIGALRM, v4_signal_handler);
    if (target == NULL || session_root == NULL || initialization_binary == NULL || dpu_binary == NULL ||
        dpus == 0u || dpus > EXECUTION_PLAN_V4_MAX_DPUS || tasklets == 0u || tasklets > EXECUTION_PLAN_V4_MAX_TASKLETS ||
        timeout_s == 0u || realpath(session_root, root_real) == NULL || stat(root_real, &root_stat) != 0 ||
        !S_ISDIR(root_stat.st_mode) || !path_exists(initialization_binary) || !path_exists(dpu_binary)) {
        v4_emit_startup_failure("hardware_profile_violation", "invalid session arguments or paths");
        v4_emit_release();
        return 2;
    }
    if (tasklets != (uint32_t)NR_TASKLETS) {
        v4_emit_startup_failure("tasklet_binary_mismatch", "--tasklets must match host binary NR_TASKLETS");
        v4_emit_release();
        return 1;
    }
    if (strcmp(target, "hardware") == 0) {
        if (rank_path == NULL || getenv("UPMEM_ALLOW_PHYSICAL_HARDWARE") == NULL ||
            strcmp(getenv("UPMEM_ALLOW_PHYSICAL_HARDWARE"), "1") != 0) {
            v4_emit_startup_failure("hardware_opt_in_missing", "hardware v4 requires rank path and UPMEM_ALLOW_PHYSICAL_HARDWARE=1");
            v4_emit_release();
            return 1;
        }
        if (getenv("DPU_BACKEND") != NULL || getenv("UPMEM_EXECUTION_MODE") != NULL) {
            v4_emit_startup_failure("hardware_profile_violation", "hardware v4 forbids backend selectors");
            v4_emit_release();
            return 1;
        }
    } else if (strcmp(target, "simulator") == 0) {
        v4_simulator_target = 1;
        if (rank_path != NULL || getenv("DPU_BACKEND") == NULL ||
            strcmp(getenv("DPU_BACKEND"), "simulator") != 0 ||
            getenv("UPMEM_EXECUTION_MODE") != NULL) {
            v4_emit_startup_failure("simulator_profile_violation", "simulator v4 requires DPU_BACKEND=simulator and no rank path or execution selector");
            v4_emit_release();
            return 1;
        }
    } else {
        v4_emit_startup_failure("hardware_profile_violation", "--target must be hardware or simulator");
        v4_emit_release();
        return 2;
    }
    v4_session_rank_path = v4_simulator_target ? NULL : rank_path;
    v4_provider_initialized = 1;
    error = v4_simulator_target
        ? upmem_v4_provider_init_simulator(&v4_provider, dpus, initialization_binary)
        : upmem_v4_provider_init_on_rank(&v4_provider, dpus, rank_path, initialization_binary);
    if (error != DPU_OK || v4_provider.observed_dpus != dpus ||
        (v4_simulator_target ? v4_provider.observed_ranks < 1u : v4_provider.observed_ranks != 1u)) {
        v4_emit_startup_failure("hardware_allocation_failed", "v4 rank allocation did not match the requested DPU set");
        v4_emit_release();
        return 1;
    }
    struct dpu_program_t *program = NULL;
    error = dpu_load(v4_provider.set, dpu_binary, &program);
    if (error != DPU_OK) {
        v4_emit_startup_failure("binary_load_failed", "v4 DPU binary load failed");
        v4_emit_release();
        return 1;
    }
    if (wave_mode && (verify_wave_binary(program, tasklets) ||
            execution_plan_sha256_file(dpu_binary, wave_binary_sha256))) {
        v4_emit_startup_failure("tasklet_binary_mismatch", "v5 wave symbols or tasklet identity mismatch");
        v4_emit_release();
        return 1;
    }
    v4_emit_ready(rank_path, dpu_binary, initialization_binary, dpus, tasklets);
    rc = 0;
    while (!v4_interrupted) {
        char line[PATH_MAX * 2u];
        char command[32], path[PATH_MAX], digest[65], extra[8];
        int fields;
        if (fgets(line, sizeof(line), stdin) == NULL) break;
        fields = sscanf(line, "%31s %4095s %64s %7s", command, path, digest, extra);
        if (fields == 1 &&
            strcmp(command, "CLOSE") == 0) break;
        if (fields != 3 || strcmp(command, wave_mode ? "SUBMIT_PACKED_WAVES" : "SUBMIT_PACKED_OPERATION") != 0) {
            v4_emit_startup_failure("request_manifest_failed",
                wave_mode ? "expected SUBMIT_PACKED_WAVES <session-basename> <sha256> or CLOSE" :
                "expected SUBMIT_PACKED_OPERATION <safe-relative-envelope> <sha256> or CLOSE");
            rc = 1;
            if (wave_mode) break;
            continue;
        }
        if (wave_mode) {
            if (execute_wave_envelope(root_real, path, digest, dpus, tasklets, timeout_s)) {
                rc = 1;
                break;
            }
            continue;
        }
        if (execute_packed_operation(root_real, path, digest, dpus, tasklets, timeout_s) != 0) {
            rc = 1;
        }
    }
    v4_emit_release();
    return rc;
}
