#define _POSIX_C_SOURCE 200809L

#include <dpu.h>

#include <errno.h>
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
#include "simplepim_provider.h"
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
static uint64_t v4_last_request_sequence = 0u;
static int v4_have_request_sequence = 0;

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
    fputs(EXECUTION_PLAN_V4_NATIVE_BACKEND_ID, stdout);
    fputs("\",\"backend_family\":\"", stdout);
    fputs(EXECUTION_PLAN_V4_NATIVE_BACKEND_FAMILY, stdout);
    fputs("\",\"profile\":\"", stdout);
    fputs(EXECUTION_PLAN_V4_NATIVE_M5_PROFILE, stdout);
    fputs("\",\"abi\":\"", stdout);
    fputs(EXECUTION_PLAN_V4_NATIVE_ABI, stdout);
    fputs("\",\"session_protocol\":\"", stdout);
    fputs(EXECUTION_PLAN_V4_NATIVE_SESSION, stdout);
    fputs("\",\"dispatch_mode\":\"", stdout);
    fputs(EXECUTION_PLAN_V4_NATIVE_DISPATCH, stdout);
    fputs("\",\"kernel_identity\":\"", stdout);
    fputs(EXECUTION_PLAN_V4_NATIVE_KERNEL, stdout);
    fputs("\",\"execution_class\":\"", stdout);
    fputs(EXECUTION_PLAN_V4_NATIVE_EXECUTION_CLASS, stdout);
    fputs("\",\"target_observed\":\"physical_hardware\",\"rank_path\":", stdout);
    json_string(stdout, rank_path);
    printf(",\"requested_dpu_count\":%u,\"allocated_dpu_count\":%u,\"tasklets_per_dpu\":%u,\"hardware_allocation_verified\":true,\"native_kernel_executed\":false,\"simulator_kernel_executed\":false,\"cpu_fallback_used\":false,\"dpu_binary_sha256\":", dpus, dpus, tasklets);
    json_string(stdout, dpu_hash);
    fputs(",\"initialization_binary_sha256\":", stdout);
    json_string(stdout, init_hash);
    fputs(",\"session_release_pending\":true}\n", stdout);
    fflush(stdout);
}

static int copy_request_to_dpus(
    const execution_plan_v4_request_t *request,
    struct dpu_set_t set,
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

static void v4_emit_response(
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
    printf("{\"event\":\"RESPONSE\",\"status\":\"%s\",\"failure_stage\":",
        failure_stage == NULL ? "completed" : "failed");
    json_string(stdout, failure_stage);
    fputs(",\"error\":", stdout);
    json_string(stdout, error_message);
    fputs(",\"backend_id\":\"", stdout);
    fputs(EXECUTION_PLAN_V4_NATIVE_BACKEND_ID, stdout);
    fputs("\",\"backend_family\":\"", stdout);
    fputs(EXECUTION_PLAN_V4_NATIVE_BACKEND_FAMILY, stdout);
    fputs("\",\"profile\":\"", stdout);
    fputs(EXECUTION_PLAN_V4_NATIVE_M5_PROFILE, stdout);
    fputs("\",\"abi\":\"", stdout);
    fputs(EXECUTION_PLAN_V4_NATIVE_ABI, stdout);
    fputs("\",\"session_protocol\":\"", stdout);
    fputs(EXECUTION_PLAN_V4_NATIVE_SESSION, stdout);
    fputs("\",\"dispatch_mode\":\"", stdout);
    fputs(EXECUTION_PLAN_V4_NATIVE_DISPATCH, stdout);
    fputs("\",\"kernel_identity\":\"", stdout);
    fputs(EXECUTION_PLAN_V4_NATIVE_KERNEL, stdout);
    fputs("\",\"target_requested\":\"hardware\",\"target_observed\":\"physical_hardware\",\"execution_class\":\"", stdout);
    fputs(EXECUTION_PLAN_V4_NATIVE_EXECUTION_CLASS, stdout);
    fputs("\",\"rank_path\":", stdout);
    json_string(stdout, v4_session_rank_path);
    printf(",\"request_sequence\":%llu,\"request_output_elements\":%llu,\"global_output_elements\":%llu,\"global_completeness\":false,\"task_contract_sha256\":\"%s\",\"request_sha256\":\"%s\",\"request_manifest_sha256\":\"%s\",\"sidecar_sha256\":\"%s\",\"bulk_set_launch_verified\":%s,\"requested_dpu_count\":%u,\"allocated_dpu_count\":%u,\"tasklets_per_dpu\":%u,\"hardware_allocation_verified\":true,\"native_kernel_executed\":%s,\"simulator_kernel_executed\":false,\"hardware_kernel_executed\":%s,\"cpu_fallback_used\":false,\"session_release_pending\":true,\"timing_scope\":\"one_bulk_request_in_persistent_session\",\"request_timing_is_bringup_only\":true,\"request_level_speedup_applicable\":false,\"hardware_functionality_evidence\":%s,\"timing\":{\"h2d_time_s\":%.9f,\"launch_time_s\":%.9f,\"d2h_time_s\":%.9f,\"output_time_s\":%.9f,\"total_route_time_s\":%.9f},\"transfer\":{\"h2d_bytes\":%llu,\"d2h_bytes\":%llu,\"total_bytes\":%llu},\"per_dpu\":[",
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
        native_kernel_executed ? "true" : "false",
        native_kernel_executed ? "true" : "false",
        native_kernel_executed && failure_stage == NULL ? "true" : "false",
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
            if (index != 0u) fputc(',', stdout);
            fprintf(stdout, "{\"dpu_id\":%u,\"tile_id\":%llu,\"completion_status\":%u,\"cycles\":%llu,\"processed_elements\":%llu,\"h2d_bytes\":%llu,\"d2h_bytes\":%llu}",
                results[index].dpu_id, (unsigned long long)results[index].tile_id,
                results[index].completion_status, (unsigned long long)results[index].cycles,
                (unsigned long long)results[index].processed_elements,
                (unsigned long long)results[index].h2d_bytes,
                (unsigned long long)results[index].d2h_bytes);
        }
    }
    fputs("]}\n", stdout);
    fflush(stdout);
}

static int execute_request(
    const char *session_root,
    const char *manifest_path,
    const char *manifest_sha256,
    uint32_t dpus,
    uint32_t tasklets,
    uint32_t timeout_s
) {
    execution_plan_v4_request_t request = {0};
    v4_request_metrics_t metrics = {0};
    v4_dpu_result_t results[EXECUTION_PLAN_V4_MAX_DPUS] = {0};
    char *error_message = NULL;
    const char *failure_stage = NULL;
    dpu_error_t error;
    int native_kernel_executed = 0;
    const double route_started = now_s();
    if (execution_plan_v4_request_load(session_root, manifest_path, manifest_sha256,
        dpus, tasklets, &request, &error_message) != 0) {
        failure_stage = "request_manifest_failed";
        metrics.total_route_time_s = now_s() - route_started;
        v4_emit_response(&request, &metrics, results, failure_stage, error_message, 0, 0);
        free(error_message);
        execution_plan_v4_request_free(&request);
        return 1;
    }
    if (execution_plan_v4_request_load_payloads(&request, &error_message) != 0) {
        failure_stage = "payload_validation_failed";
        metrics.total_route_time_s = now_s() - route_started;
        v4_emit_response(&request, &metrics, results, failure_stage, error_message, 0, 0);
        free(error_message);
        execution_plan_v4_request_free(&request);
        return 1;
    }
    if (v4_have_request_sequence && request.header.request_sequence <= v4_last_request_sequence) {
        failure_stage = "hardware_profile_violation";
        metrics.total_route_time_s = now_s() - route_started;
        v4_emit_response(&request, &metrics, results, failure_stage,
            "v4 request_sequence must increase for each SUBMIT", 0, 0);
        execution_plan_v4_request_free(&request);
        return 1;
    }
    v4_last_request_sequence = request.header.request_sequence;
    v4_have_request_sequence = 1;
    if (v4_interrupted) {
        failure_stage = v4_timeout ? "kernel_timeout" : "kernel_launch_failed";
        metrics.total_route_time_s = now_s() - route_started;
        v4_emit_response(&request, &metrics, results, failure_stage, "request interrupted before launch", 0, 0);
        execution_plan_v4_request_free(&request);
        return 1;
    }
    if (copy_request_to_dpus(&request, v4_provider.set, &metrics, &error_message) != 0) {
        failure_stage = "argument_transfer_failed";
        metrics.total_route_time_s = now_s() - route_started;
        v4_emit_response(&request, &metrics, results, failure_stage, error_message, 0, 0);
        free(error_message);
        execution_plan_v4_request_free(&request);
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
        v4_emit_response(&request, &metrics, results, failure_stage, "bulk synchronous v4 launch failed", 0,
            error == DPU_OK);
        execution_plan_v4_request_free(&request);
        return 1;
    }
    native_kernel_executed = 1;
    {
        const double started = now_s();
        if (collect_request_from_dpus(&request, v4_provider.set, &metrics, results, &error_message) != 0) {
            failure_stage = "result_transfer_failed";
        }
        metrics.output_time_s = now_s() - started;
    }
    metrics.total_route_time_s = now_s() - route_started;
    v4_emit_response(&request, &metrics, results, failure_stage, error_message, native_kernel_executed, 1);
    free(error_message);
    execution_plan_v4_request_free(&request);
    return failure_stage == NULL ? 0 : 1;
}

static void v4_usage(const char *program) {
    fprintf(stderr, "usage: %s --session-root DIR --rank-path /dev/dpu_rankN --dpus N --tasklets N --initialization-binary PATH --dpu-binary PATH [--timeout-s N]\n", program);
}

int main(int argc, char **argv) {
    const char *session_root = NULL;
    const char *rank_path = NULL;
    const char *initialization_binary = NULL;
    const char *dpu_binary = NULL;
    uint32_t dpus = 0u, tasklets = 0u, timeout_s = 60u;
    char root_real[PATH_MAX];
    struct stat root_stat;
    dpu_error_t error;
    int rc = 1;
    for (int index = 1; index < argc; index++) {
        if (strcmp(argv[index], "--session-root") == 0 && index + 1 < argc) session_root = argv[++index];
        else if (strcmp(argv[index], "--rank-path") == 0 && index + 1 < argc) rank_path = argv[++index];
        else if (strcmp(argv[index], "--dpus") == 0 && index + 1 < argc) { if (parse_u32(argv[++index], &dpus) != 0) return 2; }
        else if (strcmp(argv[index], "--tasklets") == 0 && index + 1 < argc) { if (parse_u32(argv[++index], &tasklets) != 0) return 2; }
        else if (strcmp(argv[index], "--initialization-binary") == 0 && index + 1 < argc) initialization_binary = argv[++index];
        else if (strcmp(argv[index], "--dpu-binary") == 0 && index + 1 < argc) dpu_binary = argv[++index];
        else if (strcmp(argv[index], "--timeout-s") == 0 && index + 1 < argc) { if (parse_u32(argv[++index], &timeout_s) != 0) return 2; }
        else { v4_usage(argv[0]); return 2; }
    }
    (void)signal(SIGINT, v4_signal_handler);
    (void)signal(SIGTERM, v4_signal_handler);
    (void)signal(SIGALRM, v4_signal_handler);
    if (session_root == NULL || rank_path == NULL || initialization_binary == NULL || dpu_binary == NULL ||
        dpus == 0u || dpus > EXECUTION_PLAN_V4_MAX_DPUS || tasklets == 0u || tasklets > EXECUTION_PLAN_V4_MAX_TASKLETS ||
        timeout_s == 0u || realpath(session_root, root_real) == NULL || stat(root_real, &root_stat) != 0 ||
        !S_ISDIR(root_stat.st_mode) || !path_exists(initialization_binary) || !path_exists(dpu_binary)) {
        v4_emit_startup_failure("hardware_profile_violation", "invalid session arguments or paths");
        v4_emit_release();
        return 2;
    }
    if (getenv("UPMEM_ALLOW_PHYSICAL_HARDWARE") == NULL ||
        strcmp(getenv("UPMEM_ALLOW_PHYSICAL_HARDWARE"), "1") != 0) {
        v4_emit_startup_failure("hardware_opt_in_missing", "UPMEM_ALLOW_PHYSICAL_HARDWARE=1 is required");
        v4_emit_release();
        return 1;
    }
    if (getenv("DPU_BACKEND") != NULL) {
        v4_emit_startup_failure("hardware_profile_violation", "DPU_BACKEND must be unset for v4 physical execution");
        v4_emit_release();
        return 1;
    }
    if (getenv("UPMEM_EXECUTION_MODE") != NULL) {
        v4_emit_startup_failure("hardware_profile_violation", "UPMEM_EXECUTION_MODE must be unset for v4 physical execution");
        v4_emit_release();
        return 1;
    }
    v4_session_rank_path = rank_path;
    v4_provider_initialized = 1;
    error = upmem_v4_provider_init_on_rank(&v4_provider, dpus, rank_path, initialization_binary);
    if (error != DPU_OK || v4_provider.observed_dpus != dpus || v4_provider.observed_ranks != 1u) {
        v4_emit_startup_failure("hardware_allocation_failed", "v4 rank allocation did not match the requested DPU set");
        v4_emit_release();
        return 1;
    }
    error = dpu_load(v4_provider.set, dpu_binary, NULL);
    if (error != DPU_OK) {
        v4_emit_startup_failure("binary_load_failed", "v4 DPU binary load failed");
        v4_emit_release();
        return 1;
    }
    v4_emit_ready(rank_path, dpu_binary, initialization_binary, dpus, tasklets);
    rc = 0;
    while (!v4_interrupted) {
        char line[PATH_MAX * 2u];
        char command[16], manifest[PATH_MAX], digest[65], extra[8];
        if (fgets(line, sizeof(line), stdin) == NULL) break;
        if (sscanf(line, "%15s %4095s %64s %7s", command, manifest, digest, extra) == 1 &&
            strcmp(command, "CLOSE") == 0) break;
        if (sscanf(line, "%15s %4095s %64s %7s", command, manifest, digest, extra) != 3 ||
            strcmp(command, "SUBMIT") != 0) {
            v4_emit_startup_failure("request_manifest_failed", "expected SUBMIT <safe-relative-manifest> <sha256> or CLOSE");
            rc = 1;
            continue;
        }
        if (execute_request(root_real, manifest, digest, dpus, tasklets, timeout_s) != 0) rc = 1;
    }
    v4_emit_release();
    return rc;
}
