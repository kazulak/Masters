#include <dpu.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#include "../../lib/communication/CommOps.h"
#include "../../lib/management/Management.h"
#include "../../lib/processing/ProcessingHelperHost.h"
#include "../../lib/processing/map/Map.h"
#include "../../lib/processing/zip/Zip.h"
#include "Param.h"

#define PROVIDER_ID "simplepim"
#define PROBE_ID "simplepim_va_map_zip_v1"
#define HOST_SCHEMA_VERSION "simplepim_qualification_host_v2"
#define PHYSICAL_ALLOCATION_PROFILE "backend=hw"

static double elapsed_seconds(const struct timeval *start, const struct timeval *end) {
    return (double)(end->tv_sec - start->tv_sec) +
           (double)(end->tv_usec - start->tv_usec) / 1000000.0;
}

static void fill_input(T *values, uint32_t salt) {
    for (uint32_t i = 0; i < SIMPLEPIM_QUALIFICATION_ELEMENTS; ++i) {
        values[i] = 17U + ((i * 13U + salt * 5U) % 1000U);
    }
}

static bool write_values(const char *path, const T *values, uint32_t count) {
    FILE *file = fopen(path, "wb");
    if (file == NULL) {
        return false;
    }
    size_t written = fwrite(values, sizeof(T), count, file);
    bool closed = fclose(file) == 0;
    return written == count && closed;
}

static simplepim_management_t *allocate_management_metadata(void) {
    simplepim_management_t *management = calloc(1, sizeof(*management));
    if (management == NULL) {
        return NULL;
    }
    management->curr_space = 16;
    management->tables = calloc(management->curr_space, sizeof(*management->tables));
    management->zip_args = calloc(SIMPLEPIM_QUALIFICATION_DPU_COUNT, sizeof(*management->zip_args));
    management->map_args = calloc(SIMPLEPIM_QUALIFICATION_DPU_COUNT, sizeof(*management->map_args));
    management->red_args = calloc(SIMPLEPIM_QUALIFICATION_DPU_COUNT, sizeof(*management->red_args));
    if (management->tables == NULL || management->zip_args == NULL ||
        management->map_args == NULL || management->red_args == NULL) {
        free(management->tables);
        free(management->zip_args);
        free(management->map_args);
        free(management->red_args);
        free(management);
        return NULL;
    }
    management->num_dpus = SIMPLEPIM_QUALIFICATION_DPU_COUNT;
    return management;
}

static void free_management_metadata(simplepim_management_t *management) {
    if (management == NULL) {
        return;
    }
    for (uint32_t i = 0; i < management->num_tables; ++i) {
        table_host_t *table = management->tables[i];
        if (table != NULL) {
            free(table->name);
            free(table->lens_each_dpu);
            free(table);
        }
    }
    free(management->tables);
    free(management->zip_args);
    free(management->map_args);
    free(management->red_args);
    free(management);
}

static void print_host_result(
    const char *status,
    const char *failure_stage,
    const char *reason,
    bool observed_dpus_valid,
    uint32_t observed_dpus,
    bool native_run_completed,
    bool validation_performed,
    bool host_exact_validation,
    const char *release_status,
    double input_time,
    double kernel_time,
    double output_time) {
    char observed_dpus_json[32];
    if (observed_dpus_valid) {
        snprintf(observed_dpus_json, sizeof(observed_dpus_json), "%u", observed_dpus);
    } else {
        strcpy(observed_dpus_json, "null");
    }
    printf(
        "{\"schema_version\":\"%s\",\"provider_id\":\"%s\",\"probe_id\":\"%s\","
        "\"status\":\"%s\",\"backend_profile\":\"%s\","
        "\"requested_dpu_count\":1,\"observed_dpu_count\":%s,"
        "\"configured_tasklets_per_dpu\":12,\"observed_tasklets_per_dpu\":null,"
        "\"native_run_completed\":%s,\"validation_performed\":%s,"
        "\"host_exact_validation\":%s,\"fallback\":false,"
        "\"release_status\":\"%s\",\"logical_input_bytes\":%u,"
        "\"logical_output_bytes\":%u,\"physical_transfer_bytes_available\":false,"
        "\"physical_transfer_bytes\":null,"
        "\"timing\":{\"input_s\":%.9f,\"kernel_s\":%.9f,\"output_s\":%.9f},"
        "\"failure_stage\":%s,\"reason\":%s%s%s}\n",
        HOST_SCHEMA_VERSION,
        PROVIDER_ID,
        PROBE_ID,
        status,
        PHYSICAL_ALLOCATION_PROFILE,
        observed_dpus_json,
        native_run_completed ? "true" : "false",
        validation_performed ? "true" : "false",
        host_exact_validation ? "true" : "false",
        release_status,
        (unsigned)SIMPLEPIM_QUALIFICATION_INPUT_BYTES,
        (unsigned)SIMPLEPIM_QUALIFICATION_OUTPUT_BYTES,
        input_time,
        kernel_time,
        output_time,
        failure_stage == NULL ? "null" : failure_stage,
        reason == NULL ? "" : "\"",
        reason == NULL ? "null" : reason,
        reason == NULL ? "" : "\"");
    fflush(stdout);
}

int main(int argc, char **argv) {
    int exit_code = 1;
    const char *failure_stage = "\"host_setup\"";
    const char *reason = "host_setup_failed";
    const char *release_status = "not_attempted";
    bool dpu_allocated = false;
    bool observed_dpus_valid = false;
    bool native_run_completed = false;
    bool validation_performed = false;
    bool host_exact_validation = false;
    uint32_t observed_dpus = 0;
    double input_time = 0.0;
    double kernel_time = 0.0;
    double output_time = 0.0;
    struct timeval started;
    struct timeval input_done;
    struct timeval kernel_done;
    struct timeval output_done;
    simplepim_management_t *management = NULL;
    T *a = NULL;
    T *b = NULL;
    T *expected = NULL;
    T *table_a = NULL;
    T *table_b = NULL;
    T *result = NULL;

    if (argc != 4) {
        failure_stage = "\"arguments\"";
        reason = "expected_input_a_input_b_output_paths";
        goto cleanup;
    }

    a = calloc(SIMPLEPIM_QUALIFICATION_ELEMENTS, sizeof(T));
    b = calloc(SIMPLEPIM_QUALIFICATION_ELEMENTS, sizeof(T));
    expected = calloc(SIMPLEPIM_QUALIFICATION_ELEMENTS, sizeof(T));
    if (a == NULL || b == NULL || expected == NULL) {
        failure_stage = "\"host_allocation\"";
        reason = "host_input_allocation_failed";
        goto cleanup;
    }
    fill_input(a, 0U);
    fill_input(b, 1U);
    for (uint32_t i = 0; i < SIMPLEPIM_QUALIFICATION_ELEMENTS; ++i) {
        expected[i] = a[i] + b[i];
    }
    if (!write_values(argv[1], a, SIMPLEPIM_QUALIFICATION_ELEMENTS) ||
        !write_values(argv[2], b, SIMPLEPIM_QUALIFICATION_ELEMENTS)) {
        failure_stage = "\"input_write\"";
        reason = "input_write_failed";
        goto cleanup;
    }

    management = allocate_management_metadata();
    if (management == NULL) {
        failure_stage = "\"management_allocation\"";
        reason = "management_allocation_failed";
        goto cleanup;
    }
    dpu_error_t error = dpu_alloc(
        SIMPLEPIM_QUALIFICATION_DPU_COUNT,
        PHYSICAL_ALLOCATION_PROFILE,
        &management->set);
    if (error != DPU_OK) {
        failure_stage = "\"dpu_allocation\"";
        reason = "physical_dpu_allocation_failed";
        goto cleanup;
    }
    dpu_allocated = true;
    error = dpu_get_nr_dpus(management->set, &observed_dpus);
    if (error != DPU_OK) {
        failure_stage = "\"dpu_observation\"";
        reason = "dpu_count_observation_failed";
        goto cleanup;
    }
    observed_dpus_valid = true;
    if (observed_dpus != SIMPLEPIM_QUALIFICATION_DPU_COUNT) {
        failure_stage = "\"dpu_observation\"";
        reason = "observed_dpu_count_mismatch";
        goto cleanup;
    }
    error = dpu_load(management->set, "bin/dpu_init_binary", NULL);
    if (error != DPU_OK) {
        failure_stage = "\"initialization_load\"";
        reason = "initialization_binary_load_failed";
        goto cleanup;
    }
    error = dpu_launch(management->set, DPU_SYNCHRONOUS);
    if (error != DPU_OK) {
        failure_stage = "\"initialization_launch\"";
        reason = "initialization_launch_failed";
        goto cleanup;
    }

    gettimeofday(&started, NULL);
    table_a = (T *)malloc_scatter_aligned(
        SIMPLEPIM_QUALIFICATION_ELEMENTS, sizeof(T), management);
    table_b = (T *)malloc_scatter_aligned(
        SIMPLEPIM_QUALIFICATION_ELEMENTS, sizeof(T), management);
    if (table_a == NULL || table_b == NULL) {
        failure_stage = "\"aligned_allocation\"";
        reason = "aligned_input_allocation_failed";
        goto cleanup;
    }
    memcpy(table_a, a, SIMPLEPIM_QUALIFICATION_ELEMENTS * sizeof(T));
    memcpy(table_b, b, SIMPLEPIM_QUALIFICATION_ELEMENTS * sizeof(T));
    simplepim_scatter(
        "t1", table_a, SIMPLEPIM_QUALIFICATION_ELEMENTS, sizeof(T), management);
    simplepim_scatter(
        "t2", table_b, SIMPLEPIM_QUALIFICATION_ELEMENTS, sizeof(T), management);
    gettimeofday(&input_done, NULL);
    input_time = elapsed_seconds(&started, &input_done);

    handle_t zip_handle = {
        .bin_location = "bin/dpu_zip", .so_bin_location = NULL, .func_type = ZIP};
    handle_t map_handle = {
        .bin_location = "bin/dpu_map_va_funcs", .so_bin_location = NULL, .func_type = MAP};
    table_zip("t1", "t2", "t3", &zip_handle, management);
    table_map("t3", "t4", sizeof(T), &map_handle, management, 0);
    gettimeofday(&kernel_done, NULL);
    kernel_time = elapsed_seconds(&input_done, &kernel_done);

    result = (T *)simplepim_gather("t4", management);
    if (result == NULL) {
        failure_stage = "\"result_gather\"";
        reason = "result_gather_failed";
        goto cleanup;
    }
    native_run_completed = true;
    validation_performed = true;
    host_exact_validation = true;
    for (uint32_t i = 0; i < SIMPLEPIM_QUALIFICATION_ELEMENTS; ++i) {
        if (result[i] != expected[i]) {
            host_exact_validation = false;
            break;
        }
    }
    if (!write_values(argv[3], result, SIMPLEPIM_QUALIFICATION_ELEMENTS)) {
        failure_stage = "\"output_write\"";
        reason = "output_write_failed";
        goto cleanup;
    }
    gettimeofday(&output_done, NULL);
    output_time = elapsed_seconds(&kernel_done, &output_done);
    if (!host_exact_validation) {
        failure_stage = "\"host_exact_validation\"";
        reason = "host_exact_validation_failed";
        goto cleanup;
    }
    exit_code = 0;
    failure_stage = NULL;
    reason = NULL;

cleanup:
    if (dpu_allocated) {
        dpu_error_t release_error = dpu_free(management->set);
        release_status = release_error == DPU_OK ? "released" : "failed";
        if (release_error != DPU_OK && exit_code == 0) {
            exit_code = 2;
            failure_stage = "\"release\"";
            reason = "dpu_release_failed";
        }
    }
    free(result);
    free(table_a);
    free(table_b);
    free(a);
    free(b);
    free(expected);
    free_management_metadata(management);
    print_host_result(
        exit_code == 0 ? "passed" : "failed",
        failure_stage,
        reason,
        observed_dpus_valid,
        observed_dpus,
        native_run_completed,
        validation_performed,
        host_exact_validation,
        release_status,
        input_time,
        kernel_time,
        output_time);
    return exit_code;
}
