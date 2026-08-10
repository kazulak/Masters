#include <dpu.h>

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/time.h>

#include "common.h"
#include "session_protocol.h"

#ifndef UPMEM_GENERIC_HARDWARE_MVP
#define UPMEM_GENERIC_HARDWARE_MVP 1
#endif

#if !UPMEM_GENERIC_HARDWARE_MVP
#error "resident hardware evidence requires UPMEM_GENERIC_HARDWARE_MVP=1"
#endif

#define RESIDENT_ALLOCATION_PROFILE "backend=hw"

static volatile sig_atomic_t resident_interrupted = 0;

static void resident_signal_handler(int signal_number) {
    (void)signal_number;
    resident_interrupted = 1;
}

static double resident_now_s(void) {
    struct timeval value;
    gettimeofday(&value, NULL);
    return (double)value.tv_sec + (double)value.tv_usec / 1000000.0;
}

static void resident_report_sdk_error(const char *operation, dpu_error_t error) {
    fprintf(stderr, "%s failed: %s\n", operation, dpu_error_to_string(error));
}

static int resident_read_exact(const char *path, void *buffer, size_t bytes) {
    FILE *file = fopen(path, "rb");
    size_t read_bytes;
    if (file == NULL) return 1;
    read_bytes = fread(buffer, 1, bytes, file);
    fclose(file);
    return read_bytes == bytes ? 0 : 1;
}

static int resident_write_exact(const char *path, const void *buffer, size_t bytes) {
    FILE *file = fopen(path, "wb");
    size_t written;
    if (file == NULL) return 1;
    written = fwrite(buffer, 1, bytes, file);
    if (fclose(file) != 0) return 1;
    return written == bytes ? 0 : 1;
}

static int resident_file_size(const char *path, size_t expected) {
    struct stat value;
    if (stat(path, &value) != 0 || value.st_size < 0) return 1;
    return (uintmax_t)value.st_size == (uintmax_t)expected ? 0 : 1;
}

static int resident_buffer_finite(const unsigned char *buffer, size_t bytes) {
    for (size_t offset = 0; offset < bytes; offset += sizeof(float)) {
        float value;
        memcpy(&value, buffer + offset, sizeof(value));
        if (!isfinite(value)) return 1;
    }
    return 0;
}

static void resident_release(
    struct dpu_set_t *set,
    int allocated,
    int *release_confirmed,
    dpu_error_t *error,
    const char **failure_stage,
    int *sdk_error_code,
    resident_timing_t *timing
) {
    if (!allocated) return;
    const double started = resident_now_s();
    *error = dpu_free(*set);
    timing->release_time_s = resident_now_s() - started;
    if (*error == DPU_OK) {
        *release_confirmed = 1;
    } else {
        *release_confirmed = 0;
        resident_report_sdk_error("resident dpu_free", *error);
        *sdk_error_code = (int)*error;
        *failure_stage = "hardware_release_failed";
    }
}

static const char *resident_parse_failure_stage(const char *message) {
    if (message == NULL) return "manifest_parse_failed";
    if (strstr(message, "package") != NULL) return "hardware_profile_violation";
    if (strstr(message, "profile") != NULL) return "hardware_profile_violation";
    return "manifest_parse_failed";
}

int main(int argc, char **argv) {
    resident_request_t request;
    resident_timing_t timing = {0};
    char *error_message = NULL;
    const char *failure_stage = NULL;
    int sdk_error_code = -1;
    uint32_t allocated_dpus = 0;
    uint32_t native_launch_count = 0;
    uint64_t initial_h2d_bytes = 0;
    uint64_t descriptor_h2d_bytes = 0;
    uint64_t control_h2d_bytes = 0;
    uint64_t final_d2h_bytes = 0;
    unsigned char **input_buffers = NULL;
    unsigned char **output_buffers = NULL;
    struct dpu_set_t set;
    dpu_error_t error = DPU_OK;
    int set_allocated = 0;
    int release_confirmed = 0;
    int rc = 1;
    static char interrupted_message[] = "resident host interrupted; release confirmation is required";
    static char output_nonfinite_message[] = "resident final output contains non-finite values";

    memset(&request, 0, sizeof(request));
    if (argc != 5 || strcmp(argv[1], "--resident-package") != 0 || strcmp(argv[3], "--resident-response") != 0) {
        fprintf(stderr, "usage: %s --resident-package request.json --resident-response response.json\n", argv[0]);
        return 2;
    }
    signal(SIGTERM, resident_signal_handler);
    signal(SIGINT, resident_signal_handler);
    request.session_id = (char *)malloc(20u);
    if (request.session_id != NULL) snprintf(request.session_id, 20u, "%s", "resident-unknown");
    {
        const double started = resident_now_s();
        if (resident_request_load(argv[2], &request, &error_message) != 0) {
            failure_stage = resident_parse_failure_stage(error_message);
            timing.package_parse_time_s = resident_now_s() - started;
            goto write_response;
        }
        timing.package_parse_time_s = resident_now_s() - started;
    }
    if (getenv("UPMEM_ALLOW_PHYSICAL_HARDWARE") == NULL || strcmp(getenv("UPMEM_ALLOW_PHYSICAL_HARDWARE"), "1") != 0) {
        failure_stage = "hardware_opt_in_missing";
        error_message = (char *)"UPMEM_ALLOW_PHYSICAL_HARDWARE=1 is required";
        goto release_and_write;
    }
    if (getenv("DPU_BACKEND") != NULL) {
        failure_stage = "hardware_profile_violation";
        error_message = (char *)"DPU_BACKEND must be unset";
        goto release_and_write;
    }
    if (NR_TASKLETS < 1 || NR_TASKLETS > 16 || request.header.operation_count == 0u || request.header.operation_count > RESIDENT_MAX_COMPONENT_OPS) {
        failure_stage = "hardware_profile_violation";
        error_message = (char *)"resident host requires 1 <= NR_TASKLETS <= 16 and bounded descriptors";
        goto release_and_write;
    }
    if (resident_interrupted) {
        failure_stage = "hardware_session_interrupted";
        error_message = interrupted_message;
        goto release_and_write;
    }

    input_buffers = (unsigned char **)calloc(request.input_count, sizeof(*input_buffers));
    output_buffers = (unsigned char **)calloc(request.final_count, sizeof(*output_buffers));
    if ((request.input_count != 0u && input_buffers == NULL) || (request.final_count != 0u && output_buffers == NULL)) {
        failure_stage = "hardware_allocation_failed";
        goto release_and_write;
    }
    for (size_t index = 0; index < request.input_count; index++) {
        if (resident_interrupted) {
            failure_stage = "hardware_session_interrupted";
            error_message = interrupted_message;
            goto release_and_write;
        }
        const resident_input_file_t *input = &request.inputs[index];
        if (input->slot_id >= request.header.slot_count ||
            resident_file_size(input->path, input->raw_bytes) != 0) {
            failure_stage = "initial_transfer_failed";
            goto release_and_write;
        }
        input_buffers[index] = (unsigned char *)calloc(input->transfer_bytes, 1u);
        if (input_buffers[index] == NULL || resident_read_exact(input->path, input_buffers[index], input->raw_bytes) != 0 ||
            resident_buffer_finite(input_buffers[index], input->raw_bytes) != 0) {
            failure_stage = "initial_transfer_failed";
            goto release_and_write;
        }
    }
    for (size_t index = 0; index < request.final_count; index++) {
        const resident_final_file_t *output = &request.final_outputs[index];
        if (output->slot_id >= request.header.slot_count ||
            request.slots[output->slot_id].capacity_elements < output->elements) {
            failure_stage = "hardware_profile_violation";
            goto release_and_write;
        }
        output_buffers[index] = (unsigned char *)calloc(output->transfer_bytes, 1u);
        if (output_buffers[index] == NULL) {
            failure_stage = "hardware_allocation_failed";
            goto release_and_write;
        }
    }

    {
        const double started = resident_now_s();
        error = dpu_alloc(1, RESIDENT_ALLOCATION_PROFILE, &set);
        timing.allocation_time_s = resident_now_s() - started;
        if (error != DPU_OK) {
            resident_report_sdk_error("resident dpu_alloc", error);
            sdk_error_code = (int)error;
            failure_stage = error == DPU_ERR_INVALID_PROFILE ? "hardware_profile_violation" : "hardware_allocation_failed";
            goto release_and_write;
        }
        set_allocated = 1;
        error = dpu_get_nr_dpus(set, &allocated_dpus);
        if (error != DPU_OK || allocated_dpus != 1u) {
            if (error != DPU_OK) {
                resident_report_sdk_error("resident dpu_get_nr_dpus", error);
                sdk_error_code = (int)error;
            }
            failure_stage = "hardware_allocation_failed";
            goto release_and_write;
        }
    }
    {
        const double started = resident_now_s();
        error = dpu_load(set, request.dpu_binary_path, NULL);
        timing.binary_load_time_s = resident_now_s() - started;
        if (error != DPU_OK) {
            resident_report_sdk_error("resident dpu_load", error);
            sdk_error_code = (int)error;
            failure_stage = "binary_load_failed";
            goto release_and_write;
        }
    }
    {
        const double started = resident_now_s();
        uint32_t slot_count = request.header.slot_count;
        uint32_t operation_count = request.header.operation_count;
        uint32_t pool_bytes = request.header.pool_bytes;
        resident_control_t control = {slot_count, operation_count, pool_bytes, 0u};
        error = dpu_broadcast_to(set, "RESIDENT_SLOT_DESCRIPTORS", 0, request.slots,
            request.header.slot_bytes, DPU_XFER_DEFAULT);
        if (error == DPU_OK) error = dpu_broadcast_to(set, "RESIDENT_OPERATIONS", 0, request.operations,
            request.header.operation_bytes, DPU_XFER_DEFAULT);
        timing.descriptor_h2d_time_s = resident_now_s() - started;
        descriptor_h2d_bytes = request.header.slot_bytes + request.header.operation_bytes;
        if (error != DPU_OK) {
            resident_report_sdk_error("resident descriptor transfer", error);
            sdk_error_code = (int)error;
            failure_stage = "descriptor_transfer_failed";
            goto release_and_write;
        }
        const double control_started = resident_now_s();
        error = dpu_broadcast_to(set, "RESIDENT_CONTROL", 0, &control, sizeof(control), DPU_XFER_DEFAULT);
        timing.control_h2d_time_s += resident_now_s() - control_started;
        control_h2d_bytes = sizeof(control);
        if (error != DPU_OK) {
            resident_report_sdk_error("resident control transfer", error);
            sdk_error_code = (int)error;
            failure_stage = "descriptor_transfer_failed";
            goto release_and_write;
        }
    }
    {
        const double started = resident_now_s();
        for (size_t index = 0; index < request.input_count; index++) {
            const resident_input_file_t *input = &request.inputs[index];
            error = dpu_broadcast_to(set, "RESIDENT_SLOT_POOL", request.slots[input->slot_id].offset_bytes,
                input_buffers[index], input->transfer_bytes, DPU_XFER_DEFAULT);
            if (error != DPU_OK) break;
            initial_h2d_bytes += input->transfer_bytes;
        }
        timing.initial_h2d_time_s = resident_now_s() - started;
        if (error != DPU_OK) {
            resident_report_sdk_error("resident initial slot transfer", error);
            sdk_error_code = (int)error;
            failure_stage = "initial_transfer_failed";
            goto release_and_write;
        }
    }
    for (uint32_t operation_index = 0; operation_index < request.header.operation_count; operation_index++) {
        if (resident_interrupted) {
            failure_stage = "hardware_session_interrupted";
            error_message = interrupted_message;
            goto release_and_write;
        }
        const double control_started = resident_now_s();
        const uint64_t aligned_operation_index = operation_index;
        error = dpu_broadcast_to(set, "RESIDENT_ACTIVE_OPERATION", 0, &aligned_operation_index, sizeof(aligned_operation_index), DPU_XFER_DEFAULT);
        timing.control_h2d_time_s += resident_now_s() - control_started;
        control_h2d_bytes += sizeof(aligned_operation_index);
        if (error != DPU_OK) {
            resident_report_sdk_error("resident active descriptor transfer", error);
            sdk_error_code = (int)error;
            failure_stage = "descriptor_transfer_failed";
            goto release_and_write;
        }
        const double kernel_started = resident_now_s();
        error = dpu_launch(set, DPU_SYNCHRONOUS);
        timing.kernel_time_s += resident_now_s() - kernel_started;
        if (error != DPU_OK) {
            resident_report_sdk_error("resident synchronous dpu_launch", error);
            sdk_error_code = (int)error;
            failure_stage = "kernel_launch_failed";
            goto release_and_write;
        }
        {
            resident_completion_t completion;
            struct dpu_set_t dpu_first;
            DPU_FOREACH(set, dpu_first) {
                if (dpu_copy_from(dpu_first, "RESIDENT_COMPLETION", 0, &completion, sizeof(completion)) == DPU_OK) {
                    timing.dpu_run_time_cycles += completion.dpu_run_time_cycles;
                }
                break;
            }
        }
        native_launch_count++;
    }
    if (resident_interrupted) {
        failure_stage = "hardware_session_interrupted";
        error_message = interrupted_message;
        goto release_and_write;
    }
    {
        const double started = resident_now_s();
        struct dpu_set_t dpu;
        for (size_t index = 0; index < request.final_count; index++) {
            const resident_final_file_t *output = &request.final_outputs[index];
            DPU_FOREACH(set, dpu) {
                error = dpu_copy_from(dpu, "RESIDENT_SLOT_POOL", request.slots[output->slot_id].offset_bytes,
                    output_buffers[index], output->transfer_bytes);
                break;
            }
            if (error != DPU_OK) break;
            if (resident_buffer_finite(output_buffers[index], output->raw_bytes) != 0) {
                failure_stage = "output_validation_failed";
                error_message = output_nonfinite_message;
                goto release_and_write;
            }
            final_d2h_bytes += output->transfer_bytes;
        }
        timing.final_d2h_time_s = resident_now_s() - started;
        if (error != DPU_OK) {
            resident_report_sdk_error("resident final output transfer", error);
            sdk_error_code = (int)error;
            failure_stage = "final_transfer_failed";
            goto release_and_write;
        }
    }
    {
        const double started = resident_now_s();
        for (size_t index = 0; index < request.final_count; index++) {
            resident_final_file_t *output = &request.final_outputs[index];
            if (resident_write_exact(output->path, output_buffers[index], output->raw_bytes) != 0) {
                failure_stage = "final_transfer_failed";
                goto release_and_write;
            }
            output->status = 1;
        }
        timing.output_write_time_s = resident_now_s() - started;
    }
    failure_stage = NULL;
    rc = 0;

release_and_write:
    resident_release(&set, set_allocated, &release_confirmed, &error, &failure_stage, &sdk_error_code, &timing);
    if (set_allocated && !release_confirmed && failure_stage == NULL) {
        failure_stage = "hardware_release_unverified";
    }
    if (failure_stage != NULL) rc = 1;
write_response:
    if (resident_response_write(
        argv[argc >= 5 ? 4 : 0], &request,
        failure_stage == NULL ? "completed" : "failed", failure_stage, error_message,
        allocated_dpus, sdk_error_code, &timing, native_launch_count, release_confirmed,
        initial_h2d_bytes, descriptor_h2d_bytes, control_h2d_bytes, final_d2h_bytes
    ) != 0) {
        rc = 1;
    }
    for (size_t index = 0; index < request.input_count; index++) free(input_buffers == NULL ? NULL : input_buffers[index]);
    for (size_t index = 0; index < request.final_count; index++) free(output_buffers == NULL ? NULL : output_buffers[index]);
    free(input_buffers);
    free(output_buffers);
    if (error_message != NULL && error_message != (char *)"UPMEM_ALLOW_PHYSICAL_HARDWARE=1 is required" && error_message != (char *)"DPU_BACKEND must be unset" && error_message != (char *)"resident host requires 1 <= NR_TASKLETS <= 16 and bounded descriptors" && error_message != interrupted_message && error_message != output_nonfinite_message) free(error_message);
    resident_request_free(&request);
    return rc;
}
