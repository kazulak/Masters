#define _POSIX_C_SOURCE 200809L

#include <dpu.h>

#include <errno.h>
#include <limits.h>
#include <math.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "execution_plan_common.h"
#include "execution_plan_provider.h"
#include "plan_request.h"

#ifndef PATH_MAX
#define PATH_MAX 4096
#endif

typedef struct {
    uint64_t descriptor_h2d_bytes;
    uint64_t operand_h2d_bytes;
    uint64_t reset_h2d_bytes;
    uint64_t active_operation_h2d_bytes;
    uint64_t completion_d2h_bytes;
    uint64_t cross_d2h_bytes;
    uint64_t cross_h2d_bytes;
    uint64_t final_d2h_bytes;
    uint64_t launch_count;
    uint64_t synchronize_count;
    uint64_t completion_reads;
    uint64_t cross_dpu_edge_count;
    uint32_t completed_per_dpu[EXECUTION_PLAN_MAX_DPUS];
    uint64_t distributed_dpu_cycles[EXECUTION_PLAN_V2_MAX_DPUS];
    uint64_t distributed_dpu_work_elements[EXECUTION_PLAN_V2_MAX_DPUS];
    uint64_t distributed_dpu_checksums[EXECUTION_PLAN_V2_MAX_DPUS];
    uint32_t distributed_dpu_completions[EXECUTION_PLAN_V2_MAX_DPUS];
    uint32_t distributed_dpu_output_offsets[EXECUTION_PLAN_V2_MAX_DPUS];
    uint32_t distributed_dpu_output_elements[EXECUTION_PLAN_V2_MAX_DPUS];
    uint64_t reduction_d2h_bytes;
    uint64_t reduction_partial_reads;
    uint64_t reduction_element_additions;
    uint64_t reduction_accumulator_resets;
    uint32_t reduction_participant_count;
    int native_kernel_executed;
} execution_plan_metrics_t;

typedef struct {
    double allocation_time_s;
    double binary_load_time_s;
    double descriptor_h2d_time_s;
    double operand_h2d_time_s;
    double cross_dpu_transfer_time_s;
    double launch_sync_time_s;
    double final_d2h_time_s;
    double reduction_time_s;
    double output_write_time_s;
    double release_time_s;
} execution_plan_timing_t;

static volatile sig_atomic_t execution_plan_interrupted = 0;

static void execution_plan_signal_handler(int signal_number) {
    (void)signal_number;
    execution_plan_interrupted = 1;
}

static double now_s(void) {
    struct timespec value;
    if (clock_gettime(CLOCK_MONOTONIC, &value) != 0) return 0.0;
    return (double)value.tv_sec + (double)value.tv_nsec / 1000000000.0;
}

static uint32_t align8(uint32_t bytes) {
    return (bytes + 7u) & ~7u;
}

static int read_exact(const char *path, void *buffer, size_t bytes) {
    FILE *file = path == NULL ? NULL : fopen(path, "rb");
    size_t read_bytes;
    int close_failed;
    if (file == NULL) return 1;
    read_bytes = fread(buffer, 1u, bytes, file);
    close_failed = fclose(file) != 0;
    return read_bytes != bytes || close_failed;
}

static int write_exact(const char *path, const void *buffer, size_t bytes) {
    FILE *file = path == NULL ? NULL : fopen(path, "wb");
    size_t written;
    int close_failed;
    if (file == NULL) return 1;
    written = fwrite(buffer, 1u, bytes, file);
    close_failed = fclose(file) != 0;
    return written != bytes || close_failed;
}

static int file_size_matches(const char *path, size_t expected) {
    FILE *file = path == NULL ? NULL : fopen(path, "rb");
    long size;
    if (file == NULL || fseek(file, 0, SEEK_END) != 0) {
        if (file != NULL) fclose(file);
        return 1;
    }
    size = ftell(file);
    fclose(file);
    return size < 0 || (uintmax_t)size != (uintmax_t)expected;
}

static int buffer_is_finite(const unsigned char *buffer, size_t bytes) {
    for (size_t offset = 0u; offset < bytes; offset += sizeof(float)) {
        float value;
        memcpy(&value, buffer + offset, sizeof(value));
        if (!isfinite(value)) return 1;
    }
    return 0;
}

static void json_string(FILE *file, const char *value) {
    if (value == NULL) {
        fputs("null", file);
        return;
    }
    fputc('"', file);
    for (const unsigned char *cursor = (const unsigned char *)value; *cursor != '\0'; cursor++) {
        if (*cursor == '"' || *cursor == '\\') fprintf(file, "\\%c", *cursor);
        else if (*cursor == '\n') fputs("\\n", file);
        else if (*cursor == '\r') fputs("\\r", file);
        else if (*cursor == '\t') fputs("\\t", file);
        else if (*cursor < 0x20u) fprintf(file, "\\u%04x", *cursor);
        else fputc(*cursor, file);
    }
    fputc('"', file);
}

static const char *sdk_error_text(dpu_error_t error) {
    return error == DPU_OK ? "DPU_OK" : dpu_error_to_string(error);
}

static int resolve_sibling(const char *binary_name, char output[PATH_MAX]) {
    char executable[PATH_MAX];
    ssize_t length = readlink("/proc/self/exe", executable, sizeof(executable) - 1u);
    char *slash;
    int written;
    if (length <= 0 || (size_t)length >= sizeof(executable)) return 1;
    executable[length] = '\0';
    slash = strrchr(executable, '/');
    if (slash == NULL) return 1;
    *slash = '\0';
    written = snprintf(output, PATH_MAX, "%s/%s", executable, binary_name);
    return written < 0 || (size_t)written >= PATH_MAX || access(output, R_OK) != 0;
}

static int dpu_handle_at(struct dpu_set_t set, uint32_t wanted, struct dpu_set_t *result) {
    struct dpu_set_t dpu;
    uint32_t index;
    DPU_FOREACH(set, dpu, index) {
        if (index == wanted) {
            *result = dpu;
            return 0;
        }
    }
    return 1;
}

static void report_sdk_error(const char *operation, dpu_error_t error) {
    fprintf(stderr, "%s failed: %s\n", operation, sdk_error_text(error));
}

static int prepare_inputs(
    const execution_plan_request_t *request,
    unsigned char **inputs,
    char **failure_message
) {
    for (size_t index = 0u; index < request->resident.input_count; index++) {
        const resident_input_file_t *input = &request->resident.inputs[index];
        if (input->slot_id >= request->resident.header.slot_count ||
            file_size_matches(input->path, input->raw_bytes) != 0 ||
            (inputs[index] = (unsigned char *)calloc(input->transfer_bytes, 1u)) == NULL ||
            read_exact(input->path, inputs[index], input->raw_bytes) != 0 ||
            buffer_is_finite(inputs[index], input->raw_bytes) != 0) {
            if (failure_message != NULL && *failure_message == NULL) *failure_message = strdup("initial input file or finite-value validation failed");
            return 1;
        }
    }
    return 0;
}

static int copy_package_to_dpu(
    struct dpu_set_t dpu,
    const execution_plan_request_t *request,
    unsigned char **inputs,
    execution_plan_metrics_t *metrics,
    dpu_error_t *error
) {
    const resident_control_t control = {
        request->resident.header.slot_count,
        request->resident.header.operation_count,
        request->resident.header.pool_bytes,
        0u,
    };
    *error = dpu_copy_to(dpu, "RESIDENT_SLOT_DESCRIPTORS", 0u, request->resident.slots,
        request->resident.header.slot_bytes);
    if (*error == DPU_OK) metrics->descriptor_h2d_bytes += request->resident.header.slot_bytes;
    if (*error == DPU_OK) *error = dpu_copy_to(dpu, "RESIDENT_OPERATIONS", 0u, request->resident.operations,
        request->resident.header.operation_bytes);
    if (*error == DPU_OK) metrics->descriptor_h2d_bytes += request->resident.header.operation_bytes;
    if (*error == DPU_OK) *error = dpu_copy_to(dpu, "RESIDENT_CONTROL", 0u, &control, sizeof(control));
    if (*error == DPU_OK) metrics->descriptor_h2d_bytes += sizeof(control);
    if (*error != DPU_OK) return 1;
    for (size_t index = 0u; index < request->resident.input_count; index++) {
        const resident_input_file_t *input = &request->resident.inputs[index];
        *error = dpu_copy_to(dpu, "RESIDENT_SLOT_POOL", request->resident.slots[input->slot_id].offset_bytes,
            inputs[index], input->transfer_bytes);
        if (*error != DPU_OK) return 1;
        metrics->operand_h2d_bytes += input->transfer_bytes;
    }
    return 0;
}

#if RESIDENT_OPERATION_ABI_VERSION == RESIDENT_OPERATION_ABI_V2
static const execution_plan_v2_work_unit_t *distributed_work_unit_for_dpu(
    const execution_plan_request_t *request,
    uint32_t dpu_id
) {
    for (uint32_t index = 0u; index < request->distributed_v2.work_unit_count; index++) {
        if (request->distributed_v2.work_units[index].dpu_id == dpu_id) {
            return &request->distributed_v2.work_units[index];
        }
    }
    return NULL;
}

static int copy_distributed_package_to_dpu(
    struct dpu_set_t dpu,
    uint32_t dpu_id,
    const execution_plan_request_t *request,
    unsigned char **inputs,
    execution_plan_metrics_t *metrics,
    dpu_error_t *error
) {
    const execution_plan_v2_work_unit_t *unit = distributed_work_unit_for_dpu(request, dpu_id);
    resident_operation_t operation;
    const resident_control_t control = {
        request->resident.header.slot_count,
        request->resident.header.operation_count,
        request->resident.header.pool_bytes,
        0u,
    };
    if (unit == NULL || unit->package_operation_index >= request->resident.header.operation_count) {
        *error = DPU_ERR_INVALID_PROFILE;
        return 1;
    }
    operation = request->resident.operations[unit->package_operation_index];
    operation.output_elements = unit->output_elements;
    operation.args.dpu_slice_offset = unit->output_offset;
    operation.args.dpu_slice_elements = unit->output_elements;
    operation.args.contracted_offset = unit->contracted_offset;
    operation.args.contracted_elements_slice = unit->contracted_elements;
    *error = dpu_copy_to(dpu, "RESIDENT_SLOT_DESCRIPTORS", 0u, request->resident.slots,
        request->resident.header.slot_bytes);
    if (*error == DPU_OK) metrics->descriptor_h2d_bytes += request->resident.header.slot_bytes;
    if (*error == DPU_OK) *error = dpu_copy_to(dpu, "RESIDENT_OPERATIONS", 0u, &operation,
        sizeof(operation));
    if (*error == DPU_OK) metrics->descriptor_h2d_bytes += sizeof(operation);
    if (*error == DPU_OK) *error = dpu_copy_to(dpu, "RESIDENT_CONTROL", 0u, &control, sizeof(control));
    if (*error == DPU_OK) metrics->descriptor_h2d_bytes += sizeof(control);
    if (*error != DPU_OK) return 1;
    for (size_t index = 0u; index < request->resident.input_count; index++) {
        const resident_input_file_t *input = &request->resident.inputs[index];
        *error = dpu_copy_to(dpu, "RESIDENT_SLOT_POOL", request->resident.slots[input->slot_id].offset_bytes,
            inputs[index], input->transfer_bytes);
        if (*error != DPU_OK) return 1;
        metrics->operand_h2d_bytes += input->transfer_bytes;
    }
    return 0;
}

static uint64_t checksum_f32_bytes(uint64_t checksum, const unsigned char *buffer, uint32_t elements) {
    const size_t bytes = (size_t)elements * sizeof(float);
    for (size_t index = 0u; index < bytes; index++) {
        checksum ^= buffer[index];
        checksum *= 1099511628211ULL;
    }
    return checksum;
}
#endif

static int reset_dpu(
    struct dpu_set_t dpu,
    const execution_plan_request_t *request,
    unsigned char **inputs,
    execution_plan_metrics_t *metrics,
    dpu_error_t *error
) {
    resident_completion_t completion = {0};
    uint64_t active_operation = 0u;
    *error = dpu_copy_to(dpu, "RESIDENT_COMPLETION", 0u, &completion, sizeof(completion));
    if (*error == DPU_OK) *error = dpu_copy_to(dpu, "RESIDENT_ACTIVE_OPERATION", 0u, &active_operation, sizeof(active_operation));
    if (*error == DPU_OK) metrics->reset_h2d_bytes += sizeof(completion) + sizeof(active_operation);
    if (*error != DPU_OK) return 1;
    for (size_t index = 0u; index < request->resident.input_count; index++) {
        const resident_input_file_t *input = &request->resident.inputs[index];
        *error = dpu_copy_to(dpu, "RESIDENT_SLOT_POOL", request->resident.slots[input->slot_id].offset_bytes,
            inputs[index], input->transfer_bytes);
        if (*error != DPU_OK) return 1;
        metrics->reset_h2d_bytes += input->transfer_bytes;
    }
    for (uint32_t slot = 0u; slot < request->resident.header.slot_count; slot++) {
        const resident_slot_descriptor_t *descriptor = &request->resident.slots[slot];
        if (request->producer_by_slot[slot] < 0) continue;
        const uint32_t bytes = align8(descriptor->capacity_elements * (uint32_t)sizeof(float));
        unsigned char *zero = (unsigned char *)calloc(bytes, 1u);
        if (zero == NULL) {
            *error = DPU_ERR_INTERNAL;
            return 1;
        }
        *error = dpu_copy_to(dpu, "RESIDENT_SLOT_POOL", descriptor->offset_bytes, zero, bytes);
        free(zero);
        if (*error != DPU_OK) return 1;
        metrics->reset_h2d_bytes += bytes;
    }
    return 0;
}

static int reset_all_dpus(
    struct dpu_set_t set,
    const execution_plan_request_t *request,
    unsigned char **inputs,
    execution_plan_metrics_t *metrics,
    dpu_error_t *error
) {
    struct dpu_set_t dpu;
    uint32_t index;
    DPU_FOREACH(set, dpu, index) {
        (void)index;
        if (reset_dpu(dpu, request, inputs, metrics, error) != 0) return 1;
    }
    return 0;
}

static int transfer_cross_dpu_edges(
    struct dpu_set_t set,
    const execution_plan_request_t *request,
    uint32_t wave,
    execution_plan_metrics_t *metrics,
    execution_plan_timing_t *timing,
    dpu_error_t *error
) {
    const double started = now_s();
    const uint32_t operation_count = request->resident.header.operation_count;
    for (uint32_t consumer_package = 0u; consumer_package < operation_count; consumer_package++) {
        const execution_plan_schedule_record_t *consumer =
            &request->schedule.records[request->record_for_package[consumer_package]];
        if (consumer->wave_index != wave) continue;
        for (uint32_t dependency_id = 0u; dependency_id < operation_count; dependency_id++) {
            if ((consumer->dependency_mask & (1u << dependency_id)) == 0u) continue;
            const uint32_t producer_package = request->package_for_operation[dependency_id];
            const execution_plan_schedule_record_t *producer =
                &request->schedule.records[request->record_for_package[producer_package]];
            if (producer->dpu_id == consumer->dpu_id) continue;
            const uint32_t slot_id = request->resident.operations[producer_package].slot_out_real;
            const resident_slot_descriptor_t *slot = &request->resident.slots[slot_id];
            const uint32_t bytes = align8(slot->element_count * (uint32_t)sizeof(float));
            unsigned char *handoff = (unsigned char *)calloc(bytes, 1u);
            struct dpu_set_t producer_dpu;
            struct dpu_set_t consumer_dpu;
            if (handoff == NULL || dpu_handle_at(set, producer->dpu_id, &producer_dpu) != 0 ||
                dpu_handle_at(set, consumer->dpu_id, &consumer_dpu) != 0) {
                free(handoff);
                *error = DPU_ERR_INVALID_PROFILE;
                return 1;
            }
            *error = dpu_copy_from(producer_dpu, "RESIDENT_SLOT_POOL", slot->offset_bytes, handoff, bytes);
            if (*error == DPU_OK) *error = dpu_copy_to(consumer_dpu, "RESIDENT_SLOT_POOL", slot->offset_bytes, handoff, bytes);
            free(handoff);
            if (*error != DPU_OK) return 1;
            metrics->cross_d2h_bytes += bytes;
            metrics->cross_h2d_bytes += bytes;
            metrics->cross_dpu_edge_count++;
        }
    }
    timing->cross_dpu_transfer_time_s += now_s() - started;
    return 0;
}

static int execute_wave(
    struct dpu_set_t set,
    const execution_plan_request_t *request,
    uint32_t wave,
    execution_plan_metrics_t *metrics,
    execution_plan_timing_t *timing,
    dpu_error_t *error,
    char **failure_message
) {
    const uint32_t operation_count = request->resident.header.operation_count;
    uint32_t active_package[EXECUTION_PLAN_MAX_DPUS];
    int active[EXECUTION_PLAN_MAX_DPUS] = {0};
    struct dpu_set_t handles[EXECUTION_PLAN_MAX_DPUS];
    const double started = now_s();
    memset(active_package, 0, sizeof(active_package));
    if (transfer_cross_dpu_edges(set, request, wave, metrics, timing, error) != 0) {
        if (failure_message != NULL && *failure_message == NULL) *failure_message = strdup("cross-DPU operand handoff failed");
        return 1;
    }
    for (uint32_t package_index = 0u; package_index < operation_count; package_index++) {
        const execution_plan_schedule_record_t *record =
            &request->schedule.records[request->record_for_package[package_index]];
        if (record->wave_index != wave) continue;
        if (active[record->dpu_id]) {
            if (failure_message != NULL && *failure_message == NULL) *failure_message = strdup("validated schedule assigned two operations to one DPU in a wave");
            *error = DPU_ERR_INVALID_PROFILE;
            return 1;
        }
        active[record->dpu_id] = 1;
        active_package[record->dpu_id] = package_index;
    }
    for (uint32_t dpu_id = 0u; dpu_id < request->schedule.header.dpu_count; dpu_id++) {
        if (!active[dpu_id] || dpu_handle_at(set, dpu_id, &handles[dpu_id]) != 0) {
            if (active[dpu_id]) {
                *error = DPU_ERR_INVALID_PROFILE;
                return 1;
            }
            continue;
        }
        const uint64_t active_operation = active_package[dpu_id];
        *error = dpu_copy_to(handles[dpu_id], "RESIDENT_ACTIVE_OPERATION", 0u,
            &active_operation, sizeof(active_operation));
        if (*error != DPU_OK) return 1;
        metrics->active_operation_h2d_bytes += sizeof(active_operation);
    }
    for (uint32_t dpu_id = 0u; dpu_id < request->schedule.header.dpu_count; dpu_id++) {
        if (!active[dpu_id]) continue;
        *error = dpu_launch(handles[dpu_id], DPU_ASYNCHRONOUS);
        if (*error != DPU_OK) return 1;
        metrics->launch_count++;
    }
    for (uint32_t dpu_id = 0u; dpu_id < request->schedule.header.dpu_count; dpu_id++) {
        if (!active[dpu_id]) continue;
        *error = dpu_sync(handles[dpu_id]);
        if (*error != DPU_OK) return 1;
        metrics->synchronize_count++;
        resident_completion_t completion = {0};
        *error = dpu_copy_from(handles[dpu_id], "RESIDENT_COMPLETION", 0u, &completion, sizeof(completion));
        if (*error != DPU_OK) return 1;
        metrics->completion_reads++;
        metrics->completion_d2h_bytes += sizeof(completion);
        const uint32_t package_index = active_package[dpu_id];
        const resident_operation_t *operation = &request->resident.operations[package_index];
        if (completion.magic != RESIDENT_COMPLETION_MAGIC ||
            completion.version != RESIDENT_COMPLETION_VERSION ||
            completion.active_operation_index != package_index ||
            completion.completion_status != RESIDENT_COMPLETION_COMPLETED ||
            completion.completed_operation_count != package_index + 1u ||
            completion.output_elements_processed != operation->output_elements) {
            if (failure_message != NULL && *failure_message == NULL) *failure_message = strdup("DPU completion record did not match the dispatched operation");
            *error = DPU_ERR_INVALID_PROFILE;
            return 1;
        }
        metrics->completed_per_dpu[dpu_id]++;
        metrics->native_kernel_executed = 1;
    }
    timing->launch_sync_time_s += now_s() - started;
    return 0;
}

#if RESIDENT_OPERATION_ABI_VERSION == RESIDENT_OPERATION_ABI_V2
static int validate_distributed_completion(
    const resident_completion_t *completion,
    const execution_plan_v2_work_unit_t *unit,
    uint32_t operation_index,
    char **failure_message,
    dpu_error_t *error
) {
    if (completion->magic != RESIDENT_COMPLETION_MAGIC ||
        completion->version != RESIDENT_COMPLETION_VERSION ||
        completion->active_operation_index != operation_index ||
        completion->completion_status != RESIDENT_COMPLETION_COMPLETED ||
        completion->completed_operation_count != operation_index + 1u ||
        completion->output_elements_processed != unit->output_elements) {
        if (failure_message != NULL && *failure_message == NULL) {
            *failure_message = strdup("distributed v2 completion did not match the assigned output slice");
        }
        *error = DPU_ERR_INVALID_PROFILE;
        return 1;
    }
#if RESIDENT_COMPLETION_VERSION >= 2
    {
        uint64_t processed = 0u;
        for (uint32_t tasklet = 0u; tasklet < NR_TASKLETS; tasklet++) {
            processed += completion->tasklet_processed_elements[tasklet];
        }
        if (completion->active_tasklet_count > NR_TASKLETS ||
            completion->tasklet_min_processed_elements > completion->tasklet_max_processed_elements ||
            completion->tasklet_max_processed_elements > unit->output_elements ||
            processed != unit->output_elements) {
            if (failure_message != NULL && *failure_message == NULL) {
                *failure_message = strdup("distributed v2 tasklet completion accounting did not cover the assigned slice");
            }
            *error = DPU_ERR_INVALID_PROFILE;
            return 1;
        }
    }
#endif
    return 0;
}

static int copy_distributed_output_slice(
    struct dpu_set_t dpu,
    const resident_slot_descriptor_t *slot,
    const resident_final_file_t *output,
    const execution_plan_v2_work_unit_t *unit,
    unsigned char *final_buffer,
    uint64_t expected_checksum,
    execution_plan_metrics_t *metrics,
    dpu_error_t *error,
    char **failure_message
) {
    const uint64_t raw_start = (uint64_t)unit->output_offset * sizeof(float);
    const uint64_t raw_bytes = (uint64_t)unit->output_elements * sizeof(float);
    const uint64_t raw_end = raw_start + raw_bytes;
    const uint64_t aligned_start = raw_start & ~7ULL;
    const uint64_t aligned_end = (raw_end + 7ULL) & ~7ULL;
    unsigned char *scratch;
    if (raw_end < raw_start || aligned_end < raw_end || aligned_end > output->transfer_bytes ||
        aligned_start > UINT32_MAX || aligned_end - aligned_start > SIZE_MAX ||
        (uint64_t)slot->offset_bytes + aligned_end > UINT32_MAX) {
        if (failure_message != NULL && *failure_message == NULL) {
            *failure_message = strdup("distributed v2 output slice transfer bounds are invalid");
        }
        *error = DPU_ERR_INVALID_PROFILE;
        return 1;
    }
    scratch = (unsigned char *)calloc((size_t)(aligned_end - aligned_start), 1u);
    if (scratch == NULL) {
        if (failure_message != NULL && *failure_message == NULL) *failure_message = strdup("distributed v2 output scratch allocation failed");
        *error = DPU_ERR_INTERNAL;
        return 1;
    }
    *error = dpu_copy_from(dpu, "RESIDENT_SLOT_POOL", slot->offset_bytes + (uint32_t)aligned_start,
        scratch, (size_t)(aligned_end - aligned_start));
    if (*error == DPU_OK) {
        memcpy(final_buffer + raw_start, scratch + (raw_start - aligned_start), (size_t)raw_bytes);
        metrics->final_d2h_bytes += aligned_end - aligned_start;
        if (buffer_is_finite(final_buffer + raw_start, (size_t)raw_bytes) != 0 ||
            checksum_f32_bytes(14695981039346656037ULL, final_buffer + raw_start, unit->output_elements) != expected_checksum) {
            if (failure_message != NULL && *failure_message == NULL) {
                *failure_message = strdup("distributed v2 output slice checksum or finite-value validation failed");
            }
            *error = DPU_ERR_INVALID_PROFILE;
        }
    }
    free(scratch);
    return *error == DPU_OK ? 0 : 1;
}

static int reduce_distributed_partial(
    struct dpu_set_t dpu,
    const resident_slot_descriptor_t *slot,
    const resident_final_file_t *output,
    const execution_plan_v2_work_unit_t *unit,
    unsigned char *accumulator,
    uint64_t expected_checksum,
    int initialize_accumulator,
    execution_plan_metrics_t *metrics,
    dpu_error_t *error,
    char **failure_message
) {
    unsigned char *partial = (unsigned char *)calloc(output->transfer_bytes, 1u);
    if (partial == NULL) {
        if (failure_message != NULL && *failure_message == NULL) {
            *failure_message = strdup("distributed v2 partial output allocation failed");
        }
        *error = DPU_ERR_INTERNAL;
        return 1;
    }
    *error = dpu_copy_from(dpu, "RESIDENT_SLOT_POOL", slot->offset_bytes,
        partial, output->transfer_bytes);
    if (*error == DPU_OK) {
        metrics->reduction_d2h_bytes += output->transfer_bytes;
        metrics->reduction_partial_reads++;
        if (buffer_is_finite(partial, output->raw_bytes) != 0 ||
            checksum_f32_bytes(14695981039346656037ULL, partial, unit->output_elements) != expected_checksum) {
            if (failure_message != NULL && *failure_message == NULL) {
                *failure_message = strdup("distributed v2 partial checksum or finite-value validation failed");
            }
            *error = DPU_ERR_INVALID_PROFILE;
        } else {
            if (initialize_accumulator) {
                memcpy(accumulator, partial, output->raw_bytes);
            } else {
                float *sum = (float *)accumulator;
                const float *values = (const float *)partial;
                for (uint32_t index = 0u; index < unit->output_elements; index++) {
                    sum[index] += values[index];
                    if (!isfinite(sum[index])) {
                        if (failure_message != NULL && *failure_message == NULL) {
                            *failure_message = strdup("distributed v2 host reduction produced a non-finite value");
                        }
                        *error = DPU_ERR_INVALID_PROFILE;
                        break;
                    }
                    metrics->reduction_element_additions++;
                }
            }
        }
    }
    free(partial);
    return *error == DPU_OK ? 0 : 1;
}

static int execute_distributed_repetition(
    struct dpu_set_t set,
    const execution_plan_request_t *request,
    unsigned char *final_buffer,
    execution_plan_metrics_t *metrics,
    execution_plan_timing_t *timing,
    dpu_error_t *error,
    char **failure_message
) {
    const execution_plan_v2_header_t *header = &request->distributed_v2.header;
    const resident_final_file_t *output = &request->resident.final_outputs[0];
    const resident_slot_descriptor_t *output_slot = &request->resident.slots[header->output_slot];
    struct dpu_set_t handles[EXECUTION_PLAN_V2_MAX_DPUS];
    const double launch_started = now_s();

    if (header->partition_mode == EXECUTION_PLAN_V2_PARTITION_CONTRACTED) {
        memset(final_buffer, 0, output->transfer_bytes);
        metrics->reduction_accumulator_resets++;
    }
    if (execution_plan_interrupted) {
        if (failure_message != NULL && *failure_message == NULL) *failure_message = strdup("bounded execution timeout or signal interrupted the run");
        *error = DPU_ERR_TIMEOUT;
        return 1;
    }
    for (uint32_t dpu_id = 0u; dpu_id < header->dpu_count; dpu_id++) {
        const execution_plan_v2_work_unit_t *unit = distributed_work_unit_for_dpu(request, dpu_id);
        uint64_t active_operation = header->package_operation_index;
        if (unit == NULL || dpu_handle_at(set, dpu_id, &handles[dpu_id]) != 0) {
            if (failure_message != NULL && *failure_message == NULL) *failure_message = strdup("distributed v2 DPU handle or work-unit lookup failed");
            *error = DPU_ERR_INVALID_PROFILE;
            return 1;
        }
        *error = dpu_copy_to(handles[dpu_id], "RESIDENT_ACTIVE_OPERATION", 0u,
            &active_operation, sizeof(active_operation));
        if (*error != DPU_OK) return 1;
        metrics->active_operation_h2d_bytes += sizeof(active_operation);
    }
    for (uint32_t dpu_id = 0u; dpu_id < header->dpu_count; dpu_id++) {
        *error = dpu_launch(handles[dpu_id], DPU_ASYNCHRONOUS);
        if (*error != DPU_OK) return 1;
        metrics->launch_count++;
    }
    for (uint32_t dpu_id = 0u; dpu_id < header->dpu_count; dpu_id++) {
        const execution_plan_v2_work_unit_t *unit = distributed_work_unit_for_dpu(request, dpu_id);
        resident_completion_t completion = {0};
        if (unit == NULL) {
            if (failure_message != NULL && *failure_message == NULL) *failure_message = strdup("distributed v2 DPU completion work-unit lookup failed");
            *error = DPU_ERR_INVALID_PROFILE;
            return 1;
        }
        *error = dpu_sync(handles[dpu_id]);
        if (*error != DPU_OK) return 1;
        metrics->synchronize_count++;
        *error = dpu_copy_from(handles[dpu_id], "RESIDENT_COMPLETION", 0u, &completion, sizeof(completion));
        if (*error != DPU_OK) return 1;
        metrics->completion_reads++;
        metrics->completion_d2h_bytes += sizeof(completion);
        if (validate_distributed_completion(&completion, unit, header->package_operation_index,
                failure_message, error) != 0) return 1;
#if RESIDENT_COMPLETION_VERSION >= 2
        metrics->distributed_dpu_cycles[dpu_id] += completion.dpu_run_time_cycles;
#endif
        metrics->distributed_dpu_work_elements[dpu_id] += completion.output_elements_processed;
        metrics->distributed_dpu_checksums[dpu_id] = completion.output_checksum_fnv1a64;
        metrics->distributed_dpu_completions[dpu_id]++;
        metrics->completed_per_dpu[dpu_id]++;
        metrics->distributed_dpu_output_offsets[dpu_id] = unit->output_offset;
        metrics->distributed_dpu_output_elements[dpu_id] = unit->output_elements;
        metrics->native_kernel_executed = 1;
    }
    timing->launch_sync_time_s += now_s() - launch_started;
    {
        const double output_started = now_s();
        if (header->partition_mode == EXECUTION_PLAN_V2_PARTITION_CONTRACTED) {
            for (uint32_t dpu_id = 0u; dpu_id < header->dpu_count; dpu_id++) {
                const execution_plan_v2_work_unit_t *unit = distributed_work_unit_for_dpu(request, dpu_id);
                if (unit == NULL || reduce_distributed_partial(handles[dpu_id], output_slot, output, unit,
                        final_buffer, metrics->distributed_dpu_checksums[dpu_id], dpu_id == 0u,
                        metrics, error, failure_message) != 0) return 1;
                metrics->reduction_participant_count++;
            }
        } else {
            for (uint32_t work_index = 0u; work_index < request->distributed_v2.work_unit_count; work_index++) {
                const execution_plan_v2_work_unit_t *unit = &request->distributed_v2.work_units[work_index];
                if (copy_distributed_output_slice(handles[unit->dpu_id], output_slot, output, unit, final_buffer,
                        metrics->distributed_dpu_checksums[unit->dpu_id], metrics, error, failure_message) != 0) return 1;
            }
        }
        if (buffer_is_finite(final_buffer, output->raw_bytes) != 0) {
            if (failure_message != NULL && *failure_message == NULL) *failure_message = strdup("distributed v2 final output finite-value validation failed");
            *error = DPU_ERR_INVALID_PROFILE;
            return 1;
        }
        if (header->partition_mode == EXECUTION_PLAN_V2_PARTITION_CONTRACTED) {
            timing->reduction_time_s += now_s() - output_started;
        } else {
            timing->final_d2h_time_s += now_s() - output_started;
        }
        {
            const double write_started = now_s();
            if (write_exact(output->path, final_buffer, output->raw_bytes) != 0) {
                if (failure_message != NULL && *failure_message == NULL) *failure_message = strdup("final output file could not be written");
                *error = DPU_ERR_INTERNAL;
                return 1;
            }
            timing->output_write_time_s += now_s() - write_started;
        }
    }
    return 0;
}
#endif

static int execute_repetition(
    struct dpu_set_t set,
    const execution_plan_request_t *request,
    unsigned char *final_buffer,
    execution_plan_metrics_t *metrics,
    execution_plan_timing_t *timing,
    dpu_error_t *error,
    char **failure_message
) {
    if (request->distributed_v2_mode) {
#if RESIDENT_OPERATION_ABI_VERSION == RESIDENT_OPERATION_ABI_V2
        return execute_distributed_repetition(set, request, final_buffer, metrics, timing, error, failure_message);
#else
        if (failure_message != NULL && *failure_message == NULL) {
            *failure_message = strdup("distributed v2 execution requires the resident package ABI v2 host");
        }
        *error = DPU_ERR_INVALID_PROFILE;
        return 1;
#endif
    }
    const uint32_t final_slot = request->resident.final_outputs[0].slot_id;
    const int32_t final_package = request->producer_by_slot[final_slot];
    const execution_plan_schedule_record_t *final_record =
        &request->schedule.records[request->record_for_package[(uint32_t)final_package]];
    for (uint32_t wave = 0u; wave < request->schedule.header.wave_count; wave++) {
        if (execution_plan_interrupted) {
            if (failure_message != NULL && *failure_message == NULL) *failure_message = strdup("bounded execution timeout or signal interrupted the run");
            *error = DPU_ERR_TIMEOUT;
            return 1;
        }
        if (execute_wave(set, request, wave, metrics, timing, error, failure_message) != 0) return 1;
    }
    {
        struct dpu_set_t final_dpu;
        const resident_final_file_t *output = &request->resident.final_outputs[0];
        const double started = now_s();
        if (dpu_handle_at(set, final_record->dpu_id, &final_dpu) != 0 ||
            dpu_copy_from(final_dpu, "RESIDENT_SLOT_POOL", request->resident.slots[final_slot].offset_bytes,
                final_buffer, output->transfer_bytes) != DPU_OK ||
            buffer_is_finite(final_buffer, output->raw_bytes) != 0) {
            if (failure_message != NULL && *failure_message == NULL) *failure_message = strdup("final output transfer or finite-value validation failed");
            *error = DPU_ERR_INVALID_PROFILE;
            return 1;
        }
        timing->final_d2h_time_s += now_s() - started;
        metrics->final_d2h_bytes += output->transfer_bytes;
        const double output_started = now_s();
        if (write_exact(output->path, final_buffer, output->raw_bytes) != 0) {
            if (failure_message != NULL && *failure_message == NULL) *failure_message = strdup("final output file could not be written");
            *error = DPU_ERR_INTERNAL;
            return 1;
        }
        timing->output_write_time_s += now_s() - output_started;
    }
    return 0;
}

static void write_null_or_string(FILE *file, const char *value) {
    if (value == NULL) fputs("null", file);
    else json_string(file, value);
}

static int write_response(
    const char *path,
    const execution_plan_request_t *request,
    const char *status,
    const char *failure_stage,
    const char *error_message,
    const execution_plan_provider_t *provider,
    const execution_plan_metrics_t *metrics,
    const execution_plan_timing_t *timing,
    int allocation_succeeded,
    int allocation_was_confirmed,
    int release_confirmed,
    const char *native_binary_sha256,
    const char *dpu_binary_sha256
) {
    FILE *file = path == NULL ? stdout : fopen(path, "wb");
    const int valid_request = request != NULL;
    const int distributed = valid_request && request->distributed_v2_mode;
    const int contracted_reduction = distributed &&
        request->distributed_v2.header.partition_mode == EXECUTION_PLAN_V2_PARTITION_CONTRACTED;
    const char *target_observed =
        status != NULL && strcmp(status, "completed") == 0 &&
        allocation_was_confirmed && metrics != NULL && metrics->native_kernel_executed
            ? "\"physical_hardware\""
            : provider != NULL && provider->allocation_attempted && allocation_was_confirmed
                ? "\"unknown\""
                : "\"not_allocated\"";
    const uint32_t operation_count = valid_request ? request->resident.header.operation_count : 0u;
    const uint32_t dpu_count = valid_request ? (distributed ? request->distributed_v2.header.dpu_count : request->schedule.header.dpu_count) : 0u;
    const uint64_t actual_h2d = metrics == NULL ? 0ull :
        metrics->descriptor_h2d_bytes + metrics->operand_h2d_bytes + metrics->reset_h2d_bytes +
        metrics->active_operation_h2d_bytes + metrics->cross_h2d_bytes;
    const uint64_t actual_d2h = metrics == NULL ? 0ull :
        metrics->completion_d2h_bytes + metrics->cross_d2h_bytes + metrics->final_d2h_bytes +
        metrics->reduction_d2h_bytes;
    if (file == NULL) return 1;
    fprintf(file, "{\"schema_version\":\"%s\",\"status\":",
        distributed ? EXECUTION_PLAN_V2_RESPONSE_SCHEMA : EXECUTION_PLAN_RESPONSE_SCHEMA);
    json_string(file, status == NULL ? "failed" : status);
    fprintf(file, ",\"failure_stage\":");
    write_null_or_string(file, failure_stage);
    fprintf(file, ",\"error\":");
    write_null_or_string(file, error_message);
    fprintf(file, ",\"target_requested\":\"hardware\",\"target_observed\":%s,\"backend_id\":\"upmem_sdk_hardware_execution_plan\",\"backend_family\":\"upmem_sdk\",\"execution_class\":\"resident_taskgraph\",\"kernel_strategy\":\"resident_generic_contract\",\"hardware_profile_version\":\"%s\",\"requested_dpu_count\":%u,\"allocated_dpu_count\":%u,\"tasklets_per_dpu\":%u,\"hardware_allocation_verified\":%s,\"native_kernel_executed\":%s,\"simulator_kernel_executed\":false,\"hardware_kernel_executed\":%s,\"cpu_fallback_used\":false,\"hardware_speedup_applicable\":false,\"timing_is_bringup_only\":true,\"kernel_time_s\":null,\"timing_scope\":\"host_observed_sdk_stage_boundaries\",\"validation_status\":",
        target_observed,
        distributed ? EXECUTION_PLAN_V2_PROFILE : EXECUTION_PLAN_PROFILE,
        dpu_count,
        provider != NULL && provider->allocation_used ? provider->observed_dpus : 0u,
        (unsigned)NR_TASKLETS,
        allocation_was_confirmed ? "true" : "false",
        metrics != NULL && metrics->native_kernel_executed ? "true" : "false",
        metrics != NULL && metrics->native_kernel_executed && release_confirmed ? "true" : "false");
    json_string(file, status != NULL && strcmp(status, "validated") == 0 ? "plan_valid" :
        (status != NULL && strcmp(status, "completed") == 0 ? "native_completion_verified" : "not_validated"));
    fprintf(file, ",\"exact_integer_match\":null,\"full_precision_accuracy_status\":\"python_reference_required\",\"hardware_functionality_evidence\":%s,\"package_file_sha256\":",
        status != NULL && strcmp(status, "completed") == 0 && release_confirmed ? "true" : "false");
    write_null_or_string(file, valid_request ? request->actual_package_file_sha256 : NULL);
    fprintf(file, ",\"package_file_fnv1a64_runtime\":");
    if (valid_request) fprintf(file, "\"%016llx\"", (unsigned long long)request->package_file_fnv1a64_runtime);
    else fputs("null", file);
    fprintf(file, ",\"schedule_sidecar_sha256\":");
    write_null_or_string(file, valid_request && !distributed ? request->schedule.file_sha256 : NULL);
    fprintf(file, ",\"schedule_file_fnv1a64_runtime\":");
    if (valid_request && !distributed) fprintf(file, "\"%016llx\"", (unsigned long long)request->schedule.schedule_file_fnv1a64_runtime);
    else fputs("null", file);
    fprintf(file, ",\"distributed_plan_v2_sha256\":");
    write_null_or_string(file, valid_request && distributed ? request->distributed_v2.file_sha256 : NULL);
    fprintf(file, ",\"distributed_plan_v2_file_fnv1a64_runtime\":");
    if (valid_request && distributed) fprintf(file, "\"%016llx\"", (unsigned long long)request->distributed_v2.sidecar_file_fnv1a64_runtime);
    else fputs("null", file);
    fprintf(file, ",\"native_binary_sha256\":");
    write_null_or_string(file, native_binary_sha256);
    fprintf(file, ",\"dpu_binary_sha256\":");
    write_null_or_string(file, dpu_binary_sha256);
    fprintf(file, ",\"source_identity\":null,\"execution_plan_hash\":null,\"requested_warmups\":%u,\"requested_repetitions\":%u,\"provider_identities\":{\"runtime\":\"%s\",\"kernel\":\"%s\",\"communication\":\"%s\"},\"allocation_succeeded\":%s,\"allocation_was_confirmed\":%s,\"allocation\":{\"attempted\":%s,\"confirmed\":%s,\"release_confirmed\":%s},\"timing\":{\"allocation_time_s\":%.9f,\"binary_load_time_s\":%.9f,\"descriptor_h2d_time_s\":%.9f,\"operand_h2d_time_s\":%.9f,\"cross_dpu_transfer_time_s\":%.9f,\"launch_sync_time_s\":%.9f,\"final_d2h_time_s\":%.9f",
        valid_request ? request->warmup_repetitions : 0u,
        valid_request ? request->measured_repetitions : 0u,
        EXECUTION_PLAN_PROVIDER, EXECUTION_PLAN_KERNEL_PROVIDER,
        distributed ? (contracted_reduction ? "host_mediated_sum_v1" : "none") : EXECUTION_PLAN_COMMUNICATION_PROVIDER,
        allocation_succeeded ? "true" : "false",
        allocation_was_confirmed ? "true" : "false",
        provider != NULL && provider->allocation_attempted ? "true" : "false",
        allocation_was_confirmed ? "true" : "false",
        release_confirmed ? "true" : "false",
        timing == NULL ? 0.0 : timing->allocation_time_s, timing == NULL ? 0.0 : timing->binary_load_time_s,
        timing == NULL ? 0.0 : timing->descriptor_h2d_time_s, timing == NULL ? 0.0 : timing->operand_h2d_time_s,
        timing == NULL ? 0.0 : timing->cross_dpu_transfer_time_s, timing == NULL ? 0.0 : timing->launch_sync_time_s,
        timing == NULL ? 0.0 : timing->final_d2h_time_s);
    if (distributed) fprintf(file, ",\"reduction_time_s\":%.9f", timing == NULL ? 0.0 : timing->reduction_time_s);
    fprintf(file, ",\"output_write_time_s\":%.9f,\"release_time_s\":%.9f},\"metrics\":{\"descriptor_h2d_bytes\":%llu,\"operand_h2d_bytes\":%llu,\"reset_h2d_bytes\":%llu,\"active_operation_h2d_bytes\":%llu,\"completion_d2h_bytes\":%llu,\"cross_d2h_bytes\":%llu,\"cross_h2d_bytes\":%llu,\"final_d2h_bytes\":%llu",
        timing == NULL ? 0.0 : timing->output_write_time_s,
        timing == NULL ? 0.0 : timing->release_time_s,
        metrics == NULL ? 0ull : (unsigned long long)metrics->descriptor_h2d_bytes,
        metrics == NULL ? 0ull : (unsigned long long)metrics->operand_h2d_bytes,
        metrics == NULL ? 0ull : (unsigned long long)metrics->reset_h2d_bytes,
        metrics == NULL ? 0ull : (unsigned long long)metrics->active_operation_h2d_bytes,
        metrics == NULL ? 0ull : (unsigned long long)metrics->completion_d2h_bytes,
        metrics == NULL ? 0ull : (unsigned long long)metrics->cross_d2h_bytes,
        metrics == NULL ? 0ull : (unsigned long long)metrics->cross_h2d_bytes,
        metrics == NULL ? 0ull : (unsigned long long)metrics->final_d2h_bytes);
    if (distributed) fprintf(file, ",\"reduction_d2h_bytes\":%llu,\"reduction_partial_reads\":%llu,\"reduction_element_additions\":%llu,\"reduction_accumulator_resets\":%llu,\"reduction_participant_count\":%u",
        metrics == NULL ? 0ull : (unsigned long long)metrics->reduction_d2h_bytes,
        metrics == NULL ? 0ull : (unsigned long long)metrics->reduction_partial_reads,
        metrics == NULL ? 0ull : (unsigned long long)metrics->reduction_element_additions,
        metrics == NULL ? 0ull : (unsigned long long)metrics->reduction_accumulator_resets,
        metrics == NULL ? 0u : metrics->reduction_participant_count);
    fprintf(file, ",\"actual_h2d_bytes\":%llu,\"actual_d2h_bytes\":%llu,\"actual_transfer_bytes\":%llu,\"launch_count\":%llu,\"synchronize_count\":%llu,\"completion_reads\":%llu,\"cross_dpu_edge_count\":%llu},\"operation_assignments\":[",
        (unsigned long long)actual_h2d,
        (unsigned long long)actual_d2h,
        (unsigned long long)(actual_h2d + actual_d2h),
        metrics == NULL ? 0ull : (unsigned long long)metrics->launch_count,
        metrics == NULL ? 0ull : (unsigned long long)metrics->synchronize_count,
        metrics == NULL ? 0ull : (unsigned long long)metrics->completion_reads,
        metrics == NULL ? 0ull : (unsigned long long)metrics->cross_dpu_edge_count);
    if (valid_request && !distributed) {
        for (uint32_t index = 0u; index < operation_count; index++) {
            const execution_plan_schedule_record_t *record = &request->schedule.records[request->record_for_package[index]];
            if (index != 0u) fputc(',', file);
            fprintf(file, "{\"package_operation_index\":%u,\"operation_id\":%u,\"component\":\"real\",\"task_id\":null,\"wave_index\":%u,\"dpu_id\":%u,\"dependency_bitmask\":%u,\"input_slot_ids\":[%u,%u],\"output_slot_id\":%u}",
                record->package_operation_index, record->operation_id, record->wave_index, record->dpu_id,
                record->dependency_mask, record->input_slot_a, record->input_slot_b, record->output_slot);
        }
    } else if (valid_request) {
        for (uint32_t index = 0u; index < request->distributed_v2.work_unit_count; index++) {
            const execution_plan_v2_work_unit_t *unit = &request->distributed_v2.work_units[index];
            if (index != 0u) fputc(',', file);
            fprintf(file, "{\"package_operation_index\":%u,\"operation_id\":%u,\"partition_mode\":\"%s\",\"dpu_id\":%u,\"output_offset\":%u,\"output_elements\":%u,\"contracted_offset\":%u,\"contracted_elements\":%u}",
                unit->package_operation_index, unit->operation_id, contracted_reduction ? "contracted" : "output",
                unit->dpu_id, unit->output_offset, unit->output_elements,
                unit->contracted_offset, unit->contracted_elements);
        }
    }
    fputs("],\"cross_dpu_transfers\":[", file);
    if (valid_request && !distributed) {
        int first = 1;
        for (uint32_t consumer_package = 0u; consumer_package < operation_count; consumer_package++) {
            const execution_plan_schedule_record_t *consumer = &request->schedule.records[request->record_for_package[consumer_package]];
            for (uint32_t dependency_id = 0u; dependency_id < operation_count; dependency_id++) {
                if ((consumer->dependency_mask & (1u << dependency_id)) == 0u) continue;
                const uint32_t producer_package = request->package_for_operation[dependency_id];
                const execution_plan_schedule_record_t *producer = &request->schedule.records[request->record_for_package[producer_package]];
                if (producer->dpu_id == consumer->dpu_id) continue;
                const uint32_t slot_id = request->resident.operations[producer_package].slot_out_real;
                const uint32_t bytes = align8(request->resident.slots[slot_id].element_count * sizeof(float));
                if (!first) fputc(',', file);
                first = 0;
                fprintf(file, "{\"producer_operation_id\":%u,\"consumer_operation_id\":%u,\"producer_dpu_id\":%u,\"consumer_dpu_id\":%u,\"slot_id\":%u,\"transfer_bytes\":%u,\"transport\":\"host_mediated_v1\"}",
                    producer->operation_id, consumer->operation_id, producer->dpu_id, consumer->dpu_id, slot_id, bytes);
            }
        }
    }
    fputs("]", file);
    if (distributed) fprintf(file, ",\"reduction\":{\"provider\":\"%s\",\"order\":\"ascending_dpu_id\",\"participant_count\":%u,\"partial_buffers_read\":%llu,\"d2h_bytes\":%llu,\"element_additions\":%llu,\"accumulator_reset_per_repetition\":%s}",
        contracted_reduction ? "host_mediated_sum_v1" : "none",
        metrics == NULL ? 0u : metrics->reduction_participant_count,
        metrics == NULL ? 0ull : (unsigned long long)metrics->reduction_partial_reads,
        metrics == NULL ? 0ull : (unsigned long long)metrics->reduction_d2h_bytes,
        metrics == NULL ? 0ull : (unsigned long long)metrics->reduction_element_additions,
        contracted_reduction ? "true" : "false");
    fputs(",\"completed_per_dpu\":[", file);
    if (metrics != NULL) {
        for (uint32_t index = 0u; index < dpu_count; index++) {
            if (index != 0u) fputc(',', file);
            fprintf(file, "%u", metrics->completed_per_dpu[index]);
        }
    }
    fputs("],\"distributed_work_units\":[", file);
    if (valid_request && distributed) {
        for (uint32_t index = 0u; index < request->distributed_v2.work_unit_count; index++) {
            const execution_plan_v2_work_unit_t *unit = &request->distributed_v2.work_units[index];
            if (index != 0u) fputc(',', file);
            fprintf(file, "{\"dpu_id\":%u,\"partition_mode\":\"%s\",\"output_offset\":%u,\"output_elements\":%u,\"contracted_offset\":%u,\"contracted_elements\":%u,\"runtime_cycles\":%llu,\"processed_elements\":%llu,\"output_checksum_fnv1a64\":\"%016llx\",\"completion_count\":%u}",
                unit->dpu_id, contracted_reduction ? "contracted" : "output", unit->output_offset,
                unit->output_elements, unit->contracted_offset, unit->contracted_elements,
                (unsigned long long)metrics->distributed_dpu_cycles[unit->dpu_id],
                (unsigned long long)metrics->distributed_dpu_work_elements[unit->dpu_id],
                (unsigned long long)metrics->distributed_dpu_checksums[unit->dpu_id],
                metrics->distributed_dpu_completions[unit->dpu_id]);
        }
    }
    fputs("]}\n", file);
    if (path != NULL) return fclose(file) != 0;
    return fflush(file) != 0;
}

static void usage(const char *program) {
    fprintf(stderr, "usage: %s (--validate-plan|--execute-plan) --resident-package manifest.json (--schedule plan.bin|--distributed-plan-v2 sidecar.bin) [--response result.json] [--warmups N] [--repetitions N] [--timeout-s N]\n", program);
}

int main(int argc, char **argv) {
    const char *resident_manifest = NULL;
    const char *schedule_path = NULL;
    const char *distributed_plan_path = NULL;
    const char *response_path = NULL;
    const char *failure_stage = NULL;
    const char *error_message = NULL;
    char *owned_error = NULL;
    char native_binary_sha256[65] = {0};
    char dpu_binary_sha256[65] = {0};
    char initialization_binary[PATH_MAX];
    uint32_t warmups = 1u;
    uint32_t repetitions = 1u;
    uint32_t timeout_s = 60u;
    int validate_only = 0;
    int execute = 0;
    execution_plan_request_t request;
    execution_plan_provider_t provider = {0};
    execution_plan_metrics_t metrics = {0};
    execution_plan_timing_t timing = {0};
    unsigned char **inputs = NULL;
    unsigned char *final_buffer = NULL;
    dpu_error_t error = DPU_OK;
    int allocation_succeeded = 0;
    int allocation_was_confirmed = 0;
    int release_confirmed = 0;
    int rc = 1;
    const double started = now_s();
    memset(&request, 0, sizeof(request));

    for (int index = 1; index < argc; index++) {
        if (strcmp(argv[index], "--validate-plan") == 0) validate_only = 1;
        else if (strcmp(argv[index], "--execute-plan") == 0) execute = 1;
        else if (strcmp(argv[index], "--resident-package") == 0 && index + 1 < argc) resident_manifest = argv[++index];
        else if (strcmp(argv[index], "--schedule") == 0 && index + 1 < argc) schedule_path = argv[++index];
        else if (strcmp(argv[index], "--distributed-plan-v2") == 0 && index + 1 < argc) distributed_plan_path = argv[++index];
        else if (strcmp(argv[index], "--response") == 0 && index + 1 < argc) response_path = argv[++index];
        else if (strcmp(argv[index], "--warmups") == 0 && index + 1 < argc) warmups = (uint32_t)strtoul(argv[++index], NULL, 10);
        else if (strcmp(argv[index], "--repetitions") == 0 && index + 1 < argc) repetitions = (uint32_t)strtoul(argv[++index], NULL, 10);
        else if (strcmp(argv[index], "--timeout-s") == 0 && index + 1 < argc) timeout_s = (uint32_t)strtoul(argv[++index], NULL, 10);
        else {
            usage(argv[0]);
            return 2;
        }
    }
    if ((validate_only == execute) || resident_manifest == NULL ||
        (schedule_path == NULL) == (distributed_plan_path == NULL) ||
        warmups > 1u || repetitions == 0u || repetitions > EXECUTION_PLAN_MAX_REPETITIONS || timeout_s == 0u) {
        usage(argv[0]);
        return 2;
    }
    if (execution_plan_sha256_file("/proc/self/exe", native_binary_sha256) != 0) native_binary_sha256[0] = '\0';
    if ((distributed_plan_path == NULL && execution_plan_request_load(resident_manifest, schedule_path, warmups, repetitions, &request, &owned_error) != 0) ||
        (distributed_plan_path != NULL && execution_plan_request_load_v2(resident_manifest, distributed_plan_path, warmups, repetitions, &request, &owned_error) != 0)) {
        failure_stage = owned_error != NULL && (strstr(owned_error, "schedule") != NULL || strstr(owned_error, "distributed") != NULL) ? "hardware_profile_violation" : "manifest_parse_failed";
        error_message = owned_error;
        goto write_response;
    }
    if (validate_only) {
        rc = 0;
        goto write_response;
    }
    if (getenv("UPMEM_ALLOW_PHYSICAL_HARDWARE") == NULL || strcmp(getenv("UPMEM_ALLOW_PHYSICAL_HARDWARE"), "1") != 0) {
        failure_stage = "hardware_opt_in_missing";
        error_message = "UPMEM_ALLOW_PHYSICAL_HARDWARE=1 is required";
        goto write_response;
    }
    if (getenv("DPU_BACKEND") != NULL) {
        failure_stage = "hardware_profile_violation";
        error_message = "DPU_BACKEND must be unset for the physical route";
        goto write_response;
    }
    if (resolve_sibling("dpu_simplepim_management_init", initialization_binary) != 0) {
        failure_stage = "sdk_discovery_failed";
        error_message = "SimplePIM initialization binary is missing beside the host executable";
        goto write_response;
    }
    if (execution_plan_sha256_file(request.resident.dpu_binary_path, dpu_binary_sha256) != 0) {
        failure_stage = "sdk_discovery_failed";
        error_message = "resident DPU binary hash could not be computed";
        goto write_response;
    }
    inputs = (unsigned char **)calloc(request.resident.input_count, sizeof(*inputs));
    final_buffer = (unsigned char *)calloc(request.resident.final_outputs[0].transfer_bytes, 1u);
    if ((request.resident.input_count != 0u && inputs == NULL) || final_buffer == NULL || prepare_inputs(&request, inputs, &owned_error) != 0) {
        failure_stage = "operand_transfer_failed";
        error_message = owned_error == NULL ? "input preparation failed" : owned_error;
        goto write_response;
    }
    signal(SIGTERM, execution_plan_signal_handler);
    signal(SIGINT, execution_plan_signal_handler);
    signal(SIGALRM, execution_plan_signal_handler);
    {
        const double stage = now_s();
        const uint32_t requested_dpus = request.distributed_v2_mode ? request.distributed_v2.header.dpu_count : request.schedule.header.dpu_count;
        error = execution_plan_provider_init(&provider, requested_dpus, "backend=hw", initialization_binary);
        timing.allocation_time_s = now_s() - stage;
        allocation_succeeded = provider.allocation_used && provider.management != NULL;
        allocation_was_confirmed = allocation_succeeded && error == DPU_OK &&
            provider.observed_dpus == requested_dpus;
        if (error != DPU_OK) {
            failure_stage = "hardware_allocation_failed";
            error_message = "SimplePIM management allocation or initialization failed";
            goto release_and_write;
        }
    }
    {
        const double stage = now_s();
        error = dpu_load(provider.set, request.resident.dpu_binary_path, NULL);
        timing.binary_load_time_s = now_s() - stage;
        if (error != DPU_OK) {
            report_sdk_error("resident dpu_load", error);
            failure_stage = "binary_load_failed";
            goto release_and_write;
        }
    }
    {
        const double stage = now_s();
        struct dpu_set_t dpu;
        uint32_t index;
        DPU_FOREACH(provider.set, dpu, index) {
            if (request.distributed_v2_mode) {
#if RESIDENT_OPERATION_ABI_VERSION == RESIDENT_OPERATION_ABI_V2
                if (copy_distributed_package_to_dpu(dpu, index, &request, inputs, &metrics, &error) != 0) break;
#else
                error = DPU_ERR_INVALID_PROFILE;
                break;
#endif
            } else if (copy_package_to_dpu(dpu, &request, inputs, &metrics, &error) != 0) break;
        }
        timing.descriptor_h2d_time_s = now_s() - stage;
        if (error != DPU_OK) {
            report_sdk_error("resident descriptor/input transfer", error);
            failure_stage = "argument_transfer_failed";
            goto release_and_write;
        }
    }
    alarm(timeout_s);
    for (uint32_t iteration = 0u; iteration < request.warmup_repetitions + request.measured_repetitions; iteration++) {
        const double stage = now_s();
        if (reset_all_dpus(provider.set, &request, inputs, &metrics, &error) != 0) {
            timing.operand_h2d_time_s += now_s() - stage;
            failure_stage = "operand_transfer_failed";
            goto release_and_write;
        }
        timing.operand_h2d_time_s += now_s() - stage;
        if (execute_repetition(provider.set, &request, final_buffer, &metrics, &timing, &error, &owned_error) != 0) {
            failure_stage = execution_plan_interrupted ? "kernel_timeout" : "kernel_launch_failed";
            error_message = owned_error == NULL ? "resident taskgraph execution failed" : owned_error;
            goto release_and_write;
        }
    }
    alarm(0u);
    rc = 0;
release_and_write:
    if (provider.allocation_attempted) {
        const double stage = now_s();
        error = execution_plan_provider_release(&provider);
        timing.release_time_s = now_s() - stage;
        release_confirmed = provider.release_succeeded && !provider.allocation_active;
        if (error != DPU_OK || !release_confirmed) {
            failure_stage = "hardware_release_failed";
            rc = 1;
        }
    }
write_response:
    if (write_response(response_path, request.resident_manifest_path == NULL ? NULL : &request,
        failure_stage == NULL ? (validate_only ? "validated" : "completed") : "failed", failure_stage,
        error_message, &provider, &metrics, &timing, allocation_succeeded, allocation_was_confirmed, release_confirmed,
        native_binary_sha256[0] == '\0' ? NULL : native_binary_sha256,
        dpu_binary_sha256[0] == '\0' ? NULL : dpu_binary_sha256) != 0) rc = 1;
    alarm(0u);
    for (size_t index = 0u; index < request.resident.input_count; index++) free(inputs == NULL ? NULL : inputs[index]);
    free(inputs);
    free(final_buffer);
    free(owned_error);
    execution_plan_request_free(&request);
    (void)started;
    return rc;
}
