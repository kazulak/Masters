#define _POSIX_C_SOURCE 200809L

/* Reuse the small, battle-tested SDK and file helpers from the legacy host.
 * v3 has its own main and never enters the v1/v2 dispatch paths. */
#define main execution_plan_legacy_main
#include "host.c"
#undef main

#include "distributed_plan_v3.h"
#include "execution_plan_v3_common.h"

#include <float.h>

typedef struct {
    double total_time_s;
    double launch_sync_time_s;
    double completion_read_and_validation_time_s;
    double assembly_time_s;
    double host_dequantization_time_s;
    uint64_t dpu_cycles[EXECUTION_PLAN_V3_MAX_DPUS];
    uint64_t dpu_work_elements[EXECUTION_PLAN_V3_MAX_DPUS];
    uint32_t dpu_completions[EXECUTION_PLAN_V3_MAX_DPUS];
    uint64_t reset_h2d_bytes;
    uint64_t completion_d2h_bytes;
    uint64_t output_d2h_bytes;
} v3_repeat_timing_t;

typedef struct {
    const char *path;
    const char *expected_sha256;
    double tolerance;
    char actual_sha256[65];
    double max_abs_error;
    int loaded;
    int hash_verified;
    int finite;
    int compared;
    int passed;
    int exact_match;
} v3_policy_reference_t;

typedef struct {
    const char *path;
    const char *expected_sha256;
    char actual_sha256[65];
    int loaded;
    int hash_verified;
    int compared;
    int passed;
    uint64_t mismatch_count;
} v3_integer_reference_t;

typedef struct {
    v3_repeat_timing_t *repeats;
    uint32_t repeat_count;
    uint64_t launch_count;
    uint64_t synchronize_count;
    uint64_t completion_reads;
    uint64_t kernel_launch_api_calls;
    uint64_t explicit_sync_api_calls;
    uint64_t descriptor_h2d_bytes;
    uint64_t operand_h2d_bytes;
    uint64_t reset_h2d_bytes;
    uint64_t completion_d2h_bytes;
    uint64_t final_d2h_bytes;
    uint64_t reduction_d2h_bytes;
    uint64_t reduction_element_additions;
    int launch_attempted;
    int native_kernel_executed;
} v3_metrics_t;

static void v3_json_string(FILE *file, const char *value) {
    json_string(file, value == NULL ? "" : value);
}

static int v3_policy_validation_passed(const v3_policy_reference_t *validation) {
    return validation != NULL && validation->loaded && validation->hash_verified &&
        validation->compared && validation->finite && isfinite(validation->max_abs_error) &&
        isfinite(validation->tolerance) && validation->max_abs_error <= validation->tolerance;
}

static int v3_digest_text(const char text[65], unsigned char digest[32]) {
    for (uint32_t index = 0u; index < 32u; index++) {
        unsigned int value;
        if (sscanf(text + index * 2u, "%02x", &value) != 1 || value > 0xffu) return 1;
        digest[index] = (unsigned char)value;
    }
    return 0;
}

static int v3_prepare_inputs(
    const execution_plan_request_t *request,
    unsigned char **inputs,
    char **failure_message
) {
    for (size_t index = 0u; index < request->resident.input_count; index++) {
        const resident_input_file_t *input = &request->resident.inputs[index];
        char actual_sha256[65] = {0};
        if (input->slot_id >= request->resident.header.slot_count ||
            file_size_matches(input->path, input->raw_bytes) != 0 ||
            (inputs[index] = (unsigned char *)calloc(input->transfer_bytes, 1u)) == NULL ||
            read_exact(input->path, inputs[index], input->raw_bytes) != 0 ||
            execution_plan_sha256_file(input->path, actual_sha256) != 0 ||
            input->logical_sha256 == NULL ||
            strcmp(input->logical_sha256, actual_sha256) != 0 ||
            (input->storage_kind == RESIDENT_STORAGE_FLOAT32 &&
                buffer_is_finite(inputs[index], input->raw_bytes) != 0)) {
            if (failure_message != NULL && *failure_message == NULL) {
                *failure_message = strdup(
                    "initial input file, hash, storage type, or finite-value validation failed"
                );
            }
            return 1;
        }
        if (input->storage_kind != RESIDENT_STORAGE_FLOAT32 &&
            input->storage_kind != RESIDENT_STORAGE_PACKED_INT8) {
            if (failure_message != NULL && *failure_message == NULL) {
                *failure_message = strdup("unsupported v3 initial input storage kind");
            }
            return 1;
        }
    }
    return 0;
}

static int v3_load_integer_reference(
    v3_integer_reference_t *validation,
    size_t bytes,
    unsigned char **reference,
    char **failure_message
) {
    if (validation == NULL || validation->path == NULL || validation->path[0] == '\0' ||
        validation->expected_sha256 == NULL || strlen(validation->expected_sha256) != 64u ||
        reference == NULL || file_size_matches(validation->path, bytes) != 0 ||
        execution_plan_sha256_file(validation->path, validation->actual_sha256) != 0 ||
        strcmp(validation->expected_sha256, validation->actual_sha256) != 0) {
        if (failure_message != NULL && *failure_message == NULL) {
            *failure_message = strdup("exact int32 reference file or hash is invalid");
        }
        return 1;
    }
    *reference = (unsigned char *)malloc(bytes);
    if (*reference == NULL || read_exact(validation->path, *reference, bytes) != 0) {
        free(*reference);
        *reference = NULL;
        if (failure_message != NULL && *failure_message == NULL) {
            *failure_message = strdup("exact int32 reference could not be loaded");
        }
        return 1;
    }
    validation->loaded = 1;
    validation->hash_verified = 1;
    return 0;
}

static int v3_validate_integer_reference(
    const int32_t *actual,
    const int32_t *reference,
    uint32_t elements,
    v3_integer_reference_t *validation,
    char **failure_message,
    dpu_error_t *error
) {
    if (actual == NULL || reference == NULL || validation == NULL) {
        *error = DPU_ERR_INVALID_PROFILE;
        return 1;
    }
    validation->compared = 1;
    validation->mismatch_count = 0u;
    for (uint32_t index = 0u; index < elements; index++) {
        if (actual[index] != reference[index]) validation->mismatch_count++;
    }
    validation->passed = validation->mismatch_count == 0u;
    if (!validation->passed) {
        if (failure_message != NULL && *failure_message == NULL) {
            *failure_message = strdup("assembled int32 output differs from the exact CPU reference");
        }
        *error = DPU_ERR_INVALID_PROFILE;
        return 1;
    }
    return 0;
}

static int v3_load_policy_reference(
    v3_policy_reference_t *validation, size_t bytes, unsigned char **reference,
    char **failure_message
) {
    if (validation == NULL || validation->path == NULL || validation->path[0] == '\0' ||
        reference == NULL || !isfinite(validation->tolerance) || validation->tolerance < 0.0) {
        if (failure_message != NULL && *failure_message == NULL) *failure_message = strdup(
            "a finite non-negative policy-reference tolerance and path are required");
        return 1;
    }
    if (execution_plan_sha256_file(validation->path, validation->actual_sha256) != 0 ||
        (validation->expected_sha256 != NULL && validation->expected_sha256[0] != '\0' &&
            strcmp(validation->expected_sha256, validation->actual_sha256) != 0)) {
        if (failure_message != NULL && *failure_message == NULL) *failure_message = strdup(
            "policy-reference hash could not be verified");
        return 1;
    }
    *reference = (unsigned char *)malloc(bytes);
    if (*reference == NULL || read_exact(validation->path, *reference, bytes) != 0 ||
        buffer_is_finite(*reference, bytes) != 0) {
        free(*reference); *reference = NULL;
        if (failure_message != NULL && *failure_message == NULL) *failure_message = strdup(
            "policy-reference file is missing, malformed, or non-finite");
        return 1;
    }
    validation->loaded = 1;
    validation->hash_verified = 1;
    validation->finite = 1;
    return 0;
}

static int v3_validate_policy_reference(
    const unsigned char *actual, const unsigned char *reference, uint32_t elements,
    v3_policy_reference_t *validation, char **failure_message, dpu_error_t *error
) {
    const float *actual_f32 = (const float *)actual;
    const float *reference_f32 = (const float *)reference;
    double max_abs_error = 0.0;
    int exact_match = 1;
    if (validation != NULL) validation->finite = 0;
    if (actual == NULL || reference == NULL || validation == NULL ||
        buffer_is_finite(actual, (size_t)elements * sizeof(float)) != 0) {
        *error = DPU_ERR_INVALID_PROFILE;
        return 1;
    }
    validation->compared = 1;
    for (uint32_t index = 0u; index < elements; index++) {
        const double difference = fabs((double)actual_f32[index] - (double)reference_f32[index]);
        if (!isfinite(difference)) { *error = DPU_ERR_INVALID_PROFILE; return 1; }
        if (difference > max_abs_error) max_abs_error = difference;
        if (actual_f32[index] != reference_f32[index]) exact_match = 0;
    }
    validation->max_abs_error = max_abs_error;
    validation->exact_match = exact_match;
    validation->finite = 1;
    validation->passed = validation->finite && max_abs_error <= validation->tolerance;
    if (!validation->passed) {
        if (failure_message != NULL && *failure_message == NULL) *failure_message = strdup(
            "assembled output failed the supplied CPU policy-reference tolerance");
        *error = DPU_ERR_INVALID_PROFILE;
        return 1;
    }
    return 0;
}

static int v3_copy_package_to_dpu(
    struct dpu_set_t dpu,
    uint32_t dpu_id,
    const execution_plan_request_t *request,
    const execution_plan_distributed_v3_t *plan,
    unsigned char **inputs,
    v3_metrics_t *metrics,
    dpu_error_t *error
) {
    const execution_plan_v3_work_unit_t *unit =
        execution_plan_distributed_v3_work_unit_for_dpu(plan, dpu_id);
    resident_operation_t operation;
    uint64_t active_operation;
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
    active_operation = unit->package_operation_index;
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
    if (*error == DPU_OK) *error = dpu_copy_to(dpu, "RESIDENT_CONTROL", 0u, &control,
        sizeof(control));
    if (*error == DPU_OK) metrics->descriptor_h2d_bytes += sizeof(control);
    if (*error == DPU_OK) *error = dpu_copy_to(dpu, "RESIDENT_ACTIVE_OPERATION", 0u,
        &active_operation, sizeof(active_operation));
    if (*error == DPU_OK) metrics->descriptor_h2d_bytes += sizeof(active_operation);
    if (*error != DPU_OK) return 1;
    for (size_t index = 0u; index < request->resident.input_count; index++) {
        const resident_input_file_t *input = &request->resident.inputs[index];
        *error = dpu_copy_to(dpu, "RESIDENT_SLOT_POOL",
            request->resident.slots[input->slot_id].offset_bytes, inputs[index], input->transfer_bytes);
        if (*error != DPU_OK) return 1;
        metrics->operand_h2d_bytes += input->transfer_bytes;
    }
    return 0;
}

static int v3_validate_completion(
    const resident_completion_t *completion,
    const execution_plan_v3_work_unit_t *unit,
    uint32_t operation_index,
    char **failure_message,
    dpu_error_t *error
) {
    uint64_t processed = 0u;
    if (completion->magic != RESIDENT_COMPLETION_MAGIC ||
        completion->version != RESIDENT_COMPLETION_VERSION ||
        completion->active_operation_index != operation_index ||
        completion->completion_status != RESIDENT_COMPLETION_COMPLETED ||
        completion->completed_operation_count != operation_index + 1u ||
        completion->output_elements_processed != unit->output_elements) {
        if (failure_message != NULL && *failure_message == NULL) *failure_message = strdup(
            "distributed v3 completion did not match the assigned work unit");
        *error = DPU_ERR_INVALID_PROFILE;
        return 1;
    }
    for (uint32_t tasklet = 0u; tasklet < NR_TASKLETS; tasklet++) {
        processed += completion->tasklet_processed_elements[tasklet];
    }
    if (completion->active_tasklet_count > NR_TASKLETS ||
        completion->tasklet_min_processed_elements > completion->tasklet_max_processed_elements ||
        completion->tasklet_max_processed_elements > unit->output_elements ||
        processed != unit->output_elements) {
        if (failure_message != NULL && *failure_message == NULL) *failure_message = strdup(
            "distributed v3 tasklet completion accounting did not cover the assigned work unit");
        *error = DPU_ERR_INVALID_PROFILE;
        return 1;
    }
    return 0;
}

static int v3_execute_repetition(
    struct dpu_set_t set,
    const execution_plan_request_t *request,
    const execution_plan_distributed_v3_t *plan,
    unsigned char *final_buffer,
    int32_t *raw_i32_buffer,
    v3_metrics_t *metrics,
    uint32_t repeat_index,
    const unsigned char *policy_reference,
    v3_policy_reference_t *policy_validation,
    const int32_t *integer_reference,
    v3_integer_reference_t *integer_validation,
    dpu_error_t *error,
    char **failure_message
) {
    struct dpu_set_t handles[EXECUTION_PLAN_V3_MAX_DPUS];
    resident_completion_t completions[EXECUTION_PLAN_V3_MAX_DPUS];
    const resident_final_file_t *output = &request->resident.final_outputs[0];
    const resident_slot_descriptor_t *output_slot = &request->resident.slots[plan->header.output_slot];
    const int packed_int8 =
        plan->header.numeric_mode == EXECUTION_PLAN_V3_NUMERIC_HOST_PACKED_INT8;
    const double started = now_s();
    double completion_started;
    memset(completions, 0, sizeof(completions));
    if (plan->header.partition_mode == EXECUTION_PLAN_V3_PARTITION_CONTRACTED) {
        memset(packed_int8 ? (unsigned char *)raw_i32_buffer : final_buffer, 0,
            output->transfer_bytes);
    }
    for (uint32_t dpu_id = 0u; dpu_id < plan->header.dpu_count; dpu_id++) {
        const execution_plan_v3_work_unit_t *unit =
            execution_plan_distributed_v3_work_unit_for_dpu(plan, dpu_id);
        if (unit == NULL || dpu_handle_at(set, dpu_id, &handles[dpu_id]) != 0) {
            *error = DPU_ERR_INVALID_PROFILE;
            return 1;
        }
    }
    {
        const double launch_started = now_s();
        metrics->launch_attempted = 1;
        *error = dpu_launch(set, DPU_SYNCHRONOUS);
        metrics->kernel_launch_api_calls++;
        metrics->launch_count++;
        metrics->repeats[repeat_index].launch_sync_time_s = now_s() - launch_started;
    }
    if (*error != DPU_OK) return 1;
    completion_started = now_s();
    for (uint32_t dpu_id = 0u; dpu_id < plan->header.dpu_count; dpu_id++) {
        const execution_plan_v3_work_unit_t *unit =
            execution_plan_distributed_v3_work_unit_for_dpu(plan, dpu_id);
        *error = dpu_copy_from(handles[dpu_id], "RESIDENT_COMPLETION", 0u,
            &completions[dpu_id], sizeof(completions[dpu_id]));
        if (*error != DPU_OK) return 1;
        metrics->completion_reads++;
        metrics->completion_d2h_bytes += sizeof(completions[dpu_id]);
        metrics->repeats[repeat_index].completion_d2h_bytes += sizeof(completions[dpu_id]);
        if (unit == NULL || v3_validate_completion(&completions[dpu_id], unit,
                plan->header.package_operation_index, failure_message, error) != 0) return 1;
        metrics->repeats[repeat_index].dpu_cycles[dpu_id] = completions[dpu_id].dpu_run_time_cycles;
        metrics->repeats[repeat_index].dpu_work_elements[dpu_id] = completions[dpu_id].output_elements_processed;
        metrics->repeats[repeat_index].dpu_completions[dpu_id] = 1u;
        metrics->native_kernel_executed = 1;
    }
    metrics->repeats[repeat_index].completion_read_and_validation_time_s =
        now_s() - completion_started;
    {
        const double assembly_started = now_s();
        if (plan->header.partition_mode == EXECUTION_PLAN_V3_PARTITION_CONTRACTED) {
            unsigned char *partial = (unsigned char *)calloc(output->transfer_bytes, 1u);
            if (partial == NULL) { *error = DPU_ERR_INTERNAL; return 1; }
            if (packed_int8) {
                int64_t *accumulator = (int64_t *)calloc(output->elements, sizeof(*accumulator));
                if (accumulator == NULL) {
                    free(partial); *error = DPU_ERR_INTERNAL; return 1;
                }
                for (uint32_t dpu_id = 0u; dpu_id < plan->header.dpu_count; dpu_id++) {
                    const execution_plan_v3_work_unit_t *unit =
                        execution_plan_distributed_v3_work_unit_for_dpu(plan, dpu_id);
                    *error = dpu_copy_from(handles[dpu_id], "RESIDENT_SLOT_POOL",
                        output_slot->offset_bytes, partial, output->transfer_bytes);
                    if (*error != DPU_OK) break;
                    metrics->reduction_d2h_bytes += output->transfer_bytes;
                    metrics->repeats[repeat_index].output_d2h_bytes += output->transfer_bytes;
                    if (checksum_f32_bytes(14695981039346656037ULL, partial,
                            unit->output_elements) != completions[dpu_id].output_checksum_fnv1a64) {
                        *error = DPU_ERR_INVALID_PROFILE; break;
                    }
                    const int32_t *values = (const int32_t *)partial;
                    for (uint32_t index = 0u; index < output->elements; index++) {
                        accumulator[index] += (int64_t)values[index];
                        metrics->reduction_element_additions++;
                    }
                }
                if (*error == DPU_OK) {
                    for (uint32_t index = 0u; index < output->elements; index++) {
                        if (accumulator[index] < INT32_MIN || accumulator[index] > INT32_MAX) {
                            *error = DPU_ERR_INVALID_PROFILE; break;
                        }
                        raw_i32_buffer[index] = (int32_t)accumulator[index];
                    }
                }
                free(accumulator);
            } else {
                double *accumulator = (double *)calloc(output->elements, sizeof(*accumulator));
                if (accumulator == NULL) {
                    free(partial); *error = DPU_ERR_INTERNAL; return 1;
                }
                for (uint32_t dpu_id = 0u; dpu_id < plan->header.dpu_count; dpu_id++) {
                    const execution_plan_v3_work_unit_t *unit =
                        execution_plan_distributed_v3_work_unit_for_dpu(plan, dpu_id);
                    *error = dpu_copy_from(handles[dpu_id], "RESIDENT_SLOT_POOL",
                        output_slot->offset_bytes, partial, output->transfer_bytes);
                    if (*error != DPU_OK) break;
                    metrics->reduction_d2h_bytes += output->transfer_bytes;
                    metrics->repeats[repeat_index].output_d2h_bytes += output->transfer_bytes;
                    if (buffer_is_finite(partial, output->raw_bytes) != 0 ||
                        checksum_f32_bytes(14695981039346656037ULL, partial,
                            unit->output_elements) != completions[dpu_id].output_checksum_fnv1a64) {
                        *error = DPU_ERR_INVALID_PROFILE; break;
                    }
                    const float *values = (const float *)partial;
                    for (uint32_t index = 0u; index < output->elements; index++) {
                        accumulator[index] += (double)values[index];
                        metrics->reduction_element_additions++;
                    }
                }
                if (*error == DPU_OK) {
                    float *result = (float *)final_buffer;
                    for (uint32_t index = 0u; index < output->elements; index++) {
                        if (!isfinite(accumulator[index]) || fabs(accumulator[index]) > FLT_MAX) {
                            *error = DPU_ERR_INVALID_PROFILE; break;
                        }
                        result[index] = (float)accumulator[index];
                    }
                }
                free(accumulator);
            }
            free(partial);
        } else {
            unsigned char *assembled = packed_int8
                ? (unsigned char *)raw_i32_buffer : final_buffer;
            for (uint32_t dpu_id = 0u; dpu_id < plan->header.dpu_count; dpu_id++) {
                const execution_plan_v3_work_unit_t *unit =
                    execution_plan_distributed_v3_work_unit_for_dpu(plan, dpu_id);
                const uint64_t raw_start = (uint64_t)unit->output_offset * sizeof(float);
                const uint64_t raw_bytes = (uint64_t)unit->output_elements * sizeof(float);
                const uint64_t aligned_start = raw_start & ~7ULL;
                const uint64_t aligned_end = (raw_start + raw_bytes + 7ULL) & ~7ULL;
                unsigned char *scratch = (unsigned char *)calloc((size_t)(aligned_end - aligned_start), 1u);
                if (scratch == NULL || aligned_end > output->transfer_bytes) {
                    free(scratch); *error = DPU_ERR_INVALID_PROFILE; break;
                }
                *error = dpu_copy_from(handles[dpu_id], "RESIDENT_SLOT_POOL",
                    output_slot->offset_bytes + (uint32_t)aligned_start, scratch,
                    (size_t)(aligned_end - aligned_start));
                if (*error == DPU_OK) {
                    if ((!packed_int8 && buffer_is_finite(
                            scratch + raw_start - aligned_start, (size_t)raw_bytes) != 0) ||
                        checksum_f32_bytes(14695981039346656037ULL,
                            scratch + raw_start - aligned_start, unit->output_elements) !=
                            completions[dpu_id].output_checksum_fnv1a64) {
                        *error = DPU_ERR_INVALID_PROFILE;
                    } else {
                        memcpy(assembled + raw_start, scratch + raw_start - aligned_start,
                            (size_t)raw_bytes);
                    }
                    metrics->final_d2h_bytes += aligned_end - aligned_start;
                    metrics->repeats[repeat_index].output_d2h_bytes += aligned_end - aligned_start;
                }
                free(scratch);
                if (*error != DPU_OK) break;
            }
        }
        metrics->repeats[repeat_index].assembly_time_s = now_s() - assembly_started;
    }
    if (*error != DPU_OK) {
        if (failure_message != NULL && *failure_message == NULL) *failure_message = strdup(
            packed_int8
                ? "distributed v3 int32 output assembly or int64 reduction failed"
                : "distributed v3 output assembly or float64 reduction failed");
        return 1;
    }
    if (packed_int8) {
        const resident_operation_t *operation =
            &request->resident.operations[plan->header.package_operation_index];
        const float output_scale = operation->left_scale * operation->right_scale;
        const double dequant_started = now_s();
        if (!isfinite(output_scale) || output_scale <= 0.0f ||
            v3_validate_integer_reference(raw_i32_buffer, integer_reference, output->elements,
                integer_validation, failure_message, error) != 0) {
            return 1;
        }
        for (uint32_t index = 0u; index < output->elements; index++) {
            ((float *)final_buffer)[index] = (float)raw_i32_buffer[index] * output_scale;
        }
        metrics->repeats[repeat_index].host_dequantization_time_s =
            now_s() - dequant_started;
        if (output->raw_output_path == NULL ||
            write_exact(output->raw_output_path, raw_i32_buffer, output->raw_bytes) != 0) {
            *error = DPU_ERR_INTERNAL;
            return 1;
        }
    }
    if (buffer_is_finite(final_buffer, output->raw_bytes) != 0 ||
        v3_validate_policy_reference(final_buffer, policy_reference, output->elements,
            policy_validation, failure_message, error) != 0) {
        return 1;
    }
    if (write_exact(output->path, final_buffer, output->raw_bytes) != 0) {
        *error = DPU_ERR_INTERNAL;
        return 1;
    }
    metrics->repeats[repeat_index].total_time_s = now_s() - started;
    return 0;
}

static void v3_write_response(
    const char *path, const char *status, const char *failure_stage, const char *error_message,
    const execution_plan_request_t *request, const execution_plan_distributed_v3_t *plan,
    const execution_plan_provider_t *provider, const v3_metrics_t *metrics,
    const v3_policy_reference_t *policy_validation, const char host_binary_sha256[65],
    const v3_integer_reference_t *integer_validation,
    const char staged_dpu_binary_sha256[65], const char initialization_binary_sha256[65],
    double allocation_s, double binary_load_s,
    double release_s
) {
    FILE *file = path == NULL ? stdout : fopen(path, "wb");
    char output_sha256[65] = {0};
    char raw_output_sha256[65] = {0};
    if (file == NULL) return;
    const uint32_t dpu_count = plan == NULL ? 0u : plan->header.dpu_count;
    uint64_t assigned_min = UINT64_MAX;
    uint64_t assigned_max = 0u;
    uint64_t assigned_total = 0u;
    const int allocation_confirmed = provider != NULL && provider->allocation_used &&
        provider->observed_dpus == dpu_count && provider->observed_ranks == 1u;
    const int release_confirmed = provider != NULL && provider->release_succeeded &&
        provider->allocation_active == 0;
    const uint64_t actual_h2d = metrics == NULL ? 0ull :
        metrics->descriptor_h2d_bytes + metrics->operand_h2d_bytes + metrics->reset_h2d_bytes;
    const uint64_t actual_d2h = metrics == NULL ? 0ull :
        metrics->completion_d2h_bytes + metrics->final_d2h_bytes + metrics->reduction_d2h_bytes;
    const int int8_requantization = plan != NULL &&
        plan->header.numeric_mode == EXECUTION_PLAN_V3_NUMERIC_INT8_REQUANTIZE;
    const int packed_int8 = plan != NULL &&
        plan->header.numeric_mode == EXECUTION_PLAN_V3_NUMERIC_HOST_PACKED_INT8;
    const int contracted_partition = plan != NULL &&
        plan->header.partition_mode == EXECUTION_PLAN_V3_PARTITION_CONTRACTED;
    const char *partition_strategy = contracted_partition ? "contracted" : "output";
    const char *collective_provider = contracted_partition ? "host_mediated_sum_v1" : "none";
    const char *reconstruction_provider = contracted_partition
        ? (packed_int8 ? "host_int64_reduction_v1" : "host_float64_reduction_v1")
        : "host_owned_range_assembly_v1";
    if (request != NULL && request->resident.final_count == 1u) {
        (void)execution_plan_sha256_file(
            request->resident.final_outputs[0].path, output_sha256
        );
        if (request->resident.final_outputs[0].raw_output_path != NULL) {
            (void)execution_plan_sha256_file(
                request->resident.final_outputs[0].raw_output_path, raw_output_sha256
            );
        }
    }
    if (plan != NULL && plan->work_units != NULL) {
        for (uint32_t index = 0u; index < plan->work_unit_count; index++) {
            const execution_plan_v3_work_unit_t *unit = &plan->work_units[index];
            const uint64_t assigned = plan->header.partition_mode == EXECUTION_PLAN_V3_PARTITION_CONTRACTED
                ? unit->contracted_elements : unit->output_elements;
            if (assigned < assigned_min) assigned_min = assigned;
            if (assigned > assigned_max) assigned_max = assigned;
            assigned_total += assigned;
        }
    }
    if (assigned_min == UINT64_MAX) assigned_min = 0u;
    fprintf(file, "{\"schema_version\":\"%s\",\"status\":", EXECUTION_PLAN_V3_RESPONSE_SCHEMA);
    v3_json_string(file, status == NULL ? "failed" : status);
    fputs(",\"failure_stage\":", file); if (failure_stage == NULL) fputs("null", file); else v3_json_string(file, failure_stage);
    fputs(",\"error\":", file); if (error_message == NULL) fputs("null", file); else v3_json_string(file, error_message);
    fprintf(file, ",\"target_requested\":\"hardware\",\"target_observed\":\"%s\",\"backend_id\":\"upmem_sdk_hardware_execution_plan_v3\",\"backend_family\":\"upmem_sdk\",\"execution_class\":\"resident_taskgraph\",\"kernel_strategy\":\"resident_generic_contract\",\"requested_dpu_count\":%u,\"allocated_dpu_count\":%u,\"requested_rank_path\":", provider != NULL && provider->allocation_used ? "physical_hardware" : "not_allocated", dpu_count, provider != NULL && provider->allocation_used ? provider->observed_dpus : 0u);
    v3_json_string(file, provider == NULL ? "" : provider->requested_rank_path);
    fprintf(file, ",\"observed_rank_count\":%u,\"tasklets_per_dpu\":%u,\"rank_count\":%u,\"one_rank\":%s,\"single_rank\":%s,\"partition_strategy\":\"%s\",\"dispatch_mode\":\"bulk_set_synchronous_v1\",\"kernel_launch_api_calls\":%llu,\"dpu_program_instances\":%u,\"explicit_sync_api_calls\":%llu,\"launch_count_semantics\":\"set_launch_api_calls\",\"synchronize_count_semantics\":\"explicit_dpu_sync_api_calls\",\"numeric_mode\":\"%s\",\"numeric_transport\":\"%s\",\"numeric_arithmetic\":\"%s\",\"requantization_scope\":\"%s\",\"packed_int8_transfer\":%s,\"host_quantization\":%s,\"dpu_intermediate_requantization\":false,\"simulator_kernel_executed\":false,\"cpu_fallback_used\":false,\"hardware_kernel_executed\":%s,\"native_kernel_executed\":%s,\"hardware_allocation_verified\":%s,\"hardware_release_verified\":%s,\"allocation_provider\":\"upmem_sdk_rank_profile_v1\",\"simplepim_role\":\"initialization_binary_and_management_state_only\",\"kernel_provider\":\"thesis_resident_generic_c_v3\",\"transfer_provider\":\"upmem_sdk_synchronous_v1\",\"collective_provider\":\"%s\",\"reconstruction_provider\":\"%s\",\"allocation\":{\"attempted\":%s,\"confirmed\":%s,\"release_attempted\":%s,\"release_confirmed\":%s},\"timing_scope\":\"per_repetition_total_time_s_includes_set_launch_completion_reads_and_validation_output_d2h_assembly_and_output_write; kernel_launch_sync_time_s_covers_only_dpu_launch_set_synchronous; completion_read_and_validation_time_s_covers_completion_d2h_reads_and_contract_validation\",\"timing\":{\"allocation_time_s\":%.9f,\"binary_load_time_s\":%.9f,\"release_time_s\":%.9f},\"requested_warmups\":%u,\"requested_repetitions\":%u,\"run_total_transfers\":{\"h2d_bytes\":%llu,\"d2h_bytes\":%llu,\"total_bytes\":%llu,\"descriptor_h2d_bytes\":%llu,\"operand_h2d_bytes\":%llu,\"reset_h2d_bytes\":%llu,\"completion_d2h_bytes\":%llu,\"final_d2h_bytes\":%llu,\"reduction_d2h_bytes\":%llu},\"transfers\":{\"h2d_bytes\":%llu,\"d2h_bytes\":%llu,\"actual_h2d_bytes\":%llu,\"actual_d2h_bytes\":%llu,\"actual_transfer_bytes\":%llu,\"descriptor_h2d_bytes\":%llu,\"operand_h2d_bytes\":%llu,\"reset_h2d_bytes\":%llu,\"completion_d2h_bytes\":%llu,\"final_d2h_bytes\":%llu,\"reduction_d2h_bytes\":%llu},\"load_balance\":{\"dpu_count\":%u,\"assigned_work_elements_min\":%llu,\"assigned_work_elements_max\":%llu,\"assigned_work_elements_total\":%llu,\"ratio\":%.9f},\"repetitions\":[",
        provider == NULL ? 0u : provider->observed_ranks,
        plan == NULL ? 0u : plan->header.tasklets_per_dpu,
        provider == NULL ? 0u : provider->observed_ranks,
        provider != NULL && provider->observed_ranks == 1u ? "true" : "false",
        provider != NULL && provider->observed_ranks == 1u ? "true" : "false",
        partition_strategy,
        metrics == NULL ? 0ull : (unsigned long long)metrics->kernel_launch_api_calls,
        dpu_count,
        metrics == NULL ? 0ull : (unsigned long long)metrics->explicit_sync_api_calls,
        packed_int8 ? "host_packed_int8" :
            (int8_requantization ? "per_task_resident_requantize" : "float32"),
        packed_int8 ? "host_packed_int8_mram" : "float32_mram",
        packed_int8 ? "int8_multiply_int32_accumulate" :
            (int8_requantization ? "int8_requantized" : "float32"),
        packed_int8 ? "host_initial_once_final_host_dequantize" :
            (int8_requantization ? "per_task_on_dpu" : "none"),
        packed_int8 ? "true" : "false",
        packed_int8 ? "true" : "false",
        status != NULL && strcmp(status, "completed") == 0 && metrics != NULL && metrics->native_kernel_executed ? "true" : "false",
        status != NULL && strcmp(status, "completed") == 0 && metrics != NULL && metrics->native_kernel_executed ? "true" : "false",
        allocation_confirmed ? "true" : "false",
        release_confirmed ? "true" : "false",
        collective_provider,
        reconstruction_provider,
        provider != NULL && provider->allocation_attempted ? "true" : "false",
        allocation_confirmed ? "true" : "false",
        provider != NULL && provider->release_attempted ? "true" : "false",
        release_confirmed ? "true" : "false",
        allocation_s, binary_load_s, release_s,
        request == NULL ? 0u : request->warmup_repetitions,
        request == NULL ? 0u : request->measured_repetitions,
        (unsigned long long)actual_h2d, (unsigned long long)actual_d2h,
        (unsigned long long)(actual_h2d + actual_d2h),
        metrics == NULL ? 0ull : (unsigned long long)metrics->descriptor_h2d_bytes,
        metrics == NULL ? 0ull : (unsigned long long)metrics->operand_h2d_bytes,
        metrics == NULL ? 0ull : (unsigned long long)metrics->reset_h2d_bytes,
        metrics == NULL ? 0ull : (unsigned long long)metrics->completion_d2h_bytes,
        metrics == NULL ? 0ull : (unsigned long long)metrics->final_d2h_bytes,
        metrics == NULL ? 0ull : (unsigned long long)metrics->reduction_d2h_bytes,
        (unsigned long long)actual_h2d, (unsigned long long)actual_d2h,
        (unsigned long long)actual_h2d, (unsigned long long)actual_d2h,
        (unsigned long long)(actual_h2d + actual_d2h),
        metrics == NULL ? 0ull : (unsigned long long)metrics->descriptor_h2d_bytes,
        metrics == NULL ? 0ull : (unsigned long long)metrics->operand_h2d_bytes,
        metrics == NULL ? 0ull : (unsigned long long)metrics->reset_h2d_bytes,
        metrics == NULL ? 0ull : (unsigned long long)metrics->completion_d2h_bytes,
        metrics == NULL ? 0ull : (unsigned long long)metrics->final_d2h_bytes,
        metrics == NULL ? 0ull : (unsigned long long)metrics->reduction_d2h_bytes,
        dpu_count, (unsigned long long)assigned_min, (unsigned long long)assigned_max,
        (unsigned long long)assigned_total,
        assigned_min == 0u ? 0.0 : (double)assigned_max / (double)assigned_min);
    if (metrics != NULL) for (uint32_t repeat = 0u; repeat < metrics->repeat_count; repeat++) {
        if (repeat != 0u) fputc(',', file);
        fprintf(file, "{\"repeat_id\":%u,\"warmup\":%s,\"total_time_s\":%.9f,\"launch_sync_time_s\":%.9f,\"completion_read_and_validation_time_s\":%.9f,\"assembly_time_s\":%.9f,\"host_dequantization_time_s\":%.9f,\"transfers\":{\"h2d_bytes\":%llu,\"d2h_bytes\":%llu,\"total_bytes\":%llu,\"reset_h2d_bytes\":%llu,\"completion_d2h_bytes\":%llu,\"output_d2h_bytes\":%llu},\"per_dpu\":[",
            repeat, request != NULL && repeat < request->warmup_repetitions ? "true" : "false",
            metrics->repeats[repeat].total_time_s, metrics->repeats[repeat].launch_sync_time_s,
            metrics->repeats[repeat].completion_read_and_validation_time_s,
            metrics->repeats[repeat].assembly_time_s,
            metrics->repeats[repeat].host_dequantization_time_s,
            (unsigned long long)metrics->repeats[repeat].reset_h2d_bytes,
            (unsigned long long)(metrics->repeats[repeat].completion_d2h_bytes + metrics->repeats[repeat].output_d2h_bytes),
            (unsigned long long)(metrics->repeats[repeat].reset_h2d_bytes + metrics->repeats[repeat].completion_d2h_bytes + metrics->repeats[repeat].output_d2h_bytes),
            (unsigned long long)metrics->repeats[repeat].reset_h2d_bytes,
            (unsigned long long)metrics->repeats[repeat].completion_d2h_bytes,
            (unsigned long long)metrics->repeats[repeat].output_d2h_bytes);
        for (uint32_t dpu_id = 0u; dpu_id < dpu_count; dpu_id++) {
            if (dpu_id != 0u) fputc(',', file);
            fprintf(file, "{\"dpu_id\":%u,\"runtime_cycles\":%llu,\"work_elements\":%llu,\"completion_count\":%u}",
                dpu_id, (unsigned long long)metrics->repeats[repeat].dpu_cycles[dpu_id],
                (unsigned long long)metrics->repeats[repeat].dpu_work_elements[dpu_id],
                metrics->repeats[repeat].dpu_completions[dpu_id]);
        }
        fputs("]}", file);
    }
    fprintf(file, "],\"application_visible_transfer_totals\":{\"h2d_bytes\":%llu,\"d2h_bytes\":%llu,\"total_bytes\":%llu},\"policy_reference_validation\":{\"status\":\"%s\",\"passed\":%s,\"max_abs_error\":%.9g,\"tolerance\":%.9g,\"finite\":%s,\"exact_match\":%s,\"reference_path\":",
        (unsigned long long)actual_h2d, (unsigned long long)actual_d2h,
        (unsigned long long)(actual_h2d + actual_d2h),
        v3_policy_validation_passed(policy_validation) ? "passed" :
            (policy_validation != NULL && policy_validation->loaded ? "failed" : "not_run"),
        v3_policy_validation_passed(policy_validation) ? "true" : "false",
        policy_validation == NULL ? 0.0 : policy_validation->max_abs_error,
        policy_validation == NULL ? 0.0 : policy_validation->tolerance,
        policy_validation != NULL && policy_validation->finite ? "true" : "false",
        policy_validation != NULL && policy_validation->exact_match ? "true" : "false");
    v3_json_string(file, policy_validation == NULL ? "" : policy_validation->path);
    fprintf(file, ",\"reference_sha256\":\"%s\"},\"exact_integer_validation\":{\"status\":\"%s\",\"required\":%s,\"passed\":%s,\"exact_match\":%s,\"mismatch_count\":%llu,\"reference_path\":",
        policy_validation == NULL ? "" : policy_validation->actual_sha256,
        packed_int8
            ? (integer_validation != NULL && integer_validation->passed ? "passed" : "failed")
            : "not_applicable",
        packed_int8 ? "true" : "false",
        packed_int8 && integer_validation != NULL && integer_validation->passed ? "true" : "false",
        packed_int8 && integer_validation != NULL && integer_validation->passed ? "true" : "false",
        integer_validation == NULL ? 0ull :
            (unsigned long long)integer_validation->mismatch_count);
    v3_json_string(file, integer_validation == NULL ? "" : integer_validation->path);
    fprintf(file, ",\"reference_sha256\":\"%s\"},\"output_sha256\":\"%s\",\"raw_int32_output_sha256\":\"%s\",\"launch_attempted\":%s,\"launch_count\":%llu,\"synchronize_count\":%llu,\"completion_reads\":%llu,\"kernel_launch_api_calls\":%llu,\"explicit_sync_api_calls\":%llu,\"reduction_d2h_bytes\":%llu,\"reduction_element_additions\":%llu,\"reduction_accumulator\":\"%s\",\"package_file_sha256\":\"%s\",\"distributed_plan_v3_sha256\":\"%s\",\"host_binary_sha256\":\"%s\",\"staged_dpu_binary_sha256\":\"%s\",\"initialization_binary_sha256\":\"%s\"}\n",
        integer_validation == NULL ? "" : integer_validation->actual_sha256,
        output_sha256,
        raw_output_sha256,
        metrics != NULL && metrics->launch_attempted ? "true" : "false",
        metrics == NULL ? 0ull : (unsigned long long)metrics->launch_count,
        metrics == NULL ? 0ull : (unsigned long long)metrics->synchronize_count,
        metrics == NULL ? 0ull : (unsigned long long)metrics->completion_reads,
        metrics == NULL ? 0ull : (unsigned long long)metrics->kernel_launch_api_calls,
        metrics == NULL ? 0ull : (unsigned long long)metrics->explicit_sync_api_calls,
        metrics == NULL ? 0ull : (unsigned long long)metrics->reduction_d2h_bytes,
        metrics == NULL ? 0ull : (unsigned long long)metrics->reduction_element_additions,
        packed_int8 ? "int64_then_int32" : "float64_then_float32",
        request == NULL ? "" : request->actual_package_file_sha256,
        plan == NULL || plan->file_sha256 == NULL ? "" : plan->file_sha256,
        host_binary_sha256 == NULL ? "" : host_binary_sha256,
        staged_dpu_binary_sha256 == NULL ? "" : staged_dpu_binary_sha256,
        initialization_binary_sha256 == NULL ? "" : initialization_binary_sha256);
    if (path == NULL) fflush(file); else fclose(file);
}

static void v3_usage(const char *program) {
    fprintf(stderr, "usage: %s (--validate-plan|--execute-plan) --resident-package manifest.json --distributed-plan-v3 sidecar.bin [--policy-reference reference_f32.bin] [--policy-reference-sha256 HASH] [--policy-tolerance T] [--integer-reference reference_i32.bin] [--integer-reference-sha256 HASH] [--response result.json] [--warmups N] [--repetitions N] [--timeout-s N]\n", program);
}

int main(int argc, char **argv) {
    const char *manifest_path = NULL, *sidecar_path = NULL, *response_path = NULL;
    const char *policy_reference_path = NULL, *policy_reference_sha256 = NULL;
    const char *integer_reference_path = NULL, *integer_reference_sha256 = NULL;
    char default_policy_reference[PATH_MAX] = {0};
    char host_binary_sha256[65] = {0}, staged_dpu_binary_sha256[65] = {0};
    char initialization_binary_sha256[65] = {0};
    uint32_t warmups = 1u, repetitions = 1u, timeout_s = 60u;
    int validate_only = 0, execute = 0, rc = 1, packed_int8 = 0;
    char *owned_error = NULL;
    const char *failure_stage = NULL, *error_message = NULL;
    execution_plan_request_t request = {0};
    execution_plan_distributed_v3_t plan = {0};
    execution_plan_provider_t provider = {0};
    v3_metrics_t metrics = {0};
    v3_policy_reference_t policy_validation = {.tolerance = 1.0e-5};
    v3_integer_reference_t integer_validation = {0};
    unsigned char **inputs = NULL, *final_buffer = NULL, *policy_reference = NULL;
    unsigned char *integer_reference_bytes = NULL;
    int32_t *raw_i32_buffer = NULL;
    char initialization_binary[PATH_MAX];
    dpu_error_t error = DPU_OK;
    double allocation_s = 0.0, binary_load_s = 0.0, release_s = 0.0;
    for (int index = 1; index < argc; index++) {
        if (strcmp(argv[index], "--validate-plan") == 0) validate_only = 1;
        else if (strcmp(argv[index], "--execute-plan") == 0) execute = 1;
        else if (strcmp(argv[index], "--resident-package") == 0 && index + 1 < argc) manifest_path = argv[++index];
        else if (strcmp(argv[index], "--distributed-plan-v3") == 0 && index + 1 < argc) sidecar_path = argv[++index];
        else if (strcmp(argv[index], "--policy-reference") == 0 && index + 1 < argc) policy_reference_path = argv[++index];
        else if (strcmp(argv[index], "--policy-reference-sha256") == 0 && index + 1 < argc) policy_reference_sha256 = argv[++index];
        else if (strcmp(argv[index], "--policy-tolerance") == 0 && index + 1 < argc) policy_validation.tolerance = strtod(argv[++index], NULL);
        else if (strcmp(argv[index], "--integer-reference") == 0 && index + 1 < argc) integer_reference_path = argv[++index];
        else if (strcmp(argv[index], "--integer-reference-sha256") == 0 && index + 1 < argc) integer_reference_sha256 = argv[++index];
        else if (strcmp(argv[index], "--response") == 0 && index + 1 < argc) response_path = argv[++index];
        else if (strcmp(argv[index], "--warmups") == 0 && index + 1 < argc) warmups = (uint32_t)strtoul(argv[++index], NULL, 10);
        else if (strcmp(argv[index], "--repetitions") == 0 && index + 1 < argc) repetitions = (uint32_t)strtoul(argv[++index], NULL, 10);
        else if (strcmp(argv[index], "--timeout-s") == 0 && index + 1 < argc) timeout_s = (uint32_t)strtoul(argv[++index], NULL, 10);
        else { v3_usage(argv[0]); return 2; }
    }
    if (validate_only == execute || manifest_path == NULL || sidecar_path == NULL ||
        warmups > 4u || repetitions == 0u || repetitions > EXECUTION_PLAN_V3_MAX_REPETITIONS || timeout_s == 0u) {
        v3_usage(argv[0]); return 2;
    }
    if (resident_request_load_execution_plan_v3(manifest_path, &request.resident, &owned_error) != 0) {
        failure_stage = "manifest_parse_failed"; error_message = owned_error; goto done;
    }
    request.warmup_repetitions = warmups;
    request.measured_repetitions = repetitions;
    request.resident_manifest_path = strdup(manifest_path);
    if (execution_plan_sha256_file(request.resident.package_path, request.actual_package_file_sha256) != 0) {
        failure_stage = "package_hash_failed"; error_message = "resident package hash failed"; goto done;
    }
    {
        unsigned char digest[32];
        if (v3_digest_text(request.actual_package_file_sha256, digest) != 0 ||
            execution_plan_distributed_v3_load(sidecar_path, digest, &request.resident, &plan, &owned_error) != 0) {
            failure_stage = "hardware_profile_violation"; error_message = owned_error; goto done;
        }
    }
    if (request.resident.requested_dpus != plan.header.dpu_count) {
        failure_stage = "hardware_profile_violation"; error_message = "resident and v3 DPU counts differ"; goto done;
    }
    packed_int8 = plan.header.numeric_mode == EXECUTION_PLAN_V3_NUMERIC_HOST_PACKED_INT8;
    if (packed_int8 && (integer_reference_path == NULL || integer_reference_sha256 == NULL)) {
        failure_stage = "hardware_profile_violation";
        error_message = "packed int8 execution requires an exact int32 reference and SHA-256";
        goto done;
    }
    if (validate_only) { rc = 0; goto done; }
    if (getenv("UPMEM_ALLOW_PHYSICAL_HARDWARE") == NULL || strcmp(getenv("UPMEM_ALLOW_PHYSICAL_HARDWARE"), "1") != 0) {
        failure_stage = "hardware_opt_in_missing"; error_message = "UPMEM_ALLOW_PHYSICAL_HARDWARE=1 is required"; goto done;
    }
    if (getenv("DPU_BACKEND") != NULL) { failure_stage = "hardware_profile_violation"; error_message = "DPU_BACKEND must be unset for the physical route"; goto done; }
    if (getenv("UPMEM_HW_RANK_PATH") == NULL || getenv("UPMEM_HW_RANK_PATH")[0] == '\0') {
        failure_stage = "hardware_rank_path_missing"; error_message = "UPMEM_HW_RANK_PATH is required"; goto done;
    }
    if (policy_reference_path == NULL) {
        const char *slash = strrchr(sidecar_path, '/');
        const size_t directory_bytes = slash == NULL ? 0u : (size_t)(slash - sidecar_path + 1);
        if (directory_bytes + sizeof("reference_f32.bin") > sizeof(default_policy_reference)) {
            failure_stage = "policy_reference_validation_failed"; error_message = "default policy-reference path is too long"; goto done;
        }
        if (directory_bytes != 0u) memcpy(default_policy_reference, sidecar_path, directory_bytes);
        memcpy(default_policy_reference + directory_bytes, "reference_f32.bin", sizeof("reference_f32.bin"));
        policy_reference_path = default_policy_reference;
    }
    policy_validation.path = policy_reference_path;
    policy_validation.expected_sha256 = policy_reference_sha256;
    integer_validation.path = integer_reference_path;
    integer_validation.expected_sha256 = integer_reference_sha256;
    if (resolve_sibling("dpu_simplepim_management_init", initialization_binary) != 0) { failure_stage = "sdk_discovery_failed"; error_message = "SimplePIM initialization binary is missing beside the v3 host"; goto done; }
    inputs = (unsigned char **)calloc(request.resident.input_count, sizeof(*inputs));
    final_buffer = (unsigned char *)calloc(request.resident.final_outputs[0].transfer_bytes, 1u);
    if (packed_int8) {
        raw_i32_buffer = (int32_t *)calloc(
            request.resident.final_outputs[0].transfer_bytes, 1u
        );
    }
    if ((request.resident.input_count != 0u && inputs == NULL) || final_buffer == NULL ||
        (packed_int8 && raw_i32_buffer == NULL) ||
        v3_prepare_inputs(&request, inputs, &owned_error) != 0) {
        failure_stage = "operand_transfer_failed"; error_message = owned_error == NULL ? "input preparation failed" : owned_error; goto release;
    }
    if (v3_load_policy_reference(&policy_validation, request.resident.final_outputs[0].raw_bytes,
            &policy_reference, &owned_error) != 0) {
        failure_stage = "policy_reference_validation_failed"; error_message = owned_error; goto release;
    }
    if (packed_int8 && v3_load_integer_reference(
            &integer_validation, request.resident.final_outputs[0].raw_bytes,
            &integer_reference_bytes, &owned_error) != 0) {
        failure_stage = "integer_reference_validation_failed";
        error_message = owned_error;
        goto release;
    }
    if (execution_plan_sha256_file(argv[0], host_binary_sha256) != 0 ||
        execution_plan_sha256_file(request.resident.dpu_binary_path, staged_dpu_binary_sha256) != 0 ||
        execution_plan_sha256_file(initialization_binary, initialization_binary_sha256) != 0) {
        failure_stage = "binary_hash_failed"; error_message = "host, staged DPU, or SimplePIM initialization binary hash failed"; goto release;
    }
    {
        const double started = now_s();
        error = execution_plan_provider_init_on_rank(&provider, plan.header.dpu_count,
            getenv("UPMEM_HW_RANK_PATH"), initialization_binary);
        allocation_s = now_s() - started;
        if (error != DPU_OK || provider.observed_dpus != plan.header.dpu_count) { failure_stage = "hardware_allocation_failed"; error_message = "SimplePIM allocation did not return the requested physical DPU set"; goto release; }
    }
    {
        const double started = now_s();
        error = dpu_load(provider.set, request.resident.dpu_binary_path, NULL);
        binary_load_s = now_s() - started;
        if (error != DPU_OK) { failure_stage = "binary_load_failed"; error_message = "v3 DPU binary load failed"; goto release; }
    }
    {
        struct dpu_set_t dpu; uint32_t index;
        DPU_FOREACH(provider.set, dpu, index) {
            if (v3_copy_package_to_dpu(dpu, index, &request, &plan, inputs, &metrics, &error) != 0) break;
        }
        if (error != DPU_OK) { failure_stage = "argument_transfer_failed"; error_message = "v3 descriptor or operand transfer failed"; goto release; }
    }
    metrics.repeat_count = warmups + repetitions;
    metrics.repeats = (v3_repeat_timing_t *)calloc(metrics.repeat_count, sizeof(*metrics.repeats));
    if (metrics.repeats == NULL) { failure_stage = "host_allocation_failed"; error_message = "v3 repeat metadata allocation failed"; goto release; }
    alarm(timeout_s);
    for (uint32_t repeat = 0u; repeat < metrics.repeat_count; repeat++) {
        const double repetition_started = now_s();
        if (v3_execute_repetition(provider.set, &request, &plan, final_buffer,
                raw_i32_buffer, &metrics, repeat, policy_reference, &policy_validation,
                (const int32_t *)integer_reference_bytes, &integer_validation,
                &error, &owned_error) != 0) {
            failure_stage = execution_plan_interrupted ? "kernel_timeout" :
                (integer_validation.compared && !integer_validation.passed
                    ? "integer_reference_validation_failed"
                    : (policy_validation.compared && !policy_validation.passed
                        ? "policy_reference_validation_failed" : "kernel_launch_failed"));
            error_message = owned_error == NULL ? "v3 resident execution failed" : owned_error; goto release;
        }
        metrics.repeats[repeat].total_time_s = now_s() - repetition_started;
    }
    metrics.native_kernel_executed = 1;
    alarm(0u); rc = 0;
release:
    if (provider.allocation_attempted) {
        const double started = now_s();
        error = execution_plan_provider_release(&provider);
        release_s = now_s() - started;
        if (error != DPU_OK || !provider.release_succeeded) { failure_stage = "hardware_release_failed"; rc = 1; }
    }
    if (rc == 0 && (!policy_validation.passed ||
            (packed_int8 && !integer_validation.passed) || !metrics.native_kernel_executed ||
            provider.observed_dpus != plan.header.dpu_count || provider.observed_ranks != 1u ||
            !provider.release_succeeded)) {
        failure_stage = "execution_contract_failed";
        error_message = "native completion did not satisfy the hardware and policy-reference acceptance contract";
        rc = 1;
    }
done:
    v3_write_response(response_path, failure_stage == NULL ? (validate_only ? "validated" : "completed") : "failed",
        failure_stage, error_message, request.resident_manifest_path == NULL ? NULL : &request, &plan,
        &provider, &metrics, &policy_validation, host_binary_sha256, &integer_validation,
        staged_dpu_binary_sha256,
        initialization_binary_sha256,
        allocation_s, binary_load_s, release_s);
    alarm(0u);
    for (size_t index = 0u; index < request.resident.input_count; index++) free(inputs == NULL ? NULL : inputs[index]);
    free(inputs); free(final_buffer); free(raw_i32_buffer); free(policy_reference);
    free(integer_reference_bytes); free(metrics.repeats); free(owned_error);
    execution_plan_distributed_v3_free(&plan); execution_plan_request_free(&request);
    return rc;
}
