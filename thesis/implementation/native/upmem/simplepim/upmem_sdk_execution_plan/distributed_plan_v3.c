#define _POSIX_C_SOURCE 200809L

#include "distributed_plan_v3.h"

#if RESIDENT_OPERATION_ABI_VERSION != RESIDENT_OPERATION_ABI_V2
#error "distributed_plan_v3.c must be compiled with resident operation ABI v2"
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void v3_error(char **message, const char *value) {
    if (message != NULL && *message == NULL) *message = strdup(value);
}

static int read_file(const char *path, unsigned char **payload, size_t *length) {
    FILE *file = path == NULL ? NULL : fopen(path, "rb");
    long size;
    if (file == NULL || fseek(file, 0, SEEK_END) != 0) {
        if (file != NULL) fclose(file);
        return 1;
    }
    size = ftell(file);
    if (size < 0 || (unsigned long)size > 8u * 1024u * 1024u || fseek(file, 0, SEEK_SET) != 0) {
        fclose(file);
        return 1;
    }
    *payload = (unsigned char *)malloc((size_t)size);
    if (*payload == NULL && size != 0) {
        fclose(file);
        return 1;
    }
    if (fread(*payload, 1u, (size_t)size, file) != (size_t)size || fclose(file) != 0) {
        free(*payload);
        *payload = NULL;
        return 1;
    }
    *length = (size_t)size;
    return 0;
}

static int has_input(const resident_request_t *resident, uint32_t slot_id) {
    for (size_t index = 0u; index < resident->input_count; index++) {
        if (resident->inputs[index].slot_id == slot_id) return 1;
    }
    return 0;
}

static int magic_valid(const char magic[8]) {
    return memcmp(magic, EXECUTION_PLAN_V3_MAGIC, 7u) == 0 && magic[7] == '\0';
}

static int digest_text(const char text[65], unsigned char digest[32]) {
    for (uint32_t index = 0u; index < 32u; index++) {
        unsigned int value;
        if (sscanf(text + index * 2u, "%02x", &value) != 1 || value > 0xffu) return 1;
        digest[index] = (unsigned char)value;
    }
    return 0;
}

const execution_plan_v3_work_unit_t *execution_plan_distributed_v3_work_unit_for_dpu(
    const execution_plan_distributed_v3_t *plan,
    uint32_t dpu_id
) {
    if (plan == NULL || plan->work_units == NULL) return NULL;
    for (uint32_t index = 0u; index < plan->work_unit_count; index++) {
        if (plan->work_units[index].dpu_id == dpu_id) return &plan->work_units[index];
    }
    return NULL;
}

int execution_plan_distributed_v3_load(
    const char *path,
    const unsigned char expected_package_sha256[32],
    const resident_request_t *resident,
    execution_plan_distributed_v3_t *plan,
    char **error_message
) {
    unsigned char *payload = NULL;
    size_t length = 0u;
    execution_plan_v3_header_t header;
    uint64_t expected_file_bytes;
    char file_sha256[65];
    char operation_sha256[65];
    unsigned char operation_digest[32];
    unsigned char seen[EXECUTION_PLAN_V3_MAX_DPUS] = {0};
    uint32_t cursor = 0u;

    if (path == NULL || expected_package_sha256 == NULL || resident == NULL || plan == NULL) {
        v3_error(error_message, "distributed_plan_v3_parse_failed: missing sidecar inputs");
        return 1;
    }
    memset(plan, 0, sizeof(*plan));
    if (read_file(path, &payload, &length) != 0 || length < sizeof(header)) {
        v3_error(error_message, "distributed_plan_v3_parse_failed: sidecar is unreadable or truncated");
        free(payload);
        return 1;
    }
    memcpy(&header, payload, sizeof(header));
    if (!magic_valid(header.magic) || header.version != EXECUTION_PLAN_V3_VERSION ||
        header.header_bytes != sizeof(header) || header.record_bytes != sizeof(execution_plan_v3_work_unit_t) ||
        header.reserved0 != 0u || header.reserved1 != 0u ||
        header.provider_count != EXECUTION_PLAN_V3_PROVIDER_COUNT || header.work_unit_count == 0u ||
        header.work_unit_count > EXECUTION_PLAN_V3_MAX_DPUS || header.dpu_count == 0u ||
        header.dpu_count > EXECUTION_PLAN_V3_MAX_DPUS || header.work_unit_count != header.dpu_count ||
        header.tasklets_per_dpu == 0u || header.tasklets_per_dpu > EXECUTION_PLAN_V3_MAX_TASKLETS ||
        header.tasklets_per_dpu != NR_TASKLETS ||
        (header.partition_mode != EXECUTION_PLAN_V3_PARTITION_OUTPUT &&
         header.partition_mode != EXECUTION_PLAN_V3_PARTITION_CONTRACTED) ||
        (header.numeric_mode != EXECUTION_PLAN_V3_NUMERIC_FLOAT32 &&
         header.numeric_mode != EXECUTION_PLAN_V3_NUMERIC_INT8_REQUANTIZE)) {
        v3_error(error_message, "hardware_profile_violation: distributed v3 sidecar header is invalid");
        free(payload);
        return 1;
    }
    expected_file_bytes = (uint64_t)header.header_bytes +
        (uint64_t)header.work_unit_count * (uint64_t)header.record_bytes;
    if (expected_file_bytes != (uint64_t)length ||
        memcmp(header.package_sha256, expected_package_sha256, 32u) != 0) {
        v3_error(error_message, "hardware_profile_violation: distributed v3 package binding or length is invalid");
        free(payload);
        return 1;
    }
    if (resident->header.operation_count != 1u || header.package_operation_index != 0u ||
        header.operation_id != header.package_operation_index ||
        header.package_operation_index >= resident->header.operation_count) {
        v3_error(error_message, "hardware_profile_violation: distributed v3 sidecar must bind one package operation");
        free(payload);
        return 1;
    }
    {
        const resident_operation_t *operation = &resident->operations[header.package_operation_index];
        const uint32_t expected_numeric = operation->mode == 0u
            ? EXECUTION_PLAN_V3_NUMERIC_FLOAT32 : EXECUTION_PLAN_V3_NUMERIC_INT8_REQUANTIZE;
        if (operation->kind != RESIDENT_OPERATION_CONTRACT || operation->mode > 1u ||
            expected_numeric != header.numeric_mode || operation->slot_out_real != header.output_slot ||
            operation->output_elements != header.output_elements ||
            operation->args.output_elems != header.output_elements ||
            operation->args.contracted_elems != header.contracted_elements ||
            operation->args.dpu_slice_offset != 0u ||
            operation->args.dpu_slice_elements != header.output_elements ||
            operation->args.contracted_offset != 0u ||
            operation->args.contracted_elements_slice != header.contracted_elements ||
            !has_input(resident, operation->slot_a) || !has_input(resident, operation->slot_b) ||
            header.output_slot >= resident->header.slot_count || resident->final_count != 1u ||
            resident->final_outputs[0].slot_id != header.output_slot ||
            resident->final_outputs[0].elements != header.output_elements ||
            execution_plan_sha256_bytes((const unsigned char *)operation, sizeof(*operation), operation_sha256) != 0 ||
            digest_text(operation_sha256, operation_digest) != 0 ||
            memcmp(header.operation_sha256, operation_digest, sizeof(operation_digest)) != 0) {
            v3_error(error_message, "hardware_profile_violation: distributed v3 operation identity or numeric mode is invalid");
            free(payload);
            return 1;
        }
    }
    plan->work_units = (execution_plan_v3_work_unit_t *)calloc(
        header.work_unit_count, sizeof(*plan->work_units));
    if (plan->work_units == NULL) {
        v3_error(error_message, "distributed_plan_v3_parse_failed: work-unit allocation failed");
        free(payload);
        return 1;
    }
    memcpy(plan->work_units, payload + header.header_bytes,
        (size_t)header.work_unit_count * sizeof(*plan->work_units));
    for (uint32_t index = 0u; index < header.work_unit_count; index++) {
        const execution_plan_v3_work_unit_t *unit = &plan->work_units[index];
        if (unit->package_operation_index != header.package_operation_index ||
            unit->operation_id != header.operation_id || unit->partition_mode != header.partition_mode ||
            unit->dpu_id >= header.dpu_count || seen[unit->dpu_id] ||
            unit->output_elements == 0u || unit->contracted_elements == 0u) {
            v3_error(error_message, "hardware_profile_violation: distributed v3 work-unit identity is invalid");
            execution_plan_distributed_v3_free(plan);
            free(payload);
            return 1;
        }
        if (header.partition_mode == EXECUTION_PLAN_V3_PARTITION_OUTPUT) {
            if (unit->output_offset != cursor || unit->output_offset % 2u != 0u ||
                (uint64_t)unit->output_offset + unit->output_elements > header.output_elements ||
                unit->contracted_offset != 0u || unit->contracted_elements != header.contracted_elements ||
                (index + 1u < header.work_unit_count && unit->output_elements % 2u != 0u)) {
                v3_error(error_message, "hardware_profile_violation: distributed v3 output partition is not covered on aligned boundaries");
                execution_plan_distributed_v3_free(plan);
                free(payload);
                return 1;
            }
            cursor += unit->output_elements;
        } else {
            if (unit->output_offset != 0u || unit->output_elements != header.output_elements ||
                unit->contracted_offset != cursor ||
                (uint64_t)unit->contracted_offset + unit->contracted_elements > header.contracted_elements) {
                v3_error(error_message, "hardware_profile_violation: distributed v3 contracted partition is not covered");
                execution_plan_distributed_v3_free(plan);
                free(payload);
                return 1;
            }
            cursor += unit->contracted_elements;
        }
        seen[unit->dpu_id] = 1u;
    }
    if (cursor != (header.partition_mode == EXECUTION_PLAN_V3_PARTITION_OUTPUT
            ? header.output_elements : header.contracted_elements)) {
        v3_error(error_message, "hardware_profile_violation: distributed v3 partition coverage has a gap or overlap");
        execution_plan_distributed_v3_free(plan);
        free(payload);
        return 1;
    }
    for (uint32_t dpu_id = 0u; dpu_id < header.dpu_count; dpu_id++) {
        if (!seen[dpu_id]) {
            v3_error(error_message, "hardware_profile_violation: distributed v3 DPU IDs are not dense");
            execution_plan_distributed_v3_free(plan);
            free(payload);
            return 1;
        }
    }
    if (execution_plan_sha256_file(path, file_sha256) != 0 ||
        execution_plan_hash_file(path, &plan->sidecar_file_fnv1a64_runtime) != 0) {
        v3_error(error_message, "distributed_plan_v3_hash_failed: sidecar identity could not be computed");
        execution_plan_distributed_v3_free(plan);
        free(payload);
        return 1;
    }
    plan->header = header;
    plan->work_unit_count = header.work_unit_count;
    memcpy(plan->package_file_sha256, header.package_sha256, sizeof(plan->package_file_sha256));
    plan->file_path = strdup(path);
    plan->file_sha256 = strdup(file_sha256);
    if (plan->file_path == NULL || plan->file_sha256 == NULL) {
        v3_error(error_message, "distributed_plan_v3_hash_failed: sidecar identity allocation failed");
        execution_plan_distributed_v3_free(plan);
        free(payload);
        return 1;
    }
    free(payload);
    return 0;
}

void execution_plan_distributed_v3_free(execution_plan_distributed_v3_t *plan) {
    if (plan == NULL) return;
    free(plan->work_units);
    free(plan->file_sha256);
    free(plan->file_path);
    memset(plan, 0, sizeof(*plan));
}
