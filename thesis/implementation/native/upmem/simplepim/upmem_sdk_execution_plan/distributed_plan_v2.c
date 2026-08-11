#define _POSIX_C_SOURCE 200809L

#include "distributed_plan_v2.h"

#if RESIDENT_OPERATION_ABI_VERSION != RESIDENT_OPERATION_ABI_V2
#error "distributed_plan_v2.c must be compiled with resident operation ABI v2"
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void distributed_v2_error(char **message, const char *value) {
    if (message != NULL && *message == NULL) *message = strdup(value);
}

static int distributed_v2_read_file(const char *path, unsigned char **payload, size_t *length) {
    FILE *file = path == NULL ? NULL : fopen(path, "rb");
    long size;
    if (file == NULL || fseek(file, 0, SEEK_END) != 0) {
        if (file != NULL) fclose(file);
        return 1;
    }
    size = ftell(file);
    if (size < 0 || (unsigned long)size > 1024u * 1024u || fseek(file, 0, SEEK_SET) != 0) {
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

static int distributed_v2_allowed_dpu_count(uint32_t count) {
    return count == 1u || count == 2u || count == 4u;
}

static int distributed_v2_magic_valid(const char magic[8]) {
    return memcmp(magic, EXECUTION_PLAN_V2_MAGIC, 7u) == 0 && magic[7] == '\0';
}

static int distributed_v2_has_input(const resident_request_t *resident, uint32_t slot_id) {
    for (size_t index = 0u; index < resident->input_count; index++) {
        if (resident->inputs[index].slot_id == slot_id) return 1;
    }
    return 0;
}

int execution_plan_distributed_v2_load(
    const char *path,
    const unsigned char expected_package_sha256[32],
    const resident_request_t *resident,
    execution_plan_distributed_v2_t *plan,
    char **error_message
) {
    unsigned char *payload = NULL;
    size_t length = 0u;
    execution_plan_v2_header_t header;
    uint64_t expected_file_bytes;
    char file_sha256[65];
    char operation_sha256[65];
    int seen_dpu[EXECUTION_PLAN_V2_MAX_DPUS] = {0};
    uint32_t cursor = 0u;

    if (path == NULL || expected_package_sha256 == NULL || resident == NULL || plan == NULL) {
        distributed_v2_error(error_message, "distributed_plan_v2_parse_failed: missing sidecar inputs");
        return 1;
    }
    memset(plan, 0, sizeof(*plan));
    if (distributed_v2_read_file(path, &payload, &length) != 0 || length < sizeof(header)) {
        distributed_v2_error(error_message, "distributed_plan_v2_parse_failed: sidecar is unreadable or truncated");
        free(payload);
        return 1;
    }
    memcpy(&header, payload, sizeof(header));
    if (!distributed_v2_magic_valid(header.magic) || header.version != EXECUTION_PLAN_V2_VERSION ||
        header.header_bytes != sizeof(header) || header.record_bytes != sizeof(execution_plan_v2_work_unit_t) ||
        header.reserved0 != 0u || header.reserved1 != 0u || header.provider_count != EXECUTION_PLAN_V2_PROVIDER_COUNT ||
        (header.partition_mode != EXECUTION_PLAN_V2_PARTITION_OUTPUT &&
         header.partition_mode != EXECUTION_PLAN_V2_PARTITION_CONTRACTED) ||
        header.work_unit_count == 0u || header.work_unit_count > EXECUTION_PLAN_V2_MAX_WORK_UNITS ||
        !distributed_v2_allowed_dpu_count(header.dpu_count) || header.work_unit_count != header.dpu_count ||
        header.dpu_count > EXECUTION_PLAN_V2_MAX_DPUS || header.tasklets_per_dpu != NR_TASKLETS) {
        distributed_v2_error(error_message, "hardware_profile_violation: distributed v2 sidecar header is invalid");
        free(payload);
        return 1;
    }
    expected_file_bytes = (uint64_t)header.header_bytes +
        (uint64_t)header.work_unit_count * (uint64_t)header.record_bytes;
    if (expected_file_bytes != (uint64_t)length || memcmp(header.package_sha256, expected_package_sha256, 32u) != 0) {
        distributed_v2_error(error_message, "hardware_profile_violation: distributed v2 package binding or length is invalid");
        free(payload);
        return 1;
    }
    if (resident->header.operation_count != 1u || header.package_operation_index != 0u ||
        header.operation_id != header.package_operation_index || header.package_operation_index >= resident->header.operation_count) {
        distributed_v2_error(error_message, "hardware_profile_violation: distributed v2 sidecar must bind one package operation");
        free(payload);
        return 1;
    }
    {
        const resident_operation_t *operation = &resident->operations[header.package_operation_index];
        if (operation->kind != RESIDENT_OPERATION_CONTRACT || operation->mode != 0u ||
            operation->slot_out_real != header.output_slot || operation->output_elements != header.output_elements ||
            operation->args.output_elems != header.output_elements || operation->args.contracted_elems != header.contracted_elements ||
            operation->args.dpu_slice_offset != 0u || operation->args.dpu_slice_elements != header.output_elements ||
            operation->args.contracted_offset != 0u || operation->args.contracted_elements_slice != header.contracted_elements ||
            !distributed_v2_has_input(resident, operation->slot_a) || !distributed_v2_has_input(resident, operation->slot_b) ||
            header.output_slot >= resident->header.slot_count ||
            resident->slots[header.output_slot].capacity_elements < header.output_elements ||
            resident->slots[header.output_slot].element_count < header.output_elements ||
            resident->final_count != 1u || resident->final_outputs[0].slot_id != header.output_slot ||
            resident->final_outputs[0].elements != header.output_elements ||
            (uint64_t)header.output_elements * sizeof(float) > resident->final_outputs[0].raw_bytes ||
            ((uint64_t)header.output_elements * sizeof(float) + 7u) / 8u * 8u > resident->final_outputs[0].transfer_bytes ||
            execution_plan_sha256_bytes((const unsigned char *)operation, sizeof(*operation), operation_sha256) != 0) {
            distributed_v2_error(error_message, "hardware_profile_violation: distributed v2 operation identity or output bounds mismatch");
            free(payload);
            return 1;
        }
    }
    {
        unsigned char operation_digest[32];
        for (uint32_t index = 0u; index < 32u; index++) {
            unsigned int value;
            if (sscanf(operation_sha256 + index * 2u, "%02x", &value) != 1 || value > 0xffu) {
                distributed_v2_error(error_message, "distributed_plan_v2_hash_failed: operation identity encoding failed");
                free(payload);
                return 1;
            }
            operation_digest[index] = (unsigned char)value;
        }
        if (memcmp(header.operation_sha256, operation_digest, sizeof(operation_digest)) != 0) {
            distributed_v2_error(error_message, "hardware_profile_violation: distributed v2 operation SHA-256 binding mismatch");
            free(payload);
            return 1;
        }
    }
    memcpy(plan->work_units, payload + header.header_bytes,
        (size_t)header.work_unit_count * sizeof(plan->work_units[0]));
    for (uint32_t index = 0u; index < header.work_unit_count; index++) {
        const execution_plan_v2_work_unit_t *unit = &plan->work_units[index];
        if (unit->package_operation_index != header.package_operation_index ||
            unit->operation_id != header.operation_id || unit->partition_mode != header.partition_mode ||
            unit->dpu_id >= header.dpu_count || seen_dpu[unit->dpu_id] ||
            unit->output_elements == 0u || unit->contracted_elements == 0u) {
            distributed_v2_error(error_message, "hardware_profile_violation: distributed v2 work-unit range, mode, or identity is invalid");
            free(payload);
            return 1;
        }
        if (header.partition_mode == EXECUTION_PLAN_V2_PARTITION_OUTPUT) {
            if (unit->output_offset != cursor ||
                (uint64_t)unit->output_offset + (uint64_t)unit->output_elements > (uint64_t)header.output_elements ||
                unit->contracted_offset != 0u || unit->contracted_elements != header.contracted_elements) {
                distributed_v2_error(error_message, "hardware_profile_violation: distributed v2 output range has a gap, overlap, or wrong contracted coverage");
                free(payload);
                return 1;
            }
            cursor += unit->output_elements;
        } else {
            if (unit->output_offset != 0u || unit->output_elements != header.output_elements ||
                unit->contracted_offset != cursor ||
                (uint64_t)unit->contracted_offset + (uint64_t)unit->contracted_elements > (uint64_t)header.contracted_elements) {
                distributed_v2_error(error_message, "hardware_profile_violation: distributed v2 contracted range has a gap, overlap, or wrong output coverage");
                free(payload);
                return 1;
            }
            cursor += unit->contracted_elements;
        }
        seen_dpu[unit->dpu_id] = 1;
    }
    if (cursor != (header.partition_mode == EXECUTION_PLAN_V2_PARTITION_OUTPUT
            ? header.output_elements : header.contracted_elements)) {
        distributed_v2_error(error_message, "hardware_profile_violation: distributed v2 partition coverage has a gap or overlap");
        free(payload);
        return 1;
    }
    for (uint32_t dpu_id = 0u; dpu_id < header.dpu_count; dpu_id++) {
        if (!seen_dpu[dpu_id]) {
            distributed_v2_error(error_message, "hardware_profile_violation: distributed v2 DPU IDs are not dense");
            free(payload);
            return 1;
        }
    }
    if (execution_plan_sha256_file(path, file_sha256) != 0 ||
        execution_plan_hash_file(path, &plan->sidecar_file_fnv1a64_runtime) != 0) {
        distributed_v2_error(error_message, "distributed_plan_v2_hash_failed: sidecar identity could not be computed");
        free(payload);
        return 1;
    }
    plan->header = header;
    plan->work_unit_count = header.work_unit_count;
    memcpy(plan->package_file_sha256, header.package_sha256, sizeof(plan->package_file_sha256));
    plan->file_path = strdup(path);
    plan->file_sha256 = strdup(file_sha256);
    if (plan->file_path == NULL || plan->file_sha256 == NULL) {
        distributed_v2_error(error_message, "distributed_plan_v2_hash_failed: sidecar identity allocation failed");
        free(payload);
        execution_plan_distributed_v2_free(plan);
        return 1;
    }
    free(payload);
    return 0;
}

void execution_plan_distributed_v2_free(execution_plan_distributed_v2_t *plan) {
    if (plan == NULL) return;
    free(plan->file_sha256);
    free(plan->file_path);
    memset(plan, 0, sizeof(*plan));
}
