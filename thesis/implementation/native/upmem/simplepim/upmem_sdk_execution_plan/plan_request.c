#define _POSIX_C_SOURCE 200809L

#include "plan_request.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* The resident manifest/package parser remains the single owner of the
 * existing package ABI.  This file only binds that package to the binary
 * schedule; it intentionally does not parse execution-plan JSON. */

static void request_error(char **message, const char *value) {
    if (message != NULL && *message == NULL) *message = strdup(value);
}

static int valid_slot(const resident_request_t *resident, uint32_t slot_id) {
    return resident != NULL && slot_id < resident->header.slot_count;
}

static uint32_t operation_bit(uint32_t operation_id) {
    return 1u << operation_id;
}

static int hex_digest_bytes(const char text[65], unsigned char digest[32]) {
    for (uint32_t index = 0u; index < 32u; index++) {
        unsigned int value;
        if (sscanf(text + index * 2u, "%02x", &value) != 1 || value > 0xffu) return 1;
        digest[index] = (unsigned char)value;
    }
    return 0;
}

static int validate_package(execution_plan_request_t *request, char **error_message) {
    resident_request_t *resident = &request->resident;
    const uint32_t operation_count = resident->header.operation_count;
    if (operation_count == 0u || operation_count > EXECUTION_PLAN_MAX_TASKS ||
        resident->final_count != 1u || resident->header.final_output_count != 1u ||
        resident->slot_flags == NULL || resident->operations == NULL || resident->slots == NULL) {
        request_error(error_message, "hardware_profile_violation: resident package is outside the bounded execution-plan profile");
        return 1;
    }
    if (strcmp(resident->quantization_mode, "none") != 0) {
        request_error(error_message, "hardware_profile_violation: execution-plan phase supports real float32 only");
        return 1;
    }
    for (uint32_t slot = 0u; slot < resident->header.slot_count; slot++) {
        if (resident->slots[slot].slot_id != slot) {
            request_error(error_message, "hardware_profile_violation: resident slot IDs are not dense");
            return 1;
        }
        request->producer_by_slot[slot] = -1;
    }
    for (uint32_t slot = resident->header.slot_count; slot < RESIDENT_MAX_SLOT_DESCRIPTORS; slot++) {
        request->producer_by_slot[slot] = -1;
    }
    for (uint32_t index = 0u; index < operation_count; index++) {
        const resident_operation_t *operation = &resident->operations[index];
        if (operation->kind != RESIDENT_OPERATION_CONTRACT || operation->mode != 0u ||
            operation->slot_c != RESIDENT_INVALID_SLOT || operation->slot_d != RESIDENT_INVALID_SLOT ||
            operation->slot_out_imag != RESIDENT_INVALID_SLOT ||
            !valid_slot(resident, operation->slot_a) || !valid_slot(resident, operation->slot_b) ||
            !valid_slot(resident, operation->slot_out_real) || operation->slot_a == operation->slot_b ||
            operation->slot_out_real == operation->slot_a || operation->slot_out_real == operation->slot_b) {
            request_error(error_message, "hardware_profile_violation: package contains an unsupported real operation");
            return 1;
        }
    }
    if (!valid_slot(resident, resident->final_outputs[0].slot_id) ||
        (resident->slot_flags[resident->final_outputs[0].slot_id] & RESIDENT_SLOT_FINAL_FLAG) == 0u) {
        request_error(error_message, "hardware_profile_violation: final output slot is not a package final slot");
        return 1;
    }
    return 0;
}

static int validate_schedule(execution_plan_request_t *request, char **error_message) {
    const uint32_t operation_count = request->resident.header.operation_count;
    const uint32_t wave_count = request->schedule.header.wave_count;
    int seen_operation[EXECUTION_PLAN_MAX_TASKS] = {0};
    int seen_package[EXECUTION_PLAN_MAX_TASKS] = {0};
    int dpu_seen[EXECUTION_PLAN_MAX_WAVES][EXECUTION_PLAN_MAX_DPUS] = {{0}};
    int wave_seen[EXECUTION_PLAN_MAX_WAVES] = {0};
    if (request->schedule.record_count != operation_count ||
        request->schedule.header.dpu_count < 1u || request->schedule.header.dpu_count > EXECUTION_PLAN_MAX_DPUS ||
        request->schedule.header.tasklets_per_dpu != 1u) {
        request_error(error_message, "hardware_profile_violation: schedule/package operation or resource counts differ");
        return 1;
    }
    for (uint32_t record_index = 0u; record_index < operation_count; record_index++) {
        const execution_plan_schedule_record_t *record = &request->schedule.records[record_index];
        if (record->package_operation_index >= operation_count || record->operation_id >= operation_count ||
            seen_package[record->package_operation_index] || seen_operation[record->operation_id] ||
            record->wave_index >= wave_count || record->dpu_id >= request->schedule.header.dpu_count ||
            (record->dependency_mask & ~((1u << operation_count) - 1u)) != 0u ||
            dpu_seen[record->wave_index][record->dpu_id]) {
            request_error(error_message, "hardware_profile_violation: schedule IDs, masks, waves, or DPU assignments are invalid");
            return 1;
        }
        if (record->input_slot_a != request->resident.operations[record->package_operation_index].slot_a ||
            record->input_slot_b != request->resident.operations[record->package_operation_index].slot_b ||
            record->output_slot != request->resident.operations[record->package_operation_index].slot_out_real) {
            request_error(error_message, "hardware_profile_violation: schedule slot binding does not match package operation");
            return 1;
        }
        seen_package[record->package_operation_index] = 1;
        seen_operation[record->operation_id] = 1;
        request->record_for_package[record->package_operation_index] = record_index;
        request->package_for_operation[record->operation_id] = record->package_operation_index;
        dpu_seen[record->wave_index][record->dpu_id] = 1;
        wave_seen[record->wave_index] = 1;
    }
    for (uint32_t wave = 0u; wave < wave_count; wave++) {
        if (!wave_seen[wave]) {
            request_error(error_message, "hardware_profile_violation: schedule waves are not dense");
            return 1;
        }
    }
    for (uint32_t slot = 0u; slot < request->resident.header.slot_count; slot++) {
        request->producer_by_slot[slot] = -1;
    }
    for (uint32_t wave = 0u; wave < wave_count; wave++) {
        int slot_read[RESIDENT_MAX_SLOT_DESCRIPTORS] = {0};
        int slot_written[RESIDENT_MAX_SLOT_DESCRIPTORS] = {0};
        for (uint32_t record_index = 0u; record_index < operation_count; record_index++) {
            const execution_plan_schedule_record_t *record = &request->schedule.records[record_index];
            const resident_operation_t *operation;
            const uint32_t inputs[2] = {record->input_slot_a, record->input_slot_b};
            uint32_t expected_mask = 0u;
            if (record->wave_index != wave) continue;
            operation = &request->resident.operations[record->package_operation_index];
            for (size_t input_index = 0u; input_index < 2u; input_index++) {
                const uint32_t slot_id = inputs[input_index];
                const int32_t producer = request->producer_by_slot[slot_id];
                if (slot_written[slot_id]) {
                    request_error(error_message, "hardware_profile_violation: same-wave operation reads a slot another operation writes");
                    return 1;
                }
                slot_read[slot_id] = 1;
                if (producer < 0) {
                    if ((request->resident.slot_flags[slot_id] & RESIDENT_SLOT_INITIAL_FLAG) == 0u) {
                        request_error(error_message, "hardware_profile_violation: operation input slot has no producer or initial binding");
                        return 1;
                    }
                } else {
                    const execution_plan_schedule_record_t *producer_record =
                        &request->schedule.records[request->record_for_package[(uint32_t)producer]];
                    expected_mask |= operation_bit(producer_record->operation_id);
                }
            }
            if (slot_read[operation->slot_out_real] || slot_written[operation->slot_out_real]) {
                request_error(error_message, "hardware_profile_violation: same-wave output slot aliases a live input or output");
                return 1;
            }
            if (record->dependency_mask != expected_mask) {
                request_error(error_message, "hardware_profile_violation: schedule dependency mask does not match live slot producers");
                return 1;
            }
            slot_written[operation->slot_out_real] = 1;
        }
        for (uint32_t record_index = 0u; record_index < operation_count; record_index++) {
            const execution_plan_schedule_record_t *record = &request->schedule.records[record_index];
            if (record->wave_index == wave) {
                request->producer_by_slot[record->output_slot] = (int32_t)record->package_operation_index;
            }
        }
    }
    if (request->producer_by_slot[request->resident.final_outputs[0].slot_id] < 0) {
        request_error(error_message, "hardware_profile_violation: final output has no operation producer");
        return 1;
    }
    for (uint32_t index = 0u; index < operation_count; index++) {
        if (!seen_operation[index] || !seen_package[index]) {
            request_error(error_message, "hardware_profile_violation: schedule does not cover every operation exactly once");
            return 1;
        }
    }
    return 0;
}

int execution_plan_request_load(
    const char *resident_manifest_path,
    const char *schedule_path,
    uint32_t warmup_repetitions,
    uint32_t measured_repetitions,
    execution_plan_request_t *request,
    char **error_message
) {
    if (request == NULL || resident_manifest_path == NULL || schedule_path == NULL ||
        warmup_repetitions > 1u || measured_repetitions == 0u || measured_repetitions > EXECUTION_PLAN_MAX_REPETITIONS) {
        request_error(error_message, "hardware_profile_violation: invalid execution-plan request arguments");
        return 1;
    }
    memset(request, 0, sizeof(*request));
    request->warmup_repetitions = warmup_repetitions;
    request->measured_repetitions = measured_repetitions;
    request->resident_manifest_path = strdup(resident_manifest_path);
    if (request->resident_manifest_path == NULL || resident_request_load_execution_plan(resident_manifest_path, &request->resident, error_message) != 0) {
        if (error_message != NULL && *error_message == NULL) request_error(error_message, "resident_package_parse_failed: resident request could not be loaded");
        goto failed;
    }
    if (execution_plan_sha256_file(request->resident.package_path, request->actual_package_file_sha256) != 0 ||
        execution_plan_hash_file(request->resident.package_path, &request->package_file_fnv1a64_runtime) != 0) {
        request_error(error_message, "package_hash_failed: resident package identity could not be computed");
        goto failed;
    }
    {
        unsigned char digest[32];
        if (hex_digest_bytes(request->actual_package_file_sha256, digest) != 0 ||
            execution_plan_schedule_load(schedule_path, digest, &request->schedule, error_message) != 0) goto failed;
    }
    if (request->resident.requested_dpus != 1u &&
        request->resident.requested_dpus != request->schedule.header.dpu_count) {
        request_error(error_message, "hardware_profile_violation: resident request DPU count conflicts with schedule");
        goto failed;
    }
    {
        if (memcmp(request->schedule.package_file_sha256, (unsigned char[32]){0}, 32u) == 0) {
            request_error(error_message, "hardware_profile_violation: schedule package binding is empty");
            goto failed;
        }
        unsigned char digest[32];
        if (hex_digest_bytes(request->actual_package_file_sha256, digest) != 0) {
            request_error(error_message, "package_hash_failed: package SHA-256 encoding is invalid");
            goto failed;
        }
        if (memcmp(request->schedule.package_file_sha256, digest, sizeof(digest)) != 0) {
            request_error(error_message, "hardware_profile_violation: schedule package SHA-256 does not match resident package");
            goto failed;
        }
    }
    if (validate_package(request, error_message) != 0 || validate_schedule(request, error_message) != 0) goto failed;
    return 0;
failed:
    execution_plan_request_free(request);
    return 1;
}

int execution_plan_request_validate(execution_plan_request_t *request, char **error_message) {
    return request == NULL || validate_package(request, error_message) != 0 || validate_schedule(request, error_message) != 0;
}

void execution_plan_request_free(execution_plan_request_t *request) {
    if (request == NULL) return;
    free(request->resident_manifest_path);
    execution_plan_schedule_free(&request->schedule);
    resident_request_free(&request->resident);
    memset(request, 0, sizeof(*request));
}
