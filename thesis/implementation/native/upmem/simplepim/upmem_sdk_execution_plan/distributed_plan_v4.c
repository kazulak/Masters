#define _POSIX_C_SOURCE 200809L

#include "distributed_plan_v4.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void v4_error(char **message, const char *value) {
    if (message != NULL && *message == NULL) *message = strdup(value);
}

static uint32_t align8_u32(uint64_t value) {
    if (value > UINT32_MAX - 7u) return 0u;
    return (uint32_t)((value + 7u) & ~UINT64_C(7));
}

static int overlap(uint32_t left_offset, uint32_t left_bytes, uint32_t right_offset, uint32_t right_bytes) {
    return left_bytes != 0u && right_bytes != 0u &&
        left_offset < (uint64_t)right_offset + right_bytes &&
        right_offset < (uint64_t)left_offset + left_bytes;
}

static int extent_inside(uint64_t offset, uint64_t length, uint64_t limit) {
    return offset <= limit && length <= limit - offset;
}

static int ranges_overlap_u64(uint64_t left_offset, uint64_t left_length,
    uint64_t right_offset, uint64_t right_length) {
    if (left_length == 0u || right_length == 0u) return 0;
    if (left_offset <= right_offset) return right_offset - left_offset < left_length;
    return left_offset - right_offset < right_length;
}

int execution_plan_distributed_v4_validate(
    const execution_plan_v4_header_t *header,
    const execution_plan_v4_work_unit_t *work_units,
    uint32_t expected_dpus,
    uint32_t expected_tasklets,
    char **error_message
) {
    unsigned char seen[EXECUTION_PLAN_V4_MAX_DPUS] = {0};
    uint64_t covered = 0u;
    uint64_t expected_output;
    const uint32_t element_bytes = header != NULL &&
        header->numeric_mode == EXECUTION_PLAN_V4_NUMERIC_FLOAT32 ? 4u : 1u;
    if (header == NULL || work_units == NULL || expected_dpus == 0u ||
        expected_dpus > EXECUTION_PLAN_V4_MAX_DPUS || expected_tasklets == 0u ||
        expected_tasklets > EXECUTION_PLAN_V4_MAX_TASKLETS ||
        memcmp(header->magic, EXECUTION_PLAN_V4_MAGIC, 7u) != 0 || header->magic[7] != '\0' ||
        header->version != EXECUTION_PLAN_V4_VERSION || header->header_bytes != sizeof(*header) ||
        header->work_unit_count != expected_dpus || header->dpu_count != expected_dpus ||
        header->tasklets_per_dpu != expected_tasklets || header->record_bytes != sizeof(*work_units) ||
        header->reserved0 != 0u || header->reserved1 != 0u ||
        header->partition_mode != EXECUTION_PLAN_V4_PARTITION_OUTPUT_TILE ||
        (header->numeric_mode != EXECUTION_PLAN_V4_NUMERIC_FLOAT32 &&
         header->numeric_mode != EXECUTION_PLAN_V4_NUMERIC_HOST_PACKED_INT8) ||
        header->canonical_batch_count == 0u || header->canonical_m == 0u ||
        header->canonical_n == 0u || header->canonical_k == 0u ||
        header->canonical_k > EXECUTION_PLAN_V4_MAX_CONTRACTED ||
        header->request_output_elements == 0u ||
        header->request_output_elements > header->global_output_elements) {
        v4_error(error_message, "hardware_profile_violation: v4 header is outside the bounded profile");
        return 1;
    }
    if (header->canonical_batch_count > UINT64_MAX / header->canonical_m ||
        header->canonical_batch_count * header->canonical_m > UINT64_MAX / header->canonical_n) {
        v4_error(error_message, "hardware_profile_violation: v4 output dimension overflow");
        return 1;
    }
    expected_output = header->canonical_batch_count * header->canonical_m * header->canonical_n;
    if (header->global_output_elements != expected_output) {
        v4_error(error_message, "hardware_profile_violation: v4 output element count is inconsistent");
        return 1;
    }
    for (uint32_t index = 0u; index < header->work_unit_count; index++) {
        const execution_plan_v4_work_unit_t *unit = &work_units[index];
        uint64_t a_elements, b_elements, c_elements;
        uint32_t expected_a, expected_b, expected_c;
        if (unit->local_dpu_id >= expected_dpus || seen[unit->local_dpu_id] ||
            (index != 0u && unit->local_dpu_id <= work_units[index - 1u].local_dpu_id) ||
            unit->batch_index >= header->canonical_batch_count ||
            (unit->flags & ~EXECUTION_PLAN_V4_FLAG_ZERO_WORK) != 0u) {
            v4_error(error_message, "hardware_profile_violation: v4 local IDs or flags are invalid");
            return 1;
        }
        seen[unit->local_dpu_id] = 1u;
        if ((unit->flags & EXECUTION_PLAN_V4_FLAG_ZERO_WORK) != 0u) {
            if (unit->m_elements != 0u || unit->n_elements != 0u || unit->k_elements != 0u ||
                unit->a_transfer_bytes != 0u || unit->b_transfer_bytes != 0u || unit->c_transfer_bytes != 0u) {
                v4_error(error_message, "hardware_profile_violation: zero-work unit is not empty");
                return 1;
            }
            continue;
        }
        if (unit->m_elements == 0u || unit->n_elements == 0u || unit->k_elements == 0u ||
            !extent_inside(unit->k_offset, unit->k_elements, header->canonical_k) ||
            !extent_inside(unit->m_offset, unit->m_elements, header->canonical_m) ||
            !extent_inside(unit->n_offset, unit->n_elements, header->canonical_n) ||
            (uint64_t)unit->k_elements * 128u * 128u > 2147483647u) {
            v4_error(error_message, "hardware_profile_violation: v4 tile extents are invalid");
            return 1;
        }
        a_elements = (uint64_t)unit->m_elements * unit->k_elements;
        b_elements = (uint64_t)unit->k_elements * unit->n_elements;
        c_elements = (uint64_t)unit->m_elements * unit->n_elements;
        if (a_elements > UINT64_MAX / element_bytes || b_elements > UINT64_MAX / element_bytes ||
            c_elements > UINT64_MAX / sizeof(int32_t)) {
            v4_error(error_message, "hardware_profile_violation: v4 tile byte count overflow");
            return 1;
        }
        expected_a = align8_u32(a_elements * element_bytes);
        expected_b = align8_u32(b_elements * element_bytes);
        expected_c = align8_u32(c_elements * sizeof(int32_t));
        if (expected_a == 0u || expected_b == 0u || expected_c == 0u ||
            unit->a_transfer_bytes != expected_a || unit->b_transfer_bytes != expected_b ||
            unit->c_transfer_bytes != expected_c || unit->a_offset_bytes % 8u != 0u ||
            unit->b_offset_bytes % 8u != 0u || unit->c_offset_bytes % 8u != 0u ||
            (uint64_t)unit->a_offset_bytes + unit->a_transfer_bytes > EXECUTION_PLAN_V4_MRAM_POOL_BYTES ||
            (uint64_t)unit->b_offset_bytes + unit->b_transfer_bytes > EXECUTION_PLAN_V4_MRAM_POOL_BYTES ||
            (uint64_t)unit->c_offset_bytes + unit->c_transfer_bytes > EXECUTION_PLAN_V4_MRAM_POOL_BYTES ||
            overlap(unit->a_offset_bytes, unit->a_transfer_bytes, unit->b_offset_bytes, unit->b_transfer_bytes) ||
            overlap(unit->a_offset_bytes, unit->a_transfer_bytes, unit->c_offset_bytes, unit->c_transfer_bytes) ||
            overlap(unit->b_offset_bytes, unit->b_transfer_bytes, unit->c_offset_bytes, unit->c_transfer_bytes)) {
            v4_error(error_message, "hardware_profile_violation: v4 storage is unaligned, overlapping, or too large");
            return 1;
        }
        covered += (uint64_t)unit->m_elements * unit->n_elements;
        for (uint32_t prior = 0u; prior < index; prior++) {
            const execution_plan_v4_work_unit_t *other = &work_units[prior];
            if ((other->flags & EXECUTION_PLAN_V4_FLAG_ZERO_WORK) != 0u || other->batch_index != unit->batch_index) continue;
            if (ranges_overlap_u64(unit->m_offset, unit->m_elements,
                    other->m_offset, other->m_elements) &&
                ranges_overlap_u64(unit->n_offset, unit->n_elements,
                    other->n_offset, other->n_elements)) {
                v4_error(error_message, "hardware_profile_violation: v4 output tiles overlap");
                return 1;
            }
        }
    }
    for (uint32_t dpu_id = 0u; dpu_id < expected_dpus; dpu_id++) {
        if (!seen[dpu_id]) {
            v4_error(error_message, "hardware_profile_violation: v4 DPU IDs are not dense");
            return 1;
        }
    }
    if (covered != header->request_output_elements) {
        v4_error(error_message, "hardware_profile_violation: v4 request tiles do not cover the request output");
        return 1;
    }
    return 0;
}

const execution_plan_v4_work_unit_t *execution_plan_distributed_v4_work_unit_for_dpu(
    const execution_plan_v4_work_unit_t *work_units,
    uint32_t work_unit_count,
    uint32_t dpu_id
) {
    if (work_units == NULL) return NULL;
    for (uint32_t index = 0u; index < work_unit_count; index++) {
        if (work_units[index].local_dpu_id == dpu_id) return &work_units[index];
    }
    return NULL;
}
