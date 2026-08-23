#define _POSIX_C_SOURCE 200809L

#include "plan.h"

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
/* Small SHA-256 implementation used only for evidence identity. It avoids a
 * host OpenSSL dependency on the ETH machine. */
typedef struct {
    uint32_t state[8];
    uint64_t bit_count;
    unsigned char block[64];
    size_t block_length;
} execution_plan_sha256_t;

static uint32_t rotr32(uint32_t value, uint32_t amount) {
    return (value >> amount) | (value << (32u - amount));
}

static void sha256_block(execution_plan_sha256_t *ctx, const unsigned char block[64]) {
    static const uint32_t constants[64] = {
        0x428a2f98u,0x71374491u,0xb5c0fbcfu,0xe9b5dba5u,0x3956c25bu,0x59f111f1u,0x923f82a4u,0xab1c5ed5u,
        0xd807aa98u,0x12835b01u,0x243185beu,0x550c7dc3u,0x72be5d74u,0x80deb1feu,0x9bdc06a7u,0xc19bf174u,
        0xe49b69c1u,0xefbe4786u,0x0fc19dc6u,0x240ca1ccu,0x2de92c6fu,0x4a7484aau,0x5cb0a9dcu,0x76f988dau,
        0x983e5152u,0xa831c66du,0xb00327c8u,0xbf597fc7u,0xc6e00bf3u,0xd5a79147u,0x06ca6351u,0x14292967u,
        0x27b70a85u,0x2e1b2138u,0x4d2c6dfcu,0x53380d13u,0x650a7354u,0x766a0abbu,0x81c2c92eu,0x92722c85u,
        0xa2bfe8a1u,0xa81a664bu,0xc24b8b70u,0xc76c51a3u,0xd192e819u,0xd6990624u,0xf40e3585u,0x106aa070u,
        0x19a4c116u,0x1e376c08u,0x2748774cu,0x34b0bcb5u,0x391c0cb3u,0x4ed8aa4au,0x5b9cca4fu,0x682e6ff3u,
        0x748f82eeu,0x78a5636fu,0x84c87814u,0x8cc70208u,0x90befffau,0xa4506cebu,0xbef9a3f7u,0xc67178f2u
    };
    uint32_t words[64];
    uint32_t a, b, c, d, e, f, g, h;
    for (uint32_t index = 0; index < 16u; index++) {
        words[index] = ((uint32_t)block[index * 4u] << 24u) |
            ((uint32_t)block[index * 4u + 1u] << 16u) |
            ((uint32_t)block[index * 4u + 2u] << 8u) | (uint32_t)block[index * 4u + 3u];
    }
    for (uint32_t index = 16u; index < 64u; index++) {
        uint32_t s0 = rotr32(words[index - 15u], 7u) ^ rotr32(words[index - 15u], 18u) ^ (words[index - 15u] >> 3u);
        uint32_t s1 = rotr32(words[index - 2u], 17u) ^ rotr32(words[index - 2u], 19u) ^ (words[index - 2u] >> 10u);
        words[index] = words[index - 16u] + s0 + words[index - 7u] + s1;
    }
    a = ctx->state[0]; b = ctx->state[1]; c = ctx->state[2]; d = ctx->state[3];
    e = ctx->state[4]; f = ctx->state[5]; g = ctx->state[6]; h = ctx->state[7];
    for (uint32_t index = 0; index < 64u; index++) {
        uint32_t s1 = rotr32(e, 6u) ^ rotr32(e, 11u) ^ rotr32(e, 25u);
        uint32_t choose = (e & f) ^ ((~e) & g);
        uint32_t temp1 = h + s1 + choose + constants[index] + words[index];
        uint32_t s0 = rotr32(a, 2u) ^ rotr32(a, 13u) ^ rotr32(a, 22u);
        uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
        uint32_t temp2 = s0 + majority;
        h = g; g = f; f = e; e = d + temp1; d = c; c = b; b = a; a = temp1 + temp2;
    }
    ctx->state[0] += a; ctx->state[1] += b; ctx->state[2] += c; ctx->state[3] += d;
    ctx->state[4] += e; ctx->state[5] += f; ctx->state[6] += g; ctx->state[7] += h;
}

static void sha256_init(execution_plan_sha256_t *ctx) {
    *ctx = (execution_plan_sha256_t){{
        0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
        0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u
    }, 0u, {0}, 0u};
}

static void sha256_update(execution_plan_sha256_t *ctx, const unsigned char *data, size_t length) {
    while (length != 0u) {
        size_t available = sizeof(ctx->block) - ctx->block_length;
        size_t take = length < available ? length : available;
        memcpy(ctx->block + ctx->block_length, data, take);
        ctx->block_length += take;
        ctx->bit_count += (uint64_t)take * 8u;
        data += take;
        length -= take;
        if (ctx->block_length == sizeof(ctx->block)) {
            sha256_block(ctx, ctx->block);
            ctx->block_length = 0u;
        }
    }
}

static void sha256_final(execution_plan_sha256_t *ctx, unsigned char digest[32]) {
    uint64_t bits = ctx->bit_count;
    unsigned char length_bytes[8];
    for (int index = 0; index < 8; index++) length_bytes[7 - index] = (unsigned char)(bits >> (index * 8));
    unsigned char one = 0x80u;
    sha256_update(ctx, &one, 1u);
    while (ctx->block_length != 56u) {
        unsigned char zero = 0u;
        sha256_update(ctx, &zero, 1u);
    }
    sha256_update(ctx, length_bytes, sizeof(length_bytes));
    for (uint32_t index = 0; index < 8u; index++) {
        digest[index * 4u] = (unsigned char)(ctx->state[index] >> 24u);
        digest[index * 4u + 1u] = (unsigned char)(ctx->state[index] >> 16u);
        digest[index * 4u + 2u] = (unsigned char)(ctx->state[index] >> 8u);
        digest[index * 4u + 3u] = (unsigned char)ctx->state[index];
    }
}

int execution_plan_sha256_file(const char *path, char output[65]) {
    execution_plan_sha256_t ctx;
    unsigned char buffer[4096];
    FILE *file = path == NULL ? NULL : fopen(path, "rb");
    size_t bytes;
    unsigned char digest[32];
    if (file == NULL || output == NULL) {
        if (file != NULL) fclose(file);
        return 1;
    }
    sha256_init(&ctx);
    while ((bytes = fread(buffer, 1u, sizeof(buffer), file)) != 0u) sha256_update(&ctx, buffer, bytes);
    if (ferror(file) != 0 || fclose(file) != 0) return 1;
    sha256_final(&ctx, digest);
    for (uint32_t index = 0; index < 32u; index++) snprintf(output + index * 2u, 3u, "%02x", digest[index]);
    output[64] = '\0';
    return 0;
}

int execution_plan_sha256_bytes(const unsigned char *data, size_t length, char output[65]) {
    execution_plan_sha256_t ctx;
    unsigned char digest[32];
    if (data == NULL || output == NULL) return 1;
    sha256_init(&ctx);
    sha256_update(&ctx, data, length);
    sha256_final(&ctx, digest);
    for (uint32_t index = 0; index < 32u; index++) snprintf(output + index * 2u, 3u, "%02x", digest[index]);
    output[64] = '\0';
    return 0;
}
