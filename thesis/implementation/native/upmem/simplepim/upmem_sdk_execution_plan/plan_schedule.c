#define _POSIX_C_SOURCE 200809L

#include "plan_schedule.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#define FNV1A64_OFFSET 14695981039346656037ULL
#define FNV1A64_PRIME 1099511628211ULL

static void schedule_error(char **message, const char *value) {
    if (message != NULL && *message == NULL) *message = strdup(value);
}

static int read_file(const char *path, unsigned char **data, size_t *length) {
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
    *data = (unsigned char *)malloc((size_t)size);
    if (*data == NULL && size != 0) {
        fclose(file);
        return 1;
    }
    if (fread(*data, 1u, (size_t)size, file) != (size_t)size || fclose(file) != 0) {
        free(*data);
        *data = NULL;
        return 1;
    }
    *length = (size_t)size;
    return 0;
}

uint64_t execution_plan_fnv1a64(const unsigned char *data, size_t length) {
    uint64_t value = FNV1A64_OFFSET;
    for (size_t index = 0; index < length; index++) {
        value ^= data[index];
        value *= FNV1A64_PRIME;
    }
    return value;
}

int execution_plan_hash_file(const char *path, uint64_t *hash) {
    unsigned char buffer[4096];
    FILE *file = path == NULL ? NULL : fopen(path, "rb");
    uint64_t value = FNV1A64_OFFSET;
    size_t bytes;
    if (file == NULL || hash == NULL) {
        if (file != NULL) fclose(file);
        return 1;
    }
    while ((bytes = fread(buffer, 1u, sizeof(buffer), file)) != 0u) {
        for (size_t index = 0; index < bytes; index++) {
            value ^= buffer[index];
            value *= FNV1A64_PRIME;
        }
    }
    {
        int failed = ferror(file) != 0 || fclose(file) != 0;
        if (failed) return 1;
    }
    *hash = value;
    return 0;
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

int execution_plan_schedule_load(
    const char *path,
    const unsigned char expected_package_sha256[32],
    execution_plan_schedule_t *schedule,
    char **error_message
) {
    unsigned char *payload = NULL;
    size_t length = 0u;
    execution_plan_schedule_header_t header;
    uint64_t expected_file_bytes;
    char file_sha256[65];
    if (schedule == NULL || path == NULL) {
        schedule_error(error_message, "schedule_parse_failed: missing schedule");
        return 1;
    }
    memset(schedule, 0, sizeof(*schedule));
    if (read_file(path, &payload, &length) != 0 || length < sizeof(header)) {
        schedule_error(error_message, "schedule_parse_failed: schedule is unreadable or truncated");
        free(payload);
        return 1;
    }
    memcpy(&header, payload, sizeof(header));
    if (memcmp(header.magic, EXECUTION_PLAN_SCHEDULE_MAGIC, 8u) != 0 ||
        header.version != EXECUTION_PLAN_SCHEDULE_VERSION || header.header_bytes != sizeof(header) ||
        header.record_bytes != sizeof(execution_plan_schedule_record_t) || header.reserved0 != 0u || header.reserved1 != 0u ||
        header.provider_count != 3u ||
        header.operation_count == 0u || header.operation_count > EXECUTION_PLAN_MAX_TASKS ||
        header.wave_count == 0u || header.wave_count > EXECUTION_PLAN_MAX_WAVES ||
        header.dpu_count == 0u || header.dpu_count > EXECUTION_PLAN_V1_MAX_DPUS ||
        header.tasklets_per_dpu != 1u) {
        schedule_error(error_message, "hardware_profile_violation: schedule header is invalid");
        free(payload);
        return 1;
    }
    if ((uint64_t)header.operation_count * sizeof(execution_plan_schedule_record_t) > UINT32_MAX - header.header_bytes) {
        schedule_error(error_message, "schedule_parse_failed: schedule byte count overflow");
        free(payload);
        return 1;
    }
    expected_file_bytes = (uint64_t)header.header_bytes + (uint64_t)header.operation_count * header.record_bytes;
    if (expected_file_bytes != length) {
        schedule_error(error_message, "schedule_parse_failed: schedule file length mismatch");
        free(payload);
        return 1;
    }
    if (expected_package_sha256 == NULL || memcmp(header.package_sha256, expected_package_sha256, 32u) != 0) {
        schedule_error(error_message, "hardware_profile_violation: schedule package SHA-256 binding mismatch");
        free(payload);
        return 1;
    }
    if (execution_plan_sha256_file(path, file_sha256) != 0) {
        schedule_error(error_message, "schedule_hash_failed: schedule SHA-256 could not be computed");
        free(payload);
        return 1;
    }
    memcpy(schedule->records, payload + header.header_bytes, header.operation_count * sizeof(schedule->records[0]));
    {
        int seen_package[EXECUTION_PLAN_MAX_TASKS] = {0};
        int seen_operation[EXECUTION_PLAN_MAX_TASKS] = {0};
        uint32_t wave_by_operation[EXECUTION_PLAN_MAX_TASKS] = {0};
        for (uint32_t index = 0u; index < header.operation_count; index++) {
            const execution_plan_schedule_record_t *record = &schedule->records[index];
            if (record->package_operation_index >= header.operation_count ||
                record->operation_id >= header.operation_count ||
                record->wave_index >= header.wave_count || record->dpu_id >= header.dpu_count ||
                (record->dependency_mask & ~((1u << header.operation_count) - 1u)) != 0u ||
                seen_package[record->package_operation_index] || seen_operation[record->operation_id]) {
                schedule_error(error_message, "hardware_profile_violation: schedule record IDs or ranges are invalid");
                free(payload);
                return 1;
            }
            seen_package[record->package_operation_index] = 1;
            seen_operation[record->operation_id] = 1;
            wave_by_operation[record->operation_id] = record->wave_index;
        }
        for (uint32_t index = 0u; index < header.operation_count; index++) {
            const execution_plan_schedule_record_t *record = &schedule->records[index];
            for (uint32_t dependency = 0u; dependency < header.operation_count; dependency++) {
                if ((record->dependency_mask & (1u << dependency)) != 0u &&
                    wave_by_operation[dependency] >= record->wave_index) {
                    schedule_error(error_message, "hardware_profile_violation: schedule dependency is not in an earlier wave");
                    free(payload);
                    return 1;
                }
            }
        }
    }
    schedule->header = header;
    schedule->record_count = header.operation_count;
    memcpy(schedule->package_file_sha256, header.package_sha256, sizeof(schedule->package_file_sha256));
    schedule->file_path = strdup(path);
    schedule->file_sha256 = strdup(file_sha256);
    if (schedule->file_path == NULL || schedule->file_sha256 == NULL ||
        execution_plan_hash_file(path, &schedule->schedule_file_fnv1a64_runtime) != 0) {
        schedule_error(error_message, "schedule_hash_failed: schedule identity could not be computed");
        free(payload);
        execution_plan_schedule_free(schedule);
        return 1;
    }
    free(payload);
    return 0;
}

void execution_plan_schedule_free(execution_plan_schedule_t *schedule) {
    if (schedule == NULL) return;
    free(schedule->file_sha256);
    free(schedule->file_path);
    memset(schedule, 0, sizeof(*schedule));
}
