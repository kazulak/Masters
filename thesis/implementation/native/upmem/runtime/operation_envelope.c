#define _POSIX_C_SOURCE 200809L

#include "operation_envelope.h"

#include "plan.h"

#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#ifndef PATH_MAX
#define PATH_MAX 4096
#endif

static void v4_error(char **message, const char *value) {
    if (message != NULL && *message == NULL) *message = strdup(value);
}

static uint32_t read_le32(const unsigned char *data) {
    return (uint32_t)data[0] |
        ((uint32_t)data[1] << 8u) |
        ((uint32_t)data[2] << 16u) |
        ((uint32_t)data[3] << 24u);
}

static uint64_t read_le64(const unsigned char *data) {
    uint64_t value = 0u;
    for (uint32_t index = 0u; index < 8u; index++) {
        value |= (uint64_t)data[index] << (index * 8u);
    }
    return value;
}

static int checked_add_u64(uint64_t left, uint64_t right, uint64_t *result) {
    if (left > UINT64_MAX - right) return 1;
    *result = left + right;
    return 0;
}

static int checked_mul_u64(uint64_t left, uint64_t right, uint64_t *result) {
    if (left != 0u && right > UINT64_MAX / left) return 1;
    *result = left * right;
    return 0;
}

static int extent_inside(uint64_t offset, uint64_t length, uint64_t limit) {
    return offset <= limit && length <= limit - offset;
}

static int digest_text(const char *text, unsigned char digest[32]) {
    if (text == NULL || strlen(text) != 64u) return 1;
    for (uint32_t index = 0u; index < 32u; index++) {
        const char high = text[index * 2u];
        const char low = text[index * 2u + 1u];
        unsigned char value = 0u;
        if (high >= '0' && high <= '9') value = (unsigned char)((high - '0') << 4u);
        else if (high >= 'a' && high <= 'f') value = (unsigned char)((high - 'a' + 10) << 4u);
        else return 1;
        if (low >= '0' && low <= '9') value |= (unsigned char)(low - '0');
        else if (low >= 'a' && low <= 'f') value |= (unsigned char)(low - 'a' + 10);
        else return 1;
        digest[index] = value;
    }
    return 0;
}

static int digest_bytes_matches(
    const unsigned char *data,
    size_t length,
    const unsigned char expected[32]
) {
    char actual_hex[65];
    unsigned char actual[32];
    return execution_plan_sha256_bytes(data, length, actual_hex) == 0 &&
        digest_text(actual_hex, actual) == 0 &&
        memcmp(actual, expected, sizeof(actual)) == 0;
}

typedef struct {
    uint32_t state[8];
    uint64_t bit_count;
    unsigned char block[64];
    size_t block_length;
} envelope_sha256_t;

static uint32_t rotr32(uint32_t value, uint32_t amount) {
    return (value >> amount) | (value << (32u - amount));
}

static void envelope_sha256_block(envelope_sha256_t *context, const unsigned char block[64]) {
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
    for (uint32_t index = 0u; index < 16u; index++) {
        words[index] = ((uint32_t)block[index * 4u] << 24u) |
            ((uint32_t)block[index * 4u + 1u] << 16u) |
            ((uint32_t)block[index * 4u + 2u] << 8u) |
            (uint32_t)block[index * 4u + 3u];
    }
    for (uint32_t index = 16u; index < 64u; index++) {
        uint32_t s0 = rotr32(words[index - 15u], 7u) ^
            rotr32(words[index - 15u], 18u) ^ (words[index - 15u] >> 3u);
        uint32_t s1 = rotr32(words[index - 2u], 17u) ^
            rotr32(words[index - 2u], 19u) ^ (words[index - 2u] >> 10u);
        words[index] = words[index - 16u] + s0 + words[index - 7u] + s1;
    }
    a = context->state[0]; b = context->state[1]; c = context->state[2]; d = context->state[3];
    e = context->state[4]; f = context->state[5]; g = context->state[6]; h = context->state[7];
    for (uint32_t index = 0u; index < 64u; index++) {
        uint32_t s1 = rotr32(e, 6u) ^ rotr32(e, 11u) ^ rotr32(e, 25u);
        uint32_t choose = (e & f) ^ ((~e) & g);
        uint32_t temp1 = h + s1 + choose + constants[index] + words[index];
        uint32_t s0 = rotr32(a, 2u) ^ rotr32(a, 13u) ^ rotr32(a, 22u);
        uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
        uint32_t temp2 = s0 + majority;
        h = g; g = f; f = e; e = d + temp1;
        d = c; c = b; b = a; a = temp1 + temp2;
    }
    context->state[0] += a; context->state[1] += b;
    context->state[2] += c; context->state[3] += d;
    context->state[4] += e; context->state[5] += f;
    context->state[6] += g; context->state[7] += h;
}

static void envelope_sha256_init(envelope_sha256_t *context) {
    *context = (envelope_sha256_t){
        {0x6a09e667u,0xbb67ae85u,0x3c6ef372u,0xa54ff53au,
         0x510e527fu,0x9b05688cu,0x1f83d9abu,0x5be0cd19u},
        0u, {0}, 0u
    };
}

static void envelope_sha256_update(
    envelope_sha256_t *context,
    const unsigned char *data,
    size_t length
) {
    while (length != 0u) {
        size_t available = sizeof(context->block) - context->block_length;
        size_t take = length < available ? length : available;
        memcpy(context->block + context->block_length, data, take);
        context->block_length += take;
        context->bit_count += (uint64_t)take * 8u;
        data += take;
        length -= take;
        if (context->block_length == sizeof(context->block)) {
            envelope_sha256_block(context, context->block);
            context->block_length = 0u;
        }
    }
}

static void envelope_sha256_final(envelope_sha256_t *context, unsigned char digest[32]) {
    uint64_t bit_count = context->bit_count;
    unsigned char length_bytes[8];
    for (uint32_t index = 0u; index < 8u; index++) {
        length_bytes[7u - index] = (unsigned char)(bit_count >> (index * 8u));
    }
    {
        const unsigned char one = 0x80u;
        envelope_sha256_update(context, &one, 1u);
    }
    while (context->block_length != 56u) {
        const unsigned char zero = 0u;
        envelope_sha256_update(context, &zero, 1u);
    }
    envelope_sha256_update(context, length_bytes, sizeof(length_bytes));
    for (uint32_t index = 0u; index < 8u; index++) {
        digest[index * 4u] = (unsigned char)(context->state[index] >> 24u);
        digest[index * 4u + 1u] = (unsigned char)(context->state[index] >> 16u);
        digest[index * 4u + 2u] = (unsigned char)(context->state[index] >> 8u);
        digest[index * 4u + 3u] = (unsigned char)context->state[index];
    }
}

static void digest_envelope_with_zeroed_header_digest(
    const unsigned char *data,
    size_t length,
    unsigned char digest[32]
) {
    static const unsigned char zeros[32] = {0};
    envelope_sha256_t context;
    envelope_sha256_init(&context);
    envelope_sha256_update(&context, data, 64u);
    envelope_sha256_update(&context, zeros, sizeof(zeros));
    envelope_sha256_update(&context, data + 96u, length - 96u);
    envelope_sha256_final(&context, digest);
}

static int relative_path_is_safe(const char *path) {
    const char *cursor;
    if (path == NULL || path[0] == '\0' || path[0] == '/' || path[0] == '\\') return 0;
    cursor = path;
    while (*cursor != '\0') {
        const char *start = cursor;
        while (*cursor != '\0' && *cursor != '/') cursor++;
        if ((size_t)(cursor - start) == 2u && start[0] == '.' && start[1] == '.') return 0;
        while (*cursor == '/') cursor++;
    }
    return 1;
}

static int path_inside_root(const char *root, const char *candidate) {
    size_t root_length = strlen(root);
    return strncmp(root, candidate, root_length) == 0 &&
        (candidate[root_length] == '\0' || candidate[root_length] == '/');
}

static int resolve_envelope_path(
    const char *session_root,
    const char *relative_path,
    char output[PATH_MAX]
) {
    char root[PATH_MAX];
    char joined[PATH_MAX];
    char resolved[PATH_MAX];
    struct stat info;
    if (realpath(session_root, root) == NULL || stat(root, &info) != 0 || !S_ISDIR(info.st_mode) ||
        !relative_path_is_safe(relative_path) ||
        snprintf(joined, sizeof(joined), "%s/%s", root, relative_path) >= (int)sizeof(joined) ||
        realpath(joined, resolved) == NULL || !path_inside_root(root, resolved) ||
        stat(resolved, &info) != 0 || !S_ISREG(info.st_mode) ||
        snprintf(output, PATH_MAX, "%s", resolved) >= PATH_MAX) return 1;
    return 0;
}

static int operation_descriptor_at(
    const execution_plan_v4_operation_envelope_t *operation,
    uint32_t index,
    const unsigned char **descriptor,
    char **error_message
) {
    uint64_t descriptor_offset;
    uint64_t index_offset;
    if (operation == NULL || operation->mapping == NULL || index >= operation->descriptor_count ||
        checked_mul_u64(index, EXECUTION_PLAN_V4_OPERATION_DESCRIPTOR_BYTES, &index_offset) != 0 ||
        checked_add_u64(EXECUTION_PLAN_V4_OPERATION_HEADER_BYTES, index_offset, &descriptor_offset) != 0 ||
        descriptor_offset > operation->file_size ||
        EXECUTION_PLAN_V4_OPERATION_DESCRIPTOR_BYTES > operation->file_size - (size_t)descriptor_offset) {
        v4_error(error_message, "operation_envelope_failed: descriptor index is outside the envelope");
        return 1;
    }
    *descriptor = operation->mapping + (size_t)descriptor_offset;
    return 0;
}

static void fill_operation_descriptor(
    const unsigned char *raw,
    execution_plan_v4_embedded_request_t *descriptor
) {
    descriptor->request_sequence = read_le64(raw);
    descriptor->manifest_bytes = (size_t)read_le64(raw + 16u);
    descriptor->sidecar_bytes = (size_t)read_le64(raw + 32u);
    descriptor->payload_bytes = (size_t)read_le64(raw + 48u);
    descriptor->work_unit_count = read_le32(raw + 56u);
    descriptor->request_output_elements = read_le64(raw + 64u);
    descriptor->manifest = NULL;
    descriptor->sidecar = NULL;
    descriptor->payload = NULL;
    memcpy(descriptor->manifest_sha256, raw + 72u, 32u);
    memcpy(descriptor->sidecar_sha256, raw + 104u, 32u);
    memcpy(descriptor->payload_sha256, raw + 136u, 32u);
}

static int validate_envelope(
    const char *session_root,
    execution_plan_v4_operation_envelope_t *operation,
    uint32_t expected_dpus,
    uint32_t expected_tasklets,
    char **error_message
) {
    const unsigned char *data = operation->mapping;
    const uint64_t file_size = (uint64_t)operation->file_size;
    uint32_t descriptor_count;
    uint32_t descriptor_bytes;
    uint64_t descriptors_offset;
    uint64_t body_offset;
    uint64_t total_bytes;
    uint64_t operation_sequence;
    uint64_t descriptor_table_bytes;
    uint64_t expected_body_offset;
    uint64_t cursor;
    unsigned char envelope_digest[32];
    uint64_t previous_sequence = 0u;

    if (operation->file_size < EXECUTION_PLAN_V4_OPERATION_HEADER_BYTES ||
        memcmp(data, EXECUTION_PLAN_V4_OPERATION_MAGIC, 8u) != 0 ||
        read_le32(data + 8u) != EXECUTION_PLAN_V4_OPERATION_VERSION ||
        read_le32(data + 12u) != EXECUTION_PLAN_V4_OPERATION_HEADER_BYTES) {
        v4_error(error_message, "operation_envelope_failed: UPOENV2 header or descriptor bounds are invalid");
        return 1;
    }
    descriptor_count = read_le32(data + 16u);
    descriptor_bytes = read_le32(data + 20u);
    descriptors_offset = read_le64(data + 32u);
    body_offset = read_le64(data + 40u);
    total_bytes = read_le64(data + 48u);
    operation_sequence = read_le64(data + 56u);
    if (descriptor_count == 0u || descriptor_bytes != EXECUTION_PLAN_V4_OPERATION_DESCRIPTOR_BYTES ||
        read_le32(data + 24u) != 0u || read_le32(data + 28u) != 0u ||
        descriptors_offset != EXECUTION_PLAN_V4_OPERATION_HEADER_BYTES ||
        checked_mul_u64(descriptor_count, descriptor_bytes, &descriptor_table_bytes) != 0 ||
        checked_add_u64(descriptors_offset, descriptor_table_bytes, &expected_body_offset) != 0 ||
        body_offset != expected_body_offset || total_bytes != file_size ||
        !extent_inside(descriptors_offset, descriptor_table_bytes, file_size) ||
        !extent_inside(body_offset, 0u, file_size)) {
        v4_error(error_message, "operation_envelope_failed: UPOENV2 header or descriptor bounds are invalid");
        return 1;
    }
    digest_envelope_with_zeroed_header_digest(data, operation->file_size, envelope_digest);
    if (memcmp(envelope_digest, data + 64u, sizeof(envelope_digest)) != 0) {
        v4_error(error_message, "operation_envelope_failed: envelope SHA-256 mismatch");
        return 1;
    }
    operation->descriptor_count = descriptor_count;
    operation->operation_sequence = operation_sequence;
    memcpy(operation->digest, data + 64u, sizeof(operation->digest));
    operation->requests = (execution_plan_v4_request_t *)calloc(
        descriptor_count, sizeof(*operation->requests));
    if (operation->requests == NULL) {
        v4_error(error_message, "operation_envelope_failed: prepared request allocation failed");
        return 1;
    }
    cursor = body_offset;
    for (uint32_t index = 0u; index < descriptor_count; index++) {
        const unsigned char *raw_descriptor;
        uint64_t request_sequence;
        uint64_t manifest_offset;
        uint64_t manifest_bytes;
        uint64_t sidecar_offset;
        uint64_t sidecar_bytes;
        uint64_t payload_offset;
        uint64_t payload_bytes;
        uint64_t next_cursor;
        uint64_t index_offset;
        unsigned char descriptor_copy[EXECUTION_PLAN_V4_OPERATION_DESCRIPTOR_BYTES];
        execution_plan_v4_embedded_request_t embedded = {0};
        if (checked_mul_u64(index, descriptor_bytes, &index_offset) != 0 ||
            checked_add_u64(descriptors_offset, index_offset, &index_offset) != 0 ||
            index_offset > file_size || descriptor_bytes > file_size - index_offset) {
            v4_error(error_message, "operation_envelope_failed: descriptor table arithmetic overflow");
            return 1;
        }
        raw_descriptor = data + (size_t)index_offset;
        request_sequence = read_le64(raw_descriptor);
        manifest_offset = read_le64(raw_descriptor + 8u);
        manifest_bytes = read_le64(raw_descriptor + 16u);
        sidecar_offset = read_le64(raw_descriptor + 24u);
        sidecar_bytes = read_le64(raw_descriptor + 32u);
        payload_offset = read_le64(raw_descriptor + 40u);
        payload_bytes = read_le64(raw_descriptor + 48u);
        if (index != 0u && request_sequence <= previous_sequence) {
            v4_error(error_message, "operation_envelope_failed: request descriptors are reordered");
            return 1;
        }
        previous_sequence = request_sequence;
        if (read_le32(raw_descriptor + 56u) == 0u ||
            read_le32(raw_descriptor + 60u) != 0u ||
            read_le64(raw_descriptor + 64u) == 0u ||
            manifest_bytes == 0u || sidecar_bytes == 0u || payload_bytes == 0u ||
            !extent_inside(manifest_offset, manifest_bytes, file_size) ||
            !extent_inside(sidecar_offset, sidecar_bytes, file_size) ||
            !extent_inside(payload_offset, payload_bytes, file_size) ||
            manifest_offset != cursor ||
            checked_add_u64(manifest_offset, manifest_bytes, &next_cursor) != 0 ||
            sidecar_offset != next_cursor ||
            checked_add_u64(sidecar_offset, sidecar_bytes, &next_cursor) != 0 ||
            payload_offset != next_cursor ||
            checked_add_u64(payload_offset, payload_bytes, &next_cursor) != 0) {
            v4_error(error_message, "operation_envelope_failed: descriptor body regions are not ordered and contiguous");
            return 1;
        }
        memcpy(descriptor_copy, raw_descriptor, sizeof(descriptor_copy));
        memset(descriptor_copy + 168u, 0, 32u);
        if (!digest_bytes_matches(
                descriptor_copy, sizeof(descriptor_copy), raw_descriptor + 168u)) {
            v4_error(error_message, "operation_envelope_failed: descriptor SHA-256 mismatch");
            return 1;
        }
        embedded.manifest = data + (size_t)manifest_offset;
        embedded.manifest_bytes = (size_t)manifest_bytes;
        embedded.sidecar = data + (size_t)sidecar_offset;
        embedded.sidecar_bytes = (size_t)sidecar_bytes;
        embedded.payload = data + (size_t)payload_offset;
        embedded.payload_bytes = (size_t)payload_bytes;
        embedded.request_sequence = request_sequence;
        embedded.request_output_elements = read_le64(raw_descriptor + 64u);
        embedded.work_unit_count = read_le32(raw_descriptor + 56u);
        memcpy(embedded.manifest_sha256, raw_descriptor + 72u, 32u);
        memcpy(embedded.sidecar_sha256, raw_descriptor + 104u, 32u);
        memcpy(embedded.payload_sha256, raw_descriptor + 136u, 32u);
        if (execution_plan_v4_request_prepare_embedded(
                session_root, &embedded, expected_dpus, expected_tasklets,
                &operation->requests[index], error_message) != 0) return 1;
        cursor = next_cursor;
    }
    if (cursor != file_size) {
        v4_error(error_message, "operation_envelope_failed: envelope body has a gap or trailing bytes");
        return 1;
    }
    return 0;
}

int execution_plan_v4_operation_open(
    const char *session_root,
    const char *relative_path,
    const char *submitted_sha256,
    uint32_t expected_dpus,
    uint32_t expected_tasklets,
    execution_plan_v4_operation_envelope_t *operation,
    char **error_message
) {
    char path[PATH_MAX];
    struct stat info;
    unsigned char submitted_digest[32];
    unsigned char actual_file_digest[32];
    char actual_file_digest_hex[65];
    int descriptor;
    void *mapping;
    if (operation == NULL) {
        v4_error(error_message, "operation_envelope_failed: missing operation output");
        return 1;
    }
    memset(operation, 0, sizeof(*operation));
    operation->file_descriptor = -1;
    if (session_root == NULL || relative_path == NULL || submitted_sha256 == NULL ||
        digest_text(submitted_sha256, submitted_digest) != 0 ||
        resolve_envelope_path(session_root, relative_path, path) != 0) {
        v4_error(error_message, "operation_envelope_failed: envelope path or SHA-256 is invalid");
        return 1;
    }
#ifdef O_CLOEXEC
    descriptor = open(path, O_RDONLY | O_CLOEXEC);
#else
    descriptor = open(path, O_RDONLY);
#endif
    if (descriptor < 0 || fstat(descriptor, &info) != 0 || !S_ISREG(info.st_mode) || info.st_size <= 0 ||
        (uintmax_t)info.st_size > (uintmax_t)SIZE_MAX || (uintmax_t)info.st_size > (uintmax_t)UINT64_MAX) {
        if (descriptor >= 0) close(descriptor);
        v4_error(error_message, "operation_envelope_failed: envelope file is unreadable");
        return 1;
    }
    mapping = mmap(NULL, (size_t)info.st_size, PROT_READ, MAP_PRIVATE, descriptor, 0);
    if (mapping == MAP_FAILED) {
        close(descriptor);
        v4_error(error_message, "operation_envelope_failed: envelope mmap failed");
        return 1;
    }
    operation->mapping = (const unsigned char *)mapping;
    operation->file_size = (size_t)info.st_size;
    operation->file_descriptor = descriptor;
    if (validate_envelope(session_root, operation, expected_dpus, expected_tasklets, error_message) != 0 ||
        execution_plan_sha256_bytes(
            operation->mapping, operation->file_size, actual_file_digest_hex) != 0 ||
        digest_text(actual_file_digest_hex, actual_file_digest) != 0 ||
        memcmp(submitted_digest, actual_file_digest, sizeof(submitted_digest)) != 0) {
        if (error_message != NULL && *error_message == NULL) {
            v4_error(error_message, "operation_envelope_failed: submitted envelope SHA-256 does not match file");
        }
        execution_plan_v4_operation_close(operation);
        return 1;
    }
    return 0;
}

int execution_plan_v4_operation_descriptor(
    const execution_plan_v4_operation_envelope_t *operation,
    uint32_t index,
    execution_plan_v4_embedded_request_t *descriptor,
    char **error_message
) {
    const unsigned char *raw_descriptor;
    uint64_t manifest_offset;
    uint64_t sidecar_offset;
    uint64_t payload_offset;
    if (descriptor == NULL || operation_descriptor_at(operation, index, &raw_descriptor, error_message) != 0) {
        if (descriptor == NULL) v4_error(error_message, "operation_envelope_failed: missing descriptor output");
        return 1;
    }
    memset(descriptor, 0, sizeof(*descriptor));
    fill_operation_descriptor(raw_descriptor, descriptor);
    manifest_offset = read_le64(raw_descriptor + 8u);
    sidecar_offset = read_le64(raw_descriptor + 24u);
    payload_offset = read_le64(raw_descriptor + 40u);
    descriptor->manifest = operation->mapping + (size_t)manifest_offset;
    descriptor->sidecar = operation->mapping + (size_t)sidecar_offset;
    descriptor->payload = operation->mapping + (size_t)payload_offset;
    return 0;
}

void execution_plan_v4_operation_close(
    execution_plan_v4_operation_envelope_t *operation
) {
    if (operation == NULL) return;
    if (operation->requests != NULL) {
        for (uint32_t index = 0u; index < operation->descriptor_count; index++) {
            execution_plan_v4_request_free(&operation->requests[index]);
        }
    }
    free(operation->requests);
    if (operation->mapping != NULL && operation->file_size != 0u) {
        (void)munmap((void *)operation->mapping, operation->file_size);
    }
    if (operation->file_descriptor >= 0) (void)close(operation->file_descriptor);
    memset(operation, 0, sizeof(*operation));
    operation->file_descriptor = -1;
}
