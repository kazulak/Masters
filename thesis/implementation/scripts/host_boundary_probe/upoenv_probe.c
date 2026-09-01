#include <errno.h>
#include <inttypes.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define HEADER_BYTES 96u
#define V1_DESCRIPTOR_BYTES 152u
#define V2_DESCRIPTOR_BYTES 200u
#define V1_MAX_REQUESTS 64u
#define REQUEST_HEADER_BYTES 168u
#define WORK_UNIT_BYTES 84u
#define MRAM_POOL_BYTES (512u * 1024u)
#define SHA256_BYTES 32u

typedef struct {
    uint32_t state[8];
    uint64_t bit_count;
    unsigned char block[64];
    size_t block_used;
} sha256_ctx;

static const uint32_t K[64] = {
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u,
    0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
    0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
    0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
    0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
    0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
    0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u,
    0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
    0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
    0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
    0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u,
    0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
    0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u,
    0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
    0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
    0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u
};

static uint32_t rotr(uint32_t value, unsigned int bits) {
    return (value >> bits) | (value << (32u - bits));
}

static void sha256_transform(sha256_ctx *ctx, const unsigned char *block) {
    uint32_t w[64];
    uint32_t a, b, c, d, e, f, g, h;
    unsigned int i;
    for (i = 0; i < 16u; ++i) {
        w[i] = ((uint32_t)block[i * 4u] << 24u) |
               ((uint32_t)block[i * 4u + 1u] << 16u) |
               ((uint32_t)block[i * 4u + 2u] << 8u) |
               (uint32_t)block[i * 4u + 3u];
    }
    for (i = 16u; i < 64u; ++i) {
        uint32_t s0 = rotr(w[i - 15u], 7u) ^ rotr(w[i - 15u], 18u) ^
                       (w[i - 15u] >> 3u);
        uint32_t s1 = rotr(w[i - 2u], 17u) ^ rotr(w[i - 2u], 19u) ^
                       (w[i - 2u] >> 10u);
        w[i] = w[i - 16u] + s0 + w[i - 7u] + s1;
    }
    a = ctx->state[0];
    b = ctx->state[1];
    c = ctx->state[2];
    d = ctx->state[3];
    e = ctx->state[4];
    f = ctx->state[5];
    g = ctx->state[6];
    h = ctx->state[7];
    for (i = 0; i < 64u; ++i) {
        uint32_t s1 = rotr(e, 6u) ^ rotr(e, 11u) ^ rotr(e, 25u);
        uint32_t choice = (e & f) ^ ((~e) & g);
        uint32_t temp1 = h + s1 + choice + K[i] + w[i];
        uint32_t s0 = rotr(a, 2u) ^ rotr(a, 13u) ^ rotr(a, 22u);
        uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
        uint32_t temp2 = s0 + majority;
        h = g;
        g = f;
        f = e;
        e = d + temp1;
        d = c;
        c = b;
        b = a;
        a = temp1 + temp2;
    }
    ctx->state[0] += a;
    ctx->state[1] += b;
    ctx->state[2] += c;
    ctx->state[3] += d;
    ctx->state[4] += e;
    ctx->state[5] += f;
    ctx->state[6] += g;
    ctx->state[7] += h;
}

static void sha256_init(sha256_ctx *ctx) {
    ctx->state[0] = 0x6a09e667u;
    ctx->state[1] = 0xbb67ae85u;
    ctx->state[2] = 0x3c6ef372u;
    ctx->state[3] = 0xa54ff53au;
    ctx->state[4] = 0x510e527fu;
    ctx->state[5] = 0x9b05688cu;
    ctx->state[6] = 0x1f83d9abu;
    ctx->state[7] = 0x5be0cd19u;
    ctx->bit_count = 0u;
    ctx->block_used = 0u;
}

static void sha256_update(sha256_ctx *ctx, const unsigned char *data,
                          size_t length) {
    while (length > 0u) {
        size_t available = 64u - ctx->block_used;
        size_t take = length < available ? length : available;
        memcpy(ctx->block + ctx->block_used, data, take);
        ctx->block_used += take;
        ctx->bit_count += (uint64_t)take * 8u;
        data += take;
        length -= take;
        if (ctx->block_used == 64u) {
            sha256_transform(ctx, ctx->block);
            ctx->block_used = 0u;
        }
    }
}

static void sha256_final(sha256_ctx *ctx, unsigned char digest[SHA256_BYTES]) {
    size_t i;
    uint64_t bit_count = ctx->bit_count;
    ctx->block[ctx->block_used++] = 0x80u;
    while (ctx->block_used != 56u) {
        if (ctx->block_used == 64u) {
            sha256_transform(ctx, ctx->block);
            ctx->block_used = 0u;
        }
        ctx->block[ctx->block_used++] = 0u;
    }
    for (i = 0u; i < 8u; ++i) {
        ctx->block[56u + i] = (unsigned char)(bit_count >> (56u - i * 8u));
    }
    sha256_transform(ctx, ctx->block);
    for (i = 0u; i < 8u; ++i) {
        digest[i * 4u] = (unsigned char)(ctx->state[i] >> 24u);
        digest[i * 4u + 1u] = (unsigned char)(ctx->state[i] >> 16u);
        digest[i * 4u + 2u] = (unsigned char)(ctx->state[i] >> 8u);
        digest[i * 4u + 3u] = (unsigned char)ctx->state[i];
    }
}

static uint32_t u32(const unsigned char *data) {
    return (uint32_t)data[0] | ((uint32_t)data[1] << 8u) |
           ((uint32_t)data[2] << 16u) | ((uint32_t)data[3] << 24u);
}

static uint64_t u64(const unsigned char *data) {
    uint64_t value = 0u;
    unsigned int i;
    for (i = 0u; i < 8u; ++i) {
        value |= (uint64_t)data[i] << (i * 8u);
    }
    return value;
}

static int equal_digest(const unsigned char *left, const unsigned char *right) {
    return memcmp(left, right, SHA256_BYTES) == 0;
}

static void digest_range(const unsigned char *data, size_t offset, size_t length,
                         unsigned char digest[SHA256_BYTES]) {
    sha256_ctx ctx;
    sha256_init(&ctx);
    sha256_update(&ctx, data + offset, length);
    sha256_final(&ctx, digest);
}

static void digest_with_zero(const unsigned char *data, size_t total,
                             size_t zero_offset, size_t zero_length,
                             unsigned char digest[SHA256_BYTES]) {
    static const unsigned char zeros[SHA256_BYTES] = {0};
    sha256_ctx ctx;
    sha256_init(&ctx);
    sha256_update(&ctx, data, zero_offset);
    sha256_update(&ctx, zeros, zero_length);
    sha256_update(&ctx, data + zero_offset + zero_length,
                  total - zero_offset - zero_length);
    sha256_final(&ctx, digest);
}

static int checked_add(uint64_t left, uint64_t right, uint64_t *result) {
    if (right > UINT64_MAX - left) return 0;
    *result = left + right;
    return 1;
}

static int checked_mul(uint64_t left, uint64_t right, uint64_t *result) {
    if (left != 0u && right > UINT64_MAX / left) return 0;
    *result = left * right;
    return 1;
}

static int range_valid(uint64_t offset, uint64_t length, uint64_t total) {
    return offset <= total && length <= total - offset;
}

static int range_size_valid(uint64_t offset, uint64_t length, uint64_t total) {
    return range_valid(offset, length, total) && offset <= SIZE_MAX &&
           length <= SIZE_MAX;
}

static int reject(const char *reason) {
    fprintf(stderr, "reject: %s\n", reason);
    return 1;
}

static void print_hex(const unsigned char digest[SHA256_BYTES]) {
    unsigned int i;
    for (i = 0u; i < SHA256_BYTES; ++i) printf("%02x", digest[i]);
}

static int hex_value(unsigned char value) {
    if (value >= '0' && value <= '9') return (int)(value - '0');
    if (value >= 'a' && value <= 'f') return (int)(value - 'a' + 10u);
    if (value >= 'A' && value <= 'F') return (int)(value - 'A' + 10u);
    return -1;
}

static int digest_hex_matches(const unsigned char digest[SHA256_BYTES],
                              const unsigned char *text, size_t length) {
    unsigned char expected[SHA256_BYTES];
    size_t i;
    if (length != SHA256_BYTES * 2u) return 0;
    for (i = 0u; i < SHA256_BYTES; ++i) {
        int high = hex_value(text[i * 2u]);
        int low = hex_value(text[i * 2u + 1u]);
        if (high < 0 || low < 0) return 0;
        expected[i] = (unsigned char)((high << 4) | low);
    }
    return equal_digest(digest, expected);
}

static int token(const unsigned char *data, size_t length, size_t *cursor,
                 const unsigned char **start, size_t *token_length) {
    size_t begin;
    while (*cursor < length &&
           (data[*cursor] == ' ' || data[*cursor] == '\t')) {
        *cursor += 1u;
    }
    begin = *cursor;
    while (*cursor < length && data[*cursor] != ' ' && data[*cursor] != '\t' &&
           data[*cursor] != '\n' && data[*cursor] != '\r') {
        *cursor += 1u;
    }
    if (*cursor == begin) return 0;
    *start = data + begin;
    *token_length = *cursor - begin;
    return 1;
}

static int expect_newline(const unsigned char *data, size_t length,
                          size_t *cursor) {
    if (*cursor < length && data[*cursor] == '\r') *cursor += 1u;
    if (*cursor >= length || data[*cursor] != '\n') return 0;
    *cursor += 1u;
    return 1;
}

static int token_u64(const unsigned char *text, size_t length, uint64_t *value) {
    size_t i;
    uint64_t result = 0u;
    if (length == 0u) return 0;
    for (i = 0u; i < length; ++i) {
        unsigned int digit;
        if (text[i] < '0' || text[i] > '9') return 0;
        digit = (unsigned int)(text[i] - '0');
        if (result > (UINT64_MAX - digit) / 10u) return 0;
        result = result * 10u + digit;
    }
    *value = result;
    return 1;
}

static int token_equals(const unsigned char *text, size_t length,
                        const char *expected) {
    size_t expected_length = strlen(expected);
    return length == expected_length && memcmp(text, expected, length) == 0;
}

static int aligned8_checked(uint64_t value, uint64_t *result) {
    if (value > UINT64_MAX - 7u) return 0;
    *result = (value + 7u) & ~UINT64_C(7);
    return 1;
}

static int validate_manifest_payloads(
    const unsigned char *manifest, size_t manifest_size,
    const unsigned char *sidecar, size_t sidecar_size,
    const unsigned char *payload, size_t payload_size, uint32_t work_count) {
    size_t cursor = 0u;
    uint32_t j;
    const unsigned char *text;
    size_t text_length;
    uint64_t payload_cursor = 0u;
    (void)sidecar_size;
    if (!token(manifest, manifest_size, &cursor, &text, &text_length) ||
        !token_equals(text, text_length, "sidecar") ||
        !token(manifest, manifest_size, &cursor, &text, &text_length) ||
        !expect_newline(manifest, manifest_size, &cursor)) {
        return 0;
    }
    for (j = 0u; j < work_count; ++j) {
        uint64_t dpu_id, tile_id;
        unsigned char digest[SHA256_BYTES];
        const unsigned char *a_hex;
        const unsigned char *b_hex;
        size_t a_hex_length, b_hex_length;
        uint64_t a_size = u32(sidecar + REQUEST_HEADER_BYTES +
                              (uint64_t)j * WORK_UNIT_BYTES + 60u);
        uint64_t b_size = u32(sidecar + REQUEST_HEADER_BYTES +
                              (uint64_t)j * WORK_UNIT_BYTES + 64u);
        uint64_t next;
        if (!token(manifest, manifest_size, &cursor, &text, &text_length) ||
            !token_equals(text, text_length, "dpu") ||
            !token(manifest, manifest_size, &cursor, &text, &text_length) ||
            !token_u64(text, text_length, &dpu_id) || dpu_id != j ||
            !token(manifest, manifest_size, &cursor, &text, &text_length) ||
            !token_u64(text, text_length, &tile_id) ||
            tile_id != u64(sidecar + REQUEST_HEADER_BYTES +
                           (uint64_t)j * WORK_UNIT_BYTES + 8u)) {
            return 0;
        }
        for (unsigned int path_index = 0u; path_index < 3u; ++path_index) {
            if (!token(manifest, manifest_size, &cursor, &text, &text_length)) {
                return 0;
            }
        }
        if (!token(manifest, manifest_size, &cursor, &a_hex, &a_hex_length) ||
            !token(manifest, manifest_size, &cursor, &b_hex, &b_hex_length) ||
            !expect_newline(manifest, manifest_size, &cursor) ||
            a_hex_length != 64u || b_hex_length != 64u) {
            return 0;
        }
        if (!range_size_valid(payload_cursor, a_size, (uint64_t)payload_size)) {
            return 0;
        }
        digest_range(payload, (size_t)payload_cursor, (size_t)a_size, digest);
        if (!digest_hex_matches(digest, a_hex, a_hex_length)) return 0;
        if (!checked_add(payload_cursor, a_size, &next) ||
            !range_size_valid(next, b_size, (uint64_t)payload_size)) {
            return 0;
        }
        digest_range(payload, (size_t)next, (size_t)b_size, digest);
        if (!digest_hex_matches(digest, b_hex, b_hex_length)) return 0;
        if (!checked_add(next, b_size, &payload_cursor)) return 0;
    }
    return cursor == manifest_size && payload_cursor == (uint64_t)payload_size;
}

static int validate_v4_request(const unsigned char *data, size_t size,
                              uint64_t request_sequence,
                              uint64_t manifest_offset, uint64_t manifest_bytes,
                              uint64_t sidecar_offset, uint64_t sidecar_bytes,
                              uint64_t payload_offset, uint64_t payload_bytes,
                              uint32_t descriptor_work_count,
                              uint64_t descriptor_output,
                              const unsigned char *manifest_digest) {
    const unsigned char *manifest;
    const unsigned char *sidecar;
    const unsigned char *payload;
    uint32_t work_count;
    uint32_t dpu_count;
    uint32_t numeric_mode;
    uint64_t expected_sidecar_size;
    uint64_t expected_payload = 0u;
    uint32_t j;
    if (!range_size_valid(manifest_offset, manifest_bytes, (uint64_t)size) ||
        !range_size_valid(sidecar_offset, sidecar_bytes, (uint64_t)size) ||
        !range_size_valid(payload_offset, payload_bytes, (uint64_t)size)) {
        return 0;
    }
    manifest = data + (size_t)manifest_offset;
    sidecar = data + (size_t)sidecar_offset;
    payload = data + (size_t)payload_offset;
    if (sidecar_bytes < REQUEST_HEADER_BYTES ||
        memcmp(sidecar, "UPXDPV4\0", 8u) != 0 || u32(sidecar + 8u) != 4u ||
        u32(sidecar + 12u) != REQUEST_HEADER_BYTES ||
        u32(sidecar + 32u) != 1u ||
        u32(sidecar + 36u) != WORK_UNIT_BYTES ||
        u32(sidecar + 40u) != 0u || u32(sidecar + 44u) != 0u) {
        return 0;
    }
    work_count = u32(sidecar + 16u);
    dpu_count = u32(sidecar + 20u);
    numeric_mode = u32(sidecar + 28u);
    if (work_count == 0u || work_count != descriptor_work_count ||
        dpu_count != work_count || (numeric_mode != 0u && numeric_mode != 1u) ||
        u64(sidecar + 88u) != descriptor_output ||
        !equal_digest(sidecar + 136u, manifest_digest) ||
        !checked_mul(work_count, WORK_UNIT_BYTES, &expected_sidecar_size) ||
        !checked_add(REQUEST_HEADER_BYTES, expected_sidecar_size,
                     &expected_sidecar_size) ||
        expected_sidecar_size != sidecar_bytes) {
        return 0;
    }
    for (j = 0u; j < work_count; ++j) {
        const unsigned char *unit =
            sidecar + REQUEST_HEADER_BYTES + (uint64_t)j * WORK_UNIT_BYTES;
        uint32_t flags = u32(unit + 4u);
        uint64_t a_bytes = u32(unit + 60u);
        uint64_t b_bytes = u32(unit + 64u);
        uint64_t c_bytes = u32(unit + 68u);
        uint64_t m = u32(unit + 48u);
        uint64_t n = u32(unit + 52u);
        uint64_t k = u32(unit + 56u);
        uint64_t elements;
        uint64_t expected_a, expected_b, expected_c;
        uint64_t b_offset, b_end, c_offset;
        if (u32(unit) != j || (flags & ~1u) != 0u) return 0;
        if (flags & 1u) {
            if (m != 0u || n != 0u || k != 0u || a_bytes != 0u ||
                b_bytes != 0u || c_bytes != 0u || u32(unit + 72u) != 0u ||
                u32(unit + 76u) != 0u || u32(unit + 80u) != 0u) {
                return 0;
            }
            continue;
        }
        if (m == 0u || n == 0u || k == 0u ||
            !checked_mul(m, k, &elements) ||
            !checked_mul(elements, numeric_mode == 1u ? 1u : 4u, &expected_a) ||
            !checked_mul(k, n, &elements) ||
            !checked_mul(elements, numeric_mode == 1u ? 1u : 4u, &expected_b) ||
            !checked_mul(m, n, &elements) ||
            !checked_mul(elements, 4u, &expected_c)) {
            return 0;
        }
        if (!aligned8_checked(expected_a, &expected_a) ||
            !aligned8_checked(expected_b, &expected_b) ||
            !aligned8_checked(expected_c, &expected_c) ||
            !aligned8_checked(expected_a, &b_offset)) {
            return 0;
        }
        if (!checked_add(b_offset, expected_b, &b_end) ||
            !aligned8_checked(b_end, &c_offset) ||
            !checked_add(c_offset, expected_c, &b_end) ||
            b_end > MRAM_POOL_BYTES || a_bytes != expected_a ||
            b_bytes != expected_b || c_bytes != expected_c ||
            u32(unit + 72u) != 0u || u32(unit + 76u) != b_offset ||
            u32(unit + 80u) != c_offset) {
            return 0;
        }
        if (!checked_add(expected_payload, a_bytes, &expected_payload) ||
            !checked_add(expected_payload, b_bytes, &expected_payload)) {
            return 0;
        }
    }
    if (expected_payload != payload_bytes) return 0;
    if (u64(sidecar + 96u) != request_sequence) return 0;
    return validate_manifest_payloads(
        manifest, (size_t)manifest_bytes, sidecar, (size_t)sidecar_bytes,
        payload, (size_t)payload_bytes, work_count);
}

static int validate_v1(const unsigned char *data, size_t size) {
    uint32_t count, descriptor_bytes;
    uint64_t descriptors_offset, body_offset, total_bytes, table_bytes;
    unsigned char digest[SHA256_BYTES];
    uint32_t i;
    if (size < HEADER_BYTES) return reject("truncated header");
    if (memcmp(data, "UPOENV1\0", 8u) != 0) return reject("invalid magic");
    if (u32(data + 8u) != 1u || u32(data + 12u) != HEADER_BYTES) {
        return reject("invalid envelope version or header size");
    }
    count = u32(data + 16u);
    descriptor_bytes = u32(data + 20u);
    if (count == 0u || count > V1_MAX_REQUESTS) return reject("invalid request count");
    if (descriptor_bytes != V1_DESCRIPTOR_BYTES || u32(data + 24u) != 0u ||
        u32(data + 28u) != 0u) {
        return reject("invalid descriptor layout or reserved fields");
    }
    descriptors_offset = u64(data + 32u);
    body_offset = u64(data + 40u);
    total_bytes = u64(data + 48u);
    if (!checked_mul(count, V1_DESCRIPTOR_BYTES, &table_bytes) ||
        !checked_add(descriptors_offset, table_bytes, &table_bytes) ||
        descriptors_offset != HEADER_BYTES || body_offset != table_bytes ||
        total_bytes != (uint64_t)size || !range_valid(body_offset, 0u, total_bytes)) {
        return reject("invalid envelope bounds");
    }
    digest_with_zero(data, size, 64u, SHA256_BYTES, digest);
    if (!equal_digest(digest, data + 64u)) return reject("envelope digest mismatch");
    for (i = 0u; i < count; ++i) {
        const unsigned char *descriptor =
            data + (size_t)(descriptors_offset + (uint64_t)i * V1_DESCRIPTOR_BYTES);
        uint64_t request_offset = u64(descriptor + 8u);
        uint64_t request_bytes = u64(descriptor + 16u);
        uint64_t payload_offset = u64(descriptor + 24u);
        uint64_t payload_bytes = u64(descriptor + 32u);
        uint32_t work_count = u32(descriptor + 40u);
        uint64_t request_output = u64(descriptor + 48u);
        uint64_t expected_request, expected_payload = 0u;
        unsigned char descriptor_digest[SHA256_BYTES];
        unsigned char request_digest[SHA256_BYTES];
        unsigned char payload_digest[SHA256_BYTES];
        uint32_t j;
        if (u64(descriptor) != i || u32(descriptor + 44u) != 0u ||
            work_count == 0u || work_count > V1_MAX_REQUESTS) {
            return reject("reordered descriptor or invalid descriptor fields");
        }
        if (!range_valid(request_offset, request_bytes, total_bytes) ||
            !range_valid(payload_offset, payload_bytes, total_bytes) ||
            request_offset < body_offset || payload_offset < body_offset ||
            !checked_add(request_offset, request_bytes, &expected_request) ||
            expected_request != payload_offset) {
            return reject("truncated or overlapping request ranges");
        }
        if (i > 0u) {
            const unsigned char *previous =
                data + (size_t)(descriptors_offset + (uint64_t)(i - 1u) * V1_DESCRIPTOR_BYTES);
            uint64_t previous_payload;
            if (!checked_add(u64(previous + 24u), u64(previous + 32u),
                             &previous_payload) ||
                request_offset != previous_payload) {
                return reject("request descriptors are not contiguous and ordered");
            }
        }
        if (!checked_mul(work_count, WORK_UNIT_BYTES, &expected_request) ||
            !checked_add(REQUEST_HEADER_BYTES, expected_request, &expected_request) ||
            request_bytes != expected_request ||
            !range_size_valid(request_offset, request_bytes, total_bytes) ||
            !range_size_valid(payload_offset, payload_bytes, total_bytes)) {
            return reject("request byte count does not match work-unit count");
        }
        digest_range(data, (size_t)request_offset, (size_t)request_bytes, request_digest);
        digest_range(data, (size_t)payload_offset, (size_t)payload_bytes, payload_digest);
        if (!equal_digest(request_digest, descriptor + 56u) ||
            !equal_digest(payload_digest, descriptor + 88u)) {
            return reject("request or payload digest mismatch");
        }
        digest_with_zero(descriptor, V1_DESCRIPTOR_BYTES, 120u, SHA256_BYTES,
                         descriptor_digest);
        if (!equal_digest(descriptor_digest, descriptor + 120u)) {
            return reject("descriptor digest mismatch");
        }
        {
            const unsigned char *request = data + (size_t)request_offset;
            if (memcmp(request, "UPXDPV4\0", 8u) != 0 || u32(request + 8u) != 4u ||
                u32(request + 12u) != REQUEST_HEADER_BYTES ||
                u32(request + 16u) != work_count || u32(request + 20u) != work_count ||
                u32(request + 36u) != WORK_UNIT_BYTES ||
                u64(request + 96u) != u64(descriptor) ||
                request_output != u64(request + 88u)) {
                return reject("embedded ABI-v4 request header mismatch");
            }
            for (j = 0u; j < work_count; ++j) {
                const unsigned char *unit = request + REQUEST_HEADER_BYTES +
                                            (uint64_t)j * WORK_UNIT_BYTES;
                uint32_t flags = u32(unit + 4u);
                uint64_t a_bytes = u32(unit + 60u);
                uint64_t b_bytes = u32(unit + 64u);
                if (u32(unit) != j || (flags & ~1u) != 0u ||
                    (flags & 1u && (a_bytes != 0u || b_bytes != 0u ||
                                    u32(unit + 68u) != 0u)) ||
                    !checked_add(expected_payload, a_bytes, &expected_payload) ||
                    !checked_add(expected_payload, b_bytes, &expected_payload)) {
                    return reject("reordered or invalid work-unit record");
                }
            }
        }
        if (expected_payload != payload_bytes) return reject("payload count mismatch");
    }
    {
        const unsigned char *last = data + (size_t)(descriptors_offset +
                                      (uint64_t)(count - 1u) * V1_DESCRIPTOR_BYTES);
        uint64_t end;
        if (!checked_add(u64(last + 24u), u64(last + 32u), &end) ||
            end != total_bytes) {
            return reject("body has a gap or trailing bytes");
        }
    }
    printf("{\"envelope_sha256\":\"");
    print_hex(data + 64u);
    printf("\",\"request_count\":%" PRIu32 ",\"requests\":[", count);
    for (i = 0u; i < count; ++i) {
        const unsigned char *descriptor = data + (size_t)(descriptors_offset +
                                          (uint64_t)i * V1_DESCRIPTOR_BYTES);
        printf("%s{\"descriptor_sha256\":\"", i == 0u ? "" : ",");
        print_hex(descriptor + 120u);
        printf("\",\"index\":%" PRIu32 ",\"payload_bytes\":%" PRIu64
               ",\"payload_sha256\":\"", i, u64(descriptor + 32u));
        print_hex(descriptor + 88u);
        printf("\",\"request_bytes\":%" PRIu64
               ",\"request_output_elements\":%" PRIu64
               ",\"request_sequence\":%" PRIu64
               ",\"request_sha256\":\"", u64(descriptor + 16u),
               u64(descriptor + 48u), u64(descriptor));
        print_hex(descriptor + 56u);
        printf("\",\"work_unit_count\":%" PRIu32 "}", u32(descriptor + 40u));
    }
    printf("],\"schema\":\"upoenv1_probe_summary_v1\",\"status\":\"accepted\",\"version\":1}\n");
    return 0;
}

static int validate_v2(const unsigned char *data, size_t size) {
    uint32_t count;
    uint64_t table_bytes, descriptors_end, body_offset, total_bytes;
    uint64_t cursor;
    unsigned char digest[SHA256_BYTES];
    uint32_t i;
    if (size < HEADER_BYTES) return reject("truncated header");
    if (memcmp(data, "UPOENV2\0", 8u) != 0) return reject("invalid magic");
    if (u32(data + 8u) != 2u || u32(data + 12u) != HEADER_BYTES) {
        return reject("invalid envelope version or header size");
    }
    count = u32(data + 16u);
    if (count == 0u || u32(data + 20u) != V2_DESCRIPTOR_BYTES ||
        u32(data + 24u) != 0u || u32(data + 28u) != 0u) {
        return reject("invalid descriptor layout or reserved fields");
    }
    if (!checked_mul(count, V2_DESCRIPTOR_BYTES, &table_bytes) ||
        !checked_add(HEADER_BYTES, table_bytes, &body_offset) ||
        u64(data + 32u) != HEADER_BYTES || u64(data + 40u) != body_offset ||
        !range_valid(body_offset, 0u, (uint64_t)size) ||
        u64(data + 48u) != (uint64_t)size) {
        return reject("invalid envelope bounds");
    }
    total_bytes = u64(data + 48u);
    descriptors_end = body_offset;
    (void)descriptors_end;
    digest_with_zero(data, size, 64u, SHA256_BYTES, digest);
    if (!equal_digest(digest, data + 64u)) return reject("envelope digest mismatch");
    cursor = body_offset;
    for (i = 0u; i < count; ++i) {
        const unsigned char *descriptor = data + (size_t)(HEADER_BYTES +
                                         (uint64_t)i * V2_DESCRIPTOR_BYTES);
        uint64_t manifest_offset = u64(descriptor + 8u);
        uint64_t manifest_bytes = u64(descriptor + 16u);
        uint64_t sidecar_offset = u64(descriptor + 24u);
        uint64_t sidecar_bytes = u64(descriptor + 32u);
        uint64_t payload_offset = u64(descriptor + 40u);
        uint64_t payload_bytes = u64(descriptor + 48u);
        uint64_t expected_sidecar_offset, expected_payload_offset, payload_end;
        unsigned char manifest_digest[SHA256_BYTES];
        unsigned char sidecar_digest[SHA256_BYTES];
        unsigned char payload_digest[SHA256_BYTES];
        unsigned char descriptor_digest[SHA256_BYTES];
        if (u64(descriptor) != i || u32(descriptor + 60u) != 0u ||
            manifest_offset != cursor ||
            !checked_add(manifest_offset, manifest_bytes, &expected_sidecar_offset) ||
            sidecar_offset != expected_sidecar_offset ||
            !checked_add(sidecar_offset, sidecar_bytes, &expected_payload_offset) ||
            payload_offset != expected_payload_offset ||
            !checked_add(payload_offset, payload_bytes, &payload_end) ||
            !range_size_valid(manifest_offset, manifest_bytes, total_bytes) ||
            !range_size_valid(sidecar_offset, sidecar_bytes, total_bytes) ||
            !range_size_valid(payload_offset, payload_bytes, total_bytes)) {
            return reject("truncated or overlapping ordered regions");
        }
        digest_range(data, (size_t)manifest_offset, (size_t)manifest_bytes,
                     manifest_digest);
        digest_range(data, (size_t)sidecar_offset, (size_t)sidecar_bytes,
                     sidecar_digest);
        digest_range(data, (size_t)payload_offset, (size_t)payload_bytes,
                     payload_digest);
        if (!equal_digest(manifest_digest, descriptor + 72u) ||
            !equal_digest(sidecar_digest, descriptor + 104u) ||
            !equal_digest(payload_digest, descriptor + 136u)) {
            return reject("manifest, sidecar, or payload digest mismatch");
        }
        digest_with_zero(descriptor, V2_DESCRIPTOR_BYTES, 168u, SHA256_BYTES,
                         descriptor_digest);
        if (!equal_digest(descriptor_digest, descriptor + 168u)) {
            return reject("descriptor digest mismatch");
        }
        if (!validate_v4_request(
                data, size, u64(descriptor), manifest_offset, manifest_bytes, sidecar_offset,
                sidecar_bytes, payload_offset, payload_bytes, u32(descriptor + 56u),
                u64(descriptor + 64u), descriptor + 72u)) {
            return reject("embedded ABI-v4 request or payload mismatch");
        }
        cursor = payload_end;
    }
    if (cursor != total_bytes) return reject("body has a gap or trailing bytes");
    printf("{\"descriptor_count\":%" PRIu32 ",\"envelope_sha256\":\"", count);
    print_hex(data + 64u);
    printf("\",\"operation_sequence\":%" PRIu64 ",\"requests\":[",
           u64(data + 56u));
    for (i = 0u; i < count; ++i) {
        const unsigned char *descriptor = data + (size_t)(HEADER_BYTES +
                                         (uint64_t)i * V2_DESCRIPTOR_BYTES);
        printf("%s{\"descriptor_sha256\":\"", i == 0u ? "" : ",");
        print_hex(descriptor + 168u);
        printf("\",\"index\":%" PRIu32 ",\"manifest_bytes\":%" PRIu64
               ",\"manifest_sha256\":\"", i, u64(descriptor + 16u));
        print_hex(descriptor + 72u);
        printf("\",\"payload_bytes\":%" PRIu64
               ",\"payload_sha256\":\"", u64(descriptor + 48u));
        print_hex(descriptor + 136u);
        printf("\",\"request_output_elements\":%" PRIu64
               ",\"request_sequence\":%" PRIu64
               ",\"sidecar_bytes\":%" PRIu64
               ",\"sidecar_sha256\":\"", u64(descriptor + 64u),
               u64(descriptor), u64(descriptor + 32u));
        print_hex(descriptor + 104u);
        printf("\",\"work_unit_count\":%" PRIu32 "}",
               u32(descriptor + 56u));
    }
    printf("],\"schema\":\"upoenv2_probe_summary_v1\",\"status\":\"accepted\",\"version\":2}\n");
    return 0;
}

static int validate(const unsigned char *data, size_t size) {
    if (size >= 8u && memcmp(data, "UPOENV2\0", 8u) == 0) {
        return validate_v2(data, size);
    }
    return validate_v1(data, size);
}

int main(int argc, char **argv) {
    FILE *file;
    long file_size;
    unsigned char *data;
    size_t read_size;
    int result;
    if (argc != 2) {
        fprintf(stderr, "usage: %s OPERATION.UPOENV\n", argv[0]);
        return 2;
    }
    file = fopen(argv[1], "rb");
    if (file == NULL) return reject(strerror(errno));
    if (fseek(file, 0L, SEEK_END) != 0 || (file_size = ftell(file)) < 0L ||
        fseek(file, 0L, SEEK_SET) != 0 || (uintmax_t)file_size > SIZE_MAX) {
        fclose(file);
        return reject("cannot determine input size");
    }
    data = (unsigned char *)malloc(file_size == 0L ? 1u : (size_t)file_size);
    if (data == NULL) {
        fclose(file);
        return reject("out of memory");
    }
    read_size = fread(data, 1u, (size_t)file_size, file);
    fclose(file);
    if (read_size != (size_t)file_size) {
        free(data);
        return reject("truncated input");
    }
    result = validate(data, read_size);
    free(data);
    return result;
}
