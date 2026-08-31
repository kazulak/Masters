#include <errno.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define HEADER_BYTES 96u
#define DESCRIPTOR_BYTES 152u
#define MAX_REQUESTS 64u
#define MAX_FILE_BYTES (64u * 1024u * 1024u)
#define REQUEST_HEADER_BYTES 168u
#define WORK_UNIT_BYTES 84u

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
        uint32_t s0 = rotr(w[i - 15u], 7u) ^ rotr(w[i - 15u], 18u) ^ (w[i - 15u] >> 3u);
        uint32_t s1 = rotr(w[i - 2u], 17u) ^ rotr(w[i - 2u], 19u) ^ (w[i - 2u] >> 10u);
        w[i] = w[i - 16u] + s0 + w[i - 7u] + s1;
    }
    a = ctx->state[0]; b = ctx->state[1]; c = ctx->state[2]; d = ctx->state[3];
    e = ctx->state[4]; f = ctx->state[5]; g = ctx->state[6]; h = ctx->state[7];
    for (i = 0; i < 64u; ++i) {
        uint32_t s1 = rotr(e, 6u) ^ rotr(e, 11u) ^ rotr(e, 25u);
        uint32_t choice = (e & f) ^ ((~e) & g);
        uint32_t temp1 = h + s1 + choice + K[i] + w[i];
        uint32_t s0 = rotr(a, 2u) ^ rotr(a, 13u) ^ rotr(a, 22u);
        uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
        uint32_t temp2 = s0 + majority;
        h = g; g = f; f = e; e = d + temp1;
        d = c; c = b; b = a; a = temp1 + temp2;
    }
    ctx->state[0] += a; ctx->state[1] += b; ctx->state[2] += c; ctx->state[3] += d;
    ctx->state[4] += e; ctx->state[5] += f; ctx->state[6] += g; ctx->state[7] += h;
}

static void sha256_init(sha256_ctx *ctx) {
    ctx->state[0] = 0x6a09e667u; ctx->state[1] = 0xbb67ae85u;
    ctx->state[2] = 0x3c6ef372u; ctx->state[3] = 0xa54ff53au;
    ctx->state[4] = 0x510e527fu; ctx->state[5] = 0x9b05688cu;
    ctx->state[6] = 0x1f83d9abu; ctx->state[7] = 0x5be0cd19u;
    ctx->bit_count = 0u;
    ctx->block_used = 0u;
}

static void sha256_update(sha256_ctx *ctx, const unsigned char *data, size_t length) {
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

static void sha256_final(sha256_ctx *ctx, unsigned char digest[32]) {
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
    for (i = 0; i < 8u; ++i) {
        ctx->block[56u + i] = (unsigned char)(bit_count >> (56u - i * 8u));
    }
    sha256_transform(ctx, ctx->block);
    for (i = 0; i < 8u; ++i) {
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
    for (i = 0; i < 8u; ++i) {
        value |= (uint64_t)data[i] << (i * 8u);
    }
    return value;
}

static int equal_digest(const unsigned char *left, const unsigned char *right) {
    return memcmp(left, right, 32u) == 0;
}

static void digest_range(const unsigned char *data, size_t offset, size_t length,
                         unsigned char digest[32]) {
    sha256_ctx ctx;
    sha256_init(&ctx);
    sha256_update(&ctx, data + offset, length);
    sha256_final(&ctx, digest);
}

static void digest_with_zero(const unsigned char *data, size_t total,
                             size_t zero_offset, size_t zero_length,
                             unsigned char digest[32]) {
    static const unsigned char zeros[32] = {0};
    sha256_ctx ctx;
    sha256_init(&ctx);
    sha256_update(&ctx, data, zero_offset);
    sha256_update(&ctx, zeros, zero_length);
    sha256_update(&ctx, data + zero_offset + zero_length,
                  total - zero_offset - zero_length);
    sha256_final(&ctx, digest);
}

static int range_valid(uint64_t offset, uint64_t length, uint64_t total) {
    return offset <= total && length <= total - offset;
}

static int reject(const char *reason) {
    fprintf(stderr, "reject: %s\n", reason);
    return 1;
}

static void print_hex(const unsigned char digest[32]) {
    unsigned int i;
    for (i = 0; i < 32u; ++i) {
        printf("%02x", digest[i]);
    }
}

static int validate(const unsigned char *data, size_t size) {
    uint32_t count, descriptor_bytes;
    uint64_t descriptors_offset, body_offset, total_bytes;
    unsigned char digest[32];
    uint32_t i;

    if (size < HEADER_BYTES) return reject("truncated header");
    if (memcmp(data, "UPOENV1\0", 8u) != 0) return reject("invalid magic");
    if (u32(data + 8u) != 1u || u32(data + 12u) != HEADER_BYTES)
        return reject("invalid envelope version or header size");
    count = u32(data + 16u);
    descriptor_bytes = u32(data + 20u);
    if (count == 0u || count > MAX_REQUESTS) return reject("invalid request count");
    if (descriptor_bytes != DESCRIPTOR_BYTES || u32(data + 24u) != 0u || u32(data + 28u) != 0u)
        return reject("invalid descriptor layout or reserved fields");
    descriptors_offset = u64(data + 32u);
    body_offset = u64(data + 40u);
    total_bytes = u64(data + 48u);
    if (descriptors_offset != HEADER_BYTES ||
        body_offset != descriptors_offset + (uint64_t)count * DESCRIPTOR_BYTES ||
        total_bytes != (uint64_t)size || total_bytes > MAX_FILE_BYTES)
        return reject("invalid envelope bounds");
    if (body_offset > total_bytes) return reject("truncated descriptor table");
    digest_with_zero(data, size, 64u, 32u, digest);
    if (!equal_digest(digest, data + 64u)) return reject("envelope digest mismatch");

    for (i = 0u; i < count; ++i) {
        const unsigned char *descriptor = data + descriptors_offset + (uint64_t)i * DESCRIPTOR_BYTES;
        uint64_t request_offset = u64(descriptor + 8u);
        uint64_t request_bytes = u64(descriptor + 16u);
        uint64_t payload_offset = u64(descriptor + 24u);
        uint64_t payload_bytes = u64(descriptor + 32u);
        uint32_t work_count = u32(descriptor + 40u);
        uint64_t request_output = u64(descriptor + 48u);
        unsigned char descriptor_digest[32];
        unsigned char request_digest[32];
        unsigned char payload_digest[32];
        uint64_t expected_payload = 0u;
        uint32_t j;

        if (u64(descriptor) != i || u32(descriptor + 44u) != 0u)
            return reject("reordered descriptor or invalid descriptor fields");
        if (work_count == 0u || work_count > MAX_REQUESTS)
            return reject("invalid work-unit count");
        if (!range_valid(request_offset, request_bytes, total_bytes) ||
            !range_valid(payload_offset, payload_bytes, total_bytes) ||
            request_offset < body_offset || payload_offset < body_offset ||
            request_offset + request_bytes != payload_offset)
            return reject("truncated or overlapping request ranges");
        if (i > 0u) {
            const unsigned char *previous = data + descriptors_offset + (uint64_t)(i - 1u) * DESCRIPTOR_BYTES;
            uint64_t previous_payload = u64(previous + 24u) + u64(previous + 32u);
            if (request_offset != previous_payload)
                return reject("request descriptors are not contiguous and ordered");
        }
        if (request_bytes != REQUEST_HEADER_BYTES + (uint64_t)work_count * WORK_UNIT_BYTES)
            return reject("request byte count does not match work-unit count");
        digest_range(data, (size_t)request_offset, (size_t)request_bytes, request_digest);
        digest_range(data, (size_t)payload_offset, (size_t)payload_bytes, payload_digest);
        if (!equal_digest(request_digest, descriptor + 56u) ||
            !equal_digest(payload_digest, descriptor + 88u))
            return reject("request or payload digest mismatch");
        digest_with_zero(descriptor, DESCRIPTOR_BYTES, 120u, 32u, descriptor_digest);
        if (!equal_digest(descriptor_digest, descriptor + 120u))
            return reject("descriptor digest mismatch");

        {
            const unsigned char *request = data + request_offset;
            uint32_t request_count = u32(request + 16u);
            uint64_t request_sequence = u64(request + 96u);
            if (memcmp(request, "UPXDPV4\0", 8u) != 0 || u32(request + 8u) != 4u ||
                u32(request + 12u) != REQUEST_HEADER_BYTES || request_count != work_count ||
                u32(request + 20u) != work_count || u32(request + 36u) != WORK_UNIT_BYTES ||
                request_sequence != u64(descriptor))
                return reject("embedded ABI-v4 request header mismatch");
            if (request_output != u64(request + 88u)) return reject("output descriptor mismatch");
            for (j = 0u; j < work_count; ++j) {
                const unsigned char *unit = request + REQUEST_HEADER_BYTES + (uint64_t)j * WORK_UNIT_BYTES;
                uint32_t flags = u32(unit + 4u);
                uint64_t a_bytes = u32(unit + 60u);
                uint64_t b_bytes = u32(unit + 64u);
                if (u32(unit) != j || (flags & ~1u) != 0u)
                    return reject("reordered or invalid work-unit record");
                if (flags & 1u) {
                    if (a_bytes != 0u || b_bytes != 0u || u32(unit + 68u) != 0u)
                        return reject("zero-work record carries payload data");
                }
                expected_payload += a_bytes + b_bytes;
            }
        }
        if (expected_payload != payload_bytes) return reject("payload count mismatch");
    }
    {
        const unsigned char *last = data + descriptors_offset + (uint64_t)(count - 1u) * DESCRIPTOR_BYTES;
        if (u64(last + 24u) + u64(last + 32u) != total_bytes)
            return reject("body has a gap or trailing bytes");
    }

    printf("{\"envelope_sha256\":\"");
    print_hex(data + 64u);
    printf("\",\"request_count\":%" PRIu32 ",\"requests\":[", count);
    for (i = 0u; i < count; ++i) {
        const unsigned char *descriptor = data + descriptors_offset + (uint64_t)i * DESCRIPTOR_BYTES;
        printf("%s{\"descriptor_sha256\":\"", i == 0u ? "" : ",");
        print_hex(descriptor + 120u);
        printf("\",\"index\":%" PRIu32 ",\"payload_bytes\":%" PRIu64 ",\"payload_sha256\":\"", i, u64(descriptor + 32u));
        print_hex(descriptor + 88u);
        printf("\",\"request_bytes\":%" PRIu64 ",\"request_output_elements\":%" PRIu64 ",\"request_sequence\":%" PRIu64 ",\"request_sha256\":\"", u64(descriptor + 16u), u64(descriptor + 48u), u64(descriptor));
        print_hex(descriptor + 56u);
        printf("\",\"work_unit_count\":%" PRIu32 "}", u32(descriptor + 40u));
    }
    printf("],\"schema\":\"upoenv1_probe_summary_v1\",\"status\":\"accepted\",\"version\":1}\n");
    return 0;
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
        (unsigned long)file_size > MAX_FILE_BYTES || fseek(file, 0L, SEEK_SET) != 0) {
        fclose(file);
        return reject("cannot determine input size");
    }
    data = (unsigned char *)malloc((size_t)file_size);
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
