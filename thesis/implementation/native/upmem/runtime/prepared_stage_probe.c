#define _POSIX_C_SOURCE 200809L

#include "plan.h"

#include <errno.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define PROBE_MAGIC "NHPV1\0\0\0"
#define PROBE_VERSION 1u
#define PROBE_HEADER_BYTES 40u
#define PROBE_LANE_COUNT 4u
#define PROBE_RECORD_BYTES 84u
#define PROBE_MAX_RECORDS 1000000u
#define PROBE_MAX_PAYLOAD_BYTES (64u * 1024u * 1024u)

typedef struct __attribute__((packed)) {
    char magic[8];
    uint32_t version;
    uint32_t header_bytes;
    uint32_t record_count;
    uint32_t lane_count;
    uint32_t wave_count;
    uint32_t payload_bytes;
    uint64_t seed;
} probe_header_t;

_Static_assert(sizeof(probe_header_t) == PROBE_HEADER_BYTES,
    "prepared-stage probe header layout changed");
_Static_assert(sizeof(execution_plan_v4_work_unit_t) == PROBE_RECORD_BYTES,
    "probe records must retain the v4 work-unit width");

static double now_s(void) {
    struct timespec value;
    if (clock_gettime(CLOCK_MONOTONIC, &value) != 0) return 0.0;
    return (double)value.tv_sec + (double)value.tv_nsec / 1000000000.0;
}

static int read_file(const char *path, unsigned char **data, size_t *length) {
    FILE *file = NULL;
    long size;
    unsigned char *buffer = NULL;
    if (path == NULL || data == NULL || length == NULL) return 1;
    file = fopen(path, "rb");
    if (file == NULL || fseek(file, 0, SEEK_END) != 0) goto failure;
    size = ftell(file);
    if (size < 0 || fseek(file, 0, SEEK_SET) != 0) goto failure;
    if ((uintmax_t)size > PROBE_HEADER_BYTES +
        (uintmax_t)PROBE_MAX_RECORDS * PROBE_RECORD_BYTES + PROBE_MAX_PAYLOAD_BYTES) {
        goto failure;
    }
    buffer = (unsigned char *)malloc((size_t)size == 0u ? 1u : (size_t)size);
    if (buffer == NULL || fread(buffer, 1u, (size_t)size, file) != (size_t)size) goto failure;
    if (fclose(file) != 0) {
        file = NULL;
        goto failure;
    }
    *data = buffer;
    *length = (size_t)size;
    return 0;

failure:
    if (file != NULL) fclose(file);
    free(buffer);
    return 1;
}

static int write_file(const char *path, const unsigned char *data, size_t length) {
    FILE *file = fopen(path, "wb");
    int status;
    if (file == NULL) return 1;
    status = fwrite(data, 1u, length, file) == length ? 0 : 1;
    if (fclose(file) != 0) status = 1;
    return status;
}

static int parse_iterations(const char *text, uint32_t *iterations) {
    char *end = NULL;
    unsigned long value;
    if (text == NULL || iterations == NULL || text[0] == '\0') return 1;
    errno = 0;
    value = strtoul(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || value == 0u || value > UINT32_MAX) return 1;
    *iterations = (uint32_t)value;
    return 0;
}

static int validate_packet(
    const unsigned char *packet,
    size_t packet_bytes,
    probe_header_t *header,
    size_t *canonical_bytes
) {
    size_t records_bytes;
    size_t expected_bytes;
    if (packet == NULL || header == NULL || canonical_bytes == NULL || packet_bytes < sizeof(*header)) return 1;
    memcpy(header, packet, sizeof(*header));
    if (memcmp(header->magic, PROBE_MAGIC, sizeof(header->magic)) != 0 ||
        header->version != PROBE_VERSION || header->header_bytes != PROBE_HEADER_BYTES ||
        header->lane_count != PROBE_LANE_COUNT || header->wave_count == 0u ||
        header->record_count == 0u || header->record_count > PROBE_MAX_RECORDS ||
        header->payload_bytes > PROBE_MAX_PAYLOAD_BYTES) return 1;
    records_bytes = (size_t)header->record_count * PROBE_RECORD_BYTES;
    expected_bytes = sizeof(*header) + records_bytes + header->payload_bytes;
    if (expected_bytes != packet_bytes) return 1;
    *canonical_bytes = records_bytes + header->payload_bytes;
    return 0;
}

int main(int argc, char **argv) {
    unsigned char *packet = NULL;
    unsigned char *canonical = NULL;
    size_t packet_bytes = 0u;
    size_t canonical_bytes = 0u;
    size_t records_bytes;
    probe_header_t header;
    uint32_t iterations;
    char packet_sha256[65] = {0};
    char canonical_sha256[65] = {0};
    double setup_started;
    double steady_started;
    double setup_s;
    double steady_s;
    int status = 1;

    if (argc != 4 || parse_iterations(argv[3], &iterations) != 0) {
        fprintf(stderr, "usage: %s PACKET OUTPUT ITERATIONS\n", argv[0]);
        return 2;
    }
    setup_started = now_s();
    if (read_file(argv[1], &packet, &packet_bytes) != 0 ||
        validate_packet(packet, packet_bytes, &header, &canonical_bytes) != 0 ||
        execution_plan_sha256_bytes(packet, packet_bytes, packet_sha256) != 0) {
        fprintf(stderr, "invalid prepared-stage packet\n");
        goto cleanup;
    }
    records_bytes = (size_t)header.record_count * PROBE_RECORD_BYTES;
    setup_s = now_s() - setup_started;
    steady_started = now_s();
    for (uint32_t iteration = 0u; iteration < iterations; iteration++) {
        unsigned char *current = (unsigned char *)malloc(canonical_bytes);
        if (current == NULL) goto cleanup;
        for (uint32_t record = 0u; record < header.record_count; record++) {
            memcpy(current + (size_t)record * PROBE_RECORD_BYTES,
                packet + sizeof(header) + (size_t)record * PROBE_RECORD_BYTES,
                PROBE_RECORD_BYTES);
        }
        memcpy(current + records_bytes, packet + sizeof(header) + records_bytes,
            header.payload_bytes);
        if (execution_plan_sha256_bytes(current, canonical_bytes, canonical_sha256) != 0) {
            free(current);
            goto cleanup;
        }
        free(canonical);
        canonical = current;
    }
    steady_s = now_s() - steady_started;
    if (write_file(argv[2], canonical, canonical_bytes) != 0) goto cleanup;
    printf(
        "{\"setup_s\":%.9f,\"steady_s\":%.9f,\"iterations\":%" PRIu32
        ",\"record_count\":%" PRIu32 ",\"payload_bytes\":%" PRIu32
        ",\"canonical_bytes\":%zu,\"packet_sha256\":\"%s\",\"canonical_sha256\":\"%s\"}\n",
        setup_s, steady_s, iterations, header.record_count, header.payload_bytes,
        canonical_bytes, packet_sha256, canonical_sha256);
    status = 0;

cleanup:
    free(canonical);
    free(packet);
    return status;
}
