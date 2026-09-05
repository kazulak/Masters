#define _POSIX_C_SOURCE 200809L
#include "wave_envelope.h"
#include "plan.h"
#include <fcntl.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

static int fail(char **message, const char *reason) {
    if (message && !*message) *message = strdup(reason);
    return 1;
}

static uint32_t le32(const unsigned char *p) {
    return (uint32_t)p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 |
        (uint32_t)p[3] << 24;
}

static uint64_t le64(const unsigned char *p) {
    return le32(p) | (uint64_t)le32(p + 4) << 32;
}

static int nonzero_digest(const unsigned char *p) {
    unsigned char bits = 0;
    for (unsigned i = 0; i < 32; ++i) bits |= p[i];
    return bits != 0;
}

static int digest_matches(const unsigned char *bytes, const char *hex) {
    static const char digits[] = "0123456789abcdef";
    if (!hex || strlen(hex) != 64) return 0;
    for (unsigned i = 0; i < 32; ++i)
        if (hex[2*i] != digits[bytes[i] >> 4] || hex[2*i+1] != digits[bytes[i] & 15])
            return 0;
    return 1;
}

void upmem_wave_envelope_tile(const upmem_wave_envelope_t *e,
        uint64_t index, upmem_wave_tile_t *tile) {
    memcpy(tile, e->data + UPMEM_WAVE_ENVELOPE_HEADER_BYTES +
        (size_t)e->operation_count * UPMEM_WAVE_OPERATION_BYTES +
        (size_t)index * UPMEM_WAVE_TILE_BYTES, sizeof(*tile));
}

int upmem_wave_envelope_validate(upmem_wave_envelope_t *e,
        const char *binary_sha256, uint32_t dpus, uint32_t tasklets, char **message) {
    const unsigned char *p = e->data;
    upmem_wave_operation_t operations[UPMEM_WAVE_MAX_DPUS];
    unsigned used[UPMEM_WAVE_MAX_DPUS] = {0};
    uint32_t owner[UPMEM_WAVE_MAX_DPUS];
    for (unsigned i = 0; i < UPMEM_WAVE_MAX_DPUS; ++i) owner[i] = UPMEM_WAVE_NO_OPERATION;
    if (e->size > UPMEM_WAVE_ENVELOPE_MAX_BYTES ||
            e->size < UPMEM_WAVE_ENVELOPE_HEADER_BYTES || memcmp(p, "UPWAVE1\0", 8) ||
            le32(p+8) != 1 || le32(p+12) != UPMEM_WAVE_ENVELOPE_HEADER_BYTES ||
            le32(p+16) != dpus || !dpus || dpus > UPMEM_WAVE_MAX_DPUS ||
            le32(p+20) != tasklets || !tasklets || tasklets > UPMEM_WAVE_MAX_TASKLETS ||
            le32(p+24) == 0 || le32(p+24) > dpus || le32(p+28) == 0 ||
            le32(p+32) > UPMEM_WAVE_INT8 || le32(p+36) != 0 ||
            le64(p+56) != e->size || !nonzero_digest(p+72) ||
            !nonzero_digest(p+104) || !digest_matches(p+104, binary_sha256))
        return fail(message, "invalid wave envelope header or executable identity");
    e->dpus = dpus; e->tasklets = tasklets;
    e->operation_count = le32(p+24); e->wave_count = le32(p+28);
    e->numeric_mode = le32(p+32); e->sequence = le64(p+40);
    e->control_count = le64(p+48);
    if (e->control_count != (uint64_t)e->wave_count * dpus)
        return fail(message, "invalid dense wave control count");
    const size_t table_start = UPMEM_WAVE_ENVELOPE_HEADER_BYTES +
        (size_t)e->operation_count * UPMEM_WAVE_OPERATION_BYTES;
    if (table_start > e->size || e->control_count >
            (e->size - table_start) / UPMEM_WAVE_TILE_BYTES)
        return fail(message, "truncated wave descriptor table");
    e->payload_offset = table_start + (size_t)e->control_count * UPMEM_WAVE_TILE_BYTES;
    if (le64(p+64) != e->payload_offset)
        return fail(message, "noncanonical wave payload offset");
    for (uint32_t i = 0; i < e->operation_count; ++i) {
        memcpy(&operations[i], p + UPMEM_WAVE_ENVELOPE_HEADER_BYTES +
            i * UPMEM_WAVE_OPERATION_BYTES, sizeof(operations[i]));
        const upmem_wave_operation_t *o = &operations[i];
        if (!nonzero_digest(o->node_digest) || !nonzero_digest(o->contract_digest) ||
                !o->batch_count || o->batch_count > UINT32_MAX || !o->m || !o->n ||
                !o->k || o->k > UPMEM_WAVE_MAX_K ||
                !isfinite(o->left_scale) || o->left_scale <= 0 ||
                !isfinite(o->right_scale) || o->right_scale <= 0 ||
                (e->numeric_mode == UPMEM_WAVE_FLOAT32 &&
                 (o->left_scale != 1 || o->right_scale != 1)))
            return fail(message, "invalid wave operation geometry, scale or identity");
        for (uint32_t j = 0; j < i; ++j)
            if (!memcmp(o->node_digest, operations[j].node_digest, 32))
                return fail(message, "duplicate wave operation identity");
    }
    size_t cursor = e->payload_offset;
    uint64_t previous_wave = 0, previous_request = 0;
    for (uint32_t w = 0; w < e->wave_count; ++w) {
        upmem_wave_tile_t tiles[UPMEM_WAVE_MAX_DPUS];
        unsigned active = 0;
        for (uint32_t d = 0; d < dpus; ++d) {
            upmem_wave_envelope_tile(e, (uint64_t)w * dpus + d, &tiles[d]);
            const upmem_wave_control_t *c = &tiles[d].control;
            if (!upmem_wave_control_valid(c, d, tasklets) || c->numeric_mode != e->numeric_mode)
                return fail(message, "invalid wave control or DPU ownership");
            if (d == 0) {
                if (w && (c->wave_id <= previous_wave || c->request_sequence <= previous_request))
                    return fail(message, "wave/request identities must increase");
                previous_wave = c->wave_id; previous_request = c->request_sequence;
            } else if (c->wave_id != previous_wave || c->request_sequence != previous_request)
                return fail(message, "wave controls have inconsistent identities");
            if (c->flags == UPMEM_WAVE_IDLE) {
                if (tiles[d].m_offset || tiles[d].n_offset)
                    return fail(message, "idle tile has output offsets");
                continue;
            }
            if (c->operation_index >= e->operation_count)
                return fail(message, "wave control references absent operation");
            if (owner[d] != UPMEM_WAVE_NO_OPERATION && owner[d] != c->operation_index)
                return fail(message, "DPU group changes ownership inside a prepared cohort");
            owner[d] = c->operation_index;
            ++active; used[c->operation_index] = 1;
            const upmem_wave_operation_t *o = &operations[c->operation_index];
            if (c->batch_index >= o->batch_count || tiles[d].m_offset > o->m ||
                    c->m > o->m - tiles[d].m_offset || tiles[d].n_offset > o->n ||
                    c->n > o->n - tiles[d].n_offset || c->k_offset > o->k ||
                    c->k > o->k - c->k_offset)
                return fail(message, "wave tile exceeds canonical operation geometry");
            for (uint32_t j = 0; j < d; ++j) {
                const upmem_wave_control_t *other = &tiles[j].control;
                if (other->flags == UPMEM_WAVE_IDLE || other->operation_index != c->operation_index)
                    continue;
                if (other->tile_id == c->tile_id || (other->batch_index == c->batch_index &&
                        tiles[j].m_offset < tiles[d].m_offset + c->m &&
                        tiles[d].m_offset < tiles[j].m_offset + other->m &&
                        tiles[j].n_offset < tiles[d].n_offset + c->n &&
                        tiles[d].n_offset < tiles[j].n_offset + other->n))
                    return fail(message, "duplicate tile or overlapping wave outputs");
            }
            for (unsigned plane = 0; plane < 4; ++plane) {
                size_t length = c->planes[plane].length;
                if (length > e->size - cursor)
                    return fail(message, "truncated wave input payload");
                if (!length) continue;
                size_t elements = plane < 2 ? (size_t)c->m * c->k : (size_t)c->k * c->n;
                size_t live = elements * (e->numeric_mode == UPMEM_WAVE_FLOAT32 ? 4u : 1u);
                for (size_t i = live; i < length; ++i)
                    if (p[cursor+i]) return fail(message, "nonzero wave input padding");
                for (size_t i = 0; i < elements; ++i) {
                    if (e->numeric_mode == UPMEM_WAVE_INT8) {
                        if (p[cursor+i] == 128) return fail(message, "int8 wave input exceeds symmetric range");
                    } else {
                        float value;
                        memcpy(&value, p+cursor+4*i, sizeof(value));
                        if (!isfinite(value)) return fail(message, "nonfinite wave input");
                    }
                }
                cursor += length;
            }
        }
        if (!active) return fail(message, "wave has no active DPU");
    }
    for (uint32_t i = 0; i < e->operation_count; ++i)
        if (!used[i]) return fail(message, "unused wave operation");
    if (cursor != e->size) return fail(message, "trailing wave payload bytes");
    return 0;
}

int upmem_wave_envelope_open(const char *root, const char *name, const char *sha256,
        const char *binary_sha256, uint32_t dpus, uint32_t tasklets,
        upmem_wave_envelope_t *e, char **message) {
    memset(e, 0, sizeof(*e)); e->fd = -1;
    if (!root || !name || !*name || name[0] == '.' || !sha256 || strlen(sha256) != 64)
        return fail(message, "invalid wave envelope path or digest");
    for (const char *c = name; *c; ++c)
        if (!((*c >= 'a' && *c <= 'z') || (*c >= 'A' && *c <= 'Z') ||
              (*c >= '0' && *c <= '9') || *c == '_' || *c == '-' || *c == '.'))
            return fail(message, "wave envelope must be a session-root basename");
    int dir = open(root, O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    if (dir < 0) return fail(message, "wave session root unavailable");
    e->fd = openat(dir, name, O_RDONLY | O_NOFOLLOW | O_CLOEXEC | O_NONBLOCK);
    close(dir);
    struct stat info;
    if (e->fd < 0 || fstat(e->fd, &info) || !S_ISREG(info.st_mode) || info.st_size <= 0 ||
            (uintmax_t)info.st_size > UPMEM_WAVE_ENVELOPE_MAX_BYTES) goto invalid;
    e->size = (size_t)info.st_size;
    unsigned char *snapshot = malloc(e->size);
    if (!snapshot) goto invalid;
    e->data = snapshot;
    size_t received = 0;
    while (received < e->size) {
        ssize_t count = read(e->fd, snapshot + received, e->size - received);
        if (count <= 0) goto invalid;
        received += (size_t)count;
    }
    unsigned char extra;
    if (read(e->fd, &extra, 1) != 0) goto invalid;
    close(e->fd); e->fd = -1;
    char actual[65];
    if (execution_plan_sha256_bytes(e->data, e->size, actual) || strcmp(actual, sha256))
        goto invalid;
    if (upmem_wave_envelope_validate(e, binary_sha256, dpus, tasklets, message))
        goto invalid;
    return 0;
invalid:
    upmem_wave_envelope_close(e);
    return fail(message, "wave envelope file, digest or validation failed");
}

void upmem_wave_envelope_close(upmem_wave_envelope_t *e) {
    free((void *)e->data);
    if (e->fd >= 0) close(e->fd);
    memset(e, 0, sizeof(*e)); e->fd = -1;
}
