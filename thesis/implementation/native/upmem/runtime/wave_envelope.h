#ifndef UPMEM_WAVE_ENVELOPE_H
#define UPMEM_WAVE_ENVELOPE_H

#include <stddef.h>
#include <stdint.h>
#include "wave_protocol.h"

#define UPMEM_WAVE_ENVELOPE_HEADER_BYTES 136u
#define UPMEM_WAVE_OPERATION_BYTES 112u
#define UPMEM_WAVE_TILE_BYTES 160u
#define UPMEM_WAVE_ENVELOPE_MAX_BYTES (512u * 1024u * 1024u)

typedef struct __attribute__((packed)) {
    unsigned char node_digest[32];
    unsigned char contract_digest[32];
    uint64_t batch_count, m, n, k;
    double left_scale, right_scale;
} upmem_wave_operation_t;

typedef struct __attribute__((packed)) {
    uint64_t m_offset, n_offset;
    upmem_wave_control_t control;
} upmem_wave_tile_t;

typedef struct {
    const unsigned char *data;
    size_t size, payload_offset;
    uint32_t dpus, tasklets, operation_count, wave_count, numeric_mode;
    uint64_t sequence, control_count;
    int fd;
} upmem_wave_envelope_t;

_Static_assert(sizeof(upmem_wave_operation_t) == UPMEM_WAVE_OPERATION_BYTES,
    "wave operation wire layout drift");
_Static_assert(sizeof(upmem_wave_tile_t) == UPMEM_WAVE_TILE_BYTES,
    "wave tile wire layout drift");

/* Snapshot admission limit, not a DPU geometry limit. Files may be removed after load. */
int upmem_wave_envelope_open(const char *root, const char *name, const char *sha256,
    const char *binary_sha256, uint32_t dpus, uint32_t tasklets,
    upmem_wave_envelope_t *envelope, char **error_message);
void upmem_wave_envelope_close(upmem_wave_envelope_t *envelope);
void upmem_wave_envelope_tile(const upmem_wave_envelope_t *envelope,
    uint64_t index, upmem_wave_tile_t *tile);
int upmem_wave_envelope_validate(upmem_wave_envelope_t *envelope,
    const char *binary_sha256, uint32_t dpus, uint32_t tasklets,
    char **error_message);

#endif
