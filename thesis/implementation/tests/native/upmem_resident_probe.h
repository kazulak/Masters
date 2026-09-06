#ifndef UPMEM_RESIDENT_PROBE_H
#define UPMEM_RESIDENT_PROBE_H

/* Test-only fixed pair; not an extension to the production DPU ABI. */
#include <stddef.h>
#include "wave_protocol.h"

#define RESIDENT_PROBE_VERSION 1u
#define RESIDENT_FIRST 1u
#define RESIDENT_HOST_SECOND 2u
#define RESIDENT_LOCAL_SECOND 3u

typedef struct __attribute__((packed)) {
    uint32_t version;
    uint32_t resident_side; /* 0: left, 1: right */
    uint64_t pair_id;
    upmem_wave_control_t controls[2];
    upmem_wave_span_t retained[2];
} resident_probe_plan_t;

_Static_assert(sizeof(resident_probe_plan_t) == 320u, "resident probe layout drift");
_Static_assert(__builtin_offsetof(resident_probe_plan_t, controls) == 16u, "resident controls offset");
_Static_assert(__builtin_offsetof(resident_probe_plan_t, retained) == 304u, "resident spans offset");

static inline int resident_probe_advance(uint32_t *cursor, upmem_wave_span_t span) {
    if (*cursor > UPMEM_WAVE_MRAM_BYTES || span.offset != *cursor ||
            span.length > UPMEM_WAVE_MRAM_BYTES - *cursor) return 0;
    *cursor += span.length;
    return 1;
}

static inline int resident_probe_plan_valid(const resident_probe_plan_t *p, uint32_t tasklets) {
    if (p->version != RESIDENT_PROBE_VERSION || p->resident_side > 1u || p->pair_id == 0u) return 0;
    for (uint32_t i = 0u; i < 2u; ++i) {
        const upmem_wave_control_t *c = &p->controls[i];
        if (!upmem_wave_control_valid(c, 0u, tasklets) || c->flags != 0u ||
                c->numeric_mode != UPMEM_WAVE_FLOAT32 || c->kernel != UPMEM_WAVE_KERNEL_FOUR_PRODUCT_PANEL ||
                c->operation_index != i || c->wave_id != i || c->tile_id != i ||
                c->request_sequence != p->pair_id || c->batch_index != 0u || c->k_offset != 0u) return 0;
    }
    const upmem_wave_control_t *a = &p->controls[0], *b = &p->controls[1];
    const uint64_t elements = (uint64_t)a->m * a->n;
    const uint64_t consumed = p->resident_side == 0u ? (uint64_t)b->m * b->k : (uint64_t)b->k * b->n;
    if (elements != consumed) return 0;
    const uint32_t span_bytes = (uint32_t)((elements * 4u + 7u) & ~UINT64_C(7));
    uint32_t cursor = 0u;
    for (uint32_t i = 0u; i < 8u; ++i)
        if (!resident_probe_advance(&cursor, a->planes[i])) return 0;
    for (uint32_t i = 0u; i < 2u; ++i)
        if (p->retained[i].length != span_bytes || !resident_probe_advance(&cursor, p->retained[i])) return 0;
    const uint32_t resident_start = 2u * p->resident_side;
    for (uint32_t i = 0u; i < 8u; ++i) {
        if (i == resident_start || i == resident_start + 1u) {
            if (b->planes[i].offset != p->retained[i - resident_start].offset ||
                    b->planes[i].length != p->retained[i - resident_start].length) return 0;
        } else if (!resident_probe_advance(&cursor, b->planes[i])) return 0;
    }
    return 1;
}

#endif
