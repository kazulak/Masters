#ifndef UPMEM_WAVE_PROTOCOL_H
#define UPMEM_WAVE_PROTOCOL_H

#include <stdint.h>

/* Private v5 launch contract. Not selected by the active v4 runtime yet. */
#if !defined(__BYTE_ORDER__) || __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
#error "UPMEM wave controls require little-endian host and DPU builds"
#endif

#define UPMEM_WAVE_VERSION 5u
#define UPMEM_WAVE_CONTROL_MAGIC 0x35574354u
#define UPMEM_WAVE_COMPLETION_MAGIC 0x35574350u
#define UPMEM_WAVE_MRAM_BYTES (512u * 1024u)
#define UPMEM_WAVE_MAX_DPUS 64u
#define UPMEM_WAVE_MAX_TASKLETS 24u
#define UPMEM_WAVE_MAX_K 65536u
#define UPMEM_WAVE_INT8_COMPONENT_PRODUCT (2u * 127u * 127u)
#define UPMEM_WAVE_IDLE 1u
#define UPMEM_WAVE_NO_OPERATION UINT32_MAX
#define UPMEM_WAVE_KERNEL_NONE 0u
#define UPMEM_WAVE_KERNEL_REAL_PANEL 1u
#define UPMEM_WAVE_KERNEL_FOUR_PRODUCT_PANEL 2u
#define UPMEM_WAVE_FLOAT32 0u
#define UPMEM_WAVE_INT8 1u
#define UPMEM_WAVE_PENDING 0u
#define UPMEM_WAVE_COMPLETED 1u
#define UPMEM_WAVE_FAILED 2u
#define UPMEM_WAVE_FAILURE_NONE 0u
#define UPMEM_WAVE_FAILURE_VALIDATION 1u
#define UPMEM_WAVE_FAILURE_EXECUTION 2u
#define UPMEM_WAVE_NO_PRODUCT UINT32_MAX

enum upmem_wave_plane {
    UPMEM_WAVE_A_REAL = 0,
    UPMEM_WAVE_A_IMAG = 1,
    UPMEM_WAVE_B_REAL = 2,
    UPMEM_WAVE_B_IMAG = 3,
    UPMEM_WAVE_RR = 4,
    UPMEM_WAVE_II = 5,
    UPMEM_WAVE_RI = 6,
    UPMEM_WAVE_IR = 7,
    UPMEM_WAVE_PLANE_COUNT = 8
};

typedef struct __attribute__((packed)) {
    uint32_t offset;
    uint32_t length;
} upmem_wave_span_t;

typedef struct __attribute__((packed)) {
    uint32_t magic;
    uint32_t version;
    uint32_t dpu_id;
    uint32_t tasklets;
    uint32_t flags;
    uint32_t numeric_mode;
    uint32_t kernel;
    uint32_t operation_index;
    uint64_t wave_id;
    uint64_t request_sequence;
    uint64_t tile_id;
    uint32_t batch_index;
    uint32_t m;
    uint32_t n;
    uint32_t k;
    uint32_t k_offset;
    uint32_t reserved;
    upmem_wave_span_t planes[UPMEM_WAVE_PLANE_COUNT];
} upmem_wave_control_t;

typedef struct __attribute__((packed)) {
    uint32_t magic;
    uint32_t version;
    uint32_t status;
    uint32_t dpu_id;
    uint32_t operation_index;
    uint32_t completed_product_mask;
    uint64_t wave_id;
    uint64_t request_sequence;
    uint64_t tile_id;
    uint64_t cycles;
    uint64_t processed_elements;
    uint32_t failure_stage;
    uint32_t failing_product;
} upmem_wave_completion_t;

_Static_assert(sizeof(upmem_wave_span_t) == 8u, "wave span ABI drift");
_Static_assert(sizeof(upmem_wave_control_t) == 144u, "wave control ABI drift");
_Static_assert(sizeof(upmem_wave_completion_t) == 72u, "wave completion ABI drift");
_Static_assert((uint64_t)UPMEM_WAVE_MAX_K * UPMEM_WAVE_INT8_COMPONENT_PRODUCT
    <= INT32_MAX, "wave int8 component bound exceeds int32");

/* Called by the host and tasklet zero before any MRAM access. */
static inline int upmem_wave_control_valid(const upmem_wave_control_t *c,
        uint32_t expected_dpu, uint32_t expected_tasklets) {
    uint32_t sizes[UPMEM_WAVE_PLANE_COUNT] = {0};
    if (c->magic != UPMEM_WAVE_CONTROL_MAGIC || c->version != UPMEM_WAVE_VERSION ||
            c->reserved != 0u || c->dpu_id >= UPMEM_WAVE_MAX_DPUS ||
            c->dpu_id != expected_dpu || c->tasklets != expected_tasklets ||
            c->tasklets < 1u || c->tasklets > UPMEM_WAVE_MAX_TASKLETS ||
            c->numeric_mode > UPMEM_WAVE_INT8 || c->flags > UPMEM_WAVE_IDLE) {
        return 0;
    }
    if (c->flags == UPMEM_WAVE_IDLE) {
        if (c->kernel != UPMEM_WAVE_KERNEL_NONE ||
                c->operation_index != UPMEM_WAVE_NO_OPERATION ||
                c->tile_id != 0u || c->batch_index != 0u || c->m != 0u ||
                c->n != 0u || c->k != 0u || c->k_offset != 0u) return 0;
        for (uint32_t i = 0; i < UPMEM_WAVE_PLANE_COUNT; ++i) {
            if (c->planes[i].offset != 0u || c->planes[i].length != 0u) return 0;
        }
        return 1;
    }
    if (c->operation_index >= UPMEM_WAVE_MAX_DPUS ||
            (c->kernel != UPMEM_WAVE_KERNEL_REAL_PANEL &&
             c->kernel != UPMEM_WAVE_KERNEL_FOUR_PRODUCT_PANEL) ||
            c->m == 0u || c->m > 256u || c->n == 0u || c->n > 256u ||
            c->k == 0u || c->k > UPMEM_WAVE_MAX_K ||
            (uint64_t)c->k * UPMEM_WAVE_INT8_COMPONENT_PRODUCT > INT32_MAX ||
            c->k_offset > UPMEM_WAVE_MAX_K - c->k) return 0;
    const uint64_t e = c->numeric_mode == UPMEM_WAVE_FLOAT32 ? 4u : 1u;
    const uint64_t a = ((uint64_t)c->m * c->k * e + 7u) & ~UINT64_C(7);
    const uint64_t b = ((uint64_t)c->k * c->n * e + 7u) & ~UINT64_C(7);
    const uint64_t out = ((uint64_t)c->m * c->n * 4u + 7u) & ~UINT64_C(7);
    const uint64_t total = c->kernel == UPMEM_WAVE_KERNEL_FOUR_PRODUCT_PANEL
        ? 2u * a + 2u * b + 4u * out : a + b + out;
    if (total > UPMEM_WAVE_MRAM_BYTES) return 0;
    sizes[UPMEM_WAVE_A_REAL] = (uint32_t)a;
    sizes[UPMEM_WAVE_B_REAL] = (uint32_t)b;
    sizes[UPMEM_WAVE_RR] = (uint32_t)out;
    if (c->kernel == UPMEM_WAVE_KERNEL_FOUR_PRODUCT_PANEL) {
        sizes[UPMEM_WAVE_A_IMAG] = (uint32_t)a;
        sizes[UPMEM_WAVE_B_IMAG] = (uint32_t)b;
        sizes[UPMEM_WAVE_II] = sizes[UPMEM_WAVE_RI] = sizes[UPMEM_WAVE_IR] = (uint32_t)out;
    }
    for (uint32_t i = 0; i < UPMEM_WAVE_PLANE_COUNT; ++i) {
        const upmem_wave_span_t span = c->planes[i];
        if (span.length != sizes[i] || (span.length == 0u && span.offset != 0u) ||
                span.offset % 8u != 0u || span.offset > UPMEM_WAVE_MRAM_BYTES ||
                span.length > UPMEM_WAVE_MRAM_BYTES - span.offset) return 0;
    }
    for (uint32_t i = 0; i < UPMEM_WAVE_PLANE_COUNT; ++i) {
        for (uint32_t j = 0; j < i; ++j) {
            const upmem_wave_span_t a_span = c->planes[i], b_span = c->planes[j];
            if (a_span.length && b_span.length &&
                    a_span.offset < b_span.offset + b_span.length &&
                    b_span.offset < a_span.offset + a_span.length) return 0;
        }
    }
    return 1;
}

#endif
