#include <dpu.h>
#include <dpu_types.h>
#include <pidcomm.h>

#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define PAYLOAD_BYTES 256u
#define ELEMENTS (PAYLOAD_BYTES / sizeof(int32_t))
#define START_OFFSET 0u
#define TARGET_OFFSET 0u
#define BUFFER_OFFSET (32u * 1024u * 1024u)

static void fill_input(int32_t *values, uint32_t dpu_count) {
    for (uint32_t dpu = 0; dpu < dpu_count; ++dpu) {
        for (uint32_t element = 0; element < ELEMENTS; ++element) {
            values[dpu * ELEMENTS + element] = (int32_t)(dpu + element);
        }
    }
}

static int expected_value(uint32_t dpu_count, uint32_t element) {
    int64_t result = 0;
    for (uint32_t dpu = 0; dpu < dpu_count; ++dpu) {
        result += (int64_t)dpu + element;
    }
    return (int32_t)result;
}

static void topology_for_count(uint32_t dpu_count, uint32_t axis_len[3], char comm[4]) {
    axis_len[0] = 1;
    axis_len[1] = 1;
    axis_len[2] = 1;
    memcpy(comm, "000", 4);
    if (dpu_count == 2) {
        axis_len[0] = 2;
        comm[0] = '1';
    } else if (dpu_count == 4) {
        axis_len[0] = 2;
        axis_len[1] = 2;
        comm[0] = '1';
        comm[1] = '1';
    } else if (dpu_count == 64) {
        axis_len[0] = 8;
        axis_len[1] = 8;
        comm[0] = '1';
        comm[1] = '1';
    }
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s <dpu-count>\n", argv[0]);
        return 2;
    }

    char *end = NULL;
    unsigned long parsed = strtoul(argv[1], &end, 10);
    if (end == argv[1] || *end != '\0' || parsed > UINT32_MAX) {
        fprintf(stderr, "invalid dpu count: %s\n", argv[1]);
        return 2;
    }

    uint32_t dpu_count = (uint32_t)parsed;
    uint32_t dimension = 3;
    uint32_t axis_len[3];
    char comm[4];
    if (dpu_count != 2 && dpu_count != 4 && dpu_count != 64) {
        fprintf(stderr, "unsupported candidate dpu count: %" PRIu32 "\n", dpu_count);
        return 2;
    }
    topology_for_count(dpu_count, axis_len, comm);

    const char *backend = getenv("DPU_BACKEND");
    if (backend == NULL || strcmp(backend, "hw") != 0) {
        fprintf(stderr, "physical hardware backend is required\n");
        return 1;
    }

    struct dpu_set_t dpu_set;
    DPU_ASSERT(dpu_alloc_comm(dpu_count, NULL, &dpu_set, 1));
    uint32_t observed_dpu_count = 0;
    DPU_ASSERT(dpu_get_nr_dpus(dpu_set, &observed_dpu_count));
    if (observed_dpu_count != dpu_count) {
        fprintf(stderr, "hardware topology mismatch observed=%" PRIu32 " expected=%" PRIu32 "\n", observed_dpu_count, dpu_count);
        DPU_ASSERT(dpu_free(dpu_set));
        return 1;
    }
    DPU_ASSERT(dpu_load(dpu_set, "pidcomm_bin/dpu_user", NULL));

    hypercube_manager *manager = init_hypercube_manager(dpu_set, dimension, axis_len);
    if (manager == NULL) {
        fprintf(stderr, "PID-Comm manager initialization returned NULL\n");
        DPU_ASSERT(dpu_free(dpu_set));
        return 1;
    }

    int32_t *values = calloc((size_t)dpu_count * ELEMENTS, sizeof(*values));
    if (values == NULL) {
        fprintf(stderr, "input allocation failed\n");
        free(manager);
        DPU_ASSERT(dpu_free(dpu_set));
        return 1;
    }
    fill_input(values, dpu_count);

    struct dpu_set_t dpu;
    uint32_t each_dpu;
    DPU_FOREACH_ENTANGLED_GROUP(dpu_set, dpu, each_dpu, dpu_count) {
        DPU_ASSERT(dpu_prepare_xfer(dpu, values + each_dpu * ELEMENTS));
    }
    DPU_ASSERT(dpu_push_xfer(
        dpu_set, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME, START_OFFSET,
        PAYLOAD_BYTES, DPU_XFER_DEFAULT));

    /* This is the pinned PID-Comm all-reduce entry point, with reduction type 0 for sum. */
    pidcomm_all_reduce(
        manager, comm, PAYLOAD_BYTES, START_OFFSET, TARGET_OFFSET, BUFFER_OFFSET,
        sizeof(int32_t), 0);

    DPU_FOREACH_ENTANGLED_GROUP(dpu_set, dpu, each_dpu, dpu_count) {
        DPU_ASSERT(dpu_prepare_xfer(dpu, values + each_dpu * ELEMENTS));
    }
    DPU_ASSERT(dpu_push_xfer(
        dpu_set, DPU_XFER_FROM_DPU, DPU_MRAM_HEAP_POINTER_NAME, TARGET_OFFSET,
        PAYLOAD_BYTES, DPU_XFER_DEFAULT));

    for (uint32_t dpu_id = 0; dpu_id < dpu_count; ++dpu_id) {
        for (uint32_t element = 0; element < ELEMENTS; ++element) {
            int32_t observed = values[dpu_id * ELEMENTS + element];
            int expected = expected_value(dpu_count, element);
            if (observed != expected) {
                fprintf(stderr, "all-reduce mismatch dpu=%" PRIu32 " element=%" PRIu32 " observed=%" PRId32 " expected=%d\n", dpu_id, element, observed, expected);
                free(values);
                free(manager);
                DPU_ASSERT(dpu_free(dpu_set));
                return 1;
            }
        }
    }

    printf(
        "{\"status\":\"passed\",\"dpu_count\":%" PRIu32 ",\"payload_bytes\":%u,"
        "\"payload_dtype\":\"int32\",\"operation\":\"sum_all_reduce\","
        "\"topology\":{\"dimension\":3,\"axis_lengths\":[%" PRIu32 ",%" PRIu32 ",%" PRIu32 "],\"communicator\":\"%s\"},"
        "\"hardware_observed\":true,\"fallback\":false,\"pidcomm_api\":\"pidcomm_all_reduce\"}\n",
        dpu_count, PAYLOAD_BYTES, axis_len[0], axis_len[1], axis_len[2], comm);
    free(values);
    free(manager);
    DPU_ASSERT(dpu_free(dpu_set));
    return 0;
}
