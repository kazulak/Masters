#define _POSIX_C_SOURCE 200809L

#include "request.h"

#include "plan.h"

#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

static void v4_error(char **message, const char *value) {
    if (message != NULL && *message == NULL) *message = strdup(value);
}

static uint32_t align8_u32(uint64_t value) {
    if (value > UINT32_MAX - 7u) return 0u;
    return (uint32_t)((value + 7u) & ~UINT64_C(7));
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

static int digest_is_zero(const unsigned char digest[32]) {
    unsigned char value = 0u;
    for (uint32_t index = 0u; index < 32u; index++) value |= digest[index];
    return value == 0u;
}

static int read_file(const char *path, unsigned char **data, size_t *length, size_t maximum) {
    FILE *file = path == NULL ? NULL : fopen(path, "rb");
    long size;
    if (file == NULL || fseek(file, 0, SEEK_END) != 0) {
        if (file != NULL) fclose(file);
        return 1;
    }
    size = ftell(file);
    if (size < 0 || (uintmax_t)size > (uintmax_t)maximum || fseek(file, 0, SEEK_SET) != 0) {
        fclose(file);
        return 1;
    }
    *data = (unsigned char *)malloc((size_t)size + 1u);
    if (*data == NULL) {
        fclose(file);
        return 1;
    }
    if (fread(*data, 1u, (size_t)size, file) != (size_t)size || fclose(file) != 0) {
        free(*data);
        *data = NULL;
        return 1;
    }
    (*data)[size] = 0u;
    *length = (size_t)size;
    return 0;
}

static int path_component_is_parent(const char *cursor) {
    const char *start = cursor;
    while (*cursor != '\0' && *cursor != '/') cursor++;
    return (size_t)(cursor - start) == 2u && start[0] == '.' && start[1] == '.';
}

static int relative_path_is_safe(const char *path) {
    const char *cursor;
    if (path == NULL || path[0] == '\0' || path[0] == '/' || path[0] == '\\') return 0;
    cursor = path;
    while (*cursor != '\0') {
        if (path_component_is_parent(cursor)) return 0;
        while (*cursor != '\0' && *cursor != '/') cursor++;
        while (*cursor == '/') cursor++;
    }
    return 1;
}

static int path_inside_root(const char *root, const char *candidate) {
    size_t root_length = strlen(root);
    return strncmp(root, candidate, root_length) == 0 &&
        (candidate[root_length] == '\0' || candidate[root_length] == '/');
}

static int resolve_path(
    const char *root,
    const char *relative,
    int must_exist,
    char output[PATH_MAX]
) {
    char joined[PATH_MAX];
    char resolved[PATH_MAX];
    char parent[PATH_MAX];
    char *slash;
    if (!relative_path_is_safe(relative) ||
        snprintf(joined, sizeof(joined), "%s/%s", root, relative) >= (int)sizeof(joined)) return 1;
    if (must_exist) {
        if (realpath(joined, resolved) == NULL || !path_inside_root(root, resolved)) return 1;
        if (snprintf(output, PATH_MAX, "%s", resolved) >= PATH_MAX) return 1;
        return 0;
    }
    if (snprintf(parent, sizeof(parent), "%s", joined) >= (int)sizeof(parent)) return 1;
    slash = strrchr(parent, '/');
    if (slash == NULL) return 1;
    *slash = '\0';
    if (realpath(parent, resolved) == NULL || !path_inside_root(root, resolved)) return 1;
    if (snprintf(output, PATH_MAX, "%s/%s", resolved, slash + 1) >= PATH_MAX) return 1;
    return 0;
}

static char *trim(char *value) {
    char *end;
    while (*value == ' ' || *value == '\t' || *value == '\r' || *value == '\n') value++;
    end = value + strlen(value);
    while (end > value && (end[-1] == ' ' || end[-1] == '\t' || end[-1] == '\r' || end[-1] == '\n')) --end;
    *end = '\0';
    return value;
}

static int parse_manifest(
    const char *root,
    const unsigned char *contents,
    size_t length,
    char sidecar_relative[PATH_MAX],
    execution_plan_v4_request_item_t items[EXECUTION_PLAN_V4_MAX_DPUS],
    uint32_t *item_count,
    char **error_message
) {
    char *copy = (char *)malloc(length + 1u);
    char *save = NULL;
    char *line;
    if (copy == NULL) {
        v4_error(error_message, "request_manifest_failed: manifest allocation failed");
        return 1;
    }
    memcpy(copy, contents, length);
    copy[length] = '\0';
    *item_count = 0u;
    sidecar_relative[0] = '\0';
    for (line = strtok_r(copy, "\n", &save); line != NULL; line = strtok_r(NULL, "\n", &save)) {
        char *value = trim(line);
        unsigned int dpu_id;
        unsigned long long tile_id;
        char a_path[PATH_MAX], b_path[PATH_MAX], c_path[PATH_MAX];
        char a_sha256[65], b_sha256[65];
        unsigned char digest[32];
        if (*value == '\0' || *value == '#') continue;
        if (strncmp(value, "sidecar", 7u) == 0 &&
            (value[7] == ' ' || value[7] == '\t' || value[7] == '=')) {
            char *sidecar = trim(value + 7u);
            if (*sidecar == '=') sidecar = trim(sidecar + 1u);
            if (resolve_path(root, sidecar, 1, sidecar_relative) != 0 || sidecar_relative[0] == '\0') {
                v4_error(error_message, "request_manifest_failed: sidecar path is unsafe or missing");
                free(copy);
                return 1;
            }
            continue;
        }
        if (strncmp(value, "dpu", 3u) != 0 ||
            (value[3] != ' ' && value[3] != '\t')) {
            v4_error(error_message, "request_manifest_failed: expected sidecar or dpu record");
            free(copy);
            return 1;
        }
        if (*item_count >= EXECUTION_PLAN_V4_MAX_DPUS ||
            sscanf(value + 3u, "%u %llu %4095s %4095s %4095s %64s %64s", &dpu_id, &tile_id,
                a_path, b_path, c_path, a_sha256, b_sha256) != 7 ||
            dpu_id >= EXECUTION_PLAN_V4_MAX_DPUS ||
            digest_text(a_sha256, digest) != 0 || digest_text(b_sha256, digest) != 0) {
            v4_error(error_message, "request_manifest_failed: invalid dpu record");
            free(copy);
            return 1;
        }
        if (items[dpu_id].a_path != NULL) {
            v4_error(error_message, "hardware_profile_violation: duplicate local DPU ID");
            free(copy);
            return 1;
        }
        items[dpu_id].work_unit.local_dpu_id = dpu_id;
        items[dpu_id].work_unit.tile_id = (uint64_t)tile_id;
        if (resolve_path(root, a_path, 1, a_path) != 0 ||
            resolve_path(root, b_path, 1, b_path) != 0 ||
            resolve_path(root, c_path, 0, c_path) != 0) {
            v4_error(error_message, "request_manifest_failed: payload path is unsafe");
            free(copy);
            return 1;
        }
        items[dpu_id].a_path = strdup(a_path);
        items[dpu_id].b_path = strdup(b_path);
        items[dpu_id].c_path = strdup(c_path);
        if (items[dpu_id].a_path == NULL || items[dpu_id].b_path == NULL || items[dpu_id].c_path == NULL) {
            v4_error(error_message, "request_manifest_failed: payload path allocation failed");
            free(copy);
            return 1;
        }
        memcpy(items[dpu_id].a_sha256, a_sha256, sizeof(items[dpu_id].a_sha256));
        memcpy(items[dpu_id].b_sha256, b_sha256, sizeof(items[dpu_id].b_sha256));
        (*item_count)++;
    }
    free(copy);
    if (sidecar_relative[0] == '\0' || *item_count == 0u) {
        v4_error(error_message, "request_manifest_failed: manifest has no sidecar or DPU records");
        return 1;
    }
    return 0;
}

static int ranges_overlap(uint32_t a_offset, uint32_t a_bytes, uint32_t b_offset, uint32_t b_bytes) {
    uint64_t a_end = (uint64_t)a_offset + a_bytes;
    uint64_t b_end = (uint64_t)b_offset + b_bytes;
    return a_bytes != 0u && b_bytes != 0u && a_offset < b_end && b_offset < a_end;
}

static int extent_inside(uint64_t offset, uint64_t length, uint64_t limit) {
    return offset <= limit && length <= limit - offset;
}

static int ranges_overlap_u64(uint64_t left_offset, uint64_t left_length,
    uint64_t right_offset, uint64_t right_length) {
    if (left_length == 0u || right_length == 0u) return 0;
    if (left_offset <= right_offset) return right_offset - left_offset < left_length;
    return left_offset - right_offset < right_length;
}

static int validate_work_units(
    const execution_plan_v4_header_t *header,
    const execution_plan_v4_work_unit_t *units,
    char **error_message
) {
    unsigned char seen[EXECUTION_PLAN_V4_MAX_DPUS] = {0};
    uint64_t area = 0u;
    const uint32_t element_bytes = header->numeric_mode == EXECUTION_PLAN_V4_NUMERIC_FLOAT32 ? 4u : 1u;
    for (uint32_t index = 0u; index < header->work_unit_count; index++) {
        const execution_plan_v4_work_unit_t *unit = &units[index];
        uint64_t a_elements, b_elements, c_elements;
        uint32_t expected_a, expected_b, expected_c;
        if (unit->local_dpu_id >= header->dpu_count || seen[unit->local_dpu_id] ||
            (index != 0u && unit->local_dpu_id <= units[index - 1u].local_dpu_id) ||
            unit->batch_index >= header->canonical_batch_count ||
            (unit->flags & ~EXECUTION_PLAN_V4_FLAG_ZERO_WORK) != 0u) {
            v4_error(error_message, "hardware_profile_violation: v4 DPU IDs or flags are invalid");
            return 1;
        }
        seen[unit->local_dpu_id] = 1u;
        if ((unit->flags & EXECUTION_PLAN_V4_FLAG_ZERO_WORK) != 0u) {
            if (unit->m_elements != 0u || unit->n_elements != 0u || unit->k_elements != 0u ||
                unit->a_transfer_bytes != 0u || unit->b_transfer_bytes != 0u || unit->c_transfer_bytes != 0u) {
                v4_error(error_message, "hardware_profile_violation: zero-work unit has payload or extents");
                return 1;
            }
            continue;
        }
        if (unit->m_elements == 0u || unit->n_elements == 0u || unit->k_elements == 0u ||
            !extent_inside(unit->k_offset, unit->k_elements, header->canonical_k) ||
            !extent_inside(unit->m_offset, unit->m_elements, header->canonical_m) ||
            !extent_inside(unit->n_offset, unit->n_elements, header->canonical_n) ||
            unit->k_elements > EXECUTION_PLAN_V4_MAX_CONTRACTED ||
            (uint64_t)unit->k_elements * 128u * 128u > 2147483647u) {
            v4_error(error_message, "hardware_profile_violation: v4 tile extents are outside the profile");
            return 1;
        }
        a_elements = (uint64_t)unit->m_elements * unit->k_elements;
        b_elements = (uint64_t)unit->k_elements * unit->n_elements;
        c_elements = (uint64_t)unit->m_elements * unit->n_elements;
        if (a_elements > UINT64_MAX / element_bytes || b_elements > UINT64_MAX / element_bytes ||
            c_elements > UINT64_MAX / sizeof(int32_t)) {
            v4_error(error_message, "hardware_profile_violation: v4 tile byte count overflow");
            return 1;
        }
        expected_a = align8_u32(a_elements * element_bytes);
        expected_b = align8_u32(b_elements * element_bytes);
        expected_c = align8_u32(c_elements * sizeof(int32_t));
        if (expected_a == 0u || expected_b == 0u || expected_c == 0u ||
            unit->a_transfer_bytes != expected_a || unit->b_transfer_bytes != expected_b ||
            unit->c_transfer_bytes != expected_c || unit->a_offset_bytes % 8u != 0u ||
            unit->b_offset_bytes % 8u != 0u || unit->c_offset_bytes % 8u != 0u ||
            (uint64_t)unit->a_offset_bytes + unit->a_transfer_bytes > EXECUTION_PLAN_V4_MRAM_POOL_BYTES ||
            (uint64_t)unit->b_offset_bytes + unit->b_transfer_bytes > EXECUTION_PLAN_V4_MRAM_POOL_BYTES ||
            (uint64_t)unit->c_offset_bytes + unit->c_transfer_bytes > EXECUTION_PLAN_V4_MRAM_POOL_BYTES ||
            ranges_overlap(unit->a_offset_bytes, unit->a_transfer_bytes, unit->b_offset_bytes, unit->b_transfer_bytes) ||
            ranges_overlap(unit->a_offset_bytes, unit->a_transfer_bytes, unit->c_offset_bytes, unit->c_transfer_bytes) ||
            ranges_overlap(unit->b_offset_bytes, unit->b_transfer_bytes, unit->c_offset_bytes, unit->c_transfer_bytes)) {
            v4_error(error_message, "hardware_profile_violation: v4 tile storage is invalid or exceeds MRAM");
            return 1;
        }
        area += (uint64_t)unit->m_elements * unit->n_elements;
        for (uint32_t prior = 0u; prior < index; prior++) {
            const execution_plan_v4_work_unit_t *other = &units[prior];
            if ((other->flags & EXECUTION_PLAN_V4_FLAG_ZERO_WORK) != 0u || other->batch_index != unit->batch_index) continue;
            if (ranges_overlap_u64(unit->m_offset, unit->m_elements,
                    other->m_offset, other->m_elements) &&
                ranges_overlap_u64(unit->n_offset, unit->n_elements,
                    other->n_offset, other->n_elements)) {
                v4_error(error_message, "hardware_profile_violation: v4 output tiles overlap");
                return 1;
            }
        }
    }
    for (uint32_t dpu = 0u; dpu < header->dpu_count; dpu++) {
        if (!seen[dpu]) {
            v4_error(error_message, "hardware_profile_violation: v4 DPU IDs are not dense");
            return 1;
        }
    }
    if (area != header->request_output_elements) {
        v4_error(error_message, "hardware_profile_violation: v4 request tiles do not cover the request output");
        return 1;
    }
    return 0;
}

int execution_plan_v4_request_load(
    const char *session_root,
    const char *manifest_relative_path,
    const char *submitted_manifest_sha256,
    uint32_t expected_dpus,
    uint32_t expected_tasklets,
    execution_plan_v4_request_t *request,
    char **error_message
) {
    unsigned char *manifest = NULL;
    unsigned char *sidecar = NULL;
    size_t manifest_length = 0u, sidecar_length = 0u;
    char root[PATH_MAX], manifest_path[PATH_MAX], sidecar_relative[PATH_MAX];
    unsigned char request_digest[32];
    if (request == NULL || session_root == NULL || manifest_relative_path == NULL ||
        submitted_manifest_sha256 == NULL || expected_dpus == 0u || expected_dpus > EXECUTION_PLAN_V4_MAX_DPUS ||
        expected_tasklets == 0u || expected_tasklets > EXECUTION_PLAN_V4_MAX_TASKLETS) {
        v4_error(error_message, "request_manifest_failed: invalid request arguments");
        return 1;
    }
    memset(request, 0, sizeof(*request));
    if (realpath(session_root, root) == NULL ||
        stat(root, &(struct stat){0}) != 0 ||
        resolve_path(root, manifest_relative_path, 1, manifest_path) != 0 ||
        read_file(manifest_path, &manifest, &manifest_length, 256u * 1024u) != 0 ||
        execution_plan_sha256_bytes(manifest, manifest_length, request->manifest_sha256) != 0 ||
        strcmp(request->manifest_sha256, submitted_manifest_sha256) != 0 ||
        digest_text(request->manifest_sha256, request_digest) != 0) {
        v4_error(error_message, "request_manifest_failed: manifest path or SHA-256 is invalid");
        free(manifest);
        return 1;
    }
    memcpy(request->root_path, root, strlen(root) + 1u);
    request->manifest_path = strdup(manifest_path);
    {
        execution_plan_v4_request_item_t items[EXECUTION_PLAN_V4_MAX_DPUS] = {0};
        uint32_t item_count = 0u;
        if (parse_manifest(root, manifest, manifest_length, sidecar_relative, items, &item_count, error_message) != 0) {
            free(manifest);
            return 1;
        }
        request->items = (execution_plan_v4_request_item_t *)calloc(expected_dpus, sizeof(*request->items));
        if (request->items == NULL || item_count != expected_dpus) {
            v4_error(error_message, "hardware_profile_violation: manifest DPU count differs from session");
            free(manifest);
            for (uint32_t index = 0u; index < EXECUTION_PLAN_V4_MAX_DPUS; index++) {
                free(items[index].a_path); free(items[index].b_path); free(items[index].c_path);
            }
            return 1;
        }
        for (uint32_t index = 0u; index < expected_dpus; index++) {
            if (items[index].a_path == NULL) {
                v4_error(error_message, "hardware_profile_violation: manifest local DPU IDs are not dense");
                free(manifest);
                for (uint32_t item = 0u; item < EXECUTION_PLAN_V4_MAX_DPUS; item++) {
                    free(items[item].a_path); free(items[item].b_path); free(items[item].c_path);
                }
                free(request->items);
                request->items = NULL;
                return 1;
            }
            request->items[index] = items[index];
        }
        request->item_count = item_count;
    }
    request->sidecar_path = strdup(sidecar_relative);
    if (request->sidecar_path == NULL || read_file(sidecar_relative, &sidecar, &sidecar_length,
        EXECUTION_PLAN_V4_MAX_REQUEST_BYTES) != 0 || sidecar_length < sizeof(request->header)) {
        v4_error(error_message, "sidecar_validation_failed: v4 sidecar is unreadable");
        free(manifest); free(sidecar);
        return 1;
    }
    memcpy(&request->header, sidecar, sizeof(request->header));
    if (memcmp(request->header.magic, EXECUTION_PLAN_V4_MAGIC, 7u) != 0 || request->header.magic[7] != '\0' ||
        request->header.version != EXECUTION_PLAN_V4_VERSION || request->header.header_bytes != sizeof(request->header) ||
        request->header.record_bytes != sizeof(execution_plan_v4_work_unit_t) ||
        request->header.work_unit_count != expected_dpus || request->header.dpu_count != expected_dpus ||
        request->header.tasklets_per_dpu != expected_tasklets || request->header.partition_mode != EXECUTION_PLAN_V4_PARTITION_OUTPUT_TILE ||
        (request->header.numeric_mode != EXECUTION_PLAN_V4_NUMERIC_FLOAT32 &&
         request->header.numeric_mode != EXECUTION_PLAN_V4_NUMERIC_HOST_PACKED_INT8) ||
        request->header.canonical_batch_count == 0u || request->header.canonical_m == 0u ||
        request->header.canonical_n == 0u || request->header.canonical_k == 0u ||
        request->header.request_output_elements == 0u ||
        request->header.global_output_elements != request->header.canonical_batch_count *
            request->header.canonical_m * request->header.canonical_n ||
        memcmp(request->header.request_sha256, request_digest, sizeof(request_digest)) != 0 ||
        digest_is_zero(request->header.task_contract_sha256)) {
        v4_error(error_message, "sidecar_validation_failed: v4 header is invalid or unbound");
        free(manifest); free(sidecar);
        return 1;
    }
    if (execution_plan_sha256_file(sidecar_relative, request->sidecar_sha256) != 0) {
        v4_error(error_message, "sidecar_validation_failed: v4 sidecar hash failed");
        free(manifest); free(sidecar);
        return 1;
    }
    request->work_units = (execution_plan_v4_work_unit_t *)calloc(expected_dpus, sizeof(*request->work_units));
    if (request->work_units == NULL) {
        v4_error(error_message, "sidecar_validation_failed: v4 work-unit allocation failed");
        free(manifest); free(sidecar);
        return 1;
    }
    if ((uint64_t)request->header.header_bytes +
        (uint64_t)request->header.work_unit_count * request->header.record_bytes > sidecar_length) {
        v4_error(error_message, "sidecar_validation_failed: v4 sidecar is truncated");
        free(manifest); free(sidecar);
        return 1;
    }
    memcpy(request->work_units, sidecar + request->header.header_bytes,
        expected_dpus * sizeof(*request->work_units));
    if ((uint64_t)request->header.header_bytes +
        (uint64_t)request->header.work_unit_count * request->header.record_bytes != sidecar_length ||
        execution_plan_distributed_v4_validate(&request->header, request->work_units,
            expected_dpus, expected_tasklets, error_message) != 0 ||
        validate_work_units(&request->header, request->work_units, error_message) != 0) {
        v4_error(error_message, "sidecar_validation_failed: v4 sidecar length or work units are invalid");
        free(manifest); free(sidecar);
        return 1;
    }
    for (uint32_t index = 0u; index < expected_dpus; index++) {
        if (request->items[index].work_unit.tile_id != request->work_units[index].tile_id) {
            v4_error(error_message, "request_manifest_failed: manifest tile identity differs from sidecar");
            free(manifest); free(sidecar);
            return 1;
        }
        request->items[index].work_unit = request->work_units[index];
    }
    free(manifest);
    free(sidecar);
    return 0;
}

static int read_payload(const char *path, uint32_t expected, unsigned char **payload) {
    FILE *file = fopen(path, "rb");
    long size;
    if (file == NULL || fseek(file, 0, SEEK_END) != 0) {
        if (file != NULL) fclose(file);
        return 1;
    }
    size = ftell(file);
    if (size < 0 || (uintmax_t)size != expected || fseek(file, 0, SEEK_SET) != 0) {
        fclose(file);
        return 1;
    }
    if (expected == 0u) {
        *payload = (unsigned char *)malloc(1u);
        if (*payload == NULL) {
            fclose(file);
            return 1;
        }
        fclose(file);
        return 0;
    }
    if (posix_memalign((void **)payload, 8u, expected) != 0) {
        fclose(file);
        return 1;
    }
    if (fread(*payload, 1u, expected, file) != expected || fclose(file) != 0) {
        free(*payload);
        *payload = NULL;
        return 1;
    }
    return 0;
}

static int payload_digest_matches(
    const unsigned char *payload,
    uint32_t payload_bytes,
    const char expected_sha256[65]
) {
    char actual_sha256[65];
    return payload != NULL &&
        execution_plan_sha256_bytes(payload, payload_bytes, actual_sha256) == 0 &&
        strcmp(actual_sha256, expected_sha256) == 0;
}

int execution_plan_v4_request_load_payloads(execution_plan_v4_request_t *request, char **error_message) {
    if (request == NULL || request->items == NULL) {
        v4_error(error_message, "payload_validation_failed: missing request items");
        return 1;
    }
    for (uint32_t index = 0u; index < request->item_count; index++) {
        execution_plan_v4_request_item_t *item = &request->items[index];
        if (read_payload(item->a_path, item->work_unit.a_transfer_bytes, &item->a_payload) != 0 ||
            read_payload(item->b_path, item->work_unit.b_transfer_bytes, &item->b_payload) != 0) {
            v4_error(error_message, "payload_validation_failed: A or B payload length/read failed");
            return 1;
        }
        if (!payload_digest_matches(item->a_payload, item->work_unit.a_transfer_bytes, item->a_sha256) ||
            !payload_digest_matches(item->b_payload, item->work_unit.b_transfer_bytes, item->b_sha256)) {
            v4_error(error_message, "payload_validation_failed: A or B payload SHA-256 mismatch");
            return 1;
        }
        if ((item->work_unit.flags & EXECUTION_PLAN_V4_FLAG_ZERO_WORK) != 0u) continue;
        if (posix_memalign((void **)&item->c_payload, 8u, item->work_unit.c_transfer_bytes) != 0) {
            v4_error(error_message, "payload_validation_failed: C output allocation failed");
            return 1;
        }
        memset(item->c_payload, 0, item->work_unit.c_transfer_bytes);
    }
    return 0;
}

int execution_plan_v4_request_write_output(
    const execution_plan_v4_request_item_t *item,
    char **error_message
) {
    FILE *file;
    size_t written;
    int close_status;
    if (item == NULL || item->c_path == NULL || item->c_payload == NULL) {
        v4_error(error_message, "output_manifest_failed: missing C output");
        return 1;
    }
    file = fopen(item->c_path, "wb");
    if (file == NULL) {
        v4_error(error_message, "output_manifest_failed: C output open failed");
        return 1;
    }
    written = fwrite(item->c_payload, 1u, item->work_unit.c_transfer_bytes, file);
    close_status = fclose(file);
    if (written != item->work_unit.c_transfer_bytes || close_status != 0) {
        v4_error(error_message, "output_manifest_failed: C output write failed");
        return 1;
    }
    return 0;
}

void execution_plan_v4_request_free(execution_plan_v4_request_t *request) {
    if (request == NULL) return;
    for (uint32_t index = 0u; index < request->item_count; index++) {
        free(request->items[index].a_path);
        free(request->items[index].b_path);
        free(request->items[index].c_path);
        free(request->items[index].a_payload);
        free(request->items[index].b_payload);
        free(request->items[index].c_payload);
    }
    free(request->items);
    free(request->work_units);
    free(request->manifest_path);
    free(request->sidecar_path);
    memset(request, 0, sizeof(*request));
}
