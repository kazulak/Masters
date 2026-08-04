#define _POSIX_C_SOURCE 200809L

#include <dpu.h>

#include <ctype.h>
#include <errno.h>
#include <math.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include "common.h"
#include "../upmem_sdk_generic_loop_resident/session_protocol.h"

#if !UPMEM_GENERIC_HARDWARE_MVP
#error "frontier hardware execution requires UPMEM_GENERIC_HARDWARE_MVP=1"
#endif

#define FRONTIER_ALLOCATION_PROFILE "backend=hw"
#define FNV1A64_OFFSET 14695981039346656037ULL
#define FNV1A64_PRIME 1099511628211ULL
/* The resident manifest parser encodes quantization_mode=none as mode zero;
 * resident dpu.c treats mode one as its local requantization path. */
#define FRONTIER_FLOAT32_MODE 0u

typedef struct {
    double package_parse_time_s;
    double allocation_time_s;
    double binary_load_time_s;
    double initial_h2d_time_s;
    double descriptor_h2d_time_s;
    double control_h2d_time_s;
    double wave0_launch_time_s;
    double wave0_sync_time_s;
    double inter_wave_d2h_time_s;
    double inter_wave_h2d_time_s;
    double wave1_launch_time_s;
    double wave1_sync_time_s;
    double final_d2h_time_s;
    double output_write_time_s;
    double release_time_s;
} frontier_timing_t;

typedef struct {
    resident_request_t request;
    const char *manifest_path;
    char *package_path;
    char *parse_error;
    uint64_t manifest_hash;
    int manifest_hash_valid;
    uint64_t package_hash;
    int package_hash_valid;
    uint64_t binary_hash;
    int binary_hash_valid;
    uint64_t source_hash;
    int source_hash_valid;
    unsigned char **inputs;
    unsigned char *intermediate;
    unsigned char *final_output;
    size_t intermediate_transfer_bytes;
    size_t final_transfer_bytes;
    uint64_t initial_h2d_bytes;
    int prepared;
} frontier_request_t;

typedef struct {
    uint32_t dpu_id;
    uint32_t operation_index;
    int launched;
    int synchronized;
    int completion_read;
    int completion_verified;
    resident_completion_t completion;
} frontier_task_instance_t;

typedef struct {
    const char *stage;
    const char *message;
    int wave;
    int operation;
    int dpu;
    int sdk_error_code;
} frontier_failure_t;

static volatile sig_atomic_t frontier_interrupted = 0;

static void frontier_signal_handler(int signal_number) {
    (void)signal_number;
    frontier_interrupted = 1;
}

static double frontier_now_s(void) {
    struct timespec value;
    if (clock_gettime(CLOCK_MONOTONIC, &value) != 0) return 0.0;
    return (double)value.tv_sec + (double)value.tv_nsec / 1000000000.0;
}

static void frontier_report_sdk_error(const char *operation, dpu_error_t error) {
    fprintf(stderr, "%s failed: %s\n", operation, dpu_error_to_string(error));
}

static void frontier_fail(
    frontier_failure_t *failure,
    const char *stage,
    const char *message,
    int wave,
    int operation,
    int dpu,
    dpu_error_t error
) {
    if (failure->stage != NULL) return;
    failure->stage = stage;
    failure->message = message;
    failure->wave = wave;
    failure->operation = operation;
    failure->dpu = dpu;
    failure->sdk_error_code = error == DPU_OK ? -1 : (int)error;
}

static int frontier_read_file(const char *path, unsigned char **data, size_t *length) {
    FILE *file = fopen(path, "rb");
    long size;
    int failed = 0;
    if (file == NULL || fseek(file, 0, SEEK_END) != 0) failed = 1;
    size = failed ? -1 : ftell(file);
    if (size < 0 || fseek(file, 0, SEEK_SET) != 0) failed = 1;
    if (failed) {
        if (file != NULL) (void)fclose(file);
        return 1;
    }
    *data = (unsigned char *)malloc((size_t)size + 1u);
    if (*data == NULL || fread(*data, 1u, (size_t)size, file) != (size_t)size) {
        free(*data);
        *data = NULL;
        (void)fclose(file);
        return 1;
    }
    (*data)[size] = 0;
    *length = (size_t)size;
    return fclose(file) != 0;
}

static uint64_t frontier_hash_bytes(const unsigned char *data, size_t length) {
    uint64_t value = FNV1A64_OFFSET;
    for (size_t index = 0; index < length; index++) {
        value ^= data[index];
        value *= FNV1A64_PRIME;
    }
    return value;
}

static int frontier_hash_file(const char *path, uint64_t *hash) {
    unsigned char buffer[4096];
    FILE *file = path == NULL ? NULL : fopen(path, "rb");
    size_t bytes;
    uint64_t value = FNV1A64_OFFSET;
    if (file == NULL) return 1;
    while ((bytes = fread(buffer, 1u, sizeof(buffer), file)) != 0u) {
        for (size_t index = 0; index < bytes; index++) {
            value ^= buffer[index];
            value *= FNV1A64_PRIME;
        }
    }
    {
        const int read_failed = ferror(file) != 0;
        const int close_failed = fclose(file) != 0;
        if (read_failed || close_failed) return 1;
    }
    *hash = value;
    return 0;
}

static int frontier_file_size(const char *path, size_t expected) {
    struct stat value;
    return stat(path, &value) != 0 || value.st_size < 0 || (uintmax_t)value.st_size != (uintmax_t)expected;
}

static int frontier_read_exact(const char *path, void *buffer, size_t bytes) {
    FILE *file = fopen(path, "rb");
    size_t read_bytes;
    int close_failed;
    if (file == NULL) return 1;
    read_bytes = fread(buffer, 1u, bytes, file);
    close_failed = fclose(file) != 0;
    return read_bytes == bytes && !close_failed ? 0 : 1;
}

static int frontier_write_exact(const char *path, const void *buffer, size_t bytes) {
    FILE *file = fopen(path, "wb");
    size_t written;
    int close_failed;
    if (file == NULL) return 1;
    written = fwrite(buffer, 1u, bytes, file);
    close_failed = fclose(file) != 0;
    return written == bytes && !close_failed ? 0 : 1;
}

static int frontier_buffer_finite(const unsigned char *buffer, size_t bytes) {
    for (size_t offset = 0; offset < bytes; offset += sizeof(float)) {
        float value;
        memcpy(&value, buffer + offset, sizeof(value));
        if (!isfinite(value)) return 1;
    }
    return 0;
}

static int frontier_safe_relative(const char *path) {
    const char *cursor = path;
    if (path == NULL || path[0] == '\0' || path[0] == '/') return 1;
    while (*cursor != '\0') {
        const char *start = cursor;
        while (*cursor != '\0' && *cursor != '/') cursor++;
        if ((size_t)(cursor - start) == 2u && start[0] == '.' && start[1] == '.') return 1;
        if (*cursor == '/') cursor++;
    }
    return 0;
}

static char *frontier_manifest_root(const char *path) {
    const char *slash = strrchr(path, '/');
    const size_t length = slash == NULL ? 1u : (size_t)(slash - path);
    char *root = (char *)malloc(length + 1u);
    if (root == NULL) return NULL;
    if (slash == NULL) memcpy(root, ".", 2u);
    else {
        memcpy(root, path, length);
        root[length] = '\0';
    }
    return root;
}

static char *frontier_resolve(const char *root, const char *relative) {
    const size_t root_length = strlen(root);
    const size_t relative_length = strlen(relative);
    char *path;
    if (frontier_safe_relative(relative) != 0 || root_length > SIZE_MAX - relative_length - 2u) return NULL;
    path = (char *)malloc(root_length + relative_length + 2u);
    if (path == NULL) return NULL;
    memcpy(path, root, root_length);
    path[root_length] = '/';
    memcpy(path + root_length + 1u, relative, relative_length + 1u);
    return path;
}

static const char *frontier_field_start(const char *json, const char *key) {
    char needle[128];
    if (strlen(key) + 3u > sizeof(needle)) return NULL;
    snprintf(needle, sizeof(needle), "\"%s\"", key);
    return strstr(json, needle);
}

static int frontier_manifest_string(const char *json, const char *key, char **value) {
    const char *cursor = frontier_field_start(json, key);
    const char *start;
    size_t length;
    char *copy;
    if (cursor == NULL || (cursor = strchr(cursor, ':')) == NULL) return 1;
    cursor++;
    while (isspace((unsigned char)*cursor)) cursor++;
    if (*cursor++ != '"') return 1;
    start = cursor;
    while (*cursor != '\0' && *cursor != '"') {
        if (*cursor == '\\' || (unsigned char)*cursor < 0x20u) return 1;
        cursor++;
    }
    if (*cursor != '"') return 1;
    length = (size_t)(cursor - start);
    copy = (char *)malloc(length + 1u);
    if (copy == NULL) return 1;
    memcpy(copy, start, length);
    copy[length] = '\0';
    *value = copy;
    return 0;
}

static int frontier_json_uint(const char *json, const char *key, uint32_t *value) {
    const char *cursor = frontier_field_start(json, key);
    char *end;
    unsigned long parsed;
    if (cursor == NULL || (cursor = strchr(cursor, ':')) == NULL) return 1;
    cursor++;
    while (isspace((unsigned char)*cursor)) cursor++;
    if (!isdigit((unsigned char)*cursor)) return 1;
    errno = 0;
    parsed = strtoul(cursor, &end, 10);
    if (errno != 0 || end == cursor || parsed > UINT32_MAX) return 1;
    *value = (uint32_t)parsed;
    return 0;
}

static int frontier_replace_field(char **json, size_t *length, const char *key, const char *replacement, int quoted) {
    const char *field = frontier_field_start(*json, key);
    const char *value;
    const char *end;
    size_t prefix;
    size_t suffix;
    size_t replacement_length = strlen(replacement) + (quoted ? 2u : 0u);
    char *updated;
    if (field == NULL || (value = strchr(field, ':')) == NULL) return 1;
    value++;
    while (isspace((unsigned char)*value)) value++;
    if (quoted) {
        if (*value++ != '"') return 1;
        end = value;
        while (*end != '\0' && *end != '"') {
            if (*end == '\\') return 1;
            end++;
        }
        if (*end != '"') return 1;
        value--;
        end++;
    } else {
        end = value;
        while (isdigit((unsigned char)*end)) end++;
    }
    prefix = (size_t)(value - *json);
    suffix = *length - (size_t)(end - *json);
    updated = (char *)malloc(prefix + replacement_length + suffix + 1u);
    if (updated == NULL) return 1;
    memcpy(updated, *json, prefix);
    if (quoted) {
        updated[prefix] = '"';
        memcpy(updated + prefix + 1u, replacement, strlen(replacement));
        updated[prefix + replacement_length - 1u] = '"';
    } else memcpy(updated + prefix, replacement, replacement_length);
    memcpy(updated + prefix + replacement_length, *json + (size_t)(end - *json), suffix);
    updated[prefix + replacement_length + suffix] = '\0';
    free(*json);
    *json = updated;
    *length = prefix + replacement_length + suffix;
    return 0;
}

/* The existing parser has the resident identity and one-DPU request checks.
 * Normalize only those envelope fields in a temporary file in the original
 * directory; the parser still owns all package, slot, operation, and file ABI
 * validation. */
static int frontier_make_parser_manifest(const char *manifest_path, char **parser_path, char **error_message) {
    unsigned char *raw = NULL;
    size_t length = 0u;
    char *json = NULL;
    char *root = NULL;
    char *schema = NULL;
    char *native_schema = NULL;
    char *route = NULL;
    char *backend = NULL;
    char *profile = NULL;
    char *session_protocol = NULL;
    uint32_t requested_dpus = 0u;
    int fd = -1;
    FILE *file = NULL;
    int failed = 1;
    if (frontier_read_file(manifest_path, &raw, &length) != 0) {
        *error_message = strdup("frontier manifest is unreadable");
        goto done;
    }
    json = (char *)raw;
    {
        const int identity_parse_failed = frontier_manifest_string(json, "schema_version", &schema) != 0 ||
            frontier_manifest_string(json, "native_schema_version", &native_schema) != 0 ||
            frontier_manifest_string(json, "route_id", &route) != 0 || frontier_manifest_string(json, "backend_id", &backend) != 0 ||
            frontier_manifest_string(json, "hardware_profile_version", &profile) != 0 ||
            frontier_manifest_string(json, "session_protocol", &session_protocol) != 0 ||
            frontier_json_uint(json, "requested_dpus", &requested_dpus) != 0;
        if (identity_parse_failed || strcmp(schema, RESIDENT_SESSION_SCHEMA) != 0 ||
            strcmp(native_schema, FRONTIER_SCHEMA_ID) != 0 || strcmp(route, FRONTIER_ROUTE_ID) != 0 ||
            strcmp(backend, FRONTIER_BACKEND_ID) != 0 || strcmp(profile, FRONTIER_PROFILE_ID) != 0 ||
            strcmp(session_protocol, RESIDENT_SESSION_SCHEMA) != 0 ||
            requested_dpus != FRONTIER_TWO_DPU_COUNT) {
            *error_message = strdup("hardware_profile_violation: frontier identity or physical DPU count mismatch");
            goto done;
        }
    }
    root = frontier_manifest_root(manifest_path);
    if (root == NULL || strlen(root) > SIZE_MAX - 32u) goto done;
    {
        const size_t path_length = strlen(root) + 32u;
        char *template_path = (char *)malloc(path_length);
        if (template_path == NULL) goto done;
        snprintf(template_path, path_length, "%s/.frontier-parser-XXXXXX", root);
        fd = mkstemp(template_path);
        free(root);
        root = NULL;
        if (fd < 0) {
            free(template_path);
            goto done;
        }
        *parser_path = template_path;
    }
    if (frontier_replace_field(&json, &length, "manifest_kind", RESIDENT_REQUEST_KIND, 1) != 0 ||
        frontier_replace_field(&json, &length, "route_id", RESIDENT_ROUTE_ID, 1) != 0 ||
        frontier_replace_field(&json, &length, "backend_id", RESIDENT_BACKEND_ID, 1) != 0 ||
        frontier_replace_field(&json, &length, "hardware_profile_version", RESIDENT_PROFILE_VERSION, 1) != 0 ||
        frontier_replace_field(&json, &length, "requested_dpus", "1", 0) != 0) goto done;
    file = fdopen(fd, "wb");
    if (file == NULL) goto done;
    fd = -1;
    if (fwrite(json, 1u, length, file) != length) goto done;
    {
        const int close_failed = fclose(file) != 0;
        file = NULL;
        if (close_failed) goto done;
    }
    failed = 0;
done:
    if (file != NULL) (void)fclose(file);
    if (fd >= 0) (void)close(fd);
    if (failed && parser_path != NULL && *parser_path != NULL) {
        unlink(*parser_path);
        free(*parser_path);
        *parser_path = NULL;
    }
    free(schema);
    free(native_schema);
    free(route);
    free(backend);
    free(profile);
    free(session_protocol);
    free(root);
    free(json);
    return failed;
}

static int frontier_validate_contract(const frontier_request_t *frontier, const char **reason) {
    const resident_request_t *request = &frontier->request;
    const resident_operation_t *first;
    const resident_operation_t *second;
    const resident_operation_t *combine;
    if (request->header.operation_count != FRONTIER_TWO_DPU_OPERATION_COUNT ||
        request->input_count != request->header.initial_slot_count || request->final_count != 1u ||
        request->header.final_output_count != 1u || request->final_outputs == NULL || request->quantization_mode == NULL ||
        strcmp(request->quantization_mode, "none") != 0) {
        *reason = "frontier_contract_requires_three_operations_one_final_float32_output";
        return 1;
    }
    first = &request->operations[0];
    second = &request->operations[1];
    combine = &request->operations[2];
    for (uint32_t index = 0; index < FRONTIER_TWO_DPU_OPERATION_COUNT; index++) {
        const resident_operation_t *operation = &request->operations[index];
        if (operation->kind != RESIDENT_OPERATION_CONTRACT || operation->mode != FRONTIER_FLOAT32_MODE ||
            operation->slot_out_imag != RESIDENT_INVALID_SLOT || operation->slot_out_real == RESIDENT_INVALID_SLOT) {
            *reason = "hardware_profile_violation: complex-combine and quantized operations are forbidden";
            return 1;
        }
    }
    if (first->slot_out_real == second->slot_out_real || first->slot_out_real == combine->slot_out_real ||
        second->slot_out_real == combine->slot_out_real || first->slot_a == second->slot_a ||
        first->slot_a == second->slot_b || first->slot_b == second->slot_a || first->slot_b == second->slot_b ||
        first->slot_out_real >= request->header.slot_count || second->slot_out_real >= request->header.slot_count ||
        combine->slot_a != first->slot_out_real || combine->slot_b != second->slot_out_real ||
        combine->slot_c != RESIDENT_INVALID_SLOT || combine->slot_d != RESIDENT_INVALID_SLOT ||
        combine->slot_out_real != request->final_outputs[0].slot_id ||
        combine->slot_out_real >= request->header.slot_count ||
        (request->slot_flags[combine->slot_out_real] & RESIDENT_SLOT_FINAL_FLAG) == 0u) {
        *reason = "hardware_profile_violation: frontier wave dependencies or output slots are invalid";
        return 1;
    }
    if (request->final_outputs[0].slot_id != combine->slot_out_real || request->final_outputs[0].status != 0) {
        *reason = "hardware_profile_violation: final output must be the sole real operation-two output";
        return 1;
    }
    return 0;
}

static int frontier_load_request(const char *manifest_path, frontier_request_t *frontier, const char **failure_stage) {
    char *parser_path = NULL;
    const char *reason = NULL;
    memset(frontier, 0, sizeof(*frontier));
    frontier->manifest_path = manifest_path;
    if (frontier_hash_file(manifest_path, &frontier->manifest_hash) != 0) {
        frontier->parse_error = strdup("frontier manifest hash/read failed");
        *failure_stage = "manifest_parse_failed";
        return 1;
    }
    frontier->manifest_hash_valid = 1;
    if (frontier_make_parser_manifest(manifest_path, &parser_path, &frontier->parse_error) != 0) {
        *failure_stage = "manifest_parse_failed";
        return 1;
    }
    if (resident_request_load(parser_path, &frontier->request, &frontier->parse_error) != 0) {
        *failure_stage = "resident_package_parse_failed";
        unlink(parser_path);
        free(parser_path);
        return 1;
    }
    unlink(parser_path);
    free(parser_path);
    if (frontier_validate_contract(frontier, &reason) != 0) {
        frontier->parse_error = strdup(reason);
        *failure_stage = "hardware_profile_violation";
        return 1;
    }
    if (frontier_hash_file(__FILE__, &frontier->source_hash) != 0) {
        frontier->parse_error = strdup("frontier host source hash/read failed");
        *failure_stage = "source_hash_failed";
        return 1;
    }
    frontier->source_hash_valid = 1;
    if (frontier_hash_file(frontier->request.dpu_binary_path, &frontier->binary_hash) != 0) {
        frontier->parse_error = strdup("frontier DPU binary hash/read failed");
        *failure_stage = "binary_hash_failed";
        return 1;
    }
    frontier->binary_hash_valid = 1;
    {
        unsigned char *manifest = NULL;
        size_t manifest_length = 0u;
        char *package_ref = NULL;
        char *root = frontier_manifest_root(manifest_path);
        if (root == NULL || frontier_read_file(manifest_path, &manifest, &manifest_length) != 0 ||
            frontier_manifest_string((char *)manifest, "package_path", &package_ref) != 0 ||
            (frontier->package_path = frontier_resolve(root, package_ref)) == NULL) {
            free(root);
            free(manifest);
            free(package_ref);
            *failure_stage = "package_hash_failed";
            return 1;
        }
        free(root);
        free(manifest);
        free(package_ref);
        if (frontier_hash_file(frontier->package_path, &frontier->package_hash) != 0) {
            frontier->parse_error = strdup("frontier package hash/read failed");
            *failure_stage = "package_hash_failed";
            return 1;
        }
        frontier->package_hash_valid = 1;
    }
    return 0;
}

static int frontier_prepare_buffers(frontier_request_t *frontier, const char **failure_stage) {
    resident_request_t *request = &frontier->request;
    frontier->inputs = (unsigned char **)calloc(request->input_count, sizeof(*frontier->inputs));
    if (frontier->inputs == NULL) {
        *failure_stage = "host_input_allocation_failed";
        return 1;
    }
    for (size_t index = 0; index < request->input_count; index++) {
        const resident_input_file_t *input = &request->inputs[index];
        if (frontier_file_size(input->path, input->raw_bytes) != 0 ||
            (frontier->inputs[index] = (unsigned char *)calloc(input->transfer_bytes, 1u)) == NULL ||
            frontier_read_exact(input->path, frontier->inputs[index], input->raw_bytes) != 0 ||
            frontier_buffer_finite(frontier->inputs[index], input->raw_bytes) != 0) {
            *failure_stage = "initial_tensor_prepare_failed";
            return 1;
        }
    }
    frontier->intermediate_transfer_bytes = request->operations[1].output_elements * sizeof(float);
    frontier->intermediate_transfer_bytes = (frontier->intermediate_transfer_bytes + 7u) & ~7u;
    frontier->intermediate = (unsigned char *)calloc(frontier->intermediate_transfer_bytes, 1u);
    frontier->final_transfer_bytes = request->final_outputs[0].transfer_bytes;
    frontier->final_output = (unsigned char *)calloc(frontier->final_transfer_bytes, 1u);
    if (frontier->intermediate == NULL || frontier->final_output == NULL) {
        *failure_stage = "host_output_allocation_failed";
        return 1;
    }
    frontier->prepared = 1;
    return 0;
}

static void frontier_free_request(frontier_request_t *frontier) {
    if (frontier->inputs != NULL) {
        for (size_t index = 0; index < frontier->request.input_count; index++) free(frontier->inputs[index]);
    }
    free(frontier->inputs);
    free(frontier->intermediate);
    free(frontier->final_output);
    free(frontier->package_path);
    free(frontier->parse_error);
    resident_request_free(&frontier->request);
    memset(frontier, 0, sizeof(*frontier));
}

static int frontier_transfer_package(
    struct dpu_set_t dpu,
    frontier_request_t *frontier,
    uint64_t *descriptor_package_h2d_bytes,
    dpu_error_t *error
) {
    const resident_control_t control = {
        frontier->request.header.slot_count,
        frontier->request.header.operation_count,
        frontier->request.header.pool_bytes,
        0u,
    };
    *error = dpu_copy_to(dpu, "RESIDENT_SLOT_DESCRIPTORS", 0u, frontier->request.slots, frontier->request.header.slot_bytes);
    if (*error != DPU_OK) return 1;
    *descriptor_package_h2d_bytes += frontier->request.header.slot_bytes;
    *error = dpu_copy_to(dpu, "RESIDENT_OPERATIONS", 0u, frontier->request.operations, frontier->request.header.operation_bytes);
    if (*error != DPU_OK) return 1;
    *descriptor_package_h2d_bytes += frontier->request.header.operation_bytes;
    *error = dpu_copy_to(dpu, "RESIDENT_CONTROL", 0u, &control, sizeof(control));
    if (*error != DPU_OK) return 1;
    *descriptor_package_h2d_bytes += sizeof(control);
    for (size_t index = 0; index < frontier->request.input_count; index++) {
        const resident_input_file_t *input = &frontier->request.inputs[index];
        *error = dpu_copy_to(dpu, "RESIDENT_SLOT_POOL", frontier->request.slots[input->slot_id].offset_bytes,
            frontier->inputs[index], input->transfer_bytes);
        if (*error != DPU_OK) return 1;
        frontier->initial_h2d_bytes += input->transfer_bytes;
    }
    return 0;
}

static int frontier_set_active(struct dpu_set_t dpu, uint32_t operation, uint64_t *control_h2d_bytes, dpu_error_t *error) {
    const uint64_t active_operation = operation;
    *error = dpu_copy_to(dpu, "RESIDENT_ACTIVE_OPERATION", 0u, &active_operation, sizeof(active_operation));
    if (*error == DPU_OK) *control_h2d_bytes += sizeof(active_operation);
    return *error == DPU_OK ? 0 : 1;
}

static int frontier_read_completion(
    struct dpu_set_t dpu,
    frontier_task_instance_t *task,
    const resident_operation_t *operation,
    frontier_failure_t *failure,
    dpu_error_t *error
) {
    resident_completion_t completion = {0};
    *error = dpu_copy_from(dpu, "RESIDENT_COMPLETION", 0u, &completion, sizeof(completion));
    task->completion_read = *error == DPU_OK;
    task->completion = completion;
    if (*error != DPU_OK) {
        frontier_fail(failure, "completion_sentinel_read_failed", "DPU completion sentinel read failed", task->operation_index == 2u ? 1 : 0,
            (int)task->operation_index, (int)task->dpu_id, *error);
        return 1;
    }
    if (completion.magic != RESIDENT_COMPLETION_MAGIC || completion.version != RESIDENT_COMPLETION_VERSION ||
        completion.active_operation_index != task->operation_index || completion.completion_status != RESIDENT_COMPLETION_COMPLETED ||
        completion.completed_operation_count != task->operation_index + 1u || completion.output_elements_processed != operation->output_elements) {
        frontier_fail(failure, "completion_sentinel_verification_failed", "DPU completion sentinel did not verify", task->operation_index == 2u ? 1 : 0,
            (int)task->operation_index, (int)task->dpu_id, DPU_OK);
        return 1;
    }
    task->completion_verified = 1;
    return 0;
}

static void frontier_json_string(FILE *file, const char *value) {
    if (value == NULL) {
        fputs("null", file);
        return;
    }
    fputc('"', file);
    for (const unsigned char *cursor = (const unsigned char *)value; *cursor != '\0'; cursor++) {
        if (*cursor == '"' || *cursor == '\\') fprintf(file, "\\%c", *cursor);
        else if (*cursor == '\n') fputs("\\n", file);
        else if (*cursor == '\r') fputs("\\r", file);
        else if (*cursor == '\t') fputs("\\t", file);
        else if (*cursor < 0x20u) fprintf(file, "\\u%04x", *cursor);
        else fputc(*cursor, file);
    }
    fputc('"', file);
}

static void frontier_hash_json(FILE *file, uint64_t hash, int available) {
    if (!available) fputs("null", file);
    else fprintf(file, "\"%016llx\"", (unsigned long long)hash);
}

static void frontier_failure_context(FILE *file, const frontier_failure_t *failure) {
    if (failure->stage == NULL || (failure->wave < 0 && failure->operation < 0 && failure->dpu < 0)) {
        fputs("null", file);
        return;
    }
    fprintf(file, "{\"wave_index\":%d,\"operation_index\":%d,\"dpu_index\":%d}",
        failure->wave, failure->operation, failure->dpu);
}

static void frontier_write_task_ids(
    FILE *file,
    const frontier_task_instance_t tasks[FRONTIER_TWO_DPU_OPERATION_COUNT]
) {
    int first = 1;
    for (uint32_t index = 0; index < FRONTIER_TWO_DPU_OPERATION_COUNT; index++) {
        if (!tasks[index].completion_verified) continue;
        if (!first) fputc(',', file);
        fprintf(file, "\"task_%u\"", tasks[index].operation_index);
        first = 0;
    }
}

static int frontier_write_validation(const frontier_request_t *frontier, const char *failure_stage, const char *reason) {
    const int valid = failure_stage == NULL;
    printf("{\"schema_version\":\"%s\",\"native_schema_version\":\"%s\",\"status\":\"%s\",\"valid\":%s,\"failure_stage\":",
        FRONTIER_SCHEMA_ID, FRONTIER_SCHEMA_ID,
        valid ? "valid" : "invalid", valid ? "true" : "false");
    frontier_json_string(stdout, failure_stage);
    fputs(",\"error\":", stdout);
    frontier_json_string(stdout, reason);
    printf(",\"native_execution\":false,\"allocation_attempted\":false,\"launch_attempted\":false,\"release_attempted\":false,\"requested_dpus\":2,\"tasklets_per_dpu\":1,\"operation_count\":%u,\"final_output_count\":%u,\"quantization_mode\":",
        frontier->request.header.operation_count, frontier->request.header.final_output_count);
    frontier_json_string(stdout, frontier->request.quantization_mode);
    printf(",\"route_id\":\"%s\",\"backend_id\":\"%s\",\"hardware_profile_version\":\"%s\",\"target\":\"hardware\",\"session_protocol\":\"%s\",\"profile_id\":\"%s\",\"wave_barrier_count\":2}\n",
        FRONTIER_ROUTE_ID, FRONTIER_BACKEND_ID, FRONTIER_PROFILE_ID, FRONTIER_SCHEMA_ID, FRONTIER_PROFILE_ID);
    return valid ? 0 : 1;
}

static int frontier_write_response(
    const char *path,
    const frontier_request_t *frontier,
    const frontier_failure_t *failure,
    const frontier_timing_t *timing,
    const frontier_task_instance_t tasks[FRONTIER_TWO_DPU_OPERATION_COUNT],
    uint32_t allocated_dpus,
    int allocation_attempted,
    int load_attempted,
    int load_succeeded,
    int wave0_launch_attempted,
    int wave0_sync_attempted,
    int wave0_sync_succeeded,
    int wave1_launch_attempted,
    int wave1_sync_attempted,
    int wave1_sync_succeeded,
    int inter_wave_d2h,
    int inter_wave_h2d,
    int final_d2h,
    int output_written,
    int release_attempted,
    int release_confirmed,
    uint64_t descriptor_package_h2d_bytes,
    uint64_t operation_control_h2d_bytes,
    uint64_t inter_wave_d2h_bytes,
    uint64_t inter_wave_h2d_bytes,
    uint64_t final_d2h_bytes
) {
    const resident_final_file_t *final_output = NULL;
    uint64_t final_output_file_hash = 0u;
    int final_output_file_hash_valid = 0;
    const int completed = failure->stage == NULL && allocated_dpus == FRONTIER_TWO_DPU_COUNT && load_succeeded &&
        wave0_sync_succeeded && wave1_sync_succeeded && inter_wave_d2h && inter_wave_h2d && final_d2h && output_written && release_confirmed;
    const uint64_t actual_h2d = frontier->initial_h2d_bytes + descriptor_package_h2d_bytes + operation_control_h2d_bytes + inter_wave_h2d_bytes;
    const uint64_t actual_d2h = inter_wave_d2h_bytes + final_d2h_bytes;
    const uint64_t actual_transfer = actual_h2d + actual_d2h;
    const double total_route_time_s = timing->package_parse_time_s + timing->allocation_time_s + timing->binary_load_time_s +
        timing->initial_h2d_time_s + timing->wave0_launch_time_s + timing->wave0_sync_time_s +
        timing->inter_wave_d2h_time_s + timing->inter_wave_h2d_time_s + timing->wave1_launch_time_s +
        timing->wave1_sync_time_s + timing->final_d2h_time_s + timing->output_write_time_s + timing->release_time_s;
    FILE *file;
    if (frontier->request.final_count == 1u && frontier->request.final_outputs != NULL) {
        final_output = &frontier->request.final_outputs[0];
        if (output_written && final_output->path != NULL && frontier_hash_file(final_output->path, &final_output_file_hash) == 0) {
            final_output_file_hash_valid = 1;
        }
    }
    file = path == NULL ? NULL : fopen(path, "w");
    if (file == NULL) return 1;
    fprintf(file, "{\n  \"schema_version\":\"%s\",\n  \"native_schema_version\":\"%s\",\n  \"manifest_kind\":\"frontier_two_dpu_response\",\n  \"status\":\"%s\",\n  \"failure_stage\":",
        FRONTIER_SCHEMA_ID, FRONTIER_SCHEMA_ID, completed ? "completed" : "failed");
    frontier_json_string(file, failure->stage);
    fputs(",\n  \"error\":", file);
    frontier_json_string(file, failure->message);
    fputs(",\n  \"failure_context\":", file);
    frontier_failure_context(file, failure);
    fprintf(file, ",\n  \"sdk_error_code\":%d,\n  \"route_id\":\"%s\",\n  \"profile_id\":\"%s\",\n  \"backend_id\":\"%s\",\n  \"hardware_profile_version\":\"%s\",\n  \"target\":\"hardware\",\n  \"target_requested\":\"hardware\",\n  \"target_observed\":\"%s\",\n  \"requested_dpus\":2,\n  \"allocated_dpus\":%u,\n  \"hardware_execution\":%s,\n  \"hardware_kernel_executed\":%s,\n  \"hardware_functionality_evidence\":%s,\n  \"cpu_fallback_used\":false,\n  \"simulator_kernel_executed\":false,\n  \"no_cpu_fallback\":true,\n  \"no_simulator_fallback\":true,\n  \"native_failure_fallback_used\":false,\n  \"hardware_no_fallback\":true,\n  \"performance_claim_applicable\":false,\n  \"tasklets_per_dpu\":1,\n  \"physical_dpu_count\":2,\n  \"operation_count\":3,\n  \"numeric_contract\":\"float32_real\",\n  \"numeric_mode\":\"none\",\n  \"complex_combine_used\":false,\n  \"quantization_mode\":\"none\",\n  \"allocation\":{\"attempted\":%s,\"requested_dpus\":2,\"allocated_dpus\":%u,\"profile\":\"backend=hw\",\"verified\":%s},\n  \"load\":{\"attempted\":%s,\"succeeded\":%s,\"confirmed\":%s,\"hardware\":%s},\n",
        failure->sdk_error_code, FRONTIER_ROUTE_ID, FRONTIER_PROFILE_ID, FRONTIER_BACKEND_ID, FRONTIER_PROFILE_ID,
        completed ? "hardware" : "hardware_unverified", allocated_dpus,
        completed ? "true" : "false", completed ? "true" : "false", completed ? "true" : "false",
        allocation_attempted ? "true" : "false", allocated_dpus, allocated_dpus == 2u ? "true" : "false",
        load_attempted ? "true" : "false", load_succeeded ? "true" : "false", load_succeeded ? "true" : "false", load_succeeded ? "true" : "false");
    fprintf(file, "  \"wave_plan\":[{\"wave\":0,\"assignments\":[{\"dpu\":0,\"operation\":0},{\"dpu\":1,\"operation\":1}],\"launch\":\"dpu_set_async\",\"synchronize\":\"dpu_sync_set\",\"barrier\":true},{\"wave\":1,\"assignments\":[{\"dpu\":0,\"operation\":2}],\"launch\":\"dpu0_async\",\"synchronize\":\"dpu_sync_dpu0\",\"barrier\":true}],\n  \"wave_barrier_count\":2,\n  \"co_dispatch_observed\":%s,\n  \"co_dispatch_confirmed\":%s,\n  \"overlap_measurement\":\"unmeasured\",\n  \"overlap_measured\":false,\n  \"overlap_claim\":\"unmeasured\",\n  \"overlap_evidence\":\"co_dispatch_without_overlap_measurement\",\n  \"wave0_complete_before_wave1\":%s,\n  \"completed_task_ids\":[",
        wave0_launch_attempted && wave0_sync_succeeded ? "true" : "false",
        wave0_launch_attempted && wave0_sync_succeeded ? "true" : "false",
        wave0_sync_succeeded && wave1_sync_succeeded ? "true" : "false");
    frontier_write_task_ids(file, tasks);
    fputs("],\n  \"completed_task_ids_scope\":\"wave_dependency_order_not_intra_wave_finish_order\",\n  \"barrier_count\":2,\n  \"barriers\":[{\"barrier_index\":0,\"wave_index\":0,\"completed\":", file);
    fputs(wave0_sync_succeeded ? "true" : "false", file);
    fputs("},{\"barrier_index\":1,\"wave_index\":1,\"completed\":", file);
    fputs(wave1_sync_succeeded ? "true" : "false", file);
    fputs("}],\n  \"observed_dpu_task_counts\":[", file);
    fprintf(file, "%u,%u],\n  \"launch\":{\"wave0_attempted\":%s,\"wave0_synchronized\":%s,\"wave1_attempted\":%s,\"wave1_synchronized\":%s,\"async_launch_count\":%u,\"completed\":%s,\"task_count\":3,\"barrier_count\":2},\n",
        tasks[0].completion_verified && tasks[2].completion_verified ? 2u : 0u,
        tasks[1].completion_verified ? 1u : 0u,
        wave0_launch_attempted ? "true" : "false", wave0_sync_attempted && wave0_sync_succeeded ? "true" : "false",
        wave1_launch_attempted ? "true" : "false", wave1_sync_attempted && wave1_sync_succeeded ? "true" : "false",
        (wave0_launch_attempted ? 1u : 0u) + (wave1_launch_attempted ? 1u : 0u), completed ? "true" : "false");
    fprintf(file, "  \"physical_task_instances\":[{\"instance\":0,\"dpu\":0,\"operation\":0},{\"instance\":1,\"dpu\":1,\"operation\":1},{\"instance\":2,\"dpu\":0,\"operation\":2}],\n  \"per_dpu_completed_operations\":[%s,%s],\n  \"completion_sentinels\":[",
        tasks[0].completion_verified && tasks[2].completion_verified ? "2" : "0",
        tasks[1].completion_verified ? "1" : "0");
    for (uint32_t index = 0; index < FRONTIER_TWO_DPU_OPERATION_COUNT; index++) {
        if (index != 0u) fputc(',', file);
        fprintf(file, "{\"dpu\":%u,\"operation\":%u,\"read\":%s,\"verified\":%s,\"magic\":%u,\"version\":%u,\"status\":%u,\"completed_operation_count\":%u,\"output_elements\":%u}",
            tasks[index].dpu_id, tasks[index].operation_index, tasks[index].completion_read ? "true" : "false",
            tasks[index].completion_verified ? "true" : "false", tasks[index].completion.magic, tasks[index].completion.version,
            tasks[index].completion.completion_status, tasks[index].completion.completed_operation_count,
            tasks[index].completion.output_elements_processed);
    }
    fprintf(file, "],\n  \"tasks\":[{\"task_id\":\"task_0\",\"wave_index\":0,\"dpu_id\":0,\"operation_id\":0,\"completed\":%s,\"completion_confirmed\":%s},{\"task_id\":\"task_1\",\"wave_index\":0,\"dpu_id\":1,\"operation_id\":1,\"completed\":%s,\"completion_confirmed\":%s},{\"task_id\":\"task_2\",\"wave_index\":1,\"dpu_id\":0,\"operation_id\":2,\"completed\":%s,\"completion_confirmed\":%s}],\n  \"transfer\":{\"descriptor_package_h2d_bytes\":%llu,\"initial_h2d_bytes\":%llu,\"operation_control_h2d_bytes\":%llu,\"inter_wave_h2d_bytes\":%llu,\"inter_wave_d2h_bytes\":%llu,\"final_d2h_bytes\":%llu,\"h2d_bytes\":%llu,\"d2h_bytes\":%llu,\"total_bytes\":%llu,\"transfer_invariant\":%s,\"accounting_scope\":\"sdk_argument_byte_counts\"},\n  \"actual_h2d_bytes\":%llu,\n  \"actual_d2h_bytes\":%llu,\n  \"actual_transfer_bytes\":%llu,\n  \"transfer_accounting_scope\":\"native_sdk_observed_application_visible\",\n",
        tasks[0].completion_verified ? "true" : "false", tasks[0].completion_verified ? "true" : "false",
        tasks[1].completion_verified ? "true" : "false", tasks[1].completion_verified ? "true" : "false",
        tasks[2].completion_verified ? "true" : "false", tasks[2].completion_verified ? "true" : "false",
        (unsigned long long)descriptor_package_h2d_bytes, (unsigned long long)frontier->initial_h2d_bytes,
        (unsigned long long)operation_control_h2d_bytes, (unsigned long long)inter_wave_h2d_bytes,
        (unsigned long long)inter_wave_d2h_bytes,
        (unsigned long long)final_d2h_bytes,
        (unsigned long long)actual_h2d, (unsigned long long)actual_d2h, (unsigned long long)actual_transfer,
        actual_transfer == actual_h2d + actual_d2h ? "true" : "false",
        (unsigned long long)actual_h2d, (unsigned long long)actual_d2h, (unsigned long long)actual_transfer);
    fprintf(file, "  \"stage_flags\":{\"inter_wave_d2h\":%s,\"inter_wave_h2d\":%s,\"final_d2h\":%s,\"output_written\":%s},\n  \"timing_scope\":\"two_dpu_frontier_resident_full_taskgraph_v1\",\n  \"timing\":{\"clock\":\"clock_monotonic\",\"overlap_measured\":false,\"package_parse_time_s\":%.9f,\"allocation_time_s\":%.9f,\"binary_load_time_s\":%.9f,\"initial_h2d_time_s\":%.9f,\"wave0_launch_time_s\":%.9f,\"wave0_barrier_wait_time_s\":%.9f,\"wave1_launch_time_s\":%.9f,\"wave1_barrier_wait_time_s\":%.9f,\"final_d2h_time_s\":%.9f,\"release_time_s\":%.9f,\"total_route_time_s\":%.9f,\"descriptor_h2d_time_s\":%.9f,\"control_h2d_time_s\":%.9f,\"wave0_sync_time_s\":%.9f,\"inter_wave_d2h_time_s\":%.9f,\"inter_wave_h2d_time_s\":%.9f,\"wave1_sync_time_s\":%.9f,\"output_write_time_s\":%.9f,\"kernel_time_s\":null},\n",
        inter_wave_d2h ? "true" : "false", inter_wave_h2d ? "true" : "false", final_d2h ? "true" : "false", output_written ? "true" : "false",
        timing->package_parse_time_s, timing->allocation_time_s, timing->binary_load_time_s, timing->initial_h2d_time_s,
        timing->wave0_launch_time_s, timing->wave0_sync_time_s, timing->wave1_launch_time_s, timing->wave1_sync_time_s,
        timing->final_d2h_time_s, timing->release_time_s, total_route_time_s, timing->descriptor_h2d_time_s,
        timing->control_h2d_time_s, timing->wave0_sync_time_s, timing->inter_wave_d2h_time_s, timing->inter_wave_h2d_time_s,
        timing->wave1_sync_time_s, timing->output_write_time_s);
    fprintf(file, "  \"release\":{\"attempted\":%s,\"confirmed\":%s},\n  \"hashes\":{\"manifest_fnv1a64\":",
        release_attempted ? "true" : "false", release_confirmed ? "true" : "false");
    frontier_hash_json(file, frontier->manifest_hash, frontier->manifest_hash_valid);
    fputs(",\"package_fnv1a64\":", file);
    frontier_hash_json(file, frontier->package_hash, frontier->package_hash_valid);
    fputs(",\"dpu_binary_fnv1a64\":", file);
    frontier_hash_json(file, frontier->binary_hash, frontier->binary_hash_valid);
    fputs(",\"host_source_fnv1a64\":", file);
    frontier_hash_json(file, frontier->source_hash, frontier->source_hash_valid);
    fputs(",\"final_output_file_fnv1a64\":", file);
    frontier_hash_json(file, final_output_file_hash, final_output_file_hash_valid);
    fputs("},\n  \"final_output\":", file);
    if (final_output == NULL) {
        fputs("null\n}\n", file);
    } else {
        fprintf(file, "{\"component\":\"real\",\"slot_id\":%u,\"elements\":%u,\"output_path\":",
            final_output->slot_id, final_output->elements);
        frontier_json_string(file, final_output->path);
        fputs(",\"path\":", file);
        frontier_json_string(file, final_output->path);
        fputs(",\"raw_bytes\":", file);
        fprintf(file, "%zu,\"hash_fnv1a64\":", final_output->raw_bytes);
        if (final_d2h && frontier->final_output != NULL) {
            frontier_hash_json(file, frontier_hash_bytes(frontier->final_output, final_output->raw_bytes), 1);
        } else {
            fputs("null", file);
        }
        fprintf(file, ",\"written\":%s}\n}\n", output_written ? "true" : "false");
    }
    {
        const int write_failed = ferror(file) != 0;
        const int close_failed = fclose(file) != 0;
        return write_failed || close_failed;
    }
}

int main(int argc, char **argv) {
    frontier_request_t frontier;
    frontier_timing_t timing = {0};
    frontier_failure_t failure = {0, 0, -1, -1, -1, -1};
    frontier_task_instance_t tasks[FRONTIER_TWO_DPU_OPERATION_COUNT] = {
        {0u, 0u, 0, 0, 0, 0, {0}}, {1u, 1u, 0, 0, 0, 0, {0}}, {0u, 2u, 0, 0, 0, 0, {0}}
    };
    struct dpu_set_t set = {0};
    struct dpu_set_t dpu0 = {0};
    struct dpu_set_t dpu1 = {0};
    dpu_error_t error = DPU_OK;
    const char *failure_stage = NULL;
    const char *response_path = NULL;
    uint32_t allocated_dpus = 0u;
    uint64_t descriptor_package_h2d_bytes = 0u;
    uint64_t operation_control_h2d_bytes = 0u;
    uint64_t inter_wave_d2h_bytes = 0u;
    uint64_t inter_wave_h2d_bytes = 0u;
    uint64_t final_d2h_bytes = 0u;
    int allocation_attempted = 0;
    int set_allocated = 0;
    int load_attempted = 0;
    int load_succeeded = 0;
    int wave0_launch_attempted = 0;
    int wave0_sync_attempted = 0;
    int wave0_sync_succeeded = 0;
    int wave1_launch_attempted = 0;
    int wave1_sync_attempted = 0;
    int wave1_sync_succeeded = 0;
    int inter_wave_d2h = 0;
    int inter_wave_h2d = 0;
    int final_d2h = 0;
    int output_written = 0;
    int release_attempted = 0;
    int release_confirmed = 0;
    int rc = 1;
    const double started = frontier_now_s();

    memset(&frontier, 0, sizeof(frontier));
    if (argc == 3 && strcmp(argv[1], "--validate-frontier-package") == 0) {
        const double stage_started = frontier_now_s();
        if (frontier_load_request(argv[2], &frontier, &failure_stage) == 0) failure_stage = NULL;
        timing.package_parse_time_s = frontier_now_s() - stage_started;
        rc = frontier_write_validation(&frontier, failure_stage, frontier.parse_error);
        frontier_free_request(&frontier);
        return rc;
    }
    if (argc != 5 || strcmp(argv[1], "--frontier-package") != 0 || strcmp(argv[3], "--frontier-response") != 0) {
        fprintf(stderr, "usage: %s --frontier-package request.json --frontier-response response.json\n", argv[0]);
        fprintf(stderr, "   or: %s --validate-frontier-package request.json\n", argv[0]);
        return 2;
    }
    response_path = argv[4];
    signal(SIGTERM, frontier_signal_handler);
    signal(SIGINT, frontier_signal_handler);
    {
        const double stage_started = frontier_now_s();
        if (frontier_load_request(argv[2], &frontier, &failure_stage) != 0) {
            frontier_fail(&failure, failure_stage, frontier.parse_error, -1, -1, -1, DPU_OK);
            goto write_response;
        }
        timing.package_parse_time_s = frontier_now_s() - stage_started;
    }
    if (getenv("UPMEM_ALLOW_PHYSICAL_HARDWARE") == NULL || strcmp(getenv("UPMEM_ALLOW_PHYSICAL_HARDWARE"), "1") != 0) {
        frontier_fail(&failure, "hardware_opt_in_missing", "UPMEM_ALLOW_PHYSICAL_HARDWARE=1 is required", -1, -1, -1, DPU_OK);
        goto write_response;
    }
    if (getenv("DPU_BACKEND") != NULL || NR_TASKLETS != FRONTIER_TWO_DPU_TASKLETS) {
        frontier_fail(&failure, "hardware_profile_violation", "DPU_BACKEND must be unset and NR_TASKLETS must equal one", -1, -1, -1, DPU_OK);
        goto write_response;
    }
    if (frontier_interrupted || frontier_prepare_buffers(&frontier, &failure_stage) != 0) {
        frontier_fail(&failure, failure_stage == NULL ? "host_prepare_failed" : failure_stage, "frontier host preparation failed", -1, -1, -1, DPU_OK);
        goto write_response;
    }
    {
        const double stage_started = frontier_now_s();
        allocation_attempted = 1;
        error = dpu_alloc(FRONTIER_TWO_DPU_COUNT, FRONTIER_ALLOCATION_PROFILE, &set);
        if (error == DPU_OK) set_allocated = 1;
        if (error == DPU_OK) error = dpu_get_nr_dpus(set, &allocated_dpus);
        timing.allocation_time_s = frontier_now_s() - stage_started;
        if (error != DPU_OK || allocated_dpus != FRONTIER_TWO_DPU_COUNT) {
            if (error != DPU_OK) frontier_report_sdk_error("frontier dpu_alloc", error);
            frontier_fail(&failure, error == DPU_ERR_INVALID_PROFILE ? "hardware_profile_violation" : "hardware_allocation_failed",
                "exactly two physical DPUs were not allocated", -1, -1, -1, error);
            goto release_and_write;
        }
        {
            struct dpu_set_t dpu;
            uint32_t index;
            DPU_FOREACH(set, dpu, index) {
                if (index == 0u) dpu0 = dpu;
                else if (index == 1u) dpu1 = dpu;
            }
        }
    }
    {
        const double stage_started = frontier_now_s();
        load_attempted = 1;
        error = dpu_load(set, frontier.request.dpu_binary_path, NULL);
        timing.binary_load_time_s = frontier_now_s() - stage_started;
        if (error != DPU_OK) {
            frontier_report_sdk_error("frontier dpu_load", error);
            frontier_fail(&failure, "binary_load_failed", "frontier DPU binary load failed", -1, -1, -1, error);
            goto release_and_write;
        }
        load_succeeded = 1;
    }
    {
        const double stage_started = frontier_now_s();
        struct dpu_set_t dpu;
        uint32_t index;
        DPU_FOREACH(set, dpu, index) {
            if (frontier_transfer_package(dpu, &frontier, &descriptor_package_h2d_bytes, &error) != 0) {
                frontier_fail(&failure, "initial_h2d_failed", "resident package or initial tensor transfer failed", 0, -1, (int)index, error);
                break;
            }
        }
        timing.initial_h2d_time_s = frontier_now_s() - stage_started;
        if (failure.stage != NULL) goto release_and_write;
    }
    {
        const double stage_started = frontier_now_s();
        if (frontier_set_active(dpu0, 0u, &operation_control_h2d_bytes, &error) != 0) {
            frontier_fail(&failure, "operation_control_h2d_failed", "wave zero operation assignment failed", 0, 0, 0, error);
            goto release_and_write;
        }
        if (frontier_set_active(dpu1, 1u, &operation_control_h2d_bytes, &error) != 0) {
            frontier_fail(&failure, "operation_control_h2d_failed", "wave zero operation assignment failed", 0, 1, 1, error);
            goto release_and_write;
        }
        timing.control_h2d_time_s += frontier_now_s() - stage_started;
    }
    {
        const double stage_started = frontier_now_s();
        wave0_launch_attempted = 1;
        error = dpu_launch(set, DPU_ASYNCHRONOUS);
        timing.wave0_launch_time_s = frontier_now_s() - stage_started;
        tasks[0].launched = error == DPU_OK;
        tasks[1].launched = error == DPU_OK;
        if (error != DPU_OK) {
            frontier_report_sdk_error("frontier wave zero dpu_launch", error);
            frontier_fail(&failure, "wave0_launch_failed", "wave zero asynchronous set launch failed", 0, -1, -1, error);
            goto release_and_write;
        }
    }
    {
        const double stage_started = frontier_now_s();
        wave0_sync_attempted = 1;
        error = dpu_sync(set);
        timing.wave0_sync_time_s = frontier_now_s() - stage_started;
        tasks[0].synchronized = error == DPU_OK;
        tasks[1].synchronized = error == DPU_OK;
        if (error != DPU_OK) {
            frontier_report_sdk_error("frontier wave zero dpu_sync", error);
            frontier_fail(&failure, "wave0_sync_failed", "wave zero set synchronization failed", 0, -1, -1, error);
            goto release_and_write;
        }
        wave0_sync_succeeded = 1;
        if (frontier_read_completion(dpu0, &tasks[0], &frontier.request.operations[0], &failure, &error) != 0 ||
            frontier_read_completion(dpu1, &tasks[1], &frontier.request.operations[1], &failure, &error) != 0) goto release_and_write;
    }
    {
        const double stage_started = frontier_now_s();
        error = dpu_copy_from(dpu1, "RESIDENT_SLOT_POOL", frontier.request.slots[frontier.request.operations[1].slot_out_real].offset_bytes,
            frontier.intermediate, frontier.intermediate_transfer_bytes);
        timing.inter_wave_d2h_time_s = frontier_now_s() - stage_started;
        inter_wave_d2h = error == DPU_OK && frontier_buffer_finite(frontier.intermediate, frontier.request.operations[1].output_elements * sizeof(float)) == 0;
        if (!inter_wave_d2h) {
            frontier_report_sdk_error("frontier inter-wave dpu1 d2h", error);
            frontier_fail(&failure, "inter_wave_d2h_failed", "wave zero operation-one output read failed", 0, 1, 1, error);
            goto release_and_write;
        }
        inter_wave_d2h_bytes = frontier.intermediate_transfer_bytes;
    }
    {
        const double stage_started = frontier_now_s();
        error = dpu_copy_to(dpu0, "RESIDENT_SLOT_POOL", frontier.request.slots[frontier.request.operations[1].slot_out_real].offset_bytes,
            frontier.intermediate, frontier.intermediate_transfer_bytes);
        timing.inter_wave_h2d_time_s = frontier_now_s() - stage_started;
        inter_wave_h2d = error == DPU_OK;
        if (!inter_wave_h2d) {
            frontier_report_sdk_error("frontier inter-wave dpu0 h2d", error);
            frontier_fail(&failure, "inter_wave_h2d_failed", "wave zero operation-one output write failed", 0, 1, 0, error);
            goto release_and_write;
        }
        inter_wave_h2d_bytes = frontier.intermediate_transfer_bytes;
    }
    {
        const double stage_started = frontier_now_s();
        if (frontier_set_active(dpu0, 2u, &operation_control_h2d_bytes, &error) != 0) {
            frontier_fail(&failure, "operation_control_h2d_failed", "wave one operation assignment failed", 1, 2, 0, error);
            goto release_and_write;
        }
        timing.control_h2d_time_s += frontier_now_s() - stage_started;
    }
    {
        const double stage_started = frontier_now_s();
        wave1_launch_attempted = 1;
        error = dpu_launch(dpu0, DPU_ASYNCHRONOUS);
        timing.wave1_launch_time_s = frontier_now_s() - stage_started;
        tasks[2].launched = error == DPU_OK;
        if (error != DPU_OK) {
            frontier_report_sdk_error("frontier wave one dpu0 launch", error);
            frontier_fail(&failure, "wave1_launch_failed", "wave one DPU0 asynchronous launch failed", 1, 2, 0, error);
            goto release_and_write;
        }
    }
    {
        const double stage_started = frontier_now_s();
        wave1_sync_attempted = 1;
        error = dpu_sync(dpu0);
        timing.wave1_sync_time_s = frontier_now_s() - stage_started;
        tasks[2].synchronized = error == DPU_OK;
        if (error != DPU_OK) {
            frontier_report_sdk_error("frontier wave one dpu0 sync", error);
            frontier_fail(&failure, "wave1_sync_failed", "wave one DPU0 synchronization failed", 1, 2, 0, error);
            goto release_and_write;
        }
        wave1_sync_succeeded = 1;
        if (frontier_read_completion(dpu0, &tasks[2], &frontier.request.operations[2], &failure, &error) != 0) goto release_and_write;
    }
    {
        const double stage_started = frontier_now_s();
        error = dpu_copy_from(dpu0, "RESIDENT_SLOT_POOL", frontier.request.slots[frontier.request.final_outputs[0].slot_id].offset_bytes,
            frontier.final_output, frontier.final_transfer_bytes);
        timing.final_d2h_time_s = frontier_now_s() - stage_started;
        final_d2h = error == DPU_OK && frontier_buffer_finite(frontier.final_output, frontier.request.final_outputs[0].raw_bytes) == 0;
        if (!final_d2h) {
            frontier_report_sdk_error("frontier final dpu0 d2h", error);
            frontier_fail(&failure, "final_d2h_failed", "final real output read failed", 1, 2, 0, error);
            goto release_and_write;
        }
        final_d2h_bytes = frontier.final_transfer_bytes;
    }
    {
        const double stage_started = frontier_now_s();
        output_written = frontier_write_exact(frontier.request.final_outputs[0].path, frontier.final_output, frontier.request.final_outputs[0].raw_bytes) == 0;
        timing.output_write_time_s = frontier_now_s() - stage_started;
        if (!output_written) {
            frontier_fail(&failure, "final_output_write_failed", "final output file write failed", 1, 2, 0, DPU_OK);
            goto release_and_write;
        }
    }
    goto release_and_write;

release_and_write:
    if (set_allocated) {
        const double stage_started = frontier_now_s();
        release_attempted = 1;
        error = dpu_free(set);
        timing.release_time_s = frontier_now_s() - stage_started;
        if (error == DPU_OK) release_confirmed = 1;
        else {
            frontier_report_sdk_error("frontier dpu_free", error);
            frontier_fail(&failure, "hardware_release_failed", "DPU set release was not confirmed", failure.wave, failure.operation, failure.dpu, error);
        }
    }
write_response:
    if (frontier_write_response(response_path, &frontier, &failure, &timing, tasks, allocated_dpus,
        allocation_attempted, load_attempted, load_succeeded, wave0_launch_attempted, wave0_sync_attempted,
        wave0_sync_succeeded, wave1_launch_attempted, wave1_sync_attempted, wave1_sync_succeeded,
        inter_wave_d2h, inter_wave_h2d, final_d2h, output_written, release_attempted, release_confirmed,
        descriptor_package_h2d_bytes, operation_control_h2d_bytes, inter_wave_d2h_bytes, inter_wave_h2d_bytes, final_d2h_bytes) != 0) rc = 1;
    else rc = failure.stage == NULL && release_confirmed ? 0 : 1;
    frontier_free_request(&frontier);
    (void)started;
    return rc;
}
