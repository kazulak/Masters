#define _POSIX_C_SOURCE 200809L

#include <dpu.h>

#include <ctype.h>
#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

#include "common.h"
#include "../upmem_sdk_generic_loop_resident/session_protocol.h"

#ifndef UPMEM_GENERIC_HARDWARE_MVP
#define UPMEM_GENERIC_HARDWARE_MVP 1
#endif

#if !UPMEM_GENERIC_HARDWARE_MVP
#error "two-DPU resident hardware evidence requires UPMEM_GENERIC_HARDWARE_MVP=1"
#endif

#define RESIDENT_TWO_DPU_ALLOCATION_PROFILE "backend=hw"
#define FNV1A64_OFFSET 14695981039346656037ULL
#define FNV1A64_PRIME 1099511628211ULL
#define RESIDENT_TWO_DPU_SLICE_SCHEMA "upmem_sliced_resident_execution_v1"

typedef struct {
    char *tensor_id;
    uint32_t label;
    uint32_t axis;
    uint32_t value;
} two_dpu_restriction_t;

typedef struct {
    uint32_t slice_id;
    uint32_t dpu_id;
    uint32_t sliced_label;
    uint32_t assignment_value;
    char *source_task_id;
    char *circuit_semantics_hash;
    char *tensor_network_hash;
    char *contraction_plan_hash;
    char *descriptor_sha256;
    uint64_t descriptor_fnv1a64;
    two_dpu_restriction_t restrictions[2];
    uint64_t restrictions_hash;
    uint64_t inputs_hash;
    uint64_t *input_fingerprints;
    size_t input_fingerprint_count;
} two_dpu_slice_execution_t;

typedef struct {
    uint32_t slice_id;
    const char *manifest_path;
    uint64_t manifest_hash;
    resident_request_t request;
    two_dpu_slice_execution_t execution;
    char *parse_error;
    unsigned char **inputs;
    unsigned char *partial_output;
    uint64_t package_bytes;
    uint64_t input_bytes;
    uint64_t partial_output_bytes;
    uint64_t partial_output_transfer_bytes;
    uint64_t partial_output_fnv1a64;
    uint32_t completed_operation_count;
    resident_completion_t completion_sentinel;
    uint32_t completion_sentinel_read_count;
    int package_transferred;
    int inputs_transferred;
    int completion_sentinel_read;
    int completion_sentinel_verified;
    int completion_confirmed;
    int partial_output_read;
    int partial_output_written;
} two_dpu_slice_t;

static double two_dpu_now_s(void) {
    struct timespec value;
    if (clock_gettime(CLOCK_MONOTONIC, &value) != 0) return 0.0;
    return (double)value.tv_sec + (double)value.tv_nsec / 1000000000.0;
}

static void two_dpu_report_sdk_error(const char *operation, dpu_error_t error) {
    fprintf(stderr, "%s failed: %s\n", operation, dpu_error_to_string(error));
}

static int two_dpu_file_size(const char *path, size_t expected) {
    struct stat value;
    return stat(path, &value) != 0 || value.st_size < 0 || (uintmax_t)value.st_size != (uintmax_t)expected;
}

static int two_dpu_read_exact(const char *path, void *buffer, size_t bytes) {
    FILE *file = fopen(path, "rb");
    size_t read_bytes;
    if (file == NULL) return 1;
    read_bytes = fread(buffer, 1u, bytes, file);
    fclose(file);
    return read_bytes == bytes ? 0 : 1;
}

static int two_dpu_write_exact(const char *path, const void *buffer, size_t bytes) {
    FILE *file = fopen(path, "wb");
    size_t written;
    int close_failed;
    if (file == NULL) return 1;
    written = fwrite(buffer, 1u, bytes, file);
    close_failed = fclose(file) != 0;
    return written == bytes && !close_failed ? 0 : 1;
}

static int two_dpu_buffer_finite(const unsigned char *buffer, size_t bytes) {
    for (size_t offset = 0; offset < bytes; offset += sizeof(float)) {
        float value;
        memcpy(&value, buffer + offset, sizeof(value));
        if (!isfinite(value)) return 1;
    }
    return 0;
}

static int two_dpu_hash_file(const char *path, uint64_t *hash) {
    unsigned char buffer[4096];
    FILE *file = fopen(path, "rb");
    size_t bytes;
    uint64_t value = FNV1A64_OFFSET;
    if (file == NULL) return 1;
    while ((bytes = fread(buffer, 1u, sizeof(buffer), file)) != 0u) {
        for (size_t index = 0; index < bytes; index++) {
            value ^= buffer[index];
            value *= FNV1A64_PRIME;
        }
    }
    if (ferror(file) != 0 || fclose(file) != 0) return 1;
    *hash = value;
    return 0;
}

static int two_dpu_safe_relative_path(const char *path) {
    const char *cursor = path;
    if (path == NULL || path[0] == '\0' || path[0] == '/') return 1;
    while (*cursor != '\0') {
        const char *segment = cursor;
        while (*cursor != '\0' && *cursor != '/') cursor++;
        if ((size_t)(cursor - segment) == 2u && segment[0] == '.' && segment[1] == '.') return 1;
        if (*cursor == '/') cursor++;
    }
    return 0;
}

static int two_dpu_resolve_manifest_path(const char *root, const char *relative, char **resolved) {
    const size_t root_length = strlen(root);
    const size_t relative_length = strlen(relative);
    char *path;
    if (two_dpu_safe_relative_path(relative) != 0 || root_length > SIZE_MAX - relative_length - 2u) return 1;
    path = (char *)malloc(root_length + relative_length + 2u);
    if (path == NULL) return 1;
    memcpy(path, root, root_length);
    path[root_length] = '/';
    memcpy(path + root_length + 1u, relative, relative_length + 1u);
    *resolved = path;
    return 0;
}

static uint64_t two_dpu_hash_bytes(const char *begin, const char *end) {
    uint64_t value = FNV1A64_OFFSET;
    for (const char *cursor = begin; cursor < end; cursor++) {
        value ^= (unsigned char)*cursor;
        value *= FNV1A64_PRIME;
    }
    return value;
}

static const char *two_dpu_skip_space(const char *cursor, const char *end) {
    while (cursor < end && isspace((unsigned char)*cursor)) cursor++;
    return cursor;
}

static int two_dpu_matching_end(const char *start, const char *end, char opening, char closing, const char **match) {
    int depth = 0;
    int in_string = 0;
    int escaped = 0;
    for (const char *cursor = start; cursor < end; cursor++) {
        const unsigned char character = (unsigned char)*cursor;
        if (in_string) {
            if (escaped) escaped = 0;
            else if (character == '\\') escaped = 1;
            else if (character == '"') in_string = 0;
        } else if (character == '"') in_string = 1;
        else if (character == (unsigned char)opening) depth++;
        else if (character == (unsigned char)closing && --depth == 0) {
            *match = cursor;
            return 0;
        }
    }
    return 1;
}

static int two_dpu_unique_value(const char *object, const char *end, const char *key, const char **value) {
    const size_t key_length = strlen(key);
    size_t needle_length;
    char *needle = NULL;
    const char *found = NULL;
    const char *cursor = object;
    int failed = 1;
    if (key_length > SIZE_MAX - 3u) return 1;
    needle_length = key_length + 2u;
    needle = (char *)malloc(needle_length + 1u);
    if (needle == NULL) return 1;
    needle[0] = '"';
    memcpy(needle + 1u, key, key_length);
    needle[key_length + 1u] = '"';
    needle[needle_length] = '\0';
    while ((cursor = strstr(cursor, needle)) != NULL && cursor < end) {
        if (found != NULL) goto done;
        if ((size_t)(end - cursor) < needle_length) goto done;
        found = cursor + needle_length;
        cursor = found;
    }
    if (found == NULL || found >= end) goto done;
    found = two_dpu_skip_space(found, end);
    if (found >= end || *found++ != ':') goto done;
    *value = two_dpu_skip_space(found, end);
    failed = *value >= end;
done:
    free(needle);
    return failed;
}

static int two_dpu_json_string_field(const char *object, const char *end, const char *key, char **value) {
    const char *cursor;
    const char *start;
    size_t length;
    char *copy;
    if (two_dpu_unique_value(object, end, key, &cursor) != 0 || cursor >= end || *cursor++ != '"') return 1;
    start = cursor;
    while (cursor < end && *cursor != '"') {
        if (*cursor == '\\' || (unsigned char)*cursor < 0x20u) return 1;
        cursor++;
    }
    if (cursor >= end || cursor == start) return 1;
    length = (size_t)(cursor - start);
    copy = (char *)malloc(length + 1u);
    if (copy == NULL) return 1;
    memcpy(copy, start, length);
    copy[length] = '\0';
    *value = copy;
    return 0;
}

static int two_dpu_json_uint(const char *object, const char *end, const char *key, uint32_t *value) {
    const char *cursor;
    char *number_end;
    unsigned long parsed;
    if (two_dpu_unique_value(object, end, key, &cursor) != 0 || cursor >= end || !isdigit((unsigned char)*cursor)) return 1;
    errno = 0;
    parsed = strtoul(cursor, &number_end, 10);
    if (errno != 0 || number_end == cursor || parsed > UINT32_MAX || number_end > end) return 1;
    number_end = (char *)two_dpu_skip_space(number_end, end);
    if (number_end != end && *number_end != ',' && *number_end != '}' && *number_end != ']') return 1;
    *value = (uint32_t)parsed;
    return 0;
}

static int two_dpu_json_fnv1a64_field(const char *object, const char *end, const char *key, uint64_t *value) {
    char *text = NULL;
    uint64_t parsed = 0u;
    if (two_dpu_json_string_field(object, end, key, &text) != 0) return 1;
    if (strlen(text) != 16u) {
        free(text);
        return 1;
    }
    for (const unsigned char *cursor = (const unsigned char *)text; *cursor != '\0'; cursor++) {
        uint64_t digit;
        if (*cursor >= '0' && *cursor <= '9') digit = (uint64_t)(*cursor - '0');
        else if (*cursor >= 'a' && *cursor <= 'f') digit = (uint64_t)(*cursor - 'a' + 10u);
        else {
            free(text);
            return 1;
        }
        parsed = (parsed << 4u) | digit;
    }
    free(text);
    *value = parsed;
    return 0;
}

static int two_dpu_json_container(const char *object, const char *end, const char *key, char opening, char closing,
    const char **begin, const char **container_end) {
    const char *cursor;
    if (two_dpu_unique_value(object, end, key, &cursor) != 0 || cursor >= end || *cursor != opening ||
        two_dpu_matching_end(cursor, end, opening, closing, container_end) != 0) return 1;
    *begin = cursor;
    return 0;
}

static int two_dpu_sha256(const char *value) {
    if (strlen(value) != 64u) return 1;
    for (const unsigned char *cursor = (const unsigned char *)value; *cursor != '\0'; cursor++) {
        if (!isdigit(*cursor) && (*cursor < 'a' || *cursor > 'f')) return 1;
    }
    return 0;
}

static int two_dpu_json_string_value(const char **cursor, const char *end, char **value) {
    const char *start;
    size_t length;
    char *copy;
    if (*cursor >= end || *(*cursor)++ != '"') return 1;
    start = *cursor;
    while (*cursor < end && **cursor != '"') {
        if (**cursor == '\\' || (unsigned char)**cursor < 0x20u) return 1;
        (*cursor)++;
    }
    if (*cursor >= end || *cursor == start) return 1;
    length = (size_t)(*cursor - start);
    copy = (char *)malloc(length + 1u);
    if (copy == NULL) return 1;
    memcpy(copy, start, length);
    copy[length] = '\0';
    *value = copy;
    (*cursor)++;
    return 0;
}

static int two_dpu_validate_sha256_object(const char *object, const char *object_end) {
    const char *cursor = object + 1;
    size_t count = 0u;
    while ((cursor = two_dpu_skip_space(cursor, object_end)) < object_end) {
        char *key = NULL;
        char *value = NULL;
        if (two_dpu_json_string_value(&cursor, object_end, &key) != 0 ||
            (cursor = two_dpu_skip_space(cursor, object_end)) >= object_end || *cursor++ != ':' ||
            (cursor = two_dpu_skip_space(cursor, object_end)) >= object_end ||
            two_dpu_json_string_value(&cursor, object_end, &value) != 0 || two_dpu_sha256(value) != 0) {
            free(key);
            free(value);
            return 1;
        }
        free(key);
        free(value);
        count++;
        cursor = two_dpu_skip_space(cursor, object_end);
        if (cursor == object_end) break;
        if (*cursor++ != ',') return 1;
    }
    return count >= 2u && count <= RESIDENT_MAX_SLOT_DESCRIPTORS ? 0 : 1;
}

static int two_dpu_validate_restrictions(const char *array, const char *array_end, two_dpu_slice_execution_t *execution) {
    const char *cursor = array + 1;
    uint32_t count = 0u;
    int failed = 0;
    while ((cursor = two_dpu_skip_space(cursor, array_end)) < array_end) {
        const char *object_end;
        if (count >= 2u || *cursor != '{' || two_dpu_matching_end(cursor, array_end, '{', '}', &object_end) != 0 ||
            two_dpu_json_string_field(cursor, object_end + 1, "tensor_id", &execution->restrictions[count].tensor_id) != 0 ||
            two_dpu_json_uint(cursor, object_end + 1, "label", &execution->restrictions[count].label) != 0 ||
            two_dpu_json_uint(cursor, object_end + 1, "axis", &execution->restrictions[count].axis) != 0 ||
            two_dpu_json_uint(cursor, object_end + 1, "value", &execution->restrictions[count].value) != 0 ||
            execution->restrictions[count].label != execution->sliced_label ||
            execution->restrictions[count].value != execution->assignment_value) {
            failed = 1;
            break;
        }
        count++;
        cursor = two_dpu_skip_space(object_end + 1, array_end);
        if (cursor == array_end) break;
        if (*cursor++ != ',') {
            failed = 1;
            break;
        }
        if (two_dpu_skip_space(cursor, array_end) == array_end) {
            failed = 1;
            break;
        }
    }
    if (count != 2u || execution->restrictions[0].tensor_id == NULL || execution->restrictions[1].tensor_id == NULL ||
        strcmp(execution->restrictions[0].tensor_id, execution->restrictions[1].tensor_id) == 0) failed = 1;
    return failed;
}

static void two_dpu_free_execution(two_dpu_slice_execution_t *execution) {
    free(execution->source_task_id);
    free(execution->circuit_semantics_hash);
    free(execution->tensor_network_hash);
    free(execution->contraction_plan_hash);
    free(execution->descriptor_sha256);
    free(execution->restrictions[0].tensor_id);
    free(execution->restrictions[1].tensor_id);
    free(execution->input_fingerprints);
    memset(execution, 0, sizeof(*execution));
}

static int two_dpu_load_input_fingerprints(
    const char *fnv1a64,
    const char *fnv1a64_end,
    const char *input_files,
    const char *input_files_end,
    two_dpu_slice_execution_t *execution
) {
    const char *cursor = input_files + 1;
    size_t count = 0u;
    while ((cursor = two_dpu_skip_space(cursor, input_files_end)) < input_files_end) {
        char *input_path = NULL;
        const char *object_end;
        uint64_t fingerprint;
        uint64_t *grown;
        cursor = two_dpu_skip_space(cursor, input_files_end);
        if (cursor >= input_files_end || *cursor != '{' ||
            two_dpu_matching_end(cursor, input_files_end, '{', '}', &object_end) != 0 ||
            two_dpu_json_string_field(cursor, object_end + 1, "input_path", &input_path) != 0 ||
            two_dpu_json_fnv1a64_field(fnv1a64, fnv1a64_end + 1, input_path, &fingerprint) != 0) {
            free(input_path);
            return 1;
        }
        if (count >= RESIDENT_MAX_SLOT_DESCRIPTORS) {
            free(input_path);
            return 1;
        }
        grown = (uint64_t *)realloc(
            execution->input_fingerprints,
            (count + 1u) * sizeof(*execution->input_fingerprints)
        );
        if (grown == NULL) {
            free(input_path);
            return 1;
        }
        execution->input_fingerprints = grown;
        execution->input_fingerprints[count++] = fingerprint;
        free(input_path);
        cursor = two_dpu_skip_space(object_end + 1, input_files_end);
        if (cursor == input_files_end) break;
        if (*cursor++ != ',') return 1;
    }
    execution->input_fingerprint_count = count;
    return count >= 2u && two_dpu_skip_space(cursor, input_files_end) == input_files_end ? 0 : 1;
}

static int two_dpu_load_descriptor_path(const char *manifest_path, const char *manifest_root, char **package_path) {
    FILE *file = fopen(manifest_path, "rb");
    long length;
    char *bytes = NULL;
    char *reference = NULL;
    int failed = 1;
    if (file == NULL || fseek(file, 0, SEEK_END) != 0 || (length = ftell(file)) < 0 || fseek(file, 0, SEEK_SET) != 0 ||
        (bytes = (char *)malloc((size_t)length + 1u)) == NULL || fread(bytes, 1u, (size_t)length, file) != (size_t)length) goto done;
    bytes[length] = '\0';
    if (two_dpu_json_string_field(bytes, bytes + length, "package_path", &reference) != 0 ||
        two_dpu_resolve_manifest_path(manifest_root, reference, package_path) != 0) goto done;
    failed = 0;
done:
    free(reference);
    free(bytes);
    if (file != NULL) fclose(file);
    return failed;
}

static int two_dpu_verify_slice_fingerprints(const two_dpu_slice_t *slice, const char **reason) {
    char *package_path = NULL;
    uint64_t descriptor_hash;
    if (slice->request.manifest_root == NULL ||
        two_dpu_load_descriptor_path(slice->manifest_path, slice->request.manifest_root, &package_path) != 0 ||
        two_dpu_hash_file(package_path, &descriptor_hash) != 0 ||
        descriptor_hash != slice->execution.descriptor_fnv1a64) {
        free(package_path);
        *reason = "slice_descriptor_fingerprint_mismatch";
        return 1;
    }
    free(package_path);
    if (slice->request.input_count != slice->execution.input_fingerprint_count) {
        *reason = "slice_input_fingerprint_count_mismatch";
        return 1;
    }
    for (size_t index = 0; index < slice->request.input_count; index++) {
        uint64_t input_hash;
        const resident_input_file_t *input = &slice->request.inputs[index];
        if (two_dpu_file_size(input->path, input->raw_bytes) != 0 ||
            two_dpu_hash_file(input->path, &input_hash) != 0 ||
            input_hash != slice->execution.input_fingerprints[index]) {
            *reason = "slice_restricted_input_fingerprint_mismatch";
            return 1;
        }
    }
    return 0;
}

static int two_dpu_load_slice_execution(const char *path, two_dpu_slice_execution_t *execution) {
    FILE *file = fopen(path, "rb");
    long length;
    char *bytes = NULL;
    const char *object;
    const char *object_end;
    const char *source_hashes;
    const char *source_hashes_end;
    const char *restrictions;
    const char *restrictions_end;
    const char *inputs;
    const char *inputs_end;
    const char *input_fingerprints;
    const char *input_fingerprints_end;
    const char *input_files;
    const char *input_files_end;
    char *schema = NULL;
    char *contract = NULL;
    int failed = 1;
    memset(execution, 0, sizeof(*execution));
    if (file == NULL || fseek(file, 0, SEEK_END) != 0 || (length = ftell(file)) < 0 || fseek(file, 0, SEEK_SET) != 0 ||
        (bytes = (char *)malloc((size_t)length + 1u)) == NULL || fread(bytes, 1u, (size_t)length, file) != (size_t)length) goto done;
    bytes[length] = '\0';
    if (two_dpu_json_container(bytes, bytes + length, "slice_execution", '{', '}', &object, &object_end) != 0 ||
        two_dpu_json_string_field(object, object_end + 1, "schema_version", &schema) != 0 ||
        strcmp(schema, RESIDENT_TWO_DPU_SLICE_SCHEMA) != 0 ||
        two_dpu_json_uint(object, object_end + 1, "slice_id", &execution->slice_id) != 0 ||
        two_dpu_json_uint(object, object_end + 1, "dpu_id", &execution->dpu_id) != 0 ||
        two_dpu_json_string_field(object, object_end + 1, "source_task_id", &execution->source_task_id) != 0 ||
        two_dpu_json_uint(object, object_end + 1, "sliced_label", &execution->sliced_label) != 0 ||
        two_dpu_json_string_field(object, object_end + 1, "resident_descriptor_sha256", &execution->descriptor_sha256) != 0 ||
        two_dpu_json_fnv1a64_field(object, object_end + 1, "resident_descriptor_fnv1a64", &execution->descriptor_fnv1a64) != 0 ||
        two_dpu_json_string_field(object, object_end + 1, "reconstruction_contract", &contract) != 0 ||
        strcmp(contract, "python_sum_partials") != 0 || two_dpu_sha256(execution->descriptor_sha256) != 0 ||
        two_dpu_json_container(object, object_end + 1, "source_hashes", '{', '}', &source_hashes, &source_hashes_end) != 0 ||
        two_dpu_json_string_field(source_hashes, source_hashes_end + 1, "circuit_semantics_hash", &execution->circuit_semantics_hash) != 0 ||
        two_dpu_json_string_field(source_hashes, source_hashes_end + 1, "tensor_network_hash", &execution->tensor_network_hash) != 0 ||
        two_dpu_json_string_field(source_hashes, source_hashes_end + 1, "contraction_plan_hash", &execution->contraction_plan_hash) != 0 ||
        two_dpu_sha256(execution->circuit_semantics_hash) != 0 || two_dpu_sha256(execution->tensor_network_hash) != 0 ||
        two_dpu_sha256(execution->contraction_plan_hash) != 0 ||
        two_dpu_json_uint(object, object_end + 1, "assignment_value", &execution->assignment_value) != 0 ||
        two_dpu_json_container(object, object_end + 1, "restrictions", '[', ']', &restrictions, &restrictions_end) != 0 ||
        two_dpu_validate_restrictions(restrictions, restrictions_end, execution) != 0 ||
        two_dpu_json_container(object, object_end + 1, "restricted_input_sha256", '{', '}', &inputs, &inputs_end) != 0 ||
        two_dpu_json_container(object, object_end + 1, "restricted_input_fnv1a64", '{', '}', &input_fingerprints, &input_fingerprints_end) != 0 ||
        two_dpu_validate_sha256_object(inputs, inputs_end) != 0 ||
        two_dpu_json_container(bytes, bytes + length, "initial_slots", '[', ']', &input_files, &input_files_end) != 0 ||
        two_dpu_load_input_fingerprints(input_fingerprints, input_fingerprints_end, input_files, input_files_end,
            execution) != 0) goto done;
    execution->restrictions_hash = two_dpu_hash_bytes(restrictions, restrictions_end + 1);
    execution->inputs_hash = two_dpu_hash_bytes(inputs, inputs_end + 1);
    failed = 0;
done:
    free(schema);
    free(contract);
    free(bytes);
    if (file != NULL) fclose(file);
    if (failed) two_dpu_free_execution(execution);
    return failed;
}

static void two_dpu_json_string(FILE *file, const char *value) {
    fputc('"', file);
    for (const unsigned char *cursor = (const unsigned char *)(value == NULL ? "" : value); *cursor != '\0'; cursor++) {
        if (*cursor == '"' || *cursor == '\\') fputc('\\', file);
        if (*cursor == '\n') fputs("\\n", file);
        else if (*cursor == '\t') fputs("\\t", file);
        else fputc(*cursor, file);
    }
    fputc('"', file);
}

static int two_dpu_validate_slice_graph(const two_dpu_slice_t *slice, const char **reason) {
    const resident_operation_t *operation;
    const resident_final_file_t *output;
    if (slice->request.header.operation_count == 0u || slice->request.final_count != 1u) {
        *reason = "slice_package_requires_operations_and_one_partial_output";
        return 1;
    }
    operation = &slice->request.operations[slice->request.header.operation_count - 1u];
    output = &slice->request.final_outputs[0];
    if (operation->kind != RESIDENT_OPERATION_CONTRACT || operation->slot_out_real != output->slot_id ||
        operation->output_elements != output->elements) {
        *reason = "slice_package_requires_one_real_final_contract_partial";
        return 1;
    }
    return 0;
}

static int two_dpu_validate_restriction_pair(
    const two_dpu_slice_execution_t *left,
    const two_dpu_slice_execution_t *right
) {
    uint8_t matched[2] = {0u, 0u};
    for (size_t left_index = 0; left_index < 2u; left_index++) {
        const two_dpu_restriction_t *left_restriction = &left->restrictions[left_index];
        int found = 0;
        if (left_restriction->value != 0u) return 1;
        for (size_t right_index = 0; right_index < 2u; right_index++) {
            const two_dpu_restriction_t *right_restriction = &right->restrictions[right_index];
            if (!matched[right_index] && strcmp(left_restriction->tensor_id, right_restriction->tensor_id) == 0 &&
                left_restriction->label == right_restriction->label && left_restriction->axis == right_restriction->axis &&
                right_restriction->value == 1u) {
                matched[right_index] = 1u;
                found = 1;
                break;
            }
        }
        if (!found) return 1;
    }
    return 0;
}

static int two_dpu_validate_slice_pair(const two_dpu_slice_t slices[RESIDENT_TWO_DPU_COUNT], const char **reason) {
    if (slices[0].request.quantization_mode == NULL || slices[1].request.quantization_mode == NULL ||
        (strcmp(slices[0].request.quantization_mode, "none") != 0 &&
            strcmp(slices[0].request.quantization_mode, "per_task_resident_requantize") != 0) ||
        (strcmp(slices[1].request.quantization_mode, "none") != 0 &&
            strcmp(slices[1].request.quantization_mode, "per_task_resident_requantize") != 0)) {
        *reason = "slice_packages_require_supported_quantization_mode";
        return 1;
    }
    if (strcmp(slices[0].request.quantization_mode, slices[1].request.quantization_mode) != 0) {
        *reason = "slice_packages_require_matching_quantization_mode";
        return 1;
    }
    if (strcmp(slices[0].manifest_path, slices[1].manifest_path) == 0 || slices[0].manifest_hash == slices[1].manifest_hash) {
        *reason = "slice_packages_must_be_distinct";
        return 1;
    }
    if (two_dpu_validate_slice_graph(&slices[0], reason) != 0 || two_dpu_validate_slice_graph(&slices[1], reason) != 0) return 1;
    if (slices[0].request.input_count < 2u || slices[1].request.input_count < 2u) {
        *reason = "slice_packages_require_all_initial_inputs";
        return 1;
    }
    if (slices[0].execution.slice_id != 0u || slices[1].execution.slice_id != 1u ||
        slices[0].execution.dpu_id != 0u || slices[1].execution.dpu_id != 1u ||
        slices[0].execution.assignment_value != 0u || slices[1].execution.assignment_value != 1u) {
        *reason = "slice_execution_assignment_mismatch";
        return 1;
    }
    if (strcmp(slices[0].execution.source_task_id, slices[1].execution.source_task_id) != 0 ||
        strcmp(slices[0].execution.circuit_semantics_hash, slices[1].execution.circuit_semantics_hash) != 0 ||
        strcmp(slices[0].execution.tensor_network_hash, slices[1].execution.tensor_network_hash) != 0 ||
        strcmp(slices[0].execution.contraction_plan_hash, slices[1].execution.contraction_plan_hash) != 0) {
        *reason = "slice_execution_source_identity_mismatch";
        return 1;
    }
    if (slices[0].execution.sliced_label != slices[1].execution.sliced_label) {
        *reason = "slice_execution_sliced_label_mismatch";
        return 1;
    }
    if (two_dpu_validate_restriction_pair(&slices[0].execution, &slices[1].execution) != 0) {
        *reason = "slice_execution_restrictions_mismatch";
        return 1;
    }
    if (slices[0].execution.restrictions_hash == slices[1].execution.restrictions_hash) {
        *reason = "slice_execution_restrictions_must_be_distinct";
        return 1;
    }
    if (slices[0].execution.descriptor_fnv1a64 != slices[1].execution.descriptor_fnv1a64) {
        *reason = "slice_execution_descriptor_fingerprint_mismatch";
        return 1;
    }
    if (two_dpu_verify_slice_fingerprints(&slices[0], reason) != 0 ||
        two_dpu_verify_slice_fingerprints(&slices[1], reason) != 0) return 1;
    if (slices[0].request.header.operation_count != slices[1].request.header.operation_count) {
        *reason = "slice_packages_require_matching_operation_count";
        return 1;
    }
    {
        const uint32_t operation_count = slices[0].request.header.operation_count;
        const resident_operation_t *left = &slices[0].request.operations[operation_count - 1u];
        const resident_operation_t *right = &slices[1].request.operations[operation_count - 1u];
        const resident_final_file_t *left_output = &slices[0].request.final_outputs[0];
        const resident_final_file_t *right_output = &slices[1].request.final_outputs[0];
        if (strcmp(slices[0].request.dpu_binary_path, slices[1].request.dpu_binary_path) != 0) {
            *reason = "slice_packages_require_the_same_dpu_binary";
            return 1;
        }
        if (slices[0].request.header.pool_bytes != slices[1].request.header.pool_bytes ||
            slices[0].request.header.max_rank != slices[1].request.header.max_rank ||
            memcmp(slices[0].request.operations, slices[1].request.operations,
                (size_t)operation_count * sizeof(resident_operation_t)) != 0 ||
            left->args.output_rank != right->args.output_rank ||
            memcmp(left->args.output_shape, right->args.output_shape, sizeof(left->args.output_shape)) != 0 ||
            memcmp(left->args.output_strides, right->args.output_strides, sizeof(left->args.output_strides)) != 0 ||
            left_output->elements != right_output->elements || left_output->raw_bytes != right_output->raw_bytes ||
            left_output->transfer_bytes != right_output->transfer_bytes) {
            *reason = "slice_packages_require_compatible_full_shape_outputs";
            return 1;
        }
        if (strcmp(left_output->path, right_output->path) == 0) {
            *reason = "slice_partial_output_paths_must_be_distinct";
            return 1;
        }
        for (size_t left_input = 0; left_input < slices[0].request.input_count; left_input++) {
            for (size_t right_input = 0; right_input < slices[1].request.input_count; right_input++) {
                if (strcmp(slices[0].request.inputs[left_input].path, slices[1].request.inputs[right_input].path) == 0) {
                    *reason = "slice_input_paths_must_be_distinct";
                    return 1;
                }
            }
        }
    }
    return 0;
}

static int two_dpu_load_slice(two_dpu_slice_t *slice, const char *path, const char **reason) {
    const uint32_t slice_id = slice->slice_id;
    memset(slice, 0, sizeof(*slice));
    slice->slice_id = slice_id;
    slice->manifest_path = path;
    if (two_dpu_hash_file(path, &slice->manifest_hash) != 0) {
        *reason = "slice_manifest_hash_failed";
        return 1;
    }
    if (resident_request_load(path, &slice->request, &slice->parse_error) != 0) {
        *reason = "slice_manifest_parse_failed";
        return 1;
    }
    if (two_dpu_load_slice_execution(path, &slice->execution) != 0) {
        free(slice->parse_error);
        slice->parse_error = strdup("slice_execution_parse_failed");
        *reason = "slice_execution_parse_failed";
        return 1;
    }
    return 0;
}

static int two_dpu_prepare_slice_inputs(two_dpu_slice_t *slice, const char **reason) {
    slice->inputs = (unsigned char **)calloc(slice->request.input_count, sizeof(*slice->inputs));
    if (slice->inputs == NULL && slice->request.input_count != 0u) {
        *reason = "slice_input_allocation_failed";
        return 1;
    }
    for (size_t index = 0; index < slice->request.input_count; index++) {
        const resident_input_file_t *input = &slice->request.inputs[index];
        if (input->slot_id >= slice->request.header.slot_count || two_dpu_file_size(input->path, input->raw_bytes) != 0) {
            *reason = "slice_input_file_invalid";
            return 1;
        }
        slice->inputs[index] = (unsigned char *)calloc(input->transfer_bytes, 1u);
        if (slice->inputs[index] == NULL || two_dpu_read_exact(input->path, slice->inputs[index], input->raw_bytes) != 0 ||
            two_dpu_buffer_finite(slice->inputs[index], input->raw_bytes) != 0) {
            *reason = "slice_input_load_failed";
            return 1;
        }
        slice->input_bytes += input->transfer_bytes;
    }
    slice->partial_output = (unsigned char *)calloc(slice->request.final_outputs[0].transfer_bytes, 1u);
    if (slice->partial_output == NULL) {
        *reason = "slice_partial_output_allocation_failed";
        return 1;
    }
    return 0;
}

static void two_dpu_free_slice(two_dpu_slice_t *slice) {
    if (slice->inputs != NULL) {
        for (size_t index = 0; index < slice->request.input_count; index++) free(slice->inputs[index]);
    }
    free(slice->inputs);
    free(slice->partial_output);
    free(slice->parse_error);
    two_dpu_free_execution(&slice->execution);
    resident_request_free(&slice->request);
    memset(slice, 0, sizeof(*slice));
}

static int two_dpu_transfer_slice_package(struct dpu_set_t dpu, two_dpu_slice_t *slice, dpu_error_t *error) {
    const resident_control_t control = {
        slice->request.header.slot_count,
        slice->request.header.operation_count,
        slice->request.header.pool_bytes,
        0u,
    };
    *error = dpu_copy_to(dpu, "RESIDENT_SLOT_DESCRIPTORS", 0u, slice->request.slots, slice->request.header.slot_bytes);
    if (*error == DPU_OK) *error = dpu_copy_to(dpu, "RESIDENT_OPERATIONS", 0u, slice->request.operations, slice->request.header.operation_bytes);
    if (*error == DPU_OK) *error = dpu_copy_to(dpu, "RESIDENT_CONTROL", 0u, &control, sizeof(control));
    if (*error != DPU_OK) return 1;
    slice->package_bytes = slice->request.header.slot_bytes + slice->request.header.operation_bytes + sizeof(control);
    slice->package_transferred = 1;
    for (size_t index = 0; index < slice->request.input_count; index++) {
        const resident_input_file_t *input = &slice->request.inputs[index];
        *error = dpu_copy_to(dpu, "RESIDENT_SLOT_POOL", slice->request.slots[input->slot_id].offset_bytes,
            slice->inputs[index], input->transfer_bytes);
        if (*error != DPU_OK) return 1;
    }
    slice->inputs_transferred = 1;
    return 0;
}

static int two_dpu_set_active_operation(
    struct dpu_set_t dpu,
    const two_dpu_slice_t *slice,
    uint32_t operation_index,
    dpu_error_t *error
) {
    const uint64_t active_operation = operation_index;
    if (operation_index >= slice->request.header.operation_count) return 1;
    *error = dpu_copy_to(dpu, "RESIDENT_ACTIVE_OPERATION", 0u, &active_operation, sizeof(active_operation));
    return *error == DPU_OK ? 0 : 1;
}

static int two_dpu_read_slice_partial(struct dpu_set_t dpu, two_dpu_slice_t *slice, dpu_error_t *error) {
    const resident_final_file_t *output = &slice->request.final_outputs[0];
    *error = dpu_copy_from(dpu, "RESIDENT_SLOT_POOL", slice->request.slots[output->slot_id].offset_bytes,
        slice->partial_output, output->transfer_bytes);
    if (*error != DPU_OK || two_dpu_buffer_finite(slice->partial_output, output->raw_bytes) != 0) return 1;
    slice->partial_output_bytes = output->raw_bytes;
    slice->partial_output_transfer_bytes = output->transfer_bytes;
    {
        uint64_t value = FNV1A64_OFFSET;
        for (size_t index = 0; index < output->raw_bytes; index++) {
            value ^= slice->partial_output[index];
            value *= FNV1A64_PRIME;
        }
        slice->partial_output_fnv1a64 = value;
    }
    slice->partial_output_read = 1;
    return 0;
}

static int two_dpu_read_completion_sentinel(
    struct dpu_set_t dpu,
    two_dpu_slice_t *slice,
    uint32_t operation_index,
    dpu_error_t *error
) {
    const resident_operation_t *operation = &slice->request.operations[operation_index];
    resident_completion_t sentinel = {0};
    *error = dpu_copy_from(dpu, "RESIDENT_COMPLETION", 0u, &sentinel, sizeof(sentinel));
    if (*error != DPU_OK) return 1;
    slice->completion_sentinel = sentinel;
    slice->completion_sentinel_read = 1;
    if (sentinel.magic != RESIDENT_COMPLETION_MAGIC ||
        sentinel.version != RESIDENT_COMPLETION_VERSION ||
        sentinel.active_operation_index != operation_index ||
        sentinel.completion_status != RESIDENT_COMPLETION_COMPLETED ||
        sentinel.completed_operation_count != operation_index + 1u ||
        sentinel.output_elements_processed != operation->output_elements) {
        return 1;
    }
    slice->completed_operation_count = sentinel.completed_operation_count;
    slice->completion_sentinel_verified = 1;
    slice->completion_sentinel_read_count++;
    return 0;
}

static int two_dpu_write_slice_partial(two_dpu_slice_t *slice) {
    const resident_final_file_t *output = &slice->request.final_outputs[0];
    if (!slice->partial_output_read ||
        two_dpu_write_exact(output->path, slice->partial_output, output->raw_bytes) != 0) return 1;
    slice->request.final_outputs[0].status = 1;
    slice->partial_output_written = 1;
    return 0;
}

static int two_dpu_write_validation_result(const two_dpu_slice_t slices[RESIDENT_TWO_DPU_COUNT], const char *reason) {
    printf("{\"status\":\"%s\",\"reason\":", reason == NULL ? "valid" : "invalid");
    if (reason == NULL) fputs("null", stdout); else two_dpu_json_string(stdout, reason);
    printf(",\"slice_package_paths_distinct\":%s,\"slice_package_hashes_distinct\":%s,\"native_reconstruction_performed\":false,\"reconstruction_contract\":\"python_sum_partials\"}\n",
        strcmp(slices[0].manifest_path, slices[1].manifest_path) != 0 ? "true" : "false",
        slices[0].manifest_hash != slices[1].manifest_hash ? "true" : "false");
    return reason == NULL ? 0 : 1;
}

typedef struct {
    double package_parse_time_s;
    double allocation_time_s;
    double binary_load_time_s;
    double initial_h2d_time_s;
    double operation_control_h2d_time_s;
    double launch_enqueue_time_s;
    double sync_wait_time_s;
    double final_d2h_time_s;
    double output_write_time_s;
    double release_time_s;
} two_dpu_timing_t;

static int two_dpu_write_response(
    const char *path,
    const two_dpu_slice_t slices[RESIDENT_TWO_DPU_COUNT],
    const char *status,
    const char *failure_stage,
    const char *error_message,
    uint32_t allocated_dpus,
    int release_attempted,
    int release_confirmed,
    uint32_t operation_count,
    uint32_t async_launch_count,
    uint32_t synchronize_count,
    uint64_t initial_h2d_bytes,
    uint64_t descriptor_h2d_bytes,
    uint64_t control_h2d_bytes,
    uint64_t final_d2h_bytes,
    const two_dpu_timing_t *timing,
    double elapsed_s,
    int launch_in_flight
) {
    const two_dpu_timing_t empty_timing = {0};
    const two_dpu_timing_t *current = timing == NULL ? &empty_timing : timing;
    int slices_completed = operation_count != 0u;
    uint64_t actual_h2d_bytes;
    uint64_t actual_transfer_bytes;
    int completed;
    const char *response_status;
    const char *quantization_mode = "unknown";
    if (slices[0].request.quantization_mode != NULL && slices[1].request.quantization_mode != NULL &&
        strcmp(slices[0].request.quantization_mode, slices[1].request.quantization_mode) == 0) {
        quantization_mode = slices[0].request.quantization_mode;
    }
    actual_h2d_bytes = initial_h2d_bytes + descriptor_h2d_bytes + control_h2d_bytes;
    actual_transfer_bytes = actual_h2d_bytes + final_d2h_bytes;
    for (uint32_t index = 0; index < RESIDENT_TWO_DPU_COUNT; index++) {
        const two_dpu_slice_t *slice = &slices[index];
        slices_completed = slices_completed && slice->completed_operation_count == operation_count &&
            slice->completion_sentinel_verified && slice->partial_output_read && slice->partial_output_written;
    }
    const int device_completion_confirmed = !launch_in_flight && slices_completed;
    const char *device_completion_state = device_completion_confirmed
        ? "confirmed"
        : (launch_in_flight ? "unknown_after_launch" : "not_confirmed");
    completed = strcmp(status, "completed") == 0 && failure_stage == NULL &&
        allocated_dpus == RESIDENT_TWO_DPU_COUNT && release_confirmed &&
        async_launch_count == operation_count && synchronize_count == operation_count &&
        device_completion_confirmed;
    response_status = completed ? "completed" : "failed";
    FILE *file = fopen(path, "w");
    if (file == NULL) return 1;
    fprintf(file, "{\n  \"schema_version\": \"generic_loop_resident_two_dpu_contraction_slice_v1\",\n");
    fprintf(file, "  \"manifest_kind\": \"resident_two_slice_response\",\n  \"session_id\": ");
    two_dpu_json_string(file, slices[0].request.session_id == NULL ? "resident-two-dpu-unknown" : slices[0].request.session_id);
    fprintf(file, ",\n  \"status\": \"%s\",\n  \"failure_stage\": ", response_status);
    if (failure_stage == NULL) fputs("null", file); else two_dpu_json_string(file, failure_stage);
    fprintf(file, ",\n  \"error\": ");
    if (error_message == NULL) fputs("null", file); else two_dpu_json_string(file, error_message);
    fprintf(file, ",\n  \"target_requested\": \"hardware\",\n  \"target_observed\": \"%s\",\n  \"backend_id\": \"upmem_sdk_hardware_sliced_resident_two_dpu\",\n  \"backend_family\": \"upmem_sdk\",\n  \"execution_class\": \"two_dpu_sliced_resident\",\n  \"hardware_profile_version\": \"hardware_sliced_resident_two_dpu_m2_v1\",\n", completed ? "hardware" : "hardware_unverified");
    fprintf(file, "  \"quantization_mode\": ");
    two_dpu_json_string(file, quantization_mode);
    fputs(",\n", file);
    fprintf(file, "  \"cpu_fallback_used\": false,\n  \"simulator_kernel_executed\": false,\n  \"tasklets_per_dpu\": %u,\n  \"topology\": \"two_dpu_allocation\",\n  \"operation_count\": %u,\n  \"async_launch_count\": %u,\n  \"synchronize_count\": %u,\n", (unsigned)NR_TASKLETS, operation_count, async_launch_count, synchronize_count);
    fprintf(file, "  \"device_launch_mode\": \"asynchronous_dpu_set\",\n  \"host_completion_mode\": \"blocking_sync\",\n  \"completion_evidence\": \"dpu_written_completion_sentinel_read_after_each_sync\",\n  \"native_execution_sentinel_available\": true,\n  \"device_completion_confirmed\": %s,\n  \"device_completion_state\": \"%s\",\n", device_completion_confirmed ? "true" : "false", device_completion_state);
    fprintf(file, "  \"allocation\": {\"requested_dpus\":2,\"allocated_dpus\":%u,\"profile\":\"backend=hw\",\"verified\":%s},\n",
        allocated_dpus, allocated_dpus == RESIDENT_TWO_DPU_COUNT ? "true" : "false");
    fprintf(file, "  \"launch\": {\"mode\":\"asynchronous\",\"device_launch_mode\":\"asynchronous_dpu_set\",\"host_completion_mode\":\"blocking_sync\",\"operation_count\":%u,\"async_launch_count\":%u,\"synchronize_count\":%u,\"completed\":%s},\n",
        operation_count, async_launch_count, synchronize_count, completed ? "true" : "false");
    fprintf(file, "  \"observed_operation_completion_counts\": [%u,%u],\n",
        slices[0].completed_operation_count, slices[1].completed_operation_count);
    fprintf(file, "  \"completion_sentinel_read_counts\": [%u,%u],\n",
        slices[0].completion_sentinel_read_count, slices[1].completion_sentinel_read_count);
    fprintf(file, "  \"timing_scope\": \"host_observed_sdk_stage_boundaries\",\n  \"timing_is_bringup_only\": true,\n  \"timing\": {\"clock\":\"clock_monotonic\",\"status\":\"host_stage_boundaries\",\"package_parse_time_s\":%.9f,\"allocation_time_s\":%.9f,\"binary_load_time_s\":%.9f,\"initial_h2d_time_s\":%.9f,\"operation_control_h2d_time_s\":%.9f,\"launch_enqueue_time_s\":%.9f,\"sync_wait_time_s\":%.9f,\"sync_wait_is_not_pure_kernel_time\":true,\"kernel_time_s\":null,\"final_d2h_time_s\":%.9f,\"output_write_time_s\":%.9f,\"release_time_s\":%.9f,\"total_route_time_s\":%.9f},\n",
        current->package_parse_time_s, current->allocation_time_s, current->binary_load_time_s,
        current->initial_h2d_time_s, current->operation_control_h2d_time_s,
        current->launch_enqueue_time_s, current->sync_wait_time_s,
        current->final_d2h_time_s, current->output_write_time_s, current->release_time_s, elapsed_s);
    fprintf(file, "  \"initial_h2d_bytes\":%llu,\"descriptor_h2d_bytes\":%llu,\"control_h2d_bytes\":%llu,\"active_operation_h2d_bytes\":%llu,\"final_d2h_bytes\":%llu,\"actual_h2d_bytes\":%llu,\"actual_d2h_bytes\":%llu,\"actual_transfer_bytes\":%llu,\"transfer_bytes_are_application_visible\":true,\n",
        (unsigned long long)initial_h2d_bytes, (unsigned long long)descriptor_h2d_bytes,
        (unsigned long long)control_h2d_bytes,
        (unsigned long long)(control_h2d_bytes >= (uint64_t)RESIDENT_TWO_DPU_COUNT * sizeof(resident_control_t)
            ? control_h2d_bytes - (uint64_t)RESIDENT_TWO_DPU_COUNT * sizeof(resident_control_t) : 0u),
        (unsigned long long)final_d2h_bytes, (unsigned long long)actual_h2d_bytes,
        (unsigned long long)final_d2h_bytes, (unsigned long long)actual_transfer_bytes);
    fprintf(file, "  \"native_reconstruction_performed\": false,\n  \"reconstruction_contract\": \"python_sum_partials\",\n  \"slices\": [");
    for (uint32_t index = 0; index < RESIDENT_TWO_DPU_COUNT; index++) {
        const two_dpu_slice_t *slice = &slices[index];
        const resident_final_file_t *output = slice->request.final_count == 0u ? NULL : &slice->request.final_outputs[0];
        if (index != 0u) fputc(',', file);
        fprintf(file, "{\"slice_id\":%u,\"dpu_index\":%u,\"allocated\":%s,\"release_confirmed\":%s,\"manifest_path\":",
            slice->slice_id, index, allocated_dpus == RESIDENT_TWO_DPU_COUNT ? "true" : "false", release_confirmed ? "true" : "false");
        two_dpu_json_string(file, slice->manifest_path);
        fprintf(file, ",\"manifest_fnv1a64\":\"%016llx\",\"package_transferred\":%s,\"input_count\":%zu,\"input_transfer_bytes\":%llu,\"inputs_transferred\":%s,\"partial_output_path\":",
            (unsigned long long)slice->manifest_hash, slice->package_transferred ? "true" : "false", slice->request.input_count,
            (unsigned long long)slice->input_bytes, slice->inputs_transferred ? "true" : "false");
        two_dpu_json_string(file, output == NULL ? "" : output->path);
        fprintf(file, ",\"operation_count\":%u,\"completed_operation_count\":%u,\"observed_operation_completion_count\":%u,\"operation_completion_confirmed\":%s,\"completion_sentinel_read_count\":%u,\"dpu_completion_sentinel\":{\"magic\":%u,\"version\":%u,\"active_operation_index\":%u,\"completion_status\":%u,\"completed_operation_count\":%u,\"output_elements_processed\":%u,\"output_checksum_fnv1a64\":\"%016llx\",\"read\":%s,\"verified\":%s},\"partial_output_elements\":%u,\"partial_output_bytes\":%llu,\"partial_output_raw_bytes\":%llu,\"partial_output_transfer_bytes\":%llu,\"partial_output_fnv1a64\":\"%016llx\",\"partial_output_read\":%s,\"partial_output_written\":%s,\"completion_confirmed\":%s}",
            operation_count, slice->completed_operation_count, slice->completed_operation_count,
            slice->completion_sentinel_verified ? "true" : "false",
            slice->completion_sentinel_read_count,
            slice->completion_sentinel.magic, slice->completion_sentinel.version,
            slice->completion_sentinel.active_operation_index, slice->completion_sentinel.completion_status,
            slice->completion_sentinel.completed_operation_count,
            slice->completion_sentinel.output_elements_processed,
            (unsigned long long)slice->completion_sentinel.output_checksum_fnv1a64,
            slice->completion_sentinel_read ? "true" : "false",
            slice->completion_sentinel_verified ? "true" : "false",
            output == NULL ? 0u : output->elements, (unsigned long long)slice->partial_output_bytes,
            (unsigned long long)slice->partial_output_bytes, (unsigned long long)slice->partial_output_transfer_bytes,
            (unsigned long long)slice->partial_output_fnv1a64,
            slice->partial_output_read ? "true" : "false", slice->partial_output_written ? "true" : "false",
            slice->completion_confirmed ? "true" : "false");
    }
    fprintf(file, "],\n  \"release\": {\"attempted\":%s,\"confirmed\":%s,\"device_completion_confirmed\":%s},\n  \"hardware_execution\":%s,\n  \"hardware_kernel_executed\":%s,\n  \"hardware_functionality_evidence\":%s,\n  \"elapsed_s\":%.9f\n}\n",
        release_attempted ? "true" : "false", release_confirmed ? "true" : "false",
        device_completion_confirmed ? "true" : "false",
        completed ? "true" : "false", device_completion_confirmed ? "true" : "false",
        completed ? "true" : "false", elapsed_s);
    {
        const int failed = ferror(file) != 0 || fclose(file) != 0;
        return failed ? 1 : 0;
    }
}

int main(int argc, char **argv) {
    two_dpu_slice_t slices[RESIDENT_TWO_DPU_COUNT];
    two_dpu_timing_t timing = {0};
    struct dpu_set_t set;
    dpu_error_t error = DPU_OK;
    const char *failure_stage = NULL;
    const char *error_message = NULL;
    const char *response_path = NULL;
    uint32_t allocated_dpus = 0u;
    uint32_t operation_count = 0u;
    uint32_t async_launch_count = 0u;
    uint32_t synchronize_count = 0u;
    uint64_t initial_h2d_bytes = 0u;
    uint64_t descriptor_h2d_bytes = 0u;
    uint64_t control_h2d_bytes = 0u;
    uint64_t final_d2h_bytes = 0u;
    int set_allocated = 0;
    int release_attempted = 0;
    int release_confirmed = 0;
    int launch_in_flight = 0;
    int rc = 1;
    const double started = two_dpu_now_s();

    memset(slices, 0, sizeof(slices));
    if (argc == 4 && strcmp(argv[1], "--validate-slice-packages") == 0) {
        slices[0].slice_id = 0u;
        slices[1].slice_id = 1u;
        slices[0].manifest_path = argv[2];
        slices[1].manifest_path = argv[3];
        if (two_dpu_load_slice(&slices[0], argv[2], &failure_stage) == 0 && two_dpu_load_slice(&slices[1], argv[3], &failure_stage) == 0) {
            (void)two_dpu_validate_slice_pair(slices, &failure_stage);
        }
        rc = two_dpu_write_validation_result(slices, failure_stage);
        goto cleanup;
    }
    if (argc != 7 || strcmp(argv[1], "--slice-package-0") != 0 || strcmp(argv[3], "--slice-package-1") != 0 ||
        strcmp(argv[5], "--resident-response") != 0) {
        fprintf(stderr, "usage: %s --slice-package-0 slice0.json --slice-package-1 slice1.json --resident-response response.json\n", argv[0]);
        fprintf(stderr, "   or: %s --validate-slice-packages slice0.json slice1.json\n", argv[0]);
        return 2;
    }
    response_path = argv[6];
    slices[0].slice_id = 0u;
    slices[1].slice_id = 1u;
    {
        const double stage_started = two_dpu_now_s();
        const int first_failed = two_dpu_load_slice(&slices[0], argv[2], &failure_stage);
        const int second_failed = first_failed == 0 ? two_dpu_load_slice(&slices[1], argv[4], &failure_stage) : 1;
        timing.package_parse_time_s = two_dpu_now_s() - stage_started;
        if (first_failed != 0 || second_failed != 0) {
            error_message = slices[0].parse_error != NULL ? slices[0].parse_error : slices[1].parse_error;
            goto write_response;
        }
    }
    operation_count = slices[0].request.header.operation_count;
    if (two_dpu_validate_slice_pair(slices, &failure_stage) != 0) {
        error_message = "slice packages do not describe the same operation stream";
        goto write_response;
    }
    if (getenv("UPMEM_ALLOW_PHYSICAL_HARDWARE") == NULL || strcmp(getenv("UPMEM_ALLOW_PHYSICAL_HARDWARE"), "1") != 0) {
        failure_stage = "hardware_opt_in_missing";
        error_message = "UPMEM_ALLOW_PHYSICAL_HARDWARE=1 is required";
        goto write_response;
    }
    if (getenv("DPU_BACKEND") != NULL || NR_TASKLETS != 1) {
        failure_stage = "hardware_profile_violation";
        error_message = "DPU_BACKEND must be unset and NR_TASKLETS must equal one";
        goto write_response;
    }
    if (two_dpu_prepare_slice_inputs(&slices[0], &failure_stage) != 0 || two_dpu_prepare_slice_inputs(&slices[1], &failure_stage) != 0) goto write_response;

    {
        const double stage_started = two_dpu_now_s();
        error = dpu_alloc(RESIDENT_TWO_DPU_COUNT, RESIDENT_TWO_DPU_ALLOCATION_PROFILE, &set);
        if (error == DPU_OK) set_allocated = 1;
        if (error == DPU_OK) error = dpu_get_nr_dpus(set, &allocated_dpus);
        timing.allocation_time_s = two_dpu_now_s() - stage_started;
        if (error != DPU_OK || allocated_dpus != RESIDENT_TWO_DPU_COUNT) {
            if (error != DPU_OK) two_dpu_report_sdk_error("two-DPU allocation", error);
            failure_stage = error == DPU_ERR_INVALID_PROFILE ? "hardware_profile_violation" : "hardware_allocation_failed";
            goto release_and_write;
        }
    }
    {
        const double stage_started = two_dpu_now_s();
        error = dpu_load(set, slices[0].request.dpu_binary_path, NULL);
        timing.binary_load_time_s = two_dpu_now_s() - stage_started;
        if (error != DPU_OK) {
            two_dpu_report_sdk_error("two-DPU dpu_load", error);
            failure_stage = "binary_load_failed";
            goto release_and_write;
        }
    }
    {
        const double stage_started = two_dpu_now_s();
        struct dpu_set_t dpu;
        uint32_t index;
        DPU_FOREACH(set, dpu, index) {
            if (two_dpu_transfer_slice_package(dpu, &slices[index], &error) != 0) break;
        }
        timing.initial_h2d_time_s = two_dpu_now_s() - stage_started;
        if (error != DPU_OK) {
            two_dpu_report_sdk_error("two-DPU per-slice package transfer", error);
            failure_stage = "slice_package_transfer_failed";
            goto release_and_write;
        }
        initial_h2d_bytes = slices[0].input_bytes + slices[1].input_bytes;
        descriptor_h2d_bytes = (uint64_t)(slices[0].request.header.slot_bytes + slices[0].request.header.operation_bytes) * RESIDENT_TWO_DPU_COUNT;
        control_h2d_bytes = (uint64_t)sizeof(resident_control_t) * RESIDENT_TWO_DPU_COUNT;
    }
    for (uint32_t operation_index = 0; operation_index < operation_count; operation_index++) {
        {
            const double stage_started = two_dpu_now_s();
            struct dpu_set_t dpu;
            uint32_t index;
            DPU_FOREACH(set, dpu, index) {
                if (two_dpu_set_active_operation(dpu, &slices[index], operation_index, &error) != 0) break;
                control_h2d_bytes += sizeof(uint64_t);
            }
            timing.operation_control_h2d_time_s += two_dpu_now_s() - stage_started;
            if (error != DPU_OK) {
                two_dpu_report_sdk_error("two-DPU active operation transfer", error);
                failure_stage = "operation_control_transfer_failed";
                goto release_and_write;
            }
        }
        {
            const double stage_started = two_dpu_now_s();
            error = dpu_launch(set, DPU_ASYNCHRONOUS);
            timing.launch_enqueue_time_s += two_dpu_now_s() - stage_started;
            if (error != DPU_OK) {
                two_dpu_report_sdk_error("two-DPU asynchronous dpu_launch", error);
                failure_stage = "kernel_launch_failed";
                goto release_and_write;
            }
            async_launch_count++;
            launch_in_flight = 1;
        }
        {
            const double stage_started = two_dpu_now_s();
            error = dpu_sync(set);
            timing.sync_wait_time_s += two_dpu_now_s() - stage_started;
            if (error != DPU_OK) {
                two_dpu_report_sdk_error("two-DPU blocking dpu_sync", error);
                failure_stage = "kernel_synchronize_failed";
                goto release_and_write;
            }
            synchronize_count++;
            launch_in_flight = 0;
            {
                struct dpu_set_t dpu;
                uint32_t index;
                DPU_FOREACH(set, dpu, index) {
                    if (two_dpu_read_completion_sentinel(dpu, &slices[index], operation_index, &error) != 0) break;
                }
                if (error != DPU_OK) {
                    two_dpu_report_sdk_error("two-DPU completion sentinel read", error);
                    failure_stage = "kernel_completion_sentinel_failed";
                    goto release_and_write;
                }
                if (!slices[0].completion_sentinel_verified || !slices[1].completion_sentinel_verified) {
                    failure_stage = "kernel_completion_sentinel_failed";
                    error_message = "DPU completion sentinel did not verify";
                    goto release_and_write;
                }
            }
        }
    }
    {
        const double stage_started = two_dpu_now_s();
        struct dpu_set_t dpu;
        uint32_t index;
        DPU_FOREACH(set, dpu, index) {
            if (two_dpu_read_slice_partial(dpu, &slices[index], &error) != 0) break;
            final_d2h_bytes += slices[index].partial_output_transfer_bytes;
        }
        timing.final_d2h_time_s = two_dpu_now_s() - stage_started;
        if (error != DPU_OK) two_dpu_report_sdk_error("two-DPU full partial-output read", error);
        if (error != DPU_OK || !slices[0].partial_output_read || !slices[1].partial_output_read) {
            failure_stage = "partial_output_read_failed";
            goto release_and_write;
        }
    }
    {
        const double stage_started = two_dpu_now_s();
        if (two_dpu_write_slice_partial(&slices[0]) != 0 || two_dpu_write_slice_partial(&slices[1]) != 0) {
            timing.output_write_time_s = two_dpu_now_s() - stage_started;
            failure_stage = "partial_output_write_failed";
            goto release_and_write;
        }
        timing.output_write_time_s = two_dpu_now_s() - stage_started;
        slices[0].completion_confirmed = slices[0].completion_sentinel_verified;
        slices[1].completion_confirmed = slices[1].completion_sentinel_verified;
    }
    rc = 0;

release_and_write:
    if (set_allocated) {
        const double stage_started = two_dpu_now_s();
        release_attempted = 1;
        error = dpu_free(set);
        timing.release_time_s = two_dpu_now_s() - stage_started;
        if (error == DPU_OK) release_confirmed = 1;
        else {
            two_dpu_report_sdk_error("two-DPU dpu_free", error);
            if (failure_stage == NULL) failure_stage = "hardware_release_failed";
        }
    }
    if (failure_stage != NULL) rc = 1;
write_response:
    if (two_dpu_write_response(response_path, slices, failure_stage == NULL ? "completed" : "failed", failure_stage, error_message,
        allocated_dpus, release_attempted, release_confirmed, operation_count, async_launch_count, synchronize_count,
        initial_h2d_bytes, descriptor_h2d_bytes, control_h2d_bytes, final_d2h_bytes, &timing,
        two_dpu_now_s() - started, launch_in_flight) != 0) rc = 1;
cleanup:
    two_dpu_free_slice(&slices[0]);
    two_dpu_free_slice(&slices[1]);
    return rc;
}
