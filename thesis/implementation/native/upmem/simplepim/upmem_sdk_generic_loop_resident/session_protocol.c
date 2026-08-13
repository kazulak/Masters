#include "session_protocol.h"

#include <ctype.h>
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static char *resident_copy(const char *value) {
    const size_t length = strlen(value);
    char *copy = (char *)malloc(length + 1u);
    if (copy != NULL) memcpy(copy, value, length + 1u);
    return copy;
}

static void resident_error(char **message, const char *value) {
    if (message != NULL && *message == NULL) *message = resident_copy(value);
}

static int resident_supported_tasklets(uint64_t value) {
    return value == 1u || value == 2u || value == 4u || value == 8u || value == 16u;
}

#if RESIDENT_OPERATION_ABI_VERSION == RESIDENT_OPERATION_ABI_V2
static const char *resident_binary_basename(const char *path) {
    const char *slash = path == NULL ? NULL : strrchr(path, '/');
    return slash == NULL ? path : slash + 1;
}

static int resident_binary_matches_abi(const char *path) {
    const char *basename = resident_binary_basename(path);
    return basename != NULL && strcmp(basename, RESIDENT_DPU_BINARY_NAME) == 0;
}
#endif

static int resident_binary_matches_v3_tasklets(const char *path, uint64_t tasklets) {
    char expected[32];
    const char *basename;
    if (tasklets < 1u || tasklets > 24u ||
        snprintf(expected, sizeof(expected), "dpu_resident_v3_t%llu",
            (unsigned long long)tasklets) < 0) return 0;
    basename = path == NULL ? NULL : strrchr(path, '/');
    basename = basename == NULL ? path : basename + 1;
    return basename != NULL && strcmp(basename, expected) == 0;
}

static int resident_read_file(const char *path, unsigned char **payload, size_t *length) {
    FILE *file = fopen(path, "rb");
    long size;
    if (file == NULL || fseek(file, 0, SEEK_END) != 0) {
        if (file != NULL) fclose(file);
        return 1;
    }
    size = ftell(file);
    if (size < 0 || (unsigned long)size > 8u * 1024u * 1024u || fseek(file, 0, SEEK_SET) != 0) {
        fclose(file);
        return 1;
    }
    *payload = (unsigned char *)malloc((size_t)size + 1u);
    if (*payload == NULL || fread(*payload, 1, (size_t)size, file) != (size_t)size) {
        free(*payload);
        *payload = NULL;
        fclose(file);
        return 1;
    }
    fclose(file);
    (*payload)[size] = 0;
    *length = (size_t)size;
    return 0;
}

static const char *resident_find_key(const char *object, const char *key) {
    char needle[128];
    const size_t key_length = strlen(key);
    if (key_length + 3u > sizeof(needle)) return NULL;
    snprintf(needle, sizeof(needle), "\"%s\"", key);
    return strstr(object, needle);
}

static int resident_string_field(const char *object, const char *key, char **value) {
    const char *cursor = resident_find_key(object, key);
    size_t capacity = 32u;
    size_t length = 0u;
    char *result;
    if (cursor == NULL) return 1;
    cursor = strchr(cursor, ':');
    if (cursor == NULL) return 1;
    cursor++;
    while (*cursor != '\0' && isspace((unsigned char)*cursor)) cursor++;
    if (*cursor++ != '"') return 1;
    result = (char *)malloc(capacity);
    if (result == NULL) return 1;
    while (*cursor != '\0') {
        unsigned char character = (unsigned char)*cursor++;
        if (character == '"') {
            result[length] = '\0';
            *value = result;
            return 0;
        }
        if (character == '\\') {
            character = (unsigned char)*cursor++;
            if (character != '"' && character != '\\' && character != '/' && character != 'n' && character != 't') {
                free(result);
                return 1;
            }
            if (character == 'n') character = '\n';
            if (character == 't') character = '\t';
        }
        if (character < 0x20u) {
            free(result);
            return 1;
        }
        if (length + 1u >= capacity) {
            char *grown;
            if (capacity > SIZE_MAX / 2u) {
                free(result);
                return 1;
            }
            capacity *= 2u;
            grown = (char *)realloc(result, capacity);
            if (grown == NULL) {
                free(result);
                return 1;
            }
            result = grown;
        }
        result[length++] = (char)character;
    }
    free(result);
    return 1;
}

static int resident_uint_field(const char *object, const char *key, uint64_t *value) {
    const char *cursor = resident_find_key(object, key);
    char *end = NULL;
    unsigned long long parsed;
    if (cursor == NULL) return 1;
    cursor = strchr(cursor, ':');
    if (cursor == NULL) return 1;
    cursor++;
    while (*cursor != '\0' && isspace((unsigned char)*cursor)) cursor++;
    if (!isdigit((unsigned char)*cursor)) return 1;
    errno = 0;
    parsed = strtoull(cursor, &end, 10);
    if (errno != 0 || end == cursor || parsed > UINT64_MAX) return 1;
    while (*end != '\0' && isspace((unsigned char)*end)) end++;
    if (*end != ',' && *end != '}' && *end != ']') return 1;
    *value = (uint64_t)parsed;
    return 0;
}

static const char *resident_skip_space(const char *cursor, const char *end) {
    while (cursor < end && isspace((unsigned char)*cursor)) cursor++;
    return cursor;
}

static int resident_matching_end(const char *start, const char *end, char opening, char closing, const char **match) {
    int depth = 0;
    int in_string = 0;
    int escaped = 0;
    for (const char *cursor = start; cursor < end; cursor++) {
        const unsigned char character = (unsigned char)*cursor;
        if (in_string) {
            if (escaped) escaped = 0;
            else if (character == '\\') escaped = 1;
            else if (character == '"') in_string = 0;
            continue;
        }
        if (character == '"') {
            in_string = 1;
            continue;
        }
        if (character == (unsigned char)opening) depth++;
        else if (character == (unsigned char)closing) {
            depth--;
            if (depth == 0) {
                *match = cursor;
                return 0;
            }
            if (depth < 0) return 1;
        }
    }
    return 1;
}

static int resident_manifest_array(const char *contents, const char *key, const char **begin, const char **end) {
    const char *cursor = resident_find_key(contents, key);
    const char *array_end = NULL;
    if (cursor == NULL) return 1;
    cursor = strchr(cursor, ':');
    if (cursor == NULL) return 1;
    cursor++;
    while (*cursor != '\0' && isspace((unsigned char)*cursor)) cursor++;
    if (*cursor != '[' || resident_matching_end(cursor, contents + strlen(contents), '[', ']', &array_end) != 0) return 1;
    *begin = cursor + 1;
    *end = array_end;
    return 0;
}

static int resident_next_object(const char **cursor, const char *end, const char **object, const char **object_end) {
    const char *start = resident_skip_space(*cursor, end);
    if (start >= end || *start != '{' || resident_matching_end(start, end, '{', '}', object_end) != 0) return 1;
    *object = start;
    *cursor = *object_end + 1;
    return 0;
}

static char *resident_copy_object(const char *object, const char *object_end) {
    const size_t length = (size_t)(object_end - object) + 1u;
    char *copy = (char *)malloc(length + 1u);
    if (copy == NULL) return NULL;
    memcpy(copy, object, length);
    copy[length] = '\0';
    return copy;
}

static int resident_safe_relative(const char *path) {
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

static char *resident_base(const char *path) {
    const char *slash = strrchr(path, '/');
    const size_t length = slash == NULL ? 1u : (size_t)(slash - path);
    char *base = (char *)malloc(length + 1u);
    if (base == NULL) return NULL;
    if (slash == NULL) memcpy(base, ".", 2u);
    else {
        memcpy(base, path, length);
        base[length] = '\0';
    }
    return base;
}

static char *resident_resolve(const char *base, const char *relative) {
    const size_t base_length = strlen(base);
    const size_t relative_length = strlen(relative);
    char *resolved;
    if (resident_safe_relative(relative) != 0 || relative_length > RESIDENT_MAX_PATH) return NULL;
    if (base_length > SIZE_MAX - relative_length - 2u) return NULL;
    resolved = (char *)malloc(base_length + relative_length + 2u);
    if (resolved == NULL) return NULL;
    memcpy(resolved, base, base_length);
    resolved[base_length] = '/';
    memcpy(resolved + base_length + 1u, relative, relative_length + 1u);
    return resolved;
}

static uint64_t resident_align8(uint64_t value) {
    return (value + 7u) & ~((uint64_t)7u);
}

static int resident_alignment_overflow(uint64_t value) {
    return value > UINT64_MAX - 7u;
}

static int resident_add_overflow(uint64_t left, uint64_t right, uint64_t *result) {
    if (right > UINT64_MAX - left) return 1;
    *result = left + right;
    return 0;
}

static int resident_mul_overflow(uint64_t left, uint64_t right, uint64_t *result) {
    if (left != 0u && right > UINT64_MAX / left) return 1;
    *result = left * right;
    return 0;
}

static int resident_transfer_size(
    uint32_t elements,
    uint32_t element_bytes,
    size_t *raw,
    size_t *transfer
) {
    uint64_t bytes;
    if ((element_bytes != 1u && element_bytes != sizeof(float)) ||
        resident_mul_overflow(elements, element_bytes, &bytes) || bytes > SIZE_MAX) return 1;
    *raw = (size_t)bytes;
    if (bytes > UINT64_MAX - 7u || resident_align8(bytes) > SIZE_MAX) return 1;
    *transfer = (size_t)resident_align8(bytes);
    return 0;
}

static uint32_t resident_slot_element_bytes(const resident_slot_descriptor_t *slot) {
#if RESIDENT_PACKAGE_ABI_VERSION == RESIDENT_PACKAGE_VERSION_V3
    return slot->element_bytes;
#else
    (void)slot;
    return (uint32_t)sizeof(float);
#endif
}

static uint32_t resident_slot_storage_kind(const resident_slot_descriptor_t *slot) {
#if RESIDENT_PACKAGE_ABI_VERSION == RESIDENT_PACKAGE_VERSION_V3
    return slot->storage_kind;
#else
    (void)slot;
    return RESIDENT_STORAGE_FLOAT32;
#endif
}

static const char *resident_storage_dtype(uint32_t storage_kind) {
    if (storage_kind == RESIDENT_STORAGE_FLOAT32) return "float32";
    if (storage_kind == RESIDENT_STORAGE_PACKED_INT8) return "int8";
    if (storage_kind == RESIDENT_STORAGE_INT32) return "int32";
    return NULL;
}

static int resident_ascii_identifier(const char *value) {
    if (value == NULL || value[0] == '\0') return 1;
    for (const unsigned char *cursor = (const unsigned char *)value; *cursor != '\0'; cursor++) {
        if (*cursor < 0x20u || *cursor >= 0x80u) return 1;
    }
    return 0;
}

static int resident_validate_row_major(
    const uint32_t *shape,
    const uint32_t *strides,
    uint32_t rank,
    uint32_t expected_elements
) {
    uint64_t product = 1u;
    uint64_t expected_stride = 1u;
    for (uint32_t reverse_axis = 0; reverse_axis < rank; reverse_axis++) {
        const uint32_t axis = rank - reverse_axis - 1u;
        if (shape[axis] == 0u || strides[axis] != expected_stride) return 1;
        product *= shape[axis];
        expected_stride *= shape[axis];
        if (product > UINT32_MAX || expected_stride > UINT32_MAX) return 1;
    }
    return product != expected_elements;
}

static int resident_validate_index_maps(const upmem_generic_args_t *args) {
    uint8_t left_used[UPMEM_GENERIC_MAX_RANK] = {0};
    uint8_t right_used[UPMEM_GENERIC_MAX_RANK] = {0};
    for (uint32_t output_axis = 0; output_axis < args->output_rank; output_axis++) {
        const int32_t left_axis = args->output_to_left_axes[output_axis];
        const int32_t right_axis = args->output_to_right_axes[output_axis];
        if (left_axis < -1 || right_axis < -1 ||
            left_axis >= (int32_t)args->left_rank || right_axis >= (int32_t)args->right_rank ||
            (left_axis < 0 && right_axis < 0)) return 1;
        if (left_axis >= 0) {
            if (left_used[left_axis] || args->left_shape[left_axis] != args->output_shape[output_axis]) return 1;
            left_used[left_axis] = 1;
        }
        if (right_axis >= 0) {
            if (right_used[right_axis] || args->right_shape[right_axis] != args->output_shape[output_axis]) return 1;
            right_used[right_axis] = 1;
        }
    }
    for (uint32_t contracted_axis = 0; contracted_axis < args->contracted_rank; contracted_axis++) {
        const int32_t left_axis = args->contracted_to_left_axes[contracted_axis];
        const int32_t right_axis = args->contracted_to_right_axes[contracted_axis];
        if (left_axis < 0 || right_axis < 0 ||
            left_axis >= (int32_t)args->left_rank || right_axis >= (int32_t)args->right_rank ||
            left_used[left_axis] || right_used[right_axis] || args->contracted_dims[contracted_axis] == 0u ||
            args->left_shape[left_axis] != args->contracted_dims[contracted_axis] ||
            args->right_shape[right_axis] != args->contracted_dims[contracted_axis]) return 1;
        left_used[left_axis] = 1;
        right_used[right_axis] = 1;
    }
    for (uint32_t axis = 0; axis < args->left_rank; axis++) if (!left_used[axis]) return 1;
    for (uint32_t axis = 0; axis < args->right_rank; axis++) if (!right_used[axis]) return 1;
    return 0;
}

static int resident_validate_args(const resident_operation_t *operation) {
    const upmem_generic_args_t *args = &operation->args;
    uint64_t contracted_product = 1u;
    if (args->left_rank > UPMEM_GENERIC_MAX_RANK || args->right_rank > UPMEM_GENERIC_MAX_RANK ||
        args->output_rank > UPMEM_GENERIC_MAX_RANK || args->contracted_rank > UPMEM_GENERIC_MAX_RANK ||
        args->left_elems == 0u || args->right_elems == 0u || args->output_elems == 0u || args->contracted_elems == 0u ||
        args->left_elems > UPMEM_GENERIC_MAX_ELEMS || args->right_elems > UPMEM_GENERIC_MAX_ELEMS ||
        args->output_elems > UPMEM_GENERIC_MAX_ELEMS || args->contracted_elems > UPMEM_GENERIC_MAX_ELEMS ||
        (args->operand_mode != UPMEM_GENERIC_MODE_FLOAT32_NO_QUANT &&
         args->operand_mode != UPMEM_GENERIC_MODE_INT8_SCALED &&
         args->operand_mode != UPMEM_GENERIC_MODE_HOST_PACKED_INT8) ||
        resident_validate_row_major(args->left_shape, args->left_strides, args->left_rank, args->left_elems) != 0 ||
        resident_validate_row_major(args->right_shape, args->right_strides, args->right_rank, args->right_elems) != 0 ||
        resident_validate_index_maps(args) != 0 ||
        resident_validate_row_major(args->output_shape, args->output_strides, args->output_rank, args->output_elems) != 0) return 1;
    for (uint32_t axis = 0; axis < args->contracted_rank; axis++) {
        if (args->contracted_dims[axis] == 0u || contracted_product > UINT32_MAX / args->contracted_dims[axis]) return 1;
        contracted_product *= args->contracted_dims[axis];
    }
    if (contracted_product != args->contracted_elems || args->output_elems != operation->output_elements) return 1;
#if RESIDENT_OPERATION_ABI_VERSION >= RESIDENT_OPERATION_ABI_V2
    if (args->dpu_slice_offset != 0u || args->dpu_slice_elements != args->output_elems ||
        args->contracted_offset != 0u || args->contracted_elements_slice != args->contracted_elems) return 1;
#endif
    return 0;
}

static int resident_validate_package(
    const unsigned char *payload,
    size_t length,
    resident_request_t *request,
    char **error_message
) {
    resident_package_header_t header;
    uint64_t section_end;
    int packed_package;
    if (sizeof(resident_package_header_t) != 96u || length < sizeof(header)) {
        resident_error(error_message, "resident_package_truncated_header");
        return 1;
    }
    memcpy(&header, payload, sizeof(header));
    if (memcmp(header.magic, RESIDENT_PACKAGE_MAGIC, 8u) != 0 || header.version != RESIDENT_PACKAGE_VERSION) {
        resident_error(error_message, "resident_package_abi_magic_or_version_mismatch");
        return 1;
    }
    if (header.endian != RESIDENT_PACKAGE_ENDIAN) {
        resident_error(error_message, "resident_package_bad_endian");
        return 1;
    }
    packed_package = (header.flags & RESIDENT_PACKAGE_FLAG_PACKED_INT8) != 0u;
    if (header.header_bytes != sizeof(header) || (header.header_bytes & 7u) != 0u || header.file_bytes != length) {
        resident_error(error_message, "resident_package_file_length_or_header_mismatch");
        return 1;
    }
    if (header.file_bytes > SIZE_MAX || header.slot_offset > SIZE_MAX || header.slot_bytes > SIZE_MAX ||
        header.operation_offset > SIZE_MAX || header.operation_bytes > SIZE_MAX ||
        resident_alignment_overflow(header.slot_offset) || resident_alignment_overflow(header.slot_bytes) ||
        resident_alignment_overflow(header.operation_offset) || resident_alignment_overflow(header.operation_bytes)) {
        resident_error(error_message, "resident_package_file_length_or_offset_overflow");
        return 1;
    }
    if ((header.slot_offset & 7u) != 0u || (header.slot_bytes & 7u) != 0u ||
        (header.operation_offset & 7u) != 0u || (header.operation_bytes & 7u) != 0u) {
        resident_error(error_message, "resident_package_unaligned_section");
        return 1;
    }
#if RESIDENT_PACKAGE_ABI_VERSION == RESIDENT_PACKAGE_VERSION_V3
    if ((header.flags & ~RESIDENT_PACKAGE_FLAG_PACKED_INT8) != 0u ||
#else
    if (header.flags != 0u ||
#endif
        header.reserved != 0u || header.slot_count == 0u ||
        header.slot_count > RESIDENT_MAX_SLOT_DESCRIPTORS || header.operation_count == 0u || header.operation_count > RESIDENT_MAX_COMPONENT_OPS ||
        header.graph_request_count != 1u || header.pool_bytes != RESIDENT_MRAM_POOL_BYTES ||
        header.max_rank != UPMEM_GENERIC_MAX_RANK || header.initial_slot_count == 0u ||
        header.initial_slot_count > header.slot_count ||
        header.final_output_count == 0u || header.final_output_count > 2u) {
        resident_error(error_message, "resident_package_profile_cap_or_request_mismatch");
        return 1;
    }
    if (resident_mul_overflow(header.slot_count, sizeof(resident_slot_descriptor_t), &section_end) ||
        section_end != header.slot_bytes ||
        resident_mul_overflow(header.operation_count, RESIDENT_OPERATION_BYTES, &section_end) ||
        section_end != header.operation_bytes) {
        resident_error(error_message, "resident_package_descriptor_length_overflow");
        return 1;
    }
    if (header.slot_offset < header.header_bytes ||
        resident_add_overflow(header.slot_offset, header.slot_bytes, &section_end) || section_end > header.file_bytes ||
        header.operation_offset < section_end ||
        resident_add_overflow(header.operation_offset, header.operation_bytes, &section_end) || section_end != header.file_bytes) {
        resident_error(error_message, "resident_package_section_overflow_or_overlap");
        return 1;
    }
    request->header = header;
    request->slots = (resident_slot_descriptor_t *)calloc(header.slot_count, sizeof(*request->slots));
    request->operations = (resident_operation_t *)calloc(header.operation_count, sizeof(*request->operations));
    request->slot_flags = (uint32_t *)calloc(header.slot_count, sizeof(*request->slot_flags));
    if ((header.slot_count != 0u && request->slots == NULL) ||
        (header.operation_count != 0u && request->operations == NULL) ||
        (header.slot_count != 0u && request->slot_flags == NULL)) {
        resident_error(error_message, "resident_package_descriptor_allocation_failed");
        return 1;
    }
    memcpy(request->slots, payload + header.slot_offset, (size_t)header.slot_bytes);
    memcpy(request->operations, payload + header.operation_offset, (size_t)header.operation_bytes);
    uint8_t slot_ready[RESIDENT_MAX_SLOT_DESCRIPTORS] = {0};
    for (uint32_t index = 0; index < header.slot_count; index++) {
        resident_slot_descriptor_t *slot = &request->slots[index];
        uint64_t bytes;
        const uint32_t element_bytes = resident_slot_element_bytes(slot);
        const uint32_t storage_kind = resident_slot_storage_kind(slot);
        const uint32_t flags = slot->slot_id & ~(uint32_t)RESIDENT_SLOT_ID_MASK;
        if ((slot->slot_id & RESIDENT_SLOT_ID_MASK) != index ||
            (flags & ~(RESIDENT_SLOT_INITIAL_FLAG | RESIDENT_SLOT_FINAL_FLAG)) != 0u ||
            (slot->offset_bytes & 7u) != 0u || slot->capacity_elements == 0u ||
            slot->element_count == 0u || slot->element_count > slot->capacity_elements ||
            resident_storage_dtype(storage_kind) == NULL ||
            ((storage_kind == RESIDENT_STORAGE_PACKED_INT8) != (element_bytes == 1u)) ||
            ((storage_kind == RESIDENT_STORAGE_FLOAT32 || storage_kind == RESIDENT_STORAGE_INT32) &&
                element_bytes != sizeof(float)) ||
            resident_mul_overflow(slot->capacity_elements, element_bytes, &bytes) ||
            resident_add_overflow(slot->offset_bytes, resident_align8(bytes), &section_end) || section_end > header.pool_bytes) {
            resident_error(error_message, "resident_package_slot_descriptor_invalid_or_overflow");
            return 1;
        }
#if RESIDENT_PACKAGE_ABI_VERSION == RESIDENT_PACKAGE_VERSION_V3
        {
            uint64_t logical_bytes;
            if (resident_mul_overflow(slot->element_count, element_bytes, &logical_bytes) ||
                logical_bytes != slot->logical_bytes ||
                resident_align8(bytes) != slot->transfer_bytes ||
                (slot->transfer_bytes & 7u) != 0u ||
                slot->logical_bytes > slot->transfer_bytes) {
                resident_error(error_message, "resident_package_typed_slot_bytes_invalid");
                return 1;
            }
        }
#endif
        request->slot_flags[index] = flags;
        slot_ready[index] = (flags & RESIDENT_SLOT_INITIAL_FLAG) != 0u;
        for (uint32_t other = 0; other < index; other++) {
            uint64_t this_end = (uint64_t)slot->offset_bytes + resident_align8(bytes);
            uint64_t other_bytes;
            if (resident_mul_overflow(
                    request->slots[other].capacity_elements,
                    resident_slot_element_bytes(&request->slots[other]),
                    &other_bytes)) {
                resident_error(error_message, "resident_package_slot_capacity_overflow");
                return 1;
            }
            const uint64_t other_start = request->slots[other].offset_bytes;
            const uint64_t other_end = other_start + resident_align8(other_bytes);
            if ((uint64_t)slot->offset_bytes < other_end && other_start < this_end) {
                resident_error(error_message, "resident_package_slot_overlap");
                return 1;
            }
        }
    }
    {
        uint32_t initial_flags = 0u;
        uint32_t final_flags = 0u;
        for (uint32_t index = 0; index < header.slot_count; index++) {
            if ((request->slot_flags[index] & RESIDENT_SLOT_INITIAL_FLAG) != 0u) initial_flags++;
            if ((request->slot_flags[index] & RESIDENT_SLOT_FINAL_FLAG) != 0u) final_flags++;
        }
        if (initial_flags != header.initial_slot_count || final_flags != header.final_output_count) {
            resident_error(error_message, "resident_package_slot_flag_count_mismatch");
            return 1;
        }
    }
    for (uint32_t index = 0; index < header.operation_count; index++) {
        const resident_operation_t *operation = &request->operations[index];
        const uint32_t refs[6] = {operation->slot_a, operation->slot_b, operation->slot_c,
            operation->slot_d, operation->slot_out_real, operation->slot_out_imag};
        if (operation->kind != RESIDENT_OPERATION_CONTRACT && operation->kind != RESIDENT_OPERATION_COMPLEX_COMBINE) {
            resident_error(error_message, "resident_package_operation_kind_invalid");
            return 1;
        }
        if (operation->mode >
#if RESIDENT_PACKAGE_ABI_VERSION == RESIDENT_PACKAGE_VERSION_V3
                RESIDENT_MODE_HOST_PACKED_INT8 ||
#else
                RESIDENT_MODE_PER_TASK_REQUANTIZE ||
#endif
            operation->output_elements == 0u || operation->output_elements > UPMEM_GENERIC_MAX_ELEMS) {
            resident_error(error_message, "resident_package_operation_mode_or_output_invalid");
            return 1;
        }
#if RESIDENT_OPERATION_ABI_VERSION >= RESIDENT_OPERATION_ABI_V2
        if (operation->args.dpu_slice_offset != 0u || operation->args.dpu_slice_elements != operation->output_elements) {
            resident_error(error_message, "resident_package_operation_v2_slice_metadata_invalid");
            return 1;
        }
#endif
        for (size_t ref = 0; ref < 6u; ref++) {
            if (refs[ref] != RESIDENT_INVALID_SLOT && refs[ref] >= header.slot_count) {
                resident_error(error_message, "resident_package_operation_slot_reference_invalid");
                return 1;
            }
        }
        if (operation->kind == RESIDENT_OPERATION_CONTRACT) {
            if (resident_validate_args(operation) != 0 ||
                operation->slot_a == RESIDENT_INVALID_SLOT || operation->slot_b == RESIDENT_INVALID_SLOT ||
                operation->slot_out_real == RESIDENT_INVALID_SLOT || operation->args.left_rank > UPMEM_GENERIC_MAX_RANK ||
                operation->args.right_rank > UPMEM_GENERIC_MAX_RANK || operation->args.output_rank > UPMEM_GENERIC_MAX_RANK ||
                operation->args.contracted_rank > UPMEM_GENERIC_MAX_RANK || operation->args.output_elems != operation->output_elements ||
                operation->args.left_elems > UPMEM_GENERIC_MAX_ELEMS || operation->args.right_elems > UPMEM_GENERIC_MAX_ELEMS ||
                operation->args.contracted_elems > UPMEM_GENERIC_MAX_ELEMS ||
                request->slots[operation->slot_a].capacity_elements < operation->args.left_elems ||
                request->slots[operation->slot_b].capacity_elements < operation->args.right_elems ||
                request->slots[operation->slot_out_real].capacity_elements < operation->output_elements ||
                request->slots[operation->slot_a].element_count < operation->args.left_elems ||
                request->slots[operation->slot_b].element_count < operation->args.right_elems ||
                request->slots[operation->slot_out_real].element_count < operation->output_elements ||
                !slot_ready[operation->slot_a] || !slot_ready[operation->slot_b]) {
                resident_error(error_message, "resident_package_contract_metadata_invalid");
                return 1;
            }
#if RESIDENT_PACKAGE_ABI_VERSION == RESIDENT_PACKAGE_VERSION_V3
            if (operation->mode == RESIDENT_MODE_HOST_PACKED_INT8) {
                const resident_slot_descriptor_t *left = &request->slots[operation->slot_a];
                const resident_slot_descriptor_t *right = &request->slots[operation->slot_b];
                const resident_slot_descriptor_t *output = &request->slots[operation->slot_out_real];
                if (!packed_package || operation->args.operand_mode != UPMEM_GENERIC_MODE_HOST_PACKED_INT8 ||
                    !isfinite(operation->left_scale) || !isfinite(operation->right_scale) ||
                    operation->left_scale <= 0.0f || operation->right_scale <= 0.0f ||
                    operation->args.contracted_elems > RESIDENT_PACKED_INT8_MAX_CONTRACTED ||
                    operation->args.contracted_elems >
                        (uint32_t)(INT32_MAX /
                            (RESIDENT_PACKED_INT8_MAX_ABS * RESIDENT_PACKED_INT8_MAX_ABS)) ||
                    left->storage_kind != RESIDENT_STORAGE_PACKED_INT8 ||
                    right->storage_kind != RESIDENT_STORAGE_PACKED_INT8 ||
                    output->storage_kind != RESIDENT_STORAGE_INT32) {
                    resident_error(error_message, "resident_package_packed_int8_contract_invalid");
                    return 1;
                }
            } else if (packed_package ||
                       request->slots[operation->slot_a].storage_kind != RESIDENT_STORAGE_FLOAT32 ||
                       request->slots[operation->slot_b].storage_kind != RESIDENT_STORAGE_FLOAT32 ||
                       request->slots[operation->slot_out_real].storage_kind != RESIDENT_STORAGE_FLOAT32) {
                resident_error(error_message, "resident_package_float32_storage_contract_invalid");
                return 1;
            }
#endif
            slot_ready[operation->slot_out_real] = 1;
            continue;
        } else if (operation->slot_a == RESIDENT_INVALID_SLOT || operation->slot_b == RESIDENT_INVALID_SLOT ||
                   operation->slot_c == RESIDENT_INVALID_SLOT || operation->slot_d == RESIDENT_INVALID_SLOT ||
                   operation->slot_out_real == RESIDENT_INVALID_SLOT || operation->slot_out_imag == RESIDENT_INVALID_SLOT ||
                   request->slots[operation->slot_a].capacity_elements < operation->output_elements ||
                   request->slots[operation->slot_b].capacity_elements < operation->output_elements ||
                   request->slots[operation->slot_c].capacity_elements < operation->output_elements ||
                   request->slots[operation->slot_d].capacity_elements < operation->output_elements ||
                   request->slots[operation->slot_out_real].capacity_elements < operation->output_elements ||
                   request->slots[operation->slot_out_imag].capacity_elements < operation->output_elements ||
                   request->slots[operation->slot_a].element_count < operation->output_elements ||
                   request->slots[operation->slot_b].element_count < operation->output_elements ||
                   request->slots[operation->slot_c].element_count < operation->output_elements ||
                   request->slots[operation->slot_d].element_count < operation->output_elements ||
                   request->slots[operation->slot_out_real].element_count < operation->output_elements ||
                   request->slots[operation->slot_out_imag].element_count < operation->output_elements) {
            resident_error(error_message, "resident_package_complex_combine_metadata_invalid");
            return 1;
        }
        if (!slot_ready[operation->slot_a] || !slot_ready[operation->slot_b] ||
            !slot_ready[operation->slot_c] || !slot_ready[operation->slot_d]) {
            resident_error(error_message, "hardware_profile_violation: resident package complex slot read before initialization");
            return 1;
        }
#if RESIDENT_PACKAGE_ABI_VERSION == RESIDENT_PACKAGE_VERSION_V3
        if (packed_package ||
            request->slots[operation->slot_a].storage_kind != RESIDENT_STORAGE_FLOAT32 ||
            request->slots[operation->slot_b].storage_kind != RESIDENT_STORAGE_FLOAT32 ||
            request->slots[operation->slot_c].storage_kind != RESIDENT_STORAGE_FLOAT32 ||
            request->slots[operation->slot_d].storage_kind != RESIDENT_STORAGE_FLOAT32 ||
            request->slots[operation->slot_out_real].storage_kind != RESIDENT_STORAGE_FLOAT32 ||
            request->slots[operation->slot_out_imag].storage_kind != RESIDENT_STORAGE_FLOAT32) {
            resident_error(error_message, "resident_package_complex_storage_contract_invalid");
            return 1;
        }
#endif
        slot_ready[operation->slot_out_real] = 1;
        slot_ready[operation->slot_out_imag] = 1;
    }
#if RESIDENT_PACKAGE_ABI_VERSION == RESIDENT_PACKAGE_VERSION_V3
    if (packed_package) {
        const resident_operation_t *operation = &request->operations[0];
        if (header.slot_count != 3u || header.operation_count != 1u ||
            header.initial_slot_count != 2u || header.final_output_count != 1u ||
            operation->kind != RESIDENT_OPERATION_CONTRACT ||
            operation->mode != RESIDENT_MODE_HOST_PACKED_INT8 ||
            operation->slot_c != RESIDENT_INVALID_SLOT || operation->slot_d != RESIDENT_INVALID_SLOT ||
            operation->slot_out_imag != RESIDENT_INVALID_SLOT) {
            resident_error(error_message, "resident_package_packed_int8_profile_invalid");
            return 1;
        }
    }
#endif
    for (uint32_t index = 0; index < header.slot_count; index++) {
        if ((request->slot_flags[index] & RESIDENT_SLOT_FINAL_FLAG) != 0u && !slot_ready[index]) {
            resident_error(error_message, "hardware_profile_violation: resident final slot was not produced");
            return 1;
        }
    }
    return 0;
}

static int resident_parse_file_entries(
    const char *contents,
    const char *array_key,
    const char *path_key,
    const char *base,
    resident_request_t *request,
    int final_entries,
    char **error_message
) {
    const char *cursor;
    const char *end;
    if (resident_manifest_array(contents, array_key, &cursor, &end) != 0) {
        resident_error(error_message, "manifest_parse_failed: resident file array missing");
        return 1;
    }
    const size_t expected = final_entries ? request->header.final_output_count : request->header.initial_slot_count;
    if (final_entries) {
        request->final_outputs = (resident_final_file_t *)calloc(expected, sizeof(*request->final_outputs));
    } else {
        request->inputs = (resident_input_file_t *)calloc(expected, sizeof(*request->inputs));
    }
    if (expected != 0u && (final_entries ? request->final_outputs == NULL : request->inputs == NULL)) {
        resident_error(error_message, "manifest_parse_failed: resident file entry allocation failed");
        return 1;
    }
    if (final_entries) request->final_count = expected;
    else request->input_count = expected;
    for (size_t index = 0; index < expected; index++) {
        const char *object;
        const char *object_end;
        char *object_copy = NULL;
        char *path_ref = NULL;
        char *storage_dtype_ref = NULL;
        char *raw_output_ref = NULL;
        uint64_t slot_id;
        uint64_t elements;
        uint64_t raw_bytes_field;
        uint64_t transfer_bytes_field;
        const char *expected_component = NULL;
        if (index != 0u) {
            cursor = resident_skip_space(cursor, end);
            if (cursor >= end || *cursor != ',') {
                resident_error(error_message, "manifest_parse_failed: resident file entry count is short or malformed");
                return 1;
            }
            cursor++;
        }
        if (resident_next_object(&cursor, end, &object, &object_end) != 0 ||
            (object_copy = resident_copy_object(object, object_end)) == NULL ||
            resident_uint_field(object_copy, "slot_id", &slot_id) != 0 ||
            resident_uint_field(object_copy, "elements", &elements) != 0 ||
            resident_uint_field(object_copy, "raw_bytes", &raw_bytes_field) != 0 ||
            resident_uint_field(object_copy, "transfer_bytes", &transfer_bytes_field) != 0 ||
            slot_id >= request->header.slot_count || elements == 0u || elements > UINT32_MAX ||
            resident_string_field(object_copy, path_key, &path_ref) != 0) {
            free(path_ref);
            free(object_copy);
            resident_error(error_message, "manifest_parse_failed: resident file entry invalid");
            return 1;
        }
        char *path = resident_resolve(base, path_ref);
        free(path_ref);
        size_t raw_bytes = 0u;
        size_t transfer_bytes = 0u;
        const resident_slot_descriptor_t *slot = &request->slots[slot_id];
        const uint32_t element_bytes = resident_slot_element_bytes(slot);
        const uint32_t storage_kind = resident_slot_storage_kind(slot);
        const char *expected_storage_dtype = resident_storage_dtype(storage_kind);
#if RESIDENT_PACKAGE_ABI_VERSION == RESIDENT_PACKAGE_VERSION_V3
        if (resident_string_field(object_copy, "storage_dtype", &storage_dtype_ref) != 0 ||
            expected_storage_dtype == NULL || strcmp(storage_dtype_ref, expected_storage_dtype) != 0) {
            free(path);
            free(storage_dtype_ref);
            free(object_copy);
            resident_error(error_message, "hardware_profile_violation: resident file storage dtype mismatch");
            return 1;
        }
#endif
#if RESIDENT_PACKAGE_ABI_VERSION != RESIDENT_PACKAGE_VERSION_V3
        (void)storage_dtype_ref;
        (void)raw_output_ref;
        (void)expected_storage_dtype;
#endif
        free(storage_dtype_ref);
        if (path == NULL || resident_transfer_size((uint32_t)elements, element_bytes,
                &raw_bytes, &transfer_bytes) != 0) {
            free(path);
            free(object_copy);
            resident_error(error_message, "hardware_profile_violation: resident file path or byte overflow");
            return 1;
        }
        {
            const uint32_t required_flag = final_entries ? RESIDENT_SLOT_FINAL_FLAG : RESIDENT_SLOT_INITIAL_FLAG;
            const uint32_t forbidden_flag = final_entries ? RESIDENT_SLOT_INITIAL_FLAG : RESIDENT_SLOT_FINAL_FLAG;
            uint32_t expected_elements = 0u;
            for (uint32_t operation_index = 0; operation_index < request->header.operation_count; operation_index++) {
                const resident_operation_t *operation = &request->operations[operation_index];
                if (final_entries) {
                    if (operation->slot_out_real == slot_id || operation->slot_out_imag == slot_id) {
                        expected_elements = operation->output_elements;
                        expected_component = operation->slot_out_imag == slot_id ? "imag" : "real";
                    }
                } else if (operation->kind == RESIDENT_OPERATION_CONTRACT) {
                    if (operation->slot_a == slot_id) expected_elements = operation->args.left_elems;
                    if (operation->slot_b == slot_id) expected_elements = operation->args.right_elems;
                    if (expected_elements != 0u) break;
                } else if (operation->slot_a == slot_id || operation->slot_b == slot_id ||
                           operation->slot_c == slot_id || operation->slot_d == slot_id) {
                    expected_elements = slot->element_count;
                    break;
                }
            }
            if ((request->slot_flags[slot_id] & required_flag) == 0u ||
                (request->slot_flags[slot_id] & forbidden_flag) != 0u ||
                expected_elements == 0u || elements != expected_elements ||
                elements > slot->element_count || raw_bytes_field != raw_bytes ||
                transfer_bytes_field != transfer_bytes) {
                free(path);
                free(object_copy);
                resident_error(error_message, "hardware_profile_violation: resident file entry does not bind the required slot descriptor");
                return 1;
            }
        }
        if (final_entries) {
            char *component = NULL;
            if (resident_string_field(object_copy, "component", &component) != 0 || component == NULL || component[0] == '\0') {
                free(path); free(component); free(object_copy);
                resident_error(error_message, "manifest_parse_failed: resident final component missing");
                return 1;
            }
            if (expected_component == NULL || strcmp(component, expected_component) != 0) {
                free(path);
                free(component);
                free(object_copy);
                resident_error(error_message, "hardware_profile_violation: resident final output component is substituted");
                return 1;
            }
            request->final_outputs[index].component = component;
            request->final_outputs[index].slot_id = (uint32_t)slot_id;
            request->final_outputs[index].elements = (uint32_t)elements;
            request->final_outputs[index].element_bytes = element_bytes;
            request->final_outputs[index].storage_kind = storage_kind;
            request->final_outputs[index].path = path;
#if RESIDENT_PACKAGE_ABI_VERSION == RESIDENT_PACKAGE_VERSION_V3
            if (storage_kind == RESIDENT_STORAGE_INT32) {
                if (resident_string_field(object_copy, "raw_output_path", &raw_output_ref) != 0 ||
                    (request->final_outputs[index].raw_output_path =
                        resident_resolve(base, raw_output_ref)) == NULL) {
                    free(raw_output_ref);
                    free(object_copy);
                    resident_error(error_message, "manifest_parse_failed: packed int32 raw output path missing");
                    return 1;
                }
                free(raw_output_ref);
            }
#endif
            request->final_outputs[index].raw_bytes = raw_bytes;
            request->final_outputs[index].transfer_bytes = transfer_bytes;
            request->final_outputs[index].status = 0;
        } else {
            request->inputs[index].slot_id = (uint32_t)slot_id;
            request->inputs[index].elements = (uint32_t)elements;
            request->inputs[index].element_bytes = element_bytes;
            request->inputs[index].storage_kind = storage_kind;
            request->inputs[index].path = path;
#if RESIDENT_PACKAGE_ABI_VERSION == RESIDENT_PACKAGE_VERSION_V3
            if (resident_string_field(object_copy, "logical_sha256",
                    &request->inputs[index].logical_sha256) != 0 ||
                request->inputs[index].logical_sha256 == NULL ||
                strlen(request->inputs[index].logical_sha256) != 64u) {
                free(object_copy);
                resident_error(error_message, "manifest_parse_failed: resident input hash missing");
                return 1;
            }
#endif
            request->inputs[index].raw_bytes = raw_bytes;
            request->inputs[index].transfer_bytes = transfer_bytes;
        }
        free(object_copy);
        cursor = object_end + 1;
    }
    cursor = resident_skip_space(cursor, end);
    if (cursor != end) {
        resident_error(error_message, "manifest_parse_failed: resident file entry array has extra or malformed entries");
        return 1;
    }
    return 0;
}

static int resident_request_load_profile(
    const char *manifest_path,
    uint32_t max_requested_dpus,
    int v3_profile,
    resident_request_t *request,
    char **error_message
) {
    unsigned char *manifest_bytes = NULL;
    size_t manifest_length = 0u;
    unsigned char *package_bytes = NULL;
    size_t package_length = 0u;
    char *base = NULL;
    char *dpu_ref = NULL;
    char *package_ref = NULL;
    char *session_id = NULL;
    char *package_path = NULL;
    char *route_id = NULL;
    char *backend_id = NULL;
    char *profile_version = NULL;
    char *allocation_profile = NULL;
    char *target = NULL;
    char *session_protocol = NULL;
    char *quantization_mode = NULL;
    char *package_magic = NULL;
    char *dpu_binary_abi = NULL;
    uint64_t manifest_requested_dpus = 0u;
    uint64_t manifest_requested_dpu_count = 0u;
    uint64_t manifest_tasklets = 0u;
    int failed = 1;
    if (request == NULL || manifest_path == NULL || max_requested_dpus == 0u) {
        resident_error(error_message, "manifest_parse_failed: resident request arguments invalid");
        return 1;
    }
    memset(request, 0, sizeof(*request));
    if (resident_read_file(manifest_path, &manifest_bytes, &manifest_length) != 0) {
        resident_error(error_message, "manifest_parse_failed: resident manifest unreadable");
        goto done;
    }
    if (resident_string_field((char *)manifest_bytes, "schema_version", &package_ref) != 0 || strcmp(package_ref, RESIDENT_SESSION_SCHEMA) != 0) {
        resident_error(error_message, "hardware_profile_violation: resident manifest schema mismatch");
        goto done;
    }
    free(package_ref); package_ref = NULL;
    if (resident_string_field((char *)manifest_bytes, "manifest_kind", &package_ref) != 0 || strcmp(package_ref, RESIDENT_REQUEST_KIND) != 0) {
        resident_error(error_message, "hardware_profile_violation: resident manifest kind mismatch");
        goto done;
    }
    free(package_ref); package_ref = NULL;
    if (resident_string_field((char *)manifest_bytes, "session_id", &session_id) != 0 ||
        resident_string_field((char *)manifest_bytes, "dpu_binary", &dpu_ref) != 0 ||
        resident_string_field((char *)manifest_bytes, "package_path", &package_ref) != 0 ||
        resident_string_field((char *)manifest_bytes, "route_id", &route_id) != 0 ||
        resident_string_field((char *)manifest_bytes, "backend_id", &backend_id) != 0 ||
        resident_string_field((char *)manifest_bytes, "hardware_profile_version", &profile_version) != 0 ||
        resident_string_field((char *)manifest_bytes, "target", &target) != 0 ||
        resident_string_field((char *)manifest_bytes, "sdk_allocation_profile", &allocation_profile) != 0 ||
        resident_string_field((char *)manifest_bytes, "session_protocol", &session_protocol) != 0 ||
        resident_string_field((char *)manifest_bytes, "quantization_mode", &quantization_mode) != 0 ||
        resident_string_field((char *)manifest_bytes, "package_magic", &package_magic) != 0 ||
        resident_string_field((char *)manifest_bytes, "dpu_binary_abi", &dpu_binary_abi) != 0) {
        resident_error(error_message, "manifest_parse_failed: resident manifest identity missing");
        goto done;
    }
    if (resident_uint_field((char *)manifest_bytes, "requested_dpus", &manifest_requested_dpus) != 0 ||
        resident_uint_field((char *)manifest_bytes, "requested_dpu_count", &manifest_requested_dpu_count) != 0 ||
        resident_uint_field((char *)manifest_bytes, "tasklets", &manifest_tasklets) != 0) {
        resident_error(error_message, "hardware_profile_violation: distributed package resident request DPU/tasklet identity missing");
        goto done;
    }
    if (manifest_requested_dpus != manifest_requested_dpu_count) {
        resident_error(error_message, "hardware_profile_violation: distributed package resident request DPU counts conflict");
        goto done;
    }
    {
        int binary_identity_valid;
        if (v3_profile != 0) {
            /* Package v3 retains operation ABI v2 but has typed slots and a
             * separate tasklet-keyed binary identity. */
            binary_identity_valid = strcmp(dpu_binary_abi, "dpu_resident_v3") == 0 &&
                resident_binary_matches_v3_tasklets(dpu_ref, manifest_tasklets);
        } else {
            binary_identity_valid = strcmp(dpu_binary_abi,
#if RESIDENT_OPERATION_ABI_VERSION == RESIDENT_OPERATION_ABI_V2
                "dpu_resident_v2"
#else
                "dpu_resident"
#endif
                ) == 0;
#if RESIDENT_OPERATION_ABI_VERSION == RESIDENT_OPERATION_ABI_V2
            binary_identity_valid = binary_identity_valid && resident_binary_matches_abi(dpu_ref);
#endif
        }
        if (resident_ascii_identifier(session_id) != 0 || strcmp(route_id, RESIDENT_ROUTE_ID) != 0 ||
        strcmp(backend_id, RESIDENT_BACKEND_ID) != 0 || strcmp(profile_version, RESIDENT_PROFILE_VERSION) != 0 ||
        strcmp(target, RESIDENT_TARGET) != 0 || strcmp(allocation_profile, RESIDENT_ALLOCATION_PROFILE) != 0 ||
        strcmp(session_protocol, RESIDENT_SESSION_SCHEMA) != 0 ||
        strcmp(package_magic, RESIDENT_PACKAGE_MAGIC) != 0 ||
        binary_identity_valid == 0 ||
        (strcmp(quantization_mode, "none") != 0 &&
         strcmp(quantization_mode, "per_task_resident_requantize") != 0 &&
         strcmp(quantization_mode, "host_packed_int8") != 0)) {
            resident_error(error_message, "hardware_profile_violation: resident manifest ABI or hardware identity mismatch");
            goto done;
        }
    }
    base = resident_base(manifest_path);
    request->manifest_root = base;
    base = NULL;
    request->session_id = session_id; session_id = NULL;
    request->route_id = route_id; route_id = NULL;
    request->backend_id = backend_id; backend_id = NULL;
    request->profile_version = profile_version; profile_version = NULL;
    request->allocation_profile = allocation_profile; allocation_profile = NULL;
    request->quantization_mode = quantization_mode; quantization_mode = NULL;
    request->dpu_binary_path = resident_resolve(request->manifest_root, dpu_ref);
    package_path = resident_resolve(request->manifest_root, package_ref);
    if (request->dpu_binary_path == NULL || package_path == NULL) {
        resident_error(error_message, "hardware_profile_violation: resident manifest path containment failed");
        goto done;
    }
    request->package_path = package_path;
    package_path = NULL;
    uint64_t value;
    uint64_t manifest_package_version;
    uint64_t manifest_operation_abi_version;
    uint64_t manifest_operation_bytes;
    if (resident_uint_field((char *)manifest_bytes, "package_version", &manifest_package_version) != 0 ||
        resident_uint_field((char *)manifest_bytes, "operation_abi_version", &manifest_operation_abi_version) != 0 ||
        resident_uint_field((char *)manifest_bytes, "operation_bytes", &manifest_operation_bytes) != 0 ||
        manifest_package_version != RESIDENT_PACKAGE_VERSION ||
        manifest_operation_abi_version != RESIDENT_OPERATION_ABI_VERSION ||
        manifest_operation_bytes != RESIDENT_OPERATION_BYTES) {
        resident_error(error_message, "hardware_profile_violation: resident manifest ABI identity mismatch");
        goto done;
    }
    if (manifest_requested_dpus == 0u || manifest_requested_dpus > max_requested_dpus ||
        (v3_profile != 0
            ? (manifest_tasklets < 1u || manifest_tasklets > 24u)
            : !resident_supported_tasklets(manifest_tasklets)) ||
        manifest_tasklets != NR_TASKLETS ||
        resident_uint_field((char *)manifest_bytes, "graph_request_count", &value) != 0 || value != 1u ||
        resident_uint_field((char *)manifest_bytes, "logical_task_count", &value) != 0 || value == 0u || value > RESIDENT_MAX_LOGICAL_TASKS) {
        resident_error(error_message, "hardware_profile_violation: resident request DPU/tasklet/graph limits exceeded");
        goto done;
    }
    request->requested_dpus = (uint32_t)manifest_requested_dpus;
    if (resident_uint_field((char *)manifest_bytes, "logical_task_count", &value) != 0) {
        resident_error(error_message, "manifest_parse_failed: resident logical task count missing");
        goto done;
    }
    request->logical_task_count = (uint32_t)value;
    if (resident_read_file(request->package_path, &package_bytes, &package_length) != 0 ||
        resident_validate_package(package_bytes, package_length, request, error_message) != 0) {
        if (error_message != NULL && *error_message == NULL) resident_error(error_message, "hardware_profile_violation: resident package validation failed");
        goto done;
    }
    {
        uint32_t expected_mode;
        if (strcmp(request->quantization_mode, "none") == 0) {
            expected_mode = RESIDENT_MODE_FLOAT32;
        } else if (strcmp(request->quantization_mode, "per_task_resident_requantize") == 0) {
            expected_mode = RESIDENT_MODE_PER_TASK_REQUANTIZE;
        } else if (strcmp(request->quantization_mode, "host_packed_int8") == 0) {
            expected_mode = RESIDENT_MODE_HOST_PACKED_INT8;
        } else {
            resident_error(error_message, "hardware_profile_violation: unsupported resident numeric mode");
            goto done;
        }
        for (uint32_t index = 0; index < request->header.operation_count; index++) {
            if (request->operations[index].mode != expected_mode) {
                resident_error(error_message, "hardware_profile_violation: resident manifest mode does not match operation descriptors");
                goto done;
            }
        }
    }
    if (resident_parse_file_entries((char *)manifest_bytes, "initial_slots", "input_path", request->manifest_root, request, 0, error_message) != 0 ||
        resident_parse_file_entries((char *)manifest_bytes, "final_outputs", "output_path", request->manifest_root, request, 1, error_message) != 0) {
        goto done;
    }
    if (resident_uint_field((char *)manifest_bytes, "component_operation_count", &value) != 0 ||
        value != request->header.operation_count ||
        resident_uint_field((char *)manifest_bytes, "slot_descriptor_count", &value) != 0 ||
        value != request->header.slot_count ||
        resident_uint_field((char *)manifest_bytes, "mram_pool_bytes", &value) != 0 ||
        value != request->header.pool_bytes) {
        resident_error(error_message, "hardware_profile_violation: resident manifest/package descriptor metadata mismatch");
        goto done;
    }
    if (request->input_count != request->header.initial_slot_count || request->final_count != request->header.final_output_count) {
        resident_error(error_message, "hardware_profile_violation: resident manifest/package file count mismatch");
        goto done;
    }
    for (size_t left = 0; left < request->input_count; left++) {
        for (size_t right = 0; right < left; right++) {
            if (request->inputs[left].slot_id == request->inputs[right].slot_id) {
                resident_error(error_message, "hardware_profile_violation: duplicate resident initial slot entry");
                goto done;
            }
        }
    }
    for (size_t left = 0; left < request->final_count; left++) {
        for (size_t right = 0; right < left; right++) {
            if (request->final_outputs[left].slot_id == request->final_outputs[right].slot_id ||
                strcmp(request->final_outputs[left].component, request->final_outputs[right].component) == 0) {
                resident_error(error_message, "hardware_profile_violation: duplicate resident final output entry");
                goto done;
            }
        }
    }
    if (request->final_count == 1u && strcmp(request->final_outputs[0].component, "real") != 0) {
        resident_error(error_message, "hardware_profile_violation: resident real output component is missing");
        goto done;
    }
    if (request->final_count == 2u) {
        int real_seen = 0;
        int imag_seen = 0;
        for (size_t index = 0; index < request->final_count; index++) {
            real_seen |= strcmp(request->final_outputs[index].component, "real") == 0;
            imag_seen |= strcmp(request->final_outputs[index].component, "imag") == 0;
        }
        if (!real_seen || !imag_seen) {
            resident_error(error_message, "hardware_profile_violation: resident split-complex output components are incomplete");
            goto done;
        }
    }
    for (uint32_t index = 0; index < request->header.slot_count; index++) {
        request->slots[index].slot_id = index;
    }
    failed = 0;
done:
    free(manifest_bytes);
    free(package_bytes);
    free(dpu_ref);
    free(package_ref);
    free(session_id);
    free(package_path);
    free(base);
    free(route_id);
    free(backend_id);
    free(profile_version);
    free(allocation_profile);
    free(target);
    free(session_protocol);
    free(quantization_mode);
    free(package_magic);
    free(dpu_binary_abi);
    if (failed) resident_request_free(request);
    return failed;
}

int resident_request_load(const char *manifest_path, resident_request_t *request, char **error_message) {
    return resident_request_load_profile(manifest_path, 1u, 0, request, error_message);
}

int resident_request_load_execution_plan(
    const char *manifest_path,
    resident_request_t *request,
    char **error_message
) {
    return resident_request_load_profile(manifest_path, 2u, 0, request, error_message);
}

int resident_request_load_execution_plan_v2(
    const char *manifest_path,
    resident_request_t *request,
    char **error_message
) {
    return resident_request_load_profile(manifest_path, 4u, 0, request, error_message);
}

int resident_request_load_execution_plan_v3(
    const char *manifest_path,
    resident_request_t *request,
    char **error_message
) {
#if defined(RESIDENT_V3)
    return resident_request_load_profile(manifest_path, 64u, 1, request, error_message);
#else
    (void)manifest_path;
    (void)request;
    resident_error(error_message, "hardware_profile_violation: v3 resident loader requires RESIDENT_V3");
    return 1;
#endif
}

void resident_request_free(resident_request_t *request) {
    if (request == NULL) return;
    free(request->session_id);
    free(request->dpu_binary_path);
    free(request->package_path);
    free(request->slots);
    free(request->operations);
    free(request->slot_flags);
    free(request->manifest_root);
    free(request->route_id);
    free(request->backend_id);
    free(request->profile_version);
    free(request->allocation_profile);
    free(request->quantization_mode);
    for (size_t index = 0; index < request->input_count; index++) {
        free(request->inputs[index].path);
        free(request->inputs[index].logical_sha256);
    }
    for (size_t index = 0; index < request->final_count; index++) {
        free(request->final_outputs[index].component);
        free(request->final_outputs[index].path);
        free(request->final_outputs[index].raw_output_path);
    }
    free(request->inputs);
    free(request->final_outputs);
    memset(request, 0, sizeof(*request));
}

static void resident_json_string(FILE *file, const char *value) {
    fputc('"', file);
    for (const unsigned char *cursor = (const unsigned char *)value; *cursor != '\0'; cursor++) {
        if (*cursor == '"' || *cursor == '\\') fputc('\\', file);
        if (*cursor == '\n') fputc('n', file);
        else if (*cursor == '\t') fputc('t', file);
        else fputc(*cursor, file);
    }
    fputc('"', file);
}

static const char *resident_output_reference(const resident_request_t *request, const char *path) {
    const size_t base_length = strlen(request->manifest_root == NULL ? "" : request->manifest_root);
    if (request->manifest_root != NULL && strncmp(path, request->manifest_root, base_length) == 0 && path[base_length] == '/') {
        return path + base_length + 1u;
    }
    return path;
}

int resident_response_write(
    const char *response_path,
    const resident_request_t *request,
    const char *status,
    const char *failure_stage,
    const char *error_message,
    uint32_t allocated_dpus,
    int sdk_error_code,
    const resident_timing_t *timing,
    uint32_t native_launch_count,
    int release_confirmed,
    uint64_t initial_h2d_bytes,
    uint64_t descriptor_h2d_bytes,
    uint64_t control_h2d_bytes,
    uint64_t final_d2h_bytes
) {
    const resident_timing_t empty = {0};
    const resident_timing_t *current = timing == NULL ? &empty : timing;
    FILE *file = fopen(response_path, "w");
    if (file == NULL) return 1;
    const int success = failure_stage == NULL && strcmp(status, "completed") == 0 &&
        allocated_dpus == 1u && native_launch_count == request->header.operation_count && release_confirmed;
    fprintf(file, "{\n  \"schema_version\": \"%s\",\n  \"manifest_kind\": \"%s\",\n", RESIDENT_SESSION_SCHEMA, RESIDENT_RESPONSE_KIND);
    fprintf(file, "  \"session_id\": "); resident_json_string(file, request->session_id == NULL ? "resident-unknown" : request->session_id); fprintf(file, ",\n");
    fprintf(file, "  \"route_id\": \"%s\",\n  \"backend_id\": \"%s\",\n  \"hardware_profile_version\": \"%s\",\n  \"target_requested\": \"%s\",\n  \"target_observed\": \"%s\",\n  \"sdk_allocation_profile\": \"%s\",\n  \"sdk_allocation_profile_verified\": %s,\n  \"session_protocol\": \"%s\",\n  \"quantization_mode\": ", RESIDENT_ROUTE_ID, RESIDENT_BACKEND_ID, RESIDENT_PROFILE_VERSION, RESIDENT_TARGET, success ? RESIDENT_TARGET : "hardware_unverified", RESIDENT_ALLOCATION_PROFILE, success ? "true" : "false", RESIDENT_SESSION_SCHEMA);
    resident_json_string(file, request->quantization_mode == NULL ? "unknown" : request->quantization_mode);
    fputs(",\n", file);
    fprintf(file, "  \"status\": \"%s\",\n  \"failure_stage\": ", status);
    if (failure_stage == NULL) fputs("null", file); else resident_json_string(file, failure_stage);
    fprintf(file, ",\n  \"error\": ");
    if (error_message == NULL) fputs("null", file); else resident_json_string(file, error_message);
    fprintf(file, ",\n  \"requested_dpus\": 1,\n  \"allocated_dpus\": %u,\n  \"tasklets\": %u,\n  \"graph_request_count\": 1,\n", allocated_dpus, (unsigned)NR_TASKLETS);
    fprintf(file, "  \"logical_task_count\": %u,\n  \"component_operation_count\": %u,\n  \"native_launch_count\": %u,\n  \"native_task_count\": %u,\n", request->logical_task_count, request->header.operation_count, native_launch_count, native_launch_count);
    fprintf(file, "  \"sdk_error_code\": %d,\n  \"package_parse_time_s\": %.9f,\n  \"allocation_time_s\": %.9f,\n  \"binary_load_time_s\": %.9f,\n  \"initial_h2d_time_s\": %.9f,\n  \"descriptor_h2d_time_s\": %.9f,\n  \"control_h2d_time_s\": %.9f,\n  \"kernel_time_s\": %.9f,\n  \"final_d2h_time_s\": %.9f,\n  \"output_write_time_s\": %.9f,\n  \"release_time_s\": %.9f,\n  \"dpu_run_time_cycles\": %llu,\n", sdk_error_code, current->package_parse_time_s, current->allocation_time_s, current->binary_load_time_s, current->initial_h2d_time_s, current->descriptor_h2d_time_s, current->control_h2d_time_s, current->kernel_time_s, current->final_d2h_time_s, current->output_write_time_s, current->release_time_s, (unsigned long long)current->dpu_run_time_cycles);
    fprintf(file, "  \"completion_abi_version\": %u,\n", (unsigned)RESIDENT_COMPLETION_VERSION);
#if RESIDENT_COMPLETION_VERSION >= 2
    fprintf(file, "  \"graph_cycle_sum\": %llu,\n  \"dpu_operation_cycles\": [", (unsigned long long)current->dpu_run_time_cycles);
    for (uint32_t operation = 0; operation < request->header.operation_count; operation++) {
        if (operation != 0u) fputs(",", file);
        fprintf(file, "%llu", (unsigned long long)current->operation_dpu_cycles[operation]);
    }
    fputs("],\n  \"tasklet_processed_elements\": [", file);
    for (uint32_t operation = 0; operation < request->header.operation_count; operation++) {
        if (operation != 0u) fputs(",", file);
        fputs("[", file);
        for (uint32_t tasklet = 0; tasklet < NR_TASKLETS; tasklet++) {
            if (tasklet != 0u) fputs(",", file);
            fprintf(file, "%u", current->operation_tasklet_processed_elements[operation][tasklet]);
        }
        fputs("]", file);
    }
    fputs("],\n  \"active_tasklet_count\": [", file);
    for (uint32_t operation = 0; operation < request->header.operation_count; operation++) {
        if (operation != 0u) fputs(",", file);
        fprintf(file, "%u", current->operation_active_tasklet_count[operation]);
    }
    fputs("],\n  \"idle_tasklet_count\": [", file);
    for (uint32_t operation = 0; operation < request->header.operation_count; operation++) {
        if (operation != 0u) fputs(",", file);
        fprintf(file, "%u", current->operation_idle_tasklet_count[operation]);
    }
    fputs("],\n  \"tasklet_utilization\": [", file);
    for (uint32_t operation = 0; operation < request->header.operation_count; operation++) {
        if (operation != 0u) fputs(",", file);
        fprintf(file, "%.6f", (double)current->operation_tasklet_utilization_ppm[operation] / 1000000.0);
    }
    fputs("],\n  \"tasklet_work_imbalance\": [", file);
    for (uint32_t operation = 0; operation < request->header.operation_count; operation++) {
        if (operation != 0u) fputs(",", file);
        fprintf(file, "%.6f", (double)current->operation_tasklet_work_imbalance_ppm[operation] / 1000000.0);
    }
    fputs("],\n", file);
#endif
    fprintf(file, "  \"initial_h2d_bytes\": %llu,\n  \"descriptor_h2d_bytes\": %llu,\n  \"control_h2d_bytes\": %llu,\n  \"final_d2h_bytes\": %llu,\n  \"intermediate_h2d_bytes\": 0,\n  \"intermediate_d2h_bytes\": 0,\n  \"actual_h2d_bytes\": %llu,\n  \"actual_d2h_bytes\": %llu,\n  \"actual_transfer_bytes\": %llu,\n", (unsigned long long)initial_h2d_bytes, (unsigned long long)descriptor_h2d_bytes, (unsigned long long)control_h2d_bytes, (unsigned long long)final_d2h_bytes, (unsigned long long)(initial_h2d_bytes + descriptor_h2d_bytes + control_h2d_bytes), (unsigned long long)final_d2h_bytes, (unsigned long long)(initial_h2d_bytes + descriptor_h2d_bytes + control_h2d_bytes + final_d2h_bytes));
    fprintf(file, "  \"allocation_count\": %u,\n  \"hardware_allocation_verified\": %s,\n  \"hardware_execution\": %s,\n  \"hardware_kernel_executed\": %s,\n  \"native_execution\": %s,\n  \"native_hardware_backend\": %s,\n  \"hardware_backend_verified\": %s,\n  \"simulator_kernel_executed\": false,\n  \"cpu_fallback_used\": false,\n  \"hardware_release_verified\": %s,\n  \"release_confirmed\": %s,\n  \"physical_dependency_chain_verified\": %s,\n  \"hardware_timing_available\": %s,\n  \"session_scope\": \"single_graph_request\",\n  \"persistent_session_reused\": false,\n  \"resident_slots_persist_for_graph\": true,\n  \"session_persistence_semantics\": \"one_native_graph_request_keeps_logical_slots_resident\",\n  \"steady_state_graph_execution_s\": %.9f,\n  \"final_output_only_d2h\": true,\n  \"physical_bus_bytes_available\": false,\n  \"final_outputs\": [",
        allocated_dpus,
        success ? "true" : "false", success ? "true" : "false", success ? "true" : "false", success ? "true" : "false",
        success ? "true" : "false", success ? "true" : "false",
        release_confirmed ? "true" : "false", release_confirmed ? "true" : "false",
        success ? "true" : "false", success ? "true" : "false",
        current->initial_h2d_time_s + current->descriptor_h2d_time_s + current->control_h2d_time_s + current->kernel_time_s + current->final_d2h_time_s + current->output_write_time_s);
    for (size_t index = 0; index < request->final_count; index++) {
        const resident_final_file_t *output = &request->final_outputs[index];
        if (index != 0u) fputs(",", file);
        fprintf(file, "{\"component\":"); resident_json_string(file, output->component);
        fprintf(file, ",\"slot_id\":%u,\"status\":\"%s\",\"output_path\":", output->slot_id, output->status == 1 && success ? "completed" : "not_run");
        resident_json_string(file, resident_output_reference(request, output->path));
        fprintf(file, ",\"elements\":%u,\"raw_bytes\":%zu,\"transfer_bytes\":%zu}", output->elements, output->raw_bytes, output->transfer_bytes);
    }
    fprintf(file, "]\n}\n");
    const int failed = ferror(file) != 0 || fclose(file) != 0;
    return failed ? 1 : 0;
}
