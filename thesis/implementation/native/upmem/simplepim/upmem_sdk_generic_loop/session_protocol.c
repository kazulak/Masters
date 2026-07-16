#include "session_protocol.h"

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef UPMEM_GENERIC_HARDWARE_MVP
#define UPMEM_GENERIC_HARDWARE_MVP 0
#endif

#if UPMEM_GENERIC_HARDWARE_MVP
#define SESSION_ALLOCATION_PROFILE_JSON "\"backend=hw\""
#else
#define SESSION_ALLOCATION_PROFILE_JSON "null"
#endif

typedef struct {
    const char *cursor;
    const char *end;
} json_reader;

static char *copy_string(const char *value) {
    size_t length = strlen(value);
    char *copy = (char *)malloc(length + 1u);
    if (copy != NULL) {
        memcpy(copy, value, length + 1u);
    }
    return copy;
}

static void set_error(char **error_message, const char *message) {
    if (error_message == NULL || *error_message != NULL) {
        return;
    }
    *error_message = copy_string(message);
}

static void skip_space(json_reader *reader) {
    while (reader->cursor < reader->end && isspace((unsigned char)*reader->cursor)) {
        reader->cursor++;
    }
}

static int consume(json_reader *reader, char expected) {
    skip_space(reader);
    if (reader->cursor >= reader->end || *reader->cursor != expected) {
        return 1;
    }
    reader->cursor++;
    return 0;
}

static int parse_string(json_reader *reader, char **value) {
    size_t capacity = 32u;
    size_t length = 0u;
    char *result = NULL;
    skip_space(reader);
    if (reader->cursor >= reader->end || *reader->cursor != '"') {
        return 1;
    }
    reader->cursor++;
    result = (char *)malloc(capacity);
    if (result == NULL) {
        return 1;
    }
    while (reader->cursor < reader->end) {
        unsigned char character = (unsigned char)*reader->cursor++;
        if (character == '"') {
            result[length] = '\0';
            *value = result;
            return 0;
        }
        if (character == '\\') {
            if (reader->cursor >= reader->end) {
                free(result);
                return 1;
            }
            character = (unsigned char)*reader->cursor++;
            if (character == '"' || character == '\\' || character == '/' ||
                character == 'b' || character == 'f' || character == 'n' ||
                character == 'r' || character == 't') {
                if (character == 'b') character = '\b';
                else if (character == 'f') character = '\f';
                else if (character == 'n') character = '\n';
                else if (character == 'r') character = '\r';
                else if (character == 't') character = '\t';
            } else {
                /* Session paths and identifiers are UTF-8 bytes, not \u escapes. */
                free(result);
                return 1;
            }
        } else if (character < 0x20u) {
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

static int parse_uint(json_reader *reader, uint32_t *value) {
    uint64_t result = 0u;
    int digits = 0;
    skip_space(reader);
    while (reader->cursor < reader->end && isdigit((unsigned char)*reader->cursor)) {
        result = result * 10u + (uint64_t)(*reader->cursor - '0');
        if (result > UINT32_MAX) {
            return 1;
        }
        reader->cursor++;
        digits = 1;
    }
    if (!digits) {
        return 1;
    }
    *value = (uint32_t)result;
    return 0;
}

static int skip_value(json_reader *reader);

static int skip_array(json_reader *reader) {
    if (consume(reader, '[') != 0) return 1;
    skip_space(reader);
    if (reader->cursor < reader->end && *reader->cursor == ']') {
        reader->cursor++;
        return 0;
    }
    for (;;) {
        if (skip_value(reader) != 0) return 1;
        skip_space(reader);
        if (reader->cursor < reader->end && *reader->cursor == ']') {
            reader->cursor++;
            return 0;
        }
        if (consume(reader, ',') != 0) return 1;
    }
}

static int skip_object(json_reader *reader) {
    if (consume(reader, '{') != 0) return 1;
    skip_space(reader);
    if (reader->cursor < reader->end && *reader->cursor == '}') {
        reader->cursor++;
        return 0;
    }
    for (;;) {
        char *key = NULL;
        int failed = parse_string(reader, &key) != 0;
        free(key);
        if (failed || consume(reader, ':') != 0 || skip_value(reader) != 0) return 1;
        skip_space(reader);
        if (reader->cursor < reader->end && *reader->cursor == '}') {
            reader->cursor++;
            return 0;
        }
        if (consume(reader, ',') != 0) return 1;
    }
}

static int skip_value(json_reader *reader) {
    char *string = NULL;
    int digits = 0;
    skip_space(reader);
    if (reader->cursor >= reader->end) return 1;
    if (*reader->cursor == '"') return parse_string(reader, &string) == 0 ? (free(string), 0) : 1;
    if (*reader->cursor == '{') return skip_object(reader);
    if (*reader->cursor == '[') return skip_array(reader);
    if (strncmp(reader->cursor, "true", 4) == 0) { reader->cursor += 4; return 0; }
    if (strncmp(reader->cursor, "false", 5) == 0) { reader->cursor += 5; return 0; }
    if (strncmp(reader->cursor, "null", 4) == 0) { reader->cursor += 4; return 0; }
    if (*reader->cursor == '-') reader->cursor++;
    while (reader->cursor < reader->end && isdigit((unsigned char)*reader->cursor)) {
        reader->cursor++;
        digits = 1;
    }
    if (reader->cursor < reader->end && *reader->cursor == '.') {
        reader->cursor++;
        while (reader->cursor < reader->end && isdigit((unsigned char)*reader->cursor)) {
            reader->cursor++;
            digits = 1;
        }
    }
    if (reader->cursor < reader->end &&
        (*reader->cursor == 'e' || *reader->cursor == 'E')) {
        reader->cursor++;
        if (reader->cursor < reader->end &&
            (*reader->cursor == '+' || *reader->cursor == '-')) {
            reader->cursor++;
        }
        while (reader->cursor < reader->end && isdigit((unsigned char)*reader->cursor)) {
            reader->cursor++;
            digits = 1;
        }
    }
    if (digits) return 0;
    while (reader->cursor < reader->end &&
           !isspace((unsigned char)*reader->cursor) &&
           *reader->cursor != ',' && *reader->cursor != '}' && *reader->cursor != ']') {
        reader->cursor++;
    }
    return 1;
}

static int path_is_safe_relative(const char *path) {
    const char *cursor = path;
    if (path[0] == '\0' || path[0] == '/') return 1;
    while (*cursor != '\0') {
        const char *start = cursor;
        while (*cursor != '\0' && *cursor != '/') cursor++;
        if ((size_t)(cursor - start) == 2u && start[0] == '.' && start[1] == '.') return 1;
        if (*cursor == '/') cursor++;
    }
    return 0;
}

static char *resolve_path(const char *base, const char *relative) {
    size_t base_length;
    size_t relative_length;
    char *resolved;
    if (path_is_safe_relative(relative)) return NULL;
    base_length = strlen(base);
    relative_length = strlen(relative);
    resolved = (char *)malloc(base_length + 1u + relative_length + 1u);
    if (resolved == NULL) return NULL;
    memcpy(resolved, base, base_length);
    resolved[base_length] = '/';
    memcpy(resolved + base_length + 1u, relative, relative_length + 1u);
    return resolved;
}

static char *manifest_base(const char *manifest_path) {
    const char *slash = strrchr(manifest_path, '/');
    size_t length = slash == NULL ? 1u : (size_t)(slash - manifest_path);
    char *base = (char *)malloc(length + 1u);
    if (base == NULL) return NULL;
    if (slash == NULL) memcpy(base, ".", 2u);
    else {
        memcpy(base, manifest_path, length);
        base[length] = '\0';
    }
    return base;
}

static void free_task_fields(upmem_generic_session_task *task) {
    free(task->task_id);
    free(task->args_ref);
    free(task->left_ref);
    free(task->right_ref);
    free(task->output_ref);
    free(task->args_path);
    free(task->left_path);
    free(task->right_path);
    free(task->output_path);
    memset(task, 0, sizeof(*task));
}

static int parse_task(json_reader *reader, upmem_generic_session_task *task, const char *base) {
    int has_id = 0, has_args = 0, has_left = 0, has_right = 0, has_output = 0;
    if (consume(reader, '{') != 0) return 1;
    memset(task, 0, sizeof(*task));
    for (;;) {
        char *key = NULL;
        if (parse_string(reader, &key) != 0 || consume(reader, ':') != 0) { free(key); free_task_fields(task); return 1; }
        if (strcmp(key, "task_id") == 0) { free(task->task_id); if (parse_string(reader, &task->task_id) != 0) { free(key); free_task_fields(task); return 1; } has_id = 1; }
        else if (strcmp(key, "args_path") == 0) { free(task->args_ref); if (parse_string(reader, &task->args_ref) != 0) { free(key); free_task_fields(task); return 1; } has_args = 1; }
        else if (strcmp(key, "left_path") == 0) { free(task->left_ref); if (parse_string(reader, &task->left_ref) != 0) { free(key); free_task_fields(task); return 1; } has_left = 1; }
        else if (strcmp(key, "right_path") == 0) { free(task->right_ref); if (parse_string(reader, &task->right_ref) != 0) { free(key); free_task_fields(task); return 1; } has_right = 1; }
        else if (strcmp(key, "output_path") == 0) { free(task->output_ref); if (parse_string(reader, &task->output_ref) != 0) { free(key); free_task_fields(task); return 1; } has_output = 1; }
        else if (skip_value(reader) != 0) { free(key); free_task_fields(task); return 1; }
        free(key);
        skip_space(reader);
        if (reader->cursor < reader->end && *reader->cursor == '}') { reader->cursor++; break; }
        if (consume(reader, ',') != 0) { free_task_fields(task); return 1; }
    }
    if (!has_id || !has_args || !has_left || !has_right || !has_output ||
        task->task_id == NULL || task->task_id[0] == '\0') {
        free_task_fields(task);
        return 1;
    }
    task->args_path = resolve_path(base, task->args_ref);
    task->left_path = resolve_path(base, task->left_ref);
    task->right_path = resolve_path(base, task->right_ref);
    task->output_path = resolve_path(base, task->output_ref);
    if (task->args_path == NULL || task->left_path == NULL ||
        task->right_path == NULL || task->output_path == NULL) {
        free_task_fields(task);
        return 1;
    }
    return 0;
}

static int parse_tasks(json_reader *reader, upmem_generic_session *session, const char *base) {
    if (consume(reader, '[') != 0) return 1;
    skip_space(reader);
    if (reader->cursor < reader->end && *reader->cursor == ']') { reader->cursor++; return 1; }
    for (;;) {
        upmem_generic_session_task task;
        upmem_generic_session_task *grown;
        if (session->task_count >= UPMEM_GENERIC_SESSION_MAX_TASKS ||
            parse_task(reader, &task, base) != 0) return 1;
        for (size_t i = 0; i < session->task_count; i++) {
            if (strcmp(session->tasks[i].task_id, task.task_id) == 0 ||
                strcmp(session->tasks[i].output_ref, task.output_ref) == 0) {
                free_task_fields(&task);
                return 1;
            }
        }
        grown = (upmem_generic_session_task *)realloc(
            session->tasks, (session->task_count + 1u) * sizeof(*grown)
        );
        if (grown == NULL) {
            free_task_fields(&task);
            return 1;
        }
        session->tasks = grown;
        session->tasks[session->task_count++] = task;
        skip_space(reader);
        if (reader->cursor < reader->end && *reader->cursor == ']') { reader->cursor++; return 0; }
        if (consume(reader, ',') != 0) return 1;
    }
}

static int read_manifest(const char *path, char **contents, size_t *length) {
    FILE *file = fopen(path, "rb");
    long size;
    if (file == NULL || fseek(file, 0, SEEK_END) != 0) {
        if (file) fclose(file);
        return 1;
    }
    size = ftell(file);
    if (size < 0 || (unsigned long)size > 4u * 1024u * 1024u ||
        fseek(file, 0, SEEK_SET) != 0) {
        fclose(file);
        return 1;
    }
    *contents = (char *)malloc((size_t)size + 1u);
    if (*contents == NULL ||
        fread(*contents, 1, (size_t)size, file) != (size_t)size) {
        free(*contents);
        *contents = NULL;
        fclose(file);
        return 1;
    }
    fclose(file);
    (*contents)[size] = '\0';
    *length = (size_t)size;
    return 0;
}

static int load_task_manifest(
    const char *manifest_path,
    upmem_generic_session *session,
    char **error_message,
    const char *schema,
    const char *kind,
    const char *id_key,
    int require_binary,
    int require_nonempty_tasks,
    int interactive_request
) {
    char *contents = NULL;
    char *base = NULL;
    size_t length = 0u;
    json_reader reader;
    int has_schema = 0;
    int has_kind = 0;
    int has_binary = 0;
    int has_dpus = 0;
    int has_tasklets = 0;
    int has_tasks = 0;

    memset(session, 0, sizeof(*session));
    if (read_manifest(manifest_path, &contents, &length) != 0) {
        set_error(error_message, "unable to read session manifest");
        return 1;
    }
    base = manifest_base(manifest_path);
    if (base == NULL) {
        free(contents);
        set_error(error_message, "unable to allocate manifest path");
        return 1;
    }
    reader.cursor = contents;
    reader.end = contents + length;
    if (consume(&reader, '{') != 0) goto parse_failed;
    for (;;) {
        char *key = NULL;
        if (parse_string(&reader, &key) != 0 || consume(&reader, ':') != 0) {
            free(key);
            goto parse_failed;
        }
        if (strcmp(key, "schema_version") == 0) {
            char *value = NULL;
            if (parse_string(&reader, &value) != 0) { free(key); goto parse_failed; }
            has_schema = strcmp(value, schema) == 0;
            free(value);
        } else if (strcmp(key, "manifest_kind") == 0) {
            char *value = NULL;
            if (parse_string(&reader, &value) != 0) { free(key); goto parse_failed; }
            has_kind = strcmp(value, kind) == 0;
            free(value);
        } else if (strcmp(key, id_key) == 0) {
            free(session->session_id);
            if (parse_string(&reader, &session->session_id) != 0) { free(key); goto parse_failed; }
        } else if (strcmp(key, "dpu_binary") == 0) {
            free(session->dpu_binary_ref);
            if (parse_string(&reader, &session->dpu_binary_ref) != 0) { free(key); goto parse_failed; }
            has_binary = 1;
        } else if (strcmp(key, "requested_dpus") == 0) {
            if (parse_uint(&reader, &session->requested_dpus) != 0) { free(key); goto parse_failed; }
            has_dpus = 1;
        } else if (strcmp(key, "tasklets") == 0) {
            if (parse_uint(&reader, &session->tasklets) != 0) { free(key); goto parse_failed; }
            has_tasklets = 1;
        } else if (strcmp(key, "tasks") == 0) {
            if (has_tasks || parse_tasks(&reader, session, base) != 0) { free(key); goto parse_failed; }
            has_tasks = 1;
        } else if (skip_value(&reader) != 0) {
            free(key);
            goto parse_failed;
        }
        free(key);
        skip_space(&reader);
        if (reader.cursor < reader.end && *reader.cursor == '}') {
            reader.cursor++;
            break;
        }
        if (consume(&reader, ',') != 0) goto parse_failed;
    }
    skip_space(&reader);
    if (reader.cursor != reader.end || !has_schema || !has_kind ||
        (require_binary && !has_binary) || !has_dpus || !has_tasklets ||
        (require_nonempty_tasks && (!has_tasks || session->task_count == 0u)) ||
        session->requested_dpus != 1u || session->tasklets != 1u ||
        (interactive_request && (session->session_id == NULL || session->session_id[0] == '\0')) ||
        (interactive_request && session->task_count != 1u && session->task_count != 4u)) goto parse_failed;
    if (require_binary) {
        session->dpu_binary_path = resolve_path(base, session->dpu_binary_ref);
        if (session->dpu_binary_path == NULL) goto parse_failed;
    }
    free(base);
    free(contents);
    return 0;

parse_failed:
    free(base);
    free(contents);
    set_error(error_message, "invalid upmem_generic_session_v1 manifest");
    upmem_generic_session_free(session);
    return 1;
}

int upmem_generic_session_load(
    const char *manifest_path,
    upmem_generic_session *session,
    char **error_message
) {
    return load_task_manifest(
        manifest_path, session, error_message,
        UPMEM_GENERIC_SESSION_SCHEMA,
        UPMEM_GENERIC_SESSION_INPUT_KIND,
        "session_id", 1, 1, 0
    );
}

int upmem_generic_interactive_request_load(
    const char *manifest_path,
    upmem_generic_session *request,
    char **error_message
) {
    return load_task_manifest(
        manifest_path, request, error_message,
        UPMEM_GENERIC_INTERACTIVE_SCHEMA,
        UPMEM_GENERIC_INTERACTIVE_REQUEST_KIND,
        "request_id", 0, 1, 1
    );
}

int upmem_generic_interactive_bootstrap_load(
    const char *manifest_path,
    upmem_generic_interactive_bootstrap *bootstrap,
    char **error_message
) {
    char *contents = NULL;
    char *base = NULL;
    char *session_id = NULL;
    char *binary_ref = NULL;
    size_t length = 0u;
    json_reader reader;
    uint32_t requested_dpus = 0u;
    uint32_t tasklets = 0u;
    int has_schema = 0;
    int has_kind = 0;
    int has_id = 0;
    int has_binary = 0;
    int has_dpus = 0;
    int has_tasklets = 0;

    memset(bootstrap, 0, sizeof(*bootstrap));
    if (read_manifest(manifest_path, &contents, &length) != 0) {
        set_error(error_message, "unable to read interactive bootstrap manifest");
        return 1;
    }
    base = manifest_base(manifest_path);
    if (base == NULL) goto bootstrap_failed;
    reader.cursor = contents;
    reader.end = contents + length;
    if (consume(&reader, '{') != 0) goto bootstrap_failed;
    for (;;) {
        char *key = NULL;
        if (parse_string(&reader, &key) != 0 || consume(&reader, ':') != 0) {
            free(key);
            goto bootstrap_failed;
        }
        if (strcmp(key, "schema_version") == 0) {
            char *value = NULL;
            if (parse_string(&reader, &value) != 0) { free(key); goto bootstrap_failed; }
            has_schema = strcmp(value, UPMEM_GENERIC_INTERACTIVE_SCHEMA) == 0;
            free(value);
        } else if (strcmp(key, "manifest_kind") == 0) {
            char *value = NULL;
            if (parse_string(&reader, &value) != 0) { free(key); goto bootstrap_failed; }
            has_kind = strcmp(value, UPMEM_GENERIC_INTERACTIVE_BOOTSTRAP_KIND) == 0;
            free(value);
        } else if (strcmp(key, "session_id") == 0) {
            free(session_id);
            if (parse_string(&reader, &session_id) != 0) { free(key); goto bootstrap_failed; }
            has_id = 1;
        } else if (strcmp(key, "dpu_binary") == 0) {
            free(binary_ref);
            if (parse_string(&reader, &binary_ref) != 0) { free(key); goto bootstrap_failed; }
            has_binary = 1;
        } else if (strcmp(key, "requested_dpus") == 0) {
            if (parse_uint(&reader, &requested_dpus) != 0) { free(key); goto bootstrap_failed; }
            has_dpus = 1;
        } else if (strcmp(key, "tasklets") == 0) {
            if (parse_uint(&reader, &tasklets) != 0) { free(key); goto bootstrap_failed; }
            has_tasklets = 1;
        } else if (skip_value(&reader) != 0) {
            free(key);
            goto bootstrap_failed;
        }
        free(key);
        skip_space(&reader);
        if (reader.cursor < reader.end && *reader.cursor == '}') {
            reader.cursor++;
            break;
        }
        if (consume(&reader, ',') != 0) goto bootstrap_failed;
    }
    skip_space(&reader);
    if (reader.cursor != reader.end || !has_schema || !has_kind || !has_id ||
        !has_binary || !has_dpus || !has_tasklets || session_id == NULL ||
        session_id[0] == '\0' || requested_dpus != 1u || tasklets != 1u) {
        goto bootstrap_failed;
    }
    bootstrap->dpu_binary_path = resolve_path(base, binary_ref);
    if (bootstrap->dpu_binary_path == NULL) goto bootstrap_failed;
    bootstrap->session_id = session_id;
    bootstrap->dpu_binary_ref = binary_ref;
    bootstrap->requested_dpus = requested_dpus;
    bootstrap->tasklets = tasklets;
    free(base);
    free(contents);
    return 0;

bootstrap_failed:
    free(session_id);
    free(binary_ref);
    free(base);
    free(contents);
    set_error(error_message, "invalid generic_loop_interactive_session_v1 bootstrap manifest");
    upmem_generic_interactive_bootstrap_free(bootstrap);
    return 1;
}

void upmem_generic_interactive_bootstrap_free(
    upmem_generic_interactive_bootstrap *bootstrap
) {
    if (bootstrap == NULL) return;
    free(bootstrap->session_id);
    free(bootstrap->dpu_binary_ref);
    free(bootstrap->dpu_binary_path);
    memset(bootstrap, 0, sizeof(*bootstrap));
}

void upmem_generic_session_free(upmem_generic_session *session) {
    if (session == NULL) return;
    free(session->session_id);
    free(session->dpu_binary_ref);
    free(session->dpu_binary_path);
    for (size_t i = 0; i < session->task_count; i++) {
        upmem_generic_session_task *task = &session->tasks[i];
        free_task_fields(task);
    }
    free(session->tasks);
    memset(session, 0, sizeof(*session));
}

static void write_json_string(FILE *file, const char *value) {
    const unsigned char *cursor = (const unsigned char *)(value ? value : "");
    fputc('"', file);
    for (; *cursor; cursor++) {
        if (*cursor == '"' || *cursor == '\\') { fputc('\\', file); fputc(*cursor, file); }
        else if (*cursor == '\n') fputs("\\n", file);
        else if (*cursor == '\r') fputs("\\r", file);
        else if (*cursor == '\t') fputs("\\t", file);
        else fputc(*cursor, file);
    }
    fputc('"', file);
}

static const char *task_status(int status) {
    return status == UPMEM_GENERIC_SESSION_TASK_COMPLETED ? "completed" :
        status == UPMEM_GENERIC_SESSION_TASK_FAILED ? "failed" : "not_run";
}

static int write_response_for_protocol(
    const char *response_path,
    const upmem_generic_session *session,
    const char *status,
    const char *failure_stage,
    const char *error_message,
    uint32_t allocated_dpus,
    int sdk_error_code,
    double allocation_time_s,
    double binary_load_time_s,
    double work_time_s,
    double release_time_s,
    const char *schema,
    const char *kind,
    const char *id_field,
    const char *work_field
) {
    FILE *file = fopen(response_path, "wb");
    size_t completed = 0u;
    if (file == NULL) return 1;
    for (size_t i = 0; i < session->task_count; i++) {
        completed += (size_t)(
            session->tasks[i].result_status == UPMEM_GENERIC_SESSION_TASK_COMPLETED
        );
    }
    fprintf(
        file,
        "{\n  \"schema_version\": \"%s\",\n  \"manifest_kind\": \"%s\",\n"
        "  ",
        schema,
        kind
    );
    fprintf(file, "\"%s\": ", id_field);
    write_json_string(file, session->session_id ? session->session_id : "");
    fprintf(file, ",\n  \"status\": ");
    write_json_string(file, status);
    fprintf(file, ",\n  \"failure_stage\": ");
    if (failure_stage) write_json_string(file, failure_stage); else fputs("null", file);
    fprintf(file, ",\n  \"error\": ");
    if (error_message) write_json_string(file, error_message); else fputs("null", file);
    fprintf(
        file,
        ",\n  \"requested_dpus\": %u,\n  \"allocated_dpus\": %u,\n"
        "  \"tasklets\": %u,\n  \"task_count\": %zu,\n"
        "  \"completed_task_count\": %zu,\n  \"sdk_error_code\": %d,\n"
        "  \"allocation_profile\": %s,\n"
        "  \"allocation_time_s\": %.9f,\n  \"binary_load_time_s\": %.9f,\n"
        "  \"%s\": %.9f,\n  \"release_time_s\": %.9f,\n"
        "  \"tasks\": [\n",
        session->requested_dpus, allocated_dpus, session->tasklets,
        session->task_count, completed, sdk_error_code,
        SESSION_ALLOCATION_PROFILE_JSON, allocation_time_s,
        binary_load_time_s, work_field, work_time_s, release_time_s
    );
    for (size_t i = 0; i < session->task_count; i++) {
        const upmem_generic_session_task *task = &session->tasks[i];
        fprintf(file, "    {\n      \"sequence\": %zu,\n      \"task_id\": ", i);
        write_json_string(file, task->task_id);
        fprintf(file, ",\n      \"status\": \"%s\",\n      \"failure_stage\": ", task_status(task->result_status));
        if (task->failure_stage[0]) write_json_string(file, task->failure_stage);
        else fputs("null", file);
        fprintf(
            file,
            ",\n      \"sdk_error_code\": %d,\n      \"timing\": {\n"
            "        \"input_read_time_s\": %.9f,\n        \"h2d_time_s\": %.9f,\n"
            "        \"kernel_time_s\": %.9f,\n        \"d2h_time_s\": %.9f,\n"
            "        \"output_write_time_s\": %.9f,\n        \"total_time_s\": %.9f\n"
            "      },\n      \"output\": ",
            task->sdk_error_code, task->timing.input_read_time_s,
            task->timing.h2d_time_s, task->timing.kernel_time_s,
            task->timing.d2h_time_s, task->timing.output_write_time_s,
            task->timing.total_time_s
        );
        if (task->result_status == UPMEM_GENERIC_SESSION_TASK_COMPLETED) {
            fprintf(file, "{\"path\": ");
            write_json_string(file, task->output_ref);
            fprintf(file, ", \"bytes\": %zu}", task->output_bytes);
        } else {
            fputs("null", file);
        }
        fprintf(file, "\n    }%s\n", i + 1u == session->task_count ? "" : ",");
    }
    fputs("  ]\n}\n", file);
    {
        int failed = ferror(file);
        if (fclose(file) != 0) failed = 1;
        return failed;
    }
}

int upmem_generic_session_write_response(
    const char *response_path,
    const upmem_generic_session *session,
    const char *status,
    const char *failure_stage,
    const char *error_message,
    uint32_t allocated_dpus,
    int sdk_error_code,
    double allocation_time_s,
    double binary_load_time_s,
    double batch_time_s,
    double release_time_s
) {
    return write_response_for_protocol(
        response_path, session, status, failure_stage, error_message,
        allocated_dpus, sdk_error_code, allocation_time_s, binary_load_time_s,
        batch_time_s, release_time_s,
        UPMEM_GENERIC_SESSION_SCHEMA, UPMEM_GENERIC_SESSION_OUTPUT_KIND,
        "session_id", "batch_time_s"
    );
}

int upmem_generic_interactive_request_write_response(
    const char *response_path,
    const upmem_generic_session *request,
    const char *status,
    const char *failure_stage,
    const char *error_message,
    uint32_t allocated_dpus,
    int sdk_error_code,
    double allocation_time_s,
    double binary_load_time_s,
    double request_time_s,
    double release_time_s
) {
    return write_response_for_protocol(
        response_path, request, status, failure_stage, error_message,
        allocated_dpus, sdk_error_code, allocation_time_s, binary_load_time_s,
        request_time_s, release_time_s,
        UPMEM_GENERIC_INTERACTIVE_SCHEMA,
        UPMEM_GENERIC_INTERACTIVE_RESPONSE_KIND,
        "request_id", "request_time_s"
    );
}

int upmem_generic_session_write_error_response(
    const char *response_path,
    const char *failure_stage,
    const char *error_message
) {
    upmem_generic_session empty;
    memset(&empty, 0, sizeof(empty));
    empty.requested_dpus = 1u;
    empty.tasklets = 1u;
    return upmem_generic_session_write_response(
        response_path, &empty, "failed", failure_stage, error_message,
        0u, -1, 0.0, 0.0, 0.0, 0.0
    );
}

int upmem_generic_interactive_request_write_error_response(
    const char *response_path,
    const char *failure_stage,
    const char *error_message
) {
    upmem_generic_session empty;
    memset(&empty, 0, sizeof(empty));
    empty.requested_dpus = 1u;
    empty.tasklets = 1u;
    return upmem_generic_interactive_request_write_response(
        response_path, &empty, "failed", failure_stage, error_message,
        0u, -1, 0.0, 0.0, 0.0, 0.0
    );
}
