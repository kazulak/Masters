#define _POSIX_C_SOURCE 200809L

#include <dpu.h>

#include <errno.h>
#include <inttypes.h>
#include <limits.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "communication/CommOps.h"
#include "management/Management.h"
#include "processing/ProcessingHelperHost.h"
#include "processing/map/Map.h"
#include "processing/zip/Zip.h"
#include "processing/gen_red/GenRed.h"

#define M42_SCHEMA "simplepim_rank1_dot_m4_2_v1"
#define M42_PROFILE "hardware_simplepim_rank1_dot_m4_2_v1"
#define M42_BACKEND "upmem_sdk_hardware_simplepim_rank1_dot_m4_2"
#define M42_ROUTE "upmem_tn_hardware_simplepim_rank1_dot_m4_2"
#define M42_PROVIDER "simplepim"
#define M42_SOURCE_COMMIT "1d639c53532555f01e9f71d872e7712b166d6cba"
#define M42_ALLOCATION_PROFILE "backend=hw"
#define M42_DPU_COUNT 2u
#define M42_INITIALIZATION_TASKLETS 1u
#define M42_OPERATOR_TASKLETS 12u
#define M42_VECTOR_LENGTH 256u
#define M42_WARMUPS 1u
#define M42_REPEATS 5u
#define M42_ITERATIONS (M42_WARMUPS + M42_REPEATS)
#define M42_EXTERNAL_OPERAND_BYTES (2u * M42_VECTOR_LENGTH * sizeof(int8_t))
#define M42_TABLES_PER_ITERATION 5u
#define M42_EXPECTED_TABLE_COUNT (M42_ITERATIONS * M42_TABLES_PER_ITERATION)
#define M42_SCATTER_CALLS (2u * M42_ITERATIONS)
#define M42_LOGICAL_H2D_BYTES_PER_ITERATION (2u * M42_VECTOR_LENGTH * sizeof(int32_t))
#define M42_LOGICAL_D2H_BYTES_PER_ITERATION (M42_DPU_COUNT * sizeof(int64_t))
#define M42_LOGICAL_TRANSFER_BYTES_PER_ITERATION (M42_LOGICAL_H2D_BYTES_PER_ITERATION + M42_LOGICAL_D2H_BYTES_PER_ITERATION)
#define M42_LOGICAL_H2D_BYTES_TOTAL_SESSION (M42_ITERATIONS * M42_LOGICAL_H2D_BYTES_PER_ITERATION)
#define M42_LOGICAL_D2H_BYTES_TOTAL_SESSION (M42_ITERATIONS * M42_LOGICAL_D2H_BYTES_PER_ITERATION)
#define M42_LOGICAL_TRANSFER_BYTES_TOTAL_SESSION (M42_ITERATIONS * M42_LOGICAL_TRANSFER_BYTES_PER_ITERATION)

typedef struct {
    uint32_t repeat_id;
    bool warmup;
    char input_hash[17];
    char output_hash[17];
    int64_t reference;
    int64_t result;
    bool exact_match;
    double scatter_s;
    double zip_s;
    double map_s;
    double reduce_s;
    double total_s;
} repeat_record_t;

typedef struct {
    char source_commit[80];
    char staged_source_tree_sha256[80];
    char staged_overlay_tree_sha256[80];
    char patch_sha256[80];
    char stage_manifest_hash[17];
    bool valid;
} provenance_t;

typedef struct {
    const char *status;
    const char *failure_stage;
    const char *reason;
    const char *target_observed;
    int allocated_dpus;
    bool allocation_attempted;
    bool release_attempted;
    bool release_confirmed;
    bool provider_initialized;
    bool operator_api_used;
    bool scatter_attempted;
    bool scatter_completed;
    bool zip_attempted;
    bool zip_completed;
    bool map_attempted;
    bool map_completed;
    bool reduction_attempted;
    bool reduction_completed;
    uint32_t scatter_attempt_count;
    uint32_t scatter_completed_count;
    uint32_t zip_attempt_count;
    uint32_t zip_completed_count;
    uint32_t map_attempt_count;
    uint32_t map_completed_count;
    uint32_t reduction_attempt_count;
    uint32_t reduction_completed_count;
    bool operator_validations_passed;
    bool operator_metadata_checks_passed;
    bool virtual_zip_used;
    bool map_kernel_executed;
    bool reduction_kernel_executed;
    bool host_mediated_reduction;
    bool persistent_allocation_observed;
    bool table_metrics_observed;
    bool bounded_table_growth;
    uint32_t observed_table_count;
    uint32_t mram_high_water_bytes_per_dpu;
    uint32_t mram_conservative_bound_bytes_per_dpu;
    bool all_tasks_completed;
    bool exact_validation;
    bool hardware_execution;
    bool hardware_functionality_evidence;
    bool external_operand_transport;
    uint32_t operand_input_length_bytes;
    char operand_input_hash[17];
    const char *validation_status;
    repeat_record_t records[M42_WARMUPS + M42_REPEATS];
    size_t record_count;
} response_state_t;

static double now_s(void) {
    struct timespec value;
    if (clock_gettime(CLOCK_MONOTONIC, &value) != 0) return 0.0;
    return (double)value.tv_sec + (double)value.tv_nsec / 1000000000.0;
}

static uint64_t fnv1a_update(uint64_t hash, const void *data, size_t length) {
    const unsigned char *bytes = (const unsigned char *)data;
    for (size_t i = 0; i < length; ++i) {
        hash ^= bytes[i];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

static uint64_t fnv1a_bytes(const void *data, size_t length) {
    return fnv1a_update(UINT64_C(14695981039346656037), data, length);
}

static void hash_hex(uint64_t value, char output[17]) {
    (void)snprintf(output, 17, "%016" PRIx64, value);
}

static uint64_t hash_int32_values_le(uint64_t hash, const int32_t *values, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        uint32_t bits = (uint32_t)values[i];
        unsigned char encoded[4];
        for (size_t byte = 0; byte < sizeof(encoded); ++byte) {
            encoded[byte] = (unsigned char)(bits >> (8u * byte));
        }
        hash = fnv1a_update(hash, encoded, sizeof(encoded));
    }
    return hash;
}

static uint64_t hash_int64_value_le(int64_t value) {
    uint64_t bits = (uint64_t)value;
    unsigned char encoded[8];
    for (size_t byte = 0; byte < sizeof(encoded); ++byte) {
        encoded[byte] = (unsigned char)(bits >> (8u * byte));
    }
    return fnv1a_bytes(encoded, sizeof(encoded));
}

static uint64_t hash_file(const char *path, bool *ok) {
    unsigned char buffer[4096];
    size_t count;
    uint64_t hash = UINT64_C(14695981039346656037);
    FILE *file = fopen(path, "rb");
    if (file == NULL) {
        *ok = false;
        return 0;
    }
    while ((count = fread(buffer, 1, sizeof(buffer), file)) != 0) {
        hash = fnv1a_update(hash, buffer, count);
    }
    {
        int read_error = ferror(file);
        int close_error = fclose(file);
        *ok = read_error == 0 && close_error == 0;
    }
    return hash;
}

static bool read_text(const char *path, char **text) {
    long size;
    FILE *file = fopen(path, "rb");
    if (file == NULL || fseek(file, 0, SEEK_END) != 0) {
        if (file != NULL) (void)fclose(file);
        return false;
    }
    size = ftell(file);
    if (size < 0 || fseek(file, 0, SEEK_SET) != 0) {
        (void)fclose(file);
        return false;
    }
    *text = (char *)malloc((size_t)size + 1u);
    if (*text == NULL || fread(*text, 1, (size_t)size, file) != (size_t)size) {
        free(*text);
        *text = NULL;
        (void)fclose(file);
        return false;
    }
    (*text)[size] = '\0';
    return fclose(file) == 0;
}

static bool json_field(const char *text, const char *field, char *output, size_t output_size) {
    char needle[128];
    const char *start;
    const char *end;
    size_t length;
    int written = snprintf(needle, sizeof(needle), "\"%s\":\"", field);
    if (written < 0 || (size_t)written >= sizeof(needle)) return false;
    start = strstr(text, needle);
    if (start == NULL) return false;
    start += strlen(needle);
    end = strchr(start, '"');
    if (end == NULL) return false;
    length = (size_t)(end - start);
    if (length + 1u > output_size) return false;
    memcpy(output, start, length);
    output[length] = '\0';
    return true;
}

static bool load_provenance(const char *manifest_path, provenance_t *provenance) {
    char *text = NULL;
    char expected_source_commit[80] = "";
    bool ok;
    bool hash_ok;
    uint64_t manifest_hash;
    memset(provenance, 0, sizeof(*provenance));
    if (!read_text(manifest_path, &text)) return false;
    manifest_hash = hash_file(manifest_path, &hash_ok);
    if (!hash_ok) {
        free(text);
        return false;
    }
    hash_hex(manifest_hash, provenance->stage_manifest_hash);
    ok = json_field(text, "source_commit", provenance->source_commit, sizeof(provenance->source_commit)) &&
         json_field(text, "expected_source_commit", expected_source_commit, sizeof(expected_source_commit)) &&
         json_field(text, "staged_source_tree_sha256", provenance->staged_source_tree_sha256, sizeof(provenance->staged_source_tree_sha256)) &&
         json_field(text, "staged_overlay_tree_sha256", provenance->staged_overlay_tree_sha256, sizeof(provenance->staged_overlay_tree_sha256)) &&
         json_field(text, "patch_sha256", provenance->patch_sha256, sizeof(provenance->patch_sha256)) &&
         strstr(text, "\"patch_applied\":true") != NULL &&
         strstr(text, "\"source_worktree_dirty\":false") != NULL &&
         strcmp(provenance->source_commit, M42_SOURCE_COMMIT) == 0 &&
         strcmp(expected_source_commit, M42_SOURCE_COMMIT) == 0;
    provenance->valid = ok;
    free(text);
    return ok;
}

static void json_string(FILE *file, const char *value) {
    fputc('"', file);
    if (value != NULL) {
        for (const unsigned char *cursor = (const unsigned char *)value; *cursor != '\0'; ++cursor) {
            if (*cursor == '\\' || *cursor == '"') fputc('\\', file);
            if (*cursor == '\n') {
                fputs("\\n", file);
            } else if (*cursor == '\r') {
                fputs("\\r", file);
            } else if (*cursor == '\t') {
                fputs("\\t", file);
            } else {
                fputc(*cursor, file);
            }
        }
    }
    fputc('"', file);
}

static void json_nullable_string(FILE *file, const char *value) {
    if (value == NULL) fputs("null", file);
    else json_string(file, value);
}

static void json_bool(FILE *file, bool value) {
    fputs(value ? "true" : "false", file);
}

static int write_response(
    const char *path,
    const response_state_t *state,
    const provenance_t *provenance,
    const char *hostname,
    const char *host_hash,
    const char *init_hash,
    const char *map_hash,
    const char *genred_hash,
    const char *reduce_so_hash,
    double allocation_s,
    double handle_compile_s,
    double release_s,
    bool parser_mode
) {
    FILE *file = fopen(path, "w");
    if (file == NULL) return 1;
    fprintf(file, "{\"schema_version\":"); json_string(file, M42_SCHEMA);
    fprintf(file, ",\"profile_id\":"); json_string(file, M42_PROFILE);
    fprintf(file, ",\"backend_id\":"); json_string(file, M42_BACKEND);
    fprintf(file, ",\"route_id\":"); json_string(file, M42_ROUTE);
    fprintf(file, ",\"provider_id\":"); json_string(file, M42_PROVIDER);
    fprintf(file, ",\"execution_class\":\"rank1_contraction\",\"kernel_strategy\":\"virtual_zip_map_int64_genred\",\"allocation_profile\":\"%s\"", M42_ALLOCATION_PROFILE);
    fprintf(file, ",\"target_requested\":\"physical_hardware\",\"target_observed\":"); json_string(file, state->target_observed);
    fprintf(file, ",\"requested_dpu_count\":%u,\"allocated_dpu_count\":", M42_DPU_COUNT);
    if (state->allocated_dpus < 0) fputs("null", file); else fprintf(file, "%d", state->allocated_dpus);
    fprintf(file, ",\"initialization_tasklets_per_dpu\":%u,\"operator_tasklets_per_dpu\":%u", M42_INITIALIZATION_TASKLETS, M42_OPERATOR_TASKLETS);
    fprintf(file, ",\"allocation_count\":%u,\"persistent_allocation_requested\":true,\"persistent_allocation_observed\":", state->allocation_attempted ? 1u : 0u);
    json_bool(file, state->persistent_allocation_observed);
    fprintf(file, ",\"warmup_count\":%u,\"repeat_count\":%u", M42_WARMUPS, M42_REPEATS);
    fprintf(file, ",\"operator_sequence\":[\"simplepim_scatter\",\"simplepim_scatter\",\"table_zip_virtual\",\"table_map_pair_product\",\"table_gen_red_host_reduce\"]");
    fprintf(file, ",\"external_operand_transport\":"); json_bool(file, state->external_operand_transport);
    fprintf(file, ",\"operand_input_length_bytes\":%u,\"operand_input_hash\":", state->operand_input_length_bytes);
    if (state->external_operand_transport) json_string(file, state->operand_input_hash); else fputs("null", file);
    fprintf(file, ",\"simplepim_operator_api_used\":"); json_bool(file, state->operator_api_used);
    fprintf(file, ",\"simplepim_operator_names\":[\"simplepim_scatter\",\"table_zip\",\"table_map\",\"table_gen_red\"]");
    fprintf(file, ",\"scatter_attempted\":"); json_bool(file, state->scatter_attempted);
    fprintf(file, ",\"scatter_completed\":"); json_bool(file, state->scatter_completed);
    fprintf(file, ",\"scatter_attempt_count\":%u,\"scatter_completed_count\":%u", state->scatter_attempt_count, state->scatter_completed_count);
    fprintf(file, ",\"zip_attempted\":"); json_bool(file, state->zip_attempted);
    fprintf(file, ",\"zip_completed\":"); json_bool(file, state->zip_completed);
    fprintf(file, ",\"zip_attempt_count\":%u,\"zip_completed_count\":%u", state->zip_attempt_count, state->zip_completed_count);
    fprintf(file, ",\"map_attempted\":"); json_bool(file, state->map_attempted);
    fprintf(file, ",\"map_completed\":"); json_bool(file, state->map_completed);
    fprintf(file, ",\"map_attempt_count\":%u,\"map_completed_count\":%u", state->map_attempt_count, state->map_completed_count);
    fprintf(file, ",\"genred_attempted\":"); json_bool(file, state->reduction_attempted);
    fprintf(file, ",\"genred_completed\":"); json_bool(file, state->reduction_completed);
    fprintf(file, ",\"genred_attempt_count\":%u,\"genred_completed_count\":%u", state->reduction_attempt_count, state->reduction_completed_count);
    fprintf(file, ",\"operator_validations_passed\":"); json_bool(file, state->operator_validations_passed);
    fprintf(file, ",\"operator_metadata_checks_passed\":"); json_bool(file, state->operator_metadata_checks_passed);
    fprintf(file, ",\"virtual_zip\":"); json_bool(file, state->virtual_zip_used);
    fprintf(file, ",\"map_kernel_executed\":"); json_bool(file, state->map_kernel_executed);
    fprintf(file, ",\"genred_kernel_executed\":"); json_bool(file, state->reduction_kernel_executed);
    fprintf(file, ",\"host_mediated_reduction\":"); json_bool(file, state->host_mediated_reduction);
    fprintf(file, ",\"allreduce_called\":false,\"gather_called\":false,\"final_gather_called\":false");
    fprintf(file, ",\"table_reuse\":false,\"bounded_table_growth\":"); json_bool(file, state->bounded_table_growth);
    fprintf(file, ",\"expected_table_count_session\":%u,\"observed_table_count\":", M42_EXPECTED_TABLE_COUNT);
    if (state->table_metrics_observed) fprintf(file, "%u", state->observed_table_count); else fputs("null", file);
    fprintf(file, ",\"mram_layout_bound_bytes_per_dpu\":%u,\"mram_conservative_bound_bytes_per_dpu\":%u,\"mram_high_water_bytes_per_dpu\":", state->mram_conservative_bound_bytes_per_dpu, state->mram_conservative_bound_bytes_per_dpu);
    if (state->table_metrics_observed) fprintf(file, "%u", state->mram_high_water_bytes_per_dpu); else fputs("null", file);
    fprintf(file, ",\"mram_capacity_verified\":false,\"cpu_fallback_used\":false,\"simulator_kernel_executed\":false,\"hardware_kernel_executed\":"); json_bool(file, state->hardware_execution);
    fprintf(file, ",\"thesis_direct_raw_sdk_allocation_used\":false,\"simplepim_managed_allocation\":");
    json_bool(file, state->provider_initialized && state->allocated_dpus == (int)M42_DPU_COUNT && strcmp(state->target_observed, "physical_hardware") == 0);
    fprintf(file, ",\"provider_initialized\":"); json_bool(file, state->provider_initialized);
    fprintf(file, ",\"allocation_attempted\":"); json_bool(file, state->allocation_attempted);
    fprintf(file, ",\"release_attempted\":"); json_bool(file, state->release_attempted);
    fprintf(file, ",\"release_confirmed\":"); json_bool(file, state->release_confirmed);
    fprintf(file, ",\"all_tasks_completed\":"); json_bool(file, state->all_tasks_completed);
    fprintf(file, ",\"validation_status\":"); json_string(file, state->validation_status);
    fprintf(file, ",\"exact_integer_match\":"); json_bool(file, state->exact_validation);
    fprintf(file, ",\"allocation_time_s\":%.9f,\"handle_compile_time_s\":%.9f,\"release_time_s\":%.9f", allocation_s, handle_compile_s, release_s);
    {
        double total_route_s = allocation_s + handle_compile_s + release_s;
        for (size_t i = 0; i < state->record_count; ++i) total_route_s += state->records[i].total_s;
        fprintf(file, ",\"total_route_time_s\":%.9f", total_route_s);
    }
    fprintf(file, ",\"session_iteration_count\":%u", M42_ITERATIONS);
    fprintf(file, ",\"logical_payload_h2d_bytes_per_iteration\":%zu,\"logical_payload_d2h_bytes_per_iteration\":%zu,\"logical_payload_transfer_bytes_per_iteration\":%zu", (size_t)M42_LOGICAL_H2D_BYTES_PER_ITERATION, (size_t)M42_LOGICAL_D2H_BYTES_PER_ITERATION, (size_t)M42_LOGICAL_TRANSFER_BYTES_PER_ITERATION);
    fprintf(file, ",\"logical_payload_h2d_bytes_total_session\":%zu,\"logical_payload_d2h_bytes_total_session\":%zu,\"logical_payload_transfer_bytes_total_session\":%zu", (size_t)M42_LOGICAL_H2D_BYTES_TOTAL_SESSION, (size_t)M42_LOGICAL_D2H_BYTES_TOTAL_SESSION, (size_t)M42_LOGICAL_TRANSFER_BYTES_TOTAL_SESSION);
    fprintf(file, ",\"transfer_bytes_scope\":\"application-visible logical operand payloads and per-DPU scalar partials across warmup plus measured iterations; SDK argument, control, alignment, and runtime-internal transfers excluded\"");
    fprintf(file, ",\"hardware_target_observation_method\":\"explicit_backend_hw_request_and_observed_dpu_count\"");
    fprintf(file, ",\"timing_scope\":\"physical hardware SimplePIM qualification; operator timings include API orchestration\",\"timing_is_bringup_only\":true,\"hardware_speedup_applicable\":false,\"hardware_functionality_evidence\":"); json_bool(file, state->hardware_functionality_evidence);
    fprintf(file, ",\"source_commit\":"); json_nullable_string(file, provenance->valid ? provenance->source_commit : NULL);
    fprintf(file, ",\"staged_source_tree_sha256\":"); json_nullable_string(file, provenance->valid ? provenance->staged_source_tree_sha256 : NULL);
    fprintf(file, ",\"staged_overlay_tree_sha256\":"); json_nullable_string(file, provenance->valid ? provenance->staged_overlay_tree_sha256 : NULL);
    fprintf(file, ",\"patch_sha256\":"); json_nullable_string(file, provenance->valid ? provenance->patch_sha256 : NULL);
    fprintf(file, ",\"stage_manifest_hash\":"); json_nullable_string(file, provenance->valid ? provenance->stage_manifest_hash : NULL);
    fprintf(file, ",\"hash_algorithm\":\"fnv1a64_for_runtime_artifacts\",\"hostname\":"); json_string(file, hostname);
    fprintf(file, ",\"host_binary_hash\":"); json_nullable_string(file, host_hash);
    fprintf(file, ",\"initialization_binary_hash\":"); json_nullable_string(file, init_hash);
    fprintf(file, ",\"map_binary_hash\":"); json_nullable_string(file, map_hash);
    fprintf(file, ",\"genred_binary_hash\":"); json_nullable_string(file, genred_hash);
    fprintf(file, ",\"genred_reduce_shared_object_hash\":"); json_nullable_string(file, reduce_so_hash);
    fprintf(file, ",\"status\":"); json_string(file, state->status);
    fprintf(file, ",\"failure_stage\":"); json_nullable_string(file, state->failure_stage);
    fprintf(file, ",\"reason\":"); json_nullable_string(file, state->reason);
    fprintf(file, ",\"parser_mode\":"); json_bool(file, parser_mode);
    fputs(",\"repetitions\":[", file);
    for (size_t i = 0; i < state->record_count; ++i) {
        const repeat_record_t *record = &state->records[i];
        if (i != 0) fputc(',', file);
        fprintf(file, "{\"repeat_id\":%u,\"warmup\":", record->repeat_id); json_bool(file, record->warmup);
        fprintf(file, ",\"input_hash\":"); json_string(file, record->input_hash);
        fprintf(file, ",\"output_hash\":"); json_string(file, record->output_hash);
        fprintf(file, ",\"reference_int64\":%" PRId64 ",\"result_int64\":%" PRId64 ",\"exact_integer_match\":", record->reference, record->result); json_bool(file, record->exact_match);
        fprintf(file, ",\"scatter_time_s\":%.9f,\"virtual_zip_time_s\":%.9f,\"map_time_s\":%.9f,\"reduction_time_s\":%.9f,\"total_time_s\":%.9f}", record->scatter_s, record->zip_s, record->map_s, record->reduce_s, record->total_s);
    }
    fputs("]}\n", file);
    return fclose(file) == 0 ? 0 : 1;
}

static void fill_inputs(int32_t *a, int32_t *b, int64_t *reference) {
    int64_t value = 0;
    for (uint32_t i = 0; i < M42_VECTOR_LENGTH; ++i) {
        a[i] = (int32_t)(i % 17u) - 8;
        b[i] = (int32_t)((i * 3u) % 19u) - 9;
        value += (int64_t)a[i] * (int64_t)b[i];
    }
    *reference = value;
}

static const char *load_operand_file(
    const char *path,
    int32_t *a,
    int32_t *b,
    int64_t *reference,
    uint64_t *input_hash
) {
    unsigned char raw[M42_EXTERNAL_OPERAND_BYTES];
    FILE *file = fopen(path, "rb");
    if (file == NULL) return "operand_file_open_failed";
    size_t read_count = fread(raw, 1, sizeof(raw), file);
    if (read_count != sizeof(raw)) {
        const bool read_error = ferror(file) != 0;
        (void)fclose(file);
        return read_error ? "operand_file_read_failed" : "operand_file_short";
    }
    int trailing = fgetc(file);
    if (trailing != EOF || ferror(file) != 0) {
        (void)fclose(file);
        return trailing != EOF ? "operand_file_trailing_data" : "operand_file_read_failed";
    }
    if (fclose(file) != 0) return "operand_file_read_failed";

    *input_hash = fnv1a_bytes(raw, sizeof(raw));
    *reference = 0;
    for (uint32_t i = 0; i < M42_VECTOR_LENGTH; ++i) {
        a[i] = (int32_t)(int8_t)raw[i];
        b[i] = (int32_t)(int8_t)raw[M42_VECTOR_LENGTH + i];
        *reference += (int64_t)a[i] * (int64_t)b[i];
    }
    return NULL;
}

static const char *simulator_selector_reason(void) {
    if (getenv("DPU_BACKEND") != NULL) return "DPU_BACKEND_must_be_unset";
    if (getenv("DPU_PROFILE") != NULL) return "DPU_PROFILE_must_be_unset";
    if (getenv("SIMPLEPIM_BACKEND") != NULL) return "SIMPLEPIM_BACKEND_must_be_unset";
    if (getenv("UPMEM_BACKEND") != NULL) return "UPMEM_BACKEND_must_be_unset";
    if (getenv("UPMEM_MODE") != NULL) return "UPMEM_MODE_must_be_unset";
    if (getenv("UPMEM_TARGET") != NULL) return "UPMEM_TARGET_must_be_unset";
    if (getenv("UPMEM_PROFILE") != NULL) return "UPMEM_PROFILE_must_be_unset";
    if (getenv("UPMEM_PROFILE_BASE") != NULL) return "UPMEM_PROFILE_BASE_must_be_unset";
    return NULL;
}

static uint32_t simplepim_registered_end(uint32_t start, uint32_t payload_bytes) {
    uint32_t raw_end = start + payload_bytes;
    return raw_end + (8u - raw_end % 8u);
}

static uint32_t conservative_mram_bound_bytes_per_dpu(void) {
    uint32_t input_bytes = M42_VECTOR_LENGTH * (uint32_t)sizeof(int32_t);
    uint32_t input_pad = calculate_pad_len(M42_VECTOR_LENGTH, (uint32_t)sizeof(int32_t), M42_DPU_COUNT);
    uint32_t input_extent = (input_bytes + input_pad) / M42_DPU_COUNT;
    uint32_t output_elements_per_dpu = M42_VECTOR_LENGTH / M42_DPU_COUNT;
    uint32_t offset = 0;
    for (uint32_t iteration = 0; iteration < M42_ITERATIONS; ++iteration) {
        offset += input_extent;
        offset += input_extent;
        offset = simplepim_registered_end(offset, output_elements_per_dpu * (uint32_t)(2u * sizeof(int32_t)));
        offset = simplepim_registered_end(offset, output_elements_per_dpu * (uint32_t)sizeof(int64_t));
        offset = simplepim_registered_end(offset, (uint32_t)sizeof(int64_t));
    }
    return offset;
}

static bool table_metadata_matches(
    simplepim_management_t *management,
    const char *name,
    uint32_t expected_table_count,
    uint32_t expected_start,
    uint32_t expected_end,
    uint64_t expected_length,
    uint32_t expected_type_size,
    uint32_t expected_virtual,
    uint32_t expected_length_per_dpu
) {
    table_host_t *table;
    if (management == NULL || management->num_tables != expected_table_count || !contains_table(name, management)) return false;
    table = lookup_table(name, management);
    if (table == NULL || table->lens_each_dpu == NULL || table->start != expected_start || table->end != expected_end ||
        table->len != expected_length || table->table_type_size != expected_type_size ||
        table->is_virtual_zipped != expected_virtual || management->free_space_start_pos != expected_end) {
        return false;
    }
    for (uint32_t dpu = 0; dpu < M42_DPU_COUNT; ++dpu) {
        if (table->lens_each_dpu[dpu] != expected_length_per_dpu) return false;
    }
    return true;
}

static bool zip_metadata_matches(
    simplepim_management_t *management,
    const char *name,
    uint32_t expected_table_count,
    uint32_t expected_start,
    uint32_t expected_end,
    const table_host_t *left,
    const table_host_t *right
) {
    table_host_t *table;
    if (!table_metadata_matches(
            management,
            name,
            expected_table_count,
            expected_start,
            expected_end,
            M42_VECTOR_LENGTH,
            (uint32_t)(2u * sizeof(int32_t)),
            1u,
            M42_VECTOR_LENGTH / M42_DPU_COUNT)) {
        return false;
    }
    table = lookup_table(name, management);
    return table != NULL && left != NULL && right != NULL &&
           table->start1 == left->start && table->end1 == left->end && table->type1 == left->table_type_size &&
           table->start2 == right->start && table->end2 == right->end && table->type2 == right->table_type_size;
}

static void observe_management(response_state_t *state, const simplepim_management_t *management) {
    if (management == NULL) return;
    state->table_metrics_observed = true;
    state->observed_table_count = management->num_tables;
    state->mram_high_water_bytes_per_dpu = management->free_space_start_pos;
}

static void finalise_execution_state(response_state_t *state, const simplepim_management_t *management) {
    bool all_exact = state->record_count == M42_ITERATIONS;
    observe_management(state, management);
    for (size_t i = 0; i < state->record_count; ++i) all_exact = all_exact && state->records[i].exact_match;
    state->scatter_attempted = state->scatter_attempt_count != 0u;
    state->scatter_completed = state->scatter_completed_count == M42_SCATTER_CALLS;
    state->zip_attempted = state->zip_attempt_count != 0u;
    state->zip_completed = state->zip_completed_count == M42_ITERATIONS;
    state->map_attempted = state->map_attempt_count != 0u;
    state->map_completed = state->map_completed_count == M42_ITERATIONS;
    state->reduction_attempted = state->reduction_attempt_count != 0u;
    state->reduction_completed = state->reduction_completed_count == M42_ITERATIONS;
    state->operator_validations_passed = state->scatter_completed && state->zip_completed &&
                                         state->map_completed && state->reduction_completed;
    state->operator_metadata_checks_passed = state->operator_validations_passed;
    state->all_tasks_completed = state->record_count == M42_ITERATIONS;
    state->exact_validation = all_exact;
    state->virtual_zip_used = state->zip_completed;
    state->map_kernel_executed = state->map_completed;
    state->reduction_kernel_executed = state->reduction_completed;
    state->host_mediated_reduction = state->reduction_completed;
    state->hardware_execution = state->map_completed && state->reduction_completed;
    state->bounded_table_growth = state->all_tasks_completed && state->table_metrics_observed &&
                                  state->observed_table_count == M42_EXPECTED_TABLE_COUNT &&
                                  state->mram_high_water_bytes_per_dpu <= state->mram_conservative_bound_bytes_per_dpu;
    state->persistent_allocation_observed = state->all_tasks_completed && state->allocation_attempted &&
                                            state->allocated_dpus == (int)M42_DPU_COUNT &&
                                            strcmp(state->target_observed, "physical_hardware") == 0;
    state->operator_api_used = state->operator_validations_passed && state->exact_validation;
    if (state->all_tasks_completed) state->validation_status = state->exact_validation ? "passed" : "failed";
}

static void free_handle(handle_t *handle) {
    if (handle == NULL) return;
    free(handle->bin_location);
    free(handle->so_bin_location);
    free(handle);
}

static void release_management_metadata(simplepim_management_t *management) {
    if (management == NULL) return;
    for (uint32_t i = 0; i < management->num_tables; ++i) {
        table_host_t *table = management->tables[i];
        if (table == NULL) continue;
        if (table->name != NULL && table->name[0] != '\0') free(table->name);
        free(table->lens_each_dpu);
        free(table);
    }
    free(management->tables);
    free(management->zip_args);
    free(management->map_args);
    free(management->red_args);
    free(management);
}

static bool existing_handle_artifacts(const handle_t *map, const handle_t *red) {
    return map != NULL && red != NULL && map->bin_location != NULL && red->bin_location != NULL &&
           red->so_bin_location != NULL && access(map->bin_location, R_OK) == 0 &&
           access(red->bin_location, R_OK) == 0 && access(red->so_bin_location, R_OK) == 0;
}

static void binary_hash_or_null(const char *path, char output[17], const char **value) {
    bool ok = false;
    uint64_t hash = hash_file(path, &ok);
    if (!ok) {
        *value = NULL;
        output[0] = '\0';
        return;
    }
    hash_hex(hash, output);
    *value = output;
}

static const char *resolve_host_path(const char *argv0, char output[PATH_MAX]) {
    if (realpath(argv0, output) != NULL) return output;
    if (strlen(argv0) + 1u <= PATH_MAX) {
        strcpy(output, argv0);
        return output;
    }
    return NULL;
}

static void initialise_state(response_state_t *state) {
    memset(state, 0, sizeof(*state));
    state->status = "failed";
    state->failure_stage = "host_setup";
    state->reason = "not_started";
    state->target_observed = "not_executed";
    state->allocated_dpus = -1;
    state->mram_conservative_bound_bytes_per_dpu = conservative_mram_bound_bytes_per_dpu();
    state->validation_status = "not_run";
}

static int parse_arguments(
    int argc,
    char **argv,
    const char **mode,
    const char **response,
    const char **stage_manifest,
    const char **operands_file
) {
    *mode = NULL;
    *response = "build/simplepim_rank1_dot_m4_2/response.json";
    *stage_manifest = NULL;
    *operands_file = NULL;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--mode") == 0 && i + 1 < argc) *mode = argv[++i];
        else if (strcmp(argv[i], "--response") == 0 && i + 1 < argc) *response = argv[++i];
        else if (strcmp(argv[i], "--stage-manifest") == 0 && i + 1 < argc) *stage_manifest = argv[++i];
        else if (strcmp(argv[i], "--operands-file") == 0 && i + 1 < argc) *operands_file = argv[++i];
        else return 1;
    }
    return *mode == NULL || *stage_manifest == NULL;
}

int main(int argc, char **argv) {
    const char *mode;
    const char *response_path;
    const char *stage_manifest_path;
    const char *operands_file_path;
    response_state_t state;
    provenance_t provenance;
    char hostname[256] = "unknown";
    char host_path[PATH_MAX] = "";
    char host_hash_buffer[17];
    char init_hash_buffer[17];
    char map_hash_buffer[17];
    char genred_hash_buffer[17];
    char reduce_so_hash_buffer[17];
    const char *host_hash = NULL;
    const char *init_hash = NULL;
    const char *map_hash = NULL;
    const char *genred_hash = NULL;
    const char *reduce_so_hash = NULL;
    double allocation_s = 0.0;
    double handle_compile_s = 0.0;
    double release_s = 0.0;
    bool parser_mode = false;
    int32_t external_values_a[M42_VECTOR_LENGTH];
    int32_t external_values_b[M42_VECTOR_LENGTH];
    int64_t external_reference = 0;
    uint64_t external_input_hash = 0;
    simplepim_management_t *management = NULL;
    handle_t *map_handle = NULL;
    handle_t *red_handle = NULL;
    handle_t zip_handle = {NULL, NULL, ZIP};
    bool dpu_released = false;
    int exit_code = 2;

    initialise_state(&state);
    memset(&provenance, 0, sizeof(provenance));
    (void)gethostname(hostname, sizeof(hostname) - 1u);
    if (parse_arguments(argc, argv, &mode, &response_path, &stage_manifest_path, &operands_file_path) != 0) {
        state.failure_stage = "arguments";
        state.reason = "expected_mode_response_and_stage_manifest";
        (void)write_response(response_path, &state, &provenance, hostname, NULL, NULL, NULL, NULL, NULL, 0.0, 0.0, 0.0, false);
        return exit_code;
    }
    parser_mode = strcmp(mode, "parser") == 0;
    if (!parser_mode && strcmp(mode, "execute") != 0) {
        state.failure_stage = "arguments";
        state.reason = "mode_must_be_parser_or_execute";
        (void)write_response(response_path, &state, &provenance, hostname, NULL, NULL, NULL, NULL, NULL, 0.0, 0.0, 0.0, false);
        return exit_code;
    }
    if (!load_provenance(stage_manifest_path, &provenance)) {
        state.failure_stage = "staging";
        state.reason = "invalid_simplepim_stage_manifest";
        (void)write_response(response_path, &state, &provenance, hostname, NULL, NULL, NULL, NULL, NULL, 0.0, 0.0, 0.0, parser_mode);
        return exit_code;
    }
    if (operands_file_path != NULL) {
        const char *input_error = load_operand_file(
            operands_file_path,
            external_values_a,
            external_values_b,
            &external_reference,
            &external_input_hash);
        state.external_operand_transport = true;
        if (input_error != NULL) {
            state.failure_stage = "input";
            state.reason = input_error;
            (void)write_response(response_path, &state, &provenance, hostname, NULL, NULL, NULL, NULL, NULL, 0.0, 0.0, 0.0, parser_mode);
            return exit_code;
        }
        state.operand_input_length_bytes = M42_EXTERNAL_OPERAND_BYTES;
        hash_hex(external_input_hash, state.operand_input_hash);
    }
    if (parser_mode) {
        state.status = "prepared";
        state.failure_stage = NULL;
        state.reason = NULL;
        state.validation_status = "not_run";
        exit_code = write_response(response_path, &state, &provenance, hostname, NULL, NULL, NULL, NULL, NULL, 0.0, 0.0, 0.0, true) == 0 ? 0 : 2;
        return exit_code;
    }
    if (getenv("UPMEM_ALLOW_PHYSICAL_HARDWARE") == NULL || strcmp(getenv("UPMEM_ALLOW_PHYSICAL_HARDWARE"), "1") != 0) {
        state.failure_stage = "opt_in";
        state.reason = "UPMEM_ALLOW_PHYSICAL_HARDWARE_must_equal_1";
        (void)write_response(response_path, &state, &provenance, hostname, NULL, NULL, NULL, NULL, NULL, 0.0, 0.0, 0.0, false);
        return exit_code;
    }
    {
        const char *selector_reason = simulator_selector_reason();
        if (selector_reason != NULL) {
            state.failure_stage = "hardware_profile";
            state.reason = selector_reason;
            (void)write_response(response_path, &state, &provenance, hostname, NULL, NULL, NULL, NULL, NULL, 0.0, 0.0, 0.0, false);
            return exit_code;
        }
    }
    if (resolve_host_path(argv[0], host_path) != NULL) binary_hash_or_null(host_path, host_hash_buffer, &host_hash);

    state.allocation_attempted = true;
    {
        double started = now_s();
        management = table_management_init(M42_DPU_COUNT);
        allocation_s = now_s() - started;
    }
    state.provider_initialized = management != NULL;
    if (management == NULL) {
        state.failure_stage = "allocation";
        state.reason = "simplepim_management_init_returned_null";
        goto finish;
    }
    {
        uint32_t observed = 0;
        if (dpu_get_nr_dpus(management->set, &observed) != DPU_OK) {
            state.failure_stage = "allocation";
            state.reason = "allocated_dpu_count_unavailable";
            goto finish;
        }
        state.allocated_dpus = (int)observed;
        if (observed != M42_DPU_COUNT) {
            state.failure_stage = "allocation";
            state.reason = "allocated_dpu_count_mismatch";
            goto finish;
        }
        state.target_observed = "physical_hardware";
    }
    {
        double started = now_s();
        map_handle = create_handle("dot_funcs", MAP);
        red_handle = create_handle("dot_reduce_funcs", REDUCE);
        handle_compile_s = now_s() - started;
    }
    if (!existing_handle_artifacts(map_handle, red_handle)) {
        state.failure_stage = "handle";
        state.reason = "operator_handle_build_failed";
        goto finish;
    }
    state.failure_stage = NULL;
    state.reason = NULL;
    state.validation_status = "not_run";
    binary_hash_or_null("bin/dpu_init_binary", init_hash_buffer, &init_hash);
    binary_hash_or_null(map_handle->bin_location, map_hash_buffer, &map_hash);
    binary_hash_or_null(red_handle->bin_location, genred_hash_buffer, &genred_hash);
    binary_hash_or_null(red_handle->so_bin_location, reduce_so_hash_buffer, &reduce_so_hash);
    if (host_hash == NULL || init_hash == NULL || map_hash == NULL || genred_hash == NULL || reduce_so_hash == NULL) {
        state.failure_stage = "handle";
        state.reason = "operator_artifact_hash_failed";
        goto finish;
    }

    for (uint32_t iteration = 0; iteration < M42_ITERATIONS; ++iteration) {
        int32_t values_a[M42_VECTOR_LENGTH];
        int32_t values_b[M42_VECTOR_LENGTH];
        int64_t reference;
        int64_t *result = NULL;
        void *table_a;
        void *table_b;
        table_host_t *table_a_metadata;
        table_host_t *table_b_metadata;
        repeat_record_t *record = &state.records[state.record_count];
        char name_a[32];
        char name_b[32];
        char name_zip[32];
        char name_products[32];
        char name_partial[32];
        double started;
        uint32_t table_count_before;
        uint32_t start_before;
        uint32_t expected_end;
        uint32_t input_pad = calculate_pad_len(M42_VECTOR_LENGTH, (uint32_t)sizeof(int32_t), M42_DPU_COUNT);
        uint32_t input_extent = (M42_VECTOR_LENGTH * (uint32_t)sizeof(int32_t) + input_pad) / M42_DPU_COUNT;
        uint64_t input_hash = UINT64_C(14695981039346656037);
        if (operands_file_path != NULL) {
            memcpy(values_a, external_values_a, sizeof(values_a));
            memcpy(values_b, external_values_b, sizeof(values_b));
            reference = external_reference;
        } else {
            fill_inputs(values_a, values_b, &reference);
        }
        memset(record, 0, sizeof(*record));
        record->repeat_id = iteration == 0 ? 0u : iteration - 1u;
        record->warmup = iteration == 0;
        record->reference = reference;
        if (operands_file_path != NULL) {
            input_hash = external_input_hash;
        } else {
            input_hash = hash_int32_values_le(input_hash, values_a, M42_VECTOR_LENGTH);
            input_hash = hash_int32_values_le(input_hash, values_b, M42_VECTOR_LENGTH);
        }
        hash_hex(input_hash, record->input_hash);
        (void)snprintf(name_a, sizeof(name_a), "m42_r%u_a", iteration);
        (void)snprintf(name_b, sizeof(name_b), "m42_r%u_b", iteration);
        (void)snprintf(name_zip, sizeof(name_zip), "m42_r%u_zip", iteration);
        (void)snprintf(name_products, sizeof(name_products), "m42_r%u_products", iteration);
        (void)snprintf(name_partial, sizeof(name_partial), "m42_r%u_partial", iteration);
        started = now_s();
        table_a = malloc_scatter_aligned(M42_VECTOR_LENGTH, sizeof(int32_t), management);
        table_b = malloc_scatter_aligned(M42_VECTOR_LENGTH, sizeof(int32_t), management);
        if (table_a == NULL || table_b == NULL) {
            free(table_a); free(table_b);
            state.failure_stage = "scatter";
            state.reason = "aligned_input_allocation_failed";
            goto finish;
        }
        memcpy(table_a, values_a, sizeof(values_a));
        memcpy(table_b, values_b, sizeof(values_b));
        table_count_before = management->num_tables;
        start_before = management->free_space_start_pos;
        expected_end = start_before + input_extent;
        state.scatter_attempt_count += 1u;
        simplepim_scatter(name_a, table_a, M42_VECTOR_LENGTH, sizeof(int32_t), management);
        free(table_a);
        table_a = NULL;
        if (!table_metadata_matches(
                management,
                name_a,
                table_count_before + 1u,
                start_before,
                expected_end,
                M42_VECTOR_LENGTH,
                (uint32_t)sizeof(int32_t),
                0u,
                M42_VECTOR_LENGTH / M42_DPU_COUNT)) {
            free(table_b);
            state.failure_stage = "scatter";
            state.reason = "scatter_a_table_metadata_invalid";
            goto finish;
        }
        state.scatter_completed_count += 1u;
        table_count_before = management->num_tables;
        start_before = management->free_space_start_pos;
        expected_end = start_before + input_extent;
        state.scatter_attempt_count += 1u;
        simplepim_scatter(name_b, table_b, M42_VECTOR_LENGTH, sizeof(int32_t), management);
        record->scatter_s = now_s() - started;
        free(table_b);
        table_b = NULL;
        if (!table_metadata_matches(
                management,
                name_b,
                table_count_before + 1u,
                start_before,
                expected_end,
                M42_VECTOR_LENGTH,
                (uint32_t)sizeof(int32_t),
                0u,
                M42_VECTOR_LENGTH / M42_DPU_COUNT)) {
            state.failure_stage = "scatter";
            state.reason = "scatter_b_table_metadata_invalid";
            goto finish;
        }
        state.scatter_completed_count += 1u;
        table_a_metadata = lookup_table(name_a, management);
        table_b_metadata = lookup_table(name_b, management);

        table_count_before = management->num_tables;
        start_before = management->free_space_start_pos;
        expected_end = simplepim_registered_end(
            start_before,
            (M42_VECTOR_LENGTH / M42_DPU_COUNT) * (uint32_t)(2u * sizeof(int32_t)));
        started = now_s();
        state.zip_attempt_count += 1u;
        table_zip(name_a, name_b, name_zip, &zip_handle, management);
        record->zip_s = now_s() - started;
        if (!zip_metadata_matches(
                management,
                name_zip,
                table_count_before + 1u,
                start_before,
                expected_end,
                table_a_metadata,
                table_b_metadata)) {
            state.failure_stage = "zip";
            state.reason = "virtual_zip_table_metadata_invalid";
            goto finish;
        }
        state.zip_completed_count += 1u;

        table_count_before = management->num_tables;
        start_before = management->free_space_start_pos;
        expected_end = simplepim_registered_end(
            start_before,
            (M42_VECTOR_LENGTH / M42_DPU_COUNT) * (uint32_t)sizeof(int64_t));
        started = now_s();
        state.map_attempt_count += 1u;
        table_map(name_zip, name_products, sizeof(int64_t), map_handle, management, 0u);
        record->map_s = now_s() - started;
        if (!table_metadata_matches(
                management,
                name_products,
                table_count_before + 1u,
                start_before,
                expected_end,
                M42_VECTOR_LENGTH,
                (uint32_t)sizeof(int64_t),
                0u,
                M42_VECTOR_LENGTH / M42_DPU_COUNT)) {
            state.failure_stage = "map";
            state.reason = "map_output_table_metadata_invalid";
            goto finish;
        }
        state.map_completed_count += 1u;

        table_count_before = management->num_tables;
        start_before = management->free_space_start_pos;
        expected_end = simplepim_registered_end(start_before, (uint32_t)sizeof(int64_t));
        started = now_s();
        state.reduction_attempt_count += 1u;
        result = (int64_t *)table_gen_red(name_products, name_partial, sizeof(int64_t), 1u, red_handle, management, 0u);
        record->reduce_s = now_s() - started;
        if (result == NULL) {
            state.failure_stage = "reduction";
            state.reason = "simplepim_general_reduction_returned_null";
            goto finish;
        }
        if (!table_metadata_matches(
                management,
                name_partial,
                table_count_before + 1u,
                start_before,
                expected_end,
                1u,
                (uint32_t)sizeof(int64_t),
                0u,
                1u)) {
            free(result);
            state.failure_stage = "reduction";
            state.reason = "genred_output_table_metadata_invalid";
            goto finish;
        }
        state.reduction_completed_count += 1u;
        record->result = result[0];
        record->exact_match = record->result == record->reference;
        hash_hex(hash_int64_value_le(record->result), record->output_hash);
        record->total_s = record->scatter_s + record->zip_s + record->map_s + record->reduce_s;
        free(result);
        state.record_count += 1u;
        if (!record->exact_match) {
            state.failure_stage = "validation";
            state.reason = "int64_cpu_reference_mismatch";
            state.validation_status = "failed";
            goto finish;
        }
        observe_management(&state, management);
    }

finish:
    finalise_execution_state(&state, management);
    if (management != NULL && !dpu_released) {
        double started = now_s();
        state.release_attempted = true;
        dpu_error_t release_error = dpu_free(management->set);
        release_s = now_s() - started;
        dpu_released = true;
        state.release_confirmed = release_error == DPU_OK;
        if (!state.release_confirmed && state.failure_stage == NULL) {
            state.failure_stage = "release";
            state.reason = "dpu_free_failed";
            state.status = "failed";
        }
        release_management_metadata(management);
        management = NULL;
    }
    free_handle(map_handle);
    free_handle(red_handle);
    if (state.failure_stage == NULL && state.all_tasks_completed && state.exact_validation &&
        state.operator_validations_passed && state.operator_api_used && state.hardware_execution &&
        state.bounded_table_growth && state.persistent_allocation_observed &&
        state.allocated_dpus == (int)M42_DPU_COUNT &&
        strcmp(state.target_observed, "physical_hardware") == 0 && state.release_confirmed) {
        state.status = "completed";
        state.reason = NULL;
        state.hardware_functionality_evidence = true;
        exit_code = 0;
    } else {
        if (state.status == NULL) state.status = "failed";
        exit_code = 2;
    }
    if (write_response(response_path, &state, &provenance, hostname, host_hash, init_hash, map_hash, genred_hash, reduce_so_hash, allocation_s, handle_compile_s, release_s, false) != 0) return 2;
    return exit_code;
}
