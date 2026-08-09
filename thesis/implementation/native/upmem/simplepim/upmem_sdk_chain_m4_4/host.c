#define _POSIX_C_SOURCE 200809L

#include <dpu.h>

#include <inttypes.h>
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

#define M44_SCHEMA "simplepim_chain_m4_4_v1"
#define M44_PROFILE "hardware_simplepim_chain_m4_4_v1"
#define M44_BACKEND "upmem_sdk_hardware_simplepim_chain_m4_4"
#define M44_ROUTE "upmem_tn_hardware_simplepim_chain_m4_4"
#define M44_SOURCE_COMMIT "1d639c53532555f01e9f71d872e7712b166d6cba"
#define M44_DPU_COUNT 1u
#define M44_TASKLETS 1u
#define M44_LENGTH 256u
#define M44_WARMUPS 1u
#define M44_REPEATS 5u
#define M44_ITERATIONS (M44_WARMUPS + M44_REPEATS)
#define M44_OPERAND_BYTES (3u * M44_LENGTH * (unsigned)sizeof(int8_t))
#define M44_GRAPH_MAX_BYTES 16384u
#define M44_GRAPH_SCHEMA "M44_GRAPH_BINDING_V1"
#define M44_GRAPH_CASE_ID "simplepim_two_task_chain_fixture"
#define M44_GRAPH_EXPECTED_SCALAR INT64_C(-11654)

typedef struct {
    uint32_t repeat_id;
    bool warmup;
    int64_t reference;
    int64_t result;
    bool exact;
    double scatter_s;
    double zip_s;
    double map_s;
    double reduce_s;
    double total_s;
} repetition_t;

typedef struct {
    char case_id[128];
    char circuit_semantics_hash[65];
    char tensor_network_hash[65];
    char contraction_plan_hash[65];
    char contraction_path_structure_hash[65];
    char input_sha256[65];
    char binding_sha256[65];
    bool validated;
} graph_binding_t;

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
    bool simplepim_operator_api_used;
    bool map_attempted;
    bool map_completed;
    bool genred_attempted;
    bool genred_completed;
    uint32_t map_attempt_count;
    uint32_t map_completed_count;
    uint32_t genred_attempt_count;
    uint32_t genred_completed_count;
    char input_sha256[65];
    int64_t reference;
    bool all_tasks_completed;
    bool exact_integer_match;
    bool graph_binding_validated;
    bool native_taskgraph_protocol;
    graph_binding_t graph_binding;
    repetition_t repetitions[M44_ITERATIONS];
    size_t repetition_count;
} response_state_t;

static double now_s(void) {
    struct timespec value;
    if (clock_gettime(CLOCK_MONOTONIC, &value) != 0) return 0.0;
    return (double)value.tv_sec + (double)value.tv_nsec / 1000000000.0;
}

typedef struct {
    uint32_t state[8];
    uint64_t bits;
    unsigned char block[64];
    size_t length;
} sha256_t;

static uint32_t rotr32(uint32_t value, uint32_t bits) {
    return (value >> bits) | (value << (32u - bits));
}

static void sha256_transform(sha256_t *ctx, const unsigned char block[64]) {
    static const uint32_t k[64] = {
        UINT32_C(0x428a2f98), UINT32_C(0x71374491), UINT32_C(0xb5c0fbcf), UINT32_C(0xe9b5dba5),
        UINT32_C(0x3956c25b), UINT32_C(0x59f111f1), UINT32_C(0x923f82a4), UINT32_C(0xab1c5ed5),
        UINT32_C(0xd807aa98), UINT32_C(0x12835b01), UINT32_C(0x243185be), UINT32_C(0x550c7dc3),
        UINT32_C(0x72be5d74), UINT32_C(0x80deb1fe), UINT32_C(0x9bdc06a7), UINT32_C(0xc19bf174),
        UINT32_C(0xe49b69c1), UINT32_C(0xefbe4786), UINT32_C(0x0fc19dc6), UINT32_C(0x240ca1cc),
        UINT32_C(0x2de92c6f), UINT32_C(0x4a7484aa), UINT32_C(0x5cb0a9dc), UINT32_C(0x76f988da),
        UINT32_C(0x983e5152), UINT32_C(0xa831c66d), UINT32_C(0xb00327c8), UINT32_C(0xbf597fc7),
        UINT32_C(0xc6e00bf3), UINT32_C(0xd5a79147), UINT32_C(0x06ca6351), UINT32_C(0x14292967),
        UINT32_C(0x27b70a85), UINT32_C(0x2e1b2138), UINT32_C(0x4d2c6dfc), UINT32_C(0x53380d13),
        UINT32_C(0x650a7354), UINT32_C(0x766a0abb), UINT32_C(0x81c2c92e), UINT32_C(0x92722c85),
        UINT32_C(0xa2bfe8a1), UINT32_C(0xa81a664b), UINT32_C(0xc24b8b70), UINT32_C(0xc76c51a3),
        UINT32_C(0xd192e819), UINT32_C(0xd6990624), UINT32_C(0xf40e3585), UINT32_C(0x106aa070),
        UINT32_C(0x19a4c116), UINT32_C(0x1e376c08), UINT32_C(0x2748774c), UINT32_C(0x34b0bcb5),
        UINT32_C(0x391c0cb3), UINT32_C(0x4ed8aa4a), UINT32_C(0x5b9cca4f), UINT32_C(0x682e6ff3),
        UINT32_C(0x748f82ee), UINT32_C(0x78a5636f), UINT32_C(0x84c87814), UINT32_C(0x8cc70208),
        UINT32_C(0x90befffa), UINT32_C(0xa4506ceb), UINT32_C(0xbef9a3f7), UINT32_C(0xc67178f2)
    };
    uint32_t words[64];
    for (uint32_t i = 0; i < 16u; ++i) {
        words[i] = ((uint32_t)block[i * 4u] << 24) | ((uint32_t)block[i * 4u + 1u] << 16) |
                   ((uint32_t)block[i * 4u + 2u] << 8) | (uint32_t)block[i * 4u + 3u];
    }
    for (uint32_t i = 16u; i < 64u; ++i) {
        uint32_t s0 = rotr32(words[i - 15u], 7u) ^ rotr32(words[i - 15u], 18u) ^ (words[i - 15u] >> 3);
        uint32_t s1 = rotr32(words[i - 2u], 17u) ^ rotr32(words[i - 2u], 19u) ^ (words[i - 2u] >> 10);
        words[i] = words[i - 16u] + s0 + words[i - 7u] + s1;
    }
    uint32_t a = ctx->state[0], b = ctx->state[1], c = ctx->state[2], d = ctx->state[3];
    uint32_t e = ctx->state[4], f = ctx->state[5], g = ctx->state[6], h = ctx->state[7];
    for (uint32_t i = 0; i < 64u; ++i) {
        uint32_t s1 = rotr32(e, 6u) ^ rotr32(e, 11u) ^ rotr32(e, 25u);
        uint32_t choice = (e & f) ^ ((~e) & g);
        uint32_t temp1 = h + s1 + choice + k[i] + words[i];
        uint32_t s0 = rotr32(a, 2u) ^ rotr32(a, 13u) ^ rotr32(a, 22u);
        uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
        uint32_t temp2 = s0 + majority;
        h = g; g = f; f = e; e = d + temp1; d = c; c = b; b = a; a = temp1 + temp2;
    }
    ctx->state[0] += a; ctx->state[1] += b; ctx->state[2] += c; ctx->state[3] += d;
    ctx->state[4] += e; ctx->state[5] += f; ctx->state[6] += g; ctx->state[7] += h;
}

static void sha256_init(sha256_t *ctx) {
    static const uint32_t initial[8] = {
        UINT32_C(0x6a09e667), UINT32_C(0xbb67ae85), UINT32_C(0x3c6ef372), UINT32_C(0xa54ff53a),
        UINT32_C(0x510e527f), UINT32_C(0x9b05688c), UINT32_C(0x1f83d9ab), UINT32_C(0x5be0cd19)
    };
    memcpy(ctx->state, initial, sizeof(initial));
    ctx->bits = 0;
    ctx->length = 0;
}

static void sha256_update(sha256_t *ctx, const void *data, size_t length) {
    const unsigned char *bytes = (const unsigned char *)data;
    while (length != 0u) {
        size_t take = sizeof(ctx->block) - ctx->length;
        if (take > length) take = length;
        memcpy(ctx->block + ctx->length, bytes, take);
        ctx->length += take;
        ctx->bits += (uint64_t)take * 8u;
        bytes += take;
        length -= take;
        if (ctx->length == sizeof(ctx->block)) {
            sha256_transform(ctx, ctx->block);
            ctx->length = 0;
        }
    }
}

static void sha256_final(sha256_t *ctx, unsigned char digest[32]) {
    size_t i = ctx->length;
    ctx->block[i++] = 0x80;
    while (i < 64u) ctx->block[i++] = 0;
    if (ctx->length >= 56u) {
        sha256_transform(ctx, ctx->block);
        memset(ctx->block, 0, sizeof(ctx->block));
    }
    for (i = 0; i < 8u; ++i) ctx->block[56u + i] = (unsigned char)(ctx->bits >> (56u - 8u * i));
    sha256_transform(ctx, ctx->block);
    for (i = 0; i < 8u; ++i) {
        digest[i * 4u] = (unsigned char)(ctx->state[i] >> 24);
        digest[i * 4u + 1u] = (unsigned char)(ctx->state[i] >> 16);
        digest[i * 4u + 2u] = (unsigned char)(ctx->state[i] >> 8);
        digest[i * 4u + 3u] = (unsigned char)ctx->state[i];
    }
}

static void sha256_hex(const unsigned char digest[32], char output[65]) {
    static const char digits[] = "0123456789abcdef";
    for (size_t i = 0; i < 32u; ++i) {
        output[i * 2u] = digits[digest[i] >> 4];
        output[i * 2u + 1u] = digits[digest[i] & 15u];
    }
    output[64] = '\0';
}

static bool valid_hash_text(const char *value) {
    if (value == NULL || strlen(value) != 64u) return false;
    for (size_t i = 0; i < 64u; ++i) {
        const char c = value[i];
        if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F'))) return false;
    }
    return true;
}

static bool copy_graph_field(const char *line, const char *key, char *value, size_t value_size) {
    const size_t key_len = strlen(key);
    if (strncmp(line, key, key_len) != 0 || line[key_len] != '\t') return false;
    const char *start = line + key_len + 1u;
    const size_t length = strlen(start);
    if (length == 0u || length >= value_size || strchr(start, '\t') != NULL) return false;
    memcpy(value, start, length + 1u);
    return true;
}

static bool read_graph_binding(const char *path, char *buffer, size_t *length) {
    FILE *file = fopen(path, "rb");
    if (file == NULL) return false;
    const size_t count = fread(buffer, 1, M44_GRAPH_MAX_BYTES + 1u, file);
    const bool read_error = ferror(file) != 0;
    const bool trailing = count > M44_GRAPH_MAX_BYTES;
    if (fclose(file) != 0 || read_error || trailing || count == 0u || count > M44_GRAPH_MAX_BYTES) return false;
    buffer[count] = '\0';
    if (buffer[count - 1u] != '\n' || memchr(buffer, '\0', count) != NULL || memchr(buffer, '\r', count) != NULL) return false;
    size_t line_start = 0u;
    for (size_t i = 0u; i < count; ++i) {
        if (buffer[i] == '\n') {
            if (i == line_start) return false;
            line_start = i + 1u;
        }
    }
    if (line_start != count) return false;
    *length = count;
    return true;
}

static bool validate_graph_binding_lines(char *buffer, size_t length, graph_binding_t *binding) {
    char *lines[18];
    size_t line_count = 0u;
    char *cursor = buffer;
    char *end = buffer + length;
    while (cursor < end && line_count < 18u) {
        char *newline = strchr(cursor, '\n');
        if (newline == NULL) return false;
        *newline = '\0';
        lines[line_count++] = cursor;
        cursor = newline + 1u;
    }
    if (cursor != end || line_count != 18u) return false;
    if (strcmp(lines[0], M44_GRAPH_SCHEMA) != 0) return false;
    if (!copy_graph_field(lines[1], "CASE_ID", binding->case_id, sizeof(binding->case_id)) || strcmp(binding->case_id, M44_GRAPH_CASE_ID) != 0) return false;
    if (strcmp(lines[2], "TASK_COUNT\t2") != 0) return false;
    if (!copy_graph_field(lines[3], "CIRCUIT_SEMANTICS_HASH", binding->circuit_semantics_hash, sizeof(binding->circuit_semantics_hash)) || !valid_hash_text(binding->circuit_semantics_hash)) return false;
    if (!copy_graph_field(lines[4], "TENSOR_NETWORK_HASH", binding->tensor_network_hash, sizeof(binding->tensor_network_hash)) || !valid_hash_text(binding->tensor_network_hash)) return false;
    if (!copy_graph_field(lines[5], "CONTRACTION_PLAN_HASH", binding->contraction_plan_hash, sizeof(binding->contraction_plan_hash)) || !valid_hash_text(binding->contraction_plan_hash)) return false;
    if (!copy_graph_field(lines[6], "CONTRACTION_PATH_STRUCTURE_HASH", binding->contraction_path_structure_hash, sizeof(binding->contraction_path_structure_hash)) || !valid_hash_text(binding->contraction_path_structure_hash)) return false;
    if (!copy_graph_field(lines[7], "INPUT_SHA256", binding->input_sha256, sizeof(binding->input_sha256)) || !valid_hash_text(binding->input_sha256)) return false;
    if (strcmp(lines[8], "EXPECTED_SCALAR\t-11654") != 0) return false;
    if (strcmp(lines[9], "INPUT_DTYPE\tint8") != 0) return false;
    if (strcmp(lines[10], "ACCUMULATOR_DTYPE\tint32") != 0) return false;
    if (strcmp(lines[11], "LENGTH\t256") != 0) return false;
    if (strcmp(lines[12], "PATH_COUNT\t2") != 0) return false;
    if (strcmp(lines[13], "PATH\t0\t0\t1") != 0) return false;
    if (strcmp(lines[14], "PATH\t1\t0\t1") != 0) return false;
    if (strcmp(lines[15], "TASK\t0\ttask_0\t-\tchain_a,chain_b\tresult_0\telementwise_product_i8_i8") != 0) return false;
    if (strcmp(lines[16], "TASK\t1\ttask_1\ttask_0\tchain_c,result_0\tresult_1\tscalar_product_i32_i8_reduce_i64") != 0) return false;
    if (strcmp(lines[17], "END") != 0) return false;
    binding->validated = true;
    return true;
}

static int validate_graph_binding(const char *path, const char *expected_sha256, graph_binding_t *binding) {
    char buffer[M44_GRAPH_MAX_BYTES + 1u];
    size_t length = 0u;
    if (!read_graph_binding(path, buffer, &length)) return 1;
    unsigned char digest[32];
    char actual_sha256[65];
    sha256_t digest_ctx;
    sha256_init(&digest_ctx);
    sha256_update(&digest_ctx, buffer, length);
    sha256_final(&digest_ctx, digest);
    sha256_hex(digest, actual_sha256);
    memcpy(binding->binding_sha256, actual_sha256, sizeof(binding->binding_sha256));
    if (!valid_hash_text(expected_sha256) || strcmp(actual_sha256, expected_sha256) != 0) return 2;
    if (!validate_graph_binding_lines(buffer, length, binding)) return 3;
    return 0;
}

static void json_string(FILE *file, const char *value) {
    fputc('"', file);
    if (value != NULL) {
        for (const unsigned char *p = (const unsigned char *)value; *p != '\0'; ++p) {
            if (*p == '\\' || *p == '"') fputc('\\', file);
            if (*p == '\n') fputs("\\n", file);
            else if (*p == '\r') fputs("\\r", file);
            else if (*p == '\t') fputs("\\t", file);
            else fputc(*p, file);
        }
    }
    fputc('"', file);
}

static void json_bool(FILE *file, bool value) { fputs(value ? "true" : "false", file); }

static int write_response(const char *path, const response_state_t *state) {
    FILE *file = fopen(path, "w");
    if (file == NULL) return 1;
    const bool task_graph_integrated = state->graph_binding_validated && state->native_taskgraph_protocol;
    fprintf(file, "{\"schema_version\":"); json_string(file, M44_SCHEMA);
    fprintf(file, ",\"profile_id\":"); json_string(file, M44_PROFILE);
    fprintf(file, ",\"backend_id\":"); json_string(file, M44_BACKEND);
    fprintf(file, ",\"route_id\":"); json_string(file, M44_ROUTE);
    fprintf(file, ",\"source_commit\":"); json_string(file, M44_SOURCE_COMMIT);
    fprintf(file, ",\"target_requested\":\"physical_hardware\",\"target_observed\":"); json_string(file, state->target_observed);
    fprintf(file, ",\"requested_dpu_count\":%u,\"allocated_dpu_count\":%d,\"tasklets_per_dpu\":%u,\"effective_operator_tasklets\":%u", M44_DPU_COUNT, state->allocated_dpus, M44_TASKLETS, M44_TASKLETS);
    fprintf(file, ",\"provider_initialized\":"); json_bool(file, state->provider_initialized);
    fprintf(file, ",\"allocation_attempted\":"); json_bool(file, state->allocation_attempted);
    fprintf(file, ",\"release_attempted\":"); json_bool(file, state->release_attempted);
    fprintf(file, ",\"release_confirmed\":"); json_bool(file, state->release_confirmed);
    fprintf(file, ",\"simplepim_operator_api_used\":"); json_bool(file, state->simplepim_operator_api_used);
    fprintf(file, ",\"graph_binding_validated\":"); json_bool(file, state->graph_binding_validated);
    fprintf(file, ",\"native_taskgraph_protocol\":"); json_bool(file, state->native_taskgraph_protocol);
    fprintf(file, ",\"graph_binding_sha256\":"); json_string(file, state->graph_binding.binding_sha256);
    fprintf(file, ",\"graph_binding_schema_version\":"); json_string(file, M44_GRAPH_SCHEMA);
    fprintf(file, ",\"case_id\":"); json_string(file, state->graph_binding.case_id);
    fprintf(file, ",\"circuit_semantics_hash\":"); json_string(file, state->graph_binding.circuit_semantics_hash);
    fprintf(file, ",\"tensor_network_hash\":"); json_string(file, state->graph_binding.tensor_network_hash);
    fprintf(file, ",\"contraction_plan_hash\":"); json_string(file, state->graph_binding.contraction_plan_hash);
    fprintf(file, ",\"contraction_path_structure_hash\":"); json_string(file, state->graph_binding.contraction_path_structure_hash);
    fprintf(file, ",\"binding_input_sha256\":"); json_string(file, state->graph_binding.input_sha256);
    fprintf(file, ",\"graph_binding_expected_scalar\":%" PRId64, M44_GRAPH_EXPECTED_SCALAR);
    fprintf(file, ",\"graph_binding_input_dtype\":\"int8\",\"graph_binding_accumulator_dtype\":\"int32\",\"graph_binding_length\":%u,\"graph_binding_path_count\":2", M44_LENGTH);
    fprintf(file, ",\"graph_task_count\":2,\"graph_path\":\"0,1;0,1\"");
    fprintf(file, ",\"fixture_version\":\"simplepim_chain_fixture_v1\",\"task_graph_integrated\":"); json_bool(file, task_graph_integrated);
    fprintf(file, ",\"length\":%u,\"path\":[[0,1],[0,1]],\"task_order\":[\"task_0\",\"task_1\"],\"task_dependencies\":[[],[\"task_0\"]]", M44_LENGTH);
    fprintf(file, ",\"operation_kinds\":[\"elementwise_product_i8_i8\",\"scalar_product_i32_i8_reduce_i64\"]");
    fprintf(file, ",\"hardware_kernel_executed\":"); json_bool(file, state->all_tasks_completed);
    fprintf(file, ",\"simulator_kernel_executed\":false,\"cpu_fallback_used\":false,\"hardware_speedup_applicable\":false");
    fprintf(file, ",\"task_count\":2,\"map_task_count\":2,\"genred_task_count\":1,\"map_attempt_count\":%u,\"map_completed_count\":%u,\"genred_attempt_count\":%u,\"genred_completed_count\":%u", state->map_attempt_count, state->map_completed_count, state->genred_attempt_count, state->genred_completed_count);
    fprintf(file, ",\"final_reduction_location\":\"host\",\"intermediate_residency\":\"device_mram\",\"all_tasks_completed\":"); json_bool(file, state->all_tasks_completed);
    fprintf(file, ",\"exact_integer_match\":"); json_bool(file, state->exact_integer_match);
    fprintf(file, ",\"hardware_functionality_evidence\":"); json_bool(file, state->all_tasks_completed && state->exact_integer_match && state->release_confirmed);
    fprintf(file, ",\"input_dtype\":\"int8\",\"accumulator_dtype\":\"int32\",\"input_elements_per_operand\":%u,\"input_length_bytes\":%u,\"input_sha256\":", M44_LENGTH, (unsigned)M44_OPERAND_BYTES); json_string(file, state->input_sha256);
    fprintf(file, ",\"reference_int64\":%" PRId64, state->reference);
    fprintf(file, ",\"warmup_count\":%u,\"repeat_count\":%u,\"repetition_count\":%zu", M44_WARMUPS, M44_REPEATS, state->repetition_count);
    fprintf(file, ",\"application_visible_h2d_bytes\":%u,\"application_visible_d2h_bytes\":%u,\"application_visible_transfer_bytes\":%u", (unsigned)M44_OPERAND_BYTES, (unsigned)sizeof(int64_t), (unsigned)(M44_OPERAND_BYTES + sizeof(int64_t)));
    fprintf(file, ",\"intermediate_h2d_bytes\":0,\"intermediate_d2h_bytes\":0,\"intermediate_transfer_bytes\":0");
    fprintf(file, ",\"validation_status\":"); json_string(file, state->status != NULL && strcmp(state->status, "completed") == 0 ? "passed" : "not_run");
    fprintf(file, ",\"timing_scope\":\"physical SimplePIM native chain bring-up\",\"timing_is_bringup_only\":true,\"status\":"); json_string(file, state->status);
    fprintf(file, ",\"failure_stage\":"); if (state->failure_stage == NULL) fputs("null", file); else json_string(file, state->failure_stage);
    fprintf(file, ",\"reason\":"); if (state->reason == NULL) fputs("null", file); else json_string(file, state->reason);
    fputs(",\"repetitions\":[", file);
    for (size_t i = 0; i < state->repetition_count; ++i) {
        const repetition_t *row = &state->repetitions[i];
        if (i != 0u) fputc(',', file);
        fprintf(file, "{\"repeat_id\":%u,\"warmup\":", row->repeat_id); json_bool(file, row->warmup);
        fprintf(file, ",\"reference_int64\":%" PRId64 ",\"result_int64\":%" PRId64 ",\"exact_integer_match\":", row->reference, row->result); json_bool(file, row->exact);
        fprintf(file, ",\"scatter_time_s\":%.9f,\"virtual_zip_time_s\":%.9f,\"map_time_s\":%.9f,\"reduction_time_s\":%.9f,\"total_time_s\":%.9f,\"total_route_time_s\":%.9f}", row->scatter_s, row->zip_s, row->map_s, row->reduce_s, row->total_s, row->total_s);
    }
    fputs("]}\n", file);
    return fclose(file) == 0 ? 0 : 1;
}

static bool parse_arguments(int argc, char **argv, const char **mode, const char **response, const char **operands, const char **input_sha256, const char **graph_binding, const char **graph_binding_sha256) {
    bool invalid = false;
    *mode = "execute";
    *response = "build/simplepim_chain_m4_4/execute_response.json";
    *operands = NULL;
    *input_sha256 = NULL;
    *graph_binding = NULL;
    *graph_binding_sha256 = NULL;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--mode") == 0 && i + 1 < argc) *mode = argv[++i];
        else if (strcmp(argv[i], "--response") == 0 && i + 1 < argc) *response = argv[++i];
        else if (strcmp(argv[i], "--operands-file") == 0 && i + 1 < argc) *operands = argv[++i];
        else if (strcmp(argv[i], "--input-sha256") == 0 && i + 1 < argc) *input_sha256 = argv[++i];
        else if (strcmp(argv[i], "--graph-binding") == 0 && i + 1 < argc) *graph_binding = argv[++i];
        else if (strcmp(argv[i], "--graph-binding-sha256") == 0 && i + 1 < argc) *graph_binding_sha256 = argv[++i];
        else if (strcmp(argv[i], "--stage-manifest") == 0 && i + 1 < argc) ++i;
        else invalid = true;
    }
    return !invalid;
}

static const char *load_operands(const char *path, int8_t *a, int8_t *b, int8_t *c, int64_t *reference, char sha256[65]) {
    unsigned char raw[M44_OPERAND_BYTES];
    FILE *file = fopen(path, "rb");
    if (file == NULL) return "operand_file_open_failed";
    size_t count = fread(raw, 1, sizeof(raw), file);
    if (count != sizeof(raw)) {
        bool read_error = ferror(file) != 0;
        (void)fclose(file);
        return read_error ? "operand_file_read_failed" : "operand_file_short";
    }
    int trailing = fgetc(file);
    if (trailing != EOF || ferror(file) != 0) {
        (void)fclose(file);
        return trailing != EOF ? "operand_file_trailing_data" : "operand_file_read_failed";
    }
    if (fclose(file) != 0) return "operand_file_read_failed";
    sha256_t digest_ctx;
    unsigned char digest[32];
    sha256_init(&digest_ctx);
    sha256_update(&digest_ctx, raw, sizeof(raw));
    sha256_final(&digest_ctx, digest);
    sha256_hex(digest, sha256);
    *reference = 0;
    for (uint32_t i = 0; i < M44_LENGTH; ++i) {
        a[i] = (int8_t)raw[i];
        b[i] = (int8_t)raw[M44_LENGTH + i];
        c[i] = (int8_t)raw[2u * M44_LENGTH + i];
        *reference += (int64_t)a[i] * (int64_t)b[i] * (int64_t)c[i];
    }
    return NULL;
}

static bool simulator_selector_present(void) {
    static const char *const names[] = {
        "DPU_BACKEND", "DPU_PROFILE", "SIMPLEPIM_BACKEND", "UPMEM_BACKEND", "UPMEM_MODE", "UPMEM_TARGET", "UPMEM_PROFILE", "UPMEM_PROFILE_BASE"
    };
    for (size_t i = 0; i < sizeof(names) / sizeof(names[0]); ++i) if (getenv(names[i]) != NULL) return true;
    return false;
}

static bool prepare_stage_cwd(void) {
    if (access("product_i8_i8/map.h", R_OK) == 0) return true;
    if (access("../product_i8_i8/map.h", R_OK) == 0 && chdir("..") == 0) return true;
    return false;
}

static void free_handle_safe(handle_t *handle) {
    if (handle == NULL) return;
    free(handle->bin_location);
    free(handle->so_bin_location);
    free(handle);
}

static void free_management(simplepim_management_t *management) {
    if (management == NULL) return;
    for (uint32_t i = 0; i < management->num_tables; ++i) {
        if (management->tables[i] == NULL) continue;
        free(management->tables[i]->name);
        free(management->tables[i]->lens_each_dpu);
        free(management->tables[i]);
    }
    free(management->tables);
    free(management->zip_args);
    free(management->map_args);
    free(management->red_args);
    free(management);
}

int main(int argc, char **argv) {
    response_state_t state = {0};
    const char *mode;
    const char *response_path;
    const char *operands_path;
    const char *input_sha256_arg;
    const char *graph_binding_path;
    const char *graph_binding_sha256_arg;
    int8_t a[M44_LENGTH], b[M44_LENGTH], c[M44_LENGTH];
    int64_t reference = 0;
    simplepim_management_t *management = NULL;
    handle_t *map_i8_handle = NULL;
    handle_t *map_i32_handle = NULL;
    handle_t *zip_handle = NULL;
    handle_t *red_handle = NULL;
    int exit_code = 2;

    state.status = "failed";
    state.failure_stage = "host_setup";
    state.reason = "not_started";
    state.target_observed = "not_executed";
    state.allocated_dpus = -1;
    if (!parse_arguments(argc, argv, &mode, &response_path, &operands_path, &input_sha256_arg, &graph_binding_path, &graph_binding_sha256_arg)) {
        state.failure_stage = "arguments";
        state.reason = "invalid_arguments";
        (void)write_response(response_path, &state);
        return exit_code;
    }
    if (strcmp(mode, "parser") == 0) {
        state.status = "prepared";
        state.failure_stage = NULL;
        state.reason = NULL;
        (void)write_response(response_path, &state);
        return 0;
    }
    if (strcmp(mode, "execute") != 0 || operands_path == NULL || graph_binding_path == NULL || graph_binding_sha256_arg == NULL) {
        if (operands_path == NULL) {
            state.failure_stage = "arguments";
            state.reason = "operands_file_required";
        } else if (graph_binding_path == NULL) {
            state.failure_stage = "graph_binding_read";
            state.reason = "graph_binding_required";
        } else if (graph_binding_sha256_arg == NULL) {
            state.failure_stage = "graph_binding_sha256";
            state.reason = "graph_binding_sha256_required";
        } else {
            state.failure_stage = "arguments";
            state.reason = "mode_must_be_execute_or_parser";
        }
        (void)write_response(response_path, &state);
        return exit_code;
    }
    {
        const int binding_error = validate_graph_binding(graph_binding_path, graph_binding_sha256_arg, &state.graph_binding);
        if (binding_error != 0) {
            state.failure_stage = binding_error == 1 ? "graph_binding_read" : (binding_error == 2 ? "graph_binding_sha256" : "graph_binding_contract");
            state.reason = binding_error == 1 ? "graph_binding_file_invalid" : (binding_error == 2 ? "graph_binding_sha256_mismatch" : "graph_binding_contract_invalid");
            (void)write_response(response_path, &state);
            return exit_code;
        }
        state.graph_binding_validated = true;
    }
    {
        const char *input_error = load_operands(operands_path, a, b, c, &reference, state.input_sha256);
        if (input_error != NULL) {
            state.failure_stage = "input";
            state.reason = input_error;
            (void)write_response(response_path, &state);
            return exit_code;
        }
    }
    if (input_sha256_arg != NULL && strcmp(input_sha256_arg, state.input_sha256) != 0) {
        state.failure_stage = "input";
        state.reason = "input_sha256_mismatch";
        (void)write_response(response_path, &state);
        return exit_code;
    }
    if (strcmp(state.graph_binding.input_sha256, state.input_sha256) != 0 || reference != M44_GRAPH_EXPECTED_SCALAR) {
        state.failure_stage = "graph_binding_contract";
        state.reason = strcmp(state.graph_binding.input_sha256, state.input_sha256) != 0 ? "graph_binding_input_sha256_mismatch" : "graph_binding_expected_scalar_mismatch";
        (void)write_response(response_path, &state);
        return exit_code;
    }
    state.native_taskgraph_protocol = true;
    state.reference = reference;
    if (getenv("UPMEM_ALLOW_PHYSICAL_HARDWARE") == NULL || strcmp(getenv("UPMEM_ALLOW_PHYSICAL_HARDWARE"), "1") != 0) {
        state.failure_stage = "opt_in";
        state.reason = "UPMEM_ALLOW_PHYSICAL_HARDWARE_must_equal_1";
        (void)write_response(response_path, &state);
        return exit_code;
    }
    if (simulator_selector_present()) {
        state.failure_stage = "hardware_profile";
        state.reason = "simulator_selector_must_be_unset";
        (void)write_response(response_path, &state);
        return exit_code;
    }
    if (!prepare_stage_cwd()) {
        state.failure_stage = "staging";
        state.reason = "benchmark_stage_not_found";
        (void)write_response(response_path, &state);
        return exit_code;
    }
    state.failure_stage = NULL;
    state.reason = NULL;

    state.allocation_attempted = true;
    management = table_management_init(M44_DPU_COUNT);
    state.provider_initialized = management != NULL;
    if (management == NULL) {
        state.failure_stage = "allocation";
        state.reason = "simplepim_management_init_returned_null";
        goto finish;
    }
    uint32_t observed = 0;
    if (dpu_get_nr_dpus(management->set, &observed) != DPU_OK || observed != M44_DPU_COUNT) {
        state.failure_stage = "allocation";
        state.reason = "allocated_dpu_count_mismatch";
        goto finish;
    }
    state.allocated_dpus = (int)observed;
    state.target_observed = "physical_hardware";
    map_i8_handle = create_handle("product_i8_i8", MAP);
    map_i32_handle = create_handle("product_i32_i8", MAP);
    zip_handle = create_handle("zip", ZIP);
    red_handle = create_handle("reduce_i32", REDUCE);
    if (map_i8_handle == NULL || map_i32_handle == NULL || zip_handle == NULL || red_handle == NULL) {
        state.failure_stage = "operator_setup";
        state.reason = "operator_handle_build_failed";
        goto finish;
    }

    for (uint32_t iteration = 0; iteration < M44_ITERATIONS; ++iteration) {
        repetition_t *row = &state.repetitions[state.repetition_count];
        void *table_a = NULL, *table_b = NULL, *table_c = NULL;
        int64_t *result = NULL;
        char name_a[32], name_b[32], name_c[32], name_zip0[32], name_result0[32], name_zip1[32], name_result1[32], name_final[32];
        double started;
        memset(row, 0, sizeof(*row));
        row->repeat_id = iteration == 0u ? 0u : iteration - 1u;
        row->warmup = iteration == 0u;
        row->reference = reference;
        (void)snprintf(name_a, sizeof(name_a), "m44_r%u_a", iteration);
        (void)snprintf(name_b, sizeof(name_b), "m44_r%u_b", iteration);
        (void)snprintf(name_c, sizeof(name_c), "m44_r%u_c", iteration);
        (void)snprintf(name_zip0, sizeof(name_zip0), "m44_r%u_zip0", iteration);
        (void)snprintf(name_result0, sizeof(name_result0), "m44_r%u_result0", iteration);
        (void)snprintf(name_zip1, sizeof(name_zip1), "m44_r%u_zip1", iteration);
        (void)snprintf(name_result1, sizeof(name_result1), "m44_r%u_result1", iteration);
        (void)snprintf(name_final, sizeof(name_final), "m44_r%u_final", iteration);

        started = now_s();
        table_a = malloc_scatter_aligned(M44_LENGTH, sizeof(int8_t), management);
        table_b = malloc_scatter_aligned(M44_LENGTH, sizeof(int8_t), management);
        table_c = malloc_scatter_aligned(M44_LENGTH, sizeof(int8_t), management);
        if (table_a == NULL || table_b == NULL || table_c == NULL) {
            free(table_a); free(table_b); free(table_c);
            state.failure_stage = "scatter";
            state.reason = "aligned_input_allocation_failed";
            goto finish;
        }
        memcpy(table_a, a, M44_LENGTH * sizeof(int8_t));
        memcpy(table_b, b, M44_LENGTH * sizeof(int8_t));
        memcpy(table_c, c, M44_LENGTH * sizeof(int8_t));
        simplepim_scatter(name_a, table_a, M44_LENGTH, sizeof(int8_t), management);
        simplepim_scatter(name_b, table_b, M44_LENGTH, sizeof(int8_t), management);
        simplepim_scatter(name_c, table_c, M44_LENGTH, sizeof(int8_t), management);
        free(table_a); free(table_b); free(table_c);
        if (!contains_table(name_a, management) || !contains_table(name_b, management) || !contains_table(name_c, management)) {
            state.failure_stage = "scatter";
            state.reason = "input_table_registration_failed";
            goto finish;
        }
        row->scatter_s = now_s() - started;

        started = now_s();
        table_zip(name_a, name_b, name_zip0, zip_handle, management);
        if (!contains_table(name_zip0, management)) {
            state.failure_stage = "task_0";
            state.reason = "first_virtual_zip_failed";
            goto finish;
        }
        row->zip_s = now_s() - started;

        started = now_s();
        state.map_attempt_count += 1u;
        table_map(name_zip0, name_result0, sizeof(int32_t), map_i8_handle, management, 0u);
        row->map_s = now_s() - started;
        if (!contains_table(name_result0, management)) {
            state.failure_stage = "map";
            state.reason = "first_map_output_table_registration_failed";
            goto finish;
        }

        started = now_s();
        table_zip(name_result0, name_c, name_zip1, zip_handle, management);
        row->zip_s += now_s() - started;
        if (!contains_table(name_zip1, management)) {
            state.failure_stage = "task_1";
            state.reason = "second_virtual_zip_failed";
            goto finish;
        }

        started = now_s();
        state.map_attempt_count += 1u;
        table_map(name_zip1, name_result1, sizeof(int32_t), map_i32_handle, management, 0u);
        row->map_s += now_s() - started;
        if (!contains_table(name_result1, management)) {
            state.failure_stage = "map";
            state.reason = "second_map_output_table_registration_failed";
            goto finish;
        }
        state.map_completed_count += 2u;

        started = now_s();
        state.genred_attempt_count += 1u;
        result = (int64_t *)table_gen_red(name_result1, name_final, sizeof(int64_t), 1u, red_handle, management, 0u);
        row->reduce_s = now_s() - started;
        if (result == NULL || !contains_table(name_final, management)) {
            free(result);
            state.failure_stage = "genred";
            state.reason = "final_host_reduction_failed";
            goto finish;
        }
        state.genred_completed_count += 1u;
        row->result = result[0];
        row->exact = row->result == row->reference;
        row->total_s = row->scatter_s + row->zip_s + row->map_s + row->reduce_s;
        free(result);
        state.repetition_count += 1u;
        if (!row->exact) {
            state.failure_stage = "validation";
            state.reason = "int64_reference_mismatch";
            goto finish;
        }
    }

finish:
    state.map_attempted = state.map_attempt_count != 0u;
    state.map_completed = state.map_completed_count == 2u * M44_ITERATIONS;
    state.genred_attempted = state.genred_attempt_count != 0u;
    state.genred_completed = state.genred_completed_count == M44_ITERATIONS;
    state.all_tasks_completed = state.repetition_count == M44_ITERATIONS && state.map_completed && state.genred_completed;
    state.exact_integer_match = state.repetition_count == M44_ITERATIONS;
    for (size_t i = 0; i < state.repetition_count; ++i) state.exact_integer_match = state.exact_integer_match && state.repetitions[i].exact;
    state.simplepim_operator_api_used = state.all_tasks_completed && state.exact_integer_match;
    if (management != NULL) {
        state.release_attempted = true;
        state.release_confirmed = dpu_free(management->set) == DPU_OK;
        free_management(management);
        management = NULL;
    }
    free_handle_safe(map_i8_handle);
    free_handle_safe(map_i32_handle);
    free_handle_safe(zip_handle);
    free_handle_safe(red_handle);
    if (state.failure_stage == NULL && state.all_tasks_completed && state.exact_integer_match && state.release_confirmed) {
        state.status = "completed";
        state.reason = NULL;
        state.failure_stage = NULL;
        exit_code = 0;
    } else if (state.failure_stage == NULL) {
        state.failure_stage = "execution";
        state.reason = "native_chain_incomplete";
    }
    if (write_response(response_path, &state) != 0) return 2;
    return exit_code;
}
