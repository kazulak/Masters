#define _POSIX_C_SOURCE 199309L

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "cjson/cJSON.h"
#include "quantizer.h"
#include "../Param.h"

#include <dpu.h>

#define MAX_TENSORS 256
#define MAX_DIMS 16

typedef struct {
    char key[64];
    double *real;
    double *imag;
    int n_elements;
    int shape[MAX_DIMS];
    int labels[MAX_DIMS];
    int n_dims;
} TensorEntry;

typedef struct {
    struct dpu_set_t set;
    struct dpu_set_t dpu;
} PimContext;

typedef struct {
    int task_id;
    char input_a[64];
    char input_b[64];
    char output[64];
    int m;
    int k;
    int n;
    int n_row_blocks;
    int n_col_blocks;
    int gemm_tile_invocations;
    double total_seconds;
    double host_prepare_seconds;
    double host_pack_quantize_seconds;
    double h2d_dma_seconds;
    double dpu_kernel_seconds;
    double d2h_dma_seconds;
    double host_dequantize_accumulate_seconds;
    size_t host_to_dpu_bytes;
    size_t dpu_to_host_bytes;
} TaskProfile;

static TensorEntry g_registry[MAX_TENSORS];
static int g_n_entries = 0;

static double wall_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static void die(const char *msg) {
    fprintf(stderr, "%s\n", msg);
    exit(1);
}

static cJSON *json_get(const cJSON *obj, const char *key) {
    cJSON *item = cJSON_GetObjectItem(obj, key);
    if (!item) {
        fprintf(stderr, "Missing JSON key: %s\n", key);
        exit(1);
    }
    return item;
}

static int json_int(const cJSON *obj, const char *key) {
    return json_get(obj, key)->valueint;
}

static const char *json_string(const cJSON *obj, const char *key) {
    cJSON *item = json_get(obj, key);
    if (!item->valuestring) {
        fprintf(stderr, "JSON key is not a string: %s\n", key);
        exit(1);
    }
    return item->valuestring;
}

static int read_int_array(const cJSON *obj, const char *key, int *out, int max_len) {
    cJSON *arr = json_get(obj, key);
    int n = cJSON_GetArraySize(arr);
    if (n > max_len) die("JSON array exceeds MAX_DIMS");
    for (int i = 0; i < n; i++) {
        out[i] = cJSON_GetArrayItem(arr, i)->valueint;
    }
    return n;
}

static TensorEntry *registry_get(const char *key) {
    for (int i = 0; i < g_n_entries; i++) {
        if (strcmp(g_registry[i].key, key) == 0) return &g_registry[i];
    }
    return NULL;
}

static TensorEntry *registry_insert(const char *key, int n_elements,
                                    const int *shape, const int *labels,
                                    int n_dims) {
    if (g_n_entries >= MAX_TENSORS) die("Tensor registry full");
    TensorEntry *e = &g_registry[g_n_entries++];
    memset(e, 0, sizeof(*e));
    snprintf(e->key, sizeof(e->key), "%s", key);
    e->n_elements = n_elements;
    e->n_dims = n_dims;
    memcpy(e->shape, shape, (size_t)n_dims * sizeof(int));
    memcpy(e->labels, labels, (size_t)n_dims * sizeof(int));
    e->real = (double *)calloc((size_t)n_elements, sizeof(double));
    e->imag = (double *)calloc((size_t)n_elements, sizeof(double));
    if (!e->real || !e->imag) die("calloc failed for tensor registry entry");
    return e;
}

static void registry_free_entry(const char *key) {
    for (int i = 0; i < g_n_entries; i++) {
        if (strcmp(g_registry[i].key, key) == 0) {
            free(g_registry[i].real);
            free(g_registry[i].imag);
            memmove(&g_registry[i], &g_registry[i + 1],
                    (size_t)(g_n_entries - i - 1) * sizeof(TensorEntry));
            g_n_entries--;
            return;
        }
    }
}

static int label_pos(const int *labels, int n_labels, int label) {
    for (int i = 0; i < n_labels; i++) {
        if (labels[i] == label) return i;
    }
    return -1;
}

static int flat_offset_from_order(const TensorEntry *t, const int *order,
                                  int n_order, int linear_index) {
    int bits[MAX_DIMS] = {0};
    for (int i = n_order - 1; i >= 0; i--) {
        bits[i] = linear_index & 1;
        linear_index >>= 1;
    }

    int offset = 0;
    for (int d = 0; d < t->n_dims; d++) {
        int p = label_pos(order, n_order, t->labels[d]);
        if (p < 0) die("Internal error: tensor label missing from GEMM order");
        offset = offset * t->shape[d] + bits[p];
    }
    return offset;
}

static void tensor_to_matrix(const TensorEntry *t, const int *row_labels, int n_rows_labels,
                             const int *col_labels, int n_cols_labels, double *real_out,
                             double *imag_out, int rows, int cols) {
    int order[MAX_DIMS];
    int n_order = 0;
    for (int i = 0; i < n_rows_labels; i++) order[n_order++] = row_labels[i];
    for (int i = 0; i < n_cols_labels; i++) order[n_order++] = col_labels[i];
    if (n_order != t->n_dims) die("Tensor/GEMM label count mismatch");

    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            int matrix_offset = r * cols + c;
            int tensor_offset = flat_offset_from_order(t, order, n_order, matrix_offset);
            real_out[matrix_offset] = t->real[tensor_offset];
            imag_out[matrix_offset] = t->imag[tensor_offset];
        }
    }
}

static void dispatch_gemm_tile(PimContext *pim, const int8_t *a_tile,
                               const int8_t *b_tile, int32_t *result,
                               int tile_rows, int k, int tile_cols,
                               double *t_dma_out, double *t_dpu,
                               double *t_dma_in,
                               size_t *bytes_dma_out,
                               size_t *bytes_dma_in) {
    GemmTileInput *input = (GemmTileInput *)calloc(1, sizeof(GemmTileInput));
    if (!input) die("calloc failed for GemmTileInput");
    input->tile_rows = tile_rows;
    input->k = k;
    input->tile_cols = tile_cols;
    memcpy(input->a, a_tile, (size_t)tile_rows * (size_t)k);
    memcpy(input->b, b_tile, (size_t)k * (size_t)tile_cols);

    double t0 = wall_seconds();
    DPU_ASSERT(dpu_copy_to(pim->dpu, "DPU_INPUT", 0, input,
                           sizeof(GemmTileInput)));
    double t1 = wall_seconds();
    *t_dma_out += (t1 - t0);
    *bytes_dma_out += sizeof(GemmTileInput);

    t0 = wall_seconds();
    DPU_ASSERT(dpu_launch(pim->set, DPU_SYNCHRONOUS));
    t1 = wall_seconds();
    *t_dpu += (t1 - t0);

    GemmTileOutput *output = (GemmTileOutput *)calloc(1, sizeof(GemmTileOutput));
    if (!output) die("calloc failed for GemmTileOutput");
    t0 = wall_seconds();
    DPU_ASSERT(dpu_copy_from(pim->dpu, "DPU_OUTPUT", 0, output,
                             sizeof(GemmTileOutput)));
    t1 = wall_seconds();
    *t_dma_in += (t1 - t0);
    *bytes_dma_in += sizeof(GemmTileOutput);
    memcpy(result, output->c, (size_t)tile_rows * (size_t)tile_cols * sizeof(int32_t));

    free(output);
    free(input);
}

static void execute_task(PimContext *pim, const cJSON *task, TaskProfile *profile) {
    double t_task_start = wall_seconds();

    const char *key_a = json_string(task, "input_A_key");
    const char *key_b = json_string(task, "input_B_key");
    const char *key_out = json_string(task, "output_key");

    memset(profile, 0, sizeof(*profile));
    profile->task_id = json_int(task, "task_id");
    snprintf(profile->input_a, sizeof(profile->input_a), "%s", key_a);
    snprintf(profile->input_b, sizeof(profile->input_b), "%s", key_b);
    snprintf(profile->output, sizeof(profile->output), "%s", key_out);

    int m = json_int(task, "m");
    int k = json_int(task, "k");
    int n = json_int(task, "n");
    int n_row_blocks = json_int(task, "n_row_blocks");
    int n_col_blocks = json_int(task, "n_col_blocks");
    profile->m = m;
    profile->k = k;
    profile->n = n;
    profile->n_row_blocks = n_row_blocks;
    profile->n_col_blocks = n_col_blocks;

    if (json_get(task, "needs_k_tiling")->valueint) {
        fprintf(stderr, "ERROR: task requires K-tiling (K=%d > TILE_K=%d). "
                        "Out of MVP scope.\n", k, TILE_K);
        exit(1);
    }
    if (k > TILE_K) die("ERROR: K exceeds TILE_K but needs_k_tiling was false");

    TensorEntry *a = registry_get(key_a);
    TensorEntry *b = registry_get(key_b);
    if (!a || !b) die("Missing input tensor in registry");

    int free_a[MAX_DIMS], contracted[MAX_DIMS], free_b[MAX_DIMS];
    int n_free_a = read_int_array(task, "free_A", free_a, MAX_DIMS);
    int n_contract = read_int_array(task, "contracted", contracted, MAX_DIMS);
    int n_free_b = read_int_array(task, "free_B", free_b, MAX_DIMS);

    int shape_out[MAX_DIMS], labels_out[MAX_DIMS];
    int n_dims = read_int_array(task, "shape_out", shape_out, MAX_DIMS);
    int n_label_out = read_int_array(task, "labels_out", labels_out, MAX_DIMS);
    if (n_dims != n_label_out) die("shape_out/labels_out length mismatch");

    int n_out_elements = 1;
    for (int d = 0; d < n_dims; d++) n_out_elements *= shape_out[d];
    if (n_out_elements != m * n) die("Task output element count does not equal m*n");

    double *ar = (double *)malloc((size_t)m * (size_t)k * sizeof(double));
    double *ai = (double *)malloc((size_t)m * (size_t)k * sizeof(double));
    double *br = (double *)malloc((size_t)k * (size_t)n * sizeof(double));
    double *bi = (double *)malloc((size_t)k * (size_t)n * sizeof(double));
    if (!ar || !ai || !br || !bi) die("malloc failed for GEMM matrices");

    double t0 = wall_seconds();
    tensor_to_matrix(a, free_a, n_free_a, contracted, n_contract, ar, ai, m, k);
    tensor_to_matrix(b, contracted, n_contract, free_b, n_free_b, br, bi, k, n);
    profile->host_prepare_seconds += wall_seconds() - t0;

    TensorEntry *c = registry_insert(key_out, n_out_elements, shape_out,
                                     labels_out, n_dims);

    int8_t *a_tile_i8 = (int8_t *)malloc(TILE_ROWS * TILE_K);
    int8_t *ai_tile_i8 = (int8_t *)malloc(TILE_ROWS * TILE_K);
    int8_t *b_tile_i8 = (int8_t *)malloc(TILE_K * TILE_N);
    int8_t *bi_tile_i8 = (int8_t *)malloc(TILE_K * TILE_N);
    double *a_tile_f64 = (double *)malloc(TILE_ROWS * TILE_K * sizeof(double));
    double *b_tile_f64 = (double *)malloc(TILE_K * TILE_N * sizeof(double));
    int32_t *result_i32 = (int32_t *)malloc(TILE_ROWS * TILE_N * sizeof(int32_t));
    double *result_f64 = (double *)malloc(TILE_ROWS * TILE_N * sizeof(double));
    if (!a_tile_i8 || !ai_tile_i8 || !b_tile_i8 || !bi_tile_i8 ||
        !a_tile_f64 || !b_tile_f64 || !result_i32 || !result_f64) {
        die("malloc failed for tile scratch buffers");
    }

    for (int rb = 0; rb < n_row_blocks; rb++) {
        int row_start = rb * TILE_ROWS;
        int this_rows = (row_start + TILE_ROWS <= m) ? TILE_ROWS : (m - row_start);
        for (int cb = 0; cb < n_col_blocks; cb++) {
            int col_start = cb * TILE_N;
            int this_cols = (col_start + TILE_N <= n) ? TILE_N : (n - col_start);

            double scale_ar, scale_ai, scale_br, scale_bi;
            t0 = wall_seconds();
            extract_tile(ar, a_tile_f64, m, k, row_start, 0, this_rows, k);
            quantize_f64_to_i8(a_tile_f64, a_tile_i8, (size_t)this_rows * (size_t)k,
                               &scale_ar);
            extract_tile(ai, a_tile_f64, m, k, row_start, 0, this_rows, k);
            quantize_f64_to_i8(a_tile_f64, ai_tile_i8, (size_t)this_rows * (size_t)k,
                               &scale_ai);
            extract_tile(br, b_tile_f64, k, n, 0, col_start, k, this_cols);
            quantize_f64_to_i8(b_tile_f64, b_tile_i8, (size_t)k * (size_t)this_cols,
                               &scale_br);
            extract_tile(bi, b_tile_f64, k, n, 0, col_start, k, this_cols);
            quantize_f64_to_i8(b_tile_f64, bi_tile_i8, (size_t)k * (size_t)this_cols,
                               &scale_bi);
            profile->host_pack_quantize_seconds += wall_seconds() - t0;

            dispatch_gemm_tile(pim, a_tile_i8, b_tile_i8, result_i32,
                               this_rows, k, this_cols,
                               &profile->h2d_dma_seconds,
                               &profile->dpu_kernel_seconds,
                               &profile->d2h_dma_seconds,
                               &profile->host_to_dpu_bytes,
                               &profile->dpu_to_host_bytes);
            profile->gemm_tile_invocations++;
            t0 = wall_seconds();
            memset(result_f64, 0, (size_t)this_rows * (size_t)this_cols * sizeof(double));
            dequantize_i32_accumulate(result_i32, result_f64,
                                      (size_t)this_rows * (size_t)this_cols,
                                      scale_ar, scale_br);
            accumulate_tile(c->real, result_f64, m, n, row_start, col_start,
                            this_rows, this_cols);
            profile->host_dequantize_accumulate_seconds += wall_seconds() - t0;

            dispatch_gemm_tile(pim, ai_tile_i8, bi_tile_i8, result_i32,
                               this_rows, k, this_cols,
                               &profile->h2d_dma_seconds,
                               &profile->dpu_kernel_seconds,
                               &profile->d2h_dma_seconds,
                               &profile->host_to_dpu_bytes,
                               &profile->dpu_to_host_bytes);
            profile->gemm_tile_invocations++;
            t0 = wall_seconds();
            memset(result_f64, 0, (size_t)this_rows * (size_t)this_cols * sizeof(double));
            dequantize_i32_accumulate(result_i32, result_f64,
                                      (size_t)this_rows * (size_t)this_cols,
                                      scale_ai, scale_bi);
            for (int i = 0; i < this_rows * this_cols; i++) result_f64[i] = -result_f64[i];
            accumulate_tile(c->real, result_f64, m, n, row_start, col_start,
                            this_rows, this_cols);
            profile->host_dequantize_accumulate_seconds += wall_seconds() - t0;

            dispatch_gemm_tile(pim, a_tile_i8, bi_tile_i8, result_i32,
                               this_rows, k, this_cols,
                               &profile->h2d_dma_seconds,
                               &profile->dpu_kernel_seconds,
                               &profile->d2h_dma_seconds,
                               &profile->host_to_dpu_bytes,
                               &profile->dpu_to_host_bytes);
            profile->gemm_tile_invocations++;
            t0 = wall_seconds();
            memset(result_f64, 0, (size_t)this_rows * (size_t)this_cols * sizeof(double));
            dequantize_i32_accumulate(result_i32, result_f64,
                                      (size_t)this_rows * (size_t)this_cols,
                                      scale_ar, scale_bi);
            accumulate_tile(c->imag, result_f64, m, n, row_start, col_start,
                            this_rows, this_cols);
            profile->host_dequantize_accumulate_seconds += wall_seconds() - t0;

            dispatch_gemm_tile(pim, ai_tile_i8, b_tile_i8, result_i32,
                               this_rows, k, this_cols,
                               &profile->h2d_dma_seconds,
                               &profile->dpu_kernel_seconds,
                               &profile->d2h_dma_seconds,
                               &profile->host_to_dpu_bytes,
                               &profile->dpu_to_host_bytes);
            profile->gemm_tile_invocations++;
            t0 = wall_seconds();
            memset(result_f64, 0, (size_t)this_rows * (size_t)this_cols * sizeof(double));
            dequantize_i32_accumulate(result_i32, result_f64,
                                      (size_t)this_rows * (size_t)this_cols,
                                      scale_ai, scale_br);
            accumulate_tile(c->imag, result_f64, m, n, row_start, col_start,
                            this_rows, this_cols);
            profile->host_dequantize_accumulate_seconds += wall_seconds() - t0;
        }
    }

    free(a_tile_i8);
    free(ai_tile_i8);
    free(b_tile_i8);
    free(bi_tile_i8);
    free(a_tile_f64);
    free(b_tile_f64);
    free(result_i32);
    free(result_f64);
    free(ar);
    free(ai);
    free(br);
    free(bi);

    registry_free_entry(key_a);
    registry_free_entry(key_b);
    profile->total_seconds = wall_seconds() - t_task_start;
}

static void add_profile_json(cJSON *array, const TaskProfile *p) {
    cJSON *obj = cJSON_CreateObject();
    cJSON_AddItemToArray(array, obj);

    cJSON_AddNumberToObject(obj, "task_id", p->task_id);
    cJSON_AddStringToObject(obj, "selected_route", "raw_upmem_dense");
    cJSON_AddStringToObject(obj, "selected_format", "complex_i8_tile_scaled");
    cJSON_AddStringToObject(obj, "status", "ok");
    cJSON_AddStringToObject(obj, "input_A_key", p->input_a);
    cJSON_AddStringToObject(obj, "input_B_key", p->input_b);
    cJSON_AddStringToObject(obj, "output_key", p->output);

    cJSON *shape = cJSON_AddObjectToObject(obj, "gemm_shape");
    cJSON_AddNumberToObject(shape, "m", p->m);
    cJSON_AddNumberToObject(shape, "k", p->k);
    cJSON_AddNumberToObject(shape, "n", p->n);

    cJSON *tiling = cJSON_AddObjectToObject(obj, "tiling");
    cJSON_AddNumberToObject(tiling, "n_row_blocks", p->n_row_blocks);
    cJSON_AddNumberToObject(tiling, "n_col_blocks", p->n_col_blocks);
    cJSON_AddNumberToObject(tiling, "gemm_tile_invocations",
                            p->gemm_tile_invocations);

    cJSON *timing = cJSON_AddObjectToObject(obj, "timing_seconds");
    cJSON_AddNumberToObject(timing, "host_prepare", p->host_prepare_seconds);
    cJSON_AddNumberToObject(timing, "host_pack_quantize",
                            p->host_pack_quantize_seconds);
    cJSON_AddNumberToObject(timing, "h2d_dma", p->h2d_dma_seconds);
    cJSON_AddNumberToObject(timing, "dpu_kernel", p->dpu_kernel_seconds);
    cJSON_AddNumberToObject(timing, "d2h_dma", p->d2h_dma_seconds);
    cJSON_AddNumberToObject(timing, "host_dequantize_accumulate",
                            p->host_dequantize_accumulate_seconds);
    cJSON_AddNumberToObject(timing, "host_reduce", 0.0);
    cJSON_AddNumberToObject(timing, "total", p->total_seconds);

    cJSON *bytes = cJSON_AddObjectToObject(obj, "bytes");
    cJSON_AddNumberToObject(bytes, "host_to_dpu", (double)p->host_to_dpu_bytes);
    cJSON_AddNumberToObject(bytes, "dpu_to_host", (double)p->dpu_to_host_bytes);
}

static void write_execution_log(const char *log_path, const char *json_path,
                                const char *bin_path, const char *output_path,
                                const cJSON *task_graph,
                                const TaskProfile *profiles, int n_profiles,
                                double total_seconds, int n_output_elements) {
    cJSON *log = cJSON_CreateObject();
    if (!log) die("failed to allocate execution log JSON");

    cJSON_AddStringToObject(log, "schema_version", "mvp_execution_log-0.2");
    cJSON_AddStringToObject(log, "baseline_role", "RawUPMEMProvider_baseline");
    cJSON_AddStringToObject(log, "selected_route", "raw_upmem_dense");
    cJSON_AddStringToObject(log, "selected_format", "complex_i8_tile_scaled");

    cJSON *paths = cJSON_AddObjectToObject(log, "paths");
    cJSON_AddStringToObject(paths, "task_graph", json_path);
    cJSON_AddStringToObject(paths, "tensor_data", bin_path);
    cJSON_AddStringToObject(paths, "output_amplitudes", output_path);

    cJSON *graph_schema = cJSON_GetObjectItem(task_graph, "schema_version");
    if (graph_schema && graph_schema->valuestring) {
        cJSON_AddStringToObject(log, "task_graph_schema_version",
                                graph_schema->valuestring);
    } else {
        cJSON_AddStringToObject(log, "task_graph_schema_version", "legacy_mvp");
    }

    cJSON *runtime = cJSON_AddObjectToObject(log, "runtime");
    cJSON_AddNumberToObject(runtime, "dpu_count", 1);
    cJSON_AddNumberToObject(runtime, "tile_rows", TILE_ROWS);
    cJSON_AddNumberToObject(runtime, "tile_k", TILE_K);
    cJSON_AddNumberToObject(runtime, "tile_n", TILE_N);
    cJSON_AddNumberToObject(runtime, "nr_tasklets", NR_TASKLETS);
    cJSON_AddNumberToObject(runtime, "n_output_elements", n_output_elements);

    double total_h2d = 0.0;
    double total_dpu = 0.0;
    double total_d2h = 0.0;
    double total_prepare = 0.0;
    double total_pack = 0.0;
    double total_accumulate = 0.0;
    size_t bytes_h2d = 0;
    size_t bytes_d2h = 0;
    int gemm_calls = 0;
    for (int i = 0; i < n_profiles; i++) {
        total_h2d += profiles[i].h2d_dma_seconds;
        total_dpu += profiles[i].dpu_kernel_seconds;
        total_d2h += profiles[i].d2h_dma_seconds;
        total_prepare += profiles[i].host_prepare_seconds;
        total_pack += profiles[i].host_pack_quantize_seconds;
        total_accumulate += profiles[i].host_dequantize_accumulate_seconds;
        bytes_h2d += profiles[i].host_to_dpu_bytes;
        bytes_d2h += profiles[i].dpu_to_host_bytes;
        gemm_calls += profiles[i].gemm_tile_invocations;
    }

    cJSON *summary = cJSON_AddObjectToObject(log, "summary");
    cJSON_AddNumberToObject(summary, "n_tasks", n_profiles);
    cJSON_AddNumberToObject(summary, "gemm_tile_invocations", gemm_calls);
    cJSON_AddNumberToObject(summary, "total_wall_seconds", total_seconds);
    cJSON_AddNumberToObject(summary, "host_prepare_seconds", total_prepare);
    cJSON_AddNumberToObject(summary, "host_pack_quantize_seconds", total_pack);
    cJSON_AddNumberToObject(summary, "h2d_dma_seconds", total_h2d);
    cJSON_AddNumberToObject(summary, "dpu_kernel_seconds", total_dpu);
    cJSON_AddNumberToObject(summary, "d2h_dma_seconds", total_d2h);
    cJSON_AddNumberToObject(summary, "host_dequantize_accumulate_seconds",
                            total_accumulate);
    cJSON_AddNumberToObject(summary, "host_to_dpu_bytes", (double)bytes_h2d);
    cJSON_AddNumberToObject(summary, "dpu_to_host_bytes", (double)bytes_d2h);

    cJSON *routes = cJSON_AddArrayToObject(log, "route_decisions");
    for (int i = 0; i < n_profiles; i++) {
        cJSON *route = cJSON_CreateObject();
        cJSON_AddItemToArray(routes, route);
        cJSON_AddNumberToObject(route, "task_id", profiles[i].task_id);
        cJSON_AddStringToObject(route, "selected_route", "raw_upmem_dense");
        cJSON_AddStringToObject(route, "selected_format",
                                "complex_i8_tile_scaled");
        cJSON_AddStringToObject(route, "decision_policy", "mvp_fixed_route");
        cJSON_AddStringToObject(route, "decision_reason",
                                "conference_mvp_validation_path");
    }

    cJSON *profile_array = cJSON_AddArrayToObject(log, "profiles");
    for (int i = 0; i < n_profiles; i++) {
        add_profile_json(profile_array, &profiles[i]);
    }

    char *serialized = cJSON_Print(log);
    if (!serialized) die("failed to serialize execution log");
    FILE *lf = fopen(log_path, "w");
    if (!lf) {
        perror(log_path);
        free(serialized);
        cJSON_Delete(log);
        exit(1);
    }
    fputs(serialized, lf);
    fputc('\n', lf);
    fclose(lf);

    free(serialized);
    cJSON_Delete(log);
}

int main(int argc, char **argv) {
    const char *json_path = (argc > 1) ? argv[1] : "data_exchange/task_graph.json";
    const char *bin_path = (argc > 2) ? argv[2] : "data_exchange/tensor_data.bin";
    const char *output_path = (argc > 3) ? argv[3] : "data_exchange/output_amplitudes.bin";
    const char *log_path = (argc > 4) ? argv[4] : "data_exchange/execution_log.json";

    FILE *jf = fopen(json_path, "rb");
    if (!jf) {
        perror(json_path);
        return 1;
    }
    fseek(jf, 0, SEEK_END);
    long jlen = ftell(jf);
    rewind(jf);
    char *jbuf = (char *)malloc((size_t)jlen + 1);
    if (!jbuf) die("malloc failed for JSON buffer");
    if (fread(jbuf, 1, (size_t)jlen, jf) != (size_t)jlen) die("failed to read JSON");
    fclose(jf);
    jbuf[jlen] = '\0';

    cJSON *root = cJSON_Parse(jbuf);
    free(jbuf);
    if (!root) {
        fprintf(stderr, "JSON parse error near: %s\n", cJSON_GetErrorPtr());
        return 1;
    }

    FILE *bf = fopen(bin_path, "rb");
    if (!bf) {
        perror(bin_path);
        return 1;
    }

    cJSON *init_list = json_get(root, "initial_tensors");
    int n_init = cJSON_GetArraySize(init_list);
    for (int t = 0; t < n_init; t++) {
        cJSON *it = cJSON_GetArrayItem(init_list, t);
        const char *key = json_string(it, "key");
        int n_elem = json_int(it, "n_elements");
        long off_real = (long)json_get(it, "offset_real_bytes")->valuedouble;
        long off_imag = (long)json_get(it, "offset_imag_bytes")->valuedouble;

        int shape[MAX_DIMS], labels[MAX_DIMS];
        int n_dims = read_int_array(it, "shape", shape, MAX_DIMS);
        int n_labels = read_int_array(it, "labels", labels, MAX_DIMS);
        if (n_dims != n_labels) die("initial tensor shape/labels length mismatch");

        TensorEntry *e = registry_insert(key, n_elem, shape, labels, n_dims);
        fseek(bf, off_real, SEEK_SET);
        if (fread(e->real, sizeof(double), (size_t)n_elem, bf) != (size_t)n_elem) {
            die("failed to read real tensor data");
        }
        fseek(bf, off_imag, SEEK_SET);
        if (fread(e->imag, sizeof(double), (size_t)n_elem, bf) != (size_t)n_elem) {
            die("failed to read imag tensor data");
        }
    }
    fclose(bf);

    cJSON *tasks = json_get(root, "tasks");
    int n_tasks = cJSON_GetArraySize(tasks);
    if (n_tasks <= 0) die("No contraction tasks in plan");
    if (n_tasks > MAX_TENSORS) die("Too many tasks for MVP profile buffer");

    PimContext pim;
    DPU_ASSERT(dpu_alloc(1, NULL, &pim.set));
    DPU_ASSERT(dpu_load(pim.set, "bin/dpu_gemm_int8", NULL));
    DPU_FOREACH(pim.set, pim.dpu) {
        break;
    }

    double t_total_start = wall_seconds();
    TaskProfile profiles[MAX_TENSORS];
    memset(profiles, 0, sizeof(profiles));

    for (int s = 0; s < n_tasks; s++) {
        cJSON *task = cJSON_GetArrayItem(tasks, s);
        printf("[step %d/%d] Contracting %s x %s -> %s ...\n",
               s + 1, n_tasks,
               json_string(task, "input_A_key"),
               json_string(task, "input_B_key"),
               json_string(task, "output_key"));
        execute_task(&pim, task, &profiles[s]);
    }

    double t_total = wall_seconds() - t_total_start;
    const char *final_key = json_string(cJSON_GetArrayItem(tasks, n_tasks - 1),
                                        "output_key");
    TensorEntry *final_t = registry_get(final_key);
    if (!final_t) die("Final tensor missing from registry");

    FILE *of = fopen(output_path, "wb");
    if (!of) {
        perror(output_path);
        return 1;
    }
    int32_t n_out = final_t->n_elements;
    fwrite(&n_out, sizeof(int32_t), 1, of);
    fwrite(final_t->real, sizeof(double), (size_t)n_out, of);
    fwrite(final_t->imag, sizeof(double), (size_t)n_out, of);
    fclose(of);

    double t_dma_out = 0.0;
    double t_dpu = 0.0;
    double t_dma_in = 0.0;
    double t_prepare = 0.0;
    double t_pack = 0.0;
    double t_accumulate = 0.0;
    size_t bytes_dma_out = 0;
    size_t bytes_dma_in = 0;
    int gemm_calls = 0;
    for (int s = 0; s < n_tasks; s++) {
        t_dma_out += profiles[s].h2d_dma_seconds;
        t_dpu += profiles[s].dpu_kernel_seconds;
        t_dma_in += profiles[s].d2h_dma_seconds;
        t_prepare += profiles[s].host_prepare_seconds;
        t_pack += profiles[s].host_pack_quantize_seconds;
        t_accumulate += profiles[s].host_dequantize_accumulate_seconds;
        bytes_dma_out += profiles[s].host_to_dpu_bytes;
        bytes_dma_in += profiles[s].dpu_to_host_bytes;
        gemm_calls += profiles[s].gemm_tile_invocations;
    }

    write_execution_log(log_path, json_path, bin_path, output_path, root,
                        profiles, n_tasks, t_total, n_out);

    printf("\n=== Performance Report ===\n");
    printf("Total wall time : %.6f s\n", t_total);
    printf("Host prepare    : %.6f s  (%5.1f%%)\n", t_prepare, 100.0 * t_prepare / t_total);
    printf("Host pack/quant : %.6f s  (%5.1f%%)\n", t_pack, 100.0 * t_pack / t_total);
    printf("DMA host->DPU   : %.6f s  (%5.1f%%)\n", t_dma_out, 100.0 * t_dma_out / t_total);
    printf("DPU map phase   : %.6f s  (%5.1f%%)\n", t_dpu, 100.0 * t_dpu / t_total);
    printf("DMA DPU->host   : %.6f s  (%5.1f%%)\n", t_dma_in, 100.0 * t_dma_in / t_total);
    printf("Host dequant/acc: %.6f s  (%5.1f%%)\n", t_accumulate, 100.0 * t_accumulate / t_total);
    printf("GEMM tile calls : %d\n", gemm_calls);
    printf("Bytes host->DPU : %zu\n", bytes_dma_out);
    printf("Bytes DPU->host : %zu\n", bytes_dma_in);
    printf("Output written to %s\n", output_path);
    printf("Execution log   : %s\n", log_path);
    printf("Output n_elements: %d\n", n_out);

    DPU_ASSERT(dpu_free(pim.set));
    cJSON_Delete(root);
    return 0;
}
