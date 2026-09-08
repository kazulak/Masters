/* Software-only host lifecycle stub. No SDK linkage, devices, or computation. */
#ifndef RESIDENT_MOCK_DPU_H
#define RESIDENT_MOCK_DPU_H
#include <stdio.h>
#include <string.h>
#include "upmem_resident_probe.h"
#define DPU_OK 0
#define DPU_SYNCHRONOUS 0
struct dpu_set_t { int unused; };
struct dpu_program_t { int unused; };
struct dpu_symbol_t { uint32_t size; };
#define DPU_FOREACH(set, dpu) for (int once = ((dpu) = (set), 1); once; once = 0)
#define DPU_RANK_FOREACH(set, rank) DPU_FOREACH(set, rank)
static resident_probe_plan_t mock_plan;
static uint32_t mock_command;
static int dpu_alloc(unsigned count, const char *profile, struct dpu_set_t *set) {
    if (count != 1 || strcmp(profile, "backend=simulator") != 0) return 1;
    set->unused = 0;
    fprintf(stderr, "MOCK_ALLOC\n");
    return DPU_OK;
}
static int dpu_get_nr_dpus(struct dpu_set_t set, uint32_t *count) {
    (void)set; *count = 1; return DPU_OK;
}
static int dpu_get_nr_ranks(struct dpu_set_t set, uint32_t *count) {
    return dpu_get_nr_dpus(set, count);
}
static int dpu_load(struct dpu_set_t set, const char *path, struct dpu_program_t **program) {
    static struct dpu_program_t value;
    (void)set; (void)path; if (program) *program = &value; return DPU_OK;
}
static int dpu_get_symbol(struct dpu_program_t *program, const char *name, struct dpu_symbol_t *symbol) {
    (void)program;
    symbol->size = strcmp(name, "PROBE_PLAN") == 0 ? sizeof(mock_plan) :
        strcmp(name, "PROBE_COMMAND") == 0 ? 4 :
        strcmp(name, "PROBE_COMPLETION") == 0 ? 72 : UPMEM_WAVE_MRAM_BYTES;
    return DPU_OK;
}
static int dpu_copy_to(struct dpu_set_t set, const char *name, unsigned offset, const void *data, size_t bytes) {
    (void)set; (void)offset;
    if (strcmp(name, "PROBE_PLAN") == 0) memcpy(&mock_plan, data, bytes);
    if (strcmp(name, "PROBE_COMMAND") == 0) memcpy(&mock_command, data, bytes);
    return DPU_OK;
}
static int dpu_launch(struct dpu_set_t set, int mode) {
    (void)set; (void)mode; return DPU_OK;
}
static int dpu_copy_from(struct dpu_set_t set, const char *name, unsigned offset, void *data, size_t bytes) {
    (void)set; (void)offset;
    if (strcmp(name, "PROBE_COMPLETION") == 0) {
        const upmem_wave_control_t *c = &mock_plan.controls[mock_command == RESIDENT_FIRST ? 0 : 1];
        upmem_wave_completion_t reply = {
            .magic = UPMEM_WAVE_COMPLETION_MAGIC, .version = UPMEM_WAVE_VERSION,
            .status = UPMEM_WAVE_COMPLETED, .dpu_id = c->dpu_id,
            .operation_index = c->operation_index, .completed_product_mask = 15,
            .wave_id = c->wave_id, .request_sequence = c->request_sequence,
            .tile_id = c->tile_id, .processed_elements = (uint64_t)c->m * c->n * 4,
            .failing_product = UPMEM_WAVE_NO_PRODUCT
        };
        memcpy(data, &reply, bytes);
    } else {
        memset(data, 0, bytes);
    }
    return DPU_OK;
}
static int dpu_free(struct dpu_set_t set) {
    (void)set; fprintf(stderr, "MOCK_FREE\n"); return DPU_OK;
}
#endif
