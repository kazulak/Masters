/* SDK-simulator test harness only; never allocates a physical backend. */
#include <dpu.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "upmem_resident_probe.h"

typedef struct {
    uint32_t command;
    resident_probe_plan_t plan;
    uint32_t patch_offset;
    uint32_t patch_length;
    uint8_t *patch;
} probe_request_t;

static int read_exact(void *destination, size_t bytes) {
    return fread(destination, 1, bytes, stdin) == bytes;
}

static int read_request(probe_request_t *request) {
    size_t count = fread(&request->command, 1, sizeof(request->command), stdin);
    if (count == 0u) return feof(stdin) ? 0 : -1;
    if (count != sizeof(request->command) ||
            !read_exact(&request->plan, sizeof(request->plan)) ||
            !read_exact(&request->patch_offset, sizeof(request->patch_offset)) ||
            !read_exact(&request->patch_length, sizeof(request->patch_length))) return -1;
    request->patch = NULL;
    if (request->patch_offset % 8u != 0u || request->patch_length % 8u != 0u ||
            request->patch_offset > UPMEM_WAVE_MRAM_BYTES ||
            request->patch_length > UPMEM_WAVE_MRAM_BYTES - request->patch_offset) return -1;
    if (request->patch_length != 0u) {
        request->patch = malloc(request->patch_length);
        if (request->patch == NULL || !read_exact(request->patch, request->patch_length)) {
            free(request->patch);
            request->patch = NULL;
            return -1;
        }
    }
    return 1;
}

int main(int argc, char **argv) {
    probe_request_t request = {0};
    struct dpu_set_t set;
    struct dpu_set_t dpu;
    upmem_wave_completion_t completion;
    uint8_t *arena = NULL;
    int status = 5;
    int allocated = 0;
    int result;
    if (argc != 2) return 2;
    result = read_request(&request);
    if (result < 0) return 2;
    if (result == 0) return 0;

    arena = malloc(UPMEM_WAVE_MRAM_BYTES);
    if (arena == NULL) {
        free(request.patch);
        return 3;
    }
    if (dpu_alloc(1, "backend=simulator", &set) != DPU_OK) goto cleanup;
    allocated = 1;
    if (dpu_load(set, argv[1], NULL) != DPU_OK) goto cleanup;
    DPU_FOREACH(set, dpu) { break; }

    for (;;) {
        if (dpu_copy_to(dpu, "PROBE_PLAN", 0, &request.plan, sizeof(request.plan)) != DPU_OK ||
                dpu_copy_to(dpu, "PROBE_COMMAND", 0, &request.command,
                    sizeof(request.command)) != DPU_OK ||
                (request.patch_length != 0u &&
                 dpu_copy_to(dpu, "PROBE_MRAM", request.patch_offset, request.patch,
                    request.patch_length) != DPU_OK) ||
                dpu_launch(set, DPU_SYNCHRONOUS) != DPU_OK ||
                dpu_copy_from(dpu, "PROBE_COMPLETION", 0, &completion,
                    sizeof(upmem_wave_completion_t)) != DPU_OK ||
                dpu_copy_from(dpu, "PROBE_MRAM", 0, arena,
                    UPMEM_WAVE_MRAM_BYTES) != DPU_OK)
            goto cleanup;
        if (fwrite(&completion, 1, sizeof(completion), stdout) !=
                    sizeof(upmem_wave_completion_t) ||
                fwrite(arena, 1, UPMEM_WAVE_MRAM_BYTES, stdout) != UPMEM_WAVE_MRAM_BYTES)
            goto cleanup;
        free(request.patch);
        request.patch = NULL;
        result = read_request(&request);
        if (result == 0) {
            status = 0;
            break;
        }
        if (result < 0) goto cleanup;
    }

cleanup:
    free(request.patch);
    if (arena != NULL) free(arena);
    if (allocated) {
        /* dpu_free is intentionally reached after every loaded simulator session. */
        if (dpu_free(set) != DPU_OK) status = 6;
    }
    return status;
}
