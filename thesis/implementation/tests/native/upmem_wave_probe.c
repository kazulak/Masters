/* SDK-simulator test harness only; never allocates a physical backend. */
#include <dpu.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "wave_protocol.h"

int main(int argc, char **argv) {
    struct dpu_set_t set;
    if (argc != 2) return 2;
    uint8_t *arena = malloc(UPMEM_WAVE_MRAM_BYTES);
    if (arena == NULL) return 3;
    if (dpu_alloc(1, "backend=simulator", &set) != DPU_OK) {
        free(arena);
        return 4;
    }
    int status = 5;
    if (dpu_load(set, argv[1], NULL) != DPU_OK) goto cleanup;
    struct dpu_set_t dpu;
    DPU_FOREACH(set, dpu) { break; }
    for (;;) {
        upmem_wave_control_t control;
        upmem_wave_completion_t completion;
        size_t count = fread(&control, 1, sizeof(control), stdin);
        if (count == 0 && feof(stdin)) { status = 0; break; }
        if (count != sizeof(control) ||
                fread(arena, 1, UPMEM_WAVE_MRAM_BYTES, stdin) != UPMEM_WAVE_MRAM_BYTES)
            goto cleanup;
        /* Intentionally bypass host admission to test tasklet-zero validation. */
        if (dpu_copy_to(dpu, "WAVE_CONTROL", 0, &control, sizeof(control)) != DPU_OK ||
                dpu_copy_to(dpu, "WAVE_MRAM", 0, arena, UPMEM_WAVE_MRAM_BYTES) != DPU_OK ||
                dpu_launch(set, DPU_SYNCHRONOUS) != DPU_OK ||
                dpu_copy_from(dpu, "WAVE_COMPLETION", 0, &completion, sizeof(completion)) != DPU_OK ||
                dpu_copy_from(dpu, "WAVE_MRAM", 0, arena, UPMEM_WAVE_MRAM_BYTES) != DPU_OK)
            goto cleanup;
        if (fwrite(&completion, 1, sizeof(completion), stdout) != sizeof(completion) ||
                fwrite(arena, 1, UPMEM_WAVE_MRAM_BYTES, stdout) != UPMEM_WAVE_MRAM_BYTES)
            goto cleanup;
    }
cleanup:
    if (dpu_free(set) != DPU_OK) status = 6;
    free(arena);
    return status;
}
