/* Test-only pair harness. Default diagnostics are strictly SDK simulator. */
#define _POSIX_C_SOURCE 200809L
#include <dpu.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

#ifdef __FAST_MATH__
#error "resident probe requires strict float32 and signed-zero arithmetic"
#endif

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

static volatile sig_atomic_t interrupted;
static void stop_pair(int signal_number) { interrupted = signal_number; }

static double now_s(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) return -1.0;
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static int rank_owned(void) {
    int owned = -1;
    FILE *stream = fopen("/sys/class/dpu_rank/dpu_rank1/is_owned", "r");
    if (stream != NULL) {
        if (fscanf(stream, "%d", &owned) != 1) owned = -1;
        fclose(stream);
    }
    return owned;
}

static int measure_pair(int argc, char **argv) {
    probe_request_t request = {0};
    resident_probe_plan_t saved;
    uint8_t *products = NULL;
    struct dpu_set_t set, dpu, rank;
    struct dpu_program_t *program = NULL;
    struct dpu_symbol_t symbol;
    struct stat device;
    struct sigaction action = {0};
    upmem_wave_completion_t completions[2];
    uint32_t dpus = 0, ranks = 0, enumerated = 0;
    uint64_t h2d_bytes = 0, d2h_bytes = 0;
    unsigned h2d_calls = 0, d2h_calls = 0, launches = 0;
    double opened = 0, pair = 0, closed = 0, h2d = 0, d2h = 0, launch = 0;
    double started = now_s(), section, pair_started = 0;
    int allocated = 0, released = 0, status = 5, hardware = argc == 6;
    int resident = strcmp(argv[3], "resident") == 0;
    const char *profile = "backend=simulator";
    const char *allow = getenv("UPMEM_ALLOW_PHYSICAL_HARDWARE");
    if ((!resident && strcmp(argv[3], "host_roundtrip") != 0) || started < 0) return 2;
    if (hardware) {
        /* Same explicit selector and allocation/resource pattern as simplepim_provider.c.
           This probe has no SimplePIM tables and needs no management-init launch. */
        if (strcmp(argv[4], "--rank-path") != 0 || strcmp(argv[5], "/dev/dpu_rank1") != 0 ||
                allow == NULL || strcmp(allow, "1") != 0 || getenv("DPU_BACKEND") != NULL ||
                getenv("UPMEM_EXECUTION_MODE") != NULL || getenv("UPMEM_REQUIRE_SDK_SIMULATOR") != NULL ||
                stat(argv[5], &device) != 0 || !S_ISCHR(device.st_mode)) return 2;
        profile = "backend=hw,rankPath=/dev/dpu_rank1";
    }
    if (read_request(&request) != 1) return 2;
    /* Fixed Stress16 contract121->122 corpus, T8, right resident operand; no int8. */
    if (request.command != RESIDENT_FIRST || !resident_probe_plan_valid(&request.plan, 8) ||
            request.plan.resident_side != 1 || request.patch_offset != 0 || request.patch_length != 93184 ||
            request.plan.controls[0].m != 16 || request.plan.controls[0].n != 64 || request.plan.controls[0].k != 4 ||
            request.plan.controls[1].m != 16 || request.plan.controls[1].n != 256 || request.plan.controls[1].k != 4)
        { free(request.patch); return 2; }
    saved = request.plan;
    for (unsigned phase = 0; phase < 2; ++phase)
        for (unsigned plane = 0; plane < 8; ++plane) {
            upmem_wave_span_t span = saved.controls[phase].planes[plane];
            if (span.offset > request.patch_length || span.length > request.patch_length - span.offset)
                { free(request.patch); return 2; }
        }
    products = malloc(4 * saved.controls[1].planes[4].length);
    if (products == NULL) { free(request.patch); return 3; }
    /* No SA_RESTART: interrupts must unblock stdio between the two launches. */
    sigemptyset(&action.sa_mask);
    action.sa_handler = stop_pair;
    if (sigaction(SIGINT, &action, NULL) != 0 || sigaction(SIGTERM, &action, NULL) != 0)
        goto cleanup_pair;
    action.sa_handler = SIG_IGN;
    if (sigaction(SIGPIPE, &action, NULL) != 0) goto cleanup_pair;
    if (interrupted || dpu_alloc(1, profile, &set) != DPU_OK) goto cleanup_pair;
    allocated = 1;
    if (dpu_get_nr_dpus(set, &dpus) != DPU_OK || dpu_get_nr_ranks(set, &ranks) != DPU_OK)
        goto cleanup_pair;
    DPU_RANK_FOREACH(set, rank) { (void)rank; ++enumerated; }
    if (dpus != 1 || ranks != 1 || enumerated != ranks || (hardware && rank_owned() != 1)) goto cleanup_pair;
    if (dpu_load(set, argv[1], &program) != DPU_OK) goto cleanup_pair;
    const char *names[] = {"PROBE_PLAN", "PROBE_COMMAND", "PROBE_COMPLETION", "PROBE_MRAM"};
    const uint32_t sizes[] = {sizeof(request.plan), sizeof(request.command), sizeof(completions[0]), UPMEM_WAVE_MRAM_BYTES};
    for (unsigned i = 0; i < 4; ++i)
        if (dpu_get_symbol(program, names[i], &symbol) != DPU_OK || symbol.size != sizes[i]) goto cleanup_pair;
    DPU_FOREACH(set, dpu) { break; }
    opened = now_s() - started;
    pair_started = now_s();
    for (unsigned phase = 0; phase < 2; ++phase) {
        if (interrupted) goto cleanup_pair;
        section = now_s();
        if (dpu_copy_to(dpu, "PROBE_PLAN", 0, &request.plan, sizeof(request.plan)) != DPU_OK) goto cleanup_pair;
        h2d_bytes += sizeof(request.plan); ++h2d_calls;
        if (dpu_copy_to(dpu, "PROBE_COMMAND", 0, &request.command, sizeof(request.command)) != DPU_OK) goto cleanup_pair;
        h2d_bytes += sizeof(request.command); ++h2d_calls;
        if (phase == 0 || !resident) {
            uint32_t offset = request.patch_offset, bytes = request.patch_length;
            if (dpu_copy_to(dpu, "PROBE_MRAM", offset, request.patch, bytes) != DPU_OK) goto cleanup_pair;
            h2d_bytes += bytes; ++h2d_calls;
        }
        h2d += now_s() - section;
        section = now_s();
        ++launches;
        if (dpu_launch(set, DPU_SYNCHRONOUS) != DPU_OK) goto cleanup_pair;
        launch += now_s() - section;
        section = now_s();
        if (dpu_copy_from(dpu, "PROBE_COMPLETION", 0, &completions[phase], sizeof(completions[phase])) != DPU_OK) goto cleanup_pair;
        d2h_bytes += sizeof(completions[phase]); ++d2h_calls;
        if (!upmem_wave_completion_success(&completions[phase], &request.plan.controls[phase])) goto cleanup_pair;
        if (phase == 1 || !resident) {
            upmem_wave_span_t span = request.plan.controls[phase].planes[4];
            if (dpu_copy_from(dpu, "PROBE_MRAM", span.offset, products, 4 * span.length) != DPU_OK) goto cleanup_pair;
            d2h_bytes += 4 * span.length; ++d2h_calls;
        }
        d2h += now_s() - section;
        uint32_t bytes = phase == 1 || !resident ? 4 * saved.controls[phase].planes[4].length : 0;
        if (fwrite(&completions[phase], 1, sizeof(completions[phase]), stdout) != sizeof(completions[phase]) ||
                (bytes && fwrite(products, 1, bytes, stdout) != bytes) || fflush(stdout) != 0) goto cleanup_pair;
        if (phase == 0) {
            free(request.patch); request.patch = NULL;
            if (read_request(&request) != 1 || interrupted ||
                    !resident_probe_plan_valid(&request.plan, 8) || memcmp(&saved, &request.plan, sizeof(saved)) != 0 ||
                    request.command != (resident ? RESIDENT_LOCAL_SECOND : RESIDENT_HOST_SECOND) ||
                    request.patch_offset != (resident ? 0 : saved.retained[0].offset) ||
                    request.patch_length != (resident ? 0 : 2 * saved.retained[0].length)) goto cleanup_pair;
        }
    }
    pair = now_s() - pair_started;
    status = interrupted || fgetc(stdin) != EOF || !feof(stdin) ? 5 : 0;
cleanup_pair:
    section = now_s();
    if (allocated) {
        released = dpu_free(set) == DPU_OK;
        if (!released || (hardware && rank_owned() != 0)) { released = 0; status = 6; }
    }
    closed = now_s() - section;
    fprintf(stderr, "{\"schema\":\"resident_pair_measurement_v1\",\"status\":%d,\"physical\":%s,"
        "\"arm\":\"%s\",\"pair_id\":%llu,\"dpus\":%u,\"ranks\":%u,\"tasklets\":8,"
        "\"allocated\":%d,\"release_verified\":%d,\"fallback\":false,\"launches\":%u,"
        "\"h2d_bytes\":%llu,\"d2h_bytes\":%llu,\"h2d_calls\":%u,\"d2h_calls\":%u,"
        "\"session_open_s\":%.17g,\"pair_s\":%.17g,\"session_close_s\":%.17g,"
        "\"h2d_s\":%.17g,\"d2h_s\":%.17g,\"launch_wall_s\":%.17g,\"attempt_wall_s\":%.17g}\n",
        status, hardware ? "true" : "false", argv[3], (unsigned long long)saved.pair_id,
        dpus, ranks, allocated, released, launches, (unsigned long long)h2d_bytes,
        (unsigned long long)d2h_bytes, h2d_calls, d2h_calls, opened, pair, closed,
        h2d, d2h, launch, now_s() - started);
    free(request.patch);
    free(products);
    return status;
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
    if ((argc == 4 || argc == 6) && strcmp(argv[2], "--measure") == 0)
        return measure_pair(argc, argv);
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
                fwrite(arena, 1, UPMEM_WAVE_MRAM_BYTES, stdout) != UPMEM_WAVE_MRAM_BYTES ||
                fflush(stdout) != 0)
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
