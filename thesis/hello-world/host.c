#include <dpu.h>
#include <stdio.h>

// Wskazujemy, jak nazywa sie skompilowany plik dla DPU
#define DPU_BINARY "./dpu_task"

int main() {
    struct dpu_set_t set, dpu;

    // 1. Alokujemy (wynajmujemy) 1 rdzen DPU
    DPU_ASSERT(dpu_alloc(1, "backend=simulator", &set));

    // 2. Ladowanie skompilowanego programu na DPU
    DPU_ASSERT(dpu_load(set, DPU_BINARY, NULL));

    printf("[HOST] DPU zaalokowane. Odpalamy zaplon...\n");

    // 3. Uruchomienie DPU (Host czeka, az DPU skonczy prace)
    DPU_ASSERT(dpu_launch(set, DPU_SYNCHRONOUS));

    // 4. Odczytanie logow (printow) z wnetrza DPU na ekran Hosta
    printf("[HOST] Wyniki z pamieci RAM:\n");
    printf("------------------------------------------\n");
    DPU_FOREACH(set, dpu) {
        DPU_ASSERT(dpu_log_read(dpu, stdout));
    }
    printf("------------------------------------------\n");

    // 5. Zwalniamy zasoby
    DPU_ASSERT(dpu_free(set));
    printf("[HOST] Koniec pracy. DPU zwolnione.\n");

    return 0;
}
