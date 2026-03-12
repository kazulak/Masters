#include <dpu.h>
#include <stdio.h>
#include <stdint.h>

#define DPU_BINARY "./dpu_math"
#define SIZE 8

int main() {
    struct dpu_set_t set, dpu;
    
    // Tablica testowa na Twoim PC (w procesorze Ryzen)
    uint32_t host_array[SIZE] = {2^1, 2^2, 3, 4, 5, 6, 7, 128};
    uint32_t result_array[SIZE] = {0}; // Tutaj odbierzemy wyniki

    // 1. Alokacja (z argumentem wyciszającym błędy braku sprzętu)
    DPU_ASSERT(dpu_alloc(1, "backend=simulator", &set));
    DPU_ASSERT(dpu_load(set, DPU_BINARY, NULL));

    printf("[HOST] Wysylam tablice do RAM: ");
    for(int i=0; i<SIZE; i++) printf("%d ", host_array[i]);
    printf("\n");

    // 2. Kopiowanie danych PC -> RAM (do zmiennej '__host my_array')
    DPU_FOREACH(set, dpu) {
        DPU_ASSERT(dpu_copy_to(dpu, "my_array", 0, host_array, sizeof(host_array)));
    }

    printf("[HOST] Uruchamiam obliczenia na DPU...\n");
    // 3. Start!
    DPU_ASSERT(dpu_launch(set, DPU_SYNCHRONOUS));

    // Odczyt printów z wewnątrz DPU (opcjonalne, ale fajne przy nauce)
    printf("--- Logi z wnetrza DPU ---\n");
    DPU_FOREACH(set, dpu) {
        DPU_ASSERT(dpu_log_read(dpu, stdout));
    }
    printf("--------------------------\n");

    // 4. Kopiowanie wyników RAM -> PC
    DPU_FOREACH(set, dpu) {
        DPU_ASSERT(dpu_copy_from(dpu, "my_array", 0, result_array, sizeof(result_array)));
    }

    printf("[HOST] Otrzymany wynik z RAM:  ");
    for(int i=0; i<SIZE; i++) printf("%d ", result_array[i]);
    printf("\n");

    DPU_ASSERT(dpu_free(set));
    return 0;
}
