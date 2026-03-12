#include <mram.h>
#include <stdint.h>
#include <stdio.h>

#define SIZE 8

// TYLKO __mram_noinit. To wystarczy, by tablica trafiła do wielkiego MRAM-u, 
// a Host i tak ją znajdzie dzięki samej nazwie "my_array".
__mram_noinit uint32_t my_array[SIZE];

int main() {
    // Nasz koszyk (wymuszamy wyrównanie do 8 bajtów, co jest wymagane przez transfer MRAM<->WRAM)
    uint32_t cache[SIZE] __attribute__((aligned(8)));

    // 1. Z dużej (MRAM) do koszyka (WRAM)
    mram_read(my_array, cache, sizeof(cache));

    // 2. Mnożenie!
    for (int i = 0; i < SIZE; i++) {
        printf("DPU (przed): %d ", cache[i]);
        cache[i] = cache[i] * 2 + 1;
        printf("-> (po): %d\n", cache[i]);
    }

    // 3. Z koszyka z powrotem do dużej (MRAM)
    mram_write(cache, my_array, sizeof(cache));

    return 0;
}
