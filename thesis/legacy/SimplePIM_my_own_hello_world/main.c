#include <stdio.h>
#include <stdlib.h>
#include <dpu.h>
#include <string.h>

#include "management/Management.h"
#include "processing/map/Map.h"
#include "communication/CommOps.h"
#include "processing/ProcessingHelperHost.h"
#include "Param.h"

typedef struct {
    int task_id;
    int operation_type; // 1 = Multiply, 2 = Add
    uint32_t seed_value; 
} Task;

int main() {
    printf("[ORCHESTRATOR] Booting System...\n");

    printf("[ORCHESTRATOR] Claiming DPUs...\n");
    simplepim_management_t* m = table_management_init(dpu_number);

    printf("[ORCHESTRATOR] Pre-compiling Kernel Handles...\n");
    handle_t* handle_multiply = create_handle("kernels/multiply", MAP);
    handle_t* handle_add      = create_handle("kernels/add", MAP);

    printf("[ORCHESTRATOR] Allocating Reusable DMA Buffer...\n");
    T* dma_buffer = (T*)malloc_scatter_aligned(nr_elements, sizeof(T), m);

    Task queue[4] = {
        {101, 1, 10},  // Multiply, Seed 10
        {102, 2, 50},  // Add, Seed 50
        {103, 2, 80},  // Add, Seed 80
        {104, 1, 25}   // Multiply, Seed 25
    };

    printf("\n[ORCHESTRATOR] Beginning Task Processing...\n");
    int num_tasks = 4;

    for (int t = 0; t < num_tasks; t++) {
        Task current_task = queue[t];
        printf("\n--- Processing Task ID: %d ---\n", current_task.task_id);

        // 1. Generate unique table names for this specific task
        char src_id[32], dst_id[32];
        sprintf(src_id, "data_%d", current_task.task_id);
        sprintf(dst_id, "res_%d", current_task.task_id);

        // 2. Load data into the reusable buffer
        for (uint64_t i = 0; i < nr_elements; i++) {
            dma_buffer[i] = i + current_task.seed_value;
        }

        // 3. Scatter using the UNIQUE source ID
        simplepim_scatter(src_id, dma_buffer, nr_elements, sizeof(T), m);

        // 4. Dispatch using the UNIQUE source and destination IDs
        if (current_task.operation_type == 1) {
            printf("    -> Executing MULTIPLY Kernel...\n");
            table_map(src_id, dst_id, sizeof(T), handle_multiply, m, 0);
        } else {
            printf("    -> Executing ADD Kernel...\n");
            table_map(src_id, dst_id, sizeof(T), handle_add, m, 0);
        }

        // 5. Gather using the UNIQUE destination ID
        T* res = simplepim_gather(dst_id, m);

        // 6. Verification (Checking Index 1)
        // Array value at index 1 is (1 + seed_value)
        uint32_t val_at_index_1 = 1 + current_task.seed_value;
        
        printf("    -> Verification (Index 1): Expected ");
        if (current_task.operation_type == 1) {
            printf("%u, Got %u\n", val_at_index_1 * 100, res[1]);
        } else {
            printf("%u, Got %u\n", val_at_index_1 + 100, res[1]);
        }
    }

    printf("\n[ORCHESTRATOR] All tasks complete. Shutting down.\n");
    free(dma_buffer);
    return 0;
}