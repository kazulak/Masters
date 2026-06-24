#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <string.h>

#include "management/Management.h"
#include "processing/map/Map.h"
#include "processing/zip/Zip.h"
#include "communication/CommOps.h"
#include "processing/ProcessingHelperHost.h"
#include "Param.h"

// 1. Dataflow Task Definition
typedef struct {
    int task_id;
    int is_ready;          
    int is_completed;      
    int op_type;           // 1 = Mult, 2 = Add
    int parents_needed;    // How many inputs required to start
    int parents_completed; // How many inputs have arrived
    float* host_in1;       // Data buffer 1
    float* host_in2;       // Data buffer 2 (Only used for Add)
    float* host_out;       // Result buffer
} DAG_Task;

// 2. Hardware Execution Engine
void execute_on_dpu(DAG_Task* task, simplepim_management_t* m, handle_t* h_mult, handle_t* h_add, handle_t* h_zip) {
    int tid = omp_get_thread_num();
    char id1[32], id2[32], zip_id[32], out_id[32];
    sprintf(id1, "in1_%d", task->task_id);
    sprintf(out_id, "out_%d", task->task_id);

    if (task->op_type == 1) {
        printf("[DPU Worker %d] Executing Task %d (Heavy Multiply)...\n", tid, task->task_id);
        simplepim_scatter(id1, task->host_in1, nr_elements, sizeof(T), m);
        table_map(id1, out_id, sizeof(T), h_mult, m, 0);
        
        T* res = simplepim_gather(out_id, m);
        memcpy(task->host_out, res, nr_elements * sizeof(T));
    } 
    else if (task->op_type == 2) {
        printf("[DPU Worker %d] Executing Task %d (Zip + Add)...\n", tid, task->task_id);
        sprintf(id2, "in2_%d", task->task_id);
        sprintf(zip_id, "zip_%d", task->task_id);

        simplepim_scatter(id1, task->host_in1, nr_elements, sizeof(T), m);
        simplepim_scatter(id2, task->host_in2, nr_elements, sizeof(T), m);
        
        table_zip(id1, id2, zip_id, h_zip, m);
        table_map(zip_id, out_id, sizeof(T), h_add, m, 0);
        
        T* res = simplepim_gather(out_id, m);
        memcpy(task->host_out, res, nr_elements * sizeof(T));
    }
}

int main() {
    printf("[ORCHESTRATOR] Booting Pure Dataflow Engine...\n");

    // ==========================================
    // 1. PROVISION HARDWARE
    // ==========================================
    int num_dpu_workers = 2; 
    simplepim_management_t* dpu_managers[num_dpu_workers];
    handle_t *h_mult[num_dpu_workers], *h_add[num_dpu_workers], *h_zip[num_dpu_workers];
    omp_lock_t dpu_locks[num_dpu_workers];

    for(int i = 0; i < num_dpu_workers; i++) {
        dpu_managers[i] = table_management_init(dpu_number);
        h_mult[i] = create_handle("kernels/multiply", MAP);
        h_add[i]  = create_handle("kernels/add", MAP);
        h_zip[i]  = create_handle("", ZIP); // Framework default zip
        omp_init_lock(&dpu_locks[i]);
    }

    // ==========================================
    // 2. DEFINE THE DAG (Tasks & Memory)
    // ==========================================
    int num_tasks = 7;
    DAG_Task q[7];
    for(int i=0; i<7; i++) {
        q[i].task_id = i;
        q[i].is_completed = 0;
        q[i].parents_completed = 0;
        q[i].host_in1 = malloc(nr_elements * sizeof(T));
        q[i].host_in2 = malloc(nr_elements * sizeof(T));
        q[i].host_out = malloc(nr_elements * sizeof(T));
    }

    // Train A (Tasks 0, 1, 2)
    q[0].is_ready = 1; q[0].op_type = 1; q[0].parents_needed = 0;
    q[1].is_ready = 0; q[1].op_type = 1; q[1].parents_needed = 1;
    q[2].is_ready = 0; q[2].op_type = 1; q[2].parents_needed = 1;
    
    // Train B (Tasks 3, 4, 5)
    q[3].is_ready = 1; q[3].op_type = 1; q[3].parents_needed = 0;
    q[4].is_ready = 0; q[4].op_type = 1; q[4].parents_needed = 1;
    q[5].is_ready = 0; q[5].op_type = 1; q[5].parents_needed = 1;

    // Final Reduction (Task 6)
    q[6].is_ready = 0; q[6].op_type = 2; q[6].parents_needed = 2;

    // Seed the initial data for the starting tasks
    for(int i=0; i<nr_elements; i++) {
        q[0].host_in1[i] = 1.0f; // Train A starts with 1.0
        q[3].host_in1[i] = 2.0f; // Train B starts with 2.0
    }

    // ==========================================
    // 3. DEFINE THE ROUTING EDGES
    // format: {source_task, dest_task, dest_port}
    // ==========================================
    int num_edges = 6;
    int edges[6][3] = {
        {0, 1, 1}, // T0 output -> T1 port 1
        {1, 2, 1}, // T1 output -> T2 port 1
        {2, 6, 1}, // T2 output -> T6 port 1 (Train A end)
        {3, 4, 1}, // T3 output -> T4 port 1
        {4, 5, 1}, // T4 output -> T5 port 1
        {5, 6, 2}  // T5 output -> T6 port 2 (Train B end)
    };

    // ==========================================
    // 4. PARALLEL EXECUTION ENGINE
    // ==========================================
    int completed_tasks = 0;
    omp_set_num_threads(4); 
    
    printf("\n[ORCHESTRATOR] Engine Online. Dispatching Graph...\n\n");

    #pragma omp parallel shared(q, completed_tasks)
    {
        while(completed_tasks < num_tasks) {
            int task_to_run = -1;

            #pragma omp critical
            {
                for (int i = 0; i < num_tasks; i++) {
                    if (q[i].is_ready && !q[i].is_completed) {
                        task_to_run = i;
                        q[i].is_completed = 2; 
                        break;
                    }
                }
            }

            if (task_to_run != -1) {
                DAG_Task* task = &q[task_to_run];

                // Claim a DPU lock
                int manager = -1;
                while (manager == -1) {
                    for (int j = 0; j < num_dpu_workers; j++) {
                        if (omp_test_lock(&dpu_locks[j])) { manager = j; break; }
                    }
                }

                // Execute on Silicon
                execute_on_dpu(task, dpu_managers[manager], h_mult[manager], h_add[manager], h_zip[manager]);
                omp_unset_lock(&dpu_locks[manager]);

                #pragma omp critical
                {
                    task->is_completed = 1;
                    completed_tasks++;
                    printf("  -> Task %d Output[0] = %f\n", task->task_id, task->host_out[0]);

                    // ROUTE THE DATA to dependent tasks
                    for (int e = 0; e < num_edges; e++) {
                        if (edges[e][0] == task->task_id) {
                            int child = edges[e][1];
                            int port  = edges[e][2];
                            
                            // Propagate the buffer!
                            if (port == 1) memcpy(q[child].host_in1, task->host_out, nr_elements * sizeof(T));
                            if (port == 2) memcpy(q[child].host_in2, task->host_out, nr_elements * sizeof(T));
                            
                            q[child].parents_completed++;
                            if (q[child].parents_completed == q[child].parents_needed) {
                                q[child].is_ready = 1;
                                printf("[ROUTER] Task %d unlocked by Task %d!\n", child, task->task_id);
                            }
                        }
                    }
                }
            } 
        } // end while
    } // end omp parallel

    printf("\n[ORCHESTRATOR] Graph Complete. Final Result (Index 0): %f\n", q[6].host_out[0]);
    
    // Cleanup
    for(int i=0; i<7; i++) { free(q[i].host_in1); free(q[i].host_in2); free(q[i].host_out); }
    for(int i=0; i<num_dpu_workers; i++) omp_destroy_lock(&dpu_locks[i]);

    return 0;
}