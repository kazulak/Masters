#ifndef UPMEM_SDK_GENERIC_LOOP_FRONTIER_TWO_DPU_COMMON_H
#define UPMEM_SDK_GENERIC_LOOP_FRONTIER_TWO_DPU_COMMON_H

/* The frontier route deliberately shares the resident package ABI. */
#include "../upmem_sdk_generic_loop_resident/common.h"

#define FRONTIER_TWO_DPU_COUNT 2u
#define FRONTIER_TWO_DPU_TASKLETS 1u
#define FRONTIER_TWO_DPU_OPERATION_COUNT 3u

#define FRONTIER_PROFILE_ID "hardware_frontier_two_dpu_m3_1_v1"
#define FRONTIER_BACKEND_ID "upmem_sdk_hardware_taskgraph_frontier_two_dpu"
#define FRONTIER_ROUTE_ID "upmem_tn_hardware_taskgraph_frontier_two_dpu"
#define FRONTIER_SCHEMA_ID "generic_loop_resident_frontier_two_dpu_v1"

#endif
