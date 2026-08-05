#ifndef UPMEM_SDK_GENERIC_LOOP_FRONTIER_TWO_DPU_COMMON_H
#define UPMEM_SDK_GENERIC_LOOP_FRONTIER_TWO_DPU_COMMON_H

/* The frontier route deliberately shares the resident package ABI. */
#include "../upmem_sdk_generic_loop_resident/common.h"

#define FRONTIER_TWO_DPU_COUNT 2u
#define FRONTIER_TWO_DPU_TASKLETS 1u
#define FRONTIER_TWO_DPU_OPERATION_COUNT 3u
#ifndef FRONTIER_PROVIDER_SIMPLEPIM_MANAGEMENT
#define FRONTIER_PROVIDER_SIMPLEPIM_MANAGEMENT 0
#endif

#if FRONTIER_PROVIDER_SIMPLEPIM_MANAGEMENT
#define FRONTIER_PROFILE_ID "hardware_frontier_two_dpu_m4_1_v1"
#define FRONTIER_PROVIDER_ID "simplepim_management"
#define FRONTIER_BACKEND_ID "upmem_sdk_hardware_taskgraph_simplepim_management_frontier_two_dpu"
#define FRONTIER_ROUTE_ID "upmem_tn_hardware_taskgraph_simplepim_management_frontier_two_dpu"
#define FRONTIER_CONTROL_PROVIDER "simplepim_management"
#define FRONTIER_KERNEL_PROVIDER "thesis_resident_generic_contract"
#define FRONTIER_SIMPLEPIM_MANAGEMENT_API "simplepim_management_init_physical_v1"
#define FRONTIER_ALLOCATION_SOURCE "simplepim_management"
#else
#define FRONTIER_PROFILE_ID "hardware_frontier_two_dpu_m3_1_v2"
#define FRONTIER_PROVIDER_ID "raw_sdk"
#define FRONTIER_BACKEND_ID "upmem_sdk_hardware_taskgraph_frontier_two_dpu"
#define FRONTIER_ROUTE_ID "upmem_tn_hardware_taskgraph_frontier_two_dpu"
#define FRONTIER_CONTROL_PROVIDER "raw_sdk"
#define FRONTIER_KERNEL_PROVIDER "thesis_resident_generic_contract"
#define FRONTIER_SIMPLEPIM_MANAGEMENT_API "not_used"
#define FRONTIER_ALLOCATION_SOURCE "raw_sdk"
#endif
#define FRONTIER_ALLOCATION_PROFILE "backend=hw"
#define FRONTIER_SCHEMA_ID "generic_loop_resident_frontier_two_dpu_v2"

#endif
