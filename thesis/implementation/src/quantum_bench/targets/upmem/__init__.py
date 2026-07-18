"""Active UPMEM planning, generic bridge, and environment APIs."""

# This module is a public facade; its imports are intentionally re-exported.
# ruff: noqa: F401

from quantum_bench.targets.upmem.environment import (
    DEFAULT_UPMEM_ENV_TIMEOUT_SECONDS,
    UPMEM_ENV_CHECK_SCHEMA_VERSION,
    CommandExecutionRecord,
    SampleCheckRecord,
    UpmemEnvironmentCheckResult,
    UpmemSdkDiscovery,
    UpmemToolCheck,
    build_environment_check_result,
    discover_upmem_sdk,
    run_command,
    sample_success_marker,
)
from quantum_bench.targets.upmem.frontier import (
    DOMINANT_SOURCE_HYBRID,
    DOMINANT_SOURCE_INTER_TASK,
    DOMINANT_SOURCE_INTRA_TASK,
    DOMINANT_SOURCE_SERIAL,
    MEMORY_LEVEL_L1_WRAM,
    MEMORY_LEVEL_L2_SINGLE_DPU_MRAM,
    MEMORY_LEVEL_L3_MULTI_DPU,
    MEMORY_LEVEL_L4_OUT_OF_SCOPE,
    MEMORY_LEVEL_NOT_DENSE_GEMM,
    PIM_FRONTIER_ANALYSIS_SCHEMA_VERSION,
    PimFrontierGraphAnalysis,
    PimFrontierWave,
    PimTaskMemoryAnalysis,
    UpmemResourceModel,
    analyze_task,
    analyze_task_graph,
)
from quantum_bench.targets.upmem.generic_bridge import (
    GENERIC_BRIDGE_ID,
    GENERIC_BRIDGE_SCHEMA_VERSION,
    GENERIC_LOOP_BACKEND_ID,
    GENERIC_LOOP_KERNEL_FAMILY,
    GenericBridgeBackendIdentity,
    GenericBridgeBlob,
    GenericBridgeExecutionResult,
    GenericBridgeInputManifest,
    GenericBridgeOutputManifest,
    GenericBridgeStatus,
    execute_generic_bridge,
    generic_bridge_backend_registry,
    get_generic_bridge_backend,
    read_generic_bridge_input_manifest,
    read_generic_bridge_output_manifest,
    write_generic_bridge_input_manifest,
)
from quantum_bench.targets.upmem.generic_boundary import (
    GENERIC_BOUNDARY_CASE_ID,
    GENERIC_BOUNDARY_EINSUM,
    GenericBoundaryWorkload,
    build_generic_boundary_workload,
    generic_boundary_manifest,
    is_generic_boundary_case,
)
from quantum_bench.targets.upmem.schedule import (
    DENSE_INT8_FORMAT,
    REQUIRES_TILING_NOT_IMPLEMENTED,
    UNSUPPORTED_DENSE_GEMM_SHAPE,
    UPMEM_DENSE_ESTIMATE_KEY,
    UPMEM_DENSE_TILE_PLAN_ARTIFACT_KEY,
    UPMEM_DENSE_TILE_PLAN_MODEL,
    UPMEM_PROFILE,
    UpmemDataFormat,
    UpmemHardwareProfile,
    UpmemScheduleEstimate,
    UpmemTaskEstimate,
    annotate_task_graph_with_upmem_estimates,
    estimate_dense_task,
    estimate_dense_task_graph,
    upmem_task_estimate_rows,
)
from quantum_bench.targets.upmem.synthetic_pressure import (
    SYNTHETIC_PRESSURE_ERROR,
    SYNTHETIC_PRESSURE_KIND,
    build_synthetic_pressure_task_graph,
    is_synthetic_pressure_case,
    require_synthetic_pressure_metadata,
    synthetic_pressure_initial_tensors,
    synthetic_pressure_manifest,
)
from quantum_bench.targets.upmem.tile_plan import (
    UPMEM_EXECUTION_CLASS_L1_WRAM,
    UPMEM_EXECUTION_CLASS_L2_SINGLE_DPU_MRAM,
    UPMEM_L1_KERNEL_STRATEGY,
    UPMEM_L2_ALIGNMENT_RESERVE_BYTES,
    UPMEM_L2_EFFECTIVE_WRAM_BYTES,
    UPMEM_L2_KERNEL_STRATEGY,
    UPMEM_L2_MAX_HOST_BLOB_BYTES,
    UPMEM_L2_NATIVE_MAX_DIM,
    UPMEM_L2_PER_DPU_MRAM_BYTES,
    UPMEM_L2_TILE_CANDIDATES,
    UpmemDenseTilePlan,
    UpmemL2TiledExecutionPlan,
    UpmemTileCounts,
    UpmemTileShape,
    plan_dense_task,
    plan_dense_task_graph,
    plan_l2_tiled_execution,
    upmem_dense_tile_plan_rows,
)


__all__ = [name for name in globals() if not name.startswith("_")]
