"""Immutable contracts shared by tensor-network execution backends.

This module deliberately contains records and pure validation/serialization
functions only.  Device sessions, subprocesses, and tensor buffers belong to
backend implementations, not to these contracts.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
from typing import Any, Callable

import numpy as np


class Target(str, Enum):
    CPU = "cpu"
    GPU = "gpu"
    UPMEM = "upmem"


class NumericMode(str, Enum):
    COMPLEX128 = "complex128"
    FLOAT32_REAL = "float32_real"
    HOST_PACKED_INT8_PER_TASK_V1 = "host_packed_int8_per_task_v1"

    # Compatibility names for the initial functional execution contract.
    FLOAT32 = FLOAT32_REAL
    HOST_PACKED_INT8 = HOST_PACKED_INT8_PER_TASK_V1


@dataclass(frozen=True, slots=True, kw_only=True)
class UpmemTopology:
    dpu_count: int
    tasklets_per_dpu: int
    rank_count: int = 1


@dataclass(frozen=True, slots=True, kw_only=True)
class UpmemNodeWorkPlan:
    node_id: str
    kernel_id: str
    decomposition_id: str
    placement_id: str
    reduction_id: str
    dpu_ids: tuple[int, ...] = ()
    tile_count: int | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class CpuCompileRequest:
    contraction_dag_hash: str
    numeric_mode: NumericMode = NumericMode.FLOAT32
    executor_id: str = "cpu_numpy_v1"
    node_order: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True, kw_only=True)
class UpmemCompileRequest:
    contraction_dag_hash: str
    numeric_mode: NumericMode
    topology: UpmemTopology
    kernel_id: str = "dpu_gemm_tile_v4"
    decomposition_id: str = "m5_v4_tile_decomposition"
    placement_id: str = "m5_rank_wave_placement"
    reduction_id: str = "m5_tile_host_reduction"
    node_work_plans: tuple[UpmemNodeWorkPlan, ...] = ()
    profile_id: str = "m5_whole_circuit_v4_v1"
    abi_id: str = "execution_plan_v4"
    session_id: str = "persistent_rank_session_v1"
    dispatch_id: str = "bulk_set_synchronous_v1"


@dataclass(frozen=True, slots=True, kw_only=True)
class CpuPlan:
    numeric_mode: NumericMode
    executor_id: str
    node_order: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True, kw_only=True)
class UpmemPlan:
    topology: UpmemTopology
    numeric_mode: NumericMode
    kernel_id: str
    decomposition_id: str
    placement_id: str
    reduction_id: str
    node_work_plans: tuple[UpmemNodeWorkPlan, ...] = ()
    node_order: tuple[str, ...] = ()
    profile_id: str = "m5_whole_circuit_v4_v1"
    abi_id: str = "execution_plan_v4"
    session_id: str = "persistent_rank_session_v1"
    dispatch_id: str = "bulk_set_synchronous_v1"


@dataclass(frozen=True, slots=True, kw_only=True)
class ExecutionPlan:
    contraction_dag_hash: str
    target: Target
    payload: CpuPlan | UpmemPlan


@dataclass(frozen=True, slots=True, kw_only=True)
class RunContext:
    run_id: str
    target: Target
    warmups: int = 0
    repetitions: int = 1
    timeout_s: float | None = None
    target_resources: "UpmemRuntimeResources | None" = None


@dataclass(frozen=True, slots=True, kw_only=True)
class UpmemRuntimeResources:
    """Machine-local UPMEM resources excluded from execution-plan identity."""

    session_root: str
    host_binary: str
    dpu_binary: str
    initialization_binary: str
    rank_paths: tuple[str, ...] = ()
    session_opener: Callable[[ExecutionPlan, RunContext], Any] | None = field(
        default=None, repr=False, compare=False
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class UnsupportedExecution:
    target: Target
    reason: str
    capability: str = ""


@dataclass(frozen=True, slots=True, kw_only=True)
class BackendFacts:
    """Bounded backend facts exposed without leaking arbitrary metadata."""

    backend_id: str
    profile_id: str
    abi_id: str
    session_id: str
    dispatch_id: str
    kernel_id: str
    execution_class: str
    intermediate_placement: str
    intermediate_placement_origin: str
    native_identity_verified: bool
    target_observed: str | None = None
    hardware_allocation_verified: bool = False
    hardware_release_verified: bool = False
    hardware_release_confirmed: bool = False
    requested_dpu_count: int | None = None
    allocated_dpu_count: int | None = None
    observed_rank_count: int | None = None
    tasklets_per_dpu: int | None = None
    native_kernel_executed: bool = False
    hardware_kernel_executed: bool = False
    simulator_kernel_executed: bool = False
    cpu_fallback_used: bool = False
    host_binary_sha256: str | None = None
    dpu_binary_sha256: str | None = None
    initialization_binary_sha256: str | None = None
    rank_binding_sha256: str | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class TimingBreakdown:
    planning_s: float | None = None
    compilation_s: float | None = None
    preparation_s: float | None = None
    host_quantization_s: float | None = None
    h2d_s: float | None = None
    kernel_s: float | None = None
    d2h_s: float | None = None
    reduction_s: float | None = None
    host_dequantization_s: float | None = None
    reference_s: float | None = None
    validation_s: float | None = None
    session_open_s: float | None = None
    session_close_s: float | None = None
    route_total_s: float | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class ExecutionResult:
    contraction_dag_hash: str
    target: Target
    output: np.ndarray
    timing: TimingBreakdown
    executed_node_ids: tuple[str, ...] = ()
    h2d_bytes: int | None = None
    d2h_bytes: int | None = None
    transfer_bytes: int | None = None
    output_hash: str | None = None
    backend_facts: BackendFacts | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class ExecutionFailure:
    contraction_dag_hash: str
    target: Target
    stage: str
    reason: str
    retryable: bool = False


def _canonical_value(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {
            field.name: _canonical_value(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, tuple):
        return [_canonical_value(item) for item in value]
    if isinstance(value, list):
        return [_canonical_value(item) for item in value]
    return value


def canonical_serialize(value: object) -> str:
    """Return deterministic JSON for a supported contract value."""

    return json.dumps(
        _canonical_value(value),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )


def execution_plan_hash(plan: ExecutionPlan) -> str:
    """Hash the complete execution identity, including target payload."""

    validate_execution_plan(plan)
    return hashlib.sha256(canonical_serialize(plan).encode("utf-8")).hexdigest()


def validate_execution_plan(plan: ExecutionPlan) -> None:
    """Reject malformed or target/payload-mismatched execution plans."""

    if not plan.contraction_dag_hash:
        raise ValueError("contraction_dag_hash must be non-empty")
    if plan.target is Target.CPU and not isinstance(plan.payload, CpuPlan):
        raise ValueError("CPU execution plans require a CpuPlan payload")
    if plan.target is Target.UPMEM and not isinstance(plan.payload, UpmemPlan):
        raise ValueError("UPMEM execution plans require an UpmemPlan payload")
    if plan.target is Target.GPU:
        raise ValueError("GPU execution plans are not supported by this contract")
    if isinstance(plan.payload, CpuPlan):
        if not plan.payload.executor_id:
            raise ValueError("CPU executor_id must be non-empty")
        if len(set(plan.payload.node_order)) != len(plan.payload.node_order):
            raise ValueError("CPU node_order contains duplicate node ids")
    if isinstance(plan.payload, UpmemPlan):
        topology = plan.payload.topology
        if topology.dpu_count < 1 or topology.tasklets_per_dpu < 1:
            raise ValueError("UPMEM topology counts must be positive")
        if topology.rank_count < 1:
            raise ValueError("UPMEM rank_count must be positive")
        if topology.dpu_count % topology.rank_count:
            raise ValueError("UPMEM dpu_count must be divisible by rank_count")
        for name in (
            "kernel_id",
            "decomposition_id",
            "placement_id",
            "reduction_id",
        ):
            if not getattr(plan.payload, name):
                raise ValueError(f"UPMEM {name} must be non-empty")
        node_ids = [work.node_id for work in plan.payload.node_work_plans]
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("UPMEM work plans contain duplicate node ids")
        for work in plan.payload.node_work_plans:
            if not work.node_id or not work.kernel_id:
                raise ValueError("UPMEM work plan ids must be non-empty")
            if (
                (work.tile_count is not None and work.tile_count < 0)
                or any(dpu_id < 0 for dpu_id in work.dpu_ids)
            ):
                raise ValueError("UPMEM work plan counts and ids must be non-negative")
        if len(set(plan.payload.node_order)) != len(plan.payload.node_order):
            raise ValueError("UPMEM node_order contains duplicate node ids")


def validate_upmem_runtime_resources(
    resources: UpmemRuntimeResources, topology: UpmemTopology
) -> None:
    """Validate machine-local rank bindings against logical topology."""

    if not resources.rank_paths:
        raise ValueError("UPMEM runtime resources require explicit rank_paths")
    if len(resources.rank_paths) != topology.rank_count:
        raise ValueError(
            "UPMEM runtime rank_paths must match the logical topology rank_count"
        )
    if len(set(resources.rank_paths)) != len(resources.rank_paths):
        raise ValueError("UPMEM runtime rank paths must be unique")


def validate_transfer_bytes(
    h2d_bytes: int | None,
    d2h_bytes: int | None,
    transfer_bytes: int | None,
) -> None:
    """Validate transfer accounting when all three values are available."""

    values = (h2d_bytes, d2h_bytes, transfer_bytes)
    if any(value is not None and value < 0 for value in values):
        raise ValueError("transfer byte counts must be non-negative")
    if any(value is None for value in values):
        return
    if transfer_bytes != h2d_bytes + d2h_bytes:
        raise ValueError("transfer_bytes must equal h2d_bytes + d2h_bytes")


def validate_timing(timing: TimingBreakdown) -> None:
    """Reject negative or non-finite timing observations."""

    for timing_field in fields(timing):
        value = getattr(timing, timing_field.name)
        if value is not None and (value < 0 or not math.isfinite(value)):
            raise ValueError(f"{timing_field.name} must be finite and non-negative")


def validate_execution_result(result: ExecutionResult) -> None:
    """Validate execution facts without interpreting performance claims."""

    if not result.contraction_dag_hash:
        raise ValueError("contraction_dag_hash must be non-empty")
    validate_timing(result.timing)
    validate_transfer_bytes(result.h2d_bytes, result.d2h_bytes, result.transfer_bytes)
    if result.target is Target.UPMEM and result.backend_facts is None:
        raise ValueError("UPMEM execution results require backend_facts")
