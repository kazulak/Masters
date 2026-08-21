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
class UpmemWorkUnit:
    """One statically assigned bounded v4 tile invocation."""

    node_id: str
    stable_tile_id: str
    wave: int
    logical_rank: int
    logical_dpu: int
    batch_start: int
    batch_size: int
    m_start: int
    m_size: int
    n_start: int
    n_size: int
    k_start: int
    k_size: int
    estimated_input_bytes: int
    estimated_output_bytes: int
    aligned_mram_bytes: int
    estimated_arithmetic_work: int


@dataclass(frozen=True, slots=True, kw_only=True)
class UpmemNodePlan:
    """Static physical plan for one semantic DAG node."""

    node_id: str
    node_kind: str
    canonical_shape: tuple[int, int, int, int] | None
    work_units: tuple[UpmemWorkUnit, ...]
    reduction_mode: str
    arithmetic_imbalance: float


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
    node_plans: tuple[UpmemNodePlan, ...] = ()
    profile_id: str
    abi_id: str
    session_id: str
    dispatch_id: str


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
    physical_plan_consumed: bool = False
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
        if plan.payload.numeric_mode not in {
            NumericMode.FLOAT32_REAL,
            NumericMode.HOST_PACKED_INT8_PER_TASK_V1,
        }:
            raise ValueError("UPMEM numeric mode is not supported by v4")
        _validate_upmem_node_plans(plan.payload)


def _validate_upmem_node_plans(plan: UpmemPlan) -> None:
    node_ids = [node_plan.node_id for node_plan in plan.node_plans]
    if len(node_ids) != len(set(node_ids)):
        raise ValueError("UPMEM node plans contain duplicate node ids")
    for node_plan in plan.node_plans:
        if not node_plan.node_id or node_plan.node_kind not in {"contract", "reduce"}:
            raise ValueError("UPMEM node plan identity is invalid")
        if not node_plan.reduction_mode:
            raise ValueError("UPMEM node plan reduction_mode must be non-empty")
        if not math.isfinite(node_plan.arithmetic_imbalance) or node_plan.arithmetic_imbalance < 0:
            raise ValueError("UPMEM node plan arithmetic_imbalance must be finite and non-negative")
        if node_plan.node_kind == "reduce":
            if node_plan.canonical_shape is not None or node_plan.work_units:
                raise ValueError("UPMEM reduce node plans cannot contain tile geometry")
            if node_plan.arithmetic_imbalance != 0.0:
                raise ValueError("UPMEM reduce node plan arithmetic_imbalance must be zero")
            continue
        _validate_contract_node_plan(plan, node_plan)


def _validate_contract_node_plan(plan: UpmemPlan, node_plan: UpmemNodePlan) -> None:
    shape = node_plan.canonical_shape
    if shape is None or len(shape) != 4 or any(value < 1 for value in shape):
        raise ValueError("UPMEM contract node plan requires positive canonical B/M/K/N shape")
    if not node_plan.work_units:
        raise ValueError("UPMEM contract node plan requires work units")
    batch, m, k, n = shape
    input_element_bytes = (
        1
        if plan.numeric_mode is NumericMode.HOST_PACKED_INT8_PER_TASK_V1
        else 4
    )
    dpus_per_rank = plan.topology.dpu_count // plan.topology.rank_count
    by_output: dict[tuple[int, int, int, int, int, int], list[tuple[int, int]]] = {}
    wave_slots: dict[int, set[tuple[int, int]]] = {}
    seen_tile_ids: set[str] = set()
    dpu_work = [0] * plan.topology.dpu_count
    for unit in node_plan.work_units:
        if unit.node_id != node_plan.node_id or not unit.stable_tile_id:
            raise ValueError("UPMEM work unit node or tile identity is invalid")
        if unit.stable_tile_id in seen_tile_ids:
            raise ValueError("UPMEM work units contain duplicate stable tile ids")
        seen_tile_ids.add(unit.stable_tile_id)
        if unit.wave < 0 or not 0 <= unit.logical_rank < plan.topology.rank_count:
            raise ValueError("UPMEM work unit wave or rank is out of bounds")
        if not 0 <= unit.logical_dpu < dpus_per_rank:
            raise ValueError("UPMEM work unit logical DPU is out of bounds")
        slot = (unit.logical_rank, unit.logical_dpu)
        if slot in wave_slots.setdefault(unit.wave, set()):
            raise ValueError("UPMEM work units reuse a logical DPU in one wave")
        wave_slots[unit.wave].add(slot)
        extents = (
            (unit.batch_start, unit.batch_size, batch),
            (unit.m_start, unit.m_size, m),
            (unit.n_start, unit.n_size, n),
            (unit.k_start, unit.k_size, k),
        )
        if any(start < 0 or size < 1 or start + size > limit for start, size, limit in extents):
            raise ValueError("UPMEM work unit extent is outside canonical geometry")
        left_bytes = unit.m_size * unit.k_size * input_element_bytes
        right_bytes = unit.k_size * unit.n_size * input_element_bytes
        output_bytes = unit.m_size * unit.n_size * 4
        aligned_mram = _align(left_bytes) + _align(right_bytes) + _align(output_bytes)
        if any(_align(value) > 512 * 1024 for value in (left_bytes, right_bytes, output_bytes)) or aligned_mram > 512 * 1024:
            raise ValueError("UPMEM work unit aligned A/B/C footprint exceeds 512 KiB")
        if plan.numeric_mode is NumericMode.HOST_PACKED_INT8_PER_TASK_V1 and unit.k_size * 128 * 128 > (1 << 31) - 1:
            raise ValueError("UPMEM packed K chunk exceeds int32 accumulation safety")
        expected_input = left_bytes + right_bytes
        expected_work = unit.m_size * unit.n_size * unit.k_size
        if (
            unit.estimated_input_bytes != expected_input
            or unit.estimated_output_bytes != output_bytes
            or unit.aligned_mram_bytes != aligned_mram
            or unit.estimated_arithmetic_work != expected_work
        ):
            raise ValueError("UPMEM work unit stored estimates do not match geometry")
        by_output.setdefault(
            (
                unit.batch_start,
                unit.batch_size,
                unit.m_start,
                unit.m_size,
                unit.n_start,
                unit.n_size,
            ),
            [],
        ).append((unit.k_start, unit.k_size))
        dpu_work[unit.logical_rank * dpus_per_rank + unit.logical_dpu] += expected_work
    for intervals in by_output.values():
        cursor = 0
        for start, size in sorted(intervals):
            if start != cursor:
                raise ValueError("UPMEM K chunks contain a gap or overlap")
            cursor += size
        if cursor != k:
            raise ValueError("UPMEM K chunks do not cover canonical K")
    _validate_output_coverage(by_output, batch, m, n)
    total_work = sum(dpu_work)
    expected_imbalance = max(dpu_work) / (total_work / len(dpu_work)) if total_work else 0.0
    if node_plan.arithmetic_imbalance != expected_imbalance:
        raise ValueError("UPMEM node plan arithmetic_imbalance does not match work units")


def _validate_output_coverage(
    by_output: dict[tuple[int, int, int, int, int, int], list[tuple[int, int]]],
    batch: int,
    m: int,
    n: int,
) -> None:
    batch_bounds = {0, batch}
    m_bounds = {0, m}
    n_bounds = {0, n}
    for batch_start, batch_size, m_start, m_size, n_start, n_size in by_output:
        batch_bounds.update((batch_start, batch_start + batch_size))
        m_bounds.update((m_start, m_start + m_size))
        n_bounds.update((n_start, n_start + n_size))
    sorted_batch_bounds = sorted(batch_bounds)
    sorted_m_bounds = sorted(m_bounds)
    sorted_n_bounds = sorted(n_bounds)
    for batch_start, batch_end in zip(
        sorted_batch_bounds[:-1], sorted_batch_bounds[1:], strict=True
    ):
        for m_start, m_end in zip(
            sorted_m_bounds[:-1], sorted_m_bounds[1:], strict=True
        ):
            for n_start, n_end in zip(
                sorted_n_bounds[:-1], sorted_n_bounds[1:], strict=True
            ):
                covering = sum(
                    output_batch <= batch_start and batch_end <= output_batch + batch_size
                    and output_m <= m_start and m_end <= output_m + m_size
                    and output_n <= n_start and n_end <= output_n + n_size
                    for output_batch, batch_size, output_m, m_size, output_n, n_size in by_output
                )
                if covering != 1:
                    raise ValueError("UPMEM output tiles contain a gap or overlap")


def _align(value: int) -> int:
    return ((value + 7) // 8) * 8


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
