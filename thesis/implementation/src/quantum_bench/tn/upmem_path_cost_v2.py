"""Versioned modeled costs for the bounded generic UPMEM execution policy.

This module deliberately does not alter ``upmem_path_cost_v1``.  V2 models
the existing strict generic runtime more faithfully: a task with a genuinely
complex input is executed as four real components and recombined on the host.
All byte and memory values are modeled application/runtime quantities, not
physical hardware counters or latency predictions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Literal, Mapping, Sequence

import numpy as np

from quantum_bench.core.records import ContractionTask, JsonDict
from quantum_bench.routing.generic_numeric_contract import classify_numeric
from quantum_bench.routing.generic_prepare import (
    GENERIC_OUTPUT_TILE_ELEMENTS,
    GenericTaskPreparationCaps,
    generic_structural_feasibility_from_metadata,
)
from quantum_bench.tn.graph import ContractNode
from quantum_bench.tn.network import TensorNetworkValue


UPMEM_PATH_OBJECTIVE_V2 = "upmem_path_cost_v2"
GENERIC_SINGLE_DPU_SPLIT_COMPLEX_V2 = "generic_single_dpu_split_complex_v2"
FIXED_LOG1P_GENERIC_BUDGETS_V2 = "fixed_log1p_generic_budgets_v2"
DEFAULT_UPMEM_PATH_COST_POLICY_V2 = GENERIC_SINGLE_DPU_SPLIT_COMPLEX_V2
DEFAULT_UPMEM_PATH_COST_NORMALIZATION_V2 = FIXED_LOG1P_GENERIC_BUDGETS_V2

# These are modeled capacity assumptions, not a measurement of a compiled DPU
# binary.  The generic kernel reserves three max-sized float32 MRAM regions.
UPMEM_GENERIC_MRAM_CAPACITY_BYTES = 64 * 1024 * 1024
UPMEM_GENERIC_EFFECTIVE_WRAM_BUDGET_BYTES = 60 * 1024
UPMEM_GENERIC_NATIVE_MAX_ELEMENTS = 65536
UPMEM_GENERIC_REAL_ELEMENT_BYTES = 4
UPMEM_GENERIC_SCALAR_READ_WINDOW_BYTES = 8

PathCostProfileIdV2 = Literal[
    "compute_oriented",
    "host_transfer_oriented",
    "local_movement_oriented",
    "wram_constrained",
    "synchronization_constrained",
    "balanced_literature_informed",
]


@dataclass(frozen=True)
class TaskNumericExecution:
    """How one generic task is represented by the current strict runtime."""

    representation: Literal["real_float32", "split_real_imag"]
    component_invocations: int
    recombination_flops: int
    rejection_reason: str | None = None

    @property
    def feasible(self) -> bool:
        return self.rejection_reason is None

    def to_json_dict(self) -> JsonDict:
        return {
            "representation": self.representation,
            "component_invocations": int(self.component_invocations),
            "recombination_flops": int(self.recombination_flops),
            "feasible": self.feasible,
            "rejection_reason": self.rejection_reason,
        }


@dataclass(frozen=True)
class PathCostComponentsV2:
    """Independent modeled primitives plus explicitly derived memory fields."""

    estimated_flops: int = 0
    largest_tensor_bytes: int = 0
    host_to_dpu_payload_bytes: int = 0
    dpu_to_host_payload_bytes: int = 0
    mram_dma_window_bytes_model: int = 0
    tile_iterations: int = 0
    host_completion_events: int = 0
    numeric_component_invocations: int = 0
    numeric_recombination_flops: int = 0
    numeric_representation_penalty: float = 0.0
    task_mram_payload_bytes: int = 0
    native_static_mram_reservation_bytes: int = 0
    mram_capacity_bytes: int = 0
    mram_static_reservation_pressure_ratio: float = 0.0
    mram_max_region_payload_ratio: float = 0.0
    mram_payload_pressure_ratio: float = 0.0
    known_wram_static_bytes: int = 0
    wram_budget_bytes: int = 0
    wram_known_pressure_ratio: float = 0.0
    feasibility: bool = True
    rejection_reasons: tuple[str, ...] = ()

    def to_json_dict(self) -> JsonDict:
        return {
            "estimated_flops": int(self.estimated_flops),
            "largest_tensor_bytes": int(self.largest_tensor_bytes),
            "host_to_dpu_payload_bytes": int(self.host_to_dpu_payload_bytes),
            "dpu_to_host_payload_bytes": int(self.dpu_to_host_payload_bytes),
            "mram_dma_window_bytes_model": int(self.mram_dma_window_bytes_model),
            "mram_dma_window_scope": "modeled_aligned_request_volume_not_physical_bus_bytes",
            "tile_iterations": int(self.tile_iterations),
            "host_completion_events": int(self.host_completion_events),
            "host_completion_event_scope": "one_synchronous_dpu_launch_per_real_component",
            "numeric_component_invocations": int(self.numeric_component_invocations),
            "numeric_recombination_flops": int(self.numeric_recombination_flops),
            "numeric_representation_penalty": float(self.numeric_representation_penalty),
            "task_mram_payload_bytes": int(self.task_mram_payload_bytes),
            "native_static_mram_reservation_bytes": int(self.native_static_mram_reservation_bytes),
            "mram_capacity_bytes": int(self.mram_capacity_bytes),
            "mram_static_reservation_pressure_ratio": float(self.mram_static_reservation_pressure_ratio),
            "mram_max_region_payload_ratio": float(self.mram_max_region_payload_ratio),
            "mram_payload_pressure_ratio": float(self.mram_payload_pressure_ratio),
            "mram_payload_pressure_scope": "maximum_single_buffer_payload_divided_by_fixed_native_buffer_capacity",
            "known_wram_static_bytes": int(self.known_wram_static_bytes),
            "wram_budget_bytes": int(self.wram_budget_bytes),
            "wram_known_pressure_ratio": float(self.wram_known_pressure_ratio),
            "memory_budget_scope": "configured_modeled_budget_not_measured_runtime_occupancy",
            "feasibility": bool(self.feasibility),
            "rejection_reasons": list(self.rejection_reasons),
            "metric_contract": metric_contract_v2(),
        }


@dataclass(frozen=True)
class UpmemPathCostPolicyV2:
    """Fixed single-DPU assumptions for the existing generic float32 route."""

    policy_id: str = GENERIC_SINGLE_DPU_SPLIT_COMPLEX_V2
    caps: GenericTaskPreparationCaps = field(default_factory=GenericTaskPreparationCaps)
    output_tile_elements: int = GENERIC_OUTPUT_TILE_ELEMENTS
    mram_capacity_bytes: int = UPMEM_GENERIC_MRAM_CAPACITY_BYTES
    wram_budget_bytes: int = UPMEM_GENERIC_EFFECTIVE_WRAM_BUDGET_BYTES
    native_max_tensor_elements: int = UPMEM_GENERIC_NATIVE_MAX_ELEMENTS
    input_element_bytes: int = UPMEM_GENERIC_REAL_ELEMENT_BYTES
    output_element_bytes: int = UPMEM_GENERIC_REAL_ELEMENT_BYTES
    scalar_read_window_bytes: int = UPMEM_GENERIC_SCALAR_READ_WINDOW_BYTES

    def __post_init__(self) -> None:
        if self.output_tile_elements <= 0:
            raise ValueError("output_tile_elements must be positive")
        if min(self.mram_capacity_bytes, self.wram_budget_bytes, self.native_max_tensor_elements) <= 0:
            raise ValueError("generic capacity assumptions must be positive")

    @property
    def native_static_mram_reservation_bytes(self) -> int:
        return 3 * self.native_max_tensor_elements * self.output_element_bytes

    @property
    def known_wram_static_bytes(self) -> int:
        # dpu.c has one 8-byte scalar window and a float32 output tile.
        return self.scalar_read_window_bytes + self.output_tile_elements * self.output_element_bytes

    def to_json_dict(self) -> JsonDict:
        return {
            "policy_id": self.policy_id,
            "caps": {
                "application_max_rank": int(self.caps.max_rank),
                "application_max_tensor_elements": int(self.caps.max_tensor_elements),
                "application_max_contracted_combinations": int(self.caps.max_contracted_combinations),
                "native_abi_max_tensor_elements": int(self.native_max_tensor_elements),
            },
            "output_tile_elements": int(self.output_tile_elements),
            "input_element_bytes": int(self.input_element_bytes),
            "output_element_bytes": int(self.output_element_bytes),
            "mram_capacity_bytes": int(self.mram_capacity_bytes),
            "wram_budget_bytes": int(self.wram_budget_bytes),
            "memory_budget_source": "configured_generic_dpu_policy_v2",
            "numeric_contract": "real_float32_or_split_real_imag_v2",
            "host_dpu_transfer_model": "per_real_component_operand_upload_and_output_download_no_reuse",
            "mram_wram_transfer_model": "aligned_scalar_read_windows_and_aligned_output_tile_writes",
            "host_completion_model": "one_synchronous_dpu_launch_per_real_component",
            "dpu_parallelism_model": "single_dpu_serial_tasks",
        }


@dataclass(frozen=True)
class PathCostNormalizationV2:
    normalization_id: str
    caps: tuple[tuple[str, float], ...]

    def to_json_dict(self) -> JsonDict:
        return {
            "normalization_id": self.normalization_id,
            "caps": {name: float(value) for name, value in self.caps},
            "transform": "log1p(value)/log1p(cap)",
        }

    def normalize(self, components: PathCostComponentsV2) -> JsonDict:
        values = _normalizable_values_v2(components)
        return {name: _fixed_log1p(values[name], cap) for name, cap in self.caps}


@dataclass(frozen=True)
class PathCostWeightsV2:
    estimated_flops: float = 0.0
    host_to_dpu_payload_bytes: float = 0.0
    dpu_to_host_payload_bytes: float = 0.0
    mram_dma_window_bytes_model: float = 0.0
    tile_iterations: float = 0.0
    host_completion_events: float = 0.0
    numeric_representation_penalty: float = 0.0
    mram_payload_pressure_ratio: float = 0.0
    wram_known_pressure_ratio: float = 0.0

    def to_json_dict(self) -> JsonDict:
        return {name: float(getattr(self, name)) for name in _WEIGHTED_PRIMITIVES_V2}


@dataclass(frozen=True)
class UpmemPathCostProfileV2:
    profile_id: str
    policy: UpmemPathCostPolicyV2
    normalization: PathCostNormalizationV2
    weights: PathCostWeightsV2

    def normalize(self, components: PathCostComponentsV2) -> JsonDict:
        return self.normalization.normalize(components)

    def score(self, components: PathCostComponentsV2) -> float:
        if not components.feasibility:
            return math.inf
        normalized = self.normalize(components)
        return float(sum(normalized[name] * getattr(self.weights, name) for name in _WEIGHTED_PRIMITIVES_V2))

    def to_json_dict(self) -> JsonDict:
        return {
            "profile_id": self.profile_id,
            "profile_version": "v2",
            "policy": self.policy.to_json_dict(),
            "normalization": self.normalization.to_json_dict(),
            "weights": self.weights.to_json_dict(),
            "weighted_primitives": list(_WEIGHTED_PRIMITIVES_V2),
        }


_WEIGHTED_PRIMITIVES_V2 = (
    "estimated_flops",
    "host_to_dpu_payload_bytes",
    "dpu_to_host_payload_bytes",
    "mram_dma_window_bytes_model",
    "tile_iterations",
    "host_completion_events",
    "numeric_representation_penalty",
    "mram_payload_pressure_ratio",
    "wram_known_pressure_ratio",
)


def upmem_path_cost_policy_v2(
    policy_id: str = DEFAULT_UPMEM_PATH_COST_POLICY_V2,
    *,
    caps: GenericTaskPreparationCaps | None = None,
    output_tile_elements: int = GENERIC_OUTPUT_TILE_ELEMENTS,
) -> UpmemPathCostPolicyV2:
    if policy_id != GENERIC_SINGLE_DPU_SPLIT_COMPLEX_V2:
        raise ValueError(f"unknown UPMEM v2 path cost policy: {policy_id}")
    return UpmemPathCostPolicyV2(
        policy_id=policy_id,
        caps=caps or GenericTaskPreparationCaps(),
        output_tile_elements=output_tile_elements,
    )


def fixed_log1p_generic_budgets_v2(
    policy: UpmemPathCostPolicyV2 | None = None,
) -> PathCostNormalizationV2:
    active = policy or upmem_path_cost_policy_v2()
    max_elements = int(active.caps.max_tensor_elements)
    max_contracted = int(active.caps.max_contracted_combinations)
    max_tiles = max(1, math.ceil(max_elements / active.output_tile_elements))
    max_output_write = _aligned_output_tile_bytes(max_elements, active.output_tile_elements, active.output_element_bytes)
    cap_values = {
        "estimated_flops": 4 * 2 * max_elements * max_contracted + 2 * max_elements,
        "host_to_dpu_payload_bytes": 4 * 2 * max_elements * active.input_element_bytes,
        "dpu_to_host_payload_bytes": 4 * max_elements * active.output_element_bytes,
        "mram_dma_window_bytes_model": 4 * (2 * max_elements * max_contracted * active.scalar_read_window_bytes + max_output_write),
        "tile_iterations": 4 * max_tiles,
        "host_completion_events": 4.0,
        "numeric_representation_penalty": 3.0,
        "mram_payload_pressure_ratio": 1.0,
        "wram_known_pressure_ratio": 1.0,
    }
    return PathCostNormalizationV2(
        normalization_id=FIXED_LOG1P_GENERIC_BUDGETS_V2,
        caps=tuple((name, float(cap_values[name])) for name in _WEIGHTED_PRIMITIVES_V2),
    )


def upmem_path_cost_profile_v2(
    profile_id: PathCostProfileIdV2 = "balanced_literature_informed",
    *,
    policy: UpmemPathCostPolicyV2 | None = None,
) -> UpmemPathCostProfileV2:
    active = policy or upmem_path_cost_policy_v2()
    return UpmemPathCostProfileV2(
        profile_id=profile_id,
        policy=active,
        normalization=fixed_log1p_generic_budgets_v2(active),
        weights=_profile_weights_v2(profile_id),
    )


def task_numeric_execution(
    left_is_complex: bool,
    right_is_complex: bool,
    output_elements: int,
) -> TaskNumericExecution:
    """Return the current strict-runtime decomposition for one task."""
    if left_is_complex or right_is_complex:
        return TaskNumericExecution(
            representation="split_real_imag",
            component_invocations=4,
            recombination_flops=2 * int(output_elements),
        )
    return TaskNumericExecution("real_float32", 1, 0)


def task_numeric_executions_for_path(
    network: TensorNetworkValue,
    tasks: Sequence[ContractionTask],
) -> tuple[dict[str, TaskNumericExecution], tuple[str, ...]]:
    """Derive the conservative numeric representation of every ordered task."""
    complex_by_tensor: dict[str, bool] = {}
    reasons: list[str] = []
    for tensor in network.tensors:
        classification = classify_numeric(np.asarray(tensor.array))
        if classification.has_nonfinite:
            reasons.append("nonfinite_values_not_supported")
        complex_by_tensor[tensor.spec.id] = classification.has_nonzero_imaginary
    if reasons:
        return {}, tuple(dict.fromkeys(reasons))

    executions: dict[str, TaskNumericExecution] = {}
    for task in tasks:
        try:
            left_is_complex = complex_by_tensor[task.input_tensor_ids[0]]
            right_is_complex = complex_by_tensor[task.input_tensor_ids[1]]
        except KeyError:
            return {}, ("numeric_tensor_dependency_missing",)
        output_elements = _shape_product(task.output_shape)
        execution = task_numeric_execution(left_is_complex, right_is_complex, output_elements)
        executions[task.id] = execution
        # A contraction with a complex input is conservatively treated as
        # complex even if a specific value happens to cancel its imaginary part.
        complex_by_tensor[task.output_tensor_id] = execution.representation == "split_real_imag"
    return executions, ()


def model_upmem_task_cost_v2(
    task: ContractionTask,
    policy: UpmemPathCostPolicyV2 | None = None,
    *,
    numeric_execution: TaskNumericExecution | None = None,
) -> PathCostComponentsV2:
    return _model_upmem_structural_cost_v2(
        task.input_shapes,
        task.output_shape,
        task.left_labels,
        task.right_labels,
        task.contracted_labels,
        task.output_labels,
        policy,
        numeric_execution=numeric_execution,
    )


def _model_upmem_structural_cost_v2(
    input_shapes: tuple[tuple[int, ...], tuple[int, ...]],
    output_shape: tuple[int, ...],
    left_labels: tuple[int, ...],
    right_labels: tuple[int, ...],
    contracted_labels: tuple[int, ...],
    output_labels: tuple[int, ...],
    policy: UpmemPathCostPolicyV2 | None = None,
    *,
    numeric_execution: TaskNumericExecution | None = None,
) -> PathCostComponentsV2:
    active = policy or upmem_path_cost_policy_v2()
    structural = generic_structural_feasibility_from_metadata(
        input_shapes=input_shapes,
        output_shape=output_shape,
        left_labels=left_labels,
        right_labels=right_labels,
        contracted_labels=contracted_labels,
        output_labels=output_labels,
        caps=active.caps,
        check_int32_accumulation=False,
    )
    if not structural.feasible:
        return PathCostComponentsV2(feasibility=False, rejection_reasons=structural.rejection_reasons)
    if active.native_static_mram_reservation_bytes > active.mram_capacity_bytes:
        return PathCostComponentsV2(
            feasibility=False,
            rejection_reasons=("configured_native_static_mram_reservation_exceeds_capacity",),
        )
    if active.known_wram_static_bytes > active.wram_budget_bytes:
        return PathCostComponentsV2(
            feasibility=False,
            rejection_reasons=("configured_known_wram_static_bytes_exceed_budget",),
        )
    execution = numeric_execution or task_numeric_execution(False, False, _shape_product(output_shape))
    if not execution.feasible:
        return PathCostComponentsV2(feasibility=False, rejection_reasons=(execution.rejection_reason or "numeric_contract_rejected",))
    components = _components_from_structural_v2(input_shapes, structural.metadata, active, execution)
    if components.task_mram_payload_bytes > active.native_static_mram_reservation_bytes:
        return replace(
            components,
            feasibility=False,
            rejection_reasons=("mram_live_payload_exceeds_native_static_reservation",),
        )
    if components.task_mram_payload_bytes > active.mram_capacity_bytes:
        return replace(
            components,
            feasibility=False,
            rejection_reasons=("mram_live_payload_exceeds_configured_capacity",),
        )
    return components


def model_upmem_contract_cost_v2(
    node: ContractNode,
    policy: UpmemPathCostPolicyV2 | None = None,
    *,
    numeric_execution: TaskNumericExecution | None = None,
) -> PathCostComponentsV2:
    """Model a semantic DAG contraction with the established v2 formulas."""

    shared_contracted_labels = tuple(
        label
        for label in node.contracted_labels
        if label in node.left.labels and label in node.right.labels
    )
    return _model_upmem_structural_cost_v2(
        input_shapes=(node.left.shape, node.right.shape),
        output_shape=node.output.shape,
        left_labels=node.left.labels,
        right_labels=node.right.labels,
        contracted_labels=shared_contracted_labels,
        output_labels=node.output_labels,
        policy=policy,
        numeric_execution=numeric_execution,
    )


def model_upmem_path_cost_v2(
    tasks: Sequence[ContractionTask],
    policy: UpmemPathCostPolicyV2 | None = None,
    *,
    numeric_executions: Mapping[str, TaskNumericExecution] | None = None,
) -> PathCostComponentsV2:
    active = policy or upmem_path_cost_policy_v2()
    components = [
        model_upmem_task_cost_v2(task, active, numeric_execution=(numeric_executions or {}).get(task.id))
        for task in tasks
    ]
    return combine_path_cost_components_v2(components, active)


def model_upmem_network_path_cost_v2(
    network: TensorNetworkValue,
    tasks: Sequence[ContractionTask],
    policy: UpmemPathCostPolicyV2 | None = None,
) -> PathCostComponentsV2:
    executions, reasons = task_numeric_executions_for_path(network, tasks)
    if reasons:
        return PathCostComponentsV2(feasibility=False, rejection_reasons=reasons)
    return model_upmem_path_cost_v2(tasks, policy, numeric_executions=executions)


def combine_path_cost_components_v2(
    components: Sequence[PathCostComponentsV2],
    policy: UpmemPathCostPolicyV2 | None = None,
) -> PathCostComponentsV2:
    active = policy or upmem_path_cost_policy_v2()
    if not components:
        return PathCostComponentsV2(
            native_static_mram_reservation_bytes=active.native_static_mram_reservation_bytes,
            mram_capacity_bytes=active.mram_capacity_bytes,
            mram_static_reservation_pressure_ratio=(
                active.native_static_mram_reservation_bytes / active.mram_capacity_bytes
            ),
            mram_max_region_payload_ratio=0.0,
            known_wram_static_bytes=active.known_wram_static_bytes,
            wram_budget_bytes=active.wram_budget_bytes,
            mram_payload_pressure_ratio=active.native_static_mram_reservation_bytes / active.mram_capacity_bytes,
            wram_known_pressure_ratio=active.known_wram_static_bytes / active.wram_budget_bytes,
        )
    reasons = tuple(dict.fromkeys(reason for item in components for reason in item.rejection_reasons))
    return PathCostComponentsV2(
        estimated_flops=sum(item.estimated_flops for item in components),
        largest_tensor_bytes=max(item.largest_tensor_bytes for item in components),
        host_to_dpu_payload_bytes=sum(item.host_to_dpu_payload_bytes for item in components),
        dpu_to_host_payload_bytes=sum(item.dpu_to_host_payload_bytes for item in components),
        mram_dma_window_bytes_model=sum(item.mram_dma_window_bytes_model for item in components),
        tile_iterations=sum(item.tile_iterations for item in components),
        host_completion_events=sum(item.host_completion_events for item in components),
        numeric_component_invocations=sum(item.numeric_component_invocations for item in components),
        numeric_recombination_flops=sum(item.numeric_recombination_flops for item in components),
        numeric_representation_penalty=sum(item.numeric_representation_penalty for item in components),
        task_mram_payload_bytes=max(item.task_mram_payload_bytes for item in components),
        native_static_mram_reservation_bytes=active.native_static_mram_reservation_bytes,
        mram_capacity_bytes=active.mram_capacity_bytes,
        mram_static_reservation_pressure_ratio=(
            active.native_static_mram_reservation_bytes / active.mram_capacity_bytes
        ),
        mram_max_region_payload_ratio=max(item.mram_max_region_payload_ratio for item in components),
        mram_payload_pressure_ratio=max(item.mram_payload_pressure_ratio for item in components),
        known_wram_static_bytes=active.known_wram_static_bytes,
        wram_budget_bytes=active.wram_budget_bytes,
        wram_known_pressure_ratio=max(item.wram_known_pressure_ratio for item in components),
        feasibility=all(item.feasibility for item in components),
        rejection_reasons=reasons,
    )


def _components_from_structural_v2(
    input_shapes: tuple[tuple[int, ...], tuple[int, ...]],
    metadata: Mapping[str, object],
    policy: UpmemPathCostPolicyV2,
    execution: TaskNumericExecution,
) -> PathCostComponentsV2:
    output_elements = int(metadata["output_element_count"])
    contracted_count = int(metadata["contracted_combination_count"])
    left_elements = _shape_product(input_shapes[0])
    right_elements = _shape_product(input_shapes[1])
    component_count = int(execution.component_invocations)
    output_bytes = output_elements * policy.output_element_bytes
    left_bytes = left_elements * policy.input_element_bytes
    right_bytes = right_elements * policy.input_element_bytes
    tile_iterations = int(math.ceil(output_elements / policy.output_tile_elements)) * component_count
    output_window_bytes = _aligned_output_tile_bytes(output_elements, policy.output_tile_elements, policy.output_element_bytes)
    scalar_read_window_bytes = 2 * output_elements * contracted_count * policy.scalar_read_window_bytes
    task_payload_bytes = _align8(left_bytes) + _align8(right_bytes) + output_window_bytes
    max_payload_region = max(_align8(left_bytes), _align8(right_bytes), output_window_bytes)
    max_region_capacity_bytes = policy.native_max_tensor_elements * policy.output_element_bytes
    return PathCostComponentsV2(
        estimated_flops=component_count * 2 * output_elements * contracted_count + execution.recombination_flops,
        largest_tensor_bytes=max(left_bytes, right_bytes, output_bytes),
        host_to_dpu_payload_bytes=component_count * (left_bytes + right_bytes),
        dpu_to_host_payload_bytes=component_count * output_bytes,
        mram_dma_window_bytes_model=component_count * (scalar_read_window_bytes + output_window_bytes),
        tile_iterations=tile_iterations,
        host_completion_events=component_count,
        numeric_component_invocations=component_count,
        numeric_recombination_flops=execution.recombination_flops,
        numeric_representation_penalty=float(component_count - 1),
        task_mram_payload_bytes=task_payload_bytes,
        native_static_mram_reservation_bytes=policy.native_static_mram_reservation_bytes,
        mram_capacity_bytes=policy.mram_capacity_bytes,
        mram_static_reservation_pressure_ratio=(
            policy.native_static_mram_reservation_bytes / policy.mram_capacity_bytes
        ),
        mram_max_region_payload_ratio=max_payload_region / max_region_capacity_bytes,
        mram_payload_pressure_ratio=max_payload_region / max_region_capacity_bytes,
        known_wram_static_bytes=policy.known_wram_static_bytes,
        wram_budget_bytes=policy.wram_budget_bytes,
        wram_known_pressure_ratio=policy.known_wram_static_bytes / policy.wram_budget_bytes,
    )


def _normalizable_values_v2(components: PathCostComponentsV2) -> dict[str, float]:
    return {name: float(getattr(components, name)) for name in _WEIGHTED_PRIMITIVES_V2}


def _profile_weights_v2(profile_id: str) -> PathCostWeightsV2:
    profiles = {
        "compute_oriented": PathCostWeightsV2(estimated_flops=1.0, host_completion_events=0.1),
        "host_transfer_oriented": PathCostWeightsV2(host_to_dpu_payload_bytes=0.6, dpu_to_host_payload_bytes=0.4),
        "local_movement_oriented": PathCostWeightsV2(mram_dma_window_bytes_model=0.8, tile_iterations=0.2),
        "wram_constrained": PathCostWeightsV2(mram_payload_pressure_ratio=0.6, wram_known_pressure_ratio=0.4),
        "synchronization_constrained": PathCostWeightsV2(host_completion_events=0.8, tile_iterations=0.2),
        "balanced_literature_informed": PathCostWeightsV2(
            estimated_flops=0.25,
            host_to_dpu_payload_bytes=0.15,
            dpu_to_host_payload_bytes=0.05,
            mram_dma_window_bytes_model=0.2,
            tile_iterations=0.05,
            host_completion_events=0.1,
            numeric_representation_penalty=0.1,
            mram_payload_pressure_ratio=0.05,
            wram_known_pressure_ratio=0.05,
        ),
    }
    try:
        return profiles[profile_id]
    except KeyError as exc:
        raise ValueError(f"unknown UPMEM v2 path-cost profile: {profile_id}") from exc


def _shape_product(shape: Sequence[int]) -> int:
    product = 1
    for dim in shape:
        product *= int(dim)
    return int(product)


def _align8(value: int) -> int:
    return (int(value) + 7) & ~7


def _aligned_output_tile_bytes(output_elements: int, tile_elements: int, element_bytes: int) -> int:
    remaining = int(output_elements)
    total = 0
    while remaining:
        current = min(remaining, tile_elements)
        total += _align8(current * element_bytes)
        remaining -= current
    return total


def _fixed_log1p(value: float, cap: float) -> float:
    if cap <= 0:
        return 0.0
    return float(math.log1p(max(0.0, value)) / math.log1p(cap))


def calibrate_pim_cost_model(empirical_dpu_cycles: int | None, modeled_flops: int | None) -> float | None:
    """Calculate C_PIM operational calibration ratio safely.

    C_PIM = empirical_dpu_cycles / modeled_flops
    Returns None if empirical_dpu_cycles or modeled_flops is <= 0 or None.
    """
    if empirical_dpu_cycles is None or modeled_flops is None:
        return None
    cycles = int(empirical_dpu_cycles)
    flops = int(modeled_flops)
    if cycles > 0 and flops > 0:
        return float(cycles) / float(flops)
    return None


def metric_contract_v2() -> dict[str, JsonDict]:
    """Compact machine-readable V2 metric contract mapping primitive metrics.

    Each metric maps to unit, origin analytic_model, scope, and model_id.
    """
    return {
        "estimated_flops": {
            "unit": "flops",
            "origin": "analytic_model",
            "scope": "path_total_modeled_floating_point_operations",
            "model_id": UPMEM_PATH_OBJECTIVE_V2,
        },
        "largest_tensor_bytes": {
            "unit": "bytes",
            "origin": "analytic_model",
            "scope": "maximum_single_tensor_payload_bytes",
            "model_id": UPMEM_PATH_OBJECTIVE_V2,
        },
        "host_to_dpu_payload_bytes": {
            "unit": "bytes",
            "origin": "analytic_model",
            "scope": "modeled_host_to_dpu_operand_transfer_volume",
            "model_id": UPMEM_PATH_OBJECTIVE_V2,
        },
        "dpu_to_host_payload_bytes": {
            "unit": "bytes",
            "origin": "analytic_model",
            "scope": "modeled_dpu_to_host_result_transfer_volume",
            "model_id": UPMEM_PATH_OBJECTIVE_V2,
        },
        "mram_dma_window_bytes_model": {
            "unit": "bytes",
            "origin": "analytic_model",
            "scope": "modeled_aligned_request_volume_not_physical_bus_bytes",
            "model_id": UPMEM_PATH_OBJECTIVE_V2,
        },
        "tile_iterations": {
            "unit": "iterations",
            "origin": "analytic_model",
            "scope": "modeled_output_tile_execution_iterations",
            "model_id": UPMEM_PATH_OBJECTIVE_V2,
        },
        "host_completion_events": {
            "unit": "events",
            "origin": "analytic_model",
            "scope": "one_synchronous_dpu_launch_per_real_component",
            "model_id": UPMEM_PATH_OBJECTIVE_V2,
        },
        "numeric_component_invocations": {
            "unit": "count",
            "origin": "analytic_model",
            "scope": "dpu_kernel_launches_for_real_split_components",
            "model_id": UPMEM_PATH_OBJECTIVE_V2,
        },
        "numeric_recombination_flops": {
            "unit": "flops",
            "origin": "analytic_model",
            "scope": "host_recombination_floating_point_operations",
            "model_id": UPMEM_PATH_OBJECTIVE_V2,
        },
        "numeric_representation_penalty": {
            "unit": "ratio",
            "origin": "analytic_model",
            "scope": "penalty_for_split_complex_representation",
            "model_id": UPMEM_PATH_OBJECTIVE_V2,
        },
        "task_mram_payload_bytes": {
            "unit": "bytes",
            "origin": "analytic_model",
            "scope": "modeled_task_mram_working_set_payload",
            "model_id": UPMEM_PATH_OBJECTIVE_V2,
        },
        "native_static_mram_reservation_bytes": {
            "unit": "bytes",
            "origin": "analytic_model",
            "scope": "fixed_native_mram_buffer_reservation",
            "model_id": UPMEM_PATH_OBJECTIVE_V2,
        },
        "mram_capacity_bytes": {
            "unit": "bytes",
            "origin": "analytic_model",
            "scope": "configured_modeled_budget_not_measured_runtime_occupancy",
            "model_id": UPMEM_PATH_OBJECTIVE_V2,
        },
        "mram_static_reservation_pressure_ratio": {
            "unit": "ratio",
            "origin": "analytic_model",
            "scope": "native_static_reservation_divided_by_mram_capacity",
            "model_id": UPMEM_PATH_OBJECTIVE_V2,
        },
        "mram_max_region_payload_ratio": {
            "unit": "ratio",
            "origin": "analytic_model",
            "scope": "maximum_single_buffer_payload_divided_by_fixed_native_buffer_capacity",
            "model_id": UPMEM_PATH_OBJECTIVE_V2,
        },
        "mram_payload_pressure_ratio": {
            "unit": "ratio",
            "origin": "analytic_model",
            "scope": "maximum_single_buffer_payload_divided_by_fixed_native_buffer_capacity",
            "model_id": UPMEM_PATH_OBJECTIVE_V2,
        },
        "known_wram_static_bytes": {
            "unit": "bytes",
            "origin": "analytic_model",
            "scope": "known_wram_static_reservation_bytes",
            "model_id": UPMEM_PATH_OBJECTIVE_V2,
        },
        "wram_budget_bytes": {
            "unit": "bytes",
            "origin": "analytic_model",
            "scope": "configured_modeled_budget_not_measured_runtime_occupancy",
            "model_id": UPMEM_PATH_OBJECTIVE_V2,
        },
        "wram_known_pressure_ratio": {
            "unit": "ratio",
            "origin": "analytic_model",
            "scope": "known_wram_static_bytes_divided_by_wram_budget",
            "model_id": UPMEM_PATH_OBJECTIVE_V2,
        },
    }


__all__ = [
    "DEFAULT_UPMEM_PATH_COST_NORMALIZATION_V2",
    "DEFAULT_UPMEM_PATH_COST_POLICY_V2",
    "FIXED_LOG1P_GENERIC_BUDGETS_V2",
    "GENERIC_SINGLE_DPU_SPLIT_COMPLEX_V2",
    "PathCostComponentsV2",
    "PathCostNormalizationV2",
    "PathCostWeightsV2",
    "TaskNumericExecution",
    "UPMEM_PATH_OBJECTIVE_V2",
    "UpmemPathCostPolicyV2",
    "UpmemPathCostProfileV2",
    "calibrate_pim_cost_model",
    "combine_path_cost_components_v2",
    "fixed_log1p_generic_budgets_v2",
    "metric_contract_v2",
    "model_upmem_network_path_cost_v2",
    "model_upmem_path_cost_v2",
    "model_upmem_task_cost_v2",
    "task_numeric_execution",
    "task_numeric_executions_for_path",
    "upmem_path_cost_policy_v2",
    "upmem_path_cost_profile_v2",
]
