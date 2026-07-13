"""Analysis-only generic UPMEM path cost components.

This module models the application-visible work and tensor movement of the
generic float32 contract. It does not claim hardware throughput or latency,
and it does not execute UPMEM work. The custom UPMEM planner consumes this
shared model to select a modeled path.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Literal, Sequence

import numpy as np

from quantum_bench.core.records import ContractionTask, JsonDict
from quantum_bench.routing.generic_prepare import (
    GENERIC_OUTPUT_TILE_ELEMENTS,
    GenericStructuralFeasibility,
    GenericTaskPreparationCaps,
    generic_structural_feasibility,
)
from quantum_bench.tn.network import TensorNetworkValue


GENERIC_SINGLE_DPU_FLOAT32_V1 = "generic_single_dpu_float32_v1"
FIXED_LOG1P_GENERIC_CAPS_V1 = "fixed_log1p_generic_caps_v1"
DEFAULT_UPMEM_PATH_COST_POLICY_ID = GENERIC_SINGLE_DPU_FLOAT32_V1
DEFAULT_UPMEM_PATH_COST_NORMALIZATION_ID = FIXED_LOG1P_GENERIC_CAPS_V1
DEFAULT_OUTPUT_TILE_ELEMENTS = GENERIC_OUTPUT_TILE_ELEMENTS

PathCostProfileId = Literal[
    "compute_oriented",
    "host_transfer_oriented",
    "local_movement_oriented",
    "wram_constrained",
    "synchronization_constrained",
    "balanced_literature_informed",
]


@dataclass(frozen=True)
class PathCostComponents:
    """Unweighted, integer-counted cost components for a task or path."""

    flops: int = 0
    peak_bytes: int = 0
    intermediate_writes: int = 0
    host_to_dpu_bytes: int = 0
    dpu_to_host_bytes: int = 0
    # Kept as the aggregate planning component used by the initial profile.
    host_dpu_bytes: int = 0
    mram_wram_bytes: int = 0
    local_work: int = 0
    sync_events: int = 0
    numeric_penalty: float = 0.0
    wram_pressure: float = 0.0
    tiles: int = 0
    feasibility: bool = True
    rejection_reasons: tuple[str, ...] = ()

    def to_json_dict(self) -> JsonDict:
        return {
            "flops": int(self.flops),
            "peak_bytes": int(self.peak_bytes),
            "intermediate_writes": int(self.intermediate_writes),
            "host_to_dpu_bytes": int(self.host_to_dpu_bytes),
            "dpu_to_host_bytes": int(self.dpu_to_host_bytes),
            "host_dpu_bytes": int(self.host_dpu_bytes),
            "mram_wram_bytes": int(self.mram_wram_bytes),
            "local_work": int(self.local_work),
            "sync_events": int(self.sync_events),
            "numeric_penalty": float(self.numeric_penalty),
            "wram_pressure": float(self.wram_pressure),
            "tiles": int(self.tiles),
            "feasibility": bool(self.feasibility),
            "rejection_reasons": list(self.rejection_reasons),
        }

    as_dict = to_json_dict


@dataclass(frozen=True)
class UpmemPathCostPolicy:
    """Explicit configuration for the generic single-DPU cost model."""

    policy_id: str = GENERIC_SINGLE_DPU_FLOAT32_V1
    caps: GenericTaskPreparationCaps = field(default_factory=GenericTaskPreparationCaps)
    output_tile_elements: int = DEFAULT_OUTPUT_TILE_ELEMENTS
    input_element_bytes: int = 4
    output_element_bytes: int = 4
    accumulator_element_bytes: int = 4
    check_int32_accumulation: bool = False

    def __post_init__(self) -> None:
        if self.output_tile_elements <= 0:
            raise ValueError("output_tile_elements must be positive")
        if min(self.input_element_bytes, self.output_element_bytes, self.accumulator_element_bytes) <= 0:
            raise ValueError("element byte widths must be positive")

    def to_json_dict(self) -> JsonDict:
        return {
            "policy_id": self.policy_id,
            "caps": {
                "max_rank": int(self.caps.max_rank),
                "max_tensor_elements": int(self.caps.max_tensor_elements),
                "max_contracted_combinations": int(self.caps.max_contracted_combinations),
            },
            "output_tile_elements": int(self.output_tile_elements),
            "input_element_bytes": int(self.input_element_bytes),
            "output_element_bytes": int(self.output_element_bytes),
            "accumulator_element_bytes": int(self.accumulator_element_bytes),
            "check_int32_accumulation": bool(self.check_int32_accumulation),
            "host_dpu_transfer_model": "per_task_operand_upload_and_output_download_no_reuse",
            "mram_wram_transfer_model": "generic_output_tiled_operand_scan",
            "synchronization_model": "one_modeled_completion_event_per_output_tile",
            "dpu_parallelism_model": "single_dpu_serial_tasks",
        }

    as_dict = to_json_dict


@dataclass(frozen=True)
class PathCostNormalization:
    """Fixed log1p denominators; values are derived solely from policy caps."""

    normalization_id: str
    caps: tuple[tuple[str, float], ...]

    def to_json_dict(self) -> JsonDict:
        return {
            "normalization_id": self.normalization_id,
            "caps": {name: float(value) for name, value in self.caps},
            "transform": "log1p(value)/log1p(cap)",
        }

    as_dict = to_json_dict

    def normalize(self, components: PathCostComponents) -> JsonDict:
        values = _normalizable_values(components)
        return {
            name: _fixed_log1p(values[name], cap)
            for name, cap in self.caps
            if name in values
        }


@dataclass(frozen=True)
class PathCostWeights:
    flops: float = 0.0
    peak_bytes: float = 0.0
    intermediate_writes: float = 0.0
    host_dpu_bytes: float = 0.0
    mram_wram_bytes: float = 0.0
    local_work: float = 0.0
    sync_events: float = 0.0
    numeric_penalty: float = 0.0
    wram_pressure: float = 0.0
    tiles: float = 0.0

    def to_json_dict(self) -> JsonDict:
        return {name: float(getattr(self, name)) for name in _COST_FIELDS}

    as_dict = to_json_dict


@dataclass(frozen=True)
class UpmemPathCostProfile:
    profile_id: str
    policy: UpmemPathCostPolicy
    normalization: PathCostNormalization
    weights: PathCostWeights

    def normalize(self, components: PathCostComponents) -> JsonDict:
        return self.normalization.normalize(components)

    def score(self, components: PathCostComponents) -> float:
        if not components.feasibility:
            return math.inf
        normalized = self.normalize(components)
        return float(sum(normalized[name] * getattr(self.weights, name) for name in _COST_FIELDS))

    def to_json_dict(self) -> JsonDict:
        return {
            "profile_id": self.profile_id,
            "policy": self.policy.to_json_dict(),
            "normalization": self.normalization.to_json_dict(),
            "weights": self.weights.to_json_dict(),
        }

    as_dict = to_json_dict


_COST_FIELDS = (
    "flops",
    "peak_bytes",
    "intermediate_writes",
    "host_dpu_bytes",
    "mram_wram_bytes",
    "local_work",
    "sync_events",
    "numeric_penalty",
    "wram_pressure",
    "tiles",
)


def default_upmem_path_cost_policy() -> UpmemPathCostPolicy:
    return UpmemPathCostPolicy()


def upmem_path_cost_policy(
    policy_id: str = DEFAULT_UPMEM_PATH_COST_POLICY_ID,
    *,
    caps: GenericTaskPreparationCaps | None = None,
    output_tile_elements: int = DEFAULT_OUTPUT_TILE_ELEMENTS,
) -> UpmemPathCostPolicy:
    if policy_id != GENERIC_SINGLE_DPU_FLOAT32_V1:
        raise ValueError(f"unknown UPMEM path cost policy: {policy_id}")
    return UpmemPathCostPolicy(
        policy_id=policy_id,
        caps=caps or GenericTaskPreparationCaps(),
        output_tile_elements=output_tile_elements,
    )


def fixed_log1p_generic_caps_v1(
    policy: UpmemPathCostPolicy | None = None,
) -> PathCostNormalization:
    active = policy or default_upmem_path_cost_policy()
    max_elements = int(active.caps.max_tensor_elements)
    max_contracted = int(active.caps.max_contracted_combinations)
    output_bytes = max_elements * active.output_element_bytes
    tile_count = max(1, math.ceil(max_elements / active.output_tile_elements))
    wram_pressure_cap = min(1.0, active.output_tile_elements / max(1, max_elements))
    cap_values = {
        "flops": 2 * max_elements * max_contracted,
        "peak_bytes": max_elements * max(active.input_element_bytes, active.output_element_bytes),
        "intermediate_writes": output_bytes,
        "host_dpu_bytes": (2 * max_elements * active.input_element_bytes) + output_bytes,
        "mram_wram_bytes": (2 * max_elements * max_contracted * active.input_element_bytes) + output_bytes,
        "local_work": max_elements * max_contracted,
        "sync_events": tile_count,
        "numeric_penalty": 1.0,
        "wram_pressure": wram_pressure_cap,
        "tiles": tile_count,
    }
    return PathCostNormalization(
        normalization_id=FIXED_LOG1P_GENERIC_CAPS_V1,
        caps=tuple((name, float(cap_values[name])) for name in _COST_FIELDS),
    )


def upmem_path_cost_profile(
    profile_id: PathCostProfileId = "balanced_literature_informed",
    *,
    policy: UpmemPathCostPolicy | None = None,
) -> UpmemPathCostProfile:
    active_policy = policy or default_upmem_path_cost_policy()
    weights = _profile_weights(profile_id)
    return UpmemPathCostProfile(
        profile_id=profile_id,
        policy=active_policy,
        normalization=fixed_log1p_generic_caps_v1(active_policy),
        weights=weights,
    )


def make_upmem_path_cost_profile(
    profile_id: PathCostProfileId = "balanced_literature_informed",
    *,
    policy: UpmemPathCostPolicy | None = None,
) -> UpmemPathCostProfile:
    return upmem_path_cost_profile(profile_id, policy=policy)


def model_upmem_task_cost(
    task: ContractionTask,
    policy: UpmemPathCostPolicy | None = None,
) -> PathCostComponents:
    active = policy or default_upmem_path_cost_policy()
    structural = generic_structural_feasibility(
        task,
        active.caps,
        check_int32_accumulation=active.check_int32_accumulation,
    )
    if not structural.feasible:
        return PathCostComponents(feasibility=False, rejection_reasons=structural.rejection_reasons)
    return _components_from_metadata(task, structural, active)


def model_upmem_path_cost(
    tasks: Sequence[ContractionTask],
    policy: UpmemPathCostPolicy | None = None,
) -> PathCostComponents:
    components = [model_upmem_task_cost(task, policy) for task in tasks]
    if not components:
        return PathCostComponents()
    reasons = tuple(dict.fromkeys(reason for item in components for reason in item.rejection_reasons))
    return PathCostComponents(
        flops=sum(item.flops for item in components),
        peak_bytes=max(item.peak_bytes for item in components),
        intermediate_writes=sum(item.intermediate_writes for item in components),
        host_dpu_bytes=sum(item.host_dpu_bytes for item in components),
        host_to_dpu_bytes=sum(item.host_to_dpu_bytes for item in components),
        dpu_to_host_bytes=sum(item.dpu_to_host_bytes for item in components),
        mram_wram_bytes=sum(item.mram_wram_bytes for item in components),
        local_work=sum(item.local_work for item in components),
        sync_events=sum(item.sync_events for item in components),
        numeric_penalty=sum(item.numeric_penalty for item in components),
        wram_pressure=max(item.wram_pressure for item in components),
        tiles=sum(item.tiles for item in components),
        feasibility=all(item.feasibility for item in components),
        rejection_reasons=reasons,
    )


def generic_float32_network_rejection_reasons(network: TensorNetworkValue) -> tuple[str, ...]:
    """Return numeric-contract blockers for the modeled generic path.

    The current generic UPMEM model is explicitly real float32. This check is
    shared by the custom generator and post-hoc scoring of external paths so a
    standard-library path cannot be presented as feasible for a network the
    modeled executor cannot represent.
    """
    for tensor in network.tensors:
        if np.iscomplexobj(np.asarray(tensor.array)):
            return ("complex_generic_loop_not_implemented",)
    return ()


def model_upmem_network_path_cost(
    network: TensorNetworkValue,
    tasks: Sequence[ContractionTask],
    policy: UpmemPathCostPolicy | None = None,
) -> PathCostComponents:
    """Score a full path under both structural and numeric-contract bounds."""
    components = model_upmem_path_cost(tasks, policy)
    numeric_reasons = generic_float32_network_rejection_reasons(network)
    if not numeric_reasons:
        return components
    return replace(
        components,
        feasibility=False,
        rejection_reasons=tuple(dict.fromkeys((*components.rejection_reasons, *numeric_reasons))),
    )


def upmem_path_cost(
    tasks: Sequence[ContractionTask],
    policy: UpmemPathCostPolicy | None = None,
) -> PathCostComponents:
    return model_upmem_path_cost(tasks, policy)


def normalize_upmem_path_cost(
    components: PathCostComponents,
    normalization: PathCostNormalization | None = None,
) -> JsonDict:
    return (normalization or fixed_log1p_generic_caps_v1()).normalize(components)


def _components_from_metadata(
    task: ContractionTask,
    structural: GenericStructuralFeasibility,
    policy: UpmemPathCostPolicy,
) -> PathCostComponents:
    metadata = structural.metadata
    output_elements = int(metadata["output_element_count"])
    contracted_count = int(metadata["contracted_combination_count"])
    left_elements = _shape_product(task.input_shapes[0])
    right_elements = _shape_product(task.input_shapes[1])
    tiles = int(math.ceil(output_elements / policy.output_tile_elements))
    output_bytes = output_elements * policy.output_element_bytes
    host_to_dpu_bytes = (
        left_elements * policy.input_element_bytes
        + right_elements * policy.input_element_bytes
    )
    dpu_to_host_bytes = output_bytes
    mram_wram_bytes = (
        2 * output_elements * contracted_count * policy.input_element_bytes
        + output_bytes
    )
    tile_output_elements = min(output_elements, policy.output_tile_elements)
    wram_pressure = (
        tile_output_elements * policy.output_element_bytes
        / (policy.caps.max_tensor_elements * policy.output_element_bytes)
    )
    return PathCostComponents(
        flops=2 * output_elements * contracted_count,
        peak_bytes=max(
            left_elements * policy.input_element_bytes,
            right_elements * policy.input_element_bytes,
            output_bytes,
        ),
        intermediate_writes=output_bytes,
        host_to_dpu_bytes=host_to_dpu_bytes,
        dpu_to_host_bytes=dpu_to_host_bytes,
        host_dpu_bytes=host_to_dpu_bytes + dpu_to_host_bytes,
        mram_wram_bytes=mram_wram_bytes,
        local_work=output_elements * contracted_count,
        sync_events=tiles,
        numeric_penalty=0.0,
        wram_pressure=float(wram_pressure),
        tiles=tiles,
    )


def _profile_weights(profile_id: str) -> PathCostWeights:
    profiles = {
        "compute_oriented": PathCostWeights(flops=1.0, local_work=0.35, host_dpu_bytes=0.1),
        "host_transfer_oriented": PathCostWeights(host_dpu_bytes=1.0, intermediate_writes=0.2, peak_bytes=0.1),
        "local_movement_oriented": PathCostWeights(mram_wram_bytes=1.0, local_work=0.2, tiles=0.1),
        "wram_constrained": PathCostWeights(wram_pressure=1.0, peak_bytes=0.35, intermediate_writes=0.2),
        "synchronization_constrained": PathCostWeights(sync_events=1.0, tiles=0.5, host_dpu_bytes=0.1),
        # This is a transparent sensitivity profile, not a hardware-calibrated claim.
        "balanced_literature_informed": PathCostWeights(
            flops=0.25,
            peak_bytes=0.05,
            intermediate_writes=0.05,
            host_dpu_bytes=0.2,
            mram_wram_bytes=0.2,
            local_work=0.1,
            sync_events=0.05,
            numeric_penalty=0.05,
            wram_pressure=0.05,
            tiles=0.05,
        ),
    }
    try:
        return profiles[profile_id]
    except KeyError as exc:
        raise ValueError(f"unknown UPMEM path cost profile: {profile_id}") from exc


def _normalizable_values(components: PathCostComponents) -> dict[str, float]:
    return {name: float(getattr(components, name)) for name in _COST_FIELDS}


def _fixed_log1p(value: float, cap: float) -> float:
    if cap <= 0.0:
        return 0.0
    return float(math.log1p(max(0.0, value)) / math.log1p(cap))


def _shape_product(shape: tuple[int, ...]) -> int:
    product = 1
    for dim in shape:
        product *= int(dim)
    return int(product)


__all__ = [
    "DEFAULT_OUTPUT_TILE_ELEMENTS",
    "DEFAULT_UPMEM_PATH_COST_NORMALIZATION_ID",
    "DEFAULT_UPMEM_PATH_COST_POLICY_ID",
    "FIXED_LOG1P_GENERIC_CAPS_V1",
    "GENERIC_SINGLE_DPU_FLOAT32_V1",
    "PathCostComponents",
    "PathCostNormalization",
    "PathCostProfileId",
    "PathCostWeights",
    "UpmemPathCostPolicy",
    "UpmemPathCostProfile",
    "default_upmem_path_cost_policy",
    "fixed_log1p_generic_caps_v1",
    "generic_float32_network_rejection_reasons",
    "make_upmem_path_cost_profile",
    "model_upmem_path_cost",
    "model_upmem_network_path_cost",
    "model_upmem_task_cost",
    "normalize_upmem_path_cost",
    "upmem_path_cost",
    "upmem_path_cost_policy",
    "upmem_path_cost_profile",
]
