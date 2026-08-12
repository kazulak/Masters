"""UPMEM PIM-aware contraction path optimizer using functional programming principles.

This module implements pure functional cost calculations and state transitions
to find optimal tensor network contraction paths for UPMEM PIM target hardware.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

import opt_einsum as oe
from opt_einsum.paths import PathOptimizer

from quantum_bench.core.records import ContractionTask
from quantum_bench.tn.upmem_path_cost_v2 import (
    PathCostComponentsV2,
    UpmemPathCostPolicyV2,
    model_upmem_task_cost_v2,
    upmem_path_cost_policy_v2,
)


@dataclass(frozen=True)
class PIMCostParameters:
    """Centralized immutable registry of all PIM path cost weights and policy parameters."""

    w_flops: float = 1.0
    w_h2d: float = 1.0
    w_d2h: float = 1.0
    w_mram_dma: float = 1.0
    w_wram: float = 1.0
    w_sync: float = 1.0
    w_complex_penalty: float = 1.0
    scale_h2d: float = 1.0
    scale_d2h: float = 1.0
    memory_limit: int | None = None

    @classmethod
    def from_config(cls, config: Mapping[str, Any] | None = None) -> PIMCostParameters:
        """Instantiate parameters from an abstract dictionary mapping."""
        if not config:
            return cls()
        valid_keys = {k for k in cls.__dataclass_fields__}
        filtered = {k: v for k, v in config.items() if k in valid_keys and v is not None}
        return cls(**filtered)

    def compute_scalar_cost(self, components: PathCostComponentsV2) -> float:
        """Pure calculation of scalar cost from PathCostComponentsV2."""
        if not components.feasibility:
            return math.inf
        return (
            self.w_flops * float(components.estimated_flops)
            + self.w_h2d * float(components.host_to_dpu_payload_bytes) * self.scale_h2d
            + self.w_d2h * float(components.dpu_to_host_payload_bytes) * self.scale_d2h
            + self.w_mram_dma * float(components.mram_dma_window_bytes_model)
            + self.w_wram * float(components.wram_known_pressure_ratio) * 1000.0
            + self.w_sync * float(components.host_completion_events)
            + self.w_complex_penalty * float(components.numeric_representation_penalty)
        )


@dataclass(frozen=True)
class PathSearchState:
    """Immutable state container during functional path optimization."""

    active_tensors: tuple[tuple[str, ...], ...]
    size_dict: Mapping[str, int]
    history: tuple[tuple[int, int], ...]
    total_cost: float
    parameters: PIMCostParameters
    output_labels: tuple[str, ...] = ()


def _shape_product(shape: Sequence[int]) -> int:
    result = 1
    for dim in shape:
        result *= int(dim)
    return result


def make_sim_contraction_task(
    task_id: str,
    left_labels: tuple[str, ...],
    right_labels: tuple[str, ...],
    output_labels: tuple[str, ...],
    size_dict: Mapping[str, int],
) -> ContractionTask:
    """Pure helper constructing a mock ContractionTask for cost evaluation."""
    set_i = set(left_labels)
    set_j = set(right_labels)
    out_set = set(output_labels)
    contracted_labels = tuple(sorted((set_i & set_j) - out_set))

    left_shape = tuple(size_dict[idx] for idx in left_labels)
    right_shape = tuple(size_dict[idx] for idx in right_labels)
    output_shape = tuple(size_dict[idx] for idx in output_labels)

    m = _shape_product(output_shape)
    k = max(1, _shape_product(tuple(size_dict[idx] for idx in contracted_labels)))
    n = 1
    flops = 8 * m * k
    bytes_est = (_shape_product(left_shape) + _shape_product(right_shape) + _shape_product(output_shape)) * 8

    return ContractionTask(
        id=task_id,
        input_tensor_ids=("t0", "t1"),
        output_tensor_id="tout",
        dependencies=(),
        index_expression="sim",
        input_shapes=(left_shape, right_shape),
        output_shape=output_shape,
        left_labels=left_labels,
        right_labels=right_labels,
        contracted_labels=contracted_labels,
        output_labels=output_labels,
        gemm_m=m,
        gemm_k=k,
        gemm_n=n,
        structure="dense",
        estimated_flops=flops,
        estimated_bytes=bytes_est,
    )


def calculate_pim_step_cost(
    left_labels: tuple[str, ...],
    right_labels: tuple[str, ...],
    out_labels: tuple[str, ...],
    size_dict: Mapping[str, int],
    params: PIMCostParameters,
    policy: UpmemPathCostPolicyV2 | None = None,
) -> float:
    """Pure cost calculation for a single candidate pairwise contraction step."""
    task = make_sim_contraction_task("step_cost_eval", left_labels, right_labels, out_labels, size_dict)
    active_policy = policy or upmem_path_cost_policy_v2()
    components = model_upmem_task_cost_v2(task, active_policy)
    return params.compute_scalar_cost(components)


def _derive_next_active(
    active: tuple[tuple[str, ...], ...],
    pair: tuple[int, int],
    new_tensor: tuple[str, ...],
) -> tuple[tuple[str, ...], ...]:
    i, j = pair
    remaining = [t for idx, t in enumerate(active) if idx not in {i, j}]
    remaining.append(new_tensor)
    return tuple(remaining)


def eval_pair_step(
    state: PathSearchState,
    pair: tuple[int, int],
) -> tuple[PathSearchState, float]:
    """Pure state transformation evaluating the cost of contracting a pair."""
    i, j = pair
    left_labels = state.active_tensors[i]
    right_labels = state.active_tensors[j]

    output_set = set(state.output_labels)
    set_i = set(left_labels)
    set_j = set(right_labels)

    other_indices: set[str] = set()
    for k, s in enumerate(state.active_tensors):
        if k != i and k != j:
            other_indices |= set(s)

    out_set = (set_i | set_j) & (output_set | other_indices)
    out_labels = tuple(sorted(out_set))

    step_cost = calculate_pim_step_cost(
        left_labels,
        right_labels,
        out_labels,
        state.size_dict,
        state.parameters,
    )

    next_active = _derive_next_active(state.active_tensors, pair, out_labels)

    next_state = PathSearchState(
        active_tensors=next_active,
        size_dict=state.size_dict,
        history=state.history + (pair,),
        total_cost=state.total_cost + step_cost,
        parameters=state.parameters,
        output_labels=state.output_labels,
    )

    return next_state, step_cost


def _greedy_search_pure(state: PathSearchState) -> PathSearchState:
    """Pure recursive greedy path search state transformer."""
    if len(state.active_tensors) <= 1:
        return state

    best_pair: tuple[int, int] | None = None
    best_next_state: PathSearchState | None = None
    best_step_cost = math.inf

    n_active = len(state.active_tensors)
    for i in range(n_active):
        for j in range(i + 1, n_active):
            pair = (i, j)
            candidate_state, step_cost = eval_pair_step(state, pair)
            if step_cost < best_step_cost:
                best_step_cost = step_cost
                best_pair = pair
                best_next_state = candidate_state

    if best_next_state is None:
        # Fallback to first-pair if all evaluation costs fail
        best_next_state, _ = eval_pair_step(state, (0, 1))

    return _greedy_search_pure(best_next_state)


class PIMPathCostOptimizer(PathOptimizer):
    """PathOptimizer subclass implementing opt_einsum's custom path optimizer interface."""

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        self.parameters = PIMCostParameters.from_config(config)

    def __call__(
        self,
        inputs: list[set[str]],
        output: set[str],
        size_dict: dict[str, int],
        memory_limit: int | None = None,
        **kwargs: Any,
    ) -> list[tuple[int, int]]:
        actual_params = replace(self.parameters, memory_limit=memory_limit) if memory_limit is not None else self.parameters
        initial_state = PathSearchState(
            active_tensors=tuple(tuple(sorted(s)) for s in inputs),
            size_dict=size_dict,
            history=(),
            total_cost=0.0,
            parameters=actual_params,
            output_labels=tuple(sorted(output)),
        )
        final_state = _greedy_search_pure(initial_state)
        return list(final_state.history)


def pim_path_finder_functional(
    inputs: list[set[str]],
    output: set[str],
    size_dict: dict[str, int],
    memory_limit: int | None = None,
    **kwargs: Any,
) -> list[tuple[int, int]]:
    """Pure functional path optimizer compatible with opt_einsum custom path optimizer interface."""
    optimizer = PIMPathCostOptimizer(config=kwargs.get("config"))
    return optimizer(inputs, output, size_dict, memory_limit=memory_limit, **kwargs)
