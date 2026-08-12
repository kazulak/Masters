"""UPMEM PIM-aware contraction path optimizer using functional programming principles.

This module implements pure functional cost calculations and state transitions
to find optimal tensor network contraction paths for UPMEM PIM target hardware.
"""

from __future__ import annotations

import math
from collections import Counter
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

# Empirical scaling penalty ratio applied to WRAM memory pressure ratios (wram_bytes_used / wram_capacity)
# to heavily penalize candidate steps that threaten WRAM overflow.
WRAM_PRESSURE_SCALING_PENALTY: float = 1000.0


def _sort_key(label: Any) -> tuple[int, Any]:
    """Type-aware sort key ensuring integers sort numerically (2 < 10) and strings sort alphabetically."""
    return (1 if isinstance(label, str) else 0, label)


@dataclass(frozen=True)
class PIMCostParameters:
    """Centralized immutable registry of all PIM path cost weights and policy parameters.

    Attributes:
        w_flops: Scalar weight for FLOP cost component.
        w_h2d: Scalar weight per Host-to-DPU transfer byte.
        w_d2h: Scalar weight per DPU-to-Host transfer byte.
        w_mram_dma: Scalar weight for MRAM DMA transfer volume.
        w_wram: Scalar weight for WRAM tile pressure ratio.
        w_sync: Scalar weight for host completion synchronization events.
        w_complex_penalty: Scalar penalty for split-complex numeric representations.
        scale_h2d: Multiplicative latency scale for Host-to-DPU transfers.
        scale_d2h: Multiplicative latency scale for DPU-to-Host transfers.
        memory_limit: Global upper memory budget in bytes across transfer payloads (H2D + D2H).
    """

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
        """Pure calculation of scalar cost from PathCostComponentsV2 with global memory enforcement."""
        if not components.feasibility:
            return math.inf

        # Enforce global memory limit across all payload transfers (H2D input + D2H output)
        total_transfer_bytes = components.host_to_dpu_payload_bytes + components.dpu_to_host_payload_bytes
        if self.memory_limit is not None and total_transfer_bytes > self.memory_limit:
            return math.inf

        return (
            self.w_flops * float(components.estimated_flops)
            + self.w_h2d * float(components.host_to_dpu_payload_bytes) * self.scale_h2d
            + self.w_d2h * float(components.dpu_to_host_payload_bytes) * self.scale_d2h
            + self.w_mram_dma * float(components.mram_dma_window_bytes_model)
            + self.w_wram * float(components.wram_known_pressure_ratio) * WRAM_PRESSURE_SCALING_PENALTY
            + self.w_sync * float(components.host_completion_events)
            + self.w_complex_penalty * float(components.numeric_representation_penalty)
        )


@dataclass(frozen=True)
class PathSearchState:
    """Immutable state container during functional path optimization."""

    active_tensors: tuple[tuple[Any, ...], ...]
    size_dict: Mapping[Any, int]
    history: tuple[tuple[int, int], ...]
    total_cost: float
    parameters: PIMCostParameters
    output_labels: tuple[Any, ...] = ()


def _shape_product(shape: Sequence[int]) -> int:
    result = 1
    for dim in shape:
        result *= int(dim)
    return result


def make_sim_contraction_task(
    task_id: str,
    left_labels: tuple[Any, ...],
    right_labels: tuple[Any, ...],
    output_labels: tuple[Any, ...],
    size_dict: Mapping[Any, int],
    dtype_bytes: int = 8,
) -> ContractionTask:
    """Pure helper constructing a mock ContractionTask for cost evaluation.

    Derives exact GEMM dimensions (B, M, N, K) accounting for:
    - Batched indices (B): indices present in left, right, AND output.
    - Contracted indices (K): indices present in left and right, but absent from output.
    - Left-only indices (M): indices present in left only.
    - Right-only indices (N): indices present in right only.
    """
    set_i = set(left_labels)
    set_j = set(right_labels)
    out_set = set(output_labels)

    batch_labels = tuple(sorted((set_i & set_j) & out_set, key=_sort_key))
    contracted_labels = tuple(sorted((set_i & set_j) - out_set, key=_sort_key))
    left_only = tuple(sorted(set_i - set_j, key=_sort_key))
    right_only = tuple(sorted(set_j - set_i, key=_sort_key))

    left_shape = tuple(size_dict[idx] for idx in left_labels)
    right_shape = tuple(size_dict[idx] for idx in right_labels)
    output_shape = tuple(size_dict[idx] for idx in output_labels)

    b = _shape_product(tuple(size_dict[idx] for idx in batch_labels))
    m = _shape_product(tuple(size_dict[idx] for idx in left_only))
    n = _shape_product(tuple(size_dict[idx] for idx in right_only))
    k = _shape_product(tuple(size_dict[idx] for idx in contracted_labels))

    # Complex GEMM FLOP count: 2 ops (add/mul) * 4 float muls per complex mul = 8 * B * M * N * K
    flops = 8 * b * m * n * k
    bytes_est = (_shape_product(left_shape) + _shape_product(right_shape) + _shape_product(output_shape)) * dtype_bytes

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
        gemm_m=b * m,
        gemm_k=k,
        gemm_n=n,
        structure="dense",
        estimated_flops=flops,
        estimated_bytes=bytes_est,
    )


def calculate_pim_step_cost(
    left_labels: tuple[Any, ...],
    right_labels: tuple[Any, ...],
    out_labels: tuple[Any, ...],
    size_dict: Mapping[Any, int],
    params: PIMCostParameters,
    policy: UpmemPathCostPolicyV2 | None = None,
) -> float:
    """Pure cost calculation for a single candidate pairwise contraction step."""
    task = make_sim_contraction_task("step_cost_eval", left_labels, right_labels, out_labels, size_dict)
    active_policy = policy or upmem_path_cost_policy_v2()
    components = model_upmem_task_cost_v2(task, active_policy)
    return params.compute_scalar_cost(components)


def _derive_next_active(
    active: tuple[tuple[Any, ...], ...],
    pair: tuple[int, int],
    new_tensor: tuple[Any, ...],
) -> tuple[tuple[Any, ...], ...]:
    """Pure tuple slicing helper generating the next sequence of active tensors without list allocations or iteration."""
    i, j = pair
    return active[:i] + active[i + 1 : j] + active[j + 1 :] + (new_tensor,)


def eval_pair_step(
    state: PathSearchState,
    pair: tuple[int, int],
    all_label_counts: Mapping[Any, int] | None = None,
) -> tuple[PathSearchState, float]:
    """Pure state transformation evaluating the cost of contracting a pair."""
    i, j = pair
    left_labels = state.active_tensors[i]
    right_labels = state.active_tensors[j]

    output_set = set(state.output_labels)
    set_i = set(left_labels)
    set_j = set(right_labels)

    if all_label_counts is not None:
        out_set: set[Any] = set()
        for idx in set_i | set_j:
            if idx in output_set:
                out_set.add(idx)
            else:
                in_pair_count = (idx in set_i) + (idx in set_j)
                if all_label_counts[idx] > in_pair_count:
                    out_set.add(idx)
    else:
        other_indices: set[Any] = set()
        for k, s in enumerate(state.active_tensors):
            if k != i and k != j:
                other_indices |= set(s)
        out_set = (set_i | set_j) & (output_set | other_indices)

    out_labels = tuple(sorted(out_set, key=_sort_key))

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


def _greedy_search_pure(initial_state: PathSearchState) -> PathSearchState:
    """Iterative state accumulator executing pure greedy search state transitions without recursion or silent failure."""
    current_state = initial_state
    while len(current_state.active_tensors) > 1:
        best_pair: tuple[int, int] | None = None
        best_next_state: PathSearchState | None = None
        best_step_cost = math.inf

        # Precompute label counts across active tensors once per step for O(N^3) overall path search
        all_label_counts = Counter(label for tensor in current_state.active_tensors for label in tensor)

        n_active = len(current_state.active_tensors)
        for i in range(n_active):
            for j in range(i + 1, n_active):
                pair = (i, j)
                candidate_state, step_cost = eval_pair_step(current_state, pair, all_label_counts=all_label_counts)
                if step_cost < best_step_cost:
                    best_step_cost = step_cost
                    best_pair = pair
                    best_next_state = candidate_state

        if best_next_state is None or math.isinf(best_step_cost):
            # Explicit failure when memory limit or feasibility constraints reject all candidate pairs
            raise ValueError(
                f"Contraction path search failed: no valid candidate tensor pair satisfies memory or hardware constraints "
                f"among {len(current_state.active_tensors)} active tensors (memory_limit={current_state.parameters.memory_limit})."
            )

        current_state = best_next_state

    return current_state


class PIMPathCostOptimizer(PathOptimizer):
    """PathOptimizer subclass implementing opt_einsum's custom path optimizer interface."""

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        self.parameters = PIMCostParameters.from_config(config)

    def __call__(
        self,
        inputs: list[set[Any]],
        output: set[Any],
        size_dict: dict[Any, int],
        memory_limit: int | None = None,
        **kwargs: Any,
    ) -> list[tuple[int, int]]:
        actual_params = replace(self.parameters, memory_limit=memory_limit) if memory_limit is not None else self.parameters
        initial_state = PathSearchState(
            active_tensors=tuple(tuple(sorted(s, key=_sort_key)) for s in inputs),
            size_dict=size_dict,
            history=(),
            total_cost=0.0,
            parameters=actual_params,
            output_labels=tuple(sorted(output, key=_sort_key)),
        )
        final_state = _greedy_search_pure(initial_state)
        return list(final_state.history)


def pim_path_finder_functional(
    inputs: list[set[Any]],
    output: set[Any],
    size_dict: dict[str, int],
    memory_limit: int | None = None,
    **kwargs: Any,
) -> list[tuple[int, int]]:
    """Pure functional path optimizer compatible with opt_einsum custom path optimizer interface."""
    optimizer = PIMPathCostOptimizer(config=kwargs.get("config"))
    return optimizer(inputs, output, size_dict, memory_limit=memory_limit, **kwargs)
