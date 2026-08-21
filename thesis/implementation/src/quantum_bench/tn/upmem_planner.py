"""Deterministic modeled UPMEM-aware contraction-path planning.

This module deliberately models the current bounded generic single-DPU route.
It selects a path from structural cost estimates only; it neither executes a
DPU program nor predicts hardware runtime.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from quantum_bench.core.records import TensorNetworkSpec, TensorSpec
from quantum_bench.tn.network import TensorNetworkValue
from quantum_bench.tn.planners import PlannerIdentity, PlannerResult
from quantum_bench.tn.upmem_path_cost import (
    DEFAULT_UPMEM_PATH_COST_NORMALIZATION_ID,
    DEFAULT_UPMEM_PATH_COST_POLICY_ID,
    PathCostComponents,
    UpmemPathCostProfile,
    generic_float32_network_rejection_reasons,
    model_upmem_network_path_cost,
    model_upmem_task_cost,
    upmem_path_cost_policy,
    upmem_path_cost_profile,
)
from quantum_bench.tn.upmem_path_cost_v2 import (
    DEFAULT_UPMEM_PATH_COST_NORMALIZATION_V2,
    DEFAULT_UPMEM_PATH_COST_POLICY_V2,
    UPMEM_PATH_OBJECTIVE_V2,
    PathCostComponentsV2,
    TaskNumericExecution,
    UpmemPathCostProfileV2,
    combine_path_cost_components_v2,
    model_upmem_task_cost_v2,
    task_numeric_execution,
    upmem_path_cost_policy_v2,
    upmem_path_cost_profile_v2,
)
from quantum_bench.routing.generic_numeric_contract import classify_numeric


UPMEM_PATH_OBJECTIVE_VERSION = "upmem_path_cost_v1"
UPMEM_PATH_ENGINE = "custom_upmem"
UPMEM_PATH_SELECTION_SCOPE_V1 = "local_step"
UPMEM_PATH_SELECTION_SCOPE_V2 = "projected_prefix"


class PlannerInfeasibleError(ValueError):
    """No complete path satisfies the selected modeled policy."""

    def __init__(self, message: str, *, rejection_reasons: tuple[str, ...], step_index: int | None = None) -> None:
        super().__init__(message)
        self.rejection_reasons = rejection_reasons
        self.step_index = step_index


@dataclass(frozen=True)
class _Candidate:
    pair: tuple[int, int]
    task: Any
    output_tensor: TensorSpec
    next_active: tuple[TensorSpec, ...]
    components: PathCostComponents
    score: float


class UpmemAwareGreedyPlanner:
    """Select a complete dynamic path using one fixed generic UPMEM policy."""

    def __init__(self, *, profile: UpmemPathCostProfile, options: dict[str, Any] | None = None) -> None:
        self.profile = profile
        base_options = {
            "engine": UPMEM_PATH_ENGINE,
            "algorithm": "greedy",
            "objective_version": UPMEM_PATH_OBJECTIVE_VERSION,
            "weight_profile": profile.profile_id,
            "normalization": profile.normalization.normalization_id,
            "execution_policy": profile.policy.policy_id,
            "numeric_contract": "float32_real_generic_model",
        }
        if options:
            base_options.update(options)
        self.identity = PlannerIdentity(
            planner_engine=UPMEM_PATH_ENGINE,
            planner_id=f"{UPMEM_PATH_ENGINE}.greedy.{profile.profile_id}",
            planner_kind="native_target_greedy",
            optimize_mode="greedy",
            objective=UPMEM_PATH_OBJECTIVE_VERSION,
            cost_basis=profile.policy.policy_id,
            target_estimate_key=profile.policy.policy_id,
            options=base_options,
        )

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "UpmemAwareGreedyPlanner":
        algorithm = str(config.get("algorithm", "greedy"))
        if algorithm != "greedy":
            raise ValueError(f"Unsupported custom_upmem planner algorithm: {algorithm}")
        objective_version = str(config.get("objective_version", UPMEM_PATH_OBJECTIVE_VERSION))
        if objective_version != UPMEM_PATH_OBJECTIVE_VERSION:
            raise ValueError(f"Unsupported custom_upmem objective version: {objective_version}")
        policy_id = str(config.get("execution_policy", DEFAULT_UPMEM_PATH_COST_POLICY_ID))
        normalization = str(config.get("normalization", DEFAULT_UPMEM_PATH_COST_NORMALIZATION_ID))
        if normalization != DEFAULT_UPMEM_PATH_COST_NORMALIZATION_ID:
            raise ValueError(f"Unsupported custom_upmem normalization: {normalization}")
        policy = upmem_path_cost_policy(policy_id)
        profile = upmem_path_cost_profile(str(config.get("weight_profile", "balanced_literature_informed")), policy=policy)
        return cls(profile=profile, options=config)

    def plan(self, network: TensorNetworkValue) -> PlannerResult:
        complex_reasons = generic_float32_network_rejection_reasons(network)
        if complex_reasons:
            raise PlannerInfeasibleError(
                "custom_upmem generic float32 policy cannot model complex tensor inputs",
                rejection_reasons=complex_reasons,
            )

        # Import lazily: task_graph imports PlannerResult/PathPlanner.
        from quantum_bench.tn.task_graph import derive_binary_contraction_step

        start = time.perf_counter()
        active = [tensor.spec for tensor in network.tensors]
        produced_by: dict[str, str | None] = {tensor.id: tensor.produced_by for tensor in active}
        selected_tasks = []
        path: list[tuple[int, int]] = []
        step_trace: list[dict[str, Any]] = []

        for step_index in range(max(0, len(active) - 1)):
            candidates: list[_Candidate] = []
            rejections: list[str] = []
            for i in range(len(active)):
                for j in range(i + 1, len(active)):
                    step = derive_binary_contraction_step(
                        active,
                        (i, j),
                        network.spec.output_labels,
                        produced_by=produced_by,
                        task_id=f"task_{step_index}",
                        output_id=f"result_{step_index}",
                    )
                    components = model_upmem_task_cost(step.task, self.profile.policy)
                    score = self.profile.score(components)
                    entry = {
                        "pair": [i, j],
                        "feasible": components.feasibility,
                        "score": score if np.isfinite(score) else None,
                        "components": components.to_json_dict(),
                    }
                    if components.feasibility:
                        candidates.append(
                            _Candidate(
                                pair=(i, j),
                                task=step.task,
                                output_tensor=step.output_tensor,
                                next_active=step.next_active,
                                components=components,
                                score=score,
                            )
                        )
                    else:
                        rejections.extend(components.rejection_reasons)
                    step_trace.append({"step_index": step_index, **entry})

            if not candidates:
                reasons = tuple(dict.fromkeys(rejections)) or ("no_feasible_pair",)
                raise PlannerInfeasibleError(
                    f"custom_upmem planner found no feasible pair at step {step_index}: {', '.join(reasons)}",
                    rejection_reasons=reasons,
                    step_index=step_index,
                )
            selected = min(
                candidates,
                key=lambda item: (
                    item.score,
                    item.components.tiles,
                    int(np.prod(item.task.output_shape, dtype=np.int64)),
                    item.task.estimated_flops,
                    item.pair,
                ),
            )
            path.append(selected.pair)
            selected_tasks.append(selected.task)
            produced_by[selected.output_tensor.id] = selected.task.id
            active = list(selected.next_active)

        if len(active) != 1:
            raise PlannerInfeasibleError(
                f"custom_upmem planner ended with {len(active)} active tensors",
                rejection_reasons=("incomplete_path",),
            )

        total = model_upmem_network_path_cost(network, selected_tasks, self.profile.policy)
        if not total.feasibility:
            raise PlannerInfeasibleError(
                "custom_upmem planner selected a structurally infeasible path",
                rejection_reasons=total.rejection_reasons,
            )
        planning_time_s = time.perf_counter() - start
        return PlannerResult(
            identity=self.identity,
            path=tuple(path),
            path_info_text=(
                f"deterministic {UPMEM_PATH_ENGINE} greedy path using {self.profile.profile_id} "
                f"under {self.profile.policy.policy_id}"
            ),
            largest_intermediate=max((int(np.prod(task.output_shape, dtype=np.int64)) for task in selected_tasks), default=0),
            naive_flops=None,
            optimized_flops=float(total.flops),
            planning_time_s=planning_time_s,
            metadata={
                "planner_cost_model": UPMEM_PATH_OBJECTIVE_VERSION,
                "weight_profile": self.profile.profile_id,
                "normalization": self.profile.normalization.to_json_dict(),
                "execution_policy": self.profile.policy.to_json_dict(),
                "numeric_contract": "float32_real_generic_model",
                "selection_scope": UPMEM_PATH_SELECTION_SCOPE_V1,
                "components": total.to_json_dict(),
                "normalized_components": self.profile.normalize(total),
                "modeled_score": self.profile.score(total),
                "step_trace": step_trace,
                "execution_plan_executed": False,
            },
        )


@dataclass(frozen=True)
class _ProjectedCandidate:
    pair: tuple[int, int]
    task: Any
    output_tensor: TensorSpec
    next_active: tuple[TensorSpec, ...]
    numeric_execution: TaskNumericExecution
    components: PathCostComponentsV2
    local_score: float
    projected_components: PathCostComponentsV2
    projected_score: float
    tie_break: tuple[Any, ...]


class UpmemAwareProjectedPrefixPlanner:
    """Deterministic v2 greedy planner scored against its selected prefix.

    The planner is intentionally still greedy.  At each step it selects the
    candidate with the lowest modeled score of the already selected prefix plus
    that candidate; it does not claim a global complete-path optimum.
    """

    def __init__(self, *, profile: UpmemPathCostProfileV2, options: dict[str, Any] | None = None) -> None:
        self.profile = profile
        base_options = {
            "engine": UPMEM_PATH_ENGINE,
            "algorithm": "greedy",
            "objective_version": UPMEM_PATH_OBJECTIVE_V2,
            "selection_scope": UPMEM_PATH_SELECTION_SCOPE_V2,
            "weight_profile": profile.profile_id,
            "normalization": profile.normalization.normalization_id,
            "execution_policy": profile.policy.policy_id,
            "numeric_contract": "real_float32_or_split_real_imag_v2",
        }
        if options:
            base_options.update(options)
        self.identity = PlannerIdentity(
            planner_engine=UPMEM_PATH_ENGINE,
            planner_id=f"{UPMEM_PATH_ENGINE}.greedy.v2.{profile.profile_id}",
            planner_kind="native_target_projected_prefix_greedy",
            optimize_mode="greedy",
            objective=UPMEM_PATH_OBJECTIVE_V2,
            cost_basis=profile.policy.policy_id,
            target_estimate_key=profile.policy.policy_id,
            options=base_options,
        )

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "UpmemAwareProjectedPrefixPlanner":
        algorithm = str(config.get("algorithm", "greedy"))
        if algorithm != "greedy":
            raise ValueError(f"Unsupported custom_upmem v2 planner algorithm: {algorithm}")
        objective_version = str(config.get("objective_version", UPMEM_PATH_OBJECTIVE_V2))
        if objective_version != UPMEM_PATH_OBJECTIVE_V2:
            raise ValueError(f"Unsupported custom_upmem v2 objective version: {objective_version}")
        selection_scope = str(config.get("selection_scope", UPMEM_PATH_SELECTION_SCOPE_V2))
        if selection_scope != UPMEM_PATH_SELECTION_SCOPE_V2:
            raise ValueError(f"Unsupported custom_upmem v2 selection scope: {selection_scope}")
        policy_id = str(config.get("execution_policy", DEFAULT_UPMEM_PATH_COST_POLICY_V2))
        normalization = str(config.get("normalization", DEFAULT_UPMEM_PATH_COST_NORMALIZATION_V2))
        if normalization != DEFAULT_UPMEM_PATH_COST_NORMALIZATION_V2:
            raise ValueError(f"Unsupported custom_upmem v2 normalization: {normalization}")
        policy = upmem_path_cost_policy_v2(policy_id)
        profile = upmem_path_cost_profile_v2(
            str(config.get("weight_profile", "balanced_literature_informed")),
            policy=policy,
        )
        return cls(profile=profile, options=config)

    def plan(self, network: TensorNetworkValue) -> PlannerResult:
        numeric_complex_by_tensor: dict[str, bool] = {}
        for tensor in network.tensors:
            classification = classify_numeric(np.asarray(tensor.array))
            if classification.has_nonfinite:
                raise PlannerInfeasibleError(
                    "custom_upmem v2 policy cannot model nonfinite tensor inputs",
                    rejection_reasons=("nonfinite_values_not_supported",),
                )
            numeric_complex_by_tensor[tensor.spec.id] = classification.has_nonzero_imaginary

        return plan_upmem_projected_prefix(
            network.spec,
            numeric_complex_by_tensor,
            profile=self.profile,
            identity=self.identity,
        )


def plan_upmem_projected_prefix(
    network: TensorNetworkSpec,
    complex_by_tensor: dict[str, bool],
    *,
    profile: UpmemPathCostProfileV2 | None = None,
    request_config: dict[str, Any] | None = None,
    identity: PlannerIdentity | None = None,
) -> PlannerResult:
    """Build a v2 projected-prefix path from tensor metadata and flags.

    ``complex_by_tensor`` is an explicit logical execution annotation.  This
    function never reads tensor values, allocates fabricated arrays, or
    performs execution validation.  The legacy value-based planner derives the
    same annotation from its arrays and delegates here.
    """

    active_profile = profile or upmem_path_cost_profile_v2()
    expected_ids = {tensor.id for tensor in network.tensors}
    provided_ids = set(complex_by_tensor)
    if provided_ids != expected_ids:
        missing = sorted(expected_ids - provided_ids)
        extra = sorted(provided_ids - expected_ids)
        raise ValueError(f"complex tensor flags do not match network: missing={missing} extra={extra}")
    if any(not isinstance(value, bool) for value in complex_by_tensor.values()):
        raise TypeError("complex tensor flags must be bool values")

    planner_identity = identity or _projected_prefix_identity(active_profile, request_config or {})
    start = time.perf_counter()

    # Import lazily to preserve the existing task-graph/planner import order.
    from quantum_bench.tn.task_graph import derive_binary_contraction_step

    active = [tensor for tensor in network.tensors]
    produced_by: dict[str, str | None] = {tensor.id: tensor.produced_by for tensor in active}
    numeric_flags = dict(complex_by_tensor)
    selected_tasks: list[Any] = []
    selected_components: list[PathCostComponentsV2] = []
    selected_executions: dict[str, TaskNumericExecution] = {}
    path: list[tuple[int, int]] = []
    step_trace: list[dict[str, Any]] = []

    for step_index in range(max(0, len(active) - 1)):
        candidates: list[_ProjectedCandidate] = []
        rejections: list[str] = []
        active_tensor_ids = [tensor.id for tensor in active]
        for i in range(len(active)):
            for j in range(i + 1, len(active)):
                step = derive_binary_contraction_step(
                    active,
                    (i, j),
                    network.output_labels,
                    produced_by=produced_by,
                    task_id=f"task_{step_index}",
                    output_id=f"result_{step_index}",
                )
                execution = task_numeric_execution(
                    numeric_flags[step.task.input_tensor_ids[0]],
                    numeric_flags[step.task.input_tensor_ids[1]],
                    int(np.prod(step.task.output_shape, dtype=np.int64)),
                )
                components = model_upmem_task_cost_v2(
                    step.task,
                    active_profile.policy,
                    numeric_execution=execution,
                )
                local_score = active_profile.score(components)
                projected_components = combine_path_cost_components_v2(
                    [*selected_components, components],
                    active_profile.policy,
                )
                projected_score = active_profile.score(projected_components)
                output_elements = int(np.prod(step.task.output_shape, dtype=np.int64))
                tie_break = (
                    projected_score,
                    components.tile_iterations,
                    output_elements,
                    step.task.estimated_flops,
                    (i, j),
                )
                candidate = _ProjectedCandidate(
                    pair=(i, j),
                    task=step.task,
                    output_tensor=step.output_tensor,
                    next_active=step.next_active,
                    numeric_execution=execution,
                    components=components,
                    local_score=local_score,
                    projected_components=projected_components,
                    projected_score=projected_score,
                    tie_break=tie_break,
                )
                if not components.feasibility:
                    rejections.extend(components.rejection_reasons)
                    step_trace.append(
                        _projected_trace_entry(
                            step_index, active_tensor_ids, candidate, None, False
                        )
                    )
                    continue
                candidates.append(candidate)

        if not candidates:
            reasons = tuple(dict.fromkeys(rejections)) or ("no_feasible_pair",)
            raise PlannerInfeasibleError(
                f"custom_upmem v2 planner found no feasible pair at step {step_index}: {', '.join(reasons)}",
                rejection_reasons=reasons,
                step_index=step_index,
            )

        ordered = sorted(candidates, key=lambda item: item.tie_break)
        selected = ordered[0]
        for candidate_rank, candidate in enumerate(ordered, start=1):
            step_trace.append(
                _projected_trace_entry(
                    step_index, active_tensor_ids, candidate, candidate_rank, candidate is selected
                )
            )

        path.append(selected.pair)
        selected_tasks.append(selected.task)
        selected_components.append(selected.components)
        selected_executions[selected.task.id] = selected.numeric_execution
        produced_by[selected.output_tensor.id] = selected.task.id
        numeric_flags[selected.output_tensor.id] = (
            selected.numeric_execution.representation == "split_real_imag"
        )
        active = list(selected.next_active)

    if len(active) != 1:
        raise PlannerInfeasibleError(
            f"custom_upmem v2 planner ended with {len(active)} active tensors",
            rejection_reasons=("incomplete_path",),
        )
    total = combine_path_cost_components_v2(selected_components, active_profile.policy)
    if not total.feasibility:
        raise PlannerInfeasibleError(
            "custom_upmem v2 planner selected an infeasible path",
            rejection_reasons=total.rejection_reasons,
        )
    modeled_score = active_profile.score(total)
    return PlannerResult(
        identity=planner_identity,
        path=tuple(path),
        path_info_text=(
            f"deterministic {UPMEM_PATH_ENGINE} projected-prefix greedy path using "
            f"{active_profile.profile_id} under {active_profile.policy.policy_id}"
        ),
        largest_intermediate=max(
            (int(np.prod(task.output_shape, dtype=np.int64)) for task in selected_tasks),
            default=0,
        ),
        naive_flops=None,
        optimized_flops=float(total.estimated_flops),
        planning_time_s=time.perf_counter() - start,
        metadata={
            "planner_cost_model": UPMEM_PATH_OBJECTIVE_V2,
            "selection_scope": UPMEM_PATH_SELECTION_SCOPE_V2,
            "selection_claim": "greedy_projected_prefix_not_global_path_optimum",
            "weight_profile": active_profile.profile_id,
            "normalization": active_profile.normalization.to_json_dict(),
            "execution_policy": active_profile.policy.to_json_dict(),
            "numeric_contract": "real_float32_or_split_real_imag_v2",
            "numeric_flags": {key: bool(value) for key, value in sorted(complex_by_tensor.items())},
            "components": total.to_json_dict(),
            "normalized_components": active_profile.normalize(total),
            "final_path_score": modeled_score,
            "modeled_score": modeled_score,
            "step_trace": step_trace,
            "task_numeric_executions": {
                task_id: execution.to_json_dict()
                for task_id, execution in selected_executions.items()
            },
            "execution_plan_executed": False,
        },
    )


def _projected_prefix_identity(
    profile: UpmemPathCostProfileV2,
    request_config: dict[str, Any],
) -> PlannerIdentity:
    config = dict(request_config)
    config.setdefault("engine", UPMEM_PATH_ENGINE)
    config.setdefault("algorithm", "greedy")
    config.setdefault("objective_version", UPMEM_PATH_OBJECTIVE_V2)
    config.setdefault("selection_scope", UPMEM_PATH_SELECTION_SCOPE_V2)
    config.setdefault("weight_profile", profile.profile_id)
    config.setdefault("normalization", profile.normalization.normalization_id)
    config.setdefault("execution_policy", profile.policy.policy_id)
    config.setdefault("numeric_contract", "real_float32_or_split_real_imag_v2")
    return PlannerIdentity(
        planner_engine=UPMEM_PATH_ENGINE,
        planner_id=f"{UPMEM_PATH_ENGINE}.greedy.v2.{profile.profile_id}",
        planner_kind="native_target_projected_prefix_greedy",
        optimize_mode="greedy",
        objective=UPMEM_PATH_OBJECTIVE_V2,
        cost_basis=profile.policy.policy_id,
        target_estimate_key=profile.policy.policy_id,
        options=config,
        planner_config=config,
    )


def _projected_trace_entry(
    step_index: int,
    active_tensor_ids: list[str],
    candidate: _ProjectedCandidate,
    candidate_rank: int | None,
    selected: bool,
) -> dict[str, Any]:
    components = candidate.components
    return {
        "step_index": step_index,
        "active_tensor_ids": list(active_tensor_ids),
        "pair": list(candidate.pair),
        "left_tensor_id": candidate.task.input_tensor_ids[0],
        "right_tensor_id": candidate.task.input_tensor_ids[1],
        "output_tensor_id": candidate.task.output_tensor_id,
        "output_shape": list(candidate.task.output_shape),
        "numeric_execution": candidate.numeric_execution.to_json_dict(),
        "feasible": components.feasibility,
        "local_step_score": candidate.local_score if np.isfinite(candidate.local_score) else None,
        "projected_cumulative_score": (
            candidate.projected_score if np.isfinite(candidate.projected_score) else None
        ),
        "components": components.to_json_dict(),
        "projected_cumulative_components": candidate.projected_components.to_json_dict(),
        "candidate_rank": candidate_rank,
        "tie_break": _jsonable_tie_break(candidate.tie_break),
        "selected": selected,
    }


def _jsonable_tie_break(value: tuple[Any, ...]) -> list[Any]:
    result: list[Any] = []
    for item in value:
        if isinstance(item, tuple):
            result.append(list(item))
        elif isinstance(item, np.generic):
            result.append(item.item())
        else:
            result.append(item)
    return result
__all__ = [
    "PlannerInfeasibleError",
    "UPMEM_PATH_ENGINE",
    "UPMEM_PATH_OBJECTIVE_VERSION",
    "UPMEM_PATH_SELECTION_SCOPE_V1",
    "UPMEM_PATH_SELECTION_SCOPE_V2",
    "UpmemAwareGreedyPlanner",
    "UpmemAwareProjectedPrefixPlanner",
    "plan_upmem_projected_prefix",
]
