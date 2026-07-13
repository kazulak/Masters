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

from quantum_bench.core.records import TensorSpec
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


UPMEM_PATH_OBJECTIVE_VERSION = "upmem_path_cost_v1"
UPMEM_PATH_ENGINE = "custom_upmem"


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
                "components": total.to_json_dict(),
                "normalized_components": self.profile.normalize(total),
                "modeled_score": self.profile.score(total),
                "step_trace": step_trace,
                "execution_plan_executed": False,
            },
        )
__all__ = [
    "PlannerInfeasibleError",
    "UPMEM_PATH_ENGINE",
    "UPMEM_PATH_OBJECTIVE_VERSION",
    "UpmemAwareGreedyPlanner",
]
