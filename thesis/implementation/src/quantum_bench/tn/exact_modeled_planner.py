"""Deterministic exhaustive modeled UPMEM v2 contraction-path planner."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from quantum_bench.routing.generic_numeric_contract import classify_numeric
from quantum_bench.tn.network import TensorNetworkValue
from quantum_bench.tn.planners import PlannerIdentity, PlannerResult
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
from quantum_bench.tn.upmem_planner import PlannerInfeasibleError


EXACT_MODELED_ENGINE = "exact_modeled"
EXACT_MODELED_SELECTION_SCOPE = "exact_finite_search"
DEFAULT_MAX_INPUT_TENSORS = 6
HARD_MAX_INPUT_TENSORS = 6


@dataclass(frozen=True)
class _CompletePathCandidate:
    score: float
    canonical_path: tuple[tuple[int, int], ...]
    components: PathCostComponentsV2
    tasks: tuple[Any, ...]
    executions: dict[str, TaskNumericExecution]


class ExactModeledPlanner:
    """Exhaustive search planner over pairwise contraction paths for small networks."""

    def __init__(
        self,
        *,
        profile: UpmemPathCostProfileV2 | None = None,
        max_input_tensors: int = DEFAULT_MAX_INPUT_TENSORS,
        options: dict[str, Any] | None = None,
    ) -> None:
        if max_input_tensors <= 0:
            raise ValueError("max_input_tensors must be positive")
        if max_input_tensors > HARD_MAX_INPUT_TENSORS:
            raise ValueError(
                f"max_input_tensors cannot exceed {HARD_MAX_INPUT_TENSORS}, got {max_input_tensors}"
            )
        self.max_input_tensors = int(max_input_tensors)
        self.profile = profile or upmem_path_cost_profile_v2()
        base_options = {
            "engine": EXACT_MODELED_ENGINE,
            "algorithm": "exact_modeled",
            "objective_version": UPMEM_PATH_OBJECTIVE_V2,
            "selection_scope": EXACT_MODELED_SELECTION_SCOPE,
            "max_input_tensors": self.max_input_tensors,
            "weight_profile": self.profile.profile_id,
            "normalization": self.profile.normalization.normalization_id,
            "execution_policy": self.profile.policy.policy_id,
            "numeric_contract": "real_float32_or_split_real_imag_v2",
        }
        if options:
            base_options.update(options)
        self.identity = PlannerIdentity(
            planner_engine=EXACT_MODELED_ENGINE,
            planner_id=f"{EXACT_MODELED_ENGINE}.{self.profile.profile_id}",
            planner_kind="exact_modeled_exhaustive",
            optimize_mode="exact_modeled",
            objective=UPMEM_PATH_OBJECTIVE_V2,
            cost_basis=self.profile.policy.policy_id,
            target_estimate_key=self.profile.policy.policy_id,
            options=base_options,
        )

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> ExactModeledPlanner:
        max_input_tensors = int(
            config.get("max_input_tensors", DEFAULT_MAX_INPUT_TENSORS)
        )
        objective_version = str(
            config.get("objective_version", UPMEM_PATH_OBJECTIVE_V2)
        )
        if objective_version != UPMEM_PATH_OBJECTIVE_V2:
            raise ValueError(
                f"Unsupported exact_modeled objective version: {objective_version}"
            )
        policy_id = str(
            config.get("execution_policy", DEFAULT_UPMEM_PATH_COST_POLICY_V2)
        )
        normalization = str(
            config.get("normalization", DEFAULT_UPMEM_PATH_COST_NORMALIZATION_V2)
        )
        if normalization != DEFAULT_UPMEM_PATH_COST_NORMALIZATION_V2:
            raise ValueError(
                f"Unsupported exact_modeled normalization: {normalization}"
            )
        policy = upmem_path_cost_policy_v2(policy_id)
        profile = upmem_path_cost_profile_v2(
            str(config.get("weight_profile", "balanced_literature_informed")),
            policy=policy,
        )
        return cls(profile=profile, max_input_tensors=max_input_tensors, options=config)

    def plan(self, network: TensorNetworkValue) -> PlannerResult:
        if len(network.tensors) > self.max_input_tensors:
            raise PlannerInfeasibleError(
                f"exact_modeled planner input tensor cap exceeded: {len(network.tensors)} > {self.max_input_tensors}",
                rejection_reasons=("input_tensor_cap_exceeded",),
            )

        # Lazy import derive_binary_contraction_step to avoid circular imports.
        from quantum_bench.tn.task_graph import derive_binary_contraction_step

        numeric_complex_by_tensor: dict[str, bool] = {}
        for tensor in network.tensors:
            classification = classify_numeric(np.asarray(tensor.array))
            if classification.has_nonfinite:
                raise PlannerInfeasibleError(
                    "exact_modeled policy cannot model nonfinite tensor inputs",
                    rejection_reasons=("nonfinite_values_not_supported",),
                )
            numeric_complex_by_tensor[tensor.spec.id] = (
                classification.has_nonzero_imaginary
            )

        start = time.perf_counter()
        initial_active = [tensor.spec for tensor in network.tensors]
        initial_produced_by: dict[str, str | None] = {
            tensor.id: tensor.produced_by for tensor in initial_active
        }

        complete_candidates: list[_CompletePathCandidate] = []
        explored_complete_paths = 0

        def _search(
            active: list[Any],
            produced_by: dict[str, str | None],
            complex_by_tensor: dict[str, bool],
            path_pairs: list[tuple[int, int]],
            path_tasks: list[Any],
            path_components: list[PathCostComponentsV2],
            path_executions: dict[str, TaskNumericExecution],
        ) -> None:
            nonlocal explored_complete_paths
            if len(active) == 1:
                explored_complete_paths += 1
                total_comp = combine_path_cost_components_v2(
                    path_components, self.profile.policy
                )
                if total_comp.feasibility:
                    score = self.profile.score(total_comp)
                    if math.isfinite(score):
                        complete_candidates.append(
                            _CompletePathCandidate(
                                score=score,
                                canonical_path=tuple(path_pairs),
                                components=total_comp,
                                tasks=tuple(path_tasks),
                                executions=dict(path_executions),
                            )
                        )
                return

            step_index = len(path_pairs)
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
                    left_complex = complex_by_tensor[step.task.input_tensor_ids[0]]
                    right_complex = complex_by_tensor[step.task.input_tensor_ids[1]]
                    output_elements = int(
                        np.prod(step.task.output_shape, dtype=np.int64)
                    )
                    execution = task_numeric_execution(
                        left_complex, right_complex, output_elements
                    )
                    components = model_upmem_task_cost_v2(
                        step.task,
                        self.profile.policy,
                        numeric_execution=execution,
                    )
                    next_produced_by = dict(produced_by)
                    next_produced_by[step.output_tensor.id] = step.task.id
                    next_complex = dict(complex_by_tensor)
                    next_complex[step.output_tensor.id] = (
                        execution.representation == "split_real_imag"
                    )

                    _search(
                        list(step.next_active),
                        next_produced_by,
                        next_complex,
                        [*path_pairs, (i, j)],
                        [*path_tasks, step.task],
                        [*path_components, components],
                        {**path_executions, step.task.id: execution},
                    )

        _search(
            initial_active,
            initial_produced_by,
            numeric_complex_by_tensor,
            [],
            [],
            [],
            {},
        )

        if not complete_candidates:
            raise PlannerInfeasibleError(
                "exact_modeled planner found no feasible complete path",
                rejection_reasons=("no_feasible_complete_path",),
            )

        # Deterministic tie break: score then canonical path tuple
        complete_candidates.sort(
            key=lambda candidate: (candidate.score, candidate.canonical_path)
        )
        best = complete_candidates[0]
        planning_time_s = time.perf_counter() - start

        return PlannerResult(
            identity=self.identity,
            path=best.canonical_path,
            path_info_text=(
                f"exact exhaustive search under modeled objective {self.profile.profile_id} "
                f"(explored {explored_complete_paths} complete paths, {len(complete_candidates)} feasible, "
                f"modeled-only, not hardware-optimal)"
            ),
            largest_intermediate=max(
                (
                    int(np.prod(task.output_shape, dtype=np.int64))
                    for task in best.tasks
                ),
                default=0,
            ),
            naive_flops=None,
            optimized_flops=float(best.components.estimated_flops),
            planning_time_s=planning_time_s,
            metadata={
                "explored_complete_paths": int(explored_complete_paths),
                "feasible_complete_paths": int(len(complete_candidates)),
                "selection_scope": EXACT_MODELED_SELECTION_SCOPE,
                "selection_claim": "exact_optimum_under_modeled_objective_only_not_hardware_optimal",
                "max_input_tensors": self.max_input_tensors,
                "planner_cost_model": UPMEM_PATH_OBJECTIVE_V2,
                "weight_profile": self.profile.profile_id,
                "normalization": self.profile.normalization.to_json_dict(),
                "execution_policy": self.profile.policy.to_json_dict(),
                "numeric_contract": "real_float32_or_split_real_imag_v2",
                "components": best.components.to_json_dict(),
                "normalized_components": self.profile.normalize(best.components),
                "modeled_score": float(best.score),
                "task_numeric_executions": {
                    task_id: exec.to_json_dict()
                    for task_id, exec in best.executions.items()
                },
                "execution_plan_executed": False,
            },
        )


__all__ = [
    "DEFAULT_MAX_INPUT_TENSORS",
    "EXACT_MODELED_ENGINE",
    "EXACT_MODELED_SELECTION_SCOPE",
    "HARD_MAX_INPUT_TENSORS",
    "ExactModeledPlanner",
]
