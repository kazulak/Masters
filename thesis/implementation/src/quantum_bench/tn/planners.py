from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Protocol

import opt_einsum as oe

from quantum_bench.tn.network import TensorNetworkValue, interleaved_einsum_args


@dataclass(frozen=True)
class PlannerIdentity:
    planner_engine: str
    planner_id: str
    planner_kind: str
    optimize_mode: str
    objective: str
    cost_basis: str
    target_estimate_key: str | None
    options: dict[str, Any]


@dataclass(frozen=True)
class PlannerResult:
    identity: PlannerIdentity
    path: tuple[tuple[int, ...], ...]
    path_info_text: str
    largest_intermediate: int | None
    naive_flops: float | None
    optimized_flops: float | None
    planning_time_s: float


class PathPlanner(Protocol):
    identity: PlannerIdentity

    def plan(self, network: TensorNetworkValue) -> PlannerResult:
        ...


class OptEinsumPlanner:
    def __init__(self, optimize: str = "greedy", options: dict[str, Any] | None = None) -> None:
        self.optimize = optimize
        base_options = {"engine": "opt_einsum", "optimize": optimize}
        if options:
            base_options.update(options)
        self.identity = PlannerIdentity(
            planner_engine="opt_einsum",
            planner_id=f"opt_einsum.{optimize}",
            planner_kind="external_path_optimizer",
            optimize_mode=optimize,
            objective="opt_einsum_contract_path",
            cost_basis="opt_einsum_internal",
            target_estimate_key=None,
            options=base_options,
        )

    def plan(self, network: TensorNetworkValue) -> PlannerResult:
        start = time.perf_counter()
        path, path_info = oe.contract_path(*interleaved_einsum_args(network), optimize=self.optimize)
        planning_time_s = time.perf_counter() - start
        return PlannerResult(
            identity=self.identity,
            path=tuple(tuple(int(item) for item in step) for step in path),
            path_info_text=str(path_info),
            largest_intermediate=_safe_int(getattr(path_info, "largest_intermediate", None)),
            naive_flops=_safe_float(getattr(path_info, "naive_cost", None)),
            optimized_flops=_safe_float(getattr(path_info, "opt_cost", None)),
            planning_time_s=planning_time_s,
        )


def planner_from_config(config: dict[str, Any] | None) -> PathPlanner:
    config = config or {}
    engine = str(config.get("engine", "opt_einsum"))
    optimize = str(config.get("optimize", "greedy"))
    if engine != "opt_einsum":
        raise ValueError(f"Unsupported planner engine: {engine}")
    return OptEinsumPlanner(optimize=optimize, options=dict(config))


def _safe_int(value: object) -> int | None:
    try:
        return None if value is None else int(value)
    except (TypeError, ValueError):
        return None


def _safe_float(value: object) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None
