from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
import hashlib
import json
import random
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
    planner_config: dict[str, Any] = field(default_factory=dict)
    planner_config_hash: str = ""

    def __post_init__(self) -> None:
        # Keep the old options surface while making the effective configuration explicit.
        resolved = dict(self.planner_config or self.options)
        config_hash = canonical_planner_config_hash(resolved)
        options = dict(self.options)
        options["planner_config"] = resolved
        options["planner_config_hash"] = config_hash
        object.__setattr__(self, "planner_config", resolved)
        object.__setattr__(self, "planner_config_hash", config_hash)
        object.__setattr__(self, "options", options)


@dataclass(frozen=True)
class PlannerResult:
    identity: PlannerIdentity
    path: tuple[tuple[int, ...], ...]
    path_info_text: str
    largest_intermediate: int | None
    naive_flops: float | None
    optimized_flops: float | None
    planning_time_s: float
    metadata: dict[str, Any] = field(default_factory=dict)


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
        base_options["opt_einsum_version"] = str(getattr(oe, "__version__", "unknown"))
        if optimize.startswith("random-greedy"):
            base_options.update(
                {
                    "random_seed_policy": "opt_einsum_random_optimizer_run_index_v1",
                    "random_seed_version": "opt_einsum.path_random.RandomOptimizer",
                }
            )
        self.identity = PlannerIdentity(
            planner_engine="opt_einsum",
            planner_id=f"opt_einsum.{optimize}",
            planner_kind="external_path_optimizer",
            optimize_mode=optimize,
            objective="opt_einsum_contract_path",
            cost_basis="opt_einsum_internal",
            target_estimate_key=None,
            options=base_options,
            planner_config=base_options,
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
            metadata={
                "planner_config_hash": self.identity.planner_config_hash,
                "opt_einsum_version": self.identity.planner_config["opt_einsum_version"],
                **(
                    {
                        "random_seed_policy": self.identity.planner_config["random_seed_policy"],
                        "random_seed_version": self.identity.planner_config["random_seed_version"],
                    }
                    if self.optimize.startswith("random-greedy")
                    else {}
                ),
            },
        )


class CotengraPlanner:
    """Pinned-public-API cotengra baseline adapter for planner comparisons."""

    SUPPORTED_OBJECTIVES = frozenset({"flops", "size", "write", "combo"})

    def __init__(
        self,
        *,
        objective: str = "flops",
        methods: str = "greedy",
        max_repeats: int = 1,
        options: dict[str, Any] | None = None,
        seed: int | None = None,
    ) -> None:
        if objective not in self.SUPPORTED_OBJECTIVES:
            raise ValueError(f"Unsupported cotengra planner objective: {objective}")
        self.objective = objective
        self.methods = methods
        self.max_repeats = max(1, int(max_repeats))
        option_seed = options.get("seed", 0) if options else 0
        self.seed = int(option_seed if seed is None else seed)
        base_options = {
            "engine": "cotengra",
            "objective": objective,
            "methods": methods,
            "max_repeats": self.max_repeats,
            "optlib": "random",
            "seed": self.seed,
            "parallel": False,
            "progbar": False,
            "on_trial_error": "raise",
        }
        if options:
            base_options.update(options)
        # These settings are part of this adapter's reproducibility contract.
        base_options.update(
            {
                "methods": self.methods,
                "max_repeats": self.max_repeats,
                "optlib": "random",
                "seed": self.seed,
                "parallel": False,
                "progbar": False,
                "on_trial_error": "raise",
            }
        )
        self.identity = PlannerIdentity(
            planner_engine="cotengra",
            planner_id=f"cotengra.{objective}",
            planner_kind="external_contraction_tree",
            optimize_mode=methods,
            objective=f"cotengra_{objective}",
            cost_basis="cotengra_contraction_tree",
            target_estimate_key=None,
            options=base_options,
            planner_config=base_options,
        )

    def plan(self, network: TensorNetworkValue) -> PlannerResult:
        try:
            import cotengra as ctg
        except ImportError as exc:  # pragma: no cover - dependency contract
            raise RuntimeError("cotengra planner requested but cotengra is not installed") from exc

        inputs = [tuple(tensor.spec.labels) for tensor in network.tensors]
        size_dict: dict[int, int] = {}
        for tensor in network.tensors:
            for label, size in zip(tensor.spec.labels, tensor.spec.shape):
                existing = size_dict.setdefault(int(label), int(size))
                if existing != int(size):
                    raise ValueError(f"Inconsistent dimension for tensor-network label {label}")

        start = time.perf_counter()
        random_state = random.getstate()
        random.seed(self.seed)
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Couldn't find `optuna`, `cmaes`, or `nevergrad`.*")
                optimizer = ctg.HyperOptimizer(
                    methods=self.methods,
                    minimize=self.objective,
                    max_repeats=self.max_repeats,
                    optlib="random",
                    seed=self.seed,
                    parallel=False,
                    progbar=False,
                    on_trial_error="raise",
                )
                path = optimizer(inputs, tuple(network.spec.output_labels), size_dict)
        finally:
            random.setstate(random_state)
        planning_time_s = time.perf_counter() - start
        tree = getattr(optimizer, "tree", None)
        tree_path = tree.get_path() if tree is not None and hasattr(tree, "get_path") else path
        normalized_path = tuple(tuple(int(item) for item in step) for step in tree_path)
        if any(len(step) != 2 for step in normalized_path):
            raise ValueError("cotengra adapter produced a non-pairwise contraction path")
        return PlannerResult(
            identity=self.identity,
            path=normalized_path,
            path_info_text=f"cotengra {self.objective} tree ({self.methods})",
            largest_intermediate=_safe_int(_tree_number(tree, "max_size")),
            naive_flops=None,
            optimized_flops=_safe_float(_tree_number(tree, "total_flops")),
            planning_time_s=planning_time_s,
            metadata={
                "planner_adapter": "cotengra_hyperoptimizer_v1",
                "cotengra_objective": self.objective,
                "cotengra_methods": self.methods,
                "cotengra_max_repeats": self.max_repeats,
                "cotengra_optlib": "random",
                "cotengra_seed": self.seed,
                "cotengra_parallel": False,
                "planner_config_hash": self.identity.planner_config_hash,
            },
        )


def planner_from_config(config: dict[str, Any] | None) -> PathPlanner:
    config = config or {}
    engine = str(config.get("engine", "opt_einsum"))
    if engine == "opt_einsum":
        optimize = str(config.get("optimize", "greedy"))
        return OptEinsumPlanner(optimize=optimize, options=dict(config))
    if engine == "cotengra":
        return CotengraPlanner(
            objective=str(config.get("objective", config.get("minimize", "flops"))),
            methods=str(config.get("methods", "greedy")),
            max_repeats=int(config.get("max_repeats", 1)),
            options=dict(config),
            seed=int(config.get("seed", 0)),
        )
    if engine == "custom_upmem":
        objective_version = str(config.get("objective_version", "upmem_path_cost_v1"))
        if objective_version == "upmem_path_cost_v2":
            from quantum_bench.tn.upmem_planner import UpmemAwareProjectedPrefixPlanner

            return UpmemAwareProjectedPrefixPlanner.from_config(dict(config))
        from quantum_bench.tn.upmem_planner import UpmemAwareGreedyPlanner

        return UpmemAwareGreedyPlanner.from_config(dict(config))
    if engine == "exact_modeled":
        from quantum_bench.tn.exact_modeled_planner import ExactModeledPlanner

        return ExactModeledPlanner.from_config(dict(config))
    raise ValueError(f"Unsupported planner engine: {engine}")


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


def _tree_number(tree: object | None, name: str) -> object | None:
    if tree is None:
        return None
    value = getattr(tree, name, None)
    return value() if callable(value) else value


def canonical_planner_config_hash(config: dict[str, Any]) -> str:
    """Return the stable identity hash for a resolved planner configuration."""

    encoded = json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=_json_default)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _json_default(value: object) -> str:
    return repr(value)
