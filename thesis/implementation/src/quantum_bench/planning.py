"""Functional contraction-path adapters for the canonical TN route."""

from __future__ import annotations

import hashlib
import json
import random
import time

import opt_einsum as oe

from quantum_bench.model import TensorNetwork

__all__ = ["plan_opt_einsum", "plan_cotengra"]

_PROVENANCE_KEYS = (
    "planner_engine",
    "planner_id",
    "planner_kind",
    "optimize_mode",
    "objective",
    "cost_basis",
    "planner_config",
    "planner_config_hash",
    "path_info_text",
    "largest_intermediate",
    "naive_flops",
    "optimized_flops",
    "planning_time_s",
    "dependency_versions",
)


def plan_opt_einsum(
    network: TensorNetwork,
    *,
    optimize: str = "greedy",
) -> tuple[tuple[tuple[int, int], ...], dict[str, object]]:
    """Return an opt_einsum active-list path and its provenance."""

    tensor_count = _network_preflight(network)
    shapes = tuple(tuple(int(size) for size in tensor.shape) for tensor in network.tensors)
    dependency_version = str(getattr(oe, "__version__", "unknown"))
    config: dict[str, object] = {
        "engine": "opt_einsum",
        "optimize": str(optimize),
    }
    if tensor_count == 1:
        return (), _provenance(
            planner_engine="opt_einsum",
            planner_id=f"opt_einsum.{optimize}",
            planner_kind="external_path_optimizer",
            optimize_mode=str(optimize),
            objective="opt_einsum_contract_path",
            cost_basis="opt_einsum_internal",
            config=config,
            path_info_text="one tensor; no contraction required",
            largest_intermediate=None,
            naive_flops=None,
            optimized_flops=None,
            planning_time_s=0.0,
            dependency_versions={"opt_einsum": dependency_version},
        )
    expression = _expression(network)
    start = time.perf_counter()
    path, path_info = oe.contract_path(
        expression,
        *shapes,
        shapes=True,
        optimize=optimize,
    )
    planning_time_s = time.perf_counter() - start
    normalized_path = _validate_pairwise_path(path, len(shapes))
    provenance = _provenance(
        planner_engine="opt_einsum",
        planner_id=f"opt_einsum.{optimize}",
        planner_kind="external_path_optimizer",
        optimize_mode=str(optimize),
        objective="opt_einsum_contract_path",
        cost_basis="opt_einsum_internal",
        config=config,
        path_info_text=str(path_info),
        largest_intermediate=_safe_int(getattr(path_info, "largest_intermediate", None)),
        naive_flops=_safe_float(getattr(path_info, "naive_cost", None)),
        optimized_flops=_safe_float(getattr(path_info, "opt_cost", None)),
        planning_time_s=planning_time_s,
        dependency_versions={"opt_einsum": dependency_version},
    )
    return normalized_path, provenance


def plan_cotengra(
    network: TensorNetwork,
    *,
    objective: str = "flops",
    methods: str = "greedy",
    max_repeats: int = 1,
    seed: int = 0,
) -> tuple[tuple[tuple[int, int], ...], dict[str, object]]:
    """Return a cotengra active-list path and its provenance."""

    tensor_count = _network_preflight(network)
    try:
        import cotengra as ctg
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError("cotengra is required for the cotengra planner") from exc

    if int(max_repeats) < 1:
        raise ValueError("max_repeats must be at least one")
    max_repeats = int(max_repeats)
    seed = int(seed)
    inputs = [tuple(int(label) for label in tensor.labels) for tensor in network.tensors]
    size_dict = _size_dict(network)
    cotengra_version = str(getattr(ctg, "__version__", "unknown"))
    config: dict[str, object] = {
        "engine": "cotengra",
        "methods": str(methods),
        "objective": str(objective),
        "max_repeats": max_repeats,
        "seed": seed,
        "optlib": "random",
        "parallel": False,
        "progbar": False,
        "on_trial_error": "raise",
    }
    if tensor_count == 1:
        return (), _provenance(
            planner_engine="cotengra",
            planner_id=f"cotengra.{objective}",
            planner_kind="external_contraction_tree",
            optimize_mode=str(methods),
            objective=str(objective),
            cost_basis="cotengra_contraction_tree",
            config=config,
            path_info_text="one tensor; no contraction required",
            largest_intermediate=None,
            naive_flops=None,
            optimized_flops=None,
            planning_time_s=0.0,
            dependency_versions={"cotengra": cotengra_version},
        )
    start = time.perf_counter()
    random_state = random.getstate()
    random.seed(seed)
    try:
        optimizer = ctg.HyperOptimizer(
            methods=methods,
            minimize=objective,
            max_repeats=max_repeats,
            optlib="random",
            seed=seed,
            parallel=False,
            progbar=False,
            on_trial_error="raise",
        )
        path = optimizer(inputs, tuple(network.output_labels), size_dict)
    finally:
        random.setstate(random_state)
    planning_time_s = time.perf_counter() - start
    tree = getattr(optimizer, "tree", None)
    tree_path = tree.get_path() if tree is not None and hasattr(tree, "get_path") else path
    normalized_path = _validate_pairwise_path(tree_path, len(inputs))
    provenance = _provenance(
        planner_engine="cotengra",
        planner_id=f"cotengra.{objective}",
        planner_kind="external_contraction_tree",
        optimize_mode=str(methods),
        objective=str(objective),
        cost_basis="cotengra_contraction_tree",
        config=config,
        path_info_text=f"cotengra {objective} tree ({methods})",
        largest_intermediate=_safe_int(_tree_number(tree, "max_size")),
        naive_flops=None,
        optimized_flops=_safe_float(_tree_number(tree, "total_flops")),
        planning_time_s=planning_time_s,
        dependency_versions={"cotengra": cotengra_version},
    )
    return normalized_path, provenance


def _provenance(
    *,
    planner_engine: str,
    planner_id: str,
    planner_kind: str,
    optimize_mode: str,
    objective: str,
    cost_basis: str,
    config: dict[str, object],
    path_info_text: str,
    largest_intermediate: int | None,
    naive_flops: float | None,
    optimized_flops: float | None,
    planning_time_s: float,
    dependency_versions: dict[str, str],
) -> dict[str, object]:
    result: dict[str, object] = {
        "planner_engine": planner_engine,
        "planner_id": planner_id,
        "planner_kind": planner_kind,
        "optimize_mode": optimize_mode,
        "objective": objective,
        "cost_basis": cost_basis,
        "planner_config": config,
        "planner_config_hash": _config_hash(config),
        "path_info_text": path_info_text,
        "largest_intermediate": largest_intermediate,
        "naive_flops": naive_flops,
        "optimized_flops": optimized_flops,
        "planning_time_s": planning_time_s,
        "dependency_versions": dependency_versions,
    }
    if tuple(result) != _PROVENANCE_KEYS:
        raise AssertionError("planner provenance schema drift")
    json.dumps(result, sort_keys=True, separators=(",", ":"))
    return result


def _config_hash(config: dict[str, object]) -> str:
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _network_preflight(network: TensorNetwork) -> int:
    tensor_count = len(network.tensors)
    if tensor_count == 0:
        raise ValueError("cannot plan an empty tensor network")
    return tensor_count


def _expression(network: TensorNetwork) -> str:
    if not network.tensors:
        raise ValueError("cannot plan an empty tensor network")
    symbols = {label: oe.get_symbol(int(label)) for tensor in network.tensors for label in tensor.labels}
    operands = ["".join(symbols[label] for label in tensor.labels) for tensor in network.tensors]
    output = "".join(symbols[label] for label in network.output_labels)
    return ",".join(operands) + "->" + output


def _size_dict(network: TensorNetwork) -> dict[int, int]:
    size_dict: dict[int, int] = {}
    for tensor in network.tensors:
        if len(tensor.labels) != len(tensor.shape):
            raise ValueError(f"tensor {tensor.id} labels and shape have different ranks")
        for label, size in zip(tensor.labels, tensor.shape):
            size = int(size)
            if size <= 0:
                raise ValueError(f"tensor {tensor.id} has non-positive dimension {size}")
            previous = size_dict.setdefault(int(label), size)
            if previous != size:
                raise ValueError(f"inconsistent dimension for tensor-network label {label}")
    return size_dict


def _validate_pairwise_path(path: object, tensor_count: int) -> tuple[tuple[int, int], ...]:
    if path is None:
        raise ValueError("planner returned no path")
    normalized: list[tuple[int, int]] = []
    active_count = tensor_count
    for step_index, raw_step in enumerate(path):
        try:
            step = tuple(int(item) for item in raw_step)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"planner step {step_index} is not a pair: {raw_step!r}") from exc
        if len(step) != 2:
            raise ValueError(f"planner step {step_index} is not pairwise: {step}")
        left, right = step
        if left < 0 or right < 0 or left >= active_count or right >= active_count or left == right:
            raise ValueError(f"planner step {step_index} references invalid active operands: {step}")
        normalized.append((left, right))
        active_count -= 1
    expected = max(0, tensor_count - 1)
    if len(normalized) != expected or active_count != 1:
        raise ValueError(
            f"planner returned incomplete path: {len(normalized)} steps for {tensor_count} tensors"
        )
    return tuple(normalized)


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
