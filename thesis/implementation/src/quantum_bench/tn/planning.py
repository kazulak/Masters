"""Functional contraction-planner adapters.

The adapters in this module consume only :class:`TensorNetworkSpec` metadata.
They do not inspect or mutate tensor arrays.  Existing ``PlannerIdentity`` and
``PlannerResult`` records are reused so this slice can be adopted without
changing the legacy planner/reporting path.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

import opt_einsum as oe

from quantum_bench.core.records import TensorNetworkSpec
from quantum_bench.tn.planner_records import (
    PlannerIdentity,
    PlannerResult,
    canonical_planner_config_hash,
)
from quantum_bench.tn.upmem_path_cost_v2 import (
    DEFAULT_UPMEM_PATH_COST_NORMALIZATION_V2,
    DEFAULT_UPMEM_PATH_COST_POLICY_V2,
    PathCostProfileIdV2,
    UPMEM_PATH_OBJECTIVE_V2,
)
from quantum_bench.tn.upmem_planner import (
    UPMEM_PATH_SELECTION_SCOPE_V2,
    plan_upmem_projected_prefix,
)


class PlannerEngine(str, Enum):
    """Planner engines supported by the functional adapter boundary."""

    OPT_EINSUM = "opt_einsum"
    COTENGRA = "cotengra"
    CUSTOM_UPMEM = "custom_upmem"


DEFAULT_INPUT_REPRESENTATION = "split_real_imag"
SUPPORTED_INPUT_REPRESENTATIONS = ("real_float32", "split_real_imag")


@dataclass(frozen=True, slots=True)
class PlannerRequest:
    """Immutable planner configuration with no tensor-value dependency.

    ``input_representation`` is an explicit modeling assumption for the
    metadata-only custom UPMEM planner.  The conservative default models split
    real/imaginary execution.  Real-only experiments must request
    ``real_float32`` explicitly.  Storage dtype does not override the selected
    assumption, and the value-aware legacy planner remains separate.
    """

    engine: PlannerEngine = PlannerEngine.OPT_EINSUM
    algorithm: str = "greedy"
    optimize: str = "greedy"
    objective: str = "flops"
    methods: str = "greedy"
    max_repeats: int = 1
    seed: int = 0
    objective_version: str = UPMEM_PATH_OBJECTIVE_V2
    selection_scope: str = UPMEM_PATH_SELECTION_SCOPE_V2
    weight_profile: str = "balanced_literature_informed"
    normalization: str = DEFAULT_UPMEM_PATH_COST_NORMALIZATION_V2
    execution_policy: str = DEFAULT_UPMEM_PATH_COST_POLICY_V2
    input_representation: str = DEFAULT_INPUT_REPRESENTATION
    options: tuple[tuple[str, object], ...] = ()

    def __post_init__(self) -> None:
        try:
            engine = self.engine if isinstance(self.engine, PlannerEngine) else PlannerEngine(self.engine)
        except ValueError as exc:
            raise ValueError(f"Unsupported planner engine: {self.engine!r}") from exc
        object.__setattr__(self, "engine", engine)
        if int(self.max_repeats) < 1:
            raise ValueError("max_repeats must be at least one")
        object.__setattr__(self, "max_repeats", int(self.max_repeats))
        object.__setattr__(self, "seed", int(self.seed))
        if self.input_representation not in SUPPORTED_INPUT_REPRESENTATIONS:
            raise ValueError(
                "Unsupported input representation: "
                f"{self.input_representation!r}; expected one of "
                f"{SUPPORTED_INPUT_REPRESENTATIONS}"
            )
        if isinstance(self.options, Mapping):
            items = tuple(sorted(self.options.items(), key=lambda item: str(item[0])))
            object.__setattr__(self, "options", items)
        for key, value in self.options:
            if not isinstance(key, str) or not key:
                raise ValueError("planner option keys must be non-empty strings")
            if not isinstance(value, (str, int, float, bool, type(None))):
                raise ValueError("planner option values must be JSON scalars")
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError("planner option values must be finite")

    @classmethod
    def from_config(cls, config: Mapping[str, Any] | None) -> "PlannerRequest":
        """Build a request from the existing YAML-style planner mapping."""

        values = dict(config or {})
        return cls(
            engine=values.pop("engine", PlannerEngine.OPT_EINSUM),
            algorithm=str(values.pop("algorithm", "greedy")),
            optimize=str(values.pop("optimize", "greedy")),
            objective=str(values.pop("objective", values.pop("minimize", "flops"))),
            methods=str(values.pop("methods", "greedy")),
            max_repeats=int(values.pop("max_repeats", 1)),
            seed=int(values.pop("seed", 0)),
            objective_version=str(values.pop("objective_version", UPMEM_PATH_OBJECTIVE_V2)),
            selection_scope=str(values.pop("selection_scope", UPMEM_PATH_SELECTION_SCOPE_V2)),
            weight_profile=str(values.pop("weight_profile", "balanced_literature_informed")),
            normalization=str(values.pop("normalization", DEFAULT_UPMEM_PATH_COST_NORMALIZATION_V2)),
            execution_policy=str(values.pop("execution_policy", DEFAULT_UPMEM_PATH_COST_POLICY_V2)),
            input_representation=str(
                values.pop("input_representation", DEFAULT_INPUT_REPRESENTATION)
            ),
            options=tuple(sorted(values.items(), key=lambda item: str(item[0]))),
        )

    def config(self) -> dict[str, object]:
        """Return the resolved, JSON-friendly configuration used for identity."""

        config: dict[str, object] = {"engine": self.engine.value}
        if self.engine is PlannerEngine.OPT_EINSUM:
            config["optimize"] = self.optimize
        elif self.engine is PlannerEngine.COTENGRA:
            config.update(
                {
                    "methods": self.methods,
                    "objective": self.objective,
                    "max_repeats": self.max_repeats,
                    "seed": self.seed,
                }
            )
        else:
            config.update(
                {
                    "algorithm": self.algorithm,
                    "objective_version": self.objective_version,
                    "selection_scope": self.selection_scope,
                    "weight_profile": self.weight_profile,
                    "normalization": self.normalization,
                    "execution_policy": self.execution_policy,
                    "input_representation": self.input_representation,
                }
            )
        config.update(dict(self.options))
        return config


class PlannerUnsupportedError(RuntimeError):
    """Raised when an adapter cannot satisfy its metadata-only contract."""


def plan_opt_einsum(
    network: TensorNetworkSpec,
    request: PlannerRequest | Mapping[str, Any] | None = None,
) -> PlannerResult:
    """Plan a network using pinned public ``opt_einsum`` APIs and shapes only."""

    resolved = _request(request)
    _require_engine(resolved, PlannerEngine.OPT_EINSUM)
    _reject_unused_options(resolved)
    expression = _expression(network)
    shapes = tuple(tuple(int(size) for size in tensor.shape) for tensor in network.tensors)
    start = time.perf_counter()
    path, path_info = oe.contract_path(
        expression,
        *shapes,
        shapes=True,
        optimize=resolved.optimize,
    )
    planning_time_s = time.perf_counter() - start
    normalized_path = _validate_pairwise_path(path, len(shapes))
    config = _resolved_config(resolved, opt_einsum_version=str(getattr(oe, "__version__", "unknown")))
    identity = _identity(
        engine=PlannerEngine.OPT_EINSUM,
        planner_id=f"opt_einsum.{resolved.optimize}",
        planner_kind="external_path_optimizer",
        optimize_mode=resolved.optimize,
        objective="opt_einsum_contract_path",
        cost_basis="opt_einsum_internal",
        config=config,
    )
    return PlannerResult(
        identity=identity,
        path=normalized_path,
        path_info_text=str(path_info),
        largest_intermediate=_safe_int(getattr(path_info, "largest_intermediate", None)),
        naive_flops=_safe_float(getattr(path_info, "naive_cost", None)),
        optimized_flops=_safe_float(getattr(path_info, "opt_cost", None)),
        planning_time_s=planning_time_s,
        metadata={
            "planner_config_hash": identity.planner_config_hash,
            "opt_einsum_version": config["opt_einsum_version"],
            "input_source": "tensor_network_spec_labels_and_shapes",
        },
    )


def plan_cotengra(
    network: TensorNetworkSpec,
    request: PlannerRequest | Mapping[str, Any] | None = None,
) -> PlannerResult:
    """Plan a network using cotengra's public ``HyperOptimizer`` API."""

    resolved = _request(request)
    _require_engine(resolved, PlannerEngine.COTENGRA)
    _reject_unused_options(resolved)
    try:
        import cotengra as ctg
    except ImportError as exc:  # pragma: no cover - dependency contract
        raise PlannerUnsupportedError("cotengra is not installed") from exc

    inputs = [tuple(int(label) for label in tensor.labels) for tensor in network.tensors]
    size_dict = _size_dict(network)
    config = _resolved_config(
        resolved,
        cotengra_version=str(getattr(ctg, "__version__", "unknown")),
        optlib="random",
        parallel=False,
        progbar=False,
        on_trial_error="raise",
    )
    start = time.perf_counter()
    random_state = random.getstate()
    random.seed(resolved.seed)
    try:
        optimizer = ctg.HyperOptimizer(
            methods=resolved.methods,
            minimize=resolved.objective,
            max_repeats=resolved.max_repeats,
            optlib="random",
            seed=resolved.seed,
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
    identity = _identity(
        engine=PlannerEngine.COTENGRA,
        planner_id=f"cotengra.{resolved.objective}",
        planner_kind="external_contraction_tree",
        optimize_mode=resolved.methods,
        objective=f"cotengra_{resolved.objective}",
        cost_basis="cotengra_contraction_tree",
        config=config,
    )
    return PlannerResult(
        identity=identity,
        path=normalized_path,
        path_info_text=f"cotengra {resolved.objective} tree ({resolved.methods})",
        largest_intermediate=_safe_int(_tree_number(tree, "max_size")),
        naive_flops=None,
        optimized_flops=_safe_float(_tree_number(tree, "total_flops")),
        planning_time_s=planning_time_s,
        metadata={
            "planner_adapter": "cotengra_hyperoptimizer_v1",
            "planner_config_hash": identity.planner_config_hash,
            "input_source": "tensor_network_spec_labels_and_shapes",
        },
    )


def plan_upmem_greedy(
    network: TensorNetworkSpec,
    request: PlannerRequest | Mapping[str, Any] | None = None,
) -> PlannerResult:
    resolved = _request(request)
    _require_engine(resolved, PlannerEngine.CUSTOM_UPMEM)
    _reject_unused_options(resolved)
    if resolved.algorithm != "greedy":
        raise ValueError(f"Unsupported custom_upmem planner algorithm: {resolved.algorithm}")
    if resolved.objective_version != UPMEM_PATH_OBJECTIVE_V2:
        raise ValueError(f"Unsupported custom_upmem objective version: {resolved.objective_version}")
    if resolved.selection_scope != UPMEM_PATH_SELECTION_SCOPE_V2:
        raise ValueError(f"Unsupported custom_upmem selection scope: {resolved.selection_scope}")
    if resolved.normalization != DEFAULT_UPMEM_PATH_COST_NORMALIZATION_V2:
        raise ValueError(f"Unsupported custom_upmem normalization: {resolved.normalization}")

    from quantum_bench.tn.upmem_path_cost_v2 import (
        upmem_path_cost_policy_v2,
        upmem_path_cost_profile_v2,
    )

    policy = upmem_path_cost_policy_v2(resolved.execution_policy)
    profile = upmem_path_cost_profile_v2(
        _validate_weight_profile(resolved.weight_profile),
        policy=policy,
    )
    complex_by_tensor = {
        tensor.id: resolved.input_representation == "split_real_imag"
        for tensor in network.tensors
    }
    return plan_upmem_projected_prefix(
        network,
        complex_by_tensor,
        profile=profile,
        request_config=resolved.config(),
    )


def plan_contractions(
    network: TensorNetworkSpec,
    request: PlannerRequest | Mapping[str, Any] | None = None,
) -> PlannerResult:
    """Dispatch planning explicitly while keeping adapters as pure functions."""

    resolved = _request(request)
    match resolved.engine:
        case PlannerEngine.OPT_EINSUM:
            return plan_opt_einsum(network, resolved)
        case PlannerEngine.COTENGRA:
            return plan_cotengra(network, resolved)
        case PlannerEngine.CUSTOM_UPMEM:
            return plan_upmem_greedy(network, resolved)
        case _:
            raise ValueError(f"Unsupported planner engine: {resolved.engine!r}")


def _request(request: PlannerRequest | Mapping[str, Any] | None) -> PlannerRequest:
    if request is None:
        return PlannerRequest()
    if isinstance(request, PlannerRequest):
        return request
    if isinstance(request, Mapping):
        return PlannerRequest.from_config(request)
    raise TypeError("planner request must be PlannerRequest, mapping, or None")


def _validate_weight_profile(value: str) -> PathCostProfileIdV2:
    allowed = {
        "compute_oriented",
        "host_transfer_oriented",
        "local_movement_oriented",
        "wram_constrained",
        "synchronization_constrained",
        "balanced_literature_informed",
    }
    if value not in allowed:
        raise ValueError(f"unknown UPMEM v2 path-cost profile: {value}")
    return value  # type: ignore[return-value]


def _require_engine(request: PlannerRequest, expected: PlannerEngine) -> None:
    if request.engine is not expected:
        raise ValueError(f"{expected.value} adapter received {request.engine.value}")


def _reject_unused_options(request: PlannerRequest) -> None:
    if request.options:
        names = ", ".join(key for key, _ in request.options)
        raise ValueError(f"Unsupported {request.engine.value} planner option(s): {names}")


def _expression(network: TensorNetworkSpec) -> str:
    if not network.tensors:
        raise ValueError("cannot plan an empty tensor network")
    symbols = {label: oe.get_symbol(int(label)) for tensor in network.tensors for label in tensor.labels}
    operands = ["".join(symbols[label] for label in tensor.labels) for tensor in network.tensors]
    output = "".join(symbols[label] for label in network.output_labels)
    return ",".join(operands) + "->" + output


def _size_dict(network: TensorNetworkSpec) -> dict[int, int]:
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


def _validate_pairwise_path(path: object, tensor_count: int) -> tuple[tuple[int, ...], ...]:
    normalized: list[tuple[int, ...]] = []
    active_count = tensor_count
    for step_index, raw_step in enumerate(path if path is not None else ()):
        step = tuple(int(item) for item in raw_step)
        if len(step) != 2:
            raise ValueError(f"planner step {step_index} is not pairwise: {step}")
        left, right = step
        if left < 0 or right < 0 or left >= active_count or right >= active_count or left == right:
            raise ValueError(f"planner step {step_index} references invalid active operands: {step}")
        normalized.append(step)
        active_count -= 1
    expected = max(0, tensor_count - 1)
    if len(normalized) != expected or active_count != 1:
        raise ValueError(
            f"planner returned incomplete path: {len(normalized)} steps for {tensor_count} tensors"
        )
    return tuple(normalized)


def _resolved_config(request: PlannerRequest, **extra: object) -> dict[str, object]:
    config = request.config()
    config.update(extra)
    return config


def _identity(
    *,
    engine: PlannerEngine,
    planner_id: str,
    planner_kind: str,
    optimize_mode: str,
    objective: str,
    cost_basis: str,
    config: dict[str, object],
) -> PlannerIdentity:
    return PlannerIdentity(
        planner_engine=engine.value,
        planner_id=planner_id,
        planner_kind=planner_kind,
        optimize_mode=optimize_mode,
        objective=objective,
        cost_basis=cost_basis,
        target_estimate_key=None,
        options=dict(config),
        planner_config=config,
        planner_config_hash=canonical_planner_config_hash(config),
    )


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
