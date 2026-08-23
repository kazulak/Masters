"""Small target-neutral model records for the canonical simulation route."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from copy import deepcopy
from dataclasses import dataclass
import math
from typing import Any, Literal, TypeAlias


SimulationQuery: TypeAlias = Literal["pre_measurement_statevector"]
_Scalar: TypeAlias = str | int | float | bool | None


@dataclass(frozen=True)
class CircuitOperation:
    gate: str
    wires: tuple[int, ...]
    params: tuple[float, ...] = ()


@dataclass(frozen=True)
class CircuitSpec:
    name: str
    n_qubits: int
    operations: tuple[CircuitOperation, ...]
    source: Mapping[str, Any]

    def __post_init__(self) -> None:
        frozen = _freeze(self.source)
        if not isinstance(frozen, _FrozenMapping):
            raise TypeError("CircuitSpec.source must be a mapping")
        object.__setattr__(self, "source", frozen)


@dataclass(frozen=True)
class TensorSpec:
    id: str
    labels: tuple[int, ...]
    shape: tuple[int, ...]
    structure: str
    dtype: str = "complex128"
    produced_by: str | None = None


@dataclass(frozen=True, slots=True)
class SimulationJob:
    circuit: CircuitSpec
    query: SimulationQuery = "pre_measurement_statevector"
    parameters: tuple[tuple[str, _Scalar], ...] = ()
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.query != "pre_measurement_statevector":
            raise ValueError(f"Unsupported simulation query: {self.query!r}")
        if not isinstance(self.parameters, tuple):
            raise ValueError("SimulationJob parameters must be a tuple")
        previous_key: str | None = None
        for item in self.parameters:
            if not isinstance(item, tuple) or len(item) != 2:
                raise ValueError(f"Invalid simulation parameter entry: {item!r}")
            key, value = item
            _validate_parameter(key, value)
            if previous_key is not None and key == previous_key:
                raise ValueError(f"Duplicate simulation parameter key: {key!r}")
            if previous_key is not None and key < previous_key:
                raise ValueError("SimulationJob parameters must be strictly key-sorted")
            previous_key = key
        if self.seed is not None and (isinstance(self.seed, bool) or not isinstance(self.seed, int)):
            raise ValueError("SimulationJob seed must be None or an integer")


def make_simulation_job(
    circuit: CircuitSpec,
    *,
    query: SimulationQuery = "pre_measurement_statevector",
    parameters: Iterable[tuple[str, _Scalar]] = (),
    seed: int | None = None,
) -> SimulationJob:
    """Validate and normalize simulation-job construction."""

    normalized: list[tuple[str, _Scalar]] = []
    seen: set[str] = set()
    for item in parameters:
        if not isinstance(item, tuple) or len(item) != 2:
            raise ValueError(f"Invalid simulation parameter entry: {item!r}")
        key, value = item
        _validate_parameter(key, value)
        if key in seen:
            raise ValueError(f"Duplicate simulation parameter key: {key!r}")
        seen.add(key)
        normalized.append((key, value))
    normalized.sort(key=lambda item: item[0])
    return SimulationJob(
        circuit=circuit,
        query=query,
        parameters=tuple(normalized),
        seed=seed,
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class TensorView:
    """A semantic tensor reference with optional fixed indices."""

    tensor_id: str
    labels: tuple[int, ...]
    shape: tuple[int, ...]
    slice_spec: tuple[tuple[int, int], ...] = ()


@dataclass(frozen=True, slots=True, kw_only=True)
class SliceSpec:
    """One bounded semantic slice of one contraction label."""

    node_id: str
    label: int


@dataclass(frozen=True, slots=True, kw_only=True)
class ContractNode:
    """One binary tensor contraction in the semantic graph."""

    node_id: str
    left: TensorView
    right: TensorView
    output: TensorSpec
    contracted_labels: tuple[int, ...]
    output_labels: tuple[int, ...]
    dependencies: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True, kw_only=True)
class ReduceNode:
    """Explicit sum reconstruction for sliced partial results."""

    node_id: str
    inputs: tuple[TensorView, ...]
    output: TensorSpec
    reduced_labels: tuple[int, ...] = ()
    dependencies: tuple[str, ...] = ()


GraphNode: TypeAlias = ContractNode | ReduceNode


@dataclass(frozen=True, slots=True, kw_only=True)
class ContractionDAG:
    """Planner-independent semantic contraction graph."""

    tensors: tuple[TensorSpec, ...]
    nodes: tuple[GraphNode, ...]
    output: TensorView


@dataclass(frozen=True, slots=True)
class TensorNetwork:
    """Target-neutral semantic network metadata, not an execution IR."""

    circuit: CircuitSpec
    tensors: tuple[TensorSpec, ...]
    output_labels: tuple[int, ...]
    einsum_expression: str


def _validate_parameter(key: object, value: object) -> None:
    if not isinstance(key, str) or not key:
        raise ValueError("Simulation parameter keys must be nonempty strings")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"Simulation parameter {key!r} must be finite")
    if not isinstance(value, (str, int, float, bool)) and value is not None:
        raise ValueError(f"Simulation parameter {key!r} has a non-scalar value")


class _FrozenMapping(Mapping[str, Any]):
    __slots__ = ("_items",)

    def __init__(self, items: tuple[tuple[str, Any], ...]) -> None:
        object.__setattr__(self, "_items", items)

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("frozen mapping does not support attribute assignment")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("frozen mapping does not support attribute deletion")

    def __getitem__(self, key: str) -> Any:
        for item_key, value in self._items:
            if item_key == key:
                return value
        raise KeyError(key)

    def __iter__(self):
        return (key for key, _ in self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __deepcopy__(self, memo: dict[int, Any]) -> dict[str, Any]:
        existing = memo.get(id(self))
        if existing is not None:
            return existing
        copied: dict[str, Any] = {}
        memo[id(self)] = copied
        for key, value in self._items:
            copied[key] = deepcopy(value, memo)
        return copied


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        frozen_items: list[tuple[str, Any]] = []
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("CircuitSpec.source mapping keys must be strings")
            frozen_items.append((key, _freeze(item)))
        return _FrozenMapping(tuple(frozen_items))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, float) and not math.isfinite(value):
        raise TypeError("CircuitSpec.source does not support non-finite floats")
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (set, frozenset)):
        raise TypeError("CircuitSpec.source does not support unordered sets")
    raise TypeError(
        f"CircuitSpec.source does not support values of type {type(value).__name__}"
    )


__all__ = [
    "CircuitOperation",
    "CircuitSpec",
    "ContractionDAG",
    "ContractNode",
    "GraphNode",
    "ReduceNode",
    "SimulationJob",
    "SimulationQuery",
    "SliceSpec",
    "TensorNetwork",
    "TensorSpec",
    "TensorView",
    "make_simulation_job",
]
