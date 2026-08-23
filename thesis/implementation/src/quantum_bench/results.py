"""Immutable results and failure contracts for one execution sample."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import TypeAlias

import numpy as np


JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | tuple["JsonValue", ...] | Mapping[str, "JsonValue"]


@dataclass(frozen=True, slots=True)
class Measurement:
    scope_id: str
    total_wall_s: float
    lowering_s: float | None = None
    planning_s: float | None = None
    slicing_s: float | None = None
    mapping_s: float | None = None
    session_open_s: float | None = None
    encode_s: float | None = None
    preparation_s: float | None = None
    h2d_s: float | None = None
    kernel_s: float | None = None
    host_reduce_s: float | None = None
    d2h_s: float | None = None
    decode_s: float | None = None
    rank_work_s: float | None = None
    h2d_bytes: int | None = None
    d2h_bytes: int | None = None
    energy_j: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.scope_id, str) or not self.scope_id:
            raise ValueError("scope_id must be a nonempty string")
        for name in (
            "total_wall_s",
            "lowering_s",
            "planning_s",
            "slicing_s",
            "mapping_s",
            "session_open_s",
            "encode_s",
            "preparation_s",
            "h2d_s",
            "kernel_s",
            "host_reduce_s",
            "d2h_s",
            "decode_s",
            "rank_work_s",
            "energy_j",
        ):
            value = getattr(self, name)
            if value is not None:
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ValueError(f"{name} must be a finite non-negative number")
                if not math.isfinite(float(value)) or float(value) < 0.0:
                    raise ValueError(f"{name} must be a finite non-negative number")
                object.__setattr__(self, name, float(value))
        for name in ("h2d_bytes", "d2h_bytes"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise ValueError(f"{name} must be a non-negative integer")
            if value is not None:
                object.__setattr__(self, name, int(value))


@dataclass(frozen=True, slots=True)
class ExecutionSample:
    output: np.ndarray
    measurement: Measurement
    backend_facts: Mapping[str, JsonValue]
    numeric_facts: Mapping[str, JsonValue]

    def __post_init__(self) -> None:
        if not isinstance(self.measurement, Measurement):
            raise TypeError("measurement must be a Measurement")
        output = np.array(self.output, copy=True, order="C")
        output.setflags(write=False)
        object.__setattr__(self, "output", output)
        object.__setattr__(self, "backend_facts", _freeze_mapping(self.backend_facts))
        object.__setattr__(self, "numeric_facts", _freeze_mapping(self.numeric_facts))


class UnsupportedExecution(RuntimeError):
    """A preflight rejection before runtime side effects."""

    __slots__ = ("stage", "reason", "capability")

    def __init__(self, stage: str, reason: str, capability: str) -> None:
        if not all(
            isinstance(value, str) and value for value in (stage, reason, capability)
        ):
            raise ValueError("stage, reason, and capability must be nonempty strings")
        self.stage = stage
        self.reason = reason
        self.capability = capability
        super().__init__(f"unsupported execution at {stage}: {reason} ({capability})")


class ExecutionFailed(RuntimeError):
    """A runtime attempt that failed after side effects could begin."""

    __slots__ = ("stage", "reason", "backend_facts")

    def __init__(
        self,
        stage: str,
        reason: str,
        backend_facts: Mapping[str, JsonValue],
    ) -> None:
        if not all(isinstance(value, str) and value for value in (stage, reason)):
            raise ValueError("stage and reason must be nonempty strings")
        self.stage = stage
        self.reason = reason
        self.backend_facts = _freeze_mapping(backend_facts)
        super().__init__(f"execution failed at {stage}: {reason}")


def _freeze_mapping(value: Mapping[str, JsonValue]) -> Mapping[str, JsonValue]:
    if not isinstance(value, Mapping):
        raise TypeError("facts must be a mapping")
    frozen: dict[str, JsonValue] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise TypeError("fact mapping keys must be strings")
        frozen[key] = _freeze_value(item)
    return MappingProxyType(frozen)


def _freeze_value(value: object) -> JsonValue:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("fact floats must be finite")
        return value
    if isinstance(value, Mapping):
        return _freeze_mapping(value)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_value(item) for item in value)
    raise TypeError(f"unsupported fact value type: {type(value).__name__}")


__all__ = [
    "JsonScalar",
    "JsonValue",
    "Measurement",
    "ExecutionSample",
    "UnsupportedExecution",
    "ExecutionFailed",
]
