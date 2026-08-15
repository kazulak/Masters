"""Small, typed strategy protocols and default implementations for whole-circuit execution."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import inspect
import json
import math
from pathlib import Path
from typing import Any, Mapping, Protocol, runtime_checkable

import numpy as np

from quantum_bench.core.records import ContractionTask


StrategyConfigValue = str | int | float | bool | None


class StrategyRole(str, Enum):
    DECOMPOSITION = "decomposition"
    PLACEMENT = "placement"
    KERNEL = "kernel"
    REDUCTION = "reduction"


@dataclass(frozen=True)
class StrategyIdentity:
    """Canonical scientific configuration identity for one execution strategy."""

    role: StrategyRole
    implementation_id: str
    version: str
    provider: str
    transport: str
    config: tuple[tuple[str, StrategyConfigValue], ...] = ()
    implementation_type: str = ""
    module_source_sha256: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.role, StrategyRole):
            raise TypeError("strategy identity role must be a StrategyRole")
        if not isinstance(self.config, tuple) or any(
            not isinstance(item, tuple) or len(item) != 2 for item in self.config
        ):
            raise TypeError(
                "strategy identity config must be a tuple of key/value tuples"
            )
        for field_name in ("implementation_id", "version", "provider", "transport"):
            if not getattr(self, field_name).strip():
                raise ValueError(f"strategy identity {field_name} must be non-empty")
        keys = [key for key, _ in self.config]
        if keys != sorted(keys) or len(keys) != len(set(keys)):
            raise ValueError("strategy identity config keys must be sorted and unique")
        for key, value in self.config:
            if not key:
                raise ValueError("strategy identity config keys must be non-empty")
            if not isinstance(value, (str, int, float, bool, type(None))):
                raise TypeError("strategy identity config values must be JSON scalars")
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError("strategy identity config floats must be finite")
        if not isinstance(self.implementation_type, str):
            raise TypeError("strategy implementation type must be a string")
        if self.module_source_sha256 is not None and (
            not isinstance(self.module_source_sha256, str)
            or len(self.module_source_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.module_source_sha256
            )
        ):
            raise ValueError(
                "strategy module source digest must be a SHA-256 hex digest"
            )

    def to_record(self) -> dict[str, Any]:
        return {
            "role": self.role.value,
            "implementation_id": self.implementation_id,
            "version": self.version,
            "provider": self.provider,
            "transport": self.transport,
            "config": dict(self.config),
            "implementation_type": self.implementation_type,
            "module_source_sha256": self.module_source_sha256,
        }


@dataclass(frozen=True)
class StrategyConfiguration:
    """Versioned canonical identity of the strategies used for one execution."""

    strategies: tuple[StrategyIdentity, ...]
    schema_version: str = "strategy_configuration_v2"

    def __post_init__(self) -> None:
        if not isinstance(self.strategies, tuple):
            raise TypeError("strategy configuration must contain a tuple of identities")
        if not self.schema_version.strip():
            raise ValueError("strategy configuration schema version must be non-empty")
        if any(
            not isinstance(identity, StrategyIdentity) for identity in self.strategies
        ):
            raise TypeError("strategy configuration entries must be StrategyIdentity")
        roles = [identity.role for identity in self.strategies]
        if roles != sorted(roles, key=lambda role: role.value):
            raise ValueError("strategy identities must be ordered by role")
        if len(roles) != len(set(roles)):
            raise ValueError("strategy configuration contains duplicate roles")

    def to_record(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "strategies": [identity.to_record() for identity in self.strategies],
        }

    @property
    def sha256(self) -> str:
        payload = json.dumps(
            self.to_record(), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("ascii")
        return hashlib.sha256(payload).hexdigest()


def bind_strategy_identity(
    strategy: Any, claimed_identity: StrategyIdentity
) -> StrategyIdentity:
    """Attach stable loaded-code evidence to a strategy's claimed identity."""

    strategy_type = type(strategy)
    implementation_type = f"{strategy_type.__module__}.{strategy_type.__qualname__}"
    module_source_sha256: str | None = None
    try:
        source_path = inspect.getsourcefile(strategy_type)
        if source_path is not None:
            module_source_sha256 = hashlib.sha256(
                Path(source_path).read_bytes()
            ).hexdigest()
    except (OSError, TypeError):
        pass
    # This attests the loaded type and source, not arbitrary runtime mutation.
    return StrategyIdentity(
        role=claimed_identity.role,
        implementation_id=claimed_identity.implementation_id,
        version=claimed_identity.version,
        provider=claimed_identity.provider,
        transport=claimed_identity.transport,
        config=claimed_identity.config,
        implementation_type=implementation_type,
        module_source_sha256=module_source_sha256,
    )


@runtime_checkable
class ReadyTaskOrderPolicy(Protocol):
    """Choose deterministic order among ready tasks; execution remains serial."""

    name: str

    def select_ready_tasks(
        self,
        pending: dict[str, ContractionTask],
        completed: set[str],
    ) -> list[ContractionTask]:
        """Select and order ready tasks from pending tasks given completed task IDs."""
        ...


@dataclass(frozen=True)
class SerialReadyTaskOrderPolicy:
    """Order ready tasks by ID while executing one task at a time."""

    name: str = "sequential_sorted_ready"

    def select_ready_tasks(
        self,
        pending: dict[str, ContractionTask],
        completed: set[str],
    ) -> list[ContractionTask]:
        return sorted(
            (task for task in pending.values() if set(task.dependencies) <= completed),
            key=lambda task: task.id,
        )


# Compatibility aliases for callers written before the ordering-only rename.
Scheduler = ReadyTaskOrderPolicy
TaskScheduler = ReadyTaskOrderPolicy
SequentialScheduler = SerialReadyTaskOrderPolicy
DefaultScheduler = SerialReadyTaskOrderPolicy


@runtime_checkable
class DecompositionStrategy(Protocol):
    """Protocol for decomposing a binary contraction task into bounded tiles."""

    def identity(self) -> StrategyIdentity: ...

    def decompose(
        self,
        task: ContractionTask,
        left: np.ndarray,
        right: np.ndarray,
        *,
        limits: Any = None,
    ) -> Any: ...


@runtime_checkable
class PlacementStrategy(Protocol):
    """Protocol for partitioning and placing tiles on rank-local DPUs."""

    def identity(self) -> StrategyIdentity: ...

    def place_waves(
        self,
        tiles: tuple[Any, ...],
        total_dpu_count: int,
    ) -> tuple[tuple[Any, ...], ...]: ...

    def map_wave_to_ranks(
        self,
        wave: tuple[Any, ...],
        ranks: tuple[Any, ...],
    ) -> list[tuple[Any, list[tuple[Any, int]]]]: ...


@runtime_checkable
class KernelProvider(Protocol):
    """Protocol for building work units, preparing requests, and decoding outputs."""

    def identity(self) -> StrategyIdentity: ...

    def build_work_unit(
        self,
        tile: Any,
        local_id: int,
        left: np.ndarray,
        right: np.ndarray,
        packed: bool,
    ) -> Any: ...

    def prepare_request(
        self,
        root: Path,
        *,
        profile: Any,
        lowering: Any,
        work_units: list[Any],
        task_contract_sha256: str,
        request_sequence: int,
    ) -> Any: ...

    def read_output(
        self,
        path: Path,
        tile: Any,
        *,
        packed: bool,
    ) -> np.ndarray: ...


@runtime_checkable
class ReductionProvider(Protocol):
    """Protocol for assembling partial execution results into host arrays."""

    def identity(self) -> StrategyIdentity: ...

    def reduce(
        self,
        lowering: Any,
        partials: Mapping[str, np.ndarray],
        *,
        packed: bool = False,
        scale: float = 1.0,
    ) -> np.ndarray: ...


__all__ = [
    "Scheduler",
    "TaskScheduler",
    "SequentialScheduler",
    "DefaultScheduler",
    "ReadyTaskOrderPolicy",
    "SerialReadyTaskOrderPolicy",
    "StrategyConfigValue",
    "StrategyRole",
    "StrategyIdentity",
    "StrategyConfiguration",
    "bind_strategy_identity",
    "DecompositionStrategy",
    "PlacementStrategy",
    "KernelProvider",
    "ReductionProvider",
]
