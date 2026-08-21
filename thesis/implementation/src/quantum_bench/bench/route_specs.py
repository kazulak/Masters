from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

REQUIRED_PIPELINE_ROLES = (
    "tensor_network",
    "planner",
    "numeric",
    "executor",
    "topology",
)
OPTIONAL_PIPELINE_ROLES = (
    "kernel",
    "partitioner",
    "scheduler",
    "communication",
)
KNOWN_PIPELINE_ROLES = REQUIRED_PIPELINE_ROLES + OPTIONAL_PIPELINE_ROLES


def _canonical_json(value: Mapping[str, Any]) -> str:
    """Return a deterministic JSON string for the provided mapping."""

    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return encoded.decode("utf-8")


def _hash_json(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _normalize_json_value(value: Any, *, path: str) -> Any:
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"Non-string JSON key at {path}: {type(key)!r}")
            normalized[key] = _normalize_json_value(item, path=f"{path}[{key!r}]")
        return normalized

    if isinstance(value, tuple):
        return [
            _normalize_json_value(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]

    if isinstance(value, list):
        return [
            _normalize_json_value(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]

    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        float_value = float(value)
        if not math.isfinite(float_value):
            raise ValueError(f"Non-finite value at {path}: {float_value}")
        return float_value

    if value is None or isinstance(value, bool | int | str):
        return value

    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"Non-finite value at {path}: {value}")
        return float(value)

    raise ValueError(f"Unsupported JSON type at {path}: {type(value)!r}")


def _normalize_json_mapping(values: Mapping[str, Any]) -> dict[str, Any]:
    return _normalize_json_value(dict(values), path="$")


@dataclass(frozen=True)
class PipelineParameters:
    """Immutable, canonical parameters for a route module."""

    _canonical_json: str

    def __init__(self, mapping: Mapping[str, Any] | None = None) -> None:
        if mapping is None:
            mapping = {}
        if not isinstance(mapping, Mapping):
            raise ValueError("PipelineParameters requires a top-level mapping")
        normalized = _normalize_json_mapping(mapping)
        object.__setattr__(self, "_canonical_json", _canonical_json(normalized))

    @property
    def canonical_json(self) -> str:
        return self._canonical_json

    @property
    def hash(self) -> str:
        return _hash_json(self._canonical_json)

    def to_dict(self) -> dict[str, Any]:
        return dict(json.loads(self._canonical_json))


@dataclass(frozen=True)
class ModuleSpec:
    role: str
    implementation: str
    parameters: PipelineParameters
    config_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if not self.role:
            raise ValueError("ModuleSpec.role must be non-empty")
        if not self.implementation:
            raise ValueError("ModuleSpec.implementation must be non-empty")
        if not isinstance(self.parameters, PipelineParameters):
            raise ValueError("ModuleSpec.parameters must be PipelineParameters")
        payload = {
            "role": self.role,
            "implementation": self.implementation,
            "parameters": self.parameters.to_dict(),
        }
        object.__setattr__(self, "config_hash", _hash_json(_canonical_json(payload)))


@dataclass(frozen=True)
class PipelineRoute:
    """A named route through the supported whole-circuit module roles."""

    route_id: str
    label: str
    modules: tuple[ModuleSpec, ...]
    route_config_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if not self.route_id:
            raise ValueError("PipelineRoute requires a non-empty route_id")
        if not self.label:
            raise ValueError("PipelineRoute requires a non-empty label")
        modules = tuple(self.modules)
        if not all(isinstance(module, ModuleSpec) for module in modules):
            raise ValueError("PipelineRoute modules must be ModuleSpec instances")
        roles = tuple(module.role for module in modules)
        unique_roles = set(roles)
        if len(unique_roles) != len(modules):
            raise ValueError("PipelineRoute modules must have unique roles")
        missing = set(REQUIRED_PIPELINE_ROLES) - unique_roles
        extra = unique_roles - set(KNOWN_PIPELINE_ROLES)
        if missing or extra:
            raise ValueError(
                f"PipelineRoute must define required roles {REQUIRED_PIPELINE_ROLES}; "
                f"missing={sorted(missing)} extra={sorted(extra)}"
            )
        ordered_modules = tuple(sorted(modules, key=lambda module: module.role))
        object.__setattr__(self, "modules", ordered_modules)
        payload = {
            "modules": {
                module.role: {
                    "implementation": module.implementation,
                    "parameters": module.parameters.to_dict(),
                    "config_hash": module.config_hash,
                }
                for module in ordered_modules
            }
        }
        object.__setattr__(
            self, "route_config_hash", _hash_json(_canonical_json(payload))
        )

    def module(self, role: str) -> ModuleSpec:
        for module in self.modules:
            if module.role == role:
                return module
        raise KeyError(f"Route {self.route_id} missing role {role}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "route_id": self.route_id,
            "label": self.label,
            "modules": {
                module.role: {
                    "implementation": module.implementation,
                    "parameters": module.parameters.to_dict(),
                    "config_hash": module.config_hash,
                }
                for module in self.modules
            },
            "route_config_hash": self.route_config_hash,
        }


@dataclass(frozen=True)
class ComparisonSpec:
    baseline_route: PipelineRoute
    candidate_route: PipelineRoute
    changed_roles: tuple[str, ...]
    label: str
    baseline_route_id: str = field(init=False)
    candidate_route_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not self.label:
            raise ValueError("ComparisonSpec.label must be non-empty")
        if self.baseline_route.route_id == self.candidate_route.route_id:
            raise ValueError("ComparisonSpec requires distinct routes")
        changed = tuple(self.changed_roles)
        if len(set(changed)) != len(changed):
            raise ValueError("ComparisonSpec.changed_roles must not contain duplicates")
        invalid = [role for role in changed if role not in KNOWN_PIPELINE_ROLES]
        if invalid:
            raise ValueError(
                f"ComparisonSpec contains invalid roles: {sorted(invalid)}"
            )

        baseline_modules = {
            module.role: module for module in self.baseline_route.modules
        }
        candidate_modules = {
            module.role: module for module in self.candidate_route.modules
        }
        roles = set(baseline_modules) | set(candidate_modules)
        role_differences = tuple(
            role
            for role in KNOWN_PIPELINE_ROLES
            if role in roles
            and baseline_modules.get(role) != candidate_modules.get(role)
        )
        if set(role_differences) != set(changed):
            raise ValueError(
                f"ComparisonSpec changed_roles mismatch with actual route differences. "
                f"declared={tuple(sorted(changed))} actual={tuple(sorted(role_differences))}"
            )
        object.__setattr__(self, "baseline_route_id", self.baseline_route.route_id)
        object.__setattr__(self, "candidate_route_id", self.candidate_route.route_id)
        object.__setattr__(self, "changed_roles", changed)

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "baseline_route": self.baseline_route.to_dict(),
            "candidate_route": self.candidate_route.to_dict(),
            "baseline_route_id": self.baseline_route_id,
            "candidate_route_id": self.candidate_route_id,
            "changed_roles": self.changed_roles,
        }
