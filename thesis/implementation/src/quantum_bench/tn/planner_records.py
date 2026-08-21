"""Shared immutable records for planner adapters.

This module contains no planner implementations.  It keeps the active
metadata-only planner boundary independent from the legacy planner module.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any


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
        # Preserve the legacy serialized options surface.
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


def canonical_planner_config_hash(config: dict[str, Any]) -> str:
    """Return the stable identity hash for a resolved planner configuration."""

    encoded = json.dumps(
        config,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=_json_default,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _json_default(value: object) -> str:
    return repr(value)
