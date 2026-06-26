from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from quantum_bench.core.records import JsonDict


TASK_ROUTE_DECISION_SCHEMA_VERSION = "task_route_decision_v1"
TASK_ROUTE_SUMMARY_SCHEMA_VERSION = "task_route_summary_v1"
STATIC_TASK_ROUTER_ID = "static_task_router_v1"

TaskRouteDecisionStatus = Literal["selected", "rejected", "skipped", "unavailable", "fallback"]
TaskRouteExecutionState = Literal["not_executable", "estimate_only", "fallback_available", "future_backend"]

TASK_ROUTE_STATUSES: tuple[str, ...] = ("selected", "rejected", "skipped", "unavailable", "fallback")


@dataclass(frozen=True)
class TaskRouteIdentity:
    route_id: str
    display_name: str
    route_family: str
    kernel_family: str
    hardware_target: str
    execution_mode: str
    maturity_level: int


@dataclass(frozen=True)
class TaskRouteCapabilities:
    identity: TaskRouteIdentity
    status: str
    supported_task_structures: tuple[str, ...] = ()
    can_estimate: bool = False
    can_prepare: bool = False
    can_execute: bool = False
    can_return_output: bool = False
    reason: str | None = None
    metadata: JsonDict = field(default_factory=dict)


@dataclass(frozen=True)
class TaskRouteEstimate:
    supported: bool
    estimated_flops: int
    estimated_bytes: int
    estimated_peak_memory: int | None
    wram_fit: bool | None = None
    requires_tiling: bool | None = None
    tiling_implemented: bool | None = None
    host_to_dpu_bytes: int | None = None
    dpu_to_host_bytes: int | None = None
    mram_to_wram_bytes: int | None = None
    estimated_tile_count: int | None = None
    estimated_parallel_tiles: int | None = None
    reason: str | None = None
    metadata: JsonDict = field(default_factory=dict)


@dataclass(frozen=True)
class TaskRouteExecutionStatus:
    state: TaskRouteExecutionState
    execution_implemented: bool
    can_prepare: bool
    can_execute: bool
    can_validate: bool
    reason: str | None = None


@dataclass(frozen=True)
class TaskRouteContext:
    suite_id: str
    case_id: str
    run_dir: Path | None = None
    policy: str = "analysis_only_cpu_fallback"
    decisions_artifact: str | None = None
    target_artifacts: JsonDict = field(default_factory=dict)
    backend_probes: JsonDict = field(default_factory=dict)


@dataclass(frozen=True)
class TaskRouteDecision:
    schema_version: str
    router_id: str
    case_id: str
    task_id: str
    task_index: int
    input_tensor_ids: tuple[str, str]
    output_tensor_id: str
    route_id: str
    route_family: str
    kernel_family: str
    hardware_target: str
    execution_mode: str
    maturity_level: int
    status: TaskRouteDecisionStatus
    is_selected: bool
    execution_status: TaskRouteExecutionStatus
    reason: str | None
    estimate: TaskRouteEstimate
    metadata: JsonDict = field(default_factory=dict)


@dataclass(frozen=True)
class TaskRoutingAnalysis:
    router_id: str
    case_id: str
    decisions: tuple[TaskRouteDecision, ...]
    summary: JsonDict
