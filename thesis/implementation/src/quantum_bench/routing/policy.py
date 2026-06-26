from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Literal, Mapping, Sequence

from quantum_bench.core.records import JsonDict, to_jsonable
from quantum_bench.routing.records import TaskRouteDecision


ShadowRoutePolicyId = Literal[
    "cpu-only",
    "dense-if-estimate-supported",
    "dense-if-no-tiling",
    "dense-if-bridge-ready",
]
ShadowRoutePolicyStatus = Literal["selected_cpu", "selected_dense", "blocked_to_cpu"]

SHADOW_ROUTE_POLICY_IDS: tuple[str, ...] = (
    "cpu-only",
    "dense-if-estimate-supported",
    "dense-if-no-tiling",
    "dense-if-bridge-ready",
)

SHADOW_POLICY_BLOCKERS: tuple[str, ...] = (
    "missing_dense_estimate",
    "unsupported_dense_estimate",
    "requires_tiling_not_allowed",
    "bridge_manifest_ineligible",
    "working_set_exceeds_threshold",
    "tile_count_exceeds_threshold",
    "simplepim_unavailable",
)


@dataclass(frozen=True)
class ShadowRoutePolicyConfig:
    policy_id: ShadowRoutePolicyId = "cpu-only"
    max_working_set_bytes: int | None = None
    max_tile_count: int | None = None
    allow_tiling: bool = False
    require_simplepim_configured: bool = False

    def to_json_dict(self) -> JsonDict:
        return to_jsonable(self)


@dataclass(frozen=True)
class ShadowRoutePolicyDecision:
    policy_id: str
    selected_route: str
    status: ShadowRoutePolicyStatus
    reason: str
    blockers: tuple[str, ...] = ()
    dense_estimate_supported: bool = False
    dense_estimate_present: bool = False
    metadata: JsonDict = field(default_factory=dict)

    def to_json_dict(self) -> JsonDict:
        return to_jsonable(self)


@dataclass(frozen=True)
class ShadowRoutePolicySummary:
    shadow_policy_id: str
    selected_route_counts: JsonDict
    dense_gemm_candidate_count: int
    cpu_fallback_count: int
    blocker_counts: JsonDict
    total_host_to_dpu_bytes_for_policy_dense: int
    total_dpu_to_host_bytes_for_policy_dense: int
    total_mram_to_wram_bytes_for_policy_dense: int
    max_tile_count_for_policy_dense: int

    def to_json_dict(self) -> JsonDict:
        return to_jsonable(self)


def evaluate_shadow_route_policy(
    *,
    config: ShadowRoutePolicyConfig,
    task_decisions: Sequence[TaskRouteDecision],
    dense_record: Mapping[str, object],
    simplepim_probe: Mapping[str, object],
) -> ShadowRoutePolicyDecision:
    if config.policy_id == "cpu-only":
        return ShadowRoutePolicyDecision(
            policy_id=config.policy_id,
            selected_route="cpu_fallback",
            status="selected_cpu",
            reason="policy_cpu_only",
        )

    dense_decision = _dense_decision(task_decisions)
    estimate = dense_decision.estimate if dense_decision is not None else None
    blockers: list[str] = []

    if dense_decision is None or estimate is None or estimate.reason == "missing_target_estimate":
        blockers.append("missing_dense_estimate")
        return _blocked(config, blockers, dense_decision)

    if not bool(estimate.supported):
        blockers.append("unsupported_dense_estimate")
        return _blocked(config, blockers, dense_decision)

    if config.policy_id == "dense-if-no-tiling" and estimate.requires_tiling is True:
        blockers.append("requires_tiling_not_allowed")
        return _blocked(config, blockers, dense_decision)

    if config.policy_id == "dense-if-bridge-ready" and dense_record.get("bridge_manifest_eligible") is not True:
        blockers.append("bridge_manifest_ineligible")
        return _blocked(config, blockers, dense_decision)

    if config.policy_id != "dense-if-no-tiling" and not config.allow_tiling and estimate.requires_tiling is True:
        blockers.append("requires_tiling_not_allowed")

    if config.max_working_set_bytes is not None:
        working_set = estimate.estimated_peak_memory
        if working_set is not None and int(working_set) > int(config.max_working_set_bytes):
            blockers.append("working_set_exceeds_threshold")

    if config.max_tile_count is not None:
        tile_count = estimate.estimated_tile_count
        if tile_count is not None and int(tile_count) > int(config.max_tile_count):
            blockers.append("tile_count_exceeds_threshold")

    if config.require_simplepim_configured and not _simplepim_available(simplepim_probe):
        blockers.append("simplepim_unavailable")

    if blockers:
        return _blocked(config, blockers, dense_decision)

    return ShadowRoutePolicyDecision(
        policy_id=config.policy_id,
        selected_route="dense_gemm",
        status="selected_dense",
        reason="dense_policy_selected",
        blockers=(),
        dense_estimate_supported=True,
        dense_estimate_present=True,
        metadata=_estimate_metadata(dense_decision),
    )


def summarize_shadow_route_policy(rows: Sequence[Mapping[str, object]]) -> ShadowRoutePolicySummary:
    policy_ids = {str(row.get("shadow_policy_id")) for row in rows if row.get("shadow_policy_id")}
    policy_id = next(iter(policy_ids)) if len(policy_ids) == 1 else ",".join(sorted(policy_ids))
    selected_route_counts = Counter(str(row.get("shadow_policy_selected_route")) for row in rows)
    blocker_counts = Counter()
    for row in rows:
        for blocker in row.get("shadow_policy_blockers") or ():
            blocker_counts[str(blocker)] += 1

    dense_selected_rows = [row for row in rows if row.get("shadow_policy_selected_route") == "dense_gemm"]
    return ShadowRoutePolicySummary(
        shadow_policy_id=policy_id,
        selected_route_counts=dict(sorted(selected_route_counts.items())),
        dense_gemm_candidate_count=sum(1 for row in rows if _row_dense_estimate_supported(row)),
        cpu_fallback_count=sum(1 for row in rows if row.get("shadow_policy_selected_route") == "cpu_fallback"),
        blocker_counts={blocker: int(blocker_counts.get(blocker, 0)) for blocker in SHADOW_POLICY_BLOCKERS},
        total_host_to_dpu_bytes_for_policy_dense=sum(_row_dense_estimate_int(row, "host_to_dpu_bytes") for row in dense_selected_rows),
        total_dpu_to_host_bytes_for_policy_dense=sum(_row_dense_estimate_int(row, "dpu_to_host_bytes") for row in dense_selected_rows),
        total_mram_to_wram_bytes_for_policy_dense=sum(_row_dense_estimate_int(row, "mram_to_wram_bytes") for row in dense_selected_rows),
        max_tile_count_for_policy_dense=max(
            (_row_dense_estimate_int(row, "estimated_tile_count") for row in dense_selected_rows),
            default=0,
        ),
    )


def _blocked(
    config: ShadowRoutePolicyConfig,
    blockers: Sequence[str],
    dense_decision: TaskRouteDecision | None,
) -> ShadowRoutePolicyDecision:
    reason = blockers[0] if blockers else "dense_policy_blocked"
    return ShadowRoutePolicyDecision(
        policy_id=config.policy_id,
        selected_route="cpu_fallback",
        status="blocked_to_cpu",
        reason=reason,
        blockers=tuple(blockers),
        dense_estimate_supported=bool(dense_decision is not None and dense_decision.estimate.supported),
        dense_estimate_present=dense_decision is not None,
        metadata=_estimate_metadata(dense_decision),
    )


def _dense_decision(task_decisions: Sequence[TaskRouteDecision]) -> TaskRouteDecision | None:
    return next((decision for decision in task_decisions if decision.route_id == "dense_gemm"), None)


def _estimate_metadata(dense_decision: TaskRouteDecision | None) -> JsonDict:
    if dense_decision is None:
        return {}
    estimate = dense_decision.estimate
    return {
        "wram_fit": estimate.wram_fit,
        "requires_tiling": estimate.requires_tiling,
        "tiling_implemented": estimate.tiling_implemented,
        "host_to_dpu_bytes": estimate.host_to_dpu_bytes,
        "dpu_to_host_bytes": estimate.dpu_to_host_bytes,
        "mram_to_wram_bytes": estimate.mram_to_wram_bytes,
        "estimated_tile_count": estimate.estimated_tile_count,
        "estimated_peak_memory": estimate.estimated_peak_memory,
    }


def _simplepim_available(simplepim_probe: Mapping[str, object]) -> bool:
    if bool(simplepim_probe.get("simplepim_available")):
        return True
    return str(simplepim_probe.get("simplepim_probe_status")) == "available"


def _row_dense_estimate_supported(row: Mapping[str, object]) -> bool:
    for route in row.get("candidate_routes") or ():
        if isinstance(route, Mapping) and route.get("route_id") == "dense_gemm":
            estimate = route.get("estimate")
            return isinstance(estimate, Mapping) and bool(estimate.get("supported"))
    return False


def _row_dense_estimate_int(row: Mapping[str, object], field: str) -> int:
    for route in row.get("candidate_routes") or ():
        if isinstance(route, Mapping) and route.get("route_id") == "dense_gemm":
            estimate = route.get("estimate")
            if isinstance(estimate, Mapping):
                value = estimate.get(field)
                try:
                    return max(0, int(value)) if value is not None else 0
                except (TypeError, ValueError):
                    return 0
    return 0
