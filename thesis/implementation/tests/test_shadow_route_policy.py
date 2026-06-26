from __future__ import annotations

from quantum_bench.routing.policy import (
    ShadowRoutePolicyConfig,
    evaluate_shadow_route_policy,
    summarize_shadow_route_policy,
)
from quantum_bench.routing.records import (
    TASK_ROUTE_DECISION_SCHEMA_VERSION,
    STATIC_TASK_ROUTER_ID,
    TaskRouteDecision,
    TaskRouteEstimate,
    TaskRouteExecutionStatus,
)


def _dense_decision(
    *,
    supported: bool = True,
    reason: str | None = None,
    requires_tiling: bool | None = False,
    working_set: int | None = 128,
    tile_count: int | None = 1,
    h2d: int | None = 10,
    d2h: int | None = 20,
    mram: int | None = 30,
) -> TaskRouteDecision:
    return TaskRouteDecision(
        schema_version=TASK_ROUTE_DECISION_SCHEMA_VERSION,
        router_id=STATIC_TASK_ROUTER_ID,
        case_id="case",
        task_id="task_0",
        task_index=0,
        input_tensor_ids=("a", "b"),
        output_tensor_id="c",
        route_id="dense_gemm",
        route_family="dense_gemm",
        kernel_family="dense_gemm",
        hardware_target="upmem_dpu",
        execution_mode="future_simplepim_or_native",
        maturity_level=2,
        status="skipped" if supported else "rejected",
        is_selected=False,
        execution_status=TaskRouteExecutionStatus(
            state="estimate_only",
            execution_implemented=False,
            can_prepare=False,
            can_execute=False,
            can_validate=False,
        ),
        reason=reason,
        estimate=TaskRouteEstimate(
            supported=supported,
            estimated_flops=100,
            estimated_bytes=(h2d or 0) + (d2h or 0) + (mram or 0),
            estimated_peak_memory=working_set,
            requires_tiling=requires_tiling,
            host_to_dpu_bytes=h2d,
            dpu_to_host_bytes=d2h,
            mram_to_wram_bytes=mram,
            estimated_tile_count=tile_count,
            reason=reason,
        ),
    )


def _probe(available: bool = False) -> dict[str, object]:
    return {"simplepim_available": available, "simplepim_probe_status": "available" if available else "unavailable"}


def test_cpu_only_policy_selects_cpu_with_explicit_reason() -> None:
    decision = evaluate_shadow_route_policy(
        config=ShadowRoutePolicyConfig(policy_id="cpu-only"),
        task_decisions=[_dense_decision()],
        dense_record={"bridge_manifest_eligible": True},
        simplepim_probe=_probe(),
    )

    assert decision.status == "selected_cpu"
    assert decision.selected_route == "cpu_fallback"
    assert decision.reason == "policy_cpu_only"
    assert decision.blockers == ()


def test_dense_if_estimate_supported_selects_dense() -> None:
    decision = evaluate_shadow_route_policy(
        config=ShadowRoutePolicyConfig(policy_id="dense-if-estimate-supported"),
        task_decisions=[_dense_decision()],
        dense_record={"bridge_manifest_eligible": False},
        simplepim_probe=_probe(),
    )

    assert decision.status == "selected_dense"
    assert decision.selected_route == "dense_gemm"
    assert decision.reason == "dense_policy_selected"


def test_missing_and_unsupported_estimate_precede_thresholds() -> None:
    missing = evaluate_shadow_route_policy(
        config=ShadowRoutePolicyConfig(policy_id="dense-if-estimate-supported", max_tile_count=0),
        task_decisions=[],
        dense_record={"bridge_manifest_eligible": True},
        simplepim_probe=_probe(),
    )
    unsupported = evaluate_shadow_route_policy(
        config=ShadowRoutePolicyConfig(policy_id="dense-if-estimate-supported", max_working_set_bytes=1),
        task_decisions=[_dense_decision(supported=False, reason="unsupported_dense_gemm_shape")],
        dense_record={"bridge_manifest_eligible": True},
        simplepim_probe=_probe(),
    )

    assert missing.status == "blocked_to_cpu"
    assert missing.blockers == ("missing_dense_estimate",)
    assert unsupported.status == "blocked_to_cpu"
    assert unsupported.blockers == ("unsupported_dense_estimate",)


def test_dense_if_no_tiling_ignores_allow_tiling() -> None:
    decision = evaluate_shadow_route_policy(
        config=ShadowRoutePolicyConfig(policy_id="dense-if-no-tiling", allow_tiling=True),
        task_decisions=[_dense_decision(requires_tiling=True)],
        dense_record={"bridge_manifest_eligible": True},
        simplepim_probe=_probe(),
    )

    assert decision.status == "blocked_to_cpu"
    assert decision.blockers == ("requires_tiling_not_allowed",)


def test_threshold_blockers_apply_after_policy_specific_blockers() -> None:
    bridge_blocked = evaluate_shadow_route_policy(
        config=ShadowRoutePolicyConfig(policy_id="dense-if-bridge-ready", max_working_set_bytes=1),
        task_decisions=[_dense_decision(working_set=128)],
        dense_record={"bridge_manifest_eligible": False},
        simplepim_probe=_probe(),
    )
    threshold_blocked = evaluate_shadow_route_policy(
        config=ShadowRoutePolicyConfig(policy_id="dense-if-estimate-supported", max_working_set_bytes=1, max_tile_count=0),
        task_decisions=[_dense_decision(working_set=128, tile_count=1)],
        dense_record={"bridge_manifest_eligible": True},
        simplepim_probe=_probe(),
    )

    assert bridge_blocked.blockers == ("bridge_manifest_ineligible",)
    assert threshold_blocked.blockers == ("working_set_exceeds_threshold", "tile_count_exceeds_threshold")


def test_simplepim_unavailable_blocks_only_when_required() -> None:
    not_required = evaluate_shadow_route_policy(
        config=ShadowRoutePolicyConfig(policy_id="dense-if-estimate-supported"),
        task_decisions=[_dense_decision()],
        dense_record={"bridge_manifest_eligible": True},
        simplepim_probe=_probe(available=False),
    )
    required = evaluate_shadow_route_policy(
        config=ShadowRoutePolicyConfig(policy_id="dense-if-estimate-supported", require_simplepim_configured=True),
        task_decisions=[_dense_decision()],
        dense_record={"bridge_manifest_eligible": True},
        simplepim_probe=_probe(available=False),
    )

    assert not_required.status == "selected_dense"
    assert required.status == "blocked_to_cpu"
    assert required.blockers == ("simplepim_unavailable",)


def test_policy_summary_counts_candidates_and_selected_dense_transfers() -> None:
    rows = [
        {
            "shadow_policy_id": "dense-if-estimate-supported",
            "shadow_policy_selected_route": "dense_gemm",
            "shadow_policy_blockers": [],
            "candidate_routes": [{"route_id": "dense_gemm", "estimate": {"supported": True, "host_to_dpu_bytes": 10, "dpu_to_host_bytes": 20, "mram_to_wram_bytes": 30, "estimated_tile_count": 2}}],
        },
        {
            "shadow_policy_id": "dense-if-estimate-supported",
            "shadow_policy_selected_route": "cpu_fallback",
            "shadow_policy_blockers": ["tile_count_exceeds_threshold"],
            "candidate_routes": [{"route_id": "dense_gemm", "estimate": {"supported": True, "host_to_dpu_bytes": 99, "dpu_to_host_bytes": 99, "mram_to_wram_bytes": 99, "estimated_tile_count": 9}}],
        },
    ]

    summary = summarize_shadow_route_policy(rows)

    assert summary.dense_gemm_candidate_count == 2
    assert summary.cpu_fallback_count == 1
    assert summary.blocker_counts["tile_count_exceeds_threshold"] == 1
    assert summary.total_host_to_dpu_bytes_for_policy_dense == 10
    assert summary.total_dpu_to_host_bytes_for_policy_dense == 20
    assert summary.total_mram_to_wram_bytes_for_policy_dense == 30
    assert summary.max_tile_count_for_policy_dense == 2
