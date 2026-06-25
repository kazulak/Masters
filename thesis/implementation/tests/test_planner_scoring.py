from __future__ import annotations

import pytest

from quantum_bench.bench.planner_scoring import DEFAULT_SCORING_WEIGHTS, score_planner_rows, validate_scoring_weights


def _row(
    case_id: str,
    planner_id: str,
    *,
    flops: int = 100,
    peak_bytes: int = 1000,
    host_to_dpu: int = 100,
    dpu_to_host: int = 100,
    mram_to_wram: int = 100,
    task_count: int = 4,
    estimated_total_tile_count: int = 4,
    tiling_required_task_count: int = 0,
    unsupported_task_count: int = 0,
    missing_target_estimate_count: int = 0,
    estimated_max_parallel_tiles: int = 1,
) -> dict[str, object]:
    return {
        "case_id": case_id,
        "workload_id": case_id,
        "planner_id": planner_id,
        "total_estimated_flops": flops,
        "peak_intermediate_bytes": peak_bytes,
        "total_host_to_dpu_bytes": host_to_dpu,
        "total_dpu_to_host_bytes": dpu_to_host,
        "total_mram_to_wram_bytes": mram_to_wram,
        "task_count": task_count,
        "estimated_total_tile_count": estimated_total_tile_count,
        "tiling_required_task_count": tiling_required_task_count,
        "unsupported_task_count": unsupported_task_count,
        "missing_target_estimate_count": missing_target_estimate_count,
        "estimated_max_parallel_tiles": estimated_max_parallel_tiles,
    }


def _by_planner(rows: list[dict[str, object]], case_id: str) -> dict[str, dict[str, object]]:
    return {str(row["planner_id"]): row for row in rows if row["case_id"] == case_id}


def test_scoring_normalizes_within_each_case_only() -> None:
    rows = score_planner_rows(
        [
            _row("case_a", "low_transfer", host_to_dpu=1),
            _row("case_a", "high_transfer", host_to_dpu=2),
            _row("case_b", "low_transfer", host_to_dpu=100),
            _row("case_b", "high_transfer", host_to_dpu=200),
        ],
        DEFAULT_SCORING_WEIGHTS,
    )

    case_a = _by_planner(rows, "case_a")
    case_b = _by_planner(rows, "case_b")
    assert case_a["low_transfer"]["upmem_rank"] == 1
    assert case_a["high_transfer"]["upmem_rank"] == 2
    assert case_b["low_transfer"]["upmem_rank"] == 1
    assert case_b["high_transfer"]["upmem_rank"] == 2
    assert case_a["low_transfer"]["upmem_pressure_score"] == case_b["low_transfer"]["upmem_pressure_score"]
    assert case_a["high_transfer"]["upmem_pressure_score"] == case_b["high_transfer"]["upmem_pressure_score"]


def test_one_tile_per_task_is_not_tiling_pressure() -> None:
    rows = score_planner_rows(
        [
            _row("case", "one_tile_per_task", task_count=4, estimated_total_tile_count=4),
            _row("case", "extra_tile", task_count=4, estimated_total_tile_count=5),
        ],
        DEFAULT_SCORING_WEIGHTS,
    )

    by_planner = _by_planner(rows, "case")
    one_tile_components = by_planner["one_tile_per_task"]["score_components"]
    extra_tile_components = by_planner["extra_tile"]["score_components"]
    assert one_tile_components["raw"]["extra_tile_raw"] == 0.0
    assert one_tile_components["raw"]["tiling_raw"] == 0.0
    assert extra_tile_components["raw"]["extra_tile_raw"] == 1.0
    assert extra_tile_components["raw"]["tiling_raw"] == 1.0
    assert by_planner["one_tile_per_task"]["upmem_rank"] == 1
    assert by_planner["extra_tile"]["upmem_rank"] == 2


def test_potential_parallelism_is_a_conservative_bonus() -> None:
    rows = score_planner_rows(
        [
            _row("case", "low_parallelism", estimated_max_parallel_tiles=1),
            _row("case", "higher_parallelism", estimated_max_parallel_tiles=4),
        ],
        DEFAULT_SCORING_WEIGHTS,
    )

    by_planner = _by_planner(rows, "case")
    assert by_planner["higher_parallelism"]["upmem_rank"] == 1
    assert by_planner["low_parallelism"]["upmem_rank"] == 2
    assert by_planner["higher_parallelism"]["score_weights"]["parallelism_bonus_weight"] == 0.25
    assert by_planner["higher_parallelism"]["upmem_pressure_score"] == -0.25


def test_dense_ranks_allow_ties_for_rank_one() -> None:
    rows = score_planner_rows(
        [
            _row("case", "planner_a"),
            _row("case", "planner_b"),
        ],
        DEFAULT_SCORING_WEIGHTS,
    )

    by_planner = _by_planner(rows, "case")
    assert by_planner["planner_a"]["upmem_rank"] == 1
    assert by_planner["planner_b"]["upmem_rank"] == 1
    assert by_planner["planner_a"]["flop_rank"] == 1
    assert by_planner["planner_b"]["flop_rank"] == 1


def test_flop_rank_and_upmem_rank_can_differ() -> None:
    rows = score_planner_rows(
        [
            _row("case", "low_flops_high_transfer", flops=10, host_to_dpu=1000),
            _row("case", "high_flops_low_transfer", flops=20, host_to_dpu=1),
        ],
        DEFAULT_SCORING_WEIGHTS,
    )

    by_planner = _by_planner(rows, "case")
    assert by_planner["low_flops_high_transfer"]["flop_rank"] == 1
    assert by_planner["low_flops_high_transfer"]["upmem_rank"] == 2
    assert by_planner["high_flops_low_transfer"]["flop_rank"] == 2
    assert by_planner["high_flops_low_transfer"]["upmem_rank"] == 1
    assert "lowest modeled FLOPs" in by_planner["low_flops_high_transfer"]["tradeoff_note"]
    assert "lowest modeled UPMEM pressure" in by_planner["high_flops_low_transfer"]["tradeoff_note"]


def test_invalid_scoring_weights_fail_explicitly() -> None:
    with pytest.raises(ValueError, match="Unknown planner comparison scoring weight"):
        validate_scoring_weights({"bad_weight": 1.0})
    with pytest.raises(ValueError, match="must be nonnegative"):
        validate_scoring_weights({"transfer_weight": -1.0})
    with pytest.raises(ValueError, match="must be numeric"):
        validate_scoring_weights({"transfer_weight": "high"})
