from __future__ import annotations

import json
from collections import defaultdict
from typing import Any


SCORE_MODEL = "upmem_pressure_v1"
RANK_EPSILON = 1.0e-12

DEFAULT_SCORING_WEIGHTS = {
    "transfer_weight": 1.0,
    "memory_weight": 1.0,
    "tiling_weight": 1.0,
    "unsupported_weight": 10.0,
    "missing_estimate_weight": 10.0,
    "parallelism_bonus_weight": 0.25,
}

SCORING_FORMULA = (
    "per-case min-max normalized weighted pressure: "
    "transfer + memory + tiling + unsupported + missing estimates - conservative modeled/potential parallelism bonus"
)


def scoring_metadata(weights: dict[str, float]) -> dict[str, Any]:
    return {
        "score_model": SCORE_MODEL,
        "rank_scope": "case_id",
        "normalization": "per_case_minmax",
        "rank_order": "lower_score_is_better",
        "tie_policy": f"dense_rank_epsilon_{RANK_EPSILON:g}",
        "weights": dict(weights),
        "formula": SCORING_FORMULA,
    }


def validate_scoring_weights(value: dict[str, Any] | None) -> dict[str, float]:
    weights = dict(DEFAULT_SCORING_WEIGHTS)
    if value is None:
        return weights
    unknown = sorted(set(value) - set(DEFAULT_SCORING_WEIGHTS))
    if unknown:
        raise ValueError(f"Unknown planner comparison scoring weight(s): {', '.join(unknown)}")
    for key, raw in value.items():
        if not isinstance(raw, (int, float)):
            raise ValueError(f"planner_comparison.scoring.{key} must be numeric")
        numeric = float(raw)
        if numeric < 0:
            raise ValueError(f"planner_comparison.scoring.{key} must be nonnegative")
        weights[key] = numeric
    return weights


def score_planner_rows(rows: list[dict[str, Any]], weights: dict[str, float]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["case_id"])].append(row)

    scored_rows: list[dict[str, Any]] = []
    for case_rows in grouped.values():
        components_by_planner = {str(row["planner_id"]): _raw_components(row) for row in case_rows}
        normalized_by_planner = _normalized_components(components_by_planner)
        score_by_planner = {
            planner_id: _weighted_score(normalized, weights)
            for planner_id, normalized in normalized_by_planner.items()
        }
        upmem_ranks = _dense_ranks(score_by_planner)
        flop_ranks = _dense_ranks({str(row["planner_id"]): float(row["total_estimated_flops"]) for row in case_rows})
        upmem_winners = {planner_id for planner_id, rank in upmem_ranks.items() if rank == 1}
        flop_winners = {planner_id for planner_id, rank in flop_ranks.items() if rank == 1}

        for row in case_rows:
            planner_id = str(row["planner_id"])
            raw = components_by_planner[planner_id]
            normalized = normalized_by_planner[planner_id]
            weighted = _weighted_components(normalized, weights)
            scored = dict(row)
            scored.update(
                {
                    "score_model": SCORE_MODEL,
                    "upmem_pressure_score": score_by_planner[planner_id],
                    "upmem_rank": upmem_ranks[planner_id],
                    "flop_rank": flop_ranks[planner_id],
                    "score_components": {
                        "raw": raw,
                        "normalized": normalized,
                        "weighted": weighted,
                    },
                    "score_weights": dict(weights),
                    "tradeoff_note": _tradeoff_note(scored, planner_id, flop_winners, upmem_winners),
                }
            )
            scored_rows.append(scored)

    return scored_rows


def markdown_summary(rows: list[dict[str, Any]]) -> str:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["case_id"])].append(row)
    summary = divergence_summary(rows)
    divergent = set(summary["divergent_case_ids"])

    lines = [
        "# Planner Comparison Summary",
        "",
        "Modeled UPMEM-pressure ranking is analysis only and is not used for execution.",
        f"Divergent modeled winner cases: {summary['divergent_case_count']} / {summary['case_count']}.",
        "",
        "| case_id | FLOP winner(s) | UPMEM-score winner(s) | differ? | best UPMEM score | note |",
        "|---|---|---|---:|---:|---|",
    ]
    for case_id, case_rows in sorted(grouped.items(), key=lambda item: (item[0] not in divergent, item[0])):
        flop_winners = sorted(str(row["planner_id"]) for row in case_rows if int(row["flop_rank"]) == 1)
        upmem_winners = sorted(str(row["planner_id"]) for row in case_rows if int(row["upmem_rank"]) == 1)
        best_score = min(float(row["upmem_pressure_score"]) for row in case_rows)
        differ = "yes" if set(flop_winners) != set(upmem_winners) else "no"
        note = "modeled FLOP and UPMEM-pressure winners differ" if differ == "yes" else "same modeled winner set"
        lines.append(
            f"| {case_id} | {', '.join(flop_winners)} | {', '.join(upmem_winners)} | {differ} | {best_score:.6g} | {note} |"
        )
    lines.append("")
    return "\n".join(lines)


def divergence_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["case_id"])].append(row)
    divergent_case_ids = []
    for case_id, case_rows in sorted(grouped.items()):
        flop_winners = {str(row["planner_id"]) for row in case_rows if int(row["flop_rank"]) == 1}
        upmem_winners = {str(row["planner_id"]) for row in case_rows if int(row["upmem_rank"]) == 1}
        if flop_winners != upmem_winners:
            divergent_case_ids.append(case_id)
    return {
        "rank_scope": "case_id",
        "case_count": len(grouped),
        "divergent_case_count": len(divergent_case_ids),
        "divergent_case_ids": divergent_case_ids,
    }


def csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return value


def _raw_components(row: dict[str, Any]) -> dict[str, float]:
    transfer_raw = (
        _float(row, "total_host_to_dpu_bytes")
        + _float(row, "total_dpu_to_host_bytes")
        + _float(row, "total_mram_to_wram_bytes")
    )
    task_count = _float(row, "task_count")
    estimated_total_tile_count = _float(row, "estimated_total_tile_count")
    extra_tile_raw = max(0.0, estimated_total_tile_count - task_count)
    tiling_raw = _float(row, "tiling_required_task_count") + extra_tile_raw
    return {
        "transfer_raw": transfer_raw,
        "memory_raw": _float(row, "peak_intermediate_bytes"),
        "extra_tile_raw": extra_tile_raw,
        "tiling_raw": tiling_raw,
        "unsupported_raw": _float(row, "unsupported_task_count"),
        "missing_estimate_raw": _float(row, "missing_target_estimate_count"),
        "potential_parallelism_raw": _float(row, "estimated_max_parallel_tiles"),
    }


def _normalized_components(raw_by_planner: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    component_names = next(iter(raw_by_planner.values())).keys() if raw_by_planner else ()
    mins = {name: min(raw[name] for raw in raw_by_planner.values()) for name in component_names}
    maxes = {name: max(raw[name] for raw in raw_by_planner.values()) for name in component_names}
    normalized: dict[str, dict[str, float]] = {}
    for planner_id, raw in raw_by_planner.items():
        normalized[planner_id] = {
            _normalized_name(name): _normalize(raw[name], mins[name], maxes[name])
            for name in component_names
        }
    return normalized


def _weighted_score(normalized: dict[str, float], weights: dict[str, float]) -> float:
    return (
        weights["transfer_weight"] * normalized["transfer_component"]
        + weights["memory_weight"] * normalized["memory_component"]
        + weights["tiling_weight"] * normalized["tiling_component"]
        + weights["unsupported_weight"] * normalized["unsupported_component"]
        + weights["missing_estimate_weight"] * normalized["missing_estimate_component"]
        - weights["parallelism_bonus_weight"] * normalized["potential_parallelism_benefit"]
    )


def _weighted_components(normalized: dict[str, float], weights: dict[str, float]) -> dict[str, float]:
    return {
        "transfer_weighted": weights["transfer_weight"] * normalized["transfer_component"],
        "memory_weighted": weights["memory_weight"] * normalized["memory_component"],
        "tiling_weighted": weights["tiling_weight"] * normalized["tiling_component"],
        "unsupported_weighted": weights["unsupported_weight"] * normalized["unsupported_component"],
        "missing_estimate_weighted": weights["missing_estimate_weight"] * normalized["missing_estimate_component"],
        "potential_parallelism_bonus": weights["parallelism_bonus_weight"] * normalized["potential_parallelism_benefit"],
    }


def _normalized_name(raw_name: str) -> str:
    if raw_name == "potential_parallelism_raw":
        return "potential_parallelism_benefit"
    return raw_name.replace("_raw", "_component")


def _normalize(value: float, minimum: float, maximum: float) -> float:
    if abs(maximum - minimum) <= RANK_EPSILON:
        return 0.0
    return (value - minimum) / (maximum - minimum)


def _dense_ranks(values: dict[str, float]) -> dict[str, int]:
    unique_values: list[float] = []
    for value in sorted(values.values()):
        if not unique_values or abs(value - unique_values[-1]) > RANK_EPSILON:
            unique_values.append(value)
    ranks: dict[str, int] = {}
    for key, value in values.items():
        for idx, unique in enumerate(unique_values, start=1):
            if abs(value - unique) <= RANK_EPSILON:
                ranks[key] = idx
                break
    return ranks


def _tradeoff_note(row: dict[str, Any], planner_id: str, flop_winners: set[str], upmem_winners: set[str]) -> str:
    is_flop_winner = planner_id in flop_winners
    is_upmem_winner = planner_id in upmem_winners
    if is_flop_winner and is_upmem_winner:
        note = "same modeled FLOP and UPMEM-pressure winner"
    elif is_flop_winner:
        note = "lowest modeled FLOPs, but not lowest modeled UPMEM pressure"
    elif is_upmem_winner:
        note = "lowest modeled UPMEM pressure, but not lowest modeled FLOPs"
    else:
        note = "not the modeled FLOP or UPMEM-pressure winner"
    if int(row.get("unsupported_task_count", 0) or 0) > 0:
        note += "; includes unsupported task estimates"
    if int(row.get("missing_target_estimate_count", 0) or 0) > 0:
        note += "; includes missing target estimates"
    return note


def _float(row: dict[str, Any], key: str) -> float:
    return float(row.get(key, 0) or 0)
