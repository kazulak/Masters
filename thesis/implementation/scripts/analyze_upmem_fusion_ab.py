#!/usr/bin/env python3
"""Analyze the bounded, normalized UPMEM fusion A/B packet.

This module is deliberately a pure analysis boundary.  ``analyze_rows`` only
consumes normalized rows; it does not inspect evidence directories, execute a
backend, generate candidates, or make an acceptance decision.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping, Sequence
import hashlib
import json
import math
import random
from statistics import median
from pathlib import Path
from typing import Any


ANALYSIS_VERSION = "upmem_fusion_ab_analysis_v1"
DEFAULT_SEED = 20260907
DEFAULT_BOOTSTRAP_RESAMPLES = 10_000

EXPECTED_CASES = (
    "quantization_stress_16q_l2",
    "hs_20q_d1",
    "edc_14q",
)
EXPECTED_DPU_COUNTS = (1, 4)
EXPECTED_ARMS = ("unfused", "fused")
EXPECTED_ATTEMPTS = ("warmup", "measurement")
WARMUP_BLOCKS = (0,)
MEASUREMENT_BLOCKS = (1, 2, 3, 4, 5)

IDENTITY_FIELDS = (
    "case_id",
    "dpu_count",
    "arm",
    "attempt_kind",
    "block_id",
    "sample_index",
)
TIME_FIELDS = (
    "steady_s",
    "session_open_s",
    "session_close_s",
    "kernel_s",
)
REPORTED_METRICS = ("steady_s", "session_inclusive_s", "kernel_s")
SETUP_METRICS = ("session_open_s", "session_close_s", "setup_s")
ALL_DERIVED_METRICS = REPORTED_METRICS + SETUP_METRICS
DERIVED_INPUT_METRICS = set(ALL_DERIVED_METRICS)
TIME_EPSILON_S = 1e-9


def _require_int(value: object, field: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{field} must be an integer")
    return value


def _finite_nonnegative(value: object, field: str) -> float:
    if type(value) not in (int, float):
        raise ValueError(f"{field} must be a finite non-negative number")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0.0:
        raise ValueError(f"{field} must be a finite non-negative number")
    return numeric


def _finite_positive(value: object, field: str) -> float:
    numeric = _finite_nonnegative(value, field)
    if numeric <= 0.0:
        raise ValueError(f"{field} must be positive")
    return numeric


def _raw_mad(values: Sequence[float]) -> float:
    center = median(values)
    return float(median(abs(value - center) for value in values))


def _stats(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        raise ValueError("cannot summarize an empty timing array")
    checked = [float(value) for value in values]
    return {
        "raw": checked,
        "count": len(checked),
        "median": float(median(checked)),
        "raw_mad": _raw_mad(checked),
        "min": min(checked),
        "max": max(checked),
    }


def _percentile(values: Sequence[float], fraction: float) -> float:
    if not values or not 0.0 <= fraction <= 1.0:
        raise ValueError("percentile requires non-empty values and [0, 1]")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] + weight * (ordered[upper] - ordered[lower])


def _seed_for(seed: int, *parts: object) -> int:
    """Derive stable independent streams without using process-randomized hash()."""

    encoded = "\0".join([str(seed), *(str(part) for part in parts)]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big")


def _bootstrap_summary(
    values: Sequence[float], *, seed: int, bootstrap_resamples: int
) -> dict[str, Any]:
    """Bootstrap a median while resampling complete paired observations."""

    if not values:
        raise ValueError("cannot bootstrap an empty array")
    rng = random.Random(seed)
    resampled_medians: list[float] = []
    size = len(values)
    for _ in range(bootstrap_resamples):
        indexes = [rng.randrange(size) for _ in range(size)]
        resampled_medians.append(float(median(values[index] for index in indexes)))
    return {
        "bootstrap_95_ci": [
            _percentile(resampled_medians, 0.025),
            _percentile(resampled_medians, 0.975),
        ],
        "bootstrap_resamples": bootstrap_resamples,
    }


def _bootstrap_ratio_of_medians(
    unfused: Sequence[float],
    fused: Sequence[float],
    *,
    seed: int,
    bootstrap_resamples: int,
) -> dict[str, Any]:
    """Bootstrap the primary ratio of arm medians using paired indexes."""

    if len(unfused) != len(fused) or not unfused:
        raise ValueError("paired A/B arrays must have equal non-zero length")
    rng = random.Random(seed)
    resampled_ratios: list[float] = []
    size = len(unfused)
    for _ in range(bootstrap_resamples):
        indexes = [rng.randrange(size) for _ in range(size)]
        unfused_median = median(unfused[index] for index in indexes)
        fused_median = median(fused[index] for index in indexes)
        if fused_median <= 0.0:
            raise ValueError("paired speedup denominator must be positive")
        ratio = float(unfused_median / fused_median)
        if not math.isfinite(ratio) or ratio <= 0.0:
            raise ValueError("paired speedup is not finite and positive")
        resampled_ratios.append(ratio)
    return {
        "bootstrap_95_ci": [
            _percentile(resampled_ratios, 0.025),
            _percentile(resampled_ratios, 0.975),
        ],
        "bootstrap_resamples": bootstrap_resamples,
    }


def _validate_optional_timing_fields(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    """Reject partially populated optional timing columns.

    A producer may carry an additional timing column when it is available, or
    carry it as null.  It must have one consistent state for the entire packet;
    in particular, an unavailable value is never represented by a fabricated
    zero.  Measured columns are retained as timing statistics and paired
    differences; null columns remain explicitly unavailable.
    """

    optional_fields = sorted(
        {
            key
            for row in rows
            for key in row
            if key.endswith("_s")
            and key not in TIME_FIELDS
            and key not in DERIVED_INPUT_METRICS
        }
    )
    states_by_field: dict[str, str] = {}
    for field in optional_fields:
        states = {
            "missing" if field not in row else "null" if row[field] is None else "value"
            for row in rows
        }
        if len(states) != 1:
            raise ValueError(
                f"optional timing {field} must be omitted or populated consistently"
            )
        if states == {"value"}:
            for row in rows:
                _finite_nonnegative(row[field], field)
        states_by_field[field] = next(iter(states))
    return states_by_field


def _validate_row(
    raw_row: Mapping[str, Any], index: int
) -> tuple[tuple[str, int, str, str, int], dict[str, Any]]:
    missing = [field for field in (*IDENTITY_FIELDS, *TIME_FIELDS) if field not in raw_row]
    if missing:
        raise ValueError(f"row {index} is missing {missing[0]}")

    case_id = raw_row["case_id"]
    if case_id not in EXPECTED_CASES:
        raise ValueError(f"row {index} has an unknown case_id: {case_id!r}")
    dpu_count = _require_int(raw_row["dpu_count"], f"row {index} dpu_count")
    if dpu_count not in EXPECTED_DPU_COUNTS:
        raise ValueError(f"row {index} has an unexpected dpu_count")
    arm = raw_row["arm"]
    if arm not in EXPECTED_ARMS:
        raise ValueError(f"row {index} has an unknown arm: {arm!r}")
    attempt_kind = raw_row["attempt_kind"]
    if attempt_kind not in EXPECTED_ATTEMPTS:
        raise ValueError(f"row {index} has an unknown attempt_kind")

    block_id = _require_int(raw_row["block_id"], f"row {index} block_id")
    sample_index = _require_int(raw_row["sample_index"], f"row {index} sample_index")
    if attempt_kind == "warmup":
        expected_block = 0
        expected_sample = 0
    else:
        expected_block = block_id
        expected_sample = block_id - 1
        if block_id not in MEASUREMENT_BLOCKS:
            raise ValueError(f"row {index} measurement block_id must be 1..5")
    if block_id != expected_block or sample_index != expected_sample:
        raise ValueError(
            f"row {index} has invalid sample/block semantics for {attempt_kind}"
        )

    steady_s = _finite_nonnegative(raw_row["steady_s"], f"row {index} steady_s")
    session_open_s = _finite_nonnegative(
        raw_row["session_open_s"], f"row {index} session_open_s"
    )
    session_close_s = _finite_nonnegative(
        raw_row["session_close_s"], f"row {index} session_close_s"
    )
    kernel_s = _finite_positive(raw_row["kernel_s"], f"row {index} kernel_s")
    allowed_excess = TIME_EPSILON_S * max(1.0, abs(steady_s), abs(kernel_s))
    if kernel_s > steady_s + allowed_excess:
        raise ValueError(f"row {index} kernel_s exceeds steady_s")

    session_inclusive_s = session_open_s + steady_s + session_close_s
    setup_s = session_open_s + session_close_s
    if not math.isfinite(session_inclusive_s) or not math.isfinite(setup_s):
        raise ValueError(f"row {index} derived timing is not finite")

    normalized = {
        "case_id": case_id,
        "dpu_count": dpu_count,
        "arm": arm,
        "attempt_kind": attempt_kind,
        "block_id": block_id,
        "sample_index": sample_index,
        "steady_s": steady_s,
        "session_open_s": session_open_s,
        "session_close_s": session_close_s,
        "kernel_s": kernel_s,
        # These are calculated per row before any median or pairing occurs.
        "session_inclusive_s": session_inclusive_s,
        "setup_s": setup_s,
    }
    key = (case_id, dpu_count, arm, attempt_kind, block_id)
    return key, normalized


def _paired_comparison(
    block_ids: Sequence[int],
    unfused: Sequence[float],
    fused: Sequence[float],
    *,
    seed: int,
    bootstrap_resamples: int,
) -> dict[str, Any]:
    if len(block_ids) != len(unfused) or len(unfused) != len(fused) or not unfused:
        raise ValueError("paired A/B arrays must have equal non-zero length")
    if any(value <= 0.0 for value in fused):
        raise ValueError("paired speedup denominator must be positive")

    speedups = [
        float(unfused_value / fused_value)
        for unfused_value, fused_value in zip(unfused, fused, strict=True)
    ]
    saved_time = [
        float(unfused_value - fused_value)
        for unfused_value, fused_value in zip(unfused, fused, strict=True)
    ]
    if any(not math.isfinite(value) or value <= 0.0 for value in speedups):
        raise ValueError("paired speedup is not finite and positive")

    unfused_median = float(median(unfused))
    fused_median = float(median(fused))
    if fused_median <= 0.0:
        raise ValueError("paired speedup denominator must be positive")
    ratio_of_medians = unfused_median / fused_median
    if not math.isfinite(ratio_of_medians) or ratio_of_medians <= 0.0:
        raise ValueError("paired speedup is not finite and positive")
    primary_bootstrap = _bootstrap_ratio_of_medians(
        unfused,
        fused,
        seed=_seed_for(seed, "ratio_of_arm_medians"),
        bootstrap_resamples=bootstrap_resamples,
    )
    descriptive_bootstrap = _bootstrap_summary(
        speedups,
        seed=_seed_for(seed, "median_of_block_speedups"),
        bootstrap_resamples=bootstrap_resamples,
    )
    saved_bootstrap = _bootstrap_summary(
        saved_time,
        seed=_seed_for(seed, "saved_time"),
        bootstrap_resamples=bootstrap_resamples,
    )
    return {
        "block_ids": [int(block_id) for block_id in block_ids],
        "speedups": speedups,
        "saved_time_s": saved_time,
        "primary_estimand": "ratio_of_arm_medians",
        "primary_speedup": ratio_of_medians,
        "ratio_of_medians": ratio_of_medians,
        "median_of_block_speedups": float(median(speedups)),
        "paired_saved_time_s": float(median(saved_time)),
        "saved_time_from_medians_s": unfused_median - fused_median,
        "bootstrap_95_ci": primary_bootstrap["bootstrap_95_ci"],
        "median_of_block_speedups_bootstrap_95_ci": descriptive_bootstrap[
            "bootstrap_95_ci"
        ],
        "saved_time_bootstrap_95_ci_s": saved_bootstrap["bootstrap_95_ci"],
        "bootstrap_resamples": bootstrap_resamples,
    }


def _paired_difference(
    block_ids: Sequence[int],
    unfused: Sequence[float],
    fused: Sequence[float],
    *,
    seed: int,
    bootstrap_resamples: int,
) -> dict[str, Any]:
    if len(block_ids) != len(unfused) or len(unfused) != len(fused) or not unfused:
        raise ValueError("paired A/B arrays must have equal non-zero length")
    differences = [
        float(unfused_value - fused_value)
        for unfused_value, fused_value in zip(unfused, fused, strict=True)
    ]
    bootstrap = _bootstrap_summary(
        differences,
        seed=_seed_for(seed, "difference"),
        bootstrap_resamples=bootstrap_resamples,
    )
    return {
        "block_ids": [int(block_id) for block_id in block_ids],
        "differences_s": differences,
        "median_difference_s": float(median(differences)),
        "difference_from_medians_s": float(median(unfused) - median(fused)),
        "bootstrap_95_ci_s": bootstrap["bootstrap_95_ci"],
        "bootstrap_resamples": bootstrap_resamples,
    }


def _geometric_mean(values: Sequence[float]) -> float:
    if not values or any(value <= 0.0 or not math.isfinite(value) for value in values):
        raise ValueError("geometric mean requires finite positive values")
    return float(math.exp(sum(math.log(value) for value in values) / len(values)))


def _aggregate_speedup(
    cells: Sequence[Mapping[str, Any]],
    metric: str,
    *,
    seed: int,
    bootstrap_resamples: int,
) -> dict[str, Any]:
    cell_values = [
        {
            "case_id": cell["case_id"],
            "dpu_count": cell["dpu_count"],
            "ratio_of_arm_medians": cell["comparisons"][metric][
                "ratio_of_medians"
            ],
            "median_of_block_speedups": cell["comparisons"][metric][
                "median_of_block_speedups"
            ],
        }
        for cell in cells
    ]
    arm_values = [
        (
            cell["arms"]["unfused"]["metrics"][metric]["raw"],
            cell["arms"]["fused"]["metrics"][metric]["raw"],
        )
        for cell in cells
    ]
    if not arm_values or any(
        len(unfused) != len(MEASUREMENT_BLOCKS)
        or len(fused) != len(MEASUREMENT_BLOCKS)
        for unfused, fused in arm_values
    ):
        raise ValueError("aggregate requires complete measurement block pairing")

    point = _geometric_mean(
        [item["ratio_of_arm_medians"] for item in cell_values]
    )
    rng = random.Random(_seed_for(seed, "aggregate", metric, "speedup"))
    bootstrap_values: list[float] = []
    block_count = len(MEASUREMENT_BLOCKS)
    for _ in range(bootstrap_resamples):
        indexes = [rng.randrange(block_count) for _ in range(block_count)]
        cell_ratios = []
        for unfused, fused in arm_values:
            unfused_median = median(unfused[index] for index in indexes)
            fused_median = median(fused[index] for index in indexes)
            if fused_median <= 0.0:
                raise ValueError("paired speedup denominator must be positive")
            cell_ratios.append(float(unfused_median / fused_median))
        bootstrap_values.append(_geometric_mean(cell_ratios))
    return {
        "cell_speedups": cell_values,
        "primary_estimand": "equal_cell_geometric_mean_of_ratio_of_arm_medians",
        "equal_cell_geometric_mean_speedup": point,
        "bootstrap_95_ci": [
            _percentile(bootstrap_values, 0.025),
            _percentile(bootstrap_values, 0.975),
        ],
        "bootstrap_resamples": bootstrap_resamples,
        "bootstrap_method": (
            "common_measurement_block_resampling_then_equal_cell_geometric_mean"
        ),
    }


def analyze_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    seed: int = DEFAULT_SEED,
    bootstrap_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
) -> dict[str, Any]:
    """Validate and analyze the complete six-cell fusion A/B packet.

    The five measurement blocks are paired by ``case_id``, ``dpu_count`` and
    ``block_id``.  Warmups are validated and retained in the result for audit,
    but are excluded from all measurement summaries and comparisons.
    """

    if isinstance(rows, (str, bytes, Mapping)):
        raise TypeError("rows must be an iterable of row mappings")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    if type(bootstrap_resamples) is not int or bootstrap_resamples <= 0:
        raise ValueError("bootstrap_resamples must be a positive integer")

    materialized = list(rows)
    expected_count = (
        len(EXPECTED_CASES)
        * len(EXPECTED_DPU_COUNTS)
        * len(EXPECTED_ARMS)
        * (len(WARMUP_BLOCKS) + len(MEASUREMENT_BLOCKS))
    )
    if len(materialized) != expected_count:
        raise ValueError(
            f"expected exactly {expected_count} rows, received {len(materialized)}"
        )
    mappings: list[Mapping[str, Any]] = []
    for index, row in enumerate(materialized):
        if not isinstance(row, Mapping):
            raise ValueError(f"row {index} must be a mapping")
        mappings.append(row)
    optional_states = _validate_optional_timing_fields(mappings)
    optional_fields = tuple(sorted(optional_states))

    records: dict[tuple[str, int, str, str, int], dict[str, Any]] = {}
    for index, row in enumerate(mappings):
        key, normalized = _validate_row(row, index)
        if key in records:
            raise ValueError(f"duplicate row for {key}")
        for field, state in optional_states.items():
            normalized[field] = None if state == "null" else float(row[field])
        records[key] = normalized

    expected_keys = {
        (case_id, dpu_count, arm, attempt_kind, block_id)
        for case_id in EXPECTED_CASES
        for dpu_count in EXPECTED_DPU_COUNTS
        for arm in EXPECTED_ARMS
        for attempt_kind in EXPECTED_ATTEMPTS
        for block_id in (
            WARMUP_BLOCKS if attempt_kind == "warmup" else MEASUREMENT_BLOCKS
        )
    }
    missing = sorted(expected_keys - records.keys())
    extra = sorted(records.keys() - expected_keys)
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing rows {missing[:3]}")
        if extra:
            details.append(f"unexpected rows {extra[:3]}")
        raise ValueError("packet rows are incomplete: " + "; ".join(details))

    cells: list[dict[str, Any]] = []
    for case_id in EXPECTED_CASES:
        for dpu_count in EXPECTED_DPU_COUNTS:
            cell: dict[str, Any] = {
                "case_id": case_id,
                "dpu_count": dpu_count,
                "measurement_blocks": list(MEASUREMENT_BLOCKS),
                "arms": {},
                "comparisons": {},
                "setup_differences": {},
                "paired_differences": {},
            }
            measurement_values: dict[str, dict[str, Any]] = {}
            for arm in EXPECTED_ARMS:
                warmup = records[(case_id, dpu_count, arm, "warmup", 0)]
                measurements = [
                    records[(case_id, dpu_count, arm, "measurement", block_id)]
                    for block_id in MEASUREMENT_BLOCKS
                ]
                metric_values = {
                    metric: [float(row[metric]) for row in measurements]
                    for metric in (*REPORTED_METRICS, *SETUP_METRICS)
                }
                for metric in optional_fields:
                    metric_values[metric] = (
                        [float(row[metric]) for row in measurements]
                        if optional_states[metric] == "value"
                        else None
                    )
                measurement_values[arm] = metric_values
                metric_stats = {
                    metric: _stats(values) if values is not None else None
                    for metric, values in metric_values.items()
                }
                cell["arms"][arm] = {
                    "warmup": {
                        "block_id": warmup["block_id"],
                        "sample_index": warmup["sample_index"],
                        **{
                            metric: float(warmup[metric])
                            for metric in ALL_DERIVED_METRICS
                        },
                        **{
                            metric: warmup[metric]
                            for metric in optional_fields
                        },
                    },
                    "measurement_blocks": list(MEASUREMENT_BLOCKS),
                    "metrics": metric_stats,
                }

            for metric in REPORTED_METRICS:
                cell["comparisons"][metric] = _paired_comparison(
                    MEASUREMENT_BLOCKS,
                    measurement_values["unfused"][metric],
                    measurement_values["fused"][metric],
                    seed=_seed_for(seed, case_id, dpu_count, metric),
                    bootstrap_resamples=bootstrap_resamples,
                )
            for metric in SETUP_METRICS:
                difference = _paired_difference(
                    MEASUREMENT_BLOCKS,
                    measurement_values["unfused"][metric],
                    measurement_values["fused"][metric],
                    seed=_seed_for(seed, case_id, dpu_count, metric),
                    bootstrap_resamples=bootstrap_resamples,
                )
                cell["setup_differences"][metric] = difference
                cell["paired_differences"][metric] = difference
            for metric in optional_fields:
                if optional_states[metric] == "value":
                    cell["paired_differences"][metric] = _paired_difference(
                        MEASUREMENT_BLOCKS,
                        measurement_values["unfused"][metric],
                        measurement_values["fused"][metric],
                        seed=_seed_for(seed, case_id, dpu_count, metric),
                        bootstrap_resamples=bootstrap_resamples,
                    )
                else:
                    cell["paired_differences"][metric] = None
            cells.append(cell)

    aggregate = {
        "cell_count": len(cells),
        "metrics": {
            metric: _aggregate_speedup(
                cells,
                metric,
                seed=seed,
                bootstrap_resamples=bootstrap_resamples,
            )
            for metric in REPORTED_METRICS
        },
    }
    return {
        "analysis_version": ANALYSIS_VERSION,
        "seed": seed,
        "bootstrap_resamples": bootstrap_resamples,
        "cases": list(EXPECTED_CASES),
        "dpu_counts": list(EXPECTED_DPU_COUNTS),
        "arms": list(EXPECTED_ARMS),
        "attempts": {
            "warmup": {"count_per_arm_cell": len(WARMUP_BLOCKS), "block_ids": list(WARMUP_BLOCKS)},
            "measurement": {
                "count_per_arm_cell": len(MEASUREMENT_BLOCKS),
                "block_ids": list(MEASUREMENT_BLOCKS),
            },
        },
        "row_count": len(materialized),
        "optional_timing_fields": {
            field: (
                "measured" if state == "value" else "unavailable"
            )
            for field, state in optional_states.items()
        },
        "timing_semantics": {
            "session_inclusive_s": "session_open_s + steady_s + session_close_s per row before medians",
            "setup_s": "session_open_s + session_close_s per row before paired differences",
            "primary_speedup": "median(unfused) divided by median(fused) within each cell",
            "primary_bootstrap": "paired block indexes resampled before both arm medians",
            "descriptive_speedup": "median of unfused/fused ratios for paired blocks",
            "saved_time_s": "unfused minus fused for the same cell and measurement block",
            "aggregate": "equal-cell aggregation; raw runtimes are never pooled",
        },
        "cells": cells,
        "aggregate": aggregate,
    }


def _read_rows(path: Path) -> list[Mapping[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError("input JSON must be a list of normalized rows")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_positional", nargs="?", type=Path)
    parser.add_argument("output_positional", nargs="?", type=Path)
    parser.add_argument("--input", dest="input_option", type=Path)
    parser.add_argument("--output", "--output-json", dest="output_option", type=Path)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--bootstrap-resamples", type=int, default=DEFAULT_BOOTSTRAP_RESAMPLES
    )
    args = parser.parse_args(argv)

    if args.input_option is not None and args.input_positional is not None:
        parser.error("provide input either positionally or with --input")
    if args.output_option is not None and args.output_positional is not None:
        parser.error("provide output either positionally or with --output")
    input_path = args.input_option or args.input_positional
    if input_path is None:
        parser.error("an input JSON path is required")
    output_path = args.output_option or args.output_positional

    result = analyze_rows(
        _read_rows(input_path),
        seed=args.seed,
        bootstrap_resamples=args.bootstrap_resamples,
    )
    encoded = json.dumps(result, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    if output_path is None:
        print(encoded, end="")
    else:
        output_path.write_text(encoded, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
