#!/usr/bin/env python3
"""Analyze the bounded Stress16/four-DPU fusion confirmation packet.

The input is a single, complete normalized confirmation cell.  This module
does not read the earlier fusion A/B packet, execute hardware, or make a
production-adoption decision.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping, Sequence
import json
import math
from pathlib import Path
from statistics import median
from typing import Any

from scripts.analyze_upmem_fusion_ab import (
    _bootstrap_ratio_of_medians,
    _paired_difference,
    _seed_for,
    _stats,
    _validate_row as _validate_ab_row,
    _validate_optional_timing_fields,
)


ANALYSIS_VERSION = "upmem_fusion_confirmation_analysis_v1"
DEFAULT_SEED = 20260908
DEFAULT_BOOTSTRAP_RESAMPLES = 10_000

EXPECTED_CASE_ID = "quantization_stress_16q_l2"
EXPECTED_DPU_COUNT = 4
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
DERIVED_METRICS = ("session_inclusive_s", "setup_s")
STAT_METRICS = (
    "steady_s",
    "session_inclusive_s",
    "kernel_s",
    "session_open_s",
    "session_close_s",
    "setup_s",
)
COMPARISON_METRICS = ("steady_s", "session_inclusive_s", "kernel_s")
SETUP_METRICS = ("session_open_s", "session_close_s", "setup_s")
MINIMUM_REDUCTION = 0.05
MAXIMUM_CELL_REGRESSION = 0.05


def _validate_row(
    raw_row: Mapping[str, Any], index: int
) -> tuple[tuple[str, str, int, int], dict[str, Any]]:
    ab_key, normalized = _validate_ab_row(raw_row, index)
    case_id, dpu_count, arm, attempt_kind, block_id = ab_key
    if case_id != EXPECTED_CASE_ID:
        raise ValueError(f"row {index} has the wrong case_id")
    if dpu_count != EXPECTED_DPU_COUNT:
        raise ValueError(f"row {index} has the wrong topology dpu_count")
    key = (arm, attempt_kind, block_id, normalized["sample_index"])
    return key, normalized


def _record_for_output(
    record: Mapping[str, Any], optional_fields: Sequence[str]
) -> dict[str, Any]:
    return {
        field: record[field]
        for field in (*IDENTITY_FIELDS, *TIME_FIELDS, *DERIVED_METRICS, *optional_fields)
    }


def _comparison(
    block_ids: Sequence[int],
    unfused: Sequence[float],
    fused: Sequence[float],
    *,
    metric: str,
    seed: int,
    bootstrap_resamples: int,
) -> dict[str, Any]:
    if len(block_ids) != len(unfused) or len(unfused) != len(fused) or not unfused:
        raise ValueError("paired A/B arrays must have equal non-zero length")
    unfused_median = float(median(unfused))
    fused_median = float(median(fused))
    if unfused_median <= 0.0 or fused_median <= 0.0:
        raise ValueError(f"{metric} arm medians must be positive")
    ratio_of_medians = unfused_median / fused_median
    if not math.isfinite(ratio_of_medians) or ratio_of_medians <= 0.0:
        raise ValueError(f"{metric} ratio of arm medians is not finite and positive")

    bootstrap = _bootstrap_ratio_of_medians(
        unfused,
        fused,
        seed=_seed_for(seed, metric, "ratio_of_arm_medians"),
        bootstrap_resamples=bootstrap_resamples,
    )
    return {
        "block_ids": [int(block_id) for block_id in block_ids],
        "unfused_median": unfused_median,
        "fused_median": fused_median,
        "primary_estimand": "ratio_of_arm_medians",
        "primary_speedup": ratio_of_medians,
        "ratio_of_medians": ratio_of_medians,
        "reduction_fraction": 1.0 - (fused_median / unfused_median),
        "bootstrap_95_ci": bootstrap["bootstrap_95_ci"],
        "lower_95_bound": bootstrap["bootstrap_95_ci"][0],
        "upper_95_bound": bootstrap["bootstrap_95_ci"][1],
        "bootstrap_resamples": bootstrap_resamples,
    }


def _validate_complete_packet(
    rows: Iterable[Mapping[str, Any]],
    *,
    seed: int,
    bootstrap_resamples: int,
) -> tuple[
    dict[tuple[str, str, int, int], dict[str, Any]],
    dict[str, str],
]:
    if isinstance(rows, (str, bytes, Mapping)):
        raise TypeError("rows must be an iterable of row mappings")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    if type(bootstrap_resamples) is not int or bootstrap_resamples <= 0:
        raise ValueError("bootstrap_resamples must be a positive integer")

    materialized = list(rows)
    expected_count = len(EXPECTED_ARMS) * (
        len(WARMUP_BLOCKS) + len(MEASUREMENT_BLOCKS)
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
    records: dict[tuple[str, str, int, int], dict[str, Any]] = {}
    for index, row in enumerate(mappings):
        key, normalized = _validate_row(row, index)
        if key in records:
            raise ValueError(f"duplicate row for {key}")
        for field, state in optional_states.items():
            normalized[field] = None if state == "null" else float(row[field])
        records[key] = normalized

    expected_keys = {
        (arm, "warmup", 0, 0) for arm in EXPECTED_ARMS
    } | {
        (arm, "measurement", block_id, block_id - 1)
        for arm in EXPECTED_ARMS
        for block_id in MEASUREMENT_BLOCKS
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
    return records, optional_states


def analyze_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    seed: int = DEFAULT_SEED,
    bootstrap_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
) -> dict[str, Any]:
    """Validate and summarize exactly one fresh confirmation cell."""

    records, optional_states = _validate_complete_packet(
        rows,
        seed=seed,
        bootstrap_resamples=bootstrap_resamples,
    )

    optional_fields = tuple(sorted(optional_states))
    measurement_values: dict[str, dict[str, list[float] | None]] = {}
    arms: dict[str, Any] = {}
    for arm in EXPECTED_ARMS:
        warmup = records[(arm, "warmup", 0, 0)]
        measurements = [
            records[(arm, "measurement", block_id, block_id - 1)]
            for block_id in MEASUREMENT_BLOCKS
        ]
        metric_values: dict[str, list[float] | None] = {
            metric: [float(record[metric]) for record in measurements]
            for metric in STAT_METRICS
        }
        for metric in optional_fields:
            metric_values[metric] = (
                [float(record[metric]) for record in measurements]
                if all(record[metric] is not None for record in measurements)
                else None
            )
        measurement_values[arm] = metric_values
        arms[arm] = {
            "warmup": _record_for_output(warmup, optional_fields),
            "measurement_blocks": list(MEASUREMENT_BLOCKS),
            "metrics": {
                metric: _stats(values) if values is not None else None
                for metric, values in metric_values.items()
            },
        }

    comparisons = {
        metric: _comparison(
            MEASUREMENT_BLOCKS,
            measurement_values["unfused"][metric],  # type: ignore[arg-type]
            measurement_values["fused"][metric],  # type: ignore[arg-type]
            metric=metric,
            seed=seed,
            bootstrap_resamples=bootstrap_resamples,
        )
        for metric in COMPARISON_METRICS
    }

    paired_differences: dict[str, Any] = {}
    for metric in SETUP_METRICS:
        paired_differences[metric] = _paired_difference(
            MEASUREMENT_BLOCKS,
            measurement_values["unfused"][metric],  # type: ignore[arg-type]
            measurement_values["fused"][metric],  # type: ignore[arg-type]
            seed=_seed_for(seed, metric),
            bootstrap_resamples=bootstrap_resamples,
        )
    for metric in optional_fields:
        if all(
            measurement_values[arm][metric] is not None for arm in EXPECTED_ARMS
        ):
            paired_differences[metric] = _paired_difference(
                MEASUREMENT_BLOCKS,
                measurement_values["unfused"][metric],  # type: ignore[arg-type]
                measurement_values["fused"][metric],  # type: ignore[arg-type]
                seed=_seed_for(seed, metric),
                bootstrap_resamples=bootstrap_resamples,
            )
        else:
            paired_differences[metric] = None

    primary = comparisons["session_inclusive_s"]
    reduction = primary["reduction_fraction"]
    cell_regression = (
        primary["fused_median"] / primary["unfused_median"] - 1.0
    )
    gates = {
        "minimum_session_inclusive_reduction": reduction >= MINIMUM_REDUCTION,
        "lower_95_bound_above_one": primary["lower_95_bound"] > 1.0,
        "maximum_cell_regression": cell_regression <= MAXIMUM_CELL_REGRESSION,
    }
    confirmation_pass = all(gates.values())

    cell = {
        "case_id": EXPECTED_CASE_ID,
        "dpu_count": EXPECTED_DPU_COUNT,
        "measurement_blocks": list(MEASUREMENT_BLOCKS),
        "arms": arms,
        "comparisons": comparisons,
        "setup_differences": {
            metric: paired_differences[metric] for metric in SETUP_METRICS
        },
        "paired_differences": paired_differences,
    }

    return {
        "analysis_version": ANALYSIS_VERSION,
        "seed": seed,
        "bootstrap_resamples": bootstrap_resamples,
        "case_id": EXPECTED_CASE_ID,
        "dpu_count": EXPECTED_DPU_COUNT,
        "topology": {"dpu_count": EXPECTED_DPU_COUNT},
        "arms": list(EXPECTED_ARMS),
        "attempts": {
            "warmup": {
                "count": len(WARMUP_BLOCKS),
                "block_ids": list(WARMUP_BLOCKS),
            },
            "measurement": {
                "count": len(MEASUREMENT_BLOCKS),
                "block_ids": list(MEASUREMENT_BLOCKS),
            },
        },
        "row_count": len(records),
        "measurement_blocks": list(MEASUREMENT_BLOCKS),
        "optional_timing_fields": {
            field: ("measured" if state == "value" else "unavailable")
            for field, state in optional_states.items()
        },
        "timing_semantics": {
            "session_inclusive_s": (
                "session_open_s + steady_s + session_close_s per sample before medians"
            ),
            "setup_s": "session_open_s + session_close_s per sample before summaries",
            "primary_speedup": "median(unfused) divided by median(fused)",
            "primary_bootstrap": "paired block indexes resampled before both arm medians",
            "warmup": "validated and retained for audit; excluded from measurement summaries",
            "aggregation": "single confirmation cell; no raw-runtime pooling",
        },
        "cells": [cell],
        "decision": {
            "result": "confirmation_pass" if confirmation_pass else "confirmation_fail",
            "confirmation_pass": confirmation_pass,
            "primary_metric": "session_inclusive_s",
            "unfused_median_s": primary["unfused_median"],
            "fused_median_s": primary["fused_median"],
            "ratio_of_arm_medians": primary["ratio_of_medians"],
            "reduction_fraction": reduction,
            "reduction_percent": 100.0 * reduction,
            "lower_95_bound": primary["lower_95_bound"],
            "cell_regression_fraction": cell_regression,
            "cell_regression_percent": 100.0 * cell_regression,
            "thresholds": {
                "minimum_reduction_fraction": MINIMUM_REDUCTION,
                "lower_95_bound_exclusive": 1.0,
                "maximum_cell_regression_fraction": MAXIMUM_CELL_REGRESSION,
            },
            "gates": gates,
            "production_adoption": False,
            "production_adoption_decision": "not_evaluated",
        },
        "claim_boundary": "confirmation_result_only_no_production_adoption",
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
