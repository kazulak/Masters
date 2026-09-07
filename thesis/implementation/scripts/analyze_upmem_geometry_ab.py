#!/usr/bin/env python3
"""Analyze the bounded, normalized UPMEM panel-versus-outer A/B packet.

This wrapper reuses the fusion A/B validator and paired statistics without
renaming geometry observations as fusion arms.  ``aggregate`` always covers
all six cells; ``target_region`` is the separately declared HS20 view.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping, Sequence
import json
from pathlib import Path
from typing import Any

if __package__:
    from . import analyze_upmem_fusion_ab as _shared
else:
    import analyze_upmem_fusion_ab as _shared


ANALYSIS_VERSION = "upmem_geometry_ab_analysis_v1"
DEFAULT_SEED = 20260909
DEFAULT_BOOTSTRAP_RESAMPLES = _shared.DEFAULT_BOOTSTRAP_RESAMPLES

EXPECTED_CASES = _shared.EXPECTED_CASES
EXPECTED_DPU_COUNTS = _shared.EXPECTED_DPU_COUNTS
EXPECTED_ARMS = ("panel", "outer")
EXPECTED_ATTEMPTS = _shared.EXPECTED_ATTEMPTS
WARMUP_BLOCKS = _shared.WARMUP_BLOCKS
MEASUREMENT_BLOCKS = _shared.MEASUREMENT_BLOCKS
IDENTITY_FIELDS = _shared.IDENTITY_FIELDS
TIME_FIELDS = _shared.TIME_FIELDS
REPORTED_METRICS = _shared.REPORTED_METRICS
SETUP_METRICS = _shared.SETUP_METRICS
ALL_DERIVED_METRICS = _shared.ALL_DERIVED_METRICS

TARGET_REGION_NAME = "hs20"
TARGET_REGION_CASES = ("hs_20q_d1",)
TARGET_REGION_DPU_COUNTS = EXPECTED_DPU_COUNTS

def _target_region(result: Mapping[str, Any], *, seed: int, bootstrap_resamples: int) -> dict[str, Any]:
    cells = [
        cell
        for cell in result["cells"]
        if cell["case_id"] in TARGET_REGION_CASES
        and cell["dpu_count"] in TARGET_REGION_DPU_COUNTS
    ]
    if len(cells) != len(TARGET_REGION_CASES) * len(TARGET_REGION_DPU_COUNTS):
        raise ValueError("target region must contain both HS20 DPU cells")
    return {
        "name": TARGET_REGION_NAME,
        "case_ids": list(TARGET_REGION_CASES),
        "dpu_counts": list(TARGET_REGION_DPU_COUNTS),
        "cell_count": len(cells),
        "metrics": {
            metric: _shared._aggregate_speedup(
                cells,
                metric,
                seed=seed,
                bootstrap_resamples=bootstrap_resamples,
                arm_labels=EXPECTED_ARMS,
            )
            for metric in REPORTED_METRICS
        },
    }


def analyze_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    seed: int = DEFAULT_SEED,
    bootstrap_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
) -> dict[str, Any]:
    """Validate and analyze the complete six-cell geometry A/B packet."""

    result = _shared.analyze_rows(
        rows,
        seed=seed,
        bootstrap_resamples=bootstrap_resamples,
        arm_labels=EXPECTED_ARMS,
        analysis_version=ANALYSIS_VERSION,
    )
    result["target_region"] = _target_region(
        result,
        seed=seed,
        bootstrap_resamples=bootstrap_resamples,
    )
    return result


def _read_rows(path: Path) -> list[Mapping[str, Any]]:
    return _shared._read_rows(path)


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
