#!/usr/bin/env python3
"""Paired fixed-resource serial/static DAG analysis; not formal scaling claims."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

if __package__:
    from . import analyze_upmem_fusion_ab as _shared
else:
    import analyze_upmem_fusion_ab as _shared

ANALYSIS_VERSION = "upmem_dag_ab_analysis_v1"
EXPECTED_ARMS = ("serial", "dag")
EXPECTED_DPU_COUNTS = (2, 4)
DEFAULT_SEED = 20260909


def analyze_rows(rows, *, seed=DEFAULT_SEED, bootstrap_resamples=10000):
    return _shared.analyze_rows(rows, seed=seed, bootstrap_resamples=bootstrap_resamples,
        arm_labels=EXPECTED_ARMS, dpu_counts=EXPECTED_DPU_COUNTS,
        analysis_version=ANALYSIS_VERSION)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    args = parser.parse_args(argv)
    result = analyze_rows(_shared._read_rows(args.input), seed=args.seed,
                          bootstrap_resamples=args.bootstrap_resamples)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
