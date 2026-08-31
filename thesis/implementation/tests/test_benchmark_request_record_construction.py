from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "benchmark_request_record_construction.py"
SPEC = importlib.util.spec_from_file_location("benchmark_request_record_construction", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
benchmark = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = benchmark
SPEC.loader.exec_module(benchmark)


def _attribution(path: Path) -> None:
    rows = [
        {
            "case_id": case_id,
            "dpu_count": dpu_count,
            "median_attempt_elapsed_s": 10.0,
            "raw_mad_attempt_elapsed_s": 0.1,
            "median_total_wall_s": 9.0,
            "raw_mad_total_wall_s": 0.1,
            "median_payload_record_construction_s": 2.0,
        }
        for case_id, dpu_count in benchmark.CELL_DEFINITIONS
    ]
    path.write_text(json.dumps({"measurement_cells": rows}), encoding="utf-8")


def test_host_only_benchmark_produces_gate_and_deterministic_factors(tmp_path: Path) -> None:
    attribution = tmp_path / "attribution.json"
    _attribution(attribution)
    output = tmp_path / "output"

    result = benchmark.analyze(output, attribution, repeats=3)

    assert result["go_no_go"]["decision"] in {"go", "no_go"}
    assert len(result["cells"]) == 6
    assert len(result["record_count_sweep"]) == len(benchmark.COUNT_POINTS)
    assert len(result["isolated_factors"]) == len(benchmark.PATH_LENGTH_POINTS)
    assert (output / "request_record_cost.json").is_file()
    assert (output / "request_record_cost.csv").is_file()
    assert (output / "request_template_go_no_go.json").is_file()


def test_template_fill_preserves_serialized_record_values(tmp_path: Path) -> None:
    artifact = benchmark._build(tmp_path / "request", dpu_count=4, sequence=0)

    elapsed, records = benchmark._fill_template(artifact.work_units)

    assert elapsed >= 0.0
    assert records == artifact.work_units
