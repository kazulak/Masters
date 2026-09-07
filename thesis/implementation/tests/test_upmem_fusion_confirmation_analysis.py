from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "analyze_upmem_fusion_confirmation.py"
SPEC = importlib.util.spec_from_file_location(
    "analyze_upmem_fusion_confirmation", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
analyzer = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = analyzer
SPEC.loader.exec_module(analyzer)


def _row(
    arm: str,
    attempt_kind: str,
    block_id: int,
    *,
    steady_s: float,
    session_open_s: float = 0.1,
    session_close_s: float = 0.2,
    kernel_s: float | None = None,
) -> dict[str, object]:
    return {
        "case_id": analyzer.EXPECTED_CASE_ID,
        "dpu_count": analyzer.EXPECTED_DPU_COUNT,
        "arm": arm,
        "attempt_kind": attempt_kind,
        "block_id": block_id,
        "sample_index": 0 if attempt_kind == "warmup" else block_id - 1,
        "steady_s": steady_s,
        "session_open_s": session_open_s,
        "session_close_s": session_close_s,
        "kernel_s": steady_s / 2.0 if kernel_s is None else kernel_s,
    }


def _rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for arm in analyzer.EXPECTED_ARMS:
        rows.append(
            _row(
                arm,
                "warmup",
                0,
                steady_s=20.0 if arm == "unfused" else 10.0,
            )
        )
        for block_id in analyzer.MEASUREMENT_BLOCKS:
            rows.append(
                _row(
                    arm,
                    "measurement",
                    block_id,
                    steady_s=2.0 if arm == "unfused" else 1.0,
                )
            )
    return rows


def test_confirmation_is_one_cell_and_excludes_warmup() -> None:
    result = analyzer.analyze_rows(_rows(), bootstrap_resamples=32)

    assert result["row_count"] == 12
    assert result["case_id"] == analyzer.EXPECTED_CASE_ID
    assert result["dpu_count"] == 4
    cell = result["cells"][0]  # type: ignore[index]
    assert cell["arms"]["unfused"]["warmup"]["steady_s"] == 20.0  # type: ignore[index]
    assert cell["arms"]["unfused"]["metrics"]["steady_s"] == {  # type: ignore[index]
        "raw": [2.0] * 5,
        "count": 5,
        "median": 2.0,
        "raw_mad": 0.0,
        "min": 2.0,
        "max": 2.0,
    }


def test_session_inclusive_uses_nonlinear_per_sample_medians() -> None:
    rows = _rows()
    measurements = [
        row
        for row in rows
        if row["arm"] == "unfused" and row["attempt_kind"] == "measurement"
    ]
    opens = [0.0, 0.0, 0.0, 0.0, 1.0]
    closes = [0.0, 0.0, 1.0, 1.0, 0.0]
    for row, open_s, close_s in zip(measurements, opens, closes, strict=True):
        row["session_open_s"] = open_s
        row["session_close_s"] = close_s

    result = analyzer.analyze_rows(rows, bootstrap_resamples=8)
    metrics = result["cells"][0]["arms"]["unfused"]["metrics"]  # type: ignore[index]
    assert metrics["session_inclusive_s"]["raw"] == [2.0, 2.0, 3.0, 3.0, 3.0]
    assert metrics["session_inclusive_s"]["median"] == 3.0
    assert metrics["session_inclusive_s"]["median"] != (
        metrics["session_open_s"]["median"]
        + metrics["steady_s"]["median"]
        + metrics["session_close_s"]["median"]
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda rows: rows.pop(), "exactly 12"),
        (lambda rows: rows.__setitem__(-1, dict(rows[0])), "duplicate row"),
        (
            lambda rows: rows[0].__setitem__("case_id", "hs_20q_d1"),
            "wrong case_id",
        ),
        (
            lambda rows: rows[0].__setitem__("dpu_count", 1),
            "wrong topology",
        ),
        (
            lambda rows: rows[0].__setitem__("block_id", 2),
            "sample/block",
        ),
        (
            lambda rows: rows[2].__setitem__("sample_index", 3),
            "sample/block",
        ),
    ],
)
def test_identity_topology_block_and_completeness_checks(mutation, message) -> None:
    rows = _rows()
    mutation(rows)
    with pytest.raises(ValueError, match=message):
        analyzer.analyze_rows(rows, bootstrap_resamples=4)


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_required_timer_is_rejected(bad_value: float) -> None:
    rows = _rows()
    rows[0]["steady_s"] = bad_value
    with pytest.raises(ValueError, match="finite"):
        analyzer.analyze_rows(rows, bootstrap_resamples=4)


def test_optional_measured_and_null_timers_are_preserved() -> None:
    measured = _rows()
    for row in measured:
        row["cohort_wall_sum_s"] = 0.06
        row["request_wave_s"] = None
    result = analyzer.analyze_rows(measured, bootstrap_resamples=4)
    metrics = result["cells"][0]["arms"]["unfused"]["metrics"]  # type: ignore[index]
    assert metrics["cohort_wall_sum_s"]["raw"] == [0.06] * 5
    assert metrics["request_wave_s"] is None
    assert result["optional_timing_fields"] == {  # type: ignore[comparison-overlap]
        "cohort_wall_sum_s": "measured",
        "request_wave_s": "unavailable",
    }


def test_bootstrap_is_deterministic_and_uses_confirmation_defaults() -> None:
    rows = _rows()
    first = analyzer.analyze_rows(rows, bootstrap_resamples=128)
    second = analyzer.analyze_rows(list(reversed(rows)), bootstrap_resamples=128)

    assert first == second
    assert first["seed"] == 20260908
    assert (
        first["cells"][0]["comparisons"]["session_inclusive_s"][  # type: ignore[index]
            "bootstrap_resamples"
        ]
        == 128
    )
    assert first["decision"]["production_adoption"] is False  # type: ignore[index]
    assert first["claim_boundary"] == "confirmation_result_only_no_production_adoption"


def test_cli_reads_json_list_and_writes_json(tmp_path: Path) -> None:
    input_path = tmp_path / "rows.json"
    output_path = tmp_path / "analysis.json"
    input_path.write_text(json.dumps(_rows()), encoding="utf-8")
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.analyze_upmem_fusion_confirmation",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--bootstrap-resamples",
            "4",
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT)},
    )
    assert completed.stdout == ""
    output = json.loads(output_path.read_text(encoding="utf-8"))
    assert output["analysis_version"] == analyzer.ANALYSIS_VERSION
    assert output["bootstrap_resamples"] == 4
