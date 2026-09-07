from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import random
import shutil
from statistics import median
import subprocess
import sys

import pytest

from scripts import analyze_upmem_geometry_ab as analyzer
from scripts import analyze_upmem_fusion_ab as shared


ROOT = Path(__file__).resolve().parents[1]
SHARED_SCRIPT = ROOT / "scripts" / "analyze_upmem_fusion_ab.py"
SCRIPT = ROOT / "scripts" / "analyze_upmem_geometry_ab.py"


def _row(
    case_id: str,
    dpu_count: int,
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
        "case_id": case_id,
        "dpu_count": dpu_count,
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
    for case_id in analyzer.EXPECTED_CASES:
        for dpu_count in analyzer.EXPECTED_DPU_COUNTS:
            for arm in analyzer.EXPECTED_ARMS:
                rows.append(
                    _row(
                        case_id,
                        dpu_count,
                        arm,
                        "warmup",
                        0,
                        steady_s=20.0 if arm == "panel" else 10.0,
                    )
                )
                for block_id in analyzer.MEASUREMENT_BLOCKS:
                    rows.append(
                        _row(
                            case_id,
                            dpu_count,
                            arm,
                            "measurement",
                            block_id,
                            steady_s=2.0 if arm == "panel" else 1.0,
                        )
                    )
    return rows


def _cell(result: dict[str, object], case_id: str, dpu_count: int) -> dict[str, object]:
    return next(
        cell
        for cell in result["cells"]  # type: ignore[index]
        if cell["case_id"] == case_id and cell["dpu_count"] == dpu_count  # type: ignore[index]
    )


def test_geometry_packet_has_explicit_arms_and_both_aggregate_views() -> None:
    result = analyzer.analyze_rows(_rows(), bootstrap_resamples=32)

    assert result["analysis_version"] == "upmem_geometry_ab_analysis_v1"
    assert result["seed"] == 20260909
    assert result["row_count"] == 72
    assert result["arms"] == ["panel", "outer"]
    assert result["aggregate"]["cell_count"] == 6  # type: ignore[index]
    target = result["target_region"]  # type: ignore[index]
    assert target["name"] == "hs20"
    assert target["case_ids"] == ["hs_20q_d1"]
    assert target["dpu_counts"] == [1, 4]
    assert target["cell_count"] == 2
    assert len(target["metrics"]["steady_s"]["cell_speedups"]) == 2

    encoded = json.dumps(result)
    assert "unfused" not in encoded
    assert "fused" not in encoded
    assert result["timing_semantics"]["primary_speedup"] == (  # type: ignore[index]
        "median(panel) divided by median(outer) within each cell"
    )


def test_missing_and_duplicate_geometry_rows_are_rejected() -> None:
    rows = _rows()
    with pytest.raises(ValueError, match="exactly 72"):
        analyzer.analyze_rows(rows[:-1], bootstrap_resamples=4)

    duplicate = _rows()
    duplicate[-1] = dict(duplicate[0])
    with pytest.raises(ValueError, match="duplicate row"):
        analyzer.analyze_rows(duplicate, bootstrap_resamples=4)


@pytest.mark.parametrize("field", ["dpu_count", "block_id", "sample_index"])
def test_boolean_identity_values_are_rejected(field: str) -> None:
    rows = _rows()
    rows[0][field] = True
    with pytest.raises(ValueError, match="integer"):
        analyzer.analyze_rows(rows, bootstrap_resamples=4)


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_geometry_timing_values_are_rejected(bad_value: float) -> None:
    rows = _rows()
    rows[0]["steady_s"] = bad_value
    with pytest.raises(ValueError, match="finite"):
        analyzer.analyze_rows(rows, bootstrap_resamples=4)


def test_boolean_timing_values_are_rejected() -> None:
    rows = _rows()
    rows[0]["steady_s"] = True
    with pytest.raises(ValueError, match="finite non-negative number"):
        analyzer.analyze_rows(rows, bootstrap_resamples=4)


def test_session_inclusive_is_derived_per_sample_before_medians() -> None:
    rows = _rows()
    block_values = [0.0, 0.0, 0.0, 0.0, 1.0]
    close_values = [0.0, 0.0, 1.0, 1.0, 0.0]
    for row in rows:
        if (
            row["case_id"] == analyzer.EXPECTED_CASES[0]
            and row["dpu_count"] == 1
            and row["arm"] == "panel"
            and row["attempt_kind"] == "measurement"
        ):
            block_id = row["block_id"]
            row["session_open_s"] = block_values[block_id - 1]  # type: ignore[operator]
            row["session_close_s"] = close_values[block_id - 1]  # type: ignore[operator]

    result = analyzer.analyze_rows(rows, bootstrap_resamples=8)
    stats = _cell(result, analyzer.EXPECTED_CASES[0], 1)["arms"]["panel"][  # type: ignore[index]
        "metrics"
    ]
    assert stats["session_inclusive_s"]["raw"] == [2.0, 2.0, 3.0, 3.0, 3.0]
    assert stats["session_inclusive_s"]["median"] == 3.0
    assert stats["session_inclusive_s"]["median"] != (
        stats["session_open_s"]["median"]
        + stats["steady_s"]["median"]
        + stats["session_close_s"]["median"]
    )


def test_paired_bootstrap_keeps_panel_outer_block_covariance_and_is_deterministic() -> None:
    rows = _rows()
    panel = [1.0, 1.0, 1.0, 1.0, 2.0]
    outer = [1.0, 1.0, 2.0, 2.0, 2.0]
    for row in rows:
        if (
            row["case_id"] == analyzer.EXPECTED_CASES[0]
            and row["dpu_count"] == 1
            and row["attempt_kind"] == "measurement"
        ):
            values = panel if row["arm"] == "panel" else outer
            row["steady_s"] = values[row["block_id"] - 1]  # type: ignore[operator]
            row["kernel_s"] = row["steady_s"] / 2.0  # type: ignore[operator]

    first = analyzer.analyze_rows(rows, seed=17, bootstrap_resamples=64)
    second = analyzer.analyze_rows(list(reversed(rows)), seed=17, bootstrap_resamples=64)
    third = analyzer.analyze_rows(rows, seed=18, bootstrap_resamples=64)
    assert first == second
    assert first != third

    comparison = _cell(first, analyzer.EXPECTED_CASES[0], 1)["comparisons"][  # type: ignore[index]
        "steady_s"
    ]
    assert comparison["speedups"] == [1.0, 1.0, 0.5, 0.5, 1.0]
    assert comparison["ratio_of_medians"] == 0.5

    seed = shared._seed_for(
        17,
        analyzer.EXPECTED_CASES[0],
        1,
        "steady_s",
        "ratio_of_arm_medians",
    )
    rng = random.Random(seed)
    paired_bootstrap: list[float] = []
    for _ in range(64):
        indexes = [rng.randrange(5) for _ in range(5)]
        paired_bootstrap.append(
            median(panel[index] for index in indexes)
            / median(outer[index] for index in indexes)
        )
    assert comparison["bootstrap_95_ci"] == pytest.approx(
        [
            shared._percentile(paired_bootstrap, 0.025),
            shared._percentile(paired_bootstrap, 0.975),
        ]
    )


def test_geometry_analysis_does_not_mutate_input_rows() -> None:
    rows = _rows()
    original = copy.deepcopy(rows)
    analyzer.analyze_rows(rows, bootstrap_resamples=8)
    assert rows == original


def test_target_region_cannot_hide_a_regression_in_all_six_cells() -> None:
    rows = _rows()
    for row in rows:
        if row["attempt_kind"] != "measurement":
            continue
        row["session_open_s"] = 0.0
        row["session_close_s"] = 0.0
        if row["case_id"] == "hs_20q_d1":
            row["steady_s"] = 2.0 if row["arm"] == "panel" else 1.0
        elif row["case_id"] == "quantization_stress_16q_l2":
            row["steady_s"] = 1.0 if row["arm"] == "panel" else 4.0
        else:
            row["steady_s"] = 2.0 if row["arm"] == "panel" else 1.0
        row["kernel_s"] = row["steady_s"] / 2.0  # type: ignore[operator]

    result = analyzer.analyze_rows(rows, bootstrap_resamples=16)
    all_six = result["aggregate"]["metrics"]["steady_s"]  # type: ignore[index]
    target = result["target_region"]["metrics"]["steady_s"]  # type: ignore[index]
    assert len(all_six["cell_speedups"]) == 6
    assert all_six["equal_cell_geometric_mean_speedup"] == pytest.approx(1.0)
    assert len(target["cell_speedups"]) == 2
    assert target["equal_cell_geometric_mean_speedup"] == pytest.approx(2.0)


def test_standalone_cli_works_with_only_adjacent_packet_scripts(tmp_path: Path) -> None:
    packet_root = tmp_path / "packet"
    packet_root.mkdir()
    geometry_script = packet_root / "analyze_upmem_geometry_ab.py"
    fusion_script = packet_root / "analyze_upmem_fusion_ab.py"
    shutil.copyfile(SCRIPT, geometry_script)
    shutil.copyfile(SHARED_SCRIPT, fusion_script)
    input_path = tmp_path / "rows.json"
    output_path = tmp_path / "analysis.json"
    input_path.write_text(json.dumps(_rows()), encoding="utf-8")

    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    completed = subprocess.run(
        [
            sys.executable,
            str(geometry_script),
            str(input_path),
            "--output",
            str(output_path),
            "--bootstrap-resamples",
            "4",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    assert completed.stdout == ""
    result = json.loads(output_path.read_text(encoding="utf-8"))
    assert result["analysis_version"] == "upmem_geometry_ab_analysis_v1"
    assert result["target_region"]["cell_count"] == 2
