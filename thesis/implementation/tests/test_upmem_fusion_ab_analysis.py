from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import random
from statistics import median
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "analyze_upmem_fusion_ab.py"
SPEC = importlib.util.spec_from_file_location("analyze_upmem_fusion_ab", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
analyzer = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = analyzer
SPEC.loader.exec_module(analyzer)


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
                for block_id in analyzer.WARMUP_BLOCKS:
                    rows.append(
                        _row(
                            case_id,
                            dpu_count,
                            arm,
                            "warmup",
                            block_id,
                            steady_s=20.0 if arm == "unfused" else 10.0,
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
                            steady_s=2.0 if arm == "unfused" else 1.0,
                        )
                    )
    return rows


def _cell(result: dict[str, object], case_id: str, dpu_count: int) -> dict[str, object]:
    return next(
        cell
        for cell in result["cells"]  # type: ignore[index]
        if cell["case_id"] == case_id and cell["dpu_count"] == dpu_count  # type: ignore[index]
    )


def test_complete_packet_has_six_cells_and_excludes_warmup_from_stats() -> None:
    result = analyzer.analyze_rows(_rows(), bootstrap_resamples=32)

    assert result["row_count"] == 72
    assert len(result["cells"]) == 6  # type: ignore[arg-type]
    cell = _cell(result, analyzer.EXPECTED_CASES[0], 1)
    unfused = cell["arms"]["unfused"]  # type: ignore[index]
    assert unfused["warmup"]["steady_s"] == 20.0  # type: ignore[index]
    assert unfused["metrics"]["steady_s"] == {  # type: ignore[index]
        "raw": [2.0] * 5,
        "count": 5,
        "median": 2.0,
        "raw_mad": 0.0,
        "min": 2.0,
        "max": 2.0,
    }
    assert cell["comparisons"]["steady_s"]["primary_speedup"] == 2.0  # type: ignore[index]


def test_session_inclusive_is_derived_per_row_before_median() -> None:
    rows = _rows()
    target = next(
        row
        for row in rows
        if row["case_id"] == analyzer.EXPECTED_CASES[0]
        and row["dpu_count"] == 1
        and row["arm"] == "unfused"
        and row["attempt_kind"] == "measurement"
    )
    block_id = 1
    for row in rows:
        if (
            row["case_id"] == target["case_id"]
            and row["dpu_count"] == target["dpu_count"]
            and row["arm"] == target["arm"]
            and row["attempt_kind"] == "measurement"
        ):
            row["session_open_s"] = [0.0, 0.0, 0.0, 0.0, 1.0][block_id - 1]
            row["session_close_s"] = [0.0, 0.0, 1.0, 1.0, 0.0][block_id - 1]
            block_id += 1

    result = analyzer.analyze_rows(rows, bootstrap_resamples=8)
    stats = _cell(result, analyzer.EXPECTED_CASES[0], 1)["arms"]["unfused"]["metrics"]  # type: ignore[index]
    assert stats["session_inclusive_s"]["raw"] == [2.0, 2.0, 3.0, 3.0, 3.0]
    assert stats["session_inclusive_s"]["median"] == 3.0
    assert stats["session_inclusive_s"]["median"] != (
        stats["session_open_s"]["median"]
        + stats["steady_s"]["median"]
        + stats["session_close_s"]["median"]
    )


def test_paired_speedup_uses_block_speedups_not_ratio_of_medians() -> None:
    rows = _rows()
    cell_rows = [
        row
        for row in rows
        if row["case_id"] == analyzer.EXPECTED_CASES[0]
        and row["dpu_count"] == 1
        and row["attempt_kind"] == "measurement"
    ]
    unfused = [1.0, 1.0, 1.0, 1.0, 2.0]
    fused = [1.0, 1.0, 2.0, 2.0, 2.0]
    for row in cell_rows:
        values = unfused if row["arm"] == "unfused" else fused
        row["steady_s"] = values[row["block_id"] - 1]
        row["kernel_s"] = row["steady_s"] / 2.0

    result = analyzer.analyze_rows(rows, bootstrap_resamples=16)
    comparison = _cell(result, analyzer.EXPECTED_CASES[0], 1)["comparisons"]["steady_s"]  # type: ignore[index]
    assert comparison["speedups"] == [1.0, 1.0, 0.5, 0.5, 1.0]
    assert comparison["primary_estimand"] == "ratio_of_arm_medians"
    assert comparison["ratio_of_medians"] == 0.5
    assert comparison["primary_speedup"] == 0.5
    assert comparison["median_of_block_speedups"] == 1.0
    bootstrap_seed = analyzer._seed_for(
        analyzer.DEFAULT_SEED,
        analyzer.EXPECTED_CASES[0],
        1,
        "steady_s",
        "ratio_of_arm_medians",
    )
    rng = random.Random(bootstrap_seed)
    primary_bootstrap = []
    for _ in range(16):
        indexes = [rng.randrange(5) for _ in range(5)]
        primary_bootstrap.append(
            median(unfused[index] for index in indexes)
            / median(fused[index] for index in indexes)
        )
    assert comparison["bootstrap_95_ci"] == pytest.approx(
        [
            analyzer._percentile(primary_bootstrap, 0.025),
            analyzer._percentile(primary_bootstrap, 0.975),
        ]
    )
    assert comparison["paired_saved_time_s"] == 0.0


def test_equal_cell_geometric_aggregate_does_not_pool_raw_runtimes() -> None:
    rows = _rows()
    for row in rows:
        if row["attempt_kind"] == "measurement":
            row["steady_s"] = 2.0 if row["arm"] == "unfused" else 1.0
            row["kernel_s"] = row["steady_s"] / 2.0
        if row["case_id"] == analyzer.EXPECTED_CASES[0] and row["dpu_count"] == 1:
            if row["attempt_kind"] == "measurement":
                row["steady_s"] = 100.0 if row["arm"] == "unfused" else 1.0
                row["kernel_s"] = row["steady_s"] / 2.0

    result = analyzer.analyze_rows(rows, bootstrap_resamples=16)
    aggregate = result["aggregate"]["metrics"]["steady_s"]  # type: ignore[index]
    assert aggregate["equal_cell_geometric_mean_speedup"] == pytest.approx(
        (100.0 * 2.0**5) ** (1.0 / 6.0)
    )
    assert len(aggregate["cell_speedups"]) == 6


def test_equal_cell_primary_aggregate_uses_ratio_of_arm_medians() -> None:
    rows = _rows()
    unfused = [1.0, 1.0, 1.0, 1.0, 2.0]
    fused = [1.0, 1.0, 2.0, 2.0, 2.0]
    for row in rows:
        if (
            row["case_id"] == analyzer.EXPECTED_CASES[0]
            and row["dpu_count"] == 1
            and row["attempt_kind"] == "measurement"
        ):
            values = unfused if row["arm"] == "unfused" else fused
            row["steady_s"] = values[row["block_id"] - 1]
            row["kernel_s"] = row["steady_s"] / 2.0

    result = analyzer.analyze_rows(rows, bootstrap_resamples=16)
    aggregate = result["aggregate"]["metrics"]["steady_s"]  # type: ignore[index]
    target = aggregate["cell_speedups"][0]
    assert target["ratio_of_arm_medians"] == 0.5
    assert target["median_of_block_speedups"] == 1.0
    assert aggregate["equal_cell_geometric_mean_speedup"] == pytest.approx(
        (0.5 * 2.0**5) ** (1.0 / 6.0)
    )


def test_missing_duplicate_and_bad_sample_pairs_are_rejected() -> None:
    rows = _rows()
    with pytest.raises(ValueError, match="exactly 72"):
        analyzer.analyze_rows(rows[:-1], bootstrap_resamples=4)

    duplicate = _rows()
    duplicate[-1] = dict(duplicate[0])
    with pytest.raises(ValueError, match="duplicate row"):
        analyzer.analyze_rows(duplicate, bootstrap_resamples=4)

    invalid = _rows()
    invalid[0]["sample_index"] = 1
    with pytest.raises(ValueError, match="sample/block"):
        analyzer.analyze_rows(invalid, bootstrap_resamples=4)


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
def test_nan_and_infinite_timing_values_are_rejected(bad_value: float) -> None:
    rows = _rows()
    rows[0]["steady_s"] = bad_value
    with pytest.raises(ValueError, match="finite"):
        analyzer.analyze_rows(rows, bootstrap_resamples=4)


def test_kernel_must_be_positive_and_not_exceed_steady() -> None:
    rows = _rows()
    rows[0]["kernel_s"] = 0.0
    with pytest.raises(ValueError, match="positive"):
        analyzer.analyze_rows(rows, bootstrap_resamples=4)

    rows = _rows()
    rows[0]["kernel_s"] = 2.1
    rows[0]["steady_s"] = 2.0
    with pytest.raises(ValueError, match="exceeds"):
        analyzer.analyze_rows(rows, bootstrap_resamples=4)


def test_optional_timing_columns_cannot_be_partially_zero_filled() -> None:
    rows = _rows()
    rows[0]["h2d_s"] = 0.0
    with pytest.raises(ValueError, match="optional timing"):
        analyzer.analyze_rows(rows, bootstrap_resamples=4)

    rows = _rows()
    for row in rows:
        row["h2d_s"] = None
    result = analyzer.analyze_rows(rows, bootstrap_resamples=4)
    assert result["cells"][0]["arms"]["unfused"]["metrics"]["h2d_s"] is None  # type: ignore[index]


def test_lead_optional_timing_contract_is_accepted_without_legacy_zero_fill() -> None:
    rows = _rows()
    measured = {
        "h2d_s": 0.01,
        "d2h_s": 0.02,
        "decode_s": 0.03,
        "encode_s": 0.04,
        "preparation_s": 0.05,
        "cohort_wall_sum_s": 0.06,
    }
    for row in rows:
        row.update(measured)
        row["request_build_s"] = None
        row["request_wave_s"] = None

    result = analyzer.analyze_rows(rows, bootstrap_resamples=4)
    metrics = result["cells"][0]["arms"]["unfused"]["metrics"]  # type: ignore[index]
    assert metrics["cohort_wall_sum_s"]["raw"] == [0.06] * 5
    assert metrics["request_wave_s"] is None
    paired = result["cells"][0]["paired_differences"]  # type: ignore[index]
    assert paired["cohort_wall_sum_s"]["median_difference_s"] == 0.0
    assert paired["request_wave_s"] is None
    assert result["optional_timing_fields"]["request_wave_s"] == "unavailable"  # type: ignore[index]
    assert result["optional_timing_fields"]["cohort_wall_sum_s"] == "measured"  # type: ignore[index]
    assert result["timing_semantics"]["session_inclusive_s"].startswith(  # type: ignore[index]
        "session_open_s + steady_s"
    )


def test_bootstrap_is_deterministic_and_changes_with_seed() -> None:
    rows = _rows()
    for row in rows:
        if row["attempt_kind"] == "measurement":
            block_id = row["block_id"]
            row["steady_s"] = (
                (2.0 + 0.1 * block_id)
                if row["arm"] == "unfused"
                else (1.0 + 0.03 * block_id)
            )
            row["kernel_s"] = row["steady_s"] / 2.0
    first = analyzer.analyze_rows(rows, seed=17, bootstrap_resamples=64)
    second = analyzer.analyze_rows(list(reversed(rows)), seed=17, bootstrap_resamples=64)
    third = analyzer.analyze_rows(rows, seed=18, bootstrap_resamples=64)
    assert first == second
    assert first["aggregate"] == second["aggregate"]
    assert first != third


def test_zero_speedup_denominator_is_rejected_by_pair_helper() -> None:
    with pytest.raises(ValueError, match="denominator"):
        analyzer._paired_comparison(
            [1], [1.0], [0.0], seed=1, bootstrap_resamples=4
        )


def test_cli_reads_json_list_and_writes_json(tmp_path: Path) -> None:
    input_path = tmp_path / "rows.json"
    output_path = tmp_path / "analysis.json"
    input_path.write_text(json.dumps(_rows()), encoding="utf-8")
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
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
    )
    assert completed.stdout == ""
    output = json.loads(output_path.read_text(encoding="utf-8"))
    assert output["analysis_version"] == analyzer.ANALYSIS_VERSION
    assert output["bootstrap_resamples"] == 4
