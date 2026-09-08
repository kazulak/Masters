from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import sys

import pytest

from quantum_bench.upmem.path_heuristic import FEATURE_NAMES, RawFeatureVector


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "fit_upmem_wave_paths.py"
SPEC = importlib.util.spec_from_file_location("fit_upmem_wave_paths", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
fit = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = fit
SPEC.loader.exec_module(fit)


def _raw(
    host: float,
    local: float = 100.0,
    work: float = 100.0,
    sync: float = 100.0,
    numeric: float = 0.0,
    wram: float = 1.0,
) -> RawFeatureVector:
    return RawFeatureVector(
        host_dpu_bytes=host,
        mram_wram_bytes=local,
        dpu_work=work,
        sync_events=sync,
        numeric_overhead=numeric,
        wram_pressure=wram,
    )


def _cells(raw: dict[str, RawFeatureVector], greedy: str = "g") -> dict[str, object]:
    return {
        "cell": {
            "greedy_path_id": greedy,
            "raw_features": raw,
        }
    }


def _row(
    candidate: str,
    total: float,
    *,
    cell_id: str = "cell",
    round_id: int = 1,
    block: int = 1,
    split: str = "training",
    attempt_type: str = "measurement",
    **extra: object,
) -> dict[str, object]:
    return {
        "cell_id": cell_id,
        "candidate_path_id": candidate,
        "round_id": round_id,
        "block": block,
        "attempt_type": attempt_type,
        "session_open_s": 0.0,
        "total_wall_s": total,
        "session_close_s": 0.0,
        "split": split,
        "collection_resource_admission_passed": True,
        "execution_resource_admission_passed": True,
        "startup_resource_admission_passed": True,
        **extra,
    }


def _expected(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        {
            key: row[key]
            for key in ("cell_id", "candidate_path_id", "round_id", "block", "attempt_type", "split")
        }
        for row in rows
    ]


def test_paired_log_ratios_are_not_ratio_of_medians() -> None:
    rows = [
        _row("g", 1.0),
        _row("p", 2.0),
        _row("g", 100.0, block=2),
        _row("p", 10.0, block=2),
    ]
    result = fit.fit_upmem_wave_paths(
        _cells({"g": _raw(100.0), "p": _raw(50.0)}),
        rows,
        _expected(rows),
        seed=4,
        sample_count=8,
    )

    paired_geometric = math.sqrt((1.0 / 2.0) * (100.0 / 10.0))
    ratio_of_medians = (1.0 + 100.0) / (2.0 + 10.0)
    assert dict(result.cell_speedups)["cell"] == pytest.approx(paired_geometric)
    assert dict(result.cell_speedups)["cell"] != pytest.approx(ratio_of_medians)
    assert result.paired_observation_counts == (("cell", "p", 2),)


def test_round_local_greedy_drift_and_shorter_adaptive_round_set_are_retained() -> None:
    rows = [
        _row("g", 10.0, round_id=1),
        _row("p", 5.0, round_id=1),
        _row("q", 8.0, round_id=1),
        _row("g", 100.0, round_id=2),
        _row("p", 20.0, round_id=2),
    ]
    result = fit.fit_upmem_wave_paths(
        _cells({"g": _raw(100.0), "p": _raw(50.0), "q": _raw(80.0)}),
        rows,
        _expected(rows),
        sample_count=4,
    )

    assert dict(result.cell_speedups)["cell"] == pytest.approx(math.sqrt(10.0))
    assert result.paired_observation_counts == (("cell", "p", 2), ("cell", "q", 1))


def test_greedy_only_cell_is_retained_with_zero_competitor_pairs() -> None:
    cells = {
        "greedy-only": {
            "greedy_path_id": "g0",
            "raw_features": {"g0": _raw(100.0)},
        },
        "mixed": {
            "greedy_path_id": "g1",
            "raw_features": {"g1": _raw(100.0), "p1": _raw(50.0)},
        },
    }
    rows = [
        _row("g0", 3.0, cell_id="greedy-only"),
        _row("g1", 10.0, cell_id="mixed"),
        _row("p1", 5.0, cell_id="mixed"),
    ]
    result = fit.fit_upmem_wave_paths(cells, rows, _expected(rows), sample_count=3)

    assert dict(result.cell_speedups)["greedy-only"] == pytest.approx(1.0)
    assert dict(result.cell_speedups)["mixed"] == pytest.approx(2.0)
    assert ("greedy-only", "g0", 0) in result.paired_observation_counts
    assert ("mixed", "p1", 1) in result.paired_observation_counts
    assert result.selected_path_ids[0] == ("greedy-only", "g0")


def test_declared_row_set_is_authoritative() -> None:
    rows = [_row("g", 10.0), _row("p", 5.0)]
    cells = _cells({"g": _raw(100.0), "p": _raw(50.0)})
    with pytest.raises(ValueError, match="declared expected row set"):
        fit.fit_upmem_wave_paths(cells, rows, _expected(rows) + [_expected([_row("g", 4.0, block=2)])[0]])
    with pytest.raises(ValueError, match="declared expected row set"):
        fit.fit_upmem_wave_paths(cells, rows, _expected([rows[0]]))


def test_shared_path_id_cannot_supply_another_cells_measurements() -> None:
    cells = {
        name: {"greedy_path_id": "shared", "raw_features": {"shared": _raw(100.0)}}
        for name in ("d1", "d4")
    }
    rows = [
        _row("shared", 10.0, cell_id="d1"),
        _row("shared", 10.0, cell_id="d4", attempt_type="warmup"),
    ]
    with pytest.raises(ValueError, match="zero measured observations.*d4"):
        fit.fit_upmem_wave_paths(cells, rows, _expected(rows), sample_count=2)


def test_grouped_projection_gives_equal_half_movement_and_path_id_tie() -> None:
    rows = [_row("g", 10.0), _row("a", 5.0), _row("b", 5.0)]
    result = fit.fit_upmem_wave_paths(
        _cells({
            "g": _raw(4.0, local=4.0),
            "a": _raw(1.0, local=4.0),
            "b": _raw(4.0, local=1.0),
        }),
        rows,
        _expected(rows),
        model_form="grouped",
        seed=11,
        sample_count=5,
    )

    assert result.model.mode == "grouped"
    assert result.selected_path_ids == (("cell", "a"),)
    assert dict(result.weights.as_mapping())["E_num"] == 0.0
    assert dict(result.weights.as_mapping())["P_wram"] == 0.0


def test_simplex_search_is_seeded_and_transparent() -> None:
    rows = [_row("g", 10.0), _row("p", 5.0)]
    cells = _cells({"g": _raw(100.0), "p": _raw(50.0)})
    first = fit.fit_upmem_wave_paths(cells, rows, _expected(rows), seed=29, sample_count=7)
    second = fit.fit_upmem_wave_paths(cells, rows, _expected(rows), seed=29, sample_count=7)

    assert first == second
    assert first.seed == 29
    assert first.sample_count == 7
    assert first.evaluated_weight_vectors == 7 + 4 + 1
    assert first.objective[:2] == pytest.approx(
        (first.geometric_mean_speedup, first.worst_cell_speedup)
    )


def test_six_term_keeps_numeric_and_wram_weights_inactive() -> None:
    rows = [_row("g", 10.0), _row("p", 5.0)]
    result = fit.fit_upmem_wave_paths(
        _cells({
            "g": _raw(100.0, numeric=1.0, wram=1.0),
            "p": _raw(50.0, numeric=10_000.0, wram=10_000.0),
        }),
        rows,
        _expected(rows),
        seed=2,
        sample_count=9,
    )

    assert result.model.active_features == FEATURE_NAMES[:4]
    assert result.model.zero_range_features == ("E_num", "P_wram")
    assert result.weights.numeric == 0.0
    assert result.weights.wram == 0.0


def test_zero_open_close_are_valid_but_inclusive_total_must_be_positive() -> None:
    rows = [_row("g", 0.0), _row("p", 0.0)]
    with pytest.raises(ValueError, match="strictly positive"):
        fit.fit_upmem_wave_paths(
            _cells({"g": _raw(100.0), "p": _raw(50.0)}),
            rows,
            _expected(rows),
            sample_count=2,
        )

    valid = [_row("g", 1.0), _row("p", 1.0)]
    result = fit.fit_upmem_wave_paths(
        _cells({"g": _raw(100.0), "p": _raw(50.0, local=50.0, work=50.0, sync=50.0)}),
        valid,
        _expected(valid),
        sample_count=2,
    )
    assert result.selected_path_ids == (("cell", "p"),)

    for field, value in (("session_open_s", -1.0), ("total_wall_s", math.inf), ("session_close_s", math.nan)):
        invalid = [_row("g", 1.0), _row("p", 1.0)]
        invalid[1][field] = value
        with pytest.raises(ValueError, match="finite nonnegative"):
            fit.fit_upmem_wave_paths(
                _cells({"g": _raw(100.0), "p": _raw(50.0)}),
                invalid,
                _expected(invalid),
                sample_count=2,
            )


def test_failure_fallback_and_heldout_split_are_rejected_even_when_rows_are_caller_verified() -> None:
    cells = _cells({"g": _raw(100.0), "p": _raw(50.0)})
    for marker, value, message in (
        ("status", "failed", "failure"),
        ("fallback", True, "fallback"),
    ):
        rows = [_row("g", 10.0), _row("p", 5.0, **{marker: value})]
        with pytest.raises(ValueError, match=message):
            fit.fit_upmem_wave_paths(cells, rows, _expected(rows), sample_count=2)

    rows = [_row("g", 10.0, split="validation"), _row("p", 5.0, split="validation")]
    with pytest.raises(ValueError, match="training/development"):
        fit.fit_upmem_wave_paths(cells, rows, _expected(rows), split="training", sample_count=2)


def test_collection_admission_false_is_valid_diagnostic_input() -> None:
    rows = [
        _row("g", 10.0, collection_resource_admission_passed=False),
        _row("p", 5.0, collection_resource_admission_passed=False),
    ]
    result = fit.fit_upmem_wave_paths(
        _cells({"g": _raw(100.0), "p": _raw(50.0)}),
        rows,
        _expected(rows),
        sample_count=2,
    )
    assert result.selected_path_ids == (("cell", "p"),)


@pytest.mark.parametrize(
    "field",
    (
        "collection_resource_admission_passed",
        "execution_resource_admission_passed",
        "startup_resource_admission_passed",
    ),
)
def test_missing_resource_admission_flags_are_rejected(field: str) -> None:
    rows = [_row("g", 10.0), _row("p", 5.0)]
    del rows[1][field]
    with pytest.raises(ValueError, match=f"{field} must be a boolean"):
        fit.fit_upmem_wave_paths(
            _cells({"g": _raw(100.0), "p": _raw(50.0)}),
            rows,
            _expected(rows),
            sample_count=2,
        )


@pytest.mark.parametrize(
    "field",
    (
        "collection_resource_admission_passed",
        "execution_resource_admission_passed",
        "startup_resource_admission_passed",
    ),
)
def test_nonboolean_resource_admission_flags_are_rejected(field: str) -> None:
    rows = [_row("g", 10.0), _row("p", 5.0)]
    rows[1][field] = 1
    with pytest.raises(ValueError, match=f"{field} must be a boolean"):
        fit.fit_upmem_wave_paths(
            _cells({"g": _raw(100.0), "p": _raw(50.0)}),
            rows,
            _expected(rows),
            sample_count=2,
        )


@pytest.mark.parametrize(
    "field",
    (
        "execution_resource_admission_passed",
        "startup_resource_admission_passed",
    ),
)
def test_failed_hard_resource_admission_flags_are_rejected(field: str) -> None:
    rows = [_row("g", 10.0), _row("p", 5.0)]
    rows[1][field] = False
    with pytest.raises(ValueError, match=f"{field} must be exactly True"):
        fit.fit_upmem_wave_paths(
            _cells({"g": _raw(100.0), "p": _raw(50.0)}),
            rows,
            _expected(rows),
            sample_count=2,
        )


def test_unmeasured_candidate_cannot_win_and_missing_greedy_pair_fails() -> None:
    cells = _cells({"g": _raw(100.0), "measured": _raw(90.0), "unmeasured": _raw(1.0)})
    rows = [_row("g", 10.0), _row("measured", 9.0)]
    result = fit.fit_upmem_wave_paths(cells, rows, _expected(rows), sample_count=2)
    assert result.selected_path_ids == (("cell", "measured"),)
    assert all(candidate != "unmeasured" for candidate, _ in dict(result.candidate_speedups)["cell"])

    missing_greedy = [_row("measured", 9.0)]
    with pytest.raises(ValueError, match="greedy pair"):
        fit.fit_upmem_wave_paths(cells, missing_greedy, _expected(missing_greedy), sample_count=2)


@pytest.mark.parametrize(
    "interval,six_worst,grouped_worst,expected",
    [
        ((0.01, 0.1), 1.0, 1.0, "six_term"),
        ((0.01, 0.1), 1.1, 1.0, "six_term"),
        ((0.0, 0.1), 1.1, 1.0, "grouped"),
        ((-0.1, 0.1), 1.1, 1.0, "grouped"),
        ((-0.2, -0.1), 1.1, 1.0, "grouped"),
        ((0.01, 0.1), 0.9, 1.0, "grouped"),
    ],
)
def test_preregistered_model_choice(interval, six_worst, grouped_worst, expected):
    assert fit.select_wave_model_form(
        interval, six_term_worst_speedup=six_worst,
        grouped_worst_speedup=grouped_worst,
    ) == expected


@pytest.mark.parametrize("interval", [(1.0, 0.0), (math.nan, 1.0), (0.0, math.inf), (True, 1.0)])
def test_model_choice_rejects_invalid_uncertainty(interval):
    with pytest.raises(ValueError, match="finite bounds"):
        fit.select_wave_model_form(
            interval, six_term_worst_speedup=1.0, grouped_worst_speedup=1.0,
        )


def test_model_choice_rejects_zero_speedup():
    with pytest.raises(ValueError, match="strictly positive"):
        fit.select_wave_model_form(
            (0.01, 0.1), six_term_worst_speedup=0.0, grouped_worst_speedup=1.0,
        )
