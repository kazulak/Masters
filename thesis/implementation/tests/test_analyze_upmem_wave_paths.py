from __future__ import annotations

from copy import deepcopy
import importlib.util
import math
from pathlib import Path
import random
import sys

import pytest

from quantum_bench.upmem.path_heuristic import RawFeatureVector, WeightVector


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


fit = _load("fit_upmem_wave_paths", ROOT / "scripts" / "fit_upmem_wave_paths.py")
analysis = _load(
    "analyze_upmem_wave_paths",
    ROOT / "scripts" / "analyze_upmem_wave_paths.py",
)


def _raw(host: float, local: float = 100.0) -> RawFeatureVector:
    return RawFeatureVector(
        host_dpu_bytes=host,
        mram_wram_bytes=local,
        dpu_work=100.0,
        sync_events=100.0,
        numeric_overhead=0.0,
        wram_pressure=1.0,
    )


def _cells(raw: dict[str, RawFeatureVector], cell_id: str = "cell") -> dict[str, object]:
    return {
        cell_id: {
            "greedy_path_id": "g",
            "raw_features": raw,
        }
    }


def _profile(
    profile_id: str,
    selected: dict[str, str],
) -> analysis.FixedWaveProfile:
    return analysis.FixedWaveProfile(
        profile_id=profile_id,
        weights=WeightVector.from_values(
            (1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            inactive=("E_num", "P_wram"),
        ),
        model=fit._model("six_term"),
        selected_path_ids=tuple(sorted(selected.items())),
    )


def _rows(
    values: dict[str, tuple[float, ...]],
    *,
    cell_id: str = "cell",
    round_id: str = "round-1",
    split: str = "training",
    status: str | None = None,
) -> list[dict[str, object]]:
    rows = []
    for candidate_id, timings in values.items():
        for block, total in enumerate(timings, start=1):
            row: dict[str, object] = {
                "cell_id": cell_id,
                "candidate_path_id": candidate_id,
                "round_id": round_id,
                "block": block,
                "attempt_type": "measurement",
                "session_open_s": 0.0,
                "total_wall_s": total,
                "session_close_s": 0.0,
                "split": split,
            }
            if status is not None:
                row["status"] = status
            rows.append(row)
    return rows


def _expected(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    fields = (
        "cell_id",
        "candidate_path_id",
        "round_id",
        "block",
        "attempt_type",
        "split",
    )
    return [{field: row[field] for field in fields} for row in rows]


def _run(
    cells: dict[str, object],
    rows: list[dict[str, object]],
    profiles: tuple[analysis.FixedWaveProfile, analysis.FixedWaveProfile],
    *,
    resamples: int = 25,
):
    return analysis.analyze_upmem_wave_paths(
        cells,
        rows,
        _expected(rows),
        profiles,
        bootstrap_resamples=resamples,
    )


def _cell(result: dict[str, object], profile_id: str = "a") -> dict[str, object]:
    profile = next(
        item for item in result["profile_results"] if item["profile_id"] == profile_id
    )
    return profile["cells"][0]


def test_tie_aware_ranking_and_raw_median_mad_facts() -> None:
    raw = {
        "g": _raw(100.0),
        "a": _raw(50.0),
        "b": _raw(50.0),
        "c": _raw(25.0),
    }
    rows = _rows(
        {
            "g": (10.0, 11.0, 9.0),
            "a": (5.0, 6.0, 4.0),
            "b": (8.0, 8.0, 8.0),
            "c": (2.0, 3.0, 1.0),
        }
    )
    result = _run(
        _cells(raw),
        rows,
        (_profile("a", {"cell": "a"}), _profile("b", {"cell": "c"})),
    )
    cell = _cell(result)
    assert analysis._average_ranks((1.0, 2.0, 2.0, 4.0)) == (1.0, 2.5, 2.5, 4.0)
    assert cell["selected_rank"] == 2
    assert cell["top1"] is False
    assert cell["top3"] is True
    assert cell["oracle_path_id"] == "c"
    assert cell["oracle_regret"] == pytest.approx(20.0 ** (1.0 / 3.0))
    selected_time = (0.5 * (6.0 / 11.0) * (4.0 / 9.0)) ** (1.0 / 3.0)
    oracle_time = (0.2 * (3.0 / 11.0) * (1.0 / 9.0)) ** (1.0 / 3.0)
    assert cell["headroom"] == pytest.approx(
        (1.0 - selected_time) / (1.0 - oracle_time)
    )
    candidate_a = next(
        item for item in cell["candidates"] if item["candidate_path_id"] == "a"
    )
    assert candidate_a["raw_runtime_median_s"] == pytest.approx(5.0)
    assert candidate_a["raw_runtime_mad_s"] == pytest.approx(1.0)
    scores = [item["score"] for item in cell["candidates"]]
    paired_times = [item["paired_normalized_time"] for item in cell["candidates"]]
    assert cell["score_runtime_spearman"] == pytest.approx(
        analysis._spearman(scores, paired_times)
    )


def test_greedy_oracle_has_null_headroom() -> None:
    rows = _rows({"g": (10.0, 10.0, 10.0), "p": (10.0, 10.0, 10.0)})
    result = _run(
        _cells({"g": _raw(100.0), "p": _raw(50.0)}),
        rows,
        (_profile("a", {"cell": "p"}), _profile("b", {"cell": "g"})),
    )
    cell = _cell(result)
    assert cell["oracle_path_id"] == "g"
    assert cell["oracle_regret"] == pytest.approx(1.0)
    assert cell["headroom"] is None
    assert cell["headroom_reason"] == "no_positive_measurable_headroom"


def test_primary_rank_uses_paired_round_normalization_not_pooled_raw_medians() -> None:
    cells = _cells({"g": _raw(100.0), "p": _raw(50.0)})
    rows = _rows(
        {"g": (10.0, 10.0, 10.0), "p": (1.0, 1.0, 1.0)},
        round_id="round-1",
    ) + _rows(
        {"g": (100.0, 100.0, 100.0), "p": (200.0, 200.0, 200.0)},
        round_id="round-2",
    )
    result = _run(cells, rows, (_profile("a", {"cell": "p"}), _profile("b", {"cell": "g"})))
    cell = _cell(result)
    assert cell["selected_raw_runtime_median_s"] == pytest.approx(100.5)
    assert cell["selected_paired_normalized_time"] == pytest.approx(math.sqrt(0.2))
    assert cell["selected_rank"] == 1
    assert cell["oracle_path_id"] == "p"


def test_known_selected_slowdown_is_reported() -> None:
    rows = _rows({"g": (10.0, 10.0, 10.0), "slow": (20.0, 20.0, 20.0)})
    result = _run(
        _cells({"g": _raw(100.0), "slow": _raw(50.0)}),
        rows,
        (_profile("a", {"cell": "slow"}), _profile("b", {"cell": "g"})),
    )
    cell = _cell(result)
    assert cell["selected_speedup_vs_greedy"] == pytest.approx(0.5)
    assert cell["oracle_regret"] == pytest.approx(2.0)
    assert cell["headroom"] is None


def test_paired_bootstrap_rejects_unpaired_variation_and_is_deterministic() -> None:
    rows = _rows({"g": (1.0, 10.0, 100.0), "p": (2.0, 20.0, 200.0)})
    cells = _cells({"g": _raw(100.0), "p": _raw(50.0)})
    profiles = (_profile("a", {"cell": "p"}), _profile("b", {"cell": "g"}))
    first = _run(cells, rows, profiles, resamples=40)
    second = _run(cells, rows, profiles, resamples=40)
    assert first == second
    bootstrap = first["comparison"]["bootstrap"]
    expected = -math.log(2.0)
    assert bootstrap["geometric_mean_log_difference"] == pytest.approx(expected)
    assert bootstrap["geometric_mean_log_difference_95pct"] == pytest.approx(
        (expected, expected)
    )


def test_bootstrap_shares_one_block_draw_across_cells_in_a_round() -> None:
    cells = {
        "cell-a": {"greedy_path_id": "g", "raw_features": {"g": _raw(100.0), "p": _raw(50.0)}},
        "cell-b": {"greedy_path_id": "g", "raw_features": {"g": _raw(100.0), "p": _raw(50.0)}},
    }
    rows = _rows({"g": (1.0, 1.0, 1.0), "p": (1.0, 2.0, 4.0)}, cell_id="cell-a")
    rows += _rows({"g": (1.0, 1.0, 1.0), "p": (1.0, 0.5, 0.25)}, cell_id="cell-b")
    profiles = (
        _profile("a", {"cell-a": "p", "cell-b": "p"}),
        _profile("b", {"cell-a": "g", "cell-b": "g"}),
    )
    result = _run(cells, rows, profiles, resamples=30)
    interval = result["comparison"]["bootstrap"]["geometric_mean_log_difference_95pct"]
    assert interval == pytest.approx((0.0, 0.0))


def test_missing_block_failure_and_heldout_rows_are_rejected() -> None:
    complete = _rows({"g": (10.0, 10.0, 10.0), "p": (5.0, 5.0, 5.0)})
    incomplete = [row for row in complete if not (row["candidate_path_id"] == "p" and row["block"] == 2)]
    with pytest.raises(ValueError, match="missing measurement block"):
        analysis.analyze_upmem_wave_paths(
            _cells({"g": _raw(100.0), "p": _raw(50.0)}),
            incomplete,
            _expected(incomplete),
            (_profile("a", {"cell": "p"}), _profile("b", {"cell": "g"})),
            bootstrap_resamples=2,
        )

    failed = _rows(
        {"g": (10.0, 10.0, 10.0), "p": (5.0, 5.0, 5.0)},
        status="failed",
    )
    with pytest.raises(ValueError, match="failure"):
        _run(
            _cells({"g": _raw(100.0), "p": _raw(50.0)}),
            failed,
            (_profile("a", {"cell": "p"}), _profile("b", {"cell": "g"})),
            resamples=2,
        )

    heldout = _rows(
        {"g": (10.0, 10.0, 10.0), "p": (5.0, 5.0, 5.0)},
        split="validation",
    )
    with pytest.raises(ValueError, match="training/development"):
        analysis.analyze_upmem_wave_paths(
            _cells({"g": _raw(100.0), "p": _raw(50.0)}),
            heldout,
            _expected(heldout),
            (_profile("a", {"cell": "p"}), _profile("b", {"cell": "g"})),
            bootstrap_resamples=2,
        )


def test_profile_selection_duplicates_invalid_model_and_inputs_remain_immutable() -> None:
    cells = _cells({"g": _raw(100.0), "p": _raw(50.0)})
    rows = _rows({"g": (10.0, 10.0, 10.0), "p": (5.0, 5.0, 5.0)})
    original_cells = deepcopy(cells)
    original_rows = deepcopy(rows)
    duplicate = analysis.FixedWaveProfile(
        profile_id="duplicate",
        weights=_profile("a", {"cell": "p"}).weights,
        model=fit._model("six_term"),
        selected_path_ids=(("cell", "p"), ("cell", "g")),
    )
    with pytest.raises(ValueError, match="duplicate cell"):
        _run(cells, rows, (duplicate, _profile("b", {"cell": "g"})), resamples=2)

    invalid_model = analysis.FixedWaveProfile(
        profile_id="invalid",
        weights=_profile("a", {"cell": "p"}).weights,
        model=object(),  # type: ignore[arg-type]
        selected_path_ids=(("cell", "p"),),
    )
    with pytest.raises(ValueError, match="model mode"):
        _run(cells, rows, (invalid_model, _profile("b", {"cell": "g"})), resamples=2)
    _run(cells, rows, (_profile("a", {"cell": "p"}), _profile("b", {"cell": "g"})), resamples=2)
    assert cells == original_cells
    assert rows == original_rows


def _cv_fixture() -> tuple[
    dict[str, object],
    list[dict[str, object]],
    dict[str, str],
    dict[str, str],
]:
    cells: dict[str, object] = {}
    rows: list[dict[str, object]] = []
    cell_to_circuit: dict[str, str] = {}
    cell_to_topology: dict[str, str] = {}
    for circuit_index, circuit_id in enumerate(("c1", "c2", "c3"), start=1):
        for topology_index, topology_id in enumerate(("1dpu_t8", "4dpu_t8"), start=1):
            cell_id = f"{circuit_id}:{topology_id}"
            cells[cell_id] = {
                "greedy_path_id": "g",
                "raw_features": {
                    "g": _raw(100.0),
                    "p": _raw(40.0 + circuit_index + topology_index),
                },
            }
            cell_to_circuit[cell_id] = circuit_id
            cell_to_topology[cell_id] = topology_id
            rows.extend(
                _rows(
                    {
                        "g": (10.0 + circuit_index, 11.0 + topology_index),
                        "p": (
                            5.0 + circuit_index + topology_index,
                            5.5 + circuit_index + topology_index,
                        ),
                    },
                    cell_id=cell_id,
                    round_id="shared-round",
                )
            )
    return cells, rows, cell_to_circuit, cell_to_topology


def _run_cv(
    cells: dict[str, object],
    rows: list[dict[str, object]],
    cell_to_circuit: dict[str, str],
    cell_to_topology: dict[str, str],
    **kwargs: object,
) -> dict[str, object]:
    return analysis.compare_upmem_wave_models(
        cells,
        rows,
        _expected(rows),
        cell_to_circuit,
        cell_to_topology,
        fit_sample_count=8,
        bootstrap_resamples=12,
        **kwargs,
    )


def test_instance_folds_keep_both_topologies_and_isolate_fit_inputs(monkeypatch) -> None:
    cells, rows, cell_to_circuit, cell_to_topology = _cv_fixture()
    original_fit = analysis.fit_upmem_wave_paths
    calls: list[tuple[set[str], set[str], int, int, str]] = []

    def record_fit(*args, **kwargs):
        calls.append(
            (
                set(args[0]),
                {row["cell_id"] for row in args[1]},
                kwargs["sample_count"],
                kwargs["seed"],
                kwargs["model_form"],
            )
        )
        return original_fit(*args, **kwargs)

    monkeypatch.setattr(analysis, "fit_upmem_wave_paths", record_fit)
    result = _run_cv(cells, rows, cell_to_circuit, cell_to_topology)
    assert len(calls) == 6
    circuits = {"c1", "c2", "c3"}
    for fold in result["folds"]:
        held_out = fold["held_out_circuit_id"]
        assert set(fold["held_out_topology_ids"]) == {"1dpu_t8", "4dpu_t8"}
        assert set(fold["training_circuit_ids"]) == circuits - {held_out}
        assert all(
            cell_to_circuit[cell_id] != held_out
            for cell_id in fold["training_cell_ids"]
        )
    for cell_ids, row_cell_ids, sample_count, seed, model_form in calls:
        assert sample_count == 8
        assert seed == analysis.DEFAULT_FIT_SEED
        assert model_form in {"six_term", "grouped"}
        assert len({cell_to_circuit[cell_id] for cell_id in cell_ids}) == 2
        assert row_cell_ids == cell_ids


def test_cv_pooled_comparison_calls_preregistered_helper_and_reports_empirical_facts(
    monkeypatch,
) -> None:
    cells, rows, cell_to_circuit, cell_to_topology = _cv_fixture()
    seen: dict[str, object] = {}
    original_selector = analysis.select_wave_model_form

    def record_selector(interval, **kwargs):
        seen["interval"] = interval
        seen["kwargs"] = kwargs
        return original_selector(interval, **kwargs)

    monkeypatch.setattr(analysis, "select_wave_model_form", record_selector)
    result = _run_cv(cells, rows, cell_to_circuit, cell_to_topology)
    assert result["model_selection"]["helper"] == "select_wave_model_form"
    assert result["model_selection"]["selected_model_form"] == "grouped"
    assert len(seen["interval"]) == 2
    assert seen["kwargs"]["six_term_worst_speedup"] > 0.0
    assert seen["kwargs"]["grouped_worst_speedup"] > 0.0
    assert (
        result["pooled_cv"]["fold_specific_model_metrics"]["six_term"]["cell_count"]
        == 6
    )
    assert (
        result["pooled_cv"]["fold_specific_model_metrics"]["grouped"]["cell_count"]
        == 6
    )
    assert result["pooled_cv"]["bootstrap"]["resamples"] == 12
    for fold in result["folds"]:
        diagnostics = fold["training_feature_diagnostics"]
        assert "empirical_rank" in diagnostics
        assert "declared_model_rank" not in diagnostics
        assert "pairwise_pearson_correlation" in diagnostics
        assert "pairwise_spearman_correlation" in diagnostics
        assert "distinct_value_count" in diagnostics
        assert "tie_group_count" in diagnostics
        for model in ("six_term", "grouped"):
            evaluation = fold["evaluations"][model]
            assert set(evaluation["paired_speedups"]) == set(
                fold["held_out_cell_ids"]
            )
            assert all(
                "selected_rank" in cell and "oracle_regret" in cell
                for cell in evaluation["cells"]
            )


def test_cv_rejects_non_training_split_and_invalid_topology_partition() -> None:
    cells, rows, cell_to_circuit, cell_to_topology = _cv_fixture()
    with pytest.raises(ValueError, match="training-only"):
        _run_cv(
            cells,
            rows,
            cell_to_circuit,
            cell_to_topology,
            split="development",
            stability_resamples=1,
            stability_fit_sample_count=2,
        )
    invalid_topology = dict(cell_to_topology)
    invalid_topology["c1:4dpu_t8"] = "1dpu_t8"
    with pytest.raises(ValueError, match="both wave topologies"):
        _run_cv(cells, rows, cell_to_circuit, invalid_topology)


def test_refit_stability_is_deterministic_and_reports_bounded_budget() -> None:
    cells, rows, cell_to_circuit, cell_to_topology = _cv_fixture()
    first = _run_cv(
        cells,
        rows,
        cell_to_circuit,
        cell_to_topology,
        stability_resamples=2,
        stability_fit_sample_count=4,
        stability_seed=31,
    )
    second = _run_cv(
        cells,
        rows,
        cell_to_circuit,
        cell_to_topology,
        stability_resamples=2,
        stability_fit_sample_count=4,
        stability_seed=31,
    )
    stability = first["refit_stability"]
    assert first == second
    assert stability["status"] == "ran"
    assert stability["resamples"] == 2
    assert stability["fit_sample_count"] == 4
    assert stability["fit_count"] == 12
    assert "not equivalent to full-search uncertainty" in stability["interpretation"]
    for model in ("six_term", "grouped"):
        for fold_weights in stability["weights_by_fold"][model].values():
            assert len(fold_weights) == 2
            assert all("weights" in row for row in fold_weights)
        for counts in stability["selection_frequencies"][model].values():
            assert sum(row["count"] for row in counts.values()) == 2


def test_refit_resampling_copies_complete_shared_blocks_without_fabricating_rows() -> None:
    cells = {
        "cell-a": {
            "greedy_path_id": "g",
            "raw_features": {"g": _raw(100.0), "p": _raw(50.0)},
        },
        "cell-b": {
            "greedy_path_id": "g",
            "raw_features": {"g": _raw(100.0), "p": _raw(50.0)},
        },
    }
    rows = _rows(
        {"g": (1.0, 1.0), "p": (1.0, 2.0)},
        cell_id="cell-a",
        round_id="shared-round",
    )
    rows += _rows(
        {"g": (1.0, 1.0), "p": (1.0, 0.5)},
        cell_id="cell-b",
        round_id="shared-round",
    )
    pool = analysis._measurement_pool(
        cells,
        rows,
        _expected(rows),
        "training",
    )
    sampled, expected = analysis._resample_measurement_rows(
        pool, random.Random(5), "training"
    )
    assert len(sampled) == len(expected) == len(rows)
    for block in {row["block"] for row in sampled}:
        block_rows = [row for row in sampled if row["block"] == block]
        values = {
            (row["cell_id"], row["candidate_path_id"]): row["total_wall_s"]
            for row in block_rows
        }
        assert values["cell-a", "g"] == values["cell-b", "g"] == 1.0
        assert values["cell-b", "p"] == pytest.approx(
            1.0 / values["cell-a", "p"]
        )
    assert rows != sampled
    assert all(row["attempt_type"] == "measurement" for row in sampled)


def test_cv_bootstrap_shares_round_draws_across_fold_pools() -> None:
    pools = {}
    for fold_id, values in (
        ("fold-a", {"g": (1.0, 1.0), "p": (1.0, 2.0)}),
        ("fold-b", {"g": (1.0, 1.0), "p": (1.0, 0.5)}),
    ):
        rows = _rows(values, cell_id=fold_id, round_id="shared-round")
        pools[fold_id] = analysis._measurement_pool(
            _cells({"g": _raw(100.0), "p": _raw(50.0)}, cell_id=fold_id),
            rows,
            _expected(rows),
            "training",
        )
    geo, _ = analysis._bootstrap_cv_log_differences(
        pools,
        {
            "six_term": {"fold-a": "p", "fold-b": "p"},
            "grouped": {"fold-a": "g", "fold-b": "g"},
        },
        resamples=20,
        seed=7,
    )
    assert geo == pytest.approx((0.0,) * 20)
