from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "analyze_upmem_path_v1_robustness.py"
SPEC = importlib.util.spec_from_file_location("analyze_upmem_path_v1_robustness", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
analyzer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analyzer)

BASE = ROOT / "thesis_results" / "upmem_path_heuristic_v1"
CALIBRATION = BASE / "physical_calibration" / "path_runtime_calibration.json"
CANDIDATES = BASE / "software" / "candidate_paths.json"
PROFILE = BASE / "fit" / "physical_speedup_fit_v1.json"
WEIGHT_SEARCH = BASE / "fit" / "weight_search_candidates.csv"
VALIDATION = BASE / "validation" / "heuristic_validation.json"
TEST = BASE / "test" / "heuristic_test.json"


def _run(output_dir: Path, *, resamples: int = 1_000) -> dict:
    return analyzer.analyze(
        CALIBRATION,
        CANDIDATES,
        PROFILE,
        WEIGHT_SEARCH,
        VALIDATION,
        TEST,
        output_dir,
        bootstrap_resamples=resamples,
        bootstrap_seed=analyzer.BOOTSTRAP_SEED,
    )


def test_surviving_calibration_is_sample_paired_and_historical_data_is_not_raw(
    tmp_path: Path,
) -> None:
    result = _run(tmp_path, resamples=8)
    calibration = next(
        item
        for item in result["calibration_stats"]
        if item["cell_id"] == "edc_18q:1dpu_t8"
        and item["candidate_path_id"] == "09eba679eeaacde57b3d14e532a34cfb89b93de888eb607ea84d8a211c4d2e29"
    )
    assert calibration["warmup_count"] == 1
    assert calibration["measurement_count"] == 3
    assert calibration["measurement_blocks"] == [1, 2, 3]
    assert len(calibration["measurement_observations"]) == 3
    total = calibration["metrics"]["total_wall_s"]
    values = [
        row["metrics"]["total_wall_s"]
        for row in calibration["measurement_observations"]
    ]
    assert total["median"] == analyzer._median(values)
    assert total["raw_mad"] == analyzer._mad(values)
    assert total["minimum"] == min(values)
    assert total["maximum"] == max(values)
    assert total["bootstrap_resamples"] == analyzer.BOOTSTRAP_RESAMPLES

    for split in ("validation", "test"):
        historical = result["historical_evaluation_status"][split]
        assert historical["raw_observations_available"] is False
        assert historical["raw_uncertainty_available"] is False
        assert historical["raw_artifact_recovered"] is False
        assert "no raw observations reconstructed" in historical["disposition"]
    bv = {row["topology_id"]: row for row in result["historical_bv18_interpretation"]}
    assert bv["1dpu_t8"]["classification"] == "A_greedy_is_fastest_measured"
    assert bv["4dpu_t8"]["classification"] == (
        "C_indistinguishable_within_reported_variability"
    )
    assert all(row["raw_observations_available"] is False for row in bv.values())


def test_ranking_headroom_and_correlation_outputs_are_complete(tmp_path: Path) -> None:
    result = _run(tmp_path, resamples=8)
    summaries = result["ranking_metrics"]
    assert len(summaries) == 8
    assert all(1 <= row["profile_selected_rank"] <= row["measured_candidate_count"] for row in summaries)
    assert all(1 <= row["profile_selected_rank"] for row in summaries)
    assert all("oracle_regret" in row and "greedy_regret" in row for row in summaries)
    assert all(
        row["captured_headroom"] is not None
        or row["captured_headroom_reason"] == "no_measurable_candidate_pool_headroom"
        for row in summaries
    )

    ranking_rows = list(
        csv.DictReader((tmp_path / "existing_v1_ranking.csv").open(encoding="utf-8"))
    )
    assert len(ranking_rows) == 48
    assert {row["cell_id"] for row in ranking_rows} == {
        item["cell_id"] for item in summaries
    }
    assert all(row["spearman_score_runtime"] for row in ranking_rows)

    correlation_rows = list(
        csv.DictReader(
            (tmp_path / "existing_v1_feature_correlations.csv").open(encoding="utf-8")
        )
    )
    assert {row["row_kind"] for row in correlation_rows} == {"feature", "pair"}
    assert {row["representation"] for row in correlation_rows} == {"raw", "normalized"}
    assert any(
        row["feature"] == "E_num" and row["variance_population"] == "0.0"
        for row in correlation_rows
        if row["row_kind"] == "feature"
    )
    assert any(
        row["left_feature"] == "B_host_dpu"
        and row["right_feature"] == "B_mram_wram"
        and row["spearman"]
        for row in correlation_rows
        if row["row_kind"] == "pair"
    )


def test_bootstrap_uses_all_finite_profiles_and_is_deterministic(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    _run(first)
    _run(second)
    first_json = (first / "existing_v1_robustness.json").read_bytes()
    second_json = (second / "existing_v1_robustness.json").read_bytes()
    assert first_json == second_json
    result = json.loads(first_json)
    bootstrap = result["bootstrap_refits"]
    assert bootstrap["resamples"] == 1_000
    assert bootstrap["finite_profile_count"] == 11
    assert bootstrap["method"].startswith("paired_measurement_block_bootstrap")
    assert set(bootstrap["zero_weight_fraction"]) == set(analyzer.FEATURES)

    stability_rows = list(
        csv.DictReader(
            (first / "existing_v1_weight_stability.csv").open(encoding="utf-8")
        )
    )
    assert len(stability_rows) == 1_000
    assert {int(row["resample_index"]) for row in stability_rows} == set(range(1_000))
    assert sum(int(value) for value in bootstrap["profile_frequency"].values()) == 1_000

    loo_rows = list(
        csv.DictReader(
            (first / "existing_v1_leave_one_out.csv").open(encoding="utf-8")
        )
    )
    assert len(loo_rows) == 32
    assert {row["evaluation_scope"] for row in loo_rows} == {"fit", "held_out"}
    assert len(result["leave_one_training_circuit_out"]) == 4


def test_correlation_helpers_handle_ties_and_constant_vectors() -> None:
    assert analyzer._spearman([1.0, 2.0, 2.0, 4.0], [10.0, 20.0, 20.0, 40.0]) == 1.0
    assert analyzer._spearman([1.0, 1.0], [2.0, 3.0]) is None
    assert analyzer._tie_fraction([1.0, 1.0, 2.0]) == 1 / 3
