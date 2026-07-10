from __future__ import annotations

import csv
import json
from pathlib import Path

from quantum_bench.bench.config import comparison_planner_configs, load_suite
from quantum_bench.bench.planner_compare import compare_planners
from quantum_bench.targets.upmem import UPMEM_DENSE_ESTIMATE_KEY


ROOT = Path(__file__).resolve().parents[1]


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_planner_comparison_writes_rows_and_artifacts(tmp_path: Path) -> None:
    run_dir = compare_planners(ROOT / "configs" / "suites" / "diagnostics" / "planner_compare.yml", tmp_path)

    payload = json.loads((run_dir / "planner_comparison.json").read_text(encoding="utf-8"))
    csv_rows = list(csv.DictReader((run_dir / "planner_comparison.csv").open(encoding="utf-8", newline="")))
    rows = payload["rows"]
    divergence_summary = payload["divergence_summary"]

    assert payload["schema_version"] == "planner_comparison_v2"
    assert payload["suite_id"] == "planner_compare"
    assert payload["run_id"] == run_dir.name
    assert run_dir.parent.name == "planner_comparison"
    assert run_dir.parents[1].name == "planner_compare"
    assert [config["optimize"] for config in payload["planner_configs"]] == ["greedy", "optimal"]
    assert payload["scoring"]["score_model"] == "upmem_pressure_v1"
    assert payload["scoring"]["rank_scope"] == "case_id"
    assert payload["scoring"]["normalization"] == "per_case_minmax"
    assert payload["scoring"]["rank_order"] == "lower_score_is_better"
    assert payload["scoring"]["weights"]["parallelism_bonus_weight"] == 0.25
    assert divergence_summary["rank_scope"] == "case_id"
    assert divergence_summary["case_count"] == 2
    assert divergence_summary["divergent_case_count"] == len(divergence_summary["divergent_case_ids"])
    assert len(rows) == 4
    assert len(csv_rows) == len(rows)
    assert (run_dir / "planner_comparison_summary.md").exists()
    assert (run_dir / "run_manifest.json").exists()
    assert (run_dir / "artifact_retention_manifest.json").exists()
    normalized_rows = _read_jsonl(run_dir / "normalized_records.jsonl")
    assert len(normalized_rows) == len(rows)
    assert not list((run_dir / "raw").glob("*.jsonl"))
    assert not list((run_dir / "cases").glob("*/route_decisions.jsonl"))

    seen = {(row["case_id"], row["planner_id"]) for row in rows}
    assert seen == {
        ("bell_2q", "opt_einsum.greedy"),
        ("bell_2q", "opt_einsum.optimal"),
        ("ghz_3q", "opt_einsum.greedy"),
        ("ghz_3q", "opt_einsum.optimal"),
    }

    for row in rows:
        assert row["case_id"]
        assert row["workload_id"] == row["case_id"]
        assert row["planner_engine"] == "opt_einsum"
        assert row["planner_kind"] == "external_path_optimizer"
        assert row["objective"] == "opt_einsum_contract_path"
        assert row["cost_basis"] == "opt_einsum_internal"
        assert row["task_count"] > 0
        assert row["total_estimated_flops"] >= 0
        assert row["peak_intermediate_bytes"] >= 0
        assert row["total_host_to_dpu_bytes"] >= 0
        assert row["total_dpu_to_host_bytes"] >= 0
        assert row["total_mram_to_wram_bytes"] >= 0
        assert row["unsupported_task_count"] >= 0
        assert row["tiling_required_task_count"] >= 0
        assert row["missing_target_estimate_count"] == 0
        assert row["estimated_total_tile_count"] >= 0
        assert row["estimated_max_parallel_tiles"] >= 0
        assert row["score_model"] == "upmem_pressure_v1"
        assert isinstance(row["upmem_pressure_score"], float)
        assert row["upmem_rank"] >= 1
        assert row["flop_rank"] >= 1
        assert isinstance(row["score_components"], dict)
        assert isinstance(row["score_weights"], dict)
        assert row["score_weights"]["parallelism_bonus_weight"] == 0.25
        assert row["score_components"]["raw"]["extra_tile_raw"] >= 0
        assert row["score_components"]["raw"]["tiling_raw"] >= row["tiling_required_task_count"]
        assert row["score_components"]["raw"]["potential_parallelism_raw"] == row["estimated_max_parallel_tiles"]
        assert "modeled" in row["tradeoff_note"]

        task_graph_artifact = Path(row["task_graph_artifact"])
        path_summary_artifact = Path(row["path_summary_artifact"])
        target_estimates_artifact = Path(row["target_estimates_artifact"])
        execution_bundle_artifact = Path(row["execution_bundle_artifact"])
        assert not task_graph_artifact.is_absolute()
        assert not path_summary_artifact.is_absolute()
        assert not target_estimates_artifact.is_absolute()
        assert not execution_bundle_artifact.is_absolute()
        assert (run_dir / task_graph_artifact).exists()
        assert (run_dir / path_summary_artifact).exists()
        assert (run_dir / target_estimates_artifact).exists()
        assert (run_dir / execution_bundle_artifact).exists()
        assert len(row["contraction_plan_hash"]) == 64

        task_graph = json.loads((run_dir / task_graph_artifact).read_text(encoding="utf-8"))
        path_summary = json.loads((run_dir / path_summary_artifact).read_text(encoding="utf-8"))
        task_estimates = _read_jsonl(run_dir / target_estimates_artifact)
        assert task_graph["path_summary"] == path_summary
        assert len(task_estimates) == row["task_count"]
        assert path_summary["planner_id"] == row["planner_id"]
        assert path_summary["task_count"] == row["task_count"]
        assert path_summary["total_estimated_flops"] == row["total_estimated_flops"]
        assert path_summary["peak_intermediate_bytes"] == row["peak_intermediate_bytes"]
        assert path_summary["total_host_to_dpu_bytes"] == row["total_host_to_dpu_bytes"]
        assert path_summary["total_dpu_to_host_bytes"] == row["total_dpu_to_host_bytes"]
        assert path_summary["total_mram_to_wram_bytes"] == row["total_mram_to_wram_bytes"]
        assert path_summary["unsupported_task_count"] == row["unsupported_task_count"]
        assert path_summary["tiling_required_task_count"] == row["tiling_required_task_count"]
        assert path_summary["missing_target_estimate_count"] == row["missing_target_estimate_count"]
        assert path_summary["estimated_total_tile_count"] == row["estimated_total_tile_count"]
        assert path_summary["estimated_max_parallel_tiles"] == row["estimated_max_parallel_tiles"]
        for task_row in task_estimates:
            assert task_row["estimate_key"] == UPMEM_DENSE_ESTIMATE_KEY
            assert task_row["host_to_dpu_bytes"] >= 0
            assert task_row["dpu_to_host_bytes"] >= 0
            assert task_row["mram_to_wram_bytes"] >= 0

    assert all(row["route_id"] == "planner_candidate_model" for row in normalized_rows)
    assert all(row["parallelism_evidence_type"] == "modeled" for row in normalized_rows)
    assert all(row["execution_plan_executed"] is False for row in normalized_rows)

    for case_id in {"bell_2q", "ghz_3q"}:
        case_rows = [row for row in rows if row["case_id"] == case_id]
        assert any(row["upmem_rank"] == 1 for row in case_rows)
        assert any(row["flop_rank"] == 1 for row in case_rows)
        assert sum(1 for row in case_rows if row["upmem_rank"] == 1) >= 1


def test_extended_planner_comparison_suite_is_bounded_and_scored(tmp_path: Path) -> None:
    run_dir = compare_planners(ROOT / "configs" / "suites" / "diagnostics" / "planner_compare_extended.yml", tmp_path)

    payload = json.loads((run_dir / "planner_comparison.json").read_text(encoding="utf-8"))
    csv_rows = list(csv.DictReader((run_dir / "planner_comparison.csv").open(encoding="utf-8", newline="")))
    rows = payload["rows"]
    case_ids = {row["case_id"] for row in rows}
    divergent_case_ids = set(payload["divergence_summary"]["divergent_case_ids"])

    assert payload["schema_version"] == "planner_comparison_v2"
    assert payload["suite_id"] == "planner_compare_extended"
    assert [config["optimize"] for config in payload["planner_configs"]] == ["greedy", "auto", "random-greedy"]
    assert len(rows) == 30
    assert len(csv_rows) == 30
    assert len(case_ids) == 10
    assert "edc_4q" in case_ids
    assert payload["divergence_summary"]["rank_scope"] == "case_id"
    assert payload["divergence_summary"]["case_count"] == 10
    assert payload["divergence_summary"]["divergent_case_count"] == len(divergent_case_ids)
    assert divergent_case_ids <= case_ids
    assert (run_dir / "planner_comparison_summary.md").exists()
    assert not list((run_dir / "raw").glob("*.jsonl"))
    assert not list((run_dir / "cases").glob("*/route_decisions.jsonl"))

    for case_id in case_ids:
        case_rows = [row for row in rows if row["case_id"] == case_id]
        assert len(case_rows) == 3
        assert {row["planner_id"] for row in case_rows} == {
            "opt_einsum.greedy",
            "opt_einsum.auto",
            "opt_einsum.random-greedy",
        }
        assert any(row["upmem_rank"] == 1 for row in case_rows)
        assert any(row["flop_rank"] == 1 for row in case_rows)

    for row in rows:
        assert row["score_model"] == "upmem_pressure_v1"
        assert row["workload_id"] == row["case_id"]
        for key in ("task_graph_artifact", "path_summary_artifact", "target_estimates_artifact"):
            artifact = Path(row[key])
            assert not artifact.is_absolute()
            assert (run_dir / artifact).exists()


def test_planner_comparison_suite_configs_separate_tiny_and_extended_modes() -> None:
    tiny = load_suite(ROOT / "configs" / "suites" / "diagnostics" / "planner_compare.yml")
    extended = load_suite(ROOT / "configs" / "suites" / "diagnostics" / "planner_compare_extended.yml")

    assert [config["optimize"] for config in comparison_planner_configs(tiny)] == ["greedy", "optimal"]
    assert [config["optimize"] for config in comparison_planner_configs(extended)] == ["greedy", "auto", "random-greedy"]
    assert "optimal" not in {config["optimize"] for config in comparison_planner_configs(extended)}
    assert len(extended["cases"]) == 10
    assert "edc_4q" in {case["case_id"] for case in extended["cases"]}
    assert max(int(case["circuit"].get("n_qubits", 0) or 0) for case in extended["cases"]) <= 6
