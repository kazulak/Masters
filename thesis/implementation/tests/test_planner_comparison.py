from __future__ import annotations

import csv
import json
from pathlib import Path

from quantum_bench.bench.planner_compare import compare_planners
from quantum_bench.targets.upmem import UPMEM_DENSE_ESTIMATE_KEY


ROOT = Path(__file__).resolve().parents[1]


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_planner_comparison_writes_rows_and_artifacts(tmp_path: Path) -> None:
    run_dir = compare_planners(ROOT / "configs" / "suites" / "planner_compare.yml", tmp_path)

    payload = json.loads((run_dir / "planner_comparison.json").read_text(encoding="utf-8"))
    csv_rows = list(csv.DictReader((run_dir / "planner_comparison.csv").open(encoding="utf-8", newline="")))
    rows = payload["rows"]

    assert payload["schema_version"] == "planner_comparison_v2"
    assert payload["suite_id"] == "planner_compare"
    assert payload["run_id"] == run_dir.name
    assert run_dir.name.endswith("_planner_compare")
    assert not run_dir.name.endswith("_planner_compare_planner_compare")
    assert [config["optimize"] for config in payload["planner_configs"]] == ["greedy", "optimal"]
    assert payload["scoring"]["score_model"] == "upmem_pressure_v1"
    assert payload["scoring"]["rank_scope"] == "case_id"
    assert payload["scoring"]["normalization"] == "per_case_minmax"
    assert payload["scoring"]["rank_order"] == "lower_score_is_better"
    assert payload["scoring"]["weights"]["parallelism_bonus_weight"] == 0.25
    assert len(rows) == 4
    assert len(csv_rows) == len(rows)
    assert (run_dir / "planner_comparison_summary.md").exists()
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
        assert not task_graph_artifact.is_absolute()
        assert not path_summary_artifact.is_absolute()
        assert not target_estimates_artifact.is_absolute()
        assert (run_dir / task_graph_artifact).exists()
        assert (run_dir / path_summary_artifact).exists()
        assert (run_dir / target_estimates_artifact).exists()

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

    for case_id in {"bell_2q", "ghz_3q"}:
        case_rows = [row for row in rows if row["case_id"] == case_id]
        assert any(row["upmem_rank"] == 1 for row in case_rows)
        assert any(row["flop_rank"] == 1 for row in case_rows)
        assert sum(1 for row in case_rows if row["upmem_rank"] == 1) >= 1
