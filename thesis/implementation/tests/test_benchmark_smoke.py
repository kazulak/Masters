from __future__ import annotations

import json
from pathlib import Path

from quantum_bench.bench.runner import run_suite
from quantum_bench.targets.upmem import UPMEM_DENSE_ESTIMATE_KEY


ROOT = Path(__file__).resolve().parents[1]


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _read_raw_rows(run_dir: Path) -> list[dict[str, object]]:
    raw_rows: list[dict[str, object]] = []
    for raw in sorted((run_dir / "raw").glob("*.jsonl")):
        raw_rows.extend(_read_jsonl(raw))
    return raw_rows


def _read_route_decisions(run_dir: Path) -> list[dict[str, object]]:
    decisions: list[dict[str, object]] = []
    for path in sorted((run_dir / "cases").glob("*/route_decisions.jsonl")):
        decisions.extend(_read_jsonl(path))
    return decisions


def _assert_upmem_estimate_schema(estimate: dict[str, object]) -> None:
    assert estimate["target"] == "upmem"
    assert estimate["estimate_key"] == UPMEM_DENSE_ESTIMATE_KEY
    assert estimate["model"] == "dense_int8_single_dpu_feasibility"
    assert isinstance(estimate["supported"], bool)
    assert isinstance(estimate["wram_fit"], bool)
    assert isinstance(estimate["requires_tiling"], bool)
    assert estimate["tiling_implemented"] is False
    assert estimate["gemm_m"] >= 0
    assert estimate["gemm_k"] >= 0
    assert estimate["gemm_n"] >= 0
    assert estimate["max_working_set_bytes"] >= 0
    assert estimate["estimated_tile_count"] >= 0
    assert estimate["estimated_parallel_tiles"] >= 0
    assert estimate["host_to_dpu_bytes"] >= 0
    assert estimate["dpu_to_host_bytes"] >= 0
    assert estimate["mram_to_wram_bytes"] >= 0
    assert "reject_reason" in estimate


def test_smoke_suite_writes_raw_summary_and_plots_contract(tmp_path: Path) -> None:
    run_dir = run_suite(ROOT / "configs" / "suites" / "smoke.yml", tmp_path)
    raw_rows = _read_raw_rows(run_dir)
    route_decisions = _read_route_decisions(run_dir)
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert len(raw_rows) == 8
    assert summary["record_count"] == 8
    assert (run_dir / "environment.json").exists()
    assert (run_dir / "metrics" / "metrics.csv").exists()
    assert any(row["status"] == "skipped" and row["route"] == "upmem_dense_int8_placeholder" for row in raw_rows)
    assert any(row["status"] == "passed" and row["route"] == "cpu_tn_einsum_exact" for row in raw_rows)
    for row in raw_rows:
        assert "route" + "_alias" not in row
        assert row["role"]
        assert row["simulation_method"]
        assert row["kernel_family"]
        assert row["hardware_target"]
        assert row["execution_mode"]
        assert row["output_contract"]
        assert row["validation_mode"]
    assert summary["validated_routes"]
    assert summary["skipped_or_probe_routes"]

    for case_dir in sorted((run_dir / "cases").iterdir()):
        if not case_dir.is_dir():
            continue
        task_graph = json.loads((case_dir / "task_graph.json").read_text(encoding="utf-8"))
        path_summary = json.loads((case_dir / "path_summary.json").read_text(encoding="utf-8"))
        assert path_summary == task_graph["path_summary"]
        assert path_summary["planner_engine"] == "opt_einsum"
        assert path_summary["planner_id"] == "opt_einsum.greedy"
        assert path_summary["planner_kind"] == "external_path_optimizer"
        assert path_summary["optimize_mode"] == "greedy"
        assert path_summary["objective"] == "opt_einsum_contract_path"
        assert path_summary["cost_basis"] == "opt_einsum_internal"
        assert path_summary["target_estimate_key"] is None
        assert path_summary["options"] == {"engine": "opt_einsum", "optimize": "greedy"}
        assert path_summary["task_count"] == len(task_graph["tasks"])
        assert path_summary["total_estimated_flops"] == sum(task["estimated_flops"] for task in task_graph["tasks"])
        assert path_summary["peak_intermediate_bytes"] >= path_summary["max_intermediate_bytes"]
        assert path_summary["total_host_to_dpu_bytes"] >= 0
        assert path_summary["total_dpu_to_host_bytes"] >= 0
        assert path_summary["total_mram_to_wram_bytes"] >= 0
        assert path_summary["unsupported_task_count"] >= 0
        assert path_summary["tiling_required_task_count"] >= 0
        assert path_summary["missing_target_estimate_count"] == 0
        assert path_summary["estimated_total_tile_count"] >= 0
        assert path_summary["estimated_max_parallel_tiles"] >= 0
        task_rows = _read_jsonl(case_dir / "target_estimates" / f"{UPMEM_DENSE_ESTIMATE_KEY}.jsonl")
        assert len(task_rows) == len(task_graph["tasks"])
        for task, row in zip(task_graph["tasks"], task_rows):
            estimate = task["target_estimates"][UPMEM_DENSE_ESTIMATE_KEY]
            _assert_upmem_estimate_schema(estimate)
            assert row["task_id"] == task["id"]
            assert row["input_tensor_ids"] == task["input_tensor_ids"]
            assert row["output_tensor_id"] == task["output_tensor_id"]
            _assert_upmem_estimate_schema(row)

    cpu_rows = [row for row in raw_rows if row["route"] == "cpu_tn_einsum_exact" and row["status"] == "passed"]
    assert len(cpu_rows) == 4
    for row in cpu_rows:
        route_metadata = row["route_metadata"]
        assert isinstance(route_metadata, dict)
        assert route_metadata["execution_engine"] == "task_sequence_np_einsum"
        assert route_metadata["task_count"] > 0
        assert route_metadata["peak_intermediate_bytes"] >= route_metadata["max_intermediate_tensor_bytes"]
        assert route_metadata["final_tensor_id"]
        assert route_metadata["final_tensor_labels"]
        assert route_metadata["output_labels"]
        artifact = Path(route_metadata["task_metrics_artifact"])
        assert not artifact.is_absolute()
        artifact_path = run_dir / artifact
        assert artifact_path.exists()
        metrics = _read_jsonl(artifact_path)
        assert len(metrics) == route_metadata["task_count"]
        for metric in metrics:
            assert metric["task_id"]
            assert metric["input_tensor_ids"]
            assert metric["output_tensor_id"]
            assert metric["input_shapes"]
            assert metric["output_shape"]
            assert "contracted_labels" in metric
            assert metric["estimated_flops"] >= 0
            assert metric["estimated_bytes"] >= 0
            assert metric["execution_time_s"] >= 0.0
            assert metric["intermediate_tensor_bytes"] > 0
            target_estimates = metric["target_estimates"]
            assert isinstance(target_estimates, dict)
            _assert_upmem_estimate_schema(target_estimates[UPMEM_DENSE_ESTIMATE_KEY])

    upmem_decisions = [row for row in route_decisions if row["route"] == "upmem_dense_int8_placeholder"]
    assert len(upmem_decisions) == 2
    for decision in upmem_decisions:
        metadata = decision["metadata"]
        tile_shape = decision["tile_shape"]
        assert isinstance(metadata, dict)
        assert isinstance(tile_shape, dict)
        assert decision["status"] == "skipped"
        assert decision["wram_fit"] is not None
        assert tile_shape["model"] == "untiled_dense_gemm"
        assert tile_shape["max_working_set_bytes"] >= 0
        assert tile_shape["wram_bytes"] == 64 * 1024
        assert metadata["target"] == "upmem"
        assert metadata["estimate_key"] == UPMEM_DENSE_ESTIMATE_KEY
        assert metadata["route_family"] == "dense_gemm"
        assert metadata["tiling_implemented"] is False
        artifact = Path(metadata["task_estimates_artifact"])
        assert not artifact.is_absolute()
        assert (run_dir / artifact).exists()
        for field in (
            "total_host_to_dpu_bytes",
            "total_dpu_to_host_bytes",
            "total_mram_to_wram_bytes",
            "max_working_set_bytes",
            "total_estimated_tile_count",
            "max_estimated_parallel_tiles",
        ):
            assert field in metadata
            assert metadata[field] >= 0


def test_smoke_v2_suite_runs_with_same_contract(tmp_path: Path) -> None:
    run_dir = run_suite(ROOT / "configs" / "suites" / "smoke_v2.yml", tmp_path)
    raw_rows = _read_raw_rows(run_dir)
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert len(raw_rows) == 8
    assert summary["record_count"] == 8
    assert any(row["route"] == "cpu_tn_einsum_exact" and row["role"] == "reference" for row in raw_rows)
    assert any(row["route"] == "upmem_dense_int8_placeholder" and row["role"] == "candidate" for row in raw_rows)
    assert summary["validated_routes"]
