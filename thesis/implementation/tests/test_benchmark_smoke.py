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
    assert estimate["tile_plan_model"] == "dense_int8_wram_tile_plan_v1"
    assert isinstance(estimate["supported"], bool)
    assert isinstance(estimate["wram_fit"], bool)
    assert isinstance(estimate["requires_tiling"], bool)
    assert estimate["tiling_implemented"] is False
    assert isinstance(estimate["tile_plan_available"], bool)
    assert estimate["gemm_m"] >= 0
    assert estimate["gemm_k"] >= 0
    assert estimate["gemm_n"] >= 0
    assert estimate["tile_m"] >= 0
    assert estimate["tile_k"] >= 0
    assert estimate["tile_n"] >= 0
    assert estimate["tile_count_m"] >= 0
    assert estimate["tile_count_k"] >= 0
    assert estimate["tile_count_n"] >= 0
    assert estimate["total_tile_count"] >= 0
    assert estimate["estimated_tile_count"] == estimate["total_tile_count"]
    assert estimate["max_working_set_bytes"] >= 0
    assert estimate["estimated_tile_count"] >= 0
    assert estimate["estimated_parallel_tiles"] >= 0
    assert isinstance(estimate["double_buffer_possible"], bool)
    assert isinstance(estimate["requires_host_aggregation"], bool)
    assert estimate["host_to_dpu_bytes"] >= 0
    assert estimate["dpu_to_host_bytes"] >= 0
    assert estimate["mram_to_wram_bytes"] >= 0
    assert "reject_reason" in estimate
    assert "tile_plan" in estimate


def test_smoke_suite_writes_raw_summary_and_plots_contract(tmp_path: Path) -> None:
    run_dir = run_suite(ROOT / "configs" / "suites" / "smoke.yml", tmp_path)
    raw_rows = _read_raw_rows(run_dir)
    route_decisions = _read_route_decisions(run_dir)
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    environment = json.loads((run_dir / "environment.json").read_text(encoding="utf-8"))
    assert len(raw_rows) == 8
    assert summary["record_count"] == 8
    assert (run_dir / "environment.json").exists()
    assert "simplepim" in environment
    assert "SIMPLEPIM_HOME" in environment["simplepim"]
    assert "SIMPLEPIM_BIN" in environment["simplepim"]
    assert "SIMPLEPIM_LIB" in environment["simplepim"]
    assert "available" in environment["simplepim"]
    assert "probe_status" in environment["simplepim"]
    assert "skip_reason" in environment["simplepim"]
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
        tile_plan_rows = _read_jsonl(case_dir / "target_estimates" / "upmem_dense_tile_plan.jsonl")
        assert len(task_rows) == len(task_graph["tasks"])
        assert len(tile_plan_rows) == len(task_graph["tasks"])
        for task, row, tile_plan_row in zip(task_graph["tasks"], task_rows, tile_plan_rows):
            estimate = task["target_estimates"][UPMEM_DENSE_ESTIMATE_KEY]
            _assert_upmem_estimate_schema(estimate)
            assert row["task_id"] == task["id"]
            assert row["input_tensor_ids"] == task["input_tensor_ids"]
            assert row["output_tensor_id"] == task["output_tensor_id"]
            _assert_upmem_estimate_schema(row)
            assert tile_plan_row["task_id"] == task["id"]
            assert tile_plan_row["model"] == "dense_int8_wram_tile_plan_v1"
            assert tile_plan_row["tile_m"] == estimate["tile_m"]
            assert tile_plan_row["tile_k"] == estimate["tile_k"]
            assert tile_plan_row["tile_n"] == estimate["tile_n"]
            assert tile_plan_row["total_tile_count"] == estimate["total_tile_count"]
            assert tile_plan_row["working_set_bytes"] == estimate["max_working_set_bytes"]
            assert tile_plan_row["memory_model_note"].startswith("conservative")

        task_route_decisions = _read_jsonl(case_dir / "task_route_decisions.jsonl")
        task_route_summary = json.loads((case_dir / "task_route_summary.json").read_text(encoding="utf-8"))
        task_route_ids = ["dense_gemm", "sparse", "heuristic_bypass", "transpim_support", "cpu_fallback"]
        assert task_route_summary["schema_version"] == "task_route_summary_v1"
        assert task_route_summary["router_id"] == "static_task_router_v1"
        assert task_route_summary["case_id"] == case_dir.name
        assert task_route_summary["task_count"] == len(task_graph["tasks"])
        assert task_route_summary["route_ids"] == task_route_ids
        assert task_route_summary["decision_count"] == len(task_graph["tasks"]) * len(task_route_ids)
        assert task_route_summary["decision_count"] == len(task_route_decisions)
        assert task_route_summary["selected_task_count"] == len(task_graph["tasks"])
        assert task_route_summary["fallback_task_count"] == len(task_graph["tasks"])
        assert task_route_summary["non_fallback_selected_task_count"] == 0
        assert task_route_summary["missing_dense_estimate_count"] == 0
        assert task_route_summary["status_counts"]["fallback"] == len(task_graph["tasks"])
        assert task_route_summary["status_counts"]["unavailable"] == len(task_graph["tasks"]) * 3
        assert task_route_summary["route_status_counts"]["cpu_fallback"]["fallback"] == len(task_graph["tasks"])
        assert task_route_summary["policy"] == "analysis_only_cpu_fallback"
        decisions_artifact = Path(task_route_summary["decisions_artifact"])
        assert not decisions_artifact.is_absolute()
        assert (run_dir / decisions_artifact).exists()
        assert [decision["route_id"] for decision in task_route_decisions[: len(task_route_ids)]] == task_route_ids
        dense_task_decisions = [decision for decision in task_route_decisions if decision["route_id"] == "dense_gemm"]
        fallback_task_decisions = [decision for decision in task_route_decisions if decision["route_id"] == "cpu_fallback"]
        assert len(dense_task_decisions) == len(task_graph["tasks"])
        assert len(fallback_task_decisions) == len(task_graph["tasks"])
        for decision in task_route_decisions:
            assert decision["schema_version"] == "task_route_decision_v1"
            assert decision["router_id"] == "static_task_router_v1"
            assert decision["case_id"] == case_dir.name
            assert decision["task_id"]
            assert decision["input_tensor_ids"]
            assert decision["output_tensor_id"]
            assert decision["route_id"] in task_route_ids
            assert decision["status"] in {"selected", "rejected", "skipped", "unavailable", "fallback"}
            assert decision["execution_status"]["execution_implemented"] is False
            assert "estimate" in decision
        for decision, task in zip(dense_task_decisions, task_graph["tasks"]):
            estimate = task["target_estimates"][UPMEM_DENSE_ESTIMATE_KEY]
            route_estimate = decision["estimate"]
            assert decision["is_selected"] is False
            assert decision["execution_status"]["state"] == "estimate_only"
            assert route_estimate["metadata"]["target_estimate_key"] == UPMEM_DENSE_ESTIMATE_KEY
            assert route_estimate["metadata"]["tile_plan_available"] is True
            tile_plan_artifact = Path(route_estimate["metadata"]["tile_plan_artifact"])
            assert not tile_plan_artifact.is_absolute()
            assert (run_dir / tile_plan_artifact).exists()
            assert route_estimate["metadata"]["tile_count"] == estimate["total_tile_count"]
            assert route_estimate["metadata"]["working_set_bytes"] == estimate["max_working_set_bytes"]
            assert route_estimate["metadata"]["double_buffer_possible"] == estimate["double_buffer_possible"]
            assert route_estimate["metadata"]["requires_host_aggregation"] == estimate["requires_host_aggregation"]
            assert route_estimate["metadata"]["backend"] in {"simplepim_unavailable", "simplepim_future"}
            assert isinstance(route_estimate["metadata"]["simplepim_available"], bool)
            assert route_estimate["metadata"]["simplepim_probe_status"] in {
                "available",
                "unavailable",
                "configured_but_unverified",
            }
            assert "simplepim_version" in route_estimate["metadata"]
            assert "simplepim_command_path" in route_estimate["metadata"]
            assert "simplepim_library_path" in route_estimate["metadata"]
            assert "simplepim_skip_reason" in route_estimate["metadata"]
            assert route_estimate["supported"] == estimate["supported"]
            assert route_estimate["wram_fit"] == estimate["wram_fit"]
            assert route_estimate["requires_tiling"] == estimate["requires_tiling"]
            assert route_estimate["tiling_implemented"] == estimate["tiling_implemented"]
            assert route_estimate["host_to_dpu_bytes"] == estimate["host_to_dpu_bytes"]
            assert route_estimate["dpu_to_host_bytes"] == estimate["dpu_to_host_bytes"]
            assert route_estimate["mram_to_wram_bytes"] == estimate["mram_to_wram_bytes"]
        for decision in fallback_task_decisions:
            assert decision["status"] == "fallback"
            assert decision["is_selected"] is True
            assert decision["execution_status"]["state"] == "fallback_available"

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
        assert tile_shape["model"] == "dense_wram_tile_plan"
        assert tile_shape["max_working_set_bytes"] >= 0
        assert tile_shape["wram_bytes"] == 64 * 1024
        assert metadata["target"] == "upmem"
        assert metadata["estimate_key"] == UPMEM_DENSE_ESTIMATE_KEY
        assert metadata["route_family"] == "dense_gemm"
        assert metadata["tiling_implemented"] is False
        artifact = Path(metadata["task_estimates_artifact"])
        assert not artifact.is_absolute()
        assert (run_dir / artifact).exists()
        tile_plan_artifact = Path(metadata["tile_plan_artifact"])
        assert not tile_plan_artifact.is_absolute()
        assert (run_dir / tile_plan_artifact).exists()
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
