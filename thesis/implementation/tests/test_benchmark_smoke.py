from __future__ import annotations

import json
from pathlib import Path

from quantum_bench.bench.runner import run_suite


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
        assert metadata["route_family"] == "dense_gemm"
        for field in (
            "total_host_to_dpu_bytes",
            "total_dpu_to_host_bytes",
            "total_mram_to_wram_bytes",
            "max_working_set_bytes",
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
