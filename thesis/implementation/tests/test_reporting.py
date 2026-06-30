from __future__ import annotations

import json
from pathlib import Path

from quantum_bench.bench.reporting import compare_runs, prune_run, report_run, validate_retention_mode, write_run_manifest
from quantum_bench.core.jsonio import write_json, write_jsonl


def _record(case_id: str, *, status: str = "completed", validation_status: str = "passed", kernel_family: str = "generic_loop_fallback", time_s: float = 1.0):
    return {
        "schema_version": "benchmark_result_artifact_v1",
        "source_artifact": f"cases/{case_id}/summary.json",
        "run_id": "run",
        "suite_id": "suite",
        "case_id": case_id,
        "workload_id": case_id,
        "route_id": "upmem_tn_runtime",
        "backend_id": "upmem_sdk_simulator_generic_loop",
        "kernel_family": kernel_family,
        "execution_target": "upmem",
        "contraction_execution_target": "upmem",
        "upmem_execution_mode": "sdk_simulator",
        "execution_scope": "full_taskgraph",
        "simulator_or_hardware": "simulator",
        "status": status,
        "validation_status": validation_status,
        "task_count": 2,
        "validated_task_count": 2 if validation_status == "passed" else 0,
        "unsupported_task_count": 0 if status == "completed" else 1,
        "total_wall_time_s": time_s,
        "kernel_time_s": time_s / 2,
        "build_time_s": 0.1,
        "hardware_speedup": "not_applicable",
        "validation_error_metrics": {"max_abs_error": 0.1 if validation_status == "passed" else 1.0},
        "notes": json.dumps({"policy": "generic-only", "quantization_mode": "per_task_input_quantize"}, sort_keys=True),
        "warnings": "",
    }


def _new_run(path: Path, records: list[dict]) -> Path:
    path.mkdir(parents=True)
    write_run_manifest(
        path,
        run_kind="upmem_mvp_benchmark",
        suite_id="suite",
        suite_path="suite.yml",
        policies=("generic-only",),
        quantization_modes=("per_task_input_quantize",),
        upmem_execution_mode="sdk_simulator",
        artifact_retention="compact",
        root_dir=path,
    )
    write_jsonl(path / "normalized_records.jsonl", records)
    return path


def test_summary_only_retention_is_deferred() -> None:
    try:
        validate_retention_mode("summary-only")
    except ValueError as exc:
        assert "deferred" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("summary-only should be rejected in this wave")


def test_report_run_is_non_destructive(tmp_path: Path) -> None:
    run_dir = _new_run(tmp_path / "run", [_record("bell_2q")])
    raw_artifact = run_dir / "cases" / "bell_2q" / "upmem_taskgraph_bridge" / "task_0000" / "runner_work" / "build" / "bin" / "host"
    raw_artifact.parent.mkdir(parents=True)
    raw_artifact.write_text("native-binary-placeholder", encoding="utf-8")

    result = report_run(run_dir, output_plots=False)

    assert result.status == "completed"
    assert raw_artifact.exists()
    assert (run_dir / "report_run.json").exists()
    assert (run_dir / "metrics" / "timing_breakdown.csv").exists()


def test_prune_run_compact_is_idempotent_and_rewrites_pruned_refs(tmp_path: Path) -> None:
    run_dir = _new_run(tmp_path / "run", [_record("bell_2q")])
    bridge = run_dir / "cases" / "bell_2q" / "policy" / "mode" / "upmem_taskgraph_bridge" / "task_0000" / "generic"
    output = bridge / "outputs" / "output.npy"
    output.parent.mkdir(parents=True)
    output.write_bytes(b"not-a-real-npy")
    write_json(bridge / "output_manifest.json", {"output_blob": {"relative_path": "outputs/output.npy"}, "output_path": "outputs/output.npy"})

    first = prune_run(run_dir, artifact_retention="compact")
    second = prune_run(run_dir, artifact_retention="compact")

    assert first.status == "completed"
    assert second.status == "completed"
    assert not output.exists()
    manifest = json.loads((bridge / "output_manifest.json").read_text(encoding="utf-8"))
    assert manifest["output_blob"]["status"] == "intentionally_pruned"
    assert manifest["output_path"]["status"] == "intentionally_pruned"


def test_prune_run_rejects_legacy_layout(tmp_path: Path) -> None:
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    try:
        prune_run(legacy, artifact_retention="compact")
    except ValueError as exc:
        assert "unsupported_legacy_run_layout" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("legacy runs should be rejected")


def test_compare_runs_reports_validation_kernel_and_timing_changes(tmp_path: Path) -> None:
    baseline = _new_run(tmp_path / "baseline", [_record("bell_2q", kernel_family="generic_loop_fallback", time_s=2.0)])
    candidate = _new_run(
        tmp_path / "candidate",
        [
            _record("bell_2q", validation_status="failed", kernel_family="dense_gemm", time_s=3.0),
            _record("ghz_3q", kernel_family="generic_loop_fallback", time_s=1.0),
        ],
    )

    result = compare_runs(baseline, candidate, tmp_path / "comparison")
    payload = json.loads(result.artifact_path.read_text(encoding="utf-8"))

    assert result.status == "completed"
    assert payload["final_validation_accuracy_timing"]["validation_regression_count"] >= 1
    assert payload["final_validation_accuracy_timing"]["newly_supported_count"] >= 1
    assert payload["kernel_family_mix"]["row_count"] >= 2
    assert any(row["total_wall_time_delta_s"] for row in payload["final_validation_accuracy_timing"]["rows"])
