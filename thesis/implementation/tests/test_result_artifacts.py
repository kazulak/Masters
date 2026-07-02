from __future__ import annotations

import csv
import json
from pathlib import Path

from quantum_bench.bench.generic_task_bridge import run_generic_task_bridge
from quantum_bench.bench.result_artifacts import KERNEL_FAMILIES, compare_results, load_result_records
from quantum_bench.bench.run_dirs import EVIDENCE_ARTIFACT_KIND, create_run_dir
from quantum_bench.bench.reporting import write_normalized_records, write_run_manifest


def test_kernel_family_vocabulary_contains_generic_and_dense() -> None:
    assert "dense_gemm" in KERNEL_FAMILIES
    assert "einsum_contraction" in KERNEL_FAMILIES
    assert "external_tn_contraction" in KERNEL_FAMILIES
    assert "full_state_vector" in KERNEL_FAMILIES
    assert "generic_loop_fallback" in KERNEL_FAMILIES
    assert "unsupported" in KERNEL_FAMILIES


def test_compare_results_loads_generic_summary_directory(tmp_path: Path) -> None:
    bridge = run_generic_task_bridge(tmp_path / "runs", case="bell_2q", task_index=0, execute_external=False)

    records = load_result_records([bridge.run_dir])
    assert len(records) == 1
    assert records[0]["kernel_family"] == "generic_loop_fallback"
    assert records[0]["execution_scope"] == "task_level"
    assert records[0]["hardware_speedup"] == "not_applicable"

    out_dir = tmp_path / "comparison"
    result = compare_results([bridge.run_dir], out_dir)
    payload = json.loads(result.artifact_path.read_text(encoding="utf-8"))
    summary_md = result.summary_path.read_text(encoding="utf-8")

    assert result.record_count == 1
    assert payload["schema_version"] == "compare_results_v1"
    assert payload["records"][0]["kernel_family"] == "generic_loop_fallback"
    assert payload["metadata"]["simulator_timings_are_not_hardware_speedups"] is True
    assert "Simulator timings are task-level development evidence" in summary_md
    assert (out_dir / "kernel_family_summary.csv").exists()

    with result.csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["kernel_family"] == "generic_loop_fallback"
    assert rows[0]["hardware_speedup"] == "not_applicable"
    manifest = json.loads((out_dir / "comparison_manifest.json").read_text(encoding="utf-8"))
    assert manifest["artifact_kind"] == "comparison_report"
    assert manifest["comparison_type"] == "generic_comparison"
    assert manifest["input_count"] == 1
    assert manifest["metadata"]["evidence_inputs_are_read_only"] is True


def test_compare_results_fails_gracefully_without_compatible_artifacts(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()

    try:
        compare_results([empty], tmp_path / "out")
    except ValueError as exc:
        assert "no compatible benchmark result artifacts found" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("compare_results should reject directories with no compatible artifacts")


def test_evidence_run_layout_and_compare_results_read_only_boundary(tmp_path: Path) -> None:
    run_dir = create_run_dir(tmp_path, "suite_a", artifact_kind=EVIDENCE_ARTIFACT_KIND, route_label="upmem_generic_int8")
    assert (run_dir / "config").exists()
    assert (run_dir / "cases").exists()
    assert not (run_dir / "raw").exists()
    assert not (run_dir / "metrics").exists()
    assert not (run_dir / "validation").exists()
    assert not (run_dir / "plots").exists()
    record = {
        "schema_version": "benchmark_result_artifact_v1",
        "run_id": run_dir.name,
        "suite_id": "suite_a",
        "case_id": "case_a",
        "route_id": "upmem_tn_runtime",
        "kernel_family": "generic_loop_fallback",
        "execution_target": "upmem",
        "contraction_execution_target": "upmem",
        "upmem_execution_mode": "sdk_simulator",
        "execution_scope": "full_taskgraph",
        "status": "completed",
        "validation_status": "passed",
        "task_count": 1,
        "validated_task_count": 1,
        "unsupported_task_count": 0,
        "hardware_speedup": "not_applicable",
    }
    write_run_manifest(
        run_dir,
        run_kind="upmem_mvp_benchmark",
        suite_id="suite_a",
        suite_path="suite.yml",
        artifact_kind=EVIDENCE_ARTIFACT_KIND,
        route_label="upmem_generic_int8",
        route_id="upmem_tn_runtime",
        policy="generic-only",
        quantization_mode="per_task_input_quantize",
        policies=("generic-only",),
        quantization_modes=("per_task_input_quantize",),
        upmem_execution_mode="sdk_simulator",
        artifact_retention="compact",
        root_dir=tmp_path,
    )
    write_normalized_records(run_dir, [record])

    assert run_dir.parent == tmp_path / "runs" / "evidence" / "suite_a" / "upmem_generic_int8"
    latest = tmp_path / "runs" / "latest"
    assert latest.is_symlink()
    assert latest.resolve() == run_dir.resolve()
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["artifact_kind"] == "evidence_run"
    assert manifest["route_label"] == "upmem_generic_int8"

    out_dir = tmp_path / "runs" / "comparisons" / "suite_a" / "quantization_attribution" / "manual"
    result = compare_results([run_dir], out_dir, comparison_type="quantization_attribution", root_dir=tmp_path)
    comparison_manifest = json.loads((out_dir / "comparison_manifest.json").read_text(encoding="utf-8"))
    assert result.record_count == 1
    assert comparison_manifest["artifact_kind"] == "comparison_report"
    assert comparison_manifest["comparison_type"] == "quantization_attribution"
    assert comparison_manifest["inputs"][0]["artifact_kind"] == "evidence_run"
    assert comparison_manifest["inputs"][0]["route_label"] == "upmem_generic_int8"

    try:
        compare_results([run_dir], tmp_path / "runs" / "evidence" / "suite_a" / "bad_compare", root_dir=tmp_path)
    except ValueError as exc:
        assert "must not be written under runs/evidence" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("compare_results should reject outputs inside runs/evidence")
