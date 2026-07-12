from __future__ import annotations

import csv
import json
import re
from pathlib import Path

from quantum_bench.bench.generic_task_bridge import run_generic_task_bridge
from quantum_bench.bench.result_artifacts import KERNEL_FAMILIES, compare_results, load_result_records
from quantum_bench.bench.run_dirs import EVIDENCE_ARTIFACT_KIND, LEGACY_ARTIFACT_KIND, create_run_dir
from quantum_bench.bench.reporting import write_normalized_records, write_run_manifest


def _cpu_gpu_record(
    case_id: str,
    route_id: str,
    repeat_id: int,
    *,
    total_s: float,
    compute_s: float,
    validation_status: str = "passed",
    state_output_mode: str = "full_dump",
    validation_method: str = "full_statevector",
    performance_tier: bool = False,
) -> dict:
    is_gpu = route_id == "quest_gpu_full_state_exact"
    exact_output = state_output_mode != "none"
    return {
        "schema_version": "benchmark_result_artifact_v1",
        "source_artifact": f"cases/{case_id}/simulation_backend_compare.json",
        "run_id": "run",
        "suite_id": "cpu_gpu_sweep",
        "case_id": case_id,
        "workload_id": case_id,
        "route_id": route_id,
        "backend_id": route_id,
        "backend_family": "quest",
        "benchmark_role": "serious_gpu_full_state_baseline" if is_gpu else "serious_full_state_baseline",
        "kernel_family": "full_state_vector",
        "execution_model": "full_state",
        "output_kind": "metrics_only" if not exact_output else "statevector",
        "comparison_output_kind": "not_applicable" if not exact_output else "statevector",
        "state_output_mode": state_output_mode,
        "output_contract": "metrics_only" if not exact_output else "statevector",
        "output_contract_label": "metrics_only" if not exact_output else "full_statevector",
        "output_contract_is_exact": exact_output,
        "performance_tier": performance_tier,
        "exact_output_comparable": exact_output,
        "full_statevector_validation_available": exact_output,
        "execution_target": "gpu" if is_gpu else "cpu",
        "contraction_execution_target": "gpu" if is_gpu else "cpu",
        "accelerator_kind": "amd_gpu" if is_gpu else "none",
        "gpu_backend_verified": is_gpu,
        "gpu_program_executed": is_gpu,
        "gpu_device_name": "AMD Radeon RX 6600 (gfx1032)" if is_gpu else None,
        "execution_scope": "full_circuit",
        "status": "completed",
        "validation_status": validation_status,
        "repeat_id": repeat_id,
        "measured_repeat_count": 2,
        "task_count": 0,
        "validated_task_count": 0,
        "unsupported_task_count": 0,
        "total_wall_time_s": total_s,
        "kernel_time_s": compute_s,
        "simulation_compute_time_s": compute_s,
        "timing_scope": "compute_only_native_and_process_wall" if performance_tier else "end_to_end_and_compute",
        "validation_method": validation_method,
        "native_process_wall_time_s": total_s,
        "quest_simulation_compute_time_s": compute_s,
        "state_dump_requested": exact_output,
        "state_dump_time_s": 0.001 if exact_output else 0.0,
        "repeat_layers": 1,
        "energy_joules": None,
        "energy_source": "unavailable",
        "energy_measurement_status": "unavailable",
        "validation_error_metrics": {"max_abs_error": 0.0, "l2_error": 0.0},
        "statevector_bytes": 256 if exact_output else None,
        "hardware_speedup": "not_applicable",
        "hardware_speedup_applicable": False,
        "cpu_fallback_used": False,
        "notes": "{}",
        "warnings": "",
    }


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


def test_legacy_run_updates_runs_latest_without_repository_root_symlink(tmp_path: Path) -> None:
    run_dir = create_run_dir(tmp_path, "diagnostic", artifact_kind=LEGACY_ARTIFACT_KIND)

    assert (tmp_path / "runs" / "latest").resolve() == run_dir.resolve()
    assert not (tmp_path / "latest").exists()


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
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}(?:_\d{2})?", run_dir.name)
    assert (run_dir.parent / "latest").resolve() == run_dir.resolve()
    assert (run_dir.parent.parent / "latest").resolve() == run_dir.resolve()
    latest = tmp_path / "runs" / "latest"
    assert latest.is_symlink()
    assert latest.resolve() == run_dir.resolve()
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["artifact_kind"] == "evidence_run"
    assert manifest["route_label"] == "upmem_generic_int8"
    assert manifest["timestamp"]
    assert manifest["command"]
    assert manifest["benchmark_source_commit"] is None
    assert manifest["benchmark_source_worktree_dirty"] is None
    assert manifest["repository_worktree_dirty"] is None
    assert manifest["provenance_scope"] == "thesis/implementation"
    assert manifest["git_commit"] == manifest["benchmark_source_commit"]
    assert manifest["dirty_tree"] == manifest["benchmark_source_worktree_dirty"]
    assert manifest["dirty_worktree"] == manifest["benchmark_source_worktree_dirty"]

    out_dir = tmp_path / "runs" / "comparisons" / "suite_a" / "quantization_attribution" / "manual"
    result = compare_results([run_dir], out_dir, comparison_type="quantization_attribution", root_dir=tmp_path)
    comparison_manifest = json.loads((out_dir / "comparison_manifest.json").read_text(encoding="utf-8"))
    loaded_records = load_result_records([run_dir])
    assert result.record_count == 1
    assert loaded_records[0]["timestamp"] == manifest["timestamp"]
    assert loaded_records[0]["parallelism_mode"] == "sequential"
    assert loaded_records[0]["parallelism_evidence_type"] == "executed"
    assert loaded_records[0]["execution_plan_kind"] == "sequential_upmem_taskgraph"
    assert loaded_records[0]["execution_plan_executed"] is True
    with result.csv_path.open("r", encoding="utf-8", newline="") as handle:
        header = next(csv.reader(handle))
    assert "parallelism_mode" in header
    assert "parallelism_evidence_type" in header
    assert "slicing_backend" in header
    assert "slice_count" in header
    assert "slicing_total_flops" in header
    assert "unsliced_total_flops" in header
    assert "slicing_flop_ratio" in header
    assert "slicing_flop_metric_source" in header
    assert "slicing_flop_change_kind" in header
    assert "slicing_flop_inflation_factor" in header
    assert "slice_parallel_execution" in header
    assert "slicing_reconstruction_status" in header
    assert "slicing_memory_ratio" in header
    assert "frontier_parallel_execution" in header
    assert "frontier_worker_count" in header
    assert "frontier_wave_count" in header
    assert "frontier_executed_task_count" in header
    assert "source_frontier_completed_task_count" in header
    assert "frontier_executed_parallel_task_count" in header
    assert "duplicate_contraction_check" in header
    assert "slice_reconstruction_status" in header
    assert "slice_task_execution_mode" in header
    assert "hybrid_components" in header
    assert "hybrid_ready" in header
    assert "slice_model_execution_status" in header
    assert "source_task_count" in header
    assert "source_task_completion_count" in header
    assert "slice_model_slice_count" in header
    assert "slice_model_task_count" in header
    assert "slice_model_executed_task_count" in header
    assert "slice_parallel_wave_count" in header
    assert "hybrid_reconstruction_validation_status" in header
    assert "dependency_violation_detected" in header
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


def test_legacy_slicing_flop_inflation_maps_to_ratio_without_claiming_inflation(tmp_path: Path) -> None:
    run_dir = tmp_path / "legacy_slicing"
    run_dir.mkdir()
    legacy_record = {
        "schema_version": "benchmark_result_artifact_v1",
        "run_id": "legacy",
        "suite_id": "legacy_slicing",
        "case_id": "case",
        "workload_id": "case",
        "route_id": "quimb_tn_sliced_exact",
        "backend_id": "quimb_tn_sliced_exact",
        "backend_family": "quimb",
        "kernel_family": "external_tn_contraction",
        "execution_model": "tensor_network",
        "parallelism_mode": "slicing",
        "slicing_enabled": True,
        "slicing_flop_inflation": 0.5,
        "execution_scope": "full_circuit",
        "status": "completed",
        "validation_status": "passed",
        "task_count": 1,
        "validated_task_count": 1,
        "unsupported_task_count": 0,
    }
    (run_dir / "normalized_records.jsonl").write_text(json.dumps(legacy_record) + "\n", encoding="utf-8")

    loaded = load_result_records([run_dir])

    assert loaded[0]["slicing_flop_ratio"] == 0.5
    assert loaded[0]["slicing_flop_metric_source"] == "legacy_slicing_flop_inflation"
    assert loaded[0]["slicing_flop_change_kind"] == "legacy_unknown"
    assert loaded[0]["slicing_flop_inflation_factor"] is None


def test_compare_results_writes_cpu_gpu_speedup_artifacts(tmp_path: Path) -> None:
    run_dir = create_run_dir(tmp_path, "cpu_gpu_sweep", artifact_kind=EVIDENCE_ARTIFACT_KIND, route_label="simulation_backend_compare")
    records = [
        _cpu_gpu_record("quest_qrng_4q", "quest_cpu_full_state_exact", 0, total_s=4.0, compute_s=2.0),
        _cpu_gpu_record("quest_qrng_4q", "quest_gpu_full_state_exact", 0, total_s=1.0, compute_s=0.5),
        _cpu_gpu_record("quest_qrng_4q", "quest_cpu_full_state_exact", 1, total_s=6.0, compute_s=3.0),
        _cpu_gpu_record("quest_qrng_4q", "quest_gpu_full_state_exact", 1, total_s=2.0, compute_s=1.0),
        _cpu_gpu_record("quest_bv_6q", "quest_cpu_full_state_exact", 0, total_s=3.0, compute_s=1.5),
    ]
    write_run_manifest(
        run_dir,
        run_kind="simulation_backend_compare",
        suite_id="cpu_gpu_sweep",
        suite_path="configs/suites/cpu_gpu_sweep.yml",
        artifact_kind=EVIDENCE_ARTIFACT_KIND,
        route_label="simulation_backend_compare",
        artifact_retention="compact",
        root_dir=tmp_path,
    )
    write_normalized_records(run_dir, records)

    out_dir = tmp_path / "runs" / "comparisons" / "cpu_gpu" / "quest_full_state" / "run"
    result = compare_results([run_dir], out_dir, comparison_type="cpu_gpu_sweep", root_dir=tmp_path)
    payload = json.loads(result.artifact_path.read_text(encoding="utf-8"))

    assert result.record_count == len(records)
    assert (out_dir / "cpu_gpu_speedup_pairs.csv").exists()
    assert (out_dir / "cpu_gpu_speedup_summary.csv").exists()
    assert (out_dir / "cpu_gpu_speedup_skipped_pairs.csv").exists()
    assert (out_dir / "plots" / "data" / "cpu_gpu_speedup_summary.csv").exists()
    assert payload["cpu_gpu_speedup"]["timing_fields"] == {
        "total_wall": "total_wall_time_s",
        "compute": "simulation_compute_time_s",
        "repeat": "repeat_id",
        "performance_speedup_field": "compute_speedup",
    }
    assert len(payload["cpu_gpu_speedup"]["pairs"]) == 2
    assert payload["cpu_gpu_speedup"]["summary"][0]["case_family"] == "qrng"
    assert payload["cpu_gpu_speedup"]["summary"][0]["n_qubits"] == 4
    assert payload["cpu_gpu_speedup"]["summary"][0]["matched_repeat_count"] == 2
    assert payload["cpu_gpu_speedup"]["summary"][0]["state_output_mode"] == "full_dump"
    assert payload["cpu_gpu_speedup"]["summary"][0]["validation_method"] == "full_statevector"
    assert payload["cpu_gpu_speedup"]["summary"][0]["performance_tier"] is False
    assert payload["cpu_gpu_speedup"]["summary"][0]["total_wall_speedup_median"] == 3.5
    assert payload["cpu_gpu_speedup"]["summary"][0]["compute_speedup_median"] == 3.5
    assert payload["cpu_gpu_speedup"]["skipped_pairs"][0]["reason"] == "missing_cpu_or_gpu_row"

    with (out_dir / "cpu_gpu_speedup_pairs.csv").open("r", encoding="utf-8", newline="") as handle:
        pair_rows = list(csv.DictReader(handle))
    assert {row["repeat_id"] for row in pair_rows} == {"0", "1"}
    manifest = json.loads((out_dir / "comparison_manifest.json").read_text(encoding="utf-8"))
    assert "cpu_gpu_speedup_pairs.csv" in manifest["outputs"]
    assert "plots/data/cpu_gpu_speedup_summary.csv" in manifest["outputs"]
    summary_md = result.summary_path.read_text(encoding="utf-8")
    assert "## CPU/GPU Speedup" in summary_md


def test_compare_results_accepts_cpu_gpu_performance_tier_native_status(tmp_path: Path) -> None:
    run_dir = create_run_dir(tmp_path, "cpu_gpu_performance", artifact_kind=EVIDENCE_ARTIFACT_KIND, route_label="simulation_backend_compare")
    records = [
        _cpu_gpu_record(
            "quest_qrng_10q_perf",
            "quest_cpu_full_state_exact",
            0,
            total_s=5.0,
            compute_s=4.0,
            validation_status="passed_native_status",
            state_output_mode="none",
            validation_method="native_status_gate_counts",
            performance_tier=True,
        ),
        _cpu_gpu_record(
            "quest_qrng_10q_perf",
            "quest_gpu_full_state_exact",
            0,
            total_s=2.0,
            compute_s=1.0,
            validation_status="passed_native_status",
            state_output_mode="none",
            validation_method="native_status_gate_counts",
            performance_tier=True,
        ),
    ]
    write_run_manifest(
        run_dir,
        run_kind="simulation_backend_compare",
        suite_id="cpu_gpu_performance",
        suite_path="configs/suites/manual/cpu_gpu_performance.yml",
        artifact_kind=EVIDENCE_ARTIFACT_KIND,
        route_label="simulation_backend_compare",
        artifact_retention="compact",
        root_dir=tmp_path,
    )
    write_normalized_records(run_dir, records)

    out_dir = tmp_path / "runs" / "comparisons" / "cpu_gpu" / "performance" / "run"
    result = compare_results([run_dir], out_dir, comparison_type="cpu_gpu_sweep", root_dir=tmp_path)
    payload = json.loads(result.artifact_path.read_text(encoding="utf-8"))
    pair = payload["cpu_gpu_speedup"]["pairs"][0]
    summary = payload["cpu_gpu_speedup"]["summary"][0]

    assert pair["validation_status"] == "passed_native_status"
    assert pair["state_output_mode"] == "none"
    assert pair["case_family"] == "qrng"
    assert pair["n_qubits"] == 10
    assert pair["performance_tier"] is True
    assert pair["exact_output_comparable"] is False
    assert pair["full_statevector_validation_available"] is False
    assert pair["compute_speedup"] == 4.0
    assert summary["timing_scope"] == "performance_compute"
    assert summary["validation_method"] == "native_status_gate_counts"
