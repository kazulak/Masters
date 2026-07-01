from __future__ import annotations

import json
from pathlib import Path

import csv

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


def _simulation_record(
    case_id: str,
    route_id: str,
    *,
    n_qubits: int,
    gate_count: int,
    backend_family: str,
    execution_model: str,
    time_s: float,
    max_abs_error: float = 0.0,
    probability_l1_error: float = 0.0,
) -> dict:
    benchmark_roles = {
        "quest_cpu_full_state_exact": "serious_full_state_baseline",
        "cpu_tn_einsum_exact": "internal_debug_baseline",
        "quimb_tn_exact": "serious_external_tn_baseline",
    }
    limitation_scopes = {
        "cpu_tn_einsum_exact": "Internal einsum expression/lowering engine limitation, not a tensor-network approach limitation.",
    }
    return {
        "schema_version": "benchmark_result_artifact_v1",
        "source_artifact": f"cases/{case_id}/simulation_backend_compare.json",
        "run_id": "run",
        "suite_id": "suite",
        "case_id": case_id,
        "workload_id": case_id,
        "route_id": route_id,
        "backend_id": route_id,
        "backend_family": backend_family,
        "benchmark_role": benchmark_roles.get(route_id),
        "route_role_description": f"{route_id} fixture route",
        "route_limitation_scope": limitation_scopes.get(route_id, ""),
        "kernel_family": "full_state_vector" if execution_model == "full_state" else "external_tn_contraction",
        "execution_model": execution_model,
        "execution_target": "cpu",
        "contraction_execution_target": "cpu",
        "accelerator_kind": "none",
        "execution_scope": "full_circuit" if execution_model == "full_state" else "full_taskgraph",
        "output_kind": "statevector" if execution_model == "full_state" else "final_tensor",
        "comparison_output_kind": "statevector",
        "status": "completed",
        "validation_status": "passed",
        "policy": "not_applicable",
        "quantization_mode": "not_applicable",
        "task_count": 0 if execution_model == "full_state" else 3,
        "validated_task_count": 0 if execution_model == "full_state" else 3,
        "unsupported_task_count": 0,
        "planning_time_s": 0.0 if execution_model == "full_state" else 0.01,
        "lowering_time_s": 0.0,
        "total_wall_time_s": time_s,
        "kernel_time_s": time_s,
        "simulation_compute_time_s": time_s * 0.8,
        "setup_time_s": time_s * 0.05,
        "data_transfer_time_s": 0.0,
        "validation_time_s": time_s * 0.02,
        "output_materialization_time_s": time_s * 0.03,
        "timing_scope": "end_to_end_and_compute",
        "gpu_synchronized": False,
        "validation_method": "full_statevector",
        "expected_runtime_class": "local_medium",
        "expected_memory_class": "local_bounded",
        "intended_use": "local_validation",
        "max_qubits": n_qubits,
        "manual_invocation_required": False,
        "expected_risk": [],
        "known_heavy_backends": [],
        "resource_guard_status": "executed",
        "resource_skip_reason": None,
        "repeat_id": 0,
        "measured_repeat_count": 1,
        "hardware_speedup": "not_applicable",
        "validation_error_metrics": {
            "max_abs_error": max_abs_error,
            "l2_error": max_abs_error,
            "probability_l1_error": probability_l1_error,
            "probability_max_abs_error": probability_l1_error,
        },
        "statevector_bytes": 16 * (1 << n_qubits),
        "tn_task_count": 0 if execution_model == "full_state" else 3,
        "tn_max_intermediate_bytes": None if execution_model == "full_state" else 256 * n_qubits,
        "tn_estimated_flops": None if execution_model == "full_state" else 1024 * n_qubits,
        "tn_estimated_bytes": None if execution_model == "full_state" else 2048 * n_qubits,
        "notes": json.dumps({"n_qubits": n_qubits, "gate_count": gate_count}, sort_keys=True),
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


def test_report_run_writes_thesis_plot_sources_and_inventory(tmp_path: Path) -> None:
    records = [
        _simulation_record("quest_qrng_3q", "quest_cpu_full_state_exact", n_qubits=3, gate_count=3, backend_family="quest", execution_model="full_state", time_s=0.12),
        _simulation_record("quest_qrng_3q", "cpu_tn_einsum_exact", n_qubits=3, gate_count=3, backend_family="cpu", execution_model="tensor_network", time_s=0.08, max_abs_error=1.0e-15),
        _simulation_record("quest_qrng_3q", "quimb_tn_exact", n_qubits=3, gate_count=3, backend_family="quimb", execution_model="tensor_network", time_s=0.09, max_abs_error=1.0e-15),
        _simulation_record("quest_qrng_5q", "quest_cpu_full_state_exact", n_qubits=5, gate_count=5, backend_family="quest", execution_model="full_state", time_s=0.2),
        _simulation_record("quest_qrng_5q", "cpu_tn_einsum_exact", n_qubits=5, gate_count=5, backend_family="cpu", execution_model="tensor_network", time_s=0.16, max_abs_error=1.0e-15),
        _simulation_record("quest_bv_4q", "quest_cpu_full_state_exact", n_qubits=4, gate_count=9, backend_family="quest", execution_model="full_state", time_s=0.18),
        _simulation_record("quest_bv_4q", "quimb_tn_exact", n_qubits=4, gate_count=9, backend_family="quimb", execution_model="tensor_network", time_s=0.11, max_abs_error=1.0e-15),
    ]
    run_dir = _new_run(tmp_path / "run", records)

    result = report_run(run_dir, output_plots=True)

    assert result.status == "completed"
    backend_csv = run_dir / "plots" / "data" / "backend_results.csv"
    scaling_csv = run_dir / "plots" / "data" / "runtime_scaling_by_qubits.csv"
    relative_csv = run_dir / "plots" / "data" / "relative_runtime_vs_quest_anchor.csv"
    compute_csv = run_dir / "plots" / "data" / "compute_time_by_backend_case.csv"
    total_vs_compute_csv = run_dir / "plots" / "data" / "total_vs_compute_time.csv"
    assert backend_csv.exists()
    assert scaling_csv.exists()
    assert relative_csv.exists()
    assert compute_csv.exists()
    assert total_vs_compute_csv.exists()
    with backend_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert {row["case_family"] for row in rows} >= {"qrng", "bv"}
    assert {row["expected_runtime_class"] for row in rows} == {"local_medium"}
    assert {row["benchmark_role"] for row in rows} >= {"serious_full_state_baseline", "serious_external_tn_baseline", "internal_debug_baseline"}
    internal_rows = [row for row in rows if row["route_id"] == "cpu_tn_einsum_exact"]
    assert internal_rows
    assert all("not a tensor-network approach limitation" in row["route_limitation_scope"] for row in internal_rows)
    with relative_csv.open("r", encoding="utf-8", newline="") as handle:
        relative_rows = list(csv.DictReader(handle))
    assert relative_rows
    assert all(float(row["relative_runtime"]) > 0 for row in relative_rows)

    manifest = json.loads((run_dir / "plots" / "plot_manifest.json").read_text(encoding="utf-8"))
    summary = (run_dir / "comparison_summary.md").read_text(encoding="utf-8")
    assert "## Plot Inventory" in summary
    if manifest["status"] == "completed":
        entries = {entry["plot"]: entry for entry in manifest["plots"]}
        assert "runtime_by_backend_case.png" in entries
        assert "runtime_scaling_by_qubits.png" in entries
        assert "compute_time_by_backend_case.png" in entries
        assert "total_vs_compute_time.png" in entries
        for entry in entries.values():
            assert (run_dir / entry["source_csv"]).exists()
            if entry["status"] == "generated":
                assert entry["image"]["non_empty"] is True
                assert entry["image"]["reasonable_dimensions"] is True
            else:
                assert entry["reason"]
    else:
        assert manifest["reason"] in {"matplotlib_unavailable", "plot_generation_disabled"}


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
