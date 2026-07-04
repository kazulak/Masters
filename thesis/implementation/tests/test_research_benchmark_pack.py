from __future__ import annotations

import json
from pathlib import Path

from scripts import research_benchmark_pack as pack


def _record(case_id: str, route_id: str, repeat_id: int, *, target: str, total: float, compute: float) -> dict:
    is_gpu = route_id == "quest_gpu_full_state_exact"
    return {
        "schema_version": "benchmark_result_artifact_v1",
        "suite_id": "research_cpu_gpu",
        "case_id": case_id,
        "workload_id": case_id,
        "max_qubits": 20,
        "route_id": route_id,
        "backend_id": route_id,
        "backend_family": "quest",
        "benchmark_role": "serious_gpu_full_state_baseline" if is_gpu else "serious_full_state_baseline",
        "kernel_family": "full_state_vector",
        "execution_model": "full_state",
        "contraction_execution_target": target,
        "accelerator_kind": "amd_gpu" if is_gpu else "none",
        "gpu_backend_verified": is_gpu,
        "gpu_program_executed": is_gpu,
        "gpu_device_name": "AMD Radeon RX 6600 (gfx1032)" if is_gpu else None,
        "state_output_mode": "none",
        "validation_method": "native_status_gate_counts",
        "performance_tier": True,
        "validation_status": "passed_native_status",
        "status": "completed",
        "repeat_id": repeat_id,
        "total_wall_time_s": total,
        "simulation_compute_time_s": compute,
        "timing_scope": "compute_only_native_and_process_wall",
        "energy_joules": None,
        "energy_source": "unavailable",
        "energy_measurement_status": "unavailable",
        "hardware_speedup": "not_applicable",
        "hardware_speedup_applicable": False,
        "cpu_fallback_used": False,
        "validation_error_metrics": {"max_abs_error": 0.0, "l2_error": 0.0},
    }


def test_research_pack_statistics_and_cpu_gpu_pairing() -> None:
    records = [
        _record("quest_bv_10q_research_perf", "quest_cpu_full_state_exact", 0, target="cpu", total=10.0, compute=8.0),
        _record("quest_bv_10q_research_perf", "quest_gpu_full_state_exact", 0, target="gpu", total=5.0, compute=2.0),
        _record("quest_bv_10q_research_perf", "quest_cpu_full_state_exact", 1, target="cpu", total=12.0, compute=10.0),
        _record("quest_bv_10q_research_perf", "quest_gpu_full_state_exact", 1, target="gpu", total=6.0, compute=2.5),
    ]

    stats = pack.per_case_route_stats(records)
    speedups = pack.paired_speedups(records)

    assert len(stats) == 2
    cpu = next(row for row in stats if row["route_id"] == "quest_cpu_full_state_exact")
    assert cpu["repeat_count"] == 2
    assert cpu["n_qubits"] == 10
    assert cpu["actual_n_qubits"] == 10
    assert cpu["benchmark_n_qubits"] == 10
    assert cpu["actual_n_qubits_source"] == "case_id"
    assert cpu["simulation_compute_time_s_median"] == 9.0
    assert len(speedups) == 2
    assert speedups[0]["n_qubits"] == 10
    assert speedups[0]["actual_n_qubits"] == 10
    assert speedups[0]["benchmark_n_qubits"] == 10
    assert speedups[0]["compute_speedup_cpu_over_gpu"] == 4.0
    assert speedups[0]["timing_scope"] == "performance_compute"


def test_research_pack_actual_qubits_do_not_use_output_caps() -> None:
    record = _record("quest_xor_12q_research_perf", "quest_cpu_full_state_exact", 0, target="cpu", total=1.0, compute=1.0)
    record["max_qubits"] = 99
    record["max_output_amplitudes"] = 4096

    stats = pack.per_case_route_stats([record])

    assert stats[0]["n_qubits"] == 12
    assert stats[0]["actual_n_qubits"] == 12
    assert stats[0]["benchmark_n_qubits"] == 12
    assert stats[0]["actual_n_qubits_source"] == "case_id"


def test_research_pack_actual_qubits_prefer_explicit_fields() -> None:
    record = _record("opaque_case_name", "quest_cpu_full_state_exact", 0, target="cpu", total=1.0, compute=1.0)
    record["actual_n_qubits"] = 18
    record["max_qubits"] = 99

    stats = pack.per_case_route_stats([record])

    assert stats[0]["benchmark_n_qubits"] == 18
    assert stats[0]["actual_n_qubits_source"] == "actual_n_qubits"


def test_research_pack_actual_qubits_warn_when_unresolved() -> None:
    record = _record("opaque_case_name", "quest_cpu_full_state_exact", 0, target="cpu", total=1.0, compute=1.0)
    record.pop("max_qubits", None)

    stats = pack.per_case_route_stats([record])

    assert stats[0]["actual_n_qubits"] is None
    assert stats[0]["benchmark_n_qubits"] is None
    assert stats[0]["actual_n_qubits_warning"] == "actual_qubit_count_unresolved"


def test_research_pack_cpu_tn_plot_source_uses_actual_qubits() -> None:
    record = _record("quest_bv_12q_research_tn", "quimb_tn_exact", 0, target="cpu", total=2.0, compute=1.5)
    record["backend_family"] = "quimb"
    record["benchmark_role"] = "serious_external_tn_baseline"
    record["max_qubits"] = 14

    stats = pack.per_case_route_stats([record])

    assert stats[0]["route_id"] == "quimb_tn_exact"
    assert stats[0]["n_qubits"] == 12
    assert stats[0]["benchmark_n_qubits"] == 12


def test_research_pack_rejects_unverified_gpu_and_fake_energy() -> None:
    good_cpu = _record("quest_bv_10q_research_perf", "quest_cpu_full_state_exact", 0, target="cpu", total=10.0, compute=8.0)
    bad_gpu = _record("quest_bv_10q_research_perf", "quest_gpu_full_state_exact", 0, target="gpu", total=5.0, compute=2.0)
    bad_gpu["gpu_backend_verified"] = False
    fake_energy = dict(good_cpu)
    fake_energy["case_id"] = "quest_xor_10q_research_perf"
    fake_energy["energy_joules"] = 10.0

    assert pack.paired_speedups([good_cpu, bad_gpu]) == []
    issues = pack._claim_guard_issues([bad_gpu, fake_energy])
    assert any("unverified gpu row" in issue for issue in issues)
    assert any("energy value without measured status" in issue for issue in issues)


def test_research_pack_skipped_group_result_is_visible() -> None:
    result = pack._skipped_group_result("cpu_gpu", "hip_smoke_build_failed")

    assert result["returncode"] == 0
    assert result["skipped_group"] == "cpu_gpu"
    assert result["blocker_reason"] == "hip_smoke_build_failed"
    assert result["benchmark_rows_emitted"] is False


def test_research_pack_boundary_check_detects_derived_evidence_files(tmp_path: Path) -> None:
    bad = tmp_path / "runs" / "evidence" / "suite" / "route" / "run" / "comparison_summary.md"
    bad.parent.mkdir(parents=True)
    bad.write_text("derived", encoding="utf-8")

    result = pack.validate_artifact_boundaries(tmp_path)

    assert result["status"] == "failed"
    assert result["violations"] == ["runs/evidence/suite/route/run/comparison_summary.md"]


def test_research_pack_writes_lightweight_pack(tmp_path: Path) -> None:
    out = tmp_path / "pack"
    exit_code = pack._write_pack(
        tmp_path,
        out,
        [],
        command_results=[{"command": "unit", "returncode": 0, "stdout": "", "stderr": ""}],
        selected_suite_keys=["cpu_gpu"],
    )

    assert exit_code == 0
    assert (out / "benchmark_manifest.json").exists()
    assert (out / "per_case_route_stats.csv").exists()
    assert (out / "plot_manifest.json").exists()
    summary = (out / "benchmark_summary.md").read_text(encoding="utf-8")
    assert "Next UPMEM Implementation Readiness" in summary
    manifest = json.loads((out / "benchmark_manifest.json").read_text(encoding="utf-8"))
    assert manifest["artifact_kind"] == "research_benchmark_pack"
