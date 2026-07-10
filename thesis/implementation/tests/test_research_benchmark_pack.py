from __future__ import annotations

import json
from pathlib import Path

from scripts import research_benchmark_pack as pack
from quantum_bench.bench.config import comparison_planner_configs, load_suite


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


def _generic_upmem_record(case_id: str, quantization_mode: str, *, total: float, compute: float, transfer: int) -> dict:
    float_mode = quantization_mode == "none"
    return {
        "schema_version": "benchmark_result_artifact_v1",
        "suite_id": "thesis_upmem_quantization_boundary",
        "case_id": case_id,
        "workload_id": case_id,
        "n_qubits": 7,
        "route_id": "upmem_tn_runtime",
        "backend_family": "upmem_sdk",
        "benchmark_role": "strict_upmem_sdk_simulator_generic",
        "kernel_family": "generic_loop_fallback",
        "execution_model": "tensor_network",
        "contraction_execution_target": "upmem",
        "upmem_execution_mode": "sdk_simulator",
        "policy": "generic-only",
        "quantization_mode": quantization_mode,
        "generic_only_all_tasks_used_generic_backend": True,
        "valid_primary_upmem_codepath_result": True,
        "dpu_program_invocations": 3,
        "upmem_program_executed": True,
        "cpu_fallback_used": False,
        "status": "completed",
        "validation_status": "passed",
        "repeat_id": 0,
        "total_wall_time_s": total,
        "simulation_compute_time_s": compute,
        "actual_transfer_bytes": transfer,
        "input_dtype_on_dpu": "float32" if float_mode else "int8",
        "native_unquantized_upmem_kernel_executed": float_mode,
        "hardware_speedup": "not_applicable",
        "hardware_speedup_applicable": False,
        "validation_error_metrics": {"max_abs_error": 0.0 if float_mode else 0.01, "l2_error": 0.0 if float_mode else 0.02},
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


def test_research_pack_rejects_dense_upmem_rows_and_accepts_strict_generic_rows() -> None:
    dense = _generic_upmem_record("qrng_7q_thesis_upmem_boundary", "per_task_input_quantize", total=2.0, compute=0.1, transfer=100)
    dense["policy"] = "dense-then-generic"
    dense["kernel_family"] = "dense_gemm"

    issues = pack._claim_guard_issues([dense])

    assert any("not generic-only" in issue for issue in issues)
    generic = _generic_upmem_record("qrng_7q_thesis_upmem_boundary", "per_task_input_quantize", total=2.0, compute=0.1, transfer=100)
    assert pack._claim_guard_issues([generic]) == []


def test_research_pack_separates_generic_quantization_modes_and_builds_attribution() -> None:
    float32 = _generic_upmem_record("qrng_7q_thesis_upmem_boundary", "none", total=4.0, compute=2.0, transfer=400)
    int8 = _generic_upmem_record("qrng_7q_thesis_upmem_boundary", "per_task_input_quantize", total=2.0, compute=1.0, transfer=100)

    stats = pack.per_case_route_stats([float32, int8])
    attribution = pack.upmem_quantization_attribution([float32, int8])

    assert len(stats) == 2
    assert {row["quantization_mode"] for row in stats} == {"none", "per_task_input_quantize"}
    assert len(attribution) == 1
    assert attribution[0]["same_route_comparison"] is True
    assert attribution[0]["route_runtime_ratio_none_over_quantized"] == 2.0
    assert attribution[0]["transfer_ratio_none_over_quantized"] == 4.0
    assert attribution[0]["native_unquantized_upmem_kernel_executed"] is True


def test_research_pack_quantization_attribution_rejects_different_routes_or_runs() -> None:
    float32 = _generic_upmem_record("qrng_7q_thesis_upmem_boundary", "none", total=4.0, compute=2.0, transfer=400)
    int8 = _generic_upmem_record("qrng_7q_thesis_upmem_boundary", "per_task_input_quantize", total=2.0, compute=1.0, transfer=100)
    float32["run_id"] = "run_a"
    int8["run_id"] = "run_a"
    int8["route_id"] = "another_upmem_route"

    assert pack.upmem_quantization_attribution([float32, int8]) == []

    int8["route_id"] = float32["route_id"]
    int8["run_id"] = "run_b"
    assert pack.upmem_quantization_attribution([float32, int8]) == []


def test_research_pack_preserves_generic_boundary_reason_from_record_notes() -> None:
    unsupported = _generic_upmem_record("qrng_8q_thesis_upmem_boundary", "none", total=0.0, compute=0.0, transfer=0)
    unsupported.update(
        {
            "status": "unsupported",
            "validation_status": "skipped",
            "unsupported_task_count": 1,
            "notes": '{"reason":"generic_feasibility_rank_cap_exceeded"}',
        }
    )

    rows = pack.unsupported_cases([unsupported])

    assert rows[0]["resource_skip_reason"] == "generic_feasibility_rank_cap_exceeded"


def test_research_pack_cpu_gpu_plot_rows_exclude_correctness_tier() -> None:
    performance = _record("quest_bv_10q_research_perf", "quest_cpu_full_state_exact", 0, target="cpu", total=10.0, compute=8.0)
    gpu_performance = _record("quest_bv_10q_research_perf", "quest_gpu_full_state_exact", 0, target="gpu", total=5.0, compute=2.0)
    correctness = dict(performance)
    correctness["case_id"] = "quest_bv_10q_research_correctness"
    correctness["performance_tier"] = False
    correctness["state_output_mode"] = "full_dump"
    correctness["validation_method"] = "full_statevector"
    gpu_correctness = dict(gpu_performance)
    gpu_correctness["case_id"] = correctness["case_id"]
    gpu_correctness["performance_tier"] = False
    gpu_correctness["state_output_mode"] = "full_dump"
    gpu_correctness["validation_method"] = "full_statevector"

    pairs = pack.paired_speedups([performance, gpu_performance, correctness, gpu_correctness])

    assert len(pairs) == 2
    assert len([row for row in pairs if row["performance_tier"]]) == 1


def test_research_pack_cpu_gpu_performance_summary_uses_repeat_medians() -> None:
    records = [
        _record("quest_bv_10q_research_perf", "quest_cpu_full_state_exact", 0, target="cpu", total=10.0, compute=8.0),
        _record("quest_bv_10q_research_perf", "quest_gpu_full_state_exact", 0, target="gpu", total=5.0, compute=2.0),
        _record("quest_bv_10q_research_perf", "quest_cpu_full_state_exact", 1, target="cpu", total=12.0, compute=10.0),
        _record("quest_bv_10q_research_perf", "quest_gpu_full_state_exact", 1, target="gpu", total=6.0, compute=2.5),
    ]

    summary = pack.cpu_gpu_performance_summary(pack.paired_speedups(records))

    assert len(summary) == 1
    assert summary[0]["matched_repeat_count"] == 2
    assert summary[0]["cpu_simulation_compute_time_s_median"] == 9.0
    assert summary[0]["gpu_simulation_compute_time_s_median"] == 2.25
    assert summary[0]["compute_speedup_cpu_over_gpu_median"] == 4.0


def test_research_pack_skipped_group_result_is_visible() -> None:
    result = pack._skipped_group_result("cpu_gpu", "hip_smoke_build_failed")

    assert result["returncode"] == 0
    assert result["skipped_group"] == "cpu_gpu"
    assert result["blocker_reason"] == "hip_smoke_build_failed"
    assert result["benchmark_rows_emitted"] is False


def test_research_pack_runs_upmem_boundary_through_strict_generic_mvp_command() -> None:
    argv = pack._research_suite_argv("upmem_boundary", pack.ROOT)

    assert argv[:2] == ["upmem-mvp-benchmark", "--suite"]
    assert any(item.endswith("thesis_upmem_quantization_boundary.yml") for item in argv)
    assert argv[argv.index("--policies") + 1] == "generic-only"
    assert argv[argv.index("--quantization-modes") + 1] == "none,per_task_input_quantize"
    assert "--execute-external" in argv


def test_research_suite_matrix_uses_six_families_and_seven_local_sizes() -> None:
    expected_families = {"QRNG", "BV", "XOR", "BB84", "EDC", "HS"}
    expected_sizes = {4, 6, 8, 10, 12, 14, 16}
    for suite_name in ("research_cpu_gpu.yml", "research_cpu_tn.yml", "research_planner_compare.yml"):
        suite = load_suite(pack.ROOT / "configs" / "suites" / "manual" / suite_name)
        families = {str(case["circuit"]["name"]) for case in suite["cases"]}
        sizes = {
            int(case["circuit"].get("n_qubits") or case["circuit"].get("allocated_qubits"))
            for case in suite["cases"]
        }
        assert families == expected_families
        assert sizes == expected_sizes
        assert len(suite["cases"]) == 42

    planner = load_suite(pack.RESEARCH_SUITES["planner_paths"])
    assert [item["optimize"] for item in comparison_planner_configs(planner)] == ["greedy", "auto"]
    assert pack._research_suite_argv("planner_paths", pack.ROOT)[:2] == ["compare-planners", "--suite"]


def test_research_pack_preserves_modeled_planner_candidates() -> None:
    records = [
        {
            "suite_id": "research_planner_compare",
            "case_id": "quest_bv_10q_planner",
            "n_qubits": 10,
            "route_id": "planner_candidate_model",
            "backend_id": "opt_einsum.greedy",
            "planner_id": "opt_einsum.greedy",
            "optimize_mode": "greedy",
            "contraction_plan_hash": "a" * 64,
            "planning_time_s": 0.01,
            "task_count": 12,
            "tn_estimated_flops": 1234,
            "tn_max_intermediate_bytes": 2048,
            "total_host_to_dpu_bytes": 100,
            "total_dpu_to_host_bytes": 50,
            "total_mram_to_wram_bytes": 500,
            "tiling_required_task_count": 1,
            "estimated_total_tile_count": 4,
            "estimated_max_parallel_tiles": 2,
            "upmem_pressure_score": 0.25,
            "upmem_rank": 1,
            "flop_rank": 2,
            "parallelism_evidence_type": "modeled",
            "execution_plan_executed": False,
        }
    ]

    rows = pack.planner_comparison(records)

    assert len(rows) == 1
    assert rows[0]["benchmark_n_qubits"] == 10
    assert rows[0]["planner_id"] == "opt_einsum.greedy"
    assert rows[0]["parallelism_evidence_type"] == "modeled"
    assert rows[0]["execution_plan_executed"] is False


def test_research_pack_same_plan_cpu_upmem_requires_matching_hash() -> None:
    plan_hash = "b" * 64
    cpu = {
        "suite_id": "thesis_upmem_quantization_boundary",
        "case_id": "qrng_4q_thesis_upmem",
        "route_id": "cpu_tn_einsum_exact",
        "contraction_execution_target": "cpu",
        "status": "completed",
        "simulation_compute_time_s": 0.1,
        "contraction_plan_hash": plan_hash,
    }
    upmem = _generic_upmem_record(
        "qrng_4q_thesis_upmem",
        "per_task_input_quantize",
        total=2.0,
        compute=1.0,
        transfer=100,
    )
    upmem["contraction_plan_hash"] = plan_hash

    rows = pack.same_plan_execution([cpu, upmem])

    assert len(rows) == 1
    assert rows[0]["same_plan_verified"] is True
    assert rows[0]["hardware_speedup_applicable"] is False
    upmem["contraction_plan_hash"] = "c" * 64
    assert pack.same_plan_execution([cpu, upmem]) == []


def test_research_pack_builds_full_state_tn_ratio_without_calling_it_speedup() -> None:
    common = {
        "schema_version": pack.SCHEMA_VERSION,
        "suite_id": "research_cpu_tn",
        "case_id": "quest_bv_10q_research_tn",
        "case_family": "bv",
        "benchmark_n_qubits": 10,
        "validation_passed_count": 3,
    }
    stats = [
        {**common, "route_id": "quest_cpu_full_state_exact", "simulation_compute_time_s_median": 1.0},
        {**common, "route_id": "quimb_tn_exact", "simulation_compute_time_s_median": 2.0},
        {**common, "route_id": "quimb_tn_sliced_exact", "simulation_compute_time_s_median": 3.0},
    ]

    rows = pack.full_state_tn_comparison(stats)

    assert len(rows) == 1
    assert rows[0]["quimb_unsliced_time_over_quest_time"] == 2.0
    assert rows[0]["quimb_sliced_time_over_unsliced_time"] == 1.5
    assert all("speedup" not in key for key in rows[0])


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
    assert (out / "full_state_tn_comparison.csv").exists()
    assert b"\r\n" not in (out / "per_case_route_stats.csv").read_bytes()
    assert (out / "plot_manifest.json").exists()
    summary = (out / "benchmark_summary.md").read_text(encoding="utf-8")
    assert "Next UPMEM Implementation Readiness" in summary
    manifest = json.loads((out / "benchmark_manifest.json").read_text(encoding="utf-8"))
    assert manifest["artifact_kind"] == "research_benchmark_pack"
    assert not (tmp_path / "latest").exists()
