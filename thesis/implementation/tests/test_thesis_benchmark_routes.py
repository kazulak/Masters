from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts import thesis_report
from quantum_bench.bench.config import load_suite
from quantum_bench.core.jsonio import write_jsonl
from quantum_bench.core.records import BenchmarkContext
from quantum_bench.circuits import quest_compatible_circuit
from quantum_bench.providers import route_registry
from quantum_bench.providers.exact_tn.cpu_path_replay import CpuTnPathReplayFloat64Route, CpuTnPathReplayInt8QuantizedRoute
from quantum_bench.tn import build_tensor_network, execute_task_sequence_np_einsum, plan_task_graph_with_config, with_path_cost_summary


ROOT = Path(__file__).resolve().parents[1]


def test_thesis_manual_suites_load() -> None:
    for rel_path in (
        "configs/suites/manual/thesis_full_state_cpu_gpu.yml",
        "configs/suites/manual/thesis_tn_paths_quantization.yml",
        "configs/suites/manual/thesis_upmem_quantization_boundary.yml",
    ):
        suite = load_suite(ROOT / rel_path)
        assert suite["metadata"]["manual_invocation_required"] is True
        assert suite["cases"]
        assert suite["route_policy"]["routes"]


def test_path_replay_routes_are_registered() -> None:
    routes = route_registry(ROOT)

    assert "cpu_tn_path_replay_float64" in routes
    assert "cpu_tn_path_replay_int8_quantized" in routes
    assert routes["cpu_tn_path_replay_float64"].identity.role == "diagnostic_path_replay_baseline"
    assert routes["cpu_tn_path_replay_int8_quantized"].identity.role == "diagnostic_quantized_path_replay"


def test_path_replay_float64_matches_internal_task_sequence(tmp_path: Path) -> None:
    circuit = quest_compatible_circuit("XOR", {"n_qubits": 4})
    network = build_tensor_network(circuit)
    graph = with_path_cost_summary(plan_task_graph_with_config(network, {"engine": "opt_einsum", "optimize": "greedy"}))
    context = _context(tmp_path, "cpu_tn_path_replay_float64")

    route = CpuTnPathReplayFloat64Route()
    result = route.execute(route.prepare(graph, network, context), context)
    expected, _ = execute_task_sequence_np_einsum(graph, network)

    assert result.status == "passed"
    assert np.allclose(result.output.array, expected)
    assert result.metadata["path_replay_execution"] is True
    assert result.metadata["quantization_mode"] == "none"
    assert result.metadata["per_contraction_quantization"] is False


def test_path_replay_quantized_records_per_contraction_metadata(tmp_path: Path) -> None:
    circuit = quest_compatible_circuit("BV", {"n_qubits": 4})
    network = build_tensor_network(circuit)
    graph = with_path_cost_summary(plan_task_graph_with_config(network, {"engine": "opt_einsum", "optimize": "greedy"}))
    context = _context(tmp_path, "cpu_tn_path_replay_int8_quantized")

    route = CpuTnPathReplayInt8QuantizedRoute()
    result = route.execute(route.prepare(graph, network, context), context)

    assert result.status == "passed"
    assert result.metadata["path_replay_execution"] is True
    assert result.metadata["quantization_mode"] == "per_contraction_input_quantize"
    assert result.metadata["per_contraction_quantization"] is True
    assert result.metadata["input_dtype"] == "int8_split_real_imag"
    assert result.metadata["accumulator_dtype"] == "complex128"
    assert result.metadata["total_quantization_time_s"] >= 0.0
    assert result.metadata["total_dequantization_time_s"] >= 0.0
    assert result.metadata["quantized_replay_numeric_contract"] == "int8_operand_quantize_dequantize_then_complex128_einsum"


def test_thesis_report_uses_explicit_evidence_inputs(tmp_path: Path) -> None:
    evidence = tmp_path / "runs" / "evidence" / "unit" / "route" / "run"
    records = [
        _record("quest_bv_8q_thesis_gpu", "quest_cpu_full_state_exact", 0, target="cpu", compute=8.0),
        _record("quest_bv_8q_thesis_gpu", "quest_gpu_full_state_exact", 0, target="gpu", compute=2.0),
        _record("quest_bv_8q_thesis_tn", "cpu_tn_path_replay_float64", 0, target="cpu", compute=3.0),
        _record("quest_bv_8q_thesis_tn", "cpu_tn_path_replay_int8_quantized", 0, target="cpu", compute=2.5),
        _record("bv_4q_thesis_upmem", "upmem_tn_sdk_simulator_quantized", 0, target="upmem", compute=1.0),
    ]
    write_jsonl(evidence / "normalized_records.jsonl", records)
    out = tmp_path / "runs" / "comparisons" / "thesis" / "unit"

    exit_code = thesis_report.main(["--inputs", str(evidence), "--out", str(out)])

    assert exit_code == 0
    assert (out / "full_state_cpu_gpu_by_circuit.csv").exists()
    assert (out / "tn_path_comparison_by_circuit.csv").exists()
    assert (out / "tn_quantization_comparison.csv").exists()
    assert (out / "upmem_boundary_quantization.csv").exists()
    assert (out / "benchmark_summary.md").exists()
    manifest = json.loads((out / "thesis_report_manifest.json").read_text(encoding="utf-8"))
    assert manifest["artifact_kind"] == "thesis_comparison_report"
    assert manifest["input_count"] == 1
    assert "QuEST full-state GPU only" in (out / "benchmark_summary.md").read_text(encoding="utf-8")


def _context(tmp_path: Path, route_id: str) -> BenchmarkContext:
    return BenchmarkContext(
        root_dir=tmp_path,
        run_dir=tmp_path / "run",
        suite={"suite_id": "unit"},
        case={"case_id": "unit"},
        route_config={"id": route_id, "options": {"path_strategy": "greedy"}},
        repeat_id=0,
        tolerances={},
        timeout_s=None,
        memory_guard_gib=None,
    )


def _record(case_id: str, route_id: str, repeat_id: int, *, target: str, compute: float) -> dict:
    is_gpu = route_id == "quest_gpu_full_state_exact"
    is_quant = route_id == "cpu_tn_path_replay_int8_quantized"
    return {
        "schema_version": "benchmark_result_artifact_v1",
        "suite_id": "unit",
        "case_id": case_id,
        "workload_id": case_id,
        "route_id": route_id,
        "backend_id": route_id,
        "backend_family": "quest" if "quest" in route_id else "cpu",
        "benchmark_role": "serious_gpu_full_state_baseline" if is_gpu else "diagnostic_quantized_path_replay" if is_quant else "diagnostic_path_replay_baseline",
        "kernel_family": "full_state_vector" if "quest" in route_id else "einsum_contraction",
        "execution_model": "full_state" if "quest" in route_id else "tensor_network",
        "contraction_execution_target": target,
        "accelerator_kind": "amd_gpu" if is_gpu else "upmem" if target == "upmem" else "none",
        "gpu_backend_verified": is_gpu,
        "gpu_program_executed": is_gpu,
        "gpu_device_name": "AMD Radeon RX 6600 (gfx1032)" if is_gpu else None,
        "upmem_execution_mode": "sdk_simulator" if target == "upmem" else None,
        "execution_backend": "upmem_sdk" if target == "upmem" else None,
        "upmem_program_executed": target == "upmem",
        "dpu_program_invocations": 1 if target == "upmem" else None,
        "hardware_execution": False,
        "hardware_speedup_applicable": False,
        "cpu_fallback_used": False,
        "state_output_mode": "none" if "quest" in route_id else "full_dump",
        "validation_method": "native_status_gate_counts" if "quest" in route_id else "full_statevector",
        "performance_tier": "quest" in route_id,
        "validation_status": "passed_native_status" if "quest" in route_id else "passed",
        "status": "completed",
        "repeat_id": repeat_id,
        "total_wall_time_s": compute + 0.5,
        "simulation_compute_time_s": compute,
        "quantization_mode": "per_contraction_input_quantize" if is_quant else "none",
        "per_contraction_quantization": is_quant,
        "path_replay_execution": route_id.startswith("cpu_tn_path_replay"),
        "path_strategy": "greedy" if route_id.startswith("cpu_tn_path_replay") else None,
        "input_dtype": "int8_split_real_imag" if is_quant else "complex128" if route_id.startswith("cpu_tn_path_replay") else None,
        "accumulator_dtype": "complex128" if route_id.startswith("cpu_tn_path_replay") else None,
        "quantized_replay_numeric_contract": "int8_operand_quantize_dequantize_then_complex128_einsum" if is_quant else None,
        "quantization_max_abs_error": 0.01 if is_quant else None,
        "quantization_l2_error": 0.1 if is_quant else None,
        "validation_error_metrics": {"max_abs_error": 0.02 if is_quant else 0.0, "l2_error": 0.2 if is_quant else 0.0},
    }
