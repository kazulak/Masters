from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from scripts import thesis_report
from quantum_bench.bench.config import load_suite
from quantum_bench.bench.simulation_backend_compare import _resource_guard_skip_reason
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
        "configs/suites/manual/thesis_full_state_correctness.yml",
        "configs/suites/manual/thesis_cpu_tn_quimb.yml",
        "configs/suites/manual/thesis_tn_paths_quantization.yml",
        "configs/suites/manual/thesis_planner_compare.yml",
        "configs/suites/manual/thesis_planner_sensitivity.yml",
        "configs/suites/manual/thesis_upmem_quantization_boundary.yml",
    ):
        suite = load_suite(ROOT / rel_path)
        assert suite["metadata"]["manual_invocation_required"] is True
        assert suite["cases"]
        assert suite["route_policy"]["routes"]


def test_thesis_manual_suites_cover_declared_size_targets() -> None:
    cpu_gpu = load_suite(ROOT / "configs/suites/manual/thesis_full_state_cpu_gpu.yml")
    correctness = load_suite(ROOT / "configs/suites/manual/thesis_full_state_correctness.yml")
    tn_quimb = load_suite(ROOT / "configs/suites/manual/thesis_cpu_tn_quimb.yml")
    tn_quant = load_suite(ROOT / "configs/suites/manual/thesis_tn_paths_quantization.yml")
    planner = load_suite(ROOT / "configs/suites/manual/thesis_planner_compare.yml")

    assert _sizes_for_family(cpu_gpu, "quest_qrng") == {8, 10, 12, 14, 16, 18, 20}
    assert _sizes_for_family(cpu_gpu, "quest_bv") == {8, 10, 12, 14, 16, 18, 20}
    assert _sizes_for_family(tn_quimb, "quest_qrng") == {8, 10, 12, 14, 16, 18, 20}
    assert _sizes_for_family(tn_quimb, "quest_bv") == {8, 10, 12, 14, 16, 18, 20}
    assert _sizes_for_family(tn_quant, "quest_qrng") == {8, 10, 12, 14, 16, 18, 20}
    assert _sizes_for_family(tn_quant, "quest_bv") == {8, 10, 12, 14, 16, 18, 20}
    assert _sizes_for_family(planner, "quest_qrng") == {8, 10, 12, 14, 16, 18, 20}
    assert _sizes_for_family(planner, "quest_bv") == {8, 10, 12, 14, 16, 18, 20}
    assert correctness["suite_id"] == "thesis_full_state_correctness"
    assert correctness["metadata"]["state_output_mode"] == "full_dump"
    assert tn_quimb["metadata"]["max_qubits"] == 20
    assert tn_quant["metadata"]["max_qubits"] == 20
    assert tn_quimb["warmups"] == 0
    assert tn_quant["warmups"] == 0
    assert tn_quimb["repeats"] == 3
    assert tn_quant["repeats"] == 1
    assert tn_quimb["memory_guard_gib"] == 12
    assert tn_quant["memory_guard_gib"] == 8
    quest_reference = next(route for route in tn_quimb["_route_configs"] if route["id"] == "quest_cpu_full_state_exact")
    assert quest_reference["options"]["max_output_qubits"] == 20
    assert quest_reference["options"]["max_output_amplitudes"] == 1_048_576
    quimb_route = next(route for route in tn_quimb["_route_configs"] if route["id"] == "quimb_tn_exact")
    replay_route = next(route for route in tn_quant["_route_configs"] if route["id"] == "cpu_tn_path_replay_float64")
    assert quimb_route["options"]["max_estimated_intermediate_bytes"] == 4_294_967_296
    assert replay_route["options"]["max_estimated_intermediate_bytes"] == 1_073_741_824


def test_thesis_tn_manual_suites_large_label_cases_build_taskgraphs() -> None:
    suites = [
        load_suite(ROOT / "configs/suites/manual/thesis_cpu_tn_quimb.yml"),
        load_suite(ROOT / "configs/suites/manual/thesis_tn_paths_quantization.yml"),
    ]

    for suite in suites:
        assert not any(case.get("case_skip_reason") == "internal_tensor_label_symbol_cap" for case in suite["cases"])
        for case in suite["cases"]:
            if case.get("case_skip_reason"):
                continue
            circuit = quest_compatible_circuit(case["circuit"]["name"], case["circuit"])
            network = build_tensor_network(circuit)
            graph = with_path_cost_summary(plan_task_graph_with_config(network, suite["planner"]))
            assert graph.tasks


def test_thesis_tn_suite_taskgraph_estimates_match_planner_and_pass_guards() -> None:
    tn_quimb = load_suite(ROOT / "configs/suites/manual/thesis_cpu_tn_quimb.yml")
    tn_quant = load_suite(ROOT / "configs/suites/manual/thesis_tn_paths_quantization.yml")
    quimb_routes = {route["id"]: route for route in tn_quimb["_route_configs"]}
    quant_routes = {route["id"]: route for route in tn_quant["_route_configs"]}

    for case in tn_quimb["cases"]:
        graph = _graph_for_case(case, tn_quimb)
        assert graph.path_summary.largest_intermediate is not None
        assert graph.path_summary.max_intermediate_bytes == graph.path_summary.largest_intermediate * 16
        assert _resource_guard_skip_reason(quimb_routes["quimb_tn_exact"], graph) is None
        assert _resource_guard_skip_reason(quimb_routes["quimb_tn_sliced_exact"], graph) is None

    for case in tn_quant["cases"]:
        graph = _graph_for_case(case, tn_quant)
        assert graph.path_summary.largest_intermediate is not None
        assert graph.path_summary.max_intermediate_bytes == graph.path_summary.largest_intermediate * 16
        assert _resource_guard_skip_reason(quant_routes["cpu_tn_path_replay_float64"], graph) is None
        assert _resource_guard_skip_reason(quant_routes["cpu_tn_path_replay_int8_quantized"], graph) is None


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
    assert result.metadata["contraction_plan_hash"] == graph.contraction_plan_hash
    assert result.metadata["plan_reused"] is True
    assert result.metadata["planning_in_timed_region"] is False


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
    assert result.metadata["contraction_plan_hash"] == graph.contraction_plan_hash


def test_thesis_report_uses_explicit_evidence_inputs(tmp_path: Path) -> None:
    full_state_evidence = tmp_path / "runs" / "evidence" / "unit_full_state" / "route" / "run"
    quimb_evidence = tmp_path / "runs" / "evidence" / "unit_quimb" / "route" / "run"
    quant_evidence = tmp_path / "runs" / "evidence" / "unit_quant" / "route" / "run"
    upmem_evidence = tmp_path / "runs" / "evidence" / "unit_upmem" / "route" / "run"
    write_jsonl(full_state_evidence / "normalized_records.jsonl", [
        _record("quest_bv_8q_thesis_gpu", "quest_cpu_full_state_exact", 0, target="cpu", compute=8.0),
        _record("quest_bv_8q_thesis_gpu", "quest_gpu_full_state_exact", 0, target="gpu", compute=2.0),
    ])
    write_jsonl(quimb_evidence / "normalized_records.jsonl", [
        _record("quest_bv_8q_thesis_tn", "quimb_tn_exact", 0, target="cpu", compute=1.5),
        _record("quest_bv_8q_thesis_tn", "quimb_tn_sliced_exact", 0, target="cpu", compute=1.8),
    ])
    write_jsonl(quant_evidence / "normalized_records.jsonl", [
        _record("quest_bv_8q_thesis_tn", "cpu_tn_path_replay_float64", 0, target="cpu", compute=3.0),
        _record("quest_bv_8q_thesis_tn", "cpu_tn_path_replay_int8_quantized", 0, target="cpu", compute=2.5),
    ])
    write_jsonl(upmem_evidence / "normalized_records.jsonl", [
        _record("bv_4q_thesis_upmem", "upmem_tn_sdk_simulator_quantized", 0, target="upmem", compute=1.0),
    ])
    out = tmp_path / "runs" / "comparisons" / "thesis" / "unit"

    exit_code = thesis_report.main([
        "--inputs",
        str(full_state_evidence),
        str(quimb_evidence),
        str(quant_evidence),
        str(upmem_evidence),
        "--out",
        str(out),
    ])

    assert exit_code == 0
    assert (out / "full_state_cpu_gpu_by_circuit.csv").exists()
    assert (out / "full_state_cpu_gpu_speedup_by_circuit_size.csv").exists()
    assert (out / "tn_path_comparison_by_circuit.csv").exists()
    assert (out / "tn_path_runtime_by_circuit_size.csv").exists()
    assert (out / "tn_quantization_comparison.csv").exists()
    assert (out / "tn_quantization_speedup_by_circuit_size.csv").exists()
    assert (out / "tn_quantization_error_by_circuit_size.csv").exists()
    assert (out / "upmem_boundary_quantization.csv").exists()
    assert (out / "upmem_accuracy_by_circuit_size.csv").exists()
    assert (out / "benchmark_summary.md").exists()
    manifest = json.loads((out / "thesis_report_manifest.json").read_text(encoding="utf-8"))
    assert manifest["artifact_kind"] == "thesis_comparison_report"
    assert manifest["input_count"] == 4
    assert "full_state_cpu_gpu_speedup_by_circuit_size.csv" in manifest["outputs"]
    assert "tn_quantization_speedup_by_circuit_size.csv" in manifest["outputs"]
    plot_manifest = json.loads((out / "plot_manifest.json").read_text(encoding="utf-8"))
    plot_names = {entry["plot"] for entry in plot_manifest["plots"]}
    assert "full_state_cpu_gpu_speedup_by_circuit_size.png" in plot_names
    assert "tn_quantization_runtime_by_circuit_size.png" in plot_names
    assert "upmem_accuracy_error_by_circuit_size.png" in plot_names
    assert all(entry["status"] != "skipped" for entry in plot_manifest["plots"])
    with (out / "full_state_cpu_gpu_speedup_by_circuit_size.csv").open("r", encoding="utf-8", newline="") as handle:
        summary_rows = list(csv.DictReader(handle))
    assert summary_rows[0]["case_family"] == "quest_bv"
    assert summary_rows[0]["n_qubits"] == "8"
    assert summary_rows[0]["compute_speedup_median_cpu_over_gpu"] == "4.0"
    with (out / "tn_quantization_comparison.csv").open("r", encoding="utf-8", newline="") as handle:
        quant_rows = list(csv.DictReader(handle))
    assert quant_rows[0]["comparison_scope"] == "same_route_family_cpu_diagnostic_path_replay"
    assert quant_rows[0]["contraction_execution_target"] == "cpu"
    assert quant_rows[0]["accelerator_kind"] == "none"
    assert quant_rows[0]["quantized_input_dtype"] == "int8_split_real_imag"
    assert quant_rows[0]["quantized_accumulator_dtype"] == "complex128"
    assert quant_rows[0]["compute_slowdown_quantized_over_unquantized"] == "0.8333333333333334"
    assert "CPU diagnostic replay" in quant_rows[0]["interpretation"]
    with (out / "tn_path_comparison_by_circuit.csv").open("r", encoding="utf-8", newline="") as handle:
        tn_rows = list(csv.DictReader(handle))
    quant_tn_row = next(row for row in tn_rows if row["route_id"] == "cpu_tn_path_replay_int8_quantized")
    assert quant_tn_row["thesis_route_label"] == "CPU diagnostic TN path replay int8-dequantized"
    assert quant_tn_row["contraction_execution_target"] == "cpu"
    assert quant_tn_row["max_abs_error"] == "0.02"
    with (out / "upmem_boundary_quantization.csv").open("r", encoding="utf-8", newline="") as handle:
        upmem_rows = list(csv.DictReader(handle))
    assert upmem_rows[0]["thesis_route_label"] == "UPMEM SDK simulator generic float32/no quantization"
    assert upmem_rows[0]["backend_family"] == "upmem_sdk"
    assert upmem_rows[0]["accelerator_kind"] == "upmem"
    assert upmem_rows[0]["policy"] == "generic-only"
    assert upmem_rows[0]["cpu_fallback_task_count"] == "0"
    assert upmem_rows[0]["upmem_task_count"] == "1"
    assert upmem_rows[0]["hardware_timing_available"] == "False"
    assert upmem_rows[0]["hardware_speedup_applicable"] == "False"
    assert upmem_rows[0]["max_abs_error"] == "0.0"
    summary = (out / "benchmark_summary.md").read_text(encoding="utf-8")
    assert "QuEST full-state GPU only" in summary
    assert "Full-state CPU/GPU runtime and speedup by circuit family and qubit size" in summary
    assert "CPU quantized replay rows are not UPMEM or native int8 kernel performance evidence" in summary


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


def _sizes_for_family(suite: dict, family_prefix: str) -> set[int]:
    sizes: set[int] = set()
    for case in suite["cases"]:
        case_id = str(case["case_id"])
        if not case_id.startswith(family_prefix):
            continue
        circuit = case["circuit"]
        sizes.add(int(circuit.get("n_qubits") or circuit.get("allocated_qubits")))
    return sizes


def _graph_for_case(case: dict, suite: dict):
    circuit = quest_compatible_circuit(case["circuit"]["name"], case["circuit"])
    network = build_tensor_network(circuit)
    return with_path_cost_summary(plan_task_graph_with_config(network, suite["planner"]))


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
