from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from quantum_bench.bench.reporting import write_normalized_records, write_run_manifest
from quantum_bench.circuits import builtin_circuit
from quantum_bench.core.records import (
    CircuitSpec,
    ContractionTask,
    PathSummary,
    TaskGraph,
    TensorNetworkSpec,
    TensorSpec,
    TensorValue,
)
from quantum_bench.tn import (
    build_tensor_network,
    plan_task_graph,
    with_execution_identity,
)
from quantum_bench.tn.network import TensorNetworkValue
from quantum_bench.targets.upmem.hardware_taskgraph_resident import (
    RESIDENT_CONTROL_H2D_BYTES_PER_LAUNCH,
    RESIDENT_DESCRIPTOR_CONTROL_BYTES,
    build_resident_graph_package,
    load_hardware_taskgraph_resident_suite,
)


ROOT = Path(__file__).resolve().parents[1]
RESIDENT_SUITE_PATH = ROOT / "configs" / "suites" / "upmem_hardware_taskgraph_resident_path_quantization.yml"


@dataclass(frozen=True)
class GraphCase:
    graph: TaskGraph
    network: TensorNetworkValue


def minimal_real_graph() -> GraphCase:
    network = build_tensor_network(builtin_circuit("bell_2q"))
    return GraphCase(network=network, graph=plan_task_graph(network))


def split_complex_graph() -> GraphCase:
    left = np.array([[1.0 + 1.0j, 0.0], [0.0, 1.0 - 1.0j]], dtype=np.complex128)
    right = np.array([[1.0, 0.0 + 1.0j], [1.0 - 1.0j, 1.0]], dtype=np.complex128)
    circuit = CircuitSpec("split_complex", 0, (), {"kind": "fixture"})
    left_spec = TensorSpec("left", (0, 1), left.shape, "dense", dtype="complex128")
    right_spec = TensorSpec("right", (1, 2), right.shape, "dense", dtype="complex128")
    network_spec = TensorNetworkSpec(circuit, (left_spec, right_spec), (0, 2), "ab,bc->ac")
    task = ContractionTask(
        id="task_complex",
        input_tensor_ids=("left", "right"),
        output_tensor_id="out",
        dependencies=(),
        index_expression="ab,bc->ac",
        input_shapes=(left.shape, right.shape),
        output_shape=(2, 2),
        left_labels=(0, 1),
        right_labels=(1, 2),
        contracted_labels=(1,),
        output_labels=(0, 2),
        gemm_m=2,
        gemm_k=2,
        gemm_n=2,
        structure="dense",
        estimated_flops=16,
        estimated_bytes=0,
    )
    graph = with_execution_identity(
        TaskGraph(
            network_spec,
            (task,),
            ((0, 1),),
            PathSummary("fixture", "greedy", 1, 1, None, None, "fixture"),
            0.0,
        )
    )
    network = TensorNetworkValue(
        network_spec,
        [TensorValue(left_spec, left), TensorValue(right_spec, right)],
    )
    return GraphCase(network=network, graph=graph)


def contraction_task(
    task_id: str = "task",
    *,
    shape: tuple[int, int, int] = (2, 3, 4),
    structure: str = "generic",
) -> ContractionTask:
    m, k, n = shape
    return ContractionTask(
        id=task_id,
        input_tensor_ids=(f"{task_id}_left", f"{task_id}_right"),
        output_tensor_id=f"{task_id}_out",
        dependencies=(),
        index_expression="ab,bc->ac",
        input_shapes=((m, k), (k, n)),
        output_shape=(m, n),
        left_labels=(0, 1),
        right_labels=(1, 2),
        contracted_labels=(1,),
        output_labels=(0, 2),
        gemm_m=m,
        gemm_k=k,
        gemm_n=n,
        structure=structure,
        estimated_flops=2 * m * k * n,
        estimated_bytes=m * k + k * n + m * n,
    )


def dense_task(
    task_id: str,
    m: int,
    k: int,
    n: int,
    *,
    structure: str = "dense",
) -> ContractionTask:
    return contraction_task(task_id, shape=(m, k, n), structure=structure)


def resident_package_fixture(case: GraphCase, root: Path) -> tuple[Any, dict[str, Any]]:
    package = build_resident_graph_package(
        case.graph,
        case.network,
        case_id="fixture",
        suite_id="fixture_resident",
        quantization_mode="none",
    )
    binary = root / "dpu_resident"
    binary.write_bytes(b"resident-fixture-binary")
    written = package.write(root, dpu_binary=binary, request_id="fixture-request")
    manifest = json.loads(written.manifest_path.read_text(encoding="utf-8"))
    return written, manifest


def valid_resident_response(manifest: dict[str, Any], **updates: Any) -> dict[str, Any]:
    operation_count = int(manifest["component_operation_count"])
    initial_h2d = int(manifest["initial_h2d_bytes"])
    descriptor_h2d = int(manifest["descriptor_h2d_bytes"])
    control_h2d = RESIDENT_DESCRIPTOR_CONTROL_BYTES + operation_count * RESIDENT_CONTROL_H2D_BYTES_PER_LAUNCH
    final_d2h = int(manifest["final_d2h_bytes"])
    response: dict[str, Any] = {
        "schema_version": "generic_loop_resident_graph_session_v1",
        "manifest_kind": "resident_graph_response",
        "route_id": "upmem_tn_hardware_taskgraph_resident",
        "backend_id": "upmem_sdk_hardware_taskgraph_resident",
        "hardware_profile_version": "hardware_taskgraph_single_dpu_mram_resident_v1",
        "target_requested": "hardware",
        "target_observed": "hardware",
        "sdk_allocation_profile": "backend=hw",
        "sdk_allocation_profile_verified": True,
        "session_protocol": "generic_loop_resident_graph_session_v1",
        "quantization_mode": manifest["quantization_mode"],
        "status": "completed",
        "failure_stage": None,
        "requested_dpus": 1,
        "allocated_dpus": 1,
        "tasklets": 1,
        "graph_request_count": 1,
        "native_launch_count": operation_count,
        "native_task_count": operation_count,
        "allocation_count": 1,
        "hardware_allocation_verified": True,
        "hardware_execution": True,
        "hardware_kernel_executed": True,
        "native_execution": True,
        "native_hardware_backend": True,
        "hardware_backend_verified": True,
        "simulator_kernel_executed": False,
        "cpu_fallback_used": False,
        "hardware_release_verified": True,
        "release_confirmed": True,
        "physical_dependency_chain_verified": True,
        "hardware_timing_available": True,
        "persistent_session_reused": False,
        "resident_slots_persist_for_graph": True,
        "final_output_only_d2h": True,
        "physical_bus_bytes_available": False,
        "intermediate_h2d_bytes": 0,
        "intermediate_d2h_bytes": 0,
        "initial_h2d_bytes": initial_h2d,
        "descriptor_h2d_bytes": descriptor_h2d,
        "control_h2d_bytes": control_h2d,
        "final_d2h_bytes": final_d2h,
        "actual_h2d_bytes": initial_h2d + descriptor_h2d + control_h2d,
        "actual_d2h_bytes": final_d2h,
        "actual_transfer_bytes": initial_h2d + descriptor_h2d + control_h2d + final_d2h,
        "final_outputs": [dict(item, status="completed") for item in manifest["final_outputs"]],
    }
    response.update(
        {
            key: 0.0
            for key in (
                "package_parse_time_s",
                "allocation_time_s",
                "binary_load_time_s",
                "initial_h2d_time_s",
                "descriptor_h2d_time_s",
                "control_h2d_time_s",
                "kernel_time_s",
                "final_d2h_time_s",
                "output_write_time_s",
                "release_time_s",
                "steady_state_graph_execution_s",
            )
        }
    )
    response.update(updates)
    return response


def _cpu_gpu_record(
    case_id: str,
    route_id: str,
    repeat_id: int,
    *,
    n_qubits: int,
    total_s: float,
    compute_s: float,
    verified: bool = True,
    validation_status: str = "passed",
) -> dict[str, Any]:
    is_gpu = route_id == "quest_gpu_full_state_exact"
    return {
        "schema_version": "benchmark_result_artifact_v1",
        "run_id": "fixture-run",
        "suite_id": "fixture_cpu_gpu",
        "case_id": case_id,
        "workload_id": case_id,
        "n_qubits": n_qubits,
        "max_qubits": n_qubits,
        "route_id": route_id,
        "backend_id": route_id,
        "backend_family": "quest",
        "benchmark_role": "serious_gpu_full_state_baseline" if is_gpu else "serious_full_state_baseline",
        "kernel_family": "full_state_vector",
        "execution_model": "full_state",
        "execution_target": "gpu" if is_gpu else "cpu",
        "contraction_execution_target": "gpu" if is_gpu else "cpu",
        "accelerator_kind": "amd_gpu" if is_gpu else "none",
        "gpu_backend_verified": bool(is_gpu and verified),
        "gpu_program_executed": bool(is_gpu and verified),
        "gpu_device_name": "fixture-gpu" if is_gpu and verified else None,
        "gpu_runtime_stack": "amd_rocm" if is_gpu and verified else None,
        "state_output_mode": "full_dump",
        "output_contract": "statevector",
        "output_contract_is_exact": True,
        "exact_output_comparable": True,
        "full_statevector_validation_available": True,
        "status": "completed",
        "validation_status": validation_status,
        "validation_method": "full_statevector",
        "performance_tier": False,
        "repeat_id": repeat_id,
        "measured_repeat_count": 2,
        "total_wall_time_s": total_s,
        "simulation_compute_time_s": compute_s,
        "kernel_time_s": compute_s,
        "timing_scope": "end_to_end_and_compute",
        "energy_joules": None,
        "energy_source": "unavailable",
        "energy_measurement_status": "unavailable",
        "hardware_speedup": "not_applicable",
        "hardware_speedup_applicable": False,
        "cpu_fallback_used": False,
        "validation_error_metrics": {"max_abs_error": 0.0, "l2_error": 0.0},
    }


def cpu_gpu_pair_records() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for n_qubits, factor in ((4, 1.0), (6, 1.5)):
        case_id = f"qrng_{n_qubits}q"
        for repeat_id in (0, 1):
            records.extend(
                (
                    _cpu_gpu_record(case_id, "quest_cpu_full_state_exact", repeat_id, n_qubits=n_qubits, total_s=factor * (4.0 + repeat_id), compute_s=factor * (2.0 + repeat_id)),
                    _cpu_gpu_record(case_id, "quest_gpu_full_state_exact", repeat_id, n_qubits=n_qubits, total_s=factor * (1.0 + repeat_id / 2), compute_s=factor * (0.5 + repeat_id / 4)),
                )
            )
    return records


def _generic_upmem_record(
    case_id: str,
    quantization_mode: str,
    *,
    total_s: float,
    compute_s: float,
    plan_hash: str = "p" * 64,
    status: str = "completed",
    validation_status: str = "passed",
    unsupported_reason: str | None = None,
) -> dict[str, Any]:
    quantized = quantization_mode != "none"
    record: dict[str, Any] = {
        "schema_version": "benchmark_result_artifact_v1",
        "run_id": "fixture-run",
        "suite_id": "fixture_upmem",
        "case_id": case_id,
        "workload_id": case_id,
        "n_qubits": 4,
        "route_id": "upmem_tn_runtime",
        "backend_id": "upmem_sdk_simulator_generic_loop",
        "backend_family": "upmem_sdk",
        "benchmark_role": "strict_upmem_sdk_simulator_generic",
        "kernel_family": "generic_loop_fallback",
        "execution_model": "tensor_network",
        "contraction_execution_target": "upmem",
        "upmem_execution_mode": "sdk_simulator",
        "execution_scope": "full_taskgraph",
        "policy": "generic-only",
        "quantization_mode": quantization_mode,
        "generic_only_all_tasks_used_generic_backend": True,
        "valid_primary_upmem_codepath_result": status == "completed",
        "dpu_program_invocations": 3 if status == "completed" else 0,
        "dpu_program_executed_all_tasks": status == "completed",
        "upmem_program_executed": status == "completed",
        "cpu_fallback_used": False,
        "status": status,
        "validation_status": validation_status,
        "repeat_id": 0,
        "total_wall_time_s": total_s,
        "simulation_compute_time_s": compute_s,
        "actual_h2d_bytes": 75 if status == "completed" else 0,
        "actual_d2h_bytes": 25 if status == "completed" else 0,
        "actual_transfer_bytes": 100 if status == "completed" else 0,
        "actual_transfer_bytes_invariant": "passed" if status == "completed" else None,
        "transfer_accounting_scope": "application_visible_sdk_recorded",
        "input_dtype_on_dpu": "int8" if quantized else "float32",
        "native_unquantized_upmem_kernel_executed": not quantized and status == "completed",
        "hardware_speedup": "not_applicable",
        "hardware_speedup_applicable": False,
        "validation_error_metrics": {"max_abs_error": 0.01 if quantized else 0.0, "l2_error": 0.02 if quantized else 0.0},
        "contraction_plan_hash": plan_hash,
        "full_precision_max_abs_error": 0.01 if quantized else 0.0,
        "full_precision_l2_error": 0.02 if quantized else 0.0,
        "full_precision_reference_kind": "cpu_exact_taskgraph_full_precision",
        "native_sdk_control_path": True,
        "simplepim_api_used": False,
    }
    if unsupported_reason:
        record.update(
            {
                "status": "unsupported",
                "validation_status": "skipped",
                "valid_primary_upmem_codepath_result": False,
                "resource_skip_reason": unsupported_reason,
                "unsupported_task_count": 1,
            }
        )
    return record


def tn_upmem_pair_records() -> list[dict[str, Any]]:
    plan_hash = "q" * 64
    cpu = {
        "schema_version": "benchmark_result_artifact_v1",
        "run_id": "fixture-run",
        "suite_id": "fixture_upmem",
        "case_id": "qrng_4q",
        "route_id": "cpu_tn_einsum_exact",
        "backend_id": "cpu_tn_einsum_exact",
        "backend_family": "cpu",
        "benchmark_role": "internal_debug_baseline",
        "kernel_family": "einsum_contraction",
        "execution_model": "tensor_network",
        "contraction_execution_target": "cpu",
        "status": "completed",
        "validation_status": "passed",
        "repeat_id": 0,
        "simulation_compute_time_s": 0.5,
        "total_wall_time_s": 0.6,
        "contraction_plan_hash": plan_hash,
        "hardware_speedup": "not_applicable",
        "hardware_speedup_applicable": False,
    }
    return [
        cpu,
        _generic_upmem_record("qrng_4q", "none", total_s=2.0, compute_s=1.0, plan_hash=plan_hash),
        _generic_upmem_record("qrng_4q", "per_task_input_quantize", total_s=1.6, compute_s=0.8, plan_hash=plan_hash),
        _generic_upmem_record("qrng_8q", "none", total_s=0.0, compute_s=0.0, status="unsupported", validation_status="skipped", unsupported_reason="rank_cap_exceeded"),
    ]


def planner_evidence_records() -> list[dict[str, Any]]:
    base = {
        "schema_version": "benchmark_result_artifact_v1",
        "suite_id": "fixture_planner",
        "case_id": "planner_fixture",
        "route_id": "planner_candidate_model",
        "n_qubits": 4,
        "parallelism_evidence_type": "modeled",
        "execution_plan_executed": False,
        "pim_weight_profile": "balanced_literature_informed",
        "pim_objective_version": "upmem_path_cost_v2",
        "score_model": "upmem_path_cost_v2",
        "planner_config_hash": "config-hash",
        "pim_execution_policy": {"policy_id": "generic_single_dpu_split_complex_v2"},
        "pim_normalization": {"normalization_id": "fixed_log1p_generic_budgets_v2"},
    }
    return [
        base | {"planner_id": "opt_einsum.greedy", "pim_selected": False, "pim_feasible": True, "pim_objective_score": 0.5},
        base | {"planner_id": "custom_upmem.greedy", "pim_selected": True, "pim_feasible": True, "pim_objective_score": 0.25},
    ]


def hardware_evidence_records() -> list[dict[str, Any]]:
    row = {
        "schema_version": "benchmark_result_artifact_v1",
        "suite_id": "fixture_hardware",
        "case_id": "hardware_fixture",
        "route_id": "upmem_tn_hardware_taskgraph_resident",
        "backend_id": "upmem_sdk_hardware_taskgraph_resident",
        "backend_family": "upmem_sdk",
        "benchmark_role": "physical_upmem_taskgraph",
        "kernel_family": "generic_loop_fallback",
        "execution_model": "tensor_network",
        "contraction_execution_target": "upmem",
        "upmem_execution_mode": "sdk_hardware_taskgraph_resident",
        "target_requested": "hardware",
        "target_observed": "hardware",
        "hardware_execution": True,
        "hardware_kernel_executed": True,
        "native_execution": True,
        "native_kernel_executed": True,
        "native_hardware_backend": True,
        "hardware_backend_verified": True,
        "hardware_allocation_verified": True,
        "hardware_release_verified": True,
        "release_confirmed": True,
        "hardware_profile_version": "hardware_taskgraph_single_dpu_mram_resident_v1",
        "session_protocol": "generic_loop_resident_graph_session_v1",
        "host_binary_hash": "host" * 16,
        "dpu_binary_hash": "dpu" * 21 + "d",
        "native_source_tree_hash": "source" * 10 + "so",
        "simulator_kernel_executed": False,
        "cpu_fallback_used": False,
        "requested_dpu_count": 1,
        "allocated_dpu_count": 1,
        "tasklets_per_dpu": 1,
        "allocation_count": 1,
        "task_count": 1,
        "validated_task_count": 1,
        "multi_dpu_execution": False,
        "physical_dependency_chain_verified": True,
        "persistent_session_reused": True,
        "session_scope": "case_benchmark_block",
        "path_variant_id": "resident_full_graph",
        "quantization_mode": "none",
        "input_dtype_on_dpu": "float32",
        "repeat_id": 0,
        "steady_state_graph_execution_s": 0.25,
        "status": "completed",
        "validation_status": "passed",
        "exact_integer_match": True,
        "hardware_speedup_applicable": False,
        "actual_h2d_bytes": 80,
        "actual_d2h_bytes": 16,
        "actual_transfer_bytes": 96,
        "actual_transfer_bytes_invariant": "passed",
        "timing_scope": "one_dpu_mram_resident_full_taskgraph_v1",
        "timing_is_bringup_only": False,
        "hardware_timing_available": True,
        "contraction_plan_hash": "h" * 64,
    }
    return [row]


def record_with_updates(record: dict[str, Any], **updates: Any) -> dict[str, Any]:
    result = copy.deepcopy(record)
    result.update(updates)
    return result


def write_evidence_run(root: Path, records: list[dict[str, Any]], *, suite_id: str = "fixture_suite") -> Path:
    run_dir = root / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    write_run_manifest(
        run_dir,
        run_kind="fixture_evidence",
        suite_id=suite_id,
        suite_path="fixture.yml",
        artifact_retention="compact",
        normalized_records="normalized_records.jsonl",
        summary="fixture_summary.json",
        root_dir=root,
    )
    write_normalized_records(run_dir, records)
    return run_dir


def resident_suite() -> Any:
    return load_hardware_taskgraph_resident_suite(RESIDENT_SUITE_PATH)


ResidentResponseFactory = Callable[[dict[str, Any]], dict[str, Any]]
