"""Physical one-DPU TaskGraph correctness benchmark.

This command is deliberately narrower than the SDK-simulator TaskGraph route.
It proves that each contraction in a circuit-derived TaskGraph can consume the
prior physical result and produce the next physical result on one DPU.  Native
source is built once for a run, but the initial protocol allocates, loads, and
releases the DPU for each logical task.  The resulting timings are retained as
bring-up diagnostics only and are never speedup evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import shutil
import socket
from typing import Any, Mapping

import numpy as np

from quantum_bench.bench.reporting import write_normalized_records, write_run_manifest
from quantum_bench.bench.run_dirs import (
    EVIDENCE_ARTIFACT_KIND,
    create_run_dir,
    sanitize,
)
from quantum_bench.circuits import load_circuit, manifest
from quantum_bench.core.jsonio import write_json, write_jsonl
from quantum_bench.core.records import JsonDict
from quantum_bench.environment import capture_environment
from quantum_bench.targets.upmem.hardware_session import (
    HardwareSessionBuild,
    build_hardware_session,
)
from quantum_bench.targets.upmem.hardware_taskgraph import (
    HARDWARE_TASKGRAPH_ROUTE_ID,
    HardwareTaskGraphSuite,
    hardware_taskgraph_profile_metadata,
    load_hardware_taskgraph_suite,
    validate_hardware_taskgraph_execution_request,
)
from quantum_bench.targets.upmem.hardware_taskgraph_runtime import (
    execute_hardware_taskgraph_runtime,
)
from quantum_bench.tn import (
    build_execution_bundle,
    build_tensor_network,
    execute_task_sequence_np_einsum,
    plan_task_graph_with_config,
    with_execution_identity,
)


UPMEM_HARDWARE_TASKGRAPH_BENCHMARK_SCHEMA_VERSION = (
    "upmem_hardware_taskgraph_benchmark_v1"
)
UPMEM_HARDWARE_TASKGRAPH_PLAN_SCHEMA_VERSION = "upmem_hardware_taskgraph_plan_v1"


@dataclass(frozen=True)
class UpmemHardwareTaskGraphPlanResult:
    plan_dir: Path
    summary_path: Path
    status: str


@dataclass(frozen=True)
class UpmemHardwareTaskGraphResult:
    run_dir: Path
    summary_path: Path
    status: str
    row_count: int


def prepare_upmem_hardware_taskgraph(
    root_dir: Path,
    *,
    suite_path: Path,
    build: bool = False,
    environment: Mapping[str, str] | None = None,
) -> UpmemHardwareTaskGraphPlanResult:
    """Validate and materialize a physical TaskGraph plan without a DPU call."""

    suite = load_hardware_taskgraph_suite(suite_path)
    env = dict(os.environ if environment is None else environment)
    plan_dir = _unique_dir(root_dir / "build" / "upmem_hardware_taskgraph_plan")
    plan_dir.mkdir(parents=True, exist_ok=False)
    config_dir = plan_dir / "config"
    config_dir.mkdir()
    shutil.copy2(suite.suite_path, config_dir / "resolved_suite.yml")
    write_json(
        config_dir / "hardware_profile.json",
        hardware_taskgraph_profile_metadata(suite.profile),
    )
    write_json(plan_dir / "environment.json", capture_environment(root_dir))

    case_rows: list[JsonDict] = []
    status = "prepared"
    failure_stage: str | None = None
    for case in suite.suite["cases"]:
        try:
            prepared = _prepare_case(
                root_dir,
                plan_dir / "cases" / sanitize(str(case["case_id"])),
                suite,
                case,
            )
            case_rows.append(_prepared_case_row(case, prepared))
        except Exception as exc:
            status = "failed"
            failure_stage = "hardware_profile_violation"
            case_rows.append(
                {"case_id": case.get("case_id"), "status": "failed", "reason": str(exc)}
            )
            break

    native_build: JsonDict = {"attempted": False, "status": "not_requested"}
    if status == "prepared" and build:
        try:
            native = build_hardware_session(
                root_dir,
                plan_dir / "native_session",
                profile=suite.profile,
                environment=env,
            )
            native_build = _native_build_metadata(native, plan_dir)
        except Exception as exc:
            status = "failed"
            failure_stage = _failure_stage(str(exc), default="native_build_failed")
            native_build = {"attempted": True, "status": "failed", "reason": str(exc)}

    summary_path = plan_dir / "upmem_hardware_taskgraph_plan.json"
    write_json(
        summary_path,
        {
            "schema_version": UPMEM_HARDWARE_TASKGRAPH_PLAN_SCHEMA_VERSION,
            "status": status,
            "failure_stage": failure_stage,
            "suite_id": suite.suite["suite_id"],
            "suite_path": str(suite.suite_path),
            "profile": hardware_taskgraph_profile_metadata(suite.profile),
            "prepared_cases": case_rows,
            "native_build": native_build,
            "dpu_allocation_attempted": False,
            "dpu_launch_attempted": False,
            "notes": [
                "Preparation lowers circuit workloads into TaskGraphs, writes execution bundles, and optionally builds isolated native source.",
                "Preparation never allocates or launches a DPU and never creates thesis evidence.",
            ],
        },
    )
    return UpmemHardwareTaskGraphPlanResult(plan_dir, summary_path, status)


def run_upmem_hardware_taskgraph(
    root_dir: Path,
    *,
    suite_path: Path,
    environment: Mapping[str, str] | None = None,
) -> UpmemHardwareTaskGraphResult:
    """Run the guarded physical TaskGraph correctness suite without fallback."""

    env = dict(os.environ if environment is None else environment)
    validate_hardware_taskgraph_execution_request(execute=True, environment=env)
    suite = load_hardware_taskgraph_suite(suite_path)
    run_dir = create_run_dir(
        root_dir,
        str(suite.suite["suite_id"]),
        artifact_kind=EVIDENCE_ARTIFACT_KIND,
        route_label="upmem_hw_taskgraph",
    )
    shutil.copy2(suite.suite_path, run_dir / "config" / "resolved_suite.yml")
    write_json(
        run_dir / "config" / "hardware_profile.json",
        hardware_taskgraph_profile_metadata(suite.profile),
    )
    write_json(run_dir / "environment.json", capture_environment(root_dir))
    run_manifest = write_run_manifest(
        run_dir,
        run_kind="upmem_hardware_taskgraph_correctness",
        suite_id=str(suite.suite["suite_id"]),
        suite_path=str(suite.suite_path),
        artifact_kind=EVIDENCE_ARTIFACT_KIND,
        route_label="upmem_hw_taskgraph",
        route_id=HARDWARE_TASKGRAPH_ROUTE_ID,
        backend_id=suite.profile.backend_id,
        execution_scope="physical_single_dpu_taskgraph_correctness",
        evidence_type="physical_hardware_functionality",
        upmem_execution_mode="sdk_hardware_single_dpu_taskgraph",
        artifact_retention="full",
        summary="upmem_hardware_taskgraph_summary.json",
        root_dir=root_dir,
    )

    try:
        native_build = build_hardware_session(
            root_dir,
            run_dir / "native_session",
            profile=suite.profile,
            environment=env,
        )
    except Exception as exc:
        summary_path = _write_failed_run_summary(run_dir, suite, str(exc))
        run_manifest.update(
            {
                "summary": summary_path.name,
                "hardware_available": "not_verified_by_execution",
            }
        )
        write_json(run_dir / "run_manifest.json", run_manifest)
        write_normalized_records(
            run_dir, [_build_failure_record(suite, str(exc), run_manifest)]
        )
        return UpmemHardwareTaskGraphResult(run_dir, summary_path, "failed", 1)

    records: list[JsonDict] = []
    warmup_rows: list[JsonDict] = []
    case_statuses: list[JsonDict] = []
    stop_after_failure = False
    for case in suite.suite["cases"]:
        case_id = str(case["case_id"])
        if stop_after_failure:
            case_statuses.append(
                {"case_id": case_id, "status": "not_attempted_after_prior_failure"}
            )
            continue
        try:
            prepared = _prepare_case(
                root_dir, run_dir / "cases" / sanitize(case_id), suite, case
            )
        except Exception as exc:
            failure = _build_failure_record(suite, str(exc), run_manifest, case=case)
            records.append(failure)
            case_statuses.append(
                {
                    "case_id": case_id,
                    "status": "failed",
                    "failure_stage": failure["failure_stage"],
                }
            )
            stop_after_failure = True
            continue

        case_records, case_warmups, case_status = _run_case(
            root_dir=root_dir,
            run_dir=run_dir,
            suite=suite,
            case=case,
            prepared=prepared,
            native_build=native_build,
            environment=env,
            source_commit=run_manifest.get("benchmark_source_commit"),
        )
        records.extend(case_records)
        warmup_rows.extend(case_warmups)
        case_statuses.append(case_status)
        if case_status["status"] != "passed":
            stop_after_failure = True

    completed = bool(records) and all(
        record.get("status") == "completed" for record in records
    )
    summary = {
        "schema_version": UPMEM_HARDWARE_TASKGRAPH_BENCHMARK_SCHEMA_VERSION,
        "status": "completed" if completed else "failed",
        "suite_id": suite.suite["suite_id"],
        "route_id": HARDWARE_TASKGRAPH_ROUTE_ID,
        "backend_id": suite.profile.backend_id,
        "hardware_profile_version": suite.profile.version,
        "row_count": len(records),
        "warmup_count": len(warmup_rows),
        "case_statuses": case_statuses,
        "native_build": _native_build_metadata(native_build, run_dir),
        "strict_scope": {
            "physical_circuit_taskgraph": True,
            "requested_dpu_count": 1,
            "tasklets_per_dpu": 1,
            "synchronous_execution": True,
            "native_build_reused": True,
            "logical_task_session_only": True,
            "hardware_speedup_applicable": False,
            "hardware_timing_available": False,
            "no_simulator_fallback": True,
            "no_cpu_fallback": True,
        },
        "normalized_records": "normalized_records.jsonl",
        "warmup_summary": "warmups.jsonl",
    }
    summary_path = run_dir / "upmem_hardware_taskgraph_summary.json"
    write_json(summary_path, summary)
    write_jsonl(run_dir / "warmups.jsonl", warmup_rows)
    write_normalized_records(run_dir, records)
    run_manifest.update(
        {
            "summary": summary_path.name,
            "upmem_sdk_available": "verified_by_execution"
            if completed
            else "not_verified_by_execution",
            "hardware_available": "verified_by_execution"
            if completed
            else "not_verified_by_execution",
        }
    )
    write_json(run_dir / "run_manifest.json", run_manifest)
    return UpmemHardwareTaskGraphResult(
        run_dir, summary_path, str(summary["status"]), len(records)
    )


def _prepare_case(
    root_dir: Path,
    case_dir: Path,
    suite: HardwareTaskGraphSuite,
    case: Mapping[str, Any],
) -> JsonDict:
    case_dir.mkdir(parents=True, exist_ok=True)
    circuit = load_circuit(dict(case), root_dir)
    network = build_tensor_network(circuit)
    graph = with_execution_identity(
        plan_task_graph_with_config(network, suite.suite["planner"])
    )
    reference_output, reference_metrics = execute_task_sequence_np_einsum(
        graph, network
    )
    bundle = build_execution_bundle(
        graph, case_id=str(case["case_id"]), suite_id=str(suite.suite["suite_id"])
    )
    bundle_path = case_dir / "execution_bundle.json"
    write_json(bundle_path, bundle)
    reference_path = case_dir / "reference_final_tensor.npy"
    np.save(reference_path, np.asarray(reference_output), allow_pickle=False)
    write_json(
        case_dir / "case_preparation.json",
        {
            "case_id": case["case_id"],
            "hardware_numeric_coverage": case.get("hardware_numeric_coverage"),
            "circuit": manifest(circuit),
            "task_count": len(graph.tasks),
            "reference_execution": reference_metrics,
            "execution_bundle": bundle_path.name,
            "reference_final_tensor": reference_path.name,
        },
    )
    return {
        "circuit": circuit,
        "network": network,
        "graph": graph,
        "reference_output": np.asarray(reference_output),
        "bundle_path": bundle_path,
        "reference_path": reference_path,
        "reference_metrics": reference_metrics,
    }


def _prepared_case_row(
    case: Mapping[str, Any], prepared: Mapping[str, Any]
) -> JsonDict:
    graph = prepared["graph"]
    return {
        "case_id": case["case_id"],
        "status": "prepared",
        "n_qubits": prepared["circuit"].n_qubits,
        "hardware_numeric_coverage": case.get("hardware_numeric_coverage"),
        "task_count": len(graph.tasks),
        "contraction_plan_hash": graph.contraction_plan_hash,
        "execution_bundle": str(prepared["bundle_path"].name),
    }


def _run_case(
    *,
    root_dir: Path,
    run_dir: Path,
    suite: HardwareTaskGraphSuite,
    case: Mapping[str, Any],
    prepared: Mapping[str, Any],
    native_build: HardwareSessionBuild,
    environment: Mapping[str, str],
    source_commit: object,
) -> tuple[list[JsonDict], list[JsonDict], JsonDict]:
    records: list[JsonDict] = []
    warmups: list[JsonDict] = []
    case_id = str(case["case_id"])
    modes = tuple(suite.profile.numeric_modes)
    for warmup_id in range(int(suite.suite["warmups"])):
        for quantization_mode in modes:
            result = _run_one(
                run_dir=run_dir,
                root_dir=root_dir,
                suite=suite,
                case=case,
                prepared=prepared,
                native_build=native_build,
                environment=environment,
                quantization_mode=quantization_mode,
                phase="warmup",
                iteration=warmup_id,
            )
            warmups.append(_warmup_row(case, quantization_mode, warmup_id, result))
            if result.summary.get("status") != "completed":
                return (
                    records,
                    warmups,
                    {
                        "case_id": case_id,
                        "status": "failed",
                        "phase": "warmup",
                        "attempted_warmups": warmup_id + 1,
                        "failure_stage": result.summary.get("failure_stage")
                        or result.reason,
                    },
                )

    for repeat_id in range(int(suite.suite["repeats"])):
        for quantization_mode in modes:
            result = _run_one(
                run_dir=run_dir,
                root_dir=root_dir,
                suite=suite,
                case=case,
                prepared=prepared,
                native_build=native_build,
                environment=environment,
                quantization_mode=quantization_mode,
                phase="repeat",
                iteration=repeat_id,
            )
            record = _normalized_record(
                root_dir=root_dir,
                run_dir=run_dir,
                suite=suite,
                case=case,
                prepared=prepared,
                result=result,
                repeat_id=repeat_id,
                quantization_mode=quantization_mode,
                source_commit=source_commit,
            )
            records.append(record)
            if record["status"] != "completed":
                return (
                    records,
                    warmups,
                    {
                        "case_id": case_id,
                        "status": "failed",
                        "phase": "repeat",
                        "attempted_repeats": repeat_id + 1,
                        "failure_stage": record.get("failure_stage"),
                    },
                )
    return (
        records,
        warmups,
        {
            "case_id": case_id,
            "status": "passed",
            "attempted_repeats": len(records) // len(modes),
        },
    )


def _run_one(
    *,
    root_dir: Path,
    run_dir: Path,
    suite: HardwareTaskGraphSuite,
    case: Mapping[str, Any],
    prepared: Mapping[str, Any],
    native_build: HardwareSessionBuild,
    environment: Mapping[str, str],
    quantization_mode: str,
    phase: str,
    iteration: int,
):
    work_dir = (
        native_build.session_root
        / "logical_runs"
        / sanitize(str(case["case_id"]))
        / phase
        / f"{iteration:02d}_{quantization_mode}"
    )
    result = execute_hardware_taskgraph_runtime(
        root_dir=root_dir,
        work_dir=work_dir,
        graph=prepared["graph"],
        network=prepared["network"],
        case_id=str(case["case_id"]),
        quantization_mode=quantization_mode,
        profile=suite.profile,
        environment=environment,
        reference_output=prepared["reference_output"],
        native_build=native_build,
    )
    artifact_dir = (
        run_dir
        / "cases"
        / sanitize(str(case["case_id"]))
        / "physical_taskgraph"
        / phase
        / f"{iteration:02d}_{quantization_mode}"
    )
    artifact_dir.mkdir(parents=True, exist_ok=False)
    summary_path = artifact_dir / "runtime_summary.json"
    write_json(summary_path, result.summary)
    output_path: Path | None = None
    if result.output is not None:
        output_path = artifact_dir / "final_output.npy"
        np.save(output_path, np.asarray(result.output), allow_pickle=False)
    result.summary["runtime_summary_artifact"] = str(summary_path.relative_to(run_dir))
    result.summary["final_output_artifact"] = (
        str(output_path.relative_to(run_dir)) if output_path is not None else None
    )
    write_json(summary_path, result.summary)
    return result


def _warmup_row(
    case: Mapping[str, Any], quantization_mode: str, warmup_id: int, result: Any
) -> JsonDict:
    return {
        "case_id": case["case_id"],
        "quantization_mode": quantization_mode,
        "warmup_id": warmup_id,
        "status": result.summary.get("status"),
        "reason": result.reason,
        "timing_is_bringup_only": True,
        "hardware_speedup_applicable": False,
        "summary": result.summary,
    }


def _normalized_record(
    *,
    root_dir: Path,
    run_dir: Path,
    suite: HardwareTaskGraphSuite,
    case: Mapping[str, Any],
    prepared: Mapping[str, Any],
    result: Any,
    repeat_id: int,
    quantization_mode: str,
    source_commit: object,
) -> JsonDict:
    summary = dict(result.summary)
    circuit = prepared["circuit"]
    output = result.output
    metric_status = str(summary.get("status") or "failed")
    validation = (
        summary.get("final_validation")
        if isinstance(summary.get("final_validation"), Mapping)
        else {}
    )
    full_precision = (
        summary.get("full_precision_accuracy")
        if isinstance(summary.get("full_precision_accuracy"), Mapping)
        else {}
    )
    task_metrics = (
        summary.get("task_metrics")
        if isinstance(summary.get("task_metrics"), list)
        else []
    )
    exact_integer_match = (
        bool(task_metrics)
        and quantization_mode == "per_task_input_quantize"
        and all(
            (metric.get("validation") or {}).get("max_abs_error") == 0.0
            for metric in task_metrics
            if isinstance(metric, Mapping)
        )
    )
    return {
        "schema_version": UPMEM_HARDWARE_TASKGRAPH_BENCHMARK_SCHEMA_VERSION,
        "source_artifact": str(
            (run_dir / "upmem_hardware_taskgraph_summary.json").relative_to(run_dir)
        ),
        "run_id": run_dir.name,
        "suite_id": suite.suite["suite_id"],
        "case_id": case["case_id"],
        "workload_id": case["workload_id"],
        "repeat_id": repeat_id,
        "benchmark_n_qubits": circuit.n_qubits,
        "actual_n_qubits": circuit.n_qubits,
        "actual_n_qubits_source": "circuit_spec",
        "route_id": HARDWARE_TASKGRAPH_ROUTE_ID,
        "backend_id": suite.profile.backend_id,
        "backend_family": "upmem_sdk",
        "benchmark_role": "physical_taskgraph_correctness",
        "route_role_description": "one-DPU circuit-derived TaskGraph correctness route; native build is reused, but each logical contraction opens a physical session",
        "route_limitation_scope": "bring-up correctness only; no hardware speedup, energy, scheduling, or multi-DPU claim",
        "execution_model": "tensor_network",
        "execution_plan_kind": "taskgraph_serial_physical_one_dpu",
        "execution_plan_executed": metric_status == "completed",
        "parallelism_mode": "sequential",
        "parallelism_evidence_type": "executed",
        "contraction_execution_target": "upmem",
        "execution_target": "upmem",
        "accelerator_kind": "upmem",
        "upmem_execution_mode": "sdk_hardware_single_dpu_taskgraph",
        "execution_backend": suite.profile.backend_id,
        "hardware_execution": summary.get("hardware_execution") is True,
        "hardware_kernel_executed": summary.get("hardware_kernel_executed") is True,
        "simulator_kernel_executed": False,
        "cpu_fallback_used": False,
        "hardware_functionality_evidence": True,
        "hardware_timing_available": False,
        "hardware_speedup_applicable": False,
        "speedup_claim_allowed": False,
        "target_requested": "hardware",
        "target_observed": summary.get("target_observed"),
        "hardware_profile_version": suite.profile.version,
        "session_protocol": suite.profile.session_protocol,
        "session_scope": "logical_task",
        "physical_session_build_reused": True,
        "requested_dpu_count": suite.profile.requested_dpu_count,
        "allocated_dpu_count": summary.get("allocated_dpu_count"),
        "tasklets_per_dpu": suite.profile.tasklets_per_dpu,
        "hardware_allocation_verified": (
            summary.get("hardware_execution") is True
            and summary.get("allocated_dpu_count") == suite.profile.requested_dpu_count
        ),
        "hardware_release_verified": summary.get("hardware_release_verified"),
        "kernel_family": "generic_loop_fallback",
        "generic_kernel_strategy": "mram_resident_output_tiled_v1",
        "native_max_rank": suite.profile.max_rank,
        "native_max_tensor_elements": suite.profile.max_tensor_elements,
        "generic_output_tile_elements": suite.profile.output_tile_elements,
        "mram_resident_operands": True,
        "wram_output_tiled": True,
        "quantization_mode": quantization_mode,
        "input_dtype_on_dpu": "float32" if quantization_mode == "none" else "int8",
        "accumulator_dtype_on_dpu": "float32"
        if quantization_mode == "none"
        else "int32",
        "complex_policy": suite.profile.complex_policy,
        "hardware_numeric_coverage": case.get("hardware_numeric_coverage"),
        "task_count": summary.get("task_count"),
        "upmem_task_count": summary.get("task_count"),
        "validated_task_count": summary.get("validated_task_count"),
        "unsupported_task_count": summary.get("unsupported_task_count"),
        "split_complex_component_count": sum(
            int(metric.get("split_complex_component_count") or 0)
            for metric in task_metrics
            if isinstance(metric, Mapping)
        ),
        "validation_method": (
            "physical_native_float32_task_reference_and_final_cpu_reference"
            if quantization_mode == "none"
            else "physical_native_int8_task_reference_with_final_quantization_error"
        ),
        "validation_status": summary.get("validation_status"),
        "validation_max_abs_error": summary.get("max_abs_error"),
        "max_abs_error": summary.get("max_abs_error"),
        "l2_error": summary.get("l2_error"),
        "full_precision_max_abs_error": full_precision.get("max_abs_error"),
        "quantization_max_abs_error": summary.get("quantization_max_abs_error"),
        "exact_integer_match": exact_integer_match
        if quantization_mode == "per_task_input_quantize"
        else None,
        "output_contract": "final_tensor",
        "output_contract_label": "physical_taskgraph_final_tensor",
        "output_contract_is_exact": False,
        "exact_output_comparable": False,
        "full_statevector_validation_available": False,
        "performance_tier": False,
        "status": metric_status,
        "reason": summary.get("reason"),
        "failure_stage": _failure_stage(str(summary.get("reason") or ""), default=None),
        "application_visible_h2d_bytes": summary.get("application_visible_h2d_bytes"),
        "application_visible_d2h_bytes": summary.get("application_visible_d2h_bytes"),
        "application_visible_transfer_bytes": summary.get(
            "application_visible_transfer_bytes"
        ),
        "actual_h2d_bytes": summary.get("actual_h2d_bytes"),
        "actual_d2h_bytes": summary.get("actual_d2h_bytes"),
        "actual_transfer_bytes": summary.get("actual_transfer_bytes"),
        "allocation_time_s": summary.get("allocation_time_s"),
        "binary_load_time_s": summary.get("binary_load_time_s"),
        "h2d_time_s": summary.get("h2d_time_s"),
        "kernel_time_s": summary.get("kernel_time_s"),
        "d2h_time_s": summary.get("d2h_time_s"),
        "total_quantization_time_s": summary.get("total_quantization_time_s"),
        "total_dequantization_time_s": summary.get("total_dequantization_time_s"),
        "total_build_time_s": summary.get("total_build_time_s"),
        "total_route_time_s": summary.get("total_route_time_s"),
        "timing_scope": summary.get("timing_scope"),
        "timing_is_bringup_only": True,
        "source_commit": source_commit,
        "hostname": socket.gethostname(),
        "sdk_metadata": summary.get("sdk_tools"),
        "host_binary_hash": summary.get("host_binary_hash"),
        "dpu_binary_hash": summary.get("dpu_binary_hash"),
        "input_hash": _network_input_hash(prepared["network"]),
        "output_hash": _array_hash(output) if output is not None else None,
        "execution_bundle_artifact": str(prepared["bundle_path"].relative_to(run_dir)),
        "circuit_semantics_hash": summary.get("circuit_semantics_hash"),
        "tensor_network_hash": summary.get("tensor_network_hash"),
        "contraction_plan_hash": summary.get("contraction_plan_hash"),
        "contraction_path_structure_hash": summary.get(
            "contraction_path_structure_hash"
        ),
        "plan_reused": summary.get("plan_reused"),
        "planning_in_timed_region": False,
        "executor_config_hash": summary.get("executor_config_hash"),
        "task_metrics_artifact": summary.get("runtime_summary_artifact"),
        "final_output_artifact": summary.get("final_output_artifact"),
        "notes": {
            "final_native_validation": validation,
            "full_precision_accuracy": full_precision,
            "hardware_functionality_evidence": True,
            "timing_claim_boundary": "logical-task allocation/load/release timing is bring-up-only and excluded from speedup analysis",
        },
    }


def _build_failure_record(
    suite: HardwareTaskGraphSuite,
    error: str,
    run_manifest: Mapping[str, Any],
    *,
    case: Mapping[str, Any] | None = None,
) -> JsonDict:
    return {
        "schema_version": UPMEM_HARDWARE_TASKGRAPH_BENCHMARK_SCHEMA_VERSION,
        "suite_id": suite.suite["suite_id"],
        "case_id": case.get("case_id") if case else "native_build",
        "workload_id": case.get("workload_id") if case else "native_build",
        "route_id": HARDWARE_TASKGRAPH_ROUTE_ID,
        "backend_id": suite.profile.backend_id,
        "backend_family": "upmem_sdk",
        "benchmark_role": "physical_taskgraph_correctness",
        "contraction_execution_target": "upmem",
        "upmem_execution_mode": "sdk_hardware_single_dpu_taskgraph",
        "target_requested": "hardware",
        "target_observed": None,
        "hardware_execution": False,
        "hardware_kernel_executed": False,
        "simulator_kernel_executed": False,
        "cpu_fallback_used": False,
        "hardware_speedup_applicable": False,
        "timing_is_bringup_only": True,
        "status": "failed",
        "reason": error,
        "failure_stage": _failure_stage(error, default="native_build_failed"),
        "source_commit": run_manifest.get("benchmark_source_commit"),
    }


def _write_failed_run_summary(
    run_dir: Path, suite: HardwareTaskGraphSuite, error: str
) -> Path:
    summary_path = run_dir / "upmem_hardware_taskgraph_summary.json"
    write_json(
        summary_path,
        {
            "schema_version": UPMEM_HARDWARE_TASKGRAPH_BENCHMARK_SCHEMA_VERSION,
            "status": "failed",
            "suite_id": suite.suite["suite_id"],
            "route_id": HARDWARE_TASKGRAPH_ROUTE_ID,
            "backend_id": suite.profile.backend_id,
            "failure_stage": _failure_stage(error, default="native_build_failed"),
            "reason": error,
            "hardware_speedup_applicable": False,
            "normalized_records": "normalized_records.jsonl",
        },
    )
    return summary_path


def _native_build_metadata(build: HardwareSessionBuild, root: Path) -> JsonDict:
    return {
        "attempted": True,
        "status": "passed",
        "source_tree_hash": build.source_tree_hash,
        "host_binary_hash": build.host_binary_hash,
        "dpu_binary_hash": build.dpu_binary_hash,
        "build_time_s": build.build_time_s,
        "build_command": list(build.build_command),
        "sdk_tools": build.sdk_tools,
        "session_root": str(build.session_root.relative_to(root))
        if build.session_root.is_relative_to(root)
        else str(build.session_root),
    }


def _network_input_hash(network: Any) -> str:
    digest = hashlib.sha256()
    for tensor in network.tensors:
        digest.update(tensor.spec.id.encode("utf-8"))
        digest.update(_array_hash(np.asarray(tensor.array)).encode("ascii"))
    return digest.hexdigest()


def _array_hash(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(np.asarray(array))
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(repr(tuple(int(dim) for dim in contiguous.shape)).encode("ascii"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _failure_stage(error: str, *, default: str | None) -> str | None:
    if "final_validation_failed" in error or "task_validation_failed" in error:
        return "output_validation_failed"
    known = (
        "hardware_opt_in_missing",
        "hardware_profile_violation",
        "sdk_discovery_failed",
        "native_build_failed",
        "hardware_allocation_failed",
        "binary_load_failed",
        "argument_transfer_failed",
        "operand_transfer_failed",
        "kernel_launch_failed",
        "kernel_timeout",
        "result_transfer_failed",
        "output_manifest_failed",
        "output_validation_failed",
        "hardware_release_failed",
    )
    for stage in known:
        if stage in error:
            return stage
    return default


def _unique_dir(parent: Path) -> Path:
    from datetime import datetime

    parent.mkdir(parents=True, exist_ok=True)
    base = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    candidate = parent / base
    suffix = 1
    while candidate.exists():
        candidate = parent / f"{base}_{suffix:02d}"
        suffix += 1
    return candidate
