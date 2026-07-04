from __future__ import annotations

import csv
import json
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from quantum_bench.core.jsonio import read_jsonl, write_json
from quantum_bench.core.records import JsonDict, to_jsonable
from quantum_bench.bench.run_dirs import is_within_evidence_root


RESULT_ARTIFACT_SCHEMA_VERSION = "benchmark_result_artifact_v1"
COMPARE_RESULTS_SCHEMA_VERSION = "compare_results_v1"
COMPARISON_MANIFEST_SCHEMA_VERSION = "comparison_manifest_v1"

KERNEL_FAMILIES = (
    "dense_gemm",
    "einsum_contraction",
    "external_tn_contraction",
    "full_state_vector",
    "generic_loop_fallback",
    "quantum_structured",
    "sparse_or_zero_heavy",
    "communication_collective",
    "cpu_reference_only",
    "upmem_taskgraph_quantized",
    "unsupported",
)

RESULT_FIELDS = [
    "schema_version",
    "source_artifact",
    "run_id",
    "timestamp",
    "suite_id",
    "case_id",
    "workload_id",
    "route_id",
    "backend_id",
    "backend_family",
    "benchmark_role",
    "route_role_description",
    "route_limitation_scope",
    "kernel_family",
    "execution_model",
    "parallelism_mode",
    "parallelism_evidence_type",
    "execution_plan_kind",
    "execution_plan_executed",
    "slicing_enabled",
    "frontier_scheduler_enabled",
    "intra_contraction_parallelism_source",
    "modeled_parallelism_available",
    "output_kind",
    "comparison_output_kind",
    "state_output_mode",
    "output_contract",
    "output_contract_label",
    "output_contract_is_exact",
    "performance_tier",
    "exact_output_comparable",
    "full_statevector_validation_available",
    "execution_target",
    "contraction_execution_target",
    "accelerator_kind",
    "gpu_backend_verified",
    "gpu_program_executed",
    "gpu_device_name",
    "gpu_runtime_stack",
    "upmem_execution_mode",
    "execution_backend",
    "hardware_execution",
    "hardware_timing_available",
    "hardware_speedup_applicable",
    "cpu_fallback_used",
    "dpu_program_invocations",
    "upmem_program_executed",
    "native_sdk_control_path",
    "simplepim_api_used",
    "quantization_mode",
    "input_dtype_on_dpu",
    "accumulator_dtype_on_dpu",
    "scaling_applied",
    "unquantized_mode_kind",
    "actual_h2d_bytes",
    "actual_d2h_bytes",
    "actual_transfer_bytes",
    "total_quantization_time_s",
    "total_dequantization_time_s",
    "execution_scope",
    "simulator_or_hardware",
    "status",
    "validation_status",
    "repeat_id",
    "measured_repeat_count",
    "task_count",
    "validated_task_count",
    "unsupported_task_count",
    "setup_time_s",
    "circuit_lowering_time_s",
    "planning_time_s",
    "lowering_time_s",
    "data_transfer_time_s",
    "simulation_compute_time_s",
    "validation_time_s",
    "output_materialization_time_s",
    "timing_scope",
    "gpu_synchronized",
    "validation_method",
    "native_process_wall_time_s",
    "quest_simulation_compute_time_s",
    "state_dump_requested",
    "state_dump_time_s",
    "repeat_layers",
    "energy_joules",
    "energy_source",
    "energy_measurement_status",
    "expected_runtime_class",
    "expected_memory_class",
    "intended_use",
    "max_qubits",
    "manual_invocation_required",
    "expected_risk",
    "known_heavy_backends",
    "resource_guard_status",
    "resource_skip_reason",
    "total_wall_time_s",
    "kernel_time_s",
    "total_wall_time_s_median",
    "total_wall_time_s_min",
    "total_wall_time_s_mean",
    "total_wall_time_s_std",
    "simulation_compute_time_s_median",
    "simulation_compute_time_s_min",
    "simulation_compute_time_s_mean",
    "simulation_compute_time_s_std",
    "host_transfer_time_s",
    "build_time_s",
    "launch_overhead_s",
    "simulator_relative_time",
    "hardware_speedup",
    "validation_error_metrics",
    "statevector_bytes",
    "tn_task_count",
    "tn_max_intermediate_bytes",
    "tn_estimated_flops",
    "tn_estimated_bytes",
    "dependency_metadata",
    "notes",
    "warnings",
]

SUMMARY_FIELDS = [
    "schema_version",
    "kernel_family",
    "record_count",
    "task_count",
    "validated_task_count",
    "unsupported_task_count",
    "measured_record_count",
    "simulator_record_count",
    "hardware_record_count",
]

CPU_GPU_SPEEDUP_PAIR_FIELDS = [
    "schema_version",
    "case_id",
    "case_family",
    "n_qubits",
    "repeat_id",
    "cpu_route_id",
    "gpu_route_id",
    "state_output_mode",
    "validation_method",
    "performance_tier",
    "exact_output_comparable",
    "full_statevector_validation_available",
    "timing_scope",
    "cpu_total_wall_time_s",
    "gpu_total_wall_time_s",
    "total_wall_speedup",
    "cpu_simulation_compute_time_s",
    "gpu_simulation_compute_time_s",
    "compute_speedup",
    "validation_status",
    "gpu_device_name",
]

CPU_GPU_SPEEDUP_SUMMARY_FIELDS = [
    "schema_version",
    "case_family",
    "n_qubits",
    "matched_repeat_count",
    "state_output_mode",
    "validation_method",
    "performance_tier",
    "timing_scope",
    "cpu_total_wall_time_s_median",
    "gpu_total_wall_time_s_median",
    "total_wall_speedup_median",
    "total_wall_speedup_mean",
    "total_wall_speedup_min",
    "total_wall_speedup_max",
    "cpu_simulation_compute_time_s_median",
    "gpu_simulation_compute_time_s_median",
    "compute_speedup_median",
    "compute_speedup_mean",
    "compute_speedup_min",
    "compute_speedup_max",
    "gpu_device_name",
    "validation_status",
]

CPU_GPU_SPEEDUP_SKIPPED_FIELDS = [
    "schema_version",
    "case_id",
    "repeat_id",
    "reason",
    "cpu_present",
    "gpu_present",
]


@dataclass(frozen=True)
class CompareResultsResult:
    run_dir: Path
    artifact_path: Path
    csv_path: Path
    summary_path: Path
    record_count: int
    manifest_path: Path | None = None


def normalized_task_result_from_summary(summary: JsonDict, *, source_artifact: str | None = None) -> JsonDict:
    schema = str(summary.get("schema_version", ""))
    if schema == "dense_task_bridge_v1":
        return normalize_parallelism_metadata(_dense_task_bridge_record(summary, source_artifact=source_artifact))
    if schema == "generic_task_bridge_v1":
        return normalize_parallelism_metadata(_generic_task_bridge_record(summary, source_artifact=source_artifact))
    raise ValueError(f"unsupported one-task summary schema_version: {schema}")


def normalized_upmem_taskgraph_result_from_summary(summary: JsonDict, *, source_artifact: str | None = None) -> JsonDict:
    return normalize_parallelism_metadata(_upmem_taskgraph_runtime_record(summary, source_artifact=source_artifact))


def normalized_upmem_taskgraph_records_from_summary(summary: JsonDict, *, source_artifact: str | None = None) -> list[JsonDict]:
    return [normalize_parallelism_metadata(record) for record in _upmem_taskgraph_runtime_records(summary, source_artifact=source_artifact)]


def load_result_records(inputs: Iterable[Path]) -> list[JsonDict]:
    records: list[JsonDict] = []
    for input_path in inputs:
        path = Path(input_path)
        if path.is_dir():
            canonical = path / "normalized_records.jsonl"
            if canonical.exists():
                records.extend(normalize_parallelism_metadata(record) for record in read_jsonl(canonical))
                continue
            for artifact in _discover_artifacts(path):
                records.extend(_records_from_artifact(artifact))
        else:
            records.extend(_records_from_artifact(path))
    return records


def compare_results(
    inputs: Iterable[Path],
    out_dir: Path,
    *,
    comparison_type: str = "generic_comparison",
    root_dir: Path | None = None,
) -> CompareResultsResult:
    input_paths = [Path(input_path) for input_path in inputs]
    if root_dir is not None and is_within_evidence_root(out_dir, root_dir):
        raise ValueError("compare-results output must not be written under runs/evidence; use runs/comparisons or another analysis directory")
    records = load_result_records(input_paths)
    if not records:
        raise ValueError("no compatible benchmark result artifacts found")
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = _kernel_family_summary(records)
    cpu_gpu_speedup = _cpu_gpu_speedup_payload(records) if comparison_type == "cpu_gpu_sweep" else None
    payload = {
        "schema_version": COMPARE_RESULTS_SCHEMA_VERSION,
        "record_count": len(records),
        "kernel_family_summary": summary,
        "records": records,
        "metadata": {
            "simulator_timings_are_not_hardware_speedups": True,
            "missing_gpu_or_hardware_results_are_not_fabricated": True,
        },
    }
    if cpu_gpu_speedup is not None:
        payload["cpu_gpu_speedup"] = cpu_gpu_speedup
    artifact_path = out_dir / "comparison_results.json"
    csv_path = out_dir / "comparison_results.csv"
    family_csv_path = out_dir / "kernel_family_summary.csv"
    summary_path = out_dir / "comparison_summary.md"
    manifest_path = out_dir / "comparison_manifest.json"
    write_json(artifact_path, payload)
    _write_csv(csv_path, records, RESULT_FIELDS)
    _write_csv(family_csv_path, summary, SUMMARY_FIELDS)
    extra_outputs: list[str] = []
    if cpu_gpu_speedup is not None:
        extra_outputs.extend(_write_cpu_gpu_speedup_artifacts(out_dir, cpu_gpu_speedup))
    summary_path.write_text(_summary_markdown(records, summary, cpu_gpu_speedup=cpu_gpu_speedup), encoding="utf-8")
    _write_comparison_manifest(
        manifest_path,
        out_dir=out_dir,
        input_paths=input_paths,
        comparison_type=comparison_type,
        outputs=(
            artifact_path.name,
            csv_path.name,
            family_csv_path.name,
            summary_path.name,
            *extra_outputs,
        ),
        records=records,
    )
    return CompareResultsResult(
        run_dir=out_dir,
        artifact_path=artifact_path,
        csv_path=csv_path,
        summary_path=summary_path,
        record_count=len(records),
        manifest_path=manifest_path,
    )


def _write_comparison_manifest(
    path: Path,
    *,
    out_dir: Path,
    input_paths: list[Path],
    comparison_type: str,
    outputs: Iterable[str],
    records: list[JsonDict],
) -> None:
    suite_ids = sorted({str(record.get("suite_id")) for record in records if record.get("suite_id")})
    inputs = []
    for index, input_path in enumerate(input_paths):
        manifest = _read_run_manifest(input_path)
        inputs.append(
            {
                "role": f"input_{index}",
                "path": input_path.as_posix(),
                "artifact_kind": manifest.get("artifact_kind") if manifest else None,
                "run_id": manifest.get("run_id") if manifest else input_path.name,
                "suite_id": manifest.get("suite_id") if manifest else None,
                "route_label": manifest.get("route_label") if manifest else None,
                "route_id": manifest.get("route_id") if manifest else None,
                "quantization_mode": manifest.get("quantization_mode") if manifest else None,
            }
        )
    write_json(
        path,
        {
            "schema_version": COMPARISON_MANIFEST_SCHEMA_VERSION,
            "artifact_kind": "comparison_report",
            "comparison_id": out_dir.name,
            "comparison_type": comparison_type,
            "suite_id": suite_ids[0] if len(suite_ids) == 1 else None,
            "input_count": len(input_paths),
            "record_count": len(records),
            "inputs": inputs,
            "outputs": tuple(outputs),
            "metadata": {
                "evidence_inputs_are_read_only": True,
                "comparison_outputs_are_derived_analysis": True,
            },
        },
    )


def _read_run_manifest(input_path: Path) -> JsonDict:
    manifest_path = input_path / "run_manifest.json" if input_path.is_dir() else input_path.parent / "run_manifest.json"
    if not manifest_path.exists():
        return {}
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _discover_artifacts(path: Path) -> list[Path]:
    names = {
        "dense_task_bridge_summary.json",
        "generic_task_bridge_summary.json",
        "upmem_mvp_benchmark_summary.json",
        "upmem_taskgraph_runtime_summary.json",
        "pim_bridge_eval.json",
        "simulation_backend_compare_summary.json",
    }
    return sorted(candidate for candidate in path.rglob("*.json") if candidate.name in names)


def _records_from_artifact(path: Path) -> list[JsonDict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    schema = str(payload.get("schema_version", ""))
    source = _source_artifact_label(path)
    if schema in {"dense_task_bridge_v1", "generic_task_bridge_v1"}:
        return [normalized_task_result_from_summary(payload, source_artifact=source)]
    if schema == "upmem_taskgraph_runtime_v1":
        return _upmem_taskgraph_runtime_records(payload, source_artifact=source)
    if schema == "upmem_mvp_benchmark_v1":
        return _upmem_mvp_benchmark_cpu_records(payload, source_artifact=source)
    if schema == "simulation_backend_compare_v1":
        return [normalize_parallelism_metadata(record | {"source_artifact": source}) for record in payload.get("normalized_records", [])]
    if schema == "pim_bridge_eval_v1":
        return [_pim_bridge_eval_row_record(payload, row, source_artifact=source) for row in payload.get("rows", [])]
    return []


def _source_artifact_label(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.name


def _dense_task_bridge_record(summary: JsonDict, *, source_artifact: str | None) -> JsonDict:
    validation = dict(summary.get("bridge_validation_metrics") or {})
    artifacts = dict(summary.get("artifacts") or {})
    bridge_status = summary.get("bridge_execution_status")
    metadata = dict(summary.get("metadata") or {})
    record = _base_record(
        source_artifact=source_artifact,
        run_id=_run_id_from_source(source_artifact),
        suite_id=None,
        case_id=summary.get("case_id"),
        workload_id=summary.get("case_id"),
        route_id=summary.get("route_id", "dense_gemm"),
        backend_id=summary.get("bridge_backend_id"),
        kernel_family="dense_gemm",
        execution_target=_target_from_summary(summary),
        execution_scope="task_level",
        simulator_or_hardware=_simulator_or_hardware(summary),
        status=summary.get("status"),
        validation_status=_validation_status(validation, bridge_status),
        task_count=1 if summary.get("task_id") else 0,
        validated_task_count=1 if validation.get("passed") is True else 0,
        unsupported_task_count=1 if summary.get("status") == "unsupported" else 0,
        total_wall_time_s=_float_from_nested(summary, ("bridge_result", "output_manifest", "total_time_s")),
        kernel_time_s=_float_from_nested(summary, ("bridge_result", "output_manifest", "compute_time_s")),
        build_time_s=_float_from_metadata(summary, "build_time_s"),
        validation_error_metrics=validation,
        notes=_json_string({"artifacts": artifacts, "developer_only": metadata.get("developer_only", True)}),
        warnings=_warning_text(summary),
    )
    record.update(
        {
            "execution_backend": summary.get("execution_backend"),
            "cpu_fallback_used": summary.get("cpu_fallback_used", False),
            "dpu_program_invocations": summary.get("dpu_program_invocations"),
            "upmem_program_executed": (
                summary.get("upmem_program_executed")
                if summary.get("upmem_program_executed") is not None
                else summary.get("dpu_program_executed_all_tasks")
            ),
            "hardware_execution": summary.get("hardware_execution", False),
            "hardware_timing_available": summary.get("hardware_timing_available", False),
            "hardware_speedup_applicable": summary.get("hardware_speedup_applicable", False),
        }
    )
    return _with_upmem_execution_metadata(record, summary)


def _generic_task_bridge_record(summary: JsonDict, *, source_artifact: str | None) -> JsonDict:
    validation = dict(summary.get("bridge_validation_metrics") or {})
    bridge_result = dict(summary.get("bridge_result") or {})
    output_manifest = dict(bridge_result.get("output_manifest") or {})
    output_metadata = dict(output_manifest.get("metadata") or {})
    record = _base_record(
        source_artifact=source_artifact,
        run_id=_run_id_from_source(source_artifact),
        suite_id=None,
        case_id=summary.get("case_id"),
        workload_id=summary.get("case_id"),
        route_id=summary.get("route_id", "generic_loop_fallback"),
        backend_id=summary.get("bridge_backend_id"),
        kernel_family=summary.get("kernel_family", "generic_loop_fallback"),
        execution_target=summary.get("execution_target", "upmem_simulator"),
        execution_scope=summary.get("execution_scope", "task_level"),
        simulator_or_hardware=str(output_metadata.get("target", "simulator")),
        status=summary.get("status"),
        validation_status=_validation_status(validation, summary.get("bridge_execution_status")),
        task_count=1 if summary.get("task_id") else 0,
        validated_task_count=1 if validation.get("passed") is True else 0,
        unsupported_task_count=1 if summary.get("status") == "unsupported" else 0,
        total_wall_time_s=float(output_manifest.get("total_time_s", 0.0) or 0.0),
        kernel_time_s=float(output_manifest.get("compute_time_s", 0.0) or 0.0),
        build_time_s=float(output_metadata.get("build_time_s", 0.0) or 0.0),
        validation_error_metrics=validation,
        notes=_json_string(
            {
                "artifacts": summary.get("artifacts", {}),
                "validation_target": summary.get("metadata", {}).get("validation_target"),
                "full_precision_reference_is_validation_target": False,
            }
        ),
        warnings=_warning_text(summary),
    )
    record.update(
        {
            "execution_backend": summary.get("execution_backend"),
            "cpu_fallback_used": summary.get("cpu_fallback_used", False),
            "dpu_program_invocations": summary.get("dpu_program_invocations"),
            "upmem_program_executed": summary.get("upmem_program_executed"),
            "hardware_execution": summary.get("hardware_execution", False),
            "hardware_timing_available": summary.get("hardware_timing_available", False),
            "hardware_speedup_applicable": summary.get("hardware_speedup_applicable", False),
        }
    )
    return _with_upmem_execution_metadata(record, summary)


def _pim_bridge_eval_row_record(payload: JsonDict, row: JsonDict, *, source_artifact: str | None) -> JsonDict:
    validation = dict(row.get("validation_metrics") or {})
    status = row.get("readiness_status")
    record = _base_record(
        source_artifact=source_artifact,
        run_id=str(payload.get("run_id") or _run_id_from_source(source_artifact)),
        suite_id=payload.get("suite_id"),
        case_id=row.get("case_id"),
        workload_id=row.get("workload_id"),
        route_id="dense_gemm",
        backend_id=row.get("backend_id", payload.get("backend")),
        kernel_family=row.get("kernel_family", "dense_gemm"),
        execution_target="upmem_simulator",
        execution_scope="task_level",
        simulator_or_hardware="simulator",
        status=status,
        validation_status=row.get("validation_status"),
        task_count=1,
        validated_task_count=1 if row.get("validation_status") == "passed" else 0,
        unsupported_task_count=1 if status == "unsupported" else 0,
        total_wall_time_s=float(row.get("runner_total_time_s", 0.0) or 0.0),
        kernel_time_s=float(row.get("simulator_run_time_s", 0.0) or 0.0),
        build_time_s=float(row.get("build_time_s", 0.0) or 0.0),
        validation_error_metrics=validation,
        notes=_json_string({"task_index": row.get("task_index"), "task_id": row.get("task_id")}),
        warnings="simulator_task_level_only" if row.get("backend_status") else "not_executed",
    )
    if row.get("external_command_executed") or row.get("backend_status"):
        record.update(
            {
                "contraction_execution_target": "upmem",
                "upmem_execution_mode": "sdk_simulator",
                "native_sdk_control_path": True,
                "simplepim_api_used": False,
            }
        )
    return record


def _upmem_taskgraph_runtime_records(payload: JsonDict, *, source_artifact: str | None) -> list[JsonDict]:
    counts = dict(payload.get("kernel_family_counts") or {})
    if not counts:
        return [_upmem_taskgraph_runtime_record(payload, source_artifact=source_artifact, kernel_family="unsupported", task_count=0)]
    return [
        _upmem_taskgraph_runtime_record(payload, source_artifact=source_artifact, kernel_family=str(family), task_count=int(count))
        for family, count in sorted(counts.items())
    ]


def _upmem_taskgraph_runtime_record(
    summary: JsonDict,
    *,
    source_artifact: str | None,
    kernel_family: str | None = None,
    task_count: int | None = None,
) -> JsonDict:
    final_validation = dict(summary.get("final_validation") or {})
    family = kernel_family or _dominant_kernel_family(summary)
    tasks = int(task_count if task_count is not None else summary.get("executed_tasks", 0) or 0)
    final_passed = final_validation.get("passed") is True
    record = _base_record(
        source_artifact=source_artifact,
        run_id=_run_id_from_source(source_artifact),
        suite_id=None,
        case_id=summary.get("case_id"),
        workload_id=summary.get("case_id"),
        route_id=summary.get("route_id", "upmem_tn_runtime"),
        backend_id=_json_string(summary.get("backend_counts", {})),
        kernel_family=family,
        execution_target=summary.get("contraction_execution_target", "upmem"),
        execution_scope=summary.get("execution_scope", "full_taskgraph"),
        simulator_or_hardware="simulator" if summary.get("upmem_execution_mode") == "sdk_simulator" else "not_applicable",
        status=summary.get("status"),
        validation_status="passed" if final_passed else "failed",
        task_count=tasks,
        validated_task_count=tasks if final_passed else 0,
        unsupported_task_count=int(summary.get("unsupported_tasks", 0) or 0),
        total_wall_time_s=float(summary.get("total_wall_time_s", 0.0) or 0.0),
        kernel_time_s=float(summary.get("total_kernel_time_s", 0.0) or 0.0),
        build_time_s=float(summary.get("total_build_time_s", 0.0) or 0.0),
        validation_error_metrics=final_validation,
        notes=_json_string(
            {
                "policy": summary.get("policy"),
                "quantization_mode": summary.get("quantization_mode"),
                "reason": summary.get("reason"),
                "whole_network_quantized_at_initialization": summary.get("whole_network_quantized_at_initialization"),
                "valid_primary_upmem_codepath_result": summary.get("valid_primary_upmem_codepath_result"),
                "dpu_program_executed_all_tasks": summary.get("dpu_program_executed_all_tasks"),
                "native_sdk_control_path": summary.get("native_sdk_control_path"),
                "simplepim_api_used": summary.get("simplepim_api_used"),
            }
        ),
        warnings="sdk_simulator_not_hardware_speedup",
    )
    record.update(
        {
            "execution_model": "tensor_network",
            "output_kind": "final_tensor",
            "comparison_output_kind": "final_tensor",
            "contraction_execution_target": summary.get("contraction_execution_target", "upmem"),
            "upmem_execution_mode": summary.get("upmem_execution_mode", "sdk_simulator"),
            "execution_backend": summary.get("execution_backend", "upmem_sdk"),
            "hardware_execution": summary.get("hardware_execution", False),
            "hardware_timing_available": summary.get("hardware_timing_available", False),
            "hardware_speedup_applicable": summary.get("hardware_speedup_applicable", False),
            "cpu_fallback_used": summary.get("cpu_fallback_used", False),
            "dpu_program_invocations": summary.get("dpu_program_invocations"),
            "upmem_program_executed": (
                summary.get("upmem_program_executed")
                if summary.get("upmem_program_executed") is not None
                else summary.get("dpu_program_executed_all_tasks")
            ),
            "native_sdk_control_path": summary.get("native_sdk_control_path", True),
            "simplepim_api_used": summary.get("simplepim_api_used", False),
            "quantization_mode": summary.get("quantization_mode"),
            "input_dtype_on_dpu": summary.get("input_dtype_on_dpu"),
            "accumulator_dtype_on_dpu": summary.get("accumulator_dtype_on_dpu"),
            "scaling_applied": summary.get("scaling_applied"),
            "unquantized_mode_kind": summary.get("unquantized_mode_kind"),
            "actual_h2d_bytes": summary.get("actual_h2d_bytes"),
            "actual_d2h_bytes": summary.get("actual_d2h_bytes"),
            "actual_transfer_bytes": summary.get("actual_transfer_bytes"),
            "total_quantization_time_s": summary.get("total_quantization_time_s"),
            "total_dequantization_time_s": summary.get("total_dequantization_time_s"),
        }
    )
    return record


def _upmem_mvp_benchmark_cpu_records(payload: JsonDict, *, source_artifact: str | None) -> list[JsonDict]:
    records: list[JsonDict] = []
    for record in payload.get("cpu_reference_records", []):
        if not isinstance(record, dict):
            continue
        normalized = dict(record)
        normalized["source_artifact"] = source_artifact
        normalized.setdefault("schema_version", RESULT_ARTIFACT_SCHEMA_VERSION)
        normalized.setdefault("kernel_family", "cpu_reference_only")
        normalized.setdefault("execution_target", "cpu")
        normalized.setdefault("contraction_execution_target", "cpu")
        normalized.setdefault("execution_model", "tensor_network")
        normalized.setdefault("output_kind", "final_tensor")
        normalized.setdefault("comparison_output_kind", "final_tensor")
        normalized.setdefault("execution_scope", "full_taskgraph_reference")
        normalized.setdefault("hardware_speedup", "not_applicable")
        records.append(to_jsonable(normalized))
    return records


def _with_upmem_execution_metadata(record: JsonDict, summary: JsonDict) -> JsonDict:
    execution_target = str(record.get("execution_target") or "")
    simulator = str(record.get("simulator_or_hardware") or "")
    backend = str(record.get("backend_id") or "")
    if execution_target.startswith("upmem") or simulator == "simulator" or backend.startswith("upmem_sdk_"):
        record.update(
            {
                "execution_target": "upmem",
                "contraction_execution_target": summary.get("contraction_execution_target", "upmem"),
                "upmem_execution_mode": summary.get("upmem_execution_mode", "sdk_simulator"),
                "native_sdk_control_path": summary.get("native_sdk_control_path", True),
                "simplepim_api_used": summary.get("simplepim_api_used", False),
            }
        )
    return record


def _base_record(
    *,
    source_artifact: str | None,
    run_id: str | None,
    suite_id: Any,
    case_id: Any,
    workload_id: Any,
    route_id: Any,
    backend_id: Any,
    kernel_family: Any,
    execution_target: Any,
    execution_scope: Any,
    simulator_or_hardware: Any,
    status: Any,
    validation_status: Any,
    task_count: int,
    validated_task_count: int,
    unsupported_task_count: int,
    total_wall_time_s: float,
    kernel_time_s: float,
    build_time_s: float,
    validation_error_metrics: JsonDict,
    notes: str,
    warnings: str,
) -> JsonDict:
    return {
        "schema_version": RESULT_ARTIFACT_SCHEMA_VERSION,
        "source_artifact": source_artifact,
        "run_id": run_id,
        "timestamp": None,
        "suite_id": suite_id,
        "case_id": case_id,
        "workload_id": workload_id,
        "route_id": route_id,
        "backend_id": backend_id,
        "backend_family": None,
        "benchmark_role": None,
        "route_role_description": None,
        "route_limitation_scope": None,
        "kernel_family": kernel_family if kernel_family in KERNEL_FAMILIES else "unsupported",
        "execution_model": None,
        "parallelism_mode": None,
        "parallelism_evidence_type": None,
        "execution_plan_kind": None,
        "execution_plan_executed": None,
        "slicing_enabled": False,
        "frontier_scheduler_enabled": False,
        "intra_contraction_parallelism_source": None,
        "modeled_parallelism_available": False,
        "output_kind": None,
        "comparison_output_kind": None,
        "execution_target": execution_target,
        "contraction_execution_target": None,
        "accelerator_kind": None,
        "gpu_backend_verified": False,
        "gpu_program_executed": False,
        "gpu_device_name": None,
        "gpu_runtime_stack": None,
        "upmem_execution_mode": None,
        "execution_backend": None,
        "hardware_execution": False,
        "hardware_timing_available": False,
        "hardware_speedup_applicable": False,
        "cpu_fallback_used": None,
        "dpu_program_invocations": None,
        "upmem_program_executed": None,
        "native_sdk_control_path": None,
        "simplepim_api_used": None,
        "quantization_mode": None,
        "input_dtype_on_dpu": None,
        "accumulator_dtype_on_dpu": None,
        "scaling_applied": None,
        "unquantized_mode_kind": None,
        "actual_h2d_bytes": None,
        "actual_d2h_bytes": None,
        "actual_transfer_bytes": None,
        "total_quantization_time_s": None,
        "total_dequantization_time_s": None,
        "execution_scope": execution_scope,
        "simulator_or_hardware": simulator_or_hardware,
        "status": status,
        "validation_status": validation_status,
        "repeat_id": 0,
        "measured_repeat_count": 1,
        "task_count": int(task_count),
        "validated_task_count": int(validated_task_count),
        "unsupported_task_count": int(unsupported_task_count),
        "setup_time_s": None,
        "circuit_lowering_time_s": None,
        "planning_time_s": None,
        "lowering_time_s": None,
        "data_transfer_time_s": None,
        "simulation_compute_time_s": float(kernel_time_s),
        "validation_time_s": None,
        "output_materialization_time_s": None,
        "timing_scope": "legacy_total_and_kernel",
        "gpu_synchronized": False,
        "validation_method": None,
        "expected_runtime_class": None,
        "expected_memory_class": None,
        "intended_use": None,
        "max_qubits": None,
        "manual_invocation_required": False,
        "expected_risk": None,
        "known_heavy_backends": None,
        "resource_guard_status": None,
        "resource_skip_reason": None,
        "total_wall_time_s": float(total_wall_time_s),
        "kernel_time_s": float(kernel_time_s),
        "total_wall_time_s_median": None,
        "total_wall_time_s_min": None,
        "total_wall_time_s_mean": None,
        "total_wall_time_s_std": None,
        "simulation_compute_time_s_median": None,
        "simulation_compute_time_s_min": None,
        "simulation_compute_time_s_mean": None,
        "simulation_compute_time_s_std": None,
        "host_transfer_time_s": None,
        "build_time_s": float(build_time_s),
        "launch_overhead_s": None,
        "simulator_relative_time": None,
        "hardware_speedup": "not_applicable",
        "validation_error_metrics": validation_error_metrics,
        "statevector_bytes": None,
        "tn_task_count": None,
        "tn_max_intermediate_bytes": None,
        "tn_estimated_flops": None,
        "tn_estimated_bytes": None,
        "dependency_metadata": None,
        "notes": notes,
        "warnings": warnings,
    }


def normalize_parallelism_metadata(record: JsonDict) -> JsonDict:
    normalized = dict(record)
    status = str(normalized.get("status") or "")
    executed = _parallelism_execution_plan_executed(normalized, status)
    target = str(normalized.get("contraction_execution_target") or normalized.get("execution_target") or "")
    execution_model = str(normalized.get("execution_model") or "")
    route_id = str(normalized.get("route_id") or "")
    scope = str(normalized.get("execution_scope") or "")

    normalized.setdefault("slicing_enabled", False)
    normalized.setdefault("frontier_scheduler_enabled", False)
    normalized.setdefault("modeled_parallelism_available", False)
    normalized["execution_plan_executed"] = executed

    evidence_type = normalized.get("parallelism_evidence_type")
    if evidence_type is None:
        if bool(normalized.get("modeled_parallelism_available", False)) and not executed:
            evidence_type = "modeled"
        elif executed:
            evidence_type = "executed"
        elif status in {"not_executed", "skipped", "unsupported", "failed"}:
            evidence_type = "unsupported"
        else:
            evidence_type = "not_applicable"
        normalized["parallelism_evidence_type"] = evidence_type

    if normalized.get("parallelism_mode") is None:
        normalized["parallelism_mode"] = _default_parallelism_mode(
            route_id=route_id,
            execution_model=execution_model,
            target=target,
            evidence_type=str(evidence_type),
        )

    if normalized.get("execution_plan_kind") is None:
        normalized["execution_plan_kind"] = _default_execution_plan_kind(
            route_id=route_id,
            execution_model=execution_model,
            target=target,
            scope=scope,
        )

    if normalized.get("intra_contraction_parallelism_source") is None:
        normalized["intra_contraction_parallelism_source"] = (
            "none" if normalized.get("parallelism_mode") == "sequential" else "not_applicable"
        )

    normalized["execution_plan_executed"] = bool(normalized["execution_plan_executed"])
    normalized["slicing_enabled"] = bool(normalized["slicing_enabled"])
    normalized["frontier_scheduler_enabled"] = bool(normalized["frontier_scheduler_enabled"])
    normalized["modeled_parallelism_available"] = bool(normalized["modeled_parallelism_available"])
    return to_jsonable(normalized)


def _parallelism_execution_plan_executed(record: JsonDict, status: str) -> bool:
    if record.get("execution_plan_executed") is not None:
        return bool(record.get("execution_plan_executed"))
    if status not in {"completed", "validation_failed", "executable"}:
        return False
    if record.get("resource_guard_status") not in {None, "executed"}:
        return False
    return True


def _default_parallelism_mode(*, route_id: str, execution_model: str, target: str, evidence_type: str) -> str:
    if evidence_type == "modeled":
        return "modeled_only"
    if evidence_type == "unsupported":
        return "not_applicable"
    if execution_model == "tensor_network" or target == "upmem" or route_id == "upmem_tn_sdk_simulator_quantized":
        return "sequential"
    return "not_applicable"


def _default_execution_plan_kind(*, route_id: str, execution_model: str, target: str, scope: str) -> str:
    if target == "upmem" and scope == "full_taskgraph":
        return "sequential_upmem_taskgraph"
    if target == "upmem":
        return "single_upmem_task"
    if route_id == "quimb_tn_exact":
        return "external_tn_unsliced_contract"
    if route_id == "cpu_tn_einsum_exact" or scope == "full_taskgraph_reference":
        return "sequential_taskgraph"
    if execution_model == "full_state":
        return "full_state_simulation"
    return "not_applicable"


def _kernel_family_summary(records: list[JsonDict]) -> list[JsonDict]:
    by_family: dict[str, list[JsonDict]] = {}
    for record in records:
        by_family.setdefault(str(record.get("kernel_family") or "unsupported"), []).append(record)
    summary: list[JsonDict] = []
    for family in sorted(by_family):
        family_records = by_family[family]
        summary.append(
            {
                "schema_version": RESULT_ARTIFACT_SCHEMA_VERSION,
                "kernel_family": family,
                "record_count": len(family_records),
                "task_count": sum(int(record.get("task_count", 0) or 0) for record in family_records),
                "validated_task_count": sum(int(record.get("validated_task_count", 0) or 0) for record in family_records),
                "unsupported_task_count": sum(int(record.get("unsupported_task_count", 0) or 0) for record in family_records),
                "measured_record_count": sum(1 for record in family_records if record.get("status") in {"completed", "executable"}),
                "simulator_record_count": sum(1 for record in family_records if record.get("simulator_or_hardware") == "simulator"),
                "hardware_record_count": sum(1 for record in family_records if record.get("simulator_or_hardware") == "hardware"),
            }
        )
    return summary


def _write_csv(path: Path, rows: list[JsonDict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in fieldnames})


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(to_jsonable(value), sort_keys=True, separators=(",", ":"))
    return value


def _summary_markdown(records: list[JsonDict], family_summary: list[JsonDict], *, cpu_gpu_speedup: JsonDict | None = None) -> str:
    lines = [
        "# Benchmark Result Comparison",
        "",
        f"Compatible records loaded: {len(records)}.",
        "",
        "Simulator timings are task-level development evidence, not hardware speedups.",
        "",
        "## Kernel Families",
        "",
        "| Kernel family | Records | Tasks | Validated tasks | Unsupported tasks |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in family_summary:
        lines.append(
            f"| {row['kernel_family']} | {row['record_count']} | {row['task_count']} | "
            f"{row['validated_task_count']} | {row['unsupported_task_count']} |"
        )
    if cpu_gpu_speedup is not None:
        lines.extend(
            [
                "",
                "## CPU/GPU Speedup",
                "",
                f"Matched CPU/GPU repeat pairs: {len(cpu_gpu_speedup['pairs'])}.",
                f"Skipped candidate pairs: {len(cpu_gpu_speedup['skipped_pairs'])}.",
                "Speedup is CPU time divided by GPU time. For performance-tier no-dump rows, compute speedup is the primary metric; these rows are not full-statevector validation evidence.",
                "",
                "| Family | Qubits | Repeats | Output mode | Validation | Performance tier | Median wall speedup | Median compute speedup | GPU device |",
                "| --- | ---: | ---: | --- | --- | ---: | ---: | ---: | --- |",
            ]
        )
        for row in cpu_gpu_speedup["summary"]:
            lines.append(
                f"| {row['case_family']} | {row['n_qubits']} | {row['matched_repeat_count']} | "
                f"{row.get('state_output_mode') or ''} | {row.get('validation_method') or ''} | {row.get('performance_tier')} | "
                f"{_format_float(row.get('total_wall_speedup_median'))} | "
                f"{_format_float(row.get('compute_speedup_median'))} | {row.get('gpu_device_name') or ''} |"
            )
    lines.extend(
        [
            "",
            "Missing CPU, GPU, hardware, or full-circuit results are left absent; this report does not fabricate baselines.",
            "",
        ]
    )
    return "\n".join(lines)


def _cpu_gpu_speedup_payload(records: list[JsonDict]) -> JsonDict:
    cpu_records: dict[tuple[str, int], JsonDict] = {}
    gpu_records: dict[tuple[str, int], JsonDict] = {}
    skipped: list[JsonDict] = []
    for record in records:
        route_id = str(record.get("route_id") or "")
        if route_id not in {"quest_cpu_full_state_exact", "quest_gpu_full_state_exact"}:
            continue
        case_id = str(record.get("case_id") or "")
        repeat_id = _int_value(record.get("repeat_id"))
        if not case_id or repeat_id is None:
            skipped.append(_cpu_gpu_skipped(case_id, record.get("repeat_id"), "missing_case_or_repeat", route_id == "quest_cpu_full_state_exact", route_id == "quest_gpu_full_state_exact"))
            continue
        key = (case_id, repeat_id)
        if route_id == "quest_cpu_full_state_exact":
            cpu_records[key] = record
        else:
            gpu_records[key] = record

    pairs: list[JsonDict] = []
    for key in sorted(set(cpu_records) | set(gpu_records)):
        cpu = cpu_records.get(key)
        gpu = gpu_records.get(key)
        case_id, repeat_id = key
        if cpu is None or gpu is None:
            skipped.append(_cpu_gpu_skipped(case_id, repeat_id, "missing_cpu_or_gpu_row", cpu is not None, gpu is not None))
            continue
        if not _cpu_gpu_validation_ok(cpu) or not _cpu_gpu_validation_ok(gpu):
            skipped.append(_cpu_gpu_skipped(case_id, repeat_id, "validation_not_passed", True, True))
            continue
        if str(cpu.get("state_output_mode") or "full_dump") != str(gpu.get("state_output_mode") or "full_dump"):
            skipped.append(_cpu_gpu_skipped(case_id, repeat_id, "state_output_mode_mismatch", True, True))
            continue
        if str(cpu.get("validation_method") or "") != str(gpu.get("validation_method") or ""):
            skipped.append(_cpu_gpu_skipped(case_id, repeat_id, "validation_method_mismatch", True, True))
            continue
        if bool(cpu.get("performance_tier", False)) != bool(gpu.get("performance_tier", False)):
            skipped.append(_cpu_gpu_skipped(case_id, repeat_id, "performance_tier_mismatch", True, True))
            continue
        if gpu.get("gpu_backend_verified") is not True or gpu.get("gpu_program_executed") is not True:
            skipped.append(_cpu_gpu_skipped(case_id, repeat_id, "gpu_execution_not_verified", True, True))
            continue
        cpu_total = _positive_float(cpu.get("total_wall_time_s"))
        gpu_total = _positive_float(gpu.get("total_wall_time_s"))
        cpu_compute = _positive_float(cpu.get("simulation_compute_time_s"))
        gpu_compute = _positive_float(gpu.get("simulation_compute_time_s"))
        if cpu_total is None or gpu_total is None or cpu_compute is None or gpu_compute is None:
            skipped.append(_cpu_gpu_skipped(case_id, repeat_id, "missing_positive_timing", True, True))
            continue
        family, n_qubits = _case_family_and_qubits(cpu)
        performance_tier = bool(cpu.get("performance_tier", False))
        timing_scope = "performance_compute" if performance_tier else "correctness_wall_and_compute"
        pairs.append(
            {
                "schema_version": COMPARE_RESULTS_SCHEMA_VERSION,
                "case_id": case_id,
                "case_family": family,
                "n_qubits": n_qubits,
                "repeat_id": repeat_id,
                "cpu_route_id": "quest_cpu_full_state_exact",
                "gpu_route_id": "quest_gpu_full_state_exact",
                "state_output_mode": str(cpu.get("state_output_mode") or "full_dump"),
                "validation_method": str(cpu.get("validation_method") or ""),
                "performance_tier": performance_tier,
                "exact_output_comparable": bool(cpu.get("exact_output_comparable", False)) and bool(gpu.get("exact_output_comparable", False)),
                "full_statevector_validation_available": bool(cpu.get("full_statevector_validation_available", False))
                and bool(gpu.get("full_statevector_validation_available", False)),
                "timing_scope": timing_scope,
                "cpu_total_wall_time_s": cpu_total,
                "gpu_total_wall_time_s": gpu_total,
                "total_wall_speedup": cpu_total / gpu_total,
                "cpu_simulation_compute_time_s": cpu_compute,
                "gpu_simulation_compute_time_s": gpu_compute,
                "compute_speedup": cpu_compute / gpu_compute,
                "validation_status": str(cpu.get("validation_status") or gpu.get("validation_status") or "passed"),
                "gpu_device_name": gpu.get("gpu_device_name"),
            }
        )
    return {
        "schema_version": COMPARE_RESULTS_SCHEMA_VERSION,
        "pairs": pairs,
        "summary": _cpu_gpu_speedup_summary(pairs),
        "skipped_pairs": skipped,
        "timing_fields": {
            "total_wall": "total_wall_time_s",
            "compute": "simulation_compute_time_s",
            "repeat": "repeat_id",
            "performance_speedup_field": "compute_speedup",
        },
    }


def _cpu_gpu_speedup_summary(pairs: list[JsonDict]) -> list[JsonDict]:
    grouped: dict[tuple[str, int, str, str, bool], list[JsonDict]] = {}
    for row in pairs:
        grouped.setdefault(
            (
                str(row["case_family"]),
                int(row["n_qubits"]),
                str(row.get("state_output_mode") or "full_dump"),
                str(row.get("validation_method") or ""),
                bool(row.get("performance_tier", False)),
            ),
            [],
        ).append(row)
    summary: list[JsonDict] = []
    for (family, n_qubits, state_output_mode, validation_method, performance_tier), rows in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1], item[0][2])):
        total_speedups = [float(row["total_wall_speedup"]) for row in rows]
        compute_speedups = [float(row["compute_speedup"]) for row in rows]
        cpu_totals = [float(row["cpu_total_wall_time_s"]) for row in rows]
        gpu_totals = [float(row["gpu_total_wall_time_s"]) for row in rows]
        cpu_compute = [float(row["cpu_simulation_compute_time_s"]) for row in rows]
        gpu_compute = [float(row["gpu_simulation_compute_time_s"]) for row in rows]
        devices = sorted({str(row.get("gpu_device_name") or "") for row in rows if row.get("gpu_device_name")})
        summary.append(
            {
                "schema_version": COMPARE_RESULTS_SCHEMA_VERSION,
                "case_family": family,
                "n_qubits": n_qubits,
                "matched_repeat_count": len(rows),
                "state_output_mode": state_output_mode,
                "validation_method": validation_method,
                "performance_tier": performance_tier,
                "timing_scope": "performance_compute" if performance_tier else "correctness_wall_and_compute",
                "cpu_total_wall_time_s_median": statistics.median(cpu_totals),
                "gpu_total_wall_time_s_median": statistics.median(gpu_totals),
                "total_wall_speedup_median": statistics.median(total_speedups),
                "total_wall_speedup_mean": statistics.mean(total_speedups),
                "total_wall_speedup_min": min(total_speedups),
                "total_wall_speedup_max": max(total_speedups),
                "cpu_simulation_compute_time_s_median": statistics.median(cpu_compute),
                "gpu_simulation_compute_time_s_median": statistics.median(gpu_compute),
                "compute_speedup_median": statistics.median(compute_speedups),
                "compute_speedup_mean": statistics.mean(compute_speedups),
                "compute_speedup_min": min(compute_speedups),
                "compute_speedup_max": max(compute_speedups),
                "gpu_device_name": ", ".join(devices),
                "validation_status": str(rows[0].get("validation_status") or "passed"),
            }
        )
    return summary


def _cpu_gpu_validation_ok(record: JsonDict) -> bool:
    status = str(record.get("validation_status") or "")
    return status in {"passed", "passed_native_status", "passed_runtime_only"}


def _write_cpu_gpu_speedup_artifacts(out_dir: Path, payload: JsonDict) -> list[str]:
    pair_path = out_dir / "cpu_gpu_speedup_pairs.csv"
    summary_path = out_dir / "cpu_gpu_speedup_summary.csv"
    skipped_path = out_dir / "cpu_gpu_speedup_skipped_pairs.csv"
    plot_data_path = out_dir / "plots" / "data" / "cpu_gpu_speedup_summary.csv"
    _write_csv(pair_path, payload["pairs"], CPU_GPU_SPEEDUP_PAIR_FIELDS)
    _write_csv(summary_path, payload["summary"], CPU_GPU_SPEEDUP_SUMMARY_FIELDS)
    _write_csv(skipped_path, payload["skipped_pairs"], CPU_GPU_SPEEDUP_SKIPPED_FIELDS)
    _write_csv(plot_data_path, payload["summary"], CPU_GPU_SPEEDUP_SUMMARY_FIELDS)
    outputs = [
        pair_path.relative_to(out_dir).as_posix(),
        summary_path.relative_to(out_dir).as_posix(),
        skipped_path.relative_to(out_dir).as_posix(),
        plot_data_path.relative_to(out_dir).as_posix(),
    ]
    outputs.extend(_write_cpu_gpu_speedup_plots(out_dir, payload["summary"], source_csv=plot_data_path.relative_to(out_dir).as_posix()))
    return outputs


def _write_cpu_gpu_speedup_plots(out_dir: Path, rows: list[JsonDict], *, source_csv: str) -> list[str]:
    plots_dir = out_dir / "plots"
    manifest_path = plots_dir / "plot_manifest.json"
    outputs = ["plots/plot_manifest.json"]
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        write_json(manifest_path, {"schema_version": COMPARE_RESULTS_SCHEMA_VERSION, "status": "skipped", "reason": "matplotlib_unavailable", "error": str(exc)})
        return outputs
    entries: list[JsonDict] = []
    plot_specs = (
        ("cpu_gpu_speedup_by_family_qubits.png", _plot_cpu_gpu_speedup, "CPU/GPU speedup by family and qubits"),
        ("cpu_gpu_runtime_by_family_qubits.png", _plot_cpu_gpu_runtime, "CPU/GPU runtime by family and qubits"),
    )
    for filename, plotter, title in plot_specs:
        path = plots_dir / filename
        reason = plotter(plt, path, rows)
        if reason:
            entries.append({"plot": filename, "title": title, "status": "skipped", "reason": reason, "source_csv": source_csv, "source_row_count": len(rows)})
        else:
            entries.append(
                {
                    "plot": filename,
                    "title": title,
                    "status": "generated",
                    "reason": None,
                    "source_csv": source_csv,
                    "source_row_count": len(rows),
                    "image": _plot_image_metadata(plt, path),
                }
            )
            outputs.append(f"plots/{filename}")
    write_json(
        manifest_path,
        {
            "schema_version": COMPARE_RESULTS_SCHEMA_VERSION,
            "status": "completed",
            "plots": entries,
            "written": [entry["plot"] for entry in entries if entry["status"] == "generated"],
            "skipped": [entry for entry in entries if entry["status"] == "skipped"],
        },
    )
    return outputs


def _plot_cpu_gpu_speedup(plt: Any, path: Path, rows: list[JsonDict]) -> str | None:
    if not rows:
        return "required_data_unavailable"
    families = sorted({str(row["case_family"]) for row in rows})
    use_compute = any(bool(row.get("performance_tier", False)) for row in rows)
    speedup_key = "compute_speedup_median" if use_compute else "total_wall_speedup_median"
    fig, axis = plt.subplots(figsize=(max(9.0, len(rows) * 0.32), 5.8), constrained_layout=True)
    for family in families:
        family_rows = sorted((row for row in rows if row["case_family"] == family), key=lambda row: int(row["n_qubits"]))
        axis.plot([int(row["n_qubits"]) for row in family_rows], [float(row[speedup_key]) for row in family_rows], marker="o", label=family.upper())
    axis.axhline(1.0, color="#6b7280", linewidth=1.0, linestyle="--")
    axis.set_xlabel("Qubits")
    axis.set_ylabel("CPU/GPU compute speedup" if use_compute else "CPU/GPU wall-time speedup")
    axis.set_title("CPU/GPU full-state speedup by circuit family")
    axis.grid(True, axis="y", alpha=0.3)
    axis.legend(fontsize="small", ncol=2)
    _save_plot(fig, path)
    return None


def _plot_cpu_gpu_runtime(plt: Any, path: Path, rows: list[JsonDict]) -> str | None:
    if not rows:
        return "required_data_unavailable"
    ordered = sorted(rows, key=lambda row: (str(row["case_family"]), int(row["n_qubits"])))
    labels = [f"{row['case_family']}_{row['n_qubits']}q" for row in ordered]
    x = list(range(len(labels)))
    width = 0.38
    use_compute = any(bool(row.get("performance_tier", False)) for row in rows)
    cpu_key = "cpu_simulation_compute_time_s_median" if use_compute else "cpu_total_wall_time_s_median"
    gpu_key = "gpu_simulation_compute_time_s_median" if use_compute else "gpu_total_wall_time_s_median"
    fig, axis = plt.subplots(figsize=(max(10.0, len(labels) * 0.34), 6.0), constrained_layout=True)
    axis.bar([item - width / 2 for item in x], [float(row[cpu_key]) for row in ordered], width=width, label="QuEST CPU", color="#2563eb")
    axis.bar([item + width / 2 for item in x], [float(row[gpu_key]) for row in ordered], width=width, label="QuEST HIP GPU", color="#16a34a")
    axis.set_xticks(x)
    axis.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
    axis.set_ylabel("Median compute time (s, log scale)" if use_compute else "Median wall time (s, log scale)")
    axis.set_title("CPU/GPU full-state compute runtime by circuit and size" if use_compute else "CPU/GPU full-state runtime by circuit and size")
    axis.set_yscale("log")
    axis.grid(True, axis="y", alpha=0.3)
    axis.legend()
    _save_plot(fig, path)
    return None


def _save_plot(fig: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    try:
        import matplotlib.pyplot as _plt

        _plt.close(fig)
    except Exception:  # pragma: no cover
        fig.clf()


def _plot_image_metadata(plt: Any, path: Path) -> JsonDict:
    payload: JsonDict = {"size_bytes": path.stat().st_size if path.exists() else 0}
    try:
        image = plt.imread(path)
        height, width = image.shape[:2]
        payload.update({"width_px": int(width), "height_px": int(height), "non_empty": payload["size_bytes"] > 1000})
    except Exception as exc:
        payload.update({"read_error": str(exc), "non_empty": False})
    return payload


def _cpu_gpu_skipped(case_id: Any, repeat_id: Any, reason: str, cpu_present: bool, gpu_present: bool) -> JsonDict:
    return {
        "schema_version": COMPARE_RESULTS_SCHEMA_VERSION,
        "case_id": case_id,
        "repeat_id": repeat_id,
        "reason": reason,
        "cpu_present": cpu_present,
        "gpu_present": gpu_present,
    }


def _case_family_and_qubits(record: JsonDict) -> tuple[str, int]:
    case_id = str(record.get("case_id") or "")
    match = re.match(r"^(?:quest_)?(?P<family>.+?)_(?P<qubits>\d+)q(?:_.+)?$", case_id)
    if match:
        return match.group("family"), int(match.group("qubits"))
    notes = _json_dict(record.get("notes"))
    family = str(notes.get("circuit_family") or notes.get("family") or case_id)
    n_qubits = _int_value(record.get("n_qubits")) or _int_value(notes.get("n_qubits")) or _int_value(record.get("max_qubits")) or 0
    return family, n_qubits


def _json_dict(value: Any) -> JsonDict:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str) or not value:
        return {}
    try:
        payload = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _positive_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0.0 else None


def _int_value(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _format_float(value: Any) -> str:
    parsed = _positive_float(value)
    if parsed is None:
        return ""
    return f"{parsed:.6g}"


def _run_id_from_source(source_artifact: str | None) -> str | None:
    if not source_artifact:
        return None
    path = Path(source_artifact)
    for parent in path.parents:
        if parent.name.startswith("20"):
            return parent.name
    return path.parent.name


def _validation_status(metrics: JsonDict, backend_status: Any) -> str:
    if metrics.get("passed") is True:
        return "passed"
    if metrics.get("passed") is False:
        return "failed"
    if backend_status:
        return "not_applicable"
    return "not_run"


def _target_from_summary(summary: JsonDict) -> str:
    metadata = dict(summary.get("metadata") or {})
    bridge = dict(summary.get("bridge_result") or {})
    output = dict(bridge.get("output_manifest") or {})
    output_metadata = dict(output.get("metadata") or {})
    target = output_metadata.get("target")
    if target == "hardware":
        return "upmem_hardware"
    if target == "simulator" or summary.get("bridge_backend_id", "").startswith("upmem_sdk"):
        return "upmem_simulator"
    return str(metadata.get("execution_target") or "cpu")


def _simulator_or_hardware(summary: JsonDict) -> str:
    bridge = dict(summary.get("bridge_result") or {})
    output = dict(bridge.get("output_manifest") or {})
    target = dict(output.get("metadata") or {}).get("target")
    if target in {"simulator", "hardware"}:
        return str(target)
    if str(summary.get("bridge_backend_id", "")).startswith("upmem_sdk"):
        return "simulator"
    return "not_applicable"


def _float_from_nested(payload: JsonDict, path: tuple[str, ...]) -> float:
    current: Any = payload
    for key in path:
        if not isinstance(current, dict):
            return 0.0
        current = current.get(key)
    return float(current or 0.0)


def _float_from_metadata(summary: JsonDict, key: str) -> float:
    bridge = dict(summary.get("bridge_result") or {})
    output = dict(bridge.get("output_manifest") or {})
    return float(dict(output.get("metadata") or {}).get(key, 0.0) or 0.0)


def _json_string(payload: Any) -> str:
    return json.dumps(to_jsonable(payload), sort_keys=True, separators=(",", ":"))


def _dominant_kernel_family(summary: JsonDict) -> str:
    counts = dict(summary.get("kernel_family_counts") or {})
    if not counts:
        return "unsupported"
    return max(sorted(counts), key=lambda family: int(counts[family]))


def _warning_text(summary: JsonDict) -> str:
    warnings: list[str] = ["task_level_only"]
    if summary.get("execution_target") == "upmem_simulator" or str(summary.get("bridge_backend_id", "")).startswith("upmem_sdk"):
        warnings.append("simulator_not_hardware_speedup")
    if summary.get("status") != "completed":
        warnings.append(str(summary.get("reason") or summary.get("status")))
    return ",".join(warnings)
