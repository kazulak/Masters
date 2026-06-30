from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from quantum_bench.core.jsonio import read_jsonl, write_json
from quantum_bench.core.records import JsonDict, to_jsonable


RESULT_ARTIFACT_SCHEMA_VERSION = "benchmark_result_artifact_v1"
COMPARE_RESULTS_SCHEMA_VERSION = "compare_results_v1"

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
    "kernel_family",
    "execution_model",
    "output_kind",
    "comparison_output_kind",
    "execution_target",
    "contraction_execution_target",
    "accelerator_kind",
    "upmem_execution_mode",
    "native_sdk_control_path",
    "simplepim_api_used",
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


@dataclass(frozen=True)
class CompareResultsResult:
    run_dir: Path
    artifact_path: Path
    csv_path: Path
    summary_path: Path
    record_count: int


def normalized_task_result_from_summary(summary: JsonDict, *, source_artifact: str | None = None) -> JsonDict:
    schema = str(summary.get("schema_version", ""))
    if schema == "dense_task_bridge_v1":
        return _dense_task_bridge_record(summary, source_artifact=source_artifact)
    if schema == "generic_task_bridge_v1":
        return _generic_task_bridge_record(summary, source_artifact=source_artifact)
    raise ValueError(f"unsupported one-task summary schema_version: {schema}")


def normalized_upmem_taskgraph_result_from_summary(summary: JsonDict, *, source_artifact: str | None = None) -> JsonDict:
    return _upmem_taskgraph_runtime_record(summary, source_artifact=source_artifact)


def normalized_upmem_taskgraph_records_from_summary(summary: JsonDict, *, source_artifact: str | None = None) -> list[JsonDict]:
    return _upmem_taskgraph_runtime_records(summary, source_artifact=source_artifact)


def load_result_records(inputs: Iterable[Path]) -> list[JsonDict]:
    records: list[JsonDict] = []
    for input_path in inputs:
        path = Path(input_path)
        if path.is_dir():
            canonical = path / "normalized_records.jsonl"
            if canonical.exists():
                records.extend(read_jsonl(canonical))
                continue
            for artifact in _discover_artifacts(path):
                records.extend(_records_from_artifact(artifact))
        else:
            records.extend(_records_from_artifact(path))
    return records


def compare_results(inputs: Iterable[Path], out_dir: Path) -> CompareResultsResult:
    records = load_result_records(inputs)
    if not records:
        raise ValueError("no compatible benchmark result artifacts found")
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = _kernel_family_summary(records)
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
    artifact_path = out_dir / "comparison_results.json"
    csv_path = out_dir / "comparison_results.csv"
    family_csv_path = out_dir / "kernel_family_summary.csv"
    summary_path = out_dir / "comparison_summary.md"
    write_json(artifact_path, payload)
    _write_csv(csv_path, records, RESULT_FIELDS)
    _write_csv(family_csv_path, summary, SUMMARY_FIELDS)
    summary_path.write_text(_summary_markdown(records, summary), encoding="utf-8")
    return CompareResultsResult(
        run_dir=out_dir,
        artifact_path=artifact_path,
        csv_path=csv_path,
        summary_path=summary_path,
        record_count=len(records),
    )


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
        return [to_jsonable(record | {"source_artifact": source}) for record in payload.get("normalized_records", [])]
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
            "native_sdk_control_path": summary.get("native_sdk_control_path", True),
            "simplepim_api_used": summary.get("simplepim_api_used", False),
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
        "kernel_family": kernel_family if kernel_family in KERNEL_FAMILIES else "unsupported",
        "execution_model": None,
        "output_kind": None,
        "comparison_output_kind": None,
        "execution_target": execution_target,
        "contraction_execution_target": None,
        "accelerator_kind": None,
        "upmem_execution_mode": None,
        "native_sdk_control_path": None,
        "simplepim_api_used": None,
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


def _summary_markdown(records: list[JsonDict], family_summary: list[JsonDict]) -> str:
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
    lines.extend(
        [
            "",
            "Missing CPU, GPU, hardware, or full-circuit results are left absent; this report does not fabricate baselines.",
            "",
        ]
    )
    return "\n".join(lines)


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
