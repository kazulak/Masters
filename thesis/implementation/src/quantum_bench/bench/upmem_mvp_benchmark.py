from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from quantum_bench.bench.config import load_suite
from quantum_bench.bench.reporting import (
    ARTIFACT_REFERENCE_SCHEMA_VERSION,
    artifact_ref,
    prune_run,
    validate_retention_mode,
    write_normalized_records,
    write_run_manifest,
)
from quantum_bench.bench.result_artifacts import RESULT_ARTIFACT_SCHEMA_VERSION, normalized_upmem_taskgraph_records_from_summary
from quantum_bench.bench.run_dirs import EVIDENCE_ARTIFACT_KIND, create_run_dir, sanitize
from quantum_bench.circuits import load_circuit, manifest
from quantum_bench.core.jsonio import write_json, write_jsonl
from quantum_bench.core.records import JsonDict, to_jsonable
from quantum_bench.environment import capture_environment
from quantum_bench.targets.upmem import SYNTHETIC_PRESSURE_ERROR, annotate_task_graph_with_upmem_estimates, is_synthetic_pressure_case
from quantum_bench.targets.upmem.taskgraph_runtime import (
    UPMEM_EXECUTION_MODE,
    UPMEM_TASKGRAPH_POLICIES,
    UPMEM_TASKGRAPH_QUANTIZATION_MODES,
    build_generic_taskgraph_reference,
    execute_upmem_taskgraph_runtime,
)
from quantum_bench.tn import build_tensor_network, execute_task_sequence_np_einsum, plan_task_graph_with_config, with_path_cost_summary


UPMEM_MVP_BENCHMARK_SCHEMA_VERSION = "upmem_mvp_benchmark_v1"

RESULT_FIELDS = [
    "case_id",
    "workload_id",
    "circuit_family",
    "n_qubits",
    "route",
    "policy",
    "quantization_mode",
    "status",
    "reason",
    "contraction_execution_target",
    "upmem_execution_mode",
    "whole_network_quantized_at_initialization",
    "cpu_fallback_used",
    "hardware_benchmark_result",
    "hardware_timing_available",
    "hardware_speedup_applicable",
    "total_tasks",
    "executed_tasks",
    "unsupported_tasks",
    "dense_gemm_count",
    "generic_loop_fallback_count",
    "complex_split_complex_task_count",
    "final_validation_status",
    "max_abs_error",
    "mean_abs_error",
    "l2_error",
    "norm_drift",
    "max_task_bridge_error",
    "total_wall_time_s",
    "total_kernel_time_s",
    "total_bridge_time_s",
    "total_quantization_time_s",
    "total_dequantization_time_s",
    "total_simulator_time_s",
    "total_host_orchestration_time_s",
    "actual_h2d_bytes",
    "actual_d2h_bytes",
    "actual_transfer_bytes",
    "full_precision_transfer_bytes_model",
    "transfer_compression_ratio",
    "input_dtype_on_dpu",
    "accumulator_dtype_on_dpu",
    "scaling_applied",
    "unquantized_mode_kind",
    "native_upmem_kernel_executed",
    "native_unquantized_upmem_kernel_executed",
    "cpu_reference_artifact",
    "upmem_runtime_summary_artifact",
    "upmem_task_metrics_artifact",
    "final_tensor_artifact",
]

KERNEL_FAMILY_FIELDS = ["policy", "quantization_mode", "kernel_family", "task_count"]
QUANTIZATION_FIELDS = [
    "case_id",
    "policy",
    "quantization_mode",
    "final_validation_status",
    "max_abs_error",
    "mean_abs_error",
    "l2_error",
    "norm_drift",
    "max_task_bridge_error",
]
UNSUPPORTED_FIELDS = ["case_id", "policy", "quantization_mode", "reason", "count"]
QUANTIZATION_COMPARISON_FIELDS = [
    "case_id",
    "policy",
    "same_route_comparison",
    "same_taskgraph",
    "same_kernel_family",
    "quantized_runtime_s",
    "unquantized_runtime_s",
    "quantization_runtime_speedup",
    "quantized_transfer_bytes",
    "unquantized_transfer_bytes",
    "transfer_reduction",
    "quantized_max_abs_error_vs_full_precision",
    "unquantized_max_abs_error_vs_full_precision",
    "accuracy_delta",
    "native_unquantized_upmem_kernel_executed",
]


@dataclass(frozen=True)
class UpmemMvpBenchmarkResult:
    schema_version: str
    status: str
    reason: str | None
    run_dir: Path
    summary_path: Path
    result_count: int
    summary: JsonDict = field(default_factory=dict)

    def to_json_dict(self) -> JsonDict:
        payload = to_jsonable(self)
        payload["run_dir"] = str(self.run_dir)
        payload["summary_path"] = str(self.summary_path)
        return payload


def run_upmem_mvp_benchmark(
    root_dir: Path,
    *,
    suite_path: Path,
    policies: tuple[str, ...] = ("generic-only", "dense-then-generic"),
    quantization_modes: tuple[str, ...] = ("per_task_input_quantize",),
    execute_external: bool = False,
    max_taskgraph_tasks: int = 128,
    fail_fast: bool = False,
    artifact_retention: str = "compact",
    env: Mapping[str, str] | None = None,
) -> UpmemMvpBenchmarkResult:
    suite = load_suite(suite_path)
    validate_options(
        policies=policies,
        quantization_modes=quantization_modes,
        execute_external=execute_external,
        max_taskgraph_tasks=max_taskgraph_tasks,
        artifact_retention=artifact_retention,
    )
    route_label = _upmem_mvp_route_label(policies, quantization_modes)
    run_dir = create_run_dir(
        root_dir,
        str(suite["suite_id"]),
        artifact_kind=EVIDENCE_ARTIFACT_KIND,
        route_label=route_label,
    )
    write_run_manifest(
        run_dir,
        run_kind="upmem_mvp_benchmark",
        suite_id=str(suite["suite_id"]),
        suite_path=str(suite_path),
        artifact_kind=EVIDENCE_ARTIFACT_KIND,
        route_label=route_label,
        route_id="upmem_tn_runtime",
        backend_id="upmem_sdk_simulator_generic_loop",
        policy=policies[0] if len(policies) == 1 else ",".join(policies),
        quantization_mode=quantization_modes[0] if len(quantization_modes) == 1 else ",".join(quantization_modes),
        execution_scope="suite_taskgraph",
        evidence_type="sdk_simulator",
        normalized_records="normalized_records.jsonl",
        summary="upmem_mvp_benchmark_summary.json",
        policies=policies,
        quantization_modes=quantization_modes,
        upmem_execution_mode=UPMEM_EXECUTION_MODE,
        artifact_retention=artifact_retention,
        root_dir=root_dir,
    )
    write_json(run_dir / "environment.json", capture_environment(root_dir))
    (run_dir / "config" / "resolved_suite.yml").write_text(yaml.safe_dump(suite, sort_keys=True), encoding="utf-8")
    write_json(
        run_dir / "config" / "upmem_mvp_benchmark_input.json",
        {
            "suite_path": str(suite_path),
            "suite_id": suite["suite_id"],
            "policies": policies,
            "quantization_modes": quantization_modes,
            "execute_external": execute_external,
            "max_taskgraph_tasks": max_taskgraph_tasks,
            "fail_fast": fail_fast,
            "artifact_retention": artifact_retention,
        },
    )

    result_rows: list[JsonDict] = []
    cpu_reference_records: list[JsonDict] = []
    case_records: list[JsonDict] = []

    for case_payload in suite["cases"]:
        try:
            generated = _generate_reference_case(root_dir, run_dir, suite, case_payload)
            cpu_reference_records.append(generated["cpu_reference_record"])
        except Exception as exc:
            generated = _failed_case_payload(case_payload, str(exc))
            if fail_fast:
                raise

        for policy in policies:
            for quantization_mode in quantization_modes:
                row = _run_case_policy(
                    run_dir=run_dir,
                    suite=suite,
                    case_payload=case_payload,
                    generated=generated,
                    policy=policy,
                    quantization_mode=quantization_mode,
                    execute_external=execute_external,
                    max_taskgraph_tasks=max_taskgraph_tasks,
                    env=env,
                )
                result_rows.append(row)
                case_records.append(row)
                if fail_fast and row["status"] not in {"completed", "validation_failed"}:
                    raise RuntimeError(f"UPMEM MVP benchmark failed for {row['case_id']} {policy}: {row['reason']}")

    write_jsonl(run_dir / "upmem_mvp_benchmark_cases.jsonl", case_records)
    _write_csv(run_dir / "upmem_mvp_benchmark_results.csv", result_rows, RESULT_FIELDS)
    _write_csv(run_dir / "kernel_family_summary.csv", _kernel_family_summary(result_rows), KERNEL_FAMILY_FIELDS)
    _write_csv(run_dir / "quantization_accuracy_summary.csv", _quantization_accuracy_rows(result_rows), QUANTIZATION_FIELDS)
    _write_csv(run_dir / "unsupported_reasons.csv", _unsupported_reason_rows(result_rows), UNSUPPORTED_FIELDS)
    quantization_comparison_rows = _quantization_comparison_rows(result_rows)
    _write_csv(run_dir / "quantization_comparison.csv", quantization_comparison_rows, QUANTIZATION_COMPARISON_FIELDS)
    write_json(
        run_dir / "quantization_comparison.json",
        {
            "schema_version": UPMEM_MVP_BENCHMARK_SCHEMA_VERSION,
            "comparison_kind": "same_route_generic_upmem_quantization_attribution",
            "rows": quantization_comparison_rows,
            "metadata": {
                "native_unquantized_upmem_required": True,
                "hardware_speedup_applicable": False,
                "simulator_runtime_ratio_not_hardware_speedup": True,
            },
        },
    )

    summary = _summary_payload(
        suite=suite,
        suite_path=suite_path,
        policies=policies,
        quantization_modes=quantization_modes,
        execute_external=execute_external,
        max_taskgraph_tasks=max_taskgraph_tasks,
        result_rows=result_rows,
        cpu_reference_records=cpu_reference_records,
        quantization_comparison_rows=quantization_comparison_rows,
    )
    write_json(run_dir / "upmem_mvp_benchmark_summary.json", summary)
    (run_dir / "comparison_summary.md").write_text(_summary_markdown(summary, result_rows), encoding="utf-8")
    normalized_records = _normalized_records_for_run(run_dir, cpu_reference_records)
    write_normalized_records(run_dir, normalized_records)
    if artifact_retention == "compact":
        prune_run(run_dir, artifact_retention="compact")
    status = "completed" if not any(row["status"] == "failed" for row in result_rows) else "failed"
    return UpmemMvpBenchmarkResult(
        schema_version=UPMEM_MVP_BENCHMARK_SCHEMA_VERSION,
        status=status,
        reason=None if status == "completed" else "one_or_more_case_policies_failed",
        run_dir=run_dir,
        summary_path=run_dir / "upmem_mvp_benchmark_summary.json",
        result_count=len(result_rows),
        summary=summary,
    )


def _upmem_mvp_route_label(policies: tuple[str, ...], quantization_modes: tuple[str, ...]) -> str:
    if policies == ("generic-only",):
        modes = set(quantization_modes)
        if modes == {"none"}:
            return "upmem_generic_float32"
        if modes == {"per_task_input_quantize"}:
            return "upmem_generic_int8"
        if modes == {"none", "per_task_input_quantize"}:
            return "upmem_generic_both_modes"
    return "upmem_mvp"


def validate_options(
    *,
    policies: tuple[str, ...],
    quantization_modes: tuple[str, ...],
    execute_external: bool,
    max_taskgraph_tasks: int,
    artifact_retention: str = "compact",
) -> None:
    validate_retention_mode(artifact_retention)
    if not execute_external:
        raise ValueError("upmem-mvp-benchmark requires --execute-external for strict UPMEM SDK DPU execution")
    if not policies:
        raise ValueError("--policies must contain at least one policy")
    invalid_policies = sorted(set(policies) - set(UPMEM_TASKGRAPH_POLICIES))
    if invalid_policies:
        raise ValueError(f"unsupported policies: {','.join(invalid_policies)}")
    if not quantization_modes:
        raise ValueError("--quantization-modes must contain at least one mode")
    invalid_modes = sorted(set(quantization_modes) - set(UPMEM_TASKGRAPH_QUANTIZATION_MODES))
    if invalid_modes:
        raise ValueError(f"unsupported quantization modes: {','.join(invalid_modes)}")
    if max_taskgraph_tasks < 0:
        raise ValueError("--max-taskgraph-tasks must be >= 0")


def parse_csv_choices(raw: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in raw.split(",") if item.strip())
    if not values:
        raise ValueError("comma-separated option must contain at least one value")
    return values


def _generate_reference_case(root_dir: Path, run_dir: Path, suite: JsonDict, case_payload: JsonDict) -> JsonDict:
    case_id = str(case_payload["case_id"])
    case_dir = run_dir / "cases" / sanitize(case_id)
    if is_synthetic_pressure_case(case_payload):
        raise ValueError(SYNTHETIC_PRESSURE_ERROR)
    circuit = load_circuit(case_payload, root_dir)
    network = build_tensor_network(circuit)
    graph = plan_task_graph_with_config(network, suite["planner"])
    graph, _ = annotate_task_graph_with_upmem_estimates(graph)
    graph = with_path_cost_summary(graph)
    start = time.perf_counter()
    reference_output, reference_metadata = execute_task_sequence_np_einsum(graph, network)
    cpu_time_s = time.perf_counter() - start
    cpu_reference_rel = Path("cases") / sanitize(case_id) / "cpu_reference.npy"
    (run_dir / cpu_reference_rel).parent.mkdir(parents=True, exist_ok=True)
    np.save(run_dir / cpu_reference_rel, reference_output, allow_pickle=False)
    cpu_reference_json_rel = Path("cases") / sanitize(case_id) / "cpu_reference.json"
    cpu_reference_payload = {
        "schema_version": UPMEM_MVP_BENCHMARK_SCHEMA_VERSION,
        "case_id": case_id,
        "suite_id": str(suite["suite_id"]),
        "workload_id": str(case_payload.get("workload_id", case_id)),
        "contraction_execution_target": "cpu",
        "execution_scope": "full_taskgraph_reference",
        "cpu_role": "reference_validator",
        "cpu_reference_used_to_feed_runtime_tensors": False,
        "task_count": len(graph.tasks),
        "cpu_reference_time_s": float(cpu_time_s),
        "output_shape": tuple(int(dim) for dim in reference_output.shape),
        "output_dtype": str(reference_output.dtype),
        "output_artifact": artifact_ref(run_dir, cpu_reference_rel, role="cpu_reference_tensor"),
        "metadata": _reference_metadata_payload(reference_metadata),
    }
    write_json(run_dir / cpu_reference_json_rel, cpu_reference_payload)
    write_json(case_dir / "circuit.json", manifest(circuit))
    write_json(case_dir / "task_graph.json", graph)
    write_json(case_dir / "path_summary.json", graph.path_summary)
    cpu_reference_record = _cpu_reference_record(
        run_id=run_dir.name,
        suite_id=str(suite["suite_id"]),
        case_id=case_id,
        workload_id=str(case_payload.get("workload_id", case_id)),
        task_count=len(graph.tasks),
        cpu_time_s=cpu_time_s,
        source_artifact=cpu_reference_json_rel.as_posix(),
    )
    return {
        "status": "ready",
        "case_id": case_id,
        "workload_id": str(case_payload.get("workload_id", case_id)),
        "circuit_family": str(case_payload.get("circuit", {}).get("name", circuit.name)),
        "circuit": circuit,
        "network": network,
        "graph": graph,
        "reference_output": reference_output,
        "reference_metadata": reference_metadata,
        "cpu_reference_artifact": artifact_ref(run_dir, cpu_reference_json_rel, role="cpu_reference_metadata"),
        "cpu_reference_tensor_artifact": artifact_ref(run_dir, cpu_reference_rel, role="cpu_reference_tensor"),
        "cpu_reference_record": cpu_reference_record,
    }


def _failed_case_payload(case_payload: JsonDict, reason: str) -> JsonDict:
    case_id = str(case_payload.get("case_id", "unknown"))
    return {
        "status": "failed",
        "reason": reason,
        "case_id": case_id,
        "workload_id": str(case_payload.get("workload_id", case_id)),
        "circuit_family": str(case_payload.get("circuit", {}).get("name", "unknown")),
        "circuit": None,
        "network": None,
        "graph": None,
        "reference_output": None,
        "reference_metadata": {},
        "cpu_reference_artifact": None,
        "cpu_reference_tensor_artifact": None,
    }


def _run_case_policy(
    *,
    run_dir: Path,
    suite: JsonDict,
    case_payload: JsonDict,
    generated: JsonDict,
    policy: str,
    quantization_mode: str,
    execute_external: bool,
    max_taskgraph_tasks: int,
    env: Mapping[str, str] | None,
) -> JsonDict:
    case_id = str(generated["case_id"])
    workload_id = str(generated["workload_id"])
    child_rel = Path("cases") / sanitize(case_id) / sanitize(policy) / sanitize(quantization_mode)
    child_dir = run_dir / child_rel
    runtime_summary_rel = child_rel / "upmem_taskgraph_runtime_summary.json"
    task_metrics_rel = child_rel / "upmem_taskgraph_task_metrics.jsonl"
    final_tensor_rel = child_rel / "final_tensor.npy"

    if generated["status"] != "ready":
        summary = _runtime_error_summary(case_id, policy, quantization_mode, str(generated.get("reason") or "case_setup_failed"))
        _write_child_runtime_artifacts(run_dir, runtime_summary_rel, task_metrics_rel, summary, ())
        return _row_from_runtime_summary(generated, policy, quantization_mode, summary, runtime_summary_rel, task_metrics_rel, None)
    graph = generated["graph"]
    if len(graph.tasks) > max_taskgraph_tasks:
        summary = _runtime_error_summary(case_id, policy, quantization_mode, "taskgraph_task_cap_exceeded")
        _write_child_runtime_artifacts(run_dir, runtime_summary_rel, task_metrics_rel, summary, ())
        return _row_from_runtime_summary(generated, policy, quantization_mode, summary, runtime_summary_rel, task_metrics_rel, None)

    reference_output = generated["reference_output"]
    reference_kind = "cpu_exact_taskgraph_full_precision"
    generic_reference = None
    if policy == "generic-only":
        generic_reference = build_generic_taskgraph_reference(
            graph=graph,
            network=generated["network"],
            case_id=case_id,
            quantization_mode=quantization_mode,  # type: ignore[arg-type]
        )
        if generic_reference.status != "completed":
            summary = _runtime_error_summary(
                case_id,
                policy,
                quantization_mode,
                f"generic_feasibility_{generic_reference.reason or generic_reference.status}",
            )
            summary["generic_feasibility"] = generic_reference.to_json_dict()
            _write_child_runtime_artifacts(run_dir, runtime_summary_rel, task_metrics_rel, summary, ())
            return _row_from_runtime_summary(generated, policy, quantization_mode, summary, runtime_summary_rel, task_metrics_rel, None)
        reference_output = generic_reference.output
        reference_kind = str(generic_reference.summary.get("reference_kind") or reference_kind)

    runtime = execute_upmem_taskgraph_runtime(
        graph=graph,
        network=generated["network"],
        case_id=case_id,
        policy=policy,
        quantization_mode=quantization_mode,
        bridge_root=child_dir / "upmem_taskgraph_bridge",
        execute_external=execute_external,
        reference_output=reference_output,
        reference_kind=reference_kind,
        env=env,
    )
    final_tensor_artifact = None
    if runtime.output is not None:
        (run_dir / final_tensor_rel).parent.mkdir(parents=True, exist_ok=True)
        np.save(run_dir / final_tensor_rel, runtime.output, allow_pickle=False)
        final_tensor_artifact = final_tensor_rel.as_posix()
    summary = _enriched_runtime_summary(
        run_dir=run_dir,
        generated=generated,
        policy=policy,
        quantization_mode=quantization_mode,
        runtime_summary=dict(runtime.summary),
        runtime_task_metrics=runtime.task_metrics,
        generic_reference=generic_reference,
        runtime_summary_rel=runtime_summary_rel,
        task_metrics_rel=task_metrics_rel,
        final_tensor_artifact=final_tensor_artifact,
    )
    _write_child_runtime_artifacts(run_dir, runtime_summary_rel, task_metrics_rel, summary, runtime.task_metrics)
    return _row_from_runtime_summary(generated, policy, quantization_mode, summary, runtime_summary_rel, task_metrics_rel, final_tensor_artifact)


def _enriched_runtime_summary(
    *,
    run_dir: Path,
    generated: JsonDict,
    policy: str,
    quantization_mode: str,
    runtime_summary: JsonDict,
    runtime_task_metrics: tuple[JsonDict, ...],
    generic_reference,
    runtime_summary_rel: Path,
    task_metrics_rel: Path,
    final_tensor_artifact: str | None,
) -> JsonDict:
    from quantum_bench.bench.result_artifacts import normalized_upmem_taskgraph_result_from_summary

    runtime_summary.update(
        {
            "schema_version": "upmem_taskgraph_runtime_v1",
            "suite_id": generated.get("suite_id"),
            "case_id": generated["case_id"],
            "workload_id": generated["workload_id"],
            "circuit": manifest(generated["circuit"]),
            "route_id": "upmem_tn_runtime",
            "execution_scope": "full_taskgraph",
            "policy": policy,
            "quantization_mode": quantization_mode,
            "task_metrics_artifact": _planned_artifact_ref(task_metrics_rel, role="task_metrics"),
            "final_tensor_artifact": artifact_ref(run_dir, final_tensor_artifact, role="final_tensor"),
            "reference": {
                "kind": runtime_summary.get("final_validation", {}).get("reference_kind", "cpu_exact_taskgraph_full_precision"),
                "cpu_reference_artifact": generated["cpu_reference_artifact"],
                "cpu_reference_tensor_artifact": generated["cpu_reference_tensor_artifact"],
                "cpu_reference_used_to_feed_runtime_tensors": False,
                "generic_reference": generic_reference.to_json_dict() if generic_reference is not None else None,
            },
            "artifacts": {
                "runtime_summary": _planned_artifact_ref(runtime_summary_rel, role="runtime_summary"),
                "task_metrics": _planned_artifact_ref(task_metrics_rel, role="task_metrics"),
                "final_tensor": artifact_ref(run_dir, final_tensor_artifact, role="final_tensor"),
            },
            "metadata": {
                **dict(runtime_summary.get("metadata") or {}),
                "suite_level_upmem_mvp_benchmark": True,
                "cpu_reference_used_to_feed_runtime_tensors": False,
                "strict_upmem_timing_excludes_cpu_reference_time": True,
            },
        }
    )
    runtime_summary["complex_split_complex_task_count"] = sum(
        1 for row in runtime_task_metrics if row.get("complex_representation") == "split_real_imag"
    )
    runtime_summary["max_task_bridge_error"] = _max_bridge_error(runtime_task_metrics)
    runtime_summary["normalized_result"] = normalized_upmem_taskgraph_result_from_summary(runtime_summary)
    return to_jsonable(runtime_summary)


def _runtime_error_summary(case_id: str, policy: str, quantization_mode: str, reason: str, *, status: str = "unsupported") -> JsonDict:
    from quantum_bench.bench.result_artifacts import normalized_upmem_taskgraph_result_from_summary

    summary = {
        "schema_version": "upmem_taskgraph_runtime_v1",
        "case_id": case_id,
        "status": status,
        "reason": reason,
        "policy": policy,
        "quantization_mode": quantization_mode,
        "whole_network_quantized_at_initialization": False,
        "contraction_execution_target": "upmem",
        "upmem_execution_mode": UPMEM_EXECUTION_MODE,
        "native_sdk_control_path": True,
        "simplepim_api_used": False,
        "hardware_benchmark_result": False,
        "hardware_timing_available": False,
        "hardware_speedup_applicable": False,
        "cpu_fallback_used": False,
        "valid_primary_upmem_codepath_result": False,
        "total_tasks": 0,
        "executed_tasks": 0,
        "unsupported_tasks": 1,
        "failed_tasks": 1 if status == "failed" else 0,
        "kernel_family_counts": {},
        "backend_counts": {},
        "final_validation": {"passed": False, "reason": "not_available"},
        "complex_split_complex_task_count": 0,
        "metadata": {
            "suite_level_upmem_mvp_benchmark": True,
            "cpu_reference_used_to_feed_runtime_tensors": False,
        },
    }
    summary["normalized_result"] = normalized_upmem_taskgraph_result_from_summary(summary)
    return to_jsonable(summary)


def _planned_artifact_ref(rel_path: Path, *, role: str) -> JsonDict:
    return {
        "schema_version": ARTIFACT_REFERENCE_SCHEMA_VERSION,
        "role": role,
        "relative_path": rel_path.as_posix(),
        "retained": True,
        "status": "retained",
        "prune_reason": None,
        "metadata": {},
    }


def _write_child_runtime_artifacts(
    run_dir: Path,
    runtime_summary_rel: Path,
    task_metrics_rel: Path,
    summary: JsonDict,
    task_metrics: tuple[JsonDict, ...],
) -> None:
    write_jsonl(run_dir / task_metrics_rel, list(task_metrics))
    write_json(run_dir / runtime_summary_rel, summary)


def _row_from_runtime_summary(
    generated: JsonDict,
    policy: str,
    quantization_mode: str,
    summary: JsonDict,
    runtime_summary_rel: Path,
    task_metrics_rel: Path,
    final_tensor_artifact: str | None,
) -> JsonDict:
    final_validation = dict(summary.get("final_validation") or {})
    kernel_counts = dict(summary.get("kernel_family_counts") or {})
    return to_jsonable(
        {
            "case_id": generated["case_id"],
            "workload_id": generated["workload_id"],
            "circuit_family": generated["circuit_family"],
            "n_qubits": int(getattr(generated.get("circuit"), "n_qubits", 0) or 0),
            "route": "upmem_taskgraph_runtime",
            "policy": policy,
            "quantization_mode": quantization_mode,
            "status": summary.get("status"),
            "reason": summary.get("reason"),
            "contraction_execution_target": "upmem",
            "upmem_execution_mode": UPMEM_EXECUTION_MODE,
            "whole_network_quantized_at_initialization": False,
            "cpu_fallback_used": bool(summary.get("cpu_fallback_used", False)),
            "hardware_benchmark_result": False,
            "hardware_timing_available": False,
            "hardware_speedup_applicable": False,
            "total_tasks": int(summary.get("total_tasks", 0) or 0),
            "executed_tasks": int(summary.get("executed_tasks", 0) or 0),
            "unsupported_tasks": int(summary.get("unsupported_tasks", 0) or 0),
            "dense_gemm_count": int(kernel_counts.get("dense_gemm", 0) or 0),
            "generic_loop_fallback_count": int(kernel_counts.get("generic_loop_fallback", 0) or 0),
            "complex_split_complex_task_count": int(summary.get("complex_split_complex_task_count", 0) or 0),
            "final_validation_status": "passed" if final_validation.get("passed") is True else "failed",
            "max_abs_error": final_validation.get("max_abs_error"),
            "mean_abs_error": final_validation.get("mean_abs_error"),
            "l2_error": final_validation.get("l2_error"),
            "norm_drift": final_validation.get("norm_drift"),
            "max_task_bridge_error": summary.get("max_task_bridge_error"),
            "total_wall_time_s": float(summary.get("total_wall_time_s", 0.0) or 0.0),
            "total_kernel_time_s": float(summary.get("total_kernel_time_s", 0.0) or 0.0),
            "total_bridge_time_s": float(summary.get("total_bridge_time_s", 0.0) or 0.0),
            "total_quantization_time_s": float(summary.get("total_quantization_time_s", 0.0) or 0.0),
            "total_dequantization_time_s": float(summary.get("total_dequantization_time_s", 0.0) or 0.0),
            "total_simulator_time_s": float(summary.get("total_kernel_time_s", 0.0) or 0.0),
            "total_host_orchestration_time_s": max(
                0.0,
                float(summary.get("total_wall_time_s", 0.0) or 0.0) - float(summary.get("total_kernel_time_s", 0.0) or 0.0),
            ),
            "actual_h2d_bytes": int(summary.get("actual_h2d_bytes", 0) or 0),
            "actual_d2h_bytes": int(summary.get("actual_d2h_bytes", 0) or 0),
            "actual_transfer_bytes": int(summary.get("actual_transfer_bytes", 0) or 0),
            "full_precision_transfer_bytes_model": int(summary.get("full_precision_transfer_bytes_model", 0) or 0),
            "transfer_compression_ratio": summary.get("transfer_compression_ratio"),
            "input_dtype_on_dpu": summary.get("input_dtype_on_dpu"),
            "accumulator_dtype_on_dpu": summary.get("accumulator_dtype_on_dpu"),
            "scaling_applied": summary.get("scaling_applied"),
            "unquantized_mode_kind": summary.get("unquantized_mode_kind"),
            "native_upmem_kernel_executed": bool(summary.get("dpu_program_executed_all_tasks") is True),
            "native_unquantized_upmem_kernel_executed": bool(
                summary.get("quantization_mode") == "none"
                and summary.get("input_dtype_on_dpu") == "float32"
                and summary.get("dpu_program_executed_all_tasks") is True
            ),
            "cpu_reference_artifact": generated.get("cpu_reference_artifact"),
            "upmem_runtime_summary_artifact": summary.get("artifacts", {}).get("runtime_summary"),
            "upmem_task_metrics_artifact": summary.get("artifacts", {}).get("task_metrics"),
            "final_tensor_artifact": summary.get("artifacts", {}).get("final_tensor"),
        }
    )


def _cpu_reference_record(
    *,
    run_id: str,
    suite_id: str,
    case_id: str,
    workload_id: str,
    task_count: int,
    cpu_time_s: float,
    source_artifact: str,
) -> JsonDict:
    return {
        "schema_version": RESULT_ARTIFACT_SCHEMA_VERSION,
        "source_artifact": source_artifact,
        "run_id": run_id,
        "timestamp": None,
        "suite_id": suite_id,
        "case_id": case_id,
        "workload_id": workload_id,
        "route_id": "cpu_tn_einsum_exact",
        "backend_id": "numpy_einsum",
        "kernel_family": "cpu_reference_only",
        "execution_target": "cpu",
        "execution_scope": "full_taskgraph_reference",
        "simulator_or_hardware": "not_applicable",
        "status": "completed",
        "validation_status": "reference",
        "task_count": int(task_count),
        "validated_task_count": int(task_count),
        "unsupported_task_count": 0,
        "total_wall_time_s": float(cpu_time_s),
        "kernel_time_s": float(cpu_time_s),
        "host_transfer_time_s": None,
        "build_time_s": 0.0,
        "launch_overhead_s": None,
        "simulator_relative_time": None,
        "hardware_speedup": "not_applicable",
        "validation_error_metrics": {},
        "notes": json.dumps(
            {
                "contraction_execution_target": "cpu",
                "cpu_role": "reference_validator",
                "cpu_reference_used_to_feed_runtime_tensors": False,
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        "warnings": "reference_not_upmem_execution",
        "contraction_execution_target": "cpu",
        "cpu_role": "reference_validator",
    }


def _reference_metadata_payload(metadata: JsonDict) -> JsonDict:
    payload = dict(metadata)
    payload.pop("task_metrics", None)
    return to_jsonable(payload)


def _summary_payload(
    *,
    suite: JsonDict,
    suite_path: Path,
    policies: tuple[str, ...],
    quantization_modes: tuple[str, ...],
    execute_external: bool,
    max_taskgraph_tasks: int,
    result_rows: list[JsonDict],
    cpu_reference_records: list[JsonDict],
    quantization_comparison_rows: list[JsonDict],
) -> JsonDict:
    return to_jsonable(
        {
            "schema_version": UPMEM_MVP_BENCHMARK_SCHEMA_VERSION,
            "suite_id": suite["suite_id"],
            "suite_path": str(suite_path),
            "policies": policies,
            "quantization_modes": quantization_modes,
            "execute_external": execute_external,
            "max_taskgraph_tasks": max_taskgraph_tasks,
            "case_policy_count": len(result_rows),
            "completed_count": sum(1 for row in result_rows if row["status"] == "completed"),
            "unsupported_count": sum(1 for row in result_rows if row["status"] == "unsupported"),
            "failed_count": sum(1 for row in result_rows if row["status"] == "failed"),
            "validation_failed_count": sum(1 for row in result_rows if row["status"] == "validation_failed"),
            "cpu_reference_records": cpu_reference_records,
            "upmem_rows": result_rows,
            "quantization_comparison_rows": quantization_comparison_rows,
            "quantization_comparison_count": len(quantization_comparison_rows),
            "upmem_normalized_records_are_child_runtime_summaries": False,
            "root_summary_emits_upmem_normalized_records": True,
            "root_normalized_records_are_canonical": True,
            "normalized_records_artifact": "normalized_records.jsonl",
            "metadata": {
                "developer_only": True,
                "normal_suite_routes_executed": False,
                "suite_routes_ignored": True,
                "cpu_reference_used_to_feed_runtime_tensors": False,
                "sdk_simulator_timings_are_not_hardware_speedups": True,
            },
        }
    )


def _normalized_records_for_run(run_dir: Path, cpu_reference_records: list[JsonDict]) -> list[JsonDict]:
    records = [dict(record) for record in cpu_reference_records]
    for summary_path in sorted(run_dir.glob("cases/*/*/*/upmem_taskgraph_runtime_summary.json")):
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        source = summary_path.relative_to(run_dir).as_posix()
        records.extend(normalized_upmem_taskgraph_records_from_summary(payload, source_artifact=source))
    for record in records:
        record.setdefault("run_id", run_dir.name)
    return to_jsonable(records)


def _kernel_family_summary(rows: list[JsonDict]) -> list[JsonDict]:
    grouped: dict[tuple[str, str, str], int] = {}
    for row in rows:
        for family, field in (("dense_gemm", "dense_gemm_count"), ("generic_loop_fallback", "generic_loop_fallback_count")):
            count = int(row.get(field, 0) or 0)
            if count:
                key = (str(row["policy"]), str(row["quantization_mode"]), family)
                grouped[key] = grouped.get(key, 0) + count
    return [
        {"policy": policy, "quantization_mode": mode, "kernel_family": family, "task_count": count}
        for (policy, mode, family), count in sorted(grouped.items())
    ]


def _quantization_accuracy_rows(rows: list[JsonDict]) -> list[JsonDict]:
    output: list[JsonDict] = []
    for row in rows:
        output.append(
            {
                "case_id": row["case_id"],
                "policy": row["policy"],
                "quantization_mode": row["quantization_mode"],
                "final_validation_status": row["final_validation_status"],
                "max_abs_error": row.get("max_abs_error"),
                "mean_abs_error": row.get("mean_abs_error"),
                "l2_error": row.get("l2_error"),
                "norm_drift": row.get("norm_drift"),
                "max_task_bridge_error": row.get("max_task_bridge_error"),
            }
        )
    return output


def _quantization_comparison_rows(rows: list[JsonDict]) -> list[JsonDict]:
    by_key: dict[tuple[str, str], dict[str, JsonDict]] = {}
    for row in rows:
        if row.get("policy") != "generic-only":
            continue
        if row.get("status") != "completed":
            continue
        mode = str(row.get("quantization_mode"))
        if mode not in {"none", "per_task_input_quantize"}:
            continue
        key = (str(row.get("case_id")), str(row.get("policy")))
        by_key.setdefault(key, {})[mode] = row

    output: list[JsonDict] = []
    for (case_id, policy), modes in sorted(by_key.items()):
        quantized = modes.get("per_task_input_quantize")
        unquantized = modes.get("none")
        if quantized is None or unquantized is None:
            continue
        quantized_runtime = _positive_float_or_none(quantized.get("total_wall_time_s"))
        unquantized_runtime = _positive_float_or_none(unquantized.get("total_wall_time_s"))
        quantized_transfer = _positive_float_or_none(quantized.get("actual_transfer_bytes"))
        unquantized_transfer = _positive_float_or_none(unquantized.get("actual_transfer_bytes"))
        quantized_error = _float_or_none(quantized.get("max_abs_error"))
        unquantized_error = _float_or_none(unquantized.get("max_abs_error"))
        output.append(
            {
                "case_id": case_id,
                "policy": policy,
                "same_route_comparison": True,
                "same_taskgraph": True,
                "same_kernel_family": True,
                "quantized_runtime_s": quantized_runtime,
                "unquantized_runtime_s": unquantized_runtime,
                "quantization_runtime_speedup": _ratio(unquantized_runtime, quantized_runtime),
                "quantized_transfer_bytes": quantized_transfer,
                "unquantized_transfer_bytes": unquantized_transfer,
                "transfer_reduction": _ratio(unquantized_transfer, quantized_transfer),
                "quantized_max_abs_error_vs_full_precision": quantized_error,
                "unquantized_max_abs_error_vs_full_precision": unquantized_error,
                "accuracy_delta": None if quantized_error is None or unquantized_error is None else quantized_error - unquantized_error,
                "native_unquantized_upmem_kernel_executed": bool(
                    unquantized.get("native_unquantized_upmem_kernel_executed") is True
                ),
            }
        )
    return output


def _positive_float_or_none(value: Any) -> float | None:
    number = _float_or_none(value)
    if number is None or number <= 0.0:
        return None
    return number


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator == 0.0:
        return None
    return numerator / denominator


def _max_bridge_error(task_metrics: tuple[JsonDict, ...]) -> float | None:
    values: list[float] = []
    for row in task_metrics:
        _collect_bridge_error(row.get("bridge_validation_metrics"), values)
        component_metrics = row.get("component_metrics") or {}
        if isinstance(component_metrics, dict):
            for component in component_metrics.values():
                if isinstance(component, dict):
                    _collect_bridge_error(component.get("bridge_validation_metrics"), values)
    return max(values) if values else None


def _collect_bridge_error(metrics: Any, values: list[float]) -> None:
    if not isinstance(metrics, dict):
        return
    value = metrics.get("max_abs_error")
    if value is not None:
        values.append(float(value))


def _unsupported_reason_rows(rows: list[JsonDict]) -> list[JsonDict]:
    grouped: dict[tuple[str, str, str, str], int] = {}
    for row in rows:
        if row["status"] == "completed":
            continue
        reason = str(row.get("reason") or row["status"])
        key = (str(row["case_id"]), str(row["policy"]), str(row["quantization_mode"]), reason)
        grouped[key] = grouped.get(key, 0) + 1
    return [
        {"case_id": case_id, "policy": policy, "quantization_mode": mode, "reason": reason, "count": count}
        for (case_id, policy, mode, reason), count in sorted(grouped.items())
    ]


def _summary_markdown(summary: JsonDict, rows: list[JsonDict]) -> str:
    lines = [
        "# UPMEM MVP Benchmark",
        "",
        f"Suite: `{summary['suite_id']}`",
        f"Case-policy runs: {summary['case_policy_count']}",
        f"Completed: {summary['completed_count']}",
        f"Unsupported: {summary['unsupported_count']}",
        f"Failed: {summary['failed_count']}",
        f"Validation failed: {summary['validation_failed_count']}",
        "",
        "SDK simulator mode validates the UPMEM DPU code path; it is not hardware timing or speedup evidence.",
        "CPU exact reference artifacts are used only for final validation and reporting, never to feed UPMEM runtime intermediates.",
        "",
        "## Case Policies",
        "",
        "| Case | Policy | Quantization | Status | Tasks | Dense | Generic | Max abs error |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['case_id']} | {row['policy']} | {row['quantization_mode']} | {row['status']} | "
            f"{row['total_tasks']} | {row['dense_gemm_count']} | {row['generic_loop_fallback_count']} | {row.get('max_abs_error')} |"
        )
    lines.append("")
    return "\n".join(lines)


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
