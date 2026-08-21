from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

import numpy as np

from quantum_bench.bench.reporting import write_run_manifest, write_summary_and_normalized_records
from quantum_bench.bench.result_artifacts import normalized_upmem_taskgraph_records_from_summary, normalized_upmem_taskgraph_result_from_summary
from quantum_bench.bench.run_dirs import EVIDENCE_ARTIFACT_KIND, create_run_dir
from quantum_bench.circuits import builtin_circuit, manifest
from quantum_bench.core.jsonio import write_json, write_jsonl
from quantum_bench.core.records import JsonDict, to_jsonable
from quantum_bench.environment import capture_environment
from quantum_bench.targets.upmem.evidence import (
    CONTRACTION_EXECUTION_TARGET_UPMEM,
    UPMEM_EXECUTION_MODE_SDK_SIMULATOR,
    UPMEM_SDK_SIMULATOR_EXECUTES_THROUGH,
)
from quantum_bench.targets.upmem.generic_boundary import build_generic_boundary_workload, is_generic_boundary_case
from quantum_bench.targets.upmem.schedule import annotate_task_graph_with_upmem_estimates
from quantum_bench.targets.upmem.taskgraph_runtime import (
    QUANTIZED_FINAL_VALIDATION_TOLERANCES,
    UPMEM_TASKGRAPH_POLICIES,
    UPMEM_TASKGRAPH_QUANTIZATION_MODES,
    UpmemTaskGraphPolicy,
    UpmemTaskGraphQuantizationMode,
    UpmemTaskGraphScheduleMode,
    build_generic_taskgraph_reference,
    execute_upmem_taskgraph_runtime,
)
from quantum_bench.tn.execution import execute_task_sequence_np_einsum
from quantum_bench.tn.network import build_tensor_network
from quantum_bench.tn.task_graph import plan_task_graph_with_config, with_path_cost_summary
from quantum_bench.validation import validate


UPMEM_TASKGRAPH_RUNTIME_RUN_SCHEMA_VERSION = "upmem_taskgraph_runtime_run_v1"


@dataclass(frozen=True)
class UpmemTaskGraphRuntimeRunResult:
    schema_version: str
    status: str
    reason: str | None
    run_dir: Path
    summary_path: Path
    case_id: str
    policy: str
    quantization_mode: str
    summary: JsonDict = field(default_factory=dict)

    def to_json_dict(self) -> JsonDict:
        payload = to_jsonable(self)
        payload["run_dir"] = str(self.run_dir)
        payload["summary_path"] = str(self.summary_path)
        return payload


def run_upmem_taskgraph_runtime(
    root_dir: Path,
    *,
    case: str = "bell_2q",
    n_qubits: int | None = None,
    policy: UpmemTaskGraphPolicy = "generic-only",
    quantization_mode: UpmemTaskGraphQuantizationMode = "per_task_input_quantize",
    execute_external: bool = False,
    env: Mapping[str, str] | None = None,
    schedule_mode: UpmemTaskGraphScheduleMode = "sequential",
    frontier_worker_count: int = 1,
    dpu_group_count: int = 1,
    task_assignment_strategy: str = "sequential_single_dpu",
) -> UpmemTaskGraphRuntimeRunResult:
    route_label = _upmem_taskgraph_route_label(policy, quantization_mode)
    route_id = "upmem_tn_runtime"
    run_kind = "upmem_taskgraph_runtime"
    execution_scope = "full_taskgraph"
    run_dir = create_run_dir(root_dir, case, artifact_kind=EVIDENCE_ARTIFACT_KIND, route_label=route_label)
    summary_path = run_dir / "upmem_taskgraph_runtime_summary.json"
    final_tensor_rel = Path("raw") / "final_tensor.npy"
    write_run_manifest(
        run_dir,
        run_kind=run_kind,
        suite_id=case,
        suite_path=None,
        artifact_kind=EVIDENCE_ARTIFACT_KIND,
        route_label=route_label,
        route_id=route_id,
        backend_id="upmem_sdk_simulator_generic_loop",
        policy=policy,
        quantization_mode=quantization_mode,
        execution_scope=execution_scope,
        evidence_type="sdk_simulator",
        normalized_records="normalized_records.jsonl",
        summary="upmem_taskgraph_runtime_summary.json",
        policies=(policy,),
        quantization_modes=(quantization_mode,),
        upmem_execution_mode=UPMEM_EXECUTION_MODE_SDK_SIMULATOR,
        root_dir=root_dir,
    )
    write_json(run_dir / "environment.json", capture_environment(root_dir))

    try:
        if policy not in UPMEM_TASKGRAPH_POLICIES:
            summary = _error_summary(case, policy, quantization_mode, "unsupported_policy")
            _write_summary_and_records(run_dir, summary_path, summary)
            return _result(run_dir, summary_path, summary)
        if quantization_mode not in UPMEM_TASKGRAPH_QUANTIZATION_MODES:
            summary = _error_summary(case, policy, quantization_mode, "unsupported_quantization_mode")
            _write_summary_and_records(run_dir, summary_path, summary)
            return _result(run_dir, summary_path, summary)

        boundary_case = is_generic_boundary_case(case)
        if boundary_case:
            if n_qubits is not None:
                summary = _error_summary(case, policy, quantization_mode, "generic_boundary_does_not_accept_n_qubits")
                _write_summary_and_records(run_dir, summary_path, summary)
                return _result(run_dir, summary_path, summary)
            workload = build_generic_boundary_workload(case)
            case_id = workload.case_id
            circuit_payload = workload.manifest
            network = workload.network
            graph = workload.graph
        else:
            circuit_params: JsonDict = {"name": case}
            if n_qubits is not None:
                circuit_params["n_qubits"] = int(n_qubits)
            circuit = builtin_circuit(case, circuit_params)
            case_id = circuit.name
            circuit_payload = manifest(circuit)
            network = build_tensor_network(circuit)
            graph = plan_task_graph_with_config(network, None)
            graph, _ = annotate_task_graph_with_upmem_estimates(graph)
            graph = with_path_cost_summary(graph)
        task_metrics_rel = Path("cases") / case_id / "upmem_taskgraph_task_metrics.jsonl"
        reference_output, reference_metadata = execute_task_sequence_np_einsum(graph, network)
        primary_reference_output = reference_output
        primary_reference_kind = "cpu_exact_taskgraph_full_precision"
        generic_quantized_reference = None
        if policy == "generic-only":
            generic_quantized_reference = build_generic_taskgraph_reference(
                graph=graph,
                network=network,
                case_id=case_id,
                quantization_mode=quantization_mode,
            )
            if generic_quantized_reference.status != "completed":
                summary = _error_summary(
                    case_id,
                    policy,
                    quantization_mode,
                    f"generic_quantized_reference_{generic_quantized_reference.reason or generic_quantized_reference.status}",
                )
                summary["generic_quantized_taskgraph_reference"] = generic_quantized_reference.to_json_dict()
                summary["reference"] = {
                    "kind": "generic_quantized_taskgraph_replay",
                    "status": generic_quantized_reference.status,
                    "reason": generic_quantized_reference.reason,
                    "cpu_reference_used_to_feed_runtime_tensors": False,
                    "full_precision_reference_is_task_validation_target": False,
                    "full_precision_cpu_reference": {
                        "kind": "cpu_exact_taskgraph_full_precision_accuracy_only",
                        "metadata": _reference_metadata_payload(reference_metadata),
                    },
                }
                _write_summary_and_records(run_dir, summary_path, summary)
                return _result(run_dir, summary_path, summary)
            primary_reference_output = generic_quantized_reference.output
            primary_reference_kind = str(generic_quantized_reference.summary.get("reference_kind") or "generic_quantized_taskgraph_replay")

        runtime = execute_upmem_taskgraph_runtime(
            graph=graph,
            network=network,
            case_id=case_id,
            policy=policy,
            quantization_mode=quantization_mode,
            bridge_root=run_dir / "cases" / case_id / "upmem_taskgraph_bridge",
            execute_external=execute_external,
            reference_output=primary_reference_output,
            reference_kind=primary_reference_kind,
            env=env,
            schedule_mode=schedule_mode,
            frontier_worker_count=frontier_worker_count,
            dpu_group_count=dpu_group_count,
            task_assignment_strategy=task_assignment_strategy,
        )
        write_jsonl(run_dir / task_metrics_rel, list(runtime.task_metrics))
        final_tensor_artifact: str | None = None
        if runtime.output is not None:
            (run_dir / final_tensor_rel).parent.mkdir(parents=True, exist_ok=True)
            np.save(run_dir / final_tensor_rel, runtime.output, allow_pickle=False)
            final_tensor_artifact = final_tensor_rel.as_posix()
        summary = dict(runtime.summary)
        summary.update(
            {
                "schema_version": "upmem_taskgraph_runtime_v1",
                "run_schema_version": UPMEM_TASKGRAPH_RUNTIME_RUN_SCHEMA_VERSION,
                "case_id": case_id,
                "circuit": circuit_payload,
                "route_id": route_id,
                "execution_scope": execution_scope,
                "task_metrics_artifact": task_metrics_rel.as_posix(),
                "final_tensor_artifact": final_tensor_artifact,
                "reference": {
                    "kind": primary_reference_kind,
                    "primary_final_validation_reference_kind": primary_reference_kind,
                    "metadata": _reference_metadata_payload(reference_metadata),
                    "cpu_reference_used_to_feed_runtime_tensors": False,
                    "full_precision_reference_is_task_validation_target": primary_reference_kind == "cpu_exact_taskgraph_full_precision",
                    "full_precision_cpu_reference": {
                        "kind": "cpu_exact_taskgraph_full_precision_accuracy_only"
                        if primary_reference_kind != "cpu_exact_taskgraph_full_precision"
                        else "cpu_exact_taskgraph_full_precision_primary",
                        "metadata": _reference_metadata_payload(reference_metadata),
                    },
                },
                "generic_quantized_taskgraph_reference": generic_quantized_reference.to_json_dict()
                if generic_quantized_reference is not None
                else None,
                "final_full_precision_accuracy": _final_accuracy_payload(
                    runtime.output,
                    reference_output,
                    reference_kind="cpu_exact_taskgraph_full_precision",
                ),
                "artifacts": {
                    "task_metrics": task_metrics_rel.as_posix(),
                    **({"final_tensor": final_tensor_artifact} if final_tensor_artifact else {}),
                },
                "metadata": {
                    "developer_only": True,
                    "strict_upmem_taskgraph_runtime": True,
                    "cpu_contraction_fallback_allowed": False,
                    "executes_through": UPMEM_SDK_SIMULATOR_EXECUTES_THROUGH,
                    "primary_final_validation_reference_kind": primary_reference_kind,
                    "full_precision_reference_is_task_validation_target": primary_reference_kind == "cpu_exact_taskgraph_full_precision",
                    "whole_network_quantized_at_initialization": False,
                    "normal_benchmark_routes_unchanged": True,
                    "upmem_frontier_runtime_prototype": False,
                },
            }
        )
        if boundary_case:
            summary["generic_boundary_evidence"] = _generic_boundary_runtime_evidence(graph, summary, list(runtime.task_metrics))
        summary["normalized_result"] = normalized_upmem_taskgraph_result_from_summary(summary)
        _write_summary_and_records(run_dir, summary_path, summary)
        (run_dir / "upmem_taskgraph_runtime_summary.md").write_text(_summary_markdown(summary), encoding="utf-8")
        return _result(run_dir, summary_path, summary)
    except ValueError as exc:
        summary = _error_summary(case, policy, quantization_mode, "unsupported_case_or_runtime_input", str(exc))
        _write_summary_and_records(run_dir, summary_path, summary)
        return _result(run_dir, summary_path, summary)
    except Exception as exc:  # pragma: no cover - defensive CLI boundary
        summary = _error_summary(case, policy, quantization_mode, "unexpected_runtime_harness_exception", str(exc), status="failed")
        _write_summary_and_records(run_dir, summary_path, summary)
        return _result(run_dir, summary_path, summary)


def _upmem_taskgraph_route_label(policy: str, quantization_mode: str, *, schedule_mode: str = "sequential") -> str:
    if policy == "generic-only" and quantization_mode == "none":
        return "upmem_generic_float32"
    if policy == "generic-only" and quantization_mode == "per_task_input_quantize":
        return "upmem_generic_int8"
    return "upmem_taskgraph_runtime"


def _write_summary_and_records(run_dir: Path, summary_path: Path, summary: JsonDict) -> None:
    write_summary_and_normalized_records(
        run_dir,
        summary_path,
        summary,
        lambda payload, source: normalized_upmem_taskgraph_records_from_summary(payload, source_artifact=source),
    )


def _reference_metadata_payload(metadata: JsonDict) -> JsonDict:
    payload = dict(metadata)
    payload.pop("task_metrics", None)
    return to_jsonable(payload)


def _final_accuracy_payload(output: np.ndarray | None, reference_output: np.ndarray | None, *, reference_kind: str) -> JsonDict:
    if output is None:
        return {"passed": False, "reason": "runtime_output_missing", "reference_kind": reference_kind}
    if reference_output is None:
        return {"passed": False, "reason": "reference_output_missing", "reference_kind": reference_kind}
    result = validate(output, reference_output, QUANTIZED_FINAL_VALIDATION_TOLERANCES)
    diff = np.asarray(output, dtype=np.complex128) - np.asarray(reference_output, dtype=np.complex128)
    abs_diff = np.abs(diff)
    return to_jsonable(
        {
            **result.__dict__,
            "reference_kind": reference_kind,
            "tolerance_kind": "quantized_execution_accuracy_tolerance",
            "mean_abs_error": float(abs_diff.mean()) if abs_diff.size else 0.0,
            "max_abs_error": result.max_abs_error,
            "l2_error": result.l2_error,
            "full_precision_cpu_reference_used_to_feed_runtime_tensors": False,
        }
    )


def _summary_markdown(summary: JsonDict) -> str:
    final_validation = dict(summary.get("final_validation") or {})
    full_precision_accuracy = dict(summary.get("final_full_precision_accuracy") or {})
    kernel_counts = dict(summary.get("kernel_family_counts") or {})
    lines = [
        "# UPMEM TaskGraph Runtime",
        "",
        f"Case: `{summary.get('case_id')}`",
        f"Status: `{summary.get('status')}`",
        f"Policy: `{summary.get('policy')}`",
        f"Quantization mode: `{summary.get('quantization_mode')}`",
        "",
        f"This run executes contractions through {UPMEM_SDK_SIMULATOR_EXECUTES_THROUGH}. It is not a hardware benchmark.",
        "",
        "## Kernel Usage",
        "",
        "| Kernel family | Tasks |",
        "| --- | ---: |",
    ]
    for family, count in sorted(kernel_counts.items()):
        lines.append(f"| {family} | {count} |")
    lines.extend(
        [
            "",
            "## Validation",
            "",
            f"Primary reference: `{final_validation.get('reference_kind')}`",
            f"Final validation passed: `{final_validation.get('passed')}`",
            f"Max absolute error: `{final_validation.get('max_abs_error')}`",
            f"L2 error: `{final_validation.get('l2_error')}`",
            f"Full-precision CPU accuracy max absolute error: `{full_precision_accuracy.get('max_abs_error')}`",
            "",
            "Hardware timing and speedup fields are not applicable for this simulator-mode run.",
            "",
        ]
    )
    return "\n".join(lines)


def _generic_boundary_runtime_evidence(graph, summary: JsonDict, task_metrics: list[JsonDict]) -> JsonDict:
    task = graph.tasks[0]
    first_metric = task_metrics[0] if task_metrics else {}
    return to_jsonable(
        {
            "workload_type": "generic_boundary_execution",
            "non_gemm_boundary": True,
            "einsum_expression": task.index_expression,
            "input_shapes": task.input_shapes,
            "output_shape": task.output_shape,
            "input_ranks": (len(task.input_shapes[0]), len(task.input_shapes[1])),
            "output_rank": len(task.output_shape),
            "contracted_labels": task.contracted_labels,
            "output_labels": task.output_labels,
            "native_index_metadata": first_metric.get("native_index_metadata", {}),
            "expected_native_index_metadata": {
                "output_to_left_axes": (0, 1, -1, -1),
                "output_to_right_axes": (-1, -1, 1, 2),
                "contracted_to_left_axes": (2,),
                "contracted_to_right_axes": (0,),
            },
            "validation_target": "expected_quantized_reference_output",
            "full_precision_reference_is_validation_target": False,
            "cpu_fallback_used": summary.get("cpu_fallback_used"),
            "dpu_program_invocations": summary.get("dpu_program_executed_task_count"),
            "upmem_program_executed": summary.get("dpu_program_executed_all_tasks"),
            "valid_primary_upmem_codepath_result": summary.get("valid_primary_upmem_codepath_result"),
        }
    )


def _error_summary(
    case_id: str,
    policy: str,
    quantization_mode: str,
    reason: str,
    error: str | None = None,
    *,
    status: str = "unsupported",
) -> JsonDict:
    from quantum_bench.bench.result_artifacts import normalized_upmem_taskgraph_result_from_summary

    summary = to_jsonable(
        {
            "schema_version": "upmem_taskgraph_runtime_v1",
            "run_schema_version": UPMEM_TASKGRAPH_RUNTIME_RUN_SCHEMA_VERSION,
            "case_id": case_id,
            "status": status,
            "reason": reason,
            "error": error,
            "policy": policy,
            "quantization_mode": quantization_mode,
            "whole_network_quantized_at_initialization": False,
            "contraction_execution_target": CONTRACTION_EXECUTION_TARGET_UPMEM,
            "upmem_execution_mode": UPMEM_EXECUTION_MODE_SDK_SIMULATOR,
            "native_sdk_control_path": True,
            "simplepim_api_used": False,
            "hardware_benchmark_result": False,
            "hardware_timing_available": False,
            "hardware_speedup_applicable": False,
            "cpu_fallback_used": False,
            "valid_primary_upmem_codepath_result": False,
            "total_tasks": 0,
            "executed_tasks": 0,
            "unsupported_tasks": 0,
            "failed_tasks": 1 if status == "failed" else 0,
            "kernel_family_counts": {},
            "backend_counts": {},
            "final_validation": {"passed": False, "reason": "not_available"},
            "metadata": {
                "developer_only": True,
                "strict_upmem_taskgraph_runtime": True,
                "cpu_contraction_fallback_allowed": False,
                "executes_through": UPMEM_SDK_SIMULATOR_EXECUTES_THROUGH,
            },
        }
    )
    summary["normalized_result"] = normalized_upmem_taskgraph_result_from_summary(summary)
    return summary


def _result(run_dir: Path, summary_path: Path, summary: JsonDict) -> UpmemTaskGraphRuntimeRunResult:
    return UpmemTaskGraphRuntimeRunResult(
        schema_version=UPMEM_TASKGRAPH_RUNTIME_RUN_SCHEMA_VERSION,
        status=str(summary.get("status")),
        reason=summary.get("reason"),
        run_dir=run_dir,
        summary_path=summary_path,
        case_id=str(summary.get("case_id")),
        policy=str(summary.get("policy")),
        quantization_mode=str(summary.get("quantization_mode")),
        summary=summary,
    )
