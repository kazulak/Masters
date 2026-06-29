from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

import numpy as np

from quantum_bench.bench.run_dirs import create_run_dir
from quantum_bench.circuits import builtin_circuit, manifest
from quantum_bench.core.jsonio import write_json, write_jsonl
from quantum_bench.core.records import JsonDict, to_jsonable
from quantum_bench.environment import capture_environment
from quantum_bench.targets.upmem import annotate_task_graph_with_upmem_estimates
from quantum_bench.targets.upmem.taskgraph_runtime import (
    UPMEM_TASKGRAPH_POLICIES,
    UPMEM_TASKGRAPH_QUANTIZATION_MODES,
    UpmemTaskGraphPolicy,
    UpmemTaskGraphQuantizationMode,
    execute_upmem_taskgraph_runtime,
)
from quantum_bench.tn import build_tensor_network, execute_task_sequence_np_einsum, plan_task_graph_with_config, with_path_cost_summary


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
) -> UpmemTaskGraphRuntimeRunResult:
    run_dir = create_run_dir(root_dir, f"{case}_upmem_taskgraph_runtime")
    summary_path = run_dir / "upmem_taskgraph_runtime_summary.json"
    final_tensor_rel = Path("raw") / "final_tensor.npy"
    write_json(run_dir / "environment.json", capture_environment(root_dir))

    try:
        if policy not in UPMEM_TASKGRAPH_POLICIES:
            summary = _error_summary(case, policy, quantization_mode, "unsupported_policy")
            write_json(summary_path, summary)
            return _result(run_dir, summary_path, summary)
        if quantization_mode not in UPMEM_TASKGRAPH_QUANTIZATION_MODES:
            summary = _error_summary(case, policy, quantization_mode, "unsupported_quantization_mode")
            write_json(summary_path, summary)
            return _result(run_dir, summary_path, summary)

        circuit_params: JsonDict = {"name": case}
        if n_qubits is not None:
            circuit_params["n_qubits"] = int(n_qubits)
        circuit = builtin_circuit(case, circuit_params)
        task_metrics_rel = Path("cases") / circuit.name / "upmem_taskgraph_task_metrics.jsonl"
        network = build_tensor_network(circuit)
        graph = plan_task_graph_with_config(network, None)
        graph, _ = annotate_task_graph_with_upmem_estimates(graph)
        graph = with_path_cost_summary(graph)
        reference_output, reference_metadata = execute_task_sequence_np_einsum(graph, network)

        runtime = execute_upmem_taskgraph_runtime(
            graph=graph,
            network=network,
            case_id=circuit.name,
            policy=policy,
            quantization_mode=quantization_mode,
            bridge_root=run_dir / "cases" / circuit.name / "upmem_taskgraph_bridge",
            execute_external=execute_external,
            reference_output=reference_output,
            env=env,
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
                "case_id": circuit.name,
                "circuit": manifest(circuit),
                "route_id": "upmem_tn_runtime",
                "execution_scope": "full_taskgraph",
                "task_metrics_artifact": task_metrics_rel.as_posix(),
                "final_tensor_artifact": final_tensor_artifact,
                "reference": {
                    "kind": "cpu_exact_taskgraph_full_precision_final_validation_only",
                    "metadata": _reference_metadata_payload(reference_metadata),
                    "cpu_reference_used_to_feed_runtime_tensors": False,
                },
                "artifacts": {
                    "task_metrics": task_metrics_rel.as_posix(),
                    **({"final_tensor": final_tensor_artifact} if final_tensor_artifact else {}),
                },
                "metadata": {
                    "developer_only": True,
                    "strict_upmem_taskgraph_runtime": True,
                    "cpu_contraction_fallback_allowed": False,
                    "executes_through": "UPMEM SDK DPU programs using SDK simulator mode",
                    "full_precision_reference_is_task_validation_target": False,
                    "whole_network_quantized_at_initialization": False,
                    "normal_benchmark_routes_unchanged": True,
                },
            }
        )
        from quantum_bench.bench.result_artifacts import normalized_upmem_taskgraph_result_from_summary

        summary["normalized_result"] = normalized_upmem_taskgraph_result_from_summary(summary)
        write_json(summary_path, summary)
        (run_dir / "upmem_taskgraph_runtime_summary.md").write_text(_summary_markdown(summary), encoding="utf-8")
        return _result(run_dir, summary_path, summary)
    except ValueError as exc:
        summary = _error_summary(case, policy, quantization_mode, "unsupported_case_or_runtime_input", str(exc))
        write_json(summary_path, summary)
        return _result(run_dir, summary_path, summary)
    except Exception as exc:  # pragma: no cover - defensive CLI boundary
        summary = _error_summary(case, policy, quantization_mode, "unexpected_runtime_harness_exception", str(exc), status="failed")
        write_json(summary_path, summary)
        return _result(run_dir, summary_path, summary)


def _reference_metadata_payload(metadata: JsonDict) -> JsonDict:
    payload = dict(metadata)
    payload.pop("task_metrics", None)
    return to_jsonable(payload)


def _summary_markdown(summary: JsonDict) -> str:
    final_validation = dict(summary.get("final_validation") or {})
    kernel_counts = dict(summary.get("kernel_family_counts") or {})
    lines = [
        "# UPMEM TaskGraph Runtime",
        "",
        f"Case: `{summary.get('case_id')}`",
        f"Status: `{summary.get('status')}`",
        f"Policy: `{summary.get('policy')}`",
        f"Quantization mode: `{summary.get('quantization_mode')}`",
        "",
        "This run executes contractions through UPMEM SDK DPU programs using SDK simulator mode. It is not a hardware benchmark.",
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
            f"Final validation passed: `{final_validation.get('passed')}`",
            f"Max absolute error: `{final_validation.get('max_abs_error')}`",
            f"L2 error: `{final_validation.get('l2_error')}`",
            "",
            "Hardware timing and speedup fields are not applicable for this simulator-mode run.",
            "",
        ]
    )
    return "\n".join(lines)


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
            "contraction_execution_target": "upmem",
            "upmem_execution_mode": "sdk_simulator",
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
                "executes_through": "UPMEM SDK DPU programs using SDK simulator mode",
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
