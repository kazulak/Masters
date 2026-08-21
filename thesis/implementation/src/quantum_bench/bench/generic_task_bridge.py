from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Mapping

from quantum_bench.bench.run_dirs import create_run_dir
from quantum_bench.circuits import builtin_circuit, manifest
from quantum_bench.core.jsonio import write_json
from quantum_bench.core.records import JsonDict, to_jsonable
from quantum_bench.environment import capture_environment
from quantum_bench.routing import GenericTaskPreparationInput, prepare_generic_task
from quantum_bench.targets.upmem.generic_boundary import build_generic_boundary_workload, is_generic_boundary_case
from quantum_bench.targets.upmem.generic_bridge import (
    GENERIC_LOOP_BACKEND_ID,
    execute_generic_bridge,
    write_generic_bridge_input_manifest,
)
from quantum_bench.tn.materialize import (
    TaskInputMaterializationRequest,
    materialize_task_inputs,
)
from quantum_bench.tn.network import build_tensor_network
from quantum_bench.tn.task_graph import plan_task_graph


GENERIC_TASK_BRIDGE_SCHEMA_VERSION = "generic_task_bridge_v1"
GenericTaskBridgeStatus = Literal["completed", "skipped", "unsupported", "failed"]
_SUCCESSFUL_STATUSES = {"upmem_sdk_simulator_generic_loop_executed"}


@dataclass(frozen=True)
class GenericTaskBridgeResult:
    schema_version: str
    status: GenericTaskBridgeStatus
    reason: str | None
    run_dir: Path
    summary_path: Path
    case_id: str
    task_index: int
    task_id: str | None
    backend: str
    external_command_executed: bool
    execution_implemented: bool
    summary: JsonDict = field(default_factory=dict)

    def to_json_dict(self) -> JsonDict:
        payload = to_jsonable(self)
        payload["run_dir"] = str(self.run_dir)
        payload["summary_path"] = str(self.summary_path)
        return payload


def run_generic_task_bridge(
    root_dir: Path,
    *,
    case: str = "bell_2q",
    n_qubits: int | None = None,
    task_index: int = 0,
    backend: str = GENERIC_LOOP_BACKEND_ID,
    execute_external: bool = False,
    env: Mapping[str, str] | None = None,
) -> GenericTaskBridgeResult:
    run_dir = create_run_dir(root_dir, f"{case}_generic_task_bridge")
    summary_path = run_dir / "generic_task_bridge_summary.json"
    write_json(run_dir / "environment.json", capture_environment(root_dir))

    try:
        if backend != GENERIC_LOOP_BACKEND_ID:
            summary = _error_summary("unsupported", "unsupported_backend", case, task_index, backend)
            write_json(summary_path, summary)
            return _result(run_dir, summary_path, summary)
        boundary_case = is_generic_boundary_case(case)
        if boundary_case:
            if n_qubits is not None:
                summary = _error_summary("unsupported", "generic_boundary_does_not_accept_n_qubits", case, task_index, backend)
                write_json(summary_path, summary)
                return _result(run_dir, summary_path, summary)
            workload = build_generic_boundary_workload(case)
            graph = workload.graph
            network = workload.network
            case_id = workload.case_id
            circuit_payload = workload.manifest
            initial_tensors = {tensor.spec.id: tensor for tensor in network.tensors}
        else:
            circuit_params: JsonDict = {"name": case}
            if n_qubits is not None:
                circuit_params["n_qubits"] = int(n_qubits)
            circuit = builtin_circuit(case, circuit_params)
            network = build_tensor_network(circuit)
            graph = plan_task_graph(network)
            case_id = circuit.name
            circuit_payload = manifest(circuit)
            initial_tensors = {tensor.spec.id: tensor for tensor in network.tensors}
        if task_index < 0 or task_index >= len(graph.tasks):
            summary = _error_summary("unsupported", "target_task_index_out_of_range", case_id, task_index, backend)
            write_json(summary_path, summary)
            return _result(run_dir, summary_path, summary)

        materialization = materialize_task_inputs(
            TaskInputMaterializationRequest(
                graph=graph,
                initial_tensors=initial_tensors,
                target_task_index=task_index,
            )
        )
        task = graph.tasks[task_index]
        if materialization.status not in {"initial_inputs_available", "materialized"} or materialization.left_tensor is None or materialization.right_tensor is None:
            summary = _base_summary(
                status="unsupported" if materialization.status == "unsupported" else "failed",
                reason=materialization.reason or "materialized_inputs_missing",
                case_id=case_id,
                circuit_payload=circuit_payload,
                task_index=task_index,
                task_id=task.id,
                backend=backend,
                materialization=materialization.to_json_dict(),
                boundary_evidence=_boundary_evidence(task) if boundary_case else None,
            )
            write_json(summary_path, summary)
            return _result(run_dir, summary_path, summary)

        preparation = prepare_generic_task(
            GenericTaskPreparationInput(
                task=task,
                left_tensor=materialization.left_tensor,
                right_tensor=materialization.right_tensor,
            )
        )
        if preparation.status != "prepared":
            summary = _base_summary(
                status="unsupported" if preparation.status == "unsupported_shape" else "failed",
                reason=preparation.reason,
                case_id=case_id,
                circuit_payload=circuit_payload,
                task_index=task_index,
                task_id=task.id,
                backend=backend,
                materialization=materialization.to_json_dict(),
                preparation=preparation.to_json_dict(),
                boundary_evidence=_boundary_evidence(task, preparation=preparation.to_json_dict()) if boundary_case else None,
            )
            write_json(summary_path, summary)
            return _result(run_dir, summary_path, summary)

        bridge_dir = run_dir / "bridge"
        input_manifest = write_generic_bridge_input_manifest(preparation, bridge_dir)
        bridge_result = execute_generic_bridge(
            bridge_dir / "input_manifest.json",
            backend=backend,
            execute_external=execute_external,
            env=env,
        )
        status, reason = _bridge_status(bridge_result.execution_status, bridge_result.reason)
        summary = _base_summary(
            status=status,
            reason=reason,
            case_id=case_id,
            circuit_payload=circuit_payload,
            task_index=task_index,
            task_id=task.id,
            backend=backend,
            materialization=materialization.to_json_dict(),
            preparation=preparation.to_json_dict(),
            bridge_result=bridge_result.to_json_dict(),
            artifacts={
                "input_manifest": "bridge/input_manifest.json",
                "output_manifest": "bridge/output_manifest.json",
                **({"output_blob": f"bridge/{bridge_result.output_blob_path}"} if bridge_result.output_blob_path else {}),
            },
            input_manifest=input_manifest.to_json_dict(),
            boundary_evidence=(
                _boundary_evidence(
                    task,
                    preparation=preparation.to_json_dict(),
                    bridge_result=bridge_result.to_json_dict(),
                )
                if boundary_case
                else None
            ),
        )
        write_json(summary_path, summary)
        return _result(run_dir, summary_path, summary)
    except ValueError as exc:
        summary = _error_summary("unsupported", "unsupported_builtin_case", case, task_index, backend, str(exc))
        write_json(summary_path, summary)
        return _result(run_dir, summary_path, summary)
    except Exception as exc:  # pragma: no cover - defensive harness boundary
        summary = _error_summary("failed", "unexpected_harness_exception", case, task_index, backend, str(exc))
        write_json(summary_path, summary)
        return _result(run_dir, summary_path, summary)


def _bridge_status(status: str, reason: str | None) -> tuple[GenericTaskBridgeStatus, str | None]:
    if status in _SUCCESSFUL_STATUSES:
        return "completed", None
    if status in {"skipped", "not_implemented"}:
        return "skipped", reason or status
    if status == "unsupported":
        return "unsupported", reason or status
    return "failed", reason or status


def _base_summary(
    *,
    status: GenericTaskBridgeStatus,
    reason: str | None,
    case_id: str,
    circuit_payload: JsonDict,
    task_index: int,
    task_id: str | None,
    backend: str,
    materialization: JsonDict | None = None,
    preparation: JsonDict | None = None,
    bridge_result: JsonDict | None = None,
    artifacts: JsonDict | None = None,
    input_manifest: JsonDict | None = None,
    boundary_evidence: JsonDict | None = None,
) -> JsonDict:
    validation = {}
    if bridge_result and bridge_result.get("output_manifest"):
        validation = dict(bridge_result["output_manifest"].get("validation_metrics") or {})
    bridge_status = bridge_result.get("execution_status") if bridge_result else None
    upmem_program_executed = bridge_status in _SUCCESSFUL_STATUSES
    summary = to_jsonable(
        {
            "schema_version": GENERIC_TASK_BRIDGE_SCHEMA_VERSION,
            "status": status,
            "reason": reason,
            "case_id": case_id,
            "circuit": circuit_payload,
            "task_index": task_index,
            "task_id": task_id,
            "route_id": "generic_loop_fallback",
            "bridge_backend_id": backend,
            "kernel_family": "generic_loop_fallback",
            "execution_target": "upmem_simulator",
            "contraction_execution_target": "upmem",
            "upmem_execution_mode": "sdk_simulator",
            "execution_backend": "upmem_sdk",
            "execution_scope": "task_level",
            "cpu_fallback_used": False,
            "dpu_program_invocations": 1 if upmem_program_executed else 0,
            "upmem_program_executed": upmem_program_executed,
            "native_sdk_control_path": True,
            "simplepim_api_used": False,
            "hardware_execution": False,
            "hardware_timing_available": False,
            "hardware_speedup_applicable": False,
            "materialization": materialization or {},
            "preparation": preparation or {},
            "bridge_result": bridge_result or {},
            "bridge_execution_status": bridge_result.get("execution_status") if bridge_result else None,
            "bridge_execution_reason": bridge_result.get("reason") if bridge_result else None,
            "bridge_validation_metrics": validation,
            "external_command_executed": bool(bridge_result.get("external_command_executed", False)) if bridge_result else False,
            "execution_implemented": bool(bridge_result.get("execution_implemented", False)) if bridge_result else False,
            "artifacts": artifacts or {},
            "input_manifest_audit": input_manifest or {},
            "generic_boundary_evidence": boundary_evidence or {},
            "metadata": {
                "developer_only": True,
                "one_task_only": True,
                "normal_routing_unchanged": True,
                "task_level_simulator_execution_only": True,
                "simplepim_api_used": False,
                "native_sdk_control_path": True,
                "validation_target": "expected_quantized_reference_output",
                "full_precision_reference_is_validation_target": False,
                "generic_boundary_execution": bool(boundary_evidence),
            },
        }
    )
    from quantum_bench.bench.result_artifacts import normalized_task_result_from_summary

    summary["normalized_result"] = normalized_task_result_from_summary(summary)
    return summary


def _error_summary(
    status: GenericTaskBridgeStatus,
    reason: str,
    case_id: str,
    task_index: int,
    backend: str,
    error: str | None = None,
) -> JsonDict:
    summary = to_jsonable(
        {
            "schema_version": GENERIC_TASK_BRIDGE_SCHEMA_VERSION,
            "status": status,
            "reason": reason,
            "error": error,
            "case_id": case_id,
            "task_index": task_index,
            "task_id": None,
            "route_id": "generic_loop_fallback",
            "bridge_backend_id": backend,
            "kernel_family": "generic_loop_fallback",
            "execution_target": "upmem_simulator",
            "contraction_execution_target": "upmem",
            "upmem_execution_mode": "sdk_simulator",
            "execution_backend": "upmem_sdk",
            "execution_scope": "task_level",
            "cpu_fallback_used": False,
            "dpu_program_invocations": 0,
            "upmem_program_executed": False,
            "native_sdk_control_path": True,
            "simplepim_api_used": False,
            "hardware_execution": False,
            "hardware_timing_available": False,
            "hardware_speedup_applicable": False,
            "external_command_executed": False,
            "execution_implemented": False,
            "artifacts": {},
            "metadata": {
                "developer_only": True,
                "normal_routing_unchanged": True,
                "simplepim_api_used": False,
                "native_sdk_control_path": True,
            },
        }
    )
    from quantum_bench.bench.result_artifacts import normalized_task_result_from_summary

    summary["normalized_result"] = normalized_task_result_from_summary(summary)
    return summary


def _boundary_evidence(
    task,
    *,
    preparation: JsonDict | None = None,
    bridge_result: JsonDict | None = None,
) -> JsonDict:
    native_metadata = dict((preparation or {}).get("native_index_metadata") or {})
    bridge_status = str((bridge_result or {}).get("execution_status") or "")
    upmem_program_executed = bridge_status in _SUCCESSFUL_STATUSES
    return to_jsonable(
        {
            "workload_type": "generic_boundary_execution",
            "non_gemm_boundary": True,
            "einsum_expression": task.index_expression,
            "input_shapes": task.input_shapes,
            "output_shape": task.output_shape,
            "input_ranks": (len(task.input_shapes[0]), len(task.input_shapes[1])),
            "output_rank": len(task.output_shape),
            "left_labels": task.left_labels,
            "right_labels": task.right_labels,
            "contracted_labels": task.contracted_labels,
            "output_labels": task.output_labels,
            "native_index_metadata": native_metadata,
            "expected_native_index_metadata": {
                "output_to_left_axes": (0, 1, -1, -1),
                "output_to_right_axes": (-1, -1, 1, 2),
                "contracted_to_left_axes": (2,),
                "contracted_to_right_axes": (0,),
            },
            "validation_target": "expected_quantized_reference_output",
            "full_precision_reference_is_validation_target": False,
            "cpu_fallback_used": False,
            "dpu_program_invocations": 1 if upmem_program_executed else 0,
            "upmem_program_executed": upmem_program_executed,
            "bridge_execution_status": bridge_status or None,
        }
    )


def _result(run_dir: Path, summary_path: Path, summary: JsonDict) -> GenericTaskBridgeResult:
    return GenericTaskBridgeResult(
        schema_version=GENERIC_TASK_BRIDGE_SCHEMA_VERSION,
        status=summary["status"],
        reason=summary.get("reason"),
        run_dir=run_dir,
        summary_path=summary_path,
        case_id=summary["case_id"],
        task_index=int(summary["task_index"]),
        task_id=summary.get("task_id"),
        backend=summary["bridge_backend_id"],
        external_command_executed=bool(summary.get("external_command_executed", False)),
        execution_implemented=bool(summary.get("execution_implemented", False)),
        summary=summary,
    )
