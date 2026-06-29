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
from quantum_bench.targets.upmem import (
    GENERIC_LOOP_BACKEND_ID,
    execute_generic_bridge,
    write_generic_bridge_input_manifest,
)
from quantum_bench.tn import (
    TaskInputMaterializationRequest,
    build_tensor_network,
    materialize_task_inputs,
    plan_task_graph,
)


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
        circuit_params: JsonDict = {"name": case}
        if n_qubits is not None:
            circuit_params["n_qubits"] = int(n_qubits)
        circuit = builtin_circuit(case, circuit_params)
        network = build_tensor_network(circuit)
        graph = plan_task_graph(network)
        if task_index < 0 or task_index >= len(graph.tasks):
            summary = _error_summary("unsupported", "target_task_index_out_of_range", circuit.name, task_index, backend)
            write_json(summary_path, summary)
            return _result(run_dir, summary_path, summary)

        initial_tensors = {tensor.spec.id: tensor for tensor in network.tensors}
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
                case_id=circuit.name,
                circuit_payload=manifest(circuit),
                task_index=task_index,
                task_id=task.id,
                backend=backend,
                materialization=materialization.to_json_dict(),
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
                case_id=circuit.name,
                circuit_payload=manifest(circuit),
                task_index=task_index,
                task_id=task.id,
                backend=backend,
                materialization=materialization.to_json_dict(),
                preparation=preparation.to_json_dict(),
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
            case_id=circuit.name,
            circuit_payload=manifest(circuit),
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
) -> JsonDict:
    validation = {}
    if bridge_result and bridge_result.get("output_manifest"):
        validation = dict(bridge_result["output_manifest"].get("validation_metrics") or {})
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
            "execution_scope": "task_level",
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
            "metadata": {
                "developer_only": True,
                "one_task_only": True,
                "normal_routing_unchanged": True,
                "task_level_simulator_execution_only": True,
                "simplepim_api_used": False,
                "native_sdk_control_path": True,
                "validation_target": "expected_quantized_reference_output",
                "full_precision_reference_is_validation_target": False,
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
            "execution_scope": "task_level",
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
