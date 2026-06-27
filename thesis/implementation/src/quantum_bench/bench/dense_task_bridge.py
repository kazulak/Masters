from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal, Mapping

from quantum_bench.bench.run_dirs import sanitize, update_latest_symlink
from quantum_bench.circuits import builtin_circuit, manifest
from quantum_bench.core.jsonio import write_json
from quantum_bench.core.records import JsonDict, to_jsonable
from quantum_bench.environment import capture_environment
from quantum_bench.routing import DenseTaskPreparationInput, prepare_dense_task
from quantum_bench.targets.upmem import (
    SimplePimProbeResult,
    annotate_task_graph_with_upmem_estimates,
    execute_dense_bridge,
    probe_simplepim,
    write_dense_bridge_input_manifest,
)
from quantum_bench.tn import (
    TASK_INPUT_MATERIALIZATION_SCHEMA_VERSION,
    TaskInputMaterializationRequest,
    TaskInputMaterializationResult,
    build_tensor_network,
    materialize_task_inputs,
    plan_task_graph,
)


DENSE_TASK_BRIDGE_SCHEMA_VERSION = "dense_task_bridge_v1"
DenseTaskBridgeStatus = Literal["completed", "skipped", "unsupported", "failed"]
DenseTaskBridgeMaterializationMode = Literal["initial-only", "cpu-replay"]
_BRIDGEABLE_PREPARATION_STATUSES = {"prepared", "simplepim_unavailable"}
_SUCCESSFUL_BRIDGE_STATUSES = {"mock_executed", "upmem_sdk_simulator_executed"}


@dataclass(frozen=True)
class DenseTaskBridgeResult:
    schema_version: str
    status: DenseTaskBridgeStatus
    reason: str | None
    run_dir: Path
    summary_path: Path
    case_id: str
    task_index: int | None
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


def run_dense_task_bridge(
    root_dir: Path,
    *,
    case: str = "bell_2q",
    n_qubits: int | None = None,
    task_index: int | None = None,
    backend: str = "mock_numpy_dequantized",
    execute_external: bool = False,
    materialization: DenseTaskBridgeMaterializationMode = "initial-only",
    env: Mapping[str, str] | None = None,
) -> DenseTaskBridgeResult:
    run_dir = _create_dense_task_bridge_run_dir(root_dir)
    summary_path = run_dir / "dense_task_bridge_summary.json"
    write_json(run_dir / "environment.json", capture_environment(root_dir))

    try:
        circuit = _build_builtin_circuit(case, n_qubits)
        network = build_tensor_network(circuit)
        graph = plan_task_graph(network)
        graph, _ = annotate_task_graph_with_upmem_estimates(graph)
        initial_tensors = {tensor.spec.id: tensor for tensor in network.tensors}
        probe = probe_simplepim(env=env) if env is not None else probe_simplepim()

        if materialization not in {"initial-only", "cpu-replay"}:
            summary = _base_summary(
                status="unsupported",
                reason="unsupported_materialization_mode",
                case_id=circuit.name,
                circuit_payload=manifest(circuit),
                task_index=task_index,
                task=None,
                backend=backend,
                probe=probe,
                materialization=_generic_materialization_payload(
                    materialization,
                    status="unsupported",
                    reason="unsupported_materialization_mode",
                    task_index=task_index,
                ),
            )
            write_json(summary_path, summary)
            return _result_from_summary(run_dir, summary_path, summary)

        if materialization == "cpu-replay" and task_index is not None:
            selection = _select_task_by_index(graph.tasks, task_index)
        else:
            selection = _select_task(graph.tasks, initial_tensors, task_index, probe)
        if selection["status"] != "selected":
            task = selection.get("task")
            summary = _base_summary(
                status=selection["harness_status"],
                reason=selection["reason"],
                case_id=circuit.name,
                circuit_payload=manifest(circuit),
                task_index=selection.get("task_index"),
                task=task,
                backend=backend,
                probe=probe,
                materialization=_selection_materialization_payload(
                    mode=materialization,
                    task=task,
                    task_index=selection.get("task_index"),
                    initial_tensors=initial_tensors,
                    status="unsupported",
                    reason=selection["reason"],
                ),
            )
            write_json(summary_path, summary)
            return _result_from_summary(run_dir, summary_path, summary)

        task = selection["task"]
        selected_index = int(selection["task_index"])
        materialization_payload: JsonDict
        if materialization == "cpu-replay" and task_index is not None:
            materialization_result = materialize_task_inputs(
                TaskInputMaterializationRequest(
                    graph=graph,
                    initial_tensors=initial_tensors,
                    target_task_index=selected_index,
                )
            )
            materialization_payload = _materialization_payload(materialization, materialization_result)
            if materialization_result.status not in {"initial_inputs_available", "materialized"}:
                summary = _base_summary(
                    status="failed" if materialization_result.status == "failed" else "unsupported",
                    reason=materialization_result.reason,
                    case_id=circuit.name,
                    circuit_payload=manifest(circuit),
                    task_index=selected_index,
                    task=task,
                    backend=backend,
                    probe=probe,
                    materialization=materialization_payload,
                )
                write_json(summary_path, summary)
                return _result_from_summary(run_dir, summary_path, summary)
            if materialization_result.left_tensor is None or materialization_result.right_tensor is None:
                summary = _base_summary(
                    status="failed",
                    reason="materialized_inputs_missing",
                    case_id=circuit.name,
                    circuit_payload=manifest(circuit),
                    task_index=selected_index,
                    task=task,
                    backend=backend,
                    probe=probe,
                    materialization=materialization_payload,
                )
                write_json(summary_path, summary)
                return _result_from_summary(run_dir, summary_path, summary)
            preparation = _prepare_task_from_tensors(
                task,
                materialization_result.left_tensor,
                materialization_result.right_tensor,
                probe,
            )
        else:
            materialization_payload = _selection_materialization_payload(
                mode=materialization,
                task=task,
                task_index=selected_index,
                initial_tensors=initial_tensors,
                status="initial_inputs_available",
                reason=None,
            )
            preparation = selection.get("preparation") or _prepare_task(task, initial_tensors, probe)
        if not _is_bridgeable_preparation(preparation):
            status, reason = _preparation_status(preparation)
            summary = _base_summary(
                status=status,
                reason=reason,
                case_id=circuit.name,
                circuit_payload=manifest(circuit),
                task_index=selected_index,
                task=task,
                backend=backend,
                probe=probe,
                preparation=preparation,
                materialization=materialization_payload,
            )
            write_json(summary_path, summary)
            return _result_from_summary(run_dir, summary_path, summary)

        bridge_dir = run_dir / "bridge"
        input_manifest = write_dense_bridge_input_manifest(preparation, bridge_dir)
        bridge_result = execute_dense_bridge(
            bridge_dir / "input_manifest.json",
            backend=backend,
            execute_external=execute_external,
            env=env,
        )
        status, reason = _bridge_status(bridge_result.execution_status, backend, bridge_result.reason)
        summary = _base_summary(
            status=status,
            reason=reason,
            case_id=circuit.name,
            circuit_payload=manifest(circuit),
            task_index=selected_index,
            task=task,
            backend=backend,
            probe=probe,
            preparation=preparation,
            bridge_result=bridge_result,
            input_manifest=input_manifest,
            run_dir=run_dir,
            materialization=materialization_payload,
        )
        write_json(summary_path, summary)
        return _result_from_summary(run_dir, summary_path, summary)
    except ValueError as exc:
        summary = _error_summary(
            status="unsupported",
            reason="unsupported_builtin_case",
            error=str(exc),
            case_id=case,
            task_index=task_index,
            backend=backend,
            materialization_mode=materialization,
        )
        write_json(summary_path, summary)
        return _result_from_summary(run_dir, summary_path, summary)
    except Exception as exc:  # pragma: no cover - defensive harness boundary
        summary = _error_summary(
            status="failed",
            reason="unexpected_harness_exception",
            error=str(exc),
            case_id=case,
            task_index=task_index,
            backend=backend,
            materialization_mode=materialization,
        )
        write_json(summary_path, summary)
        return _result_from_summary(run_dir, summary_path, summary)


def _build_builtin_circuit(case: str, n_qubits: int | None):
    if case.lower() == "bell_2q" and n_qubits not in {None, 2}:
        raise ValueError("bell_2q only supports --n-qubits 2")
    params: JsonDict = {"name": case}
    if n_qubits is not None:
        params["n_qubits"] = n_qubits
    return builtin_circuit(case, params)


def _select_task(
    tasks: tuple[object, ...],
    initial_tensors: dict[str, object],
    task_index: int | None,
    probe: SimplePimProbeResult,
) -> JsonDict:
    if task_index is not None:
        if task_index < 0 or task_index >= len(tasks):
            return {
                "status": "not_selected",
                "harness_status": "unsupported",
                "reason": "task_index_out_of_range",
                "task_index": task_index,
            }
        task = tasks[task_index]
        if not _inputs_available(task, initial_tensors):
            return {
                "status": "not_selected",
                "harness_status": "unsupported",
                "reason": "intermediate_tensor_inputs_not_materialized",
                "task_index": task_index,
                "task": task,
            }
        return {"status": "selected", "task": task, "task_index": task_index}

    for index, task in enumerate(tasks):
        if _inputs_available(task, initial_tensors):
            preparation = _prepare_task(task, initial_tensors, probe)
            if _is_bridgeable_preparation(preparation):
                return {"status": "selected", "task": task, "task_index": index, "preparation": preparation}
    return {
        "status": "not_selected",
        "harness_status": "skipped",
        "reason": "no_initial_input_dense_task_available",
        "task_index": None,
    }


def _select_task_by_index(tasks: tuple[object, ...], task_index: int) -> JsonDict:
    if task_index < 0 or task_index >= len(tasks):
        return {
            "status": "not_selected",
            "harness_status": "unsupported",
            "reason": "target_task_index_out_of_range",
            "task_index": task_index,
        }
    return {"status": "selected", "task": tasks[task_index], "task_index": task_index}


def _inputs_available(task: object, initial_tensors: dict[str, object]) -> bool:
    return all(tensor_id in initial_tensors for tensor_id in task.input_tensor_ids)


def _prepare_task(task: object, initial_tensors: dict[str, object], probe: SimplePimProbeResult):
    return prepare_dense_task(
        DenseTaskPreparationInput(
            task=task,
            left_tensor=initial_tensors[task.input_tensor_ids[0]],
            right_tensor=initial_tensors[task.input_tensor_ids[1]],
            simplepim_probe=probe,
        )
    )


def _prepare_task_from_tensors(
    task: object,
    left_tensor: object,
    right_tensor: object,
    probe: SimplePimProbeResult,
):
    return prepare_dense_task(
        DenseTaskPreparationInput(
            task=task,
            left_tensor=left_tensor,
            right_tensor=right_tensor,
            simplepim_probe=probe,
        )
    )


def _is_bridgeable_preparation(preparation: object) -> bool:
    return (
        preparation.status in _BRIDGEABLE_PREPARATION_STATUSES
        and preparation.prepared_operands is not None
        and preparation.tile_plan is not None
        and preparation.tile_plan.get("requires_tiling") is not True
        and preparation.tile_plan.get("requires_host_aggregation") is not True
    )


def _preparation_status(preparation: object) -> tuple[DenseTaskBridgeStatus, str]:
    if preparation.status in {"unsupported_shape", "requires_executable_tiling_not_implemented"}:
        return "unsupported", str(preparation.reason or preparation.status)
    if preparation.status == "failed":
        return "failed", str(preparation.reason or "dense_preparation_failed")
    return "unsupported", str(preparation.reason or f"non_bridgeable_preparation_status:{preparation.status}")


def _bridge_status(bridge_status: str, backend: str, bridge_reason: str | None) -> tuple[DenseTaskBridgeStatus, str | None]:
    if bridge_status in _SUCCESSFUL_BRIDGE_STATUSES:
        return "completed", None
    if bridge_status == "stub_executed":
        return "completed", bridge_reason or "external_stub_contract_executed"
    if backend in {"simplepim_external", "simplepim_external_stub"} and bridge_status in {"skipped", "not_implemented"}:
        return "skipped", bridge_reason or bridge_status
    if bridge_status == "unsupported":
        return "unsupported", bridge_reason or bridge_status
    if bridge_status == "failed":
        return "failed", bridge_reason or bridge_status
    return "failed", bridge_reason or f"unexpected_bridge_status:{bridge_status}"


def _base_summary(
    *,
    status: DenseTaskBridgeStatus,
    reason: str | None,
    case_id: str,
    circuit_payload: JsonDict,
    task_index: int | None,
    task: object | None,
    backend: str,
    probe: SimplePimProbeResult,
    preparation: object | None = None,
    bridge_result: object | None = None,
    input_manifest: object | None = None,
    run_dir: Path | None = None,
    materialization: JsonDict | None = None,
) -> JsonDict:
    bridge_output_manifest = bridge_result.output_manifest if bridge_result is not None else None
    summary: JsonDict = {
        "schema_version": DENSE_TASK_BRIDGE_SCHEMA_VERSION,
        "status": status,
        "reason": reason,
        "case_id": case_id,
        "workload": circuit_payload,
        "task_index": task_index,
        "task_id": task.id if task is not None else None,
        "route_id": "dense_gemm",
        "bridge_backend_id": backend,
        "bridge_execution_status": bridge_result.execution_status if bridge_result is not None else None,
        "bridge_execution_reason": bridge_result.reason if bridge_result is not None else None,
        "bridge_execution_error": bridge_result.error if bridge_result is not None else None,
        "bridge_execution_error_type": bridge_result.error_type if bridge_result is not None else None,
        "simplepim_probe": probe.to_json_dict(),
        "tile_plan": preparation.tile_plan if preparation is not None else None,
        "fixed_point_conversion": {
            "left": to_jsonable(preparation.left_conversion) if preparation is not None else None,
            "right": to_jsonable(preparation.right_conversion) if preparation is not None else None,
        },
        "preparation": {
            "status": preparation.status if preparation is not None else None,
            "reason": preparation.reason if preparation is not None else None,
            "error": preparation.error if preparation is not None else None,
            "validation_metrics": to_jsonable(preparation.validation_metrics) if preparation is not None else None,
        },
        "bridge_validation_metrics": (
            bridge_output_manifest.validation_metrics if bridge_output_manifest is not None else None
        ),
        "materialization": materialization or _generic_materialization_payload(
            "initial-only",
            status="unsupported",
            reason="materialization_not_recorded",
            task_index=task_index,
        ),
        "artifacts": _artifact_paths(run_dir, bridge_result),
        "external_command_executed": (
            bool(bridge_result.external_command_executed) if bridge_result is not None else False
        ),
        "execution_implemented": (
            bool(bridge_result.execution_implemented) if bridge_result is not None else False
        ),
        "metadata": {
            "developer_only": True,
            "one_task_only": True,
            "normal_routing_unchanged": True,
            "bridge_manifest_written": input_manifest is not None,
        },
    }
    return summary


def _artifact_paths(run_dir: Path | None, bridge_result: object | None) -> JsonDict:
    if run_dir is None:
        return {}
    paths: JsonDict = {"input_manifest": "bridge/input_manifest.json"}
    if bridge_result is not None and bridge_result.output_manifest_path is not None:
        paths["output_manifest"] = f"bridge/{bridge_result.output_manifest_path}"
    if bridge_result is not None and bridge_result.output_blob_path is not None:
        paths["output_blob"] = f"bridge/{bridge_result.output_blob_path}"
    return paths


def _error_summary(
    *,
    status: DenseTaskBridgeStatus,
    reason: str,
    error: str,
    case_id: str,
    task_index: int | None,
    backend: str,
    materialization_mode: str = "initial-only",
) -> JsonDict:
    return {
        "schema_version": DENSE_TASK_BRIDGE_SCHEMA_VERSION,
        "status": status,
        "reason": reason,
        "case_id": case_id,
        "workload": None,
        "task_index": task_index,
        "task_id": None,
        "route_id": "dense_gemm",
        "bridge_backend_id": backend,
        "bridge_execution_status": None,
        "bridge_execution_reason": None,
        "bridge_execution_error": None,
        "bridge_execution_error_type": None,
        "simplepim_probe": None,
        "tile_plan": None,
        "fixed_point_conversion": {"left": None, "right": None},
        "preparation": {"status": None, "reason": None, "error": error, "validation_metrics": None},
        "bridge_validation_metrics": None,
        "materialization": _generic_materialization_payload(
            materialization_mode,
            status=status,
            reason=reason,
            task_index=task_index,
        ),
        "artifacts": {},
        "external_command_executed": False,
        "execution_implemented": False,
        "metadata": {"developer_only": True, "one_task_only": True, "normal_routing_unchanged": True},
    }


def _materialization_payload(mode: str, result: TaskInputMaterializationResult) -> JsonDict:
    payload = result.to_json_dict()
    payload["mode"] = mode
    return payload


def _selection_materialization_payload(
    *,
    mode: str,
    task: object | None,
    task_index: int | None,
    initial_tensors: Mapping[str, object],
    status: str,
    reason: str | None,
) -> JsonDict:
    if task is None:
        return _generic_materialization_payload(mode, status=status, reason=reason, task_index=task_index)
    return {
        "mode": mode,
        "schema_version": TASK_INPUT_MATERIALIZATION_SCHEMA_VERSION,
        "status": status,
        "reason": reason,
        "target_task_id": task.id,
        "target_task_index": task_index,
        "selected_input_tensor_ids": task.input_tensor_ids,
        "input_sources": {
            tensor_id: _input_source_record(tensor_id, initial_tensors.get(tensor_id))
            for tensor_id in task.input_tensor_ids
        },
        "replayed_task_count": 0,
        "replayed_task_ids": [],
        "replay_time_s": 0.0,
        "peak_materialized_bytes": _retained_tensor_bytes(initial_tensors),
        "dead_tensor_release_implemented": False,
        "step_metrics": [],
        "error": None,
    }


def _generic_materialization_payload(
    mode: str,
    *,
    status: str,
    reason: str | None,
    task_index: int | None,
) -> JsonDict:
    return {
        "mode": mode,
        "schema_version": TASK_INPUT_MATERIALIZATION_SCHEMA_VERSION,
        "status": status,
        "reason": reason,
        "target_task_id": None,
        "target_task_index": task_index,
        "selected_input_tensor_ids": None,
        "input_sources": {},
        "replayed_task_count": 0,
        "replayed_task_ids": [],
        "replay_time_s": 0.0,
        "peak_materialized_bytes": 0,
        "dead_tensor_release_implemented": False,
        "step_metrics": [],
        "error": None,
    }


def _input_source_record(tensor_id: str, tensor: object | None) -> JsonDict:
    if tensor is None:
        return {"source": "unavailable", "tensor_id": tensor_id}
    array = getattr(tensor, "array", None)
    spec = getattr(tensor, "spec", None)
    return {
        "source": "initial",
        "tensor_id": tensor_id,
        "labels": getattr(spec, "labels", ()),
        "shape": getattr(spec, "shape", ()),
        "dtype": getattr(spec, "dtype", None),
        "produced_by": getattr(spec, "produced_by", None),
        "replayed_task_index": None,
        "nbytes": int(getattr(array, "nbytes", 0) or 0),
    }


def _retained_tensor_bytes(tensors: Mapping[str, object]) -> int:
    total = 0
    for tensor in tensors.values():
        total += int(getattr(getattr(tensor, "array", None), "nbytes", 0) or 0)
    return total


def _result_from_summary(run_dir: Path, summary_path: Path, summary: JsonDict) -> DenseTaskBridgeResult:
    return DenseTaskBridgeResult(
        schema_version=DENSE_TASK_BRIDGE_SCHEMA_VERSION,
        status=summary["status"],
        reason=summary["reason"],
        run_dir=run_dir,
        summary_path=summary_path,
        case_id=summary["case_id"],
        task_index=summary["task_index"],
        task_id=summary["task_id"],
        backend=summary["bridge_backend_id"],
        external_command_executed=bool(summary["external_command_executed"]),
        execution_implemented=bool(summary["execution_implemented"]),
        summary=summary,
    )


def _create_dense_task_bridge_run_dir(root_dir: Path) -> Path:
    runs_dir = root_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = f"{stamp}_{sanitize('dense_task_bridge')}"
    run_dir = runs_dir / base
    suffix = 1
    while run_dir.exists():
        run_dir = runs_dir / f"{base}_{suffix:02d}"
        suffix += 1
    run_dir.mkdir(parents=True)
    update_latest_symlink(runs_dir, run_dir)
    return run_dir
