from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path
from typing import Any, Literal, Mapping

import numpy as np
import yaml

from quantum_bench.bench.config import DEFAULTS, load_suite
from quantum_bench.bench.run_dirs import create_run_dir
from quantum_bench.circuits import builtin_circuit, load_circuit, manifest
from quantum_bench.core.jsonio import write_json, write_jsonl
from quantum_bench.core.records import (
    ContractionTask,
    JsonDict,
    TensorSpec,
    TensorValue,
    to_jsonable,
)
from quantum_bench.environment import capture_environment
from quantum_bench.routing import (
    SHADOW_ROUTE_POLICY_IDS,
    DenseTaskPreparationInput,
    ShadowRoutePolicyConfig,
    ShadowRoutePolicyId,
    TaskRouteContext,
    evaluate_shadow_route_policy,
    prepare_dense_task,
    route_task_graph,
    summarize_shadow_route_policy,
)
from quantum_bench.targets.upmem import (
    SIMPLEPIM_PROBE_KEY,
    UPMEM_DENSE_ESTIMATE_KEY,
    annotate_task_graph_with_upmem_estimates,
    execute_dense_bridge,
    probe_simplepim,
    write_dense_bridge_input_manifest,
)
from quantum_bench.tn import TensorNetworkValue, build_tensor_network, plan_task_graph_with_config, with_path_cost_summary
from quantum_bench.validation import compute_reference, validate


SHADOW_ROUTED_RUNTIME_SCHEMA_VERSION = "shadow_routed_runtime_v1"
ShadowDenseMode = Literal["none", "prepare", "bridge", "stub"]
ShadowBridgeBackend = Literal["none", "mock_numpy_dequantized", "simplepim_external_stub"]

_BRIDGEABLE_PREPARATION_STATUSES = {"prepared", "simplepim_unavailable"}

RUNTIME_FIELDS = [
    "case_id",
    "workload_id",
    "circuit_family",
    "n_qubits",
    "planner_engine",
    "planner_id",
    "optimize_mode",
    "task_index",
    "task_id",
    "input_tensor_ids",
    "output_tensor_id",
    "authoritative_route",
    "selected_authoritative_route",
    "candidate_routes",
    "router_selected_route",
    "router_status",
    "router_reason",
    "cpu_execution_status",
    "cpu_execution_reason",
    "cpu_execution_time_s",
    "output_shape",
    "output_bytes",
    "live_tensor_bytes_after_task",
    "dense_shadow_enabled",
    "dense_shadow_mode",
    "dense_shadow_status",
    "dense_shadow_reason",
    "dense_prepare_status",
    "dense_prepare_reason",
    "bridge_manifest_eligible",
    "bridge_manifest_reason",
    "bridge_artifact_written",
    "bridge_artifact_path",
    "bridge_backend",
    "bridge_status",
    "bridge_reason",
    "bridge_validation_metrics",
    "external_command_executed",
    "execution_implemented",
    "native_kernel_executed",
    "shadow_policy_id",
    "shadow_policy_selected_route",
    "shadow_policy_status",
    "shadow_policy_reason",
    "shadow_policy_blockers",
    "final_route_used_for_tensor",
]


def run_shadow_routed_runtime(
    root_dir: Path,
    *,
    suite_path: Path | None = None,
    case: str | None = None,
    n_qubits: int | None = None,
    dense_shadow: ShadowDenseMode = "prepare",
    bridge_backend: ShadowBridgeBackend = "none",
    execute_external: bool = False,
    max_bridge_artifacts: int = 0,
    shadow_route_policy: ShadowRoutePolicyId = "cpu-only",
    env: Mapping[str, str] | None = None,
) -> Path:
    _validate_options(
        suite_path=suite_path,
        case=case,
        dense_shadow=dense_shadow,
        bridge_backend=bridge_backend,
        execute_external=execute_external,
        max_bridge_artifacts=max_bridge_artifacts,
        shadow_route_policy=shadow_route_policy,
        env=env,
    )

    suite: dict[str, Any] | None
    cases: list[dict[str, Any]]
    suite_id: str
    planner_config: dict[str, Any]
    if suite_path is not None:
        suite = load_suite(suite_path)
        cases = [dict(item) for item in suite["cases"]]
        suite_id = str(suite["suite_id"])
        planner_config = dict(suite["planner"])
    else:
        suite = None
        circuit = _single_builtin_circuit(str(case), n_qubits)
        suite_id = circuit.name
        planner_config = dict(DEFAULTS["planner"])
        circuit_payload: JsonDict = {"kind": "builtin", "name": str(case)}
        if n_qubits is not None:
            circuit_payload["n_qubits"] = int(n_qubits)
        cases = [
            {
                "case_id": circuit.name,
                "workload_id": circuit.name,
                "circuit": circuit_payload,
                "_preloaded_circuit": circuit,
            }
        ]

    run_dir = create_run_dir(root_dir, f"{suite_id}_shadow_routed_runtime")
    write_json(run_dir / "environment.json", capture_environment(root_dir))
    if suite is not None:
        (run_dir / "config" / "resolved_suite.yml").write_text(
            yaml.safe_dump(suite, sort_keys=True),
            encoding="utf-8",
        )
    else:
        write_json(
            run_dir / "config" / "shadow_routed_runtime_input.json",
            {
                "case": case,
                "n_qubits": n_qubits,
                "planner": planner_config,
                "dense_shadow": dense_shadow,
                "bridge_backend": bridge_backend,
                "execute_external": execute_external,
                "max_bridge_artifacts": max_bridge_artifacts,
                "shadow_route_policy": shadow_route_policy,
            },
        )

    probe = probe_simplepim(env=env) if env is not None else probe_simplepim()
    rows: list[JsonDict] = []
    case_summaries: list[JsonDict] = []
    bridge_artifacts_written = 0
    for case_payload in cases:
        case_rows, case_summary, bridge_artifacts_written = _run_shadow_case(
            root_dir=root_dir,
            run_dir=run_dir,
            suite_id=suite_id,
            case_payload=dict(case_payload),
            planner_config=planner_config,
            dense_shadow=dense_shadow,
            bridge_backend=bridge_backend,
            execute_external=execute_external,
            max_bridge_artifacts=max_bridge_artifacts,
            shadow_route_policy=shadow_route_policy,
            bridge_artifacts_written=bridge_artifacts_written,
            probe=probe,
            env=env,
        )
        rows.extend(case_rows)
        case_summaries.append(case_summary)

    summary = _run_summary(rows, case_summaries)
    payload: JsonDict = {
        "schema_version": SHADOW_ROUTED_RUNTIME_SCHEMA_VERSION,
        "run_id": run_dir.name,
        "suite_id": suite_id,
        "source": "suite" if suite_path is not None else "case",
        "suite_path": str(suite_path) if suite_path is not None else None,
        "planner": planner_config,
        "dense_shadow": dense_shadow,
        "bridge_backend": bridge_backend,
        "execute_external": execute_external,
        "max_bridge_artifacts": max_bridge_artifacts,
        "shadow_route_policy": shadow_route_policy,
        "simplepim_probe": probe.to_json_dict(),
        "status": summary["status"],
        "summary": summary,
        "case_summaries": case_summaries,
        "rows": rows,
    }
    write_json(run_dir / "shadow_routed_runtime.json", payload)
    _write_runtime_csv(run_dir / "shadow_routed_runtime.csv", rows)
    (run_dir / "shadow_routed_runtime_summary.md").write_text(
        _runtime_markdown(summary, case_summaries),
        encoding="utf-8",
    )
    return run_dir


def validate_cli_options(
    *,
    suite_path: Path | None,
    case: str | None,
    dense_shadow: str,
    bridge_backend: str,
    execute_external: bool,
    max_bridge_artifacts: int,
    shadow_route_policy: str,
    env: Mapping[str, str] | None = None,
) -> None:
    _validate_options(
        suite_path=suite_path,
        case=case,
        dense_shadow=dense_shadow,
        bridge_backend=bridge_backend,
        execute_external=execute_external,
        max_bridge_artifacts=max_bridge_artifacts,
        shadow_route_policy=shadow_route_policy,
        env=env,
    )


def _run_shadow_case(
    *,
    root_dir: Path,
    run_dir: Path,
    suite_id: str,
    case_payload: dict[str, Any],
    planner_config: dict[str, Any],
    dense_shadow: ShadowDenseMode,
    bridge_backend: ShadowBridgeBackend,
    execute_external: bool,
    max_bridge_artifacts: int,
    shadow_route_policy: ShadowRoutePolicyId,
    bridge_artifacts_written: int,
    probe: Any,
    env: Mapping[str, str] | None,
) -> tuple[list[JsonDict], JsonDict, int]:
    circuit = case_payload.pop("_preloaded_circuit", None)
    if circuit is None:
        circuit = load_circuit(case_payload, root_dir)
    network = build_tensor_network(circuit)
    graph = plan_task_graph_with_config(network, planner_config)
    graph, _ = annotate_task_graph_with_upmem_estimates(graph)
    graph = with_path_cost_summary(graph)

    case_id = str(case_payload["case_id"])
    workload_id = str(case_payload.get("workload_id", case_id))
    circuit_family = str(case_payload.get("circuit", {}).get("name", circuit.name))
    route_context = TaskRouteContext(
        suite_id=suite_id,
        case_id=case_id,
        run_dir=run_dir,
        policy="shadow_runtime_cpu_fallback_authoritative",
        backend_probes={SIMPLEPIM_PROBE_KEY: probe.to_json_dict()},
    )
    routing = route_task_graph(graph, route_context)
    decisions_by_task = _decisions_by_task(routing.decisions)
    policy_config = ShadowRoutePolicyConfig(policy_id=shadow_route_policy)
    simplepim_probe = probe.to_json_dict()

    rows: list[JsonDict] = []
    tensors = {tensor.spec.id: np.asarray(tensor.array, dtype=np.complex128) for tensor in network.tensors}
    labels = {tensor.spec.id: tensor.spec.labels for tensor in network.tensors}
    live_ids = set(tensors)
    remaining_uses = _remaining_input_uses(graph)
    peak_live_tensor_bytes = _live_tensor_bytes(tensors, live_ids)
    max_output_tensor_bytes = 0
    total_cpu_time_s = 0.0
    warnings: list[JsonDict] = []
    case_status = "passed"
    case_reason: str | None = None
    final_tensor_id = graph.tasks[-1].output_tensor_id if graph.tasks else None

    if not graph.tasks:
        output, empty_final_id, empty_final_labels, transposed = _execute_empty_graph(graph, network)
        final_tensor_id = empty_final_id
        final_validation, reference_time_s = _validate_final_output(output, network, planner_config)
        case_status = "passed" if final_validation["passed"] else "failed"
        case_reason = None if final_validation["passed"] else "final_validation_failed"
        case_summary = _case_summary(
            case_id=case_id,
            workload_id=workload_id,
            circuit_manifest=manifest(circuit),
            graph=graph,
            status=case_status,
            reason=case_reason,
            final_tensor_id=final_tensor_id,
            final_tensor_labels=empty_final_labels,
            final_transpose_applied=transposed,
            final_validation=final_validation,
            reference_time_s=reference_time_s,
            total_cpu_time_s=0.0,
            peak_live_tensor_bytes=int(output.nbytes),
            max_output_tensor_bytes=int(output.nbytes),
            warning_count=0,
            warnings=[],
        )
        write_jsonl(run_dir / "cases" / case_id / "shadow_routed_runtime.jsonl", rows)
        return rows, case_summary, bridge_artifacts_written

    for task_index, task in enumerate(graph.tasks):
        row_started = time.perf_counter()
        task_decisions = decisions_by_task.get(task.id, [])
        router_selected = _selected_router_decision(task_decisions)

        dense_record, bridge_artifacts_written = _shadow_dense_evidence(
            run_dir=run_dir,
            case_id=case_id,
            task_index=task_index,
            task=task,
            tensors=tensors,
            labels=labels,
            dense_shadow=dense_shadow,
            bridge_backend=bridge_backend,
            execute_external=execute_external,
            max_bridge_artifacts=max_bridge_artifacts,
            bridge_artifacts_written=bridge_artifacts_written,
            probe=probe,
            env=env,
        )
        policy_decision = evaluate_shadow_route_policy(
            config=policy_config,
            task_decisions=task_decisions,
            dense_record=dense_record,
            simplepim_probe=simplepim_probe,
        )
        if dense_record.get("shadow_warning"):
            warnings.append(
                {
                    "task_id": task.id,
                    "task_index": task_index,
                    "reason": dense_record["shadow_warning"],
                }
            )

        cpu_payload = _execute_cpu_fallback_task(task, tensors)
        cpu_status = cpu_payload["status"]
        cpu_reason = cpu_payload["reason"]
        output_shape: tuple[int, ...] | None = None
        output_bytes = 0
        if cpu_status == "passed":
            output = np.asarray(cpu_payload["output"], dtype=np.complex128)
            tensors[task.output_tensor_id] = output
            labels[task.output_tensor_id] = task.output_labels
            live_ids.add(task.output_tensor_id)
            output_shape = tuple(int(dim) for dim in output.shape)
            output_bytes = int(output.nbytes)
            max_output_tensor_bytes = max(max_output_tensor_bytes, output_bytes)
            peak_live_tensor_bytes = max(peak_live_tensor_bytes, _live_tensor_bytes(tensors, live_ids))
            total_cpu_time_s += float(cpu_payload["execution_time_s"])
            _release_dead_inputs(task, tensors, labels, live_ids, remaining_uses)
        else:
            case_status = "failed"
            case_reason = cpu_reason or "cpu_fallback_failed"

        live_tensor_bytes = _live_tensor_bytes(tensors, live_ids)
        row = {
            "case_id": case_id,
            "workload_id": workload_id,
            "circuit_family": circuit_family,
            "n_qubits": int(circuit.n_qubits),
            "planner_engine": graph.path_summary.planner_engine,
            "planner_id": graph.path_summary.planner_id,
            "optimize_mode": graph.path_summary.optimize_mode,
            "task_index": task_index,
            "task_id": task.id,
            "input_tensor_ids": task.input_tensor_ids,
            "output_tensor_id": task.output_tensor_id,
            "authoritative_route": "cpu_fallback",
            "selected_authoritative_route": "cpu_fallback",
            "candidate_routes": [_candidate_route_payload(decision) for decision in task_decisions],
            "router_selected_route": router_selected.get("route_id"),
            "router_status": router_selected.get("status"),
            "router_reason": router_selected.get("reason"),
            "cpu_execution_status": cpu_status,
            "cpu_execution_reason": cpu_reason,
            "cpu_execution_time_s": float(cpu_payload.get("execution_time_s", time.perf_counter() - row_started)),
            "output_shape": output_shape,
            "output_bytes": output_bytes,
            "live_tensor_bytes_after_task": live_tensor_bytes,
            **dense_record,
            "shadow_policy_id": policy_decision.policy_id,
            "shadow_policy_selected_route": policy_decision.selected_route,
            "shadow_policy_status": policy_decision.status,
            "shadow_policy_reason": policy_decision.reason,
            "shadow_policy_blockers": policy_decision.blockers,
            "final_route_used_for_tensor": "cpu_fallback" if cpu_status == "passed" else None,
        }
        rows.append(to_jsonable(row))
        if cpu_status != "passed":
            break

    final_validation: JsonDict | None = None
    reference_time_s: float | None = None
    final_transpose_applied = False
    final_tensor_labels: tuple[int, ...] | None = None
    if case_status == "passed":
        try:
            if final_tensor_id is None or final_tensor_id not in tensors:
                raise ValueError(f"Task sequence did not produce final tensor {final_tensor_id}")
            final_tensor_labels = labels[final_tensor_id]
            output, final_transpose_applied = _order_final_tensor(
                tensors[final_tensor_id],
                final_tensor_labels,
                graph.network.output_labels,
            )
            final_validation, reference_time_s = _validate_final_output(output, network, planner_config)
            if not final_validation["passed"]:
                case_status = "failed"
                case_reason = "final_validation_failed"
        except Exception as exc:
            case_status = "failed"
            case_reason = "final_validation_exception"
            final_validation = {"passed": False, "error": str(exc)}
            reference_time_s = None

    case_summary = _case_summary(
        case_id=case_id,
        workload_id=workload_id,
        circuit_manifest=manifest(circuit),
        graph=graph,
        status=case_status,
        reason=case_reason,
        final_tensor_id=final_tensor_id,
        final_tensor_labels=final_tensor_labels,
        final_transpose_applied=final_transpose_applied,
        final_validation=final_validation,
        reference_time_s=reference_time_s,
        total_cpu_time_s=total_cpu_time_s,
        peak_live_tensor_bytes=peak_live_tensor_bytes,
        max_output_tensor_bytes=max_output_tensor_bytes,
        warning_count=len(warnings),
        warnings=warnings,
    )
    write_jsonl(run_dir / "cases" / case_id / "shadow_routed_runtime.jsonl", rows)
    return rows, case_summary, bridge_artifacts_written


def _shadow_dense_evidence(
    *,
    run_dir: Path,
    case_id: str,
    task_index: int,
    task: ContractionTask,
    tensors: Mapping[str, np.ndarray],
    labels: Mapping[str, tuple[int, ...]],
    dense_shadow: ShadowDenseMode,
    bridge_backend: ShadowBridgeBackend,
    execute_external: bool,
    max_bridge_artifacts: int,
    bridge_artifacts_written: int,
    probe: Any,
    env: Mapping[str, str] | None,
) -> tuple[JsonDict, int]:
    base = _base_dense_record(dense_shadow, bridge_backend)
    if dense_shadow == "none":
        base["dense_shadow_status"] = "not_requested"
        base["dense_shadow_reason"] = "dense_shadow_disabled"
        return base, bridge_artifacts_written
    if not _inputs_available(task, tensors):
        base["dense_shadow_status"] = "skipped"
        base["dense_shadow_reason"] = "task_inputs_unavailable_before_cpu_fallback"
        return base, bridge_artifacts_written

    try:
        left_tensor = _tensor_value_for(task.input_tensor_ids[0], tensors, labels)
        right_tensor = _tensor_value_for(task.input_tensor_ids[1], tensors, labels)
        preparation = prepare_dense_task(
            DenseTaskPreparationInput(
                task=task,
                left_tensor=left_tensor,
                right_tensor=right_tensor,
                simplepim_probe=probe,
            )
        )
    except Exception as exc:  # pragma: no cover - defensive shadow boundary
        base["dense_shadow_status"] = "warning"
        base["dense_shadow_reason"] = "dense_preparation_exception"
        base["dense_prepare_status"] = "failed"
        base["dense_prepare_reason"] = str(exc)
        base["shadow_warning"] = f"dense_preparation_exception:{exc}"
        return base, bridge_artifacts_written

    base["dense_prepare_status"] = preparation.status
    base["dense_prepare_reason"] = preparation.reason
    base["dense_shadow_status"] = "prepared" if preparation.status in _BRIDGEABLE_PREPARATION_STATUSES else "warning"
    base["dense_shadow_reason"] = preparation.reason

    eligible, bridge_reason = _bridge_manifest_eligibility(preparation)
    base["bridge_manifest_eligible"] = eligible
    base["bridge_manifest_reason"] = bridge_reason
    if not eligible:
        if dense_shadow in {"bridge", "stub"}:
            base["dense_shadow_status"] = "warning"
            base["shadow_warning"] = bridge_reason or "bridge_manifest_not_eligible"
        return base, bridge_artifacts_written

    if dense_shadow == "prepare":
        return base, bridge_artifacts_written

    if bridge_artifacts_written >= max_bridge_artifacts:
        base["dense_shadow_status"] = "prepared"
        base["dense_shadow_reason"] = "bridge_artifact_cap_reached"
        return base, bridge_artifacts_written

    bridge_dir = run_dir / "cases" / case_id / "dense_bridge" / f"task_{task_index:04d}"
    write_dense_bridge_input_manifest(preparation, bridge_dir)
    input_manifest_path = bridge_dir / "input_manifest.json"
    base["bridge_artifact_written"] = True
    base["bridge_artifact_path"] = _relative_to_run(input_manifest_path, run_dir)
    bridge_artifacts_written += 1

    if dense_shadow == "bridge" and bridge_backend == "none":
        base["dense_shadow_status"] = "prepared"
        base["dense_shadow_reason"] = "bridge_input_manifest_written"
        return base, bridge_artifacts_written

    if dense_shadow == "bridge" and bridge_backend == "mock_numpy_dequantized":
        bridge_result = execute_dense_bridge(
            input_manifest_path,
            backend="mock_numpy_dequantized",
            execute_external=False,
            env=env,
        )
        _apply_bridge_result(base, bridge_result, bridge_dir, run_dir, expected_success="mock_executed")
        return base, bridge_artifacts_written

    if dense_shadow == "stub" and bridge_backend == "simplepim_external_stub":
        bridge_result = execute_dense_bridge(
            input_manifest_path,
            backend="simplepim_external_stub",
            execute_external=execute_external,
            env=env,
        )
        _apply_bridge_result(base, bridge_result, bridge_dir, run_dir, expected_success="stub_executed")
        if bridge_result.execution_status == "stub_executed":
            base["bridge_validation_metrics"] = {"status": "not_applicable", "reason": "external_stub_no_output_blob"}
        return base, bridge_artifacts_written

    base["dense_shadow_status"] = "warning"
    base["dense_shadow_reason"] = "unsupported_shadow_bridge_combination"
    base["shadow_warning"] = "unsupported_shadow_bridge_combination"
    return base, bridge_artifacts_written


def _base_dense_record(dense_shadow: ShadowDenseMode, bridge_backend: ShadowBridgeBackend) -> JsonDict:
    return {
        "dense_shadow_enabled": dense_shadow != "none",
        "dense_shadow_mode": dense_shadow,
        "dense_shadow_status": None,
        "dense_shadow_reason": None,
        "dense_prepare_status": None,
        "dense_prepare_reason": None,
        "bridge_manifest_eligible": False,
        "bridge_manifest_reason": None,
        "bridge_artifact_written": False,
        "bridge_artifact_path": None,
        "bridge_backend": bridge_backend,
        "bridge_status": None,
        "bridge_reason": None,
        "bridge_validation_metrics": None,
        "external_command_executed": False,
        "execution_implemented": False,
        "native_kernel_executed": False,
    }


def _apply_bridge_result(
    row: JsonDict,
    bridge_result: Any,
    bridge_dir: Path,
    run_dir: Path,
    *,
    expected_success: str,
) -> None:
    row["bridge_status"] = bridge_result.execution_status
    row["bridge_reason"] = bridge_result.reason
    row["external_command_executed"] = bool(bridge_result.external_command_executed)
    row["execution_implemented"] = bool(bridge_result.execution_implemented)
    row["native_kernel_executed"] = bool(bridge_result.metadata.get("native_kernel_executed", False))
    if bridge_result.output_manifest_path is not None:
        row["bridge_output_manifest_path"] = _relative_to_run(bridge_dir / bridge_result.output_manifest_path, run_dir)
    if bridge_result.output_blob_path is not None:
        row["bridge_output_blob_path"] = _relative_to_run(bridge_dir / bridge_result.output_blob_path, run_dir)
    output_manifest = bridge_result.output_manifest
    if output_manifest is not None and output_manifest.validation_metrics:
        row["bridge_validation_metrics"] = output_manifest.validation_metrics
    elif bridge_result.execution_status != "mock_executed":
        row["bridge_validation_metrics"] = {"status": "not_applicable", "reason": "no_numeric_bridge_output"}
    if bridge_result.execution_status == expected_success:
        row["dense_shadow_status"] = "checked"
        row["dense_shadow_reason"] = bridge_result.reason
        return
    row["dense_shadow_status"] = "warning"
    row["dense_shadow_reason"] = bridge_result.reason or bridge_result.execution_status
    row["shadow_warning"] = f"bridge_{bridge_result.execution_status}:{row['dense_shadow_reason']}"


def _execute_cpu_fallback_task(task: ContractionTask, tensors: Mapping[str, np.ndarray]) -> JsonDict:
    left_id, right_id = task.input_tensor_ids
    if left_id not in tensors or right_id not in tensors:
        missing = [tensor_id for tensor_id in task.input_tensor_ids if tensor_id not in tensors]
        return {
            "status": "failed",
            "reason": "missing_cpu_fallback_input_tensor",
            "error": f"Task {task.id} references unavailable tensor(s): {', '.join(missing)}",
            "execution_time_s": 0.0,
            "output": None,
        }
    try:
        start = time.perf_counter()
        output = np.einsum(task.index_expression, tensors[left_id], tensors[right_id], optimize=False)
        execution_time_s = time.perf_counter() - start
        output = np.asarray(output, dtype=np.complex128)
        if tuple(int(dim) for dim in output.shape) != task.output_shape:
            return {
                "status": "failed",
                "reason": "cpu_fallback_output_shape_mismatch",
                "error": f"Task {task.id} produced {output.shape}, expected {task.output_shape}",
                "execution_time_s": execution_time_s,
                "output": None,
            }
        return {
            "status": "passed",
            "reason": None,
            "error": None,
            "execution_time_s": execution_time_s,
            "output": output,
        }
    except Exception as exc:  # pragma: no cover - defensive runtime boundary
        return {
            "status": "failed",
            "reason": "cpu_fallback_exception",
            "error": str(exc),
            "execution_time_s": 0.0,
            "output": None,
        }


def _release_dead_inputs(
    task: ContractionTask,
    tensors: dict[str, np.ndarray],
    labels: dict[str, tuple[int, ...]],
    live_ids: set[str],
    remaining_uses: dict[str, int],
) -> None:
    for input_id in task.input_tensor_ids:
        remaining_uses[input_id] = remaining_uses.get(input_id, 0) - 1
        if remaining_uses[input_id] <= 0 and input_id != task.output_tensor_id:
            live_ids.discard(input_id)
            tensors.pop(input_id, None)
            labels.pop(input_id, None)


def _validate_final_output(
    output: np.ndarray,
    network: TensorNetworkValue,
    planner_config: Mapping[str, Any],
) -> tuple[JsonDict, float]:
    optimize = str(planner_config.get("optimize", "greedy"))
    reference, reference_time_s = compute_reference(network, optimize=optimize)
    return to_jsonable(validate(output, reference)), float(reference_time_s)


def _execute_empty_graph(graph: Any, network: TensorNetworkValue) -> tuple[np.ndarray, str, tuple[int, ...], bool]:
    if len(network.tensors) != 1:
        raise ValueError(
            f"Cannot execute empty TaskGraph with {len(network.tensors)} original tensors; expected exactly one tensor"
        )
    tensor = network.tensors[0]
    output, transposed = _order_final_tensor(
        np.asarray(tensor.array, dtype=np.complex128),
        tensor.spec.labels,
        graph.network.output_labels,
    )
    return output, tensor.spec.id, tensor.spec.labels, transposed


def _order_final_tensor(
    array: np.ndarray,
    actual_labels: tuple[int, ...],
    output_labels: tuple[int, ...],
) -> tuple[np.ndarray, bool]:
    if actual_labels == output_labels:
        return np.asarray(array, dtype=np.complex128), False
    if len(actual_labels) != len(output_labels) or set(actual_labels) != set(output_labels):
        raise ValueError(f"Final tensor labels {actual_labels} do not match requested output labels {output_labels}")
    axes = tuple(actual_labels.index(label) for label in output_labels)
    return np.asarray(np.transpose(array, axes), dtype=np.complex128), True


def _tensor_value_for(
    tensor_id: str,
    tensors: Mapping[str, np.ndarray],
    labels: Mapping[str, tuple[int, ...]],
) -> TensorValue:
    array = np.asarray(tensors[tensor_id], dtype=np.complex128)
    return TensorValue(
        TensorSpec(
            id=tensor_id,
            labels=labels[tensor_id],
            shape=tuple(int(dim) for dim in array.shape),
            structure="dense",
            dtype=str(array.dtype),
        ),
        array,
    )


def _decisions_by_task(decisions: tuple[Any, ...]) -> dict[str, list[Any]]:
    grouped: dict[str, list[Any]] = {}
    for decision in decisions:
        grouped.setdefault(decision.task_id, []).append(decision)
    return grouped


def _selected_router_decision(decisions: list[Any]) -> JsonDict:
    selected = next((decision for decision in decisions if decision.is_selected), None)
    if selected is None:
        selected = next((decision for decision in decisions if decision.route_id == "cpu_fallback"), None)
    if selected is None:
        return {"route_id": None, "status": None, "reason": "router_selected_route_missing"}
    return {"route_id": selected.route_id, "status": selected.status, "reason": selected.reason}


def _candidate_route_payload(decision: Any) -> JsonDict:
    return {
        "route_id": decision.route_id,
        "route_family": decision.route_family,
        "kernel_family": decision.kernel_family,
        "hardware_target": decision.hardware_target,
        "execution_mode": decision.execution_mode,
        "maturity_level": decision.maturity_level,
        "status": decision.status,
        "is_selected": decision.is_selected,
        "reason": decision.reason,
        "execution_status": to_jsonable(decision.execution_status),
        "estimate": to_jsonable(decision.estimate),
        "metadata": to_jsonable(decision.metadata),
    }


def _bridge_manifest_eligibility(preparation: Any) -> tuple[bool, str | None]:
    if preparation is None:
        return False, "dense_preparation_missing"
    if preparation.prepared_operands is None:
        return False, "prepared_operands_missing"
    if preparation.tile_plan is None:
        return False, "tile_plan_missing"
    if preparation.tile_plan.get("requires_tiling") is True:
        return False, "requires_tiling_not_implemented"
    if preparation.tile_plan.get("requires_host_aggregation") is True:
        return False, "requires_host_aggregation_not_representable"
    if preparation.status not in _BRIDGEABLE_PREPARATION_STATUSES:
        return False, f"non_bridgeable_preparation_status:{preparation.status}"
    if preparation.left_conversion is None or preparation.right_conversion is None:
        return False, "conversion_records_missing"
    if preparation.left_matrix_shape is None or preparation.right_matrix_shape is None:
        return False, "matrix_shapes_missing"
    return True, None


def _inputs_available(task: ContractionTask, tensors: Mapping[str, Any]) -> bool:
    return all(tensor_id in tensors for tensor_id in task.input_tensor_ids)


def _remaining_input_uses(graph: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for task in graph.tasks:
        for tensor_id in task.input_tensor_ids:
            counts[tensor_id] = counts.get(tensor_id, 0) + 1
    return counts


def _live_tensor_bytes(tensors: Mapping[str, np.ndarray], live_ids: set[str]) -> int:
    return int(sum(tensors[tensor_id].nbytes for tensor_id in live_ids if tensor_id in tensors))


def _case_summary(
    *,
    case_id: str,
    workload_id: str,
    circuit_manifest: JsonDict,
    graph: Any,
    status: str,
    reason: str | None,
    final_tensor_id: str | None,
    final_tensor_labels: tuple[int, ...] | None,
    final_transpose_applied: bool,
    final_validation: JsonDict | None,
    reference_time_s: float | None,
    total_cpu_time_s: float,
    peak_live_tensor_bytes: int,
    max_output_tensor_bytes: int,
    warning_count: int,
    warnings: list[JsonDict],
) -> JsonDict:
    return to_jsonable(
        {
            "case_id": case_id,
            "workload_id": workload_id,
            "circuit": circuit_manifest,
            "task_count": len(graph.tasks),
            "status": status,
            "reason": reason,
            "final_tensor_id": final_tensor_id,
            "final_tensor_labels": final_tensor_labels,
            "output_labels": graph.network.output_labels,
            "final_transpose_applied": final_transpose_applied,
            "final_validation": final_validation,
            "reference_time_s": reference_time_s,
            "total_cpu_time_s": total_cpu_time_s,
            "peak_live_tensor_bytes": peak_live_tensor_bytes,
            "max_output_tensor_bytes": max_output_tensor_bytes,
            "dead_tensor_release_policy": "remaining_use_counts_by_occurrence",
            "selected_authoritative_route": "cpu_fallback",
            "dense_shadow_warning_count": warning_count,
            "dense_shadow_warnings": warnings,
        }
    )


def _run_summary(rows: list[JsonDict], case_summaries: list[JsonDict]) -> JsonDict:
    failed_cases = [case for case in case_summaries if case["status"] != "passed"]
    warning_count = sum(int(case.get("dense_shadow_warning_count", 0) or 0) for case in case_summaries)
    policy_summary = summarize_shadow_route_policy(rows).to_json_dict()
    return {
        "status": "failed" if failed_cases else "completed",
        "case_count": len(case_summaries),
        "task_count": len(rows),
        "passed_case_count": len(case_summaries) - len(failed_cases),
        "failed_case_count": len(failed_cases),
        "warning_count": warning_count,
        "cpu_fallback_authoritative_task_count": sum(
            1 for row in rows if row.get("selected_authoritative_route") == "cpu_fallback"
        ),
        "dense_shadow_enabled_task_count": sum(1 for row in rows if row.get("dense_shadow_enabled")),
        "bridge_manifest_eligible_task_count": sum(1 for row in rows if row.get("bridge_manifest_eligible")),
        "bridge_artifact_written_count": sum(1 for row in rows if row.get("bridge_artifact_written")),
        "external_command_executed_count": sum(1 for row in rows if row.get("external_command_executed")),
        "native_kernel_executed_count": sum(1 for row in rows if row.get("native_kernel_executed")),
        "shadow_policy_summary": policy_summary,
    }


def _runtime_markdown(summary: JsonDict, case_summaries: list[JsonDict]) -> str:
    lines = [
        "# Shadow Routed Runtime",
        "",
        "Developer-only full TaskGraph runtime. CPU fallback is authoritative; dense route checks are shadow evidence only.",
        "",
        f"- Status: {summary['status']}",
        f"- Cases: {summary['case_count']}",
        f"- Tasks: {summary['task_count']}",
        f"- CPU fallback authoritative tasks: {summary['cpu_fallback_authoritative_task_count']}",
        f"- Dense shadow warnings: {summary['warning_count']}",
        f"- Bridge artifacts written: {summary['bridge_artifact_written_count']}",
        f"- External commands executed: {summary['external_command_executed_count']}",
        f"- Native kernels executed: {summary['native_kernel_executed_count']}",
        f"- Shadow policy: {summary['shadow_policy_summary']['shadow_policy_id']}",
        f"- Shadow dense selections: {summary['shadow_policy_summary']['selected_route_counts'].get('dense_gemm', 0)}",
        "",
        "## Cases",
        "",
        "| Case | Status | Tasks | Final Validation | Warnings |",
        "| --- | --- | ---: | --- | ---: |",
    ]
    for case in case_summaries:
        validation = case.get("final_validation") or {}
        passed = validation.get("passed")
        lines.append(
            f"| {case['case_id']} | {case['status']} | {case['task_count']} | {passed} | {case['dense_shadow_warning_count']} |"
        )
    if not case_summaries:
        lines.append("| none | n/a | 0 | n/a | 0 |")
    lines.append("")
    return "\n".join(lines)


def _write_runtime_csv(path: Path, rows: list[JsonDict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RUNTIME_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in RUNTIME_FIELDS})


def _csv_value(value: Any) -> Any:
    value = to_jsonable(value)
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    if value is None:
        return ""
    return value


def _single_builtin_circuit(case: str, n_qubits: int | None):
    params: JsonDict = {"name": case}
    if n_qubits is not None:
        params["n_qubits"] = int(n_qubits)
    return builtin_circuit(case, params)


def _relative_to_run(path: Path, run_dir: Path) -> str:
    try:
        return path.relative_to(run_dir).as_posix()
    except ValueError:
        return path.name


def _validate_options(
    *,
    suite_path: Path | None,
    case: str | None,
    dense_shadow: str,
    bridge_backend: str,
    execute_external: bool,
    max_bridge_artifacts: int,
    shadow_route_policy: str,
    env: Mapping[str, str] | None,
) -> None:
    if (suite_path is None) == (case is None):
        raise ValueError("shadow-routed-runtime requires exactly one of --suite or --case")
    if dense_shadow not in {"none", "prepare", "bridge", "stub"}:
        raise ValueError("--dense-shadow must be one of: none, prepare, bridge, stub")
    if shadow_route_policy not in SHADOW_ROUTE_POLICY_IDS:
        raise ValueError("--shadow-route-policy must be one of: " + ", ".join(SHADOW_ROUTE_POLICY_IDS))
    if shadow_route_policy == "dense-if-bridge-ready" and dense_shadow == "none":
        raise ValueError("--shadow-route-policy dense-if-bridge-ready requires --dense-shadow prepare, bridge, or stub")
    if bridge_backend not in {"none", "mock_numpy_dequantized", "simplepim_external_stub"}:
        raise ValueError("--bridge-backend must be one of: none, mock_numpy_dequantized, simplepim_external_stub")
    if max_bridge_artifacts < 0:
        raise ValueError("--max-bridge-artifacts must be >= 0")
    if dense_shadow in {"none", "prepare"}:
        if bridge_backend != "none":
            raise ValueError(f"--dense-shadow {dense_shadow} requires --bridge-backend none")
        if execute_external:
            raise ValueError(f"--execute-external is invalid with --dense-shadow {dense_shadow}")
        if max_bridge_artifacts != 0:
            raise ValueError(f"--max-bridge-artifacts must be 0 with --dense-shadow {dense_shadow}")
    if dense_shadow == "bridge":
        if execute_external:
            raise ValueError("--execute-external is only valid with --dense-shadow stub")
        if bridge_backend == "simplepim_external_stub":
            raise ValueError("--bridge-backend simplepim_external_stub requires --dense-shadow stub")
        if bridge_backend == "mock_numpy_dequantized" and max_bridge_artifacts == 0:
            raise ValueError("--bridge-backend mock_numpy_dequantized requires --max-bridge-artifacts > 0")
    if dense_shadow == "stub":
        if bridge_backend != "simplepim_external_stub":
            raise ValueError("--dense-shadow stub requires --bridge-backend simplepim_external_stub")
        if not execute_external:
            raise ValueError("--dense-shadow stub requires --execute-external")
        if max_bridge_artifacts == 0:
            raise ValueError("--dense-shadow stub requires --max-bridge-artifacts > 0")
        environment = env if env is not None else os.environ
        if not environment.get("SIMPLEPIM_STUB_BIN"):
            raise ValueError("--dense-shadow stub requires SIMPLEPIM_STUB_BIN to be configured")
