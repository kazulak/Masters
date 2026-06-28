from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any, Literal, Mapping

import yaml

from quantum_bench.bench.config import DEFAULTS, load_suite
from quantum_bench.bench.run_dirs import create_run_dir, sanitize
from quantum_bench.circuits import builtin_circuit, load_circuit, manifest
from quantum_bench.core.jsonio import write_json, write_jsonl
from quantum_bench.core.records import JsonDict, TaskGraph, to_jsonable
from quantum_bench.environment import capture_environment
from quantum_bench.routing import DenseTaskPreparationInput, prepare_dense_task
from quantum_bench.targets.upmem import (
    UPMEM_DENSE_ESTIMATE_KEY,
    annotate_task_graph_with_upmem_estimates,
    build_synthetic_pressure_task_graph,
    dense_bridge_backend_manifest_eligibility,
    dense_bridge_manifest_eligibility,
    execute_dense_bridge,
    is_synthetic_pressure_case,
    synthetic_pressure_initial_tensors,
    synthetic_pressure_manifest,
    plan_l2_tiled_execution,
    probe_simplepim,
    write_dense_bridge_validation_diagnostics,
    write_dense_bridge_input_manifest,
)
from quantum_bench.tn import (
    TaskInputMaterializationRequest,
    build_tensor_network,
    materialize_task_inputs,
    plan_task_graph_with_config,
    with_path_cost_summary,
)


PIM_BRIDGE_EVAL_SCHEMA_VERSION = "pim_bridge_eval_v1"
PimBridgeTaskSelection = Literal["all", "eligible-only", "first-supported", "first-n"]

_PARAMETERIZED_CASES = {"ghz_chain", "qrng", "bv", "bernstein_vazirani", "xor", "parity", "bb84", "bb_n", "edc", "dense_coding", "hs", "hidden_shift"}
_SUCCESSFUL_BACKEND_STATUSES = {"upmem_sdk_simulator_executed"}

TASK_FIELDS = [
    "case_id",
    "workload_id",
    "circuit_family",
    "n_qubits",
    "circuit_parameters",
    "planner_engine",
    "planner_id",
    "optimize_mode",
    "task_index",
    "task_id",
    "input_tensor_ids",
    "output_tensor_id",
    "gemm_m",
    "gemm_k",
    "gemm_n",
    "complex_execution_mode",
    "requires_tiling",
    "requires_host_aggregation",
    "route_dtype",
    "supported_by_dense_estimate",
    "estimated_flops",
    "host_to_dpu_bytes",
    "dpu_to_host_bytes",
    "mram_to_wram_bytes",
    "tile_count",
    "working_set_bytes",
    "materialization_status",
    "materialization_reason",
    "dense_prepare_status",
    "dense_prepare_reason",
    "bridge_manifest_eligible",
    "bridge_artifact_written",
    "bridge_artifact_path",
    "readiness_status",
    "backend_id",
    "backend_status",
    "blocker_reason",
    "validation_status",
    "validation_metrics",
    "max_abs_error",
    "max_rel_error",
    "build_time_s",
    "runner_total_time_s",
    "simulator_run_time_s",
    "kernel_invocation_count",
    "output_manifest_path",
    "output_blob_path",
    "diagnostic_path",
    "diagnostic_conclusion",
    "external_command_executed",
    "execution_implemented",
    "simulator_kernel_executed",
    "hardware_kernel_executed",
]

CASE_FIELDS = [
    "case_id",
    "workload_id",
    "circuit_family",
    "n_qubits",
    "task_count",
    "analyzed_task_count",
    "attempted_task_count",
    "executed_task_count",
    "validated_task_count",
    "failed_task_count",
    "unsupported_task_count",
    "skipped_task_count",
    "blocker_counts",
    "dense_candidate_count",
    "simulator_executable_count",
    "total_estimated_flops",
    "total_estimated_transfer_bytes",
    "total_runner_time_s",
    "total_simulator_time_s",
    "total_build_time_s",
    "max_working_set_bytes",
    "max_tile_count",
    "final_cpu_validation_status",
]


def run_pim_bridge_eval(
    root_dir: Path,
    *,
    suite_path: Path | None = None,
    case: str | None = None,
    n_qubits: int | None = None,
    backend: str = "upmem_sdk_simulator_dense",
    execute_external: bool = False,
    dry_run: bool = False,
    max_tasks_per_case: int = 64,
    max_executed_tasks_per_case: int = 2,
    task_selection: PimBridgeTaskSelection = "eligible-only",
    timeout_seconds: float = 60.0,
    planner: str | None = None,
    output_plots: bool = True,
    debug_failures: bool = False,
    compare_mock_on_failure: bool = False,
    keep_failure_artifacts: bool = False,
    env: Mapping[str, str] | None = None,
) -> Path:
    validate_cli_options(
        suite_path=suite_path,
        case=case,
        n_qubits=n_qubits,
        backend=backend,
        execute_external=execute_external,
        dry_run=dry_run,
        max_tasks_per_case=max_tasks_per_case,
        max_executed_tasks_per_case=max_executed_tasks_per_case,
        task_selection=task_selection,
        timeout_seconds=timeout_seconds,
    )

    suite: dict[str, Any] | None
    cases: list[dict[str, Any]]
    suite_id: str
    planner_config: dict[str, Any]
    source: str
    if suite_path is not None:
        suite = load_suite(suite_path)
        cases = list(suite["cases"])
        suite_id = str(suite["suite_id"])
        planner_config = dict(suite["planner"])
        source = "suite"
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
        source = "case"
    if planner:
        planner_config["optimize"] = str(planner)

    run_dir = create_run_dir(root_dir, _run_id(suite_id))
    write_json(run_dir / "environment.json", capture_environment(root_dir))
    if suite is not None:
        (run_dir / "config" / "resolved_suite.yml").write_text(yaml.safe_dump(suite, sort_keys=True), encoding="utf-8")
    else:
        write_json(
            run_dir / "config" / "pim_bridge_eval_input.json",
            {
                "case": case,
                "n_qubits": n_qubits,
                "backend": backend,
                "execute_external": execute_external,
                "dry_run": dry_run,
                "max_tasks_per_case": max_tasks_per_case,
                "max_executed_tasks_per_case": max_executed_tasks_per_case,
                "task_selection": task_selection,
                "timeout_seconds": timeout_seconds,
                "planner": planner_config,
                "debug_failures": debug_failures,
                "compare_mock_on_failure": compare_mock_on_failure,
                "keep_failure_artifacts": keep_failure_artifacts,
            },
        )

    probe = probe_simplepim(env=env) if env is not None else probe_simplepim()
    rows: list[JsonDict] = []
    case_summaries: list[JsonDict] = []
    execution_env = _execution_environment(env, timeout_seconds)
    for case_payload in cases:
        case_rows, case_summary = _evaluate_case(
            root_dir=root_dir,
            run_dir=run_dir,
            case_payload=case_payload,
            planner_config=planner_config,
            probe=probe,
            backend=backend,
            execute_external=execute_external,
            dry_run=dry_run or not execute_external,
            max_tasks_per_case=max_tasks_per_case,
            max_executed_tasks_per_case=max_executed_tasks_per_case,
            task_selection=task_selection,
            debug_failures=debug_failures or compare_mock_on_failure,
            compare_mock_on_failure=compare_mock_on_failure,
            env=execution_env,
        )
        rows.extend(case_rows)
        case_summaries.append(case_summary)

    summary = _run_summary(rows, case_summaries)
    failure_diagnostics = _failure_diagnostics(rows)
    payload = {
        "schema_version": PIM_BRIDGE_EVAL_SCHEMA_VERSION,
        "run_id": run_dir.name,
        "suite_id": suite_id,
        "source": source,
        "suite_path": str(suite_path) if suite_path is not None else None,
        "backend": backend,
        "execute_external": execute_external,
        "dry_run": dry_run or not execute_external,
        "task_selection": task_selection,
        "max_tasks_per_case": max_tasks_per_case,
        "max_executed_tasks_per_case": max_executed_tasks_per_case,
        "timeout_seconds": timeout_seconds,
        "planner": planner_config,
        "debug_failures": debug_failures,
        "compare_mock_on_failure": compare_mock_on_failure,
        "keep_failure_artifacts": keep_failure_artifacts,
        "simplepim_probe": probe.to_json_dict(),
        "summary": summary,
        "case_summaries": case_summaries,
        "failure_diagnostics": failure_diagnostics,
        "rows": rows,
        "metadata": {
            "developer_only": True,
            "per_task_backend_evidence_only": True,
            "full_circuit_pim_acceleration": False,
            "normal_suite_routes_executed": False,
            "suite_routes_ignored": True,
            "cpu_tn_remains_correctness_authority": True,
        },
    }
    write_json(run_dir / "pim_bridge_eval.json", payload)
    if failure_diagnostics:
        write_json(run_dir / "pim_bridge_eval_failures.json", {"schema_version": PIM_BRIDGE_EVAL_SCHEMA_VERSION, "failures": failure_diagnostics})
    _write_csv(run_dir / "pim_bridge_eval.csv", rows, TASK_FIELDS)
    _write_csv(run_dir / "pim_bridge_eval_cases.csv", case_summaries, CASE_FIELDS)
    (run_dir / "pim_bridge_eval_summary.md").write_text(_summary_markdown(summary, case_summaries), encoding="utf-8")
    if output_plots:
        _write_plots(run_dir, rows, case_summaries)
    return run_dir


def validate_cli_options(
    *,
    suite_path: Path | None,
    case: str | None,
    n_qubits: int | None,
    backend: str,
    execute_external: bool,
    dry_run: bool,
    max_tasks_per_case: int,
    max_executed_tasks_per_case: int,
    task_selection: str,
    timeout_seconds: float,
) -> None:
    if (suite_path is None) == (case is None):
        raise ValueError("pim-bridge-eval requires exactly one of --suite or --case")
    if backend != "upmem_sdk_simulator_dense":
        raise ValueError("--backend must be upmem_sdk_simulator_dense in Wave 2E.2")
    if dry_run and execute_external:
        raise ValueError("--dry-run cannot be combined with --execute-external")
    if max_tasks_per_case < 1:
        raise ValueError("--max-tasks-per-case must be >= 1")
    if max_executed_tasks_per_case < 0:
        raise ValueError("--max-executed-tasks-per-case must be >= 0")
    if task_selection not in {"all", "eligible-only", "first-supported", "first-n"}:
        raise ValueError("--task-selection must be one of: all, eligible-only, first-supported, first-n")
    if timeout_seconds <= 0.0:
        raise ValueError("--timeout-seconds must be > 0")
    if case is not None:
        _validate_case_n_qubits(str(case), n_qubits)


def _evaluate_case(
    *,
    root_dir: Path,
    run_dir: Path,
    case_payload: dict[str, Any],
    planner_config: dict[str, Any],
    probe: Any,
    backend: str,
    execute_external: bool,
    dry_run: bool,
    max_tasks_per_case: int,
    max_executed_tasks_per_case: int,
    task_selection: PimBridgeTaskSelection,
    debug_failures: bool,
    compare_mock_on_failure: bool,
    env: Mapping[str, str],
) -> tuple[list[JsonDict], JsonDict]:
    case_payload = dict(case_payload)
    synthetic_case = is_synthetic_pressure_case(case_payload)
    if is_synthetic_pressure_case(case_payload):
        graph = build_synthetic_pressure_task_graph(case_payload)
        circuit = graph.network.circuit
        graph, _ = annotate_task_graph_with_upmem_estimates(graph)
        graph = with_path_cost_summary(graph)
        initial_tensors = {}
        circuit_payload = synthetic_pressure_manifest(graph)
    else:
        circuit = case_payload.pop("_preloaded_circuit", None)
        if circuit is None:
            circuit = load_circuit(case_payload, root_dir)
        network = build_tensor_network(circuit)
        graph = plan_task_graph_with_config(network, planner_config)
        graph, _ = annotate_task_graph_with_upmem_estimates(graph)
        graph = with_path_cost_summary(graph)
        initial_tensors = {tensor.spec.id: tensor for tensor in network.tensors}
        circuit_payload = manifest(circuit)
    case_id = str(case_payload["case_id"])
    workload_id = str(case_payload.get("workload_id", case_id))
    circuit_family = str(case_payload.get("circuit", {}).get("name", circuit.name))
    selected_indices = _selected_task_indices(graph, task_selection, max_tasks_per_case)
    if synthetic_case and selected_indices:
        initial_tensors = synthetic_pressure_initial_tensors(graph)
    rows: list[JsonDict] = []
    execution_attempts = 0
    found_supported = False

    for task_index in selected_indices:
        task = graph.tasks[task_index]
        materialization = materialize_task_inputs(
            TaskInputMaterializationRequest(
                graph=graph,
                initial_tensors=initial_tensors,
                target_task_index=task_index,
            )
        )
        preparation = None
        if materialization.status in {"initial_inputs_available", "materialized"} and materialization.left_tensor is not None and materialization.right_tensor is not None:
            preparation = prepare_dense_task(
                DenseTaskPreparationInput(
                    task=task,
                    left_tensor=materialization.left_tensor,
                    right_tensor=materialization.right_tensor,
                    simplepim_probe=probe,
                )
            )
        row = _base_task_row(
            case_id=case_id,
            workload_id=workload_id,
            circuit_family=circuit_family,
            n_qubits=int(circuit.n_qubits),
            circuit_payload=circuit_payload,
            graph=graph,
            task_index=task_index,
            materialization=materialization,
            preparation=preparation,
            backend=backend,
            dry_run=dry_run,
        )
        if row["bridge_manifest_eligible"]:
            found_supported = True

        if row["bridge_manifest_eligible"] and not dry_run:
            if execution_attempts >= max_executed_tasks_per_case:
                row["readiness_status"] = "skipped"
                row["blocker_reason"] = "execution_cap_reached"
            else:
                row = _execute_backend_task(
                    row=row,
                    run_dir=run_dir,
                    case_id=case_id,
                    task_index=task_index,
                    preparation=preparation,
                    backend=backend,
                    execute_external=execute_external,
                    debug_failures=debug_failures,
                    compare_mock_on_failure=compare_mock_on_failure,
                    env=env,
                )
                execution_attempts += 1
        rows.append(to_jsonable(row))
        if task_selection == "first-supported" and found_supported:
            break

    write_jsonl(run_dir / "cases" / case_id / "pim_bridge_eval_tasks.jsonl", rows)
    return rows, _case_summary(case_id, workload_id, circuit_family, int(circuit.n_qubits), graph, rows)


def _selected_task_indices(graph: TaskGraph, task_selection: PimBridgeTaskSelection, max_tasks_per_case: int) -> list[int]:
    all_indices = list(range(len(graph.tasks)))
    if task_selection in {"all", "first-n"}:
        return all_indices[:max_tasks_per_case]
    candidates = [index for index, task in enumerate(graph.tasks) if _cheap_estimate_candidate(task)]
    if task_selection == "first-supported":
        return candidates[:max_tasks_per_case]
    return candidates[:max_tasks_per_case]


def _cheap_estimate_candidate(task: Any) -> bool:
    estimate = task.target_estimates.get(UPMEM_DENSE_ESTIMATE_KEY, {})
    if not isinstance(estimate, dict) or not estimate.get("supported", False):
        return False
    gemm_m = int(getattr(task, "gemm_m", 0) or 0)
    gemm_k = int(getattr(task, "gemm_k", 0) or 0)
    gemm_n = int(getattr(task, "gemm_n", 0) or 0)
    if min(gemm_m, gemm_k, gemm_n) <= 0:
        return False
    if estimate.get("requires_tiling", False):
        return plan_l2_tiled_execution(gemm_m, gemm_k, gemm_n).supported
    if estimate.get("requires_host_aggregation", False):
        return False
    return True


def _base_task_row(
    *,
    case_id: str,
    workload_id: str,
    circuit_family: str,
    n_qubits: int,
    circuit_payload: JsonDict,
    graph: TaskGraph,
    task_index: int,
    materialization: Any,
    preparation: Any,
    backend: str,
    dry_run: bool,
) -> JsonDict:
    task = graph.tasks[task_index]
    estimate = task.target_estimates.get(UPMEM_DENSE_ESTIMATE_KEY, {})
    tile_plan = estimate.get("tile_plan") if isinstance(estimate, dict) else {}
    if not isinstance(tile_plan, dict):
        tile_plan = {}
    prep_payload = preparation.to_json_dict() if preparation is not None else {}
    bridge_manifest_eligible, bridge_reason = dense_bridge_backend_manifest_eligibility(preparation, backend)
    blocker = _blocker_reason(materialization, preparation, estimate, bridge_manifest_eligible, bridge_reason)
    readiness_status = "dry_run_ready" if bridge_manifest_eligible and dry_run else ("executable" if bridge_manifest_eligible else "unsupported")
    conversion_records = prep_payload.get("conversion_records") if isinstance(prep_payload, dict) else None
    route_dtype = _route_dtype(conversion_records)
    return {
        "case_id": case_id,
        "workload_id": workload_id,
        "circuit_family": circuit_family,
        "n_qubits": n_qubits,
        "circuit_parameters": circuit_payload.get("source", {}),
        "planner_engine": graph.path_summary.planner_engine,
        "planner_id": graph.path_summary.planner_id,
        "optimize_mode": graph.path_summary.optimize_mode,
        "task_index": task_index,
        "task_id": task.id,
        "input_tensor_ids": task.input_tensor_ids,
        "output_tensor_id": task.output_tensor_id,
        "gemm_m": task.gemm_m,
        "gemm_k": task.gemm_k,
        "gemm_n": task.gemm_n,
        "complex_execution_mode": _complex_execution_mode(conversion_records),
        "requires_tiling": bool(estimate.get("requires_tiling", False)) if isinstance(estimate, dict) else False,
        "requires_host_aggregation": bool(estimate.get("requires_host_aggregation", tile_plan.get("requires_host_aggregation", False))) if isinstance(estimate, dict) else False,
        "route_dtype": route_dtype,
        "supported_by_dense_estimate": bool(estimate.get("supported", False)) if isinstance(estimate, dict) else False,
        "estimated_flops": int(task.estimated_flops),
        "host_to_dpu_bytes": int(estimate.get("host_to_dpu_bytes", 0) or 0) if isinstance(estimate, dict) else 0,
        "dpu_to_host_bytes": int(estimate.get("dpu_to_host_bytes", 0) or 0) if isinstance(estimate, dict) else 0,
        "mram_to_wram_bytes": int(estimate.get("mram_to_wram_bytes", 0) or 0) if isinstance(estimate, dict) else 0,
        "tile_count": int(estimate.get("estimated_tile_count", tile_plan.get("total_tile_count", 0)) or 0) if isinstance(estimate, dict) else 0,
        "working_set_bytes": int(estimate.get("max_working_set_bytes", tile_plan.get("working_set_bytes", 0)) or 0) if isinstance(estimate, dict) else 0,
        "materialization_status": materialization.status,
        "materialization_reason": materialization.reason,
        "dense_prepare_status": prep_payload.get("status"),
        "dense_prepare_reason": prep_payload.get("reason"),
        "bridge_manifest_eligible": bridge_manifest_eligible,
        "bridge_artifact_written": False,
        "bridge_artifact_path": None,
        "readiness_status": readiness_status,
        "backend_id": backend,
        "backend_status": None,
        "blocker_reason": blocker,
        "validation_status": "not_run",
        "validation_metrics": None,
        "max_abs_error": None,
        "max_rel_error": None,
        "build_time_s": 0.0,
        "runner_total_time_s": 0.0,
        "simulator_run_time_s": 0.0,
        "kernel_invocation_count": 0,
        "output_manifest_path": None,
        "output_blob_path": None,
        "diagnostic_path": None,
        "external_command_executed": False,
        "execution_implemented": False,
        "simulator_kernel_executed": False,
        "hardware_kernel_executed": False,
    }


def _execute_backend_task(
    *,
    row: JsonDict,
    run_dir: Path,
    case_id: str,
    task_index: int,
    preparation: Any,
    backend: str,
    execute_external: bool,
    debug_failures: bool,
    compare_mock_on_failure: bool,
    env: Mapping[str, str],
) -> JsonDict:
    bridge_dir = run_dir / "cases" / sanitize(case_id) / "dense_bridge" / f"task_{task_index:04d}"
    write_dense_bridge_input_manifest(preparation, bridge_dir)
    input_manifest_path = bridge_dir / "input_manifest.json"
    bridge_result = execute_dense_bridge(input_manifest_path, backend=backend, execute_external=execute_external, env=env)
    row = dict(row)
    row["bridge_artifact_written"] = True
    row["bridge_artifact_path"] = input_manifest_path.relative_to(run_dir).as_posix()
    row["backend_status"] = bridge_result.execution_status
    row["blocker_reason"] = bridge_result.reason if bridge_result.execution_status not in _SUCCESSFUL_BACKEND_STATUSES else None
    row["output_manifest_path"] = _bridge_result_path(bridge_result.output_manifest_path, bridge_dir, run_dir)
    row["output_blob_path"] = _bridge_result_path(bridge_result.output_blob_path, bridge_dir, run_dir)
    row["external_command_executed"] = bridge_result.external_command_executed
    row["execution_implemented"] = bridge_result.execution_implemented
    metadata = {}
    if bridge_result.output_manifest is not None:
        metadata.update(dict(bridge_result.output_manifest.metadata))
        validation = dict(bridge_result.output_manifest.validation_metrics)
        row["validation_metrics"] = validation
        row["validation_status"] = "passed" if validation.get("passed") is True else ("failed" if validation else "not_applicable")
        row["max_abs_error"] = validation.get("max_abs_error")
        row["max_rel_error"] = validation.get("relative_l2_error")
        row["runner_total_time_s"] = _float_or_zero(metadata.get("runner_total_time_s", bridge_result.output_manifest.total_time_s))
        row["simulator_run_time_s"] = _float_or_zero(metadata.get("simulator_run_time_s", bridge_result.output_manifest.compute_time_s))
        row["build_time_s"] = _float_or_zero(metadata.get("build_time_s"))
        row["kernel_invocation_count"] = int(metadata.get("kernel_invocation_count", 0) or 0)
    metadata.update(dict(bridge_result.metadata))
    row["simulator_kernel_executed"] = bool(metadata.get("simulator_kernel_executed", False))
    row["hardware_kernel_executed"] = bool(metadata.get("hardware_kernel_executed", False))
    if bridge_result.execution_status in _SUCCESSFUL_BACKEND_STATUSES:
        row["readiness_status"] = "executable"
    elif bridge_result.execution_status in {"skipped", "not_implemented"}:
        row["readiness_status"] = "skipped"
    elif bridge_result.execution_status == "unsupported":
        row["readiness_status"] = "unsupported"
    else:
        row["readiness_status"] = "failed"
    if debug_failures and _needs_diagnostics(row, bridge_result):
        diagnostics = write_dense_bridge_validation_diagnostics(
            bridge_dir,
            case_id=str(row["case_id"]),
            task_index=task_index,
            task_id=str(row["task_id"]),
            compare_mock=compare_mock_on_failure,
        )
        row["diagnostic_path"] = (bridge_dir / "validation_diagnostics.json").relative_to(run_dir).as_posix()
        row["diagnostic_conclusion"] = diagnostics.get("conclusion")
    return row


def _case_summary(
    case_id: str,
    workload_id: str,
    circuit_family: str,
    n_qubits: int,
    graph: TaskGraph,
    rows: list[JsonDict],
) -> JsonDict:
    blocker_counts: dict[str, int] = {}
    for row in rows:
        reason = row.get("blocker_reason")
        if reason:
            blocker_counts[str(reason)] = blocker_counts.get(str(reason), 0) + 1
    total_transfer = sum(int(row["host_to_dpu_bytes"] or 0) + int(row["dpu_to_host_bytes"] or 0) + int(row["mram_to_wram_bytes"] or 0) for row in rows)
    return {
        "case_id": case_id,
        "workload_id": workload_id,
        "circuit_family": circuit_family,
        "n_qubits": n_qubits,
        "task_count": len(graph.tasks),
        "analyzed_task_count": len(rows),
        "attempted_task_count": sum(1 for row in rows if row["backend_status"] is not None),
        "executed_task_count": sum(1 for row in rows if row["backend_status"] in _SUCCESSFUL_BACKEND_STATUSES),
        "validated_task_count": sum(1 for row in rows if row["validation_status"] == "passed"),
        "failed_task_count": sum(1 for row in rows if row["readiness_status"] == "failed"),
        "unsupported_task_count": sum(1 for row in rows if row["readiness_status"] == "unsupported"),
        "skipped_task_count": sum(1 for row in rows if row["readiness_status"] == "skipped"),
        "blocker_counts": blocker_counts,
        "dense_candidate_count": sum(1 for row in rows if row["supported_by_dense_estimate"]),
        "simulator_executable_count": sum(1 for row in rows if row["bridge_manifest_eligible"]),
        "total_estimated_flops": sum(int(row["estimated_flops"] or 0) for row in rows),
        "total_estimated_transfer_bytes": total_transfer,
        "total_runner_time_s": sum(_float_or_zero(row["runner_total_time_s"]) for row in rows),
        "total_simulator_time_s": sum(_float_or_zero(row["simulator_run_time_s"]) for row in rows),
        "total_build_time_s": sum(_float_or_zero(row["build_time_s"]) for row in rows),
        "max_working_set_bytes": max((int(row["working_set_bytes"] or 0) for row in rows), default=0),
        "max_tile_count": max((int(row["tile_count"] or 0) for row in rows), default=0),
        "final_cpu_validation_status": "not_run",
    }


def _run_summary(rows: list[JsonDict], case_summaries: list[JsonDict]) -> JsonDict:
    blocker_counts: dict[str, int] = {}
    for row in rows:
        reason = row.get("blocker_reason")
        if reason:
            blocker_counts[str(reason)] = blocker_counts.get(str(reason), 0) + 1
    return {
        "case_count": len(case_summaries),
        "task_count": sum(int(case["task_count"]) for case in case_summaries),
        "analyzed_task_count": len(rows),
        "attempted_task_count": sum(int(case["attempted_task_count"]) for case in case_summaries),
        "executed_task_count": sum(int(case["executed_task_count"]) for case in case_summaries),
        "validated_task_count": sum(int(case["validated_task_count"]) for case in case_summaries),
        "failed_task_count": sum(int(case["failed_task_count"]) for case in case_summaries),
        "unsupported_task_count": sum(int(case["unsupported_task_count"]) for case in case_summaries),
        "skipped_task_count": sum(int(case["skipped_task_count"]) for case in case_summaries),
        "blocker_counts": blocker_counts,
        "dense_candidate_count": sum(int(case["dense_candidate_count"]) for case in case_summaries),
        "simulator_executable_count": sum(int(case["simulator_executable_count"]) for case in case_summaries),
        "total_estimated_transfer_bytes": sum(int(case["total_estimated_transfer_bytes"]) for case in case_summaries),
        "total_runner_time_s": sum(_float_or_zero(case["total_runner_time_s"]) for case in case_summaries),
        "total_simulator_time_s": sum(_float_or_zero(case["total_simulator_time_s"]) for case in case_summaries),
        "total_build_time_s": sum(_float_or_zero(case["total_build_time_s"]) for case in case_summaries),
    }


def _failure_diagnostics(rows: list[JsonDict]) -> list[JsonDict]:
    failures = []
    for row in rows:
        if row.get("diagnostic_path"):
            failures.append(
                {
                    "case_id": row.get("case_id"),
                    "task_index": row.get("task_index"),
                    "task_id": row.get("task_id"),
                    "backend_status": row.get("backend_status"),
                    "blocker_reason": row.get("blocker_reason"),
                    "diagnostic_conclusion": row.get("diagnostic_conclusion"),
                    "diagnostic_path": row.get("diagnostic_path"),
                }
            )
    return failures


def _needs_diagnostics(row: JsonDict, bridge_result: Any) -> bool:
    if bridge_result.execution_status == "failed":
        return True
    if row.get("validation_status") == "failed":
        return True
    if row.get("blocker_reason") == "validation_failed":
        return True
    return False


def _summary_markdown(summary: JsonDict, case_summaries: list[JsonDict]) -> str:
    lines = [
        "# PIM Bridge Evaluation",
        "",
        "Developer-only per-task evaluation for the UPMEM SDK simulator dense bridge. This is not full-circuit PIM acceleration.",
        "",
        f"- Cases evaluated: {summary['case_count']}",
        f"- TaskGraph tasks: {summary['task_count']}",
        f"- Analyzed tasks: {summary['analyzed_task_count']}",
        f"- Simulator execution attempts: {summary['attempted_task_count']}",
        f"- Executed simulator tasks: {summary['executed_task_count']}",
        f"- Validated simulator tasks: {summary['validated_task_count']}",
        f"- Dense bridge eligible tasks: {summary['simulator_executable_count']}",
        "",
        "## Dominant Blockers",
        "",
    ]
    blockers = summary.get("blocker_counts") or {}
    if blockers:
        for reason, count in sorted(blockers.items(), key=lambda item: (-int(item[1]), item[0])):
            lines.append(f"- {reason}: {count}")
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Cases",
            "",
            "| Case | Family | Qubits | Tasks | Eligible | Attempts | Executed | Validated | Max Tile Count |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for case in case_summaries:
        lines.append(
            "| {case_id} | {family} | {n_qubits} | {tasks} | {eligible} | {attempts} | {executed} | {validated} | {max_tile} |".format(
                case_id=case["case_id"],
                family=case["circuit_family"],
                n_qubits=case["n_qubits"],
                tasks=case["task_count"],
                eligible=case["simulator_executable_count"],
                attempts=case["attempted_task_count"],
                executed=case["executed_task_count"],
                validated=case["validated_task_count"],
                max_tile=case["max_tile_count"],
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Backend timings are bring-up timings. Build time, runner overhead, and simulator time are reported separately and should not be presented as final performance evidence.",
            "Next implementation priorities should be chosen from the blocker counts: larger shape support, executable tiling, hardware backend bring-up, SimplePIM GEMM, or sparse route work.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_csv(path: Path, rows: list[JsonDict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in fields})


def _write_plots(run_dir: Path, rows: list[JsonDict], case_summaries: list[JsonDict]) -> None:
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(plots_dir / ".matplotlib"))
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    labels = [str(case["case_id"]) for case in case_summaries]
    if not labels:
        return

    _bar_plot(plt, plots_dir / "executed_tasks_by_case.png", labels, [int(case["executed_task_count"]) for case in case_summaries], "Executed Simulator Tasks", "Tasks")
    _bar_plot(plt, plots_dir / "runner_total_time_by_case.png", labels, [float(case["total_runner_time_s"]) for case in case_summaries], "Runner Total Time By Case", "Seconds")
    _bar_plot(plt, plots_dir / "simulator_run_time_by_case.png", labels, [float(case["total_simulator_time_s"]) for case in case_summaries], "Simulator Run Time By Case", "Seconds")
    _bar_plot(plt, plots_dir / "transfer_estimate_by_case.png", labels, [int(case["total_estimated_transfer_bytes"]) for case in case_summaries], "Estimated Transfer Bytes By Case", "Bytes")
    _blocker_plot(plt, plots_dir / "blockers_by_case.png", case_summaries)
    _task_count_line_plot(plt, plots_dir / "task_count_by_qubits.png", case_summaries)


def _bar_plot(plt: Any, path: Path, labels: list[str], values: list[float], title: str, ylabel: str) -> None:
    figure, axis = plt.subplots(figsize=(max(6.0, len(labels) * 0.55), 4.0))
    axis.bar(labels, values, color="#3b82f6")
    axis.set_title(title)
    axis.set_ylabel(ylabel)
    axis.tick_params(axis="x", rotation=45)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _blocker_plot(plt: Any, path: Path, case_summaries: list[JsonDict]) -> None:
    labels = [str(case["case_id"]) for case in case_summaries]
    reasons = sorted({reason for case in case_summaries for reason in dict(case.get("blocker_counts") or {}).keys()})
    figure, axis = plt.subplots(figsize=(max(6.0, len(labels) * 0.65), 4.5))
    bottoms = [0] * len(labels)
    for reason in reasons:
        values = [int((case.get("blocker_counts") or {}).get(reason, 0)) for case in case_summaries]
        axis.bar(labels, values, bottom=bottoms, label=reason)
        bottoms = [bottom + value for bottom, value in zip(bottoms, values)]
    axis.set_title("Blockers By Case")
    axis.set_ylabel("Tasks")
    axis.tick_params(axis="x", rotation=45)
    if reasons:
        axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _task_count_line_plot(plt: Any, path: Path, case_summaries: list[JsonDict]) -> None:
    by_family: dict[str, list[JsonDict]] = {}
    for case in case_summaries:
        by_family.setdefault(str(case["circuit_family"]), []).append(case)
    figure, axis = plt.subplots(figsize=(6.5, 4.0))
    for family, cases in sorted(by_family.items()):
        ordered = sorted(cases, key=lambda item: (int(item["n_qubits"]), str(item["case_id"])))
        axis.plot([int(case["n_qubits"]) for case in ordered], [int(case["task_count"]) for case in ordered], marker="o", label=family)
    axis.set_title("Task Count By Qubits")
    axis.set_xlabel("Qubits")
    axis.set_ylabel("TaskGraph Tasks")
    if by_family:
        axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _blocker_reason(materialization: Any, preparation: Any, estimate: Any, eligible: bool, bridge_reason: str | None) -> str | None:
    if materialization.status not in {"initial_inputs_available", "materialized"}:
        return materialization.reason or "task_inputs_not_materialized"
    if not isinstance(estimate, dict) or not estimate.get("supported", False):
        return (estimate.get("reject_reason") if isinstance(estimate, dict) else None) or "unsupported_dense_estimate"
    if preparation is None:
        return "dense_preparation_missing"
    if not eligible:
        return bridge_reason or preparation.reason or "bridge_manifest_ineligible"
    return None


def _single_builtin_circuit(case: str, n_qubits: int | None):
    _validate_case_n_qubits(case, n_qubits)
    params: JsonDict = {"name": case}
    if n_qubits is not None:
        params["n_qubits"] = int(n_qubits)
    return builtin_circuit(case, params)


def _validate_case_n_qubits(case: str, n_qubits: int | None) -> None:
    lowered = case.lower()
    if lowered == "bell_2q":
        if n_qubits not in {None, 2}:
            raise ValueError("bell_2q only supports --n-qubits 2")
        return
    if lowered in _PARAMETERIZED_CASES and n_qubits is None:
        raise ValueError(f"{case} requires --n-qubits")


def _execution_environment(env: Mapping[str, str] | None, timeout_seconds: float) -> dict[str, str]:
    environment = dict(os.environ if env is None else env)
    environment["UPMEM_DENSE_SIM_TIMEOUT_SECONDS"] = str(float(timeout_seconds))
    return environment


def _run_id(suite_id: str) -> str:
    sanitized = sanitize(suite_id)
    if sanitized in {"pim_bridge_eval", "pim_bridge_eval_quick"} or sanitized.endswith("_pim_bridge_eval"):
        return sanitized
    return f"{sanitized}_pim_bridge_eval"


def _bridge_result_path(path: Path | str | None, bridge_dir: Path, run_dir: Path) -> str | None:
    if path is None:
        return None
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = bridge_dir / candidate
    try:
        return candidate.relative_to(run_dir).as_posix()
    except ValueError:
        return candidate.as_posix()


def _route_dtype(conversion_records: Any) -> str | None:
    if not isinstance(conversion_records, dict):
        return None
    values = {
        str(record.get("route_dtype"))
        for record in conversion_records.values()
        if isinstance(record, dict) and record.get("route_dtype")
    }
    if not values:
        return None
    if len(values) == 1:
        return next(iter(values))
    return ",".join(sorted(values))


def _complex_execution_mode(conversion_records: Any) -> str | None:
    if not isinstance(conversion_records, dict):
        return None
    policies = {
        str(record.get("complex_policy"))
        for record in conversion_records.values()
        if isinstance(record, dict) and record.get("complex_policy")
    }
    if not policies:
        return None
    if len(policies) == 1:
        policy = next(iter(policies))
        if policy == "split_real_imag_last_axis":
            return "split_complex_four_gemm"
        return policy
    return ",".join(sorted(policies))


def _float_or_zero(value: Any) -> float:
    try:
        return 0.0 if value is None else float(value)
    except (TypeError, ValueError):
        return 0.0


def _csv_value(value: Any) -> Any:
    value = to_jsonable(value)
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    if value is None:
        return ""
    return value
