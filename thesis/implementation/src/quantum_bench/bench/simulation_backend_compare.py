from __future__ import annotations

import csv
import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from quantum_bench.bench.config import load_suite, route_config_for
from quantum_bench.bench.reporting import artifact_ref, prune_run, report_run, validate_retention_mode, write_normalized_records, write_run_manifest
from quantum_bench.bench.result_artifacts import RESULT_ARTIFACT_SCHEMA_VERSION
from quantum_bench.bench.run_dirs import create_run_dir, sanitize
from quantum_bench.bench.simulation_backend_probe import probe_simulation_backends
from quantum_bench.circuits import load_circuit, manifest
from quantum_bench.core.jsonio import write_json, write_jsonl
from quantum_bench.core.records import BenchmarkContext, JsonDict, RouteResult, to_jsonable
from quantum_bench.environment import capture_environment
from quantum_bench.providers import route_registry
from quantum_bench.providers.base import ExecutionRoute
from quantum_bench.targets.upmem import SYNTHETIC_PRESSURE_ERROR, is_synthetic_pressure_case
from quantum_bench.tn import build_tensor_network, plan_task_graph_with_config, with_path_cost_summary
from quantum_bench.validation import probability_error_metrics, tensor_to_quest_statevector, validate, validation_result_to_dict


SIMULATION_BACKEND_COMPARE_SCHEMA_VERSION = "simulation_backend_compare_v1"

RESULT_FIELDS = [
    "case_id",
    "workload_id",
    "anchor_route_id",
    "route_id",
    "backend_family",
    "benchmark_role",
    "route_role_description",
    "route_limitation_scope",
    "kernel_family",
    "execution_model",
    "contraction_execution_target",
    "accelerator_kind",
    "execution_scope",
    "output_kind",
    "comparison_output_kind",
    "status",
    "validation_status",
    "error_direction",
    "n_qubits",
    "gate_count",
    "two_qubit_gate_count",
    "statevector_bytes",
    "tn_task_count",
    "tn_max_intermediate_bytes",
    "tn_estimated_flops",
    "tn_estimated_bytes",
    "planning_time_s",
    "lowering_time_s",
    "total_wall_time_s",
    "kernel_time_s",
    "max_abs_error",
    "l2_error",
    "norm_drift",
    "probability_l1_error",
    "probability_max_abs_error",
    "statevector_artifact",
    "final_tensor_artifact",
    "dependency_metadata",
    "repeat_id",
    "measured_repeat_count",
    "setup_time_s",
    "circuit_lowering_time_s",
    "data_transfer_time_s",
    "simulation_compute_time_s",
    "validation_time_s",
    "output_materialization_time_s",
    "timing_scope",
    "gpu_synchronized",
    "validation_method",
    "expected_runtime_class",
    "expected_memory_class",
    "intended_use",
    "max_qubits",
    "manual_invocation_required",
    "expected_risk",
    "known_heavy_backends",
    "resource_guard_status",
    "resource_skip_reason",
    "total_wall_time_s_median",
    "total_wall_time_s_min",
    "total_wall_time_s_mean",
    "total_wall_time_s_std",
    "simulation_compute_time_s_median",
    "simulation_compute_time_s_min",
    "simulation_compute_time_s_mean",
    "simulation_compute_time_s_std",
]


@dataclass(frozen=True)
class SimulationBackendCompareResult:
    run_dir: Path
    summary_path: Path
    status: str
    case_count: int


@dataclass(frozen=True)
class ComparableRouteRun:
    route: ExecutionRoute
    result: RouteResult
    statevector: np.ndarray
    statevector_rel: Path
    final_tensor_rel: Path | None
    output_kind: str
    comparison_output_kind: str
    repeat_id: int
    output_materialization_time_s: float


def run_simulation_backend_compare(
    root_dir: Path,
    *,
    suite_path: Path,
    artifact_retention: str = "compact",
) -> SimulationBackendCompareResult:
    validate_retention_mode(artifact_retention)
    suite = load_suite(suite_path)
    _validate_suite_routes(suite)
    run_dir = create_run_dir(root_dir, f"{suite['suite_id']}_simulation_backend_compare")
    write_run_manifest(
        run_dir,
        run_kind="simulation_backend_compare",
        suite_id=str(suite["suite_id"]),
        suite_path=str(suite_path),
        policies=("not_applicable",),
        quantization_modes=("not_applicable",),
        artifact_retention=artifact_retention,
        root_dir=root_dir,
    )
    write_json(run_dir / "environment.json", capture_environment(root_dir))
    (run_dir / "config" / "resolved_suite.yml").write_text(yaml.safe_dump(suite, sort_keys=True), encoding="utf-8")

    rows: list[JsonDict] = []
    comparison_rows: list[JsonDict] = []
    case_rows: list[JsonDict] = []
    normalized_records: list[JsonDict] = []
    optional_backend_reports: list[JsonDict] = []
    for case_payload in suite["cases"]:
        case_result = _run_case(root_dir, run_dir, suite, case_payload)
        rows.extend(case_result["rows"])
        comparison_rows.extend(case_result["comparisons"])
        case_rows.append(case_result["case"])
        optional_backend_reports.extend(case_result["optional_backend_reports"])
        normalized_records.extend(case_result["normalized_records"])

    write_jsonl(run_dir / "simulation_backend_compare_cases.jsonl", case_rows)
    _write_csv(run_dir / "simulation_backend_compare_results.csv", rows, RESULT_FIELDS)
    _write_csv(run_dir / "simulation_backend_compare_pairs.csv", comparison_rows, _fields(comparison_rows))
    backend_probe = probe_simulation_backends(root_dir)
    summary = _summary_payload(
        suite=suite,
        suite_path=suite_path,
        rows=rows,
        case_rows=case_rows,
        comparison_rows=comparison_rows,
        normalized_records=normalized_records,
        backend_probe=backend_probe,
        optional_backend_reports=optional_backend_reports,
    )
    write_json(run_dir / "simulation_backend_compare_summary.json", summary)
    (run_dir / "comparison_summary.md").write_text(_summary_markdown(summary, comparison_rows), encoding="utf-8")
    write_normalized_records(run_dir, normalized_records)
    report_run(run_dir, output_plots=True)
    if artifact_retention == "compact":
        prune_run(run_dir, artifact_retention="compact")
    return SimulationBackendCompareResult(
        run_dir=run_dir,
        summary_path=run_dir / "simulation_backend_compare_summary.json",
        status="completed",
        case_count=len(case_rows),
    )


def _validate_suite_routes(suite: JsonDict) -> None:
    routes = list(suite.get("route_policy", {}).get("routes") or ())
    if "quest_cpu_full_state_exact" not in routes:
        raise ValueError("simulation backend comparison suite must include quest_cpu_full_state_exact")
    if len(routes) < 2:
        raise ValueError("simulation backend comparison suite must include at least two comparable routes")


def _run_case(root_dir: Path, run_dir: Path, suite: JsonDict, case_payload: JsonDict) -> JsonDict:
    if is_synthetic_pressure_case(case_payload):
        raise ValueError(SYNTHETIC_PRESSURE_ERROR)
    if case_payload.get("circuit", {}).get("kind") != "quest_compatible":
        raise ValueError("simulation-backend-compare requires quest_compatible deterministic circuits")

    case_id = str(case_payload["case_id"])
    case_dir = run_dir / "cases" / sanitize(case_id)
    circuit = load_circuit(case_payload, root_dir)
    if not circuit.source.get("deterministic_unitary", False):
        raise ValueError(f"{case_id} is not a deterministic unitary statevector comparison case")
    network = build_tensor_network(circuit)
    graph = with_path_cost_summary(plan_task_graph_with_config(network, suite["planner"]))
    write_json(case_dir / "circuit.json", manifest(circuit))
    write_json(case_dir / "task_graph.json", graph)
    write_json(case_dir / "path_summary.json", graph.path_summary)

    anchor_route_id = _anchor_route_id(suite)
    warmups = int(suite.get("warmups", 0) or 0)
    repeats = int(suite.get("repeats", 1) or 1)
    validation_method = _validation_method(suite)
    circuit_meta = manifest(circuit)
    gate_counts = circuit_meta["gate_counts"]
    resource_profile = _resource_profile(suite)
    common = {
        "case_id": case_id,
        "workload_id": str(case_payload.get("workload_id", case_id)),
        "suite_id": suite["suite_id"],
        "anchor_route_id": anchor_route_id,
        "n_qubits": circuit.n_qubits,
        "gate_count": int(gate_counts["total"]),
        "two_qubit_gate_count": int(gate_counts["2q"]),
        "tn_task_count": len(graph.tasks),
        "tn_max_intermediate_bytes": int(graph.path_summary.max_intermediate_bytes),
        "tn_estimated_flops": int(graph.path_summary.total_estimated_flops),
        "tn_estimated_bytes": int(sum(task.estimated_bytes for task in graph.tasks)),
        **resource_profile,
    }
    executable_routes: list[ExecutionRoute] = []
    skipped_rows: list[JsonDict] = []
    optional_backend_reports: list[JsonDict] = []
    routes = route_registry(root_dir)
    for route_id in suite["route_policy"]["routes"]:
        route_config = route_config_for(suite, route_id)
        required = bool(route_config.get("required")) or route_id == anchor_route_id
        route = routes.get(route_id)
        if route is None:
            if required:
                raise ValueError(f"Unknown required route: {route_id}")
            optional_backend_reports.append(_optional_backend_report(case_id, route_id, "unknown_route"))
            continue
        capabilities = route.capabilities()
        if not capabilities.can_return_output:
            if required:
                raise ValueError(f"Required route {route_id} does not return comparable output")
            optional_backend_reports.append(_optional_backend_report(case_id, route_id, "not_output_comparable"))
            continue
        resource_skip_reason = _resource_guard_skip_reason(route_config, graph)
        if resource_skip_reason:
            if required:
                raise RuntimeError(resource_skip_reason)
            optional_backend_reports.append(_optional_backend_report(case_id, route_id, resource_skip_reason))
            skipped_rows.extend(
                _skipped_route_rows(
                    common,
                    route=route,
                    route_metadata=_route_benchmark_metadata(route_config, route),
                    repeats=repeats,
                    reason=resource_skip_reason,
                    validation_method=validation_method,
                    guard_status="resource_guard_skipped",
                )
            )
            continue
        can_execute, reason = route.can_execute(
            graph,
            _context(root_dir, run_dir, suite, case_payload, route_config, repeat_id=0),
        )
        if not can_execute:
            if required:
                raise RuntimeError(reason or f"{route_id} cannot execute")
            optional_backend_reports.append(_optional_backend_report(case_id, route_id, reason or "route_unavailable"))
            continue
        executable_routes.append(route)

    if not any(route.name == anchor_route_id for route in executable_routes):
        raise RuntimeError(f"comparison anchor {anchor_route_id} did not execute")

    for warmup_id in range(warmups):
        for route in executable_routes:
            try:
                _execute_route(root_dir, run_dir, suite, case_payload, graph, network, route, repeat_id=-(warmup_id + 1))
            except (MemoryError, RuntimeError, ValueError) as exc:
                if route.name == anchor_route_id or bool(route_config_for(suite, route.name).get("required")):
                    raise
                optional_backend_reports.append(_optional_backend_report(case_id, route.name, f"warmup_failed:{exc}"))

    measured_runs: list[ComparableRouteRun] = []
    for repeat_id in range(repeats):
        for route in executable_routes:
            try:
                measured_runs.append(_execute_route(root_dir, run_dir, suite, case_payload, graph, network, route, repeat_id=repeat_id))
            except MemoryError as exc:
                if route.name == anchor_route_id or bool(route_config_for(suite, route.name).get("required")):
                    raise
                reason = f"memory_error:{exc}"
                optional_backend_reports.append(_optional_backend_report(case_id, route.name, reason))
                skipped_rows.append(
                    _skipped_route_row(
                        common,
                        route=route,
                        route_metadata=_route_benchmark_metadata(route_config_for(suite, route.name), route),
                        repeat_id=repeat_id,
                        repeats=repeats,
                        reason=reason,
                        validation_method=validation_method,
                        guard_status="execution_failed",
                        status="failed",
                    )
                )
            except RuntimeError as exc:
                if route.name == anchor_route_id or bool(route_config_for(suite, route.name).get("required")):
                    raise
                reason = str(exc)
                optional_backend_reports.append(_optional_backend_report(case_id, route.name, reason))
                skipped_rows.append(
                    _skipped_route_row(
                        common,
                        route=route,
                        route_metadata=_route_benchmark_metadata(route_config_for(suite, route.name), route),
                        repeat_id=repeat_id,
                        repeats=repeats,
                        reason=reason,
                        validation_method=validation_method,
                        guard_status="execution_failed",
                        status="failed",
                    )
                )
            except ValueError as exc:
                if route.name == anchor_route_id or bool(route_config_for(suite, route.name).get("required")):
                    raise
                reason = f"value_error:{exc}"
                optional_backend_reports.append(_optional_backend_report(case_id, route.name, reason))
                skipped_rows.append(
                    _skipped_route_row(
                        common,
                        route=route,
                        route_metadata=_route_benchmark_metadata(route_config_for(suite, route.name), route),
                        repeat_id=repeat_id,
                        repeats=repeats,
                        reason=reason,
                        validation_method=validation_method,
                        guard_status="execution_failed",
                        status="failed",
                    )
                )

    runs_by_repeat_route = {(run.repeat_id, run.route.name): run for run in measured_runs}
    for repeat_id in range(repeats):
        if (repeat_id, anchor_route_id) not in runs_by_repeat_route:
            raise RuntimeError(f"comparison anchor {anchor_route_id} did not execute for repeat {repeat_id}")

    route_runs = measured_runs

    rows: list[JsonDict] = list(skipped_rows)
    comparisons: list[JsonDict] = []
    for row in skipped_rows:
        comparisons.append(_comparison_row(row, anchor_route_id))
    for run in route_runs:
        repeat_anchor = runs_by_repeat_route[(run.repeat_id, anchor_route_id)]
        validation_start = time.perf_counter()
        validation = validate(run.statevector, repeat_anchor.statevector, suite["tolerances"])
        validation_metrics = validation_result_to_dict(validation)
        validation_metrics.update(probability_error_metrics(run.statevector, repeat_anchor.statevector))
        validation_time_s = time.perf_counter() - validation_start
        validation_metrics["error_direction"] = _error_direction(run.route.name, anchor_route_id)
        row = _row_with_metrics(
            {
                **common,
                "route_id": run.route.name,
                "backend_family": run.route.backend_family,
                **_route_benchmark_metadata(route_config_for(suite, run.route.name), run.route),
                "kernel_family": run.route.identity.kernel_family,
                "execution_model": _execution_model(run.route.identity.simulation_method),
                "contraction_execution_target": _target(run.route.identity.hardware_target),
                "accelerator_kind": _accelerator_kind(run.route.identity.hardware_target),
                "execution_scope": _execution_scope(run.route.identity.simulation_method),
                "output_kind": run.output_kind,
                "comparison_output_kind": run.comparison_output_kind,
                "status": "completed" if validation.passed else "validation_failed",
                "validation_status": "passed" if validation.passed else "failed",
                "error_direction": validation_metrics["error_direction"],
                "statevector_bytes": int(run.statevector.nbytes),
                "planning_time_s": float(run.result.profile.planning_s),
                "lowering_time_s": float(run.result.profile.lowering_s),
                "total_wall_time_s": float(run.result.profile.total_s),
                "kernel_time_s": float(run.result.profile.kernel_s),
                "repeat_id": int(run.repeat_id),
                "measured_repeat_count": repeats,
                "setup_time_s": float(run.result.profile.prepare_s),
                "circuit_lowering_time_s": float(run.result.profile.lowering_s),
                "data_transfer_time_s": float(run.result.profile.h2d_s + run.result.profile.d2h_s),
                "simulation_compute_time_s": float(run.result.profile.kernel_s),
                "validation_time_s": float(validation_time_s),
                "output_materialization_time_s": float(run.output_materialization_time_s),
                "timing_scope": "end_to_end_and_compute",
                "gpu_synchronized": bool(run.result.metadata.get("gpu_synchronized", False)),
                "validation_method": validation_method,
                "resource_guard_status": "executed",
                "resource_skip_reason": None,
                "validation_metrics": validation_metrics,
                "statevector_artifact": artifact_ref(run_dir, run.statevector_rel, role=f"{run.route.name}_statevector"),
                "final_tensor_artifact": artifact_ref(run_dir, run.final_tensor_rel, role=f"{run.route.name}_final_tensor"),
                "dependency_metadata": _dependency_metadata(run),
                "route_metadata": run.result.metadata,
            }
        )
        rows.append(row)
        comparisons.append(_comparison_row(row, anchor_route_id))
    _add_repeat_aggregates(rows)

    case_summary = {
        **common,
        "schema_version": SIMULATION_BACKEND_COMPARE_SCHEMA_VERSION,
        "basis_order": "quest_little_endian_integer_index",
        "routes": list(suite["route_policy"]["routes"]),
        "executed_routes": [route.name for route in executable_routes],
        "skipped_route_count": sum(1 for row in rows if row.get("validation_status") == "skipped"),
        "anchor_route_id": anchor_route_id,
        "route_count": len(executable_routes),
        "warmup_runs": warmups,
        "measured_runs": repeats,
        "validation_method": validation_method,
        "validation_status": "passed" if all(row["validation_status"] in {"passed", "skipped"} for row in rows) else "failed",
        "backend_families": sorted({row["backend_family"] for row in rows}),
        "execution_models": sorted({row["execution_model"] for row in rows}),
        "optional_backend_reports": optional_backend_reports,
        "rows": rows,
        "comparisons": comparisons,
    }
    write_json(case_dir / "simulation_backend_compare.json", case_summary)
    return {
        "rows": rows,
        "comparisons": comparisons,
        "case": to_jsonable(case_summary),
        "optional_backend_reports": optional_backend_reports,
        "normalized_records": [_normalized_record(run_dir, row, case_id=case_id) for row in rows],
    }


def _context(root_dir: Path, run_dir: Path, suite: JsonDict, case_payload: JsonDict, route_config: JsonDict, *, repeat_id: int) -> BenchmarkContext:
    return BenchmarkContext(
        root_dir,
        run_dir,
        suite,
        case_payload,
        route_config,
        repeat_id,
        suite["tolerances"],
        suite.get("timeout_s"),
        suite.get("memory_guard_gib"),
    )


def _execute_route(
    root_dir: Path,
    run_dir: Path,
    suite: JsonDict,
    case_payload: JsonDict,
    graph: Any,
    network: Any,
    route: ExecutionRoute,
    *,
    repeat_id: int,
) -> ComparableRouteRun:
    case_id = str(case_payload["case_id"])
    context = _context(root_dir, run_dir, suite, case_payload, route_config_for(suite, route.name), repeat_id=repeat_id)
    result = route.execute(route.prepare(graph, network, context), context)
    if result.status != "passed" or result.output.array is None:
        raise RuntimeError(result.error or f"{route.name} failed")
    array = np.asarray(result.output.array, dtype=np.complex128)
    repeat_label = f"repeat_{repeat_id}" if repeat_id >= 0 else f"warmup_{abs(repeat_id) - 1}"
    route_dir = Path("cases") / sanitize(case_id) / "routes" / sanitize(route.name) / repeat_label
    (run_dir / route_dir).mkdir(parents=True, exist_ok=True)
    output_write_time_s = 0.0
    if route.identity.output_contract == "statevector":
        statevector = _statevector_from_state_output(array, graph.network.circuit.n_qubits, route.name)
        state_rel = route_dir / "statevector.npy"
        write_start = time.perf_counter()
        np.save(run_dir / state_rel, statevector, allow_pickle=False)
        output_write_time_s += time.perf_counter() - write_start
        return ComparableRouteRun(route, result, statevector, state_rel, None, "statevector", "statevector", repeat_id, output_write_time_s)
    if route.identity.output_contract == "final_tensor":
        statevector = tensor_to_quest_statevector(array)
        tensor_rel = route_dir / "final_tensor.npy"
        state_rel = route_dir / "statevector_quest_order.npy"
        write_start = time.perf_counter()
        np.save(run_dir / tensor_rel, array, allow_pickle=False)
        np.save(run_dir / state_rel, statevector, allow_pickle=False)
        output_write_time_s += time.perf_counter() - write_start
        return ComparableRouteRun(route, result, statevector, state_rel, tensor_rel, "final_tensor", "statevector_from_final_tensor", repeat_id, output_write_time_s)
    raise RuntimeError(f"{route.name} output contract {route.identity.output_contract} is not comparable")


def _resource_profile(suite: JsonDict) -> JsonDict:
    metadata = dict(suite.get("metadata") or {})
    return {
        "expected_runtime_class": metadata.get("expected_runtime_class"),
        "expected_memory_class": metadata.get("expected_memory_class"),
        "intended_use": metadata.get("intended_use"),
        "max_qubits": metadata.get("max_qubits") or metadata.get("statevector_cap_qubits"),
        "manual_invocation_required": bool(metadata.get("manual_invocation_required", False)),
        "expected_risk": metadata.get("expected_risk") or (),
        "known_heavy_backends": metadata.get("known_heavy_backends") or (),
    }


def _resource_guard_skip_reason(route_config: JsonDict, graph: Any) -> str | None:
    options = dict(route_config.get("options") or {})
    if "max_estimated_intermediate_bytes" not in options and "max_estimated_flops" not in options:
        return None
    fallback_reason = str(options.get("resource_skip_reason") or "resource_guard_exceeded")
    allow_missing = bool(options.get("allow_missing_estimate", False))
    max_intermediate = options.get("max_estimated_intermediate_bytes")
    if max_intermediate is not None:
        estimate = getattr(graph.path_summary, "max_intermediate_bytes", None)
        if estimate is None:
            return None if allow_missing else "unavailable_estimate"
        if int(estimate) > int(max_intermediate):
            return f"{fallback_reason}:estimated_intermediate_bytes={int(estimate)}:limit={int(max_intermediate)}"
    max_flops = options.get("max_estimated_flops")
    if max_flops is not None:
        estimate = getattr(graph.path_summary, "total_estimated_flops", None)
        if estimate is None:
            return None if allow_missing else "unavailable_estimate"
        if int(estimate) > int(max_flops):
            return f"{fallback_reason}:estimated_flops={int(estimate)}:limit={int(max_flops)}"
    return None


def _route_benchmark_metadata(route_config: JsonDict, route: ExecutionRoute) -> JsonDict:
    defaults = {
        "quest_cpu_full_state_exact": (
            "serious_full_state_baseline",
            "Serious CPU full-state baseline and comparison anchor.",
            "Statevector output is capped by suite options for exact comparison.",
        ),
        "quimb_tn_exact": (
            "serious_external_tn_baseline",
            "Serious external tensor-network baseline using Quimb/cotengra-compatible execution.",
            "External exact TN baseline; heavy cases may still be resource guarded.",
        ),
        "cpu_tn_einsum_exact": (
            "internal_debug_baseline",
            "Internal NumPy einsum tensor-network route for small correctness and diagnostic checks.",
            "Internal einsum expression/lowering engine limitation, not a tensor-network approach limitation.",
        ),
    }
    default_role, default_description, default_limitation = defaults.get(
        route.name,
        (route.identity.role, f"{route.identity.display_name} route.", ""),
    )
    return {
        "benchmark_role": route_config.get("benchmark_role") or default_role,
        "route_role_description": route_config.get("route_role_description") or default_description,
        "route_limitation_scope": route_config.get("route_limitation_scope") or default_limitation,
    }


def _skipped_route_rows(
    common: JsonDict,
    *,
    route: ExecutionRoute,
    route_metadata: JsonDict,
    repeats: int,
    reason: str,
    validation_method: str,
    guard_status: str,
    status: str = "not_executed",
) -> list[JsonDict]:
    return [
        _skipped_route_row(
            common,
            route=route,
            route_metadata=route_metadata,
            repeat_id=repeat_id,
            repeats=repeats,
            reason=reason,
            validation_method=validation_method,
            guard_status=guard_status,
            status=status,
        )
        for repeat_id in range(repeats)
    ]


def _skipped_route_row(
    common: JsonDict,
    *,
    route: ExecutionRoute,
    route_metadata: JsonDict,
    repeat_id: int,
    repeats: int,
    reason: str,
    validation_method: str,
    guard_status: str,
    status: str = "not_executed",
) -> JsonDict:
    return _row_with_metrics(
        {
            **common,
            "route_id": route.name,
            "backend_family": route.backend_family,
            **route_metadata,
            "kernel_family": route.identity.kernel_family,
            "execution_model": _execution_model(route.identity.simulation_method),
            "contraction_execution_target": _target(route.identity.hardware_target),
            "accelerator_kind": _accelerator_kind(route.identity.hardware_target),
            "execution_scope": _execution_scope(route.identity.simulation_method),
            "output_kind": route.identity.output_contract,
            "comparison_output_kind": "not_applicable",
            "status": status,
            "validation_status": "skipped",
            "error_direction": "not_applicable",
            "statevector_bytes": None,
            "planning_time_s": 0.0,
            "lowering_time_s": 0.0,
            "total_wall_time_s": 0.0,
            "kernel_time_s": 0.0,
            "repeat_id": int(repeat_id),
            "measured_repeat_count": int(repeats),
            "setup_time_s": 0.0,
            "circuit_lowering_time_s": 0.0,
            "data_transfer_time_s": 0.0,
            "simulation_compute_time_s": 0.0,
            "validation_time_s": 0.0,
            "output_materialization_time_s": 0.0,
            "timing_scope": "not_executed",
            "gpu_synchronized": False,
            "validation_method": validation_method,
            "resource_guard_status": guard_status,
            "resource_skip_reason": reason,
            "unsupported_task_count": 1,
            "validation_metrics": {},
            "statevector_artifact": None,
            "final_tensor_artifact": None,
            "dependency_metadata": {},
            "route_metadata": {"not_executed_reason": reason},
        }
    )


def _statevector_from_state_output(array: np.ndarray, n_qubits: int, route_id: str) -> np.ndarray:
    state = np.asarray(array, dtype=np.complex128).reshape(-1)
    expected = 1 << int(n_qubits)
    if state.size != expected:
        raise RuntimeError(f"{route_id} emitted {state.size} amplitudes, expected {expected}")
    return state


def _anchor_route_id(suite: JsonDict) -> str:
    for route_config in suite.get("_route_configs", []):
        if route_config.get("role") == "comparison_anchor":
            return str(route_config["id"])
    routes = list(suite.get("route_policy", {}).get("routes") or ())
    if "quest_cpu_full_state_exact" in routes:
        return "quest_cpu_full_state_exact"
    raise ValueError("simulation backend comparison suite must define a comparison anchor")


def _row_with_metrics(row: JsonDict) -> JsonDict:
    metrics = dict(row.get("validation_metrics") or {})
    row = dict(row)
    row["max_abs_error"] = metrics.get("max_abs_error")
    row["l2_error"] = metrics.get("l2_error")
    row["norm_drift"] = metrics.get("norm_drift")
    row["probability_l1_error"] = metrics.get("probability_l1_error")
    row["probability_max_abs_error"] = metrics.get("probability_max_abs_error")
    return to_jsonable(row)


def _validation_method(suite: JsonDict) -> str:
    metadata = suite.get("metadata") or {}
    return str(metadata.get("validation_method") or "full_statevector")


def _add_repeat_aggregates(rows: list[JsonDict]) -> None:
    grouped: dict[tuple[str, str], list[JsonDict]] = {}
    for row in rows:
        grouped.setdefault((str(row.get("case_id")), str(row.get("route_id"))), []).append(row)
    for group in grouped.values():
        total_values = [float(row.get("total_wall_time_s") or 0.0) for row in group]
        compute_values = [float(row.get("simulation_compute_time_s") or 0.0) for row in group]
        total_stats = _repeat_stats(total_values)
        compute_stats = _repeat_stats(compute_values)
        for row in group:
            row.update(
                {
                    "total_wall_time_s_median": total_stats["median"],
                    "total_wall_time_s_min": total_stats["min"],
                    "total_wall_time_s_mean": total_stats["mean"],
                    "total_wall_time_s_std": total_stats["std"],
                    "simulation_compute_time_s_median": compute_stats["median"],
                    "simulation_compute_time_s_min": compute_stats["min"],
                    "simulation_compute_time_s_mean": compute_stats["mean"],
                    "simulation_compute_time_s_std": compute_stats["std"],
                }
            )


def _repeat_stats(values: list[float]) -> JsonDict:
    if not values:
        return {"median": None, "min": None, "mean": None, "std": None}
    return {
        "median": float(statistics.median(values)),
        "min": float(min(values)),
        "mean": float(statistics.mean(values)),
        "std": float(statistics.pstdev(values)) if len(values) > 1 else 0.0,
    }


def _comparison_row(row: JsonDict, anchor_route_id: str) -> JsonDict:
    metrics = dict(row.get("validation_metrics") or {})
    return to_jsonable(
        {
            "schema_version": SIMULATION_BACKEND_COMPARE_SCHEMA_VERSION,
            "case_id": row["case_id"],
            "workload_id": row["workload_id"],
            "anchor_route_id": anchor_route_id,
            "route_id": row["route_id"],
            "backend_family": row["backend_family"],
            "benchmark_role": row.get("benchmark_role"),
            "route_limitation_scope": row.get("route_limitation_scope"),
            "execution_model": row["execution_model"],
            "validation_status": row["validation_status"],
            "error_direction": row["error_direction"],
            "max_abs_error": metrics.get("max_abs_error"),
            "l2_error": metrics.get("l2_error"),
            "norm_drift": metrics.get("norm_drift"),
            "probability_l1_error": metrics.get("probability_l1_error"),
            "probability_max_abs_error": metrics.get("probability_max_abs_error"),
            "repeat_id": row.get("repeat_id"),
            "validation_method": row.get("validation_method"),
        }
    )


def _normalized_record(run_dir: Path, row: JsonDict, *, case_id: str) -> JsonDict:
    metrics = dict(row.get("validation_metrics") or {})
    return {
        "schema_version": RESULT_ARTIFACT_SCHEMA_VERSION,
        "source_artifact": f"cases/{sanitize(case_id)}/simulation_backend_compare.json",
        "run_id": run_dir.name,
        "timestamp": None,
        "suite_id": row.get("suite_id"),
        "case_id": row.get("case_id"),
        "workload_id": row.get("workload_id"),
        "route_id": row.get("route_id"),
        "backend_id": row.get("route_id"),
        "backend_family": row.get("backend_family"),
        "benchmark_role": row.get("benchmark_role"),
        "route_role_description": row.get("route_role_description"),
        "route_limitation_scope": row.get("route_limitation_scope"),
        "kernel_family": row.get("kernel_family"),
        "execution_model": row.get("execution_model"),
        "execution_target": row.get("contraction_execution_target"),
        "contraction_execution_target": row.get("contraction_execution_target"),
        "accelerator_kind": row.get("accelerator_kind"),
        "upmem_execution_mode": None,
        "native_sdk_control_path": None,
        "simplepim_api_used": None,
        "execution_scope": row.get("execution_scope"),
        "output_kind": row.get("output_kind"),
        "comparison_output_kind": row.get("comparison_output_kind"),
        "simulator_or_hardware": "not_applicable",
        "policy": "not_applicable",
        "quantization_mode": "not_applicable",
        "status": row.get("status"),
        "validation_status": row.get("validation_status"),
        "task_count": int(row.get("tn_task_count", 0) or 0),
        "validated_task_count": int(row.get("tn_task_count", 0) or 0) if row.get("validation_status") == "passed" else 0,
        "unsupported_task_count": int(row.get("unsupported_task_count", 0) or 0),
        "planning_time_s": row.get("planning_time_s"),
        "lowering_time_s": row.get("lowering_time_s"),
        "total_wall_time_s": float(row.get("total_wall_time_s", 0.0) or 0.0),
        "kernel_time_s": float(row.get("kernel_time_s", 0.0) or 0.0),
        "repeat_id": int(row.get("repeat_id", 0) or 0),
        "measured_repeat_count": int(row.get("measured_repeat_count", 1) or 1),
        "setup_time_s": row.get("setup_time_s"),
        "circuit_lowering_time_s": row.get("circuit_lowering_time_s"),
        "data_transfer_time_s": row.get("data_transfer_time_s"),
        "simulation_compute_time_s": row.get("simulation_compute_time_s"),
        "validation_time_s": row.get("validation_time_s"),
        "output_materialization_time_s": row.get("output_materialization_time_s"),
        "timing_scope": row.get("timing_scope"),
        "gpu_synchronized": bool(row.get("gpu_synchronized", False)),
        "validation_method": row.get("validation_method"),
        "expected_runtime_class": row.get("expected_runtime_class"),
        "expected_memory_class": row.get("expected_memory_class"),
        "intended_use": row.get("intended_use"),
        "max_qubits": row.get("max_qubits"),
        "manual_invocation_required": bool(row.get("manual_invocation_required", False)),
        "expected_risk": row.get("expected_risk"),
        "known_heavy_backends": row.get("known_heavy_backends"),
        "resource_guard_status": row.get("resource_guard_status"),
        "resource_skip_reason": row.get("resource_skip_reason"),
        "total_wall_time_s_median": row.get("total_wall_time_s_median"),
        "total_wall_time_s_min": row.get("total_wall_time_s_min"),
        "total_wall_time_s_mean": row.get("total_wall_time_s_mean"),
        "total_wall_time_s_std": row.get("total_wall_time_s_std"),
        "simulation_compute_time_s_median": row.get("simulation_compute_time_s_median"),
        "simulation_compute_time_s_min": row.get("simulation_compute_time_s_min"),
        "simulation_compute_time_s_mean": row.get("simulation_compute_time_s_mean"),
        "simulation_compute_time_s_std": row.get("simulation_compute_time_s_std"),
        "host_transfer_time_s": None,
        "build_time_s": 0.0,
        "launch_overhead_s": None,
        "simulator_relative_time": None,
        "hardware_speedup": "not_applicable",
        "validation_error_metrics": metrics,
        "statevector_bytes": row.get("statevector_bytes"),
        "tn_task_count": row.get("tn_task_count"),
        "tn_max_intermediate_bytes": row.get("tn_max_intermediate_bytes"),
        "tn_estimated_flops": row.get("tn_estimated_flops"),
        "tn_estimated_bytes": row.get("tn_estimated_bytes"),
        "dependency_metadata": row.get("dependency_metadata"),
        "notes": json.dumps(
            {
                "anchor_route_id": row.get("anchor_route_id"),
                "error_direction": row.get("error_direction"),
                "n_qubits": row.get("n_qubits"),
                "gate_count": row.get("gate_count"),
                "two_qubit_gate_count": row.get("two_qubit_gate_count"),
                "dependency_metadata": row.get("dependency_metadata"),
                "benchmark_role": row.get("benchmark_role"),
                "route_role_description": row.get("route_role_description"),
                "route_limitation_scope": row.get("route_limitation_scope"),
                "repeat_id": row.get("repeat_id"),
                "validation_method": row.get("validation_method"),
                "timing_scope": row.get("timing_scope"),
                "resource_profile": {
                    "expected_runtime_class": row.get("expected_runtime_class"),
                    "expected_memory_class": row.get("expected_memory_class"),
                    "intended_use": row.get("intended_use"),
                    "max_qubits": row.get("max_qubits"),
                    "manual_invocation_required": bool(row.get("manual_invocation_required", False)),
                    "expected_risk": row.get("expected_risk"),
                    "known_heavy_backends": row.get("known_heavy_backends"),
                },
                "resource_guard_status": row.get("resource_guard_status"),
                "resource_skip_reason": row.get("resource_skip_reason"),
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        "warnings": row.get("resource_skip_reason") or "",
    }


def _summary_payload(
    *,
    suite: JsonDict,
    suite_path: Path,
    rows: list[JsonDict],
    case_rows: list[JsonDict],
    comparison_rows: list[JsonDict],
    normalized_records: list[JsonDict],
    backend_probe: JsonDict,
    optional_backend_reports: list[JsonDict],
) -> JsonDict:
    return to_jsonable(
        {
            "schema_version": SIMULATION_BACKEND_COMPARE_SCHEMA_VERSION,
            "suite_id": suite["suite_id"],
            "suite_path": str(suite_path),
            "case_count": len(case_rows),
            "record_count": len(rows),
            "passed_case_count": sum(1 for row in case_rows if row["validation_status"] == "passed"),
            "failed_case_count": sum(1 for row in case_rows if row["validation_status"] != "passed"),
            "routes": list(suite["route_policy"]["routes"]),
            "anchor_route_id": _anchor_route_id(suite),
            "execution_models": sorted({str(row.get("execution_model")) for row in rows}),
            "backend_families": sorted({str(row.get("backend_family")) for row in rows}),
            "benchmark_roles": sorted({str(row.get("benchmark_role")) for row in rows}),
            "root_normalized_records_are_canonical": True,
            "normalized_records_artifact": "normalized_records.jsonl",
            "quest_metrics_only_route_is_not_output_comparable": True,
            "statevector_retention_policy": "compact_retains_statevectors_under_configured_caps",
            "warmup_runs": int(suite.get("warmups", 0) or 0),
            "measured_runs": int(suite.get("repeats", 1) or 1),
            "validation_method": str((suite.get("metadata") or {}).get("validation_method") or "full_statevector"),
            "resource_profile": _resource_profile(suite),
            "gpu_execution_backend_added": bool((backend_probe.get("gpu_probe") or {}).get("gpu_execution_backend_added")),
            "gpu_benchmark_records_emitted": any(row.get("contraction_execution_target") == "gpu" and row.get("status") == "completed" for row in rows),
            "backend_probe": backend_probe,
            "optional_backend_reports": optional_backend_reports,
            "rows": rows,
            "cases": case_rows,
            "comparisons": comparison_rows,
            "normalized_records": normalized_records,
        }
    )


def _summary_markdown(summary: JsonDict, comparison_rows: list[JsonDict]) -> str:
    lines = [
        "# Simulation Backend Comparison",
        "",
        f"Suite: `{summary['suite_id']}`",
        f"Cases: {summary['case_count']}",
        f"Anchor route: `{summary['anchor_route_id']}`",
        "",
        "Comparable backends execute the same deterministic unitary circuit semantics.",
        "Error directions are labeled per route; the anchor is a comparison anchor, not an implicit claim that every other backend is numerically subordinate.",
        "GPU feasibility is reported separately and does not create benchmark rows without real GPU execution.",
        "",
        "## Resource Profile",
        "",
        "| Field | Value |",
        "| --- | --- |",
    ]
    profile = summary.get("resource_profile") or {}
    for key in (
        "expected_runtime_class",
        "expected_memory_class",
        "intended_use",
        "max_qubits",
        "manual_invocation_required",
        "expected_risk",
        "known_heavy_backends",
    ):
        lines.append(f"| {key} | {profile.get(key)} |")
    lines.extend(
        [
            "",
            "## Backend Metadata",
            "",
            "| Route | Benchmark role | Backend | Model | Target | Output | Limitation scope |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    seen: set[str] = set()
    for row in summary["rows"]:
        route_id = str(row["route_id"])
        if route_id in seen:
            continue
        seen.add(route_id)
        lines.append(
            f"| {route_id} | {row.get('benchmark_role')} | {row['backend_family']} | {row['execution_model']} | "
            f"{row['contraction_execution_target']} | {row['output_kind']} | {row.get('route_limitation_scope')} |"
        )
    lines.extend(
        [
            "",
            "## Output Agreement",
            "",
            "| Case | Route | Status | Max abs error | L2 error | Probability L1 error |",
            "| --- | --- | --- | ---: | ---: | ---: |",
        ]
    )
    for row in comparison_rows:
        lines.append(
            f"| {row['case_id']} | {row['route_id']} | {row['validation_status']} | "
            f"{row.get('max_abs_error')} | {row.get('l2_error')} | {row.get('probability_l1_error')} |"
        )
    lines.extend(
        [
            "",
            "## Timing Breakdown",
            "",
            "| Route | Repeat | Total wall time s | Compute time s | Setup s | Transfer s | Validation s | Output write s |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in summary["rows"]:
        lines.append(
            f"| {row['route_id']} | {row.get('repeat_id')} | {row.get('total_wall_time_s')} | {row.get('simulation_compute_time_s')} | "
            f"{row.get('setup_time_s')} | {row.get('data_transfer_time_s')} | {row.get('validation_time_s')} | "
            f"{row.get('output_materialization_time_s')} |"
        )
    lines.extend(
        [
            "",
            "## TN Path / Intermediate Metrics",
            "",
            "| Case | Route | TN tasks | Max intermediate bytes | Estimated FLOPs | Estimated bytes |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in summary["rows"]:
        if row.get("execution_model") == "tensor_network":
            lines.append(
                f"| {row['case_id']} | {row['route_id']} | {row.get('tn_task_count')} | "
                f"{row.get('tn_max_intermediate_bytes')} | {row.get('tn_estimated_flops')} | {row.get('tn_estimated_bytes')} |"
            )
    lines.extend(
        [
            "",
            "## Optional Backend Feasibility",
            "",
            f"- GPU execution backend added: `{summary.get('gpu_execution_backend_added')}`",
            f"- GPU benchmark records emitted: `{summary.get('gpu_benchmark_records_emitted')}`",
        ]
    )
    optional = summary.get("optional_backend_reports") or []
    if optional:
        for item in optional:
            lines.append(f"- `{item['route_id']}` for `{item['case_id']}`: {item['reason']}")
    else:
        lines.append("- No optional comparable backend was skipped.")
    lines.append("")
    return "\n".join(lines)


def _fields(rows: list[JsonDict]) -> list[str]:
    if not rows:
        return ["case_id"]
    fields = set()
    for row in rows:
        fields.update(row)
    preferred = [
        "schema_version",
        "case_id",
        "workload_id",
        "anchor_route_id",
        "route_id",
        "backend_family",
        "execution_model",
        "validation_status",
        "max_abs_error",
        "l2_error",
        "probability_l1_error",
    ]
    return preferred + sorted(fields - set(preferred))


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


def _execution_model(simulation_method: str) -> str:
    return "full_state" if "full_state" in simulation_method else "tensor_network"


def _execution_scope(simulation_method: str) -> str:
    return "full_circuit" if "full_state" in simulation_method else "full_taskgraph"


def _target(hardware_target: str) -> str:
    if "gpu" in hardware_target:
        return "gpu"
    if "upmem" in hardware_target:
        return "upmem"
    return "cpu"


def _accelerator_kind(hardware_target: str) -> str:
    if "amd" in hardware_target:
        return "amd_gpu"
    if "nvidia" in hardware_target or "cuda" in hardware_target:
        return "nvidia_gpu"
    if "gpu" in hardware_target:
        return "gpu"
    if "upmem" in hardware_target:
        return "upmem"
    return "none"


def _error_direction(route_id: str, anchor_route_id: str) -> str:
    if route_id == anchor_route_id:
        return "self_reference"
    return f"{route_id}_minus_{anchor_route_id}_statevector"


def _dependency_metadata(run: ComparableRouteRun) -> JsonDict:
    metadata = dict(run.result.metadata or {})
    return to_jsonable(
        {
            "dependency_versions": metadata.get("dependency_versions"),
            "quest": metadata.get("quest"),
            "external_library": metadata.get("external_library", run.route.backend_family not in {"cpu", "quest"}),
            "route_metadata_keys": sorted(metadata),
        }
    )


def _optional_backend_report(case_id: str, route_id: str, reason: str) -> JsonDict:
    return {"case_id": case_id, "route_id": route_id, "status": "not_executed", "reason": reason}
