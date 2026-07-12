from __future__ import annotations

import ctypes
import gc
import json
import multiprocessing
import statistics
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from quantum_bench.bench.config import load_suite, route_config_for
from quantum_bench.bench.reporting import artifact_ref, prune_run, validate_retention_mode, write_normalized_records, write_run_manifest
from quantum_bench.bench.result_artifacts import RESULT_ARTIFACT_SCHEMA_VERSION, normalize_parallelism_metadata
from quantum_bench.bench.run_dirs import EVIDENCE_ARTIFACT_KIND, create_run_dir, sanitize
from quantum_bench.bench.simulation_backend_probe import probe_simulation_backends
from quantum_bench.circuits import load_circuit, manifest
from quantum_bench.core.jsonio import read_jsonl, write_json, write_jsonl
from quantum_bench.core.records import BenchmarkContext, JsonDict, PathSummary, RouteResult, TaskGraph, TensorNetworkSpec, to_jsonable
from quantum_bench.environment import capture_environment
from quantum_bench.providers import route_registry
from quantum_bench.providers.base import ExecutionRoute
from quantum_bench.targets.upmem import SYNTHETIC_PRESSURE_ERROR, is_synthetic_pressure_case
from quantum_bench.tn import (
    TensorNetworkValue,
    build_execution_bundle,
    build_tensor_network,
    plan_task_graph_with_config,
    with_execution_identity,
    with_path_cost_summary,
)
from quantum_bench.validation import probability_error_metrics, tensor_to_quest_statevector, validate, validation_result_to_dict


SIMULATION_BACKEND_COMPARE_SCHEMA_VERSION = "simulation_backend_compare_v1"

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
    statevector: np.ndarray | None
    statevector_rel: Path | None
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
    run_dir = create_run_dir(
        root_dir,
        str(suite["suite_id"]),
        artifact_kind=EVIDENCE_ARTIFACT_KIND,
        route_label="simulation_backend_compare",
    )
    write_run_manifest(
        run_dir,
        run_kind="simulation_backend_compare",
        suite_id=str(suite["suite_id"]),
        suite_path=str(suite_path),
        artifact_kind=EVIDENCE_ARTIFACT_KIND,
        route_label="simulation_backend_compare",
        execution_scope="suite_backend_comparison",
        evidence_type="benchmark_execution",
        normalized_records="normalized_records.jsonl",
        summary="simulation_backend_compare_summary.json",
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
        case_result = _run_case_with_optional_isolation(root_dir, run_dir, suite, case_payload)
        rows.extend(case_result["rows"])
        comparison_rows.extend(case_result["comparisons"])
        case_rows.append(case_result["case"])
        optional_backend_reports.extend(case_result["optional_backend_reports"])
        normalized_records.extend(case_result["normalized_records"])

        if artifact_retention == "compact":
            # Compact evidence should not retain every repeat's full output
            # until a manual scaling suite has finished.
            write_jsonl(run_dir / "simulation_backend_compare_cases.jsonl", case_rows)
            write_normalized_records(run_dir, normalized_records)
            prune_run(run_dir, artifact_retention="compact")
            case_rows = read_jsonl(run_dir / "simulation_backend_compare_cases.jsonl")
            normalized_records = read_jsonl(run_dir / "normalized_records.jsonl")
            _release_completed_case_memory()

    if artifact_retention != "compact":
        write_jsonl(run_dir / "simulation_backend_compare_cases.jsonl", case_rows)
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
    if artifact_retention != "compact":
        write_normalized_records(run_dir, normalized_records)
    return SimulationBackendCompareResult(
        run_dir=run_dir,
        summary_path=run_dir / "simulation_backend_compare_summary.json",
        status="completed",
        case_count=len(case_rows),
    )


def _release_completed_case_memory() -> None:
    gc.collect()
    try:
        libc = ctypes.CDLL(None)
        trim = libc.malloc_trim
    except (AttributeError, OSError):
        return
    trim(0)


def _run_case_with_optional_isolation(root_dir: Path, run_dir: Path, suite: JsonDict, case_payload: JsonDict) -> JsonDict:
    if not bool((suite.get("metadata") or {}).get("case_process_isolation", False)):
        return _run_case(root_dir, run_dir, suite, case_payload)

    timeout_s = suite.get("timeout_s")
    timeout = float(timeout_s) if timeout_s is not None else None
    with tempfile.NamedTemporaryFile(prefix="quantum_bench_case_", suffix=".json", delete=False) as handle:
        result_path = Path(handle.name)
    result_path.unlink(missing_ok=True)
    try:
        process_context = multiprocessing.get_context("fork")
    except ValueError:  # pragma: no cover - Windows fallback for local development
        process_context = multiprocessing.get_context("spawn")
    process = process_context.Process(
        target=_isolated_case_worker,
        args=(str(root_dir), str(run_dir), suite, case_payload, str(result_path)),
    )
    process.start()
    process.join(timeout)
    try:
        if process.is_alive():
            process.terminate()
            process.join(5.0)
            return _isolated_case_failure(root_dir, run_dir, suite, case_payload, "case_process_timeout")
        if process.exitcode != 0:
            return _isolated_case_failure(
                root_dir,
                run_dir,
                suite,
                case_payload,
                f"case_process_exit_{process.exitcode}",
            )
        if not result_path.exists():
            return _isolated_case_failure(root_dir, run_dir, suite, case_payload, "case_process_result_missing")
        return json.loads(result_path.read_text(encoding="utf-8"))
    finally:
        result_path.unlink(missing_ok=True)


def _isolated_case_worker(
    root_dir: str,
    run_dir: str,
    suite: JsonDict,
    case_payload: JsonDict,
    result_path: str,
) -> None:
    try:
        result = _run_case(Path(root_dir), Path(run_dir), suite, case_payload)
        write_json(Path(result_path), result)
    except Exception as exc:
        write_json(Path(result_path), {"error": f"{type(exc).__name__}:{exc}"})
        raise


def _isolated_case_failure(
    root_dir: Path,
    run_dir: Path,
    suite: JsonDict,
    case_payload: JsonDict,
    reason: str,
) -> JsonDict:
    routes = route_registry(root_dir)
    circuit = load_circuit(case_payload, root_dir)
    return _skipped_case_result(
        root_dir,
        run_dir,
        suite,
        case_payload,
        routes,
        circuit,
        reason=reason,
    )


def _validate_suite_routes(suite: JsonDict) -> None:
    routes = list(suite.get("route_policy", {}).get("routes") or ())
    if "quest_cpu_full_state_exact" not in routes:
        raise ValueError("simulation backend comparison suite must include quest_cpu_full_state_exact")
    if len(routes) < 2:
        raise ValueError("simulation backend comparison suite must include at least two comparable routes")


def _suite_uses_only_full_state_routes(suite: JsonDict, routes: dict[str, ExecutionRoute]) -> bool:
    for route_id in suite["route_policy"]["routes"]:
        route = routes.get(str(route_id))
        if route is None:
            return False
        if route.identity.output_contract != "statevector" or "full_state" not in route.identity.simulation_method:
            return False
    return True


def _route_requires_internal_taskgraph(route: ExecutionRoute) -> bool:
    """Return whether a route needs the repository's TaskGraph lowering.

    QuEST full-state routes and external-library TN routes have their own
    execution plans. Requiring the local TaskGraph for those routes makes an
    internal lowering limit look like a backend limitation.
    """
    if "full_state" in route.identity.simulation_method:
        # Test harness routes often execute the internal NumPy TaskGraph while
        # presenting a full-state output. Real QuEST routes are external
        # processes and do not need that lowering.
        return route.identity.execution_mode in {"test", "in_process_fake"}
    return not route.identity.execution_mode.startswith("in_process_external_library")


def _suite_requires_internal_taskgraph(suite: JsonDict, routes: dict[str, ExecutionRoute]) -> bool:
    return any(
        _route_requires_internal_taskgraph(route)
        for route_id in suite["route_policy"]["routes"]
        if (route := routes.get(str(route_id))) is not None
    )


def _full_state_only_graph(circuit: Any) -> tuple[TensorNetworkValue, TaskGraph]:
    network_spec = TensorNetworkSpec(
        circuit=circuit,
        tensors=(),
        output_labels=tuple(range(int(circuit.n_qubits))),
        einsum_expression="full_state_only",
    )
    network = TensorNetworkValue(network_spec, [])
    return network, _graph_without_internal_taskgraph(network, reason="all selected routes return statevector outputs")


def _graph_without_internal_taskgraph(network: TensorNetworkValue, *, reason: str) -> TaskGraph:
    estimated_statevector_bytes = int((1 << int(network.spec.circuit.n_qubits)) * np.dtype(np.complex128).itemsize)
    path_summary = PathSummary(
        planner="not_applicable",
        optimize="not_applicable",
        path_length=0,
        largest_intermediate=1 << int(network.spec.circuit.n_qubits),
        naive_flops=0.0,
        optimized_flops=0.0,
        text=f"{reason}; internal TaskGraph planning not required",
        planner_engine="not_applicable",
        planner_id="full_state_only",
        planner_kind="full_state_only",
        optimize_mode="not_applicable",
        objective="not_applicable",
        cost_basis="not_applicable",
        options={"reason": reason},
        task_count=0,
        total_estimated_flops=0,
        peak_intermediate_bytes=estimated_statevector_bytes,
        max_intermediate_bytes=estimated_statevector_bytes,
    )
    return TaskGraph(
        network=network.spec,
        tasks=(),
        path=(),
        path_summary=path_summary,
        planning_time_s=0.0,
    )


def _run_case(root_dir: Path, run_dir: Path, suite: JsonDict, case_payload: JsonDict) -> JsonDict:
    if is_synthetic_pressure_case(case_payload):
        raise ValueError(SYNTHETIC_PRESSURE_ERROR)
    if case_payload.get("circuit", {}).get("kind") != "quest_compatible":
        raise ValueError("simulation-backend-compare requires quest_compatible deterministic circuits")

    case_id = str(case_payload["case_id"])
    case_dir = run_dir / "cases" / sanitize(case_id)
    routes = route_registry(root_dir)
    circuit = load_circuit(case_payload, root_dir)
    if not circuit.source.get("deterministic_unitary", False):
        raise ValueError(f"{case_id} is not a deterministic unitary statevector comparison case")
    case_skip_reason = case_payload.get("case_skip_reason")
    if case_skip_reason:
        return _skipped_case_result(
            root_dir,
            run_dir,
            suite,
            case_payload,
            routes,
            circuit,
            reason=str(case_skip_reason),
        )
    if _suite_uses_only_full_state_routes(suite, routes):
        network, graph = _full_state_only_graph(circuit)
        execution_bundle_rel = None
    else:
        network = build_tensor_network(circuit)
        if _suite_requires_internal_taskgraph(suite, routes):
            graph = with_execution_identity(with_path_cost_summary(plan_task_graph_with_config(network, suite["planner"])))
            execution_bundle_rel = Path("cases") / sanitize(case_id) / "execution_bundle.json"
            write_json(
                run_dir / execution_bundle_rel,
                build_execution_bundle(graph, case_id=case_id, suite_id=str(suite["suite_id"])),
            )
        else:
            graph = _graph_without_internal_taskgraph(
                network,
                reason="selected external tensor-network routes plan their own contraction trees",
            )
            execution_bundle_rel = None
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
        resource_estimate = None
        if _route_uses_resource_guard(route_config):
            try:
                resource_estimate = route.estimate(
                    graph,
                    _context(root_dir, run_dir, suite, case_payload, route_config, repeat_id=0),
                )
            except (MemoryError, RuntimeError, ValueError) as exc:
                resource_estimate = None
                resource_skip_reason = _resource_guard_estimate_failure(route_config, exc)
            else:
                resource_skip_reason = _resource_guard_skip_reason(route_config, graph, estimate=resource_estimate)
        else:
            resource_skip_reason = None
        if resource_skip_reason:
            if _route_failure_is_fatal(route_config, route, anchor_route_id):
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
            if _route_failure_is_fatal(route_config, route, anchor_route_id):
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
                route_config = route_config_for(suite, route.name)
                if _route_failure_is_fatal(route_config, route, anchor_route_id):
                    raise
                optional_backend_reports.append(_optional_backend_report(case_id, route.name, f"warmup_failed:{exc}"))

    measured_runs: list[ComparableRouteRun] = []
    for repeat_id in range(repeats):
        for route in executable_routes:
            try:
                measured_runs.append(_execute_route(root_dir, run_dir, suite, case_payload, graph, network, route, repeat_id=repeat_id))
            except MemoryError as exc:
                route_config = route_config_for(suite, route.name)
                if _route_failure_is_fatal(route_config, route, anchor_route_id):
                    raise
                reason = f"memory_error:{exc}"
                optional_backend_reports.append(_optional_backend_report(case_id, route.name, reason))
                skipped_rows.append(
                    _skipped_route_row(
                        common,
                        route=route,
                        route_metadata=_route_benchmark_metadata(route_config, route),
                        repeat_id=repeat_id,
                        repeats=repeats,
                        reason=reason,
                        validation_method=validation_method,
                        guard_status="execution_failed",
                        status="failed",
                    )
                )
            except RuntimeError as exc:
                route_config = route_config_for(suite, route.name)
                if _route_failure_is_fatal(route_config, route, anchor_route_id):
                    raise
                reason = str(exc)
                optional_backend_reports.append(_optional_backend_report(case_id, route.name, reason))
                skipped_rows.append(
                    _skipped_route_row(
                        common,
                        route=route,
                        route_metadata=_route_benchmark_metadata(route_config, route),
                        repeat_id=repeat_id,
                        repeats=repeats,
                        reason=reason,
                        validation_method=validation_method,
                        guard_status="execution_failed",
                        status="failed",
                    )
                )
            except ValueError as exc:
                route_config = route_config_for(suite, route.name)
                if _route_failure_is_fatal(route_config, route, anchor_route_id):
                    raise
                reason = f"value_error:{exc}"
                optional_backend_reports.append(_optional_backend_report(case_id, route.name, reason))
                skipped_rows.append(
                    _skipped_route_row(
                        common,
                        route=route,
                        route_metadata=_route_benchmark_metadata(route_config, route),
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
        result_metadata = dict(run.result.metadata or {})
        if run.statevector is not None and repeat_anchor.statevector is not None:
            validation_start = time.perf_counter()
            validation = validate(run.statevector, repeat_anchor.statevector, suite["tolerances"])
            validation_metrics = validation_result_to_dict(validation)
            validation_metrics.update(probability_error_metrics(run.statevector, repeat_anchor.statevector))
            validation_time_s = time.perf_counter() - validation_start
            validation_status = "passed" if validation.passed else "failed"
            row_status = "completed" if validation.passed else "validation_failed"
            error_direction = _error_direction(run.route.name, anchor_route_id)
        else:
            validation_metrics = {
                "passed": True,
                "validation_method": result_metadata.get("validation_method") or validation_method,
                "native_status": (result_metadata.get("quest") or {}).get("status"),
                "exact_output_comparable": False,
                "full_statevector_validation_available": False,
            }
            validation_time_s = 0.0
            validation_status = str(result_metadata.get("validation_status") or "passed_native_status")
            row_status = "completed"
            error_direction = "not_applicable"
        validation_metrics["error_direction"] = error_direction
        row = _row_with_metrics(
            {
                **common,
                "route_id": run.route.name,
                "backend_family": run.route.backend_family,
                **_route_benchmark_metadata(route_config_for(suite, run.route.name), run.route),
                "kernel_family": run.route.identity.kernel_family,
                "execution_model": _execution_model(run.route.identity.simulation_method),
                "parallelism_mode": result_metadata.get("parallelism_mode"),
                "parallelism_evidence_type": result_metadata.get("parallelism_evidence_type"),
                "execution_plan_kind": result_metadata.get("execution_plan_kind"),
                "execution_plan_executed": result_metadata.get("execution_plan_executed"),
                "circuit_semantics_hash": result_metadata.get("circuit_semantics_hash"),
                "tensor_network_hash": result_metadata.get("tensor_network_hash"),
                "contraction_plan_hash": result_metadata.get("contraction_plan_hash"),
                "plan_reused": result_metadata.get("plan_reused"),
                "planning_in_timed_region": result_metadata.get("planning_in_timed_region"),
                "executor_config_hash": result_metadata.get("executor_config_hash"),
                "execution_bundle_artifact": (
                    artifact_ref(run_dir, execution_bundle_rel, role="execution_bundle")
                    if result_metadata.get("contraction_plan_hash") and execution_bundle_rel is not None
                    else None
                ),
                "slicing_enabled": bool(result_metadata.get("slicing_enabled", False)),
                "slicing_backend": result_metadata.get("slicing_backend"),
                "slicing_strategy": result_metadata.get("slicing_strategy"),
                "slice_count": result_metadata.get("slice_count"),
                "sliced_indices": result_metadata.get("sliced_indices"),
                "sliced_index_sizes": result_metadata.get("sliced_index_sizes"),
                "slicing_total_flops": result_metadata.get("slicing_total_flops"),
                "unsliced_total_flops": result_metadata.get("unsliced_total_flops"),
                "slicing_flop_ratio": result_metadata.get("slicing_flop_ratio"),
                "slicing_flop_metric_source": result_metadata.get("slicing_flop_metric_source"),
                "slicing_flop_change_kind": result_metadata.get("slicing_flop_change_kind"),
                "slicing_flop_inflation_factor": result_metadata.get("slicing_flop_inflation_factor"),
                "slicing_flop_inflation": result_metadata.get("slicing_flop_inflation"),
                "slicing_max_intermediate_size": result_metadata.get("slicing_max_intermediate_size"),
                "unsliced_max_intermediate_size": result_metadata.get("unsliced_max_intermediate_size"),
                "slicing_memory_ratio": result_metadata.get("slicing_memory_ratio"),
                "slicing_memory_reduction_factor": result_metadata.get("slicing_memory_reduction_factor"),
                "slicing_reconstruction_status": result_metadata.get("slicing_reconstruction_status"),
                "slice_aware_taskgraph_available": bool(result_metadata.get("slice_aware_taskgraph_available", False)),
                "slice_reconstruction_required": result_metadata.get("slice_reconstruction_required"),
                "slice_reconstruction_status": result_metadata.get("slice_reconstruction_status"),
                "slice_task_execution_mode": result_metadata.get("slice_task_execution_mode"),
                "slice_parallel_execution": bool(result_metadata.get("slice_parallel_execution", False)),
                "slice_worker_count": result_metadata.get("slice_worker_count"),
                "frontier_scheduler_enabled": bool(result_metadata.get("frontier_scheduler_enabled", False)),
                "frontier_parallel_execution": bool(result_metadata.get("frontier_parallel_execution", False)),
                "frontier_worker_count": result_metadata.get("frontier_worker_count"),
                "frontier_wave_count": result_metadata.get("frontier_wave_count"),
                "max_frontier_width": result_metadata.get("max_frontier_width"),
                "mean_frontier_width": result_metadata.get("mean_frontier_width"),
                "frontier_executed_task_count": result_metadata.get("frontier_executed_task_count"),
                "source_frontier_completed_task_count": result_metadata.get("source_frontier_completed_task_count"),
                "frontier_executed_parallel_task_count": result_metadata.get("frontier_executed_parallel_task_count"),
                "executed_parallel_task_count": result_metadata.get("executed_parallel_task_count"),
                "scheduler_overhead_s": result_metadata.get("scheduler_overhead_s"),
                "duplicate_contraction_check": result_metadata.get("duplicate_contraction_check"),
                "missing_dependency_check": result_metadata.get("missing_dependency_check"),
                "hybrid_components": result_metadata.get("hybrid_components"),
                "hybrid_ready": bool(result_metadata.get("hybrid_ready", False)),
                "slice_model_execution_status": result_metadata.get("slice_model_execution_status"),
                "source_task_count": result_metadata.get("source_task_count"),
                "source_task_completion_count": result_metadata.get("source_task_completion_count"),
                "slice_model_slice_count": result_metadata.get("slice_model_slice_count"),
                "slice_model_task_count": result_metadata.get("slice_model_task_count"),
                "slice_model_executed_task_count": result_metadata.get("slice_model_executed_task_count"),
                "slice_parallel_wave_count": result_metadata.get("slice_parallel_wave_count"),
                "hybrid_reconstruction_validation_status": result_metadata.get("hybrid_reconstruction_validation_status"),
                "hybrid_reconstruction_max_abs_error": result_metadata.get("hybrid_reconstruction_max_abs_error"),
                "dependency_violation_detected": bool(result_metadata.get("dependency_violation_detected", False)),
                "hybrid_execution_node_count": result_metadata.get("hybrid_execution_node_count"),
                "intra_contraction_parallelism_source": result_metadata.get("intra_contraction_parallelism_source"),
                "modeled_parallelism_available": bool(result_metadata.get("modeled_parallelism_available", False)),
                "contraction_execution_target": _target(run.route.identity.hardware_target),
                "upmem_execution_mode": result_metadata.get("upmem_execution_mode"),
                "execution_backend": result_metadata.get("execution_backend"),
                "hardware_execution": bool(result_metadata.get("hardware_execution", False)),
                "hardware_timing_available": bool(result_metadata.get("hardware_timing_available", False)),
                "hardware_speedup_applicable": bool(result_metadata.get("hardware_speedup_applicable", False)),
                "cpu_fallback_used": bool(result_metadata.get("cpu_fallback_used", False)),
                "native_sdk_control_path": result_metadata.get("native_sdk_control_path"),
                "simplepim_api_used": result_metadata.get("simplepim_api_used"),
                "policy": result_metadata.get("policy", "not_applicable"),
                "quantization_mode": result_metadata.get("quantization_mode", "not_applicable"),
                "per_contraction_quantization": bool(result_metadata.get("per_contraction_quantization", False)),
                "input_dtype": result_metadata.get("input_dtype"),
                "accumulator_dtype": result_metadata.get("accumulator_dtype"),
                "path_replay_execution": bool(result_metadata.get("path_replay_execution", False)),
                "path_strategy": result_metadata.get("path_strategy"),
                "path_planner_engine": result_metadata.get("path_planner_engine"),
                "path_replay_task_count": result_metadata.get("path_replay_task_count"),
                "quantized_replay_numeric_contract": result_metadata.get("quantized_replay_numeric_contract"),
                "quantization_source_bytes": result_metadata.get("quantization_source_bytes"),
                "quantization_converted_bytes": result_metadata.get("quantization_converted_bytes"),
                "quantization_transfer_reduction_ratio": result_metadata.get("quantization_transfer_reduction_ratio"),
                "quantization_max_abs_error": result_metadata.get("quantization_max_abs_error"),
                "quantization_l2_error": result_metadata.get("quantization_l2_error"),
                "quantization_clipping_count": result_metadata.get("quantization_clipping_count"),
                "quantization_saturation_count": result_metadata.get("quantization_saturation_count"),
                "total_quantization_time_s": result_metadata.get("total_quantization_time_s"),
                "total_dequantization_time_s": result_metadata.get("total_dequantization_time_s"),
                "scaling_applied": result_metadata.get("scaling_applied"),
                "input_dtype_on_dpu": result_metadata.get("input_dtype_on_dpu"),
                "accumulator_dtype_on_dpu": result_metadata.get("accumulator_dtype_on_dpu"),
                "unquantized_mode_kind": result_metadata.get("unquantized_mode_kind"),
                "actual_h2d_bytes": result_metadata.get("actual_h2d_bytes"),
                "actual_d2h_bytes": result_metadata.get("actual_d2h_bytes"),
                "actual_transfer_bytes": result_metadata.get("actual_transfer_bytes"),
                "cpu_fallback_task_count": result_metadata.get("cpu_fallback_task_count"),
                "upmem_task_count": result_metadata.get("upmem_task_count"),
                "dpu_program_invocations": result_metadata.get("dpu_program_invocations"),
                "upmem_program_executed": result_metadata.get("upmem_program_executed"),
                "accelerator_kind": result_metadata.get("accelerator_kind") or _accelerator_kind(run.route.identity.hardware_target),
                "gpu_backend_verified": bool(result_metadata.get("gpu_backend_verified", False)),
                "gpu_program_executed": bool(result_metadata.get("gpu_program_executed", False)),
                "gpu_device_name": result_metadata.get("gpu_device_name"),
                "gpu_runtime_stack": result_metadata.get("gpu_runtime_stack"),
                "execution_scope": _execution_scope(run.route.identity.simulation_method),
                "output_kind": run.output_kind,
                "comparison_output_kind": run.comparison_output_kind,
                "status": row_status,
                "validation_status": validation_status,
                "error_direction": validation_metrics["error_direction"],
                "statevector_bytes": int(run.statevector.nbytes) if run.statevector is not None else None,
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
                "state_output_mode": result_metadata.get("state_output_mode") or "full_dump",
                "output_contract": result_metadata.get("output_contract") or run.result.output.contract,
                "output_contract_label": result_metadata.get("output_contract_label"),
                "output_contract_is_exact": bool(result_metadata.get("output_contract_is_exact", run.statevector is not None)),
                "output_contract_note": result_metadata.get("output_contract_note"),
                "performance_tier": bool(result_metadata.get("performance_tier", False)),
                "exact_output_comparable": bool(result_metadata.get("exact_output_comparable", run.statevector is not None)),
                "full_statevector_validation_available": bool(result_metadata.get("full_statevector_validation_available", run.statevector is not None)),
                "native_process_wall_time_s": result_metadata.get("native_process_wall_time_s"),
                "quest_simulation_compute_time_s": result_metadata.get("quest_simulation_compute_time_s"),
                "state_dump_requested": bool(result_metadata.get("state_dump_requested", run.statevector is not None)),
                "state_dump_time_s": result_metadata.get("state_dump_time_s"),
                "repeat_layers": result_metadata.get("repeat_layers"),
                "energy_joules": run.result.energy_joules,
                "energy_source": run.result.energy_source,
                "energy_measurement_status": result_metadata.get("energy_measurement_status"),
                "gpu_synchronized": bool(result_metadata.get("gpu_synchronized", False)),
                "tn_task_count": result_metadata.get("tn_task_count", common["tn_task_count"]),
                "tn_max_intermediate_bytes": result_metadata.get("tn_max_intermediate_bytes", common["tn_max_intermediate_bytes"]),
                "tn_estimated_flops": result_metadata.get("tn_estimated_flops", common["tn_estimated_flops"]),
                "tn_estimated_bytes": result_metadata.get("tn_estimated_bytes", common["tn_estimated_bytes"]),
                "validation_method": result_metadata.get("validation_method") or validation_method,
                "resource_guard_status": "executed",
                "resource_skip_reason": None,
                "validation_metrics": validation_metrics,
                "statevector_artifact": artifact_ref(run_dir, run.statevector_rel, role=f"{run.route.name}_statevector") if run.statevector_rel is not None else None,
                "final_tensor_artifact": artifact_ref(run_dir, run.final_tensor_rel, role=f"{run.route.name}_final_tensor") if run.final_tensor_rel is not None else None,
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
        "validation_status": "passed" if all(row["validation_status"] in {"passed", "passed_native_status", "passed_runtime_only", "skipped"} for row in rows) else "failed",
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


def _skipped_case_result(
    root_dir: Path,
    run_dir: Path,
    suite: JsonDict,
    case_payload: JsonDict,
    routes: dict[str, ExecutionRoute],
    circuit: Any,
    *,
    reason: str,
) -> JsonDict:
    case_id = str(case_payload["case_id"])
    case_dir = run_dir / "cases" / sanitize(case_id)
    network, graph = _full_state_only_graph(circuit)
    write_json(case_dir / "circuit.json", manifest(circuit))
    write_json(case_dir / "task_graph.json", graph)
    write_json(case_dir / "path_summary.json", graph.path_summary)
    anchor_route_id = _anchor_route_id(suite)
    repeats = int(suite.get("repeats", 1) or 1)
    validation_method = _validation_method(suite)
    circuit_meta = manifest(circuit)
    gate_counts = circuit_meta["gate_counts"]
    common = {
        "case_id": case_id,
        "workload_id": str(case_payload.get("workload_id", case_id)),
        "suite_id": suite["suite_id"],
        "anchor_route_id": anchor_route_id,
        "n_qubits": circuit.n_qubits,
        "gate_count": int(gate_counts["total"]),
        "two_qubit_gate_count": int(gate_counts["2q"]),
        "tn_task_count": 0,
        "tn_max_intermediate_bytes": 0,
        "tn_estimated_flops": 0,
        "tn_estimated_bytes": 0,
        **_resource_profile(suite),
    }
    rows: list[JsonDict] = []
    optional_backend_reports: list[JsonDict] = []
    for route_id in suite["route_policy"]["routes"]:
        route = routes.get(str(route_id))
        if route is None:
            optional_backend_reports.append(_optional_backend_report(case_id, str(route_id), "unknown_route"))
            continue
        route_metadata = _route_benchmark_metadata(route_config_for(suite, route.name), route)
        rows.extend(
            _skipped_route_rows(
                common,
                route=route,
                route_metadata=route_metadata,
                repeats=repeats,
                reason=reason,
                validation_method=validation_method,
                guard_status="case_resource_guard_skipped",
                status="not_executed",
            )
        )
        optional_backend_reports.append(_optional_backend_report(case_id, route.name, reason))
    comparisons = [_comparison_row(row, anchor_route_id) for row in rows]
    case_summary = {
        **common,
        "schema_version": SIMULATION_BACKEND_COMPARE_SCHEMA_VERSION,
        "basis_order": "quest_little_endian_integer_index",
        "routes": list(suite["route_policy"]["routes"]),
        "executed_routes": [],
        "skipped_route_count": len(rows),
        "anchor_route_id": anchor_route_id,
        "route_count": 0,
        "warmup_runs": int(suite.get("warmups", 0) or 0),
        "measured_runs": repeats,
        "validation_method": validation_method,
        "validation_status": "passed",
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
    if result.status != "passed":
        raise RuntimeError(result.error or f"{route.name} failed")
    repeat_label = f"repeat_{repeat_id}" if repeat_id >= 0 else f"warmup_{abs(repeat_id) - 1}"
    route_dir = Path("cases") / sanitize(case_id) / "routes" / sanitize(route.name) / repeat_label
    (run_dir / route_dir).mkdir(parents=True, exist_ok=True)
    output_write_time_s = 0.0
    output_contract = result.output.contract or route.identity.output_contract
    output_kind = str((result.output.metadata or {}).get("output_kind") or output_contract)
    if output_contract == "metrics_only":
        return ComparableRouteRun(route, result, None, None, None, output_kind, "not_applicable", repeat_id, output_write_time_s)
    if result.output.array is None:
        raise RuntimeError(result.error or f"{route.name} did not emit comparable output")
    array = np.asarray(result.output.array, dtype=np.complex128)
    if output_contract == "statevector":
        statevector = _statevector_from_state_output(array, graph.network.circuit.n_qubits, route.name)
        state_rel = route_dir / "statevector.npy"
        write_start = time.perf_counter()
        np.save(run_dir / state_rel, statevector, allow_pickle=False)
        output_write_time_s += time.perf_counter() - write_start
        return ComparableRouteRun(route, result, statevector, state_rel, None, "statevector", "statevector", repeat_id, output_write_time_s)
    if output_contract == "final_tensor":
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


def _route_uses_resource_guard(route_config: JsonDict) -> bool:
    options = dict(route_config.get("options") or {})
    return "max_estimated_intermediate_bytes" in options or "max_estimated_flops" in options


def _resource_guard_estimate_failure(route_config: JsonDict, exc: Exception) -> str | None:
    options = dict(route_config.get("options") or {})
    if bool(options.get("allow_missing_estimate", False)):
        return None
    fallback_reason = str(options.get("resource_skip_reason") or "resource_guard_exceeded")
    return f"{fallback_reason}:estimate_failed:{type(exc).__name__}:{exc}"


def _resource_guard_skip_reason(route_config: JsonDict, graph: Any, *, estimate: Any | None = None) -> str | None:
    options = dict(route_config.get("options") or {})
    if "max_estimated_intermediate_bytes" not in options and "max_estimated_flops" not in options:
        return None
    fallback_reason = str(options.get("resource_skip_reason") or "resource_guard_exceeded")
    allow_missing = bool(options.get("allow_missing_estimate", False))
    estimate_metadata = dict(getattr(estimate, "metadata", {}) or {}) if estimate is not None else {}
    if estimate_metadata.get("estimate_status") == "failed":
        if allow_missing:
            return None
        return f"{fallback_reason}:estimate_failed:{estimate_metadata.get('estimate_error', 'unknown')}"
    max_intermediate = options.get("max_estimated_intermediate_bytes")
    if max_intermediate is not None:
        intermediate_estimate = (
            getattr(estimate, "estimated_peak_memory", None)
            if estimate is not None
            else getattr(graph.path_summary, "max_intermediate_bytes", None)
        )
        if intermediate_estimate is None:
            return None if allow_missing else "unavailable_estimate"
        if int(intermediate_estimate) > int(max_intermediate):
            return f"{fallback_reason}:estimated_intermediate_bytes={int(intermediate_estimate)}:limit={int(max_intermediate)}"
    max_flops = options.get("max_estimated_flops")
    if max_flops is not None:
        flops_estimate = (
            getattr(estimate, "estimated_flops", None)
            if estimate is not None
            else getattr(graph.path_summary, "total_estimated_flops", None)
        )
        if flops_estimate is None:
            return None if allow_missing else "unavailable_estimate"
        if int(flops_estimate) > int(max_flops):
            return f"{fallback_reason}:estimated_flops={int(flops_estimate)}:limit={int(max_flops)}"
    return None


def _route_is_internal_diagnostic(route_config: JsonDict, route: ExecutionRoute) -> bool:
    route_metadata = _route_benchmark_metadata(route_config, route)
    return (
        str(route_config.get("role") or "") == "optional_diagnostic"
        or str(route_metadata.get("benchmark_role") or "") == "internal_debug_baseline"
    )


def _route_failure_is_fatal(route_config: JsonDict, route: ExecutionRoute, anchor_route_id: str) -> bool:
    if route.name == anchor_route_id:
        return True
    if _route_is_internal_diagnostic(route_config, route):
        return False
    return bool(route_config.get("required"))


def _route_benchmark_metadata(route_config: JsonDict, route: ExecutionRoute) -> JsonDict:
    defaults = {
        "quest_cpu_full_state_exact": (
            "serious_full_state_baseline",
            "Serious CPU full-state baseline and comparison anchor.",
            "Statevector output is capped by suite options for exact comparison.",
        ),
        "quest_gpu_full_state_exact": (
            "serious_gpu_full_state_baseline",
            "Serious GPU full-state baseline for direct CPU/GPU QuEST comparison.",
            "Requires verified GPU execution; unavailable candidates must not emit benchmark rows.",
        ),
        "quimb_tn_exact": (
            "serious_external_tn_baseline",
            "Serious external tensor-network baseline using Quimb/cotengra-compatible execution.",
            "External exact TN baseline; heavy cases may still be resource guarded.",
        ),
        "quimb_tn_sliced_exact": (
            "explicit_slicing_evidence",
            "Explicit Quimb/cotengra sliced tensor-network evidence route.",
            "Sliced contraction tree evidence; slice workers are not parallelized in this route.",
        ),
        "cpu_tn_einsum_exact": (
            "internal_debug_baseline",
            "Internal NumPy einsum tensor-network route for small correctness and diagnostic checks.",
            "Internal einsum expression/lowering engine limitation, not a tensor-network approach limitation.",
        ),
        "cpu_tn_frontier_exact": (
            "internal_frontier_diagnostic",
            "Internal CPU TaskGraph frontier scheduler diagnostic route.",
            "Diagnostic graph-level frontier execution evidence; not a serious external TN baseline.",
        ),
        "cpu_tn_path_replay_float64": (
            "diagnostic_path_replay_baseline",
            "Internal CPU TaskGraph path-replay route for contraction-path attribution.",
            "Diagnostic same-path replay baseline; not a serious external TN baseline.",
        ),
        "cpu_tn_path_replay_int8_quantized": (
            "diagnostic_quantized_path_replay",
            "Internal CPU TaskGraph path-replay route with per-contraction operand quantization.",
            "Diagnostic quantization attribution route; uses dequantized complex128 accumulation, not a native int8 accelerator.",
        ),
        "upmem_tn_sdk_simulator_quantized": (
            "strict_upmem_sdk_simulator_quantized",
            "Strict quantized UPMEM SDK TaskGraph runtime using SDK simulator mode.",
            "Per-task int8/int32 quantized execution; SDK simulator mode only, no hardware timing.",
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
            "parallelism_mode": None,
            "parallelism_evidence_type": None,
            "execution_plan_kind": None,
            "execution_plan_executed": False,
            "slicing_enabled": False,
            "slicing_backend": None,
            "slicing_strategy": None,
            "slice_count": None,
            "sliced_indices": None,
            "sliced_index_sizes": None,
            "slicing_total_flops": None,
            "unsliced_total_flops": None,
            "slicing_flop_ratio": None,
            "slicing_flop_metric_source": None,
            "slicing_flop_change_kind": None,
            "slicing_flop_inflation_factor": None,
            "slicing_flop_inflation": None,
            "slicing_max_intermediate_size": None,
            "unsliced_max_intermediate_size": None,
            "slicing_memory_ratio": None,
            "slicing_memory_reduction_factor": None,
            "slicing_reconstruction_status": None,
            "slice_aware_taskgraph_available": False,
            "slice_reconstruction_required": None,
            "slice_reconstruction_status": None,
            "slice_task_execution_mode": None,
            "slice_parallel_execution": False,
            "slice_worker_count": None,
            "frontier_scheduler_enabled": False,
            "frontier_parallel_execution": False,
            "frontier_worker_count": None,
            "frontier_wave_count": None,
            "max_frontier_width": None,
            "mean_frontier_width": None,
            "frontier_executed_task_count": None,
            "source_frontier_completed_task_count": None,
            "frontier_executed_parallel_task_count": None,
            "executed_parallel_task_count": None,
            "scheduler_overhead_s": None,
            "duplicate_contraction_check": None,
            "missing_dependency_check": None,
            "hybrid_components": None,
            "hybrid_ready": False,
            "slice_model_execution_status": None,
            "source_task_count": None,
            "source_task_completion_count": None,
            "slice_model_slice_count": None,
            "slice_model_task_count": None,
            "slice_model_executed_task_count": None,
            "slice_parallel_wave_count": None,
            "hybrid_reconstruction_validation_status": None,
            "hybrid_reconstruction_max_abs_error": None,
            "dependency_violation_detected": False,
            "hybrid_execution_node_count": None,
            "intra_contraction_parallelism_source": None,
            "modeled_parallelism_available": False,
            "contraction_execution_target": _target(route.identity.hardware_target),
            "upmem_execution_mode": None,
            "execution_backend": None,
            "hardware_execution": False,
            "hardware_timing_available": False,
            "hardware_speedup_applicable": False,
            "cpu_fallback_used": False,
            "native_sdk_control_path": None,
            "simplepim_api_used": None,
            "policy": "not_applicable",
            "quantization_mode": "not_applicable",
            "per_contraction_quantization": False,
            "input_dtype": None,
            "accumulator_dtype": None,
            "path_replay_execution": False,
            "path_strategy": None,
            "path_planner_engine": None,
            "path_replay_task_count": None,
            "quantized_replay_numeric_contract": None,
            "quantization_source_bytes": None,
            "quantization_converted_bytes": None,
            "quantization_transfer_reduction_ratio": None,
            "quantization_max_abs_error": None,
            "quantization_l2_error": None,
            "quantization_clipping_count": None,
            "quantization_saturation_count": None,
            "total_quantization_time_s": None,
            "total_dequantization_time_s": None,
            "scaling_applied": None,
            "dpu_program_invocations": None,
            "upmem_program_executed": False,
            "accelerator_kind": _accelerator_kind(route.identity.hardware_target),
            "gpu_backend_verified": False,
            "gpu_program_executed": False,
            "gpu_device_name": None,
            "gpu_runtime_stack": None,
            "execution_scope": _execution_scope(route.identity.simulation_method),
            "output_kind": route.identity.output_contract,
            "comparison_output_kind": "not_applicable",
            "status": status,
            "validation_status": "skipped",
            "error_direction": "not_applicable",
            "statevector_bytes": None,
            "state_output_mode": "not_executed",
            "output_contract": route.identity.output_contract,
            "output_contract_label": "not_executed",
            "output_contract_is_exact": False,
            "output_contract_note": "Route did not execute.",
            "performance_tier": False,
            "exact_output_comparable": False,
            "full_statevector_validation_available": False,
            "native_process_wall_time_s": None,
            "quest_simulation_compute_time_s": None,
            "state_dump_requested": False,
            "state_dump_time_s": None,
            "repeat_layers": None,
            "energy_joules": None,
            "energy_source": "unavailable",
            "energy_measurement_status": "unavailable",
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
            "state_output_mode": row.get("state_output_mode"),
            "performance_tier": row.get("performance_tier"),
            "exact_output_comparable": row.get("exact_output_comparable"),
            "full_statevector_validation_available": row.get("full_statevector_validation_available"),
        }
    )


def _normalized_record(run_dir: Path, row: JsonDict, *, case_id: str) -> JsonDict:
    metrics = dict(row.get("validation_metrics") or {})
    return normalize_parallelism_metadata({
        "schema_version": RESULT_ARTIFACT_SCHEMA_VERSION,
        "source_artifact": f"cases/{sanitize(case_id)}/simulation_backend_compare.json",
        "run_id": run_dir.name,
        "timestamp": None,
        "suite_id": row.get("suite_id"),
        "case_id": row.get("case_id"),
        "workload_id": row.get("workload_id"),
        "n_qubits": row.get("n_qubits"),
        "route_id": row.get("route_id"),
        "backend_id": row.get("route_id"),
        "backend_family": row.get("backend_family"),
        "benchmark_role": row.get("benchmark_role"),
        "route_role_description": row.get("route_role_description"),
        "route_limitation_scope": row.get("route_limitation_scope"),
        "kernel_family": row.get("kernel_family"),
        "execution_model": row.get("execution_model"),
        "parallelism_mode": row.get("parallelism_mode"),
        "parallelism_evidence_type": row.get("parallelism_evidence_type"),
        "execution_plan_kind": row.get("execution_plan_kind"),
        "execution_plan_executed": row.get("execution_plan_executed"),
        "slicing_enabled": bool(row.get("slicing_enabled", False)),
        "slicing_backend": row.get("slicing_backend"),
        "slicing_strategy": row.get("slicing_strategy"),
        "slice_count": row.get("slice_count"),
        "sliced_indices": row.get("sliced_indices"),
        "sliced_index_sizes": row.get("sliced_index_sizes"),
        "slicing_total_flops": row.get("slicing_total_flops"),
        "unsliced_total_flops": row.get("unsliced_total_flops"),
        "slicing_flop_ratio": row.get("slicing_flop_ratio"),
        "slicing_flop_metric_source": row.get("slicing_flop_metric_source"),
        "slicing_flop_change_kind": row.get("slicing_flop_change_kind"),
        "slicing_flop_inflation_factor": row.get("slicing_flop_inflation_factor"),
        "slicing_flop_inflation": row.get("slicing_flop_inflation"),
        "slicing_max_intermediate_size": row.get("slicing_max_intermediate_size"),
        "unsliced_max_intermediate_size": row.get("unsliced_max_intermediate_size"),
        "slicing_memory_ratio": row.get("slicing_memory_ratio"),
        "slicing_memory_reduction_factor": row.get("slicing_memory_reduction_factor"),
        "slicing_reconstruction_status": row.get("slicing_reconstruction_status"),
        "slice_aware_taskgraph_available": bool(row.get("slice_aware_taskgraph_available", False)),
        "slice_reconstruction_required": row.get("slice_reconstruction_required"),
        "slice_reconstruction_status": row.get("slice_reconstruction_status"),
        "slice_task_execution_mode": row.get("slice_task_execution_mode"),
        "slice_parallel_execution": bool(row.get("slice_parallel_execution", False)),
        "slice_worker_count": row.get("slice_worker_count"),
        "frontier_scheduler_enabled": bool(row.get("frontier_scheduler_enabled", False)),
        "frontier_parallel_execution": bool(row.get("frontier_parallel_execution", False)),
        "frontier_worker_count": row.get("frontier_worker_count"),
        "frontier_wave_count": row.get("frontier_wave_count"),
        "max_frontier_width": row.get("max_frontier_width"),
        "mean_frontier_width": row.get("mean_frontier_width"),
        "frontier_executed_task_count": row.get("frontier_executed_task_count"),
        "source_frontier_completed_task_count": row.get("source_frontier_completed_task_count"),
        "frontier_executed_parallel_task_count": row.get("frontier_executed_parallel_task_count"),
        "executed_parallel_task_count": row.get("executed_parallel_task_count"),
        "scheduler_overhead_s": row.get("scheduler_overhead_s"),
        "duplicate_contraction_check": row.get("duplicate_contraction_check"),
        "missing_dependency_check": row.get("missing_dependency_check"),
        "hybrid_components": row.get("hybrid_components"),
        "hybrid_ready": bool(row.get("hybrid_ready", False)),
        "slice_model_execution_status": row.get("slice_model_execution_status"),
        "source_task_count": row.get("source_task_count"),
        "source_task_completion_count": row.get("source_task_completion_count"),
        "slice_model_slice_count": row.get("slice_model_slice_count"),
        "slice_model_task_count": row.get("slice_model_task_count"),
        "slice_model_executed_task_count": row.get("slice_model_executed_task_count"),
        "slice_parallel_wave_count": row.get("slice_parallel_wave_count"),
        "hybrid_reconstruction_validation_status": row.get("hybrid_reconstruction_validation_status"),
        "hybrid_reconstruction_max_abs_error": row.get("hybrid_reconstruction_max_abs_error"),
        "dependency_violation_detected": bool(row.get("dependency_violation_detected", False)),
        "hybrid_execution_node_count": row.get("hybrid_execution_node_count"),
        "intra_contraction_parallelism_source": row.get("intra_contraction_parallelism_source"),
        "modeled_parallelism_available": bool(row.get("modeled_parallelism_available", False)),
        "execution_target": row.get("contraction_execution_target"),
        "contraction_execution_target": row.get("contraction_execution_target"),
        "accelerator_kind": row.get("accelerator_kind"),
        "gpu_backend_verified": bool(row.get("gpu_backend_verified", False)),
        "gpu_program_executed": bool(row.get("gpu_program_executed", False)),
        "gpu_device_name": row.get("gpu_device_name"),
        "gpu_runtime_stack": row.get("gpu_runtime_stack"),
        "upmem_execution_mode": row.get("upmem_execution_mode"),
        "execution_backend": row.get("execution_backend"),
        "native_sdk_control_path": row.get("native_sdk_control_path"),
        "simplepim_api_used": row.get("simplepim_api_used"),
        "execution_scope": row.get("execution_scope"),
        "output_kind": row.get("output_kind"),
        "comparison_output_kind": row.get("comparison_output_kind"),
        "state_output_mode": row.get("state_output_mode"),
        "output_contract": row.get("output_contract"),
        "output_contract_label": row.get("output_contract_label"),
        "output_contract_is_exact": bool(row.get("output_contract_is_exact", False)),
        "output_contract_note": row.get("output_contract_note"),
        "performance_tier": bool(row.get("performance_tier", False)),
        "exact_output_comparable": bool(row.get("exact_output_comparable", False)),
        "full_statevector_validation_available": bool(row.get("full_statevector_validation_available", False)),
        "native_process_wall_time_s": row.get("native_process_wall_time_s"),
        "quest_simulation_compute_time_s": row.get("quest_simulation_compute_time_s"),
        "state_dump_requested": bool(row.get("state_dump_requested", False)),
        "state_dump_time_s": row.get("state_dump_time_s"),
        "repeat_layers": row.get("repeat_layers"),
        "energy_joules": row.get("energy_joules"),
        "energy_source": row.get("energy_source"),
        "energy_measurement_status": row.get("energy_measurement_status"),
        "simulator_or_hardware": "simulator" if row.get("upmem_execution_mode") == "sdk_simulator" else "not_applicable",
        "policy": row.get("policy", "not_applicable"),
        "quantization_mode": row.get("quantization_mode", "not_applicable"),
        "per_contraction_quantization": bool(row.get("per_contraction_quantization", False)),
        "input_dtype": row.get("input_dtype"),
        "accumulator_dtype": row.get("accumulator_dtype"),
        "path_replay_execution": bool(row.get("path_replay_execution", False)),
        "path_strategy": row.get("path_strategy"),
        "path_planner_engine": row.get("path_planner_engine"),
        "path_replay_task_count": row.get("path_replay_task_count"),
        "quantized_replay_numeric_contract": row.get("quantized_replay_numeric_contract"),
        "quantization_source_bytes": row.get("quantization_source_bytes"),
        "quantization_converted_bytes": row.get("quantization_converted_bytes"),
        "quantization_transfer_reduction_ratio": row.get("quantization_transfer_reduction_ratio"),
        "quantization_max_abs_error": row.get("quantization_max_abs_error"),
        "quantization_l2_error": row.get("quantization_l2_error"),
        "quantization_clipping_count": row.get("quantization_clipping_count"),
        "quantization_saturation_count": row.get("quantization_saturation_count"),
        "input_dtype_on_dpu": row.get("input_dtype_on_dpu"),
        "accumulator_dtype_on_dpu": row.get("accumulator_dtype_on_dpu"),
        "scaling_applied": row.get("scaling_applied"),
        "unquantized_mode_kind": row.get("unquantized_mode_kind"),
        "actual_h2d_bytes": row.get("actual_h2d_bytes"),
        "actual_d2h_bytes": row.get("actual_d2h_bytes"),
        "actual_transfer_bytes": row.get("actual_transfer_bytes"),
        "status": row.get("status"),
        "validation_status": row.get("validation_status"),
        "max_abs_error": metrics.get("max_abs_error"),
        "l2_error": metrics.get("l2_error"),
        "norm_drift": metrics.get("norm_drift"),
        "probability_l1_error": metrics.get("probability_l1_error"),
        "probability_max_abs_error": metrics.get("probability_max_abs_error"),
        "task_count": int(row.get("tn_task_count", 0) or 0),
        "validated_task_count": int(row.get("tn_task_count", 0) or 0) if row.get("validation_status") in {"passed", "passed_native_status", "passed_runtime_only"} else 0,
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
        "total_quantization_time_s": row.get("total_quantization_time_s"),
        "total_dequantization_time_s": row.get("total_dequantization_time_s"),
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
        "hardware_execution": bool(row.get("hardware_execution", False)),
        "hardware_timing_available": bool(row.get("hardware_timing_available", False)),
        "hardware_speedup_applicable": bool(row.get("hardware_speedup_applicable", False)),
        "cpu_fallback_used": bool(row.get("cpu_fallback_used", False)),
        "cpu_fallback_task_count": row.get("cpu_fallback_task_count"),
        "upmem_task_count": row.get("upmem_task_count"),
        "dpu_program_invocations": row.get("dpu_program_invocations"),
        "upmem_program_executed": bool(row.get("upmem_program_executed", False)),
        "validation_error_metrics": metrics,
        "statevector_bytes": row.get("statevector_bytes"),
        "statevector_artifact": row.get("statevector_artifact"),
        "final_tensor_artifact": row.get("final_tensor_artifact"),
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
                "gpu_backend_verified": bool(row.get("gpu_backend_verified", False)),
                "gpu_program_executed": bool(row.get("gpu_program_executed", False)),
                "gpu_device_name": row.get("gpu_device_name"),
                "gpu_runtime_stack": row.get("gpu_runtime_stack"),
                "upmem_execution_mode": row.get("upmem_execution_mode"),
                "execution_backend": row.get("execution_backend"),
                "hardware_execution": bool(row.get("hardware_execution", False)),
                "hardware_timing_available": bool(row.get("hardware_timing_available", False)),
                "hardware_speedup_applicable": bool(row.get("hardware_speedup_applicable", False)),
                "cpu_fallback_used": bool(row.get("cpu_fallback_used", False)),
                "native_sdk_control_path": row.get("native_sdk_control_path"),
                "simplepim_api_used": row.get("simplepim_api_used"),
                "dpu_program_invocations": row.get("dpu_program_invocations"),
                "upmem_program_executed": bool(row.get("upmem_program_executed", False)),
                "quantization_mode": row.get("quantization_mode"),
                "per_contraction_quantization": bool(row.get("per_contraction_quantization", False)),
                "input_dtype": row.get("input_dtype"),
                "accumulator_dtype": row.get("accumulator_dtype"),
                "path_replay_execution": bool(row.get("path_replay_execution", False)),
                "path_strategy": row.get("path_strategy"),
                "path_planner_engine": row.get("path_planner_engine"),
                "path_replay_task_count": row.get("path_replay_task_count"),
                "quantized_replay_numeric_contract": row.get("quantized_replay_numeric_contract"),
                "quantization_source_bytes": row.get("quantization_source_bytes"),
                "quantization_converted_bytes": row.get("quantization_converted_bytes"),
                "quantization_transfer_reduction_ratio": row.get("quantization_transfer_reduction_ratio"),
                "quantization_max_abs_error": row.get("quantization_max_abs_error"),
                "quantization_l2_error": row.get("quantization_l2_error"),
                "quantization_clipping_count": row.get("quantization_clipping_count"),
                "quantization_saturation_count": row.get("quantization_saturation_count"),
                "benchmark_role": row.get("benchmark_role"),
                "route_role_description": row.get("route_role_description"),
                "route_limitation_scope": row.get("route_limitation_scope"),
                "repeat_id": row.get("repeat_id"),
                "validation_method": row.get("validation_method"),
                "timing_scope": row.get("timing_scope"),
                "state_output_mode": row.get("state_output_mode"),
                "output_contract": row.get("output_contract"),
                "output_contract_label": row.get("output_contract_label"),
                "output_contract_is_exact": bool(row.get("output_contract_is_exact", False)),
                "output_contract_note": row.get("output_contract_note"),
                "performance_tier": bool(row.get("performance_tier", False)),
                "exact_output_comparable": bool(row.get("exact_output_comparable", False)),
                "full_statevector_validation_available": bool(row.get("full_statevector_validation_available", False)),
                "native_process_wall_time_s": row.get("native_process_wall_time_s"),
                "quest_simulation_compute_time_s": row.get("quest_simulation_compute_time_s"),
                "state_dump_requested": bool(row.get("state_dump_requested", False)),
                "state_dump_time_s": row.get("state_dump_time_s"),
                "total_quantization_time_s": row.get("total_quantization_time_s"),
                "total_dequantization_time_s": row.get("total_dequantization_time_s"),
                "repeat_layers": row.get("repeat_layers"),
                "energy_source": row.get("energy_source"),
                "energy_measurement_status": row.get("energy_measurement_status"),
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
    })


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
            "case_records_artifact": "simulation_backend_compare_cases.jsonl",
            "case_artifacts_directory": "cases",
            "quest_metrics_only_route_is_not_output_comparable": True,
            "statevector_retention_policy": "compact_prunes_validated_output_tensors_and_retains_metadata",
            "warmup_runs": int(suite.get("warmups", 0) or 0),
            "measured_runs": int(suite.get("repeats", 1) or 1),
            "validation_method": str((suite.get("metadata") or {}).get("validation_method") or "full_statevector"),
            "state_output_modes": sorted({str(row.get("state_output_mode") or "unspecified") for row in rows}),
            "performance_tier": any(bool(row.get("performance_tier", False)) for row in rows),
            "performance_tier_record_count": sum(1 for row in rows if bool(row.get("performance_tier", False))),
            "exact_output_comparable_record_count": sum(1 for row in rows if bool(row.get("exact_output_comparable", False))),
            "metrics_only_record_count": sum(1 for row in rows if row.get("output_kind") == "metrics_only"),
            "timing_scope_note": (
                "Performance-tier rows use native compute timing plus process wall time and do not include full statevector validation."
                if any(bool(row.get("performance_tier", False)) for row in rows)
                else "Correctness-tier rows use full statevector output validation."
            ),
            "resource_profile": _resource_profile(suite),
            "gpu_execution_backend_added": bool((backend_probe.get("gpu_probe") or {}).get("gpu_execution_backend_added")),
            "gpu_benchmark_records_emitted": any(row.get("contraction_execution_target") == "gpu" and row.get("status") == "completed" for row in rows),
            "backend_probe": backend_probe,
            "optional_backend_reports": optional_backend_reports,
        }
    )


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
            "gpu_backend_verified": metadata.get("gpu_backend_verified"),
            "gpu_program_executed": metadata.get("gpu_program_executed"),
            "gpu_device_name": metadata.get("gpu_device_name"),
            "gpu_runtime_stack": metadata.get("gpu_runtime_stack"),
            "gpu_toolkit_metadata": metadata.get("gpu_toolkit_metadata"),
            "external_library": metadata.get("external_library", run.route.backend_family not in {"cpu", "quest"}),
            "route_metadata_keys": sorted(metadata),
        }
    )


def _optional_backend_report(case_id: str, route_id: str, reason: str) -> JsonDict:
    return {"case_id": case_id, "route_id": route_id, "status": "not_executed", "reason": reason}
