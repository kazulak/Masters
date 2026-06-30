from __future__ import annotations

import csv
import json
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
    route_runs: list[ComparableRouteRun] = []
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
        can_execute, reason = route.can_execute(
            graph,
            _context(root_dir, run_dir, suite, case_payload, route_config),
        )
        if not can_execute:
            if required:
                raise RuntimeError(reason or f"{route_id} cannot execute")
            optional_backend_reports.append(_optional_backend_report(case_id, route_id, reason or "route_unavailable"))
            continue
        try:
            route_runs.append(_execute_route(root_dir, run_dir, suite, case_payload, graph, network, route))
        except RuntimeError as exc:
            if required:
                raise
            optional_backend_reports.append(_optional_backend_report(case_id, route_id, str(exc)))

    runs_by_route = {run.route.name: run for run in route_runs}
    if anchor_route_id not in runs_by_route:
        raise RuntimeError(f"comparison anchor {anchor_route_id} did not execute")
    anchor_run = runs_by_route[anchor_route_id]

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
        "tn_task_count": len(graph.tasks),
        "tn_max_intermediate_bytes": int(graph.path_summary.max_intermediate_bytes),
        "tn_estimated_flops": int(graph.path_summary.total_estimated_flops),
        "tn_estimated_bytes": int(sum(task.estimated_bytes for task in graph.tasks)),
    }
    rows: list[JsonDict] = []
    comparisons: list[JsonDict] = []
    for run in route_runs:
        validation = validate(run.statevector, anchor_run.statevector, suite["tolerances"])
        validation_metrics = validation_result_to_dict(validation)
        validation_metrics.update(probability_error_metrics(run.statevector, anchor_run.statevector))
        validation_metrics["error_direction"] = _error_direction(run.route.name, anchor_route_id)
        row = _row_with_metrics(
            {
                **common,
                "route_id": run.route.name,
                "backend_family": run.route.backend_family,
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
                "validation_metrics": validation_metrics,
                "statevector_artifact": artifact_ref(run_dir, run.statevector_rel, role=f"{run.route.name}_statevector"),
                "final_tensor_artifact": artifact_ref(run_dir, run.final_tensor_rel, role=f"{run.route.name}_final_tensor"),
                "dependency_metadata": _dependency_metadata(run),
                "route_metadata": run.result.metadata,
            }
        )
        rows.append(row)
        comparisons.append(_comparison_row(row, anchor_route_id))

    case_summary = {
        **common,
        "schema_version": SIMULATION_BACKEND_COMPARE_SCHEMA_VERSION,
        "basis_order": "quest_little_endian_integer_index",
        "routes": [run.route.name for run in route_runs],
        "anchor_route_id": anchor_route_id,
        "route_count": len(route_runs),
        "validation_status": "passed" if all(row["validation_status"] == "passed" for row in rows) else "failed",
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


def _context(root_dir: Path, run_dir: Path, suite: JsonDict, case_payload: JsonDict, route_config: JsonDict) -> BenchmarkContext:
    return BenchmarkContext(
        root_dir,
        run_dir,
        suite,
        case_payload,
        route_config,
        0,
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
) -> ComparableRouteRun:
    case_id = str(case_payload["case_id"])
    context = _context(root_dir, run_dir, suite, case_payload, route_config_for(suite, route.name))
    result = route.execute(route.prepare(graph, network, context), context)
    if result.status != "passed" or result.output.array is None:
        raise RuntimeError(result.error or f"{route.name} failed")
    array = np.asarray(result.output.array, dtype=np.complex128)
    route_dir = Path("cases") / sanitize(case_id) / "routes" / sanitize(route.name)
    (run_dir / route_dir).mkdir(parents=True, exist_ok=True)
    if route.identity.output_contract == "statevector":
        statevector = _statevector_from_state_output(array, graph.network.circuit.n_qubits, route.name)
        state_rel = route_dir / "statevector.npy"
        np.save(run_dir / state_rel, statevector, allow_pickle=False)
        return ComparableRouteRun(route, result, statevector, state_rel, None, "statevector", "statevector")
    if route.identity.output_contract == "final_tensor":
        statevector = tensor_to_quest_statevector(array)
        tensor_rel = route_dir / "final_tensor.npy"
        state_rel = route_dir / "statevector_quest_order.npy"
        np.save(run_dir / tensor_rel, array, allow_pickle=False)
        np.save(run_dir / state_rel, statevector, allow_pickle=False)
        return ComparableRouteRun(route, result, statevector, state_rel, tensor_rel, "final_tensor", "statevector_from_final_tensor")
    raise RuntimeError(f"{route.name} output contract {route.identity.output_contract} is not comparable")


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
            "execution_model": row["execution_model"],
            "validation_status": row["validation_status"],
            "error_direction": row["error_direction"],
            "max_abs_error": metrics.get("max_abs_error"),
            "l2_error": metrics.get("l2_error"),
            "norm_drift": metrics.get("norm_drift"),
            "probability_l1_error": metrics.get("probability_l1_error"),
            "probability_max_abs_error": metrics.get("probability_max_abs_error"),
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
        "unsupported_task_count": 0,
        "planning_time_s": row.get("planning_time_s"),
        "lowering_time_s": row.get("lowering_time_s"),
        "total_wall_time_s": float(row.get("total_wall_time_s", 0.0) or 0.0),
        "kernel_time_s": float(row.get("kernel_time_s", 0.0) or 0.0),
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
        "notes": json.dumps(
            {
                "anchor_route_id": row.get("anchor_route_id"),
                "error_direction": row.get("error_direction"),
                "n_qubits": row.get("n_qubits"),
                "gate_count": row.get("gate_count"),
                "two_qubit_gate_count": row.get("two_qubit_gate_count"),
                "dependency_metadata": row.get("dependency_metadata"),
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        "warnings": "",
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
            "root_normalized_records_are_canonical": True,
            "normalized_records_artifact": "normalized_records.jsonl",
            "quest_metrics_only_route_is_not_output_comparable": True,
            "statevector_retention_policy": "compact_retains_statevectors_under_configured_caps",
            "gpu_execution_backend_added": False,
            "gpu_benchmark_records_emitted": False,
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
        "## Backend Metadata",
        "",
        "| Route | Backend | Model | Target | Output |",
        "| --- | --- | --- | --- | --- |",
    ]
    seen: set[str] = set()
    for row in summary["rows"]:
        route_id = str(row["route_id"])
        if route_id in seen:
            continue
        seen.add(route_id)
        lines.append(
            f"| {route_id} | {row['backend_family']} | {row['execution_model']} | "
            f"{row['contraction_execution_target']} | {row['output_kind']} |"
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
            "| Route | Total wall time s | Kernel time s | Planning time s | Lowering time s |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in summary["rows"]:
        lines.append(
            f"| {row['route_id']} | {row.get('total_wall_time_s')} | {row.get('kernel_time_s')} | "
            f"{row.get('planning_time_s')} | {row.get('lowering_time_s')} |"
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
