from __future__ import annotations

import csv
import json
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
from quantum_bench.circuits import load_circuit, manifest
from quantum_bench.core.jsonio import write_json, write_jsonl
from quantum_bench.core.records import BenchmarkContext, JsonDict, to_jsonable
from quantum_bench.environment import capture_environment
from quantum_bench.providers import route_registry
from quantum_bench.targets.upmem import SYNTHETIC_PRESSURE_ERROR, is_synthetic_pressure_case
from quantum_bench.tn import build_tensor_network, execute_task_sequence_np_einsum, plan_task_graph_with_config, with_path_cost_summary
from quantum_bench.validation import probability_error_metrics, tensor_to_quest_statevector, validate, validation_result_to_dict


SIMULATION_BACKEND_COMPARE_SCHEMA_VERSION = "simulation_backend_compare_v1"

RESULT_FIELDS = [
    "case_id",
    "workload_id",
    "route_id",
    "backend_family",
    "execution_model",
    "contraction_execution_target",
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
    "total_wall_time_s",
    "kernel_time_s",
    "max_abs_error",
    "l2_error",
    "norm_drift",
    "probability_l1_error",
    "probability_max_abs_error",
    "statevector_artifact",
]


@dataclass(frozen=True)
class SimulationBackendCompareResult:
    run_dir: Path
    summary_path: Path
    status: str
    case_count: int


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
    pair_rows: list[JsonDict] = []
    normalized_records: list[JsonDict] = []
    for case_payload in suite["cases"]:
        case_result = _run_case(root_dir, run_dir, suite, case_payload)
        rows.extend(case_result["rows"])
        pair_rows.append(case_result["pair"])
        normalized_records.extend(case_result["normalized_records"])

    write_jsonl(run_dir / "simulation_backend_compare_cases.jsonl", pair_rows)
    _write_csv(run_dir / "simulation_backend_compare_results.csv", rows, RESULT_FIELDS)
    _write_csv(run_dir / "simulation_backend_compare_pairs.csv", pair_rows, _pair_fields(pair_rows))
    summary = _summary_payload(suite=suite, suite_path=suite_path, rows=rows, pair_rows=pair_rows, normalized_records=normalized_records)
    write_json(run_dir / "simulation_backend_compare_summary.json", summary)
    (run_dir / "comparison_summary.md").write_text(_summary_markdown(summary, pair_rows), encoding="utf-8")
    write_normalized_records(run_dir, normalized_records)
    report_run(run_dir, output_plots=True)
    if artifact_retention == "compact":
        prune_run(run_dir, artifact_retention="compact")
    return SimulationBackendCompareResult(
        run_dir=run_dir,
        summary_path=run_dir / "simulation_backend_compare_summary.json",
        status="completed",
        case_count=len(pair_rows),
    )


def _validate_suite_routes(suite: JsonDict) -> None:
    routes = set(suite.get("route_policy", {}).get("routes") or ())
    required = {"cpu_tn_einsum_exact", "quest_cpu_full_state_exact"}
    missing = sorted(required - routes)
    if missing:
        raise ValueError(f"simulation backend comparison suite must include routes: {', '.join(missing)}")


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

    tn_start = time.perf_counter()
    tn_output, tn_metadata = execute_task_sequence_np_einsum(graph, network)
    tn_time_s = time.perf_counter() - tn_start
    tn_state = tensor_to_quest_statevector(tn_output)

    tn_tensor_rel = Path("cases") / sanitize(case_id) / "cpu_tn_final_tensor.npy"
    tn_state_rel = Path("cases") / sanitize(case_id) / "cpu_tn_statevector_quest_order.npy"
    np.save(run_dir / tn_tensor_rel, tn_output, allow_pickle=False)
    np.save(run_dir / tn_state_rel, tn_state, allow_pickle=False)

    quest_result = _run_quest_route(root_dir, run_dir, suite, case_payload, graph, network)
    if quest_result.output.array is None or quest_result.status != "passed":
        raise RuntimeError(quest_result.error or "quest_cpu_full_state_exact failed")
    quest_state = np.asarray(quest_result.output.array, dtype=np.complex128)
    quest_state_rel = Path("cases") / sanitize(case_id) / "quest_statevector.npy"
    np.save(run_dir / quest_state_rel, quest_state, allow_pickle=False)

    validation = validate(quest_state, tn_state, suite["tolerances"])
    validation_metrics = validation_result_to_dict(validation)
    validation_metrics.update(probability_error_metrics(quest_state, tn_state))
    validation_metrics["error_direction"] = "quest_minus_cpu_tn_statevector"

    circuit_meta = manifest(circuit)
    one_two = circuit_meta["gate_counts"]
    common = {
        "case_id": case_id,
        "workload_id": str(case_payload.get("workload_id", case_id)),
        "suite_id": suite["suite_id"],
        "n_qubits": circuit.n_qubits,
        "gate_count": int(one_two["total"]),
        "two_qubit_gate_count": int(one_two["2q"]),
        "tn_task_count": len(graph.tasks),
        "tn_max_intermediate_bytes": int(graph.path_summary.max_intermediate_bytes),
        "tn_estimated_flops": int(graph.path_summary.total_estimated_flops),
        "tn_estimated_bytes": int(sum(task.estimated_bytes for task in graph.tasks)),
        "validation_metrics": validation_metrics,
        "validation_status": "passed" if validation.passed else "failed",
    }
    tn_row = {
        **common,
        "route_id": "cpu_tn_einsum_exact",
        "backend_family": "cpu",
        "execution_model": "tensor_network",
        "contraction_execution_target": "cpu",
        "execution_scope": "full_taskgraph",
        "output_kind": "final_tensor",
        "comparison_output_kind": "statevector_from_final_tensor",
        "status": "completed" if validation.passed else "validation_failed",
        "error_direction": "quest_minus_cpu_tn_statevector",
        "statevector_bytes": int(tn_state.nbytes),
        "total_wall_time_s": float(tn_time_s),
        "kernel_time_s": float(tn_time_s),
        "statevector_artifact": artifact_ref(run_dir, tn_state_rel, role="cpu_tn_statevector"),
        "final_tensor_artifact": artifact_ref(run_dir, tn_tensor_rel, role="cpu_tn_final_tensor"),
    }
    quest_row = {
        **common,
        "route_id": "quest_cpu_full_state_exact",
        "backend_family": "quest",
        "execution_model": "full_state",
        "contraction_execution_target": "cpu",
        "execution_scope": "full_circuit",
        "output_kind": "statevector",
        "comparison_output_kind": "statevector",
        "status": "completed" if validation.passed else "validation_failed",
        "error_direction": "quest_minus_cpu_tn_statevector",
        "statevector_bytes": int(quest_state.nbytes),
        "total_wall_time_s": float(quest_result.profile.total_s),
        "kernel_time_s": float(quest_result.profile.kernel_s),
        "statevector_artifact": artifact_ref(run_dir, quest_state_rel, role="quest_statevector"),
        "quest_state_dump_artifact": quest_result.output.metadata.get("state_dump_artifact"),
    }
    pair = {
        **common,
        "schema_version": SIMULATION_BACKEND_COMPARE_SCHEMA_VERSION,
        "quest_route_id": "quest_cpu_full_state_exact",
        "tn_route_id": "cpu_tn_einsum_exact",
        "quest_statevector_artifact": quest_row["statevector_artifact"],
        "cpu_tn_statevector_artifact": tn_row["statevector_artifact"],
        "cpu_tn_final_tensor_artifact": tn_row["final_tensor_artifact"],
        "basis_order": "quest_little_endian_integer_index",
        "tn_execution_metadata": _metadata_without_task_metrics(tn_metadata),
        "quest_metadata": quest_result.metadata,
    }
    rows = [_row_with_metrics(tn_row), _row_with_metrics(quest_row)]
    write_json(case_dir / "simulation_backend_compare.json", pair)
    return {
        "rows": rows,
        "pair": to_jsonable(pair),
        "normalized_records": [_normalized_record(run_dir, row, case_id=case_id) for row in rows],
    }


def _run_quest_route(root_dir: Path, run_dir: Path, suite: JsonDict, case_payload: JsonDict, graph: Any, network: Any) -> Any:
    routes = route_registry(root_dir)
    route = routes["quest_cpu_full_state_exact"]
    route_config = route_config_for(suite, "quest_cpu_full_state_exact")
    context = BenchmarkContext(
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
    can_execute, reason = route.can_execute(graph, context)
    if not can_execute:
        raise RuntimeError(reason or "quest_cpu_full_state_exact cannot execute")
    return route.execute(route.prepare(graph, network, context), context)


def _row_with_metrics(row: JsonDict) -> JsonDict:
    metrics = dict(row.get("validation_metrics") or {})
    row = dict(row)
    row["max_abs_error"] = metrics.get("max_abs_error")
    row["l2_error"] = metrics.get("l2_error")
    row["norm_drift"] = metrics.get("norm_drift")
    row["probability_l1_error"] = metrics.get("probability_l1_error")
    row["probability_max_abs_error"] = metrics.get("probability_max_abs_error")
    return to_jsonable(row)


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
        "kernel_family": "full_state_vector" if row.get("execution_model") == "full_state" else "einsum_contraction",
        "execution_model": row.get("execution_model"),
        "execution_target": "cpu",
        "contraction_execution_target": "cpu",
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
                "error_direction": row.get("error_direction"),
                "n_qubits": row.get("n_qubits"),
                "gate_count": row.get("gate_count"),
                "two_qubit_gate_count": row.get("two_qubit_gate_count"),
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        "warnings": "",
    }


def _metadata_without_task_metrics(metadata: JsonDict) -> JsonDict:
    payload = dict(metadata)
    payload.pop("task_metrics", None)
    return to_jsonable(payload)


def _summary_payload(
    *,
    suite: JsonDict,
    suite_path: Path,
    rows: list[JsonDict],
    pair_rows: list[JsonDict],
    normalized_records: list[JsonDict],
) -> JsonDict:
    return to_jsonable(
        {
            "schema_version": SIMULATION_BACKEND_COMPARE_SCHEMA_VERSION,
            "suite_id": suite["suite_id"],
            "suite_path": str(suite_path),
            "case_count": len(pair_rows),
            "record_count": len(rows),
            "passed_case_count": sum(1 for row in pair_rows if row["validation_status"] == "passed"),
            "failed_case_count": sum(1 for row in pair_rows if row["validation_status"] != "passed"),
            "routes": ["quest_cpu_full_state_exact", "cpu_tn_einsum_exact"],
            "execution_models": ["full_state", "tensor_network"],
            "root_normalized_records_are_canonical": True,
            "normalized_records_artifact": "normalized_records.jsonl",
            "quest_metrics_only_route_is_not_output_comparable": True,
            "statevector_retention_policy": "compact_retains_statevectors_under_configured_caps",
            "rows": rows,
            "pairs": pair_rows,
            "normalized_records": normalized_records,
        }
    )


def _summary_markdown(summary: JsonDict, pair_rows: list[JsonDict]) -> str:
    lines = [
        "# Simulation Backend Comparison",
        "",
        f"Suite: `{summary['suite_id']}`",
        f"Cases: {summary['case_count']}",
        "",
        "QuEST CPU full-state and CPU tensor-network outputs are compared against each other on identical deterministic unitary circuits.",
        "The error direction is `quest_minus_cpu_tn_statevector`; CPU TN is not described as an unquestioned authority in this report.",
        "",
        "| Case | Status | Max abs error | L2 error | Probability L1 error | TN tasks |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in pair_rows:
        metrics = dict(row.get("validation_metrics") or {})
        lines.append(
            f"| {row['case_id']} | {row['validation_status']} | {metrics.get('max_abs_error')} | "
            f"{metrics.get('l2_error')} | {metrics.get('probability_l1_error')} | {row.get('tn_task_count')} |"
        )
    lines.append("")
    return "\n".join(lines)


def _pair_fields(rows: list[JsonDict]) -> list[str]:
    if not rows:
        return ["case_id"]
    fields = set()
    for row in rows:
        fields.update(row)
    preferred = [
        "schema_version",
        "case_id",
        "workload_id",
        "validation_status",
        "n_qubits",
        "gate_count",
        "two_qubit_gate_count",
        "tn_task_count",
        "tn_max_intermediate_bytes",
        "tn_estimated_flops",
        "tn_estimated_bytes",
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
