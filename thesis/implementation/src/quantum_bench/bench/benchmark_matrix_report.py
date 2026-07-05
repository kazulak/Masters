from __future__ import annotations

import csv
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml

from quantum_bench.bench.run_dirs import create_run_dir, sanitize
from quantum_bench.circuits import load_circuit, manifest
from quantum_bench.core.jsonio import write_json
from quantum_bench.core.records import JsonDict, TaskGraph, to_jsonable
from quantum_bench.environment import capture_environment
from quantum_bench.targets.upmem import (
    MEMORY_LEVEL_L1_WRAM,
    MEMORY_LEVEL_L2_SINGLE_DPU_MRAM,
    MEMORY_LEVEL_L3_MULTI_DPU,
    MEMORY_LEVEL_L4_OUT_OF_SCOPE,
    UpmemResourceModel,
    analyze_task_graph,
    build_synthetic_pressure_task_graph,
    is_synthetic_pressure_case,
    synthetic_pressure_manifest,
)
from quantum_bench.targets.upmem.external_libs import candidate_status_payload_from_report
from quantum_bench.tn import build_tensor_network, plan_task_graph_with_config, with_path_cost_summary
from quantum_bench.targets.upmem import annotate_task_graph_with_upmem_estimates


BENCHMARK_MATRIX_SCHEMA_VERSION = "benchmark_matrix_v1"
UPMEM_ROUTE_CATEGORY = "upmem_tn_runtime"
TOP_LEVEL_ROUTE_CATEGORIES = {
    "cpu_tn_exact",
    "gpu_tn_exact",
    "cpu_full_state",
    "gpu_full_state",
    UPMEM_ROUTE_CATEGORY,
}
UPMEM_INTERNAL_CLASSES = {
    MEMORY_LEVEL_L1_WRAM,
    MEMORY_LEVEL_L2_SINGLE_DPU_MRAM,
    MEMORY_LEVEL_L3_MULTI_DPU,
    MEMORY_LEVEL_L4_OUT_OF_SCOPE,
}

MATRIX_FIELDS = [
    "case_id",
    "workload_family",
    "workload_type",
    "n_qubits",
    "depth",
    "circuit_parameters",
    "gate_counts",
    "route_category",
    "route_id",
    "route_status",
    "execution_scope",
    "target",
    "validation_scope",
    "evidence_type",
    "output_authority",
    "validation_policy",
    "resource_model_id",
    "upmem_l1_task_count",
    "upmem_l2_task_count",
    "upmem_l3_task_count",
    "upmem_l4_task_count",
    "upmem_l1_implemented_task_count",
    "upmem_l2_modeled_task_count",
    "upmem_l3_modeled_task_count",
    "upmem_executable_task_count_current",
    "upmem_modeled_task_count",
    "max_frontier_width",
    "mean_frontier_width",
    "estimated_dpu_occupancy",
    "inter_task_parallelism_potential",
    "intra_task_parallelism_potential",
    "hybrid_parallelism_potential",
    "max_modeled_dpu_group_size",
    "estimated_transfer_bytes",
    "estimated_host_aggregation_bytes",
    "l1_l2_compute_backend_candidates",
    "l3_communication_backend_candidates",
    "simplepim_candidate_status",
    "pid_comm_candidate_status",
    "native_sdk_control_status",
    "recommended_next_backend_work",
]

PRESSURE_TASK_FIELDS = [
    "resource_model_id",
    "case_id",
    "workload_family",
    "workload_type",
    "task_index",
    "task_id",
    "memory_level",
    "current_backend_executable",
    "gemm_m",
    "gemm_k",
    "gemm_n",
    "full_task_bytes",
    "working_set_bytes",
    "estimated_host_to_dpu_bytes",
    "estimated_dpu_to_host_bytes",
    "estimated_mram_to_wram_bytes",
    "frontier_wave_index",
    "dominant_source",
]

PRESSURE_CASE_FIELDS = [
    "resource_model_id",
    "case_id",
    "workload_family",
    "workload_type",
    "n_qubits",
    "task_count",
    "wave_count",
    "memory_level_counts",
    "dominant_source_counts",
    "max_frontier_width",
    "mean_frontier_width",
    "mean_estimated_dpu_occupancy",
    "max_modeled_dpu_group_size",
    "total_estimated_transfer_bytes",
    "first_l2",
    "first_l3",
]

RESOURCE_MODEL_FIELDS = [
    "resource_model_id",
    "available_dpus",
    "effective_wram_bytes",
    "per_dpu_mram_bytes",
    "aggregate_mram_bytes",
    "case_count",
    "task_count",
    "memory_level_counts",
    "dominant_source_counts",
    "first_l2_l3_real_circuits",
    "first_l2_l3_synthetic_pressure",
]


def run_benchmark_matrix_report(
    root_dir: Path,
    matrix_path: Path,
    *,
    output_plots: bool = True,
    external_libs_report_path: Path | None = None,
) -> Path:
    matrix = load_benchmark_matrix(matrix_path)
    candidate_payload = _external_candidate_payload(external_libs_report_path)
    run_dir = create_run_dir(root_dir, "benchmark_matrix_report")
    write_json(run_dir / "environment.json", capture_environment(root_dir))
    (run_dir / "config" / "benchmark_matrix.yml").write_text(yaml.safe_dump(matrix, sort_keys=True), encoding="utf-8")
    if external_libs_report_path is not None:
        (run_dir / "config" / "external_pim_libraries_source.json").write_text(
            json.dumps({"path": str(external_libs_report_path)}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    route_categories = list(matrix["route_categories"])
    resource_models = _resource_models(matrix)
    workloads = list(matrix["workloads"])
    planner_config = dict(matrix.get("planner") or {"engine": "opt_einsum", "optimize": "greedy"})

    matrix_rows: list[JsonDict] = []
    pressure_task_rows: list[JsonDict] = []
    pressure_case_rows: list[JsonDict] = []
    for workload in workloads:
        graph, workload_manifest = _workload_graph(root_dir, workload, planner_config)
        workload_base = _workload_base(workload, workload_manifest)
        for route in route_categories:
            if route["route_category"] != UPMEM_ROUTE_CATEGORY:
                matrix_rows.append(_non_upmem_matrix_row(workload_base, route))
                continue
            for model_id, model in resource_models.items():
                analysis = analyze_task_graph(graph, model)
                case_row = _pressure_case_row(model_id, workload_base, analysis.summary())
                pressure_case_rows.append(case_row)
                pressure_task_rows.extend(_pressure_task_rows(model_id, workload_base, analysis))
                matrix_rows.append(_upmem_matrix_row(workload_base, route, model_id, analysis.summary(), analysis, candidate_payload))

    resource_rows = _resource_model_rows(resource_models, pressure_case_rows)
    summary = _summary(matrix_rows, pressure_case_rows, resource_rows)
    payload = {
        "schema_version": BENCHMARK_MATRIX_SCHEMA_VERSION,
        "run_id": run_dir.name,
        "matrix_id": matrix.get("matrix_id"),
        "summary": summary,
        "route_categories": route_categories,
        "upmem_runtime": matrix["upmem_runtime"],
        "resource_models": {key: model.as_dict() for key, model in resource_models.items()},
        "benchmark_matrix_rows": matrix_rows,
        "pim_pressure_case_rows": pressure_case_rows,
        "pim_pressure_task_rows": pressure_task_rows,
        "pim_pressure_resource_model_rows": resource_rows,
        "metadata": {
            "developer_only": True,
            "analysis_only": True,
            "providers_executed": False,
            "upmem_kernels_executed": False,
            "gpu_code_executed": False,
            "upmem_l1_l2_l3_are_internal_classes": True,
            "external_libs_report_loaded": external_libs_report_path is not None,
        },
    }
    write_json(run_dir / "benchmark_matrix.json", payload)
    _write_csv(run_dir / "benchmark_matrix.csv", matrix_rows, MATRIX_FIELDS)
    _write_csv(run_dir / "pim_pressure_tasks.csv", pressure_task_rows, PRESSURE_TASK_FIELDS)
    _write_csv(run_dir / "pim_pressure_cases.csv", pressure_case_rows, PRESSURE_CASE_FIELDS)
    _write_csv(run_dir / "pim_pressure_resource_models.csv", resource_rows, RESOURCE_MODEL_FIELDS)
    (run_dir / "benchmark_matrix_summary.md").write_text(_summary_markdown(summary, resource_rows), encoding="utf-8")
    if output_plots:
        _write_plots(run_dir, matrix_rows, pressure_case_rows, resource_rows)
    return run_dir


def load_benchmark_matrix(path: Path) -> JsonDict:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("benchmark matrix must be a YAML mapping")
    matrix = dict(data)
    validate_benchmark_matrix(matrix)
    return matrix


def validate_benchmark_matrix(matrix: JsonDict) -> None:
    routes = matrix.get("route_categories")
    if not isinstance(routes, list) or not routes:
        raise ValueError("benchmark matrix must define route_categories")
    categories = [str(route.get("route_category", "")) for route in routes if isinstance(route, dict)]
    if len(categories) != len(routes):
        raise ValueError("each route category entry must be a mapping with route_category")
    illegal = [category for category in categories if category in {"upmem_l1", "upmem_l2", "upmem_l3"}]
    if illegal:
        raise ValueError("L1/L2/L3 must be internal UPMEM classes, not top-level route categories")
    unknown = [category for category in categories if category not in TOP_LEVEL_ROUTE_CATEGORIES]
    if unknown:
        raise ValueError(f"unknown top-level route categories: {unknown}")
    if categories.count(UPMEM_ROUTE_CATEGORY) != 1:
        raise ValueError("benchmark matrix must contain exactly one upmem_tn_runtime route category")
    for route in routes:
        _validate_route_entry(route)

    upmem_runtime = matrix.get("upmem_runtime")
    if not isinstance(upmem_runtime, dict):
        raise ValueError("benchmark matrix must define upmem_runtime")
    internal = set(upmem_runtime.get("implemented_execution_classes", ())) | set(upmem_runtime.get("modeled_execution_classes", ())) | set(upmem_runtime.get("planned_execution_classes", ()))
    if not internal <= UPMEM_INTERNAL_CLASSES:
        raise ValueError("upmem_runtime execution classes must be L1/L2/L3/L4 internal classes")

    models = matrix.get("resource_models")
    if not isinstance(models, list) or not models:
        raise ValueError("benchmark matrix must define resource_models")
    for model in models:
        _resource_model_from_mapping(model)

    workloads = matrix.get("workloads")
    if not isinstance(workloads, list) or not workloads:
        raise ValueError("benchmark matrix must define workloads")
    for workload in workloads:
        _validate_workload(workload)


def _validate_route_entry(route: JsonDict) -> None:
    required = {
        "route_category",
        "route_id",
        "route_status",
        "execution_scope",
        "target",
        "validation_scope",
        "evidence_type",
        "output_authority",
        "validation_policy",
    }
    missing = sorted(required - set(route))
    if missing:
        raise ValueError(f"route category {route.get('route_category')} is missing fields: {missing}")
    if route["route_category"] == UPMEM_ROUTE_CATEGORY:
        if route["execution_scope"] == "full_circuit" or route["output_authority"] == "authoritative":
            raise ValueError("current upmem_tn_runtime rows must not claim full-circuit authority")
    if route["route_category"] == "cpu_full_state" and route["route_id"] == "quest_cpu_full_state_exact":
        if route["output_authority"] != "authoritative" or route["validation_policy"] != "full_exact":
            raise ValueError("QuEST CPU full-state category must record authoritative/full_exact semantics")
    if route["route_category"] == "gpu_tn_exact":
        if route["route_status"] != "planned" or route["evidence_type"] != "planned":
            raise ValueError("GPU TN matrix category must remain planned until a verified GPU TN route exists")
        if route["output_authority"] == "authoritative" or route["validation_policy"] == "full_exact":
            raise ValueError("planned GPU TN matrix category must not claim authoritative/full_exact semantics")
    if route["route_category"] == "gpu_full_state":
        if route["route_id"] != "quest_gpu_full_state_exact" or route["route_status"] != "implemented_optional":
            raise ValueError("GPU full-state matrix category must reference the optional verified QuEST GPU route")
        if route["evidence_type"] != "measured_when_verified" or route["output_authority"] != "authoritative_when_verified":
            raise ValueError("GPU full-state matrix category must be conditional on verified GPU execution")


def _validate_workload(workload: Any) -> None:
    if not isinstance(workload, dict) or not workload.get("id"):
        raise ValueError("benchmark matrix workloads must define id")
    circuit = workload.get("circuit")
    if not isinstance(circuit, dict) or not circuit.get("name"):
        raise ValueError(f"workload {workload.get('id')} must define circuit.name")
    if circuit.get("kind") == "synthetic_pressure":
        metadata = workload.get("metadata")
        if not isinstance(metadata, dict):
            raise ValueError("synthetic pressure workloads must define metadata")
        if metadata.get("workload_type") != "synthetic_pressure" or metadata.get("execution_scope") != "model_only":
            raise ValueError("synthetic pressure workloads must be model_only")
        if metadata.get("not_real_quantum_circuit") is not True:
            raise ValueError("synthetic pressure workloads must mark not_real_quantum_circuit true")


def _workload_graph(root_dir: Path, workload: JsonDict, planner_config: JsonDict) -> tuple[TaskGraph, JsonDict]:
    case = {key: value for key, value in workload.items() if key != "id"}
    case["case_id"] = workload["id"]
    case["workload_id"] = workload["id"]
    if is_synthetic_pressure_case(case):
        graph = build_synthetic_pressure_task_graph(case)
        return graph, synthetic_pressure_manifest(graph)
    circuit = load_circuit(case, root_dir)
    network = build_tensor_network(circuit)
    graph = plan_task_graph_with_config(network, planner_config)
    graph, _ = annotate_task_graph_with_upmem_estimates(graph)
    graph = with_path_cost_summary(graph)
    return graph, manifest(circuit)


def _workload_base(workload: JsonDict, workload_manifest: JsonDict) -> JsonDict:
    metadata = dict(workload.get("metadata") or {})
    source = dict(workload_manifest.get("source") or {})
    workload_type = str(metadata.get("workload_type") or ("synthetic_pressure" if source.get("kind") == "synthetic_pressure" else "real_circuit"))
    return {
        "case_id": str(workload["id"]),
        "workload_family": str(workload.get("workload_family") or workload.get("family") or workload_manifest.get("source", {}).get("name") or workload_manifest.get("name")),
        "workload_type": workload_type,
        "n_qubits": int(workload_manifest.get("n_qubits", 0) or 0),
        "depth": int(workload_manifest.get("depth_proxy", 0) or 0),
        "circuit_parameters": source,
        "gate_counts": workload_manifest.get("gate_counts", {}),
        "pressure_scale": metadata.get("pressure_scale", workload_manifest.get("n_qubits", 0)),
    }


def _non_upmem_matrix_row(base: JsonDict, route: JsonDict) -> JsonDict:
    return {
        **_empty_upmem_fields(),
        **_not_applicable_candidate_fields(),
        **base,
        "route_category": route["route_category"],
        "route_id": route["route_id"],
        "route_status": route["route_status"],
        "execution_scope": route["execution_scope"],
        "target": route["target"],
        "validation_scope": route["validation_scope"],
        "evidence_type": route["evidence_type"],
        "output_authority": route["output_authority"],
        "validation_policy": route["validation_policy"],
        "resource_model_id": "not_applicable",
    }


def _upmem_matrix_row(
    base: JsonDict,
    route: JsonDict,
    resource_model_id: str,
    summary: JsonDict,
    analysis: Any,
    candidate_payload: JsonDict,
) -> JsonDict:
    tasks = list(analysis.tasks)
    l1_executable = sum(1 for task in tasks if task.memory_level == MEMORY_LEVEL_L1_WRAM and task.current_backend_executable)
    dominant_sources = set(summary.get("dominant_source_counts", {}))
    return {
        **base,
        "route_category": route["route_category"],
        "route_id": route["route_id"],
        "route_status": route["route_status"],
        "execution_scope": route["execution_scope"],
        "target": route["target"],
        "validation_scope": route["validation_scope"],
        "evidence_type": route["evidence_type"],
        "output_authority": route["output_authority"],
        "validation_policy": route["validation_policy"],
        "resource_model_id": resource_model_id,
        "upmem_l1_task_count": int(summary.get("l1_task_count", 0) or 0),
        "upmem_l2_task_count": int(summary.get("l2_task_count", 0) or 0),
        "upmem_l3_task_count": int(summary.get("l3_task_count", 0) or 0),
        "upmem_l4_task_count": int(summary.get("l4_task_count", 0) or 0),
        "upmem_l1_implemented_task_count": l1_executable,
        "upmem_l2_modeled_task_count": int(summary.get("l2_task_count", 0) or 0),
        "upmem_l3_modeled_task_count": int(summary.get("l3_task_count", 0) or 0),
        "upmem_executable_task_count_current": l1_executable,
        "upmem_modeled_task_count": int(summary.get("task_count", 0) or 0),
        "max_frontier_width": int(summary.get("max_frontier_width", 0) or 0),
        "mean_frontier_width": float(summary.get("mean_frontier_width", 0.0) or 0.0),
        "estimated_dpu_occupancy": float(summary.get("mean_estimated_dpu_occupancy", 0.0) or 0.0),
        "inter_task_parallelism_potential": "inter_task" in dominant_sources or "hybrid" in dominant_sources,
        "intra_task_parallelism_potential": "intra_task" in dominant_sources or "hybrid" in dominant_sources,
        "hybrid_parallelism_potential": "hybrid" in dominant_sources,
        "max_modeled_dpu_group_size": int(summary.get("max_modeled_dpu_group_size", 0) or 0),
        "estimated_transfer_bytes": int(summary.get("total_estimated_transfer_bytes", 0) or 0),
        "estimated_host_aggregation_bytes": "unknown",
        **candidate_payload,
    }


def _empty_upmem_fields() -> JsonDict:
    return {
        "upmem_l1_task_count": 0,
        "upmem_l2_task_count": 0,
        "upmem_l3_task_count": 0,
        "upmem_l4_task_count": 0,
        "upmem_l1_implemented_task_count": 0,
        "upmem_l2_modeled_task_count": 0,
        "upmem_l3_modeled_task_count": 0,
        "upmem_executable_task_count_current": 0,
        "upmem_modeled_task_count": 0,
        "max_frontier_width": 0,
        "mean_frontier_width": 0.0,
        "estimated_dpu_occupancy": 0.0,
        "inter_task_parallelism_potential": False,
        "intra_task_parallelism_potential": False,
        "hybrid_parallelism_potential": False,
        "max_modeled_dpu_group_size": 0,
        "estimated_transfer_bytes": 0,
        "estimated_host_aggregation_bytes": "not_applicable",
    }


def _not_applicable_candidate_fields() -> JsonDict:
    return {
        "l1_l2_compute_backend_candidates": "not_applicable",
        "l3_communication_backend_candidates": "not_applicable",
        "simplepim_candidate_status": "not_applicable",
        "pid_comm_candidate_status": "not_applicable",
        "native_sdk_control_status": "not_applicable",
        "recommended_next_backend_work": "not_applicable",
    }


def _external_candidate_payload(external_libs_report_path: Path | None) -> JsonDict:
    if external_libs_report_path is None:
        return candidate_status_payload_from_report(None)
    payload = json.loads(external_libs_report_path.read_text(encoding="utf-8"))
    return candidate_status_payload_from_report(payload)


def _pressure_case_row(resource_model_id: str, base: JsonDict, summary: JsonDict) -> JsonDict:
    return {
        "resource_model_id": resource_model_id,
        "case_id": base["case_id"],
        "workload_family": base["workload_family"],
        "workload_type": base["workload_type"],
        "n_qubits": base["n_qubits"],
        "pressure_scale": base["pressure_scale"],
        "task_count": int(summary.get("task_count", 0) or 0),
        "wave_count": int(summary.get("wave_count", 0) or 0),
        "memory_level_counts": summary.get("memory_level_counts", {}),
        "dominant_source_counts": summary.get("dominant_source_counts", {}),
        "max_frontier_width": int(summary.get("max_frontier_width", 0) or 0),
        "mean_frontier_width": float(summary.get("mean_frontier_width", 0.0) or 0.0),
        "mean_estimated_dpu_occupancy": float(summary.get("mean_estimated_dpu_occupancy", 0.0) or 0.0),
        "max_modeled_dpu_group_size": int(summary.get("max_modeled_dpu_group_size", 0) or 0),
        "total_estimated_transfer_bytes": int(summary.get("total_estimated_transfer_bytes", 0) or 0),
        "first_l2": int(summary.get("l2_task_count", 0) or 0) > 0,
        "first_l3": int(summary.get("l3_task_count", 0) or 0) > 0,
    }


def _pressure_task_rows(resource_model_id: str, base: JsonDict, analysis: Any) -> list[JsonDict]:
    return [
        {
            "resource_model_id": resource_model_id,
            "case_id": base["case_id"],
            "workload_family": base["workload_family"],
            "workload_type": base["workload_type"],
            "task_index": task.task_index,
            "task_id": task.task_id,
            "memory_level": task.memory_level,
            "current_backend_executable": task.current_backend_executable,
            "gemm_m": task.gemm_m,
            "gemm_k": task.gemm_k,
            "gemm_n": task.gemm_n,
            "full_task_bytes": task.full_task_bytes,
            "working_set_bytes": task.working_set_bytes,
            "estimated_host_to_dpu_bytes": task.estimated_host_to_dpu_bytes,
            "estimated_dpu_to_host_bytes": task.estimated_dpu_to_host_bytes,
            "estimated_mram_to_wram_bytes": task.estimated_mram_to_wram_bytes,
            "frontier_wave_index": task.frontier_wave_index,
            "dominant_source": task.dominant_source,
        }
        for task in analysis.tasks
    ]


def _resource_model_rows(resource_models: dict[str, UpmemResourceModel], case_rows: list[JsonDict]) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for model_id, model in resource_models.items():
        model_cases = [row for row in case_rows if row["resource_model_id"] == model_id]
        memory_counts: Counter[str] = Counter()
        source_counts: Counter[str] = Counter()
        for case in model_cases:
            memory_counts.update({str(key): int(value) for key, value in dict(case["memory_level_counts"]).items()})
            source_counts.update({str(key): int(value) for key, value in dict(case["dominant_source_counts"]).items()})
        rows.append(
            {
                "resource_model_id": model_id,
                "available_dpus": model.available_dpus,
                "effective_wram_bytes": model.effective_wram_bytes,
                "per_dpu_mram_bytes": model.per_dpu_mram_bytes,
                "aggregate_mram_bytes": model.aggregate_mram_bytes,
                "case_count": len(model_cases),
                "task_count": sum(int(case["task_count"]) for case in model_cases),
                "memory_level_counts": dict(sorted(memory_counts.items())),
                "dominant_source_counts": dict(sorted(source_counts.items())),
                "first_l2_l3_real_circuits": _first_l2_l3(model_cases, "real_circuit"),
                "first_l2_l3_synthetic_pressure": _first_l2_l3(model_cases, "synthetic_pressure"),
            }
        )
    return rows


def _first_l2_l3(case_rows: list[JsonDict], workload_type: str) -> JsonDict:
    by_family: dict[str, list[JsonDict]] = defaultdict(list)
    for row in case_rows:
        if row["workload_type"] == workload_type:
            by_family[str(row["workload_family"])].append(row)
    result: JsonDict = {}
    for family, rows in sorted(by_family.items()):
        ordered = sorted(rows, key=lambda item: (float(item.get("pressure_scale", item.get("n_qubits", 0)) or 0), str(item["case_id"])))
        first_l2 = next((row for row in ordered if row.get("first_l2")), None)
        first_l3 = next((row for row in ordered if row.get("first_l3")), None)
        result[family] = {
            "first_l2_case_id": first_l2["case_id"] if first_l2 else None,
            "first_l3_case_id": first_l3["case_id"] if first_l3 else None,
        }
    return result


def _summary(matrix_rows: list[JsonDict], pressure_case_rows: list[JsonDict], resource_rows: list[JsonDict]) -> JsonDict:
    route_status_counts = Counter(str(row["route_status"]) for row in matrix_rows)
    execution_scope_counts = Counter(str(row["execution_scope"]) for row in matrix_rows)
    categories = sorted({str(row["route_category"]) for row in matrix_rows})
    upmem_categories = sorted({category for category in categories if category.startswith("upmem")})
    return {
        "matrix_row_count": len(matrix_rows),
        "pressure_case_row_count": len(pressure_case_rows),
        "resource_model_count": len(resource_rows),
        "route_categories": categories,
        "upmem_top_level_categories": upmem_categories,
        "route_status_counts": dict(sorted(route_status_counts.items())),
        "execution_scope_counts": dict(sorted(execution_scope_counts.items())),
        "upmem_is_unified_runtime": upmem_categories == [UPMEM_ROUTE_CATEGORY],
        "current_upmem_full_circuit_speedup_available": False,
    }


def _summary_markdown(summary: JsonDict, resource_rows: list[JsonDict]) -> str:
    lines = [
        "# Benchmark Matrix Report",
        "",
        "This report is a scaffold for final thesis evaluation. It does not execute providers, GPU code, UPMEM kernels, or SimplePIM commands.",
        "",
        "## Matrix Scope",
        "",
        f"- Matrix rows: {summary['matrix_row_count']}",
        f"- Route categories: {', '.join(summary['route_categories'])}",
        f"- UPMEM top-level category: {', '.join(summary['upmem_top_level_categories'])}",
        f"- UPMEM full-circuit speedup available now: {summary['current_upmem_full_circuit_speedup_available']}",
        "",
        "UPMEM is one unified runtime in the final comparison. L1/L2/L3 are internal scheduler-selected execution classes, not separate top-level competitors.",
        "Current UPMEM evidence is limited to L1 task-level simulator dense bridge subset evidence plus L2/L3 model-only pressure evidence.",
        "Current UPMEM task timings must not be reported as full-circuit speedup.",
        "",
        "## Resource Models",
        "",
        "| Resource model | Tasks | Memory levels | Dominant sources |",
        "|---|---:|---|---|",
    ]
    for row in resource_rows:
        lines.append(
            "| {id} | {tasks} | {levels} | {sources} |".format(
                id=row["resource_model_id"],
                tasks=row["task_count"],
                levels=_counts_text(row["memory_level_counts"]),
                sources=_counts_text(row["dominant_source_counts"]),
            )
        )
    lines.extend(
        [
            "",
            "## Next Backend Work Suggested",
            "",
            "Use the internal UPMEM class distribution to choose between L2 tiling, L3 distributed GEMM, hardware execution, SimplePIM GEMM feasibility, SparseP, or route-aware path selection.",
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


def _write_plots(run_dir: Path, matrix_rows: list[JsonDict], pressure_case_rows: list[JsonDict], resource_rows: list[JsonDict]) -> None:
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(plots_dir / ".matplotlib"))
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    _simple_count_plot(plt, plots_dir / "route_status_matrix.png", Counter(str(row["route_status"]) for row in matrix_rows), "Route Status Matrix")
    _simple_count_plot(plt, plots_dir / "execution_scope_matrix.png", Counter(str(row["execution_scope"]) for row in matrix_rows), "Execution Scope Matrix")
    _stacked_resource_plot(plt, plots_dir / "upmem_internal_classes_by_resource_model.png", resource_rows, "memory_level_counts", "UPMEM Internal Classes By Resource Model")
    _stacked_case_plot(plt, plots_dir / "upmem_internal_memory_levels_by_case.png", pressure_case_rows, "memory_level_counts", "UPMEM Internal Memory Levels By Case")
    _stacked_case_plot(plt, plots_dir / "inter_vs_intra_pressure_by_case.png", pressure_case_rows, "dominant_source_counts", "Inter Vs Intra Pressure By Case")
    _first_l2_l3_plot(plt, plots_dir / "first_l2_l3_by_family.png", resource_rows)


def _simple_count_plot(plt: Any, path: Path, counts: Counter[str], title: str) -> None:
    labels = sorted(counts)
    figure, axis = plt.subplots(figsize=(max(5.0, len(labels) * 0.75), 4.0))
    axis.bar(labels, [counts[label] for label in labels], color="#2563eb")
    axis.set_title(title)
    axis.set_ylabel("Rows")
    axis.tick_params(axis="x", rotation=35)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _stacked_resource_plot(plt: Any, path: Path, rows: list[JsonDict], key: str, title: str) -> None:
    labels = [str(row["resource_model_id"]) for row in rows]
    _stacked_plot(plt, path, labels, [dict(row.get(key) or {}) for row in rows], title)


def _stacked_case_plot(plt: Any, path: Path, rows: list[JsonDict], key: str, title: str) -> None:
    labels = [f"{row['case_id']}:{row['resource_model_id']}" for row in rows]
    _stacked_plot(plt, path, labels, [dict(row.get(key) or {}) for row in rows], title)


def _stacked_plot(plt: Any, path: Path, labels: list[str], count_rows: list[dict[str, int]], title: str) -> None:
    categories = sorted({category for counts in count_rows for category in counts})
    figure, axis = plt.subplots(figsize=(max(6.0, len(labels) * 0.55), 4.5))
    bottoms = [0] * len(labels)
    for category in categories:
        values = [int(counts.get(category, 0)) for counts in count_rows]
        axis.bar(labels, values, bottom=bottoms, label=category)
        bottoms = [bottom + value for bottom, value in zip(bottoms, values)]
    axis.set_title(title)
    axis.set_ylabel("Count")
    axis.tick_params(axis="x", rotation=45)
    if categories:
        axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _first_l2_l3_plot(plt: Any, path: Path, resource_rows: list[JsonDict]) -> None:
    labels = [str(row["resource_model_id"]) for row in resource_rows]
    values = [len(dict(row.get("first_l2_l3_synthetic_pressure") or {})) for row in resource_rows]
    _simple_count_plot(plt, path, Counter(dict(zip(labels, values))), "First L2/L3 Synthetic Families By Resource Model")


def _resource_models(matrix: JsonDict) -> dict[str, UpmemResourceModel]:
    return {
        str(item["id"]): _resource_model_from_mapping(item)
        for item in matrix["resource_models"]
    }


def _resource_model_from_mapping(item: Any) -> UpmemResourceModel:
    if not isinstance(item, dict) or not item.get("id"):
        raise ValueError("resource model entries must define id")
    return UpmemResourceModel(
        available_dpus=int(item.get("available_dpus", 64)),
        per_dpu_wram_bytes=int(item.get("per_dpu_wram_bytes", 64 * 1024)),
        effective_wram_bytes=int(item.get("effective_wram_bytes", 60 * 1024)),
        per_dpu_mram_bytes=int(item.get("per_dpu_mram_bytes", 64 * 1024 * 1024)),
        max_task_group_dpus=int(item.get("max_task_group_dpus", item.get("available_dpus", 64))),
    )


def _counts_text(value: Any) -> str:
    counts = dict(value or {})
    if not counts:
        return "none"
    return ", ".join(f"{key}: {counts[key]}" for key in sorted(counts))


def _csv_value(value: Any) -> Any:
    value = to_jsonable(value)
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    if value is None:
        return ""
    return value
