from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import yaml

from quantum_bench.bench.config import comparison_planner_configs, comparison_scoring_weights, load_suite
from quantum_bench.bench.planner_scoring import csv_value, divergence_summary, markdown_summary, score_planner_rows, scoring_metadata
from quantum_bench.bench.reporting import prune_run, write_run_manifest
from quantum_bench.bench.run_dirs import EVIDENCE_ARTIFACT_KIND, create_run_dir, sanitize
from quantum_bench.circuits import load_circuit
from quantum_bench.core.jsonio import write_json, write_jsonl
from quantum_bench.core.records import TaskGraph
from quantum_bench.environment import capture_environment
from quantum_bench.targets.upmem import (
    UPMEM_DENSE_ESTIMATE_KEY,
    annotate_task_graph_with_upmem_estimates,
    upmem_task_estimate_rows,
)
from quantum_bench.tn import (
    build_execution_bundle,
    build_tensor_network,
    executor_config_hash,
    plan_task_graph_with_config,
    with_path_cost_summary,
)


COMPARISON_SCHEMA_VERSION = "planner_comparison_v2"

COMPARISON_FIELDS = [
    "case_id",
    "workload_id",
    "circuit_family",
    "n_qubits",
    "depth",
    "planner_engine",
    "planner_id",
    "planner_kind",
    "optimize_mode",
    "objective",
    "cost_basis",
    "target_estimate_key",
    "task_count",
    "total_estimated_flops",
    "peak_intermediate_bytes",
    "total_host_to_dpu_bytes",
    "total_dpu_to_host_bytes",
    "total_mram_to_wram_bytes",
    "unsupported_task_count",
    "tiling_required_task_count",
    "missing_target_estimate_count",
    "estimated_total_tile_count",
    "estimated_max_parallel_tiles",
    "circuit_semantics_hash",
    "tensor_network_hash",
    "contraction_plan_hash",
    "planning_time_s",
    "execution_bundle_artifact",
    "task_graph_artifact",
    "path_summary_artifact",
    "target_estimates_artifact",
    "score_model",
    "upmem_pressure_score",
    "upmem_rank",
    "flop_rank",
    "score_components",
    "score_weights",
    "tradeoff_note",
]


def compare_planners(suite_path: Path, root_dir: Path) -> Path:
    suite = load_suite(suite_path)
    planner_configs = comparison_planner_configs(suite)
    scoring_weights = comparison_scoring_weights(suite)
    run_suite_id = _comparison_run_suite_id(str(suite["suite_id"]))
    run_dir = create_run_dir(
        root_dir,
        run_suite_id,
        artifact_kind=EVIDENCE_ARTIFACT_KIND,
        route_label="planner_comparison",
    )
    (run_dir / "config" / "resolved_suite.yml").write_text(yaml.safe_dump(suite, sort_keys=True), encoding="utf-8")
    write_json(run_dir / "environment.json", capture_environment(root_dir))

    rows: list[dict[str, Any]] = []
    for case in suite["cases"]:
        rows.extend(_compare_case(case, str(suite["suite_id"]), planner_configs, root_dir, run_dir))
    rows = score_planner_rows(rows, scoring_weights)

    payload = {
        "schema_version": COMPARISON_SCHEMA_VERSION,
        "suite_id": suite["suite_id"],
        "run_id": run_dir.name,
        "planner_configs": planner_configs,
        "scoring": scoring_metadata(scoring_weights),
        "divergence_summary": divergence_summary(rows),
        "rows": rows,
    }
    write_json(run_dir / "planner_comparison.json", payload)
    _write_comparison_csv(run_dir / "planner_comparison.csv", rows)
    (run_dir / "planner_comparison_summary.md").write_text(markdown_summary(rows), encoding="utf-8")
    write_jsonl(
        run_dir / "normalized_records.jsonl",
        [_normalized_planner_record(payload, row) for row in rows],
    )
    write_run_manifest(
        run_dir,
        run_kind="planner_comparison",
        suite_id=run_suite_id,
        suite_path=str(suite_path),
        artifact_kind=EVIDENCE_ARTIFACT_KIND,
        route_label="planner_comparison",
        route_id="planner_candidate_model",
        backend_id="opt_einsum",
        execution_scope="contraction_planning",
        evidence_type="modeled_planner_comparison",
        normalized_records="normalized_records.jsonl",
        summary="planner_comparison.json",
        artifact_retention="full",
        root_dir=root_dir,
    )
    prune_run(run_dir, artifact_retention="full")
    return run_dir


def _compare_case(
    case: dict[str, Any],
    suite_id: str,
    planner_configs: list[dict[str, Any]],
    root_dir: Path,
    run_dir: Path,
) -> list[dict[str, Any]]:
    circuit = load_circuit(case, root_dir)
    network = build_tensor_network(circuit)
    case_id = str(case["case_id"])
    workload_id = str(case.get("workload_id", case_id))
    used_planner_dirs: dict[str, int] = {}
    rows: list[dict[str, Any]] = []

    for planner_config in planner_configs:
        graph = plan_task_graph_with_config(network, planner_config)
        graph, _ = annotate_task_graph_with_upmem_estimates(graph)
        graph = with_path_cost_summary(graph)
        planner_dir_name = _unique_planner_dir_name(graph.path_summary.planner_id, used_planner_dirs)
        artifacts = _write_planner_artifacts(
            run_dir,
            suite_id,
            case_id,
            planner_dir_name,
            graph,
        )
        rows.append(_comparison_row(case, circuit.name, circuit.n_qubits, len(circuit.operations), workload_id, graph, artifacts))

    return rows


def _write_planner_artifacts(
    run_dir: Path,
    suite_id: str,
    case_id: str,
    planner_dir_name: str,
    graph: TaskGraph,
) -> dict[str, str]:
    planner_root = Path("cases") / case_id / "planners" / planner_dir_name
    task_graph_artifact = planner_root / "task_graph.json"
    path_summary_artifact = planner_root / "path_summary.json"
    target_estimates_artifact = planner_root / "target_estimates" / f"{UPMEM_DENSE_ESTIMATE_KEY}.jsonl"
    execution_bundle_artifact = planner_root / "execution_bundle.json"
    write_json(run_dir / task_graph_artifact, graph)
    write_json(run_dir / path_summary_artifact, graph.path_summary)
    write_jsonl(run_dir / target_estimates_artifact, upmem_task_estimate_rows(graph))
    write_json(
        run_dir / execution_bundle_artifact,
        build_execution_bundle(graph, case_id=case_id, suite_id=suite_id),
    )
    return {
        "task_graph_artifact": task_graph_artifact.as_posix(),
        "path_summary_artifact": path_summary_artifact.as_posix(),
        "target_estimates_artifact": target_estimates_artifact.as_posix(),
        "execution_bundle_artifact": execution_bundle_artifact.as_posix(),
    }


def _comparison_row(
    case: dict[str, Any],
    circuit_name: str,
    n_qubits: int,
    depth: int,
    workload_id: str,
    graph: TaskGraph,
    artifacts: dict[str, str],
) -> dict[str, Any]:
    summary = graph.path_summary
    return {
        "case_id": str(case["case_id"]),
        "workload_id": workload_id,
        "circuit_family": str(case.get("circuit", {}).get("name", circuit_name)),
        "n_qubits": n_qubits,
        "depth": depth,
        "planner_engine": summary.planner_engine,
        "planner_id": summary.planner_id,
        "planner_kind": summary.planner_kind,
        "optimize_mode": summary.optimize_mode,
        "objective": summary.objective,
        "cost_basis": summary.cost_basis,
        "target_estimate_key": summary.target_estimate_key,
        "task_count": summary.task_count,
        "total_estimated_flops": summary.total_estimated_flops,
        "peak_intermediate_bytes": summary.peak_intermediate_bytes,
        "total_host_to_dpu_bytes": summary.total_host_to_dpu_bytes,
        "total_dpu_to_host_bytes": summary.total_dpu_to_host_bytes,
        "total_mram_to_wram_bytes": summary.total_mram_to_wram_bytes,
        "unsupported_task_count": summary.unsupported_task_count,
        "tiling_required_task_count": summary.tiling_required_task_count,
        "missing_target_estimate_count": summary.missing_target_estimate_count,
        "estimated_total_tile_count": summary.estimated_total_tile_count,
        "estimated_max_parallel_tiles": summary.estimated_max_parallel_tiles,
        "circuit_semantics_hash": graph.circuit_semantics_hash,
        "tensor_network_hash": graph.tensor_network_hash,
        "contraction_plan_hash": graph.contraction_plan_hash,
        "planning_time_s": graph.planning_time_s,
        **artifacts,
    }


def _normalized_planner_record(payload: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "benchmark_result_artifact_v1",
        "source_artifact": "planner_comparison.json",
        "run_id": payload["run_id"],
        "suite_id": _comparison_run_suite_id(str(payload["suite_id"])),
        "case_id": row["case_id"],
        "workload_id": row["workload_id"],
        "n_qubits": row["n_qubits"],
        "route_id": "planner_candidate_model",
        "backend_id": row["planner_id"],
        "backend_family": "opt_einsum",
        "benchmark_role": "contraction_path_candidate",
        "kernel_family": "cpu_reference_only",
        "execution_model": "tensor_network_planning",
        "parallelism_mode": "not_executed",
        "parallelism_evidence_type": "modeled",
        "execution_plan_kind": "taskgraph_contraction_path",
        "execution_plan_executed": False,
        "circuit_semantics_hash": row["circuit_semantics_hash"],
        "tensor_network_hash": row["tensor_network_hash"],
        "contraction_plan_hash": row["contraction_plan_hash"],
        "plan_reused": False,
        "planning_in_timed_region": False,
        "executor_config_hash": executor_config_hash("planner_candidate_model", {"planner_id": row["planner_id"]}),
        "execution_bundle_artifact": row["execution_bundle_artifact"],
        "contraction_execution_target": "modeled",
        "accelerator_kind": "none",
        "execution_scope": "contraction_planning",
        "status": "completed",
        "validation_status": "not_applicable",
        "task_count": row["task_count"],
        "planning_time_s": row["planning_time_s"],
        "planner_engine": row["planner_engine"],
        "planner_id": row["planner_id"],
        "planner_kind": row["planner_kind"],
        "optimize_mode": row["optimize_mode"],
        "objective": row["objective"],
        "cost_basis": row["cost_basis"],
        "tn_estimated_flops": row["total_estimated_flops"],
        "tn_max_intermediate_bytes": row["peak_intermediate_bytes"],
        "total_host_to_dpu_bytes": row["total_host_to_dpu_bytes"],
        "total_dpu_to_host_bytes": row["total_dpu_to_host_bytes"],
        "total_mram_to_wram_bytes": row["total_mram_to_wram_bytes"],
        "unsupported_task_count": row["unsupported_task_count"],
        "tiling_required_task_count": row["tiling_required_task_count"],
        "estimated_total_tile_count": row["estimated_total_tile_count"],
        "estimated_max_parallel_tiles": row["estimated_max_parallel_tiles"],
        "score_model": row.get("score_model"),
        "upmem_pressure_score": row.get("upmem_pressure_score"),
        "upmem_rank": row.get("upmem_rank"),
        "flop_rank": row.get("flop_rank"),
        "hardware_execution": False,
        "hardware_speedup_applicable": False,
        "cpu_fallback_used": False,
    }


def _unique_planner_dir_name(planner_id: str, used: dict[str, int]) -> str:
    base = sanitize(planner_id)
    count = used.get(base, 0)
    used[base] = count + 1
    if count == 0:
        return base
    return f"{base}_{count + 1:02d}"


def _comparison_run_suite_id(suite_id: str) -> str:
    if suite_id == "planner_compare" or suite_id.endswith("_planner_compare"):
        return suite_id
    return f"{suite_id}_planner_compare"


def _write_comparison_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COMPARISON_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: csv_value(row.get(field)) for field in COMPARISON_FIELDS})
