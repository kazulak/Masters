from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

import yaml

from quantum_bench.bench.config import comparison_pim_objective_config, comparison_planner_configs, comparison_scoring_weights, load_suite
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
    contraction_path_structure_hash,
    executor_config_hash,
    plan_task_graph_with_config,
    with_path_cost_summary,
)
from quantum_bench.tn.planner_motifs import build_planner_motif_workload, is_planner_motif_case
from quantum_bench.tn.upmem_path_cost import (
    DEFAULT_UPMEM_PATH_COST_NORMALIZATION_ID,
    PathCostComponents,
    model_upmem_network_path_cost,
    upmem_path_cost_policy,
    upmem_path_cost_profile,
)
from quantum_bench.tn.upmem_planner import PlannerInfeasibleError, UPMEM_PATH_OBJECTIVE_VERSION


COMPARISON_SCHEMA_VERSION = "planner_comparison_v3"

COMPARISON_FIELDS = [
    "case_id",
    "workload_id",
    "circuit_family",
    "n_qubits",
    "depth",
    "workload_kind",
    "not_real_quantum_circuit",
    "planner_motif",
    "network_tensor_count",
    "network_index_count",
    "network_max_rank",
    "network_max_tensor_elements",
    "network_size_proxy",
    "planner_engine",
    "planner_id",
    "planner_kind",
    "optimize_mode",
    "objective",
    "cost_basis",
    "target_estimate_key",
    "candidate_status",
    "candidate_failure_reason",
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
    "contraction_path_structure_hash",
    "planning_time_s",
    "execution_bundle_artifact",
    "task_graph_artifact",
    "path_summary_artifact",
    "target_estimates_artifact",
    "planner_cost_components_artifact",
    "planner_step_trace_artifact",
    "score_model",
    "upmem_pressure_score",
    "upmem_rank",
    "flop_rank",
    "score_components",
    "score_weights",
    "tradeoff_note",
    "pim_objective_version",
    "pim_weight_profile",
    "pim_normalization",
    "pim_execution_policy",
    "pim_feasible",
    "pim_rejection_reasons",
    "pim_estimated_flops",
    "pim_peak_intermediate_bytes",
    "pim_total_intermediate_write_bytes",
    "pim_estimated_host_to_dpu_bytes",
    "pim_estimated_dpu_to_host_bytes",
    "pim_estimated_host_dpu_bytes",
    "pim_estimated_mram_to_wram_bytes",
    "pim_estimated_dpu_local_work",
    "pim_estimated_sync_events",
    "pim_estimated_numerical_penalty",
    "pim_estimated_wram_pressure",
    "pim_estimated_tile_count",
    "pim_objective_components",
    "pim_normalized_components",
    "pim_objective_score",
    "pim_objective_rank",
    "pim_pareto_dominated",
    "pim_selected",
]


def compare_planners(suite_path: Path, root_dir: Path) -> Path:
    suite = load_suite(suite_path)
    planner_configs = comparison_planner_configs(suite)
    scoring_weights = comparison_scoring_weights(suite)
    pim_objective_config = comparison_pim_objective_config(suite)
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
        rows.extend(
            _compare_case(
                case,
                str(suite["suite_id"]),
                planner_configs,
                pim_objective_config,
                root_dir,
                run_dir,
            )
        )
    rows = score_planner_rows(rows, scoring_weights)
    rows = _score_pim_objective_rows(rows)

    payload = {
        "schema_version": COMPARISON_SCHEMA_VERSION,
        "suite_id": suite["suite_id"],
        "run_id": run_dir.name,
        "planner_configs": planner_configs,
        "scoring": scoring_metadata(scoring_weights),
        "pim_objective": pim_objective_config,
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
        backend_id="planner_comparison",
        execution_scope="contraction_planning",
        evidence_type="modeled_planner_comparison",
        normalized_records="normalized_records.jsonl",
        summary="planner_comparison.json",
        artifact_retention="full",
        root_dir=root_dir,
    )
    prune_run(run_dir, artifact_retention="full")
    return run_dir


def _load_planner_workload(
    case: dict[str, Any],
    root_dir: Path,
) -> tuple[Any, Any, dict[str, Any]]:
    """Load a circuit or a deliberately modeled-only planner motif.

    Planner motifs never pass through normal benchmark routes. Their metadata
    is copied into candidate rows so reports cannot mistake them for circuit
    performance evidence.
    """
    if is_planner_motif_case(case):
        workload = build_planner_motif_workload(case, root_dir)
        return workload.circuit, workload.network, dict(workload.metadata)

    circuit = load_circuit(case, root_dir)
    network = build_tensor_network(circuit)
    labels = {label for tensor in network.spec.tensors for label in tensor.labels}
    return circuit, network, {
        "workload_kind": "quantum_circuit",
        "not_real_quantum_circuit": False,
        "planner_motif": None,
        "network_tensor_count": len(network.spec.tensors),
        "network_index_count": len(labels),
        "network_max_rank": max((len(tensor.labels) for tensor in network.spec.tensors), default=0),
        "network_max_tensor_elements": max(
            (math.prod(tensor.shape) for tensor in network.spec.tensors),
            default=0,
        ),
        "network_size_proxy": len(network.spec.tensors),
    }


def _compare_case(
    case: dict[str, Any],
    suite_id: str,
    planner_configs: list[dict[str, Any]],
    pim_objective_config: dict[str, str],
    root_dir: Path,
    run_dir: Path,
) -> list[dict[str, Any]]:
    circuit, network, workload_metadata = _load_planner_workload(case, root_dir)
    case_id = str(case["case_id"])
    workload_id = str(case.get("workload_id", case_id))
    used_planner_dirs: dict[str, int] = {}
    rows: list[dict[str, Any]] = []

    for planner_config in planner_configs:
        profile = _pim_profile_for_config(planner_config, pim_objective_config)
        try:
            graph = plan_task_graph_with_config(network, planner_config)
            graph, _ = annotate_task_graph_with_upmem_estimates(graph)
            graph = with_path_cost_summary(graph)
            components = model_upmem_network_path_cost(network, graph.tasks, profile.policy)
            planner_dir_name = _unique_planner_dir_name(graph.path_summary.planner_id, used_planner_dirs)
            artifacts = _write_planner_artifacts(
                run_dir,
                suite_id,
                case_id,
                planner_dir_name,
                graph,
                components,
            )
            rows.append(
                _comparison_row(
                    case,
                    circuit.name,
                    circuit.n_qubits,
                    len(circuit.operations),
                    workload_id,
                    workload_metadata,
                    graph,
                    artifacts,
                    components,
                    profile,
                )
            )
        except PlannerInfeasibleError as exc:
            rows.append(
                _rejected_comparison_row(
                    case,
                    circuit.name,
                    circuit.n_qubits,
                    len(circuit.operations),
                    workload_id,
                    workload_metadata,
                    planner_config,
                    profile,
                    str(exc),
                    exc.rejection_reasons,
                    candidate_status="rejected",
                )
            )
        except (RuntimeError, ValueError) as exc:
            rows.append(
                _rejected_comparison_row(
                    case,
                    circuit.name,
                    circuit.n_qubits,
                    len(circuit.operations),
                    workload_id,
                    workload_metadata,
                    planner_config,
                    profile,
                    f"{type(exc).__name__}: {exc}",
                    ("planner_candidate_failed",),
                    candidate_status="failed",
                )
            )

    return rows


def _write_planner_artifacts(
    run_dir: Path,
    suite_id: str,
    case_id: str,
    planner_dir_name: str,
    graph: TaskGraph,
    components: PathCostComponents,
) -> dict[str, str]:
    planner_root = Path("cases") / case_id / "planners" / planner_dir_name
    task_graph_artifact = planner_root / "task_graph.json"
    path_summary_artifact = planner_root / "path_summary.json"
    target_estimates_artifact = planner_root / "target_estimates" / f"{UPMEM_DENSE_ESTIMATE_KEY}.jsonl"
    execution_bundle_artifact = planner_root / "execution_bundle.json"
    planner_cost_components_artifact = planner_root / "planner_cost_components.json"
    planner_step_trace_artifact = planner_root / "planner_step_trace.json"
    write_json(run_dir / task_graph_artifact, graph)
    write_json(run_dir / path_summary_artifact, graph.path_summary)
    write_jsonl(run_dir / target_estimates_artifact, upmem_task_estimate_rows(graph))
    write_json(
        run_dir / execution_bundle_artifact,
        build_execution_bundle(graph, case_id=case_id, suite_id=suite_id),
    )
    write_json(run_dir / planner_cost_components_artifact, components.to_json_dict())
    write_json(
        run_dir / planner_step_trace_artifact,
        list(graph.path_summary.planner_metadata.get("step_trace") or []),
    )
    return {
        "task_graph_artifact": task_graph_artifact.as_posix(),
        "path_summary_artifact": path_summary_artifact.as_posix(),
        "target_estimates_artifact": target_estimates_artifact.as_posix(),
        "execution_bundle_artifact": execution_bundle_artifact.as_posix(),
        "planner_cost_components_artifact": planner_cost_components_artifact.as_posix(),
        "planner_step_trace_artifact": planner_step_trace_artifact.as_posix(),
    }


def _comparison_row(
    case: dict[str, Any],
    circuit_name: str,
    n_qubits: int,
    depth: int,
    workload_id: str,
    workload_metadata: dict[str, Any],
    graph: TaskGraph,
    artifacts: dict[str, str],
    components: PathCostComponents,
    profile: Any,
) -> dict[str, Any]:
    summary = graph.path_summary
    return {
        "case_id": str(case["case_id"]),
        "workload_id": workload_id,
        "circuit_family": str(case.get("circuit", {}).get("name", circuit_name)),
        "n_qubits": n_qubits,
        "depth": depth,
        **workload_metadata,
        "planner_engine": summary.planner_engine,
        "planner_id": summary.planner_id,
        "planner_kind": summary.planner_kind,
        "optimize_mode": summary.optimize_mode,
        "objective": summary.objective,
        "cost_basis": summary.cost_basis,
        "target_estimate_key": summary.target_estimate_key,
        "candidate_status": "completed",
        "candidate_failure_reason": None,
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
        "contraction_path_structure_hash": contraction_path_structure_hash(graph),
        "planning_time_s": graph.planning_time_s,
        **_pim_component_fields(components, profile),
        **artifacts,
    }


def _pim_profile_for_config(planner_config: dict[str, Any], defaults: dict[str, str]) -> Any:
    objective_version = str(planner_config.get("objective_version", defaults["objective_version"]))
    if objective_version != UPMEM_PATH_OBJECTIVE_VERSION:
        raise ValueError(f"Unsupported modeled UPMEM objective version: {objective_version}")
    normalization_id = str(planner_config.get("normalization", defaults["normalization"]))
    if normalization_id != DEFAULT_UPMEM_PATH_COST_NORMALIZATION_ID:
        raise ValueError(f"Unsupported modeled UPMEM normalization: {normalization_id}")
    policy_id = str(planner_config.get("execution_policy", defaults["execution_policy"]))
    profile_id = str(planner_config.get("weight_profile", defaults["weight_profile"]))
    return upmem_path_cost_profile(profile_id, policy=upmem_path_cost_policy(policy_id))


def _pim_component_fields(components: PathCostComponents, profile: Any) -> dict[str, Any]:
    score = profile.score(components)
    return {
        "pim_objective_version": UPMEM_PATH_OBJECTIVE_VERSION,
        "pim_weight_profile": profile.profile_id,
        "pim_normalization": profile.normalization.to_json_dict(),
        "pim_execution_policy": profile.policy.to_json_dict(),
        "pim_feasible": bool(components.feasibility),
        "pim_rejection_reasons": list(components.rejection_reasons),
        "pim_estimated_flops": int(components.flops),
        "pim_peak_intermediate_bytes": int(components.peak_bytes),
        "pim_total_intermediate_write_bytes": int(components.intermediate_writes),
        "pim_estimated_host_to_dpu_bytes": int(components.host_to_dpu_bytes),
        "pim_estimated_dpu_to_host_bytes": int(components.dpu_to_host_bytes),
        "pim_estimated_host_dpu_bytes": int(components.host_dpu_bytes),
        "pim_estimated_mram_to_wram_bytes": int(components.mram_wram_bytes),
        "pim_estimated_dpu_local_work": int(components.local_work),
        "pim_estimated_sync_events": int(components.sync_events),
        "pim_estimated_numerical_penalty": float(components.numeric_penalty),
        "pim_estimated_wram_pressure": float(components.wram_pressure),
        "pim_estimated_tile_count": int(components.tiles),
        "pim_objective_components": components.to_json_dict(),
        "pim_normalized_components": profile.normalize(components) if components.feasibility else None,
        "pim_objective_score": float(score) if math.isfinite(score) else None,
        "pim_objective_rank": None,
        "pim_pareto_dominated": None,
        "pim_selected": False,
    }


def _rejected_comparison_row(
    case: dict[str, Any],
    circuit_name: str,
    n_qubits: int,
    depth: int,
    workload_id: str,
    workload_metadata: dict[str, Any],
    planner_config: dict[str, Any],
    profile: Any,
    failure_reason: str,
    rejection_reasons: tuple[str, ...],
    *,
    candidate_status: str,
) -> dict[str, Any]:
    identity = _planner_identity_fields(planner_config, profile)
    components = PathCostComponents(feasibility=False, rejection_reasons=tuple(rejection_reasons))
    return {
        "case_id": str(case["case_id"]),
        "workload_id": workload_id,
        "circuit_family": str(case.get("circuit", {}).get("name", circuit_name)),
        "n_qubits": n_qubits,
        "depth": depth,
        **workload_metadata,
        **identity,
        "candidate_status": candidate_status,
        "candidate_failure_reason": failure_reason,
        "task_count": None,
        "total_estimated_flops": None,
        "peak_intermediate_bytes": None,
        "total_host_to_dpu_bytes": None,
        "total_dpu_to_host_bytes": None,
        "total_mram_to_wram_bytes": None,
        "unsupported_task_count": None,
        "tiling_required_task_count": None,
        "missing_target_estimate_count": None,
        "estimated_total_tile_count": None,
        "estimated_max_parallel_tiles": None,
        "circuit_semantics_hash": None,
        "tensor_network_hash": None,
        "contraction_plan_hash": None,
        "contraction_path_structure_hash": None,
        "planning_time_s": None,
        "execution_bundle_artifact": None,
        "task_graph_artifact": None,
        "path_summary_artifact": None,
        "target_estimates_artifact": None,
        "planner_cost_components_artifact": None,
        "planner_step_trace_artifact": None,
        "score_model": None,
        "upmem_pressure_score": None,
        "upmem_rank": None,
        "flop_rank": None,
        "score_components": None,
        "score_weights": None,
        "tradeoff_note": "planner candidate rejected before path construction",
        **_pim_component_fields(components, profile),
    }


def _planner_identity_fields(planner_config: dict[str, Any], profile: Any) -> dict[str, Any]:
    engine = str(planner_config.get("engine", "opt_einsum"))
    if engine == "opt_einsum":
        optimize = str(planner_config.get("optimize", "greedy"))
        return {
            "planner_engine": engine,
            "planner_id": f"opt_einsum.{optimize}",
            "planner_kind": "external_path_optimizer",
            "optimize_mode": optimize,
            "objective": "opt_einsum_contract_path",
            "cost_basis": "opt_einsum_internal",
            "target_estimate_key": None,
        }
    if engine == "cotengra":
        objective = str(planner_config.get("objective", planner_config.get("minimize", "flops")))
        return {
            "planner_engine": engine,
            "planner_id": f"cotengra.{objective}",
            "planner_kind": "external_contraction_tree",
            "optimize_mode": str(planner_config.get("methods", "greedy")),
            "objective": f"cotengra_{objective}",
            "cost_basis": "cotengra_contraction_tree",
            "target_estimate_key": None,
        }
    if engine == "custom_upmem":
        return {
            "planner_engine": engine,
            "planner_id": f"custom_upmem.greedy.{profile.profile_id}",
            "planner_kind": "native_target_greedy",
            "optimize_mode": "greedy",
            "objective": UPMEM_PATH_OBJECTIVE_VERSION,
            "cost_basis": profile.policy.policy_id,
            "target_estimate_key": profile.policy.policy_id,
        }
    return {
        "planner_engine": engine,
        "planner_id": engine,
        "planner_kind": "unknown",
        "optimize_mode": "",
        "objective": "",
        "cost_basis": "",
        "target_estimate_key": None,
    }


def _score_pim_objective_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_case: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_case.setdefault(str(row["case_id"]), []).append(dict(row))

    scored: list[dict[str, Any]] = []
    for case_rows in by_case.values():
        feasible = [
            row
            for row in case_rows
            if row.get("candidate_status") == "completed"
            and row.get("pim_feasible") is True
            and row.get("pim_objective_score") is not None
        ]
        for row in case_rows:
            row["pim_pareto_dominated"] = _pim_pareto_dominated(row, feasible) if row in feasible else None

        by_profile: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
        for row in feasible:
            key = (
                str(row.get("pim_weight_profile")),
                _stable_json_text(row.get("pim_normalization")),
                _stable_json_text(row.get("pim_execution_policy")),
            )
            by_profile.setdefault(key, []).append(row)
        for profile_rows in by_profile.values():
            ordered = sorted(profile_rows, key=lambda row: (float(row["pim_objective_score"]), str(row["planner_id"])))
            rank = 0
            prior_score: float | None = None
            for index, row in enumerate(ordered, start=1):
                score = float(row["pim_objective_score"])
                if prior_score is None or not math.isclose(score, prior_score, rel_tol=1.0e-12, abs_tol=1.0e-12):
                    rank = index
                    prior_score = score
                row["pim_objective_rank"] = rank
                row["pim_selected"] = rank == 1
        scored.extend(case_rows)
    return scored


def _pim_pareto_dominated(row: dict[str, Any], feasible_rows: list[dict[str, Any]]) -> bool:
    if row not in feasible_rows:
        return False
    values = _pim_component_vector(row)
    for other in feasible_rows:
        if other is row:
            continue
        other_values = _pim_component_vector(other)
        if all(other_values[key] <= values[key] for key in values) and any(other_values[key] < values[key] for key in values):
            return True
    return False


def _pim_component_vector(row: dict[str, Any]) -> dict[str, float]:
    return {
        "flops": float(row.get("pim_estimated_flops", 0) or 0),
        "peak_bytes": float(row.get("pim_peak_intermediate_bytes", 0) or 0),
        "host_to_dpu_bytes": float(row.get("pim_estimated_host_to_dpu_bytes", 0) or 0),
        "dpu_to_host_bytes": float(row.get("pim_estimated_dpu_to_host_bytes", 0) or 0),
        "mram_wram_bytes": float(row.get("pim_estimated_mram_to_wram_bytes", 0) or 0),
        "tiles": float(row.get("pim_estimated_tile_count", 0) or 0),
        "sync_events": float(row.get("pim_estimated_sync_events", 0) or 0),
        "wram_pressure": float(row.get("pim_estimated_wram_pressure", 0) or 0),
        "numeric_penalty": float(row.get("pim_estimated_numerical_penalty", 0) or 0),
    }


def _stable_json_text(value: Any) -> str:
    return yaml.safe_dump(value, sort_keys=True)


def _normalized_planner_record(payload: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "benchmark_result_artifact_v1",
        "source_artifact": "planner_comparison.json",
        "run_id": payload["run_id"],
        "suite_id": _comparison_run_suite_id(str(payload["suite_id"])),
        "case_id": row["case_id"],
        "workload_id": row["workload_id"],
        "n_qubits": row["n_qubits"],
        "workload_kind": row.get("workload_kind"),
        "not_real_quantum_circuit": row.get("not_real_quantum_circuit"),
        "planner_motif": row.get("planner_motif"),
        "network_tensor_count": row.get("network_tensor_count"),
        "network_index_count": row.get("network_index_count"),
        "network_max_rank": row.get("network_max_rank"),
        "network_max_tensor_elements": row.get("network_max_tensor_elements"),
        "network_size_proxy": row.get("network_size_proxy"),
        "route_id": "planner_candidate_model",
        "backend_id": row["planner_id"],
        "backend_family": row["planner_engine"],
        "benchmark_role": "contraction_path_candidate",
        "kernel_family": "cpu_reference_only",
        "execution_model": "tensor_network_planning",
        "parallelism_mode": "not_executed",
        "parallelism_evidence_type": "modeled",
        "execution_plan_kind": "taskgraph_contraction_path",
        "execution_plan_executed": False,
        "circuit_semantics_hash": row.get("circuit_semantics_hash"),
        "tensor_network_hash": row.get("tensor_network_hash"),
        "contraction_plan_hash": row.get("contraction_plan_hash"),
        "contraction_path_structure_hash": row.get("contraction_path_structure_hash"),
        "plan_reused": False,
        "planning_in_timed_region": False,
        "executor_config_hash": executor_config_hash("planner_candidate_model", {"planner_id": row["planner_id"]}),
        "execution_bundle_artifact": row.get("execution_bundle_artifact"),
        "planner_cost_components_artifact": row.get("planner_cost_components_artifact"),
        "planner_step_trace_artifact": row.get("planner_step_trace_artifact"),
        "contraction_execution_target": "modeled",
        "accelerator_kind": "none",
        "execution_scope": "contraction_planning",
        "status": "completed" if row.get("candidate_status") == "completed" else "rejected",
        "validation_status": "not_applicable",
        "task_count": row.get("task_count"),
        "planning_time_s": row.get("planning_time_s"),
        "planner_engine": row["planner_engine"],
        "planner_id": row["planner_id"],
        "planner_kind": row["planner_kind"],
        "optimize_mode": row["optimize_mode"],
        "objective": row["objective"],
        "cost_basis": row["cost_basis"],
        "tn_estimated_flops": row.get("total_estimated_flops"),
        "tn_max_intermediate_bytes": row.get("peak_intermediate_bytes"),
        "total_host_to_dpu_bytes": row.get("total_host_to_dpu_bytes"),
        "total_dpu_to_host_bytes": row.get("total_dpu_to_host_bytes"),
        "total_mram_to_wram_bytes": row.get("total_mram_to_wram_bytes"),
        "unsupported_task_count": row.get("unsupported_task_count"),
        "tiling_required_task_count": row.get("tiling_required_task_count"),
        "estimated_total_tile_count": row.get("estimated_total_tile_count"),
        "estimated_max_parallel_tiles": row.get("estimated_max_parallel_tiles"),
        "score_model": row.get("score_model"),
        "upmem_pressure_score": row.get("upmem_pressure_score"),
        "upmem_rank": row.get("upmem_rank"),
        "flop_rank": row.get("flop_rank"),
        "candidate_status": row.get("candidate_status"),
        "candidate_failure_reason": row.get("candidate_failure_reason"),
        "pim_objective_version": row.get("pim_objective_version"),
        "pim_weight_profile": row.get("pim_weight_profile"),
        "pim_normalization": row.get("pim_normalization"),
        "pim_execution_policy": row.get("pim_execution_policy"),
        "pim_feasible": row.get("pim_feasible"),
        "pim_rejection_reasons": row.get("pim_rejection_reasons"),
        "pim_objective_score": row.get("pim_objective_score"),
        "pim_objective_rank": row.get("pim_objective_rank"),
        "pim_pareto_dominated": row.get("pim_pareto_dominated"),
        "pim_selected": row.get("pim_selected"),
        "pim_estimated_flops": row.get("pim_estimated_flops"),
        "pim_peak_intermediate_bytes": row.get("pim_peak_intermediate_bytes"),
        "pim_total_intermediate_write_bytes": row.get("pim_total_intermediate_write_bytes"),
        "pim_estimated_host_to_dpu_bytes": row.get("pim_estimated_host_to_dpu_bytes"),
        "pim_estimated_dpu_to_host_bytes": row.get("pim_estimated_dpu_to_host_bytes"),
        "pim_estimated_host_dpu_bytes": row.get("pim_estimated_host_dpu_bytes"),
        "pim_estimated_mram_to_wram_bytes": row.get("pim_estimated_mram_to_wram_bytes"),
        "pim_estimated_dpu_local_work": row.get("pim_estimated_dpu_local_work"),
        "pim_estimated_sync_events": row.get("pim_estimated_sync_events"),
        "pim_estimated_numerical_penalty": row.get("pim_estimated_numerical_penalty"),
        "pim_estimated_wram_pressure": row.get("pim_estimated_wram_pressure"),
        "pim_estimated_tile_count": row.get("pim_estimated_tile_count"),
        "pim_objective_components": row.get("pim_objective_components"),
        "pim_normalized_components": row.get("pim_normalized_components"),
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
