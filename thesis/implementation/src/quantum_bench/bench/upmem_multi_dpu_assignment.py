from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from quantum_bench.bench.config import load_suite
from quantum_bench.bench.reporting import write_normalized_records, write_run_manifest
from quantum_bench.bench.run_dirs import EVIDENCE_ARTIFACT_KIND, create_run_dir, sanitize
from quantum_bench.circuits import load_circuit, manifest
from quantum_bench.core.jsonio import write_json
from quantum_bench.core.records import ContractionTask, JsonDict, to_jsonable
from quantum_bench.environment import capture_environment
from quantum_bench.targets.upmem import UPMEM_DENSE_ESTIMATE_KEY, annotate_task_graph_with_upmem_estimates
from quantum_bench.tn import build_tensor_network, plan_task_graph_with_config, with_path_cost_summary
from quantum_bench.tn.execution import frontier_waves


SCHEMA_VERSION = "upmem_multi_dpu_assignment_v1"
ROUTE_ID = "upmem_multi_dpu_assignment_model"
ROUTE_LABEL = "upmem_multi_dpu_assignment"
STRATEGIES = (
    "sequential_single_dpu",
    "frontier_round_robin_dpu_groups",
    "frontier_size_aware_dpu_groups",
)

ASSIGNMENT_FIELDS = [
    "case_id",
    "workload_id",
    "circuit_family",
    "n_qubits",
    "frontier_wave_index",
    "task_index",
    "task_id",
    "dpu_group_id",
    "dependencies",
    "estimated_flops",
    "estimated_h2d_bytes",
    "estimated_d2h_bytes",
    "estimated_transfer_bytes",
]


@dataclass(frozen=True)
class UpmemMultiDpuAssignmentResult:
    run_dir: Path
    plan_path: Path
    normalized_records_path: Path
    status: str
    case_count: int


def run_upmem_multi_dpu_assignment(
    root_dir: Path,
    *,
    suite_path: Path,
    dpu_group_count: int = 4,
    strategy: str = "frontier_round_robin_dpu_groups",
) -> UpmemMultiDpuAssignmentResult:
    if dpu_group_count < 1:
        raise ValueError("--dpu-groups must be >= 1")
    if strategy not in STRATEGIES:
        raise ValueError(f"unsupported --strategy: {strategy}")
    suite = load_suite(suite_path)
    run_dir = create_run_dir(
        root_dir,
        str(suite["suite_id"]),
        artifact_kind=EVIDENCE_ARTIFACT_KIND,
        route_label=ROUTE_LABEL,
    )
    relative_suite_path = _display_path(suite_path, root_dir)
    write_run_manifest(
        run_dir,
        run_kind="upmem_multi_dpu_assignment",
        suite_id=str(suite["suite_id"]),
        suite_path=relative_suite_path,
        artifact_kind=EVIDENCE_ARTIFACT_KIND,
        route_label=ROUTE_LABEL,
        route_id=ROUTE_ID,
        backend_id="upmem_sdk_assignment_model",
        execution_scope="taskgraph_assignment_model",
        evidence_type="modeled_only",
        normalized_records="normalized_records.jsonl",
        summary="upmem_multi_dpu_assignment_plan.json",
        upmem_execution_mode="sdk_simulator",
        artifact_retention="compact",
        command="upmem-multi-dpu-assignment",
        root_dir=root_dir,
    )
    write_json(run_dir / "environment.json", capture_environment(root_dir))
    (run_dir / "config" / "resolved_suite.yml").write_text(yaml.safe_dump(suite, sort_keys=True), encoding="utf-8")

    case_plans: list[JsonDict] = []
    assignment_rows: list[JsonDict] = []
    normalized_records: list[JsonDict] = []
    for case_payload in suite["cases"]:
        case_plan, case_assignments, record = _plan_case(
            root_dir=root_dir,
            suite=suite,
            case_payload=dict(case_payload),
            dpu_group_count=dpu_group_count,
            strategy=strategy,
        )
        case_plans.append(case_plan)
        assignment_rows.extend(case_assignments)
        normalized_records.append(record)
        write_json(run_dir / "cases" / sanitize(str(case_payload["case_id"])) / "upmem_multi_dpu_assignment_case.json", case_plan)

    summary = _summary(case_plans)
    plan = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_dir.name,
        "suite_id": str(suite["suite_id"]),
        "route_id": ROUTE_ID,
        "scheduler_strategy": strategy,
        "evidence_type": "modeled_only",
        "dpu_group_count": int(dpu_group_count),
        "summary": summary,
        "cases": case_plans,
        "metadata": {
            "upmem_kernels_executed": False,
            "dpu_programs_executed": False,
            "execution_plan_executed": False,
            "hardware_execution": False,
            "hardware_speedup_applicable": False,
            "cpu_fallback_used": False,
            "modeled_only": True,
        },
    }
    plan_path = run_dir / "upmem_multi_dpu_assignment_plan.json"
    write_json(plan_path, plan)
    _write_csv(run_dir / "upmem_multi_dpu_assignments.csv", assignment_rows, ASSIGNMENT_FIELDS)
    records_path = write_normalized_records(run_dir, normalized_records)
    return UpmemMultiDpuAssignmentResult(
        run_dir=run_dir,
        plan_path=plan_path,
        normalized_records_path=records_path,
        status="completed",
        case_count=len(case_plans),
    )


def _plan_case(
    *,
    root_dir: Path,
    suite: JsonDict,
    case_payload: JsonDict,
    dpu_group_count: int,
    strategy: str,
) -> tuple[JsonDict, list[JsonDict], JsonDict]:
    case_id = str(case_payload["case_id"])
    workload_id = str(case_payload.get("workload_id", case_id))
    circuit = load_circuit(case_payload, root_dir)
    network = build_tensor_network(circuit)
    graph = plan_task_graph_with_config(network, dict(suite["planner"]))
    graph, _ = annotate_task_graph_with_upmem_estimates(graph)
    graph = with_path_cost_summary(graph)
    waves = frontier_waves(graph)
    assignments, validation = _assign_waves(waves, dpu_group_count=dpu_group_count, strategy=strategy)
    widths = [len(wave) for wave in waves]
    case_prefix = {
        "case_id": case_id,
        "workload_id": workload_id,
        "circuit_family": str(case_payload.get("circuit", {}).get("name", circuit.name)),
        "n_qubits": int(circuit.n_qubits),
    }
    assignment_rows = [{**case_prefix, **assignment} for assignment in assignments]
    total_h2d = sum(int(row["estimated_h2d_bytes"]) for row in assignment_rows)
    total_d2h = sum(int(row["estimated_d2h_bytes"]) for row in assignment_rows)
    total_flops = sum(int(row["estimated_flops"]) for row in assignment_rows)
    used_group_counts = [
        len({int(row["dpu_group_id"]) for row in assignment_rows if int(row["frontier_wave_index"]) == wave_index})
        for wave_index in range(len(waves))
    ]
    occupancy = (
        sum(used / max(1, dpu_group_count) for used in used_group_counts) / len(used_group_counts)
        if used_group_counts
        else 0.0
    )
    imbalance = _load_imbalance_ratio(assignment_rows, dpu_group_count)
    case_plan = {
        **case_prefix,
        "schema_version": SCHEMA_VERSION,
        "route_id": ROUTE_ID,
        "circuit": manifest(circuit),
        "planner_engine": graph.path_summary.planner_engine,
        "planner_id": graph.path_summary.planner_id,
        "optimize_mode": graph.path_summary.optimize_mode,
        "scheduler_strategy": strategy,
        "evidence_type": "modeled_only",
        "dpu_group_count": int(dpu_group_count),
        "task_count": len(graph.tasks),
        "frontier_wave_count": len(waves),
        "max_frontier_width": max(widths, default=0),
        "mean_frontier_width": sum(widths) / len(widths) if widths else 0.0,
        "assigned_task_count": len(assignments),
        "executed_dpu_task_count": 0,
        "unassigned_task_count": 0,
        "assigned_h2d_bytes": int(total_h2d),
        "assigned_d2h_bytes": int(total_d2h),
        "assigned_transfer_bytes": int(total_h2d + total_d2h),
        "assigned_estimated_flops": int(total_flops),
        "modeled_dpu_occupancy": float(occupancy),
        "modeled_load_imbalance_ratio": float(imbalance),
        "dpu_assignment_validation_status": validation["status"],
        "duplicate_assignment_check": validation["duplicate_assignment_check"],
        "missing_dependency_check": validation["missing_dependency_check"],
        "dependency_violation_detected": validation["dependency_violation_detected"],
        "frontier_waves": _wave_payloads(waves, assignment_rows),
    }
    record = {
        "schema_version": "benchmark_result_artifact_v1",
        "suite_id": str(suite["suite_id"]),
        **case_prefix,
        "route_id": ROUTE_ID,
        "backend_family": "upmem_sdk",
        "benchmark_role": "modeled_upmem_multi_dpu_assignment",
        "kernel_family": "modeled_task_assignment",
        "execution_model": "tensor_network",
        "parallelism_mode": "modeled_only",
        "parallelism_evidence_type": "modeled",
        "execution_plan_kind": "upmem_multi_dpu_assignment_model",
        "execution_plan_executed": False,
        "upmem_parallelism_mode": "sequential" if strategy == "sequential_single_dpu" else "frontier_multi_dpu",
        "upmem_parallelism_evidence_type": "modeled",
        "task_assignment_strategy": strategy,
        "dpu_group_count": int(dpu_group_count),
        "frontier_wave_count": case_plan["frontier_wave_count"],
        "max_frontier_width": case_plan["max_frontier_width"],
        "mean_frontier_width": case_plan["mean_frontier_width"],
        "assigned_task_count": case_plan["assigned_task_count"],
        "executed_dpu_task_count": 0,
        "unassigned_task_count": 0,
        "dpu_assignment_plan_artifact": "upmem_multi_dpu_assignment_plan.json",
        "dpu_assignment_validation_status": case_plan["dpu_assignment_validation_status"],
        "assigned_h2d_bytes": int(total_h2d),
        "assigned_d2h_bytes": int(total_d2h),
        "modeled_dpu_occupancy": float(occupancy),
        "modeled_load_imbalance_ratio": float(imbalance),
        "contraction_execution_target": "upmem",
        "accelerator_kind": "upmem",
        "upmem_execution_mode": "sdk_simulator",
        "execution_backend": "upmem_sdk",
        "hardware_execution": False,
        "hardware_timing_available": False,
        "hardware_speedup_applicable": False,
        "cpu_fallback_used": False,
        "cpu_fallback_task_count": 0,
        "native_sdk_control_path": True,
        "simplepim_api_used": False,
        "task_count": len(graph.tasks),
        "upmem_task_count": 0,
        "dpu_program_invocations": 0,
        "upmem_program_executed": False,
        "status": "modeled",
        "validation_status": "not_applicable_modeled_only",
        "output_kind": "not_applicable",
        "comparison_output_kind": "not_applicable",
        "output_contract": "metadata_only",
        "exact_output_comparable": False,
        "full_statevector_validation_available": False,
        "timing_scope": "not_executed",
        "total_wall_time_s": 0.0,
        "simulation_compute_time_s": 0.0,
        "validated_task_count": 0,
        "unsupported_task_count": 0,
        "duplicate_contraction_check": validation["duplicate_assignment_check"],
        "missing_dependency_check": validation["missing_dependency_check"],
        "dependency_violation_detected": validation["dependency_violation_detected"],
    }
    return to_jsonable(case_plan), to_jsonable(assignment_rows), to_jsonable(record)


def _assign_waves(
    waves: list[list[ContractionTask]],
    *,
    dpu_group_count: int,
    strategy: str,
) -> tuple[list[JsonDict], JsonDict]:
    completed: set[str] = set()
    seen: set[str] = set()
    assignments: list[JsonDict] = []
    missing_dependency = False
    duplicate = False
    for wave_index, wave in enumerate(waves):
        wave_assignments = _assign_wave(wave, dpu_group_count=dpu_group_count, strategy=strategy)
        for task_index, dpu_group_id in wave_assignments:
            task = wave[task_index]
            if task.id in seen:
                duplicate = True
            missing = [dependency for dependency in task.dependencies if dependency not in completed]
            if missing:
                missing_dependency = True
            seen.add(task.id)
            assignments.append(
                {
                    "frontier_wave_index": int(wave_index),
                    "task_index": int(_task_number(task)),
                    "task_id": task.id,
                    "dpu_group_id": int(dpu_group_id),
                    "dependencies": task.dependencies,
                    "estimated_flops": int(task.estimated_flops),
                    "estimated_h2d_bytes": int(_task_estimate(task).get("host_to_dpu_bytes", 0) or 0),
                    "estimated_d2h_bytes": int(_task_estimate(task).get("dpu_to_host_bytes", 0) or 0),
                    "estimated_transfer_bytes": int(
                        (_task_estimate(task).get("host_to_dpu_bytes", 0) or 0)
                        + (_task_estimate(task).get("dpu_to_host_bytes", 0) or 0)
                    ),
                }
            )
        completed.update(task.id for task in wave)
    validation_status = "failed" if duplicate or missing_dependency else "passed"
    return assignments, {
        "status": validation_status,
        "duplicate_assignment_check": "failed" if duplicate else "passed",
        "missing_dependency_check": "failed" if missing_dependency else "passed",
        "dependency_violation_detected": bool(missing_dependency),
    }


def _assign_wave(wave: list[ContractionTask], *, dpu_group_count: int, strategy: str) -> list[tuple[int, int]]:
    if strategy == "sequential_single_dpu":
        return [(idx, 0) for idx, _ in enumerate(wave)]
    if strategy == "frontier_round_robin_dpu_groups":
        return [(idx, idx % dpu_group_count) for idx, _ in enumerate(wave)]
    loads = [0 for _ in range(dpu_group_count)]
    assigned: list[tuple[int, int]] = []
    order = sorted(range(len(wave)), key=lambda idx: (_task_load(wave[idx]), wave[idx].id), reverse=True)
    for idx in order:
        group_id = min(range(dpu_group_count), key=lambda group: (loads[group], group))
        loads[group_id] += _task_load(wave[idx])
        assigned.append((idx, group_id))
    return sorted(assigned, key=lambda item: item[0])


def _wave_payloads(waves: list[list[ContractionTask]], assignment_rows: list[JsonDict]) -> list[JsonDict]:
    payloads: list[JsonDict] = []
    for wave_index, wave in enumerate(waves):
        rows = [row for row in assignment_rows if int(row["frontier_wave_index"]) == wave_index]
        payloads.append(
            {
                "frontier_wave_index": int(wave_index),
                "ready_task_count": len(wave),
                "task_ids": tuple(task.id for task in wave),
                "assigned_groups": sorted({int(row["dpu_group_id"]) for row in rows}),
                "assignments": [
                    {
                        "task_id": row["task_id"],
                        "dpu_group_id": row["dpu_group_id"],
                        "estimated_h2d_bytes": row["estimated_h2d_bytes"],
                        "estimated_d2h_bytes": row["estimated_d2h_bytes"],
                        "estimated_flops": row["estimated_flops"],
                    }
                    for row in rows
                ],
            }
        )
    return payloads


def _summary(case_plans: list[JsonDict]) -> JsonDict:
    return {
        "case_count": len(case_plans),
        "task_count": sum(int(case.get("task_count", 0) or 0) for case in case_plans),
        "assigned_task_count": sum(int(case.get("assigned_task_count", 0) or 0) for case in case_plans),
        "executed_dpu_task_count": 0,
        "unassigned_task_count": sum(int(case.get("unassigned_task_count", 0) or 0) for case in case_plans),
        "max_frontier_width": max((int(case.get("max_frontier_width", 0) or 0) for case in case_plans), default=0),
        "total_assigned_h2d_bytes": sum(int(case.get("assigned_h2d_bytes", 0) or 0) for case in case_plans),
        "total_assigned_d2h_bytes": sum(int(case.get("assigned_d2h_bytes", 0) or 0) for case in case_plans),
        "dpu_assignment_validation_status": (
            "passed"
            if all(case.get("dpu_assignment_validation_status") == "passed" for case in case_plans)
            else "failed"
        ),
        "hardware_execution": False,
        "hardware_speedup_applicable": False,
    }


def _task_estimate(task: ContractionTask) -> JsonDict:
    return dict(task.target_estimates.get(UPMEM_DENSE_ESTIMATE_KEY) or {})


def _task_load(task: ContractionTask) -> int:
    estimate = _task_estimate(task)
    return int(estimate.get("host_to_dpu_bytes", 0) or 0) + int(estimate.get("dpu_to_host_bytes", 0) or 0) or int(task.estimated_bytes)


def _task_number(task: ContractionTask) -> int:
    try:
        return int(task.id.rsplit("_", 1)[-1])
    except ValueError:
        return 0


def _load_imbalance_ratio(rows: list[JsonDict], dpu_group_count: int) -> float:
    if not rows:
        return 0.0
    loads = [0 for _ in range(dpu_group_count)]
    for row in rows:
        loads[int(row["dpu_group_id"])] += int(row["estimated_transfer_bytes"])
    nonzero = [load for load in loads if load > 0]
    if not nonzero:
        return 0.0
    mean = sum(nonzero) / len(nonzero)
    return max(nonzero) / mean if mean else 0.0


def _write_csv(path: Path, rows: list[JsonDict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in fields})


def _csv_value(value: Any) -> Any:
    value = to_jsonable(value)
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    if value is None:
        return ""
    return value


def _display_path(path: Path, root_dir: Path) -> str:
    try:
        return path.resolve().relative_to(root_dir.resolve()).as_posix()
    except ValueError:
        return path.as_posix()
