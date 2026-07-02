from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from quantum_bench.bench.config import load_suite
from quantum_bench.bench.run_dirs import create_run_dir
from quantum_bench.circuits import load_circuit
from quantum_bench.core.jsonio import write_json, write_jsonl
from quantum_bench.core.records import JsonDict, to_jsonable
from quantum_bench.environment import capture_environment
from quantum_bench.targets.upmem import SYNTHETIC_PRESSURE_ERROR, annotate_task_graph_with_upmem_estimates, is_synthetic_pressure_case
from quantum_bench.targets.upmem.taskgraph_runtime import UPMEM_TASKGRAPH_QUANTIZATION_MODES, build_generic_taskgraph_reference
from quantum_bench.tn import build_tensor_network, plan_task_graph_with_config, with_path_cost_summary


UPMEM_GENERIC_FEASIBILITY_SCHEMA_VERSION = "upmem_generic_feasibility_v1"

CASE_FIELDS = [
    "case_id",
    "workload_id",
    "circuit_family",
    "n_qubits",
    "quantization_mode",
    "status",
    "reason",
    "total_tasks",
    "completed_tasks",
    "unsupported_tasks",
    "failed_tasks",
    "unsupported_task_index",
    "unsupported_task_id",
    "unsupported_reason",
    "input_dtype_on_dpu",
    "accumulator_dtype_on_dpu",
    "scaling_applied",
    "actual_transfer_bytes_model",
    "full_precision_transfer_bytes_model",
    "transfer_compression_ratio_model",
]

TASK_FIELDS = [
    "case_id",
    "quantization_mode",
    "task_index",
    "task_id",
    "status",
    "reason",
    "selected_kernel_family",
    "input_dtype_on_dpu",
    "accumulator_dtype_on_dpu",
    "scaling_applied",
    "output_shape",
    "contracted_combination_count",
]


@dataclass(frozen=True)
class UpmemGenericFeasibilityResult:
    schema_version: str
    status: str
    reason: str | None
    run_dir: Path
    summary_path: Path
    row_count: int
    summary: JsonDict = field(default_factory=dict)

    def to_json_dict(self) -> JsonDict:
        payload = to_jsonable(self)
        payload["run_dir"] = str(self.run_dir)
        payload["summary_path"] = str(self.summary_path)
        return payload


def run_upmem_generic_feasibility(
    root_dir: Path,
    *,
    suite_path: Path,
    quantization_modes: tuple[str, ...] = ("none", "per_task_input_quantize"),
    max_taskgraph_tasks: int = 128,
) -> UpmemGenericFeasibilityResult:
    suite = load_suite(suite_path)
    _validate_options(quantization_modes=quantization_modes, max_taskgraph_tasks=max_taskgraph_tasks)
    run_dir = create_run_dir(root_dir, f"{suite['suite_id']}_upmem_generic_feasibility")
    write_json(run_dir / "environment.json", capture_environment(root_dir))
    write_json(
        run_dir / "config" / "upmem_generic_feasibility_input.json",
        {
            "suite_id": suite["suite_id"],
            "suite_path": str(suite_path),
            "quantization_modes": quantization_modes,
            "max_taskgraph_tasks": max_taskgraph_tasks,
            "dpu_programs_executed": False,
        },
    )

    case_rows: list[JsonDict] = []
    task_rows: list[JsonDict] = []
    for case_payload in suite["cases"]:
        case_rows_for_case, task_rows_for_case = _scan_case(
            root_dir=root_dir,
            suite=suite,
            case_payload=case_payload,
            quantization_modes=quantization_modes,
            max_taskgraph_tasks=max_taskgraph_tasks,
        )
        case_rows.extend(case_rows_for_case)
        task_rows.extend(task_rows_for_case)

    _write_csv(run_dir / "upmem_generic_feasibility_cases.csv", case_rows, CASE_FIELDS)
    _write_csv(run_dir / "upmem_generic_feasibility_tasks.csv", task_rows, TASK_FIELDS)
    write_jsonl(run_dir / "upmem_generic_feasibility_tasks.jsonl", task_rows)
    summary = _summary_payload(suite, suite_path, quantization_modes, max_taskgraph_tasks, case_rows)
    write_json(run_dir / "upmem_generic_feasibility.json", {"schema_version": UPMEM_GENERIC_FEASIBILITY_SCHEMA_VERSION, "summary": summary, "rows": case_rows})
    summary_path = run_dir / "upmem_generic_feasibility_summary.md"
    summary_path.write_text(_summary_markdown(summary, case_rows), encoding="utf-8")
    status = "completed"
    return UpmemGenericFeasibilityResult(
        schema_version=UPMEM_GENERIC_FEASIBILITY_SCHEMA_VERSION,
        status=status,
        reason=None,
        run_dir=run_dir,
        summary_path=summary_path,
        row_count=len(case_rows),
        summary=summary,
    )


def parse_csv_choices(raw: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in raw.split(",") if item.strip())
    if not values:
        raise ValueError("comma-separated option must contain at least one value")
    return values


def _validate_options(*, quantization_modes: tuple[str, ...], max_taskgraph_tasks: int) -> None:
    invalid = sorted(set(quantization_modes) - set(UPMEM_TASKGRAPH_QUANTIZATION_MODES))
    if invalid:
        raise ValueError(f"unsupported quantization modes: {','.join(invalid)}")
    if "persistent_network_quantized" in quantization_modes:
        raise ValueError("persistent_network_quantized is not implemented for generic feasibility scanning")
    if max_taskgraph_tasks < 0:
        raise ValueError("--max-taskgraph-tasks must be >= 0")


def _scan_case(
    *,
    root_dir: Path,
    suite: JsonDict,
    case_payload: JsonDict,
    quantization_modes: tuple[str, ...],
    max_taskgraph_tasks: int,
) -> tuple[list[JsonDict], list[JsonDict]]:
    case_id = str(case_payload["case_id"])
    workload_id = str(case_payload.get("workload_id", case_id))
    if is_synthetic_pressure_case(case_payload):
        row = _case_error_row(case_id, workload_id, "synthetic_pressure", 0, quantization_modes[0], SYNTHETIC_PRESSURE_ERROR)
        return [row | {"quantization_mode": mode} for mode in quantization_modes], []
    try:
        circuit = load_circuit(case_payload, root_dir)
        network = build_tensor_network(circuit)
        graph = plan_task_graph_with_config(network, suite["planner"])
        graph, _ = annotate_task_graph_with_upmem_estimates(graph)
        graph = with_path_cost_summary(graph)
    except Exception as exc:
        rows = [
            _case_error_row(case_id, workload_id, str(case_payload.get("circuit", {}).get("name", "unknown")), 0, mode, str(exc))
            for mode in quantization_modes
        ]
        return rows, []

    family = str(case_payload.get("circuit", {}).get("name", circuit.name))
    n_qubits = int(getattr(circuit, "n_qubits", 0) or 0)
    if len(graph.tasks) > max_taskgraph_tasks:
        rows = [
            _case_error_row(case_id, workload_id, family, n_qubits, mode, "taskgraph_task_cap_exceeded", total_tasks=len(graph.tasks))
            for mode in quantization_modes
        ]
        return rows, []

    case_rows: list[JsonDict] = []
    task_rows: list[JsonDict] = []
    for mode in quantization_modes:
        reference = build_generic_taskgraph_reference(graph=graph, network=network, case_id=case_id, quantization_mode=mode)  # type: ignore[arg-type]
        case_rows.append(_case_row(case_id, workload_id, family, n_qubits, mode, reference))
        for metric in reference.task_metrics:
            task_rows.append(_task_row(case_id, mode, metric))
    return case_rows, task_rows


def _case_error_row(
    case_id: str,
    workload_id: str,
    family: str,
    n_qubits: int,
    quantization_mode: str,
    reason: str,
    *,
    total_tasks: int = 0,
) -> JsonDict:
    return {
        "case_id": case_id,
        "workload_id": workload_id,
        "circuit_family": family,
        "n_qubits": int(n_qubits),
        "quantization_mode": quantization_mode,
        "status": "unsupported",
        "reason": reason,
        "total_tasks": int(total_tasks),
        "completed_tasks": 0,
        "unsupported_tasks": 1,
        "failed_tasks": 0,
        "unsupported_task_index": None,
        "unsupported_task_id": None,
        "unsupported_reason": reason,
        "input_dtype_on_dpu": None,
        "accumulator_dtype_on_dpu": None,
        "scaling_applied": None,
        "actual_transfer_bytes_model": 0,
        "full_precision_transfer_bytes_model": 0,
        "transfer_compression_ratio_model": None,
    }


def _case_row(case_id: str, workload_id: str, family: str, n_qubits: int, quantization_mode: str, reference) -> JsonDict:
    summary = dict(reference.summary or {})
    unsupported = next((row for row in reference.task_metrics if row.get("status") != "completed"), None)
    actual_transfer = int(summary.get("actual_transfer_bytes", 0) or 0)
    full_transfer = int(summary.get("full_precision_transfer_bytes_model", 0) or 0)
    return {
        "case_id": case_id,
        "workload_id": workload_id,
        "circuit_family": family,
        "n_qubits": int(n_qubits),
        "quantization_mode": quantization_mode,
        "status": reference.status,
        "reason": reference.reason,
        "total_tasks": int(summary.get("total_tasks", len(reference.task_metrics)) or 0),
        "completed_tasks": int(summary.get("completed_tasks", 0) or 0),
        "unsupported_tasks": int(summary.get("unsupported_tasks", 0) or 0),
        "failed_tasks": int(summary.get("failed_tasks", 0) or 0),
        "unsupported_task_index": unsupported.get("task_index") if unsupported else None,
        "unsupported_task_id": unsupported.get("task_id") if unsupported else None,
        "unsupported_reason": unsupported.get("reason") if unsupported else None,
        "input_dtype_on_dpu": _unique(reference.task_metrics, "input_dtype_on_dpu"),
        "accumulator_dtype_on_dpu": _unique(reference.task_metrics, "accumulator_dtype_on_dpu"),
        "scaling_applied": _unique(reference.task_metrics, "scaling_applied"),
        "actual_transfer_bytes_model": actual_transfer,
        "full_precision_transfer_bytes_model": full_transfer,
        "transfer_compression_ratio_model": None if actual_transfer <= 0 else full_transfer / actual_transfer,
    }


def _task_row(case_id: str, quantization_mode: str, metric: JsonDict) -> JsonDict:
    native = dict(metric.get("native_index_metadata") or {})
    return {
        "case_id": case_id,
        "quantization_mode": quantization_mode,
        "task_index": metric.get("task_index"),
        "task_id": metric.get("task_id"),
        "status": metric.get("status"),
        "reason": metric.get("reason"),
        "selected_kernel_family": metric.get("selected_kernel_family"),
        "input_dtype_on_dpu": metric.get("input_dtype_on_dpu"),
        "accumulator_dtype_on_dpu": metric.get("accumulator_dtype_on_dpu"),
        "scaling_applied": metric.get("scaling_applied"),
        "output_shape": metric.get("output_shape"),
        "contracted_combination_count": native.get("contracted_combination_count"),
    }


def _summary_payload(
    suite: JsonDict,
    suite_path: Path,
    quantization_modes: tuple[str, ...],
    max_taskgraph_tasks: int,
    rows: list[JsonDict],
) -> JsonDict:
    return to_jsonable(
        {
            "schema_version": UPMEM_GENERIC_FEASIBILITY_SCHEMA_VERSION,
            "suite_id": suite["suite_id"],
            "suite_path": str(suite_path),
            "quantization_modes": quantization_modes,
            "max_taskgraph_tasks": max_taskgraph_tasks,
            "case_mode_rows": len(rows),
            "completed_count": sum(1 for row in rows if row["status"] == "completed"),
            "unsupported_count": sum(1 for row in rows if row["status"] == "unsupported"),
            "failed_count": sum(1 for row in rows if row["status"] == "failed"),
            "blocker_counts": _blocker_counts(rows),
            "dpu_programs_executed": False,
            "metadata": {
                "scanner_only": True,
                "cpu_fallback_used": False,
                "strict_runtime_can_use_completed_rows": True,
            },
        }
    )


def _blocker_counts(rows: list[JsonDict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        if row["status"] == "completed":
            continue
        reason = str(row.get("unsupported_reason") or row.get("reason") or "unknown")
        counts[reason] = counts.get(reason, 0) + 1
    return dict(sorted(counts.items()))


def _summary_markdown(summary: JsonDict, rows: list[JsonDict]) -> str:
    lines = [
        "# UPMEM Generic Feasibility",
        "",
        f"Suite: `{summary['suite_id']}`",
        f"Rows: {summary['case_mode_rows']}",
        f"Completed: {summary['completed_count']}",
        f"Unsupported: {summary['unsupported_count']}",
        "",
        "This scanner does not execute UPMEM DPU programs. It checks whether the current bounded generic contract can cover each TaskGraph.",
        "",
        "| Case | Quantization | Status | Tasks | Blocker |",
        "| --- | --- | --- | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['case_id']} | {row['quantization_mode']} | {row['status']} | {row['total_tasks']} | {row.get('unsupported_reason') or ''} |"
        )
    lines.append("")
    return "\n".join(lines)


def _unique(rows: tuple[JsonDict, ...], field: str) -> Any:
    values = {row.get(field) for row in rows if row.get(field) is not None}
    if not values:
        return None
    if len(values) == 1:
        return next(iter(values))
    return sorted(str(value) for value in values)


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
