from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any, Literal, Mapping

import yaml

from quantum_bench.bench.config import DEFAULTS, load_suite
from quantum_bench.bench.run_dirs import create_run_dir
from quantum_bench.circuits import builtin_circuit, load_circuit, manifest
from quantum_bench.core.jsonio import write_json, write_jsonl
from quantum_bench.core.records import JsonDict, TaskGraph, to_jsonable
from quantum_bench.environment import capture_environment
from quantum_bench.routing import DenseTaskPreparationInput, prepare_dense_task
from quantum_bench.targets.upmem import (
    UPMEM_DENSE_ESTIMATE_KEY,
    annotate_task_graph_with_upmem_estimates,
    dense_bridge_manifest_eligibility,
    execute_dense_bridge,
    probe_simplepim,
    write_dense_bridge_input_manifest,
)
from quantum_bench.tn import (
    TaskInputMaterializationRequest,
    build_tensor_network,
    materialize_task_inputs,
    plan_task_graph_with_config,
    with_path_cost_summary,
)


DENSE_ROUTE_COVERAGE_SCHEMA_VERSION = "dense_route_coverage_v1"

DenseRouteCoverageBridgeBackend = Literal["none", "simplepim_external_stub"]

COVERAGE_FIELDS = [
    "case_id",
    "workload_id",
    "circuit_family",
    "n_qubits",
    "planner_engine",
    "planner_id",
    "optimize_mode",
    "task_index",
    "task_id",
    "input_tensor_ids",
    "output_tensor_id",
    "input_shapes",
    "output_shape",
    "gemm_m",
    "gemm_k",
    "gemm_n",
    "input_sources",
    "materialization_status",
    "materialization_reason",
    "replayed_task_count",
    "replay_time_s",
    "peak_materialized_bytes",
    "dense_prepare_status",
    "dense_prepare_reason",
    "dense_prepare_error",
    "conversion_dtype",
    "conversion_error_metrics",
    "validation_metrics",
    "supported_by_dense_estimate",
    "requires_tiling",
    "tiling_implemented",
    "tile_count",
    "working_set_bytes",
    "host_to_dpu_bytes",
    "dpu_to_host_bytes",
    "mram_to_wram_bytes",
    "requires_host_aggregation",
    "estimate_reject_reason",
    "simplepim_probe_status",
    "bridge_manifest_eligible",
    "bridge_artifact_written",
    "bridge_artifact_path",
    "external_stub_eligible",
    "external_stub_checked",
    "external_stub_status",
    "external_stub_reason",
    "external_command_executed",
    "execution_implemented",
    "final_readiness_level",
    "readiness_reason",
]


def run_dense_route_coverage(
    root_dir: Path,
    *,
    suite_path: Path | None = None,
    case: str | None = None,
    n_qubits: int | None = None,
    bridge_backend: DenseRouteCoverageBridgeBackend = "none",
    execute_external: bool = False,
    max_bridge_artifacts: int = 0,
    env: Mapping[str, str] | None = None,
) -> Path:
    _validate_options(
        suite_path=suite_path,
        case=case,
        bridge_backend=bridge_backend,
        execute_external=execute_external,
        max_bridge_artifacts=max_bridge_artifacts,
        env=env,
    )

    suite: dict[str, Any] | None
    cases: list[dict[str, Any]]
    suite_id: str
    planner_config: dict[str, Any]
    if suite_path is not None:
        suite = load_suite(suite_path)
        cases = list(suite["cases"])
        suite_id = str(suite["suite_id"])
        planner_config = dict(suite["planner"])
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

    run_dir = create_run_dir(root_dir, f"{suite_id}_dense_route_coverage")
    write_json(run_dir / "environment.json", capture_environment(root_dir))
    if suite is not None:
        (run_dir / "config" / "resolved_suite.yml").write_text(
            yaml.safe_dump(suite, sort_keys=True),
            encoding="utf-8",
        )
    else:
        write_json(
            run_dir / "config" / "dense_route_coverage_input.json",
            {
                "case": case,
                "n_qubits": n_qubits,
                "planner": planner_config,
                "bridge_backend": bridge_backend,
                "execute_external": execute_external,
                "max_bridge_artifacts": max_bridge_artifacts,
            },
        )

    probe = probe_simplepim(env=env) if env is not None else probe_simplepim()
    rows: list[JsonDict] = []
    bridge_artifacts_written = 0
    for case_payload in cases:
        case_rows, bridge_artifacts_written = _analyze_case(
            root_dir=root_dir,
            run_dir=run_dir,
            case_payload=case_payload,
            planner_config=planner_config,
            probe=probe,
            bridge_backend=bridge_backend,
            execute_external=execute_external,
            max_bridge_artifacts=max_bridge_artifacts,
            bridge_artifacts_written=bridge_artifacts_written,
            env=env,
        )
        rows.extend(case_rows)

    summary = _coverage_summary(rows)
    payload = {
        "schema_version": DENSE_ROUTE_COVERAGE_SCHEMA_VERSION,
        "run_id": run_dir.name,
        "suite_id": suite_id,
        "source": "suite" if suite_path is not None else "case",
        "suite_path": str(suite_path) if suite_path is not None else None,
        "planner": planner_config,
        "bridge_backend": bridge_backend,
        "execute_external": execute_external,
        "max_bridge_artifacts": max_bridge_artifacts,
        "simplepim_probe": probe.to_json_dict(),
        "summary": summary,
        "rows": rows,
    }
    write_json(run_dir / "dense_route_coverage.json", payload)
    _write_coverage_csv(run_dir / "dense_route_coverage.csv", rows)
    (run_dir / "dense_route_coverage_summary.md").write_text(_coverage_markdown(summary, rows), encoding="utf-8")
    return run_dir


def validate_cli_options(
    *,
    suite_path: Path | None,
    case: str | None,
    bridge_backend: str,
    execute_external: bool,
    max_bridge_artifacts: int,
    env: Mapping[str, str] | None = None,
) -> None:
    _validate_options(
        suite_path=suite_path,
        case=case,
        bridge_backend=bridge_backend,
        execute_external=execute_external,
        max_bridge_artifacts=max_bridge_artifacts,
        env=env,
    )


def _analyze_case(
    *,
    root_dir: Path,
    run_dir: Path,
    case_payload: dict[str, Any],
    planner_config: dict[str, Any],
    probe: Any,
    bridge_backend: DenseRouteCoverageBridgeBackend,
    execute_external: bool,
    max_bridge_artifacts: int,
    bridge_artifacts_written: int,
    env: Mapping[str, str] | None,
) -> tuple[list[JsonDict], int]:
    circuit = case_payload.pop("_preloaded_circuit", None)
    if circuit is None:
        circuit = load_circuit(case_payload, root_dir)
    network = build_tensor_network(circuit)
    graph = plan_task_graph_with_config(network, planner_config)
    graph, _ = annotate_task_graph_with_upmem_estimates(graph)
    graph = with_path_cost_summary(graph)
    initial_tensors = {tensor.spec.id: tensor for tensor in network.tensors}
    case_id = str(case_payload["case_id"])
    workload_id = str(case_payload.get("workload_id", case_id))
    circuit_family = str(case_payload.get("circuit", {}).get("name", circuit.name))
    case_rows: list[JsonDict] = []

    for task_index, task in enumerate(graph.tasks):
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

        row = _coverage_row(
            case_id=case_id,
            workload_id=workload_id,
            circuit_family=circuit_family,
            n_qubits=int(circuit.n_qubits),
            graph=graph,
            task_index=task_index,
            materialization=materialization,
            preparation=preparation,
            probe=probe,
        )
        if row["bridge_manifest_eligible"] and bridge_artifacts_written < max_bridge_artifacts:
            row, bridge_artifacts_written = _write_optional_bridge_artifact(
                row=row,
                run_dir=run_dir,
                case_id=case_id,
                task_index=task_index,
                preparation=preparation,
                bridge_backend=bridge_backend,
                execute_external=execute_external,
                bridge_artifacts_written=bridge_artifacts_written,
                env=env,
            )
        case_rows.append(row)

    write_jsonl(run_dir / "cases" / case_id / "dense_route_coverage.jsonl", case_rows)
    return case_rows, bridge_artifacts_written


def _coverage_row(
    *,
    case_id: str,
    workload_id: str,
    circuit_family: str,
    n_qubits: int,
    graph: TaskGraph,
    task_index: int,
    materialization: Any,
    preparation: Any,
    probe: Any,
) -> JsonDict:
    task = graph.tasks[task_index]
    estimate = task.target_estimates.get(UPMEM_DENSE_ESTIMATE_KEY, {})
    tile_plan = estimate.get("tile_plan") if isinstance(estimate, dict) else None
    if not isinstance(tile_plan, dict):
        tile_plan = {}
    prep_payload = preparation.to_json_dict() if preparation is not None else {}
    conversion_records = prep_payload.get("conversion_records") or {}
    left_conversion = conversion_records.get("left") if isinstance(conversion_records, dict) else None
    right_conversion = conversion_records.get("right") if isinstance(conversion_records, dict) else None
    bridge_manifest_eligible, bridge_reason = dense_bridge_manifest_eligibility(preparation)
    row = {
        "case_id": case_id,
        "workload_id": workload_id,
        "circuit_family": circuit_family,
        "n_qubits": n_qubits,
        "planner_engine": graph.path_summary.planner_engine,
        "planner_id": graph.path_summary.planner_id,
        "optimize_mode": graph.path_summary.optimize_mode,
        "task_index": task_index,
        "task_id": task.id,
        "input_tensor_ids": task.input_tensor_ids,
        "output_tensor_id": task.output_tensor_id,
        "input_shapes": task.input_shapes,
        "output_shape": task.output_shape,
        "gemm_m": task.gemm_m,
        "gemm_k": task.gemm_k,
        "gemm_n": task.gemm_n,
        "input_sources": materialization.input_sources,
        "materialization_status": materialization.status,
        "materialization_reason": materialization.reason,
        "replayed_task_count": materialization.replayed_task_count,
        "replay_time_s": materialization.replay_time_s,
        "peak_materialized_bytes": materialization.peak_materialized_bytes,
        "dense_prepare_status": prep_payload.get("status"),
        "dense_prepare_reason": prep_payload.get("reason"),
        "dense_prepare_error": prep_payload.get("error"),
        "conversion_dtype": _conversion_dtype(left_conversion, right_conversion),
        "conversion_error_metrics": _conversion_error_metrics(left_conversion, right_conversion),
        "validation_metrics": prep_payload.get("validation_metrics"),
        "supported_by_dense_estimate": bool(estimate.get("supported", False)) if isinstance(estimate, dict) else False,
        "requires_tiling": bool(estimate.get("requires_tiling", False)) if isinstance(estimate, dict) else False,
        "tiling_implemented": bool(estimate.get("tiling_implemented", False)) if isinstance(estimate, dict) else False,
        "tile_count": int(estimate.get("estimated_tile_count", tile_plan.get("total_tile_count", 0)) or 0) if isinstance(estimate, dict) else 0,
        "working_set_bytes": int(estimate.get("max_working_set_bytes", tile_plan.get("working_set_bytes", 0)) or 0) if isinstance(estimate, dict) else 0,
        "host_to_dpu_bytes": int(estimate.get("host_to_dpu_bytes", 0) or 0) if isinstance(estimate, dict) else 0,
        "dpu_to_host_bytes": int(estimate.get("dpu_to_host_bytes", 0) or 0) if isinstance(estimate, dict) else 0,
        "mram_to_wram_bytes": int(estimate.get("mram_to_wram_bytes", 0) or 0) if isinstance(estimate, dict) else 0,
        "requires_host_aggregation": bool(estimate.get("requires_host_aggregation", tile_plan.get("requires_host_aggregation", False))) if isinstance(estimate, dict) else False,
        "estimate_reject_reason": estimate.get("reject_reason") if isinstance(estimate, dict) else None,
        "simplepim_probe_status": probe.simplepim_probe_status,
        "bridge_manifest_eligible": bridge_manifest_eligible,
        "bridge_artifact_written": False,
        "bridge_artifact_path": None,
        "external_stub_eligible": bridge_manifest_eligible,
        "external_stub_checked": False,
        "external_stub_status": None,
        "external_stub_reason": None,
        "external_command_executed": False,
        "execution_implemented": False,
        "final_readiness_level": None,
        "readiness_reason": bridge_reason,
    }
    level, reason = _readiness_level(row)
    row["final_readiness_level"] = level
    row["readiness_reason"] = reason
    return to_jsonable(row)


def _write_optional_bridge_artifact(
    *,
    row: JsonDict,
    run_dir: Path,
    case_id: str,
    task_index: int,
    preparation: Any,
    bridge_backend: DenseRouteCoverageBridgeBackend,
    execute_external: bool,
    bridge_artifacts_written: int,
    env: Mapping[str, str] | None,
) -> tuple[JsonDict, int]:
    bridge_dir = run_dir / "cases" / case_id / "dense_bridge" / f"task_{task_index:04d}"
    write_dense_bridge_input_manifest(preparation, bridge_dir)
    input_manifest_path = bridge_dir / "input_manifest.json"
    rel_input_manifest = input_manifest_path.relative_to(run_dir).as_posix()
    row = dict(row)
    row["bridge_artifact_written"] = True
    row["bridge_artifact_path"] = rel_input_manifest
    row["readiness_reason"] = "bridge_manifest_written"
    bridge_artifacts_written += 1

    if bridge_backend == "simplepim_external_stub":
        bridge_result = execute_dense_bridge(
            input_manifest_path,
            backend="simplepim_external_stub",
            execute_external=execute_external,
            env=env,
        )
        row["external_stub_status"] = bridge_result.execution_status
        row["external_stub_reason"] = bridge_result.reason
        row["external_command_executed"] = bridge_result.external_command_executed
        row["execution_implemented"] = bridge_result.execution_implemented
        if execute_external:
            row["external_stub_checked"] = True
        if bridge_result.execution_status == "stub_executed":
            row["final_readiness_level"] = "external_stub_ready"
            row["readiness_reason"] = "external_stub_contract_executed"
        elif bridge_result.execution_status in {"failed", "skipped", "not_implemented", "unsupported"}:
            row["final_readiness_level"] = "bridge_manifest_ready"
            row["readiness_reason"] = f"external_stub_{bridge_result.execution_status}:{bridge_result.reason}"
    return row, bridge_artifacts_written


def _readiness_level(row: JsonDict) -> tuple[str, str | None]:
    if row["materialization_status"] not in {"initial_inputs_available", "materialized"}:
        return "not_materializable", row["materialization_reason"] or "task_inputs_not_materialized"
    if row["supported_by_dense_estimate"] is False:
        return "blocked_unsupported_shape", row["estimate_reject_reason"] or row["dense_prepare_reason"]
    if row["requires_tiling"] is True and row["tiling_implemented"] is False:
        reason = row["estimate_reject_reason"] or row["dense_prepare_reason"] or "requires_tiling_not_implemented"
        if row.get("requires_host_aggregation") and "host_aggregation" not in str(reason):
            reason = f"{reason};requires_host_aggregation"
        return "blocked_requires_tiling", reason
    if row["dense_prepare_status"] == "failed":
        return "dense_prepare_failed", row["dense_prepare_reason"] or "dense_preparation_failed"
    if row["bridge_manifest_eligible"]:
        return "bridge_manifest_ready", row["readiness_reason"]
    if row["dense_prepare_status"] is not None:
        return "dense_prepare_ready", row["readiness_reason"] or row["dense_prepare_reason"]
    return "materialized_only", row["readiness_reason"]


def _conversion_dtype(left_conversion: Any, right_conversion: Any) -> str | None:
    dtypes = {
        str(record.get("route_dtype"))
        for record in (left_conversion, right_conversion)
        if isinstance(record, dict) and record.get("route_dtype")
    }
    if not dtypes:
        return None
    if len(dtypes) == 1:
        return next(iter(dtypes))
    return ",".join(sorted(dtypes))


def _conversion_error_metrics(left_conversion: Any, right_conversion: Any) -> JsonDict | None:
    if not isinstance(left_conversion, dict) and not isinstance(right_conversion, dict):
        return None
    return {
        "left": _conversion_errors(left_conversion),
        "right": _conversion_errors(right_conversion),
    }


def _conversion_errors(record: Any) -> JsonDict | None:
    if not isinstance(record, dict):
        return None
    return {
        "quantization_error": record.get("quantization_error"),
        "dequantization_error": record.get("dequantization_error"),
        "clipping_count": record.get("clipping_count"),
        "saturation_count": record.get("saturation_count"),
        "scale": record.get("scale"),
    }


def _coverage_summary(rows: list[JsonDict]) -> JsonDict:
    readiness_by_case: dict[str, dict[str, int]] = {}
    for row in rows:
        case_id = str(row["case_id"])
        level = str(row["final_readiness_level"])
        readiness_by_case.setdefault(case_id, {})
        readiness_by_case[case_id][level] = readiness_by_case[case_id].get(level, 0) + 1
    return {
        "total_tasks": len(rows),
        "tasks_materializable_by_cpu_replay": sum(
            1 for row in rows if row["materialization_status"] in {"initial_inputs_available", "materialized"}
        ),
        "tasks_dense_preparable": sum(
            1
            for row in rows
            if row["dense_prepare_status"]
            in {"prepared", "simplepim_unavailable", "requires_executable_tiling_not_implemented"}
        ),
        "tasks_requiring_tiling": sum(1 for row in rows if row["requires_tiling"]),
        "tasks_blocked_by_unsupported_estimate": sum(1 for row in rows if row["supported_by_dense_estimate"] is False),
        "tasks_bridge_manifest_eligible": sum(1 for row in rows if row["bridge_manifest_eligible"]),
        "tasks_external_stub_eligible": sum(1 for row in rows if row["external_stub_eligible"]),
        "tasks_external_stub_checked": sum(1 for row in rows if row["external_stub_checked"]),
        "total_host_to_dpu_bytes": sum(int(row["host_to_dpu_bytes"] or 0) for row in rows),
        "total_dpu_to_host_bytes": sum(int(row["dpu_to_host_bytes"] or 0) for row in rows),
        "total_mram_to_wram_bytes": sum(int(row["mram_to_wram_bytes"] or 0) for row in rows),
        "max_tile_count": max((int(row["tile_count"] or 0) for row in rows), default=0),
        "readiness_counts_by_case": readiness_by_case,
    }


def _coverage_markdown(summary: JsonDict, rows: list[JsonDict]) -> str:
    lines = [
        "# Dense Route Coverage",
        "",
        "Developer-only readiness analysis for the dense UPMEM/SimplePIM route. No normal routed execution is performed.",
        "",
        f"- Total tasks: {summary['total_tasks']}",
        f"- Materializable by CPU replay: {summary['tasks_materializable_by_cpu_replay']}",
        f"- Dense-preparable tasks: {summary['tasks_dense_preparable']}",
        f"- Bridge-manifest eligible tasks: {summary['tasks_bridge_manifest_eligible']}",
        f"- External-stub checked tasks: {summary['tasks_external_stub_checked']}",
        f"- Total H2D bytes: {summary['total_host_to_dpu_bytes']}",
        f"- Total D2H bytes: {summary['total_dpu_to_host_bytes']}",
        f"- Total MRAM-WRAM bytes: {summary['total_mram_to_wram_bytes']}",
        f"- Max tile count: {summary['max_tile_count']}",
        "",
        "## Readiness By Case",
        "",
        "| Case | Readiness Counts |",
        "| --- | --- |",
    ]
    for case_id, counts in sorted(summary["readiness_counts_by_case"].items()):
        counts_text = ", ".join(f"{level}: {count}" for level, count in sorted(counts.items()))
        lines.append(f"| {case_id} | {counts_text} |")
    if not rows:
        lines.append("| none | no tasks |")
    lines.append("")
    return "\n".join(lines)


def _write_coverage_csv(path: Path, rows: list[JsonDict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COVERAGE_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in COVERAGE_FIELDS})


def _csv_value(value: Any) -> Any:
    value = to_jsonable(value)
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    if value is None:
        return ""
    return value


def _single_builtin_circuit(case: str, n_qubits: int | None):
    params: JsonDict = {"name": case}
    if n_qubits is not None:
        params["n_qubits"] = int(n_qubits)
    return builtin_circuit(case, params)


def _validate_options(
    *,
    suite_path: Path | None,
    case: str | None,
    bridge_backend: str,
    execute_external: bool,
    max_bridge_artifacts: int,
    env: Mapping[str, str] | None,
) -> None:
    if (suite_path is None) == (case is None):
        raise ValueError("dense-route-coverage requires exactly one of --suite or --case")
    if bridge_backend not in {"none", "simplepim_external_stub"}:
        raise ValueError("bridge backend must be one of: none, simplepim_external_stub")
    if max_bridge_artifacts < 0:
        raise ValueError("--max-bridge-artifacts must be >= 0")
    if execute_external and bridge_backend == "none":
        raise ValueError("--execute-external requires --bridge-backend simplepim_external_stub")
    if execute_external and max_bridge_artifacts == 0:
        raise ValueError("--execute-external requires --max-bridge-artifacts > 0")
    environment = env if env is not None else os.environ
    if execute_external and not environment.get("SIMPLEPIM_STUB_BIN"):
        raise ValueError("--execute-external requires SIMPLEPIM_STUB_BIN to be configured")
