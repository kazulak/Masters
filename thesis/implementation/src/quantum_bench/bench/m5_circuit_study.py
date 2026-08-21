"""Whole-circuit M5 study orchestration.

This module is intentionally a coordinator, not another execution engine.  A
study expands explicit planner, numeric-policy, engine and topology variants,
then sends one :class:`~quantum_bench.tn.graph.ContractionDAG` through the
functional compile/execute boundary.  Physical engines are injected by the
caller so planning and CI never open a device or silently substitute a
simulator.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import yaml

from quantum_bench.bench.reporting import write_normalized_records, write_run_manifest
from quantum_bench.bench.run_dirs import EVIDENCE_ARTIFACT_KIND, create_run_dir
from quantum_bench.bench.m5_circuit_routes import (
    admit_route_observation,
    apply_rank_path_override,
    build_pipeline_catalog,
    route_comparison_ids,
    select_pipeline_routes,
    selected_pipeline_comparisons,
    validate_route_adapter,
)
from quantum_bench.circuits import load_circuit, manifest as circuit_manifest
from quantum_bench.core.jsonio import write_json
from quantum_bench.core.records import TensorNetworkSpec
from quantum_bench.execution import (
    CpuCompileRequest,
    ExecutionFailure,
    ExecutionResult,
    NumericMode,
    RunContext,
    Target,
    UnsupportedExecution,
    UpmemCompileRequest,
    UpmemRuntimeResources,
    UpmemTopology,
    canonical_serialize,
    compile_execution,
    execute,
    execution_plan_hash,
)
from quantum_bench.tn.graph import (
    ContractionDAG,
    ContractNode,
    ReduceNode,
    TensorView,
    build_contraction_dag,
    contraction_dag_hash,
)
from quantum_bench.tn.network import build_tensor_network_data
from quantum_bench.tn.planning import PlannerRequest, plan_contractions
from quantum_bench.tn.planner_records import PlannerResult
from quantum_bench.whole_circuit import (
    DeviceTopology,
    Float32RealPolicy,
    HostPackedInt8Policy,
)


SCHEMA_VERSION = 2
LEGACY_SCHEMA_VERSION = 1
LEGACY_STUDY_SCHEMA_VERSION = "m5_circuit_study_v1"
STUDY_SCHEMA_VERSION = "m5_circuit_study_v2"
STUDY_RECORD_SCHEMA_VERSION = "m5_circuit_study_record_v2"
RANK_PATH_PATTERN = re.compile(r"^/dev/dpu_rank[0-9]+$")
ROUTE_LABEL = "m5_circuit_study"
DEFAULT_TIMEOUT_S = 300.0
DEFAULT_MAX_LIVE_BYTES = 512 * 1024 * 1024
DEFAULT_TOLERANCES = {
    "max_abs_error": 1.0e-5,
    "l2_error": 1.0e-5,
    "max_rel_error": 1.0e-4,
    "norm_drift": 1.0e-5,
    "quantized_max_abs_error": 0.25,
    "quantized_l2_error": 0.25,
    "quantized_max_rel_error": 1.0,
    "quantized_norm_drift": 0.25,
}


@dataclass(frozen=True)
class _Plan:
    case: dict[str, Any]
    planner: dict[str, Any]
    circuit: Any
    spec: TensorNetworkSpec
    inputs: dict[str, np.ndarray]
    planner_result: PlannerResult
    dag: ContractionDAG
    dag_hash: str
    circuit_semantics_hash: str
    tensor_network_hash: str
    resources: dict[str, Any]
    preflight: dict[str, Any]


def load_study_config(path: Path) -> dict[str, Any]:
    """Load and validate the compact ``m5_circuit_study_v1`` schema."""

    path = Path(path)
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Study {path} must contain a mapping")
    raw_version = value.get("schema_version", SCHEMA_VERSION)
    if raw_version not in {
        SCHEMA_VERSION,
        LEGACY_SCHEMA_VERSION,
        STUDY_SCHEMA_VERSION,
        LEGACY_STUDY_SCHEMA_VERSION,
    }:
        raise ValueError(
            f"Study {path} must use schema_version: {STUDY_SCHEMA_VERSION}"
        )

    defaults = value.get("defaults") or {}
    if not isinstance(defaults, dict):
        raise ValueError("defaults must be a mapping")
    cases = _named_entries(value, "cases", required=True)
    planners = _named_entries(value, "planner_variants", required=True)
    policies = _named_entries(value, "numeric_policies", required=True)
    engines = _named_entries(
        value, "engine_variants", aliases=("engines",), required=True
    )

    normalized_cases = []
    for entry in cases:
        circuit = entry.get("circuit")
        if not isinstance(circuit, dict) or not circuit.get("name"):
            raise ValueError(f"Case {entry['id']} must define circuit.name")
        normalized_cases.append({**entry, "case_id": str(entry["id"])})

    normalized_planners = []
    for entry in planners:
        config = entry.get("planner", entry.get("config"))
        if config is None:
            config = {
                key: val for key, val in entry.items() if key not in {"id", "label"}
            }
        if not isinstance(config, dict) or not config:
            raise ValueError(f"Planner variant {entry['id']} must define planner")
        normalized_planners.append(
            {
                "id": str(entry["id"]),
                "label": entry.get("label", str(entry["id"])),
                "planner": dict(config),
            }
        )

    normalized_policies = []
    for entry in policies:
        allowed_policy_keys = {"id", "label", "policy", "name"}
        unknown_policy_keys = sorted(set(entry) - allowed_policy_keys)
        if unknown_policy_keys:
            raise ValueError(
                f"Numeric policy {entry['id']} has unsupported keys: "
                f"{', '.join(unknown_policy_keys)}"
            )
        policy_name = str(entry.get("policy", entry.get("name", entry["id"])))
        policy_name = {
            "float32": "float32_real",
            "int8": "host_packed_int8",
            "host_packed_int8_per_task_v1": "host_packed_int8",
        }.get(policy_name, policy_name)
        if policy_name not in {"float32_real", "host_packed_int8"}:
            raise ValueError(f"Unsupported numeric policy {policy_name!r}")
        normalized_policies.append(
            {
                "id": str(entry["id"]),
                "policy": policy_name,
                "label": entry.get("label", str(entry["id"])),
            }
        )

    normalized_engines = []
    for entry in engines:
        engine_id = str(entry.get("engine", entry.get("engine_id", entry["id"])))
        topology = _normalize_topology(entry.get("topology") or entry)
        timeout_enforcement = str(
            entry.get(
                "timeout_enforcement",
                "posthoc_observation" if topology["backend"] == "cpu" else "",
            )
        )
        if timeout_enforcement not in {"engine_subprocess", "posthoc_observation"}:
            raise ValueError(
                f"Engine variant {entry['id']} must declare timeout_enforcement"
            )
        if topology["backend"] != "cpu" and timeout_enforcement != "engine_subprocess":
            raise ValueError(
                f"Physical engine variant {entry['id']} must enforce timeout in its subprocess"
            )
        normalized_engines.append(
            {
                "id": str(entry["id"]),
                "engine": engine_id,
                "label": entry.get("label", str(entry["id"])),
                "topology": topology,
                "timeout_enforcement": timeout_enforcement,
                "executor_config": _executor_config_fields(entry),
            }
        )

    limits = defaults.get("resource_limits", value.get("resource_limits", {})) or {}
    if not isinstance(limits, dict):
        raise ValueError("resource_limits must be a mapping")
    limits = {
        "max_live_bytes": int(limits.get("max_live_bytes", DEFAULT_MAX_LIVE_BYTES)),
        "max_output_bytes": int(limits.get("max_output_bytes", 0)),
        "element_bytes": int(limits.get("element_bytes", 4)),
        "max_tasks": int(limits.get("max_tasks", 0)),
    }
    if limits["max_live_bytes"] < 1 or limits["element_bytes"] < 1:
        raise ValueError(
            "resource_limits.max_live_bytes and element_bytes must be positive"
        )

    result = {
        "schema_version": STUDY_SCHEMA_VERSION,
        "study_id": str(value.get("study_id") or path.stem),
        "metadata": dict(value.get("metadata") or {}),
        "cases": normalized_cases,
        "planner_variants": normalized_planners,
        "numeric_policies": normalized_policies,
        "engine_variants": normalized_engines,
        "warmups": int(value.get("warmups", defaults.get("warmups", 0))),
        "repeats": int(value.get("repeats", defaults.get("repeats", 1))),
        "timeout_s": float(
            value.get("timeout_s", defaults.get("timeout_s", DEFAULT_TIMEOUT_S))
        ),
        "resource_limits": limits,
        "tolerances": {**DEFAULT_TOLERANCES, **dict(value.get("tolerances") or {})},
        "_study_path": str(path.resolve()),
    }
    if result["warmups"] < 0 or result["repeats"] < 1 or result["timeout_s"] <= 0:
        raise ValueError(
            "warmups must be >= 0, repeats >= 1, and timeout_s must be positive"
        )
    routes, comparisons = build_pipeline_catalog(result)
    result["pipeline_routes"] = routes
    result["pipeline_comparisons"] = comparisons
    return result


def plan_study(
    root: Path, path: Path, *, route_ids: Iterable[str] | None = None
) -> Path:
    """Resolve every circuit/planner combination without opening an engine."""

    root = Path(root).resolve()
    config = load_study_config(Path(path))
    selected_routes = select_pipeline_routes(config, route_ids)
    config["selected_route_ids"] = [route["route_id"] for route in selected_routes]
    plans = _build_plans(root, config, config["selected_route_ids"])
    plan_parent = root / "build" / "m5_circuit_study_plan"
    plan_parent.mkdir(parents=True, exist_ok=True)
    plan_dir = plan_parent / _timestamp()
    suffix = 1
    while plan_dir.exists():
        plan_dir = plan_parent / f"{_timestamp()}_{suffix:02d}"
        suffix += 1
    plan_dir.mkdir(parents=True, exist_ok=False)
    payload = _plan_manifest(
        config,
        plans,
        plan_dir,
        hardware_opened=False,
        selected_route_ids=config["selected_route_ids"],
    )
    write_json(plan_dir / "resolved_study.yml.json", config)
    write_json(plan_dir / "resolved_plan.json", payload)
    return plan_dir / "resolved_plan.json"


def run_study(
    root: Path,
    path: Path,
    *,
    engine_factories: Mapping[str, Any] | None = None,
    rank_paths: list[str] | None = None,
    route_ids: Iterable[str] | None = None,
) -> Path:
    """Run all declared combinations and persist normalized study evidence."""

    root = Path(root).resolve()
    config = load_study_config(Path(path))
    if rank_paths is not None:
        config = apply_rank_path_override(config, rank_paths)
    selected_routes = select_pipeline_routes(config, route_ids)
    config["selected_route_ids"] = [route["route_id"] for route in selected_routes]
    selected_comparisons = selected_pipeline_comparisons(
        config, set(config["selected_route_ids"])
    )
    plans = _build_plans(root, config, config["selected_route_ids"])
    run_dir = create_run_dir(
        root,
        config["study_id"],
        artifact_kind=EVIDENCE_ARTIFACT_KIND,
        route_label=ROUTE_LABEL,
    )
    write_run_manifest(
        run_dir,
        run_kind=STUDY_SCHEMA_VERSION,
        suite_id=config["study_id"],
        suite_path=config["_study_path"],
        artifact_kind=EVIDENCE_ARTIFACT_KIND,
        route_label=ROUTE_LABEL,
        execution_scope="whole_circuit_contraction_dag",
        evidence_type="benchmark_execution",
        normalized_records="normalized_records.jsonl",
        summary="m5_circuit_study_summary.json",
        root_dir=root,
    )
    write_json(run_dir / "config" / "resolved_study.json", config)
    write_json(
        run_dir / "plan_manifest.json",
        _plan_manifest(
            config,
            plans,
            run_dir,
            hardware_opened=False,
            selected_route_ids=config["selected_route_ids"],
        ),
    )

    factories = dict(engine_factories or {})
    engines_by_id = {item["id"]: item for item in config["engine_variants"]}
    policies_by_id = {item["id"]: item for item in config["numeric_policies"]}
    selected_route_ids_set = set(config["selected_route_ids"])
    records: list[dict[str, Any]] = []
    warmup_failures: list[dict[str, Any]] = []
    for planned in plans:
        case = planned.case
        planned_routes = [
            route
            for route in selected_routes
            if route["planner_id"] == planned.planner["id"]
        ]
        anchor: dict[str, Any] | None = None
        policy_anchors: dict[str, dict[str, Any]] = {}
        if planned.preflight["status"] == "supported":
            anchor = _run_anchor(planned, config, Float32RealPolicy())
            declared_policies = {
                str(policies_by_id[route["numeric_policy_id"]]["policy"])
                for route in planned_routes
            }
            for policy_name, policy in (
                ("float32_real", Float32RealPolicy()),
                ("host_packed_int8", HostPackedInt8Policy()),
            ):
                if policy_name not in declared_policies:
                    continue
                policy_anchors[policy_name] = (
                    anchor
                    if policy_name == "float32_real"
                    else _run_anchor(planned, config, policy)
                )

        for route in planned_routes:
            engine_variant = engines_by_id[route["engine_id"]]
            policy_variant = policies_by_id[route["numeric_policy_id"]]
            comparison_ids = route_comparison_ids(
                config,
                route["route_id"],
                selected_route_ids_set,
            )
            compilation_started = time.perf_counter()
            try:
                execution_plan = _compile_route_plan(
                    planned,
                    str(engine_variant["topology"]["backend"]),
                    _numeric_mode_from_policy(_policy(str(policy_variant["policy"]))),
                    engine_variant,
                )
                compilation_time_s = time.perf_counter() - compilation_started
            except Exception as exc:
                compilation_time_s = time.perf_counter() - compilation_started
                for repeat_id in range(config["repeats"]):
                    row_base = _row_base(
                        config, planned, engine_variant, policy_variant, route,
                        comparison_ids, repeat_id, anchor,
                    )
                    records.append(
                        {
                            **row_base,
                            "status": "failed",
                            "support_status": "failed",
                            "failure_stage": "execution_plan_compilation_failed",
                            "error": str(exc),
                            "compilation_time_s": compilation_time_s,
                        }
                    )
                continue
            for repeat_id in range(config["repeats"]):
                row_base = _row_base(
                    config,
                    planned,
                    engine_variant,
                    policy_variant,
                    route,
                    comparison_ids,
                    repeat_id,
                    anchor,
                )
                if planned.preflight["status"] != "supported":
                    records.append(
                        {
                            **row_base,
                            "status": "unsupported",
                            "support_status": "unsupported",
                            "failure_stage": "preflight_resource_limit",
                            "error": planned.preflight["reason"],
                            "repeat_count": 0,
                            "compilation_time_s": compilation_time_s,
                        }
                    )
                    break
                if isinstance(execution_plan, UnsupportedExecution):
                    records.append(
                        {
                            **row_base,
                            "status": "unsupported",
                            "support_status": "unsupported",
                            "failure_stage": "execution_plan_unsupported",
                            "error": execution_plan.reason,
                            "execution_target": execution_plan.target.value,
                            "compilation_time_s": compilation_time_s,
                        }
                    )
                    break
                anchor_failure = _anchor_failure(
                    anchor=anchor,
                    policy_anchor=policy_anchors.get(str(policy_variant["policy"])),
                )
                if anchor_failure is not None:
                    records.append(
                        {
                            **row_base,
                            "status": "failed",
                            "support_status": "failed",
                            "failure_stage": anchor_failure[0],
                            "error": anchor_failure[1],
                        }
                    )
                    continue
                if repeat_id == 0:
                    for warmup_id in range(config["warmups"]):
                        try:
                            _execute_combo(
                                planned,
                                config,
                                route,
                                engine_variant,
                                policy_variant,
                                factories,
                                execution_plan,
                                compilation_time_s,
                                warmup_id,
                                anchor,
                                policy_anchors.get(str(policy_variant["policy"])),
                            )
                        except Exception as exc:
                            warmup_failures.append(
                                {
                                    "case_id": case["case_id"],
                                    "planner_id": planned.planner["id"],
                                    "engine_id": engine_variant["id"],
                                    "numeric_policy_id": policy_variant["id"],
                                    "route_id": route["route_id"],
                                    "warmup_id": warmup_id,
                                    "error": str(exc),
                                }
                            )
                try:
                    executed = _execute_combo(
                        planned,
                        config,
                        route,
                        engine_variant,
                        policy_variant,
                        factories,
                        execution_plan,
                        compilation_time_s,
                        repeat_id,
                        anchor,
                        policy_anchors.get(str(policy_variant["policy"])),
                    )
                    records.append({**row_base, **executed})
                except Exception as exc:
                    adapter_validation = getattr(exc, "details", None)
                    records.append(
                        {
                            **row_base,
                            "status": "failed",
                            "support_status": "failed",
                            "failure_stage": _failure_stage(exc),
                            "error": str(exc),
                            "compilation_time_s": compilation_time_s,
                            "no_fallback_used": True,
                            "route_adapter_validation": adapter_validation,
                        }
                    )
        # Do not retain full state arrays across planner/case iterations.
        anchor = None
        policy_anchors.clear()

    write_normalized_records(run_dir, records)
    summary = {
        "schema_version": STUDY_SCHEMA_VERSION,
        "study_id": config["study_id"],
        "status": "completed"
        if not any(row["status"] == "failed" for row in records) and not warmup_failures
        else "failed",
        "record_count": len(records),
        "completed_count": sum(row["status"] == "completed" for row in records),
        "unsupported_count": sum(row["status"] == "unsupported" for row in records),
        "failed_count": sum(row["status"] == "failed" for row in records),
        "warmup_failures": warmup_failures,
        "selected_route_ids": config["selected_route_ids"],
        "pipeline_routes": selected_routes,
        "pipeline_comparisons": selected_comparisons,
        "normalized_records": "normalized_records.jsonl",
        "plan_manifest": "plan_manifest.json",
        "energy": {
            "status": "unavailable",
            "source": None,
            "reason": "No energy provider is part of M5.5",
        },
    }
    write_json(run_dir / "m5_circuit_study_summary.json", summary)
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    manifest.update(
        {
            "study_schema_version": STUDY_SCHEMA_VERSION,
            "hardware_opened": any(
                bool(row.get("hardware_execution_attempted"))
                or bool(row.get("hardware_execution_verified"))
                or bool(row.get("hardware_allocation_verified"))
                or bool(row.get("native_kernel_executed"))
                for row in records
            ),
            "rank_paths_resolved": list(config.get("_rank_paths_resolved", [])),
            "energy_status": "unavailable",
            "selected_route_ids": config["selected_route_ids"],
            "pipeline_routes": selected_routes,
            "pipeline_comparisons": selected_comparisons,
        }
    )
    write_json(run_dir / "run_manifest.json", manifest)
    return run_dir


def _build_plans(
    root: Path, config: dict[str, Any], selected_route_ids: Iterable[str] | None = None
) -> list[_Plan]:
    plans: list[_Plan] = []
    selected_planner_ids: set[str] | None = None
    if selected_route_ids is not None:
        selected_planner_ids = {
            route["planner_id"]
            for route in select_pipeline_routes(config, selected_route_ids)
        }
    for case in config["cases"]:
        circuit = load_circuit(case, root)
        spec, inputs = build_tensor_network_data(circuit)
        for planner in config["planner_variants"]:
            if (
                selected_planner_ids is not None
                and planner["id"] not in selected_planner_ids
            ):
                continue
            request = PlannerRequest.from_config(planner["planner"])
            planner_result = plan_contractions(spec, request)
            dag = build_contraction_dag(spec, planner_result.path)
            dag_hash = contraction_dag_hash(dag)
            circuit_hash = _circuit_semantics_hash(circuit)
            network_hash = _tensor_network_hash(spec, circuit_hash)
            resources = _estimate_resources(dag, config["resource_limits"])
            preflight = _preflight(resources, config["resource_limits"])
            plans.append(
                _Plan(
                    case,
                    planner,
                    circuit,
                    spec,
                    inputs,
                    planner_result,
                    dag,
                    dag_hash,
                    circuit_hash,
                    network_hash,
                    resources,
                    preflight,
                )
            )
    return plans


def _estimate_resources(dag: ContractionDAG, limits: dict[str, int]) -> dict[str, Any]:
    """Estimate configured execution storage while caller inputs remain live."""

    element_bytes = limits["element_bytes"]
    initial_tensor_ids = {spec.id for spec in dag.tensors}
    sizes = {
        spec.id: int(np.prod(spec.shape, dtype=np.int64)) * element_bytes
        for spec in dag.tensors
    }
    remaining: dict[str, int] = {}
    ordered_nodes = _topological_dag_nodes(dag)
    for node in ordered_nodes:
        for view in _node_views(node):
            remaining[view.tensor_id] = remaining.get(view.tensor_id, 0) + 1
    live = sum(sizes.values())
    peak = live
    for node in ordered_nodes:
        output_bytes = _tensor_bytes(node.output.shape, element_bytes)
        sizes[node.output.id] = output_bytes
        live += output_bytes
        peak = max(peak, live)
        for view in _node_views(node):
            remaining[view.tensor_id] -= 1
            if (
                remaining[view.tensor_id] == 0
                and view.tensor_id not in initial_tensor_ids
            ):
                live -= sizes[view.tensor_id]
    final_node = _final_dag_node(dag)
    if final_node is None and dag.nodes:
        raise ValueError("DAG output must be produced by exactly one final node")
    final_bytes = _tensor_bytes(dag.output.shape, element_bytes)
    return {
        "final_output_bytes": final_bytes,
        "peak_live_bytes": peak,
        "initial_tensor_bytes": sum(
            _tensor_bytes(spec.shape, element_bytes) for spec in dag.tensors
        ),
        "initial_host_input_bytes": sum(
            _tensor_bytes(spec.shape, np.dtype(spec.dtype).itemsize)
            for spec in dag.tensors
        ),
        "byte_accounting_basis": "configured_execution_element_width",
        "configured_execution_element_bytes": element_bytes,
        "dag_node_count": len(dag.nodes),
        # Compatibility alias for report consumers; this is a DAG-node count.
        "task_count": len(dag.nodes),
        # Compatibility alias for the configured execution element width.
        "element_bytes": element_bytes,
    }


def _node_views(node: ContractNode | ReduceNode) -> tuple[TensorView, ...]:
    if isinstance(node, ContractNode):
        return node.left, node.right
    return node.inputs


def _tensor_bytes(shape: tuple[int, ...], element_bytes: int) -> int:
    return int(np.prod(shape, dtype=np.int64)) * element_bytes


def _topological_dag_nodes(dag: ContractionDAG) -> tuple[ContractNode | ReduceNode, ...]:
    """Return a deterministic dependency order without trusting ``dag.nodes`` order."""

    nodes = {node.node_id: node for node in dag.nodes}
    if len(nodes) != len(dag.nodes):
        raise ValueError("ContractionDAG contains duplicate node IDs")
    indegree = {node_id: len(node.dependencies) for node_id, node in nodes.items()}
    dependants: dict[str, list[str]] = {node_id: [] for node_id in nodes}
    for node in dag.nodes:
        for dependency in node.dependencies:
            if dependency not in nodes:
                raise ValueError(
                    f"DAG node {node.node_id} has unknown dependency {dependency}"
                )
            dependants[dependency].append(node.node_id)

    ready = sorted(node_id for node_id, count in indegree.items() if count == 0)
    ordered: list[ContractNode | ReduceNode] = []
    while ready:
        node_id = ready.pop(0)
        ordered.append(nodes[node_id])
        for dependant in sorted(dependants[node_id]):
            indegree[dependant] -= 1
            if indegree[dependant] == 0:
                ready.append(dependant)
        ready.sort()
    if len(ordered) != len(dag.nodes):
        raise ValueError("ContractionDAG dependencies contain a cycle")
    return tuple(ordered)


def _preflight(resources: dict[str, Any], limits: dict[str, int]) -> dict[str, Any]:
    reasons = []
    if resources["peak_live_bytes"] > limits["max_live_bytes"]:
        reasons.append(
            f"peak_live_bytes={resources['peak_live_bytes']} exceeds max_live_bytes={limits['max_live_bytes']}"
        )
    if (
        limits["max_output_bytes"]
        and resources["final_output_bytes"] > limits["max_output_bytes"]
    ):
        reasons.append(
            f"final_output_bytes={resources['final_output_bytes']} exceeds max_output_bytes={limits['max_output_bytes']}"
        )
    if limits["max_tasks"] and resources["task_count"] > limits["max_tasks"]:
        reasons.append(
            f"task_count={resources['task_count']} exceeds max_tasks={limits['max_tasks']}"
        )
    return {
        "status": "unsupported" if reasons else "supported",
        "reason": "; ".join(reasons) if reasons else None,
    }


def _run_anchor(planned: _Plan, config: dict[str, Any], policy: Any) -> dict[str, Any]:
    start = time.perf_counter()
    try:
        numeric_mode = _numeric_mode_from_policy(policy)
        compiled = compile_execution(
            planned.dag,
            CpuCompileRequest(
                contraction_dag_hash=planned.dag_hash,
                numeric_mode=numeric_mode,
            ),
        )
        if isinstance(compiled, UnsupportedExecution):
            raise RuntimeError(f"CPU anchor unsupported: {compiled.reason}")
        result = execute(
            compiled,
            planned.dag,
            planned.inputs,
            RunContext(run_id="m5_cpu_anchor", target=Target.CPU),
        )
        if isinstance(result, ExecutionFailure):
            raise RuntimeError(f"CPU anchor failed: {result.stage}: {result.reason}")
    except Exception as exc:
        return {
            "output": None,
            "metadata": {},
            "time_s": time.perf_counter() - start,
            "status": "failed",
            "error": str(exc),
        }
    return {
        "output": result.output,
        "metadata": _execution_metadata(result, None, None),
        "time_s": time.perf_counter() - start,
        "reference_s": result.timing.route_total_s,
        "status": "available",
    }


def _anchor_failure(
    *, anchor: dict[str, Any] | None, policy_anchor: dict[str, Any] | None
) -> tuple[str, str] | None:
    if anchor is None or anchor.get("output") is None:
        return (
            "anchor_generation_failed",
            str((anchor or {}).get("error", "CPU float32 anchor unavailable")),
        )
    if policy_anchor is None or policy_anchor.get("output") is None:
        return (
            "reference_unavailable",
            str((policy_anchor or {}).get("error", "CPU policy reference unavailable")),
        )
    return None


def _execute_combo(
    planned: _Plan,
    config: dict[str, Any],
    route: Mapping[str, Any],
    engine_variant: dict[str, Any],
    policy_variant: dict[str, Any],
    factories: dict[str, Any],
    execution_plan: Any,
    compilation_time_s: float,
    repeat_id: int,
    anchor: dict[str, Any] | None,
    policy_anchor: dict[str, Any] | None,
) -> dict[str, Any]:
    adapter_validation = validate_route_adapter(
        route, planned.planner, policy_variant, engine_variant
    )
    topology = _legacy_topology(engine_variant["topology"])
    policy = _policy(str(policy_variant["policy"]))
    backend = str(engine_variant["topology"]["backend"])
    if isinstance(execution_plan, UnsupportedExecution):
        raise RuntimeError("unsupported execution plans must be handled before execution")

    resources = None
    if execution_plan.target is Target.UPMEM:
        engine = _resolve_engine(
            engine_variant, topology, factories, timeout_s=config["timeout_s"]
        )
        resources = _upmem_runtime_resources(
            engine,
            topology,
            execution_plan,
            expected_rank_paths=tuple(
                str(path) for path in engine_variant["topology"]["rank_paths"]
            ),
            timeout_s=config["timeout_s"],
        )
    result = execute(
        execution_plan,
        planned.dag,
        planned.inputs,
        RunContext(
            run_id=f"{planned.case['case_id']}:{route['route_id']}:{repeat_id}",
            target=execution_plan.target,
            timeout_s=config["timeout_s"],
            target_resources=resources,
        ),
    )
    if isinstance(result, ExecutionFailure):
        raise RuntimeError(f"{result.stage}: {result.reason}")
    elapsed = result.timing.route_total_s
    if elapsed is not None and elapsed > config["timeout_s"]:
        raise TimeoutError(
            f"whole-circuit route exceeded timeout_s={config['timeout_s']}"
        )
    output = np.asarray(result.output)
    metadata = _execution_metadata(result, execution_plan, engine_variant)
    expected_policy = policy_anchor["output"] if policy_anchor else None
    expected_full = anchor["output"] if anchor else None
    policy_validation = _validation(
        output,
        expected_policy,
        config["tolerances"],
        quantized=_is_quantized_policy(policy),
    )
    full_validation = _validation(
        output,
        expected_full,
        config["tolerances"],
        quantized=_is_quantized_policy(policy),
    )
    failed_validation = (
        full_validation["status"] == "failed" or policy_validation["status"] == "failed"
    )
    engine_metadata = _engine_metadata(metadata)
    hardware = _hardware_contract(engine_metadata, engine_variant["topology"])
    hardware_failure = backend != "cpu" and not hardware["verified"]
    route_observation = admit_route_observation(
        route, engine_metadata, backend=backend
    )
    route_observation_failure = not bool(route_observation["passed"])
    output_hash = _array_hash(output)
    executor_hash = _executor_config_hash(
        engine_variant, policy, topology, engine_metadata
    )
    timing = _timing_metadata(result)
    transfer = _transfer_metadata(result)
    h2d_bytes = result.h2d_bytes
    d2h_bytes = result.d2h_bytes
    transfer_bytes = result.transfer_bytes
    transfer_accounting_verified = bool(
        h2d_bytes is not None
        and d2h_bytes is not None
        and transfer_bytes is not None
        and h2d_bytes >= 0
        and d2h_bytes >= 0
        and transfer_bytes == h2d_bytes + d2h_bytes
    )
    measurement_repetitions_sufficient = (
        config["warmups"] >= 1 and config["repeats"] >= 3
    )
    timing_is_bringup_only = not measurement_repetitions_sufficient
    host_dag_node_completion_coverage = set(result.executed_node_ids) == {
        node.node_id for node in planned.dag.nodes
    } and len(result.executed_node_ids) == len(planned.dag.nodes)
    timing_breakdown = {
        key: value
        for key, value in {
            "preparation_time_s": result.timing.preparation_s,
            "h2d_time_s": result.timing.h2d_s,
            "kernel_time_s": result.timing.kernel_s,
            "d2h_time_s": result.timing.d2h_s,
            "host_quantization_time_s": result.timing.host_quantization_s,
            "host_dequantization_time_s": result.timing.host_dequantization_s,
            "reduction_time_s": result.timing.reduction_s,
            "session_open_s": result.timing.session_open_s,
            "session_close_s": result.timing.session_close_s,
            "graph_execution_s": _graph_execution_time(result),
        }.items()
        if value is not None
    }
    physical_timing_complete = _physical_timing_complete(timing_breakdown)
    hardware_speedup_applicable = bool(
        backend != "cpu"
        and hardware["verified"]
        and measurement_repetitions_sufficient
        and not failed_validation
        and host_dag_node_completion_coverage
        and transfer_accounting_verified
        and physical_timing_complete
        and not route_observation_failure
        and not bool(engine_metadata.get("cpu_fallback_used", False))
        and not bool(engine_metadata.get("simulator_kernel_executed", False))
    )
    return {
        "status": "failed"
        if failed_validation or hardware_failure or route_observation_failure
        else "completed",
        "support_status": "supported",
        "failure_stage": (
            "route_observation_mismatch"
            if route_observation_failure
            else hardware["failure_stage"]
            if hardware_failure
            else "output_validation_failed"
            if failed_validation
            else None
        ),
        "error": (
            route_observation["reason"]
            if route_observation_failure
            else hardware["reason"]
            if hardware_failure
            else "output validation failed"
            if failed_validation
            else None
        ),
        "planning_time_s": planned.planner_result.planning_time_s,
        "compilation_time_s": compilation_time_s,
        "whole_route_including_session_lifecycle_s": result.timing.route_total_s,
        "graph_execution_s": timing_breakdown.get("graph_execution_s"),
        "session_open_s": timing_breakdown.get("session_open_s"),
        "session_close_s": timing_breakdown.get("session_close_s"),
        "outer_elapsed_s": result.timing.route_total_s,
        "total_route_time_s": result.timing.route_total_s,
        "timing_scope": "whole_route_including_session_lifecycle",
        "timeout_enforcement": engine_variant["timeout_enforcement"],
        "executor_config_hash": executor_hash,
        "timing": timing,
        "timing_breakdown": timing_breakdown,
        "preparation_time_s": timing_breakdown.get("preparation_time_s"),
        "h2d_time_s": timing_breakdown.get("h2d_time_s"),
        "kernel_time_s": timing_breakdown.get("kernel_time_s"),
        "d2h_time_s": timing_breakdown.get("d2h_time_s"),
        "host_quantization_time_s": timing_breakdown.get("host_quantization_time_s"),
        "host_dequantization_time_s": timing_breakdown.get(
            "host_dequantization_time_s"
        ),
        "validation_status": full_validation["status"],
        "scientific_validation_status": full_validation["status"],
        "policy_reference_validation": policy_validation,
        "full_precision_accuracy": full_validation,
        "output_shape": list(output.shape),
        "output_dtype": str(output.dtype),
        "output_sha256": output_hash,
        "output_hash": output_hash,
        "output_norm": float(np.linalg.norm(output.ravel())),
        "validation_max_abs_error": full_validation.get("max_abs_error"),
        "validation_l2_error": full_validation.get("l2_error"),
        "route_adapter_validation": adapter_validation,
        "route_observation_admission": route_observation,
        "validation_norm_drift": full_validation.get("norm_drift"),
        "task_metrics": (),
        "complete_task_count": len(planned.dag.nodes),
        "executed_task_count": len(result.executed_node_ids),
        # Backward-compatible field: this records host-coordinator DAG-node
        # completion coverage, not native-kernel exactly-once proof.
        "exact_once": host_dag_node_completion_coverage,
        "exact_once_scope": "host_dag_node_completion_per_route",
        "host_dag_node_completion_coverage": host_dag_node_completion_coverage,
        "cpu_fallback_used": bool(engine_metadata.get("cpu_fallback_used", False)),
        "simulator_kernel_executed": bool(
            engine_metadata.get("simulator_kernel_executed", False)
        ),
        "no_fallback_used": not bool(
            engine_metadata.get("cpu_fallback_used", False)
            or engine_metadata.get("simulator_kernel_executed", False)
        ),
        "hardware_execution_attempted": bool(
            backend != "cpu"
            and (
                hardware["native_kernel_executed"]
                or hardware["allocation_verified"]
                or hardware["release_verified"]
            )
        ),
        "hardware_execution_verified": hardware["verified"],
        "measurement_repetitions_sufficient": measurement_repetitions_sufficient,
        "timing_is_bringup_only": timing_is_bringup_only,
        "hardware_speedup_applicable": hardware_speedup_applicable,
        "native_kernel_executed": hardware["native_kernel_executed"],
        "hardware_kernel_executed": hardware["hardware_kernel_executed"],
        "hardware_allocation_verified": hardware["allocation_verified"],
        "hardware_release_verified": hardware["release_verified"],
        "target_observed": _target_observed(engine_metadata),
        "native_identity_verified": bool(
            engine_metadata.get("native_identity_verified", False)
        ),
        "physical_plan_consumed": bool(
            engine_metadata.get("physical_plan_consumed", False)
        ),
        "graph_intermediate_placement_origin": engine_metadata.get(
            "graph_intermediate_placement_origin"
        ),
        "observed_rank_count": engine_metadata.get("observed_rank_count"),
        "allocated_dpu_count": engine_metadata.get("allocated_dpu_count"),
        "observed_tasklets_per_dpu": engine_metadata.get(
            "observed_tasklets_per_dpu", engine_metadata.get("tasklets_per_dpu")
        ),
        "rank_binding_sha256": engine_metadata.get("rank_binding_sha256"),
        "application_visible_h2d_bytes": h2d_bytes,
        "application_visible_d2h_bytes": d2h_bytes,
        "application_visible_transfer_bytes": transfer_bytes,
        "transfer_accounting_verified": transfer_accounting_verified,
        "transfer": dict(transfer),
        "engine_metadata": engine_metadata,
        "energy_joules": None,
        "energy_status": "unavailable",
        "execution_plan_hash": execution_plan_hash(execution_plan),
        "execution_plan_schema_version": "tn_execution_plan_v1",
        "execution_target": execution_plan.target.value,
        "contraction_dag_hash": planned.dag_hash,
        "contraction_dag_schema_version": "contraction_dag_v2",
        "planner_config_hash": planned.planner_result.identity.planner_config_hash,
    }


def _numeric_mode_from_policy(policy: Any) -> NumericMode:
    if _is_quantized_policy(policy):
        return NumericMode.HOST_PACKED_INT8_PER_TASK_V1
    if str(getattr(policy, "name", "")) == "float32_real":
        return NumericMode.FLOAT32_REAL
    raise ValueError(f"unsupported M5 numeric policy {getattr(policy, 'name', None)!r}")


def _legacy_topology(topology: Mapping[str, Any]) -> DeviceTopology:
    return DeviceTopology(
        backend=str(topology["backend"]),
        device_ids=tuple(str(value) for value in topology["device_ids"]),
        tasklets_per_device=int(topology["tasklets_per_device"]),
    )


def _compile_route_plan(
    planned: _Plan,
    backend: str,
    numeric_mode: NumericMode,
    engine_variant: Mapping[str, Any],
) -> Any:
    dag_hash = planned.dag_hash
    if backend == "cpu":
        if str(engine_variant["engine"]) != "numpy_cpu":
            raise ValueError(
                "CPU functional routes require engine: numpy_cpu before execution"
            )
        return compile_execution(
            planned.dag,
            CpuCompileRequest(
                contraction_dag_hash=dag_hash,
                numeric_mode=numeric_mode,
                executor_id=str(engine_variant["engine"]),
            ),
        )
    topology = engine_variant["topology"]
    return compile_execution(
        planned.dag,
        UpmemCompileRequest(
            contraction_dag_hash=dag_hash,
            numeric_mode=numeric_mode,
            topology=UpmemTopology(
                dpu_count=len(topology["device_ids"]),
                tasklets_per_dpu=int(topology["tasklets_per_device"]),
                rank_count=len(topology["rank_paths"]),
            ),
        ),
    )


def _upmem_runtime_resources(
    engine: Any,
    topology: DeviceTopology,
    execution_plan: Any,
    *,
    expected_rank_paths: tuple[str, ...],
    timeout_s: float,
) -> UpmemRuntimeResources:
    """Bind a public M5 engine instance to the functional UPMEM adapter.

    Machine-local binary paths and rank device paths deliberately remain in the
    run context.  The compiled plan contains only logical topology and v4
    execution choices.
    """

    required = (
        "session_root",
        "host_binary",
        "dpu_binary",
        "initialization_binary",
        "rank_paths",
    )
    missing = [name for name in required if not getattr(engine, name, None)]
    if missing:
        raise RuntimeError(
            "injected M5 engine does not expose required runtime resources: "
            + ", ".join(missing)
        )
    if not expected_rank_paths:
        raise RuntimeError("M5 runtime binding requires suite-resolved rank paths")
    actual_rank_paths = tuple(str(path) for path in engine.rank_paths)
    if actual_rank_paths != expected_rank_paths:
        raise RuntimeError(
            "injected M5 engine rank_paths do not match suite-resolved topology"
        )

    def open_session(_plan: Any, _context: RunContext) -> Any:
        return engine.open_session(
            _policy_for_mode(execution_plan.payload.numeric_mode), topology
        )

    return UpmemRuntimeResources(
        session_root=str(engine.session_root),
        host_binary=str(engine.host_binary),
        dpu_binary=str(engine.dpu_binary),
        initialization_binary=str(engine.initialization_binary),
        rank_paths=actual_rank_paths,
        session_opener=open_session,
    )


def _policy_for_mode(mode: NumericMode) -> Any:
    return (
        HostPackedInt8Policy()
        if mode is NumericMode.HOST_PACKED_INT8_PER_TASK_V1
        else Float32RealPolicy()
    )


def _execution_metadata(
    result: ExecutionResult,
    execution_plan: Any | None,
    engine_variant: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Translate the narrow functional result into the historic M5 row surface."""

    if result.target is Target.CPU:
        return {
            "execution_engine": "numpy_cpu",
            "engine": "numpy_cpu",
            "task_metrics": (),
            "executed_order": result.executed_node_ids,
        }
    facts = result.backend_facts
    if facts is None or execution_plan is None:
        raise RuntimeError("UPMEM execution result is missing backend facts")
    return {
        "backend_id": facts.backend_id,
        "physical_profile": facts.profile_id,
        "profile": facts.profile_id,
        "abi": facts.abi_id,
        "abi_version": facts.abi_id,
        "session_protocol": facts.session_id,
        "dispatch_mode": facts.dispatch_id,
        "kernel_identity": facts.kernel_id,
        "execution_class": facts.execution_class,
        "graph_intermediate_placement": facts.intermediate_placement,
        "graph_intermediate_placement_origin": facts.intermediate_placement_origin,
        "native_identity_verified": facts.native_identity_verified,
        "target_observed": facts.target_observed,
        "hardware_allocation_verified": facts.hardware_allocation_verified,
        "hardware_release_verified": facts.hardware_release_verified,
        "hardware_release_confirmed": facts.hardware_release_confirmed,
        "requested_dpu_count": facts.requested_dpu_count,
        "allocated_dpu_count": facts.allocated_dpu_count,
        "observed_rank_count": facts.observed_rank_count,
        "tasklets_per_dpu": facts.tasklets_per_dpu,
        "observed_tasklets_per_dpu": facts.tasklets_per_dpu,
        "native_kernel_executed": facts.native_kernel_executed,
        "hardware_kernel_executed": facts.hardware_kernel_executed,
        "simulator_kernel_executed": facts.simulator_kernel_executed,
        "cpu_fallback_used": facts.cpu_fallback_used,
        "physical_plan_consumed": facts.physical_plan_consumed,
        "host_binary_sha256": facts.host_binary_sha256,
        "dpu_binary_sha256": facts.dpu_binary_sha256,
        "initialization_binary_sha256": facts.initialization_binary_sha256,
        "rank_binding_sha256": facts.rank_binding_sha256,
        "application_visible_h2d_bytes": result.h2d_bytes,
        "application_visible_d2h_bytes": result.d2h_bytes,
        "application_visible_transfer_bytes": result.transfer_bytes,
        "h2d_bytes": result.h2d_bytes,
        "d2h_bytes": result.d2h_bytes,
        "transfer_bytes": result.transfer_bytes,
        "task_metrics": (),
        "executed_order": result.executed_node_ids,
    }


def _timing_metadata(result: ExecutionResult) -> dict[str, float]:
    timing = result.timing
    return {
        key: value
        for key, value in {
            "total_s": timing.route_total_s,
            "preparation_time_s": timing.preparation_s,
            "h2d_time_s": timing.h2d_s,
            "kernel_time_s": timing.kernel_s,
            "d2h_time_s": timing.d2h_s,
            "host_quantization_time_s": timing.host_quantization_s,
            "host_dequantization_time_s": timing.host_dequantization_s,
            "reduction_time_s": timing.reduction_s,
            "session_open_s": timing.session_open_s,
            "session_close_s": timing.session_close_s,
        }.items()
        if value is not None
    }


def _transfer_metadata(result: ExecutionResult) -> dict[str, int]:
    values = {
        "application_visible_h2d_bytes": result.h2d_bytes,
        "h2d_bytes": result.h2d_bytes,
        "application_visible_d2h_bytes": result.d2h_bytes,
        "d2h_bytes": result.d2h_bytes,
        "application_visible_transfer_bytes": result.transfer_bytes,
        "transfer_bytes": result.transfer_bytes,
    }
    return {key: value for key, value in values.items() if value is not None}


def _graph_execution_time(result: ExecutionResult) -> float | None:
    total = result.timing.route_total_s
    if total is None:
        return None
    return max(
        0.0,
        total
        - float(result.timing.session_open_s or 0.0)
        - float(result.timing.session_close_s or 0.0),
    )


def _row_base(
    config: dict[str, Any],
    planned: _Plan,
    engine: dict[str, Any],
    policy: dict[str, Any],
    route: Mapping[str, Any],
    comparison_ids: list[str],
    repeat_id: int,
    anchor: dict[str, Any] | None,
) -> dict[str, Any]:
    circuit = planned.circuit
    cm = circuit_manifest(circuit)
    topology = engine["topology"]
    return {
        "normalized_record_schema_version": STUDY_RECORD_SCHEMA_VERSION,
        "study_id": config["study_id"],
        "case_id": planned.case["case_id"],
        "family": planned.case.get("family", circuit.source.get("name", circuit.name)),
        "circuit_family": planned.case.get(
            "family", circuit.source.get("name", circuit.name)
        ),
        "qubits": int(circuit.n_qubits),
        "n_qubits": int(circuit.n_qubits),
        "depth": int(
            planned.case.get("depth", cm.get("depth_proxy", len(circuit.operations)))
        ),
        "planner_id": planned.planner["id"],
        "planner_config": planned.planner["planner"],
        "planner_config_hash": planned.planner_result.identity.planner_config_hash,
        "planner_hash": planned.planner_result.identity.planner_config_hash,
        "route_id": route["route_id"],
        "route_label": route["label"],
        "route_config_hash": route["route_config_hash"],
        "route_modules": route["modules"],
        "comparison_ids": comparison_ids,
        "engine_id": engine["id"],
        "engine": engine["engine"],
        "backend_id": engine["engine"],
        "executor_config_hash": _executor_config_hash(
            engine,
            _policy(str(policy["policy"])),
            DeviceTopology(
                backend=topology["backend"],
                device_ids=tuple(topology["device_ids"]),
                tasklets_per_device=int(topology["tasklets_per_device"]),
            ),
            {},
        ),
        "backend_family": topology["backend"],
        "execution_class": (
            "physical_hardware" if topology["backend"] != "cpu" else "cpu_diagnostic"
        ),
        "target_requested": topology["backend"],
        "target_observed": None,
        "observed_rank_count": None,
        "allocated_dpu_count": None,
        "observed_tasklets_per_dpu": None,
        "numeric_policy_id": policy["id"],
        "numeric_policy": policy["policy"],
        "topology": topology,
        "rank_paths": topology.get("rank_paths", []),
        "rank_count": len(topology.get("rank_paths", [])),
        "dpu_count": len(topology.get("device_ids", [])),
        "tasklets_per_dpu": topology.get("tasklets_per_device", 1),
        "circuit_semantics_hash": planned.circuit_semantics_hash,
        "tensor_network_hash": planned.tensor_network_hash,
        # Compatibility aliases: historical plan fields now carry the
        # canonical semantic ContractionDAG identity.
        "contraction_plan_hash": planned.dag_hash,
        "contraction_path_structure_hash": planned.dag_hash,
        "contraction_dag_hash": planned.dag_hash,
        "contraction_dag_schema_version": "contraction_dag_v2",
        "execution_plan_hash": None,
        "execution_plan_schema_version": "tn_execution_plan_v1",
        "execution_target": "upmem" if topology["backend"] != "cpu" else "cpu",
        "dag_node_count": len(planned.dag.nodes),
        # Compatibility alias for report consumers; this is a DAG-node count.
        "complete_task_count": len(planned.dag.nodes),
        "planning_time_s": planned.planner_result.planning_time_s,
        "preflight": planned.preflight,
        "anchor_status": "available"
        if anchor and anchor.get("output") is not None
        else "unavailable",
        "anchor_case_planner_key": [
            planned.case["case_id"],
            planned.dag_hash,
        ],
        "anchor_match_key": {
            "case_id": planned.case["case_id"],
            "circuit_semantics_hash": planned.circuit_semantics_hash,
            "planner_hash": planned.planner_result.identity.planner_config_hash,
        },
        "repeat_id": repeat_id,
        "exact_once": None,
        "exact_once_scope": "host_dag_node_completion_per_route",
        "host_dag_node_completion_coverage": None,
        "validation_status": "not_run",
        "scientific_validation_status": "not_run",
        "policy_reference_validation": {"status": "not_run"},
        "full_precision_accuracy": {"status": "not_run"},
        "validation_max_abs_error": None,
        "validation_l2_error": None,
        "validation_norm_drift": None,
        "output_shape": None,
        "output_dtype": None,
        "output_sha256": None,
        "output_norm": None,
        "no_fallback_used": True,
        "cpu_fallback_used": False,
        "simulator_kernel_executed": False,
        "timing_scope": "not_run",
        "timeout_enforcement": engine["timeout_enforcement"],
        "whole_route_including_session_lifecycle_s": None,
        "graph_execution_s": None,
        "session_open_s": None,
        "session_close_s": None,
        "h2d_time_s": None,
        "kernel_time_s": None,
        "d2h_time_s": None,
        "host_quantization_time_s": None,
        "host_dequantization_time_s": None,
        "application_visible_h2d_bytes": None,
        "application_visible_d2h_bytes": None,
        "application_visible_transfer_bytes": None,
        "transfer_accounting_verified": False,
        "timing_breakdown": {},
        "outer_elapsed_s": None,
        "energy_joules": None,
        "energy_status": "unavailable",
        "energy_source": None,
        "measurement_repetitions_sufficient": config["warmups"] >= 1
        and config["repeats"] >= 3,
        "timing_is_bringup_only": not (
            config["warmups"] >= 1 and config["repeats"] >= 3
        ),
        "hardware_speedup_applicable": False,
        "hardware_execution_attempted": False,
        "hardware_execution_verified": False,
        "native_kernel_executed": False,
        "hardware_kernel_executed": False,
        "hardware_allocation_verified": False,
        "hardware_release_verified": False,
        "failure_stage": None,
    }


def _validation(
    actual: np.ndarray,
    expected: np.ndarray | None,
    tolerances: dict[str, Any],
    *,
    quantized: bool,
) -> dict[str, Any]:
    if expected is None:
        return {
            "status": "unavailable",
            "reason": "CPU float32 same-plan anchor unavailable",
        }
    expected = np.asarray(expected)
    if actual.shape != expected.shape:
        return {
            "status": "failed",
            "reason": "shape mismatch",
            "actual_shape": list(actual.shape),
            "expected_shape": list(expected.shape),
        }
    delta = np.asarray(actual, dtype=np.complex128) - np.asarray(
        expected, dtype=np.complex128
    )
    max_abs = float(np.max(np.abs(delta))) if delta.size else 0.0
    l2 = float(np.linalg.norm(delta.ravel()))
    denom = np.maximum(np.abs(np.asarray(expected, dtype=np.complex128)), 1.0e-30)
    max_rel = float(np.max(np.abs(delta) / denom)) if delta.size else 0.0
    expected_norm = float(np.linalg.norm(np.asarray(expected).ravel()))
    actual_norm = float(np.linalg.norm(np.asarray(actual).ravel()))
    norm_drift = abs(actual_norm - expected_norm)
    prefix = "quantized_" if quantized else ""
    passed = (
        max_abs
        <= float(tolerances.get(prefix + "max_abs_error", tolerances["max_abs_error"]))
        and l2 <= float(tolerances.get(prefix + "l2_error", tolerances["l2_error"]))
        and max_rel
        <= float(tolerances.get(prefix + "max_rel_error", tolerances["max_rel_error"]))
        and norm_drift
        <= float(tolerances.get(prefix + "norm_drift", tolerances["norm_drift"]))
    )
    return {
        "status": "passed" if passed else "failed",
        "max_abs_error": max_abs,
        "l2_error": l2,
        "max_rel_error": max_rel,
        "norm_drift": float(norm_drift),
        "expected_norm": expected_norm,
        "actual_norm": actual_norm,
    }


def _resolve_engine(
    variant: dict[str, Any],
    topology: DeviceTopology,
    factories: dict[str, Any],
    *,
    timeout_s: float,
) -> Any:
    engine_id = str(variant["engine"])
    if engine_id == "numpy_cpu":
        raise RuntimeError("CPU routes execute through the functional CPU adapter")
    factory = factories.get(variant["id"], factories.get(engine_id))
    if factory is None:
        raise RuntimeError(f"No engine factory registered for {engine_id}")
    if not inspect.isclass(factory) and hasattr(factory, "open_session"):
        return factory
    if not callable(factory):
        raise TypeError(f"Engine factory for {engine_id} is not callable")
    try:
        signature = inspect.signature(factory)
    except (TypeError, ValueError):
        return factory()
    kwargs = {}
    if "topology" in signature.parameters:
        kwargs["topology"] = topology
    if "engine_variant" in signature.parameters:
        kwargs["engine_variant"] = variant
    if "timeout_s" in signature.parameters:
        kwargs["timeout_s"] = timeout_s
    return factory(**kwargs) if kwargs else factory()


def _policy(name: str) -> Any:
    if name == "float32_real":
        return Float32RealPolicy()
    if name == "host_packed_int8":
        return HostPackedInt8Policy()
    raise ValueError(f"Unsupported numeric policy {name}")


def _is_quantized_policy(policy: Any) -> bool:
    """Identify the host-packed int8 runtime policy independent of its ID."""

    return str(getattr(policy, "name", "")) in {
        "host_packed_int8",
        "host_packed_int8_per_task_v1",
    }


def _normalize_topology(value: Mapping[str, Any]) -> dict[str, Any]:
    backend = str(value.get("backend", "cpu"))
    rank_paths = [str(item) for item in value.get("rank_paths", value.get("ranks", []))]
    device_ids = [str(item) for item in value.get("device_ids", [])]
    if not device_ids:
        dpus = int(value.get("dpus", value.get("dpu_count", 0)))
        if backend == "cpu":
            device_ids = ["cpu"]
        elif dpus > 0:
            device_ids = [f"dpu:{index}" for index in range(dpus)]
    tasklets = int(value.get("tasklets_per_device", value.get("tasklets", 1)))
    if tasklets < 1:
        raise ValueError("topology tasklets must be >= 1")
    if backend != "cpu" and not rank_paths:
        raise ValueError("physical topology must declare explicit rank_paths")
    if any(RANK_PATH_PATTERN.fullmatch(path) is None for path in rank_paths):
        raise ValueError("rank paths must match ^/dev/dpu_rank[0-9]+$")
    if len(set(rank_paths)) != len(rank_paths):
        raise ValueError("UPMEM rank paths must be unique")
    if (
        backend != "cpu"
        and not value.get("device_ids")
        and not value.get("dpus", value.get("dpu_count", 0))
    ):
        raise ValueError("physical topology must declare device_ids or dpus")
    return {
        "backend": backend,
        "device_ids": device_ids,
        "rank_paths": rank_paths,
        "tasklets_per_device": tasklets,
    }


def _named_entries(
    value: Mapping[str, Any],
    key: str,
    *,
    aliases: tuple[str, ...] = (),
    required: bool = False,
) -> list[dict[str, Any]]:
    entries = value.get(key)
    if entries is None:
        for alias in aliases:
            entries = value.get(alias)
            if entries is not None:
                break
    if entries is None:
        if required:
            raise ValueError(f"Study must define {key}")
        return []
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"{key} must be a non-empty list")
    result = []
    seen = set()
    for index, entry in enumerate(entries):
        if isinstance(entry, str):
            entry = {"id": entry}
        if not isinstance(entry, dict) or not entry.get("id"):
            raise ValueError(f"{key}[{index}] must define id")
        entry = dict(entry)
        entry["id"] = str(entry["id"])
        if entry["id"] in seen:
            raise ValueError(f"Duplicate {key} id {entry['id']}")
        seen.add(entry["id"])
        result.append(entry)
    return result


def _final_dag_node(dag: ContractionDAG) -> ContractNode | ReduceNode | None:
    """Return the unique node producing the DAG output, when it exists."""

    final = [node for node in dag.nodes if node.output.id == dag.output.tensor_id]
    if len(final) > 1:
        raise ValueError(f"Expected one final DAG node, found {len(final)}")
    return final[0] if final else None


def _circuit_semantics_hash(circuit: Any) -> str:
    return _semantic_hash(
        {
            "schema": "m5_circuit_semantics_v2",
            "name": circuit.name,
            "n_qubits": int(circuit.n_qubits),
            "operations": [
                {
                    "gate": operation.gate,
                    "wires": operation.wires,
                    "params": operation.params,
                }
                for operation in circuit.operations
            ],
        }
    )


def _tensor_network_hash(spec: TensorNetworkSpec, circuit_hash: str) -> str:
    return _semantic_hash(
        {
            "schema": "m5_tensor_network_v2",
            "circuit_semantics_hash": circuit_hash,
            "tensors": [
                {
                    "id": tensor.id,
                    "labels": tensor.labels,
                    "shape": tensor.shape,
                    "structure": tensor.structure,
                    "dtype": tensor.dtype,
                    "produced_by": tensor.produced_by,
                }
                for tensor in spec.tensors
            ],
            "output_labels": spec.output_labels,
            "einsum_expression": spec.einsum_expression,
        }
    )


def _semantic_hash(payload: Any) -> str:
    """Hash an explicit M5 v2 semantic payload using the shared serializer."""

    return hashlib.sha256(canonical_serialize(payload).encode("utf-8")).hexdigest()


def _engine_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(metadata)
    session_metadata = metadata.get("session_metadata", {})
    if isinstance(session_metadata, Mapping):
        for key, value in session_metadata.items():
            result.setdefault(key, value)
    byte_totals: dict[str, int] = {}
    timing_totals: dict[str, float] = {}
    boolean_any: dict[str, bool] = {}
    identity_keys = {
        "target_observed",
        "physical_profile",
        "hardware_profile",
        "profile",
        "abi",
        "abi_version",
        "protocol_version",
        "numeric_transport",
        "session_protocol",
        "dispatch_mode",
        "kernel_identity",
        "kernel_family",
        "kernel_strategy",
        "graph_intermediate_placement",
        "backend_id",
        "execution_class",
        "observed_rank_count",
        "allocated_dpu_count",
        "observed_tasklets_per_dpu",
        "tasklets_per_dpu",
    }
    for item in metadata.get("task_metrics", ()):
        if isinstance(item, Mapping):
            for key, value in item.items():
                if key.endswith("bytes") and isinstance(value, (int, float)):
                    byte_totals[key] = byte_totals.get(key, 0) + int(value)
                if key.endswith("_s") and isinstance(value, (int, float)):
                    timing_totals[key] = timing_totals.get(key, 0.0) + float(value)
                if (
                    key.endswith("used")
                    or key.endswith("executed")
                    or key.endswith("verified")
                ) and isinstance(value, bool):
                    existing = result.get(key)
                    boolean_any[key] = (
                        boolean_any.get(key, False)
                        or (existing if isinstance(existing, bool) else False)
                        or value
                    )
                if key in identity_keys and value is not None:
                    result.setdefault(key, value)
            nested_timing = item.get("timing")
            if isinstance(nested_timing, Mapping):
                for key, value in nested_timing.items():
                    if (
                        key.endswith("_s")
                        and key not in item
                        and isinstance(value, (int, float))
                        and not isinstance(value, bool)
                    ):
                        timing_totals[key] = timing_totals.get(key, 0.0) + float(value)
    result.update(byte_totals)
    result.update(timing_totals)
    result.update(boolean_any)
    return result


def _executor_config_fields(entry: Mapping[str, Any]) -> dict[str, Any]:
    """Return only stable executor identity fields from a suite entry."""

    identity_keys = (
        "physical_profile",
        "hardware_profile",
        "profile",
        "abi",
        "abi_version",
        "protocol_version",
        "session_protocol",
        "dispatch_mode",
        "kernel_identity",
        "kernel_family",
        "kernel_strategy",
        "kernel_policy",
        "backend_id",
        "execution_class",
    )
    result = dict(entry.get("executor_config") or {})
    for key in identity_keys:
        if key in entry:
            result[key] = entry[key]
    return result


def _executor_config_hash(
    variant: Mapping[str, Any],
    _policy: Any,
    _topology: DeviceTopology,
    metadata: Mapping[str, Any],
) -> str:
    """Hash the stable executor contract, excluding ablation dimensions.

    Numeric policy, planner/path, variant labels, and physical placement are
    represented by separate evidence fields.  They deliberately do not enter
    this compatibility identity so the same implementation can be paired
    across numeric modes and DPU scaling points.
    """

    executor_config = {
        key: value
        for key, value in dict(variant.get("executor_config") or {}).items()
        if key not in {"numeric_policy", "numeric_transport"}
    }
    for key in (
        "physical_profile",
        "hardware_profile",
        "profile",
        "abi",
        "abi_version",
        "protocol_version",
        "session_protocol",
        "dispatch_mode",
        "kernel_identity",
        "kernel_family",
        "kernel_strategy",
        "kernel_policy",
        "backend_id",
        "execution_class",
    ):
        if key in metadata and metadata[key] is not None:
            executor_config[key] = metadata[key]
    payload = {
        "engine_id": str(variant.get("engine", "")),
        "executor_config": executor_config,
    }
    encoded = json.dumps(
        _jsonable(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _target_observed(metadata: Mapping[str, Any]) -> Any:
    for key in ("target_observed", "observed_target", "target", "device_name"):
        value = metadata.get(key)
        if value is not None:
            return value
    return None


def _hardware_contract(
    metadata: Mapping[str, Any], expected_topology: Mapping[str, Any]
) -> dict[str, Any]:
    target_observed = _target_observed(metadata)
    requested_rank_count = len(expected_topology.get("rank_paths", []))
    requested_dpu_count = len(expected_topology.get("device_ids", []))
    requested_tasklets = int(expected_topology.get("tasklets_per_device", 1))
    observed_rank_count = metadata.get("observed_rank_count")
    observed_dpu_count = metadata.get("allocated_dpu_count")
    observed_tasklets = metadata.get(
        "observed_tasklets_per_dpu", metadata.get("tasklets_per_dpu")
    )
    native = bool(
        metadata.get("native_kernel_executed", False)
        or metadata.get("native_execution", False)
    )
    hardware_kernel = bool(metadata.get("hardware_kernel_executed", False))
    allocation = bool(
        metadata.get("hardware_allocation_verified", False)
        or metadata.get("allocation_verified", False)
    )
    release = bool(
        metadata.get("hardware_release_verified", False)
        or metadata.get("release_verified", False)
        or str(metadata.get("resource_release_status", "")).lower()
        in {"released", "passed", "verified", "clean"}
    )
    simulator = bool(metadata.get("simulator_kernel_executed", False))
    cpu_fallback = bool(metadata.get("cpu_fallback_used", False))
    physical_plan = bool(metadata.get("physical_plan_consumed", False))
    verified = (
        target_observed == "physical_hardware"
        and native
        and hardware_kernel
        and allocation
        and release
        and observed_rank_count == requested_rank_count
        and observed_dpu_count == requested_dpu_count
        and observed_tasklets == requested_tasklets
        and physical_plan
        and not simulator
        and not cpu_fallback
    )
    if verified:
        stage = None
        reason = None
    else:
        stage = "hardware_execution_unverified"
        reason = (
            "physical row lacks physical target, matching observed topology, "
            "compiled-plan consumption, verified native allocation, hardware "
            "execution, release, or no-fallback metadata"
        )
    return {
        "verified": verified,
        "native_kernel_executed": native,
        "hardware_kernel_executed": hardware_kernel,
        "allocation_verified": allocation,
        "release_verified": release,
        "physical_plan_consumed": physical_plan,
        "failure_stage": stage,
        "reason": reason,
    }


def _first_number(*sources: Mapping[str, Any], keys: Iterable[str]) -> float | None:
    for source in sources:
        for key in keys:
            value = source.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return float(value)
    return None


def _first_byte_count(*sources: Mapping[str, Any], keys: Iterable[str]) -> int | None:
    for source in sources:
        for key in keys:
            if key not in source:
                continue
            value = source[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return None
            number = float(value)
            if not np.isfinite(number) or number < 0 or not number.is_integer():
                return None
            return int(value)
    return None


def _physical_timing_complete(timing: Mapping[str, Any]) -> bool:
    required = ("h2d_time_s", "kernel_time_s", "d2h_time_s")
    return all(
        isinstance(timing.get(key), (int, float))
        and not isinstance(timing.get(key), bool)
        and np.isfinite(float(timing[key]))
        and float(timing[key]) >= 0.0
        for key in required
    )


def _array_hash(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _failure_stage(exc: Exception) -> str:
    failure_stage = getattr(exc, "failure_stage", None)
    if isinstance(failure_stage, str) and failure_stage:
        return failure_stage
    if "No engine factory registered" in str(exc):
        return "engine_factory_missing"
    if "terminal metadata is not physically verified" in str(exc):
        return "hardware_execution_unverified"
    if isinstance(exc, TimeoutError):
        return "timeout"
    if isinstance(exc, ValueError):
        return (
            "output_validation_failed"
            if "validation" in str(exc).lower()
            else "execution_failed"
        )
    return "engine_execution_failed"


def _plan_manifest(
    config: dict[str, Any],
    plans: Iterable[_Plan],
    artifact_dir: Path,
    *,
    hardware_opened: bool,
    selected_route_ids: Iterable[str],
) -> dict[str, Any]:
    selected_ids = set(selected_route_ids)
    return {
        "schema_version": STUDY_SCHEMA_VERSION,
        "study_id": config["study_id"],
        "hardware_opened": hardware_opened,
        "artifact_dir": str(artifact_dir),
        "selected_route_ids": list(selected_route_ids),
        "pipeline_routes": [
            route
            for route in config["pipeline_routes"]
            if route["route_id"] in selected_ids
        ],
        "pipeline_comparisons": selected_pipeline_comparisons(config, selected_ids),
        "plans": [
            {
                "case_id": plan.case["case_id"],
                "family": plan.case.get("family"),
                "planner_id": plan.planner["id"],
                "planner_config": plan.planner["planner"],
                "planner_config_hash": plan.planner_result.identity.planner_config_hash,
                "circuit": circuit_manifest(plan.circuit),
                "circuit_semantics_hash": plan.circuit_semantics_hash,
                "tensor_network_hash": plan.tensor_network_hash,
                # Compatibility aliases: historical plan fields now carry the
                # canonical semantic ContractionDAG identity.
                "contraction_plan_hash": plan.dag_hash,
                "contraction_path_structure_hash": plan.dag_hash,
                "contraction_dag_hash": plan.dag_hash,
                "contraction_dag_schema_version": "contraction_dag_v2",
                "execution_plan_schema_version": "tn_execution_plan_v1",
                "dag_node_count": len(plan.dag.nodes),
                # Compatibility alias for report consumers; this is a DAG-node count.
                "task_count": len(plan.dag.nodes),
                "planning_time_s": plan.planner_result.planning_time_s,
                "resources": plan.resources,
                "preflight": plan.preflight,
            }
            for plan in plans
        ],
    }


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")


__all__ = [
    "load_study_config",
    "plan_study",
    "run_study",
]
