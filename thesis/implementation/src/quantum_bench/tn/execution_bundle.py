from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from typing import Any

from quantum_bench.core.records import JsonDict, TaskGraph, to_jsonable


EXECUTION_BUNDLE_SCHEMA_VERSION = "execution_bundle_v1"


def with_execution_identity(graph: TaskGraph) -> TaskGraph:
    hashes = execution_hashes(graph)
    existing = {
        "circuit_semantics_hash": graph.circuit_semantics_hash,
        "tensor_network_hash": graph.tensor_network_hash,
        "contraction_plan_hash": graph.contraction_plan_hash,
    }
    for key, value in existing.items():
        if value and value != hashes[key]:
            raise ValueError(f"TaskGraph {key} does not match its canonical content")
    return replace(graph, **hashes)


def execution_hashes(graph: TaskGraph) -> dict[str, str]:
    circuit_payload = _circuit_payload(graph)
    circuit_hash = canonical_hash(circuit_payload)
    network_payload = _network_payload(graph, circuit_hash)
    network_hash = canonical_hash(network_payload)
    plan_payload = _plan_payload(graph, network_hash)
    return {
        "circuit_semantics_hash": circuit_hash,
        "tensor_network_hash": network_hash,
        "contraction_plan_hash": canonical_hash(plan_payload),
    }


def build_execution_bundle(
    graph: TaskGraph,
    *,
    case_id: str,
    suite_id: str,
) -> JsonDict:
    identified = with_execution_identity(graph)
    return to_jsonable(
        {
            "schema_version": EXECUTION_BUNDLE_SCHEMA_VERSION,
            "case_id": case_id,
            "suite_id": suite_id,
            "circuit_semantics_hash": identified.circuit_semantics_hash,
            "tensor_network_hash": identified.tensor_network_hash,
            "contraction_plan_hash": identified.contraction_plan_hash,
            "circuit": _circuit_payload(identified),
            "tensor_network": _network_payload(identified, identified.circuit_semantics_hash),
            "planner": _planner_payload(identified),
            "contraction_plan": _plan_payload(identified, identified.tensor_network_hash),
            "output_contract": {
                "output_labels": identified.network.output_labels,
                "dtype": "complex128",
            },
            "provenance": {
                "planning_time_s": float(identified.planning_time_s),
                "planning_in_timed_region": False,
            },
        }
    )


def validate_execution_bundle(bundle: JsonDict, graph: TaskGraph) -> None:
    if bundle.get("schema_version") != EXECUTION_BUNDLE_SCHEMA_VERSION:
        raise ValueError("unsupported execution bundle schema")
    hashes = execution_hashes(graph)
    for key, expected in hashes.items():
        if bundle.get(key) != expected:
            raise ValueError(f"execution bundle {key} mismatch")


def execution_identity_metadata(graph: TaskGraph, *, plan_reused: bool) -> JsonDict:
    identified = with_execution_identity(graph)
    return {
        "circuit_semantics_hash": identified.circuit_semantics_hash,
        "tensor_network_hash": identified.tensor_network_hash,
        "contraction_plan_hash": identified.contraction_plan_hash,
        "plan_reused": bool(plan_reused),
        "planning_in_timed_region": False,
    }


def executor_config_hash(route_id: str, config: dict[str, Any] | None = None) -> str:
    return canonical_hash({"route_id": route_id, "config": config or {}})


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(to_jsonable(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _circuit_payload(graph: TaskGraph) -> JsonDict:
    circuit = graph.network.circuit
    return to_jsonable(
        {
            "name": circuit.name,
            "n_qubits": int(circuit.n_qubits),
            "operations": [
                {"gate": operation.gate, "wires": operation.wires, "params": operation.params}
                for operation in circuit.operations
            ],
        }
    )


def _network_payload(graph: TaskGraph, circuit_hash: str) -> JsonDict:
    return to_jsonable(
        {
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
                for tensor in graph.network.tensors
            ],
            "output_labels": graph.network.output_labels,
            "einsum_expression": graph.network.einsum_expression,
        }
    )


def _planner_payload(graph: TaskGraph) -> JsonDict:
    summary = graph.path_summary
    return to_jsonable(
        {
            "planner_engine": summary.planner_engine,
            "planner_id": summary.planner_id,
            "planner_kind": summary.planner_kind,
            "optimize_mode": summary.optimize_mode,
            "objective": summary.objective,
            "cost_basis": summary.cost_basis,
            "options": summary.options,
        }
    )


def _plan_payload(graph: TaskGraph, network_hash: str) -> JsonDict:
    return to_jsonable(
        {
            "tensor_network_hash": network_hash,
            "planner": _planner_payload(graph),
            "path": graph.path,
            "tasks": [
                {
                    "id": task.id,
                    "input_tensor_ids": task.input_tensor_ids,
                    "output_tensor_id": task.output_tensor_id,
                    "dependencies": task.dependencies,
                    "index_expression": task.index_expression,
                    "input_shapes": task.input_shapes,
                    "output_shape": task.output_shape,
                    "left_labels": task.left_labels,
                    "right_labels": task.right_labels,
                    "contracted_labels": task.contracted_labels,
                    "output_labels": task.output_labels,
                    "gemm_m": task.gemm_m,
                    "gemm_k": task.gemm_k,
                    "gemm_n": task.gemm_n,
                    "structure": task.structure,
                }
                for task in graph.tasks
            ],
            "output_labels": graph.network.output_labels,
        }
    )
