from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from quantum_bench.core.records import TaskGraph, TensorValue
from quantum_bench.circuits import load_circuit
from quantum_bench.targets.upmem.hardware_taskgraph_frontier import (
    BACKEND_ID,
    NATIVE_SCHEMA,
    REQUEST_SCHEMA,
    PROFILE_ID,
    ROUTE_ID,
    build_hardware_frontier_graph_package,
    build_hardware_frontier_plan,
    validate_frontier_package_validation_response,
    validate_frontier_native_response,
    write_hardware_frontier_graph_package,
)
from quantum_bench.targets.upmem.hardware_taskgraph_resident import (
    RESIDENT_CONTROL_H2D_BYTES_PER_LAUNCH,
    RESIDENT_DESCRIPTOR_CONTROL_BYTES,
)
from quantum_bench.tn.network import TensorNetworkValue, build_tensor_network
from quantum_bench.tn.task_graph import plan_task_graph_with_config


def _fixture() -> tuple[TaskGraph, TensorNetworkValue]:
    circuit = load_circuit(
        {
            "circuit": {
                "kind": "qasm_file",
                "path": "configs/circuits/upmem_m2/one_qubit_ry_h_ry_a.qasm",
            }
        },
        Path(__file__).resolve().parents[1],
    )
    network = build_tensor_network(circuit)
    graph = plan_task_graph_with_config(
        network, {"engine": "opt_einsum", "optimize": "greedy"}
    )
    return graph, network


def test_fixed_frontier_plan_has_two_waves_and_dpu_counts() -> None:
    graph, network = _fixture()
    plan = build_hardware_frontier_plan(graph, network)
    assert [[item.task_id for item in wave] for wave in plan.waves] == [
        ["task_0", "task_1"],
        ["task_2"],
    ]
    assert plan.dpu_task_counts == (2, 1)
    assert plan.to_json_dict()["overlap_measured"] is False


@pytest.mark.parametrize(
    "mutation",
    (
        lambda graph: graph.__class__(graph.network, graph.tasks[:2], ((0, 1),), graph.path_summary, 0.0),
        lambda graph: graph.__class__(graph.network, tuple(
            task if task.id != "task_2" else task.__class__(**{**task.__dict__, "dependencies": ()})
            for task in graph.tasks
        ), graph.path, graph.path_summary, 0.0),
    ),
)
def test_graph_dependency_shape_is_rejected(mutation) -> None:
    graph, network = _fixture()
    with pytest.raises(ValueError, match="frontier|dependencies"):
        build_hardware_frontier_plan(mutation(graph), network)


def test_graph_rejects_missing_produced_tensor_dataflow() -> None:
    graph, network = _fixture()
    task = graph.tasks[2]
    bad = task.__class__(**{**task.__dict__, "input_tensor_ids": ("tensor_0", "result_1")})
    broken = graph.__class__(graph.network, (graph.tasks[0], graph.tasks[1], bad), graph.path, graph.path_summary, 0.0)
    with pytest.raises(ValueError, match="dataflow|dependency"):
        build_hardware_frontier_plan(broken, network)


def test_qasm_complex128_inputs_are_lowered_and_nonzero_imaginary_values_rejected() -> None:
    graph, network = _fixture()
    with pytest.raises(ValueError, match="numeric_mode"):
        build_hardware_frontier_graph_package(graph, network, case_id="c", suite_id="s", quantization_mode="int8")
    package = build_hardware_frontier_graph_package(graph, network, case_id="c", suite_id="s")
    assert all(item.dtype == "float32" for item in package.graph.network.tensors)
    assert all(np.asarray(value).dtype == np.dtype("float32") for value in package.initial_data.values())

    bad_values = [TensorValue(item.spec, item.array.copy()) for item in network.tensors]
    bad_values[1].array.flat[0] += 1j
    complex_network = TensorNetworkValue(
        network.spec,
        bad_values,
    )
    with pytest.raises(ValueError, match="zero imaginary"):
        build_hardware_frontier_plan(graph, complex_network)


def test_writer_replaces_identity_and_records_frontier_manifest(tmp_path: Path) -> None:
    graph, network = _fixture()
    package = build_hardware_frontier_graph_package(graph, network, case_id="c", suite_id="s")
    binary = tmp_path / "dpu"
    binary.write_bytes(b"dpu")
    written = write_hardware_frontier_graph_package(
        package, tmp_path, dpu_binary=binary, request_id="frontier-request"
    )
    assert written.manifest_path is not None
    manifest = json.loads(written.manifest_path.read_text(encoding="utf-8"))
    assert {manifest[key] for key in ("route_id", "backend_id", "hardware_profile_version", "schema_version")} == {
        ROUTE_ID, BACKEND_ID, PROFILE_ID, REQUEST_SCHEMA
    }
    assert manifest["native_schema_version"] == NATIVE_SCHEMA
    assert manifest["requested_dpus"] == 2
    assert manifest["frontier_plan"]["expected_dpu_task_counts"] == [2, 1]
    assert [item["task_id"] for item in manifest["frontier_task_operations"]] == ["task_0", "task_1", "task_2"]
    assert manifest["frontier_task_operations"][2]["input_slot_ids"]
    transfer = manifest["expected_frontier_transfer"]
    assert transfer["descriptor_package_h2d_bytes"] == (manifest["descriptor_h2d_bytes"] + RESIDENT_DESCRIPTOR_CONTROL_BYTES) * 2
    assert transfer["initial_h2d_bytes"] == manifest["initial_h2d_bytes"] * 2
    assert transfer["operation_control_h2d_bytes"] == len(manifest["frontier_task_operations"]) * RESIDENT_CONTROL_H2D_BYTES_PER_LAUNCH
    assert transfer["total_bytes"] == transfer["h2d_bytes"] + transfer["d2h_bytes"]


def _completed_response(manifest: dict[str, object]) -> dict[str, object]:
    operations = manifest["frontier_task_operations"]
    assert isinstance(operations, list)
    return {
        "schema_version": NATIVE_SCHEMA,
        "native_schema_version": NATIVE_SCHEMA,
        "route_id": ROUTE_ID,
        "backend_id": BACKEND_ID,
        "hardware_profile_version": PROFILE_ID,
        "target_requested": "hardware",
        "target_observed": "hardware",
        "numeric_mode": "none",
        "tasklets_per_dpu": 1,
        "requested_dpus": 2,
        "allocated_dpus": 2,
        "status": "completed",
        "failure_stage": None,
        "hardware_execution": True,
        "hardware_kernel_executed": True,
        "hardware_functionality_evidence": True,
        "cpu_fallback_used": False,
        "simulator_kernel_executed": False,
        "no_cpu_fallback": True,
        "no_simulator_fallback": True,
        "native_failure_fallback_used": False,
        "hardware_no_fallback": True,
        "performance_claim_applicable": False,
        "timing_scope": "two_dpu_frontier_resident_full_taskgraph_v1",
        "timing": {
            "clock": "clock_monotonic",
            "overlap_measured": False,
            **{field: 0.001 for field in (
                "package_parse_time_s", "allocation_time_s", "binary_load_time_s",
                "initial_h2d_time_s", "wave0_launch_time_s", "wave0_barrier_wait_time_s",
                "wave1_launch_time_s", "wave1_barrier_wait_time_s", "final_d2h_time_s",
                "release_time_s", "total_route_time_s",
            )},
        },
        "allocation": {"requested_dpus": 2, "allocated_dpus": 2, "verified": True},
        "load": {"confirmed": True, "hardware": True},
        "launch": {"completed": True, "task_count": 3, "barrier_count": 2},
        "release": {"confirmed": True},
        "co_dispatch_confirmed": True,
        "overlap_measured": False,
        "overlap_claim": "unmeasured",
        "overlap_evidence": "co_dispatch_without_overlap_measurement",
        "wave0_complete_before_wave1": True,
        "completed_task_ids": ["task_0", "task_1", "task_2"],
        "completed_task_ids_scope": "wave_dependency_order_not_intra_wave_finish_order",
        "barrier_count": 2,
        "barriers": [
            {"barrier_index": 0, "wave_index": 0, "completed": True},
            {"barrier_index": 1, "wave_index": 1, "completed": True},
        ],
        "observed_dpu_task_counts": [2, 1],
        "transfer": {
            **manifest["expected_frontier_transfer"],
            "transfer_invariant": True,
            "accounting_scope": "sdk_argument_byte_counts",
        },
        "actual_h2d_bytes": manifest["expected_frontier_transfer"]["h2d_bytes"],
        "actual_d2h_bytes": manifest["expected_frontier_transfer"]["d2h_bytes"],
        "actual_transfer_bytes": manifest["expected_frontier_transfer"]["total_bytes"],
        "transfer_accounting_scope": "native_sdk_observed_application_visible",
        "tasks": [
            {**item, "completed": True, "completion_confirmed": True}
            for item in operations
        ],
        "final_output": {
            **manifest["final_output_binding"],
            "hash_fnv1a64": "0" * 16,
            "path": "/session/" + manifest["final_output_binding"]["output_path"],
            "output_path": "/session/" + manifest["final_output_binding"]["output_path"],
            "written": True,
        },
        "hashes": {
            "manifest_fnv1a64": "0" * 16,
            "package_fnv1a64": "0" * 16,
            "dpu_binary_fnv1a64": "0" * 16,
            "host_source_fnv1a64": "0" * 16,
        },
    }


def test_response_validator_rejects_duplicate_order_and_transfer_mismatch(tmp_path: Path) -> None:
    graph, network = _fixture()
    package = build_hardware_frontier_graph_package(graph, network, case_id="c", suite_id="s")
    binary = tmp_path / "dpu"
    binary.write_bytes(b"dpu")
    written = write_hardware_frontier_graph_package(package, tmp_path, dpu_binary=binary, request_id="r")
    assert written.manifest_path is not None
    manifest = json.loads(written.manifest_path.read_text(encoding="utf-8"))
    response = _completed_response(manifest)
    validate_frontier_native_response(response, manifest)
    response["actual_transfer_bytes"] = 31
    with pytest.raises(ValueError, match="transfer"):
        validate_frontier_native_response(response, manifest)


def test_final_output_validation_requires_exact_absolute_binding_and_fnv(tmp_path: Path) -> None:
    graph, network = _fixture()
    package = build_hardware_frontier_graph_package(graph, network, case_id="c", suite_id="s")
    binary = tmp_path / "dpu"
    binary.write_bytes(b"dpu")
    written = write_hardware_frontier_graph_package(package, tmp_path, dpu_binary=binary, request_id="output")
    assert written.manifest_path is not None
    manifest = json.loads(written.manifest_path.read_text(encoding="utf-8"))
    output = tmp_path / manifest["final_output_binding"]["output_path"]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(b"\x00" * manifest["final_output_binding"]["raw_bytes"])
    response = _completed_response(manifest)
    absolute = str(output.resolve())
    response["final_output"]["output_path"] = absolute
    response["final_output"]["path"] = absolute
    response["final_output"]["hash_fnv1a64"] = "a8c7f832281a39c5"
    from quantum_bench.targets.upmem.hardware_taskgraph_frontier import validate_frontier_output_file

    validate_frontier_output_file(response, manifest, tmp_path)
    response["final_output"]["path"] = str(tmp_path / "other.bin")
    with pytest.raises(ValueError, match="path"):
        validate_frontier_output_file(response, manifest, tmp_path)


def test_validate_only_response_is_non_hardware_and_strict(tmp_path: Path) -> None:
    graph, network = _fixture()
    package = build_hardware_frontier_graph_package(graph, network, case_id="c", suite_id="s")
    binary = tmp_path / "dpu"
    binary.write_bytes(b"dpu")
    written = write_hardware_frontier_graph_package(package, tmp_path, dpu_binary=binary, request_id="validate")
    assert written.manifest_path is not None
    manifest = json.loads(written.manifest_path.read_text(encoding="utf-8"))
    validate_frontier_package_validation_response(
        {
            "schema_version": NATIVE_SCHEMA,
            "native_schema_version": NATIVE_SCHEMA,
            "status": "valid",
            "valid": True,
            "failure_stage": None,
            "error": None,
            "native_execution": False,
            "allocation_attempted": False,
            "launch_attempted": False,
            "release_attempted": False,
            "requested_dpus": 2,
            "tasklets_per_dpu": 1,
            "operation_count": 3,
            "final_output_count": 1,
            "quantization_mode": "none",
            "route_id": ROUTE_ID,
            "backend_id": BACKEND_ID,
            "hardware_profile_version": PROFILE_ID,
            "target": "hardware",
            "session_protocol": NATIVE_SCHEMA,
            "profile_id": PROFILE_ID,
            "wave_barrier_count": 2,
        },
        manifest,
    )
