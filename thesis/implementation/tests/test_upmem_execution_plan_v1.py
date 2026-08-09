from __future__ import annotations

from dataclasses import replace
import struct
from pathlib import Path

import pytest

from quantum_bench.circuits import load_circuit
from quantum_bench.targets.upmem.execution_plan_v1 import (
    PLACEMENT_FRONTIER,
    PLACEMENT_SINGLE,
    build_request_manifest,
    compile_plan,
    package_bytes_for,
    parse_plan_json,
    parse_schedule,
    serialize_plan_json,
    serialize_schedule,
    SCHEDULE_HEADER_BYTES,
    SCHEDULE_RECORD_BYTES,
    validate_plan,
    validate_schedule,
)
from quantum_bench.targets.upmem.hardware_taskgraph_frontier import (
    build_hardware_frontier_graph_package,
)
from quantum_bench.tn.network import build_tensor_network
from quantum_bench.tn.task_graph import plan_task_graph_with_config


ROOT = Path(__file__).resolve().parents[1]
QASM = "configs/circuits/upmem_m2/one_qubit_ry_h_ry_a.qasm"


def _fixture():
    circuit = load_circuit({"circuit": {"kind": "qasm_file", "path": QASM}}, ROOT)
    network = build_tensor_network(circuit)
    graph = plan_task_graph_with_config(network, {"engine": "opt_einsum", "optimize": "greedy"})
    package = build_hardware_frontier_graph_package(
        graph, network, case_id="execution-plan-test", suite_id="execution-plan-test"
    )
    return graph, package


def test_plans_and_sidecars_are_deterministic_and_preserve_lowered_identity() -> None:
    graph, package = _fixture()
    first = compile_plan(graph, package, placement_policy=PLACEMENT_FRONTIER)
    second = compile_plan(graph, package, placement_policy=PLACEMENT_FRONTIER)

    assert serialize_plan_json(first) == serialize_plan_json(second)
    assert serialize_schedule(first) == serialize_schedule(second)
    assert first.source_tensor_network_hash == graph.tensor_network_hash
    assert first.package_tensor_network_hash == package.graph.tensor_network_hash
    assert first.source_tensor_network_hash != first.package_tensor_network_hash
    assert first.execution_plan_hash == second.execution_plan_hash
    assert first.execution_plan_hash != first.schedule_sidecar_sha256
    assert SCHEDULE_HEADER_BYTES == 80
    assert SCHEDULE_RECORD_BYTES == 32


def test_single_dpu_policy_is_sequential_and_has_no_cross_dpu_edges() -> None:
    graph, package = _fixture()
    plan = compile_plan(graph, package, placement_policy=PLACEMENT_SINGLE)

    assert plan.requested_dpu_count == 1
    assert plan.waves == (("task_0",), ("task_1",), ("task_2",))
    assert [item.dpu_id for item in plan.assignments] == [0, 0, 0]
    assert plan.transfer_edges == ()


def test_two_dpu_frontier_policy_assigns_expected_counts_and_handoff() -> None:
    graph, package = _fixture()
    plan = compile_plan(graph, package, placement_policy=PLACEMENT_FRONTIER)

    assert plan.waves == (("task_0", "task_1"), ("task_2",))
    assert [item.dpu_id for item in plan.assignments] == [0, 1, 0]
    assert [sum(item.dpu_id == dpu for item in plan.assignments) for dpu in (0, 1)] == [2, 1]
    assert len(plan.transfer_edges) == 1
    edge = plan.transfer_edges[0]
    assert (edge.producer_task_id, edge.consumer_task_id) == ("task_1", "task_2")
    assert edge.element_count == 4
    assert edge.transfer_bytes == 16
    assert plan.host_to_dpu_bytes == 16
    assert plan.dpu_to_host_bytes == 16


def test_placement_changes_execution_hash_but_not_source_contraction_hash() -> None:
    graph, package = _fixture()
    single = compile_plan(graph, package, placement_policy=PLACEMENT_SINGLE)
    frontier = compile_plan(graph, package, placement_policy=PLACEMENT_FRONTIER)

    assert single.execution_plan_hash != frontier.execution_plan_hash
    assert single.source_contraction_plan_hash == frontier.source_contraction_plan_hash
    assert single.package_contraction_plan_hash == frontier.package_contraction_plan_hash


def test_json_round_trip_and_request_manifest_keep_sidecar_out_of_h2d() -> None:
    graph, package = _fixture()
    plan = compile_plan(graph, package, placement_policy=PLACEMENT_FRONTIER)
    parsed = parse_plan_json(serialize_plan_json(plan))
    sidecar = serialize_schedule(plan)
    manifest = build_request_manifest(
        plan,
        package,
        sidecar,
        package_path="resident_graph_package.bin",
        schedule_path="execution_plan.bin",
        dpu_binary="dpu.bin",
    )

    assert parsed.execution_plan_hash == plan.execution_plan_hash
    assert manifest["source_identity"] == {
        "circuit_semantics_hash": plan.source_circuit_semantics_hash,
        "tensor_network_hash": plan.source_tensor_network_hash,
        "contraction_plan_hash": plan.source_contraction_plan_hash,
    }
    assert manifest["package_identity"] == {
        "circuit_semantics_hash": plan.package_circuit_semantics_hash,
        "tensor_network_hash": plan.package_tensor_network_hash,
        "contraction_plan_hash": plan.package_contraction_plan_hash,
    }
    assert manifest["source_tensor_network_hash"] == graph.tensor_network_hash
    assert manifest["package_tensor_network_hash"] == package.graph.tensor_network_hash
    assert manifest["source_tensor_network_hash"] != manifest["package_tensor_network_hash"]
    assert manifest["package_file_sha256"] == plan.package_file_sha256
    assert manifest["schedule_sidecar_sha256"] == plan.schedule_sidecar_sha256
    assert manifest["schedule_sidecar_h2d_bytes"] == 0
    assert manifest["schedule_sidecar_scope"] == "host_metadata_not_h2d"


def test_request_manifest_rejects_lowered_graph_as_source_identity() -> None:
    graph, package = _fixture()
    plan = compile_plan(graph, package, placement_policy=PLACEMENT_FRONTIER)

    with pytest.raises(ValueError, match="source graph identity mismatch"):
        build_request_manifest(
            plan,
            package,
            serialize_schedule(plan),
            package_path="resident_graph_package.bin",
            schedule_path="execution_plan.bin",
            dpu_binary="dpu.bin",
            source_graph=package.graph,
        )


def test_malformed_schedule_and_caps_are_rejected() -> None:
    graph, package = _fixture()
    plan = compile_plan(graph, package, placement_policy=PLACEMENT_FRONTIER)
    sidecar = serialize_schedule(plan)

    with pytest.raises(ValueError, match="magic|version"):
        parse_schedule(b"bad" + sidecar[3:])
    with pytest.raises(ValueError, match="truncated|length"):
        parse_schedule(sidecar[:-1])

    too_many_operations = bytearray(sidecar)
    struct.pack_into("<I", too_many_operations, 16, 9)
    with pytest.raises(ValueError, match="cap|length"):
        parse_schedule(too_many_operations)

    parsed = parse_schedule(sidecar)
    assert parsed.records == tuple(
        (
            item.package_operation_index,
            item.operation_id,
            item.dependency_bitmask,
            item.wave_index,
            item.dpu_id,
            item.input_slot_ids[0],
            item.input_slot_ids[1],
            item.output_slot_id,
        )
        for item in plan.assignments
    )


def test_future_wave_and_bad_dpu_assignments_are_rejected() -> None:
    graph, package = _fixture()
    plan = compile_plan(graph, package, placement_policy=PLACEMENT_FRONTIER)
    first = plan.assignments[0]
    future_dependency = replace(first, dependency_operation_ids=(2,), dependency_bitmask=1 << 2)
    with pytest.raises(ValueError, match="earlier wave"):
        validate_plan(replace(plan, assignments=(future_dependency, *plan.assignments[1:])))

    bad_dpu = replace(plan.assignments[0], dpu_id=2)
    with pytest.raises(ValueError, match="out of range"):
        validate_plan(replace(plan, assignments=(bad_dpu, *plan.assignments[1:])))


def test_package_mutation_and_assignment_hash_mismatch_are_rejected() -> None:
    graph, package = _fixture()
    plan = compile_plan(graph, package, placement_policy=PLACEMENT_FRONTIER)
    sidecar = serialize_schedule(plan)
    package_bytes = bytearray(package_bytes_for(package))
    package_bytes[-1] ^= 1

    with pytest.raises(ValueError, match="package|magic|version"):
        validate_schedule(sidecar, plan, package_bytes=bytes(package_bytes))

    payload = plan.to_json()
    payload["task_assignments"][0]["dpu_id"] = 1
    with pytest.raises(ValueError, match="hash|assignment|DPU"):
        parse_plan_json(payload)

    malformed = plan.to_json()
    malformed["task_assignments"].append("not-an-object")
    with pytest.raises(ValueError, match="task_assignments"):
        parse_plan_json(malformed)


def test_transfer_edges_must_equal_derived_cross_dpu_dependencies() -> None:
    graph, package = _fixture()
    plan = compile_plan(graph, package, placement_policy=PLACEMENT_FRONTIER)
    with pytest.raises(ValueError, match="transfer edges"):
        validate_plan(replace(plan, transfer_edges=()), graph=graph, package=package)


def test_block_one_rejects_non_real_multi_component_packages() -> None:
    from tests.support import split_complex_graph
    from quantum_bench.targets.upmem.hardware_taskgraph_resident import build_resident_graph_package

    case = split_complex_graph()
    package = build_resident_graph_package(
        case.graph, case.network, case_id="complex", suite_id="complex", quantization_mode="none"
    )
    with pytest.raises(ValueError, match="multi-component|real float32|real contract"):
        compile_plan(case.graph, package)
