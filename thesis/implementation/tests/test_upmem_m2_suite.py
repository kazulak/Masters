from __future__ import annotations

import json
from pathlib import Path
import struct

import numpy as np
import pytest
import yaml

import quantum_bench.bench.upmem_hardware_sliced_resident_mvp as mvp
from quantum_bench.bench.config import load_suite
from quantum_bench.circuits import load_circuit
from quantum_bench.targets.upmem.hardware_taskgraph_sliced_resident import (
    build_two_slice_resident_graph_packages,
    build_two_slice_resident_plan,
    reconstruct_host_slice_outputs,
    validate_written_two_slice_packages,
    write_two_slice_resident_graph_packages,
)
from quantum_bench.targets.upmem.hardware_taskgraph_resident import (
    RESIDENT_PACKAGE_HEADER_FORMAT,
    RESIDENT_SLOT_FORMAT,
    RESIDENT_SLOT_INITIAL_FLAG,
    RESIDENT_SLOT_BYTES,
    RESIDENT_OPERATION_BYTES,
    _encode_package,
    validate_resident_graph_package_file,
    validate_resident_graph_package_bytes,
)
from quantum_bench.tn import (
    build_tensor_network,
    execute_task_sequence_np_einsum,
    plan_task_graph_with_config,
)
from quantum_bench.tn.slicing import build_slice_aware_taskgraph_model


ROOT = Path(__file__).resolve().parents[1]
SUITE_PATH = ROOT / "configs" / "suites" / "upmem_hardware_sliced_resident_mvp.yml"
M2_1_SUITE_PATH = ROOT / "configs" / "suites" / "upmem_hardware_sliced_resident_m2_1.yml"


def test_m2_suite_is_strict_and_reconstructs_qasm_cases() -> None:
    raw = yaml.safe_load(SUITE_PATH.read_text(encoding="utf-8"))
    suite = load_suite(SUITE_PATH)
    profile = raw["metadata"]["hardware_profile"]
    route = raw["routes"][0]

    assert raw["metadata"]["hardware_sliced_resident_m2_schema_version"] == (
        "upmem_hardware_sliced_resident_m2_v1"
    )
    assert suite["warmups"] == 1
    assert suite["repeats"] == 3
    assert suite["metadata"]["quantization_mode"] == "none"
    assert len(raw["routes"]) == 1
    assert route["id"] == "upmem_tn_hardware_sliced_resident_two_dpu"
    assert route["options"]["backend_id"] == (
        "upmem_sdk_hardware_sliced_resident_two_dpu"
    )
    assert profile["requested_dpu_count"] == 2
    assert profile["slices"] == 2
    assert profile["tasklets_per_dpu"] == 1
    assert profile["numeric_modes"] == ["none"]
    assert profile["device_launch_mode"] == "asynchronous_dpu_set"
    assert profile["host_completion_mode"] == "blocking_sync"
    assert profile["performance_claim_applicable"] is False
    assert profile["device_launch_mode"] == "asynchronous_dpu_set"
    assert profile["host_completion_mode"] == "blocking_sync"
    assert raw["metadata"]["claim_boundary"].endswith("no_speedup_claim")
    assert len(suite["cases"]) == 3

    for case in suite["cases"]:
        circuit = load_circuit(case, ROOT)
        network = build_tensor_network(circuit)
        graph = plan_task_graph_with_config(network, suite["planner"])

        assert circuit.n_qubits == 1
        assert len(circuit.operations) == 1
        assert len(graph.tasks) == 1
        task = graph.tasks[0]
        assert task.dependencies == ()
        assert task.output_labels == graph.network.output_labels
        assert task.contracted_labels
        assert task.gemm_k == 2
        assert all(
            tensor.spec.dtype == "complex128" and not np.any(tensor.array.imag)
            for tensor in network.tensors
        )

        unsliced, _ = execute_task_sequence_np_einsum(graph, network)
        expected = np.asarray(case["expected_output"], dtype=np.complex128)
        np.testing.assert_allclose(unsliced, expected, atol=1.0e-12, rtol=1.0e-12)

        plan = build_two_slice_resident_plan(graph, network)
        assert len(plan.slice_plans) == 2
        assert [item.dpu_id for item in plan.slice_plans] == [0, 1]
        assert plan.execution_plan.placement.resources.requested_dpu_count == 2
        assert plan.execution_plan.placement.resources.requested_tasklets_per_dpu == 1

        packages = build_two_slice_resident_graph_packages(
            plan,
            case_id=case["case_id"],
            suite_id=suite["suite_id"],
            quantization_mode=suite["metadata"]["quantization_mode"],
        )
        assert len(packages) == 2
        assert [item.dpu_id for item in packages] == [0, 1]
        partials = {
            item.slice_id: execute_task_sequence_np_einsum(
                item.package.graph, item.network
            )[0]
            for item in packages
        }
        reconstructed = reconstruct_host_slice_outputs(plan, partials)
        np.testing.assert_allclose(reconstructed, unsliced, atol=1.0e-6, rtol=1.0e-6)


@pytest.mark.parametrize("gate", ["x", "h", "z"])
def test_m2_qasm_files_are_single_gate_real_circuits(gate: str) -> None:
    path = ROOT / "configs" / "circuits" / "upmem_m2" / f"one_qubit_{gate}.qasm"
    circuit = load_circuit({"circuit": {"kind": "qasm_file", "path": str(path)}}, ROOT)

    assert circuit.n_qubits == 1
    assert [(operation.gate, operation.wires) for operation in circuit.operations] == [
        (gate, (0,))
    ]


def test_m2_1_fixture_builds_dependent_hx_graph_and_useful_cpu_slices(tmp_path) -> None:
    raw = yaml.safe_load(M2_1_SUITE_PATH.read_text(encoding="utf-8"))
    suite = load_suite(M2_1_SUITE_PATH)
    profile = raw["metadata"]["hardware_profile"]
    route = raw["routes"][0]
    case = suite["cases"][0]

    assert raw["suite_id"] == "upmem_hardware_sliced_resident_m2_1"
    assert raw["metadata"]["hardware_sliced_resident_m2_1_schema_version"] == (
        "upmem_hardware_sliced_resident_m2_1_v1"
    )
    assert raw["metadata"]["fixture_scope"] == (
        "two_operation_h_then_x_full_graph_replicated_prefix"
    )
    assert raw["metadata"]["require_nonzero_slice_partials"] is True
    assert suite["warmups"] == 1
    assert suite["repeats"] == 3
    assert len(suite["cases"]) == 1
    assert profile["requested_dpu_count"] == 2
    assert profile["slices"] == 2
    assert profile["tasklets_per_dpu"] == 1
    assert profile["performance_claim_applicable"] is False
    assert route["id"] == "upmem_tn_hardware_sliced_resident_two_dpu"
    assert route["options"]["backend_id"] == (
        "upmem_sdk_hardware_sliced_resident_two_dpu"
    )
    assert "no_speedup" in raw["metadata"]["claim_boundary"]
    assert "scaling" in raw["metadata"]["claim_boundary"]
    assert "energy" in raw["metadata"]["claim_boundary"]

    circuit = load_circuit(case, ROOT)
    network = build_tensor_network(circuit)
    graph = plan_task_graph_with_config(network, suite["planner"])
    assert circuit.n_qubits == 1
    assert [(operation.gate, operation.wires) for operation in circuit.operations] == [
        ("h", (0,)),
        ("x", (0,)),
    ]
    assert len(network.tensors) == 3
    assert len(graph.tasks) == 2
    assert graph.tasks[0].dependencies == ()
    assert graph.tasks[1].dependencies == (graph.tasks[0].id,)

    source_tensor_ids = {tensor.spec.id for tensor in network.tensors}
    assert graph.tasks[1].input_tensor_ids[1] == graph.tasks[0].output_tensor_id
    assert graph.tasks[0].output_tensor_id not in source_tensor_ids

    unsliced, _ = execute_task_sequence_np_einsum(graph, network)
    expected = np.asarray(case["expected_output"], dtype=np.complex128)
    np.testing.assert_allclose(unsliced, expected, atol=1.0e-12, rtol=1.0e-12)

    model = build_slice_aware_taskgraph_model(
        graph, max_slice_count=2, sliced_task_id=graph.tasks[1].id
    )
    internal_label = graph.tasks[1].contracted_labels[0]
    assert model.sliced_task_id == graph.tasks[1].id
    assert model.sliced_indices == (internal_label,)
    assert internal_label not in graph.tasks[0].contracted_labels
    assert [task.slice_id for task in model.slice_tasks] == [0, 1]

    tensors = {tensor.spec.id: tensor.array for tensor in network.tensors}
    prefix_task = graph.tasks[0]
    terminal_task = graph.tasks[1]
    prefix = np.einsum(
        "a,ab->b",
        tensors[prefix_task.input_tensor_ids[0]],
        tensors[prefix_task.input_tensor_ids[1]],
    )
    terminal_tensor_id = terminal_task.input_tensor_ids[0]
    terminal_tensor = tensors[terminal_tensor_id]
    partials = {
        slice_id: np.asarray(
            prefix[slice_id] * terminal_tensor[slice_id, :], dtype=np.complex128
        )
        for slice_id in (0, 1)
    }
    assert all(np.linalg.norm(partial) > 1.0e-7 for partial in partials.values())
    np.testing.assert_allclose(
        partials[0] + partials[1], unsliced, atol=1.0e-12, rtol=1.0e-12
    )

    prepared = mvp._prepare_case(
        ROOT,
        case,
        mvp.load_m2_suite(M2_1_SUITE_PATH),
    )
    plan = prepared["plan"]
    packages = build_two_slice_resident_graph_packages(
        plan,
        case_id=case["case_id"],
        suite_id=suite["suite_id"],
        quantization_mode=suite["metadata"]["quantization_mode"],
    )
    prefix_output_id = graph.tasks[0].output_tensor_id
    for item in packages:
        assert len(item.package.graph.tasks) == 2
        assert item.package.graph.tasks[0].output_tensor_id == prefix_output_id
        assert item.package.graph.tasks[1].dependencies == (graph.tasks[0].id,)
        prefix_slot = item.package.allocation.logical_to_slot[
            f"{prefix_output_id}::real"
        ]
        assert prefix_slot not in item.package.initial_data

    dpu_binary = tmp_path / "dpu_resident"
    dpu_binary.write_bytes(b"fixture")
    written = write_two_slice_resident_graph_packages(
        packages,
        tmp_path,
        dpu_binary=dpu_binary,
        request_id_prefix="m2-1-slot-alias-regression",
    )
    preflight = validate_written_two_slice_packages(plan, written)
    assert preflight["validated"] is True

    for item in written:
        package = item.package
        manifest = json.loads(package.manifest_path.read_text(encoding="utf-8"))
        binary_metadata = validate_resident_graph_package_file(package.package_path)
        initial_slots = {entry["slot_id"] for entry in manifest["initial_slots"]}
        final_slots = {entry["slot_id"] for entry in manifest["final_outputs"]}

        # The native parser requires these file-backed roles to be disjoint.
        assert initial_slots.isdisjoint(final_slots)
        assert initial_slots == set(binary_metadata["initial_slot_ids"])
        assert final_slots == set(binary_metadata["final_slot_ids"])
        assert binary_metadata["slot_count"] == 5
        assert binary_metadata["initial_slot_count"] == 3
        assert binary_metadata["final_output_component_count"] == 1
        assert package.allocation.mram_used_bytes == 40
        assert manifest["slot_descriptor_count"] == 5
        assert manifest["initial_h2d_bytes"] == 24
        assert manifest["descriptor_h2d_bytes"] == (
            5 * RESIDENT_SLOT_BYTES + 2 * RESIDENT_OPERATION_BYTES
        )
        assert manifest["control_h2d_bytes"] == 32
        assert manifest["final_d2h_bytes"] == 8
        assert (
            manifest["initial_h2d_bytes"]
            + manifest["descriptor_h2d_bytes"]
            + manifest["control_h2d_bytes"]
            + manifest["final_d2h_bytes"]
            == 1712
        )


def test_python_package_validator_rejects_native_dual_role_slot(tmp_path) -> None:
    loaded = mvp.load_m2_suite(M2_1_SUITE_PATH)
    case = loaded.suite["cases"][0]
    prepared = mvp._prepare_case(ROOT, case, loaded)
    packages = build_two_slice_resident_graph_packages(
        prepared["plan"],
        case_id=case["case_id"],
        suite_id=loaded.suite["suite_id"],
        quantization_mode="none",
    )
    package = packages[0].package
    payload = bytearray(
        _encode_package(package.allocation.slots, package.operations)
    )
    header = struct.unpack_from(RESIDENT_PACKAGE_HEADER_FORMAT, payload, 0)
    slot_offset = int(header[6])
    final_slot = package.allocation.final_components[0][1]
    encoded_id, offset, capacity, elements = struct.unpack_from(
        RESIDENT_SLOT_FORMAT,
        payload,
        slot_offset + final_slot * RESIDENT_SLOT_BYTES,
    )
    struct.pack_into(
        RESIDENT_SLOT_FORMAT,
        payload,
        slot_offset + final_slot * RESIDENT_SLOT_BYTES,
        encoded_id | RESIDENT_SLOT_INITIAL_FLAG,
        offset,
        capacity,
        elements,
    )
    header_values = list(header)
    header_values[14] += 1
    struct.pack_into(RESIDENT_PACKAGE_HEADER_FORMAT, payload, 0, *header_values)

    with pytest.raises(match="initial_final_slot_alias"):
        validate_resident_graph_package_bytes(bytes(payload))
