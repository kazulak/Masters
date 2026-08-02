from __future__ import annotations

from pathlib import Path

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


def test_m2_1_fixture_builds_dependent_hx_graph_and_useful_cpu_slices() -> None:
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
