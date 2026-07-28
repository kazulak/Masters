from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

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


ROOT = Path(__file__).resolve().parents[1]
SUITE_PATH = ROOT / "configs" / "suites" / "upmem_hardware_sliced_resident_mvp.yml"


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
    assert profile["performance_claim_applicable"] is False
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
