import numpy as np
import pytest

from quantum_bench.targets.upmem.simplepim_chain_task import (
    CHAIN_INTERMEDIATE_ID,
    build_simplepim_chain_workload,
    validate_simplepim_chain_workload,
)
from quantum_bench.targets.upmem.hardware_taskgraph_resident import (
    build_resident_graph_package,
    build_resident_policy_reference,
)


def test_chain_fixture_is_deterministic_and_exact() -> None:
    first = build_simplepim_chain_workload()
    second = build_simplepim_chain_workload()
    assert first.reference_int64 == -11654
    assert first.operand_sha256 == second.operand_sha256
    assert all(np.array_equal(left, right) for left, right in zip(first.operands, second.operands, strict=True))


def test_chain_has_two_tasks_and_resident_intermediate_dependency() -> None:
    workload = build_simplepim_chain_workload()
    first, second = workload.graph.tasks
    assert first.output_tensor_id == CHAIN_INTERMEDIATE_ID
    assert second.dependencies == (first.id,)
    assert CHAIN_INTERMEDIATE_ID in second.input_tensor_ids


def test_chain_validates_and_has_four_tiles() -> None:
    workload = build_simplepim_chain_workload()
    validate_simplepim_chain_workload(workload)
    assert [(tile.start, tile.stop) for tile in workload.tiles] == [(0, 64), (64, 128), (128, 192), (192, 256)]


def test_chain_rejects_wrong_dependency() -> None:
    workload = build_simplepim_chain_workload()
    tasks = list(workload.graph.tasks)
    tasks[1] = tasks[1].__class__(**{**tasks[1].__dict__, "dependencies": ()})
    broken = workload.__class__(**{**workload.__dict__, "graph": workload.graph.__class__(**{**workload.graph.__dict__, "tasks": tuple(tasks)})})
    with pytest.raises(ValueError, match="dependency"):
        validate_simplepim_chain_workload(broken)


def test_chain_lowers_to_resident_package_with_intermediate_slot() -> None:
    workload = build_simplepim_chain_workload()
    package = build_resident_graph_package(
        workload.graph,
        workload.network,
        case_id=workload.graph.network.circuit.name,
        suite_id="phase_a_local",
        quantization_mode="none",
    )
    assert len(package.operations) == 2
    assert package.allocation.logical_to_slot[CHAIN_INTERMEDIATE_ID + "::real"] not in {
        package.allocation.logical_to_slot[tensor.spec.id + "::real"] for tensor in workload.network.tensors
    }
    assert package.allocation.mram_used_bytes > 0


def test_chain_resident_policy_reference_matches_fixture_reference() -> None:
    workload = build_simplepim_chain_workload()
    result = build_resident_policy_reference(workload.graph, workload.network, quantization_mode="none")
    assert result["status"] == "completed"
    assert result["output_hash"]
    assert int(np.real(np.asarray(result["output"]).reshape(()))) == workload.reference_int64
    assert result["task_metrics"][-1]["task_id"] == workload.graph.tasks[-1].id
