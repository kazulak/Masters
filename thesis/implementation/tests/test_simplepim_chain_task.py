import numpy as np
import pytest

import quantum_bench.targets.upmem.hardware_taskgraph_resident as resident
from quantum_bench.targets.upmem.simplepim_chain_task import (
    CHAIN_EXPECTED_PATH,
    CHAIN_INPUT_IDS,
    CHAIN_INTERMEDIATE_ID,
    CHAIN_OUTPUT_ID,
    build_simplepim_chain_workload,
    validate_simplepim_chain_workload,
)
from quantum_bench.targets.upmem.hardware_taskgraph_resident import (
    build_resident_graph_package,
    build_resident_policy_reference,
    validate_resident_graph_package_bytes,
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
    assert workload.graph.path == CHAIN_EXPECTED_PATH
    assert first.input_tensor_ids == CHAIN_INPUT_IDS[:2]
    assert first.output_tensor_id == CHAIN_INTERMEDIATE_ID
    assert second.input_tensor_ids == (CHAIN_INPUT_IDS[2], CHAIN_INTERMEDIATE_ID)
    assert second.dependencies == (first.id,)
    assert second.output_tensor_id == CHAIN_OUTPUT_ID


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
    package_metadata = validate_resident_graph_package_bytes(
        resident._encode_package(package.allocation.slots, package.operations)
    )
    assert len(package.operations) == 2
    input_slots = {
        package.allocation.logical_to_slot[f"{tensor_id}::real"]
        for tensor_id in CHAIN_INPUT_IDS
    }
    intermediate_slot = package.allocation.logical_to_slot[f"{CHAIN_INTERMEDIATE_ID}::real"]
    final_slot = package.allocation.logical_to_slot[f"{CHAIN_OUTPUT_ID}::real"]
    assert intermediate_slot not in input_slots
    assert final_slot not in input_slots | {intermediate_slot}
    assert package_metadata["initial_slot_count"] == len(CHAIN_INPUT_IDS)
    assert package_metadata["final_output_component_count"] == 1
    assert package_metadata["slot_count"] == len(package.allocation.slots)
    assert package_metadata["operation_count"] == len(package.operations)
    assert package_metadata["initial_slot_ids"] == sorted(input_slots)
    assert package_metadata["final_slot_ids"] == [final_slot]
    assert {slot["slot_id"] for slot in package_metadata["slot_descriptors"] if slot["initial"]} == input_slots
    assert {slot["slot_id"] for slot in package_metadata["slot_descriptors"] if slot["final"]} == {final_slot}
    assert {
        package.operations[0].slot_a,
        package.operations[0].slot_b,
        package.operations[1].slot_a,
        package.operations[1].slot_b,
        package.operations[0].slot_out_real,
        package.operations[1].slot_out_real,
    } == input_slots | {intermediate_slot, final_slot}
    assert package.operations[0].slot_a == package.allocation.logical_to_slot[f"{CHAIN_INPUT_IDS[0]}::real"]
    assert package.operations[0].slot_b == package.allocation.logical_to_slot[f"{CHAIN_INPUT_IDS[1]}::real"]
    assert package.operations[0].slot_out_real == intermediate_slot
    assert package.operations[1].slot_a == package.allocation.logical_to_slot[f"{CHAIN_INPUT_IDS[2]}::real"]
    assert package.operations[1].slot_b == intermediate_slot
    assert package.operations[1].slot_out_real == final_slot
    assert package.allocation.mram_used_bytes > 0


def test_chain_resident_policy_reference_matches_fixture_reference() -> None:
    workload = build_simplepim_chain_workload()
    result = build_resident_policy_reference(workload.graph, workload.network, quantization_mode="none")
    assert result["status"] == "completed"
    assert result["output_hash"]
    output = np.asarray(result["output"]).reshape(())
    assert float(np.imag(output)) == 0.0
    assert int(np.real(output)) == workload.reference_int64
    assert result["task_metrics"][-1]["task_id"] == workload.graph.tasks[-1].id
