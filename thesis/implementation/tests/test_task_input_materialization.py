from __future__ import annotations

import json
from dataclasses import replace

import numpy as np

from quantum_bench.circuits import builtin_circuit
from quantum_bench.routing import DenseTaskPreparationInput, prepare_dense_task
from quantum_bench.targets.upmem import SimplePimProbeResult
from quantum_bench.tn import (
    TaskInputMaterializationRequest,
    build_tensor_network,
    materialize_task_inputs,
    plan_task_graph,
)
from quantum_bench.tn.contract import contract_binary_task


def _available_probe() -> SimplePimProbeResult:
    return SimplePimProbeResult(
        simplepim_available=True,
        simplepim_probe_status="available",
        simplepim_bin="/tmp/simplepim",
        simplepim_command_path="/tmp/simplepim",
        metadata={"external_command_executed": False, "source": "unit_test"},
    )


def _bell_graph_and_tensors():
    circuit = builtin_circuit("bell_2q")
    network = build_tensor_network(circuit)
    graph = plan_task_graph(network)
    initial = {tensor.spec.id: tensor for tensor in network.tensors}
    return graph, initial


def test_materializes_later_task_inputs_by_replaying_predecessors() -> None:
    graph, initial = _bell_graph_and_tensors()
    target_index = _first_task_with_intermediate_inputs(graph, initial)
    result = materialize_task_inputs(
        TaskInputMaterializationRequest(graph=graph, initial_tensors=initial, target_task_index=target_index)
    )
    target_task = graph.tasks[target_index]
    replayed_tasks = graph.tasks[:target_index]
    direct_outputs = {
        task.output_tensor_id: contract_binary_task(
            task,
            np.asarray(initial[task.input_tensor_ids[0]].array, dtype=np.complex128),
            np.asarray(initial[task.input_tensor_ids[1]].array, dtype=np.complex128),
        )
        for task in replayed_tasks
    }

    assert result.status == "materialized"
    assert result.reason is None
    assert result.target_task_index == target_index
    assert result.target_task_id == target_task.id
    assert result.selected_input_tensor_ids == target_task.input_tensor_ids
    assert result.replayed_task_count == len(replayed_tasks)
    assert result.replayed_task_ids == tuple(task.id for task in replayed_tasks)
    assert result.dead_tensor_release_implemented is False
    assert result.peak_materialized_bytes >= sum(output.nbytes for output in direct_outputs.values())
    assert result.left_tensor is not None
    assert result.right_tensor is not None
    assert result.input_sources[target_task.input_tensor_ids[0]]["source"] == "replayed"
    assert result.input_sources[target_task.input_tensor_ids[1]]["source"] == "replayed"
    assert result.left_tensor.spec.id in direct_outputs
    assert result.right_tensor.spec.id in direct_outputs
    assert result.left_tensor.spec.dtype == "complex128"
    assert result.right_tensor.spec.dtype == "complex128"
    np.testing.assert_allclose(result.left_tensor.array, direct_outputs[result.left_tensor.spec.id])
    np.testing.assert_allclose(result.right_tensor.array, direct_outputs[result.right_tensor.spec.id])


def test_materialized_inputs_are_accepted_by_dense_preparation() -> None:
    graph, initial = _bell_graph_and_tensors()
    target_index = _first_task_with_intermediate_inputs(graph, initial)
    materialized = materialize_task_inputs(
        TaskInputMaterializationRequest(graph=graph, initial_tensors=initial, target_task_index=target_index)
    )

    assert materialized.left_tensor is not None
    assert materialized.right_tensor is not None
    prepared = prepare_dense_task(
        DenseTaskPreparationInput(
            task=graph.tasks[target_index],
            left_tensor=materialized.left_tensor,
            right_tensor=materialized.right_tensor,
            simplepim_probe=_available_probe(),
        )
    )

    assert prepared.status == "prepared"
    assert prepared.input_tensor_ids == graph.tasks[target_index].input_tensor_ids
    assert prepared.prepared_operands is not None


def test_initial_input_task_returns_stable_input_sources_without_replay() -> None:
    graph, initial = _bell_graph_and_tensors()
    result = materialize_task_inputs(
        TaskInputMaterializationRequest(graph=graph, initial_tensors=initial, target_task_index=0)
    )
    task = graph.tasks[0]

    assert result.status == "initial_inputs_available"
    assert result.reason is None
    assert result.replayed_task_count == 0
    assert result.replayed_task_ids == ()
    assert result.left_tensor is initial[task.input_tensor_ids[0]]
    assert result.right_tensor is initial[task.input_tensor_ids[1]]
    assert set(result.input_sources) == set(task.input_tensor_ids)
    assert all(source["source"] == "initial" for source in result.input_sources.values())


def test_invalid_target_selection_returns_explicit_reasons() -> None:
    graph, initial = _bell_graph_and_tensors()

    out_of_range = materialize_task_inputs(
        TaskInputMaterializationRequest(graph=graph, initial_tensors=initial, target_task_index=999)
    )
    missing_id = materialize_task_inputs(
        TaskInputMaterializationRequest(graph=graph, initial_tensors=initial, target_task_id="missing")
    )
    ambiguous = materialize_task_inputs(
        TaskInputMaterializationRequest(
            graph=graph,
            initial_tensors=initial,
            target_task_index=0,
            target_task_id=graph.tasks[0].id,
        )
    )

    assert out_of_range.status == "unsupported"
    assert out_of_range.reason == "target_task_index_out_of_range"
    assert missing_id.status == "unsupported"
    assert missing_id.reason == "target_task_id_not_found"
    assert ambiguous.status == "unsupported"
    assert ambiguous.reason == "ambiguous_target_selection"


def test_replay_output_shape_mismatch_returns_failed_result() -> None:
    graph, initial = _bell_graph_and_tensors()
    bad_first_task = replace(graph.tasks[0], output_shape=(999,))
    bad_graph = replace(graph, tasks=(bad_first_task, *graph.tasks[1:]))
    target_index = _first_task_with_intermediate_inputs(bad_graph, initial)

    result = materialize_task_inputs(
        TaskInputMaterializationRequest(graph=bad_graph, initial_tensors=initial, target_task_index=target_index)
    )

    assert result.status == "failed"
    assert result.reason == "replay_output_shape_mismatch"
    assert result.error == "replay_output_shape_mismatch"


def test_materialization_json_omits_raw_arrays() -> None:
    graph, initial = _bell_graph_and_tensors()
    target_index = _first_task_with_intermediate_inputs(graph, initial)
    result = materialize_task_inputs(
        TaskInputMaterializationRequest(graph=graph, initial_tensors=initial, target_task_index=target_index)
    )
    payload = result.to_json_dict()
    encoded = json.dumps(payload)

    assert "left_tensor" not in payload
    assert "right_tensor" not in payload
    assert "array" not in encoded
    assert payload["status"] == "materialized"


def _first_task_with_intermediate_inputs(graph, initial: dict) -> int:
    initial_ids = set(initial)
    for index, task in enumerate(graph.tasks):
        if any(tensor_id not in initial_ids for tensor_id in task.input_tensor_ids):
            return index
    raise AssertionError("expected at least one task with intermediate inputs")
