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
    result = materialize_task_inputs(
        TaskInputMaterializationRequest(graph=graph, initial_tensors=initial, target_task_index=1)
    )
    target_task = graph.tasks[1]
    replayed_task = graph.tasks[0]
    direct = np.einsum(
        replayed_task.index_expression,
        np.asarray(initial[replayed_task.input_tensor_ids[0]].array, dtype=np.complex128),
        np.asarray(initial[replayed_task.input_tensor_ids[1]].array, dtype=np.complex128),
        optimize=False,
    )
    direct = np.asarray(direct, dtype=np.complex128)

    assert result.status == "materialized"
    assert result.reason is None
    assert result.target_task_index == 1
    assert result.target_task_id == target_task.id
    assert result.selected_input_tensor_ids == target_task.input_tensor_ids
    assert result.replayed_task_count == 1
    assert result.replayed_task_ids == (replayed_task.id,)
    assert result.dead_tensor_release_implemented is False
    assert result.peak_materialized_bytes >= direct.nbytes
    assert result.left_tensor is not None
    assert result.right_tensor is not None
    assert result.input_sources[target_task.input_tensor_ids[0]]["source"] == "initial"
    assert result.input_sources[target_task.input_tensor_ids[1]]["source"] == "replayed"
    assert result.right_tensor.spec.id == replayed_task.output_tensor_id
    assert result.right_tensor.spec.labels == replayed_task.output_labels
    assert result.right_tensor.spec.shape == replayed_task.output_shape
    assert result.right_tensor.spec.dtype == "complex128"
    assert result.right_tensor.spec.produced_by == replayed_task.id
    np.testing.assert_allclose(result.right_tensor.array, direct)


def test_materialized_inputs_are_accepted_by_dense_preparation() -> None:
    graph, initial = _bell_graph_and_tensors()
    materialized = materialize_task_inputs(
        TaskInputMaterializationRequest(graph=graph, initial_tensors=initial, target_task_index=1)
    )

    assert materialized.left_tensor is not None
    assert materialized.right_tensor is not None
    prepared = prepare_dense_task(
        DenseTaskPreparationInput(
            task=graph.tasks[1],
            left_tensor=materialized.left_tensor,
            right_tensor=materialized.right_tensor,
            simplepim_probe=_available_probe(),
        )
    )

    assert prepared.status == "prepared"
    assert prepared.input_tensor_ids == graph.tasks[1].input_tensor_ids
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

    result = materialize_task_inputs(
        TaskInputMaterializationRequest(graph=bad_graph, initial_tensors=initial, target_task_index=1)
    )

    assert result.status == "failed"
    assert result.reason == "replay_output_shape_mismatch"
    assert result.error == "replay_output_shape_mismatch"


def test_materialization_json_omits_raw_arrays() -> None:
    graph, initial = _bell_graph_and_tensors()
    result = materialize_task_inputs(
        TaskInputMaterializationRequest(graph=graph, initial_tensors=initial, target_task_index=1)
    )
    payload = result.to_json_dict()
    encoded = json.dumps(payload)

    assert "left_tensor" not in payload
    assert "right_tensor" not in payload
    assert "array" not in encoded
    assert payload["status"] == "materialized"
