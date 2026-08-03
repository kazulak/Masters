from __future__ import annotations

from dataclasses import replace
import json

import numpy as np
import pytest

from quantum_bench.circuits import builtin_circuit, gate_matrix
from quantum_bench.core.records import (
    CircuitOperation,
    CircuitSpec,
    PathSummary,
    TaskGraph,
    TensorSpec,
    TensorValue,
)
from quantum_bench.formats.fixed_point import (
    FixedPointSpec,
    dequantize_fixed_point,
    quantize_fixed_point,
)
from quantum_bench.routing import GenericTaskPreparationInput, prepare_generic_task
from quantum_bench.tn import (
    build_execution_bundle,
    build_tensor_network,
    contraction_path_structure_hash,
    execute_task_sequence_np_einsum,
    execute_task_sliced_sequence_np_einsum,
    plan_task_graph,
    plan_task_graph_with_config,
    validate_execution_bundle,
    with_execution_identity,
)
from quantum_bench.tn.materialize import TaskInputMaterializationRequest, materialize_task_inputs
from quantum_bench.validation import compute_reference, validate

from .support import contraction_task, minimal_real_graph


def test_circuit_library_is_deterministic_and_unitary() -> None:
    first = builtin_circuit("quantization_stress", {"n_qubits": 4, "repeat_layers": 2})
    second = builtin_circuit("quantization_stress", {"n_qubits": 4, "repeat_layers": 2})

    assert first == second
    assert first.source["deterministic_unitary"] is True
    assert {operation.gate for operation in first.operations} == {"h", "rz", "cx"}
    for gate, params in (("h", ()), ("rz", (0.37,)), ("cx", ())):
        matrix = gate_matrix(gate, params)
        identity = np.eye(matrix.shape[0], dtype=np.complex128)
        np.testing.assert_allclose(matrix.conj().T @ matrix, identity, atol=1.0e-12)
        state = np.arange(1, matrix.shape[0] + 1, dtype=np.complex128)
        state /= np.linalg.norm(state)
        np.testing.assert_allclose(np.linalg.norm(matrix @ state), 1.0, atol=1.0e-12)

    composed = build_tensor_network(first)
    composed_state, _ = compute_reference(composed)
    np.testing.assert_allclose(np.linalg.norm(composed_state.ravel()), 1.0, atol=1.0e-12)


def test_ry_matrix_is_real_unitary_and_uses_gate_matrix_contract() -> None:
    theta = 0.73
    matrix = gate_matrix("ry", (theta,))
    half_theta = theta / 2.0
    expected = np.array(
        [
            [np.cos(half_theta), -np.sin(half_theta)],
            [np.sin(half_theta), np.cos(half_theta)],
        ],
        dtype=np.complex128,
    )

    assert matrix.dtype == np.complex128
    np.testing.assert_allclose(matrix, expected, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(matrix.conj().T @ matrix, np.eye(2), atol=1.0e-12)


def test_tn_lowering_preserves_output_labels_and_einsum_contract(minimal_graph) -> None:
    case = minimal_graph
    assert case.graph.network.output_labels == case.network.spec.output_labels
    assert case.graph.network.einsum_expression
    assert len(case.graph.tasks) == len(case.graph.path)
    assert all(task.output_labels for task in case.graph.tasks)
    assert case.graph.path_summary.missing_target_estimate_count == len(case.graph.tasks)


def test_task_graph_identity_is_deterministic(minimal_graph) -> None:
    second = minimal_real_graph()

    assert minimal_graph.graph.circuit_semantics_hash == second.graph.circuit_semantics_hash
    assert minimal_graph.graph.tensor_network_hash == second.graph.tensor_network_hash
    assert minimal_graph.graph.contraction_plan_hash == second.graph.contraction_plan_hash
    assert contraction_path_structure_hash(minimal_graph.graph) == contraction_path_structure_hash(second.graph)


def test_execution_bundle_validates_and_separates_executor_identity(minimal_graph) -> None:
    bundle = build_execution_bundle(minimal_graph.graph, case_id="bell_2q", suite_id="fixture")
    validate_execution_bundle(bundle, minimal_graph.graph)

    assert bundle["provenance"]["planning_in_timed_region"] is False
    assert bundle["contraction_plan_hash"] == minimal_graph.graph.contraction_plan_hash


def test_execution_bundle_rejects_changed_task(minimal_graph) -> None:
    task = replace(minimal_graph.graph.tasks[0], structure="changed")
    changed = with_execution_identity(
        replace(
            minimal_graph.graph,
            tasks=(task, *minimal_graph.graph.tasks[1:]),
            circuit_semantics_hash="",
            tensor_network_hash="",
            contraction_plan_hash="",
        )
    )

    assert changed.circuit_semantics_hash == minimal_graph.graph.circuit_semantics_hash
    assert changed.tensor_network_hash == minimal_graph.graph.tensor_network_hash
    assert changed.contraction_plan_hash != minimal_graph.graph.contraction_plan_hash


def test_stale_execution_identity_is_rejected(minimal_graph) -> None:
    with pytest.raises(ValueError, match="contraction_plan_hash"):
        with_execution_identity(replace(minimal_graph.graph, contraction_plan_hash="0" * 64))


@pytest.mark.parametrize(
    ("name", "params"),
    [("bell_2q", None), ("bv", {"n_qubits": 4}), ("xor", {"n_qubits": 4})],
)
def test_task_sequence_matches_exact_reference(name: str, params: dict | None) -> None:
    network = build_tensor_network(builtin_circuit(name, params))
    graph = plan_task_graph(network)
    actual, metadata = execute_task_sequence_np_einsum(graph, network)
    reference, _ = compute_reference(network)

    assert validate(actual, reference).passed
    assert metadata["task_count"] == len(graph.tasks)
    assert metadata["final_tensor_id"] == graph.tasks[-1].output_tensor_id
    assert metadata["peak_intermediate_bytes"] >= metadata["max_intermediate_tensor_bytes"]


def test_task_sequence_reorders_final_tensor_by_labels() -> None:
    network = build_tensor_network(builtin_circuit("bv", {"n_qubits": 4}))
    graph = plan_task_graph(network)
    actual, metadata = execute_task_sequence_np_einsum(graph, network)
    reference, _ = compute_reference(network)

    assert metadata["final_transpose_applied"] is True
    np.testing.assert_allclose(actual, reference, atol=1.0e-12)


def test_internal_slicing_preserves_taskgraph_result() -> None:
    network = build_tensor_network(builtin_circuit("bv", {"n_qubits": 4}))
    graph = plan_task_graph(network)
    expected, _ = execute_task_sequence_np_einsum(graph, network)
    actual, metadata = execute_task_sliced_sequence_np_einsum(graph, network, max_slice_count=2)

    np.testing.assert_allclose(actual, expected, atol=1.0e-12)
    assert metadata["slice_model_execution_status"] == "executed"
    assert metadata["slice_reconstruction_status"] == "completed"
    assert metadata["dependency_violation_detected"] is False


def test_empty_single_tensor_graph_returns_initial_state() -> None:
    circuit = builtin_circuit("qrng", {"n_qubits": 1})
    idle = type(circuit)(circuit.name, circuit.n_qubits, (), circuit.source)
    network = build_tensor_network(idle)
    graph = TaskGraph(
        network=network.spec,
        tasks=(),
        path=(),
        path_summary=PathSummary("fixture", "manual", 0, None, None, None, "fixture"),
        planning_time_s=0.0,
    )
    actual, metadata = execute_task_sequence_np_einsum(graph, network)

    assert metadata["task_count"] == 0
    np.testing.assert_array_equal(actual, np.array([1.0, 0.0], dtype=np.complex128))


def test_empty_multi_tensor_graph_is_rejected() -> None:
    circuit = builtin_circuit("qrng", {"n_qubits": 2})
    idle = type(circuit)(circuit.name, circuit.n_qubits, (), circuit.source)
    network = build_tensor_network(idle)
    graph = TaskGraph(
        network=network.spec,
        tasks=(),
        path=(),
        path_summary=PathSummary("fixture", "manual", 0, None, None, None, "fixture"),
        planning_time_s=0.0,
    )

    with pytest.raises(ValueError, match="empty TaskGraph"):
        execute_task_sequence_np_einsum(graph, network)


def test_duplicate_wire_is_rejected() -> None:
    circuit = CircuitSpec(
        "duplicate-wire",
        2,
        (CircuitOperation("cx", (0, 0)),),
        {},
    )

    with pytest.raises(ValueError, match="Duplicate wire 0"):
        build_tensor_network(circuit)


@pytest.mark.parametrize(
    ("dtype", "expected_scale"),
    [("int8", 1.0 / 127.0), ("int16", 1.0 / 32767.0)],
)
def test_fixed_point_scale_and_round_trip(dtype: str, expected_scale: float) -> None:
    converted = quantize_fixed_point(np.array([-1.0, 0.0, 1.0], dtype=np.float32), FixedPointSpec(route_dtype=dtype))

    assert converted.record.scale == pytest.approx(expected_scale)
    assert converted.record.rounding == "nearest_even"
    assert converted.record.clipping_count == 0
    assert converted.record.converted_bytes < converted.record.source_bytes
    assert np.max(np.abs(converted.record.dequantization_error.max_abs_error)) <= expected_scale


def test_fixed_point_rounding_and_clipping_boundaries() -> None:
    converted = quantize_fixed_point(
        np.array([0.5, 1.5, -0.5, -1.5, 127.5, -127.5], dtype=np.float32),
        FixedPointSpec(route_dtype="int8", scale=1.0),
    )

    np.testing.assert_array_equal(converted.array, np.array([0, 2, 0, -2, 127, -127], dtype=np.int8))
    assert converted.record.clipping_count == 2
    assert converted.record.saturation_count == 2


def test_int8_preparation_matches_einsum_int32_accumulation_and_dequantization() -> None:
    task = contraction_task("int8_reference", shape=(2, 2, 2))
    left = np.array([[0.5, 127.5], [-127.5, 1.5]], dtype=np.float32)
    right = np.array([[1.5, -127.5], [127.5, -0.5]], dtype=np.float32)
    preparation = GenericTaskPreparationInput(
        task=task,
        left_tensor=TensorValue(TensorSpec("int8_reference_left", task.left_labels, left.shape, "dense", dtype="float32"), left),
        right_tensor=TensorValue(TensorSpec("int8_reference_right", task.right_labels, right.shape, "dense", dtype="float32"), right),
        quantization_mode="per_task_input_quantize",
        fixed_point_spec=FixedPointSpec(route_dtype="int8", scale=1.0),
    )

    result = prepare_generic_task(preparation)

    assert result.status == "prepared"
    assert result.left_conversion is not None and result.right_conversion is not None
    operands = result.prepared_operands
    assert operands is not None
    expected_left = np.array([[0, 127], [-127, 2]], dtype=np.int8)
    expected_right = np.array([[2, -127], [127, 0]], dtype=np.int8)
    np.testing.assert_array_equal(operands.left_quantized, expected_left)
    np.testing.assert_array_equal(operands.right_quantized, expected_right)
    assert result.left_conversion.clipping_count == 2
    assert result.right_conversion.clipping_count == 2

    accumulator = np.einsum(
        "ik,kj->ij",
        expected_left.astype(np.int32),
        expected_right.astype(np.int32),
        dtype=np.int32,
    )
    np.testing.assert_array_equal(
        operands.expected_quantized_reference_output,
        accumulator.astype(np.float64),
    )
    np.testing.assert_allclose(
        dequantize_fixed_point(operands.left_quantized, result.left_conversion, dtype=np.float64),
        expected_left.astype(np.float64),
    )
    np.testing.assert_allclose(operands.expected_reference_output, accumulator.astype(np.float64))

    zero = np.zeros((2, 2), dtype=np.float32)
    zero_preparation = preparation.__class__(
        task=task,
        left_tensor=TensorValue(TensorSpec(task.input_tensor_ids[0], task.left_labels, zero.shape, "dense", dtype="float32"), zero),
        right_tensor=TensorValue(TensorSpec(task.input_tensor_ids[1], task.right_labels, zero.shape, "dense", dtype="float32"), zero),
        quantization_mode="per_task_input_quantize",
    )
    zero_result = prepare_generic_task(zero_preparation)
    assert zero_result.status == "prepared"
    assert zero_result.left_conversion is not None
    assert zero_result.right_conversion is not None
    assert zero_result.left_conversion.scale == 1.0
    assert zero_result.right_conversion.scale == 1.0
    np.testing.assert_array_equal(zero_result.prepared_operands.expected_reference_output, np.zeros((2, 2)))


def test_fixed_point_zero_and_error_metrics_are_safe() -> None:
    converted = quantize_fixed_point(np.zeros(4, dtype=np.float32))

    assert converted.record.scale == 1.0
    assert converted.record.dequantization_error.relative_l2_error == 0.0
    assert converted.record.status == "converted"


def test_fixed_point_complex_split_is_explicit_and_json_safe() -> None:
    converted = quantize_fixed_point(
        np.array([1.0 + 2.0j, -0.5 + 0.25j], dtype=np.complex64),
        FixedPointSpec(complex_policy="split_real_imag_last_axis"),
    )

    assert converted.record.representation == "split_complex_real_imag"
    assert converted.array.shape == (2, 2)
    assert json.dumps(converted.record.__dict__, default=lambda value: value.__dict__)


def test_fixed_point_complex_default_policy_rejects() -> None:
    with pytest.raises(ValueError, match="complex_policy"):
        quantize_fixed_point(np.array([1.0 + 1.0j]))


def test_fixed_point_rejects_unsupported_dtype_and_invalid_spec() -> None:
    with pytest.raises(ValueError, match="Unsupported source dtype"):
        quantize_fixed_point(np.array([1, 2], dtype=np.int32))
    with pytest.raises(ValueError, match="scale must be positive"):
        quantize_fixed_point(np.ones(2), FixedPointSpec(scale=0.0))


def test_materializer_replays_predecessors_without_raw_arrays_in_json(minimal_graph) -> None:
    initial = {tensor.spec.id: tensor for tensor in minimal_graph.network.tensors}
    target_index = next(
        index
        for index, task in enumerate(minimal_graph.graph.tasks)
        if any(tensor_id not in initial for tensor_id in task.input_tensor_ids)
    )
    result = materialize_task_inputs(
        TaskInputMaterializationRequest(minimal_graph.graph, initial, target_task_index=target_index)
    )

    payload = result.to_json_dict()
    assert result.status == "materialized"
    assert result.replayed_task_count == target_index
    assert "array" not in json.dumps(payload)


def test_materializer_reports_selection_boundaries(minimal_graph) -> None:
    initial = {tensor.spec.id: tensor for tensor in minimal_graph.network.tensors}
    out_of_range = materialize_task_inputs(TaskInputMaterializationRequest(minimal_graph.graph, initial, target_task_index=999))
    missing_id = materialize_task_inputs(TaskInputMaterializationRequest(minimal_graph.graph, initial, target_task_id="missing"))

    assert out_of_range.reason == "target_task_index_out_of_range"
    assert missing_id.reason == "target_task_id_not_found"


def test_planner_configuration_is_explicit_and_changes_only_plan_identity(minimal_graph) -> None:
    auto = plan_task_graph_with_config(minimal_graph.network, {"engine": "opt_einsum", "optimize": "auto"})
    greedy = plan_task_graph_with_config(minimal_graph.network, {"engine": "opt_einsum", "optimize": "greedy"})

    assert auto.network == greedy.network
    assert auto.path_summary.options["planner_config_hash"] != greedy.path_summary.options["planner_config_hash"]
    assert auto.circuit_semantics_hash == greedy.circuit_semantics_hash
    assert auto.tensor_network_hash == greedy.tensor_network_hash
