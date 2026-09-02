from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import quantum_bench.quantized_contraction as qc
from quantum_bench.circuits import builtin_circuit
from quantum_bench.cpu import run_cpu_once
from quantum_bench.lowering import build_contraction_dag, lower_tensor_network
from quantum_bench.model import (
    ContractNode,
    ContractionDAG,
    ReduceNode,
    TensorSpec,
    TensorView,
    make_simulation_job,
)
from quantum_bench.planning import plan_opt_einsum


HISTORICAL_FLOAT32 = "split_complex_float32_v1"
HISTORICAL_INT8 = "split_complex_int8_shared_scale_v1"
DATA_PATH = Path(__file__).parent / "data" / "complex_int8_shared_scale_v1.json"


def _contract_node(
    *,
    left_id: str = "left",
    right_id: str = "right",
    output_id: str = "output",
    left_labels: tuple[int, ...] = (0, 1),
    right_labels: tuple[int, ...] = (1, 2),
    left_shape: tuple[int, ...] = (2, 3),
    right_shape: tuple[int, ...] = (3, 4),
    output_labels: tuple[int, ...] = (0, 2),
    output_shape: tuple[int, ...] = (2, 4),
    contracted_labels: tuple[int, ...] = (1,),
    dependencies: tuple[str, ...] = (),
) -> ContractNode:
    left = TensorSpec(left_id, left_labels, left_shape, "dense", dtype="complex128")
    right = TensorSpec(right_id, right_labels, right_shape, "dense", dtype="complex128")
    output = TensorSpec(
        output_id,
        output_labels,
        output_shape,
        "dense",
        dtype="complex128",
        produced_by="node",
    )
    return ContractNode(
        node_id=output_id,
        left=TensorView(tensor_id=left.id, labels=left.labels, shape=left.shape),
        right=TensorView(tensor_id=right.id, labels=right.labels, shape=right.shape),
        output=output,
        contracted_labels=contracted_labels,
        output_labels=output_labels,
        dependencies=dependencies,
    )


def _chain_dag() -> tuple[ContractionDAG, dict[str, np.ndarray]]:
    a = TensorSpec("a", (0, 1), (2, 3), "dense", dtype="complex128")
    b = TensorSpec("b", (1, 2), (3, 4), "dense", dtype="complex128")
    c = TensorSpec("c", (2, 3), (4, 2), "dense", dtype="complex128")
    p = TensorSpec("p", (0, 2), (2, 4), "dense", dtype="complex128", produced_by="first")
    q = TensorSpec("q", (0, 3), (2, 2), "dense", dtype="complex128", produced_by="second")
    first = ContractNode(
        node_id="first",
        left=TensorView(tensor_id="a", labels=a.labels, shape=a.shape),
        right=TensorView(tensor_id="b", labels=b.labels, shape=b.shape),
        output=p,
        contracted_labels=(1,),
        output_labels=(0, 2),
    )
    second = ContractNode(
        node_id="second",
        left=TensorView(tensor_id="p", labels=p.labels, shape=p.shape),
        right=TensorView(tensor_id="c", labels=c.labels, shape=c.shape),
        output=q,
        contracted_labels=(2,),
        output_labels=(0, 3),
        dependencies=("first",),
    )
    dag = ContractionDAG(
        tensors=(a, b, c),
        nodes=(second, first),
        output=TensorView(tensor_id="q", labels=q.labels, shape=q.shape),
    )
    inputs = {
        "a": np.array(
            [[1.0 + 0.3j, -2.1 + 3.7j, 0.01 - 0.02j], [4.2 - 1.4j, -0.4 + 0.8j, 2.2 + 1.9j]],
            dtype=np.complex128,
        ),
        "b": np.array(
            [[0.7 - 1.2j, 2.5 + 0.1j, -4.0 + 0.5j, 0.03j], [3.1 + 2.4j, -0.8 - 1.7j, 1.3 + 0.2j, 2.8 - 0.6j], [5.0 - 0.4j, 0.6 + 3.2j, -1.1 + 0.9j, -2.4 - 1.5j]],
            dtype=np.complex128,
        ),
        "c": np.array(
            [[1.0 - 0.2j, -0.4 + 0.7j], [2.2 + 0.4j, 0.1 - 1.3j], [0.02 + 0.03j, 4.0 - 0.6j], [1.8 + 1.1j, -2.0 + 0.5j]],
            dtype=np.complex128,
        ),
    }
    return dag, inputs


def _dag_from_circuit(name: str, params: dict | None = None) -> tuple[ContractionDAG, dict[str, np.ndarray]]:
    network, inputs = lower_tensor_network(
        make_simulation_job(builtin_circuit(name, params))
    )
    path, _ = plan_opt_einsum(network, optimize="greedy")
    return build_contraction_dag(network, path), inputs


def test_golden_policy_vectors_are_exact() -> None:
    payload = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    assert payload["policy_id"] == qc.POLICY_ID
    for vector in payload["vectors"]:
        value = np.array(
            [complex(item["real"], item["imag"]) for item in vector["input"]],
            dtype=np.complex128,
        )
        encoded = qc.quantize_complex_shared_scale(value)
        assert encoded.scale == vector["scale"]
        np.testing.assert_array_equal(encoded.q_real, np.array(vector["q_real"], dtype=np.int8))
        np.testing.assert_array_equal(encoded.q_imag, np.array(vector["q_imag"], dtype=np.int8))
        expected = np.array(
            [complex(item["real"], item["imag"]) for item in vector["dequantized"]],
            dtype=np.complex64,
        )
        np.testing.assert_array_equal(qc.dequantize_complex(encoded), expected)


def test_policy_id_scale_dtype_shared_planes_and_read_only_payload() -> None:
    source = np.array([1.0 + 4.0j, -2.0 - 3.0j], dtype=np.complex128)
    encoded = qc.quantize_complex_shared_scale(source)
    assert qc.POLICY_ID == "complex_int8_shared_scale_v1"
    assert type(encoded.scale) is float
    assert encoded.scale == 4.0 / 127.0
    assert encoded.q_real.dtype == np.dtype(np.int8)
    assert encoded.q_imag.dtype == np.dtype(np.int8)
    assert not encoded.q_real.flags.writeable
    assert not encoded.q_imag.flags.writeable
    with pytest.raises((ValueError, TypeError)):
        encoded.q_real[0] = 0
    source[0] = 99.0 + 99.0j
    assert encoded.q_imag[0] == 127


def test_ties_to_even_for_positive_and_negative_halfway_values() -> None:
    values = np.array(
        [0.5, 1.5, 2.5, -0.5, -1.5, -2.5, 127.0], dtype=np.complex128
    )
    encoded = qc.quantize_complex_shared_scale(values)
    np.testing.assert_array_equal(encoded.q_real, np.array([0, 2, 2, 0, -2, -2, 127], dtype=np.int8))
    np.testing.assert_array_equal(encoded.q_imag, np.zeros(7, dtype=np.int8))


@pytest.mark.parametrize(
    "value",
    [
        np.zeros((2, 3), dtype=np.complex128),
        np.array([1.0, -2.0, 3.0], dtype=np.float64),
        np.array([1j, -2j, 3j], dtype=np.complex128),
        np.array([1.0 + 1.0j, -2.0 - 2.0j], dtype=np.complex128),
        np.array([1.0e-30 + 2.0e-30j, -3.0e-30j], dtype=np.complex128),
        np.array([1.0e300 + 1.0e-300j, -2.0e299 + 3.0e-301j], dtype=np.complex128),
        np.array([1.0, -1.0, 127.0, -127.0], dtype=np.float64),
    ],
)
def test_adversarial_finite_inputs_have_symmetric_int8_diagnostics(value: np.ndarray) -> None:
    encoded = qc.quantize_complex_shared_scale(value)
    assert np.all(encoded.q_real >= -127)
    assert np.all(encoded.q_real <= 127)
    assert np.all(encoded.q_imag >= -127)
    assert np.all(encoded.q_imag <= 127)
    assert not np.any(encoded.q_real == -128)
    assert not np.any(encoded.q_imag == -128)
    assert encoded.diagnostics.max_abs_component >= 0.0
    if np.any(np.asarray(value) != 0):
        assert encoded.diagnostics.min_nonzero_abs_component is not None


def test_zero_rule_and_exact_boundary_counts() -> None:
    zero = qc.quantize_complex_shared_scale(np.zeros(4, dtype=np.complex128))
    assert zero.scale == 1.0
    assert zero.diagnostics.zero_count == 8
    assert zero.diagnostics.clipping_count == 0

    boundary = qc.quantize_complex_shared_scale(
        np.array([127.0 - 127.0j, 0.0 + 0.0j], dtype=np.complex128)
    )
    assert boundary.diagnostics.boundary_saturation_count == 2
    assert boundary.diagnostics.clipping_count == 0
    assert boundary.diagnostics.q_real_min == 0
    assert boundary.diagnostics.q_imag_min == -127


def test_clipping_is_explicit_and_never_emits_minus_128() -> None:
    rounded, clipping = qc._quantize_plane(np.array([128.0, -128.0, 127.0, -127.0]), 1.0)
    np.testing.assert_array_equal(rounded, np.array([127, -127, 127, -127], dtype=np.int8))
    assert clipping == 2
    assert not np.any(rounded == -128)


@pytest.mark.parametrize("bad", [np.array([np.nan + 0j]), np.array([np.inf + 0j]), np.array([-np.inf + 0j])])
def test_nonfinite_inputs_are_rejected_without_sanitization(bad: np.ndarray) -> None:
    with pytest.raises(ValueError, match="finite"):
        qc.quantize_complex_shared_scale(bad)


def test_scale_underflow_is_rejected_explicitly() -> None:
    tiny = np.array([np.nextafter(0.0, 1.0) + 0.0j], dtype=np.complex128)
    with pytest.raises(ValueError, match="underflow"):
        qc.quantize_complex_shared_scale(tiny)


def test_noncontiguous_input_is_encoded_as_a_contiguous_read_only_copy() -> None:
    source = np.arange(48, dtype=np.float64).reshape(6, 8)[:, ::2]
    assert not source.flags.c_contiguous
    encoded = qc.quantize_complex_shared_scale(source + 1j * source[:, ::-1])
    assert encoded.q_real.flags.c_contiguous
    assert encoded.q_imag.flags.c_contiguous
    assert not encoded.q_real.flags.writeable


def test_dequantize_accepts_explicit_planes_and_returns_historical_complex64() -> None:
    q_real = np.array([1, -2], dtype=np.int8)
    q_imag = np.array([3, -4], dtype=np.int8)
    decoded = qc.dequantize_complex(q_real, q_imag, 0.5)
    assert decoded.dtype == np.dtype(np.complex64)
    assert not decoded.flags.writeable
    np.testing.assert_array_equal(decoded, np.array([0.5 + 1.5j, -1.0 - 2.0j], dtype=np.complex64))
    with pytest.raises(ValueError, match="outside"):
        qc.dequantize_complex(np.array([-128], dtype=np.int8), np.array([0], dtype=np.int8), 1.0)


def test_integer_contraction_has_four_labeled_products_and_exact_signs() -> None:
    node = _contract_node(
        left_shape=(2, 3), right_shape=(3, 4), output_shape=(2, 4)
    )
    a_real = np.array([[1, -2, 3], [4, 5, -6]], dtype=np.int8)
    a_imag = np.array([[2, 3, -1], [-2, 1, 4]], dtype=np.int8)
    b_real = np.array([[3, -1, 2, 4], [5, 2, -3, 1], [-2, 4, 1, -5]], dtype=np.int8)
    b_imag = np.array([[1, 4, -2, 3], [-3, 2, 5, -1], [2, -4, 1, 6]], dtype=np.int8)
    a = qc.QuantizedComplexTensor(a_real, a_imag, 1.0)
    b = qc.QuantizedComplexTensor(b_real, b_imag, 1.0)
    products = qc.contract_integer_products(a, b, node)
    assert all(array.dtype == np.dtype(np.int64) for array in products.as_tuple())
    np.testing.assert_array_equal(products.rr, a_real.astype(np.int64) @ b_real.astype(np.int64))
    np.testing.assert_array_equal(products.ii, a_imag.astype(np.int64) @ b_imag.astype(np.int64))
    np.testing.assert_array_equal(products.ri, a_real.astype(np.int64) @ b_imag.astype(np.int64))
    np.testing.assert_array_equal(products.ir, a_imag.astype(np.int64) @ b_real.astype(np.int64))
    decoded = qc.decode_integer_products(products, a.scale, b.scale)
    expected = (a_real.astype(np.int64) + 1j * a_imag.astype(np.int64)) @ (
        b_real.astype(np.int64) + 1j * b_imag.astype(np.int64)
    )
    np.testing.assert_array_equal(decoded, expected.astype(np.complex64))


def test_scalar_k1_and_odd_non_square_contractions() -> None:
    scalar_node = _contract_node(
        output_id="scalar",
        left_labels=(),
        right_labels=(),
        left_shape=(),
        right_shape=(),
        output_labels=(),
        output_shape=(),
        contracted_labels=(),
    )
    scalar_a = qc.QuantizedComplexTensor(np.array(2, dtype=np.int8), np.array(1, dtype=np.int8), 1.0)
    scalar_b = qc.QuantizedComplexTensor(np.array(3, dtype=np.int8), np.array(4, dtype=np.int8), 1.0)
    scalar_products = qc.contract_integer_products(scalar_a, scalar_b, scalar_node)
    scalar = qc.decode_integer_products(scalar_products, 1.0, 1.0)
    assert scalar.shape == ()
    assert scalar.item() == -2.0 + 11.0j

    odd_node = _contract_node()
    geometry = qc.contraction_geometry(odd_node)
    assert (geometry.B, geometry.M, geometry.N, geometry.K) == (1, 2, 4, 3)
    left = qc.quantize_complex_shared_scale(np.arange(6, dtype=np.float64).reshape(2, 3) + 1j)
    right = qc.quantize_complex_shared_scale(np.arange(12, dtype=np.float64).reshape(3, 4) - 2j)
    result, facts = qc.contract_complex_int8_reference(left, right, odd_node)
    assert result.shape == (2, 4)
    assert (facts.B, facts.M, facts.N, facts.K) == (1, 2, 4, 3)
    assert facts.integer_multiply_accumulate_count == 4 * 2 * 4 * 3


def test_accumulator_bounds_use_full_component_bound_at_exact_edges() -> None:
    safe_k = qc.INT32_MAX // (2 * qc.INT8_MAX**2)
    assert qc.int32_accumulator_safe(safe_k)
    assert not qc.int32_accumulator_safe(safe_k + 1)
    safe = qc.accumulator_bounds(safe_k)
    unsafe = qc.accumulator_bounds(safe_k + 1)
    assert safe.lane_bound == safe_k * 127**2
    assert safe.component_bound <= qc.INT32_MAX
    assert unsafe.component_bound > qc.INT32_MAX
    assert qc.int64_accumulator_safe(1)
    assert qc.theoretical_accumulator_bound(3) == 2 * 3 * 127**2


def test_historical_cpu_int8_is_equal_to_the_new_int64_reference_when_safe() -> None:
    node = _contract_node()
    a_spec = TensorSpec("left", (0, 1), (2, 3), "dense", dtype="complex128")
    b_spec = TensorSpec("right", (1, 2), (3, 4), "dense", dtype="complex128")
    dag = ContractionDAG(
        tensors=(a_spec, b_spec),
        nodes=(node,),
        output=TensorView(tensor_id="output", labels=(0, 2), shape=(2, 4)),
    )
    inputs = {
        "left": np.array([[1 + 2j, -3 + 4j, 2 - 1j], [4 - 2j, 1 + 3j, -2 - 5j]], dtype=np.complex128),
        "right": np.array([[2 - 1j, 1 + 2j, -3 + 1j, 4], [1 + 3j, -2 + 1j, 5 - 2j, 2 + 4j], [3, -1 - 2j, 1 + 1j, -4 + 2j]], dtype=np.complex128),
    }
    historical = run_cpu_once(dag, inputs, HISTORICAL_INT8).output
    ours, _ = qc.contract_complex_int8_reference(inputs["left"], inputs["right"], node)
    np.testing.assert_array_equal(ours, historical)


def test_quantized_replay_preserves_order_shape_and_requantizes_chain() -> None:
    dag, inputs = _chain_dag()
    replay = qc.replay_quantized_dag(dag, inputs)
    replay_again = qc.replay_quantized_dag(dag, inputs)
    assert tuple(trace.node_id for trace in replay.traces) == ("first", "second")
    assert replay.output.shape == (2, 2)
    assert replay.output.dtype == np.dtype(np.complex64)
    assert replay.float32_output.shape == (2, 2)
    assert replay.quantized_intermediates["p"].dtype == np.dtype(np.complex64)
    assert replay.traces[0].requantization_event_count == 0
    assert replay.traces[1].requantization_event_count == 1
    assert replay.traces == replay_again.traces
    np.testing.assert_array_equal(replay.output, replay_again.output)
    np.testing.assert_array_equal(replay.float32_output, run_cpu_once(dag, inputs, HISTORICAL_FLOAT32).output)


def test_local_and_cumulative_error_are_separate_records() -> None:
    dag, inputs = _chain_dag()
    replay = qc.replay_quantized_dag(dag, inputs)
    first, second = replay.traces
    assert first.local_max_abs_error_vs_same_node_float32 == first.cumulative_max_abs_error_vs_same_node_float32
    assert second.local_max_abs_error_vs_same_node_float32 != second.cumulative_max_abs_error_vs_same_node_float32
    assert second.observed_local_error == second.local_max_abs_error_vs_same_node_float32
    assert second.ideal_local_max_abs_error >= 0.0
    assert second.theoretical_local_error_bound >= second.ideal_local_max_abs_error


def test_reduce_nodes_are_complex64_and_unquantized() -> None:
    left = TensorSpec("left", (0,), (2,), "dense", dtype="complex128")
    right = TensorSpec("right", (0,), (2,), "dense", dtype="complex128")
    output = TensorSpec("sum", (0,), (2,), "dense", dtype="complex128", produced_by="reduce")
    dag = ContractionDAG(
        tensors=(left, right),
        nodes=(
            ReduceNode(
                node_id="reduce",
                inputs=(
                    TensorView(tensor_id="left", labels=(0,), shape=(2,)),
                    TensorView(tensor_id="right", labels=(0,), shape=(2,)),
                ),
                output=output,
            ),
        ),
        output=TensorView(tensor_id="sum", labels=(0,), shape=(2,)),
    )
    inputs = {
        "left": np.array([1 + 2j, 3 - 4j], dtype=np.complex128),
        "right": np.array([5 - 1j, -2 + 7j], dtype=np.complex128),
    }
    replay = qc.replay_quantized_dag(dag, inputs)
    assert replay.traces == ()
    np.testing.assert_array_equal(
        replay.output,
        np.add.reduce(
            (inputs["left"].astype(np.complex64), inputs["right"].astype(np.complex64)),
            axis=0,
            dtype=np.complex64,
        ),
    )


def test_final_metrics_are_phase_sensitive_and_retain_float32_floor() -> None:
    dag, inputs = _chain_dag()
    replay = qc.replay_quantized_dag(dag, inputs)
    assert replay.max_abs_error_vs_float32_same_dag >= 0.0
    assert replay.relative_l2_vs_float32_same_dag >= 0.0
    assert replay.norm_drift_vs_float32_same_dag >= 0.0
    assert replay.max_abs_error_vs_complex128 >= 0.0
    assert replay.relative_l2_vs_complex128 >= 0.0
    assert replay.norm_drift_vs_complex128 >= 0.0
    assert replay.max_abs_error_float32_vs_complex128 >= 0.0
    assert replay.relative_l2_float32_vs_complex128 >= 0.0
    assert replay.norm_drift_float32_vs_complex128 >= 0.0


def test_static_and_logical_policy_facts_are_physical_claim_neutral() -> None:
    static = qc.numeric_policy_description()
    logical = qc.logical_policy_cost_facts(10, contraction_count=3, B=2, M=3, N=4, K=5)
    assert static["policy_id"] == qc.POLICY_ID
    assert static["bytes_per_complex_operand"] == 2
    assert static["scale_metadata_size"] == 8
    assert static["four_real_product_count"] == 4
    assert static["accumulator_requirement"] == "2*K*127^2"
    assert logical["logical_int8_operand_bytes"] == 2 * 10 + 8
    assert logical["logical_float32_operand_bytes"] == 8 * 10
    assert logical["quantization_event_count"] == 6
    assert logical["integer_multiply_accumulate_count"] == 4 * 2 * 3 * 4 * 5
    assert "measured_h2d_bytes" not in logical
    assert "mram_traffic_bytes" not in logical


def test_bell2_analytic_fixture_matches_both_references() -> None:
    dag, inputs = _dag_from_circuit("bell_2q")
    replay = qc.replay_quantized_dag(dag, inputs)
    np.testing.assert_array_equal(replay.float32_output, run_cpu_once(dag, inputs, HISTORICAL_FLOAT32).output)
    np.testing.assert_allclose(replay.complex128_output, run_cpu_once(dag, inputs, HISTORICAL_FLOAT32).output, atol=2e-7, rtol=2e-7)
    assert len(replay.traces) == len([node for node in dag.nodes if isinstance(node, ContractNode)])


@pytest.mark.parametrize(
    ("name", "params"),
    [
        ("quantization_stress", {"n_qubits": 4, "repeat_layers": 2}),
        ("ghz_chain", {"n_qubits": 18}),
        ("hs", {"allocated_qubits": 18, "depth": 1}),
        ("quantization_stress", {"n_qubits": 18, "repeat_layers": 2}),
    ],
    ids=("Stress4", "GHZ18", "HS18", "Stress18"),
)
def test_required_software_circuit_characterizations(
    name: str, params: dict
) -> None:
    dag, inputs = _dag_from_circuit(name, params)
    replay = qc.replay_quantized_dag(dag, inputs)
    assert len(replay.traces) == len([node for node in dag.nodes if isinstance(node, ContractNode)])
    assert replay.output.shape == dag.output.shape
    assert replay.output.dtype == np.dtype(np.complex64)
    assert np.all(np.isfinite(replay.output))
    assert np.all(np.isfinite(replay.float32_output))
    assert np.all(np.isfinite(replay.complex128_output))
    assert all(trace.int64_accumulator_safe for trace in replay.traces)
