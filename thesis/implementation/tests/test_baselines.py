"""Focused tests for the direct tensor-network baselines."""

from __future__ import annotations

import ast
import inspect
import random
import re
from types import MappingProxyType

import numpy as np
import pytest

from quantum_bench import baselines
from quantum_bench.circuits import builtin_circuit
from quantum_bench.model import CircuitOperation, CircuitSpec, make_simulation_job
from quantum_bench.results import ExecutionFailed, ExecutionSample, UnsupportedExecution


def _job(name: str = "bell_2q"):
    return make_simulation_job(builtin_circuit(name))


def _complex_job():
    circuit = CircuitSpec(
        "complex_fixture",
        1,
        (CircuitOperation("h", (0,)), CircuitOperation("s", (0,))),
        {"kind": "fixture"},
    )
    return make_simulation_job(circuit)


def _order_distinguishing_complex_job():
    circuit = CircuitSpec(
        "complex_order_fixture",
        2,
        (
            CircuitOperation("h", (0,)),
            CircuitOperation("s", (0,)),
            CircuitOperation("h", (1,)),
            CircuitOperation("t", (1,)),
            CircuitOperation("cx", (0, 1)),
        ),
        {"kind": "fixture", "purpose": "axis-order regression"},
    )
    return make_simulation_job(circuit)


@pytest.mark.parametrize("runner", [baselines.run_quimb, baselines.run_cotengra])
def test_bell_has_canonical_statevector(runner):
    sample = runner(_job())
    expected = np.array([1, 0, 0, 1], dtype=np.complex128) / np.sqrt(2)
    np.testing.assert_allclose(sample.output, expected)


def test_nonzero_imaginary_amplitude_and_route_equality():
    job = _complex_job()
    quimb_sample = baselines.run_quimb(job)
    cotengra_sample = baselines.run_cotengra(job)
    expected = np.array([1, 1j], dtype=np.complex128) / np.sqrt(2)
    np.testing.assert_allclose(quimb_sample.output, expected)
    np.testing.assert_allclose(cotengra_sample.output, expected)
    np.testing.assert_allclose(quimb_sample.output, cotengra_sample.output)


@pytest.mark.parametrize("runner", [baselines.run_quimb, baselines.run_cotengra])
def test_multiqubit_complex_fixture_preserves_quest_basis_order(runner):
    sample = runner(_order_distinguishing_complex_job())
    phase = np.exp(1j * np.pi / 4)
    expected = np.array([1, 1j * phase, phase, 1j], dtype=np.complex128) / 2
    np.testing.assert_allclose(sample.output, expected)


def test_lowering_once_input_unchanged_and_output_deterministic(monkeypatch):
    job = _job()
    original_lower = baselines.lower_tensor_network
    calls = 0

    def counted_lower(value):
        nonlocal calls
        calls += 1
        return original_lower(value)

    monkeypatch.setattr(baselines, "lower_tensor_network", counted_lower)
    first = baselines.run_quimb(job)
    second = baselines.run_quimb(job)

    assert calls == 2
    assert job == _job()
    np.testing.assert_array_equal(first.output, second.output)
    assert first.backend_facts == second.backend_facts
    assert first.numeric_facts == second.numeric_facts


def test_timing_facts_and_result_are_immutable():
    sample = baselines.run_cotengra(_job(), methods="greedy", max_repeats=1)
    assert isinstance(sample, ExecutionSample)
    assert sample.output.dtype == np.dtype("complex128")
    assert not sample.output.flags.writeable
    assert sample.measurement.scope_id == "simulation_end_to_end_v1"
    assert sample.measurement.total_wall_s >= 0.0
    assert sample.measurement.lowering_s is not None
    assert sample.measurement.planning_s is not None
    assert sample.measurement.kernel_s is not None
    assert sample.measurement.decode_s is not None
    assert sample.backend_facts["backend_id"] == "cotengra_tn_v1"
    assert sample.backend_facts["methods"] == "greedy"
    assert sample.backend_facts["max_repeats"] == 1
    assert sample.backend_facts["hardware_execution"] is False
    assert sample.backend_facts["deterministic_planning_seed"] == 0
    assert sample.backend_facts["deterministic_planning_rngs"] == (
        "python_random",
        "numpy_legacy",
        "cotengra_hyperoptimizer",
    )
    assert "optimizer_seed" not in sample.backend_facts
    assert re.fullmatch(
        r"[0-9a-f]{64}", sample.backend_facts["contraction_path_fingerprint"]
    )
    assert sample.backend_facts["contraction_path_length"] > 0
    assert sample.numeric_facts["output_dtype"] == "complex128"
    assert isinstance(sample.backend_facts, MappingProxyType)
    with pytest.raises(ValueError):
        sample.output[0] = 0
    with pytest.raises(TypeError):
        sample.backend_facts["backend_id"] = "changed"


@pytest.mark.parametrize(
    "runner, kwargs",
    [
        (baselines.run_quimb, {"optimize": "invalid"}),
        (baselines.run_cotengra, {"methods": "invalid"}),
        (baselines.run_cotengra, {"max_repeats": 0}),
    ],
)
def test_invalid_configuration_is_unsupported(runner, kwargs):
    with pytest.raises(UnsupportedExecution) as error:
        runner(_job(), **kwargs)
    assert error.value.stage == "preflight"


def test_random_quimb_optimizer_is_unsupported():
    with pytest.raises(UnsupportedExecution) as error:
        baselines.run_quimb(_job(), optimize="random-greedy")
    assert error.value.stage == "preflight"


def test_auto_quimb_optimizer_is_unsupported():
    with pytest.raises(UnsupportedExecution) as error:
        baselines.run_quimb(_job(), optimize="auto")
    assert error.value.stage == "preflight"


def test_cotengra_invalid_prefix_is_unsupported():
    with pytest.raises(UnsupportedExecution) as error:
        baselines.run_cotengra(_job(), methods="greedy-not-a-method")
    assert error.value.stage == "preflight"


def test_cotengra_labels_is_an_accepted_deterministic_method():
    first = baselines.run_cotengra(_job("ghz_4q"), methods="labels")
    second = baselines.run_cotengra(_job("ghz_4q"), methods="labels")
    assert first.backend_facts["methods"] == "labels"
    assert (
        first.backend_facts["contraction_path_fingerprint"]
        == second.backend_facts["contraction_path_fingerprint"]
    )


@pytest.mark.parametrize("runner", [baselines.run_quimb, baselines.run_cotengra])
def test_contraction_path_provenance_is_sha256_and_stable(runner):
    first = runner(_job("ghz_4q"))
    second = runner(_job("ghz_4q"))

    first_fingerprint = first.backend_facts["contraction_path_fingerprint"]
    second_fingerprint = second.backend_facts["contraction_path_fingerprint"]
    assert isinstance(first_fingerprint, str)
    assert re.fullmatch(r"[0-9a-f]{64}", first_fingerprint)
    assert first_fingerprint == second_fingerprint
    assert first.backend_facts["contraction_path_length"] > 0
    assert (
        first.backend_facts["contraction_path_length"]
        == second.backend_facts["contraction_path_length"]
    )


def test_cotengra_path_is_deterministic_across_external_rng_states():
    original_python_state = random.getstate()
    original_numpy_state = np.random.get_state()
    try:
        random.seed(17)
        np.random.seed(19)
        first = baselines.run_cotengra(_job("ghz_4q"))

        random.seed(101)
        np.random.seed(103)
        second = baselines.run_cotengra(_job("ghz_4q"))
    finally:
        random.setstate(original_python_state)
        np.random.set_state(original_numpy_state)

    first_path = first.backend_facts["contraction_path_fingerprint"]
    second_path = second.backend_facts["contraction_path_fingerprint"]
    assert isinstance(first_path, str) and first_path
    assert first_path == second_path


def test_cotengra_restores_python_and_numpy_rng_states():
    original_python_state = random.getstate()
    original_numpy_state = np.random.get_state()
    try:
        random.seed(211)
        np.random.seed(223)
        expected_python_state = random.getstate()
        expected_numpy_state = np.random.get_state()

        baselines.run_cotengra(_job("ghz_4q"))

        assert random.getstate() == expected_python_state
        _assert_numpy_rng_state_equal(np.random.get_state(), expected_numpy_state)
    finally:
        random.setstate(original_python_state)
        np.random.set_state(original_numpy_state)


def test_nonfinite_decoded_output_is_a_decode_failure(monkeypatch):
    monkeypatch.setattr(
        baselines,
        "_tensor_to_quest_statevector",
        lambda tensor: np.array([np.nan + 0j], dtype=np.complex128),
    )
    with pytest.raises(ExecutionFailed) as error:
        baselines.run_quimb(_job())
    assert error.value.stage == "decode"


def test_unexpected_contraction_error_reports_kernel_stage(monkeypatch):
    original_contract = baselines._build_quimb_network

    def failing_network(qtn, network, inputs):
        tensor_network, output_inds = original_contract(qtn, network, inputs)

        def fail(*args, **kwargs):
            raise RuntimeError("fixture contraction failure")

        tensor_network.contract = fail
        return tensor_network, output_inds

    monkeypatch.setattr(baselines, "_build_quimb_network", failing_network)
    with pytest.raises(ExecutionFailed) as error:
        baselines.run_quimb(_job())
    assert error.value.stage == "planning"


def test_import_boundary_is_canonical_and_public_api_is_function_only():
    tree = ast.parse(inspect.getsource(baselines))
    forbidden = {
        "providers",
        "routing",
        "TaskGraph",
        "TensorNetworkValue",
        "ContractionDAG",
        "execution",
        "registry",
    }
    imports = [
        node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    ]
    assert all(not any(item in module for item in forbidden) for module in imports)
    assert set(baselines.__all__) == {"run_quimb", "run_cotengra"}
    assert all(callable(getattr(baselines, name)) for name in baselines.__all__)


def _assert_numpy_rng_state_equal(actual, expected):
    assert actual[0] == expected[0]
    np.testing.assert_array_equal(actual[1], expected[1])
    assert actual[2:] == expected[2:]
