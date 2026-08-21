import numpy as np
import pytest

from quantum_bench.circuits.library import builtin_circuit
from quantum_bench.tn.network import (
    build_tensor_network_data,
    validate_tensor_inputs,
)


def test_network_structure_and_values_are_separate():
    network, inputs = build_tensor_network_data(builtin_circuit("bell_2q"))

    assert tuple(inputs) == tuple(tensor.id for tensor in network.tensors)
    assert all(not hasattr(tensor, "array") for tensor in network.tensors)
    assert all(isinstance(array, np.ndarray) for array in inputs.values())


def test_tensor_input_validation_rejects_missing_and_wrong_shape():
    network, inputs = build_tensor_network_data(builtin_circuit("bell_2q"))

    missing = dict(inputs)
    missing.pop(next(iter(missing)))
    with pytest.raises(ValueError, match="missing"):
        validate_tensor_inputs(network, missing)

    first_id = next(iter(inputs))
    wrong = dict(inputs)
    wrong[first_id] = np.zeros((3,), dtype=np.complex128)
    with pytest.raises(ValueError, match="shape"):
        validate_tensor_inputs(network, wrong)


def test_tensor_input_validation_rejects_extra_ids():
    network, inputs = build_tensor_network_data(builtin_circuit("bell_2q"))
    extra = dict(inputs)
    extra["unexpected"] = np.zeros((1,), dtype=np.complex128)

    with pytest.raises(ValueError, match="extra"):
        validate_tensor_inputs(network, extra)


def test_tensor_input_validation_rejects_dtype_mismatch():
    network, inputs = build_tensor_network_data(builtin_circuit("bell_2q"))
    first_id = next(iter(inputs))
    wrong = dict(inputs)
    wrong[first_id] = np.zeros(inputs[first_id].shape, dtype=np.float32)

    with pytest.raises(ValueError, match="dtype"):
        validate_tensor_inputs(network, wrong)
