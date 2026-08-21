from dataclasses import replace

import numpy as np
import pytest

from quantum_bench.circuits.library import builtin_circuit
from quantum_bench.tn.network import (
    TensorInput,
    TensorInputs,
    build_tensor_network_data,
    tensor_input_map,
    validate_tensor_inputs,
)


def test_network_structure_and_values_are_separate():
    network, inputs = build_tensor_network_data(builtin_circuit("bell_2q"))

    assert tuple(value.tensor_id for value in inputs.values) == tuple(
        tensor.id for tensor in network.tensors
    )
    assert all(not hasattr(tensor, "array") for tensor in network.tensors)
    assert set(tensor_input_map(inputs)) == {tensor.id for tensor in network.tensors}


def test_tensor_input_validation_rejects_missing_and_wrong_shape():
    network, inputs = build_tensor_network_data(builtin_circuit("bell_2q"))

    with pytest.raises(ValueError, match="missing"):
        validate_tensor_inputs(network, TensorInputs(values=inputs.values[:-1]))

    first = inputs.values[0]
    wrong = replace(first, array=np.zeros((3,), dtype=np.complex128))
    with pytest.raises(ValueError, match="shape"):
        validate_tensor_inputs(
            network,
            TensorInputs(values=(wrong, *inputs.values[1:])),
        )


def test_tensor_input_validation_rejects_duplicate_ids():
    network, inputs = build_tensor_network_data(builtin_circuit("bell_2q"))
    duplicate = TensorInput(
        tensor_id=inputs.values[0].tensor_id,
        array=inputs.values[0].array,
    )

    with pytest.raises(ValueError, match="duplicate"):
        validate_tensor_inputs(
            network,
            TensorInputs(values=(*inputs.values, duplicate)),
        )


def test_tensor_input_validation_rejects_dtype_mismatch():
    network, inputs = build_tensor_network_data(builtin_circuit("bell_2q"))
    wrong = replace(
        inputs.values[0],
        array=np.zeros(inputs.values[0].array.shape, dtype=np.float32),
    )

    with pytest.raises(ValueError, match="dtype"):
        validate_tensor_inputs(
            network,
            TensorInputs(values=(wrong, *inputs.values[1:])),
        )
