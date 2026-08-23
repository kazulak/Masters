from __future__ import annotations

import numpy as np
import pytest

from quantum_bench.core.records import TensorSpec
from quantum_bench.model import ContractNode, TensorView
from quantum_bench.execution.contracts import NumericMode
from quantum_bench.execution.numeric import (
    MAX_INT32_SAFE_K,
    contract_encoded_node,
    contract_node,
    decode_contraction_output,
    encode_tensor,
)


def _matrix_node() -> ContractNode:
    return ContractNode(
        node_id="contract_0",
        left=TensorView(tensor_id="left", labels=(0, 1), shape=(2, 3)),
        right=TensorView(tensor_id="right", labels=(1, 2), shape=(3, 2)),
        output=TensorSpec(id="out", labels=(0, 2), shape=(2, 2), structure="dense"),
        contracted_labels=(1,),
        output_labels=(0, 2),
    )


def _node_with_k(k: int) -> ContractNode:
    return ContractNode(
        node_id="contract_k",
        left=TensorView(tensor_id="left", labels=(0, 1), shape=(1, k)),
        right=TensorView(tensor_id="right", labels=(1, 2), shape=(k, 1)),
        output=TensorSpec(id="out", labels=(0, 2), shape=(1, 1), structure="dense"),
        contracted_labels=(1,),
        output_labels=(0, 2),
    )


def test_host_int8_zero_tensor_uses_unit_scale_and_no_saturation() -> None:
    payload, scale, saturation = encode_tensor(
        np.zeros((2, 3), dtype=np.float64),
        NumericMode.HOST_PACKED_INT8_PER_TASK_V1,
    )

    assert payload.dtype == np.int8
    assert payload.flags.c_contiguous
    assert scale == 1.0
    assert saturation == 0
    np.testing.assert_array_equal(payload, 0)


def test_real_mode_accepts_zero_imaginary_complex_and_rejects_invalid_values() -> None:
    zero_imag = np.ones((2, 2), dtype=np.complex128)
    payload, scale, saturation = encode_tensor(zero_imag, NumericMode.FLOAT32_REAL)
    assert payload.dtype == np.float32
    assert scale == 1.0
    assert saturation == 0
    np.testing.assert_array_equal(payload, 1.0)
    float64_payload, float64_scale, _ = encode_tensor(
        zero_imag.real.astype(np.float64), NumericMode.HOST_PACKED_INT8_PER_TASK_V1
    )
    complex_payload, complex_scale, _ = encode_tensor(
        zero_imag, NumericMode.HOST_PACKED_INT8_PER_TASK_V1
    )
    float32_payload, float32_scale, _ = encode_tensor(
        zero_imag.real.astype(np.float32), NumericMode.HOST_PACKED_INT8_PER_TASK_V1
    )
    np.testing.assert_array_equal(float64_payload, float32_payload)
    np.testing.assert_array_equal(complex_payload, float32_payload)
    assert float64_scale == float32_scale
    assert complex_scale == float32_scale

    nonzero_imag = zero_imag.copy()
    nonzero_imag[0, 0] = 1.0 + 1.0j
    with pytest.raises(ValueError, match="real-valued"):
        encode_tensor(nonzero_imag, NumericMode.FLOAT32_REAL)

    nonfinite = np.array([[1.0, np.inf]], dtype=np.float32)
    with pytest.raises(ValueError, match="finite"):
        encode_tensor(nonfinite, NumericMode.FLOAT32_REAL)
    with pytest.raises(ValueError, match="finite"):
        encode_tensor(nonfinite, NumericMode.HOST_PACKED_INT8_PER_TASK_V1)

    with pytest.raises(ValueError, match="finite"):
        encode_tensor(
            np.array([1.0 + np.inf * 1j], dtype=np.complex128),
            NumericMode.COMPLEX128,
        )


def test_encoded_contract_returns_int32_and_decodes_with_operand_scales() -> None:
    node = _matrix_node()
    left = np.array([[0.25, -1.5, 2.0], [3.25, 0.5, -0.75]])
    right = np.array([[1.0, -2.0], [0.25, 0.5], [-1.25, 0.75]])
    left_payload, left_scale, _ = encode_tensor(
        left, NumericMode.HOST_PACKED_INT8_PER_TASK_V1
    )
    right_payload, right_scale, _ = encode_tensor(
        right, NumericMode.HOST_PACKED_INT8_PER_TASK_V1
    )

    accumulator = contract_encoded_node(
        node,
        left_payload,
        right_payload,
        NumericMode.HOST_PACKED_INT8_PER_TASK_V1,
    )
    output = decode_contraction_output(
        accumulator,
        NumericMode.HOST_PACKED_INT8_PER_TASK_V1,
        left_scale * right_scale,
    )

    assert accumulator.dtype == np.int32
    np.testing.assert_array_equal(
        output,
        contract_node(
            node,
            left,
            right,
            NumericMode.HOST_PACKED_INT8_PER_TASK_V1,
        ),
    )


def test_complex128_boundary_preserves_dtype() -> None:
    values = np.array([1.0 + 2.0j, -3.0j], dtype=np.complex64)
    payload, scale, saturation = encode_tensor(values, NumericMode.COMPLEX128)

    assert payload.dtype == np.complex128
    assert scale == 1.0
    assert saturation == 0
    np.testing.assert_array_equal(payload, values)


def test_int8_contract_requires_encoded_operands_and_safe_contracted_k() -> None:
    node = _matrix_node()
    with pytest.raises(ValueError, match="encoded operands"):
        contract_encoded_node(
            node,
            np.ones((2, 3), dtype=np.float32),
            np.ones((3, 2), dtype=np.int8),
            NumericMode.HOST_PACKED_INT8_PER_TASK_V1,
        )

    # The boundary check is intentionally exercised without executing this
    # large contraction; the next K must be rejected before einsum.
    unsafe_node = _node_with_k(MAX_INT32_SAFE_K + 1)
    with pytest.raises(ValueError, match="int32 accumulation"):
        contract_encoded_node(
            unsafe_node,
            np.ones((1, MAX_INT32_SAFE_K + 1), dtype=np.int8),
            np.ones((MAX_INT32_SAFE_K + 1, 1), dtype=np.int8),
            NumericMode.HOST_PACKED_INT8_PER_TASK_V1,
        )


def test_int8_bound_counts_one_sided_contracted_labels() -> None:
    node = ContractNode(
        node_id="one_sided",
        left=TensorView(tensor_id="left", labels=(0, 1), shape=(1, 1)),
        right=TensorView(
            tensor_id="right",
            labels=(1, 2),
            shape=(1, MAX_INT32_SAFE_K + 1),
        ),
        output=TensorSpec(id="out", labels=(0,), shape=(1,), structure="dense"),
        contracted_labels=(1, 2),
        output_labels=(0,),
    )
    with pytest.raises(ValueError, match="int32 accumulation"):
        contract_encoded_node(
            node,
            np.ones((1, 1), dtype=np.int8),
            np.ones((1, MAX_INT32_SAFE_K + 1), dtype=np.int8),
            NumericMode.HOST_PACKED_INT8_PER_TASK_V1,
        )


def test_decode_validates_scale_only_for_int8() -> None:
    values = np.array([1.0], dtype=np.float32)
    np.testing.assert_array_equal(
        decode_contraction_output(values, NumericMode.FLOAT32_REAL, 0.0), values
    )
    np.testing.assert_array_equal(
        decode_contraction_output(values, NumericMode.COMPLEX128, np.nan), values
    )
    with pytest.raises(ValueError, match="finite and positive"):
        decode_contraction_output(
            np.array([1], dtype=np.int32),
            NumericMode.HOST_PACKED_INT8_PER_TASK_V1,
            0.0,
        )
