from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from quantum_bench.model import ContractNode, TensorSpec, TensorView
from quantum_bench.numerics import (
    INT8_QUANTIZED_MAX_ABS,
    int32_accumulator_safe,
    theoretical_int32_accumulator_bound,
    EncodedComplexTensor,
    NumericPolicy,
    contract_complex_products,
    decode_complex_products,
    encode_complex_tensor,
)


FLOAT: NumericPolicy = "split_complex_float32_v1"
INT8: NumericPolicy = "complex_int8_shared_scale_v1"


def _matrix_node(
    *,
    left_labels: tuple[int, ...] = (41, 900),
    right_labels: tuple[int, ...] = (900, 73),
    left_shape: tuple[int, ...] = (2, 3),
    right_shape: tuple[int, ...] = (3, 2),
    contracted_labels: tuple[int, ...] = (900,),
    output_labels: tuple[int, ...] = (41, 73),
) -> ContractNode:
    return ContractNode(
        node_id="matrix",
        left=TensorView(tensor_id="left", labels=left_labels, shape=left_shape),
        right=TensorView(tensor_id="right", labels=right_labels, shape=right_shape),
        output=TensorSpec(
            id="out", labels=output_labels, shape=(left_shape[0], right_shape[1]), structure="dense"
        ),
        contracted_labels=contracted_labels,
        output_labels=output_labels,
    )


def _encode_pair(left: np.ndarray, right: np.ndarray, policy: NumericPolicy):
    return encode_complex_tensor(left, policy), encode_complex_tensor(right, policy)


def test_public_exports_are_exact() -> None:
    import quantum_bench.numerics as numerics

    assert numerics.__all__ == [
        "INT8_QUANTIZED_MAX_ABS",
        "INT8_MAX_PRODUCT",
        "INT8_COMPONENT_PRODUCT",
        "NumericPolicy",
        "EncodedComplexTensor",
        "encode_complex_tensor",
        "contract_complex_products",
        "decode_complex_products",
        "theoretical_int32_accumulator_bound",
        "int32_accumulator_safe",
    ]
    assert numerics.INT8_QUANTIZED_MAX_ABS == 127


@pytest.mark.parametrize("value", [np.array([1.0, np.nan]), np.array([np.inf + 0j])])
def test_encoding_rejects_nonfinite_values(value: np.ndarray) -> None:
    with pytest.raises(ValueError, match="finite"):
        encode_complex_tensor(value, FLOAT)


def test_float_encoding_preserves_split_complex_planes() -> None:
    value = np.array([1.0 + 2.0j, -2.0 - 1.0j], dtype=np.complex128)
    encoded = encode_complex_tensor(value, FLOAT)

    assert encoded.real.dtype == np.dtype(np.float32)
    assert encoded.imag.dtype == np.dtype(np.float32)
    assert encoded.scale == 1.0
    np.testing.assert_array_equal(encoded.real, [1.0, -2.0])
    np.testing.assert_array_equal(encoded.imag, [2.0, -1.0])


def test_float64_value_that_overflows_float32_is_rejected() -> None:
    with pytest.raises(ValueError, match="float32"):
        encode_complex_tensor(np.array([np.finfo(np.float64).max]), FLOAT)


def test_int8_uses_shared_scale_and_zero_fallback() -> None:
    encoded = encode_complex_tensor(np.array([1.0 + 4.0j, -2.0 - 1.0j]), INT8)

    assert encoded.scale == pytest.approx(4.0 / 127.0)
    assert encoded.real.dtype == np.dtype(np.int8)
    assert encoded.imag.dtype == np.dtype(np.int8)
    np.testing.assert_array_equal(encoded.real, [32, -64])
    np.testing.assert_array_equal(encoded.imag, [127, -32])

    zero = encode_complex_tensor(np.zeros(3, dtype=np.complex64), INT8)
    assert zero.scale == 1.0
    assert zero.saturation_real == zero.saturation_imag == 0
    np.testing.assert_array_equal(zero.real, 0)
    np.testing.assert_array_equal(zero.imag, 0)


def test_int8_endpoint_is_not_saturation_but_out_of_range_is() -> None:
    ordinary = encode_complex_tensor(np.array([4.0 + 0j]), INT8)
    assert ordinary.real[0] == INT8_QUANTIZED_MAX_ABS
    assert ordinary.saturation_real == 0


def test_int8_quantization_emits_only_within_bounded_magnitude() -> None:
    values = np.linspace(-1000.0, 1000.0, 500) + 1j * np.linspace(-500.0, 500.0, 500)
    encoded = encode_complex_tensor(values, INT8)
    assert np.all(encoded.real >= -INT8_QUANTIZED_MAX_ABS)
    assert np.all(encoded.real <= INT8_QUANTIZED_MAX_ABS)
    assert np.all(encoded.imag >= -INT8_QUANTIZED_MAX_ABS)
    assert np.all(encoded.imag <= INT8_QUANTIZED_MAX_ABS)
    assert int(np.min(encoded.real)) >= -127
    assert int(np.max(encoded.real)) <= 127
    assert int(np.min(encoded.imag)) >= -127
    assert int(np.max(encoded.imag)) <= 127
    assert -128 not in encoded.real
    assert -128 not in encoded.imag


def test_int8_rounding_is_nearest_even_and_deterministic() -> None:
    source = np.array([0.5 + 0j, 1.5 + 0j, -0.5 + 0j, -1.5 + 0j, 127.0 + 0j])
    first = encode_complex_tensor(source, INT8)
    second = encode_complex_tensor(source, INT8)

    np.testing.assert_array_equal(first.real, [0, 2, 0, -2, 127])
    assert first.saturation_real == 0
    assert first.saturation_imag == 0
    np.testing.assert_array_equal(first.real, second.real)
    assert first.scale == second.scale


def test_int8_scale_underflow_is_rejected_before_division() -> None:
    tiny = np.nextafter(np.float64(0.0), np.float64(1.0))
    with pytest.raises(ValueError, match="underflow"):
        encode_complex_tensor(np.array([tiny + 0j], dtype=np.complex128), INT8)


def test_encoded_planes_are_detached_and_readonly() -> None:
    source = np.array([1.0 + 2.0j, 3.0 + 4.0j])
    encoded = encode_complex_tensor(source, FLOAT)

    assert encoded.real.flags.owndata
    assert encoded.real.flags.c_contiguous
    assert not encoded.real.flags.writeable
    assert not encoded.imag.flags.writeable
    source[0] = 99.0
    assert encoded.real[0] == 1.0
    with pytest.raises(ValueError):
        encoded.real[0] = 0.0


def test_float_products_follow_labels_and_four_product_formula() -> None:
    node = _matrix_node()
    left = np.array([[1 + 2j, 3 - 1j, -2 + 1j], [4 + 0j, -1 + 2j, 2 - 3j]])
    right = np.array([[2 - 1j, 1 + 3j], [-2 + 2j, 3 + 0j], [1 - 1j, -1 + 2j]])
    left_encoded, right_encoded = _encode_pair(left, right, FLOAT)
    products = contract_complex_products(node, left_encoded, right_encoded, FLOAT)
    expected_products = (
        np.einsum("ik,kj->ij", left_encoded.real, right_encoded.real, dtype=np.float32, optimize=False),
        np.einsum("ik,kj->ij", left_encoded.imag, right_encoded.imag, dtype=np.float32, optimize=False),
        np.einsum("ik,kj->ij", left_encoded.real, right_encoded.imag, dtype=np.float32, optimize=False),
        np.einsum("ik,kj->ij", left_encoded.imag, right_encoded.real, dtype=np.float32, optimize=False),
    )

    for actual, expected in zip(products, expected_products):
        np.testing.assert_array_equal(actual, expected)
    np.testing.assert_allclose(
        decode_complex_products(products, 1.0, 1.0, FLOAT),
        (left @ right).astype(np.complex64),
        atol=0,
        rtol=0,
    )
    assert all(product.dtype == np.dtype(np.float32) for product in products)
    assert all(product.flags.owndata and not product.flags.writeable for product in products)


def test_int8_products_decode_and_repeat_deterministically() -> None:
    node = _matrix_node()
    left = np.array([[0.25 + 1j, -1.5 - 0.5j, 2.0 + 0j], [3.25 + 1j, 0.5 - 2j, -0.75 + 0.5j]])
    right = np.array([[1.0 - 1j, -2.0 + 0.25j], [0.25 + 0.5j, 0.5 - 1j], [-1.25 + 1j, 0.75 + 0j]])
    left_encoded, right_encoded = _encode_pair(left, right, INT8)
    products = contract_complex_products(node, left_encoded, right_encoded, INT8)
    repeated = contract_complex_products(node, *_encode_pair(left, right, INT8), INT8)

    assert all(product.dtype == np.dtype(np.int32) for product in products)
    for product, other in zip(products, repeated):
        np.testing.assert_array_equal(product, other)
    decoded = decode_complex_products(products, left_encoded.scale, right_encoded.scale, INT8)
    assert decoded.dtype == np.dtype(np.complex64)
    assert not decoded.flags.writeable
    np.testing.assert_allclose(decoded, left @ right, rtol=0.08, atol=0.08)


def test_decode_uses_explicit_rr_ii_ri_ir_order() -> None:
    products = (
        np.array([[10]], dtype=np.int32),
        np.array([[3]], dtype=np.int32),
        np.array([[4]], dtype=np.int32),
        np.array([[-2]], dtype=np.int32),
    )
    decoded = decode_complex_products(products, 0.5, 2.0, INT8)
    np.testing.assert_array_equal(decoded, np.array([[7.0 + 2.0j]], dtype=np.complex64))


def test_outer_product_preserves_requested_output_order() -> None:
    node = ContractNode(
        node_id="outer",
        left=TensorView(tensor_id="left", labels=(10,), shape=(2,)),
        right=TensorView(tensor_id="right", labels=(20,), shape=(3,)),
        output=TensorSpec(id="out", labels=(20, 10), shape=(3, 2), structure="dense"),
        contracted_labels=(),
        output_labels=(20, 10),
    )
    left = np.array([1.0 + 1j, 2.0 - 1j])
    right = np.array([3.0 + 0j, 4.0 + 1j, 5.0 - 2j])
    products = contract_complex_products(node, *_encode_pair(left, right, FLOAT), FLOAT)
    np.testing.assert_array_equal(
        decode_complex_products(products, 1.0, 1.0, FLOAT), right[:, None] * left[None, :]
    )


def test_contract_rejects_unsafe_int32_k_before_einsum() -> None:
    safe_k = np.iinfo(np.int32).max // (2 * INT8_QUANTIZED_MAX_ABS**2)

    def node_with_k(k: int) -> ContractNode:
        return _matrix_node(
            left_labels=(0, 1),
            right_labels=(1, 2),
            left_shape=(1, k),
            right_shape=(k, 1),
            contracted_labels=(1,),
            output_labels=(0, 2),
        )

    safe = node_with_k(safe_k)
    left_safe = EncodedComplexTensor(
        np.zeros((1, safe_k), dtype=np.int8), np.zeros((1, safe_k), dtype=np.int8), 1.0, 0, 0
    )
    right_safe = EncodedComplexTensor(
        np.zeros((safe_k, 1), dtype=np.int8), np.zeros((safe_k, 1), dtype=np.int8), 1.0, 0, 0
    )
    assert contract_complex_products(safe, left_safe, right_safe, INT8)[0].dtype == np.dtype(np.int32)

    unsafe = node_with_k(safe_k + 1)
    left = EncodedComplexTensor(
        np.zeros((1, safe_k + 1), dtype=np.int8), np.zeros((1, safe_k + 1), dtype=np.int8), 1.0, 0, 0
    )
    right = EncodedComplexTensor(
        np.zeros((safe_k + 1, 1), dtype=np.int8), np.zeros((safe_k + 1, 1), dtype=np.int8), 1.0, 0, 0
    )
    with pytest.raises(ValueError, match="int32 accumulation"):
        contract_complex_products(unsafe, left, right, INT8)


def test_full_component_int32_bound_is_explicit() -> None:
    safe_k = np.iinfo(np.int32).max // (2 * INT8_QUANTIZED_MAX_ABS**2)
    assert theoretical_int32_accumulator_bound(safe_k) <= np.iinfo(np.int32).max
    assert int32_accumulator_safe(safe_k)
    assert not int32_accumulator_safe(safe_k + 1)
    assert theoretical_int32_accumulator_bound(3) == 2 * 3 * 127**2


def test_malformed_encoded_values_and_descriptors_are_rejected() -> None:
    with pytest.raises(ValueError, match="share dtype"):
        EncodedComplexTensor(np.ones(2, dtype=np.float32), np.ones(2, dtype=np.int8), 1.0, 0, 0)
    with pytest.raises(ValueError, match="finite"):
        EncodedComplexTensor(np.array([np.inf], dtype=np.float32), np.zeros(1, dtype=np.float32), 1.0, 0, 0)
    with pytest.raises(ValueError, match="saturation"):
        EncodedComplexTensor(np.ones(2, dtype=np.int8), np.ones(2, dtype=np.int8), 1.0, True, 0)
    with pytest.raises(ValueError, match="scale exactly one"):
        EncodedComplexTensor(np.ones(2, dtype=np.float32), np.ones(2, dtype=np.float32), 2.0, 0, 0)

    node = _matrix_node()
    left, right = _encode_pair(np.ones((2, 3), dtype=np.complex64), np.ones((3, 2), dtype=np.complex64), FLOAT)
    bad_output = replace(node, output=TensorSpec(id="out", labels=(41, 73), shape=(3, 2), structure="dense"))
    with pytest.raises(ValueError, match="output descriptor shape"):
        contract_complex_products(bad_output, left, right, FLOAT)


def test_decode_rejects_bad_shapes_dtypes_scales_and_int64_overflow() -> None:
    products = tuple(np.ones((2, 2), dtype=np.float32) for _ in range(4))
    with pytest.raises(ValueError, match="exactly one"):
        decode_complex_products(products, 2.0, 1.0, FLOAT)
    with pytest.raises(ValueError, match="finite and positive"):
        decode_complex_products(products, 0.0, 1.0, FLOAT)
    with pytest.raises(ValueError, match="equal shapes"):
        decode_complex_products((*products[:3], np.ones((1, 1), dtype=np.float32)), 1.0, 1.0, FLOAT)
    with pytest.raises(ValueError, match="float32"):
        decode_complex_products(tuple(np.ones((2, 2), dtype=np.float64) for _ in range(4)), 1.0, 1.0, FLOAT)

    maximum = np.iinfo(np.int64).max
    overflowing = (
        np.zeros((1,), dtype=np.int64),
        np.zeros((1,), dtype=np.int64),
        np.array([maximum], dtype=np.int64),
        np.ones((1,), dtype=np.int64),
    )
    with pytest.raises(ValueError, match="overflow"):
        decode_complex_products(overflowing, 1.0, 1.0, INT8)


def test_unsupported_policy_is_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        encode_complex_tensor(np.ones(1), "legacy")  # type: ignore[arg-type]
