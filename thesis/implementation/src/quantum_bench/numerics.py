"""Small, target-neutral complex numeric policies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from quantum_bench.model import ContractNode


NumericPolicy = Literal[
    "split_complex_float32_v1",
    "split_complex_int8_shared_scale_v1",
]


@dataclass(frozen=True, slots=True)
class EncodedComplexTensor:
    real: np.ndarray
    imag: np.ndarray
    scale: float
    saturation_real: int
    saturation_imag: int

    def __post_init__(self) -> None:
        real = _readonly_copy(self.real)
        imag = _readonly_copy(self.imag)
        if real.shape != imag.shape:
            raise ValueError("real and imaginary planes must have the same shape")
        if real.dtype != imag.dtype or real.dtype not in (np.dtype(np.float32), np.dtype(np.int8)):
            raise ValueError("encoded planes must share dtype float32 or int8")
        if real.dtype == np.dtype(np.float32) and (
            not np.all(np.isfinite(real)) or not np.all(np.isfinite(imag))
        ):
            raise ValueError("encoded float32 planes must be finite")
        try:
            scale = float(self.scale)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("encoded tensor scale must be finite and positive") from exc
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError("encoded tensor scale must be finite and positive")
        if real.dtype == np.dtype(np.float32) and scale != 1.0:
            raise ValueError("float32 encoded planes require scale exactly one")
        if not _valid_saturation_count(self.saturation_real, real.size) or not _valid_saturation_count(
            self.saturation_imag, imag.size
        ):
            raise ValueError("saturation counts must be integers within plane size")
        object.__setattr__(self, "real", real)
        object.__setattr__(self, "imag", imag)
        object.__setattr__(self, "scale", scale)
        object.__setattr__(self, "saturation_real", int(self.saturation_real))
        object.__setattr__(self, "saturation_imag", int(self.saturation_imag))


def encode_complex_tensor(
    value: np.ndarray,
    policy: NumericPolicy,
) -> EncodedComplexTensor:
    """Encode one complex tensor into separate real and imaginary planes."""

    _validate_policy(policy)
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError("numeric tensor input is required")
    if not np.all(np.isfinite(array)):
        raise ValueError("numeric policy requires finite tensor values")

    if policy == "split_complex_float32_v1":
        with np.errstate(over="ignore", invalid="ignore"):
            if np.iscomplexobj(array):
                real = np.asarray(array.real, dtype=np.float32)
                imag = np.asarray(array.imag, dtype=np.float32)
            else:
                real = np.asarray(array, dtype=np.float32)
                imag = np.zeros_like(real, dtype=np.float32)
        if not np.all(np.isfinite(real)) or not np.all(np.isfinite(imag)):
            raise ValueError("finite input is not representable as float32")
        return EncodedComplexTensor(
            real=real,
            imag=imag,
            scale=1.0,
            saturation_real=0,
            saturation_imag=0,
        )

    if np.iscomplexobj(array):
        real = np.asarray(array.real, dtype=np.float64)
        imag = np.asarray(array.imag, dtype=np.float64)
    else:
        real = np.asarray(array, dtype=np.float64)
        imag = np.zeros_like(real)
    if not np.all(np.isfinite(real)) or not np.all(np.isfinite(imag)):
        raise ValueError("int8 quantization requires finite float64 planes")
    max_abs = max(float(np.max(np.abs(real), initial=0.0)), float(np.max(np.abs(imag), initial=0.0)))
    scale = max_abs / 127.0 if max_abs > 0.0 else 1.0
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("int8 quantization scale underflowed or is nonfinite")
    quantized_real, saturation_real = _quantize_plane(real, scale)
    quantized_imag, saturation_imag = _quantize_plane(imag, scale)
    return EncodedComplexTensor(
        real=quantized_real,
        imag=quantized_imag,
        scale=scale,
        saturation_real=saturation_real,
        saturation_imag=saturation_imag,
    )


def contract_complex_products(
    node: ContractNode,
    left: EncodedComplexTensor,
    right: EncodedComplexTensor,
    policy: NumericPolicy,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return the four real products ``rr, ii, ri, ir``."""

    _validate_policy(policy)
    _validate_operand(node.left.shape, node.left.labels, left, policy, "left")
    _validate_operand(node.right.shape, node.right.labels, right, policy, "right")
    _validate_contraction_labels(node)

    contracted_k = 1
    for label in node.contracted_labels:
        if label in node.left.labels:
            contracted_k *= node.left.shape[node.left.labels.index(label)]
        else:
            contracted_k *= node.right.shape[node.right.labels.index(label)]
    if policy == "split_complex_int8_shared_scale_v1":
        if contracted_k * (127**2) > np.iinfo(np.int32).max:
            raise ValueError("int8 contraction exceeds int32 accumulation safety bound")
        dtype = np.dtype(np.int32)
    else:
        dtype = np.dtype(np.float32)

    left_indices, right_indices, output_indices = _einsum_indices(node)
    products = []
    for left_plane, right_plane in (
        (left.real, right.real),
        (left.imag, right.imag),
        (left.real, right.imag),
        (left.imag, right.real),
    ):
        with np.errstate(over="ignore", invalid="ignore"):
            product = np.einsum(
                left_plane,
                left_indices,
                right_plane,
                right_indices,
                output_indices,
                dtype=dtype,
                optimize=False,
            )
        if not np.all(np.isfinite(product)):
            raise ValueError("complex product produced a nonfinite result")
        products.append(_readonly_copy(product))
    return tuple(products)  # type: ignore[return-value]


def decode_complex_products(
    products: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    left_scale: float,
    right_scale: float,
    policy: NumericPolicy,
) -> np.ndarray:
    """Combine four products and decode them to a read-only complex64 array."""

    _validate_policy(policy)
    if len(products) != 4:
        raise ValueError("four complex product arrays are required")
    arrays = tuple(np.asarray(product) for product in products)
    if any(array.shape != arrays[0].shape for array in arrays[1:]):
        raise ValueError("complex product arrays must have equal shapes")
    if not np.isfinite(left_scale) or not np.isfinite(right_scale):
        raise ValueError("product scales must be finite and positive")
    if left_scale <= 0.0 or right_scale <= 0.0:
        raise ValueError("product scales must be finite and positive")

    if policy == "split_complex_float32_v1":
        if left_scale != 1.0 or right_scale != 1.0:
            raise ValueError("float32 product scales must be exactly one")
        if any(array.dtype != np.dtype(np.float32) for array in arrays):
            raise ValueError("float32 products must have dtype float32")
        if any(not np.all(np.isfinite(array)) for array in arrays):
            raise ValueError("float32 products must be finite")
        with np.errstate(over="ignore", invalid="ignore"):
            real = np.subtract(arrays[0], arrays[1], dtype=np.float32)
            imag = np.add(arrays[2], arrays[3], dtype=np.float32)
        return _complex64_result(real, imag)

    product_dtype = arrays[0].dtype
    if product_dtype not in (np.dtype(np.int32), np.dtype(np.int64)) or any(
        array.dtype != product_dtype for array in arrays
    ):
        raise ValueError("int8 products must be homogeneous int32 or int64")
    scale = float(left_scale) * float(right_scale)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("product scale must be finite and positive")
    rr, ii, ri, ir = (array.astype(np.int64, copy=False) for array in arrays)
    if product_dtype == np.dtype(np.int64):
        _check_int64_subtract(rr, ii)
        _check_int64_add(ri, ir)
    with np.errstate(over="ignore", invalid="ignore"):
        real = rr - ii
        imag = ri + ir
    return _complex64_result(real.astype(np.float64) * scale, imag.astype(np.float64) * scale)


def _validate_policy(policy: str) -> None:
    if policy not in {
        "split_complex_float32_v1",
        "split_complex_int8_shared_scale_v1",
    }:
        raise ValueError(f"unsupported numeric policy: {policy!r}")


def _readonly_copy(value: np.ndarray) -> np.ndarray:
    copied = np.array(value, copy=True, order="C")
    copied.setflags(write=False)
    return copied


def _valid_saturation_count(value: object, plane_size: int) -> bool:
    return (
        not isinstance(value, (bool, np.bool_))
        and isinstance(value, (int, np.integer))
        and 0 <= int(value) <= plane_size
    )


def _quantize_plane(value: np.ndarray, scale: float) -> tuple[np.ndarray, int]:
    rounded = np.rint(value / scale)
    saturation = int(np.count_nonzero((rounded < -127) | (rounded > 127)))
    clipped = np.clip(rounded, -127, 127).astype(np.int8)
    return clipped, saturation


def _validate_operand(
    shape: tuple[int, ...],
    labels: tuple[int, ...],
    operand: EncodedComplexTensor,
    policy: NumericPolicy,
    name: str,
) -> None:
    if operand.real.shape != shape or operand.imag.shape != shape:
        raise ValueError(f"{name} encoded planes do not match node shape")
    expected = (
        np.dtype(np.float32)
        if policy == "split_complex_float32_v1"
        else np.dtype(np.int8)
    )
    if operand.real.dtype != expected or operand.imag.dtype != expected:
        raise ValueError(f"{name} encoded planes have the wrong dtype")
    if len(labels) != len(shape) or len(set(labels)) != len(labels):
        raise ValueError(f"{name} labels must be unique and match its rank")


def _validate_contraction_labels(node: ContractNode) -> None:
    for labels in (
        node.left.labels,
        node.right.labels,
        node.output.labels,
        node.contracted_labels,
        node.output_labels,
    ):
        if any(isinstance(label, (bool, np.bool_)) or not isinstance(label, (int, np.integer)) for label in labels):
            raise ValueError("contraction labels must be integers")
    if len(node.left.labels) != len(node.left.shape) or len(node.right.labels) != len(node.right.shape):
        raise ValueError("node labels must match operand ranks")
    if len(node.output.labels) != len(node.output.shape):
        raise ValueError("output labels must match output rank")
    if tuple(node.output.labels) != tuple(node.output_labels):
        raise ValueError("node output descriptor labels must match output_labels")
    if len(set(node.contracted_labels)) != len(node.contracted_labels):
        raise ValueError("contracted labels must be unique")
    if len(set(node.output_labels)) != len(node.output_labels):
        raise ValueError("output labels must be unique")
    left_labels = set(node.left.labels)
    right_labels = set(node.right.labels)
    input_labels = left_labels | right_labels
    contracted_labels = set(node.contracted_labels)
    output_labels = set(node.output_labels)
    if contracted_labels & output_labels:
        raise ValueError("contracted and output labels must be disjoint")
    if contracted_labels | output_labels != input_labels:
        raise ValueError("contracted and output labels must cover all input labels")
    for label in node.contracted_labels:
        if label not in input_labels:
            raise ValueError("every contracted label must occur in an operand")
    for label in left_labels & right_labels:
        left_dim = node.left.shape[node.left.labels.index(label)]
        right_dim = node.right.shape[node.right.labels.index(label)]
        if left_dim != right_dim:
            raise ValueError("shared label dimensions must agree")
    dimensions = {
        label: node.left.shape[axis]
        for axis, label in enumerate(node.left.labels)
    }
    dimensions.update(
        {
            label: node.right.shape[axis]
            for axis, label in enumerate(node.right.labels)
            if label not in dimensions
        }
    )
    expected_output_shape = tuple(dimensions[label] for label in node.output_labels)
    if tuple(node.output.shape) != expected_output_shape:
        raise ValueError("node output descriptor shape does not match labels")


def _einsum_indices(node: ContractNode) -> tuple[list[int], list[int], list[int]]:
    labels = list(dict.fromkeys((*node.left.labels, *node.right.labels, *node.output_labels)))
    if len(labels) > 52:
        raise ValueError("contraction uses too many distinct labels for NumPy einsum")
    mapping = {label: index for index, label in enumerate(labels)}
    return (
        [mapping[label] for label in node.left.labels],
        [mapping[label] for label in node.right.labels],
        [mapping[label] for label in node.output_labels],
    )


def _check_int64_add(left: np.ndarray, right: np.ndarray) -> None:
    info = np.iinfo(np.int64)
    right_positive = np.maximum(right, 0)
    right_negative = np.minimum(right, 0)
    positive_overflow = (right > 0) & (left > info.max - right_positive)
    negative_overflow = (right < 0) & (left < info.min - right_negative)
    if np.any(positive_overflow | negative_overflow):
        raise ValueError("complex int64 combination would overflow")


def _check_int64_subtract(left: np.ndarray, right: np.ndarray) -> None:
    info = np.iinfo(np.int64)
    right_positive = np.maximum(right, 0)
    right_negative = np.minimum(right, 0)
    positive_overflow = (right < 0) & (left > info.max + right_negative)
    negative_overflow = (right > 0) & (left < info.min + right_positive)
    if np.any(positive_overflow | negative_overflow):
        raise ValueError("complex int64 combination would overflow")


def _complex64_result(real: np.ndarray, imag: np.ndarray) -> np.ndarray:
    output = np.empty(real.shape, dtype=np.complex64, order="C")
    with np.errstate(over="ignore", invalid="ignore"):
        output.real = np.asarray(real, dtype=np.float32)
        output.imag = np.asarray(imag, dtype=np.float32)
    if not np.all(np.isfinite(output)):
        raise ValueError("complex64 result is nonfinite")
    output.setflags(write=False)
    return output


__all__ = [
    "NumericPolicy",
    "EncodedComplexTensor",
    "encode_complex_tensor",
    "contract_complex_products",
    "decode_complex_products",
]
