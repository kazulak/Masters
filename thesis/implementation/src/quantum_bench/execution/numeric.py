"""Pure numeric policies for the functional CPU executor."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from quantum_bench.formats.fixed_point import FixedPointSpec, quantize_fixed_point

if TYPE_CHECKING:
    from quantum_bench.execution.contracts import NumericMode
    from quantum_bench.tn.graph import ContractNode


MAX_INT32_SAFE_K = (2**31 - 1) // (128 * 128)


def encode_tensor(
    array: np.ndarray, mode: "NumericMode"
) -> tuple[np.ndarray, float, int]:
    """Encode one operand for a numeric execution policy.

    The function is deliberately side-effect free from the caller's point of
    view: it returns a contiguous payload, the scale needed to decode an
    int8 result, and the number of clipped values.  The packed-int8 policy
    intentionally canonicalizes float64 and mathematically-real complex input
    through float32, matching the active CPU policy.  The fixed-point helper is
    used only as the established signed, symmetric, nearest-even conversion;
    its conversion timing is not part of this boundary.
    """

    from quantum_bench.execution.contracts import NumericMode

    value = np.asarray(array)
    if mode is NumericMode.COMPLEX128:
        if not np.all(np.isfinite(value)):
            raise ValueError("Numeric policy requires finite complex-valued tensors")
        return np.ascontiguousarray(value, dtype=np.complex128), 1.0, 0
    if mode is NumericMode.FLOAT32_REAL:
        real = _real_float32(value)
        return np.ascontiguousarray(real, dtype=np.float32), 1.0, 0
    if mode is NumericMode.HOST_PACKED_INT8_PER_TASK_V1:
        real = _real_float32(value)
        converted = quantize_fixed_point(real, FixedPointSpec(route_dtype="int8"))
        return (
            np.ascontiguousarray(converted.array, dtype=np.int8),
            float(converted.record.scale),
            int(converted.record.saturation_count),
        )
    raise ValueError(f"Unsupported numeric mode: {mode!r}")


def contract_encoded_node(
    node: "ContractNode",
    left: np.ndarray,
    right: np.ndarray,
    mode: "NumericMode",
) -> np.ndarray:
    """Contract already encoded operands and return the raw accumulator.

    Encoding is intentionally separate from this function.  Callers that
    need component timings encode with :func:`encode_tensor` first.  For
    ``HOST_PACKED_INT8_PER_TASK_V1`` the result is an int32 accumulator; the
    caller must decode it with the product of the operand scales.
    """

    from quantum_bench.execution.contracts import NumericMode

    if mode is NumericMode.HOST_PACKED_INT8_PER_TASK_V1:
        if np.asarray(left).dtype != np.dtype(np.int8) or np.asarray(
            right
        ).dtype != np.dtype(np.int8):
            raise ValueError(
                "HOST_PACKED_INT8 contraction requires int8 encoded operands"
            )
        contracted_k = math.prod(
            _label_dimension(node, label) for label in node.contracted_labels
        )
        if contracted_k > MAX_INT32_SAFE_K:
            raise ValueError(
                "HOST_PACKED_INT8 contraction exceeds int32 accumulation safety bound"
            )

    if mode is NumericMode.COMPLEX128:
        dtype = np.complex128
    elif mode is NumericMode.FLOAT32_REAL:
        dtype = np.float32
    elif mode is NumericMode.HOST_PACKED_INT8_PER_TASK_V1:
        dtype = np.int32
    else:
        raise ValueError(f"Unsupported numeric mode: {mode!r}")
    return _contract(
        left,
        node.left.labels,
        right,
        node.right.labels,
        node.output_labels,
        dtype=dtype,
    )


def decode_contraction_output(
    accumulator: np.ndarray,
    mode: "NumericMode",
    scale: float,
) -> np.ndarray:
    """Decode one raw contraction accumulator to the route output dtype."""

    from quantum_bench.execution.contracts import NumericMode

    if mode is NumericMode.COMPLEX128:
        return np.asarray(accumulator, dtype=np.complex128)
    if mode is NumericMode.FLOAT32_REAL:
        return np.asarray(accumulator, dtype=np.float32)
    if mode is NumericMode.HOST_PACKED_INT8_PER_TASK_V1:
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError("contraction output scale must be finite and positive")
        return np.asarray(accumulator, dtype=np.float32) * np.float32(scale)
    raise ValueError(f"Unsupported numeric mode: {mode!r}")


def contract_node(
    node: "ContractNode",
    left: np.ndarray,
    right: np.ndarray,
    mode: "NumericMode",
) -> np.ndarray:
    """Compatibility convenience for one untimed raw-input contraction."""

    left_payload, left_scale, _ = encode_tensor(left, mode)
    right_payload, right_scale, _ = encode_tensor(right, mode)
    accumulator = contract_encoded_node(node, left_payload, right_payload, mode)
    return decode_contraction_output(accumulator, mode, left_scale * right_scale)


def reduce_values(values: tuple[np.ndarray, ...]) -> np.ndarray:
    """Sum already-produced values without applying another numeric policy."""

    if not values:
        raise ValueError("Cannot reduce an empty value sequence")
    return np.sum(np.stack(values, axis=0), axis=0)


def _real_float32(array: np.ndarray) -> np.ndarray:
    value = np.asarray(array)
    if np.iscomplexobj(value) and np.any(np.imag(value) != 0):
        raise ValueError("Numeric policy requires real-valued tensors")
    if np.iscomplexobj(value) and not np.all(np.isfinite(value)):
        raise ValueError("Numeric policy requires finite complex-valued tensors")
    result = np.asarray(np.real(value), dtype=np.float32)
    if not np.all(np.isfinite(result)):
        raise ValueError("Numeric policy requires finite real-valued tensors")
    return result


def _label_dimension(node: "ContractNode", label: int) -> int:
    if label in node.left.labels:
        return node.left.shape[node.left.labels.index(label)]
    return node.right.shape[node.right.labels.index(label)]


def _contract(
    left: np.ndarray,
    left_labels: tuple[int, ...],
    right: np.ndarray,
    right_labels: tuple[int, ...],
    output_labels: tuple[int, ...],
    *,
    dtype: np.dtype,
) -> np.ndarray:
    compact = _compact_labels(left_labels, right_labels, output_labels)
    result = np.einsum(
        np.asarray(left, dtype=dtype),
        compact[0],
        np.asarray(right, dtype=dtype),
        compact[1],
        compact[2],
        optimize=False,
        dtype=dtype,
    )
    return np.asarray(result, dtype=dtype)


def _compact_labels(
    left: tuple[int, ...],
    right: tuple[int, ...],
    output: tuple[int, ...],
) -> tuple[list[int], list[int], list[int]]:
    labels: dict[int, int] = {}
    for label in left + right + output:
        if label not in labels:
            labels[label] = len(labels)
    return (
        [labels[label] for label in left],
        [labels[label] for label in right],
        [labels[label] for label in output],
    )


__all__ = [
    "MAX_INT32_SAFE_K",
    "contract_encoded_node",
    "contract_node",
    "decode_contraction_output",
    "encode_tensor",
    "reduce_values",
]
