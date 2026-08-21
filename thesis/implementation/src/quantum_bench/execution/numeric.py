"""Pure numeric policies for the functional CPU executor."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from quantum_bench.formats.fixed_point import FixedPointSpec, quantize_fixed_point

if TYPE_CHECKING:
    from quantum_bench.execution.contracts import NumericMode
    from quantum_bench.tn.graph import ContractNode


def contract_node(
    node: "ContractNode",
    left: np.ndarray,
    right: np.ndarray,
    mode: "NumericMode",
) -> np.ndarray:
    """Contract one graph node using the selected numeric representation."""

    from quantum_bench.execution.contracts import NumericMode

    if mode is NumericMode.COMPLEX128:
        return _contract(
            left,
            node.left.labels,
            right,
            node.right.labels,
            node.output_labels,
            dtype=np.complex128,
        )
    if mode is NumericMode.FLOAT32_REAL:
        return _contract_real_float32(node, left, right)
    if mode is NumericMode.HOST_PACKED_INT8_PER_TASK_V1:
        return _contract_host_packed_int8(node, left, right)
    raise ValueError(f"Unsupported CPU numeric mode: {mode!r}")


def reduce_values(values: tuple[np.ndarray, ...]) -> np.ndarray:
    """Sum already-produced values without applying another numeric policy."""

    if not values:
        raise ValueError("Cannot reduce an empty value sequence")
    return np.sum(np.stack(values, axis=0), axis=0)


def _contract_real_float32(
    node: "ContractNode", left: np.ndarray, right: np.ndarray
) -> np.ndarray:
    left_value = _real_float32(left)
    right_value = _real_float32(right)
    return _contract(
        left_value,
        node.left.labels,
        right_value,
        node.right.labels,
        node.output_labels,
        dtype=np.float32,
    )


def _contract_host_packed_int8(
    node: "ContractNode", left: np.ndarray, right: np.ndarray
) -> np.ndarray:
    left_q = quantize_fixed_point(
        _real_float32(left), FixedPointSpec(route_dtype="int8")
    )
    right_q = quantize_fixed_point(
        _real_float32(right), FixedPointSpec(route_dtype="int8")
    )
    integer_output = _contract(
        left_q.array,
        node.left.labels,
        right_q.array,
        node.right.labels,
        node.output_labels,
        dtype=np.int32,
    )
    scale = np.float32(left_q.record.scale * right_q.record.scale)
    return np.asarray(integer_output, dtype=np.float32) * scale


def _real_float32(array: np.ndarray) -> np.ndarray:
    value = np.asarray(array)
    if np.iscomplexobj(value) and np.any(np.imag(value) != 0):
        raise ValueError("Numeric policy requires real-valued tensors")
    return np.asarray(np.real(value), dtype=np.float32)


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


__all__ = ["contract_node", "reduce_values"]
