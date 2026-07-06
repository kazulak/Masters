from __future__ import annotations

import numpy as np

from quantum_bench.core.indices import is_label_list_einsum_expression
from quantum_bench.core.records import ContractionTask


def contract_binary_task(
    task: ContractionTask,
    left: np.ndarray,
    right: np.ndarray,
    *,
    dtype: np.dtype | type = np.complex128,
) -> np.ndarray:
    left_array = np.asarray(left, dtype=dtype)
    right_array = np.asarray(right, dtype=dtype)
    if is_label_list_einsum_expression(task.index_expression):
        result = _contract_binary_label_lists(task, left_array, right_array)
    else:
        result = np.einsum(task.index_expression, left_array, right_array, optimize=False)
    return np.asarray(result, dtype=dtype)


def _contract_binary_label_lists(task: ContractionTask, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    output_set = set(task.output_labels)
    contracted = tuple(label for label in task.left_labels if label in set(task.right_labels) and label not in output_set)
    left_only_reduced = tuple(label for label in task.left_labels if label not in output_set and label not in contracted)
    right_only_reduced = tuple(label for label in task.right_labels if label not in output_set and label not in contracted)
    left, left_labels = _sum_out_labels(left, task.left_labels, left_only_reduced)
    right, right_labels = _sum_out_labels(right, task.right_labels, right_only_reduced)
    free_left = tuple(label for label in left_labels if label not in contracted)
    free_right = tuple(label for label in right_labels if label not in contracted)
    left_order = free_left + contracted
    right_order = contracted + free_right

    for label in contracted:
        if label not in left_labels or label not in right_labels:
            raise ValueError(f"contracted_label_missing:{label}")
        left_dim = left.shape[left_labels.index(label)]
        right_dim = right.shape[right_labels.index(label)]
        if left_dim != right_dim:
            raise ValueError(f"contracted_label_dimension_mismatch:{label}")

    left_reordered = _transpose_to_labels(left, left_labels, left_order)
    right_reordered = _transpose_to_labels(right, right_labels, right_order)
    m = _shape_product(left_reordered.shape[: len(free_left)])
    k_left = _shape_product(left_reordered.shape[len(free_left) :])
    k_right = _shape_product(right_reordered.shape[: len(contracted)])
    n = _shape_product(right_reordered.shape[len(contracted) :])
    if k_left != k_right:
        raise ValueError("contracted_dimension_product_mismatch")

    matrix = left_reordered.reshape((m, k_left)) @ right_reordered.reshape((k_right, n))
    matrix_labels = free_left + free_right
    matrix_shape = tuple(_label_dim(label, task, left, right) for label in matrix_labels)
    output = matrix.reshape(matrix_shape)
    if matrix_labels != task.output_labels:
        output = _transpose_to_labels(output, matrix_labels, task.output_labels)
    if tuple(int(dim) for dim in output.shape) != task.output_shape:
        raise ValueError(f"binary_contraction_output_shape_mismatch:{output.shape}!={task.output_shape}")
    return output


def _transpose_to_labels(array: np.ndarray, current_labels: tuple[int, ...], target_labels: tuple[int, ...]) -> np.ndarray:
    if current_labels == target_labels:
        return array
    if set(current_labels) != set(target_labels):
        raise ValueError("transpose_label_set_mismatch")
    axes = tuple(current_labels.index(label) for label in target_labels)
    return np.transpose(array, axes)


def _sum_out_labels(array: np.ndarray, labels: tuple[int, ...], reduced: tuple[int, ...]) -> tuple[np.ndarray, tuple[int, ...]]:
    if not reduced:
        return array, labels
    axes = tuple(labels.index(label) for label in reduced)
    result = np.sum(array, axis=axes)
    remaining = tuple(label for label in labels if label not in set(reduced))
    return np.asarray(result), remaining


def _label_dim(label: int, task: ContractionTask, left: np.ndarray, right: np.ndarray) -> int:
    if label in task.left_labels:
        return int(left.shape[task.left_labels.index(label)])
    if label in task.right_labels:
        return int(right.shape[task.right_labels.index(label)])
    raise ValueError(f"output_label_missing_from_inputs:{label}")


def _shape_product(shape: tuple[int, ...]) -> int:
    product = 1
    for dim in shape:
        product *= int(dim)
    return product
