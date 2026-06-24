from __future__ import annotations

import numpy as np
import opt_einsum as oe

from .model import TensorValue


def density(array: np.ndarray) -> float:
    return 0.0 if array.size == 0 else float(np.count_nonzero(array) / array.size)


def index_symbols(label_groups: list[tuple[int, ...]], output_labels: tuple[int, ...]) -> dict[int, str]:
    all_labels = sorted(set(label for group in label_groups for label in group) | set(output_labels))
    return {label: oe.get_symbol(index) for index, label in enumerate(all_labels)}


def label_dim(label: int, left: TensorValue, right: TensorValue) -> int:
    for tensor in (left, right):
        if label in tensor.labels:
            return int(tensor.array.shape[tensor.labels.index(label)])
    raise ValueError(f"Label {label} not found in operands")


def shape_product(values) -> int:
    result = 1
    for value in values:
        result *= int(value)
    return result

