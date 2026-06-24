from __future__ import annotations


_SYMBOLS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"


def index_symbols(label_sets: list[tuple[int, ...]], output_labels: tuple[int, ...]) -> dict[int, str]:
    labels = sorted({label for labels in label_sets for label in labels} | set(output_labels))
    if len(labels) > len(_SYMBOLS):
        raise ValueError("Too many tensor indices for NumPy einsum symbol set")
    return {label: _SYMBOLS[idx] for idx, label in enumerate(labels)}


def shape_product(values: tuple[int, ...] | list[int]) -> int:
    product = 1
    for value in values:
        product *= int(value)
    return product
