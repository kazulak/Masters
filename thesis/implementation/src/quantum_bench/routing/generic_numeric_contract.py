"""Small, shared numeric contract for generic routes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


NumericKind = Literal["real", "complex_zero_imag", "complex_nonzero", "nonfinite"]


@dataclass(frozen=True)
class NumericClassification:
    kind: NumericKind
    is_complex: bool
    has_nonzero_imaginary: bool
    has_nonfinite: bool


def classify_numeric(value: object) -> NumericClassification:
    """Classify numeric values without changing or discarding components."""
    array = np.asarray(value)
    is_complex = bool(np.iscomplexobj(array))
    try:
        finite = bool(np.all(np.isfinite(array)))
    except TypeError:
        finite = False
    if not finite:
        return NumericClassification("nonfinite", is_complex, False, True)
    has_nonzero_imaginary = bool(is_complex and np.any(array.imag != 0))
    kind: NumericKind = "complex_nonzero" if has_nonzero_imaginary else (
        "complex_zero_imag" if is_complex else "real"
    )
    return NumericClassification(kind, is_complex, has_nonzero_imaginary, False)
