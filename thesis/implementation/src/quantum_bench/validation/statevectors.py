from __future__ import annotations

from typing import Any

import numpy as np

from quantum_bench.core.records import JsonDict


STATEVECTOR_METRICS_SCHEMA_VERSION = "statevector_metrics_v1"
QUEST_BASIS_ORDER = "quest_little_endian_integer_index"


def tensor_to_quest_statevector(tensor: np.ndarray) -> np.ndarray:
    """Flatten a wire-ordered TN final tensor into QuEST basis-index order."""
    array = np.asarray(tensor, dtype=np.complex128)
    if array.ndim == 0:
        return array.reshape(1)
    if any(dim != 2 for dim in array.shape):
        raise ValueError(f"statevector conversion requires qubit dimensions of 2, got shape {array.shape}")
    output = np.empty(1 << array.ndim, dtype=np.complex128)
    for index in range(output.size):
        bits = tuple((index >> wire) & 1 for wire in range(array.ndim))
        output[index] = array[bits]
    return output


def probability_distribution(statevector: np.ndarray) -> np.ndarray:
    state = np.asarray(statevector, dtype=np.complex128).ravel()
    return np.abs(state) ** 2


def probability_error_metrics(actual: np.ndarray, expected: np.ndarray) -> JsonDict:
    actual_prob = probability_distribution(actual)
    expected_prob = probability_distribution(expected)
    diff = actual_prob - expected_prob
    abs_diff = np.abs(diff)
    return {
        "schema_version": STATEVECTOR_METRICS_SCHEMA_VERSION,
        "probability_l1_error": float(np.sum(abs_diff)),
        "probability_max_abs_error": float(abs_diff.max()) if abs_diff.size else 0.0,
        "actual_probability_norm": float(np.sum(actual_prob)),
        "expected_probability_norm": float(np.sum(expected_prob)),
    }


def statevector_memory_metadata(statevector: np.ndarray) -> JsonDict:
    array = np.asarray(statevector)
    return {
        "shape": tuple(int(dim) for dim in array.shape),
        "dtype": str(array.dtype),
        "size": int(array.size),
        "nbytes": int(array.nbytes),
    }


def validation_result_to_dict(result: Any) -> JsonDict:
    return {
        "passed": bool(result.passed),
        "max_abs_error": float(result.max_abs_error),
        "l2_error": float(result.l2_error),
        "max_rel_error": float(result.max_rel_error),
        "norm_drift": float(result.norm_drift),
        "fidelity": float(result.fidelity),
        "reference_norm": float(result.reference_norm),
        "actual_norm": float(result.actual_norm),
        "tolerance": dict(result.tolerance),
    }
