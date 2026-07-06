from __future__ import annotations

import time

import numpy as np
import opt_einsum as oe

from quantum_bench.core.indices import is_label_list_einsum_expression
from quantum_bench.core.records import ValidationResult
from quantum_bench.tn.network import TensorNetworkValue, interleaved_einsum_args


DEFAULT_TOLERANCES = {
    "max_abs_error": 1.0e-9,
    "l2_error": 1.0e-8,
    "max_rel_error": 1.0e-8,
    "norm_drift": 1.0e-8,
    "min_fidelity": 1.0 - 1.0e-9,
}


def compute_reference(network: TensorNetworkValue, optimize: str = "greedy") -> tuple[np.ndarray, float]:
    start = time.perf_counter()
    if is_label_list_einsum_expression(network.spec.einsum_expression):
        result = oe.contract(*interleaved_einsum_args(network), optimize=optimize)
    else:
        result = np.einsum(network.spec.einsum_expression, *(tensor.array for tensor in network.tensors), optimize=optimize)
    return np.asarray(result, dtype=np.complex128), time.perf_counter() - start


def validate(actual: np.ndarray, reference: np.ndarray, tolerances: dict | None = None) -> ValidationResult:
    tol = {**DEFAULT_TOLERANCES, **(tolerances or {})}
    actual = np.asarray(actual, dtype=np.complex128).reshape(reference.shape)
    reference = np.asarray(reference, dtype=np.complex128)
    diff = actual - reference
    abs_diff = np.abs(diff)
    ref_abs = np.abs(reference)
    abs_tol = float(tol["max_abs_error"])
    max_abs = float(abs_diff.max()) if abs_diff.size else 0.0
    l2 = float(np.linalg.norm(diff.ravel()))
    rel_mask = abs_diff > abs_tol
    denom = np.maximum(ref_abs[rel_mask], 1.0e-300)
    max_rel = float((abs_diff[rel_mask] / denom).max()) if np.any(rel_mask) else 0.0
    actual_norm = float(np.linalg.norm(actual.ravel()))
    reference_norm = float(np.linalg.norm(reference.ravel()))
    norm_drift = abs(actual_norm - reference_norm)
    if actual_norm == 0.0 and reference_norm == 0.0:
        fidelity = 1.0
    elif actual_norm == 0.0 or reference_norm == 0.0:
        fidelity = 0.0
    else:
        fidelity = float(abs(np.vdot(reference.ravel(), actual.ravel())) ** 2 / ((reference_norm**2) * (actual_norm**2)))
    passed = (
        max_abs <= abs_tol
        and l2 <= float(tol["l2_error"])
        and max_rel <= float(tol["max_rel_error"])
        and norm_drift <= float(tol["norm_drift"])
        and fidelity >= float(tol["min_fidelity"])
    )
    return ValidationResult(passed, max_abs, l2, max_rel, norm_drift, fidelity, reference_norm, actual_norm, tol)
