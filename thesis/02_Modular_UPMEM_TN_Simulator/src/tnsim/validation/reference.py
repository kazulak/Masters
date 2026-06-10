from __future__ import annotations

import time

import numpy as np

from tnsim.core.model import TensorNetwork


def compute_reference(network: TensorNetwork, optimize: str) -> tuple[np.ndarray, float]:
    start = time.perf_counter()
    result = np.einsum(network.einsum_expression, *(tensor.array for tensor in network.tensors), optimize=optimize)
    return result, time.perf_counter() - start


def validation_record(actual: np.ndarray, reference: np.ndarray, config: dict, validation_seconds: float) -> dict:
    actual = np.asarray(actual, dtype=np.complex128).reshape(reference.shape)
    diff = actual - reference
    abs_diff = np.abs(diff)
    ref_abs = np.abs(reference)
    max_abs = float(abs_diff.max()) if abs_diff.size else 0.0
    denom = np.maximum(ref_abs, 1.0e-300)
    max_rel = float((abs_diff / denom).max()) if abs_diff.size else 0.0
    actual_norm = float(np.linalg.norm(actual.ravel()))
    reference_norm = float(np.linalg.norm(reference.ravel()))
    norm_drift = abs(actual_norm - reference_norm)
    if actual_norm == 0.0 and reference_norm == 0.0:
        fidelity = 1.0
    elif actual_norm == 0.0 or reference_norm == 0.0:
        fidelity = 0.0
    else:
        fidelity = float(
            abs(np.vdot(reference.ravel(), actual.ravel())) ** 2
            / ((reference_norm**2) * (actual_norm**2))
        )
    tolerances = config["validation"]["tolerances"]
    passed = (
        max_abs <= float(tolerances["max_abs_error"])
        and max_rel <= float(tolerances["max_rel_error"])
        and norm_drift <= float(tolerances["norm_drift"])
        and fidelity >= float(tolerances["min_fidelity"])
    )
    return {
        "schema_version": "validation_record_stage1a-0.1",
        "experiment_id": config["experiment"]["id"],
        "reference": config["validation"]["reference"],
        "metrics": {
            "max_abs_error": max_abs,
            "max_rel_error": max_rel,
            "norm_drift": norm_drift,
            "fidelity": fidelity,
            "actual_norm": actual_norm,
            "reference_norm": reference_norm,
        },
        "tolerances": tolerances,
        "passed": passed,
        "validation_seconds": validation_seconds,
    }

