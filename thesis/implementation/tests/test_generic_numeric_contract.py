from __future__ import annotations

import numpy as np

from quantum_bench.routing.generic_numeric_contract import classify_numeric
from quantum_bench.targets.upmem.runtime_evidence import transfer_accounting


def test_numeric_classifier_distinguishes_real_and_complex_categories() -> None:
    assert classify_numeric(np.array([1, 2], dtype=np.int64)).kind == "real"
    zero_imag = classify_numeric(np.array([1 + 0j], dtype=np.complex128))
    assert zero_imag.kind == "complex_zero_imag"
    assert zero_imag.is_complex is True
    nonzero = classify_numeric(np.array([1 + 2j], dtype=np.complex128))
    assert nonzero.kind == "complex_nonzero"
    assert nonzero.has_nonzero_imaginary is True


def test_numeric_classifier_prioritizes_nonfinite_values() -> None:
    result = classify_numeric(np.array([1 + np.nan * 1j], dtype=np.complex128))

    assert result.kind == "nonfinite"
    assert result.has_nonfinite is True


def test_transfer_accounting_separates_prepared_and_sdk_observed_values() -> None:
    result = transfer_accounting(
        64,
        24,
        declared_total_bytes=88,
        recorded_by_sdk=True,
        prepared_h2d_bytes=48,
        prepared_d2h_bytes=16,
        control_bytes=16,
        alignment_padding_bytes=8,
    )

    assert result["actual_transfer_bytes"] == 88
    assert result["prepared_payload_h2d_bytes"] == 48
    assert result["prepared_payload_d2h_bytes"] == 16
    assert result["sdk_observed_h2d_bytes"] == 64
    assert result["sdk_observed_d2h_bytes"] == 24
    assert result["transfer_components"]["h2d_application_visible_payload_bytes"] == 48
    assert result["transfer_components"]["d2h_application_visible_payload_bytes"] == 16
    assert result["transfer_components"]["sdk_observed_h2d_nonpayload_bytes"] == 16
    assert result["transfer_components"]["sdk_observed_d2h_nonpayload_bytes"] == 8
    assert result["transfer_components"]["control_structure_bytes"] == 16
    assert result["transfer_components"]["alignment_padding_bytes"] == 8
