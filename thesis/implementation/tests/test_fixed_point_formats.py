from __future__ import annotations

import json

import numpy as np
import pytest

from quantum_bench.core.records import to_jsonable
from quantum_bench.formats import (
    FixedPointSpec,
    conversion_error_metrics,
    dequantize_fixed_point,
    quantize_fixed_point,
)


def test_float64_to_int8_round_trip_records_error_metrics() -> None:
    source = np.array([-1.0, -0.5, 0.0, 0.25, 1.0], dtype=np.float64)
    converted = quantize_fixed_point(source, FixedPointSpec(route_dtype="int8"))
    restored = dequantize_fixed_point(converted, dtype=np.float64)

    assert converted.array.dtype == np.int8
    assert converted.record.source_dtype == "float64"
    assert converted.record.route_dtype == "int8"
    assert converted.record.scale == pytest.approx(1.0 / 127.0)
    assert converted.record.zero_point == 0
    assert converted.record.signed is True
    assert converted.record.symmetric is True
    assert converted.record.shape == source.shape
    assert converted.record.converted_shape == source.shape
    assert converted.record.source_bytes == source.nbytes
    assert converted.record.converted_bytes == converted.array.nbytes
    assert converted.record.status == "converted"
    assert converted.record.reject_reason is None
    assert converted.record.quantization_error.max_abs_error <= converted.record.scale / 2.0
    assert converted.record.dequantization_error.l2_error == pytest.approx(np.linalg.norm(restored - source))
    assert np.max(np.abs(restored - source)) <= converted.record.scale / 2.0


def test_float32_to_int16_round_trip_uses_source_dtype_by_default() -> None:
    source = np.array([-2.0, -0.25, 0.25, 2.0], dtype=np.float32)
    converted = quantize_fixed_point(source, FixedPointSpec(route_dtype="int16"))
    restored = dequantize_fixed_point(converted)

    assert converted.array.dtype == np.int16
    assert restored.dtype == np.float32
    assert converted.record.source_dtype == "float32"
    assert converted.record.scale == pytest.approx(2.0 / 32767.0)
    assert np.max(np.abs(restored.astype(np.float64) - source.astype(np.float64))) <= converted.record.scale


def test_zero_array_uses_unit_scale_and_round_trips_exactly() -> None:
    source = np.zeros((2, 3), dtype=np.float64)
    converted = quantize_fixed_point(source, FixedPointSpec(route_dtype="int8"))
    restored = dequantize_fixed_point(converted, dtype=np.float64)

    assert converted.record.scale == 1.0
    assert converted.record.min_quantized == 0
    assert converted.record.max_quantized == 0
    assert converted.record.clipping_count == 0
    assert converted.record.saturation_count == 0
    assert np.array_equal(converted.array, np.zeros_like(source, dtype=np.int8))
    assert np.array_equal(restored, source)
    assert converted.record.dequantization_error.relative_l2_error == 0.0


def test_scale_is_deterministic_for_repeated_calls() -> None:
    source = np.array([0.125, -0.75, 0.5], dtype=np.float64)
    first = quantize_fixed_point(source, FixedPointSpec(route_dtype="int8"))
    second = quantize_fixed_point(source, FixedPointSpec(route_dtype="int8"))

    assert first.record.scale == second.record.scale
    assert np.array_equal(first.array, second.array)


def test_explicit_scale_records_clipping_and_saturation() -> None:
    source = np.array([-10.0, 0.0, 10.0], dtype=np.float64)
    converted = quantize_fixed_point(source, FixedPointSpec(route_dtype="int8", scale=0.01))

    assert converted.record.scale == 0.01
    assert converted.record.clipping_count == 2
    assert converted.record.saturation_count == 2
    assert converted.record.min_quantized == -127
    assert converted.record.max_quantized == 127
    assert np.array_equal(converted.array, np.array([-127, 0, 127], dtype=np.int8))


def test_error_metrics_handle_zero_reference_norm_safely() -> None:
    exact_zero = conversion_error_metrics(np.zeros(3), np.zeros(3))
    nonzero_error = conversion_error_metrics(np.zeros(3), np.ones(3))

    assert exact_zero.max_abs_error == 0.0
    assert exact_zero.l2_error == 0.0
    assert exact_zero.relative_l2_error == 0.0
    assert nonzero_error.max_abs_error == 1.0
    assert nonzero_error.l2_error == pytest.approx(np.sqrt(3.0))
    assert nonzero_error.relative_l2_error is None


def test_complex_split_real_imag_round_trip() -> None:
    source = np.array([1.0 + 0.5j, -0.25 - 1.0j], dtype=np.complex128)
    spec = FixedPointSpec(route_dtype="int8", complex_policy="split_real_imag_last_axis")
    converted = quantize_fixed_point(source, spec)
    restored = dequantize_fixed_point(converted, dtype=np.complex128)

    assert converted.array.dtype == np.int8
    assert converted.array.shape == (2, 2)
    assert converted.record.shape == source.shape
    assert converted.record.converted_shape == (2, 2)
    assert converted.record.representation == "split_complex_real_imag"
    assert converted.record.complex_policy == "split_real_imag_last_axis"
    assert converted.record.metadata["complex_split_axis"] == -1
    assert np.max(np.abs(restored - source)) <= converted.record.scale / 2.0


def test_complex_default_policy_rejects_explicitly() -> None:
    source = np.array([1.0 + 1.0j], dtype=np.complex128)

    with pytest.raises(ValueError, match="Complex fixed-point conversion requires"):
        quantize_fixed_point(source, FixedPointSpec(route_dtype="int8"))


def test_unsupported_source_dtype_and_invalid_spec_are_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported source dtype"):
        quantize_fixed_point(np.array([1, 2, 3], dtype=np.int64), FixedPointSpec(route_dtype="int8"))

    with pytest.raises(ValueError, match="Unsupported fixed-point route dtype"):
        quantize_fixed_point(np.array([1.0], dtype=np.float64), FixedPointSpec(route_dtype="uint8"))  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="scale must be positive"):
        quantize_fixed_point(np.array([1.0], dtype=np.float64), FixedPointSpec(route_dtype="int8", scale=0.0))


def test_conversion_record_is_json_serializable() -> None:
    source = np.array([0.0, 1.0], dtype=np.float64)
    converted = quantize_fixed_point(source, FixedPointSpec(route_dtype="int8"))

    payload = to_jsonable(converted.record)
    json.dumps(payload)
    assert payload["route_dtype"] == "int8"
    assert payload["dequantization_error"]["max_abs_error"] >= 0.0
