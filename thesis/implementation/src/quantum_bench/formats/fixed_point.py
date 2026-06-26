from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from quantum_bench.core.records import JsonDict


RouteDType = Literal["int8", "int16"]
ComplexPolicy = Literal["reject", "split_real_imag_last_axis"]


@dataclass(frozen=True)
class FixedPointSpec:
    route_dtype: RouteDType = "int8"
    scale: float | None = None
    zero_point: int = 0
    signed: bool = True
    symmetric: bool = True
    quantization_mode: str = "symmetric"
    rounding: str = "nearest_even"
    clipping: bool = True
    complex_policy: ComplexPolicy = "reject"


@dataclass(frozen=True)
class ConversionErrorMetrics:
    max_abs_error: float
    l2_error: float
    relative_l2_error: float | None


@dataclass(frozen=True)
class FixedPointConversionRecord:
    source_dtype: str
    route_dtype: str
    scale: float
    zero_point: int
    signed: bool
    symmetric: bool
    quantization_mode: str
    rounding: str
    clipping: bool
    complex_policy: str
    representation: str
    shape: tuple[int, ...]
    converted_shape: tuple[int, ...]
    source_bytes: int
    converted_bytes: int
    conversion_time_s: float
    quantization_error: ConversionErrorMetrics
    dequantization_error: ConversionErrorMetrics
    status: str
    reject_reason: str | None
    clipping_count: int
    saturation_count: int
    min_quantized: int | None
    max_quantized: int | None
    metadata: JsonDict = field(default_factory=dict)


@dataclass(frozen=True)
class ConvertedTensor:
    array: np.ndarray
    record: FixedPointConversionRecord


def quantize_fixed_point(array: Any, spec: FixedPointSpec | None = None) -> ConvertedTensor:
    spec = spec or FixedPointSpec()
    _validate_spec(spec)
    source = np.asarray(array)
    if not _is_supported_source_dtype(source.dtype):
        raise ValueError(f"Unsupported source dtype for fixed-point conversion: {source.dtype}")

    start = time.perf_counter()
    packed, representation = _pack_source_array(source, spec)
    scale = _resolve_scale(packed, spec)
    qmin, qmax, dtype = _quantized_range(spec.route_dtype)
    unrounded = packed / scale + spec.zero_point
    rounded = np.rint(unrounded)
    clipping_mask = (rounded < qmin) | (rounded > qmax)
    converted = np.clip(rounded, qmin, qmax).astype(dtype)
    dequantized_packed = (converted.astype(np.float64) - spec.zero_point) * scale
    dequantized = _unpack_dequantized(dequantized_packed, source.shape, representation)
    conversion_time_s = time.perf_counter() - start

    record = FixedPointConversionRecord(
        source_dtype=str(source.dtype),
        route_dtype=spec.route_dtype,
        scale=float(scale),
        zero_point=spec.zero_point,
        signed=spec.signed,
        symmetric=spec.symmetric,
        quantization_mode=spec.quantization_mode,
        rounding=spec.rounding,
        clipping=spec.clipping,
        complex_policy=spec.complex_policy,
        representation=representation,
        shape=tuple(int(dim) for dim in source.shape),
        converted_shape=tuple(int(dim) for dim in converted.shape),
        source_bytes=int(source.nbytes),
        converted_bytes=int(converted.nbytes),
        conversion_time_s=float(conversion_time_s),
        quantization_error=conversion_error_metrics(packed, dequantized_packed),
        dequantization_error=conversion_error_metrics(source, dequantized),
        status="converted",
        reject_reason=None,
        clipping_count=int(np.count_nonzero(clipping_mask)),
        saturation_count=int(np.count_nonzero((converted == qmin) | (converted == qmax))),
        min_quantized=_optional_int(converted.min()) if converted.size else None,
        max_quantized=_optional_int(converted.max()) if converted.size else None,
        metadata={
            "format": "fixed_point_symmetric",
            "qmin": qmin,
            "qmax": qmax,
            "complex_split_axis": -1 if representation == "split_complex_real_imag" else None,
        },
    )
    return ConvertedTensor(converted, record)


def dequantize_fixed_point(
    converted: ConvertedTensor | np.ndarray,
    record_or_spec: FixedPointConversionRecord | FixedPointSpec | None = None,
    dtype: np.dtype | type | str | None = None,
) -> np.ndarray:
    if isinstance(converted, ConvertedTensor):
        array = converted.array
        record = converted.record if record_or_spec is None else record_or_spec
    else:
        array = np.asarray(converted)
        if record_or_spec is None:
            raise ValueError("record_or_spec is required when dequantizing a raw array")
        record = record_or_spec

    scale = _record_scale(record)
    zero_point = _record_zero_point(record)
    representation = _record_representation(record)
    source_shape = _record_shape(record, array)
    result = (array.astype(np.float64) - zero_point) * scale
    result = _unpack_dequantized(result, source_shape, representation)

    if dtype is not None:
        return result.astype(dtype)
    if isinstance(record, FixedPointConversionRecord):
        source_dtype = np.dtype(record.source_dtype)
        if np.issubdtype(source_dtype, np.floating) or np.issubdtype(source_dtype, np.complexfloating):
            return result.astype(source_dtype)
    return result


def conversion_error_metrics(reference: Any, actual: Any) -> ConversionErrorMetrics:
    reference_array = np.asarray(reference)
    actual_array = np.asarray(actual)
    diff = actual_array - reference_array
    if diff.size == 0:
        return ConversionErrorMetrics(0.0, 0.0, 0.0)
    abs_diff = np.abs(diff)
    max_abs_error = float(np.max(abs_diff))
    l2_error = float(np.linalg.norm(diff.ravel()))
    reference_norm = float(np.linalg.norm(reference_array.ravel()))
    if reference_norm == 0.0:
        relative_l2_error = 0.0 if l2_error == 0.0 else None
    else:
        relative_l2_error = l2_error / reference_norm
    return ConversionErrorMetrics(max_abs_error, l2_error, relative_l2_error)


def _validate_spec(spec: FixedPointSpec) -> None:
    if spec.route_dtype not in {"int8", "int16"}:
        raise ValueError(f"Unsupported fixed-point route dtype: {spec.route_dtype}")
    if not spec.signed or not spec.symmetric or spec.quantization_mode != "symmetric":
        raise ValueError("Only signed symmetric fixed-point quantization is implemented")
    if spec.zero_point != 0:
        raise ValueError("Only zero_point=0 is implemented for symmetric quantization")
    if spec.rounding != "nearest_even":
        raise ValueError("Only nearest_even rounding is implemented")
    if not spec.clipping:
        raise ValueError("Fixed-point conversion requires clipping to be enabled")
    if spec.complex_policy not in {"reject", "split_real_imag_last_axis"}:
        raise ValueError(f"Unsupported complex policy: {spec.complex_policy}")
    if spec.scale is not None and spec.scale <= 0.0:
        raise ValueError("Fixed-point scale must be positive")


def _is_supported_source_dtype(dtype: np.dtype) -> bool:
    return np.issubdtype(dtype, np.floating) or np.issubdtype(dtype, np.complexfloating)


def _pack_source_array(array: np.ndarray, spec: FixedPointSpec) -> tuple[np.ndarray, str]:
    if np.issubdtype(array.dtype, np.complexfloating):
        if spec.complex_policy != "split_real_imag_last_axis":
            raise ValueError("Complex fixed-point conversion requires complex_policy='split_real_imag_last_axis'")
        packed = np.stack((array.real, array.imag), axis=-1).astype(np.float64, copy=False)
        return packed, "split_complex_real_imag"
    return array.astype(np.float64, copy=False), "real"


def _resolve_scale(packed: np.ndarray, spec: FixedPointSpec) -> float:
    if spec.scale is not None:
        return float(spec.scale)
    if packed.size == 0:
        return 1.0
    max_abs = float(np.max(np.abs(packed)))
    if max_abs == 0.0:
        return 1.0
    _, qmax, _ = _quantized_range(spec.route_dtype)
    return max_abs / qmax


def _quantized_range(route_dtype: str) -> tuple[int, int, np.dtype]:
    if route_dtype == "int8":
        return -127, 127, np.dtype(np.int8)
    if route_dtype == "int16":
        return -32767, 32767, np.dtype(np.int16)
    raise ValueError(f"Unsupported fixed-point route dtype: {route_dtype}")


def _unpack_dequantized(packed: np.ndarray, source_shape: tuple[int, ...], representation: str) -> np.ndarray:
    if representation == "split_complex_real_imag":
        if packed.shape != (*source_shape, 2):
            raise ValueError(f"Converted complex tensor shape {packed.shape} does not match source shape {source_shape}")
        return packed[..., 0] + 1j * packed[..., 1]
    return packed


def _record_scale(record: FixedPointConversionRecord | FixedPointSpec) -> float:
    scale = record.scale
    if scale is None:
        raise ValueError("A concrete fixed-point scale is required for dequantization")
    return float(scale)


def _record_zero_point(record: FixedPointConversionRecord | FixedPointSpec) -> int:
    return int(record.zero_point)


def _record_representation(record: FixedPointConversionRecord | FixedPointSpec) -> str:
    if isinstance(record, FixedPointConversionRecord):
        return record.representation
    return "split_complex_real_imag" if record.complex_policy == "split_real_imag_last_axis" else "real"


def _record_shape(record: FixedPointConversionRecord | FixedPointSpec, array: np.ndarray) -> tuple[int, ...]:
    if isinstance(record, FixedPointConversionRecord):
        return record.shape
    representation = _record_representation(record)
    if representation == "split_complex_real_imag":
        if not array.shape or array.shape[-1] != 2:
            raise ValueError("Converted complex tensor must have a final real/imag axis of length 2")
        return tuple(int(dim) for dim in array.shape[:-1])
    return tuple(int(dim) for dim in array.shape)


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    return int(value)
