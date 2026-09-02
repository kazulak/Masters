"""CPU-only numerical qualification for ``complex_int8_shared_scale_v1``.

This module is deliberately separate from the accepted runtime numerical
policies.  It is an analysis oracle for the policy that a later physical
implementation may reproduce.

For a complex tensor ``X`` the real and imaginary planes share one scale::

    s = max(max(abs(real(X))), max(abs(imag(X)))) / 127

with ``s == 1`` for an all-zero tensor.  Each component is encoded with
nearest-even rounding and clipping to ``[-127, 127]``.  A contraction uses
four explicitly named integer products and is decoded with the single scale
product ``s_A * s_B``.

The replay functions intentionally keep reductions in complex64 host
arithmetic.  Only binary ``ContractNode`` operations are quantized.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Iterator

import numpy as np

from quantum_bench.lowering import validate_contraction_dag, validate_dag_inputs
from quantum_bench.model import ContractNode, ContractionDAG, ReduceNode, TensorView
from quantum_bench.numerics import (
    contract_complex_products,
    decode_complex_products,
    encode_complex_tensor,
)


POLICY_ID = "complex_int8_shared_scale_v1"
"""The analysis-only identifier for this fixed numerical policy."""

INT8_MAX = 127
INT8_MIN = -127
SCALE_METADATA_BYTES = np.dtype(np.float64).itemsize
FLOAT32_COMPLEX_BYTES = 2 * np.dtype(np.float32).itemsize
INT8_COMPLEX_BYTES = 2 * np.dtype(np.int8).itemsize
INT32_MAX = int(np.iinfo(np.int32).max)
INT64_MAX = int(np.iinfo(np.int64).max)


@dataclass(frozen=True, slots=True)
class QuantizationDiagnostics:
    """Deterministic facts about one shared-scale tensor encoding."""

    real_min: float
    real_max: float
    imag_min: float
    imag_max: float
    q_real_min: int
    q_real_max: int
    q_imag_min: int
    q_imag_max: int
    real_zero_count: int
    imag_zero_count: int
    zero_count: int
    real_clipping_count: int
    imag_clipping_count: int
    clipping_count: int
    real_boundary_saturation_count: int
    imag_boundary_saturation_count: int
    boundary_saturation_count: int
    max_abs_component: float
    min_nonzero_abs_component: float | None

    @property
    def saturation_real(self) -> int:
        """Historical-compatible strict clipping count for the real plane."""

        return self.real_clipping_count

    @property
    def saturation_imag(self) -> int:
        """Historical-compatible strict clipping count for the imaginary plane."""

        return self.imag_clipping_count

    @property
    def max_nonzero_abs_component(self) -> float:
        """The largest absolute nonzero scalar component."""

        return self.max_abs_component


@dataclass(frozen=True, slots=True)
class QuantizedComplexTensor:
    """Immutable split-complex int8 data with one shared Python-float scale."""

    q_real: np.ndarray
    q_imag: np.ndarray
    scale: float
    diagnostics: QuantizationDiagnostics | None = None

    def __post_init__(self) -> None:
        real = np.asarray(self.q_real)
        imag = np.asarray(self.q_imag)
        if real.shape != imag.shape:
            raise ValueError("quantized real and imaginary planes must match")
        if real.dtype != np.dtype(np.int8) or imag.dtype != np.dtype(np.int8):
            raise ValueError("quantized planes must have dtype int8")
        if np.any(real < INT8_MIN) or np.any(real > INT8_MAX):
            raise ValueError("quantized real plane contains a value outside [-127, 127]")
        if np.any(imag < INT8_MIN) or np.any(imag > INT8_MAX):
            raise ValueError("quantized imaginary plane contains a value outside [-127, 127]")
        try:
            scale = float(self.scale)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("quantization scale must be finite and positive") from exc
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError("quantization scale must be finite and positive")
        real = _readonly_copy(real)
        imag = _readonly_copy(imag)
        diagnostics = self.diagnostics
        if diagnostics is None:
            diagnostics = _diagnostics_from_encoded_planes(real, imag, scale)
        object.__setattr__(self, "q_real", real)
        object.__setattr__(self, "q_imag", imag)
        object.__setattr__(self, "scale", scale)
        object.__setattr__(self, "diagnostics", diagnostics)

    @property
    def real(self) -> np.ndarray:
        """Alias matching the historical split-plane record."""

        return self.q_real

    @property
    def imag(self) -> np.ndarray:
        """Alias matching the historical split-plane record."""

        return self.q_imag

    @property
    def encoded_element_count(self) -> int:
        return int(self.q_real.size)

    @property
    def saturation_real(self) -> int:
        return self.diagnostics.real_clipping_count

    @property
    def saturation_imag(self) -> int:
        return self.diagnostics.imag_clipping_count

    def as_tuple(self) -> tuple[np.ndarray, np.ndarray, float, QuantizationDiagnostics]:
        """Return planes, scale, and diagnostics without exposing mutable state."""

        return self.q_real, self.q_imag, self.scale, self.diagnostics


def quantize_complex_shared_scale(value: np.ndarray) -> QuantizedComplexTensor:
    """Quantize a finite real or complex tensor under the fixed v1 policy."""

    real, imag = _finite_float64_planes(value)
    max_abs_component = max(_max_abs(real), _max_abs(imag))
    scale = float(max_abs_component / float(INT8_MAX)) if max_abs_component else 1.0
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("quantization scale underflowed or is nonfinite")

    q_real, real_clipping = _quantize_plane(real, scale)
    q_imag, imag_clipping = _quantize_plane(imag, scale)
    diagnostics = _make_diagnostics(
        real,
        imag,
        q_real,
        q_imag,
        real_clipping,
        imag_clipping,
        max_abs_component,
    )
    return QuantizedComplexTensor(q_real, q_imag, scale, diagnostics)


def dequantize_complex(
    q_real: QuantizedComplexTensor | np.ndarray,
    q_imag: np.ndarray | None = None,
    scale: float | None = None,
) -> np.ndarray:
    """Decode split int8 planes to a finite, read-only historical complex64."""

    if isinstance(q_real, QuantizedComplexTensor):
        if q_imag is not None or scale is not None:
            raise ValueError("a QuantizedComplexTensor already contains imag and scale")
        real = q_real.q_real
        imag = q_real.q_imag
        decode_scale = q_real.scale
    else:
        if q_imag is None or scale is None:
            raise ValueError("q_real, q_imag, and scale are required")
        real = np.asarray(q_real)
        imag = np.asarray(q_imag)
        decode_scale = _positive_finite_scale(scale)
        _validate_encoded_planes(real, imag)
    with np.errstate(over="ignore", invalid="ignore"):
        real64 = np.asarray(real, dtype=np.float64) * float(decode_scale)
        imag64 = np.asarray(imag, dtype=np.float64) * float(decode_scale)
    return _readonly_complex64(real64, imag64)


@dataclass(frozen=True, slots=True)
class ContractionGeometry:
    """Matrixized geometry facts for a binary tensor contraction."""

    B: int
    M: int
    N: int
    K: int

    @property
    def output_elements(self) -> int:
        return self.B * self.M * self.N


@dataclass(frozen=True, slots=True)
class AccumulatorBounds:
    """Worst-case integer accumulator facts for one local K extent."""

    K: int
    lane_bound: int
    component_bound: int
    int32_safe: bool
    int64_safe: bool

    @property
    def theoretical_int32_accumulator_bound(self) -> int:
        return self.component_bound

    @property
    def theoretical_int64_accumulator_bound(self) -> int:
        return self.component_bound


@dataclass(frozen=True, slots=True)
class IntegerContractionProducts:
    """The four explicitly named int64 product accumulators."""

    rr: np.ndarray
    ii: np.ndarray
    ri: np.ndarray
    ir: np.ndarray

    def __post_init__(self) -> None:
        arrays = tuple(np.asarray(item) for item in (self.rr, self.ii, self.ri, self.ir))
        if any(array.dtype != np.dtype(np.int64) for array in arrays):
            raise ValueError("integer product accumulators must have dtype int64")
        if any(array.shape != arrays[0].shape for array in arrays[1:]):
            raise ValueError("integer product accumulators must have equal shapes")
        for name, array in zip(("rr", "ii", "ri", "ir"), arrays, strict=True):
            object.__setattr__(self, name, _readonly_copy(array))

    def as_tuple(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return self.rr, self.ii, self.ri, self.ir


@dataclass(frozen=True, slots=True)
class ContractionFacts:
    """Small facts record returned by a standalone int8 contraction."""

    B: int
    M: int
    N: int
    K: int
    left_scale: float
    right_scale: float
    output_scale: float
    int32_theoretical_accumulator_bound: int
    int64_theoretical_accumulator_bound: int
    int32_safe: bool
    int64_safe: bool
    integer_multiply_accumulate_count: int
    logical_encoded_bytes: int
    logical_float32_complex_bytes: int


@dataclass(frozen=True, slots=True)
class ErrorMetrics:
    """Phase-sensitive error metrics matching the repository convention."""

    max_abs_error: float
    relative_l2: float
    norm_drift: float


@dataclass(frozen=True, slots=True)
class ContractionTrace:
    """Per-contract quantization and error facts for a replay."""

    node_id: str
    operand_a_shape: tuple[int, ...]
    operand_b_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    B: int
    M: int
    N: int
    K: int
    operand_a_scale: float
    operand_b_scale: float
    operand_a_max_abs_component: float
    operand_b_max_abs_component: float
    operand_a_min_nonzero_abs_component: float | None
    operand_b_min_nonzero_abs_component: float | None
    operand_a_real_min: float
    operand_a_real_max: float
    operand_a_imag_min: float
    operand_a_imag_max: float
    operand_b_real_min: float
    operand_b_real_max: float
    operand_b_imag_min: float
    operand_b_imag_max: float
    operand_a_encoded_element_count: int
    operand_b_encoded_element_count: int
    operand_a_q_real_min: int
    operand_a_q_real_max: int
    operand_a_q_imag_min: int
    operand_a_q_imag_max: int
    operand_b_q_real_min: int
    operand_b_q_real_max: int
    operand_b_q_imag_min: int
    operand_b_q_imag_max: int
    operand_a_zero_count: int
    operand_b_zero_count: int
    operand_a_clipping_count: int
    operand_b_clipping_count: int
    operand_a_boundary_saturation_count: int
    operand_b_boundary_saturation_count: int
    int32_theoretical_accumulator_bound: int
    int32_accumulator_safe: bool
    int64_theoretical_accumulator_bound: int
    int64_accumulator_safe: bool
    local_max_abs_error_vs_same_node_float32: float
    local_relative_l2_vs_same_node_float32: float
    local_norm_drift_vs_same_node_float32: float
    cumulative_max_abs_error_vs_same_node_float32: float
    cumulative_relative_l2_vs_same_node_float32: float
    cumulative_norm_drift_vs_same_node_float32: float
    theoretical_local_error_bound: float
    ideal_local_max_abs_error: float
    observed_local_error: float
    rounding_bound_applicable: bool
    scale_computation_count: int
    quantization_event_count: int
    requantization_event_count: int
    logical_encoded_bytes: int
    logical_float32_complex_bytes: int
    nominal_operand_compression_ratio: float
    scale_metadata_bytes: int
    integer_multiply_accumulate_count: int

    @property
    def output_max_abs_error_vs_same_node_float32(self) -> float:
        """Local sensitivity under the requested name."""

        return self.local_max_abs_error_vs_same_node_float32

    @property
    def cumulative_output_error(self) -> float:
        return self.cumulative_max_abs_error_vs_same_node_float32


@dataclass(frozen=True, slots=True)
class QuantizedDAGReplay:
    """Complete CPU-only policy replay and phase-sensitive comparisons."""

    output: np.ndarray
    float32_output: np.ndarray
    complex128_output: np.ndarray
    traces: tuple[ContractionTrace, ...]
    float32_intermediates: Mapping[str, np.ndarray]
    quantized_intermediates: Mapping[str, np.ndarray]
    max_abs_error_vs_float32_same_dag: float
    relative_l2_vs_float32_same_dag: float
    norm_drift_vs_float32_same_dag: float
    max_abs_error_vs_complex128: float
    relative_l2_vs_complex128: float
    norm_drift_vs_complex128: float
    max_abs_error_float32_vs_complex128: float
    relative_l2_float32_vs_complex128: float
    norm_drift_float32_vs_complex128: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "output", _readonly_array(self.output))
        object.__setattr__(self, "float32_output", _readonly_array(self.float32_output))
        object.__setattr__(self, "complex128_output", _readonly_array(self.complex128_output))
        object.__setattr__(
            self,
            "float32_intermediates",
            _readonly_array_mapping(self.float32_intermediates),
        )
        object.__setattr__(
            self,
            "quantized_intermediates",
            _readonly_array_mapping(self.quantized_intermediates),
        )

    @property
    def final_output(self) -> np.ndarray:
        return self.output

    @property
    def node_trace(self) -> tuple[ContractionTrace, ...]:
        return self.traces

    def __iter__(self) -> Iterator[object]:
        """Allow the convenient ``output, trace = replay(...)`` spelling."""

        yield self.output
        yield self.traces


def contraction_geometry(node: ContractNode) -> ContractionGeometry:
    """Return deterministic ``B, M, N, K`` facts for an existing ContractNode."""

    _validate_contract_node(node)
    left_labels = set(node.left.labels)
    right_labels = set(node.right.labels)
    output_labels = tuple(node.output_labels)
    batch = [label for label in output_labels if label in left_labels and label in right_labels]
    left_free = [label for label in output_labels if label in left_labels and label not in right_labels]
    right_free = [label for label in output_labels if label in right_labels and label not in left_labels]
    dimensions = _label_dimensions(node)
    return ContractionGeometry(
        B=_product(dimensions[label] for label in batch),
        M=_product(dimensions[label] for label in left_free),
        N=_product(dimensions[label] for label in right_free),
        K=_product(dimensions[label] for label in node.contracted_labels),
    )


def contraction_dimensions(node: ContractNode) -> tuple[int, int, int, int]:
    """Return ``(B, M, N, K)`` for callers that prefer a plain tuple."""

    geometry = contraction_geometry(node)
    return geometry.B, geometry.M, geometry.N, geometry.K


def bmnk(node: ContractNode) -> tuple[int, int, int, int]:
    """Short alias for :func:`contraction_dimensions`."""

    return contraction_dimensions(node)


def accumulator_bounds(K: int) -> AccumulatorBounds:
    """Return lane and full-component worst-case bounds for one K extent.

    A lane has ``K * 127**2`` magnitude.  Combining the two same-sign lanes
    gives the conservative full real/imaginary component bound
    ``2 * K * 127**2``.  The latter is the safety fact used here for both
    int32 qualification and int64 reference preflight.
    """

    k = _positive_integer(K, "K")
    lane_bound = k * INT8_MAX * INT8_MAX
    component_bound = 2 * lane_bound
    return AccumulatorBounds(
        K=k,
        lane_bound=lane_bound,
        component_bound=component_bound,
        int32_safe=component_bound <= INT32_MAX,
        int64_safe=component_bound <= INT64_MAX,
    )


def theoretical_accumulator_bound(K: int) -> int:
    return accumulator_bounds(K).component_bound


def int32_accumulator_safe(K: int) -> bool:
    return accumulator_bounds(K).int32_safe


def int64_accumulator_safe(K: int) -> bool:
    return accumulator_bounds(K).int64_safe


def contract_integer_products(
    left: QuantizedComplexTensor,
    right: QuantizedComplexTensor,
    node: ContractNode,
) -> IntegerContractionProducts:
    """Compute rr, ii, ri, and ir explicitly with checked int64 arithmetic."""

    _validate_contract_node(node)
    _validate_quantized_operand(left, node.left.shape, "left")
    _validate_quantized_operand(right, node.right.shape, "right")
    geometry = contraction_geometry(node)
    bounds = accumulator_bounds(geometry.K)
    if not bounds.int64_safe:
        raise ValueError(
            "int8 contraction exceeds the int64 full-component accumulation bound"
        )
    left_indices, right_indices, output_indices = _einsum_indices(node)
    left_real = np.asarray(left.q_real, dtype=np.int64)
    left_imag = np.asarray(left.q_imag, dtype=np.int64)
    right_real = np.asarray(right.q_real, dtype=np.int64)
    right_imag = np.asarray(right.q_imag, dtype=np.int64)
    return IntegerContractionProducts(
        _integer_einsum(left_real, left_indices, right_real, right_indices, output_indices),
        _integer_einsum(left_imag, left_indices, right_imag, right_indices, output_indices),
        _integer_einsum(left_real, left_indices, right_imag, right_indices, output_indices),
        _integer_einsum(left_imag, left_indices, right_real, right_indices, output_indices),
    )


def decode_integer_products(
    products: IntegerContractionProducts,
    left_scale: float,
    right_scale: float,
) -> np.ndarray:
    """Combine four int64 lanes and decode their shared product scale."""

    output_scale = _product_scale(left_scale, right_scale)
    _check_int64_subtract(products.rr, products.ii)
    _check_int64_add(products.ri, products.ir)
    with np.errstate(over="ignore", invalid="ignore"):
        real = np.subtract(products.rr, products.ii, dtype=np.int64).astype(np.float64)
        imag = np.add(products.ri, products.ir, dtype=np.int64).astype(np.float64)
        real *= output_scale
        imag *= output_scale
    return _readonly_complex64(real, imag)


def contract_complex_int8_reference(
    left: np.ndarray | QuantizedComplexTensor,
    right: np.ndarray | QuantizedComplexTensor,
    node: ContractNode,
) -> tuple[np.ndarray, ContractionFacts]:
    """Quantize operands if needed, contract, and return output plus node facts."""

    left_encoded = (
        left if isinstance(left, QuantizedComplexTensor) else quantize_complex_shared_scale(left)
    )
    right_encoded = (
        right if isinstance(right, QuantizedComplexTensor) else quantize_complex_shared_scale(right)
    )
    products = contract_integer_products(left_encoded, right_encoded, node)
    result = decode_integer_products(products, left_encoded.scale, right_encoded.scale)
    geometry = contraction_geometry(node)
    bounds = accumulator_bounds(geometry.K)
    left_count = left_encoded.encoded_element_count
    right_count = right_encoded.encoded_element_count
    return result, ContractionFacts(
        B=geometry.B,
        M=geometry.M,
        N=geometry.N,
        K=geometry.K,
        left_scale=left_encoded.scale,
        right_scale=right_encoded.scale,
        output_scale=_product_scale(left_encoded.scale, right_encoded.scale),
        int32_theoretical_accumulator_bound=bounds.component_bound,
        int64_theoretical_accumulator_bound=bounds.component_bound,
        int32_safe=bounds.int32_safe,
        int64_safe=bounds.int64_safe,
        integer_multiply_accumulate_count=4 * geometry.output_elements * geometry.K,
        logical_encoded_bytes=2 * (left_count + right_count) + 2 * SCALE_METADATA_BYTES,
        logical_float32_complex_bytes=FLOAT32_COMPLEX_BYTES * (left_count + right_count),
    )


def replay_quantized_dag(
    dag: ContractionDAG,
    inputs: Mapping[str, np.ndarray],
) -> QuantizedDAGReplay:
    """Replay every binary DAG contraction with per-operand shared-scale int8.

    Inputs are first materialized as the accepted float32 representation.  A
    local sensitivity calculation quantizes only the float32-reference
    operands at a node.  The cumulative calculation feeds each dequantized
    complex64 contraction output into the next node and quantizes it again.
    ``ReduceNode`` remains an unquantized complex64 host reduction.
    """

    validate_contraction_dag(dag)
    validate_dag_inputs(dag, inputs)
    order = _topological_order(dag)
    float32_working = {
        tensor_id: _to_complex64(np.asarray(value)) for tensor_id, value in inputs.items()
    }
    cumulative_working = {
        tensor_id: _to_complex64(np.asarray(value)) for tensor_id, value in inputs.items()
    }
    float32_intermediates: dict[str, np.ndarray] = {}
    quantized_intermediates: dict[str, np.ndarray] = {}
    traces: list[ContractionTrace] = []
    produced_tensors = {node.output.id for node in order}

    for node in order:
        if isinstance(node, ContractNode):
            float_left = _materialize_view(node.left, float32_working)
            float_right = _materialize_view(node.right, float32_working)
            float_result = _float32_contract(node, float_left, float_right)
            float32_working[node.output.id] = float_result
            float32_intermediates[node.output.id] = float_result

            local_left = quantize_complex_shared_scale(float_left)
            local_right = quantize_complex_shared_scale(float_right)
            local_products = contract_integer_products(local_left, local_right, node)
            local_result = decode_integer_products(
                local_products, local_left.scale, local_right.scale
            )

            cumulative_left_value = _materialize_view(node.left, cumulative_working)
            cumulative_right_value = _materialize_view(node.right, cumulative_working)
            cumulative_left = quantize_complex_shared_scale(cumulative_left_value)
            cumulative_right = quantize_complex_shared_scale(cumulative_right_value)
            cumulative_products = contract_integer_products(
                cumulative_left, cumulative_right, node
            )
            cumulative_result = decode_integer_products(
                cumulative_products, cumulative_left.scale, cumulative_right.scale
            )
            cumulative_working[node.output.id] = cumulative_result
            quantized_intermediates[node.output.id] = cumulative_result
            traces.append(
                _make_trace(
                    node,
                    float_left,
                    float_right,
                    float_result,
                    local_left,
                    local_right,
                    local_products,
                    local_result,
                    cumulative_left,
                    cumulative_right,
                    cumulative_result,
                    produced_tensors,
                )
            )
        elif isinstance(node, ReduceNode):
            float_values = [_materialize_view(view, float32_working) for view in node.inputs]
            cumulative_values = [
                _materialize_view(view, cumulative_working) for view in node.inputs
            ]
            float_result = _complex64_reduce(float_values)
            cumulative_result = _complex64_reduce(cumulative_values)
            float32_working[node.output.id] = float_result
            cumulative_working[node.output.id] = cumulative_result
            float32_intermediates[node.output.id] = float_result
            quantized_intermediates[node.output.id] = cumulative_result
        else:  # pragma: no cover - GraphNode is closed by the model contract.
            raise TypeError(f"unsupported DAG node: {type(node).__name__}")

    float32_output = _to_complex64(_materialize_view(dag.output, float32_working))
    quantized_output = _to_complex64(_materialize_view(dag.output, cumulative_working))
    # This is intentionally the repository's existing complex128 reference.
    from quantum_bench.cpu import run_complex128_reference

    complex128_output = run_complex128_reference(dag, inputs)
    int8_vs_float32 = _error_metrics(quantized_output, float32_output)
    int8_vs_complex128 = _error_metrics(quantized_output, complex128_output)
    float32_vs_complex128 = _error_metrics(float32_output, complex128_output)
    return QuantizedDAGReplay(
        output=quantized_output,
        float32_output=float32_output,
        complex128_output=complex128_output,
        traces=tuple(traces),
        float32_intermediates=float32_intermediates,
        quantized_intermediates=quantized_intermediates,
        max_abs_error_vs_float32_same_dag=int8_vs_float32.max_abs_error,
        relative_l2_vs_float32_same_dag=int8_vs_float32.relative_l2,
        norm_drift_vs_float32_same_dag=int8_vs_float32.norm_drift,
        max_abs_error_vs_complex128=int8_vs_complex128.max_abs_error,
        relative_l2_vs_complex128=int8_vs_complex128.relative_l2,
        norm_drift_vs_complex128=int8_vs_complex128.norm_drift,
        max_abs_error_float32_vs_complex128=float32_vs_complex128.max_abs_error,
        relative_l2_float32_vs_complex128=float32_vs_complex128.relative_l2,
        norm_drift_float32_vs_complex128=float32_vs_complex128.norm_drift,
    )


def policy_facts(
    *,
    complex_values: int | None = None,
    contraction_count: int = 0,
    B: int = 1,
    M: int = 1,
    N: int = 1,
    K: int = 1,
) -> dict[str, object]:
    """Return static and optional logical facts for future path scoring.

    The byte fields are logical encoded sizes.  They are not H2D bytes, MRAM
    traffic, WRAM traffic, or measured physical transfer reductions.
    """

    contractions = _nonnegative_integer(contraction_count, "contraction_count")
    geometry = ContractionGeometry(
        B=_positive_integer(B, "B"),
        M=_positive_integer(M, "M"),
        N=_positive_integer(N, "N"),
        K=_positive_integer(K, "K"),
    )
    bounds = accumulator_bounds(geometry.K)
    facts: dict[str, object] = {
        "numeric_policy": POLICY_ID,
        "policy_id": POLICY_ID,
        "bytes_per_complex_operand": INT8_COMPLEX_BYTES,
        "scale_metadata_size": SCALE_METADATA_BYTES,
        "scale_dtype": "float64",
        "accumulator_dtype": "int64_reference",
        "accumulator_requirement": "2*K*127^2",
        "four_real_product_count": 4,
        "scale_computations_per_contraction": 2,
        "scale_reduction_count": 2 * contractions,
        "quantization_events_per_contraction": 2,
        "requantization_policy": "per_contraction_per_operand",
        "quantization_event_count": 2 * contractions,
        "requantization_event_count_requires_dag_trace": True,
        "dequantized_output_elements_per_contraction": geometry.output_elements,
        "int32_theoretical_accumulator_bound": bounds.component_bound,
        "int32_accumulator_safe": bounds.int32_safe,
        "int64_theoretical_accumulator_bound": bounds.component_bound,
        "int64_accumulator_safe": bounds.int64_safe,
        "B": geometry.B,
        "M": geometry.M,
        "N": geometry.N,
        "K": geometry.K,
        "integer_multiply_accumulate_count": 4 * geometry.output_elements * geometry.K,
    }
    if complex_values is not None:
        values = _nonnegative_integer(complex_values, "complex_values")
        logical_int8 = INT8_COMPLEX_BYTES * values + SCALE_METADATA_BYTES
        logical_float32 = FLOAT32_COMPLEX_BYTES * values
        facts.update(
            {
                "complex_values": values,
                "scalar_components_quantized": 2 * values,
                "logical_int8_operand_bytes": logical_int8,
                "logical_float32_operand_bytes": logical_float32,
                "logical_encoded_operand_bytes": logical_int8,
                "logical_float32_complex_bytes": logical_float32,
                "nominal_operand_compression_ratio": (
                    float(logical_float32 / logical_int8) if logical_int8 else float("inf")
                ),
                "theoretical_logical_transfer_byte_reduction": logical_float32 - logical_int8,
            }
        )
    return facts


def numeric_policy_description() -> dict[str, object]:
    """Return static policy facts without importing path-planning code."""

    return policy_facts()


def logical_policy_cost_facts(
    complex_values: int,
    *,
    contraction_count: int = 0,
    B: int = 1,
    M: int = 1,
    N: int = 1,
    K: int = 1,
) -> dict[str, object]:
    """Return policy facts including logical encoded sizes for ``complex_values``."""

    return policy_facts(
        complex_values=complex_values,
        contraction_count=contraction_count,
        B=B,
        M=M,
        N=N,
        K=K,
    )


def _finite_float64_planes(value: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError("numeric tensor input is required")
    with np.errstate(over="ignore", invalid="ignore"):
        if np.iscomplexobj(array):
            real = np.asarray(array.real, dtype=np.float64)
            imag = np.asarray(array.imag, dtype=np.float64)
        else:
            real = np.asarray(array, dtype=np.float64)
            imag = np.zeros_like(real, dtype=np.float64)
    if not np.all(np.isfinite(real)) or not np.all(np.isfinite(imag)):
        raise ValueError("complex int8 quantization requires finite float64 components")
    return real, imag


def _quantize_plane(value: np.ndarray, scale: float) -> tuple[np.ndarray, int]:
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        scaled = np.asarray(value, dtype=np.float64) / scale
        rounded = np.rint(scaled)
    if not np.all(np.isfinite(rounded)):
        raise ValueError("quantization produced a nonfinite rounded component")
    clipping_mask = (rounded < INT8_MIN) | (rounded > INT8_MAX)
    clipped = np.clip(rounded, INT8_MIN, INT8_MAX).astype(np.int8)
    return _readonly_copy(clipped), int(np.count_nonzero(clipping_mask))


def _make_diagnostics(
    real: np.ndarray,
    imag: np.ndarray,
    q_real: np.ndarray,
    q_imag: np.ndarray,
    real_clipping: int,
    imag_clipping: int,
    max_abs_component: float,
) -> QuantizationDiagnostics:
    nonzero_components = np.concatenate(
        (np.abs(real).reshape(-1), np.abs(imag).reshape(-1))
    )
    nonzero_components = nonzero_components[nonzero_components != 0.0]
    return QuantizationDiagnostics(
        real_min=_plane_min(real),
        real_max=_plane_max(real),
        imag_min=_plane_min(imag),
        imag_max=_plane_max(imag),
        q_real_min=_plane_min_int(q_real),
        q_real_max=_plane_max_int(q_real),
        q_imag_min=_plane_min_int(q_imag),
        q_imag_max=_plane_max_int(q_imag),
        real_zero_count=int(np.count_nonzero(q_real == 0)),
        imag_zero_count=int(np.count_nonzero(q_imag == 0)),
        zero_count=int(np.count_nonzero(q_real == 0) + np.count_nonzero(q_imag == 0)),
        real_clipping_count=real_clipping,
        imag_clipping_count=imag_clipping,
        clipping_count=real_clipping + imag_clipping,
        real_boundary_saturation_count=int(np.count_nonzero(np.abs(q_real) == INT8_MAX)),
        imag_boundary_saturation_count=int(np.count_nonzero(np.abs(q_imag) == INT8_MAX)),
        boundary_saturation_count=int(
            np.count_nonzero(np.abs(q_real) == INT8_MAX)
            + np.count_nonzero(np.abs(q_imag) == INT8_MAX)
        ),
        max_abs_component=float(max_abs_component),
        min_nonzero_abs_component=(
            float(np.min(nonzero_components)) if nonzero_components.size else None
        ),
    )


def _diagnostics_from_encoded_planes(
    q_real: np.ndarray, q_imag: np.ndarray, scale: float
) -> QuantizationDiagnostics:
    real = np.asarray(q_real, dtype=np.float64) * scale
    imag = np.asarray(q_imag, dtype=np.float64) * scale
    return _make_diagnostics(
        real,
        imag,
        q_real,
        q_imag,
        0,
        0,
        max(_max_abs(real), _max_abs(imag)),
    )


def _validate_encoded_planes(real: np.ndarray, imag: np.ndarray) -> None:
    if real.shape != imag.shape:
        raise ValueError("quantized real and imaginary planes must match")
    if real.dtype != np.dtype(np.int8) or imag.dtype != np.dtype(np.int8):
        raise ValueError("quantized planes must have dtype int8")
    if np.any(real < INT8_MIN) or np.any(real > INT8_MAX):
        raise ValueError("quantized real plane contains a value outside [-127, 127]")
    if np.any(imag < INT8_MIN) or np.any(imag > INT8_MAX):
        raise ValueError("quantized imaginary plane contains a value outside [-127, 127]")


def _validate_quantized_operand(
    operand: QuantizedComplexTensor, shape: tuple[int, ...], name: str
) -> None:
    if not isinstance(operand, QuantizedComplexTensor):
        raise TypeError(f"{name} must be a QuantizedComplexTensor")
    if operand.q_real.shape != shape or operand.q_imag.shape != shape:
        raise ValueError(f"{name} encoded planes do not match node shape")


def _validate_contract_node(node: ContractNode) -> None:
    if not isinstance(node, ContractNode):
        raise TypeError("contraction specification must be a ContractNode")
    for labels, shape, name in (
        (node.left.labels, node.left.shape, "left"),
        (node.right.labels, node.right.shape, "right"),
        (node.output.labels, node.output.shape, "output"),
    ):
        if len(labels) != len(shape) or len(set(labels)) != len(labels):
            raise ValueError(f"{name} labels must be unique and match its shape")
        if any(isinstance(label, (bool, np.bool_)) or not isinstance(label, (int, np.integer)) for label in labels):
            raise ValueError("contraction labels must be integer values")
    if len(node.contracted_labels) != len(set(node.contracted_labels)):
        raise ValueError("contracted labels must be unique")
    if len(node.output_labels) != len(set(node.output_labels)):
        raise ValueError("output labels must be unique")
    if any(
        isinstance(label, (bool, np.bool_)) or not isinstance(label, (int, np.integer))
        for label in (*node.contracted_labels, *node.output_labels)
    ):
        raise ValueError("contraction labels must be integer values")
    if set(node.contracted_labels) & set(node.output_labels):
        raise ValueError("contracted and output labels must be disjoint")
    input_labels = set(node.left.labels) | set(node.right.labels)
    if set(node.contracted_labels) | set(node.output_labels) != input_labels:
        raise ValueError("contraction labels do not cover all input labels")
    for label in set(node.left.labels) & set(node.right.labels):
        left_dim = node.left.shape[node.left.labels.index(label)]
        right_dim = node.right.shape[node.right.labels.index(label)]
        if left_dim != right_dim:
            raise ValueError("shared contraction label dimensions must agree")
    dimensions = _label_dimensions(node)
    if tuple(node.output.shape) != tuple(dimensions[label] for label in node.output_labels):
        raise ValueError("contraction output shape does not match output labels")
    if tuple(node.output.labels) != tuple(node.output_labels):
        raise ValueError("contraction output labels do not match output descriptor")


def _label_dimensions(node: ContractNode) -> dict[int, int]:
    dimensions = {
        int(label): int(size)
        for label, size in zip(node.left.labels, node.left.shape, strict=True)
    }
    for label, size in zip(node.right.labels, node.right.shape, strict=True):
        existing = dimensions.get(int(label))
        if existing is not None and existing != int(size):
            raise ValueError("shared contraction label dimensions must agree")
        dimensions[int(label)] = int(size)
    return dimensions


def _einsum_indices(node: ContractNode) -> tuple[list[int], list[int], list[int]]:
    labels = list(dict.fromkeys((*node.left.labels, *node.right.labels, *node.output_labels)))
    if len(labels) > 52:
        raise ValueError("contraction uses too many distinct labels for NumPy einsum")
    mapping = {label: index for index, label in enumerate(labels)}
    return (
        [mapping[label] for label in node.left.labels],
        [mapping[label] for label in node.right.labels],
        [mapping[label] for label in node.output_labels],
    )


def _integer_einsum(
    left: np.ndarray,
    left_indices: list[int],
    right: np.ndarray,
    right_indices: list[int],
    output_indices: list[int],
) -> np.ndarray:
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.einsum(
            left,
            left_indices,
            right,
            right_indices,
            output_indices,
            dtype=np.int64,
            optimize=False,
        )
    return _readonly_copy(np.asarray(result, dtype=np.int64))


def _positive_finite_scale(value: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("scale must be finite and positive") from exc
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError("scale must be finite and positive")
    return result


def _product_scale(left: float, right: float) -> float:
    left_scale = _positive_finite_scale(left)
    right_scale = _positive_finite_scale(right)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        result = float(np.float64(left_scale) * np.float64(right_scale))
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError("product scale overflowed or underflowed")
    return result


def _check_int64_add(left: np.ndarray, right: np.ndarray) -> None:
    info = np.iinfo(np.int64)
    right_positive = np.maximum(right, 0)
    right_negative = np.minimum(right, 0)
    positive_overflow = (right > 0) & (left > info.max - right_positive)
    negative_overflow = (right < 0) & (left < info.min - right_negative)
    if np.any(positive_overflow | negative_overflow):
        raise ValueError("int64 imaginary accumulator combination would overflow")


def _check_int64_subtract(left: np.ndarray, right: np.ndarray) -> None:
    info = np.iinfo(np.int64)
    right_positive = np.maximum(right, 0)
    right_negative = np.minimum(right, 0)
    positive_overflow = (right < 0) & (left > info.max + right_negative)
    negative_overflow = (right > 0) & (left < info.min + right_positive)
    if np.any(positive_overflow | negative_overflow):
        raise ValueError("int64 real accumulator combination would overflow")


def _float32_contract(
    node: ContractNode, left: np.ndarray, right: np.ndarray
) -> np.ndarray:
    left_encoded = encode_complex_tensor(left, "split_complex_float32_v1")
    right_encoded = encode_complex_tensor(right, "split_complex_float32_v1")
    products = contract_complex_products(
        node,
        left_encoded,
        right_encoded,
        "split_complex_float32_v1",
    )
    result = decode_complex_products(
        products,
        left_encoded.scale,
        right_encoded.scale,
        "split_complex_float32_v1",
    )
    if tuple(result.shape) != tuple(node.output.shape):
        raise ValueError(f"node {node.node_id} produced the wrong output shape")
    return _to_complex64(result)


def _complex64_reduce(values: list[np.ndarray]) -> np.ndarray:
    if not values:
        raise ValueError("a reduction requires at least one input")
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.add.reduce(tuple(_to_complex64(value) for value in values), axis=0, dtype=np.complex64)
    return _to_complex64(result)


def _topological_order(dag: ContractionDAG) -> tuple[ContractNode | ReduceNode, ...]:
    nodes = {node.node_id: node for node in dag.nodes}
    remaining = {node.node_id: set(node.dependencies) for node in dag.nodes}
    ordered: list[ContractNode | ReduceNode] = []
    while remaining:
        ready = sorted(node_id for node_id, dependencies in remaining.items() if not dependencies)
        if not ready:
            raise ValueError("ContractionDAG contains a dependency cycle")
        for node_id in ready:
            ordered.append(nodes[node_id])
            del remaining[node_id]
        for dependencies in remaining.values():
            dependencies.difference_update(ready)
    return tuple(ordered)


def _materialize_view(view: TensorView, tensors: Mapping[str, np.ndarray]) -> np.ndarray:
    try:
        array = np.asarray(tensors[view.tensor_id])
    except KeyError as exc:
        raise ValueError(f"tensor {view.tensor_id!r} is unavailable") from exc
    if not view.slice_spec:
        result = array
    else:
        indices: list[slice | int] = [slice(None)] * array.ndim
        fixed_axes: set[int] = set()
        for axis, value in view.slice_spec:
            if axis in fixed_axes or axis < 0 or axis >= array.ndim:
                raise ValueError(f"tensor view {view.tensor_id!r} has an invalid fixed axis")
            if value < 0 or value >= array.shape[axis]:
                raise ValueError(f"tensor view {view.tensor_id!r} has an invalid fixed value")
            fixed_axes.add(axis)
            indices[axis] = int(value)
        result = array[tuple(indices)]
    if tuple(result.shape) != tuple(view.shape):
        raise ValueError(
            f"tensor view {view.tensor_id!r} produced {result.shape}; expected {view.shape}"
        )
    return np.asarray(result)


def _make_trace(
    node: ContractNode,
    float_left: np.ndarray,
    float_right: np.ndarray,
    float_result: np.ndarray,
    local_left: QuantizedComplexTensor,
    local_right: QuantizedComplexTensor,
    local_products: IntegerContractionProducts,
    local_result: np.ndarray,
    cumulative_left: QuantizedComplexTensor,
    cumulative_right: QuantizedComplexTensor,
    cumulative_result: np.ndarray,
    produced_tensors: set[str],
) -> ContractionTrace:
    geometry = contraction_geometry(node)
    bounds = accumulator_bounds(geometry.K)
    local_reference = _complex128_contract(node, float_left, float_right)
    local_ideal = _decode_products_complex128(
        local_products, local_left.scale, local_right.scale
    )
    local_metrics = _error_metrics(local_result, float_result)
    cumulative_metrics = _error_metrics(cumulative_result, float_result)
    bound, rounding_bound_applicable = _theoretical_local_error_bound(
        float_left,
        float_right,
        local_left,
        local_right,
    )
    local_ideal_error = _error_metrics(local_ideal, local_reference).max_abs_error
    observed_local_error = float(
        np.linalg.norm(
            np.asarray(local_ideal, dtype=np.complex128)
            - np.asarray(local_reference, dtype=np.complex128)
        )
    )
    cumulative_requantizations = sum(
        int(view.tensor_id in produced_tensors) for view in (node.left, node.right)
    )
    encoded_elements = local_left.encoded_element_count + local_right.encoded_element_count
    logical_encoded_bytes = 2 * encoded_elements + 2 * SCALE_METADATA_BYTES
    logical_float32_bytes = FLOAT32_COMPLEX_BYTES * encoded_elements
    diagnostics_a = cumulative_left.diagnostics
    diagnostics_b = cumulative_right.diagnostics
    return ContractionTrace(
        node_id=node.node_id,
        operand_a_shape=tuple(node.left.shape),
        operand_b_shape=tuple(node.right.shape),
        output_shape=tuple(node.output.shape),
        B=geometry.B,
        M=geometry.M,
        N=geometry.N,
        K=geometry.K,
        operand_a_scale=cumulative_left.scale,
        operand_b_scale=cumulative_right.scale,
        operand_a_max_abs_component=diagnostics_a.max_abs_component,
        operand_b_max_abs_component=diagnostics_b.max_abs_component,
        operand_a_min_nonzero_abs_component=diagnostics_a.min_nonzero_abs_component,
        operand_b_min_nonzero_abs_component=diagnostics_b.min_nonzero_abs_component,
        operand_a_real_min=diagnostics_a.real_min,
        operand_a_real_max=diagnostics_a.real_max,
        operand_a_imag_min=diagnostics_a.imag_min,
        operand_a_imag_max=diagnostics_a.imag_max,
        operand_b_real_min=diagnostics_b.real_min,
        operand_b_real_max=diagnostics_b.real_max,
        operand_b_imag_min=diagnostics_b.imag_min,
        operand_b_imag_max=diagnostics_b.imag_max,
        operand_a_encoded_element_count=cumulative_left.encoded_element_count,
        operand_b_encoded_element_count=cumulative_right.encoded_element_count,
        operand_a_q_real_min=diagnostics_a.q_real_min,
        operand_a_q_real_max=diagnostics_a.q_real_max,
        operand_a_q_imag_min=diagnostics_a.q_imag_min,
        operand_a_q_imag_max=diagnostics_a.q_imag_max,
        operand_b_q_real_min=diagnostics_b.q_real_min,
        operand_b_q_real_max=diagnostics_b.q_real_max,
        operand_b_q_imag_min=diagnostics_b.q_imag_min,
        operand_b_q_imag_max=diagnostics_b.q_imag_max,
        operand_a_zero_count=diagnostics_a.zero_count,
        operand_b_zero_count=diagnostics_b.zero_count,
        operand_a_clipping_count=diagnostics_a.clipping_count,
        operand_b_clipping_count=diagnostics_b.clipping_count,
        operand_a_boundary_saturation_count=diagnostics_a.boundary_saturation_count,
        operand_b_boundary_saturation_count=diagnostics_b.boundary_saturation_count,
        int32_theoretical_accumulator_bound=bounds.component_bound,
        int32_accumulator_safe=bounds.int32_safe,
        int64_theoretical_accumulator_bound=bounds.component_bound,
        int64_accumulator_safe=bounds.int64_safe,
        local_max_abs_error_vs_same_node_float32=local_metrics.max_abs_error,
        local_relative_l2_vs_same_node_float32=local_metrics.relative_l2,
        local_norm_drift_vs_same_node_float32=local_metrics.norm_drift,
        cumulative_max_abs_error_vs_same_node_float32=cumulative_metrics.max_abs_error,
        cumulative_relative_l2_vs_same_node_float32=cumulative_metrics.relative_l2,
        cumulative_norm_drift_vs_same_node_float32=cumulative_metrics.norm_drift,
        theoretical_local_error_bound=bound,
        ideal_local_max_abs_error=local_ideal_error,
        observed_local_error=observed_local_error,
        rounding_bound_applicable=rounding_bound_applicable,
        scale_computation_count=2,
        quantization_event_count=2,
        requantization_event_count=cumulative_requantizations,
        logical_encoded_bytes=logical_encoded_bytes,
        logical_float32_complex_bytes=logical_float32_bytes,
        nominal_operand_compression_ratio=(
            float(logical_float32_bytes / logical_encoded_bytes)
            if logical_encoded_bytes
            else float("inf")
        ),
        scale_metadata_bytes=2 * SCALE_METADATA_BYTES,
        integer_multiply_accumulate_count=4 * geometry.output_elements * geometry.K,
    )


def _theoretical_local_error_bound(
    left: np.ndarray,
    right: np.ndarray,
    left_encoded: QuantizedComplexTensor,
    right_encoded: QuantizedComplexTensor,
) -> tuple[float, bool]:
    left64 = np.asarray(left, dtype=np.complex128)
    right64 = np.asarray(right, dtype=np.complex128)
    left_hat = _dequantize_complex128(left_encoded)
    right_hat = _dequantize_complex128(right_encoded)
    left_error = left64 - left_hat
    right_error = right64 - right_hat
    clipping = (
        left_encoded.diagnostics.clipping_count > 0
        or right_encoded.diagnostics.clipping_count > 0
    )
    if clipping:
        left_error_bound = float(np.linalg.norm(left_error))
        right_error_bound = float(np.linalg.norm(right_error))
    else:
        left_error_bound = np.sqrt(left64.size / 2.0) * left_encoded.scale
        right_error_bound = np.sqrt(right64.size / 2.0) * right_encoded.scale
    bound = left_error_bound * float(np.linalg.norm(right64)) + float(
        np.linalg.norm(left_hat)
    ) * right_error_bound
    return float(bound), not clipping


def _complex128_contract(node: ContractNode, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_indices, right_indices, output_indices = _einsum_indices(node)
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.einsum(
            np.asarray(left, dtype=np.complex128),
            left_indices,
            np.asarray(right, dtype=np.complex128),
            right_indices,
            output_indices,
            dtype=np.complex128,
            optimize=False,
        )
    if not np.all(np.isfinite(result)):
        raise ValueError("complex128 local reference produced a nonfinite result")
    return np.asarray(result, dtype=np.complex128)


def _decode_products_complex128(
    products: IntegerContractionProducts, left_scale: float, right_scale: float
) -> np.ndarray:
    scale = _product_scale(left_scale, right_scale)
    _check_int64_subtract(products.rr, products.ii)
    _check_int64_add(products.ri, products.ir)
    with np.errstate(over="ignore", invalid="ignore"):
        real = np.subtract(products.rr, products.ii, dtype=np.int64).astype(np.float64) * scale
        imag = np.add(products.ri, products.ir, dtype=np.int64).astype(np.float64) * scale
    result = real + 1j * imag
    if not np.all(np.isfinite(result)):
        raise ValueError("ideal dequantized local result is nonfinite")
    return np.asarray(result, dtype=np.complex128)


def _error_metrics(actual: np.ndarray, expected: np.ndarray) -> ErrorMetrics:
    actual_state = np.asarray(actual).reshape(-1, order="F")
    expected_state = np.asarray(expected).reshape(-1, order="F")
    if actual_state.shape != expected_state.shape:
        raise ValueError("error metric operands must have matching shapes")
    difference = actual_state - expected_state
    max_abs = float(np.max(np.abs(difference), initial=0.0))
    actual_norm = float(np.linalg.norm(actual_state))
    expected_norm = float(np.linalg.norm(expected_state))
    relative_l2 = (
        float(np.linalg.norm(difference) / expected_norm)
        if expected_norm
        else float(np.linalg.norm(difference))
    )
    return ErrorMetrics(max_abs, relative_l2, abs(actual_norm - expected_norm))


def _readonly_copy(value: np.ndarray) -> np.ndarray:
    copied = np.array(value, copy=True, order="C")
    copied.setflags(write=False)
    return copied


def _readonly_array(value: np.ndarray) -> np.ndarray:
    return _readonly_copy(np.asarray(value))


def _readonly_array_mapping(value: Mapping[str, np.ndarray]) -> Mapping[str, np.ndarray]:
    copied = {key: _readonly_array(array) for key, array in value.items()}
    return MappingProxyType(copied)


def _readonly_complex64(real: np.ndarray, imag: np.ndarray) -> np.ndarray:
    if real.shape != imag.shape:
        raise ValueError("complex result planes must have equal shapes")
    if not np.all(np.isfinite(real)) or not np.all(np.isfinite(imag)):
        raise ValueError("complex64 result is nonfinite")
    f32_limit = np.finfo(np.float32).max
    if np.any(np.abs(real) > f32_limit) or np.any(np.abs(imag) > f32_limit):
        raise ValueError("complex result cannot be represented as finite complex64")
    output = np.empty(real.shape, dtype=np.complex64, order="C")
    with np.errstate(over="ignore", invalid="ignore"):
        output.real = np.asarray(real, dtype=np.float32)
        output.imag = np.asarray(imag, dtype=np.float32)
    if not np.all(np.isfinite(output)):
        raise ValueError("complex64 result is nonfinite")
    output.setflags(write=False)
    return output


def _to_complex64(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError("execution result must be numeric")
    with np.errstate(over="ignore", invalid="ignore"):
        real = np.asarray(array.real, dtype=np.float64)
        imag = np.asarray(array.imag, dtype=np.float64)
    return _readonly_complex64(real, imag)


def _dequantize_complex128(value: QuantizedComplexTensor) -> np.ndarray:
    with np.errstate(over="ignore", invalid="ignore"):
        real = np.asarray(value.q_real, dtype=np.float64) * value.scale
        imag = np.asarray(value.q_imag, dtype=np.float64) * value.scale
    result = real + 1j * imag
    if not np.all(np.isfinite(result)):
        raise ValueError("complex128 dequantization is nonfinite")
    return np.asarray(result, dtype=np.complex128)


def _plane_min(value: np.ndarray) -> float:
    return float(np.min(value)) if value.size else 0.0


def _plane_max(value: np.ndarray) -> float:
    return float(np.max(value)) if value.size else 0.0


def _plane_min_int(value: np.ndarray) -> int:
    return int(np.min(value)) if value.size else 0


def _plane_max_int(value: np.ndarray) -> int:
    return int(np.max(value)) if value.size else 0


def _max_abs(value: np.ndarray) -> float:
    return float(np.max(np.abs(value))) if value.size else 0.0


def _product(values: Any) -> int:
    result = 1
    for value in values:
        result *= int(value)
    return result


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a positive integer")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _nonnegative_integer(value: object, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a nonnegative integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return result


__all__ = [
    "AccumulatorBounds",
    "ContractionFacts",
    "ContractionGeometry",
    "ContractionTrace",
    "ErrorMetrics",
    "FLOAT32_COMPLEX_BYTES",
    "INT32_MAX",
    "INT64_MAX",
    "INT8_MAX",
    "INT8_MIN",
    "INT8_COMPLEX_BYTES",
    "IntegerContractionProducts",
    "POLICY_ID",
    "QuantizationDiagnostics",
    "QuantizedComplexTensor",
    "QuantizedDAGReplay",
    "SCALE_METADATA_BYTES",
    "accumulator_bounds",
    "bmnk",
    "contraction_dimensions",
    "contraction_geometry",
    "contract_complex_int8_reference",
    "contract_integer_products",
    "decode_integer_products",
    "dequantize_complex",
    "int32_accumulator_safe",
    "int64_accumulator_safe",
    "logical_policy_cost_facts",
    "numeric_policy_description",
    "policy_facts",
    "quantize_complex_shared_scale",
    "replay_quantized_dag",
    "theoretical_accumulator_bound",
]
