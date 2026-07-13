from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from quantum_bench.core.records import ContractionTask, JsonDict, TensorValue, to_jsonable
from quantum_bench.formats import (
    FixedPointConversionRecord,
    FixedPointSpec,
    conversion_error_metrics,
    quantize_fixed_point,
)
from quantum_bench.tn.contract import contract_binary_task


GENERIC_TASK_PREPARATION_SCHEMA_VERSION = "generic_task_preparation_v1"
GENERIC_TASK_ROUTE_ID = "generic_loop_fallback"
INT8_MAX_ABS_VALUE = 127
INT32_MAX_VALUE = (2**31) - 1
GENERIC_MODE_INT8_SCALED = "int8_scaled"
GENERIC_MODE_FLOAT32_NO_QUANT = "float32_no_quant"
GENERIC_KERNEL_STRATEGY = "mram_resident_output_tiled_v1"
GENERIC_OUTPUT_TILE_ELEMENTS = 256

GenericTaskPreparationStatus = Literal["prepared", "unsupported_shape", "failed"]
GenericQuantizationMode = Literal["per_task_input_quantize", "none"]


@dataclass(frozen=True)
class GenericTaskPreparationCaps:
    max_rank: int = 16
    max_tensor_elements: int = 65536
    max_contracted_combinations: int = 4096


@dataclass(frozen=True)
class GenericStructuralFeasibility:
    """Pure structural result shared by preparation and analysis callers."""

    feasible: bool
    metadata: JsonDict = field(default_factory=dict)
    rejection_reasons: tuple[str, ...] = ()

    @property
    def reason(self) -> str | None:
        return self.rejection_reasons[0] if self.rejection_reasons else None


@dataclass(frozen=True)
class GenericTaskPreparedOperands:
    left_quantized: np.ndarray
    right_quantized: np.ndarray
    expected_quantized_reference_output: np.ndarray
    full_precision_reference_output: np.ndarray
    left_operand: np.ndarray | None = None
    right_operand: np.ndarray | None = None
    expected_reference_output: np.ndarray | None = None
    operand_mode: str = GENERIC_MODE_INT8_SCALED


@dataclass(frozen=True)
class GenericTaskPreparationInput:
    task: ContractionTask
    left_tensor: TensorValue
    right_tensor: TensorValue
    fixed_point_spec: FixedPointSpec = field(default_factory=lambda: FixedPointSpec(route_dtype="int8", complex_policy="reject"))
    caps: GenericTaskPreparationCaps = field(default_factory=GenericTaskPreparationCaps)
    route_id: str = GENERIC_TASK_ROUTE_ID
    quantization_mode: GenericQuantizationMode = "per_task_input_quantize"


@dataclass(frozen=True)
class GenericTaskPreparationResult:
    schema_version: str
    route_id: str
    task_id: str
    status: GenericTaskPreparationStatus
    reason: str | None
    error: str | None
    input_tensor_ids: tuple[str, str]
    output_tensor_id: str
    left_labels: tuple[int, ...]
    right_labels: tuple[int, ...]
    contracted_labels: tuple[int, ...]
    output_labels: tuple[int, ...]
    input_shapes: tuple[tuple[int, ...], tuple[int, ...]]
    output_shape: tuple[int, ...]
    left_strides: tuple[int, ...]
    right_strides: tuple[int, ...]
    output_strides: tuple[int, ...]
    output_to_left_axes: tuple[int, ...]
    output_to_right_axes: tuple[int, ...]
    contracted_to_left_axes: tuple[int, ...]
    contracted_to_right_axes: tuple[int, ...]
    contracted_dims: tuple[int, ...]
    output_element_count: int
    contracted_combination_count: int
    fixed_point_spec: FixedPointSpec
    left_conversion: FixedPointConversionRecord | None
    right_conversion: FixedPointConversionRecord | None
    validation_metrics: JsonDict
    full_precision_error_metrics: JsonDict
    caps: GenericTaskPreparationCaps
    external_command_executed: bool
    execution_implemented: bool
    metadata: JsonDict = field(default_factory=dict)
    prepared_operands: GenericTaskPreparedOperands | None = field(default=None, repr=False, compare=False)

    def to_json_dict(self) -> JsonDict:
        return to_jsonable(
            {
                "schema_version": self.schema_version,
                "route_id": self.route_id,
                "task_id": self.task_id,
                "status": self.status,
                "reason": self.reason,
                "error": self.error,
                "input_tensor_ids": self.input_tensor_ids,
                "output_tensor_id": self.output_tensor_id,
                "left_labels": self.left_labels,
                "right_labels": self.right_labels,
                "contracted_labels": self.contracted_labels,
                "output_labels": self.output_labels,
                "input_shapes": self.input_shapes,
                "output_shape": self.output_shape,
                "native_index_metadata": {
                    "left_strides": self.left_strides,
                    "right_strides": self.right_strides,
                    "output_strides": self.output_strides,
                    "output_to_left_axes": self.output_to_left_axes,
                    "output_to_right_axes": self.output_to_right_axes,
                    "contracted_to_left_axes": self.contracted_to_left_axes,
                    "contracted_to_right_axes": self.contracted_to_right_axes,
                    "contracted_dims": self.contracted_dims,
                    "output_element_count": self.output_element_count,
                    "contracted_combination_count": self.contracted_combination_count,
                },
                "fixed_point_spec": self.fixed_point_spec,
                "conversion_records": {
                    "left": self.left_conversion,
                    "right": self.right_conversion,
                },
                "validation_metrics": self.validation_metrics,
                "full_precision_error_metrics": self.full_precision_error_metrics,
                "caps": self.caps,
                "external_command_executed": self.external_command_executed,
                "execution_implemented": self.execution_implemented,
                "metadata": self.metadata,
            }
        )


def prepare_generic_task(preparation: GenericTaskPreparationInput) -> GenericTaskPreparationResult:
    try:
        return _prepare_generic_task(preparation)
    except Exception as exc:  # pragma: no cover - defensive boundary
        return _base_result(preparation, status="failed", reason="host_preparation_exception", error=str(exc))


def generic_loop_reference_int32(
    left_quantized: np.ndarray,
    right_quantized: np.ndarray,
    *,
    output_shape: tuple[int, ...],
    left_strides: tuple[int, ...],
    right_strides: tuple[int, ...],
    output_strides: tuple[int, ...],
    output_to_left_axes: tuple[int, ...],
    output_to_right_axes: tuple[int, ...],
    contracted_to_left_axes: tuple[int, ...],
    contracted_to_right_axes: tuple[int, ...],
    contracted_dims: tuple[int, ...],
) -> np.ndarray:
    output = np.zeros(output_shape, dtype=np.int32)
    flat_left = np.asarray(left_quantized, dtype=np.int8).ravel()
    flat_right = np.asarray(right_quantized, dtype=np.int8).ravel()
    flat_output = output.ravel()
    output_element_count = int(output.size)
    contracted_count = _shape_product(contracted_dims)

    for output_linear in range(output_element_count):
        output_coords = _decode_index(output_linear, output_shape, output_strides)
        total = 0
        for contracted_linear in range(contracted_count):
            contracted_coords = _decode_index(contracted_linear, contracted_dims, _row_major_strides(contracted_dims))
            left_offset = _mapped_offset(output_coords, contracted_coords, output_to_left_axes, contracted_to_left_axes, left_strides)
            right_offset = _mapped_offset(output_coords, contracted_coords, output_to_right_axes, contracted_to_right_axes, right_strides)
            total += int(flat_left[left_offset]) * int(flat_right[right_offset])
        flat_output[output_linear] = total
    return output


def generic_loop_reference_float32(
    left: np.ndarray,
    right: np.ndarray,
    *,
    output_shape: tuple[int, ...],
    left_strides: tuple[int, ...],
    right_strides: tuple[int, ...],
    output_strides: tuple[int, ...],
    output_to_left_axes: tuple[int, ...],
    output_to_right_axes: tuple[int, ...],
    contracted_to_left_axes: tuple[int, ...],
    contracted_to_right_axes: tuple[int, ...],
    contracted_dims: tuple[int, ...],
) -> np.ndarray:
    output = np.zeros(output_shape, dtype=np.float32)
    flat_left = np.asarray(left, dtype=np.float32).ravel()
    flat_right = np.asarray(right, dtype=np.float32).ravel()
    flat_output = output.ravel()
    output_element_count = int(output.size)
    contracted_count = _shape_product(contracted_dims)

    for output_linear in range(output_element_count):
        output_coords = _decode_index(output_linear, output_shape, output_strides)
        total = np.float32(0.0)
        for contracted_linear in range(contracted_count):
            contracted_coords = _decode_index(contracted_linear, contracted_dims, _row_major_strides(contracted_dims))
            left_offset = _mapped_offset(output_coords, contracted_coords, output_to_left_axes, contracted_to_left_axes, left_strides)
            right_offset = _mapped_offset(output_coords, contracted_coords, output_to_right_axes, contracted_to_right_axes, right_strides)
            total = np.float32(total + np.float32(flat_left[left_offset] * flat_right[right_offset]))
        flat_output[output_linear] = total
    return output


def _prepare_generic_task(preparation: GenericTaskPreparationInput) -> GenericTaskPreparationResult:
    task = preparation.task
    mismatch = _validate_tensor_binding(task, preparation.left_tensor, preparation.right_tensor)
    if mismatch is not None:
        return _base_result(preparation, status="unsupported_shape", reason=mismatch)

    left_array = _real_array_or_none(np.asarray(preparation.left_tensor.array))
    right_array = _real_array_or_none(np.asarray(preparation.right_tensor.array))
    if left_array is None or right_array is None:
        return _base_result(preparation, status="unsupported_shape", reason="complex_generic_loop_not_implemented")
    if preparation.quantization_mode not in {"per_task_input_quantize", "none"}:
        return _base_result(preparation, status="unsupported_shape", reason=f"unsupported_quantization_mode:{preparation.quantization_mode}")
    if preparation.quantization_mode == "per_task_input_quantize" and preparation.fixed_point_spec.route_dtype != "int8":
        return _base_result(preparation, status="unsupported_shape", reason="unsupported_dtype")

    metadata_or_reason = _native_index_metadata(
        task,
        preparation.caps,
        check_int32_accumulation=preparation.quantization_mode == "per_task_input_quantize",
    )
    if isinstance(metadata_or_reason, str):
        return _base_result(preparation, status="unsupported_shape", reason=metadata_or_reason)
    metadata = metadata_or_reason

    if preparation.quantization_mode == "none":
        return _prepare_generic_float32_task(preparation, left_array, right_array, metadata)
    return _prepare_generic_int8_task(preparation, left_array, right_array, metadata)


def _prepare_generic_int8_task(
    preparation: GenericTaskPreparationInput,
    left_array: np.ndarray,
    right_array: np.ndarray,
    metadata: JsonDict,
) -> GenericTaskPreparationResult:
    task = preparation.task
    left_source_bytes = _float32_transfer_nbytes(preparation.left_tensor.array)
    right_source_bytes = _float32_transfer_nbytes(preparation.right_tensor.array)
    quantization_started = _perf_counter()
    left_converted = quantize_fixed_point(left_array, preparation.fixed_point_spec)
    right_converted = quantize_fixed_point(right_array, preparation.fixed_point_spec)
    quantization_time_s = _perf_counter() - quantization_started
    full_precision_reference = contract_binary_task(task, left_array, right_array)

    reference_started = _perf_counter()
    int32_reference = generic_loop_reference_int32(
        left_converted.array,
        right_converted.array,
        output_shape=task.output_shape,
        left_strides=metadata["left_strides"],
        right_strides=metadata["right_strides"],
        output_strides=metadata["output_strides"],
        output_to_left_axes=metadata["output_to_left_axes"],
        output_to_right_axes=metadata["output_to_right_axes"],
        contracted_to_left_axes=metadata["contracted_to_left_axes"],
        contracted_to_right_axes=metadata["contracted_to_right_axes"],
        contracted_dims=metadata["contracted_dims"],
    )
    int32_dequantized = int32_reference.astype(np.float64) * float(left_converted.record.scale) * float(right_converted.record.scale)
    dequantization_time_s = _perf_counter() - reference_started
    expected_quantized_reference = int32_dequantized
    validation = conversion_error_metrics(expected_quantized_reference, int32_dequantized)
    full_precision_error = conversion_error_metrics(full_precision_reference, expected_quantized_reference)

    if tuple(expected_quantized_reference.shape) != task.output_shape:
        return _base_result(preparation, status="unsupported_shape", reason="output_shape_mismatch")

    operands = GenericTaskPreparedOperands(
        left_quantized=left_converted.array,
        right_quantized=right_converted.array,
        expected_quantized_reference_output=expected_quantized_reference,
        full_precision_reference_output=full_precision_reference,
        left_operand=left_converted.array,
        right_operand=right_converted.array,
        expected_reference_output=expected_quantized_reference,
        operand_mode=GENERIC_MODE_INT8_SCALED,
    )
    return _base_result(
        preparation,
        status="prepared",
        reason=None,
        left_conversion=left_converted.record,
        right_conversion=right_converted.record,
        validation_metrics={
            "reference_kind": "expected_quantized_reference_vs_python_generic_loop_int32",
            "max_abs_error": validation.max_abs_error,
            "l2_error": validation.l2_error,
            "relative_l2_error": validation.relative_l2_error,
            "passed": validation.max_abs_error == 0.0,
        },
        full_precision_error_metrics={
            "reference_kind": "full_precision_vs_expected_quantized_reference",
            "max_abs_error": full_precision_error.max_abs_error,
            "l2_error": full_precision_error.l2_error,
            "relative_l2_error": full_precision_error.relative_l2_error,
        },
        metadata={
            "kernel_family": "generic_loop_fallback",
            "quantization_mode": "per_task_input_quantize",
            "operand_mode": GENERIC_MODE_INT8_SCALED,
            "input_dtype_on_dpu": "int8",
            "accumulator_dtype_on_dpu": "int32",
            "output_dtype_on_dpu": "int32",
            "unquantized_mode_kind": None,
            "scaling_applied": True,
            "validation_target": "expected_quantized_reference_output",
            "full_precision_reference_is_validation_target": False,
            "quantization_time_s": float(quantization_time_s),
            "dequantization_time_s": float(dequantization_time_s),
            "float32_reference_time_s": 0.0,
            "actual_h2d_bytes_model": int(left_converted.array.nbytes + right_converted.array.nbytes),
            "actual_d2h_bytes_model": int(int32_reference.nbytes),
            "full_precision_h2d_bytes_model": left_source_bytes + right_source_bytes,
            "full_precision_d2h_bytes_model": _float32_transfer_nbytes(full_precision_reference),
            **metadata,
        },
        prepared_operands=operands,
    )


def _prepare_generic_float32_task(
    preparation: GenericTaskPreparationInput,
    left_array: np.ndarray,
    right_array: np.ndarray,
    metadata: JsonDict,
) -> GenericTaskPreparationResult:
    task = preparation.task
    left_source_bytes = _float32_transfer_nbytes(preparation.left_tensor.array)
    right_source_bytes = _float32_transfer_nbytes(preparation.right_tensor.array)
    left_operand = np.asarray(left_array, dtype=np.float32)
    right_operand = np.asarray(right_array, dtype=np.float32)
    full_precision_reference = contract_binary_task(task, left_array, right_array)
    reference_started = _perf_counter()
    float32_reference = generic_loop_reference_float32(
        left_operand,
        right_operand,
        output_shape=task.output_shape,
        left_strides=metadata["left_strides"],
        right_strides=metadata["right_strides"],
        output_strides=metadata["output_strides"],
        output_to_left_axes=metadata["output_to_left_axes"],
        output_to_right_axes=metadata["output_to_right_axes"],
        contracted_to_left_axes=metadata["contracted_to_left_axes"],
        contracted_to_right_axes=metadata["contracted_to_right_axes"],
        contracted_dims=metadata["contracted_dims"],
    )
    float32_reference_time_s = _perf_counter() - reference_started
    validation = conversion_error_metrics(float32_reference, float32_reference)
    full_precision_error = conversion_error_metrics(full_precision_reference, float32_reference)
    if tuple(float32_reference.shape) != task.output_shape:
        return _base_result(preparation, status="unsupported_shape", reason="output_shape_mismatch")

    operands = GenericTaskPreparedOperands(
        left_quantized=left_operand,
        right_quantized=right_operand,
        expected_quantized_reference_output=float32_reference,
        full_precision_reference_output=full_precision_reference,
        left_operand=left_operand,
        right_operand=right_operand,
        expected_reference_output=float32_reference,
        operand_mode=GENERIC_MODE_FLOAT32_NO_QUANT,
    )
    return _base_result(
        preparation,
        status="prepared",
        reason=None,
        left_conversion=None,
        right_conversion=None,
        validation_metrics={
            "reference_kind": "expected_float32_reference_vs_python_generic_loop_float32",
            "max_abs_error": validation.max_abs_error,
            "l2_error": validation.l2_error,
            "relative_l2_error": validation.relative_l2_error,
            "passed": validation.max_abs_error == 0.0,
        },
        full_precision_error_metrics={
            "reference_kind": "full_precision_vs_expected_float32_reference",
            "max_abs_error": full_precision_error.max_abs_error,
            "l2_error": full_precision_error.l2_error,
            "relative_l2_error": full_precision_error.relative_l2_error,
        },
        metadata={
            "kernel_family": "generic_loop_fallback",
            "quantization_mode": "none",
            "operand_mode": GENERIC_MODE_FLOAT32_NO_QUANT,
            "input_dtype_on_dpu": "float32",
            "accumulator_dtype_on_dpu": "float32",
            "output_dtype_on_dpu": "float32",
            "unquantized_mode_kind": GENERIC_MODE_FLOAT32_NO_QUANT,
            "scaling_applied": False,
            "validation_target": "expected_float32_reference_output",
            "full_precision_reference_is_validation_target": False,
            "quantization_time_s": 0.0,
            "dequantization_time_s": 0.0,
            "float32_reference_time_s": float(float32_reference_time_s),
            "actual_h2d_bytes_model": int(left_operand.nbytes + right_operand.nbytes),
            "actual_d2h_bytes_model": int(float32_reference.nbytes),
            "full_precision_h2d_bytes_model": left_source_bytes + right_source_bytes,
            "full_precision_d2h_bytes_model": _float32_transfer_nbytes(full_precision_reference),
            **metadata,
        },
        prepared_operands=operands,
    )


def _base_result(
    preparation: GenericTaskPreparationInput,
    *,
    status: GenericTaskPreparationStatus,
    reason: str | None,
    error: str | None = None,
    left_conversion: FixedPointConversionRecord | None = None,
    right_conversion: FixedPointConversionRecord | None = None,
    validation_metrics: JsonDict | None = None,
    full_precision_error_metrics: JsonDict | None = None,
    metadata: JsonDict | None = None,
    prepared_operands: GenericTaskPreparedOperands | None = None,
) -> GenericTaskPreparationResult:
    task = preparation.task
    native_metadata = metadata or {}
    return GenericTaskPreparationResult(
        schema_version=GENERIC_TASK_PREPARATION_SCHEMA_VERSION,
        route_id=preparation.route_id,
        task_id=task.id,
        status=status,
        reason=reason,
        error=error,
        input_tensor_ids=task.input_tensor_ids,
        output_tensor_id=task.output_tensor_id,
        left_labels=task.left_labels,
        right_labels=task.right_labels,
        contracted_labels=task.contracted_labels,
        output_labels=task.output_labels,
        input_shapes=task.input_shapes,
        output_shape=task.output_shape,
        left_strides=tuple(native_metadata.get("left_strides", ())),
        right_strides=tuple(native_metadata.get("right_strides", ())),
        output_strides=tuple(native_metadata.get("output_strides", ())),
        output_to_left_axes=tuple(native_metadata.get("output_to_left_axes", ())),
        output_to_right_axes=tuple(native_metadata.get("output_to_right_axes", ())),
        contracted_to_left_axes=tuple(native_metadata.get("contracted_to_left_axes", ())),
        contracted_to_right_axes=tuple(native_metadata.get("contracted_to_right_axes", ())),
        contracted_dims=tuple(native_metadata.get("contracted_dims", ())),
        output_element_count=int(native_metadata.get("output_element_count", 0) or 0),
        contracted_combination_count=int(native_metadata.get("contracted_combination_count", 0) or 0),
        fixed_point_spec=preparation.fixed_point_spec,
        left_conversion=left_conversion,
        right_conversion=right_conversion,
        validation_metrics=validation_metrics or {},
        full_precision_error_metrics=full_precision_error_metrics or {},
        caps=preparation.caps,
        external_command_executed=False,
        execution_implemented=False,
        metadata=native_metadata,
        prepared_operands=prepared_operands,
    )


def _validate_tensor_binding(task: ContractionTask, left_tensor: TensorValue, right_tensor: TensorValue) -> str | None:
    if tuple(task.input_tensor_ids) != (left_tensor.spec.id, right_tensor.spec.id):
        return "tensor_id_mismatch"
    if tuple(left_tensor.spec.labels) != tuple(task.left_labels) or tuple(right_tensor.spec.labels) != tuple(task.right_labels):
        return "label_mismatch"
    if tuple(np.asarray(left_tensor.array).shape) != tuple(task.input_shapes[0]):
        return "left_shape_mismatch"
    if tuple(np.asarray(right_tensor.array).shape) != tuple(task.input_shapes[1]):
        return "right_shape_mismatch"
    return None


def _real_array_or_none(array: np.ndarray) -> np.ndarray | None:
    if np.iscomplexobj(array):
        if np.any(np.abs(array.imag) > 0.0):
            return None
        return array.real.astype(np.float64, copy=False)
    return array


def generic_structural_feasibility(
    task: ContractionTask,
    caps: GenericTaskPreparationCaps = GenericTaskPreparationCaps(),
    *,
    check_int32_accumulation: bool = True,
) -> GenericStructuralFeasibility:
    """Return generic-loop shape metadata without touching tensor values.

    The validation order and reason strings are the generic preparation contract.
    Keeping this function value-free lets planners use the same rejection rules
    without importing or executing the preparation path.
    """
    left_shape = tuple(int(dim) for dim in task.input_shapes[0])
    right_shape = tuple(int(dim) for dim in task.input_shapes[1])
    output_shape = tuple(int(dim) for dim in task.output_shape)
    if max(len(left_shape), len(right_shape), len(output_shape), len(task.contracted_labels)) > caps.max_rank:
        return GenericStructuralFeasibility(False, rejection_reasons=("rank_cap_exceeded",))
    if max(_shape_product(left_shape), _shape_product(right_shape), _shape_product(output_shape)) > caps.max_tensor_elements:
        return GenericStructuralFeasibility(False, rejection_reasons=("element_count_cap_exceeded",))

    try:
        output_to_left_axes = tuple(task.left_labels.index(label) if label in task.left_labels else -1 for label in task.output_labels)
        output_to_right_axes = tuple(task.right_labels.index(label) if label in task.right_labels else -1 for label in task.output_labels)
        contracted_to_left_axes = tuple(task.left_labels.index(label) for label in task.contracted_labels)
        contracted_to_right_axes = tuple(task.right_labels.index(label) for label in task.contracted_labels)
        contracted_dims = tuple(left_shape[axis] for axis in contracted_to_left_axes)
    except ValueError:
        return GenericStructuralFeasibility(False, rejection_reasons=("label_mapping_invalid",))
    for label, left_axis, right_axis in zip(task.contracted_labels, contracted_to_left_axes, contracted_to_right_axes):
        if left_shape[left_axis] != right_shape[right_axis]:
            return GenericStructuralFeasibility(False, rejection_reasons=("label_mapping_invalid",))
    if any(left_axis < 0 and right_axis < 0 for left_axis, right_axis in zip(output_to_left_axes, output_to_right_axes)):
        return GenericStructuralFeasibility(False, rejection_reasons=("label_mapping_invalid",))

    contracted_count = _shape_product(contracted_dims)
    if contracted_count > caps.max_contracted_combinations:
        return GenericStructuralFeasibility(False, rejection_reasons=("contracted_combination_cap_exceeded",))
    if check_int32_accumulation and contracted_count * INT8_MAX_ABS_VALUE * INT8_MAX_ABS_VALUE > INT32_MAX_VALUE:
        return GenericStructuralFeasibility(False, rejection_reasons=("int32_accumulation_overflow_risk",))

    output_element_count = _shape_product(output_shape)
    output_tile_count = (output_element_count + GENERIC_OUTPUT_TILE_ELEMENTS - 1) // GENERIC_OUTPUT_TILE_ELEMENTS
    return GenericStructuralFeasibility(
        True,
        metadata={
            "left_strides": _row_major_strides(left_shape),
            "right_strides": _row_major_strides(right_shape),
            "output_strides": _row_major_strides(output_shape),
            "output_to_left_axes": output_to_left_axes,
            "output_to_right_axes": output_to_right_axes,
            "contracted_to_left_axes": contracted_to_left_axes,
            "contracted_to_right_axes": contracted_to_right_axes,
            "contracted_dims": contracted_dims,
            "output_element_count": output_element_count,
            "contracted_combination_count": contracted_count,
            "generic_kernel_strategy": GENERIC_KERNEL_STRATEGY,
            "native_max_rank": caps.max_rank,
            "native_max_tensor_elements": caps.max_tensor_elements,
            "generic_output_tile_elements": GENERIC_OUTPUT_TILE_ELEMENTS,
            "generic_output_tile_count": output_tile_count,
            "mram_resident_operands": True,
            "wram_output_tiled": True,
            "mram_tiled_task_count": int(output_tile_count > 1),
            "mram_read_bytes_model": int(output_element_count * contracted_count * 2 * 8),
            "mram_write_bytes_model": int((output_element_count * 4 + 7) & ~7),
        },
    )


def _native_index_metadata(task: ContractionTask, caps: GenericTaskPreparationCaps, *, check_int32_accumulation: bool = True) -> JsonDict | str:
    feasibility = generic_structural_feasibility(
        task,
        caps,
        check_int32_accumulation=check_int32_accumulation,
    )
    return feasibility.metadata if feasibility.feasible else (feasibility.reason or "generic_structural_rejection")


def _row_major_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    if not shape:
        return ()
    strides: list[int] = []
    product = 1
    for dim in reversed(shape):
        strides.insert(0, product)
        product *= int(dim)
    return tuple(strides)


def _shape_product(shape: tuple[int, ...]) -> int:
    product = 1
    for dim in shape:
        product *= int(dim)
    return int(product)


def _float32_transfer_nbytes(array: object) -> int:
    """Model native float32 transfer without casting away complex components."""
    value = np.asarray(array)
    component_count = 2 if np.iscomplexobj(value) else 1
    return int(value.size * component_count * np.dtype(np.float32).itemsize)


def _decode_index(linear: int, shape: tuple[int, ...], strides: tuple[int, ...]) -> tuple[int, ...]:
    if not shape:
        return ()
    remaining = int(linear)
    coords: list[int] = []
    for dim, stride in zip(shape, strides):
        coord = remaining // int(stride)
        coords.append(int(coord))
        remaining -= coord * int(stride)
    return tuple(coords)


def _mapped_offset(
    output_coords: tuple[int, ...],
    contracted_coords: tuple[int, ...],
    output_axis_map: tuple[int, ...],
    contracted_axis_map: tuple[int, ...],
    strides: tuple[int, ...],
) -> int:
    offset = 0
    for output_axis, tensor_axis in enumerate(output_axis_map):
        if tensor_axis >= 0:
            offset += int(output_coords[output_axis]) * int(strides[tensor_axis])
    for contracted_axis, tensor_axis in enumerate(contracted_axis_map):
        offset += int(contracted_coords[contracted_axis]) * int(strides[tensor_axis])
    return int(offset)


def _perf_counter() -> float:
    # Local wrapper keeps import surface stable for tests that monkeypatch time elsewhere.
    import time

    return time.perf_counter()
