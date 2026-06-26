from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from quantum_bench.core.records import ContractionTask, JsonDict, TensorValue, to_jsonable
from quantum_bench.formats import (
    FixedPointConversionRecord,
    FixedPointSpec,
    conversion_error_metrics,
    dequantize_fixed_point,
    quantize_fixed_point,
)
from quantum_bench.targets.upmem import (
    REQUIRES_TILING_NOT_IMPLEMENTED,
    UPMEM_DENSE_ESTIMATE_KEY,
    SimplePimProbeResult,
    UpmemTaskEstimate,
    estimate_dense_task,
    probe_simplepim,
)


DENSE_TASK_PREPARATION_SCHEMA_VERSION = "dense_task_preparation_v1"
DENSE_TASK_ROUTE_ID = "dense_gemm"

DenseTaskPreparationStatus = Literal[
    "prepared",
    "skipped",
    "unsupported_shape",
    "requires_executable_tiling_not_implemented",
    "simplepim_unavailable",
    "failed",
]


@dataclass(frozen=True)
class DenseTaskValidationMetrics:
    reference_check_max_abs_error: float
    reference_check_l2_error: float
    reference_check_relative_l2_error: float | None
    dequantized_output_max_abs_error: float
    dequantized_output_l2_error: float
    dequantized_output_relative_l2_error: float | None
    reference_output_norm: float
    dequantized_output_norm: float
    passed_reference_shape_check: bool


@dataclass(frozen=True)
class DenseTaskPreparedOperands:
    left_matrix: np.ndarray
    right_matrix: np.ndarray
    left_quantized: np.ndarray
    right_quantized: np.ndarray
    left_dequantized: np.ndarray
    right_dequantized: np.ndarray
    reference_output: np.ndarray
    dequantized_output: np.ndarray
    output_labels: tuple[int, ...]


@dataclass(frozen=True)
class DenseTaskPreparationInput:
    task: ContractionTask
    left_tensor: TensorValue
    right_tensor: TensorValue
    fixed_point_spec: FixedPointSpec = field(
        default_factory=lambda: FixedPointSpec(route_dtype="int8", complex_policy="split_real_imag_last_axis")
    )
    simplepim_probe: SimplePimProbeResult | None = None
    route_id: str = DENSE_TASK_ROUTE_ID
    target_estimate_key: str = UPMEM_DENSE_ESTIMATE_KEY


@dataclass(frozen=True)
class DenseTaskPreparationResult:
    schema_version: str
    route_id: str
    task_id: str
    status: DenseTaskPreparationStatus
    reason: str | None
    error: str | None
    input_tensor_ids: tuple[str, str]
    output_tensor_id: str
    left_labels: tuple[int, ...]
    right_labels: tuple[int, ...]
    contracted_labels: tuple[int, ...]
    ordered_contracted_labels: tuple[int, ...]
    left_free_labels: tuple[int, ...]
    right_free_labels: tuple[int, ...]
    gemm_output_labels: tuple[int, ...]
    output_labels: tuple[int, ...]
    input_shapes: tuple[tuple[int, ...], tuple[int, ...]]
    output_shape: tuple[int, ...]
    gemm_m: int
    gemm_k: int
    gemm_n: int
    left_matrix_shape: tuple[int, int] | None
    right_matrix_shape: tuple[int, int] | None
    fixed_point_spec: FixedPointSpec
    left_conversion: FixedPointConversionRecord | None
    right_conversion: FixedPointConversionRecord | None
    simplepim_probe: JsonDict
    tile_plan: JsonDict | None
    upmem_task_estimate: JsonDict | None
    validation_metrics: DenseTaskValidationMetrics | None
    external_command_executed: bool
    execution_implemented: bool
    metadata: JsonDict = field(default_factory=dict)
    prepared_operands: DenseTaskPreparedOperands | None = None

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
                "ordered_contracted_labels": self.ordered_contracted_labels,
                "left_free_labels": self.left_free_labels,
                "right_free_labels": self.right_free_labels,
                "gemm_output_labels": self.gemm_output_labels,
                "output_labels": self.output_labels,
                "input_shapes": self.input_shapes,
                "output_shape": self.output_shape,
                "gemm_m": self.gemm_m,
                "gemm_k": self.gemm_k,
                "gemm_n": self.gemm_n,
                "left_matrix_shape": self.left_matrix_shape,
                "right_matrix_shape": self.right_matrix_shape,
                "fixed_point_spec": self.fixed_point_spec,
                "conversion_records": {
                    "left": self.left_conversion,
                    "right": self.right_conversion,
                },
                "simplepim_probe": self.simplepim_probe,
                "tile_plan": self.tile_plan,
                "upmem_task_estimate": self.upmem_task_estimate,
                "validation_metrics": self.validation_metrics,
                "external_command_executed": self.external_command_executed,
                "execution_implemented": self.execution_implemented,
                "metadata": self.metadata,
            }
        )


@dataclass(frozen=True)
class _DenseLowering:
    left_matrix: np.ndarray
    right_matrix: np.ndarray
    reference_output: np.ndarray
    output_from_gemm: np.ndarray
    ordered_contracted_labels: tuple[int, ...]
    left_free_labels: tuple[int, ...]
    right_free_labels: tuple[int, ...]
    gemm_output_labels: tuple[int, ...]


def prepare_dense_task(preparation: DenseTaskPreparationInput) -> DenseTaskPreparationResult:
    probe = preparation.simplepim_probe or probe_simplepim()
    try:
        return _prepare_dense_task(preparation, probe)
    except Exception as exc:  # pragma: no cover - defensive boundary
        return _base_result(
            preparation=preparation,
            probe=probe,
            status="failed",
            reason="host_preparation_exception",
            error=str(exc),
        )


def _prepare_dense_task(
    preparation: DenseTaskPreparationInput,
    probe: SimplePimProbeResult,
) -> DenseTaskPreparationResult:
    mismatch_reason = _validate_tensor_binding(preparation.task, preparation.left_tensor, preparation.right_tensor)
    if mismatch_reason is not None:
        return _base_result(
            preparation=preparation,
            probe=probe,
            status="unsupported_shape",
            reason=mismatch_reason,
        )

    estimate = estimate_dense_task(preparation.task)
    if not estimate.supported:
        return _base_result(
            preparation=preparation,
            probe=probe,
            status="unsupported_shape",
            reason=estimate.reject_reason or "unsupported_dense_gemm_shape",
            estimate=estimate,
        )

    try:
        lowering = _lower_task_to_gemm(preparation.task, preparation.left_tensor, preparation.right_tensor)
    except ValueError as exc:
        return _base_result(
            preparation=preparation,
            probe=probe,
            status="unsupported_shape",
            reason=str(exc),
            estimate=estimate,
        )
    left_converted = quantize_fixed_point(lowering.left_matrix, preparation.fixed_point_spec)
    right_converted = quantize_fixed_point(lowering.right_matrix, preparation.fixed_point_spec)
    left_dequantized = dequantize_fixed_point(left_converted)
    right_dequantized = dequantize_fixed_point(right_converted)
    dequantized_matrix = left_dequantized @ right_dequantized
    dequantized_output = _restore_output_order(
        dequantized_matrix,
        lowering.left_free_labels,
        lowering.right_free_labels,
        preparation.task.output_labels,
        preparation.task.output_shape,
    )
    validation_metrics = _validation_metrics(lowering.reference_output, lowering.output_from_gemm, dequantized_output)

    prepared_operands = DenseTaskPreparedOperands(
        left_matrix=lowering.left_matrix,
        right_matrix=lowering.right_matrix,
        left_quantized=left_converted.array,
        right_quantized=right_converted.array,
        left_dequantized=left_dequantized,
        right_dequantized=right_dequantized,
        reference_output=lowering.reference_output,
        dequantized_output=dequantized_output,
        output_labels=preparation.task.output_labels,
    )

    status: DenseTaskPreparationStatus
    reason: str | None
    if estimate.requires_tiling or estimate.tile_plan.requires_host_aggregation:
        status = "requires_executable_tiling_not_implemented"
        reason = estimate.reject_reason or REQUIRES_TILING_NOT_IMPLEMENTED
        if estimate.tile_plan.requires_host_aggregation:
            reason = "requires_host_aggregation_not_implemented"
    elif not _simplepim_ready(probe):
        status = "simplepim_unavailable"
        reason = probe.skip_reason or probe.simplepim_probe_status
    else:
        status = "prepared"
        reason = None

    return _base_result(
        preparation=preparation,
        probe=probe,
        status=status,
        reason=reason,
        estimate=estimate,
        lowering=lowering,
        left_conversion=left_converted.record,
        right_conversion=right_converted.record,
        validation_metrics=validation_metrics,
        prepared_operands=prepared_operands,
    )


def _base_result(
    preparation: DenseTaskPreparationInput,
    probe: SimplePimProbeResult,
    status: DenseTaskPreparationStatus,
    reason: str | None,
    error: str | None = None,
    estimate: UpmemTaskEstimate | None = None,
    lowering: _DenseLowering | None = None,
    left_conversion: FixedPointConversionRecord | None = None,
    right_conversion: FixedPointConversionRecord | None = None,
    validation_metrics: DenseTaskValidationMetrics | None = None,
    prepared_operands: DenseTaskPreparedOperands | None = None,
) -> DenseTaskPreparationResult:
    task = preparation.task
    tile_plan = estimate.tile_plan.as_summary() if estimate is not None else None
    task_estimate = estimate.as_task_estimate() if estimate is not None else None
    return DenseTaskPreparationResult(
        schema_version=DENSE_TASK_PREPARATION_SCHEMA_VERSION,
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
        ordered_contracted_labels=lowering.ordered_contracted_labels if lowering else (),
        left_free_labels=lowering.left_free_labels if lowering else (),
        right_free_labels=lowering.right_free_labels if lowering else (),
        gemm_output_labels=lowering.gemm_output_labels if lowering else (),
        output_labels=task.output_labels,
        input_shapes=task.input_shapes,
        output_shape=task.output_shape,
        gemm_m=task.gemm_m,
        gemm_k=task.gemm_k,
        gemm_n=task.gemm_n,
        left_matrix_shape=tuple(int(dim) for dim in lowering.left_matrix.shape) if lowering else None,
        right_matrix_shape=tuple(int(dim) for dim in lowering.right_matrix.shape) if lowering else None,
        fixed_point_spec=preparation.fixed_point_spec,
        left_conversion=left_conversion,
        right_conversion=right_conversion,
        simplepim_probe=probe.to_json_dict(),
        tile_plan=tile_plan,
        upmem_task_estimate=task_estimate,
        validation_metrics=validation_metrics,
        external_command_executed=False,
        execution_implemented=False,
        metadata={
            "target_estimate_key": preparation.target_estimate_key,
            "prepared_means": "host-side dense-route preparation succeeded; no SimplePIM execution happened",
            "complex_fixed_point_policy": preparation.fixed_point_spec.complex_policy,
            "complex_kernel_mapping_implemented": False,
            "status_priority": (
                "failed > unsupported_shape > requires_executable_tiling_not_implemented > "
                "simplepim_unavailable > prepared"
            ),
        },
        prepared_operands=prepared_operands,
    )


def _validate_tensor_binding(task: ContractionTask, left_tensor: TensorValue, right_tensor: TensorValue) -> str | None:
    if tuple(task.input_tensor_ids) != (left_tensor.spec.id, right_tensor.spec.id):
        return "tensor_id_mismatch"
    if tuple(left_tensor.spec.labels) != tuple(task.left_labels):
        return "left_label_mismatch"
    if tuple(right_tensor.spec.labels) != tuple(task.right_labels):
        return "right_label_mismatch"
    if tuple(left_tensor.spec.shape) != tuple(task.input_shapes[0]):
        return "left_shape_mismatch"
    if tuple(right_tensor.spec.shape) != tuple(task.input_shapes[1]):
        return "right_shape_mismatch"
    if np.asarray(left_tensor.array).shape != tuple(left_tensor.spec.shape):
        return "left_array_shape_mismatch"
    if np.asarray(right_tensor.array).shape != tuple(right_tensor.spec.shape):
        return "right_array_shape_mismatch"
    if len(tuple(task.left_labels)) != len(set(task.left_labels)):
        return "left_labels_not_unique"
    if len(tuple(task.right_labels)) != len(set(task.right_labels)):
        return "right_labels_not_unique"
    if len(tuple(task.output_labels)) != len(set(task.output_labels)):
        return "output_labels_not_unique"
    for label in task.contracted_labels:
        if label not in task.left_labels or label not in task.right_labels:
            return "contracted_label_missing_from_input"
        left_dim = task.input_shapes[0][task.left_labels.index(label)]
        right_dim = task.input_shapes[1][task.right_labels.index(label)]
        if left_dim != right_dim:
            return "contracted_label_dimension_mismatch"
    expected_output_labels = tuple(label for label in task.left_labels if label not in task.contracted_labels) + tuple(
        label for label in task.right_labels if label not in task.contracted_labels
    )
    if sorted(expected_output_labels) != sorted(task.output_labels):
        return "output_label_mismatch"
    return None


def _lower_task_to_gemm(task: ContractionTask, left_tensor: TensorValue, right_tensor: TensorValue) -> _DenseLowering:
    contracted_labels = tuple(task.contracted_labels)
    left_free = tuple(label for label in task.left_labels if label not in contracted_labels)
    right_free = tuple(label for label in task.right_labels if label not in contracted_labels)
    contracted = tuple(label for label in task.left_labels if label in contracted_labels)

    for label in contracted:
        if label not in task.right_labels:
            raise ValueError(f"Contracted label {label} is missing from right tensor")

    left_order = left_free + contracted
    right_order = contracted + right_free
    left_array = np.asarray(left_tensor.array)
    right_array = np.asarray(right_tensor.array)
    left_reordered = _transpose_to_labels(left_array, task.left_labels, left_order)
    right_reordered = _transpose_to_labels(right_array, task.right_labels, right_order)

    left_matrix = left_reordered.reshape((task.gemm_m, task.gemm_k))
    right_matrix = right_reordered.reshape((task.gemm_k, task.gemm_n))
    reference_output = np.einsum(task.index_expression, left_array, right_array, optimize=False)
    gemm_output_matrix = left_matrix @ right_matrix
    output_from_gemm = _restore_output_order(
        gemm_output_matrix,
        left_free,
        right_free,
        task.output_labels,
        task.output_shape,
    )
    return _DenseLowering(
        left_matrix=left_matrix,
        right_matrix=right_matrix,
        reference_output=reference_output,
        output_from_gemm=output_from_gemm,
        ordered_contracted_labels=contracted,
        left_free_labels=left_free,
        right_free_labels=right_free,
        gemm_output_labels=left_free + right_free,
    )


def _restore_output_order(
    matrix_output: np.ndarray,
    left_free: tuple[int, ...],
    right_free: tuple[int, ...],
    output_labels: tuple[int, ...],
    output_shape: tuple[int, ...],
) -> np.ndarray:
    gemm_labels = left_free + right_free
    gemm_shape = _shape_for_labels(output_labels, output_shape, gemm_labels)
    tensor_output = np.asarray(matrix_output).reshape(gemm_shape)
    return _transpose_to_labels(tensor_output, gemm_labels, output_labels)


def _shape_for_labels(
    output_labels: tuple[int, ...],
    output_shape: tuple[int, ...],
    desired_labels: tuple[int, ...],
) -> tuple[int, ...]:
    return tuple(int(output_shape[output_labels.index(label)]) for label in desired_labels)


def _transpose_to_labels(array: np.ndarray, current_labels: tuple[int, ...], desired_labels: tuple[int, ...]) -> np.ndarray:
    if tuple(current_labels) != tuple(desired_labels):
        axes = tuple(current_labels.index(label) for label in desired_labels)
        return np.transpose(array, axes)
    return array


def _validation_metrics(
    reference_output: np.ndarray,
    output_from_gemm: np.ndarray,
    dequantized_output: np.ndarray,
) -> DenseTaskValidationMetrics:
    reference_check = conversion_error_metrics(reference_output, output_from_gemm)
    dequantized_check = conversion_error_metrics(reference_output, dequantized_output)
    return DenseTaskValidationMetrics(
        reference_check_max_abs_error=reference_check.max_abs_error,
        reference_check_l2_error=reference_check.l2_error,
        reference_check_relative_l2_error=reference_check.relative_l2_error,
        dequantized_output_max_abs_error=dequantized_check.max_abs_error,
        dequantized_output_l2_error=dequantized_check.l2_error,
        dequantized_output_relative_l2_error=dequantized_check.relative_l2_error,
        reference_output_norm=float(np.linalg.norm(np.asarray(reference_output).ravel())),
        dequantized_output_norm=float(np.linalg.norm(np.asarray(dequantized_output).ravel())),
        passed_reference_shape_check=tuple(reference_output.shape) == tuple(dequantized_output.shape),
    )


def _simplepim_ready(probe: SimplePimProbeResult) -> bool:
    return probe.simplepim_available and probe.simplepim_probe_status == "available"
