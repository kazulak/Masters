from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

from quantum_bench.core.records import ContractionTask, JsonDict, to_jsonable
from quantum_bench.targets.upmem.evidence import (
    CONTRACTION_EXECUTION_TARGET_UPMEM,
    UPMEM_EXECUTION_MODE_SDK_SIMULATOR,
)


UPMEM_TASKGRAPH_RUNTIME_SCHEMA_VERSION = "upmem_taskgraph_runtime_v1"
UPMEM_TASKGRAPH_TASK_METRIC_SCHEMA_VERSION = "upmem_taskgraph_task_metric_v1"
GENERIC_QUANTIZED_TASKGRAPH_REFERENCE_SCHEMA_VERSION = "generic_quantized_taskgraph_reference_v1"
GENERIC_QUANTIZED_TASKGRAPH_REFERENCE_TASK_SCHEMA_VERSION = "generic_quantized_taskgraph_reference_task_v1"

GENERIC_KERNEL_STRATEGY = "mram_resident_output_tiled_v1"
GENERIC_NATIVE_MAX_RANK = 16
GENERIC_NATIVE_MAX_TENSOR_ELEMENTS = 65536
GENERIC_OUTPUT_TILE_ELEMENTS = 256
APPLICATION_VISIBLE_SDK_TRANSFER_SCOPE = "application_visible_sdk_recorded"
APPLICATION_VISIBLE_PREPARATION_TRANSFER_SCOPE = "application_visible_preparation_model"

UpmemTaskGraphStatus = Literal["completed", "unsupported", "failed", "validation_failed"]

CONTRACTION_EXECUTION_TARGET = CONTRACTION_EXECUTION_TARGET_UPMEM
UPMEM_EXECUTION_MODE = UPMEM_EXECUTION_MODE_SDK_SIMULATOR


def transfer_accounting(
    actual_h2d_bytes: int | None,
    actual_d2h_bytes: int | None,
    *,
    declared_total_bytes: int | None = None,
    recorded_by_sdk: bool = True,
) -> JsonDict:
    """Return a checked application-visible transfer accounting record.

    SDK simulator artifacts do not expose physical DIMM/bus counters. Values
    are therefore deliberately scoped to application-visible byte records or
    preparation models, never presented as physical hardware traffic.
    """

    h2d = int(actual_h2d_bytes or 0)
    d2h = int(actual_d2h_bytes or 0)
    if h2d < 0 or d2h < 0:
        raise ValueError("actual transfer byte counts must be nonnegative")
    total = h2d + d2h
    if declared_total_bytes is not None and int(declared_total_bytes) != total:
        raise ValueError(
            f"actual_transfer_bytes invariant failed: {declared_total_bytes} != {h2d} + {d2h}"
        )
    return {
        "actual_h2d_bytes": h2d,
        "actual_d2h_bytes": d2h,
        "actual_transfer_bytes": total,
        "actual_transfer_bytes_invariant": "passed",
        "transfer_accounting_scope": (
            APPLICATION_VISIBLE_SDK_TRANSFER_SCOPE
            if recorded_by_sdk
            else APPLICATION_VISIBLE_PREPARATION_TRANSFER_SCOPE
        ),
        "physical_bus_bytes_available": False,
        "transfer_components": {
            "h2d_application_visible_payload_bytes": h2d,
            "d2h_application_visible_payload_bytes": d2h,
            "quantization_scale_bytes": None,
            "shape_index_metadata_bytes": None,
            "control_structure_bytes": None,
            "alignment_padding_bytes": None,
            "unobserved_sdk_overhead_bytes": None,
        },
    }


@dataclass(frozen=True)
class UpmemTaskGraphRuntimeResult:
    schema_version: str
    status: UpmemTaskGraphStatus
    reason: str | None
    case_id: str
    policy: str
    quantization_mode: str
    output: np.ndarray | None
    output_labels: tuple[int, ...] | None
    final_validation: JsonDict
    summary: JsonDict
    task_metrics: tuple[JsonDict, ...]
    artifacts: JsonDict = field(default_factory=dict)

    def to_json_dict(self) -> JsonDict:
        return to_jsonable(
            {
                "schema_version": self.schema_version,
                "status": self.status,
                "reason": self.reason,
                "case_id": self.case_id,
                "policy": self.policy,
                "quantization_mode": self.quantization_mode,
                "output_labels": self.output_labels,
                "final_validation": self.final_validation,
                "summary": self.summary,
                "task_metrics": self.task_metrics,
                "artifacts": self.artifacts,
            }
        )


@dataclass(frozen=True)
class GenericQuantizedTaskGraphReference:
    schema_version: str
    status: Literal["completed", "unsupported", "failed"]
    reason: str | None
    case_id: str
    output: np.ndarray | None
    output_labels: tuple[int, ...] | None
    summary: JsonDict
    task_metrics: tuple[JsonDict, ...]

    def to_json_dict(self) -> JsonDict:
        return to_jsonable(
            {
                "schema_version": self.schema_version,
                "status": self.status,
                "reason": self.reason,
                "case_id": self.case_id,
                "output_shape": tuple(int(dim) for dim in np.asarray(self.output).shape) if self.output is not None else None,
                "output_dtype": str(np.asarray(self.output).dtype) if self.output is not None else None,
                "output_labels": self.output_labels,
                "summary": self.summary,
                "task_metrics": self.task_metrics,
            }
        )


def _unavailable_full_precision_accuracy() -> JsonDict:
    return {
        "full_precision_reference_kind": None,
        "full_precision_max_abs_error": None,
        "full_precision_l2_error": None,
        "full_precision_relative_l2_error": None,
        "available": False,
    }


def _generic_reference_base_task_metric(
    case_id: str,
    task_index: int,
    task: ContractionTask,
    *,
    status: str,
    reason: str | None,
    task_started: float,
    preparation: JsonDict | None = None,
    output: np.ndarray | None = None,
    component: str | None = None,
    component_metrics: JsonDict | None = None,
    complex_representation: str | None = None,
    validation_metrics: JsonDict | None = None,
    full_precision_metrics: JsonDict | None = None,
) -> JsonDict:
    output_array = np.asarray(output) if output is not None else None
    native_metadata = dict((preparation or {}).get("native_index_metadata") or {})
    prep_metadata = dict((preparation or {}).get("metadata") or {})
    conversion_summary = _conversion_summary(preparation=preparation, component_metrics=component_metrics)
    full_precision_metrics = _full_precision_metrics(full_precision_metrics, preparation)
    return to_jsonable(
        {
            "schema_version": GENERIC_QUANTIZED_TASKGRAPH_REFERENCE_TASK_SCHEMA_VERSION,
            "case_id": case_id,
            "task_index": task_index,
            "task_id": task.id,
            "input_tensor_ids": task.input_tensor_ids,
            "output_tensor_id": task.output_tensor_id,
            "status": status,
            "reason": reason,
            "component": component,
            "index_expression": task.index_expression,
            "input_shapes": task.input_shapes,
            "output_shape": tuple(int(dim) for dim in output_array.shape) if output_array is not None else task.output_shape,
            "input_ranks": (len(task.input_shapes[0]), len(task.input_shapes[1])),
            "output_rank": len(task.output_shape),
            "left_labels": task.left_labels,
            "right_labels": task.right_labels,
            "contracted_labels": task.contracted_labels,
            "output_labels": task.output_labels,
            "complex_quantization_scope": "per_task_operands" if complex_representation else None,
            "reference_kind": "generic_quantized_task_reference",
            "selected_kernel_family": prep_metadata.get("kernel_family", "generic_loop_fallback"),
            "validation_target": prep_metadata.get("validation_target", "expected_quantized_reference_output"),
            "full_precision_reference_is_validation_target": False,
            **_full_precision_metric_fields(full_precision_metrics),
            **conversion_summary,
            "complex_representation": complex_representation
            or conversion_summary.get("complex_representation")
            or ("real" if output_array is not None and not np.iscomplexobj(output_array) else None),
            "quantization_mode": prep_metadata.get("quantization_mode"),
            "operand_mode": prep_metadata.get("operand_mode"),
            "input_dtype_on_dpu": prep_metadata.get("input_dtype_on_dpu"),
            "accumulator_dtype_on_dpu": prep_metadata.get("accumulator_dtype_on_dpu"),
            "output_dtype_on_dpu": prep_metadata.get("output_dtype_on_dpu"),
            "unquantized_mode_kind": prep_metadata.get("unquantized_mode_kind"),
            "scaling_applied": prep_metadata.get("scaling_applied"),
            "quantization_time_s": float(prep_metadata.get("quantization_time_s", 0.0) or 0.0),
            "dequantization_time_s": float(prep_metadata.get("dequantization_time_s", 0.0) or 0.0),
            "float32_reference_time_s": float(prep_metadata.get("float32_reference_time_s", 0.0) or 0.0),
            "actual_h2d_bytes_model": int(prep_metadata.get("actual_h2d_bytes_model", 0) or 0),
            "actual_d2h_bytes_model": int(prep_metadata.get("actual_d2h_bytes_model", 0) or 0),
            "actual_transfer_bytes_model": int(prep_metadata.get("actual_h2d_bytes_model", 0) or 0)
            + int(prep_metadata.get("actual_d2h_bytes_model", 0) or 0),
            "full_precision_h2d_bytes_model": int(prep_metadata.get("full_precision_h2d_bytes_model", 0) or 0),
            "full_precision_d2h_bytes_model": int(prep_metadata.get("full_precision_d2h_bytes_model", 0) or 0),
            "full_precision_transfer_bytes_model": int(prep_metadata.get("full_precision_h2d_bytes_model", 0) or 0)
            + int(prep_metadata.get("full_precision_d2h_bytes_model", 0) or 0),
            "preparation_status": (preparation or {}).get("status"),
            "preparation_reason": (preparation or {}).get("reason"),
            "native_index_metadata": native_metadata,
            "conversion_records": (preparation or {}).get("conversion_records") or {},
            "validation_metrics": validation_metrics or (preparation or {}).get("validation_metrics") or {},
            "full_precision_error_metrics": (preparation or {}).get("full_precision_error_metrics") or {},
            "caps": (preparation or {}).get("caps") or {},
            "output_dtype": str(output_array.dtype) if output_array is not None else None,
            "output_bytes": int(output_array.nbytes) if output_array is not None else 0,
            "estimated_flops": task.estimated_flops,
            "estimated_bytes": task.estimated_bytes,
            "component_metrics": component_metrics or {},
            "reference_task_wall_time_s": float(time.perf_counter() - task_started),
        }
    )


def _generic_reference_summary(
    *,
    case_id: str,
    status: str,
    reason: str | None,
    started: float,
    task_metrics: list[JsonDict],
    quantization_mode: str = "per_task_input_quantize",
    final_tensor_id: str,
    final_tensor_labels: tuple[int, ...],
    final_transpose_applied: bool,
    peak_live_tensor_bytes: int,
) -> JsonDict:
    return to_jsonable(
        {
            "schema_version": GENERIC_QUANTIZED_TASKGRAPH_REFERENCE_SCHEMA_VERSION,
            "case_id": case_id,
            "status": status,
            "reason": reason,
            "reference_kind": _generic_reference_kind(quantization_mode),
            "task_reference_kind": "generic_quantized_task_reference",
            "quantization_mode": quantization_mode,
            "validation_target": "per_task_expected_float32_reference_output"
            if quantization_mode == "none"
            else "per_task_expected_quantized_reference_output",
            "full_precision_reference_is_task_validation_target": False,
            "whole_network_quantized_at_initialization": False,
            "total_tasks": len(task_metrics),
            "completed_tasks": sum(1 for row in task_metrics if row.get("status") == "completed"),
            "unsupported_tasks": sum(1 for row in task_metrics if row.get("status") == "unsupported"),
            "failed_tasks": sum(1 for row in task_metrics if row.get("status") == "failed"),
            "complex_split_tasks": sum(1 for row in task_metrics if row.get("complex_representation") == "split_real_imag"),
            "final_tensor_id": final_tensor_id,
            "final_tensor_labels": final_tensor_labels,
            "final_transpose_applied": final_transpose_applied,
            "peak_live_tensor_bytes": int(peak_live_tensor_bytes),
            "total_wall_time_s": float(time.perf_counter() - started),
            "input_dtype_on_dpu": _unique_or_none(task_metrics, "input_dtype_on_dpu"),
            "accumulator_dtype_on_dpu": _unique_or_none(task_metrics, "accumulator_dtype_on_dpu"),
            "output_dtype_on_dpu": _unique_or_none(task_metrics, "output_dtype_on_dpu"),
            "unquantized_mode_kind": _unique_or_none(task_metrics, "unquantized_mode_kind"),
            "scaling_applied": _unique_or_none(task_metrics, "scaling_applied"),
            "actual_h2d_bytes_model": int(sum(int(row.get("actual_h2d_bytes_model", 0) or 0) for row in task_metrics)),
            "actual_d2h_bytes_model": int(sum(int(row.get("actual_d2h_bytes_model", 0) or 0) for row in task_metrics)),
            "actual_transfer_bytes": int(sum(int(row.get("actual_transfer_bytes_model", 0) or 0) for row in task_metrics)),
            "full_precision_h2d_bytes_model": int(sum(int(row.get("full_precision_h2d_bytes_model", 0) or 0) for row in task_metrics)),
            "full_precision_d2h_bytes_model": int(sum(int(row.get("full_precision_d2h_bytes_model", 0) or 0) for row in task_metrics)),
            "full_precision_transfer_bytes_model": int(
                sum(int(row.get("full_precision_transfer_bytes_model", 0) or 0) for row in task_metrics)
            ),
        }
    )


def _generic_reference_stop_result(
    case_id: str,
    status: Literal["unsupported", "failed"],
    reason: str,
    started: float,
    task_metrics: list[JsonDict],
    quantization_mode: str = "per_task_input_quantize",
) -> GenericQuantizedTaskGraphReference:
    summary = _generic_reference_summary(
        case_id=case_id,
        status=status,
        reason=reason,
        started=started,
        task_metrics=task_metrics,
        quantization_mode=quantization_mode,
        final_tensor_id="",
        final_tensor_labels=(),
        final_transpose_applied=False,
        peak_live_tensor_bytes=0,
    )
    return GenericQuantizedTaskGraphReference(
        schema_version=GENERIC_QUANTIZED_TASKGRAPH_REFERENCE_SCHEMA_VERSION,
        status=status,
        reason=reason,
        case_id=case_id,
        output=None,
        output_labels=None,
        summary=summary,
        task_metrics=tuple(task_metrics),
    )


def _generic_reference_kind(quantization_mode: str) -> str:
    return "generic_float32_taskgraph_replay" if quantization_mode == "none" else "generic_quantized_taskgraph_replay"


def _base_task_metric(
    case_id: str,
    task_index: int,
    task: ContractionTask,
    policy: str,
    quantization_mode: str,
    *,
    status: str,
    reason: str | None,
    task_started: float,
    selected_kernel_family: str | None = None,
    backend_id: str | None = None,
    preparation: JsonDict | None = None,
    bridge_result: JsonDict | None = None,
    output: np.ndarray | None = None,
    bridge_artifact_path: Path | None = None,
    component: str | None = None,
    component_metrics: JsonDict | None = None,
    complex_representation: str | None = None,
    validation_metrics: JsonDict | None = None,
    full_precision_metrics: JsonDict | None = None,
    dense_reject_reason: str | None = None,
) -> JsonDict:
    output_array = np.asarray(output) if output is not None else None
    bridge_manifest = dict((bridge_result or {}).get("output_manifest") or {})
    bridge_metadata = dict(bridge_manifest.get("metadata") or {})
    bridge_validation = dict(bridge_manifest.get("validation_metrics") or {})
    conversion_records = dict((preparation or {}).get("conversion_records") or {})
    prep_metadata = dict((preparation or {}).get("metadata") or {})
    actual_h2d_bytes = bridge_metadata.get("actual_h2d_bytes", prep_metadata.get("actual_h2d_bytes_model"))
    actual_d2h_bytes = bridge_metadata.get("actual_d2h_bytes", prep_metadata.get("actual_d2h_bytes_model"))
    transfer = transfer_accounting(
        actual_h2d_bytes,
        actual_d2h_bytes,
        recorded_by_sdk=("actual_h2d_bytes" in bridge_metadata or "actual_d2h_bytes" in bridge_metadata),
    )
    full_precision_h2d_bytes = prep_metadata.get("full_precision_h2d_bytes_model", bridge_metadata.get("full_precision_h2d_bytes_model"))
    full_precision_d2h_bytes = prep_metadata.get("full_precision_d2h_bytes_model", bridge_metadata.get("full_precision_d2h_bytes_model"))
    conversion_summary = _conversion_summary(preparation=preparation, component_metrics=component_metrics)
    full_precision_metrics = _full_precision_metrics(full_precision_metrics, preparation)
    return to_jsonable(
        {
            "schema_version": UPMEM_TASKGRAPH_TASK_METRIC_SCHEMA_VERSION,
            "case_id": case_id,
            "task_index": task_index,
            "task_id": task.id,
            "input_tensor_ids": task.input_tensor_ids,
            "output_tensor_id": task.output_tensor_id,
            "selected_kernel_family": selected_kernel_family,
            "backend_id": backend_id,
            "policy": policy,
            "quantization_mode": quantization_mode,
            "whole_network_quantized_at_initialization": False,
            "operand_mode": bridge_metadata.get("operand_mode", prep_metadata.get("operand_mode")),
            "input_dtype_on_dpu": bridge_metadata.get("input_dtype_on_dpu", prep_metadata.get("input_dtype_on_dpu")),
            "accumulator_dtype_on_dpu": bridge_metadata.get("accumulator_dtype_on_dpu", prep_metadata.get("accumulator_dtype_on_dpu")),
            "output_dtype_on_dpu": bridge_metadata.get("output_dtype_on_dpu", prep_metadata.get("output_dtype_on_dpu")),
            "unquantized_mode_kind": bridge_metadata.get("unquantized_mode_kind", prep_metadata.get("unquantized_mode_kind")),
            "scaling_applied": bridge_metadata.get("scaling_applied", prep_metadata.get("scaling_applied")),
            "generic_kernel_strategy": bridge_metadata.get("generic_kernel_strategy", prep_metadata.get("generic_kernel_strategy", GENERIC_KERNEL_STRATEGY)),
            "native_max_rank": bridge_metadata.get("native_max_rank", prep_metadata.get("native_max_rank", GENERIC_NATIVE_MAX_RANK)),
            "native_max_tensor_elements": bridge_metadata.get("native_max_tensor_elements", prep_metadata.get("native_max_tensor_elements", GENERIC_NATIVE_MAX_TENSOR_ELEMENTS)),
            "generic_output_tile_elements": bridge_metadata.get("generic_output_tile_elements", prep_metadata.get("generic_output_tile_elements", GENERIC_OUTPUT_TILE_ELEMENTS)),
            "generic_output_tile_count": bridge_metadata.get("generic_output_tile_count", prep_metadata.get("generic_output_tile_count")),
            "mram_resident_operands": bridge_metadata.get("mram_resident_operands", prep_metadata.get("mram_resident_operands", True)),
            "wram_output_tiled": bridge_metadata.get("wram_output_tiled", prep_metadata.get("wram_output_tiled", True)),
            "mram_tiled_task_count": bridge_metadata.get("mram_tiled_task_count", prep_metadata.get("mram_tiled_task_count", 0)),
            "mram_read_bytes_model": bridge_metadata.get("mram_read_bytes_model", prep_metadata.get("mram_read_bytes_model", 0)),
            "mram_write_bytes_model": bridge_metadata.get("mram_write_bytes_model", prep_metadata.get("mram_write_bytes_model", 0)),
            "complex_quantization_scope": "per_task_operands" if complex_representation else None,
            **_full_precision_metric_fields(full_precision_metrics),
            **conversion_summary,
            "complex_representation": complex_representation
            or conversion_summary.get("complex_representation")
            or ("real" if output_array is not None and not np.iscomplexobj(output_array) else None),
            "contraction_execution_target": CONTRACTION_EXECUTION_TARGET,
            "upmem_execution_mode": UPMEM_EXECUTION_MODE,
            "dpu_program_executed": status == "completed",
            "native_sdk_control_path": True,
            "simplepim_api_used": False,
            "hardware_timing_available": False,
            "hardware_speedup_applicable": False,
            "cpu_contraction_fallback_used": False,
            "status": status,
            "reason": reason,
            "dense_reject_reason": dense_reject_reason,
            "component": component,
            "input_shapes": task.input_shapes,
            "output_shape": tuple(int(dim) for dim in output_array.shape) if output_array is not None else task.output_shape,
            "output_dtype": str(output_array.dtype) if output_array is not None else None,
            "output_bytes": int(output_array.nbytes) if output_array is not None else 0,
            "estimated_flops": task.estimated_flops,
            "estimated_bytes": task.estimated_bytes,
            "preparation_status": (preparation or {}).get("status"),
            "preparation_reason": (preparation or {}).get("reason"),
            "native_index_metadata": (preparation or {}).get("native_index_metadata"),
            "conversion_records": conversion_records,
            "bridge_execution_status": (bridge_result or {}).get("execution_status"),
            "bridge_reason": (bridge_result or {}).get("reason"),
            "bridge_validation_metrics": validation_metrics or bridge_validation,
            "bridge_total_time_s": float(bridge_manifest.get("total_time_s", 0.0) or 0.0),
            "kernel_time_s": float(bridge_manifest.get("compute_time_s", 0.0) or 0.0),
            "build_time_s": float(bridge_metadata.get("build_time_s", 0.0) or 0.0),
            "quantization_time_s": float(prep_metadata.get("quantization_time_s", 0.0) or 0.0),
            "dequantization_time_s": float(prep_metadata.get("dequantization_time_s", 0.0) or 0.0),
            "float32_reference_time_s": float(prep_metadata.get("float32_reference_time_s", 0.0) or 0.0),
            **transfer,
            "full_precision_h2d_bytes_model": int(full_precision_h2d_bytes or 0),
            "full_precision_d2h_bytes_model": int(full_precision_d2h_bytes or 0),
            "full_precision_transfer_bytes_model": int((full_precision_h2d_bytes or 0) + (full_precision_d2h_bytes or 0)),
            "runtime_task_wall_time_s": float(time.perf_counter() - task_started),
            "bridge_artifact_path": _metric_artifact_path(bridge_artifact_path),
            "component_metrics": component_metrics or {},
            "runtime_tensor_source": "upmem_output_blob" if status == "completed" else None,
            "cpu_reference_artifact_used_as_runtime_input": False,
        }
    )


def _schedule_metadata(
    *,
    schedule_mode: str,
    frontier_worker_count: int,
    dpu_group_count: int,
    task_assignment_strategy: str,
    frontier_widths: tuple[int, ...],
    scheduler_overhead_s: float,
    executed_task_count: int,
    duplicate_contraction_check: str,
    missing_dependency_check: str,
    dependency_violation_detected: bool,
) -> JsonDict:
    frontier_enabled = schedule_mode == "frontier"
    assigned_task_count = int(sum(frontier_widths)) if frontier_enabled else None
    return to_jsonable(
        {
            "parallelism_mode": "frontier" if frontier_enabled else "sequential",
            "parallelism_evidence_type": "executed",
            "execution_plan_kind": "upmem_frontier_assignment_scheduler" if frontier_enabled else "sequential_upmem_taskgraph",
            "execution_plan_executed": True,
            "frontier_scheduler_enabled": frontier_enabled,
            "frontier_parallel_execution": bool(frontier_enabled and frontier_worker_count > 1 and any(width > 1 for width in frontier_widths)),
            "frontier_worker_count": int(frontier_worker_count) if frontier_enabled else None,
            "frontier_wave_count": len(frontier_widths) if frontier_enabled else None,
            "max_frontier_width": max(frontier_widths, default=0) if frontier_enabled else None,
            "mean_frontier_width": (sum(frontier_widths) / len(frontier_widths)) if frontier_enabled and frontier_widths else None,
            "frontier_executed_task_count": int(executed_task_count) if frontier_enabled else None,
            "frontier_executed_parallel_task_count": 0 if frontier_enabled else None,
            "executed_parallel_task_count": 0 if frontier_enabled else None,
            "scheduler_overhead_s": float(scheduler_overhead_s) if frontier_enabled else None,
            "duplicate_contraction_check": duplicate_contraction_check if frontier_enabled else None,
            "missing_dependency_check": missing_dependency_check if frontier_enabled else None,
            "dependency_violation_detected": bool(dependency_violation_detected) if frontier_enabled else False,
            "upmem_parallelism_mode": "frontier_multi_dpu" if frontier_enabled else "sequential",
            "upmem_parallelism_evidence_type": "sdk_simulator_executed" if frontier_enabled else None,
            "task_assignment_strategy": task_assignment_strategy if frontier_enabled else None,
            "dpu_group_count": int(dpu_group_count) if frontier_enabled else None,
            "assigned_task_count": assigned_task_count,
            "executed_dpu_task_count": int(executed_task_count) if frontier_enabled else None,
            "unassigned_task_count": max(0, int(assigned_task_count or 0) - int(executed_task_count)) if frontier_enabled else None,
            "dpu_assignment_validation_status": (
                "passed"
                if frontier_enabled and duplicate_contraction_check == "passed" and missing_dependency_check == "passed" and not dependency_violation_detected
                else ("failed" if frontier_enabled else None)
            ),
            "hardware_execution": False,
            "hardware_timing_available": False,
            "hardware_speedup_applicable": False,
        }
    )


def _summary_payload(
    *,
    case_id: str,
    policy: str,
    quantization_mode: str,
    status: str,
    reason: str | None,
    started: float,
    task_metrics: list[JsonDict],
    kernel_family_counts: dict[str, int],
    backend_counts: dict[str, int],
    final_validation: JsonDict,
    final_full_precision_accuracy: JsonDict | None = None,
    final_tensor_id: str,
    final_tensor_labels: tuple[int, ...],
    final_transpose_applied: bool,
    total_bridge_time_s: float,
    total_kernel_time_s: float,
    total_build_time_s: float,
    peak_live_tensor_bytes: int,
    schedule_metadata: JsonDict | None = None,
    execution_metadata: JsonDict | None = None,
) -> JsonDict:
    schedule_metadata = schedule_metadata or _schedule_metadata(
        schedule_mode="sequential",
        frontier_worker_count=1,
        dpu_group_count=1,
        task_assignment_strategy="sequential_single_dpu",
        frontier_widths=tuple(1 for _ in task_metrics),
        scheduler_overhead_s=0.0,
        executed_task_count=sum(1 for row in task_metrics if row.get("status") == "completed"),
        duplicate_contraction_check="passed",
        missing_dependency_check="passed",
        dependency_violation_detected=False,
    )
    executed_tasks = sum(1 for row in task_metrics if row.get("status") == "completed")
    dpu_invocations = sum(1 for row in task_metrics if row.get("dpu_program_executed") is True)
    total_actual_h2d_bytes = sum(int(row.get("actual_h2d_bytes", 0) or 0) for row in task_metrics)
    total_actual_d2h_bytes = sum(int(row.get("actual_d2h_bytes", 0) or 0) for row in task_metrics)
    total_full_precision_h2d_bytes = sum(int(row.get("full_precision_h2d_bytes_model", 0) or 0) for row in task_metrics)
    total_full_precision_d2h_bytes = sum(int(row.get("full_precision_d2h_bytes_model", 0) or 0) for row in task_metrics)
    total_actual_transfer_bytes = total_actual_h2d_bytes + total_actual_d2h_bytes
    total_full_precision_transfer_bytes = total_full_precision_h2d_bytes + total_full_precision_d2h_bytes
    transfer = transfer_accounting(
        total_actual_h2d_bytes,
        total_actual_d2h_bytes,
        declared_total_bytes=total_actual_transfer_bytes,
        recorded_by_sdk=all(
            row.get("transfer_accounting_scope") == APPLICATION_VISIBLE_SDK_TRANSFER_SCOPE
            for row in task_metrics
            if row.get("status") == "completed"
        ),
    )
    total_clipping = sum(int(row.get("quantization_clipping_count", 0) or 0) for row in task_metrics)
    total_saturation = sum(int(row.get("quantization_saturation_count", 0) or 0) for row in task_metrics)
    total_left_clipping = sum(int(row.get("left_quantization_clipping_count", 0) or 0) for row in task_metrics)
    total_right_clipping = sum(int(row.get("right_quantization_clipping_count", 0) or 0) for row in task_metrics)
    total_left_saturation = sum(int(row.get("left_quantization_saturation_count", 0) or 0) for row in task_metrics)
    total_right_saturation = sum(int(row.get("right_quantization_saturation_count", 0) or 0) for row in task_metrics)
    transfer_compression_ratio = (
        float(total_full_precision_transfer_bytes) / float(total_actual_transfer_bytes)
        if total_actual_transfer_bytes > 0
        else None
    )
    dpu_all = bool(task_metrics) and all(row.get("dpu_program_executed") is True for row in task_metrics)
    target_all = bool(task_metrics) and all(row.get("contraction_execution_target") == CONTRACTION_EXECUTION_TARGET for row in task_metrics)
    mode_all = bool(task_metrics) and all(row.get("upmem_execution_mode") == UPMEM_EXECUTION_MODE for row in task_metrics)
    generic_only_all_generic = bool(task_metrics) and all(
        row.get("status") == "completed"
        and row.get("selected_kernel_family") == "generic_loop_fallback"
        and row.get("backend_id") == "upmem_sdk_simulator_generic_loop"
        for row in task_metrics
    )
    no_cpu_feed = bool(task_metrics) and all(
        row.get("runtime_tensor_source") == "upmem_output_blob" and row.get("cpu_reference_artifact_used_as_runtime_input") is False
        for row in task_metrics
        if row.get("status") == "completed"
    )
    valid_primary = (
        status == "completed"
        and target_all
        and mode_all
        and dpu_all
        and no_cpu_feed
        and final_validation.get("passed") is True
        and all(row.get("cpu_contraction_fallback_used") is False for row in task_metrics)
    )
    return to_jsonable(
        {
            "schema_version": UPMEM_TASKGRAPH_RUNTIME_SCHEMA_VERSION,
            "case_id": case_id,
            "status": status,
            "reason": reason,
            **dict(execution_metadata or {}),
            "policy": policy,
            "quantization_mode": quantization_mode,
            "whole_network_quantized_at_initialization": False,
            "contraction_execution_target": CONTRACTION_EXECUTION_TARGET,
            "upmem_execution_mode": UPMEM_EXECUTION_MODE,
            "native_sdk_control_path": True,
            "simplepim_api_used": False,
            "input_dtype_on_dpu": _unique_or_none(task_metrics, "input_dtype_on_dpu"),
            "accumulator_dtype_on_dpu": _unique_or_none(task_metrics, "accumulator_dtype_on_dpu"),
            "output_dtype_on_dpu": _unique_or_none(task_metrics, "output_dtype_on_dpu"),
            "unquantized_mode_kind": _unique_or_none(task_metrics, "unquantized_mode_kind"),
            "scaling_applied": _unique_or_none(task_metrics, "scaling_applied"),
            "hardware_benchmark_result": False,
            "hardware_timing_available": False,
            "hardware_speedup_applicable": False,
            "cpu_fallback_used": False,
            "cpu_fallback_task_count": 0,
            "dpu_program_executed_all_tasks": dpu_all,
            "dpu_program_invocations": dpu_invocations,
            "runtime_tensor_sources_all_upmem_output_blobs": no_cpu_feed,
            "generic_only_all_tasks_used_generic_backend": generic_only_all_generic if policy == "generic-only" else None,
            "valid_primary_upmem_codepath_result": valid_primary,
            "total_tasks": len(task_metrics),
            "executed_tasks": executed_tasks,
            "upmem_task_count": executed_tasks,
            "unsupported_tasks": sum(1 for row in task_metrics if row.get("status") == "unsupported"),
            "failed_tasks": sum(1 for row in task_metrics if row.get("status") == "failed"),
            "dpu_program_executed_task_count": dpu_invocations,
            "kernel_family_counts": kernel_family_counts,
            "backend_counts": backend_counts,
            "final_tensor_id": final_tensor_id,
            "final_tensor_labels": final_tensor_labels,
            "final_transpose_applied": final_transpose_applied,
            "final_validation": final_validation,
            "final_full_precision_accuracy": final_full_precision_accuracy or _unavailable_full_precision_accuracy(),
            "full_precision_reference_kind": (final_full_precision_accuracy or {}).get("full_precision_reference_kind"),
            "full_precision_max_abs_error": (final_full_precision_accuracy or {}).get("full_precision_max_abs_error"),
            "full_precision_l2_error": (final_full_precision_accuracy or {}).get("full_precision_l2_error"),
            "full_precision_relative_l2_error": (final_full_precision_accuracy or {}).get("full_precision_relative_l2_error"),
            "total_wall_time_s": float(time.perf_counter() - started),
            "total_bridge_time_s": float(total_bridge_time_s),
            "total_kernel_time_s": float(total_kernel_time_s),
            "total_build_time_s": float(total_build_time_s),
            "total_quantization_time_s": float(sum(float(row.get("quantization_time_s", 0.0) or 0.0) for row in task_metrics)),
            "total_dequantization_time_s": float(sum(float(row.get("dequantization_time_s", 0.0) or 0.0) for row in task_metrics)),
            "total_float32_reference_time_s": float(sum(float(row.get("float32_reference_time_s", 0.0) or 0.0) for row in task_metrics)),
            **transfer,
            "full_precision_h2d_bytes_model": int(total_full_precision_h2d_bytes),
            "full_precision_d2h_bytes_model": int(total_full_precision_d2h_bytes),
            "full_precision_transfer_bytes_model": int(total_full_precision_transfer_bytes),
            "transfer_compression_ratio": transfer_compression_ratio,
            "left_quantization_clipping_count": int(total_left_clipping),
            "right_quantization_clipping_count": int(total_right_clipping),
            "quantization_clipping_count": int(total_clipping),
            "left_quantization_saturation_count": int(total_left_saturation),
            "right_quantization_saturation_count": int(total_right_saturation),
            "quantization_saturation_count": int(total_saturation),
            "left_quantization_scale": _unique_or_none(task_metrics, "left_quantization_scale"),
            "right_quantization_scale": _unique_or_none(task_metrics, "right_quantization_scale"),
            "quantization_scales": {
                "left": [row.get("left_quantization_scale") for row in task_metrics if row.get("left_quantization_scale") is not None],
                "right": [row.get("right_quantization_scale") for row in task_metrics if row.get("right_quantization_scale") is not None],
            },
            "complex_representation": _unique_or_none(task_metrics, "complex_representation"),
            "generic_kernel_strategy": _unique_or_none(task_metrics, "generic_kernel_strategy") or GENERIC_KERNEL_STRATEGY,
            "native_max_rank": _unique_or_none(task_metrics, "native_max_rank") or GENERIC_NATIVE_MAX_RANK,
            "native_max_tensor_elements": _unique_or_none(task_metrics, "native_max_tensor_elements") or GENERIC_NATIVE_MAX_TENSOR_ELEMENTS,
            "generic_output_tile_elements": _unique_or_none(task_metrics, "generic_output_tile_elements") or GENERIC_OUTPUT_TILE_ELEMENTS,
            "generic_output_tile_count": int(sum(int(row.get("generic_output_tile_count", 0) or 0) for row in task_metrics)),
            "mram_resident_operands": all(row.get("mram_resident_operands", True) is True for row in task_metrics),
            "wram_output_tiled": all(row.get("wram_output_tiled", True) is True for row in task_metrics),
            "mram_tiled_task_count": int(sum(int(row.get("mram_tiled_task_count", 0) or 0) for row in task_metrics)),
            "mram_read_bytes_model": int(sum(int(row.get("mram_read_bytes_model", 0) or 0) for row in task_metrics)),
            "mram_write_bytes_model": int(sum(int(row.get("mram_write_bytes_model", 0) or 0) for row in task_metrics)),
            "peak_live_tensor_bytes": int(peak_live_tensor_bytes),
            "task_metrics_artifact": None,
            "final_tensor_artifact": None,
            **schedule_metadata,
        }
    )
def _stop_result(
    *,
    case_id: str,
    policy: str,
    quantization_mode: str,
    status: UpmemTaskGraphStatus,
    reason: str,
    started: float,
    task_metrics: list[JsonDict],
    schedule_metadata: JsonDict | None = None,
    execution_metadata: JsonDict | None = None,
) -> UpmemTaskGraphRuntimeResult:
    summary = _summary_payload(
        case_id=case_id,
        policy=policy,
        quantization_mode=quantization_mode,
        status=status,
        reason=reason,
        started=started,
        task_metrics=task_metrics,
        kernel_family_counts=_counts(task_metrics, "selected_kernel_family"),
        backend_counts=_counts(task_metrics, "backend_id"),
        final_validation={"passed": False, "reason": "not_available"},
        final_tensor_id="",
        final_tensor_labels=(),
        final_transpose_applied=False,
        total_bridge_time_s=sum(float(row.get("bridge_total_time_s", 0.0) or 0.0) for row in task_metrics),
        total_kernel_time_s=sum(float(row.get("kernel_time_s", 0.0) or 0.0) for row in task_metrics),
        total_build_time_s=sum(float(row.get("build_time_s", 0.0) or 0.0) for row in task_metrics),
        peak_live_tensor_bytes=0,
        schedule_metadata=schedule_metadata,
        execution_metadata=execution_metadata,
    )
    return UpmemTaskGraphRuntimeResult(
        schema_version=UPMEM_TASKGRAPH_RUNTIME_SCHEMA_VERSION,
        status=status,
        reason=reason,
        case_id=case_id,
        policy=policy,
        quantization_mode=quantization_mode,
        output=None,
        output_labels=None,
        final_validation=summary["final_validation"],
        summary=summary,
        task_metrics=tuple(task_metrics),
    )


def _unsupported_result(
    case_id: str,
    policy: str,
    quantization_mode: str,
    reason: str,
    started: float,
    execution_metadata: JsonDict | None = None,
) -> UpmemTaskGraphRuntimeResult:
    return _stop_result(
        case_id=case_id,
        policy=policy,
        quantization_mode=quantization_mode,
        status="unsupported",
        reason=reason,
        started=started,
        task_metrics=[],
        execution_metadata=execution_metadata,
    )


def _counts(rows: list[JsonDict], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = row.get(key)
        if value:
            counts[str(value)] = counts.get(str(value), 0) + 1
    return counts


def _unique_or_none(rows: list[JsonDict], key: str) -> object | None:
    values = {row.get(key) for row in rows if row.get(key) is not None}
    if len(values) == 1:
        return next(iter(values))
    if not values:
        return None
    return "mixed"


def _full_precision_metric_fields(metrics: JsonDict) -> JsonDict:
    return {
        "full_precision_reference_kind": metrics.get("reference_kind"),
        "full_precision_max_abs_error": metrics.get("max_abs_error"),
        "full_precision_l2_error": metrics.get("l2_error"),
        "full_precision_relative_l2_error": metrics.get("relative_l2_error"),
    }


def _full_precision_metrics(explicit: JsonDict | None, preparation: JsonDict | None) -> JsonDict:
    metrics = dict(explicit or (preparation or {}).get("full_precision_error_metrics") or {})
    if metrics:
        return metrics
    dense_validation = dict((preparation or {}).get("validation_metrics") or {})
    if "dequantized_output_max_abs_error" not in dense_validation:
        return {}
    return {
        "reference_kind": "full_precision_reference_vs_dequantized_output",
        "max_abs_error": dense_validation.get("dequantized_output_max_abs_error"),
        "l2_error": dense_validation.get("dequantized_output_l2_error"),
        "relative_l2_error": dense_validation.get("dequantized_output_relative_l2_error"),
    }


def _conversion_summary(
    *,
    preparation: JsonDict | None,
    component_metrics: JsonDict | None,
) -> JsonDict:
    records = dict((preparation or {}).get("conversion_records") or {})
    if records:
        left = dict(records.get("left") or {})
        right = dict(records.get("right") or {})
        return {
            "complex_representation": (
                "split_real_imag"
                if left.get("representation") == "split_complex_real_imag"
                or right.get("representation") == "split_complex_real_imag"
                else "real"
            ),
            "left_quantization_scale": left.get("scale"),
            "right_quantization_scale": right.get("scale"),
            "quantization_scales": {"left": left.get("scale"), "right": right.get("scale")},
            "left_quantization_clipping_count": int(left.get("clipping_count", 0) or 0),
            "right_quantization_clipping_count": int(right.get("clipping_count", 0) or 0),
            "quantization_clipping_count": int(left.get("clipping_count", 0) or 0)
            + int(right.get("clipping_count", 0) or 0),
            "left_quantization_saturation_count": int(left.get("saturation_count", 0) or 0),
            "right_quantization_saturation_count": int(right.get("saturation_count", 0) or 0),
            "quantization_saturation_count": int(left.get("saturation_count", 0) or 0)
            + int(right.get("saturation_count", 0) or 0),
        }

    component_rows = [row for row in (component_metrics or {}).values() if isinstance(row, dict)]
    if not component_rows:
        return {
            "complex_representation": _unique_or_none(component_rows, "complex_representation"),
            "left_quantization_scale": None,
            "right_quantization_scale": None,
            "quantization_scales": None,
            "left_quantization_clipping_count": 0,
            "right_quantization_clipping_count": 0,
            "quantization_clipping_count": 0,
            "left_quantization_saturation_count": 0,
            "right_quantization_saturation_count": 0,
            "quantization_saturation_count": 0,
        }
    left_scales = [row.get("left_quantization_scale") for row in component_rows if row.get("left_quantization_scale") is not None]
    right_scales = [row.get("right_quantization_scale") for row in component_rows if row.get("right_quantization_scale") is not None]
    left_scale = left_scales[0] if left_scales and all(value == left_scales[0] for value in left_scales) else None
    right_scale = right_scales[0] if right_scales and all(value == right_scales[0] for value in right_scales) else None
    return {
        "complex_representation": _unique_or_none(component_rows, "complex_representation"),
        "left_quantization_scale": left_scale,
        "right_quantization_scale": right_scale,
        "quantization_scales": {"left": left_scales, "right": right_scales},
        "left_quantization_clipping_count": sum(int(row.get("left_quantization_clipping_count", 0) or 0) for row in component_rows),
        "right_quantization_clipping_count": sum(int(row.get("right_quantization_clipping_count", 0) or 0) for row in component_rows),
        "quantization_clipping_count": sum(int(row.get("quantization_clipping_count", 0) or 0) for row in component_rows),
        "left_quantization_saturation_count": sum(int(row.get("left_quantization_saturation_count", 0) or 0) for row in component_rows),
        "right_quantization_saturation_count": sum(int(row.get("right_quantization_saturation_count", 0) or 0) for row in component_rows),
        "quantization_saturation_count": sum(int(row.get("quantization_saturation_count", 0) or 0) for row in component_rows),
    }
def _metric_artifact_path(path: Path | None) -> str | None:
    if path is None:
        return None
    parts = path.parts
    for index in range(len(parts) - 1, -1, -1):
        if parts[index] == "cases":
            return Path(*parts[index:]).as_posix()
    return path.name
