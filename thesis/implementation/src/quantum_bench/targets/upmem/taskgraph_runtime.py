from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Mapping

import numpy as np

from quantum_bench.core.records import ContractionTask, JsonDict, TensorSpec, TensorValue, to_jsonable
from quantum_bench.formats import conversion_error_metrics
from quantum_bench.routing import DenseTaskPreparationInput, GenericTaskPreparationInput, prepare_dense_task, prepare_generic_task
from quantum_bench.targets.upmem.dense_bridge import (
    dense_bridge_backend_manifest_eligibility,
    execute_dense_bridge,
    write_dense_bridge_input_manifest,
)
from quantum_bench.targets.upmem.evidence import (
    CONTRACTION_EXECUTION_TARGET_UPMEM,
    UPMEM_EXECUTION_MODE_SDK_SIMULATOR,
)
from quantum_bench.targets.upmem.generic_bridge import execute_generic_bridge, write_generic_bridge_input_manifest
from quantum_bench.tn.execution import frontier_waves, live_tensor_bytes, order_final_tensor, release_dead_inputs, remaining_input_uses
from quantum_bench.tn.execution_bundle import execution_identity_metadata, executor_config_hash, with_execution_identity
from quantum_bench.tn.network import TensorNetworkValue
from quantum_bench.validation import validate


UPMEM_TASKGRAPH_RUNTIME_SCHEMA_VERSION = "upmem_taskgraph_runtime_v1"
UPMEM_TASKGRAPH_TASK_METRIC_SCHEMA_VERSION = "upmem_taskgraph_task_metric_v1"
GENERIC_QUANTIZED_TASKGRAPH_REFERENCE_SCHEMA_VERSION = "generic_quantized_taskgraph_reference_v1"
GENERIC_QUANTIZED_TASKGRAPH_REFERENCE_TASK_SCHEMA_VERSION = "generic_quantized_taskgraph_reference_task_v1"

UPMEM_TASKGRAPH_POLICIES = ("generic-only", "dense-then-generic", "dense-only")
UPMEM_TASKGRAPH_QUANTIZATION_MODES = ("per_task_input_quantize", "none", "persistent_network_quantized")
GENERIC_KERNEL_STRATEGY = "mram_resident_output_tiled_v1"
GENERIC_NATIVE_MAX_RANK = 16
GENERIC_NATIVE_MAX_TENSOR_ELEMENTS = 65536
GENERIC_OUTPUT_TILE_ELEMENTS = 256

UpmemTaskGraphPolicy = Literal["generic-only", "dense-then-generic", "dense-only"]
UpmemTaskGraphQuantizationMode = Literal["per_task_input_quantize", "none", "persistent_network_quantized"]


def upmem_taskgraph_executor_config(
    *,
    policy: str,
    quantization_mode: str,
    schedule_mode: str = "sequential",
    frontier_worker_count: int = 1,
    dpu_group_count: int = 1,
    task_assignment_strategy: str = "sequential_single_dpu",
) -> JsonDict:
    return {
        "policy": policy,
        "quantization_mode": quantization_mode,
        "schedule_mode": schedule_mode,
        "frontier_worker_count": frontier_worker_count,
        "dpu_group_count": dpu_group_count,
        "task_assignment_strategy": task_assignment_strategy,
        "generic_kernel_strategy": GENERIC_KERNEL_STRATEGY,
        "native_max_rank": GENERIC_NATIVE_MAX_RANK,
        "native_max_tensor_elements": GENERIC_NATIVE_MAX_TENSOR_ELEMENTS,
        "generic_output_tile_elements": GENERIC_OUTPUT_TILE_ELEMENTS,
    }
UpmemTaskGraphStatus = Literal["completed", "unsupported", "failed", "validation_failed"]
UpmemTaskGraphScheduleMode = Literal["sequential", "frontier"]

CONTRACTION_EXECUTION_TARGET = CONTRACTION_EXECUTION_TARGET_UPMEM
UPMEM_EXECUTION_MODE = UPMEM_EXECUTION_MODE_SDK_SIMULATOR
QUANTIZED_FINAL_VALIDATION_TOLERANCES = {
    "max_abs_error": 0.25,
    "l2_error": 2.0,
    "max_rel_error": 10.0,
    "norm_drift": 2.0,
    "min_fidelity": 0.0,
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


def build_generic_taskgraph_reference(
    *,
    graph,
    network: TensorNetworkValue,
    case_id: str,
    quantization_mode: UpmemTaskGraphQuantizationMode = "per_task_input_quantize",
) -> GenericQuantizedTaskGraphReference:
    """Replay a TaskGraph using the generic per-task native numeric contract.

    This is the CPU reference for strict generic-only UPMEM validation. It
    intentionally mirrors the native generic path rather than full-precision
    einsum: `per_task_input_quantize` uses int8 x int8 -> int32 with
    dequantization, while `none` uses float32 operands and float32 accumulation.
    Complex tensors use the same split-real-imag four-component contract as the
    generic runtime.
    """

    started = time.perf_counter()
    if not graph.tasks:
        return _generic_reference_stop_result(
            case_id,
            "unsupported",
            "empty_task_graph_not_supported",
            started,
            [],
            quantization_mode=quantization_mode,
        )

    tensors = {tensor.spec.id: np.asarray(tensor.array) for tensor in network.tensors}
    labels = {tensor.spec.id: tensor.spec.labels for tensor in network.tensors}
    live_ids = set(tensors)
    remaining_uses = remaining_input_uses(graph)
    final_tensor_id = graph.tasks[-1].output_tensor_id
    final_labels: tuple[int, ...] | None = None
    task_metrics: list[JsonDict] = []
    peak_live_bytes = live_tensor_bytes(tensors, live_ids)

    for task_index, task in enumerate(graph.tasks):
        task_started = time.perf_counter()
        if not _inputs_available(task, tensors):
            missing = [tensor_id for tensor_id in task.input_tensor_ids if tensor_id not in tensors]
            metric = _generic_reference_base_task_metric(
                case_id,
                task_index,
                task,
                status="unsupported",
                reason=f"missing:{','.join(missing)}",
                task_started=task_started,
            )
            return _generic_reference_stop_result(
                case_id,
                "unsupported",
                "runtime_input_tensor_missing",
                started,
                task_metrics + [metric],
                quantization_mode=quantization_mode,
            )

        left_tensor = _tensor_value_for(task.input_tensor_ids[0], task, tensors, labels, side="left")
        right_tensor = _tensor_value_for(task.input_tensor_ids[1], task, tensors, labels, side="right")
        reference = _generic_quantized_task_reference(
            task=task,
            task_index=task_index,
            case_id=case_id,
            left_tensor=left_tensor,
            right_tensor=right_tensor,
            task_started=task_started,
            quantization_mode=quantization_mode,
        )
        task_metrics.append(reference["metric"])
        if reference["status"] != "completed":
            return _generic_reference_stop_result(
                case_id,
                "unsupported" if reference["status"] == "unsupported" else "failed",
                str(reference["reason"]),
                started,
                task_metrics,
                quantization_mode=quantization_mode,
            )

        output = np.asarray(reference["output"])
        tensors[task.output_tensor_id] = output
        labels[task.output_tensor_id] = task.output_labels
        live_ids.add(task.output_tensor_id)
        final_labels = task.output_labels
        release_dead_inputs(task.input_tensor_ids, task.output_tensor_id, final_tensor_id, tensors, labels, live_ids, remaining_uses)
        peak_live_bytes = max(peak_live_bytes, live_tensor_bytes(tensors, live_ids))

    if final_tensor_id not in tensors or final_labels is None:
        return _generic_reference_stop_result(
            case_id,
            "failed",
            "final_tensor_missing",
            started,
            task_metrics,
            quantization_mode=quantization_mode,
        )

    final_output, final_transposed = order_final_tensor(np.asarray(tensors[final_tensor_id]), final_labels, graph.network.output_labels)
    summary = _generic_reference_summary(
        case_id=case_id,
        status="completed",
        reason=None,
        started=started,
        task_metrics=task_metrics,
        quantization_mode=quantization_mode,
        final_tensor_id=final_tensor_id,
        final_tensor_labels=final_labels,
        final_transpose_applied=final_transposed,
        peak_live_tensor_bytes=peak_live_bytes,
    )
    return GenericQuantizedTaskGraphReference(
        schema_version=GENERIC_QUANTIZED_TASKGRAPH_REFERENCE_SCHEMA_VERSION,
        status="completed",
        reason=None,
        case_id=case_id,
        output=np.asarray(final_output),
        output_labels=graph.network.output_labels,
        summary=summary,
        task_metrics=tuple(task_metrics),
    )


def build_generic_quantized_taskgraph_reference(
    *,
    graph,
    network: TensorNetworkValue,
    case_id: str,
) -> GenericQuantizedTaskGraphReference:
    return build_generic_taskgraph_reference(
        graph=graph,
        network=network,
        case_id=case_id,
        quantization_mode="per_task_input_quantize",
    )


def execute_upmem_taskgraph_runtime(
    *,
    graph,
    network: TensorNetworkValue,
    case_id: str,
    policy: UpmemTaskGraphPolicy,
    quantization_mode: UpmemTaskGraphQuantizationMode,
    bridge_root: Path,
    execute_external: bool,
    reference_output: np.ndarray | None = None,
    reference_kind: str = "cpu_exact_taskgraph_full_precision",
    env: Mapping[str, str] | None = None,
    schedule_mode: UpmemTaskGraphScheduleMode = "sequential",
    frontier_worker_count: int = 1,
    dpu_group_count: int = 1,
    task_assignment_strategy: str = "sequential_single_dpu",
) -> UpmemTaskGraphRuntimeResult:
    started = time.perf_counter()
    graph = with_execution_identity(graph)
    execution_metadata = {
        **execution_identity_metadata(graph, plan_reused=True),
        "executor_config_hash": executor_config_hash(
            "upmem_tn_runtime",
            upmem_taskgraph_executor_config(
                policy=policy,
                quantization_mode=quantization_mode,
                schedule_mode=schedule_mode,
                frontier_worker_count=frontier_worker_count,
                dpu_group_count=dpu_group_count,
                task_assignment_strategy=task_assignment_strategy,
            ),
        ),
    }
    if policy not in UPMEM_TASKGRAPH_POLICIES:
        return _unsupported_result(case_id, policy, quantization_mode, "unsupported_policy", started, execution_metadata)
    if schedule_mode not in {"sequential", "frontier"}:
        return _unsupported_result(case_id, policy, quantization_mode, f"unsupported_schedule_mode:{schedule_mode}", started, execution_metadata)
    if frontier_worker_count < 1:
        return _unsupported_result(case_id, policy, quantization_mode, "frontier_worker_count_must_be_positive", started, execution_metadata)
    if dpu_group_count < 1:
        return _unsupported_result(case_id, policy, quantization_mode, "dpu_group_count_must_be_positive", started, execution_metadata)
    if schedule_mode == "frontier" and frontier_worker_count > 1:
        return _unsupported_result(case_id, policy, quantization_mode, "frontier_worker_count_gt_1_not_implemented", started, execution_metadata)
    if quantization_mode not in {"per_task_input_quantize", "none"}:
        return _unsupported_result(case_id, policy, quantization_mode, f"unsupported_quantization_mode:{quantization_mode}", started, execution_metadata)
    if quantization_mode == "none" and policy != "generic-only":
        return _unsupported_result(case_id, policy, quantization_mode, "quantization_none_requires_generic_only", started, execution_metadata)
    if not execute_external:
        return _unsupported_result(case_id, policy, quantization_mode, "external_execution_required", started, execution_metadata)
    if not graph.tasks:
        return _unsupported_result(case_id, policy, quantization_mode, "empty_task_graph_not_supported", started, execution_metadata)

    tensors = {tensor.spec.id: np.asarray(tensor.array) for tensor in network.tensors}
    labels = {tensor.spec.id: tensor.spec.labels for tensor in network.tensors}
    live_ids = set(tensors)
    remaining_uses = remaining_input_uses(graph)
    final_tensor_id = graph.tasks[-1].output_tensor_id
    task_metrics: list[JsonDict] = []
    kernel_family_counts: dict[str, int] = {}
    backend_counts: dict[str, int] = {}
    total_bridge_time_s = 0.0
    total_kernel_time_s = 0.0
    total_build_time_s = 0.0
    peak_live_bytes = live_tensor_bytes(tensors, live_ids)
    final_labels: tuple[int, ...] | None = None
    scheduler_started = time.perf_counter()
    schedule_waves = frontier_waves(graph) if schedule_mode == "frontier" else tuple((task,) for task in graph.tasks)
    scheduler_overhead_s = time.perf_counter() - scheduler_started
    frontier_widths = tuple(len(wave) for wave in schedule_waves)
    max_frontier_width = max(frontier_widths, default=0)
    mean_frontier_width = (sum(frontier_widths) / len(frontier_widths)) if frontier_widths else 0.0
    task_indices = {task.id: index for index, task in enumerate(graph.tasks)}
    completed_task_ids: set[str] = set()
    executed_task_ids: set[str] = set()
    dependency_violation_detected = False
    missing_dependency_check = "passed"
    duplicate_contraction_check = "passed"

    for wave_index, wave in enumerate(schedule_waves):
        for task in wave:
            task_index = task_indices[task.id]
            task_started = time.perf_counter()
            if task.id in executed_task_ids:
                duplicate_contraction_check = "failed"
                return _stop_result(
                    case_id=case_id,
                    policy=policy,
                    quantization_mode=quantization_mode,
                    status="failed",
                    reason="duplicate_contraction_detected",
                    started=started,
                    task_metrics=task_metrics
                    + [
                        _base_task_metric(
                            case_id,
                            task_index,
                            task,
                            policy,
                            quantization_mode,
                            status="failed",
                            reason="duplicate_contraction_detected",
                            task_started=task_started,
                        )
                    ],
                    schedule_metadata=_schedule_metadata(
                        schedule_mode=schedule_mode,
                        frontier_worker_count=frontier_worker_count,
                        dpu_group_count=dpu_group_count,
                        task_assignment_strategy=task_assignment_strategy,
                        frontier_widths=frontier_widths,
                        scheduler_overhead_s=scheduler_overhead_s,
                        executed_task_count=len(executed_task_ids),
                        duplicate_contraction_check=duplicate_contraction_check,
                        missing_dependency_check=missing_dependency_check,
                        dependency_violation_detected=dependency_violation_detected,
                    ),
                    execution_metadata=execution_metadata,
                )
            missing_dependencies = [dependency for dependency in task.dependencies if dependency not in completed_task_ids]
            if missing_dependencies:
                dependency_violation_detected = True
                missing_dependency_check = "failed"
                return _stop_result(
                    case_id=case_id,
                    policy=policy,
                    quantization_mode=quantization_mode,
                    status="failed",
                    reason="runtime_task_dependency_missing",
                    started=started,
                    task_metrics=task_metrics
                    + [
                        _base_task_metric(
                            case_id,
                            task_index,
                            task,
                            policy,
                            quantization_mode,
                            status="failed",
                            reason=f"missing_dependencies:{','.join(missing_dependencies)}",
                            task_started=task_started,
                        )
                    ],
                    schedule_metadata=_schedule_metadata(
                        schedule_mode=schedule_mode,
                        frontier_worker_count=frontier_worker_count,
                        dpu_group_count=dpu_group_count,
                        task_assignment_strategy=task_assignment_strategy,
                        frontier_widths=frontier_widths,
                        scheduler_overhead_s=scheduler_overhead_s,
                        executed_task_count=len(executed_task_ids),
                        duplicate_contraction_check=duplicate_contraction_check,
                        missing_dependency_check=missing_dependency_check,
                        dependency_violation_detected=dependency_violation_detected,
                    ),
                    execution_metadata=execution_metadata,
                )
            if not _inputs_available(task, tensors):
                missing = [tensor_id for tensor_id in task.input_tensor_ids if tensor_id not in tensors]
                return _stop_result(
                    case_id=case_id,
                    policy=policy,
                    quantization_mode=quantization_mode,
                    status="unsupported",
                    reason="runtime_input_tensor_missing",
                    started=started,
                    task_metrics=task_metrics
                    + [
                        _base_task_metric(
                            case_id,
                            task_index,
                            task,
                            policy,
                            quantization_mode,
                            status="unsupported",
                            reason=f"missing:{','.join(missing)}",
                            task_started=task_started,
                        )
                    ],
                    schedule_metadata=_schedule_metadata(
                        schedule_mode=schedule_mode,
                        frontier_worker_count=frontier_worker_count,
                        dpu_group_count=dpu_group_count,
                        task_assignment_strategy=task_assignment_strategy,
                        frontier_widths=frontier_widths,
                        scheduler_overhead_s=scheduler_overhead_s,
                        executed_task_count=len(executed_task_ids),
                        duplicate_contraction_check=duplicate_contraction_check,
                        missing_dependency_check=missing_dependency_check,
                        dependency_violation_detected=dependency_violation_detected,
                    ),
                    execution_metadata=execution_metadata,
                )

            left_tensor = _tensor_value_for(task.input_tensor_ids[0], task, tensors, labels, side="left")
            right_tensor = _tensor_value_for(task.input_tensor_ids[1], task, tensors, labels, side="right")
            bridge_dir = bridge_root / f"wave_{wave_index:04d}" / f"task_{task_index:04d}" if schedule_mode == "frontier" else bridge_root / f"task_{task_index:04d}"
            task_result = _execute_task_by_policy(
                task=task,
                task_index=task_index,
                case_id=case_id,
                left_tensor=left_tensor,
                right_tensor=right_tensor,
                bridge_dir=bridge_dir,
                policy=policy,
                quantization_mode=quantization_mode,
                execute_external=execute_external,
                env=env,
                task_started=task_started,
            )
            metric = task_result["metric"]
            if schedule_mode == "frontier":
                metric["frontier_wave_index"] = int(wave_index)
                metric["dpu_group_id"] = _assigned_dpu_group(wave_index, task_index, dpu_group_count, task_assignment_strategy)
            task_metrics.append(metric)
            if task_result["status"] != "completed":
                return _stop_result(
                    case_id=case_id,
                    policy=policy,
                    quantization_mode=quantization_mode,
                    status="unsupported" if task_result["status"] == "unsupported" else "failed",
                    reason=str(task_result["reason"]),
                    started=started,
                    task_metrics=task_metrics,
                    schedule_metadata=_schedule_metadata(
                        schedule_mode=schedule_mode,
                        frontier_worker_count=frontier_worker_count,
                        dpu_group_count=dpu_group_count,
                        task_assignment_strategy=task_assignment_strategy,
                        frontier_widths=frontier_widths,
                        scheduler_overhead_s=scheduler_overhead_s,
                        executed_task_count=len(executed_task_ids),
                        duplicate_contraction_check=duplicate_contraction_check,
                        missing_dependency_check=missing_dependency_check,
                        dependency_violation_detected=dependency_violation_detected,
                    ),
                    execution_metadata=execution_metadata,
                )

            output = np.asarray(task_result["output"])
            tensors[task.output_tensor_id] = output
            labels[task.output_tensor_id] = task.output_labels
            live_ids.add(task.output_tensor_id)
            final_labels = task.output_labels
            completed_task_ids.add(task.id)
            executed_task_ids.add(task.id)
            release_dead_inputs(task.input_tensor_ids, task.output_tensor_id, final_tensor_id, tensors, labels, live_ids, remaining_uses)
            peak_live_bytes = max(peak_live_bytes, live_tensor_bytes(tensors, live_ids))
            kernel_family_counts[str(metric["selected_kernel_family"])] = kernel_family_counts.get(str(metric["selected_kernel_family"]), 0) + 1
            backend_counts[str(metric["backend_id"])] = backend_counts.get(str(metric["backend_id"]), 0) + 1
            total_bridge_time_s += float(metric.get("bridge_total_time_s", 0.0) or 0.0)
            total_kernel_time_s += float(metric.get("kernel_time_s", 0.0) or 0.0)
            total_build_time_s += float(metric.get("build_time_s", 0.0) or 0.0)

    if final_tensor_id not in tensors or final_labels is None:
        return _stop_result(
            case_id=case_id,
            policy=policy,
            quantization_mode=quantization_mode,
            status="failed",
            reason="final_tensor_missing",
            started=started,
            task_metrics=task_metrics,
            execution_metadata=execution_metadata,
        )

    final_output, final_transposed = order_final_tensor(np.asarray(tensors[final_tensor_id]), final_labels, graph.network.output_labels)
    final_validation = _final_validation(final_output, reference_output, reference_kind=reference_kind)
    status: UpmemTaskGraphStatus = "completed" if final_validation.get("passed") else "validation_failed"
    reason = None if status == "completed" else "final_validation_failed"
    summary = _summary_payload(
        case_id=case_id,
        policy=policy,
        quantization_mode=quantization_mode,
        status=status,
        reason=reason,
        started=started,
        task_metrics=task_metrics,
        kernel_family_counts=kernel_family_counts,
        backend_counts=backend_counts,
        final_validation=final_validation,
        final_tensor_id=final_tensor_id,
        final_tensor_labels=final_labels,
        final_transpose_applied=final_transposed,
        total_bridge_time_s=total_bridge_time_s,
        total_kernel_time_s=total_kernel_time_s,
        total_build_time_s=total_build_time_s,
        peak_live_tensor_bytes=peak_live_bytes,
        schedule_metadata=_schedule_metadata(
            schedule_mode=schedule_mode,
            frontier_worker_count=frontier_worker_count,
            dpu_group_count=dpu_group_count,
            task_assignment_strategy=task_assignment_strategy,
            frontier_widths=frontier_widths,
            scheduler_overhead_s=scheduler_overhead_s,
            executed_task_count=len(executed_task_ids),
            duplicate_contraction_check=duplicate_contraction_check,
            missing_dependency_check=missing_dependency_check,
            dependency_violation_detected=dependency_violation_detected,
        ),
        execution_metadata=execution_metadata,
    )
    return UpmemTaskGraphRuntimeResult(
        schema_version=UPMEM_TASKGRAPH_RUNTIME_SCHEMA_VERSION,
        status=status,
        reason=reason,
        case_id=case_id,
        policy=policy,
        quantization_mode=quantization_mode,
        output=np.asarray(final_output),
        output_labels=graph.network.output_labels,
        final_validation=final_validation,
        summary=summary,
        task_metrics=tuple(task_metrics),
    )


def _execute_task_by_policy(
    *,
    task: ContractionTask,
    task_index: int,
    case_id: str,
    left_tensor: TensorValue,
    right_tensor: TensorValue,
    bridge_dir: Path,
    policy: str,
    quantization_mode: str,
    execute_external: bool,
    env: Mapping[str, str] | None,
    task_started: float,
) -> JsonDict:
    if policy in {"dense-only", "dense-then-generic"}:
        dense = _execute_dense_task(
            task=task,
            task_index=task_index,
            case_id=case_id,
            left_tensor=left_tensor,
            right_tensor=right_tensor,
            bridge_dir=bridge_dir / "dense",
            policy=policy,
            quantization_mode=quantization_mode,
            execute_external=execute_external,
            env=env,
            task_started=task_started,
        )
        if dense["status"] == "completed" or policy == "dense-only":
            return dense
        dense_reject_reason = str(dense["reason"])
    else:
        dense_reject_reason = "policy_generic_only"

    generic = _execute_generic_task(
        task=task,
        task_index=task_index,
        case_id=case_id,
        left_tensor=left_tensor,
        right_tensor=right_tensor,
        bridge_dir=bridge_dir / "generic",
        policy=policy,
        quantization_mode=quantization_mode,
        execute_external=execute_external,
        env=env,
        task_started=task_started,
        dense_reject_reason=dense_reject_reason,
    )
    return generic


def _execute_dense_task(
    *,
    task: ContractionTask,
    task_index: int,
    case_id: str,
    left_tensor: TensorValue,
    right_tensor: TensorValue,
    bridge_dir: Path,
    policy: str,
    quantization_mode: str,
    execute_external: bool,
    env: Mapping[str, str] | None,
    task_started: float,
) -> JsonDict:
    if quantization_mode == "none":
        return {
            "status": "unsupported",
            "reason": "dense_quantization_none_not_implemented",
            "metric": _base_task_metric(
                case_id,
                task_index,
                task,
                policy,
                quantization_mode,
                status="unsupported",
                reason="dense_quantization_none_not_implemented",
                task_started=task_started,
                selected_kernel_family="dense_gemm",
                backend_id="upmem_sdk_simulator_dense",
            ),
        }
    preparation = prepare_dense_task(DenseTaskPreparationInput(task=task, left_tensor=left_tensor, right_tensor=right_tensor))
    eligible, reason = dense_bridge_backend_manifest_eligibility(preparation, "upmem_sdk_simulator_dense")
    if not eligible:
        return {
            "status": "unsupported",
            "reason": reason or preparation.reason or preparation.status,
            "metric": _base_task_metric(
                case_id,
                task_index,
                task,
                policy,
                quantization_mode,
                status="unsupported",
                reason=reason or preparation.reason or preparation.status,
                task_started=task_started,
                selected_kernel_family="dense_gemm",
                backend_id="upmem_sdk_simulator_dense",
                preparation=preparation.to_json_dict(),
            ),
        }
    write_dense_bridge_input_manifest(preparation, bridge_dir)
    result = execute_dense_bridge(bridge_dir / "input_manifest.json", backend="upmem_sdk_simulator_dense", execute_external=execute_external, env=env)
    output = _load_bridge_output(bridge_dir, result.output_blob_path)
    if result.execution_status != "upmem_sdk_simulator_executed" or output is None:
        return {
            "status": "failed" if result.execution_status == "failed" else "unsupported",
            "reason": result.reason or result.execution_status,
            "metric": _base_task_metric(
                case_id,
                task_index,
                task,
                policy,
                quantization_mode,
                status="failed" if result.execution_status == "failed" else "unsupported",
                reason=result.reason or result.execution_status,
                task_started=task_started,
                selected_kernel_family="dense_gemm",
                backend_id="upmem_sdk_simulator_dense",
                preparation=preparation.to_json_dict(),
                bridge_result=result.to_json_dict(),
            ),
        }
    metric = _base_task_metric(
        case_id,
        task_index,
        task,
        policy,
        quantization_mode,
        status="completed",
        reason=None,
        task_started=task_started,
        selected_kernel_family="dense_gemm",
        backend_id="upmem_sdk_simulator_dense",
        preparation=preparation.to_json_dict(),
        bridge_result=result.to_json_dict(),
        output=output,
        bridge_artifact_path=bridge_dir,
    )
    return {"status": "completed", "reason": None, "output": np.asarray(output), "metric": metric}


def _execute_generic_task(
    *,
    task: ContractionTask,
    task_index: int,
    case_id: str,
    left_tensor: TensorValue,
    right_tensor: TensorValue,
    bridge_dir: Path,
    policy: str,
    quantization_mode: str,
    execute_external: bool,
    env: Mapping[str, str] | None,
    task_started: float,
    dense_reject_reason: str,
) -> JsonDict:
    left_array = np.asarray(left_tensor.array)
    right_array = np.asarray(right_tensor.array)
    has_nonzero_imaginary = (
        (np.iscomplexobj(left_array) and bool(np.any(np.abs(left_array.imag) > 0.0)))
        or (np.iscomplexobj(right_array) and bool(np.any(np.abs(right_array.imag) > 0.0)))
    )
    if has_nonzero_imaginary:
        return _execute_generic_split_complex_task(
            task=task,
            task_index=task_index,
            case_id=case_id,
            left_tensor=left_tensor,
            right_tensor=right_tensor,
            bridge_dir=bridge_dir,
            policy=policy,
            quantization_mode=quantization_mode,
            execute_external=execute_external,
            env=env,
            task_started=task_started,
            dense_reject_reason=dense_reject_reason,
        )
    if np.iscomplexobj(left_array):
        left_tensor = _component_tensor(left_tensor, left_array.real)
    if np.iscomplexobj(right_array):
        right_tensor = _component_tensor(right_tensor, right_array.real)
    return _execute_generic_real_component(
        task=task,
        task_index=task_index,
        case_id=case_id,
        left_tensor=left_tensor,
        right_tensor=right_tensor,
        bridge_dir=bridge_dir,
        policy=policy,
        quantization_mode=quantization_mode,
        execute_external=execute_external,
        env=env,
        task_started=task_started,
        component="real",
        dense_reject_reason=dense_reject_reason,
    )


def _execute_generic_split_complex_task(
    *,
    task: ContractionTask,
    task_index: int,
    case_id: str,
    left_tensor: TensorValue,
    right_tensor: TensorValue,
    bridge_dir: Path,
    policy: str,
    quantization_mode: str,
    execute_external: bool,
    env: Mapping[str, str] | None,
    task_started: float,
    dense_reject_reason: str,
) -> JsonDict:
    left = np.asarray(left_tensor.array)
    right = np.asarray(right_tensor.array)
    components = {
        "ar_br": (left.real, right.real),
        "ai_bi": (left.imag, right.imag),
        "ar_bi": (left.real, right.imag),
        "ai_br": (left.imag, right.real),
    }
    outputs: dict[str, np.ndarray] = {}
    expected: dict[str, np.ndarray] = {}
    component_metrics: dict[str, JsonDict] = {}
    for name, (left_part, right_part) in components.items():
        component_result = _execute_generic_real_component(
            task=task,
            task_index=task_index,
            case_id=case_id,
            left_tensor=_component_tensor(left_tensor, left_part),
            right_tensor=_component_tensor(right_tensor, right_part),
            bridge_dir=bridge_dir / name,
            policy=policy,
            quantization_mode=quantization_mode,
            execute_external=execute_external,
            env=env,
            task_started=task_started,
            component=name,
            dense_reject_reason=dense_reject_reason,
        )
        component_metrics[name] = component_result["metric"]
        if component_result["status"] != "completed":
            metric = _base_task_metric(
                case_id,
                task_index,
                task,
                policy,
                quantization_mode,
                status=component_result["status"],
                reason=f"split_complex_component_{name}:{component_result['reason']}",
                task_started=task_started,
                selected_kernel_family="generic_loop_fallback",
                backend_id="upmem_sdk_simulator_generic_loop",
                component_metrics=component_metrics,
                complex_representation="split_real_imag",
                dense_reject_reason=dense_reject_reason,
            )
            return {"status": component_result["status"], "reason": metric["reason"], "metric": metric}
        outputs[name] = np.asarray(component_result["output"], dtype=np.float64)
        expected[name] = np.asarray(component_result["expected_quantized_reference_output"], dtype=np.float64)

    output = (outputs["ar_br"] - outputs["ai_bi"]) + 1j * (outputs["ar_bi"] + outputs["ai_br"])
    expected_complex = (expected["ar_br"] - expected["ai_bi"]) + 1j * (expected["ar_bi"] + expected["ai_br"])
    validation = conversion_error_metrics(expected_complex, output)
    tolerance = 1.0e-5 if quantization_mode == "none" else 1.0e-8
    metric = _base_task_metric(
        case_id,
        task_index,
        task,
        policy,
        quantization_mode,
        status="completed",
        reason=None,
        task_started=task_started,
        selected_kernel_family="generic_loop_fallback",
        backend_id="upmem_sdk_simulator_generic_loop",
        output=output,
        component_metrics=component_metrics,
        complex_representation="split_real_imag",
        dense_reject_reason=dense_reject_reason,
        validation_metrics={
            "reference_kind": "complex_quantized_dequantized_reference_vs_split_complex_generic",
            "max_abs_error": validation.max_abs_error,
            "l2_error": validation.l2_error,
            "relative_l2_error": validation.relative_l2_error,
            "passed": validation.max_abs_error <= tolerance,
            "max_abs_tolerance": tolerance,
        },
    )
    metric["bridge_artifact_path"] = _metric_artifact_path(bridge_dir)
    metric["split_complex_component_count"] = 4
    return {"status": "completed", "reason": None, "output": output, "metric": metric}


def _execute_generic_real_component(
    *,
    task: ContractionTask,
    task_index: int,
    case_id: str,
    left_tensor: TensorValue,
    right_tensor: TensorValue,
    bridge_dir: Path,
    policy: str,
    quantization_mode: str,
    execute_external: bool,
    env: Mapping[str, str] | None,
    task_started: float,
    component: str,
    dense_reject_reason: str,
) -> JsonDict:
    preparation = prepare_generic_task(
        GenericTaskPreparationInput(
            task=task,
            left_tensor=left_tensor,
            right_tensor=right_tensor,
            quantization_mode=quantization_mode,  # type: ignore[arg-type]
        )
    )
    if preparation.status != "prepared":
        return {
            "status": "unsupported" if preparation.status == "unsupported_shape" else "failed",
            "reason": preparation.reason or preparation.status,
            "metric": _base_task_metric(
                case_id,
                task_index,
                task,
                policy,
                quantization_mode,
                status="unsupported" if preparation.status == "unsupported_shape" else "failed",
                reason=preparation.reason or preparation.status,
                task_started=task_started,
                selected_kernel_family="generic_loop_fallback",
                backend_id="upmem_sdk_simulator_generic_loop",
                preparation=preparation.to_json_dict(),
                component=component,
                dense_reject_reason=dense_reject_reason,
            ),
        }
    write_generic_bridge_input_manifest(preparation, bridge_dir)
    result = execute_generic_bridge(bridge_dir / "input_manifest.json", execute_external=execute_external, env=env)
    output = _load_bridge_output(bridge_dir, result.output_blob_path)
    status = "completed" if result.execution_status == "upmem_sdk_simulator_generic_loop_executed" and output is not None else (
        "failed" if result.execution_status == "failed" else "unsupported"
    )
    reason = None if status == "completed" else result.reason or result.execution_status
    metric = _base_task_metric(
        case_id,
        task_index,
        task,
        policy,
        quantization_mode,
        status=status,
        reason=reason,
        task_started=task_started,
        selected_kernel_family="generic_loop_fallback",
        backend_id="upmem_sdk_simulator_generic_loop",
        preparation=preparation.to_json_dict(),
        bridge_result=result.to_json_dict(),
        output=output,
        bridge_artifact_path=bridge_dir,
        component=component,
        dense_reject_reason=dense_reject_reason,
    )
    expected = None
    if preparation.prepared_operands is not None:
        expected = np.asarray(
            preparation.prepared_operands.expected_reference_output
            if preparation.prepared_operands.expected_reference_output is not None
            else preparation.prepared_operands.expected_quantized_reference_output
        )
    return {"status": status, "reason": reason, "output": output, "metric": metric, "expected_quantized_reference_output": expected}


def _generic_quantized_task_reference(
    *,
    task: ContractionTask,
    task_index: int,
    case_id: str,
    left_tensor: TensorValue,
    right_tensor: TensorValue,
    task_started: float,
    quantization_mode: str,
) -> JsonDict:
    left_array = np.asarray(left_tensor.array)
    right_array = np.asarray(right_tensor.array)
    has_nonzero_imaginary = (
        (np.iscomplexobj(left_array) and bool(np.any(np.abs(left_array.imag) > 0.0)))
        or (np.iscomplexobj(right_array) and bool(np.any(np.abs(right_array.imag) > 0.0)))
    )
    if has_nonzero_imaginary:
        return _generic_quantized_split_complex_reference(
            task=task,
            task_index=task_index,
            case_id=case_id,
            left_tensor=left_tensor,
            right_tensor=right_tensor,
            task_started=task_started,
            quantization_mode=quantization_mode,
        )
    if np.iscomplexobj(left_array):
        left_tensor = _component_tensor(left_tensor, left_array.real)
    if np.iscomplexobj(right_array):
        right_tensor = _component_tensor(right_tensor, right_array.real)
    return _generic_quantized_real_component_reference(
        task=task,
        task_index=task_index,
        case_id=case_id,
        left_tensor=left_tensor,
        right_tensor=right_tensor,
        task_started=task_started,
        component="real",
        quantization_mode=quantization_mode,
    )


def _generic_quantized_split_complex_reference(
    *,
    task: ContractionTask,
    task_index: int,
    case_id: str,
    left_tensor: TensorValue,
    right_tensor: TensorValue,
    task_started: float,
    quantization_mode: str,
) -> JsonDict:
    left = np.asarray(left_tensor.array)
    right = np.asarray(right_tensor.array)
    components = {
        "ar_br": (left.real, right.real),
        "ai_bi": (left.imag, right.imag),
        "ar_bi": (left.real, right.imag),
        "ai_br": (left.imag, right.real),
    }
    expected: dict[str, np.ndarray] = {}
    component_metrics: dict[str, JsonDict] = {}
    for name, (left_part, right_part) in components.items():
        component_result = _generic_quantized_real_component_reference(
            task=task,
            task_index=task_index,
            case_id=case_id,
            left_tensor=_component_tensor(left_tensor, left_part),
            right_tensor=_component_tensor(right_tensor, right_part),
            task_started=task_started,
            component=name,
            quantization_mode=quantization_mode,
        )
        component_metrics[name] = component_result["metric"]
        if component_result["status"] != "completed":
            metric = _generic_reference_base_task_metric(
                case_id,
                task_index,
                task,
                status=component_result["status"],
                reason=f"split_complex_component_{name}:{component_result['reason']}",
                task_started=task_started,
                component_metrics=component_metrics,
                complex_representation="split_real_imag",
            )
            return {"status": component_result["status"], "reason": metric["reason"], "metric": metric}
        expected[name] = np.asarray(component_result["output"], dtype=np.float64)

    output = (expected["ar_br"] - expected["ai_bi"]) + 1j * (expected["ar_bi"] + expected["ai_br"])
    metric = _generic_reference_base_task_metric(
        case_id,
        task_index,
        task,
        status="completed",
        reason=None,
        task_started=task_started,
        output=output,
        component_metrics=component_metrics,
        complex_representation="split_real_imag",
        validation_metrics={
            "reference_kind": "complex_quantized_dequantized_reference_from_four_real_generic_replays",
            "validation_target": "combined_complex_quantized_dequantized_reference",
            "passed": True,
            "max_abs_error": 0.0,
            "l2_error": 0.0,
            "relative_l2_error": 0.0,
        },
    )
    metric["split_complex_component_count"] = 4
    return {"status": "completed", "reason": None, "output": output, "metric": metric}


def _generic_quantized_real_component_reference(
    *,
    task: ContractionTask,
    task_index: int,
    case_id: str,
    left_tensor: TensorValue,
    right_tensor: TensorValue,
    task_started: float,
    component: str,
    quantization_mode: str,
) -> JsonDict:
    preparation = prepare_generic_task(
        GenericTaskPreparationInput(
            task=task,
            left_tensor=left_tensor,
            right_tensor=right_tensor,
            quantization_mode=quantization_mode,  # type: ignore[arg-type]
        )
    )
    if preparation.status != "prepared" or preparation.prepared_operands is None:
        status = "unsupported" if preparation.status == "unsupported_shape" else "failed"
        return {
            "status": status,
            "reason": preparation.reason or preparation.status,
            "metric": _generic_reference_base_task_metric(
                case_id,
                task_index,
                task,
                status=status,
                reason=preparation.reason or preparation.status,
                task_started=task_started,
                preparation=preparation.to_json_dict(),
                component=component,
            ),
        }

    output = np.asarray(preparation.prepared_operands.expected_quantized_reference_output)
    return {
        "status": "completed",
        "reason": None,
        "output": output,
        "metric": _generic_reference_base_task_metric(
            case_id,
            task_index,
            task,
            status="completed",
            reason=None,
            task_started=task_started,
            preparation=preparation.to_json_dict(),
            output=output,
            component=component,
            validation_metrics={
                **dict(preparation.validation_metrics),
                "validation_target": preparation.metadata.get("validation_target", "expected_quantized_reference_output"),
            },
        ),
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
) -> JsonDict:
    output_array = np.asarray(output) if output is not None else None
    native_metadata = dict((preparation or {}).get("native_index_metadata") or {})
    prep_metadata = dict((preparation or {}).get("metadata") or {})
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
            "complex_representation": complex_representation or ("real" if output_array is not None and not np.iscomplexobj(output_array) else None),
            "complex_quantization_scope": "per_task_operands" if complex_representation else None,
            "reference_kind": "generic_quantized_task_reference",
            "selected_kernel_family": prep_metadata.get("kernel_family", "generic_loop_fallback"),
            "validation_target": prep_metadata.get("validation_target", "expected_quantized_reference_output"),
            "full_precision_reference_is_validation_target": False,
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
    full_precision_h2d_bytes = prep_metadata.get("full_precision_h2d_bytes_model", bridge_metadata.get("full_precision_h2d_bytes_model"))
    full_precision_d2h_bytes = prep_metadata.get("full_precision_d2h_bytes_model", bridge_metadata.get("full_precision_d2h_bytes_model"))
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
            "complex_representation": complex_representation or ("real" if output_array is not None and not np.iscomplexobj(output_array) else None),
            "complex_quantization_scope": "per_task_operands" if complex_representation else None,
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
            "actual_h2d_bytes": int(actual_h2d_bytes or 0),
            "actual_d2h_bytes": int(actual_d2h_bytes or 0),
            "actual_transfer_bytes": int((actual_h2d_bytes or 0) + (actual_d2h_bytes or 0)),
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


def _assigned_dpu_group(wave_index: int, task_index: int, dpu_group_count: int, task_assignment_strategy: str) -> int:
    if dpu_group_count <= 1:
        return 0
    if task_assignment_strategy == "sequential_single_dpu":
        return 0
    if task_assignment_strategy == "frontier_size_aware_dpu_groups":
        return int(task_index % dpu_group_count)
    return int((wave_index + task_index) % dpu_group_count)


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
            "total_wall_time_s": float(time.perf_counter() - started),
            "total_bridge_time_s": float(total_bridge_time_s),
            "total_kernel_time_s": float(total_kernel_time_s),
            "total_build_time_s": float(total_build_time_s),
            "total_quantization_time_s": float(sum(float(row.get("quantization_time_s", 0.0) or 0.0) for row in task_metrics)),
            "total_dequantization_time_s": float(sum(float(row.get("dequantization_time_s", 0.0) or 0.0) for row in task_metrics)),
            "total_float32_reference_time_s": float(sum(float(row.get("float32_reference_time_s", 0.0) or 0.0) for row in task_metrics)),
            "actual_h2d_bytes": int(total_actual_h2d_bytes),
            "actual_d2h_bytes": int(total_actual_d2h_bytes),
            "actual_transfer_bytes": int(total_actual_transfer_bytes),
            "full_precision_h2d_bytes_model": int(total_full_precision_h2d_bytes),
            "full_precision_d2h_bytes_model": int(total_full_precision_d2h_bytes),
            "full_precision_transfer_bytes_model": int(total_full_precision_transfer_bytes),
            "transfer_compression_ratio": transfer_compression_ratio,
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


def _final_validation(output: np.ndarray, reference_output: np.ndarray | None, *, reference_kind: str) -> JsonDict:
    if reference_output is None:
        return {"passed": False, "reason": "reference_output_missing", "reference_kind": reference_kind}
    result = validate(output, reference_output, QUANTIZED_FINAL_VALIDATION_TOLERANCES)
    diff = np.asarray(output, dtype=np.complex128) - np.asarray(reference_output, dtype=np.complex128)
    abs_diff = np.abs(diff)
    return to_jsonable(
        {
            **result.__dict__,
            "reference_kind": reference_kind,
            "tolerance_kind": "quantized_execution_tolerance",
            "mean_abs_error": float(abs_diff.mean()) if abs_diff.size else 0.0,
            "max_abs_error": result.max_abs_error,
            "l2_error": result.l2_error,
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


def _inputs_available(task: ContractionTask, tensors: Mapping[str, np.ndarray]) -> bool:
    return all(tensor_id in tensors for tensor_id in task.input_tensor_ids)


def _tensor_value_for(
    tensor_id: str,
    task: ContractionTask,
    tensors: Mapping[str, np.ndarray],
    labels: Mapping[str, tuple[int, ...]],
    *,
    side: str,
) -> TensorValue:
    array = np.asarray(tensors[tensor_id])
    expected_shape = task.input_shapes[0] if side == "left" else task.input_shapes[1]
    return TensorValue(
        TensorSpec(tensor_id, labels[tensor_id], tuple(int(dim) for dim in expected_shape), "dense", dtype=str(array.dtype)),
        array,
    )


def _component_tensor(tensor: TensorValue, array: np.ndarray) -> TensorValue:
    return TensorValue(
        TensorSpec(tensor.spec.id, tensor.spec.labels, tensor.spec.shape, tensor.spec.structure, dtype=str(np.asarray(array).dtype)),
        np.asarray(array, dtype=np.float64),
    )


def _load_bridge_output(bridge_dir: Path, relative_path: str | None) -> np.ndarray | None:
    if not relative_path:
        return None
    path = Path(relative_path)
    if path.is_absolute():
        raise ValueError(f"bridge output path must be relative: {relative_path}")
    resolved = (bridge_dir / path).resolve()
    try:
        resolved.relative_to(bridge_dir.resolve())
    except ValueError as exc:
        raise ValueError(f"bridge output path escapes bridge directory: {relative_path}") from exc
    if not resolved.exists():
        return None
    return np.load(resolved, allow_pickle=False)


def _metric_artifact_path(path: Path | None) -> str | None:
    if path is None:
        return None
    parts = path.parts
    for index in range(len(parts) - 1, -1, -1):
        if parts[index] == "cases":
            return Path(*parts[index:]).as_posix()
    return path.name
