from __future__ import annotations

import time
from pathlib import Path
from typing import Literal, Mapping

import numpy as np

from quantum_bench.core.records import ContractionTask, JsonDict, TensorValue
from quantum_bench.routing import GenericTaskPreparationInput, prepare_generic_task
from quantum_bench.targets.upmem.evidence import (
    CONTRACTION_EXECUTION_TARGET_UPMEM,
    UPMEM_EXECUTION_MODE_SDK_SIMULATOR,
)
from quantum_bench.targets.upmem.generic_bridge import execute_generic_bridge, write_generic_bridge_input_manifest
import quantum_bench.targets.upmem.numeric_reference as _numeric_reference
import quantum_bench.targets.upmem.runtime_evidence as _runtime_evidence
from quantum_bench.targets.upmem.numeric_reference import (
    _component_tensor,
    _complex_split_reference_metrics,
    _final_validation,
    _full_precision_accuracy,
    _tensor_value_for,
)
from quantum_bench.targets.upmem.runtime_evidence import (
    _base_task_metric,
    _metric_artifact_path,
    _schedule_metadata,
    _stop_result,
    _summary_payload,
    _unsupported_result,
)
from quantum_bench.routing.generic_numeric_contract import classify_numeric
from quantum_bench.tn.execution import live_tensor_bytes, order_final_tensor, release_dead_inputs, remaining_input_uses
from quantum_bench.tn.execution_bundle import execution_identity_metadata, executor_config_hash, with_execution_identity
from quantum_bench.tn.network import TensorNetworkValue


QUANTIZED_FINAL_VALIDATION_TOLERANCES = _numeric_reference.QUANTIZED_FINAL_VALIDATION_TOLERANCES
GenericQuantizedTaskGraphReference = _numeric_reference.GenericQuantizedTaskGraphReference
build_generic_quantized_taskgraph_reference = _numeric_reference.build_generic_quantized_taskgraph_reference
build_generic_taskgraph_reference = _numeric_reference.build_generic_taskgraph_reference
GENERIC_QUANTIZED_TASKGRAPH_REFERENCE_SCHEMA_VERSION = _runtime_evidence.GENERIC_QUANTIZED_TASKGRAPH_REFERENCE_SCHEMA_VERSION
GENERIC_QUANTIZED_TASKGRAPH_REFERENCE_TASK_SCHEMA_VERSION = _runtime_evidence.GENERIC_QUANTIZED_TASKGRAPH_REFERENCE_TASK_SCHEMA_VERSION
UPMEM_TASKGRAPH_RUNTIME_SCHEMA_VERSION = _runtime_evidence.UPMEM_TASKGRAPH_RUNTIME_SCHEMA_VERSION
UPMEM_TASKGRAPH_TASK_METRIC_SCHEMA_VERSION = _runtime_evidence.UPMEM_TASKGRAPH_TASK_METRIC_SCHEMA_VERSION
UpmemTaskGraphRuntimeResult = _runtime_evidence.UpmemTaskGraphRuntimeResult
UpmemTaskGraphStatus = _runtime_evidence.UpmemTaskGraphStatus


UPMEM_TASKGRAPH_POLICIES = ("generic-only",)
UPMEM_TASKGRAPH_QUANTIZATION_MODES = ("per_task_input_quantize", "none")
GENERIC_KERNEL_STRATEGY = _runtime_evidence.GENERIC_KERNEL_STRATEGY
GENERIC_NATIVE_MAX_RANK = _runtime_evidence.GENERIC_NATIVE_MAX_RANK
GENERIC_NATIVE_MAX_TENSOR_ELEMENTS = _runtime_evidence.GENERIC_NATIVE_MAX_TENSOR_ELEMENTS
GENERIC_OUTPUT_TILE_ELEMENTS = _runtime_evidence.GENERIC_OUTPUT_TILE_ELEMENTS

UpmemTaskGraphPolicy = Literal["generic-only"]
UpmemTaskGraphQuantizationMode = Literal["per_task_input_quantize", "none"]


def upmem_taskgraph_executor_config(
    *,
    policy: str,
    quantization_mode: str,
    schedule_mode: str = "sequential",
    frontier_worker_count: int = 1,
    dpu_group_count: int = 1,
    task_assignment_strategy: str = "sequential_single_dpu",
    tasklets_per_dpu: int = 1,
) -> JsonDict:
    return {
        "policy": policy,
        "quantization_mode": quantization_mode,
        "schedule_mode": schedule_mode,
        "frontier_worker_count": frontier_worker_count,
        "dpu_group_count": dpu_group_count,
        "task_assignment_strategy": task_assignment_strategy,
        "tasklets_per_dpu": tasklets_per_dpu,
        "generic_kernel_strategy": GENERIC_KERNEL_STRATEGY,
        "native_max_rank": GENERIC_NATIVE_MAX_RANK,
        "native_max_tensor_elements": GENERIC_NATIVE_MAX_TENSOR_ELEMENTS,
        "generic_output_tile_elements": GENERIC_OUTPUT_TILE_ELEMENTS,
    }
UpmemTaskGraphScheduleMode = Literal["sequential"]

CONTRACTION_EXECUTION_TARGET = CONTRACTION_EXECUTION_TARGET_UPMEM
UPMEM_EXECUTION_MODE = UPMEM_EXECUTION_MODE_SDK_SIMULATOR


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
    full_precision_reference_output: np.ndarray | None = None,
    full_precision_reference_kind: str = "cpu_exact_taskgraph_full_precision",
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
    if schedule_mode != "sequential":
        return _unsupported_result(case_id, policy, quantization_mode, f"unsupported_schedule_mode:{schedule_mode}", started, execution_metadata)
    if frontier_worker_count < 1:
        return _unsupported_result(case_id, policy, quantization_mode, "frontier_worker_count_must_be_positive", started, execution_metadata)
    if dpu_group_count < 1:
        return _unsupported_result(case_id, policy, quantization_mode, "dpu_group_count_must_be_positive", started, execution_metadata)
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
    schedule_waves = tuple((task,) for task in graph.tasks)
    scheduler_overhead_s = time.perf_counter() - scheduler_started
    frontier_widths = tuple(len(wave) for wave in schedule_waves)
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
            bridge_dir = bridge_root / f"task_{task_index:04d}"
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
    final_full_precision_accuracy = _full_precision_accuracy(
        final_output,
        full_precision_reference_output,
        reference_kind=full_precision_reference_kind,
    )
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
        final_full_precision_accuracy=final_full_precision_accuracy,
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
    return _execute_generic_task(
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
        dense_reject_reason="policy_generic_only",
    )


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
    left_classification = classify_numeric(left_array)
    right_classification = classify_numeric(right_array)
    if left_classification.has_nonfinite or right_classification.has_nonfinite:
        metric = _base_task_metric(
            case_id, task_index, task, policy, quantization_mode,
            status="unsupported", reason="nonfinite_values_not_supported",
            task_started=task_started, selected_kernel_family="generic_loop_fallback",
            backend_id="upmem_sdk_simulator_generic_loop", dense_reject_reason=dense_reject_reason,
        )
        return {"status": "unsupported", "reason": metric["reason"], "metric": metric}
    has_nonzero_imaginary = left_classification.has_nonzero_imaginary or right_classification.has_nonzero_imaginary
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

    output, validation_metrics, full_precision_metrics = _complex_split_reference_metrics(
        task=task,
        left=left,
        right=right,
        outputs=outputs,
        expected=expected,
        quantization_mode=quantization_mode,
    )
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
        validation_metrics=validation_metrics,
        full_precision_metrics=full_precision_metrics,
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
        dpu_run_time_cycles=getattr(result, "dpu_run_time_cycles", 0),
    )
    expected = None
    if preparation.prepared_operands is not None:
        expected = np.asarray(
            preparation.prepared_operands.expected_reference_output
            if preparation.prepared_operands.expected_reference_output is not None
            else preparation.prepared_operands.expected_quantized_reference_output
        )
    return {"status": status, "reason": reason, "output": output, "metric": metric, "expected_quantized_reference_output": expected}



def _inputs_available(task: ContractionTask, tensors: Mapping[str, np.ndarray]) -> bool:
    return all(tensor_id in tensors for tensor_id in task.input_tensor_ids)


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
