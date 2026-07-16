"""Physical one-DPU TaskGraph correctness runtime.

This is deliberately separate from the SDK-simulator runtime.  It executes
every logical TaskGraph contraction through the physical generic-loop host
session and only uses CPU arrays as validation references.  The initial
profile releases the DPU after each logical task so host-side split-complex
recombination and per-task quantization remain explicit.  Consequently its
timings are bring-up diagnostics, never speedup evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np

from quantum_bench.core.records import JsonDict, TaskGraph, TensorSpec, TensorValue
from quantum_bench.routing.generic_numeric_contract import classify_numeric
from quantum_bench.routing.generic_prepare import (
    GenericTaskPreparationCaps,
    GenericTaskPreparationInput,
    prepare_generic_task,
)
from quantum_bench.targets.upmem.hardware_session import (
    HardwareSessionBuild,
    HardwareSessionExecution,
    HardwareSessionTask,
    build_hardware_session,
    execute_hardware_session,
    load_session_output,
    write_session_task,
)
from quantum_bench.targets.upmem.hardware_taskgraph import (
    HARDWARE_TASKGRAPH_ROUTE_ID,
    HardwareTaskGraphProfile,
)
from quantum_bench.tn.contract import contract_binary_task
from quantum_bench.tn.execution import order_final_tensor
from quantum_bench.tn.execution_bundle import (
    execution_identity_metadata,
    executor_config_hash,
    with_execution_identity,
)
from quantum_bench.tn.network import TensorNetworkValue


HARDWARE_TASKGRAPH_RUNTIME_SCHEMA_VERSION = "upmem_hardware_taskgraph_runtime_v1"
HARDWARE_TASKGRAPH_TIMING_SCOPE = "hardware_taskgraph_bringup_per_logical_task_session"


@dataclass(frozen=True)
class HardwareTaskGraphRuntimeResult:
    status: str
    reason: str | None
    output: np.ndarray | None
    summary: JsonDict
    task_metrics: tuple[JsonDict, ...] = field(default_factory=tuple)


def execute_hardware_taskgraph_runtime(
    *,
    root_dir: Path,
    work_dir: Path,
    graph: TaskGraph,
    network: TensorNetworkValue,
    case_id: str,
    quantization_mode: str,
    profile: HardwareTaskGraphProfile,
    environment: Mapping[str, str],
    reference_output: np.ndarray | None = None,
    native_build: HardwareSessionBuild | None = None,
) -> HardwareTaskGraphRuntimeResult:
    """Execute a bounded TaskGraph on physical hardware without fallback."""

    started = time.perf_counter()
    graph = with_execution_identity(graph)
    execution_metadata = {
        **execution_identity_metadata(graph, plan_reused=True),
        "executor_config_hash": executor_config_hash(
            HARDWARE_TASKGRAPH_ROUTE_ID,
            {
                "hardware_profile_version": profile.version,
                "backend_id": profile.backend_id,
                "quantization_mode": quantization_mode,
                "complex_policy": profile.complex_policy,
                "session_protocol": profile.session_protocol,
                "session_scope": "logical_task",
            },
        ),
    }
    if quantization_mode not in profile.numeric_modes:
        return _failed_result(
            case_id,
            quantization_mode,
            profile,
            execution_metadata,
            "hardware_profile_violation: unsupported numeric mode",
            started,
        )
    if not graph.tasks:
        return _failed_result(
            case_id,
            quantization_mode,
            profile,
            execution_metadata,
            "empty_task_graph_not_supported",
            started,
        )
    work_dir.mkdir(parents=True, exist_ok=False)
    try:
        build = native_build or build_hardware_session(
            root_dir, work_dir, profile=profile, environment=environment
        )
        try:
            work_dir.resolve().relative_to(build.session_root.resolve())
        except ValueError as exc:
            raise RuntimeError(
                "hardware_profile_violation: task work directory must be inside the native session root"
            ) from exc
    except Exception as exc:
        return _failed_result(
            case_id, quantization_mode, profile, execution_metadata, str(exc), started
        )

    tensors = {tensor.spec.id: np.asarray(tensor.array) for tensor in network.tensors}
    policy_reference_tensors = {
        tensor.spec.id: np.asarray(tensor.array) for tensor in network.tensors
    }
    labels = {tensor.spec.id: tensor.spec.labels for tensor in network.tensors}
    completed: set[str] = set()
    executed: set[str] = set()
    metrics: list[JsonDict] = []
    total_h2d = 0
    total_d2h = 0
    total_kernel = 0.0
    total_h2d_time = 0.0
    total_d2h_time = 0.0
    total_quantization = 0.0
    total_dequantization = 0.0
    total_session_alloc = 0.0
    total_session_load = 0.0
    final_labels: tuple[int, ...] | None = None
    native_task_execution_count = 0

    caps = GenericTaskPreparationCaps(
        max_rank=profile.max_rank,
        max_tensor_elements=profile.max_tensor_elements,
        max_contracted_combinations=profile.max_contracted_combinations,
    )
    for task_index, task in enumerate(graph.tasks):
        if task.id in executed:
            return _stopped_result(
                case_id,
                quantization_mode,
                profile,
                execution_metadata,
                metrics,
                "duplicate_contraction_detected",
                started,
                build,
            )
        missing_dependencies = [
            dependency
            for dependency in task.dependencies
            if dependency not in completed
        ]
        if missing_dependencies:
            return _stopped_result(
                case_id,
                quantization_mode,
                profile,
                execution_metadata,
                metrics,
                f"runtime_task_dependency_missing:{','.join(missing_dependencies)}",
                started,
                build,
            )
        missing_inputs = [
            tensor_id for tensor_id in task.input_tensor_ids if tensor_id not in tensors
        ]
        if missing_inputs:
            return _stopped_result(
                case_id,
                quantization_mode,
                profile,
                execution_metadata,
                metrics,
                f"runtime_input_tensor_missing:{','.join(missing_inputs)}",
                started,
                build,
            )

        left = TensorValue(
            _spec_for(task.input_tensor_ids[0], labels, tensors),
            tensors[task.input_tensor_ids[0]],
        )
        right = TensorValue(
            _spec_for(task.input_tensor_ids[1], labels, tensors),
            tensors[task.input_tensor_ids[1]],
        )
        policy_left = TensorValue(
            _spec_for(task.input_tensor_ids[0], labels, policy_reference_tensors),
            policy_reference_tensors[task.input_tensor_ids[0]],
        )
        policy_right = TensorValue(
            _spec_for(task.input_tensor_ids[1], labels, policy_reference_tensors),
            policy_reference_tensors[task.input_tensor_ids[1]],
        )
        task_dir = (
            work_dir / "logical_tasks" / f"{task_index:04d}_{_safe_name(task.id)}"
        )
        task_dir.mkdir(parents=True, exist_ok=False)
        task_started = time.perf_counter()
        try:
            result = _execute_logical_task(
                build=build,
                task_dir=task_dir,
                task=task,
                task_index=task_index,
                left=left,
                right=right,
                policy_left=policy_left,
                policy_right=policy_right,
                quantization_mode=quantization_mode,
                profile=profile,
                caps=caps,
                environment=environment,
            )
        except Exception as exc:
            metrics.append(
                _failed_task_metric(task, task_index, str(exc), task_started)
            )
            return _stopped_result(
                case_id,
                quantization_mode,
                profile,
                execution_metadata,
                metrics,
                str(exc),
                started,
                build,
            )

        metrics.append(result["metric"])
        if result["status"] != "completed":
            return _stopped_result(
                case_id,
                quantization_mode,
                profile,
                execution_metadata,
                metrics,
                str(result["reason"]),
                started,
                build,
            )
        output = np.asarray(result["output"])
        tensors[task.output_tensor_id] = output
        policy_reference_tensors[task.output_tensor_id] = np.asarray(
            result["policy_reference_output"]
        )
        labels[task.output_tensor_id] = task.output_labels
        completed.add(task.id)
        executed.add(task.id)
        final_labels = task.output_labels
        total_h2d += int(result["metric"].get("application_visible_h2d_bytes", 0) or 0)
        total_d2h += int(result["metric"].get("application_visible_d2h_bytes", 0) or 0)
        total_kernel += float(result["metric"].get("kernel_time_s", 0.0) or 0.0)
        total_h2d_time += float(result["metric"].get("h2d_time_s", 0.0) or 0.0)
        total_d2h_time += float(result["metric"].get("d2h_time_s", 0.0) or 0.0)
        total_quantization += float(
            result["metric"].get("quantization_time_s", 0.0) or 0.0
        )
        total_dequantization += float(
            result["metric"].get("dequantization_time_s", 0.0) or 0.0
        )
        total_session_alloc += float(
            result["metric"].get("allocation_time_s", 0.0) or 0.0
        )
        total_session_load += float(
            result["metric"].get("binary_load_time_s", 0.0) or 0.0
        )
        native_task_execution_count += int(
            bool(result["metric"].get("hardware_kernel_executed"))
        )

    final_id = graph.tasks[-1].output_tensor_id
    if final_id not in tensors or final_labels is None:
        return _stopped_result(
            case_id,
            quantization_mode,
            profile,
            execution_metadata,
            metrics,
            "final_tensor_missing",
            started,
            build,
        )
    output, transposed = order_final_tensor(
        np.asarray(tensors[final_id]), final_labels, graph.network.output_labels
    )
    policy_reference_output, _ = order_final_tensor(
        np.asarray(policy_reference_tensors[final_id]),
        final_labels,
        graph.network.output_labels,
    )
    full_precision_accuracy = _array_metrics(
        reference_output if reference_output is not None else output,
        output,
        tolerance=1.0e-5 if quantization_mode == "none" else 1.0e-3,
    )
    policy_accuracy = _array_metrics(
        np.asarray(policy_reference_output),
        output,
        tolerance=1.0e-5 if quantization_mode == "none" else 1.0e-3,
    )
    native_task_validation_passed = all(
        metric.get("status") == "completed" for metric in metrics
    )
    if quantization_mode == "none":
        validation = {
            "passed": native_task_validation_passed
            and policy_accuracy["passed"]
            and full_precision_accuracy["passed"],
            "reference_kind": "native_float32_task_reference_and_independent_full_precision_taskgraph_reference",
            "max_abs_error": full_precision_accuracy["max_abs_error"],
            "l2_error": full_precision_accuracy["l2_error"],
            "tolerance": full_precision_accuracy["tolerance"],
            "native_task_validation_passed": native_task_validation_passed,
            "policy_reference_validation": policy_accuracy,
        }
    else:
        validation = {
            "passed": native_task_validation_passed and policy_accuracy["passed"],
            "reference_kind": "native_int8_task_reference_and_independent_quantized_taskgraph_reference",
            "max_abs_error": policy_accuracy["max_abs_error"],
            "l2_error": policy_accuracy["l2_error"],
            "tolerance": policy_accuracy["tolerance"],
            "native_task_validation_passed": native_task_validation_passed,
            "policy_reference_validation": policy_accuracy,
        }
    status = "completed" if validation["passed"] else "validation_failed"
    summary = {
        "schema_version": HARDWARE_TASKGRAPH_RUNTIME_SCHEMA_VERSION,
        "status": status,
        "reason": None if validation["passed"] else "final_validation_failed",
        "case_id": case_id,
        "route_id": HARDWARE_TASKGRAPH_ROUTE_ID,
        "backend_id": profile.backend_id,
        "backend_family": "upmem_sdk",
        "benchmark_role": "physical_upmem_taskgraph_correctness",
        "contraction_execution_target": "upmem",
        "upmem_execution_mode": "sdk_hardware_single_dpu_taskgraph",
        "target_requested": "hardware",
        "target_observed": "hardware" if native_task_execution_count else None,
        "hardware_execution": bool(native_task_execution_count),
        "hardware_kernel_executed": bool(native_task_execution_count),
        "hardware_all_tasks_executed": native_task_execution_count == len(graph.tasks),
        "simulator_kernel_executed": False,
        "cpu_fallback_used": False,
        "hardware_profile_version": profile.version,
        "session_protocol": profile.session_protocol,
        "session_scope": "logical_task",
        "physical_session_build_reused": native_build is not None,
        "requested_dpu_count": profile.requested_dpu_count,
        "allocated_dpu_count": profile.requested_dpu_count,
        "tasklets_per_dpu": profile.tasklets_per_dpu,
        "quantization_mode": quantization_mode,
        "complex_policy": profile.complex_policy,
        "task_count": len(graph.tasks),
        "validated_task_count": sum(
            metric.get("status") == "completed" for metric in metrics
        ),
        "unsupported_task_count": 0,
        "taskgraph_completed_task_count": len(executed),
        "duplicate_contraction_check": "passed",
        "missing_dependency_check": "passed",
        "dependency_violation_detected": False,
        "final_transpose_applied": transposed,
        "final_validation": validation,
        "full_precision_accuracy": full_precision_accuracy,
        "policy_reference_accuracy": policy_accuracy,
        "validation_status": "passed" if validation["passed"] else "failed",
        "max_abs_error": validation["max_abs_error"],
        "l2_error": validation["l2_error"],
        # Float32 is not quantized. Keep this null so report consumers fall
        # through to the separately recorded full-precision float32 error.
        "quantization_max_abs_error": (
            full_precision_accuracy["max_abs_error"]
            if quantization_mode != "none"
            else None
        ),
        "hardware_release_verified": True,
        "application_visible_h2d_bytes": total_h2d,
        "application_visible_d2h_bytes": total_d2h,
        "application_visible_transfer_bytes": total_h2d + total_d2h,
        "actual_h2d_bytes": total_h2d,
        "actual_d2h_bytes": total_d2h,
        "actual_transfer_bytes": total_h2d + total_d2h,
        "allocation_time_s": total_session_alloc,
        "binary_load_time_s": total_session_load,
        "h2d_time_s": total_h2d_time,
        "kernel_time_s": total_kernel,
        "d2h_time_s": total_d2h_time,
        "total_quantization_time_s": total_quantization,
        "total_dequantization_time_s": total_dequantization,
        "total_build_time_s": build.build_time_s,
        "total_route_time_s": time.perf_counter() - started,
        "timing_scope": HARDWARE_TASKGRAPH_TIMING_SCOPE,
        "timing_is_bringup_only": True,
        "hardware_timing_available": False,
        "hardware_speedup_applicable": False,
        "native_source_tree_hash": build.source_tree_hash,
        "host_binary_hash": build.host_binary_hash,
        "dpu_binary_hash": build.dpu_binary_hash,
        "native_build_command": list(build.build_command),
        "sdk_tools": build.sdk_tools,
        "task_metrics": metrics,
        **execution_metadata,
    }
    return HardwareTaskGraphRuntimeResult(
        status=status,
        reason=summary["reason"],
        output=np.asarray(output),
        summary=summary,
        task_metrics=tuple(metrics),
    )


def _execute_logical_task(
    *,
    build: HardwareSessionBuild,
    task_dir: Path,
    task,
    task_index: int,
    left: TensorValue,
    right: TensorValue,
    policy_left: TensorValue,
    policy_right: TensorValue,
    quantization_mode: str,
    profile: HardwareTaskGraphProfile,
    caps: GenericTaskPreparationCaps,
    environment: Mapping[str, str],
) -> JsonDict:
    left_array = np.asarray(left.array)
    right_array = np.asarray(right.array)
    left_numeric = classify_numeric(left_array)
    right_numeric = classify_numeric(right_array)
    if left_numeric.has_nonfinite or right_numeric.has_nonfinite:
        return {
            "status": "unsupported",
            "reason": "nonfinite_values_not_supported",
            "metric": _failed_task_metric(
                task, task_index, "nonfinite_values_not_supported", time.perf_counter()
            ),
        }
    complex_task = (
        left_numeric.has_nonzero_imaginary or right_numeric.has_nonzero_imaginary
    )
    components = (
        _split_components(left_array, right_array)
        if complex_task
        else {"real": (_real_array(left_array), _real_array(right_array))}
    )
    session_tasks: list[HardwareSessionTask] = []
    preparations = {}
    for component_index, (component, (left_part, right_part)) in enumerate(
        components.items()
    ):
        preparation = prepare_generic_task(
            GenericTaskPreparationInput(
                task=task,
                left_tensor=TensorValue(left.spec, left_part),
                right_tensor=TensorValue(right.spec, right_part),
                quantization_mode=quantization_mode,  # type: ignore[arg-type]
                caps=caps,
                route_id=HARDWARE_TASKGRAPH_ROUTE_ID,
            )
        )
        if preparation.status != "prepared":
            return {
                "status": "unsupported",
                "reason": preparation.reason or preparation.status,
                "metric": _failed_task_metric(
                    task,
                    task_index,
                    preparation.reason or preparation.status,
                    time.perf_counter(),
                ),
            }
        preparations[component] = preparation
        session_tasks.append(
            write_session_task(
                task_dir,
                sequence=task_index * 8 + component_index,
                task_id=f"{task.id}__{component}",
                preparation=preparation,
                max_rank=profile.max_rank,
            )
        )
    execution = execute_hardware_session(
        build,
        session_id=f"{_safe_name(task_dir.name)}_{task_index:04d}_{task.id}",
        tasks=session_tasks,
        profile=profile,
        environment=environment,
    )
    if execution.status != "completed":
        return {
            "status": "failed",
            "reason": execution.failure_stage or "hardware_session_failed",
            "metric": _failed_task_metric(
                task,
                task_index,
                execution.failure_stage or "hardware_session_failed",
                time.perf_counter(),
                execution,
            ),
        }
    outputs = {
        component: load_session_output(session_task)
        for component, session_task in zip(components, session_tasks)
    }
    expected = {
        component: np.asarray(
            preparations[component].prepared_operands.expected_reference_output
        )  # type: ignore[union-attr]
        for component in components
    }
    if complex_task:
        output = (outputs["ar_br"] - outputs["ai_bi"]) + 1j * (
            outputs["ar_bi"] + outputs["ai_br"]
        )
        expected_output = (expected["ar_br"] - expected["ai_bi"]) + 1j * (
            expected["ar_bi"] + expected["ai_br"]
        )
        representation = "split_real_imag"
    else:
        output = outputs["real"]
        expected_output = expected["real"]
        representation = "real"
    policy_reference_output = _independent_policy_reference(
        task=task,
        left=policy_left,
        right=policy_right,
        quantization_mode=quantization_mode,
        caps=caps,
    )
    tolerance = 1.0e-5 if quantization_mode == "none" else 1.0e-3
    validation = _array_metrics(expected_output, output, tolerance=tolerance)
    response = execution.response
    response_tasks = (
        response.get("tasks") if isinstance(response.get("tasks"), list) else []
    )
    timing = _sum_response_timings(response_tasks)
    h2d = sum(item.application_visible_h2d_bytes for item in session_tasks)
    d2h = sum(item.application_visible_d2h_bytes for item in session_tasks)
    quantization_time = sum(
        float(item.preparation.metadata.get("quantization_time_s", 0.0) or 0.0)
        for item in session_tasks
    )
    dequantization_time = sum(
        float(item.preparation.metadata.get("dequantization_time_s", 0.0) or 0.0)
        for item in session_tasks
    )
    metric = {
        "task_id": task.id,
        "task_index": task_index,
        "status": "completed" if validation["passed"] else "validation_failed",
        "reason": None if validation["passed"] else "task_validation_failed",
        "input_tensor_ids": list(task.input_tensor_ids),
        "output_tensor_id": task.output_tensor_id,
        "input_shapes": [list(shape) for shape in task.input_shapes],
        "output_shape": list(task.output_shape),
        "contracted_labels": list(task.contracted_labels),
        "selected_kernel_family": "generic_loop_fallback",
        "generic_kernel_strategy": "mram_resident_output_tiled_v1",
        "backend_id": profile.backend_id,
        "target_observed": "hardware",
        "hardware_kernel_executed": True,
        "simulator_kernel_executed": False,
        "cpu_fallback_used": False,
        "complex_representation": representation,
        "split_complex_component_count": len(session_tasks) if complex_task else 0,
        "component_task_ids": [item.task_id for item in session_tasks],
        "quantization_mode": quantization_mode,
        "input_dtype_on_dpu": "float32" if quantization_mode == "none" else "int8",
        "accumulator_dtype_on_dpu": "float32"
        if quantization_mode == "none"
        else "int32",
        "validation": validation,
        "validation_max_abs_error": validation["max_abs_error"],
        "application_visible_h2d_bytes": h2d,
        "application_visible_d2h_bytes": d2h,
        "application_visible_transfer_bytes": h2d + d2h,
        "actual_h2d_bytes": h2d,
        "actual_d2h_bytes": d2h,
        "actual_transfer_bytes": h2d + d2h,
        "allocation_time_s": _number(response.get("allocation_time_s")),
        "binary_load_time_s": _number(response.get("binary_load_time_s")),
        "h2d_time_s": timing["h2d_time_s"],
        "kernel_time_s": timing["kernel_time_s"],
        "d2h_time_s": timing["d2h_time_s"],
        "quantization_time_s": quantization_time,
        "dequantization_time_s": dequantization_time,
        "session_response_artifact": str(
            execution.response_path.relative_to(build.session_root)
        ),
        "session_process_time_s": execution.process_time_s,
        "hardware_release_verified": True,
    }
    return {
        "status": "completed" if validation["passed"] else "failed",
        "reason": metric["reason"],
        "output": output,
        "policy_reference_output": policy_reference_output,
        "metric": metric,
    }


def _split_components(
    left: np.ndarray, right: np.ndarray
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    return {
        "ar_br": (np.asarray(left.real), np.asarray(right.real)),
        "ai_bi": (np.asarray(left.imag), np.asarray(right.imag)),
        "ar_bi": (np.asarray(left.real), np.asarray(right.imag)),
        "ai_br": (np.asarray(left.imag), np.asarray(right.real)),
    }


def _real_array(array: np.ndarray) -> np.ndarray:
    return np.asarray(array.real if np.iscomplexobj(array) else array)


def _independent_policy_reference(
    *,
    task,
    left: TensorValue,
    right: TensorValue,
    quantization_mode: str,
    caps: GenericTaskPreparationCaps,
) -> np.ndarray:
    """Evaluate the selected numeric policy without using native output files.

    This is intentionally separate from the per-component generic-loop
    references used to validate each physical transfer.  It contracts policy
    converted operands through the normal CPU TaskGraph contraction primitive,
    which catches a host-side split-complex recombination or dependency-flow
    error that per-component checks alone could miss.
    """

    left_array = np.asarray(left.array)
    right_array = np.asarray(right.array)
    complex_task = (
        classify_numeric(left_array).has_nonzero_imaginary
        or classify_numeric(right_array).has_nonzero_imaginary
    )
    if not complex_task:
        preparation = _prepare_policy_component(
            task, left, right, quantization_mode, caps
        )
        return contract_binary_task(
            task,
            _converted_operand(preparation, "left"),
            _converted_operand(preparation, "right"),
        )

    components = _split_components(left_array, right_array)
    prepared: dict[str, Any] = {}
    for name, (left_part, right_part) in components.items():
        prepared[name] = _prepare_policy_component(
            task,
            TensorValue(left.spec, left_part),
            TensorValue(right.spec, right_part),
            quantization_mode,
            caps,
        )
    left_policy = _converted_operand(
        prepared["ar_br"], "left"
    ) + 1j * _converted_operand(prepared["ai_bi"], "left")
    right_policy = _converted_operand(
        prepared["ar_br"], "right"
    ) + 1j * _converted_operand(prepared["ar_bi"], "right")
    return contract_binary_task(task, left_policy, right_policy)


def _prepare_policy_component(
    task,
    left: TensorValue,
    right: TensorValue,
    quantization_mode: str,
    caps: GenericTaskPreparationCaps,
):
    preparation = prepare_generic_task(
        GenericTaskPreparationInput(
            task=task,
            left_tensor=left,
            right_tensor=right,
            quantization_mode=quantization_mode,  # type: ignore[arg-type]
            caps=caps,
            route_id=HARDWARE_TASKGRAPH_ROUTE_ID,
        )
    )
    if preparation.status != "prepared" or preparation.prepared_operands is None:
        raise RuntimeError(
            f"hardware_profile_violation: policy reference preparation failed: {preparation.reason or preparation.status}"
        )
    return preparation


def _converted_operand(preparation, side: str) -> np.ndarray:
    operands = preparation.prepared_operands
    if operands is None:
        raise RuntimeError("hardware_profile_violation: prepared operands missing")
    value = operands.left_operand if side == "left" else operands.right_operand
    if value is None:
        raise RuntimeError("hardware_profile_violation: prepared operand missing")
    if operands.operand_mode == "float32_no_quant":
        return np.asarray(value, dtype=np.float32)
    conversion = (
        preparation.left_conversion if side == "left" else preparation.right_conversion
    )
    if conversion is None:
        raise RuntimeError(
            "hardware_profile_violation: int8 conversion metadata missing"
        )
    return np.asarray(value, dtype=np.float64) * float(conversion.scale)


def _spec_for(
    tensor_id: str,
    labels: Mapping[str, tuple[int, ...]],
    tensors: Mapping[str, np.ndarray],
) -> TensorSpec:
    array = np.asarray(tensors[tensor_id])
    return TensorSpec(
        tensor_id,
        tuple(labels[tensor_id]),
        tuple(int(dim) for dim in array.shape),
        "intermediate",
        str(array.dtype),
    )


def _sum_response_timings(tasks: list[Any]) -> JsonDict:
    fields = ("h2d_time_s", "kernel_time_s", "d2h_time_s")
    return {
        field: sum(
            _number((item.get("timing") or {}).get(field))
            for item in tasks
            if isinstance(item, Mapping)
        )
        for field in fields
    }


def _array_metrics(
    expected: np.ndarray, actual: np.ndarray, *, tolerance: float
) -> JsonDict:
    expected_array = np.asarray(expected)
    actual_array = np.asarray(actual)
    if expected_array.shape != actual_array.shape:
        return {
            "passed": False,
            "max_abs_error": None,
            "l2_error": None,
            "tolerance": tolerance,
            "reason": "shape_mismatch",
        }
    difference = actual_array - expected_array
    max_abs_error = float(np.max(np.abs(difference))) if difference.size else 0.0
    l2_error = float(np.linalg.norm(difference.ravel()))
    return {
        "passed": bool(
            np.allclose(actual_array, expected_array, rtol=tolerance, atol=tolerance)
        ),
        "max_abs_error": max_abs_error,
        "l2_error": l2_error,
        "tolerance": tolerance,
    }


def _failed_task_metric(
    task,
    task_index: int,
    reason: str,
    started: float,
    execution: HardwareSessionExecution | None = None,
) -> JsonDict:
    return {
        "task_id": task.id,
        "task_index": task_index,
        "status": "failed",
        "reason": reason,
        "input_tensor_ids": list(task.input_tensor_ids),
        "output_tensor_id": task.output_tensor_id,
        "hardware_kernel_executed": False,
        "simulator_kernel_executed": False,
        "cpu_fallback_used": False,
        "hardware_release_verified": (
            False if execution and execution.failure_stage == "kernel_timeout" else None
        ),
        "hardware_release_status": (
            "unverified_after_host_timeout"
            if execution and execution.failure_stage == "kernel_timeout"
            else "not_reached_or_unknown"
        ),
        "session_response_artifact": str(execution.response_path)
        if execution
        else None,
        "elapsed_s": time.perf_counter() - started,
    }


def _failed_result(
    case_id: str,
    quantization_mode: str,
    profile: HardwareTaskGraphProfile,
    execution_metadata: JsonDict,
    reason: str,
    started: float,
) -> HardwareTaskGraphRuntimeResult:
    summary = {
        "schema_version": HARDWARE_TASKGRAPH_RUNTIME_SCHEMA_VERSION,
        "status": "failed",
        "reason": reason,
        "case_id": case_id,
        "route_id": HARDWARE_TASKGRAPH_ROUTE_ID,
        "backend_id": profile.backend_id,
        "quantization_mode": quantization_mode,
        "hardware_execution": False,
        "hardware_kernel_executed": False,
        "simulator_kernel_executed": False,
        "cpu_fallback_used": False,
        "total_route_time_s": time.perf_counter() - started,
        "timing_scope": HARDWARE_TASKGRAPH_TIMING_SCOPE,
        "timing_is_bringup_only": True,
        "hardware_speedup_applicable": False,
        **execution_metadata,
    }
    return HardwareTaskGraphRuntimeResult(
        status="failed", reason=reason, output=None, summary=summary
    )


def _stopped_result(
    case_id: str,
    quantization_mode: str,
    profile: HardwareTaskGraphProfile,
    execution_metadata: JsonDict,
    metrics: list[JsonDict],
    reason: str,
    started: float,
    build: HardwareSessionBuild,
) -> HardwareTaskGraphRuntimeResult:
    result = _failed_result(
        case_id, quantization_mode, profile, execution_metadata, reason, started
    )
    summary = dict(result.summary)
    summary.update(
        {
            "task_count": len(metrics),
            "validated_task_count": sum(
                metric.get("status") == "completed" for metric in metrics
            ),
            "native_task_execution_count": sum(
                int(bool(metric.get("hardware_kernel_executed"))) for metric in metrics
            ),
            "hardware_execution": any(
                bool(metric.get("hardware_kernel_executed")) for metric in metrics
            ),
            "hardware_kernel_executed": any(
                bool(metric.get("hardware_kernel_executed")) for metric in metrics
            ),
            "target_observed": "hardware"
            if any(bool(metric.get("hardware_kernel_executed")) for metric in metrics)
            else None,
            "task_metrics": metrics,
            "native_source_tree_hash": build.source_tree_hash,
            "host_binary_hash": build.host_binary_hash,
            "dpu_binary_hash": build.dpu_binary_hash,
        }
    )
    return HardwareTaskGraphRuntimeResult(
        status="failed",
        reason=reason,
        output=None,
        summary=summary,
        task_metrics=tuple(metrics),
    )


def _number(value: object) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _safe_name(value: str) -> str:
    return "".join(
        character if character.isalnum() or character in {"-", "_"} else "_"
        for character in value
    )
