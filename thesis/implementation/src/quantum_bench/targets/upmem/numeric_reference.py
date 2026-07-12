from __future__ import annotations

import time
from typing import Literal, Mapping

import numpy as np

from quantum_bench.core.records import ContractionTask, JsonDict, TensorSpec, TensorValue, to_jsonable
from quantum_bench.formats import conversion_error_metrics
from quantum_bench.routing import GenericTaskPreparationInput, prepare_generic_task
from quantum_bench.targets.upmem.runtime_evidence import (
    GENERIC_QUANTIZED_TASKGRAPH_REFERENCE_SCHEMA_VERSION,
    GenericQuantizedTaskGraphReference,
    _generic_reference_base_task_metric,
    _generic_reference_stop_result,
    _generic_reference_summary,
    _unavailable_full_precision_accuracy,
)
from quantum_bench.tn.contract import contract_binary_task
from quantum_bench.tn.execution import live_tensor_bytes, order_final_tensor, release_dead_inputs, remaining_input_uses
from quantum_bench.tn.network import TensorNetworkValue

UpmemTaskGraphQuantizationMode = Literal["per_task_input_quantize", "none", "persistent_network_quantized"]

QUANTIZED_FINAL_VALIDATION_TOLERANCES = {
    "max_abs_error": 0.25,
    "l2_error": 2.0,
    "max_rel_error": 10.0,
    "norm_drift": 2.0,
    "min_fidelity": 0.0,
}


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
    full_precision_error = conversion_error_metrics(contract_binary_task(task, left, right), output)
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
        full_precision_metrics={
            "reference_kind": "full_precision_vs_expected_quantized_reference",
            "max_abs_error": full_precision_error.max_abs_error,
            "l2_error": full_precision_error.l2_error,
            "relative_l2_error": full_precision_error.relative_l2_error,
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


def _complex_split_reference_metrics(
    *,
    task: ContractionTask,
    left: np.ndarray,
    right: np.ndarray,
    outputs: Mapping[str, np.ndarray],
    expected: Mapping[str, np.ndarray],
    quantization_mode: str,
) -> tuple[np.ndarray, JsonDict, JsonDict]:
    output = (outputs["ar_br"] - outputs["ai_bi"]) + 1j * (outputs["ar_bi"] + outputs["ai_br"])
    expected_complex = (expected["ar_br"] - expected["ai_bi"]) + 1j * (expected["ar_bi"] + expected["ai_br"])
    full_precision_error = conversion_error_metrics(contract_binary_task(task, left, right), expected_complex)
    validation = conversion_error_metrics(expected_complex, output)
    tolerance = 1.0e-5 if quantization_mode == "none" else 1.0e-8
    return (
        output,
        {
            "reference_kind": "complex_quantized_dequantized_reference_vs_split_complex_generic",
            "max_abs_error": validation.max_abs_error,
            "l2_error": validation.l2_error,
            "relative_l2_error": validation.relative_l2_error,
            "passed": validation.max_abs_error <= tolerance,
            "max_abs_tolerance": tolerance,
        },
        {
            "reference_kind": "full_precision_vs_expected_quantized_reference",
            "max_abs_error": full_precision_error.max_abs_error,
            "l2_error": full_precision_error.l2_error,
            "relative_l2_error": full_precision_error.relative_l2_error,
        },
    )


def _final_validation(output: np.ndarray, reference_output: np.ndarray | None, *, reference_kind: str) -> JsonDict:
    if reference_output is None:
        return {"passed": False, "reason": "reference_output_missing", "reference_kind": reference_kind}
    from quantum_bench.targets.upmem.taskgraph_runtime import QUANTIZED_FINAL_VALIDATION_TOLERANCES
    from quantum_bench.validation import validate

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


def _full_precision_accuracy(
    output: np.ndarray,
    reference_output: np.ndarray | None,
    *,
    reference_kind: str,
) -> JsonDict:
    if reference_output is None:
        return {
            **_unavailable_full_precision_accuracy(),
            "full_precision_reference_kind": reference_kind,
        }
    metrics = conversion_error_metrics(reference_output, output)
    return {
        "full_precision_reference_kind": reference_kind,
        "full_precision_max_abs_error": metrics.max_abs_error,
        "full_precision_l2_error": metrics.l2_error,
        "full_precision_relative_l2_error": metrics.relative_l2_error,
        "available": True,
    }
