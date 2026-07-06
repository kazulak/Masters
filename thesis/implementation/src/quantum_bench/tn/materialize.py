from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Literal, Mapping

import numpy as np

from quantum_bench.core.records import ContractionTask, JsonDict, TaskGraph, TensorSpec, TensorValue, to_jsonable
from quantum_bench.tn.contract import contract_binary_task


TASK_INPUT_MATERIALIZATION_SCHEMA_VERSION = "task_input_materialization_v1"

TaskInputMaterializationStatus = Literal["initial_inputs_available", "materialized", "unsupported", "failed"]


@dataclass(frozen=True)
class TaskInputReplayMetric:
    task_id: str
    input_tensor_ids: tuple[str, str]
    output_tensor_id: str
    input_shapes: tuple[tuple[int, ...], tuple[int, ...]]
    output_shape: tuple[int, ...]
    output_labels: tuple[int, ...]
    output_dtype: str
    execution_time_s: float
    output_bytes: int
    retained_tensor_bytes: int


@dataclass(frozen=True)
class TaskInputMaterializationRequest:
    graph: TaskGraph
    initial_tensors: Mapping[str, TensorValue]
    target_task_index: int | None = None
    target_task_id: str | None = None


@dataclass(frozen=True)
class TaskInputMaterializationResult:
    schema_version: str
    status: TaskInputMaterializationStatus
    reason: str | None
    target_task_index: int | None
    target_task_id: str | None
    selected_input_tensor_ids: tuple[str, str] | None
    input_sources: JsonDict
    replayed_task_count: int
    replayed_task_ids: tuple[str, ...]
    replay_time_s: float
    peak_materialized_bytes: int
    dead_tensor_release_implemented: bool
    step_metrics: tuple[TaskInputReplayMetric, ...] = ()
    left_tensor: TensorValue | None = field(default=None, repr=False, compare=False)
    right_tensor: TensorValue | None = field(default=None, repr=False, compare=False)
    error: str | None = None

    def to_json_dict(self) -> JsonDict:
        return to_jsonable(
            {
                "schema_version": self.schema_version,
                "status": self.status,
                "reason": self.reason,
                "target_task_index": self.target_task_index,
                "target_task_id": self.target_task_id,
                "selected_input_tensor_ids": self.selected_input_tensor_ids,
                "input_sources": self.input_sources,
                "replayed_task_count": self.replayed_task_count,
                "replayed_task_ids": self.replayed_task_ids,
                "replay_time_s": self.replay_time_s,
                "peak_materialized_bytes": self.peak_materialized_bytes,
                "dead_tensor_release_implemented": self.dead_tensor_release_implemented,
                "step_metrics": self.step_metrics,
                "error": self.error,
            }
        )


def materialize_task_inputs(request: TaskInputMaterializationRequest) -> TaskInputMaterializationResult:
    started = time.perf_counter()
    target_selection = _resolve_target(request)
    if isinstance(target_selection, TaskInputMaterializationResult):
        return target_selection

    target_index, target_task = target_selection
    tensor_map = dict(request.initial_tensors)
    source_map = {
        tensor_id: _source_record(tensor, source="initial", produced_by=tensor.spec.produced_by)
        for tensor_id, tensor in tensor_map.items()
    }
    peak_bytes = _retained_tensor_bytes(tensor_map)

    if _inputs_available(target_task, tensor_map):
        return _success_result(
            status="initial_inputs_available",
            target_task=target_task,
            target_task_index=target_index,
            tensor_map=tensor_map,
            source_map=source_map,
            step_metrics=(),
            replay_time_s=float(time.perf_counter() - started),
            peak_materialized_bytes=peak_bytes,
        )

    step_metrics: list[TaskInputReplayMetric] = []
    replayed_task_ids: list[str] = []

    for step_index, task in enumerate(request.graph.tasks[:target_index]):
        if not _inputs_available(task, tensor_map):
            missing = tuple(tensor_id for tensor_id in task.input_tensor_ids if tensor_id not in tensor_map)
            return _unsupported_result(
                reason="missing_predecessor_input_tensor",
                target_task=target_task,
                target_task_index=target_index,
                selected_input_tensor_ids=target_task.input_tensor_ids,
                input_sources=_selected_input_sources(target_task, source_map),
                replayed_task_ids=tuple(replayed_task_ids),
                step_metrics=tuple(step_metrics),
                replay_time_s=float(time.perf_counter() - started),
                peak_materialized_bytes=peak_bytes,
                error=f"Task {task.id} is missing input tensor(s): {', '.join(missing)}",
            )

        try:
            tensor, metric = _replay_task(task, tensor_map)
        except ValueError as exc:
            return _failed_result(
                reason=str(exc),
                target_task=target_task,
                target_task_index=target_index,
                selected_input_tensor_ids=target_task.input_tensor_ids,
                input_sources=_selected_input_sources(target_task, source_map),
                replayed_task_ids=tuple(replayed_task_ids),
                step_metrics=tuple(step_metrics),
                replay_time_s=float(time.perf_counter() - started),
                peak_materialized_bytes=peak_bytes,
                error=str(exc),
            )
        except Exception as exc:  # pragma: no cover - defensive materializer boundary
            return _failed_result(
                reason="replay_exception",
                target_task=target_task,
                target_task_index=target_index,
                selected_input_tensor_ids=target_task.input_tensor_ids,
                input_sources=_selected_input_sources(target_task, source_map),
                replayed_task_ids=tuple(replayed_task_ids),
                step_metrics=tuple(step_metrics),
                replay_time_s=float(time.perf_counter() - started),
                peak_materialized_bytes=peak_bytes,
                error=str(exc),
            )

        tensor_map[task.output_tensor_id] = tensor
        source_map[task.output_tensor_id] = _source_record(
            tensor,
            source="replayed",
            produced_by=task.id,
            replayed_task_index=step_index,
        )
        step_metrics.append(metric)
        replayed_task_ids.append(task.id)
        peak_bytes = max(peak_bytes, metric.retained_tensor_bytes)

        if _inputs_available(target_task, tensor_map):
            return _success_result(
                status="materialized",
                target_task=target_task,
                target_task_index=target_index,
                tensor_map=tensor_map,
                source_map=source_map,
                step_metrics=tuple(step_metrics),
                replay_time_s=float(time.perf_counter() - started),
                peak_materialized_bytes=peak_bytes,
            )

    return _unsupported_result(
        reason="missing_predecessor_input_tensor",
        target_task=target_task,
        target_task_index=target_index,
        selected_input_tensor_ids=target_task.input_tensor_ids,
        input_sources=_selected_input_sources(target_task, source_map),
        replayed_task_ids=tuple(replayed_task_ids),
        step_metrics=tuple(step_metrics),
        replay_time_s=float(time.perf_counter() - started),
        peak_materialized_bytes=peak_bytes,
        error=f"Target task {target_task.id} inputs were not available after replaying predecessors",
    )


def _resolve_target(
    request: TaskInputMaterializationRequest,
) -> tuple[int, ContractionTask] | TaskInputMaterializationResult:
    if request.target_task_index is not None and request.target_task_id is not None:
        return _selection_error("ambiguous_target_selection", request.target_task_index, request.target_task_id)
    if request.target_task_index is None and request.target_task_id is None:
        return _selection_error("target_selection_required", None, None)
    if request.target_task_index is not None:
        index = request.target_task_index
        if index < 0 or index >= len(request.graph.tasks):
            return _selection_error("target_task_index_out_of_range", index, None)
        return index, request.graph.tasks[index]

    target_id = str(request.target_task_id)
    for index, task in enumerate(request.graph.tasks):
        if task.id == target_id:
            return index, task
    return _selection_error("target_task_id_not_found", None, target_id)


def _selection_error(
    reason: str,
    target_task_index: int | None,
    target_task_id: str | None,
) -> TaskInputMaterializationResult:
    return TaskInputMaterializationResult(
        schema_version=TASK_INPUT_MATERIALIZATION_SCHEMA_VERSION,
        status="unsupported",
        reason=reason,
        target_task_index=target_task_index,
        target_task_id=target_task_id,
        selected_input_tensor_ids=None,
        input_sources={},
        replayed_task_count=0,
        replayed_task_ids=(),
        replay_time_s=0.0,
        peak_materialized_bytes=0,
        dead_tensor_release_implemented=False,
    )


def _success_result(
    *,
    status: Literal["initial_inputs_available", "materialized"],
    target_task: ContractionTask,
    target_task_index: int,
    tensor_map: Mapping[str, TensorValue],
    source_map: Mapping[str, JsonDict],
    step_metrics: tuple[TaskInputReplayMetric, ...],
    replay_time_s: float,
    peak_materialized_bytes: int,
) -> TaskInputMaterializationResult:
    left_id, right_id = target_task.input_tensor_ids
    return TaskInputMaterializationResult(
        schema_version=TASK_INPUT_MATERIALIZATION_SCHEMA_VERSION,
        status=status,
        reason=None,
        target_task_index=target_task_index,
        target_task_id=target_task.id,
        selected_input_tensor_ids=target_task.input_tensor_ids,
        input_sources=_selected_input_sources(target_task, source_map),
        replayed_task_count=len(step_metrics),
        replayed_task_ids=tuple(metric.task_id for metric in step_metrics),
        replay_time_s=replay_time_s,
        peak_materialized_bytes=int(peak_materialized_bytes),
        dead_tensor_release_implemented=False,
        step_metrics=step_metrics,
        left_tensor=tensor_map[left_id],
        right_tensor=tensor_map[right_id],
    )


def _unsupported_result(
    *,
    reason: str,
    target_task: ContractionTask,
    target_task_index: int,
    selected_input_tensor_ids: tuple[str, str] | None,
    input_sources: JsonDict,
    replayed_task_ids: tuple[str, ...],
    step_metrics: tuple[TaskInputReplayMetric, ...],
    replay_time_s: float,
    peak_materialized_bytes: int,
    error: str | None = None,
) -> TaskInputMaterializationResult:
    return TaskInputMaterializationResult(
        schema_version=TASK_INPUT_MATERIALIZATION_SCHEMA_VERSION,
        status="unsupported",
        reason=reason,
        target_task_index=target_task_index,
        target_task_id=target_task.id,
        selected_input_tensor_ids=selected_input_tensor_ids,
        input_sources=input_sources,
        replayed_task_count=len(replayed_task_ids),
        replayed_task_ids=replayed_task_ids,
        replay_time_s=replay_time_s,
        peak_materialized_bytes=int(peak_materialized_bytes),
        dead_tensor_release_implemented=False,
        step_metrics=step_metrics,
        error=error,
    )


def _failed_result(
    *,
    reason: str,
    target_task: ContractionTask,
    target_task_index: int,
    selected_input_tensor_ids: tuple[str, str] | None,
    input_sources: JsonDict,
    replayed_task_ids: tuple[str, ...],
    step_metrics: tuple[TaskInputReplayMetric, ...],
    replay_time_s: float,
    peak_materialized_bytes: int,
    error: str | None = None,
) -> TaskInputMaterializationResult:
    return TaskInputMaterializationResult(
        schema_version=TASK_INPUT_MATERIALIZATION_SCHEMA_VERSION,
        status="failed",
        reason=reason,
        target_task_index=target_task_index,
        target_task_id=target_task.id,
        selected_input_tensor_ids=selected_input_tensor_ids,
        input_sources=input_sources,
        replayed_task_count=len(replayed_task_ids),
        replayed_task_ids=replayed_task_ids,
        replay_time_s=replay_time_s,
        peak_materialized_bytes=int(peak_materialized_bytes),
        dead_tensor_release_implemented=False,
        step_metrics=step_metrics,
        error=error,
    )


def _replay_task(task: ContractionTask, tensor_map: Mapping[str, TensorValue]) -> tuple[TensorValue, TaskInputReplayMetric]:
    left = tensor_map[task.input_tensor_ids[0]]
    right = tensor_map[task.input_tensor_ids[1]]
    started = time.perf_counter()
    output = contract_binary_task(task, np.asarray(left.array, dtype=np.complex128), np.asarray(right.array, dtype=np.complex128))
    output = np.asarray(output, dtype=np.complex128)
    execution_time_s = float(time.perf_counter() - started)
    if tuple(int(dim) for dim in output.shape) != task.output_shape:
        raise ValueError("replay_output_shape_mismatch")

    tensor = TensorValue(
        TensorSpec(
            id=task.output_tensor_id,
            labels=task.output_labels,
            shape=task.output_shape,
            structure=task.structure,
            dtype=str(output.dtype),
            produced_by=task.id,
        ),
        output,
    )
    retained_bytes = _retained_tensor_bytes({**tensor_map, task.output_tensor_id: tensor})
    metric = TaskInputReplayMetric(
        task_id=task.id,
        input_tensor_ids=task.input_tensor_ids,
        output_tensor_id=task.output_tensor_id,
        input_shapes=task.input_shapes,
        output_shape=task.output_shape,
        output_labels=task.output_labels,
        output_dtype=str(output.dtype),
        execution_time_s=execution_time_s,
        output_bytes=int(output.nbytes),
        retained_tensor_bytes=retained_bytes,
    )
    return tensor, metric


def _inputs_available(task: ContractionTask, tensor_map: Mapping[str, TensorValue]) -> bool:
    return all(tensor_id in tensor_map for tensor_id in task.input_tensor_ids)


def _selected_input_sources(task: ContractionTask, source_map: Mapping[str, JsonDict]) -> JsonDict:
    return {tensor_id: dict(source_map.get(tensor_id, {"source": "unavailable"})) for tensor_id in task.input_tensor_ids}


def _source_record(
    tensor: TensorValue,
    *,
    source: str,
    produced_by: str | None,
    replayed_task_index: int | None = None,
) -> JsonDict:
    return {
        "source": source,
        "tensor_id": tensor.spec.id,
        "labels": tensor.spec.labels,
        "shape": tensor.spec.shape,
        "dtype": tensor.spec.dtype,
        "produced_by": produced_by,
        "replayed_task_index": replayed_task_index,
        "nbytes": int(np.asarray(tensor.array).nbytes),
    }


def _retained_tensor_bytes(tensor_map: Mapping[str, TensorValue]) -> int:
    return int(sum(np.asarray(tensor.array).nbytes for tensor in tensor_map.values()))
