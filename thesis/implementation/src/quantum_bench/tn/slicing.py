from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from quantum_bench.core.records import ContractionTask, TaskGraph


JsonDict = dict[str, Any]


@dataclass(frozen=True)
class SliceIndexAssignment:
    label: int
    value: int

    def to_json(self) -> JsonDict:
        return {"label": self.label, "value": self.value}


@dataclass(frozen=True)
class SliceInputRestriction:
    tensor_id: str
    label: int
    axis: int
    value: int

    def to_json(self) -> JsonDict:
        return {"tensor_id": self.tensor_id, "label": self.label, "axis": self.axis, "value": self.value}


@dataclass(frozen=True)
class SliceModelTask:
    id: str
    source_task_id: str
    slice_id: int
    assignment: SliceIndexAssignment
    dependencies: tuple[str, ...]
    input_tensor_ids: tuple[str, str]
    partial_output_tensor_id: str
    input_restrictions: tuple[SliceInputRestriction, ...]
    output_shape: tuple[int, ...]
    output_labels: tuple[int, ...]

    def to_json(self) -> JsonDict:
        return {
            "id": self.id,
            "source_task_id": self.source_task_id,
            "slice_id": self.slice_id,
            "assignment": self.assignment.to_json(),
            "dependencies": list(self.dependencies),
            "input_tensor_ids": list(self.input_tensor_ids),
            "partial_output_tensor_id": self.partial_output_tensor_id,
            "input_restrictions": [restriction.to_json() for restriction in self.input_restrictions],
            "output_shape": list(self.output_shape),
            "output_labels": list(self.output_labels),
        }


@dataclass(frozen=True)
class SliceReconstructionStep:
    id: str
    source_task_id: str
    dependencies: tuple[str, ...]
    input_tensor_ids: tuple[str, ...]
    output_tensor_id: str
    operation: str = "sum_partials"

    def to_json(self) -> JsonDict:
        return {
            "id": self.id,
            "source_task_id": self.source_task_id,
            "dependencies": list(self.dependencies),
            "input_tensor_ids": list(self.input_tensor_ids),
            "output_tensor_id": self.output_tensor_id,
            "operation": self.operation,
        }


@dataclass(frozen=True)
class SliceDependencyRewrite:
    task_id: str
    old_dependency: str
    new_dependency: str

    def to_json(self) -> JsonDict:
        return {"task_id": self.task_id, "old_dependency": self.old_dependency, "new_dependency": self.new_dependency}


@dataclass(frozen=True)
class SliceAwareTaskGraphModel:
    available: bool
    rejection_reason: str | None
    source_task_count: int
    slice_model_kind: str
    sliced_task_id: str | None
    sliced_indices: tuple[int, ...]
    slice_model_slice_count: int
    slice_model_task_count: int
    slice_tasks: tuple[SliceModelTask, ...]
    reconstruction_step: SliceReconstructionStep | None
    downstream_dependency_rewrites: tuple[SliceDependencyRewrite, ...]
    hybrid_ready: bool = False
    slice_model_execution_status: str = "model_only"

    def to_metadata(self) -> JsonDict:
        reconstruction_required = self.reconstruction_step is not None
        return {
            "slice_aware_taskgraph_available": self.available,
            "source_task_count": self.source_task_count,
            "slice_model_kind": self.slice_model_kind,
            "slice_model_slice_count": self.slice_model_slice_count,
            "slice_model_task_count": self.slice_model_task_count,
            "slice_model_execution_status": self.slice_model_execution_status,
            "slice_model_rejection_reason": self.rejection_reason,
            "sliced_task_id": self.sliced_task_id,
            "sliced_indices": list(self.sliced_indices),
            "slice_model_sliced_indices": list(self.sliced_indices),
            "slice_assignments": [task.assignment.to_json() | {"slice_id": task.slice_id} for task in self.slice_tasks],
            "slice_model_tasks": [task.to_json() for task in self.slice_tasks],
            "slice_reconstruction_required": reconstruction_required,
            "slice_reconstruction_status": "model_only" if reconstruction_required else "not_applicable",
            "slice_reconstruction_step": self.reconstruction_step.to_json() if self.reconstruction_step else None,
            "slice_dependency_rewrites": [rewrite.to_json() for rewrite in self.downstream_dependency_rewrites],
            "slice_task_execution_mode": "model_only" if self.available else "not_applicable",
            "hybrid_ready": False,
        }


def build_slice_aware_taskgraph_model(
    graph: TaskGraph,
    *,
    max_slice_count: int = 4,
    sliced_task_id: str | None = None,
) -> SliceAwareTaskGraphModel:
    if max_slice_count < 2:
        raise ValueError("max_slice_count must be >= 2")
    selected = _select_sliced_task(
        graph, max_slice_count=max_slice_count, sliced_task_id=sliced_task_id
    )
    if selected is None:
        return _unavailable_model(graph, _unsupported_reason(graph, max_slice_count=max_slice_count))
    task, label, dim = selected
    slice_tasks = tuple(_build_slice_task(task, label, slice_id) for slice_id in range(dim))
    reconstruction = SliceReconstructionStep(
        id=f"{task.id}__slice_reconstruct",
        source_task_id=task.id,
        dependencies=tuple(slice_task.id for slice_task in slice_tasks),
        input_tensor_ids=tuple(slice_task.partial_output_tensor_id for slice_task in slice_tasks),
        output_tensor_id=task.output_tensor_id,
    )
    rewrites = tuple(
        SliceDependencyRewrite(downstream.id, task.id, reconstruction.id)
        for downstream in graph.tasks
        if task.id in downstream.dependencies
    )
    return SliceAwareTaskGraphModel(
        available=True,
        rejection_reason=None,
        source_task_count=len(graph.tasks),
        slice_model_kind="single_task_single_index_model",
        sliced_task_id=task.id,
        sliced_indices=(label,),
        slice_model_slice_count=dim,
        slice_model_task_count=len(slice_tasks),
        slice_tasks=slice_tasks,
        reconstruction_step=reconstruction,
        downstream_dependency_rewrites=rewrites,
    )


def validate_slice_aware_taskgraph_model(model: SliceAwareTaskGraphModel) -> tuple[bool, str | None]:
    if not model.available:
        return False, model.rejection_reason
    slice_ids = [task.id for task in model.slice_tasks]
    if len(slice_ids) != len(set(slice_ids)):
        return False, "duplicate_slice_task_id"
    if model.slice_model_task_count != model.slice_model_slice_count:
        return False, "slice_task_count_mismatch"
    if model.slice_model_task_count != len(model.slice_tasks):
        return False, "slice_model_task_count_mismatch"
    if len(model.sliced_indices) != 1:
        return False, "unsupported_sliced_index_count"
    sliced_label = model.sliced_indices[0]
    for task in model.slice_tasks:
        if task.assignment.label != sliced_label:
            return False, "slice_assignment_label_mismatch"
        if len(task.input_restrictions) != 2:
            return False, "slice_input_restriction_count_mismatch"
    if model.reconstruction_step is None:
        return False, "missing_reconstruction_step"
    expected_dependencies = tuple(slice_ids)
    if model.reconstruction_step.dependencies != expected_dependencies:
        return False, "reconstruction_dependency_mismatch"
    if tuple(model.reconstruction_step.input_tensor_ids) != tuple(task.partial_output_tensor_id for task in model.slice_tasks):
        return False, "reconstruction_input_mismatch"
    if model.hybrid_ready:
        return False, "model_must_not_be_hybrid_ready"
    if model.slice_model_execution_status != "model_only":
        return False, "unexpected_execution_status"
    return True, None


def _select_sliced_task(
    graph: TaskGraph,
    *,
    max_slice_count: int,
    sliced_task_id: str | None = None,
) -> tuple[ContractionTask, int, int] | None:
    for task in graph.tasks:
        if sliced_task_id is not None and task.id != sliced_task_id:
            continue
        for label in task.contracted_labels:
            dim = _task_label_dim(task, label)
            if 1 < dim <= max_slice_count:
                return task, label, dim
    if sliced_task_id is not None and not any(
        task.id == sliced_task_id for task in graph.tasks
    ):
        raise ValueError(f"Unknown sliced task id: {sliced_task_id}")
    return None


def _unsupported_reason(graph: TaskGraph, *, max_slice_count: int) -> str:
    if not graph.tasks:
        return "no_contraction_tasks"
    saw_too_large_candidate = False
    for task in graph.tasks:
        for label in task.contracted_labels:
            dim = _task_label_dim(task, label)
            if dim > max_slice_count:
                saw_too_large_candidate = True
    return "slice_count_exceeds_cap" if saw_too_large_candidate else "no_supported_sliced_label"


def _unavailable_model(graph: TaskGraph, reason: str) -> SliceAwareTaskGraphModel:
    return SliceAwareTaskGraphModel(
        available=False,
        rejection_reason=reason,
        source_task_count=len(graph.tasks),
        slice_model_kind="unavailable",
        sliced_task_id=None,
        sliced_indices=(),
        slice_model_slice_count=0,
        slice_model_task_count=0,
        slice_tasks=(),
        reconstruction_step=None,
        downstream_dependency_rewrites=(),
        slice_model_execution_status="unsupported",
    )


def _build_slice_task(task: ContractionTask, label: int, slice_id: int) -> SliceModelTask:
    assignment = SliceIndexAssignment(label=label, value=slice_id)
    return SliceModelTask(
        id=f"{task.id}__slice_{slice_id}",
        source_task_id=task.id,
        slice_id=slice_id,
        assignment=assignment,
        dependencies=task.dependencies,
        input_tensor_ids=task.input_tensor_ids,
        partial_output_tensor_id=f"{task.output_tensor_id}__slice_{slice_id}",
        input_restrictions=_input_restrictions(task, label, slice_id),
        output_shape=task.output_shape,
        output_labels=task.output_labels,
    )


def _input_restrictions(task: ContractionTask, label: int, value: int) -> tuple[SliceInputRestriction, ...]:
    restrictions: list[SliceInputRestriction] = []
    for tensor_id, labels in zip(task.input_tensor_ids, (task.left_labels, task.right_labels), strict=True):
        if label in labels:
            restrictions.append(SliceInputRestriction(tensor_id=tensor_id, label=label, axis=labels.index(label), value=value))
    return tuple(restrictions)


def _task_label_dim(task: ContractionTask, label: int) -> int:
    if label in task.left_labels:
        return int(task.input_shapes[0][task.left_labels.index(label)])
    if label in task.right_labels:
        return int(task.input_shapes[1][task.right_labels.index(label)])
    raise ValueError(f"Task {task.id} does not contain label {label}")
