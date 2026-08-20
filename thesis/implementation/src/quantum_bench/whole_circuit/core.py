from __future__ import annotations

import time
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Protocol

import numpy as np

from quantum_bench.core.records import ContractionTask, TaskGraph, TensorValue
from quantum_bench.tn.execution_bundle import contraction_path_structure_hash
from quantum_bench.whole_circuit.strategies import (
    ReadyTaskOrderPolicy,
    SerialReadyTaskOrderPolicy,
)


@dataclass(frozen=True)
class DeviceTopology:
    """Execution placement description kept independent of the engine."""

    backend: str = "cpu"
    device_ids: tuple[str, ...] = ("cpu",)
    tasklets_per_device: int = 1

    def __post_init__(self) -> None:
        if not self.device_ids:
            raise ValueError("DeviceTopology requires at least one device")
        if self.tasklets_per_device < 1:
            raise ValueError("tasklets_per_device must be >= 1")


class NumericPolicy(Protocol):
    name: str

    def contract(
        self,
        task: ContractionTask,
        left: np.ndarray,
        right: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Execute one contraction under this policy's numeric contract."""


@dataclass(frozen=True)
class EngineTaskResult:
    output: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)


class TaskExecutionSession(Protocol):
    def execute(
        self,
        task: ContractionTask,
        left: np.ndarray,
        right: np.ndarray,
    ) -> EngineTaskResult: ...

    def close(self) -> dict[str, Any] | None: ...


class TaskExecutionEngine(Protocol):
    name: str

    def open_session(
        self,
        policy: NumericPolicy,
        topology: DeviceTopology = DeviceTopology(),
    ) -> TaskExecutionSession: ...


class TensorStore(Protocol):
    def seed(
        self, tensors: tuple[TensorValue, ...], use_counts: dict[str, int]
    ) -> None: ...

    def put(self, tensor_id: str, value: np.ndarray) -> None: ...

    def get(self, tensor_id: str) -> np.ndarray: ...

    def consume(self, tensor_id: str) -> bool:
        """Consume one use and return whether the tensor was released."""
        ...

    def contains(self, tensor_id: str) -> bool: ...

    def live_ids(self) -> tuple[str, ...]: ...


class InMemoryTensorStore:
    """Lifetime-aware, single-execution store for the CPU implementation.

    Create a fresh store for each graph execution. Re-seeding is rejected so
    release history and stale values cannot leak into a later measurement.
    """

    def __init__(self) -> None:
        self._values: dict[str, np.ndarray] = {}
        self._remaining_uses: dict[str, int] = {}
        self._seeded = False
        self.released_tensor_ids: list[str] = []

    def seed(
        self, tensors: tuple[TensorValue, ...], use_counts: dict[str, int]
    ) -> None:
        if self._seeded:
            raise RuntimeError(
                "TensorStore is single-use and has already been seeded; create a new store for each execution"
            )
        self._seeded = True
        self._remaining_uses = {
            tensor_id: int(count) for tensor_id, count in use_counts.items()
        }
        for tensor in tensors:
            if tensor.spec.id in self._values:
                raise ValueError(f"Duplicate tensor id: {tensor.spec.id}")
            self._values[tensor.spec.id] = np.asarray(tensor.array)
            self._remaining_uses.setdefault(tensor.spec.id, 0)

    def put(self, tensor_id: str, value: np.ndarray) -> None:
        if tensor_id in self._values:
            raise ValueError(f"Tensor id already exists: {tensor_id}")
        self._values[tensor_id] = np.asarray(value)
        self._remaining_uses.setdefault(tensor_id, 0)

    def get(self, tensor_id: str) -> np.ndarray:
        try:
            return self._values[tensor_id]
        except KeyError as exc:
            raise KeyError(f"Tensor is unavailable: {tensor_id}") from exc

    def consume(self, tensor_id: str) -> bool:
        if tensor_id not in self._values:
            raise KeyError(f"Tensor is unavailable: {tensor_id}")
        remaining = self._remaining_uses.get(tensor_id, 0) - 1
        self._remaining_uses[tensor_id] = remaining
        if remaining <= 0:
            del self._values[tensor_id]
            self.released_tensor_ids.append(tensor_id)
            return True
        return False

    def contains(self, tensor_id: str) -> bool:
        return tensor_id in self._values

    def live_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._values))


@dataclass(frozen=True)
class NumpyCpuSession:
    policy: NumericPolicy
    topology: DeviceTopology
    closed: bool = False

    def execute(
        self,
        task: ContractionTask,
        left: np.ndarray,
        right: np.ndarray,
    ) -> EngineTaskResult:
        if self.closed:
            raise RuntimeError("NumpyCpuSession is closed")
        start = time.perf_counter()
        output, policy_metadata = self.policy.contract(task, left, right)
        return EngineTaskResult(
            output=np.asarray(output),
            metadata={
                **policy_metadata,
                "engine": "numpy_cpu",
                "device": self.topology.device_ids[0],
                "execution_time_s": time.perf_counter() - start,
            },
        )

    def close(self) -> dict[str, Any]:
        object.__setattr__(self, "closed", True)
        return {}


class NumpyCpuEngine:
    name = "numpy_cpu"

    def open_session(
        self,
        policy: NumericPolicy,
        topology: DeviceTopology = DeviceTopology(),
    ) -> NumpyCpuSession:
        if topology.backend != "cpu":
            raise ValueError(
                f"NumpyCpuEngine requires cpu topology, got {topology.backend}"
            )
        return NumpyCpuSession(policy=policy, topology=topology)


@dataclass(frozen=True)
class WholeGraphExecution:
    output: np.ndarray
    metadata: dict[str, Any]


class WholeGraphExecutor:
    """Execute every task once using a replaceable engine and tensor store."""

    def __init__(
        self,
        engine: TaskExecutionEngine,
        policy: NumericPolicy,
        *,
        topology: DeviceTopology | None = None,
        store: TensorStore | None = None,
        task_order_policy: ReadyTaskOrderPolicy | None = None,
        scheduler: ReadyTaskOrderPolicy | None = None,
    ) -> None:
        if task_order_policy is not None and scheduler is not None:
            raise ValueError("provide task_order_policy or scheduler, not both")
        self.engine = engine
        self.policy = policy
        self.topology = topology or DeviceTopology()
        self.store = store or InMemoryTensorStore()
        self.task_order_policy = (
            task_order_policy or scheduler or SerialReadyTaskOrderPolicy()
        )
        self.scheduler = self.task_order_policy

    def execute(self, graph: TaskGraph, network: Any) -> WholeGraphExecution:
        self._validate_graph(graph)
        use_counts = _input_use_counts(graph)
        self.store.seed(tuple(network.tensors), use_counts)
        final_id = _final_tensor_id(graph)
        route_start = time.perf_counter()
        session_open_start = time.perf_counter()
        session = self.engine.open_session(self.policy, self.topology)
        session_open_s = time.perf_counter() - session_open_start
        task_records: list[dict[str, Any]] = []
        completed: set[str] = set()
        pending = {task.id: task for task in graph.tasks}
        executed_order: list[str] = []
        graph_start = time.perf_counter()
        session_metadata: dict[str, Any] = {}
        try:
            while pending:
                ready = self.task_order_policy.select_ready_tasks(
                    MappingProxyType(dict(pending)), frozenset(completed)
                )
                if not ready:
                    unresolved = {
                        task_id: task.dependencies
                        for task_id, task in sorted(pending.items())
                    }
                    raise ValueError(
                        f"TaskGraph dependencies are cyclic or unresolved: {unresolved}"
                    )
                ready_ids = tuple(task.id for task in ready)
                if len(set(ready_ids)) != len(ready_ids):
                    raise ValueError("Task order policy returned duplicate task ids")
                if any(task_id not in pending for task_id in ready_ids):
                    raise ValueError(
                        "Task order policy returned a task that is not pending"
                    )
                if any(
                    not set(pending[task_id].dependencies) <= completed
                    for task_id in ready_ids
                ):
                    raise ValueError(
                        "Task order policy returned a task with unresolved dependencies"
                    )
                if any(task is not pending[task.id] for task in ready):
                    raise ValueError(
                        "Task order policy must return canonical pending task objects"
                    )
                for task_id in ready_ids:
                    task = pending[task_id]
                    left_id, right_id = task.input_tensor_ids
                    left = self.store.get(left_id)
                    right = self.store.get(right_id)
                    result = session.execute(task, left, right)
                    output = np.asarray(result.output)
                    if tuple(output.shape) != tuple(task.output_shape):
                        raise ValueError(
                            f"Task {task.id} output shape {output.shape} does not match {task.output_shape}"
                        )
                    self.store.put(task.output_tensor_id, output)
                    self.store.consume(left_id)
                    self.store.consume(right_id)
                    record = {
                        "task_id": task.id,
                        "input_tensor_ids": task.input_tensor_ids,
                        "output_tensor_id": task.output_tensor_id,
                        "dependencies": task.dependencies,
                        "output_shape": task.output_shape,
                        **result.metadata,
                    }
                    task_records.append(record)
                    completed.add(task.id)
                    executed_order.append(task.id)
                    del pending[task.id]
        finally:
            graph_execution_s = time.perf_counter() - graph_start
            session_close_start = time.perf_counter()
            close_metadata = session.close()
            session_close_s = time.perf_counter() - session_close_start
            if close_metadata is not None:
                session_metadata = dict(close_metadata)

        output = self.store.get(final_id)
        final_task = next(
            task for task in graph.tasks if task.output_tensor_id == final_id
        )
        output, transposed = _order_final_tensor(
            output, final_task.output_labels, graph.network.output_labels
        )
        total_route_s = time.perf_counter() - route_start
        task_time_s = float(
            sum(float(record.get("execution_time_s", 0.0)) for record in task_records)
        )
        metadata = {
            "execution_engine": self.engine.name,
            "numeric_policy": self.policy.name,
            "task_order_policy": self.task_order_policy.name,
            "task_execution_mode": "serial",
            "device_topology": {
                "backend": self.topology.backend,
                "device_ids": self.topology.device_ids,
                "tasklets_per_device": self.topology.tasklets_per_device,
            },
            "task_count": len(task_records),
            "executed_order": tuple(executed_order),
            "task_metrics": tuple(task_records),
            "session_metadata": session_metadata,
            "timing": {
                "session_open_s": session_open_s,
                "allocation_time_s": session_open_s,
                "graph_execution_s": graph_execution_s,
                "session_close_s": session_close_s,
                "total_route_s": total_route_s,
                "total_s": total_route_s,
                "task_execution_s": task_time_s,
                "orchestration_s": max(0.0, total_route_s - task_time_s),
                "per_task_s": {
                    record["task_id"]: float(record.get("execution_time_s", 0.0))
                    for record in task_records
                },
            },
            "final_tensor_id": final_id,
            "final_transpose_applied": transposed,
            "live_tensor_ids": self.store.live_ids(),
            "released_tensor_ids": tuple(
                getattr(self.store, "released_tensor_ids", ())
            ),
            "circuit_semantics_hash": graph.circuit_semantics_hash,
            "tensor_network_hash": graph.tensor_network_hash,
            "contraction_plan_hash": graph.contraction_plan_hash,
            "contraction_path_structure_hash": contraction_path_structure_hash(graph),
        }
        return WholeGraphExecution(output=np.asarray(output), metadata=metadata)

    @staticmethod
    def _validate_graph(graph: TaskGraph) -> None:
        task_ids = [task.id for task in graph.tasks]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("TaskGraph contains duplicate task ids")
        output_ids = [task.output_tensor_id for task in graph.tasks]
        if len(output_ids) != len(set(output_ids)):
            raise ValueError("TaskGraph contains duplicate output tensor ids")
        task_id_set = set(task_ids)
        producer_by_tensor = {task.output_tensor_id: task.id for task in graph.tasks}
        for task in graph.tasks:
            unknown = set(task.dependencies) - task_id_set
            if unknown:
                raise ValueError(
                    f"Task {task.id} has unknown dependencies: {sorted(unknown)}"
                )
            expected_dependencies = {
                producer_by_tensor[tensor_id]
                for tensor_id in task.input_tensor_ids
                if tensor_id in producer_by_tensor
            }
            declared_dependencies = set(task.dependencies)
            missing = expected_dependencies - declared_dependencies
            extra = declared_dependencies - expected_dependencies
            if missing or extra:
                raise ValueError(
                    f"Task {task.id} dependencies do not match produced inputs: "
                    f"missing={sorted(missing)}, extra={sorted(extra)}"
                )


def _input_use_counts(graph: TaskGraph) -> dict[str, int]:
    counts: dict[str, int] = {}
    for task in graph.tasks:
        for tensor_id in task.input_tensor_ids:
            counts[tensor_id] = counts.get(tensor_id, 0) + 1
    return counts


def _final_tensor_id(graph: TaskGraph) -> str:
    outputs = {task.output_tensor_id for task in graph.tasks}
    consumed = {
        tensor_id for task in graph.tasks for tensor_id in task.input_tensor_ids
    }
    final_ids = outputs - consumed
    if len(final_ids) != 1:
        raise ValueError(
            f"TaskGraph must have exactly one final tensor, got {sorted(final_ids)}"
        )
    return next(iter(final_ids))


def _order_final_tensor(
    array: np.ndarray,
    actual_labels: tuple[int, ...],
    output_labels: tuple[int, ...],
) -> tuple[np.ndarray, bool]:
    if actual_labels == output_labels:
        return np.asarray(array), False
    if len(actual_labels) != len(output_labels) or set(actual_labels) != set(
        output_labels
    ):
        raise ValueError(
            f"Final tensor labels {actual_labels} do not match requested output labels {output_labels}"
        )
    axes = tuple(actual_labels.index(label) for label in output_labels)
    return np.asarray(np.transpose(array, axes)), True
