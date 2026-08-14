from __future__ import annotations

import numpy as np
import pytest

from quantum_bench.circuits import builtin_circuit
from quantum_bench.core.records import (
    ContractionTask,
    PathSummary,
    TaskGraph,
    TensorNetworkSpec,
    TensorSpec,
    TensorValue,
)
from quantum_bench.tn import build_tensor_network, plan_task_graph
from quantum_bench.tn.contract import contract_binary_task
from quantum_bench.tn.execution import execute_task_sequence_np_einsum
from quantum_bench.whole_circuit import (
    DeviceTopology,
    EngineTaskResult,
    Float32RealPolicy,
    HostPackedInt8Policy,
    InMemoryTensorStore,
    NumpyCpuEngine,
    WholeGraphExecutor,
)


def test_builtin_circuit_executes_as_a_complete_graph() -> None:
    network = build_tensor_network(builtin_circuit("bell_2q"))
    graph = plan_task_graph(network)

    result = WholeGraphExecutor(NumpyCpuEngine(), Float32RealPolicy()).execute(
        graph, network
    )

    assert result.metadata["task_count"] == len(graph.tasks)
    assert result.metadata["executed_order"] == tuple(task.id for task in graph.tasks)
    assert result.metadata["contraction_plan_hash"] == graph.contraction_plan_hash
    assert result.metadata["contraction_path_structure_hash"]
    assert result.output.shape == (2, 2)


def _chain_graph() -> tuple[TaskGraph, object]:
    circuit = builtin_circuit("bell_2q")
    left = TensorSpec("left", (0,), (2,), "dense", dtype="float32")
    right = TensorSpec("right", (0, 1), (2, 2), "dense", dtype="float32")
    third = TensorSpec("third", (1,), (2,), "dense", dtype="float32")
    network_spec = TensorNetworkSpec(circuit, (left, right, third), (), "a,ab,b->")
    first = ContractionTask(
        "first",
        ("left", "right"),
        "mid",
        (),
        "a,ab->b",
        ((2,), (2, 2)),
        (2,),
        (0,),
        (0, 1),
        (0,),
        (1,),
        1,
        2,
        1,
        "dense",
        4,
        0,
    )
    second = ContractionTask(
        "second",
        ("mid", "third"),
        "out",
        ("first",),
        "b,b->",
        ((2,), (2,)),
        (),
        (0,),
        (0,),
        (0,),
        (),
        1,
        2,
        1,
        "dense",
        4,
        0,
    )
    graph = TaskGraph(
        network_spec,
        (second, first),
        ((0, 1), (0, 1)),
        PathSummary("test", "test", 2, 1, None, None, "test"),
        0.0,
    )
    network = type(
        "Network",
        (),
        {
            "tensors": (
                TensorValue(left, np.array([1, 2], dtype=np.float32)),
                TensorValue(right, np.eye(2, dtype=np.float32)),
                TensorValue(third, np.array([3, 4], dtype=np.float32)),
            )
        },
    )()
    return graph, network


def _branching_graph() -> tuple[TaskGraph, object]:
    circuit = builtin_circuit("bell_2q")
    left = TensorSpec("left", (0,), (2,), "dense", dtype="float32")
    matrix = TensorSpec("matrix", (0, 1), (2, 2), "dense", dtype="float32")
    scale_a = TensorSpec("scale_a", (), (), "dense", dtype="float32")
    scale_b = TensorSpec("scale_b", (), (), "dense", dtype="float32")
    network_spec = TensorNetworkSpec(
        circuit,
        (left, matrix, scale_a, scale_b),
        (),
        "a,ab,,->",
    )
    shared = ContractionTask(
        "shared",
        ("left", "matrix"),
        "shared_out",
        (),
        "a,ab->b",
        ((2,), (2, 2)),
        (2,),
        (0,),
        (0, 1),
        (0,),
        (1,),
        1,
        2,
        1,
        "dense",
        4,
        0,
    )
    branch_a = ContractionTask(
        "branch_a",
        ("shared_out", "scale_a"),
        "branch_a_out",
        ("shared",),
        "b,->b",
        ((2,), ()),
        (2,),
        (1,),
        (),
        (),
        (1,),
        2,
        1,
        1,
        "dense",
        2,
        0,
    )
    branch_b = ContractionTask(
        "branch_b",
        ("shared_out", "scale_b"),
        "branch_b_out",
        ("shared",),
        "b,->b",
        ((2,), ()),
        (2,),
        (1,),
        (),
        (),
        (1,),
        2,
        1,
        1,
        "dense",
        2,
        0,
    )
    final = ContractionTask(
        "final",
        ("branch_a_out", "branch_b_out"),
        "out",
        ("branch_a", "branch_b"),
        "b,b->",
        ((2,), (2,)),
        (),
        (1,),
        (1,),
        (1,),
        (),
        1,
        2,
        1,
        "dense",
        4,
        0,
    )
    graph = TaskGraph(
        network_spec,
        (final, branch_b, shared, branch_a),
        ((0, 1),) * 4,
        PathSummary("test", "test", 4, 2, None, None, "test"),
        0.0,
    )
    network = type(
        "Network",
        (),
        {
            "tensors": (
                TensorValue(left, np.array([1, 2], dtype=np.float32)),
                TensorValue(matrix, np.eye(2, dtype=np.float32)),
                TensorValue(scale_a, np.array(2, dtype=np.float32)),
                TensorValue(scale_b, np.array(3, dtype=np.float32)),
            )
        },
    )()
    return graph, network


class RecordingSession:
    def __init__(
        self, calls: list[str], close_metadata: dict[str, object] | None = None
    ) -> None:
        self.calls = calls
        self.close_metadata = close_metadata
        self.closed = False

    def execute(
        self, task: ContractionTask, left: np.ndarray, right: np.ndarray
    ) -> EngineTaskResult:
        self.calls.append(task.id)
        return EngineTaskResult(
            contract_binary_task(task, left, right, dtype=np.float32)
        )

    def close(self) -> dict[str, object] | None:
        self.closed = True
        return self.close_metadata


class RecordingEngine:
    name = "recording"

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.session: RecordingSession | None = None

    def open_session(
        self, policy: object, topology: DeviceTopology
    ) -> RecordingSession:
        self.session = RecordingSession(
            self.calls, {"release_confirmed": True, "session_id": "test"}
        )
        return self.session


def test_fake_engine_executes_each_task_after_dependencies() -> None:
    graph, network = _chain_graph()
    engine = RecordingEngine()

    result = WholeGraphExecutor(engine, Float32RealPolicy()).execute(graph, network)

    assert engine.calls == ["first", "second"]
    assert result.metadata["task_count"] == 2
    assert result.metadata["session_metadata"] == {
        "release_confirmed": True,
        "session_id": "test",
    }


def test_route_timing_includes_open_graph_and_close() -> None:
    graph, network = _chain_graph()

    result = WholeGraphExecutor(NumpyCpuEngine(), Float32RealPolicy()).execute(
        graph, network
    )

    assert result.metadata["session_metadata"] == {}
    timing = result.metadata["timing"]
    assert timing["session_open_s"] >= 0.0
    assert timing["allocation_time_s"] == timing["session_open_s"]
    assert timing["graph_execution_s"] >= 0.0
    assert timing["session_close_s"] >= 0.0
    assert timing["total_route_s"] >= timing["session_open_s"]
    assert timing["total_route_s"] >= timing["graph_execution_s"]
    assert timing["total_route_s"] >= timing["session_close_s"]
    assert timing["total_s"] == timing["total_route_s"]


class FailingSession(RecordingSession):
    def execute(
        self, task: ContractionTask, left: np.ndarray, right: np.ndarray
    ) -> EngineTaskResult:
        self.calls.append(task.id)
        raise RuntimeError("task failed")


class FailingEngine(RecordingEngine):
    def open_session(self, policy: object, topology: DeviceTopology) -> FailingSession:
        self.session = FailingSession(self.calls)
        return self.session


def test_session_closes_when_task_execution_raises() -> None:
    graph, network = _chain_graph()
    engine = FailingEngine()

    with pytest.raises(RuntimeError, match="task failed"):
        WholeGraphExecutor(engine, Float32RealPolicy()).execute(graph, network)

    assert engine.session is not None
    assert engine.session.closed is True


def test_dependency_must_match_produced_input() -> None:
    graph, network = _chain_graph()
    second = graph.tasks[0]
    malformed = second.__class__(**{**second.__dict__, "dependencies": ()})
    malformed_graph = TaskGraph(
        graph.network,
        (malformed, graph.tasks[1]),
        graph.path,
        graph.path_summary,
        graph.planning_time_s,
    )

    with pytest.raises(ValueError, match="dependencies do not match produced inputs"):
        WholeGraphExecutor(NumpyCpuEngine(), Float32RealPolicy()).execute(
            malformed_graph, network
        )


def test_float32_cpu_matches_existing_task_contraction_semantics() -> None:
    network = build_tensor_network(builtin_circuit("bell_2q"))
    graph = plan_task_graph(network)
    expected, _ = execute_task_sequence_np_einsum(graph, network)
    actual = (
        WholeGraphExecutor(NumpyCpuEngine(), Float32RealPolicy())
        .execute(graph, network)
        .output
    )

    assert actual.dtype == np.float32
    assert np.all(np.isfinite(actual))
    assert actual.shape == (2, 2)
    np.testing.assert_allclose(actual, expected, rtol=1.0e-5, atol=1.0e-6)


def test_host_packed_int8_is_deterministic_and_handles_zero_scale() -> None:
    left = TensorSpec("left", (0,), (2,), "dense", dtype="float32")
    right = TensorSpec("right", (0,), (2,), "dense", dtype="float32")
    spec = TensorNetworkSpec(builtin_circuit("bell_2q"), (left, right), (), "a,a->")
    task = ContractionTask(
        "task",
        ("left", "right"),
        "out",
        (),
        "a,a->",
        ((2,), (2,)),
        (),
        (0,),
        (0,),
        (0,),
        (),
        1,
        2,
        1,
        "dense",
        4,
        0,
    )
    graph = TaskGraph(
        spec,
        (task,),
        ((0, 1),),
        PathSummary("test", "test", 1, 1, None, None, "test"),
        0.0,
    )
    network = type(
        "Network",
        (),
        {
            "tensors": (
                TensorValue(left, np.zeros(2, dtype=np.float32)),
                TensorValue(right, np.zeros(2, dtype=np.float32)),
            )
        },
    )()
    policy = HostPackedInt8Policy()

    first = WholeGraphExecutor(NumpyCpuEngine(), policy).execute(graph, network)
    second = WholeGraphExecutor(NumpyCpuEngine(), policy).execute(graph, network)

    np.testing.assert_array_equal(first.output, second.output)
    assert first.metadata["task_metrics"][0]["zero_scale_fallback_used"] is True
    assert first.metadata["numeric_policy"] == "host_packed_int8_per_task_v1"


def test_dead_intermediates_are_released() -> None:
    graph, network = _chain_graph()
    store = InMemoryTensorStore()

    result = WholeGraphExecutor(
        NumpyCpuEngine(), Float32RealPolicy(), store=store
    ).execute(graph, network)

    assert "mid" in result.metadata["released_tensor_ids"]
    assert result.metadata["live_tensor_ids"] == ("out",)


def test_intermediate_fanout_remains_live_until_both_consumers_finish() -> None:
    graph, network = _branching_graph()
    engine = RecordingEngine()

    result = WholeGraphExecutor(engine, Float32RealPolicy()).execute(graph, network)

    assert engine.calls == ["shared", "branch_a", "branch_b", "final"]
    assert result.output == pytest.approx(30.0)
    assert result.metadata["released_tensor_ids"].count("shared_out") == 1
    assert result.metadata["live_tensor_ids"] == ("out",)


def test_tensor_store_rejects_reuse_across_executions() -> None:
    graph, network = _chain_graph()
    executor = WholeGraphExecutor(NumpyCpuEngine(), Float32RealPolicy())
    executor.execute(graph, network)

    with pytest.raises(RuntimeError, match="single-use"):
        executor.execute(graph, network)
