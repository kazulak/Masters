from __future__ import annotations

from dataclasses import replace
from typing import AbstractSet, Any, Mapping

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
from quantum_bench.whole_circuit.core import (
    NumpyCpuEngine,
    WholeGraphExecutor,
)
from quantum_bench.whole_circuit.policies import Float32RealPolicy
from quantum_bench.whole_circuit.strategies import (
    DecompositionStrategy,
    DefaultScheduler,
    KernelProvider,
    PlacementStrategy,
    ReadyTaskOrderPolicy,
    ReductionProvider,
    Scheduler,
    SerialReadyTaskOrderPolicy,
    SequentialScheduler,
    StrategyIdentity,
    StrategyRole,
    TaskScheduler,
)


def _diamond_graph() -> tuple[TaskGraph, object]:
    """Create a diamond TaskGraph: root -> (branch_1, branch_2) -> join."""
    circuit = builtin_circuit("bell_2q")
    spec_a = TensorSpec("a", (0,), (2,), "dense", dtype="float32")
    spec_b = TensorSpec("b", (0, 1), (2, 2), "dense", dtype="float32")
    spec_c = TensorSpec("c", (1,), (2,), "dense", dtype="float32")
    spec_d = TensorSpec("d", (1,), (2,), "dense", dtype="float32")
    network_spec = TensorNetworkSpec(
        circuit,
        (spec_a, spec_b, spec_c, spec_d),
        (),
        "a,ab,b,b->",
    )
    root = ContractionTask(
        "t0_root",
        ("a", "b"),
        "root_out",
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
    branch_1 = ContractionTask(
        "t1_branch_first",
        ("root_out", "c"),
        "b1_out",
        ("t0_root",),
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
    branch_2 = ContractionTask(
        "t2_branch_second",
        ("root_out", "d"),
        "b2_out",
        ("t0_root",),
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
    join = ContractionTask(
        "t3_join",
        ("b1_out", "b2_out"),
        "final_out",
        ("t1_branch_first", "t2_branch_second"),
        ",->",
        ((), ()),
        (),
        (),
        (),
        (),
        (),
        1,
        1,
        1,
        "dense",
        1,
        0,
    )
    graph = TaskGraph(
        network_spec,
        (join, branch_2, branch_1, root),
        ((0, 1),) * 4,
        PathSummary("diamond", "diamond", 4, 1, None, None, "diamond"),
        0.0,
    )
    network = type(
        "Network",
        (),
        {
            "tensors": (
                TensorValue(spec_a, np.array([1.0, 2.0], dtype=np.float32)),
                TensorValue(
                    spec_b, np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
                ),
                TensorValue(spec_c, np.array([3.0, 4.0], dtype=np.float32)),
                TensorValue(spec_d, np.array([5.0, 6.0], dtype=np.float32)),
            )
        },
    )()
    return graph, network


def test_scheduler_protocols_and_aliases() -> None:
    policy = SerialReadyTaskOrderPolicy()
    assert isinstance(policy, ReadyTaskOrderPolicy)
    assert isinstance(policy, Scheduler)
    assert isinstance(policy, TaskScheduler)
    assert policy.name == "sequential_sorted_ready"

    default_scheduler = DefaultScheduler()
    assert isinstance(default_scheduler, ReadyTaskOrderPolicy)
    assert isinstance(SequentialScheduler(), ReadyTaskOrderPolicy)
    assert default_scheduler.name == "sequential_sorted_ready"


def test_strategy_protocols_pure_contracts() -> None:
    class CustomDecomp:
        name = "custom_decomp"

        def identity(self) -> StrategyIdentity:
            return StrategyIdentity(
                StrategyRole.DECOMPOSITION, self.name, "1", "test", "none"
            )

        def decompose(
            self,
            task: ContractionTask,
            left: np.ndarray,
            right: np.ndarray,
            *,
            limits: Any = None,
        ) -> Any:
            return {"task": task, "left": left, "right": right}

    class CustomPlacement:
        name = "custom_placement"

        def identity(self) -> StrategyIdentity:
            return StrategyIdentity(
                StrategyRole.PLACEMENT, self.name, "1", "test", "none"
            )

        def place(
            self,
            decomposition: Any,
            **kwargs: Any,
        ) -> Any:
            return (decomposition,)

        def place_waves(
            self,
            tiles: tuple[Any, ...],
            total_dpu_count: int,
        ) -> tuple[tuple[Any, ...], ...]:
            return (tiles,)

        def map_wave_to_ranks(
            self,
            wave: tuple[Any, ...],
            ranks: tuple[Any, ...],
        ) -> list[tuple[Any, list[tuple[Any, int]]]]:
            return [(ranks[0], [(tile, index) for index, tile in enumerate(wave)])]

    class CustomKernel:
        name = "custom_kernel"

        def identity(self) -> StrategyIdentity:
            return StrategyIdentity(StrategyRole.KERNEL, self.name, "1", "test", "none")

        def execute_kernel(
            self,
            work_units: Any,
            **kwargs: Any,
        ) -> Any:
            return work_units

        def build_work_unit(
            self,
            tile: Any,
            local_id: int,
            left: np.ndarray,
            right: np.ndarray,
            packed: bool,
        ) -> Any:
            return {"tile": tile, "local_id": local_id}

        def prepare_request(
            self,
            root: Any,
            *,
            profile: Any,
            lowering: Any,
            work_units: list[Any],
            task_contract_sha256: str,
            request_sequence: int,
        ) -> Any:
            return {"work_units": work_units}

        def read_output(
            self,
            path: Any,
            tile: Any,
            *,
            packed: bool,
        ) -> np.ndarray:
            return np.zeros((1, 1), dtype=np.float32)

    class CustomReduction:
        name = "custom_reduction"

        def identity(self) -> StrategyIdentity:
            return StrategyIdentity(
                StrategyRole.REDUCTION, self.name, "1", "test", "none"
            )

        def reduce(
            self,
            lowering: Any,
            partials: Mapping[str, np.ndarray],
            *,
            packed: bool = False,
            scale: float = 1.0,
        ) -> np.ndarray:
            return np.ones((1, 1), dtype=np.float32)

    assert isinstance(CustomDecomp(), DecompositionStrategy)
    assert isinstance(CustomPlacement(), PlacementStrategy)
    assert isinstance(CustomKernel(), KernelProvider)
    assert isinstance(CustomReduction(), ReductionProvider)


def test_default_scheduler_exact_order_and_dependencies() -> None:
    graph, network = _diamond_graph()
    executor = WholeGraphExecutor(NumpyCpuEngine(), Float32RealPolicy())
    result = executor.execute(graph, network)

    assert result.metadata["task_order_policy"] == "sequential_sorted_ready"
    assert result.metadata["task_execution_mode"] == "serial"
    assert result.metadata["executed_order"] == (
        "t0_root",
        "t1_branch_first",
        "t2_branch_second",
        "t3_join",
    )
    # root: [1, 2] @ eye = [1, 2]
    # b1: [1, 2] . [3, 4] = 3 + 8 = 11
    # b2: [1, 2] . [5, 6] = 5 + 12 = 17
    # join: 11 * 17 = 187
    assert result.output == pytest.approx(187.0)


class ReverseReadyTaskOrderPolicy:
    """Test policy that orders ready tasks in reverse alphabetical order."""

    name = "reverse_ready_task_order"

    def __init__(self) -> None:
        self.invocations: list[list[str]] = []

    def select_ready_tasks(
        self,
        pending: Mapping[str, ContractionTask],
        completed: set[str],
    ) -> list[ContractionTask]:
        ready = sorted(
            (task for task in pending.values() if set(task.dependencies) <= completed),
            key=lambda task: task.id,
            reverse=True,
        )
        self.invocations.append([t.id for t in ready])
        return ready


class ReadOnlyStateTaskOrderPolicy:
    name = "read_only_state_task_order"

    def __init__(self) -> None:
        self.pending_was_read_only = False
        self.completed_was_immutable = False

    def select_ready_tasks(
        self,
        pending: Mapping[str, ContractionTask],
        completed: AbstractSet[str],
    ) -> list[ContractionTask]:
        self.pending_was_read_only = not hasattr(pending, "__setitem__")
        self.completed_was_immutable = not hasattr(completed, "add")
        return sorted(
            (task for task in pending.values() if set(task.dependencies) <= completed),
            key=lambda task: task.id,
        )


def test_injected_task_order_policy_is_invoked_and_alters_execution_order() -> None:
    graph, network = _diamond_graph()
    policy = ReverseReadyTaskOrderPolicy()
    executor = WholeGraphExecutor(
        NumpyCpuEngine(), Float32RealPolicy(), task_order_policy=policy
    )
    result = executor.execute(graph, network)

    assert len(policy.invocations) > 0
    assert result.metadata["task_order_policy"] == "reverse_ready_task_order"
    assert result.metadata["task_execution_mode"] == "serial"
    # When root is completed, both t1 and t2 become ready.
    # Reverse scheduler executes t2_branch_second before t1_branch_first.
    assert result.metadata["executed_order"] == (
        "t0_root",
        "t2_branch_second",
        "t1_branch_first",
        "t3_join",
    )
    assert result.output == pytest.approx(187.0)


def test_task_order_policy_receives_read_only_state_snapshots() -> None:
    graph, network = _diamond_graph()
    policy = ReadOnlyStateTaskOrderPolicy()

    result = WholeGraphExecutor(
        NumpyCpuEngine(), Float32RealPolicy(), task_order_policy=policy
    ).execute(graph, network)

    assert policy.pending_was_read_only
    assert policy.completed_was_immutable
    assert result.output == pytest.approx(187.0)


def test_task_order_policy_rejects_forged_pending_task() -> None:
    graph, network = _diamond_graph()

    class ForgedTaskPolicy:
        name = "forged_task_policy"

        def select_ready_tasks(
            self,
            pending: Mapping[str, ContractionTask],
            completed: set[str],
        ) -> list[ContractionTask]:
            ready = sorted(
                (
                    task
                    for task in pending.values()
                    if set(task.dependencies) <= completed
                ),
                key=lambda task: task.id,
            )
            return [
                replace(task, input_tensor_ids=("root_out", "d"))
                if task.id == "t1_branch_first"
                else task
                for task in ready
            ]

    executor = WholeGraphExecutor(
        NumpyCpuEngine(), Float32RealPolicy(), task_order_policy=ForgedTaskPolicy()
    )
    with pytest.raises(ValueError, match="canonical pending task"):
        executor.execute(graph, network)


def test_executor_task_order_policy_equivalence_and_scheduler_alias() -> None:
    graph, network = _diamond_graph()

    default_executor = WholeGraphExecutor(NumpyCpuEngine(), Float32RealPolicy())
    explicit_executor = WholeGraphExecutor(
        NumpyCpuEngine(), Float32RealPolicy(), scheduler=SequentialScheduler()
    )

    res_default = default_executor.execute(graph, network)
    res_explicit = explicit_executor.execute(graph, network)

    np.testing.assert_array_equal(res_default.output, res_explicit.output)
    assert (
        res_default.metadata["executed_order"]
        == res_explicit.metadata["executed_order"]
    )
    assert (
        res_default.metadata["task_order_policy"]
        == res_explicit.metadata["task_order_policy"]
    )
