from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import quantum_bench.targets.upmem.hardware_taskgraph_runtime as runtime
from quantum_bench.core.records import (
    CircuitSpec,
    ContractionTask,
    PathSummary,
    TaskGraph,
    TensorSpec,
    TensorValue,
    TensorNetworkSpec,
)
from quantum_bench.tn.network import TensorNetworkValue
from quantum_bench.targets.upmem.hardware_session import HardwareSessionExecution
from quantum_bench.targets.upmem.hardware_taskgraph import load_hardware_taskgraph_suite


ROOT = Path(__file__).resolve().parents[1]
PROFILE = load_hardware_taskgraph_suite(
    ROOT / "configs/suites/upmem_hardware_taskgraph_correctness.yml"
).profile


def _task(
    task_id: str,
    left: str,
    right: str,
    output: str,
    deps: tuple[str, ...],
    left_labels: tuple[int, ...],
    right_labels: tuple[int, ...],
    output_labels: tuple[int, ...],
    contracted: tuple[int, ...],
) -> ContractionTask:
    return ContractionTask(
        task_id,
        (left, right),
        output,
        deps,
        "ab,bc->ac",
        (tuple(2 for _ in left_labels), tuple(2 for _ in right_labels)),
        tuple(2 for _ in output_labels),
        left_labels,
        right_labels,
        contracted,
        output_labels,
        2,
        2,
        2,
        "dense",
        8,
        64,
    )


def _graph(
    left: np.ndarray, right: np.ndarray, *, second: bool = False
) -> tuple[TaskGraph, TensorNetworkValue]:
    specs = (
        TensorSpec("a", (0, 1), left.shape, "dense"),
        TensorSpec("b", (1, 2), right.shape, "dense"),
    )
    tasks = [_task("first", "a", "b", "mid", (), (0, 1), (1, 2), (0, 2), (1,))]
    arrays = [TensorValue(specs[0], left), TensorValue(specs[1], right)]
    if second:
        c = np.asarray([[2.0, -1.0], [1.0, 3.0]])
        specs += (TensorSpec("c", (2, 3), c.shape, "dense"),)
        arrays.append(TensorValue(specs[2], c))
        tasks.append(
            _task("second", "mid", "c", "out", ("first",), (0, 2), (2, 3), (0, 3), (2,))
        )
        output_labels = (0, 3)
    else:
        output_labels = (0, 2)
    circuit = CircuitSpec("test", 2, (), {})
    network_spec = TensorNetworkSpec(circuit, tuple(specs), output_labels, "ab,bc->ac")
    network = TensorNetworkValue(network_spec, arrays)
    graph = TaskGraph(
        network_spec,
        tuple(tasks),
        (),
        PathSummary("test", "greedy", len(tasks), None, None, None, ""),
        0.0,
    )
    return graph, network


def _build(root: Path) -> SimpleNamespace:
    root.mkdir(parents=True)
    return SimpleNamespace(
        session_root=root,
        build_dir=root,
        build_time_s=0.0,
        source_tree_hash="s",
        host_binary_hash="h",
        dpu_binary_hash="d",
        build_command=("fake",),
        sdk_tools={},
    )


def _fake_executor(calls: list[list[str]], *, perturb: float = 0.0):
    def execute(
        build: object,
        *,
        session_id: str,
        tasks: list[object],
        profile: object,
        environment: object,
    ) -> HardwareSessionExecution:
        calls.append([task.task_id for task in tasks])
        for task in tasks:
            expected = np.asarray(
                task.preparation.prepared_operands.expected_reference_output
            )
            if task.operand_mode == "float32_no_quant":
                expected.astype("<f4").tofile(task.output_path)
            else:
                scale = task.output_scale
                np.rint(expected / scale).astype("<i4").tofile(task.output_path)
        return HardwareSessionExecution(
            "completed",
            None,
            build.session_root / (session_id + ".json"),
            {"tasks": [{"timing": {}} for _ in tasks]},
            0.0,
            (),
            "",
            "",
        )

    return execute


def test_dependent_graph_uses_first_native_output_as_second_input(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    graph, network = _graph(
        np.arange(4, dtype=float).reshape(2, 2),
        np.arange(4, dtype=float).reshape(2, 2),
        second=True,
    )
    calls: list[list[str]] = []
    monkeypatch.setattr(runtime, "execute_hardware_session", _fake_executor(calls))
    result = runtime.execute_hardware_taskgraph_runtime(
        root_dir=tmp_path,
        work_dir=tmp_path / "native" / "run",
        graph=graph,
        network=network,
        case_id="dependent",
        quantization_mode="none",
        profile=PROFILE,
        environment={},
        native_build=_build(tmp_path / "native"),
    )
    assert result.status == "completed"
    assert calls == [["first__real"], ["second__real"]]
    assert np.allclose(
        result.output,
        np.einsum(
            "ab,bc,cd->ad",
            network.tensors[0].array,
            network.tensors[1].array,
            network.tensors[2].array,
        ),
    )
    assert result.summary["cpu_fallback_used"] is False


def test_split_complex_invokes_four_components_and_reconstructs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    left = np.array([[1 + 2j, 3 - 1j], [2 + 0.5j, -1 + 4j]])
    right = np.array([[2 - 1j, 1 + 3j], [4 + 2j, -2 + 0.5j]])
    graph, network = _graph(left, right)
    calls: list[list[str]] = []
    monkeypatch.setattr(runtime, "execute_hardware_session", _fake_executor(calls))
    result = runtime.execute_hardware_taskgraph_runtime(
        root_dir=tmp_path,
        work_dir=tmp_path / "native" / "run",
        graph=graph,
        network=network,
        case_id="complex",
        quantization_mode="none",
        profile=PROFILE,
        environment={},
        native_build=_build(tmp_path / "native"),
    )
    assert result.status == "completed"
    assert calls == [["first__ar_br", "first__ai_bi", "first__ar_bi", "first__ai_br"]]
    assert result.task_metrics[0]["split_complex_component_count"] == 4
    assert np.allclose(result.output, np.einsum("ab,bc->ac", left, right))


def test_int8_validation_reports_policy_and_full_precision_error_independently(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    graph, network = _graph(
        np.array([[0.1, 1.7], [-2.2, 3.3]]), np.array([[1.4, -0.7], [2.6, 0.2]])
    )
    monkeypatch.setattr(runtime, "execute_hardware_session", _fake_executor([]))
    result = runtime.execute_hardware_taskgraph_runtime(
        root_dir=tmp_path,
        work_dir=tmp_path / "native" / "run",
        graph=graph,
        network=network,
        case_id="int8",
        quantization_mode="per_task_input_quantize",
        profile=PROFILE,
        environment={},
        native_build=_build(tmp_path / "native"),
        reference_output=np.einsum(
            "ab,bc->ac", network.tensors[0].array, network.tensors[1].array
        ),
    )
    assert result.status == "completed"
    assert result.summary["final_validation"]["reference_kind"].startswith(
        "native_int8"
    )
    assert result.summary["policy_reference_accuracy"]["passed"] is True
    assert result.summary["full_precision_accuracy"]["max_abs_error"] > 0.0
    assert (
        result.summary["quantization_max_abs_error"]
        == result.summary["full_precision_accuracy"]["max_abs_error"]
    )


def test_float32_does_not_emit_a_quantization_error_value(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    graph, network = _graph(
        np.array([[0.1, 1.7], [-2.2, 3.3]]),
        np.array([[1.4, -0.7], [2.6, 0.2]]),
    )
    monkeypatch.setattr(runtime, "execute_hardware_session", _fake_executor([]))

    result = runtime.execute_hardware_taskgraph_runtime(
        root_dir=tmp_path,
        work_dir=tmp_path / "native" / "run",
        graph=graph,
        network=network,
        case_id="float32",
        quantization_mode="none",
        profile=PROFILE,
        environment={},
        native_build=_build(tmp_path / "native"),
    )

    assert result.status == "completed"
    assert result.summary["quantization_max_abs_error"] is None
    assert result.summary["full_precision_accuracy"]["max_abs_error"] is not None


def test_native_failure_stops_without_fallback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    graph, network = _graph(np.ones((2, 2)), np.ones((2, 2)), second=True)
    calls: list[list[str]] = []

    def failed(*args: object, **kwargs: object) -> HardwareSessionExecution:
        calls.append([task.task_id for task in kwargs["tasks"]])
        return HardwareSessionExecution(
            "failed", "kernel_failed", tmp_path / "response.json", {}, 0.0, (), "", ""
        )

    monkeypatch.setattr(runtime, "execute_hardware_session", failed)
    result = runtime.execute_hardware_taskgraph_runtime(
        root_dir=tmp_path,
        work_dir=tmp_path / "native" / "run",
        graph=graph,
        network=network,
        case_id="failed",
        quantization_mode="none",
        profile=PROFILE,
        environment={},
        native_build=_build(tmp_path / "native"),
    )
    assert result.status == "failed"
    assert result.reason == "kernel_failed"
    assert calls == [["first__real"]]
    assert result.summary["cpu_fallback_used"] is False
