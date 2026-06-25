from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from quantum_bench.bench.config import load_suite
from quantum_bench.circuits import builtin_circuit
from quantum_bench.core.records import BenchmarkContext, PathSummary, TaskGraph
from quantum_bench.providers.exact_tn.cpu_einsum import CpuTnEinsumExactRoute
from quantum_bench.tn import build_tensor_network, plan_task_graph
from quantum_bench.validation import compute_reference, validate


ROOT = Path(__file__).resolve().parents[1]


def _context() -> BenchmarkContext:
    suite = load_suite(ROOT / "configs" / "suites" / "smoke.yml")
    return BenchmarkContext(
        root_dir=ROOT,
        run_dir=ROOT / "runs" / "test",
        suite=suite,
        case=suite["cases"][0],
        route_config=suite["_route_configs"][0],
        repeat_id=0,
        tolerances=suite["tolerances"],
        timeout_s=suite.get("timeout_s"),
        memory_guard_gib=suite.get("memory_guard_gib"),
    )


def _execute_builtin(name: str, params: dict | None = None):
    route = CpuTnEinsumExactRoute()
    circuit = builtin_circuit(name, params)
    network = build_tensor_network(circuit)
    graph = plan_task_graph(network)
    result = route.execute(route.prepare(graph, network, _context()), _context())
    reference, _ = compute_reference(network)
    return graph, result, reference


def test_cpu_task_sequence_matches_reference_for_bell() -> None:
    graph, result, reference = _execute_builtin("bell_2q")
    metrics = result.metadata["task_metrics"]

    assert result.status == "passed"
    assert validate(result.output.array, reference).passed
    assert result.metadata["execution_engine"] == "task_sequence_np_einsum"
    assert result.metadata["task_count"] == len(graph.tasks)
    assert len(metrics) == len(graph.tasks)
    assert result.metadata["peak_intermediate_bytes"] >= result.metadata["max_intermediate_tensor_bytes"]
    for metric in metrics:
        assert metric.task_id
        assert metric.input_tensor_ids
        assert metric.output_tensor_id
        assert metric.input_shapes
        assert metric.output_shape
        assert metric.estimated_flops >= 0
        assert metric.estimated_bytes >= 0
        assert metric.execution_time_s >= 0.0
        assert metric.intermediate_tensor_bytes > 0


def test_cpu_task_sequence_reorders_final_tensor_by_labels() -> None:
    graph, result, reference = _execute_builtin("bv", {"n_qubits": 4})

    assert result.metadata["final_tensor_labels"] != result.metadata["output_labels"]
    assert result.metadata["final_transpose_applied"] is True
    assert result.output.shape == reference.shape
    assert validate(result.output.array, reference).passed
    assert np.max(np.abs(result.output.array - reference)) < 1.0e-12


def test_empty_single_tensor_graph_returns_tensor_directly() -> None:
    route = CpuTnEinsumExactRoute()
    circuit = builtin_circuit("qrng", {"n_qubits": 1})
    idle_circuit = type(circuit)(circuit.name, circuit.n_qubits, (), circuit.source)
    network = build_tensor_network(idle_circuit)
    graph = TaskGraph(
        network=network.spec,
        tasks=(),
        path=(),
        path_summary=PathSummary("unit_test", "manual", 0, None, None, None, ""),
        planning_time_s=0.0,
    )
    result = route.execute(route.prepare(graph, network, _context()), _context())

    assert len(graph.tasks) == 0
    assert result.metadata["task_count"] == 0
    assert result.metadata["task_metrics"] == []
    assert result.metadata["final_transpose_applied"] is False
    assert np.array_equal(result.output.array, np.array([1.0, 0.0], dtype=np.complex128))


def test_empty_multi_tensor_graph_is_rejected() -> None:
    route = CpuTnEinsumExactRoute()
    circuit = builtin_circuit("qrng", {"n_qubits": 2})
    idle_circuit = type(circuit)(circuit.name, circuit.n_qubits, (), circuit.source)
    network = build_tensor_network(idle_circuit)
    empty_graph = TaskGraph(
        network=network.spec,
        tasks=(),
        path=(),
        path_summary=PathSummary("unit_test", "manual", 0, None, None, None, ""),
        planning_time_s=0.0,
    )

    with pytest.raises(ValueError, match="Cannot execute empty TaskGraph"):
        route.execute(route.prepare(empty_graph, network, _context()), _context())
