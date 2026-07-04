from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from quantum_bench.bench.config import load_suite
from quantum_bench.circuits import builtin_circuit
from quantum_bench.core.records import BenchmarkContext, PathSummary, TaskGraph
from quantum_bench.providers.exact_tn.cpu_einsum import CpuTnEinsumExactRoute
from quantum_bench.tn import (
    build_slice_aware_taskgraph_model,
    build_tensor_network,
    execute_task_hybrid_slice_frontier_np_einsum,
    execute_task_sequence_np_einsum,
    execute_task_sliced_sequence_np_einsum,
    plan_task_graph,
    validate_slice_aware_taskgraph_model,
)
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


def test_slice_aware_taskgraph_model_expands_one_contraction_without_execution() -> None:
    circuit = builtin_circuit("bv", {"n_qubits": 4})
    network = build_tensor_network(circuit)
    graph = plan_task_graph(network)

    model = build_slice_aware_taskgraph_model(graph, max_slice_count=2)
    valid, reason = validate_slice_aware_taskgraph_model(model)
    metadata = model.to_metadata()

    assert valid is True
    assert reason is None
    assert model.available is True
    assert model.slice_model_kind == "single_task_single_index_model"
    assert model.slice_model_execution_status == "model_only"
    assert model.slice_model_slice_count == 2
    assert model.slice_model_task_count == 2
    assert metadata["slice_aware_taskgraph_available"] is True
    assert metadata["slice_model_slice_count"] == 2
    assert metadata["slice_model_task_count"] == 2
    assert metadata["slice_model_execution_status"] == "model_only"
    assert metadata["source_task_count"] == len(graph.tasks)
    assert metadata["slice_reconstruction_required"] is True
    assert metadata["slice_reconstruction_status"] == "model_only"
    assert metadata["slice_reconstruction_step"]["operation"] == "sum_partials"
    assert metadata["slice_task_execution_mode"] == "model_only"
    assert metadata["hybrid_ready"] is False
    assert metadata["slice_model_sliced_indices"] == list(model.sliced_indices)
    assert len(metadata["slice_model_tasks"]) == model.slice_model_task_count
    assert len(metadata["slice_dependency_rewrites"]) == len(model.downstream_dependency_rewrites)
    assert "slice_count" not in metadata
    assert "parallelism_mode" not in metadata

    selected_task = next(task for task in graph.tasks if task.id == model.sliced_task_id)
    assert model.sliced_indices == selected_task.contracted_labels[:1]
    assert model.reconstruction_step is not None
    assert model.reconstruction_step.output_tensor_id == selected_task.output_tensor_id
    assert model.reconstruction_step.dependencies == tuple(task.id for task in model.slice_tasks)
    assert model.reconstruction_step.input_tensor_ids == tuple(task.partial_output_tensor_id for task in model.slice_tasks)

    assert len(model.downstream_dependency_rewrites) >= 1
    for rewrite in model.downstream_dependency_rewrites:
        assert rewrite.old_dependency == selected_task.id
        assert rewrite.new_dependency == model.reconstruction_step.id

    for slice_id, slice_task in enumerate(model.slice_tasks):
        assert slice_task.id == f"{selected_task.id}__slice_{slice_id}"
        assert slice_task.source_task_id == selected_task.id
        assert slice_task.assignment.value == slice_id
        assert slice_task.assignment.label == model.sliced_indices[0]
        assert slice_task.dependencies == selected_task.dependencies
        assert slice_task.input_tensor_ids == selected_task.input_tensor_ids
        assert slice_task.output_shape == selected_task.output_shape
        assert slice_task.output_labels == selected_task.output_labels
        assert len(slice_task.input_restrictions) == 2
        assert metadata["slice_model_tasks"][slice_id]["partial_output_tensor_id"] == slice_task.partial_output_tensor_id
        assert len(metadata["slice_model_tasks"][slice_id]["input_restrictions"]) == 2


def test_slice_aware_taskgraph_model_reports_unsupported_empty_graph() -> None:
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

    model = build_slice_aware_taskgraph_model(graph)
    valid, reason = validate_slice_aware_taskgraph_model(model)
    metadata = model.to_metadata()

    assert valid is False
    assert reason == "no_contraction_tasks"
    assert model.available is False
    assert model.rejection_reason == "no_contraction_tasks"
    assert metadata["slice_aware_taskgraph_available"] is False
    assert metadata["slice_model_execution_status"] == "unsupported"
    assert metadata["slice_model_rejection_reason"] == "no_contraction_tasks"
    assert metadata["source_task_count"] == 0
    assert metadata["slice_model_slice_count"] == 0
    assert metadata["slice_model_task_count"] == 0
    assert metadata["slice_model_tasks"] == []
    assert metadata["slice_reconstruction_step"] is None
    assert metadata["slice_dependency_rewrites"] == []
    assert metadata["slice_task_execution_mode"] == "not_applicable"
    assert metadata["slice_reconstruction_required"] is False
    assert metadata["hybrid_ready"] is False
    assert "slice_count" not in metadata


def test_executable_sequential_internal_slicing_matches_taskgraph_baseline() -> None:
    circuit = builtin_circuit("bv", {"n_qubits": 4})
    network = build_tensor_network(circuit)
    graph = plan_task_graph(network)
    expected, _ = execute_task_sequence_np_einsum(graph, network)

    output, metadata = execute_task_sliced_sequence_np_einsum(graph, network, max_slice_count=2)

    np.testing.assert_allclose(output, expected, atol=1.0e-12)
    assert metadata["parallelism_mode"] == "slicing"
    assert metadata["parallelism_evidence_type"] == "executed"
    assert metadata["slice_model_execution_status"] == "executed"
    assert metadata["slice_task_execution_mode"] == "sequential"
    assert metadata["slice_model_executed_task_count"] == metadata["slice_model_task_count"] == 2
    assert metadata["source_task_count"] == len(graph.tasks)
    assert metadata["source_task_completion_count"] == metadata["source_task_count"]
    assert metadata["slice_reconstruction_status"] == "completed"
    assert metadata["hybrid_ready"] is False
    assert metadata["frontier_scheduler_enabled"] is False
    assert metadata["hybrid_reconstruction_validation_status"] == "passed"
    assert metadata["dependency_violation_detected"] is False
    assert set(metadata["executed_source_task_ids"]) == {task.id for task in graph.tasks}
    assert len(metadata["executed_source_task_ids"]) == len(graph.tasks)
    assert len(metadata["executed_slice_task_ids"]) == metadata["slice_model_task_count"]
    assert metadata["slice_parallel_wave_count"] == 0


def test_hybrid_slice_frontier_matches_taskgraph_baseline_for_worker_counts() -> None:
    circuit = builtin_circuit("xor", {"n_qubits": 4})
    network = build_tensor_network(circuit)
    graph = plan_task_graph(network)
    expected, _ = execute_task_sequence_np_einsum(graph, network)

    for worker_count in (1, 2):
        output, metadata = execute_task_hybrid_slice_frontier_np_einsum(
            graph,
            network,
            frontier_worker_count=worker_count,
            max_slice_count=2,
        )

        np.testing.assert_allclose(output, expected, atol=1.0e-12)
        assert metadata["parallelism_mode"] == "hybrid"
        assert metadata["parallelism_evidence_type"] == "executed"
        assert metadata["hybrid_components"] == ["slicing", "frontier"]
        assert metadata["slicing_backend"] == "internal_taskgraph"
        assert metadata["frontier_scheduler_enabled"] is True
        assert metadata["frontier_worker_count"] == worker_count
        assert metadata["frontier_parallel_execution"] is (worker_count > 1)
        if worker_count > 1:
            assert metadata["slice_parallel_wave_count"] > 0
        else:
            assert metadata["slice_parallel_wave_count"] == 0
        assert metadata["slice_parallel_execution"] is (metadata["slice_parallel_wave_count"] > 0)
        assert metadata["slice_task_execution_mode"] == "frontier_scheduled"
        assert metadata["hybrid_ready"] is True
        assert metadata["slice_model_execution_status"] == "executed"
        assert metadata["slice_reconstruction_status"] == "completed"
        assert metadata["hybrid_reconstruction_validation_status"] == "passed"
        assert metadata["source_task_count"] == len(graph.tasks)
        assert metadata["source_task_completion_count"] == metadata["source_task_count"]
        assert metadata["source_frontier_completed_task_count"] == len(graph.tasks)
        assert metadata["frontier_executed_task_count"] == metadata["hybrid_execution_node_count"]
        assert metadata["duplicate_contraction_check"] == "passed"
        assert metadata["missing_dependency_check"] == "passed"
        assert metadata["dependency_violation_detected"] is False
        assert set(metadata["executed_source_task_ids"]) == {task.id for task in graph.tasks}
        assert len(metadata["executed_source_task_ids"]) == len(graph.tasks)
        assert len(metadata["executed_slice_task_ids"]) == metadata["slice_model_task_count"]
