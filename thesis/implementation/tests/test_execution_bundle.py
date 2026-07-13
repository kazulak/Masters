from __future__ import annotations

from dataclasses import replace

import pytest

from quantum_bench.circuits import builtin_circuit
from quantum_bench.tn import (
    build_execution_bundle,
    contraction_path_structure_hash,
    build_tensor_network,
    execution_identity_metadata,
    executor_config_hash,
    plan_task_graph_with_config,
    validate_execution_bundle,
    with_execution_identity,
)


def _graph(optimize: str = "greedy", **config_overrides):
    network = build_tensor_network(builtin_circuit("ghz_chain", {"n_qubits": 4}))
    config = {"engine": "opt_einsum", "optimize": optimize, **config_overrides}
    return plan_task_graph_with_config(network, config)


def test_execution_bundle_hashes_are_stable_and_validate() -> None:
    first = _graph()
    second = _graph()

    assert first.circuit_semantics_hash == second.circuit_semantics_hash
    assert first.tensor_network_hash == second.tensor_network_hash
    assert first.contraction_plan_hash == second.contraction_plan_hash

    bundle = build_execution_bundle(first, case_id="ghz_4q", suite_id="test")
    validate_execution_bundle(bundle, second)
    assert bundle["contraction_plan_hash"] == first.contraction_plan_hash
    assert bundle["contraction_path_structure_hash"] == contraction_path_structure_hash(first)
    assert bundle["provenance"]["planning_in_timed_region"] is False


def test_semantic_task_change_changes_plan_hash_only() -> None:
    graph = _graph()
    task = graph.tasks[0]
    changed_task = replace(task, structure="test_changed_structure")
    changed = replace(
        graph,
        tasks=(changed_task, *graph.tasks[1:]),
        circuit_semantics_hash="",
        tensor_network_hash="",
        contraction_plan_hash="",
    )
    changed = with_execution_identity(changed)

    assert changed.circuit_semantics_hash == graph.circuit_semantics_hash
    assert changed.tensor_network_hash == graph.tensor_network_hash
    assert changed.contraction_plan_hash != graph.contraction_plan_hash


def test_stale_execution_identity_is_rejected() -> None:
    graph = _graph()
    stale = replace(graph, contraction_plan_hash="0" * 64)

    with pytest.raises(ValueError, match="contraction_plan_hash"):
        with_execution_identity(stale)


def test_bundle_mismatch_is_rejected() -> None:
    greedy = _graph("greedy")
    auto = _graph("auto")
    bundle = build_execution_bundle(greedy, case_id="ghz_4q", suite_id="test")

    if greedy.contraction_plan_hash == auto.contraction_plan_hash:
        changed = {**bundle, "contraction_plan_hash": "0" * 64}
        with pytest.raises(ValueError, match="contraction_plan_hash"):
            validate_execution_bundle(changed, greedy)
    else:
        with pytest.raises(ValueError, match="contraction_plan_hash"):
            validate_execution_bundle(bundle, auto)


def test_execution_metadata_and_executor_hash_separate_plan_from_route_config() -> None:
    graph = _graph()
    metadata = execution_identity_metadata(graph, plan_reused=True)

    assert metadata["contraction_plan_hash"] == graph.contraction_plan_hash
    assert metadata["contraction_path_structure_hash"] == contraction_path_structure_hash(graph)
    assert metadata["plan_reused"] is True
    assert metadata["planning_in_timed_region"] is False
    assert executor_config_hash("cpu", {"quantization_mode": "none"}) != executor_config_hash(
        "cpu", {"quantization_mode": "int8"}
    )


def test_wave_2e65_tiled_strategy_relationships_and_hash_invariants() -> None:
    graph = _graph()
    cpu_bundle = build_execution_bundle(graph, case_id="wave_2e65", suite_id="cpu")
    upmem_bundle = build_execution_bundle(graph, case_id="wave_2e65", suite_id="upmem")
    strategy = {
        "name": "mram_resident_output_tiled_v1",
        "max_rank": 16,
        "max_tensor_elements": 65536,
        "max_contracted_combinations": 4096,
        "output_tile": 256,
    }
    cpu_executor = executor_config_hash("cpu", {"strategy": strategy, "quantization_mode": "none"})
    upmem_executor = executor_config_hash("upmem", {"strategy": strategy, "quantization_mode": "none"})
    int8_executor = executor_config_hash("upmem", {"strategy": strategy, "quantization_mode": "int8"})

    assert strategy["output_tile"] <= strategy["max_tensor_elements"]
    assert strategy["max_contracted_combinations"] <= strategy["max_tensor_elements"]
    assert cpu_bundle["contraction_plan_hash"] == upmem_bundle["contraction_plan_hash"] == graph.contraction_plan_hash
    assert cpu_executor != upmem_executor
    assert upmem_executor != int8_executor


def test_structure_hash_excludes_planner_identity_but_keeps_pairwise_structure() -> None:
    graph = _graph()
    changed_summary = replace(graph.path_summary, planner_id="test.other", planner_engine="test")
    changed = with_execution_identity(replace(graph, path_summary=changed_summary, contraction_plan_hash=""))

    assert contraction_path_structure_hash(changed) == contraction_path_structure_hash(graph)
    assert changed.contraction_plan_hash != graph.contraction_plan_hash


def test_planner_config_hash_changes_executor_identity_but_not_structure_hash() -> None:
    first = _graph(planner_run_label="first")
    second = _graph(planner_run_label="second")

    assert first.path == second.path
    assert first.path_summary.options["planner_config_hash"] != second.path_summary.options["planner_config_hash"]
    assert first.contraction_plan_hash != second.contraction_plan_hash
    assert build_execution_bundle(first, case_id="ghz_4q", suite_id="test")["planner"] != build_execution_bundle(
        second, case_id="ghz_4q", suite_id="test"
    )["planner"]
    assert contraction_path_structure_hash(first) == contraction_path_structure_hash(second)
