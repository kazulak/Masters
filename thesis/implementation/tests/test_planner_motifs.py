from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from quantum_bench.bench.planner_compare import compare_planners
from quantum_bench.circuits import load_circuit
from quantum_bench.tn import contraction_path_structure_hash, execute_task_sequence_np_einsum, plan_task_graph_with_config
from quantum_bench.tn.planner_motifs import SUPPORTED_PLANNER_MOTIFS, build_planner_motif_workload
from quantum_bench.tn.upmem_path_cost import model_upmem_network_path_cost
from quantum_bench.tn.upmem_planner import PlannerInfeasibleError


ROOT = Path(__file__).resolve().parents[1]


def _case(name: str) -> dict:
    return {
        "case_id": f"planner_motif_{name}",
        "circuit": {"kind": "planner_motif", "name": name},
        "metadata": {
            "workload_type": "synthetic_planner_motif",
            "execution_scope": "model_only",
            "not_real_quantum_circuit": True,
        },
    }


@pytest.mark.parametrize("name", sorted(SUPPORTED_PLANNER_MOTIFS))
def test_planner_motifs_are_real_valued_modeled_only_networks(name: str) -> None:
    workload = build_planner_motif_workload(_case(name))

    assert workload.circuit.n_qubits == 0
    assert workload.metadata["not_real_quantum_circuit"] is True
    assert workload.metadata["execution_scope"] == "model_only"
    assert workload.network.spec.output_labels == ()
    assert len(workload.network.tensors) == workload.metadata["network_tensor_count"]
    assert all(np.isrealobj(tensor.array) for tensor in workload.network.tensors)

    graph = plan_task_graph_with_config(
        workload.network,
        {"engine": "custom_upmem", "weight_profile": "balanced_literature_informed"},
    )
    assert len(graph.tasks) == len(workload.network.tensors) - 1
    assert graph.path_summary.planner_engine == "custom_upmem"


def test_planner_motif_profiles_can_select_different_paths() -> None:
    workload = build_planner_motif_workload(_case("flop_memory_tradeoff"))
    compute = plan_task_graph_with_config(
        workload.network,
        {"engine": "custom_upmem", "weight_profile": "compute_oriented"},
    )
    wram = plan_task_graph_with_config(
        workload.network,
        {"engine": "custom_upmem", "weight_profile": "wram_constrained"},
    )

    assert compute.path != wram.path
    assert compute.contraction_plan_hash != wram.contraction_plan_hash
    assert contraction_path_structure_hash(compute) != contraction_path_structure_hash(wram)


def test_custom_upmem_path_contracts_to_the_same_result_as_standard_path() -> None:
    workload = build_planner_motif_workload(_case("grid"))
    standard = plan_task_graph_with_config(
        workload.network,
        {"engine": "opt_einsum", "optimize": "greedy"},
    )
    custom = plan_task_graph_with_config(
        workload.network,
        {"engine": "custom_upmem", "weight_profile": "balanced_literature_informed"},
    )

    expected, _ = execute_task_sequence_np_einsum(standard, workload.network)
    actual, _ = execute_task_sequence_np_einsum(custom, workload.network)

    assert np.allclose(actual, expected, rtol=1.0e-12, atol=1.0e-12)


def test_custom_upmem_rejects_complex_dtype_even_when_imaginary_values_are_zero() -> None:
    workload = build_planner_motif_workload(_case("chain"))
    complex_network = replace(
        workload.network,
        tensors=tuple(
            replace(tensor, array=np.asarray(tensor.array, dtype=np.complex64))
            for tensor in workload.network.tensors
        ),
    )

    with pytest.raises(PlannerInfeasibleError, match="complex tensor inputs") as exc_info:
        plan_task_graph_with_config(
            complex_network,
            {"engine": "custom_upmem", "weight_profile": "balanced_literature_informed"},
        )

    assert exc_info.value.rejection_reasons == ("complex_generic_loop_not_implemented",)


def test_shared_upmem_cost_model_rejects_complex_network_for_all_planner_engines() -> None:
    workload = build_planner_motif_workload(_case("chain"))
    graph = plan_task_graph_with_config(workload.network, {"engine": "opt_einsum", "optimize": "greedy"})
    complex_network = replace(
        workload.network,
        tensors=tuple(
            replace(tensor, array=np.asarray(tensor.array, dtype=np.complex64))
            for tensor in workload.network.tensors
        ),
    )

    components = model_upmem_network_path_cost(complex_network, graph.tasks)

    assert components.feasibility is False
    assert components.rejection_reasons == ("complex_generic_loop_not_implemented",)


def test_planner_motif_is_rejected_by_normal_circuit_loader(tmp_path) -> None:
    with pytest.raises(ValueError, match="modeled-planning-only"):
        load_circuit(_case("chain"), tmp_path)


def test_planner_objective_motif_suite_records_modeled_only_metadata(tmp_path) -> None:
    run_dir = compare_planners(ROOT / "configs/suites/diagnostics/planner_objective_motifs.yml", tmp_path)
    payload = json.loads((run_dir / "planner_comparison.json").read_text(encoding="utf-8"))

    assert {row["planner_motif"] for row in payload["rows"]} == SUPPORTED_PLANNER_MOTIFS
    assert all(row["workload_kind"] == "planner_motif" for row in payload["rows"])
    assert all(row["not_real_quantum_circuit"] is True for row in payload["rows"])
    assert all(row["candidate_status"] == "completed" for row in payload["rows"])
    assert any(row["planner_id"].startswith("custom_upmem.") for row in payload["rows"])
