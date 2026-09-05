from __future__ import annotations

from copy import deepcopy
import importlib.util
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from quantum_bench.circuits import builtin_circuit
from quantum_bench.lowering import build_contraction_dag, contraction_dag_hash, lower_tensor_network
from quantum_bench.model import make_simulation_job
from quantum_bench.planning import plan_opt_einsum
from quantum_bench.upmem.plan import UpmemTopology, UpmemWorkUnit


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/characterize_upmem_execution.py"
SPEC = importlib.util.spec_from_file_location("execution_census", SCRIPT)
census = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(census)


def cell(policy=census.POLICIES[0], dpus=1):
    network, _ = lower_tensor_network(make_simulation_job(builtin_circuit("bell_2q", {})))
    path, _ = plan_opt_einsum(network, optimize="greedy")
    return {
        "cell_id": "fixture", "circuit_id": "bell_fixture", "circuit": {"name": "bell_2q", "parameters": {}},
        "candidate_path_id": "fixture", "path": [list(p) for p in path], "path_roles": ["greedy"],
        "logical_plan_id": contraction_dag_hash(build_contraction_dag(network, path)),
        "numeric_policy": policy, "topology": {"dpu_count": dpus, "rank_count": 1, "tasklets_per_dpu": 8},
    }


def test_selection_deduplicates_roles_and_breaks_ties_by_id():
    def candidate(key, greedy, flops, peak):
        return {"candidate_path_id": key, "is_greedy": greedy, "logical_plan_id": key,
                "path": [[0, 1]], "conventional_features": {"flops": flops, "peak_intermediate_elements": peak}}
    values = [candidate("z", False, 1, 1), candidate("b", True, 2, 1), candidate("a", False, 1, 1)]
    selected = census.select_paths({"candidates": values})
    assert [(p["candidate_path_id"], p["path_roles"]) for p in selected] == [
        ("a", ["minimum_flops", "minimum_peak"]), ("b", ["greedy"])]
    assert census.select_paths({"candidates": list(reversed(values))}) == selected
    with pytest.raises(ValueError, match="exactly one"):
        census.select_paths({"candidates": [values[0]]})


def test_frozen_manifest_is_deterministic_development_only():
    first, hashes = census.frozen_cells()
    second, _ = census.frozen_cells()
    assert first == second
    assert len(first) <= 48
    assert len({row["cell_id"] for row in first}) == len(first)
    assert {r["circuit_id"] for r in first} == set(census.CIRCUITS)
    assert sum("greedy" in r["path_roles"] for r in first) == 16
    assert set(hashes) == {"pilot", "generalization"}
    for circuit_id in census.CIRCUITS:
        by_topology = {}
        for row in first:
            if row["circuit_id"] == circuit_id:
                by_topology.setdefault((row["numeric_policy"], row["topology"]["dpu_count"]), set()).add(row["candidate_path_id"])
        assert len(by_topology) == 4
        assert all(paths == next(iter(by_topology.values())) for paths in by_topology.values())


@pytest.mark.parametrize("policy", census.POLICIES)
@pytest.mark.parametrize("dpus", [1, 4])
def test_both_policy_plans_keep_partial_waves_and_correct_counters(policy, dpus):
    result = census.characterize_cell(cell(policy, dpus))
    assert result["status"] == "eligible"
    assert result == census.characterize_cell(cell(policy, dpus))
    assert result["totals"]["lane_envelope_submissions"] == 4 * len(result["operations"])
    assert result["totals"]["dpu_launch_count"] == 4 * result["totals"]["wave_count"]
    assert result["totals"]["descriptor_count"] == dpus * result["totals"]["request_count"]
    assert result["totals"]["output_file_count"] == 4 * result["totals"]["work_unit_count"]
    assert all(op["measured_timing"] is None for op in result["operations"])
    if dpus == 4:
        assert result["totals"]["idle_dpu_slots"] > 0


def test_explicit_limits_and_identity_fail_closed():
    fixture = cell()
    missing = {**fixture, "logical_plan_id": None,
               "retained_infeasibility_reasons": ["semantic_identity_expansion_exceeds_preregistered_bound"]}
    rejected = census.characterize_cell(missing)
    assert rejected["rejection_reasons"] == ["retained_candidate_has_no_logical_identity"]
    assert rejected["retained_infeasibility_reasons"] == missing["retained_infeasibility_reasons"]
    assert census.characterize_cell(fixture, memory_limit=0)["rejection_reasons"] == ["host_memory_limit"]
    assert census.characterize_cell(fixture, work_limit=0)["rejection_reasons"] == ["planned_work_unit_limit"]
    corrupt = deepcopy(fixture)
    corrupt["logical_plan_id"] = "0" * 64
    with pytest.raises(ValueError, match="identity mismatch"):
        census.characterize_cell(corrupt)


def test_odd_geometry_split_k_counts_launches_not_envelopes():
    node = SimpleNamespace(node_id="contract", left=SimpleNamespace(labels=(0, 1), shape=(3, 5)),
                           right=SimpleNamespace(labels=(1, 2), shape=(5, 7)),
                           output=SimpleNamespace(labels=(0, 2), shape=(3, 7)))
    units = tuple(UpmemWorkUnit(node_id="contract", stable_tile_id=str(i), wave=i,
                               logical_rank=0, logical_dpu=0, batch_start=0, batch_size=1,
                               m_start=0, m_size=3, n_start=0, n_size=7, k_start=start, k_size=k,
                               estimated_input_bytes=4 * 10 * k, estimated_output_bytes=84,
                               aligned_mram_bytes=256, estimated_arithmetic_work=21 * k)
                  for i, (start, k) in enumerate(((0, 3), (3, 2))))
    facts = census.operation_facts(node, units, census.POLICIES[0],
                                   UpmemTopology(dpu_count=4, tasklets_per_dpu=8))
    assert (facts["b"], facts["m"], facts["k"], facts["n"]) == (1, 3, 5, 7)
    assert facts["output_tile_count"] == 1
    assert facts["k_chunk_count"] == 2
    assert facts["lane_envelope_submissions"] == 4
    assert facts["dpu_launch_count"] == 8
    assert facts["idle_dpu_slots"] == 6


def test_timeout_is_retained_not_replaced(monkeypatch):
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], 0.01)
    monkeypatch.setattr(census.subprocess, "run", timeout)
    result = census.isolated_cell(cell())
    assert result["status"] == "rejected"
    assert result["rejection_reasons"] == ["lowering_timeout"]


def test_nonempty_output_is_not_overwritten(tmp_path):
    (tmp_path / "existing").write_text("retained")
    with pytest.raises(ValueError, match="empty"):
        census.write_census(tmp_path)
    assert (tmp_path / "existing").read_text() == "retained"
