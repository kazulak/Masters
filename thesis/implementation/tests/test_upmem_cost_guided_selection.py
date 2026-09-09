from copy import deepcopy

import pytest

from tests.test_upmem_cost_guided_search import search, _network
from quantum_bench.planning import plan_opt_einsum
from quantum_bench.upmem.path_heuristic import LaunchCostScales


SCALES = LaunchCostScales(1, 1, 1, 1, 1)
WEIGHTS = (10, 0, 0, 0, 0)
NORMALIZATION = {"scales": SCALES.as_mapping()}
PROFILE = {"model_id": "upmem_launch_cost_v1", "integer_weights": list(WEIGHTS),
           "normalization_hash": search.record_hash(NORMALIZATION)}


def candidate(identifier, cost, flops):
    return {"path_id": identifier, "tree_flops": flops, "status": "eligible",
            "facts": {"model_id": "upmem_launch_cost_v1", "H": cost, "P": 0, "N": 0, "launches": []}}


def trace(objective, *records):
    records = deepcopy(records)
    for row in records:
        row["score"] = row["tree_flops"] if objective == "cotengra_tree_flops_v1" else row["facts"]["H"]
        row["told_objective"] = row["score"]
    return {"completed": True, "proposal_count": 128,
            "objective_id": objective, "workload_id": "w", "cell_id": "c/t", "circuit_id": "c",
            "stage": "test", "master_seed": 1, "sampler_seed": 2, "profile_id": search.record_hash(PROFILE),
            "trace_context": {"stage": "test", "objective_id": objective, "profile_id": search.record_hash(PROFILE),
                              "caller_binding": {"normalization_hash": search.record_hash(NORMALIZATION)}},
            "trace": [records[i % len(records)] for i in range(128)]}


def test_initial_mandatory_roles_and_farthest_first_are_deterministic():
    values = {row["path_id"]: row for row in (
        candidate("g", 10, 10), candidate("f", 8, 1), candidate("r", 1, 3),
        candidate("far", 100, 200), candidate("middle", 12, 4),
    )}
    result = search.choose_round_paths(values, "g", scales=SCALES, weights=WEIGHTS, initial_cap=4)
    assert result["path_ids"] == ["f", "far", "g", "r"]
    assert {k: result["roles"][k] for k in ("G", "F", "R")} == {"G": "g", "F": "f", "R": "r"}
    assert result == search.choose_round_paths(dict(reversed(tuple(values.items()))), "g", scales=SCALES, weights=WEIGHTS, initial_cap=4)


def test_feedback_only_selects_unmeasured_paths_and_keeps_fresh_greedy():
    values = {row["path_id"]: row for row in (
        candidate("g", 10, 10), candidate("old", 1, 1), candidate("new", 2, 4), candidate("far", 50, 60),
    )}
    result = search.choose_round_paths(values, "g", scales=SCALES, weights=WEIGHTS, previously_measured={"g", "old"})
    assert result["path_ids"] == ["far", "g", "new"]
    assert result["roles"]["new_best"] == "new"
    skipped = search.choose_round_paths(values, "g", scales=SCALES, weights=WEIGHTS, previously_measured=set(values))
    assert skipped["path_ids"] == []
    assert skipped["reason"] == "no_new_eligible_candidate"


def test_u_never_receives_the_better_reranked_f_pool_candidate():
    g = candidate("g", 10, 10)
    f = trace("cotengra_tree_flops_v1", candidate("f", 8, 1), candidate("r", 1, 3))
    u = trace("upmem_launch_cost_v1", candidate("u", 3, 4))
    assert search.choose_evaluation_paths(g, f, u, profile=PROFILE, normalization=NORMALIZATION) == {"G": "g", "F": "f", "R": "r", "U": "u"}
    drift = deepcopy(u)
    drift["profile_id"] = "changed"
    with pytest.raises(ValueError, match="binding mismatch"):
        search.choose_evaluation_paths(g, f, drift, profile=PROFILE, normalization=NORMALIZATION)
    drift = deepcopy(u)
    drift["trace_context"]["caller_binding"]["normalization_hash"] = "different"
    with pytest.raises(ValueError, match="binding mismatch"):
        search.choose_evaluation_paths(g, f, drift, profile=PROFILE, normalization=NORMALIZATION)
    profile_drift = {**PROFILE, "integer_weights": [0, 10, 0, 0, 0]}
    with pytest.raises(ValueError, match="supplied profile"):
        search.choose_evaluation_paths(g, f, u, profile=profile_drift, normalization=NORMALIZATION)
    score_drift = deepcopy(u)
    score_drift["trace"][0]["score"] = 100
    with pytest.raises(ValueError, match="frozen objective"):
        search.choose_evaluation_paths(g, f, score_drift, profile=PROFILE, normalization=NORMALIZATION)


def test_greedy_flops_use_same_cotengra_scale_as_search():
    from cotengra import ContractionTree
    network = _network()
    path, _ = plan_opt_einsum(network)
    expected = ContractionTree.from_path(
        [t.labels for t in network.tensors], network.output_labels, search._size_dict(network), path=path,
    ).total_flops()
    assert search.conventional_tree_flops(network, path) == expected
