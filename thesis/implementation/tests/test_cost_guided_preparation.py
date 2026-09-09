from copy import deepcopy
import json

import pytest

from tests.test_upmem_cost_guided_search import search, _network
from quantum_bench.upmem.path_heuristic import path_id


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    study, _, _ = search.load_study()
    study = deepcopy(study)
    study["executor"]["topologies"] = [study["executor"]["topologies"][0]]
    workload = {"workload_id": "fixture", "instances": [
        {"instance_id": "train", "family": "family", "split": "training"},
        {"instance_id": "test", "family": "family", "split": "test"},
    ]}
    path = [[0, 1], [0, 1], [0, 1]]
    facts = {"model_id": "upmem_launch_cost_v1", "H": 1, "P": 1, "N": 1,
             "launches": [{"M": [1], "W": [1]}]}
    greedy = {"path": path, "path_id": path_id(path, circuit_id="train"), "tree_flops": 10.0,
              "facts": facts, "circuit_id": "train", "family": "family",
              "split": "training", "topology_id": "1dpu_t8"}
    normalization = {"scales": {k: 1.0 for k in ("H", "P", "N", "M", "W")},
                     "greedy_cells": {"train/1dpu_t8": greedy}}
    monkeypatch.setattr(search, "research_binding", lambda _: {"source_sha": "a" * 40})
    monkeypatch.setattr(search, "prepare_normalization", lambda *_: deepcopy(normalization))
    directory = tmp_path / "study"
    search.initialize_preparation(directory, study, workload)
    monkeypatch.setattr(search, "_cell_inputs", lambda *_: (
        workload["instances"][0], study["executor"]["topologies"][0], _network(),
    ))

    def engine(network, **kwargs):
        rows = []
        common_seeds = {k: kwargs[k] for k in ("master_seed", "workload_id", "cell_id", "stage")}
        sampler_seed = search._seed_from_identity(**common_seeds, seed_domain="optuna_tpe_sampler", proposal_ordinal=None)
        score = 10.0 if kwargs["objective_id"] == "cotengra_tree_flops_v1" else 1.0
        for index in range(128):
            row = {"proposal_index": index, "tell_order": index + 1, "path": path,
                   "path_id": greedy["path_id"], "tree_flops": 10.0, "score": score,
                   "told_objective": score, "facts": facts, "status": "eligible",
                   "rejection_reason": None, "duplicate": index > 0, "trial_number": index,
                   "params": {"costmod": 1.0, "temperature": 0.1}, "sampler_seed": sampler_seed,
                   "proposal_seed": search._seed_from_identity(**common_seeds, seed_domain="cotengra_random_greedy_proposal", proposal_ordinal=index),
                   **{k: kwargs[k] for k in ("stage", "objective_id", "profile_id")}}
            rows.append(row)
            kwargs["record_callback"](row)
        return {"schema": "upmem_cost_guided_search_trace_v1", "completed": True, "proposal_count": 128,
                "startup_trials": 16, "master_seed": kwargs["master_seed"], "sampler_seed": sampler_seed,
                "generator": {"optimizer": "cotengra.RandomGreedyOptimizer", "max_repeats": 1,
                              "accel": False, "parallel": False, "simplify": True},
                "seed_derivation": "sha256_sorted_compact_utf8_json_first_four_bytes_big_endian",
                "best_path_id": greedy["path_id"],
                **{k: kwargs[k] for k in ("cell_id", "circuit_id", "workload_id", "stage", "objective_id", "profile_id")},
                "trace_context": {"caller_binding": kwargs["trace_context"],
                                  **{k: kwargs[k] for k in ("stage", "objective_id", "profile_id")}}, "trace": rows}
    monkeypatch.setattr(search, "run_cost_guided_search", engine)
    return directory, study, workload


def test_initial_trace_is_durable_deduplicated_and_cannot_rerun(prepared):
    directory, study, workload = prepared
    result = search.initial_cell_search(directory, "train/1dpu_t8", study, workload)
    assert result["proposal_count"] == 128
    folder = directory / "initial" / search.record_hash("train/1dpu_t8")
    assert len((folder / "search_trace.jsonl").read_text().splitlines()) == 128
    with pytest.raises(FileExistsError):
        search.initial_cell_search(directory, "train/1dpu_t8", study, workload)
    budget = {"development_cells": 1, "initial_paths_per_cell": 4, "stages": {"initial": 16}}
    manifest = search.freeze_initial_round(directory, study, workload, budget)
    assert manifest["expected_attempts"] == 4
    assert manifest["physical_admission"] == "not_performed"
    assert set(manifest["cells"]["train/1dpu_t8"]["selection"]["roles"]) == {"G", "F", "R"}
    with pytest.raises(FileExistsError):
        search.freeze_initial_round(directory, study, workload, budget)
    with pytest.raises(ValueError, match="already frozen"):
        search.initial_cell_search(directory, "train/1dpu_t8", study, workload)


def test_failed_trace_preserves_prefix_and_cannot_refill(prepared, monkeypatch):
    directory, study, workload = prepared
    def failed(network, **kwargs):
        kwargs["record_callback"]({"proposal_index": 0, "status": "aborted"})
        raise RuntimeError("demonstrated failure")
    monkeypatch.setattr(search, "run_cost_guided_search", failed)
    with pytest.raises(RuntimeError, match="demonstrated failure"):
        search.initial_cell_search(directory, "train/1dpu_t8", study, workload)
    folder = directory / "initial" / search.record_hash("train/1dpu_t8")
    assert (folder / "failed.json").exists()
    assert not (folder / "completed.json").exists()
    assert len((folder / "search_trace.jsonl").read_text().splitlines()) == 1
    with pytest.raises(FileExistsError):
        search.initial_cell_search(directory, "train/1dpu_t8", study, workload)
    with pytest.raises(ValueError, match="Failed cell"):
        search.freeze_initial_round(directory, study, workload, {})


def test_preparation_rejects_source_drift_and_evaluation_cells(prepared, monkeypatch):
    directory, study, workload = prepared
    with pytest.raises(ValueError, match="development"):
        search.initial_cell_search(directory, "test/1dpu_t8", study, workload)
    monkeypatch.setattr(search, "research_binding", lambda _: {"source_sha": "b" * 40})
    with pytest.raises(ValueError, match="binding changed"):
        search.initial_cell_search(directory, "train/1dpu_t8", study, workload)


def test_freeze_requires_all_declared_traces_and_identical_incremental_records(prepared):
    directory, study, workload = prepared
    budget = {"development_cells": 1, "initial_paths_per_cell": 4, "stages": {"initial": 16}}
    with pytest.raises(FileNotFoundError):
        search.freeze_initial_round(directory, study, workload, budget)
    search.initial_cell_search(directory, "train/1dpu_t8", study, workload)
    folder = directory / "initial" / search.record_hash("train/1dpu_t8")
    with (folder / "search_trace.jsonl").open("a") as stream:
        stream.write(json.dumps({"extra": True}) + "\n")
    with pytest.raises(ValueError, match="Incremental trace"):
        search.freeze_initial_round(directory, study, workload, budget)
    assert not (directory / "initial_round.json").exists()


def test_exclusive_json_writer_does_not_overwrite(tmp_path):
    path = tmp_path / "record.json"
    search._write_new_json(path, {"first": True})
    with pytest.raises(FileExistsError):
        search._write_new_json(path, {"second": True})
    assert json.loads(path.read_text()) == {"first": True}


def test_duplicate_path_with_conflicting_facts_is_rejected():
    row = {"path_id": "p", "facts": {"H": 1}, "tree_flops": 1, "status": "eligible"}
    changed = deepcopy(row)
    changed["facts"]["H"] = 2
    with pytest.raises(ValueError, match="Conflicting deterministic facts"):
        search.eligible_candidate_pool({"trace": [row]}, changed)


@pytest.mark.parametrize("change", ["seed", "startup", "inner_repeats", "objective", "infeasible_tell", "duplicate", "nan_score"])
def test_freeze_rejects_corrupt_engine_provenance_even_when_jsonl_matches(prepared, change):
    directory, study, workload = prepared
    search.initial_cell_search(directory, "train/1dpu_t8", study, workload)
    folder = directory / "initial" / search.record_hash("train/1dpu_t8")
    path = folder / "completed.json"
    trace = json.loads(path.read_text())
    if change == "seed":
        trace["trace"][0]["proposal_seed"] += 1
    elif change == "startup":
        trace["startup_trials"] = 15
    elif change == "inner_repeats":
        trace["generator"]["max_repeats"] = 2
    elif change == "objective":
        trace["trace"][0]["objective_id"] = "another_objective"
    elif change == "infeasible_tell":
        trace["trace"][0].update(status="known_infeasible", rejection_reason="fixture", score=None, told_objective=0)
    elif change == "duplicate":
        trace["trace"][0]["duplicate"] = True
    else:
        trace["trace"][0]["score"] = float("nan")
    path.write_text(json.dumps(trace))
    (folder / "search_trace.jsonl").write_text("".join(json.dumps(row) + "\n" for row in trace["trace"]))
    with pytest.raises(ValueError):
        search.freeze_initial_round(directory, study, workload, {"development_cells": 1, "initial_paths_per_cell": 4, "stages": {"initial": 16}})
    assert not (directory / "initial_round.json").exists()
