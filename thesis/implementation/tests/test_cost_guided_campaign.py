from copy import deepcopy

import pytest

from tests.test_cost_guided_preparation import prepared  # noqa: F401
from tests.test_upmem_cost_guided_search import search
from quantum_bench.upmem.path_heuristic import path_id


def synthetic_observations(manifest, binding, study):
    identity = {"source_sha": binding["source_sha"], "execution_source": study["executor"]["source"],
                "policy_id": study["executor"]["numeric_policy"], "timing_scope": "steady_execution_v1",
                "round_manifest_hash": search.record_hash(manifest), "experiment_id": manifest["stage"]}
    rows = []
    for cell_id, cell in manifest["cells"].items():
        for path in cell["candidates"]:
            for attempt, blocks in (("warmup", manifest["warmup_blocks"]), ("measurement", manifest["measurement_blocks"])):
                for block in blocks:
                    rows.append({"cell_id": cell_id, "path_id": path, "round_id": manifest["stage"],
                                 "block": block, "attempt_type": attempt, "identity": identity,
                                 "split": cell["split"], "status": "success", "validation": "pass", "fallback": False,
                                 "full_precision_passed": True, "policy_reference_passed": True,
                                 "execution_resource_admission_passed": True, "startup_resource_admission_passed": True,
                                 "session_open_s": 1.0, "total_wall_s": 10.0 if path == cell["selection"]["roles"]["G"] else 5.0,
                                 "session_close_s": 1.0})
    return rows


@pytest.fixture
def campaign(request, monkeypatch):
    directory, study, workload = request.getfixturevalue("prepared")
    calls = []
    def verify(archives, manifest, workload, binding, study):
        calls.append(manifest["stage"])
        return {"rows": synthetic_observations(manifest, binding, study),
                "archives": [{"path": str(path), "sha256": "0" * 64} for path in archives],
                "physical_stage_elapsed_s": 1.0}
    monkeypatch.setattr(search, "verify_round_archives", verify)
    budget = {"development_cells": 1, "evaluation_cells": 1, "initial_paths_per_cell": 4,
              "stages": {"initial": 16, "feedback_1": 12, "feedback_2": 12, "evaluation": 24}}
    return directory, study, workload, budget, calls


def initial(campaign):
    directory, study, workload, budget, _ = campaign
    search.initial_cell_search(directory, "train/1dpu_t8", study, workload)
    search.freeze_initial_round(directory, study, workload, budget)
    search.accept_round(directory, "initial", [directory / "copy1", directory / "copy2"], study, workload)
    return search.fit_accepted_rounds(directory, "initial", study, workload)


def test_no_fit_without_accepted_raw_round_and_no_evaluation_leakage(campaign):
    directory, study, workload, _, _ = campaign
    with pytest.raises(FileNotFoundError):
        search.fit_accepted_rounds(directory, "initial", study, workload)
    with pytest.raises(ValueError, match="forbidden"):
        search.fit_accepted_rounds(directory, "evaluation", study, workload)
    with pytest.raises(FileNotFoundError):
        search.feedback_cell_search(directory, "feedback_1", "train/1dpu_t8", study, workload)


def test_prelaunch_budget_requires_verified_predecessors_and_no_prior_invocation(campaign, monkeypatch):
    directory, study, workload, _, _ = campaign
    manifest = {"expected_attempts": 4, "cells": {"one": {"candidates": {"g": {}}}}}
    with pytest.raises(FileNotFoundError):
        search.remaining_execution_budget(directory, "feedback_1", manifest, study, workload)
    budget = search.remaining_execution_budget(directory, "initial", manifest, study, workload)
    assert budget["cumulative_attempts"] == 4
    assert budget["remaining_physical_time_s"] == 86400
    for state in ("running", "accepted", "failed"):
        marker = directory / f"initial.{state}.json"
        marker.write_text("{}")
        with pytest.raises(ValueError, match="cannot execute again"):
            search.remaining_execution_budget(directory, "initial", manifest, study, workload)
        marker.unlink()
    with pytest.raises(ValueError, match="stage budget"):
        search.remaining_execution_budget(directory, "initial", {**manifest, "expected_attempts": 8}, study, workload)
    monkeypatch.setattr(search, "_accepted_round", lambda *_: (
        {"expected_attempts": 192}, {"physical_stage_elapsed_s": 86400},
    ))
    with pytest.raises(ValueError, match="time budget exhausted"):
        search.remaining_execution_budget(directory, "feedback_1", manifest, study, workload)
    small = deepcopy(study)
    small["campaign"]["effective_attempt_cap"] = 192
    with pytest.raises(ValueError, match="attempt budget exceeded"):
        search.remaining_execution_budget(directory, "feedback_1", manifest, small, workload)


def test_empty_feedback_rounds_stop_after_two_without_refill(campaign):
    directory, study, workload, budget, calls = campaign
    profile = initial(campaign)
    assert len(search._read_json(directory / "initial_fit/grid.json")) == 1001
    for stage in ("feedback_1", "feedback_2"):
        search.feedback_cell_search(directory, stage, "train/1dpu_t8", study, workload)
        manifest = search.freeze_feedback_round(directory, stage, study, workload, budget)
        assert manifest["expected_attempts"] == 0
        assert manifest["skipped_cells"]["train/1dpu_t8"]["reason"] == "no_new_eligible_candidate"
        search.accept_round(directory, stage, [], study, workload)
        fitted = search.fit_accepted_rounds(directory, stage, study, workload)
        assert fitted["integer_weights"] == profile["integer_weights"]
    assert set(calls) == {"initial"}
    frozen = search.freeze_pretest(directory, study, workload)
    assert frozen["evaluation_cells"] == ["test/1dpu_t8"]
    with pytest.raises(ValueError, match="after pretest"):
        search.feedback_cell_search(directory, "feedback_1", "train/1dpu_t8", study, workload)
    with pytest.raises(ValueError, match="forbidden"):
        search.fit_accepted_rounds(directory, "feedback_2", study, workload)
    with pytest.raises(FileExistsError):
        search.freeze_pretest(directory, study, workload)
    with pytest.raises(ValueError, match="forbidden"):
        search._stage_prefix("feedback_3")


def test_feedback_uses_new_paths_and_verified_observations(campaign, monkeypatch):
    directory, study, workload, budget, _ = campaign
    initial(campaign)
    engine = search.run_cost_guided_search
    new_path = [[0, 2], [0, 1], [0, 1]]
    new_id = path_id(new_path, circuit_id="train")
    def changed(network, **kwargs):
        retain = kwargs["record_callback"]
        def transform(row):
            row.update(path=new_path, path_id=new_id, score=0.0, told_objective=0.0,
                       facts={"model_id": "upmem_launch_cost_v1", "H": 0, "P": 0, "N": 0, "launches": []})
            retain(row)
        return engine(network, **{**kwargs, "record_callback": transform}) | {"best_path_id": new_id}
    monkeypatch.setattr(search, "run_cost_guided_search", changed)
    search.feedback_cell_search(directory, "feedback_1", "train/1dpu_t8", study, workload)
    manifest = search.freeze_feedback_round(directory, "feedback_1", study, workload, budget)
    assert manifest["expected_attempts"] == 8
    assert manifest["cells"]["train/1dpu_t8"]["selection"]["roles"]["new_best"] == new_id
    search.accept_round(directory, "feedback_1", [directory / "copy3", directory / "copy4"], study, workload)
    fitted = search.fit_accepted_rounds(directory, "feedback_1", study, workload)
    assert len(fitted["accepted_round_hashes"]) == 2
    result = search._read_json(directory / "feedback_1_fit/fit.json")
    assert result["cells"]["train/1dpu_t8"]["path_id"] == new_id
    with pytest.raises(ValueError, match="cannot be replaced"):
        search.accept_round(directory, "feedback_1", [], study, workload)


def test_changed_profile_and_accepted_data_are_rejected(campaign):
    directory, study, workload, _, _ = campaign
    initial(campaign)
    path = directory / "initial_fit/profile.json"
    profile = search._read_json(path)
    changed = deepcopy(profile)
    changed["integer_weights"] = [10, 0, 0, 0, 0]
    import json
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="binding changed"):
        search.feedback_cell_search(directory, "feedback_1", "train/1dpu_t8", study, workload)
    path.write_text(json.dumps(profile))
    accepted_path = directory / "initial.accepted.json"
    accepted = search._read_json(accepted_path)
    accepted["rows"][0]["total_wall_s"] = 123
    accepted_path.write_text(json.dumps(accepted))
    with pytest.raises(ValueError, match="raw evidence"):
        search._accepted_round(directory, "initial", study, workload)


def test_final_evaluation_requires_both_traces_and_freezes_all_roles(campaign, monkeypatch):
    from tests.test_upmem_cost_guided_search import _network
    directory, study, workload, budget, _ = campaign
    initial(campaign)
    for stage in ("feedback_1", "feedback_2"):
        search.feedback_cell_search(directory, stage, "train/1dpu_t8", study, workload)
        search.freeze_feedback_round(directory, stage, study, workload, budget)
        search.accept_round(directory, stage, [], study, workload)
        search.fit_accepted_rounds(directory, stage, study, workload)
    with pytest.raises(FileNotFoundError):
        search.evaluation_cell_search(directory, "test/1dpu_t8", "upmem_launch_cost_v1", study, workload)
    search.freeze_pretest(directory, study, workload)
    engine = search.run_cost_guided_search
    def test_engine(network, **kwargs):
        retain = kwargs["record_callback"]
        def transform(row):
            row["path_id"] = path_id(row["path"], circuit_id="test")
            retain(row)
        result = engine(network, **{**kwargs, "record_callback": transform})
        result["best_path_id"] = result["trace"][0]["path_id"]
        return result
    monkeypatch.setattr(search, "run_cost_guided_search", test_engine)
    monkeypatch.setattr(search, "_cell_inputs", lambda *_: (workload["instances"][1], study["executor"]["topologies"][0], _network()))
    monkeypatch.setattr(search, "conventional_tree_flops", lambda *_: 10.0)
    facts = search._read_json(directory / "normalization.json")["greedy_cells"]["train/1dpu_t8"]["facts"]
    monkeypatch.setattr(search, "lower_candidate", lambda *_: search.GuidedEvaluation(0, facts))
    search.evaluation_cell_search(directory, "test/1dpu_t8", "cotengra_tree_flops_v1", study, workload)
    with pytest.raises(FileNotFoundError):
        search.freeze_evaluation_round(directory, study, workload, budget)
    search.evaluation_cell_search(directory, "test/1dpu_t8", "upmem_launch_cost_v1", study, workload)
    manifest = search.freeze_evaluation_round(directory, study, workload, budget)
    cell = manifest["cells"]["test/1dpu_t8"]
    assert set(cell["selection"]["roles"]) == {"G", "F", "R", "U"}
    assert manifest["expected_attempts"] == 6 * len(set(cell["selection"]["roles"].values()))
    assert manifest["measurement_blocks"] == [1, 2, 3, 4, 5]
    with pytest.raises(ValueError, match="already frozen"):
        search.evaluation_cell_search(directory, "test/1dpu_t8", "upmem_launch_cost_v1", study, workload)
    with pytest.raises(ValueError, match="forbidden"):
        search.fit_accepted_rounds(directory, "feedback_2", study, workload)
