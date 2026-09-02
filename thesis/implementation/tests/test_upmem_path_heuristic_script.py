from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
import sys


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "upmem_path_heuristic.py"
SPEC = importlib.util.spec_from_file_location("upmem_path_heuristic_script", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
script = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = script
SPEC.loader.exec_module(script)


def test_candidate_generation_is_seeded_deduplicated_and_greedy_is_retained(monkeypatch) -> None:
    config = script.load_config()
    config["candidate_generation"]["one_trial_searches"] = 4
    monkeypatch.setattr(
        script,
        "plan_opt_einsum",
        lambda network, optimize: (
            ((0, 1),),
            {"planner_config_hash": "g" * 64},
        ),
    )

    seen = []

    def fake_cotengra(network, *, objective, methods, max_repeats, seed):
        seen.append((objective, methods, max_repeats, seed))
        path = ((0, 1),) if seed % 2 == 0 else ((1, 0),)
        return path, {"planner_config_hash": f"{seed:064x}"}

    monkeypatch.setattr(script, "plan_cotengra", fake_cotengra)
    candidates, timings = script._candidate_paths(object(), "circuit", config)
    assert [item["source_seed"] for item in candidates] == [None]
    assert candidates[0]["is_greedy"] is True
    assert seen == [
        ("flops", "greedy", 1, seed)
        for seed in range(20260902, 20260906)
    ]
    assert timings["candidate_generation_s"] >= 0.0


def _candidate_record(path_id: str, host_bytes: float, *, greedy: bool) -> dict[str, object]:
    features = {
        "B_host_dpu": host_bytes,
        "B_mram_wram": 100.0,
        "I_dpu": 100.0,
        "N_sync": 100.0,
        "E_num": 0.0,
        "P_wram": 1.0,
    }
    return {
        "candidate_path_id": path_id,
        "source_kind": "fixture",
        "is_greedy": greedy,
        "conventional_features": {
            "flops": 10.0,
            "macs": 5.0,
            "peak_intermediate_elements": 2.0,
            "peak_intermediate_bytes": 32.0,
            "total_intermediate_writes": 2.0,
            "maximum_intermediate_rank": 2,
            "contraction_count": 1,
        },
        "topologies": [
            {
                "topology_id": "1dpu_t8",
                "feasible": True,
                "physical_plan_id": f"physical-{path_id}",
                "features": features,
            }
        ],
    }


def test_offline_fit_uses_only_training_measurements_and_writes_every_evaluation(tmp_path: Path) -> None:
    greedy = "a" * 64
    candidate = "b" * 64
    dataset = {
        "source_sha": "c" * 40,
        "circuits": [
            {
                "circuit_id": "train",
                "candidates": [
                    _candidate_record(greedy, 100.0, greedy=True),
                    _candidate_record(candidate, 50.0, greedy=False),
                ],
            }
        ],
    }
    calibration = {
        "source_sha": dataset["source_sha"],
        "candidate_set_sha256": script._sha256_bytes(script._canonical_bytes(dataset)),
        "cells": [
            {
                "cell_id": "train:1dpu_t8",
                "circuit_id": "train",
                "topology_id": "1dpu_t8",
                "greedy_path_id": greedy,
                "candidate_path_ids": [greedy, candidate],
            }
        ],
    }
    candidates_path = tmp_path / "candidate_paths.json"
    calibration_path = tmp_path / "calibration.json"
    runtimes_path = tmp_path / "runtime.csv"
    candidates_path.write_text(json.dumps(dataset), encoding="utf-8")
    calibration_path.write_text(json.dumps(calibration), encoding="utf-8")
    with runtimes_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "split", "attempt_type", "cell_id", "candidate_path_id",
                "total_wall_s", "source_sha", "timing_scope", "status",
                "validation", "fallback", "physical_plan_id", "block",
            ),
        )
        writer.writeheader()
        for block in (1, 2, 3):
            for candidate_id, runtime in ((greedy, 10.0), (candidate, 5.0)):
                writer.writerow(
                    {
                        "split": "training", "attempt_type": "measurement",
                        "cell_id": "train:1dpu_t8", "candidate_path_id": candidate_id,
                        "total_wall_s": runtime, "source_sha": dataset["source_sha"],
                        "timing_scope": "steady_execution_v1", "status": "success",
                        "validation": "passed", "fallback": "false",
                        "physical_plan_id": f"physical-{candidate_id}", "block": block,
                    }
                )
    output = tmp_path / "fit"
    result = script.fit(
        candidates_path, calibration_path, runtimes_path, output,
        samples=4, seed=7,
    )
    assert result.geometric_mean_speedup == 2.0
    profile = json.loads((output / "physical_speedup_fit_v1.json").read_text(encoding="utf-8"))
    assert profile["selected_path_ids"] == {"train:1dpu_t8": candidate}
    with (output / "weight_search_candidates.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == result.evaluated_weight_vectors


def test_physical_lowering_timeout_is_an_explicit_infeasible_candidate() -> None:
    config = script.load_config()
    item = {
        "candidate_path_id": "e" * 64,
        "path": ((0, 1),),
        "source_kind": "cotengra_one_trial",
        "source_seed": 20260902,
        "planner_config_hash": "f" * 64,
        "is_greedy": False,
    }
    record, rows, candidate = script._infeasible_candidate_record(
        circuit_id="fixture",
        split="training",
        item=item,
        config=config,
        reason="physical_lowering_timeout_60s",
    )
    assert candidate is None
    assert record["conventional_features"] is None
    assert len(rows) == 2
    assert all(row["feasible"] is False for row in rows)
    assert all(
        topology["infeasibility_reason"] == "physical_lowering_timeout_60s"
        for topology in record["topologies"]
    )


def test_frozen_profile_selects_validation_paths_without_timing(tmp_path: Path) -> None:
    greedy = "a" * 64
    candidate = "b" * 64
    records = [
        _candidate_record(greedy, 100.0, greedy=True),
        _candidate_record(candidate, 50.0, greedy=False),
    ]
    for record in records:
        second = dict(record["topologies"][0])
        second["topology_id"] = "4dpu_t8"
        second["physical_plan_id"] = f"physical-4d-{record['candidate_path_id']}"
        record["topologies"].append(second)
    dataset = {
        "source_sha": "c" * 40,
        "circuits": [{
            "circuit_id": "held-out",
            "split": "validation",
            "candidates": records,
        }],
    }
    dataset_path = tmp_path / "candidate_paths.json"
    dataset_path.write_bytes(script._canonical_bytes(dataset))
    profile = {
        "source_sha": dataset["source_sha"],
        "candidate_set_sha256": script._sha256_bytes(script._canonical_bytes(dataset)),
        "weights": {
            "B_host_dpu": 1.0,
            "B_mram_wram": 0.0,
            "I_dpu": 0.0,
            "N_sync": 0.0,
            "E_num": 0.0,
            "P_wram": 0.0,
        },
        "feature_model": {
            "mode": "six_term",
            "active_features": [
                "B_host_dpu", "B_mram_wram", "I_dpu", "N_sync", "E_num", "P_wram"
            ],
            "zero_range_features": [],
            "correlated_pairs": [],
            "matrix_rank": 6,
            "rank_tolerance": 1.0e-12,
            "reason": "fixture",
        },
    }
    profile_path = tmp_path / "profile.json"
    profile_path.write_bytes(script._canonical_bytes(profile))
    output = tmp_path / "validation.json"
    result = script.evaluate_frozen_profile(
        dataset_path, profile_path, output, split="validation"
    )
    assert result["timing_used_for_selection"] is False
    assert {row["upmem_selected_path_id"] for row in result["selections"]} == {
        candidate
    }
    assert len(result["selections"]) == 2
