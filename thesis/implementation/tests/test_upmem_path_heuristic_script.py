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
    assert [item["source_seed"] for item in candidates] == [None, 20260903]
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
        "candidate_set_sha256": "d" * 64,
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
                "total_wall_s",
            ),
        )
        writer.writeheader()
        writer.writerow(
            {
                "split": "training", "attempt_type": "measurement",
                "cell_id": "train:1dpu_t8", "candidate_path_id": greedy,
                "total_wall_s": 10.0,
            }
        )
        writer.writerow(
            {
                "split": "training", "attempt_type": "measurement",
                "cell_id": "train:1dpu_t8", "candidate_path_id": candidate,
                "total_wall_s": 5.0,
            }
        )
        writer.writerow(
            {
                "split": "test", "attempt_type": "measurement",
                "cell_id": "train:1dpu_t8", "candidate_path_id": greedy,
                "total_wall_s": 0.01,
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
