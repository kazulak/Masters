from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
import sys

import pytest


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


def test_deterministic_admission_precedes_process_isolation(monkeypatch) -> None:
    config = script.load_config()
    definition = {"name": "bell_2q", "parameters": {}}
    circuit = script.builtin_circuit(definition["name"], definition["parameters"])
    network, _ = script.lower_tensor_network(script.make_simulation_job(circuit))
    path, provenance = script.plan_opt_einsum(network, optimize="greedy")
    item = {
        "candidate_path_id": script.path_id(path, circuit_id="fixture"),
        "path": path,
        "source_kind": "opt_einsum_greedy",
        "source_seed": None,
        "planner_config_hash": provenance["planner_config_hash"],
        "is_greedy": True,
    }
    monkeypatch.setattr(
        script,
        "_estimated_work_unit_count",
        lambda dag: config["candidate_generation"]["maximum_planned_work_units"] + 1,
    )
    monkeypatch.setattr(
        script.multiprocessing,
        "get_context",
        lambda method: (_ for _ in ()).throw(AssertionError("worker was started")),
    )
    record, _, candidate = script._serialized_candidate_with_admission(
        circuit_id="fixture",
        split="training",
        definition=definition,
        item=item,
        config=config,
    )
    assert candidate is None
    assert record["topologies"][0]["infeasibility_reason"] == (
        "estimated_work_unit_count_exceeds_preregistered_bound"
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


def _calibration_fixture(tmp_path: Path) -> tuple[Path, Path, Path, dict, tuple[dict, ...], tuple[dict, ...]]:
    candidate_id = "a" * 64
    logical_plan_id = "b" * 64
    physical_plan_id = "c" * 64
    problem = "d" * 64
    tensor_structure = "e" * 64
    candidate_source = "f" * 40
    physical_source = "1" * 40
    experiment_id = "2" * 64
    run_id = "run-fixture"
    dataset = {
        "schema_version": script.SCHEMA_VERSION,
        "source_sha": candidate_source,
        "circuits": [{
            "circuit_id": "fixture",
            "split": "training",
            "problem_id": problem,
            "tensor_network_structure_id": tensor_structure,
            "candidates": [{
                "candidate_path_id": candidate_id,
                "source_kind": "fixture",
                "is_greedy": True,
                "logical_plan_id": logical_plan_id,
                "conventional_features": {
                    "flops": 1.0,
                    "macs": 1.0,
                    "peak_intermediate_elements": 1.0,
                    "peak_intermediate_bytes": 16.0,
                    "total_intermediate_writes": 1.0,
                    "maximum_intermediate_rank": 1,
                    "contraction_count": 1,
                },
                "topologies": [{
                    "topology_id": "1dpu_t8",
                    "feasible": True,
                    "physical_plan_id": physical_plan_id,
                    "topology": {
                        "dpu_count": 1,
                        "rank_count": 1,
                        "tasklets_per_dpu": 8,
                    },
                    "resource_admission": {
                        "collection_resource_admission_passed": True,
                    },
                }],
            }],
        }],
    }
    calibration = {
        "schema_version": "upmem_path_calibration_candidate_set_v1",
        "source_sha": candidate_source,
        "candidate_set_sha256": script._sha256_bytes(script._canonical_bytes(dataset)),
        "timing_used_for_selection": False,
        "cells": [{
            "cell_id": "fixture:1dpu_t8",
            "circuit_id": "fixture",
            "topology_id": "1dpu_t8",
            "greedy_path_id": candidate_id,
            "candidate_path_ids": [candidate_id],
        }],
    }
    manifest = {
        "status": "completed",
        "source_worktree_dirty": False,
        "source_commit": physical_source,
        "experiment_id": experiment_id,
        "run_id": run_id,
        "configuration": {
            "experiment": {
                "experiment_id": experiment_id,
                "collection": {
                    "claim_policy": "diagnostic_v1",
                    "warmup_blocks": 1,
                    "measurement_blocks": 3,
                    "session_policy": "fresh_session_per_attempt_v1",
                },
                "matrix": [{
                    "case_id": "fixture",
                    "plan_id": f"path_{candidate_id}",
                    "route_ids": ["1dpu_t8"],
                }],
            },
            "environment": {
                "host": "fixture-host",
                "requested_rank_paths": ["/dev/dpu_rank1"],
            },
        },
    }
    base_backend = {
        "target_observed": "physical_hardware",
        "physical_target_verified": True,
        "hardware_kernel_executed": True,
        "simulator_kernel_executed": False,
        "cpu_fallback_used": False,
        "collection_resource_admission_passed": True,
        "execution_resource_admission_passed": True,
        "startup_resource_admission_passed": True,
        "requested_dpus": 1,
        "allocated_dpus": 1,
        "active_dpus": 1,
        "tasklets_per_dpu": 8,
        "rank_count": 1,
        "request_transport": script.CALIBRATION_TRANSPORT,
        "arithmetic_weighted_tasklet_utilization": 1.0,
        "arithmetic_weighted_dpu_slot_utilization": 1.0,
        "dominant_work_wave_utilization": 1.0,
        "total_wave_count": 1,
        "fully_populated_wave_count": 1,
        "active_dpu_ids": [[0, 0]],
        "active_rank_indices": [0],
        "operation_facts": [],
    }
    terminal = {
        "target_observed": "physical_hardware",
        "physical_target_verified": True,
        "hardware_kernel_executed": True,
        "simulator_kernel_executed": False,
        "cpu_fallback_used": False,
        "allocation_verified": True,
        "hardware_allocation_verified": True,
        "binary_identity_verified": True,
        "native_identity_verified": True,
        "hardware_release_verified": True,
        "startup_resource_admission_passed": True,
        "requested_dpu_count": 1,
        "allocated_dpu_count": 1,
        "observed_dpu_count": 1,
        "observed_tasklets_per_dpu": 8,
        "startup_requested_dpu_count": 1,
        "startup_allocated_dpu_count": 1,
        "startup_requested_tasklets_per_dpu": 8,
    }
    validation = {
        "accuracy_qualified": True,
        "full_precision_threshold_applicable": True,
        "full_precision_passed": True,
        "policy_reference_applicable": True,
        "policy_reference_passed": True,
        "max_abs_error": 0.0,
        "relative_l2_error": 0.0,
        "norm_drift": 0.0,
        "phase_aligned_max_abs_error": 0.0,
    }
    samples = []
    sessions = []
    for block, attempt_kind in ((0, "warmup"), (1, "measurement"), (2, "measurement"), (3, "measurement")):
        session_id = f"session-{block}"
        samples.append({
            "experiment_id": experiment_id,
            "run_id": run_id,
            "case_id": "fixture",
            "plan_id": f"path_{candidate_id}",
            "route_id": "1dpu_t8",
            "block_id": block,
            "attempt_kind": attempt_kind,
            "status": "success",
            "sample_id": f"sample-{block}",
            "sample_index": block,
            "order_index": block,
            "session_instance_id": session_id,
            "observed_affinity": [0],
            "output_sha256": "3" * 64,
            "identities": {
                "problem_id": problem,
                "tensor_network_structure_id": tensor_structure,
                "logical_plan_id": logical_plan_id,
                "physical_plan_id": physical_plan_id,
                "executable_id": "4" * 64,
                "validation_policy_id": "5" * 64,
            },
            "validation": validation,
            "measurement": {
                "scope_id": "steady_execution_v1",
                "total_wall_s": 10.0 + block,
                "kernel_s": 2.0,
                "h2d_s": 0.1,
                "d2h_s": 0.1,
                "h2d_bytes": 100,
                "d2h_bytes": 200,
                "preparation_s": 0.2,
            },
            "backend_facts": base_backend,
        })
        sessions.append({
            "experiment_id": experiment_id,
            "run_id": run_id,
            "case_id": "fixture",
            "plan_id": f"path_{candidate_id}",
            "route_id": "1dpu_t8",
            "status": "success",
            "session_instance_id": session_id,
            "open_s": 0.5,
            "session_close_s": 0.25,
            "release_attempted": True,
            "release_succeeded": True,
            "release_verified": True,
            "terminal_backend_facts": terminal,
        })
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "manifest.json").write_text("manifest\n", encoding="utf-8")
    (raw_dir / "samples.jsonl").write_text("samples\n", encoding="utf-8")
    (raw_dir / "sessions.jsonl").write_text("sessions\n", encoding="utf-8")
    candidate_path = tmp_path / "candidate_paths.json"
    calibration_path = tmp_path / "calibration_candidate_set.json"
    candidate_path.write_bytes(script._canonical_bytes(dataset))
    calibration_path.write_bytes(script._canonical_bytes(calibration))
    return raw_dir, candidate_path, calibration_path, manifest, tuple(samples), tuple(sessions)


def test_extract_calibration_emits_raw_rows_and_separates_source_commits(
    tmp_path: Path, monkeypatch
) -> None:
    raw_dir, candidate_path, calibration_path, manifest, samples, sessions = _calibration_fixture(tmp_path)
    monkeypatch.setattr(script, "load_artifacts", lambda path: (manifest, samples, sessions))
    output_dir = tmp_path / "calibration"
    result = script.extract_calibration(raw_dir, candidate_path, calibration_path, output_dir)
    assert result["sample_count"] == 4
    assert result["session_count"] == 4
    assert result["candidate_generation_source_sha"] == "f" * 40
    assert result["physical_execution_source_sha"] == "1" * 40
    assert len(result["observations"]) == 4
    assert {row["block"] for row in result["observations"]} == {0, 1, 2, 3}
    assert (output_dir / "path_runtime_calibration.csv").exists()
    table = list(csv.DictReader((output_dir / "path_runtime_calibration.csv").open(encoding="utf-8")))
    assert len(table) == 4
    assert table[0]["source_sha"] == "f" * 40
    assert table[0]["candidate_generation_source_sha"] == "f" * 40
    assert table[0]["physical_execution_source_sha"] == "1" * 40
    assert table[0]["timing_scope"] == "steady_execution_v1"
    assert table[0]["fallback"] == "false"
    assert json.loads(table[0]["backend_facts_json"])["request_transport"] == "packed_operation_v1"
    emitted = json.loads((output_dir / "path_runtime_calibration.json").read_text(encoding="utf-8"))
    assert emitted["observations"][0]["raw_sample"]["sample_id"] == "sample-0"
    assert emitted["observations"][0]["raw_session"]["session_instance_id"] == "session-0"


def test_extract_calibration_rejects_incomplete_block_set(tmp_path: Path, monkeypatch) -> None:
    raw_dir, candidate_path, calibration_path, manifest, samples, sessions = _calibration_fixture(tmp_path)
    monkeypatch.setattr(script, "load_artifacts", lambda path: (manifest, samples[:-1], sessions[:-1]))
    with pytest.raises(ValueError, match="canonical evidence count"):
        script.extract_calibration(raw_dir, candidate_path, calibration_path, tmp_path / "out")


def test_extract_calibration_cli_alias_invokes_strict_extractor(tmp_path: Path, monkeypatch, capsys) -> None:
    raw_dir, candidate_path, calibration_path, manifest, samples, sessions = _calibration_fixture(tmp_path)
    monkeypatch.setattr(script, "load_artifacts", lambda path: (manifest, samples, sessions))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "upmem_path_heuristic.py", "extract", "--raw-dir", str(raw_dir),
            "--candidate-paths", str(candidate_path), "--calibration-set",
            str(calibration_path), "--output-dir", str(tmp_path / "cli-out"),
        ],
    )
    script.main()
    assert json.loads(capsys.readouterr().out)["observation_count"] == 4
