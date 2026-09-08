from __future__ import annotations

import csv
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

import pytest
import yaml


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "upmem_path_heuristic.py"
SPEC = importlib.util.spec_from_file_location("upmem_path_heuristic_script", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
script = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = script
SPEC.loader.exec_module(script)


def _wave_config() -> dict[str, object]:
    config = deepcopy(script.load_config())
    config["execution_profile"] = script.EXECUTION_PROFILE
    config["execution_contract"] = dict(script.WAVE_EXECUTION_CONTRACT)
    config["score_id"] = script.WAVE_COST_MODEL_ID
    config["topologies"] = [config["topologies"][0]]
    return config


def test_execution_contract_is_explicit_and_legacy_inputs_are_not_migrated() -> None:
    legacy = script.load_config()
    assert script.execution_contract(legacy) is None
    wave = _wave_config()
    assert script.execution_contract(wave) == script.WAVE_EXECUTION_CONTRACT

    wrong_numeric = deepcopy(wave)
    wrong_numeric["execution_contract"]["numeric_policy"] = "complex_int8_shared_scale_v1"
    with pytest.raises(ValueError, match="numeric_policy"):
        script.execution_contract(wrong_numeric)

    wrong_topology = deepcopy(wave)
    wrong_topology["topologies"][0]["rank_count"] = 2
    with pytest.raises(ValueError, match="one rank"):
        script.execution_contract(wrong_topology)

    legacy_profile_mix = deepcopy(wave)
    legacy_profile_mix["score_id"] = script.COST_MODEL_ID
    with pytest.raises(ValueError, match="score_id"):
        script.execution_contract(legacy_profile_mix)

    frozen_profile_mix = deepcopy(wave)
    frozen_profile_mix["calibration"]["frozen_v1_profile"] = {}
    with pytest.raises(ValueError, match="legacy frozen"):
        script.execution_contract(frozen_profile_mix)

    missing_profile = {
        "execution_contract": dict(script.WAVE_EXECUTION_CONTRACT),
        "score_id": script.WAVE_COST_MODEL_ID,
    }
    with pytest.raises(ValueError, match="explicit execution_profile"):
        script.execution_contract(missing_profile)


def test_wave_candidate_persists_actual_plan_and_declared_facts(monkeypatch) -> None:
    config = _wave_config()
    definition = {"name": "hs", "parameters": {"n_qubits": 8, "depth": 1}}
    circuit = script.builtin_circuit(definition["name"], definition["parameters"])
    network, _ = script.lower_tensor_network(script.make_simulation_job(circuit))
    path, provenance = script.plan_opt_einsum(network, optimize="greedy")
    item = {
        "candidate_path_id": script.path_id(path, circuit_id="wave-fixture"),
        "path": path,
        "source_kind": "opt_einsum_greedy",
        "source_seed": None,
        "planner_config_hash": provenance["planner_config_hash"],
        "is_greedy": True,
    }
    monkeypatch.setattr(script, "_host_memory_estimate", lambda inputs, dag: 10**18)
    record, rows, candidate = script._serialized_candidate_with_admission(
        circuit_id="wave-fixture",
        split="training",
        definition=definition,
        item=item,
        config=config,
    )

    assert candidate is not None and candidate.feasible_for("1dpu_t8")
    assert record["execution_profile"] == script.EXECUTION_PROFILE
    assert record["execution_contract"] == script.WAVE_EXECUTION_CONTRACT
    assert record["score_id"] == script.WAVE_COST_MODEL_ID
    topology_record = record["topologies"][0]
    resources = topology_record["topology"]
    assert resources == {"dpu_count": 1, "rank_count": 1, "tasklets_per_dpu": 8}
    assert topology_record["execution_contract"] == script.WAVE_EXECUTION_CONTRACT
    assert topology_record["logical_plan_id"] == record["logical_plan_id"]
    assert topology_record["feasible"] is True
    assert topology_record["memory_admission"]["passed"] is True

    dag = script.build_contraction_dag(network, path)
    topology = script.UpmemTopology(**resources)
    serial = script.plan_upmem(
        dag, numeric_policy=script.NUMERIC_POLICY, topology=topology
    )
    wave = script.plan_upmem(
        dag,
        numeric_policy=script.NUMERIC_POLICY,
        topology=topology,
        schedule_policy=script.WAVE_EXECUTION_CONTRACT["schedule_policy"],
    )
    assert record["logical_plan_id"] == script.contraction_dag_hash(dag)
    assert topology_record["physical_plan_id"] == script.physical_plan_id(wave)
    assert script.physical_plan_id(serial) != topology_record["physical_plan_id"]

    facts = topology_record["wave_facts"]
    assert facts["plan"]["physical_plan_id"] == topology_record["physical_plan_id"]
    assert facts["plan"]["logical_plan_id"] == record["logical_plan_id"]
    assert topology_record["features"] == facts["raw"]
    raw_mapping = facts["raw"]
    assert raw_mapping["E_num"] == 0.0
    assert raw_mapping["P_wram"] > 0.0
    assert json.dumps(record, sort_keys=True)
    assert set(rows[0]) == set(script.WAVE_FEATURE_COLUMNS)
    assert rows[0]["execution_profile"] == script.EXECUTION_PROFILE
    assert rows[0]["score_id"] == script.WAVE_COST_MODEL_ID
    assert rows[0]["I_dpu"] == raw_mapping["I_dpu"]
    assert rows[0]["wave_critical_real_mac_sum"] == facts["totals"][
        "wave_critical_real_mac_sum"
    ]
    assert rows[0]["wave_critical_mram_bytes"] == raw_mapping["B_mram_wram"]
    assert rows[0]["declared_executor_memory_estimate_bytes"] == facts[
        "host_buffers"
    ]["declared_executor_memory_estimate_bytes"]
    dataset = {
        "execution_profile": script.EXECUTION_PROFILE,
        "execution_contract": script.WAVE_EXECUTION_CONTRACT,
        "score_id": script.WAVE_COST_MODEL_ID,
        "circuits": [{"circuit_id": "wave-fixture", "candidates": [record]}],
    }
    assert script._validate_dataset_execution_contract(dataset) == (
        script.WAVE_EXECUTION_CONTRACT
    )


def test_wave_features_use_critical_work_without_aggregate_double_count() -> None:
    circuit = script.builtin_circuit("bell_2q", {})
    network, _ = script.lower_tensor_network(script.make_simulation_job(circuit))
    path, _ = script.plan_opt_einsum(network, optimize="greedy")
    dag = script.build_contraction_dag(network, path)
    topology = script.UpmemTopology(dpu_count=1, rank_count=1, tasklets_per_dpu=8)
    plan = script.plan_upmem(
        dag,
        numeric_policy=script.NUMERIC_POLICY,
        topology=topology,
        schedule_policy="static_dag_waves_v1",
    )
    facts = script.extract_wave_path_features(
        dag, plan, fuse_complex=True, geometry_policy="panel_only_v1"
    )
    raw = facts["raw"].as_mapping()
    totals = facts["totals"]
    assert raw["I_dpu"] == totals["wave_critical_real_mac_sum"]
    assert raw["I_dpu"] <= totals["real_mac_count"]
    assert raw["B_mram_wram"] == sum(
        max(
            slot["local_traffic"]["mram_aligned_transfer_bytes_estimate"]
            for slot in wave["slots"]
        )
        for wave in facts["waves"]
    )
    assert raw["N_sync"] == sum(facts["sync_components"].values())


def test_wave_build_binds_dataset_and_calibration_contract(monkeypatch) -> None:
    config = _wave_config()
    config["circuits"] = [{
        "circuit_id": "wave-fixture",
        "split": "training",
        "circuit": {
            "kind": "builtin",
            "name": "hs",
            "parameters": {"n_qubits": 8, "depth": 1},
        },
    }]
    config["candidate_generation"]["one_trial_searches"] = 0

    def candidate_paths(network, circuit_id, config, **kwargs):
        del config, kwargs
        path, provenance = script.plan_opt_einsum(network, optimize="greedy")
        return [
            {
                "candidate_path_id": script.path_id(path, circuit_id=circuit_id),
                "path": path,
                "source_kind": "opt_einsum_greedy",
                "source_seed": None,
                "planner_config_hash": provenance["planner_config_hash"],
                "is_greedy": True,
            }
        ], {"candidate_generation_s": 0.0, "cotengra_search_s": 0.0}

    monkeypatch.setattr(script, "_candidate_paths", candidate_paths)
    dataset, features, rankings, calibration, _timings = script.build_dataset(config)

    assert dataset["execution_profile"] == script.EXECUTION_PROFILE
    assert dataset["execution_contract"] == script.WAVE_EXECUTION_CONTRACT
    assert dataset["score_id"] == script.WAVE_COST_MODEL_ID
    assert calibration["execution_contract"] == script.WAVE_EXECUTION_CONTRACT
    assert calibration["score_id"] == script.WAVE_COST_MODEL_ID
    assert len(features) == 1
    assert set(features[0]) == set(script.WAVE_FEATURE_COLUMNS)
    assert rankings[0]["candidate_path_id"] == (
        dataset["circuits"][0]["candidates"][0]["candidate_path_id"]
    )
    assert calibration["cells"][0]["candidate_path_ids"] == [
        rankings[0]["candidate_path_id"]
    ]


def test_wave_calibration_role_dedup_does_not_refill_physical_choices() -> None:
    raw = script.RawFeatureVector(1.0, 1.0, 1.0, 1.0, 0.0, 1.0)
    candidates = tuple(
        script.PathCandidate.synthetic(
            path_id,
            raw,
            topology="1dpu_t8",
            flops=flops,
            peak_intermediate=peak,
            intermediate_writes=writes,
            is_greedy=path_id == "greedy",
        )
        for path_id, flops, peak, writes in (
            ("greedy", 1.0, 1.0, 1.0),
            ("flops", 2.0, 2.0, 2.0),
            ("peak", 3.0, 3.0, 3.0),
            ("writes", 4.0, 4.0, 4.0),
            ("diverse", 5.0, 5.0, 5.0),
            ("unused", 6.0, 6.0, 6.0),
        )
    )
    records = {
        item.path_id: {
            "topologies": [{
                "topology_id": "1dpu_t8",
                "physical_plan_id": "shared-physical",
            }]
        }
        for item in candidates[:-1]
    }
    records["unused"] = {
        "topologies": [{
            "topology_id": "1dpu_t8",
            "physical_plan_id": "unused-physical",
        }]
    }
    model = script.FeatureModelDecision(
        mode="six_term",
        active_features=(),
        zero_range_features=("B_host_dpu", "B_mram_wram", "I_dpu", "N_sync", "E_num", "P_wram"),
        correlated_pairs=(),
        matrix_rank=0,
        rank_tolerance=0.0,
        reason="fixture",
    )

    selected = script._wave_calibration_candidates(
        candidates,
        "1dpu_t8",
        limit=6,
        model=model,
        greedy_path_id="greedy",
        records=records,
    )

    assert [item.path_id for item in selected] == ["greedy"]
    _selected, roles = script._wave_calibration_candidates(
        candidates,
        "1dpu_t8",
        limit=6,
        model=model,
        greedy_path_id="greedy",
        records=records,
        return_roles=True,
    )
    assert roles == (
        {"role": "greedy", "candidate_path_id": "greedy"},
        {"role": "minimum_flops", "candidate_path_id": "greedy"},
        {"role": "minimum_peak_intermediate", "candidate_path_id": "greedy"},
        {"role": "minimum_writes", "candidate_path_id": "greedy"},
        {"role": "equal_wave_cost", "candidate_path_id": "diverse"},
        {"role": "feature_diverse", "candidate_path_id": "diverse"},
    )


@pytest.mark.parametrize("operation", ["fit", "evaluate"])
def test_wave_downstream_operations_reject_incomplete_inputs(
    tmp_path: Path, operation: str
) -> None:
    dataset_path = tmp_path / "wave.json"
    dataset_path.write_text(
        json.dumps(
            {
                "execution_profile": script.EXECUTION_PROFILE,
                "execution_contract": script.WAVE_EXECUTION_CONTRACT,
                "score_id": script.WAVE_COST_MODEL_ID,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        if operation == "fit":
            script.fit(
                dataset_path,
                tmp_path / "missing-calibration.json",
                tmp_path / "missing-runtime.csv",
                tmp_path / "fit",
                samples=1,
                seed=1,
            )
        elif operation == "extract":
            script.extract_calibration(
                tmp_path / "missing-raw",
                dataset_path,
                tmp_path / "missing-calibration.json",
                tmp_path / "extract",
            )
        else:
            script.evaluate_frozen_profile(
                dataset_path,
                tmp_path / "missing-profile.json",
                tmp_path / "evaluate.json",
                split="validation",
            )


def test_calibration_splits_default_to_training_and_accept_validation() -> None:
    assert script._calibration_splits({"calibration": {}}) == {"training"}
    assert script._calibration_splits(
        {"calibration": {"splits": ["training", "validation"]}}
    ) == {"training", "validation"}


@pytest.mark.parametrize("splits", [[], ["test"], ["training", "test"]])
def test_calibration_splits_reject_non_calibration_partitions(splits: list[str]) -> None:
    with pytest.raises(ValueError, match="calibration splits"):
        script._calibration_splits({"calibration": {"splits": splits}})


def test_collection_admission_reasons_are_derived_from_existing_facts() -> None:
    assert script._collection_admission_reasons(
        {
            "tasklet_row_sufficiency_passed": True,
            "dominant_work_wave_allocated_dpu_slots": 4,
            "dominant_work_wave_populated_dpu_slots": 2,
        }
    ) == ("dominant_work_wave_underfilled",)
    assert script._collection_admission_reasons(
        {
            "tasklet_row_sufficiency_passed": False,
            "dominant_work_wave_allocated_dpu_slots": 1,
            "dominant_work_wave_populated_dpu_slots": 1,
        }
    ) == ("tasklet_row_sufficiency",)


def test_generalization_calibration_includes_frozen_profile_selection() -> None:
    greedy_id = "a" * 64
    selected_id = "b" * 64
    candidates = tuple(
        script._candidate_from_record(record)
        for record in (
            _candidate_record(greedy_id, 100.0, greedy=True),
            _candidate_record(selected_id, 50.0, greedy=False),
        )
    )
    selected, roles = script._generalization_calibration_candidates(
        candidates,
        "1dpu_t8",
        limit=6,
        weights=script.WeightVector.from_values(
            {"B_host_dpu": 1.0}
        ),
        model=script.explicit_feature_model("six_term"),
        greedy_path_id=greedy_id,
    )

    assert {item.path_id for item in selected} == {greedy_id, selected_id}
    assert next(
        item["candidate_path_id"]
        for item in roles
        if item["role"] == "frozen_v1_selected"
    ) == selected_id


def test_candidate_pool_hashes_are_per_circuit_and_deterministic() -> None:
    dataset = {
        "source_sha": "a" * 40,
        "workload_manifest_sha256": "b" * 64,
        "circuits": [
            {
                "circuit_id": "fixture",
                "candidates": [
                    {"candidate_path_id": "c" * 64},
                    {"candidate_path_id": "d" * 64},
                ],
            }
        ],
    }

    first = script._candidate_pool_hashes(dataset)
    second = script._candidate_pool_hashes(dataset)
    assert first == second
    assert first["workload_manifest_sha256"] == "b" * 64
    assert first["circuits"][0]["candidate_count"] == 2
    assert len(first["circuits"][0]["candidate_pool_sha256"]) == 64


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
                "candidate_generation_source_sha", "physical_execution_source_sha",
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
                        "candidate_generation_source_sha": dataset["source_sha"],
                        "physical_execution_source_sha": "d" * 40,
                        "timing_scope": "steady_execution_v1", "status": "success",
                        "validation": "passed", "fallback": "false",
                        "physical_plan_id": f"physical-{candidate_id}", "block": block,
                    }
                )
    output = tmp_path / "fit"
    result = script.fit(
        candidates_path, calibration_path, runtimes_path, output,
        samples=4, seed=7, model_form="grouped",
    )
    assert result.geometric_mean_speedup == 2.0
    profile = json.loads((output / "physical_speedup_fit_v1.json").read_text(encoding="utf-8"))
    assert profile["selected_path_ids"] == {"train:1dpu_t8": candidate}
    assert profile["candidate_generation_source_sha"] == "c" * 40
    assert profile["physical_execution_source_sha"] == "d" * 40
    assert profile["requested_model_form"] == "grouped"
    assert profile["fit_splits"] == ["training"]
    assert profile["calibration_set_sha256"] == script._file_sha256(calibration_path)
    assert profile["runtime_table_sha256"] == script._file_sha256(runtimes_path)
    with (output / "weight_search_candidates.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert 1 <= len(rows) <= result.evaluated_weight_vectors
    assert sum(int(row["equivalent_weight_vector_count"]) for row in rows) == (
        result.evaluated_weight_vectors
    )
    assert profile["weight_search_candidate_rows"] == len(rows)


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


def _calibration_fixture(
    tmp_path: Path, *, split: str = "training"
) -> tuple[Path, Path, Path, dict, tuple[dict, ...], tuple[dict, ...]]:
    candidate_id = "a" * 64
    logical_plan_id = "b" * 64
    physical_plan_id = "c" * 64
    problem = "d" * 64
    tensor_structure = "e" * 64
    candidate_source = "f" * 40
    physical_source = "1" * 40
    experiment_id = "8" * 64
    run_id = "run-fixture"
    dataset = {
        "schema_version": script.SCHEMA_VERSION,
        "source_sha": candidate_source,
        "circuits": [{
            "circuit_id": "fixture",
            "split": split,
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
                        "tasklet_row_sufficiency_passed": True,
                        "dominant_work_wave_tasklet_row_sufficiency_passed": True,
                        "dominant_work_wave_allocated_dpu_slots": 1,
                        "dominant_work_wave_populated_dpu_slots": 1,
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
        "backend_id": "upmem_final_plan_v1",
        "execution_class": "upmem_v4_real_tile",
        "target_observed": "physical_hardware",
        "physical_target_verified": True,
        "hardware_kernel_executed": True,
        "simulator_kernel_executed": False,
        "cpu_fallback_used": False,
        "tasklet_row_sufficiency_passed": True,
        "dominant_work_wave_tasklet_row_sufficiency_passed": True,
        "dominant_work_wave_allocated_dpu_slots": 1,
        "dominant_work_wave_populated_dpu_slots": 1,
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
        "backend_id": "upmem_sdk_hardware_v4_tile_session",
        "execution_class": "physical_v4_output_tile",
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


def _wave_calibration_fixture(
    tmp_path: Path,
) -> tuple[Path, Path, Path, dict, tuple[dict, ...], tuple[dict, ...]]:
    raw_dir, candidate_path, calibration_path, manifest, base_samples, base_sessions = (
        _calibration_fixture(tmp_path)
    )
    dataset = json.loads(candidate_path.read_text(encoding="utf-8"))
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    dataset.update(
        {
            "execution_profile": script.EXECUTION_PROFILE,
            "execution_contract": dict(script.WAVE_EXECUTION_CONTRACT),
            "score_id": script.WAVE_COST_MODEL_ID,
            "preregistration_sha256": "5" * 64,
        }
    )
    candidate = dataset["circuits"][0]["candidates"][0]
    candidate.update(
        {
            "source_kind": "opt_einsum_greedy",
            "source_seed": None,
            "planner_config_hash": "6" * 64,
            "execution_profile": script.EXECUTION_PROFILE,
            "execution_contract": dict(script.WAVE_EXECUTION_CONTRACT),
            "score_id": script.WAVE_COST_MODEL_ID,
        }
    )
    raw = script.RawFeatureVector(1.0, 2.0, 3.0, 4.0, 0.0, 5.0)
    topology = candidate["topologies"][0]
    topology.update(
        {
            "execution_profile": script.EXECUTION_PROFILE,
            "execution_contract": dict(script.WAVE_EXECUTION_CONTRACT),
            "score_id": script.WAVE_COST_MODEL_ID,
            "memory_admission": {
                "declared_executor_memory_estimate_bytes": 128,
                "configured_budget_bytes": 256,
                "configured_reserve_bytes": 0,
                "required_bytes": 128,
                "passed": True,
            },
            "host_memory_estimate_bytes": 128,
            "wave_facts": {
                "cost_model_id": script.WAVE_COST_MODEL_ID,
                "plan": {
                    **script.WAVE_EXECUTION_CONTRACT,
                    "logical_plan_id": candidate["logical_plan_id"],
                    "physical_plan_id": topology["physical_plan_id"],
                    "dpu_count": 1,
                    "rank_count": 1,
                    "tasklets_per_dpu": 8,
                    "kernel_identity": script.WAVE_KERNEL_IMPLEMENTATION,
                },
                "raw": raw.as_mapping(),
            },
            "features": raw.as_mapping(),
        }
    )
    candidate["topologies"][0]["topology"] = {
        "dpu_count": 1,
        "rank_count": 1,
        "tasklets_per_dpu": 8,
    }
    calibration.update(
        {
            "execution_profile": script.EXECUTION_PROFILE,
            "execution_contract": dict(script.WAVE_EXECUTION_CONTRACT),
            "score_id": script.WAVE_COST_MODEL_ID,
        }
    )
    environment_id = "6" * 64
    validation_policy_id = "7" * 64
    manifest["source_commit"] = script.EXECUTION_SOURCE
    manifest["environment_id"] = environment_id
    manifest["validation_policy_id"] = validation_policy_id
    experiment = manifest["configuration"]["experiment"]
    experiment["routes"] = {
        "1dpu_t8": {
            "executor": "upmem_physical",
            "numeric_policy": script.NUMERIC_POLICY,
            "options": {
                "dpu_count": 1,
                "rank_count": 1,
                "tasklets_per_dpu": 8,
                "rank_paths": ["/dev/dpu_rank1"],
                "request_transport": "packed_wave_v1",
                "schedule_policy": "static_dag_waves_v1",
                "fuse_complex": True,
                "geometry_policy": "panel_only_v1",
                "dpu_binary": "../native/dpu_wave_v5_t8",
                "host_binary": "../native/host_wave_v4_t8",
                "initialization_binary": "../native/init_t8",
            },
        }
    }
    manifest["configuration"]["identity_bindings"] = [{
        "case_id": "fixture",
        "plan_id": f"path_{candidate['candidate_path_id']}",
        "route_id": "1dpu_t8",
        "problem_id": "d" * 64,
        "tensor_network_structure_id": "e" * 64,
        "logical_plan_id": candidate["logical_plan_id"],
        "physical_plan_id": topology["physical_plan_id"],
        "executable_id": "4" * 64,
        "environment_id": environment_id,
        "validation_policy_id": validation_policy_id,
    }]
    samples = []
    sessions = []
    for base_sample, base_session in zip(base_samples, base_sessions, strict=True):
        sample = deepcopy(base_sample)
        session = deepcopy(base_session)
        # The canonical scheduler resets sample_index within each block.
        sample["sample_index"] = 0
        sample["order_index"] = 0
        sample["identities"] = {
            **sample["identities"],
            "environment_id": environment_id,
            "validation_policy_id": validation_policy_id,
        }
        sample["numeric_facts"] = {
            "numeric_policy": script.NUMERIC_POLICY,
            "operand_records": [],
            "operations": [],
            "raw_lane_records": [],
            "saturation_real": 0,
            "saturation_imag": 0,
        }
        sample["failure"] = None
        sample["backend_facts"] = {
            **sample["backend_facts"],
            "logical_plan_id": candidate["logical_plan_id"],
            "physical_plan_id": topology["physical_plan_id"],
            "output_hash": sample["output_sha256"],
            "request_transport": "packed_wave_v1",
            "schedule_policy": "static_dag_waves_v1",
            "complex_launch_policy": script.WAVE_COMPLEX_LAUNCH_POLICY,
            "geometry_kernel_policy": "panel_only_v1",
            "kernel_implementation_id": script.WAVE_KERNEL_IMPLEMENTATION,
            "physical_plan_consumed": True,
            "test_double_execution": False,
            "rank_response_timing_scope": "cohort_counters_on_first_node_v1",
            "tasklet_row_sufficiency_passed": True,
            "dominant_work_wave_tasklet_row_sufficiency_passed": True,
            "execution_active_dpu_count": 1,
            "execution_active_rank_count": 1,
            "execution_resource_admission_reasons": [],
            "startup_resource_admission_reasons": [],
            "active_ranks": [0],
        }
        session["session_protocol_id"] = script.WAVE_SESSION_PROTOCOL
        session["failure"] = None
        session["terminal_backend_facts"] = {
            **session["terminal_backend_facts"],
            "request_transport": "packed_wave_v1",
            "schedule_policy": "static_dag_waves_v1",
            "complex_launch_policy": script.WAVE_COMPLEX_LAUNCH_POLICY,
            "geometry_kernel_policy": "panel_only_v1",
            "kernel_implementation_id": script.WAVE_KERNEL_IMPLEMENTATION,
            "native_kernel_executed": True,
            "simulator_target_verified": False,
            "ready_verified": True,
            "hardware_release_attempted": True,
            "hardware_release_confirmed": True,
            "hardware_release_succeeded": True,
            "physical_profile": "prepared_wave_v1",
            "hardware_profile": "prepared_wave_v1",
            "observed_rank_count": 1,
            "tasklets_per_dpu": 8,
            "test_double_execution": False,
            "backend_family": "upmem_sdk",
            "kernel_provider": script.WAVE_KERNEL_IMPLEMENTATION,
            "kernel_strategy": script.WAVE_KERNEL_IMPLEMENTATION,
            "kernel_identity": script.WAVE_KERNEL_IMPLEMENTATION,
            "dispatch": "bulk_set_synchronous_v1",
            "dispatch_mode": "bulk_set_synchronous_v1",
            "active_dpu_ids": [[0, 0]],
            "active_rank_indices": [0],
            "startup_resource_admission_reasons": [],
            "dpu_binary_path": "../native/dpu_wave_v5_t8",
            "host_binary_path": "../native/host_wave_v4_t8",
            "initialization_binary_path": "../native/init_t8",
            "dpu_binary_sha256": "8" * 64,
            "host_binary_sha256": "9" * 64,
            "initialization_binary_sha256": "a" * 64,
            "source_root": "../implementation",
            "strategy_config_hash": "b" * 64,
            "strategy_identity": {
                "request_transport": "packed_wave_v1",
                "complex_launch_policy": script.WAVE_COMPLEX_LAUNCH_POLICY,
                "geometry_kernel_policy": "panel_only_v1",
                "kernel_identity": script.WAVE_KERNEL_IMPLEMENTATION,
            },
        }
        samples.append(sample)
        sessions.append(session)
    preregistration_dir = raw_dir.parent / "preregistration"
    preregistration_dir.mkdir()
    (preregistration_dir / "binary_sha256.json").write_bytes(
        script._canonical_bytes(
            {
                "../native/dpu_wave_v5_t8": "8" * 64,
                "../native/host_wave_v4_t8": "9" * 64,
                "../native/init_t8": "a" * 64,
            }
        )
    )
    physical_config = yaml.safe_load(
        (
            script.ROOT / "configs" / "tn_benchmark_upmem_path_calibration_v1.yml"
        ).read_text(encoding="utf-8")
    )
    physical_config["experiment_id"] = "wave-calibration-fixture"
    physical_config["cases"] = {
        "fixture": {
            "circuit": {
                "kind": "builtin",
                "name": "hs",
                "parameters": {"n_qubits": 8, "depth": 1},
                "path": None,
            }
        }
    }
    physical_config["plans"] = {
        f"path_{candidate['candidate_path_id']}": {
            "planner": {"engine": "opt_einsum", "mode": "greedy"},
            "slicing": None,
        }
    }
    physical_config["routes"] = {
        "1dpu_t8": {
            "executor": "upmem_physical",
            "numeric_policy": script.NUMERIC_POLICY,
            "options": {
                "dpu_count": 1,
                "rank_count": 1,
                "tasklets_per_dpu": 8,
                "session_root": "../runs/upmem_sessions/wave_fixture",
                "rank_paths": ["/dev/dpu_rank1"],
                "request_transport": "packed_wave_v1",
                "schedule_policy": "static_dag_waves_v1",
                "fuse_complex": True,
                "geometry_policy": "panel_only_v1",
                "dpu_binary": "../native/dpu_wave_v5_t8",
                "host_binary": "../native/host_wave_v4_t8",
                "initialization_binary": "../native/init_t8",
            },
        }
    }
    physical_config["matrix"] = [{
        "case_id": "fixture",
        "plan_id": f"path_{candidate['candidate_path_id']}",
        "route_ids": ["1dpu_t8"],
    }]
    physical_path = preregistration_dir / "physical.yml"
    physical_path.write_text(
        yaml.safe_dump(physical_config, sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )
    normalized = json.loads(
        script.canonical_json(script.load_experiment_config(physical_path))
    )
    manifest["configuration"]["experiment"] = normalized
    manifest["experiment_id"] = normalized["experiment_id"]
    normalized_options = normalized["routes"]["1dpu_t8"]["options"]
    binary_hashes = {
        normalized_options[field]: digest
        for field, digest in (
            ("dpu_binary", "8" * 64),
            ("host_binary", "9" * 64),
            ("initialization_binary", "a" * 64),
        )
    }
    (preregistration_dir / "binary_sha256.json").write_bytes(
        script._canonical_bytes(binary_hashes)
    )
    for sample, session in zip(samples, sessions, strict=True):
        sample["experiment_id"] = normalized["experiment_id"]
        session["experiment_id"] = normalized["experiment_id"]
        for field in ("dpu_binary", "host_binary", "initialization_binary"):
            terminal = session["terminal_backend_facts"]
            terminal[f"{field}_path"] = normalized_options[field]
    configuration_sha = script._file_sha256(physical_path)
    normalized_sha = script._sha256_bytes(script._canonical_bytes(normalized))
    calibration["candidate_set_sha256"] = script._sha256_bytes(
        script._canonical_bytes(dataset)
    )
    candidate_path.write_bytes(script._canonical_bytes(dataset))
    calibration_path.write_bytes(script._canonical_bytes(calibration))
    sidecar = {
        "schema_version": "upmem_path_experiment_provenance_v1",
        "source_sha": dataset["source_sha"],
        "candidate_set_sha256": calibration["candidate_set_sha256"],
        "preregistration_sha256": dataset["preregistration_sha256"],
        "mode": "calibration",
        "stage": script._wave_stage_metadata(calibration),
        "execution_profile": script.EXECUTION_PROFILE,
        "execution_contract": dict(script.WAVE_EXECUTION_CONTRACT),
        "score_id": script.WAVE_COST_MODEL_ID,
        "numeric_policy": script.NUMERIC_POLICY,
        "primary_quantity": script.WAVE_PRIMARY_QUANTITY,
        "timing_scope": script.CALIBRATION_TIMING_SCOPE,
        "calibration_set_sha256": script._file_sha256(calibration_path),
        "configuration_sha256": configuration_sha,
        "normalized_configuration_sha256": normalized_sha,
    }
    sidecar_path = preregistration_dir / "physical.yml.provenance.json"
    sidecar_path.write_bytes(script._canonical_bytes(sidecar))
    checksum_lines = []
    for path in sorted(preregistration_dir.iterdir()):
        checksum_lines.append(
            f"{script._file_sha256(path)}  {path.name}\n"
        )
    (preregistration_dir / "SHA256SUMS").write_text(
        "".join(checksum_lines), encoding="ascii"
    )
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
    assert table[0]["request_transport"] == "packed_operation_v1"
    assert table[0]["output_sha256"] == "3" * 64
    emitted = json.loads((output_dir / "path_runtime_calibration.json").read_text(encoding="utf-8"))
    assert emitted["observations"][0]["sample_id"] == "sample-0"
    assert emitted["observations"][0]["session_instance_id"] == "session-0"


def test_extract_wave_calibration_emits_private_profile_binding(
    tmp_path: Path, monkeypatch
) -> None:
    raw_dir, candidate_path, calibration_path, manifest, samples, sessions = (
        _wave_calibration_fixture(tmp_path)
    )
    monkeypatch.setattr(script, "load_artifacts", lambda path: (manifest, samples, sessions))
    output_dir = tmp_path / "wave-calibration"
    result = script.extract_calibration(raw_dir, candidate_path, calibration_path, output_dir)

    assert result["profile_schema_version"] == "physical_speedup_fit_v1"
    assert result["execution_profile"] == script.EXECUTION_PROFILE
    assert result["execution_contract"] == script.WAVE_EXECUTION_CONTRACT
    assert result["score_id"] == script.WAVE_COST_MODEL_ID
    assert result["primary_quantity"] == "session_inclusive_s"
    assert result["timing_scope"] == "steady_execution_v1"
    assert result["normalization"] == "log((candidate+1)/(greedy+1))"
    assert result["round_id"] == (
        f"{script.WAVE_INITIAL_STAGE}:{manifest['experiment_id']}"
    )
    assert result["candidate_generation_source_sha"] == "f" * 40
    assert result["physical_execution_source_sha"] == script.EXECUTION_SOURCE
    assert set(result["private_provenance"]["checksums"]) == {
        "physical.yml",
        "physical.yml.provenance.json",
        "binary_sha256.json",
        "SHA256SUMS",
    }
    assert len(result["observations"]) == 4
    assert result["observations"][0]["session_inclusive_s"] == 10.75
    assert {(row["sample_index"], row["order_index"]) for row in result["observations"]} == {
        (0, 0)
    }
    table = list(
        csv.DictReader(
            (output_dir / "path_runtime_calibration.csv").open(encoding="utf-8")
        )
    )
    assert set(table[0]) == set(script.WAVE_CALIBRATION_COLUMNS)
    assert table[0]["execution_profile"] == script.EXECUTION_PROFILE
    assert json.loads(table[0]["execution_contract_json"]) == script.WAVE_EXECUTION_CONTRACT
    assert table[0]["primary_quantity"] == "session_inclusive_s"
    assert table[0]["round_id"] == result["round_id"]
    emitted = json.loads(
        (output_dir / "path_runtime_calibration.json").read_text(encoding="utf-8")
    )
    assert emitted["observations"][0]["execution_contract_json"] == table[0][
        "execution_contract_json"
    ]


def test_extract_wave_calibration_accepts_underutilized_diagnostic_facts(
    tmp_path: Path, monkeypatch
) -> None:
    raw_dir, candidate_path, calibration_path, manifest, base_samples, base_sessions = (
        _wave_calibration_fixture(tmp_path)
    )
    dataset = json.loads(candidate_path.read_text(encoding="utf-8"))
    topology = dataset["circuits"][0]["candidates"][0]["topologies"][0]
    topology["resource_admission"].update(
        {
            "tasklet_row_sufficiency_passed": False,
            "dominant_work_wave_tasklet_row_sufficiency_passed": False,
            "dominant_work_wave_allocated_dpu_slots": 1,
            "dominant_work_wave_populated_dpu_slots": 0,
            "collection_resource_admission_passed": False,
        }
    )
    candidate_set_sha = script._sha256_bytes(script._canonical_bytes(dataset))
    candidate_path.write_bytes(script._canonical_bytes(dataset))
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    calibration["candidate_set_sha256"] = candidate_set_sha
    calibration_path.write_bytes(script._canonical_bytes(calibration))
    _refresh_wave_calibration_archive(raw_dir, calibration_path, candidate_set_sha)

    samples = [deepcopy(item) for item in base_samples]
    for sample in samples:
        sample["backend_facts"].update(
            {
                "tasklet_row_sufficiency_passed": False,
                "dominant_work_wave_tasklet_row_sufficiency_passed": False,
                "dominant_work_wave_allocated_dpu_slots": 1,
                "dominant_work_wave_populated_dpu_slots": 0,
                "collection_resource_admission_passed": False,
            }
        )
    monkeypatch.setattr(
        script,
        "load_artifacts",
        lambda path: (manifest, tuple(samples), base_sessions),
    )
    result = script.extract_calibration(
        raw_dir, candidate_path, calibration_path, tmp_path / "underutilized"
    )

    assert result["all_resource_admission_passed"] is False
    assert all(
        row["collection_resource_admission_passed"] is False
        for row in result["observations"]
    )
    runtime_path = tmp_path / "underutilized" / "path_runtime_calibration.json"
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    row_tampered = deepcopy(runtime)
    row_tampered["observations"][0]["collection_resource_admission_passed"] = True
    with pytest.raises(ValueError, match="candidate plan facts"):
        script._wave_stage_fit_inputs(
            json.loads(candidate_path.read_text(encoding="utf-8")),
            json.loads(calibration_path.read_text(encoding="utf-8")),
            row_tampered,
        )
    runtime["all_resource_admission_passed"] = True
    runtime_path.write_bytes(script._canonical_bytes(runtime))
    with pytest.raises(ValueError, match="aggregate"):
        script._wave_stage_fit_inputs(
            json.loads(candidate_path.read_text(encoding="utf-8")),
            json.loads(calibration_path.read_text(encoding="utf-8")),
            runtime,
        )


def test_extract_wave_calibration_rejects_archived_stage_drift(
    tmp_path: Path, monkeypatch
) -> None:
    raw_dir, candidate_path, calibration_path, manifest, samples, sessions = (
        _wave_calibration_fixture(tmp_path)
    )
    sidecar_path = raw_dir.parent / "preregistration" / "physical.yml.provenance.json"
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["stage"]["selection_profile_sha256"] = "1" * 64
    sidecar_path.write_bytes(script._canonical_bytes(sidecar))
    checksum_path = raw_dir.parent / "preregistration" / "SHA256SUMS"
    checksum_lines = []
    for line in checksum_path.read_text(encoding="ascii").splitlines():
        if line.endswith("  physical.yml.provenance.json"):
            line = f"{script._file_sha256(sidecar_path)}  physical.yml.provenance.json"
        checksum_lines.append(line)
    checksum_path.write_text("\n".join(checksum_lines) + "\n", encoding="ascii")
    monkeypatch.setattr(script, "load_artifacts", lambda path: (manifest, samples, sessions))
    with pytest.raises(ValueError, match="archive provenance has a mismatched stage"):
        script.extract_calibration(
            raw_dir, candidate_path, calibration_path, tmp_path / "out"
        )


@pytest.mark.parametrize(
    "corruption,match",
    [
        ("transport", "request_transport"),
        ("numeric", "numeric policy"),
        ("logical", "logical_plan_id"),
        ("binary", "deployment manifest"),
        ("duplicate_session", "fresh session"),
        ("source", "physical execution source"),
    ],
)
def test_extract_wave_calibration_rejects_corrupt_contracts(
    tmp_path: Path, monkeypatch, corruption: str, match: str
) -> None:
    raw_dir, candidate_path, calibration_path, manifest, base_samples, base_sessions = (
        _wave_calibration_fixture(tmp_path)
    )
    samples = [deepcopy(item) for item in base_samples]
    sessions = [deepcopy(item) for item in base_sessions]
    if corruption == "transport":
        samples[0]["backend_facts"]["request_transport"] = "packed_operation_v1"
    elif corruption == "numeric":
        samples[0]["numeric_facts"]["numeric_policy"] = "complex_int8_shared_scale_v1"
    elif corruption == "logical":
        samples[0]["identities"]["logical_plan_id"] = "c" * 64
    elif corruption == "binary":
        sessions[0]["terminal_backend_facts"]["dpu_binary_sha256"] = "f" * 64
    elif corruption == "duplicate_session":
        samples[1]["session_instance_id"] = samples[0]["session_instance_id"]
    else:
        manifest = deepcopy(manifest)
        manifest["source_commit"] = "1" * 40
    monkeypatch.setattr(script, "load_artifacts", lambda path: (manifest, samples, sessions))
    with pytest.raises(ValueError, match=match):
        script.extract_calibration(
            raw_dir, candidate_path, calibration_path, tmp_path / "out"
        )


def test_extract_calibration_preserves_validation_split(
    tmp_path: Path, monkeypatch
) -> None:
    raw_dir, candidate_path, calibration_path, manifest, samples, sessions = (
        _calibration_fixture(tmp_path, split="validation")
    )
    monkeypatch.setattr(script, "load_artifacts", lambda path: (manifest, samples, sessions))
    output_dir = tmp_path / "calibration"
    script.extract_calibration(raw_dir, candidate_path, calibration_path, output_dir)
    table = list(
        csv.DictReader(
            (output_dir / "path_runtime_calibration.csv").open(encoding="utf-8")
        )
    )
    assert {row["split"] for row in table} == {"validation"}


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


def _extract_wave_fixture(
    tmp_path: Path, monkeypatch
) -> tuple[Path, Path, Path, Path]:
    raw_dir, candidate_path, calibration_path, manifest, samples, sessions = (
        _wave_calibration_fixture(tmp_path)
    )
    monkeypatch.setattr(script, "load_artifacts", lambda path: (manifest, samples, sessions))
    extract_dir = tmp_path / "wave-extract"
    script.extract_calibration(raw_dir, candidate_path, calibration_path, extract_dir)
    return (
        candidate_path,
        calibration_path,
        extract_dir / "path_runtime_calibration.json",
        extract_dir,
    )


def _fit_wave_fixture(
    tmp_path: Path, monkeypatch, *, model_form: str = "six_term"
) -> tuple[Path, Path, Path, Any]:
    candidate_path, calibration_path, runtime_path, _extract_dir = _extract_wave_fixture(
        tmp_path, monkeypatch
    )
    fit_dir = tmp_path / f"fit-{model_form}"
    result = script.fit(
        candidate_path,
        calibration_path,
        runtime_path,
        fit_dir,
        samples=4,
        seed=17,
        model_form=model_form,
    )
    return candidate_path, fit_dir / "physical_speedup_fit_v1.json", runtime_path, result


def _refresh_wave_calibration_archive(
    raw_dir: Path, calibration_path: Path, candidate_set_sha: str
) -> None:
    archive_root = raw_dir.parent / "preregistration"
    sidecar_path = archive_root / "physical.yml.provenance.json"
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["candidate_set_sha256"] = candidate_set_sha
    sidecar["calibration_set_sha256"] = script._file_sha256(calibration_path)
    sidecar_path.write_bytes(script._canonical_bytes(sidecar))
    checksum_path = archive_root / "SHA256SUMS"
    checksum_lines = []
    for line in checksum_path.read_text(encoding="ascii").splitlines():
        if line.endswith("  physical.yml.provenance.json"):
            line = f"{script._file_sha256(sidecar_path)}  physical.yml.provenance.json"
        checksum_lines.append(line)
    checksum_path.write_text("\n".join(checksum_lines) + "\n", encoding="ascii")


def _wave_fixed_pool_fixture(
    tmp_path: Path,
    monkeypatch,
    *,
    duplicate_physical: bool = False,
) -> tuple[Path, Path, Path, Path, Path]:
    raw_dir, candidate_path, calibration_path, manifest, samples, sessions = (
        _wave_calibration_fixture(tmp_path)
    )
    dataset = json.loads(candidate_path.read_text(encoding="utf-8"))
    circuit = dataset["circuits"][0]
    base = deepcopy(circuit["candidates"][0])
    base_physical = base["topologies"][0]["physical_plan_id"]
    for index, physical_id in enumerate(
        (
            base_physical if duplicate_physical else "4" * 64,
            base_physical if duplicate_physical else "5" * 64,
        ),
        start=2,
    ):
        candidate = deepcopy(base)
        logical_id = str(index + 2) * 64
        candidate.update(
            {
                "candidate_path_id": str(index) * 64,
                "logical_plan_id": logical_id,
                "source_kind": "cotengra_one_trial",
                "source_seed": index,
                "planner_config_hash": str(index + 6) * 64,
                "is_greedy": False,
            }
        )
        topology = candidate["topologies"][0]
        raw = script.RawFeatureVector(
            float(index), float(index + 1), float(index + 2), float(index + 3), 0.0, 1.0
        )
        topology["logical_plan_id"] = logical_id
        topology["physical_plan_id"] = physical_id
        topology["wave_facts"]["plan"]["logical_plan_id"] = logical_id
        topology["wave_facts"]["plan"]["physical_plan_id"] = physical_id
        topology["wave_facts"]["raw"] = raw.as_mapping()
        topology["features"] = raw.as_mapping()
        circuit["candidates"].append(candidate)
    dataset["circuits"][0]["unique_candidate_count"] = len(circuit["candidates"])
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    calibration["candidate_set_sha256"] = script._sha256_bytes(
        script._canonical_bytes(dataset)
    )
    candidate_path.write_bytes(script._canonical_bytes(dataset))
    calibration_path.write_bytes(script._canonical_bytes(calibration))
    _refresh_wave_calibration_archive(
        raw_dir, calibration_path, calibration["candidate_set_sha256"]
    )
    monkeypatch.setattr(script, "load_artifacts", lambda path: (manifest, samples, sessions))
    extract_dir = tmp_path / "initial-extract"
    script.extract_calibration(raw_dir, candidate_path, calibration_path, extract_dir)
    fit_dir = tmp_path / "initial-fit"
    script.fit(
        candidate_path,
        calibration_path,
        extract_dir / "path_runtime_calibration.json",
        fit_dir,
        samples=4,
        seed=17,
        model_form="six_term",
    )
    adaptive_calibration_path = tmp_path / "adaptive-calibration.json"
    profile_path = fit_dir / "physical_speedup_fit_v1.json"
    script.propose_wave_calibration_stage(
        candidate_path,
        (calibration_path,),
        profile_path,
        adaptive_calibration_path,
        round_ordinal=1,
    )
    return (
        candidate_path,
        calibration_path,
        extract_dir / "path_runtime_calibration.json",
        profile_path,
        adaptive_calibration_path,
    )


def _wave_adaptive_runtime(
    initial_runtime_path: Path,
    adaptive_calibration_path: Path,
    candidate_path: Path,
    profile_path: Path,
    output_path: Path,
) -> Path:
    initial = json.loads(initial_runtime_path.read_text(encoding="utf-8"))
    calibration = json.loads(adaptive_calibration_path.read_text(encoding="utf-8"))
    dataset = json.loads(candidate_path.read_text(encoding="utf-8"))
    candidate_by_id = {
        candidate["candidate_path_id"]: candidate
        for candidate in dataset["circuits"][0]["candidates"]
    }
    experiment_id = "2" * 64
    run_id = "adaptive-run-1"
    round_id = f"{script.WAVE_ADAPTIVE_STAGE}:1:{experiment_id}"
    artifacts = {
        "manifest.json": "c" * 64,
        "samples.jsonl": "d" * 64,
        "sessions.jsonl": "e" * 64,
    }
    template_by_block = {
        int(row["block"]): row for row in initial["observations"]
    }
    observations = []
    for cell in calibration["cells"]:
        for candidate_id in cell["candidate_path_ids"]:
            candidate = candidate_by_id[candidate_id]
            topology = candidate["topologies"][0]
            for block in range(4):
                row = deepcopy(template_by_block[block])
                row.update(
                    {
                        "candidate_path_id": candidate_id,
                        "plan_id": f"path_{candidate_id}",
                        "logical_plan_id": candidate["logical_plan_id"],
                        "physical_plan_id": topology["physical_plan_id"],
                        "experiment_id": experiment_id,
                        "run_id": run_id,
                        "round_id": round_id,
                        "calibration_set_sha256": script._sha256_bytes(
                            script._canonical_bytes(calibration)
                        ),
                        "sample_id": f"adaptive-sample-{candidate_id[:4]}-{block}",
                        "session_instance_id": f"adaptive-session-{candidate_id[:4]}-{block}",
                        "raw_artifact_sha256": artifacts,
                    }
                )
                observations.append(row)
    runtime = {
        **initial,
        "experiment_id": experiment_id,
        "run_id": run_id,
        "round_id": round_id,
        "stage_experiment_id": round_id,
        "stage_id": script.WAVE_ADAPTIVE_STAGE,
        "round_ordinal": 1,
        "prior_stage_hashes": [
            script._file_sha256(
                adaptive_calibration_path.parent / "calibration_candidate_set.json"
            )
        ],
        "selection_profile_sha256": script._file_sha256(profile_path),
        "timing_used_for_selection": True,
        "calibration_set_sha256": script._sha256_bytes(
            script._canonical_bytes(calibration)
        ),
        "raw_artifact_sha256": artifacts,
        "expected_candidate_cell_count": len(
            [
                item
                for cell in calibration["cells"]
                for item in cell["candidate_path_ids"]
            ]
        ),
        "sample_count": len(observations),
        "session_count": len(observations),
        "observations": observations,
    }
    output_path.write_bytes(script._canonical_bytes(runtime))
    return output_path


def test_wave_adaptive_candidate_set_is_deterministic_and_does_not_refill(
    tmp_path: Path, monkeypatch
) -> None:
    candidate_path, _initial_calibration, _runtime, _profile, adaptive_path = (
        _wave_fixed_pool_fixture(tmp_path, monkeypatch)
    )
    calibration = json.loads(adaptive_path.read_text(encoding="utf-8"))
    cell = calibration["cells"][0]
    roles = {item["role"]: item["candidate_path_id"] for item in cell["candidate_roles"]}
    assert calibration["stage_id"] == script.WAVE_ADAPTIVE_STAGE
    assert calibration["round_ordinal"] == 1
    assert calibration["timing_used_for_selection"] is True
    assert calibration["prior_stage_hashes"]
    assert calibration["selection_profile_sha256"]
    assert roles["greedy"] == cell["greedy_path_id"]
    assert roles["incumbent"] == cell["greedy_path_id"]
    assert cell["candidate_path_ids"][0] == cell["greedy_path_id"]
    assert len(cell["candidate_path_ids"]) == 2
    assert roles["new_fixed_pool_candidate"] in cell["candidate_path_ids"]
    assert json.loads(candidate_path.read_text(encoding="utf-8"))["execution_profile"] == (
        script.EXECUTION_PROFILE
    )


def test_wave_adaptive_candidate_set_marks_exhaustion_after_physical_dedup(
    tmp_path: Path, monkeypatch
) -> None:
    _candidate, _initial_calibration, _runtime, _profile, adaptive_path = (
        _wave_fixed_pool_fixture(tmp_path, monkeypatch, duplicate_physical=True)
    )
    calibration = json.loads(adaptive_path.read_text(encoding="utf-8"))
    cell = calibration["cells"][0]
    assert cell["candidate_path_ids"] == [cell["greedy_path_id"]]
    assert cell["candidate_roles"][2] == {
        "role": "new_fixed_pool_candidate",
        "candidate_path_id": None,
    }
    assert cell["no_more_candidate"] is True
    assert calibration["exhausted_cells"] == [cell["cell_id"]]


def test_wave_fit_stages_preserves_round_pairing_and_profile_fields(
    tmp_path: Path, monkeypatch
) -> None:
    candidate_path, initial_calibration, initial_runtime, profile_path, adaptive_path = (
        _wave_fixed_pool_fixture(tmp_path, monkeypatch)
    )
    adaptive_runtime = _wave_adaptive_runtime(
        initial_runtime,
        adaptive_path,
        candidate_path,
        profile_path,
        tmp_path / "adaptive-runtime.json",
    )
    output_dir = tmp_path / "multi-fit"
    result = script.fit_wave_stages(
        candidate_path,
        ((initial_calibration, initial_runtime), (adaptive_path, adaptive_runtime)),
        output_dir,
        samples=4,
        seed=17,
        model_form="six_term",
    )
    profile = json.loads(
        (output_dir / "physical_speedup_fit_v1.json").read_text(encoding="utf-8")
    )
    dataset = json.loads(candidate_path.read_text(encoding="utf-8"))
    greedy_id = dataset["circuits"][0]["candidates"][0]["candidate_path_id"]
    extra_id = dataset["circuits"][0]["candidates"][1]["candidate_path_id"]
    assert result is not None
    assert profile["stage_count"] == 2
    assert profile["adaptive_round_count"] == 1
    assert profile["stage_id"] == script.WAVE_ADAPTIVE_STAGE
    assert profile["round_ordinal"] == 1
    assert profile["stage_metadata"][0]["stage_id"] == script.WAVE_INITIAL_STAGE
    assert profile["stage_metadata"][1]["stage_id"] == script.WAVE_ADAPTIVE_STAGE
    assert profile["stage_calibration_sha256"] == [
        script._file_sha256(initial_calibration),
        script._file_sha256(adaptive_path),
    ]
    assert profile["training_cell_ids"] == ["fixture:1dpu_t8"]
    assert set(profile["selected_path_ids"]) == {"fixture:1dpu_t8"}
    assert set(profile["candidate_speedups"]["fixture:1dpu_t8"]) == {
        greedy_id,
        extra_id,
    }
    assert ("fixture:1dpu_t8", extra_id, 3) in [
        tuple(item) for item in result.paired_observation_counts
    ]


def test_wave_stage_provenance_rejects_boolean_integer_fields(
    tmp_path: Path, monkeypatch
) -> None:
    _candidate_path, profile_path, _runtime_path, _result = _fit_wave_fixture(
        tmp_path, monkeypatch
    )
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    mutations = (
        ("adaptive_round_count", lambda value: value.update(adaptive_round_count=False)),
        ("round_ordinal", lambda value: value.update(round_ordinal=False)),
        (
            "stage_metadata round_ordinal",
            lambda value: value["stage_metadata"][0].update(round_ordinal=False),
        ),
    )
    for _name, mutate in mutations:
        invalid = deepcopy(profile)
        mutate(invalid)
        with pytest.raises(ValueError):
            script.validate_wave_stage_provenance(invalid)


@pytest.mark.parametrize(
    "corruption,match",
    [
        ("duplicate_sample", "reuse sample"),
        ("source", "candidate_generation_source_sha"),
        ("split", "split"),
        ("hash", "candidate_set_sha256"),
        ("prior", "prior_stage_hashes"),
        ("run", "distinct run"),
    ],
)
def test_wave_fit_stages_rejects_cross_stage_drift_and_reuse(
    tmp_path: Path, monkeypatch, corruption: str, match: str
) -> None:
    candidate_path, initial_calibration, initial_runtime, profile_path, adaptive_path = (
        _wave_fixed_pool_fixture(tmp_path, monkeypatch)
    )
    adaptive_runtime_path = _wave_adaptive_runtime(
        initial_runtime,
        adaptive_path,
        candidate_path,
        profile_path,
        tmp_path / "adaptive-runtime.json",
    )
    runtime = json.loads(adaptive_runtime_path.read_text(encoding="utf-8"))
    if corruption == "duplicate_sample":
        initial = json.loads(initial_runtime.read_text(encoding="utf-8"))
        runtime["observations"][0]["sample_id"] = initial["observations"][0]["sample_id"]
    elif corruption == "source":
        runtime["candidate_generation_source_sha"] = "0" * 40
    elif corruption == "split":
        runtime["observations"][0]["split"] = "validation"
    elif corruption == "hash":
        runtime["candidate_set_sha256"] = "0" * 64
    elif corruption == "prior":
        calibration = json.loads(adaptive_path.read_text(encoding="utf-8"))
        calibration["prior_stage_hashes"] = ["0" * 64]
        adaptive_path.write_bytes(script._canonical_bytes(calibration))
        calibration_sha = script._file_sha256(adaptive_path)
        runtime["calibration_set_sha256"] = calibration_sha
        runtime["prior_stage_hashes"] = ["0" * 64]
        for row in runtime["observations"]:
            row["calibration_set_sha256"] = calibration_sha
    else:
        initial = json.loads(initial_runtime.read_text(encoding="utf-8"))
        runtime["run_id"] = initial["run_id"]
        for row in runtime["observations"]:
            row["run_id"] = initial["run_id"]
    adaptive_runtime_path.write_bytes(script._canonical_bytes(runtime))
    with pytest.raises(ValueError, match=match):
        script.fit_wave_stages(
            candidate_path,
            ((initial_calibration, initial_runtime), (adaptive_path, adaptive_runtime_path)),
            tmp_path / "failed-fit",
            samples=4,
            seed=17,
            model_form="six_term",
        )


def _load_qualifier_for_owned_integration_test() -> Any:
    path = Path(__file__).resolve().parents[1] / "scripts" / "qualify_upmem_path_candidates.py"
    spec = importlib.util.spec_from_file_location("qualify_wave_profile_integration", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_wave_fit_profile_is_accepted_by_qualifier(
    tmp_path: Path, monkeypatch
) -> None:
    candidate_path, profile_path, _runtime_path, result = _fit_wave_fixture(
        tmp_path, monkeypatch
    )
    dataset = json.loads(candidate_path.read_text(encoding="utf-8"))
    qualifier = _load_qualifier_for_owned_integration_test()
    loaded_profile, _profile_hash, weights, model = qualifier._load_frozen_profile(
        profile_path,
        dataset=dataset,
        dataset_hash=script._sha256_bytes(script._canonical_bytes(dataset)),
    )
    assert loaded_profile["score_id"] == script.WAVE_COST_MODEL_ID
    assert loaded_profile["physical_execution_source_sha"] == script.EXECUTION_SOURCE
    assert loaded_profile["source_sha"] == dataset["source_sha"]
    assert loaded_profile["candidate_generation_source_sha"] == dataset["source_sha"]
    assert loaded_profile["primary_quantity"] == script.WAVE_PRIMARY_QUANTITY
    assert loaded_profile["timing_scope"] == script.CALIBRATION_TIMING_SCOPE
    assert loaded_profile["normalization"] == script.WAVE_NORMALIZATION
    assert weights == result.weights
    assert model.mode == "six_term"


@pytest.mark.parametrize(
    "corruption,match",
    [
        ("hash", "candidate_set_sha256"),
        ("split", "split"),
        ("timing", "timing_scope"),
    ],
)
def test_wave_fit_rejects_runtime_hash_split_and_timing_corruption(
    tmp_path: Path, monkeypatch, corruption: str, match: str
) -> None:
    candidate_path, calibration_path, runtime_path, _extract_dir = _extract_wave_fixture(
        tmp_path, monkeypatch
    )
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    if corruption == "hash":
        runtime["candidate_set_sha256"] = "0" * 64
    elif corruption == "split":
        runtime["observations"][0]["split"] = "validation"
    else:
        runtime["observations"][0]["timing_scope"] = "serial_nodes_v1"
    runtime_path.write_bytes(script._canonical_bytes(runtime))
    with pytest.raises(ValueError, match=match):
        script.fit(
            candidate_path,
            calibration_path,
            runtime_path,
            tmp_path / "fit",
            samples=4,
            seed=17,
            model_form="six_term",
        )


def test_wave_grouped_and_six_term_selection_parity(
    tmp_path: Path, monkeypatch
) -> None:
    candidate_path, calibration_path, runtime_path, _extract_dir = _extract_wave_fixture(
        tmp_path, monkeypatch
    )
    six = script.fit(
        candidate_path,
        calibration_path,
        runtime_path,
        tmp_path / "six",
        samples=4,
        seed=17,
        model_form="six_term",
    )
    grouped = script.fit(
        candidate_path,
        calibration_path,
        runtime_path,
        tmp_path / "grouped",
        samples=4,
        seed=17,
        model_form="grouped",
    )
    assert six.selected_path_ids == grouped.selected_path_ids
    assert six.weights.numeric == grouped.weights.numeric == 0.0
    assert six.weights.wram == grouped.weights.wram == 0.0


def test_wave_evaluate_uses_the_full_heldout_wave_pool(
    tmp_path: Path, monkeypatch
) -> None:
    raw_dir, candidate_path, calibration_path, manifest, samples, sessions = (
        _wave_calibration_fixture(tmp_path)
    )
    dataset = json.loads(candidate_path.read_text(encoding="utf-8"))
    heldout = deepcopy(dataset["circuits"][0])
    heldout["circuit_id"] = "heldout"
    heldout["split"] = "validation"
    dataset["circuits"].append(heldout)
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    calibration["candidate_set_sha256"] = script._sha256_bytes(
        script._canonical_bytes(dataset)
    )
    candidate_path.write_bytes(script._canonical_bytes(dataset))
    calibration_path.write_bytes(script._canonical_bytes(calibration))
    _refresh_wave_calibration_archive(
        raw_dir, calibration_path, calibration["candidate_set_sha256"]
    )
    monkeypatch.setattr(script, "load_artifacts", lambda path: (manifest, samples, sessions))
    extract_dir = tmp_path / "extract"
    script.extract_calibration(raw_dir, candidate_path, calibration_path, extract_dir)
    fit_dir = tmp_path / "fit"
    script.fit(
        candidate_path,
        calibration_path,
        extract_dir / "path_runtime_calibration.json",
        fit_dir,
        samples=4,
        seed=17,
        model_form="grouped",
    )
    monkeypatch.setattr(
        script,
        "WAVE_TOPOLOGY_RESOURCES",
        {"1dpu_t8": {"dpu_count": 1, "rank_count": 1, "tasklets_per_dpu": 8}},
    )
    result = script.evaluate_frozen_profile(
        candidate_path,
        fit_dir / "physical_speedup_fit_v1.json",
        tmp_path / "selection.json",
        split="validation",
    )
    assert len(result["selections"]) == 1
    assert result["selections"][0]["upmem_selected_path_id"] == (
        result["selections"][0]["greedy_path_id"]
    )
    assert result["score_id"] == script.WAVE_COST_MODEL_ID
