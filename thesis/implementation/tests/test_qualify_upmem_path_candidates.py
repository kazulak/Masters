from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import yaml


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "qualify_upmem_path_candidates.py"
SPEC = importlib.util.spec_from_file_location("qualify_upmem_path_candidates", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
qualify = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = qualify
SPEC.loader.exec_module(qualify)


@pytest.mark.parametrize("simulator", [False, True])
@pytest.mark.parametrize("topology,dpus", [("1dpu_t8", 1), ("4dpu_t8", 4)])
def test_frozen_wave_route_uses_qualified_execution_policy(simulator, topology, dpus):
    from quantum_bench.experiment import upmem_execution_policy

    route = qualify._route(topology, simulator=simulator, prepared_waves=True)
    options = route["options"]
    policy = upmem_execution_policy(options)
    assert options["dpu_count"] == dpus
    assert options["rank_count"] == 1
    assert options["tasklets_per_dpu"] == 8
    assert options["dpu_binary"].endswith("/dpu_wave_v5_t8")
    assert policy["request_transport"] == "packed_wave_v1"
    assert policy["schedule_policy"] == "static_dag_waves_v1"
    assert policy["fuse_complex"] is True
    assert policy["geometry_policy"] == "panel_only_v1"
    assert route["numeric_policy"] == "split_complex_float32_v1"
    assert options.get("rank_paths") == (None if simulator else ["/dev/dpu_rank1"])


def test_path_route_rejects_unknown_topology_instead_of_using_four_dpus():
    with pytest.raises(ValueError, match="topology"):
        qualify._route("not-a-topology", simulator=True)


def test_wave_route_binds_explicit_absolute_execution_root():
    root = Path("/frozen-executor/thesis/implementation")
    route = qualify._route(
        "4dpu_t8", simulator=False, prepared_waves=True, execution_root=root
    )
    for field in ("session_root", "host_binary", "dpu_binary", "initialization_binary"):
        assert Path(route["options"][field]).is_relative_to(root)
    with pytest.raises(ValueError, match="absolute implementation path"):
        qualify._route(
            "4dpu_t8", simulator=False, prepared_waves=True,
            execution_root=Path("relative-checkout"),
        )


def test_final_system_study_manifest_has_consistent_finite_execution_contract():
    config = qualify._load(
        qualify.ROOT / "configs" / "upmem_final_system_path_study_v2.json"
    )
    contract = qualify.execution_contract(config)
    assert contract["cost_model_id"] == "upmem_slr_wave_cost_v1"
    assert config["host_memory_admission_bytes"] == 512 * 1024 * 1024
    assert config["host_memory_policy"]["reserve_bytes"] == 0
    assert config["timing_policy"]["scope_id"] == "steady_execution_v1"
    assert config["timing_policy"]["primary_quantity"] == "session_inclusive_s"
    assert config["adaptive_search"]["maximum_rounds"] == 3
    assert config["candidate_pool"]["candidate_generation_after_first_timing"] is False
    stages = config["budget"]["stages"]
    calculated = []
    for stage in stages:
        paths = stage.get("unique_paths_per_cell")
        if paths is None:
            paths = stage.get("unique_paths_per_cell_per_round")
        if paths is None:
            paths = len(stage.get("arms", stage.get("path_roles", [])))
        attempts = (
            len(stage["instances"]) * len(stage["topologies"]) * paths
            * stage.get("maximum_rounds", 1)
            * (stage["warmup_blocks"] + stage["measurement_blocks"])
        )
        assert attempts == stage["attempts"]
        calculated.append(attempts)
    assert sum(calculated) == config["budget"]["attempt_ceiling"] == 540
    assert config["budget"]["physical_stage_elapsed_ceiling_s"] == 24 * 60 * 60
    comparison = config["model_comparison"]
    assert comparison["data_split"] == "training_only"
    assert comparison["no_validation_or_test_timing"] is True
    assert comparison["bootstrap_resamples"] == 2000
    assert comparison["bootstrap_seed"] == 20260904
    assert comparison["confidence_level"] == 0.95
    assert comparison["stability_resamples"] == 200
    assert comparison["stability_fit_sample_count"] == 1000
    assert comparison["stability_seed"] == 20260905
    assert comparison["bootstrap_unit"] == "complete paired block within each physical round"
    identifiers = [case["circuit_id"] for case in config["circuits"]]
    assert len(identifiers) == len(set(identifiers))
    assert {case["circuit_id"] for case in config["circuits"] if case["split"] == "test"} == {
        "ghz_chain_15q", "xor_17q"
    }


def _candidate(identifier: str, *, greedy: bool, seed: int | None, host: int) -> dict[str, object]:
    topologies = []
    for topology in ("1dpu_t8", "4dpu_t8"):
        topologies.append(
            {
                "topology_id": topology,
                "topology": {
                    "dpu_count": 1 if topology == "1dpu_t8" else 4,
                    "rank_count": 1,
                    "tasklets_per_dpu": 8,
                },
                "feasible": True,
                "physical_plan_id": f"physical-{topology}-{identifier}",
                "resource_admission": {
                    "collection_resource_admission_passed": True,
                },
                "features": {
                    "B_host_dpu": host,
                    "B_mram_wram": 20,
                    "I_dpu": 30,
                    "N_sync": 40,
                    "E_num": 0,
                    "P_wram": 1,
                },
            }
        )
    return {
        "candidate_path_id": identifier,
        "is_greedy": greedy,
        "source_kind": "opt_einsum_greedy" if greedy else "cotengra_one_trial",
        "source_seed": seed,
        "planner_config_hash": f"planner-{identifier}",
        "logical_plan_id": f"logical-{identifier}",
        "conventional_features": {
            "flops": 10 if greedy else 9,
            "macs": 5,
            "peak_intermediate_elements": 4,
            "peak_intermediate_bytes": 64,
            "total_intermediate_writes": 4,
            "maximum_intermediate_rank": 2,
            "contraction_count": 1,
        },
        "topologies": topologies,
    }


def _mark_wave_dataset(dataset: dict[str, object]) -> dict[str, object]:
    metadata = {
        "execution_profile": "kernel_schedule_system_v1",
        "execution_contract": qualify.execution_contract(
            {"execution_profile": "kernel_schedule_system_v1"}
        ),
        "score_id": "upmem_slr_wave_cost_v1",
    }
    dataset.update(metadata)
    for circuit in dataset["circuits"]:
        for candidate in circuit["candidates"]:
            candidate.update(metadata)
            for topology in candidate["topologies"]:
                resources = topology["topology"]
                topology.update(metadata)
                raw = {
                    key: float(value) for key, value in topology["features"].items()
                }
                topology["wave_facts"] = {
                    "cost_model_id": metadata["execution_contract"]["cost_model_id"],
                    "plan": {
                        "logical_plan_id": candidate["logical_plan_id"],
                        "physical_plan_id": topology["physical_plan_id"],
                        "schedule_policy": "static_dag_waves_v1",
                        "numeric_policy": qualify.FLOAT32,
                        "geometry_policy": "panel_only_v1",
                        "fuse_complex": True,
                        "request_transport": "packed_wave_v1",
                        "kernel_identity": "dpu_panel_dispatch_v5_v1",
                        "rank_count": resources["rank_count"],
                        "dpu_count": resources["dpu_count"],
                        "tasklets_per_dpu": resources["tasklets_per_dpu"],
                    },
                    "raw": raw,
                }
                estimate = 100
                topology["host_memory_estimate_bytes"] = estimate
                topology["memory_admission"] = {
                    "scope": "declared_prepared_wave_executor_bytes_v1",
                    "declared_executor_memory_estimate_bytes": estimate,
                    "configured_budget_bytes": 512 * 1024 * 1024,
                    "configured_reserve_bytes": 0,
                    "required_bytes": estimate,
                    "passed": True,
                }
                topology["resource_admission"].update(
                    {
                        "tasklet_row_sufficiency_passed": True,
                        "dominant_work_wave_tasklet_row_sufficiency_passed": True,
                        "dominant_work_wave_allocated_dpu_slots": resources["dpu_count"],
                        "dominant_work_wave_populated_dpu_slots": resources["dpu_count"],
                    }
                )
    return dataset


def _evaluation_fixture(
    tmp_path: Path, *, split: str = "validation", wave: bool = False
) -> tuple[Path, Path, Path, str, str]:
    greedy = "a" * 64
    candidate = "b" * 64
    dataset = {
        "source_sha": "1" * 40,
        "preregistration_sha256": "2" * 64,
        "circuits": [
            {
                "circuit_id": "held-out",
                "split": split,
                "circuit": {
                    "kind": "builtin",
                    "name": "bell_2q",
                    "parameters": {},
                },
                "candidates": [
                    _candidate(greedy, greedy=True, seed=None, host=100),
                    _candidate(candidate, greedy=False, seed=20260903, host=50),
                ],
            }
        ],
    }
    if wave:
        _mark_wave_dataset(dataset)
    selection = {
        "schema_version": "upmem_path_frozen_selection_v1",
        "score_id": (
            "upmem_slr_wave_cost_v1" if wave else "upmem_slr_cost_v1"
        ),
        "source_sha": dataset["source_sha"],
        "candidate_set_sha256": qualify._candidate_set_sha256(dataset),
        "split": split,
        "timing_used_for_selection": False,
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
            "active_features": ["B_host_dpu"],
            "zero_range_features": [],
            "correlated_pairs": [],
            "matrix_rank": 1,
            "rank_tolerance": 0.0,
            "reason": "test profile",
        },
        "selections": [
            {
                "circuit_id": "held-out",
                "split": split,
                "topology_id": topology,
                "greedy_path_id": greedy,
                "minimum_flops_path_id": candidate,
                "upmem_selected_path_id": candidate,
            }
            for topology in ("1dpu_t8", "4dpu_t8")
        ],
    }
    if wave:
        selection.update(
            {
                "execution_profile": dataset["execution_profile"],
                "execution_contract": dataset["execution_contract"],
                "primary_quantity": "session_inclusive_s",
                "timing_scope": "steady_execution_v1",
                "numeric_policy": qualify.FLOAT32,
                "preregistration_sha256": dataset["preregistration_sha256"],
            }
        )
    dataset_path = tmp_path / "dataset.json"
    selection_path = tmp_path / "selection.json"
    profile = {
        "schema_version": "physical_speedup_fit_v1",
        "score_id": (
            "upmem_slr_wave_cost_v1" if wave else "upmem_slr_cost_v1"
        ),
        "normalization": "log((candidate+1)/(greedy+1))",
        "source_sha": dataset["source_sha"],
        "candidate_generation_source_sha": dataset["source_sha"],
        "candidate_set_sha256": qualify._candidate_set_sha256(dataset),
        "weights": selection["weights"],
        "feature_model": selection["feature_model"],
    }
    if wave:
        stage = {
            "stage_id": "initial_training",
            "round_ordinal": 0,
            "prior_stage_hashes": [],
            "selection_profile_sha256": None,
            "timing_used_for_selection": False,
            "experiment_id": "5" * 64,
            "run_id": "fixture-initial-run",
            "round_id": "initial_training:" + "5" * 64,
            "calibration_set_sha256": "3" * 64,
            "runtime_table_sha256": "4" * 64,
        }
        profile.update(
            {
                **stage,
                "fit_splits": ["training"],
                "stage_count": 1,
                "adaptive_round_count": 0,
                "stage_calibration_sha256": ["3" * 64],
                "stage_runtime_table_sha256": ["4" * 64],
                "stage_metadata": [stage],
                "execution_profile": dataset["execution_profile"],
                "execution_contract": dataset["execution_contract"],
                "primary_quantity": "session_inclusive_s",
                "timing_scope": "steady_execution_v1",
                "numeric_policy": qualify.FLOAT32,
                "preregistration_sha256": dataset["preregistration_sha256"],
                "physical_execution_source_sha": dataset["execution_contract"][
                    "execution_source"
                ],
                "calibration_set_sha256": "3" * 64,
                "runtime_table_sha256": "4" * 64,
            }
        )
    profile_path = tmp_path / "profile.json"
    dataset_path.write_bytes(qualify._canonical_bytes(dataset))
    selection_path.write_bytes(qualify._canonical_bytes(selection))
    profile_path.write_bytes(qualify._canonical_bytes(profile))
    selection["fitted_profile_sha256"] = hashlib.sha256(
        qualify._canonical_bytes(profile)
    ).hexdigest()
    selection_path.write_bytes(qualify._canonical_bytes(selection))
    return dataset_path, selection_path, profile_path, greedy, candidate


@pytest.mark.parametrize("split", ("validation", "test"))
def test_prepare_evaluation_config_uses_frozen_selection_once_per_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, split: str
) -> None:
    dataset_path, selection_path, profile_path, greedy, candidate = _evaluation_fixture(
        tmp_path, split=split
    )

    def fail_if_consulted(_path: Path) -> dict[tuple[str, str], str]:
        raise AssertionError("evaluation mode consulted timing/ranking input")

    monkeypatch.setattr(qualify, "_ranking_best", fail_if_consulted)
    monkeypatch.setattr(qualify, "_regenerate", lambda circuit, selected: (object(), {}))
    output = tmp_path / "validation.yml"
    config = qualify.prepare_config(
        dataset_path=dataset_path,
        selection_path=selection_path,
        output_path=output,
        mode="evaluation",
        profile_path=profile_path,
        split=split,
    )

    assert config["experiment_id"] == f"upmem-path-heuristic-evaluation-{split}-v1"
    assert config["collection"]["warmup_blocks"] == 1
    assert config["collection"]["measurement_blocks"] == 5
    assert set(config["routes"]) == {"1dpu_t8", "4dpu_t8"}
    assert all(route["executor"] == "upmem_physical" for route in config["routes"].values())
    assert len(config["plans"]) == 2
    assert set(config["plans"]) == {f"path_{greedy}", f"path_{candidate}"}
    assert config["matrix"] == [
        {
            "case_id": "held-out",
            "plan_id": f"path_{greedy}",
            "route_ids": ["1dpu_t8", "4dpu_t8"],
        },
        {
            "case_id": "held-out",
            "plan_id": f"path_{candidate}",
            "route_ids": ["1dpu_t8", "4dpu_t8"],
        },
    ]
    provenance = json.loads(
        output.with_suffix(".yml.provenance.json").read_text(encoding="utf-8")
    )
    assert provenance["selection_split"] == split
    assert provenance["fitted_profile_sha256"] == hashlib.sha256(
        qualify._canonical_bytes(
            json.loads(profile_path.read_text(encoding="utf-8"))
        )
    ).hexdigest()
    assert len(provenance["selected"]) == 4
    assert provenance["candidate_set_sha256"] == qualify._candidate_set_sha256(
        json.loads(dataset_path.read_text(encoding="utf-8"))
    )


def test_prepare_evaluation_config_can_target_strict_sdk_simulator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset_path, selection_path, profile_path, greedy, candidate = _evaluation_fixture(tmp_path)
    monkeypatch.setattr(qualify, "_regenerate", lambda circuit, selected: (object(), {}))
    output = tmp_path / "sdk.yml"
    config = qualify.prepare_config(
        dataset_path=dataset_path,
        selection_path=selection_path,
        output_path=output,
        mode="evaluation",
        profile_path=profile_path,
        split="validation",
        execution_target="sdk",
    )

    assert config["experiment_id"] == "upmem-path-heuristic-evaluation-validation-sdk-v1"
    assert config["collection"]["warmup_blocks"] == 0
    assert config["collection"]["measurement_blocks"] == 1
    assert set(config["routes"]) == {"1dpu_t8"}
    assert all(item["route_ids"] == ["1dpu_t8"] for item in config["matrix"])
    assert all(
        route["executor"] == "upmem_sdk_simulator"
        for route in config["routes"].values()
    )
    assert all("rank_paths" not in route["options"] for route in config["routes"].values())
    assert set(config["plans"]) == {f"path_{greedy}", f"path_{candidate}"}
    provenance = json.loads(
        output.with_suffix(".yml.provenance.json").read_text(encoding="utf-8")
    )
    assert provenance["execution_target"] == "sdk"
    assert {row["topology_id"] for row in provenance["selection_roles"]} == {
        "1dpu_t8",
        "4dpu_t8",
    }


@pytest.mark.parametrize("mode", ("six_term", "grouped"))
def test_wave_profile_rejects_weight_for_inactive_feature(tmp_path: Path, mode: str) -> None:
    dataset_path, _, profile_path, _, _ = _evaluation_fixture(tmp_path, wave=True)
    dataset = json.loads(dataset_path.read_text())
    profile = json.loads(profile_path.read_text())
    profile["feature_model"]["mode"] = mode
    profile["feature_model"]["active_features"] = []
    profile_path.write_bytes(qualify._canonical_bytes(profile))
    with pytest.raises(ValueError, match="inactive feature"):
        qualify._load_frozen_profile(
            profile_path, dataset=dataset,
            dataset_hash=qualify._candidate_set_sha256(dataset),
        )


@pytest.mark.parametrize(
    "corruption",
    ("missing_chain", "runtime_hash", "prior_hash", "experiment", "timing_selection"),
)
def test_wave_qualifier_rejects_corrupt_stage_provenance(tmp_path, corruption):
    dataset_path, _, profile_path, _, _ = _evaluation_fixture(tmp_path, wave=True)
    dataset = json.loads(dataset_path.read_text())
    profile = json.loads(profile_path.read_text())
    if corruption == "missing_chain":
        del profile["stage_metadata"]
    elif corruption == "runtime_hash":
        profile["stage_runtime_table_sha256"][0] = "f" * 64
    elif corruption == "prior_hash":
        profile["stage_metadata"][0]["prior_stage_hashes"] = ["f" * 64]
    elif corruption == "experiment":
        profile["experiment_id"] = "f" * 64
    else:
        profile["stage_metadata"][0]["timing_used_for_selection"] = True
    profile_path.write_bytes(qualify._canonical_bytes(profile))
    with pytest.raises(ValueError, match="stage"):
        qualify._load_frozen_profile(
            profile_path, dataset=dataset,
            dataset_hash=qualify._candidate_set_sha256(dataset),
        )


@pytest.mark.parametrize("fit_splits", [None, [], ["validation"], ["training", "test"]])
def test_wave_qualifier_requires_training_only_profile(tmp_path, fit_splits):
    dataset_path, _, profile_path, _, _ = _evaluation_fixture(tmp_path, wave=True)
    dataset = json.loads(dataset_path.read_text())
    profile = json.loads(profile_path.read_text())
    profile["fit_splits"] = fit_splits
    profile_path.write_bytes(qualify._canonical_bytes(profile))
    with pytest.raises(ValueError, match="training only"):
        qualify._load_frozen_profile(
            profile_path, dataset=dataset,
            dataset_hash=qualify._candidate_set_sha256(dataset),
        )


def test_prepare_wave_evaluation_keeps_both_sdk_topologies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset_path, selection_path, profile_path, greedy, candidate = _evaluation_fixture(
        tmp_path, wave=True
    )
    monkeypatch.setattr(qualify, "_regenerate", lambda circuit, selected: (object(), {}))
    monkeypatch.setattr(qualify, "_verify_wave_plan", lambda dag, topology, topology_id: None)

    physical = qualify.prepare_config(
        dataset_path=dataset_path,
        selection_path=selection_path,
        output_path=tmp_path / "wave-physical.yml",
        mode="evaluation",
        profile_path=profile_path,
        split="validation",
    )
    assert physical["experiment_id"] == "upmem-final-system-path-evaluation-validation-v2"
    provenance = qualify._load(tmp_path / "wave-physical.yml.provenance.json")
    assert provenance["stage"] == {
        "stage_id": "validation", "round_ordinal": 0,
        "prior_stage_hashes": [], "timing_used_for_selection": False,
        "selection_profile_sha256": provenance["fitted_profile_sha256"],
    }
    assert physical["collection"]["warmup_blocks"] == 1
    assert physical["collection"]["measurement_blocks"] == 5
    assert set(physical["routes"]) == {"1dpu_t8", "4dpu_t8"}
    assert set(physical["plans"]) == {f"path_{greedy}", f"path_{candidate}"}
    assert all(
        set(item["route_ids"]) == {"1dpu_t8", "4dpu_t8"}
        for item in physical["matrix"]
    )

    sdk = qualify.prepare_config(
        dataset_path=dataset_path,
        selection_path=selection_path,
        output_path=tmp_path / "wave-sdk.yml",
        mode="evaluation",
        profile_path=profile_path,
        split="validation",
        execution_target="sdk",
    )
    assert sdk["experiment_id"] == "upmem-final-system-path-evaluation-validation-sdk-v2"
    assert sdk["collection"]["warmup_blocks"] == 0
    assert sdk["collection"]["measurement_blocks"] == 1
    assert set(sdk["routes"]) == {"1dpu_t8", "4dpu_t8"}
    assert all(
        route["executor"] == "upmem_sdk_simulator"
        and "rank_paths" not in route["options"]
        for route in sdk["routes"].values()
    )
    assert all(
        set(item["route_ids"]) == {"1dpu_t8", "4dpu_t8"}
        for item in sdk["matrix"]
    )


@pytest.mark.parametrize("target,warmups,measurements", [("physical", 1, 5), ("sdk", 0, 1)])
def test_confirmation_uses_only_frozen_measured_training_paths(tmp_path, monkeypatch, target, warmups, measurements):
    dataset_path, _, profile_path, greedy, selected = _evaluation_fixture(
        tmp_path, wave=True, split="training"
    )
    profile = json.loads(profile_path.read_text())
    cells = [f"held-out:{topology}" for topology in ("1dpu_t8", "4dpu_t8")]
    profile["selected_path_ids"] = dict.fromkeys(cells, selected)
    profile["candidate_speedups"] = {cell: {greedy: 1.0, selected: 2.0} for cell in cells}
    profile_path.write_bytes(qualify._canonical_bytes(profile))
    monkeypatch.setattr(qualify, "_regenerate", lambda circuit, candidate: (object(), {}))
    monkeypatch.setattr(qualify, "_verify_wave_plan", lambda *args: None)
    packet = qualify.prepare_config(
        dataset_path=dataset_path, profile_path=profile_path,
        output_path=tmp_path / "confirmation.yml", mode="confirmation",
        split="training", execution_target=target,
    )
    assert packet["collection"]["warmup_blocks"] == warmups
    provenance = qualify._load(tmp_path / "confirmation.yml.provenance.json")
    assert provenance["stage"] == {
        "stage_id": "development_confirmation", "round_ordinal": 0,
        "prior_stage_hashes": [], "timing_used_for_selection": False,
        "selection_profile_sha256": provenance["fitted_profile_sha256"],
    }
    assert packet["collection"]["measurement_blocks"] == measurements
    assert len(packet["matrix"]) == 2
    assert set(packet["plans"]) == {f"path_{greedy}", f"path_{selected}"}
    assert all(
        route["executor"] == ("upmem_sdk_simulator" if target == "sdk" else "upmem_physical")
        for route in packet["routes"].values()
    )
    with pytest.raises(ValueError, match="training split"):
        qualify.prepare_config(
            dataset_path=dataset_path, profile_path=profile_path,
            output_path=tmp_path / "invalid.yml", mode="confirmation", split="test",
        )
    profile["candidate_speedups"][cells[0]].pop(selected)
    profile_path.write_bytes(qualify._canonical_bytes(profile))
    with pytest.raises(ValueError, match="measured-pool score"):
        qualify.prepare_config(
            dataset_path=dataset_path, profile_path=profile_path,
            output_path=tmp_path / "unmeasured.yml", mode="confirmation", split="training",
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("execution_profile", "wrong_profile", "execution_profile"),
        ("primary_quantity", "steady_only_s", "primary_quantity"),
        ("timing_scope", "session_inclusive_v1", "timing_scope"),
        ("numeric_policy", "complex128", "numeric_policy"),
    ],
)
def test_wave_evaluation_rejects_profile_metadata_corruption(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: str,
    message: str,
) -> None:
    dataset_path, selection_path, profile_path, _, _ = _evaluation_fixture(
        tmp_path, wave=True
    )
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    profile[field] = value
    profile_path.write_bytes(qualify._canonical_bytes(profile))
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selection["fitted_profile_sha256"] = hashlib.sha256(
        qualify._canonical_bytes(profile)
    ).hexdigest()
    selection_path.write_bytes(qualify._canonical_bytes(selection))
    monkeypatch.setattr(qualify, "_regenerate", lambda circuit, selected: (object(), {}))

    with pytest.raises(ValueError, match=message):
        qualify.prepare_config(
            dataset_path=dataset_path,
            selection_path=selection_path,
            output_path=tmp_path / "corrupt-profile.yml",
            mode="evaluation",
            profile_path=profile_path,
            split="validation",
        )


def test_wave_evaluation_rejects_profile_weight_corruption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset_path, selection_path, profile_path, _, _ = _evaluation_fixture(
        tmp_path, wave=True
    )
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    profile["weights"]["B_host_dpu"] = 0.0
    profile["weights"]["B_mram_wram"] = 1.0
    profile_path.write_bytes(qualify._canonical_bytes(profile))
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selection["fitted_profile_sha256"] = hashlib.sha256(
        qualify._canonical_bytes(profile)
    ).hexdigest()
    selection_path.write_bytes(qualify._canonical_bytes(selection))
    monkeypatch.setattr(qualify, "_regenerate", lambda circuit, selected: (object(), {}))

    with pytest.raises(ValueError, match="inactive feature"):
        qualify.prepare_config(
            dataset_path=dataset_path,
            selection_path=selection_path,
            output_path=tmp_path / "corrupt-weights.yml",
            mode="evaluation",
            profile_path=profile_path,
            split="validation",
        )


def test_wave_evaluation_rejects_selection_contract_and_selected_id_corruption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset_path, selection_path, profile_path, greedy, _ = _evaluation_fixture(
        tmp_path, wave=True
    )
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selection["execution_contract"] = {
        **selection["execution_contract"],
        "request_transport": "packed_operation_v1",
    }
    selection_path.write_bytes(qualify._canonical_bytes(selection))
    monkeypatch.setattr(qualify, "_regenerate", lambda circuit, selected: (object(), {}))
    with pytest.raises(ValueError, match="selection has a mismatched wave execution_contract"):
        qualify.prepare_config(
            dataset_path=dataset_path,
            selection_path=selection_path,
            output_path=tmp_path / "wrong-contract.yml",
            mode="evaluation",
            profile_path=profile_path,
            split="validation",
        )

    selection["execution_contract"] = qualify.execution_contract(
        {"execution_profile": "kernel_schedule_system_v1"}
    )
    selection["selections"][0]["upmem_selected_path_id"] = greedy
    selection_path.write_bytes(qualify._canonical_bytes(selection))
    with pytest.raises(ValueError, match="not selected by frozen profile"):
        qualify.prepare_config(
            dataset_path=dataset_path,
            selection_path=selection_path,
            output_path=tmp_path / "wrong-selected-id.yml",
            mode="evaluation",
            profile_path=profile_path,
            split="validation",
        )


def test_wave_evaluation_rejects_raw_features_that_disagree_with_wave_facts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset_path, selection_path, profile_path, _, _ = _evaluation_fixture(
        tmp_path, wave=True
    )
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    topology = dataset["circuits"][0]["candidates"][0]["topologies"][0]
    topology["features"]["B_host_dpu"] += 1
    dataset_path.write_bytes(qualify._canonical_bytes(dataset))
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    profile["candidate_set_sha256"] = qualify._candidate_set_sha256(dataset)
    profile_path.write_bytes(qualify._canonical_bytes(profile))
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selection["candidate_set_sha256"] = qualify._candidate_set_sha256(dataset)
    selection["fitted_profile_sha256"] = hashlib.sha256(
        qualify._canonical_bytes(profile)
    ).hexdigest()
    selection_path.write_bytes(qualify._canonical_bytes(selection))
    monkeypatch.setattr(qualify, "_regenerate", lambda circuit, selected: (object(), {}))

    with pytest.raises(ValueError, match="raw features do not match wave facts"):
        qualify.prepare_config(
            dataset_path=dataset_path,
            selection_path=selection_path,
            output_path=tmp_path / "wrong-features.yml",
            mode="evaluation",
            profile_path=profile_path,
            split="validation",
        )


def test_wave_evaluation_rejects_memory_admission_over_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset_path, selection_path, profile_path, _, _ = _evaluation_fixture(
        tmp_path, wave=True
    )
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    topology = dataset["circuits"][0]["candidates"][0]["topologies"][0]
    over_budget = 512 * 1024 * 1024 + 1
    topology["memory_admission"]["declared_executor_memory_estimate_bytes"] = over_budget
    topology["memory_admission"]["required_bytes"] = over_budget
    topology["host_memory_estimate_bytes"] = over_budget
    dataset_path.write_bytes(qualify._canonical_bytes(dataset))
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    profile["candidate_set_sha256"] = qualify._candidate_set_sha256(dataset)
    profile_path.write_bytes(qualify._canonical_bytes(profile))
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selection["candidate_set_sha256"] = qualify._candidate_set_sha256(dataset)
    selection["fitted_profile_sha256"] = hashlib.sha256(
        qualify._canonical_bytes(profile)
    ).hexdigest()
    selection_path.write_bytes(qualify._canonical_bytes(selection))
    monkeypatch.setattr(qualify, "_regenerate", lambda circuit, selected: (object(), {}))

    with pytest.raises(ValueError, match="exceeds frozen executor budget"):
        qualify.prepare_config(
            dataset_path=dataset_path,
            selection_path=selection_path,
            output_path=tmp_path / "over-budget.yml",
            mode="evaluation",
            profile_path=profile_path,
            split="validation",
        )


def test_prepare_config_accepts_explicit_replacement_experiment_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset_path, selection_path, profile_path, _, _ = _evaluation_fixture(tmp_path)
    monkeypatch.setattr(qualify, "_regenerate", lambda circuit, selected: (object(), {}))
    config = qualify.prepare_config(
        dataset_path=dataset_path,
        selection_path=selection_path,
        output_path=tmp_path / "replacement.yml",
        mode="evaluation",
        profile_path=profile_path,
        split="validation",
        experiment_id="upmem-path-heuristic-generalization-validation-v1",
    )
    assert config["experiment_id"] == (
        "upmem-path-heuristic-generalization-validation-v1"
    )

    with pytest.raises(ValueError, match="experiment_id"):
        qualify.prepare_config(
            dataset_path=dataset_path,
            selection_path=selection_path,
            output_path=tmp_path / "invalid.yml",
            mode="evaluation",
            profile_path=profile_path,
            split="validation",
            experiment_id="",
        )


def test_prepare_evaluation_requires_explicit_frozen_profile(
    tmp_path: Path,
) -> None:
    dataset_path, selection_path, _, _, _ = _evaluation_fixture(tmp_path)

    with pytest.raises(ValueError, match="frozen profile"):
        qualify.prepare_config(
            dataset_path=dataset_path,
            selection_path=selection_path,
            output_path=tmp_path / "missing-profile.yml",
            mode="evaluation",
            split="validation",
        )


def test_prepare_evaluation_rejects_forged_upmem_selected_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset_path, selection_path, profile_path, greedy, _ = _evaluation_fixture(tmp_path)
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selection["selections"][0]["upmem_selected_path_id"] = greedy
    selection_path.write_bytes(qualify._canonical_bytes(selection))
    monkeypatch.setattr(qualify, "_regenerate", lambda circuit, selected: (object(), {}))

    with pytest.raises(ValueError, match="not selected by frozen profile"):
        qualify.prepare_config(
            dataset_path=dataset_path,
            selection_path=selection_path,
            output_path=tmp_path / "forged.yml",
            mode="evaluation",
            profile_path=profile_path,
            split="validation",
        )


@pytest.mark.parametrize(
    ("mutate_profile", "message"),
    [
        (
            lambda profile: profile.update({"candidate_set_sha256": "f" * 64}),
            "frozen profile candidate-set identity",
        ),
        (
            lambda profile: profile.update({"source_sha": "f" * 40}),
            "frozen profile source identity",
        ),
    ],
)
def test_prepare_evaluation_rejects_profile_dataset_identity_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutate_profile,
    message: str,
) -> None:
    dataset_path, selection_path, profile_path, _, _ = _evaluation_fixture(tmp_path)
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    mutate_profile(profile)
    profile_path.write_bytes(qualify._canonical_bytes(profile))
    monkeypatch.setattr(qualify, "_regenerate", lambda circuit, selected: (object(), {}))

    with pytest.raises(ValueError, match=message):
        qualify.prepare_config(
            dataset_path=dataset_path,
            selection_path=selection_path,
            output_path=tmp_path / "mismatched-profile.yml",
            mode="evaluation",
            profile_path=profile_path,
            split="validation",
        )


def test_prepare_evaluation_rejects_selection_profile_identity_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset_path, selection_path, profile_path, _, _ = _evaluation_fixture(tmp_path)
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selection["fitted_profile_sha256"] = "f" * 64
    selection_path.write_bytes(qualify._canonical_bytes(selection))
    monkeypatch.setattr(qualify, "_regenerate", lambda circuit, selected: (object(), {}))

    with pytest.raises(ValueError, match="frozen-profile identity"):
        qualify.prepare_config(
            dataset_path=dataset_path,
            selection_path=selection_path,
            output_path=tmp_path / "mismatched-selection.yml",
            mode="evaluation",
            profile_path=profile_path,
            split="validation",
        )


def test_prepare_evaluation_rejects_optional_workload_identity_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset_path, selection_path, profile_path, _, _ = _evaluation_fixture(tmp_path)
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    dataset["workload_manifest_sha256"] = "2" * 64
    dataset_path.write_bytes(qualify._canonical_bytes(dataset))
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    profile["candidate_set_sha256"] = qualify._candidate_set_sha256(dataset)
    profile["workload_manifest_sha256"] = dataset["workload_manifest_sha256"]
    profile_path.write_bytes(qualify._canonical_bytes(profile))
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selection["candidate_set_sha256"] = qualify._candidate_set_sha256(dataset)
    selection["fitted_profile_sha256"] = hashlib.sha256(
        qualify._canonical_bytes(profile)
    ).hexdigest()
    selection["workload_manifest_sha256"] = "f" * 64
    selection_path.write_bytes(qualify._canonical_bytes(selection))
    monkeypatch.setattr(qualify, "_regenerate", lambda circuit, selected: (object(), {}))

    with pytest.raises(ValueError, match="workload_manifest_sha256"):
        qualify.prepare_config(
            dataset_path=dataset_path,
            selection_path=selection_path,
            output_path=tmp_path / "mismatched-workload.yml",
            mode="evaluation",
            profile_path=profile_path,
            split="validation",
        )


def test_qualify_frozen_selection_replays_each_unique_candidate_deterministically(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset_path, selection_path, profile_path, greedy, candidate = _evaluation_fixture(tmp_path)
    calls: list[str] = []
    reference = np.asarray([1.0 + 2.0j, 3.0 + 4.0j], dtype=np.complex128)
    actual = np.asarray([1.0 + 2.0j, 3.0 + 4.0j], dtype=np.complex64)
    monkeypatch.setattr(qualify, "builtin_circuit", lambda name, params: object())
    monkeypatch.setattr(
        qualify, "make_simulation_job", lambda spec: spec
    )
    monkeypatch.setattr(
        qualify, "lower_tensor_network", lambda job: (object(), {"input": actual})
    )
    monkeypatch.setattr(
        qualify,
        "plan_opt_einsum",
        lambda network, optimize: ("reference-path", {}),
    )
    monkeypatch.setattr(
        qualify, "build_contraction_dag", lambda network, path: "reference-dag"
    )
    monkeypatch.setattr(
        qualify, "run_complex128_reference", lambda dag, inputs: reference
    )
    monkeypatch.setattr(
        qualify,
        "_regenerate",
        lambda circuit, selected: (
            calls.append(selected["candidate_path_id"])
            or (f"logical-{selected['candidate_path_id']}", {})
        ),
    )
    monkeypatch.setattr(
        qualify,
        "contraction_dag_hash",
        lambda dag: dag,
    )
    monkeypatch.setattr(
        qualify,
        "run_cpu_once",
        lambda dag, inputs, numeric_policy: SimpleNamespace(output=actual),
    )
    first_output = tmp_path / "first.json"
    first = qualify.qualify_frozen_selection(
        dataset_path,
        selection_path,
        first_output,
        split="validation",
        profile_path=profile_path,
    )
    second_output = tmp_path / "second.json"
    second = qualify.qualify_frozen_selection(
        dataset_path,
        selection_path,
        second_output,
        split="validation",
        profile_path=profile_path,
    )

    assert first == second
    assert first["selection_contract_passed"] is True
    assert first["qualified_candidate_count"] == 2
    assert first["all_passed"] is True
    assert calls == [greedy, candidate, greedy, candidate]
    assert [row["candidate_path_id"] for row in first["candidates"]] == [
        greedy,
        candidate,
    ]
    assert first["candidates"][0]["roles"] == ["greedy"]
    assert first["candidates"][1]["roles"] == [
        "minimum_flops",
        "upmem_selected",
    ]
    assert first["candidates"][0]["topology_ids"] == ["1dpu_t8", "4dpu_t8"]
    assert first["candidates"][0]["errors"]["max_absolute_error"] == 0.0
    assert len(first["candidates"][0]["output_sha256"]) == 64
    assert len(first["candidates"][0]["reference_output_sha256"]) == 64
    assert first_output.read_bytes() == second_output.read_bytes()


def test_qualify_frozen_selection_records_cpu_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset_path, selection_path, profile_path, _, candidate = _evaluation_fixture(tmp_path)
    reference = np.asarray([1.0 + 0.0j], dtype=np.complex128)
    monkeypatch.setattr(qualify, "builtin_circuit", lambda name, params: object())
    monkeypatch.setattr(qualify, "make_simulation_job", lambda spec: spec)
    monkeypatch.setattr(
        qualify, "lower_tensor_network", lambda job: (object(), {"input": reference})
    )
    monkeypatch.setattr(
        qualify, "plan_opt_einsum", lambda network, optimize: ("path", {})
    )
    monkeypatch.setattr(
        qualify, "build_contraction_dag", lambda network, path: "dag"
    )
    monkeypatch.setattr(
        qualify, "run_complex128_reference", lambda dag, inputs: reference
    )
    monkeypatch.setattr(
        qualify,
        "_regenerate",
        lambda circuit, selected: (selected["candidate_path_id"], {}),
    )
    monkeypatch.setattr(qualify, "contraction_dag_hash", lambda dag: dag)
    monkeypatch.setattr(
        qualify,
        "run_cpu_once",
        lambda dag, inputs, numeric_policy: SimpleNamespace(
            output=np.asarray([9.0 + 0.0j], dtype=np.complex64)
        ),
    )

    output = tmp_path / "failure.json"
    result = qualify.qualify_frozen_selection(
        dataset_path,
        selection_path,
        output,
        split="validation",
        profile_path=profile_path,
    )

    assert result["all_passed"] is False
    assert all(not row["passed"] for row in result["candidates"])
    assert all(row["error"] is None for row in result["candidates"])
    assert result["candidates"][0]["max_absolute_error"] == 8.0
    assert json.loads(output.read_text(encoding="utf-8")) == result


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda selection: selection.update({"candidate_set_sha256": "f" * 64}), "candidate-set identity"),
        (lambda selection: selection.update({"split": "test"}), "selection split"),
        (
            lambda selection: selection.update({"timing_used_for_selection": True}),
            "timing-independent",
        ),
        (
            lambda selection: selection["selections"][0].update(
                {"greedy_path_id": "b" * 64}
            ),
            "not deterministic",
        ),
    ],
)
def test_prepare_evaluation_rejects_frozen_selection_contract_violations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    change,
    message: str,
) -> None:
    dataset_path, selection_path, profile_path, _, _ = _evaluation_fixture(tmp_path)
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    change(selection)
    selection_path.write_bytes(qualify._canonical_bytes(selection))
    monkeypatch.setattr(qualify, "_regenerate", lambda circuit, selected: (object(), {}))
    with pytest.raises(ValueError, match=message):
        qualify.prepare_config(
            dataset_path=dataset_path,
            selection_path=selection_path,
            output_path=tmp_path / "validation.yml",
            mode="evaluation",
            profile_path=profile_path,
            split="validation",
        )


@pytest.mark.parametrize("bad_value", [None, "false"])
def test_prepare_wave_evaluation_rejects_missing_or_nonbool_collection_admission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bad_value: object
) -> None:
    dataset_path, selection_path, profile_path, _, _ = _evaluation_fixture(
        tmp_path, wave=True
    )
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    admission = dataset["circuits"][0]["candidates"][1]["topologies"][1][
        "resource_admission"
    ]
    if bad_value is None:
        del admission["collection_resource_admission_passed"]
    else:
        admission["collection_resource_admission_passed"] = bad_value
    dataset_path.write_bytes(qualify._canonical_bytes(dataset))
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    profile["candidate_set_sha256"] = qualify._candidate_set_sha256(dataset)
    profile_path.write_bytes(qualify._canonical_bytes(profile))
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selection["candidate_set_sha256"] = qualify._candidate_set_sha256(dataset)
    selection["fitted_profile_sha256"] = hashlib.sha256(
        qualify._canonical_bytes(profile)
    ).hexdigest()
    selection_path.write_bytes(qualify._canonical_bytes(selection))
    monkeypatch.setattr(qualify, "_regenerate", lambda circuit, selected: (object(), {}))
    with pytest.raises(ValueError, match="resource admission|boolean"):
        qualify.prepare_config(
            dataset_path=dataset_path,
            selection_path=selection_path,
            output_path=tmp_path / "validation.yml",
            mode="evaluation",
            profile_path=profile_path,
            split="validation",
        )


def test_prepare_legacy_evaluation_still_rejects_failed_collection_admission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset_path, selection_path, profile_path, _, _ = _evaluation_fixture(tmp_path)
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    dataset["circuits"][0]["candidates"][1]["topologies"][1]["resource_admission"][
        "collection_resource_admission_passed"
    ] = False
    dataset_path.write_bytes(qualify._canonical_bytes(dataset))
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    profile["candidate_set_sha256"] = qualify._candidate_set_sha256(dataset)
    profile_path.write_bytes(qualify._canonical_bytes(profile))
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selection["candidate_set_sha256"] = qualify._candidate_set_sha256(dataset)
    selection["fitted_profile_sha256"] = hashlib.sha256(
        qualify._canonical_bytes(profile)
    ).hexdigest()
    selection["selections"][1]["minimum_flops_path_id"] = "a" * 64
    selection_path.write_bytes(qualify._canonical_bytes(selection))
    monkeypatch.setattr(qualify, "_regenerate", lambda circuit, selected: (object(), {}))
    with pytest.raises(ValueError, match="resource admission"):
        qualify.prepare_config(
            dataset_path=dataset_path,
            selection_path=selection_path,
            output_path=tmp_path / "legacy-validation.yml",
            mode="evaluation",
            profile_path=profile_path,
            split="validation",
        )


def test_prepare_wave_diagnostic_accepts_failed_collection_admission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset_path, selection_path, profile_path, greedy, selected = _evaluation_fixture(
        tmp_path, wave=True
    )
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    candidate = dataset["circuits"][0]["candidates"][1]
    for topology in candidate["topologies"]:
        admission = topology["resource_admission"]
        admission["collection_resource_admission_passed"] = False
        if topology["topology"]["dpu_count"] == 1:
            admission["tasklet_row_sufficiency_passed"] = False
            admission["dominant_work_wave_tasklet_row_sufficiency_passed"] = False
        else:
            admission["dominant_work_wave_populated_dpu_slots"] = 3
    dataset_path.write_bytes(qualify._canonical_bytes(dataset))
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    profile["candidate_set_sha256"] = qualify._candidate_set_sha256(dataset)
    profile_path.write_bytes(qualify._canonical_bytes(profile))
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selection["candidate_set_sha256"] = qualify._candidate_set_sha256(dataset)
    selection["fitted_profile_sha256"] = hashlib.sha256(
        qualify._canonical_bytes(profile)
    ).hexdigest()
    selection_path.write_bytes(qualify._canonical_bytes(selection))
    monkeypatch.setattr(qualify, "_regenerate", lambda circuit, selected: (object(), {}))
    monkeypatch.setattr(qualify, "_verify_wave_plan", lambda *args: None)

    config = qualify.prepare_config(
        dataset_path=dataset_path,
        selection_path=selection_path,
        output_path=tmp_path / "diagnostic.yml",
        mode="evaluation",
        profile_path=profile_path,
        split="validation",
    )

    assert config["collection"]["claim_policy"] == "diagnostic_v1"
    assert set(config["plans"]) == {f"path_{greedy}", f"path_{selected}"}
    assert all(
        topology["resource_admission"]["collection_resource_admission_passed"] is False
        for topology in candidate["topologies"]
    )


@pytest.mark.parametrize("prepared_waves", [False, True])
@pytest.mark.parametrize("wrong_split", [None, "validation", "test"])
def test_prepare_calibration_config_preserves_candidate_seed_and_collection(
    tmp_path: Path, monkeypatch, prepared_waves: bool, wrong_split: str | None
) -> None:
    greedy = "a" * 64
    candidate = "b" * 64
    dataset = {
        "source_sha": "1" * 40,
        "preregistration_sha256": "2" * 64,
        "circuits": [
            {
                "circuit_id": "train",
                "split": "training",
                "circuit": {"kind": "builtin", "name": "bell_2q", "parameters": {}},
                "candidates": [
                    _candidate(greedy, greedy=True, seed=None, host=100),
                    _candidate(candidate, greedy=False, seed=20260903, host=50),
                ],
            }
        ]
    }
    if prepared_waves:
        _mark_wave_dataset(dataset)
        metadata = {
            "execution_profile": dataset["execution_profile"],
            "execution_contract": dataset["execution_contract"],
            "score_id": dataset["score_id"],
        }
        contract = dataset["execution_contract"]
    if wrong_split is not None:
        dataset["circuits"][0]["split"] = wrong_split
    calibration = {
        "source_sha": dataset["source_sha"],
        "candidate_set_sha256": qualify._candidate_set_sha256(dataset),
        "cells": [
            {
                "cell_id": "train:1dpu_t8",
                "circuit_id": "train",
                "topology_id": "1dpu_t8",
                "candidate_path_ids": [greedy, candidate],
            }
        ]
    }
    rankings = tmp_path / "rankings.csv"
    if prepared_waves:
        calibration.update(metadata)
    rankings.write_text(
        "circuit_id,topology_id,equal_weight_rank,candidate_path_id\n"
        f"train,1dpu_t8,1,{candidate}\n"
        f"train,4dpu_t8,1,{candidate}\n",
        encoding="utf-8",
    )
    dataset_path = tmp_path / "dataset.json"
    calibration_path = tmp_path / "calibration.json"
    dataset_path.write_text(json.dumps(dataset), encoding="utf-8")
    calibration_path.write_text(json.dumps(calibration), encoding="utf-8")
    monkeypatch.setattr(qualify, "_regenerate", lambda circuit, selected: (object(), {}))
    output = tmp_path / "campaign.yml"
    verified = []
    monkeypatch.setattr(
        qualify, "_verify_wave_plan",
        lambda dag, topology, topology_id: verified.append(topology_id),
    )
    if prepared_waves and wrong_split is not None:
        for mode in ("calibration", "sdk"):
            with pytest.raises(ValueError, match="training only"):
                qualify.prepare_config(
                    dataset_path=dataset_path, calibration_path=calibration_path,
                    output_path=output, mode=mode,
                )
            assert not output.exists()
        return
    config = qualify.prepare_config(
        dataset_path=dataset_path,
        calibration_path=calibration_path,
        rankings_path=rankings,
        output_path=output,
        mode="calibration",
    )
    assert config["collection"]["warmup_blocks"] == 1
    assert config["collection"]["measurement_blocks"] == 3
    assert config["plans"][f"path_{candidate}"]["planner"] == {
        "engine": "cotengra",
        "mode": "greedy",
        "max_repeats": 1,
        "seed": 20260903,
    }
    loaded = yaml.safe_load(output.read_text(encoding="utf-8"))
    assert loaded["matrix"][0]["route_ids"] == ["1dpu_t8"]
    assert loaded["routes"]["1dpu_t8"]["options"]["rank_paths"] == [
        "/dev/dpu_rank1"
    ]
    provenance = json.loads(
        output.with_suffix(".yml.provenance.json").read_text(encoding="utf-8")
    )
    assert provenance["candidate_set_sha256"] == qualify._candidate_set_sha256(dataset)
    if prepared_waves:
        assert verified == ["1dpu_t8", "1dpu_t8"]
        assert config["experiment_id"] == "upmem-final-system-path-calibration-v2"
        assert provenance["execution_contract"] == contract
        assert config["routes"]["1dpu_t8"]["options"]["schedule_policy"] == "static_dag_waves_v1"
        assert config["routes"]["4dpu_t8"]["options"]["request_transport"] == "packed_wave_v1"
        sdk = qualify.prepare_config(
            dataset_path=dataset_path,
            rankings_path=rankings,
            output_path=tmp_path / "sdk.yml",
            mode="sdk",
        )
        assert {route for item in sdk["matrix"] for route in item["route_ids"]} == {
            "1dpu_t8", "4dpu_t8"
        }
        assert all(
            route["executor"] == "upmem_sdk_simulator" for route in sdk["routes"].values()
        )
        assert sdk["experiment_id"] == "upmem-final-system-path-sdk-v2"
        calibration["execution_contract"] = {
            **contract, "request_transport": "packed_operation_v1"
        }
        calibration_path.write_text(json.dumps(calibration), encoding="utf-8")
        rejected = tmp_path / "wrong-policy.yml"
        with pytest.raises(ValueError, match="contract"):
            qualify.prepare_config(
                dataset_path=dataset_path, calibration_path=calibration_path,
                output_path=rejected, mode="calibration",
            )
        assert not rejected.exists()


@pytest.mark.parametrize("defect", ["missing_contract", "wrong_transport", "wrong_score"])
def test_prepare_rejects_invalid_wave_identity_before_writing(tmp_path: Path, defect: str):
    dataset = {"execution_profile": "kernel_schedule_system_v1"}
    contract = qualify.execution_contract(dataset)
    dataset.update(execution_contract=contract, score_id=contract["cost_model_id"])
    if defect == "missing_contract":
        del dataset["execution_contract"]
    elif defect == "wrong_transport":
        dataset["execution_contract"]["request_transport"] = "packed_operation_v1"
    else:
        dataset["score_id"] = "upmem_slr_cost_v1"
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(json.dumps(dataset), encoding="utf-8")
    output = tmp_path / "campaign.yml"
    with pytest.raises(ValueError, match="contract|score"):
        qualify.prepare_config(dataset_path=dataset_path, output_path=output, mode="calibration")
    assert not output.exists()


@pytest.mark.parametrize("dpus", [1])
def test_wave_plan_identity_is_recomputed_with_real_lowering(dpus):
    from quantum_bench.upmem.plan import UpmemTopology, physical_plan_id, plan_upmem

    spec = qualify.builtin_circuit("bell_2q", {})
    network, _ = qualify.lower_tensor_network(qualify.make_simulation_job(spec))
    path, _ = qualify.plan_opt_einsum(network, optimize="greedy")
    dag = qualify.build_contraction_dag(network, path)
    resources = {"dpu_count": dpus, "rank_count": 1, "tasklets_per_dpu": 8}
    plan = plan_upmem(
        dag, numeric_policy=qualify.FLOAT32, topology=UpmemTopology(**resources),
        schedule_policy="static_dag_waves_v1",
    )
    record = {
        "topology": resources,
        "physical_plan_id": physical_plan_id(plan),
        "resource_admission": qualify.collection_resource_admission(plan),
    }
    assert record["resource_admission"]["collection_resource_admission_passed"] is False
    qualify._verify_wave_plan(dag, record, f"{dpus}dpu_t8")
    record["physical_plan_id"] = "0" * 64
    with pytest.raises(ValueError, match="physical-plan"):
        qualify._verify_wave_plan(dag, record, f"{dpus}dpu_t8")
    record["topology"] = {**resources, "tasklets_per_dpu": 4}
    with pytest.raises(ValueError, match="topology"):
        qualify._verify_wave_plan(dag, record, f"{dpus}dpu_t8")


def test_wave_plan_rejects_identity_correct_forged_bell_4d_for_hard_coverage():
    from quantum_bench.upmem.plan import UpmemTopology, physical_plan_id, plan_upmem

    spec = qualify.builtin_circuit("bell_2q", {})
    network, _ = qualify.lower_tensor_network(qualify.make_simulation_job(spec))
    path, _ = qualify.plan_opt_einsum(network, optimize="greedy")
    dag = qualify.build_contraction_dag(network, path)
    resources = {"dpu_count": 4, "rank_count": 1, "tasklets_per_dpu": 8}
    plan = plan_upmem(
        dag, numeric_policy=qualify.FLOAT32, topology=UpmemTopology(**resources),
        schedule_policy="static_dag_waves_v1",
    )
    record = {
        "feasible": True,
        "topology": resources,
        "logical_plan_id": plan.logical_plan_id,
        "physical_plan_id": physical_plan_id(plan),
        "resource_admission": qualify.collection_resource_admission(plan),
    }

    with pytest.raises(
        ValueError, match="planned_execution_resource_admission_failed"
    ):
        qualify._verify_wave_plan(dag, record, "4dpu_t8")


def test_prepare_rejects_candidate_infeasible_for_selected_topology(
    tmp_path: Path, monkeypatch
) -> None:
    identifier = "a" * 64
    candidate = _candidate(identifier, greedy=True, seed=None, host=100)
    candidate["topologies"][1]["feasible"] = False
    dataset = {
        "source_sha": "1" * 40,
        "preregistration_sha256": "2" * 64,
        "circuits": [{
            "circuit_id": "train",
            "split": "training",
            "circuit": {"kind": "builtin", "name": "bell_2q", "parameters": {}},
            "candidates": [candidate],
        }],
    }
    calibration = {
        "source_sha": dataset["source_sha"],
        "candidate_set_sha256": qualify._candidate_set_sha256(dataset),
        "cells": [{
            "cell_id": "train:4dpu_t8",
            "circuit_id": "train",
            "topology_id": "4dpu_t8",
            "candidate_path_ids": [identifier],
        }],
    }
    dataset_path = tmp_path / "dataset.json"
    calibration_path = tmp_path / "calibration.json"
    rankings = tmp_path / "rankings.csv"
    dataset_path.write_text(json.dumps(dataset), encoding="utf-8")
    calibration_path.write_text(json.dumps(calibration), encoding="utf-8")
    rankings.write_text(
        "circuit_id,topology_id,equal_weight_rank,candidate_path_id\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(qualify, "_regenerate", lambda circuit, selected: (object(), {}))
    try:
        qualify.prepare_config(
            dataset_path=dataset_path,
            calibration_path=calibration_path,
            rankings_path=rankings,
            output_path=tmp_path / "campaign.yml",
            mode="calibration",
        )
    except ValueError as exc:
        assert "infeasible" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("infeasible topology candidate was accepted")


def test_planner_config_rejects_unknown_candidate_source() -> None:
    candidate = {"source_kind": "timing_selected"}
    try:
        qualify._planner_config(candidate)
    except ValueError as exc:
        assert "unsupported candidate source" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("unknown candidate source was accepted")
