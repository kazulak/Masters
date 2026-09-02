from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest
import yaml


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "qualify_upmem_path_candidates.py"
SPEC = importlib.util.spec_from_file_location("qualify_upmem_path_candidates", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
qualify = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = qualify
SPEC.loader.exec_module(qualify)


def _candidate(identifier: str, *, greedy: bool, seed: int | None, host: int) -> dict[str, object]:
    topologies = []
    for topology in ("1dpu_t8", "4dpu_t8"):
        topologies.append(
            {
                "topology_id": topology,
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


def _evaluation_fixture(
    tmp_path: Path, *, split: str = "validation"
) -> tuple[Path, Path, str, str]:
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
    selection = {
        "schema_version": "upmem_path_frozen_selection_v1",
        "source_sha": dataset["source_sha"],
        "candidate_set_sha256": qualify._candidate_set_sha256(dataset),
        "split": split,
        "timing_used_for_selection": False,
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
    dataset_path = tmp_path / "dataset.json"
    selection_path = tmp_path / "selection.json"
    dataset_path.write_bytes(qualify._canonical_bytes(dataset))
    selection_path.write_bytes(qualify._canonical_bytes(selection))
    return dataset_path, selection_path, greedy, candidate


@pytest.mark.parametrize("split", ("validation", "test"))
def test_prepare_evaluation_config_uses_frozen_selection_once_per_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, split: str
) -> None:
    dataset_path, selection_path, greedy, candidate = _evaluation_fixture(
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
    assert len(provenance["selected"]) == 4
    assert provenance["candidate_set_sha256"] == qualify._candidate_set_sha256(
        json.loads(dataset_path.read_text(encoding="utf-8"))
    )


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
    dataset_path, selection_path, _, _ = _evaluation_fixture(tmp_path)
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
            split="validation",
        )


def test_prepare_evaluation_rejects_candidate_without_resource_admission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset_path, selection_path, _, _ = _evaluation_fixture(tmp_path)
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    dataset["circuits"][0]["candidates"][1]["topologies"][1]["resource_admission"][
        "collection_resource_admission_passed"
    ] = False
    dataset_path.write_bytes(qualify._canonical_bytes(dataset))
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selection["candidate_set_sha256"] = qualify._candidate_set_sha256(dataset)
    selection["selections"][1]["minimum_flops_path_id"] = "a" * 64
    selection_path.write_bytes(qualify._canonical_bytes(selection))
    monkeypatch.setattr(qualify, "_regenerate", lambda circuit, selected: (object(), {}))
    with pytest.raises(ValueError, match="resource admission"):
        qualify.prepare_config(
            dataset_path=dataset_path,
            selection_path=selection_path,
            output_path=tmp_path / "validation.yml",
            mode="evaluation",
            split="validation",
        )


def test_prepare_calibration_config_preserves_candidate_seed_and_collection(tmp_path: Path, monkeypatch) -> None:
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
