from __future__ import annotations

from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys

from quantum_bench.circuits import manifest, quest_compatible_circuit
from quantum_bench.evidence import problem_id
from quantum_bench.model import make_simulation_job, validate_circuit_spec


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "upmem_pimutation_full6_workload_v1.json"
STUDY_CONFIG_PATH = ROOT / "configs" / "upmem_pimutation_full6_path_study_v1.json"
GENERATOR_SHA256 = (
    "9092807e727911689a4e261a6929eb73d38fd00f374c113deec57404850a036e"
)

SCRIPT_SPEC = importlib.util.spec_from_file_location(
    "pimutation_path_study_loader",
    ROOT / "scripts" / "upmem_path_heuristic.py",
)
assert SCRIPT_SPEC is not None and SCRIPT_SPEC.loader is not None
path_study = importlib.util.module_from_spec(SCRIPT_SPEC)
sys.modules[SCRIPT_SPEC.name] = path_study
SCRIPT_SPEC.loader.exec_module(path_study)

EXPECTED = {
    ("training", "QRNG"): {
        "parameters": {"n_qubits": 16, "repeat_layers": 1},
        "gate_counts": {"1q": 16, "2q": 0, "total": 16, "by_gate": {"h": 16}},
        "problem_id": "a6045ec53af50fc5f8824a569b60bab08db30c38359d182f807eb0ac140b9e9f",
    },
    ("training", "BV"): {
        "parameters": {"n_qubits": 16, "repeat_layers": 1},
        "gate_counts": {
            "1q": 32,
            "2q": 15,
            "total": 47,
            "by_gate": {"cx": 15, "h": 31, "x": 1},
        },
        "problem_id": "9f765c1150b5971420ec5771f639009649d48011264c771eebd22cd2366b23f2",
    },
    ("training", "XOR"): {
        "parameters": {"n_qubits": 16, "repeat_layers": 1},
        "gate_counts": {"1q": 0, "2q": 15, "total": 15, "by_gate": {"cx": 15}},
        "problem_id": "9740a124fa210e68b259ee8204318518ea5e12384ef1a4b710acead6ad61b1b6",
    },
    ("training", "BB84"): {
        "parameters": {"n_qubits": 16, "repeat_layers": 1},
        "gate_counts": {
            "1q": 32,
            "2q": 0,
            "total": 32,
            "by_gate": {"h": 16, "x": 16},
        },
        "problem_id": "029cbcc1a5f5cc921538191f1ec0eab0eaf283ec8855b34fd3c4b6ad443e7c28",
    },
    ("training", "EDC"): {
        "parameters": {"n_qubits": 16, "repeat_layers": 1},
        "gate_counts": {
            "1q": 32,
            "2q": 30,
            "total": 62,
            "by_gate": {"cx": 30, "h": 16, "x": 16},
        },
        "problem_id": "2d56402bfaa7d6d4de84ecfec2fafd0d0f3cc56bd21b2131f33d2ee2220ce47a",
    },
    ("training", "HiddenSubgroup"): {
        "parameters": {
            "allocated_qubits": 16,
            "logical_qubits": 8,
            "depth": 1,
            "repeat_layers": 1,
        },
        "gate_counts": {
            "1q": 48,
            "2q": 16,
            "total": 64,
            "by_gate": {"cx": 16, "h": 32, "x": 16},
        },
        "problem_id": "78795d600b4e1cf7418708f241df6c7ede9581291b574b9b703adbe0058aeab0",
    },
    ("test", "QRNG"): {
        "parameters": {"n_qubits": 18, "repeat_layers": 2},
        "gate_counts": {"1q": 36, "2q": 0, "total": 36, "by_gate": {"h": 36}},
        "problem_id": "cb7b21d5ad7a3100d31166cc0e98d0e933d2e4bfcd690242a3c4377f5f8c6a71",
    },
    ("test", "BV"): {
        "parameters": {"n_qubits": 18, "repeat_layers": 2},
        "gate_counts": {
            "1q": 72,
            "2q": 34,
            "total": 106,
            "by_gate": {"cx": 34, "h": 70, "x": 2},
        },
        "problem_id": "770904e0258a3cd71523b2494d3076c5fb2fa095612aaf6f242b738c970021db",
    },
    ("test", "XOR"): {
        "parameters": {"n_qubits": 18, "repeat_layers": 2},
        "gate_counts": {"1q": 0, "2q": 34, "total": 34, "by_gate": {"cx": 34}},
        "problem_id": "2c6179f0609fdbc46b27343eb4c13f2887af2d74ffa9bbe699cb2721d5fd5e59",
    },
    ("test", "BB84"): {
        "parameters": {"n_qubits": 18, "repeat_layers": 2},
        "gate_counts": {
            "1q": 72,
            "2q": 0,
            "total": 72,
            "by_gate": {"h": 36, "x": 36},
        },
        "problem_id": "8874950f694b39ce3706b11f474a4182f9132767e7da5d5c804bbcbd817e79fe",
    },
    ("test", "EDC"): {
        "parameters": {"n_qubits": 15, "repeat_layers": 1},
        "gate_counts": {
            "1q": 30,
            "2q": 28,
            "total": 58,
            "by_gate": {"cx": 28, "h": 15, "x": 15},
        },
        "problem_id": "fea43ee6674bb142e28992be699105db85bb67ca942fff806024ef6ef17f332d",
    },
    ("test", "HiddenSubgroup"): {
        "parameters": {
            "allocated_qubits": 18,
            "logical_qubits": 9,
            "depth": 1,
            "repeat_layers": 2,
        },
        "gate_counts": {
            "1q": 108,
            "2q": 36,
            "total": 144,
            "by_gate": {"cx": 36, "h": 72, "x": 36},
        },
        "problem_id": "d5cc4204743731bf96dbb3f14963366ec0e25457206479d50442afc72ead5003",
    },
}


def test_finalized_manifest_covers_six_families_in_both_splits() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    assert config["status"] == "workload_definitions_frozen"
    assert config["decision_status"] == "workload_definitions_frozen"
    instances = config["instances"]
    assert len(instances) == 12
    assert {(item["split"], item["family"]) for item in instances} == set(EXPECTED)
    assert {
        item["family"] for item in instances if item["split"] == "training"
    } == {
        "QRNG",
        "BV",
        "XOR",
        "BB84",
        "EDC",
        "HiddenSubgroup",
    }
    assert {
        item["family"] for item in instances if item["split"] == "test"
    } == {
        "QRNG",
        "BV",
        "XOR",
        "BB84",
        "EDC",
        "HiddenSubgroup",
    }
    assert len({item["instance_id"] for item in instances}) == 12
    assert len(
        {
            item["canonical_circuit"]["operation_identity"]["problem_id"]
            for item in instances
        }
    ) == 12


def test_finalized_manifest_matches_existing_quest_core_and_hash_metadata() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    instances = config["instances"]
    domains = {
        item["canonical_circuit"]["operation_identity"]["domain"]
        for item in instances
    }
    assert len(domains) == 1
    assert all(
        re.fullmatch(r"[0-9a-f]{64}", item["canonical_circuit"]["operation_identity"]["problem_id"])
        for item in instances
    )

    for item in instances:
        expected = EXPECTED[(item["split"], item["family"])]
        definition = item["circuit"]
        assert definition["kind"] == "quest_compatible"
        assert definition["parameters"] == expected["parameters"]
        circuit = quest_compatible_circuit(definition["name"], definition["parameters"])
        validate_circuit_spec(circuit)
        facts = manifest(circuit)
        expected_counts = {
            **facts["gate_counts"],
            "by_gate": dict(sorted(Counter(op.gate for op in circuit.operations).items())),
        }
        canonical = item["canonical_circuit"]
        assert canonical["n_qubits"] == facts["n_qubits"]
        assert canonical["depth_proxy"] == facts["depth_proxy"]
        assert canonical["gate_set"] == facts["gate_set"]
        assert canonical["gate_counts"] == expected_counts == expected["gate_counts"]
        computed_id = problem_id(make_simulation_job(circuit))
        assert canonical["operation_identity"]["problem_id"] == computed_id
        assert canonical["operation_identity"]["problem_id"] == expected["problem_id"]

    generator = config["provenance"]["generator_source"]
    generator_path = ROOT.parent.parent / generator["path"]
    assert generator["sha256"] == GENERATOR_SHA256
    assert hashlib.sha256(generator_path.read_bytes()).hexdigest() == GENERATOR_SHA256

    for evidence in config["provenance"]["evidence_files"]:
        assert isinstance(evidence["path"], str) and evidence["path"]
        assert re.fullmatch(r"[0-9a-f]{64}", evidence["sha256"])


def test_study_loader_matches_manifest_and_declared_budget() -> None:
    manifest_config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    study = path_study.load_config(STUDY_CONFIG_PATH)

    assert study["status"] == manifest_config["status"] == "workload_definitions_frozen"
    assert study["workload_manifest"] == {
        "path": "thesis/implementation/configs/upmem_pimutation_full6_workload_v1.json",
        "workload_id": manifest_config["workload_id"],
        "status": manifest_config["status"],
    }
    assert set(study["execution_contract"]) == set(path_study.WAVE_EXECUTION_CONTRACT)
    assert study["execution_contract"] == path_study.WAVE_EXECUTION_CONTRACT
    assert study["rank_count"] == 1
    assert {topology["rank_count"] for topology in study["topologies"]} == {1}
    assert study["host_memory_admission_bytes"] == 536870912
    assert study["execution_policy"]["host_memory_reserve_bytes"] == 0
    assert study["candidate_generation"][
        "maximum_semantic_identity_expansion_units"
    ] == 1_000_000
    assert "_wave_equal_weights" in study["candidate_pool"]["initial_selection_score"]
    assert "_wave_score" in study["candidate_pool"]["initial_selection_score"]
    assert study["adaptive_search"]["proposal_model_form"] == "six_term"
    expected_circuits = [
        {
            "circuit_id": item["instance_id"],
            "split": item["split"],
            "circuit": item["circuit"],
        }
        for item in manifest_config["instances"]
    ]
    assert study["circuits"] == expected_circuits
    assert {
        (item["split"], item["family"])
        for item in study["declared_workload"]
    } == set(EXPECTED)

    budget = study["budget"]
    assert budget["attempt_ceiling"] == 792
    assert budget["historical_development_attempts"] == 92
    assert budget["aggregate_p6_attempt_ceiling_including_historical_development"] == 884
    assert budget["maximum_adaptive_rounds"] == 1
    assert budget["no_separate_validation_physical_stage"] is True
    assert sum(stage["attempts"] for stage in budget["stages"]) == 792
    assert [stage["attempts"] for stage in budget["stages"]] == [288, 144, 144, 216]
    assert budget["stages"][1]["maximum_rounds"] == 1

    rejected = manifest_config["decision"]["rejected_uniform_proposals"]
    assert [
        item["semantic_identity_expansion_units_lower_bound"] for item in rejected
    ] == [1_048_852, 1_048_820]
    assert all(
        item["preregistered_bound"]
        == study["candidate_generation"]["maximum_semantic_identity_expansion_units"]
        and item["semantic_identity_expansion_units_lower_bound"]
        > item["preregistered_bound"]
        and item["status"] == "rejected_before_timing"
        and item["physical_attempts"] == 0
        and "early-cutoff lower bound" in item["measurement_semantics"]
        for item in rejected
    )


def test_historical_development_data_is_excluded_from_final_fit() -> None:
    manifest_config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    study = path_study.load_config(STUDY_CONFIG_PATH)

    prior = manifest_config["preserved_prior_development"]
    assert prior["p6_92_attempt_record"] == "thesis/implementation/runs/p6-preparation/"
    assert prior["final_fit_eligible"] is False
    assert prior["raw_records_modified"] is False
    assert prior["old_physical_budget_reused"] is False
    evidence = study["evidence_policy"]
    assert evidence["exclude_historical_92_from_final_fit"] is True
    assert evidence["mix_prior_pilot_records_into_new_fit"] is False
    assert evidence["reuse_prior_pilot_weights"] is False
    assert evidence["reuse_prior_pilot_profile"] is False
    assert all(
        item["historical_optimization_status"] == "not_admitted_from_prior_fit"
        for item in study["declared_workload"]
    )
