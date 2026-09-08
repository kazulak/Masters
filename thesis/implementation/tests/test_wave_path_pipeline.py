"""Real software handoff; this never invokes an SDK or physical session."""

import importlib
import importlib.util
import hashlib
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "wave_pipeline_qualifier", ROOT / "scripts" / "qualify_upmem_path_candidates.py"
)
assert SPEC is not None and SPEC.loader is not None
qualifier = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = qualifier
SPEC.loader.exec_module(qualifier)
generator = importlib.import_module("upmem_path_heuristic")


def test_real_wave_candidate_generation_to_cpu_and_execution_packet(tmp_path, monkeypatch):
    config = generator.load_config(ROOT / "configs" / "upmem_final_system_path_study_v2.json")
    config["circuits"] = [{
        "circuit_id": "stress14_pipeline_fixture",
        "split": "training",
        "circuit": {
            "kind": "builtin", "name": "quantization_stress",
            "parameters": {"n_qubits": 14, "repeat_layers": 2},
        },
    }]
    config["candidate_generation"]["one_trial_searches"] = 1
    config_path = tmp_path / "fixture.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    output = tmp_path / "generated"
    generator.generate(config_path, output)
    dataset_path = output / "candidate_paths.json"
    rankings_path = output / "candidate_rankings.csv"
    calibration_path = output / "calibration_candidate_set.json"
    dataset = json.loads(dataset_path.read_text())
    calibration = json.loads(calibration_path.read_text())
    expected_stage = {
        (cell["circuit_id"], cell["topology_id"], candidate_id)
        for cell in calibration["cells"]
        for candidate_id in cell["candidate_path_ids"]
    }
    assert dataset["execution_contract"] == generator.WAVE_EXECUTION_CONTRACT
    assert dataset["score_id"] == "upmem_slr_wave_cost_v1"
    assert dataset["circuits"][0]["candidates"]

    cpu = qualifier.qualify_cpu(dataset_path, rankings_path, tmp_path / "cpu.json")
    assert cpu["all_passed"] is True

    def reject_representative_substitution(*args, **kwargs):
        raise AssertionError("explicit calibration set must not be replaced by representatives")

    monkeypatch.setattr(
        qualifier, "_representative_ids_for_topology", reject_representative_substitution
    )
    for mode in ("sdk", "calibration"):
        packet_path = tmp_path / f"{mode}.yml"
        packet = qualifier.prepare_config(
            dataset_path=dataset_path, rankings_path=rankings_path,
            calibration_path=calibration_path, output_path=packet_path, mode=mode,
        )
        assert set(packet["routes"]) == {"1dpu_t8", "4dpu_t8"}
        assert {
            (item["case_id"], route_id, item["plan_id"].removeprefix("path_"))
            for item in packet["matrix"]
            for route_id in item["route_ids"]
        } == expected_stage
        if mode == "sdk":
            assert packet["collection"]["warmup_blocks"] == 0
            assert packet["collection"]["measurement_blocks"] == 1
            assert all(
                route["executor"] == "upmem_sdk_simulator"
                and "rank_paths" not in route["options"]
                for route in packet["routes"].values()
            )
        for route in packet["routes"].values():
            assert route["options"]["request_transport"] == "packed_wave_v1"
            assert route["options"]["schedule_policy"] == "static_dag_waves_v1"
        provenance = json.loads(packet_path.with_suffix(".yml.provenance.json").read_text())
        assert provenance["configuration_sha256"] == hashlib.sha256(packet_path.read_bytes()).hexdigest()
        normalized = json.loads(qualifier.canonical_json(qualifier.load_experiment_config(packet_path)))
        assert provenance["normalized_configuration_sha256"] == hashlib.sha256(
            qualifier._canonical_bytes(normalized)
        ).hexdigest()
        assert provenance["calibration_set_sha256"] == hashlib.sha256(
            calibration_path.read_bytes()
        ).hexdigest()
        assert "stage_id" not in normalized
        if mode == "calibration":
            raw_root = tmp_path / "archived-stage" / "raw"
            raw_root.mkdir(parents=True)
            archived = raw_root.parent / "preregistration"
            archived.mkdir()
            (archived / "physical.yml").write_bytes(packet_path.read_bytes())
            sidecar = archived / "physical.yml.provenance.json"
            sidecar.write_bytes(qualifier._canonical_bytes(provenance))
            binary_hashes = {
                route["options"][field]: "0" * 64
                for route in packet["routes"].values()
                for field in ("dpu_binary", "host_binary", "initialization_binary")
            }
            (archived / "binary_sha256.json").write_bytes(
                qualifier._canonical_bytes(binary_hashes)
            )
            (archived / "SHA256SUMS").write_text("".join(
                f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n"
                for path in sorted(archived.iterdir())
            ))
            bound = generator._wave_private_archive(
                raw_root, sidecar, {"configuration": {"experiment": normalized}}
            )
            assert bound["binary_manifest"] == binary_hashes
            assert bound["configuration_sha256"] == provenance["configuration_sha256"]
        assert provenance["execution_contract"] == dataset["execution_contract"]
        assert {row["topology_id"] for row in provenance["selected"]} == {
            "1dpu_t8", "4dpu_t8"
        }

    calibration.update({
        "stage_id": "adaptive_training", "round_ordinal": 1,
        "prior_stage_hashes": ["1" * 64],
        "selection_profile_sha256": "2" * 64,
        "timing_used_for_selection": True,
    })
    adaptive_path = tmp_path / "adaptive-fixture.json"
    adaptive_path.write_bytes(generator._canonical_bytes(calibration))
    for mode in ("sdk", "calibration"):
        packet_path = tmp_path / f"adaptive-{mode}.yml"
        packet = qualifier.prepare_config(
            dataset_path=dataset_path, calibration_path=adaptive_path,
            output_path=packet_path, mode=mode,
        )
        suffix = "-sdk" if mode == "sdk" else ""
        assert packet["experiment_id"] == f"upmem-final-system-path-adaptive-1{suffix}-v2"
        provenance = json.loads(packet_path.with_suffix(".yml.provenance.json").read_text())
        assert provenance["stage"]["round_ordinal"] == 1
        assert provenance["stage"]["selection_profile_sha256"] == "2" * 64
        assert "stage_id" not in packet


def test_tiny_bell_is_rejected_for_planned_execution_coverage():
    config = generator.load_config(ROOT / "configs" / "upmem_final_system_path_study_v2.json")
    config["circuits"] = [{
        "circuit_id": "bell_pipeline_fixture", "split": "training",
        "circuit": {"kind": "builtin", "name": "bell_2q", "parameters": {}},
    }]
    config["candidate_generation"]["one_trial_searches"] = 1
    definition = config["circuits"][0]["circuit"]
    circuit = generator.builtin_circuit(definition["name"], definition["parameters"])
    network, _inputs = generator.lower_tensor_network(
        generator.make_simulation_job(circuit)
    )
    path, provenance = generator.plan_opt_einsum(network, optimize="greedy")
    item = {
        "candidate_path_id": generator.path_id(path, circuit_id="bell_pipeline_fixture"),
        "path": path,
        "source_kind": "opt_einsum_greedy",
        "source_seed": None,
        "planner_config_hash": provenance["planner_config_hash"],
        "is_greedy": True,
    }
    record, rows, candidate = generator._serialized_candidate_with_admission(
        circuit_id="bell_pipeline_fixture",
        split="training",
        definition=definition,
        item=item,
        config=config,
    )

    topology = next(item for item in record["topologies"] if item["topology_id"] == "4dpu_t8")
    assert candidate is not None
    assert topology["feasible"] is False
    assert "planned_execution_resource_admission_failed:" in topology["infeasibility_reason"]
    assert topology["wave_facts"] is None
    assert topology["features"] == {}
    assert topology["resource_admission"]["collection_resource_admission_passed"] is False
    assert any(
        row["topology_id"] == "4dpu_t8"
        and row["feasible"] is False
        and "planned_execution_resource_admission_failed:" in row["infeasibility_reason"]
        for row in rows
    )
    with pytest.raises(ValueError, match="greedy candidate is infeasible"):
        generator.build_dataset(config)
