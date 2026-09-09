from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest

from quantum_bench.circuits import load_circuit, parse_openqasm2
from quantum_bench.evidence import problem_id
from quantum_bench.model import make_simulation_job, validate_circuit_spec


ROOT = Path(__file__).resolve().parents[1]
WORKLOAD_PATH = ROOT / "configs" / "upmem_family_aligned_workload_v1.json"
STUDY_PATH = ROOT / "configs" / "upmem_family_aligned_path_study_v1.json"
QASM_ROOT = ROOT / "configs" / "qasm" / "upmem_family_aligned_v1"


def _load_script(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


path_study = _load_script(
    "upmem_path_heuristic_packet_test",
    ROOT / "scripts" / "upmem_path_heuristic.py",
)
qualify = _load_script(
    "qualify_upmem_path_candidates_packet_test",
    ROOT / "scripts" / "qualify_upmem_path_candidates.py",
)
family_workload = _load_script(
    "upmem_family_workload_packet_test",
    ROOT / "scripts" / "upmem_family_workload.py",
)


def _manifest() -> dict[str, object]:
    return json.loads(WORKLOAD_PATH.read_text(encoding="utf-8"))


def _study() -> dict[str, object]:
    return path_study.load_config(STUDY_PATH)


def test_packet_has_exact_qasm_hash_identity_widths_and_split_agreement() -> None:
    manifest = _manifest()
    study = _study()
    instances = manifest["instances"]
    study_circuits = study["circuits"]
    assert len(instances) == len(study_circuits) == 12
    assert manifest["status"] == study["status"] == "workload_definitions_frozen"
    assert manifest["definition_policy"]["qasm_file_private_parameters"] == [
        "qasm_sha256"
    ]
    assert set(manifest["qualification_status"]) == {
        "exposure",
        "resource_admission",
        "candidate_pools",
        "software",
        "sdk",
        "physical",
    }
    assert study["identity_requirements"]["candidate_generation_source_sha256"] == (
        "pending_candidate_generation"
    )

    by_id = {item["instance_id"]: item for item in instances}
    study_by_id = {item["circuit_id"]: item for item in study_circuits}
    assert set(by_id) == set(study_by_id)
    assert len({item["canonical_circuit"]["operation_identity"]["problem_id"] for item in instances}) == 12
    assert {item["family"] for item in instances if item["split"] == "training"} == {
        "QRNG",
        "BV",
        "BB84",
        "EDC",
        "HS",
        "XOR",
    }
    assert {item["family"] for item in instances if item["split"] == "test"} == {
        "QRNG",
        "BV",
        "BB84",
        "EDC",
        "HS",
        "XOR",
    }

    metadata = {
        item["instance_id"]: item for item in family_workload.all_instance_metadata()
    }
    for instance_id, item in by_id.items():
        source = item["source"]
        definition = item["circuit"]
        qasm_path = ROOT / definition["path"]
        assert qasm_path == QASM_ROOT / f"{instance_id}.qasm"
        assert definition["name"] is None
        assert set(definition["parameters"]) == {"qasm_sha256"}
        assert definition["parameters"]["qasm_sha256"] == source["qasm_sha256"]
        qasm = qasm_path.read_bytes()
        assert hashlib.sha256(qasm).hexdigest() == source["qasm_sha256"]
        assert qasm == family_workload.build_family_qasm(
            item["family_key"], metadata[instance_id]["parameters"]
        )

        circuit = path_study._circuit_from_definition(definition)
        validate_circuit_spec(circuit)
        parsed = parse_openqasm2(qasm_path)
        assert parsed.operations == circuit.operations
        assert circuit.n_qubits == item["canonical_circuit"]["n_qubits"]
        assert problem_id(make_simulation_job(circuit)) == item["canonical_circuit"]["operation_identity"]["problem_id"]

        widths = item["widths"]
        if item["family"] == "EDC":
            data = item["family_parameters"]["data_qubits"]
            assert widths == {
                "logical_qubits": 1,
                "encoded_data_qubits": data,
                "syndrome_qubits": data - 1,
                "ancilla_qubits": data - 1,
                "allocated_qubits": 2 * data - 1,
                "total_qubits": 2 * data - 1,
            }
        if item["family"] == "HS":
            assert widths["pair_count"] == widths["allocated_qubits"] // 2
            assert widths["pair_layout"] == "q[2j] control, q[2j+1] target"
        assert study_by_id[instance_id]["split"] == item["split"]
        assert study_by_id[instance_id]["family_key"] == item["family_key"]
        assert study_by_id[instance_id]["canonical_problem_id"] == item["canonical_circuit"]["operation_identity"]["problem_id"]
        assert study_by_id[instance_id]["circuit"] == definition


def test_private_qasm_parse_and_preparation_strip_only_private_hash(tmp_path: Path) -> None:
    manifest = _manifest()
    for item in manifest["instances"]:
        definition = item["circuit"]
        private_circuit = path_study._circuit_from_definition(definition)
        private_id = problem_id(make_simulation_job(private_circuit))
        public, binding = qualify._prepared_circuit_definition(definition)
        assert public["name"] is None
        assert public["parameters"] == {}
        assert public["path"] == (
            f"qasm/{definition['parameters']['qasm_sha256']}/"
            f"{Path(definition['path']).name}"
        )
        assert binding["qasm_sha256"] == definition["parameters"]["qasm_sha256"]

        bundle = tmp_path / item["instance_id"]
        staged = bundle / public["path"]
        staged.parent.mkdir(parents=True)
        staged.write_bytes((ROOT / definition["path"]).read_bytes())
        relocated = load_circuit({"circuit": public}, bundle)
        validate_circuit_spec(relocated)
        assert problem_id(make_simulation_job(relocated)) == private_id


@pytest.mark.parametrize(
    "instance_id",
    [item["instance_id"] for item in json.loads(WORKLOAD_PATH.read_text())["instances"]],
)
def test_greedy_only_wave_admission_passes_both_topologies(instance_id: str) -> None:
    study = _study()
    config = deepcopy(study)
    config["candidate_generation"]["one_trial_searches"] = 0
    circuit_record = next(
        item for item in config["circuits"] if item["circuit_id"] == instance_id
    )
    definition = circuit_record["circuit"]
    circuit = path_study._circuit_from_definition(definition)
    network, _inputs = path_study.lower_tensor_network(
        path_study.make_simulation_job(circuit)
    )
    path, provenance = path_study.plan_opt_einsum(network, optimize="greedy")
    item = {
        "candidate_path_id": path_study.path_id(path, circuit_id=instance_id),
        "path": path,
        "source_kind": "opt_einsum_greedy",
        "source_seed": None,
        "planner_config_hash": provenance["planner_config_hash"],
        "is_greedy": True,
    }
    record, _rows, candidate = path_study._serialized_candidate_with_admission(
        circuit_id=instance_id,
        split=circuit_record["split"],
        definition=definition,
        item=item,
        config=config,
    )

    assert candidate is not None
    assert set(candidate.feasible_topologies) == {"1dpu_t8", "4dpu_t8"}
    for topology in record["topologies"]:
        assert topology["feasible"] is True
        assert topology["resource_admission"] is not None
        assert type(topology["resource_admission"]["tasklet_row_sufficiency_passed"]) is bool
        assert type(topology["resource_admission"]["collection_resource_admission_passed"]) is bool
        assert topology["memory_admission"]["passed"] is True
