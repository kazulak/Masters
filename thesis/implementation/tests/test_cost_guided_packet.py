from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest
import yaml

from quantum_bench import cli
from quantum_bench.circuits import builtin_circuit
from quantum_bench.evidence import problem_id
from quantum_bench.experiment import load_experiment_config
from quantum_bench.planning import plan_opt_einsum
from quantum_bench.upmem.execution_features import extract_launch_cost_features


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("cost_guided_packet_qualifier", ROOT / "scripts/qualify_upmem_path_candidates.py")
qualifier = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = qualifier
SPEC.loader.exec_module(qualifier)


def _packet(definition):
    circuit = qualifier._circuit_from_definition(definition)
    job = qualifier.make_simulation_job(circuit)
    workload = {"workload_id": "tiny-fixture", "instances": [{
        "instance_id": "stress6", "family": "Stress", "split": "training", "circuit": definition,
        "canonical_circuit": {"operation_identity": {"problem_id": problem_id(job)}},
    }, {
        "instance_id": "test-unused", "family": "Stress", "split": "test",
        "circuit": {"kind": "qasm_file", "path": "/never-read-heldout.qasm"},
    }]}
    network, _ = qualifier.lower_tensor_network(job)
    path, _ = plan_opt_einsum(network)
    path = [sorted(pair) for pair in path]
    identifier = qualifier.path_id(path, circuit_id="stress6")
    dag = qualifier.build_contraction_dag(network, path)
    manifest = {
        "study_id": "upmem_cost_guided_path_study_v1", "stage": "initial",
        "binding_hash": "a" * 64, "profile_hash": "b" * 64, "normalization_hash": "c" * 64,
        "warmup_blocks": [0], "measurement_blocks": [1, 2, 3], "expected_attempts": 8,
        "physical_admission": "not_performed", "cells": {},
    }
    for topology_id, dpu_count in (("1dpu_t8", 1), ("4dpu_t8", 4)):
        plan = qualifier.plan_upmem(
            dag, numeric_policy=qualifier.FLOAT32,
            topology=qualifier.UpmemTopology(dpu_count=dpu_count, rank_count=1, tasklets_per_dpu=8),
            schedule_policy="static_dag_waves_v1",
        )
        qualifier._require_wave_execution_coverage(plan)
        facts = extract_launch_cost_features(dag, plan)
        execution = facts.pop("execution")
        facts.update(
            logical_plan_id=qualifier.contraction_dag_hash(dag),
            physical_plan_id=qualifier.physical_plan_id(plan),
            resource_admission=qualifier.collection_resource_admission(plan),
            declared_host_bytes=execution["host_buffers"]["declared_executor_memory_estimate_bytes"],
        )
        manifest["cells"][f"stress6/{topology_id}"] = {
            "circuit_id": "stress6", "family": "Stress", "topology_id": topology_id, "split": "training",
            "selection": {"path_ids": [identifier], "roles": {"G": identifier, "F": identifier, "R": identifier}},
            "candidates": {identifier: {"path_id": identifier, "path": path, "tree_flops": 1.0, "facts": facts}},
        }
    return manifest, workload


@pytest.fixture
def packet():
    return _packet({"kind": "builtin", "name": "quantization_stress", "parameters": {"n_qubits": 6, "repeat_layers": 1}})


def _prepare(packet, **kwargs):
    return qualifier.prepare_cost_guided_config(
        *packet, execution_root=ROOT, experiment_id="cost-guided-initial-fixture-v1", **kwargs
    )


@pytest.mark.parametrize("simulator", [False, True])
def test_exact_selected_paths_routes_counts_and_no_search(packet, tmp_path, monkeypatch, simulator):
    before = deepcopy(packet)

    def forbidden(*args, **kwargs):
        pytest.fail("preparation invoked search, legacy validation, file writing or execution")

    for name in ("plan_opt_einsum", "plan_cotengra", "_regenerate", "_validate_dataset_execution_contract", "_validate_calibration_execution_contract"):
        monkeypatch.setattr(qualifier, name, forbidden)
    with monkeypatch.context() as pure:
        pure.setattr(Path, "write_text", forbidden)
        pure.setattr(Path, "write_bytes", forbidden)
        config, provenance = _prepare(packet, simulator=simulator)
    assert packet == before
    assert config["defaults"]["timeout_s"] == 120
    assert config["collection"]["machine_policy"]["affinity"]["expected_cpus"] == [0]
    assert (config["collection"]["warmup_blocks"], config["collection"]["measurement_blocks"]) == ((0, 1) if simulator else (1, 3))
    assert provenance["expected_physical_attempts"] == 8
    assert provenance["expected_prepared_attempts"] == (2 if simulator else 8)
    assert len(provenance["selected_cells"]) == 2
    assert all(row["roles"] == ["F", "G", "R"] for row in provenance["selected_cells"])
    assert len(config["plans"]) == len(config["matrix"]) == 1
    assert config["matrix"][0]["route_ids"] == ["1dpu_t8", "4dpu_t8"]
    for topology, route in config["routes"].items():
        assert route["executor"] == ("upmem_sdk_simulator" if simulator else "upmem_physical")
        assert route["options"]["dpu_count"] == (1 if topology == "1dpu_t8" else 4)
        assert route["options"]["tasklets_per_dpu"] == 8
        assert route["options"]["fuse_complex"] is True
        assert route["options"]["geometry_policy"] == "panel_only_v1"
        assert route["options"]["schedule_policy"] == "static_dag_waves_v1"
        assert route["options"]["request_transport"] == "packed_wave_v1"
        assert ("rank_paths" not in route["options"]) is simulator
    path = tmp_path / "experiment.yml"
    path.write_text(yaml.safe_dump(config))
    loaded = load_experiment_config(path)
    for plan in loaded["plans"].values():
        assert plan["planner"]["engine"] == "frozen_path"
        job = cli._job(loaded["cases"]["stress6"])
        _, _, dag, _ = cli._plan_dag(job, plan)
        assert qualifier.contraction_dag_hash(dag) == plan["planner"]["logical_plan_id"]
    expected_hash = hashlib.sha256(json.dumps(packet[0], sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    assert provenance["manifest_record_sha256"] == expected_hash


@pytest.mark.parametrize("mutation", [
    "path", "path_id", "logical_plan_id", "physical_plan_id", "expected_attempts",
    "split", "family", "topology_id", "missing_cell", "extra_cell", "duplicate_instance",
    "duplicate_path", "extra_candidate", "role", "stage", "blocks", "binding_hash",
    "problem", "memory", "facts",
])
def test_tampered_packet_rejected(packet, mutation):
    manifest, workload = deepcopy(packet)
    cell = next(iter(manifest["cells"].values()))
    identifier, candidate = next(iter(cell["candidates"].items()))
    if mutation == "path":
        candidate["path"][0] = [False, 1]
    elif mutation == "path_id":
        candidate["path_id"] = "0" * 64
    elif mutation in {"logical_plan_id", "physical_plan_id"}:
        candidate["facts"][mutation] = "0" * 64
    elif mutation == "expected_attempts":
        manifest[mutation] += 4
    elif mutation in {"split", "family", "topology_id"}:
        cell[mutation] = "test"
    elif mutation == "missing_cell":
        manifest["cells"].popitem()
    elif mutation == "extra_cell":
        manifest["cells"]["test-unused/1dpu_t8"] = deepcopy(cell)
    elif mutation == "duplicate_instance":
        workload["instances"].append(deepcopy(workload["instances"][0]))
    elif mutation == "duplicate_path":
        cell["selection"]["path_ids"].append(identifier)
    elif mutation == "extra_candidate":
        cell["candidates"]["extra"] = deepcopy(candidate)
    elif mutation == "role":
        cell["selection"]["roles"]["G"] = "missing"
    elif mutation == "stage":
        manifest["stage"] = "feedback_1"
    elif mutation == "blocks":
        manifest["measurement_blocks"] = [1, 2]
    elif mutation == "binding_hash":
        manifest[mutation] = "wrong"
    elif mutation == "problem":
        workload["instances"][0]["canonical_circuit"]["operation_identity"]["problem_id"] = "0" * 64
    elif mutation == "memory":
        candidate["facts"]["declared_host_bytes"] = 2**30
    else:
        candidate["facts"]["H"] += 1
    with pytest.raises((ValueError, TypeError)):
        _prepare((manifest, workload))


@pytest.mark.parametrize("experiment_id", ["", " ", " leading", None])
def test_requires_explicit_experiment_id(packet, experiment_id):
    with pytest.raises(ValueError, match="experiment_id"):
        qualifier.prepare_cost_guided_config(*packet, execution_root=ROOT, experiment_id=experiment_id)


def test_handcrafted_manifest_cannot_expand_initial_path_cap(packet):
    manifest, workload = deepcopy(packet)
    for cell in manifest["cells"].values():
        candidate = next(iter(cell["candidates"].values()))
        ids = [f"candidate-{index}" for index in range(5)]
        cell["selection"] = {"path_ids": ids, "roles": dict(zip(("G", "F", "R", "diverse_3", "diverse_4"), ids))}
        cell["candidates"] = {identifier: deepcopy(candidate) for identifier in ids}
    manifest["expected_attempts"] = 40
    with pytest.raises(ValueError, match="at most 4"):
        _prepare((manifest, workload))


def test_handcrafted_manifest_cannot_expand_initial_attempt_cap(packet):
    manifest, workload = deepcopy(packet)
    template = deepcopy(workload["instances"][0])
    initial_cells = deepcopy(manifest["cells"])
    workload["instances"] = []
    manifest["cells"] = {}
    for index in range(25):
        circuit_id = f"instance-{index}"
        workload["instances"].append({**template, "instance_id": circuit_id})
        for original in initial_cells.values():
            cell = {**deepcopy(original), "circuit_id": circuit_id}
            manifest["cells"][f"{circuit_id}/{cell['topology_id']}"] = cell
    manifest["expected_attempts"] = 200
    with pytest.raises(ValueError, match="at most 192"):
        _prepare((manifest, workload))


def test_qasm_binding_staging_and_relocation(tmp_path):
    circuit = builtin_circuit("quantization_stress", {"n_qubits": 6, "repeat_layers": 1})
    lines = ['OPENQASM 2.0;', 'include "qelib1.inc";', 'qreg q[6];']
    for op in circuit.operations:
        params = "(" + ",".join(str(value) for value in op.params) + ")" if op.params else ""
        lines.append(f"{op.gate}{params} " + ",".join(f"q[{wire}]" for wire in op.wires) + ";")
    source = tmp_path / "fixture.qasm"
    source.write_text("\n".join(lines) + "\n")
    payload = source.read_bytes()
    definition = {"kind": "qasm_file", "name": None, "path": str(source), "parameters": {"qasm_sha256": hashlib.sha256(payload).hexdigest()}}
    packet = _packet(definition)
    config, provenance = _prepare(packet, simulator=True)
    assert config["cases"]["stress6"]["circuit"]["parameters"] == {}
    assert config["cases"]["stress6"]["circuit"]["name"] is None
    binding, = provenance["qasm_source_bindings"]
    bundle = tmp_path / "relocated"
    staged = bundle / binding["prepared_path"]
    staged.parent.mkdir(parents=True)
    staged.write_bytes(payload)
    path = bundle / "sdk.yml"
    path.write_text(yaml.safe_dump(config))
    loaded = load_experiment_config(path)
    job = cli._job(loaded["cases"]["stress6"])
    assert problem_id(job) == packet[1]["instances"][0]["canonical_circuit"]["operation_identity"]["problem_id"]
    for plan in loaded["plans"].values():
        cli._plan_dag(job, plan)
    source.write_bytes(payload + b"// drift\n")
    with pytest.raises(ValueError, match="qasm_sha256 does not match source bytes"):
        _prepare(packet)
