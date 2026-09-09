from copy import deepcopy
import hashlib
import json

import numpy as np
import pytest
import yaml

from quantum_bench import cli, planning
from quantum_bench.circuits import builtin_circuit
from quantum_bench.cpu import run_cpu_once
from quantum_bench.evidence import canonical_json, tensor_network_structure_id
from quantum_bench.experiment import _normalize_plan, load_experiment_config
from quantum_bench.lowering import build_contraction_dag, contraction_dag_hash, lower_tensor_network
from quantum_bench.model import make_simulation_job
from quantum_bench.upmem.plan import UpmemTopology, physical_plan_id, plan_upmem
from quantum_bench.report import verify_artifacts
from tests.test_cli_report import _numpy_config


@pytest.fixture
def frozen():
    job = make_simulation_job(builtin_circuit("bell_2q"))
    network, inputs = lower_tensor_network(job)
    # An explicit complete active-list path; no planner is used to construct it.
    path = [[0, 1] for _ in range(len(network.tensors) - 1)]
    dag = build_contraction_dag(network, path)
    config = {"planner": {
        "engine": "frozen_path", "mode": "replay", "path": path,
        "tensor_network_structure_id": tensor_network_structure_id(network),
        "logical_plan_id": contraction_dag_hash(dag),
    }, "slicing": None}
    return job, network, inputs, dag, config


def test_public_config_load_canonical_lists_and_immutable_input(tmp_path, frozen):
    config = deepcopy(frozen[-1])
    config["planner"]["path"] = [pair[::-1] for pair in config["planner"]["path"]]
    before = deepcopy(config)
    raw = yaml.safe_load(_numpy_config())
    raw["plans"]["greedy"] = config
    path = tmp_path / "replay.yml"
    path.write_text(yaml.safe_dump(raw))
    loaded = load_experiment_config(path)
    assert json.loads(canonical_json(loaded["plans"]["greedy"])) == frozen[-1]
    assert config == before
    assert isinstance(loaded["plans"]["greedy"]["planner"]["path"], tuple)
    assert cli._plan_dag(frozen[0], loaded["plans"]["greedy"])[2] == frozen[3]


@pytest.mark.parametrize("path", [
    None, "0,1", ((0, 1),), [(0, 1)], [[False, 1]], [[0.0, 1]],
    [["0", 1]], [[-1, 0]], [[0, 0]], [[0]], [[0, 1, 2]], [[0, 2]],
    [[0, 1], [0, 2]],
])
def test_strict_path_normalization_rejects_invalid_indices_and_types(frozen, path):
    config = deepcopy(frozen[-1])
    config["planner"]["path"] = path
    with pytest.raises(ValueError):
        _normalize_plan(config, "plans.replay")
    with pytest.raises(ValueError):
        planning.plan_frozen_path(
            frozen[1], path,
            tensor_network_structure_id=config["planner"]["tensor_network_structure_id"],
            logical_plan_id=config["planner"]["logical_plan_id"],
        )


@pytest.mark.parametrize("field", ["tensor_network_structure_id", "logical_plan_id"])
@pytest.mark.parametrize("value", [None, "", "a" * 63, "A" * 64, "g" * 64, 1])
def test_identity_format_rejected(frozen, field, value):
    config = deepcopy(frozen[-1])
    config["planner"][field] = value
    with pytest.raises((TypeError, ValueError)):
        _normalize_plan(config, "plans.replay")
    with pytest.raises((TypeError, ValueError)):
        cli._plan_dag(frozen[0], config)


@pytest.mark.parametrize("change", ["slicing", "mode", "extra", "missing"])
def test_strict_planner_fields(frozen, change):
    config = deepcopy(frozen[-1])
    if change == "slicing":
        config["slicing"] = {"node_id": "contract_0", "minimum_slice_count": 2}
    elif change == "mode":
        config["planner"]["mode"] = "greedy"
    elif change == "extra":
        config["planner"]["seed"] = 0
    else:
        del config["planner"]["logical_plan_id"]
    with pytest.raises(ValueError):
        _normalize_plan(config, "plans.replay")
    if change in {"slicing", "mode"}:
        with pytest.raises(ValueError, match="without slicing"):
            cli._plan_dag(frozen[0], config)


@pytest.mark.parametrize("change", ["network", "logical", "path", "incomplete"])
def test_mismatch_fails_before_allocator_or_external_search(frozen, monkeypatch, change):
    config = deepcopy(frozen[-1])
    if change == "network":
        config["planner"]["tensor_network_structure_id"] = "0" * 64
    elif change == "logical":
        config["planner"]["logical_plan_id"] = "0" * 64
    elif change == "path":
        config["planner"]["path"][0] = [1, 2]
    else:
        config["planner"]["path"].pop()

    def forbidden(*args, **kwargs):
        pytest.fail("replay invoked path search or executor allocation")

    for module in (planning, cli):
        monkeypatch.setattr(module, "plan_opt_einsum", forbidden)
        monkeypatch.setattr(module, "plan_cotengra", forbidden)
    monkeypatch.setattr(cli, "open_upmem", forbidden)
    monkeypatch.setattr(cli, "open_upmem_simulator", forbidden)
    with pytest.raises(ValueError, match="mismatch|incomplete"):
        cli._plan_dag(frozen[0], config)


@pytest.mark.parametrize("dpu_count", [1, 4])
def test_exact_replay_cpu_and_upmem_identity_parity(frozen, monkeypatch, dpu_count):
    def forbidden(*args, **kwargs):
        pytest.fail("replay invoked external search")

    for module in (planning, cli):
        monkeypatch.setattr(module, "plan_opt_einsum", forbidden)
        monkeypatch.setattr(module, "plan_cotengra", forbidden)
    network, inputs, dag, provenance = cli._plan_dag(frozen[0], frozen[-1])
    assert tensor_network_structure_id(network) == frozen[-1]["planner"]["tensor_network_structure_id"]
    assert contraction_dag_hash(dag) == contraction_dag_hash(frozen[3])
    policy = "split_complex_float32_v1"
    cpu = run_cpu_once(dag, inputs, policy)
    expected_cpu = run_cpu_once(frozen[3], frozen[2], policy)
    np.testing.assert_array_equal(cpu.output, expected_cpu.output)
    np.testing.assert_allclose(cpu.output.reshape(-1), [1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)])
    topology = UpmemTopology(dpu_count=dpu_count, tasklets_per_dpu=8, rank_count=1)
    actual_plan = plan_upmem(dag, numeric_policy=policy, topology=topology, schedule_policy="static_dag_waves_v1")
    expected_plan = plan_upmem(frozen[3], numeric_policy=policy, topology=topology, schedule_policy="static_dag_waves_v1")
    assert physical_plan_id(actual_plan) == physical_plan_id(expected_plan)
    assert provenance["planner_id"] == "frozen_path.replay"
    assert provenance["dependency_versions"] == {}
    assert provenance["planning_time_s"] >= 0
    encoded = json.dumps(provenance["planner_config"], sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    assert provenance["planner_config_hash"] == hashlib.sha256(encoded.encode()).hexdigest()


def test_reversed_pairs_have_identical_plan_and_config_hash(frozen):
    config = deepcopy(frozen[-1])
    config["planner"]["path"] = [pair[::-1] for pair in config["planner"]["path"]]
    original = cli._plan_dag(frozen[0], frozen[-1])
    reversed_pairs = cli._plan_dag(frozen[0], config)
    assert original[2] == reversed_pairs[2]
    assert original[3]["planner_config_hash"] == reversed_pairs[3]["planner_config_hash"]


def test_cpu_runner_replay_preserves_existing_evidence_schema(tmp_path, frozen):
    config = yaml.safe_load(_numpy_config(warmups=1, repetitions=2))
    config["plans"]["greedy"] = frozen[-1]
    path = tmp_path / "replay.yml"
    path.write_text(yaml.safe_dump(config))
    output = tmp_path / "evidence"
    result = cli.run_command(str(path), str(output), allow_physical=False)
    assert result["status"] == "completed"
    report = verify_artifacts(output)
    assert report["status"] == "completed"
    assert report["failed_count"] == report["unsupported_count"] == 0
    samples = [json.loads(line) for line in (output / "samples.jsonl").read_text().splitlines()]
    assert len(samples) == 3
    for sample in samples:
        assert sample["status"] == "success"
        assert sample["identities"]["logical_plan_id"] == frozen[-1]["planner"]["logical_plan_id"]
        assert sample["identities"]["tensor_network_structure_id"] == frozen[-1]["planner"]["tensor_network_structure_id"]


def test_single_tensor_empty_path_is_complete():
    from quantum_bench.model import TensorNetwork, TensorSpec

    network = TensorNetwork(builtin_circuit("bell_2q"), (TensorSpec("only", (0,), (2,), "dense"),), (0,), "a->a")
    expected = build_contraction_dag(network, [])
    path, _ = planning.plan_frozen_path(
        network, [], tensor_network_structure_id=tensor_network_structure_id(network),
        logical_plan_id=contraction_dag_hash(expected),
    )
    assert path == ()


@pytest.mark.parametrize("planner", [
    {"engine": "opt_einsum", "mode": "greedy"},
    {"engine": "opt_einsum", "mode": "optimal"},
    {"engine": "cotengra", "mode": "greedy", "max_repeats": 1, "seed": 7},
    {"engine": "cotengra", "mode": "labels", "max_repeats": 2, "seed": 0},
])
def test_conventional_plan_normalization_byte_equivalent(planner):
    plan = {"planner": planner, "slicing": None}
    before = json.dumps(plan, sort_keys=True, separators=(",", ":"))
    assert json.dumps(_normalize_plan(plan, "plans.old"), sort_keys=True, separators=(",", ":")) == before
