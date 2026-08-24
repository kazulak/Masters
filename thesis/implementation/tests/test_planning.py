from __future__ import annotations

import builtins
import hashlib
import json
import random
import subprocess
import sys
from pathlib import Path

import pytest

from quantum_bench.circuits import builtin_circuit
from quantum_bench.model import TensorNetwork, TensorSpec
from quantum_bench import planning
from quantum_bench.planning import plan_cotengra, plan_opt_einsum


ROOT = Path(__file__).parents[1]
PROVENANCE_KEYS = {
    "planner_engine",
    "planner_id",
    "planner_kind",
    "optimize_mode",
    "objective",
    "cost_basis",
    "planner_config",
    "planner_config_hash",
    "path_info_text",
    "largest_intermediate",
    "naive_flops",
    "optimized_flops",
    "planning_time_s",
    "dependency_versions",
}


def _chain_network() -> TensorNetwork:
    return TensorNetwork(
        circuit=builtin_circuit("bell_2q"),
        tensors=(
            TensorSpec("left", (0, 1), (2, 3), "dense"),
            TensorSpec("middle", (1, 2), (3, 4), "dense"),
            TensorSpec("right", (2, 3), (4, 5), "dense"),
        ),
        output_labels=(0, 3),
        einsum_expression="ab,bc,cd->ad",
    )


def _empty_network() -> TensorNetwork:
    return TensorNetwork(
        circuit=builtin_circuit("bell_2q"),
        tensors=(),
        output_labels=(),
        einsum_expression="->",
    )


def _singleton_network() -> TensorNetwork:
    return TensorNetwork(
        circuit=builtin_circuit("bell_2q"),
        tensors=(TensorSpec("only", (0,), (2,), "dense"),),
        output_labels=(0,),
        einsum_expression="a->a",
    )


def _assert_complete_pairwise_path(path: tuple[tuple[int, int], ...], tensor_count: int) -> None:
    active_count = tensor_count
    for step in path:
        assert len(step) == 2
        assert 0 <= step[0] < active_count
        assert 0 <= step[1] < active_count
        assert step[0] != step[1]
        active_count -= 1
    assert len(path) == tensor_count - 1
    assert active_count == 1


def test_opt_einsum_returns_pairwise_path_and_json_provenance() -> None:
    path, provenance = plan_opt_einsum(_chain_network())

    _assert_complete_pairwise_path(path, 3)
    assert set(provenance) == PROVENANCE_KEYS
    assert provenance["planner_engine"] == "opt_einsum"
    assert provenance["planner_config"]["engine"] == "opt_einsum"
    assert provenance["planner_config_hash"]
    assert "opt_einsum_version" not in provenance["planner_config"]
    assert set(provenance["dependency_versions"]) == {"opt_einsum"}
    assert json.loads(json.dumps(provenance)) == provenance


def test_opt_einsum_is_deterministic_and_hashes_resolved_configuration() -> None:
    first_path, first = plan_opt_einsum(_chain_network(), optimize="greedy")
    second_path, second = plan_opt_einsum(_chain_network(), optimize="greedy")
    changed_path, changed = plan_opt_einsum(_chain_network(), optimize="auto")

    assert first_path == second_path
    assert first["planner_config_hash"] == second["planner_config_hash"]
    assert first["planner_config_hash"] != changed["planner_config_hash"]
    _assert_complete_pairwise_path(changed_path, 3)
    assert set(first["dependency_versions"]) == {"opt_einsum"}


def test_planner_config_hash_excludes_dependency_metadata_and_timing() -> None:
    _, provenance = plan_opt_einsum(_chain_network())
    config = provenance["planner_config"]
    expected = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()
    changed_metadata = dict(provenance)
    changed_metadata["dependency_versions"] = {"opt_einsum": "different"}
    changed_metadata["planning_time_s"] = 123.0

    assert provenance["planner_config_hash"] == expected
    assert changed_metadata["planner_config_hash"] == expected


def test_empty_network_is_rejected_consistently() -> None:
    with pytest.raises(ValueError, match="^cannot plan an empty tensor network$"):
        plan_opt_einsum(_empty_network())
    with pytest.raises(ValueError, match="^cannot plan an empty tensor network$"):
        plan_cotengra(_empty_network())


def test_singleton_network_returns_empty_path_without_optimizer_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(*args: object, **kwargs: object) -> None:
        raise AssertionError("optimizer must not be called for a singleton network")

    monkeypatch.setattr(planning.oe, "contract_path", fail)
    opt_path, opt_provenance = plan_opt_einsum(_singleton_network())
    assert opt_path == ()
    assert set(opt_provenance) == PROVENANCE_KEYS
    assert opt_provenance["planning_time_s"] == 0.0

    pytest.importorskip("cotengra")
    import cotengra

    monkeypatch.setattr(cotengra, "HyperOptimizer", fail)
    cot_path, cot_provenance = plan_cotengra(_singleton_network())
    assert cot_path == ()
    assert set(cot_provenance) == PROVENANCE_KEYS
    assert cot_provenance["planning_time_s"] == 0.0


def test_cotengra_is_deterministic_and_restores_python_random_state() -> None:
    pytest.importorskip("cotengra")
    network = _chain_network()
    random.seed(12345)
    before = random.getstate()
    first_path, first = plan_cotengra(network, max_repeats=1, seed=0)
    after = random.getstate()
    second_path, second = plan_cotengra(network, max_repeats=1, seed=0)

    assert before == after
    assert first_path == second_path
    assert first["planner_config_hash"] == second["planner_config_hash"]
    assert first["objective"] == "flops"
    assert "cotengra_version" not in first["planner_config"]
    _assert_complete_pairwise_path(first_path, 3)
    assert set(first) == PROVENANCE_KEYS


def test_root_planning_is_lazy_and_does_not_import_upmem() -> None:
    script = """
import sys
import quantum_bench.planning
assert 'cotengra' not in sys.modules
assert not any(name.startswith('quantum_bench.upmem') for name in sys.modules)
assert not any(name.startswith('quantum_bench.targets.upmem') for name in sys.modules)
"""
    subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        env={"PYTHONPATH": "src"},
        check=True,
        capture_output=True,
        text=True,
    )


def test_missing_cotengra_is_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    original_import = builtins.__import__

    def reject_cotengra(name: str, *args: object, **kwargs: object):
        if name == "cotengra":
            raise ImportError("simulated missing cotengra")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_cotengra)
    with pytest.raises(RuntimeError, match="cotengra is required"):
        plan_cotengra(_chain_network())


def test_invalid_or_incomplete_paths_are_rejected() -> None:
    from quantum_bench.planning import _validate_pairwise_path

    with pytest.raises(ValueError, match="pairwise"):
        _validate_pairwise_path(((0, 1, 2),), 2)
    with pytest.raises(ValueError, match="incomplete"):
        _validate_pairwise_path(((0, 1),), 4)
