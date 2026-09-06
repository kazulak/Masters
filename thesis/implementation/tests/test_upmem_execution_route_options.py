from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from quantum_bench.evidence import identity_hash
from quantum_bench.experiment import (
    default_validation_policy_id,
    load_experiment_config,
    upmem_execution_policy,
)


_DEFAULT_POLICY = {
    "request_transport": "packed_operation_v1",
    "schedule_policy": "serial_nodes_v1",
    "fuse_complex": False,
    "geometry_policy": "panel_only_v1",
}


def _route_options(executor: str) -> dict[str, object]:
    if executor == "numpy_dag":
        return {}
    options: dict[str, object] = {
        "dpu_count": 1,
        "rank_count": 1,
        "tasklets_per_dpu": 1,
        "session_root": "native",
        "host_binary": "native/host",
        "dpu_binary": "native/dpu",
        "initialization_binary": "native/init",
    }
    if executor == "upmem_physical":
        options["rank_paths"] = ["/dev/dpu_rank0"]
    return options


def _config(
    *,
    executor: str = "upmem_sdk_simulator",
    options: dict[str, object] | None = None,
) -> dict[str, object]:
    route_options = _route_options(executor)
    if options:
        route_options.update(options)
    return {
        "schema_version": "tn_benchmark_v3",
        "experiment_id": "route-options-focused",
        "defaults": {"timeout_s": 2.5},
        "collection": {
            "claim_policy": "diagnostic_v1",
            "base_seed": 7,
            "warmup_blocks": 0,
            "measurement_blocks": 1,
            "session_policy": "fresh_session_per_attempt_v1",
            "block_cooldown_s": 0.0,
            "machine_policy": {
                "machine_exclusivity": {"mode": "observed_v1"},
                "cpu_governor": {"mode": "observed_v1"},
                "affinity": {"mode": "observed_v1", "expected_cpus": None},
                "numa_policy": {"mode": "observed_v1"},
                "background_load": {
                    "mode": "observed_v1",
                    "max_load1_per_online_cpu": None,
                },
            },
        },
        "cases": {
            "case": {
                "circuit": {
                    "kind": "builtin",
                    "name": "bell_2q",
                    "path": None,
                    "parameters": {},
                }
            }
        },
        "plans": {
            "plan": {
                "planner": {"engine": "opt_einsum", "mode": "greedy"},
                "slicing": None,
            }
        },
        "routes": {
            "route": {
                "executor": executor,
                "numeric_policy": (
                    "split_complex_float32_v1"
                    if executor != "quest_cpu"
                    else None
                ),
                "options": route_options,
            }
        },
        "matrix": [
            {"case_id": "case", "plan_id": "plan", "route_ids": ["route"]}
        ],
    }


def _write_config(path: Path, value: dict[str, object]) -> None:
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def _load(tmp_path: Path, value: dict[str, object]) -> dict[str, object]:
    path = tmp_path / "config.yml"
    _write_config(path, value)
    return dict(load_experiment_config(path))


@pytest.mark.parametrize("executor", ["upmem_sdk_simulator", "upmem_physical"])
def test_legacy_upmem_options_keep_normalized_shape_and_identity(
    tmp_path: Path, executor: str
) -> None:
    value = _config(executor=executor)
    expected_raw = deepcopy(value)
    config = _load(tmp_path, value)

    options = config["routes"]["route"]["options"]
    assert set(options) == set(_route_options(executor))
    assert upmem_execution_policy(options) == _DEFAULT_POLICY
    assert config["experiment_id"] == identity_hash(
        "quantum_bench.experiment_id.v3",
        {
            "label": expected_raw["experiment_id"],
            "configuration": expected_raw,
            "validation_policy_id": default_validation_policy_id(),
        },
    )


@pytest.mark.parametrize("executor", ["upmem_sdk_simulator", "upmem_physical"])
@pytest.mark.parametrize(
    "policy",
    [
        {"request_transport": "packed_operation_v1"},
        {"request_transport": "packed_wave_v1"},
        {
            "request_transport": "packed_wave_v1",
            "schedule_policy": "static_dag_waves_v1",
        },
        {"request_transport": "packed_wave_v1", "fuse_complex": True},
        {
            "request_transport": "packed_wave_v1",
            "geometry_policy": "outer_k1_v1",
        },
        {
            "request_transport": "packed_wave_v1",
            "schedule_policy": "static_dag_waves_v1",
            "fuse_complex": True,
            "geometry_policy": "outer_k1_v1",
        },
    ],
)
def test_valid_upmem_policy_combinations_are_preserved(
    tmp_path: Path, executor: str, policy: dict[str, object]
) -> None:
    value = _config(executor=executor, options=policy)
    config = _load(tmp_path, value)

    normalized_options = config["routes"]["route"]["options"]
    assert {key: normalized_options[key] for key in policy} == policy
    assert upmem_execution_policy(normalized_options) == {
        **_DEFAULT_POLICY,
        **policy,
    }


@pytest.mark.parametrize(
    "field,value",
    [
        ("request_transport", True),
        ("request_transport", 1),
        ("schedule_policy", True),
        ("schedule_policy", 1),
        ("fuse_complex", 1),
        ("fuse_complex", 0),
        ("geometry_policy", True),
        ("geometry_policy", 1),
    ],
)
def test_helper_rejects_wrong_policy_types(field: str, value: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        upmem_execution_policy({field: value})


@pytest.mark.parametrize(
    "field,value",
    [
        ("request_transport", "directory_v1"),
        ("schedule_policy", "dynamic_v1"),
        ("geometry_policy", "outer_k2_v1"),
    ],
)
def test_helper_rejects_unknown_policy_values(field: str, value: object) -> None:
    with pytest.raises(ValueError, match="unsupported"):
        upmem_execution_policy({field: value})


@pytest.mark.parametrize("executor", ["upmem_sdk_simulator", "upmem_physical"])
@pytest.mark.parametrize(
    "field,value",
    [
        ("schedule_policy", "static_dag_waves_v1"),
        ("fuse_complex", True),
        ("geometry_policy", "outer_k1_v1"),
    ],
)
def test_nondefault_policy_requires_explicit_prepared_transport(
    tmp_path: Path, executor: str, field: str, value: object
) -> None:
    with pytest.raises(ValueError, match="packed_wave_v1"):
        _load(tmp_path, _config(executor=executor, options={field: value}))


@pytest.mark.parametrize("executor", ["upmem_sdk_simulator", "upmem_physical"])
@pytest.mark.parametrize("options", [{"rank_count": 2}, {"dpu_count": 65}])
def test_prepared_transport_has_one_rank_and_at_most_64_dpus(
    tmp_path: Path, executor: str, options: dict[str, object]
) -> None:
    with pytest.raises(ValueError, match="exactly one rank.*64 DPUs"):
        _load(
            tmp_path,
            _config(
                executor=executor,
                options={"request_transport": "packed_wave_v1", **options},
            ),
        )


def test_unknown_upmem_option_and_cpu_policy_option_are_rejected(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="fields must be exact"):
        _load(tmp_path, _config(options={"unknown_option": True}))

    cpu = _config(executor="numpy_dag", options={"request_transport": "packed_wave_v1"})
    with pytest.raises(ValueError, match="fields must be exact"):
        _load(tmp_path, cpu)


def test_explicit_legacy_defaults_are_allowed_but_are_identity_visible(
    tmp_path: Path,
) -> None:
    legacy = _load(tmp_path, _config())
    explicit_value = _config(options=deepcopy(_DEFAULT_POLICY))
    explicit = _load(tmp_path, explicit_value)

    explicit_options = explicit["routes"]["route"]["options"]
    assert set(explicit_options) == set(_route_options("upmem_sdk_simulator")) | set(
        _DEFAULT_POLICY
    )
    for key, value in _DEFAULT_POLICY.items():
        assert explicit_options[key] == value
    assert explicit["experiment_id"] != legacy["experiment_id"]
