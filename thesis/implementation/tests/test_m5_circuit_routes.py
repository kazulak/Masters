from __future__ import annotations

from copy import deepcopy

import pytest

from quantum_bench.bench.m5_circuit_routes import (
    RouteAdapterError,
    admit_route_observation,
    apply_rank_path_override,
    build_pipeline_catalog,
    select_pipeline_routes,
    validate_route_adapter,
)


def _config() -> dict:
    return {
        "planner_variants": [
            {
                "id": "greedy",
                "label": "Greedy",
                "planner": {"engine": "opt_einsum", "optimize": "greedy"},
            },
            {
                "id": "auto",
                "label": "Auto",
                "planner": {"engine": "opt_einsum", "optimize": "auto"},
            },
        ],
        "numeric_policies": [
            {"id": "float", "label": "Float32", "policy": "float32_real"},
            {"id": "int8", "label": "Int8", "policy": "host_packed_int8"},
        ],
        "engine_variants": [
            {
                "id": "cpu",
                "label": "CPU",
                "engine": "numpy_cpu",
                "topology": {
                    "backend": "cpu",
                    "device_ids": ["cpu"],
                    "tasklets_per_device": 1,
                },
                "executor_config": {},
            },
            {
                "id": "upmem",
                "label": "UPMEM",
                "engine": "upmem_v4",
                "topology": {
                    "backend": "upmem",
                    "device_ids": ["dpu:0", "dpu:1"],
                    "rank_paths": ["/dev/dpu_rank0"],
                    "tasklets_per_device": 1,
                },
                "executor_config": {
                    "profile": "profile-v1",
                    "abi": "abi-v1",
                    "session_protocol": "session-v1",
                    "dispatch_mode": "bulk-synchronous",
                    "kernel_identity": "kernel-v1",
                    "execution_class": "physical",
                },
            },
        ],
    }


def _catalog_config() -> dict:
    config = _config()
    routes, comparisons = build_pipeline_catalog(config)
    config["pipeline_routes"] = routes
    config["pipeline_comparisons"] = comparisons
    return config


def test_catalog_is_deterministic_and_preserves_route_and_comparison_identity() -> None:
    first_routes, first_comparisons = build_pipeline_catalog(_config())
    second_routes, second_comparisons = build_pipeline_catalog(_config())

    assert first_routes == second_routes
    assert first_comparisons == second_comparisons
    assert [route["route_id"] for route in first_routes] == [
        "greedy__float__cpu",
        "greedy__float__upmem",
        "greedy__int8__cpu",
        "greedy__int8__upmem",
        "auto__float__cpu",
        "auto__float__upmem",
        "auto__int8__cpu",
        "auto__int8__upmem",
    ]
    assert all(
        set(route["modules"])
        == {
            "tensor_network",
            "planner",
            "numeric",
            "executor",
            "topology",
            "kernel",
            "partitioner",
            "scheduler",
            "communication",
        }
        for route in first_routes
    )
    assert all(
        route["modules"]["planner"]["implementation"]
        == route["modules"]["planner"]["parameters"]["engine"]
        for route in first_routes
    )
    assert any(item["changed_roles"] == ["numeric"] for item in first_comparisons)
    assert any(item["changed_roles"] == ["planner"] for item in first_comparisons)


def test_selection_preserves_catalog_order_and_rejects_invalid_ids() -> None:
    config = _catalog_config()
    selected = select_pipeline_routes(
        config, ["auto__int8__upmem", "greedy__float__cpu"]
    )
    assert [route["route_id"] for route in selected] == [
        "greedy__float__cpu",
        "auto__int8__upmem",
    ]
    with pytest.raises(ValueError, match="unknown route ids"):
        select_pipeline_routes(config, ["unknown"])
    with pytest.raises(ValueError, match="selection cannot be empty"):
        select_pipeline_routes(config, [])
    with pytest.raises(ValueError, match="duplicate route ids"):
        select_pipeline_routes(config, ["greedy__float__cpu", "greedy__float__cpu"])


def test_rank_override_is_immutable_and_regenerates_route_hashes() -> None:
    config = _catalog_config()
    original = deepcopy(config)
    overridden = apply_rank_path_override(config, ["/dev/dpu_rank1"])

    assert config == original
    route_id = "greedy__float__upmem"
    original_route = next(
        route for route in config["pipeline_routes"] if route["route_id"] == route_id
    )
    overridden_route = next(
        route
        for route in overridden["pipeline_routes"]
        if route["route_id"] == route_id
    )
    assert original_route["route_config_hash"] != overridden_route["route_config_hash"]
    assert overridden_route["modules"]["topology"]["parameters"]["rank_paths"] == [
        "/dev/dpu_rank1"
    ]
    assert [item["comparison_id"] for item in config["pipeline_comparisons"]] == [
        item["comparison_id"] for item in overridden["pipeline_comparisons"]
    ]


def test_adapter_rejects_unimplemented_optional_module_and_preserves_details() -> None:
    config = _catalog_config()
    route = next(
        route
        for route in config["pipeline_routes"]
        if route["route_id"] == "greedy__float__upmem"
    )
    mutated = deepcopy(route)
    mutated["modules"]["communication"]["implementation"] = "pid_comm_v1"

    with pytest.raises(RouteAdapterError) as raised:
        validate_route_adapter(
            mutated,
            config["planner_variants"][0],
            config["numeric_policies"][0],
            config["engine_variants"][1],
        )
    assert raised.value.failure_stage == "route_adapter_mismatch"
    assert raised.value.details["status"] == "failed"
    assert raised.value.details["mismatches"]


def test_observation_admission_accepts_matching_cpu_and_physical_metadata() -> None:
    config = _catalog_config()
    cpu_route = next(
        route
        for route in config["pipeline_routes"]
        if route["route_id"] == "greedy__float__cpu"
    )
    upmem_route = next(
        route
        for route in config["pipeline_routes"]
        if route["route_id"] == "greedy__float__upmem"
    )

    assert admit_route_observation(
        cpu_route, {"execution_engine": "numpy_cpu"}, backend="cpu"
    )["passed"]
    physical = admit_route_observation(
        upmem_route,
        {
            "profile": "profile-v1",
            "abi": "abi-v1",
            "session_protocol": "session-v1",
            "dispatch_mode": "bulk-synchronous",
            "kernel_identity": "kernel-v1",
            "execution_class": "physical",
            "graph_intermediate_placement": "host_managed",
        },
        backend="upmem",
    )
    assert physical["passed"]

    mismatch = admit_route_observation(
        upmem_route, {"kernel_identity": "wrong-kernel"}, backend="upmem"
    )
    assert mismatch["status"] == "failed"
    assert any("kernel_identity" in item for item in mismatch["mismatches"])
