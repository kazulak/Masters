"""Pure route policy for the M5 whole-circuit study.

This module turns the study's planner, numeric-policy, and engine axes into
immutable route records.  It also validates that the fixed M5 executor profile
and the observed native metadata agree with the selected route.  It performs
no YAML loading, planning, execution, or artifact I/O.
"""

from __future__ import annotations

import re
from copy import deepcopy
from typing import Any, Iterable, Mapping

from quantum_bench.bench.route_specs import (
    ComparisonSpec,
    ModuleSpec,
    PipelineParameters,
    PipelineRoute,
)


RANK_PATH_PATTERN = re.compile(r"^/dev/dpu_rank[0-9]+$")


class RouteAdapterError(ValueError):
    """A selected route does not describe the fixed M5 executor profile."""

    failure_stage = "route_adapter_mismatch"

    def __init__(self, details: Mapping[str, Any]) -> None:
        self.details = dict(details)
        super().__init__(
            "selected route does not match the currently implemented M5 adapter: "
            + "; ".join(self.details.get("mismatches", ()))
        )


def apply_rank_path_override(
    config: Mapping[str, Any], rank_paths: Iterable[str]
) -> dict[str, Any]:
    """Return a copied configuration with explicit physical rank paths.

    Portable suite files contain placeholder rank paths.  This operation is a
    pure resolution step: it validates explicit paths, updates only physical
    engine topology, and regenerates route contracts without mutating input.
    """

    supplied = [str(path).strip() for path in rank_paths if str(path).strip()]
    if not supplied:
        raise ValueError("at least one explicit UPMEM rank path is required")
    if any(RANK_PATH_PATTERN.fullmatch(path) is None for path in supplied):
        raise ValueError("rank paths must match ^/dev/dpu_rank[0-9]+$")
    if len(set(supplied)) != len(supplied):
        raise ValueError("UPMEM rank paths must be unique")

    resolved = deepcopy(dict(config))
    for variant in resolved.get("engine_variants", []):
        topology = variant.get("topology", {})
        if topology.get("backend") == "cpu":
            continue
        expected = len(topology.get("rank_paths", []))
        if expected < 1:
            raise ValueError(
                f"physical engine {variant.get('id', '<unknown>')} has no declared rank count"
            )
        if len(supplied) < expected:
            raise ValueError(
                f"physical engine {variant.get('id', '<unknown>')} requires {expected} rank paths; "
                f"received {len(supplied)}"
            )
        topology["rank_paths"] = supplied[:expected]
        variant["topology"] = topology

    resolved["_rank_paths_resolved"] = supplied
    routes, comparisons = build_pipeline_catalog(resolved)
    resolved["pipeline_routes"] = routes
    resolved["pipeline_comparisons"] = comparisons
    return resolved


def build_pipeline_catalog(
    config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Expand normalized M5 axes into routes and one-role comparisons."""

    routes: list[PipelineRoute] = []
    route_entries: list[dict[str, Any]] = []
    for planner in config["planner_variants"]:
        for policy in config["numeric_policies"]:
            for engine in config["engine_variants"]:
                route = build_pipeline_route(planner, policy, engine)
                routes.append(route)
                route_entries.append(
                    {
                        "planner_id": planner["id"],
                        "numeric_policy_id": policy["id"],
                        "engine_id": engine["id"],
                        **route.to_dict(),
                    }
                )

    comparisons: list[dict[str, Any]] = []
    for baseline_index, baseline in enumerate(routes):
        baseline_modules = {module.role: module for module in baseline.modules}
        for candidate in routes[baseline_index + 1 :]:
            candidate_modules = {module.role: module for module in candidate.modules}
            changed_roles = tuple(
                role
                for role in sorted(set(baseline_modules) | set(candidate_modules))
                if baseline_modules.get(role) != candidate_modules.get(role)
            )
            if len(changed_roles) != 1:
                continue
            comparison = ComparisonSpec(
                baseline_route=baseline,
                candidate_route=candidate,
                changed_roles=changed_roles,
                label=f"{changed_roles[0]}: {baseline.label} -> {candidate.label}",
            )
            comparison_payload = comparison.to_dict()
            comparison_payload["changed_roles"] = list(comparison.changed_roles)
            comparisons.append(
                {
                    "comparison_id": (
                        f"{comparison.baseline_route_id}__vs__"
                        f"{comparison.candidate_route_id}"
                    ),
                    **comparison_payload,
                }
            )
    return route_entries, comparisons


def build_pipeline_route(
    planner: Mapping[str, Any], policy: Mapping[str, Any], engine: Mapping[str, Any]
) -> PipelineRoute:
    """Describe the fixed M5 executor profile for one normalized axis tuple."""

    topology = dict(engine["topology"])
    executor_config = dict(engine["executor_config"])
    backend = str(topology["backend"])
    if backend != "cpu":
        kernel_implementation = str(
            executor_config.get("kernel_identity", "kernel_identity_unreported")
        )
        partitioner = ModuleSpec(
            "partitioner",
            "v4_output_k_tile_partitioner_v1",
            PipelineParameters(
                {
                    "partition_axes": ["output", "k"],
                    "scope": "per_contraction_task",
                }
            ),
        )
        scheduler_parameters = {
            "contraction_dag_execution": "functional_sequential_v2",
            "intra_task_execution": "v4 output/K tiles dispatched in rank waves",
            "frontier_concurrency": False,
        }
        communication = ModuleSpec(
            "communication",
            "host_managed_graph_intermediates_v1",
            PipelineParameters(
                {
                    "intermediate_residency": "host_memory",
                    "between_task_path": "host_mediated",
                    "pid_comm": False,
                }
            ),
        )
    else:
        kernel_implementation = "numpy_binary_contraction_policy_v1"
        partitioner = ModuleSpec("partitioner", "none", PipelineParameters({}))
        scheduler_parameters = {
            "contraction_dag_execution": "functional_sequential_v2",
            "frontier_concurrency": False,
        }
        communication = ModuleSpec(
            "communication",
            "host_memory_intermediates_v1",
            PipelineParameters({"intermediate_residency": "host_memory"}),
        )

    return PipelineRoute(
        route_id=f"{planner['id']}__{policy['id']}__{engine['id']}",
        label=f"{planner['label']} | {policy['label']} | {engine['label']}",
        modules=(
            ModuleSpec("tensor_network", "quantum_gate_tn_v1", PipelineParameters({})),
            ModuleSpec(
                "planner",
                str(planner["planner"]["engine"]),
                PipelineParameters(planner["planner"]),
            ),
            ModuleSpec("numeric", str(policy["policy"]), PipelineParameters({})),
            ModuleSpec(
                "executor", str(engine["engine"]), PipelineParameters(executor_config)
            ),
            ModuleSpec("topology", backend, PipelineParameters(topology)),
            ModuleSpec("kernel", kernel_implementation, PipelineParameters({})),
            partitioner,
            ModuleSpec(
                "scheduler",
                "functional_contraction_dag_sequential_v2",
                PipelineParameters(scheduler_parameters),
            ),
            communication,
        ),
    )


def validate_route_adapter(
    route: Mapping[str, Any],
    planner_variant: Mapping[str, Any],
    policy_variant: Mapping[str, Any],
    engine_variant: Mapping[str, Any],
) -> dict[str, Any]:
    """Fail closed unless a selected route equals the implemented M5 profile."""

    expected_route = build_pipeline_route(
        planner_variant, policy_variant, engine_variant
    )
    expected = expected_route.to_dict()
    actual_modules = route.get("modules")
    mismatches: list[str] = []
    if route.get("planner_id") != planner_variant["id"]:
        mismatches.append("planner_id does not match the planned TaskGraph")
    if route.get("numeric_policy_id") != policy_variant["id"]:
        mismatches.append("numeric_policy_id does not match the selected policy")
    if route.get("engine_id") != engine_variant["id"]:
        mismatches.append("engine_id does not match the selected engine")
    if route.get("route_id") != expected_route.route_id:
        mismatches.append("route_id does not match the selected planner/policy/engine")
    if route.get("route_config_hash") != expected_route.route_config_hash:
        mismatches.append("route_config_hash does not match the implemented route")
    if actual_modules != expected["modules"]:
        mismatches.append(
            "one or more route modules do not match implemented M5 behavior"
        )
    details = {
        "status": "passed" if not mismatches else "failed",
        "requested_route_id": route.get("route_id"),
        "requested_route_config_hash": route.get("route_config_hash"),
        "expected_route_id": expected_route.route_id,
        "expected_route_config_hash": expected_route.route_config_hash,
        "expected_modules": expected["modules"],
        "mismatches": mismatches,
    }
    if mismatches:
        raise RouteAdapterError(details)
    return details


def admit_route_observation(
    route: Mapping[str, Any], metadata: Mapping[str, Any], *, backend: str
) -> dict[str, Any]:
    """Compare a requested route with observed engine/native metadata."""

    modules = route["modules"]
    executor = modules["executor"]
    if backend == "cpu":
        observed_engine = metadata.get("execution_engine", metadata.get("engine"))
        expected_engine = executor["implementation"]
        passed = observed_engine == expected_engine
        return {
            "status": "passed" if passed else "failed",
            "passed": passed,
            "mode": "cpu_fixed_adapter",
            "expected": {"execution_engine": expected_engine},
            "observed": {"execution_engine": observed_engine},
            "mismatches": []
            if passed
            else ["observed execution_engine does not match requested CPU executor"],
            "reason": None
            if passed
            else "requested CPU executor does not match observed execution engine",
        }

    executor_parameters = executor["parameters"]
    expected = {
        "profile": executor_parameters.get("profile"),
        "abi": executor_parameters.get("abi", executor_parameters.get("abi_version")),
        "session_protocol": executor_parameters.get("session_protocol"),
        "dispatch_mode": executor_parameters.get("dispatch_mode"),
        "kernel_identity": modules["kernel"]["implementation"],
        "execution_class": executor_parameters.get("execution_class"),
        "graph_intermediate_placement": "host_managed",
    }
    observed = {
        "profile": metadata.get("profile", metadata.get("physical_profile")),
        "abi": metadata.get("abi", metadata.get("abi_version")),
        "session_protocol": metadata.get("session_protocol"),
        "dispatch_mode": metadata.get("dispatch_mode"),
        "kernel_identity": metadata.get("kernel_identity"),
        "execution_class": metadata.get("execution_class"),
        "graph_intermediate_placement": metadata.get("graph_intermediate_placement"),
    }
    mismatches = [
        f"{key}: expected {expected_value!r}, observed {observed[key]!r}"
        for key, expected_value in expected.items()
        if expected_value is None or observed[key] != expected_value
    ]
    passed = not mismatches
    return {
        "status": "passed" if passed else "failed",
        "passed": passed,
        "mode": "physical_observation",
        "expected": expected,
        "observed": observed,
        "mismatches": mismatches,
        "reason": None
        if passed
        else "requested physical route does not match observed native execution: "
        + "; ".join(mismatches),
    }


def select_pipeline_routes(
    config: Mapping[str, Any], route_ids: Iterable[str] | None
) -> list[dict[str, Any]]:
    """Return selected route records, preserving catalog order."""

    routes = [dict(route) for route in config.get("pipeline_routes", [])]
    if not routes:
        raise ValueError("study has no pipeline routes")
    if route_ids is None:
        return routes
    requested = [str(route_id).strip() for route_id in route_ids]
    if not requested or any(not route_id for route_id in requested):
        raise ValueError("explicit route selection cannot be empty")
    if len(set(requested)) != len(requested):
        raise ValueError(
            "explicit route selection must not contain duplicate route ids"
        )
    known = {route["route_id"] for route in routes}
    unknown = sorted(set(requested) - known)
    if unknown:
        raise ValueError(f"unknown route ids: {', '.join(unknown)}")
    selected = [route for route in routes if route["route_id"] in set(requested)]
    if not selected:
        raise ValueError("explicit route selection cannot be empty")
    return selected


def route_comparison_ids(
    config: Mapping[str, Any], route_id: str, selected_route_ids: set[str]
) -> list[str]:
    """Return admitted one-role comparisons that include one selected route."""

    return [
        str(comparison["comparison_id"])
        for comparison in config.get("pipeline_comparisons", [])
        if route_id
        in {
            comparison["baseline_route_id"],
            comparison["candidate_route_id"],
        }
        and comparison["baseline_route_id"] in selected_route_ids
        and comparison["candidate_route_id"] in selected_route_ids
    ]


def selected_pipeline_comparisons(
    config: Mapping[str, Any], selected_route_ids: set[str]
) -> list[dict[str, Any]]:
    """Return comparisons whose two routes are selected."""

    return [
        dict(comparison)
        for comparison in config.get("pipeline_comparisons", [])
        if comparison["baseline_route_id"] in selected_route_ids
        and comparison["candidate_route_id"] in selected_route_ids
    ]


__all__ = [
    "RouteAdapterError",
    "admit_route_observation",
    "apply_rank_path_override",
    "build_pipeline_catalog",
    "build_pipeline_route",
    "route_comparison_ids",
    "select_pipeline_routes",
    "selected_pipeline_comparisons",
    "validate_route_adapter",
]
