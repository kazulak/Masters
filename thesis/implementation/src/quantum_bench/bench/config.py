from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from quantum_bench.bench.planner_scoring import DEFAULT_SCORING_WEIGHTS, validate_scoring_weights
from quantum_bench.validation import DEFAULT_TOLERANCES


DEFAULTS = {
    "warmups": 0,
    "repeats": 1,
    "timeout_s": None,
    "memory_guard_gib": None,
    "thread_counts": [None],
    "planner": {"engine": "opt_einsum", "optimize": "greedy"},
    "route_policy": {"routes": ["cpu_tn_einsum_exact"], "fail_fast": False},
    "tolerances": DEFAULT_TOLERANCES,
}


def load_suite(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Suite {path} must contain a YAML mapping")
    schema_version = int(data.get("schema_version", 0))
    if schema_version != 2:
        raise ValueError(f"Suite {path} must use schema_version: 2")
    suite = normalize_v2_suite(data)
    suite["suite_id"] = str(suite.get("suite_id") or path.stem)
    suite["planner"] = {**DEFAULTS["planner"], **(suite.get("planner") or {})}
    suite["route_policy"] = {**DEFAULTS["route_policy"], **(suite.get("route_policy") or {})}
    suite["tolerances"] = {**DEFAULT_TOLERANCES, **(suite.get("tolerances") or {})}
    suite["_suite_path"] = str(path)
    validate_suite(suite)
    return suite


def normalize_v2_suite(data: dict[str, Any]) -> dict[str, Any]:
    defaults = data.get("defaults") or {}
    suite = {**DEFAULTS, **defaults}
    suite["schema_version"] = 2
    suite["suite_id"] = data.get("suite_id")
    suite["metadata"] = data.get("metadata") or {}
    workloads = data.get("workloads")
    if not isinstance(workloads, list) or not workloads:
        raise ValueError("Suite schema v2 must define a non-empty workloads list")
    cases = []
    for workload in workloads:
        if not isinstance(workload, dict) or not workload.get("id"):
            raise ValueError("Every v2 workload must define id")
        case = {key: value for key, value in workload.items() if key != "id"}
        case["case_id"] = workload["id"]
        case["workload_id"] = workload["id"]
        cases.append(case)
    routes = data.get("routes")
    if not isinstance(routes, list) or not routes:
        raise ValueError("Suite schema v2 must define a non-empty routes list")
    route_configs = [_normalize_route_entry(route) for route in routes]
    suite["cases"] = cases
    suite["route_policy"] = {
        "routes": [entry["id"] for entry in route_configs],
        "fail_fast": bool(data.get("fail_fast", False)),
    }
    suite["planner_comparison"] = _normalize_planner_comparison(data.get("planner_comparison"))
    validation = data.get("validation") or {}
    suite["tolerances"] = validation.get("tolerances", DEFAULT_TOLERANCES)
    suite["validation"] = validation
    suite["_schema_version"] = 2
    suite["_route_configs"] = route_configs
    return suite


def comparison_planner_configs(suite: dict[str, Any]) -> list[dict[str, Any]]:
    comparison = suite.get("planner_comparison") or {}
    planners = comparison.get("planners") or [
        {"engine": "opt_einsum", "optimize": "greedy"},
        {"engine": "opt_einsum", "optimize": "optimal"},
    ]
    return [dict(planner) for planner in planners]


def comparison_scoring_weights(suite: dict[str, Any]) -> dict[str, float]:
    comparison = suite.get("planner_comparison") or {}
    scoring = comparison.get("scoring") if isinstance(comparison, dict) else None
    return validate_scoring_weights(scoring)


def comparison_pim_objective_config(suite: dict[str, Any]) -> dict[str, str]:
    comparison = suite.get("planner_comparison") or {}
    value = comparison.get("pim_objective") if isinstance(comparison, dict) else None
    defaults = {
        "objective_version": "upmem_path_cost_v1",
        "weight_profile": "balanced_literature_informed",
        "normalization": "fixed_log1p_generic_caps_v1",
        "execution_policy": "generic_single_dpu_float32_v1",
    }
    if value is None:
        return defaults
    if not isinstance(value, dict):
        raise ValueError("planner_comparison.pim_objective must be a mapping")
    unknown = sorted(set(value) - set(defaults))
    if unknown:
        raise ValueError(f"Unknown planner_comparison.pim_objective field(s): {', '.join(unknown)}")
    return {key: str(value.get(key, default)) for key, default in defaults.items()}


def route_config_for(suite: dict[str, Any], route_id: str) -> dict[str, Any]:
    for entry in suite.get("_route_configs", []):
        if entry["id"] == route_id:
            return entry
    return _normalize_route_entry({"id": route_id, "required": False})


def _normalize_route_entry(entry: Any) -> dict[str, Any]:
    if isinstance(entry, str):
        entry = {"id": entry}
    if not isinstance(entry, dict) or not entry.get("id"):
        raise ValueError("Route entries must define id")
    route_id = str(entry["id"])
    return {
        "id": route_id,
        "role": entry.get("role"),
        "benchmark_role": entry.get("benchmark_role"),
        "route_role_description": entry.get("route_role_description"),
        "route_limitation_scope": entry.get("route_limitation_scope"),
        "required": bool(entry.get("required", False)),
        "options": entry.get("options") or {},
    }


def _normalize_planner_comparison(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("planner_comparison must be a mapping")
    normalized: dict[str, Any] = {}
    planners = value.get("planners")
    if planners is not None:
        if not isinstance(planners, list) or not planners:
            raise ValueError("planner_comparison.planners must be a non-empty list")
        normalized_planners = []
        for planner in planners:
            if not isinstance(planner, dict):
                raise ValueError("planner_comparison.planners entries must be mappings")
            normalized_planners.append(dict(planner))
        normalized["planners"] = normalized_planners
    scoring = value.get("scoring")
    if scoring is not None:
        if not isinstance(scoring, dict):
            raise ValueError("planner_comparison.scoring must be a mapping")
        normalized["scoring"] = validate_scoring_weights(scoring)
    elif "scoring" in value:
        normalized["scoring"] = dict(DEFAULT_SCORING_WEIGHTS)
    pim_objective = value.get("pim_objective")
    if pim_objective is not None:
        if not isinstance(pim_objective, dict):
            raise ValueError("planner_comparison.pim_objective must be a mapping")
        normalized["pim_objective"] = dict(pim_objective)
    return normalized


def validate_suite(suite: dict[str, Any]) -> None:
    if "cases" not in suite or not isinstance(suite["cases"], list) or not suite["cases"]:
        raise ValueError("Suite must define a non-empty cases list")
    if int(suite["repeats"]) < 1:
        raise ValueError("Suite repeats must be >= 1")
    if int(suite["warmups"]) < 0:
        raise ValueError("Suite warmups must be >= 0")
    routes = suite["route_policy"].get("routes")
    if not isinstance(routes, list) or not routes:
        raise ValueError("route_policy.routes must be a non-empty list")
    for idx, case in enumerate(suite["cases"]):
        if not isinstance(case, dict):
            raise ValueError(f"Case {idx} must be a mapping")
        if not case.get("case_id"):
            raise ValueError(f"Case {idx} must define case_id")
        circuit = case.get("circuit")
        if not isinstance(circuit, dict) or not circuit.get("name"):
            raise ValueError(f"Case {case.get('case_id', idx)} must define circuit.name")


def suite_path(value: str, root_dir: Path) -> Path:
    candidate = Path(value)
    if candidate.exists():
        return candidate.resolve()
    preset = root_dir / "configs" / "suites" / value
    if preset.exists():
        return preset.resolve()
    preset_yml = root_dir / "configs" / "suites" / f"{value}.yml"
    if preset_yml.exists():
        return preset_yml.resolve()
    raise FileNotFoundError(f"Suite not found: {value}")
