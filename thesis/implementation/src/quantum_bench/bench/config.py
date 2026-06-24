from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

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
    workloads = data.get("workloads")
    if not isinstance(workloads, list) or not workloads:
        raise ValueError("Suite schema v2 must define a non-empty workloads list")
    cases = []
    for workload in workloads:
        if not isinstance(workload, dict) or not workload.get("id"):
            raise ValueError("Every v2 workload must define id")
        case = {key: value for key, value in workload.items() if key != "id"}
        case["case_id"] = workload["id"]
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
    validation = data.get("validation") or {}
    suite["tolerances"] = validation.get("tolerances", DEFAULT_TOLERANCES)
    suite["validation"] = validation
    suite["_schema_version"] = 2
    suite["_route_configs"] = route_configs
    return suite


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
        "required": bool(entry.get("required", False)),
        "options": entry.get("options") or {},
    }


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
