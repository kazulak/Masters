"""Experiment-owned repetition and session lifecycle orchestration."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import hashlib
import json
from math import isfinite
import os
from pathlib import Path
import time
from types import MappingProxyType
from typing import Any
from uuid import UUID, uuid4

import numpy as np
import yaml

from quantum_bench.evidence import (
    append_sample,
    append_session,
    canonical_json,
    collection_policy_id,
    identity_hash,
    sample_id,
    validate_sample,
    validate_session,
)
from quantum_bench.results import (
    ExecutionFailed,
    ExecutionSample,
    JsonValue,
    Measurement,
    UnsupportedExecution,
)


_IDENTITY_FIELDS = {
    "problem_id",
    "tensor_network_structure_id",
    "logical_plan_id",
    "physical_plan_id",
    "executable_id",
    "environment_id",
    "validation_policy_id",
}
_REQUIRED_IDENTITY_FIELDS = {
    "problem_id",
    "environment_id",
    "validation_policy_id",
}
_MEASUREMENT_FIELDS = (
    "scope_id",
    "total_wall_s",
    "lowering_s",
    "planning_s",
    "slicing_s",
    "mapping_s",
    "session_open_s",
    "encode_s",
    "preparation_s",
    "h2d_s",
    "kernel_s",
    "host_reduce_s",
    "d2h_s",
    "decode_s",
    "rank_work_s",
    "h2d_bytes",
    "d2h_bytes",
    "energy_j",
)
_TIMING_SCOPES = frozenset({"simulation_end_to_end_v1", "steady_execution_v1"})
_VALIDATION_FIELDS = frozenset(
    {
        "policy_reference_applicable",
        "policy_reference_passed",
        "full_precision_threshold_applicable",
        "full_precision_passed",
        "accuracy_qualified",
        "max_abs_error",
        "relative_l2_error",
    }
)
_DEFAULT_VALIDATION_POLICY = {
    "policy": "complex128_reference_metrics_v1",
    "reference_dtype": "complex128",
    "float32_atol": 1.0e-5,
    "float32_rtol": 1.0e-5,
    "int8_policy_reference": "exact_raw_lane_records_v1",
    "int8_full_precision_rule": "report_error_without_universal_threshold_v1",
}
_DEFAULT_VALIDATION_POLICY_ID = identity_hash(
    "quantum_bench.validation_policy_id.v1", _DEFAULT_VALIDATION_POLICY
)

_CONFIG_SCHEMAS = frozenset({"tn_benchmark_v2", "tn_benchmark_v3"})
_CONFIG_FIELDS = frozenset(
    {
        "schema_version",
        "experiment_id",
        "defaults",
        "collection",
        "cases",
        "plans",
        "routes",
        "matrix",
    }
)
_DEFAULT_FIELDS = frozenset({"timeout_s"})
_COLLECTION_V2_FIELDS = frozenset(
    {
        "base_seed",
        "warmup_blocks",
        "measurement_blocks",
        "session_policy",
        "cooldown_s",
        "machine_policy",
    }
)
_COLLECTION_V3_FIELDS = frozenset(
    {
        "claim_policy",
        "base_seed",
        "warmup_blocks",
        "measurement_blocks",
        "session_policy",
        "block_cooldown_s",
        "machine_policy",
    }
)
_MACHINE_POLICY_FIELDS = frozenset(
    {
        "machine_exclusivity",
        "cpu_governor",
        "affinity",
        "numa_policy",
        "background_load",
    }
)
_SESSION_POLICIES = frozenset({"fresh_session_per_attempt_v1"})
_OBSERVATION_POLICIES = frozenset({"observed_v1"})
_CLAIM_POLICIES = frozenset({"diagnostic_v1", "physical_performance_v1"})
_MACHINE_POLICY_MODE_FIELDS = frozenset({"mode"})
_AFFINITY_POLICY_FIELDS = frozenset({"mode", "expected_cpus"})
_BACKGROUND_LOAD_POLICY_FIELDS = frozenset(
    {"mode", "max_load1_per_online_cpu"}
)
_MACHINE_POLICY_MODES = {
    "machine_exclusivity": frozenset({"observed_v1", "operator_attested_v1"}),
    "cpu_governor": frozenset({"observed_v1", "performance_required_v1"}),
    "affinity": frozenset({"observed_v1", "exact_required_v1"}),
    "numa_policy": frozenset({"observed_v1", "operator_attested_v1"}),
    "background_load": frozenset({"observed_v1"}),
}
_CASE_FIELDS = frozenset({"circuit"})
_CIRCUIT_FIELDS = frozenset({"kind", "name", "path", "parameters"})
_PLAN_FIELDS = frozenset({"planner", "slicing"})
_PLANNER_FIELDS = frozenset({"engine", "mode", "max_repeats", "seed"})
_SLICING_FIELDS = frozenset({"node_id", "minimum_slice_count"})
_ROUTE_FIELDS = frozenset({"executor", "numeric_policy", "options"})
_MATRIX_FIELDS = frozenset({"case_id", "plan_id", "route_ids"})
_EXECUTORS = frozenset(
    {
        "numpy_dag",
        "quimb",
        "cotengra",
        "quest_cpu",
        "quest_gpu",
        "upmem_sdk_simulator",
        "upmem_physical",
    }
)
_PLAN_REQUIRED_EXECUTORS = frozenset(
    {"numpy_dag", "upmem_sdk_simulator", "upmem_physical"}
)
_NUMERIC_POLICIES = frozenset(
    {"split_complex_float32_v1", "split_complex_int8_shared_scale_v1"}
)
_CIRCUIT_KINDS = frozenset({"builtin", "quest_compatible", "qasm_file"})
_PLANNER_ENGINES = frozenset({"opt_einsum", "cotengra"})
_OPT_EINSUM_MODES = frozenset({"greedy", "optimal"})
_COTENGRA_MODES = frozenset({"greedy", "labels"})
_ROUTE_OPTIONS = {
    "numpy_dag": frozenset(),
    "quimb": frozenset({"optimize"}),
    "cotengra": frozenset({"methods", "max_repeats"}),
    "quest_cpu": frozenset({"runner"}),
    "quest_gpu": frozenset({"verification_path"}),
    "upmem_sdk_simulator": frozenset(
        {
            "dpu_count",
            "rank_count",
            "tasklets_per_dpu",
            "session_root",
            "host_binary",
            "dpu_binary",
            "initialization_binary",
        }
    ),
    "upmem_physical": frozenset(
        {
            "dpu_count",
            "rank_count",
            "tasklets_per_dpu",
            "session_root",
            "host_binary",
            "dpu_binary",
            "initialization_binary",
            "rank_paths",
        }
    ),
}
_PATH_OPTIONS = {
    "runner",
    "verification_path",
    "session_root",
    "host_binary",
    "dpu_binary",
    "initialization_binary",
}


class _StrictSafeLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate mapping keys."""


def _strict_mapping(
    loader: _StrictSafeLoader, node: yaml.MappingNode, deep: bool = False
):
    if not isinstance(node, yaml.MappingNode):
        raise ValueError("YAML mappings must use mapping nodes")
    mapping: dict[object, object] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise ValueError(f"duplicate YAML key: {key!r}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_StrictSafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _strict_mapping
)


def _config_mapping(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a mapping")
    if any(not isinstance(key, str) for key in value):
        raise ValueError(f"{field} keys must be strings")
    return value


def _config_fields(
    value: object, expected: frozenset[str], field: str
) -> Mapping[str, object]:
    mapping = _config_mapping(value, field)
    actual = set(mapping)
    if actual != expected:
        raise ValueError(
            f"{field} fields must be exact; missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}"
        )
    return mapping


def _config_id(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a nonempty string")
    return value


def _config_string(value: object, field: str) -> str:
    return _config_id(value, field)


def _config_int(value: object, field: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{field} must be an integer >= {minimum}")
    return value


def _config_scalar(value: object, field: str) -> object:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and isfinite(value):
        return value
    raise ValueError(f"{field} must contain only finite JSON scalar values")


def _absolute_config_path(value: object, root: Path, field: str) -> str:
    path = _config_string(value, field)
    return str((root / path).resolve())


def _existing_config_file(value: object, root: Path, field: str) -> str:
    path = Path(_absolute_config_path(value, root, field))
    if not path.is_file():
        raise ValueError(f"{field} does not name an existing file: {path}")
    return str(path)


def _freeze_config(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze_config(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_config(item) for item in value)
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(
        f"unsupported normalized configuration value: {type(value).__name__}"
    )


def _load_config_yaml(path: Path) -> Mapping[str, object]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            value = yaml.load(stream, Loader=_StrictSafeLoader)
    except yaml.YAMLError as error:
        raise ValueError(f"invalid YAML configuration: {error}") from error
    if value is None:
        raise ValueError("configuration must not be empty")
    return _config_mapping(value, "configuration")


def _normalize_circuit(value: object, root: Path, field: str) -> dict[str, object]:
    circuit = dict(_config_fields(value, _CIRCUIT_FIELDS, field))
    kind = _config_string(circuit["kind"], f"{field}.kind")
    if kind not in _CIRCUIT_KINDS:
        raise ValueError(f"{field}.kind has an unsupported value: {kind}")
    name = circuit["name"]
    path = circuit["path"]
    if kind == "qasm_file":
        if name is not None:
            raise ValueError(f"{field}.name must be null for qasm_file")
        circuit["path"] = _existing_config_file(path, root, f"{field}.path")
    else:
        if path is not None:
            raise ValueError(f"{field}.path must be null for {kind}")
        circuit["name"] = _config_string(name, f"{field}.name")
    parameters = _config_mapping(circuit["parameters"], f"{field}.parameters")
    circuit["parameters"] = {
        _config_string(key, f"{field}.parameters key"): _config_scalar(
            item, f"{field}.parameters.{key}"
        )
        for key, item in parameters.items()
    }
    return circuit


def _normalize_plan(value: object, field: str) -> dict[str, object]:
    plan = dict(_config_fields(value, _PLAN_FIELDS, field))
    planner = dict(_config_fields(plan["planner"], _PLANNER_FIELDS, f"{field}.planner"))
    engine = _config_string(planner["engine"], f"{field}.planner.engine")
    if engine not in _PLANNER_ENGINES:
        raise ValueError(f"{field}.planner.engine has an unsupported value: {engine}")
    mode = _config_string(planner["mode"], f"{field}.planner.mode")
    allowed_modes = _OPT_EINSUM_MODES if engine == "opt_einsum" else _COTENGRA_MODES
    if mode not in allowed_modes:
        raise ValueError(f"{field}.planner.mode has an unsupported value: {mode}")
    planner["max_repeats"] = _config_int(
        planner["max_repeats"], f"{field}.planner.max_repeats", minimum=1
    )
    seed = planner["seed"]
    if seed is not None:
        planner["seed"] = _config_int(seed, f"{field}.planner.seed")
    slicing = plan["slicing"]
    if slicing is not None:
        slicing_map = dict(_config_fields(slicing, _SLICING_FIELDS, f"{field}.slicing"))
        slicing_map["node_id"] = _config_id(
            slicing_map["node_id"], f"{field}.slicing.node_id"
        )
        slicing_map["minimum_slice_count"] = _config_int(
            slicing_map["minimum_slice_count"],
            f"{field}.slicing.minimum_slice_count",
            minimum=2,
        )
        plan["slicing"] = slicing_map
    plan["planner"] = planner
    return plan


def _normalize_route(value: object, root: Path, field: str) -> dict[str, object]:
    route = dict(_config_fields(value, _ROUTE_FIELDS, field))
    executor = _config_string(route["executor"], f"{field}.executor")
    if executor not in _EXECUTORS:
        raise ValueError(f"{field}.executor has an unsupported value: {executor}")
    numeric_policy = route["numeric_policy"]
    if executor in _PLAN_REQUIRED_EXECUTORS:
        if numeric_policy not in _NUMERIC_POLICIES:
            raise ValueError(f"{field}.numeric_policy is required for {executor}")
    elif numeric_policy is not None:
        raise ValueError(f"{field}.numeric_policy must be null for {executor}")
    options = dict(
        _config_fields(route["options"], _ROUTE_OPTIONS[executor], f"{field}.options")
    )
    for option in _PATH_OPTIONS.intersection(options):
        options[option] = _absolute_config_path(
            options[option], root, f"{field}.options.{option}"
        )
    if "rank_paths" in options:
        paths = options["rank_paths"]
        if not isinstance(paths, list) or not paths:
            raise ValueError(f"{field}.options.rank_paths must be a nonempty list")
        options["rank_paths"] = tuple(
            _absolute_config_path(path, root, f"{field}.options.rank_paths")
            for path in paths
        )
    for option in ("dpu_count", "rank_count", "tasklets_per_dpu", "max_repeats"):
        if option in options:
            options[option] = _config_int(
                options[option], f"{field}.options.{option}", minimum=1
            )
    if "optimize" in options:
        if options["optimize"] not in _OPT_EINSUM_MODES:
            raise ValueError(f"{field}.options.optimize has an unsupported value")
    if "methods" in options:
        if options["methods"] not in _COTENGRA_MODES:
            raise ValueError(f"{field}.options.methods has an unsupported value")
    if executor in {"upmem_sdk_simulator", "upmem_physical"}:
        dpu_count = int(options["dpu_count"])
        rank_count = int(options["rank_count"])
        tasklets = int(options["tasklets_per_dpu"])
        if tasklets > 24:
            raise ValueError(f"{field}.options.tasklets_per_dpu must be <= 24")
        if executor == "upmem_sdk_simulator":
            if dpu_count != 1 or rank_count != 1:
                raise ValueError(
                    f"{field} simulator topology requires one DPU and one rank"
                )
        else:
            if dpu_count % rank_count:
                raise ValueError(
                    f"{field}.options.dpu_count must be divisible by rank_count"
                )
            if dpu_count // rank_count > 64:
                raise ValueError(f"{field} supports at most 64 DPUs per rank")
            if len(options["rank_paths"]) != rank_count:
                raise ValueError(
                    f"{field}.options.rank_paths count must equal rank_count"
                )
    route["options"] = options
    return route


def _normalize_v3_machine_policy(value: object) -> dict[str, object]:
    policy = dict(_config_fields(value, _MACHINE_POLICY_FIELDS, "collection.machine_policy"))
    normalized: dict[str, object] = {}
    for field in ("machine_exclusivity", "cpu_governor", "numa_policy"):
        entry = dict(
            _config_fields(
                policy[field], _MACHINE_POLICY_MODE_FIELDS, f"collection.machine_policy.{field}"
            )
        )
        mode = _config_string(entry["mode"], f"collection.machine_policy.{field}.mode")
        if mode not in _MACHINE_POLICY_MODES[field]:
            raise ValueError(f"collection.machine_policy.{field}.mode has an unsupported value")
        normalized[field] = {"mode": mode}

    affinity = dict(
        _config_fields(
            policy["affinity"], _AFFINITY_POLICY_FIELDS, "collection.machine_policy.affinity"
        )
    )
    affinity_mode = _config_string(affinity["mode"], "collection.machine_policy.affinity.mode")
    if affinity_mode not in _MACHINE_POLICY_MODES["affinity"]:
        raise ValueError("collection.machine_policy.affinity.mode has an unsupported value")
    expected_cpus = affinity["expected_cpus"]
    if expected_cpus is not None:
        if not isinstance(expected_cpus, list) or not expected_cpus:
            raise ValueError("collection.machine_policy.affinity.expected_cpus must be a nonempty list or null")
        normalized_cpus = tuple(
            _config_int(cpu, "collection.machine_policy.affinity.expected_cpus", minimum=0)
            for cpu in expected_cpus
        )
        if len(set(normalized_cpus)) != len(normalized_cpus):
            raise ValueError("collection.machine_policy.affinity.expected_cpus must be unique")
        expected_cpus = normalized_cpus
    if affinity_mode == "observed_v1" and expected_cpus is not None:
        raise ValueError("observed affinity policy requires null expected_cpus")
    if affinity_mode == "exact_required_v1" and expected_cpus is None:
        raise ValueError("exact affinity policy requires expected_cpus")
    normalized["affinity"] = {"mode": affinity_mode, "expected_cpus": expected_cpus}

    background = dict(
        _config_fields(
            policy["background_load"],
            _BACKGROUND_LOAD_POLICY_FIELDS,
            "collection.machine_policy.background_load",
        )
    )
    background_mode = _config_string(
        background["mode"], "collection.machine_policy.background_load.mode"
    )
    if background_mode not in _MACHINE_POLICY_MODES["background_load"]:
        raise ValueError("collection.machine_policy.background_load.mode has an unsupported value")
    threshold = background["max_load1_per_online_cpu"]
    if threshold is not None:
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
            raise ValueError("collection.machine_policy.background_load.max_load1_per_online_cpu must be finite and >= 0")
        if not isfinite(float(threshold)) or float(threshold) < 0:
            raise ValueError("collection.machine_policy.background_load.max_load1_per_online_cpu must be finite and >= 0")
        threshold = float(threshold)
    normalized["background_load"] = {
        "mode": background_mode,
        "max_load1_per_online_cpu": threshold,
    }
    return normalized


def _collection_policy_id_v1(collection: Mapping[str, object]) -> str:
    return collection_policy_id(collection, configuration_schema="tn_benchmark_v2")


def _collection_policy_id_v2(collection: Mapping[str, object]) -> str:
    return collection_policy_id(collection, configuration_schema="tn_benchmark_v3")


def _experiment_id_v2(payload: Mapping[str, object]) -> str:
    return identity_hash("quantum_bench.experiment_id.v2", payload)


def _experiment_id_v3(payload: Mapping[str, object]) -> str:
    return identity_hash("quantum_bench.experiment_id.v3", payload)


def load_experiment_config(path: str | os.PathLike[str]) -> Mapping[str, object]:
    """Load and strictly validate one immutable v2 or v3 benchmark config."""

    config_path = Path(path)
    if not config_path.exists() or not config_path.is_file():
        raise ValueError(f"configuration path does not name a file: {config_path}")
    root = config_path.resolve().parent
    raw = _load_config_yaml(config_path)
    config = dict(_config_fields(raw, _CONFIG_FIELDS, "configuration"))
    schema_version = config["schema_version"]
    if schema_version not in _CONFIG_SCHEMAS:
        raise ValueError("configuration has an invalid schema_version")
    experiment_label = _config_id(config["experiment_id"], "experiment_id")
    defaults = dict(_config_fields(config["defaults"], _DEFAULT_FIELDS, "defaults"))
    timeout_s = defaults["timeout_s"]
    if isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float)):
        raise ValueError("defaults.timeout_s must be finite and > 0")
    if not isfinite(float(timeout_s)) or float(timeout_s) <= 0:
        raise ValueError("defaults.timeout_s must be finite and > 0")
    config["defaults"] = defaults

    collection = dict(
        _config_fields(
            config["collection"],
            _COLLECTION_V2_FIELDS if schema_version == "tn_benchmark_v2" else _COLLECTION_V3_FIELDS,
            "collection",
        )
    )
    collection["base_seed"] = _config_int(
        collection["base_seed"], "collection.base_seed"
    )
    collection["warmup_blocks"] = _config_int(
        collection["warmup_blocks"], "collection.warmup_blocks"
    )
    collection["measurement_blocks"] = _config_int(
        collection["measurement_blocks"],
        "collection.measurement_blocks",
        minimum=1,
    )
    collection["session_policy"] = _config_string(
        collection["session_policy"], "collection.session_policy"
    )
    if collection["session_policy"] not in _SESSION_POLICIES:
        raise ValueError("collection.session_policy has an unsupported value")
    cooldown_field = "cooldown_s" if schema_version == "tn_benchmark_v2" else "block_cooldown_s"
    cooldown_s = collection[cooldown_field]
    if isinstance(cooldown_s, bool) or not isinstance(cooldown_s, (int, float)):
        raise ValueError(f"collection.{cooldown_field} must be finite and >= 0")
    if not isfinite(float(cooldown_s)) or float(cooldown_s) < 0:
        raise ValueError(f"collection.{cooldown_field} must be finite and >= 0")
    collection[cooldown_field] = float(cooldown_s)
    if schema_version == "tn_benchmark_v2":
        machine_policy = dict(
            _config_fields(
                collection["machine_policy"],
                _MACHINE_POLICY_FIELDS,
                "collection.machine_policy",
            )
        )
        for field, value in machine_policy.items():
            machine_policy[field] = _config_string(
                value, f"collection.machine_policy.{field}"
            )
            if machine_policy[field] not in _OBSERVATION_POLICIES:
                raise ValueError(
                    f"collection.machine_policy.{field} has an unsupported value"
                )
        collection["machine_policy"] = machine_policy
    else:
        collection["claim_policy"] = _config_string(
            collection["claim_policy"], "collection.claim_policy"
        )
        if collection["claim_policy"] not in _CLAIM_POLICIES:
            raise ValueError("collection.claim_policy has an unsupported value")
        collection["machine_policy"] = _normalize_v3_machine_policy(
            collection["machine_policy"]
        )
        if collection["claim_policy"] == "physical_performance_v1":
            machine_policy = collection["machine_policy"]
            if collection["warmup_blocks"] != 2 or collection["measurement_blocks"] != 30:
                raise ValueError("physical_performance_v1 requires two warmup and 30 measurement blocks")
            if machine_policy["machine_exclusivity"]["mode"] != "operator_attested_v1":
                raise ValueError("physical_performance_v1 requires operator-attested exclusivity")
            if machine_policy["cpu_governor"]["mode"] != "performance_required_v1":
                raise ValueError("physical_performance_v1 requires the performance CPU governor")
            if machine_policy["affinity"]["mode"] != "exact_required_v1":
                raise ValueError("physical_performance_v1 requires exact affinity")
            if machine_policy["numa_policy"]["mode"] != "operator_attested_v1":
                raise ValueError("physical_performance_v1 requires operator-attested NUMA policy")
            if machine_policy["background_load"]["max_load1_per_online_cpu"] is None:
                raise ValueError("physical_performance_v1 requires a background-load threshold")
    config["collection"] = collection
    config["collection_policy_id"] = (
        _collection_policy_id_v1(collection)
        if schema_version == "tn_benchmark_v2"
        else _collection_policy_id_v2(collection)
    )

    cases_raw = _config_mapping(config["cases"], "cases")
    if not cases_raw:
        raise ValueError("cases must be nonempty")
    cases: dict[str, object] = {}
    for case_id, case in cases_raw.items():
        case_key = _config_id(case_id, "case id")
        case_map = _config_fields(case, _CASE_FIELDS, f"cases.{case_key}")
        cases[case_key] = {
            "circuit": _normalize_circuit(
                case_map["circuit"], root, f"cases.{case_key}.circuit"
            )
        }
    config["cases"] = cases

    plans_raw = _config_mapping(config["plans"], "plans")
    plans = {
        _config_id(plan_id, "plan id"): _normalize_plan(plan, f"plans.{plan_id}")
        for plan_id, plan in plans_raw.items()
    }
    config["plans"] = plans

    routes_raw = _config_mapping(config["routes"], "routes")
    if not routes_raw:
        raise ValueError("routes must be nonempty")
    routes = {
        _config_id(route_id, "route id"): _normalize_route(
            route, root, f"routes.{route_id}"
        )
        for route_id, route in routes_raw.items()
    }
    config["routes"] = routes

    matrix_raw = config["matrix"]
    if not isinstance(matrix_raw, list) or not matrix_raw:
        raise ValueError("matrix must be a nonempty list")
    matrix: list[dict[str, object]] = []
    selected_case_plan_routes: set[tuple[str, str | None, str]] = set()
    for index, item in enumerate(matrix_raw):
        entry = dict(_config_fields(item, _MATRIX_FIELDS, f"matrix[{index}]"))
        case_id = _config_id(entry["case_id"], f"matrix[{index}].case_id")
        if case_id not in cases:
            raise ValueError(f"matrix[{index}] references unknown case_id")
        route_ids = entry["route_ids"]
        if not isinstance(route_ids, list) or not route_ids:
            raise ValueError(f"matrix[{index}].route_ids must be a nonempty list")
        normalized_routes = tuple(
            _config_id(route_id, f"matrix[{index}].route_ids") for route_id in route_ids
        )
        if len(set(normalized_routes)) != len(normalized_routes):
            raise ValueError(f"matrix[{index}].route_ids must be unique")
        if any(route_id not in routes for route_id in normalized_routes):
            raise ValueError(f"matrix[{index}] references an unknown route_id")
        plan_id = entry["plan_id"]
        if plan_id is not None:
            plan_id = _config_id(plan_id, f"matrix[{index}].plan_id")
            if plan_id not in plans:
                raise ValueError(f"matrix[{index}] references an unknown plan_id")
        requires_plan = tuple(
            routes[route_id]["executor"] in _PLAN_REQUIRED_EXECUTORS
            for route_id in normalized_routes
        )
        if any(requires_plan) and not all(requires_plan):
            raise ValueError(
                f"matrix[{index}] cannot mix plan-required and planless routes"
            )
        if any(requires_plan) != (plan_id is not None):
            raise ValueError(f"matrix[{index}].plan_id is incompatible with its routes")
        for route_id in normalized_routes:
            key = (case_id, plan_id, route_id)
            if key in selected_case_plan_routes:
                raise ValueError(
                    "matrix entries must select each case_id, plan_id, and route_id "
                    "combination once"
                )
            selected_case_plan_routes.add(key)
        matrix.append(
            {"case_id": case_id, "plan_id": plan_id, "route_ids": normalized_routes}
        )
    config["matrix"] = matrix
    experiment_identity_payload = {
        "label": experiment_label,
        "configuration": raw,
        "validation_policy_id": _DEFAULT_VALIDATION_POLICY_ID,
    }
    config["experiment_identity_payload"] = experiment_identity_payload
    config["experiment_id"] = (
        _experiment_id_v2(experiment_identity_payload)
        if schema_version == "tn_benchmark_v2"
        else _experiment_id_v3(experiment_identity_payload)
    )
    frozen = _freeze_config(config)
    if not isinstance(frozen, Mapping):  # pragma: no cover
        raise TypeError("normalized configuration must be a mapping")
    return frozen


def default_validation_policy() -> Mapping[str, JsonValue]:
    """Return the one immutable validation policy used by reset experiments."""

    frozen = _freeze_config(_DEFAULT_VALIDATION_POLICY)
    if not isinstance(frozen, Mapping):  # pragma: no cover - constant is a mapping.
        raise TypeError("default validation policy must be a mapping")
    return frozen


def default_validation_policy_id() -> str:
    """Return the canonical identity of the reset validation policy."""

    return _DEFAULT_VALIDATION_POLICY_ID


def _attempt_schedule(
    *,
    warmups: int,
    repetitions: int,
    attempts: tuple[tuple[str, int, int, int], ...] | None,
) -> tuple[tuple[str, int, int, int], ...]:
    """Normalize ``(kind, index, block, order)`` attempt tuples."""

    if attempts is None:
        attempts = tuple(
            (kind, index, block_id, 0)
            for kind, count, block_offset in (
                ("warmup", warmups, 0),
                ("measurement", repetitions, warmups),
            )
            for index in range(count)
            for block_id in (block_offset + index,)
        )
    if not isinstance(attempts, tuple):
        raise TypeError("attempts must be a tuple of attempt tuples or None")
    normalized: list[tuple[str, int, int, int]] = []
    for attempt in attempts:
        if not isinstance(attempt, tuple) or len(attempt) != 4:
            raise TypeError("attempts must contain (kind, index, block, order) tuples")
        kind, sample_index, block_id, order_index = attempt
        if kind not in {"warmup", "measurement"}:
            raise ValueError("attempt kind must be warmup or measurement")
        for value, field in (
            (sample_index, "attempt sample index"),
            (block_id, "attempt block ID"),
            (order_index, "attempt order index"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{field} must be a non-negative integer")
        normalized.append((kind, sample_index, block_id, order_index))
    if len(set(normalized)) != len(normalized):
        raise ValueError("attempts must not contain duplicate attempt tuples")
    return tuple(normalized)


def run_direct_samples(
    *,
    run_id: str,
    experiment_id: str,
    case_id: str,
    route_id: str,
    plan_id: str | None = None,
    identities: Mapping[str, JsonValue],
    warmups: int,
    repetitions: int,
    run_once: Callable[[], ExecutionSample],
    samples_path: str | os.PathLike[str],
    validate: Callable[[ExecutionSample], Mapping[str, JsonValue]] | None = None,
    attempts: tuple[tuple[str, int, int, int], ...] | None = None,
) -> tuple[Mapping[str, JsonValue], ...]:
    """Run and append all warmup and measurement samples for a direct route."""

    normalized_identities = _validate_arguments(
        run_id=run_id,
        experiment_id=experiment_id,
        case_id=case_id,
        route_id=route_id,
        plan_id=plan_id,
        identities=identities,
        warmups=warmups,
        repetitions=repetitions,
        samples_path=samples_path,
        run_once=run_once,
    )
    schedule = _attempt_schedule(
        warmups=warmups, repetitions=repetitions, attempts=attempts
    )
    _reject_planned_sample_id_collisions(
        samples_path, _planned_sample_ids(run_id, case_id, route_id, plan_id, schedule)
    )

    rows: list[Mapping[str, JsonValue]] = []
    for attempt_kind, sample_index, block_id, order_index in schedule:
        row = _run_sample(
            run_id=run_id,
            experiment_id=experiment_id,
            case_id=case_id,
            route_id=route_id,
            plan_id=plan_id,
            identities=normalized_identities,
            attempt_kind=attempt_kind,
            sample_index=sample_index,
            block_id=block_id,
            order_index=order_index,
            session_instance_id=None,
            invoke=lambda: run_once(),
            validate=validate,
        )
        append_sample(samples_path, row)
        rows.append(row)
    return tuple(rows)


def run_session_samples(
    *,
    run_id: str,
    experiment_id: str,
    case_id: str,
    route_id: str,
    plan_id: str | None = None,
    identities: Mapping[str, JsonValue],
    warmups: int,
    repetitions: int,
    session_protocol_id: str,
    open_session: Callable[[], Any],
    inputs: Mapping[str, Any],
    samples_path: str | os.PathLike[str],
    sessions_path: str | os.PathLike[str],
    validate: Callable[[ExecutionSample], Mapping[str, JsonValue]] | None = None,
    attempts: tuple[tuple[str, int, int, int], ...] | None = None,
) -> tuple[tuple[Mapping[str, JsonValue], ...], Mapping[str, JsonValue]]:
    """Run samples on one persistent session and append its lifecycle record."""

    normalized_identities = _validate_arguments(
        run_id=run_id,
        experiment_id=experiment_id,
        case_id=case_id,
        route_id=route_id,
        plan_id=plan_id,
        identities=identities,
        warmups=warmups,
        repetitions=repetitions,
        samples_path=samples_path,
        sessions_path=sessions_path,
        session_protocol_id=session_protocol_id,
        open_session=open_session,
        inputs=inputs,
    )
    schedule = _attempt_schedule(
        warmups=warmups, repetitions=repetitions, attempts=attempts
    )
    _reject_planned_sample_id_collisions(
        samples_path, _planned_sample_ids(run_id, case_id, route_id, plan_id, schedule)
    )

    session_instance_id = str(uuid4())
    if session_instance_id in _existing_session_ids(sessions_path):
        raise ValueError(
            "generated session_instance_id already exists in sessions_path"
        )
    open_started = time.perf_counter()
    try:
        session = open_session()
    except UnsupportedExecution as exc:
        open_s = time.perf_counter() - open_started
        session_row = _session_row(
            run_id=run_id,
            experiment_id=experiment_id,
            case_id=case_id,
            route_id=route_id,
            plan_id=plan_id,
            session_instance_id=session_instance_id,
            session_protocol_id=session_protocol_id,
            open_s=open_s,
            session_close_s=None,
            terminal_backend_facts={},
            failure={"stage": exc.stage, "reason": exc.reason},
        )
        append_session(sessions_path, session_row)
        return _session_open_failure_rows(
            run_id=run_id,
            experiment_id=experiment_id,
            case_id=case_id,
            route_id=route_id,
            plan_id=plan_id,
            identities=normalized_identities,
            session_instance_id=session_instance_id,
            failure=exc,
            schedule=schedule,
            samples_path=samples_path,
        ), session_row
    except ExecutionFailed as exc:
        open_s = time.perf_counter() - open_started
        session_row = _session_row(
            run_id=run_id,
            experiment_id=experiment_id,
            case_id=case_id,
            route_id=route_id,
            plan_id=plan_id,
            session_instance_id=session_instance_id,
            session_protocol_id=session_protocol_id,
            open_s=open_s,
            session_close_s=None,
            terminal_backend_facts=exc.backend_facts,
            failure={"stage": exc.stage, "reason": exc.reason},
        )
        append_session(sessions_path, session_row)
        return _session_open_failure_rows(
            run_id=run_id,
            experiment_id=experiment_id,
            case_id=case_id,
            route_id=route_id,
            plan_id=plan_id,
            identities=normalized_identities,
            session_instance_id=session_instance_id,
            failure=exc,
            schedule=schedule,
            samples_path=samples_path,
        ), session_row
    except Exception as exc:
        open_s = time.perf_counter() - open_started
        session_row = _session_row(
            run_id=run_id,
            experiment_id=experiment_id,
            case_id=case_id,
            route_id=route_id,
            plan_id=plan_id,
            session_instance_id=session_instance_id,
            session_protocol_id=session_protocol_id,
            open_s=open_s,
            session_close_s=None,
            terminal_backend_facts={},
            failure={"stage": "session_open", "reason": _unexpected_reason(exc)},
        )
        append_session(sessions_path, session_row)
        return _session_open_failure_rows(
            run_id=run_id,
            experiment_id=experiment_id,
            case_id=case_id,
            route_id=route_id,
            plan_id=plan_id,
            identities=normalized_identities,
            session_instance_id=session_instance_id,
            failure=ExecutionFailed("session_open", _unexpected_reason(exc), {}),
            schedule=schedule,
            samples_path=samples_path,
        ), session_row

    open_s = time.perf_counter() - open_started
    rows: list[Mapping[str, JsonValue]] = []
    sample_failure: Mapping[str, JsonValue] | None = None
    interface_failure: Mapping[str, JsonValue] | None = None
    try:
        run_method = getattr(session, "run_once", None)
    except Exception as exc:
        run_method = None
        interface_failure = {
            "stage": "session_open",
            "reason": _unexpected_reason(exc),
        }
    try:
        close_method = getattr(session, "close", None)
    except Exception as exc:
        close_method = None
        interface_failure = {
            "stage": "session_open",
            "reason": _unexpected_reason(exc),
        }
    if interface_failure is None and not callable(run_method):
        interface_failure = {
            "stage": "session_open",
            "reason": "opened session must expose callable run_once",
        }
    if interface_failure is None and not callable(close_method):
        interface_failure = {
            "stage": "session_open",
            "reason": "opened session must expose callable close",
        }

    try:
        if interface_failure is None:
            for attempt_kind, sample_index, block_id, order_index in schedule:
                row = _run_sample(
                    run_id=run_id,
                    experiment_id=experiment_id,
                    case_id=case_id,
                    route_id=route_id,
                    plan_id=plan_id,
                    identities=normalized_identities,
                    attempt_kind=attempt_kind,
                    sample_index=sample_index,
                    block_id=block_id,
                    order_index=order_index,
                    session_instance_id=session_instance_id,
                    invoke=lambda: run_method(inputs),
                    persistent_session=True,
                    validate=validate,
                )
                append_sample(samples_path, row)
                rows.append(row)
                if row["status"] != "success":
                    failure = row["failure"]
                    if not isinstance(failure, Mapping):  # pragma: no cover
                        raise TypeError("failed sample has no failure mapping")
                    sample_failure = {
                        "stage": failure["stage"],
                        "reason": failure["reason"],
                    }
                    break
    finally:
        close_failure: Mapping[str, JsonValue] | None = None
        terminal_backend_facts: Mapping[str, JsonValue] = {}
        session_close_s: float | None = None
        if callable(close_method):
            close_started = time.perf_counter()
            try:
                terminal_backend_facts = _plain_json(close_method())
                if not isinstance(terminal_backend_facts, Mapping):
                    raise TypeError("session close must return a mapping")
            except ExecutionFailed as exc:
                terminal_backend_facts = _plain_json(exc.backend_facts)
                close_failure = {"stage": exc.stage, "reason": exc.reason}
            except Exception as exc:
                terminal_backend_facts = {}
                close_failure = {
                    "stage": "session_close",
                    "reason": _unexpected_reason(exc),
                }
            session_close_s = time.perf_counter() - close_started
        else:
            close_failure = {
                "stage": "session_close",
                "reason": "opened session must expose callable close",
            }

        (
            release_attempted,
            release_succeeded,
            release_verified,
            release_inconsistent,
        ) = _release_facts(terminal_backend_facts)

        if close_failure is not None:
            failure = close_failure
        elif release_inconsistent:
            failure = {
                "stage": "session_close",
                "reason": "hardware release facts are inconsistent",
            }
        elif interface_failure is not None:
            failure = interface_failure
        elif sample_failure is not None:
            failure = sample_failure
        elif not (release_attempted and release_succeeded and release_verified):
            failure = {
                "stage": "session_close",
                "reason": "hardware release was not fully verified",
            }
        else:
            failure = None

        session_row = _session_row(
            run_id=run_id,
            experiment_id=experiment_id,
            case_id=case_id,
            route_id=route_id,
            plan_id=plan_id,
            session_instance_id=session_instance_id,
            session_protocol_id=session_protocol_id,
            open_s=open_s,
            session_close_s=session_close_s,
            terminal_backend_facts=terminal_backend_facts,
            release_attempted=release_attempted,
            release_succeeded=release_succeeded,
            release_verified=release_verified,
            failure=failure,
        )
        append_session(sessions_path, session_row)

    return tuple(rows), session_row


def _session_open_failure_rows(
    *,
    run_id: str,
    experiment_id: str,
    case_id: str,
    route_id: str,
    plan_id: str | None,
    identities: Mapping[str, JsonValue],
    session_instance_id: str,
    failure: UnsupportedExecution | ExecutionFailed,
    schedule: tuple[tuple[str, int, int, int], ...],
    samples_path: str | os.PathLike[str],
) -> tuple[Mapping[str, JsonValue], ...]:
    """Emit one failed row when a fresh session cannot open for an attempt."""

    if not schedule:
        return ()
    attempt_kind, sample_index, block_id, order_index = schedule[0]

    def raise_failure() -> ExecutionSample:
        raise failure

    row = _run_sample(
        run_id=run_id,
        experiment_id=experiment_id,
        case_id=case_id,
        route_id=route_id,
        plan_id=plan_id,
        identities=identities,
        attempt_kind=attempt_kind,
        sample_index=sample_index,
        block_id=block_id,
        order_index=order_index,
        session_instance_id=session_instance_id,
        invoke=raise_failure,
    )
    append_sample(samples_path, row)
    return (row,)


def _validate_arguments(**values: Any) -> dict[str, JsonValue]:
    _canonical_uuid4(values["run_id"], "run_id")
    _sha256_string(values["experiment_id"], "experiment_id")
    for field in ("case_id", "route_id"):
        _nonempty_string(values[field], field)
    if "plan_id" in values and values["plan_id"] is not None:
        _nonempty_string(values["plan_id"], "plan_id")
    if "session_protocol_id" in values:
        _nonempty_string(values["session_protocol_id"], "session_protocol_id")
    for field in ("warmups", "repetitions"):
        value = values[field]
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{field} must be a non-negative integer")
        if value < 0:
            raise ValueError(f"{field} must be a non-negative integer")
    for field in ("run_once", "open_session"):
        if field in values and not callable(values[field]):
            raise TypeError(f"{field} must be callable")
    if "inputs" in values and not isinstance(values["inputs"], Mapping):
        raise TypeError("inputs must be a mapping")
    for field in ("samples_path", "sessions_path"):
        if field in values and not isinstance(values[field], (str, os.PathLike)):
            raise TypeError(f"{field} must be path-like")

    identities = values["identities"]
    if not isinstance(identities, Mapping):
        raise TypeError("identities must be a mapping")
    if set(identities) != _IDENTITY_FIELDS:
        raise ValueError("identities must match the evidence identity schema exactly")
    normalized = _plain_json(identities)
    if not isinstance(normalized, dict):  # pragma: no cover - guarded above
        raise TypeError("identities must be a mapping")
    for field in _REQUIRED_IDENTITY_FIELDS:
        _sha256_string(normalized[field], f"identities.{field}")
    tensor_network_id = normalized["tensor_network_structure_id"]
    logical_plan_id = normalized["logical_plan_id"]
    if (tensor_network_id is None) != (logical_plan_id is None):
        raise ValueError(
            "tensor_network_structure_id and logical_plan_id must be null together"
        )
    if tensor_network_id is not None:
        _sha256_string(tensor_network_id, "identities.tensor_network_structure_id")
    if logical_plan_id is not None:
        _sha256_string(logical_plan_id, "identities.logical_plan_id")
    for field in ("physical_plan_id", "executable_id"):
        if normalized[field] is not None:
            _sha256_string(normalized[field], f"identities.{field}")
    return normalized


def _run_sample(
    *,
    run_id: str,
    experiment_id: str,
    case_id: str,
    route_id: str,
    plan_id: str | None,
    identities: Mapping[str, JsonValue],
    attempt_kind: str,
    sample_index: int,
    block_id: int,
    order_index: int,
    session_instance_id: str | None,
    invoke: Callable[[], Any],
    persistent_session: bool = False,
    validate: Callable[[ExecutionSample], Mapping[str, JsonValue]] | None = None,
) -> Mapping[str, JsonValue]:
    base: dict[str, JsonValue] = {
        "schema_version": "evidence_sample_v3",
        "sample_id": sample_id(
            run_id,
            case_id,
            route_id,
            attempt_kind,
            sample_index,
            plan_id=plan_id,
            block_id=block_id,
            order_index=order_index,
        ),
        "run_id": run_id,
        "experiment_id": experiment_id,
        "case_id": case_id,
        "plan_id": plan_id,
        "route_id": route_id,
        "attempt_kind": attempt_kind,
        "sample_index": sample_index,
        "block_id": block_id,
        "order_index": order_index,
        "observed_affinity": _observed_affinity(),
        "background_load_1m": _background_load_1m(),
        "session_instance_id": session_instance_id,
        "identities": identities,
        "validation": None,
    }
    try:
        sample = invoke()
        if not isinstance(sample, ExecutionSample):
            raise TypeError("run_once must return ExecutionSample")
        if sample.measurement.scope_id not in _TIMING_SCOPES:
            return {
                **base,
                "status": "failed",
                "measurement": None,
                "backend_facts": {},
                "numeric_facts": {},
                "output_sha256": None,
                "failure": {
                    "stage": "timing_contract",
                    "reason": "samples require a frozen timing scope",
                },
            }
        if persistent_session and not _has_steady_session_timing(sample.measurement):
            return {
                **base,
                "status": "failed",
                "measurement": None,
                "backend_facts": {},
                "numeric_facts": {},
                "output_sha256": None,
                "failure": {
                    "stage": "timing_contract",
                    "reason": "persistent session samples require steady_execution_v1 timing",
                },
            }
        measurement = _measurement_mapping(sample.measurement)
        backend_facts = _plain_json(sample.backend_facts)
        numeric_facts = _plain_json(sample.numeric_facts)
        output_sha256 = _output_hash(sample.output)
        validation: Mapping[str, JsonValue] | None = None
        if validate is not None:
            try:
                validation = _validation_mapping(validate(sample))
            except Exception as exc:
                return {
                    **base,
                    "status": "failed",
                    "measurement": None,
                    "backend_facts": backend_facts,
                    "numeric_facts": numeric_facts,
                    "output_sha256": None,
                    "failure": {
                        "stage": "validation",
                        "reason": _validation_failure_reason(exc),
                    },
                }
        return {
            **base,
            "status": "success",
            "measurement": measurement,
            "backend_facts": backend_facts,
            "numeric_facts": numeric_facts,
            "output_sha256": output_sha256,
            "validation": validation,
            "failure": None,
        }
    except UnsupportedExecution as exc:
        return {
            **base,
            "status": "unsupported",
            "measurement": None,
            "backend_facts": {},
            "numeric_facts": {},
            "output_sha256": None,
            "failure": {
                "stage": exc.stage,
                "reason": exc.reason,
                "capability": exc.capability,
            },
        }
    except ExecutionFailed as exc:
        return {
            **base,
            "status": "failed",
            "measurement": None,
            "backend_facts": _plain_json(exc.backend_facts),
            "numeric_facts": {},
            "output_sha256": None,
            "failure": {"stage": exc.stage, "reason": exc.reason},
        }
    except Exception as exc:
        return {
            **base,
            "status": "failed",
            "measurement": None,
            "backend_facts": {},
            "numeric_facts": {},
            "output_sha256": None,
            "failure": {"stage": "execution", "reason": _unexpected_reason(exc)},
        }


def _observed_affinity() -> list[int] | None:
    if not hasattr(os, "sched_getaffinity"):
        return None
    try:
        return sorted(os.sched_getaffinity(0))
    except OSError:
        return None


def _background_load_1m() -> float | None:
    try:
        value = float(os.getloadavg()[0])
    except (AttributeError, OSError):
        return None
    return value if isfinite(value) and value >= 0.0 else None


def _measurement_mapping(measurement: Measurement) -> Mapping[str, JsonValue]:
    return {field: getattr(measurement, field) for field in _MEASUREMENT_FIELDS}


def _validation_mapping(
    value: Mapping[str, JsonValue],
) -> Mapping[str, JsonValue]:
    if not isinstance(value, Mapping):
        raise TypeError("validation must be a mapping")
    if set(value) != _VALIDATION_FIELDS:
        raise ValueError("validation fields must match the validation schema exactly")

    normalized = _plain_json(value)
    if not isinstance(normalized, dict):  # pragma: no cover - guarded above
        raise TypeError("validation must be a mapping")
    for field in (
        "policy_reference_applicable",
        "full_precision_threshold_applicable",
        "accuracy_qualified",
    ):
        if not isinstance(normalized[field], bool):
            raise TypeError(f"validation.{field} must be a boolean")
    for applicable, passed in (
        ("policy_reference_applicable", "policy_reference_passed"),
        ("full_precision_threshold_applicable", "full_precision_passed"),
    ):
        if normalized[applicable]:
            if not isinstance(normalized[passed], bool):
                raise TypeError(
                    f"validation.{passed} must be a boolean when applicable"
                )
        elif normalized[passed] is not None:
            raise ValueError(f"validation.{passed} must be null when not applicable")
    if not (
        normalized["policy_reference_applicable"]
        or normalized["full_precision_threshold_applicable"]
    ):
        raise ValueError("validation must include at least one applicable comparison")
    expected_accuracy = (
        normalized["full_precision_threshold_applicable"]
        and normalized["full_precision_passed"] is True
    )
    if normalized["accuracy_qualified"] != expected_accuracy:
        raise ValueError(
            "validation.accuracy_qualified must equal full-precision qualification"
        )
    for field in ("max_abs_error", "relative_l2_error"):
        error = normalized[field]
        if isinstance(error, bool) or not isinstance(error, (int, float)):
            raise TypeError(f"validation.{field} must be a finite non-negative number")
        if not isfinite(float(error)) or float(error) < 0.0:
            raise ValueError(f"validation.{field} must be a finite non-negative number")
        normalized[field] = float(error)
    return normalized


def _validation_failure_reason(exc: Exception) -> str:
    message = " ".join(str(exc).split())
    reason = f"validator error: {type(exc).__name__}"
    if message:
        reason = f"{reason}: {message}"
    return reason[:256]


def _has_steady_session_timing(measurement: Measurement) -> bool:
    return measurement.scope_id == "steady_execution_v1" and all(
        getattr(measurement, field) is None
        for field in (
            "lowering_s",
            "planning_s",
            "slicing_s",
            "mapping_s",
            "session_open_s",
        )
    )


def _planned_sample_ids(
    run_id: str,
    case_id: str,
    route_id: str,
    plan_id: str | None,
    attempts: tuple[tuple[str, int, int, int], ...],
) -> frozenset[str]:
    return frozenset(
        sample_id(
            run_id,
            case_id,
            route_id,
            attempt_kind,
            sample_index,
            plan_id=plan_id,
            block_id=block_id,
            order_index=order_index,
        )
        for attempt_kind, sample_index, block_id, order_index in attempts
    )


def _reject_planned_sample_id_collisions(
    samples_path: str | os.PathLike[str], planned_ids: frozenset[str]
) -> None:
    if not planned_ids:
        return
    collisions = _existing_sample_ids(samples_path).intersection(planned_ids)
    if collisions:
        raise ValueError("planned sample IDs already exist in samples_path")


def _existing_sample_ids(path: str | os.PathLike[str]) -> set[str]:
    target = Path(path)
    if not target.exists():
        return set()
    text = target.read_text(encoding="utf-8")
    if not text:
        return set()
    if not text.endswith("\n"):
        raise ValueError("existing samples JSONL must be newline-terminated")

    sample_ids: set[str] = set()
    for index, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            raise ValueError("existing samples JSONL must not contain blank lines")
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"existing samples JSONL line {index} is invalid JSON"
            ) from error
        if not isinstance(record, Mapping):
            raise TypeError(f"existing samples JSONL line {index} must be a mapping")
        validate_sample(record)
        current_id = record["sample_id"]
        if current_id in sample_ids:
            raise ValueError("existing samples JSONL contains duplicate sample_id")
        sample_ids.add(current_id)
    return sample_ids


def _existing_session_ids(path: str | os.PathLike[str]) -> set[str]:
    target = Path(path)
    if not target.exists():
        return set()
    text = target.read_text(encoding="utf-8")
    if not text:
        return set()
    if not text.endswith("\n"):
        raise ValueError("existing sessions JSONL must be newline-terminated")

    session_ids: set[str] = set()
    for index, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            raise ValueError("existing sessions JSONL must not contain blank lines")
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"existing sessions JSONL line {index} is invalid JSON"
            ) from error
        if not isinstance(record, Mapping):
            raise TypeError(f"existing sessions JSONL line {index} must be a mapping")
        validate_session(record)
        current_id = record["session_instance_id"]
        if current_id in session_ids:
            raise ValueError(
                "existing sessions JSONL contains duplicate session_instance_id"
            )
        session_ids.add(current_id)
    return session_ids


def _session_row(
    *,
    run_id: str,
    experiment_id: str,
    case_id: str,
    route_id: str,
    plan_id: str | None,
    session_instance_id: str,
    session_protocol_id: str,
    open_s: float | None,
    session_close_s: float | None,
    terminal_backend_facts: Mapping[str, JsonValue],
    release_attempted: bool = False,
    release_succeeded: bool = False,
    release_verified: bool = False,
    failure: Mapping[str, JsonValue] | None = None,
) -> Mapping[str, JsonValue]:
    return {
        "schema_version": "evidence_session_v1",
        "run_id": run_id,
        "experiment_id": experiment_id,
        "case_id": case_id,
        "plan_id": plan_id,
        "route_id": route_id,
        "session_instance_id": session_instance_id,
        "session_protocol_id": session_protocol_id,
        "open_s": open_s,
        "session_close_s": session_close_s,
        "status": "success" if failure is None else "failed",
        "terminal_backend_facts": _plain_json(terminal_backend_facts),
        "release_attempted": release_attempted,
        "release_succeeded": release_succeeded,
        "release_verified": release_verified,
        "failure": failure,
    }


def _plain_json(value: object) -> Any:
    return json.loads(canonical_json(value))


def _output_hash(output: np.ndarray) -> str:
    dtype = output.dtype
    if dtype.fields is not None or dtype.kind not in {"b", "i", "u", "f", "c"}:
        raise TypeError(
            "output dtype must be a scalar bool, integer, float, or complex"
        )
    if dtype.kind in {"f", "c"} and not np.isfinite(output).all():
        raise ValueError("output values must be finite")
    array = np.asarray(output, dtype=dtype.newbyteorder("<"), order="C")
    digest = hashlib.sha256()
    digest.update(
        canonical_json(
            {
                "domain": "quantum_bench.output_sha256",
                "version": 1,
                "dtype": array.dtype.str,
                "shape": array.shape,
            }
        ).encode("ascii")
    )
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _release_facts(
    terminal_backend_facts: Mapping[str, JsonValue],
) -> tuple[bool, bool, bool, bool]:
    attempted_raw = terminal_backend_facts.get("hardware_release_attempted") is True
    succeeded_raw = terminal_backend_facts.get("hardware_release_succeeded") is True
    verified_raw = terminal_backend_facts.get("hardware_release_verified") is True
    inconsistent = (succeeded_raw and not attempted_raw) or (
        verified_raw and not succeeded_raw
    )
    release_attempted = attempted_raw
    release_succeeded = attempted_raw and succeeded_raw
    release_verified = release_succeeded and verified_raw
    return release_attempted, release_succeeded, release_verified, inconsistent


def _nonempty_string(value: object, field: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string")
    if not value:
        raise ValueError(f"{field} must be nonempty")


def _sha256_string(value: object, field: str) -> None:
    _nonempty_string(value, field)
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{field} must be a lowercase SHA-256 hex digest")
    if any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{field} must be a lowercase SHA-256 hex digest")


def _canonical_uuid4(value: object, field: str) -> None:
    _nonempty_string(value, field)
    try:
        parsed = UUID(value)
    except ValueError:
        raise ValueError(f"{field} must be a canonical UUID4 string") from None
    if parsed.version != 4 or str(parsed) != value:
        raise ValueError(f"{field} must be a canonical UUID4 string")


def _unexpected_reason(exc: Exception) -> str:
    message = str(exc).strip()
    return f"{type(exc).__name__}: {message}" if message else type(exc).__name__


__all__ = [
    "default_validation_policy",
    "default_validation_policy_id",
    "load_experiment_config",
    "run_direct_samples",
    "run_session_samples",
]
