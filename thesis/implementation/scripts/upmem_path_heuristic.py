#!/usr/bin/env python3
"""Generate and fit the finite UPMEM-aware path-heuristic dataset."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from collections.abc import Mapping
from functools import lru_cache
import hashlib
from importlib import metadata
import importlib.util
import json
import math
import multiprocessing
from pathlib import Path
import queue as queue_module
import re
import subprocess
import sys
import time
from typing import Any

from quantum_bench.circuits import (
    builtin_circuit,
    parse_openqasm2,
    quest_compatible_circuit,
)
from quantum_bench.evidence import (
    canonical_json,
    load_artifacts,
    problem_id,
    tensor_network_structure_id,
)
from quantum_bench.experiment import load_experiment_config
from quantum_bench.lowering import build_contraction_dag, contraction_dag_hash, lower_tensor_network
from quantum_bench.model import ContractNode, make_simulation_job, validate_circuit_spec
from quantum_bench.planning import plan_cotengra, plan_opt_einsum
from quantum_bench.upmem.path_heuristic import (
    COST_MODEL_ID,
    ConventionalPathFeatures,
    FEATURE_NAMES,
    FeatureModelDecision,
    GROUP_FEATURE_NAMES,
    PathCandidate,
    RawFeatureVector,
    RuntimeMeasurement,
    TrainingCell,
    WeightFitResult,
    WeightVector,
    WAVE_COST_MODEL_ID,
    choose_feature_model,
    equal_model_weights,
    explicit_feature_model,
    explain_score,
    extract_conventional_features,
    extract_plan_features,
    extract_wave_path_features,
    feature_dependency_metadata,
    fit_weights,
    normalize_features,
    path_id,
    score_features,
    score_wave_path_features,
    select_calibration_candidates,
    select_best_candidate,
)
from quantum_bench.upmem.plan import (
    UpmemPlan,
    UpmemTopology,
    collection_resource_admission,
    physical_plan_id,
    plan_upmem,
    _canonical_dimensions,
)
from quantum_bench.upmem.tiling import _choose_tile_shape, tile_limits_for_numeric_mode


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "upmem_path_heuristic_v1.json"
NUMERIC_POLICY = "split_complex_float32_v1"
SCHEMA_VERSION = "upmem_path_candidate_dataset_v1"
GENERALIZATION_STUDY_ID = "upmem_path_heuristic_generalization_v1"
EXECUTION_PROFILE = "kernel_schedule_system_v1"
EXECUTION_SOURCE = "459935f586fdd16c82013838e6d27a12604c3093"
WAVE_EXECUTION_CONTRACT = {
    "schedule_policy": "static_dag_waves_v1",
    "request_transport": "packed_wave_v1",
    "fuse_complex": True,
    "geometry_policy": "panel_only_v1",
    "numeric_policy": NUMERIC_POLICY,
    "cost_model_id": WAVE_COST_MODEL_ID,
    "execution_source": EXECUTION_SOURCE,
}
_EXPECTED_CONTRACT_UNSET = object()
GENERALIZATION_WORKLOAD_MANIFEST = (
    ROOT
    / "thesis_results"
    / "upmem_path_heuristic_generalization_v1"
    / "workload"
    / "thesis_workload_manifest.json"
)
SERIAL_FEATURE_COLUMNS = (
    "circuit_id", "split", "candidate_path_id", "source_kind", "source_seed",
    "is_greedy", "topology_id", "feasible", "infeasibility_reason",
    "logical_plan_id", "physical_plan_id", "flops", "macs",
    "peak_intermediate_elements", "peak_intermediate_bytes",
    "total_intermediate_writes", "maximum_intermediate_rank",
    "contraction_count", "B_host_dpu", "B_mram_wram", "I_dpu", "N_sync",
    "E_num", "P_wram", "h2d_bytes", "d2h_bytes", "work_unit_count",
    "wave_count", "packed_operation_count", "dpu_launch_count",
    "host_reduce_count", "barrier_events", "partial_wave_count",
    "tasklet_utilization", "dpu_utilization", "host_memory_estimate_bytes",
    "semantic_identity_expansion_units",
)
WAVE_FEATURE_COLUMNS = SERIAL_FEATURE_COLUMNS + (
    "execution_profile", "score_id", "execution_contract_json",
    "cohort_count", "original_wave_count", "launch_count",
    "active_slot_launch_count", "idle_slot_launch_count", "product_count",
    "real_mac_count", "wave_critical_real_mac_sum", "control_bytes",
    "completion_bytes", "input_payload_bytes", "output_payload_bytes",
    "barrier_tasklet_calls", "wave_critical_barrier_events",
    "wave_critical_h2d_bytes", "wave_critical_d2h_bytes",
    "wave_critical_mram_bytes", "static_peak_mram_bytes",
    "known_wram_buffers_bytes", "declared_executor_memory_estimate_bytes",
    "declared_executor_persistent_bytes", "declared_executor_peak_workspace_bytes",
    "declared_executor_live_payload_bytes", "memory_admission_required_bytes",
    "memory_admission_limit_bytes", "memory_admission_reserve_bytes",
    "memory_admission_passed",
)
# Keep the historical public default; generation selects the tuple implied by
# the explicit execution profile.
FEATURE_COLUMNS = SERIAL_FEATURE_COLUMNS
CALIBRATION_COLUMNS = (
    "split", "attempt_type", "cell_id", "circuit_id", "topology_id",
    "candidate_path_id", "plan_id", "route_id", "block", "sample_index",
    "order_index", "sample_id", "session_instance_id", "experiment_id", "run_id",
    "source_sha", "candidate_generation_source_sha", "physical_execution_source_sha",
    "candidate_set_sha256", "calibration_set_sha256", "problem_id",
    "tensor_network_structure_id", "logical_plan_id", "physical_plan_id",
    "executable_id", "validation_policy_id", "output_sha256", "status",
    "validation", "fallback", "max_abs_error", "relative_l2_error",
    "norm_drift", "phase_aligned_max_abs_error", "full_precision_passed",
    "policy_reference_passed",
    "timing_scope", "total_wall_s", "session_open_s", "session_close_s",
    "session_inclusive_s", "kernel_s", "h2d_s", "d2h_s", "h2d_bytes", "d2h_bytes",
    "preparation_s", "planning_s", "lowering_s", "mapping_s", "slicing_s",
    "host_reduce_s", "rank_work_s", "request_build_s", "request_wave_s",
    "request_artifact_build_s", "payload_record_staging_s",
    "request_work_unit_materialization_s", "request_payload_materialization_s",
    "request_payload_hashing_s", "request_payload_file_write_s",
    "request_manifest_sidecar_staging_s", "request_build_residual_s",
    "request_payload_record_count", "request_payload_bytes_staged",
    "request_payload_bytes_hashed", "request_payload_files_created",
    "requested_dpus", "allocated_dpus", "active_dpus", "tasklets_per_dpu",
    "rank_count", "active_rank_count", "target_observed", "request_transport",
    "collection_resource_admission_passed", "execution_resource_admission_passed",
    "startup_resource_admission_passed", "physical_target_verified",
    "hardware_kernel_executed", "simulator_kernel_executed", "cpu_fallback_used",
    "binary_identity_verified", "native_identity_verified", "hardware_release_verified",
    "tasklet_utilization", "dpu_utilization", "dominant_wave_utilization",
    "total_wave_count", "fully_populated_wave_count", "active_dpu_ids_json",
    "active_rank_indices_json", "requested_rank_paths_json",
)
CALIBRATION_SCHEMA_VERSION = "upmem_path_runtime_calibration_v1"
CALIBRATION_TIMING_SCOPE = "steady_execution_v1"
CALIBRATION_TRANSPORT = "packed_operation_v1"
WAVE_CALIBRATION_COLUMNS = CALIBRATION_COLUMNS + (
    "profile_schema_version", "execution_profile", "execution_contract_json",
    "score_id", "primary_quantity", "round_id", "experiment_output_sha256",
    "runtime_facts_output_sha256",
)
WAVE_CALIBRATION_PROFILE = "physical_speedup_fit_v1"
WAVE_PRIMARY_QUANTITY = "session_inclusive_s"
WAVE_NORMALIZATION = "log((candidate+1)/(greedy+1))"
WAVE_INITIAL_STAGE = "initial_training"
WAVE_ADAPTIVE_STAGE = "adaptive_training"
WAVE_MAX_ADAPTIVE_ROUNDS = 3
WAVE_SESSION_PROTOCOL = "upmem_prepared_wave_abi_v5"
WAVE_KERNEL_IMPLEMENTATION = "dpu_panel_dispatch_v5_v1"
WAVE_COMPLEX_LAUNCH_POLICY = "fused_when_admitted_v1"
WAVE_PRETEST_SCHEMA = "upmem_path_wave_pretest_extraction_v1"
WAVE_PRETEST_MEASUREMENT_BLOCKS = 5
WAVE_TOPOLOGY_RESOURCES = {
    "1dpu_t8": {"dpu_count": 1, "rank_count": 1, "tasklets_per_dpu": 8},
    "4dpu_t8": {"dpu_count": 4, "rank_count": 1, "tasklets_per_dpu": 8},
}


def _canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode("ascii")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _source_sha() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False
    )
    value = result.stdout.strip()
    if result.returncode or len(value) != 40:
        raise ValueError("cannot determine source SHA")
    return value


def _version(package: str) -> str:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return "unavailable"


def _resolve_qasm_path(path: object) -> Path:
    if not isinstance(path, str) or not path.strip():
        raise ValueError("qasm_file path must be a nonempty string")
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    return candidate.resolve()


def _qasm_source(definition: Mapping[str, Any]) -> tuple[Path, str]:
    parameters = definition.get("parameters", {})
    if not isinstance(parameters, Mapping):
        raise ValueError("qasm_file circuit parameters must be an object")
    if set(parameters) != {"qasm_sha256"}:
        raise ValueError("qasm_file parameters must contain only qasm_sha256")
    expected = parameters.get("qasm_sha256")
    if (
        not isinstance(expected, str)
        or len(expected) != 64
        or expected != expected.lower()
        or any(character not in "0123456789abcdef" for character in expected)
    ):
        raise ValueError("qasm_file parameters require a lowercase qasm_sha256")
    path = _resolve_qasm_path(definition.get("path"))
    payload = path.read_bytes()
    actual = _sha256_bytes(payload)
    if actual != expected:
        raise ValueError("qasm_file qasm_sha256 does not match source bytes")
    # The legacy parser skips declarations; constrain this private input route.
    lines = [line.split("//", 1)[0].strip() for line in payload.decode("utf-8").splitlines()]
    lines = [line for line in lines if line]
    if not lines or re.fullmatch(r"OPENQASM\s+2\.0\s*;", lines[0]) is None:
        raise ValueError("qasm_file requires an OpenQASM 2.0 declaration")
    if any(line.startswith("OPENQASM") for line in lines[1:]):
        raise ValueError("qasm_file has duplicate or unsupported version declarations")
    includes = [line for line in lines if line.startswith("include")]
    if len(includes) > 1 or any(
        re.fullmatch(r'include\s+"qelib1\.inc"\s*;', line) is None for line in includes
    ):
        raise ValueError("qasm_file only supports one optional qelib1.inc include")
    registers = [line for line in lines if line.startswith("qreg")]
    if len(registers) != 1 or re.fullmatch(r"qreg\s+q\[\d+\]\s*;", registers[0]) is None:
        raise ValueError("qasm_file requires exactly one qreg q[N] declaration")
    return path, expected


def _circuit_from_definition(definition: Mapping[str, Any]) -> Any:
    """Construct the declared circuit kind without changing legacy defaults."""

    if not isinstance(definition, Mapping):
        raise ValueError("circuit definition must be an object")
    kind = definition.get("kind", "builtin")
    name = definition.get("name")
    parameters = definition.get("parameters", {})
    if not isinstance(kind, str) or not kind:
        raise ValueError("circuit definition kind must be a nonempty string")
    if not isinstance(parameters, Mapping):
        raise ValueError("circuit definition parameters must be an object")
    if kind == "qasm_file":
        if name is not None:
            raise ValueError("qasm_file circuit name must be null")
        path, expected = _qasm_source(definition)
        circuit = parse_openqasm2(path)
        if _sha256_bytes(path.read_bytes()) != expected:
            raise ValueError("qasm_file source changed while parsing")
        validate_circuit_spec(circuit)
        return circuit
    if not isinstance(name, str) or not name:
        raise ValueError("circuit definition name must be a nonempty string")
    normalized_parameters = dict(parameters)
    if kind == "builtin":
        return builtin_circuit(name, normalized_parameters)
    if kind == "quest_compatible":
        return quest_compatible_circuit(name, normalized_parameters)
    raise ValueError(f"unsupported circuit kind: {kind!r}")


def execution_contract(
    config_or_dataset: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Resolve the explicit wave contract without migrating legacy inputs."""

    if not isinstance(config_or_dataset, Mapping):
        raise TypeError("execution contract input must be a mapping")
    profile = config_or_dataset.get("execution_profile")
    declared = config_or_dataset.get("execution_contract")
    if profile is None:
        if declared is not None or config_or_dataset.get("score_id") == WAVE_COST_MODEL_ID:
            raise ValueError(
                "wave execution metadata requires explicit execution_profile"
            )
        return None
    if profile != EXECUTION_PROFILE:
        raise ValueError(f"unsupported execution profile: {profile!r}")

    expected = dict(WAVE_EXECUTION_CONTRACT)
    if declared is not None:
        if not isinstance(declared, Mapping):
            raise ValueError("execution_contract does not match frozen wave policy")
        declared_mapping = dict(declared)
        if declared_mapping != expected:
            mismatched = next(
                (
                    field
                    for field, value in expected.items()
                    if declared_mapping.get(field) != value
                ),
                "keys",
            )
            raise ValueError(
                f"execution_contract does not match frozen wave policy: {mismatched}"
            )
    if "score_id" in config_or_dataset and config_or_dataset["score_id"] != expected[
        "cost_model_id"
    ]:
        raise ValueError("wave execution profile has a mismatched score_id")
    calibration = config_or_dataset.get("calibration")
    if isinstance(calibration, Mapping) and "frozen_v1_profile" in calibration:
        raise ValueError("wave execution profile cannot mix the legacy frozen profile")
    for field, value in expected.items():
        if field in config_or_dataset and config_or_dataset[field] != value:
            raise ValueError(f"wave execution profile has a mismatched {field}")
    for policy_name in ("execution_policy", "route_policy"):
        policy = config_or_dataset.get(policy_name)
        if policy is None:
            continue
        if not isinstance(policy, Mapping):
            raise ValueError(f"{policy_name} must be an object")
        for field, value in expected.items():
            if field in policy and policy[field] != value:
                raise ValueError(f"wave execution profile has a mismatched {field}")
        if "rank_count" in policy and policy["rank_count"] != 1:
            raise ValueError("wave execution profile requires exactly one rank")
    if "rank_count" in config_or_dataset and config_or_dataset["rank_count"] != 1:
        raise ValueError("wave execution profile requires exactly one rank")
    topologies = config_or_dataset.get("topologies")
    if topologies is not None:
        if not isinstance(topologies, list) or not topologies:
            raise ValueError("wave execution profile requires topology records")
        for topology in topologies:
            if not isinstance(topology, Mapping) or int(topology.get("rank_count", 0)) != 1:
                raise ValueError("wave execution profile requires exactly one rank")
    return expected


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("schema_version") != "upmem_path_heuristic_preregistration_v1":
        raise ValueError("unrecognized path-heuristic preregistration")
    if record.get("numeric_policy") != NUMERIC_POLICY:
        raise ValueError("physical v1 requires split-complex float32")
    execution_contract(record)
    return record


def _topologies(config: dict[str, Any]) -> tuple[tuple[str, UpmemTopology], ...]:
    return tuple(
        (
            str(item["topology_id"]),
            UpmemTopology(
                dpu_count=int(item["dpu_count"]),
                rank_count=int(item["rank_count"]),
                tasklets_per_dpu=int(item["tasklets_per_dpu"]),
            ),
        )
        for item in config["topologies"]
    )


def _calibration_splits(config: Mapping[str, Any]) -> frozenset[str]:
    calibration = _mapping(config.get("calibration"), "calibration config")
    result = frozenset(
        str(value) for value in calibration.get("splits", ("training",))
    )
    if not result or not result <= {"training", "validation"}:
        raise ValueError("calibration splits must be training and/or validation")
    return result


def _frozen_calibration_profile(
    config: Mapping[str, Any],
) -> tuple[WeightVector, FeatureModelDecision, str] | None:
    calibration = _mapping(config.get("calibration"), "calibration config")
    if "frozen_v1_profile" not in calibration:
        return None
    profile_spec = _mapping(
        calibration.get("frozen_v1_profile"),
        "calibration frozen_v1_profile",
    )
    profile_path = Path(str(profile_spec.get("path", "")))
    if not profile_path.is_absolute():
        profile_path = ROOT / profile_path
    expected_sha = _required_sha(
        profile_spec.get("sha256"), "calibration frozen profile sha256", 64
    )
    actual_sha = _file_sha256(profile_path)
    if actual_sha != expected_sha:
        raise ValueError("calibration frozen profile checksum mismatch")
    profile = _mapping(
        json.loads(profile_path.read_text(encoding="utf-8")),
        "calibration frozen profile",
    )
    if profile.get("score_id") != COST_MODEL_ID:
        raise ValueError("calibration frozen profile uses a different score")
    return (
        WeightVector.from_values(_mapping(profile.get("weights"), "profile weights")),
        _model_from_profile(dict(profile)),
        actual_sha,
    )


def _generalization_calibration_candidates(
    candidates: tuple[PathCandidate, ...],
    topology_id: str,
    *,
    limit: int,
    weights: WeightVector,
    model: FeatureModelDecision,
    greedy_path_id: str,
) -> tuple[tuple[PathCandidate, ...], tuple[dict[str, str], ...]]:
    """Select the six preregistered roles without consulting physical timing."""

    feasible = tuple(
        sorted(
            (item for item in candidates if item.feasible_for(topology_id)),
            key=lambda item: item.path_id,
        )
    )
    greedy = next(item for item in feasible if item.path_id == greedy_path_id)
    normalized = {
        item.path_id: normalize_features(
            item.raw_for(topology_id), greedy.raw_for(topology_id)
        )
        for item in feasible
    }
    frozen_selected = select_best_candidate(
        feasible,
        topology_id,
        weights,
        model=model,
        greedy_path_id=greedy.path_id,
    )
    farthest = min(
        feasible,
        key=lambda item: (
            -math.sqrt(sum(value * value for value in normalized[item.path_id].values)),
            item.path_id,
        ),
    )
    role_candidates = (
        ("greedy", greedy),
        ("minimum_flops", min(feasible, key=lambda item: (item.conventional.flops, item.path_id))),
        (
            "minimum_peak_intermediate",
            min(
                feasible,
                key=lambda item: (
                    item.conventional.peak_intermediate_elements,
                    item.path_id,
                ),
            ),
        ),
        (
            "minimum_writes",
            min(
                feasible,
                key=lambda item: (
                    item.conventional.total_intermediate_writes,
                    item.path_id,
                ),
            ),
        ),
        ("frozen_v1_selected", frozen_selected),
        ("feature_diverse", farthest),
    )
    selected: list[PathCandidate] = []
    selected_ids: set[str] = set()
    roles = []
    for role, candidate in role_candidates:
        roles.append({"role": role, "candidate_path_id": candidate.path_id})
        if candidate.path_id not in selected_ids and len(selected) < limit:
            selected.append(candidate)
            selected_ids.add(candidate.path_id)
    if len(selected) < min(limit, len(feasible)):
        ordered_diverse = sorted(
            feasible,
            key=lambda item: (
                -math.sqrt(
                    sum(value * value for value in normalized[item.path_id].values)
                ),
                item.path_id,
            ),
        )
        for candidate in ordered_diverse:
            if candidate.path_id not in selected_ids:
                selected.append(candidate)
                selected_ids.add(candidate.path_id)
            if len(selected) == min(limit, len(feasible)):
                break
    return tuple(selected), tuple(roles)


def _host_memory_estimate(inputs: dict[str, Any], dag: Any) -> int:
    input_bytes = sum(int(value.nbytes) for value in inputs.values())
    # Logical intermediates are complex128 before the physical float32 lowering.
    outputs = [int(_product(node.output.shape)) * 16 for node in dag.nodes]
    # Bound one packed float32 transport copy and one native input copy in
    # addition to the logical tensors retained by the host execution shell.
    packed_transport_bytes = input_bytes // 2 + sum(outputs) // 2
    native_copy_bytes = packed_transport_bytes
    return input_bytes + sum(outputs) + packed_transport_bytes + native_copy_bytes


def _collection_admission_reasons(admission: Mapping[str, Any]) -> tuple[str, ...]:
    reasons = []
    if admission.get("tasklet_row_sufficiency_passed") is not True:
        reasons.append("tasklet_row_sufficiency")
    if (
        admission.get("dominant_work_wave_allocated_dpu_slots")
        != admission.get("dominant_work_wave_populated_dpu_slots")
    ):
        reasons.append("dominant_work_wave_underfilled")
    return tuple(reasons or ("unspecified",))


_WAVE_SCALING_BOOLEAN_FIELDS = (
    "tasklet_row_sufficiency_passed",
    "dominant_work_wave_tasklet_row_sufficiency_passed",
    "collection_resource_admission_passed",
)
_WAVE_SCALING_INTEGER_FIELDS = (
    "dominant_work_wave_allocated_dpu_slots",
    "dominant_work_wave_populated_dpu_slots",
)


def _wave_scaling_admission(
    admission: Mapping[str, Any],
    topology: Mapping[str, Any] | UpmemTopology,
    *,
    expected: Mapping[str, Any] | None = None,
    field: str = "wave scaling admission",
) -> dict[str, Any]:
    """Validate the plan-derived scaling facts without making them feasibility gates."""

    if isinstance(topology, UpmemTopology):
        dpu_count = topology.dpu_count
    else:
        dpu_count = topology.get("dpu_count")
        if isinstance(dpu_count, bool) or not isinstance(dpu_count, int):
            raise ValueError(f"{field} has an invalid topology dpu_count")
    if dpu_count <= 0:
        raise ValueError(f"{field} has a non-positive topology dpu_count")

    values: dict[str, Any] = {}
    for name in _WAVE_SCALING_BOOLEAN_FIELDS:
        value = admission.get(name)
        if type(value) is not bool:
            raise ValueError(f"{field} {name} must be a boolean")
        values[name] = value
    for name in _WAVE_SCALING_INTEGER_FIELDS:
        value = admission.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{field} {name} must be a non-negative integer")
        values[name] = value
    if values["dominant_work_wave_allocated_dpu_slots"] != dpu_count:
        raise ValueError(f"{field} allocated DPU slots do not match the topology")
    if (
        values["dominant_work_wave_populated_dpu_slots"]
        > values["dominant_work_wave_allocated_dpu_slots"]
    ):
        raise ValueError(f"{field} populated DPU slots exceed allocated slots")
    if (
        values["dominant_work_wave_tasklet_row_sufficiency_passed"]
        != values["tasklet_row_sufficiency_passed"]
    ):
        raise ValueError(f"{field} tasklet-row flags are inconsistent")
    derived_collection = values["tasklet_row_sufficiency_passed"] and (
        dpu_count == 1
        or values["dominant_work_wave_populated_dpu_slots"]
        == values["dominant_work_wave_allocated_dpu_slots"]
    )
    if values["collection_resource_admission_passed"] != derived_collection:
        raise ValueError(f"{field} collection admission is inconsistent with scaling facts")

    if expected is not None:
        expected_values = _wave_scaling_admission(
            expected, topology, field=f"{field} expected"
        )
        for name in (*_WAVE_SCALING_BOOLEAN_FIELDS, *_WAVE_SCALING_INTEGER_FIELDS):
            if values[name] != expected_values[name]:
                raise ValueError(f"{field} {name} does not match candidate plan facts")
    return values


def _estimated_work_unit_count(dag: Any) -> int:
    limits = tile_limits_for_numeric_mode("float32")
    result = 0
    for node in dag.nodes:
        if not isinstance(node, ContractNode):
            continue
        batch, m_size, n_size, k_size = _canonical_dimensions(node)
        tile_m, tile_k, tile_n = _choose_tile_shape(
            m_size, k_size, n_size, limits
        )
        result += (
            batch
            * ((m_size + tile_m - 1) // tile_m)
            * ((n_size + tile_n - 1) // tile_n)
            * ((k_size + tile_k - 1) // tile_k)
        )
    return result


def _semantic_identity_expansion_units(dag: Any, *, stop_after: int) -> int:
    """Bound recursive escaped-subtree expansion in the frozen DAG identity."""

    depths: dict[str, int] = {}
    total = 0
    for node in dag.nodes:
        depth = 1 + max((depths[item] for item in node.dependencies), default=0)
        depths[node.node_id] = depth
        total += 1 << depth
        if total > stop_after:
            return total
    return total


def _product(values: Any) -> int:
    result = 1
    for value in values:
        result *= int(value)
    return result


def _cotengra_trial_worker(
    circuit_name: str,
    circuit_parameters: dict[str, Any],
    objective: str,
    methods: str,
    seed: int,
    queue: Any,
    circuit_kind: str = "builtin",
    circuit_definition: Mapping[str, Any] | None = None,
) -> None:
    try:
        definition = (
            dict(circuit_definition)
            if circuit_definition is not None
            else {
                "kind": circuit_kind,
                "name": circuit_name,
                "parameters": circuit_parameters,
            }
        )
        if isinstance(definition.get("parameters"), Mapping):
            definition["parameters"] = dict(definition["parameters"])
        circuit = _circuit_from_definition(definition)
        network, _ = lower_tensor_network(make_simulation_job(circuit))
        path, provenance = plan_cotengra(
            network,
            objective=objective,
            methods=methods,
            max_repeats=1,
            seed=seed,
        )
        queue.put((path, provenance["planner_config_hash"], None))
    except BaseException as exc:  # child must return a finite failure record
        queue.put((None, None, f"{type(exc).__name__}:{exc}"))


def _isolated_cotengra_trial(
    *,
    circuit_kind: str = "builtin",
    circuit_name: str,
    circuit_parameters: dict[str, Any],
    objective: str,
    methods: str,
    seed: int,
    circuit_definition: Mapping[str, Any] | None = None,
) -> tuple[Any, dict[str, str]]:
    # The supported UPMEM hosts are Linux. Fork keeps imports and immutable
    # circuit metadata copy-on-write while still releasing each optimizer tree
    # when its one-trial child exits.
    context = multiprocessing.get_context("fork")
    queue = context.Queue(maxsize=1)
    process = context.Process(
        target=_cotengra_trial_worker,
        args=(
            circuit_name,
            circuit_parameters,
            objective,
            methods,
            seed,
            queue,
            circuit_kind,
            circuit_definition,
        ),
    )
    process.start()
    deadline = time.monotonic() + 300.0
    result = None
    while result is None:
        try:
            result = queue.get_nowait()
        except queue_module.Empty:
            if not process.is_alive():
                process.join()
                raise RuntimeError(
                    f"isolated cotengra trial {seed} returned no result: {process.exitcode}"
                )
            if time.monotonic() >= deadline:
                process.kill()
                process.join()
                raise RuntimeError(f"isolated cotengra trial {seed} timed out")
            time.sleep(0.01)
    path, config_hash, error = result
    process.join()
    queue.close()
    if process.exitcode != 0 or error is not None:
        raise RuntimeError(
            f"isolated cotengra trial {seed} failed: {error or process.exitcode}"
        )
    return path, {"planner_config_hash": str(config_hash)}


def _candidate_paths(
    network: Any,
    circuit_id: str,
    config: dict[str, Any],
    *,
    isolate_trials: bool = False,
    circuit_definition: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    generation = config["candidate_generation"]
    started = time.perf_counter()
    greedy_path, greedy_provenance = plan_opt_einsum(network, optimize="greedy")
    raw = [
        {
            "path": greedy_path,
            "source_kind": "opt_einsum_greedy",
            "source_seed": None,
            "is_greedy": True,
            "planner_config_hash": greedy_provenance["planner_config_hash"],
        }
    ]
    search_started = time.perf_counter()
    for trial in range(int(generation["one_trial_searches"])):
        seed = int(generation["master_seed"]) + trial
        if isolate_trials:
            if circuit_definition is None:
                raise ValueError("isolated trials require a circuit definition")
            candidate_path, provenance = _isolated_cotengra_trial(
                circuit_kind=str(circuit_definition.get("kind", "builtin")),
                circuit_name=str(circuit_definition.get("name")),
                circuit_parameters=dict(circuit_definition["parameters"]),
                objective=str(generation["cotengra_objective"]),
                methods=str(generation["cotengra_method"]),
                seed=seed,
                circuit_definition=dict(circuit_definition),
            )
        else:
            candidate_path, provenance = plan_cotengra(
                network,
                objective=str(generation["cotengra_objective"]),
                methods=str(generation["cotengra_method"]),
                max_repeats=1,
                seed=seed,
            )
        raw.append(
            {
                "path": candidate_path,
                "source_kind": "cotengra_one_trial",
                "source_seed": seed,
                "is_greedy": False,
                "planner_config_hash": provenance["planner_config_hash"],
            }
        )
    search_elapsed = time.perf_counter() - search_started
    unique: dict[str, dict[str, Any]] = {}
    for item in raw:
        identifier = path_id(item["path"], circuit_id=circuit_id)
        item = {**item, "candidate_path_id": identifier}
        previous = unique.get(identifier)
        if previous is None or (
            previous["source_seed"] is not None
            and (item["source_seed"] is None or item["source_seed"] < previous["source_seed"])
        ):
            unique[identifier] = item
        elif item["is_greedy"]:
            previous["is_greedy"] = True
    candidates = sorted(unique.values(), key=lambda item: (not item["is_greedy"], item["candidate_path_id"]))
    return candidates, {
        "candidate_generation_s": time.perf_counter() - started,
        "cotengra_search_s": search_elapsed,
    }


def _feature_columns(config_or_dataset: Mapping[str, Any]) -> tuple[str, ...]:
    return WAVE_FEATURE_COLUMNS if execution_contract(config_or_dataset) else SERIAL_FEATURE_COLUMNS


def _jsonable_wave_facts(facts: Mapping[str, Any]) -> dict[str, Any]:
    raw = facts.get("raw")
    if not isinstance(raw, RawFeatureVector):
        raise TypeError("wave facts must contain a RawFeatureVector")
    return {**dict(facts), "raw": raw.as_mapping()}


def _validate_wave_facts_context(
    facts: Mapping[str, Any],
    contract: Mapping[str, Any],
    topology: UpmemTopology,
) -> None:
    if facts.get("cost_model_id") != contract["cost_model_id"]:
        raise ValueError("wave facts have a mismatched cost model")
    plan = _mapping(facts.get("plan"), "wave facts plan")
    for field in (
        "schedule_policy", "request_transport", "fuse_complex", "geometry_policy",
        "numeric_policy",
    ):
        if plan.get(field) != contract[field]:
            raise ValueError(f"wave facts have a mismatched {field}")
    if plan.get("rank_count") != topology.rank_count or topology.rank_count != 1:
        raise ValueError("wave facts have a mismatched rank count")
    if plan.get("dpu_count") != topology.dpu_count:
        raise ValueError("wave facts have a mismatched DPU count")
    if plan.get("tasklets_per_dpu") != topology.tasklets_per_dpu:
        raise ValueError("wave facts have a mismatched tasklet count")
    raw = facts.get("raw")
    if not isinstance(raw, RawFeatureVector):
        raise TypeError("wave facts must contain a RawFeatureVector")
    if raw.numeric_overhead != 0.0:
        raise ValueError("wave facts must keep E_num inactive")


def _require_wave_execution_coverage(plan: UpmemPlan) -> None:
    """Require every declared one-rank topology slot to have planned work."""

    if not isinstance(plan, UpmemPlan):
        raise TypeError("wave execution coverage requires the final UpmemPlan")
    topology = plan.topology
    expected_ranks = {0}
    if topology.rank_count != 1:
        raise ValueError(
            "planned_execution_resource_admission_failed:"
            f"expected_one_rank_scope_observed={topology.rank_count}"
        )
    expected_slots = {(0, dpu) for dpu in range(topology.dpu_count)}
    observed_slots = {
        (unit.logical_rank, unit.logical_dpu)
        for stage in plan.stages
        if stage.kind == "contract_batch"
        for unit in stage.work_units
    }
    observed_ranks = {rank for rank, _dpu in observed_slots}
    if observed_ranks != expected_ranks or observed_slots != expected_slots:
        raise ValueError(
            "planned_execution_resource_admission_failed:"
            f"expected_active_ranks={sorted(expected_ranks)!r},"
            f"observed_active_ranks={sorted(observed_ranks)!r},"
            f"expected_active_slots={sorted(expected_slots)!r},"
            f"observed_active_slots={sorted(observed_slots)!r}"
        )


def _wave_memory_admission(
    facts: Mapping[str, Any], config: Mapping[str, Any]
) -> dict[str, Any]:
    buffers = _mapping(facts.get("host_buffers"), "wave facts host_buffers")
    estimate = buffers.get("declared_executor_memory_estimate_bytes")
    if isinstance(estimate, bool) or not isinstance(estimate, int) or estimate < 0:
        raise ValueError("wave facts declared memory estimate must be nonnegative integer")
    budget = config.get("host_memory_admission_bytes")
    if isinstance(budget, bool) or not isinstance(budget, int) or budget < 0:
        raise ValueError("host_memory_admission_bytes must be a nonnegative integer")
    reserve = config.get("host_memory_admission_reserve_bytes", 0)
    if isinstance(reserve, bool) or not isinstance(reserve, int) or reserve < 0:
        raise ValueError("host_memory_admission_reserve_bytes must be a nonnegative integer")
    required = estimate + reserve
    passed = required <= budget
    return {
        "scope": buffers.get("declared_executor_memory_scope"),
        "declared_executor_memory_estimate_bytes": estimate,
        "configured_budget_bytes": budget,
        "configured_reserve_bytes": reserve,
        "required_bytes": required,
        "passed": passed,
    }


def _wave_feature_values(
    plan: Any,
    facts: Mapping[str, Any],
    resource_admission: Mapping[str, Any],
    memory_admission: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    totals = _mapping(facts.get("totals"), "wave facts totals")
    waves = facts.get("waves")
    if not isinstance(waves, list):
        raise ValueError("wave facts waves must be a list")
    raw = facts.get("raw")
    if not isinstance(raw, RawFeatureVector):
        raise TypeError("wave facts must contain a RawFeatureVector")

    def wave_maximum(field: str) -> int:
        return sum(
            max((int(_mapping(slot, "wave slot").get(field, 0)) for slot in _mapping(wave, "wave").get("slots", [])), default=0)
            for wave in waves
        )

    partial_wave_count = sum(
        sum(bool(_mapping(slot, "wave slot").get("active")) for slot in _mapping(wave, "wave").get("slots", []))
        < int(plan.topology.dpu_count)
        for wave in waves
    )
    work_unit_count = sum(
        len(stage.work_units)
        for stage in plan.stages
        if stage.kind == "contract_batch"
    )
    buffers = _mapping(facts.get("host_buffers"), "wave facts host_buffers")
    static_memory = _mapping(facts.get("static_memory"), "wave facts static_memory")
    sync = _mapping(facts.get("sync_components"), "wave facts sync_components")
    return {
        **raw.as_mapping(),
        "execution_profile": EXECUTION_PROFILE,
        "score_id": contract["cost_model_id"],
        "execution_contract_json": _json_text(contract),
        "h2d_bytes": int(totals["h2d_bytes"]),
        "d2h_bytes": int(totals["d2h_bytes"]),
        "work_unit_count": work_unit_count,
        "wave_count": int(totals["launch_count"]),
        "packed_operation_count": int(totals["cohort_count"]),
        "dpu_launch_count": int(totals["launch_count"]),
        "host_reduce_count": int(totals["host_reduce_count"]),
        "barrier_events": int(totals["barrier_events"]),
        "partial_wave_count": partial_wave_count,
        "tasklet_utilization": float(
            resource_admission["arithmetic_weighted_tasklet_utilization"]
        ),
        "dpu_utilization": float(
            resource_admission["arithmetic_weighted_dpu_slot_utilization"]
        ),
        "host_memory_estimate_bytes": int(
            memory_admission["declared_executor_memory_estimate_bytes"]
        ),
        "cohort_count": int(totals["cohort_count"]),
        "original_wave_count": int(totals["original_wave_count"]),
        "launch_count": int(totals["launch_count"]),
        "active_slot_launch_count": int(totals["active_slot_launch_count"]),
        "idle_slot_launch_count": int(totals["idle_slot_launch_count"]),
        "product_count": int(totals["product_count"]),
        "real_mac_count": int(totals["real_mac_count"]),
        "wave_critical_real_mac_sum": int(totals["wave_critical_real_mac_sum"]),
        "control_bytes": int(totals["control_bytes"]),
        "completion_bytes": int(totals["completion_bytes"]),
        "input_payload_bytes": int(totals["input_payload_bytes"]),
        "output_payload_bytes": int(totals["output_payload_bytes"]),
        "barrier_tasklet_calls": int(totals["barrier_tasklet_calls"]),
        "wave_critical_barrier_events": int(sync["wave_critical_barrier_events"]),
        "wave_critical_h2d_bytes": wave_maximum("h2d_bytes"),
        "wave_critical_d2h_bytes": wave_maximum("d2h_bytes"),
        # ``mram_bytes`` is the per-slot storage span. The raw frozen feature
        # is the corresponding critical local-transfer maximum.
        "wave_critical_mram_bytes": int(raw.mram_wram_bytes),
        "static_peak_mram_bytes": int(static_memory["static_peak_mram_bytes"]),
        "known_wram_buffers_bytes": int(static_memory["known_wram_buffers_bytes"]),
        "declared_executor_memory_estimate_bytes": int(
            buffers["declared_executor_memory_estimate_bytes"]
        ),
        "declared_executor_persistent_bytes": int(
            buffers["declared_executor_persistent_bytes"]
        ),
        "declared_executor_peak_workspace_bytes": int(
            buffers["declared_executor_peak_workspace_bytes"]
        ),
        "declared_executor_live_payload_bytes": int(
            buffers["declared_executor_live_payload_bytes"]
        ),
        "memory_admission_required_bytes": int(memory_admission["required_bytes"]),
        "memory_admission_limit_bytes": int(memory_admission["configured_budget_bytes"]),
        "memory_admission_reserve_bytes": int(memory_admission["configured_reserve_bytes"]),
        "memory_admission_passed": bool(memory_admission["passed"]),
    }


def _wave_facts_from_record(
    record: Mapping[str, Any], topology_id: str
) -> dict[str, Any]:
    topology = next(
        (
            item for item in record.get("topologies", [])
            if item.get("topology_id") == topology_id
        ),
        None,
    )
    if not isinstance(topology, Mapping):
        raise ValueError(f"candidate lacks topology {topology_id}")
    facts = topology.get("wave_facts")
    if not isinstance(facts, Mapping):
        raise ValueError("wave candidate lacks inspectable wave facts")
    result = dict(facts)
    result["raw"] = RawFeatureVector.from_mapping(
        _mapping(result.get("raw"), "wave candidate raw features")
    )
    return result


def _wave_score(
    candidate_record: Mapping[str, Any],
    reference_record: Mapping[str, Any],
    topology_id: str,
    weights: WeightVector | None,
) -> float:
    if weights is None:
        # With no eligible active dimensions, every candidate is tied and the
        # caller's path-ID ordering is the complete deterministic policy.
        return 0.0
    return score_wave_path_features(
        _wave_facts_from_record(candidate_record, topology_id),
        _wave_facts_from_record(reference_record, topology_id),
        weights,
        cost_model_id=WAVE_COST_MODEL_ID,
    )


def _wave_equal_weights(model: FeatureModelDecision) -> WeightVector | None:
    if not model.active_features:
        return None
    weights = equal_model_weights(model)
    values = list(weights.as_tuple())
    values[4] = 0.0
    values[5] = 0.0
    if sum(values) <= 0.0:
        return None
    if not weights.numeric and not weights.wram:
        return weights
    return WeightVector.from_values(values, inactive=("E_num", "P_wram"))


def _wave_physical_plan_id(
    record: Mapping[str, Any], topology_id: str
) -> str:
    topologies = record.get("topologies")
    if not isinstance(topologies, list):
        raise ValueError("wave candidate lacks topology records")
    topology = next(
        (
            item for item in topologies
            if isinstance(item, Mapping) and item.get("topology_id") == topology_id
        ),
        None,
    )
    if not isinstance(topology, Mapping):
        raise ValueError(f"wave candidate lacks topology {topology_id}")
    physical_id = topology.get("physical_plan_id")
    if not isinstance(physical_id, str) or not physical_id:
        raise ValueError("wave candidate lacks physical-plan identity")
    return physical_id


def _wave_calibration_candidates(
    candidates: tuple[PathCandidate, ...],
    topology_id: str,
    *,
    limit: int,
    model: FeatureModelDecision,
    greedy_path_id: str,
    records: Mapping[str, Mapping[str, Any]],
    return_roles: bool = False,
) -> tuple[PathCandidate, ...] | tuple[tuple[PathCandidate, ...], tuple[dict[str, str], ...]]:
    """Select calibration paths using only eligible wave dimensions."""

    feasible = tuple(
        sorted(
            (item for item in candidates if item.feasible_for(topology_id)),
            key=lambda item: item.path_id,
        )
    )
    greedy = next(item for item in feasible if item.path_id == greedy_path_id)
    normalized = {
        item.path_id: normalize_features(
            item.raw_for(topology_id), greedy.raw_for(topology_id)
        )
        for item in feasible
    }
    weights = _wave_equal_weights(model)

    def score(item: PathCandidate) -> float:
        return _wave_score(
            records[item.path_id], records[greedy.path_id], topology_id, weights
        )

    selected: list[PathCandidate] = []
    selected_ids: set[str] = set()
    selected_physical_ids: set[str] = set()

    def add(item: PathCandidate) -> None:
        if len(selected) >= limit or item.path_id in selected_ids:
            return
        physical_id = _wave_physical_plan_id(records[item.path_id], topology_id)
        if physical_id in selected_physical_ids:
            return
        selected.append(item)
        selected_ids.add(item.path_id)
        selected_physical_ids.add(physical_id)

    role_candidates = (
        ("greedy", greedy),
        (
            "minimum_flops",
            min(feasible, key=lambda item: (item.conventional.flops, item.path_id)),
        ),
        (
            "minimum_peak_intermediate",
            min(
                feasible,
                key=lambda item: (
                    item.conventional.peak_intermediate_elements,
                    item.path_id,
                ),
            ),
        ),
        (
            "minimum_writes",
            min(
                feasible,
                key=lambda item: (
                    item.conventional.total_intermediate_writes,
                    item.path_id,
                ),
            ),
        ),
        ("equal_wave_cost", min(feasible, key=lambda item: (score(item), item.path_id))),
    )

    for _role, item in role_candidates:
        add(item)

    if model.mode == "six_term":
        eligible = tuple(
            name
            for name in model.active_features
            if name not in {"E_num", "P_wram"}
        )

        def projected(item: PathCandidate) -> tuple[float, ...]:
            return tuple(normalized[item.path_id][name] for name in eligible)
    else:
        eligible = model.active_features

        def projected(item: PathCandidate) -> tuple[float, ...]:
            return model.project(normalized[item.path_id])

    greedy_vector = projected(greedy)

    def distance_from_greedy(item: PathCandidate) -> tuple[float, str]:
        vector = projected(item)
        distance = math.sqrt(
            sum((left - right) ** 2 for left, right in zip(vector, greedy_vector))
        )
        return (-distance, item.path_id)

    # The preregistered role list has one feature-diverse role. Do not refill
    # after its path deduplicates an earlier role.
    feature_diverse = min(feasible, key=distance_from_greedy)
    add(feature_diverse)
    roles = tuple(
        [
            {"role": role, "candidate_path_id": item.path_id}
            for role, item in (*role_candidates, ("feature_diverse", feature_diverse))
        ]
    )
    selected_result = tuple(selected)
    if return_roles:
        return selected_result, roles
    return selected_result


def _serialize_wave_candidate(
    *,
    circuit_id: str,
    split: str,
    network: Any,
    inputs: dict[str, Any],
    item: dict[str, Any],
    config: dict[str, Any],
    contract: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], PathCandidate | None]:
    del inputs
    dag = build_contraction_dag(network, item["path"])
    conventional = extract_conventional_features(dag)
    logical_id = contraction_dag_hash(dag)
    feature_pairs: list[tuple[str, RawFeatureVector]] = []
    feasible_topologies: list[str] = []
    topology_records: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    feature_columns = _feature_columns(config)
    for topology_id, topology in _topologies(config):
        feasible = True
        reason = None
        plan = None
        resource_admission = None
        wave_facts = None
        memory_admission = None
        try:
            plan = plan_upmem(
                dag,
                numeric_policy=contract["numeric_policy"],
                topology=topology,
                schedule_policy=contract["schedule_policy"],
            )
            resource_admission = collection_resource_admission(plan)
            _require_wave_execution_coverage(plan)
            wave_facts = extract_wave_path_features(
                dag,
                plan,
                fuse_complex=contract["fuse_complex"],
                geometry_policy=contract["geometry_policy"],
            )
        except Exception as exc:
            feasible = False
            reason = f"{type(exc).__name__}:{exc}"
        if wave_facts is not None and plan is not None:
            _wave_scaling_admission(resource_admission, topology)
            _validate_wave_facts_context(wave_facts, contract, topology)
            memory_admission = _wave_memory_admission(wave_facts, config)
            if not memory_admission["passed"]:
                feasible = False
                reason = (
                    "declared_executor_memory_admission_failed:"
                    f"{memory_admission['required_bytes']}>"
                    f"{memory_admission['configured_budget_bytes']}"
                )

        physical_id = physical_plan_id(plan) if plan is not None else None
        wave_json = _jsonable_wave_facts(wave_facts) if wave_facts is not None else None
        if feasible and wave_facts is not None and plan is not None:
            feature_pairs.append((topology_id, wave_facts["raw"]))
            feasible_topologies.append(topology_id)
        if wave_facts is not None and plan is not None:
            wave_values = _wave_feature_values(
                plan,
                wave_facts,
                resource_admission,
                memory_admission,
                contract,
            )
        else:
            wave_values = {}
        topology_record = {
            "topology_id": topology_id,
            "topology": asdict(topology),
            "logical_plan_id": logical_id,
            "feasible": feasible,
            "infeasibility_reason": reason,
            "physical_plan_id": physical_id,
            "resource_admission": resource_admission,
            "memory_admission": memory_admission,
            "features": wave_facts["raw"].as_mapping() if wave_facts is not None else {},
            "wave_facts": wave_json,
            "host_memory_estimate_bytes": (
                memory_admission["declared_executor_memory_estimate_bytes"]
                if memory_admission is not None else None
            ),
            "execution_profile": EXECUTION_PROFILE,
            "execution_contract": dict(contract),
            "score_id": contract["cost_model_id"],
        }
        topology_records.append(topology_record)
        row = {
            "circuit_id": circuit_id,
            "split": split,
            "candidate_path_id": item["candidate_path_id"],
            "source_kind": item["source_kind"],
            "source_seed": item["source_seed"],
            "is_greedy": item["is_greedy"],
            "topology_id": topology_id,
            "feasible": feasible,
            "infeasibility_reason": reason or "",
            "logical_plan_id": logical_id,
            "physical_plan_id": physical_id or "",
            **conventional.as_mapping(),
            **wave_values,
        }
        rows.append({column: row.get(column, "") for column in feature_columns})

    candidate = PathCandidate(
        path_id=item["candidate_path_id"],
        conventional=conventional,
        features_by_topology=tuple(feature_pairs),
        feasible_topologies=tuple(feasible_topologies),
        is_greedy=bool(item["is_greedy"]),
        source=str(item["source_kind"]),
    )
    record = {
        "candidate_path_id": item["candidate_path_id"],
        "path": [list(step) for step in item["path"]],
        "source_kind": item["source_kind"],
        "source_seed": item["source_seed"],
        "planner_config_hash": item["planner_config_hash"],
        "is_greedy": item["is_greedy"],
        "logical_plan_id": logical_id,
        "conventional_features": conventional.as_mapping(),
        "execution_profile": EXECUTION_PROFILE,
        "execution_contract": dict(contract),
        "score_id": contract["cost_model_id"],
        "topologies": topology_records,
    }
    return record, rows, candidate


def _serialize_candidate(
    *,
    circuit_id: str,
    split: str,
    network: Any,
    inputs: dict[str, Any],
    item: dict[str, Any],
    config: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], PathCandidate | None]:
    contract = execution_contract(config)
    if contract is not None:
        return _serialize_wave_candidate(
            circuit_id=circuit_id,
            split=split,
            network=network,
            inputs=inputs,
            item=item,
            config=config,
            contract=contract,
        )
    dag = build_contraction_dag(network, item["path"])
    conventional = extract_conventional_features(dag)
    logical_id = contraction_dag_hash(dag)
    memory_estimate = _host_memory_estimate(inputs, dag)
    memory_limit = int(config["host_memory_admission_bytes"])
    estimated_work_units = _estimated_work_unit_count(dag)
    work_unit_limit = int(config["candidate_generation"]["maximum_planned_work_units"])
    if estimated_work_units > work_unit_limit:
        return _infeasible_candidate_record(
            circuit_id=circuit_id,
            split=split,
            item=item,
            config=config,
            reason="estimated_work_unit_count_exceeds_preregistered_bound",
            conventional=conventional,
            logical_plan_id=logical_id,
            host_memory_estimate_bytes=memory_estimate,
            estimated_work_unit_count=estimated_work_units,
        )
    feature_pairs: list[tuple[str, RawFeatureVector]] = []
    feasible_topologies: list[str] = []
    topology_records: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for topology_id, topology in _topologies(config):
        feasible = memory_estimate <= memory_limit
        reason = None if feasible else "host_memory_estimate_exceeds_preregistered_bound"
        plan = None
        facts = None
        admission = None
        try:
            if feasible:
                plan = plan_upmem(dag, numeric_policy=NUMERIC_POLICY, topology=topology)
                admission = collection_resource_admission(plan)
                facts = extract_plan_features(plan)
                if not admission["collection_resource_admission_passed"]:
                    feasible = False
                    reasons = _collection_admission_reasons(admission)
                    reason = "collection_resource_admission_failed:" + ",".join(reasons)
        except Exception as exc:
            feasible = False
            reason = f"{type(exc).__name__}:{exc}"
        if plan is not None and facts is not None:
            physical_id = physical_plan_id(plan)
            fact_values = facts.as_mapping()
        else:
            physical_id = None
            fact_values = {}
        if feasible and plan is not None and facts is not None:
            feature_pairs.append((topology_id, facts.raw))
            feasible_topologies.append(topology_id)
        topology_records.append(
            {
                "topology_id": topology_id,
                "topology": asdict(topology),
                "feasible": feasible,
                "infeasibility_reason": reason,
                "physical_plan_id": physical_id,
                "resource_admission": admission,
                "features": fact_values,
                "host_memory_estimate_bytes": memory_estimate,
            }
        )
        row = {
            "circuit_id": circuit_id,
            "split": split,
            "candidate_path_id": item["candidate_path_id"],
            "source_kind": item["source_kind"],
            "source_seed": item["source_seed"],
            "is_greedy": item["is_greedy"],
            "topology_id": topology_id,
            "feasible": feasible,
            "infeasibility_reason": reason or "",
            "logical_plan_id": logical_id,
            "physical_plan_id": physical_id or "",
            **conventional.as_mapping(),
            **fact_values,
            "host_memory_estimate_bytes": memory_estimate,
        }
        rows.append({column: row.get(column, "") for column in _feature_columns(config)})
    candidate = PathCandidate(
        path_id=item["candidate_path_id"],
        conventional=conventional,
        features_by_topology=tuple(feature_pairs),
        feasible_topologies=tuple(feasible_topologies),
        is_greedy=bool(item["is_greedy"]),
        source=str(item["source_kind"]),
    )
    record = {
        "candidate_path_id": item["candidate_path_id"],
        "path": [list(step) for step in item["path"]],
        "source_kind": item["source_kind"],
        "source_seed": item["source_seed"],
        "planner_config_hash": item["planner_config_hash"],
        "is_greedy": item["is_greedy"],
        "logical_plan_id": logical_id,
        "conventional_features": conventional.as_mapping(),
        "topologies": topology_records,
    }
    return record, rows, candidate


def _serialized_candidate_with_admission(
    *,
    circuit_id: str,
    split: str,
    definition: dict[str, Any],
    item: dict[str, Any],
    config: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], PathCandidate | None]:
    # Perform deterministic, target-neutral admission before entering native
    # planning in a child. This avoids paying or timing out physical lowering
    # for paths already known to violate the frozen campaign bounds.
    circuit = _circuit_from_definition(definition)
    network, inputs = lower_tensor_network(make_simulation_job(circuit))
    dag = build_contraction_dag(network, item["path"])
    conventional = extract_conventional_features(dag)
    memory_estimate = _host_memory_estimate(inputs, dag)
    estimated_work_units = _estimated_work_unit_count(dag)
    work_unit_limit = int(config["candidate_generation"]["maximum_planned_work_units"])
    memory_limit = int(config["host_memory_admission_bytes"])
    identity_limit = int(
        config["candidate_generation"]["maximum_semantic_identity_expansion_units"]
    )
    identity_expansion = _semantic_identity_expansion_units(
        dag, stop_after=identity_limit
    )
    if estimated_work_units > work_unit_limit:
        return _infeasible_candidate_record(
            circuit_id=circuit_id,
            split=split,
            item=item,
            config=config,
            reason="estimated_work_unit_count_exceeds_preregistered_bound",
            conventional=conventional,
            host_memory_estimate_bytes=memory_estimate,
            estimated_work_unit_count=estimated_work_units,
        )
    contract = execution_contract(config)
    if contract is None and memory_estimate > memory_limit:
        return _infeasible_candidate_record(
            circuit_id=circuit_id,
            split=split,
            item=item,
            config=config,
            reason="host_memory_estimate_exceeds_preregistered_bound",
            conventional=conventional,
            host_memory_estimate_bytes=memory_estimate,
            estimated_work_unit_count=estimated_work_units,
        )
    if identity_expansion > identity_limit:
        return _infeasible_candidate_record(
            circuit_id=circuit_id,
            split=split,
            item=item,
            config=config,
            reason="semantic_identity_expansion_exceeds_preregistered_bound",
            conventional=conventional,
            host_memory_estimate_bytes=memory_estimate,
            estimated_work_unit_count=estimated_work_units,
            semantic_identity_expansion_units=identity_expansion,
        )
    started = time.perf_counter()
    result = _serialize_candidate(
        circuit_id=circuit_id,
        split=split,
        network=network,
        inputs=inputs,
        item=item,
        config=config,
    )
    record, rows, candidate = result
    record["semantic_identity_expansion_units"] = identity_expansion
    for topology in record["topologies"]:
        topology["semantic_identity_expansion_units"] = identity_expansion
    for row in rows:
        row["semantic_identity_expansion_units"] = identity_expansion
    timeout_s = float(config["candidate_generation"]["physical_lowering_timeout_s"])
    elapsed = time.perf_counter() - started
    if elapsed > timeout_s:
        raise RuntimeError(
            "candidate lowering exceeded the generation guard; candidate "
            f"membership was not changed: {item['candidate_path_id']} "
            f"({elapsed:.3f}s > {timeout_s:g}s)"
        )
    return record, rows, candidate


def _infeasible_candidate_record(
    *,
    circuit_id: str,
    split: str,
    item: dict[str, Any],
    config: dict[str, Any],
    reason: str,
    conventional: ConventionalPathFeatures | None = None,
    logical_plan_id: str | None = None,
    host_memory_estimate_bytes: int | None = None,
    estimated_work_unit_count: int | None = None,
    semantic_identity_expansion_units: int | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], None]:
    contract = execution_contract(config)
    topology_records = []
    rows = []
    for topology_id, topology in _topologies(config):
        topology_records.append(
            {
                "topology_id": topology_id,
                "topology": asdict(topology),
                "feasible": False,
                "infeasibility_reason": reason,
                "physical_plan_id": None,
                "resource_admission": None,
                "features": {},
                "host_memory_estimate_bytes": host_memory_estimate_bytes,
                "estimated_work_unit_count": estimated_work_unit_count,
                "semantic_identity_expansion_units": semantic_identity_expansion_units,
            }
        )
        if contract is not None:
            topology_records[-1].update(
                {
                    "wave_facts": None,
                    "memory_admission": None,
                    "execution_profile": EXECUTION_PROFILE,
                    "execution_contract": dict(contract),
                    "score_id": contract["cost_model_id"],
                }
            )
        row = {
            "circuit_id": circuit_id,
            "split": split,
            "candidate_path_id": item["candidate_path_id"],
            "source_kind": item["source_kind"],
            "source_seed": item["source_seed"],
            "is_greedy": item["is_greedy"],
            "topology_id": topology_id,
            "feasible": False,
            "infeasibility_reason": reason,
        }
        if contract is not None:
            row.update(
                {
                    "execution_profile": EXECUTION_PROFILE,
                    "score_id": contract["cost_model_id"],
                    "execution_contract_json": _json_text(contract),
                }
            )
        rows.append({column: row.get(column, "") for column in _feature_columns(config)})
    record = {
        "candidate_path_id": item["candidate_path_id"],
        "path": [list(step) for step in item["path"]],
        "source_kind": item["source_kind"],
        "source_seed": item["source_seed"],
        "planner_config_hash": item["planner_config_hash"],
        "is_greedy": item["is_greedy"],
        "logical_plan_id": logical_plan_id,
        "conventional_features": (
            conventional.as_mapping() if conventional is not None else None
        ),
        "semantic_identity_expansion_units": semantic_identity_expansion_units,
        "topologies": topology_records,
    }
    if contract is not None:
        record.update(
            {
                "execution_profile": EXECUTION_PROFILE,
                "execution_contract": dict(contract),
                "score_id": contract["cost_model_id"],
            }
        )
    return (
        record,
        rows,
        None,
    )


def build_dataset(
    config: dict[str, Any],
    *,
    candidate_partition: tuple[int, int] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, float]]:
    contract = execution_contract(config)
    source_sha = _source_sha()
    circuit_records = []
    feature_rows: list[dict[str, Any]] = []
    ranking_rows: list[dict[str, Any]] = []
    calibration_cells = []
    total_generation_s = 0.0
    total_feature_s = 0.0
    calibration_splits = _calibration_splits(config)
    calibration_profile = _frozen_calibration_profile(config)
    for circuit_spec in config["circuits"]:
        circuit_id = str(circuit_spec["circuit_id"])
        split = str(circuit_spec["split"])
        definition = circuit_spec["circuit"]
        circuit = _circuit_from_definition(definition)
        job = make_simulation_job(circuit)
        network, _ = lower_tensor_network(job)
        raw_candidates, timing = _candidate_paths(
            network,
            circuit_id,
            config,
            isolate_trials=True,
            circuit_definition=definition,
        )
        if candidate_partition is not None:
            partition_index, partition_count = candidate_partition
            raw_candidates = [raw_candidates[0]] + [
                item
                for index, item in enumerate(raw_candidates[1:])
                if index % partition_count == partition_index
            ]
        total_generation_s += timing["candidate_generation_s"]
        candidate_records = []
        path_candidates = []
        feature_started = time.perf_counter()
        for item in raw_candidates:
            record, rows, candidate = _serialized_candidate_with_admission(
                circuit_id=circuit_id,
                split=split,
                definition=definition,
                item=item,
                config=config,
            )
            candidate_records.append(record)
            feature_rows.extend(rows)
            if candidate is not None:
                path_candidates.append(candidate)
        total_feature_s += time.perf_counter() - feature_started
        records_by_id = {
            record["candidate_path_id"]: record for record in candidate_records
        }
        for topology_id, _ in _topologies(config):
            feasible = [candidate for candidate in path_candidates if candidate.feasible_for(topology_id)]
            greedy = next((candidate for candidate in feasible if candidate.is_greedy), None)
            if greedy is None:
                raise ValueError(f"greedy candidate is infeasible for {circuit_id}/{topology_id}")
            normalized = tuple(
                normalize_features(candidate.raw_for(topology_id), greedy.raw_for(topology_id))
                for candidate in feasible
            )
            model = choose_feature_model(normalized)
            weights = _wave_equal_weights(model) if contract is not None else equal_model_weights(model)

            def candidate_score(candidate: PathCandidate) -> float:
                if contract is None:
                    assert weights is not None
                    return score_features(
                        candidate.raw_for(topology_id),
                        greedy.raw_for(topology_id),
                        weights,
                        model=model,
                    )
                return _wave_score(
                    records_by_id[candidate.path_id],
                    records_by_id[greedy.path_id],
                    topology_id,
                    weights,
                )

            ordered = sorted(
                feasible,
                key=lambda candidate: (
                    candidate_score(candidate),
                    candidate.path_id,
                ),
            )
            for rank, candidate in enumerate(ordered, start=1):
                ranking_rows.append(
                    {
                        "circuit_id": circuit_id,
                        "split": split,
                        "topology_id": topology_id,
                        "candidate_path_id": candidate.path_id,
                        "equal_weight_rank": rank,
                        "equal_weight_score": candidate_score(candidate),
                        "feature_model": model.mode,
                    }
                )
            if split in calibration_splits:
                if contract is not None:
                    selected, roles = _wave_calibration_candidates(
                        tuple(feasible),
                        topology_id,
                        limit=int(config["calibration"]["candidates_per_cell_maximum"]),
                        model=model,
                        greedy_path_id=greedy.path_id,
                        records=records_by_id,
                        return_roles=True,
                    )
                elif calibration_profile is None:
                    selected = select_calibration_candidates(
                        feasible,
                        topology_id,
                        limit=int(config["calibration"]["candidates_per_cell_maximum"]),
                        model=model,
                        greedy_path_id=greedy.path_id,
                    )
                    roles: tuple[dict[str, str], ...] = ()
                else:
                    calibration_weights, calibration_model, _ = calibration_profile
                    selected, roles = _generalization_calibration_candidates(
                        tuple(feasible),
                        topology_id,
                        limit=int(config["calibration"]["candidates_per_cell_maximum"]),
                        weights=calibration_weights,
                        model=calibration_model,
                        greedy_path_id=greedy.path_id,
                    )
                calibration_cells.append(
                    {
                        "cell_id": f"{circuit_id}:{topology_id}",
                        "circuit_id": circuit_id,
                        "topology_id": topology_id,
                        "greedy_path_id": greedy.path_id,
                        "feature_model": asdict(model),
                        "candidate_roles": list(roles),
                        "candidate_path_ids": [candidate.path_id for candidate in selected],
                    }
                )
        circuit_records.append(
            {
                "circuit_id": circuit_id,
                "split": split,
                "circuit": definition,
                "problem_id": problem_id(job),
                "tensor_network_structure_id": tensor_network_structure_id(network),
                "requested_cotengra_trials": int(config["candidate_generation"]["one_trial_searches"]),
                "unique_candidate_count": len(candidate_records),
                "duplicate_count": 1 + int(config["candidate_generation"]["one_trial_searches"]) - len(candidate_records),
                "candidates": candidate_records,
            }
        )
    config_hash = _sha256_bytes(_canonical_bytes(config))
    dataset = {
        "schema_version": SCHEMA_VERSION,
        "source_sha": source_sha,
        "score_id": contract["cost_model_id"] if contract is not None else COST_MODEL_ID,
        "preregistration_sha256": config_hash,
        "dependency_versions": {
            "numpy": _version("numpy"),
            "opt_einsum": _version("opt_einsum"),
            "cotengra": _version("cotengra"),
            "quimb": _version("quimb"),
        },
        "feature_dependencies": [asdict(item) for item in feature_dependency_metadata()],
        "circuits": circuit_records,
    }
    if contract is not None:
        dataset["execution_profile"] = EXECUTION_PROFILE
        dataset["execution_contract"] = dict(contract)
        dataset["inactive_score_features"] = ["E_num", "P_wram"]
    calibration = {
        "schema_version": "upmem_path_calibration_candidate_set_v1",
        "source_sha": source_sha,
        "candidate_set_sha256": _sha256_bytes(_canonical_bytes(dataset)),
        "timing_used_for_selection": False,
        "cells": calibration_cells,
    }
    if contract is not None:
        calibration["execution_profile"] = EXECUTION_PROFILE
        calibration["execution_contract"] = dict(contract)
        calibration["score_id"] = contract["cost_model_id"]
        calibration["inactive_score_features"] = ["E_num", "P_wram"]
    if calibration_profile is not None:
        _, calibration_model, calibration_profile_sha = calibration_profile
        calibration["selection_profile_sha256"] = calibration_profile_sha
        calibration["selection_profile_model"] = asdict(calibration_model)
    return dataset, feature_rows, ranking_rows, calibration, {
        "candidate_generation_s": total_generation_s,
        "feature_extraction_s": total_feature_s,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: tuple[str, ...]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _candidate_pool_hashes(dataset: Mapping[str, Any]) -> dict[str, Any]:
    contract = execution_contract(dataset)
    circuits = []
    for circuit in dataset["circuits"]:
        payload = {
            "circuit_id": circuit["circuit_id"],
            "candidate_path_ids": [
                candidate["candidate_path_id"] for candidate in circuit["candidates"]
            ],
        }
        if contract is not None:
            payload.update(
                {
                    "execution_profile": EXECUTION_PROFILE,
                    "execution_contract": dict(contract),
                    "score_id": contract["cost_model_id"],
                }
            )
        circuits.append(
            {
                "circuit_id": circuit["circuit_id"],
                "candidate_count": len(payload["candidate_path_ids"]),
                "candidate_pool_sha256": _sha256_bytes(_canonical_bytes(payload)),
            }
        )
    result = {
        "schema_version": "upmem_path_candidate_pool_hashes_v1",
        "source_sha": dataset["source_sha"],
        "candidate_set_sha256": _sha256_bytes(_canonical_bytes(dict(dataset))),
        "workload_manifest_sha256": dataset.get("workload_manifest_sha256"),
        "circuits": circuits,
    }
    if contract is not None:
        result.update(
            {
                "execution_profile": EXECUTION_PROFILE,
                "execution_contract": dict(contract),
                "score_id": contract["cost_model_id"],
            }
        )
    return result


def _workload_manifest_sha(
    config: Mapping[str, Any], config_path: Path
) -> str | None:
    if config.get("study_id") != GENERALIZATION_STUDY_ID:
        return None
    manifest = _mapping(
        json.loads(GENERALIZATION_WORKLOAD_MANIFEST.read_text(encoding="utf-8")),
        "generalization workload manifest",
    )
    config_sha = _file_sha256(config_path)
    if manifest.get("config_sha256") != config_sha:
        raise ValueError("generalization workload manifest does not match preregistration")
    return _file_sha256(GENERALIZATION_WORKLOAD_MANIFEST)


def _mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    return value


def _required_sha(value: object, field: str, length: int = 40) -> str:
    if not isinstance(value, str) or len(value) != length:
        raise ValueError(f"{field} must be a {length}-character SHA-256 value")
    try:
        int(value, 16)
    except ValueError:
        raise ValueError(f"{field} must be hexadecimal") from None
    return value


def _finite_nonnegative(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{field} must be finite and nonnegative")
    return result


def _json_text(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _wave_topology_resources(
    topology_id: object, topology: Mapping[str, Any]
) -> dict[str, int]:
    if not isinstance(topology_id, str) or topology_id not in WAVE_TOPOLOGY_RESOURCES:
        raise ValueError(
            "wave calibration requires topology 1dpu_t8 or 4dpu_t8"
        )
    expected = WAVE_TOPOLOGY_RESOURCES[str(topology_id)]
    actual = {
        field: topology.get(field)
        for field in ("dpu_count", "rank_count", "tasklets_per_dpu")
    }
    if actual != expected:
        raise ValueError(
            f"wave topology {topology_id} must use one rank and eight tasklets"
        )
    return dict(expected)


def _wave_stage_round_id(
    stage_id: str, round_ordinal: int, experiment_id: str
) -> str:
    """Keep each private round key tied to its stage and experiment identity."""

    if stage_id == WAVE_INITIAL_STAGE:
        return f"{WAVE_INITIAL_STAGE}:{experiment_id}"
    return f"{stage_id}:{round_ordinal}:{experiment_id}"


def _wave_round_id(experiment_id: str) -> str:
    """Return the established private key for the initial training round."""

    return _wave_stage_round_id(WAVE_INITIAL_STAGE, 0, experiment_id)


def _wave_stage_metadata(
    value: Mapping[str, Any], *, field: str = "calibration"
) -> dict[str, Any]:
    """Validate the small private stage binding shared by calibration artifacts."""

    stage_id = value.get("stage_id", WAVE_INITIAL_STAGE)
    if stage_id == WAVE_INITIAL_STAGE:
        round_ordinal = value.get("round_ordinal", 0)
        if (
            isinstance(round_ordinal, bool)
            or not isinstance(round_ordinal, int)
            or round_ordinal != 0
        ):
            raise ValueError(f"{field} initial stage must use round_ordinal 0")
        prior_hashes = value.get("prior_stage_hashes", [])
        if prior_hashes != []:
            raise ValueError(f"{field} initial stage must have no prior stages")
        selection_profile_sha = value.get("selection_profile_sha256")
        if selection_profile_sha not in (None, ""):
            raise ValueError(f"{field} initial stage cannot use a selection profile")
        timing_used = value.get("timing_used_for_selection", False)
        if timing_used is not False:
            raise ValueError(
                f"{field} initial stage must not use timing for selection"
            )
    elif stage_id == WAVE_ADAPTIVE_STAGE:
        round_ordinal = value.get("round_ordinal")
        if (
            isinstance(round_ordinal, bool)
            or not isinstance(round_ordinal, int)
            or not 1 <= round_ordinal <= WAVE_MAX_ADAPTIVE_ROUNDS
        ):
            raise ValueError(
                f"{field} adaptive stage round_ordinal must be 1.."
                f"{WAVE_MAX_ADAPTIVE_ROUNDS}"
            )
        prior_hashes = value.get("prior_stage_hashes")
        if not isinstance(prior_hashes, list) or len(prior_hashes) != round_ordinal:
            raise ValueError(
                f"{field} adaptive stage must list exactly its prior stages"
            )
        prior_hashes = [
            _wave_lower_sha(item, f"{field} prior_stage_hash", 64)
            for item in prior_hashes
        ]
        if len(set(prior_hashes)) != len(prior_hashes):
            raise ValueError(f"{field} prior stages must be distinct")
        selection_profile_sha = _wave_lower_sha(
            value.get("selection_profile_sha256"),
            f"{field} selection_profile_sha256",
            64,
        )
        timing_used = value.get("timing_used_for_selection")
        if timing_used is not True:
            raise ValueError(
                f"{field} adaptive stage must truthfully use timing for selection"
            )
    else:
        raise ValueError(f"{field} has an unsupported stage_id: {stage_id!r}")
    return {
        "stage_id": stage_id,
        "round_ordinal": int(round_ordinal),
        "prior_stage_hashes": list(prior_hashes),
        "selection_profile_sha256": selection_profile_sha,
        "timing_used_for_selection": timing_used,
    }


def _validate_calibration_execution_contract(
    calibration: Mapping[str, Any], contract: Mapping[str, Any] | None
) -> None:
    actual = execution_contract(calibration)
    if actual != (dict(contract) if contract is not None else None):
        raise ValueError("calibration execution contract does not match candidate dataset")
    if contract is None:
        return
    for field, value in (
        ("execution_profile", EXECUTION_PROFILE),
        ("execution_contract", dict(contract)),
        ("score_id", contract["cost_model_id"]),
    ):
        if calibration.get(field) != value:
            raise ValueError(f"calibration is missing wave {field}")


def _validate_dataset_execution_contract(
    dataset: Mapping[str, Any],
    expected_contract: Mapping[str, Any] | None | object = _EXPECTED_CONTRACT_UNSET,
) -> dict[str, Any] | None:
    contract = execution_contract(dataset)
    expected = (
        contract
        if expected_contract is _EXPECTED_CONTRACT_UNSET
        else dict(expected_contract) if expected_contract is not None else None
    )
    if contract != expected:
        raise ValueError("candidate dataset execution contract does not match config")
    if contract is None:
        if dataset.get("score_id") not in {None, COST_MODEL_ID}:
            raise ValueError("legacy candidate dataset has a mismatched score_id")
        return None
    for field, value in (
        ("execution_profile", EXECUTION_PROFILE),
        ("execution_contract", dict(contract)),
        ("score_id", contract["cost_model_id"]),
    ):
        if dataset.get(field) != value:
            raise ValueError(f"candidate dataset is missing wave {field}")
    circuits = dataset.get("circuits")
    if not isinstance(circuits, list):
        raise ValueError("wave candidate dataset must contain circuits")
    for circuit in circuits:
        circuit_mapping = _mapping(circuit, "candidate circuit")
        candidates = circuit_mapping.get("candidates")
        if not isinstance(candidates, list):
            raise ValueError("wave candidate circuit must contain candidates")
        for candidate in candidates:
            candidate_mapping = _mapping(candidate, "candidate")
            for field, value in (
                ("execution_profile", EXECUTION_PROFILE),
                ("execution_contract", dict(contract)),
                ("score_id", contract["cost_model_id"]),
            ):
                if candidate_mapping.get(field) != value:
                    raise ValueError(f"candidate is missing wave {field}")
            topologies = candidate_mapping.get("topologies")
            if not isinstance(topologies, list):
                raise ValueError("wave candidate must contain topology records")
            for topology in topologies:
                topology_mapping = _mapping(topology, "candidate topology")
                for field, value in (
                    ("execution_profile", EXECUTION_PROFILE),
                    ("execution_contract", dict(contract)),
                    ("score_id", contract["cost_model_id"]),
                ):
                    if topology_mapping.get(field) != value:
                        raise ValueError(f"candidate topology is missing wave {field}")
                topology_spec = dict(
                    _mapping(topology_mapping.get("topology"), "candidate topology resources")
                )
                try:
                    topology_value = UpmemTopology(**topology_spec)
                except (TypeError, ValueError) as exc:
                    raise ValueError("candidate topology resources are invalid") from exc
                _wave_topology_resources(
                    topology_mapping.get("topology_id"), topology_spec
                )
                facts_value = topology_mapping.get("wave_facts")
                if facts_value is None:
                    if topology_mapping.get("feasible") is True:
                        raise ValueError("feasible wave candidate lacks wave facts")
                    continue
                admission = _mapping(
                    topology_mapping.get("resource_admission"),
                    "candidate resource admission",
                )
                _wave_scaling_admission(admission, topology_value)
                facts = _wave_facts_from_record(
                    candidate_mapping, str(topology_mapping["topology_id"])
                )
                _validate_wave_facts_context(facts, contract, topology_value)
                plan = _mapping(facts.get("plan"), "wave facts plan")
                if plan.get("logical_plan_id") != candidate_mapping.get("logical_plan_id"):
                    raise ValueError("wave facts logical-plan identity mismatch")
                if plan.get("physical_plan_id") != topology_mapping.get("physical_plan_id"):
                    raise ValueError("wave facts physical-plan identity mismatch")
                if topology_mapping.get("features") != facts["raw"].as_mapping():
                    raise ValueError("wave topology raw features do not match wave facts")
    return dict(contract)


def _reject_unadapted_wave_dataset(
    dataset: Mapping[str, Any], operation: str
) -> None:
    contract = execution_contract(dataset)
    if contract is not None:
        raise ValueError(
            f"{operation} does not support execution_profile={EXECUTION_PROFILE}; "
            "wave fit/extract/evaluate is not yet adapted"
        )


def _file_sha256(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _load_frozen_wave_binary_manifest(raw_root: Path) -> dict[str, str]:
    """Load the deployment hash map adjacent to a finalized raw artifact set."""

    path = raw_root.parent / "preregistration" / "binary_sha256.json"
    if not path.is_file():
        raise ValueError("wave calibration requires a frozen deployment binary manifest")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("frozen deployment binary manifest is unreadable") from exc
    manifest = _mapping(value, "frozen deployment binary manifest")
    if not manifest:
        raise ValueError("frozen deployment binary manifest is empty")
    result: dict[str, str] = {}
    for binary_path, digest in manifest.items():
        if not isinstance(binary_path, str) or not binary_path:
            raise ValueError("frozen deployment binary manifest has an invalid path")
        result[binary_path] = _required_sha(
            digest, f"frozen deployment hash for {binary_path}", 64
        )
    return result


def _operation_timing_total(
    sample: Mapping[str, Any], field: str
) -> float | None:
    facts = _mapping(sample.get("backend_facts"), "sample backend_facts")
    operations = facts.get("operation_facts")
    if not isinstance(operations, list):
        return None
    values: list[float] = []
    for index, operation in enumerate(operations):
        operation_mapping = _mapping(operation, f"operation_facts[{index}]")
        timing = _mapping(
            operation_mapping.get("timing"), f"operation_facts[{index}].timing"
        )
        value = timing.get(field)
        if value is None:
            return None
        values.append(_finite_nonnegative(value, f"operation timing {field}"))
    return sum(values)


def _validate_null_logical_plan_candidate(
    candidate: Mapping[str, Any], circuit_id: str, candidate_id: str
) -> None:
    """Allow only generator-recorded, non-runnable infeasible candidates."""

    if candidate.get("is_greedy") is not False:
        raise ValueError(
            f"candidate {circuit_id}/{candidate_id} has a null logical_plan_id "
            "without being an unselected non-greedy record"
        )
    topologies = candidate.get("topologies")
    if not isinstance(topologies, list) or not topologies:
        raise ValueError(
            f"candidate {circuit_id}/{candidate_id} has a null logical_plan_id "
            "without topology infeasibility records"
        )
    for topology in topologies:
        topology_mapping = _mapping(
            topology, f"candidate topology {circuit_id}/{candidate_id}"
        )
        if topology_mapping.get("feasible") is not False:
            raise ValueError(
                f"candidate {circuit_id}/{candidate_id} has a null logical_plan_id "
                "but is not explicitly infeasible"
            )
        reason = topology_mapping.get("infeasibility_reason")
        if not isinstance(reason, str) or not reason:
            raise ValueError(
                f"candidate {circuit_id}/{candidate_id} has a null logical_plan_id "
                "without an explicit infeasibility reason"
            )
        if (
            topology_mapping.get("physical_plan_id") is not None
            or topology_mapping.get("resource_admission") is not None
            or topology_mapping.get("memory_admission") is not None
            or topology_mapping.get("wave_facts") is not None
        ):
            raise ValueError(
                f"candidate {circuit_id}/{candidate_id} has a null logical_plan_id "
                "but retains executable plan facts"
            )


def _calibration_candidate_index(
    dataset: Mapping[str, Any], calibration: Mapping[str, Any]
) -> tuple[
    dict[tuple[str, str, str], dict[str, Any]],
    dict[str, dict[str, Any]],
    str,
    str,
]:
    contract = execution_contract(dataset)
    if contract is None:
        _reject_unadapted_wave_dataset(dataset, "calibration extraction")
    else:
        _validate_dataset_execution_contract(dataset, contract)
    _validate_calibration_execution_contract(calibration, contract)
    stage_metadata = _wave_stage_metadata(calibration) if contract is not None else None
    if dataset.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("candidate dataset has an invalid schema version")
    if calibration.get("schema_version") != "upmem_path_calibration_candidate_set_v1":
        raise ValueError("calibration candidate set has an invalid schema version")
    candidate_source = _required_sha(dataset.get("source_sha"), "candidate source_sha")
    if calibration.get("source_sha") != candidate_source:
        raise ValueError("calibration source_sha does not match candidate dataset")
    candidate_set_sha = _sha256_bytes(_canonical_bytes(dict(dataset)))
    if calibration.get("candidate_set_sha256") != candidate_set_sha:
        raise ValueError("calibration candidate-set identity does not match dataset")
    if contract is None:
        if calibration.get("timing_used_for_selection") is not False:
            raise ValueError("legacy calibration candidate selection must not use timing")
    elif calibration.get("timing_used_for_selection") is not stage_metadata[
        "timing_used_for_selection"
    ]:
        raise ValueError("calibration timing-selection flag does not match its stage")

    circuit_map: dict[str, dict[str, Any]] = {}
    candidate_map: dict[tuple[str, str], dict[str, Any]] = {}
    circuits = dataset.get("circuits")
    if not isinstance(circuits, list) or not circuits:
        raise ValueError("candidate dataset must contain circuits")
    for circuit in circuits:
        circuit_mapping = dict(_mapping(circuit, "candidate circuit"))
        circuit_id = str(circuit_mapping.get("circuit_id", ""))
        if not circuit_id or circuit_id in circuit_map:
            raise ValueError("candidate dataset has duplicate or empty circuit IDs")
        circuit_map[circuit_id] = circuit_mapping
        candidates = circuit_mapping.get("candidates")
        if not isinstance(candidates, list):
            raise ValueError(f"candidate list is missing for {circuit_id}")
        for candidate in candidates:
            candidate_mapping = dict(_mapping(candidate, "candidate"))
            candidate_id = str(candidate_mapping.get("candidate_path_id", ""))
            key = (circuit_id, candidate_id)
            if not candidate_id or key in candidate_map:
                raise ValueError("candidate dataset has duplicate or empty path IDs")
            if contract is not None:
                _required_sha(
                    candidate_id,
                    f"candidate path ID {circuit_id}/{candidate_id}",
                    64,
                )
                if (
                    candidate_mapping.get("logical_plan_id") is None
                    and "logical_plan_id" in candidate_mapping
                ):
                    _validate_null_logical_plan_candidate(
                        candidate_mapping, circuit_id, candidate_id
                    )
                else:
                    _required_sha(
                        candidate_mapping.get("logical_plan_id"),
                        f"candidate logical_plan_id {circuit_id}/{candidate_id}",
                        64,
                    )
                source_kind = candidate_mapping.get("source_kind")
                if source_kind not in {"opt_einsum_greedy", "cotengra_one_trial"}:
                    raise ValueError(
                        f"wave candidate has an unsupported source kind: {source_kind!r}"
                    )
                _required_sha(
                    candidate_mapping.get("planner_config_hash"),
                    f"candidate planner provenance {circuit_id}/{candidate_id}",
                    64,
                )
                source_seed = candidate_mapping.get("source_seed")
                if source_kind == "opt_einsum_greedy":
                    if source_seed is not None:
                        raise ValueError("greedy wave candidate must not have a source seed")
                elif (
                    isinstance(source_seed, bool)
                    or not isinstance(source_seed, int)
                    or source_seed < 0
                ):
                    raise ValueError("cotengra wave candidate must have a nonnegative source seed")
            candidate_map[key] = candidate_mapping

    cells = calibration.get("cells")
    if not isinstance(cells, list) or not cells:
        raise ValueError("calibration candidate set must contain cells")
    expected: dict[tuple[str, str, str], dict[str, Any]] = {}
    cell_map: dict[str, dict[str, Any]] = {}
    cell_topology_keys: set[tuple[str, str]] = set()
    for cell in cells:
        cell_mapping = dict(_mapping(cell, "calibration cell"))
        cell_id = str(cell_mapping.get("cell_id", ""))
        circuit_id = str(cell_mapping.get("circuit_id", ""))
        topology_id = str(cell_mapping.get("topology_id", ""))
        if not cell_id or cell_id in cell_map:
            raise ValueError("calibration cells must have unique nonempty IDs")
        if circuit_id not in circuit_map:
            raise ValueError(f"calibration cell references unknown circuit: {circuit_id}")
        allowed_splits = {"training", "validation"}
        if contract is not None:
            allowed_splits = {"training"}
        if circuit_map[circuit_id].get("split") not in allowed_splits:
            raise ValueError(
                f"calibration cell references a non-calibration split: {circuit_id}"
            )
        if contract is not None and cell_mapping.get("split", "training") != "training":
            raise ValueError("wave calibration cells must be training only")
        if (circuit_id, topology_id) in cell_topology_keys:
            raise ValueError("calibration cells must have unique circuit/topology pairs")
        cell_topology_keys.add((circuit_id, topology_id))
        path_ids = cell_mapping.get("candidate_path_ids")
        if not isinstance(path_ids, list) or not path_ids:
            raise ValueError(f"calibration cell has no candidate paths: {cell_id}")
        if len(set(path_ids)) != len(path_ids):
            raise ValueError(f"calibration cell repeats a candidate path: {cell_id}")
        greedy_id = cell_mapping.get("greedy_path_id")
        if greedy_id not in path_ids:
            raise ValueError(f"calibration cell greedy path is not selected: {cell_id}")
        cell_map[cell_id] = cell_mapping
        for candidate_id in path_ids:
            candidate = candidate_map.get((circuit_id, str(candidate_id)))
            if candidate is None:
                raise ValueError(
                    f"calibration cell references unknown candidate: {cell_id}/{candidate_id}"
                )
            topologies = candidate.get("topologies")
            if not isinstance(topologies, list):
                raise ValueError(f"candidate lacks topology records: {candidate_id}")
            topology = next(
                (item for item in topologies if item.get("topology_id") == topology_id),
                None,
            )
            if not isinstance(topology, Mapping):
                raise ValueError(f"candidate lacks topology {topology_id}: {candidate_id}")
            if topology.get("feasible") is not True:
                raise ValueError(
                    f"calibration candidate is infeasible: {cell_id}/{candidate_id}"
                )
            if contract is not None:
                topology_resources = _mapping(
                    topology.get("topology"),
                    f"candidate topology {cell_id}/{candidate_id}",
                )
                _wave_topology_resources(topology_id, topology_resources)
                memory_admission = _mapping(
                    topology.get("memory_admission"),
                    f"candidate memory admission {cell_id}/{candidate_id}",
                )
                memory_values = {
                    field: memory_admission.get(field)
                    for field in (
                        "declared_executor_memory_estimate_bytes",
                        "configured_budget_bytes",
                        "configured_reserve_bytes",
                        "required_bytes",
                    )
                }
                if any(
                    isinstance(value, bool) or not isinstance(value, int) or value < 0
                    for value in memory_values.values()
                ):
                    raise ValueError(
                        f"candidate memory admission facts are invalid: {cell_id}/{candidate_id}"
                    )
                if (
                    memory_values["required_bytes"]
                    != memory_values["declared_executor_memory_estimate_bytes"]
                    + memory_values["configured_reserve_bytes"]
                    or memory_admission.get("passed") is not True
                    or memory_values["required_bytes"]
                    > memory_values["configured_budget_bytes"]
                ):
                    raise ValueError(
                        f"candidate memory admission is not passed: {cell_id}/{candidate_id}"
                    )
            admission = _mapping(
                topology.get("resource_admission"),
                f"candidate resource admission {cell_id}/{candidate_id}",
            )
            if contract is not None:
                _wave_scaling_admission(
                    admission,
                    _mapping(
                        topology.get("topology"),
                        f"candidate topology resources {cell_id}/{candidate_id}",
                    ),
                )
            elif admission.get("collection_resource_admission_passed") is not True:
                raise ValueError(
                    f"calibration candidate lacks resource admission: {cell_id}/{candidate_id}"
                )
            physical_plan_id_value = topology.get("physical_plan_id")
            _required_sha(
                physical_plan_id_value,
                f"candidate physical_plan_id {cell_id}/{candidate_id}",
                64,
            )
            expected[(circuit_id, topology_id, str(candidate_id))] = {
                "cell_id": cell_id,
                "circuit": circuit_map[circuit_id],
                "candidate": candidate,
                "topology": dict(topology),
                "greedy_path_id": str(greedy_id),
            }
    return expected, cell_map, candidate_set_sha, candidate_source


def _manifest_calibration_contract(
    manifest: Mapping[str, Any],
    expected: Mapping[tuple[str, str, str], Mapping[str, Any]],
    contract: Mapping[str, Any] | None = None,
    binary_manifest: Mapping[str, str] | None = None,
    *,
    stage_metadata: Mapping[str, Any] | None = None,
    measurement_blocks: int = 3,
    bind_stage_metadata: bool = True,
) -> tuple[str, str, dict[str, Any]]:
    if manifest.get("status") != "completed":
        raise ValueError("raw evidence manifest must be completed")
    if manifest.get("source_worktree_dirty") is not False:
        raise ValueError("physical execution source must be clean")
    physical_source = _required_sha(
        manifest.get("source_commit"), "physical execution source_commit"
    )
    experiment_id = _required_sha(manifest.get("experiment_id"), "experiment_id", 64)
    run_id = manifest.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("manifest run_id must be nonempty")
    configuration = _mapping(manifest.get("configuration"), "manifest configuration")
    experiment = _mapping(configuration.get("experiment"), "manifest experiment")
    collection = _mapping(experiment.get("collection"), "experiment collection")
    if collection.get("claim_policy") != "diagnostic_v1":
        raise ValueError("calibration evidence must use diagnostic_v1")
    if (
        collection.get("warmup_blocks") != 1
        or collection.get("measurement_blocks") != measurement_blocks
    ):
        raise ValueError(
            "calibration evidence must use one warmup and "
            f"{measurement_blocks} measurements"
        )
    if collection.get("session_policy") != "fresh_session_per_attempt_v1":
        raise ValueError("calibration evidence must use fresh sessions")
    if experiment.get("experiment_id") != experiment_id:
        raise ValueError("manifest experiment identity is inconsistent")

    expected_matrix = {
        (circuit_id, f"path_{candidate_id}", topology_id)
        for circuit_id, topology_id, candidate_id in expected
    }
    actual_matrix: list[tuple[str, str, str]] = []
    matrix = experiment.get("matrix")
    if not isinstance(matrix, list):
        raise ValueError("manifest experiment matrix is missing")
    for item in matrix:
        matrix_item = _mapping(item, "experiment matrix item")
        case_id = matrix_item.get("case_id")
        plan_id = matrix_item.get("plan_id")
        route_ids = matrix_item.get("route_ids")
        if not isinstance(case_id, str) or not isinstance(plan_id, str):
            raise ValueError("experiment matrix identity is invalid")
        if not isinstance(route_ids, list) or not route_ids:
            raise ValueError("experiment matrix route list is invalid")
        for route_id in route_ids:
            if not isinstance(route_id, str):
                raise ValueError("experiment matrix route ID is invalid")
            actual_matrix.append((case_id, plan_id, route_id))
    if len(actual_matrix) != len(set(actual_matrix)) or set(actual_matrix) != expected_matrix:
        raise ValueError("manifest matrix does not match calibration cell/path set exactly")
    if contract is not None:
        if physical_source != contract["execution_source"]:
            raise ValueError(
                "physical execution source does not match the wave execution contract"
            )
        if binary_manifest is None:
            raise ValueError(
                "wave calibration requires a frozen deployment binary manifest"
            )
        environment = dict(
            _mapping(configuration.get("environment"), "environment")
        )
        rank_paths = environment.get("requested_rank_paths")
        if (
            not isinstance(rank_paths, list)
            or len(rank_paths) != 1
            or any(not isinstance(path, str) or not path for path in rank_paths)
        ):
            raise ValueError("wave environment must declare exactly one rank path")
        routes = _mapping(experiment.get("routes"), "manifest experiment routes")
        route_ids = set(routes)
        if not route_ids <= set(WAVE_TOPOLOGY_RESOURCES):
            raise ValueError("wave manifest contains an unsupported route")
        expected_route_ids = {topology_id for _case, topology_id, _candidate in expected}
        if not expected_route_ids <= route_ids:
            raise ValueError("wave manifest is missing a calibration route")
        binary_bindings: dict[str, dict[str, dict[str, str]]] = {}
        for route_id, route_value in routes.items():
            route = _mapping(route_value, f"wave route {route_id}")
            if route.get("executor") != "upmem_physical":
                raise ValueError("wave calibration route must use the physical executor")
            if route.get("numeric_policy") != contract["numeric_policy"]:
                raise ValueError("wave route numeric policy does not match contract")
            options = _mapping(route.get("options"), f"wave route {route_id} options")
            resources = _wave_topology_resources(
                route_id, {
                    field: options.get(field)
                    for field in ("dpu_count", "rank_count", "tasklets_per_dpu")
                }
            )
            for field, value in (
                ("request_transport", contract["request_transport"]),
                ("schedule_policy", contract["schedule_policy"]),
                ("fuse_complex", contract["fuse_complex"]),
                ("geometry_policy", contract["geometry_policy"]),
            ):
                if options.get(field) != value:
                    raise ValueError(f"wave route {field} does not match contract")
            if options.get("rank_paths") != rank_paths:
                raise ValueError("wave route rank paths do not match environment")
            route_binary_bindings: dict[str, dict[str, str]] = {}
            for field in ("dpu_binary", "host_binary", "initialization_binary"):
                binary_path = options.get(field)
                if not isinstance(binary_path, str) or not binary_path:
                    raise ValueError(f"wave route lacks {field}")
                binary_sha = binary_manifest.get(binary_path)
                if binary_sha is None:
                    raise ValueError(
                        f"wave route {field} is absent from the frozen deployment manifest"
                    )
                route_binary_bindings[field] = {
                    "path": binary_path,
                    "sha256": binary_sha,
                }
            binary_bindings[route_id] = route_binary_bindings
            if resources["rank_count"] != 1 or resources["tasklets_per_dpu"] != 8:
                raise ValueError("wave calibration routes require one rank and eight tasklets")
        binding_values = configuration.get("identity_bindings")
        if not isinstance(binding_values, list):
            raise ValueError("wave manifest must declare identity_bindings")
        manifest_environment_id = _required_sha(
            manifest.get("environment_id"), "manifest environment_id", 64
        )
        manifest_validation_id = _required_sha(
            manifest.get("validation_policy_id"),
            "manifest validation_policy_id",
            64,
        )
        identity_bindings: dict[tuple[str, str, str], dict[str, Any]] = {}
        for value in binding_values:
            binding = dict(_mapping(value, "manifest identity binding"))
            key = (
                binding.get("case_id"),
                binding.get("plan_id"),
                binding.get("route_id"),
            )
            if any(not isinstance(item, str) or not item for item in key):
                raise ValueError("manifest identity binding has invalid route identity")
            if key in identity_bindings:
                raise ValueError("manifest identity bindings contain duplicates")
            for field in (
                "problem_id", "tensor_network_structure_id", "logical_plan_id",
                "physical_plan_id", "executable_id", "environment_id",
                "validation_policy_id",
            ):
                _required_sha(
                    binding.get(field),
                    f"manifest identity binding {field}",
                    64,
                )
            if binding["environment_id"] != manifest_environment_id:
                raise ValueError("manifest identity binding environment mismatch")
            if binding["validation_policy_id"] != manifest_validation_id:
                raise ValueError("manifest identity binding validation mismatch")
            identity_bindings[key] = binding
        if set(identity_bindings) != expected_matrix:
            raise ValueError("wave identity bindings do not match calibration set exactly")
        result = {
            "experiment_id": experiment_id,
            "collection": dict(collection),
            "environment": environment,
            "binary_bindings": binary_bindings,
            "identity_bindings": identity_bindings,
        }
        if bind_stage_metadata:
            expected_stage = dict(stage_metadata or {
                "stage_id": WAVE_INITIAL_STAGE,
                "round_ordinal": 0,
                "prior_stage_hashes": [],
                "selection_profile_sha256": None,
                "timing_used_for_selection": False,
            })
            stage_id = expected_stage["stage_id"]
            result.update(
                {
                    "round_id": _wave_stage_round_id(
                        stage_id, expected_stage["round_ordinal"], experiment_id
                    ),
                    "stage_id": stage_id,
                    "round_ordinal": expected_stage["round_ordinal"],
                    "prior_stage_hashes": list(expected_stage["prior_stage_hashes"]),
                    "selection_profile_sha256": expected_stage[
                        "selection_profile_sha256"
                    ],
                    "timing_used_for_selection": expected_stage[
                        "timing_used_for_selection"
                    ],
                    "profile_schema_version": WAVE_CALIBRATION_PROFILE,
                }
            )
        return physical_source, run_id, result
    return physical_source, run_id, {
        "experiment_id": experiment_id,
        "collection": dict(collection),
        "environment": dict(_mapping(configuration.get("environment"), "environment")),
    }


def _joined_backend_facts(
    sample: Mapping[str, Any],
    session: Mapping[str, Any],
    *,
    allow_null_overrides: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    sample_facts = dict(_mapping(sample.get("backend_facts"), "sample backend_facts"))
    terminal = dict(
        _mapping(session.get("terminal_backend_facts"), "terminal backend facts")
    )
    # These identify different evidence scopes: the aggregate physical plan and
    # the native rank session. They are both retained in their original records.
    scope_specific_fields = {"backend_id", "execution_class"}
    for field in (set(sample_facts) & set(terminal)) - scope_specific_fields:
        if allow_null_overrides and (
            sample_facts[field] is None or terminal[field] is None
        ):
            continue
        if sample_facts[field] != terminal[field]:
            raise ValueError(f"sample/session backend fact conflict: {field}")
    joined = dict(sample_facts)
    for field, value in terminal.items():
        if allow_null_overrides and joined.get(field) is None:
            joined[field] = value
        else:
            joined.setdefault(field, value)
    return joined, terminal


def _fact_matches_expected(actual: Any, expected: Any) -> bool:
    if type(expected) is bool:
        return type(actual) is bool and actual is expected
    return actual == expected


def _require_backend_contract(
    sample: Mapping[str, Any],
    session: Mapping[str, Any],
    facts: Mapping[str, Any],
    topology: Mapping[str, Any],
    contract: Mapping[str, Any] | None = None,
    binary_bindings: Mapping[str, Mapping[str, str]] | None = None,
    *,
    claim_policy: str | None = None,
) -> None:
    if session.get("status") != "success":
        raise ValueError("calibration contains a non-success session")
    for field in ("release_attempted", "release_succeeded", "release_verified"):
        if session.get(field) is not True:
            raise ValueError(f"session {field} must be true")
    expected_terminal = {
        "target_observed": "physical_hardware",
        "physical_target_verified": True,
        "hardware_kernel_executed": True,
        "simulator_kernel_executed": False,
        "cpu_fallback_used": False,
        "allocation_verified": True,
        "hardware_allocation_verified": True,
        "binary_identity_verified": True,
        "native_identity_verified": True,
        "hardware_release_verified": True,
        "startup_resource_admission_passed": True,
    }
    terminal = _mapping(session.get("terminal_backend_facts"), "terminal backend facts")
    for field, value in expected_terminal.items():
        if not _fact_matches_expected(terminal.get(field), value):
            raise ValueError(f"terminal physical fact {field} is not qualified")
    expected_topology = _mapping(topology.get("topology"), "candidate topology")
    dpu_count = int(expected_topology["dpu_count"])
    rank_count = int(expected_topology["rank_count"])
    tasklets = int(expected_topology["tasklets_per_dpu"])
    diagnostic_scaling = contract is not None and claim_policy == "diagnostic_v1"
    expected_facts = {
        "target_observed": "physical_hardware",
        "physical_target_verified": True,
        "hardware_kernel_executed": True,
        "simulator_kernel_executed": False,
        "cpu_fallback_used": False,
        "execution_resource_admission_passed": True,
        "startup_resource_admission_passed": True,
        "requested_dpus": dpu_count,
        "allocated_dpus": dpu_count,
        "active_dpus": dpu_count,
        "tasklets_per_dpu": tasklets,
        "rank_count": rank_count,
        "request_transport": (
            contract["request_transport"]
            if contract is not None
            else CALIBRATION_TRANSPORT
        ),
    }
    if not diagnostic_scaling:
        expected_facts["collection_resource_admission_passed"] = True
    for field, value in expected_facts.items():
        if not _fact_matches_expected(facts.get(field), value):
            raise ValueError(f"sample physical fact {field} is not qualified")
    for field, value in {
        "requested_dpu_count": dpu_count,
        "allocated_dpu_count": dpu_count,
        "observed_dpu_count": dpu_count,
        "observed_tasklets_per_dpu": tasklets,
        "startup_requested_dpu_count": dpu_count,
        "startup_allocated_dpu_count": dpu_count,
        "startup_requested_tasklets_per_dpu": tasklets,
    }.items():
        if terminal.get(field) != value:
            raise ValueError(f"terminal resource fact {field} is not qualified")
    if sample.get("validation", {}).get("accuracy_qualified") is not True:
        raise ValueError("calibration sample accuracy is not qualified")
    validation = _mapping(sample.get("validation"), "sample validation")
    for applicable, passed in (
        ("full_precision_threshold_applicable", "full_precision_passed"),
        ("policy_reference_applicable", "policy_reference_passed"),
    ):
        if validation.get(applicable) is True and validation.get(passed) is not True:
            raise ValueError(f"calibration sample validation {passed} is not true")
    if not isinstance(sample.get("output_sha256"), str):
        raise ValueError("calibration sample lacks output hash")
    if contract is None:
        return

    candidate_admission = _mapping(
        topology.get("resource_admission"), "candidate resource admission"
    )
    _wave_scaling_admission(candidate_admission, expected_topology)
    if diagnostic_scaling:
        _wave_scaling_admission(
            facts,
            expected_topology,
            expected=candidate_admission,
            field="sample wave scaling admission",
        )
    elif candidate_admission.get("collection_resource_admission_passed") is not True:
        raise ValueError("non-diagnostic wave execution requires collection admission")

    if sample.get("failure") is not None or session.get("failure") is not None:
        raise ValueError("wave calibration contains a failure record")
    if session.get("session_protocol_id") != WAVE_SESSION_PROTOCOL:
        raise ValueError("wave calibration uses an unexpected session protocol")
    numeric_facts = _mapping(sample.get("numeric_facts"), "sample numeric facts")
    if numeric_facts.get("numeric_policy") != contract["numeric_policy"]:
        raise ValueError("sample numeric policy does not match wave contract")
    wave_fact_expectations = [
        ("request_transport", contract["request_transport"]),
        ("schedule_policy", contract["schedule_policy"]),
        ("complex_launch_policy", WAVE_COMPLEX_LAUNCH_POLICY),
        ("geometry_kernel_policy", contract["geometry_policy"]),
        ("kernel_implementation_id", WAVE_KERNEL_IMPLEMENTATION),
        ("physical_plan_consumed", True),
        ("test_double_execution", False),
        ("rank_response_timing_scope", "cohort_counters_on_first_node_v1"),
        ("execution_active_dpu_count", dpu_count),
        ("execution_active_rank_count", 1),
    ]
    if not diagnostic_scaling:
        wave_fact_expectations.extend(
            [
                ("tasklet_row_sufficiency_passed", True),
                ("dominant_work_wave_tasklet_row_sufficiency_passed", True),
            ]
        )
    for field, value in wave_fact_expectations:
        if not _fact_matches_expected(facts.get(field), value):
            raise ValueError(f"sample wave fact {field} is not qualified")
    for field in (
        "execution_resource_admission_reasons",
        "startup_resource_admission_reasons",
    ):
        if facts.get(field) != []:
            raise ValueError(f"sample wave resource reasons are not empty: {field}")
    if facts.get("active_ranks") != [0]:
        raise ValueError("sample wave active rank set is not exactly one rank")
    for field, value in (
        ("request_transport", contract["request_transport"]),
        ("schedule_policy", contract["schedule_policy"]),
        ("complex_launch_policy", WAVE_COMPLEX_LAUNCH_POLICY),
        ("geometry_kernel_policy", contract["geometry_policy"]),
        ("kernel_implementation_id", WAVE_KERNEL_IMPLEMENTATION),
        ("target_observed", "physical_hardware"),
        ("physical_target_verified", True),
        ("hardware_kernel_executed", True),
        ("native_kernel_executed", True),
        ("simulator_kernel_executed", False),
        ("simulator_target_verified", False),
        ("cpu_fallback_used", False),
        ("ready_verified", True),
        ("hardware_release_attempted", True),
        ("hardware_release_confirmed", True),
        ("hardware_release_succeeded", True),
        ("hardware_release_verified", True),
        ("test_double_execution", False),
        ("backend_family", "upmem_sdk"),
        ("kernel_provider", WAVE_KERNEL_IMPLEMENTATION),
        ("kernel_strategy", WAVE_KERNEL_IMPLEMENTATION),
        ("kernel_identity", WAVE_KERNEL_IMPLEMENTATION),
        ("dispatch", "bulk_set_synchronous_v1"),
        ("dispatch_mode", "bulk_set_synchronous_v1"),
    ):
        if not _fact_matches_expected(terminal.get(field), value):
            raise ValueError(f"terminal wave fact {field} is not qualified")
    if terminal.get("physical_profile") != "prepared_wave_v1":
        raise ValueError("terminal physical profile is not prepared_wave_v1")
    if terminal.get("hardware_profile") != "prepared_wave_v1":
        raise ValueError("terminal hardware profile is not prepared_wave_v1")
    if terminal.get("observed_rank_count") != 1:
        raise ValueError("terminal rank count is not one")
    if terminal.get("tasklets_per_dpu") != tasklets:
        raise ValueError("terminal tasklet count does not match route")
    if terminal.get("active_rank_indices") != [0]:
        raise ValueError("terminal active rank set is not exactly one rank")
    active_dpu_ids = terminal.get("active_dpu_ids")
    if not isinstance(active_dpu_ids, list) or len(active_dpu_ids) != dpu_count:
        raise ValueError("terminal active DPU set does not match route")
    if terminal.get("startup_resource_admission_reasons") != []:
        raise ValueError("terminal startup resource admission has reasons")
    for field in (
        "dpu_binary_path", "host_binary_path", "initialization_binary_path",
        "source_root",
    ):
        if not isinstance(terminal.get(field), str) or not terminal[field]:
            raise ValueError(f"terminal wave fact {field} is missing")
    for field in (
        "dpu_binary_sha256", "host_binary_sha256", "initialization_binary_sha256",
        "strategy_config_hash",
    ):
        _required_sha(terminal.get(field), f"terminal wave fact {field}", 64)
    strategy = _mapping(terminal.get("strategy_identity"), "terminal strategy identity")
    for field, value in (
        ("request_transport", contract["request_transport"]),
        ("complex_launch_policy", WAVE_COMPLEX_LAUNCH_POLICY),
        ("geometry_kernel_policy", contract["geometry_policy"]),
        ("kernel_identity", WAVE_KERNEL_IMPLEMENTATION),
    ):
        if strategy.get(field) != value:
            raise ValueError(f"terminal strategy identity {field} is not qualified")
    if binary_bindings is None:
        raise ValueError("wave calibration lacks frozen deployment binary bindings")
    for field in ("dpu_binary", "host_binary", "initialization_binary"):
        binding = _mapping(binary_bindings.get(field), f"wave binary binding {field}")
        if terminal.get(f"{field}_path") != binding["path"]:
            raise ValueError(f"terminal {field} path does not match deployment manifest")
        if terminal.get(f"{field}_sha256") != binding["sha256"]:
            raise ValueError(f"terminal {field} SHA does not match deployment manifest")
    _required_sha(sample.get("output_sha256"), "sample output_sha256", 64)
    _required_sha(facts.get("output_hash"), "sample backend_facts.output_hash", 64)


def _calibration_row(
    *,
    sample: Mapping[str, Any],
    session: Mapping[str, Any],
    facts: Mapping[str, Any],
    terminal: Mapping[str, Any],
    expected_item: Mapping[str, Any],
    physical_source: str,
    candidate_source: str,
    candidate_set_sha: str,
    calibration_set_sha: str | None,
    raw_hashes: Mapping[str, Any],
    contract: Mapping[str, Any] | None = None,
    round_id: str | None = None,
) -> dict[str, Any]:
    measurement = _mapping(sample.get("measurement"), "sample measurement")
    if measurement.get("scope_id") != CALIBRATION_TIMING_SCOPE:
        raise ValueError("calibration sample timing scope is not steady_execution_v1")
    total_wall = _finite_nonnegative(measurement.get("total_wall_s"), "total_wall_s")
    open_s = _finite_nonnegative(session.get("open_s"), "session open_s")
    close_s = _finite_nonnegative(session.get("session_close_s"), "session_close_s")
    row: dict[str, Any] = {
        "split": expected_item["circuit"]["split"],
        "attempt_type": sample["attempt_kind"],
        "cell_id": expected_item["cell_id"],
        "circuit_id": sample["case_id"],
        "topology_id": sample["route_id"],
        "candidate_path_id": str(sample["plan_id"])[len("path_"):],
        "plan_id": sample["plan_id"],
        "route_id": sample["route_id"],
        "block": sample["block_id"],
        "sample_index": sample["sample_index"],
        "order_index": sample["order_index"],
        "sample_id": sample["sample_id"],
        "session_instance_id": sample["session_instance_id"],
        "experiment_id": sample["experiment_id"],
        "run_id": sample["run_id"],
        "source_sha": candidate_source,
        "candidate_generation_source_sha": candidate_source,
        "physical_execution_source_sha": physical_source,
        "candidate_set_sha256": candidate_set_sha,
        "calibration_set_sha256": calibration_set_sha,
        "problem_id": sample["identities"]["problem_id"],
        "tensor_network_structure_id": sample["identities"]["tensor_network_structure_id"],
        "logical_plan_id": sample["identities"]["logical_plan_id"],
        "physical_plan_id": sample["identities"]["physical_plan_id"],
        "executable_id": sample["identities"]["executable_id"],
        "validation_policy_id": sample["identities"]["validation_policy_id"],
        "output_sha256": sample["output_sha256"],
        "status": sample["status"],
        "validation": "passed",
        "fallback": "false",
        "max_abs_error": sample["validation"].get("max_abs_error"),
        "relative_l2_error": sample["validation"].get("relative_l2_error"),
        "norm_drift": sample["validation"].get("norm_drift"),
        "phase_aligned_max_abs_error": sample["validation"].get(
            "phase_aligned_max_abs_error"
        ),
        "full_precision_passed": sample["validation"].get("full_precision_passed"),
        "policy_reference_passed": sample["validation"].get(
            "policy_reference_passed"
        ),
        "timing_scope": measurement["scope_id"],
        "total_wall_s": total_wall,
        "session_open_s": open_s,
        "session_close_s": close_s,
        "session_inclusive_s": open_s + total_wall + close_s,
        "kernel_s": measurement.get("kernel_s"),
        "h2d_s": measurement.get("h2d_s"),
        "d2h_s": measurement.get("d2h_s"),
        "h2d_bytes": measurement.get("h2d_bytes"),
        "d2h_bytes": measurement.get("d2h_bytes"),
        "preparation_s": measurement.get("preparation_s"),
        "planning_s": measurement.get("planning_s"),
        "lowering_s": measurement.get("lowering_s"),
        "mapping_s": measurement.get("mapping_s"),
        "slicing_s": measurement.get("slicing_s"),
        "host_reduce_s": measurement.get("host_reduce_s"),
        "rank_work_s": measurement.get("rank_work_s"),
        "request_build_s": _operation_timing_total(sample, "request_build_sum_s"),
        "request_wave_s": _operation_timing_total(sample, "request_wave_wall_sum_s"),
        "request_artifact_build_s": _operation_timing_total(sample, "request_artifact_build_sum_s"),
        "payload_record_staging_s": _operation_timing_total(sample, "request_payload_record_staging_sum_s"),
        "request_work_unit_materialization_s": _operation_timing_total(sample, "request_work_unit_materialization_sum_s"),
        "request_payload_materialization_s": _operation_timing_total(sample, "request_payload_materialization_sum_s"),
        "request_payload_hashing_s": _operation_timing_total(sample, "request_payload_hashing_sum_s"),
        "request_payload_file_write_s": _operation_timing_total(sample, "request_payload_file_write_sum_s"),
        "request_manifest_sidecar_staging_s": _operation_timing_total(sample, "request_manifest_sidecar_staging_sum_s"),
        "request_build_residual_s": _operation_timing_total(sample, "request_build_residual_sum_s"),
        "request_payload_record_count": facts.get("request_payload_record_count"),
        "request_payload_bytes_staged": facts.get("request_payload_bytes_staged"),
        "request_payload_bytes_hashed": facts.get("request_payload_bytes_hashed"),
        "request_payload_files_created": facts.get("request_payload_files_created"),
        "requested_dpus": facts["requested_dpus"],
        "allocated_dpus": facts["allocated_dpus"],
        "active_dpus": facts["active_dpus"],
        "tasklets_per_dpu": facts["tasklets_per_dpu"],
        "rank_count": facts["rank_count"],
        "active_rank_count": facts.get("execution_active_rank_count"),
        "target_observed": facts["target_observed"],
        "request_transport": facts["request_transport"],
        "collection_resource_admission_passed": facts["collection_resource_admission_passed"],
        "execution_resource_admission_passed": facts["execution_resource_admission_passed"],
        "startup_resource_admission_passed": facts["startup_resource_admission_passed"],
        "physical_target_verified": facts["physical_target_verified"],
        "hardware_kernel_executed": facts["hardware_kernel_executed"],
        "simulator_kernel_executed": facts["simulator_kernel_executed"],
        "cpu_fallback_used": facts["cpu_fallback_used"],
        "binary_identity_verified": facts.get("binary_identity_verified"),
        "native_identity_verified": facts.get("native_identity_verified"),
        "hardware_release_verified": terminal["hardware_release_verified"],
        "tasklet_utilization": facts.get("arithmetic_weighted_tasklet_utilization"),
        "dpu_utilization": facts.get("arithmetic_weighted_dpu_slot_utilization"),
        "dominant_wave_utilization": facts.get("dominant_work_wave_utilization"),
        "total_wave_count": facts.get("total_wave_count"),
        "fully_populated_wave_count": facts.get("fully_populated_wave_count"),
        "active_dpu_ids_json": _json_text(facts.get("active_dpu_ids")),
        "active_rank_indices_json": _json_text(facts.get("active_rank_indices")),
        "requested_rank_paths_json": _json_text(raw_hashes["rank_paths"]),
    }
    if contract is not None:
        if not isinstance(round_id, str) or not round_id:
            raise ValueError("wave calibration requires a private round identity")
        experiment_output_sha = _required_sha(
            sample.get("output_sha256"), "sample output_sha256", 64
        )
        runtime_facts_output_sha = _required_sha(
            facts.get("output_hash"), "sample backend_facts.output_hash", 64
        )
        row.update(
            {
                "profile_schema_version": WAVE_CALIBRATION_PROFILE,
                "execution_profile": EXECUTION_PROFILE,
                "execution_contract_json": _json_text(contract),
                "score_id": contract["cost_model_id"],
                "primary_quantity": WAVE_PRIMARY_QUANTITY,
                "round_id": round_id,
                "experiment_output_sha256": experiment_output_sha,
                "runtime_facts_output_sha256": runtime_facts_output_sha,
            }
        )
    for field in (
        "kernel_s", "h2d_s", "d2h_s", "preparation_s", "planning_s", "lowering_s",
        "mapping_s", "slicing_s", "host_reduce_s", "rank_work_s",
        "max_abs_error", "relative_l2_error", "norm_drift",
        "phase_aligned_max_abs_error",
    ):
        if row[field] is not None:
            row[field] = _finite_nonnegative(row[field], field)
    for field in (
        "h2d_bytes", "d2h_bytes", "request_payload_record_count",
        "request_payload_bytes_staged", "request_payload_bytes_hashed",
        "request_payload_files_created", "requested_dpus", "allocated_dpus",
        "active_dpus", "tasklets_per_dpu", "rank_count", "active_rank_count",
        "total_wave_count", "fully_populated_wave_count",
    ):
        if row[field] is not None and (
            isinstance(row[field], bool) or not isinstance(row[field], int)
        ):
            raise ValueError(f"{field} must be an integer when present")
    return row


def extract_calibration(
    raw_dir: Path,
    candidate_path: Path,
    calibration_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Extract a strict physical calibration table from canonical evidence."""

    dataset = _mapping(
        json.loads(candidate_path.read_text(encoding="utf-8")), "candidate dataset"
    )
    contract = execution_contract(dataset)
    calibration = _mapping(
        json.loads(calibration_path.read_text(encoding="utf-8")),
        "calibration candidate set",
    )
    stage_metadata = _wave_stage_metadata(calibration) if contract is not None else None
    expected, cells, candidate_set_sha, candidate_source = _calibration_candidate_index(
        dataset, calibration
    )
    calibration_set_sha = (
        _file_sha256(calibration_path)
        if contract is not None
        else _sha256_bytes(_canonical_bytes(dict(calibration)))
    )
    raw_root = Path(raw_dir)
    manifest, samples, sessions = load_artifacts(raw_dir)
    private_archive = None
    if contract is not None:
        private_archive = _wave_private_archive(
            raw_root,
            raw_root.parent / "preregistration" / "physical.yml.provenance.json",
            manifest,
        )
        _validate_wave_calibration_archive_provenance(
            private_archive,
            dataset,
            calibration,
            stage_metadata,
            contract,
            candidate_set_sha=candidate_set_sha,
            candidate_source=candidate_source,
            calibration_set_sha=calibration_set_sha,
        )
    binary_manifest = (
        private_archive["binary_manifest"] if private_archive is not None else None
    )
    physical_source, run_id, manifest_contract = _manifest_calibration_contract(
        manifest,
        expected,
        contract,
        binary_manifest,
        stage_metadata=stage_metadata,
    )
    experiment_id = manifest_contract["experiment_id"]
    round_id = manifest_contract.get("round_id")
    sessions_by_id: dict[str, Mapping[str, Any]] = {}
    for session in sessions:
        session_id = session.get("session_instance_id")
        if not isinstance(session_id, str) or session_id in sessions_by_id:
            raise ValueError("sessions must have unique session_instance_id values")
        sessions_by_id[session_id] = session
    expected_observations = len(expected) * 4
    if len(samples) != expected_observations or len(sessions) != expected_observations:
        raise ValueError(
            "canonical evidence count does not match calibration set and block schedule"
        )
    raw_hashes = {
        "manifest": _file_sha256(raw_root / "manifest.json"),
        "samples": _file_sha256(raw_root / "samples.jsonl"),
        "sessions": _file_sha256(raw_root / "sessions.jsonl"),
        "rank_paths": manifest_contract["environment"].get("requested_rank_paths", []),
    }
    seen: set[tuple[str, str, int, str]] = set()
    seen_session_ids: set[str] = set()
    rows: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    for sample in samples:
        if sample.get("experiment_id") != experiment_id or sample.get("run_id") != run_id:
            raise ValueError("sample experiment/run identity does not match manifest")
        case_id = sample.get("case_id")
        route_id = sample.get("route_id")
        plan_id = sample.get("plan_id")
        if not isinstance(case_id, str) or not isinstance(route_id, str):
            raise ValueError("sample case/route identity is invalid")
        if not isinstance(plan_id, str) or not plan_id.startswith("path_"):
            raise ValueError("sample plan_id is not a candidate path plan")
        candidate_id = plan_id.removeprefix("path_")
        expected_item = expected.get((case_id, route_id, candidate_id))
        if expected_item is None:
            raise ValueError(f"sample is outside the exact calibration set: {plan_id}")
        attempt_kind = sample.get("attempt_kind")
        block_id = sample.get("block_id")
        if contract is not None and (
            isinstance(block_id, bool) or not isinstance(block_id, int)
        ):
            raise ValueError("wave calibration block identity must be an integer")
        if (attempt_kind, block_id) not in {
            ("warmup", 0), ("measurement", 1), ("measurement", 2), ("measurement", 3)
        }:
            raise ValueError("sample is outside blocks 0..3 with one warmup")
        key = (
            str(expected_item["cell_id"]),
            candidate_id,
            int(block_id),
            str(attempt_kind),
        )
        if key in seen:
            raise ValueError(f"duplicate calibration observation: {key}")
        seen.add(key)
        if sample.get("status") != "success":
            raise ValueError("calibration contains a non-success sample")
        session_id = sample.get("session_instance_id")
        if not isinstance(session_id, str) or session_id not in sessions_by_id:
            raise ValueError("sample references a missing session")
        if contract is not None and session_id in seen_session_ids:
            raise ValueError("wave calibration must use one fresh session per attempt")
        seen_session_ids.add(session_id)
        session = sessions_by_id[session_id]
        for field in ("experiment_id", "run_id", "case_id", "plan_id", "route_id"):
            if session.get(field) != sample.get(field):
                raise ValueError(f"sample/session {field} identity mismatch")
        facts, terminal = _joined_backend_facts(
            sample,
            session,
            allow_null_overrides=contract is not None,
        )
        _require_backend_contract(
            sample,
            session,
            facts,
            expected_item["topology"],
            contract,
            manifest_contract.get("binary_bindings", {}).get(route_id),
            claim_policy=manifest_contract["collection"]["claim_policy"],
        )
        identities = _mapping(sample.get("identities"), "sample identities")
        circuit = _mapping(expected_item["circuit"], "candidate circuit")
        candidate = _mapping(expected_item["candidate"], "candidate")
        topology = _mapping(expected_item["topology"], "candidate topology")
        identity_expected = {
            "problem_id": circuit["problem_id"],
            "tensor_network_structure_id": circuit["tensor_network_structure_id"],
            "logical_plan_id": candidate["logical_plan_id"],
            "physical_plan_id": topology["physical_plan_id"],
        }
        for field, value in identity_expected.items():
            if identities.get(field) != value:
                raise ValueError(f"sample identity {field} does not match candidate")
        if contract is not None:
            for field, value in (
                ("logical_plan_id", candidate["logical_plan_id"]),
                ("physical_plan_id", topology["physical_plan_id"]),
            ):
                if facts.get(field) != value:
                    raise ValueError(f"sample/backend {field} does not match candidate")
            binding = manifest_contract["identity_bindings"].get(
                (case_id, plan_id, route_id)
            )
            if binding is None:
                raise ValueError("sample identity is outside manifest identity_bindings")
            for field in (
                "problem_id", "tensor_network_structure_id", "logical_plan_id",
                "physical_plan_id", "executable_id", "environment_id",
                "validation_policy_id",
            ):
                if identities.get(field) != binding[field]:
                    raise ValueError(f"sample identity {field} does not match manifest")
        row = _calibration_row(
            sample=sample,
            session=session,
            facts=facts,
            terminal=terminal,
            expected_item=expected_item,
            physical_source=physical_source,
            candidate_source=candidate_source,
            candidate_set_sha=candidate_set_sha,
            calibration_set_sha=calibration_set_sha,
            raw_hashes=raw_hashes,
            contract=contract,
            round_id=round_id,
        )
        rows.append(row)
        observation = dict(row)
        if contract is not None:
            observation["raw_artifact_sha256"] = {
                "manifest.json": raw_hashes["manifest"],
                "samples.jsonl": raw_hashes["samples"],
                "sessions.jsonl": raw_hashes["sessions"],
            }
        observations.append(observation)
    expected_keys = {
        (str(item["cell_id"]), candidate_id, block, attempt)
        for (_case_id, _topology_id, candidate_id), item in expected.items()
        for block, attempt in (
            (0, "warmup"), (1, "measurement"), (2, "measurement"), (3, "measurement")
        )
    }
    if seen != expected_keys:
        missing = sorted(expected_keys - seen)
        extra = sorted(seen - expected_keys)
        raise ValueError(f"calibration observations are not exact (missing={missing}, extra={extra})")
    if set(sessions_by_id) != {str(sample["session_instance_id"]) for sample in samples}:
        raise ValueError("sessions are not in a one-to-one relation with samples")
    if contract is not None and seen_session_ids != set(sessions_by_id):
        raise ValueError("wave calibration sessions do not match the exact sample set")
    rows.sort(key=lambda row: (row["cell_id"], row["candidate_path_id"], row["block"]))
    observations.sort(key=lambda row: (row["cell_id"], row["candidate_path_id"], row["block"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    columns = WAVE_CALIBRATION_COLUMNS if contract is not None else CALIBRATION_COLUMNS
    _write_csv(output_dir / "path_runtime_calibration.csv", rows, columns)
    result = {
        "schema_version": CALIBRATION_SCHEMA_VERSION,
        "source_sha": candidate_source,
        "source_sha_semantics": "candidate_generation_source_sha",
        "candidate_generation_source_sha": candidate_source,
        "physical_execution_source_sha": physical_source,
        "candidate_set_sha256": candidate_set_sha,
        "calibration_set_sha256": calibration_set_sha,
        "experiment_id": experiment_id,
        "run_id": run_id,
        "claim_policy": "diagnostic_v1",
        "timing_scope": CALIBRATION_TIMING_SCOPE,
        "numeric_policy": "split_complex_float32_v1",
        "request_transport": CALIBRATION_TRANSPORT,
        "collection": {
            "warmup_blocks": 1,
            "measurement_blocks": 3,
            "blocks": [0, 1, 2, 3],
            "attempts_per_candidate_cell": 4,
        },
        "expected_cell_count": len(cells),
        "expected_candidate_cell_count": len(expected),
        "sample_count": len(samples),
        "session_count": len(sessions),
        "all_successful_physical_sessions": True,
        "all_resource_admission_passed": all(
            row["collection_resource_admission_passed"] is True
            and row["execution_resource_admission_passed"] is True
            and row["startup_resource_admission_passed"] is True
            for row in rows
        ),
        "all_accuracy_qualified": True,
        "fallback_used": False,
        "raw_artifact_sha256": {
            "manifest.json": raw_hashes["manifest"],
            "samples.jsonl": raw_hashes["samples"],
            "sessions.jsonl": raw_hashes["sessions"],
        },
        "environment": manifest_contract["environment"],
        "cells": [dict(value) for _, value in sorted(cells.items())],
        "observations": observations,
    }
    if contract is not None:
        result.update(
            {
                "profile_schema_version": WAVE_CALIBRATION_PROFILE,
                "execution_profile": EXECUTION_PROFILE,
                "execution_contract": dict(contract),
                "score_id": contract["cost_model_id"],
                "primary_quantity": WAVE_PRIMARY_QUANTITY,
                "normalization": WAVE_NORMALIZATION,
                "round_id": round_id,
                "stage_id": manifest_contract["stage_id"],
                "round_ordinal": manifest_contract["round_ordinal"],
                "prior_stage_hashes": list(manifest_contract["prior_stage_hashes"]),
                "selection_profile_sha256": manifest_contract[
                    "selection_profile_sha256"
                ],
                "timing_used_for_selection": manifest_contract[
                    "timing_used_for_selection"
                ],
                "stage_experiment_id": round_id,
                "numeric_policy": contract["numeric_policy"],
                "request_transport": contract["request_transport"],
                "private_provenance": {
                    "configuration_sha256": private_archive["configuration_sha256"],
                    "normalized_configuration_sha256": private_archive[
                        "normalized_configuration_sha256"
                    ],
                    "configuration_path": private_archive["configuration_path"],
                    "provenance_path": private_archive["provenance_path"],
                    "binary_manifest_path": private_archive["binary_manifest_path"],
                    "checksums": dict(private_archive["private_hashes"]),
                },
                "execution_provenance": dict(private_archive["provenance"]),
            }
        )
    (output_dir / "path_runtime_calibration.json").write_bytes(_canonical_bytes(result))
    return result


def generate(
    config_path: Path,
    output_dir: Path,
    *,
    check: bool = False,
    circuit_ids: tuple[str, ...] = (),
    candidate_partition: tuple[int, int] | None = None,
) -> dict[str, float]:
    full_config = load_config(config_path)
    config = full_config
    if circuit_ids:
        requested = set(circuit_ids)
        known = {str(item["circuit_id"]) for item in full_config["circuits"]}
        if not requested <= known:
            raise ValueError(f"unknown circuit shard IDs: {sorted(requested - known)!r}")
        config = {
            **full_config,
            "circuits": [
                item for item in full_config["circuits"]
                if item["circuit_id"] in requested
            ],
        }
    feature_columns = _feature_columns(config)
    dataset, features, rankings, calibration, timings = build_dataset(
        config, candidate_partition=candidate_partition
    )
    preregistration_sha = _sha256_bytes(_canonical_bytes(full_config))
    dataset["preregistration_sha256"] = preregistration_sha
    workload_manifest_sha = _workload_manifest_sha(full_config, config_path)
    if workload_manifest_sha is not None:
        dataset["workload_manifest_sha256"] = workload_manifest_sha
    calibration["candidate_set_sha256"] = _sha256_bytes(_canonical_bytes(dataset))
    outputs: dict[str, bytes] = {
        "candidate_paths.json": _canonical_bytes(dataset),
        "calibration_candidate_set.json": _canonical_bytes(calibration),
        "candidate_pool_hashes.json": _canonical_bytes(
            _candidate_pool_hashes(dataset)
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, contents in outputs.items():
        path = output_dir / filename
        if check:
            if path.read_bytes() != contents:
                raise ValueError(f"{filename} differs from deterministic recomputation")
        else:
            path.write_bytes(contents)
    feature_path = output_dir / "candidate_features.csv"
    ranking_path = output_dir / "candidate_rankings.csv"
    ranking_columns = (
        "circuit_id", "split", "topology_id", "candidate_path_id",
        "equal_weight_rank", "equal_weight_score", "feature_model",
    )
    if check:
        import io

        for path, rows, columns in (
            (feature_path, features, feature_columns),
            (ranking_path, rankings, ranking_columns),
        ):
            stream = io.StringIO(newline="")
            writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
            if path.read_bytes() != stream.getvalue().encode("utf-8"):
                raise ValueError(f"{path.name} differs from deterministic recomputation")
    else:
        _write_csv(feature_path, features, feature_columns)
        _write_csv(ranking_path, rankings, ranking_columns)
        (output_dir / "planning_timing.json").write_bytes(_canonical_bytes(timings))
    return timings


def merge_shards(
    config_path: Path,
    shard_dirs: tuple[Path, ...],
    output_dir: Path,
) -> dict[str, float]:
    """Merge exact-source circuit shards into one canonical frozen dataset."""

    if not shard_dirs:
        raise ValueError("at least one candidate shard is required")
    config = load_config(config_path)
    expected_order = [str(item["circuit_id"]) for item in config["circuits"]]
    expected_preregistration = _sha256_bytes(_canonical_bytes(config))
    datasets = [json.loads((path / "candidate_paths.json").read_text()) for path in shard_dirs]
    calibrations = [
        json.loads((path / "calibration_candidate_set.json").read_text())
        for path in shard_dirs
    ]
    contract = execution_contract(config)
    for dataset in datasets:
        _validate_dataset_execution_contract(dataset, contract)
    for calibration in calibrations:
        _validate_calibration_execution_contract(calibration, contract)
    base = {key: value for key, value in datasets[0].items() if key != "circuits"}
    if base["preregistration_sha256"] != expected_preregistration:
        raise ValueError("candidate shard does not match preregistration")
    circuit_parts: dict[str, list[dict[str, Any]]] = {}
    feature_rows: list[dict[str, Any]] = []
    ranking_rows: list[dict[str, Any]] = []
    calibration_cells: list[dict[str, Any]] = []
    timings = {"candidate_generation_s": 0.0, "feature_extraction_s": 0.0}
    for shard_dir, dataset, calibration in zip(
        shard_dirs, datasets, calibrations, strict=True
    ):
        if {key: value for key, value in dataset.items() if key != "circuits"} != base:
            raise ValueError("candidate shards have mixed source or dependency provenance")
        if calibration["source_sha"] != dataset["source_sha"]:
            raise ValueError("candidate shard calibration source mismatch")
        if calibration["candidate_set_sha256"] != _sha256_bytes(
            _canonical_bytes(dataset)
        ):
            raise ValueError("candidate shard checksum mismatch")
        for circuit in dataset["circuits"]:
            circuit_id = str(circuit["circuit_id"])
            circuit_parts.setdefault(circuit_id, []).append(circuit)
        with (shard_dir / "candidate_features.csv").open(
            newline="", encoding="utf-8"
        ) as stream:
            feature_rows.extend(csv.DictReader(stream))
        with (shard_dir / "candidate_rankings.csv").open(
            newline="", encoding="utf-8"
        ) as stream:
            ranking_rows.extend(csv.DictReader(stream))
        shard_timing = json.loads((shard_dir / "planning_timing.json").read_text())
        for key in timings:
            timings[key] += float(shard_timing[key])
    if set(circuit_parts) != set(expected_order):
        raise ValueError("candidate shards do not cover the frozen circuit set exactly")
    circuits: dict[str, dict[str, Any]] = {}
    for circuit_id, parts in circuit_parts.items():
        identity = {
            key: value for key, value in parts[0].items()
            if key not in {"candidates", "unique_candidate_count", "duplicate_count"}
        }
        candidates: dict[str, dict[str, Any]] = {}
        for part in parts:
            if {
                key: value for key, value in part.items()
                if key not in {"candidates", "unique_candidate_count", "duplicate_count"}
            } != identity:
                raise ValueError(f"mixed circuit shard identity: {circuit_id}")
            for candidate in part["candidates"]:
                candidate_id = candidate["candidate_path_id"]
                if candidate_id in candidates and candidates[candidate_id] != candidate:
                    raise ValueError(f"mixed duplicate candidate: {candidate_id}")
                candidates[candidate_id] = candidate
        ordered = sorted(
            candidates.values(),
            key=lambda item: (not item["is_greedy"], item["candidate_path_id"]),
        )
        circuits[circuit_id] = {
            **identity,
            "unique_candidate_count": len(ordered),
            "duplicate_count": 1
            + int(identity["requested_cotengra_trials"])
            - len(ordered),
            "candidates": ordered,
        }
    dataset = {**base, "circuits": [circuits[item] for item in expected_order]}
    circuit_order = {value: index for index, value in enumerate(expected_order)}
    topology_order = {
        str(item["topology_id"]): index
        for index, item in enumerate(config["topologies"])
    }
    candidate_order = {
        (circuit_id, candidate["candidate_path_id"]): index
        for circuit_id, circuit in circuits.items()
        for index, candidate in enumerate(circuit["candidates"])
    }
    feature_by_key = {
        (row["circuit_id"], row["candidate_path_id"], row["topology_id"]): row
        for row in feature_rows
    }
    feature_rows = list(feature_by_key.values())
    feature_rows.sort(
        key=lambda row: (
            circuit_order[row["circuit_id"]],
            candidate_order[(row["circuit_id"], row["candidate_path_id"])],
            topology_order[row["topology_id"]],
        )
    )
    ranking_rows = []
    calibration_cells = []
    calibration_splits = _calibration_splits(config)
    calibration_profile = _frozen_calibration_profile(config)
    for circuit in dataset["circuits"]:
        candidates = tuple(_candidate_from_record(item) for item in circuit["candidates"])
        records_by_id = {
            item["candidate_path_id"]: item for item in circuit["candidates"]
        }
        for topology_id in topology_order:
            feasible = tuple(item for item in candidates if item.feasible_for(topology_id))
            greedy = next(item for item in feasible if item.is_greedy)
            normalized = tuple(
                normalize_features(item.raw_for(topology_id), greedy.raw_for(topology_id))
                for item in feasible
            )
            model = choose_feature_model(normalized)
            weights = _wave_equal_weights(model) if contract is not None else equal_model_weights(model)

            def candidate_score(candidate: PathCandidate) -> float:
                if contract is None:
                    assert weights is not None
                    return score_features(
                        candidate.raw_for(topology_id),
                        greedy.raw_for(topology_id),
                        weights,
                        model=model,
                    )
                return _wave_score(
                    records_by_id[candidate.path_id],
                    records_by_id[greedy.path_id],
                    topology_id,
                    weights,
                )

            ordered = sorted(
                feasible,
                key=lambda item: (
                    candidate_score(item),
                    item.path_id,
                ),
            )
            for rank, candidate in enumerate(ordered, start=1):
                ranking_rows.append({
                    "circuit_id": circuit["circuit_id"],
                    "split": circuit["split"],
                    "topology_id": topology_id,
                    "candidate_path_id": candidate.path_id,
                    "equal_weight_rank": rank,
                    "equal_weight_score": candidate_score(candidate),
                    "feature_model": model.mode,
                })
            if circuit["split"] in calibration_splits:
                if contract is not None:
                    selected, roles = _wave_calibration_candidates(
                        feasible,
                        topology_id,
                        limit=int(config["calibration"]["candidates_per_cell_maximum"]),
                        model=model,
                        greedy_path_id=greedy.path_id,
                        records=records_by_id,
                        return_roles=True,
                    )
                elif calibration_profile is None:
                    selected = select_calibration_candidates(
                        feasible,
                        topology_id,
                        limit=int(config["calibration"]["candidates_per_cell_maximum"]),
                        model=model,
                        greedy_path_id=greedy.path_id,
                    )
                    roles = ()
                else:
                    calibration_weights, calibration_model, _ = calibration_profile
                    selected, roles = _generalization_calibration_candidates(
                        feasible,
                        topology_id,
                        limit=int(config["calibration"]["candidates_per_cell_maximum"]),
                        weights=calibration_weights,
                        model=calibration_model,
                        greedy_path_id=greedy.path_id,
                    )
                calibration_cells.append({
                    "cell_id": f"{circuit['circuit_id']}:{topology_id}",
                    "circuit_id": circuit["circuit_id"],
                    "topology_id": topology_id,
                    "greedy_path_id": greedy.path_id,
                    "feature_model": asdict(model),
                    "candidate_roles": list(roles),
                    "candidate_path_ids": [item.path_id for item in selected],
                })
    calibration = {
        "schema_version": "upmem_path_calibration_candidate_set_v1",
        "source_sha": dataset["source_sha"],
        "candidate_set_sha256": _sha256_bytes(_canonical_bytes(dataset)),
        "timing_used_for_selection": False,
        "cells": calibration_cells,
    }
    if contract is not None:
        calibration["execution_profile"] = EXECUTION_PROFILE
        calibration["execution_contract"] = dict(contract)
        calibration["score_id"] = contract["cost_model_id"]
        calibration["inactive_score_features"] = ["E_num", "P_wram"]
    if calibration_profile is not None:
        _, calibration_model, calibration_profile_sha = calibration_profile
        calibration["selection_profile_sha256"] = calibration_profile_sha
        calibration["selection_profile_model"] = asdict(calibration_model)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "candidate_paths.json").write_bytes(_canonical_bytes(dataset))
    (output_dir / "calibration_candidate_set.json").write_bytes(
        _canonical_bytes(calibration)
    )
    (output_dir / "candidate_pool_hashes.json").write_bytes(
        _canonical_bytes(_candidate_pool_hashes(dataset))
    )
    _write_csv(
        output_dir / "candidate_features.csv",
        feature_rows,
        _feature_columns(config),
    )
    _write_csv(
        output_dir / "candidate_rankings.csv",
        ranking_rows,
        (
            "circuit_id", "split", "topology_id", "candidate_path_id",
            "equal_weight_rank", "equal_weight_score", "feature_model",
        ),
    )
    (output_dir / "planning_timing.json").write_bytes(_canonical_bytes(timings))
    return timings


def _candidate_from_record(record: dict[str, Any]) -> PathCandidate:
    feasible = []
    pairs = []
    for item in record["topologies"]:
        if item["feasible"]:
            raw = RawFeatureVector.from_mapping(item["features"])
            pairs.append((str(item["topology_id"]), raw))
            feasible.append(str(item["topology_id"]))
    return PathCandidate(
        path_id=str(record["candidate_path_id"]),
        conventional=ConventionalPathFeatures(**record["conventional_features"]),
        features_by_topology=tuple(pairs),
        feasible_topologies=tuple(feasible),
        is_greedy=bool(record["is_greedy"]),
        source=str(record["source_kind"]),
    )


@lru_cache(maxsize=1)
def _wave_fit_function() -> Any:
    """Load the pure wave fitter without importing the qualifier."""

    fitter_path = Path(__file__).with_name("fit_upmem_wave_paths.py")
    spec = importlib.util.spec_from_file_location(
        "_upmem_wave_paths_fitter", fitter_path
    )
    if spec is None or spec.loader is None:
        raise ValueError("wave fitter module is unavailable")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(spec.name, None)
        raise
    return module.fit_upmem_wave_paths


def _wave_json_mapping(path: Path, field: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{field} must be an extracted JSON object") from exc
    return dict(_mapping(value, field))


def _wave_lower_sha(value: object, field: str, length: int) -> str:
    result = _required_sha(value, field, length)
    if result != result.lower():
        raise ValueError(f"{field} must use lowercase hexadecimal")
    return result


_WAVE_PILOT_FIELDS = (
    "pilot_weights",
    "migrated_pilot_weights",
    "prior_pilot_weights",
    "legacy_weights",
    "pilot_profile",
    "migrated_profile",
    "reuse_prior_pilot_weights",
    "reuse_prior_calibration_profile",
    "old_pilot_weight_or_profile_imported",
    "mix_lost_raw_calibration",
)


def _wave_stage_fit_inputs(
    dataset: Mapping[str, Any],
    calibration: Mapping[str, Any],
    runtime: Mapping[str, Any],
) -> dict[str, Any]:
    contract = execution_contract(dataset)
    if contract is None:
        raise ValueError("wave fitting requires an explicit execution contract")
    _validate_dataset_execution_contract(dataset, contract)
    _validate_calibration_execution_contract(calibration, contract)
    stage = _wave_stage_metadata(calibration)
    expected, cell_map, candidate_sha, candidate_source = _calibration_candidate_index(
        dataset, calibration
    )
    candidate_source = _wave_lower_sha(candidate_source, "candidate source_sha", 40)
    _wave_lower_sha(
        dataset.get("preregistration_sha256"), "preregistration_sha256", 64
    )
    for field in _WAVE_PILOT_FIELDS:
        if runtime.get(field) not in (None, False, "", [], {}):
            raise ValueError("wave fitting rejects migrated pilot calibration data")
        if calibration.get(field) not in (None, False, "", [], {}):
            raise ValueError("wave fitting rejects migrated pilot calibration data")

    calibration_sha = _sha256_bytes(_canonical_bytes(dict(calibration)))
    runtime_fields = {
        "schema_version": CALIBRATION_SCHEMA_VERSION,
        "profile_schema_version": WAVE_CALIBRATION_PROFILE,
        "execution_profile": EXECUTION_PROFILE,
        "execution_contract": dict(contract),
        "score_id": WAVE_COST_MODEL_ID,
        "primary_quantity": WAVE_PRIMARY_QUANTITY,
        "timing_scope": CALIBRATION_TIMING_SCOPE,
        "numeric_policy": contract["numeric_policy"],
        "request_transport": contract["request_transport"],
        "normalization": WAVE_NORMALIZATION,
        "source_sha": candidate_source,
        "candidate_generation_source_sha": candidate_source,
        "physical_execution_source_sha": EXECUTION_SOURCE,
        "candidate_set_sha256": candidate_sha,
        "calibration_set_sha256": calibration_sha,
        "stage_id": stage["stage_id"],
        "round_ordinal": stage["round_ordinal"],
        "prior_stage_hashes": stage["prior_stage_hashes"],
        "selection_profile_sha256": stage["selection_profile_sha256"],
        "timing_used_for_selection": stage["timing_used_for_selection"],
    }
    for field, value in runtime_fields.items():
        if runtime.get(field) != value:
            raise ValueError(f"wave runtime calibration has a mismatched {field}")
    if runtime.get("physical_execution_source_sha") != contract["execution_source"]:
        raise ValueError("wave runtime physical execution source is not frozen 459")

    experiment_id = _wave_lower_sha(
        runtime.get("experiment_id"), "wave calibration experiment_id", 64
    )
    round_id = _wave_stage_round_id(
        stage["stage_id"], stage["round_ordinal"], experiment_id
    )
    if runtime.get("round_id") != round_id:
        raise ValueError("wave runtime round does not match its calibration stage")
    if runtime.get("stage_experiment_id") != round_id:
        raise ValueError("wave runtime stage experiment identity does not match round")
    if not isinstance(runtime.get("run_id"), str) or not runtime["run_id"]:
        raise ValueError("wave runtime run_id must be nonempty")
    collection = runtime.get("collection")
    if collection != {
        "warmup_blocks": 1,
        "measurement_blocks": 3,
        "blocks": [0, 1, 2, 3],
        "attempts_per_candidate_cell": 4,
    }:
        raise ValueError("wave runtime calibration must use one warmup and three measurements")
    if runtime.get("all_successful_physical_sessions") is not True:
        raise ValueError("wave runtime calibration contains an unsuccessful session")
    all_resource_admission = runtime.get("all_resource_admission_passed")
    if type(all_resource_admission) is not bool:
        raise ValueError("wave runtime calibration has an invalid resource admission aggregate")
    if (
        all_resource_admission is False
        and runtime.get("claim_policy") != "diagnostic_v1"
    ):
        raise ValueError("wave runtime calibration contains failed resource admission")
    if runtime.get("all_accuracy_qualified") is not True:
        raise ValueError("wave runtime calibration contains unqualified accuracy")
    if runtime.get("fallback_used") is not False:
        raise ValueError("wave runtime calibration contains fallback execution")
    artifact_hashes = _mapping(
        runtime.get("raw_artifact_sha256"), "wave runtime raw artifact hashes"
    )
    for name in ("manifest.json", "samples.jsonl", "sessions.jsonl"):
        _wave_lower_sha(artifact_hashes.get(name), f"raw artifact {name}", 64)

    circuit_splits = {
        str(circuit["circuit_id"]): str(circuit.get("split"))
        for circuit in dataset["circuits"]
    }
    if any(
        circuit_splits[str(cell["circuit_id"])] != "training"
        for cell in cell_map.values()
    ):
        raise ValueError("wave stage fitting is training-only")

    observations = runtime.get("observations")
    if not isinstance(observations, list):
        raise ValueError("wave fitting requires extracted JSON observations")
    expected_count = len(expected) * 4
    for field, value in (
        ("expected_cell_count", len(cell_map)),
        ("expected_candidate_cell_count", len(expected)),
        ("sample_count", expected_count),
        ("session_count", expected_count),
    ):
        if runtime.get(field) != value:
            raise ValueError(f"wave runtime {field} does not match calibration set")
    if len(observations) != expected_count:
        raise ValueError("wave runtime observation count does not match calibration set")

    expected_by_cell_path = {
        (str(item["cell_id"]), candidate_id): item
        for (_circuit_id, _topology_id, candidate_id), item in expected.items()
    }
    runtime_run_id = runtime["run_id"]
    expected_keys = {
        (str(item["cell_id"]), candidate_id, round_id, block, attempt_type)
        for (_circuit_id, _topology_id, candidate_id), item in expected.items()
        for block, attempt_type in (
            (0, "warmup"),
            (1, "measurement"),
            (2, "measurement"),
            (3, "measurement"),
        )
    }
    expected_rows: list[dict[str, Any]] = []
    for (_circuit_id, _topology_id, candidate_id), item in sorted(expected.items()):
        for block, attempt_type in (
            (0, "warmup"),
            (1, "measurement"),
            (2, "measurement"),
            (3, "measurement"),
        ):
            expected_rows.append(
                {
                    "cell_id": item["cell_id"],
                    "candidate_path_id": candidate_id,
                    "round_id": round_id,
                    "block": block,
                    "attempt_type": attempt_type,
                    "split": "training",
                }
            )

    observed_keys: set[tuple[str, str, str, int, str]] = set()
    sample_ids: set[str] = set()
    session_ids: set[str] = set()
    for row_value in observations:
        row = _mapping(row_value, "wave runtime observation")
        cell_id = row.get("cell_id")
        candidate_id = row.get("candidate_path_id")
        if not isinstance(cell_id, str) or not isinstance(candidate_id, str):
            raise ValueError("wave runtime observation identity is invalid")
        item = expected_by_cell_path.get((cell_id, candidate_id))
        if item is None:
            raise ValueError("wave runtime observation is outside the calibration set")
        topology_id = str(item["topology"].get("topology_id"))
        attempt_type = row.get("attempt_type")
        block = row.get("block")
        if attempt_type not in {"warmup", "measurement"}:
            raise ValueError("wave runtime observation has an invalid attempt type")
        if attempt_type == "warmup" and block != 0:
            raise ValueError("wave warmup must use block zero")
        if attempt_type == "measurement" and block not in {1, 2, 3}:
            raise ValueError("wave measurement must use blocks one through three")
        key = (cell_id, candidate_id, round_id, block, attempt_type)
        if key in observed_keys:
            raise ValueError("wave runtime observations contain duplicate identities")
        observed_keys.add(key)
        sample_id = row.get("sample_id")
        session_id = row.get("session_instance_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError("wave runtime observation lacks sample_id")
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("wave runtime observation lacks session_instance_id")
        if sample_id in sample_ids:
            raise ValueError("wave runtime observations reuse a sample_id")
        if session_id in session_ids:
            raise ValueError("wave runtime observations reuse a session_instance_id")
        sample_ids.add(sample_id)
        session_ids.add(session_id)
        expected_row_fields = {
            "split": "training",
            "round_id": round_id,
            "experiment_id": experiment_id,
            "run_id": runtime_run_id,
            "circuit_id": item["circuit"]["circuit_id"],
            "topology_id": topology_id,
            "plan_id": f"path_{candidate_id}",
            "candidate_generation_source_sha": candidate_source,
            "source_sha": candidate_source,
            "physical_execution_source_sha": EXECUTION_SOURCE,
            "candidate_set_sha256": candidate_sha,
            "calibration_set_sha256": calibration_sha,
            "profile_schema_version": WAVE_CALIBRATION_PROFILE,
            "execution_profile": EXECUTION_PROFILE,
            "score_id": WAVE_COST_MODEL_ID,
            "primary_quantity": WAVE_PRIMARY_QUANTITY,
            "timing_scope": CALIBRATION_TIMING_SCOPE,
            "request_transport": contract["request_transport"],
            "validation": "passed",
            "fallback": "false",
            "status": "success",
            "logical_plan_id": item["candidate"]["logical_plan_id"],
            "physical_plan_id": item["topology"]["physical_plan_id"],
        }
        for field, value in expected_row_fields.items():
            if row.get(field) != value:
                raise ValueError(f"wave runtime observation has a mismatched {field}")
        for field in (
            "collection_resource_admission_passed",
            "execution_resource_admission_passed",
            "startup_resource_admission_passed",
        ):
            if type(row.get(field)) is not bool:
                raise ValueError(f"wave runtime observation has an invalid {field}")
        for field in (
            "execution_resource_admission_passed",
            "startup_resource_admission_passed",
        ):
            if row[field] is not True:
                raise ValueError(f"wave runtime observation has failed {field}")
        candidate_admission = _mapping(
            item["topology"].get("resource_admission"),
            "wave runtime candidate resource admission",
        )
        expected_collection_admission = candidate_admission.get(
            "collection_resource_admission_passed"
        )
        if type(expected_collection_admission) is not bool:
            raise ValueError(
                "wave runtime candidate collection admission fact is not boolean"
            )
        if row["collection_resource_admission_passed"] is not expected_collection_admission:
            raise ValueError(
                "wave runtime collection admission does not match candidate plan facts"
            )
        contract_json = row.get("execution_contract_json")
        try:
            row_contract = json.loads(contract_json)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("wave runtime observation lacks execution contract") from exc
        if row_contract != dict(contract):
            raise ValueError("wave runtime observation has a mismatched execution contract")
        row_artifacts = row.get("raw_artifact_sha256")
        if dict(_mapping(
            row_artifacts, "wave runtime observation artifact hashes"
        )) != dict(artifact_hashes):
            raise ValueError("wave runtime observation artifact hashes do not match header")
        open_s = _finite_nonnegative(row.get("session_open_s"), "session_open_s")
        steady_s = _finite_nonnegative(row.get("total_wall_s"), "total_wall_s")
        close_s = _finite_nonnegative(row.get("session_close_s"), "session_close_s")
        inclusive = open_s + steady_s + close_s
        if inclusive <= 0.0:
            raise ValueError("session-inclusive time must be finite and strictly positive")
        reported = _finite_nonnegative(
            row.get("session_inclusive_s"), "session_inclusive_s"
        )
        if not math.isclose(reported, inclusive, rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError("session-inclusive time does not equal raw timing components")

    if observed_keys != expected_keys:
        missing = sorted(expected_keys - observed_keys)
        extra = sorted(observed_keys - expected_keys)
        raise ValueError(
            f"wave runtime observations are not exact (missing={missing}, extra={extra})"
        )
    observed_all_resource_admission = all(
        row["collection_resource_admission_passed"] is True
        and row["execution_resource_admission_passed"] is True
        and row["startup_resource_admission_passed"] is True
        for row in observations
    )
    if all_resource_admission != observed_all_resource_admission:
        raise ValueError("wave runtime resource admission aggregate is inconsistent")

    cells_for_fit: dict[str, dict[str, Any]] = {}
    for cell_id, cell in sorted(cell_map.items()):
        circuit_id = str(cell["circuit_id"])
        topology_id = str(cell["topology_id"])
        raw_features: dict[str, dict[str, float]] = {}
        for candidate_id in cell["candidate_path_ids"]:
            item = expected[(circuit_id, topology_id, str(candidate_id))]
            facts = _wave_facts_from_record(item["candidate"], topology_id)
            raw_features[str(candidate_id)] = facts["raw"].as_mapping()
        cells_for_fit[cell_id] = {
            "cell_id": cell_id,
            "greedy_path_id": str(cell["greedy_path_id"]),
            "raw_features": raw_features,
        }
    return {
        "cells": cells_for_fit,
        "observations": [dict(row) for row in observations],
        "expected_rows": expected_rows,
        "experiment_id": experiment_id,
        "run_id": runtime_run_id,
        "round_id": round_id,
        "candidate_sha": candidate_sha,
        "candidate_source": candidate_source,
        "calibration_sha": calibration_sha,
        "artifact_hashes": dict(artifact_hashes),
        "cell_map": {cell_id: dict(value) for cell_id, value in cell_map.items()},
        "sample_ids": frozenset(sample_ids),
        "session_ids": frozenset(session_ids),
        "stage": stage,
    }


def _wave_initial_fit_inputs(
    dataset: Mapping[str, Any],
    calibration: Mapping[str, Any],
    runtime: Mapping[str, Any],
) -> tuple[
    dict[str, dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    str,
    str,
    str,
    str,
]:
    """Retain the narrow initial-stage helper while sharing stage validation."""

    values = _wave_stage_fit_inputs(dataset, calibration, runtime)
    if values["stage"]["stage_id"] != WAVE_INITIAL_STAGE:
        raise ValueError("wave initial fitting requires the initial training stage")
    return (
        values["cells"],
        values["observations"],
        values["expected_rows"],
        values["experiment_id"],
        values["round_id"],
        values["candidate_sha"],
        values["calibration_sha"],
    )


def validate_wave_stage_provenance(
    profile: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the private stage chain carried by a wave fit profile."""

    stage_count = profile.get("stage_count")
    adaptive_round_count = profile.get("adaptive_round_count")
    if (
        isinstance(stage_count, bool)
        or not isinstance(stage_count, int)
        or not 1 <= stage_count <= WAVE_MAX_ADAPTIVE_ROUNDS + 1
        or isinstance(adaptive_round_count, bool)
        or not isinstance(adaptive_round_count, int)
        or adaptive_round_count != stage_count - 1
    ):
        raise ValueError("wave profile has an invalid stage count")
    stage_hashes = profile.get("stage_calibration_sha256")
    runtime_hashes = profile.get("stage_runtime_table_sha256")
    stage_metadata = profile.get("stage_metadata")
    if (
        not isinstance(stage_hashes, list)
        or len(stage_hashes) != stage_count
        or not isinstance(runtime_hashes, list)
        or len(runtime_hashes) != stage_count
        or not isinstance(stage_metadata, list)
        or len(stage_metadata) != stage_count
    ):
        raise ValueError("wave profile lacks exact stage metadata")
    for index, digest in enumerate(stage_hashes):
        _wave_lower_sha(digest, f"wave stage calibration hash {index}", 64)
    for index, digest in enumerate(runtime_hashes):
        _wave_lower_sha(digest, f"wave stage runtime hash {index}", 64)
    if len(set(stage_hashes)) != len(stage_hashes):
        raise ValueError("wave profile reuses a calibration stage hash")
    if len(set(runtime_hashes)) != len(runtime_hashes):
        raise ValueError("wave profile reuses a runtime stage hash")
    experiments: set[str] = set()
    runs: set[str] = set()
    for index, value in enumerate(stage_metadata):
        stage = _mapping(value, "wave profile stage metadata")
        expected_stage_id = WAVE_INITIAL_STAGE if index == 0 else WAVE_ADAPTIVE_STAGE
        stage_round_ordinal = stage.get("round_ordinal")
        if (
            stage.get("stage_id") != expected_stage_id
            or isinstance(stage_round_ordinal, bool)
            or not isinstance(stage_round_ordinal, int)
            or stage_round_ordinal != index
            or stage.get("calibration_set_sha256") != stage_hashes[index]
            or stage.get("runtime_table_sha256") != runtime_hashes[index]
            or stage.get("prior_stage_hashes") != stage_hashes[:index]
        ):
            raise ValueError("wave profile stage metadata is inconsistent")
        experiment_id = _wave_lower_sha(
            stage.get("experiment_id"), "wave stage experiment_id", 64
        )
        run_id = stage.get("run_id")
        if not isinstance(run_id, str) or not run_id:
            raise ValueError("wave stage run_id must be nonempty")
        if experiment_id in experiments or run_id in runs:
            raise ValueError("wave profile reuses a stage experiment or run")
        if stage.get("round_id") != _wave_stage_round_id(
            expected_stage_id, index, experiment_id
        ):
            raise ValueError("wave profile stage round identity is inconsistent")
        if index == 0:
            if stage.get("timing_used_for_selection") is not False:
                raise ValueError("wave initial profile stage must not use timing selection")
            if stage.get("selection_profile_sha256") not in (None, ""):
                raise ValueError("wave initial profile stage cannot use a selection profile")
        else:
            if stage.get("timing_used_for_selection") is not True:
                raise ValueError("wave adaptive profile stage must use timing selection")
            _wave_lower_sha(
                stage.get("selection_profile_sha256"),
                "wave adaptive selection_profile_sha256",
                64,
            )
        experiments.add(experiment_id)
        runs.add(run_id)
    if (
        profile.get("calibration_set_sha256") != stage_hashes[0]
        or profile.get("runtime_table_sha256") != runtime_hashes[0]
    ):
        raise ValueError("wave profile base stage hashes are inconsistent")
    final = stage_metadata[-1]
    final_stage_id = WAVE_INITIAL_STAGE if stage_count == 1 else WAVE_ADAPTIVE_STAGE
    final_round_ordinal = profile.get("round_ordinal")
    if (
        profile.get("stage_id") != final_stage_id
        or isinstance(final_round_ordinal, bool)
        or not isinstance(final_round_ordinal, int)
        or final_round_ordinal != stage_count - 1
        or profile.get("prior_stage_hashes") != final["prior_stage_hashes"]
        or profile.get("selection_profile_sha256") != final["selection_profile_sha256"]
        or profile.get("timing_used_for_selection")
        != final["timing_used_for_selection"]
        or profile.get("experiment_id") != final["experiment_id"]
        or profile.get("round_id") != final["round_id"]
    ):
        raise ValueError("wave profile final stage identity is inconsistent")
    return {
        "stage_count": stage_count,
        "adaptive_round_count": adaptive_round_count,
        "stage_calibration_sha256": list(stage_hashes),
        "stage_runtime_table_sha256": list(runtime_hashes),
        "stage_metadata": [dict(_mapping(value, "wave profile stage metadata"))
                            for value in stage_metadata],
    }


def _wave_selection_profile(
    profile_path: Path,
    dataset: Mapping[str, Any],
    contract: Mapping[str, Any],
    candidate_sha: str,
    prior_stage_hashes: list[str],
) -> tuple[dict[str, Any], str, WeightVector, FeatureModelDecision]:
    profile = _wave_json_mapping(profile_path, "wave training selection profile")
    profile_sha = _file_sha256(profile_path)
    if profile_sha != _sha256_bytes(_canonical_bytes(profile)):
        raise ValueError("wave training selection profile must use canonical JSON")
    required = {
        "schema_version": WAVE_CALIBRATION_PROFILE,
        "execution_profile": EXECUTION_PROFILE,
        "execution_contract": dict(contract),
        "score_id": WAVE_COST_MODEL_ID,
        "primary_quantity": WAVE_PRIMARY_QUANTITY,
        "timing_scope": CALIBRATION_TIMING_SCOPE,
        "source_sha": dataset.get("source_sha"),
        "candidate_generation_source_sha": dataset.get("source_sha"),
        "physical_execution_source_sha": contract["execution_source"],
        "candidate_set_sha256": candidate_sha,
        "fit_splits": ["training"],
    }
    for field, value in required.items():
        if profile.get(field) != value:
            raise ValueError(f"wave training selection profile has a mismatched {field}")
    provenance = validate_wave_stage_provenance(profile)
    if provenance["stage_calibration_sha256"] != prior_stage_hashes:
        raise ValueError("wave selection profile does not cover the prior stage set")
    if provenance["stage_count"] != len(prior_stage_hashes):
        raise ValueError("wave selection profile stage count does not match prior stages")
    _wave_lower_sha(dataset.get("source_sha"), "source_sha", 40)
    _wave_lower_sha(profile.get("physical_execution_source_sha"), "physical execution source", 40)
    _wave_lower_sha(profile.get("candidate_generation_source_sha"), "candidate source", 40)
    raw_weights = profile.get("weights")
    if not isinstance(raw_weights, dict) or set(raw_weights) != set(FEATURE_NAMES):
        raise ValueError("wave training selection profile weights are invalid")
    if any(raw_weights[field] != 0 for field in ("E_num", "P_wram")):
        raise ValueError("wave training selection profile activates an inactive feature")
    try:
        model = _model_from_profile(profile)
        weights = WeightVector.from_values(
            raw_weights, inactive=("E_num", "P_wram")
        )
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        raise ValueError("wave training selection profile model is invalid") from exc
    if model.mode != profile.get("requested_model_form"):
        raise ValueError("wave training selection profile model form is inconsistent")
    if weights.numeric != 0.0 or weights.wram != 0.0:
        raise ValueError("wave training selection profile activates an inactive feature")
    return profile, profile_sha, weights, model


def propose_wave_calibration_stage(
    candidate_path: Path,
    prior_calibration_paths: tuple[Path, ...],
    profile_path: Path,
    output_path: Path,
    *,
    round_ordinal: int,
) -> dict[str, Any]:
    """Build one deterministic adaptive training set from the frozen pool."""

    if (
        isinstance(round_ordinal, bool)
        or not isinstance(round_ordinal, int)
        or not 1 <= round_ordinal <= WAVE_MAX_ADAPTIVE_ROUNDS
    ):
        raise ValueError(
            f"adaptive round_ordinal must be 1..{WAVE_MAX_ADAPTIVE_ROUNDS}"
        )
    if len(prior_calibration_paths) != round_ordinal:
        raise ValueError("adaptive stage requires all prior calibration sets in order")
    dataset = _mapping(
        json.loads(candidate_path.read_text(encoding="utf-8")), "candidate dataset"
    )
    contract = execution_contract(dataset)
    if contract is None:
        raise ValueError("adaptive wave calibration requires an explicit execution contract")
    _validate_dataset_execution_contract(dataset, contract)
    candidate_sha = _sha256_bytes(_canonical_bytes(dict(dataset)))

    prior_calibrations: list[dict[str, Any]] = []
    prior_hashes: list[str] = []
    prior_cell_maps: list[dict[str, Any]] = []
    for index, path_value in enumerate(prior_calibration_paths):
        path = Path(path_value)
        calibration = _wave_json_mapping(path, "prior wave calibration candidate set")
        _calibration_candidate_index(dataset, calibration)
        stage = _wave_stage_metadata(calibration, field="prior calibration")
        expected_id = WAVE_INITIAL_STAGE if index == 0 else WAVE_ADAPTIVE_STAGE
        if stage["stage_id"] != expected_id or stage["round_ordinal"] != index:
            raise ValueError("prior wave calibration stages are not contiguous")
        if stage["prior_stage_hashes"] != prior_hashes:
            raise ValueError("prior wave calibration hash chain is inconsistent")
        digest = _file_sha256(path)
        if digest != _sha256_bytes(_canonical_bytes(calibration)):
            raise ValueError("prior wave calibration candidate set must use canonical JSON")
        prior_calibrations.append(calibration)
        prior_hashes.append(digest)
        prior_cell_maps.append({
            str(item["cell_id"]): dict(item)
            for item in calibration["cells"]
        })

    profile, profile_sha, weights, model = _wave_selection_profile(
        Path(profile_path), dataset, contract, candidate_sha, prior_hashes
    )
    initial_cells = prior_calibrations[0].get("cells")
    if not isinstance(initial_cells, list) or not initial_cells:
        raise ValueError("initial wave calibration has no training cells")
    initial_signatures = {
        str(item["cell_id"]): (
            str(item["circuit_id"]),
            str(item["topology_id"]),
            str(item["greedy_path_id"]),
        )
        for item in initial_cells
    }
    for prior_cells in prior_cell_maps[1:]:
        signatures = {
            cell_id: (
                str(item["circuit_id"]),
                str(item["topology_id"]),
                str(item["greedy_path_id"]),
            )
            for cell_id, item in prior_cells.items()
        }
        if signatures != initial_signatures:
            raise ValueError("prior adaptive calibration changed the training cell matrix")

    selected_by_cell = profile.get("selected_path_ids")
    if not isinstance(selected_by_cell, dict) or set(selected_by_cell) != set(initial_signatures):
        raise ValueError("wave training selection profile lacks exact training cells")
    candidate_speedups = profile.get("candidate_speedups")
    if not isinstance(candidate_speedups, dict):
        raise ValueError("wave training selection profile lacks measured candidate speedups")
    for cell_id, selected_path_id in selected_by_cell.items():
        measured = candidate_speedups.get(cell_id)
        if (
            not isinstance(selected_path_id, str)
            or not isinstance(measured, dict)
            or selected_path_id not in measured
        ):
            raise ValueError(
                "wave training selection profile incumbent lacks measured speedup"
            )

    circuits = {
        str(circuit["circuit_id"]): dict(circuit)
        for circuit in dataset["circuits"]
    }
    calibration_cells: list[dict[str, Any]] = []
    exhausted_cells: list[str] = []
    for initial_cell in sorted(initial_cells, key=lambda item: str(item["cell_id"])):
        cell = dict(initial_cell)
        cell_id = str(cell["cell_id"])
        circuit_id = str(cell["circuit_id"])
        topology_id = str(cell["topology_id"])
        circuit = circuits.get(circuit_id)
        if circuit is None:
            raise ValueError("adaptive calibration references an unknown circuit")
        feasible = _wave_evaluation_pool(circuit, topology_id, contract)
        by_id = {str(item["candidate_path_id"]): item for item in feasible}
        greedy_id = str(cell["greedy_path_id"])
        if greedy_id not in by_id:
            raise ValueError("adaptive calibration cell lost its greedy candidate")
        incumbent_id = selected_by_cell.get(cell_id)
        if not isinstance(incumbent_id, str) or incumbent_id not in by_id:
            raise ValueError("wave training selection profile incumbent is outside the fixed pool")
        prior_ids: set[str] = set()
        for prior_cells in prior_cell_maps:
            prior_ids.update(
                str(candidate_id)
                for candidate_id in prior_cells[cell_id]["candidate_path_ids"]
            )
        if incumbent_id not in prior_ids:
            raise ValueError("wave training selection profile incumbent was not measured")
        prior_physical_ids = {
            _wave_physical_plan_id(by_id[candidate_id], topology_id)
            for candidate_id in prior_ids
            if candidate_id in by_id
        }
        greedy = by_id[greedy_id]
        ordered_new = sorted(
            (
                candidate
                for candidate_id, candidate in by_id.items()
                if candidate_id not in prior_ids
                and _wave_physical_plan_id(candidate, topology_id)
                not in prior_physical_ids
            ),
            key=lambda candidate: (
                _wave_score(candidate, greedy, topology_id, weights),
                str(candidate["candidate_path_id"]),
            ),
        )
        new_id = (
            str(ordered_new[0]["candidate_path_id"]) if ordered_new else None
        )
        roles = [
            {"role": "greedy", "candidate_path_id": greedy_id},
            {"role": "incumbent", "candidate_path_id": incumbent_id},
            {"role": "new_fixed_pool_candidate", "candidate_path_id": new_id},
        ]
        selected: list[str] = []
        selected_physical_ids: set[str] = set()
        for role in roles:
            candidate_id = role["candidate_path_id"]
            if candidate_id is None:
                continue
            physical_id = _wave_physical_plan_id(by_id[candidate_id], topology_id)
            if physical_id in selected_physical_ids:
                continue
            selected.append(candidate_id)
            selected_physical_ids.add(physical_id)
        if not selected or selected[0] != greedy_id:
            raise ValueError("adaptive physical deduplication must retain greedy first")
        exhausted = new_id is None
        if exhausted:
            exhausted_cells.append(cell_id)
        calibration_cells.append(
            {
                "cell_id": cell_id,
                "circuit_id": circuit_id,
                "topology_id": topology_id,
                "greedy_path_id": greedy_id,
                "feature_model": cell.get("feature_model", asdict(model)),
                "candidate_roles": roles,
                "candidate_path_ids": selected,
                "prior_candidate_path_ids": sorted(prior_ids),
                "prior_physical_plan_ids": sorted(prior_physical_ids),
                "no_more_candidate": exhausted,
            }
        )

    result: dict[str, Any] = {
        "schema_version": "upmem_path_calibration_candidate_set_v1",
        "source_sha": dataset["source_sha"],
        "candidate_set_sha256": candidate_sha,
        "timing_used_for_selection": True,
        "stage_id": WAVE_ADAPTIVE_STAGE,
        "round_ordinal": round_ordinal,
        "prior_stage_hashes": prior_hashes,
        "selection_profile_sha256": profile_sha,
        "selection_profile_model": asdict(model),
        "selection_profile_source": "training_only_frozen_profile",
        "selection_rule": "greedy_then_incumbent_then_best_score_unmeasured_fixed_pool",
        "deduplicate_physical_choices": True,
        "unused_deduplicated_slots_are_refilled": False,
        "exhausted_cells": exhausted_cells,
        "execution_profile": EXECUTION_PROFILE,
        "execution_contract": dict(contract),
        "score_id": contract["cost_model_id"],
        "inactive_score_features": ["E_num", "P_wram"],
        "cells": calibration_cells,
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(_canonical_bytes(result))
    return result


def _fit_wave_initial(
    candidate_path: Path,
    dataset: Mapping[str, Any],
    calibration_path: Path,
    runtime_path: Path,
    output_dir: Path,
    *,
    samples: int,
    seed: int,
    model_form: str,
    fit_splits: tuple[str, ...],
) -> Any:
    if fit_splits != ("training",):
        raise ValueError("wave initial fitting accepts the training split only")
    return fit_wave_stages(
        candidate_path,
        ((calibration_path, runtime_path),),
        output_dir,
        samples=samples,
        seed=seed,
        model_form=model_form,
    )


def fit_wave_stages(
    candidate_path: Path,
    stage_pairs: tuple[tuple[Path, Path], ...],
    output_dir: Path,
    *,
    samples: int,
    seed: int,
    model_form: str = "six_term",
) -> Any:
    """Fit one initial and up to three explicit training wave stages."""

    if model_form == "auto":
        raise ValueError(
            "wave model comparison is pending; choose six_term or grouped explicitly"
        )
    if model_form not in {"six_term", "grouped"}:
        raise ValueError("wave model form must be six_term or grouped")
    if not stage_pairs:
        raise ValueError("wave fitting requires an initial calibration/runtime stage")
    if len(stage_pairs) > WAVE_MAX_ADAPTIVE_ROUNDS + 1:
        raise ValueError(
            f"wave fitting accepts at most {WAVE_MAX_ADAPTIVE_ROUNDS} adaptive stages"
        )
    dataset = _mapping(
        json.loads(candidate_path.read_text(encoding="utf-8")), "candidate dataset"
    )
    contract = execution_contract(dataset)
    if contract is None:
        raise ValueError("wave fitting requires an explicit execution contract")
    _validate_dataset_execution_contract(dataset, contract)

    stage_values: list[dict[str, Any]] = []
    stage_records: list[dict[str, Any]] = []
    stage_calibration_hashes: list[str] = []
    stage_runtime_hashes: list[str] = []
    experiment_ids: set[str] = set()
    run_ids: set[str] = set()
    sample_ids: set[str] = set()
    session_ids: set[str] = set()
    merged_cells: dict[str, dict[str, Any]] = {}
    initial_cell_signatures: dict[str, tuple[str, str, str]] | None = None
    candidate_sha: str | None = None
    candidate_source: str | None = None

    for index, pair in enumerate(stage_pairs):
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise TypeError("each wave stage must be a (calibration_path, runtime_path) pair")
        calibration_path, runtime_path = (Path(pair[0]), Path(pair[1]))
        calibration = _wave_json_mapping(
            calibration_path, "wave calibration candidate set"
        )
        runtime = _wave_json_mapping(runtime_path, "wave runtime calibration")
        values = _wave_stage_fit_inputs(dataset, calibration, runtime)
        calibration_sha = values["calibration_sha"]
        if _file_sha256(calibration_path) != calibration_sha:
            raise ValueError("wave calibration candidate set must use canonical JSON")
        runtime_sha = _file_sha256(runtime_path)
        if runtime_sha != _sha256_bytes(_canonical_bytes(runtime)):
            raise ValueError("wave runtime calibration must use canonical JSON")
        stage = values["stage"]
        expected_stage_id = (
            WAVE_INITIAL_STAGE if index == 0 else WAVE_ADAPTIVE_STAGE
        )
        expected_ordinal = index
        if (
            stage["stage_id"] != expected_stage_id
            or stage["round_ordinal"] != expected_ordinal
        ):
            raise ValueError("wave stages must be initial then contiguous adaptive rounds")
        if stage["prior_stage_hashes"] != stage_calibration_hashes:
            raise ValueError("wave adaptive stage prior_stage_hashes do not match stage order")
        if values["experiment_id"] in experiment_ids:
            raise ValueError("wave stages must use distinct experiment_id values")
        if values["run_id"] in run_ids:
            raise ValueError("wave stages must use distinct run_id values")
        if sample_ids.intersection(values["sample_ids"]):
            raise ValueError("wave stages reuse sample identities")
        if session_ids.intersection(values["session_ids"]):
            raise ValueError("wave stages reuse session identities")
        if candidate_sha is None:
            candidate_sha = values["candidate_sha"]
            candidate_source = values["candidate_source"]
        elif (
            values["candidate_sha"] != candidate_sha
            or values["candidate_source"] != candidate_source
        ):
            raise ValueError("wave stages drift in candidate source or candidate set")

        signatures = {
            cell_id: (
                str(cell["circuit_id"]),
                str(cell["topology_id"]),
                str(cell["greedy_path_id"]),
            )
            for cell_id, cell in values["cell_map"].items()
        }
        if initial_cell_signatures is None:
            initial_cell_signatures = signatures
        elif signatures != initial_cell_signatures:
            raise ValueError("wave stages must preserve the exact training cell matrix")
        for cell_id, cell in values["cells"].items():
            previous = merged_cells.get(cell_id)
            if previous is None:
                merged_cells[cell_id] = {
                    "cell_id": cell["cell_id"],
                    "greedy_path_id": cell["greedy_path_id"],
                    "raw_features": dict(cell["raw_features"]),
                }
                continue
            if previous["greedy_path_id"] != cell["greedy_path_id"]:
                raise ValueError("wave stages drift in greedy path identity")
            for candidate_id, raw in cell["raw_features"].items():
                if (
                    candidate_id in previous["raw_features"]
                    and previous["raw_features"][candidate_id] != raw
                ):
                    raise ValueError("wave stages drift in candidate wave features")
                previous["raw_features"][candidate_id] = raw

        experiment_ids.add(values["experiment_id"])
        run_ids.add(values["run_id"])
        sample_ids.update(values["sample_ids"])
        session_ids.update(values["session_ids"])
        stage_calibration_hashes.append(calibration_sha)
        stage_runtime_hashes.append(runtime_sha)
        stage_values.append(values)
        stage_records.append(
            {
                **stage,
                "experiment_id": values["experiment_id"],
                "run_id": values["run_id"],
                "round_id": values["round_id"],
                "calibration_set_sha256": calibration_sha,
                "runtime_table_sha256": runtime_sha,
                "raw_artifact_sha256": values["artifact_hashes"],
            }
        )

    assert candidate_sha is not None and candidate_source is not None
    observations = [
        row
        for values in stage_values
        for row in values["observations"]
    ]
    expected_rows = [
        row
        for values in stage_values
        for row in values["expected_rows"]
    ]
    fit_function = _wave_fit_function()
    result = fit_function(
        merged_cells,
        observations,
        expected_rows,
        split="training",
        model_form=model_form,
        seed=seed,
        sample_count=samples,
    )
    contract = dict(contract)
    preregistration_sha = _wave_lower_sha(
        dataset.get("preregistration_sha256"), "preregistration_sha256", 64
    )
    source_sha = _wave_lower_sha(dataset.get("source_sha"), "source_sha", 40)
    final_stage = stage_records[-1]
    profile: dict[str, Any] = {
        "schema_version": WAVE_CALIBRATION_PROFILE,
        "execution_profile": EXECUTION_PROFILE,
        "execution_contract": contract,
        "score_id": WAVE_COST_MODEL_ID,
        "primary_quantity": WAVE_PRIMARY_QUANTITY,
        "timing_scope": CALIBRATION_TIMING_SCOPE,
        "numeric_policy": contract["numeric_policy"],
        "normalization": WAVE_NORMALIZATION,
        "source_sha": source_sha,
        "source_sha_semantics": "candidate_generation_source_sha",
        "candidate_generation_source_sha": source_sha,
        "physical_execution_source_sha": EXECUTION_SOURCE,
        "reporting_tool_source_sha": _source_sha(),
        "preregistration_sha256": preregistration_sha,
        "candidate_set_sha256": candidate_sha,
        "calibration_set_sha256": stage_calibration_hashes[0],
        "runtime_table_sha256": stage_runtime_hashes[0],
        "stage_calibration_sha256": stage_calibration_hashes,
        "stage_runtime_table_sha256": stage_runtime_hashes,
        "stage_id": final_stage["stage_id"],
        "round_ordinal": final_stage["round_ordinal"],
        "experiment_id": final_stage["experiment_id"],
        "round_id": final_stage["round_id"],
        "prior_stage_hashes": list(final_stage["prior_stage_hashes"]),
        "selection_profile_sha256": final_stage["selection_profile_sha256"],
        "timing_used_for_selection": final_stage["timing_used_for_selection"],
        "stage_count": len(stage_records),
        "adaptive_round_count": len(stage_records) - 1,
        "stage_metadata": stage_records,
        "requested_model_form": model_form,
        "fit_splits": ["training"],
        "training_cell_ids": sorted(merged_cells),
        "weights": result.weights.as_mapping(),
        "feature_model": asdict(result.model),
        "selected_path_ids": dict(result.selected_path_ids),
        "cell_speedups": dict(result.cell_speedups),
        "candidate_speedups": {
            cell_id: dict(values)
            for cell_id, values in result.candidate_speedups
        },
        "paired_observation_counts": [
            list(value) for value in result.paired_observation_counts
        ],
        "geometric_mean_speedup": result.geometric_mean_speedup,
        "worst_cell_speedup": result.worst_cell_speedup,
        "minimum_cell_speedup": result.worst_cell_speedup,
        "objective": result.objective,
        "improved_cell_count": sum(
            value > 1.0 for _, value in result.cell_speedups
        ),
        "weight_search_seed": seed,
        "random_weight_samples": samples,
        "evaluated_weight_vectors": result.evaluated_weight_vectors,
        "weight_search_candidate_rows": 0,
        "weight_search_candidates_semantics": "pure_wave_fitter_result",
        "primary_objective": "geometric_mean_greedy_relative_speedup",
    }
    for field in ("workload_id", "workload_sha256", "workload_manifest_sha256"):
        if field in dataset:
            profile[field] = dataset[field]
    output_dir.mkdir(parents=True, exist_ok=True)
    profile_bytes = _canonical_bytes(profile)
    (output_dir / "physical_speedup_fit_v1.json").write_bytes(profile_bytes)
    (output_dir / "weight_search_summary.json").write_bytes(profile_bytes)
    return result


def fit(
    candidate_path: Path,
    calibration_path: Path,
    runtime_path: Path,
    output_dir: Path,
    *,
    samples: int,
    seed: int,
    model_form: str = "auto",
    fit_splits: tuple[str, ...] = ("training",),
) -> WeightFitResult:
    dataset = _mapping(
        json.loads(candidate_path.read_text(encoding="utf-8")), "candidate dataset"
    )
    contract = execution_contract(dataset)
    if contract is not None:
        return _fit_wave_initial(
            candidate_path,
            dataset,
            calibration_path,
            runtime_path,
            output_dir,
            samples=samples,
            seed=seed,
            model_form=model_form,
            fit_splits=fit_splits,
        )
    _reject_unadapted_wave_dataset(dataset, "weight fitting")
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    _validate_calibration_execution_contract(calibration, None)
    candidate_set_sha = _sha256_bytes(_canonical_bytes(dataset))
    if calibration.get("candidate_set_sha256") != candidate_set_sha:
        raise ValueError("calibration candidate-set identity does not match dataset")
    if calibration.get("source_sha") != dataset.get("source_sha"):
        raise ValueError("calibration source identity does not match dataset")
    allowed_fit_splits = {"training", "validation"}
    if not fit_splits or not set(fit_splits) <= allowed_fit_splits:
        raise ValueError("fit splits must contain training and/or validation")
    if model_form not in {"auto", "six_term", "grouped"}:
        raise ValueError("model form must be auto, six_term, or grouped")
    circuit_splits = {
        str(circuit["circuit_id"]): str(circuit.get("split", "training"))
        for circuit in dataset["circuits"]
    }
    by_circuit = {
        circuit["circuit_id"]: {
            candidate["candidate_path_id"]: _candidate_from_record(candidate)
            for candidate in circuit["candidates"]
        }
        for circuit in dataset["circuits"]
    }
    cells = []
    selected_calibration_cells = [
        item
        for item in calibration["cells"]
        if circuit_splits[str(item["circuit_id"])] in fit_splits
    ]
    if not selected_calibration_cells:
        raise ValueError("fit scope contains no calibration cells")
    for item in selected_calibration_cells:
        candidates = tuple(by_circuit[item["circuit_id"]][path] for path in item["candidate_path_ids"])
        cells.append(
            TrainingCell(
                cell_id=item["cell_id"],
                topology=item["topology_id"],
                candidates=candidates,
                greedy_path_id=item["greedy_path_id"],
            )
        )
    measurements = []
    physical_execution_sources: set[str] = set()
    expected_cells = {cell.cell_id: cell for cell in cells}
    expected_physical_plans = {
        (item["cell_id"], candidate_id): next(
            topology["physical_plan_id"]
            for topology in next(
                candidate
                for candidate in next(
                    circuit for circuit in dataset["circuits"]
                    if circuit["circuit_id"] == item["circuit_id"]
                )["candidates"]
                if candidate["candidate_path_id"] == candidate_id
            )["topologies"]
            if topology["topology_id"] == item["topology_id"]
        )
        for item in selected_calibration_cells
        for candidate_id in item["candidate_path_ids"]
    }
    with runtime_path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            row_split = str(row.get("split"))
            if row_split not in allowed_fit_splits:
                raise ValueError("calibration runtime table contains an invalid split")
            if row_split not in fit_splits:
                continue
            if row.get("attempt_type") not in {"warmup", "measurement"}:
                raise ValueError("calibration runtime table has an invalid attempt type")
            if row.get("candidate_generation_source_sha") != dataset["source_sha"]:
                raise ValueError(
                    "calibration runtime candidate source does not match candidate dataset"
                )
            physical_source = str(row.get("physical_execution_source_sha", ""))
            if len(physical_source) != 40 or any(
                character not in "0123456789abcdef" for character in physical_source
            ):
                raise ValueError("calibration runtime physical source is not a full SHA")
            physical_execution_sources.add(physical_source)
            if row.get("timing_scope") != "steady_execution_v1":
                raise ValueError("calibration runtime table has an invalid timing scope")
            if row.get("status") != "success":
                raise ValueError("calibration runtime table contains a failed attempt")
            if row.get("validation") not in {"true", "passed", "1"}:
                raise ValueError("calibration runtime table contains an invalid output")
            if row.get("fallback") not in {"false", "0"}:
                raise ValueError("calibration runtime table contains fallback execution")
            cell = expected_cells.get(str(row.get("cell_id")))
            if cell is None:
                raise ValueError("calibration runtime table references an unknown cell")
            key = (cell.cell_id, str(row.get("candidate_path_id")))
            if row.get("physical_plan_id") != expected_physical_plans.get(key):
                raise ValueError("calibration runtime physical-plan identity mismatch")
            if row["attempt_type"] == "warmup":
                continue
            block = int(row["block"])
            if block not in {1, 2, 3}:
                raise ValueError("calibration measurement block must be 1, 2, or 3")
            measurements.append(
                RuntimeMeasurement(
                    cell_id=str(row["cell_id"]),
                    candidate_id=str(row["candidate_path_id"]),
                    runtime_s=float(row["total_wall_s"]),
                    split="train",
                    source_sha=physical_source,
                    timing_scope=str(row["timing_scope"]),
                    status=str(row["status"]),
                    observation_id=str(block),
                )
            )
    if len(physical_execution_sources) != 1:
        raise ValueError("calibration runtime table has mixed physical execution sources")
    physical_execution_source = next(iter(physical_execution_sources))
    output_dir.mkdir(parents=True, exist_ok=True)
    search_path = output_dir / "weight_search_candidates.csv"
    columns = (
        "weights_json", "selected_path_ids_json", "cell_speedups_json",
        "geometric_mean_speedup", "minimum_cell_speedup", "improved_cell_count",
        "equivalent_weight_vector_count",
    )
    representatives: dict[tuple[tuple[str, str], ...], WeightFitResult] = {}
    outcome_counts: dict[tuple[tuple[str, str], ...], int] = {}

    def result_order(item: WeightFitResult) -> tuple[float, float, int, tuple[float, ...]]:
        return (
            round(item.geometric_mean_speedup, 12),
            round(item.minimum_cell_speedup, 12),
            item.improved_cell_count,
            tuple(-value for value in item.weights.as_tuple()),
        )

    def record(item: WeightFitResult) -> None:
        key = item.selected_path_ids
        outcome_counts[key] = outcome_counts.get(key, 0) + 1
        current = representatives.get(key)
        if current is None or result_order(item) > result_order(current):
            representatives[key] = item

    model = None if model_form == "auto" else explicit_feature_model(model_form)
    result = fit_weights(
        tuple(cells), tuple(measurements), seed=seed,
        model=model,
        random_sample_count=samples, evaluation_callback=record,
    )
    ordered_representatives = sorted(
        representatives.values(),
        key=lambda item: (
            tuple(item.selected_path_ids),
            tuple(item.weights.as_tuple()),
        ),
    )
    with search_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for item in ordered_representatives:
            writer.writerow(
                {
                    "weights_json": json.dumps(item.weights.as_mapping(), sort_keys=True, separators=(",", ":")),
                    "selected_path_ids_json": json.dumps(dict(item.selected_path_ids), sort_keys=True, separators=(",", ":")),
                    "cell_speedups_json": json.dumps(dict(item.cell_speedups), sort_keys=True, separators=(",", ":")),
                    "geometric_mean_speedup": item.geometric_mean_speedup,
                    "minimum_cell_speedup": item.minimum_cell_speedup,
                    "improved_cell_count": item.improved_cell_count,
                    "equivalent_weight_vector_count": outcome_counts[item.selected_path_ids],
                }
            )
    profile = {
        "schema_version": "physical_speedup_fit_v1",
        "score_id": COST_MODEL_ID,
        "source_sha": dataset["source_sha"],
        "source_sha_semantics": "candidate_generation_source_sha",
        "candidate_generation_source_sha": dataset["source_sha"],
        "physical_execution_source_sha": physical_execution_source,
        "reporting_tool_source_sha": _source_sha(),
        "candidate_set_sha256": candidate_set_sha,
        "calibration_set_sha256": _file_sha256(calibration_path),
        "runtime_table_sha256": _file_sha256(runtime_path),
        "weights": result.weights.as_mapping(),
        "feature_model": asdict(result.model),
        "requested_model_form": model_form,
        "fit_splits": list(fit_splits),
        "training_cell_ids": [cell.cell_id for cell in cells],
        "selected_path_ids": dict(result.selected_path_ids),
        "cell_speedups": dict(result.cell_speedups),
        "geometric_mean_speedup": result.geometric_mean_speedup,
        "minimum_cell_speedup": result.minimum_cell_speedup,
        "improved_cell_count": result.improved_cell_count,
        "weight_search_seed": seed,
        "random_weight_samples": samples,
        "evaluated_weight_vectors": result.evaluated_weight_vectors,
        "weight_search_candidate_rows": len(ordered_representatives),
        "weight_search_candidates_semantics": (
            "best_lexicographic_weight_vector_per_distinct_training_path_selection"
        ),
        "normalization": "log((candidate+1)/(greedy+1))",
        "primary_objective": "geometric_mean_greedy_relative_speedup",
    }
    (output_dir / "physical_speedup_fit_v1.json").write_bytes(_canonical_bytes(profile))
    (output_dir / "weight_search_summary.json").write_bytes(_canonical_bytes(profile))
    return result


def _model_from_profile(profile: dict[str, Any]) -> FeatureModelDecision:
    value = profile["feature_model"]
    return FeatureModelDecision(
        mode=value["mode"],
        active_features=tuple(value["active_features"]),
        zero_range_features=tuple(value["zero_range_features"]),
        correlated_pairs=tuple(tuple(pair) for pair in value["correlated_pairs"]),
        matrix_rank=int(value["matrix_rank"]),
        rank_tolerance=float(value["rank_tolerance"]),
        reason=str(value["reason"]),
    )


def _wave_profile_for_evaluation(
    profile_path: Path,
    dataset: Mapping[str, Any],
    contract: Mapping[str, Any],
    candidate_sha: str,
) -> tuple[dict[str, Any], str, WeightVector, FeatureModelDecision]:
    profile = _wave_json_mapping(profile_path, "wave frozen profile")
    profile_hash = _sha256_bytes(_canonical_bytes(profile))
    required_fields = {
        "schema_version": WAVE_CALIBRATION_PROFILE,
        "execution_profile": EXECUTION_PROFILE,
        "execution_contract": dict(contract),
        "score_id": WAVE_COST_MODEL_ID,
        "primary_quantity": WAVE_PRIMARY_QUANTITY,
        "timing_scope": CALIBRATION_TIMING_SCOPE,
        "numeric_policy": contract["numeric_policy"],
        "normalization": WAVE_NORMALIZATION,
        "source_sha": dataset.get("source_sha"),
        "candidate_generation_source_sha": dataset.get("source_sha"),
        "physical_execution_source_sha": contract["execution_source"],
        "candidate_set_sha256": candidate_sha,
        "fit_splits": ["training"],
    }
    for field, value in required_fields.items():
        if profile.get(field) != value:
            raise ValueError(f"wave frozen profile has a mismatched {field}")
    source_sha = _wave_lower_sha(dataset.get("source_sha"), "source_sha", 40)
    _wave_lower_sha(
        profile.get("candidate_generation_source_sha"),
        "candidate_generation_source_sha",
        40,
    )
    _wave_lower_sha(
        profile.get("physical_execution_source_sha"),
        "physical_execution_source_sha",
        40,
    )
    _wave_lower_sha(
        dataset.get("preregistration_sha256"), "preregistration_sha256", 64
    )
    if profile.get("preregistration_sha256") != dataset["preregistration_sha256"]:
        raise ValueError("wave frozen profile preregistration identity does not match dataset")
    for field in ("calibration_set_sha256", "runtime_table_sha256"):
        _wave_lower_sha(profile.get(field), field, 64)
    provenance = validate_wave_stage_provenance(profile)
    stage_count = provenance["stage_count"]
    stage_metadata = provenance["stage_metadata"]
    if "reporting_tool_source_sha" in profile:
        _wave_lower_sha(
            profile.get("reporting_tool_source_sha"), "reporting_tool_source_sha", 40
        )
    experiment_id = _wave_lower_sha(
        profile.get("experiment_id"), "wave profile experiment_id", 64
    )
    final_stage_id = WAVE_INITIAL_STAGE if stage_count == 1 else WAVE_ADAPTIVE_STAGE
    if profile.get("round_id") != _wave_stage_round_id(
        final_stage_id, stage_count - 1, experiment_id
    ):
        raise ValueError("wave frozen profile round identity is inconsistent")
    final_stage = stage_metadata[-1]
    if profile.get("prior_stage_hashes") != final_stage["prior_stage_hashes"]:
        raise ValueError("wave frozen profile prior stage hashes are inconsistent")
    if profile.get("selection_profile_sha256") != final_stage[
        "selection_profile_sha256"
    ]:
        raise ValueError("wave frozen profile selection profile binding is inconsistent")
    if profile.get("timing_used_for_selection") != final_stage[
        "timing_used_for_selection"
    ]:
        raise ValueError("wave frozen profile timing-selection state is inconsistent")
    for field in _WAVE_PILOT_FIELDS:
        if profile.get(field) not in (None, False, "", [], {}):
            raise ValueError("wave frozen profile contains migrated pilot weights or profile")
    if profile.get("requested_model_form") not in {"six_term", "grouped"}:
        raise ValueError("wave frozen profile requires an explicit model form")
    try:
        model = _model_from_profile(profile)
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        raise ValueError("wave frozen profile feature model is invalid") from exc
    if model.mode != profile["requested_model_form"] or not model.active_features:
        raise ValueError("wave frozen profile model form is inconsistent")
    if model.mode == "six_term":
        allowed = set(FEATURE_NAMES[:4])
    else:
        allowed = set(GROUP_FEATURE_NAMES)
    if not set(model.active_features) <= allowed:
        raise ValueError("wave frozen profile activates an unsupported feature")
    raw_weights = profile.get("weights")
    if not isinstance(raw_weights, dict) or set(raw_weights) != set(FEATURE_NAMES):
        raise ValueError("wave frozen profile weights must contain six canonical features")
    for feature in ("E_num", "P_wram"):
        value = raw_weights[feature]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value != 0:
            raise ValueError(f"wave frozen profile {feature} weight must be zero")
    try:
        weights = WeightVector.from_values(
            raw_weights, inactive=("E_num", "P_wram")
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("wave frozen profile weights are invalid") from exc
    if weights.numeric != 0.0 or weights.wram != 0.0:
        raise ValueError("wave frozen profile E_num and P_wram weights must be zero")
    if model.mode == "six_term":
        if any(
            float(raw_weights[feature]) != 0.0 and feature not in model.active_features
            for feature in FEATURE_NAMES
        ):
            raise ValueError("wave frozen profile assigns weight to an inactive feature")
    else:
        grouped = {
            "movement": weights.host_dpu + weights.mram_wram,
            "compute": weights.dpu_work,
            "coordination": weights.sync,
        }
        if any(value != 0.0 and feature not in model.active_features
               for feature, value in grouped.items()):
            raise ValueError("wave frozen profile assigns weight to an inactive feature")
        if not math.isclose(
            weights.host_dpu,
            weights.mram_wram,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ValueError("wave grouped profile requires equal-half movement weights")
    if profile.get("source_sha") != source_sha:
        raise ValueError("wave frozen profile source does not match dataset")
    return profile, profile_hash, weights, model


def _wave_evaluation_pool(
    circuit: Mapping[str, Any],
    topology_id: str,
    contract: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    candidates = circuit.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError("wave evaluation circuit must contain candidates")
    feasible: list[dict[str, Any]] = []
    for candidate_value in candidates:
        candidate = dict(_mapping(candidate_value, "wave evaluation candidate"))
        matches = [
            item for item in candidate.get("topologies", [])
            if isinstance(item, Mapping) and item.get("topology_id") == topology_id
        ]
        if len(matches) != 1:
            raise ValueError(
                f"wave candidate {candidate.get('candidate_path_id')} must have one {topology_id} topology"
            )
        topology = dict(matches[0])
        if topology.get("feasible") is not True:
            continue
        topology_spec = _mapping(topology.get("topology"), "wave topology resources")
        _wave_topology_resources(topology_id, topology_spec)
        admission = _mapping(
            topology.get("resource_admission"), "wave resource admission"
        )
        _wave_scaling_admission(admission, topology_spec)
        memory = _mapping(topology.get("memory_admission"), "wave memory admission")
        estimate = memory.get("declared_executor_memory_estimate_bytes")
        budget = memory.get("configured_budget_bytes")
        reserve = memory.get("configured_reserve_bytes")
        required = memory.get("required_bytes")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in (estimate, budget, reserve, required)
        ):
            raise ValueError("wave evaluation memory admission facts are invalid")
        if (
            memory.get("passed") is not True
            or required != estimate + reserve
            or required > budget
            or topology.get("host_memory_estimate_bytes") != estimate
        ):
            raise ValueError("wave evaluation candidate lacks passed memory admission")
        facts = _wave_facts_from_record(candidate, topology_id)
        if topology.get("features") != facts["raw"].as_mapping():
            raise ValueError("wave evaluation raw features do not match wave facts")
        feasible.append(candidate)
    if not feasible:
        raise ValueError(f"no admitted wave candidates for {circuit.get('circuit_id')}/{topology_id}")
    greedy = [candidate for candidate in feasible if candidate.get("is_greedy") is True]
    if len(greedy) != 1:
        raise ValueError(
            f"wave evaluation requires exactly one feasible greedy candidate for "
            f"{circuit.get('circuit_id')}/{topology_id}"
        )
    return tuple(feasible)


def _wave_normalized_configuration(path: Path) -> tuple[dict[str, Any], str]:
    """Load the archived YAML through the canonical experiment validator."""

    normalized = json.loads(canonical_json(load_experiment_config(path)))
    normalized_mapping = dict(_mapping(normalized, "normalized physical configuration"))
    return normalized_mapping, _sha256_bytes(_canonical_bytes(normalized_mapping))


def _wave_sha256sums(archive_root: Path) -> dict[str, str]:
    """Read the fixed archive checksum file without mixing binary identities."""

    path = archive_root / "SHA256SUMS"
    try:
        lines = path.read_text(encoding="ascii").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError("frozen preregistration SHA256SUMS is unreadable") from exc
    checksums: dict[str, str] = {}
    for line_number, line in enumerate(lines, start=1):
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            raise ValueError(f"frozen preregistration SHA256SUMS line {line_number} is invalid")
        digest = _wave_lower_sha(
            parts[0], f"frozen preregistration checksum line {line_number}", 64
        )
        name = parts[1].strip()
        if name.startswith("*"):
            name = name[1:]
        while name.startswith("./"):
            name = name[2:]
        if name.startswith("preregistration/"):
            name = name[len("preregistration/"):]
        if name in checksums:
            raise ValueError(f"frozen preregistration SHA256SUMS duplicates {name}")
        checksums[name] = digest
    required = {"physical.yml", "physical.yml.provenance.json", "binary_sha256.json"}
    if not required <= set(checksums):
        missing = sorted(required - set(checksums))
        raise ValueError(
            "frozen preregistration SHA256SUMS lacks required files: "
            f"{', '.join(missing)}"
        )
    return checksums


def _wave_private_checksum(
    checksums: Mapping[str, str], filename: str, path: Path
) -> str:
    digest = checksums.get(filename)
    if digest is None:
        raise ValueError(f"frozen preregistration checksum manifest lacks {filename}")
    actual = _file_sha256(path)
    if digest != actual:
        raise ValueError(f"frozen preregistration checksum mismatch: {filename}")
    return actual


def _wave_private_archive(
    raw_root: Path,
    provenance_path: Path,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify the fixed prelaunch archive used by the physical runner."""

    archive_root = raw_root.parent / "preregistration"
    configuration_path = archive_root / "physical.yml"
    archived_provenance_path = archive_root / "physical.yml.provenance.json"
    binary_manifest_path = archive_root / "binary_sha256.json"
    checksums_path = archive_root / "SHA256SUMS"
    for path in (
        configuration_path,
        archived_provenance_path,
        binary_manifest_path,
        checksums_path,
    ):
        if not path.is_file():
            raise ValueError(f"frozen preregistration archive lacks {path.name}")
    supplied_path = Path(provenance_path)
    try:
        supplied_bytes = supplied_path.read_bytes()
        archived_bytes = archived_provenance_path.read_bytes()
    except OSError as exc:
        raise ValueError("wave pretest provenance sidecar is unreadable") from exc
    if supplied_bytes != archived_bytes:
        raise ValueError(
            "wave pretest provenance must be the fixed archived physical.yml sidecar"
        )
    provenance = _wave_json_mapping(
        archived_provenance_path, "wave physical configuration provenance"
    )
    if archived_bytes != _canonical_bytes(provenance):
        raise ValueError("wave physical configuration provenance is not canonical JSON")
    checksums = _wave_sha256sums(archive_root)
    binary_manifest = _load_frozen_wave_binary_manifest(raw_root)
    private_hashes = {
        "physical.yml": _wave_private_checksum(
            checksums, "physical.yml", configuration_path
        ),
        "physical.yml.provenance.json": _wave_private_checksum(
            checksums, "physical.yml.provenance.json", archived_provenance_path
        ),
        "binary_sha256.json": _wave_private_checksum(
            checksums, "binary_sha256.json", binary_manifest_path
        ),
        "SHA256SUMS": _file_sha256(checksums_path),
    }
    normalized, normalized_sha = _wave_normalized_configuration(configuration_path)
    configuration_sha = _file_sha256(configuration_path)
    if provenance.get("configuration_sha256") != configuration_sha:
        raise ValueError("wave provenance configuration_sha256 does not match physical.yml")
    if provenance.get("normalized_configuration_sha256") != normalized_sha:
        raise ValueError(
            "wave provenance normalized_configuration_sha256 does not match physical.yml"
        )
    manifest_experiment = _mapping(
        _mapping(manifest.get("configuration"), "manifest configuration").get("experiment"),
        "manifest normalized experiment configuration",
    )
    manifest_normalized = json.loads(canonical_json(manifest_experiment))
    if manifest_normalized != normalized:
        raise ValueError(
            "manifest.configuration.experiment does not match archived physical.yml"
        )
    manifest_normalized_sha = _sha256_bytes(_canonical_bytes(manifest_normalized))
    if manifest_normalized_sha != normalized_sha:
        raise ValueError("manifest normalized configuration hash does not match archive")
    return {
        "configuration_path": str(configuration_path),
        "provenance_path": str(archived_provenance_path),
        "binary_manifest_path": str(binary_manifest_path),
        "configuration_sha256": configuration_sha,
        "normalized_configuration_sha256": normalized_sha,
        "binary_manifest": binary_manifest,
        "checksums": checksums,
        "private_hashes": private_hashes,
        "provenance": provenance,
    }


def _validate_wave_calibration_archive_provenance(
    archive: Mapping[str, Any],
    dataset: Mapping[str, Any],
    calibration: Mapping[str, Any],
    stage_metadata: Mapping[str, Any],
    contract: Mapping[str, Any],
    *,
    candidate_set_sha: str,
    candidate_source: str,
    calibration_set_sha: str,
) -> dict[str, Any]:
    """Bind the archived sidecar to the exact calibration caller inputs."""

    provenance = dict(
        _mapping(archive.get("provenance"), "wave calibration archive provenance")
    )
    expected = {
        "schema_version": "upmem_path_experiment_provenance_v1",
        "source_sha": candidate_source,
        "candidate_set_sha256": candidate_set_sha,
        "preregistration_sha256": dataset.get("preregistration_sha256"),
        "mode": "calibration",
        "stage": dict(stage_metadata),
        "execution_profile": EXECUTION_PROFILE,
        "execution_contract": dict(contract),
        "score_id": contract["cost_model_id"],
        "numeric_policy": contract["numeric_policy"],
        "primary_quantity": WAVE_PRIMARY_QUANTITY,
        "timing_scope": CALIBRATION_TIMING_SCOPE,
        "calibration_set_sha256": calibration_set_sha,
    }
    for field, value in expected.items():
        if provenance.get(field) != value:
            raise ValueError(
                f"wave calibration archive provenance has a mismatched {field}"
            )
    _wave_lower_sha(provenance["source_sha"], "archive provenance source_sha", 40)
    _wave_lower_sha(
        provenance["candidate_set_sha256"],
        "archive provenance candidate_set_sha256",
        64,
    )
    _wave_lower_sha(
        provenance["preregistration_sha256"],
        "archive provenance preregistration_sha256",
        64,
    )
    _wave_lower_sha(
        provenance["calibration_set_sha256"],
        "archive provenance calibration_set_sha256",
        64,
    )
    optional_source = provenance.get("physical_execution_source_sha")
    if optional_source is not None and optional_source != contract["execution_source"]:
        raise ValueError(
            "wave calibration archive provenance has a mismatched physical source"
        )
    _wave_stage_metadata(provenance["stage"], field="archive provenance stage")
    if provenance["stage"] != dict(stage_metadata):
        raise ValueError("wave calibration archive stage does not match calibration input")
    if calibration.get("candidate_set_sha256") != candidate_set_sha:
        raise ValueError("calibration candidate-set identity is not archive-bound")
    return provenance


def _wave_pretest_stage(
    provenance: Mapping[str, Any],
    *,
    mode: str,
    split: str,
    profile_hash: str,
) -> dict[str, Any]:
    value = _mapping(provenance.get("stage"), "wave pretest provenance stage")
    expected_stage_id = (
        "development_confirmation" if mode == "confirmation" else split
    )
    required = {
        "stage_id",
        "round_ordinal",
        "prior_stage_hashes",
        "selection_profile_sha256",
        "timing_used_for_selection",
    }
    if set(value) != required:
        raise ValueError("wave pretest provenance stage tuple is incomplete")
    if value["stage_id"] != expected_stage_id:
        raise ValueError("wave pretest provenance stage does not match mode/split")
    if (
        isinstance(value["round_ordinal"], bool)
        or not isinstance(value["round_ordinal"], int)
        or value["round_ordinal"] != 0
    ):
        raise ValueError("wave pretest provenance stage round_ordinal must be zero")
    if value["prior_stage_hashes"] != []:
        raise ValueError("wave pretest provenance stage cannot have prior stages")
    if value["selection_profile_sha256"] != profile_hash:
        raise ValueError("wave pretest provenance stage profile hash is inconsistent")
    if value["timing_used_for_selection"] is not False:
        raise ValueError("wave pretest selection must be timing-independent")
    return dict(value)


def _wave_pretest_provenance(
    provenance_path: Path,
    provenance: Mapping[str, Any],
    dataset: Mapping[str, Any],
    contract: Mapping[str, Any],
    *,
    mode: str,
    split: str,
    candidate_sha: str,
    profile_hash: str,
    archive: Mapping[str, Any],
) -> dict[str, Any]:
    del provenance_path
    expected = {
        "schema_version": "upmem_path_experiment_provenance_v1",
        "source_sha": dataset.get("source_sha"),
        "candidate_set_sha256": candidate_sha,
        "preregistration_sha256": dataset.get("preregistration_sha256"),
        "mode": mode,
        "selection_split": split,
        "execution_target": "physical",
        "execution_profile": EXECUTION_PROFILE,
        "execution_contract": dict(contract),
        "score_id": contract["cost_model_id"],
        "numeric_policy": contract["numeric_policy"],
        "primary_quantity": WAVE_PRIMARY_QUANTITY,
        "timing_scope": CALIBRATION_TIMING_SCOPE,
        "calibration_set_sha256": None,
        "fitted_profile_sha256": profile_hash,
    }
    for field, value in expected.items():
        if provenance.get(field) != value:
            raise ValueError(f"wave pretest provenance has a mismatched {field}")
    for field, length in (
        ("source_sha", 40),
        ("candidate_set_sha256", 64),
        ("preregistration_sha256", 64),
        ("configuration_sha256", 64),
        ("normalized_configuration_sha256", 64),
        ("fitted_profile_sha256", 64),
    ):
        _wave_lower_sha(provenance.get(field), f"wave provenance {field}", length)
    if provenance["configuration_sha256"] != archive["configuration_sha256"]:
        raise ValueError("wave provenance configuration hash is not archive-bound")
    if provenance["normalized_configuration_sha256"] != archive[
        "normalized_configuration_sha256"
    ]:
        raise ValueError("wave provenance normalized configuration hash is not archive-bound")
    if not isinstance(provenance.get("profile_path"), str) or not provenance["profile_path"]:
        raise ValueError("wave pretest provenance lacks profile_path")
    selection_path = provenance.get("selection_path")
    if mode == "confirmation" and selection_path is not None:
        raise ValueError("confirmation provenance must not name a selection artifact")
    if mode == "evaluation" and (
        not isinstance(selection_path, str) or not selection_path
    ):
        raise ValueError("evaluation provenance requires selection_path")
    selected = provenance.get("selected")
    roles = provenance.get("selection_roles")
    if not isinstance(selected, list) or not selected:
        raise ValueError("wave pretest provenance lacks selected paths")
    if not isinstance(roles, list) or not roles:
        raise ValueError("wave pretest provenance lacks selection roles")
    _wave_pretest_stage(provenance, mode=mode, split=split, profile_hash=profile_hash)
    return dict(provenance)


def _wave_pretest_selection(
    dataset: Mapping[str, Any],
    provenance: Mapping[str, Any],
    profile: Mapping[str, Any],
    weights: WeightVector,
    model: FeatureModelDecision,
    contract: Mapping[str, Any],
    *,
    mode: str,
    split: str,
) -> dict[tuple[str, str, str], dict[str, Any]]:
    circuits = {
        str(item["circuit_id"]): dict(_mapping(item, "wave pretest circuit"))
        for item in dataset["circuits"]
    }
    expected_circuits = {
        circuit_id for circuit_id, circuit in circuits.items()
        if circuit.get("split") == split
    }
    selected_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    for value in provenance["selected"]:
        item = _mapping(value, "wave pretest selected path")
        circuit_id = item.get("circuit_id")
        topology_id = item.get("topology_id")
        candidate_id = item.get("candidate_path_id")
        if not all(isinstance(value, str) for value in (circuit_id, topology_id, candidate_id)):
            raise ValueError("wave pretest selected path identity is invalid")
        key = (circuit_id, topology_id, candidate_id)
        if key in selected_by_key:
            raise ValueError("wave pretest selected paths contain duplicates")
        if circuit_id not in circuits or circuits[circuit_id].get("split") != split:
            raise ValueError("wave pretest selected path has a mismatched circuit split")
        if topology_id not in WAVE_TOPOLOGY_RESOURCES:
            raise ValueError("wave pretest selected path has an unsupported topology")
        for field in ("candidate_path_id", "logical_plan_id", "physical_plan_id"):
            _wave_lower_sha(item.get(field), f"selected {field}", 64)
        candidate = next(
            (
                dict(_mapping(candidate_value, "wave pretest candidate"))
                for candidate_value in circuits[circuit_id]["candidates"]
                if isinstance(candidate_value, Mapping)
                and candidate_value.get("candidate_path_id") == candidate_id
            ),
            None,
        )
        if candidate is None:
            raise ValueError("wave pretest selected path is outside candidate dataset")
        if item["logical_plan_id"] != candidate.get("logical_plan_id"):
            raise ValueError("wave pretest selected logical-plan identity is inconsistent")
        topology = next(
            (
                dict(_mapping(topology_value, "wave pretest topology"))
                for topology_value in candidate.get("topologies", [])
                if isinstance(topology_value, Mapping)
                and topology_value.get("topology_id") == topology_id
            ),
            None,
        )
        if topology is None or item["physical_plan_id"] != topology.get("physical_plan_id"):
            raise ValueError("wave pretest selected physical-plan identity is inconsistent")
        selected_by_key[key] = {"selected": dict(item), "candidate": candidate, "topology": topology}
    if {key[0] for key in selected_by_key} != expected_circuits:
        raise ValueError("wave pretest provenance does not cover the requested split")

    role_by_key: dict[tuple[str, str, str], set[str]] = {}
    for value in provenance["selection_roles"]:
        item = _mapping(value, "wave pretest selection role")
        values = (item.get("circuit_id"), item.get("topology_id"), item.get("candidate_path_id"))
        if not all(isinstance(value, str) for value in values):
            raise ValueError("wave pretest selection role identity is invalid")
        key = values
        role_values = item.get("roles")
        if not isinstance(role_values, list) or not role_values or len(set(role_values)) != len(role_values):
            raise ValueError("wave pretest selection roles are invalid")
        if key not in selected_by_key:
            raise ValueError("wave pretest role is outside selected paths")
        if key in role_by_key:
            raise ValueError("wave pretest selection roles contain duplicates")
        role_by_key[key] = set(role_values)
    if set(role_by_key) != set(selected_by_key):
        raise ValueError("wave pretest roles do not cover selected paths exactly")
    expected_role_names = {"greedy", "upmem_selected"} if mode == "confirmation" else {
        "greedy", "minimum_flops", "upmem_selected"
    }

    expected: dict[tuple[str, str, str], dict[str, Any]] = {}
    profile_selected = profile.get("selected_path_ids")
    profile_measured = profile.get("candidate_speedups")
    for (circuit_id, topology_id), cell_items in sorted(
        ((key[:2], value) for key, value in selected_by_key.items()),
        key=lambda value: value[0],
    ):
        del cell_items
        circuit = circuits[circuit_id]
        pool = _wave_evaluation_pool(circuit, topology_id, contract)
        by_id = {str(item["candidate_path_id"]): item for item in pool}
        greedy = [item for item in pool if item.get("is_greedy") is True]
        if len(greedy) != 1:
            raise ValueError("wave pretest requires exactly one feasible greedy candidate")
        greedy_id = str(greedy[0]["candidate_path_id"])
        cell_id = f"{circuit_id}:{topology_id}"
        if mode == "confirmation":
            if not isinstance(profile_measured, Mapping) or not isinstance(profile_selected, Mapping):
                raise ValueError("confirmation requires frozen measured-pool profile fields")
            measured_ids = profile_measured.get(cell_id)
            if not isinstance(measured_ids, Mapping) or not measured_ids or not set(measured_ids) <= set(by_id):
                raise ValueError("confirmation profile measured pool is invalid")
            measured_candidates = [by_id[str(path_id)] for path_id in sorted(measured_ids)]
            if greedy_id not in {str(item["candidate_path_id"]) for item in measured_candidates}:
                raise ValueError("confirmation profile measured pool lacks greedy")
            frozen_selected = profile_selected.get(cell_id)
            expected_score_id = min(
                (
                    (_wave_score(item, greedy[0], topology_id, weights), str(item["candidate_path_id"]))
                    for item in measured_candidates
                ),
                key=lambda value: (value[0], value[1]),
            )[1]
            if frozen_selected != expected_score_id:
                raise ValueError("confirmation path does not match frozen pretest profile")
        else:
            flop_id = str(min(
                pool,
                key=lambda item: (
                    _mapping(item.get("conventional_features"), "wave conventional features")["flops"],
                    str(item["candidate_path_id"]),
                ),
            )["candidate_path_id"])
            expected_score_id = min(
                (
                    (_wave_score(item, greedy[0], topology_id, weights), str(item["candidate_path_id"]))
                    for item in pool
                ),
                key=lambda value: (value[0], value[1]),
            )[1]
        expected_role_ids = {
            "greedy": greedy_id,
            "upmem_selected": expected_score_id,
        }
        if mode == "evaluation":
            expected_role_ids["minimum_flops"] = flop_id
        expected_cell_keys = {
            (circuit_id, topology_id, candidate_id)
            for candidate_id in expected_role_ids.values()
        }
        actual_cell_keys = {
            key for key in selected_by_key if key[:2] == (circuit_id, topology_id)
        }
        if actual_cell_keys != expected_cell_keys:
            raise ValueError("wave pretest selected paths do not match frozen role set")
        actual_roles = {
            role for key in actual_cell_keys for role in role_by_key[key]
        }
        if actual_roles != expected_role_names:
            raise ValueError("wave pretest selection roles are incomplete")
        for role, candidate_id in expected_role_ids.items():
            if role_by_key[(circuit_id, topology_id, candidate_id)] != {
                role for role, expected_id in expected_role_ids.items() if expected_id == candidate_id
            }:
                raise ValueError("wave pretest role identity is inconsistent")
        for candidate_id in sorted(expected_role_ids.values()):
            item = selected_by_key[(circuit_id, topology_id, candidate_id)]
            expected[(circuit_id, topology_id, candidate_id)] = {
                "cell_id": cell_id,
                "circuit": circuit,
                "candidate": item["candidate"],
                "topology": item["topology"],
                "greedy_path_id": greedy_id,
            }
    return expected


def extract_wave_pretest(
    raw_dir: Path,
    candidate_path: Path,
    provenance_path: Path,
    profile_path: Path,
    output_dir: Path,
    *,
    mode: str,
    split: str,
) -> dict[str, Any]:
    """Extract physical 1+5 confirmation or held-out wave observations."""

    if mode not in {"confirmation", "evaluation"}:
        raise ValueError("wave pretest mode must be confirmation or evaluation")
    if (mode == "confirmation" and split != "training") or (
        mode == "evaluation" and split not in {"validation", "test"}
    ):
        raise ValueError("wave pretest mode and split are incompatible")
    dataset = _wave_json_mapping(candidate_path, "wave candidate dataset")
    contract = execution_contract(dataset)
    if contract is None:
        raise ValueError("wave pretest extraction requires an explicit execution contract")
    _validate_dataset_execution_contract(dataset, contract)
    candidate_sha = _sha256_bytes(_canonical_bytes(dataset))
    profile, profile_hash, weights, model = _wave_profile_for_evaluation(
        profile_path, dataset, contract, candidate_sha
    )
    if _file_sha256(profile_path) != profile_hash:
        raise ValueError("wave frozen profile must use canonical JSON")
    manifest, samples, sessions = load_artifacts(raw_dir)
    archive = _wave_private_archive(Path(raw_dir), provenance_path, manifest)
    provenance = _wave_pretest_provenance(
        provenance_path,
        archive["provenance"],
        dataset,
        contract,
        mode=mode,
        split=split,
        candidate_sha=candidate_sha,
        profile_hash=profile_hash,
        archive=archive,
    )
    expected = _wave_pretest_selection(
        dataset,
        provenance,
        profile,
        weights,
        model,
        contract,
        mode=mode,
        split=split,
    )
    physical_source, run_id, manifest_contract = _manifest_calibration_contract(
        manifest,
        expected,
        contract,
        archive["binary_manifest"],
        measurement_blocks=WAVE_PRETEST_MEASUREMENT_BLOCKS,
        bind_stage_metadata=False,
    )
    experiment_id = manifest_contract["experiment_id"]
    round_id = f"{provenance['stage']['stage_id']}:{experiment_id}"
    sessions_by_id: dict[str, Mapping[str, Any]] = {}
    for session in sessions:
        session_id = session.get("session_instance_id")
        if not isinstance(session_id, str) or session_id in sessions_by_id:
            raise ValueError("wave pretest sessions must have unique identities")
        sessions_by_id[session_id] = session
    expected_count = len(expected) * (1 + WAVE_PRETEST_MEASUREMENT_BLOCKS)
    if len(samples) != expected_count or len(sessions) != expected_count:
        raise ValueError("wave pretest evidence count does not match the exact 1+5 set")
    raw_root = Path(raw_dir)
    raw_hashes = {
        "manifest": _file_sha256(raw_root / "manifest.json"),
        "samples": _file_sha256(raw_root / "samples.jsonl"),
        "sessions": _file_sha256(raw_root / "sessions.jsonl"),
        "rank_paths": manifest_contract["environment"].get("requested_rank_paths", []),
    }
    seen: set[tuple[str, str, int, str]] = set()
    seen_sessions: set[str] = set()
    rows: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    for sample in samples:
        if sample.get("experiment_id") != experiment_id or sample.get("run_id") != run_id:
            raise ValueError("wave pretest sample experiment/run identity does not match manifest")
        case_id = sample.get("case_id")
        route_id = sample.get("route_id")
        plan_id = sample.get("plan_id")
        if not isinstance(case_id, str) or not isinstance(route_id, str):
            raise ValueError("wave pretest sample case/route identity is invalid")
        if not isinstance(plan_id, str) or not plan_id.startswith("path_"):
            raise ValueError("wave pretest sample plan identity is invalid")
        candidate_id = plan_id.removeprefix("path_")
        expected_item = expected.get((case_id, route_id, candidate_id))
        if expected_item is None:
            raise ValueError("wave pretest sample is outside the exact selected path set")
        attempt = sample.get("attempt_kind")
        block = sample.get("block_id")
        if isinstance(block, bool) or not isinstance(block, int):
            raise ValueError("wave pretest block identity must be an integer")
        if (attempt, block) == ("warmup", 0):
            pass
        elif attempt == "measurement" and 1 <= block <= WAVE_PRETEST_MEASUREMENT_BLOCKS:
            pass
        else:
            raise ValueError("wave pretest sample is outside the exact 1+5 schedule")
        key = (str(expected_item["cell_id"]), candidate_id, block, str(attempt))
        if key in seen:
            raise ValueError("wave pretest observations contain duplicate attempts")
        seen.add(key)
        if sample.get("status") != "success":
            raise ValueError("wave pretest contains a non-success sample")
        session_id = sample.get("session_instance_id")
        if not isinstance(session_id, str) or session_id not in sessions_by_id:
            raise ValueError("wave pretest sample references a missing session")
        if session_id in seen_sessions:
            raise ValueError("wave pretest requires one fresh session per attempt")
        seen_sessions.add(session_id)
        session = sessions_by_id[session_id]
        for field in ("experiment_id", "run_id", "case_id", "plan_id", "route_id"):
            if session.get(field) != sample.get(field):
                raise ValueError(f"wave pretest sample/session {field} identity mismatch")
        facts, terminal = _joined_backend_facts(
            sample, session, allow_null_overrides=True
        )
        _require_backend_contract(
            sample,
            session,
            facts,
            expected_item["topology"],
            contract,
            manifest_contract["binary_bindings"].get(route_id),
            claim_policy=manifest_contract["collection"]["claim_policy"],
        )
        validation = _mapping(sample.get("validation"), "wave pretest validation")
        if (
            validation.get("policy_reference_applicable") is not True
            or validation.get("policy_reference_passed") is not True
        ):
            raise ValueError("wave pretest sample lacks a passed policy replay")
        identities = _mapping(sample.get("identities"), "wave pretest identities")
        circuit = _mapping(expected_item["circuit"], "wave pretest circuit")
        candidate = _mapping(expected_item["candidate"], "wave pretest candidate")
        topology = _mapping(expected_item["topology"], "wave pretest topology")
        identity_expected = {
            "problem_id": circuit["problem_id"],
            "tensor_network_structure_id": circuit["tensor_network_structure_id"],
            "logical_plan_id": candidate["logical_plan_id"],
            "physical_plan_id": topology["physical_plan_id"],
        }
        for field, value in identity_expected.items():
            if identities.get(field) != value:
                raise ValueError(f"wave pretest sample identity {field} does not match candidate")
        for field, value in (
            ("logical_plan_id", candidate["logical_plan_id"]),
            ("physical_plan_id", topology["physical_plan_id"]),
        ):
            if facts.get(field) != value:
                raise ValueError(f"wave pretest backend {field} does not match candidate")
        binding = manifest_contract["identity_bindings"].get((case_id, plan_id, route_id))
        if binding is None:
            raise ValueError("wave pretest sample identity is outside manifest bindings")
        for field in (
            "problem_id", "tensor_network_structure_id", "logical_plan_id",
            "physical_plan_id", "executable_id", "environment_id",
            "validation_policy_id",
        ):
            if identities.get(field) != binding[field]:
                raise ValueError(f"wave pretest sample identity {field} does not match manifest")
        row = _calibration_row(
            sample=sample,
            session=session,
            facts=facts,
            terminal=terminal,
            expected_item=expected_item,
            physical_source=physical_source,
            candidate_source=str(dataset["source_sha"]),
            candidate_set_sha=candidate_sha,
            calibration_set_sha=None,
            raw_hashes=raw_hashes,
            contract=contract,
            round_id=round_id,
        )
        inclusive = row["session_open_s"] + row["total_wall_s"] + row["session_close_s"]
        if not math.isfinite(inclusive) or inclusive <= 0.0:
            raise ValueError("wave pretest session-inclusive time must be positive")
        if row["session_inclusive_s"] != inclusive:
            raise ValueError("wave pretest session-inclusive time is inconsistent")
        rows.append(row)
        observation = dict(row)
        observation["raw_artifact_sha256"] = {
            "manifest.json": raw_hashes["manifest"],
            "samples.jsonl": raw_hashes["samples"],
            "sessions.jsonl": raw_hashes["sessions"],
        }
        observations.append(observation)
    expected_keys = {
        (str(item["cell_id"]), candidate_id, block, attempt)
        for (_case_id, _topology_id, candidate_id), item in expected.items()
        for block, attempt in (
            [(0, "warmup")]
            + [
                (index, "measurement")
                for index in range(1, WAVE_PRETEST_MEASUREMENT_BLOCKS + 1)
            ]
        )
    }
    if seen != expected_keys:
        raise ValueError("wave pretest observations do not match the exact 1+5 set")
    if seen_sessions != set(sessions_by_id):
        raise ValueError("wave pretest sessions do not match the exact sample set")
    rows.sort(key=lambda row: (row["cell_id"], row["candidate_path_id"], row["block"]))
    observations.sort(key=lambda row: (row["cell_id"], row["candidate_path_id"], row["block"]))
    private_provenance = {
        "configuration_sha256": archive["configuration_sha256"],
        "normalized_configuration_sha256": archive["normalized_configuration_sha256"],
        "configuration_path": archive["configuration_path"],
        "provenance_path": archive["provenance_path"],
        "binary_manifest_path": archive["binary_manifest_path"],
        "checksums": dict(archive["private_hashes"]),
    }
    result: dict[str, Any] = {
        "schema_version": WAVE_PRETEST_SCHEMA,
        "mode": mode,
        "split": split,
        "stage": dict(provenance["stage"]),
        "round_id": round_id,
        "source_sha": dataset["source_sha"],
        "candidate_generation_source_sha": dataset["source_sha"],
        "physical_execution_source_sha": physical_source,
        "candidate_set_sha256": candidate_sha,
        "calibration_set_sha256": None,
        "fitted_profile_sha256": profile_hash,
        "execution_profile": EXECUTION_PROFILE,
        "execution_contract": dict(contract),
        "score_id": contract["cost_model_id"],
        "numeric_policy": contract["numeric_policy"],
        "primary_quantity": WAVE_PRIMARY_QUANTITY,
        "timing_scope": CALIBRATION_TIMING_SCOPE,
        "normalization": WAVE_NORMALIZATION,
        "experiment_id": experiment_id,
        "run_id": run_id,
        "collection": {
            "warmup_blocks": 1,
            "measurement_blocks": WAVE_PRETEST_MEASUREMENT_BLOCKS,
            "blocks": list(range(WAVE_PRETEST_MEASUREMENT_BLOCKS + 1)),
            "attempts_per_candidate_cell": 1 + WAVE_PRETEST_MEASUREMENT_BLOCKS,
        },
        "expected_cell_count": len({item["cell_id"] for item in expected.values()}),
        "expected_candidate_cell_count": len(expected),
        "sample_count": len(samples),
        "session_count": len(sessions),
        "all_successful_physical_sessions": True,
        "all_resource_admission_passed": all(
            row["collection_resource_admission_passed"] is True
            and row["execution_resource_admission_passed"] is True
            and row["startup_resource_admission_passed"] is True
            for row in rows
        ),
        "all_accuracy_qualified": True,
        "all_policy_replays_passed": True,
        "fallback_used": False,
        "timing_used_for_selection": False,
        "raw_artifact_sha256": {
            "manifest.json": raw_hashes["manifest"],
            "samples.jsonl": raw_hashes["samples"],
            "sessions.jsonl": raw_hashes["sessions"],
        },
        "private_provenance": private_provenance,
        "execution_provenance": dict(provenance),
        "binary_bindings": manifest_contract["binary_bindings"],
        "environment": manifest_contract["environment"],
        "selected": [dict(item) for item in provenance["selected"]],
        "selection_roles": [dict(item) for item in provenance["selection_roles"]],
        "raw_samples": [dict(item) for item in samples],
        "raw_sessions": [dict(item) for item in sessions],
        "observations": observations,
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(
        output_dir / "path_runtime_wave_pretest.csv", rows, WAVE_CALIBRATION_COLUMNS
    )
    (output_dir / "path_runtime_wave_pretest.json").write_bytes(_canonical_bytes(result))
    return result


def _evaluate_wave_profile(
    dataset: Mapping[str, Any],
    profile_path: Path,
    output_path: Path,
    *,
    split: str,
) -> dict[str, Any]:
    contract = execution_contract(dataset)
    if contract is None:
        raise ValueError("wave evaluation requires an explicit execution contract")
    _validate_dataset_execution_contract(dataset, contract)
    candidate_sha = _sha256_bytes(_canonical_bytes(dict(dataset)))
    profile, profile_hash, weights, model = _wave_profile_for_evaluation(
        profile_path, dataset, contract, candidate_sha
    )
    selections: list[dict[str, Any]] = []
    for circuit in dataset["circuits"]:
        if circuit.get("split") != split:
            continue
        for topology_id in WAVE_TOPOLOGY_RESOURCES:
            feasible = _wave_evaluation_pool(circuit, topology_id, contract)
            greedy = next(candidate for candidate in feasible if candidate["is_greedy"] is True)
            selected_id, selected_score = min(
                (
                    (
                        candidate["candidate_path_id"],
                        _wave_score(candidate, greedy, topology_id, weights),
                    )
                    for candidate in feasible
                ),
                key=lambda item: (item[1], item[0]),
            )
            selected = next(
                candidate
                for candidate in feasible
                if candidate["candidate_path_id"] == selected_id
            )
            flop_best = min(
                feasible,
                key=lambda candidate: (
                    _mapping(
                        candidate.get("conventional_features"),
                        "wave conventional features",
                    )["flops"],
                    candidate["candidate_path_id"],
                ),
            )
            selected_facts = _wave_facts_from_record(selected, topology_id)
            greedy_facts = _wave_facts_from_record(greedy, topology_id)
            selections.append(
                {
                    "circuit_id": circuit["circuit_id"],
                    "split": split,
                    "topology_id": topology_id,
                    "greedy_path_id": greedy["candidate_path_id"],
                    "minimum_flops_path_id": flop_best["candidate_path_id"],
                    "upmem_selected_path_id": selected_id,
                    "upmem_score": selected_score,
                    "explanation": [
                        row.as_mapping()
                        for row in explain_score(
                            selected_facts["raw"],
                            greedy_facts["raw"],
                            weights,
                            model=model,
                        )
                    ],
                }
            )
    if not selections:
        raise ValueError(f"candidate dataset contains no {split} circuits")
    result = {
        "schema_version": "upmem_path_frozen_selection_v1",
        "execution_profile": EXECUTION_PROFILE,
        "execution_contract": dict(contract),
        "score_id": WAVE_COST_MODEL_ID,
        "primary_quantity": WAVE_PRIMARY_QUANTITY,
        "timing_scope": CALIBRATION_TIMING_SCOPE,
        "numeric_policy": contract["numeric_policy"],
        "normalization": WAVE_NORMALIZATION,
        "source_sha": dataset["source_sha"],
        "source_sha_semantics": "candidate_generation_source_sha",
        "candidate_generation_source_sha": dataset["source_sha"],
        "physical_execution_source_sha": contract["execution_source"],
        "reporting_tool_source_sha": profile.get("reporting_tool_source_sha"),
        "preregistration_sha256": dataset["preregistration_sha256"],
        "candidate_set_sha256": candidate_sha,
        "fitted_profile_sha256": profile_hash,
        "split": split,
        "timing_used_for_selection": False,
        "weights": profile["weights"],
        "feature_model": profile["feature_model"],
        "selections": selections,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(_canonical_bytes(result))
    return result


def evaluate_frozen_profile(
    candidate_path: Path,
    profile_path: Path,
    output_path: Path,
    *,
    split: str,
) -> dict[str, Any]:
    """Select held-out paths without consulting physical timing."""

    if split not in {"validation", "test"}:
        raise ValueError("frozen-profile evaluation is limited to validation or test")
    dataset = _mapping(
        json.loads(candidate_path.read_text(encoding="utf-8")), "candidate dataset"
    )
    contract = execution_contract(dataset)
    if contract is not None:
        if split not in {"validation", "test"}:
            raise ValueError("frozen-profile evaluation is limited to validation or test")
        return _evaluate_wave_profile(
            dataset, profile_path, output_path, split=split
        )
    _reject_unadapted_wave_dataset(dataset, "frozen-profile evaluation")
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    candidate_sha = _sha256_bytes(_canonical_bytes(dataset))
    if profile["candidate_set_sha256"] != candidate_sha:
        raise ValueError("fitted profile does not match candidate dataset")
    if profile["source_sha"] != dataset["source_sha"]:
        raise ValueError("fitted profile source does not match candidate dataset")
    weights = WeightVector.from_values(profile["weights"])
    model = _model_from_profile(profile)
    selections = []
    for circuit in dataset["circuits"]:
        if circuit["split"] != split:
            continue
        candidates = tuple(_candidate_from_record(item) for item in circuit["candidates"])
        for topology_id, _ in _topologies(load_config()):
            feasible = tuple(item for item in candidates if item.feasible_for(topology_id))
            greedy = next(item for item in feasible if item.is_greedy)
            flop_best = min(
                feasible,
                key=lambda item: (item.conventional.flops, item.path_id),
            )
            selected = select_best_candidate(
                feasible,
                topology_id,
                weights,
                model=model,
                greedy_path_id=greedy.path_id,
            )
            selections.append(
                {
                    "circuit_id": circuit["circuit_id"],
                    "split": split,
                    "topology_id": topology_id,
                    "greedy_path_id": greedy.path_id,
                    "minimum_flops_path_id": flop_best.path_id,
                    "upmem_selected_path_id": selected.path_id,
                    "upmem_score": score_features(
                        selected.raw_for(topology_id),
                        greedy.raw_for(topology_id),
                        weights,
                        model=model,
                    ),
                    "explanation": [
                        row.as_mapping()
                        for row in explain_score(
                            selected.raw_for(topology_id),
                            greedy.raw_for(topology_id),
                            weights,
                            model=model,
                        )
                    ],
                }
            )
    if not selections:
        raise ValueError(f"candidate dataset contains no {split} circuits")
    result = {
        "schema_version": "upmem_path_frozen_selection_v1",
        "score_id": COST_MODEL_ID,
        "source_sha": dataset["source_sha"],
        "candidate_set_sha256": candidate_sha,
        "fitted_profile_sha256": _sha256_bytes(_canonical_bytes(profile)),
        "split": split,
        "timing_used_for_selection": False,
        "weights": weights.as_mapping(),
        "feature_model": asdict(model),
        "selections": selections,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(_canonical_bytes(result))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    generate_parser = subparsers.add_parser("generate")
    generate_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    generate_parser.add_argument("--output-dir", type=Path, required=True)
    generate_parser.add_argument("--check", action="store_true")
    generate_parser.add_argument("--circuit-id", action="append", default=[])
    generate_parser.add_argument("--candidate-part-index", type=int)
    generate_parser.add_argument("--candidate-part-count", type=int)
    merge_parser = subparsers.add_parser("merge")
    merge_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    merge_parser.add_argument("--shard-dir", type=Path, action="append", required=True)
    merge_parser.add_argument("--output-dir", type=Path, required=True)
    fit_parser = subparsers.add_parser("fit")
    fit_parser.add_argument("--candidate-paths", type=Path, required=True)
    fit_parser.add_argument("--calibration-set", type=Path, required=True)
    fit_parser.add_argument("--runtime-table", type=Path, required=True)
    fit_parser.add_argument("--output-dir", type=Path, required=True)
    fit_parser.add_argument("--samples", type=int, default=100_000)
    fit_parser.add_argument("--seed", type=int, default=20260903)
    fit_parser.add_argument(
        "--model-form", choices=("auto", "six_term", "grouped"), default="auto"
    )
    fit_parser.add_argument(
        "--fit-split",
        choices=("training", "validation"),
        action="append",
        default=[],
    )
    stages_parser = subparsers.add_parser("fit-wave-stages")
    stages_parser.add_argument("--candidate-paths", type=Path, required=True)
    stages_parser.add_argument(
        "--stage",
        nargs=2,
        action="append",
        metavar=("CALIBRATION", "RUNTIME"),
        required=True,
    )
    stages_parser.add_argument("--output-dir", type=Path, required=True)
    stages_parser.add_argument("--samples", type=int, default=100_000)
    stages_parser.add_argument("--seed", type=int, default=20260903)
    stages_parser.add_argument(
        "--model-form", choices=("six_term", "grouped"), default="six_term"
    )
    extract_parser = subparsers.add_parser(
        "extract-calibration", aliases=("extract",)
    )
    extract_parser.add_argument("--raw-dir", type=Path, required=True)
    extract_parser.add_argument("--candidate-paths", type=Path, required=True)
    extract_parser.add_argument("--calibration-set", type=Path, required=True)
    extract_parser.add_argument("--output-dir", type=Path, required=True)
    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--candidate-paths", type=Path, required=True)
    evaluate_parser.add_argument("--profile", type=Path, required=True)
    evaluate_parser.add_argument("--split", choices=("validation", "test"), required=True)
    evaluate_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "generate":
        partition = None
        if args.candidate_part_index is not None or args.candidate_part_count is not None:
            if args.candidate_part_index is None or args.candidate_part_count is None:
                raise ValueError("candidate partition requires both index and count")
            if not 0 <= args.candidate_part_index < args.candidate_part_count:
                raise ValueError("candidate partition index is outside its count")
            partition = (args.candidate_part_index, args.candidate_part_count)
        timings = generate(
            args.config,
            args.output_dir,
            check=args.check,
            circuit_ids=tuple(args.circuit_id),
            candidate_partition=partition,
        )
        print(json.dumps(timings, sort_keys=True))
    elif args.command == "merge":
        timings = merge_shards(
            args.config, tuple(args.shard_dir), args.output_dir
        )
        print(json.dumps(timings, sort_keys=True))
    elif args.command == "fit":
        result = fit(
            args.candidate_paths, args.calibration_set, args.runtime_table,
            args.output_dir,
            samples=args.samples,
            seed=args.seed,
            model_form=args.model_form,
            fit_splits=tuple(args.fit_split or ("training",)),
        )
        print(json.dumps({"weights": result.weights.as_mapping(), "geometric_mean_speedup": result.geometric_mean_speedup}, sort_keys=True))
    elif args.command == "fit-wave-stages":
        result = fit_wave_stages(
            args.candidate_paths,
            tuple((Path(calibration), Path(runtime)) for calibration, runtime in args.stage),
            args.output_dir,
            samples=args.samples,
            seed=args.seed,
            model_form=args.model_form,
        )
        print(json.dumps({"weights": result.weights.as_mapping(), "geometric_mean_speedup": result.geometric_mean_speedup}, sort_keys=True))
    elif args.command in {"extract-calibration", "extract"}:
        result = extract_calibration(
            args.raw_dir,
            args.candidate_paths,
            args.calibration_set,
            args.output_dir,
        )
        print(json.dumps({"observation_count": result["sample_count"]}, sort_keys=True))
    else:
        result = evaluate_frozen_profile(
            args.candidate_paths,
            args.profile,
            args.output,
            split=args.split,
        )
        print(json.dumps({"selection_count": len(result["selections"])}))


if __name__ == "__main__":
    main()
