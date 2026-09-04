"""Small command-line coordinator for the reset tensor-network benchmark route."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import time
from typing import Any

import numpy as np

from quantum_bench.baselines import (
    run_cotengra,
    run_quest_cpu,
    run_quest_gpu,
    run_quimb,
)
from quantum_bench.circuits import load_circuit
from quantum_bench.cpu import (
    replay_upmem_plan_once,
    run_complex128_reference,
    run_cpu_once,
)
from quantum_bench.evidence import (
    canonical_json,
    environment_id,
    executable_id,
    finalize_artifacts,
    identity_hash,
    new_run_id,
    problem_id,
    tensor_network_structure_id,
    write_manifest,
)
from quantum_bench.experiment import (
    default_validation_policy,
    default_validation_policy_id,
    load_experiment_config,
    run_direct_samples,
    run_session_samples,
)
from quantum_bench.lowering import (
    build_contraction_dag,
    choose_slice_labels,
    contraction_dag_hash,
    lower_tensor_network,
    slice_contraction,
)
from quantum_bench.model import ContractNode, SimulationJob, make_simulation_job
from quantum_bench.planning import plan_cotengra, plan_opt_einsum
from quantum_bench.results import (
    ExecutionFailed,
    ExecutionSample,
    JsonValue,
    UnsupportedExecution,
)
from quantum_bench.upmem.plan import (
    UpmemPlan,
    UpmemResources,
    UpmemTopology,
    collection_resource_admission,
    physical_plan_id,
    plan_upmem,
)
from quantum_bench.upmem.runtime import open_upmem, open_upmem_simulator


_PLAN_SCHEMA = "tn_benchmark_plan_v1"
_SESSION_PROTOCOL_ID = "upmem_real_tile_abi_v4"
_UPMEM_EXECUTORS = frozenset({"upmem_sdk_simulator", "upmem_physical"})
_PLAN_EXECUTORS = frozenset({"numpy_dag", *_UPMEM_EXECUTORS})
_THREAD_ENVIRONMENT_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def _plain(value: object) -> Any:
    """Return JSON values without leaking mutable config implementation types."""

    return json.loads(canonical_json(value))


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _source_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=_repo_root(),
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        raise ValueError(f"cannot determine source commit: {exc}") from exc
    commit = result.stdout.strip()
    if result.returncode != 0 or len(commit) != 40:
        raise ValueError("cannot determine source commit")
    return commit


def _worktree_dirty() -> bool:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=_repo_root(),
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        raise ValueError(f"cannot inspect source worktree: {exc}") from exc
    if result.returncode != 0:
        raise ValueError("cannot inspect source worktree")
    return bool(result.stdout.strip())


def _sha256_file(path: str) -> str | None:
    candidate = Path(path)
    if not candidate.is_file():
        return None
    try:
        return hashlib.sha256(candidate.read_bytes()).hexdigest()
    except OSError:
        return None


def _observed_affinity() -> list[int] | None:
    if not hasattr(os, "sched_getaffinity"):
        return None
    try:
        return sorted(os.sched_getaffinity(0))
    except OSError:
        return None


def _online_logical_cpu_count() -> int | None:
    try:
        value = int(os.sysconf("SC_NPROCESSORS_ONLN"))
    except (AttributeError, OSError, ValueError):
        value = os.cpu_count() or 0
    return value if value > 0 else None


def _environment(
    config: Mapping[str, object],
    machine_preflight: Mapping[str, JsonValue],
) -> tuple[str, Mapping[str, JsonValue]]:
    rank_paths: list[str] = []
    for route in config["routes"].values():
        options = route["options"]
        if route["executor"] == "upmem_physical":
            rank_paths.extend(options["rank_paths"])
    observed_affinity = machine_preflight.get("observed_affinity")
    observed_governors = machine_preflight.get("observed_cpu_governors")
    facts: Mapping[str, JsonValue] = {
        "host": platform.node(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "blas": _numpy_blas_identity(),
        "thread_environment": {
            name: os.environ.get(name) for name in _THREAD_ENVIRONMENT_VARIABLES
        },
        "affinity": observed_affinity,
        "selected_cpu_ids": machine_preflight.get("selected_cpu_ids"),
        "requested_rank_paths": sorted(set(rank_paths)),
        "upmem_sdk_version": _tool_version(("dpu-pkg-config", "--version")),
        "collection_machine_policy": config["collection"]["machine_policy"],
        "initial_background_load_1m": _background_load_1m(),
        "observed_cpu_governors": observed_governors,
        "observed_numa_nodes": _numa_nodes(),
        "machine_preflight": machine_preflight,
    }
    return environment_id(facts), facts


def _numpy_blas_identity() -> Mapping[str, JsonValue]:
    identity: dict[str, JsonValue] = {"name": None, "version": None}
    try:
        configuration = np.show_config(mode="dicts")
    except (AttributeError, TypeError):
        return identity
    if not isinstance(configuration, Mapping):
        return identity
    dependencies = configuration.get("Build Dependencies")
    if not isinstance(dependencies, Mapping):
        return identity
    blas = dependencies.get("blas")
    if not isinstance(blas, Mapping) or blas.get("found") is False:
        return identity
    for field in ("name", "version"):
        value = blas.get(field)
        if isinstance(value, str) and value.strip():
            identity[field] = value.strip()
    return identity


def _background_load_1m() -> float | None:
    try:
        value = float(os.getloadavg()[0])
    except (AttributeError, OSError):
        return None
    return value if np.isfinite(value) and value >= 0.0 else None


def _cpu_governors(cpu_ids: list[int] | None) -> Mapping[str, JsonValue]:
    if cpu_ids is None:
        return {}
    governors: dict[str, JsonValue] = {}
    root = Path("/sys/devices/system/cpu")
    for cpu_id in cpu_ids:
        path = root / f"cpu{cpu_id}" / "cpufreq" / "scaling_governor"
        try:
            value = path.read_text(encoding="utf-8").strip()
        except OSError:
            value = ""
        governors[str(cpu_id)] = value or None
    return governors


def _numa_nodes() -> list[str]:
    root = Path("/sys/devices/system/node")
    try:
        return sorted(
            path.name
            for path in root.iterdir()
            if path.is_dir() and path.name.startswith("node") and path.name[4:].isdigit()
        )
    except OSError:
        return []


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def _physical_rank_paths(config: Mapping[str, object]) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                path
                for route in config["routes"].values()
                if route["executor"] == "upmem_physical"
                for path in route["options"]["rank_paths"]
            }
        )
    )


def _rank_paths_accessible(rank_paths: tuple[str, ...]) -> bool:
    return all(
        Path(path).exists() and os.access(path, os.R_OK | os.W_OK)
        for path in rank_paths
    )


def _machine_preflight(config: Mapping[str, object]) -> Mapping[str, JsonValue]:
    """Observe static machine conditions before a physical-performance schedule."""

    collection = config["collection"]
    machine_policy = collection["machine_policy"]
    physical_performance = collection.get("claim_policy") == "physical_performance_v1"
    checked_at = _utc_now()
    affinity = _observed_affinity()
    governors = _cpu_governors(affinity)
    expected_affinity = machine_policy["affinity"]["expected_cpus"]
    selected_cpu_ids = expected_affinity if expected_affinity is not None else affinity
    numa_nodes = _numa_nodes()
    rank_paths = _physical_rank_paths(config)
    rank_paths_accessible = _rank_paths_accessible(rank_paths)
    sdk_version = _tool_version(("dpu-pkg-config", "--version"))
    initial_load1 = _background_load_1m()
    online_cpu_count = _online_logical_cpu_count()
    initial_load_per_cpu = (
        initial_load1 / online_cpu_count
        if initial_load1 is not None and online_cpu_count is not None
        else None
    )
    exclusivity_attested = os.environ.get("QUANTUM_BENCH_EXCLUSIVITY_ATTESTED") == "1"
    numa_attested = os.environ.get("QUANTUM_BENCH_NUMA_ATTESTED") == "1"
    reasons: list[str] = []

    if physical_performance:
        if not exclusivity_attested:
            reasons.append("machine_exclusivity_not_attested")
        if not numa_attested:
            reasons.append("numa_policy_not_attested")
        if not governors or set(governors.values()) != {"performance"}:
            reasons.append("cpu_governor_not_performance")
        if affinity is None or tuple(affinity) != tuple(expected_affinity):
            reasons.append("process_affinity_mismatch")
        if not rank_paths_accessible:
            reasons.append("rank_paths_inaccessible")
        if sdk_version is None:
            reasons.append("upmem_sdk_unavailable")
        threshold = machine_policy["background_load"]["max_load1_per_online_cpu"]
        if initial_load_per_cpu is None or initial_load_per_cpu > threshold:
            reasons.append("initial_background_load_exceeds_threshold")

    return {
        "machine_preflight_passed": not reasons,
        "machine_preflight_reasons": tuple(reasons),
        "checked_at_utc": checked_at,
        "operator_exclusivity_attested": exclusivity_attested,
        "exclusivity_attestation_mode": machine_policy["machine_exclusivity"]["mode"]
        if isinstance(machine_policy["machine_exclusivity"], Mapping)
        else machine_policy["machine_exclusivity"],
        "exclusivity_attestation_recorded_at_utc": (
            checked_at if exclusivity_attested else None
        ),
        "numa_attested": numa_attested,
        "numa_attestation_mode": machine_policy["numa_policy"]["mode"]
        if isinstance(machine_policy["numa_policy"], Mapping)
        else machine_policy["numa_policy"],
        "numa_attestation_recorded_at_utc": checked_at if numa_attested else None,
        "governor_verified": bool(governors)
        and set(governors.values()) == {"performance"},
        "observed_cpu_governors": governors,
        "selected_cpu_ids": selected_cpu_ids,
        "affinity_verified": (
            tuple(affinity) == tuple(machine_policy["affinity"]["expected_cpus"])
            if isinstance(machine_policy["affinity"], Mapping)
            and machine_policy["affinity"]["expected_cpus"] is not None
            and affinity is not None
            else False
        ),
        "observed_affinity": affinity,
        "observed_numa_nodes": numa_nodes,
        "rank_paths_accessible": rank_paths_accessible,
        "requested_rank_paths": rank_paths,
        "sdk_version": sdk_version,
        "initial_load1": initial_load1,
        "online_logical_cpu_count": online_cpu_count,
        "initial_load1_per_online_cpu": initial_load_per_cpu,
        "background_load_preflight_passed": (
            initial_load_per_cpu
            <= machine_policy["background_load"]["max_load1_per_online_cpu"]
            if physical_performance and initial_load_per_cpu is not None
            else not physical_performance
        ),
    }


def _tool_version(command: tuple[str, ...]) -> str | None:
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=5.0,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    value = result.stdout.strip()
    return value if result.returncode == 0 and value else None


def _job(case: Mapping[str, object]) -> SimulationJob:
    circuit_config = dict(case["circuit"])
    parameters = dict(circuit_config.pop("parameters"))
    circuit_config.update(parameters)
    circuit = load_circuit({"circuit": circuit_config}, _repo_root())
    return make_simulation_job(circuit)


def _plan_dag(
    job: SimulationJob, plan_config: Mapping[str, object]
) -> tuple[object, Mapping[str, np.ndarray], object, Mapping[str, object]]:
    network, inputs = lower_tensor_network(job)
    planner = plan_config["planner"]
    if planner["engine"] == "opt_einsum":
        path, provenance = plan_opt_einsum(network, optimize=planner["mode"])
    elif planner["engine"] == "cotengra":
        path, provenance = plan_cotengra(
            network,
            methods=planner["mode"],
            max_repeats=planner["max_repeats"],
            seed=planner["seed"],
        )
    else:  # Config validation prevents this branch.
        raise ValueError(f"unsupported planner engine: {planner['engine']}")
    dag = build_contraction_dag(network, path)
    slicing = plan_config["slicing"]
    if slicing is not None:
        node = next(
            (
                candidate
                for candidate in dag.nodes
                if isinstance(candidate, ContractNode)
                and candidate.node_id == slicing["node_id"]
            ),
            None,
        )
        if node is None:
            raise ValueError(f"slicing node is not a contraction: {slicing['node_id']}")
        labels = choose_slice_labels(
            node, minimum_slice_count=slicing["minimum_slice_count"]
        )
        dag = slice_contraction(dag, node_id=node.node_id, labels=labels)
    return network, inputs, dag, provenance


def _reference_dag(
    job: SimulationJob,
) -> tuple[object, Mapping[str, np.ndarray], object]:
    """Build a deterministic complex128 reference outside every route timer."""

    network, inputs = lower_tensor_network(job)
    path, _ = plan_opt_einsum(network, optimize="greedy")
    return network, inputs, build_contraction_dag(network, path)


def _statevector(value: np.ndarray) -> np.ndarray:
    """Use the frozen wire-order to QuEST-index conversion without old adapters."""

    return np.asarray(value, dtype=np.complex128).reshape(-1, order="F")


def _error_metrics(
    actual: np.ndarray, expected: np.ndarray
) -> tuple[float, float, float, float]:
    actual_state = _statevector(actual)
    expected_state = _statevector(expected)
    _require_matching_shape(actual_state, expected_state)
    difference = actual_state - expected_state
    max_abs = float(np.max(np.abs(difference), initial=0.0))
    actual_norm = float(np.linalg.norm(actual_state))
    denominator = float(np.linalg.norm(expected_state))
    relative_l2 = (
        float(np.linalg.norm(difference) / denominator)
        if denominator
        else float(np.linalg.norm(difference))
    )
    norm_drift = abs(actual_norm - denominator)
    overlap = np.vdot(expected_state, actual_state)
    phase = np.exp(-1j * np.angle(overlap)) if overlap != 0.0 else 1.0
    phase_aligned_max_abs = float(
        np.max(np.abs(actual_state * phase - expected_state), initial=0.0)
    )
    return max_abs, relative_l2, norm_drift, phase_aligned_max_abs


def _require_matching_shape(actual: np.ndarray, expected: np.ndarray) -> None:
    if tuple(actual.shape) != tuple(expected.shape):
        raise ValueError(
            f"validation output shape mismatch: {actual.shape} != {expected.shape}"
        )


def _validation(
    *,
    sample: ExecutionSample,
    policy_reference: ExecutionSample | None,
    full_reference: np.ndarray,
    numeric_policy: str | None,
    require_raw_lanes: bool,
) -> Mapping[str, JsonValue]:
    validation_policy = default_validation_policy()
    atol = float(validation_policy["float32_atol"])
    rtol = float(validation_policy["float32_rtol"])
    relative_l2_max = float(validation_policy["float32_relative_l2_max"])
    norm_drift_max = float(validation_policy["float32_norm_drift_max"])
    max_abs, relative_l2, norm_drift, phase_aligned_max_abs = _error_metrics(
        sample.output, full_reference
    )
    policy_applicable = policy_reference is not None
    policy_passed: bool | None = None
    if policy_reference is not None:
        _require_matching_shape(sample.output, policy_reference.output)
        if numeric_policy == "complex_int8_shared_scale_v1":
            actual_lanes = sample.numeric_facts.get("raw_lane_records")
            expected_lanes = policy_reference.numeric_facts.get("raw_lane_records")
            raw_lanes_match = (
                actual_lanes is not None
                and expected_lanes is not None
                and canonical_json(actual_lanes) == canonical_json(expected_lanes)
            )
            output_match = np.array_equal(sample.output, policy_reference.output)
            policy_passed = (
                raw_lanes_match and output_match if require_raw_lanes else output_match
            )
        else:
            _, policy_relative_l2, policy_norm_drift, _ = _error_metrics(
                sample.output, policy_reference.output
            )
            policy_passed = bool(
                np.allclose(
                    sample.output,
                    policy_reference.output,
                    atol=atol,
                    rtol=rtol,
                )
                and policy_relative_l2 <= relative_l2_max
                and policy_norm_drift <= norm_drift_max
            )
    full_applicable = numeric_policy != "complex_int8_shared_scale_v1"
    full_passed = (
        bool(
            np.allclose(
                _statevector(sample.output),
                _statevector(full_reference),
                atol=atol,
                rtol=rtol,
            )
            and relative_l2 <= relative_l2_max
            and norm_drift <= norm_drift_max
        )
        if full_applicable
        else None
    )
    accuracy_qualified = full_applicable and full_passed is True
    return {
        "policy_reference_applicable": policy_applicable,
        "policy_reference_passed": policy_passed,
        "full_precision_threshold_applicable": full_applicable,
        "full_precision_passed": full_passed,
        "accuracy_qualified": accuracy_qualified,
        "max_abs_error": max_abs,
        "relative_l2_error": relative_l2,
        "norm_drift": norm_drift,
        "phase_aligned_max_abs_error": phase_aligned_max_abs,
    }


def _topology(route: Mapping[str, object]) -> UpmemTopology:
    options = route["options"]
    return UpmemTopology(
        dpu_count=options["dpu_count"],
        rank_count=options["rank_count"],
        tasklets_per_dpu=options["tasklets_per_dpu"],
    )


def _resources(route: Mapping[str, object]) -> UpmemResources:
    options = route["options"]
    return UpmemResources(
        session_root=options["session_root"],
        host_binary=options["host_binary"],
        dpu_binary=options["dpu_binary"],
        initialization_binary=options["initialization_binary"],
        rank_paths=tuple(options.get("rank_paths", ())),
    )


def _dependency_versions(names: tuple[str, ...]) -> Mapping[str, JsonValue]:
    versions: dict[str, JsonValue] = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _executable_identity(route: Mapping[str, object]) -> str | None:
    executor = route["executor"]
    options = route["options"]
    files: dict[str, str | None] = {}
    if executor == "quest_cpu":
        files["runner"] = _sha256_file(options["runner"])
    elif executor == "quest_gpu":
        files["verification"] = _sha256_file(options["verification_path"])
    elif executor in _UPMEM_EXECUTORS:
        for name in ("host_binary", "dpu_binary", "initialization_binary"):
            files[name] = _sha256_file(options[name])
    if executor in {"quest_cpu", "quest_gpu", *_UPMEM_EXECUTORS} and any(
        digest is None for digest in files.values()
    ):
        return None
    payload: Mapping[str, JsonValue] = {
        "executor": executor,
        "abi_version": 4 if executor in _UPMEM_EXECUTORS else None,
        "static_file_sha256": files,
        "request_transport": (
            "packed_operation_v1" if executor in _UPMEM_EXECUTORS else None
        ),
        "source_commit": _source_commit()
        if executor in {"numpy_dag", "quimb", "cotengra"}
        else None,
        "dependency_versions": _dependency_versions(
            {
                "numpy_dag": ("numpy", "opt_einsum"),
                "quimb": ("numpy", "quimb", "opt_einsum"),
                "cotengra": ("numpy", "quimb", "cotengra", "opt_einsum"),
            }.get(executor, ())
        ),
    }
    return executable_id(payload)


def _identities(
    *,
    job: SimulationJob,
    network: object | None,
    dag: object | None,
    upmem_plan: UpmemPlan | None,
    route: Mapping[str, object],
    environment: str,
) -> Mapping[str, JsonValue]:
    uses_plan = route["executor"] in _PLAN_EXECUTORS
    return {
        "problem_id": problem_id(job),
        "tensor_network_structure_id": (
            tensor_network_structure_id(network)
            if uses_plan and network is not None
            else None
        ),
        "logical_plan_id": contraction_dag_hash(dag)
        if uses_plan and dag is not None
        else None,
        "physical_plan_id": physical_plan_id(upmem_plan)
        if upmem_plan is not None
        else None,
        "executable_id": _executable_identity(route),
        "environment_id": environment,
        "validation_policy_id": default_validation_policy_id(),
    }


def _identity_binding(
    *,
    case_id: str,
    plan_id: str | None,
    route_id: str,
    identities: Mapping[str, JsonValue],
) -> dict[str, JsonValue]:
    """Build the manifest declaration for one selected experiment route."""

    return {
        "case_id": case_id,
        "plan_id": plan_id,
        "route_id": route_id,
        **dict(identities),
    }


def _stage_summary(plan: UpmemPlan) -> list[dict[str, object]]:
    return [
        {
            "stage_id": stage.stage_id,
            "kind": stage.kind,
            "node_ids": list(stage.node_ids),
            "work_unit_count": len(stage.work_units),
        }
        for stage in plan.stages
    ]


def _plan_document(config: Mapping[str, object]) -> Mapping[str, object]:
    entries: list[dict[str, object]] = []
    for matrix_item in config["matrix"]:
        plan_id = matrix_item["plan_id"]
        if plan_id is None:
            job = _job(config["cases"][matrix_item["case_id"]])
            for route_id in matrix_item["route_ids"]:
                entries.append(
                    {
                        "case_id": matrix_item["case_id"],
                        "plan_id": None,
                        "route_id": route_id,
                        "problem_id": problem_id(job),
                        "tensor_network_structure_id": None,
                        "logical_plan_id": None,
                        "planner_provenance": None,
                        "dag": None,
                        "upmem": None,
                    }
                )
            continue
        job = _job(config["cases"][matrix_item["case_id"]])
        network, _, dag, provenance = _plan_dag(job, config["plans"][plan_id])
        stable_provenance = {
            key: value for key, value in provenance.items() if key != "planning_time_s"
        }
        base = {
            "case_id": matrix_item["case_id"],
            "plan_id": plan_id,
            "problem_id": problem_id(job),
            "tensor_network_structure_id": tensor_network_structure_id(network),
            "logical_plan_id": contraction_dag_hash(dag),
            "planner_provenance": stable_provenance,
            "dag": {
                "node_count": len(dag.nodes),
                "output_tensor_id": dag.output.tensor_id,
            },
        }
        for route_id in matrix_item["route_ids"]:
            route = config["routes"][route_id]
            if route["executor"] not in _UPMEM_EXECUTORS:
                entries.append({**base, "route_id": route_id, "upmem": None})
                continue
            try:
                compiled = plan_upmem(
                    dag,
                    numeric_policy=route["numeric_policy"],
                    topology=_topology(route),
                )
                entries.append(
                    {
                        **base,
                        "route_id": route_id,
                        "upmem": {
                            "physical_plan_id": physical_plan_id(compiled),
                            "kernel_policy": compiled.kernel_policy,
                            "topology": {
                                "dpu_count": compiled.topology.dpu_count,
                                "rank_count": compiled.topology.rank_count,
                                "tasklets_per_dpu": compiled.topology.tasklets_per_dpu,
                            },
                            "stages": _stage_summary(compiled),
                        },
                    }
                )
            except UnsupportedExecution as exc:
                entries.append(
                    {
                        **base,
                        "route_id": route_id,
                        "upmem": None,
                        "unsupported": {
                            "stage": exc.stage,
                            "reason": exc.reason,
                            "capability": exc.capability,
                        },
                    }
                )
    return {
        "schema_version": _PLAN_SCHEMA,
        "experiment_id": config["experiment_id"],
        "validation_policy_id": default_validation_policy_id(),
        "entries": sorted(
            entries,
            key=lambda entry: (
                entry["case_id"],
                "" if entry["plan_id"] is None else entry["plan_id"],
                entry["route_id"],
            ),
        ),
    }


def plan_command(config_path: str, output: str) -> Mapping[str, object]:
    config = load_experiment_config(config_path)
    document = _plan_document(config)
    target = _prepare_output(output)
    (target / "plan.json").write_text(canonical_json(document) + "\n", encoding="utf-8")
    return {
        "status": "planned",
        "artifact": str(target / "plan.json"),
        "entry_count": len(document["entries"]),
    }


def _prepare_output(output: str) -> Path:
    target = Path(output)
    if target.exists() and not target.is_dir():
        raise ValueError("output must be a directory path")
    if target.exists() and any(target.iterdir()):
        raise ValueError("output directory must be absent or empty")
    target.mkdir(parents=True, exist_ok=True)
    return target


def _expected_counts(config: Mapping[str, object]) -> Mapping[str, int]:
    selected_routes = sum(len(item["route_ids"]) for item in config["matrix"])
    collection = config["collection"]
    attempt_count = collection["warmup_blocks"] + collection["measurement_blocks"]
    return {
        "warmup": selected_routes * collection["warmup_blocks"],
        "measurement": selected_routes * collection["measurement_blocks"],
        "sessions": sum(
            route_id in config["routes"]
            and config["routes"][route_id]["executor"] in _UPMEM_EXECUTORS
            for item in config["matrix"]
            for route_id in item["route_ids"]
        )
        * attempt_count,
    }


def _collection_configuration_id(
    matrix_item: Mapping[str, object], route_id: str
) -> str:
    return identity_hash(
        "quantum_bench.collection_configuration_id.v1",
        {
            "case_id": matrix_item["case_id"],
            "plan_id": matrix_item["plan_id"],
            "route_id": route_id,
        },
    )


def _collection_order_key(
    *,
    base_seed: int,
    experiment_id: str,
    block_id: int,
    configuration_id: str,
) -> bytes:
    return hashlib.sha256(
        b"quantum_bench.collection_order.v1\0"
        + str(base_seed).encode("ascii")
        + b"\0"
        + experiment_id.encode("ascii")
        + b"\0"
        + str(block_id).encode("ascii")
        + b"\0"
        + configuration_id.encode("ascii")
    ).digest()


def _scheduled_attempts(
    config: Mapping[str, object],
    selected: list[tuple[Mapping[str, object], str, Mapping[str, object]]],
) -> tuple[
    tuple[Mapping[str, object], str, Mapping[str, object], tuple[str, int, int, int]],
    ...,
]:
    """Return deterministic warmup/measurement blocks for every selected route."""

    collection = config["collection"]
    base_seed = collection["base_seed"]
    if not isinstance(base_seed, int):  # guarded by config validation
        raise TypeError("collection.base_seed must be an integer")
    blocks = (
        ("warmup", collection["warmup_blocks"], 0),
        (
            "measurement",
            collection["measurement_blocks"],
            collection["warmup_blocks"],
        ),
    )
    schedule: list[
        tuple[Mapping[str, object], str, Mapping[str, object], tuple[str, int, int, int]]
    ] = []
    for attempt_kind, block_count, block_offset in blocks:
        if not isinstance(block_count, int) or not isinstance(block_offset, int):
            raise TypeError("collection block counts must be integers")
        for sample_index in range(block_count):
            block_id = block_offset + sample_index
            ordered = sorted(
                selected,
                key=lambda entry: (
                    _collection_order_key(
                        base_seed=base_seed,
                        experiment_id=config["experiment_id"],
                        block_id=block_id,
                        configuration_id=_collection_configuration_id(
                            entry[0], entry[1]
                        ),
                    ),
                    _collection_configuration_id(entry[0], entry[1]),
                ),
            )
            schedule.extend(
                (item, route_id, route, (attempt_kind, sample_index, block_id, order))
                for order, (item, route_id, route) in enumerate(ordered)
            )
    return tuple(schedule)


def _require_collection_resource_admission(
    plan: UpmemPlan, *, physical_performance_campaign: bool
) -> None:
    """Reject physical-performance scaling without dominant-work resources."""

    if not physical_performance_campaign:
        return
    admission = collection_resource_admission(plan)
    if not admission["tasklet_row_sufficiency_passed"]:
        raise UnsupportedExecution(
            "collection_admission",
            "tasklet scaling requires each dominant-work unit to provide one output row per tasklet",
            "upmem_tasklet_work_unit_rows",
        )
    if plan.topology.dpu_count > 1 and not admission[
        "collection_resource_admission_passed"
    ]:
        raise UnsupportedExecution(
            "collection_admission",
            "DPU scaling requires a fully populated dominant work-unit wave",
            "upmem_dpu_wave_utilization",
        )


def _require_physical_opt_in(*, allow_physical: bool) -> None:
    if not allow_physical:
        raise ValueError("physical UPMEM requires --allow-physical")
    if os.environ.get("UPMEM_ALLOW_PHYSICAL_HARDWARE") != "1":
        raise ValueError("physical UPMEM requires UPMEM_ALLOW_PHYSICAL_HARDWARE=1")


def _collection_cooldown(config: Mapping[str, object]) -> None:
    collection = config["collection"]
    cooldown_s = collection.get("block_cooldown_s", collection.get("cooldown_s"))
    if not isinstance(cooldown_s, float):  # guarded by config validation
        raise TypeError("collection cooldown must be a float")
    if cooldown_s:
        time.sleep(cooldown_s)


def _complete_collection_block(
    config: Mapping[str, object],
    scheduled: tuple[
        tuple[Mapping[str, object], str, Mapping[str, object], tuple[str, int, int, int]],
        ...,
    ],
    schedule_index: int,
) -> None:
    """Apply cooldown once after a complete non-final collection block."""

    if schedule_index + 1 >= len(scheduled):
        return
    current_block = scheduled[schedule_index][3][2]
    next_block = scheduled[schedule_index + 1][3][2]
    if current_block != next_block:
        _collection_cooldown(config)


def _raise_physical_collection_failure(
    rows: tuple[Mapping[str, JsonValue], ...],
    session: Mapping[str, JsonValue],
) -> None:
    """Propagate a persisted physical failure after session cleanup."""

    failed_sample = next(
        (row for row in rows if row["status"] != "success"), None
    )
    if failed_sample is not None:
        failure = failed_sample["failure"]
        if not isinstance(failure, Mapping):
            raise ValueError("physical UPMEM sample failure is missing details")
        stage = failure.get("stage")
        reason = failure.get("reason")
        if not isinstance(stage, str) or not isinstance(reason, str):
            raise ValueError("physical UPMEM sample failure has invalid details")
        if failed_sample["status"] == "unsupported":
            capability = failure.get("capability")
            if not isinstance(capability, str):
                raise ValueError(
                    "physical UPMEM unsupported sample lacks a capability"
                )
            raise UnsupportedExecution(stage, reason, capability)
        backend_facts = failed_sample.get("backend_facts", {})
        raise ExecutionFailed(
            stage,
            reason,
            backend_facts if isinstance(backend_facts, Mapping) else {},
        )

    if session["status"] != "success":
        failure = session["failure"]
        if not isinstance(failure, Mapping):
            raise ValueError("physical UPMEM session failure is missing details")
        stage = failure.get("stage")
        reason = failure.get("reason")
        if not isinstance(stage, str) or not isinstance(reason, str):
            raise ValueError("physical UPMEM session failure has invalid details")
        terminal_backend_facts = session.get("terminal_backend_facts", {})
        raise ExecutionFailed(
            stage,
            reason,
            terminal_backend_facts
            if isinstance(terminal_backend_facts, Mapping)
            else {},
        )


def _direct_runner(
    route: Mapping[str, object],
    job: SimulationJob,
    dag: object | None,
    inputs: Mapping[str, np.ndarray] | None,
    timeout_s: float,
) -> Callable[[], ExecutionSample]:
    executor = route["executor"]
    options = route["options"]
    if executor == "numpy_dag":
        assert dag is not None and inputs is not None
        return lambda: run_cpu_once(dag, inputs, route["numeric_policy"])
    if executor == "quimb":
        return lambda: run_quimb(job, optimize=options["optimize"])
    if executor == "cotengra":
        return lambda: run_cotengra(
            job,
            methods=options["methods"],
            max_repeats=options["max_repeats"],
        )
    if executor == "quest_cpu":
        return lambda: run_quest_cpu(
            job, runner=Path(options["runner"]), timeout_s=timeout_s
        )
    if executor == "quest_gpu":
        return lambda: run_quest_gpu(
            job,
            verification_path=Path(options["verification_path"]),
            timeout_s=timeout_s,
        )
    raise ValueError(f"executor {executor!r} is not direct")


def _run_config(
    config_path: str,
    output: str,
    *,
    allow_physical: bool,
    qualification_only: bool,
) -> Mapping[str, object]:
    config = load_experiment_config(config_path)
    selected = [
        (item, route_id, config["routes"][route_id])
        for item in config["matrix"]
        for route_id in item["route_ids"]
    ]
    scheduled = _scheduled_attempts(config, selected)
    if qualification_only:
        if any(route["executor"] != "upmem_physical" for _, _, route in selected):
            raise ValueError("qualify accepts only upmem_physical routes")
        _require_physical_opt_in(allow_physical=allow_physical)
        if _worktree_dirty():
            raise ValueError("physical qualification requires a clean Git worktree")
    elif any(route["executor"] == "upmem_physical" for _, _, route in selected):
        _require_physical_opt_in(allow_physical=allow_physical)

    target = _prepare_output(output)
    machine_preflight = _machine_preflight(config)
    env_id, env_facts = _environment(config, machine_preflight)
    run_id = new_run_id()
    manifest = {
        "schema_version": "evidence_manifest_v2",
        "run_id": run_id,
        "experiment_id": config["experiment_id"],
        "collection_policy_id": config["collection_policy_id"],
        "environment_id": env_id,
        "validation_policy_id": default_validation_policy_id(),
        "created_at_utc": datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
        "source_commit": _source_commit(),
        "source_worktree_dirty": _worktree_dirty(),
        "configuration": {
            "experiment": _plain(config),
            "environment": env_facts,
            "validation_policy": _plain(default_validation_policy()),
            "identity_bindings": [],
        },
        "expected_counts": _expected_counts(config),
        "files": {
            "manifest": "manifest.json",
            "samples": "samples.jsonl",
            "sessions": "sessions.jsonl",
        },
        "status": "running",
    }
    write_manifest(target / "manifest.json", manifest)
    # Direct-only campaigns still need the complete three-file artifact set.
    (target / "samples.jsonl").touch()
    (target / "sessions.jsonl").touch()

    if machine_preflight["machine_preflight_passed"] is False:
        finalize_artifacts(target, status="failed")
        return {
            "status": "failed",
            "artifact": str(target / "manifest.json"),
            "run_id": run_id,
        }

    jobs: dict[str, SimulationJob] = {}
    named_plans: dict[
        tuple[str, str], tuple[object, Mapping[str, np.ndarray], object]
    ] = {}
    references: dict[str, np.ndarray] = {}
    identity_bindings: dict[
        tuple[str, str | None, str], dict[str, JsonValue]
    ] = {}

    def bind_route(
        *,
        case_id: str,
        plan_id: str | None,
        route_id: str,
        identities: Mapping[str, JsonValue],
    ) -> None:
        key = (case_id, plan_id, route_id)
        binding = _identity_binding(
            case_id=case_id,
            plan_id=plan_id,
            route_id=route_id,
            identities=identities,
        )
        previous = identity_bindings.setdefault(key, binding)
        if previous != binding:
            raise ValueError(f"selected route has conflicting identities: {key}")

    def persist_identity_bindings() -> None:
        manifest["configuration"]["identity_bindings"] = [
            identity_bindings[key]
            for key in sorted(
                identity_bindings,
                key=lambda key: (
                    key[0],
                    "" if key[1] is None else key[1],
                    key[2],
                ),
            )
        ]
        write_manifest(target / "manifest.json", manifest)

    try:
        for schedule_index, (matrix_item, route_id, route, attempt) in enumerate(scheduled):
            case_id = matrix_item["case_id"]
            job = jobs.setdefault(case_id, _job(config["cases"][case_id]))
            plan_id = matrix_item["plan_id"]
            network = inputs = dag = upmem_plan = None
            if plan_id is not None:
                key = (case_id, plan_id)
                if key not in named_plans:
                    planned_network, planned_inputs, planned_dag, _ = _plan_dag(
                        job, config["plans"][plan_id]
                    )
                    named_plans[key] = (planned_network, planned_inputs, planned_dag)
                network, inputs, dag = named_plans[key]
            if case_id not in references:
                reference_network, reference_inputs, reference_dag = _reference_dag(job)
                del reference_network
                references[case_id] = _statevector(
                    run_complex128_reference(reference_dag, reference_inputs)
                )
            full_reference = references[case_id]
            if route["executor"] in _PLAN_EXECUTORS:
                assert network is not None and dag is not None and inputs is not None
            if route["executor"] in _UPMEM_EXECUTORS:
                try:
                    upmem_plan = plan_upmem(
                        dag,
                        numeric_policy=route["numeric_policy"],
                        topology=_topology(route),
                    )
                    _require_collection_resource_admission(
                        upmem_plan,
                        physical_performance_campaign=(
                            route["executor"] == "upmem_physical"
                            and config["collection"]["claim_policy"]
                            == "physical_performance_v1"
                        ),
                    )
                except UnsupportedExecution as exc:
                    identities = _identities(
                        job=job,
                        network=network,
                        dag=dag,
                        upmem_plan=None,
                        route=route,
                        environment=env_id,
                    )
                    bind_route(
                        case_id=case_id,
                        plan_id=plan_id,
                        route_id=route_id,
                        identities=identities,
                    )

                    def unsupported(
                        failure: UnsupportedExecution = exc,
                    ) -> ExecutionSample:
                        raise UnsupportedExecution(
                            stage=failure.stage,
                            reason=failure.reason,
                            capability=failure.capability,
                        )

                    run_direct_samples(
                        run_id=run_id,
                        experiment_id=config["experiment_id"],
                        case_id=case_id,
                        route_id=route_id,
                        plan_id=plan_id,
                        identities=identities,
                        warmups=0,
                        repetitions=0,
                        run_once=unsupported,
                        samples_path=target / "samples.jsonl",
                        attempts=(attempt,),
                    )
                    _complete_collection_block(config, scheduled, schedule_index)
                    continue
            identities = _identities(
                job=job,
                network=network,
                dag=dag,
                upmem_plan=upmem_plan,
                route=route,
                environment=env_id,
            )
            bind_route(
                case_id=case_id,
                plan_id=plan_id,
                route_id=route_id,
                identities=identities,
            )

            if route["executor"] in _UPMEM_EXECUTORS:
                assert dag is not None and inputs is not None and upmem_plan is not None
                policy_reference = replay_upmem_plan_once(dag, upmem_plan, inputs)

                def validate(
                    sample: ExecutionSample,
                    reference: ExecutionSample = policy_reference,
                    full: np.ndarray = full_reference,
                    policy: str = route["numeric_policy"],
                ) -> Mapping[str, JsonValue]:
                    return _validation(
                        sample=sample,
                        policy_reference=reference,
                        full_reference=full,
                        numeric_policy=policy,
                        require_raw_lanes=True,
                    )

                resources = _resources(route)
                opener = (
                    (
                        lambda: open_upmem_simulator(
                            dag,
                            upmem_plan,
                            resources,
                            timeout_s=config["defaults"]["timeout_s"],
                        )
                    )
                    if route["executor"] == "upmem_sdk_simulator"
                    else (
                        lambda: open_upmem(
                            dag,
                            upmem_plan,
                            resources,
                            timeout_s=config["defaults"]["timeout_s"],
                        )
                    )
                )
                rows, session = run_session_samples(
                    run_id=run_id,
                    experiment_id=config["experiment_id"],
                    case_id=case_id,
                    route_id=route_id,
                    plan_id=plan_id,
                    identities=identities,
                    warmups=0,
                    repetitions=0,
                    session_protocol_id=_SESSION_PROTOCOL_ID,
                    open_session=opener,
                    inputs=inputs,
                    samples_path=target / "samples.jsonl",
                    sessions_path=target / "sessions.jsonl",
                    validate=validate,
                    attempts=(attempt,),
                )
                if route["executor"] == "upmem_physical":
                    _raise_physical_collection_failure(rows, session)
                _complete_collection_block(config, scheduled, schedule_index)
                continue

            if route["executor"] == "numpy_dag":
                assert dag is not None and inputs is not None
                policy_reference = run_cpu_once(dag, inputs, route["numeric_policy"])

                def validate(
                    sample: ExecutionSample,
                    reference: ExecutionSample = policy_reference,
                    full: np.ndarray = full_reference,
                    policy: str = route["numeric_policy"],
                ) -> Mapping[str, JsonValue]:
                    return _validation(
                        sample=sample,
                        policy_reference=reference,
                        full_reference=full,
                        numeric_policy=policy,
                        require_raw_lanes=False,
                    )

            else:

                def validate(
                    sample: ExecutionSample,
                    full: np.ndarray = full_reference,
                ) -> Mapping[str, JsonValue]:
                    return _validation(
                        sample=sample,
                        policy_reference=None,
                        full_reference=full,
                        numeric_policy=None,
                        require_raw_lanes=False,
                    )

            run_direct_samples(
                run_id=run_id,
                experiment_id=config["experiment_id"],
                case_id=case_id,
                route_id=route_id,
                plan_id=plan_id,
                identities=identities,
                warmups=0,
                repetitions=0,
                run_once=_direct_runner(
                    route, job, dag, inputs, config["defaults"]["timeout_s"]
                ),
                samples_path=target / "samples.jsonl",
                validate=validate,
                attempts=(attempt,),
            )
            _complete_collection_block(config, scheduled, schedule_index)
    except Exception:
        try:
            persist_identity_bindings()
            finalize_artifacts(target, status="failed")
        except Exception:
            pass
        raise

    persist_identity_bindings()

    try:
        finalize_artifacts(target, status="completed")
        status = "completed"
    except Exception:
        finalize_artifacts(target, status="failed")
        status = "failed"
    return {
        "status": status,
        "artifact": str(target / "manifest.json"),
        "run_id": run_id,
    }


def run_command(
    config_path: str, output: str, *, allow_physical: bool
) -> Mapping[str, object]:
    return _run_config(
        config_path, output, allow_physical=allow_physical, qualification_only=False
    )


def qualify_command(
    config_path: str, output: str, *, allow_physical: bool
) -> Mapping[str, object]:
    return _run_config(
        config_path, output, allow_physical=allow_physical, qualification_only=True
    )


def _report_command(input_dir: str, output: str) -> Mapping[str, object]:
    try:
        from quantum_bench.report import report_artifacts
    except ImportError as exc:
        raise ValueError(
            "report command is unavailable: quantum_bench.report is not implemented"
        ) from exc
    return report_artifacts(input_dir, output)


def _verify_command(input_dir: str) -> Mapping[str, object]:
    try:
        from quantum_bench.report import verify_artifacts
    except ImportError as exc:
        raise ValueError(
            "verify command is unavailable: quantum_bench.report is not implemented"
        ) from exc
    return verify_artifacts(input_dir)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="quantum-bench")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "run", "qualify"):
        command = commands.add_parser(name)
        command.add_argument("--config", required=True)
        command.add_argument("--output", required=True)
        if name in {"run", "qualify"}:
            command.add_argument("--allow-physical", action="store_true")
    for name in ("report", "verify"):
        command = commands.add_parser(name)
        command.add_argument("--input", required=True)
        if name == "report":
            command.add_argument("--output", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "plan":
            result = plan_command(args.config, args.output)
        elif args.command == "run":
            result = run_command(
                args.config, args.output, allow_physical=args.allow_physical
            )
        elif args.command == "qualify":
            result = qualify_command(
                args.config, args.output, allow_physical=args.allow_physical
            )
        elif args.command == "report":
            result = _report_command(args.input, args.output)
        else:
            result = _verify_command(args.input)
    except (OSError, ValueError, UnsupportedExecution) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, sort_keys=True))
        return 2
    except Exception as exc:
        print(
            json.dumps(
                {"status": "failed", "error": f"{type(exc).__name__}: {exc}"},
                sort_keys=True,
            )
        )
        return 2
    print(json.dumps(_plain(result), sort_keys=True))
    return 0 if result.get("status") in {"planned", "completed"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
