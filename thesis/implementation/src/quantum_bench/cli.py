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
from quantum_bench.results import ExecutionSample, JsonValue, UnsupportedExecution
from quantum_bench.upmem.plan import (
    UpmemPlan,
    UpmemResources,
    UpmemTopology,
    physical_plan_id,
    plan_upmem,
)
from quantum_bench.upmem.runtime import open_upmem, open_upmem_simulator


_PLAN_SCHEMA = "tn_benchmark_plan_v1"
_SESSION_PROTOCOL_ID = "upmem_real_tile_abi_v4"
_UPMEM_EXECUTORS = frozenset({"upmem_sdk_simulator", "upmem_physical"})
_PLAN_EXECUTORS = frozenset({"numpy_dag", *_UPMEM_EXECUTORS})


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


def _environment(config: Mapping[str, object]) -> tuple[str, Mapping[str, JsonValue]]:
    rank_paths: list[str] = []
    for route in config["routes"].values():
        options = route["options"]
        if route["executor"] == "upmem_physical":
            rank_paths.extend(options["rank_paths"])
    affinity: list[int] | None = None
    if hasattr(os, "sched_getaffinity"):
        try:
            affinity = sorted(os.sched_getaffinity(0))
        except OSError:
            affinity = None
    facts: Mapping[str, JsonValue] = {
        "host": platform.node(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "affinity": affinity,
        "requested_rank_paths": sorted(set(rank_paths)),
        "upmem_sdk_version": _tool_version(("dpu-pkg-config", "--version")),
    }
    return environment_id(facts), facts


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


def _error_metrics(actual: np.ndarray, expected: np.ndarray) -> tuple[float, float]:
    actual_state = _statevector(actual)
    expected_state = _statevector(expected)
    _require_matching_shape(actual_state, expected_state)
    difference = actual_state - expected_state
    max_abs = float(np.max(np.abs(difference), initial=0.0))
    denominator = float(np.linalg.norm(_statevector(expected)))
    relative_l2 = (
        float(np.linalg.norm(difference) / denominator)
        if denominator
        else float(np.linalg.norm(difference))
    )
    return max_abs, relative_l2


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
    max_abs, relative_l2 = _error_metrics(sample.output, full_reference)
    policy_applicable = policy_reference is not None
    policy_passed: bool | None = None
    if policy_reference is not None:
        _require_matching_shape(sample.output, policy_reference.output)
        if numeric_policy == "split_complex_int8_shared_scale_v1":
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
            policy_passed = bool(
                np.allclose(
                    sample.output, policy_reference.output, atol=1.0e-5, rtol=1.0e-5
                )
            )
    full_applicable = numeric_policy != "split_complex_int8_shared_scale_v1"
    full_passed = (
        bool(
            np.allclose(
                _statevector(sample.output),
                _statevector(full_reference),
                atol=1.0e-5,
                rtol=1.0e-5,
            )
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
    return {
        "warmup": selected_routes * config["defaults"]["warmups"],
        "measurement": selected_routes * config["defaults"]["repetitions"],
        "sessions": sum(
            route_id in config["routes"]
            and config["routes"][route_id]["executor"] in _UPMEM_EXECUTORS
            for item in config["matrix"]
            for route_id in item["route_ids"]
        ),
    }


def _require_physical_opt_in(*, allow_physical: bool) -> None:
    if not allow_physical:
        raise ValueError("physical UPMEM requires --allow-physical")
    if os.environ.get("UPMEM_ALLOW_PHYSICAL_HARDWARE") != "1":
        raise ValueError("physical UPMEM requires UPMEM_ALLOW_PHYSICAL_HARDWARE=1")


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
    if qualification_only:
        if any(route["executor"] != "upmem_physical" for _, _, route in selected):
            raise ValueError("qualify accepts only upmem_physical routes")
        _require_physical_opt_in(allow_physical=allow_physical)
        if _worktree_dirty():
            raise ValueError("physical qualification requires a clean Git worktree")
    elif any(route["executor"] == "upmem_physical" for _, _, route in selected):
        _require_physical_opt_in(allow_physical=allow_physical)

    target = _prepare_output(output)
    env_id, env_facts = _environment(config)
    run_id = new_run_id()
    manifest = {
        "schema_version": "evidence_manifest_v1",
        "run_id": run_id,
        "experiment_id": config["experiment_id"],
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
        for matrix_item, route_id, route in selected:
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
                        warmups=config["defaults"]["warmups"],
                        repetitions=config["defaults"]["repetitions"],
                        run_once=unsupported,
                        samples_path=target / "samples.jsonl",
                    )
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
                run_session_samples(
                    run_id=run_id,
                    experiment_id=config["experiment_id"],
                    case_id=case_id,
                    route_id=route_id,
                    plan_id=plan_id,
                    identities=identities,
                    warmups=config["defaults"]["warmups"],
                    repetitions=config["defaults"]["repetitions"],
                    session_protocol_id=_SESSION_PROTOCOL_ID,
                    open_session=opener,
                    inputs=inputs,
                    samples_path=target / "samples.jsonl",
                    sessions_path=target / "sessions.jsonl",
                    validate=validate,
                )
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
                warmups=config["defaults"]["warmups"],
                repetitions=config["defaults"]["repetitions"],
                run_once=_direct_runner(
                    route, job, dag, inputs, config["defaults"]["timeout_s"]
                ),
                samples_path=target / "samples.jsonl",
                validate=validate,
            )
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
