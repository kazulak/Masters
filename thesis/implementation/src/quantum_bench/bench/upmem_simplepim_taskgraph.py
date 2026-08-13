"""Small contract-driven engine for bounded physical SimplePIM TaskGraph studies.

M4.5 remains the default public route. New, immutable study contracts can
reuse the same suite, plan, resident-package, native adapter, and normalized
record flow without cloning this physical route.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import importlib
import json
import math
from numbers import Integral
import os
from pathlib import Path
import shutil
import time
from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np

from quantum_bench.bench.config import load_suite
from quantum_bench.bench.reporting import write_normalized_records, write_run_manifest
from quantum_bench.bench.run_dirs import EVIDENCE_ARTIFACT_KIND, create_run_dir, sanitize
from quantum_bench.circuits import load_circuit
from quantum_bench.core.jsonio import write_json, write_jsonl
from quantum_bench.core.records import JsonDict, TaskGraph, TensorValue, to_jsonable
from quantum_bench.environment import capture_environment
from quantum_bench.targets.upmem.execution_plan_v1 import (
    ExecutionPlan,
    PLACEMENT_FRONTIER,
    PLACEMENT_SINGLE,
    compile_plan,
    serialize_plan_json,
    serialize_schedule,
    validate_schedule,
    validate_plan,
)
from quantum_bench.targets.upmem.hardware_taskgraph_resident import (
    ResidentGraphPackage,
    build_resident_graph_package,
    validate_resident_graph_package_file,
)
from quantum_bench.targets.upmem.hardware_session import hardware_environment_metadata
from quantum_bench.tn import (
    build_tensor_network,
    execute_task_sequence_np_einsum,
    plan_task_graph_with_config,
    with_execution_identity,
)
from quantum_bench.tn.network import TensorNetworkValue


SUITE_ID = "upmem_hardware_simplepim_taskgraph"
SCHEMA_VERSION = "upmem_hardware_simplepim_taskgraph_v1"
PROFILE_VERSION = "hardware_simplepim_taskgraph_m4_5_v1"
ROUTE_ID = "upmem_tn_hardware_simplepim_taskgraph"
BACKEND_ID = "upmem_hardware_simplepim_taskgraph"
CLAIM_BOUNDARY = "physical_functionality_and_bringup_timing_only"
NATIVE_TARGET_MODULE = "quantum_bench.targets.upmem.simplepim_taskgraph_executor"
ADAPTER_SESSION_SCHEMA = "upmem_execution_plan_adapter_session_v1"
NATIVE_BACKEND_ID = "upmem_sdk_hardware_execution_plan"
PLACEMENTS = (PLACEMENT_SINGLE, PLACEMENT_FRONTIER)
TASKLETS_PER_DPU = 1
WARMUPS = 1
REPEATS = 3
DEFAULT_TOLERANCE = 1.0e-6
DEVICE_LAUNCH_MODE = "asynchronous_per_dpu"
SYNCHRONIZATION_POLICY = "synchronous_wave_barriers"
FAILURE_STAGES = frozenset("hardware_opt_in_missing hardware_profile_violation sdk_discovery_failed native_build_failed package_preparation_failed execution_plan_compile_failed hardware_allocation_failed binary_load_failed argument_transfer_failed operand_transfer_failed kernel_launch_failed kernel_timeout result_transfer_failed output_manifest_failed output_validation_failed hardware_release_failed".split())


class Block2NativeTarget(Protocol):
    """Stable Python adapter seam owned by Block 2."""

    def build(self, build_dir: Path, *, prepare_only: bool = True) -> Mapping[str, Any]: ...

    def validate(self, request_path: Path, *, timeout_s: float) -> Mapping[str, Any]: ...

    def execute(self, request_path: Path, *, timeout_s: float) -> Mapping[str, Any]: ...


class NativeExecutionFailure(RuntimeError):
    def __init__(self, stage: str, message: str) -> None:
        super().__init__(message)
        self.stage = stage if stage in FAILURE_STAGES else "kernel_launch_failed"


@dataclass(frozen=True)
class TransferEdgeExpectation:
    """One immutable host-mediated edge required by a route study."""

    producer_task_id: str
    consumer_task_id: str
    producer_dpu_id: int
    consumer_dpu_id: int
    slot_id: int
    element_count: int
    transfer_bytes: int


@dataclass(frozen=True)
class RouteStudyContract:
    """Immutable limits and identities for one thin physical route study."""

    suite_id: str
    schema_version: str
    profile_version: str
    route_id: str
    backend_id: str
    claim_boundary: str
    route_label: str
    execution_scope: str
    benchmark_role: str
    placements: tuple[str, ...]
    case_id: str
    qasm_path: str
    warmups: int
    repeats: int
    tolerance: float
    max_logical_tasks: int
    max_waves: int
    max_frontier_width: int
    expected_path: tuple[tuple[int, int], ...] | None = None
    expected_wave_widths: tuple[int, ...] | None = None
    expected_dpu_task_counts: tuple[int, ...] | None = None
    expected_assignment_dpu_ids: tuple[int, ...] | None = None
    expected_transfer_edges: tuple[TransferEdgeExpectation, ...] | None = None
    require_rank_evidence: bool = False
    allow_slot_reuse: bool = False
    include_failures_in_normalized_records: bool = False


M45_CONTRACT = RouteStudyContract(
    suite_id=SUITE_ID,
    schema_version=SCHEMA_VERSION,
    profile_version=PROFILE_VERSION,
    route_id=ROUTE_ID,
    backend_id=BACKEND_ID,
    claim_boundary=CLAIM_BOUNDARY,
    route_label="upmem_simplepim_taskgraph",
    execution_scope="one_or_two_dpu_descriptor_driven_taskgraph",
    benchmark_role="physical_simplepim_shared_taskgraph_functionality",
    placements=PLACEMENTS,
    case_id="one_qubit_ry_h_ry_a",
    qasm_path="configs/circuits/upmem_m2/one_qubit_ry_h_ry_a.qasm",
    warmups=WARMUPS,
    repeats=REPEATS,
    tolerance=DEFAULT_TOLERANCE,
    max_logical_tasks=8,
    max_waves=8,
    max_frontier_width=2,
    require_rank_evidence=True,
)


@dataclass(frozen=True)
class PreparedPlacement:
    case_id: str
    request_id: str
    placement_policy: str
    source_graph: TaskGraph
    package_graph: TaskGraph
    package: ResidentGraphPackage
    plan: ExecutionPlan
    plan_path: Path
    schedule_path: Path
    request_path: Path
    package_path: Path
    source_output: np.ndarray
    package_file_sha256: str
    schedule_sidecar_sha256: str
    requested_rank_path: str | None


def load_upmem_simplepim_taskgraph_suite(path: Path) -> dict[str, Any]:
    """Resolve and enforce the one fixed physical M4.5 suite."""
    return load_route_study_suite(path, M45_CONTRACT)


def load_route_study_suite(path: Path, contract: RouteStudyContract) -> dict[str, Any]:
    """Resolve one bounded study without allowing its hardware profile to drift."""

    suite = load_suite(path)
    profile = suite.get("metadata", {}).get("hardware_profile")
    if suite.get("suite_id") != contract.suite_id or not isinstance(profile, Mapping):
        raise ValueError("hardware_profile_violation: unexpected SimplePIM TaskGraph suite")
    expected = {
        "hardware_profile_version": contract.profile_version,
        "target": "hardware",
        "backend_id": contract.backend_id,
        "route_id": contract.route_id,
        "tasklets_per_dpu": TASKLETS_PER_DPU,
        "numeric_mode": "none",
        "device_launch_mode": DEVICE_LAUNCH_MODE,
        "synchronization_policy": SYNCHRONIZATION_POLICY,
        "fully_synchronous_kernel_launch": False,
        "performance_claim_applicable": False,
    }
    if any(profile.get(key) != value for key, value in expected.items()):
        raise ValueError("hardware_profile_violation: study hardware profile is not fixed")
    if bool(profile.get("allow_slot_reuse", False)) != contract.allow_slot_reuse:
        raise ValueError("hardware_profile_violation: study slot-reuse policy differs")
    if (suite.get("warmups"), suite.get("repeats")) != (contract.warmups, contract.repeats):
        raise ValueError("hardware_profile_violation: study warmup/repeat contract differs")
    if suite.get("planner") != {"engine": "opt_einsum", "optimize": "greedy"}:
        raise ValueError("hardware_profile_violation: study requires opt_einsum greedy")
    if suite.get("route_policy", {}).get("routes") != [contract.route_id]:
        raise ValueError("hardware_profile_violation: study route identity differs")
    cases = suite.get("cases")
    if not isinstance(cases, list) or len(cases) != 1:
        raise ValueError("hardware_profile_violation: study requires one workload")
    case = cases[0]
    if case.get("case_id") != contract.case_id:
        raise ValueError("hardware_profile_violation: study case differs")
    circuit = case.get("circuit")
    if (
        not isinstance(circuit, Mapping)
        or circuit.get("name") != contract.case_id
        or circuit.get("path") != contract.qasm_path
    ):
        raise ValueError("hardware_profile_violation: study circuit differs")
    if case.get("placements") != list(contract.placements):
        raise ValueError("hardware_profile_violation: study placements differ")
    if float(suite.get("tolerances", {}).get("max_abs_error", 0.0)) != contract.tolerance:
        raise ValueError("hardware_profile_violation: study tolerance differs")
    for key, expected_value in (
        ("max_logical_tasks", contract.max_logical_tasks),
        ("max_waves", contract.max_waves),
        ("max_frontier_width", contract.max_frontier_width),
    ):
        if profile.get(key) != expected_value:
            raise ValueError(f"hardware_profile_violation: study {key} differs")
    if contract.expected_dpu_task_counts is not None and profile.get("requested_dpu_count") != len(contract.expected_dpu_task_counts):
        raise ValueError("hardware_profile_violation: study requested DPU count differs")
    return suite


def _contract_json(contract: RouteStudyContract) -> JsonDict:
    return {
        "suite_id": contract.suite_id,
        "profile_version": contract.profile_version,
        "route_id": contract.route_id,
        "backend_id": contract.backend_id,
        "placements": list(contract.placements),
        "case_id": contract.case_id,
        "qasm_path": contract.qasm_path,
        "warmups": contract.warmups,
        "repeats": contract.repeats,
        "tolerance": contract.tolerance,
        "max_logical_tasks": contract.max_logical_tasks,
        "max_waves": contract.max_waves,
        "max_frontier_width": contract.max_frontier_width,
        "expected_path": None if contract.expected_path is None else [list(step) for step in contract.expected_path],
        "expected_wave_widths": None if contract.expected_wave_widths is None else list(contract.expected_wave_widths),
        "expected_dpu_task_counts": None if contract.expected_dpu_task_counts is None else list(contract.expected_dpu_task_counts),
        "expected_assignment_dpu_ids": None
        if contract.expected_assignment_dpu_ids is None
        else list(contract.expected_assignment_dpu_ids),
        "expected_transfer_edges": None
        if contract.expected_transfer_edges is None
        else [
            {
                "producer_task_id": edge.producer_task_id,
                "consumer_task_id": edge.consumer_task_id,
                "producer_dpu_id": edge.producer_dpu_id,
                "consumer_dpu_id": edge.consumer_dpu_id,
                "slot_id": edge.slot_id,
                "element_count": edge.element_count,
                "transfer_bytes": edge.transfer_bytes,
            }
            for edge in contract.expected_transfer_edges
        ],
        "require_rank_evidence": contract.require_rank_evidence,
        "allow_slot_reuse": contract.allow_slot_reuse,
        "include_failures_in_normalized_records": contract.include_failures_in_normalized_records,
    }


def _validate_study_graph(contract: RouteStudyContract, graph: TaskGraph) -> None:
    if not 1 <= len(graph.tasks) <= contract.max_logical_tasks:
        raise ValueError("hardware_profile_violation: study logical-task cap exceeded")
    if contract.expected_path is not None and graph.path != contract.expected_path:
        raise ValueError("hardware_profile_violation: study contraction path differs")


def _validate_study_plan(contract: RouteStudyContract, graph: TaskGraph, plan: ExecutionPlan) -> None:
    _validate_study_graph(contract, graph)
    widths = tuple(len(wave) for wave in plan.waves)
    if not 1 <= plan.wave_count <= contract.max_waves:
        raise ValueError("hardware_profile_violation: study wave cap exceeded")
    if any(width > contract.max_frontier_width for width in widths):
        raise ValueError("hardware_profile_violation: study frontier-width cap exceeded")
    if contract.expected_wave_widths is not None and widths != contract.expected_wave_widths:
        raise ValueError("hardware_profile_violation: study wave widths differ")
    if contract.expected_dpu_task_counts is not None:
        expected = contract.expected_dpu_task_counts
        observed = tuple(
            sum(item.dpu_id == dpu for item in plan.assignments)
            for dpu in range(plan.requested_dpu_count)
        )
        if plan.requested_dpu_count != len(expected) or observed != expected:
            raise ValueError("hardware_profile_violation: study DPU task counts differ")
    if contract.expected_assignment_dpu_ids is not None:
        observed_assignment = tuple(item.dpu_id for item in plan.assignments)
        if observed_assignment != contract.expected_assignment_dpu_ids:
            raise ValueError("hardware_profile_violation: study DPU assignment differs")
    if contract.expected_transfer_edges is not None:
        observed_edges = tuple(
            TransferEdgeExpectation(
                producer_task_id=edge.producer_task_id,
                consumer_task_id=edge.consumer_task_id,
                producer_dpu_id=edge.producer_dpu_id,
                consumer_dpu_id=edge.consumer_dpu_id,
                slot_id=edge.slot_id,
                element_count=edge.element_count,
                transfer_bytes=edge.transfer_bytes,
            )
            for edge in plan.transfer_edges
        )
        if observed_edges != contract.expected_transfer_edges:
            raise ValueError("hardware_profile_violation: study transfer edges differ")


def prepare(
    root_dir: Path,
    *,
    suite_path: Path,
    build: bool = False,
    environment: Mapping[str, str] | None = None,
    native_target: Block2NativeTarget | None = None,
    plan_compiler: Callable[..., ExecutionPlan] | None = None,
) -> dict[str, Any]:
    """Build/parser-check plans only; never allocate or launch a DPU."""

    return prepare_route_study(
        root_dir,
        suite_path=suite_path,
        contract=M45_CONTRACT,
        build=build,
        environment=environment,
        native_target=native_target,
        plan_compiler=plan_compiler,
    )


def prepare_route_study(
    root_dir: Path,
    *,
    suite_path: Path,
    contract: RouteStudyContract,
    build: bool = False,
    environment: Mapping[str, str] | None = None,
    native_target: Block2NativeTarget | None = None,
    plan_compiler: Callable[..., ExecutionPlan] | None = None,
) -> dict[str, Any]:
    """Prepare a contract-bound route without allocating or launching a DPU."""

    suite = load_route_study_suite(suite_path, contract)
    plan_dir = _unique_dir(root_dir / "build" / f"{contract.suite_id}_plan")
    (plan_dir / "config").mkdir(parents=True)
    write_json(plan_dir / "config" / "resolved_suite.json", suite)
    write_json(plan_dir / "environment.json", capture_environment(root_dir))
    native_build: Mapping[str, Any] = {
        "status": "not_requested",
        "prepare_only": True,
        "allocation_attempted": False,
        "launch_attempted": False,
    }
    target = native_target or _load_block2_target()
    if build:
        native_build = _build_native(target, plan_dir / "native_build")
    dpu_binary = _path_value(native_build, "dpu_binary") or (
        plan_dir / "native_session" / "unbuilt_dpu_binary"
    )
    case = suite["cases"][0]
    source, package_graph, package_network, reference = _build_case(root_dir, suite, case, contract)
    package = _write_package(
        plan_dir, case, package_graph, package_network, reference, dpu_binary, contract
    )
    compiler = plan_compiler or compile_plan
    placements: list[PreparedPlacement] = []
    for placement in contract.placements:
        prepared = _prepare_placement(
            plan_dir,
            case,
            placement,
            source_graph=source,
            package_graph=package_graph,
            package=package,
            source_output=reference,
            dpu_binary=dpu_binary,
            plan_compiler=compiler,
            contract=contract,
            requested_rank_path=None,
        )
        _validate_prepared_request(target, prepared.request_path, float(suite["timeout_s"]))
        placements.append(prepared)
    result: JsonDict = {
        "schema_version": contract.schema_version,
        "suite_id": contract.suite_id,
        "route_id": contract.route_id,
        "backend_id": contract.backend_id,
        "status": "prepared",
        "claim_boundary": contract.claim_boundary,
        "study_contract": _contract_json(contract),
        "placements": [_placement_json(item) for item in placements],
        "native_build": dict(native_build),
        "preparation_mode": "parser_only_and_plan_validation",
        "dpu_allocation_attempted": False,
        "dpu_launch_attempted": False,
        "allocation_status": "not_attempted",
        "launch_status": "not_attempted",
        "package_file_sha256": _sha256_file(placements[0].package_path),
    }
    artifact = plan_dir / f"{contract.suite_id}_plan.json"
    write_json(artifact, result)
    return {"plan_dir": str(plan_dir), "artifact": str(artifact), **result}


def execute(
    root_dir: Path,
    *,
    suite_path: Path,
    environment: Mapping[str, str] | None = None,
    native_target: Block2NativeTarget | None = None,
    plan_compiler: Callable[..., ExecutionPlan] | None = None,
) -> dict[str, Any]:
    """Execute both placements through Block 2 with no fallback or retry."""

    return execute_route_study(
        root_dir,
        suite_path=suite_path,
        contract=M45_CONTRACT,
        environment=environment,
        native_target=native_target,
        plan_compiler=plan_compiler,
    )


def execute_route_study(
    root_dir: Path,
    *,
    suite_path: Path,
    contract: RouteStudyContract,
    environment: Mapping[str, str] | None = None,
    native_target: Block2NativeTarget | None = None,
    plan_compiler: Callable[..., ExecutionPlan] | None = None,
) -> dict[str, Any]:
    """Execute one contract-bound physical route study with no fallback or retry."""

    env = dict(os.environ if environment is None else environment)
    _require_hardware_opt_in(env)
    _require_contract_environment(contract, env)
    suite = load_route_study_suite(suite_path, contract)
    profile = dict(suite["metadata"]["hardware_profile"])
    run_dir = create_run_dir(
        root_dir,
        contract.suite_id,
        artifact_kind=EVIDENCE_ARTIFACT_KIND,
        route_label=contract.route_label,
    )
    shutil.copy2(suite_path, run_dir / "config" / "resolved_suite.yml")
    write_json(run_dir / "config" / "hardware_profile.json", profile)
    write_json(run_dir / "environment.json", capture_environment(root_dir))
    manifest = write_run_manifest(
        run_dir,
        run_kind=contract.schema_version,
        suite_id=contract.suite_id,
        suite_path=str(suite_path),
        artifact_kind=EVIDENCE_ARTIFACT_KIND,
        route_label=contract.route_label,
        route_id=contract.route_id,
        backend_id=contract.backend_id,
        execution_scope=contract.execution_scope,
        evidence_type="physical_hardware_functionality_only",
        upmem_execution_mode="simplepim_taskgraph_block2",
        artifact_retention="full",
        summary=f"{contract.suite_id}_summary.json",
        root_dir=root_dir,
    )
    target = native_target or _load_block2_target()
    native_build = _build_native(target, run_dir / "native_build")
    dpu_binary = _path_value(native_build, "dpu_binary")
    if dpu_binary is None:
        raise NativeExecutionFailure("native_build_failed", "native target did not return dpu_binary")
    case = suite["cases"][0]
    source, package_graph, package_network, reference = _build_case(root_dir, suite, case, contract)
    package = _write_package(
        run_dir, case, package_graph, package_network, reference, dpu_binary, contract
    )
    compiler = plan_compiler or compile_plan
    placements = [
        _prepare_placement(
            run_dir,
            case,
            placement,
            source_graph=source,
            package_graph=package_graph,
            package=package,
            source_output=reference,
            dpu_binary=dpu_binary,
            plan_compiler=compiler,
            contract=contract,
            requested_rank_path=env.get("UPMEM_HW_RANK_PATH"),
        )
        for placement in contract.placements
    ]
    measured: list[JsonDict] = []
    warmups: list[JsonDict] = []
    failures: list[JsonDict] = []
    stopped = False
    for item in placements:
        try:
            session = target.execute(item.request_path, timeout_s=float(suite["timeout_s"]))
            session_warmups, session_measured = _session_records(
                run_dir,
                item,
                session,
                native_build=native_build,
                contract=contract,
            )
            warmups.extend(session_warmups)
            measured.extend(session_measured)
        except TimeoutError as exc:
            failures.append(_failure_record(item, native_build, "kernel_timeout", str(exc), contract))
            stopped = True
        except NativeExecutionFailure as exc:
            failures.append(_failure_record(item, native_build, exc.stage, str(exc), contract))
            stopped = True
        except (OSError, RuntimeError, TypeError, ValueError, KeyError) as exc:
            stage = getattr(exc, "failure_stage", _failure_stage(str(exc)))
            failures.append(_failure_record(item, native_build, str(stage), str(exc), contract))
            stopped = True
        if stopped:
            break
    normalized_records = measured + failures if contract.include_failures_in_normalized_records else measured
    write_normalized_records(run_dir, normalized_records)
    write_jsonl(run_dir / "warmup_records.jsonl", warmups)
    write_jsonl(run_dir / "session_failures.jsonl", failures)
    expected_rows = len(contract.placements) * int(suite["repeats"])
    completed = not stopped and len(measured) == expected_rows and all(
        row["status"] == "completed" for row in measured
    )
    summary: JsonDict = {
        "schema_version": contract.schema_version,
        "suite_id": contract.suite_id,
        "route_id": contract.route_id,
        "backend_id": contract.backend_id,
        "status": "completed" if completed else "failed",
        "row_count": len(measured),
        "measured_row_count": len(measured),
        "failure_record_count": len(failures),
        "normalized_record_count": len(normalized_records),
        "warmup_count": len(warmups),
        "repeat_count": len(measured),
        "normalized_records": "normalized_records.jsonl",
        "warmup_records": "warmup_records.jsonl",
        "session_failures": "session_failures.jsonl",
        "session_failure_count": len(failures),
        "claim_boundary": contract.claim_boundary,
        "hardware_speedup_applicable": False,
        "placements": [_placement_json(item) for item in placements],
        "native_build": native_build,
        "study_contract": _contract_json(contract),
    }
    artifact = run_dir / f"{contract.suite_id}_summary.json"
    write_json(artifact, summary)
    manifest.update(
        {
            "summary": artifact.name,
            "hardware_available": "verified_by_execution" if completed else "not_verified",
        }
    )
    write_json(run_dir / "run_manifest.json", manifest)
    return {"run_dir": str(run_dir), "artifact": str(artifact), "status": summary["status"], "row_count": len(measured)}


def _prepare_placement(
    output_root: Path,
    case: Mapping[str, Any],
    placement: str,
    *,
    source_graph: TaskGraph,
    package_graph: TaskGraph,
    package: ResidentGraphPackage,
    source_output: np.ndarray,
    dpu_binary: Path,
    plan_compiler: Callable[..., ExecutionPlan],
    contract: RouteStudyContract,
    requested_rank_path: str | None,
) -> PreparedPlacement:
    directory = output_root / "cases" / sanitize(str(case["case_id"])) / sanitize(placement)
    directory.mkdir(parents=True, exist_ok=False)
    package_path = package.package_path
    if package_path is None:
        raise ValueError("manifest_parse_failed: resident package path is missing")
    package_bytes = package_path.read_bytes()
    package_hash = _sha256_bytes(package_bytes)
    validate_resident_graph_package_file(package_path)
    plan = plan_compiler(source_graph, package, placement_policy=placement)
    _validate_study_plan(contract, source_graph, plan)
    validate_plan(
        plan,
        graph=source_graph,
        package=package,
        package_bytes=package_bytes,
    )
    if plan.package_file_sha256 != package_hash:
        raise ValueError("package_file_sha256 does not match final resident package bytes")
    schedule_bytes = serialize_schedule(plan)
    validate_schedule(schedule_bytes, plan, package_bytes=package_bytes)
    schedule_path = directory / "execution_schedule.bin"
    schedule_path.write_bytes(schedule_bytes)
    schedule_hash = _sha256_bytes(schedule_bytes)
    if plan.schedule_sidecar_sha256 != schedule_hash:
        raise ValueError("schedule_sidecar_sha256 does not match final sidecar bytes")
    plan_path = directory / "execution_plan.json"
    write_json(plan_path, json.loads(serialize_plan_json(plan).decode("utf-8")))
    reference_path = directory / "cpu_reference.npy"
    np.save(reference_path, source_output, allow_pickle=False)
    request = _request_manifest(
        plan,
        package_path=_relative_path(package_path, directory),
        schedule_path=schedule_path.name,
        dpu_binary=_relative_path(dpu_binary, directory),
        schedule_bytes=schedule_bytes,
    )
    request.update({
            "target": "hardware",
            "route_id": contract.route_id,
            "backend_id": contract.backend_id,
            "hardware_profile_version": contract.profile_version,
            "placement_policy": plan.placement_policy,
            "request_id": f"{case['case_id']}-{sanitize(placement)}",
            "upmem_execution_plan_hash": plan.execution_plan_hash,
            "requested_warmups": contract.warmups,
            "requested_repetitions": contract.repeats,
            "device_launch_mode": DEVICE_LAUNCH_MODE,
            "synchronization_policy": SYNCHRONIZATION_POLICY,
            "fully_synchronous_kernel_launch": False,
            "persistent_allocation_required": True,
            "schedule_h2d_bytes": 0,
            "hardware_speedup_applicable": False,
            "source_circuit_semantics_hash": plan.source_circuit_semantics_hash,
            "source_tensor_network_hash": plan.source_tensor_network_hash,
            "source_contraction_plan_hash": plan.source_contraction_plan_hash,
            "package_circuit_semantics_hash": plan.package_circuit_semantics_hash,
            "package_tensor_network_hash": plan.package_tensor_network_hash,
            "package_contraction_plan_hash": plan.package_contraction_plan_hash,
            "requested_rank_path": requested_rank_path,
        }
    )
    request_path = directory / "block2_request.json"
    write_json(request_path, request)
    return PreparedPlacement(
        case_id=str(case["case_id"]),
        request_id=str(request["request_id"]),
        placement_policy=placement,
        source_graph=source_graph,
        package_graph=package_graph,
        package=package,
        plan=plan,
        plan_path=plan_path,
        schedule_path=schedule_path,
        request_path=request_path,
        package_path=package_path,
        source_output=np.asarray(source_output),
        package_file_sha256=package_hash,
        schedule_sidecar_sha256=schedule_hash,
        requested_rank_path=requested_rank_path,
    )


def _build_case(
    root_dir: Path,
    suite: Mapping[str, Any],
    case: Mapping[str, Any],
    contract: RouteStudyContract,
) -> tuple[TaskGraph, TaskGraph, TensorNetworkValue, np.ndarray]:
    circuit_root = root_dir
    circuit_path = case.get("circuit", {}).get("path")
    if isinstance(circuit_path, str) and not (root_dir / circuit_path).is_file():
        circuit_root = Path(__file__).resolve().parents[3]
    circuit = load_circuit(dict(case), circuit_root)
    network = build_tensor_network(circuit)
    source = with_execution_identity(plan_task_graph_with_config(network, dict(suite["planner"])))
    _validate_study_graph(contract, source)
    _require_real_network(network)
    reference, _ = execute_task_sequence_np_einsum(source, network)
    package_graph, package_network = _lower_real_float32(source, network)
    return source, package_graph, package_network, np.asarray(reference)


def _request_manifest(
    plan: ExecutionPlan,
    *,
    package_path: str,
    schedule_path: str,
    dpu_binary: str,
    schedule_bytes: bytes,
) -> JsonDict:
    """Construct one request from the compiled plan; never patch DPU counts."""

    schedule_hash = hashlib.sha256(schedule_bytes).hexdigest()
    if plan.schedule_sidecar_sha256 != schedule_hash:
        raise ValueError("schedule_sidecar_sha256 does not match request sidecar")
    return {
        "schema_version": "upmem_execution_plan_request_v1",
        "manifest_kind": "upmem_execution_plan_request",
        "runtime_provider_id": plan.runtime_provider_id,
        "kernel_provider_id": plan.kernel_provider_id,
        "communication_provider_id": plan.communication_provider_id,
        "numeric_mode": plan.numeric_mode,
        "placement_policy": plan.placement_policy,
        "requested_dpu_count": plan.requested_dpu_count,
        "tasklets_per_dpu": plan.tasklets_per_dpu,
        "package_path": package_path,
        "schedule_path": schedule_path,
        "dpu_binary": dpu_binary,
        "package_file_sha256": plan.package_file_sha256,
        "schedule_sidecar_sha256": schedule_hash,
        "schedule_sidecar_h2d_bytes": 0,
        "schedule_sidecar_scope": "host_metadata_not_h2d",
        "execution_plan_hash": plan.execution_plan_hash,
        "source_identity": {
            "circuit_semantics_hash": plan.source_circuit_semantics_hash,
            "tensor_network_hash": plan.source_tensor_network_hash,
            "contraction_plan_hash": plan.source_contraction_plan_hash,
        },
        "package_identity": {
            "circuit_semantics_hash": plan.package_circuit_semantics_hash,
            "tensor_network_hash": plan.package_tensor_network_hash,
            "contraction_plan_hash": plan.package_contraction_plan_hash,
        },
        "final_outputs": [to_jsonable(item) for item in plan.final_outputs],
    }


def _write_package(
    output_root: Path,
    case: Mapping[str, Any],
    package_graph: TaskGraph,
    package_network: TensorNetworkValue,
    reference: np.ndarray,
    dpu_binary: Path,
    contract: RouteStudyContract,
) -> ResidentGraphPackage:
    package = build_resident_graph_package(
        package_graph,
        package_network,
        case_id=str(case["case_id"]),
        suite_id=contract.suite_id,
        quantization_mode="none",
        full_precision_output=reference,
        allow_slot_reuse=contract.allow_slot_reuse,
    )
    _validate_real_package(package)
    artifact = package.write(
        output_root,
        dpu_binary=dpu_binary,
        request_id=str(case["case_id"]),
    )
    if artifact.package_path is None:
        raise ValueError("manifest_parse_failed: package writer returned no package")
    validate_resident_graph_package_file(artifact.package_path)
    return artifact


def _session_records(
    run_dir: Path,
    prepared: PreparedPlacement,
    session: Mapping[str, Any],
    *,
    native_build: Mapping[str, Any],
    contract: RouteStudyContract,
) -> tuple[list[JsonDict], list[JsonDict]]:
    if not isinstance(session, dict):
        raise NativeExecutionFailure("output_manifest_failed", "adapter session is not mutable")
    if session.get("status") != "completed":
        _validate_adapter_session(session, prepared, contract)
    _complete_session_validation(session, prepared, contract)
    _validate_adapter_session(session, prepared, contract)
    session_path = prepared.request_path.parent / "adapter_session.json"
    write_json(session_path, dict(session))
    warmups: list[JsonDict] = []
    measured: list[JsonDict] = []
    for warmup, repetitions in (
        (True, [item for item in session["repetitions"] if item["warmup"] is True]),
        (False, [item for item in session["repetitions"] if item["warmup"] is False]),
    ):
        for repetition in repetitions:
            row = _base_record(
                prepared,
                native_build=native_build,
                warmup=warmup,
                repeat_id=repetition["repeat_id"],
                contract=contract,
            )
            row.update(_adapter_record(session, repetition, session_path, run_dir))
            row.update({
                "status": "completed",
                "validation_status": "not_individually_collected",
                "scientific_validation_status": "session_validation_passed_final_output_only",
                "session_validation_id": session["session_validation"]["validation_id"],
                "session_validation_status": "passed",
                "repeat_output_validation_status": "not_individually_collected",
                "validation_max_abs_error": None,
                "validation_tolerance_abs": None,
                "session_validation_max_abs_error": session["session_validation"]["max_abs_error"],
                "session_validation_tolerance_abs": contract.tolerance,
                "failure_stage": None,
            })
            (warmups if warmup else measured).append(row)
    return warmups, measured


def _validate_adapter_session(
    session: Mapping[str, Any], prepared: PreparedPlacement, contract: RouteStudyContract
) -> None:
    if not isinstance(session, Mapping):
        raise NativeExecutionFailure("output_manifest_failed", "adapter session is not a mapping")
    if session.get("status") != "completed":
        raise NativeExecutionFailure(
            str(session.get("failure_stage") or "kernel_launch_failed"),
            str(session.get("reason") or session.get("error") or "adapter session failed"),
        )
    _require_fields(session, (
        "schema_version", "returncode", "request_id", "backend_id", "target_requested",
        "target_observed", "execution_plan_hash", "package_file_sha256",
        "schedule_sidecar_sha256", "requested_dpu_count", "allocated_dpu_count",
        "tasklets_per_dpu", "allocation_attempted", "allocation_count",
        "allocation_succeeded", "persistent_allocation_observed", "release_confirmed", "hardware_allocation_verified",
        "native_kernel_executed", "hardware_kernel_executed", "simulator_kernel_executed",
        "cpu_fallback_used", "hardware_speedup_applicable", "device_launch_mode",
        "synchronization_policy", "fully_synchronous_kernel_launch", "requested_warmups",
        "requested_repetitions", "native_session_count", "logical_task_count",
        "session_completion_scope", "aggregate_completed_per_dpu",
        "aggregate_total_task_completion_count", "aggregate_session_completion_id",
        "aggregate_session_completion_status",
        "total_task_completion_count", "exactly_once_execution_verified",
        "wave_barrier_count_total", "operation_assignments", "cross_dpu_transfer",
        "schedule_h2d_bytes", "session_timing", "session_transfer", "session_validation", "repetitions",
    ), "adapter session")
    expected = {
        "schema_version": ADAPTER_SESSION_SCHEMA,
        "returncode": 0,
        "request_id": prepared.request_id,
        "backend_id": NATIVE_BACKEND_ID,
        "target_requested": "hardware",
        "target_observed": "physical_hardware",
        "execution_plan_hash": prepared.plan.execution_plan_hash,
        "package_file_sha256": prepared.package_file_sha256,
        "schedule_sidecar_sha256": prepared.schedule_sidecar_sha256,
        "requested_dpu_count": prepared.plan.requested_dpu_count,
        "allocated_dpu_count": prepared.plan.requested_dpu_count,
        "tasklets_per_dpu": TASKLETS_PER_DPU,
        "allocation_attempted": True,
        "allocation_count": 1,
        "allocation_succeeded": True,
        "persistent_allocation_observed": True,
        "release_confirmed": True,
        "hardware_allocation_verified": True,
        "native_kernel_executed": True,
        "hardware_kernel_executed": True,
        "simulator_kernel_executed": False,
        "cpu_fallback_used": False,
        "hardware_speedup_applicable": False,
        "device_launch_mode": DEVICE_LAUNCH_MODE,
        "synchronization_policy": SYNCHRONIZATION_POLICY,
        "fully_synchronous_kernel_launch": False,
        "requested_warmups": contract.warmups,
        "requested_repetitions": contract.repeats,
        "native_session_count": 1,
        "logical_task_count": prepared.plan.logical_task_count,
        "session_completion_scope": "aggregate_across_warmups_and_repetitions",
        "aggregate_completed_per_dpu": [
            sum(item.dpu_id == dpu for item in prepared.plan.assignments)
            * (contract.warmups + contract.repeats)
            for dpu in range(prepared.plan.requested_dpu_count)
        ],
        "aggregate_total_task_completion_count": prepared.plan.logical_task_count * (contract.warmups + contract.repeats),
        "aggregate_session_completion_status": "passed",
        "total_task_completion_count": prepared.plan.logical_task_count * (contract.warmups + contract.repeats),
        "exactly_once_execution_verified": True,
        "wave_barrier_count_total": prepared.plan.wave_count * (contract.warmups + contract.repeats),
        "operation_assignments": [to_jsonable(item) for item in prepared.plan.assignments],
        "schedule_h2d_bytes": 0,
    }
    for key, value in expected.items():
        if session[key] != value:
            raise NativeExecutionFailure("output_manifest_failed", f"adapter session field {key} mismatch")
    _validate_rank_evidence(session, prepared, contract)
    aggregate_completion_id = session["aggregate_session_completion_id"]
    if not isinstance(aggregate_completion_id, str) or not aggregate_completion_id.startswith("aggregate_session_completion:"):
        raise NativeExecutionFailure("output_manifest_failed", "aggregate session completion ID is invalid")
    for identity_name, expected_identity in (
        ("source_identity", {
            "circuit_semantics_hash": prepared.plan.source_circuit_semantics_hash,
            "tensor_network_hash": prepared.plan.source_tensor_network_hash,
            "contraction_plan_hash": prepared.plan.source_contraction_plan_hash,
        }),
        ("package_identity", {
            "circuit_semantics_hash": prepared.plan.package_circuit_semantics_hash,
            "tensor_network_hash": prepared.plan.package_tensor_network_hash,
            "contraction_plan_hash": prepared.plan.package_contraction_plan_hash,
        }),
    ):
        if session.get(identity_name) != expected_identity:
            raise NativeExecutionFailure(
                "output_manifest_failed",
                f"adapter session {identity_name} differs from execution plan",
            )
    _validate_completed_session_validation(session["session_validation"], prepared, contract)
    transfer = session["cross_dpu_transfer"]
    if not isinstance(transfer, Mapping) or transfer.get("count") != len(prepared.plan.transfer_edges) or transfer.get("bytes") != prepared.plan.total_cross_dpu_transfer_bytes:
        raise NativeExecutionFailure("output_manifest_failed", "cross-DPU transfer evidence mismatch")
    _validate_timing(session["session_timing"], (
        "allocation_time_s", "binary_load_time_s", "descriptor_h2d_time_s",
        "release_time_s", "total_session_time_s",
    ))
    repetitions = session["repetitions"]
    if not isinstance(repetitions, list) or len(repetitions) != contract.warmups + contract.repeats:
        raise NativeExecutionFailure("output_manifest_failed", "adapter repetition count mismatch")
    warmup_ids = [item.get("repeat_id") for item in repetitions if item.get("warmup") is True]
    measured_ids = [item.get("repeat_id") for item in repetitions if item.get("warmup") is False]
    if warmup_ids != list(range(contract.warmups)) or measured_ids != list(range(contract.repeats)):
        raise NativeExecutionFailure("output_manifest_failed", "adapter repetition identities mismatch")
    for repetition in repetitions:
        _validate_repetition(repetition, prepared, aggregate_completion_id)
        if repetition["validation_id"] != session["session_validation"]["validation_id"]:
            raise NativeExecutionFailure("output_manifest_failed", "repetition validation ID differs from session")
    _validate_session_transfer(session["session_transfer"], repetitions)


def _validate_rank_evidence(
    session: Mapping[str, Any], prepared: PreparedPlacement, contract: RouteStudyContract
) -> None:
    """Require rank identity only for studies that execute against a fixed rank."""

    if not contract.require_rank_evidence:
        return
    _require_fields(session, ("requested_rank_path", "observed_rank_count"), "adapter session")
    # Physical execution always uses PreparedPlacement, which records the
    # requested rank before native build. Keep the validator reusable for
    # parser-only adapter fixtures that predate that field.
    requested_rank_path = getattr(prepared, "requested_rank_path", session["requested_rank_path"])
    if not isinstance(requested_rank_path, str) or not requested_rank_path:
        raise NativeExecutionFailure("hardware_profile_violation", "M6a rank path was not requested")
    if session["requested_rank_path"] != requested_rank_path:
        raise NativeExecutionFailure("output_manifest_failed", "native rank path differs from request")
    if session["observed_rank_count"] != 1:
        raise NativeExecutionFailure("output_manifest_failed", "native session did not observe one rank")


def _complete_session_validation(
    session: dict[str, Any], prepared: PreparedPlacement, contract: RouteStudyContract
) -> None:
    """Validate the one final output retained by a persistent native session."""

    validation = _require_fields(
        session.get("session_validation"),
        (
            "validation_id", "status", "scope", "output",
            "final_output_path", "output_sha256", "output_provenance",
        ),
        "session validation",
    )
    if not isinstance(validation["validation_id"], str) or not validation["validation_id"]:
        raise NativeExecutionFailure("output_manifest_failed", "session validation ID is invalid")
    if validation["scope"] != "final_session_output_only":
        raise NativeExecutionFailure("output_manifest_failed", "session validation scope is invalid")
    if validation["status"] != "collected":
        raise NativeExecutionFailure("output_validation_failed", "session output validation did not complete")
    output = _extract_output(validation, prepared.request_path.parent, prepared.source_output.shape)
    error = _max_abs_error(output, prepared.source_output.real)
    if error > contract.tolerance:
        raise NativeExecutionFailure("output_validation_failed", f"max_abs_error={error}")
    session["session_validation"] = {
        **validation,
        "status": "passed",
        "max_abs_error": error,
        "tolerance_abs": contract.tolerance,
    }


def _validate_completed_session_validation(
    validation: Any, prepared: PreparedPlacement, contract: RouteStudyContract
) -> None:
    value = _require_fields(
        validation,
        (
            "validation_id", "status", "scope", "output", "final_output_path",
            "output_sha256", "output_provenance", "max_abs_error", "tolerance_abs",
        ),
        "completed session validation",
    )
    if value["status"] != "passed" or value["scope"] != "final_session_output_only":
        raise NativeExecutionFailure("output_validation_failed", "session validation is not passed")
    if not isinstance(value["validation_id"], str) or not value["validation_id"]:
        raise NativeExecutionFailure("output_manifest_failed", "completed session validation ID is invalid")
    if value["tolerance_abs"] != contract.tolerance:
        raise NativeExecutionFailure("output_manifest_failed", "session validation tolerance differs")
    error = value["max_abs_error"]
    if isinstance(error, bool) or not isinstance(error, (int, float)) or not math.isfinite(float(error)) or error < 0 or error > contract.tolerance:
        raise NativeExecutionFailure("output_validation_failed", "session validation error is invalid")


def _validate_repetition(
    repetition: Mapping[str, Any],
    prepared: PreparedPlacement,
    aggregate_completion_id: str,
) -> None:
    _require_fields(repetition, (
        "repeat_id", "warmup", "status", "scheduled_task_count", "wave_barrier_count",
        "launch_count", "synchronize_count", "device_launch_mode", "synchronization_policy",
        "fully_synchronous_kernel_launch", "timing", "transfer", "validation_id",
        "repeat_output_validation_status", "session_completion_scope",
        "repeat_completion_observation_status", "aggregate_session_completion_id",
        "aggregate_session_completion_status",
    ), "adapter repetition")
    expected = {
        "status": "completed",
        "scheduled_task_count": prepared.plan.logical_task_count,
        "wave_barrier_count": prepared.plan.wave_count,
        "launch_count": prepared.plan.logical_task_count,
        "synchronize_count": prepared.plan.logical_task_count,
        "device_launch_mode": DEVICE_LAUNCH_MODE,
        "synchronization_policy": SYNCHRONIZATION_POLICY,
        "fully_synchronous_kernel_launch": False,
        "session_completion_scope": "aggregate_across_warmups_and_repetitions",
        "repeat_completion_observation_status": "not_individually_collected",
        "aggregate_session_completion_id": aggregate_completion_id,
        "aggregate_session_completion_status": "passed",
    }
    for key, value in expected.items():
        if repetition[key] != value:
            raise NativeExecutionFailure("output_manifest_failed", f"adapter repetition field {key} mismatch")
    if not isinstance(repetition["validation_id"], str) or not repetition["validation_id"]:
        raise NativeExecutionFailure("output_manifest_failed", "adapter repetition validation ID is invalid")
    if repetition["repeat_output_validation_status"] != "not_individually_collected":
        raise NativeExecutionFailure("output_manifest_failed", "adapter repetition claims output validation")
    _validate_timing(repetition["timing"], (
        "operand_h2d_time_s", "cross_dpu_transfer_time_s", "wave_launch_sync_time_s",
        "final_d2h_time_s", "total_repetition_time_s",
    ))
    transfer = repetition["transfer"]
    _require_fields(transfer, ("h2d_bytes", "d2h_bytes", "total_bytes"), "repetition transfer")
    if not all(_is_uint(transfer[key]) for key in ("h2d_bytes", "d2h_bytes", "total_bytes")) or transfer["total_bytes"] != transfer["h2d_bytes"] + transfer["d2h_bytes"]:
        raise NativeExecutionFailure("output_manifest_failed", "transfer-byte invariant failed")


def _validate_session_transfer(
    transfer: Any, repetitions: Sequence[Mapping[str, Any]]
) -> None:
    """Admit only session totals that reconcile with per-repetition records."""

    _require_fields(
        transfer,
        (
            "initial_h2d_bytes",
            "actual_h2d_bytes",
            "actual_d2h_bytes",
            "actual_transfer_bytes",
        ),
        "adapter session transfer",
    )
    if not all(_is_uint(transfer[key]) for key in transfer):
        raise NativeExecutionFailure("output_manifest_failed", "session transfer bytes are invalid")
    repeated_h2d = sum(item["transfer"]["h2d_bytes"] for item in repetitions)
    repeated_d2h = sum(item["transfer"]["d2h_bytes"] for item in repetitions)
    if transfer["actual_h2d_bytes"] != transfer["initial_h2d_bytes"] + repeated_h2d:
        raise NativeExecutionFailure("output_manifest_failed", "session H2D bytes do not reconcile")
    if transfer["actual_d2h_bytes"] != repeated_d2h:
        raise NativeExecutionFailure("output_manifest_failed", "session D2H bytes do not reconcile")
    if transfer["actual_transfer_bytes"] != transfer["actual_h2d_bytes"] + transfer["actual_d2h_bytes"]:
        raise NativeExecutionFailure("output_manifest_failed", "session transfer-byte invariant failed")


def _base_record(
    prepared: PreparedPlacement,
    *,
    native_build: Mapping[str, Any],
    warmup: bool,
    repeat_id: int | None,
    contract: RouteStudyContract,
) -> JsonDict:
    source = _identity(prepared.source_graph)
    package = _identity(prepared.package_graph)
    row: JsonDict = {
        "schema_version": "normalized_records_v1", "case_id": prepared.case_id,
        "route_id": contract.route_id, "backend_id": contract.backend_id,
        "benchmark_role": contract.benchmark_role,
        "execution_model": "descriptor_driven_simplepim_taskgraph",
        "placement_policy": prepared.placement_policy,
        "device_launch_mode": DEVICE_LAUNCH_MODE,
        "synchronization_policy": SYNCHRONIZATION_POLICY,
        "fully_synchronous_kernel_launch": False,
        "upmem_execution_plan_hash": prepared.plan.execution_plan_hash,
        "execution_plan_hash": prepared.plan.execution_plan_hash,
        "package_file_sha256": prepared.package_file_sha256,
        "schedule_sidecar_sha256": prepared.schedule_sidecar_sha256,
        "runtime_provider_id": prepared.plan.runtime_provider_id,
        "kernel_provider_id": prepared.plan.kernel_provider_id,
        "communication_provider_id": prepared.plan.communication_provider_id,
        "requested_dpu_count": prepared.plan.requested_dpu_count,
        "tasklets_per_dpu": TASKLETS_PER_DPU, "source_task_count": prepared.plan.logical_task_count,
        "wave_count": prepared.plan.wave_count,
        "assignments": [to_jsonable(item) for item in prepared.plan.assignments],
        "cross_dpu_transfer_count": len(prepared.plan.transfer_edges),
        "cross_dpu_transfer_bytes": prepared.plan.total_cross_dpu_transfer_bytes,
        "hardware_execution": True, "hardware_speedup_applicable": False,
        "hardware_functionality_evidence": True, "timing_is_bringup_only": True,
        "cpu_fallback_used": False, "simulator_kernel_executed": False,
        "native_kernel_executed": False, "hardware_kernel_executed": False,
        "release_confirmed": False, "warmup": warmup, "repeat_id": repeat_id,
        "validation_status": "not_run", "scientific_validation_status": "not_run",
        "failure_stage": None,
        "allow_slot_reuse": contract.allow_slot_reuse,
        "resident_slot_reuse_limited": not contract.allow_slot_reuse,
        "requested_rank_path": prepared.requested_rank_path,
        "observed_rank_count": None,
    }
    row.update({f"source_{key}": value for key, value in source.items()})
    row.update({f"package_{key}": value for key, value in package.items()})
    row.update({key: source[key] for key in ("circuit_semantics_hash", "tensor_network_hash", "contraction_plan_hash")})
    row.update({key: None for key in ("allocated_dpu_count", "timing_scope", "allocation_time_s", "binary_load_time_s", "initial_h2d_time_s", "kernel_time_s", "inter_wave_transfer_time_s", "final_d2h_time_s", "total_route_time_s", "actual_h2d_bytes", "actual_d2h_bytes", "actual_transfer_bytes")})
    row.update({"host_binary_hash": _binary_hash(_path_value(native_build, "host_binary")), "dpu_binary_hash": _binary_hash(_path_value(native_build, "dpu_binary"))})
    return row


def _adapter_record(
    session: Mapping[str, Any],
    repetition: Mapping[str, Any],
    session_path: Path,
    run_dir: Path,
) -> JsonDict:
    session_timing = session["session_timing"]
    timing = repetition["timing"]
    transfer = repetition["transfer"]
    return {
        "allocated_dpu_count": session["allocated_dpu_count"],
        "scheduled_task_count": repetition["scheduled_task_count"],
        "session_completion_scope": session["session_completion_scope"],
        "aggregate_completed_per_dpu": session["aggregate_completed_per_dpu"],
        "aggregate_total_task_completion_count": session["aggregate_total_task_completion_count"],
        "aggregate_session_completion_id": session["aggregate_session_completion_id"],
        "aggregate_session_completion_status": repetition["aggregate_session_completion_status"],
        "aggregate_exactly_once_execution_verified": session["exactly_once_execution_verified"],
        "repeat_completion_observation_status": repetition["repeat_completion_observation_status"],
        "native_session_count": session["native_session_count"],
        "persistent_allocation_observed": session["persistent_allocation_observed"],
        "native_kernel_executed": session["native_kernel_executed"],
        "hardware_kernel_executed": session["hardware_kernel_executed"],
        "simulator_kernel_executed": session["simulator_kernel_executed"],
        "cpu_fallback_used": session["cpu_fallback_used"],
        "release_confirmed": session["release_confirmed"],
        "requested_rank_path": session.get("requested_rank_path"),
        "observed_rank_count": session.get("observed_rank_count"),
        "timing_scope": "host_observed_wave_launch_and_barrier_boundaries",
        "adapter_session_path": str(session_path.relative_to(run_dir)),
        "allocation_time_s": session_timing["allocation_time_s"],
        "binary_load_time_s": session_timing["binary_load_time_s"],
        "descriptor_h2d_time_s": session_timing["descriptor_h2d_time_s"],
        "initial_h2d_time_s": timing["operand_h2d_time_s"],
        "kernel_time_s": None,
        "wave_launch_sync_time_s": timing["wave_launch_sync_time_s"],
        "inter_wave_transfer_time_s": timing["cross_dpu_transfer_time_s"],
        "final_d2h_time_s": timing["final_d2h_time_s"],
        "total_repetition_time_s": timing["total_repetition_time_s"],
        "total_route_time_s": None,
        "actual_h2d_bytes": transfer["h2d_bytes"],
        "actual_d2h_bytes": transfer["d2h_bytes"],
        "actual_transfer_bytes": transfer["total_bytes"],
        "wave_barrier_count": repetition["wave_barrier_count"],
        "launch_count": repetition["launch_count"],
        "synchronize_count": repetition["synchronize_count"],
    }


def _failure_record(
    prepared: PreparedPlacement,
    native_build: Mapping[str, Any],
    stage: str,
    reason: str,
    contract: RouteStudyContract,
) -> JsonDict:
    row = _base_record(
        prepared, native_build=native_build, warmup=False, repeat_id=None, contract=contract
    )
    row.update({"status": "failed", "failure_stage": stage, "reason": reason})
    return row


def _placement_json(item: PreparedPlacement) -> JsonDict:
    return {
        "case_id": item.case_id,
        "placement_policy": item.placement_policy,
        "plan_path": str(item.plan_path),
        "schedule_path": str(item.schedule_path),
        "request_path": str(item.request_path),
        "package_path": str(item.package_path),
        "package_file_sha256": item.package_file_sha256,
        "schedule_sidecar_sha256": item.schedule_sidecar_sha256,
        "upmem_execution_plan_hash": item.plan.execution_plan_hash,
        "requested_dpu_count": item.plan.requested_dpu_count,
        "task_count": item.plan.logical_task_count,
        "wave_count": item.plan.wave_count,
        "assignments": [to_jsonable(assignment) for assignment in item.plan.assignments],
        "cross_dpu_transfer_count": len(item.plan.transfer_edges),
        "cross_dpu_transfer_bytes": item.plan.total_cross_dpu_transfer_bytes,
    }


def _lower_real_float32(graph: TaskGraph, network: TensorNetworkValue) -> tuple[TaskGraph, TensorNetworkValue]:
    specs = tuple(replace(tensor.spec, dtype="float32") for tensor in network.tensors)
    lowered_spec = replace(network.spec, tensors=specs)
    values = []
    for spec, tensor in zip(specs, network.tensors, strict=True):
        array = np.asarray(np.asarray(tensor.array).real, dtype=np.float32)
        if not np.all(np.isfinite(array)):
            raise ValueError("hardware_profile_violation: float32 lowering produced nonfinite values")
        values.append(TensorValue(spec, array))
    lowered_graph = with_execution_identity(replace(graph, network=lowered_spec, circuit_semantics_hash="", tensor_network_hash="", contraction_plan_hash=""))
    return lowered_graph, TensorNetworkValue(lowered_spec, values)


def _require_real_network(network: TensorNetworkValue) -> None:
    for tensor in network.tensors:
        array = np.asarray(tensor.array)
        if np.iscomplexobj(array) and np.any(np.asarray(array).imag != 0):
            raise ValueError("hardware_profile_violation: split-complex packages are unsupported")


def _validate_real_package(package: ResidentGraphPackage) -> None:
    if len(package.allocation.final_components) != 1:
        raise ValueError("hardware_profile_violation: multi-component package is unsupported")
    if any(operation.component != "real" or operation.kind != "contract" or operation.mode != "none" for operation in package.operations):
        raise ValueError("hardware_profile_violation: package is not real-only")


def _validate_prepared_request(
    target: Block2NativeTarget,
    request_path: Path,
    timeout_s: float,
) -> None:
    """Require Block 2's parser validation to be dry-run and successful."""

    try:
        value = target.validate(request_path, timeout_s=timeout_s)
    except NativeExecutionFailure:
        raise
    except TimeoutError as exc:
        raise NativeExecutionFailure("package_preparation_failed", "adapter validation timed out") from exc
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        stage = getattr(exc, "failure_stage", "package_preparation_failed")
        raise NativeExecutionFailure(str(stage), str(exc)) from exc
    if not isinstance(value, Mapping):
        raise NativeExecutionFailure("package_preparation_failed", "adapter validation result is not a mapping")
    if value.get("status") != "passed":
        raise NativeExecutionFailure(
            str(value.get("failure_stage") or "package_preparation_failed"),
            str(value.get("reason") or value.get("error") or "adapter validation did not pass"),
        )
    if value.get("allocation_attempted") is not False or value.get("launch_attempted") is not False:
        raise NativeExecutionFailure(
            "hardware_profile_violation",
            "adapter validation must explicitly report no allocation and no launch",
        )


def _load_block2_target() -> Block2NativeTarget:
    try:
        module = importlib.import_module(NATIVE_TARGET_MODULE)
    except ImportError as exc:
        raise RuntimeError(f"block2_native_target_missing: {NATIVE_TARGET_MODULE}") from exc
    target = getattr(module, "TASKGRAPH_TARGET", module)
    if any(not callable(getattr(target, name, None)) for name in ("build", "validate", "execute")):
        raise RuntimeError("block2_native_target_invalid: build(), validate(), and execute() are required")
    return target


def _build_native(target: Block2NativeTarget, build_dir: Path) -> Mapping[str, Any]:
    """Build the adapter target once without entering a hardware session."""

    try:
        value = target.build(build_dir, prepare_only=True)
    except NativeExecutionFailure:
        raise
    except TimeoutError as exc:
        raise NativeExecutionFailure("native_build_failed", "native build timed out") from exc
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        stage = getattr(exc, "failure_stage", "native_build_failed")
        raise NativeExecutionFailure(str(stage), str(exc)) from exc
    if not isinstance(value, Mapping):
        raise NativeExecutionFailure("native_build_failed", "native build result is not a mapping")
    _validate_prepare_build(value)
    for key in ("host_binary", "dpu_binary"):
        path = _path_value(value, key)
        if path is None or not path.is_file():
            raise NativeExecutionFailure("native_build_failed", f"native build did not produce {key}")
    return dict(value)


def _require_hardware_opt_in(environment: Mapping[str, str]) -> None:
    if environment.get("UPMEM_ALLOW_PHYSICAL_HARDWARE") != "1":
        raise NativeExecutionFailure("hardware_opt_in_missing", "UPMEM_ALLOW_PHYSICAL_HARDWARE=1 is required")
    if environment.get("DPU_BACKEND", "").lower() in {"simulator", "cpu", "mock"} or environment.get("UPMEM_EXECUTION_MODE", "").lower() in {"simulator", "cpu", "mock"}:
        raise NativeExecutionFailure("hardware_profile_violation", "simulator/CPU selectors are forbidden")


def _require_contract_environment(
    contract: RouteStudyContract, environment: Mapping[str, str]
) -> None:
    try:
        rank_path = hardware_environment_metadata(environment)["upmem_rank_path_requested"]
    except ValueError as exc:
        raise NativeExecutionFailure("hardware_profile_violation", str(exc)) from exc
    if contract.require_rank_evidence and not rank_path:
        raise NativeExecutionFailure(
            "hardware_profile_violation", "UPMEM_HW_RANK_PATH is required for this route"
        )


def _validate_prepare_build(value: Mapping[str, Any]) -> None:
    if value.get("status") != "built":
        raise NativeExecutionFailure("native_build_failed", str(value.get("reason") or "native build failed"))
    if value.get("prepare_only") is not True:
        raise NativeExecutionFailure("hardware_profile_violation", "native build must explicitly be prepare-only")
    if value.get("allocation_attempted") is not False or value.get("launch_attempted") is not False:
        raise NativeExecutionFailure("hardware_profile_violation", "prepare/build must not allocate or launch a DPU")


def _require_fields(value: Any, fields: Sequence[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise NativeExecutionFailure("output_manifest_failed", f"{label} is not a mapping")
    missing = [field for field in fields if field not in value]
    if missing:
        raise NativeExecutionFailure(
            "output_manifest_failed",
            f"{label} is missing required fields: {', '.join(missing)}",
        )
    return value


def _is_uint(value: Any) -> bool:
    return isinstance(value, Integral) and not isinstance(value, bool) and value >= 0


def _validate_timing(value: Any, fields: Sequence[str]) -> Mapping[str, Any]:
    timing = _require_fields(value, fields, "timing")
    for field in fields:
        item = timing[field]
        if item is None:
            continue
        if isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(float(item)) or item < 0:
            raise NativeExecutionFailure("output_manifest_failed", f"timing field {field} is invalid")
    return timing


def _extract_output(response: Mapping[str, Any], root: Path, shape: Sequence[int]) -> np.ndarray:
    value = response.get("output", response.get("final_output"))
    if isinstance(value, (list, tuple)):
        result = np.asarray(value, dtype=np.float64).reshape(tuple(shape))
    else:
        path_value = response.get("final_output_path")
        if not isinstance(path_value, str) or not path_value:
            raise NativeExecutionFailure("output_manifest_failed", "final output is missing")
        path = (root / path_value).resolve()
        try:
            path.relative_to(root.resolve())
        except ValueError as exc:
            raise NativeExecutionFailure("output_manifest_failed", "final output escapes run") from exc
        if not path.is_file():
            raise NativeExecutionFailure("result_transfer_failed", "final output file is missing")
        result = np.fromfile(path, dtype="<f4").astype(np.float64).reshape(tuple(shape))
    if not np.all(np.isfinite(result)):
        raise NativeExecutionFailure("output_validation_failed", "final output contains nonfinite values")
    return result


def _max_abs_error(actual: np.ndarray, expected: np.ndarray) -> float:
    if actual.shape != expected.shape:
        raise NativeExecutionFailure("output_validation_failed", "output shape mismatch")
    return float(np.max(np.abs(actual - expected))) if actual.size else 0.0


def _failure_stage(message: str) -> str:
    lowered = message.lower()
    if "timeout" in lowered:
        return "kernel_timeout"
    if "allocation" in lowered:
        return "hardware_allocation_failed"
    if "binary" in lowered or "load" in lowered:
        return "binary_load_failed"
    if "transfer" in lowered:
        return "result_transfer_failed"
    if "output" in lowered or "manifest" in lowered:
        return "output_manifest_failed"
    if "validation" in lowered:
        return "output_validation_failed"
    return "kernel_launch_failed"


def _identity(graph: TaskGraph) -> JsonDict:
    return {
        "circuit_semantics_hash": graph.circuit_semantics_hash,
        "tensor_network_hash": graph.tensor_network_hash,
        "contraction_plan_hash": graph.contraction_plan_hash,
    }


def _relative_path(path: Path, base: Path) -> str:
    return os.path.relpath(path.resolve(), base.resolve()).replace(os.sep, "/")


def _path_value(value: Mapping[str, Any], key: str) -> Path | None:
    item = value.get(key)
    return Path(item) if isinstance(item, (str, Path)) else None


def _binary_hash(path: Path | None) -> str | None:
    return _sha256_file(path) if path is not None and path.is_file() else None


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _unique_dir(parent: Path) -> Path:
    parent.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    candidate = parent / stamp
    index = 1
    while candidate.exists():
        candidate = parent / f"{stamp}_{index:02d}"
        index += 1
    candidate.mkdir()
    return candidate


__all__ = [
    "BACKEND_ID",
    "Block2NativeTarget",
    "CLAIM_BOUNDARY",
    "FAILURE_STAGES",
    "M45_CONTRACT",
    "PLACEMENT_FRONTIER",
    "PLACEMENT_SINGLE",
    "PROFILE_VERSION",
    "ROUTE_ID",
    "SCHEMA_VERSION",
    "SUITE_ID",
    "RouteStudyContract",
    "execute",
    "execute_route_study",
    "load_route_study_suite",
    "load_upmem_simplepim_taskgraph_suite",
    "prepare",
    "prepare_route_study",
]
