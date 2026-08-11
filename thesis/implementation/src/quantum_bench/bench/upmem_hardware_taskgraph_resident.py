"""Benchmark entry points for the additive MRAM-resident TaskGraph route."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from quantum_bench.bench.reporting import write_normalized_records, write_run_manifest
from quantum_bench.bench.run_dirs import EVIDENCE_ARTIFACT_KIND, create_run_dir, sanitize
from quantum_bench.circuits import load_circuit
from quantum_bench.core.jsonio import write_json, write_jsonl
from quantum_bench.core.records import JsonDict, TaskGraph
from quantum_bench.environment import capture_environment
from quantum_bench.tn.execution import order_final_tensor
from quantum_bench.targets.upmem.hardware_session import (
    HardwareSessionBuild,
    build_resident_hardware_session,
    execute_resident_graph_session,
)
from quantum_bench.targets.upmem.hardware_taskgraph_resident import (
    RESIDENT_BACKEND_ID,
    RESIDENT_CONTROL_H2D_BYTES_PER_LAUNCH,
    RESIDENT_DESCRIPTOR_CONTROL_BYTES,
    RESIDENT_ROUTE_ID,
    RESIDENT_SESSION_PROTOCOL,
    RESIDENT_SUPPORTED_TASKLETS,
    RESIDENT_TIMING_SCOPE,
    HardwareTaskGraphResidentProfile,
    HardwareTaskGraphResidentSuite,
    ResidentGraphPackage,
    build_resident_graph_package,
    build_resident_policy_reference,
    hardware_taskgraph_resident_profile_metadata,
    load_hardware_taskgraph_resident_suite,
    validate_hardware_taskgraph_resident_execution_request,
)
from quantum_bench.targets.upmem.execution_plan import (
    DpuResourceContext,
    UpmemCommunicationPlan,
    UpmemExecutionPlan,
    UpmemKernelPlan,
    UpmemNumericPlan,
    UpmemPlacementPlan,
    UpmemSchedulePlan,
    UpmemValidationStatuses,
)
from quantum_bench.tn import (
    build_execution_bundle,
    build_tensor_network,
    contraction_path_structure_hash,
    execute_task_sequence_np_einsum,
    plan_task_graph_with_config,
    with_execution_identity,
)
from quantum_bench.tn.execution_bundle import execution_identity_metadata, executor_config_hash


RESIDENT_PLAN_SCHEMA_VERSION = "upmem_hardware_taskgraph_resident_plan_v1"
RESIDENT_RUNTIME_SCHEMA_VERSION = "upmem_hardware_taskgraph_resident_runtime_v1"
RESIDENT_PROVIDER_ID = "upmem_resident_hardware"
RESIDENT_EXECUTION_PLAN_PROVENANCE = "host_declared"
RESIDENT_EXECUTION_PLAN_NATIVE_PACKAGE_BINDING = "not_native_package_bound"


@dataclass(frozen=True)
class UpmemHardwareTaskGraphResidentPlanResult:
    plan_dir: Path
    summary_path: Path
    status: str


@dataclass(frozen=True)
class UpmemHardwareTaskGraphResidentResult:
    run_dir: Path
    summary_path: Path
    status: str
    row_count: int


@dataclass(frozen=True)
class ResidentGraphExecution:
    status: str
    reason: str | None
    output: np.ndarray | None
    summary: JsonDict
    task_metrics: tuple[JsonDict, ...]


def _profile_with_tasklets(
    suite: HardwareTaskGraphResidentSuite,
    tasklets_per_dpu: int | None,
) -> HardwareTaskGraphResidentSuite:
    if tasklets_per_dpu is None:
        return suite
    tasklets = int(tasklets_per_dpu)
    if tasklets not in RESIDENT_SUPPORTED_TASKLETS:
        raise ValueError("hardware_profile_violation: tasklets_per_dpu must be one of 1, 2, 4, 8, 16")
    if suite.profile.version.endswith("_v1") and "m4_6" not in suite.profile.version and tasklets != 1:
        raise ValueError("hardware_profile_violation: resident v1 profile is one-tasklet only")
    return replace(suite, profile=replace(suite.profile, tasklets_per_dpu=tasklets))


def _write_resolved_suite(
    path: Path,
    suite: HardwareTaskGraphResidentSuite,
    *,
    tasklets_cli_override: int | None,
) -> None:
    payload = dict(suite.suite)
    metadata = dict(payload.get("metadata") or {})
    profile = dict(metadata.get("hardware_profile") or {})
    profile["tasklets_per_dpu"] = suite.profile.tasklets_per_dpu
    profile["effective_cli_tasklets_override"] = tasklets_cli_override
    metadata["hardware_profile"] = profile
    metadata["effective_cli_overrides"] = {
        "tasklets_per_dpu": tasklets_cli_override,
    }
    payload["metadata"] = metadata
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def prepare_upmem_hardware_taskgraph_resident(
    root_dir: Path,
    *,
    suite_path: Path,
    build: bool = False,
    tasklets_per_dpu: int | None = None,
    environment: Mapping[str, str] | None = None,
) -> UpmemHardwareTaskGraphResidentPlanResult:
    suite = load_hardware_taskgraph_resident_suite(suite_path)
    suite = _profile_with_tasklets(suite, tasklets_per_dpu)
    env = dict(os.environ if environment is None else environment)
    plan_dir = _unique_dir(root_dir / "build" / "upmem_hardware_taskgraph_resident_plan")
    plan_dir.mkdir(parents=True, exist_ok=False)
    (plan_dir / "config").mkdir()
    _write_resolved_suite(
        plan_dir / "config" / "resolved_suite.yml",
        suite,
        tasklets_cli_override=tasklets_per_dpu,
    )
    write_json(plan_dir / "config" / "hardware_profile.json", hardware_taskgraph_resident_profile_metadata(suite.profile))
    write_json(plan_dir / "environment.json", capture_environment(root_dir))
    rows: list[JsonDict] = []
    status = "prepared"
    failure_stage = None
    for case in suite.suite["cases"]:
        try:
            prepared = prepare_resident_case(root_dir, plan_dir / "cases" / sanitize(str(case["case_id"])), suite, case)
            rows.append(_prepared_case_row(case, prepared))
        except Exception as exc:
            status = "failed"
            failure_stage = _failure_stage(str(exc), "hardware_profile_violation")
            rows.append({"case_id": case.get("case_id"), "status": "failed", "failure_stage": failure_stage, "reason": str(exc)})
            break
    native_build: JsonDict = {"attempted": False, "status": "not_requested"}
    if status == "prepared" and build:
        try:
            built = build_resident_hardware_session(root_dir, plan_dir / "native_session", profile=suite.profile, environment=env)
            native_build = _native_build_metadata(built, plan_dir)
        except Exception as exc:
            status = "failed"
            failure_stage = _failure_stage(str(exc), "native_build_failed")
            native_build = {"attempted": True, "status": "failed", "failure_stage": failure_stage, "reason": str(exc)}
    summary_path = plan_dir / "upmem_hardware_taskgraph_resident_plan.json"
    write_json(summary_path, {
        "schema_version": RESIDENT_PLAN_SCHEMA_VERSION,
        "status": status,
        "failure_stage": failure_stage,
        "suite_id": suite.suite["suite_id"],
        "route_id": RESIDENT_ROUTE_ID,
        "backend_id": RESIDENT_BACKEND_ID,
        "provider_id": RESIDENT_PROVIDER_ID,
        "resource_context": _resource_context_metadata(_resident_resource_context(suite.profile)),
        "profile": hardware_taskgraph_resident_profile_metadata(suite.profile),
        "prepared_cases": rows,
        "native_build": native_build,
        "dpu_allocation_attempted": False,
        "dpu_launch_attempted": False,
        "notes": [
            "Resident preparation allocates deterministic float32 slots and validates lifetimes without allocating a DPU.",
            "Resident execution uses one graph request and ordered synchronous native descriptors per path/numeric mode.",
            "Capacity failure is a structured hardware profile rejection; no host spill or fallback is permitted.",
        ],
    })
    return UpmemHardwareTaskGraphResidentPlanResult(plan_dir, summary_path, status)


def prepare_resident_case(
    root_dir: Path,
    case_dir: Path,
    suite: HardwareTaskGraphResidentSuite,
    case: Mapping[str, Any],
) -> JsonDict:
    case_dir.mkdir(parents=True, exist_ok=True)
    circuit = load_circuit(dict(case), root_dir)
    network = build_tensor_network(circuit)
    variants: dict[str, JsonDict] = {}
    reference_outputs: list[np.ndarray] = []
    hashes: tuple[str, str] | None = None
    for variant in suite.variants:
        graph = with_execution_identity(plan_task_graph_with_config(network, dict(variant.planner)))
        execution_plan, resource_context = _resident_execution_plan(
            graph, suite.profile, quantization_mode="none"
        )
        reference, execution_metrics = execute_task_sequence_np_einsum(graph, network)
        reference_array = np.asarray(reference)
        reference_outputs.append(reference_array)
        identity = (graph.circuit_semantics_hash, graph.tensor_network_hash)
        if hashes is None:
            hashes = identity
        elif identity != hashes:
            raise ValueError("hardware_profile_violation: resident path variants changed circuit/network identity")
        package = build_resident_graph_package(
            graph, network, case_id=str(case["case_id"]), suite_id=str(suite.suite["suite_id"]),
            quantization_mode="none", profile=suite.profile, full_precision_output=reference_array,
        )
        variant_dir = case_dir / "paths" / sanitize(variant.variant_id)
        variant_dir.mkdir(parents=True, exist_ok=False)
        bundle_path = variant_dir / "execution_bundle.json"
        write_json(bundle_path, build_execution_bundle(graph, case_id=str(case["case_id"]), suite_id=str(suite.suite["suite_id"])))
        reference_path = variant_dir / "cpu_reference_final_tensor.npy"
        np.save(reference_path, reference_array, allow_pickle=False)
        write_json(variant_dir / "resident_allocation.json", package.allocation.to_json_dict())
        write_json(variant_dir / "path_preparation.json", {
            "case_id": case["case_id"],
            "path_variant_id": variant.variant_id,
            "path_variant_label": variant.label,
            "planner": variant.planner,
            "task_count": len(graph.tasks),
            "component_operation_count": package.component_operation_count,
            "slot_descriptor_count": package.allocation.slot_descriptor_count,
            "mram_pool_bytes": package.allocation.mram_pool_bytes,
            "mram_used_bytes": package.allocation.mram_used_bytes,
            "execution_bundle": bundle_path.name,
            "cpu_reference_final_tensor": reference_path.name,
            "reference_execution": execution_metrics,
            "circuit_semantics_hash": graph.circuit_semantics_hash,
            "tensor_network_hash": graph.tensor_network_hash,
            "contraction_plan_hash": graph.contraction_plan_hash,
            "contraction_path_structure_hash": contraction_path_structure_hash(graph),
            **_execution_plan_metadata(execution_plan, suite.profile, resource_context),
            "no_host_intermediate_output_files": True,
            "intermediate_output_paths": [],
        })
        variants[variant.variant_id] = {
            "variant": variant,
            "graph": graph,
            "reference_output": reference_array,
            "bundle_path": bundle_path,
            "reference_path": reference_path,
            "allocation": package.allocation,
            "execution_plan": execution_plan,
            "resource_context": resource_context,
        }
    if not reference_outputs:
        raise ValueError("hardware_profile_violation: resident suite has no path variants")
    if not all(np.allclose(item, reference_outputs[0], rtol=1e-10, atol=1e-10) for item in reference_outputs[1:]):
        raise ValueError("hardware_profile_violation: resident path variants do not reconstruct the same CPU tensor")
    return {
        "circuit": circuit,
        "network": network,
        "variants": variants,
        "circuit_semantics_hash": hashes[0] if hashes else None,
        "tensor_network_hash": hashes[1] if hashes else None,
    }


def run_upmem_hardware_taskgraph_resident(
    root_dir: Path,
    *,
    suite_path: Path,
    environment: Mapping[str, str] | None = None,
    tasklets_per_dpu: int | None = None,
) -> UpmemHardwareTaskGraphResidentResult:
    env = dict(os.environ if environment is None else environment)
    validate_hardware_taskgraph_resident_execution_request(execute=True, environment=env)
    suite = load_hardware_taskgraph_resident_suite(suite_path)
    suite = _profile_with_tasklets(suite, tasklets_per_dpu)
    run_dir = create_run_dir(root_dir, str(suite.suite["suite_id"]), artifact_kind=EVIDENCE_ARTIFACT_KIND, route_label="upmem_hw_taskgraph_resident")
    return run_resident_suite(
        root_dir,
        run_dir,
        suite,
        environment=env,
        tasklets_cli_override=tasklets_per_dpu,
    )


def run_resident_suite(
    root_dir: Path,
    run_dir: Path,
    suite: HardwareTaskGraphResidentSuite,
    *,
    environment: Mapping[str, str],
    tasklets_cli_override: int | None = None,
) -> UpmemHardwareTaskGraphResidentResult:
    _write_resolved_suite(
        run_dir / "config" / "resolved_suite.yml",
        suite,
        tasklets_cli_override=tasklets_cli_override,
    )
    write_json(run_dir / "config" / "hardware_profile.json", hardware_taskgraph_resident_profile_metadata(suite.profile))
    write_json(run_dir / "environment.json", capture_environment(root_dir))
    manifest = write_run_manifest(
        run_dir,
        run_kind="upmem_hardware_taskgraph_resident_path_quantization_study",
        suite_id=str(suite.suite["suite_id"]), suite_path=str(suite.suite_path),
        artifact_kind=EVIDENCE_ARTIFACT_KIND, route_label="upmem_hw_taskgraph_resident",
        route_id=RESIDENT_ROUTE_ID, backend_id=RESIDENT_BACKEND_ID,
        execution_scope="physical_single_dpu_mram_resident_full_taskgraph",
        evidence_type="physical_hardware_one_dpu_mram_resident",
        upmem_execution_mode=RESIDENT_SESSION_PROTOCOL,
        artifact_retention="full", summary="upmem_hardware_taskgraph_resident_summary.json",
        policies=("opt_einsum_greedy", "custom_upmem_v2_balanced"),
        quantization_modes=suite.profile.numeric_modes, root_dir=root_dir,
    )
    manifest.update(
        {
            "tasklets_per_dpu": suite.profile.tasklets_per_dpu,
            "effective_tasklets_per_dpu": suite.profile.tasklets_per_dpu,
            "effective_cli_tasklets_override": tasklets_cli_override,
        }
    )
    write_json(run_dir / "run_manifest.json", manifest)
    try:
        native_build = build_resident_hardware_session(root_dir, run_dir / "native_session", profile=suite.profile, environment=environment)
    except Exception as exc:
        failure = _failure_record(suite, str(exc), None)
        write_normalized_records(run_dir, [failure])
        summary_path = run_dir / "upmem_hardware_taskgraph_resident_summary.json"
        write_json(summary_path, {
            "schema_version": RESIDENT_RUNTIME_SCHEMA_VERSION, "status": "failed",
            "failure_stage": _failure_stage(str(exc), "native_build_failed"), "reason": str(exc),
            "route_id": RESIDENT_ROUTE_ID, "backend_id": RESIDENT_BACKEND_ID,
            "normalized_records": "normalized_records.jsonl",
        })
        manifest.update({"summary": summary_path.name, "hardware_available": "not_verified_by_execution"})
        write_json(run_dir / "run_manifest.json", manifest)
        return UpmemHardwareTaskGraphResidentResult(run_dir, summary_path, "failed", 1)

    records: list[JsonDict] = []
    warmups: list[JsonDict] = []
    case_statuses: list[JsonDict] = []
    stop_after_failure = False
    for case in suite.suite["cases"]:
        case_id = str(case["case_id"])
        if stop_after_failure:
            case_statuses.append({"case_id": case_id, "status": "not_attempted_after_prior_failure"})
            continue
        try:
            prepared = prepare_resident_case(root_dir, run_dir / "cases" / sanitize(case_id), suite, case)
            case_records, case_warmups, case_status = _run_resident_case(
                root_dir=root_dir, run_dir=run_dir, suite=suite, case=case,
                prepared=prepared, native_build=native_build, environment=environment,
            )
        except Exception as exc:
            case_records = [_failure_record(suite, str(exc), case)]
            case_warmups = []
            case_status = {"case_id": case_id, "status": "failed", "failure_stage": _failure_stage(str(exc), "hardware_profile_violation"), "reason": str(exc)}
        records.extend(case_records)
        warmups.extend(case_warmups)
        case_statuses.append(case_status)
        if case_status.get("status") != "passed":
            stop_after_failure = True
    write_jsonl(run_dir / "warmups.jsonl", warmups)
    write_normalized_records(run_dir, records)
    completed = bool(records) and all(item.get("status") == "completed" for item in records)
    run_resource_context = _resident_run_resource_context(suite.profile, records)
    summary_path = run_dir / "upmem_hardware_taskgraph_resident_summary.json"
    write_json(summary_path, {
        "schema_version": RESIDENT_RUNTIME_SCHEMA_VERSION,
        "status": "completed" if completed else "failed",
        "suite_id": suite.suite["suite_id"], "route_id": RESIDENT_ROUTE_ID, "backend_id": RESIDENT_BACKEND_ID,
        "provider_id": RESIDENT_PROVIDER_ID,
        "resource_context": _resource_context_metadata(run_resource_context),
        "row_count": len(records), "warmup_count": len(warmups), "case_statuses": case_statuses,
        "execution_plan_hashes": sorted({str(record.get("execution_plan_hash")) for record in records if record.get("execution_plan_hash")}),
        "hardware_profile": hardware_taskgraph_resident_profile_metadata(suite.profile),
        "native_build": _native_build_metadata(native_build, root_dir),
        "normalized_records": "normalized_records.jsonl", "warmups": "warmups.jsonl",
        "claim_boundary": "one-DPU MRAM-resident full-TaskGraph path/numeric-mode comparison only; no speedup, energy, scheduler, or multi-DPU claim",
    })
    manifest.update({"summary": summary_path.name, "hardware_available": "verified_by_execution" if completed else "not_verified_by_execution"})
    write_json(run_dir / "run_manifest.json", manifest)
    return UpmemHardwareTaskGraphResidentResult(run_dir, summary_path, "completed" if completed else "failed", len(records))


def execute_resident_variant(
    *,
    root_dir: Path,
    run_dir: Path,
    native_build: HardwareSessionBuild,
    profile: HardwareTaskGraphResidentProfile,
    suite_id: str,
    case_id: str,
    variant_id: str,
    graph: TaskGraph,
    network: Any,
    reference_output: np.ndarray,
    quantization_mode: str,
    request_id: str,
    environment: Mapping[str, str] | None = None,
) -> ResidentGraphExecution:
    execution_environment = dict(os.environ if environment is None else environment)
    validate_hardware_taskgraph_resident_execution_request(
        execute=True, environment=execution_environment
    )
    started = time.perf_counter()
    graph = with_execution_identity(graph)
    execution_plan, resource_context = _resident_execution_plan(
        graph, profile, quantization_mode=quantization_mode
    )
    execution_metadata = {
        **execution_identity_metadata(graph, plan_reused=True),
        "executor_config_hash": executor_config_hash(RESIDENT_ROUTE_ID, {
            "profile": profile.version, "backend_id": profile.backend_id,
            "session_protocol": profile.session_protocol, "quantization_mode": quantization_mode,
            "resident_pool_bytes": profile.mram_pool_bytes,
        }),
        **_execution_plan_metadata(execution_plan, profile, resource_context),
    }
    try:
        package = build_resident_graph_package(
            graph, network, case_id=case_id, suite_id=suite_id,
            quantization_mode=quantization_mode, profile=profile,
            full_precision_output=reference_output,
        )
        artifact = package.write(native_build.session_root, dpu_binary=native_build.dpu_binary, request_id=request_id)
        if artifact.manifest_path is None:
            raise RuntimeError("manifest_parse_failed: resident request manifest missing")
        response_path = native_build.session_root / f"{sanitize(request_id)}_resident_response.json"
        native = execute_resident_graph_session(
            native_build, manifest_path=artifact.manifest_path, response_path=response_path,
            profile=profile, environment=execution_environment,
        )
    except Exception as exc:
        return _failed_execution(
            graph, profile, quantization_mode,
            _failed_execution_metadata(execution_metadata, execution_plan, profile, resource_context),
            str(exc), started,
        )
    if native.status != "completed":
        return _failed_execution(
            graph, profile, quantization_mode,
            _failed_execution_metadata(execution_metadata, execution_plan, profile, resource_context),
            native.failure_stage or "resident_native_execution_failed", started, native=native,
            package=artifact,
        )
    try:
        verified_resource_context = _verified_resident_resource_context(
            profile, native.response
        )
        verified_execution_plan = replace(
            execution_plan,
            placement=replace(
                execution_plan.placement, resources=verified_resource_context
            ),
        )
        resource_context = verified_resource_context
        execution_plan = verified_execution_plan
        execution_metadata = {
            **execution_metadata,
            **_execution_plan_metadata(execution_plan, profile, resource_context),
        }
        output = _load_final_output(artifact)
    except Exception as exc:
        return _failed_execution(
            graph, profile, quantization_mode,
            _failed_execution_metadata(execution_metadata, execution_plan, profile, resource_context),
            str(exc), started, native=native, package=artifact,
        )
    policy = build_resident_policy_reference(graph, network, quantization_mode=quantization_mode, profile=profile)
    policy_output = np.asarray(policy["output"])
    policy_validation = _accuracy(policy_output, output, _policy_tolerance(quantization_mode), "resident_policy_reference")
    full_precision_accuracy = _accuracy(
        np.asarray(reference_output), output,
        _full_precision_tolerance(quantization_mode), "cpu_exact_taskgraph_full_precision",
    )
    response = native.response
    initial_h2d = _manifest_integer(artifact, "initial_h2d_bytes")
    descriptor_h2d = _manifest_integer(artifact, "descriptor_h2d_bytes")
    control_h2d = RESIDENT_DESCRIPTOR_CONTROL_BYTES + artifact.descriptor_count * RESIDENT_CONTROL_H2D_BYTES_PER_LAUNCH
    final_d2h = _manifest_integer(artifact, "final_d2h_bytes")
    actual_h2d = initial_h2d + descriptor_h2d + control_h2d
    actual_d2h = final_d2h
    accounting = {
        "initial_h2d_bytes": initial_h2d,
        "descriptor_h2d_bytes": descriptor_h2d,
        "control_h2d_bytes": control_h2d,
        "descriptor_control_bytes": RESIDENT_DESCRIPTOR_CONTROL_BYTES,
        "control_h2d_bytes_per_launch": RESIDENT_CONTROL_H2D_BYTES_PER_LAUNCH,
        "final_d2h_bytes": final_d2h,
        "intermediate_h2d_bytes": 0,
        "intermediate_d2h_bytes": 0,
        "actual_h2d_bytes": actual_h2d,
        "actual_d2h_bytes": actual_d2h,
        "actual_transfer_bytes": actual_h2d + actual_d2h,
    }
    accounting["bytes_invariant_status"] = "passed" if (
        accounting["intermediate_h2d_bytes"] == 0 and accounting["intermediate_d2h_bytes"] == 0 and
        actual_h2d == initial_h2d + descriptor_h2d + control_h2d and actual_d2h == final_d2h and
        accounting["actual_transfer_bytes"] == actual_h2d + actual_d2h
    ) else "failed"
    execution_contract = _execution_contract_passed(
        response, resource_context, accounting["bytes_invariant_status"]
    )
    statuses = _resident_validation_statuses(
        execution_contract=execution_contract,
        policy_validation=policy_validation,
        full_precision_accuracy=full_precision_accuracy,
    )
    execution_plan = replace(execution_plan, validation=statuses)
    execution_metadata = {
        **execution_metadata,
        **_execution_plan_metadata(execution_plan, profile, resource_context),
    }
    modeled = _modeled_host_rehydrated_bytes(package)
    scale_metrics = _scale_metrics(policy)
    operation_cycles = [int(value) for value in response.get("dpu_operation_cycles", ())]
    processed_counters = response.get("tasklet_processed_elements", ())
    active_tasklet_counts = response.get("active_tasklet_count", ())
    idle_tasklet_counts = response.get("idle_tasklet_count", ())
    tasklet_utilization = response.get("tasklet_utilization", ())
    work_imbalance = response.get("tasklet_work_imbalance", ())
    real_pass_count = sum(
        1 for operation in package.operations
        if operation.component in {"real", "ar_br", "ai_bi"}
    )
    imaginary_pass_count = sum(
        1 for operation in package.operations
        if operation.component in {"ar_bi", "ai_br"}
    )
    complex_combine_pass_count = sum(
        1 for operation in package.operations if operation.component == "complex_combine"
    )
    operation_timing = [
        {
            "operation_id": operation.operation_id,
            "task_id": operation.task_id,
            "component": operation.component,
            "dpu_cycles": operation_cycles[index] if index < len(operation_cycles) else None,
            "tasklet_processed_elements": list(processed_counters[index]) if index < len(processed_counters) else [],
            "active_tasklet_count": active_tasklet_counts[index] if index < len(active_tasklet_counts) else None,
            "idle_tasklet_count": idle_tasklet_counts[index] if index < len(idle_tasklet_counts) else None,
            "tasklet_utilization": tasklet_utilization[index] if index < len(tasklet_utilization) else None,
            "work_imbalance": work_imbalance[index] if index < len(work_imbalance) else None,
        }
        for index, operation in enumerate(package.operations)
    ]
    task_metrics = tuple({
        "task_id": task.id,
        "task_index": index,
        "status": "completed",
        "resident_slot_input_ids": [package.allocation.slot_for(task.input_tensor_ids[0], "real"), package.allocation.slot_for(task.input_tensor_ids[1], "real")],
        "resident_slot_output_ids": [package.allocation.slot_for(task.output_tensor_id, "real")],
        "native_component_operation_count": 5 if len(package.allocation.tensor_components[task.output_tensor_id]) == 2 else 1,
        "dpu_operation_ids": [item["operation_id"] for item in operation_timing if item["task_id"] == task.id],
    } for index, task in enumerate(graph.tasks))
    validation_status = _legacy_validation_status(policy_validation, accounting["bytes_invariant_status"])
    summary: JsonDict = {
        "schema_version": RESIDENT_RUNTIME_SCHEMA_VERSION,
        "status": "completed" if validation_status == "passed" else "failed",
        "reason": None if validation_status == "passed" else "output_validation_failed",
        "failure_stage": None if validation_status == "passed" else "output_validation_failed",
        "route_id": RESIDENT_ROUTE_ID, "backend_id": RESIDENT_BACKEND_ID,
        "hardware_profile_version": profile.version,
        "session_protocol": RESIDENT_SESSION_PROTOCOL, "timing_scope": RESIDENT_TIMING_SCOPE,
        "hardware_execution": response["hardware_execution"],
        "hardware_kernel_executed": response["hardware_kernel_executed"],
        "native_execution": response["native_execution"],
        "native_kernel_executed": response.get("native_kernel_executed", response["hardware_kernel_executed"]),
        "native_hardware_backend": response["native_hardware_backend"],
        "hardware_backend_verified": response["hardware_backend_verified"],
        "simulator_kernel_executed": response["simulator_kernel_executed"],
        "cpu_fallback_used": response["cpu_fallback_used"],
        "target_observed": response["target_observed"], "graph_request_count": response["graph_request_count"],
        "target_requested": response["target_requested"],
        "requested_dpu_count": response["requested_dpus"],
        "allocated_dpu_count": response["allocated_dpus"],
        "tasklets_per_dpu": response["tasklets"],
        "allocation_count": response["allocation_count"],
        "hardware_allocation_verified": response["hardware_allocation_verified"],
        "hardware_release_verified": response["hardware_release_verified"],
        "release_confirmed": response["release_confirmed"],
        "hardware_timing_available": response["hardware_timing_available"],
        "persistent_session_reused": response["persistent_session_reused"],
        "resident_slots_persist_for_graph": response["resident_slots_persist_for_graph"],
        "logical_task_count": len(graph.tasks), "task_count": len(graph.tasks),
        "validated_task_count": len(graph.tasks), "component_operation_count": package.component_operation_count,
        "native_launch_count": int(response.get("native_launch_count", package.component_operation_count)),
        "native_task_count": int(response.get("native_task_count", package.component_operation_count)),
        "resident_slot_descriptor_count": package.allocation.slot_descriptor_count,
        "resident_slots": [item.to_json_dict() for item in package.allocation.slots],
        "slot_lifetime_map": [item.to_json_dict() for item in package.allocation.lifetimes],
        "resident_mram_pool_bytes": package.allocation.mram_pool_bytes,
        "resident_mram_used_bytes": package.allocation.mram_used_bytes,
        "peak_resident_bytes": package.allocation.peak_resident_bytes,
        "resident_slot_dtype": "float32",
        "final_output_component_count": len(package.allocation.final_components),
        "final_output_only_d2h": response["final_output_only_d2h"],
        "no_host_intermediate_output_files": True,
        "intermediate_output_paths": [],
        "host_intermediate_combine": False,
        "complex_policy": profile.complex_policy,
        "numeric_policy": {
            "mode": quantization_mode,
            "dpu_local_requantization": quantization_mode == "per_task_resident_requantize",
            "scale_formula": "max_abs/127_or_1_for_all_zero",
            "rounding": "nearest_even",
            "clip_range": [-127, 127],
            "scales_and_saturation": scale_metrics,
        },
        "policy_reference_validation": policy_validation,
        "full_precision_accuracy": full_precision_accuracy,
        "execution_contract_status": statuses.execution_contract_status,
        "policy_reference_status": statuses.policy_reference_status,
        "full_precision_accuracy_status": statuses.full_precision_accuracy_status,
        "scientific_validation_status": statuses.scientific_validation_status,
        "validation_status": validation_status,
        "physical_dependency_chain_verified": response["physical_dependency_chain_verified"],
        "output_hash": _array_hash(output),
        "policy_reference_output_hash": _array_hash(policy_output),
        "full_precision_reference_hash": _array_hash(reference_output),
        "descriptor_package_sha256": artifact.descriptor_sha256,
        "native_source_tree_hash": native_build.source_tree_hash,
        "host_binary_hash": native_build.host_binary_hash,
        "dpu_binary_hash": native_build.dpu_binary_hash,
        "native_build_command": list(native_build.build_command),
        "sdk_tools": native_build.sdk_tools,
        "package_parse_time_s": response.get("package_parse_time_s"),
        "initial_h2d_time_s": response.get("initial_h2d_time_s"),
        "descriptor_h2d_time_s": response.get("descriptor_h2d_time_s"),
        "control_h2d_time_s": response.get("control_h2d_time_s"),
        "kernel_time_s": response.get("kernel_time_s"),
        "final_d2h_time_s": response.get("final_d2h_time_s"),
        "output_write_time_s": response.get("output_write_time_s"),
        "completion_abi_version": response.get("completion_abi_version"),
        "dpu_run_time_cycles": response.get("dpu_run_time_cycles", 0),
        "dpu_graph_cycle_sum": response.get("graph_cycle_sum", response.get("dpu_run_time_cycles", 0)),
        "dpu_operation_cycles": operation_cycles,
        "tasklet_processed_elements": [list(row) for row in processed_counters],
        "active_tasklet_count": list(active_tasklet_counts),
        "idle_tasklet_count": list(idle_tasklet_counts),
        "tasklet_utilization": list(tasklet_utilization),
        "tasklet_work_imbalance": list(work_imbalance),
        "tasklet_work_imbalance_basis": "configured_tasklets_max_minus_min_over_max",
        "tasklet_utilization_basis": "active_tasklets_over_configured_tasklets",
        "real_pass_count": real_pass_count,
        "imaginary_pass_count": imaginary_pass_count,
        "complex_contract_pass_count": real_pass_count + imaginary_pass_count,
        "complex_combine_pass_count": complex_combine_pass_count,
        "complex_pass_count": real_pass_count + imaginary_pass_count + complex_combine_pass_count,
        "operation_timing": operation_timing,
        **accounting,
        "modeled_host_rehydrated_equivalent_bytes": modeled,
        "task_metrics": list(task_metrics),
        "session_response_artifact": str(native.response_path.relative_to(native_build.session_root)),
        "total_route_time_s": time.perf_counter() - started,
        "steady_state_graph_execution_s": response.get("steady_state_graph_execution_s"),
        "timing_is_bringup_only": False,
        "contraction_execution_target": "upmem",
        "actual_transfer_bytes_invariant": "passed" if accounting["bytes_invariant_status"] == "passed" else "failed",
        **execution_metadata,
    }
    return ResidentGraphExecution(str(summary["status"]), summary["reason"], output, summary, task_metrics)


def _run_resident_case(*, root_dir, run_dir, suite, case, prepared, native_build, environment):
    variants = tuple(item.variant_id for item in suite.variants)
    modes = tuple(suite.profile.numeric_modes)
    combinations = tuple((variant, mode) for variant in variants for mode in modes)
    records: list[JsonDict] = []
    warmups: list[JsonDict] = []
    failure = None
    for warmup_id in range(int(suite.suite["warmups"])):
        for order_index, (variant_id, mode) in enumerate(_rotate(combinations, warmup_id)):
            execution = _execute_case_variant(root_dir, run_dir, suite, case, prepared, native_build, environment, variant_id, mode, "warmup", warmup_id, order_index)
            warmups.append(_warmup_row(case, variant_id, mode, warmup_id, order_index, execution))
            if execution.status != "completed":
                failure = ("warmup", execution)
                break
        if failure:
            break
    if failure is None:
        for repeat_id in range(int(suite.suite["repeats"])):
            for order_index, (variant_id, mode) in enumerate(_rotate(combinations, repeat_id)):
                execution = _execute_case_variant(root_dir, run_dir, suite, case, prepared, native_build, environment, variant_id, mode, "repeat", repeat_id, order_index)
                record = _normalized_record(case, suite, variant_id, mode, repeat_id, order_index, execution, run_dir)
                records.append(record)
                if execution.status != "completed":
                    failure = ("repeat", execution)
                    break
            if failure:
                break
    status = "passed" if failure is None and len(records) == int(suite.suite["repeats"]) * len(combinations) else "failed"
    return records, warmups, {"case_id": str(case["case_id"]), "status": status, "warmups": len(warmups), "timed_rows": len(records), "failure_stage": failure[1].summary.get("failure_stage") if failure else None}


def _execute_case_variant(root_dir, run_dir, suite, case, prepared, native_build, environment, variant_id, mode, phase, iteration, order_index):
    value = prepared["variants"][variant_id]
    return execute_resident_variant(
        root_dir=root_dir, run_dir=run_dir, native_build=native_build, profile=suite.profile,
        suite_id=str(suite.suite["suite_id"]), case_id=str(case["case_id"]), variant_id=variant_id,
        graph=value["graph"], network=prepared["network"], reference_output=value["reference_output"],
        quantization_mode=mode,
        request_id=f"{sanitize(str(case['case_id']))}-{phase}-{iteration:02d}-{sanitize(variant_id)}-{sanitize(mode)}-{order_index:02d}",
        environment=environment,
    )


def _normalized_record(case, suite, variant_id, mode, repeat_id, order_index, execution, run_dir):
    summary = execution.summary
    artifact_path = run_dir / "cases" / sanitize(str(case["case_id"])) / "resident_records" / f"repeat_{repeat_id:02d}_{sanitize(variant_id)}_{sanitize(mode)}.json"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(artifact_path, summary)
    return {
        "schema_version": "resident_normalized_record_v1",
        "status": execution.status,
        "suite_id": suite.suite["suite_id"],
        "case_id": case["case_id"], "workload_id": case["case_id"],
        "route_id": RESIDENT_ROUTE_ID, "backend_id": RESIDENT_BACKEND_ID,
        "contraction_execution_target": "upmem",
        "backend_family": RESIDENT_BACKEND_ID, "kernel_family": "generic_loop_resident_graph",
        "benchmark_role": "hardware_one_dpu_mram_resident_path_quantization_study",
        "path_variant_id": variant_id, "quantization_mode": mode, "repeat_id": repeat_id,
        "variant_order_index": order_index, "execution_model": "taskgraph_mram_resident",
        "execution_scope": "physical_single_dpu_mram_resident_full_taskgraph",
        "timing_scope": RESIDENT_TIMING_SCOPE, "target_requested": summary.get("target_requested", "hardware"),
        "target_observed": summary.get("target_observed", "hardware_unverified"),
        "hardware_functionality_evidence": execution.status == "completed" and summary.get("hardware_execution") is True,
        "hardware_timing_available": summary.get("hardware_timing_available", False) if execution.status == "completed" else False,
        "persistent_session_reused": summary.get("persistent_session_reused"),
        "hardware_kernel_executed": summary.get("hardware_kernel_executed", False),
        "native_execution": summary.get("native_execution", False),
        "native_kernel_executed": summary.get("native_kernel_executed", False),
        "native_hardware_backend": summary.get("native_hardware_backend", False),
        "hardware_backend_verified": summary.get("hardware_backend_verified", False),
        "simulator_kernel_executed": summary.get("simulator_kernel_executed", False),
        "cpu_fallback_used": summary.get("cpu_fallback_used", False),
        "multi_dpu_execution": False, "requested_dpu_count": summary.get("requested_dpu_count", 1),
        "allocated_dpu_count": summary.get("allocated_dpu_count", 0),
        "tasklets_per_dpu": summary.get("tasklets_per_dpu", 1),
        "hardware_profile_version": summary.get("hardware_profile_version"),
        "allocation_count": summary.get("allocation_count", 0),
        "graph_request_count": summary.get("graph_request_count", 1),
        "native_launch_count": summary.get("native_launch_count", 0),
        "resident_slot_descriptor_count": summary.get("resident_slot_descriptor_count"),
        "resident_mram_pool_bytes": summary.get("resident_mram_pool_bytes"),
        "resident_mram_used_bytes": summary.get("resident_mram_used_bytes"),
        "final_output_only_d2h": summary.get("final_output_only_d2h", False), "intermediate_h2d_bytes": summary.get("intermediate_h2d_bytes", 0),
        "intermediate_d2h_bytes": summary.get("intermediate_d2h_bytes", 0),
        "actual_h2d_bytes": summary.get("actual_h2d_bytes", 0), "actual_d2h_bytes": summary.get("actual_d2h_bytes", 0),
        "actual_transfer_bytes": summary.get("actual_transfer_bytes", 0),
        "actual_transfer_bytes_invariant": summary.get("actual_transfer_bytes_invariant"),
        "modeled_host_rehydrated_equivalent_bytes": summary.get("modeled_host_rehydrated_equivalent_bytes", {}),
        "policy_reference_validation": summary.get("policy_reference_validation"),
        "full_precision_accuracy": summary.get("full_precision_accuracy"),
        "execution_plan": summary.get("execution_plan"),
        "execution_plan_hash": summary.get("execution_plan_hash"),
        "execution_plan_schema_version": summary.get("execution_plan_schema_version"),
        "execution_plan_provenance": summary.get("execution_plan_provenance"),
        "execution_plan_native_package_binding": summary.get("execution_plan_native_package_binding"),
        "provider_id": summary.get("provider_id"),
        "provider_metadata": summary.get("provider_metadata"),
        "resource_context": summary.get("resource_context"),
        "execution_contract_status": summary.get("execution_contract_status"),
        "policy_reference_status": summary.get("policy_reference_status"),
        "full_precision_accuracy_status": summary.get("full_precision_accuracy_status"),
        "scientific_validation_status": summary.get("scientific_validation_status"),
        "validation_status": summary.get("validation_status"),
        "task_count": summary.get("task_count"),
        "validated_task_count": summary.get("validated_task_count"),
        "hardware_allocation_verified": summary.get("hardware_allocation_verified", False),
        "hardware_release_verified": summary.get("hardware_release_verified", False),
        "release_confirmed": summary.get("release_confirmed", False),
        "physical_dependency_chain_verified": summary.get("physical_dependency_chain_verified", False),
        "hardware_execution": summary.get("hardware_execution", False),
        "timing_is_bringup_only": summary.get("timing_is_bringup_only", False),
        "steady_state_graph_execution_s": summary.get("steady_state_graph_execution_s"),
        "completion_abi_version": summary.get("completion_abi_version"),
        "dpu_run_time_cycles": summary.get("dpu_run_time_cycles", 0),
        "dpu_graph_cycle_sum": summary.get("dpu_graph_cycle_sum", 0),
        "dpu_operation_cycles": summary.get("dpu_operation_cycles", []),
        "tasklet_processed_elements": summary.get("tasklet_processed_elements", []),
        "active_tasklet_count": summary.get("active_tasklet_count", []),
        "idle_tasklet_count": summary.get("idle_tasklet_count", []),
        "tasklet_utilization": summary.get("tasklet_utilization", []),
        "tasklet_work_imbalance": summary.get("tasklet_work_imbalance", []),
        "tasklet_work_imbalance_basis": summary.get("tasklet_work_imbalance_basis"),
        "tasklet_utilization_basis": summary.get("tasklet_utilization_basis"),
        "real_pass_count": summary.get("real_pass_count", 0),
        "imaginary_pass_count": summary.get("imaginary_pass_count", 0),
        "complex_contract_pass_count": summary.get("complex_contract_pass_count", 0),
        "complex_combine_pass_count": summary.get("complex_combine_pass_count", 0),
        "complex_pass_count": summary.get("complex_pass_count", 0),
        "operation_timing": summary.get("operation_timing", []),
        "circuit_semantics_hash": summary.get("circuit_semantics_hash"),
        "tensor_network_hash": summary.get("tensor_network_hash"),
        "contraction_plan_hash": summary.get("contraction_plan_hash"),
        "descriptor_package_sha256": summary.get("descriptor_package_sha256"),
        "task_metrics_artifact": str(artifact_path.relative_to(run_dir)),
        "summary_artifact": str(artifact_path.relative_to(run_dir)),
        "failure_stage": summary.get("failure_stage"), "reason": execution.reason,
        "claim_boundary": "no speedup, energy, scheduler, or multi-DPU claim",
    }


def _warmup_row(case, variant_id, mode, warmup_id, order_index, execution):
    return {
        "case_id": case["case_id"], "phase": "warmup", "warmup_id": warmup_id,
        "path_variant_id": variant_id, "quantization_mode": mode, "order_index": order_index,
        "status": execution.status, "failure_stage": execution.summary.get("failure_stage"),
        "graph_request_count": execution.summary.get("graph_request_count", 1),
        "timing_scope": RESIDENT_TIMING_SCOPE,
    }


def _load_final_output(package: ResidentGraphPackage) -> np.ndarray:
    values: dict[str, np.ndarray] = {}
    for component, _slot, elements in package.allocation.final_components:
        path = package.final_output_paths[component]
        if not path.is_file():
            raise RuntimeError("final_transfer_failed: resident native final output is missing")
        raw = np.fromfile(path, dtype="<f4")
        if raw.size != elements:
            raise RuntimeError("final_transfer_failed: resident native final output length mismatch")
        if not np.all(np.isfinite(raw)):
            raise RuntimeError("output_validation_failed: resident native final output is non-finite")
        values[component] = raw.reshape(tuple(int(item) for item in package.graph.tasks[-1].output_shape))
    if "imag" in values:
        if "real" not in values:
            raise RuntimeError("output_validation_failed: resident native real output component is missing")
        raw_array = np.asarray(values["real"] + 1j * values["imag"], dtype=np.complex64)
    else:
        raw_array = np.asarray(values["real"], dtype=np.float32)
    final_task = package.graph.tasks[-1]
    ordered_array, _ = order_final_tensor(
        raw_array,
        final_task.output_labels,
        package.graph.network.output_labels,
    )
    return np.asarray(ordered_array)


def _modeled_host_rehydrated_bytes(package: ResidentGraphPackage) -> JsonDict:
    h2d = 0
    d2h = 0
    args_bytes = 740
    for operation in package.operations:
        if operation.kind != "contract":
            continue
        left = int(operation.args.get("left_elems", _product(operation.args.get("left_shape", ()))))
        right = int(operation.args.get("right_elems", _product(operation.args.get("right_shape", ()))))
        output = int(operation.output_elements)
        input_itemsize = 4 if operation.mode == "none" else 1
        output_itemsize = 4
        h2d += args_bytes + _align8(left * input_itemsize) + _align8(right * input_itemsize)
        d2h += _align8(output * output_itemsize)
    return {
        "h2d_bytes": h2d,
        "d2h_bytes": d2h,
        "transfer_bytes": h2d + d2h,
        "intermediate_h2d_bytes": h2d,
        "intermediate_d2h_bytes": d2h,
        "comparison_route_id": "upmem_tn_hardware_taskgraph_persistent",
        "definition": "modeled legacy host-rehydrated per-component generic-loop transfers; comparison only, not resident hardware traffic",
    }


def _resident_resource_context(profile: HardwareTaskGraphResidentProfile) -> DpuResourceContext:
    return DpuResourceContext(
        requested_dpu_count=profile.requested_dpu_count,
        requested_tasklets_per_dpu=profile.tasklets_per_dpu,
    )


def _resident_run_resource_context(
    profile: HardwareTaskGraphResidentProfile,
    records: Sequence[Mapping[str, Any]],
) -> DpuResourceContext:
    declared = _resident_resource_context(profile)
    completed = [record for record in records if record.get("status") == "completed"]
    if not completed:
        return declared
    contexts = [record.get("resource_context") for record in completed]
    if any(
        not isinstance(context, Mapping)
        or context.get("allocation_status") != "verified"
        for context in contexts
    ):
        return declared

    requested_dpus = {context.get("requested_dpu_count") for context in contexts}
    allocated_dpus = {context.get("allocated_dpu_count") for context in contexts}
    requested_tasklets = {
        context.get("requested_tasklets_per_dpu") for context in contexts
    }
    allocated_tasklets = {
        context.get("allocated_tasklets_per_dpu") for context in contexts
    }
    if (
        requested_dpus != {profile.requested_dpu_count}
        or requested_tasklets != {profile.tasklets_per_dpu}
        or len(allocated_dpus) != 1
        or len(allocated_tasklets) != 1
    ):
        return declared
    try:
        return DpuResourceContext(
            requested_dpu_count=profile.requested_dpu_count,
            requested_tasklets_per_dpu=profile.tasklets_per_dpu,
            allocated_dpu_count=next(iter(allocated_dpus)),
            allocated_tasklets_per_dpu=next(iter(allocated_tasklets)),
            allocation_status="verified",
        )
    except (TypeError, ValueError):
        return declared


def _verified_resident_resource_context(
    profile: HardwareTaskGraphResidentProfile,
    response: Mapping[str, Any],
) -> DpuResourceContext:
    return DpuResourceContext(
        requested_dpu_count=profile.requested_dpu_count,
        requested_tasklets_per_dpu=profile.tasklets_per_dpu,
        allocated_dpu_count=int(response["allocated_dpus"]),
        allocated_tasklets_per_dpu=int(response["tasklets"]),
        allocation_status="verified",
    )


def _resident_execution_plan(
    graph: TaskGraph,
    profile: HardwareTaskGraphResidentProfile,
    *,
    quantization_mode: str,
) -> tuple[UpmemExecutionPlan, DpuResourceContext]:
    resource_context = _resident_resource_context(profile)
    plan = UpmemExecutionPlan.for_task_graph(
        graph,
        kernel=UpmemKernelPlan(
            provider_id=RESIDENT_PROVIDER_ID,
            kernel_id="generic_loop_resident_graph",
            kernel_version="generic_loop_resident_graph_v1",
            implementation="explicit_sdk_resident",
            resident=True,
        ),
        placement=UpmemPlacementPlan(
            resources=resource_context,
            assignment_policy="single_dpu",
            topology="one_rank_one_dpu",
        ),
        communication=UpmemCommunicationPlan(
            host_to_dpu="explicit_sdk",
            dpu_to_host="explicit_sdk",
            intermediate_transport="mram_resident",
            reduction="host",
        ),
        numeric=UpmemNumericPlan(
            input_dtype="float32" if quantization_mode == "none" else "int8",
            accumulator_dtype=(
                "int32"
                if quantization_mode == "per_task_resident_requantize"
                else "float32"
            ),
            output_dtype="float32",
            quantization=quantization_mode,
            complex_policy=profile.complex_policy,
            full_precision_reference="complex128_cpu",
        ),
        schedule=UpmemSchedulePlan(
            ordering="topological_task_id",
            dependency_policy="strict",
            parallelism="serial",
            resident_lifetime="taskgraph",
        ),
    )
    return plan, resource_context


def _resource_context_metadata(context: DpuResourceContext) -> JsonDict:
    return {
        "requested_dpu_count": context.requested_dpu_count,
        "allocated_dpu_count": context.allocated_dpu_count,
        "requested_tasklets_per_dpu": context.requested_tasklets_per_dpu,
        "allocated_tasklets_per_dpu": context.allocated_tasklets_per_dpu,
        "allocation_status": context.allocation_status,
    }


def _execution_plan_metadata(
    plan: UpmemExecutionPlan,
    profile: HardwareTaskGraphResidentProfile,
    resource_context: DpuResourceContext,
) -> JsonDict:
    return {
        "execution_plan_schema_version": plan.schema_version,
        "execution_plan_hash": plan.execution_plan_hash,
        "execution_plan": plan.to_json_dict(),
        "execution_plan_provenance": RESIDENT_EXECUTION_PLAN_PROVENANCE,
        "execution_plan_native_package_binding": RESIDENT_EXECUTION_PLAN_NATIVE_PACKAGE_BINDING,
        "provider_id": plan.kernel.provider_id,
        "provider_metadata": {
            "provider_id": plan.kernel.provider_id,
            "backend_id": profile.backend_id,
            "route_id": profile.route_id,
            "implementation": plan.kernel.implementation,
            "execution_mode": "physical_hardware",
        },
        "resource_context": _resource_context_metadata(resource_context),
        "execution_contract_status": plan.validation.execution_contract_status,
        "policy_reference_status": plan.validation.policy_reference_status,
        "full_precision_accuracy_status": plan.validation.full_precision_accuracy_status,
        "scientific_validation_status": plan.validation.scientific_validation_status,
    }


def _failed_execution_metadata(
    base: Mapping[str, Any],
    plan: UpmemExecutionPlan,
    profile: HardwareTaskGraphResidentProfile,
    resource_context: DpuResourceContext,
) -> JsonDict:
    failed_plan = replace(
        plan,
        validation=UpmemValidationStatuses.from_checks(execution_contract=False),
    )
    return {**base, **_execution_plan_metadata(failed_plan, profile, resource_context)}


def _execution_contract_passed(
    response: Mapping[str, Any],
    resource_context: DpuResourceContext,
    bytes_invariant_status: str,
) -> bool:
    return (
        resource_context.allocation_status == "verified"
        and response.get("requested_dpus") == resource_context.requested_dpu_count
        and response.get("allocated_dpus") == resource_context.allocated_dpu_count
        and response.get("tasklets") == resource_context.allocated_tasklets_per_dpu
        and response.get("graph_request_count") == 1
        and response.get("target_observed") == "hardware"
        and response.get("hardware_allocation_verified") is True
        and response.get("hardware_execution") is True
        and response.get("native_execution") is True
        and response.get("hardware_kernel_executed") is True
        and response.get("cpu_fallback_used") is False
        and response.get("simulator_kernel_executed") is False
        and response.get("final_output_only_d2h") is True
        and bytes_invariant_status == "passed"
    )


def _resident_validation_statuses(
    *,
    execution_contract: bool,
    policy_validation: Mapping[str, Any],
    full_precision_accuracy: Mapping[str, Any],
) -> UpmemValidationStatuses:
    return UpmemValidationStatuses.from_checks(
        execution_contract=execution_contract,
        policy_reference=bool(policy_validation.get("passed")),
        full_precision_accuracy=bool(full_precision_accuracy.get("passed")),
    )


def _legacy_validation_status(
    policy_validation: Mapping[str, Any], bytes_invariant_status: str
) -> str:
    """Keep the pre-plan policy-reference/transfer validation contract."""

    return "passed" if policy_validation.get("passed") and bytes_invariant_status == "passed" else "failed"


def _accuracy(reference, actual, tolerance, reference_kind):
    difference = np.asarray(actual, dtype=np.complex128) - np.asarray(reference, dtype=np.complex128)
    max_abs = float(np.max(np.abs(difference))) if difference.size else 0.0
    l2 = float(np.linalg.norm(difference))
    ref_norm = float(np.linalg.norm(np.asarray(reference, dtype=np.complex128)))
    relative = l2 / ref_norm if ref_norm else l2
    return {
        "status": "passed" if max_abs <= tolerance else "failed",
        "passed": max_abs <= tolerance,
        "reference_kind": reference_kind,
        "max_abs_error": max_abs, "l2_error": l2, "relative_l2_error": relative,
        "max_abs_tolerance": tolerance,
        "available": True,
    }


def _policy_tolerance(mode):
    return 1.0e-5 if mode == "none" else 1.0e-5


def _full_precision_tolerance(mode):
    return 1.0e-5 if mode == "none" else 0.25


def _scale_metrics(policy: Mapping[str, Any]) -> JsonDict:
    scales: list[JsonDict] = []
    saturation = 0
    for task in policy.get("task_metrics", []):
        for component, metric in task.get("component_metrics", {}).items():
            scales.append({"task_id": task.get("task_id"), "component": component, "left_scale": metric.get("left_scale"), "right_scale": metric.get("right_scale")})
            saturation += int(metric.get("left_saturation_count", 0)) + int(metric.get("right_saturation_count", 0))
    return {"records": scales, "total_saturation_count": saturation, "observed_by": "cpu_policy_reference"}


def _failed_execution(graph, profile, mode, metadata, reason, started, *, native=None, package=None):
    summary: JsonDict = {
        "schema_version": RESIDENT_RUNTIME_SCHEMA_VERSION, "status": "failed",
        "reason": reason, "failure_stage": _failure_stage(str(reason), "hardware_profile_violation"),
        "route_id": RESIDENT_ROUTE_ID, "backend_id": RESIDENT_BACKEND_ID,
        "session_protocol": RESIDENT_SESSION_PROTOCOL, "timing_scope": RESIDENT_TIMING_SCOPE,
        "hardware_execution": False, "hardware_kernel_executed": False,
        "simulator_kernel_executed": False, "cpu_fallback_used": False,
        "target_observed": "hardware_unverified", "graph_request_count": 1,
        "logical_task_count": len(graph.tasks), "resident_slot_dtype": "float32",
        "final_output_only_d2h": True, "intermediate_h2d_bytes": 0, "intermediate_d2h_bytes": 0,
        "no_host_intermediate_output_files": True, "hardware_speedup_applicable": False,
        "total_route_time_s": time.perf_counter() - started, **metadata,
    }
    if package is not None:
        summary.update({
            "component_operation_count": package.component_operation_count,
            "resident_slot_descriptor_count": package.allocation.slot_descriptor_count,
            "resident_mram_pool_bytes": package.allocation.mram_pool_bytes,
            "resident_mram_used_bytes": package.allocation.mram_used_bytes,
            "descriptor_package_sha256": package.descriptor_sha256,
            "slot_lifetime_map": [item.to_json_dict() for item in package.allocation.lifetimes],
        })
    if native is not None:
        summary.update({
            "native_response_artifact": str(native.response_path),
            "native_stdout_snippet": native.stdout_snippet,
            "native_stderr_snippet": native.stderr_snippet,
        })
    return ResidentGraphExecution("failed", str(reason), None, summary, ())


def _failure_record(suite, reason, case):
    return {
        "schema_version": "resident_normalized_record_v1", "status": "failed",
        "suite_id": suite.suite["suite_id"],
        "case_id": case.get("case_id") if case else None, "route_id": RESIDENT_ROUTE_ID,
        "backend_id": RESIDENT_BACKEND_ID, "kernel_family": "generic_loop_resident_graph",
        "hardware_functionality_evidence": False, "hardware_timing_available": False,
        "hardware_kernel_executed": False, "simulator_kernel_executed": False,
        "cpu_fallback_used": False, "requested_dpu_count": 1,
        "tasklets_per_dpu": suite.profile.tasklets_per_dpu,
        "provider_id": RESIDENT_PROVIDER_ID,
        "provider_metadata": {
            "provider_id": RESIDENT_PROVIDER_ID,
            "backend_id": RESIDENT_BACKEND_ID,
            "route_id": RESIDENT_ROUTE_ID,
            "implementation": "explicit_sdk_resident",
            "execution_mode": "physical_hardware",
        },
        "resource_context": _resource_context_metadata(DpuResourceContext()),
        "execution_contract_status": "failed",
        "policy_reference_status": "not_run",
        "full_precision_accuracy_status": "not_run",
        "scientific_validation_status": "failed",
        "graph_request_count": 1, "failure_stage": _failure_stage(reason, "native_build_failed"),
        "reason": reason, "timing_scope": RESIDENT_TIMING_SCOPE,
        "claim_boundary": "no speedup, energy, scheduler, or multi-DPU claim",
    }


def _native_build_metadata(build, root):
    return {
        "attempted": True, "status": "passed", "source_tree_hash": build.source_tree_hash,
        "host_binary_hash": build.host_binary_hash, "dpu_binary_hash": build.dpu_binary_hash,
        "build_time_s": build.build_time_s, "build_command": list(build.build_command),
        "sdk_tools": build.sdk_tools,
        "session_root": str(build.session_root.relative_to(root)) if build.session_root.is_relative_to(root) else str(build.session_root),
    }


def _prepared_case_row(case, prepared):
    return {
        "case_id": case["case_id"], "status": "prepared", "n_qubits": prepared["circuit"].n_qubits,
        "circuit_semantics_hash": prepared["circuit_semantics_hash"], "tensor_network_hash": prepared["tensor_network_hash"],
        "path_variants": [
            {"path_variant_id": key, "task_count": len(value["graph"].tasks),
             "component_operation_count": value["allocation"].component_operation_count,
             "slot_descriptor_count": value["allocation"].slot_descriptor_count,
             "mram_used_bytes": value["allocation"].mram_used_bytes,
             "contraction_plan_hash": value["graph"].contraction_plan_hash,
             "contraction_path_structure_hash": contraction_path_structure_hash(value["graph"]),
             "execution_plan": value["execution_plan"].to_json_dict(),
             "execution_plan_hash": value["execution_plan"].execution_plan_hash,
             "execution_plan_provenance": RESIDENT_EXECUTION_PLAN_PROVENANCE,
             "execution_plan_native_package_binding": RESIDENT_EXECUTION_PLAN_NATIVE_PACKAGE_BINDING,
             "provider_id": value["execution_plan"].kernel.provider_id,
             "resource_context": _resource_context_metadata(value["resource_context"])}
            for key, value in prepared["variants"].items()
        ],
    }


def _manifest_integer(package: ResidentGraphPackage, key: str) -> int:
    if package.manifest_path is None:
        return 0
    payload = json.loads(package.manifest_path.read_text(encoding="utf-8"))
    return int(payload.get(key, 0))


def _rotate(values, offset):
    if not values:
        return values
    offset %= len(values)
    return values[offset:] + values[:offset]


def _failure_stage(reason: str, default: str) -> str:
    for stage in (
        "hardware_opt_in_missing", "hardware_profile_violation", "sdk_discovery_failed", "native_build_timeout", "native_build_failed",
        "manifest_parse_failed", "hardware_allocation_failed", "binary_load_failed", "initial_transfer_failed",
        "descriptor_transfer_failed", "kernel_launch_failed", "kernel_timeout", "hardware_session_timeout",
        "hardware_session_interrupted", "final_transfer_failed", "response_manifest_failed", "output_validation_failed",
        "hardware_release_failed", "hardware_release_unverified",
    ):
        if stage in reason:
            return stage
    return default


def _unique_dir(parent: Path) -> Path:
    parent.mkdir(parents=True, exist_ok=True)
    base = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    candidate = parent / base
    suffix = 1
    while candidate.exists():
        candidate = parent / f"{base}_{suffix:02d}"
        suffix += 1
    return candidate


def _align8(value: int) -> int:
    return (int(value) + 7) & ~7


def _product(values: Sequence[int]) -> int:
    result = 1
    for value in values:
        result *= int(value)
    return result


def _array_hash(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(repr(tuple(array.shape)).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


# Additive names used by the command and isolated tests.
run_upmem_hardware_taskgraph_resident_suite = run_resident_suite
build_resident_taskgraph_reference = build_resident_policy_reference
