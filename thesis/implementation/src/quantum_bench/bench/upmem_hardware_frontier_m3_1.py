"""Strict M3.1 two-DPU frontier hardware benchmark."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any, Mapping

import numpy as np
import yaml

from quantum_bench.bench.reporting import artifact_ref, write_normalized_records, write_run_manifest
from quantum_bench.bench.run_dirs import EVIDENCE_ARTIFACT_KIND, create_run_dir, sanitize
from quantum_bench.circuits import load_circuit, manifest as circuit_manifest
from quantum_bench.core.jsonio import write_json, write_jsonl
from quantum_bench.core.records import JsonDict
from quantum_bench.environment import capture_environment
from quantum_bench.targets.upmem.hardware_frontier_session import (
    HardwareFrontierExecution,
    build_hardware_frontier_session,
    execute_hardware_frontier_session,
    parse_hardware_frontier_profile,
)
from quantum_bench.targets.upmem.hardware_taskgraph_frontier import (
    BACKEND_ID,
    NATIVE_SCHEMA,
    NUMERIC_MODE,
    REQUEST_SCHEMA,
    ROUTE_ID,
    TIMING_SCOPE,
    build_hardware_frontier_graph_package,
    build_hardware_frontier_plan,
    validate_frontier_native_response,
    validate_frontier_package_validation_response,
    validate_hardware_frontier_graph,
    write_hardware_frontier_graph_package,
)
from quantum_bench.tn.execution import execute_task_frontier_np_einsum
from quantum_bench.tn.execution_bundle import (
    build_execution_bundle,
    executor_config_hash,
    with_execution_identity,
)
from quantum_bench.tn.network import build_tensor_network
from quantum_bench.tn.task_graph import plan_task_graph_with_config


SUITE_SCHEMA_VERSION = "upmem_hardware_frontier_m3_1_v1"
PLAN_SCHEMA_VERSION = "upmem_hardware_frontier_m3_1_plan_v1"
RUN_SCHEMA_VERSION = "upmem_hardware_frontier_m3_1_run_v1"
PROVIDER_ID = "upmem_frontier_hardware_m3_1"
CLAIM_BOUNDARY = (
    "functionality_and_host_observed_bringup_timing_only; no speedup, scaling, "
    "concurrency, or energy claim"
)
EXPECTED_PATH = [[0, 1], [0, 1], [0, 1]]
EXPECTED_QASM = "configs/circuits/upmem_m2/one_qubit_ry_h_ry_a.qasm"
EXPECTED_WORKLOAD_ID = "m3_1_ry_h_ry_a_opt_einsum_greedy"
EXPECTED_SOURCE_TASK_COUNT = 3
EXPECTED_PHYSICAL_TASK_INSTANCE_COUNT = 3
EXPECTED_WAVE_COUNT = 2
EXPECTED_DPU_TASK_COUNTS = [2, 1]
EXPECTED_METADATA = {
    "purpose": "strict_m3_1_two_dpu_frontier_hardware_functionality",
    "manual_invocation_required": True,
    "deterministic_unitary_only": True,
    "hardware_frontier_m3_1_schema_version": SUITE_SCHEMA_VERSION,
    "native_hardware_profile_version": "hardware_frontier_two_dpu_m3_1_v2",
    "claim_boundary": "functionality_and_host_observed_bringup_timing_only_no_speedup_scaling_concurrency_or_energy_claim",
}
EXPECTED_PROFILE = {
    "hardware_profile_version": "hardware_frontier_two_dpu_m3_1_v2",
    "target": "hardware",
    "backend_id": BACKEND_ID,
    "route_id": ROUTE_ID,
    "native_schema": "generic_loop_resident_frontier_two_dpu_v2",
    "requested_dpu_count": 2,
    "tasklets_per_dpu": 1,
    "numeric_mode": NUMERIC_MODE,
    "numeric_modes": [NUMERIC_MODE],
    "synchronous_execution": True,
    "device_launch_mode": "asynchronous_dpu_set",
    "host_completion_mode": "blocking_sync",
    "performance_claim_applicable": False,
}
M31_EXECUTOR_CONFIG = {
    "profile": EXPECTED_PROFILE["hardware_profile_version"],
    "backend_id": BACKEND_ID,
    "session_protocol": REQUEST_SCHEMA,
    "native_schema": NATIVE_SCHEMA,
    "numeric_mode": NUMERIC_MODE,
    "requested_dpu_count": EXPECTED_PROFILE["requested_dpu_count"],
    "tasklets_per_dpu": EXPECTED_PROFILE["tasklets_per_dpu"],
    "execution_plan_kind": "taskgraph_frontier_scheduler",
    "device_launch_mode": EXPECTED_PROFILE["device_launch_mode"],
    "host_completion_mode": EXPECTED_PROFILE["host_completion_mode"],
}
CANONICAL_SUITE_PATH = (
    Path(__file__).resolve().parents[3]
    / "configs"
    / "suites"
    / "upmem_hardware_frontier_m3_1.yml"
)


@dataclass(frozen=True)
class M31Suite:
    path: Path
    suite: dict[str, Any]
    profile: Any


@dataclass(frozen=True)
class PreparedCase:
    case: Mapping[str, Any]
    case_dir: Path
    circuit: Any
    network: Any
    graph: Any
    plan: Any
    package: Any
    reference: np.ndarray


@dataclass(frozen=True)
class UpmemHardwareFrontierM31PlanResult:
    plan_dir: Path
    summary_path: Path
    status: str


@dataclass(frozen=True)
class UpmemHardwareFrontierM31Result:
    run_dir: Path
    summary_path: Path
    status: str
    row_count: int


def load_upmem_hardware_frontier_m3_1_suite(path: Path) -> M31Suite:
    """Load only the committed, single-workload M3.1 suite contract."""

    resolved = path.resolve()
    if resolved != CANONICAL_SUITE_PATH.resolve():
        raise ValueError("hardware_profile_violation: M3.1 requires the committed suite")
    raw = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("hardware_profile_violation: M3.1 suite must be a mapping")
    suite = _strict_suite_shape(raw, resolved)
    profile_data = dict(raw["metadata"]["hardware_profile"])
    if "timeout_s" not in profile_data:
        profile_data["timeout_s"] = raw["defaults"].get("timeout_s", 30.0)
    profile = parse_hardware_frontier_profile(profile_data)
    return M31Suite(resolved, suite, profile)


def prepare_upmem_hardware_frontier_m3_1(
    root_dir: Path,
    *,
    suite_path: Path,
    build: bool = False,
    environment: Mapping[str, str] | None = None,
) -> UpmemHardwareFrontierM31PlanResult:
    suite = load_upmem_hardware_frontier_m3_1_suite(suite_path)
    env = dict(os.environ if environment is None else environment)
    plan_dir = _unique_dir(root_dir / "build" / "upmem_hardware_frontier_m3_1_plan")
    plan_dir.mkdir(parents=True, exist_ok=False)
    (plan_dir / "config").mkdir()
    shutil.copy2(suite.path, plan_dir / "config" / "resolved_suite.yml")
    write_json(plan_dir / "config" / "hardware_profile.json", _profile_json(suite.profile))
    write_json(plan_dir / "environment.json", capture_environment(root_dir))

    built = None
    native_build: JsonDict = {"attempted": False, "status": "not_requested"}
    status = "prepared"
    failure_stage: str | None = None
    failure_reason: str | None = None
    prepared: PreparedCase | None = None
    try:
        case = suite.suite["cases"][0]
        prepared = _prepare_case(root_dir, plan_dir / "cases" / sanitize(case["case_id"]), suite, case)
        if build:
            built = build_hardware_frontier_session(
                root_dir,
                plan_dir / "native_session",
                profile=suite.profile,
                environment=env,
            )
            native_build = _build_json(built, plan_dir)
        written_package = _write_package(
            prepared,
            plan_dir / "native_session",
            built.dpu_binary if built is not None else plan_dir / "native_session" / "native" / "build" / "bin" / "dpu_frontier_two_dpu",
            request_id="prepare",
        )
        prepared = replace(prepared, package=written_package)
        if built is not None:
            validation = _native_validate_only(built, prepared.package.manifest_path)
            write_json(plan_dir / "native_session" / "prepare_validation.json", validation)
    except Exception as exc:
        status = "failed"
        failure_stage = _failure_stage(str(exc), "hardware_profile_violation")
        failure_reason = str(exc)

    summary_path = plan_dir / "upmem_hardware_frontier_m3_1_plan.json"
    write_json(summary_path, {
        "schema_version": PLAN_SCHEMA_VERSION,
        "status": status,
        "failure_stage": failure_stage,
        "failure_reason": failure_reason,
        "suite_id": suite.suite["suite_id"],
        "route_id": ROUTE_ID,
        "backend_id": BACKEND_ID,
        "provider_id": PROVIDER_ID,
        "profile": _profile_json(suite.profile),
        "prepared_case": _prepared_json(prepared) if prepared is not None else None,
        "native_build": native_build,
        "native_validation": "native_session/prepare_validation.json" if built is not None and status == "prepared" else None,
        "dpu_allocation_attempted": False,
        "dpu_launch_attempted": False,
        "hardware": False,
        "simulator_kernel_executed": False,
        "cpu_fallback_used": False,
        "requested_environment": _requested_environment(env),
        "claim_boundary": CLAIM_BOUNDARY,
    })
    return UpmemHardwareFrontierM31PlanResult(plan_dir, summary_path, status)


def run_upmem_hardware_frontier_m3_1(
    root_dir: Path,
    *,
    suite_path: Path,
    environment: Mapping[str, str] | None = None,
) -> UpmemHardwareFrontierM31Result:
    env = dict(os.environ if environment is None else environment)
    _require_execution_environment(env)
    suite = load_upmem_hardware_frontier_m3_1_suite(suite_path)
    run_dir = create_run_dir(
        root_dir,
        str(suite.suite["suite_id"]),
        artifact_kind=EVIDENCE_ARTIFACT_KIND,
        route_label="upmem_hw_frontier_m3_1",
    )
    return _run_suite(root_dir, run_dir, suite, env)


def _run_suite(root_dir: Path, run_dir: Path, suite: M31Suite, env: Mapping[str, str]) -> UpmemHardwareFrontierM31Result:
    shutil.copy2(suite.path, run_dir / "config" / "resolved_suite.yml")
    write_json(run_dir / "config" / "hardware_profile.json", _profile_json(suite.profile))
    write_json(run_dir / "environment.json", capture_environment(root_dir))
    run_manifest = write_run_manifest(
        run_dir,
        run_kind="upmem_hardware_frontier_m3_1",
        suite_id=suite.suite["suite_id"],
        suite_path=str(suite.path),
        artifact_kind=EVIDENCE_ARTIFACT_KIND,
        route_label="upmem_hw_frontier_m3_1",
        route_id=ROUTE_ID,
        backend_id=BACKEND_ID,
        execution_scope="physical_two_dpu_frontier_full_taskgraph",
        evidence_type="executed_dispatch_only",
        upmem_execution_mode="frontier_graph_request",
        artifact_retention="full",
        summary="upmem_hardware_frontier_m3_1_summary.json",
        policies=("opt_einsum_greedy",),
        quantization_modes=(NUMERIC_MODE,),
        root_dir=root_dir,
    )
    run_manifest["requested_environment"] = _requested_environment(env)
    records: list[JsonDict] = []
    warmups: list[JsonDict] = []
    case = suite.suite["cases"][0]
    build = None
    prepared = None
    failure_stage = None
    failure_reason = None
    try:
        build = build_hardware_frontier_session(
            root_dir, run_dir / "native_session", profile=suite.profile, environment=env
        )
        prepared = _prepare_case(root_dir, run_dir / "cases" / sanitize(case["case_id"]), suite, case)
        for warmup_id in range(int(suite.suite["warmups"])):
            warmup = _execute_request(
                build, prepared, suite, env, f"warmup-{warmup_id:02d}", warmup=True
            )
            warmups.append(warmup)
            if warmup["status"] != "completed" and suite.suite["route_policy"]["fail_fast"]:
                break
        warmup_failed = bool(warmups and warmups[-1]["status"] != "completed")
        if not warmup_failed or not suite.suite["route_policy"]["fail_fast"]:
            for repeat_id in range(int(suite.suite["repeats"])):
                measured = _execute_request(
                    build, prepared, suite, env, f"measured-{repeat_id:02d}", warmup=False
                )
                records.append(measured)
                if measured["status"] != "completed" and suite.suite["route_policy"]["fail_fast"]:
                    break
    except Exception as exc:
        failure_stage = _failure_stage(str(exc), "native_build_failed")
        failure_reason = str(exc)

    write_jsonl(run_dir / "warmups.jsonl", warmups)
    # The measured matrix is intentionally exactly five rows on the successful path.
    write_normalized_records(run_dir, records)
    passed = (
        bool(records)
        and len(records) == 5
        and len(warmups) == 1
        and all(row.get("status") == "completed" for row in records)
        and all(row.get("status") == "completed" for row in warmups)
    )
    if not passed and failure_stage is None:
        failure_stage = next(
            (
                str(row.get("failure_stage"))
                for row in [*warmups, *records]
                if row.get("failure_stage")
            ),
            "execution_failed",
        )
    if failure_reason is None and failure_stage is not None:
        failure_reason = next(
            (
                str(row.get("reason"))
                for row in [*warmups, *records]
                if row.get("failure_stage")
            ),
            None,
        )
    has_observed_execution = any(row.get("native_execution") is True for row in [*warmups, *records])
    summary_path = run_dir / "upmem_hardware_frontier_m3_1_summary.json"
    write_json(summary_path, {
        "schema_version": RUN_SCHEMA_VERSION,
        "status": "completed" if passed else "failed",
        "failure_stage": failure_stage,
        "failure_reason": failure_reason,
        "suite_id": suite.suite["suite_id"],
        "route_id": ROUTE_ID,
        "backend_id": BACKEND_ID,
        "provider_id": PROVIDER_ID,
        "row_count": len(records),
        "warmup_count": len(warmups),
        "expected_source_task_count": EXPECTED_SOURCE_TASK_COUNT,
        "expected_physical_task_instance_count": EXPECTED_PHYSICAL_TASK_INSTANCE_COUNT,
        "expected_wave_count": EXPECTED_WAVE_COUNT,
        "expected_dpu_task_counts": list(EXPECTED_DPU_TASK_COUNTS),
        **_summary_execution_metrics(records),
        "duplicate_contraction_check": "passed" if passed else "not_run",
        "missing_dependency_check": "passed" if passed else "not_run",
        "dependency_check": "passed" if passed else "not_run",
        "frontier_scheduler_enabled": True,
        "parallelism_mode": "frontier",
        "evidence_type": "executed_dispatch_only" if has_observed_execution else "not_observed",
        "overlap_measured": False,
        "transfer_accounting_scope": "native_sdk_observed_application_visible",
        "transfer_invariant_status": "passed" if has_observed_execution else "not_run",
        "timing_evidence": "host_observed",
        "kernel_timing_available": False,
        "validation_status": "passed" if passed else "failed",
        "measured_matrix": "normalized_records.jsonl",
        "warmups": "warmups.jsonl",
        "native_build": _build_json(build, run_dir) if build is not None else {"attempted": True, "status": "failed"},
        "hardware": passed,
        "simulator_kernel_executed": False,
        "cpu_fallback_used": False,
        "fallback_used": False,
        "native_observed_transfer_only": has_observed_execution,
        "claim_boundary": CLAIM_BOUNDARY,
        "no_speedup_claim": True,
        "no_scaling_claim": True,
        "no_concurrency_claim": True,
        "no_energy_claim": True,
        "normalized_records": "normalized_records.jsonl",
        "requested_environment": _requested_environment(env),
    })
    run_manifest.update({
        "summary": summary_path.name,
        "hardware_available": "verified_by_execution" if passed else "not_verified_by_execution",
        "upmem_sdk_available": "verified_by_execution" if passed else "not_verified_by_execution",
        "evidence_type": "executed_dispatch_only",
        "claim_boundary": CLAIM_BOUNDARY,
        "overlap_measured": False,
        "no_speedup_claim": True,
        "no_scaling_claim": True,
        "no_concurrency_claim": True,
        "no_energy_claim": True,
    })
    write_json(run_dir / "run_manifest.json", run_manifest)
    return UpmemHardwareFrontierM31Result(run_dir, summary_path, "completed" if passed else "failed", len(records))


def _prepare_case(root_dir: Path, case_dir: Path, suite: M31Suite, case: Mapping[str, Any]) -> PreparedCase:
    case_dir.mkdir(parents=True, exist_ok=True)
    circuit = load_circuit(dict(case), root_dir)
    network = build_tensor_network(circuit)
    graph = with_execution_identity(plan_task_graph_with_config(network, dict(suite.suite["planner"])))
    validate_hardware_frontier_graph(graph, network)
    if [list(step) for step in graph.path] != EXPECTED_PATH:
        raise ValueError("hardware_profile_violation: unexpected M3.1 contraction path")
    reference, reference_metrics = execute_task_frontier_np_einsum(graph, network, frontier_worker_count=2)
    reference = np.asarray(reference, dtype=np.complex128)
    expected_output = np.asarray(case["expected_output"], dtype=np.float64)
    if reference.shape != expected_output.shape or not np.allclose(reference.real, expected_output, rtol=0.0, atol=1.0e-6):
        raise ValueError("hardware_profile_violation: CPU reference differs from committed expected output")
    plan = build_hardware_frontier_plan(graph, network)
    package = build_hardware_frontier_graph_package(
        graph,
        network,
        case_id=str(case["case_id"]),
        suite_id=str(suite.suite["suite_id"]),
        quantization_mode=NUMERIC_MODE,
        full_precision_output=reference,
    )
    write_json(case_dir / "circuit.json", circuit_manifest(circuit))
    write_json(case_dir / "task_graph.json", graph)
    write_json(case_dir / "execution_bundle.json", build_execution_bundle(graph, case_id=str(case["case_id"]), suite_id=str(suite.suite["suite_id"])))
    np.save(case_dir / "cpu_reference_final_tensor.npy", reference, allow_pickle=False)
    write_json(case_dir / "frontier_plan.json", plan.to_json_dict())
    write_json(case_dir / "reference_execution.json", reference_metrics)
    return PreparedCase(case, case_dir, circuit, network, graph, plan, package, reference)


def _write_package(prepared: PreparedCase, session_root: Path, dpu_binary: Path, *, request_id: str) -> Any:
    artifact = write_hardware_frontier_graph_package(
        prepared.package, session_root, dpu_binary=dpu_binary, request_id=request_id
    )
    if artifact.manifest_path is None:
        raise RuntimeError("manifest_parse_failed: frontier request manifest is missing")
    _complete_frontier_manifest(artifact.manifest_path)
    return artifact


def _complete_frontier_manifest(manifest_path: Path) -> None:
    """Keep the owned request contract explicit for the current target package writer."""

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("manifest_parse_failed: frontier manifest is not an object")
    operations = manifest.get("frontier_task_operations")
    if not isinstance(operations, list):
        raise ValueError("manifest_parse_failed: frontier operation plan is missing")
    for operation in operations:
        if isinstance(operation, dict):
            operation.setdefault("component", "real")
            operation.setdefault("kind", "contract")
            operation.setdefault("mode", NUMERIC_MODE)
    write_json(manifest_path, manifest)


def _execute_request(build: Any, prepared: PreparedCase, suite: M31Suite, env: Mapping[str, str], request_id: str, *, warmup: bool) -> JsonDict:
    started = time.perf_counter()
    native: HardwareFrontierExecution | None = None
    validated_native_response = False
    try:
        package = _write_package(prepared, build.session_root, build.dpu_binary, request_id=request_id)
        response_path = build.session_root / f"{sanitize(request_id)}_response.json"
        native = execute_hardware_frontier_session(
            build,
            manifest_path=package.manifest_path,
            response_path=response_path,
            profile=suite.profile,
            environment=env,
        )
        manifest = json.loads(package.manifest_path.read_text(encoding="utf-8"))
        if native.status != "completed":
            return _record(
                suite,
                prepared,
                native,
                None,
                request_id,
                warmup,
                started,
                "failed",
                native.failure_stage or "native frontier request failed",
                run_dir=build.session_root.parent,
                failure_stage=native.failure_stage or "native_host_failed",
            )
        validate_frontier_native_response(native.response, manifest)
        validated_native_response = True
        actual = _load_final_output(build.session_root, manifest, prepared.graph.tasks[-1].output_shape)
        if not np.allclose(actual, prepared.reference, rtol=0.0, atol=1.0e-6):
            raise RuntimeError("output_validation_failed: final output differs from CPU reference")
        return _record(
            suite,
            prepared,
            native,
            actual,
            request_id,
            warmup,
            started,
            "completed",
            None,
            run_dir=build.session_root.parent,
            validated_native_response=validated_native_response,
        )
    except Exception as exc:
        return _record(
            suite,
            prepared,
            native,
            None,
            request_id,
            warmup,
            started,
            "failed",
            f"response_validation_error: {exc}",
            run_dir=build.session_root.parent,
            failure_stage=(
                native.failure_stage
                if native is not None and native.failure_stage
                else "response_evidence_invalid"
            ),
            validated_native_response=validated_native_response,
        )


def _record(
    suite: M31Suite,
    prepared: PreparedCase,
    native: HardwareFrontierExecution | None,
    actual: np.ndarray | None,
    request_id: str,
    warmup: bool,
    started: float,
    status: str,
    reason: str | None,
    *,
    run_dir: Path,
    failure_stage: str | None = None,
    validated_native_response: bool = False,
) -> JsonDict:
    response = native.response if native is not None else {}
    timing = response.get("timing") if isinstance(response.get("timing"), Mapping) else {}
    observed_execution = status == "completed" and validated_native_response
    observed = _observed_execution_metrics(response) if observed_execution else _empty_execution_metrics()
    response_artifact = None
    if native is not None:
        response_path = getattr(native, "response_path", None)
        if response_path is not None:
            try:
                response_rel = Path(response_path).resolve().relative_to(run_dir.resolve())
            except (OSError, ValueError):
                pass
            else:
                response_artifact = artifact_ref(
                    run_dir, response_rel, role="native_response"
                )
    execution_bundle_rel = prepared.case_dir.relative_to(run_dir) / "execution_bundle.json"
    validation_max_abs_error = (
        float(np.max(np.abs(np.asarray(actual) - prepared.reference)))
        if observed_execution and actual is not None
        else None
    )
    return {
        "schema_version": "upmem_hardware_frontier_m3_1_record_v1",
        "status": status,
        "suite_id": suite.suite["suite_id"],
        "case_id": prepared.case["case_id"],
        "request_id": request_id,
        "warmup": warmup,
        "repeat_id": None if warmup else int(request_id.rsplit("-", 1)[-1]),
        "route_id": ROUTE_ID,
        "backend_id": BACKEND_ID,
        "provider_id": PROVIDER_ID,
        "numeric_mode": NUMERIC_MODE,
        "planner": "opt_einsum_greedy",
        "expected_path": [list(step) for step in prepared.graph.path],
        "execution_target": "hardware",
        "hardware": observed_execution and response.get("hardware_execution") is True,
        "hardware_execution": response.get("hardware_execution") is True if observed_execution else False,
        "hardware_functionality_evidence": response.get("hardware_functionality_evidence") is True if observed_execution else False,
        "hardware_kernel_executed": response.get("hardware_kernel_executed") is True if observed_execution else False,
        "simulator_kernel_executed": response.get("simulator_kernel_executed") is True if observed_execution else False,
        "cpu_fallback_used": response.get("cpu_fallback_used") is True if observed_execution else False,
        "fallback_used": (
            response.get("cpu_fallback_used") is True or response.get("simulator_kernel_executed") is True
        ) if observed_execution else False,
        "native_execution": observed_execution,
        "expected_source_task_count": EXPECTED_SOURCE_TASK_COUNT,
        "expected_physical_task_instance_count": EXPECTED_PHYSICAL_TASK_INSTANCE_COUNT,
        "expected_wave_count": EXPECTED_WAVE_COUNT,
        "expected_dpu_task_counts": list(EXPECTED_DPU_TASK_COUNTS),
        "requested_dpu_count": response.get("requested_dpus") if observed_execution else None,
        "allocated_dpu_count": response.get("allocated_dpus") if observed_execution else None,
        "tasklets_per_dpu": response.get("tasklets_per_dpu") if observed_execution else None,
        "co_dispatch_observed": response.get("co_dispatch_observed") is True if observed_execution else False,
        "co_dispatch_confirmed": response.get("co_dispatch_confirmed") is True if observed_execution else False,
        **observed,
        "duplicate_contraction_check": "passed" if observed_execution else None,
        "missing_dependency_check": "passed" if observed_execution else None,
        "dependency_check": "passed" if observed_execution else None,
        "frontier_scheduler_enabled": True,
        "frontier_parallel_execution": False,
        "parallelism_mode": "frontier",
        "parallelism_evidence_type": "executed_dispatch_only" if observed_execution else "not_observed",
        "evidence_type": "executed_dispatch_only" if observed_execution else "not_observed",
        "overlap_measured": False,
        "transfer_accounting_scope": response.get("transfer_accounting_scope", "native_sdk_observed_application_visible"),
        "transfer_invariant_status": "passed" if observed_execution else None,
        "native_observed_transfer_only": observed_execution,
        "actual_h2d_bytes": response.get("actual_h2d_bytes") if observed_execution else None,
        "actual_d2h_bytes": response.get("actual_d2h_bytes") if observed_execution else None,
        "actual_transfer_bytes": response.get("actual_transfer_bytes") if observed_execution else None,
        "timing_scope": TIMING_SCOPE,
        "timing_evidence": "host_observed",
        "timing_is_bringup_only": True,
        "host_observed_timing": True,
        "kernel_timing_available": False,
        "native_timing": dict(timing),
        "validation_status": "passed" if observed_execution else "failed",
        "scientific_validation_status": "passed" if observed_execution else "failed",
        "validation_tolerance_abs": 1.0e-6,
        "validation_max_abs_error": validation_max_abs_error,
        "output_shape": list(actual.shape) if actual is not None else None,
        "execution_plan_kind": "taskgraph_frontier_scheduler",
        "execution_plan_executed": observed_execution,
        "circuit_semantics_hash": prepared.graph.circuit_semantics_hash,
        "tensor_network_hash": prepared.graph.tensor_network_hash,
        "contraction_plan_hash": prepared.graph.contraction_plan_hash,
        "executor_config_hash": executor_config_hash(ROUTE_ID, M31_EXECUTOR_CONFIG),
        "execution_bundle_artifact": artifact_ref(
            run_dir, execution_bundle_rel, role="execution_bundle"
        ),
        "hardware_timing_available": False,
        "speedup_claim_applicable": False,
        "scaling_claim_applicable": False,
        "concurrency_claim_applicable": False,
        "energy_claim_applicable": False,
        "claim_boundary": CLAIM_BOUNDARY,
        "elapsed_host_s": time.perf_counter() - started,
        "failure_stage": failure_stage,
        "reason": reason,
        "native_response": response if native is not None else None,
        "native_response_artifact": response_artifact,
        "native_failure_stage": native.failure_stage if native is not None else None,
        "native_failure_context": response.get("failure_context") if native is not None else None,
        "native_error": response.get("error") if native is not None else None,
        "native_session_command": list(getattr(native, "command", ())) if native is not None else None,
        "native_stdout_snippet": getattr(native, "stdout_snippet", None) if native is not None else None,
        "native_stderr_snippet": getattr(native, "stderr_snippet", None) if native is not None else None,
        "native_timed_out": getattr(native, "timed_out", None) if native is not None else None,
        "native_cleanup_confirmed": getattr(native, "cleanup_confirmed", None) if native is not None else None,
    }


def _load_final_output(root: Path, manifest: Mapping[str, Any], shape: Any) -> np.ndarray:
    binding = manifest.get("final_output_binding")
    if not isinstance(binding, Mapping) or binding.get("component") != "real":
        raise ValueError("output_validation_failed: final output binding is invalid")
    relative = binding.get("output_path")
    elements = binding.get("elements")
    if not isinstance(relative, str) or not isinstance(elements, int) or elements < 0:
        raise ValueError("output_validation_failed: final output path or size is invalid")
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError("output_validation_failed: final output escapes session root") from exc
    if not path.is_file():
        raise ValueError("output_validation_failed: final output file is missing")
    values = np.fromfile(path, dtype="<f4")
    if values.size != elements or tuple(values.shape) != (elements,):
        raise ValueError("output_validation_failed: final output element count mismatch")
    if int(np.prod(tuple(shape))) != elements:
        raise ValueError("output_validation_failed: final output shape mismatch")
    return values.reshape(tuple(shape)).astype(np.complex128)


def _empty_execution_metrics() -> JsonDict:
    return {
        "source_task_count": None,
        "source_task_completion_count": None,
        "physical_task_instance_count": None,
        "completed_physical_task_instance_count": None,
        "physical_instance_count": None,
        "completed_physical_instance_count": None,
        "frontier_wave_count": None,
        "barrier_count": None,
        "observed_dpu_task_counts": None,
    }


def _observed_execution_metrics(response: Mapping[str, Any]) -> JsonDict:
    """Project counts only from fields validated in the native response."""

    metrics = _empty_execution_metrics()
    launch = response.get("launch")
    tasks = response.get("tasks")
    physical_instances = response.get("physical_task_instances")
    wave_plan = response.get("wave_plan")
    per_dpu_completed = response.get("per_dpu_completed_operations")
    if isinstance(launch, Mapping) and isinstance(launch.get("task_count"), int):
        metrics["source_task_count"] = launch["task_count"]
    if isinstance(tasks, list):
        metrics["source_task_completion_count"] = sum(
            1
            for task in tasks
            if isinstance(task, Mapping)
            and task.get("completed") is True
            and task.get("completion_confirmed") is True
        )
    if isinstance(physical_instances, list):
        metrics["physical_task_instance_count"] = len(physical_instances)
        metrics["physical_instance_count"] = len(physical_instances)
    if isinstance(per_dpu_completed, list) and all(isinstance(item, int) for item in per_dpu_completed):
        completed = sum(per_dpu_completed)
        metrics["completed_physical_task_instance_count"] = completed
        metrics["completed_physical_instance_count"] = completed
    if isinstance(wave_plan, list):
        metrics["frontier_wave_count"] = len(wave_plan)
    if isinstance(response.get("barrier_count"), int):
        metrics["barrier_count"] = response["barrier_count"]
    counts = response.get("observed_dpu_task_counts")
    if isinstance(counts, list) and all(isinstance(item, int) for item in counts):
        metrics["observed_dpu_task_counts"] = list(counts)
    return metrics


def _summary_execution_metrics(records: list[JsonDict]) -> JsonDict:
    for row in records:
        if row.get("status") == "completed" and row.get("native_execution") is True:
            return {
                key: row.get(key)
                for key in (
                    "source_task_count",
                    "source_task_completion_count",
                    "physical_task_instance_count",
                    "completed_physical_task_instance_count",
                    "frontier_wave_count",
                    "barrier_count",
                    "observed_dpu_task_counts",
                )
            }
    return {
        "source_task_count": None,
        "source_task_completion_count": None,
        "physical_task_instance_count": None,
        "completed_physical_task_instance_count": None,
        "frontier_wave_count": None,
        "barrier_count": None,
        "observed_dpu_task_counts": None,
    }


def _requested_environment(environment: Mapping[str, str]) -> JsonDict:
    keys = (
        "UPMEM_ALLOW_PHYSICAL_HARDWARE",
        "DPU_BACKEND",
        "UPMEM_HOME",
        "UPMEM_PROFILE",
        "UPMEM_PROFILE_BASE",
        "PATH",
    )
    return {key: environment.get(key) for key in keys}


def _native_validate_only(build: Any, manifest_path: Path | None) -> JsonDict:
    if manifest_path is None:
        raise RuntimeError("manifest_parse_failed: validate-only manifest is missing")
    command = (str(build.host_binary), "--validate-frontier-package", str(manifest_path))
    completed = subprocess.run(command, cwd=build.build_dir, capture_output=True, text=True, check=False, timeout=30.0)
    payload = _last_json_line(completed.stdout)
    if completed.returncode != 0 or payload is None:
        raise RuntimeError("response_evidence_invalid: native validate-only failed")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_frontier_package_validation_response(payload, manifest)
    return {"status": "passed", "command": list(command), "response": payload, "allocation_attempted": False, "launch_attempted": False}


def _strict_suite_shape(raw: dict[str, Any], path: Path) -> dict[str, Any]:
    if raw.get("schema_version") != 2 or raw.get("suite_id") != "upmem_hardware_frontier_m3_1" or raw.get("fail_fast") is not True:
        raise ValueError("hardware_profile_violation: M3.1 suite identity differs")
    metadata = raw.get("metadata")
    defaults = raw.get("defaults")
    routes = raw.get("routes")
    workloads = raw.get("workloads")
    if not isinstance(metadata, dict) or set(metadata) != set(EXPECTED_METADATA) | {"hardware_profile"}:
        raise ValueError("hardware_profile_violation: M3.1 suite schema differs")
    if any(metadata.get(key) != value for key, value in EXPECTED_METADATA.items()):
        raise ValueError("hardware_profile_violation: M3.1 suite metadata differs")
    profile = metadata.get("hardware_profile")
    if not isinstance(profile, dict) or set(profile) - (set(EXPECTED_PROFILE) | {"timeout_s"}) or set(EXPECTED_PROFILE) - set(profile):
        raise ValueError("hardware_profile_violation: M3.1 hardware profile is incomplete")
    if any(profile.get(key) != value for key, value in EXPECTED_PROFILE.items()):
        raise ValueError("hardware_profile_violation: M3.1 hardware profile differs")
    if not isinstance(defaults, dict) or set(defaults) != {"warmups", "repeats", "timeout_s", "planner"} or defaults.get("warmups") != 1 or defaults.get("repeats") != 5 or defaults.get("planner") != {"engine": "opt_einsum", "optimize": "greedy"}:
        raise ValueError("hardware_profile_violation: M3.1 warmup/repeat/planner contract differs")
    if not isinstance(routes, list) or len(routes) != 1 or not isinstance(routes[0], dict):
        raise ValueError("hardware_profile_violation: M3.1 route contract differs")
    route = routes[0]
    if set(route) != {"id", "role", "required", "options"} or route.get("id") != ROUTE_ID or route.get("role") != "required_m3_1_frontier_hardware_workload" or route.get("required") is not True or route.get("options") != {"backend_id": BACKEND_ID, "numeric_mode": NUMERIC_MODE, "requested_dpu_count": 2, "tasklets_per_dpu": 1}:
        raise ValueError("hardware_profile_violation: M3.1 route contract differs")
    if not isinstance(workloads, list) or len(workloads) != 1:
        raise ValueError("hardware_profile_violation: M3.1 requires one workload")
    workload = workloads[0]
    if not isinstance(workload, dict) or workload.get("id") != EXPECTED_WORKLOAD_ID:
        raise ValueError("hardware_profile_violation: M3.1 workload identity differs")
    circuit = workload.get("circuit")
    if circuit != {"kind": "qasm_file", "name": "one_qubit_ry_h_ry_a", "path": EXPECTED_QASM}:
        raise ValueError("hardware_profile_violation: M3.1 requires the committed RY-H-RY A QASM")
    if workload.get("expected_path") != EXPECTED_PATH or workload.get("numeric_mode") != NUMERIC_MODE:
        raise ValueError("hardware_profile_violation: M3.1 workload path/numeric contract differs")
    suite = {
        "schema_version": 2,
        "suite_id": raw["suite_id"],
        "metadata": metadata,
        "warmups": 1,
        "repeats": 5,
        "planner": dict(defaults["planner"]),
        "cases": [{key: value for key, value in workload.items() if key != "id"} | {"case_id": workload["id"], "workload_id": workload["id"]}],
        "route_policy": {"routes": [ROUTE_ID], "fail_fast": True},
        "_suite_path": str(path),
    }
    return suite


def _require_execution_environment(environment: Mapping[str, str]) -> None:
    if environment.get("UPMEM_ALLOW_PHYSICAL_HARDWARE") != "1":
        raise ValueError("hardware_opt_in_missing: UPMEM_ALLOW_PHYSICAL_HARDWARE=1 is required")
    if environment.get("DPU_BACKEND"):
        raise ValueError("hardware_profile_violation: DPU_BACKEND must be unset")


def _profile_json(profile: Any) -> JsonDict:
    return {
        "hardware_profile_version": profile.version,
        "target": profile.target,
        "backend_id": profile.backend_id,
        "route_id": profile.route_id,
        "native_schema": profile.native_schema,
        "requested_dpu_count": profile.requested_dpu_count,
        "tasklets_per_dpu": profile.tasklets_per_dpu,
        "numeric_mode": profile.numeric_mode,
        "numeric_modes": list(profile.numeric_modes),
        "synchronous_execution": profile.synchronous_execution,
        "device_launch_mode": profile.device_launch_mode,
        "host_completion_mode": profile.host_completion_mode,
        "timeout_s": profile.timeout_s,
        "performance_claim_applicable": profile.performance_claim_applicable,
    }


def _prepared_json(prepared: PreparedCase | None) -> JsonDict | None:
    if prepared is None:
        return None
    return {
        "case_id": prepared.case["case_id"],
        "qasm_path": EXPECTED_QASM,
        "n_qubits": prepared.circuit.n_qubits,
        "task_count": len(prepared.graph.tasks),
        "path": [list(step) for step in prepared.graph.path],
        "frontier_plan": prepared.plan.to_json_dict(),
        "cpu_reference": "cases/%s/cpu_reference_final_tensor.npy" % sanitize(prepared.case["case_id"]),
        "package": prepared.package.to_json_dict(),
        "frontier_manifest": (
            json.loads(prepared.package.manifest_path.read_text(encoding="utf-8"))
            if prepared.package.manifest_path is not None
            else None
        ),
    }


def _build_json(build: Any, root: Path) -> JsonDict:
    return {
        "attempted": True,
        "status": "passed",
        "source_tree_hash": build.source_tree_hash,
        "host_binary_hash": build.host_binary_hash,
        "dpu_binary_hash": build.dpu_binary_hash,
        "build_time_s": build.build_time_s,
        "build_command": list(build.build_command),
        "sdk_tools": build.sdk_tools,
        "session_root": str(build.session_root.relative_to(root)) if build.session_root.is_relative_to(root) else str(build.session_root),
    }


def _failure_record(suite: M31Suite, case: Mapping[str, Any], stage: str, reason: str, repeat_id: int) -> JsonDict:
    return {
        "schema_version": "upmem_hardware_frontier_m3_1_record_v1",
        "status": "failed",
        "suite_id": suite.suite["suite_id"],
        "case_id": case.get("case_id"),
        "repeat_id": repeat_id,
        "route_id": ROUTE_ID,
        "backend_id": BACKEND_ID,
        "numeric_mode": NUMERIC_MODE,
        "hardware": False,
        "simulator_kernel_executed": False,
        "cpu_fallback_used": False,
        "expected_source_task_count": EXPECTED_SOURCE_TASK_COUNT,
        "expected_physical_task_instance_count": EXPECTED_PHYSICAL_TASK_INSTANCE_COUNT,
        "expected_wave_count": EXPECTED_WAVE_COUNT,
        "expected_dpu_task_counts": list(EXPECTED_DPU_TASK_COUNTS),
        **_empty_execution_metrics(),
        "frontier_scheduler_enabled": True,
        "parallelism_mode": "frontier",
        "parallelism_evidence_type": "not_observed",
        "evidence_type": "not_observed",
        "overlap_measured": False,
        "validation_status": "failed",
        "scientific_validation_status": "failed",
        "failure_stage": stage,
        "reason": reason,
        "claim_boundary": CLAIM_BOUNDARY,
    }


def _failure_stage(reason: str, default: str) -> str:
    stage = reason.split(":", 1)[0].strip()
    if stage and " " not in stage:
        return stage
    return default


def _last_json_line(value: str) -> JsonDict | None:
    for line in reversed(value.splitlines()):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def _unique_dir(parent: Path) -> Path:
    parent.mkdir(parents=True, exist_ok=True)
    base = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    candidate = parent / base
    suffix = 1
    while candidate.exists():
        candidate = parent / f"{base}_{suffix:02d}"
        suffix += 1
    return candidate


__all__ = [
    "CANONICAL_SUITE_PATH",
    "M31Suite",
    "UpmemHardwareFrontierM31PlanResult",
    "UpmemHardwareFrontierM31Result",
    "load_upmem_hardware_frontier_m3_1_suite",
    "prepare_upmem_hardware_frontier_m3_1",
    "run_upmem_hardware_frontier_m3_1",
]
