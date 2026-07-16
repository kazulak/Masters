"""Guarded physical one-DPU generic TaskGraph functionality MVP.

The route intentionally executes one deterministic synthetic real-valued
contraction.  It establishes only physical execution and exact validation of
the generic loop; it is not a quantum benchmark or a speedup experiment.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import socket
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from quantum_bench.bench.reporting import write_normalized_records, write_run_manifest
from quantum_bench.bench.run_dirs import EVIDENCE_ARTIFACT_KIND, create_run_dir
from quantum_bench.core.jsonio import write_json
from quantum_bench.core.records import ContractionTask, JsonDict, TensorSpec, TensorValue
from quantum_bench.environment.capture import capture_environment
from quantum_bench.formats import FixedPointSpec
from quantum_bench.routing.generic_prepare import (
    GenericTaskPreparationCaps,
    GenericTaskPreparationInput,
    prepare_generic_task,
)
from quantum_bench.targets.upmem.environment import discover_upmem_sdk
from quantum_bench.targets.upmem.generic_bridge import (
    GenericBridgeExecutionResult,
    execute_generic_bridge,
    write_generic_bridge_input_manifest,
)
from quantum_bench.targets.upmem.hardware_generic_mvp import (
    HARDWARE_GENERIC_MVP_BACKEND_ID,
    HARDWARE_GENERIC_MVP_OUTPUT_TILE_ELEMENTS,
    HARDWARE_GENERIC_MVP_SDK_ALLOCATION_PROFILE,
    HardwareGenericMvpCase,
    HardwareGenericMvpSuite,
    hardware_generic_mvp_profile_metadata,
    load_hardware_generic_mvp_suite,
    validate_hardware_generic_mvp_manifest,
)


UPMEM_HARDWARE_GENERIC_MVP_BENCHMARK_SCHEMA_VERSION = "upmem_hardware_generic_mvp_benchmark_v1"
UPMEM_HARDWARE_GENERIC_MVP_PLAN_SCHEMA_VERSION = "upmem_hardware_generic_mvp_plan_v1"


@dataclass(frozen=True)
class UpmemHardwareGenericMvpResult:
    run_dir: Path
    summary_path: Path
    status: str
    row_count: int


@dataclass(frozen=True)
class UpmemHardwareGenericMvpPlanResult:
    plan_dir: Path
    summary_path: Path
    status: str


def validate_hardware_generic_execution_request(
    *, execute: bool, environment: Mapping[str, str] | None = None
) -> None:
    env = os.environ if environment is None else environment
    if not execute:
        raise ValueError("generic hardware MVP execution requires --execute")
    if env.get("UPMEM_ALLOW_PHYSICAL_HARDWARE") != "1":
        raise ValueError("UPMEM_ALLOW_PHYSICAL_HARDWARE=1 is required for physical UPMEM execution")
    if env.get("DPU_BACKEND"):
        raise ValueError("DPU_BACKEND must be unset for the physical generic MVP")


def prepare_upmem_hardware_generic_mvp(
    root_dir: Path,
    *,
    suite_path: Path,
    build: bool = False,
    environment: Mapping[str, str] | None = None,
) -> UpmemHardwareGenericMvpPlanResult:
    suite = load_hardware_generic_mvp_suite(suite_path)
    env = dict(os.environ if environment is None else environment)
    plan_dir = _unique_build_dir(root_dir / "build" / "upmem_hardware_generic_mvp_plan")
    plan_dir.mkdir(parents=True, exist_ok=False)
    config_dir = plan_dir / "config"
    config_dir.mkdir()
    shutil.copy2(suite.suite_path, config_dir / "resolved_suite.yml")
    write_json(config_dir / "hardware_profile.json", hardware_generic_mvp_profile_metadata(suite.profile))
    write_json(plan_dir / "environment.json", _hardware_environment(root_dir, env))

    prepared: list[JsonDict] = []
    status, failure_stage = "prepared", None
    for case in suite.cases:
        bridge_dir = plan_dir / "cases" / case.case_id / "bridge"
        try:
            manifest = _prepare_case_bridge(case, bridge_dir, suite)
            prepared.append(
                {
                    "case_id": case.case_id,
                    "input_manifest": str((bridge_dir / "input_manifest.json").relative_to(plan_dir)),
                    "task_id": manifest.task_id,
                    "output_shape": list(manifest.output_shape),
                    "output_tile_count": 2,
                    "status": "prepared",
                }
            )
        except Exception as exc:
            status, failure_stage = "failed", "hardware_profile_violation"
            prepared.append({"case_id": case.case_id, "status": "failed", "error": str(exc)})
            break
    native_build: JsonDict = {"attempted": False, "status": "not_requested"}
    if status == "prepared" and build:
        native_build = _build_hardware_native_source(root_dir, plan_dir, env, suite.profile.timeout_s)
        if native_build["status"] != "passed":
            status = "failed"
            failure_stage = "sdk_discovery_failed" if native_build.get("reason") == "sdk_tools_missing" else "native_build_failed"
    summary_path = plan_dir / "hardware_generic_mvp_plan.json"
    write_json(
        summary_path,
        {
            "schema_version": UPMEM_HARDWARE_GENERIC_MVP_PLAN_SCHEMA_VERSION,
            "status": status,
            "failure_stage": failure_stage,
            "suite_id": suite.suite_id,
            "suite_path": str(suite.suite_path),
            "profile": hardware_generic_mvp_profile_metadata(suite.profile),
            "prepared_cases": prepared,
            "native_build": native_build,
            "dpu_allocation_attempted": False,
            "dpu_launch_attempted": False,
            "notes": [
                "Preparation writes one deterministic generic TaskGraph input and may build isolated native source.",
                "Preparation never allocates or launches a DPU and never creates thesis evidence.",
            ],
        },
    )
    return UpmemHardwareGenericMvpPlanResult(plan_dir, summary_path, status)


def run_upmem_hardware_generic_mvp(
    root_dir: Path,
    *,
    suite_path: Path,
    environment: Mapping[str, str] | None = None,
) -> UpmemHardwareGenericMvpResult:
    env = dict(os.environ if environment is None else environment)
    validate_hardware_generic_execution_request(execute=True, environment=env)
    suite = load_hardware_generic_mvp_suite(suite_path)
    run_dir = create_run_dir(root_dir, suite.suite_id, artifact_kind=EVIDENCE_ARTIFACT_KIND, route_label="upmem_hw_generic")
    shutil.copy2(suite.suite_path, run_dir / "config" / "resolved_suite.yml")
    write_json(run_dir / "config" / "hardware_profile.json", hardware_generic_mvp_profile_metadata(suite.profile))
    write_json(run_dir / "environment.json", _hardware_environment(root_dir, env))
    run_manifest = write_run_manifest(
        run_dir,
        run_kind="upmem_hardware_generic_mvp",
        suite_id=suite.suite_id,
        suite_path=str(suite.suite_path),
        artifact_kind=EVIDENCE_ARTIFACT_KIND,
        route_label="upmem_hw_generic",
        route_id=suite.profile.route_id,
        backend_id=suite.profile.backend_id,
        execution_scope="hardware_bringup_functionality_only",
        evidence_type="physical_hardware_functionality",
        upmem_execution_mode="sdk_hardware_single_dpu",
        artifact_retention="full",
        summary="upmem_hardware_generic_mvp_summary.json",
        root_dir=root_dir,
    )
    records: list[JsonDict] = []
    case_statuses: list[JsonDict] = []
    for case in suite.cases:
        failed = False
        for repeat_id in range(suite.profile.repetitions):
            repeat_dir = run_dir / "cases" / case.case_id / f"repeat_{repeat_id:02d}"
            bridge_dir = repeat_dir / "bridge"
            started = time.perf_counter()
            try:
                _prepare_case_bridge(case, bridge_dir, suite)
                result = execute_generic_bridge(
                    bridge_dir / "input_manifest.json",
                    backend=HARDWARE_GENERIC_MVP_BACKEND_ID,
                    execute_external=True,
                    env=env,
                    timeout_seconds=suite.profile.timeout_s,
                )
                record = _normalized_record(
                    run_dir=run_dir,
                    suite=suite,
                    case=case,
                    repeat_id=repeat_id,
                    result=result,
                    input_manifest_path=bridge_dir / "input_manifest.json",
                    elapsed_s=time.perf_counter() - started,
                    source_commit=run_manifest.get("benchmark_source_commit"),
                )
            except Exception as exc:
                record = _failed_preparation_record(suite, case, repeat_id, str(exc), time.perf_counter() - started, run_manifest.get("benchmark_source_commit"))
            records.append(record)
            if record["status"] != "completed":
                case_statuses.append({"case_id": case.case_id, "status": "failed", "attempted_repeats": repeat_id + 1, "failure_stage": record.get("failure_stage")})
                failed = True
                break
        if not failed:
            case_statuses.append({"case_id": case.case_id, "status": "passed", "attempted_repeats": suite.profile.repetitions})
        if failed:
            break
    completed = bool(records) and all(record.get("status") == "completed" for record in records)
    summary_path = run_dir / "upmem_hardware_generic_mvp_summary.json"
    write_json(
        summary_path,
        {
            "schema_version": UPMEM_HARDWARE_GENERIC_MVP_BENCHMARK_SCHEMA_VERSION,
            "status": "completed" if completed else "failed",
            "suite_id": suite.suite_id,
            "route_id": suite.profile.route_id,
            "backend_id": suite.profile.backend_id,
            "hardware_profile_version": suite.profile.version,
            "row_count": len(records),
            "case_statuses": case_statuses,
            "strict_scope": {
                "synthetic_real_taskgraph_mvp": True,
                "not_real_quantum_circuit": True,
                "requested_dpu_count": 1,
                "tasklets_per_dpu": 1,
                "synchronous_execution": True,
                "output_tile_count": 2,
                "hardware_speedup_applicable": False,
                "no_simulator_fallback": True,
                "no_cpu_fallback": True,
            },
            "normalized_records": "normalized_records.jsonl",
        },
    )
    write_normalized_records(run_dir, records)
    run_manifest.update(
        {
            "summary": summary_path.name,
            "upmem_sdk_available": "verified_by_execution" if completed else "not_verified_by_execution",
            "hardware_available": "verified_by_execution" if completed else "not_verified_by_execution",
        }
    )
    write_json(run_dir / "run_manifest.json", run_manifest)
    return UpmemHardwareGenericMvpResult(run_dir, summary_path, "completed" if completed else "failed", len(records))


def _prepare_case_bridge(case: HardwareGenericMvpCase, bridge_dir: Path, suite: HardwareGenericMvpSuite) -> Any:
    task = _generic_task(case)
    left = TensorValue(TensorSpec("left", (0, 1, 2), (2, 2, 2), "dense", dtype="float32"), case.left_int8.astype(np.float32))
    right = TensorValue(TensorSpec("right", (2, 3, 4), (2, 2, 2), "dense", dtype="float32"), case.right_int8.astype(np.float32))
    preparation = prepare_generic_task(
        GenericTaskPreparationInput(
            task=task,
            left_tensor=left,
            right_tensor=right,
            fixed_point_spec=FixedPointSpec(route_dtype="int8", scale=1.0, complex_policy="reject"),
            caps=GenericTaskPreparationCaps(max_rank=4, max_tensor_elements=16, max_contracted_combinations=2),
            route_id=suite.profile.route_id,
        )
    )
    if preparation.status != "prepared" or preparation.prepared_operands is None:
        raise ValueError(f"hardware_profile_violation: generic preparation failed: {preparation.reason}")
    if not np.array_equal(preparation.prepared_operands.left_quantized, case.left_int8) or not np.array_equal(preparation.prepared_operands.right_quantized, case.right_int8):
        raise ValueError("hardware_profile_violation: identity int8 conversion is not exact")
    bridge_dir.mkdir(parents=True, exist_ok=True)
    manifest = write_generic_bridge_input_manifest(
        preparation,
        bridge_dir,
        backend_id=HARDWARE_GENERIC_MVP_BACKEND_ID,
        execution_target="upmem_hardware",
    )
    manifest_path = bridge_dir / "input_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    native = dict(payload["native_index_metadata"])
    native.update(
        {
            "generic_output_tile_elements": HARDWARE_GENERIC_MVP_OUTPUT_TILE_ELEMENTS,
            "generic_output_tile_count": 2,
            "mram_tiled_task_count": 1,
        }
    )
    payload["native_index_metadata"] = native
    payload["metadata"] = {
        **dict(payload.get("metadata") or {}),
        **hardware_generic_mvp_profile_metadata(suite.profile),
        "generic_output_tile_elements": HARDWARE_GENERIC_MVP_OUTPUT_TILE_ELEMENTS,
        "generic_output_tile_count": 2,
        "mram_tiled_task_count": 1,
        "taskgraph_task_count": 1,
        "taskgraph_completed_task_count": 1,
        "taskgraph_identity_hash": _task_hash(task),
    }
    validate_hardware_generic_mvp_manifest(payload, profile=suite.profile)
    write_json(manifest_path, payload)
    return manifest


def _generic_task(case: HardwareGenericMvpCase) -> ContractionTask:
    return ContractionTask(
        id=f"{case.case_id}_task_0",
        input_tensor_ids=("left", "right"),
        output_tensor_id="output",
        dependencies=(),
        index_expression="abc,cde->abde",
        input_shapes=((2, 2, 2), (2, 2, 2)),
        output_shape=(2, 2, 2, 2),
        left_labels=(0, 1, 2),
        right_labels=(2, 3, 4),
        contracted_labels=(2,),
        output_labels=(0, 1, 3, 4),
        gemm_m=4,
        gemm_k=2,
        gemm_n=4,
        structure="generic",
        estimated_flops=64,
        estimated_bytes=int(case.left_int8.nbytes + case.right_int8.nbytes + 16 * np.dtype("int32").itemsize),
    )


def _normalized_record(
    *,
    run_dir: Path,
    suite: HardwareGenericMvpSuite,
    case: HardwareGenericMvpCase,
    repeat_id: int,
    result: GenericBridgeExecutionResult,
    input_manifest_path: Path,
    elapsed_s: float,
    source_commit: object,
) -> JsonDict:
    output = result.output_manifest
    metadata = dict(output.metadata) if output is not None else {}
    validation = dict(output.validation_metrics) if output is not None else {}
    transfer = dict(metadata.get("application_visible_transfer_bytes") or {})
    status_json = metadata.get("hardware_status_json") if isinstance(metadata.get("hardware_status_json"), Mapping) else {}
    succeeded = (
        result.execution_status == "upmem_sdk_hardware_generic_loop_executed"
        and bool(validation.get("exact_integer_passed"))
        and metadata.get("hardware_kernel_executed") is True
        and metadata.get("native_kernel_executed") is True
        and metadata.get("simulator_kernel_executed") is False
        and metadata.get("cpu_fallback_used") is False
        and status_json.get("success") is True
        and status_json.get("allocation_profile") == HARDWARE_GENERIC_MVP_SDK_ALLOCATION_PROFILE
        and status_json.get("requested_dpus") == 1
        and status_json.get("allocated_dpus") == 1
        and status_json.get("tasklets") == 1
        and transfer.get("total") == int(transfer.get("h2d", -1)) + int(transfer.get("d2h", -1))
    )
    hashes = dict(metadata.get("hashes") or {})
    return {
        "schema_version": UPMEM_HARDWARE_GENERIC_MVP_BENCHMARK_SCHEMA_VERSION,
        "run_id": run_dir.name,
        "suite_id": suite.suite_id,
        "case_id": case.case_id,
        "workload_id": case.case_id,
        "repeat_id": repeat_id,
        "measured_repeat_count": suite.profile.repetitions,
        "route_id": suite.profile.route_id,
        "backend_id": suite.profile.backend_id,
        "backend_family": "upmem_sdk",
        "benchmark_role": "hardware_generic_taskgraph_functionality_mvp",
        "route_role_description": "one synthetic real-valued TaskGraph contraction on one physical DPU",
        "route_limitation_scope": "functionality evidence only; not a quantum circuit benchmark and no speedup, energy, scaling, or scheduler claim",
        "kernel_family": "generic_loop_fallback",
        "execution_model": "binary_tensor_contraction",
        "contraction_execution_target": "upmem",
        "execution_target": "upmem",
        "accelerator_kind": "upmem",
        "upmem_execution_mode": "sdk_hardware_single_dpu",
        "execution_backend": suite.profile.backend_id,
        "target_requested": "hardware",
        "target_observed": "hardware" if succeeded else None,
        "execution_class": suite.profile.execution_class,
        "kernel_strategy": suite.profile.kernel_strategy,
        "hardware_profile_version": suite.profile.version,
        "sdk_allocation_profile": metadata.get("sdk_allocation_profile"),
        "sdk_allocation_profile_verified": metadata.get("sdk_allocation_profile") == HARDWARE_GENERIC_MVP_SDK_ALLOCATION_PROFILE,
        "requested_dpu_count": 1,
        "allocated_dpu_count": status_json.get("allocated_dpus"),
        "tasklets_per_dpu": 1,
        "hardware_allocation_verified": succeeded,
        "native_kernel_executed": succeeded,
        "hardware_kernel_executed": succeeded,
        "simulator_kernel_executed": False,
        "upmem_program_executed": succeeded,
        "cpu_fallback_used": False,
        "simplepim_api_used": False,
        "synthetic_real_taskgraph_mvp": True,
        "not_real_quantum_circuit": True,
        "taskgraph_task_count": 1,
        "taskgraph_completed_task_count": 1 if succeeded else 0,
        "taskgraph_identity_hash": _task_hash(_generic_task(case)),
        "source_task_count": 1,
        "source_task_completion_count": 1 if succeeded else 0,
        "input_dtype": "int8",
        "input_dtype_on_dpu": "int8",
        "accumulator_dtype": "int32",
        "accumulator_dtype_on_dpu": "int32",
        "complex_representation": "real",
        "quantization_mode": "fixed_scale_identity_int8",
        "per_contraction_quantization": False,
        "validation_method": "exact_int8_x_int8_to_int32_cpu_generic_loop_reference",
        "validation_status": "passed" if succeeded else "failed",
        "exact_integer_match": bool(validation.get("exact_integer_passed")),
        "validation_max_abs_error": validation.get("max_abs_error"),
        "max_abs_error": validation.get("max_abs_error"),
        "l2_error": validation.get("l2_error"),
        "generic_output_tile_elements": HARDWARE_GENERIC_MVP_OUTPUT_TILE_ELEMENTS,
        "generic_output_tile_count": 2,
        "mram_resident_operands": True,
        "wram_output_tiled": True,
        "mram_tiled_task_count": 1,
        "application_visible_h2d_bytes": transfer.get("h2d"),
        "application_visible_d2h_bytes": transfer.get("d2h"),
        "application_visible_transfer_bytes": transfer.get("total"),
        "actual_h2d_bytes": transfer.get("h2d"),
        "actual_d2h_bytes": transfer.get("d2h"),
        "actual_transfer_bytes": transfer.get("total"),
        "allocation_time_s": metadata.get("allocation_time_s"),
        "binary_load_time_s": metadata.get("binary_load_time_s"),
        "h2d_time_s": metadata.get("h2d_time_s"),
        "kernel_time_s": metadata.get("kernel_time_s"),
        "d2h_time_s": metadata.get("d2h_time_s"),
        "reconstruction_time_s": metadata.get("reconstruction_time_s"),
        "total_route_time_s": elapsed_s,
        "total_wall_time_s": elapsed_s,
        "timing_scope": "hardware_bringup_functionality_only",
        "timing_is_bringup_only": True,
        "hardware_execution": succeeded,
        "hardware_timing_available": False,
        "hardware_speedup_applicable": False,
        "hardware_functionality_evidence": True,
        "speedup_claim_allowed": False,
        "source_commit": source_commit,
        "hostname": socket.gethostname(),
        "sdk_metadata": metadata.get("sdk_tools"),
        "host_binary_hash": hashes.get("host_binary"),
        "dpu_binary_hash": hashes.get("dpu_binary"),
        "input_hash": hashes.get("input_manifest"),
        "output_hash": hashes.get("output"),
        "input_manifest_artifact": str(input_manifest_path.relative_to(run_dir)),
        "output_manifest_artifact": result.output_manifest_path,
        "failure_stage": None if succeeded else str(metadata.get("failure_stage") or metadata.get("reason") or result.reason),
        "status": "completed" if succeeded else "failed",
        "notes": "No fallback or retry is permitted; physical output is exact int32 functionality evidence only.",
    }


def _failed_preparation_record(suite: HardwareGenericMvpSuite, case: HardwareGenericMvpCase, repeat_id: int, error: str, elapsed_s: float, source_commit: object) -> JsonDict:
    return {
        "schema_version": UPMEM_HARDWARE_GENERIC_MVP_BENCHMARK_SCHEMA_VERSION,
        "suite_id": suite.suite_id,
        "case_id": case.case_id,
        "workload_id": case.case_id,
        "repeat_id": repeat_id,
        "route_id": suite.profile.route_id,
        "backend_id": suite.profile.backend_id,
        "backend_family": "upmem_sdk",
        "benchmark_role": "hardware_generic_taskgraph_functionality_mvp",
        "contraction_execution_target": "upmem",
        "target_requested": "hardware",
        "requested_dpu_count": 1,
        "tasklets_per_dpu": 1,
        "cpu_fallback_used": False,
        "simulator_kernel_executed": False,
        "hardware_execution": False,
        "hardware_speedup_applicable": False,
        "hardware_functionality_evidence": True,
        "validation_status": "failed",
        "timing_scope": "hardware_bringup_functionality_only",
        "timing_is_bringup_only": True,
        "total_route_time_s": elapsed_s,
        "total_wall_time_s": elapsed_s,
        "source_commit": source_commit,
        "hostname": socket.gethostname(),
        "failure_stage": "hardware_profile_violation",
        "status": "failed",
        "error": error,
    }


def _hardware_environment(root_dir: Path, environment: Mapping[str, str]) -> JsonDict:
    return {
        **capture_environment(root_dir),
        "hostname": socket.gethostname(),
        "upmem_hardware_generic_mvp": {
            "sdk_discovery": discover_upmem_sdk(env=environment).to_json_dict(),
            "sdk_allocation_profile": HARDWARE_GENERIC_MVP_SDK_ALLOCATION_PROFILE,
            "UPMEM_PROFILE_present": "UPMEM_PROFILE" in environment,
            "UPMEM_PROFILE_BASE_present": "UPMEM_PROFILE_BASE" in environment,
            "DPU_BACKEND_present": bool(environment.get("DPU_BACKEND")),
            "physical_child_environment_sanitized": True,
            "UPMEM_ALLOW_PHYSICAL_HARDWARE": environment.get("UPMEM_ALLOW_PHYSICAL_HARDWARE") == "1",
            "hardware_timing_available": False,
            "energy_measurement_status": "not_measured",
        },
    }


def _build_hardware_native_source(root_dir: Path, plan_dir: Path, environment: Mapping[str, str], timeout_s: float) -> JsonDict:
    sdk = discover_upmem_sdk(env=environment)
    if not sdk.upmem_sdk_detected or any(not tool.available for tool in sdk.tools if tool.name in {"dpu-upmem-dpurte-clang", "dpu-pkg-config"}):
        return {"attempted": True, "status": "failed", "reason": "sdk_tools_missing", "sdk_discovery": sdk.to_json_dict()}
    build_dir = plan_dir / "native_build"
    shutil.copytree(root_dir / "native" / "upmem" / "simplepim" / "upmem_sdk_generic_loop", build_dir)
    child_env = dict(environment)
    for name in ("UPMEM_PROFILE", "UPMEM_PROFILE_BASE", "DPU_BACKEND"):
        child_env.pop(name, None)
    command = ["make", "clean", "all", "MAX_RANK=4", "MAX_ELEMS=16", "OUTPUT_TILE_ELEMS=8", "NR_TASKLETS=1", "UPMEM_GENERIC_HARDWARE_MVP=1"]
    try:
        completed = subprocess.run(command, cwd=build_dir, env=child_env, capture_output=True, text=True, check=False, timeout=timeout_s)
    except subprocess.TimeoutExpired as exc:
        return {"attempted": True, "status": "failed", "reason": "native_build_timeout", "command": command, "stdout_snippet": _snippet(exc.stdout), "stderr_snippet": _snippet(exc.stderr)}
    return {"attempted": True, "status": "passed" if completed.returncode == 0 else "failed", "reason": None if completed.returncode == 0 else "native_build_failed", "command": command, "returncode": completed.returncode, "stdout_snippet": _snippet(completed.stdout), "stderr_snippet": _snippet(completed.stderr), "dpu_allocation_attempted": False}


def _task_hash(task: ContractionTask) -> str:
    payload = {
        "id": task.id,
        "inputs": task.input_tensor_ids,
        "output": task.output_tensor_id,
        "expression": task.index_expression,
        "input_shapes": task.input_shapes,
        "output_shape": task.output_shape,
        "labels": (task.left_labels, task.right_labels, task.contracted_labels, task.output_labels),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _unique_build_dir(parent: Path) -> Path:
    parent.mkdir(parents=True, exist_ok=True)
    base = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
    candidate, suffix = parent / base, 1
    while candidate.exists():
        candidate = parent / f"{base}_{suffix:02d}"
        suffix += 1
    return candidate


def _snippet(value: object, limit: int = 4000) -> str:
    if value is None:
        return ""
    text = value.decode("utf-8", errors="replace") if isinstance(value, bytes) else str(value)
    return text if len(text) <= limit else text[:limit] + "\n...[truncated]"
