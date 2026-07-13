"""Physical single-DPU dense functionality MVP.

The command in this module is intentionally separate from the general UPMEM
TaskGraph runtime.  It proves only that one thesis-owned int8 dense contraction
can run on one physical DPU and match a CPU int32 reference.  Its records are
bring-up evidence, never speedup evidence.
"""

from __future__ import annotations

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
from quantum_bench.core.records import (
    ContractionTask,
    JsonDict,
    TensorSpec,
    TensorValue,
)
from quantum_bench.environment.capture import capture_environment
from quantum_bench.formats import FixedPointSpec
from quantum_bench.routing.dense_prepare import (
    DenseTaskPreparationInput,
    prepare_dense_task,
)
from quantum_bench.targets.upmem.dense_bridge import (
    DenseBridgeExecutionResult,
    execute_dense_bridge,
    write_dense_bridge_input_manifest,
)
from quantum_bench.targets.upmem.environment import discover_upmem_sdk
from quantum_bench.targets.upmem.hardware_mvp import (
    HARDWARE_MVP_BACKEND_ID,
    HardwareMvpCase,
    HardwareMvpSuite,
    hardware_mvp_profile_metadata,
    load_hardware_mvp_suite,
    validate_hardware_mvp_manifest,
)


UPMEM_HARDWARE_MVP_BENCHMARK_SCHEMA_VERSION = "upmem_hardware_mvp_benchmark_v1"
UPMEM_HARDWARE_MVP_PLAN_SCHEMA_VERSION = "upmem_hardware_mvp_plan_v1"
UPMEM_HARDWARE_MVP_FAILURE_STAGES = frozenset(
    {
        "hardware_opt_in_missing",
        "hardware_profile_violation",
        "sdk_discovery_failed",
        "native_build_failed",
        "hardware_allocation_failed",
        "binary_load_failed",
        "argument_transfer_failed",
        "operand_transfer_failed",
        "kernel_launch_failed",
        "kernel_timeout",
        "result_transfer_failed",
        "output_manifest_failed",
        "output_validation_failed",
        "hardware_release_failed",
    }
)


@dataclass(frozen=True)
class UpmemHardwareMvpResult:
    run_dir: Path
    summary_path: Path
    status: str
    row_count: int


@dataclass(frozen=True)
class UpmemHardwareMvpPlanResult:
    plan_dir: Path
    summary_path: Path
    status: str


def validate_hardware_execution_request(
    *, execute: bool, environment: Mapping[str, str] | None = None
) -> None:
    """Fail closed before creating hardware evidence when opt-in is absent."""

    env = os.environ if environment is None else environment
    if not execute:
        raise ValueError("hardware MVP execution requires --execute")
    if env.get("UPMEM_ALLOW_PHYSICAL_HARDWARE") != "1":
        raise ValueError(
            "UPMEM_ALLOW_PHYSICAL_HARDWARE=1 is required for physical UPMEM execution"
        )
    if env.get("DPU_BACKEND"):
        raise ValueError("DPU_BACKEND must be unset for the physical UPMEM MVP")
    if env.get("UPMEM_PROFILE", "hw") != "hw":
        raise ValueError("UPMEM_PROFILE must be hw for the physical UPMEM MVP")


def prepare_upmem_hardware_mvp(
    root_dir: Path,
    *,
    suite_path: Path,
    build: bool = False,
    environment: Mapping[str, str] | None = None,
) -> UpmemHardwareMvpPlanResult:
    """Write deterministic inputs and optionally build native code without DPU allocation."""

    suite = load_hardware_mvp_suite(suite_path)
    env = dict(os.environ if environment is None else environment)
    plan_dir = _unique_build_dir(root_dir / "build" / "upmem_hardware_mvp_plan")
    plan_dir.mkdir(parents=True, exist_ok=False)
    config_dir = plan_dir / "config"
    config_dir.mkdir()
    shutil.copy2(suite.suite_path, config_dir / "resolved_suite.yml")
    write_json(
        config_dir / "hardware_profile.json",
        hardware_mvp_profile_metadata(suite.profile),
    )

    environment_payload = _hardware_environment(root_dir, env)
    write_json(plan_dir / "environment.json", environment_payload)
    prepared: list[JsonDict] = []
    status = "prepared"
    failure_stage: str | None = None
    for case in suite.cases:
        bridge_dir = plan_dir / "cases" / case.case_id / "bridge"
        try:
            manifest, reference_path = _prepare_case_bridge(
                case, bridge_dir, suite.profile
            )
            prepared.append(
                {
                    "case_id": case.case_id,
                    "input_manifest": str(
                        (bridge_dir / "input_manifest.json").relative_to(plan_dir)
                    ),
                    "expected_accumulator": str(reference_path.relative_to(plan_dir)),
                    "gemm_m": manifest.gemm_m,
                    "gemm_k": manifest.gemm_k,
                    "gemm_n": manifest.gemm_n,
                    "status": "prepared",
                }
            )
        except Exception as exc:
            status = "failed"
            failure_stage = "hardware_profile_violation"
            prepared.append(
                {"case_id": case.case_id, "status": "failed", "error": str(exc)}
            )
            break

    native_build: JsonDict = {"attempted": False, "status": "not_requested"}
    if status == "prepared" and build:
        native_build = _build_hardware_native_source(
            root_dir, plan_dir, env, suite.profile.timeout_s
        )
        if native_build["status"] != "passed":
            status = "failed"
            failure_stage = (
                "sdk_discovery_failed"
                if native_build.get("reason") == "sdk_tools_missing"
                else "native_build_failed"
            )

    payload = {
        "schema_version": UPMEM_HARDWARE_MVP_PLAN_SCHEMA_VERSION,
        "status": status,
        "failure_stage": failure_stage,
        "suite_id": suite.suite_id,
        "suite_path": str(suite.suite_path),
        "profile": hardware_mvp_profile_metadata(suite.profile),
        "prepared_cases": prepared,
        "native_build": native_build,
        "dpu_allocation_attempted": False,
        "dpu_launch_attempted": False,
        "notes": [
            "Preparation writes deterministic bridge inputs and may build an isolated native binary.",
            "Preparation never allocates a DPU, launches a DPU program, or creates thesis evidence.",
        ],
    }
    summary_path = plan_dir / "hardware_mvp_plan.json"
    write_json(summary_path, payload)
    return UpmemHardwareMvpPlanResult(
        plan_dir=plan_dir, summary_path=summary_path, status=status
    )


def run_upmem_hardware_mvp(
    root_dir: Path,
    *,
    suite_path: Path,
    environment: Mapping[str, str] | None = None,
) -> UpmemHardwareMvpResult:
    """Run the fixed physical hardware suite sequentially, without retry/fallback."""

    env = dict(os.environ if environment is None else environment)
    validate_hardware_execution_request(execute=True, environment=env)
    suite = load_hardware_mvp_suite(suite_path)
    run_dir = create_run_dir(
        root_dir,
        suite.suite_id,
        artifact_kind=EVIDENCE_ARTIFACT_KIND,
        route_label="upmem_hw_dense",
    )
    shutil.copy2(suite.suite_path, run_dir / "config" / "resolved_suite.yml")
    write_json(
        run_dir / "config" / "hardware_profile.json",
        hardware_mvp_profile_metadata(suite.profile),
    )
    write_json(run_dir / "environment.json", _hardware_environment(root_dir, env))
    run_manifest = write_run_manifest(
        run_dir,
        run_kind="upmem_hardware_mvp",
        suite_id=suite.suite_id,
        suite_path=str(suite.suite_path),
        artifact_kind=EVIDENCE_ARTIFACT_KIND,
        route_label="upmem_hw_dense",
        route_id=suite.profile.route_id,
        backend_id=suite.profile.backend_id,
        execution_scope="hardware_bringup_functionality_only",
        evidence_type="physical_hardware_functionality",
        upmem_execution_mode="sdk_hardware_single_dpu",
        artifact_retention="full",
        root_dir=root_dir,
    )

    records: list[JsonDict] = []
    case_statuses: list[JsonDict] = []
    stop_after_failure = False
    for case in suite.cases:
        if stop_after_failure:
            case_statuses.append(
                {"case_id": case.case_id, "status": "not_attempted_after_prior_failure"}
            )
            continue
        case_records, case_summary = _run_case(
            root_dir=root_dir,
            run_dir=run_dir,
            suite=suite,
            case=case,
            environment=env,
            source_commit=run_manifest.get("benchmark_source_commit"),
        )
        records.extend(case_records)
        case_statuses.append(case_summary)
        if case_summary["status"] != "passed":
            stop_after_failure = True

    status = (
        "completed"
        if records and all(record.get("status") == "completed" for record in records)
        else "failed"
    )
    summary = {
        "schema_version": UPMEM_HARDWARE_MVP_BENCHMARK_SCHEMA_VERSION,
        "status": status,
        "suite_id": suite.suite_id,
        "route_id": suite.profile.route_id,
        "backend_id": suite.profile.backend_id,
        "hardware_profile_version": suite.profile.version,
        "row_count": len(records),
        "case_statuses": case_statuses,
        "strict_scope": {
            "requested_dpu_count": 1,
            "tasklets_per_dpu": 1,
            "synchronous_execution": True,
            "hardware_speedup_applicable": False,
            "hardware_functionality_evidence": True,
            "no_simulator_fallback": True,
            "no_cpu_fallback": True,
        },
        "normalized_records": "normalized_records.jsonl",
    }
    summary_path = run_dir / "upmem_hardware_mvp_summary.json"
    write_json(summary_path, summary)
    write_normalized_records(run_dir, records)
    return UpmemHardwareMvpResult(
        run_dir=run_dir,
        summary_path=summary_path,
        status=status,
        row_count=len(records),
    )


def _run_case(
    *,
    root_dir: Path,
    run_dir: Path,
    suite: HardwareMvpSuite,
    case: HardwareMvpCase,
    environment: Mapping[str, str],
    source_commit: object,
) -> tuple[list[JsonDict], JsonDict]:
    records: list[JsonDict] = []
    for repeat_id in range(suite.profile.repetitions):
        repeat_dir = run_dir / "cases" / case.case_id / f"repeat_{repeat_id:02d}"
        bridge_dir = repeat_dir / "bridge"
        started = time.perf_counter()
        try:
            manifest, reference_path = _prepare_case_bridge(
                case, bridge_dir, suite.profile
            )
            result = execute_dense_bridge(
                bridge_dir / "input_manifest.json",
                backend=HARDWARE_MVP_BACKEND_ID,
                execute_external=True,
                env=environment,
            )
            record = _normalized_record(
                run_dir=run_dir,
                suite=suite,
                case=case,
                repeat_id=repeat_id,
                result=result,
                input_manifest_path=bridge_dir / "input_manifest.json",
                expected_accumulator_path=reference_path,
                elapsed_s=time.perf_counter() - started,
                source_commit=source_commit,
            )
        except (
            Exception
        ) as exc:  # A local preparation error cannot execute any fallback.
            record = _failed_preparation_record(
                suite=suite,
                case=case,
                repeat_id=repeat_id,
                error=str(exc),
                elapsed_s=time.perf_counter() - started,
                source_commit=source_commit,
            )
        records.append(record)
        if record["status"] != "completed":
            return records, {
                "case_id": case.case_id,
                "status": "failed",
                "attempted_repeats": len(records),
                "failure_stage": record.get("failure_stage"),
            }
    return records, {
        "case_id": case.case_id,
        "status": "passed",
        "attempted_repeats": len(records),
    }


def _prepare_case_bridge(
    case: HardwareMvpCase,
    bridge_dir: Path,
    profile: Any,
) -> tuple[Any, Path]:
    task = _dense_task(case)
    left = TensorValue(
        TensorSpec(
            "left", (0, 1), tuple(case.left_int8.shape), "dense", dtype="float32"
        ),
        case.left_int8.astype(np.float32),
    )
    right = TensorValue(
        TensorSpec(
            "right", (1, 2), tuple(case.right_int8.shape), "dense", dtype="float32"
        ),
        case.right_int8.astype(np.float32),
    )
    preparation = prepare_dense_task(
        DenseTaskPreparationInput(
            task=task,
            left_tensor=left,
            right_tensor=right,
            fixed_point_spec=FixedPointSpec(
                route_dtype="int8", scale=1.0, complex_policy="reject"
            ),
            route_id=profile.route_id,
        )
    )
    if preparation.prepared_operands is None:
        raise ValueError(
            f"hardware_profile_violation: dense preparation failed: {preparation.reason}"
        )
    if not np.array_equal(preparation.prepared_operands.left_quantized, case.left_int8):
        raise ValueError(
            "hardware_profile_violation: left int8 quantization is not exact"
        )
    if not np.array_equal(
        preparation.prepared_operands.right_quantized, case.right_int8
    ):
        raise ValueError(
            "hardware_profile_violation: right int8 quantization is not exact"
        )
    bridge_dir.mkdir(parents=True, exist_ok=True)
    manifest = write_dense_bridge_input_manifest(preparation, bridge_dir)
    manifest_payload = json.loads(
        (bridge_dir / "input_manifest.json").read_text(encoding="utf-8")
    )
    validate_hardware_mvp_manifest(manifest_payload, profile=profile)
    reference_path = bridge_dir / "references" / "expected_accumulator_i32.npy"
    np.save(reference_path, case.expected_accumulator.astype("<i4"), allow_pickle=False)
    manifest_payload["metadata"] = {
        **dict(manifest_payload.get("metadata") or {}),
        **hardware_mvp_profile_metadata(profile),
        "expected_accumulator": {
            "relative_path": reference_path.relative_to(bridge_dir).as_posix(),
            "dtype": "<i4",
            "shape": list(case.expected_accumulator.shape),
            "nbytes": int(case.expected_accumulator.astype("<i4").nbytes),
            "reference_kind": "exact_int8_x_int8_to_int32_cpu_reference",
        },
    }
    write_json(bridge_dir / "input_manifest.json", manifest_payload)
    return manifest, reference_path


def _dense_task(case: HardwareMvpCase) -> ContractionTask:
    return ContractionTask(
        id=f"{case.case_id}_dense_task",
        input_tensor_ids=("left", "right"),
        output_tensor_id="output",
        dependencies=(),
        index_expression="ab,bc->ac",
        input_shapes=(tuple(case.left_int8.shape), tuple(case.right_int8.shape)),
        output_shape=(case.m, case.n),
        left_labels=(0, 1),
        right_labels=(1, 2),
        contracted_labels=(1,),
        output_labels=(0, 2),
        gemm_m=case.m,
        gemm_k=case.k,
        gemm_n=case.n,
        structure="dense",
        estimated_flops=2 * case.m * case.k * case.n,
        estimated_bytes=int(
            case.left_int8.nbytes
            + case.right_int8.nbytes
            + case.expected_accumulator.nbytes
        ),
    )


def _normalized_record(
    *,
    run_dir: Path,
    suite: HardwareMvpSuite,
    case: HardwareMvpCase,
    repeat_id: int,
    result: DenseBridgeExecutionResult,
    input_manifest_path: Path,
    expected_accumulator_path: Path,
    elapsed_s: float,
    source_commit: object,
) -> JsonDict:
    output = result.output_manifest
    metadata = dict(output.metadata) if output is not None else {}
    transfer = dict(metadata.get("application_visible_transfer_bytes") or {})
    validation = dict(output.validation_metrics) if output is not None else {}
    stage = str(
        metadata.get("hardware_stage")
        or metadata.get("reason")
        or result.reason
        or "output_manifest_failed"
    )
    succeeded = result.execution_status == "upmem_sdk_hardware_executed" and bool(
        validation.get("exact_integer_passed")
    )
    native_executed = succeeded and bool(metadata.get("hardware_kernel_executed"))
    status_json = (
        metadata.get("hardware_status_json")
        if isinstance(metadata.get("hardware_status_json"), dict)
        else {}
    )
    allocated = (
        status_json.get("allocated_dpus") if isinstance(status_json, dict) else None
    )
    record: JsonDict = {
        "schema_version": UPMEM_HARDWARE_MVP_BENCHMARK_SCHEMA_VERSION,
        "run_id": run_dir.name,
        "suite_id": suite.suite_id,
        "case_id": case.case_id,
        "workload_id": case.case_id,
        "repeat_id": repeat_id,
        "measured_repeat_count": suite.profile.repetitions,
        "n_qubits": None,
        "route_id": suite.profile.route_id,
        "backend_id": suite.profile.backend_id,
        "backend_family": "upmem_sdk",
        "benchmark_role": "hardware_functionality_mvp",
        "route_role_description": "single-DPU int8 dense contraction physical-hardware bring-up",
        "route_limitation_scope": "functionality evidence only; no speedup, energy, scaling, or scheduler claim",
        "kernel_family": "dense_gemm",
        "execution_model": "binary_tensor_contraction",
        "contraction_execution_target": "upmem",
        "execution_target": "upmem",
        "accelerator_kind": "upmem",
        "upmem_execution_mode": "sdk_hardware_single_dpu",
        "execution_backend": suite.profile.backend_id,
        "target_requested": "hardware",
        "target_observed": "hardware" if native_executed else None,
        "execution_class": suite.profile.execution_class,
        "kernel_strategy": suite.profile.kernel_strategy,
        "hardware_profile_version": suite.profile.version,
        "requested_dpu_count": suite.profile.requested_dpu_count,
        "allocated_dpu_count": allocated,
        "tasklets_per_dpu": suite.profile.tasklets_per_dpu,
        "hardware_allocation_verified": allocated == 1 and native_executed,
        "native_kernel_executed": native_executed,
        "hardware_kernel_executed": native_executed,
        "simulator_kernel_executed": False,
        "upmem_program_executed": native_executed,
        "cpu_fallback_used": False,
        "simplepim_api_used": False,
        "input_dtype": "int8",
        "input_dtype_on_dpu": "int8",
        "accumulator_dtype": "int32",
        "accumulator_dtype_on_dpu": "int32",
        "complex_representation": "real",
        "quantization_mode": "fixed_scale_identity_int8",
        "per_contraction_quantization": False,
        "validation_method": "exact_int8_x_int8_to_int32_cpu_reference",
        "validation_status": "passed" if succeeded else "failed",
        "exact_integer_match": bool(validation.get("exact_integer_passed")),
        "validation_max_abs_error": validation.get("max_abs_error"),
        "max_abs_error": validation.get("max_abs_error"),
        "l2_error": validation.get("l2_error"),
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
        "total_route_time_s": elapsed_s,
        "total_wall_time_s": elapsed_s,
        "timing_scope": "hardware_bringup_functionality_only",
        "timing_is_bringup_only": True,
        "hardware_execution": native_executed,
        "hardware_timing_available": False,
        "hardware_speedup_applicable": False,
        "hardware_functionality_evidence": True,
        "speedup_claim_allowed": False,
        "source_commit": source_commit,
        "hostname": socket.gethostname(),
        "sdk_metadata": metadata.get("sdk_metadata"),
        "compiler_metadata": metadata.get("compiler_metadata"),
        "host_binary_hash": (metadata.get("hashes") or {}).get("host_binary"),
        "dpu_binary_hash": (metadata.get("hashes") or {}).get("dpu_binary"),
        "input_hash": (metadata.get("hashes") or {}).get("input_manifest"),
        "output_hash": (metadata.get("hashes") or {}).get("output"),
        "input_manifest_artifact": str(input_manifest_path.relative_to(run_dir)),
        "expected_accumulator_artifact": str(
            expected_accumulator_path.relative_to(run_dir)
        ),
        "output_manifest_artifact": result.output_manifest_path,
        "failure_stage": None if succeeded else stage,
        "status": "completed" if succeeded else "failed",
        "notes": "No fallback or retry is permitted for the physical hardware MVP.",
    }
    return record


def _failed_preparation_record(
    *,
    suite: HardwareMvpSuite,
    case: HardwareMvpCase,
    repeat_id: int,
    error: str,
    elapsed_s: float,
    source_commit: object,
) -> JsonDict:
    return {
        "schema_version": UPMEM_HARDWARE_MVP_BENCHMARK_SCHEMA_VERSION,
        "suite_id": suite.suite_id,
        "case_id": case.case_id,
        "workload_id": case.case_id,
        "repeat_id": repeat_id,
        "route_id": suite.profile.route_id,
        "backend_id": suite.profile.backend_id,
        "backend_family": "upmem_sdk",
        "benchmark_role": "hardware_functionality_mvp",
        "contraction_execution_target": "upmem",
        "target_requested": "hardware",
        "execution_class": suite.profile.execution_class,
        "hardware_profile_version": suite.profile.version,
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
    captured = capture_environment(root_dir)
    sdk = discover_upmem_sdk(env=environment).to_json_dict()
    return {
        **captured,
        "hostname": socket.gethostname(),
        "upmem_hardware_mvp": {
            "sdk_discovery": sdk,
            "UPMEM_PROFILE": environment.get("UPMEM_PROFILE", "hw"),
            "DPU_BACKEND_present": bool(environment.get("DPU_BACKEND")),
            "UPMEM_ALLOW_PHYSICAL_HARDWARE": environment.get(
                "UPMEM_ALLOW_PHYSICAL_HARDWARE"
            )
            == "1",
            "hardware_timing_available": False,
            "energy_measurement_status": "not_measured",
        },
    }


def _build_hardware_native_source(
    root_dir: Path, plan_dir: Path, environment: Mapping[str, str], timeout_s: float
) -> JsonDict:
    sdk = discover_upmem_sdk(env=environment)
    if not sdk.upmem_sdk_detected or any(
        not tool.available
        for tool in sdk.tools
        if tool.name in {"dpu-upmem-dpurte-clang", "dpu-pkg-config"}
    ):
        return {
            "attempted": True,
            "status": "failed",
            "reason": "sdk_tools_missing",
            "sdk_discovery": sdk.to_json_dict(),
        }
    source = root_dir / "native" / "upmem" / "simplepim" / "upmem_sdk_dense"
    build_dir = plan_dir / "native_build"
    shutil.copytree(source, build_dir)
    child_env = dict(environment)
    child_env.pop("DPU_BACKEND", None)
    child_env["UPMEM_PROFILE"] = "hw"
    command = [
        "make",
        "clean",
        "all",
        "MAX_DIM=4",
        "L2_MAX_DIM=4",
        "L2_TILE_MAX_DIM=4",
        "NR_TASKLETS=1",
        "UPMEM_DENSE_HARDWARE_MVP=1",
    ]
    try:
        completed = subprocess.run(
            command,
            cwd=build_dir,
            env=child_env,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "attempted": True,
            "status": "failed",
            "reason": "native_build_timeout",
            "command": command,
            "stdout_snippet": _snippet(exc.stdout),
            "stderr_snippet": _snippet(exc.stderr),
        }
    return {
        "attempted": True,
        "status": "passed" if completed.returncode == 0 else "failed",
        "reason": None if completed.returncode == 0 else "native_build_failed",
        "command": command,
        "returncode": completed.returncode,
        "stdout_snippet": _snippet(completed.stdout),
        "stderr_snippet": _snippet(completed.stderr),
        "dpu_allocation_attempted": False,
    }


def _unique_build_dir(parent: Path) -> Path:
    parent.mkdir(parents=True, exist_ok=True)
    base = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
    candidate = parent / base
    suffix = 1
    while candidate.exists():
        candidate = parent / f"{base}_{suffix:02d}"
        suffix += 1
    return candidate


def _snippet(value: object, limit: int = 4000) -> str:
    if value is None:
        return ""
    text = (
        value.decode("utf-8", errors="replace")
        if isinstance(value, bytes)
        else str(value)
    )
    return text if len(text) <= limit else text[:limit] + "\n...[truncated]"
