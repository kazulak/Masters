"""Block 2 adapter for the bounded SimplePIM execution-plan host.

Block 1 owns the JSON execution plan and the binary schedule.  This module
only validates those artifacts, finds the resident package manifest emitted by
Block 1, and invokes the native host once for the request.  It deliberately has
no simulator or CPU execution path.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import struct
import subprocess
import time
from typing import Any, Mapping

from quantum_bench.core.records import to_jsonable
from quantum_bench.targets.upmem.execution_plan_v1 import (
    ExecutionPlan,
    parse_plan_json,
    validate_schedule,
)


IMPLEMENTATION_ROOT = Path(__file__).resolve().parents[4]
NATIVE_SOURCE_DIR = IMPLEMENTATION_ROOT / "native" / "upmem" / "simplepim"
EXECUTION_PLAN_SOURCE_DIR = NATIVE_SOURCE_DIR / "upmem_sdk_execution_plan"
SIMPLEPIM_ROOT = IMPLEMENTATION_ROOT / "external" / "SimplePIM"
HOST_BINARY_NAME = "host_upmem_execution_plan"
DPU_BINARY_NAME = "dpu_resident"
NATIVE_RESPONSE_SCHEMA = "upmem_execution_plan_native_v1"
ADAPTER_SESSION_SCHEMA = "upmem_execution_plan_adapter_session_v1"
NATIVE_BACKEND_ID = "upmem_sdk_hardware_execution_plan"
DEVICE_LAUNCH_MODE = "asynchronous_per_dpu"
SYNCHRONIZATION_POLICY = "synchronous_wave_barriers"
BUILD_TIMEOUT_S = 180.0
MAX_TIMEOUT_S = 24 * 60 * 60


class NativeAdapterError(RuntimeError):
    """A fail-closed Block 2 error with a route-compatible failure stage."""

    def __init__(self, stage: str, message: str) -> None:
        self.failure_stage = stage
        super().__init__(f"{stage}: {message}")


@dataclass(frozen=True)
class _Build:
    build_dir: Path
    host_binary: Path
    dpu_binary: Path
    command: tuple[str, ...]


_LAST_BUILD: _Build | None = None


def build(build_dir: Path, prepare_only: bool = True) -> dict[str, Any]:
    """Build the host/DPU pair without allocating or launching hardware."""

    if not isinstance(prepare_only, bool):
        raise TypeError("prepare_only must be a bool")
    destination = Path(build_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    if not EXECUTION_PLAN_SOURCE_DIR.is_dir():
        raise NativeAdapterError("native_build_failed", "execution-plan source tree is missing")
    if not SIMPLEPIM_ROOT.is_dir():
        raise NativeAdapterError("sdk_discovery_failed", "external SimplePIM source tree is missing")

    # The Makefile intentionally keeps the execution-plan source tree
    # relocatable.  Copy the three sibling trees it references so all produced
    # binaries remain inside the run directory passed by Block 3.
    source_root = destination / "source" / "native" / "upmem" / "simplepim"
    for name in (
        "upmem_sdk_execution_plan",
        "upmem_sdk_generic_loop_resident",
        "upmem_sdk_generic_loop_frontier_two_dpu",
    ):
        shutil.copytree(NATIVE_SOURCE_DIR / name, source_root / name, dirs_exist_ok=True)

    make = shutil.which("make")
    if make is None:
        raise NativeAdapterError("sdk_discovery_failed", "make is not available")
    source = source_root / "upmem_sdk_execution_plan"
    command = (
        make,
        "clean",
        "all",
        f"SIMPLEPIM_ROOT={SIMPLEPIM_ROOT}",
        "NR_TASKLETS=1",
        "UPMEM_GENERIC_HARDWARE_MVP=1",
    )
    try:
        completed = subprocess.run(
            command,
            cwd=source,
            env=dict(os.environ),
            capture_output=True,
            text=True,
            timeout=BUILD_TIMEOUT_S,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise NativeAdapterError("native_build_failed", "native build timed out") from exc
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "native build command failed").strip()
        raise NativeAdapterError("native_build_failed", detail[-4000:])

    host_binary = source / "bin" / HOST_BINARY_NAME
    dpu_binary = source / "bin" / DPU_BINARY_NAME
    if not host_binary.is_file() or not dpu_binary.is_file():
        raise NativeAdapterError("native_build_failed", "native build did not produce host and DPU binaries")
    global _LAST_BUILD
    _LAST_BUILD = _Build(destination, host_binary.resolve(), dpu_binary.resolve(), tuple(command))
    return {
        "status": "built",
        "prepare_only": prepare_only,
        "allocation_attempted": False,
        "launch_attempted": False,
        "build_dir": str(destination),
        "native_source_dir": str(source),
        "host_binary": str(host_binary.resolve()),
        "dpu_binary": str(dpu_binary.resolve()),
        "build_command": list(command),
        "stdout": (completed.stdout or "")[-4000:],
        "stderr": (completed.stderr or "")[-4000:],
    }


def execute(request_path: Path, timeout_s: float) -> dict[str, Any]:
    """Execute one validated placement through one native host process."""

    request_file = Path(request_path).resolve()
    if not request_file.is_file():
        raise NativeAdapterError("manifest_parse_failed", "Block 2 request is missing")
    timeout = _positive_timeout(timeout_s)
    _require_physical_environment()
    request = _read_object(request_file, "Block 2 request")
    plan, schedule_path, package_path, manifest_path, dpu_path = _load_request(
        request_file, request
    )
    host_binary = _host_binary_for(dpu_path)

    response_path = request_file.parent / (
        f".native_execution_response_{os.getpid()}_{time.time_ns()}.json"
    )
    final_output_path = request_file.parent / "native_final_output.bin"
    for stale in (response_path, final_output_path):
        try:
            stale.unlink()
        except FileNotFoundError:
            pass

    command = (
        str(host_binary),
        "--execute-plan",
        "--resident-package",
        str(manifest_path),
        "--schedule",
        str(schedule_path),
        "--response",
        str(response_path),
        "--warmups",
        str(request["requested_warmups"]),
        "--repetitions",
        str(request["requested_repetitions"]),
        "--timeout-s",
        str(max(1, int(math.ceil(timeout)))),
    )
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            cwd=host_binary.parent.parent,
            env=dict(os.environ),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise NativeAdapterError("kernel_timeout", "native host process timed out") from exc
    elapsed_s = time.perf_counter() - started

    if not response_path.is_file():
        raise NativeAdapterError("output_manifest_failed", "native response is missing")
    try:
        response = _read_object(response_path, "native response")
    except NativeAdapterError:
        raise
    if completed.returncode != 0:
        stage = str(response.get("failure_stage") or "kernel_launch_failed")
        raise NativeAdapterError(stage, str(response.get("error") or "native host returned nonzero"))
    if response.get("status") != "completed":
        raise NativeAdapterError(
            str(response.get("failure_stage") or "kernel_launch_failed"),
            str(response.get("error") or "native execution did not complete"),
        )

    _validate_native_response(
        response,
        request=request,
        plan=plan,
        package_path=package_path,
        schedule_path=schedule_path,
        dpu_path=dpu_path,
        manifest_path=manifest_path,
    )
    final_output = _stage_native_output(
        request_file.parent,
        manifest_path,
        plan,
        final_output_path,
    )
    return _normalize_session(
        response,
        request=request,
        plan=plan,
        elapsed_s=elapsed_s,
        response_path=response_path,
        final_output_path=final_output,
        command=command,
    )


def validate(request_path: Path, timeout_s: float) -> dict[str, Any]:
    """Validate one request through the native parser without touching hardware."""

    request_file = Path(request_path).resolve()
    if not request_file.is_file():
        raise NativeAdapterError("manifest_parse_failed", "Block 2 request is missing")
    timeout = _positive_timeout(timeout_s)
    request = _read_object(request_file, "Block 2 request")
    plan, schedule_path, _package_path, manifest_path, dpu_path = _load_request(
        request_file, request
    )
    host_binary = _host_binary_for(dpu_path)
    response_path = request_file.parent / (
        f".native_plan_validation_{os.getpid()}_{time.time_ns()}.json"
    )
    try:
        response_path.unlink()
    except FileNotFoundError:
        pass
    command = (
        str(host_binary),
        "--validate-plan",
        "--resident-package",
        str(manifest_path),
        "--schedule",
        str(schedule_path),
        "--response",
        str(response_path),
        "--warmups",
        str(request["requested_warmups"]),
        "--repetitions",
        str(request["requested_repetitions"]),
        "--timeout-s",
        str(max(1, int(math.ceil(timeout)))),
    )
    try:
        completed = subprocess.run(
            command,
            cwd=host_binary.parent.parent,
            env=dict(os.environ),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise NativeAdapterError("kernel_timeout", "native plan validation timed out") from exc
    if not response_path.is_file():
        raise NativeAdapterError("output_manifest_failed", "native validation response is missing")
    response = _read_object(response_path, "native response")
    if completed.returncode != 0 or response.get("status") != "validated":
        stage = str(response.get("failure_stage") or "execution_plan_compile_failed")
        raise NativeAdapterError(
            stage,
            str(response.get("error") or "native plan validation failed"),
        )
    _validate_native_validation_response(response, plan=plan)
    return {
        **response,
        # The native host reports its parser state as ``validated``.  The
        # Python adapter contract uses ``passed`` for a successful dry-run so
        # the route can distinguish it from a native execution session.
        "status": "passed",
        "native_validation_status": "validated",
        "allocation_attempted": False,
        "launch_attempted": False,
        "dpu_allocation_attempted": False,
        "dpu_launch_attempted": False,
        "native_response_path": str(response_path),
        "native_command": list(command),
    }


def _load_request(
    request_file: Path, request: Mapping[str, Any]
) -> tuple[ExecutionPlan, Path, Path, Path, Path]:
    _require(request, "schema_version", "upmem_execution_plan_request_v1")
    _require(request, "manifest_kind", "upmem_execution_plan_request")
    plan_path = request_file.parent / "execution_plan.json"
    try:
        plan = parse_plan_json(plan_path.read_bytes())
    except (OSError, ValueError) as exc:
        raise NativeAdapterError("execution_plan_compile_failed", str(exc)) from exc

    schedule_path = _resolve_ref(request_file.parent, request.get("schedule_path"), "schedule")
    package_ref = request.get("package_path")
    package_path = _resolve_ref(request_file.parent, package_ref, "resident package")
    if package_path.suffix.lower() == ".json":
        package_manifest = _read_object(package_path, "resident package manifest")
        package_path = _manifest_package_path(package_path, package_manifest)
    if not package_path.is_file():
        raise NativeAdapterError("package_preparation_failed", "resident package binary is missing")
    if not schedule_path.is_file():
        raise NativeAdapterError("execution_plan_compile_failed", "execution schedule is missing")
    package_bytes = _read_bytes(package_path, "resident package")
    schedule_bytes = _read_bytes(schedule_path, "execution schedule")
    package_hash = _sha256(package_bytes)
    schedule_hash = _sha256(schedule_bytes)
    if request.get("package_file_sha256") != package_hash:
        raise NativeAdapterError("package_preparation_failed", "request package hash does not match bytes")
    if request.get("schedule_sidecar_sha256") != schedule_hash:
        raise NativeAdapterError("execution_plan_compile_failed", "request schedule hash does not match bytes")
    if plan.package_file_sha256 != package_hash or plan.schedule_sidecar_sha256 != schedule_hash:
        raise NativeAdapterError("execution_plan_compile_failed", "plan/package/schedule hash binding mismatch")
    try:
        validate_schedule(schedule_bytes, plan, package_bytes=package_bytes)
    except ValueError as exc:
        raise NativeAdapterError("execution_plan_compile_failed", str(exc)) from exc

    _validate_request_binding(request, plan)
    manifest_path = _find_resident_manifest(request_file, request, package_path)
    manifest = _read_object(manifest_path, "resident package manifest")
    if _manifest_package_path(manifest_path, manifest) != package_path.resolve():
        raise NativeAdapterError("package_preparation_failed", "resident manifest package binding mismatch")
    dpu_path = _resolve_ref(request_file.parent, request.get("dpu_binary"), "DPU binary")
    manifest_dpu = manifest.get("dpu_binary")
    if not isinstance(manifest_dpu, str) or not manifest_dpu:
        raise NativeAdapterError("package_preparation_failed", "resident manifest DPU binding is missing")
    if (manifest_path.parent / manifest_dpu).resolve() != dpu_path:
        raise NativeAdapterError("package_preparation_failed", "resident manifest DPU binding mismatch")
    if not dpu_path.is_file():
        raise NativeAdapterError("native_build_failed", "built DPU binary is missing")
    return plan, schedule_path, package_path, manifest_path, dpu_path


def _validate_request_binding(request: Mapping[str, Any], plan: ExecutionPlan) -> None:
    integer_fields = {
        "requested_dpu_count": plan.requested_dpu_count,
        "tasklets_per_dpu": plan.tasklets_per_dpu,
    }
    for key, expected in integer_fields.items():
        if request.get(key) != expected:
            raise NativeAdapterError("execution_plan_compile_failed", f"request {key} differs from plan")
    if request.get("execution_plan_hash") != plan.execution_plan_hash:
        raise NativeAdapterError("execution_plan_compile_failed", "request execution-plan hash differs from plan")
    if request.get("upmem_execution_plan_hash") not in (None, plan.execution_plan_hash):
        raise NativeAdapterError("execution_plan_compile_failed", "request execution-plan identity differs from plan")
    if request.get("schedule_h2d_bytes") not in (None, 0):
        raise NativeAdapterError("hardware_profile_violation", "execution schedule must remain host metadata")
    for identity_name, values in (
        (
            "source_identity",
            {
                "circuit_semantics_hash": plan.source_circuit_semantics_hash,
                "tensor_network_hash": plan.source_tensor_network_hash,
                "contraction_plan_hash": plan.source_contraction_plan_hash,
            },
        ),
        (
            "package_identity",
            {
                "circuit_semantics_hash": plan.package_circuit_semantics_hash,
                "tensor_network_hash": plan.package_tensor_network_hash,
                "contraction_plan_hash": plan.package_contraction_plan_hash,
            },
        ),
    ):
        if request.get(identity_name) != values:
            raise NativeAdapterError("execution_plan_compile_failed", f"request {identity_name} differs from plan")
    for prefix, values in (("source", request.get("source_identity")), ("package", request.get("package_identity"))):
        if not isinstance(values, Mapping):
            raise NativeAdapterError("execution_plan_compile_failed", f"request {prefix} identity is missing")
        for key, value in values.items():
            if request.get(f"{prefix}_{key}") != value:
                raise NativeAdapterError("execution_plan_compile_failed", f"request flattened {prefix} identity differs")
    if request.get("final_outputs") != [to_jsonable(item) for item in plan.final_outputs]:
        raise NativeAdapterError("execution_plan_compile_failed", "request final output binding differs from plan")
    for key, positive in (("requested_warmups", False), ("requested_repetitions", True)):
        value = request.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or (positive and value <= 0) or value < 0:
            raise NativeAdapterError("execution_plan_compile_failed", f"request {key} is invalid")
    if request["requested_warmups"] > 1 or request["requested_repetitions"] > 16:
        raise NativeAdapterError("execution_plan_compile_failed", "request repetition cap exceeded")


def _find_resident_manifest(
    request_file: Path, request: Mapping[str, Any], package_path: Path
) -> Path:
    explicit_keys = (
        "resident_manifest_path",
        "package_manifest_path",
        "resident_package_manifest",
    )
    candidates: list[Path] = []
    for key in explicit_keys:
        value = request.get(key)
        if isinstance(value, str) and value:
            candidates.append(_resolve_ref(request_file.parent, value, "resident manifest"))
    if package_path.suffix.lower() == ".json":
        candidates.append(package_path)
    for parent in (package_path.parent, *package_path.parents, request_file.parent, *request_file.parent.parents):
        candidates.extend(sorted(parent.glob("*_resident_request.json")))
    seen: set[Path] = set()
    matches: list[Path] = []
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen or not candidate.is_file():
            continue
        seen.add(candidate)
        try:
            payload = _read_object(candidate, "resident package manifest")
            if _manifest_package_path(candidate, payload) == package_path.resolve():
                matches.append(candidate)
        except NativeAdapterError:
            continue
    if not matches:
        raise NativeAdapterError("package_preparation_failed", "resident package manifest is missing")
    if len(matches) > 1:
        request_id = request.get("request_id")
        exact = [item for item in matches if item.stem.startswith(str(request_id or ""))]
        if len(exact) == 1:
            return exact[0]
        raise NativeAdapterError("package_preparation_failed", "resident package manifest is ambiguous")
    return matches[0]


def _validate_native_response(
    response: Mapping[str, Any],
    *,
    request: Mapping[str, Any],
    plan: ExecutionPlan,
    package_path: Path,
    schedule_path: Path,
    dpu_path: Path,
    manifest_path: Path,
) -> None:
    if response.get("schema_version") != NATIVE_RESPONSE_SCHEMA:
        raise NativeAdapterError("output_manifest_failed", "native response schema is invalid")
    if response.get("target_requested") != "hardware" or response.get("target_observed") != "physical_hardware":
        raise NativeAdapterError("output_manifest_failed", "native response is not physical hardware evidence")
    expected_flags = {
        "hardware_allocation_verified": True,
        "native_kernel_executed": True,
        "simulator_kernel_executed": False,
        "hardware_kernel_executed": True,
        "cpu_fallback_used": False,
        "hardware_speedup_applicable": False,
        "timing_is_bringup_only": True,
        "hardware_functionality_evidence": True,
    }
    for key, expected in expected_flags.items():
        if response.get(key) is not expected:
            raise NativeAdapterError("output_manifest_failed", f"native response {key} is unsafe")
    if response.get("backend_id") != NATIVE_BACKEND_ID:
        raise NativeAdapterError("output_manifest_failed", "native backend identity is invalid")
    if response.get("requested_dpu_count") != plan.requested_dpu_count or response.get("allocated_dpu_count") != plan.requested_dpu_count:
        raise NativeAdapterError("output_manifest_failed", "native DPU count does not match request")
    if response.get("tasklets_per_dpu") != plan.tasklets_per_dpu:
        raise NativeAdapterError("output_manifest_failed", "native tasklet count does not match request")
    allocation = response.get("allocation")
    if not isinstance(allocation, Mapping) or allocation.get("attempted") is not True or allocation.get("release_confirmed") is not True or not _allocation_succeeded(response):
        raise NativeAdapterError("output_manifest_failed", "native allocation/release evidence is incomplete")
    if response.get("requested_warmups") != request["requested_warmups"] or response.get("requested_repetitions") != request["requested_repetitions"]:
        raise NativeAdapterError("output_manifest_failed", "native repetition request differs")
    package_hash = _sha256(_read_bytes(package_path, "resident package"))
    schedule_hash = _sha256(_read_bytes(schedule_path, "execution schedule"))
    if response.get("package_file_sha256") != package_hash or response.get("schedule_sidecar_sha256") != schedule_hash:
        raise NativeAdapterError("output_manifest_failed", "native package or schedule hash is stale")
    if request.get("package_file_sha256") != package_hash or request.get("schedule_sidecar_sha256") != schedule_hash:
        raise NativeAdapterError("output_manifest_failed", "request package or schedule hash is stale")
    if response.get("dpu_binary_sha256") != _sha256(_read_bytes(dpu_path, "DPU binary")):
        raise NativeAdapterError("output_manifest_failed", "native DPU binary hash is stale")
    _validate_native_assignments(response.get("operation_assignments"), plan)
    _validate_native_transfers(response.get("cross_dpu_transfers"), plan)
    _validate_native_metrics(response.get("metrics"), plan, request)
    if _manifest_package_path(manifest_path, _read_object(manifest_path, "resident package manifest")) != package_path.resolve():
        raise NativeAdapterError("output_manifest_failed", "native manifest package binding changed")


def _validate_native_validation_response(
    response: Mapping[str, Any], *, plan: ExecutionPlan
) -> None:
    if response.get("schema_version") != NATIVE_RESPONSE_SCHEMA:
        raise NativeAdapterError("output_manifest_failed", "native validation response schema is invalid")
    expected = {
        "status": "validated",
        "target_requested": "hardware",
        "target_observed": "not_allocated",
        "backend_id": NATIVE_BACKEND_ID,
        "allocated_dpu_count": 0,
        "tasklets_per_dpu": plan.tasklets_per_dpu,
        "hardware_allocation_verified": False,
        "native_kernel_executed": False,
        "simulator_kernel_executed": False,
        "hardware_kernel_executed": False,
        "cpu_fallback_used": False,
        "hardware_speedup_applicable": False,
        "timing_is_bringup_only": True,
        "hardware_functionality_evidence": False,
        "validation_status": "plan_valid",
        "package_file_sha256": plan.package_file_sha256,
        "schedule_sidecar_sha256": plan.schedule_sidecar_sha256,
    }
    for key, expected_value in expected.items():
        if response.get(key) != expected_value:
            raise NativeAdapterError(
                "output_manifest_failed",
                f"native validation response {key} is invalid",
            )
    if response.get("requested_dpu_count") != plan.requested_dpu_count:
        raise NativeAdapterError("output_manifest_failed", "native validation DPU count differs from plan")
    allocation = response.get("allocation")
    if not isinstance(allocation, Mapping) or any(
        allocation.get(key) is not False
        for key in ("attempted", "confirmed", "release_confirmed")
    ):
        raise NativeAdapterError("output_manifest_failed", "native validation attempted hardware allocation")


def _validate_native_assignments(value: Any, plan: ExecutionPlan) -> None:
    if not isinstance(value, list) or len(value) != plan.operation_count:
        raise NativeAdapterError("output_manifest_failed", "native operation assignment count is invalid")
    expected = {
        item.package_operation_index: item
        for item in plan.assignments
    }
    observed: set[int] = set()
    for item in value:
        if not isinstance(item, Mapping):
            raise NativeAdapterError("output_manifest_failed", "native operation assignment is malformed")
        package_index = item.get("package_operation_index")
        if not isinstance(package_index, int) or package_index in observed or package_index not in expected:
            raise NativeAdapterError("output_manifest_failed", "native operation assignment identity is invalid")
        assignment = expected[package_index]
        checks = {
            "operation_id": assignment.operation_id,
            "component": "real",
            "wave_index": assignment.wave_index,
            "dpu_id": assignment.dpu_id,
            "dependency_bitmask": assignment.dependency_bitmask,
            "input_slot_ids": list(assignment.input_slot_ids),
            "output_slot_id": assignment.output_slot_id,
        }
        for key, expected_value in checks.items():
            if item.get(key) != expected_value:
                raise NativeAdapterError("output_manifest_failed", f"native assignment {key} differs from schedule")
        if item.get("task_id") not in (None, assignment.task_id):
            raise NativeAdapterError("output_manifest_failed", "native assignment task identity differs")
        observed.add(package_index)


def _validate_native_transfers(value: Any, plan: ExecutionPlan) -> None:
    if not isinstance(value, list):
        raise NativeAdapterError("output_manifest_failed", "native transfer list is missing")
    expected = {
        (
            item.producer_operation_id,
            item.consumer_operation_id,
            item.producer_dpu_id,
            item.consumer_dpu_id,
            item.slot_id,
            item.transfer_bytes,
        )
        for item in plan.transfer_edges
    }
    observed: set[tuple[int, int, int, int, int, int]] = set()
    for item in value:
        if not isinstance(item, Mapping) or item.get("transport") != "host_mediated_v1":
            raise NativeAdapterError("output_manifest_failed", "native transfer transport is invalid")
        key = tuple(item.get(name) for name in (
            "producer_operation_id", "consumer_operation_id", "producer_dpu_id",
            "consumer_dpu_id", "slot_id", "transfer_bytes",
        ))
        if len(key) != 6 or any(not isinstance(part, int) for part in key) or key not in expected or key in observed:
            raise NativeAdapterError("output_manifest_failed", "native transfer differs from schedule")
        observed.add(key)
    if observed != expected:
        raise NativeAdapterError("output_manifest_failed", "native transfer set is incomplete")


def _validate_native_metrics(value: Any, plan: ExecutionPlan, request: Mapping[str, Any]) -> None:
    if not isinstance(value, Mapping):
        raise NativeAdapterError("output_manifest_failed", "native metrics are missing")
    total_iterations = request["requested_warmups"] + request["requested_repetitions"]
    expected_counts = [
        sum(item.dpu_id == dpu for item in plan.assignments) * total_iterations
        for dpu in range(plan.requested_dpu_count)
    ]
    completed_per_dpu = value.get("completed_per_dpu")
    if completed_per_dpu != expected_counts:
        raise NativeAdapterError("output_manifest_failed", "native completion counts differ from schedule")
    expected_total = plan.logical_task_count * total_iterations
    if sum(completed_per_dpu) != expected_total:
        raise NativeAdapterError("output_manifest_failed", "native aggregate completion total differs from schedule")
    for key, expected in (
        ("launch_count", expected_total),
        ("synchronize_count", expected_total),
        ("completion_reads", expected_total),
        ("cross_dpu_edge_count", len(plan.transfer_edges) * total_iterations),
    ):
        if value.get(key) != expected:
            raise NativeAdapterError("output_manifest_failed", f"native metric {key} differs from schedule")
    integer_keys = (
        "descriptor_h2d_bytes", "operand_h2d_bytes", "reset_h2d_bytes",
        "cross_d2h_bytes", "cross_h2d_bytes", "final_d2h_bytes",
        "actual_h2d_bytes", "actual_d2h_bytes", "actual_transfer_bytes",
    )
    for key in integer_keys:
        if not isinstance(value.get(key), int) or isinstance(value.get(key), bool) or value[key] < 0:
            raise NativeAdapterError("output_manifest_failed", f"native metric {key} is invalid")
    if value["actual_h2d_bytes"] != value["descriptor_h2d_bytes"] + value["operand_h2d_bytes"] + value["reset_h2d_bytes"] + value["cross_h2d_bytes"]:
        raise NativeAdapterError("output_manifest_failed", "native H2D byte invariant failed")
    if value["actual_d2h_bytes"] != value["cross_d2h_bytes"] + value["final_d2h_bytes"] or value["actual_transfer_bytes"] != value["actual_h2d_bytes"] + value["actual_d2h_bytes"]:
        raise NativeAdapterError("output_manifest_failed", "native transfer byte invariant failed")
    for key in ("reset_h2d_bytes", "cross_d2h_bytes", "cross_h2d_bytes", "final_d2h_bytes"):
        if value[key] % total_iterations != 0:
            raise NativeAdapterError("output_manifest_failed", f"native metric {key} is not repetition-aligned")


def _stage_native_output(root: Path, manifest_path: Path, plan: ExecutionPlan, destination: Path) -> Path:
    manifest = _read_object(manifest_path, "resident package manifest")
    outputs = manifest.get("final_outputs")
    if not isinstance(outputs, list) or len(outputs) != 1 or not isinstance(outputs[0], Mapping):
        raise NativeAdapterError("result_transfer_failed", "resident final output binding is missing")
    output = outputs[0]
    expected = plan.final_outputs[0]
    if output.get("component") != expected.component or output.get("slot_id") != expected.slot_id or output.get("elements") != expected.element_count:
        raise NativeAdapterError("result_transfer_failed", "resident final output binding differs from plan")
    output_ref = output.get("output_path")
    if not isinstance(output_ref, str) or not output_ref:
        raise NativeAdapterError("result_transfer_failed", "resident final output path is missing")
    source = (manifest_path.parent / output_ref).resolve()
    if not source.is_file():
        raise NativeAdapterError("result_transfer_failed", "native final output is missing")
    payload = _read_bytes(source, "native final output")
    raw_bytes = output.get("raw_bytes")
    if raw_bytes != len(payload) or raw_bytes != expected.element_count * 4:
        raise NativeAdapterError("result_transfer_failed", "native final output byte count is invalid")
    if len(payload) % 4 or any(not math.isfinite(item[0]) for item in struct.iter_unpack("<f", payload)):
        raise NativeAdapterError("output_validation_failed", "native final output is nonfinite")
    try:
        destination.unlink()
    except FileNotFoundError:
        pass
    try:
        shutil.copyfile(source, destination)
    except OSError as exc:
        raise NativeAdapterError("result_transfer_failed", "native final output could not be staged") from exc
    return destination


def _read_float32_output(path: Path, element_count: int) -> list[float]:
    payload = _read_bytes(path, "staged native final output")
    if len(payload) != element_count * 4:
        raise NativeAdapterError("result_transfer_failed", "staged native final output byte count is invalid")
    values = [item[0] for item in struct.iter_unpack("<f", payload)]
    if len(values) != element_count or any(not math.isfinite(value) for value in values):
        raise NativeAdapterError("output_validation_failed", "staged native final output is invalid")
    return values


def _normalize_session(
    response: Mapping[str, Any],
    *,
    request: Mapping[str, Any],
    plan: ExecutionPlan,
    elapsed_s: float,
    response_path: Path,
    final_output_path: Path,
    command: tuple[str, ...],
) -> dict[str, Any]:
    metrics = response["metrics"]
    total_iterations = request["requested_warmups"] + request["requested_repetitions"]
    repeat_h2d = metrics["reset_h2d_bytes"] // total_iterations + metrics["cross_h2d_bytes"] // total_iterations
    repeat_d2h = metrics["cross_d2h_bytes"] // total_iterations + metrics["final_d2h_bytes"] // total_iterations
    assignments = [to_jsonable(item) for item in plan.assignments]
    transfers = [to_jsonable(item) for item in plan.transfer_edges]
    final_path_ref = final_output_path.relative_to(response_path.parent).as_posix()
    final_output = _read_float32_output(
        final_output_path, plan.final_outputs[0].element_count
    )
    output_hash = _sha256(_read_bytes(final_output_path, "native final output"))
    validation_id = "final_session_output:" + hashlib.sha256(
        f"{plan.execution_plan_hash}:{output_hash}".encode("ascii")
    ).hexdigest()
    aggregate_completion_id = "aggregate_session_completion:" + hashlib.sha256(
        json.dumps(
            {
                "execution_plan_hash": plan.execution_plan_hash,
                "completed_per_dpu": metrics["completed_per_dpu"],
                "total_iterations": total_iterations,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    ).hexdigest()
    native_timing = response.get("timing")
    if not isinstance(native_timing, Mapping):
        raise NativeAdapterError("output_manifest_failed", "native timing object is missing")
    session_timing = {
        "allocation_time_s": native_timing.get("allocation_time_s"),
        "binary_load_time_s": native_timing.get("binary_load_time_s"),
        "descriptor_h2d_time_s": native_timing.get("descriptor_h2d_time_s"),
        "release_time_s": native_timing.get("release_time_s"),
        "total_session_time_s": elapsed_s,
    }
    repetitions_out = []
    for index in range(total_iterations):
        repetitions_out.append(
            {
                "repeat_id": index if index < request["requested_warmups"] else index - request["requested_warmups"],
                "warmup": index < request["requested_warmups"],
                "status": "completed",
                "scheduled_task_count": plan.logical_task_count,
                "wave_barrier_count": plan.wave_count,
                "launch_count": plan.logical_task_count,
                "synchronize_count": plan.logical_task_count,
                "device_launch_mode": DEVICE_LAUNCH_MODE,
                "synchronization_policy": SYNCHRONIZATION_POLICY,
                "fully_synchronous_kernel_launch": False,
                # The native host reports aggregate stage timings. Null is
                # intentional: splitting those observations would fabricate
                # per-repetition hardware timing.
                "timing": {
                    "operand_h2d_time_s": None,
                    "cross_dpu_transfer_time_s": None,
                    "wave_launch_sync_time_s": None,
                    "final_d2h_time_s": None,
                    "total_repetition_time_s": None,
                },
                "transfer": {
                    "h2d_bytes": repeat_h2d,
                    "d2h_bytes": repeat_d2h,
                    "total_bytes": repeat_h2d + repeat_d2h,
                },
                "validation_id": validation_id,
                "repeat_output_validation_status": "not_individually_collected",
                "session_completion_scope": "aggregate_across_warmups_and_repetitions",
                "repeat_completion_observation_status": "not_individually_collected",
                "aggregate_session_completion_id": aggregate_completion_id,
                "aggregate_session_completion_status": "passed",
            }
        )
    return {
        "schema_version": ADAPTER_SESSION_SCHEMA,
        "status": "completed",
        "returncode": 0,
        "request_id": request["request_id"],
        "backend_id": NATIVE_BACKEND_ID,
        "target_requested": "hardware",
        "target_observed": "physical_hardware",
        "execution_plan_hash": plan.execution_plan_hash,
        "upmem_execution_plan_hash": plan.execution_plan_hash,
        "package_file_sha256": plan.package_file_sha256,
        "schedule_sidecar_sha256": plan.schedule_sidecar_sha256,
        "source_identity": dict(request["source_identity"]),
        "package_identity": dict(request["package_identity"]),
        "requested_dpu_count": plan.requested_dpu_count,
        "allocated_dpu_count": response["allocated_dpu_count"],
        "tasklets_per_dpu": plan.tasklets_per_dpu,
        "allocation_attempted": response["allocation"]["attempted"],
        "allocation_count": 1,
        "allocation_succeeded": _allocation_succeeded(response),
        "persistent_allocation_observed": _allocation_succeeded(response),
        "release_confirmed": response["allocation"]["release_confirmed"],
        "hardware_allocation_verified": response["hardware_allocation_verified"],
        "native_kernel_executed": response["native_kernel_executed"],
        "hardware_kernel_executed": response["hardware_kernel_executed"],
        "simulator_kernel_executed": response["simulator_kernel_executed"],
        "cpu_fallback_used": response["cpu_fallback_used"],
        "hardware_speedup_applicable": response["hardware_speedup_applicable"],
        "device_launch_mode": DEVICE_LAUNCH_MODE,
        "synchronization_policy": SYNCHRONIZATION_POLICY,
        "fully_synchronous_kernel_launch": False,
        "requested_warmups": request["requested_warmups"],
        "requested_repetitions": request["requested_repetitions"],
        "native_session_count": 1,
        "logical_task_count": plan.logical_task_count,
        "session_completion_scope": "aggregate_across_warmups_and_repetitions",
        "aggregate_completed_per_dpu": list(metrics["completed_per_dpu"]),
        "aggregate_total_task_completion_count": sum(metrics["completed_per_dpu"]),
        "aggregate_session_completion_id": aggregate_completion_id,
        "aggregate_session_completion_status": "passed",
        "total_task_completion_count": plan.logical_task_count * total_iterations,
        "exactly_once_execution_verified": True,
        "wave_barrier_count_total": plan.wave_count * total_iterations,
        "operation_assignments": assignments,
        "cross_dpu_transfer": {
            "count": len(plan.transfer_edges),
            "bytes": plan.total_cross_dpu_transfer_bytes,
        },
        "cross_dpu_transfers": transfers,
        "schedule_h2d_bytes": 0,
        "session_timing": session_timing,
        "session_transfer": {
            "initial_h2d_bytes": metrics["descriptor_h2d_bytes"] + metrics["operand_h2d_bytes"],
            "actual_h2d_bytes": metrics["actual_h2d_bytes"],
            "actual_d2h_bytes": metrics["actual_d2h_bytes"],
            "actual_transfer_bytes": metrics["actual_transfer_bytes"],
        },
        "native_response_path": str(response_path),
        "native_command": list(command),
        "native_metrics": dict(metrics),
        "session_validation": {
            "validation_id": validation_id,
            "status": "collected",
            "scope": "final_session_output_only",
            "output": final_output,
            "final_output_path": final_path_ref,
            "output_sha256": output_hash,
            "output_provenance": "native_final_output_after_requested_repetitions",
        },
        "repetitions": repetitions_out,
    }


def _allocation_succeeded(response: Mapping[str, Any]) -> bool:
    """Accept a durable allocation success flag after the DPU set is released."""

    allocation = response.get("allocation")
    if not isinstance(allocation, Mapping):
        return False
    return (
        allocation.get("succeeded", allocation.get("confirmed")) is True
        or response.get("hardware_allocation_verified") is True
    )


def _host_binary_for(dpu_path: Path) -> Path:
    global _LAST_BUILD
    if _LAST_BUILD is not None and dpu_path == _LAST_BUILD.dpu_binary:
        host = _LAST_BUILD.host_binary
        if host.is_file():
            return host
    candidate = dpu_path.parent / HOST_BINARY_NAME
    if candidate.is_file():
        return candidate.resolve()
    raise NativeAdapterError("native_build_failed", "host binary for request DPU binary is missing")


def _manifest_package_path(manifest_path: Path, manifest: Mapping[str, Any]) -> Path:
    if manifest.get("schema_version") != "generic_loop_resident_graph_session_v1" or manifest.get("manifest_kind") != "resident_graph_request":
        raise NativeAdapterError("package_preparation_failed", "resident manifest schema is invalid")
    value = manifest.get("package_path")
    if not isinstance(value, str) or not value:
        raise NativeAdapterError("package_preparation_failed", "resident manifest package path is missing")
    return (manifest_path.parent / value).resolve()


def _resolve_ref(base: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value or not value.isascii():
        raise NativeAdapterError("manifest_parse_failed", f"{label} path is invalid")
    return (base / value).resolve()


def _read_object(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise NativeAdapterError("output_manifest_failed" if label == "native response" else "manifest_parse_failed", f"{label} is invalid") from exc
    if not isinstance(payload, dict):
        raise NativeAdapterError("output_manifest_failed" if label == "native response" else "manifest_parse_failed", f"{label} is not an object")
    return payload


def _read_bytes(path: Path, label: str) -> bytes:
    try:
        return path.read_bytes()
    except OSError as exc:
        raise NativeAdapterError("result_transfer_failed", f"{label} is unreadable") from exc


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require(mapping: Mapping[str, Any], key: str, expected: Any) -> None:
    if mapping.get(key) != expected:
        raise NativeAdapterError("manifest_parse_failed", f"request {key} is invalid")


def _positive_timeout(value: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise NativeAdapterError("kernel_timeout", "timeout_s is invalid") from exc
    if not math.isfinite(result) or result <= 0 or result > MAX_TIMEOUT_S:
        raise NativeAdapterError("kernel_timeout", "timeout_s is outside the supported range")
    return result


def _require_physical_environment() -> None:
    if os.environ.get("UPMEM_ALLOW_PHYSICAL_HARDWARE") != "1":
        raise NativeAdapterError("hardware_opt_in_missing", "UPMEM_ALLOW_PHYSICAL_HARDWARE=1 is required")
    if os.environ.get("DPU_BACKEND"):
        raise NativeAdapterError("hardware_profile_violation", "DPU_BACKEND is forbidden for physical execution")
    if os.environ.get("UPMEM_EXECUTION_MODE", "").lower() in {"simulator", "cpu", "mock"}:
        raise NativeAdapterError("hardware_profile_violation", "simulator/CPU execution mode is forbidden")


__all__ = [
    "ADAPTER_SESSION_SCHEMA",
    "NativeAdapterError",
    "build",
    "execute",
    "validate",
]
