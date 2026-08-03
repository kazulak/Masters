"""Bounded native adapter for the committed two-DPU sliced-resident path."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
from typing import Any, Mapping, Sequence

from quantum_bench.core.records import JsonDict
from quantum_bench.targets.upmem.environment import discover_upmem_sdk
from quantum_bench.targets.upmem.hardware_session import (
    HardwareSessionBuild,
    _hash_file,
    _hash_tree,
    _run_command as _run_build_command,
    _snippet,
)
from quantum_bench.targets.upmem.hardware_taskgraph_resident import (
    RESIDENT_MAX_COMPONENT_OPS,
    RESIDENT_MAX_ELEMENTS,
    RESIDENT_MAX_LOGICAL_TASKS,
    RESIDENT_MAX_RANK,
    RESIDENT_MAX_SLOT_DESCRIPTORS,
    RESIDENT_MRAM_POOL_BYTES,
    RESIDENT_OUTPUT_TILE_ELEMENTS,
)


TWO_DPU_SOURCE_DIR = "upmem_sdk_generic_loop_resident_two_dpu"
RESIDENT_SOURCE_DIR = "upmem_sdk_generic_loop_resident"
BUILD_TIMEOUT_S = 120.0
CLEANUP_GRACE_S = 2.0
PROFILE_VERSION = "hardware_sliced_resident_two_dpu_m2_v1"
BACKEND_ID = "upmem_sdk_hardware_sliced_resident_two_dpu"
ROUTE_ID = "upmem_tn_hardware_sliced_resident_two_dpu"
_BUILD_CAPS = {
    "max_rank": RESIDENT_MAX_RANK,
    "max_tensor_elements": RESIDENT_MAX_ELEMENTS,
    "max_logical_tasks": RESIDENT_MAX_LOGICAL_TASKS,
    "max_component_ops": RESIDENT_MAX_COMPONENT_OPS,
    "max_slot_descriptors": RESIDENT_MAX_SLOT_DESCRIPTORS,
    "mram_pool_bytes": RESIDENT_MRAM_POOL_BYTES,
    "output_tile_elements": RESIDENT_OUTPUT_TILE_ELEMENTS,
}


@dataclass(frozen=True)
class SlicedResidentHardwareProfile:
    version: str
    target: str
    backend_id: str
    route_id: str
    requested_dpu_count: int
    slices: int
    tasklets_per_dpu: int
    numeric_mode: str
    synchronous_execution: bool
    device_launch_mode: str
    host_completion_mode: str
    timeout_s: float
    performance_claim_applicable: bool
    max_rank: int = RESIDENT_MAX_RANK
    max_tensor_elements: int = RESIDENT_MAX_ELEMENTS
    max_logical_tasks: int = RESIDENT_MAX_LOGICAL_TASKS
    max_component_ops: int = RESIDENT_MAX_COMPONENT_OPS
    max_slot_descriptors: int = RESIDENT_MAX_SLOT_DESCRIPTORS
    mram_pool_bytes: int = RESIDENT_MRAM_POOL_BYTES
    output_tile_elements: int = RESIDENT_OUTPUT_TILE_ELEMENTS


@dataclass(frozen=True)
class SlicedResidentSessionExecution:
    status: str
    failure_stage: str | None
    response_path: Path
    response: JsonDict
    process_time_s: float
    command: tuple[str, ...]
    stdout_snippet: str
    stderr_snippet: str
    timed_out: bool
    cleanup_confirmed: bool


def build_sliced_resident_hardware_session(
    root_dir: Path,
    session_root: Path,
    *,
    profile: SlicedResidentHardwareProfile | Mapping[str, Any],
    environment: Mapping[str, str],
) -> HardwareSessionBuild:
    profile = _coerce_profile(profile)
    make_path, sdk_tools = _required_build_tools(environment)
    source_parent = root_dir / "native" / "upmem" / "simplepim"
    source_dirs = (
        source_parent / TWO_DPU_SOURCE_DIR,
        source_parent / RESIDENT_SOURCE_DIR,
    )
    if any(not source.is_dir() for source in source_dirs):
        raise RuntimeError(
            "native_build_failed: two-DPU resident source tree is missing"
        )

    source_snapshot = session_root / "native" / "src"
    build_parent = session_root / "native" / "build"
    _copy_sibling_sources(source_dirs, source_snapshot)
    _copy_sibling_sources(source_dirs, build_parent)
    build_dir = build_parent / TWO_DPU_SOURCE_DIR
    command = (
        str(make_path),
        "clean",
        "all",
        f"MAX_RANK={int(profile.max_rank)}",
        f"MAX_ELEMS={int(profile.max_tensor_elements)}",
        f"RESIDENT_MAX_LOGICAL_TASKS={int(profile.max_logical_tasks)}",
        f"RESIDENT_MAX_COMPONENT_OPS={int(profile.max_component_ops)}",
        f"RESIDENT_MAX_SLOT_DESCRIPTORS={int(profile.max_slot_descriptors)}",
        f"RESIDENT_MRAM_POOL_BYTES={int(profile.mram_pool_bytes)}",
        f"RESIDENT_OUTPUT_TILE_ELEMS={int(profile.output_tile_elements)}",
        "NR_TASKLETS=1",
        "UPMEM_GENERIC_HARDWARE_MVP=1",
    )
    started = time.perf_counter()
    completed = _run_build_command(
        command,
        cwd=build_dir,
        env=_physical_build_env(environment),
        timeout_s=BUILD_TIMEOUT_S,
    )
    build_time_s = time.perf_counter() - started
    if completed["returncode"] != 0:
        stage = (
            "native_build_timeout" if completed["timed_out"] else "native_build_failed"
        )
        detail = completed["stderr_snippet"] or completed.get("error") or "make failed"
        raise RuntimeError(f"{stage}: {detail}")

    host_binary = build_dir / "bin" / "host_two_dpu"
    dpu_binary = build_dir / "bin" / "dpu_resident_two_dpu"
    if not host_binary.is_file() or not dpu_binary.is_file():
        raise RuntimeError(
            "native_build_failed: expected two-DPU binaries were not produced"
        )
    return HardwareSessionBuild(
        session_root=session_root,
        source_snapshot=source_snapshot,
        build_dir=build_dir,
        host_binary=host_binary,
        dpu_binary=dpu_binary,
        source_tree_hash=_hash_tree(source_snapshot),
        host_binary_hash=_hash_file(host_binary),
        dpu_binary_hash=_hash_file(dpu_binary),
        build_time_s=build_time_s,
        build_command=command,
        sdk_tools=sdk_tools,
    )


def execute_sliced_resident_hardware_session(
    build: HardwareSessionBuild,
    *,
    manifest_paths: Sequence[Path],
    response_path: Path,
    profile: SlicedResidentHardwareProfile | Mapping[str, Any],
    environment: Mapping[str, str],
) -> SlicedResidentSessionExecution:
    """Execute exactly two canonical slice manifests with no fallback path."""

    profile = _coerce_profile(profile)
    if environment.get("UPMEM_ALLOW_PHYSICAL_HARDWARE") != "1":
        raise ValueError(
            "hardware_opt_in_missing: UPMEM_ALLOW_PHYSICAL_HARDWARE=1 is required"
        )
    _reject_simulator_selectors(environment)
    if len(manifest_paths) != 2:
        raise ValueError(
            "hardware_profile_violation: exactly two slice manifests are required"
        )

    root = build.session_root.resolve()
    manifests = tuple(
        _canonical_session_path(root, path, "slice manifest") for path in manifest_paths
    )
    if manifests[0] == manifests[1]:
        raise ValueError("hardware_profile_violation: slice manifests must be distinct")
    response = _canonical_session_path(root, response_path, "response")
    command = (
        str(build.host_binary.resolve()),
        "--slice-package-0",
        str(manifests[0]),
        "--slice-package-1",
        str(manifests[1]),
        "--resident-response",
        str(response),
    )
    completed = _run_physical_command(
        command,
        cwd=build.build_dir,
        env=_physical_build_env(environment),
        timeout_s=float(profile.timeout_s),
    )
    payload = _load_response(response)
    response_loaded = payload is not None
    payload = payload or {}
    response_completed = response_loaded and _response_completed(payload)
    failure_stage = _execution_failure_stage(completed, payload, response_loaded)
    return SlicedResidentSessionExecution(
        status="completed"
        if completed["returncode"] == 0 and response_completed
        else "failed",
        failure_stage=None
        if completed["returncode"] == 0 and response_completed
        else failure_stage,
        response_path=response,
        response=payload,
        process_time_s=float(completed["elapsed_s"]),
        command=command,
        stdout_snippet=str(completed["stdout_snippet"]),
        stderr_snippet=str(completed["stderr_snippet"]),
        timed_out=bool(completed["timed_out"]),
        cleanup_confirmed=bool(completed["cleanup_confirmed"]),
    )


def parse_sliced_resident_hardware_profile(
    value: Mapping[str, Any],
) -> SlicedResidentHardwareProfile:
    """Parse the two-DPU suite profile and supply the frozen build caps."""

    if not isinstance(value, Mapping):
        raise ValueError(
            "hardware_profile_violation: sliced resident profile must be a mapping"
        )
    expected = {
        "hardware_profile_version": PROFILE_VERSION,
        "target": "hardware",
        "backend_id": BACKEND_ID,
        "route_id": ROUTE_ID,
        "requested_dpu_count": 2,
        "slices": 2,
        "tasklets_per_dpu": 1,
        "numeric_modes": ["none"],
        "synchronous_execution": True,
        "device_launch_mode": "asynchronous_dpu_set",
        "host_completion_mode": "blocking_sync",
        "performance_claim_applicable": False,
    }
    allowed = set(expected) | {"timeout_s"} | set(_BUILD_CAPS)
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(
            "hardware_profile_violation: sliced resident profile keys differ"
        )
    for key, expected_value in expected.items():
        if key in {"device_launch_mode", "host_completion_mode"} and key not in value:
            continue
        if value.get(key) != expected_value:
            raise ValueError(
                f"hardware_profile_violation: {key} must be {expected_value!r}"
            )
    timeout = value.get("timeout_s")
    if (
        not isinstance(timeout, (int, float))
        or isinstance(timeout, bool)
        or not math.isfinite(float(timeout))
        or timeout <= 0
    ):
        raise ValueError(
            "hardware_profile_violation: timeout_s must be finite and positive"
        )
    for key, expected_value in _BUILD_CAPS.items():
        if key in value and value[key] != expected_value:
            raise ValueError(
                f"hardware_profile_violation: {key} must be {expected_value}"
            )
    return SlicedResidentHardwareProfile(
        version=PROFILE_VERSION,
        target="hardware",
        backend_id=BACKEND_ID,
        route_id=ROUTE_ID,
        requested_dpu_count=2,
        slices=2,
        tasklets_per_dpu=1,
        numeric_mode="none",
        synchronous_execution=True,
        device_launch_mode=str(value.get("device_launch_mode", "asynchronous_dpu_set")),
        host_completion_mode=str(value.get("host_completion_mode", "blocking_sync")),
        timeout_s=float(timeout),
        performance_claim_applicable=False,
        **_BUILD_CAPS,
    )


def _coerce_profile(
    profile: SlicedResidentHardwareProfile | Mapping[str, Any],
) -> SlicedResidentHardwareProfile:
    if isinstance(profile, SlicedResidentHardwareProfile):
        return parse_sliced_resident_hardware_profile(
            {
                "hardware_profile_version": profile.version,
                "target": profile.target,
                "backend_id": profile.backend_id,
                "route_id": profile.route_id,
                "requested_dpu_count": profile.requested_dpu_count,
                "slices": profile.slices,
                "tasklets_per_dpu": profile.tasklets_per_dpu,
                "numeric_modes": [profile.numeric_mode],
                "synchronous_execution": profile.synchronous_execution,
                "device_launch_mode": profile.device_launch_mode,
                "host_completion_mode": profile.host_completion_mode,
                "timeout_s": profile.timeout_s,
                "performance_claim_applicable": profile.performance_claim_applicable,
                **{key: getattr(profile, key) for key in _BUILD_CAPS},
            }
        )
    return parse_sliced_resident_hardware_profile(profile)


def _required_build_tools(environment: Mapping[str, str]) -> tuple[str, JsonDict]:
    sdk = discover_upmem_sdk(env=environment)
    by_name = {tool.name: tool for tool in sdk.tools}
    make_path = shutil.which("make", path=environment.get("PATH"))
    names = ("dpu-upmem-dpurte-clang", "dpu-pkg-config")
    missing = [
        name for name in names if not by_name.get(name) or not by_name[name].available
    ]
    if not make_path:
        missing.append("make")
    if missing:
        raise RuntimeError(
            "sdk_discovery_failed: missing required UPMEM SDK tools: "
            + ", ".join(sorted(missing))
        )
    return str(make_path), {
        "make": str(make_path),
        **{
            name: str(by_name[name].path) if by_name[name].path else None
            for name in names
        },
    }


def _copy_sibling_sources(sources: Sequence[Path], destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    for source in sources:
        shutil.copytree(
            source,
            destination / source.name,
            ignore=shutil.ignore_patterns("bin", "__pycache__", "*.pyc"),
        )


def _canonical_session_path(root: Path, path: Path, label: str) -> Path:
    try:
        result = path.resolve()
        result.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            f"hardware_profile_violation: {label} must be inside session root"
        ) from exc
    return result


def _reject_simulator_selectors(environment: Mapping[str, str]) -> None:
    backend = environment.get("DPU_BACKEND")
    if backend:
        raise ValueError("hardware_profile_violation: DPU_BACKEND must be unset")
    for name in ("UPMEM_PROFILE", "UPMEM_PROFILE_BASE"):
        value = environment.get(name, "")
        if "sim" in value.lower():
            raise ValueError(f"hardware_profile_violation: {name} selects a simulator")


def _physical_build_env(environment: Mapping[str, str]) -> dict[str, str]:
    result = dict(environment)
    for name in ("DPU_BACKEND", "UPMEM_PROFILE", "UPMEM_PROFILE_BASE"):
        result.pop(name, None)
    return result


def _run_physical_command(
    command: Sequence[str], *, cwd: Path, env: Mapping[str, str], timeout_s: float
) -> JsonDict:
    started = time.perf_counter()
    process: subprocess.Popen[str] | None = None
    try:
        process = subprocess.Popen(
            tuple(command),
            cwd=cwd,
            env=dict(env),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        try:
            stdout, stderr = process.communicate(timeout=timeout_s)
            return {
                "returncode": process.returncode,
                "elapsed_s": time.perf_counter() - started,
                "timed_out": False,
                "cleanup_confirmed": process.returncode == 0,
                "stdout_snippet": _snippet(stdout),
                "stderr_snippet": _snippet(stderr),
            }
        except subprocess.TimeoutExpired as exc:
            stdout, stderr = _snippet(exc.stdout), _snippet(exc.stderr)
            _terminate_process_group(process, signal.SIGTERM)
            try:
                out, err = process.communicate(timeout=CLEANUP_GRACE_S)
            except subprocess.TimeoutExpired as grace_exc:
                stdout += _snippet(grace_exc.stdout)
                stderr += _snippet(grace_exc.stderr)
                _terminate_process_group(process, signal.SIGKILL)
                try:
                    out, err = process.communicate(timeout=CLEANUP_GRACE_S)
                except subprocess.TimeoutExpired as kill_exc:
                    stdout += _snippet(kill_exc.stdout)
                    stderr += _snippet(kill_exc.stderr)
                    return {
                        "returncode": process.returncode,
                        "elapsed_s": time.perf_counter() - started,
                        "timed_out": True,
                        "cleanup_confirmed": False,
                        "stdout_snippet": stdout,
                        "stderr_snippet": stderr,
                    }
            return {
                "returncode": process.returncode,
                "elapsed_s": time.perf_counter() - started,
                "timed_out": True,
                # A host signal cannot prove that dpu_free ran.  Never turn a
                # post-timeout exit code into a hardware-release assertion.
                "cleanup_confirmed": False,
                "stdout_snippet": stdout + _snippet(out),
                "stderr_snippet": stderr + _snippet(err),
            }
    except OSError as exc:
        return {
            "returncode": None,
            "elapsed_s": time.perf_counter() - started,
            "timed_out": False,
            "cleanup_confirmed": False,
            "stdout_snippet": "",
            "stderr_snippet": _snippet(exc),
            "error": str(exc),
        }


def _terminate_process_group(
    process: subprocess.Popen[str], signum: signal.Signals
) -> None:
    try:
        os.killpg(process.pid, signum)
    except (ProcessLookupError, PermissionError, OSError):
        if signum == signal.SIGKILL:
            process.kill()
        else:
            process.terminate()


def _load_response(path: Path) -> JsonDict | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _finite_nonnegative(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) >= 0.0
    )


def _response_completed(response: Mapping[str, Any]) -> bool:
    allocation = response.get("allocation")
    launch = response.get("launch")
    release = response.get("release")
    timing = response.get("timing")
    slices = response.get("slices")
    operation_count = response.get("operation_count")
    observed_counts = response.get("observed_operation_completion_counts")
    if not isinstance(operation_count, int) or isinstance(operation_count, bool) or operation_count < 1:
        return False
    if not isinstance(slices, list) or len(slices) != 2:
        return False
    if not isinstance(observed_counts, list) or observed_counts != [operation_count, operation_count]:
        return False
    timing_fields = (
        "package_parse_time_s",
        "allocation_time_s",
        "binary_load_time_s",
        "initial_h2d_time_s",
        "operation_control_h2d_time_s",
        "launch_enqueue_time_s",
        "sync_wait_time_s",
        "final_d2h_time_s",
        "output_write_time_s",
        "release_time_s",
        "total_route_time_s",
    )
    if (
        response.get("backend_id") != BACKEND_ID
        or response.get("backend_family") != "upmem_sdk"
        or response.get("target_requested") != "hardware"
        or response.get("target_observed") != "hardware"
        or response.get("hardware_profile_version") != PROFILE_VERSION
        or response.get("tasklets_per_dpu") != 1
        or response.get("simulator_kernel_executed") is not False
        or response.get("hardware_kernel_executed") is not True
        or response.get("cpu_fallback_used") is not False
        or response.get("hardware_functionality_evidence") is not True
        or (
            operation_count > 1
            and response.get("native_execution_sentinel_available") is not True
        )
        or (
            operation_count > 1
            and response.get("completion_evidence")
            != "dpu_written_completion_sentinel_read_after_each_sync"
        )
        or (
            operation_count > 1
            and response.get("device_completion_confirmed") is not True
        )
        or response.get("device_launch_mode") != "asynchronous_dpu_set"
        or response.get("host_completion_mode") != "blocking_sync"
        or response.get("timing_scope") != "host_observed_sdk_stage_boundaries"
        or response.get("timing_is_bringup_only") is not True
        or not isinstance(timing, Mapping)
        or timing.get("clock") != "clock_monotonic"
        or timing.get("sync_wait_is_not_pure_kernel_time") is not True
        or timing.get("kernel_time_s") is not None
        or any(not _finite_nonnegative(timing.get(field)) for field in timing_fields)
    ):
        return False
    if operation_count > 1 and any(
        not isinstance(response.get(name), int)
        or isinstance(response.get(name), bool)
        or response.get(name) < 0
        for name in ("actual_h2d_bytes", "actual_d2h_bytes", "actual_transfer_bytes")
    ):
        return False
    if operation_count > 1 and response["actual_transfer_bytes"] != (
        response["actual_h2d_bytes"] + response["actual_d2h_bytes"]
    ):
        return False
    if operation_count > 1 and response.get("completion_sentinel_read_counts") != [
        operation_count,
        operation_count,
    ]:
        return False
    for expected_slice_id, entry in enumerate(slices):
        if not isinstance(entry, Mapping):
            return False
        if (
            entry.get("slice_id") != expected_slice_id
            or entry.get("dpu_index") != expected_slice_id
            or entry.get("allocated") is not True
            or entry.get("release_confirmed") is not True
            or entry.get("package_transferred") is not True
            or entry.get("inputs_transferred") is not True
            or entry.get("partial_output_read") is not True
            or entry.get("partial_output_written") is not True
            or entry.get("completion_confirmed") is not True
            or entry.get("operation_count") != operation_count
            or entry.get("completed_operation_count") != operation_count
            or entry.get("observed_operation_completion_count") != operation_count
            or entry.get("operation_completion_confirmed") is not True
            or (
                operation_count > 1
                and entry.get("completion_sentinel_read_count") != operation_count
            )
            or not isinstance(entry.get("input_count"), int)
            or entry.get("input_count", 0) < 2
        ):
            return False
        sentinel = entry.get("dpu_completion_sentinel")
        if operation_count > 1 and (
            not isinstance(sentinel, Mapping)
            or sentinel.get("verified") is not True
            or sentinel.get("active_operation_index") != operation_count - 1
            or sentinel.get("completion_status") != 1
            or sentinel.get("completed_operation_count") != operation_count
            or sentinel.get("output_elements_processed") != entry.get("partial_output_elements")
        ):
            return False
        raw_bytes = entry.get("partial_output_raw_bytes", entry.get("partial_output_bytes"))
        legacy_bytes = entry.get("partial_output_bytes")
        transfer_bytes = entry.get("partial_output_transfer_bytes")
        if (
            not isinstance(raw_bytes, int)
            or isinstance(raw_bytes, bool)
            or raw_bytes < 0
            or legacy_bytes != raw_bytes
            or not isinstance(transfer_bytes, int)
            or isinstance(transfer_bytes, bool)
            or transfer_bytes < raw_bytes
        ):
            return False
    return (
        response.get("status") == "completed"
        and response.get("failure_stage") is None
        and response.get("hardware_execution") is True
        and response.get("cpu_fallback_used") is False
        and isinstance(allocation, Mapping)
        and allocation.get("requested_dpus") == 2
        and allocation.get("allocated_dpus") == 2
        and allocation.get("verified") is True
        and isinstance(launch, Mapping)
        and launch.get("mode") == "asynchronous"
        and launch.get("device_launch_mode") == "asynchronous_dpu_set"
        and launch.get("host_completion_mode") == "blocking_sync"
        and launch.get("operation_count") == operation_count
        and launch.get("async_launch_count") == operation_count
        and launch.get("synchronize_count") == operation_count
        and launch.get("completed") is True
        and response.get("async_launch_count") == operation_count
        and response.get("synchronize_count") == operation_count
        and isinstance(release, Mapping)
        and release.get("confirmed") is True
        and (
            operation_count == 1
            or release.get("device_completion_confirmed") is True
        )
        and (
            operation_count == 1
            or all(
                isinstance(entry.get("dpu_completion_sentinel"), Mapping)
                and entry["dpu_completion_sentinel"].get("verified") is True
                for entry in slices
            )
        )
    )


def _execution_failure_stage(
    completed: Mapping[str, Any], response: Mapping[str, Any], response_loaded: bool
) -> str:
    if completed["timed_out"]:
        return "kernel_timeout"
    native_stage = response.get("failure_stage")
    if isinstance(native_stage, str) and native_stage:
        return native_stage
    if not response_loaded:
        return "response_transport_failed"
    if completed["returncode"] != 0:
        return "native_host_failed"
    return "response_evidence_invalid"
