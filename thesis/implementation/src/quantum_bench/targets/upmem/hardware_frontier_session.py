"""Bounded physical-session adapter for the M3.1 frontier native host."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import shutil
import time
from typing import Any, Mapping

from quantum_bench.core.records import JsonDict
from quantum_bench.targets.upmem.environment import discover_upmem_sdk
from quantum_bench.targets.upmem.hardware_session import (
    HardwareSessionBuild,
    _copy_source_tree,
    _hash_file,
    _run_command,
    _run_resident_command,
    _sanitised_hardware_env,
)
from quantum_bench.targets.upmem.hardware_taskgraph_frontier import (
    BACKEND_ID,
    NATIVE_SCHEMA,
    NUMERIC_MODE,
    PROFILE_ID,
    REQUEST_SCHEMA,
    REQUESTED_DPUS,
    ROUTE_ID,
    TASKLETS_PER_DPU,
    _validate_manifest_identity,
    validate_frontier_output_file,
    validate_frontier_package_against_manifest,
    validate_frontier_native_response,
)


NATIVE_SOURCE_DIR = "upmem_sdk_generic_loop_frontier_two_dpu"
HOST_BINARY_NAME = "host_frontier_two_dpu"
DPU_BINARY_NAME = "dpu_frontier_two_dpu"
BUILD_TIMEOUT_S = 120.0
DEFAULT_TIMEOUT_S = 30.0


@dataclass(frozen=True)
class HardwareFrontierProfile:
    version: str
    target: str
    backend_id: str
    route_id: str
    native_schema: str
    requested_dpu_count: int
    tasklets_per_dpu: int
    numeric_mode: str
    numeric_modes: tuple[str, ...]
    synchronous_execution: bool
    device_launch_mode: str
    host_completion_mode: str
    timeout_s: float
    performance_claim_applicable: bool


@dataclass(frozen=True)
class HardwareFrontierExecution:
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


def parse_hardware_frontier_profile(
    value: HardwareFrontierProfile | Mapping[str, Any],
) -> HardwareFrontierProfile:
    if isinstance(value, HardwareFrontierProfile):
        value = {
            "hardware_profile_version": value.version,
            "target": value.target,
            "backend_id": value.backend_id,
            "route_id": value.route_id,
            "native_schema": value.native_schema,
            "requested_dpu_count": value.requested_dpu_count,
            "tasklets_per_dpu": value.tasklets_per_dpu,
            "numeric_mode": value.numeric_mode,
            "numeric_modes": list(value.numeric_modes),
            "synchronous_execution": value.synchronous_execution,
            "device_launch_mode": value.device_launch_mode,
            "host_completion_mode": value.host_completion_mode,
            "timeout_s": value.timeout_s,
            "performance_claim_applicable": value.performance_claim_applicable,
        }
    if not isinstance(value, Mapping):
        raise ValueError("hardware_profile_violation: frontier profile must be a mapping")
    expected = {
        "hardware_profile_version": PROFILE_ID,
        "target": "hardware",
        "backend_id": BACKEND_ID,
        "route_id": ROUTE_ID,
        "native_schema": NATIVE_SCHEMA,
        "requested_dpu_count": REQUESTED_DPUS,
        "tasklets_per_dpu": TASKLETS_PER_DPU,
        "numeric_mode": NUMERIC_MODE,
        "numeric_modes": [NUMERIC_MODE],
        "synchronous_execution": True,
        "device_launch_mode": "asynchronous_dpu_set",
        "host_completion_mode": "blocking_sync",
        "performance_claim_applicable": False,
    }
    allowed = set(expected) | {"timeout_s"}
    if set(value) - allowed:
        raise ValueError("hardware_profile_violation: frontier profile keys differ")
    for key, expected_value in expected.items():
        actual = value.get(key)
        if key not in value or type(actual) is not type(expected_value) or actual != expected_value:
            raise ValueError(f"hardware_profile_violation: {key} must be {expected_value!r}")
    timeout = value.get("timeout_s", DEFAULT_TIMEOUT_S)
    if (
        not isinstance(timeout, (int, float))
        or isinstance(timeout, bool)
        or not math.isfinite(float(timeout))
        or float(timeout) <= 0
    ):
        raise ValueError("hardware_profile_violation: timeout_s must be finite and positive")
    return HardwareFrontierProfile(
        version=PROFILE_ID,
        target="hardware",
        backend_id=BACKEND_ID,
        route_id=ROUTE_ID,
        native_schema=NATIVE_SCHEMA,
        requested_dpu_count=REQUESTED_DPUS,
        tasklets_per_dpu=TASKLETS_PER_DPU,
        numeric_mode=NUMERIC_MODE,
        numeric_modes=(NUMERIC_MODE,),
        synchronous_execution=True,
        device_launch_mode="asynchronous_dpu_set",
        host_completion_mode=str(value.get("host_completion_mode", "blocking_sync")),
        timeout_s=float(timeout),
        performance_claim_applicable=False,
    )


def build_hardware_frontier_session(
    root_dir: Path,
    session_root: Path,
    *,
    profile: HardwareFrontierProfile | Mapping[str, Any],
    environment: Mapping[str, str],
) -> HardwareSessionBuild:
    """Snapshot and compile the committed M3.1 native source tree."""

    parse_hardware_frontier_profile(profile)
    make_path, sdk_tools = _required_build_tools(environment)
    source = root_dir / "native" / "upmem" / "simplepim" / NATIVE_SOURCE_DIR
    if not source.is_dir():
        raise RuntimeError("native_build_failed: frontier native source tree is missing")
    resident_source = source.parent / "upmem_sdk_generic_loop_resident"
    if not resident_source.is_dir():
        raise RuntimeError("native_build_failed: resident native ABI source tree is missing")
    isolated_parent = session_root / "native"
    source_snapshot = session_root / "native" / "src"
    build_dir = session_root / "native" / "build"
    _copy_source_tree(resident_source, isolated_parent / resident_source.name)
    _copy_source_tree(source, source_snapshot)
    _copy_source_tree(source, build_dir)
    source_tree_hash = _hash_combined_source_tree(source_snapshot, isolated_parent / resident_source.name)
    started = time.perf_counter()
    command = (
        str(make_path),
        "clean",
        "all",
        "NR_TASKLETS=1",
        "UPMEM_GENERIC_HARDWARE_MVP=1",
        "FRONTIER_TASK_COUNT=3",
        "FRONTIER_BARRIER_COUNT=2",
    )
    completed = _run_build_command(
        command,
        cwd=build_dir,
        env=_sanitised_hardware_env(environment),
        timeout_s=BUILD_TIMEOUT_S,
    )
    if completed.get("returncode") != 0:
        stage = "native_build_timeout" if completed.get("timed_out") else "native_build_failed"
        detail = completed.get("stderr_snippet") or completed.get("error") or "make failed"
        raise RuntimeError(f"{stage}: {detail}")
    host_binary = build_dir / "bin" / HOST_BINARY_NAME
    dpu_binary = build_dir / "bin" / DPU_BINARY_NAME
    if not host_binary.is_file() or not dpu_binary.is_file():
        raise RuntimeError("native_build_failed: expected frontier host and DPU binaries were not produced")
    return HardwareSessionBuild(
        session_root=session_root,
        source_snapshot=source_snapshot,
        build_dir=build_dir,
        host_binary=host_binary,
        dpu_binary=dpu_binary,
        source_tree_hash=source_tree_hash,
        host_binary_hash=_hash_file(host_binary),
        dpu_binary_hash=_hash_file(dpu_binary),
        build_time_s=time.perf_counter() - started,
        build_command=command,
        sdk_tools=sdk_tools,
    )


def execute_hardware_frontier_session(
    build: HardwareSessionBuild,
    *,
    manifest_path: Path,
    response_path: Path,
    profile: HardwareFrontierProfile | Mapping[str, Any],
    environment: Mapping[str, str],
) -> HardwareFrontierExecution:
    """Execute one native frontier request and preserve native failure stages."""

    selected = parse_hardware_frontier_profile(profile)
    _require_physical_opt_in(environment)
    root = build.session_root.resolve()
    manifest = _read_manifest(_inside_session(root, manifest_path, "frontier manifest"))
    response = _inside_session(root, response_path, "frontier response")
    _validate_frontier_manifest(manifest, build, root)
    command = (
        str(build.host_binary.resolve()),
        "--frontier-package",
        str(manifest_path.resolve()),
        "--frontier-response",
        str(response),
    )
    completed = _run_frontier_command(
        command,
        cwd=build.build_dir,
        env=_sanitised_hardware_env(environment),
        timeout_s=selected.timeout_s,
    )
    payload = _load_response(response)
    failure_stage = _failure_stage(completed, payload)
    valid = False
    validation_error: str | None = None
    if payload:
        try:
            validate_frontier_native_response(payload, manifest)
            validate_frontier_output_file(payload, manifest, root)
            _validate_native_hashes(payload, manifest, build, root, manifest_path)
            valid = True
        except ValueError as exc:
            validation_error = str(exc)
            valid = False
    status = "completed" if completed.get("returncode") == 0 and valid else "failed"
    if status == "failed" and failure_stage is None:
        failure_stage = "response_evidence_invalid" if payload else "response_transport_failed"
    stderr = str(completed.get("stderr_snippet", ""))
    if validation_error:
        stderr += f"\nresponse validation failed: {validation_error}"
    if completed.get("timed_out"):
        stderr += "\nphysical DPU release is unverified after frontier host timeout"
    cleanup_confirmed = (
        bool(payload.get("release", {}).get("confirmed"))
        if isinstance(payload, Mapping) and isinstance(payload.get("release"), Mapping)
        else bool(completed.get("cleanup_confirmed", False))
    )
    return HardwareFrontierExecution(
        status=status,
        failure_stage=failure_stage,
        response_path=response,
        response=payload or {},
        process_time_s=float(completed.get("elapsed_s", 0.0)),
        command=command,
        stdout_snippet=str(completed.get("stdout_snippet", "")),
        stderr_snippet=stderr,
        timed_out=bool(completed.get("timed_out")),
        cleanup_confirmed=cleanup_confirmed,
    )


def validate_hardware_frontier_session(
    manifest_path: Path,
    response_path: Path,
    *,
    profile: HardwareFrontierProfile | Mapping[str, Any],
    build: HardwareSessionBuild | None = None,
    session_root: Path | None = None,
) -> JsonDict:
    """Validate an already-produced native response without executing it."""

    parse_hardware_frontier_profile(profile)
    root = (
        session_root
        if session_root is not None
        else build.session_root
        if build is not None
        else manifest_path.parent
    ).resolve()
    manifest = _read_manifest(_inside_session(root, manifest_path, "frontier manifest"))
    _validate_manifest_identity(manifest)
    _validate_frontier_manifest(manifest, build, root)
    response = _load_response(_inside_session(root, response_path, "frontier response"))
    if not response:
        raise ValueError("response_transport_failed: frontier response is missing or invalid")
    validate_frontier_native_response(response, manifest)
    validate_frontier_output_file(response, manifest, root)
    _validate_native_hashes(response, manifest, build, root, manifest_path)
    return response


def _required_build_tools(environment: Mapping[str, str]) -> tuple[str, JsonDict]:
    sdk = discover_upmem_sdk(env=environment)
    by_name = {tool.name: tool for tool in sdk.tools}
    make_path = shutil.which("make", path=environment.get("PATH"))
    required = ("dpu-upmem-dpurte-clang", "dpu-pkg-config")
    missing = [name for name in required if not by_name.get(name) or not by_name[name].available]
    if not make_path:
        missing.append("make")
    if missing:
        raise RuntimeError("sdk_discovery_failed: missing required UPMEM SDK tools: " + ", ".join(sorted(missing)))
    return str(make_path), {
        "make": str(make_path),
        **{name: str(by_name[name].path) if by_name[name].path else None for name in required},
    }


def _require_physical_opt_in(environment: Mapping[str, str]) -> None:
    if environment.get("UPMEM_ALLOW_PHYSICAL_HARDWARE") != "1":
        raise ValueError("hardware_opt_in_missing: UPMEM_ALLOW_PHYSICAL_HARDWARE=1 is required")
    if environment.get("DPU_BACKEND"):
        raise ValueError("hardware_profile_violation: DPU_BACKEND must be unset")
    for name in ("UPMEM_PROFILE", "UPMEM_PROFILE_BASE"):
        if "sim" in environment.get(name, "").lower():
            raise ValueError(f"hardware_profile_violation: {name} selects a simulator")


def _inside_session(root: Path, path: Path, label: str) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"hardware_profile_violation: {label} must be inside session root") from exc
    return resolved


def _read_manifest(path: Path) -> JsonDict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("manifest_parse_failed: frontier request manifest is invalid") from exc
    if not isinstance(payload, dict):
        raise ValueError("manifest_parse_failed: frontier request manifest is not an object")
    return payload


def _load_response(path: Path) -> JsonDict | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _validate_frontier_manifest(
    manifest: Mapping[str, Any], build: HardwareSessionBuild | None, root: Path
) -> None:
    _validate_manifest_identity(manifest)
    for key, expected in (
        ("schema_version", REQUEST_SCHEMA),
        ("native_schema_version", NATIVE_SCHEMA),
        ("route_id", ROUTE_ID),
        ("backend_id", BACKEND_ID),
        ("hardware_profile_version", PROFILE_ID),
        ("target", "hardware"),
        ("session_protocol", REQUEST_SCHEMA),
        ("requested_dpus", REQUESTED_DPUS),
        ("tasklets", TASKLETS_PER_DPU),
        ("tasklets_per_dpu", TASKLETS_PER_DPU),
        ("numeric_mode", NUMERIC_MODE),
        ("quantization_mode", NUMERIC_MODE),
        ("barrier_count", 2),
    ):
        if manifest.get(key) != expected:
            raise ValueError(f"hardware_profile_violation: frontier manifest {key} mismatch")
    if manifest.get("expected_dpu_task_counts") != [2, 1]:
        raise ValueError("hardware_profile_violation: frontier manifest DPU counts mismatch")
    if manifest.get("overlap_measured") is not False:
        raise ValueError("hardware_profile_violation: frontier overlap must be unmeasured")
    output_binding = manifest.get("final_output_binding")
    if not isinstance(output_binding, Mapping):
        raise ValueError("manifest_parse_failed: frontier final output binding is missing")
    output_ref = output_binding.get("output_path")
    if not isinstance(output_ref, str) or not output_ref:
        raise ValueError("manifest_parse_failed: frontier final output path is missing")
    output_path = root / output_ref
    resolved_root = root.resolve()
    try:
        output_path.resolve().relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError("hardware_profile_violation: frontier final output escapes session root") from exc
    if output_path.is_symlink():
        raise ValueError("hardware_profile_violation: frontier final output is a symlink")
    package_ref = manifest.get("package_path")
    if not isinstance(package_ref, str) or not package_ref:
        raise ValueError("manifest_parse_failed: frontier package path is missing")
    package_path = (root / package_ref).resolve()
    try:
        package_path.relative_to(root)
    except ValueError as exc:
        raise ValueError("hardware_profile_violation: frontier package escapes session root") from exc
    if not package_path.is_file():
        raise ValueError("manifest_parse_failed: frontier package file is missing")
    try:
        validate_frontier_package_against_manifest(package_path.read_bytes(), manifest)
    except (OSError, ValueError) as exc:
        raise ValueError("manifest_parse_failed: frontier package does not match manifest") from exc
    dpu_ref = manifest.get("dpu_binary")
    if not isinstance(dpu_ref, str) or not dpu_ref:
        raise ValueError("manifest_parse_failed: frontier DPU binary path is invalid")
    dpu_path = (root / dpu_ref).resolve()
    try:
        dpu_path.relative_to(root)
    except ValueError as exc:
        raise ValueError("hardware_profile_violation: frontier DPU binary escapes session root") from exc
    if build is not None and dpu_path != build.dpu_binary.resolve():
        raise ValueError("hardware_profile_violation: frontier DPU binary does not match build")
    if build is not None:
        _validate_build_hashes(build)


def _failure_stage(completed: Mapping[str, Any], response: Mapping[str, Any] | None) -> str | None:
    if completed.get("timed_out"):
        return "hardware_session_timeout"
    native_stage = response.get("failure_stage") if response else None
    if isinstance(native_stage, str) and native_stage:
        return native_stage
    if not response:
        return "response_transport_failed"
    if completed.get("returncode") != 0:
        return "native_host_failed"
    return None


_run_build_command = _run_command
_run_frontier_command = _run_resident_command


def _validate_build_hashes(build: HardwareSessionBuild) -> None:
    if build.source_tree_hash:
        resident = build.source_snapshot.parent / "upmem_sdk_generic_loop_resident"
        if _hash_combined_source_tree(build.source_snapshot, resident) != build.source_tree_hash:
            raise ValueError("hardware_profile_violation: combined native source snapshot hash mismatch")
    for field, path in (
        ("host_binary_hash", build.host_binary),
        ("dpu_binary_hash", build.dpu_binary),
    ):
        expected = getattr(build, field)
        if expected and (not path.is_file() or _hash_file(path) != expected):
            raise ValueError(f"hardware_profile_violation: {field} mismatch")


def _validate_native_hashes(
    response: Mapping[str, Any],
    manifest: Mapping[str, Any],
    build: HardwareSessionBuild | None,
    root: Path,
    manifest_path: Path,
) -> None:
    hashes = response.get("hashes")
    if not isinstance(hashes, Mapping):
        raise ValueError("response_evidence_invalid: native hashes are missing")
    package_ref = manifest.get("package_path")
    dpu_ref = manifest.get("dpu_binary")
    if not isinstance(package_ref, str) or not isinstance(dpu_ref, str):
        raise ValueError("manifest_parse_failed: native hash paths are missing")
    package_path = (root / package_ref).resolve()
    dpu_path = (root / dpu_ref).resolve()
    expected_paths: dict[str, Path] = {
        "manifest_fnv1a64": manifest_path.resolve(),
        "package_fnv1a64": package_path,
        "dpu_binary_fnv1a64": dpu_path,
    }
    # The C host hashes __FILE__, which is the copied build-tree host source.
    if build is not None:
        source_path = build.build_dir / "host.c"
        if not source_path.is_file():
            source_path = build.source_snapshot / "host.c"
        expected_paths["host_source_fnv1a64"] = source_path
    else:
        expected_paths["host_source_fnv1a64"] = root / "native" / "src" / "host.c"
    for key, path in expected_paths.items():
        if not path.is_file():
            if key == "manifest_fnv1a64":
                continue
            raise ValueError(f"response_evidence_invalid: native hash source {key} is missing")
        expected = _fnv1a64(path.read_bytes())
        if hashes.get(key) != expected:
            raise ValueError(f"response_evidence_invalid: native hash {key} mismatch")


def _hash_combined_source_tree(frontier: Path, resident: Path) -> str:
    digest = hashlib.sha256()
    for label, path in (("frontier", frontier), ("resident", resident)):
        if not path.is_dir():
            raise ValueError("hardware_profile_violation: combined native source snapshot is incomplete")
        for item in sorted(path.rglob("*")):
            if item.is_file():
                digest.update(f"{label}/{item.relative_to(path).as_posix()}".encode("utf-8"))
                digest.update(_hash_file(item).encode("ascii"))
    return digest.hexdigest()


def _fnv1a64(value: bytes) -> str:
    result = 14695981039346656037
    for byte in value:
        result ^= byte
        result = (result * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return f"{result:016x}"

build_frontier_hardware_session = build_hardware_frontier_session
execute_frontier_hardware_session = execute_hardware_frontier_session
validate_frontier_hardware_session = validate_hardware_frontier_session


__all__ = [
    "BACKEND_ID",
    "DPU_BINARY_NAME",
    "HOST_BINARY_NAME",
    "HardwareFrontierExecution",
    "HardwareFrontierProfile",
    "NATIVE_SCHEMA",
    "NATIVE_SOURCE_DIR",
    "PROFILE_ID",
    "ROUTE_ID",
    "build_frontier_hardware_session",
    "build_hardware_frontier_session",
    "execute_frontier_hardware_session",
    "execute_hardware_frontier_session",
    "parse_hardware_frontier_profile",
    "validate_frontier_hardware_session",
    "validate_hardware_frontier_session",
]
