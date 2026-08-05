"""SimplePIM-management adapter for the small M4.1 frontier fixture.

This module intentionally owns only the provider seam.  The package format,
frontier validation and thesis-owned resident kernel remain the M3.1 contracts.
SimplePIM manages the physical allocation; the native host still performs the
custom package load, transfers, launch and synchronous release.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import subprocess
import time
from typing import Any, Mapping

from quantum_bench.core.records import JsonDict
from quantum_bench.targets.upmem.hardware_frontier_session import (
    BUILD_TIMEOUT_S,
    DEFAULT_TIMEOUT_S,
    HardwareFrontierExecution,
    _failure_stage,
    _inside_session,
    _load_response,
    _require_physical_opt_in,
    _read_manifest,
    _required_build_tools,
    _validate_frontier_manifest,
    _validate_native_hashes,
    _hash_combined_source_tree,
)
from quantum_bench.targets.upmem.hardware_session import (
    HardwareSessionBuild,
    _copy_source_tree,
    _hash_file,
    _run_command,
    _run_resident_command,
    _sanitised_hardware_env,
)
from quantum_bench.targets.upmem.hardware_taskgraph_frontier import (
    BACKEND_ID as RAW_BACKEND_ID,
    PROFILE_ID as RAW_PROFILE_ID,
    ROUTE_ID as RAW_ROUTE_ID,
    validate_frontier_native_response,
    validate_frontier_output_file,
)


NATIVE_SOURCE_DIR = "upmem_sdk_generic_loop_frontier_two_dpu"
HOST_BINARY_NAME = "host_frontier_two_dpu_simplepim_management"
DPU_BINARY_NAME = "dpu_frontier_two_dpu_simplepim_management"
INITIALIZATION_BINARY_NAME = "dpu_simplepim_management_init"
SIMPLEPIM_SOURCE_COMMIT = "1d639c53532555f01e9f71d872e7712b166d6cba"
SIMPLEPIM_STAGE_RELATIVE = Path("build/simplepim_frontier_two_dpu/staged/SimplePIM")
SIMPLEPIM_STAGE_MARKER = "management_profile_manifest.json"
M41_PROFILE_ID = "hardware_frontier_two_dpu_m4_1_v1"
M41_BACKEND_ID = "upmem_sdk_hardware_taskgraph_simplepim_management_frontier_two_dpu"
M41_ROUTE_ID = "upmem_tn_hardware_taskgraph_simplepim_management_frontier_two_dpu"
SIMPLEPIM_PROVIDER_ID = "simplepim_management"
SIMPLEPIM_MANAGEMENT_API = "simplepim_management_init_physical_v1"


@dataclass(frozen=True)
class SimplePimFrontierProfile:
    version: str
    target: str
    requested_dpu_count: int
    tasklets_per_dpu: int
    numeric_mode: str
    timeout_s: float
    performance_claim_applicable: bool


@dataclass(frozen=True)
class SimplePimFrontierBuild:
    raw: HardwareSessionBuild
    simplepim: HardwareSessionBuild
    simplepim_root: Path
    simplepim_source_commit: str
    simplepim_source_dirty: bool
    simplepim_staged_source_tree_sha256: str
    simplepim_patch_sha256: str
    simplepim_stage_manifest: Path
    simplepim_stage_manifest_sha256: str
    simplepim_initialization_binary_hash: str
    build_result: JsonDict

    @property
    def session_root(self) -> Path:
        return self.simplepim.session_root

    @property
    def build_dir(self) -> Path:
        return self.simplepim.build_dir


def parse_simplepim_frontier_profile(
    value: SimplePimFrontierProfile | Mapping[str, Any],
) -> SimplePimFrontierProfile:
    if isinstance(value, SimplePimFrontierProfile):
        return value
    if not isinstance(value, Mapping):
        raise ValueError("hardware_profile_violation: M4.1 profile must be a mapping")
    expected = {
        "hardware_profile_version": M41_PROFILE_ID,
        "target": "hardware",
        "requested_dpu_count": 2,
        "tasklets_per_dpu": 1,
        "numeric_mode": "none",
        "performance_claim_applicable": False,
    }
    for key, expected_value in expected.items():
        if value.get(key) != expected_value:
            raise ValueError(f"hardware_profile_violation: {key} must be {expected_value!r}")
    timeout = value.get("timeout_s", DEFAULT_TIMEOUT_S)
    if not isinstance(timeout, (int, float)) or isinstance(timeout, bool) or float(timeout) <= 0:
        raise ValueError("hardware_profile_violation: timeout_s must be positive")
    return SimplePimFrontierProfile(
        version=M41_PROFILE_ID,
        target="hardware",
        requested_dpu_count=2,
        tasklets_per_dpu=1,
        numeric_mode="none",
        timeout_s=float(timeout),
        performance_claim_applicable=False,
    )


def build_simplepim_frontier_session(
    root_dir: Path,
    session_root: Path,
    *,
    profile: SimplePimFrontierProfile | Mapping[str, Any],
    environment: Mapping[str, str],
) -> SimplePimFrontierBuild:
    """Build raw and SimplePIM provider binaries in one isolated session."""

    parse_simplepim_frontier_profile(profile)
    make_path, sdk_tools = _required_build_tools(environment)
    source = root_dir / "native" / "upmem" / "simplepim" / NATIVE_SOURCE_DIR
    resident_source = source.parent / "upmem_sdk_generic_loop_resident"
    simplepim_root = (root_dir / "external" / "SimplePIM").resolve()
    if not source.is_dir() or not resident_source.is_dir():
        raise RuntimeError("native_build_failed: frontier native sources are missing")
    if not (simplepim_root / ".git").exists():
        raise RuntimeError("native_build_failed: pinned external/SimplePIM checkout is missing")
    source_commit = _git_commit(simplepim_root)
    source_dirty = _git_dirty(simplepim_root)
    if source_dirty:
        raise RuntimeError("native_build_failed: external/SimplePIM checkout is dirty")
    if source_commit != SIMPLEPIM_SOURCE_COMMIT:
        raise RuntimeError(
            "native_build_failed: SimplePIM source commit mismatch "
            f"expected {SIMPLEPIM_SOURCE_COMMIT}, got {source_commit}"
        )

    isolated_parent = session_root / "native"
    source_snapshot = isolated_parent / "src"
    build_dir = isolated_parent / "build"
    _copy_source_tree(resident_source, isolated_parent / resident_source.name)
    _copy_source_tree(source, source_snapshot)
    _copy_source_tree(source, build_dir)
    source_tree_hash = _hash_combined_source_tree(
        source_snapshot, isolated_parent / resident_source.name
    )
    patch_path = build_dir / "simplepim_management_profile.patch"
    stage_manifest = build_dir / SIMPLEPIM_STAGE_RELATIVE / SIMPLEPIM_STAGE_MARKER
    command = (
        str(make_path),
        "clean",
        "all",
        "simplepim-all",
        "NR_TASKLETS=1",
        "UPMEM_GENERIC_HARDWARE_MVP=1",
        "FRONTIER_TASK_COUNT=3",
        "FRONTIER_BARRIER_COUNT=2",
        f"SIMPLEPIM_ROOT={simplepim_root}",
    )
    started = time.perf_counter()
    result = _run_command(
        command,
        cwd=build_dir,
        env=_sanitised_hardware_env(environment),
        timeout_s=BUILD_TIMEOUT_S,
    )
    build_result = {
        **result,
        "command": list(command),
        "cwd": str(build_dir),
        "simplepim_root": str(simplepim_root),
        "simplepim_source_commit": source_commit,
    }
    if result.get("returncode") != 0:
        stage = "native_build_timeout" if result.get("timed_out") else "native_build_failed"
        detail = result.get("stderr_snippet") or result.get("error") or "make failed"
        raise RuntimeError(f"{stage}: {detail}")
    if not stage_manifest.is_file():
        raise RuntimeError("native_build_failed: SimplePIM stage marker is missing")
    stage_payload = json.loads(stage_manifest.read_text(encoding="utf-8"))
    if stage_payload.get("source_commit") != source_commit or stage_payload.get("patch_applied") is not True:
        raise RuntimeError("native_build_failed: SimplePIM stage marker is invalid")
    patch_sha256 = hashlib.sha256(patch_path.read_bytes()).hexdigest()
    if stage_payload.get("patch_sha256") != patch_sha256:
        raise RuntimeError("native_build_failed: SimplePIM patch hash mismatch")
    staged_source_tree_sha256 = stage_payload.get("staged_source_tree_sha256")
    if not isinstance(staged_source_tree_sha256, str) or not staged_source_tree_sha256:
        raise RuntimeError("native_build_failed: SimplePIM staged source hash is missing")
    initialization_binary = build_dir / "bin" / INITIALIZATION_BINARY_NAME
    if not initialization_binary.is_file():
        raise RuntimeError("native_build_failed: SimplePIM initialization binary is missing")

    raw_build = HardwareSessionBuild(
        session_root=session_root,
        source_snapshot=source_snapshot,
        build_dir=build_dir,
        host_binary=build_dir / "bin" / "host_frontier_two_dpu",
        dpu_binary=build_dir / "bin" / "dpu_frontier_two_dpu",
        source_tree_hash=source_tree_hash,
        host_binary_hash=_hash_file(build_dir / "bin" / "host_frontier_two_dpu"),
        dpu_binary_hash=_hash_file(build_dir / "bin" / "dpu_frontier_two_dpu"),
        build_time_s=time.perf_counter() - started,
        build_command=command,
        sdk_tools=sdk_tools,
    )
    simplepim_build = HardwareSessionBuild(
        session_root=session_root,
        source_snapshot=source_snapshot,
        build_dir=build_dir,
        host_binary=build_dir / "bin" / HOST_BINARY_NAME,
        dpu_binary=build_dir / "bin" / DPU_BINARY_NAME,
        source_tree_hash=source_tree_hash,
        host_binary_hash=_hash_file(build_dir / "bin" / HOST_BINARY_NAME),
        dpu_binary_hash=_hash_file(build_dir / "bin" / DPU_BINARY_NAME),
        build_time_s=time.perf_counter() - started,
        build_command=command,
        sdk_tools=sdk_tools,
    )
    return SimplePimFrontierBuild(
        raw=raw_build,
        simplepim=simplepim_build,
        simplepim_root=simplepim_root,
        simplepim_source_commit=source_commit,
        simplepim_source_dirty=source_dirty,
        simplepim_staged_source_tree_sha256=staged_source_tree_sha256,
        simplepim_patch_sha256=patch_sha256,
        simplepim_stage_manifest=stage_manifest,
        simplepim_stage_manifest_sha256=hashlib.sha256(stage_manifest.read_bytes()).hexdigest(),
        simplepim_initialization_binary_hash=_hash_file(initialization_binary),
        build_result=build_result,
    )


def execute_simplepim_frontier_session(
    build: SimplePimFrontierBuild,
    *,
    manifest_path: Path,
    response_path: Path,
    profile: SimplePimFrontierProfile | Mapping[str, Any],
    environment: Mapping[str, str],
) -> HardwareFrontierExecution:
    """Execute exactly one SimplePIM-managed physical frontier request."""

    selected = parse_simplepim_frontier_profile(profile)
    _require_physical_opt_in(environment)
    base = build.simplepim
    root = base.session_root.resolve()
    manifest_file = _inside_session(root, manifest_path, "frontier manifest")
    response_file = _inside_session(root, response_path, "frontier response")
    manifest = _read_manifest(manifest_file)
    _validate_frontier_manifest(manifest, base, root)
    command = (
        str(base.host_binary.resolve()),
        "--frontier-package",
        str(manifest_file.resolve()),
        "--frontier-response",
        str(response_file),
    )
    completed = _run_resident_command(
        command,
        cwd=base.build_dir,
        env=_sanitised_hardware_env(environment),
        timeout_s=selected.timeout_s,
    )
    payload = _load_response(response_file)
    failure_stage = _failure_stage(completed, payload)
    valid = False
    validation_error: str | None = None
    if payload:
        try:
            validate_simplepim_frontier_response(payload, manifest)
            validate_frontier_output_file(payload, manifest, root)
            _validate_native_hashes(payload, manifest, base, root, manifest_file)
            valid = True
        except ValueError as exc:
            validation_error = str(exc)
    status = "completed" if completed.get("returncode") == 0 and valid else "failed"
    if status == "failed" and failure_stage is None:
        failure_stage = "response_evidence_invalid" if payload else "response_transport_failed"
    stderr = str(completed.get("stderr_snippet", ""))
    if validation_error:
        stderr += f"\nresponse validation failed: {validation_error}"
    if completed.get("timed_out"):
        stderr += "\nphysical DPU release is unverified after SimplePIM host timeout"
    cleanup_confirmed = bool(
        payload.get("release", {}).get("confirmed")
        if isinstance(payload, Mapping) and isinstance(payload.get("release"), Mapping)
        else completed.get("cleanup_confirmed", False)
    )
    return HardwareFrontierExecution(
        status=status,
        failure_stage=failure_stage,
        response_path=response_file,
        response=payload or {},
        process_time_s=float(completed.get("elapsed_s", 0.0)),
        command=command,
        stdout_snippet=str(completed.get("stdout_snippet", "")),
        stderr_snippet=stderr,
        timed_out=bool(completed.get("timed_out")),
        cleanup_confirmed=cleanup_confirmed,
    )


def validate_simplepim_frontier_response(
    response: Mapping[str, Any], manifest: Mapping[str, Any]
) -> None:
    """Validate SimplePIM provider truth plus the unchanged M3.1 ABI contract."""

    if response.get("provider_id") != SIMPLEPIM_PROVIDER_ID:
        raise ValueError("response_evidence_invalid: SimplePIM provider identity mismatch")
    required = {
        "control_provider": SIMPLEPIM_PROVIDER_ID,
        "kernel_provider": "thesis_resident_generic_contract",
        "simplepim_management_api_used": SIMPLEPIM_MANAGEMENT_API,
        "provider_init_called": True,
        "provider_init_succeeded": True,
        "simplepim_management_init_called": True,
        "simplepim_management_allocation_used": True,
        "simplepim_management_object_created": True,
        "allocation_source": SIMPLEPIM_PROVIDER_ID,
        "allocation_profile": "backend=hw",
        "simplepim_operator_api_used": False,
        "simplepim_operator_names": [],
        "simplepim_kernel_executed": False,
        "raw_sdk_direct_allocation_used": False,
        "raw_sdk_load_used": True,
        "raw_sdk_transfer_used": True,
        "raw_sdk_launch_used": True,
        "raw_sdk_sync_used": True,
        "raw_sdk_control_calls_used": True,
        "any_task_completed": True,
        "thesis_owned_kernel_executed": True,
        "thesis_resident_kernel_executed": True,
        "simplepim_heap_used": False,
        "simplepim_table_transport_used": False,
        "all_tasks_completed": True,
        "complete_taskgraph_executed": True,
        "provider_release_attempted": True,
        "provider_release_succeeded": True,
        "provider_release_error": 0,
        "timing_is_bringup_only": True,
        "hardware_speedup_applicable": False,
    }
    for key, expected in required.items():
        if response.get(key) != expected:
            raise ValueError(f"response_evidence_invalid: SimplePIM field {key} is invalid")
    normalized = dict(response)
    normalized.update(
        route_id=RAW_ROUTE_ID,
        backend_id=RAW_BACKEND_ID,
        hardware_profile_version=RAW_PROFILE_ID,
    )
    validate_frontier_native_response(normalized, manifest)


def simplepim_build_metadata(build: SimplePimFrontierBuild, root: Path) -> JsonDict:
    stage_rel = build.simplepim_stage_manifest.resolve().relative_to(root.resolve())
    return {
        "build_result": dict(build.build_result),
        "simplepim_root": str(build.simplepim_root),
        "simplepim_source_commit": build.simplepim_source_commit,
        "simplepim_source_dirty": build.simplepim_source_dirty,
        "simplepim_staged_source_tree_sha256": build.simplepim_staged_source_tree_sha256,
        "simplepim_patch_sha256": build.simplepim_patch_sha256,
        "simplepim_management_api": SIMPLEPIM_MANAGEMENT_API,
        "simplepim_management_extension": "table_management_init_with_profile",
        "simplepim_stage_manifest_sha256": build.simplepim_stage_manifest_sha256,
        "simplepim_initialization_binary_hash": build.simplepim_initialization_binary_hash,
        "simplepim_stage_manifest": stage_rel.as_posix(),
        "raw": _build_metadata(build.raw, root),
        "simplepim_management": _build_metadata(build.simplepim, root),
    }


def _build_metadata(build: HardwareSessionBuild, root: Path) -> JsonDict:
    return {
        "host_binary": build.host_binary.resolve().relative_to(root.resolve()).as_posix(),
        "dpu_binary": build.dpu_binary.resolve().relative_to(root.resolve()).as_posix(),
        "host_binary_hash": build.host_binary_hash,
        "dpu_binary_hash": build.dpu_binary_hash,
        "source_tree_hash": build.source_tree_hash,
        "build_command": list(build.build_command),
        "build_time_s": build.build_time_s,
        "sdk_tools": build.sdk_tools,
    }


def _git_commit(path: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "--verify", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError("native_build_failed: unable to resolve SimplePIM source commit")
    return completed.stdout.strip()


def _git_dirty(path: Path) -> bool:
    completed = subprocess.run(
        ["git", "-C", str(path), "status", "--porcelain", "--untracked-files=all"],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError("native_build_failed: unable to inspect SimplePIM checkout")
    return bool(completed.stdout.strip())


__all__ = [
    "M41_BACKEND_ID",
    "M41_PROFILE_ID",
    "M41_ROUTE_ID",
    "SIMPLEPIM_MANAGEMENT_API",
    "SIMPLEPIM_PROVIDER_ID",
    "SimplePimFrontierBuild",
    "SimplePimFrontierProfile",
    "build_simplepim_frontier_session",
    "execute_simplepim_frontier_session",
    "parse_simplepim_frontier_profile",
    "simplepim_build_metadata",
    "validate_simplepim_frontier_response",
]
