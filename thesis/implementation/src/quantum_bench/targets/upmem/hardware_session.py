"""Bounded native-session adapter for physical generic UPMEM contractions.

This adapter owns the Python side of ``upmem_generic_session_v1``.  It builds
the thesis-owned DPU source once per benchmark run and invokes the native host
session for one ordered logical TaskGraph step at a time.  A logical step may
contain one real contraction or four split-complex components.  The host still
owns allocation, load, transfers, synchronous launch, and release; Python
never substitutes a CPU contraction for a native result.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shutil
import struct
import subprocess
import time
from typing import Any, Mapping, Sequence

import numpy as np

from quantum_bench.core.records import JsonDict
from quantum_bench.routing.generic_prepare import (
    GENERIC_MODE_FLOAT32_NO_QUANT,
    GenericTaskPreparationResult,
)
from quantum_bench.targets.upmem.environment import discover_upmem_sdk
from quantum_bench.targets.upmem.hardware_taskgraph import HardwareTaskGraphProfile


HARDWARE_GENERIC_SESSION_SCHEMA_VERSION = "upmem_generic_session_v1"
HARDWARE_GENERIC_SESSION_INPUT_KIND = "upmem_generic_session_input"
HARDWARE_GENERIC_SESSION_OUTPUT_KIND = "upmem_generic_session_response"
HARDWARE_GENERIC_SESSION_MAX_TASKS = 1024


@dataclass(frozen=True)
class HardwareSessionBuild:
    session_root: Path
    source_snapshot: Path
    build_dir: Path
    host_binary: Path
    dpu_binary: Path
    source_tree_hash: str
    host_binary_hash: str
    dpu_binary_hash: str
    build_time_s: float
    build_command: tuple[str, ...]
    sdk_tools: JsonDict


@dataclass(frozen=True)
class HardwareSessionTask:
    task_id: str
    args_path: Path
    left_path: Path
    right_path: Path
    output_path: Path
    output_shape: tuple[int, ...]
    operand_mode: str
    output_scale: float
    preparation: GenericTaskPreparationResult
    application_visible_h2d_bytes: int
    application_visible_d2h_bytes: int

    @property
    def application_visible_transfer_bytes(self) -> int:
        return self.application_visible_h2d_bytes + self.application_visible_d2h_bytes


@dataclass(frozen=True)
class HardwareSessionExecution:
    status: str
    failure_stage: str | None
    response_path: Path
    response: JsonDict
    process_time_s: float
    command: tuple[str, ...]
    stdout_snippet: str
    stderr_snippet: str


def build_hardware_session(
    root_dir: Path,
    session_root: Path,
    *,
    profile: HardwareTaskGraphProfile,
    environment: Mapping[str, str],
) -> HardwareSessionBuild:
    """Build the bounded native source once for a hardware TaskGraph run."""

    sdk = discover_upmem_sdk(env=environment)
    required = {
        tool.name: tool
        for tool in sdk.tools
        if tool.name in {"make", "dpu-upmem-dpurte-clang", "dpu-pkg-config"}
    }
    missing = sorted(name for name, tool in required.items() if not tool.available)
    if missing:
        raise RuntimeError(
            "sdk_discovery_failed: missing required UPMEM SDK tools: "
            + ", ".join(missing)
        )

    source = root_dir / "native" / "upmem" / "simplepim" / "upmem_sdk_generic_loop"
    source_snapshot = session_root / "native" / "src"
    build_dir = session_root / "native" / "build"
    _copy_source_tree(source, source_snapshot)
    _copy_source_tree(source, build_dir)
    child_env = _sanitised_hardware_env(environment)
    command = (
        "make",
        "clean",
        "all",
        f"MAX_RANK={profile.max_rank}",
        f"MAX_ELEMS={profile.max_tensor_elements}",
        f"OUTPUT_TILE_ELEMS={profile.output_tile_elements}",
        "NR_TASKLETS=1",
        # Compile the existing native ``backend=hw`` allocation profile into
        # this dedicated physical route.  The sanitised child environment
        # removes simulator selectors, so a failure cannot silently become a
        # simulator run.
        "UPMEM_GENERIC_HARDWARE_MVP=1",
    )
    started = time.perf_counter()
    completed = _run_command(
        command, cwd=build_dir, env=child_env, timeout_s=profile.timeout_s
    )
    build_time_s = time.perf_counter() - started
    if completed["returncode"] != 0:
        raise RuntimeError("native_build_failed: " + completed["stderr_snippet"])
    host_binary = build_dir / "bin" / "host"
    dpu_binary = build_dir / "bin" / "dpu_generic"
    if not host_binary.is_file() or not dpu_binary.is_file():
        raise RuntimeError(
            "native_build_failed: expected native host and DPU binaries were not produced"
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
        sdk_tools={
            name: str(tool.path) if tool.path else None
            for name, tool in required.items()
        },
    )


def write_session_task(
    task_root: Path,
    *,
    sequence: int,
    task_id: str,
    preparation: GenericTaskPreparationResult,
    max_rank: int,
) -> HardwareSessionTask:
    """Persist one prepared real component for the native session protocol."""

    if preparation.status != "prepared" or preparation.prepared_operands is None:
        raise ValueError(f"hardware_profile_violation: task {task_id} is not prepared")
    operands = preparation.prepared_operands
    task_dir = task_root / "tasks" / f"{sequence:04d}_{_safe_name(task_id)}"
    task_dir.mkdir(parents=True, exist_ok=False)
    args_path = task_dir / "args.bin"
    left_path = task_dir / "left.bin"
    right_path = task_dir / "right.bin"
    output_path = task_dir / "output.bin"
    native = {
        "left_rank": len(preparation.input_shapes[0]),
        "right_rank": len(preparation.input_shapes[1]),
        "output_rank": len(preparation.output_shape),
        "contracted_rank": len(preparation.contracted_dims),
        "left_shape": preparation.input_shapes[0],
        "right_shape": preparation.input_shapes[1],
        "output_shape": preparation.output_shape,
        "contracted_dims": preparation.contracted_dims,
        "left_strides": preparation.left_strides,
        "right_strides": preparation.right_strides,
        "output_strides": preparation.output_strides,
        "output_to_left_axes": preparation.output_to_left_axes,
        "output_to_right_axes": preparation.output_to_right_axes,
        "contracted_to_left_axes": preparation.contracted_to_left_axes,
        "contracted_to_right_axes": preparation.contracted_to_right_axes,
        "output_element_count": preparation.output_element_count,
        "contracted_combination_count": preparation.contracted_combination_count,
        "operand_mode": operands.operand_mode,
    }
    # The serialized struct must use the binary's compiled rank, not the
    # individual task rank.  Otherwise C would read past args.bin.
    args_payload = pack_generic_args(native, max_rank=max_rank)
    args_path.write_bytes(args_payload)
    input_dtype = (
        np.dtype("<f4")
        if operands.operand_mode == GENERIC_MODE_FLOAT32_NO_QUANT
        else np.dtype("int8")
    )
    np.asarray(operands.left_operand, dtype=input_dtype).ravel().tofile(left_path)
    np.asarray(operands.right_operand, dtype=input_dtype).ravel().tofile(right_path)
    output_itemsize = 4
    h2d = (
        len(args_payload)
        + _align8(left_path.stat().st_size)
        + _align8(right_path.stat().st_size)
    )
    d2h = _align8(int(preparation.output_element_count) * output_itemsize)
    return HardwareSessionTask(
        task_id=task_id,
        args_path=args_path,
        left_path=left_path,
        right_path=right_path,
        output_path=output_path,
        output_shape=tuple(int(dim) for dim in preparation.output_shape),
        operand_mode=operands.operand_mode,
        output_scale=_output_scale(preparation),
        preparation=preparation,
        application_visible_h2d_bytes=h2d,
        application_visible_d2h_bytes=d2h,
    )


def execute_hardware_session(
    build: HardwareSessionBuild,
    *,
    session_id: str,
    tasks: Sequence[HardwareSessionTask],
    profile: HardwareTaskGraphProfile,
    environment: Mapping[str, str],
) -> HardwareSessionExecution:
    if not tasks:
        raise ValueError(
            "hardware_profile_violation: native session requires at least one task"
        )
    if len(tasks) > HARDWARE_GENERIC_SESSION_MAX_TASKS:
        raise ValueError(
            "hardware_profile_violation: native session task limit exceeded"
        )
    for task in tasks:
        for path in (task.args_path, task.left_path, task.right_path, task.output_path):
            try:
                path.resolve().relative_to(build.session_root.resolve())
            except ValueError as exc:
                raise ValueError(
                    "hardware_profile_violation: session task paths must be inside the native session root"
                ) from exc
    manifest_path = build.session_root / f"{_safe_name(session_id)}_manifest.json"
    response_path = build.session_root / f"{_safe_name(session_id)}_response.json"
    payload = {
        "schema_version": HARDWARE_GENERIC_SESSION_SCHEMA_VERSION,
        "manifest_kind": HARDWARE_GENERIC_SESSION_INPUT_KIND,
        "session_id": session_id,
        "dpu_binary": _relative(build.session_root, build.dpu_binary),
        "requested_dpus": profile.requested_dpu_count,
        "tasklets": profile.tasklets_per_dpu,
        "tasks": [
            {
                "task_id": task.task_id,
                "args_path": _relative(build.session_root, task.args_path),
                "left_path": _relative(build.session_root, task.left_path),
                "right_path": _relative(build.session_root, task.right_path),
                "output_path": _relative(build.session_root, task.output_path),
            }
            for task in tasks
        ],
    }
    # The native parser accepts UTF-8 bytes but intentionally does not decode
    # JSON ``\\uXXXX`` escapes.  Keep identifiers and paths representable on
    # non-ASCII worktrees without weakening its safe-relative-path checks.
    manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    command = (
        str(build.host_binary),
        "--session-manifest",
        str(manifest_path),
        "--response-manifest",
        str(response_path),
    )
    completed = _run_command(
        command,
        cwd=build.build_dir,
        env=_sanitised_hardware_env(environment),
        timeout_s=profile.timeout_s + 5.0,
    )
    response = _load_response(response_path)
    failure_stage = (
        response.get("failure_stage") if isinstance(response, dict) else None
    )
    response_ok = _session_response_valid(response, tasks, profile)
    status = "completed" if completed["returncode"] == 0 and response_ok else "failed"
    if status == "failed" and not failure_stage:
        failure_stage = (
            "kernel_timeout" if completed["timed_out"] else "output_manifest_failed"
        )
    return HardwareSessionExecution(
        status=status,
        failure_stage=str(failure_stage) if failure_stage else None,
        response_path=response_path,
        response=response,
        process_time_s=float(completed["elapsed_s"]),
        command=command,
        stdout_snippet=str(completed["stdout_snippet"]),
        stderr_snippet=(
            str(completed["stderr_snippet"])
            + (
                "\nphysical DPU release is unverified after host-process timeout; inspect allocation before rerunning"
                if completed["timed_out"]
                else ""
            )
        ),
    )


def load_session_output(task: HardwareSessionTask) -> np.ndarray:
    if not task.output_path.is_file():
        raise RuntimeError(
            "result_transfer_failed: native session did not write task output"
        )
    if task.operand_mode == GENERIC_MODE_FLOAT32_NO_QUANT:
        raw = np.fromfile(task.output_path, dtype="<f4")
        return raw.reshape(task.output_shape).astype(np.float64)
    raw = np.fromfile(task.output_path, dtype="<i4").reshape(task.output_shape)
    return raw.astype(np.float64) * task.output_scale


def pack_generic_args(native: Mapping[str, Any], *, max_rank: int) -> bytes:
    """Pack ``upmem_generic_args_t`` exactly as the C generic-loop ABI expects."""

    if max_rank < 1:
        raise ValueError("max_rank must be positive")

    def unsigned(name: str) -> list[int]:
        values = [int(value) for value in native.get(name, ())]
        return (values + [0] * max_rank)[:max_rank]

    def signed(name: str) -> list[int]:
        values = [int(value) for value in native.get(name, ())]
        return (values + [-1] * max_rank)[:max_rank]

    mode = 1 if str(native.get("operand_mode")) == GENERIC_MODE_FLOAT32_NO_QUANT else 0
    fields = [
        int(native["left_rank"]),
        int(native["right_rank"]),
        int(native["output_rank"]),
        int(native["contracted_rank"]),
        _product(tuple(int(value) for value in native["left_shape"])),
        _product(tuple(int(value) for value in native["right_shape"])),
        int(native["output_element_count"]),
        int(native["contracted_combination_count"]),
        mode,
    ]
    for key in (
        "left_shape",
        "right_shape",
        "output_shape",
        "contracted_dims",
        "left_strides",
        "right_strides",
        "output_strides",
    ):
        fields.extend(unsigned(key))
    signed_fields: list[int] = []
    for key in (
        "output_to_left_axes",
        "output_to_right_axes",
        "contracted_to_left_axes",
        "contracted_to_right_axes",
    ):
        signed_fields.extend(signed(key))
    return struct.pack(
        "<" + "I" * len(fields) + "i" * len(signed_fields), *(fields + signed_fields)
    )


def _session_response_valid(
    response: JsonDict,
    tasks: Sequence[HardwareSessionTask],
    profile: HardwareTaskGraphProfile,
) -> bool:
    if response.get("schema_version") != HARDWARE_GENERIC_SESSION_SCHEMA_VERSION:
        return False
    if response.get("manifest_kind") != HARDWARE_GENERIC_SESSION_OUTPUT_KIND:
        return False
    if (
        response.get("status") != "completed"
        or response.get("failure_stage") is not None
    ):
        return False
    if response.get("requested_dpus") != profile.requested_dpu_count:
        return False
    if response.get("allocated_dpus") != profile.requested_dpu_count:
        return False
    if response.get("tasklets") != profile.tasklets_per_dpu:
        return False
    raw_tasks = response.get("tasks")
    if not isinstance(raw_tasks, list) or len(raw_tasks) != len(tasks):
        return False
    return all(
        item.get("task_id") == task.task_id and item.get("status") == "completed"
        for item, task in zip(raw_tasks, tasks)
    )


def _output_scale(preparation: GenericTaskPreparationResult) -> float:
    left = preparation.left_conversion
    right = preparation.right_conversion
    if left is None or right is None:
        return 1.0
    return float(left.scale) * float(right.scale)


def _relative(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _product(values: tuple[int, ...]) -> int:
    result = 1
    for value in values:
        result *= value
    return result


def _align8(value: int) -> int:
    return (int(value) + 7) & ~7


def _safe_name(value: str) -> str:
    return "".join(
        character if character.isalnum() or character in {"-", "_"} else "_"
        for character in value
    )


def _copy_source_tree(source: Path, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(
        source,
        destination,
        ignore=shutil.ignore_patterns("bin", "__pycache__", "*.pyc"),
    )


def _sanitised_hardware_env(environment: Mapping[str, str]) -> dict[str, str]:
    result = dict(environment)
    for name in ("UPMEM_PROFILE", "UPMEM_PROFILE_BASE", "DPU_BACKEND"):
        result.pop(name, None)
    return result


def _run_command(
    command: Sequence[str], *, cwd: Path, env: Mapping[str, str], timeout_s: float
) -> JsonDict:
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            tuple(command),
            cwd=cwd,
            env=dict(env),
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_s,
        )
        return {
            "returncode": completed.returncode,
            "elapsed_s": time.perf_counter() - started,
            "timed_out": False,
            "stdout_snippet": _snippet(completed.stdout),
            "stderr_snippet": _snippet(completed.stderr),
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "returncode": None,
            "elapsed_s": time.perf_counter() - started,
            "timed_out": True,
            "stdout_snippet": _snippet(exc.stdout),
            "stderr_snippet": _snippet(exc.stderr),
        }


def _load_response(path: Path) -> JsonDict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _hash_tree(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(path.rglob("*")):
        if item.is_file():
            digest.update(item.relative_to(path).as_posix().encode("utf-8"))
            digest.update(_hash_file(item).encode("ascii"))
    return digest.hexdigest()


def _snippet(value: object, limit: int = 4000) -> str:
    if value is None:
        return ""
    text = (
        value.decode("utf-8", errors="replace")
        if isinstance(value, bytes)
        else str(value)
    )
    return text if len(text) <= limit else text[:limit] + "\n...[truncated]"
