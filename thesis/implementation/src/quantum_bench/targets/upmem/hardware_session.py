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
import math
from pathlib import Path
import queue
import signal
import shutil
import struct
import subprocess
import threading
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
from quantum_bench.targets.upmem.hardware_taskgraph_resident import (
    RESIDENT_ALLOCATION_PROFILE,
    RESIDENT_BACKEND_ID,
    RESIDENT_CONTROL_H2D_BYTES_PER_LAUNCH,
    RESIDENT_COMPLEX_POLICY,
    RESIDENT_DESCRIPTOR_CONTROL_BYTES,
    RESIDENT_MAX_COMPONENT_OPS,
    RESIDENT_MAX_CONTRACTED_COMBINATIONS,
    RESIDENT_MAX_ELEMENTS,
    RESIDENT_MAX_LOGICAL_TASKS,
    RESIDENT_MAX_RANK,
    RESIDENT_MAX_SLOT_DESCRIPTORS,
    RESIDENT_MRAM_POOL_BYTES,
    RESIDENT_NUMERIC_MODES,
    RESIDENT_OPERATION_BYTES,
    RESIDENT_PROFILE_VERSION,
    RESIDENT_M46_PROFILE_VERSION,
    RESIDENT_M46_OUTPUT_TILE_ELEMENTS,
    RESIDENT_SUPPORTED_TASKLETS,
    RESIDENT_ROUTE_ID,
    RESIDENT_SESSION_PROTOCOL,
    RESIDENT_TIMING_SCOPE,
    RESIDENT_OUTPUT_TILE_ELEMENTS,
)


HARDWARE_GENERIC_SESSION_SCHEMA_VERSION = "upmem_generic_session_v1"
HARDWARE_GENERIC_SESSION_INPUT_KIND = "upmem_generic_session_input"
HARDWARE_GENERIC_SESSION_OUTPUT_KIND = "upmem_generic_session_response"
HARDWARE_GENERIC_SESSION_MAX_TASKS = 1024
HARDWARE_INTERACTIVE_SESSION_SCHEMA_VERSION = "generic_loop_interactive_session_v1"
HARDWARE_INTERACTIVE_BOOTSTRAP_KIND = "bootstrap"
HARDWARE_INTERACTIVE_REQUEST_KIND = "request"
HARDWARE_INTERACTIVE_RESPONSE_KIND = "response"


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


@dataclass(frozen=True)
class HardwareSessionClose:
    status: str
    failure_stage: str | None
    release_confirmed: bool
    release_time_s: float | None
    process_returncode: int | None
    stdout_snippet: str
    stderr_snippet: str


class HardwareInteractiveSessionError(RuntimeError):
    def __init__(
        self, failure_stage: str, message: str, *, stdout: str = "", stderr: str = ""
    ) -> None:
        super().__init__(f"{failure_stage}: {message}")
        self.failure_stage = failure_stage
        self.stdout_snippet = stdout
        self.stderr_snippet = stderr


class HardwareInteractiveSession:
    """Long-lived native host for ordered one- or four-component requests."""

    def __init__(
        self,
        process: subprocess.Popen[str],
        *,
        session_root: Path,
        command: tuple[str, ...],
        session_id: str,
        profile: HardwareTaskGraphProfile,
        stdout_queue: "queue.Queue[str | None]",
        stdout_chunks: list[str],
        stderr_chunks: list[str],
        stdout_lock: threading.Lock,
        stderr_lock: threading.Lock,
        startup_metadata: JsonDict,
    ) -> None:
        self._process = process
        self._session_root = session_root.resolve()
        self._command = command
        self._session_id = session_id
        self._profile = profile
        self._stdout_queue = stdout_queue
        self._stdout_chunks = stdout_chunks
        self._stderr_chunks = stderr_chunks
        self._stdout_lock = stdout_lock
        self._stderr_lock = stderr_lock
        self._startup_metadata = dict(startup_metadata)
        self._request_sequence = 0
        self._closed = False
        self._close_result: HardwareSessionClose | None = None

    @property
    def process(self) -> subprocess.Popen[str]:
        return self._process

    @property
    def command(self) -> tuple[str, ...]:
        return self._command

    @property
    def startup_metadata(self) -> JsonDict:
        """Read-only copy of native allocation/load readiness metadata."""

        return dict(self._startup_metadata)

    def submit(
        self,
        tasks: Sequence[HardwareSessionTask],
        *,
        request_id: str | None = None,
        timeout_s: float | None = None,
    ) -> HardwareSessionExecution:
        if self._closed:
            raise HardwareInteractiveSessionError(
                "session_closed", "interactive session is closed"
            )
        if len(tasks) not in (1, 4):
            raise ValueError(
                "hardware_profile_violation: interactive requests require 1 or 4 tasks"
            )
        for task in tasks:
            self._assert_contained(task.args_path)
            self._assert_contained(task.left_path)
            self._assert_contained(task.right_path)
            self._assert_contained(task.output_path)

        sequence = self._request_sequence
        self._request_sequence += 1
        request_name = request_id or f"request-{sequence:04d}"
        request_dir = self._session_root
        request_dir.mkdir(parents=True, exist_ok=True)
        request_path = request_dir / f"{sequence:04d}_{_safe_name(request_name)}.json"
        response_path = (
            request_dir / f"{sequence:04d}_{_safe_name(request_name)}_response.json"
        )
        payload = {
            "schema_version": HARDWARE_INTERACTIVE_SESSION_SCHEMA_VERSION,
            "manifest_kind": HARDWARE_INTERACTIVE_REQUEST_KIND,
            "request_id": request_name,
            "requested_dpus": 1,
            "tasklets": 1,
            "tasks": [
                {
                    "task_id": task.task_id,
                    "args_path": _relative(request_dir, task.args_path),
                    "left_path": _relative(request_dir, task.left_path),
                    "right_path": _relative(request_dir, task.right_path),
                    "output_path": _relative(request_dir, task.output_path),
                }
                for task in tasks
            ],
        }
        try:
            request_path.write_text(
                json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False)
                + "\n",
                encoding="utf-8",
            )
            request_ref = _relative(self._session_root, request_path)
            response_ref = _relative(self._session_root, response_path)
            self._write_control(f"REQUEST {request_ref} {response_ref}\n")
            started = time.perf_counter()
            event = self._wait_for_event(timeout_s or self._profile.timeout_s)
            if event.get("event") != "response":
                raise HardwareInteractiveSessionError(
                    str(event.get("failure_stage") or "protocol_error"),
                    str(
                        event.get("error")
                        or "native interactive protocol returned an unexpected event"
                    ),
                    stdout=self._stdout_snippet(),
                    stderr=self._stderr_snippet(),
                )
            response = _load_response(response_path)
            if not _interactive_response_shape_valid(
                response, request_name, tasks, self._session_root
            ):
                raise HardwareInteractiveSessionError(
                    "response_manifest_failed",
                    "native interactive response was missing or invalid",
                    stdout=self._stdout_snippet(),
                    stderr=self._stderr_snippet(),
                )
            response_failure = response.get("failure_stage")
            status = (
                "completed"
                if response.get("status") == "completed" and response_failure is None
                else "failed"
            )
            if status == "failed":
                self._consume_terminal_close()
            return HardwareSessionExecution(
                status=status,
                failure_stage=str(response_failure) if response_failure else None,
                response_path=response_path,
                response=response,
                process_time_s=time.perf_counter() - started,
                command=self._command,
                stdout_snippet=self._stdout_snippet(),
                stderr_snippet=self._stderr_snippet(),
            )
        except subprocess.TimeoutExpired as exc:
            self._abort_process()
            raise HardwareInteractiveSessionError(
                "request_timeout",
                "timed out waiting for native interactive response",
                stdout=self._stdout_snippet(),
                stderr=self._stderr_snippet(),
            ) from exc
        except HardwareInteractiveSessionError:
            self._consume_terminal_close()
            raise
        except queue.Empty as exc:
            self._abort_process()
            raise HardwareInteractiveSessionError(
                "request_timeout",
                "timed out waiting for native interactive response",
                stdout=self._stdout_snippet(),
                stderr=self._stderr_snippet(),
            ) from exc
        except OSError as exc:
            raise HardwareInteractiveSessionError(
                "request_protocol_failed",
                str(exc),
                stdout=self._stdout_snippet(),
                stderr=self._stderr_snippet(),
            ) from exc

    def close(self, *, timeout_s: float | None = None) -> HardwareSessionClose:
        if self._close_result is not None:
            return self._close_result
        if self._closed:
            return self._make_close("release_unconfirmed", False, None)
        try:
            self._write_control("CLOSE\n")
            event = self._wait_for_event(timeout_s or self._profile.timeout_s)
            if event.get("event") != "closed":
                # A protocol mismatch cannot safely leave a physical allocation
                # alive.  Do not try another close command: terminate the host
                # and preserve the failed release state for the caller.
                self._abort_process()
                return self._make_close(
                    str(event.get("failure_stage") or "close_protocol_failed"),
                    False,
                    None,
                )
            confirmed = (
                event.get("status") == "closed" and event.get("released") is True
            )
            self._process.wait(timeout=timeout_s or self._profile.timeout_s)
            result = self._make_close(
                None
                if confirmed
                else str(event.get("failure_stage") or "release_unconfirmed"),
                confirmed,
                float(event["release_time_s"])
                if isinstance(event.get("release_time_s"), (int, float))
                else None,
            )
            self._closed = True
            self._close_result = result
            return result
        except HardwareInteractiveSessionError as exc:
            self._abort_process()
            return self._make_close(exc.failure_stage, False, None)
        except (queue.Empty, subprocess.TimeoutExpired):
            self._abort_process()
            return self._make_close("close_timeout", False, None)
        except (BrokenPipeError, OSError):
            self._abort_process()
            return self._make_close("close_protocol_failed", False, None)

    def __enter__(self) -> "HardwareInteractiveSession":
        return self

    def __exit__(self, _type: object, _value: object, _traceback: object) -> None:
        self.close()

    def _assert_contained(self, path: Path) -> None:
        try:
            path.resolve().relative_to(self._session_root)
        except ValueError as exc:
            raise ValueError(
                "hardware_profile_violation: session task paths must be inside the native session root"
            ) from exc

    def _write_control(self, line: str) -> None:
        if self._process.stdin is None:
            raise HardwareInteractiveSessionError(
                "request_protocol_failed", "native stdin is unavailable"
            )
        self._process.stdin.write(line)
        self._process.stdin.flush()

    def _wait_for_event(self, timeout_s: float) -> JsonDict:
        item = self._stdout_queue.get(timeout=timeout_s)
        if item is None:
            raise HardwareInteractiveSessionError(
                "process_eof",
                "native interactive host closed stdout",
                stdout=self._stdout_snippet(),
                stderr=self._stderr_snippet(),
            )
        try:
            payload = json.loads(item)
        except json.JSONDecodeError as exc:
            raise HardwareInteractiveSessionError(
                "protocol_error",
                "native interactive host emitted invalid JSON",
                stdout=self._stdout_snippet(),
                stderr=self._stderr_snippet(),
            ) from exc
        if not isinstance(payload, dict):
            raise HardwareInteractiveSessionError(
                "protocol_error", "native event was not an object"
            )
        if payload.get("event") == "error":
            raise HardwareInteractiveSessionError(
                str(payload.get("failure_stage") or "protocol_error"),
                str(payload.get("error") or "native interactive session failed"),
                stdout=self._stdout_snippet(),
                stderr=self._stderr_snippet(),
            )
        return payload

    def _consume_terminal_close(self) -> None:
        try:
            event = self._wait_for_event(1.0)
        except (HardwareInteractiveSessionError, queue.Empty):
            self._closed = True
            return
        if event.get("event") == "closed":
            self._closed = True
            self._close_result = self._make_close(
                None
                if event.get("released") is True
                else str(event.get("failure_stage") or "release_unconfirmed"),
                event.get("released") is True,
                float(event["release_time_s"])
                if isinstance(event.get("release_time_s"), (int, float))
                else None,
            )

    def _abort_process(self) -> None:
        self._closed = True
        if self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=1.0)

    def _stdout_snippet(self) -> str:
        with self._stdout_lock:
            return _snippet("".join(self._stdout_chunks))

    def _stderr_snippet(self) -> str:
        with self._stderr_lock:
            return _snippet("".join(self._stderr_chunks))

    def _make_close(
        self, stage: str | None, confirmed: bool, release_time_s: float | None
    ) -> HardwareSessionClose:
        result = HardwareSessionClose(
            status="closed" if confirmed else "failed",
            failure_stage=stage,
            release_confirmed=confirmed,
            release_time_s=release_time_s,
            process_returncode=self._process.poll(),
            stdout_snippet=self._stdout_snippet(),
            stderr_snippet=self._stderr_snippet(),
        )
        # The physical study never retries a release automatically. Cache both
        # confirmed and failed outcomes so later callers observe one stable
        # release verdict rather than sending another control command.
        self._close_result = result
        return result


def build_hardware_session(
    root_dir: Path,
    session_root: Path,
    *,
    profile: HardwareTaskGraphProfile,
    environment: Mapping[str, str],
) -> HardwareSessionBuild:
    """Build the bounded native source once for a hardware TaskGraph run."""

    sdk = discover_upmem_sdk(env=environment)
    tools_by_name = {tool.name: tool for tool in sdk.tools}
    make_path = shutil.which("make", path=environment.get("PATH"))
    required_names = ("make", "dpu-upmem-dpurte-clang", "dpu-pkg-config")
    missing = [
        name for name in required_names
        if (name == "make" and not make_path)
        or (name != "make" and not tools_by_name.get(name))
        or (name != "make" and not tools_by_name[name].available)
    ]
    if missing:
        raise RuntimeError(
            "sdk_discovery_failed: missing required UPMEM SDK tools: "
            + ", ".join(sorted(missing))
        )
    required = {name: tools_by_name[name] for name in required_names if name != "make"}

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


def start_hardware_session(
    build: HardwareSessionBuild,
    *,
    session_id: str,
    profile: HardwareTaskGraphProfile,
    environment: Mapping[str, str],
    readiness_timeout_s: float | None = None,
) -> HardwareInteractiveSession:
    """Start one native host that owns one DPU until ``close`` is confirmed."""

    if not session_id:
        raise ValueError(
            "hardware_profile_violation: interactive session_id must be non-empty"
        )
    if profile.requested_dpu_count != 1 or profile.tasklets_per_dpu < 1 or profile.tasklets_per_dpu > 16:
        raise ValueError(
            "hardware_profile_violation: interactive session requires one DPU and 1..16 tasklets"
        )
    root = build.session_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    _relative(root, build.dpu_binary)
    bootstrap_path = root / f"{_safe_name(session_id)}_interactive_bootstrap.json"
    bootstrap_path.write_text(
        json.dumps(
            {
                "schema_version": HARDWARE_INTERACTIVE_SESSION_SCHEMA_VERSION,
                "manifest_kind": HARDWARE_INTERACTIVE_BOOTSTRAP_KIND,
                "session_id": session_id,
                "dpu_binary": _relative(root, build.dpu_binary),
                "requested_dpus": 1,
                "tasklets": profile.tasklets_per_dpu,
            },
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    command = (
        str(build.host_binary),
        "--interactive-session",
        "--bootstrap-manifest",
        str(bootstrap_path),
    )
    try:
        process = subprocess.Popen(
            command,
            cwd=build.build_dir,
            env=_sanitised_hardware_env(environment),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
    except OSError as exc:
        raise HardwareInteractiveSessionError("startup_failed", str(exc)) from exc

    stdout_queue: "queue.Queue[str | None]" = queue.Queue()
    stdout_chunks: list[str] = []
    stderr_chunks: list[str] = []
    stdout_lock = threading.Lock()
    stderr_lock = threading.Lock()

    def drain_stdout() -> None:
        stream = process.stdout
        if stream is not None:
            for line in iter(stream.readline, ""):
                with stdout_lock:
                    stdout_chunks.append(line)
                stdout_queue.put(line)
        stdout_queue.put(None)

    def drain_stderr() -> None:
        stream = process.stderr
        if stream is not None:
            for chunk in iter(lambda: stream.read(4096), ""):
                with stderr_lock:
                    stderr_chunks.append(chunk)

    threading.Thread(
        target=drain_stdout, name="upmem-session-stdout", daemon=True
    ).start()
    threading.Thread(
        target=drain_stderr, name="upmem-session-stderr", daemon=True
    ).start()
    session = HardwareInteractiveSession(
        process,
        session_root=root,
        command=command,
        session_id=session_id,
        profile=profile,
        stdout_queue=stdout_queue,
        stdout_chunks=stdout_chunks,
        stderr_chunks=stderr_chunks,
        stdout_lock=stdout_lock,
        stderr_lock=stderr_lock,
        startup_metadata={},
    )
    try:
        ready = session._wait_for_event(readiness_timeout_s or profile.timeout_s)
        if (
            ready.get("event") != "ready"
            or ready.get("status") != "ready"
            or ready.get("requested_dpus") != 1
            or ready.get("allocated_dpus") != 1
        ):
            raise HardwareInteractiveSessionError(
                str(ready.get("failure_stage") or "startup_protocol_failed"),
                "native interactive host did not report one-DPU readiness",
                stdout=session._stdout_snippet(),
                stderr=session._stderr_snippet(),
            )
        session._startup_metadata.update(
            {
                "requested_dpus": ready.get("requested_dpus"),
                "allocated_dpus": ready.get("allocated_dpus"),
                "allocation_time_s": ready.get("allocation_time_s"),
                "binary_load_time_s": ready.get("binary_load_time_s"),
            }
        )
    except (queue.Empty, HardwareInteractiveSessionError) as exc:
        session._abort_process()
        if isinstance(exc, HardwareInteractiveSessionError):
            raise
        raise HardwareInteractiveSessionError(
            "readiness_timeout",
            "timed out waiting for native interactive readiness",
            stdout=session._stdout_snippet(),
            stderr=session._stderr_snippet(),
        ) from exc
    return session


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


def _interactive_response_shape_valid(
    response: JsonDict,
    request_id: str,
    tasks: Sequence[HardwareSessionTask],
    session_root: Path,
) -> bool:
    if (
        response.get("schema_version") != HARDWARE_INTERACTIVE_SESSION_SCHEMA_VERSION
        or response.get("manifest_kind") != HARDWARE_INTERACTIVE_RESPONSE_KIND
        or response.get("request_id") != request_id
        or response.get("requested_dpus") != 1
        or response.get("allocated_dpus") != 1
    ):
        return False
    raw_tasks = response.get("tasks")
    if not isinstance(raw_tasks, list) or len(raw_tasks) != len(tasks):
        return False
    for item, task in zip(raw_tasks, tasks):
        if (
            not isinstance(item, dict)
            or item.get("task_id") != task.task_id
            or item.get("status") not in {"completed", "failed", "not_run"}
        ):
            return False
        if item.get("status") == "completed":
            output = item.get("output")
            if not isinstance(output, dict) or output.get("path") != _relative(
                session_root, task.output_path
            ):
                return False
    return True


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
    except OSError as exc:
        return {
            "returncode": None,
            "elapsed_s": time.perf_counter() - started,
            "timed_out": False,
            "stdout_snippet": "",
            "stderr_snippet": _snippet(exc),
            "error": str(exc),
        }


def _run_resident_command(
    command: Sequence[str], *, cwd: Path, env: Mapping[str, str], timeout_s: float
) -> JsonDict:
    """Run the resident host with a bounded graceful cleanup window."""

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
            stdout, stderr = process.communicate(timeout=float(timeout_s))
            return {
                "returncode": process.returncode,
                "elapsed_s": time.perf_counter() - started,
                "timed_out": False,
                "cleanup_attempted": False,
                "cleanup_confirmed": process.returncode == 0,
                "stdout_snippet": _snippet(stdout),
                "stderr_snippet": _snippet(stderr),
            }
        except subprocess.TimeoutExpired as exc:
            process.terminate()
            stdout = _snippet(exc.stdout)
            stderr = _snippet(exc.stderr)
            cleanup_confirmed = False
            try:
                out, err = process.communicate(timeout=RESIDENT_NATIVE_CLEANUP_GRACE_S)
                stdout += _snippet(out)
                stderr += _snippet(err)
                cleanup_confirmed = process.returncode is not None
            except subprocess.TimeoutExpired as grace_exc:
                stdout += _snippet(grace_exc.stdout)
                stderr += _snippet(grace_exc.stderr)
                process.send_signal(signal.SIGINT)
                try:
                    out, err = process.communicate(timeout=RESIDENT_NATIVE_CLEANUP_GRACE_S)
                    stdout += _snippet(out)
                    stderr += _snippet(err)
                    cleanup_confirmed = process.returncode is not None
                except subprocess.TimeoutExpired:
                    process.kill()
                    out, err = process.communicate()
                    stdout += _snippet(out)
                    stderr += _snippet(err)
            return {
                "returncode": process.returncode,
                "elapsed_s": time.perf_counter() - started,
                "timed_out": True,
                "cleanup_attempted": True,
                "cleanup_confirmed": cleanup_confirmed,
                "stdout_snippet": _snippet(stdout),
                "stderr_snippet": _snippet(stderr),
            }
    except OSError as exc:
        return {
            "returncode": None,
            "elapsed_s": time.perf_counter() - started,
            "timed_out": False,
            "cleanup_attempted": process is not None,
            "cleanup_confirmed": False,
            "stdout_snippet": "",
            "stderr_snippet": _snippet(exc),
            "error": str(exc),
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


# ---------------------------------------------------------------------------
# Additive MRAM-resident graph package protocol.
# The legacy generic-loop and generic_loop_interactive_session_v1 APIs above
# intentionally remain unchanged.  Resident callers use a sibling native
# source tree and these separate entry points.

RESIDENT_NATIVE_SOURCE_DIR = "upmem_sdk_generic_loop_resident"
RESIDENT_NATIVE_SCHEMA_VERSION = "generic_loop_resident_graph_session_v1"
RESIDENT_NATIVE_INPUT_KIND = "resident_graph_request"
RESIDENT_NATIVE_OUTPUT_KIND = "resident_graph_response"
RESIDENT_NATIVE_BUILD_TIMEOUT_S = 120.0
RESIDENT_NATIVE_CLEANUP_GRACE_S = 2.0


def _validate_resident_profile(profile: Any) -> None:
    try:
        actual = profile.to_json_dict()
    except AttributeError as exc:
        raise ValueError("hardware_profile_violation: resident profile is not serializable") from exc
    if actual.get("hardware_profile_version") not in {
        RESIDENT_PROFILE_VERSION,
        RESIDENT_M46_PROFILE_VERSION,
    }:
        raise ValueError("hardware_profile_violation: resident profile version is unsupported")
    if actual.get("requested_dpu_count") != 1:
        raise ValueError("hardware_profile_violation: resident profile must request one DPU")
    tasklets = actual.get("tasklets_per_dpu")
    if tasklets not in RESIDENT_SUPPORTED_TASKLETS:
        raise ValueError("hardware_profile_violation: resident profile tasklet count is unsupported")
    if actual.get("hardware_profile_version") == RESIDENT_PROFILE_VERSION and tasklets != 1:
        raise ValueError("hardware_profile_violation: resident v1 profile is one-tasklet only")
    expected_tile = RESIDENT_OUTPUT_TILE_ELEMENTS if actual["hardware_profile_version"] == RESIDENT_PROFILE_VERSION else RESIDENT_M46_OUTPUT_TILE_ELEMENTS
    if actual.get("output_tile_elements") != expected_tile:
        raise ValueError("hardware_profile_violation: resident output tile does not match profile")
    if actual.get("session_protocol") != RESIDENT_SESSION_PROTOCOL:
        raise ValueError("hardware_profile_violation: resident session protocol is unsupported")
    for key, value in {
        "target": "hardware",
        "backend_id": RESIDENT_BACKEND_ID,
        "route_id": RESIDENT_ROUTE_ID,
        "timing_scope": RESIDENT_TIMING_SCOPE,
        "max_rank": RESIDENT_MAX_RANK,
        "max_tensor_elements": RESIDENT_MAX_ELEMENTS,
        "max_logical_tasks": RESIDENT_MAX_LOGICAL_TASKS,
        "max_component_ops": RESIDENT_MAX_COMPONENT_OPS,
        "max_slot_descriptors": RESIDENT_MAX_SLOT_DESCRIPTORS,
        "mram_pool_bytes": RESIDENT_MRAM_POOL_BYTES,
        "max_contracted_combinations": RESIDENT_MAX_CONTRACTED_COMBINATIONS,
        "numeric_modes": list(RESIDENT_NUMERIC_MODES),
        "complex_policy": RESIDENT_COMPLEX_POLICY,
        "synchronous_execution": True,
        "performance_claim_applicable": False,
    }.items():
        if actual.get(key) != value:
            raise ValueError(f"hardware_profile_violation: resident profile {key} mismatch")
    timeout = actual.get("timeout_s")
    if not isinstance(timeout, (int, float)) or isinstance(timeout, bool) or not math.isfinite(float(timeout)) or float(timeout) <= 0:
        raise ValueError("hardware_profile_violation: resident profile timeout must be finite and positive")


@dataclass(frozen=True)
class ResidentGraphSessionExecution:
    status: str
    failure_stage: str | None
    response_path: Path
    response: JsonDict
    process_time_s: float
    command: tuple[str, ...]
    stdout_snippet: str
    stderr_snippet: str
    dpu_run_time_cycles: int = 0


def build_resident_hardware_session(
    root_dir: Path,
    session_root: Path,
    *,
    profile: Any,
    environment: Mapping[str, str],
) -> HardwareSessionBuild:
    """Build the separate resident host/DPU binary pair once per run.

    Building is a no-allocation preparation operation.  The physical guard is
    enforced by the execute entry point and the native host immediately before
    allocation, so a prepare-only build remains usable on a machine without a
    DPU.
    """
    _validate_resident_profile(profile)
    sdk = discover_upmem_sdk(env=environment)
    tools_by_name = {tool.name: tool for tool in sdk.tools}
    make_path = shutil.which("make", path=environment.get("PATH"))
    required_names = ("dpu-upmem-dpurte-clang", "dpu-pkg-config")
    missing = []
    if not make_path:
        missing.append("make")
    missing.extend(
        name
        for name in required_names
        if name not in tools_by_name or not tools_by_name[name].available
    )
    if missing:
        raise RuntimeError(
            "sdk_discovery_failed: missing required UPMEM SDK tools: "
            + ", ".join(missing)
        )
    required = {name: tools_by_name[name] for name in required_names}

    source = root_dir / "native" / "upmem" / "simplepim" / RESIDENT_NATIVE_SOURCE_DIR
    source_snapshot = session_root / "native" / "src"
    build_dir = session_root / "native" / "build"
    if not source.is_dir():
        raise RuntimeError("native_build_failed: resident native source tree is missing")
    _copy_source_tree(source, source_snapshot)
    _copy_source_tree(source, build_dir)
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
        f"NR_TASKLETS={int(profile.tasklets_per_dpu)}",
        f"PROFILE_VERSION={profile.version}",
        f"COMPLETION_VERSION={2 if profile.version == RESIDENT_M46_PROFILE_VERSION else 1}",
        "UPMEM_GENERIC_HARDWARE_MVP=1",
    )
    started = time.perf_counter()
    completed = _run_command(
        command,
        cwd=build_dir,
        env=_sanitised_hardware_env(environment),
        timeout_s=RESIDENT_NATIVE_BUILD_TIMEOUT_S,
    )
    build_time_s = time.perf_counter() - started
    if completed["returncode"] != 0:
        stage = "native_build_failed"
        if completed.get("timed_out"):
            stage = "native_build_timeout"
        detail = completed.get("stderr_snippet") or completed.get("error") or "native build command failed"
        raise RuntimeError(f"{stage}: {detail}")
    host_binary = build_dir / "bin" / "host"
    dpu_binary = build_dir / "bin" / "dpu_resident"
    if not host_binary.is_file() or not dpu_binary.is_file():
        raise RuntimeError(
            "native_build_failed: expected resident host and DPU binaries were not produced"
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
            "make": str(make_path),
            **{name: str(tool.path) if tool.path else None for name, tool in required.items()},
        },
    )


def execute_resident_graph_session(
    build: HardwareSessionBuild,
    *,
    manifest_path: Path,
    response_path: Path,
    profile: Any,
    environment: Mapping[str, str],
) -> ResidentGraphSessionExecution:
    """Run one complete graph request through the resident native host."""

    _validate_resident_profile(profile)
    if environment.get("UPMEM_ALLOW_PHYSICAL_HARDWARE") != "1":
        raise ValueError(
            "hardware_opt_in_missing: UPMEM_ALLOW_PHYSICAL_HARDWARE=1 is required"
        )
    if environment.get("DPU_BACKEND"):
        raise ValueError("hardware_profile_violation: DPU_BACKEND must be unset")
    root = build.session_root.resolve()
    try:
        manifest_path.resolve().relative_to(root)
        response_path.resolve().relative_to(root)
    except ValueError as exc:
        raise ValueError(
            "hardware_profile_violation: resident manifest paths must be inside session root"
        ) from exc
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("manifest_parse_failed: resident request manifest is invalid") from exc
    if not isinstance(manifest, dict):
        raise ValueError("manifest_parse_failed: resident request manifest is not an object")
    if manifest.get("schema_version") != RESIDENT_NATIVE_SCHEMA_VERSION:
        raise ValueError("hardware_profile_violation: resident request schema mismatch")
    if manifest.get("manifest_kind") != RESIDENT_NATIVE_INPUT_KIND:
        raise ValueError("hardware_profile_violation: resident request kind mismatch")
    if not isinstance(manifest.get("session_id"), str) or not manifest["session_id"] or not manifest["session_id"].isascii():
        raise ValueError("hardware_profile_violation: resident session identifiers must be non-empty ASCII")
    for key, expected in (
        ("route_id", "upmem_tn_hardware_taskgraph_resident"),
        ("backend_id", "upmem_sdk_hardware_taskgraph_resident"),
        ("hardware_profile_version", profile.version),
        ("target", "hardware"),
        ("sdk_allocation_profile", "backend=hw"),
        ("session_protocol", RESIDENT_NATIVE_SCHEMA_VERSION),
    ):
        if manifest.get(key) != expected:
            raise ValueError(f"hardware_profile_violation: resident request {key} mismatch")
    if manifest.get("graph_request_count") != 1:
        raise ValueError("hardware_profile_violation: resident graph request count must be one")
    if manifest.get("requested_dpus") != 1 or manifest.get("tasklets") not in RESIDENT_SUPPORTED_TASKLETS:
        raise ValueError("hardware_profile_violation: resident request requires one DPU and 1, 2, 4, 8, or 16 tasklets")
    if manifest.get("tasklets") != profile.tasklets_per_dpu:
        raise ValueError("hardware_profile_violation: resident request tasklet count does not match profile")
    if manifest.get("target") != "hardware" or manifest.get("sdk_allocation_profile") != RESIDENT_ALLOCATION_PROFILE:
        raise ValueError("hardware_profile_violation: resident request hardware allocation identity mismatch")
    package_ref = manifest.get("package_path")
    if not isinstance(package_ref, str) or not package_ref:
        raise ValueError("manifest_parse_failed: resident package path is missing")
    try:
        package_path = (manifest_path.parent / package_ref).resolve()
        package_path.relative_to(root)
    except ValueError as exc:
        raise ValueError("hardware_profile_violation: resident package path escapes session root") from exc
    if not package_path.is_file():
        raise ValueError("manifest_parse_failed: resident package file is missing")
    dpu_ref = manifest.get("dpu_binary")
    if not isinstance(dpu_ref, str) or not dpu_ref or not dpu_ref.isascii():
        raise ValueError("manifest_parse_failed: resident DPU binary path is invalid")
    try:
        dpu_path = (manifest_path.parent / dpu_ref).resolve()
        dpu_path.relative_to(root)
    except ValueError as exc:
        raise ValueError("hardware_profile_violation: resident DPU binary path escapes session root") from exc
    if not dpu_path.is_file() or dpu_path != build.dpu_binary.resolve():
        raise ValueError("hardware_profile_violation: resident DPU binary path does not match the built binary")
    from quantum_bench.targets.upmem.hardware_taskgraph_resident import (
        validate_resident_graph_package_file,
    )

    try:
        package_metadata = validate_resident_graph_package_file(
            package_path, profile=profile, operation_abi_version=1
        )
    except (OSError, ValueError) as exc:
        raise ValueError(f"hardware_profile_violation: resident package validation failed: {exc}") from exc
    _validate_resident_manifest(manifest, package_metadata, root, profile)
    if package_metadata.get("graph_request_count") != 1:
        raise ValueError("hardware_profile_violation: resident package graph request count must be one")
    if manifest.get("component_operation_count") != package_metadata.get("operation_count"):
        raise ValueError("hardware_profile_violation: resident manifest/package operation count mismatch")
    command = (
        str(build.host_binary),
        "--resident-package",
        str(manifest_path),
        "--resident-response",
        str(response_path),
    )
    completed = _run_resident_command(
        command,
        cwd=build.build_dir,
        env=_sanitised_hardware_env(environment),
        timeout_s=float(profile.timeout_s),
    )
    response = _load_response(response_path)
    failure_stage = response.get("failure_stage") if isinstance(response, dict) else None
    if completed.get("timed_out"):
        status = "failed"
        failure_stage = "hardware_session_timeout"
    elif not _resident_response_valid(response, manifest, profile):
        if not failure_stage:
            failure_stage = "response_manifest_failed"
        status = "failed"
    else:
        status = "completed" if completed.get("returncode") == 0 else "failed"
        if status == "failed" and not failure_stage:
            failure_stage = "hardware_session_timeout" if completed.get("timed_out") else "kernel_launch_failed"
    stderr = str(completed.get("stderr_snippet", ""))
    if completed.get("timed_out"):
        stderr += "\nphysical DPU release is unverified after resident host-process timeout; inspect allocation before rerunning"
    return ResidentGraphSessionExecution(
        status=status,
        failure_stage=str(failure_stage) if failure_stage else None,
        response_path=response_path,
        response=response,
        process_time_s=float(completed.get("elapsed_s", 0.0)),
        command=command,
        stdout_snippet=str(completed.get("stdout_snippet", "")),
        stderr_snippet=stderr,
        dpu_run_time_cycles=int(response.get("dpu_run_time_cycles", 0) if isinstance(response, dict) else 0),
    )


def _validate_resident_manifest(
    manifest: Mapping[str, Any],
    package_metadata: Mapping[str, Any],
    root: Path,
    profile: Any,
) -> None:
    def integer(value: Any, key: str, *, positive: bool = False) -> int:
        if not isinstance(value, int) or isinstance(value, bool) or (positive and value <= 0):
            raise ValueError(f"manifest_parse_failed: resident manifest integer field {key} is invalid")
        return value

    logical_tasks = integer(manifest.get("logical_task_count"), "logical_task_count", positive=True)
    if logical_tasks > profile.max_logical_tasks:
        raise ValueError("hardware_profile_violation: resident logical task count exceeds profile")
    operation_count = integer(manifest.get("component_operation_count"), "component_operation_count", positive=True)
    slot_count = integer(manifest.get("slot_descriptor_count"), "slot_descriptor_count", positive=True)
    if (
        manifest.get("package_magic") != "UPRGPCK1"
        or manifest.get("package_version") != 1
        or manifest.get("operation_abi_version") != 1
        or manifest.get("operation_bytes") != 784
        or manifest.get("dpu_binary_abi") != "dpu_resident"
        or package_metadata.get("package_magic") != "UPRGPCK1"
        or package_metadata.get("operation_abi_version") != 1
    ):
        raise ValueError("hardware_profile_violation: resident manifest ABI identity mismatch")
    if operation_count != package_metadata.get("operation_count") or slot_count != package_metadata.get("slot_count"):
        raise ValueError("hardware_profile_violation: resident manifest/package descriptor count mismatch")
    if manifest.get("mram_pool_bytes") != profile.mram_pool_bytes:
        raise ValueError("hardware_profile_violation: resident manifest MRAM pool mismatch")
    if manifest.get("quantization_mode") not in profile.numeric_modes:
        raise ValueError("hardware_profile_violation: resident manifest numeric mode mismatch")
    expected_mode = 0 if manifest.get("quantization_mode") == "none" else 1
    if package_metadata.get("operation_modes") != [expected_mode] * operation_count:
        raise ValueError("hardware_profile_violation: resident manifest mode does not match operation descriptors")
    if manifest.get("timing_scope") != RESIDENT_TIMING_SCOPE:
        raise ValueError("hardware_profile_violation: resident manifest timing scope mismatch")
    if manifest.get("no_host_intermediate_output_files") is not True or manifest.get("intermediate_output_paths") != []:
        raise ValueError("hardware_profile_violation: resident manifest permits host intermediate outputs")

    descriptors = {
        int(item["slot_id"]): item
        for item in package_metadata.get("slot_descriptors", ())
        if isinstance(item, Mapping) and isinstance(item.get("slot_id"), int)
    }
    initial_ids = {int(value) for value in package_metadata.get("initial_slot_ids", ())}
    final_ids = {int(value) for value in package_metadata.get("final_slot_ids", ())}

    def entry_ids(entries: Any, label: str) -> set[int]:
        if not isinstance(entries, list):
            raise ValueError(f"manifest_parse_failed: resident {label} entries are missing")
        result: set[int] = set()
        for entry in entries:
            if not isinstance(entry, Mapping) or not isinstance(entry.get("slot_id"), int) or isinstance(entry.get("slot_id"), bool):
                raise ValueError(f"manifest_parse_failed: resident {label} slot id is invalid")
            result.add(int(entry["slot_id"]))
        return result

    def validate_path(value: Any, key: str) -> None:
        if not isinstance(value, str) or not value or not value.isascii():
            raise ValueError(f"manifest_parse_failed: resident manifest {key} path is invalid")
        try:
            resolved = (root / value).resolve()
            resolved.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"hardware_profile_violation: resident manifest {key} path escapes session root") from exc

    def validate_bytes(entry: Mapping[str, Any], path_key: str) -> tuple[int, int, int]:
        slot_id = integer(entry.get("slot_id"), "slot_id")
        elements = integer(entry.get("elements"), "elements", positive=True)
        raw_bytes = integer(entry.get("raw_bytes"), "raw_bytes")
        transfer_bytes = integer(entry.get("transfer_bytes"), "transfer_bytes")
        if raw_bytes != elements * 4 or transfer_bytes != _align8(raw_bytes):
            raise ValueError("hardware_profile_violation: resident manifest byte fields are inconsistent")
        validate_path(entry.get(path_key), path_key)
        descriptor = descriptors.get(slot_id)
        if descriptor is None or elements > int(descriptor["element_count"]):
            raise ValueError("hardware_profile_violation: resident manifest entry exceeds its slot descriptor")
        return slot_id, raw_bytes, transfer_bytes

    initial_entries = manifest.get("initial_slots")
    initial_entry_ids = entry_ids(initial_entries, "initial slot")
    if len(initial_entries) != len(initial_ids) or initial_entry_ids != initial_ids:
        raise ValueError("hardware_profile_violation: resident initial slot entries do not match package flags")
    initial_transfer = 0
    for entry in initial_entries:
        if not isinstance(entry, Mapping) or entry.get("slot_id") not in initial_ids:
            raise ValueError("manifest_parse_failed: resident initial slot entry is invalid")
        slot_id, _raw, transfer = validate_bytes(entry, "input_path")
        if descriptors[slot_id].get("initial") is not True or descriptors[slot_id].get("final") is True:
            raise ValueError("hardware_profile_violation: resident initial slot flag binding is invalid")
        initial_transfer += transfer

    final_entries = manifest.get("final_outputs")
    final_entry_ids = entry_ids(final_entries, "final output")
    if len(final_entries) != len(final_ids) or final_entry_ids != final_ids:
        raise ValueError("hardware_profile_violation: resident final output entries do not match package flags")
    components: set[str] = set()
    final_transfer = 0
    for entry in final_entries:
        if not isinstance(entry, Mapping):
            raise ValueError("manifest_parse_failed: resident final output entry is invalid")
        component = entry.get("component")
        if component not in {"real", "imag"} or component in components:
            raise ValueError("hardware_profile_violation: resident final output component flags are invalid")
        components.add(component)
        slot_id, _raw, transfer = validate_bytes(entry, "output_path")
        if descriptors[slot_id].get("initial") is True or descriptors[slot_id].get("final") is not True:
            raise ValueError("hardware_profile_violation: resident final output flag binding is invalid")
        final_transfer += transfer
    expected_components = {"real"} if len(final_ids) == 1 else {"real", "imag"}
    if components != expected_components:
        raise ValueError("hardware_profile_violation: resident final output components are incomplete")

    expected_descriptor = _align8(slot_count * 16) + _align8(operation_count * RESIDENT_OPERATION_BYTES)
    expected_control = RESIDENT_DESCRIPTOR_CONTROL_BYTES + operation_count * RESIDENT_CONTROL_H2D_BYTES_PER_LAUNCH
    if manifest.get("initial_h2d_bytes") != initial_transfer:
        raise ValueError("hardware_profile_violation: resident initial transfer accounting mismatch")
    if manifest.get("descriptor_h2d_bytes") != expected_descriptor:
        raise ValueError("hardware_profile_violation: resident descriptor transfer accounting mismatch")
    if manifest.get("descriptor_control_bytes") != RESIDENT_DESCRIPTOR_CONTROL_BYTES:
        raise ValueError("hardware_profile_violation: resident descriptor control accounting mismatch")
    if manifest.get("control_h2d_bytes_per_launch") != RESIDENT_CONTROL_H2D_BYTES_PER_LAUNCH:
        raise ValueError("hardware_profile_violation: resident control transfer accounting mismatch")
    if manifest.get("final_d2h_bytes") != final_transfer or manifest.get("intermediate_h2d_bytes") != 0 or manifest.get("intermediate_d2h_bytes") != 0:
        raise ValueError("hardware_profile_violation: resident final transfer accounting mismatch")
    if manifest.get("control_h2d_bytes") not in (None, expected_control):
        raise ValueError("hardware_profile_violation: resident control transfer total mismatch")


def _resident_response_valid(response: JsonDict, manifest: Mapping[str, Any], profile: Any) -> bool:
    def exact_integer(value: Any, expected: int) -> bool:
        return isinstance(value, int) and not isinstance(value, bool) and value == expected

    def finite_time(value: Any) -> bool:
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)) and float(value) >= 0.0

    if response.get("schema_version") != RESIDENT_NATIVE_SCHEMA_VERSION:
        return False
    if response.get("manifest_kind") != RESIDENT_NATIVE_OUTPUT_KIND:
        return False
    if response.get("status") != "completed" or response.get("failure_stage") is not None:
        return False
    if response.get("route_id") != RESIDENT_ROUTE_ID or response.get("backend_id") != RESIDENT_BACKEND_ID:
        return False
    if response.get("hardware_profile_version") != profile.version or response.get("target_requested") != "hardware":
        return False
    if response.get("target_observed") != "hardware" or response.get("sdk_allocation_profile") != RESIDENT_ALLOCATION_PROFILE:
        return False
    if response.get("sdk_allocation_profile_verified") is not True or response.get("session_protocol") != RESIDENT_NATIVE_SCHEMA_VERSION:
        return False
    if response.get("quantization_mode") != manifest.get("quantization_mode"):
        return False
    if not exact_integer(response.get("requested_dpus"), 1) or not exact_integer(response.get("allocated_dpus"), 1):
        return False
    if not exact_integer(response.get("tasklets"), profile.tasklets_per_dpu) or not exact_integer(response.get("tasklets"), manifest.get("tasklets")) or not exact_integer(response.get("graph_request_count"), 1):
        return False
    operation_count = manifest.get("component_operation_count")
    if not isinstance(operation_count, int) or not exact_integer(response.get("native_launch_count"), operation_count) or not exact_integer(response.get("native_task_count"), operation_count):
        return False
    for key in (
        "hardware_allocation_verified", "hardware_execution", "native_execution", "native_hardware_backend",
        "hardware_backend_verified", "hardware_release_verified", "release_confirmed",
        "physical_dependency_chain_verified", "hardware_timing_available", "hardware_kernel_executed",
    ):
        if response.get(key) is not True:
            return False
    if response.get("simulator_kernel_executed") is not False or response.get("cpu_fallback_used") is not False:
        return False
    if response.get("persistent_session_reused") is not False or response.get("resident_slots_persist_for_graph") is not True:
        return False
    if response.get("final_output_only_d2h") is not True or response.get("physical_bus_bytes_available") is not False:
        return False
    if not exact_integer(response.get("allocation_count"), 1):
        return False
    expected_control = RESIDENT_DESCRIPTOR_CONTROL_BYTES + operation_count * RESIDENT_CONTROL_H2D_BYTES_PER_LAUNCH
    expected_values = {
        "initial_h2d_bytes": manifest.get("initial_h2d_bytes"),
        "descriptor_h2d_bytes": manifest.get("descriptor_h2d_bytes"),
        "control_h2d_bytes": expected_control,
        "final_d2h_bytes": manifest.get("final_d2h_bytes"),
        "intermediate_h2d_bytes": 0,
        "intermediate_d2h_bytes": 0,
    }
    for key, expected in expected_values.items():
        if not exact_integer(response.get(key), int(expected)):
            return False
    actual_h2d = expected_values["initial_h2d_bytes"] + expected_values["descriptor_h2d_bytes"] + expected_values["control_h2d_bytes"]
    actual_d2h = expected_values["final_d2h_bytes"]
    if not exact_integer(response.get("actual_h2d_bytes"), actual_h2d) or not exact_integer(response.get("actual_d2h_bytes"), actual_d2h) or not exact_integer(response.get("actual_transfer_bytes"), actual_h2d + actual_d2h):
        return False
    for key in (
        "package_parse_time_s", "allocation_time_s", "binary_load_time_s", "initial_h2d_time_s",
        "descriptor_h2d_time_s", "control_h2d_time_s", "kernel_time_s", "final_d2h_time_s",
        "output_write_time_s", "release_time_s", "steady_state_graph_execution_s",
    ):
        if not finite_time(response.get(key)):
            return False
    completion_version = response.get("completion_abi_version")
    expected_completion_version = 2 if profile.version == RESIDENT_M46_PROFILE_VERSION else 1
    if completion_version != expected_completion_version:
        return False
    if not isinstance(response.get("dpu_run_time_cycles"), int) or isinstance(response.get("dpu_run_time_cycles"), bool):
        return False
    if response.get("dpu_run_time_cycles", 0) < 0:
        return False
    if expected_completion_version >= 2:
        operation_cycles = response.get("dpu_operation_cycles")
        graph_cycle_sum = response.get("graph_cycle_sum")
        counters = response.get("tasklet_processed_elements")
        active_counts = response.get("active_tasklet_count")
        idle_counts = response.get("idle_tasklet_count")
        utilization = response.get("tasklet_utilization")
        imbalance = response.get("tasklet_work_imbalance")
        if (
            not isinstance(operation_cycles, list)
            or len(operation_cycles) != operation_count
            or not isinstance(counters, list)
            or len(counters) != operation_count
            or not isinstance(active_counts, list)
            or len(active_counts) != operation_count
            or not isinstance(idle_counts, list)
            or len(idle_counts) != operation_count
            or not isinstance(utilization, list)
            or len(utilization) != operation_count
            or not isinstance(imbalance, list)
            or len(imbalance) != operation_count
            or graph_cycle_sum != response.get("dpu_run_time_cycles")
            or sum(operation_cycles) != graph_cycle_sum
        ):
            return False
        for cycle, row, active, idle, used, ratio in zip(
            operation_cycles, counters, active_counts, idle_counts, utilization, imbalance
        ):
            if (
                not isinstance(cycle, int)
                or isinstance(cycle, bool)
                or cycle < 0
                or not isinstance(row, list)
                or len(row) != profile.tasklets_per_dpu
                or any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in row)
                or not isinstance(active, int)
                or isinstance(active, bool)
                or active < 0
                or active > profile.tasklets_per_dpu
                or not isinstance(idle, int)
                or isinstance(idle, bool)
                or idle != profile.tasklets_per_dpu - active
                or not finite_time(used)
                or float(used) > 1.0
                or not finite_time(ratio)
                or float(ratio) > 1.0
            ):
                return False
    final_outputs = response.get("final_outputs")
    expected_outputs = manifest.get("final_outputs")
    if not isinstance(final_outputs, list) or not isinstance(expected_outputs, list):
        return False
    if len(final_outputs) != len(expected_outputs):
        return False
    return all(
        isinstance(actual, dict)
        and actual.get("status") == "completed"
        and actual.get("component") == expected.get("component")
        and actual.get("slot_id") == expected.get("slot_id")
        and actual.get("output_path") == expected.get("output_path")
        and actual.get("elements") == expected.get("elements")
        and actual.get("raw_bytes") == expected.get("raw_bytes")
        and actual.get("transfer_bytes") == expected.get("transfer_bytes")
        for actual, expected in zip(final_outputs, expected_outputs)
    )


# Names used by isolated resident tests and downstream callers.
build_hardware_session_resident = build_resident_hardware_session
execute_hardware_resident_graph_session = execute_resident_graph_session
