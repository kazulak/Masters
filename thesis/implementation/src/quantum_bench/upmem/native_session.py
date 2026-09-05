"""One persistent native-process lifecycle for packed UPMEM execution."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
from pathlib import PurePosixPath
import queue
import re
import stat
import subprocess
import threading
import time
from typing import Any, Callable, Mapping, Sequence

from quantum_bench.upmem.protocol import (
    COMPLETION_BYTES,
    CONTROL_BYTES,
    EXECUTION_TARGET_PHYSICAL,
    EXECUTION_TARGET_SIMULATOR,
    FLAG_ZERO_WORK,
    native_execution_identity,
    REQUEST_TRANSPORT_PACKED_OPERATION,
    REQUEST_TRANSPORT_PACKED_WAVE,
    STATUS_COMPLETED,
    V4Error,
    V4Profile,
    V4ProtocolError,
    V4RequestArtifact,
)
from quantum_bench.upmem.packed_operation import PackedOperation
from quantum_bench.upmem.packed_wave import (
    MAX_ENVELOPE_BYTES, WaveOperation, WaveTile, pack_wave_envelope,
)
from quantum_bench.upmem.wave_protocol import COMPLETION, CONTROL
from quantum_bench.upmem.wave_result import decode_wave_results


_RANK_PATH = re.compile(r"^/dev/dpu_rank[0-9]+$")
_HEX_SHA256 = re.compile(r"^[0-9a-fA-F]{64}$")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_relative(value: str | Path) -> str:
    raw = str(value)
    if not raw or "\\" in raw:
        raise ValueError(f"unsafe v4 relative path: {value!s}")
    parts = raw.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError(f"unsafe v4 relative path: {value!s}")
    parsed = PurePosixPath(raw)
    if parsed.is_absolute():
        raise ValueError(f"unsafe v4 relative path: {value!s}")
    return parsed.as_posix()


def _relative_to(root: Path, path: Path, *, must_exist: bool = False) -> str:
    candidate = path.resolve(strict=must_exist)
    try:
        relative = candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"path is outside v4 session root: {path}") from exc
    return _safe_relative(relative.as_posix())


class _OutputPump:
    def __init__(
        self,
        stream: Any,
        *,
        line_limit: int,
        retained_limit: int,
        output_queue: "queue.Queue[tuple[str, str | None]] | None",
    ) -> None:
        self.stream = stream
        self.line_limit = line_limit
        self.retained_limit = retained_limit
        self.output_queue = output_queue
        self.chunks: list[str] = []
        self.size = 0
        self.total_size = 0
        self.retained_truncated = False
        self.overflowed = False
        self.queue_overflowed = False
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self.thread.start()

    def _run(self) -> None:
        try:
            while True:
                line = self.stream.readline(self.line_limit + 1)
                if line in ("", b""):
                    self._emit("eof", None)
                    return
                text = (
                    line.decode(errors="replace")
                    if isinstance(line, bytes)
                    else str(line)
                )
                encoded = len(text.encode("utf-8", errors="replace"))
                if encoded > self.line_limit:
                    self.overflowed = True
                    if self.output_queue is not None:
                        self._emit("overflow", None)
                        return
                    self._retain(text, encoded)
                    continue
                self._retain(text, encoded)
                if not self._emit("line", text):
                    return
        except BaseException:
            self._emit("eof", None)

    def _emit(self, kind: str, value: str | None) -> bool:
        if self.output_queue is None:
            return True
        try:
            self.output_queue.put_nowait((kind, value))
        except queue.Full:
            self.overflowed = True
            self.queue_overflowed = True
            return False
        return True

    def _retain(self, text: str, encoded: int) -> None:
        self.total_size += encoded
        self.size += encoded
        self.chunks.append(text)
        if self.size <= self.retained_limit:
            return
        self.retained_truncated = True
        while len(self.chunks) > 1:
            removed = self.chunks.pop(0)
            self.size -= len(removed.encode("utf-8", errors="replace"))
            if self.size <= self.retained_limit:
                return
        tail = self.chunks[0].encode("utf-8", errors="replace")[-self.retained_limit :]
        self.chunks[0] = tail.decode("utf-8", errors="ignore")
        self.size = len(self.chunks[0].encode("utf-8", errors="replace"))


@dataclass(frozen=True)
class V4Release:
    event: Mapping[str, Any]
    release_confirmed: bool
    stdout: str
    stderr: str
    stdout_truncated: bool = False
    stderr_truncated: bool = False
    stdout_total_bytes: int = 0
    stderr_total_bytes: int = 0
    stdout_limit_exceeded: bool = False
    stderr_limit_exceeded: bool = False


class V4Session:
    """Persistent one-rank client, with an explicit experimental wave profile."""

    def __init__(
        self,
        process: Any,
        *,
        session_root: Path,
        profile: V4Profile,
        command: tuple[str, ...],
        stdout_pump: _OutputPump,
        stderr_pump: _OutputPump,
        events: "queue.Queue[tuple[str, str | None]]",
    ) -> None:
        self.process = process
        self.session_root = session_root.resolve()
        self.profile = profile
        self.command = command
        self._stdout_pump = stdout_pump
        self._stderr_pump = stderr_pump
        self._events = events
        self._last_sequence = -1
        self._last_wave_sequence = -1
        self._wave_plan_sha256: bytes | None = None
        self._closed = False
        self._poisoned = False
        self._release: V4Release | None = None
        self._operation_lock = threading.Lock()
        self.startup: Mapping[str, Any] = {}

    def _enter_operation(self) -> None:
        if not self._operation_lock.acquire(blocking=False):
            raise V4Error(
                "session_busy", "v4 session already has an active operation"
            )

    def _leave_operation(self) -> None:
        self._operation_lock.release()

    @classmethod
    def start(
        cls,
        command: Sequence[str | Path],
        *,
        session_root: Path,
        profile: V4Profile,
        environment: Mapping[str, str] | None = None,
        popen_factory: Callable[..., Any] = subprocess.Popen,
    ) -> "V4Session":
        root = Path(session_root).resolve()
        if not root.is_dir():
            raise V4Error(
                "hardware_profile_violation",
                "session_root must be an existing directory",
            )
        env = dict(os.environ if environment is None else environment)
        if profile.execution_target == EXECUTION_TARGET_PHYSICAL:
            if profile.rank_path is None or not _RANK_PATH.fullmatch(profile.rank_path):
                raise V4Error(
                    "hardware_profile_violation",
                    "an explicit /dev/dpu_rankN is required",
                )
            if env.get("UPMEM_ALLOW_PHYSICAL_HARDWARE") != "1":
                raise V4Error(
                    "hardware_opt_in_missing",
                    "UPMEM_ALLOW_PHYSICAL_HARDWARE=1 is required",
                )
            forbidden = next(
                (
                    name
                    for name in ("DPU_BACKEND", "UPMEM_EXECUTION_MODE")
                    if name in env
                ),
                None,
            )
            if forbidden is not None:
                raise V4Error(
                    "hardware_profile_violation",
                    f"{forbidden} must be unset for physical v4",
                )
        elif profile.execution_target == EXECUTION_TARGET_SIMULATOR:
            if profile.rank_path is not None:
                raise V4Error(
                    "simulator_profile_violation",
                    "v4 simulator sessions forbid rank_path",
                )
            backend = env.get("DPU_BACKEND")
            if backend not in {None, "simulator"}:
                raise V4Error(
                    "simulator_profile_violation",
                    "DPU_BACKEND must be unset or simulator for v4 simulator execution",
                )
            if "UPMEM_EXECUTION_MODE" in env:
                raise V4Error(
                    "simulator_profile_violation",
                    "UPMEM_EXECUTION_MODE must be unset for v4 simulator execution",
                )
            env["DPU_BACKEND"] = "simulator"
        else:  # pragma: no cover - V4Profile validates this contract.
            raise V4Error("simulator_profile_violation", "unsupported v4 target")
        argv = tuple(str(value) for value in command)
        if not argv:
            raise V4Error("hardware_profile_violation", "v4 command cannot be empty")
        if profile.request_transport == REQUEST_TRANSPORT_PACKED_WAVE:
            if "--wave-v5" not in argv:
                argv += ("--wave-v5",)
        elif "--wave-v5" in argv:
            raise V4Error("hardware_profile_violation", "wave command requires wave profile")
        try:
            process = popen_factory(
                argv,
                cwd=str(root),
                env=env,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
        except OSError as exc:
            raise V4Error("sdk_discovery_failed", str(exc)) from exc
        # A failed wave may emit RESPONSE, RELEASE and EOF without another command.
        event_limit = 3 if profile.request_transport == REQUEST_TRANSPORT_PACKED_WAVE else 2
        events: "queue.Queue[tuple[str, str | None]]" = queue.Queue(maxsize=event_limit)
        stdout_pump = _OutputPump(
            process.stdout,
            line_limit=profile.max_stdout_bytes,
            retained_limit=profile.max_retained_output_bytes,
            output_queue=events,
        )
        stderr_pump = _OutputPump(
            process.stderr,
            line_limit=profile.max_stderr_bytes,
            retained_limit=profile.max_retained_output_bytes,
            output_queue=None,
        )
        stdout_pump.start()
        stderr_pump.start()
        session = cls(
            process,
            session_root=root,
            profile=profile,
            command=argv,
            stdout_pump=stdout_pump,
            stderr_pump=stderr_pump,
            events=events,
        )
        event = session._next_event(profile.timeout_s)
        if event.get("event") != "READY" or event.get("status") != "ready":
            session._poisoned = True
            session.close()
            raise V4ProtocolError(
                str(event.get("failure_stage") or "hardware_allocation_failed"),
                str(event.get("error") or "v4 native process did not become ready"),
            )
        try:
            session._validate_ready(event)
        except BaseException:
            # READY has already crossed the native allocation boundary.  A
            # malformed identity or allocation record must not leak the
            # process or the SDK allocation while preserving the original
            # validation failure for the caller.
            session._poisoned = True
            try:
                session.close()
            except BaseException:
                session._terminate()
            raise
        session.startup = dict(event)
        return session

    def _stdout_text(self) -> str:
        return "".join(self._stdout_pump.chunks)

    def _stderr_text(self) -> str:
        return "".join(self._stderr_pump.chunks)

    def _release_record(
        self, event: Mapping[str, Any], release_confirmed: bool
    ) -> V4Release:
        return V4Release(
            event,
            release_confirmed,
            self._stdout_text(),
            self._stderr_text(),
            stdout_truncated=self._stdout_pump.retained_truncated,
            stderr_truncated=self._stderr_pump.retained_truncated,
            stdout_total_bytes=self._stdout_pump.total_size,
            stderr_total_bytes=self._stderr_pump.total_size,
            stdout_limit_exceeded=self._stdout_pump.overflowed,
            stderr_limit_exceeded=self._stderr_pump.overflowed,
        )

    def _next_event(self, timeout_s: float) -> Mapping[str, Any]:
        deadline = time.monotonic() + timeout_s
        while True:
            if self._stdout_pump.queue_overflowed:
                self._poisoned = True
                self._terminate()
                raise V4ProtocolError(
                    "protocol_output_limit", "v4 stdout event queue exceeded its limit"
                )
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self._poisoned = True
                self._terminate()
                raise V4ProtocolError(
                    "kernel_timeout", "timed out waiting for v4 native event"
                )
            try:
                kind, value = self._events.get(timeout=remaining)
            except queue.Empty as exc:
                self._poisoned = True
                self._terminate()
                raise V4ProtocolError(
                    "kernel_timeout", "timed out waiting for v4 native event"
                ) from exc
            if self._stdout_pump.queue_overflowed:
                self._poisoned = True
                self._terminate()
                raise V4ProtocolError(
                    "protocol_output_limit", "v4 stdout event queue exceeded its limit"
                )
            if kind == "overflow":
                self._poisoned = True
                self._terminate()
                raise V4ProtocolError(
                    "protocol_output_limit", "v4 stdout exceeded configured limit"
                )
            if kind == "eof":
                self._poisoned = True
                raise V4ProtocolError(
                    "kernel_launch_failed", "v4 native stdout closed before an event"
                )
            assert value is not None
            try:
                event = json.loads(value)
            except json.JSONDecodeError as exc:
                self._poisoned = True
                self._terminate()
                raise V4ProtocolError(
                    "protocol_error", "v4 native emitted malformed JSON"
                ) from exc
            if not isinstance(event, Mapping):
                self._poisoned = True
                self._terminate()
                raise V4ProtocolError(
                    "protocol_error", "v4 native event is not an object"
                )
            return event

    def _validate_ready(self, event: Mapping[str, Any]) -> None:
        simulator = self.profile.execution_target == EXECUTION_TARGET_SIMULATOR
        required = {
            "target_requested": "simulator" if simulator else "hardware",
            "target_observed": "sdk_simulator" if simulator else "physical_hardware",
            "requested_dpu_count": self.profile.dpu_count,
            "allocated_dpu_count": self.profile.dpu_count,
            "tasklets_per_dpu": self.profile.tasklets_per_dpu,
            "allocation_verified": True,
            "hardware_allocation_verified": not simulator,
            "simulator_kernel_executed": False,
            "cpu_fallback_used": False,
            "rank_path": None if simulator else self.profile.rank_path,
        }
        for field, expected in required.items():
            if event.get(field) != expected:
                raise V4ProtocolError(
                    "hardware_allocation_failed",
                    f"READY field {field!r} is not verified",
                )
        observed_transport = event.get("request_transport")
        if observed_transport != self.profile.request_transport:
            raise V4ProtocolError(
                "hardware_allocation_failed",
                "READY request transport does not match the selected profile",
            )
        self._validate_native_identity(event, event_name="READY")
        if self.profile.request_transport == REQUEST_TRANSPORT_PACKED_WAVE:
            for field in ("dpu_binary_sha256", "initialization_binary_sha256"):
                value = event.get(field)
                if not isinstance(value, str) or not _HEX_SHA256.fullmatch(value):
                    raise V4ProtocolError("protocol_error", f"READY {field} is invalid")

    def _validate_native_identity(
        self, event: Mapping[str, Any], *, event_name: str
    ) -> None:
        """Require the identity compiled into the native host protocol."""

        for field, expected in native_execution_identity(
            self.profile.execution_target, self.profile.request_transport
        ).items():
            if event.get(field) != expected:
                raise V4ProtocolError(
                    "protocol_error",
                    f"{event_name} native identity field {field!r} is not verified",
                )

    def _write(self, text: str) -> None:
        if self._closed or self._poisoned:
            raise V4Error("session_closed", "v4 session is closed or poisoned")
        self._send_raw(text)

    def _send_raw(self, text: str) -> None:
        if self.process.stdin is None:
            raise V4ProtocolError(
                "kernel_launch_failed", "v4 native stdin is unavailable"
            )
        try:
            self.process.stdin.write(text)
            self.process.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            self._poisoned = True
            self._terminate()
            raise V4ProtocolError(
                "kernel_launch_failed", "v4 native stdin failed"
            ) from exc

    def submit(
        self, artifact: V4RequestArtifact, *, timeout_s: float | None = None
    ) -> Mapping[str, Any]:
        self._enter_operation()
        try:
            return self._submit_unlocked(artifact, timeout_s=timeout_s)
        finally:
            self._leave_operation()

    def _submit_unlocked(
        self, artifact: V4RequestArtifact, *, timeout_s: float | None = None
    ) -> Mapping[str, Any]:
        if self._closed or self._poisoned:
            raise V4Error("session_closed", "v4 session is closed or poisoned")
        if self.profile.request_transport == REQUEST_TRANSPORT_PACKED_WAVE:
            raise V4Error("hardware_profile_violation", "wave sessions require submit_waves")
        if artifact.root.resolve() != self.session_root:
            raise V4Error(
                "request_manifest_failed",
                "request artifact root differs from session root",
            )
        sequence = artifact.request_sequence
        if sequence <= self._last_sequence:
            raise V4Error(
                "hardware_profile_violation", "v4 request sequence must increase"
            )
        submit_started = time.perf_counter()
        artifact_validation_started = time.perf_counter()
        manifest_rel = _relative_to(
            self.session_root, artifact.manifest_path, must_exist=True
        )
        actual_manifest_hash = _file_sha256(artifact.manifest_path)
        if actual_manifest_hash != artifact.manifest_sha256:
            self._poisoned = True
            self._terminate()
            raise V4ProtocolError(
                "request_manifest_failed", "request manifest hash mismatch"
            )
        if _file_sha256(artifact.sidecar_path) != artifact.sidecar_sha256:
            self._poisoned = True
            self._terminate()
            raise V4ProtocolError(
                "sidecar_validation_failed", "request sidecar hash mismatch"
            )
        self._validate_artifact_payloads(artifact)
        artifact_validation_s = time.perf_counter() - artifact_validation_started
        self._last_sequence = sequence
        protocol_write_started = time.perf_counter()
        self._write(f"SUBMIT {manifest_rel} {artifact.manifest_sha256}\n")
        protocol_write_s = time.perf_counter() - protocol_write_started
        response_wait_started = time.perf_counter()
        event = self._next_event(timeout_s or self.profile.timeout_s)
        response_wait_s = time.perf_counter() - response_wait_started
        if event.get("event") != "RESPONSE":
            self._poisoned = True
            self._close_unlocked()
            raise V4ProtocolError(
                "protocol_error", "v4 response event has the wrong type"
            )
        response_validation_started = time.perf_counter()
        try:
            self._validate_response(event, artifact)
        except V4Error:
            self._poisoned = True
            self._close_unlocked()
            raise
        response_validation_s = time.perf_counter() - response_validation_started
        response = dict(event)
        response["host_submit_timing"] = {
            "artifact_validation_s": float(artifact_validation_s),
            "protocol_write_s": float(protocol_write_s),
            "response_wait_s": float(response_wait_s),
            "response_validation_s": float(response_validation_s),
            "total_submit_s": float(time.perf_counter() - submit_started),
        }
        return response

    def submit_packed(
        self, operation: PackedOperation, *, timeout_s: float | None = None
    ) -> Mapping[str, Any]:
        self._enter_operation()
        try:
            return self._submit_packed_unlocked(operation, timeout_s=timeout_s)
        finally:
            self._leave_operation()

    def _submit_packed_unlocked(
        self, operation: PackedOperation, *, timeout_s: float | None = None
    ) -> Mapping[str, Any]:
        """Submit one variable-length operation envelope to the native host.

        The native host returns a small operation event and a JSONL response
        file containing the unchanged per-request ABI-v4 response records.
        Keeping those records intact lets the existing response validator and
        runtime accounting remain the single correctness boundary.
        """

        if self._closed or self._poisoned:
            raise V4Error("session_closed", "v4 session is closed or poisoned")
        if not isinstance(operation, PackedOperation):
            raise TypeError("packed submission requires a PackedOperation")
        if operation.root.resolve() != self.session_root:
            raise V4Error(
                "request_manifest_failed",
                "packed operation root differs from session root",
            )
        if self.profile.request_transport != REQUEST_TRANSPORT_PACKED_OPERATION:
            raise V4Error(
                "hardware_profile_violation",
                "packed submission requires packed_operation_v1 profile",
            )
        sequences = tuple(request.request_sequence for request in operation.requests)
        if not sequences or any(
            right <= left for left, right in zip(sequences, sequences[1:])
        ):
            raise V4Error(
                "hardware_profile_violation",
                "packed request sequences must increase",
            )
        if sequences[0] <= self._last_sequence:
            raise V4Error(
                "hardware_profile_violation", "v4 request sequence must increase"
            )
        operation_path = _relative_to(
            self.session_root, operation.path, must_exist=True
        )
        actual_hash = _file_sha256(operation.path)
        if actual_hash != operation.sha256:
            self._poisoned = True
            self._terminate()
            raise V4ProtocolError(
                "request_manifest_failed", "packed operation hash mismatch"
            )
        submit_started = time.perf_counter()
        protocol_write_started = time.perf_counter()
        self._write(
            f"SUBMIT_PACKED_OPERATION {operation_path} {operation.sha256}\n"
        )
        protocol_write_s = time.perf_counter() - protocol_write_started
        response_wait_started = time.perf_counter()
        event = self._next_event(timeout_s or self.profile.timeout_s)
        response_wait_s = time.perf_counter() - response_wait_started
        if event.get("event") != "OPERATION_RESPONSE":
            self._poisoned = True
            self._close_unlocked()
            raise V4ProtocolError(
                "protocol_error", "packed response event has the wrong type"
            )
        responses: list[Mapping[str, Any]] = []
        try:
            response_count, completed_request_count, failed_request_index = (
                self._validate_packed_operation_metadata(
                    event,
                    expected_count=len(operation.requests),
                    expected_operation_sequence=operation.operation_sequence,
                )
            )
            responses = self._read_packed_operation_responses(
                event, response_count=response_count
            )
            if event.get("status") == "completed":
                response_validation_started = time.perf_counter()
                for response, request in zip(
                    responses, operation.requests, strict=True
                ):
                    self._validate_response(response, request)
                response_validation_s = time.perf_counter() - response_validation_started
            else:
                response_validation_started = time.perf_counter()
                if failed_request_index is not None:
                    for response, request in zip(
                        responses[:completed_request_count],
                        operation.requests[:completed_request_count],
                        strict=True,
                    ):
                        self._validate_response(response, request)
                    self._validate_partial_response(
                        responses[completed_request_count],
                        operation.requests[failed_request_index],
                    )
                else:
                    for response, request in zip(
                        responses, operation.requests
                    ):
                        self._validate_response(response, request)
                response_validation_s = time.perf_counter() - response_validation_started
        except V4Error as exc:
            self._raise_packed_operation_failure(
                event, exc, responses=responses
            )
        # Operation events use the protocol's JSON status string.  The
        # numeric STATUS_COMPLETED constant is reserved for DPU results.
        if event.get("status") != "completed":
            failure = V4ProtocolError(
                str(event.get("failure_stage") or "request_execution_failed"),
                str(event.get("error") or "packed operation failed"),
            )
            self._raise_packed_operation_failure(event, failure, responses=responses)
        response_validation_s = time.perf_counter() - response_validation_started
        self._last_sequence = sequences[-1]
        result = dict(event)
        result["responses"] = tuple(dict(response) for response in responses)
        result["host_submit_timing"] = {
            "artifact_validation_s": 0.0,
            "protocol_write_s": float(protocol_write_s),
            "response_wait_s": float(response_wait_s),
            "response_validation_s": float(response_validation_s),
            "total_submit_s": float(time.perf_counter() - submit_started),
        }
        return result

    def submit_waves(
        self, *, plan_sha256: bytes, dpu_binary_sha256: bytes, sequence: int,
        operations: tuple[WaveOperation, ...], waves: tuple[tuple[WaveTile, ...], ...],
        timeout_s: float | None = None,
    ) -> Mapping[str, Any]:
        """Execute one admitted cohort using the existing single-owner process.

        Input/result files remain session-owned until the caller cleans them up.
        Results expose views of one immutable snapshot, not separate plane copies.
        The whole-DAG caller must bind these tables to its exact physical plan.
        The timeout is cooperative between bounded packing/I/O/decoding phases;
        process termination can add the existing two-second teardown grace.
        """
        self._enter_operation()
        try:
            return self._submit_waves_unlocked(
                plan_sha256=plan_sha256, dpu_binary_sha256=dpu_binary_sha256,
                sequence=sequence, operations=operations, waves=waves,
                timeout_s=timeout_s,
            )
        finally:
            self._leave_operation()

    def _submit_waves_unlocked(
        self, *, plan_sha256: bytes, dpu_binary_sha256: bytes, sequence: int,
        operations: tuple[WaveOperation, ...], waves: tuple[tuple[WaveTile, ...], ...],
        timeout_s: float | None,
    ) -> Mapping[str, Any]:
        if self._closed or self._poisoned:
            raise V4Error("session_closed", "native session is closed or poisoned")
        if self.profile.request_transport != REQUEST_TRANSPORT_PACKED_WAVE:
            raise V4Error("hardware_profile_violation", "submit_waves requires packed_wave_v1")
        timeout = self.profile.timeout_s if timeout_s is None else timeout_s
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("wave timeout must be finite and positive")
        started = time.perf_counter()
        deadline = time.monotonic() + timeout
        if type(plan_sha256) is not bytes or type(dpu_binary_sha256) is not bytes:
            raise ValueError("wave plan and binary digests must be immutable bytes")
        if dpu_binary_sha256.hex() != self.startup["dpu_binary_sha256"].lower():
            raise ValueError("wave DPU binary differs from verified startup")
        if self._wave_plan_sha256 is not None and plan_sha256 != self._wave_plan_sha256:
            raise ValueError("wave physical plan changed within the session")
        if type(sequence) is not int or sequence <= self._last_wave_sequence:
            raise ValueError("wave envelope sequence must increase")
        data = pack_wave_envelope(
            plan_sha256=plan_sha256, dpu_binary_sha256=dpu_binary_sha256,
            sequence=sequence, operations=operations, waves=waves,
            numeric_mode=self.profile.numeric_mode_code,
        )
        if (len(waves[0]) != self.profile.dpu_count
                or waves[0][0].control.tasklets != self.profile.tasklets_per_dpu):
            raise ValueError("wave resources differ from the opened session")
        if waves[0][0].control.request_sequence <= self._last_sequence:
            raise ValueError("wave request sequence must increase across submissions")
        controls = tuple(tile.control for wave in waves for tile in wave)
        output_bytes = sum(COMPLETION.size + sum(
            c.m * c.n * 4 for _, length in c.planes[4:] if length
        ) for c in controls)
        if output_bytes > MAX_ENVELOPE_BYTES:
            raise ValueError("wave result exceeds the 512-MiB snapshot admission limit")
        request_name = f"wave-operation-{sequence:020d}.bin"
        request_path = self.session_root / request_name
        request_hash = hashlib.sha256(data).hexdigest()
        fd = os.open(request_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
        envelope_bytes = len(data)
        del data
        preparation_s = time.perf_counter() - started
        event: Mapping[str, Any] = {}
        try:
            if time.monotonic() >= deadline:
                raise V4ProtocolError("kernel_timeout", "wave preparation exceeded deadline")
            write_started = time.perf_counter()
            self._write(f"SUBMIT_PACKED_WAVES {request_name} {request_hash}\n")
            write_s = time.perf_counter() - write_started
            wait_started = time.perf_counter()
            event = self._next_event(max(0.001, deadline - time.monotonic()))
            wait_s = time.perf_counter() - wait_started
            validation_started = time.perf_counter()
            if event.get("event") != "WAVE_RESPONSE":
                raise V4ProtocolError("protocol_error", "expected WAVE_RESPONSE")
            if event.get("status") != "completed":
                raise V4ProtocolError(str(event.get("failure_stage") or "request_execution_failed"),
                                      str(event.get("error") or "prepared cohort failed"))
            expected = {
                "sequence": sequence, "completed_wave_count": len(waves),
                "launch_count": len(waves), "completed_result_count": len(controls),
                "allocated_dpu_count": self.profile.dpu_count,
                "tasklets_per_dpu": self.profile.tasklets_per_dpu,
                "cpu_fallback_used": False, "target_observed": self.profile.execution_target,
                "envelope_bytes": envelope_bytes, "native_snapshot_bytes": envelope_bytes,
                "operation_count": len(operations), "control_count": len(controls),
                "h2d_bytes": sum(CONTROL.size + sum(n for _, n in c.planes[:4]) for c in controls),
                "d2h_bytes": sum(COMPLETION.size + sum(n for _, n in c.planes[4:]) for c in controls),
                "input_payload_bytes": sum(sum(n for _, n in c.planes[:4]) for c in controls),
            }
            for field, value in expected.items():
                if type(event.get(field)) is not type(value) or event[field] != value:
                    raise V4ProtocolError("response_validation_failed", f"wave {field} mismatch")
            for field in ("failure_stage", "error", "failed_wave_index", "failed_dpu_id",
                          "failed_operation_index", "failed_completion_mask",
                          "failed_completion_status", "failed_completion_stage", "failed_product"):
                if field not in event or event[field] is not None:
                    raise V4ProtocolError("response_validation_failed", f"completed wave has {field}")
            for field in ("h2d_time_s", "kernel_time_s", "d2h_time_s", "output_time_s",
                          "total_route_time_s"):
                value = event.get(field)
                if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                    raise V4ProtocolError("response_validation_failed", f"invalid wave {field}")
            name = f"wave-result-{sequence:020d}.bin"
            if event.get("response_path") != name:
                raise V4ProtocolError("response_validation_failed", "wave result path mismatch")
            if time.monotonic() >= deadline:
                raise V4ProtocolError("kernel_timeout", "wave response exceeded deadline")
            snapshot = self._read_wave_snapshot(name, output_bytes)
            if time.monotonic() >= deadline:
                raise V4ProtocolError("kernel_timeout", "wave result read exceeded deadline")
            if hashlib.sha256(snapshot).hexdigest() != event.get("response_sha256"):
                raise V4ProtocolError("response_validation_failed", "wave result hash mismatch")
            if time.monotonic() >= deadline:
                raise V4ProtocolError("kernel_timeout", "wave result hashing exceeded deadline")
            results = decode_wave_results(snapshot, waves)
            if time.monotonic() >= deadline:
                raise V4ProtocolError("kernel_timeout", "wave response exceeded deadline")
        except (V4Error, ValueError, OSError) as exc:
            error = exc if isinstance(exc, V4Error) else V4ProtocolError(
                "response_validation_failed", str(exc)
            )
            error.wave_response = dict(event)  # type: ignore[attr-defined]
            error.backend_facts = {  # type: ignore[attr-defined]
                **event, "request_path": request_name, "request_sha256": request_hash,
            }
            self._poisoned = True
            try:
                self._close_unlocked(timeout_s=max(0.001, deadline - time.monotonic()))
            except BaseException:
                self._terminate()
            raise error from (exc if error is not exc else None)
        self._wave_plan_sha256 = plan_sha256
        self._last_wave_sequence = sequence
        self._last_sequence = waves[-1][0].control.request_sequence
        return {
            **event, "results": results, "request_path": request_name,
            "request_sha256": request_hash,
            "host_submit_timing": {
                "preparation_s": preparation_s, "protocol_write_s": write_s,
                "response_wait_s": wait_s,
                "response_validation_s": time.perf_counter() - validation_started,
                "total_submit_s": time.perf_counter() - started,
            },
        }

    def _read_wave_snapshot(self, name: str, expected_bytes: int) -> bytes:
        fd = os.open(self.session_root / name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(fd, "rb") as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_size != expected_bytes:
                raise ValueError("wave result is not a regular file of the expected size")
            data = stream.read(expected_bytes + 1)
        if len(data) != expected_bytes:
            raise ValueError("wave result size changed during read")
        return data

    def _raise_packed_operation_failure(
        self,
        event: Mapping[str, Any],
        error: V4Error,
        *,
        responses: Sequence[Mapping[str, Any]] = (),
    ) -> None:
        """Preserve native partial-result metadata before poisoning the session."""

        operation_response = dict(event)
        partial_responses = tuple(dict(response) for response in responses)
        facts = {
            "operation_sequence": event.get("operation_sequence"),
            "response_path": event.get("response_path"),
            "response_sha256": event.get("response_sha256"),
            "response_count": event.get("response_count"),
            "completed_request_count": event.get("completed_request_count"),
            "failed_request_index": event.get("failed_request_index"),
        }
        # V4Error predates packed-operation partial responses and intentionally
        # remains unchanged here.  Attach the private protocol facts directly
        # to this exception so callers can retain the response file and hash.
        error.operation_response = operation_response  # type: ignore[attr-defined]
        error.partial_responses = partial_responses  # type: ignore[attr-defined]
        error.backend_facts = facts  # type: ignore[attr-defined]
        error.response_path = event.get("response_path")  # type: ignore[attr-defined]
        error.response_sha256 = event.get("response_sha256")  # type: ignore[attr-defined]
        error.response_count = event.get("response_count")  # type: ignore[attr-defined]
        error.completed_request_count = event.get(  # type: ignore[attr-defined]
            "completed_request_count"
        )
        error.failed_request_index = event.get(  # type: ignore[attr-defined]
            "failed_request_index"
        )
        self._poisoned = True
        self._close_unlocked()
        raise error

    def _validate_packed_operation_metadata(
        self,
        event: Mapping[str, Any],
        *,
        expected_count: int,
        expected_operation_sequence: int,
    ) -> tuple[int, int, int | None]:
        response_count = self._nonnegative_int(
            event.get("response_count"), "response_count"
        )
        completed_request_count = self._nonnegative_int(
            event.get("completed_request_count"), "completed_request_count"
        )
        if response_count > expected_count:
            raise V4ProtocolError(
                "response_validation_failed",
                "packed response count exceeds the operation descriptor count",
            )
        if completed_request_count > response_count:
            raise V4ProtocolError(
                "response_validation_failed",
                "packed completed request count exceeds response count",
            )
        failed_raw = event.get("failed_request_index")
        failed_request_index = (
            None
            if failed_raw is None
            else self._nonnegative_int(failed_raw, "failed_request_index")
        )
        if failed_request_index is not None:
            if failed_request_index >= expected_count:
                raise V4ProtocolError(
                    "response_validation_failed",
                    "packed failed request index exceeds the operation descriptor count",
                )
            if completed_request_count != failed_request_index:
                raise V4ProtocolError(
                    "response_validation_failed",
                    "packed completed count does not precede the failed request",
                )
            if response_count != completed_request_count + 1:
                raise V4ProtocolError(
                    "response_validation_failed",
                    "packed partial response count is inconsistent with the failed request",
                )
        if event.get("operation_sequence") != expected_operation_sequence and not (
            event.get("status") != "completed"
            and event.get("operation_sequence") == 0
            and response_count == 0
            and completed_request_count == 0
            and failed_request_index is None
        ):
            raise V4ProtocolError(
                "protocol_error", "packed response operation sequence mismatch"
            )
        if event.get("status") == "completed":
            if (
                response_count != expected_count
                or completed_request_count != expected_count
                or failed_request_index is not None
            ):
                raise V4ProtocolError(
                    "response_validation_failed",
                    "completed packed response metadata is inconsistent",
                )
        return response_count, completed_request_count, failed_request_index

    def _read_packed_operation_responses(
        self, event: Mapping[str, Any], *, response_count: int
    ) -> list[Mapping[str, Any]]:
        response_rel = event.get("response_path")
        response_digest = event.get("response_sha256")
        if response_count == 0 and response_rel is None and response_digest is None:
            return []
        if not isinstance(response_rel, str) or not isinstance(response_digest, str):
            raise V4ProtocolError(
                "protocol_error", "packed response file identity is missing"
            )
        if not _HEX_SHA256.fullmatch(response_digest):
            raise V4ProtocolError(
                "response_validation_failed", "packed response hash is malformed"
            )
        try:
            response_path = self.session_root / _safe_relative(response_rel)
            response_path.resolve().relative_to(self.session_root)
            actual_response_digest = _file_sha256(response_path)
            lines = response_path.read_text(encoding="utf-8").splitlines()
            responses: list[Mapping[str, Any]] = []
            for line in lines:
                value = json.loads(line)
                if not isinstance(value, Mapping):
                    raise ValueError("packed response record is not an object")
                responses.append(value)
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
            raise V4ProtocolError(
                "response_validation_failed", "packed response JSONL is invalid"
            ) from exc
        if actual_response_digest != response_digest:
            raise V4ProtocolError(
                "response_validation_failed", "packed response hash mismatch"
            )
        if len(responses) != response_count:
            raise V4ProtocolError(
                "response_validation_failed",
                "packed response count does not match the operation metadata",
            )
        return responses

    def _validate_partial_response(
        self, response: Mapping[str, Any], artifact: Any
    ) -> None:
        if response.get("event") != "RESPONSE":
            raise V4ProtocolError(
                "response_validation_failed",
                "packed partial response has the wrong event type",
            )
        if response.get("status") != "failed":
            raise V4ProtocolError(
                "response_validation_failed",
                "packed failed request record is not marked failed",
            )
        failure_stage = response.get("failure_stage")
        if not isinstance(failure_stage, str) or not failure_stage:
            raise V4ProtocolError(
                "response_validation_failed",
                "packed failed request record lacks a failure stage",
            )
        if response.get("error") in (None, ""):
            raise V4ProtocolError(
                "response_validation_failed",
                "packed failed request record lacks an error",
            )
        self._validate_native_identity(response, event_name="PARTIAL_RESPONSE")
        if response.get("request_sequence") not in (0, artifact.request_sequence):
            raise V4ProtocolError(
                "response_validation_failed",
                "packed failed request sequence does not match the descriptor",
            )

    def _validate_artifact_payloads(self, artifact: V4RequestArtifact) -> None:
        """Reject changed staged operands before the native process can use them."""

        for record in artifact.work_units:
            for name, relative_path, expected_digest in (
                ("A", record.a_path, record.a_sha256),
                ("B", record.b_path, record.b_sha256),
            ):
                if not _HEX_SHA256.fullmatch(expected_digest):
                    self._poisoned = True
                    self._terminate()
                    raise V4ProtocolError(
                        "payload_validation_failed",
                        f"v4 {name} payload digest is malformed",
                    )
                path = self.session_root / _safe_relative(relative_path)
                try:
                    actual_digest = _file_sha256(path)
                except OSError as exc:
                    self._poisoned = True
                    self._terminate()
                    raise V4ProtocolError(
                        "payload_validation_failed",
                        f"v4 {name} payload is unavailable",
                    ) from exc
                if actual_digest != expected_digest:
                    self._poisoned = True
                    self._terminate()
                    raise V4ProtocolError(
                        "payload_validation_failed",
                        f"v4 {name} payload digest mismatch",
                    )

    def _validate_response(
        self, event: Mapping[str, Any], artifact: V4RequestArtifact
    ) -> None:
        self._validate_native_identity(event, event_name="RESPONSE")
        if event.get("status") != "completed":
            raise V4ProtocolError(
                str(event.get("failure_stage") or "kernel_launch_failed"),
                str(event.get("error") or "v4 request failed"),
            )
        simulator = self.profile.execution_target == EXECUTION_TARGET_SIMULATOR
        required = {
            "target_requested": "simulator" if simulator else "hardware",
            "target_observed": "sdk_simulator" if simulator else "physical_hardware",
            "global_completeness": False,
            "task_contract_sha256": artifact.task_contract_sha256,
            "request_sha256": artifact.manifest_sha256,
            "request_manifest_sha256": artifact.manifest_sha256,
            "sidecar_sha256": artifact.sidecar_sha256,
            "rank_path": None if simulator else self.profile.rank_path,
            "dispatch_mode": "bulk_set_synchronous_v1",
            "bulk_set_launch_verified": True,
            "requested_dpu_count": self.profile.dpu_count,
            "allocated_dpu_count": self.profile.dpu_count,
            "tasklets_per_dpu": self.profile.tasklets_per_dpu,
            "allocation_verified": True,
            "hardware_allocation_verified": not simulator,
            "native_kernel_executed": True,
            "hardware_kernel_executed": not simulator,
            "simulator_kernel_executed": simulator,
            "cpu_fallback_used": False,
            "hardware_functionality_evidence": not simulator,
            "simulator_functionality_evidence": simulator,
        }
        if event.get("global_output_elements") != artifact.global_output_elements:
            raise V4ProtocolError(
                "protocol_error", "v4 response global output size mismatch"
            )
        for field, expected in required.items():
            if event.get(field) != expected:
                raise V4ProtocolError(
                    "protocol_error", f"v4 response field {field!r} is not verified"
                )
        if event.get("request_sequence") != artifact.request_sequence:
            raise V4ProtocolError("protocol_error", "v4 response sequence mismatch")
        if event.get("request_output_elements") != artifact.request_output_elements:
            raise V4ProtocolError(
                "protocol_error", "v4 response output coverage mismatch"
            )
        if (
            not isinstance(event.get("per_dpu"), list)
            or len(event["per_dpu"]) != self.profile.dpu_count
        ):
            raise V4ProtocolError(
                "protocol_error", "v4 response lacks one result per requested DPU"
            )
        transfer = event.get("transfer")
        if not isinstance(transfer, Mapping):
            raise V4ProtocolError(
                "protocol_error", "v4 response transfer summary is missing"
            )
        summary_h2d = self._nonnegative_int(
            transfer.get("h2d_bytes"), "transfer.h2d_bytes"
        )
        summary_d2h = self._nonnegative_int(
            transfer.get("d2h_bytes"), "transfer.d2h_bytes"
        )
        summary_total = self._nonnegative_int(
            transfer.get("total_bytes"), "transfer.total_bytes"
        )
        if summary_total != summary_h2d + summary_d2h:
            raise V4ProtocolError(
                "protocol_error", "v4 response transfer total is inconsistent"
            )

        records_by_dpu = {record.local_dpu_id: record for record in artifact.work_units}
        seen_dpu_ids: set[int] = set()
        per_dpu_h2d = 0
        per_dpu_d2h = 0
        for result in event["per_dpu"]:
            if not isinstance(result, Mapping):
                raise V4ProtocolError(
                    "protocol_error", "v4 response per-DPU record is invalid"
                )
            dpu_id = self._nonnegative_int(result.get("dpu_id"), "per_dpu.dpu_id")
            if dpu_id in seen_dpu_ids or dpu_id not in records_by_dpu:
                raise V4ProtocolError(
                    "protocol_error", "v4 response DPU IDs are not dense and unique"
                )
            seen_dpu_ids.add(dpu_id)
            record = records_by_dpu[dpu_id]
            if result.get("tile_id") != record.tile_id:
                raise V4ProtocolError("protocol_error", "v4 response tile ID mismatch")
            if result.get("completion_status") != STATUS_COMPLETED:
                raise V4ProtocolError(
                    "protocol_error", "v4 response completion status is not completed"
                )
            expected_elements = (
                0
                if record.flags & FLAG_ZERO_WORK
                else record.m_elements * record.n_elements
            )
            if result.get("processed_elements") != expected_elements:
                raise V4ProtocolError(
                    "protocol_error", "v4 response processed element count mismatch"
                )
            expected_h2d = (
                record.a_transfer_bytes + record.b_transfer_bytes + CONTROL_BYTES
            )
            expected_d2h = record.c_transfer_bytes + COMPLETION_BYTES
            result_h2d = self._nonnegative_int(
                result.get("h2d_bytes"), "per_dpu.h2d_bytes"
            )
            result_d2h = self._nonnegative_int(
                result.get("d2h_bytes"), "per_dpu.d2h_bytes"
            )
            if result_h2d != expected_h2d or result_d2h != expected_d2h:
                raise V4ProtocolError(
                    "protocol_error", "v4 response per-DPU transfer accounting mismatch"
                )
            per_dpu_h2d += result_h2d
            per_dpu_d2h += result_d2h
        if seen_dpu_ids != set(range(self.profile.dpu_count)):
            raise V4ProtocolError(
                "protocol_error", "v4 response is missing a dense DPU result"
            )
        if (summary_h2d, summary_d2h) != (per_dpu_h2d, per_dpu_d2h):
            raise V4ProtocolError(
                "protocol_error", "v4 response aggregate transfer accounting mismatch"
            )

    @staticmethod
    def _nonnegative_int(value: Any, field: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise V4ProtocolError(
                "protocol_error",
                f"v4 response field {field!r} is not a non-negative integer",
            )
        return value

    def close(self, *, timeout_s: float | None = None) -> V4Release:
        self._enter_operation()
        try:
            return self._close_unlocked(timeout_s=timeout_s)
        finally:
            self._leave_operation()

    def _close_unlocked(self, *, timeout_s: float | None = None) -> V4Release:
        if self._release is not None:
            return self._release
        if self._closed:
            return self._release_record({}, False)
        alive_before_close = self.process.poll() is None
        self._closed = True
        sent_close = True
        try:
            self._send_raw("CLOSE\n")
        except V4Error:
            self._poisoned = True
            sent_close = False
        close_deadline = time.monotonic() + (timeout_s or self.profile.timeout_s)
        event: Mapping[str, Any] = {}
        confirmed = False
        if (alive_before_close and sent_close) or (
            self.profile.request_transport == REQUEST_TRANSPORT_PACKED_WAVE
        ):
            try:
                event = self._next_event(max(0.001, close_deadline - time.monotonic()))
                confirmed = (
                    event.get("event") == "RELEASE"
                    and event.get("status") == "released"
                    and event.get("release_succeeded") is True
                    and event.get("dpu_free_called_once") is True
                )
            except V4Error:
                self._poisoned = True
        if confirmed:
            try:
                returncode = self.process.wait(
                    timeout=max(0.001, close_deadline - time.monotonic())
                )
                confirmed = returncode == 0
            except (OSError, subprocess.SubprocessError):
                confirmed = False
        if not confirmed:
            self._terminate()
        for pump in (self._stdout_pump, self._stderr_pump):
            pump.thread.join(timeout=max(0.0, close_deadline - time.monotonic()))
        pumps_finished = all(
            not pump.thread.is_alive()
            for pump in (self._stdout_pump, self._stderr_pump)
        )
        confirmed = bool(
            confirmed
            and pumps_finished
            and not self._stdout_pump.overflowed
            and not self._stderr_pump.overflowed
        )
        release = self._release_record(event, confirmed)
        self._release = release
        return release

    def _terminate(self) -> None:
        try:
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=1.0)
        except (OSError, subprocess.SubprocessError):
            pass

    def __enter__(self) -> "V4Session":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()


V4HardwareSession = V4Session


__all__ = [
    "V4HardwareSession",
    "V4Release",
    "V4Session",
]
