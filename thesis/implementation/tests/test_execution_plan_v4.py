from __future__ import annotations

from collections.abc import Callable
import hashlib
import json
from pathlib import Path
import queue
import struct
import time

import pytest

from quantum_bench.targets.upmem import execution_plan_v4 as v4


TASK_HASH = "ab" * 32


class FakeStream:
    def __init__(self) -> None:
        self._lines: queue.Queue[str] = queue.Queue()
        self.read_sizes: list[int] = []

    def emit(self, line: str) -> None:
        self._lines.put(line)

    def readline(self, size: int = -1) -> str:
        self.read_sizes.append(size)
        return self._lines.get()


class FakeStdin:
    def __init__(self, callback: Callable[[str], None], process: "FakeProcess") -> None:
        self._callback = callback
        self._process = process
        self.commands: list[str] = []

    def write(self, value: str) -> int:
        if self._process.poll() is not None:
            raise BrokenPipeError("fake process is stopped")
        self.commands.append(value)
        self._callback(value)
        return len(value)

    def flush(self) -> None:
        return None


class FakeProcess:
    def __init__(
        self, response_factory: Callable[[str], str | dict[str, object] | None]
    ) -> None:
        self.stdout = FakeStream()
        self.stderr = FakeStream()
        self._returncode: int | None = None
        self.response_factory = response_factory
        self.stdin = FakeStdin(self._handle, self)
        self.stdout.emit(
            json.dumps(
                {
                    "event": "READY",
                    "status": "ready",
                    "target_observed": "physical_hardware",
                    "rank_path": "/dev/dpu_rank0",
                    "requested_dpu_count": 2,
                    "allocated_dpu_count": 2,
                    "tasklets_per_dpu": 1,
                    "hardware_allocation_verified": True,
                    "simulator_kernel_executed": False,
                    "cpu_fallback_used": False,
                }
            )
            + "\n"
        )

    def _handle(self, value: str) -> None:
        command = value.strip()
        if command.startswith("SUBMIT "):
            response = self.response_factory(command)
            if response is not None:
                self.stdout.emit(
                    (response if isinstance(response, str) else json.dumps(response))
                    + "\n"
                )
        elif command == "CLOSE":
            self.stdout.emit(
                json.dumps(
                    {
                        "event": "RELEASE",
                        "status": "released",
                        "release_succeeded": True,
                        "dpu_free_called_once": True,
                    }
                )
                + "\n"
            )
            self._returncode = 0
            self.stdout.emit("")
            self.stderr.emit("")

    def poll(self) -> int | None:
        return self._returncode

    def terminate(self) -> None:
        self._returncode = -15
        self.stdout.emit("")
        self.stderr.emit("")

    def kill(self) -> None:
        self._returncode = -9
        self.stdout.emit("")
        self.stderr.emit("")

    def wait(self, timeout: float | None = None) -> int:
        del timeout
        if self._returncode is None:
            self._returncode = 0
        return self._returncode


def _float_payload(count: int) -> bytes:
    return struct.pack("<" + "f" * count, *range(1, count + 1))


def _int8_payload(count: int) -> bytes:
    return bytes(((index % 7) - 3) & 0xFF for index in range(count))


def _artifact(
    tmp_path: Path,
    *,
    numeric_mode: str = v4.NUMERIC_FLOAT32,
    request_sequence: int = 0,
) -> v4.V4RequestArtifact:
    profile = v4.V4Profile(dpu_count=2, numeric_mode=numeric_mode)
    element_bytes = 4 if numeric_mode == v4.NUMERIC_FLOAT32 else 1
    payload = _float_payload(4) if element_bytes == 4 else _int8_payload(4)
    return v4.build_v4_request(
        tmp_path,
        profile=profile,
        canonical_batch_count=1,
        canonical_m=2,
        canonical_n=2,
        canonical_k=2,
        work_units=[
            v4.V4WorkUnit(
                local_dpu_id=0,
                tile_id=7,
                batch_index=0,
                m_offset=0,
                n_offset=0,
                k_offset=0,
                m_elements=2,
                n_elements=2,
                k_elements=2,
                a_payload=payload,
                b_payload=payload,
            )
        ],
        task_contract_sha256=TASK_HASH,
        request_sequence=request_sequence,
    )


def _valid_response(artifact: v4.V4RequestArtifact) -> dict[str, object]:
    per_dpu: list[dict[str, int]] = []
    h2d = 0
    d2h = 0
    for record in artifact.work_units:
        record_h2d = (
            record.a_transfer_bytes + record.b_transfer_bytes + v4.CONTROL_BYTES
        )
        record_d2h = record.c_transfer_bytes + v4.COMPLETION_BYTES
        h2d += record_h2d
        d2h += record_d2h
        per_dpu.append(
            {
                "dpu_id": record.local_dpu_id,
                "tile_id": record.tile_id,
                "completion_status": v4.STATUS_COMPLETED,
                "processed_elements": 0
                if record.flags & v4.FLAG_ZERO_WORK
                else record.m_elements * record.n_elements,
                "h2d_bytes": record_h2d,
                "d2h_bytes": record_d2h,
            }
        )
    return {
        "event": "RESPONSE",
        "status": "completed",
        "target_requested": "hardware",
        "target_observed": "physical_hardware",
        "rank_path": "/dev/dpu_rank0",
        "request_sequence": artifact.request_sequence,
        "request_output_elements": artifact.request_output_elements,
        "global_output_elements": artifact.global_output_elements,
        "global_completeness": False,
        "task_contract_sha256": artifact.task_contract_sha256,
        "request_sha256": artifact.manifest_sha256,
        "request_manifest_sha256": artifact.manifest_sha256,
        "sidecar_sha256": artifact.sidecar_sha256,
        "dispatch_mode": "bulk_set_synchronous_v1",
        "bulk_set_launch_verified": True,
        "requested_dpu_count": 2,
        "allocated_dpu_count": 2,
        "tasklets_per_dpu": 1,
        "hardware_allocation_verified": True,
        "native_kernel_executed": True,
        "hardware_kernel_executed": True,
        "simulator_kernel_executed": False,
        "cpu_fallback_used": False,
        "hardware_functionality_evidence": True,
        "transfer": {"h2d_bytes": h2d, "d2h_bytes": d2h, "total_bytes": h2d + d2h},
        "per_dpu": per_dpu,
    }


def _session(
    tmp_path: Path,
    artifact: v4.V4RequestArtifact,
    response_factory: Callable[[str], str | dict[str, object] | None],
    *,
    profile: v4.V4Profile | None = None,
) -> tuple[v4.V4Session, FakeProcess]:
    process: FakeProcess | None = None

    def factory(*args: object, **kwargs: object) -> FakeProcess:
        del args, kwargs
        nonlocal process
        process = FakeProcess(response_factory)
        return process

    session = v4.V4Session.start(
        ["fake-v4-host"],
        session_root=tmp_path,
        profile=profile
        or v4.V4Profile(dpu_count=2, rank_path="/dev/dpu_rank0", timeout_s=0.2),
        environment={"UPMEM_ALLOW_PHYSICAL_HARDWARE": "1"},
        popen_factory=factory,
    )
    assert process is not None
    return session, process


def test_native_struct_formats_and_field_order_match_c() -> None:
    assert v4.HEADER_FORMAT == "<8s10I7Q32s32s"
    assert v4.WORK_UNIT_FORMAT == "<2I5Q9I"
    assert v4.CONTROL_FORMAT == "<18I"
    assert v4.COMPLETION_FORMAT == "<4I3Q"
    assert v4.HEADER_BYTES == 168
    assert v4.WORK_UNIT_BYTES == 84
    assert v4.CONTROL_BYTES == 72
    assert v4.COMPLETION_BYTES == 40

    header = v4.V4Header(
        canonical_batch_count=2,
        canonical_m=3,
        canonical_n=4,
        canonical_k=5,
        global_output_elements=24,
        request_output_elements=12,
        request_sequence=9,
        task_contract_sha256=bytes.fromhex(TASK_HASH),
        request_sha256=bytes.fromhex("cd" * 32),
        work_unit_count=2,
        dpu_count=2,
        tasklets_per_dpu=1,
        numeric_mode=v4.NUMERIC_MODE_HOST_PACKED_INT8,
    )
    assert v4.V4Header.unpack(header.pack()) == header
    record = v4.V4WorkUnitRecord(
        0,
        0,
        17,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        16,
        24,
        32,
        48,
        64,
        "a",
        "b",
        "c",
        "cd" * 32,
        "ef" * 32,
    )
    assert (
        v4.V4WorkUnitRecord.unpack(
            record.pack(), paths=("a", "b", "c"), payload_sha256=("cd" * 32, "ef" * 32)
        )
        == record
    )
    completion = v4.V4Completion(1, 2, 3, 4, 5, 6, 7)
    assert v4.V4Completion.unpack(completion.pack()) == completion
    assert struct.unpack("<4I3Q", completion.pack())[-1] == 7


def test_builder_pads_float32_and_int8_and_fills_zero_work(tmp_path: Path) -> None:
    float_artifact = _artifact(tmp_path / "float")
    float_unit = float_artifact.work_units[0]
    zero_unit = float_artifact.work_units[1]
    assert (
        float_unit.a_transfer_bytes,
        float_unit.b_transfer_bytes,
        float_unit.c_transfer_bytes,
    ) == (16, 16, 16)
    assert zero_unit.flags == v4.FLAG_ZERO_WORK
    assert (
        float_artifact.request_output_elements,
        float_artifact.global_output_elements,
    ) == (4, 4)
    assert (float_artifact.root / zero_unit.a_path).read_bytes() == b""
    assert zero_unit.a_sha256 == hashlib.sha256(b"").hexdigest()
    assert zero_unit.b_sha256 == hashlib.sha256(b"").hexdigest()

    int_artifact = _artifact(
        tmp_path / "int8", numeric_mode=v4.NUMERIC_HOST_PACKED_INT8
    )
    int_unit = int_artifact.work_units[0]
    assert (
        int_unit.a_transfer_bytes,
        int_unit.b_transfer_bytes,
        int_unit.c_transfer_bytes,
    ) == (8, 8, 16)
    assert len((int_artifact.root / int_unit.a_path).read_bytes()) == 8
    assert int_artifact.manifest_sha256 == v4._file_sha256(int_artifact.manifest_path)
    assert int_artifact.sidecar_sha256 == v4._file_sha256(int_artifact.sidecar_path)
    assert ".." not in int_artifact.manifest_path.read_text(encoding="utf-8")
    manifest = int_artifact.manifest_path.read_text(encoding="utf-8")
    assert int_unit.a_sha256 in manifest
    assert int_unit.b_sha256 in manifest


def test_builder_supports_batch_and_k_chunk_coverage(tmp_path: Path) -> None:
    profile = v4.V4Profile(dpu_count=2, numeric_mode=v4.NUMERIC_HOST_PACKED_INT8)
    unit_payload = _int8_payload(4)
    artifact = v4.build_v4_request(
        tmp_path,
        profile=profile,
        canonical_batch_count=2,
        canonical_m=2,
        canonical_n=2,
        canonical_k=4,
        work_units=[
            v4.V4WorkUnit(0, 1, 0, 0, 0, 2, 2, 2, 2, unit_payload, unit_payload),
            v4.V4WorkUnit(1, 2, 1, 0, 0, 0, 2, 2, 2, unit_payload, unit_payload),
        ],
        task_contract_sha256=TASK_HASH,
        request_sequence=4,
    )
    assert artifact.header.canonical_batch_count == 2
    assert artifact.header.request_output_elements == 8
    assert artifact.work_units[0].k_offset == 2
    assert artifact.work_units[1].batch_index == 1
    assert all(record.c_transfer_bytes == 16 for record in artifact.work_units)


def test_builder_rejects_unsafe_paths_and_int8_k_overflow(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unsafe"):
        v4._safe_relative("../outside.bin")
    with pytest.raises(ValueError, match="unsafe"):
        v4._safe_relative("/absolute.bin")
    with pytest.raises(ValueError, match="unsafe"):
        v4._safe_relative("nested\\outside.bin")
    profile = v4.V4Profile(dpu_count=1, numeric_mode=v4.NUMERIC_HOST_PACKED_INT8)
    with pytest.raises(ValueError, match="canonical dimensions exceed native bounds"):
        v4.build_v4_request(
            tmp_path / "overflow",
            profile=profile,
            canonical_batch_count=1,
            canonical_m=1,
            canonical_n=1,
            canonical_k=v4.MAX_CONTRACTED + 1,
            work_units=[
                v4.V4WorkUnit(0, 1, 0, 0, 0, 0, 1, 1, v4.MAX_CONTRACTED + 1, b"x", b"x")
            ],
            task_contract_sha256=TASK_HASH,
            request_sequence=0,
        )


def test_physical_opt_in_and_backend_environment_guards(tmp_path: Path) -> None:
    with pytest.raises(v4.V4Error, match="hardware_opt_in_missing"):
        v4.V4Session.start(
            ["fake"],
            session_root=tmp_path,
            profile=v4.V4Profile(dpu_count=1, rank_path="/dev/dpu_rank0"),
            environment={},
            popen_factory=FakeProcess,
        )
    for variable in ("DPU_BACKEND", "UPMEM_EXECUTION_MODE"):
        with pytest.raises(v4.V4Error, match=variable):
            v4.V4Session.start(
                ["fake"],
                session_root=tmp_path,
                profile=v4.V4Profile(dpu_count=1, rank_path="/dev/dpu_rank0"),
                environment={
                    "UPMEM_ALLOW_PHYSICAL_HARDWARE": "1",
                    variable: "simulator",
                },
                popen_factory=FakeProcess,
            )


def test_successful_submit_and_release(tmp_path: Path) -> None:
    artifact = _artifact(tmp_path)
    session, process = _session(
        tmp_path, artifact, lambda command: _valid_response(artifact)
    )
    response = session.submit(artifact)
    assert response["global_completeness"] is False
    assert process.stdin.commands == [
        f"SUBMIT requests/0000000000000000/manifest.txt {artifact.manifest_sha256}\n"
    ]
    release = session.close()
    assert release.release_confirmed is True
    assert release.event["event"] == "RELEASE"
    assert process.stdin.commands[-1] == "CLOSE\n"


def test_persistent_session_bounds_each_event_not_cumulative_stdout(
    tmp_path: Path,
) -> None:
    artifacts = tuple(
        _artifact(tmp_path, request_sequence=sequence) for sequence in range(8)
    )
    by_hash = {artifact.manifest_sha256: artifact for artifact in artifacts}

    def response(command: str) -> dict[str, object]:
        artifact = next(item for digest, item in by_hash.items() if digest in command)
        return _valid_response(artifact)

    profile = v4.V4Profile(
        dpu_count=2,
        rank_path="/dev/dpu_rank0",
        timeout_s=0.2,
        max_stdout_bytes=2048,
        max_retained_output_bytes=512,
    )
    session, _ = _session(tmp_path, artifacts[0], response, profile=profile)
    for artifact in artifacts:
        assert session.submit(artifact)["status"] == "completed"
    release = session.close()
    assert release.release_confirmed is True
    assert release.stdout_total_bytes > profile.max_stdout_bytes
    assert release.stdout_truncated is True
    assert len(release.stdout.encode("utf-8")) <= profile.max_retained_output_bytes


def test_persistent_session_still_rejects_one_oversized_event(tmp_path: Path) -> None:
    artifact = _artifact(tmp_path)
    response = _valid_response(artifact)
    response["padding"] = "x" * 4096
    profile = v4.V4Profile(
        dpu_count=2,
        rank_path="/dev/dpu_rank0",
        timeout_s=0.2,
        max_stdout_bytes=2048,
        max_retained_output_bytes=512,
    )
    session, process = _session(
        tmp_path, artifact, lambda command: response, profile=profile
    )
    with pytest.raises(v4.V4ProtocolError) as exc_info:
        session.submit(artifact)
    assert exc_info.value.failure_stage == "protocol_output_limit"
    assert profile.max_stdout_bytes + 1 in process.stdout.read_sizes


def test_non_protocol_output_is_drained_and_retained_within_byte_limit() -> None:
    stderr = FakeStream()
    pump = v4._OutputPump(
        stderr,
        line_limit=16,
        retained_limit=8,
        output_queue=None,
    )
    pump.start()
    stderr.emit("x" * 64 + "\n")
    stderr.emit("sentinel\n")
    stderr.emit("")
    pump.thread.join(timeout=1.0)
    assert not pump.thread.is_alive()
    assert pump.overflowed is True
    assert pump.total_size == 74
    assert pump.retained_truncated is True
    assert len("".join(pump.chunks).encode("utf-8")) <= 8
    assert "".join(pump.chunks).endswith("inel\n")


def test_protocol_event_queue_overflow_fails_closed_without_growing() -> None:
    stdout = FakeStream()
    events: queue.Queue[tuple[str, str | None]] = queue.Queue(maxsize=1)
    pump = v4._OutputPump(
        stdout,
        line_limit=64,
        retained_limit=32,
        output_queue=events,
    )
    stdout.emit("{}\n")
    stdout.emit("{}\n")
    pump.start()
    pump.thread.join(timeout=1.0)
    assert not pump.thread.is_alive()
    assert events.qsize() == 1
    assert pump.overflowed is True
    assert pump.queue_overflowed is True
    assert len("".join(pump.chunks).encode("utf-8")) <= 32


def test_oversized_stderr_invalidates_physical_release(tmp_path: Path) -> None:
    artifact = _artifact(tmp_path)
    profile = v4.V4Profile(
        dpu_count=2,
        rank_path="/dev/dpu_rank0",
        timeout_s=0.2,
        max_stderr_bytes=64,
        max_retained_output_bytes=32,
    )
    session, process = _session(
        tmp_path,
        artifact,
        lambda command: _valid_response(artifact),
        profile=profile,
    )
    process.stderr.emit("x" * 128 + "\n")
    assert session.submit(artifact)["status"] == "completed"
    release = session.close()
    assert release.release_confirmed is False
    assert release.stderr_limit_exceeded is True
    assert release.stderr_truncated is True
    assert len(release.stderr.encode("utf-8")) <= profile.max_retained_output_bytes


def test_multibyte_diagnostic_tail_respects_byte_limit() -> None:
    stream = FakeStream()
    pump = v4._OutputPump(
        stream,
        line_limit=64,
        retained_limit=7,
        output_queue=None,
    )
    pump.start()
    stream.emit("é" * 10 + "\n")
    stream.emit("")
    pump.thread.join(timeout=1.0)
    assert not pump.thread.is_alive()
    assert pump.retained_truncated is True
    assert len("".join(pump.chunks).encode("utf-8")) <= 7


def test_manifest_hash_mismatch_is_rejected(tmp_path: Path) -> None:
    artifact = _artifact(tmp_path)
    session, process = _session(tmp_path, artifact, lambda command: None)
    artifact.manifest_path.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(v4.V4ProtocolError, match="hash mismatch"):
        session.submit(artifact)
    assert process.poll() is not None
    assert session.close().release_confirmed is False


def test_payload_tamper_with_unchanged_length_is_rejected_before_submit(
    tmp_path: Path,
) -> None:
    artifact = _artifact(tmp_path)
    session, process = _session(tmp_path, artifact, lambda command: None)
    payload_path = artifact.root / artifact.work_units[0].a_path
    payload = bytearray(payload_path.read_bytes())
    payload[0] ^= 0x01
    payload_path.write_bytes(payload)
    with pytest.raises(v4.V4ProtocolError, match="payload digest mismatch") as exc_info:
        session.submit(artifact)
    assert exc_info.value.failure_stage == "payload_validation_failed"
    assert not process.stdin.commands


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda response: response["per_dpu"].append(response["per_dpu"][0]),
            "lacks one result",
        ),
        (
            lambda response: response["per_dpu"].__setitem__(
                1, {**response["per_dpu"][1], "dpu_id": 0}
            ),
            "dense and unique",
        ),
        (
            lambda response: response["per_dpu"].__setitem__(
                0, {**response["per_dpu"][0], "tile_id": 999}
            ),
            "tile ID mismatch",
        ),
        (
            lambda response: response["per_dpu"].__setitem__(
                0, {**response["per_dpu"][0], "completion_status": 2}
            ),
            "completion status",
        ),
        (
            lambda response: response["per_dpu"].__setitem__(
                0, {**response["per_dpu"][0], "processed_elements": 999}
            ),
            "processed element",
        ),
        (
            lambda response: response["per_dpu"].__setitem__(
                0, {**response["per_dpu"][0], "h2d_bytes": 0}
            ),
            "per-DPU transfer",
        ),
        (
            lambda response: response["transfer"].__setitem__("total_bytes", 1),
            "transfer total",
        ),
        (
            lambda response: response["transfer"].update(
                {
                    "h2d_bytes": 0,
                    "total_bytes": response["transfer"]["d2h_bytes"],
                }
            ),
            "aggregate transfer",
        ),
    ],
)
def test_response_validation_rejects_malformed_per_dpu_evidence(
    tmp_path: Path,
    mutate: Callable[[dict[str, object]], None],
    message: str,
) -> None:
    artifact = _artifact(tmp_path)
    response = _valid_response(artifact)
    mutate(response)
    session, _ = _session(tmp_path, artifact, lambda command: response)
    with pytest.raises(v4.V4ProtocolError, match=message):
        session.submit(artifact)


def test_malformed_and_fallback_responses_poison_the_session(tmp_path: Path) -> None:
    artifact = _artifact(tmp_path)
    session, _ = _session(tmp_path, artifact, lambda command: "not-json")
    with pytest.raises(v4.V4ProtocolError, match="malformed JSON"):
        session.submit(artifact)
    assert session.close().release_confirmed is False

    artifact2 = _artifact(tmp_path / "fallback")
    fallback = _valid_response(artifact2)
    fallback["cpu_fallback_used"] = True
    session2, process2 = _session(
        tmp_path / "fallback", artifact2, lambda command: fallback
    )
    with pytest.raises(v4.V4ProtocolError, match="cpu_fallback_used"):
        session2.submit(artifact2)
    assert process2.stdin.commands[-1] == "CLOSE\n"
    assert session2.close().release_confirmed is True


def test_timeout_does_not_fabricate_release(tmp_path: Path) -> None:
    artifact = _artifact(tmp_path)
    session, process = _session(tmp_path, artifact, lambda command: None)
    session.profile = v4.V4Profile(
        dpu_count=2, rank_path="/dev/dpu_rank0", timeout_s=0.03
    )
    started = time.monotonic()
    with pytest.raises(v4.V4ProtocolError, match="timed out"):
        session.submit(artifact)
    assert time.monotonic() - started < 1.0
    assert process.poll() is not None
    assert session.close().release_confirmed is False
