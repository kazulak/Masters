from __future__ import annotations

from collections.abc import Callable
import hashlib
import json
from pathlib import Path
import queue
import struct
import threading
import time

import pytest

import quantum_bench.upmem.native_session as session_v4
import quantum_bench.upmem.protocol as v4
from quantum_bench.upmem.packed_operation import (
    PackedOperation,
    PackedV4Request,
    build_packed_v4_request,
    pack_operation,
)


TASK_HASH = "ab" * 32


def test_public_build_request_alias_is_preserved() -> None:
    assert v4.build_request is v4.build_v4_request


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
        self,
        response_factory: Callable[[str], str | dict[str, object] | None],
        *,
        profile: v4.V4Profile | None = None,
        ready_overrides: dict[str, object] | None = None,
    ) -> None:
        profile = profile or v4.V4Profile(
            dpu_count=2,
            rank_path="/dev/dpu_rank0",
        )
        simulator = profile.execution_target == v4.EXECUTION_TARGET_SIMULATOR
        self.stdout = FakeStream()
        self.stderr = FakeStream()
        self._returncode: int | None = None
        self.terminate_calls = 0
        self.kill_calls = 0
        self.response_factory = response_factory
        self.stdin = FakeStdin(self._handle, self)
        ready = {
            "event": "READY",
            "status": "ready",
            "target_requested": "simulator" if simulator else "hardware",
            "target_observed": ("sdk_simulator" if simulator else "physical_hardware"),
            "rank_path": None if simulator else profile.rank_path,
            "requested_dpu_count": profile.dpu_count,
            "allocated_dpu_count": profile.dpu_count,
            "tasklets_per_dpu": profile.tasklets_per_dpu,
            "hardware_allocation_verified": not simulator,
            "allocation_verified": True,
            "simulator_kernel_executed": False,
            "cpu_fallback_used": False,
            "request_transport": profile.request_transport,
            **v4.native_execution_identity(profile.execution_target),
        }
        ready.update(ready_overrides or {})
        self.stdout.emit(json.dumps(ready) + "\n")

    def _handle(self, value: str) -> None:
        command = value.strip()
        if command.startswith(("SUBMIT ", "SUBMIT_PACKED_OPERATION ", "SUBMIT_PACKED_WAVES ")):
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
        self.terminate_calls += 1
        self._returncode = -15
        self.stdout.emit("")
        self.stderr.emit("")

    def kill(self) -> None:
        self.kill_calls += 1
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
    dpu_count: int = 2,
) -> v4.V4RequestArtifact:
    profile = v4.V4Profile(dpu_count=dpu_count, numeric_mode=numeric_mode)
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


def _packed_operation(
    tmp_path: Path, *, count: int = 3
) -> tuple[PackedOperation, tuple[PackedV4Request, ...], v4.V4Profile]:
    profile = v4.V4Profile(
        dpu_count=1,
        rank_path="/dev/dpu_rank0",
        request_transport=v4.REQUEST_TRANSPORT_PACKED_OPERATION,
        timeout_s=0.2,
    )
    requests = tuple(
        build_packed_v4_request(
            tmp_path,
            profile=profile,
            canonical_batch_count=1,
            canonical_m=2,
            canonical_n=2,
            canonical_k=2,
            work_units=[
                v4.V4WorkUnit(
                    local_dpu_id=0,
                    tile_id=sequence + 7,
                    batch_index=0,
                    m_offset=0,
                    n_offset=0,
                    k_offset=0,
                    m_elements=2,
                    n_elements=2,
                    k_elements=2,
                    a_payload=_float_payload(4),
                    b_payload=_float_payload(4),
                )
            ],
            task_contract_sha256=TASK_HASH,
            request_sequence=sequence,
        )
        for sequence in range(count)
    )
    operation = pack_operation(
        tmp_path,
        requests=requests,
        operation_sequence=0,
        filename="packed/operation.bin",
    )
    operation.path.parent.mkdir(parents=True, exist_ok=True)
    operation.path.write_bytes(operation.data)
    return operation, requests, profile


def _valid_response(
    artifact: v4.V4RequestArtifact | PackedV4Request,
    *,
    profile: v4.V4Profile | None = None,
) -> dict[str, object]:
    profile = profile or v4.V4Profile(
        dpu_count=2,
        rank_path="/dev/dpu_rank0",
    )
    simulator = profile.execution_target == v4.EXECUTION_TARGET_SIMULATOR
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
        "target_requested": "simulator" if simulator else "hardware",
        "target_observed": "sdk_simulator" if simulator else "physical_hardware",
        "rank_path": None if simulator else profile.rank_path,
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
        "requested_dpu_count": profile.dpu_count,
        "allocated_dpu_count": profile.dpu_count,
        "tasklets_per_dpu": profile.tasklets_per_dpu,
        "hardware_allocation_verified": not simulator,
        "allocation_verified": True,
        "native_kernel_executed": True,
        "hardware_kernel_executed": not simulator,
        "simulator_kernel_executed": simulator,
        "cpu_fallback_used": False,
        "hardware_functionality_evidence": not simulator,
        "simulator_functionality_evidence": simulator,
        **v4.native_execution_identity(profile.execution_target),
        "transfer": {"h2d_bytes": h2d, "d2h_bytes": d2h, "total_bytes": h2d + d2h},
        "per_dpu": per_dpu,
    }


def _packed_failure_event(
    tmp_path: Path,
    operation: PackedOperation,
    requests: tuple[PackedV4Request, ...],
    profile: v4.V4Profile,
    failed_index: int,
) -> dict[str, object]:
    records = [
        _valid_response(request, profile=profile) for request in requests[: failed_index + 1]
    ]
    records[-1]["status"] = "failed"
    records[-1]["failure_stage"] = "request_execution_failed"
    records[-1]["error"] = "injected embedded request failure"
    response_path = tmp_path / "results" / "operation_0000000000000000.jsonl"
    response_path.parent.mkdir(parents=True, exist_ok=True)
    response_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    return {
        "event": "OPERATION_RESPONSE",
        "status": "failed",
        "failure_stage": "request_execution_failed",
        "error": "embedded request execution failed; operation stopped",
        "operation_sequence": operation.operation_sequence,
        "response_path": "results/operation_0000000000000000.jsonl",
        "response_count": failed_index + 1,
        "completed_request_count": failed_index,
        "failed_request_index": failed_index,
        "response_sha256": hashlib.sha256(response_path.read_bytes()).hexdigest(),
    }


def _session(
    tmp_path: Path,
    artifact: v4.V4RequestArtifact,
    response_factory: Callable[[str], str | dict[str, object] | None],
    *,
    profile: v4.V4Profile | None = None,
) -> tuple[session_v4.V4Session, FakeProcess]:
    process: FakeProcess | None = None
    selected_profile = profile or v4.V4Profile(
        dpu_count=2,
        rank_path="/dev/dpu_rank0",
        timeout_s=0.2,
    )

    def factory(*args: object, **kwargs: object) -> FakeProcess:
        del args, kwargs
        nonlocal process
        process = FakeProcess(response_factory, profile=selected_profile)
        return process

    session = session_v4.V4Session.start(
        ["fake-v4-host"],
        session_root=tmp_path,
        profile=selected_profile,
        environment=(
            {"DPU_BACKEND": "simulator"}
            if selected_profile.execution_target == v4.EXECUTION_TARGET_SIMULATOR
            else {"UPMEM_ALLOW_PHYSICAL_HARDWARE": "1"}
        ),
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


def test_session_rejects_conflicting_native_response_identity(tmp_path: Path) -> None:
    artifact = _artifact(tmp_path)
    response = _valid_response(artifact)
    response["kernel_identity"] = "wrong-native-kernel"
    session, _ = _session(tmp_path, artifact, lambda command: response)
    with pytest.raises(
        v4.V4ProtocolError, match="native identity field 'kernel_identity'"
    ):
        session.submit(artifact)

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
    for path in (
        "",
        ".",
        "..",
        "../outside.bin",
        "/absolute.bin",
        "nested//outside.bin",
        "nested/./outside.bin",
        "nested/../outside.bin",
        "nested\\outside.bin",
    ):
        with pytest.raises(ValueError, match="unsafe"):
            v4._safe_relative(path)
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
        session_v4.V4Session.start(
            ["fake"],
            session_root=tmp_path,
            profile=v4.V4Profile(dpu_count=1, rank_path="/dev/dpu_rank0"),
            environment={},
            popen_factory=FakeProcess,
        )
    for variable in ("DPU_BACKEND", "UPMEM_EXECUTION_MODE"):
        with pytest.raises(v4.V4Error, match=variable):
            session_v4.V4Session.start(
                ["fake"],
                session_root=tmp_path,
                profile=v4.V4Profile(dpu_count=1, rank_path="/dev/dpu_rank0"),
                environment={
                    "UPMEM_ALLOW_PHYSICAL_HARDWARE": "1",
                    variable: "simulator",
                },
                popen_factory=FakeProcess,
            )


def test_simulator_profile_environment_ready_and_response_contract(
    tmp_path: Path,
) -> None:
    profile = v4.V4Profile(
        dpu_count=1,
        execution_target=v4.EXECUTION_TARGET_SIMULATOR,
        timeout_s=0.2,
    )
    artifact = _artifact(tmp_path, dpu_count=1)
    session, process = _session(
        tmp_path,
        artifact,
        lambda _command: _valid_response(artifact, profile=profile),
        profile=profile,
    )
    assert process.stdin.commands == []
    assert session.startup["target_requested"] == "simulator"
    assert session.startup["target_observed"] == "sdk_simulator"
    assert session.startup["rank_path"] is None
    assert session.startup["hardware_allocation_verified"] is False
    response = session.submit(artifact)
    assert response["simulator_kernel_executed"] is True
    assert response["hardware_kernel_executed"] is False
    assert response["hardware_functionality_evidence"] is False
    assert response["simulator_functionality_evidence"] is True
    assert session.close().release_confirmed is True


def test_simulator_profile_rejects_crossed_target_labels(tmp_path: Path) -> None:
    profile = v4.V4Profile(
        dpu_count=1,
        execution_target=v4.EXECUTION_TARGET_SIMULATOR,
        timeout_s=0.2,
    )
    artifact = _artifact(tmp_path)
    session, _ = _session(
        tmp_path,
        artifact,
        lambda _command: _valid_response(artifact),
        profile=profile,
    )
    with pytest.raises(v4.V4ProtocolError, match="native identity"):
        session.submit(artifact)


def test_simulator_profile_rejects_crossed_ready_target_label(tmp_path: Path) -> None:
    simulator_profile = v4.V4Profile(
        dpu_count=1,
        execution_target=v4.EXECUTION_TARGET_SIMULATOR,
        timeout_s=0.2,
    )
    physical_profile = v4.V4Profile(
        dpu_count=1,
        rank_path="/dev/dpu_rank0",
        timeout_s=0.2,
    )

    with pytest.raises(v4.V4ProtocolError, match="READY field 'target_requested'"):
        session_v4.V4Session.start(
            ["fake"],
            session_root=tmp_path,
            profile=simulator_profile,
            environment={"DPU_BACKEND": "simulator"},
            popen_factory=lambda *_args, **_kwargs: FakeProcess(
                lambda _: None,
                profile=physical_profile,
            ),
        )


@pytest.mark.parametrize(
    "environment",
    [
        {"DPU_BACKEND": "hardware"},
        {"DPU_BACKEND": "simulator", "UPMEM_EXECUTION_MODE": "simulator"},
    ],
)
def test_simulator_profile_rejects_conflicting_backend_environment(
    tmp_path: Path, environment: dict[str, str]
) -> None:
    with pytest.raises(v4.V4Error, match="simulator_profile_violation"):
        session_v4.V4Session.start(
            ["fake"],
            session_root=tmp_path,
            profile=v4.V4Profile(
                dpu_count=1,
                execution_target=v4.EXECUTION_TARGET_SIMULATOR,
            ),
            environment=environment,
            popen_factory=lambda *_args, **_kwargs: FakeProcess(lambda _: None),
        )


def test_simulator_profile_forbids_rank_path() -> None:
    with pytest.raises(ValueError, match="forbid rank_path"):
        v4.V4Profile(
            dpu_count=1,
            execution_target=v4.EXECUTION_TARGET_SIMULATOR,
            rank_path="/dev/dpu_rank0",
        )


def test_simulator_profile_accepts_one_rank_multiple_dpus_with_native_limit() -> None:
    profile = v4.V4Profile(
        dpu_count=64,
        execution_target=v4.EXECUTION_TARGET_SIMULATOR,
    )
    assert profile.dpu_count == 64
    with pytest.raises(ValueError, match=r"\[1, 64\]"):
        v4.V4Profile(
            dpu_count=65,
            execution_target=v4.EXECUTION_TARGET_SIMULATOR,
        )


def test_ready_validation_failure_terminates_native_process(tmp_path: Path) -> None:
    profile = v4.V4Profile(
        dpu_count=1,
        rank_path="/dev/dpu_rank0",
        timeout_s=0.2,
    )
    process: FakeProcess | None = None

    def factory(*_args: object, **_kwargs: object) -> FakeProcess:
        nonlocal process
        process = FakeProcess(
            lambda _command: None,
            profile=profile,
            ready_overrides={"allocation_verified": False},
        )
        return process

    with pytest.raises(v4.V4ProtocolError, match="READY field"):
        session_v4.V4Session.start(
            ["fake-v4-host"],
            session_root=tmp_path,
            profile=profile,
            environment={"UPMEM_ALLOW_PHYSICAL_HARDWARE": "1"},
            popen_factory=factory,
        )
    assert process is not None
    assert process.poll() is not None


def test_successful_submit_and_release(tmp_path: Path) -> None:
    artifact = _artifact(tmp_path)
    session, process = _session(
        tmp_path, artifact, lambda command: _valid_response(artifact)
    )
    response = session.submit(artifact)
    assert response["global_completeness"] is False
    submit_timing = response["host_submit_timing"]
    assert set(submit_timing) == {
        "artifact_validation_s",
        "protocol_write_s",
        "response_wait_s",
        "response_validation_s",
        "total_submit_s",
    }
    assert all(value >= 0.0 for value in submit_timing.values())
    assert submit_timing["total_submit_s"] >= sum(
        value for field, value in submit_timing.items() if field != "total_submit_s"
    )
    assert process.stdin.commands == [
        f"SUBMIT requests/0000000000000000/manifest.txt {artifact.manifest_sha256}\n"
    ]
    release = session.close()
    assert release.release_confirmed is True
    assert release.event["event"] == "RELEASE"
    assert process.stdin.commands[-1] == "CLOSE\n"


def test_session_rejects_overlapping_submit_packed_submit_and_close(
    tmp_path: Path,
) -> None:
    operation, requests, profile = _packed_operation(tmp_path)
    artifact = _artifact(tmp_path, dpu_count=1)
    response_path = tmp_path / "results" / "operation_0000000000000000.jsonl"
    response_path.parent.mkdir(parents=True, exist_ok=True)
    response_path.write_text(
        "".join(
            json.dumps(_valid_response(request, profile=profile)) + "\n"
            for request in requests
        ),
        encoding="utf-8",
    )
    packed_response = {
        "event": "OPERATION_RESPONSE",
        "status": "completed",
        "operation_sequence": operation.operation_sequence,
        "response_path": "results/operation_0000000000000000.jsonl",
        "response_count": len(requests),
        "completed_request_count": len(requests),
        "failed_request_index": None,
        "response_sha256": hashlib.sha256(response_path.read_bytes()).hexdigest(),
    }
    submit_started = threading.Event()
    allow_response = threading.Event()
    reentrant_errors: list[v4.V4Error] = []

    def delayed_response(_command: str) -> dict[str, object]:
        for operation_call in (
            lambda: session.submit(artifact),
            lambda: session.submit_packed(operation),
            session.close,
        ):
            try:
                operation_call()
            except v4.V4Error as error:
                reentrant_errors.append(error)
            else:  # pragma: no cover - the guard must reject this call.
                raise AssertionError("reentrant operation unexpectedly executed")
        submit_started.set()
        assert allow_response.wait(timeout=1.0)
        return packed_response

    session, process = _session(
        tmp_path, requests[0], delayed_response, profile=profile
    )
    result: list[object] = []

    def submit() -> None:
        try:
            result.append(session.submit_packed(operation))
        except BaseException as exc:  # pragma: no cover - asserted below.
            result.append(exc)

    worker = threading.Thread(target=submit)
    worker.start()
    assert submit_started.wait(timeout=1.0)
    with pytest.raises(v4.V4Error, match="session_busy") as submit_error:
        session.submit(artifact)
    assert submit_error.value.failure_stage == "session_busy"
    with pytest.raises(v4.V4Error, match="session_busy") as packed_error:
        session.submit_packed(operation)
    assert packed_error.value.failure_stage == "session_busy"
    with pytest.raises(v4.V4Error, match="session_busy") as close_error:
        session.close()
    assert close_error.value.failure_stage == "session_busy"
    assert session._closed is False
    assert [error.failure_stage for error in reentrant_errors] == [
        "session_busy",
        "session_busy",
        "session_busy",
    ]
    assert process.stdin.commands == [
        f"SUBMIT_PACKED_OPERATION packed/operation.bin {operation.sha256}\n"
    ]

    allow_response.set()
    worker.join(timeout=1.0)
    assert not worker.is_alive()
    assert len(result) == 1
    assert isinstance(result[0], dict)
    release = session.close()
    assert release.release_confirmed is True
    assert session.close() is release


@pytest.mark.parametrize("failed_index", [0, 1])
def test_packed_failure_preserves_and_validates_partial_response(
    tmp_path: Path, failed_index: int
) -> None:
    operation, requests, profile = _packed_operation(tmp_path)
    response = _packed_failure_event(
        tmp_path, operation, requests, profile, failed_index
    )
    session, process = _session(
        tmp_path,
        requests[0],
        lambda _command: response,
        profile=profile,
    )

    with pytest.raises(v4.V4ProtocolError) as exc_info:
        session.submit_packed(operation)

    error = exc_info.value
    assert error.failure_stage == "request_execution_failed"
    assert error.operation_response == response
    assert error.response_path == response["response_path"]
    assert error.response_sha256 == response["response_sha256"]
    assert error.completed_request_count == failed_index
    assert error.failed_request_index == failed_index
    assert len(error.partial_responses) == failed_index + 1
    assert error.partial_responses[-1]["status"] == "failed"
    response_path = tmp_path / str(response["response_path"])
    assert response_path.is_file()
    assert hashlib.sha256(response_path.read_bytes()).hexdigest() == response[
        "response_sha256"
    ]
    assert process.stdin.commands == [
        f"SUBMIT_PACKED_OPERATION packed/operation.bin {operation.sha256}\n",
        "CLOSE\n",
    ]
    assert session.close().release_confirmed is True


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
    pump = session_v4._OutputPump(
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
    pump = session_v4._OutputPump(
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
    pump = session_v4._OutputPump(
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


def _wave_session(tmp_path, mutate=lambda event, path: None):
    from quantum_bench.upmem.packed_wave import WaveOperation, WaveTile
    from quantum_bench.upmem.wave_protocol import (
        FOUR_PRODUCT_PANEL, NO_PRODUCT, WaveCompletion, WaveControl, product_layout,
    )

    profile = v4.V4Profile(dpu_count=1, tasklets_per_dpu=8,
                           execution_target=v4.EXECUTION_TARGET_SIMULATOR,
                           request_transport=v4.REQUEST_TRANSPORT_PACKED_WAVE)
    control = WaveControl(0, 8, 0, 0, FOUR_PRODUCT_PANEL, 0, 0, 10, 0, 0,
                          1, 1, 1, 0, product_layout(1, 1, 1, numeric_mode=0,
                                                    kernel=FOUR_PRODUCT_PANEL))
    tile = WaveTile(control, 0, 0, (struct.pack('<f', 2.) + b'\0' * 4,) * 4)
    kwargs = dict(plan_sha256=b'p' * 32, dpu_binary_sha256=b'b' * 32, sequence=1,
                  operations=(WaveOperation(b'n' * 32, b'c' * 32, 1, 1, 1, 1, 1., 1.),),
                  waves=((tile,),))
    completion = WaveCompletion(1, 0, 0, 15, 0, 10, 0, 42, 4, 0, NO_PRODUCT)

    def respond(command):
        _, name, digest = command.split()
        payload = (tmp_path / name).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == digest
        result = completion.to_bytes() + struct.pack('<4f', 4., 4., 4., 4.)
        path = tmp_path / 'wave-result-00000000000000000001.bin'
        path.write_bytes(result)
        event = dict(event='WAVE_RESPONSE', status='completed', sequence=1,
                     completed_wave_count=1, launch_count=1, completed_result_count=1,
                     allocated_dpu_count=1, tasklets_per_dpu=8, cpu_fallback_used=False,
                     target_observed='sdk_simulator', envelope_bytes=len(payload),
                     native_snapshot_bytes=len(payload), operation_count=1, control_count=1,
                     h2d_bytes=144 + 32, d2h_bytes=72 + 32, input_payload_bytes=32,
                     response_path=path.name, response_sha256=hashlib.sha256(result).hexdigest(),
                     native_output_buffer_bytes=256 * 256 * 4)
        event.update({key: None for key in ('failure_stage', 'error', 'failed_wave_index',
                     'failed_dpu_id', 'failed_operation_index', 'failed_completion_mask',
                     'failed_completion_status', 'failed_completion_stage', 'failed_product')})
        event.update({key: 0.001 for key in ('h2d_time_s', 'kernel_time_s', 'd2h_time_s',
                                           'output_time_s', 'total_route_time_s')})
        mutate(event, path)
        return event

    process = FakeProcess(respond, profile=profile, ready_overrides={
        **v4.native_execution_identity(profile.execution_target, profile.request_transport),
        'dpu_binary_sha256': (b'b' * 32).hex(), 'initialization_binary_sha256': 'ab' * 32,
    })
    session = session_v4.V4Session.start(
        ('fake-native',), session_root=tmp_path, profile=profile,
        environment={}, popen_factory=lambda *a, **kw: process,
    )
    return session, process, kwargs


def test_wave_client_reuses_lifecycle_and_exposes_validated_results(tmp_path):
    session, process, kwargs = _wave_session(tmp_path)
    assert session.command[-1] == '--wave-v5'
    assert session.profile.to_dict()['profile'] == 'prepared_wave_v1'
    result = session.submit_waves(**kwargs)
    assert tuple(bytes(p) for p in result['results'][0][0]) == (struct.pack('<f', 4.),) * 4
    assert result['host_submit_timing']['total_submit_s'] >= 0
    assert (tmp_path / result['request_path']).stat().st_mode & 0o777 == 0o600
    with pytest.raises(ValueError, match='sequence'):
        session.submit_waves(**kwargs)
    with pytest.raises(ValueError, match='physical plan'):
        session.submit_waves(**{**kwargs, 'sequence': 2, 'plan_sha256': b'q' * 32})
    assert sum(cmd.startswith('SUBMIT') for cmd in process.stdin.commands) == 1
    assert session.close().release_confirmed
    assert session.close().release_confirmed
    assert process.stdin.commands.count('CLOSE\n') == 1


@pytest.mark.parametrize('field,value', [
    ('sequence', 2), ('completed_wave_count', 0), ('completed_result_count', 2),
    ('launch_count', 0), ('allocated_dpu_count', 4), ('tasklets_per_dpu', 3),
    ('cpu_fallback_used', 0), ('target_observed', 'physical_hardware'),
    ('envelope_bytes', 0), ('native_snapshot_bytes', 0), ('operation_count', 2),
    ('control_count', 2), ('input_payload_bytes', 0), ('h2d_bytes', 0), ('d2h_bytes', 0),
    ('kernel_time_s', float('nan')), ('output_time_s', -1),
    ('failed_dpu_id', 0), ('response_path', '../escape'), ('response_sha256', 'ab' * 32),
    ('event', 'RESPONSE'),
])
def test_wave_client_rejects_false_success_and_poison_session(tmp_path, field, value):
    session, process, kwargs = _wave_session(tmp_path, lambda e, p: e.update({field: value}))
    with pytest.raises(v4.V4ProtocolError) as caught:
        session.submit_waves(**kwargs)
    assert caught.value.wave_response[field] == value or field == 'kernel_time_s'
    assert caught.value.backend_facts['request_path'].startswith('wave-operation-')
    with pytest.raises(v4.V4Error, match='closed'):
        session.submit_waves(**{**kwargs, 'sequence': 2})
    assert sum(cmd.startswith('SUBMIT') for cmd in process.stdin.commands) == 1
    assert session.close().release_confirmed


@pytest.mark.parametrize('kind', ['truncated', 'extra', 'completion', 'symlink', 'fifo'])
def test_wave_client_reads_one_bounded_regular_result_snapshot(tmp_path, kind):
    def mutate(event, path):
        data = path.read_bytes()
        if kind == 'truncated':
            path.write_bytes(data[:-1])
        elif kind == 'extra':
            path.write_bytes(data + b'\0')
        elif kind == 'completion':
            path.write_bytes(data[:24] + struct.pack('<Q', 999) + data[32:])
            event['response_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        else:
            path.unlink()
            if kind == 'symlink':
                target = tmp_path / 'target'
                target.write_bytes(data)
                path.symlink_to(target)
            else:
                session_v4.os.mkfifo(path)
    session, _, kwargs = _wave_session(tmp_path, mutate)
    with pytest.raises(v4.V4ProtocolError):
        session.submit_waves(**kwargs)
    assert session.close().release_confirmed


def test_wave_client_preserves_partial_failure_without_retry(tmp_path):
    def mutate(event, path):
        event.update(status='failed', failure_stage='completion_validation_failed',
                     error='second wave fault', completed_wave_count=0,
                     completed_result_count=0, failed_wave_index=0,
                     failed_dpu_id=0, failed_operation_index=0, failed_product=2)
    session, process, kwargs = _wave_session(tmp_path, mutate)
    with pytest.raises(v4.V4ProtocolError, match='second wave fault') as caught:
        session.submit_waves(**kwargs)
    assert caught.value.wave_response['failed_product'] == 2
    assert (tmp_path / caught.value.wave_response['response_path']).is_file()
    assert process.stdin.commands[-1] == 'CLOSE\n'


def test_wave_client_reentry_and_preflight_reject_without_submission(tmp_path):
    session, process, kwargs = _wave_session(tmp_path)
    session._enter_operation()
    try:
        with pytest.raises(v4.V4Error, match='active operation'):
            session.submit_waves(**kwargs)
        with pytest.raises(v4.V4Error, match='active operation'):
            session.close()
    finally:
        session._leave_operation()
    with pytest.raises(ValueError, match='binary'):
        session.submit_waves(**{**kwargs, 'dpu_binary_sha256': b'x' * 32})
    with pytest.raises(ValueError, match='timeout'):
        session.submit_waves(**kwargs, timeout_s=float('nan'))
    assert not process.stdin.commands
    session.close()


def test_wave_client_timeout_poisoning_reuses_native_cleanup(tmp_path):
    session, process, kwargs = _wave_session(tmp_path)
    process.response_factory = lambda command: None
    with pytest.raises(v4.V4ProtocolError, match='timed out'):
        session.submit_waves(**kwargs, timeout_s=0.03)
    assert process.poll() is not None
    assert not session.close().release_confirmed
    assert sum(cmd.startswith('SUBMIT') for cmd in process.stdin.commands) == 1


@pytest.mark.parametrize('field,value', [
    ('abi', 'execution_plan_v4'), ('request_transport', 'packed_operation_v1'),
    ('kernel_identity', 'dpu_real_tile_v4_wram_panel_v1'),
    ('dpu_binary_sha256', 'missing'), ('initialization_binary_sha256', None),
])
def test_wave_ready_requires_explicit_protocol_and_binary_identity(tmp_path, field, value):
    profile = v4.V4Profile(dpu_count=1, execution_target=v4.EXECUTION_TARGET_SIMULATOR,
                           request_transport=v4.REQUEST_TRANSPORT_PACKED_WAVE)
    process = FakeProcess(lambda command: None, profile=profile, ready_overrides={
        **v4.native_execution_identity(profile.execution_target, profile.request_transport),
        'dpu_binary_sha256': 'ab' * 32, 'initialization_binary_sha256': 'bc' * 32,
        field: value,
    })
    with pytest.raises(v4.V4ProtocolError):
        session_v4.V4Session.start(('fake',), session_root=tmp_path, profile=profile,
                                  environment={}, popen_factory=lambda *a, **kw: process)
    assert process.poll() is not None
    assert process.stdin.commands == ['CLOSE\n']
