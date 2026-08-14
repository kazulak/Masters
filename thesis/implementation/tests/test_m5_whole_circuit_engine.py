from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import threading
import time
from typing import Any

import numpy as np
import pytest

from quantum_bench.core.indices import LABEL_LIST_EINSUM_SENTINEL
from quantum_bench.core.records import ContractionTask
from quantum_bench.targets.upmem.m5_whole_circuit_engine import (
    M5WholeCircuitEngine,
    M5WholeCircuitSession,
)
from quantum_bench.targets.upmem.execution_plan_v4 import V4ProtocolError
from quantum_bench.whole_circuit.core import DeviceTopology
from quantum_bench.whole_circuit.policies import Float32RealPolicy, HostPackedInt8Policy


def _task(k: int = 5, *, m: int = 3, n: int = 4) -> ContractionTask:
    return ContractionTask(
        id="fixture",
        input_tensor_ids=("left", "right"),
        output_tensor_id="out",
        dependencies=(),
        index_expression=f"{LABEL_LIST_EINSUM_SENTINEL}:fixture",
        input_shapes=((m, k), (k, n)),
        output_shape=(m, n),
        left_labels=(0, 1),
        right_labels=(1, 2),
        contracted_labels=(1,),
        output_labels=(0, 2),
        gemm_m=m,
        gemm_k=k,
        gemm_n=n,
        structure="dense",
        estimated_flops=0,
        estimated_bytes=0,
    )


def _binaries(root: Path) -> tuple[Path, Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    paths = tuple(root / name for name in ("host", "dpu", "init"))
    for path in paths:
        path.write_bytes(path.name.encode("ascii"))
    paths[0].chmod(paths[0].stat().st_mode | 0o100)
    return paths


@dataclass
class _Release:
    release_confirmed: bool = True
    stdout: str = ""
    stderr: str = ""
    stdout_truncated: bool = False
    stderr_truncated: bool = False
    stdout_total_bytes: int = 0
    stderr_total_bytes: int = 0
    stdout_limit_exceeded: bool = False
    stderr_limit_exceeded: bool = False
    event: dict[str, Any] = field(default_factory=dict)


@dataclass
class _FakeSession:
    profile: Any
    binary_provenance: dict[str, str] = field(default_factory=dict)
    startup: dict[str, Any] = field(default_factory=dict)
    delay_s: float = 0.0
    fail_submit: bool = False
    release_confirmed: bool = True
    stdout: str = ""
    stderr: str = ""
    stdout_truncated: bool = False
    stderr_truncated: bool = False
    stdout_total_bytes: int = 0
    stderr_total_bytes: int = 0
    stdout_limit_exceeded: bool = False
    stderr_limit_exceeded: bool = False
    returncode: int | None = 0
    closed: bool = False
    submissions: list[Any] = field(default_factory=list)
    submitted_timeouts: list[float] = field(default_factory=list)
    barrier: Any = None

    def __post_init__(self) -> None:
        self.startup = {
            "event": "READY",
            "status": "ready",
            "target_observed": "physical_hardware",
            "requested_dpu_count": self.profile.dpu_count,
            "allocated_dpu_count": self.profile.dpu_count,
            "tasklets_per_dpu": self.profile.tasklets_per_dpu,
            "hardware_allocation_verified": True,
            **self.binary_provenance,
        }

    def submit(
        self, artifact: Any, *, timeout_s: float | None = None
    ) -> dict[str, Any]:
        if timeout_s is not None:
            self.submitted_timeouts.append(float(timeout_s))
        self.submissions.append(artifact)
        if self.fail_submit:
            raise RuntimeError("submission failed")
        if self.barrier is not None:
            self.barrier.enter()
        if timeout_s is not None and self.delay_s > timeout_s:
            time.sleep(max(0.0, timeout_s))
            raise V4ProtocolError(
                "kernel_timeout", "fake request exceeded the graph deadline"
            )
        time.sleep(self.delay_s)
        for record in artifact.work_units:
            if record.flags:
                continue
            a_dtype = (
                np.int8
                if self.profile.numeric_mode_name == "host_packed_int8"
                else np.dtype("<f4")
            )
            a = np.fromfile(
                artifact.root / record.a_path,
                dtype=a_dtype,
                count=record.m_elements * record.k_elements,
            ).reshape(record.m_elements, record.k_elements)
            b = np.fromfile(
                artifact.root / record.b_path,
                dtype=a_dtype,
                count=record.k_elements * record.n_elements,
            ).reshape(record.k_elements, record.n_elements)
            output = (
                a.astype(np.int64) @ b.astype(np.int64) if a_dtype == np.int8 else a @ b
            )
            (artifact.root / record.c_path).write_bytes(
                np.asarray(
                    output, dtype="<i4" if a_dtype == np.int8 else "<f4"
                ).tobytes()
            )
        return {
            "status": "completed",
            "target_observed": "physical_hardware",
            "native_kernel_executed": True,
            "hardware_kernel_executed": True,
            "simulator_kernel_executed": False,
            "cpu_fallback_used": False,
            "hardware_allocation_verified": True,
            "allocated_dpu_count": self.profile.dpu_count,
            "requested_dpu_count": self.profile.dpu_count,
            "tasklets_per_dpu": self.profile.tasklets_per_dpu,
            "request_sequence": artifact.request_sequence,
            "bulk_set_launch_verified": True,
            "transfer": {"h2d_bytes": 10, "d2h_bytes": 5, "total_bytes": 15},
            "timing": {"h2d_time_s": 0.01, "launch_time_s": 0.02, "d2h_time_s": 0.01},
        }

    def close(self, *, timeout_s: float | None = None) -> _Release:
        del timeout_s
        self.closed = True
        return _Release(
            release_confirmed=self.release_confirmed,
            stdout=self.stdout,
            stderr=self.stderr,
            stdout_truncated=self.stdout_truncated,
            stderr_truncated=self.stderr_truncated,
            stdout_total_bytes=self.stdout_total_bytes,
            stderr_total_bytes=self.stderr_total_bytes,
            stdout_limit_exceeded=self.stdout_limit_exceeded,
            stderr_limit_exceeded=self.stderr_limit_exceeded,
            event={"returncode": self.returncode},
        )


def _engine(
    tmp_path: Path,
    *,
    ranks: int = 1,
    dpu_count: int = 2,
    sessions: list[_FakeSession] | None = None,
    delay_s: float = 0.0,
    barrier: Any = None,
    timeout_s: float = 60.0,
    startup_delay_s: float = 0.0,
    startup_delays_s: tuple[float, ...] = (),
) -> M5WholeCircuitEngine:
    created: list[_FakeSession] = sessions if sessions is not None else []
    host_binary, dpu_binary, initialization_binary = _binaries(tmp_path / "binaries")
    binary_provenance = {
        "host_binary_sha256": hashlib.sha256(host_binary.read_bytes()).hexdigest(),
        "dpu_binary_sha256": hashlib.sha256(dpu_binary.read_bytes()).hexdigest(),
        "initialization_binary_sha256": hashlib.sha256(
            initialization_binary.read_bytes()
        ).hexdigest(),
    }

    def factory(command: Any, *, session_root: Path, profile: Any) -> _FakeSession:
        del command, session_root
        session_index = len(created)
        session = _FakeSession(
            profile=profile,
            binary_provenance=binary_provenance,
            delay_s=delay_s,
            barrier=barrier,
        )
        created.append(session)
        delay = (
            startup_delays_s[session_index]
            if session_index < len(startup_delays_s)
            else startup_delay_s
        )
        time.sleep(delay)
        return session

    return M5WholeCircuitEngine(
        session_root=tmp_path,
        host_binary=host_binary,
        dpu_binary=dpu_binary,
        initialization_binary=initialization_binary,
        rank_paths=tuple(f"/dev/dpu_rank{i}" for i in range(ranks)),
        dpu_count=dpu_count,
        timeout_s=timeout_s,
        session_factory=factory,
    )


def _topology(count: int) -> DeviceTopology:
    return DeviceTopology(
        backend="upmem", device_ids=tuple(f"dpu{i}" for i in range(count))
    )


@dataclass
class _SubmissionBarrier:
    participants: int
    entered: int = 0
    both_entered: threading.Event = field(default_factory=threading.Event)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def enter(self) -> None:
        with self.lock:
            self.entered += 1
            if self.entered == self.participants:
                self.both_entered.set()
        if not self.both_entered.wait(timeout=2.0):
            raise AssertionError("rank submissions were serialized")


def test_float_and_packed_int8_match_reference_across_k_chunks(tmp_path: Path) -> None:
    left = np.arange(900, dtype=np.float32).reshape(3, 300) / 7
    right = np.arange(1200, dtype=np.float32).reshape(300, 4) / 5
    engine = _engine(tmp_path)
    task = _task(300)
    float_session = engine.open_session(Float32RealPolicy(), _topology(2))
    float_result = float_session.execute(task, left, right)
    np.testing.assert_allclose(float_result.output, left @ right, rtol=1e-6, atol=1e-6)
    assert float_result.metadata["k_chunk_count"] == 2
    assert len(float_result.metadata["request_manifest_hashes"]) == 2
    assert float_result.metadata["application_visible_transfer_bytes"] == 30
    assert (
        float_result.metadata["application_visible_transfer_bytes"]
        == float_result.metadata["application_visible_h2d_bytes"]
        + float_result.metadata["application_visible_d2h_bytes"]
    )
    assert float_session.close()["hardware_release_confirmed"]
    packed_session = engine.open_session(HostPackedInt8Policy(), _topology(2))
    packed_result = packed_session.execute(task, left, right)
    expected, _ = HostPackedInt8Policy().contract(task, left, right)
    np.testing.assert_allclose(packed_result.output, expected, rtol=0, atol=1e-5)
    assert packed_result.metadata["packed_int8_transport"]
    assert packed_result.metadata["host_quantization_time_s"] >= 0.0
    assert packed_result.metadata["host_dequantization_time_s"] >= 0.0
    assert packed_result.metadata["graph_intermediate_placement"] == "host_managed"
    assert packed_result.metadata["profile"] == "m5_whole_circuit_v4_v1"
    assert packed_result.metadata["abi"] == "execution_plan_v4"
    assert packed_result.metadata["session"] == "persistent_rank_session_v1"
    assert packed_result.metadata["dispatch"] == "bulk_set_synchronous_v1"
    assert packed_result.metadata["kernel"] == "dpu_gemm_tile_v4"
    assert (
        packed_result.metadata["transfer_accounting_scope"]
        == "application_visible_sdk_recorded"
    )
    assert (
        float_result.metadata["task_structure_sha256"]
        == packed_result.metadata["task_structure_sha256"]
    )
    assert (
        float_result.metadata["request_contract_sha256"]
        != packed_result.metadata["request_contract_sha256"]
    )


def test_request_contract_is_deterministic_and_binds_int8_data(tmp_path: Path) -> None:
    task = _task(5)
    left = np.arange(15, dtype=np.float32).reshape(3, 5)
    right = np.arange(20, dtype=np.float32).reshape(5, 4)
    session = _engine(tmp_path, dpu_count=1).open_session(
        HostPackedInt8Policy(), _topology(1)
    )
    first = session.execute(task, left, right)
    second = session.execute(task, left, right)
    changed = left.copy()
    changed[0, 0] = 100.0
    third = session.execute(task, changed, right)
    assert (
        first.metadata["task_structure_sha256"]
        == second.metadata["task_structure_sha256"]
        == third.metadata["task_structure_sha256"]
    )
    assert (
        first.metadata["request_contract_sha256"]
        == second.metadata["request_contract_sha256"]
    )
    assert (
        first.metadata["request_contract_sha256"]
        != third.metadata["request_contract_sha256"]
    )
    assert first.metadata["left_scale"] != third.metadata["left_scale"]
    session.close()


def test_binary_hashes_and_roots_are_provenance_only(tmp_path: Path) -> None:
    engine = _engine(tmp_path, dpu_count=1)
    session = engine.open_session(Float32RealPolicy(), _topology(1))
    result = session.execute(
        _task(), np.ones((3, 5), dtype=np.float32), np.ones((5, 4), dtype=np.float32)
    )
    terminal = session.close()
    for metadata in (result.metadata, terminal):
        assert metadata["source_root"].endswith("/thesis/implementation")
        assert metadata["session_root"] == str(tmp_path.resolve())
        for label in ("host_binary", "dpu_binary", "initialization_binary"):
            path = Path(metadata[f"{label}_path"])
            assert path.is_absolute()
            assert (
                metadata[f"{label}_sha256"]
                == hashlib.sha256(path.read_bytes()).hexdigest()
            )
    assert "source_root" not in result.metadata.get("executor_config", {})
    assert "session_root" not in result.metadata.get("executor_config", {})


def test_ready_binary_hash_mismatch_denies_admission(tmp_path: Path) -> None:
    session = _engine(tmp_path, dpu_count=1).open_session(
        Float32RealPolicy(), _topology(1)
    )
    session.execute(
        _task(), np.ones((3, 5), dtype=np.float32), np.ones((5, 4), dtype=np.float32)
    )
    session.ranks[0].session.startup["dpu_binary_sha256"] = "0" * 64
    terminal = session.close()
    assert terminal["target_observed"] == "not_verified"
    assert terminal["hardware_allocation_verified"] is True
    assert terminal["binary_identity_verified"] is False
    assert terminal["native_kernel_executed"] is True
    assert terminal["hardware_kernel_executed"] is True


def test_close_retains_bounded_rank_diagnostics_on_success(tmp_path: Path) -> None:
    session = _engine(tmp_path, dpu_count=1).open_session(
        Float32RealPolicy(), _topology(1)
    )
    native = session.ranks[0].session
    native.stdout = "native stdout\n"
    native.stderr = "native stderr\n"
    native.stdout_truncated = True
    native.stdout_total_bytes = 100_000
    native.returncode = 0
    terminal = session.close()
    assert terminal["native_diagnostics"] == [
        {
            "rank_index": 0,
            "rank_path": "/dev/dpu_rank0",
            "stdout": "native stdout\n",
            "stderr": "native stderr\n",
            "stdout_truncated": True,
            "stderr_truncated": False,
            "stdout_total_bytes": 100_000,
            "stderr_total_bytes": 0,
            "stdout_limit_exceeded": False,
            "stderr_limit_exceeded": False,
            "returncode": 0,
            "release_confirmed": True,
        }
    ]
    json.dumps(terminal)


def test_close_retains_rank_diagnostics_on_release_failure(tmp_path: Path) -> None:
    session = _engine(tmp_path, dpu_count=1).open_session(
        Float32RealPolicy(), _topology(1)
    )
    native = session.ranks[0].session
    native.release_confirmed = False
    native.stdout = "release stdout\n"
    native.stderr = "release stderr\n"
    native.returncode = 7
    terminal = session.close()
    assert terminal["native_diagnostics"][0] == {
        "rank_index": 0,
        "rank_path": "/dev/dpu_rank0",
        "stdout": "release stdout\n",
        "stderr": "release stderr\n",
        "stdout_truncated": False,
        "stderr_truncated": False,
        "stdout_total_bytes": 0,
        "stderr_total_bytes": 0,
        "stdout_limit_exceeded": False,
        "stderr_limit_exceeded": False,
        "returncode": 7,
        "release_confirmed": False,
    }
    assert terminal["failure_stage"] == "hardware_release_failed"


def test_release_failure_preserves_execution_event_semantics(tmp_path: Path) -> None:
    session = _engine(tmp_path, dpu_count=1).open_session(
        Float32RealPolicy(), _topology(1)
    )
    session.execute(
        _task(), np.ones((3, 5), dtype=np.float32), np.ones((5, 4), dtype=np.float32)
    )
    session.ranks[0].session.release_confirmed = False
    terminal = session.close()
    assert terminal["target_observed"] == "not_verified"
    assert terminal["native_kernel_executed"] is True
    assert terminal["hardware_kernel_executed"] is True
    assert terminal["failure_stage"] == "hardware_release_failed"


def test_missing_or_non_executable_binary_fails_before_session_allocation(
    tmp_path: Path,
) -> None:
    host_binary, dpu_binary, initialization_binary = _binaries(tmp_path / "binaries")
    host_binary.chmod(host_binary.stat().st_mode & ~0o111)
    with pytest.raises(ValueError, match="not executable"):
        M5WholeCircuitEngine(
            session_root=tmp_path / "session",
            host_binary=host_binary,
            dpu_binary=dpu_binary,
            initialization_binary=initialization_binary,
            rank_paths=("/dev/dpu_rank0",),
            dpu_count=1,
        )


def test_int8_uses_one_full_operand_scale_not_tile_scales(tmp_path: Path) -> None:
    engine = _engine(tmp_path, dpu_count=1)
    session = engine.open_session(HostPackedInt8Policy(), _topology(1))
    task = _task(300)
    left = np.ones((3, 300), dtype=np.float32)
    left[:, 299] = 100
    result = session.execute(task, left, np.ones((300, 4), dtype=np.float32))
    assert result.metadata["left_scale"] == pytest.approx(100 / 127)
    assert result.metadata["k_chunk_count"] == 2


def test_topology_and_startup_failures_are_closed(tmp_path: Path) -> None:
    engine = _engine(tmp_path, ranks=2, dpu_count=2)
    with pytest.raises(ValueError, match="device count"):
        engine.open_session(Float32RealPolicy(), _topology(1))
    closed: list[_FakeSession] = []
    calls = 0

    def factory(command: Any, *, session_root: Path, profile: Any) -> _FakeSession:
        nonlocal calls
        del command, session_root
        calls += 1
        if calls == 2:
            raise RuntimeError("second rank failed")
        session = _FakeSession(profile=profile)
        closed.append(session)
        return session

    host_binary, dpu_binary, initialization_binary = _binaries(
        tmp_path / "failure" / "binaries"
    )
    failing = M5WholeCircuitEngine(
        session_root=tmp_path / "failure",
        host_binary=host_binary,
        dpu_binary=dpu_binary,
        initialization_binary=initialization_binary,
        rank_paths=("/dev/dpu_rank0", "/dev/dpu_rank1"),
        dpu_count=2,
        session_factory=factory,
    )
    with pytest.raises(RuntimeError, match="second rank"):
        failing.open_session(Float32RealPolicy(), _topology(2))
    assert closed[0].closed


def test_rank_submissions_are_concurrent_and_fail_closed(tmp_path: Path) -> None:
    engine = _engine(
        tmp_path,
        ranks=2,
        dpu_count=2,
        barrier=_SubmissionBarrier(participants=2),
    )
    session = engine.open_session(Float32RealPolicy(), _topology(2))
    session.execute(
        _task(m=300),
        np.ones((300, 5), dtype=np.float32),
        np.ones((5, 4), dtype=np.float32),
    )
    assert session.close()["hardware_release_confirmed"]
    sessions: list[_FakeSession] = []
    bad = _engine(tmp_path / "bad", dpu_count=1, sessions=sessions)
    failing_session = bad.open_session(Float32RealPolicy(), _topology(1))
    failing_session.ranks[0].session.fail_submit = True
    with pytest.raises(RuntimeError, match="submission failed"):
        failing_session.execute(
            _task(),
            np.ones((3, 5), dtype=np.float32),
            np.ones((5, 4), dtype=np.float32),
        )
    terminal = failing_session.close()
    assert terminal["cpu_fallback_used"] is False
    assert terminal["hardware_kernel_executed"] is False
    assert terminal["failure_stage"] == "hardware_task_execution_failed"


def test_session_uses_one_deadline_across_multiple_task_requests(
    tmp_path: Path,
) -> None:
    sessions: list[_FakeSession] = []
    engine = _engine(
        tmp_path,
        dpu_count=1,
        sessions=sessions,
        delay_s=0.03,
        timeout_s=0.05,
    )
    session = engine.open_session(Float32RealPolicy(), _topology(1))
    left = np.ones((3, 5), dtype=np.float32)
    right = np.ones((5, 4), dtype=np.float32)
    session.execute(_task(), left, right)
    with pytest.raises(V4ProtocolError, match="kernel_timeout"):
        session.execute(_task(), left, right)
    assert len(sessions[0].submitted_timeouts) == 2
    assert sessions[0].submitted_timeouts[1] < sessions[0].submitted_timeouts[0]
    sessions[0].release_confirmed = False
    terminal = session.close()
    assert terminal["failure_stage"] == "kernel_timeout"
    assert terminal["primary_failure_stage"] == "kernel_timeout"
    assert terminal["release_failure_stage"] == "hardware_release_failed"
    assert terminal["native_kernel_executed"] is True


def test_rank_startup_is_inside_the_whole_graph_deadline(tmp_path: Path) -> None:
    sessions: list[_FakeSession] = []
    engine = _engine(
        tmp_path,
        dpu_count=1,
        sessions=sessions,
        timeout_s=0.01,
        startup_delay_s=0.02,
    )
    with pytest.raises(V4ProtocolError, match="kernel_timeout"):
        engine.open_session(Float32RealPolicy(), _topology(1))
    assert sessions and sessions[0].closed


def test_late_second_rank_releases_every_opened_rank(tmp_path: Path) -> None:
    sessions: list[_FakeSession] = []
    engine = _engine(
        tmp_path,
        ranks=2,
        dpu_count=2,
        sessions=sessions,
        timeout_s=0.02,
        startup_delays_s=(0.0, 0.03),
    )
    with pytest.raises(V4ProtocolError, match="kernel_timeout"):
        engine.open_session(Float32RealPolicy(), _topology(2))
    assert len(sessions) == 2
    assert all(session.closed for session in sessions)


def test_request_cleanup_and_release_are_required(tmp_path: Path) -> None:
    engine = _engine(tmp_path, dpu_count=1)
    session = engine.open_session(Float32RealPolicy(), _topology(1))
    session.execute(
        _task(), np.ones((3, 5), dtype=np.float32), np.ones((5, 4), dtype=np.float32)
    )
    assert not list((tmp_path / "rank_00" / "requests").iterdir())
    assert session.close()["target_observed"] == "physical_hardware"
    failed_release = _engine(tmp_path / "failed-release", dpu_count=1)
    failed_session = failed_release.open_session(Float32RealPolicy(), _topology(1))
    failed_session.ranks[0].session.release_confirmed = False
    assert failed_session.close()["target_observed"] == "not_verified"


def test_close_without_submitted_request_cannot_claim_hardware_execution(
    tmp_path: Path,
) -> None:
    session = _engine(tmp_path, dpu_count=1).open_session(
        Float32RealPolicy(), _topology(1)
    )
    terminal = session.close()
    assert terminal["target_observed"] == "not_verified"
    assert terminal["native_kernel_executed"] is False
    assert terminal["hardware_kernel_executed"] is False
    assert terminal["successful_request_count"] == 0
    assert terminal["allocated_dpu_count"] == 1
    assert terminal["hardware_release_verified"] is True
    assert terminal["observed_rank_count"] == 1
    assert terminal["observed_tasklets_per_dpu"] == 1


def test_cleanup_rejects_requests_root(tmp_path: Path) -> None:
    requests_root = tmp_path / "requests"
    requests_root.mkdir()
    artifact = type(
        "Artifact",
        (),
        {
            "root": tmp_path,
            "request_dir": requests_root,
            "manifest_path": requests_root / "manifest.json",
            "sidecar_path": requests_root / "sidecar.bin",
        },
    )()
    with pytest.raises(RuntimeError, match="requests root"):
        M5WholeCircuitSession._delete_request_dir(artifact)


def test_engine_supports_batched_permuted_output_labels(tmp_path: Path) -> None:
    task = ContractionTask(
        id="batched",
        input_tensor_ids=("left", "right"),
        output_tensor_id="out",
        dependencies=(),
        index_expression="bmk,bkn->nbm",
        input_shapes=((2, 3, 4), (2, 4, 5)),
        output_shape=(5, 2, 3),
        left_labels=(5, 0, 1),
        right_labels=(5, 1, 2),
        contracted_labels=(1,),
        output_labels=(2, 5, 0),
        gemm_m=1,
        gemm_k=1,
        gemm_n=1,
        structure="dense",
        estimated_flops=0,
        estimated_bytes=0,
    )
    left = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    right = np.arange(40, dtype=np.float32).reshape(2, 4, 5)
    result = (
        _engine(tmp_path, dpu_count=2)
        .open_session(Float32RealPolicy(), _topology(2))
        .execute(task, left, right)
    )
    expected = np.einsum("bmk,bkn->nbm", left, right)
    np.testing.assert_allclose(result.output, expected, rtol=1e-6, atol=1e-6)
