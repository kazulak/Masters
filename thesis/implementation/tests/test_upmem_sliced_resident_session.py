from __future__ import annotations

from dataclasses import FrozenInstanceError
import hashlib
import json
import subprocess
from pathlib import Path

import pytest

import quantum_bench.targets.upmem.hardware_sliced_resident_session as session


ROOT = Path(__file__).resolve().parents[1]


def _profile_mapping(**updates: object) -> dict[str, object]:
    profile = {
        "hardware_profile_version": session.PROFILE_VERSION,
        "target": "hardware",
        "backend_id": session.BACKEND_ID,
        "route_id": session.ROUTE_ID,
        "requested_dpu_count": 2,
        "slices": 2,
        "tasklets_per_dpu": 1,
        "numeric_modes": ["none"],
        "synchronous_execution": True,
        "timeout_s": 1.0,
        "performance_claim_applicable": False,
    }
    profile.update(updates)
    return profile


def _completed_response() -> dict[str, object]:
    return {
        "status": "completed",
        "failure_stage": None,
        "hardware_execution": True,
        "cpu_fallback_used": False,
        "allocation": {"requested_dpus": 2, "allocated_dpus": 2, "verified": True},
        "launch": {"mode": "asynchronous", "completed": True},
        "release": {"confirmed": True},
    }


def _fake_build(monkeypatch) -> None:
    monkeypatch.setattr(
        session,
        "_required_build_tools",
        lambda environment: ("make", {"make": "make", "dpu-pkg-config": "fake"}),
    )

    def run(command, *, cwd, env, timeout_s):
        (cwd / "bin").mkdir(parents=True)
        (cwd / "bin" / "host_two_dpu").write_bytes(b"host")
        (cwd / "bin" / "dpu_resident_two_dpu").write_bytes(b"dpu")
        return {
            "returncode": 0,
            "timed_out": False,
            "stdout_snippet": "",
            "stderr_snippet": "",
        }

    monkeypatch.setattr(session, "_run_build_command", run)


def _build(tmp_path: Path, monkeypatch) -> session.HardwareSessionBuild:
    _fake_build(monkeypatch)
    return session.build_sliced_resident_hardware_session(
        ROOT, tmp_path / "session", profile=_profile_mapping(), environment={}
    )


def _paths(build: session.HardwareSessionBuild) -> tuple[Path, Path, Path]:
    first = build.session_root / "slice-0.json"
    second = build.session_root / "slice-1.json"
    response = build.session_root / "response.json"
    first.write_text("{}", encoding="utf-8")
    second.write_text("{}", encoding="utf-8")
    return first, second, response


def _install_host(
    monkeypatch, response_path: Path, payload: object, *, returncode: int = 0
):
    calls = []

    class Process:
        pid = 123

        def __init__(self) -> None:
            self.returncode = returncode

        def communicate(self, timeout):
            response_path.write_text(json.dumps(payload), encoding="utf-8")
            return "host stdout", "host stderr"

    def popen(command, **kwargs):
        calls.append((command, kwargs))
        return Process()

    monkeypatch.setattr(session.subprocess, "Popen", popen)
    return calls


def test_profile_parser_supplies_frozen_canonical_build_caps() -> None:
    profile = session.parse_sliced_resident_hardware_profile(_profile_mapping())

    assert profile.requested_dpu_count == profile.slices == 2
    assert profile.tasklets_per_dpu == 1
    assert profile.numeric_mode == "none"
    assert profile.max_rank == session.RESIDENT_MAX_RANK
    assert profile.max_tensor_elements == session.RESIDENT_MAX_ELEMENTS
    with pytest.raises(FrozenInstanceError):
        profile.timeout_s = 2.0


@pytest.mark.parametrize(
    "updates",
    (
        {"requested_dpu_count": 1},
        {"tasklets_per_dpu": 2},
        {"numeric_modes": ["per_task_resident_requantize"]},
        {"max_rank": 17},
    ),
)
def test_profile_parser_rejects_noncanonical_hardware_shape(updates) -> None:
    with pytest.raises(ValueError, match="hardware_profile_violation"):
        session.parse_sliced_resident_hardware_profile(_profile_mapping(**updates))


def test_build_snapshots_both_native_siblings_and_hashes_binaries(
    tmp_path, monkeypatch
) -> None:
    build = _build(tmp_path, monkeypatch)

    snapshot = build.source_snapshot
    assert (snapshot / session.TWO_DPU_SOURCE_DIR / "Makefile").is_file()
    assert (snapshot / session.RESIDENT_SOURCE_DIR / "dpu.c").is_file()
    assert "../upmem_sdk_generic_loop_resident/dpu.c" in (
        snapshot / session.TWO_DPU_SOURCE_DIR / "Makefile"
    ).read_text(encoding="utf-8")
    assert build.source_tree_hash == session._hash_tree(snapshot)
    assert build.host_binary_hash == hashlib.sha256(b"host").hexdigest()
    assert build.dpu_binary_hash == hashlib.sha256(b"dpu").hexdigest()


def test_build_command_is_one_tasklet_physical_prepare_only(
    tmp_path, monkeypatch
) -> None:
    calls = []
    _fake_build(monkeypatch)

    def run(command, *, cwd, env, timeout_s):
        calls.append((command, cwd, env))
        (cwd / "bin").mkdir(parents=True)
        (cwd / "bin" / "host_two_dpu").write_bytes(b"host")
        (cwd / "bin" / "dpu_resident_two_dpu").write_bytes(b"dpu")
        return {
            "returncode": 0,
            "timed_out": False,
            "stdout_snippet": "",
            "stderr_snippet": "",
        }

    monkeypatch.setattr(session, "_run_build_command", run)
    build = session.build_sliced_resident_hardware_session(
        ROOT,
        tmp_path / "session",
        profile=_profile_mapping(),
        environment={"DPU_BACKEND": "simulator"},
    )

    command, cwd, env = calls[0]
    assert command == build.build_command
    assert "NR_TASKLETS=1" in command
    assert "UPMEM_GENERIC_HARDWARE_MVP=1" in command
    assert cwd == build.build_dir
    assert "DPU_BACKEND" not in env


def test_execute_requires_explicit_physical_opt_in(tmp_path, monkeypatch) -> None:
    build = _build(tmp_path, monkeypatch)
    first, second, response = _paths(build)
    monkeypatch.setattr(session.subprocess, "Popen", pytest.fail)

    with pytest.raises(ValueError, match="UPMEM_ALLOW_PHYSICAL_HARDWARE=1"):
        session.execute_sliced_resident_hardware_session(
            build,
            manifest_paths=(first, second),
            response_path=response,
            profile=_profile_mapping(),
            environment={},
        )


@pytest.mark.parametrize(
    "environment",
    (
        {"UPMEM_ALLOW_PHYSICAL_HARDWARE": "1", "DPU_BACKEND": "simulator"},
        {"UPMEM_ALLOW_PHYSICAL_HARDWARE": "1", "UPMEM_PROFILE": "backend=simulator"},
    ),
)
def test_execute_rejects_simulator_selectors(
    tmp_path, monkeypatch, environment
) -> None:
    build = _build(tmp_path, monkeypatch)
    first, second, response = _paths(build)
    monkeypatch.setattr(session.subprocess, "Popen", pytest.fail)

    with pytest.raises(ValueError, match="(DPU_BACKEND|selects a simulator)"):
        session.execute_sliced_resident_hardware_session(
            build,
            manifest_paths=(first, second),
            response_path=response,
            profile=_profile_mapping(),
            environment=environment,
        )


def test_execute_constructs_host_command_and_admits_complete_response(
    tmp_path, monkeypatch
) -> None:
    build = _build(tmp_path, monkeypatch)
    first, second, response = _paths(build)
    calls = _install_host(monkeypatch, response, _completed_response())
    result = session.execute_sliced_resident_hardware_session(
        build,
        manifest_paths=(first, second),
        response_path=response,
        profile=_profile_mapping(),
        environment={"UPMEM_ALLOW_PHYSICAL_HARDWARE": "1"},
    )

    assert result.status == "completed"
    assert result.response == _completed_response()
    assert result.command == (
        str(build.host_binary.resolve()),
        "--slice-package-0",
        str(first.resolve()),
        "--slice-package-1",
        str(second.resolve()),
        "--resident-response",
        str(response.resolve()),
    )
    assert calls == [
        (
            result.command,
            {
                "cwd": build.build_dir,
                "env": {"UPMEM_ALLOW_PHYSICAL_HARDWARE": "1"},
                "stdout": subprocess.PIPE,
                "stderr": subprocess.PIPE,
                "text": True,
                "start_new_session": True,
            },
        )
    ]


def test_execute_rejects_empty_response_even_when_host_succeeds(
    tmp_path, monkeypatch
) -> None:
    build = _build(tmp_path, monkeypatch)
    first, second, response = _paths(build)
    _install_host(monkeypatch, response, {})

    result = session.execute_sliced_resident_hardware_session(
        build,
        manifest_paths=(first, second),
        response_path=response,
        profile=_profile_mapping(),
        environment={"UPMEM_ALLOW_PHYSICAL_HARDWARE": "1"},
    )

    assert result.status == "failed"
    assert result.failure_stage == "response_evidence_invalid"


def test_execute_reports_native_failed_response(tmp_path, monkeypatch) -> None:
    build = _build(tmp_path, monkeypatch)
    first, second, response = _paths(build)
    _install_host(
        monkeypatch,
        response,
        {"status": "failed", "failure_stage": "hardware_allocation_failed"},
        returncode=1,
    )

    result = session.execute_sliced_resident_hardware_session(
        build,
        manifest_paths=(first, second),
        response_path=response,
        profile=_profile_mapping(),
        environment={"UPMEM_ALLOW_PHYSICAL_HARDWARE": "1"},
    )

    assert result.status == "failed"
    assert result.failure_stage == "hardware_allocation_failed"


def test_execute_timeout_keeps_hardware_cleanup_unverified(
    tmp_path, monkeypatch
) -> None:
    build = _build(tmp_path, monkeypatch)
    first, second, response = _paths(build)
    starts, signals = [], []

    class Process:
        pid = 456
        returncode = -15
        calls = 0

        def communicate(self, timeout):
            self.calls += 1
            if self.calls == 1:
                raise subprocess.TimeoutExpired(
                    "host", timeout, output="partial", stderr="late"
                )
            return "", ""

    process = Process()
    monkeypatch.setattr(
        session.subprocess,
        "Popen",
        lambda *args, **kwargs: starts.append((args, kwargs)) or process,
    )
    monkeypatch.setattr(
        session.os, "killpg", lambda pid, signum: signals.append((pid, signum))
    )
    result = session.execute_sliced_resident_hardware_session(
        build,
        manifest_paths=(first, second),
        response_path=response,
        profile=_profile_mapping(),
        environment={"UPMEM_ALLOW_PHYSICAL_HARDWARE": "1"},
    )

    assert result.status == "failed"
    assert result.failure_stage == "kernel_timeout"
    assert result.timed_out is True
    assert result.cleanup_confirmed is False
    assert len(starts) == 1
    assert signals == [(456, session.signal.SIGTERM)]
