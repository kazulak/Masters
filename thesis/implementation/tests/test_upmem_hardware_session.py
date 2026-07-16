from __future__ import annotations

import json
from pathlib import Path
import queue
from types import SimpleNamespace

import pytest

import quantum_bench.targets.upmem.hardware_session as session
from quantum_bench.targets.upmem.hardware_taskgraph import load_hardware_taskgraph_suite


ROOT = Path(__file__).resolve().parents[1]
PROFILE = load_hardware_taskgraph_suite(
    ROOT / "configs/suites/upmem_hardware_taskgraph_correctness.yml"
).profile


def _build(root: Path) -> session.HardwareSessionBuild:
    build_dir = root / "native" / "build"
    build_dir.mkdir(parents=True)
    return session.HardwareSessionBuild(
        session_root=root,
        source_snapshot=root / "native" / "src",
        build_dir=build_dir,
        host_binary=build_dir / "bin" / "host",
        dpu_binary=build_dir / "bin" / "dpu_generic",
        source_tree_hash="source",
        host_binary_hash="host",
        dpu_binary_hash="dpu",
        build_time_s=0.0,
        build_command=("make",),
        sdk_tools={},
    )


def _task(
    root: Path, *, outside: bool = False, sequence: int = 0, task_id: str = "task"
) -> SimpleNamespace:
    task_root = root / "tasks" / f"{sequence:04d}_{task_id}"
    task_root.mkdir(parents=True)
    paths = [
        task_root / name for name in ("args.bin", "left.bin", "right.bin", "output.bin")
    ]
    for path in paths:
        path.write_bytes(b"x")
    if outside:
        paths[-1] = root.parent / "outside.bin"
        paths[-1].write_bytes(b"x")
    return SimpleNamespace(
        task_id=task_id,
        args_path=paths[0],
        left_path=paths[1],
        right_path=paths[2],
        output_path=paths[3],
    )


class _FakeStream:
    def __init__(self) -> None:
        self.lines: queue.Queue[str | None] = queue.Queue()

    def readline(self) -> str:
        item = self.lines.get()
        return "" if item is None else item

    def push(self, line: str) -> None:
        self.lines.put(line)

    def read(self, _size: int = -1) -> str:
        return ""


class _FakeStdin:
    def __init__(self, process: "_FakeProcess") -> None:
        self.process = process
        self.lines: list[str] = []

    def write(self, value: str) -> int:
        self.lines.append(value)
        self.process.handle(value)
        return len(value)

    def flush(self) -> None:
        return None


class _FakeProcess:
    def __init__(self, mode: str = "normal") -> None:
        self.mode = mode
        self.command: tuple[str, ...] | None = None
        self.kwargs: dict[str, object] = {}
        self.stdout = _FakeStream()
        self.stderr = _FakeStream()
        self.stdin = _FakeStdin(self)
        self.returncode: int | None = None
        self.start_count = 0

    def start(self, command: tuple[str, ...], kwargs: dict[str, object]) -> None:
        self.start_count += 1
        self.command = command
        self.kwargs = kwargs
        self.emit(
            {
                "schema_version": session.HARDWARE_INTERACTIVE_SESSION_SCHEMA_VERSION,
                "event": "ready",
                "status": "ready",
                "requested_dpus": 1,
                "allocated_dpus": 1,
                "allocation_time_s": 0.25,
                "binary_load_time_s": 0.5,
            }
        )

    def emit(self, payload: dict[str, object]) -> None:
        self.stdout.push(json.dumps(payload) + "\n")

    def handle(self, value: str) -> None:
        line = value.strip()
        if line == "CLOSE":
            if self.mode == "close_protocol_failure":
                self.emit(
                    {
                        "schema_version": session.HARDWARE_INTERACTIVE_SESSION_SCHEMA_VERSION,
                        "event": "response",
                        "status": "completed",
                    }
                )
                return
            self.returncode = 0
            self.emit(
                {
                    "schema_version": session.HARDWARE_INTERACTIVE_SESSION_SCHEMA_VERSION,
                    "event": "closed",
                    "status": "closed",
                    "released": True,
                    "release_time_s": 0.125,
                }
            )
            return
        if self.mode == "timeout":
            return
        if self.mode == "protocol_failure":
            self.emit(
                {
                    "schema_version": session.HARDWARE_INTERACTIVE_SESSION_SCHEMA_VERSION,
                    "event": "error",
                    "status": "failed",
                    "failure_stage": "protocol_error",
                    "error": "fake protocol failure",
                }
            )
            self.emit(
                {
                    "schema_version": session.HARDWARE_INTERACTIVE_SESSION_SCHEMA_VERSION,
                    "event": "closed",
                    "status": "closed",
                    "released": True,
                    "release_time_s": 0.1,
                }
            )
            return
        _, request_ref, response_ref = line.split(" ")
        root = Path(self.command[3]).parent  # type: ignore[index]
        request = json.loads((root / request_ref).read_text())
        tasks = []
        for index, item in enumerate(request["tasks"]):
            tasks.append(
                {
                    "sequence": index,
                    "task_id": item["task_id"],
                    "status": "completed",
                    "failure_stage": None,
                    "sdk_error_code": 0,
                    "timing": {
                        "input_read_time_s": 0.01,
                        "h2d_time_s": 0.02,
                        "kernel_time_s": 0.03,
                        "d2h_time_s": 0.04,
                        "output_write_time_s": 0.05,
                        "total_time_s": 0.15,
                    },
                    "output": {"path": item["output_path"], "bytes": 4},
                }
            )
        response = {
            "schema_version": session.HARDWARE_INTERACTIVE_SESSION_SCHEMA_VERSION,
            "manifest_kind": session.HARDWARE_INTERACTIVE_RESPONSE_KIND,
            "request_id": request["request_id"],
            "status": "completed",
            "failure_stage": None,
            "requested_dpus": 1,
            "allocated_dpus": 1,
            "tasklets": 1,
            "tasks": tasks,
        }
        (root / response_ref).write_text(json.dumps(response))
        self.emit(
            {
                "schema_version": session.HARDWARE_INTERACTIVE_SESSION_SCHEMA_VERSION,
                "event": "response",
                "status": "completed",
                "response_path": response_ref,
            }
        )

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        self.returncode = 0 if self.returncode is None else self.returncode
        return self.returncode

    def terminate(self) -> None:
        self.returncode = -15

    def kill(self) -> None:
        self.returncode = -9


def _fake_popen_factory(fake: _FakeProcess):
    def fake_popen(command: tuple[str, ...], **kwargs: object) -> _FakeProcess:
        fake.start(command, kwargs)
        return fake

    return fake_popen


def test_session_rejects_task_path_outside_containment(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="paths must be inside"):
        session.execute_hardware_session(
            _build(tmp_path / "session"),
            session_id="case",
            tasks=[_task(tmp_path / "session", outside=True)],
            profile=PROFILE,
            environment={},
        )


def test_session_command_does_not_receive_simulator_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    build = _build(tmp_path / "session")
    task = _task(build.session_root)
    captured: dict[str, object] = {}

    def fake_run(
        command: object, *, cwd: Path, env: dict[str, str], timeout_s: float
    ) -> dict[str, object]:
        captured["env"] = env
        response_path = Path(list(command)[-1])
        response_path.write_text(
            json.dumps(
                {
                    "schema_version": session.HARDWARE_GENERIC_SESSION_SCHEMA_VERSION,
                    "manifest_kind": session.HARDWARE_GENERIC_SESSION_OUTPUT_KIND,
                    "status": "completed",
                    "failure_stage": None,
                    "requested_dpus": 1,
                    "allocated_dpus": 1,
                    "tasklets": 1,
                    "tasks": [{"task_id": "task", "status": "completed"}],
                }
            )
        )
        return {
            "returncode": 0,
            "elapsed_s": 0.0,
            "timed_out": False,
            "stdout_snippet": "",
            "stderr_snippet": "",
        }

    monkeypatch.setattr(session, "_run_command", fake_run)
    result = session.execute_hardware_session(
        build,
        session_id="case",
        tasks=[task],
        profile=PROFILE,
        environment={
            "DPU_BACKEND": "simulator",
            "UPMEM_PROFILE": "backend=sim",
            "KEEP": "yes",
        },
    )

    assert result.status == "completed"
    assert captured["env"] == {"KEEP": "yes"}
    assert result.command[1] == "--session-manifest"


def test_failed_native_session_is_failed_without_cpu_fallback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    build = _build(tmp_path / "session")
    task = _task(build.session_root)

    def fake_run(*args: object, **kwargs: object) -> dict[str, object]:
        return {
            "returncode": 1,
            "elapsed_s": 0.0,
            "timed_out": False,
            "stdout_snippet": "",
            "stderr_snippet": "native failed",
        }

    monkeypatch.setattr(session, "_run_command", fake_run)
    result = session.execute_hardware_session(
        build, session_id="case", tasks=[task], profile=PROFILE, environment={}
    )

    assert result.status == "failed"
    assert result.failure_stage == "output_manifest_failed"


def test_transfer_accounting_is_additive() -> None:
    task = SimpleNamespace(
        application_visible_h2d_bytes=24, application_visible_d2h_bytes=16
    )
    assert (
        session.HardwareSessionTask.application_visible_transfer_bytes.__get__(task)
        == 40
    )


def test_interactive_session_has_one_start_ordered_requests_and_confirmed_close(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    build = _build(tmp_path / "session")
    fake = _FakeProcess()
    monkeypatch.setattr(session.subprocess, "Popen", _fake_popen_factory(fake))
    interactive = session.start_hardware_session(
        build,
        session_id="case",
        profile=PROFILE,
        environment={
            "DPU_BACKEND": "simulator",
            "UPMEM_PROFILE": "backend=sim",
            "KEEP": "yes",
        },
    )

    assert interactive.startup_metadata == {
        "requested_dpus": 1,
        "allocated_dpus": 1,
        "allocation_time_s": 0.25,
        "binary_load_time_s": 0.5,
    }
    first = interactive.submit([_task(build.session_root, task_id="first")])
    second_tasks = [
        _task(build.session_root, sequence=index + 1, task_id=f"second-{index}")
        for index in range(4)
    ]
    second = interactive.submit(second_tasks)
    assert [first.response["request_id"], second.response["request_id"]] == [
        "request-0000",
        "request-0001",
    ]
    assert [item["task_id"] for item in second.response["tasks"]] == [
        "second-0",
        "second-1",
        "second-2",
        "second-3",
    ]
    assert fake.kwargs["env"] == {"KEEP": "yes"}
    assert fake.start_count == 1
    assert len(fake.stdin.lines) == 2
    closed = interactive.close()
    assert closed.status == "closed"
    assert closed.release_confirmed is True
    assert len(fake.stdin.lines) == 3


def test_interactive_protocol_failure_is_explicit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    build = _build(tmp_path / "session")
    fake = _FakeProcess(mode="protocol_failure")
    monkeypatch.setattr(session.subprocess, "Popen", _fake_popen_factory(fake))
    interactive = session.start_hardware_session(
        build, session_id="case", profile=PROFILE, environment={}
    )
    with pytest.raises(session.HardwareInteractiveSessionError, match="protocol_error"):
        interactive.submit([_task(build.session_root)])
    assert interactive.close().release_confirmed is True


def test_interactive_timeout_is_explicit_and_does_not_claim_release(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    build = _build(tmp_path / "session")
    fake = _FakeProcess(mode="timeout")
    monkeypatch.setattr(session.subprocess, "Popen", _fake_popen_factory(fake))
    interactive = session.start_hardware_session(
        build, session_id="case", profile=PROFILE, environment={}
    )
    with pytest.raises(
        session.HardwareInteractiveSessionError, match="request_timeout"
    ):
        interactive.submit([_task(build.session_root)], timeout_s=0.01)
    assert fake.returncode is not None


def test_unexpected_close_event_terminates_native_process_and_caches_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    build = _build(tmp_path / "session")
    fake = _FakeProcess(mode="close_protocol_failure")
    monkeypatch.setattr(session.subprocess, "Popen", _fake_popen_factory(fake))
    interactive = session.start_hardware_session(
        build, session_id="case", profile=PROFILE, environment={}
    )

    closed = interactive.close()

    assert closed.status == "failed"
    assert closed.failure_stage == "close_protocol_failed"
    assert closed.release_confirmed is False
    assert fake.returncode == -15
    assert interactive.close() == closed


def test_interactive_task_paths_cannot_escape_session_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    build = _build(tmp_path / "session")
    fake = _FakeProcess()
    monkeypatch.setattr(session.subprocess, "Popen", _fake_popen_factory(fake))
    interactive = session.start_hardware_session(
        build, session_id="case", profile=PROFILE, environment={}
    )
    task = _task(build.session_root, outside=True)
    with pytest.raises(ValueError, match="paths must be inside"):
        interactive.submit([task])
    assert interactive.close().release_confirmed is True
