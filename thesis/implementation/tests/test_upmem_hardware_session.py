from __future__ import annotations

import json
from pathlib import Path
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


def _task(root: Path, *, outside: bool = False) -> SimpleNamespace:
    task_root = root / "tasks" / "0000_task"
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
        task_id="task",
        args_path=paths[0],
        left_path=paths[1],
        right_path=paths[2],
        output_path=paths[3],
    )


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
