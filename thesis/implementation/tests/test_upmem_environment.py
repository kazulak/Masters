from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from quantum_bench.bench import __main__ as bench_main
import quantum_bench.bench.upmem_env_check as upmem_env_check_module
from quantum_bench.bench.upmem_env_check import run_upmem_env_check
from quantum_bench.targets.upmem.environment import (
    CommandExecutionRecord,
    SampleCheckRecord,
    build_environment_check_result,
    discover_simplepim_source,
    discover_upmem_sdk,
    run_command,
    sample_success_marker,
)


def _fake_simplepim_tree(path: Path) -> Path:
    (path / "benchmarks" / "va").mkdir(parents=True)
    (path / "benchmarks" / "va" / "Makefile").write_text("va:\n\t@true\n", encoding="utf-8")
    (path / "lib").mkdir(parents=True)
    return path


def _passed(command: tuple[str, ...] | list[str], **kwargs: object) -> CommandExecutionRecord:
    return CommandExecutionRecord(
        command=tuple(command),
        cwd=kwargs.get("cwd_label") if isinstance(kwargs.get("cwd_label"), str) else None,
        return_code=0,
        timed_out=False,
        stdout_snippet="ok",
        stderr_snippet="",
        elapsed_time_s=0.001,
        status="passed",
    )


def _failed(command: tuple[str, ...] | list[str], **kwargs: object) -> CommandExecutionRecord:
    return CommandExecutionRecord(
        command=tuple(command),
        cwd=kwargs.get("cwd_label") if isinstance(kwargs.get("cwd_label"), str) else None,
        return_code=2,
        timed_out=False,
        stdout_snippet="",
        stderr_snippet="failed",
        elapsed_time_s=0.001,
        status="failed",
    )


def test_discover_upmem_sdk_from_env_and_path() -> None:
    lookup = lambda command: f"/sdk/bin/{command}" if command in {"dpu-upmem-dpurte-clang", "dpu-pkg-config"} else None
    result = discover_upmem_sdk(
        env={"UPMEM_HOME": "/sdk"},
        path_lookup=lookup,
        command_runner=_passed,
    )
    payload = result.to_json_dict()

    assert result.upmem_sdk_detected is True
    assert result.upmem_sdk_home == "/sdk"
    assert payload["tools"][0]["probe_status"] == "passed"
    assert {tool["name"] for tool in payload["tools"] if tool["available"]} == {
        "dpu-upmem-dpurte-clang",
        "dpu-pkg-config",
    }


def test_discover_upmem_sdk_unavailable_without_env_or_tools() -> None:
    result = discover_upmem_sdk(env={}, path_lookup=lambda command: None, command_runner=_passed)

    assert result.upmem_sdk_detected is False
    assert all(tool.available is False for tool in result.tools)


def test_discover_upmem_sdk_checks_upmem_home_bin(tmp_path: Path) -> None:
    sdk_home = tmp_path / "upmem"
    bin_dir = sdk_home / "bin"
    bin_dir.mkdir(parents=True)
    for name in ("dpu-upmem-dpurte-clang", "dpu-pkg-config"):
        (bin_dir / name).write_text("# fake tool\n", encoding="utf-8")

    result = discover_upmem_sdk(
        env={"UPMEM_HOME": str(sdk_home)},
        path_lookup=lambda command: None,
        command_runner=_passed,
    )
    available_paths = {tool.name: tool.path for tool in result.tools if tool.available}

    assert result.upmem_sdk_detected is True
    assert available_paths["dpu-upmem-dpurte-clang"] == str(bin_dir / "dpu-upmem-dpurte-clang")
    assert available_paths["dpu-pkg-config"] == str(bin_dir / "dpu-pkg-config")


def test_discover_simplepim_source_cli_env_fallback_and_missing(tmp_path: Path) -> None:
    root_dir = tmp_path / "implementation"
    root_dir.mkdir()
    cli_tree = _fake_simplepim_tree(tmp_path / "cli_simplepim")
    env_tree = _fake_simplepim_tree(tmp_path / "env_simplepim")
    fallback_tree = _fake_simplepim_tree(tmp_path / "legacy" / "extern" / "SimplePIM")

    cli = discover_simplepim_source(root_dir, simplepim_home_override=str(cli_tree), env={"SIMPLEPIM_HOME": str(env_tree)})
    env = discover_simplepim_source(root_dir, env={"SIMPLEPIM_HOME": str(env_tree)})
    fallback = discover_simplepim_source(root_dir, env={})
    missing_root = tmp_path / "isolated" / "implementation"
    missing_root.mkdir(parents=True)
    missing = discover_simplepim_source(missing_root, env={})

    assert cli.simplepim_source == "cli"
    assert cli.simplepim_home == str(cli_tree)
    assert env.simplepim_source == "environment"
    assert env.simplepim_home == str(env_tree)
    assert fallback.simplepim_source == "repo_fallback"
    assert fallback.simplepim_home == str(fallback_tree)
    assert missing.simplepim_detected is False
    assert missing.simplepim_source == "none"


def test_command_runner_success_missing_and_timeout() -> None:
    success = run_command((sys.executable, "-c", "print('hello')"), timeout_seconds=5)
    missing = run_command(("definitely-missing-upmem-env-check-command",), timeout_seconds=1)
    timed_out = run_command((sys.executable, "-c", "import time; time.sleep(1)"), timeout_seconds=0.01)

    assert success.status == "passed"
    assert success.return_code == 0
    assert "hello" in success.stdout_snippet
    assert missing.status == "unavailable"
    assert missing.return_code is None
    assert timed_out.status == "timed_out"
    assert timed_out.timed_out is True


def test_command_runner_bounds_stdout_and_stderr() -> None:
    result = run_command(
        (
            sys.executable,
            "-c",
            "import sys; print('x'*6000); print('y'*6000, file=sys.stderr)",
        ),
        timeout_seconds=5,
    )

    assert result.status == "passed"
    assert len(result.stdout_snippet) < 4100
    assert len(result.stderr_snippet) < 4100
    assert "truncated" in result.stdout_snippet
    assert "truncated" in result.stderr_snippet


def test_build_result_simulator_availability_values(tmp_path: Path) -> None:
    sdk = discover_upmem_sdk(
        env={"UPMEM_HOME": "/sdk"},
        path_lookup=lambda command: f"/sdk/{command}" if command in {"dpu-upmem-dpurte-clang", "dpu-pkg-config"} else None,
        command_runner=_passed,
    )
    simplepim = discover_simplepim_source(tmp_path, simplepim_home_override=str(_fake_simplepim_tree(tmp_path / "SimplePIM")))
    not_verified = build_environment_check_result(target="simulator", timeout_seconds=10, sdk=sdk, simplepim=simplepim)
    failed_run = build_environment_check_result(
        target="simulator",
        timeout_seconds=10,
        sdk=sdk,
        simplepim=simplepim,
        sample_run=SampleCheckRecord(attempted=True, status="failed", target="simulator"),
    )
    passed_run = build_environment_check_result(
        target="simulator",
        timeout_seconds=10,
        sdk=sdk,
        simplepim=simplepim,
        sample_run=SampleCheckRecord(attempted=True, status="passed", target="simulator"),
    )

    assert not_verified.simulator_available == "not_verified"
    assert failed_run.simulator_available is False
    assert passed_run.simulator_available is True


def test_run_upmem_env_check_unavailable_environment_writes_skipped_artifact(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(upmem_env_check_module, "capture_environment", lambda root_dir: {})
    run_dir, artifact_path, status = run_upmem_env_check(
        tmp_path / "implementation",
        env={},
        path_lookup=lambda command: None,
        command_runner=_passed,
    )
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))

    assert status == "skipped"
    assert payload["status"] == "skipped"
    assert payload["upmem_sdk_detected"] is False
    assert payload["simplepim_detected"] is False
    assert (run_dir / "environment.json").exists()


def test_run_sample_failure_writes_failed_json_status(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(upmem_env_check_module, "capture_environment", lambda root_dir: {})
    simplepim = _fake_simplepim_tree(tmp_path / "SimplePIM")

    def copy_fake(src: Path, dst: Path) -> None:
        _fake_simplepim_tree(dst)

    def runner(command: tuple[str, ...] | list[str], **kwargs: object) -> CommandExecutionRecord:
        if tuple(command) == ("make",):
            return _failed(command, **kwargs)
        return _passed(command, **kwargs)

    run_dir, artifact_path, status = run_upmem_env_check(
        tmp_path / "implementation",
        run_sample=True,
        target="simulator",
        simplepim_home=str(simplepim),
        env={"UPMEM_HOME": "/sdk"},
        path_lookup=lambda command: f"/sdk/{command}" if command in {"dpu-upmem-dpurte-clang", "dpu-pkg-config"} else None,
        command_runner=runner,
        copy_simplepim_source_func=copy_fake,
    )
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))

    assert status == "failed"
    assert payload["status"] == "failed"
    assert payload["sample_build"]["attempted"] is True
    assert payload["sample_build"]["status"] == "failed"
    assert payload["sample_build"]["workspace_path"] == "sample_work/SimplePIM"
    assert payload["sample_run"]["status"] == "skipped"
    assert (run_dir / "simplepim_sample_build.json").exists()
    assert (run_dir / "simplepim_sample_run.json").exists()


def test_run_sample_success_uses_simulator_overlay_and_relative_workspace(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(upmem_env_check_module, "capture_environment", lambda root_dir: {})
    simplepim = _fake_simplepim_tree(tmp_path / "SimplePIM")
    seen_env: list[dict[str, str]] = []

    def copy_fake(src: Path, dst: Path) -> None:
        _fake_simplepim_tree(dst)

    def runner(command: tuple[str, ...] | list[str], **kwargs: object) -> CommandExecutionRecord:
        if tuple(command) == ("./bin/host",):
            seen_env.append(dict(kwargs["env"]))
            return CommandExecutionRecord(tuple(command), kwargs.get("cwd_label"), 0, False, "the result is correct", "", 0.001, "passed")
        return _passed(command, **kwargs)

    _, artifact_path, status = run_upmem_env_check(
        tmp_path / "implementation",
        run_sample=True,
        target="auto",
        simplepim_home=str(simplepim),
        env={"UPMEM_HOME": "/sdk"},
        path_lookup=lambda command: f"/sdk/{command}" if command in {"dpu-upmem-dpurte-clang", "dpu-pkg-config"} else None,
        command_runner=runner,
        copy_simplepim_source_func=copy_fake,
    )
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))

    assert status == "completed"
    assert payload["simulator_available"] is True
    assert payload["sample_run"]["target"] == "simulator"
    assert payload["sample_run"]["result_detected"] is True
    assert payload["sample_run"]["workspace_path"] == "sample_work/SimplePIM"
    assert seen_env[0]["DPU_BACKEND"] == "simulator"


def test_sample_success_marker_is_best_effort() -> None:
    assert sample_success_marker("the result is correct") is True
    assert sample_success_marker("program completed") is False


def test_cli_dispatch_returns_zero_even_for_failed_status(monkeypatch, capsys, tmp_path: Path) -> None:
    artifact = tmp_path / "upmem_environment_check.json"
    artifact.write_text(json.dumps({"status": "failed"}), encoding="utf-8")

    def fake_run(root_dir: Path, **kwargs: object):
        run_dir = tmp_path / "fake_run"
        run_dir.mkdir()
        return run_dir, artifact, "failed"

    monkeypatch.setattr(upmem_env_check_module, "run_upmem_env_check", fake_run)
    monkeypatch.setattr(
        "sys.argv",
        [
            "python -m quantum_bench.bench",
            "upmem-env-check",
            "--run-sample",
            "--target",
            "simulator",
        ],
    )

    assert bench_main.main() == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "failed"
