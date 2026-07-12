from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Callable, Mapping

from quantum_bench.bench.run_dirs import create_run_dir
from quantum_bench.core.jsonio import write_json
from quantum_bench.core.records import JsonDict
from quantum_bench.environment import capture_environment
from quantum_bench.targets.upmem.environment import (
    DEFAULT_UPMEM_ENV_TIMEOUT_SECONDS,
    CommandExecutionRecord,
    SampleCheckRecord,
    build_environment_check_result,
    discover_simplepim_source,
    discover_upmem_sdk,
    run_command,
    sample_success_marker,
)


UpmemEnvTarget = str


def run_upmem_env_check(
    root_dir: Path,
    *,
    run_sample: bool = False,
    target: UpmemEnvTarget = "auto",
    timeout_seconds: float = DEFAULT_UPMEM_ENV_TIMEOUT_SECONDS,
    simplepim_home: str | None = None,
    env: Mapping[str, str] | None = None,
    path_lookup: Callable[[str], str | None] | None = None,
    command_runner: Callable[..., CommandExecutionRecord] = run_command,
    copy_simplepim_source_func: Callable[[Path, Path], None] | None = None,
) -> tuple[Path, Path, str]:
    _validate_options(target=target, timeout_seconds=timeout_seconds)
    run_dir = create_run_dir(root_dir, "upmem_env_check")
    write_json(run_dir / "environment.json", capture_environment(root_dir))

    sdk = discover_upmem_sdk(
        env=env,
        path_lookup=path_lookup,
        command_runner=command_runner,
        timeout_seconds=timeout_seconds,
    )
    simplepim = discover_simplepim_source(root_dir, simplepim_home_override=simplepim_home, env=env)

    sample_build: SampleCheckRecord | None = None
    sample_run: SampleCheckRecord | None = None
    notes: list[str] = []
    if run_sample:
        sample_build, sample_run = _run_sample_checks(
            run_dir=run_dir,
            simplepim_home=simplepim.simplepim_home,
            sdk_detected=sdk.upmem_sdk_detected,
            target=target,
            timeout_seconds=timeout_seconds,
            env=env,
            command_runner=command_runner,
            copy_simplepim_source_func=copy_simplepim_source_func or copy_simplepim_source,
        )
        write_json(run_dir / "simplepim_sample_build.json", sample_build)
        write_json(run_dir / "simplepim_sample_run.json", sample_run)
    elif target == "hardware":
        notes.append("Hardware target requested without --run-sample; no hardware workload was executed.")

    result = build_environment_check_result(
        target=target,
        timeout_seconds=timeout_seconds,
        sdk=sdk,
        simplepim=simplepim,
        sample_build=sample_build,
        sample_run=sample_run,
        notes=notes,
    )
    artifact_path = run_dir / "upmem_environment_check.json"
    write_json(artifact_path, result)
    (run_dir / "upmem_env_check_summary.md").write_text(
        _summary_markdown(result.to_json_dict()),
        encoding="utf-8",
    )
    return run_dir, artifact_path, result.status


def copy_simplepim_source(src: Path, dst: Path) -> None:
    shutil.copytree(
        src,
        dst,
        ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc"),
    )


def _run_sample_checks(
    *,
    run_dir: Path,
    simplepim_home: str | None,
    sdk_detected: bool,
    target: str,
    timeout_seconds: float,
    env: Mapping[str, str] | None,
    command_runner: Callable[..., CommandExecutionRecord],
    copy_simplepim_source_func: Callable[[Path, Path], None],
) -> tuple[SampleCheckRecord, SampleCheckRecord]:
    if simplepim_home is None:
        skipped = SampleCheckRecord(attempted=False, status="skipped", error="SimplePIM source tree is unavailable")
        return skipped, SampleCheckRecord(attempted=False, status="skipped", target=_sample_target(target), error="SimplePIM source tree is unavailable")
    if not sdk_detected:
        skipped = SampleCheckRecord(attempted=False, status="skipped", error="UPMEM SDK/toolchain is unavailable")
        return skipped, SampleCheckRecord(attempted=False, status="skipped", target=_sample_target(target), error="UPMEM SDK/toolchain is unavailable")

    source = Path(simplepim_home)
    workspace = run_dir / "sample_work" / "SimplePIM"
    try:
        copy_simplepim_source_func(source, workspace)
    except Exception as exc:
        failed = SampleCheckRecord(
            attempted=True,
            status="failed",
            workspace_path=_relative_to_run(workspace, run_dir),
            error=f"Failed to copy SimplePIM source tree: {exc}",
        )
        return failed, SampleCheckRecord(attempted=False, status="skipped", target=_sample_target(target), error="Sample build workspace was not created")

    sample_cwd = workspace / "benchmarks" / "va"
    sample_cwd_label = _relative_to_run(sample_cwd, run_dir)
    build_result = command_runner(
        ("make",),
        cwd=sample_cwd,
        cwd_label=sample_cwd_label,
        env=_base_env(env),
        timeout_seconds=timeout_seconds,
    )
    build = _sample_record_from_command(
        build_result,
        attempted=True,
        workspace_path=_relative_to_run(workspace, run_dir),
    )
    if build.status != "passed":
        return build, SampleCheckRecord(
            attempted=False,
            status="skipped",
            target=_sample_target(target),
            workspace_path=_relative_to_run(workspace, run_dir),
            error="Sample build did not pass",
        )

    run_target = _sample_target(target)
    run_env = _base_env(env)
    if run_target == "simulator":
        run_env["DPU_BACKEND"] = "simulator"
    run_result = command_runner(
        ("./bin/host",),
        cwd=sample_cwd,
        cwd_label=sample_cwd_label,
        env=run_env,
        timeout_seconds=timeout_seconds,
    )
    run = _sample_record_from_command(
        run_result,
        attempted=True,
        workspace_path=_relative_to_run(workspace, run_dir),
        target=run_target,
        result_detected=sample_success_marker(run_result.stdout_snippet, run_result.stderr_snippet),
    )
    return build, run


def _sample_record_from_command(
    result: CommandExecutionRecord,
    *,
    attempted: bool,
    workspace_path: str,
    target: str | None = None,
    result_detected: bool | None = None,
) -> SampleCheckRecord:
    status = "passed" if result.status == "passed" else "failed"
    return SampleCheckRecord(
        attempted=attempted,
        status=status,
        command=result.command,
        return_code=result.return_code,
        timed_out=result.timed_out,
        stdout_snippet=result.stdout_snippet,
        stderr_snippet=result.stderr_snippet,
        workspace_path=workspace_path,
        target=target,
        result_detected=result_detected,
        error=result.error,
    )


def _base_env(env: Mapping[str, str] | None) -> dict[str, str]:
    return dict(os.environ if env is None else env)


def _sample_target(target: str) -> str:
    return "simulator" if target == "auto" else target


def _summary_markdown(payload: JsonDict) -> str:
    lines = [
        "# UPMEM Environment Check",
        "",
        f"- Status: {payload['status']}",
        f"- Target: {payload['target']}",
        f"- UPMEM SDK detected: {payload['upmem_sdk_detected']}",
        f"- SimplePIM detected: {payload['simplepim_detected']}",
        f"- Simulator available: {payload['simulator_available']}",
        f"- Hardware probe status: {payload['hardware_probe_status']}",
        "",
        "## Tools",
        "",
        "| Tool | Available | Probe Status | Path |",
        "| --- | --- | --- | --- |",
    ]
    for tool in payload["tools"]:
        lines.append(f"| {tool['name']} | {tool['available']} | {tool['probe_status']} | {tool.get('path') or ''} |")
    lines.extend(
        [
            "",
            "## Next Step",
            "",
            str(payload["next_recommended_backend_step"]),
            "",
        ]
    )
    return "\n".join(lines)


def _validate_options(*, target: str, timeout_seconds: float) -> None:
    if target not in {"auto", "simulator", "hardware"}:
        raise ValueError("--target must be one of: auto, simulator, hardware")
    if timeout_seconds <= 0:
        raise ValueError("--timeout-seconds must be > 0")


def _relative_to_run(path: Path, run_dir: Path) -> str:
    try:
        return path.relative_to(run_dir).as_posix()
    except ValueError:
        return path.name
