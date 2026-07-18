from __future__ import annotations

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
    discover_upmem_sdk,
    run_command,
)


UpmemEnvTarget = str


def run_upmem_env_check(
    root_dir: Path,
    *,
    run_sample: bool = False,
    target: UpmemEnvTarget = "auto",
    timeout_seconds: float = DEFAULT_UPMEM_ENV_TIMEOUT_SECONDS,
    env: Mapping[str, str] | None = None,
    path_lookup: Callable[[str], str | None] | None = None,
    command_runner: Callable[..., CommandExecutionRecord] = run_command,
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

    sample_build = SampleCheckRecord(
        attempted=False,
        status="skipped" if run_sample else "not_requested",
        error=(
            "The retired SimplePIM sample is not part of the active environment check; "
            "use the generic SDK loop route for simulator execution."
            if run_sample
            else None
        ),
    )
    sample_run = SampleCheckRecord(
        attempted=False,
        status="skipped" if run_sample else "not_requested",
        target="simulator" if target == "auto" else target,
        error=(
            "The retired SimplePIM sample is not part of the active environment check; "
            "use the generic SDK loop route for simulator execution."
            if run_sample
            else None
        ),
    )
    notes = (
        "SimplePIM and dense sample probes are retired; generic SDK loop execution is the active UPMEM simulator path.",
    ) if run_sample else ()
    if target == "hardware" and not run_sample:
        notes = (*notes, "Hardware target requested without --run-sample; no hardware workload was executed.")

    result = build_environment_check_result(
        target=target,
        timeout_seconds=timeout_seconds,
        sdk=sdk,
        sample_build=sample_build,
        sample_run=sample_run,
        notes=notes,
    )
    artifact_path = run_dir / "upmem_environment_check.json"
    write_json(artifact_path, result)
    (run_dir / "upmem_env_check_summary.md").write_text(
        _summary_markdown(result.to_json_dict()), encoding="utf-8"
    )
    return run_dir, artifact_path, result.status


def _summary_markdown(payload: JsonDict) -> str:
    lines = [
        "# UPMEM Environment Check",
        "",
        f"- Status: {payload['status']}",
        f"- Target: {payload['target']}",
        f"- UPMEM SDK detected: {payload['upmem_sdk_detected']}",
        "- SimplePIM probe: retired",
        f"- Simulator available: {payload['simulator_available']}",
        f"- Hardware probe status: {payload['hardware_probe_status']}",
        "",
        "## Tools",
        "",
        "| Tool | Available | Probe Status | Path |",
        "| --- | --- | --- | --- |",
    ]
    for tool in payload["tools"]:
        lines.append(
            f"| {tool['name']} | {tool['available']} | {tool['probe_status']} | {tool.get('path') or ''} |"
        )
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
