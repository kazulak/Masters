from __future__ import annotations

import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, Mapping, Sequence

from quantum_bench.core.records import JsonDict, to_jsonable


UPMEM_ENV_CHECK_SCHEMA_VERSION = "upmem_environment_check_v1"
DEFAULT_UPMEM_ENV_TIMEOUT_SECONDS = 10.0
SNIPPET_LIMIT = 4000

UpmemEnvCheckStatus = Literal["completed", "partial", "skipped", "failed"]
UpmemEnvTarget = Literal["auto", "simulator", "hardware"]
SimulatorAvailability = bool | Literal["not_verified"]

UPMEM_TOOL_PROBES: dict[str, tuple[str, ...]] = {
    "dpu-upmem-dpurte-clang": ("--version",),
    "dpu-pkg-config": ("--version",),
    "dpu-lldb": ("--version",),
    "dpu-diag": ("--help",),
    "dpu-profiling": ("--help",),
}
UPMEM_CORE_TOOLS = ("dpu-upmem-dpurte-clang", "dpu-pkg-config")


@dataclass(frozen=True)
class CommandExecutionRecord:
    command: tuple[str, ...]
    cwd: str | None
    return_code: int | None
    timed_out: bool
    stdout_snippet: str
    stderr_snippet: str
    elapsed_time_s: float
    status: str
    error: str | None = None

    def to_json_dict(self) -> JsonDict:
        return to_jsonable(self)


@dataclass(frozen=True)
class UpmemToolCheck:
    name: str
    path: str | None
    available: bool
    probe_status: str
    command: tuple[str, ...] = ()
    return_code: int | None = None
    timed_out: bool = False
    stdout_snippet: str = ""
    stderr_snippet: str = ""

    def to_json_dict(self) -> JsonDict:
        return to_jsonable(self)


@dataclass(frozen=True)
class UpmemSdkDiscovery:
    upmem_sdk_detected: bool
    upmem_sdk_home: str | None
    tools: tuple[UpmemToolCheck, ...]

    def to_json_dict(self) -> JsonDict:
        return to_jsonable(self)


@dataclass(frozen=True)
class SampleCheckRecord:
    attempted: bool
    status: str
    command: tuple[str, ...] = ()
    return_code: int | None = None
    timed_out: bool = False
    stdout_snippet: str = ""
    stderr_snippet: str = ""
    workspace_path: str | None = None
    target: str | None = None
    result_detected: bool | None = None
    error: str | None = None

    def to_json_dict(self) -> JsonDict:
        return to_jsonable(self)


@dataclass(frozen=True)
class UpmemEnvironmentCheckResult:
    schema_version: str
    status: UpmemEnvCheckStatus
    target: str
    timeout_seconds: float
    upmem_sdk_detected: bool
    upmem_sdk_home: str | None
    simplepim_detected: bool
    simplepim_home: str | None
    simplepim_source: str
    simplepim_stub_bin: str | None
    simulator_available: SimulatorAvailability
    hardware_probe_status: str
    tools: tuple[UpmemToolCheck, ...]
    sample_build: SampleCheckRecord
    sample_run: SampleCheckRecord
    notes: tuple[str, ...] = ()
    next_recommended_backend_step: str = ""

    def to_json_dict(self) -> JsonDict:
        return to_jsonable(self)


PathLookup = Callable[[str], str | None]
CommandRunner = Callable[[Sequence[str]], CommandExecutionRecord]


def run_command(
    argv: Sequence[str],
    *,
    cwd: Path | None = None,
    cwd_label: str | None = None,
    env: Mapping[str, str] | None = None,
    timeout_seconds: float = DEFAULT_UPMEM_ENV_TIMEOUT_SECONDS,
) -> CommandExecutionRecord:
    started = time.perf_counter()
    command = tuple(str(part) for part in argv)
    try:
        completed = subprocess.run(
            list(command),
            cwd=cwd,
            env=dict(env) if env is not None else None,
            capture_output=True,
            text=True,
            check=False,
            timeout=float(timeout_seconds),
        )
        status = "passed" if completed.returncode == 0 else "failed"
        return CommandExecutionRecord(
            command=command,
            cwd=cwd_label if cwd_label is not None else (str(cwd) if cwd is not None else None),
            return_code=int(completed.returncode),
            timed_out=False,
            stdout_snippet=_bounded_snippet(completed.stdout),
            stderr_snippet=_bounded_snippet(completed.stderr),
            elapsed_time_s=float(time.perf_counter() - started),
            status=status,
        )
    except subprocess.TimeoutExpired as exc:
        return CommandExecutionRecord(
            command=command,
            cwd=cwd_label if cwd_label is not None else (str(cwd) if cwd is not None else None),
            return_code=None,
            timed_out=True,
            stdout_snippet=_bounded_snippet(_decode_timeout_output(exc.stdout)),
            stderr_snippet=_bounded_snippet(_decode_timeout_output(exc.stderr)),
            elapsed_time_s=float(time.perf_counter() - started),
            status="timed_out",
            error=f"Command timed out after {timeout_seconds} seconds",
        )
    except FileNotFoundError as exc:
        return CommandExecutionRecord(
            command=command,
            cwd=cwd_label if cwd_label is not None else (str(cwd) if cwd is not None else None),
            return_code=None,
            timed_out=False,
            stdout_snippet="",
            stderr_snippet="",
            elapsed_time_s=float(time.perf_counter() - started),
            status="unavailable",
            error=str(exc),
        )
    except OSError as exc:
        return CommandExecutionRecord(
            command=command,
            cwd=cwd_label if cwd_label is not None else (str(cwd) if cwd is not None else None),
            return_code=None,
            timed_out=False,
            stdout_snippet="",
            stderr_snippet="",
            elapsed_time_s=float(time.perf_counter() - started),
            status="failed",
            error=str(exc),
        )


def discover_upmem_sdk(
    *,
    env: Mapping[str, str] | None = None,
    path_lookup: PathLookup | None = None,
    command_runner: Callable[..., CommandExecutionRecord] = run_command,
    timeout_seconds: float = DEFAULT_UPMEM_ENV_TIMEOUT_SECONDS,
) -> UpmemSdkDiscovery:
    probe_env = env if env is not None else os.environ
    lookup = path_lookup or shutil.which
    upmem_home = _clean_path(probe_env.get("UPMEM_HOME"))
    tools: list[UpmemToolCheck] = []
    for name, args in UPMEM_TOOL_PROBES.items():
        path = _find_tool(name, upmem_home=upmem_home, path_lookup=lookup)
        if not path:
            tools.append(UpmemToolCheck(name=name, path=None, available=False, probe_status="unavailable"))
            continue
        command = (str(path), *args)
        result = command_runner(command, timeout_seconds=timeout_seconds)
        tools.append(
            UpmemToolCheck(
                name=name,
                path=str(path),
                available=True,
                probe_status=result.status,
                command=result.command,
                return_code=result.return_code,
                timed_out=result.timed_out,
                stdout_snippet=result.stdout_snippet,
                stderr_snippet=result.stderr_snippet,
            )
        )
    core_available = all(any(tool.name == name and tool.available for tool in tools) for name in UPMEM_CORE_TOOLS)
    return UpmemSdkDiscovery(
        upmem_sdk_detected=bool(upmem_home or core_available),
        upmem_sdk_home=upmem_home,
        tools=tuple(tools),
    )


def build_environment_check_result(
    *,
    target: str,
    timeout_seconds: float,
    sdk: UpmemSdkDiscovery,
    simplepim: object | None = None,
    sample_build: SampleCheckRecord | None = None,
    sample_run: SampleCheckRecord | None = None,
    notes: Sequence[str] = (),
) -> UpmemEnvironmentCheckResult:
    build_record = sample_build or SampleCheckRecord(attempted=False, status="not_requested")
    run_record = sample_run or SampleCheckRecord(attempted=False, status="not_requested", target=target)
    simulator_available = _simulator_availability(target, run_record)
    status = _overall_status(sdk, build_record, run_record)
    return UpmemEnvironmentCheckResult(
        schema_version=UPMEM_ENV_CHECK_SCHEMA_VERSION,
        status=status,
        target=target,
        timeout_seconds=float(timeout_seconds),
        upmem_sdk_detected=sdk.upmem_sdk_detected,
        upmem_sdk_home=sdk.upmem_sdk_home,
        simplepim_detected=False,
        simplepim_home=None,
        simplepim_source="retired",
        simplepim_stub_bin=None,
        simulator_available=simulator_available,
        hardware_probe_status="not_verified",
        tools=sdk.tools,
        sample_build=build_record,
        sample_run=run_record,
        notes=tuple(notes),
        next_recommended_backend_step=_next_step(status, build_record, run_record),
    )


def sample_success_marker(stdout: str, stderr: str = "") -> bool:
    text = f"{stdout}\n{stderr}".lower()
    return "result is correct" in text or "the result is correct" in text


def _overall_status(
    sdk: UpmemSdkDiscovery,
    sample_build: SampleCheckRecord,
    sample_run: SampleCheckRecord,
) -> UpmemEnvCheckStatus:
    if not sdk.upmem_sdk_detected:
        return "skipped"
    if sample_build.attempted and sample_build.status == "failed":
        return "failed"
    if sample_run.attempted and sample_run.status == "failed":
        return "failed"
    core_tools_ok = _core_tool_probes_passed(sdk.tools)
    if not sdk.upmem_sdk_detected or not core_tools_ok:
        return "partial"
    if sample_build.status == "skipped" or sample_run.status == "skipped":
        return "partial"
    if sample_build.attempted and sample_build.status != "passed":
        return "partial"
    if sample_run.attempted and sample_run.status != "passed":
        return "partial"
    return "completed"


def _core_tool_probes_passed(tools: Sequence[UpmemToolCheck]) -> bool:
    by_name = {tool.name: tool for tool in tools}
    for name in UPMEM_CORE_TOOLS:
        tool = by_name.get(name)
        if tool is None or not tool.available or tool.probe_status != "passed":
            return False
    return True


def _simulator_availability(target: str, sample_run: SampleCheckRecord) -> SimulatorAvailability:
    if target not in {"simulator", "auto"}:
        return "not_verified"
    if not sample_run.attempted:
        return "not_verified"
    return sample_run.status == "passed"


def _next_step(status: str, sample_build: SampleCheckRecord, sample_run: SampleCheckRecord) -> str:
    if status == "completed" and sample_run.attempted:
        return "Use the strict generic UPMEM SDK-simulator route for bounded TaskGraph evidence."
    if status == "completed":
        return "Use upmem-mvp-benchmark for strict generic SDK-simulator evidence."
    if sample_build.status == "failed":
        return "Fix the UPMEM SDK/toolchain configuration before generic simulator execution."
    if sample_run.status == "failed":
        return "Fix the UPMEM SDK simulator configuration before generic runtime execution."
    return "Resolve the missing UPMEM SDK/toolchain configuration before generic simulator execution."


def _bounded_snippet(value: str | None, limit: int = SNIPPET_LIMIT) -> str:
    if value is None:
        return ""
    text = str(value)
    if len(text) <= limit:
        return text
    return text[:limit] + "\n...[truncated]"


def _decode_timeout_output(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _clean_path(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    if not stripped:
        return None
    return str(Path(stripped).expanduser())


def _find_tool(name: str, *, upmem_home: str | None, path_lookup: PathLookup) -> str | None:
    if upmem_home:
        candidate = Path(upmem_home) / "bin" / name
        if candidate.exists():
            return str(candidate)
    return path_lookup(name)
