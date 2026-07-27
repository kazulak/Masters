"""Guarded M1 physical provider qualification orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping

from quantum_bench.bench.reporting import write_run_manifest
from quantum_bench.bench.run_dirs import (
    EVIDENCE_ARTIFACT_KIND,
    LEGACY_ARTIFACT_KIND,
    create_run_dir,
)
from quantum_bench.core.jsonio import write_json, write_jsonl
from quantum_bench.providers.qualification import (
    PHYSICAL_TARGET,
    ProviderCatalog,
    ProviderSpec,
    RunnerQualification,
    fingerprint_catalog,
    fingerprint_tools,
    load_provider_catalog,
    parse_runner_result,
    provider_source_fingerprint,
    require_qualification_repository_source,
    require_provider_source,
    resolve_catalog_path,
    resolve_provider_path,
    resolve_root,
    sha256_file,
    simplepim_runner_schema_errors,
    source_gate_failure,
)


PLAN_SCHEMA_VERSION = "provider_qualification_plan_v1"
RESULT_SCHEMA_VERSION = "provider_qualification_result_v1"
NORMALIZED_SCHEMA_VERSION = "provider_qualification_normalized_record_v1"
PLAN_ROOT = "build/provider_qualification"
RAW_PREPARE_RESULT = "raw_runner_prepare.json"
RAW_EXECUTION_PREFLIGHT_RESULT = "raw_runner_preflight.json"
RAW_EXECUTION_RESULT = "raw_runner_result.json"
SUMMARY_RESULT = "provider_qualification.json"
PREPARE_TIMEOUT_SECONDS = 120.0
EXECUTE_TIMEOUT_SECONDS = 600.0
PROCESS_GROUP_GRACE_SECONDS = 2.0
CANONICAL_ROOT = Path(__file__).resolve().parents[3]
CANONICAL_CATALOG_PATH = "configs/qualification/upmem_provider_m1.yml"
CANONICAL_RUNNER_PATH = "native/upmem/simplepim/simplepim_qualification_runner.py"
CANONICAL_PROVIDER_SOURCE_PATH = "external/SimplePIM"
CANONICAL_PINNED_COMMIT = "1d639c53532555f01e9f71d872e7712b166d6cba"
CANONICAL_REPOSITORY_SOURCE_PATHS = (
    "configs/qualification/upmem_provider_m1.yml",
    "native/upmem/simplepim/simplepim_qualification_runner.py",
    "native/upmem/simplepim/qualification",
    "src/quantum_bench/providers/qualification.py",
    "src/quantum_bench/bench/provider_qualification.py",
    "src/quantum_bench/bench/__main__.py",
    "Makefile",
)
SIMULATOR_ENV_KEYS = (
    "DPU_BACKEND",
    "UPMEM_MODE",
    "UPMEM_TARGET",
    "UPMEM_PROFILE",
    "UPMEM_PROFILE_BASE",
)
SIMULATOR_ALIASES = {
    "sim",
    "simulation",
    "simulator",
    "fsim",
    "casim",
    "functional_simulator",
    "cycle_accurate_simulator",
}


@dataclass(frozen=True)
class ProviderQualificationPlan:
    plan_dir: Path
    plan_path: Path
    status: str


@dataclass(frozen=True)
class ProviderQualificationResult:
    run_dir: Path
    result_path: Path
    raw_result_path: Path
    normalized_records_path: Path
    manifest_path: Path
    status: str


@dataclass(frozen=True)
class _ExecutionClass:
    name: str
    artifact_kind: str
    success_status: str
    canonical_evidence: bool


@dataclass(frozen=True)
class _ExecutionPreflight:
    command: list[str]
    completed: subprocess.CompletedProcess[str] | None
    invocation_error: str | None
    timeout_cleanup: Mapping[str, Any] | None
    payload: dict[str, Any] | None
    raw_hash: str | None
    raw_error: str | None
    contract_errors: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return (
            self.completed is not None
            and self.completed.returncode == 0
            and self.invocation_error is None
            and self.raw_error is None
            and not self.contract_errors
            and self.payload is not None
        )

    @property
    def reason(self) -> str | None:
        if self.passed:
            return None
        return (
            self.invocation_error
            or self.raw_error
            or (self.contract_errors[0] if self.contract_errors else None)
            or (
                f"runner prepare exited with return code {self.completed.returncode}"
                if self.completed is not None
                else "runner prepare invocation failed"
            )
        )


@dataclass(frozen=True)
class _ProcProcess:
    pid: int
    ppid: int
    process_group: int
    session: int
    start_time: int
    state: str


_CANONICAL_EXECUTION = _ExecutionClass(
    name="canonical_physical_evidence",
    artifact_kind=EVIDENCE_ARTIFACT_KIND,
    success_status="qualified",
    canonical_evidence=True,
)
_TEST_EXECUTION = _ExecutionClass(
    name="internal_test_non_evidence",
    artifact_kind=LEGACY_ARTIFACT_KIND,
    success_status="test_passed_non_evidence",
    canonical_evidence=False,
)
_TEST_EXECUTION_HOOK = object()


def prepare_provider_qualification(
    root_dir: Path,
    *,
    catalog_path: Path,
    provider_id: str | None = None,
    environment: Mapping[str, str] | None = None,
) -> ProviderQualificationPlan:
    root, catalog = _resolved_inputs(root_dir, catalog_path)
    selected = [catalog.get(provider_id)] if provider_id else list(catalog.providers)
    env = dict(os.environ if environment is None else environment)
    tool_fingerprints = fingerprint_tools(env)
    runner_environment = {"HOST_CC": str(tool_fingerprints["host_cc"]["runner_value"])}
    runner_env = {**env, **runner_environment}
    plan_dir = _create_unique_plan_dir(root)
    provider_rows: list[dict[str, Any]] = []
    executable_failures = False

    for provider in selected:
        row = _plan_provider(provider, root)
        if not provider.executable or provider.status != "executable":
            row["preparation_status"] = "blocked"
            row["runner_prepare"] = None
            provider_rows.append(row)
            continue

        source_fingerprint = provider_source_fingerprint(provider, root)
        source_failure = source_gate_failure(provider, source_fingerprint)
        row["source_fingerprint"] = source_fingerprint
        if source_failure:
            row["preparation_status"] = "failed"
            row["preparation_reason"] = source_failure
            row["runner_prepare"] = None
            executable_failures = True
            provider_rows.append(row)
            continue

        runner_prepare = _prepare_runner(
            root,
            plan_dir,
            provider,
            runner_env,
            runner_environment,
            tool_fingerprints["host_cc"],
            tool_fingerprints["dpu-upmem-dpurte-clang"],
        )
        row["runner_prepare"] = runner_prepare
        row["preparation_status"] = runner_prepare["status"]
        row["preparation_reason"] = runner_prepare.get("reason")
        executable_failures = (
            executable_failures or runner_prepare["status"] != "prepared"
        )
        provider_rows.append(row)

    status = "failed" if executable_failures else "prepared"
    plan = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "status": status,
        "catalog_id": catalog.catalog_id,
        "catalog_path": _relative_path(catalog.path, root),
        "selected_provider": provider_id,
        "providers": provider_rows,
        "catalog_fingerprint": fingerprint_catalog(catalog, root),
        "tool_fingerprints": tool_fingerprints,
        "runner_environment": runner_environment,
        "execution_policy": {
            "mode": "prepare-only",
            "build_attempted": False,
            "dpu_allocation_attempted": False,
            "dpu_launch_attempted": False,
            "runner_execute_invoked": False,
        },
    }
    plan_path = (plan_dir / "plan.json").resolve()
    write_json(plan_path, plan)
    return ProviderQualificationPlan(plan_dir, plan_path, status)


def execute_provider_qualification(
    root_dir: Path,
    *,
    catalog_path: Path,
    provider_id: str | None = None,
    environment: Mapping[str, str] | None = None,
) -> ProviderQualificationResult:
    return _execute_provider_qualification(
        root_dir,
        catalog_path=catalog_path,
        provider_id=provider_id,
        environment=environment,
        execution_class=_CANONICAL_EXECUTION,
    )


def _execute_provider_qualification_for_test(
    root_dir: Path,
    *,
    catalog_path: Path,
    provider_id: str | None = None,
    environment: Mapping[str, str] | None = None,
    hook: object,
) -> ProviderQualificationResult:
    if hook is not _TEST_EXECUTION_HOOK:
        raise ValueError("invalid internal qualification test hook")
    return _execute_provider_qualification(
        root_dir,
        catalog_path=catalog_path,
        provider_id=provider_id,
        environment=environment,
        execution_class=_TEST_EXECUTION,
    )


def _execute_provider_qualification(
    root_dir: Path,
    *,
    catalog_path: Path,
    provider_id: str | None,
    environment: Mapping[str, str] | None,
    execution_class: _ExecutionClass,
) -> ProviderQualificationResult:
    root, catalog = _resolved_inputs(root_dir, catalog_path)
    provider = _select_executable_provider(catalog, provider_id)
    if execution_class.canonical_evidence:
        _require_canonical_execution_inputs(root, catalog, provider)
    env = dict(os.environ if environment is None else environment)
    _validate_physical_request(provider, env)
    source_fingerprint = require_provider_source(provider, root)
    repository_source_fingerprint = (
        require_qualification_repository_source(
            root,
            CANONICAL_REPOSITORY_SOURCE_PATHS,
        )
        if execution_class.canonical_evidence
        else {
            "classification": _TEST_EXECUTION.name,
            "gate_enforced": False,
        }
    )
    tool_fingerprints = fingerprint_tools(env)
    runner_environment = {"HOST_CC": str(tool_fingerprints["host_cc"]["runner_value"])}
    runner_env = {**env, **runner_environment}
    runner_path = _resolved_runner(root, provider)
    runner_fingerprint = {
        "path": _relative_path(runner_path, root),
        "sha256": sha256_file(runner_path),
        "host_cc": tool_fingerprints["host_cc"],
    }

    run_dir = create_run_dir(
        root,
        "provider_qualification",
        artifact_kind=execution_class.artifact_kind,
        route_label=provider.provider_id,
    ).resolve()
    workdir = (run_dir / "runner_work").resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    raw_preflight_path = (run_dir / RAW_EXECUTION_PREFLIGHT_RESULT).resolve()
    raw_result_path = (run_dir / RAW_EXECUTION_RESULT).resolve()
    result_path = (run_dir / SUMMARY_RESULT).resolve()
    normalized_path = (run_dir / "normalized_records.jsonl").resolve()
    manifest_path = (run_dir / "run_manifest.json").resolve()
    preflight = _execution_preflight(
        root=root,
        runner_path=runner_path,
        workdir=workdir,
        raw_path=raw_preflight_path,
        environment=runner_env,
        provider=provider,
        expected_host_cc=tool_fingerprints["host_cc"],
        expected_dpu_cc=tool_fingerprints["dpu-upmem-dpurte-clang"],
    )
    command = [
        sys.executable,
        str(runner_path),
        "--execute",
        "--workdir",
        str(workdir),
        "--json-output",
        str(raw_result_path),
    ]
    raw_result_path.unlink(missing_ok=True)
    if preflight.passed:
        completed, invocation_error, timeout_cleanup = _run_runner_process(
            command,
            cwd=root,
            environment=runner_env,
            timeout_seconds=EXECUTE_TIMEOUT_SECONDS,
        )
    else:
        completed = None
        invocation_error = f"runner preflight failed: {preflight.reason}"
        timeout_cleanup = None

    payload, raw_hash, raw_error = _read_designated_raw_result(raw_result_path)
    qualification = (
        parse_runner_result(
            payload,
            provider,
            expected_host_cc=tool_fingerprints["host_cc"],
            expected_dpu_cc=tool_fingerprints["dpu-upmem-dpurte-clang"],
            expected_preflight=preflight.payload,
        )
        if payload is not None
        else _empty_qualification(raw_error or "runner result is unavailable")
    )
    status = (
        execution_class.success_status
        if completed is not None
        and completed.returncode == 0
        and raw_error is None
        and qualification.status == "qualified"
        else "failed"
    )
    failure_reason = _execution_failure_reason(
        status=status,
        invocation_error=invocation_error,
        completed=completed,
        raw_error=raw_error,
        qualification=qualification,
    )
    resource_release_status = _resource_release_status(
        status,
        qualification,
        execution_class,
    )
    summary = _result_summary(
        root=root,
        catalog=catalog,
        provider=provider,
        qualification=qualification,
        status=status,
        failure_reason=failure_reason,
        raw_preflight_path=raw_preflight_path,
        raw_preflight_hash=preflight.raw_hash,
        preflight=preflight,
        raw_result_path=raw_result_path,
        raw_hash=raw_hash,
        source_fingerprint=source_fingerprint,
        tool_fingerprints=tool_fingerprints,
        runner_fingerprint=runner_fingerprint,
        command=command,
        runner_environment=runner_environment,
        completed=completed,
        timeout_cleanup=timeout_cleanup,
        resource_release_status=resource_release_status,
        execution_class=execution_class,
        repository_source_fingerprint=repository_source_fingerprint,
    )
    write_json(result_path, summary)
    write_jsonl(
        normalized_path,
        [
            _normalized_record(
                provider,
                qualification,
                status,
                failure_reason,
                raw_hash,
                resource_release_status,
                execution_class,
            )
        ],
    )
    manifest = write_run_manifest(
        run_dir,
        run_kind="provider_qualification",
        suite_id=catalog.catalog_id,
        suite_path=_relative_path(catalog.path, root),
        artifact_kind=execution_class.artifact_kind,
        route_label=provider.provider_id,
        route_id="provider_qualification",
        backend_id=provider.provider_id,
        execution_scope=(
            "physical_upmem_provider_qualification"
            if execution_class.canonical_evidence
            else "internal_provider_qualification_test"
        ),
        evidence_type=execution_class.name,
        normalized_records=normalized_path.name,
        summary=result_path.name,
        upmem_execution_mode=PHYSICAL_TARGET,
        artifact_retention="compact",
        command=_recorded_runner_command(command, runner_environment),
        root_dir=root,
    )
    manifest.update(
        {
            "status": status,
            "provider_id": provider.provider_id,
            "raw_runner_result": raw_result_path.name
            if raw_result_path.is_file()
            else None,
            "raw_runner_result_sha256": raw_hash,
            "raw_runner_preflight": (
                raw_preflight_path.name if raw_preflight_path.is_file() else None
            ),
            "raw_runner_preflight_sha256": preflight.raw_hash,
            "runner_preflight_status": ("passed" if preflight.passed else "failed"),
            "source_fingerprint": source_fingerprint,
            "tool_fingerprints": tool_fingerprints,
            "runner_fingerprint": runner_fingerprint,
            "runner_environment": runner_environment,
            "resource_release_status": resource_release_status,
            "result_classification": execution_class.name,
            "canonical_evidence": execution_class.canonical_evidence,
            "repository_source_fingerprint": repository_source_fingerprint,
            "hardware_available": (
                "verified_by_execution"
                if status == "qualified" and execution_class.canonical_evidence
                else "not_verified_by_canonical_execution"
            ),
        }
    )
    write_json(manifest_path, manifest)
    return ProviderQualificationResult(
        run_dir,
        result_path,
        raw_result_path,
        normalized_path,
        manifest_path,
        status,
    )


def _resolved_inputs(
    root_dir: Path, catalog_path: Path
) -> tuple[Path, ProviderCatalog]:
    root = resolve_root(root_dir)
    catalog = load_provider_catalog(resolve_catalog_path(root, catalog_path))
    return root, catalog


def _require_canonical_execution_inputs(
    root: Path,
    catalog: ProviderCatalog,
    provider: ProviderSpec,
) -> None:
    canonical_root = CANONICAL_ROOT.resolve()
    canonical_catalog = (canonical_root / CANONICAL_CATALOG_PATH).resolve()
    if root != canonical_root:
        raise ValueError(
            f"public --execute requires canonical qualification root {canonical_root}"
        )
    if catalog.path != canonical_catalog:
        raise ValueError(
            "custom provider catalogs are prepare-only; public --execute requires "
            f"{CANONICAL_CATALOG_PATH}"
        )
    if catalog.catalog_id != "upmem_provider_m1":
        raise ValueError("canonical qualification catalog id is invalid")
    expected = {
        "provider id": (provider.provider_id, "simplepim"),
        "runner": (provider.runner, CANONICAL_RUNNER_PATH),
        "source": (provider.source_path, CANONICAL_PROVIDER_SOURCE_PATH),
        "pinned commit": (provider.pinned_commit, CANONICAL_PINNED_COMMIT),
    }
    for label, (actual, required) in expected.items():
        if actual != required:
            raise ValueError(f"canonical SimplePIM {label} must be {required!r}")
    runner = resolve_provider_path(root, provider.runner or "", label="provider runner")
    source = resolve_provider_path(
        root,
        provider.source_path or "",
        label="provider source",
    )
    if runner != (canonical_root / CANONICAL_RUNNER_PATH).resolve():
        raise ValueError("canonical SimplePIM runner path does not resolve exactly")
    if source != (canonical_root / CANONICAL_PROVIDER_SOURCE_PATH).resolve():
        raise ValueError("canonical SimplePIM source path does not resolve exactly")


def _validate_physical_request(
    provider: ProviderSpec, environment: Mapping[str, str]
) -> None:
    if not provider.executable or provider.status != "executable":
        reason = provider.reason or "provider is blocked in the M1 catalog"
        raise ValueError(f"provider {provider.provider_id} is not executable: {reason}")
    if environment.get("UPMEM_ALLOW_PHYSICAL_HARDWARE") != "1":
        raise ValueError("UPMEM_ALLOW_PHYSICAL_HARDWARE=1 is required for --execute")
    for key in SIMULATOR_ENV_KEYS:
        value = str(environment.get(key, ""))
        if _is_simulator_selector(value):
            raise ValueError(
                f"{key}={value} is forbidden for physical provider qualification"
            )


def _is_simulator_selector(value: str) -> bool:
    lowered = value.strip().lower()
    if not lowered:
        return False
    tokens = {token for token in re.split(r"[^a-z0-9_]+", lowered) if token}
    return (
        "simulator" in lowered
        or lowered in SIMULATOR_ALIASES
        or bool(tokens & SIMULATOR_ALIASES)
    )


def _select_executable_provider(
    catalog: ProviderCatalog, provider_id: str | None
) -> ProviderSpec:
    if provider_id:
        return catalog.get(provider_id)
    candidates = [
        provider
        for provider in catalog.providers
        if provider.executable and provider.status == "executable"
    ]
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise ValueError("catalog has no executable provider")
    ids = ", ".join(provider.provider_id for provider in candidates)
    raise ValueError(
        f"--provider is required when multiple executable providers exist: {ids}"
    )


def _plan_provider(provider: ProviderSpec, root: Path) -> dict[str, Any]:
    runner_path = (
        resolve_provider_path(root, provider.runner, label="provider runner")
        if provider.runner
        else None
    )
    return {
        "id": provider.provider_id,
        "name": provider.name,
        "catalog_status": provider.status,
        "executable": provider.executable,
        "reason": provider.reason,
        "runner": provider.runner,
        "runner_available": bool(runner_path and runner_path.is_file()),
        "runner_contract": {
            "schema_version": provider.runner_schema_version,
            "provider_id": provider.provider_id,
            "probe_id": provider.probe_id,
        },
        "requested_counts": {
            "dpus": provider.requested_dpus,
            "tasklets": provider.requested_tasklets,
        },
        "lane": provider.lane,
    }


def _prepare_runner(
    root: Path,
    plan_dir: Path,
    provider: ProviderSpec,
    environment: Mapping[str, str],
    runner_environment: Mapping[str, str],
    expected_host_cc: Mapping[str, Any],
    expected_dpu_cc: Mapping[str, Any],
) -> dict[str, Any]:
    runner_path = _resolved_runner(root, provider)
    provider_dir = (plan_dir / "providers" / provider.provider_id).resolve()
    provider_dir.mkdir(parents=True, exist_ok=False)
    workdir = (provider_dir / "work").resolve()
    raw_path = (provider_dir / RAW_PREPARE_RESULT).resolve()
    raw_path.unlink(missing_ok=True)
    command = [
        sys.executable,
        str(runner_path),
        "--prepare-only",
        "--workdir",
        str(workdir),
        "--json-output",
        str(raw_path),
    ]
    completed, invocation_error, timeout_cleanup = _run_runner_process(
        command,
        cwd=root,
        environment=environment,
        timeout_seconds=PREPARE_TIMEOUT_SECONDS,
    )
    if completed is None:
        return {
            "status": "failed",
            "reason": invocation_error or "runner invocation failed",
            "returncode": None,
            "raw_result": None,
            "raw_result_sha256": None,
            "command": command,
            "environment": dict(runner_environment),
            "timeout_cleanup": timeout_cleanup,
        }
    payload, raw_hash, raw_error = _read_designated_raw_result(raw_path)
    contract_errors = (
        simplepim_runner_schema_errors(
            payload,
            provider,
            mode="prepare",
            expected_host_cc=expected_host_cc,
            expected_dpu_cc=expected_dpu_cc,
        )
        if payload is not None
        else ()
    )
    status = (
        "prepared"
        if completed.returncode == 0
        and invocation_error is None
        and raw_error is None
        and not contract_errors
        else "failed"
    )
    reason = (
        "strict runner prepare contract passed"
        if status == "prepared"
        else invocation_error
        or raw_error
        or (contract_errors[0] if contract_errors else None)
        or f"runner prepare exited with return code {completed.returncode}"
    )
    return {
        "status": status,
        "reason": reason,
        "returncode": completed.returncode,
        "raw_result": raw_path.name if raw_path.is_file() else None,
        "raw_result_sha256": raw_hash,
        "runner_stdout": _snippet(completed.stdout),
        "runner_stderr": _snippet(completed.stderr),
        "runner_contract_errors": contract_errors,
        "command": command,
        "environment": dict(runner_environment),
        "timeout_cleanup": timeout_cleanup,
    }


def _execution_preflight(
    *,
    root: Path,
    runner_path: Path,
    workdir: Path,
    raw_path: Path,
    environment: Mapping[str, str],
    provider: ProviderSpec,
    expected_host_cc: Mapping[str, Any],
    expected_dpu_cc: Mapping[str, Any],
) -> _ExecutionPreflight:
    raw_path.unlink(missing_ok=True)
    command = [
        sys.executable,
        str(runner_path),
        "--prepare-only",
        "--workdir",
        str(workdir),
        "--json-output",
        str(raw_path),
    ]
    completed, invocation_error, timeout_cleanup = _run_runner_process(
        command,
        cwd=root,
        environment=environment,
        timeout_seconds=PREPARE_TIMEOUT_SECONDS,
    )
    payload, raw_hash, raw_error = _read_designated_raw_result(raw_path)
    contract_errors = (
        simplepim_runner_schema_errors(
            payload,
            provider,
            mode="prepare",
            expected_host_cc=expected_host_cc,
            expected_dpu_cc=expected_dpu_cc,
        )
        if payload is not None
        else ()
    )
    return _ExecutionPreflight(
        command=command,
        completed=completed,
        invocation_error=invocation_error,
        timeout_cleanup=timeout_cleanup,
        payload=payload,
        raw_hash=raw_hash,
        raw_error=raw_error,
        contract_errors=contract_errors,
    )


def _resolved_runner(root: Path, provider: ProviderSpec) -> Path:
    if not provider.runner:
        raise ValueError(f"provider {provider.provider_id} has no runner")
    runner = resolve_provider_path(root, provider.runner, label="provider runner")
    if not runner.is_file():
        raise ValueError(f"provider runner is not a file: {runner}")
    return runner


def _create_unique_plan_dir(root: Path) -> Path:
    parent = (root / PLAN_ROOT).resolve()
    parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
    plan_dir = parent / stamp
    suffix = 1
    while plan_dir.exists():
        plan_dir = parent / f"{stamp}_{suffix:02d}"
        suffix += 1
    plan_dir.mkdir(parents=False, exist_ok=False)
    return plan_dir.resolve()


def _run_runner_process(
    command: list[str],
    *,
    cwd: Path,
    environment: Mapping[str, str],
    timeout_seconds: float,
) -> tuple[
    subprocess.CompletedProcess[str] | None,
    str | None,
    dict[str, Any] | None,
]:
    try:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=dict(environment),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
    except OSError as exc:
        return None, str(exc), None
    try:
        stdout, stderr = process.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        stdout, stderr, cleanup = _terminate_runner_process_group(process)
        returncode = process.returncode
        completed = subprocess.CompletedProcess(
            command,
            returncode if returncode is not None else -signal.SIGKILL,
            stdout,
            stderr,
        )
        return (
            completed,
            f"runner timed out after {timeout_seconds:g} seconds",
            cleanup,
        )
    return (
        subprocess.CompletedProcess(command, process.returncode, stdout, stderr),
        None,
        None,
    )


def _terminate_runner_process_group(
    process: subprocess.Popen[str],
) -> tuple[str, str, dict[str, Any]]:
    process_group = process.pid
    signal_errors: list[str] = []
    process_table = _proc_process_table()
    descendant_identities = {
        item.pid: item for item in _descendant_processes(process.pid, process_table)
    }
    descendants_before_sigterm = tuple(sorted(descendant_identities))
    detached_before_sigterm = tuple(
        sorted(
            item.pid
            for item in descendant_identities.values()
            if item.process_group != process_group or item.session != process_group
        )
    )
    sigterm_sent = _send_process_group_signal(
        process_group,
        signal.SIGTERM,
        signal_errors,
    )
    descendant_sigterm_sent = _signal_proc_processes(
        descendant_identities.values(),
        signal.SIGTERM,
        signal_errors,
    )
    communicated = False
    stdout = ""
    stderr = ""
    try:
        stdout, stderr = process.communicate(timeout=PROCESS_GROUP_GRACE_SECONDS)
        communicated = True
    except subprocess.TimeoutExpired:
        pass

    leader_exited_before_group_probe = process.poll() is not None
    live_members_after_sigterm = _live_process_group_members(process_group)
    after_sigterm_table = _proc_process_table()
    descendant_identities.update(
        {
            item.pid: item
            for root in tuple(descendant_identities.values())
            for item in _descendant_processes(root.pid, after_sigterm_table)
        }
    )
    descendants_after_sigterm = _live_proc_processes(
        descendant_identities,
        after_sigterm_table,
    )
    sigkill_sent = False
    if _process_group_exists(process_group):
        sigkill_sent = _send_process_group_signal(
            process_group,
            signal.SIGKILL,
            signal_errors,
        )
    descendant_sigkill_sent = _signal_proc_processes(
        descendants_after_sigterm,
        signal.SIGKILL,
        signal_errors,
    )

    if not communicated:
        try:
            stdout, stderr = process.communicate(timeout=PROCESS_GROUP_GRACE_SECONDS)
        except subprocess.TimeoutExpired as exc:
            stdout = _timeout_output(exc.output)
            stderr = _timeout_output(exc.stderr)
            if process.poll() is None:
                process.kill()
                stdout, stderr = process.communicate()

    live_members_after_cleanup = _wait_for_live_process_group_exit(
        process_group,
        PROCESS_GROUP_GRACE_SECONDS,
    )
    descendants_after_cleanup = _wait_for_proc_process_exit(
        descendant_identities,
        PROCESS_GROUP_GRACE_SECONDS,
    )
    return (
        stdout,
        stderr,
        {
            "process_group": process_group,
            "sigterm_sent": sigterm_sent,
            "sigkill_sent": sigkill_sent,
            "descendant_sigterm_sent": list(descendant_sigterm_sent),
            "descendant_sigkill_sent": list(descendant_sigkill_sent),
            "process_exited": process.returncode is not None,
            "group_probe_performed": True,
            "leader_exited_before_group_probe": leader_exited_before_group_probe,
            "live_members_after_sigterm": list(live_members_after_sigterm),
            "live_members_after_cleanup": list(live_members_after_cleanup),
            "process_group_terminated": not live_members_after_cleanup,
            "descendant_pids_before_sigterm": list(descendants_before_sigterm),
            "detached_descendant_pids_before_sigterm": list(detached_before_sigterm),
            "descendant_pids_after_sigterm": [
                item.pid for item in descendants_after_sigterm
            ],
            "descendant_pids_after_cleanup": [
                item.pid for item in descendants_after_cleanup
            ],
            "descendant_tree_terminated": not descendants_after_cleanup,
            "signal_errors": signal_errors,
        },
    )


def _send_process_group_signal(
    process_group: int,
    signum: signal.Signals,
    errors: list[str],
) -> bool:
    try:
        os.killpg(process_group, signum)
    except ProcessLookupError:
        return False
    except OSError as exc:
        errors.append(f"{signum.name}:{exc}")
        return False
    return True


def _process_group_exists(process_group: int) -> bool:
    try:
        os.killpg(process_group, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _proc_process_table(
    proc_root: Path = Path("/proc"),
) -> dict[int, _ProcProcess]:
    table: dict[int, _ProcProcess] = {}
    try:
        candidates = proc_root.iterdir()
    except OSError:
        return table
    for candidate in candidates:
        if not candidate.name.isdigit():
            continue
        try:
            stat_text = (candidate / "stat").read_text(encoding="ascii")
            fields = stat_text[stat_text.rindex(")") + 2 :].split()
            item = _ProcProcess(
                pid=int(candidate.name),
                state=fields[0],
                ppid=int(fields[1]),
                process_group=int(fields[2]),
                session=int(fields[3]),
                start_time=int(fields[19]),
            )
        except (OSError, ValueError, IndexError):
            continue
        if item.state not in {"X", "Z"}:
            table[item.pid] = item
    return table


def _descendant_processes(
    root_pid: int,
    table: Mapping[int, _ProcProcess],
) -> tuple[_ProcProcess, ...]:
    parents = {root_pid}
    descendants: dict[int, _ProcProcess] = {}
    while parents:
        children = {
            pid: item
            for pid, item in table.items()
            if item.ppid in parents and pid not in descendants
        }
        if not children:
            break
        descendants.update(children)
        parents = set(children)
    return tuple(descendants[pid] for pid in sorted(descendants))


def _live_proc_processes(
    identities: Mapping[int, _ProcProcess],
    table: Mapping[int, _ProcProcess] | None = None,
) -> tuple[_ProcProcess, ...]:
    current = _proc_process_table() if table is None else table
    return tuple(
        identity
        for pid, identity in sorted(identities.items())
        if pid in current and current[pid].start_time == identity.start_time
    )


def _signal_proc_processes(
    processes: Iterable[_ProcProcess],
    signum: signal.Signals,
    errors: list[str],
) -> tuple[int, ...]:
    sent: list[int] = []
    current = _proc_process_table()
    for process in processes:
        observed = current.get(process.pid)
        if observed is None or observed.start_time != process.start_time:
            continue
        try:
            os.kill(process.pid, signum)
        except ProcessLookupError:
            continue
        except OSError as exc:
            errors.append(f"{signum.name}:pid={process.pid}:{exc}")
        else:
            sent.append(process.pid)
    return tuple(sorted(sent))


def _wait_for_proc_process_exit(
    identities: Mapping[int, _ProcProcess],
    timeout_seconds: float,
) -> tuple[_ProcProcess, ...]:
    deadline = time.monotonic() + timeout_seconds
    while True:
        live = _live_proc_processes(identities)
        if not live or time.monotonic() >= deadline:
            return live
        time.sleep(0.01)


def _live_process_group_members(
    process_group: int,
    proc_root: Path = Path("/proc"),
) -> tuple[int, ...]:
    members: list[int] = []
    try:
        candidates = proc_root.iterdir()
    except OSError:
        return ()
    for candidate in candidates:
        if not candidate.name.isdigit():
            continue
        try:
            stat_text = (candidate / "stat").read_text(encoding="ascii")
            fields = stat_text[stat_text.rindex(")") + 2 :].split()
            state = fields[0]
            member_group = int(fields[2])
        except (OSError, ValueError, IndexError):
            continue
        if member_group == process_group and state not in {"X", "Z"}:
            members.append(int(candidate.name))
    return tuple(sorted(members))


def _wait_for_live_process_group_exit(
    process_group: int,
    timeout_seconds: float,
) -> tuple[int, ...]:
    deadline = time.monotonic() + timeout_seconds
    while True:
        members = _live_process_group_members(process_group)
        if not members or time.monotonic() >= deadline:
            return members
        time.sleep(0.01)


def _timeout_output(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _recorded_runner_command(
    command: list[str],
    runner_environment: Mapping[str, str],
) -> str:
    assignments = " ".join(
        f"{key}={shlex.quote(value)}"
        for key, value in sorted(runner_environment.items())
    )
    return f"{assignments} {shlex.join(command)}"


def _read_designated_raw_result(
    path: Path,
) -> tuple[dict[str, Any] | None, str | None, str | None]:
    if not path.is_file():
        return None, None, f"runner did not create designated JSON result {path.name}"
    raw_hash = sha256_file(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, raw_hash, f"invalid designated runner JSON: {exc}"
    if not isinstance(value, dict):
        return None, raw_hash, "designated runner JSON must be an object"
    return value, raw_hash, None


def _execution_failure_reason(
    *,
    status: str,
    invocation_error: str | None,
    completed: subprocess.CompletedProcess[str] | None,
    raw_error: str | None,
    qualification: RunnerQualification,
) -> str | None:
    if status != "failed":
        return None
    if invocation_error:
        return invocation_error
    if raw_error:
        return raw_error
    if completed is None:
        return "runner invocation failed"
    if completed.returncode != 0:
        return f"runner exited with return code {completed.returncode}"
    return qualification.reason


def _result_summary(
    *,
    root: Path,
    catalog: ProviderCatalog,
    provider: ProviderSpec,
    qualification: RunnerQualification,
    status: str,
    failure_reason: str | None,
    raw_preflight_path: Path,
    raw_preflight_hash: str | None,
    preflight: _ExecutionPreflight,
    raw_result_path: Path,
    raw_hash: str | None,
    source_fingerprint: Mapping[str, Any],
    tool_fingerprints: Mapping[str, Any],
    runner_fingerprint: Mapping[str, Any],
    command: list[str],
    runner_environment: Mapping[str, str],
    completed: subprocess.CompletedProcess[str] | None,
    timeout_cleanup: Mapping[str, Any] | None,
    resource_release_status: str,
    execution_class: _ExecutionClass,
    repository_source_fingerprint: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": status,
        "qualified": status == "qualified",
        "result_classification": execution_class.name,
        "canonical_evidence": execution_class.canonical_evidence,
        "provider_id": provider.provider_id,
        "provider_name": provider.name,
        "catalog": {
            "id": catalog.catalog_id,
            "path": _relative_path(catalog.path, root),
            "sha256": sha256_file(catalog.path),
        },
        "runner_contract": {
            "schema_version": provider.runner_schema_version,
            "provider_id": provider.provider_id,
            "probe_id": provider.probe_id,
        },
        "runner": {
            **dict(runner_fingerprint),
            "returncode": completed.returncode if completed else None,
            "stdout": _snippet(completed.stdout if completed else ""),
            "stderr": _snippet(completed.stderr if completed else ""),
            "command": command,
            "recorded_command": _recorded_runner_command(command, runner_environment),
            "environment": dict(runner_environment),
            "timeout_cleanup": (
                dict(timeout_cleanup) if timeout_cleanup is not None else None
            ),
        },
        "raw_runner_result": {
            "path": raw_result_path.name if raw_result_path.is_file() else None,
            "sha256": raw_hash,
        },
        "raw_runner_preflight": {
            "path": (raw_preflight_path.name if raw_preflight_path.is_file() else None),
            "sha256": raw_preflight_hash,
            "status": "passed" if preflight.passed else "failed",
            "reason": preflight.reason,
            "returncode": (
                preflight.completed.returncode
                if preflight.completed is not None
                else None
            ),
            "command": preflight.command,
            "contract_errors": preflight.contract_errors,
            "timeout_cleanup": (
                dict(preflight.timeout_cleanup)
                if preflight.timeout_cleanup is not None
                else None
            ),
        },
        "source_fingerprint": dict(source_fingerprint),
        "repository_source_fingerprint": dict(repository_source_fingerprint),
        "tool_fingerprints": dict(tool_fingerprints),
        "requested_counts": {
            "dpus": provider.requested_dpus,
            "tasklets": provider.requested_tasklets,
        },
        "observed_counts": {
            "dpus": qualification.observed_dpus,
            "tasklets": qualification.observed_tasklets,
        },
        "configured_tasklets_per_dpu": qualification.configured_tasklets,
        "hardware_preflight_verified": qualification.hardware_preflight_verified,
        "target": qualification.target,
        "native_execution": qualification.native_execution,
        "validation_performed": qualification.validation_performed,
        "exact_validation": qualification.exact_validation,
        "release_status": qualification.release_status,
        "resource_release_status": resource_release_status,
        "fallback_used": qualification.fallback_used,
        "simulator_kernel_executed": qualification.simulator_kernel_executed,
        "runner_status": qualification.runner_status,
        "runner_contract_errors": qualification.contract_errors,
        "failure_reason": failure_reason,
        "normalized_records": "normalized_records.jsonl",
        "run_manifest": "run_manifest.json",
    }


def _normalized_record(
    provider: ProviderSpec,
    qualification: RunnerQualification,
    status: str,
    failure_reason: str | None,
    raw_hash: str | None,
    resource_release_status: str,
    execution_class: _ExecutionClass,
) -> dict[str, Any]:
    return {
        "schema_version": NORMALIZED_SCHEMA_VERSION,
        "provider_id": provider.provider_id,
        "probe_id": provider.probe_id,
        "status": status,
        "result_classification": execution_class.name,
        "canonical_evidence": execution_class.canonical_evidence,
        "requested_dpu_count": provider.requested_dpus,
        "observed_dpu_count": qualification.observed_dpus,
        "requested_tasklet_count": provider.requested_tasklets,
        "configured_tasklet_count": qualification.configured_tasklets,
        "observed_tasklet_count": qualification.observed_tasklets,
        "hardware_preflight_verified": qualification.hardware_preflight_verified,
        "target": qualification.target,
        "native_execution": qualification.native_execution,
        "validation_performed": qualification.validation_performed,
        "validation_status": (
            "exact" if qualification.exact_validation is True else "failed"
        ),
        "release_status": qualification.release_status,
        "resource_release_status": resource_release_status,
        "fallback_used": qualification.fallback_used,
        "simulator_kernel_executed": qualification.simulator_kernel_executed,
        "raw_runner_result_sha256": raw_hash,
        "reason": failure_reason or qualification.reason,
    }


def _resource_release_status(
    status: str,
    qualification: RunnerQualification,
    execution_class: _ExecutionClass,
) -> str:
    return (
        "confirmed"
        if execution_class.canonical_evidence
        and status == "qualified"
        and qualification.status == "qualified"
        and qualification.release_status == "released"
        else (
            "test_only_unverified"
            if not execution_class.canonical_evidence and status != "failed"
            else "unconfirmed"
        )
    )


def _empty_qualification(reason: str) -> RunnerQualification:
    return RunnerQualification(
        status="failed",
        reason=reason,
        contract_errors=(reason,),
        runner_status=None,
        hardware_preflight_verified=None,
        target=None,
        observed_dpus=None,
        configured_tasklets=None,
        observed_tasklets=None,
        native_execution=None,
        validation_performed=None,
        exact_validation=None,
        release_status=None,
        fallback_used=None,
        simulator_kernel_executed=None,
    )


def _relative_path(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def _snippet(value: str) -> str | None:
    text = value.strip()
    return text[-2000:] if text else None
