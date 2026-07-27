"""Canonical orchestration for the M1 physical provider qualification."""
from __future__ import annotations
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
from typing import Any, Mapping, NamedTuple
from quantum_bench.bench.reporting import write_run_manifest
from quantum_bench.bench.run_dirs import EVIDENCE_ARTIFACT_KIND, create_run_dir
from quantum_bench.core.jsonio import write_json, write_jsonl
from quantum_bench.providers.qualification import (
    ProviderCatalog,
    ProviderSpec,
    RunnerQualification,
    fingerprint_catalog,
    fingerprint_tools,
    load_provider_catalog,
    parse_runner_result,
    provider_source_fingerprint,
    qualification_repository_fingerprint,
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
PREPARE_TIMEOUT_SECONDS = 120.0
EXECUTE_TIMEOUT_SECONDS = 600.0
CANONICAL_CATALOG_PATH = "configs/qualification/upmem_provider_m1.yml"
CANONICAL_CATALOG_ID = "upmem_provider_m1"
CANONICAL_PROVIDER_ID = "simplepim"
CANONICAL_RUNNER_PATH = "native/upmem/simplepim/simplepim_qualification_runner.py"
CANONICAL_SOURCE_PATH = "external/SimplePIM"
CANONICAL_PINNED_COMMIT = "1d639c53532555f01e9f71d872e7712b166d6cba"
CANONICAL_REPOSITORY_SOURCE_PATHS = (CANONICAL_CATALOG_PATH, CANONICAL_RUNNER_PATH, "native/upmem/simplepim/qualification", "src/quantum_bench/providers/qualification.py", "src/quantum_bench/bench/provider_qualification.py", "src/quantum_bench/bench/__main__.py", "Makefile")
SIMULATOR_ENV_KEYS = ("DPU_BACKEND", "DPU_PROFILE", "SIMPLEPIM_BACKEND", "UPMEM_BACKEND", "UPMEM_MODE", "UPMEM_TARGET", "UPMEM_PROFILE", "UPMEM_PROFILE_BASE")
SIMULATOR_ALIASES = {"sim", "simulation", "simulator", "fsim", "casim", "functional_simulator"}
ProviderQualificationPlan = NamedTuple("ProviderQualificationPlan", [("plan_dir", Path), ("plan_path", Path), ("status", str)])
ProviderQualificationResult = NamedTuple("ProviderQualificationResult", [("run_dir", Path), ("result_path", Path), ("raw_result_path", Path), ("normalized_records_path", Path), ("manifest_path", Path), ("status", str)])
def prepare_provider_qualification(
    root_dir: Path,
    *,
    catalog_path: Path,
    provider_id: str | None = None,
    environment: Mapping[str, str] | None = None,
) -> ProviderQualificationPlan:
    root, catalog = _inputs(root_dir, catalog_path)
    selected = [catalog.get(provider_id)] if provider_id else list(catalog.providers)
    env = dict(os.environ if environment is None else environment)
    tools = fingerprint_tools(env)
    plan_dir = _unique_plan_dir(root)
    rows: list[dict[str, Any]] = []
    failed = False
    for provider in selected:
        row = _provider_row(root, provider)
        if not provider.executable or provider.status == "blocked":
            row.update(
                preparation_status="blocked",
                preparation_reason=provider.reason,
                runner_prepare=None,
            )
            rows.append(row)
            continue
        source = provider_source_fingerprint(provider, root)
        row["source_fingerprint"] = source
        if reason := source_gate_failure(provider, source):
            row.update(
                preparation_status="failed",
                preparation_reason=reason,
                runner_prepare=None,
            )
            failed = True
        else:
            prepared = _call_runner(
                root,
                plan_dir / "providers" / provider.provider_id,
                provider,
                "--prepare-only",
                env,
                tools["host_cc"],
            )
            row.update(
                preparation_status=prepared["status"],
                preparation_reason=prepared.get("reason"),
                runner_prepare=prepared,
            )
            failed = failed or prepared["status"] != "prepared"
        rows.append(row)
    plan = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "status": "failed" if failed else "prepared",
        "catalog_id": catalog.catalog_id,
        "catalog_path": _rel(catalog.path, root),
        "selected_provider": provider_id,
        "providers": rows,
        "catalog_fingerprint": fingerprint_catalog(catalog, root),
        "tool_fingerprints": tools,
        "runner_environment": {"HOST_CC": tools["host_cc"]["runner_value"]},
        "execution_policy": {
            "mode": "prepare-only",
            "build_attempted": False,
            "dpu_allocation_attempted": False,
            "dpu_launch_attempted": False,
            "runner_execute_invoked": False,
        },
    }
    path = plan_dir / "plan.json"
    write_json(path, plan)
    return ProviderQualificationPlan(plan_dir, path, plan["status"])
def execute_provider_qualification(
    root_dir: Path,
    *,
    catalog_path: Path,
    provider_id: str | None = None,
    environment: Mapping[str, str] | None = None,
) -> ProviderQualificationResult:
    return _execute(
        root_dir,
        catalog_path=catalog_path,
        provider_id=provider_id,
        environment=environment,
        test_hook=False,
    )
_TEST_EXECUTION_HOOK = object()
def _execute_provider_qualification_for_test(
    root_dir: Path,
    *,
    catalog_path: Path,
    provider_id: str,
    environment: Mapping[str, str],
    hook: object,
) -> ProviderQualificationResult:
    if hook is not _TEST_EXECUTION_HOOK:
        raise ValueError("internal qualification test hook required")
    return _execute(
        root_dir,
        catalog_path=catalog_path,
        provider_id=provider_id,
        environment=environment,
        test_hook=True,
    )
def _execute(
    root_dir: Path,
    *,
    catalog_path: Path,
    provider_id: str | None,
    environment: Mapping[str, str] | None,
    test_hook: bool,
) -> ProviderQualificationResult:
    root = resolve_root(root_dir)
    catalog = load_provider_catalog(resolve_catalog_path(root, catalog_path))
    provider = _select(catalog, provider_id)
    if not test_hook:
        _require_canonical_identity(catalog, provider)
        repository = require_qualification_repository_source(root, CANONICAL_REPOSITORY_SOURCE_PATHS)
    else:
        repository = qualification_repository_fingerprint(root, (_rel(catalog.path, root), provider.runner or "", provider.source_path or ""))
    source = require_provider_source(provider, root)
    runner_path = resolve_provider_path(root, provider.runner or "", label="provider runner")
    runner_fingerprint = {"path": provider.runner, "sha256": sha256_file(runner_path)}
    env = dict(os.environ if environment is None else environment)
    _reject_simulator_env(env)
    tools = fingerprint_tools(env)
    prepared_dir = _unique_plan_dir(root) / f"execute-{provider.provider_id}"
    prepared = _call_runner(root, prepared_dir, provider, "--prepare-only", env, tools["host_cc"])
    raw_preflight = Path(prepared["raw_path"]) if prepared.get("raw_path") else prepared_dir / "raw_runner_prepare.json"
    payload = _read_json(raw_preflight)
    errors = simplepim_runner_schema_errors(payload, provider, mode="prepare", expected_host_cc=tools["host_cc"])
    if prepared["status"] != "prepared" or errors:
        raise ValueError(prepared.get("reason") or errors[0] or "runner prepare failed")
    if require_provider_source(provider, root) != source:
        raise ValueError("provider source changed after qualification preflight")
    run_dir = create_run_dir(
        root,
        "provider_qualification",
        artifact_kind=EVIDENCE_ARTIFACT_KIND,
        route_label=provider.provider_id,
    )
    evidence_preflight = run_dir / "raw_runner_preflight.json"
    shutil.copy2(raw_preflight, evidence_preflight)
    provider_dir = run_dir / "provider"
    command_result = _call_runner(
        root,
        provider_dir,
        provider,
        "--execute",
        env,
        tools["host_cc"],
        timeout=EXECUTE_TIMEOUT_SECONDS,
        expected_preflight=payload,
    )
    raw_source = Path(command_result["raw_path"])
    raw_path = run_dir / "raw_runner_result.json"
    if raw_source.is_file():
        shutil.copy2(raw_source, raw_path)
    result_payload = _read_json(raw_path) if raw_path.is_file() else _failed_payload(provider, command_result)
    qualification = parse_runner_result(
        result_payload,
        provider,
        expected_host_cc=tools["host_cc"],
        expected_preflight=payload,
    )
    status = "qualified" if command_result["returncode"] == 0 and qualification.status == "qualified" else "failed"
    reason = None if status == "qualified" else command_result.get("reason") or qualification.reason
    classification = "canonical_physical_evidence" if not test_hook else "internal_test_non_evidence"
    release = "confirmed" if status == "qualified" and qualification.release_status == "released" else "unconfirmed"
    summary = _summary(
        root,
        catalog,
        provider,
        source,
        tools,
        command_result,
        result_payload,
        qualification,
        status,
        reason,
        evidence_preflight,
        raw_path,
        repository,
        runner_fingerprint,
        classification,
        release,
    )
    result_path = run_dir / "provider_qualification.json"
    write_json(result_path, summary)
    normalized = run_dir / "normalized_records.jsonl"
    write_jsonl(
        normalized,
        [_normalized(provider, qualification, status, reason, raw_path, summary, classification, release)],
    )
    manifest = write_run_manifest(
        run_dir,
        run_kind="provider_qualification",
        suite_id="provider_qualification",
        suite_path=_rel(catalog.path, root),
        artifact_kind=EVIDENCE_ARTIFACT_KIND,
        route_label=provider.provider_id,
        route_id=provider.provider_id,
        execution_scope="physical_hardware",
        evidence_type="qualification",
        normalized_records=normalized.name,
        summary=result_path.name,
        artifact_retention="compact",
        command=shlex.join(command_result["command"]),
        root_dir=root,
    )
    manifest["qualification"] = {"execution_classification": classification, "repository_fingerprint": repository, "runner_fingerprint": runner_fingerprint, "raw_preflight_sha256": sha256_file(evidence_preflight), "raw_result_sha256": sha256_file(raw_path) if raw_path.is_file() else None, "resource_release_status": release}
    manifest_path = run_dir / "run_manifest.json"
    write_json(manifest_path, manifest)
    return ProviderQualificationResult(run_dir, result_path, raw_path, normalized, manifest_path, status)
def _inputs(root_dir: Path, catalog_path: Path) -> tuple[Path, ProviderCatalog]:
    root = resolve_root(root_dir)
    return root, load_provider_catalog(resolve_catalog_path(root, catalog_path))
def _require_canonical_identity(catalog: ProviderCatalog, provider: ProviderSpec) -> None:
    expected = (CANONICAL_CATALOG_ID, CANONICAL_PROVIDER_ID, CANONICAL_RUNNER_PATH, CANONICAL_SOURCE_PATH, CANONICAL_PINNED_COMMIT)
    actual = (catalog.catalog_id, provider.provider_id, provider.runner, provider.source_path, provider.pinned_commit)
    if actual != expected:
        raise ValueError("--execute requires the canonical catalog/provider identity and relative source paths")
def _select(catalog: ProviderCatalog, provider_id: str | None) -> ProviderSpec:
    if provider_id:
        return catalog.get(provider_id)
    candidates = [p for p in catalog.providers if p.executable and p.status == "executable"]
    if len(candidates) != 1:
        raise ValueError("--provider is required unless the catalog has one executable provider")
    return candidates[0]
def _provider_row(root: Path, provider: ProviderSpec) -> dict[str, Any]:
    runner = resolve_provider_path(root, provider.runner, label="provider runner") if provider.runner else None
    return {"id": provider.provider_id, "name": provider.name, "catalog_status": provider.status, "executable": provider.executable, "reason": provider.reason, "runner": provider.runner, "runner_available": bool(runner and runner.is_file()), "runner_contract": {"schema_version": provider.runner_schema_version, "probe_id": provider.probe_id}, "requested_counts": {"dpus": provider.requested_dpus, "tasklets": provider.requested_tasklets}, "lane": provider.lane}
def _call_runner(
    root: Path,
    plan_dir: Path,
    provider: ProviderSpec,
    mode: str,
    env: Mapping[str, str],
    host_cc: Mapping[str, Any],
    *,
    timeout: float = PREPARE_TIMEOUT_SECONDS,
    expected_preflight: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not provider.runner:
        return {"status": "failed", "reason": "provider has no runner"}
    runner = resolve_provider_path(root, provider.runner, label="provider runner")
    plan_dir.mkdir(parents=True, exist_ok=False)
    workdir = plan_dir / "work"
    raw_path = plan_dir / ("raw_runner_prepare.json" if mode == "--prepare-only" else "raw_runner_result.json")
    command = [
        sys.executable,
        str(runner),
        mode,
        "--workdir",
        str(workdir),
        "--json-output",
        str(raw_path),
    ]
    runner_env = dict(env)
    runner_env["HOST_CC"] = str(host_cc["runner_value"])
    completed, invocation_error, cleanup = _run_process(command, root, runner_env, timeout)
    payload = _read_json(raw_path) if raw_path.is_file() else None
    errors = simplepim_runner_schema_errors(
        payload,
        provider,
        mode="prepare" if mode == "--prepare-only" else "execute",
        expected_host_cc=host_cc,
        expected_preflight=expected_preflight,
    )
    status = "prepared" if mode == "--prepare-only" and completed and completed.returncode == 0 and not errors else "failed"
    if mode == "--execute":
        status = "passed" if completed and completed.returncode == 0 and payload and payload.get("status") == "passed" and not errors else "failed"
    reason = invocation_error or (errors[0] if errors else None) or (f"runner exited with return code {completed.returncode}" if completed and completed.returncode else None)
    return {"status": status, "reason": reason, "returncode": completed.returncode if completed else None, "raw_path": str(raw_path), "raw_sha256": sha256_file(raw_path) if raw_path.is_file() else None, "command": command, "environment": {"HOST_CC": runner_env["HOST_CC"]}, "stdout": _snippet(completed.stdout if completed else ""), "stderr": _snippet(completed.stderr if completed else ""), "timeout_cleanup": cleanup, "contract_errors": errors}
def _run_process(command: list[str], cwd: Path, env: Mapping[str, str], timeout: float) -> tuple[subprocess.CompletedProcess[str] | None, str | None, dict[str, Any] | None]:
    try:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=dict(env),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
    except OSError as exc:
        return None, str(exc), None
    try:
        stdout, stderr = process.communicate(timeout=timeout)
        return (
            subprocess.CompletedProcess(command, process.returncode, stdout, stderr),
            None,
            None,
        )
    except subprocess.TimeoutExpired as exc:
        errors: list[str] = []
        output_complete = True
        try:
            os.killpg(process.pid, signal.SIGTERM)
            term = True
        except OSError as error:
            errors.append(str(error))
            term = False
        try:
            stdout, stderr = process.communicate(timeout=2)
        except subprocess.TimeoutExpired as term_timeout:
            try:
                os.killpg(process.pid, signal.SIGKILL)
                killed = True
            except OSError as error:
                errors.append(str(error))
                killed = False
            try:
                stdout, stderr = process.communicate(timeout=2)
            except subprocess.TimeoutExpired as kill_timeout:
                output_complete = False
                stdout = _timeout_text(kill_timeout.output or term_timeout.output or exc.output)
                stderr = _timeout_text(kill_timeout.stderr or term_timeout.stderr or exc.stderr)
                for stream in (getattr(process, "stdout", None), getattr(process, "stderr", None)):
                    if stream is not None:
                        stream.close()
        cleanup = {
            "attempted": True,
            "verified": False,
            "verification": "unavailable",
            "output_capture_complete": output_complete,
            "sigterm_sent": term,
            "sigkill_sent": killed if "killed" in locals() else False,
            "process_exited": process.returncode is not None,
            "signal_errors": errors,
        }
        return (
            subprocess.CompletedProcess(
                command,
                process.returncode,
                stdout or _timeout_text(exc.output),
                stderr or _timeout_text(exc.stderr),
            ),
            f"runner timed out after {timeout:g} seconds",
            cleanup,
        )
def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None
def _summary(
    root: Path,
    catalog: ProviderCatalog,
    provider: ProviderSpec,
    source: Mapping[str, Any],
    tools: Mapping[str, Any],
    executed: Mapping[str, Any],
    payload: Mapping[str, Any],
    qualification: RunnerQualification,
    status: str,
    reason: str | None,
    raw_preflight: Path,
    raw_result: Path,
    repository: Mapping[str, Any],
    runner_fingerprint: Mapping[str, Any],
    classification: str,
    release: str,
) -> dict[str, Any]:
    return {"schema_version": RESULT_SCHEMA_VERSION, "status": status, "qualified": status == "qualified", "execution_classification": classification, "canonical_evidence": classification == "canonical_physical_evidence", "resource_release_status": release, "provider_id": provider.provider_id, "provider_name": provider.name, "catalog": {"id": catalog.catalog_id, "path": _rel(catalog.path, root), "sha256": sha256_file(catalog.path)}, "runner": {"command": executed.get("command"), "returncode": executed.get("returncode"), "environment": executed.get("environment"), "stdout": executed.get("stdout"), "stderr": executed.get("stderr"), "timeout_cleanup": executed.get("timeout_cleanup")}, "runner_fingerprint": dict(runner_fingerprint), "repository_fingerprint": dict(repository), "raw_runner_preflight": {"path": raw_preflight.name, "sha256": sha256_file(raw_preflight) if raw_preflight.is_file() else None}, "raw_runner_result": {"path": raw_result.name if raw_result.is_file() else None, "sha256": sha256_file(raw_result) if raw_result.is_file() else None}, "source_fingerprint": dict(source), "tool_fingerprints": dict(tools), "requested_counts": {"dpus": provider.requested_dpus, "tasklets": provider.requested_tasklets}, "observed_counts": {"dpus": qualification.observed_dpus, "tasklets": qualification.observed_tasklets}, "configured_tasklets_per_dpu": qualification.configured_tasklets, "hardware_preflight_verified": qualification.hardware_preflight_verified, "target": qualification.target, "native_execution": qualification.native_execution, "validation_performed": qualification.validation_performed, "exact_validation": qualification.exact_validation, "release_status": qualification.release_status, "fallback_used": qualification.fallback_used, "simulator_kernel_executed": qualification.simulator_kernel_executed, "runner_status": qualification.runner_status, "runner_contract_errors": qualification.contract_errors, "failure_reason": reason, "raw_payload": dict(payload), "normalized_records": "normalized_records.jsonl", "run_manifest": "run_manifest.json"}
def _normalized(
    provider: ProviderSpec,
    qualification: RunnerQualification,
    status: str,
    reason: str | None,
    raw_path: Path,
    summary: Mapping[str, Any],
    classification: str,
    release: str,
) -> dict[str, Any]:
    return {"schema_version": NORMALIZED_SCHEMA_VERSION, "provider_id": provider.provider_id, "probe_id": provider.probe_id, "status": status, "execution_classification": classification, "resource_release_status": release, "requested_dpu_count": provider.requested_dpus, "observed_dpu_count": qualification.observed_dpus, "requested_tasklet_count": provider.requested_tasklets, "configured_tasklet_count": qualification.configured_tasklets, "observed_tasklet_count": qualification.observed_tasklets, "target": qualification.target, "validation_status": "exact" if qualification.exact_validation else "failed", "release_status": qualification.release_status, "raw_runner_result_sha256": summary["raw_runner_result"]["sha256"], "reason": reason or qualification.reason}
def _failed_payload(provider: ProviderSpec, command: Mapping[str, Any]) -> dict[str, Any]:
    zero = "0" * 64
    return {"schema_version": provider.runner_schema_version, "provider_id": provider.provider_id, "probe_id": provider.probe_id, "status": "failed", "target": None, "target_observed": None, "requested_dpu_count": provider.requested_dpus, "observed_dpu_count": None, "configured_tasklets_per_dpu": provider.requested_tasklets, "observed_tasklets_per_dpu": None, "hardware_preflight_verified": False, "device_evidence": [], "native_execution": False, "validation_performed": False, "exact_validation": False, "fallback": False, "simulator_kernel_executed": False, "release_status": "unknown", "backend_profile": "backend=hw", "source_hash": zero, "source_hashes": {"combined_sha256": zero}, "command_fingerprint": zero, "effective_compilers": {}, "staged_patch": {"path": "patches/simplepim-map-unroll-rest.patch"}, "binary_hashes": {}, "input_hashes": {}, "output_hash": None, "logical_transfer_bytes": {}, "payload_sizes_8_byte_aligned": True, "physical_transfer_bytes_available": False, "physical_transfer_bytes": None, "timing": {}, "failure_stage": "runner", "reason": command.get("reason") or "runner failed"}
def _reject_simulator_env(env: Mapping[str, str]) -> None:
    for key in SIMULATOR_ENV_KEYS:
        value = env.get(key)
        if value and any(alias in value.lower() for alias in SIMULATOR_ALIASES) or (value and "simulator" in value.lower()):
            raise ValueError(f"{key}={value} is forbidden for physical provider qualification")
def _unique_plan_dir(root: Path) -> Path:
    parent = root / PLAN_ROOT
    parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
    path = parent / stamp
    index = 1
    while path.exists():
        path = parent / f"{stamp}_{index:02d}"
        index += 1
    path.mkdir()
    return path.resolve()
def _rel(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)
def _snippet(value: str) -> str | None:
    text = value.strip()
    return text[-2000:] if text else None
def _timeout_text(value: str | bytes | None) -> str:
    return value.decode(errors="replace") if isinstance(value, bytes) else value or ""
