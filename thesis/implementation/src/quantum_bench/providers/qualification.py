"""Strict metadata and result checks for the SimplePIM M1 qualification."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
from typing import Any, Mapping, Sequence

import yaml


CATALOG_SCHEMA_VERSION = "provider_qualification_catalog_v1"
SIMPLEPIM_PROVIDER_ID = "simplepim"
SIMPLEPIM_RUNNER_SCHEMA_VERSION = "simplepim_provider_qualification_v1"
SIMPLEPIM_PROBE_ID = "simplepim_va_map_zip_v1"
PHYSICAL_TARGET = "physical_hardware"
SIMPLEPIM_HOST_SCHEMA_VERSION = "simplepim_qualification_host_v2"
SIMPLEPIM_PATCH_PATH = "patches/simplepim-map-unroll-rest.patch"
SIMPLEPIM_LOGICAL_INPUT_BYTES = 2048
SIMPLEPIM_LOGICAL_OUTPUT_BYTES = 1024
SIMPLEPIM_LOGICAL_TOTAL_BYTES = 3072

_RESULT_REQUIRED_KEYS = frozenset(
    {
        "schema_version",
        "provider_id",
        "probe_id",
        "status",
        "target",
        "target_observed",
        "requested_dpu_count",
        "observed_dpu_count",
        "configured_tasklets_per_dpu",
        "observed_tasklets_per_dpu",
        "hardware_preflight_verified",
        "device_evidence",
        "native_execution",
        "validation_performed",
        "exact_validation",
        "fallback",
        "simulator_kernel_executed",
        "release_status",
        "backend_profile",
        "source_hash",
        "source_hashes",
        "command_fingerprint",
        "effective_compilers",
        "staged_patch",
        "binary_hashes",
        "input_hashes",
        "output_hash",
        "logical_transfer_bytes",
        "payload_sizes_8_byte_aligned",
        "physical_transfer_bytes_available",
        "physical_transfer_bytes",
        "timing",
        "failure_stage",
        "reason",
    }
)
_RESULT_OPTIONAL_KEYS = frozenset(
    {
        "build",
        "commands",
        "execution",
        "host_result",
        "timeout_cleanup",
    }
)
_SOURCE_HASH_KEYS = frozenset(
    {
        "combined_sha256",
        "owned_qualification_sha256",
        "upstream_library_sha256",
        "upstream_map_processing_sha256",
        "patch_sha256",
        "staged_source_before_patch_sha256",
        "staged_source_after_patch_sha256",
        "staged_patch_sha256",
        "upstream_submodule",
    }
)
_PATCH_KEYS = frozenset(
    {
        "path",
        "sha256",
        "staged_sha256",
        "applied",
        "replacement_count",
        "command_fingerprint",
        "staged_source_before_sha256",
        "staged_source_after_sha256",
        "staged_target_sha256",
    }
)
_COMPILER_ROLES = {
    "host_cc": "gcc",
    "dpu_cc": "dpu-upmem-dpurte-clang",
}
_COMPILER_IDENTITY_KEYS = frozenset({"command", "available", "path", "sha256"})
_BINARY_NAMES = frozenset({"host", "dpu_init_binary", "dpu_zip", "dpu_map_va_funcs"})
_INPUT_NAMES = frozenset({"a_u32", "b_u32"})
_DEVICE_REQUIRED_KEYS = frozenset(
    {
        "path",
        "exists",
        "character_device",
        "readable",
        "writable",
        "sysfs_path",
        "sysfs_exists",
    }
)
_DEVICE_OPTIONAL_KEYS = frozenset({"error"})
_HOST_RESULT_KEYS = frozenset(
    {
        "schema_version",
        "provider_id",
        "probe_id",
        "status",
        "backend_profile",
        "requested_dpu_count",
        "observed_dpu_count",
        "configured_tasklets_per_dpu",
        "observed_tasklets_per_dpu",
        "native_run_completed",
        "validation_performed",
        "host_exact_validation",
        "fallback",
        "release_status",
        "logical_input_bytes",
        "logical_output_bytes",
        "physical_transfer_bytes_available",
        "physical_transfer_bytes",
        "timing",
        "failure_stage",
        "reason",
    }
)

Json = dict[str, Any]


@dataclass(frozen=True)
class ProviderSpec:
    provider_id: str
    name: str
    status: str
    reason: str | None
    executable: bool
    runner: str | None
    source_path: str | None
    pinned_commit: str | None
    requested_dpus: int | None
    requested_tasklets: int | None
    runner_schema_version: str | None
    probe_id: str | None
    lane: Json


@dataclass(frozen=True)
class ProviderCatalog:
    catalog_id: str
    schema_version: str
    providers: tuple[ProviderSpec, ...]
    path: Path

    def get(self, provider_id: str) -> ProviderSpec:
        for provider in self.providers:
            if provider.provider_id == provider_id:
                return provider
        known = ", ".join(provider.provider_id for provider in self.providers)
        raise ValueError(f"unknown provider {provider_id!r}; choose one of: {known}")


@dataclass(frozen=True)
class RunnerQualification:
    status: str
    reason: str
    contract_errors: tuple[str, ...]
    runner_status: str | None
    hardware_preflight_verified: bool | None
    target: str | None
    observed_dpus: int | None
    configured_tasklets: int | None
    observed_tasklets: int | None
    native_execution: bool | None
    validation_performed: bool | None
    exact_validation: bool | None
    release_status: str | None
    fallback_used: bool | None
    simulator_kernel_executed: bool | None


def resolve_root(root_dir: Path) -> Path:
    root = root_dir.expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"qualification root is not a directory: {root}")
    return root


def resolve_catalog_path(root_dir: Path, catalog_path: Path) -> Path:
    root = resolve_root(root_dir)
    candidate = catalog_path.expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    resolved = candidate.resolve()
    if not resolved.is_file():
        raise ValueError(f"provider catalog is not a file: {resolved}")
    return resolved


def resolve_provider_path(root_dir: Path, configured_path: str, *, label: str) -> Path:
    root = resolve_root(root_dir)
    candidate = Path(configured_path).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            f"{label} must remain under qualification root: {resolved}"
        ) from exc
    return resolved


def load_provider_catalog(path: Path) -> ProviderCatalog:
    resolved = path.expanduser().resolve()
    try:
        data = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"cannot read provider catalog {resolved}: {exc}") from exc
    if not isinstance(data, Mapping):
        raise ValueError("provider catalog must be a mapping")
    schema_version = _required_string(data, "schema_version")
    if schema_version != CATALOG_SCHEMA_VERSION:
        raise ValueError(f"unsupported provider catalog schema: {schema_version}")
    raw_providers = data.get("providers")
    if not isinstance(raw_providers, list) or not raw_providers:
        raise ValueError("provider catalog must contain a non-empty providers list")

    providers: list[ProviderSpec] = []
    seen: set[str] = set()
    for raw in raw_providers:
        if not isinstance(raw, Mapping):
            raise ValueError("each provider catalog entry must be a mapping")
        provider_id = _required_string(raw, "id")
        if provider_id in seen:
            raise ValueError(f"duplicate provider id: {provider_id}")
        seen.add(provider_id)
        status = _required_string(raw, "status")
        if status not in {"executable", "blocked"}:
            raise ValueError(
                f"provider {provider_id} has unsupported status {status!r}"
            )
        executable = _required_bool(raw, "executable")
        source = raw.get("source")
        source = source if isinstance(source, Mapping) else {}
        hardware = raw.get("hardware")
        hardware = hardware if isinstance(hardware, Mapping) else {}
        contract = raw.get("runner_contract")
        contract = contract if isinstance(contract, Mapping) else {}
        lane = raw.get("lane")
        provider = ProviderSpec(
            provider_id=provider_id,
            name=_optional_string(raw.get("name")) or provider_id,
            status=status,
            reason=_optional_string(raw.get("reason")),
            executable=executable,
            runner=_optional_string(raw.get("runner")),
            source_path=_optional_string(source.get("path")),
            pinned_commit=_optional_string(source.get("pinned_commit")),
            requested_dpus=_optional_positive_int(hardware.get("requested_dpus")),
            requested_tasklets=_optional_positive_int(
                hardware.get("requested_tasklets")
            ),
            runner_schema_version=_optional_string(contract.get("schema_version")),
            probe_id=_optional_string(contract.get("probe_id")),
            lane=dict(lane) if isinstance(lane, Mapping) else {},
        )
        _validate_provider_spec(provider)
        providers.append(provider)

    return ProviderCatalog(
        catalog_id=_required_string(data, "catalog_id"),
        schema_version=schema_version,
        providers=tuple(sorted(providers, key=lambda item: item.provider_id)),
        path=resolved,
    )


def provider_source_fingerprint(provider: ProviderSpec, root_dir: Path) -> Json:
    if not provider.source_path:
        return {
            "path": None,
            "pinned_commit": provider.pinned_commit,
            "head_commit": None,
            "commit_matches_pin": False,
            "clean": False,
            "git_root_matches_source": False,
            "status_sha256": None,
        }
    source = resolve_provider_path(
        root_dir, provider.source_path, label="provider source"
    )
    head = _git_output(source, ("rev-parse", "HEAD"))
    git_root = _git_output(source, ("rev-parse", "--show-toplevel"))
    status = _git_output(
        source,
        ("status", "--porcelain=v1", "--untracked-files=all"),
        allow_empty=True,
    )
    root_matches = bool(git_root and Path(git_root).resolve() == source)
    return {
        "path": _relative_path(source, resolve_root(root_dir)),
        "pinned_commit": provider.pinned_commit,
        "head_commit": head,
        "commit_matches_pin": bool(head and head == provider.pinned_commit),
        "clean": status == "",
        "git_root_matches_source": root_matches,
        "status_sha256": _hash_text(status) if status is not None else None,
    }


def source_gate_failure(
    provider: ProviderSpec, fingerprint: Mapping[str, Any]
) -> str | None:
    if not provider.executable or provider.status != "executable":
        return None
    if not fingerprint.get("head_commit"):
        return (
            f"provider {provider.provider_id} source is not an available Git checkout"
        )
    if fingerprint.get("git_root_matches_source") is not True:
        return (
            f"provider {provider.provider_id} source path is not the Git checkout root"
        )
    if fingerprint.get("commit_matches_pin") is not True:
        return (
            f"provider {provider.provider_id} source HEAD "
            f"{fingerprint.get('head_commit')} does not match pinned commit {provider.pinned_commit}"
        )
    if fingerprint.get("clean") is not True:
        return f"provider {provider.provider_id} source worktree is not clean"
    return None


def require_provider_source(provider: ProviderSpec, root_dir: Path) -> Json:
    fingerprint = provider_source_fingerprint(provider, root_dir)
    failure = source_gate_failure(provider, fingerprint)
    if failure:
        raise ValueError(failure)
    return fingerprint


def qualification_repository_fingerprint(
    root_dir: Path,
    configured_paths: Sequence[str],
) -> Json:
    root = resolve_root(root_dir)
    git_root_text = _git_output(root, ("rev-parse", "--show-toplevel"))
    if git_root_text is None:
        return {
            "git_root": None,
            "head_commit": None,
            "paths": list(configured_paths),
            "tracked_files": [],
            "untracked_files": [],
            "actual_files": [],
            "all_files_tracked": False,
            "clean": False,
            "status_sha256": None,
            "content_sha256": None,
        }
    git_root = Path(git_root_text).resolve()
    pathspecs: list[str] = []
    for configured_path in configured_paths:
        resolved = resolve_provider_path(
            root,
            configured_path,
            label="qualification repository source",
        )
        if not resolved.exists():
            raise ValueError(f"qualification repository source is missing: {resolved}")
        pathspec = resolved.relative_to(git_root).as_posix()
        pathspecs.append(pathspec)

    tracked = _git_bytes(git_root, ("ls-files", "-z", "--", *pathspecs))
    untracked = _git_bytes(
        git_root,
        ("ls-files", "-z", "--others", "--exclude-standard", "--", *pathspecs),
    )
    status = _git_bytes(
        git_root,
        ("status", "--porcelain=v1", "-z", "--untracked-files=all", "--", *pathspecs),
    )
    tracked_files = (
        sorted(item.decode("utf-8") for item in tracked.split(b"\0") if item)
        if tracked is not None
        else []
    )
    untracked_files = (
        sorted(item.decode("utf-8") for item in untracked.split(b"\0") if item)
        if untracked is not None
        else []
    )
    actual_sorted = sorted(
        relative
        for relative in set(tracked_files) | set(untracked_files)
        if (git_root / relative).is_file() or (git_root / relative).is_symlink()
    )
    content = hashlib.sha256()
    for relative in actual_sorted:
        path = git_root / relative
        content.update(relative.encode("utf-8"))
        content.update(b"\0")
        content.update(sha256_file(path).encode("ascii"))
        content.update(b"\n")
    return {
        "git_root": str(git_root),
        "head_commit": _git_output(git_root, ("rev-parse", "HEAD")),
        "paths": list(configured_paths),
        "tracked_files": tracked_files,
        "untracked_files": untracked_files,
        "actual_files": actual_sorted,
        "all_files_tracked": (
            untracked_files == []
            and tracked is not None
            and untracked is not None
            and all(
                (git_root / relative).is_file() or (git_root / relative).is_symlink()
                for relative in tracked_files
            )
        ),
        "clean": status == b"",
        "status_sha256": (
            hashlib.sha256(status).hexdigest() if status is not None else None
        ),
        "content_sha256": content.hexdigest(),
    }


def require_qualification_repository_source(
    root_dir: Path,
    configured_paths: Sequence[str],
) -> Json:
    fingerprint = qualification_repository_fingerprint(root_dir, configured_paths)
    if fingerprint["head_commit"] is None:
        raise ValueError("qualification repository is not a Git checkout")
    if fingerprint["all_files_tracked"] is not True:
        raise ValueError(
            "qualification repository source must be fully tracked before --execute"
        )
    if fingerprint["clean"] is not True:
        raise ValueError(
            "qualification repository source must be clean before --execute"
        )
    return fingerprint


def resolve_host_cc(environment: Mapping[str, str] | None = None) -> Json:
    env = environment or {}
    configured_value = env.get("HOST_CC")
    configured = configured_value.strip() if configured_value is not None else "gcc"
    if not configured:
        raise ValueError("HOST_CC must name one compiler executable")
    try:
        command = shlex.split(configured)
    except ValueError as exc:
        raise ValueError(f"HOST_CC is not a valid compiler selector: {exc}") from exc
    if len(command) != 1:
        raise ValueError("HOST_CC must name one compiler executable without arguments")
    resolved_raw = shutil.which(command[0], path=env.get("PATH"))
    if resolved_raw is None:
        raise ValueError(f"HOST_CC compiler is unavailable: {command[0]}")
    resolved = Path(resolved_raw).resolve()
    if not resolved.is_file():
        raise ValueError(f"HOST_CC compiler is not a file: {resolved}")
    runner_value = "gcc" if configured_value is None else str(resolved)
    return {
        "available": True,
        "configured": configured,
        "runner_value": runner_value,
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "version": _tool_version(resolved),
    }


def fingerprint_tools(environment: Mapping[str, str] | None = None) -> Json:
    env = environment or {}
    search_path = env.get("PATH")
    configured = (
        ("python", sys.executable),
        ("make", shutil.which("make", path=search_path)),
        (
            "dpu-upmem-dpurte-clang",
            shutil.which("dpu-upmem-dpurte-clang", path=search_path),
        ),
        ("dpu-pkg-config", shutil.which("dpu-pkg-config", path=search_path)),
    )
    result: Json = {}
    for name, raw_path in configured:
        path = Path(raw_path).resolve() if raw_path else None
        version = _tool_version(path) if path else None
        result[name] = {
            "available": bool(path and path.is_file()),
            "path": str(path) if path else None,
            "sha256": sha256_file(path) if path and path.is_file() else None,
            "version": version,
        }
    result["host_cc"] = resolve_host_cc(env)
    return result


def fingerprint_catalog(catalog: ProviderCatalog, root_dir: Path) -> Json:
    root = resolve_root(root_dir)
    providers: Json = {}
    for provider in catalog.providers:
        runner = (
            resolve_provider_path(root, provider.runner, label="provider runner")
            if provider.runner
            else None
        )
        providers[provider.provider_id] = {
            "runner": {
                "path": _relative_path(runner, root) if runner else None,
                "exists": bool(runner and runner.is_file()),
                "sha256": sha256_file(runner) if runner and runner.is_file() else None,
            },
            "source": provider_source_fingerprint(provider, root)
            if provider.source_path
            else None,
        }
    return {
        "catalog_sha256": sha256_file(catalog.path),
        "benchmark_source_commit": _git_output(root, ("rev-parse", "HEAD")),
        "providers": providers,
    }


def simplepim_runner_schema_errors(
    payload: Mapping[str, Any],
    provider: ProviderSpec,
    *,
    mode: str,
    expected_host_cc: Mapping[str, Any] | None = None,
    expected_dpu_cc: Mapping[str, Any] | None = None,
    expected_preflight: Mapping[str, Any] | None = None,
) -> tuple[str, ...]:
    if mode not in {"prepare", "execute"}:
        raise ValueError(f"unsupported SimplePIM runner mode: {mode}")
    errors: list[str] = []
    allowed_optional = (
        frozenset({"commands"})
        if mode == "prepare"
        else frozenset(
            {
                "commands",
                "build",
                "execution",
                "host_result",
                "timeout_cleanup",
            }
        )
    )
    _check_exact_keys(
        payload,
        _RESULT_REQUIRED_KEYS,
        allowed_optional,
        "runner result",
        errors,
    )
    _expect_exact_string(
        payload,
        "schema_version",
        provider.runner_schema_version or SIMPLEPIM_RUNNER_SCHEMA_VERSION,
        errors,
    )
    _expect_exact_string(payload, "provider_id", provider.provider_id, errors)
    _expect_exact_string(
        payload, "probe_id", provider.probe_id or SIMPLEPIM_PROBE_ID, errors
    )

    status = _strict_string(payload, "status", errors)
    allowed_statuses = {"prepared"} if mode == "prepare" else {"passed", "failed"}
    if status is not None and status not in allowed_statuses:
        errors.append(
            f"status must be one of {sorted(allowed_statuses)!r} in {mode} mode"
        )
    target = _strict_nullable_string(payload, "target", errors)
    target_observed = _strict_nullable_string(payload, "target_observed", errors)
    if target not in {None, PHYSICAL_TARGET}:
        errors.append(f"target must be null or {PHYSICAL_TARGET!r}")
    if target_observed != target:
        errors.append("target_observed must equal target")

    requested_dpus = _strict_int(payload, "requested_dpu_count", errors)
    observed_dpus = _strict_nullable_int(payload, "observed_dpu_count", errors)
    configured_tasklets = _strict_int(payload, "configured_tasklets_per_dpu", errors)
    if payload.get("observed_tasklets_per_dpu") is not None:
        errors.append("observed_tasklets_per_dpu must be null")
    if (
        provider.requested_dpus is not None
        and requested_dpus != provider.requested_dpus
    ):
        errors.append(
            f"requested_dpu_count must equal catalog request {provider.requested_dpus}"
        )
    if (
        provider.requested_tasklets is not None
        and configured_tasklets != provider.requested_tasklets
    ):
        errors.append(
            "configured_tasklets_per_dpu must equal requested count "
            f"{provider.requested_tasklets}"
        )

    hardware_preflight = _strict_bool(payload, "hardware_preflight_verified", errors)
    native_execution = _strict_bool(payload, "native_execution", errors)
    validation_performed = _strict_bool(payload, "validation_performed", errors)
    exact_validation = _strict_bool(payload, "exact_validation", errors)
    fallback = _strict_bool(payload, "fallback", errors)
    simulator = _strict_bool(payload, "simulator_kernel_executed", errors)
    payload_aligned = _strict_bool(payload, "payload_sizes_8_byte_aligned", errors)
    physical_bytes_available = _strict_bool(
        payload, "physical_transfer_bytes_available", errors
    )
    if fallback is not False:
        errors.append("fallback must be false")
    if simulator is not False:
        errors.append("simulator_kernel_executed must be false")
    if payload_aligned is not True:
        errors.append("payload_sizes_8_byte_aligned must be true")
    if physical_bytes_available is not False:
        errors.append("physical_transfer_bytes_available must be false")
    if payload.get("physical_transfer_bytes") is not None:
        errors.append("physical_transfer_bytes must be null")

    backend_profile = _strict_string(payload, "backend_profile", errors)
    if backend_profile != "backend=hw":
        errors.append("backend_profile must be 'backend=hw'")
    release_status = _strict_string(payload, "release_status", errors)
    if release_status not in {"not_attempted", "released", "failed", "unknown"}:
        errors.append("release_status is invalid")
    failure_stage = _strict_nullable_string(payload, "failure_stage", errors)
    reason = _strict_nullable_string(payload, "reason", errors)
    if status != "passed" and (reason is None or not reason):
        errors.append("non-passed reason must be a non-empty string")

    device_evidence = _validate_device_evidence(payload, errors)
    source_hash = _validate_nullable_sha256(payload, "source_hash", errors)
    source_hashes = _validate_source_hashes(payload, status, errors)
    if source_hashes is not None and source_hash != source_hashes.get(
        "combined_sha256"
    ):
        errors.append("source_hash must equal source_hashes.combined_sha256")
    command_fingerprint = _validate_nullable_sha256(
        payload, "command_fingerprint", errors
    )
    effective_compilers = _validate_effective_compilers(payload, errors)
    if expected_host_cc is not None:
        _validate_expected_compiler(
            effective_compilers,
            "host_cc",
            expected_host_cc,
            errors,
        )
    if expected_dpu_cc is not None:
        _validate_expected_compiler(
            effective_compilers,
            "dpu_cc",
            expected_dpu_cc,
            errors,
        )
    if expected_preflight is not None:
        _validate_preflight_binding(payload, expected_preflight, errors)
    staged_patch = _validate_staged_patch(payload, errors)
    if (
        source_hashes is not None
        and staged_patch is not None
        and "patch_sha256" in source_hashes
        and source_hashes.get("patch_sha256") != staged_patch.get("sha256")
    ):
        errors.append("source and staged patch SHA-256 values must agree")
    if (
        source_hashes is not None
        and staged_patch is not None
        and staged_patch.get("applied") is True
    ):
        staged_links = {
            "staged_source_before_patch_sha256": "staged_source_before_sha256",
            "staged_source_after_patch_sha256": "staged_source_after_sha256",
            "staged_patch_sha256": "staged_sha256",
        }
        for source_key, patch_key in staged_links.items():
            if source_hashes.get(source_key) != staged_patch.get(patch_key):
                errors.append(
                    f"source_hashes.{source_key} must equal staged_patch.{patch_key}"
                )
        if source_hashes.get("upstream_map_processing_sha256") != staged_patch.get(
            "staged_target_sha256"
        ):
            errors.append(
                "source_hashes.upstream_map_processing_sha256 must equal "
                "staged_patch.staged_target_sha256"
            )
    binary_hashes = _validate_artifact_hashes(
        payload, "binary_hashes", _BINARY_NAMES, errors
    )
    input_hashes = _validate_artifact_hashes(
        payload,
        "input_hashes",
        _INPUT_NAMES,
        errors,
        allow_null=status == "failed",
    )
    output_hash = _validate_nullable_sha256(payload, "output_hash", errors)
    _validate_logical_bytes(payload, errors)
    _validate_timing(payload.get("timing"), "timing", errors)
    if status in {"prepared", "passed"}:
        if not _is_sha256(source_hash):
            errors.append(f"{status} source_hash must be SHA-256")
        if not _is_sha256(command_fingerprint):
            errors.append(f"{status} command_fingerprint must be SHA-256")
        if staged_patch is not None and not _is_sha256(staged_patch.get("sha256")):
            errors.append(f"{status} staged_patch.sha256 must be SHA-256")

    commands = payload.get("commands")
    if commands is not None:
        _validate_commands(commands, errors)
        _validate_command_compilers(commands, effective_compilers, errors)
        if command_fingerprint is not None and command_fingerprint != _hash_json(
            commands
        ):
            errors.append("command_fingerprint does not match commands")
        if (
            isinstance(staged_patch, Mapping)
            and isinstance(commands, Mapping)
            and staged_patch.get("command_fingerprint")
            != _hash_json(commands.get("patch"))
        ):
            errors.append(
                "staged_patch.command_fingerprint does not match commands.patch"
            )
    build_evidence = payload.get("build")
    execution_evidence = payload.get("execution")
    if build_evidence is not None:
        expected = commands.get("build") if isinstance(commands, Mapping) else None
        _validate_command_evidence(
            build_evidence,
            expected,
            "build",
            errors,
        )
    if execution_evidence is not None:
        expected = commands.get("run") if isinstance(commands, Mapping) else None
        _validate_command_evidence(
            execution_evidence,
            expected,
            "execution",
            errors,
        )
    if "host_result" in payload:
        _validate_host_result(payload["host_result"], provider, errors)
    if "timeout_cleanup" in payload:
        _validate_timeout_cleanup(payload["timeout_cleanup"], "timeout_cleanup", errors)

    if target == PHYSICAL_TARGET and not (
        hardware_preflight is True and native_execution is True
    ):
        errors.append(
            "physical target requires hardware preflight and native execution"
        )

    if status == "prepared":
        prepared_invariants = (
            (commands is not None, "prepared result must contain commands"),
            (target is None, "prepared target must be null"),
            (observed_dpus is None, "prepared observed_dpu_count must be null"),
            (
                hardware_preflight is False,
                "prepared hardware_preflight_verified must be false",
            ),
            (device_evidence == [], "prepared device_evidence must be empty"),
            (native_execution is False, "prepared native_execution must be false"),
            (
                validation_performed is False,
                "prepared validation_performed must be false",
            ),
            (exact_validation is False, "prepared exact_validation must be false"),
            (
                release_status == "not_attempted",
                "prepared release_status must be 'not_attempted'",
            ),
            (
                staged_patch is not None and staged_patch.get("applied") is False,
                "prepared staged_patch.applied must be false",
            ),
            (
                staged_patch is not None and staged_patch.get("replacement_count") == 0,
                "prepared staged_patch.replacement_count must be zero",
            ),
            (
                staged_patch is not None
                and staged_patch.get("staged_target_sha256") is None,
                "prepared staged target hash must be null",
            ),
            (
                staged_patch is not None and staged_patch.get("staged_sha256") is None,
                "prepared copied patch hash must be null",
            ),
            (
                staged_patch is not None
                and staged_patch.get("staged_source_before_sha256") is None
                and staged_patch.get("staged_source_after_sha256") is None,
                "prepared staged source hashes must be null",
            ),
            (
                source_hashes is not None
                and source_hashes.get("staged_source_before_patch_sha256") is None
                and source_hashes.get("staged_source_after_patch_sha256") is None
                and source_hashes.get("staged_patch_sha256") is None,
                "prepared source staging hashes must be null",
            ),
            (binary_hashes == {}, "prepared binary_hashes must be empty"),
            (input_hashes == {}, "prepared input_hashes must be empty"),
            (output_hash is None, "prepared output_hash must be null"),
            (payload.get("timing") == {}, "prepared timing must be empty"),
            (failure_stage is None, "prepared failure_stage must be null"),
        )
        errors.extend(message for valid, message in prepared_invariants if not valid)

    if status == "passed":
        passed_invariants = (
            (target == PHYSICAL_TARGET, f"passed target must be {PHYSICAL_TARGET!r}"),
            (
                observed_dpus == provider.requested_dpus,
                "passed observed_dpu_count must equal the catalog request",
            ),
            (
                hardware_preflight is True,
                "passed hardware_preflight_verified must be true",
            ),
            (
                _device_evidence_proves_access(device_evidence),
                "passed device_evidence must prove accessible physical hardware",
            ),
            (native_execution is True, "passed native_execution must be true"),
            (
                validation_performed is True,
                "passed validation_performed must be true",
            ),
            (exact_validation is True, "passed exact_validation must be true"),
            (
                release_status == "released",
                "passed release_status must be 'released'",
            ),
            (
                staged_patch is not None and staged_patch.get("applied") is True,
                "passed staged_patch.applied must be true",
            ),
            (
                staged_patch is not None and staged_patch.get("replacement_count") == 2,
                "passed staged_patch.replacement_count must equal two",
            ),
            (
                staged_patch is not None
                and _is_sha256(staged_patch.get("staged_target_sha256")),
                "passed staged target hash must be SHA-256",
            ),
            (
                staged_patch is not None
                and staged_patch.get("staged_sha256") == staged_patch.get("sha256"),
                "passed copied patch hash must equal the tracked patch hash",
            ),
            (
                staged_patch is not None
                and _is_sha256(staged_patch.get("staged_source_before_sha256"))
                and _is_sha256(staged_patch.get("staged_source_after_sha256"))
                and staged_patch.get("staged_source_before_sha256")
                != staged_patch.get("staged_source_after_sha256"),
                "passed staged source hashes must prove a source change",
            ),
            (
                _compiler_available(effective_compilers, "host_cc"),
                "passed host compiler must be available",
            ),
            (
                _compiler_available(effective_compilers, "dpu_cc"),
                "passed DPU compiler must be available",
            ),
            (
                binary_hashes is not None and set(binary_hashes) == _BINARY_NAMES,
                "passed binary_hashes must contain the four qualification binaries",
            ),
            (
                binary_hashes is not None
                and all(_is_sha256(value) for value in binary_hashes.values()),
                "passed binary hashes must all be SHA-256",
            ),
            (
                input_hashes is not None and set(input_hashes) == _INPUT_NAMES,
                "passed input_hashes must contain a_u32 and b_u32",
            ),
            (
                input_hashes is not None
                and all(_is_sha256(value) for value in input_hashes.values()),
                "passed input hashes must all be SHA-256",
            ),
            (_is_sha256(output_hash), "passed output_hash must be SHA-256"),
            (failure_stage is None, "passed failure_stage must be null"),
            (reason is None, "passed reason must be null"),
            (commands is not None, "passed result must contain commands"),
            (
                _command_evidence_passed(build_evidence),
                "passed result must contain successful build evidence",
            ),
            (
                _command_evidence_passed(execution_evidence),
                "passed result must contain successful execution evidence",
            ),
            ("host_result" in payload, "passed result must contain host_result"),
            (
                _host_result_matches_pass(
                    payload.get("host_result"),
                    observed_dpus,
                    configured_tasklets,
                    release_status,
                ),
                "passed host_result must prove native validation and release",
            ),
            (
                source_hashes is not None
                and staged_patch is not None
                and source_hash
                == source_hashes.get("combined_sha256")
                == source_hashes.get("staged_source_after_patch_sha256")
                == staged_patch.get("staged_source_after_sha256"),
                "passed source hash must bind to the staged source after patch",
            ),
        )
        errors.extend(message for valid, message in passed_invariants if not valid)

    if status == "failed":
        if failure_stage is None or not failure_stage:
            errors.append("failed result must name failure_stage")
        if target == PHYSICAL_TARGET and observed_dpus is None:
            errors.append("physical failed result must retain observed_dpu_count")

    return tuple(errors)


def parse_runner_result(
    payload: Mapping[str, Any],
    provider: ProviderSpec,
    *,
    expected_host_cc: Mapping[str, Any] | None = None,
    expected_dpu_cc: Mapping[str, Any] | None = None,
    expected_preflight: Mapping[str, Any] | None = None,
) -> RunnerQualification:
    errors = list(
        simplepim_runner_schema_errors(
            payload,
            provider,
            mode="execute",
            expected_host_cc=expected_host_cc,
            expected_dpu_cc=expected_dpu_cc,
            expected_preflight=expected_preflight,
        )
    )
    runner_status = _strict_string(payload, "status", errors)
    hardware_preflight = _strict_bool(payload, "hardware_preflight_verified", errors)
    target = _strict_string(payload, "target", errors)
    target_observed = _strict_string(payload, "target_observed", errors)
    requested_dpus = _strict_int(payload, "requested_dpu_count", errors)
    observed_dpus = _strict_int(payload, "observed_dpu_count", errors)
    configured_tasklets = _strict_int(payload, "configured_tasklets_per_dpu", errors)
    observed_tasklets = _strict_nullable_int(
        payload, "observed_tasklets_per_dpu", errors
    )
    native_execution = _strict_bool(payload, "native_execution", errors)
    validation_performed = _strict_bool(payload, "validation_performed", errors)
    exact_validation = _strict_bool(payload, "exact_validation", errors)
    release_status = _strict_string(payload, "release_status", errors)
    fallback = _strict_bool(payload, "fallback", errors)
    simulator = _strict_bool(payload, "simulator_kernel_executed", errors)
    backend_profile = _strict_string(payload, "backend_profile", errors)
    _strict_list(payload, "device_evidence", errors)
    staged_patch = _strict_mapping(payload, "staged_patch", errors)
    payload_aligned = _strict_bool(payload, "payload_sizes_8_byte_aligned", errors)
    physical_bytes_available = _strict_bool(
        payload, "physical_transfer_bytes_available", errors
    )
    if "physical_transfer_bytes" not in payload:
        errors.append("physical_transfer_bytes is required")
    elif payload["physical_transfer_bytes"] is not None:
        errors.append("physical_transfer_bytes must be null")
    failure_stage = payload.get("failure_stage")
    if failure_stage is not None and type(failure_stage) is not str:
        errors.append("failure_stage must be null or a string")
    _strict_nullable_string(payload, "reason", errors)

    if (
        provider.requested_dpus is not None
        and requested_dpus != provider.requested_dpus
    ):
        errors.append(
            f"requested_dpu_count must equal catalog request {provider.requested_dpus}"
        )
    if provider.requested_dpus is not None and observed_dpus != provider.requested_dpus:
        errors.append(
            f"observed_dpu_count must equal requested count {provider.requested_dpus}"
        )
    if (
        provider.requested_tasklets is not None
        and configured_tasklets != provider.requested_tasklets
    ):
        errors.append(
            "configured_tasklets_per_dpu must equal requested count "
            f"{provider.requested_tasklets}"
        )
    required_truths = (
        (runner_status == "passed", "runner status must be 'passed'"),
        (hardware_preflight is True, "hardware_preflight_verified must be true"),
        (target == PHYSICAL_TARGET, f"target must be {PHYSICAL_TARGET!r}"),
        (target_observed == target, "target_observed must equal target"),
        (native_execution is True, "native_execution must be true"),
        (validation_performed is True, "validation_performed must be true"),
        (exact_validation is True, "exact_validation must be true"),
        (release_status == "released", "release_status must be 'released'"),
        (fallback is False, "fallback must be false"),
        (simulator is False, "simulator_kernel_executed must be false"),
        (backend_profile == "backend=hw", "backend_profile must be 'backend=hw'"),
        (
            isinstance(staged_patch, Mapping) and staged_patch.get("applied") is True,
            "staged_patch.applied must be true",
        ),
        (
            observed_tasklets is None,
            "observed_tasklets_per_dpu must be null",
        ),
        (payload_aligned is True, "payload_sizes_8_byte_aligned must be true"),
        (
            physical_bytes_available is False,
            "physical_transfer_bytes_available must be false",
        ),
        (failure_stage is None, "failure_stage must be null for a passed result"),
    )
    errors.extend(message for passed, message in required_truths if not passed)
    status = "qualified" if not errors else "failed"
    raw_reason = payload.get("reason")
    reason = (
        "strict physical SimplePIM qualification contract passed"
        if status == "qualified"
        else (
            f"runner reported {runner_status}: {raw_reason}"
            if type(raw_reason) is str and runner_status != "passed"
            else errors[0]
        )
    )
    return RunnerQualification(
        status=status,
        reason=reason,
        contract_errors=tuple(errors),
        runner_status=runner_status,
        hardware_preflight_verified=hardware_preflight,
        target=target,
        observed_dpus=observed_dpus,
        configured_tasklets=configured_tasklets,
        observed_tasklets=observed_tasklets,
        native_execution=native_execution,
        validation_performed=validation_performed,
        exact_validation=exact_validation,
        release_status=release_status,
        fallback_used=fallback,
        simulator_kernel_executed=simulator,
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_provider_spec(provider: ProviderSpec) -> None:
    if provider.status == "executable" and not provider.executable:
        raise ValueError(f"provider {provider.provider_id} status/executable disagree")
    if provider.status == "blocked" and provider.executable:
        raise ValueError(
            f"blocked provider {provider.provider_id} cannot be executable"
        )
    if not provider.executable:
        return
    required = {
        "runner": provider.runner,
        "source.path": provider.source_path,
        "source.pinned_commit": provider.pinned_commit,
        "hardware.requested_dpus": provider.requested_dpus,
        "hardware.requested_tasklets": provider.requested_tasklets,
        "runner_contract.schema_version": provider.runner_schema_version,
        "runner_contract.probe_id": provider.probe_id,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise ValueError(
            f"executable provider {provider.provider_id} is missing: {', '.join(missing)}"
        )
    if provider.provider_id != SIMPLEPIM_PROVIDER_ID:
        raise ValueError("the M1 harness executes only the SimplePIM provider")
    if provider.runner_schema_version != SIMPLEPIM_RUNNER_SCHEMA_VERSION:
        raise ValueError("SimplePIM runner schema does not match the M1 contract")
    if provider.probe_id != SIMPLEPIM_PROBE_ID:
        raise ValueError("SimplePIM probe id does not match the M1 contract")


def _git_output(
    path: Path,
    args: tuple[str, ...],
    *,
    allow_empty: bool = False,
) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(path), *args],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    value = completed.stdout.strip()
    return value if value or allow_empty else None


def _git_bytes(path: Path, args: tuple[str, ...]) -> bytes | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(path), *args],
            capture_output=True,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout if completed.returncode == 0 else None


def _tool_version(path: Path) -> str | None:
    try:
        completed = subprocess.run(
            [str(path), "--version"],
            capture_output=True,
            text=True,
            check=False,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    lines = (completed.stdout or completed.stderr).splitlines()
    return lines[0].strip() if lines else None


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _relative_path(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def _required_string(data: Mapping[str, Any], key: str) -> str:
    value = data.get(key)
    if type(value) is not str or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value.strip()


def _required_bool(data: Mapping[str, Any], key: str) -> bool:
    value = data.get(key)
    if type(value) is not bool:
        raise ValueError(f"{key} must be a boolean")
    return value


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    if type(value) is not str or not value.strip():
        raise ValueError("optional string values must be non-empty strings")
    return value.strip()


def _optional_positive_int(value: Any) -> int | None:
    if value is None:
        return None
    if type(value) is not int or value <= 0:
        raise ValueError("counts must be positive integers")
    return value


def _expect_exact_string(
    payload: Mapping[str, Any],
    key: str,
    expected: str,
    errors: list[str],
) -> None:
    value = payload.get(key)
    if type(value) is not str:
        errors.append(f"{key} must be a string")
    elif value != expected:
        errors.append(f"{key} must be {expected!r}, got {value!r}")


def _strict_string(
    payload: Mapping[str, Any], key: str, errors: list[str]
) -> str | None:
    value = payload.get(key)
    if type(value) is not str:
        errors.append(f"{key} must be a string")
        return None
    return value


def _strict_bool(
    payload: Mapping[str, Any], key: str, errors: list[str]
) -> bool | None:
    value = payload.get(key)
    if type(value) is not bool:
        errors.append(f"{key} must be a boolean")
        return None
    return value


def _strict_int(payload: Mapping[str, Any], key: str, errors: list[str]) -> int | None:
    value = payload.get(key)
    if type(value) is not int or value <= 0:
        errors.append(f"{key} must be a positive integer")
        return None
    return value


def _strict_nullable_int(
    payload: Mapping[str, Any], key: str, errors: list[str]
) -> int | None:
    value = payload.get(key)
    if value is None:
        return None
    if type(value) is not int or value <= 0:
        errors.append(f"{key} must be null or a positive integer")
        return None
    return value


def _strict_mapping(
    payload: Mapping[str, Any], key: str, errors: list[str]
) -> Mapping[str, Any] | None:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        errors.append(f"{key} must be an object")
        return None
    return value


def _strict_list(
    payload: Mapping[str, Any], key: str, errors: list[str]
) -> list[Any] | None:
    value = payload.get(key)
    if type(value) is not list:
        errors.append(f"{key} must be an array")
        return None
    return value


def _strict_nullable_string(
    payload: Mapping[str, Any], key: str, errors: list[str]
) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if type(value) is not str:
        errors.append(f"{key} must be null or a string")
        return None
    return value


def _check_exact_keys(
    value: Mapping[str, Any],
    required: frozenset[str],
    optional: frozenset[str],
    label: str,
    errors: list[str],
) -> None:
    keys = set(value)
    invalid_keys = [key for key in keys if type(key) is not str]
    if invalid_keys:
        errors.append(f"{label} keys must be strings")
        return
    missing = sorted(required.difference(keys))
    unknown = sorted(keys.difference(required | optional))
    if missing:
        errors.append(f"{label} is missing fields: {missing}")
    if unknown:
        errors.append(f"{label} has unknown fields: {unknown}")


def _validate_nullable_sha256(
    payload: Mapping[str, Any],
    key: str,
    errors: list[str],
) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if not _is_sha256(value):
        errors.append(f"{key} must be null or a lowercase SHA-256 digest")
        return None
    return value


def _validate_source_hashes(
    payload: Mapping[str, Any],
    status: str | None,
    errors: list[str],
) -> Mapping[str, Any] | None:
    value = _strict_mapping(payload, "source_hashes", errors)
    if value is None:
        return None
    keys = set(value)
    minimal_keys = {
        "combined_sha256",
        "staged_source_before_patch_sha256",
        "staged_source_after_patch_sha256",
        "staged_patch_sha256",
    }
    if status == "failed" and keys == minimal_keys:
        for key in minimal_keys:
            digest = value.get(key)
            if digest is not None and not _is_sha256(digest):
                errors.append(f"source_hashes.{key} must be null or SHA-256")
        return value
    _check_exact_keys(value, _SOURCE_HASH_KEYS, frozenset(), "source_hashes", errors)
    always_hashed = _SOURCE_HASH_KEYS - {
        "upstream_submodule",
        "staged_source_before_patch_sha256",
        "staged_source_after_patch_sha256",
        "staged_patch_sha256",
    }
    for key in always_hashed:
        if not _is_sha256(value.get(key)):
            errors.append(f"source_hashes.{key} must be a lowercase SHA-256 digest")
    for key in (
        "staged_source_before_patch_sha256",
        "staged_source_after_patch_sha256",
        "staged_patch_sha256",
    ):
        digest = value.get(key)
        if digest is not None and not _is_sha256(digest):
            errors.append(f"source_hashes.{key} must be null or SHA-256")
    upstream = value.get("upstream_submodule")
    if type(upstream) is not str or not upstream:
        errors.append("source_hashes.upstream_submodule must be a non-empty string")
    return value


def _validate_staged_patch(
    payload: Mapping[str, Any],
    errors: list[str],
) -> Mapping[str, Any] | None:
    value = _strict_mapping(payload, "staged_patch", errors)
    if value is None:
        return None
    _check_exact_keys(value, _PATCH_KEYS, frozenset(), "staged_patch", errors)
    if value.get("path") != SIMPLEPIM_PATCH_PATH:
        errors.append(f"staged_patch.path must be {SIMPLEPIM_PATCH_PATH!r}")
    digest = value.get("sha256")
    if digest is not None and not _is_sha256(digest):
        errors.append("staged_patch.sha256 must be null or SHA-256")
    staged_digest = value.get("staged_sha256")
    if staged_digest is not None and not _is_sha256(staged_digest):
        errors.append("staged_patch.staged_sha256 must be null or SHA-256")
    if type(value.get("applied")) is not bool:
        errors.append("staged_patch.applied must be a boolean")
    replacement_count = value.get("replacement_count")
    if type(replacement_count) is not int or replacement_count < 0:
        errors.append("staged_patch.replacement_count must be a non-negative integer")
    target_digest = value.get("staged_target_sha256")
    if target_digest is not None and not _is_sha256(target_digest):
        errors.append("staged_patch.staged_target_sha256 must be null or SHA-256")
    command_digest = value.get("command_fingerprint")
    if command_digest is not None and not _is_sha256(command_digest):
        errors.append("staged_patch.command_fingerprint must be null or SHA-256")
    for key in (
        "staged_source_before_sha256",
        "staged_source_after_sha256",
    ):
        source_digest = value.get(key)
        if source_digest is not None and not _is_sha256(source_digest):
            errors.append(f"staged_patch.{key} must be null or SHA-256")
    if value.get("applied") is True:
        before = value.get("staged_source_before_sha256")
        after = value.get("staged_source_after_sha256")
        applied_invariants = (
            replacement_count == 2,
            staged_digest == digest,
            _is_sha256(before),
            _is_sha256(after),
            before != after,
            _is_sha256(target_digest),
        )
        if not all(applied_invariants):
            errors.append("applied staged_patch evidence is incomplete")
    elif replacement_count != 0:
        errors.append("unapplied staged_patch replacement_count must be zero")
    return value


def _validate_effective_compilers(
    payload: Mapping[str, Any],
    errors: list[str],
) -> Mapping[str, Any] | None:
    value = _strict_mapping(payload, "effective_compilers", errors)
    if value is None:
        return None
    roles = frozenset(_COMPILER_ROLES)
    _check_exact_keys(value, roles, frozenset(), "effective_compilers", errors)
    for role, expected_command in _COMPILER_ROLES.items():
        identity = value.get(role)
        label = f"effective_compilers.{role}"
        if not isinstance(identity, Mapping):
            errors.append(f"{label} must be an object")
            continue
        _check_exact_keys(
            identity,
            _COMPILER_IDENTITY_KEYS,
            frozenset(),
            label,
            errors,
        )
        if identity.get("command") != expected_command:
            errors.append(f"{label}.command must be {expected_command!r}")
        available = identity.get("available")
        if type(available) is not bool:
            errors.append(f"{label}.available must be a boolean")
        path = identity.get("path")
        if path is not None and (type(path) is not str or not path):
            errors.append(f"{label}.path must be null or a non-empty string")
        digest = identity.get("sha256")
        if digest is not None and not _is_sha256(digest):
            errors.append(f"{label}.sha256 must be null or SHA-256")
        if type(available) is bool and available != (
            type(path) is str and bool(path) and _is_sha256(digest)
        ):
            errors.append(f"{label} availability evidence is inconsistent")
        if (
            available is True
            and type(path) is str
            and _is_sha256(digest)
            and (not Path(path).is_file() or sha256_file(Path(path)) != digest)
        ):
            errors.append(f"{label} compiler file evidence is invalid")
    return value


def _validate_expected_compiler(
    compilers: Mapping[str, Any] | None,
    role: str,
    expected: Mapping[str, Any],
    errors: list[str],
) -> None:
    identity = compilers.get(role) if compilers is not None else None
    if not isinstance(identity, Mapping):
        return
    comparisons = {
        "available": expected.get("available"),
        "path": expected.get("path"),
        "sha256": expected.get("sha256"),
    }
    for key, expected_value in comparisons.items():
        if identity.get(key) != expected_value:
            errors.append(
                f"effective_compilers.{role}.{key} does not match "
                "outer runner provenance"
            )


def _validate_preflight_binding(
    payload: Mapping[str, Any],
    preflight: Mapping[str, Any],
    errors: list[str],
) -> None:
    direct_keys = ("command_fingerprint", "effective_compilers", "commands")
    for key in direct_keys:
        if payload.get(key) != preflight.get(key):
            errors.append(f"execution {key} does not match canonical preflight")
    execution_source = payload.get("source_hashes")
    preflight_source = preflight.get("source_hashes")
    if isinstance(execution_source, Mapping) and isinstance(preflight_source, Mapping):
        for key in (
            "owned_qualification_sha256",
            "patch_sha256",
            "upstream_submodule",
        ):
            if execution_source.get(key) != preflight_source.get(key):
                errors.append(
                    f"execution source_hashes.{key} does not match canonical preflight"
                )
    execution_patch = payload.get("staged_patch")
    preflight_patch = preflight.get("staged_patch")
    if isinstance(execution_patch, Mapping) and isinstance(preflight_patch, Mapping):
        for key in ("path", "sha256", "command_fingerprint"):
            if execution_patch.get(key) != preflight_patch.get(key):
                errors.append(
                    f"execution staged_patch.{key} does not match canonical preflight"
                )


def _validate_artifact_hashes(
    payload: Mapping[str, Any],
    key: str,
    allowed_names: frozenset[str],
    errors: list[str],
    *,
    allow_null: bool = False,
) -> Mapping[str, Any] | None:
    value = _strict_mapping(payload, key, errors)
    if value is None:
        return None
    unknown = sorted(set(value).difference(allowed_names))
    if unknown:
        errors.append(f"{key} has unknown artifacts: {unknown}")
    for name, digest in value.items():
        if type(name) is not str or not name:
            errors.append(f"{key} artifact names must be non-empty strings")
        if digest is None and allow_null:
            continue
        if not _is_sha256(digest):
            errors.append(f"{key}.{name} must be a lowercase SHA-256 digest")
    return value


def _validate_device_evidence(
    payload: Mapping[str, Any],
    errors: list[str],
) -> list[Any] | None:
    value = _strict_list(payload, "device_evidence", errors)
    if value is None:
        return None
    for index, item in enumerate(value):
        label = f"device_evidence[{index}]"
        if not isinstance(item, Mapping):
            errors.append(f"{label} must be an object")
            continue
        _check_exact_keys(
            item,
            _DEVICE_REQUIRED_KEYS,
            _DEVICE_OPTIONAL_KEYS,
            label,
            errors,
        )
        for key in ("path", "sysfs_path"):
            if type(item.get(key)) is not str or not item.get(key):
                errors.append(f"{label}.{key} must be a non-empty string")
        for key in (
            "exists",
            "character_device",
            "readable",
            "writable",
            "sysfs_exists",
        ):
            if type(item.get(key)) is not bool:
                errors.append(f"{label}.{key} must be a boolean")
        if "error" in item and (type(item["error"]) is not str or not item["error"]):
            errors.append(f"{label}.error must be a non-empty string")
    return value


def _device_evidence_proves_access(value: list[Any] | None) -> bool:
    return bool(
        value
        and any(
            isinstance(item, Mapping)
            and item.get("exists") is True
            and item.get("character_device") is True
            and item.get("readable") is True
            and item.get("writable") is True
            for item in value
        )
    )


def _validate_logical_bytes(
    payload: Mapping[str, Any],
    errors: list[str],
) -> None:
    value = _strict_mapping(payload, "logical_transfer_bytes", errors)
    if value is None:
        return
    required = frozenset({"h2d", "d2h", "total", "scope"})
    _check_exact_keys(value, required, frozenset(), "logical_transfer_bytes", errors)
    expected = {
        "h2d": SIMPLEPIM_LOGICAL_INPUT_BYTES,
        "d2h": SIMPLEPIM_LOGICAL_OUTPUT_BYTES,
        "total": SIMPLEPIM_LOGICAL_TOTAL_BYTES,
    }
    for key, expected_value in expected.items():
        actual = value.get(key)
        if type(actual) is not int or actual != expected_value:
            errors.append(f"logical_transfer_bytes.{key} must equal {expected_value}")
    if value.get("scope") != "logical_application_payload_only":
        errors.append(
            "logical_transfer_bytes.scope must be 'logical_application_payload_only'"
        )
    h2d = value.get("h2d")
    d2h = value.get("d2h")
    total = value.get("total")
    if (
        type(h2d) is int
        and type(d2h) is int
        and type(total) is int
        and h2d + d2h != total
    ):
        errors.append("logical transfer total must equal h2d plus d2h")
    if type(h2d) is int and type(d2h) is int and (h2d % 8 != 0 or d2h % 8 != 0):
        errors.append("logical transfer byte counts must be 8-byte aligned")


def _validate_timing(value: Any, label: str, errors: list[str]) -> None:
    if not isinstance(value, Mapping):
        errors.append(f"{label} must be an object")
        return
    for key, duration in value.items():
        if type(key) is not str or not key:
            errors.append(f"{label} keys must be non-empty strings")
        if (
            type(duration) not in {int, float}
            or not math.isfinite(duration)
            or duration < 0
        ):
            errors.append(f"{label}.{key} must be a finite non-negative number")


def _validate_commands(value: Any, errors: list[str]) -> None:
    if not isinstance(value, Mapping):
        errors.append("commands must be an object")
        return
    required = frozenset({"patch", "build", "run"})
    _check_exact_keys(value, required, frozenset(), "commands", errors)
    for key in required:
        command = value.get(key)
        if (
            type(command) is not list
            or not command
            or any(type(argument) is not str or not argument for argument in command)
        ):
            errors.append(f"commands.{key} must be a non-empty string array")


def _compiler_available(
    compilers: Mapping[str, Any] | None,
    role: str,
) -> bool:
    if compilers is None:
        return False
    identity = compilers.get(role)
    return isinstance(identity, Mapping) and identity.get("available") is True


def _validate_command_compilers(
    commands: Any,
    compilers: Mapping[str, Any] | None,
    errors: list[str],
) -> None:
    if not isinstance(commands, Mapping) or compilers is None:
        return
    build = commands.get("build")
    host_identity = compilers.get("host_cc")
    dpu_identity = compilers.get("dpu_cc")
    if type(build) is not list:
        return
    if isinstance(host_identity, Mapping):
        host_path = host_identity.get("path")
        if type(host_path) is str and f"HOST_CC={host_path}" not in build:
            errors.append(
                "commands.build HOST_CC must match effective_compilers.host_cc.path"
            )
    if isinstance(dpu_identity, Mapping):
        dpu_command = dpu_identity.get("command")
        if type(dpu_command) is str and f"DPU_CC={dpu_command}" not in build:
            errors.append(
                "commands.build DPU_CC must match effective_compilers.dpu_cc.command"
            )


def _validate_command_evidence(
    value: Any,
    expected_command: Any,
    label: str,
    errors: list[str],
) -> None:
    if not isinstance(value, Mapping):
        errors.append(f"{label} must be an object")
        return
    required = frozenset(
        {
            "command",
            "command_fingerprint",
            "status",
            "returncode",
            "wall_s",
            "stdout_tail",
            "stderr_tail",
        }
    )
    optional = frozenset({"timeout_cleanup"})
    _check_exact_keys(value, required, optional, label, errors)
    command = value.get("command")
    if (
        type(command) is not list
        or not command
        or any(type(argument) is not str or not argument for argument in command)
    ):
        errors.append(f"{label}.command must be a non-empty string array")
    elif expected_command is not None and command != list(expected_command):
        errors.append(f"{label}.command must equal the planned command")
    if value.get("command_fingerprint") != _hash_json(command):
        errors.append(f"{label}.command_fingerprint does not match command")
    if value.get("status") not in {"passed", "failed", "timeout"}:
        errors.append(f"{label}.status is invalid")
    returncode = value.get("returncode")
    if returncode is not None and type(returncode) is not int:
        errors.append(f"{label}.returncode must be null or an integer")
    if value.get("status") == "passed" and returncode != 0:
        errors.append(f"{label} passed status requires returncode zero")
    if value.get("status") == "failed" and returncode == 0:
        errors.append(f"{label} failed status cannot have returncode zero")
    for key in ("stdout_tail", "stderr_tail"):
        if type(value.get(key)) is not str:
            errors.append(f"{label}.{key} must be a string")
    _validate_non_negative_number(value.get("wall_s"), f"{label}.wall_s", errors)
    if "timeout_cleanup" in value:
        _validate_timeout_cleanup(
            value["timeout_cleanup"], f"{label}.timeout_cleanup", errors
        )


def _command_evidence_passed(value: Any) -> bool:
    return (
        isinstance(value, Mapping)
        and value.get("status") == "passed"
        and value.get("returncode") == 0
    )


def _validate_host_result(
    value: Any,
    provider: ProviderSpec,
    errors: list[str],
) -> None:
    if not isinstance(value, Mapping):
        errors.append("host_result must be an object")
        return
    _check_exact_keys(
        value,
        _HOST_RESULT_KEYS,
        frozenset(),
        "host_result",
        errors,
    )
    expected = {
        "schema_version": SIMPLEPIM_HOST_SCHEMA_VERSION,
        "provider_id": provider.provider_id,
        "probe_id": provider.probe_id or SIMPLEPIM_PROBE_ID,
        "backend_profile": "backend=hw",
        "requested_dpu_count": provider.requested_dpus,
        "configured_tasklets_per_dpu": provider.requested_tasklets,
        "observed_tasklets_per_dpu": None,
        "fallback": False,
        "logical_input_bytes": SIMPLEPIM_LOGICAL_INPUT_BYTES,
        "logical_output_bytes": SIMPLEPIM_LOGICAL_OUTPUT_BYTES,
        "physical_transfer_bytes_available": False,
        "physical_transfer_bytes": None,
    }
    for key, expected_value in expected.items():
        if value.get(key) != expected_value or (
            type(expected_value) is bool and type(value.get(key)) is not bool
        ):
            errors.append(f"host_result.{key} is invalid")
    if value.get("status") not in {"passed", "failed"}:
        errors.append("host_result.status is invalid")
    observed = value.get("observed_dpu_count")
    if observed is not None and (type(observed) is not int or observed <= 0):
        errors.append("host_result.observed_dpu_count must be null or positive")
    for key in (
        "native_run_completed",
        "validation_performed",
        "host_exact_validation",
    ):
        if type(value.get(key)) is not bool:
            errors.append(f"host_result.{key} must be a boolean")
    if value.get("release_status") not in {
        "not_attempted",
        "released",
        "failed",
        "unknown",
    }:
        errors.append("host_result.release_status is invalid")
    _validate_timing(value.get("timing"), "host_result.timing", errors)
    stage = value.get("failure_stage")
    if stage is not None and (type(stage) is not str or not stage):
        errors.append("host_result.failure_stage must be null or non-empty")
    reason = value.get("reason")
    if reason is not None and (type(reason) is not str or not reason):
        errors.append("host_result.reason must be null or a non-empty string")
    if value.get("status") == "passed":
        required_truths = (
            observed == provider.requested_dpus,
            value.get("native_run_completed") is True,
            value.get("validation_performed") is True,
            value.get("host_exact_validation") is True,
            value.get("release_status") == "released",
            stage is None,
            reason is None,
        )
        if not all(required_truths):
            errors.append("passed host_result does not satisfy its strict contract")


def _host_result_matches_pass(
    value: Any,
    observed_dpus: int | None,
    configured_tasklets: int | None,
    release_status: str | None,
) -> bool:
    return bool(
        isinstance(value, Mapping)
        and value.get("status") == "passed"
        and value.get("observed_dpu_count") == observed_dpus
        and value.get("configured_tasklets_per_dpu") == configured_tasklets
        and value.get("native_run_completed") is True
        and value.get("validation_performed") is True
        and value.get("host_exact_validation") is True
        and value.get("release_status") == release_status == "released"
        and value.get("failure_stage") is None
        and value.get("reason") is None
    )


def _validate_timeout_cleanup(
    value: Any,
    label: str,
    errors: list[str],
) -> None:
    if not isinstance(value, Mapping):
        errors.append(f"{label} must be an object")
        return
    required = frozenset(
        {
            "process_group",
            "sigterm_sent",
            "sigkill_sent",
            "process_exited",
            "group_probe_performed",
            "leader_exited_before_group_probe",
            "live_members_after_sigterm",
            "live_members_after_cleanup",
            "process_group_terminated",
            "signal_errors",
        }
    )
    _check_exact_keys(value, required, frozenset(), label, errors)
    process_group = value.get("process_group")
    if type(process_group) is not int or process_group <= 0:
        errors.append(f"{label}.process_group must be a positive integer")
    for key in (
        "sigterm_sent",
        "sigkill_sent",
        "process_exited",
        "group_probe_performed",
        "leader_exited_before_group_probe",
        "process_group_terminated",
    ):
        if type(value.get(key)) is not bool:
            errors.append(f"{label}.{key} must be a boolean")
    for key in ("live_members_after_sigterm", "live_members_after_cleanup"):
        members = value.get(key)
        if type(members) is not list or any(
            type(member) is not int or member <= 0 for member in members
        ):
            errors.append(f"{label}.{key} must be an array of positive PIDs")
    signal_errors = value.get("signal_errors")
    if type(signal_errors) is not list or any(
        type(message) is not str for message in signal_errors
    ):
        errors.append(f"{label}.signal_errors must be a string array")


def _validate_non_negative_number(
    value: Any,
    label: str,
    errors: list[str],
) -> None:
    if type(value) not in {int, float} or not math.isfinite(value) or value < 0:
        errors.append(f"{label} must be a finite non-negative number")


def _is_sha256(value: Any) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _hash_json(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
