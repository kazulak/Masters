"""Small shared contract for the physical SimplePIM qualification lane."""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
from typing import Any, Mapping, NamedTuple, Sequence
import yaml
CATALOG_SCHEMA_VERSION = "provider_qualification_catalog_v1"
SIMPLEPIM_PROVIDER_ID = "simplepim"
SIMPLEPIM_RUNNER_SCHEMA_VERSION = "simplepim_provider_qualification_v1"
SIMPLEPIM_PROBE_ID = "simplepim_va_map_zip_v1"
SIMPLEPIM_HOST_SCHEMA_VERSION = "simplepim_qualification_host_v2"
PHYSICAL_TARGET = "physical_hardware"
SIMPLEPIM_PATCH_PATH = "patches/simplepim-map-unroll-rest.patch"
SIMPLEPIM_LOGICAL_INPUT_BYTES = 2048
SIMPLEPIM_LOGICAL_OUTPUT_BYTES = 1024
SIMPLEPIM_LOGICAL_TOTAL_BYTES = 3072
Json = dict[str, Any]
REQUIRED_RESULT_FIELDS = frozenset({"schema_version", "provider_id", "probe_id", "status", "target", "target_observed", "requested_dpu_count", "observed_dpu_count", "configured_tasklets_per_dpu", "observed_tasklets_per_dpu", "hardware_preflight_verified", "device_evidence", "native_execution", "validation_performed", "exact_validation", "fallback", "simulator_kernel_executed", "release_status", "backend_profile", "source_hash", "source_hashes", "command_fingerprint", "effective_compilers", "staged_patch", "binary_hashes", "input_hashes", "output_hash", "logical_transfer_bytes", "payload_sizes_8_byte_aligned", "physical_transfer_bytes_available", "physical_transfer_bytes", "timing", "failure_stage", "reason"})
BINARY_NAMES = ("host", "dpu_init_binary", "dpu_zip", "dpu_map_va_funcs")
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
        raise ValueError(f"unknown provider {provider_id!r}")
RunnerQualification = NamedTuple("RunnerQualification", [("status", str), ("reason", str), ("contract_errors", tuple[str, ...]), ("runner_status", str | None), ("hardware_preflight_verified", bool | None), ("target", str | None), ("observed_dpus", int | None), ("configured_tasklets", int | None), ("observed_tasklets", int | None), ("native_execution", bool | None), ("validation_performed", bool | None), ("exact_validation", bool | None), ("release_status", str | None), ("fallback_used", bool | None), ("simulator_kernel_executed", bool | None)])
def _string(value: Any, label: str, *, optional: bool = False) -> str | None:
    if value is None and optional:
        return None
    if type(value) is not str or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value.strip()
def _positive(value: Any, label: str) -> int | None:
    if value is None:
        return None
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value
def resolve_root(root_dir: Path) -> Path:
    root = root_dir.expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"qualification root is not a directory: {root}")
    return root
def resolve_catalog_path(root_dir: Path, catalog_path: Path) -> Path:
    root = resolve_root(root_dir)
    path = catalog_path.expanduser()
    path = path if path.is_absolute() else root / path
    path = path.resolve()
    if not path.is_file():
        raise ValueError(f"provider catalog is not a file: {path}")
    return path
def resolve_provider_path(root_dir: Path, configured_path: str, *, label: str) -> Path:
    root = resolve_root(root_dir)
    path = Path(configured_path).expanduser()
    path = (path if path.is_absolute() else root / path).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} must remain under qualification root: {path}") from exc
    return path
def load_provider_catalog(path: Path) -> ProviderCatalog:
    resolved = path.expanduser().resolve()
    try:
        data = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"cannot read provider catalog {resolved}: {exc}") from exc
    if not isinstance(data, Mapping) or data.get("schema_version") != CATALOG_SCHEMA_VERSION:
        raise ValueError("unsupported provider catalog")
    raw_providers = data.get("providers")
    if not isinstance(raw_providers, list) or not raw_providers:
        raise ValueError("provider catalog must contain providers")
    providers: list[ProviderSpec] = []
    seen: set[str] = set()
    for raw in raw_providers:
        if not isinstance(raw, Mapping):
            raise ValueError("provider entries must be mappings")
        provider_id = _string(raw.get("id"), "provider id")
        assert provider_id is not None
        if provider_id in seen:
            raise ValueError(f"duplicate provider id: {provider_id}")
        seen.add(provider_id)
        status = _string(raw.get("status"), f"provider {provider_id} status")
        assert status is not None
        executable = raw.get("executable")
        if type(executable) is not bool or status not in {"executable", "blocked"}:
            raise ValueError(f"invalid status/executable for provider {provider_id}")
        source = raw.get("source") if isinstance(raw.get("source"), Mapping) else {}
        hardware = raw.get("hardware") if isinstance(raw.get("hardware"), Mapping) else {}
        contract = raw.get("runner_contract") if isinstance(raw.get("runner_contract"), Mapping) else {}
        provider = ProviderSpec(
            provider_id,
            _string(raw.get("name"), "provider name", optional=True) or provider_id,
            status,
            _string(raw.get("reason"), "provider reason", optional=True),
            executable,
            _string(raw.get("runner"), "runner", optional=True),
            _string(source.get("path"), "source.path", optional=True),
            _string(source.get("pinned_commit"), "source.pinned_commit", optional=True),
            _positive(hardware.get("requested_dpus"), "hardware.requested_dpus"),
            _positive(hardware.get("requested_tasklets"), "hardware.requested_tasklets"),
            _string(contract.get("schema_version"), "runner schema", optional=True),
            _string(contract.get("probe_id"), "probe id", optional=True),
            dict(raw.get("lane")) if isinstance(raw.get("lane"), Mapping) else {},
        )
        if provider.status == "executable" and not provider.executable:
            raise ValueError(f"executable provider {provider_id} is disabled")
        if provider.status == "blocked" and provider.executable:
            raise ValueError(f"blocked provider {provider_id} is executable")
        if provider.executable:
            required = (
                provider.runner,
                provider.source_path,
                provider.pinned_commit,
                provider.requested_dpus,
                provider.requested_tasklets,
                provider.runner_schema_version,
                provider.probe_id,
            )
            if any(item is None for item in required):
                raise ValueError(f"executable provider {provider_id} has missing fields")
        providers.append(provider)
    return ProviderCatalog(
        _string(data.get("catalog_id"), "catalog_id") or "",
        CATALOG_SCHEMA_VERSION,
        tuple(sorted(providers, key=lambda p: p.provider_id)),
        resolved,
    )
def _git(path: Path, *args: str, allow_empty: bool = False) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), *args],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode:
        return None
    value = result.stdout.strip()
    return value if value or allow_empty else None
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()
def _relative(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)
def provider_source_fingerprint(provider: ProviderSpec, root_dir: Path) -> Json:
    if not provider.source_path:
        return {
            "path": None,
            "pinned_commit": provider.pinned_commit,
            "head_commit": None,
            "commit_matches_pin": False,
            "clean": False,
            "git_root_matches_source": False,
        }
    root = resolve_root(root_dir)
    source = resolve_provider_path(root, provider.source_path, label="provider source")
    head = _git(source, "rev-parse", "HEAD")
    git_root = _git(source, "rev-parse", "--show-toplevel")
    status = _git(source, "status", "--porcelain=v1", "--untracked-files=all", allow_empty=True)
    return {
        "path": _relative(source, root),
        "pinned_commit": provider.pinned_commit,
        "head_commit": head,
        "commit_matches_pin": head == provider.pinned_commit,
        "clean": status == "",
        "git_root_matches_source": bool(git_root and Path(git_root).resolve() == source),
        "status_sha256": _hash_text(status) if status is not None else None,
    }
def source_gate_failure(provider: ProviderSpec, fingerprint: Mapping[str, Any]) -> str | None:
    if not provider.executable:
        return None
    if not fingerprint.get("head_commit"):
        return f"provider {provider.provider_id} source is not a Git checkout"
    if not fingerprint.get("git_root_matches_source"):
        return f"provider {provider.provider_id} source path is not the checkout root"
    if not fingerprint.get("commit_matches_pin"):
        return f"provider {provider.provider_id} source is not pinned to {provider.pinned_commit}"
    if not fingerprint.get("clean"):
        return f"provider {provider.provider_id} source worktree is not clean"
    return None
def require_provider_source(provider: ProviderSpec, root_dir: Path) -> Json:
    fingerprint = provider_source_fingerprint(provider, root_dir)
    if failure := source_gate_failure(provider, fingerprint):
        raise ValueError(failure)
    return fingerprint
def qualification_repository_fingerprint(root_dir: Path, paths: Sequence[str]) -> Json:
    root = resolve_root(root_dir)
    git_root = _git(root, "rev-parse", "--show-toplevel")
    if not git_root:
        return {
            "head_commit": None,
            "paths": list(paths),
            "clean": False,
            "all_files_tracked": False,
        }
    repo = Path(git_root).resolve()
    resolved = [resolve_provider_path(root, path, label="qualification source") for path in paths]
    missing = [str(path) for path in resolved if not path.exists()]
    if missing:
        raise ValueError(f"qualification source is missing: {missing[0]}")
    specs = [str(path.relative_to(repo)) for path in resolved]
    status = _git(
        repo,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "--",
        *specs,
        allow_empty=True,
    )
    untracked = _git(
        repo,
        "ls-files",
        "--others",
        "--exclude-standard",
        "--",
        *specs,
        allow_empty=True,
    )
    return {
        "git_root": str(repo),
        "head_commit": _git(repo, "rev-parse", "HEAD"),
        "paths": list(paths),
        "clean": status == "",
        "all_files_tracked": untracked == "",
        "status_sha256": _hash_text(status or ""),
    }
def require_qualification_repository_source(root_dir: Path, paths: Sequence[str]) -> Json:
    fingerprint = qualification_repository_fingerprint(root_dir, paths)
    if fingerprint.get("head_commit") is None:
        raise ValueError("qualification repository is not a Git checkout")
    if not fingerprint.get("all_files_tracked") or not fingerprint.get("clean"):
        raise ValueError("qualification repository source must be tracked and clean before --execute")
    return fingerprint
def _tool_version(path: Path) -> str | None:
    try:
        result = subprocess.run(
            [str(path), "--version"],
            capture_output=True,
            text=True,
            check=False,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return (result.stdout or result.stderr).splitlines()[0].strip() if (result.stdout or result.stderr) else None
def resolve_host_cc(environment: Mapping[str, str] | None = None) -> Json:
    env = environment or {}
    configured = env.get("HOST_CC", "gcc").strip()
    try:
        command = shlex.split(configured)
    except ValueError as exc:
        raise ValueError(f"HOST_CC is invalid: {exc}") from exc
    if len(command) != 1:
        raise ValueError("HOST_CC must name one compiler without arguments")
    raw = shutil.which(command[0], path=env.get("PATH"))
    if not raw or not Path(raw).is_file():
        raise ValueError(f"HOST_CC compiler is unavailable: {command[0]}")
    path = Path(raw).resolve()
    return {
        "command": command[0],
        "configured": configured,
        "runner_value": "gcc" if "HOST_CC" not in env else str(path),
        "available": True,
        "path": str(path),
        "sha256": sha256_file(path),
        "version": _tool_version(path),
    }
def fingerprint_tools(environment: Mapping[str, str] | None = None) -> Json:
    env = environment or {}
    values: Json = {}
    for name in ("python", "make", "dpu-upmem-dpurte-clang", "dpu-pkg-config"):
        raw = sys.executable if name == "python" else shutil.which(name, path=env.get("PATH"))
        path = Path(raw).resolve() if raw else None
        values[name] = {
            "available": bool(path and path.is_file()),
            "path": str(path) if path else None,
            "sha256": sha256_file(path) if path and path.is_file() else None,
            "version": _tool_version(path) if path and path.is_file() else None,
        }
    values["host_cc"] = resolve_host_cc(env)
    return values
def fingerprint_catalog(catalog: ProviderCatalog, root_dir: Path) -> Json:
    root = resolve_root(root_dir)
    rows: Json = {}
    for provider in catalog.providers:
        runner = resolve_provider_path(root, provider.runner, label="provider runner") if provider.runner else None
        rows[provider.provider_id] = {
            "runner": {
                "path": _relative(runner, root) if runner else None,
                "sha256": sha256_file(runner) if runner and runner.is_file() else None,
            },
            "source": provider_source_fingerprint(provider, root) if provider.source_path else None,
        }
    return {
        "catalog_sha256": sha256_file(catalog.path),
        "benchmark_source_commit": _git(root, "rev-parse", "HEAD"),
        "providers": rows,
    }
def _hash_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)
def _identity_matches(actual: Any, expected: Any) -> bool:
    return isinstance(actual, Mapping) and isinstance(expected, Mapping) and actual.get("available") is True and all(actual.get(key) == expected.get(key) for key in ("command", "path", "sha256"))
def _device_evidence_proves_hardware(value: Any) -> bool:
    return isinstance(value, list) and any(isinstance(item, Mapping) and isinstance(item.get("path"), str) and bool(item["path"]) and all(item.get(key) is True for key in ("exists", "character_device", "readable", "writable")) for item in value)
def simplepim_runner_schema_errors(
    payload: Mapping[str, Any] | None,
    provider: ProviderSpec,
    *,
    mode: str,
    expected_host_cc: Mapping[str, Any] | None = None,
    expected_dpu_cc: Mapping[str, Any] | None = None,
    expected_preflight: Mapping[str, Any] | None = None,
) -> tuple[str, ...]:
    if not isinstance(payload, Mapping):
        return ("runner result must be an object",)
    errors: list[str] = []
    missing = REQUIRED_RESULT_FIELDS - payload.keys()
    if missing:
        errors.append(f"missing required fields: {', '.join(sorted(missing))}")
    if payload.get("schema_version") != (provider.runner_schema_version or SIMPLEPIM_RUNNER_SCHEMA_VERSION):
        errors.append("schema_version mismatch")
    if payload.get("provider_id") != provider.provider_id or payload.get("probe_id") != (provider.probe_id or SIMPLEPIM_PROBE_ID):
        errors.append("provider/probe mismatch")
    status = payload.get("status")
    if status not in ({"prepared"} if mode == "prepare" else {"passed", "failed"}):
        errors.append("invalid status")
    if payload.get("target") != payload.get("target_observed"):
        errors.append("target_observed must equal target")
    if payload.get("target") not in {None, PHYSICAL_TARGET}:
        errors.append("invalid target")
    if payload.get("requested_dpu_count") != provider.requested_dpus:
        errors.append("requested DPU count mismatch")
    if payload.get("configured_tasklets_per_dpu") != provider.requested_tasklets:
        errors.append("configured tasklet count mismatch")
    if payload.get("observed_tasklets_per_dpu") is not None:
        errors.append("observed tasklets must remain null")
    for key in (
        "hardware_preflight_verified",
        "native_execution",
        "validation_performed",
        "exact_validation",
        "fallback",
        "simulator_kernel_executed",
        "payload_sizes_8_byte_aligned",
        "physical_transfer_bytes_available",
    ):
        if type(payload.get(key)) is not bool:
            errors.append(f"{key} must be boolean")
    if payload.get("fallback") is not False:
        errors.append("fallback is not permitted")
    if payload.get("simulator_kernel_executed") is not False:
        errors.append("simulator execution is not permitted")
    if payload.get("backend_profile") != "backend=hw":
        errors.append("hardware backend profile required")
    if payload.get("payload_sizes_8_byte_aligned") is not True or payload.get("physical_transfer_bytes_available") is not False or payload.get("physical_transfer_bytes") is not None:
        errors.append("transfer contract mismatch")
    if payload.get("release_status") not in {
        "not_attempted",
        "released",
        "failed",
        "unknown",
    }:
        errors.append("invalid release status")
    if status != "passed" and not isinstance(payload.get("reason"), str):
        errors.append("failed result requires a reason")
    hashes = payload.get("source_hashes")
    if not isinstance(hashes, Mapping) or payload.get("source_hash") != hashes.get("combined_sha256"):
        errors.append("source hash does not bind to source_hashes")
    patch = payload.get("staged_patch")
    if not isinstance(patch, Mapping) or patch.get("path") != SIMPLEPIM_PATCH_PATH:
        errors.append("staged patch evidence is missing")
    if status == "prepared":
        inert = (
            payload.get("target") is None
            and payload.get("observed_dpu_count") is None
            and payload.get("hardware_preflight_verified") is False
            and payload.get("device_evidence") == []
            and payload.get("native_execution") is False
            and payload.get("validation_performed") is False
            and payload.get("exact_validation") is False
            and payload.get("release_status") == "not_attempted"
            and payload.get("binary_hashes") == {}
            and payload.get("input_hashes") == {}
            and payload.get("output_hash") is None
            and isinstance(patch, Mapping) and patch.get("applied") is False
            and patch.get("staged_sha256") is None
            and isinstance(hashes, Mapping) and all(hashes.get(key) is None for key in ("staged_source_before_patch_sha256", "staged_source_after_patch_sha256", "staged_patch_sha256"))
            and all(key not in payload for key in ("build", "execution", "host_result", "timeout_cleanup"))
        )
        if not inert:
            errors.append("prepared result must not claim staging, build, or hardware execution")
    if expected_host_cc and payload.get("effective_compilers", {}).get("host_cc", {}).get("sha256") != expected_host_cc.get("sha256"):
        errors.append("host compiler provenance mismatch")
    if expected_preflight and payload.get("requested_dpu_count") != expected_preflight.get("requested_dpu_count"):
        errors.append("execute/preflight mismatch")
    if status == "passed":
        devices = payload.get("device_evidence")
        build, execution, host = payload.get("build"), payload.get("execution"), payload.get("host_result")
        binaries, inputs = payload.get("binary_hashes"), payload.get("input_hashes")
        compilers = payload.get("effective_compilers")
        preflight_compilers = expected_preflight.get("effective_compilers") if isinstance(expected_preflight, Mapping) else None
        preflight_hashes = expected_preflight.get("source_hashes") if isinstance(expected_preflight, Mapping) else None
        preflight_patch = expected_preflight.get("staged_patch") if isinstance(expected_preflight, Mapping) else None
        post_patch = hashes.get("staged_source_after_patch_sha256") if isinstance(hashes, Mapping) else None
        patch_hash = patch.get("sha256") if isinstance(patch, Mapping) else None
        checks = (
            (payload.get("target") == PHYSICAL_TARGET and payload.get("target_observed") == PHYSICAL_TARGET, "physical target not proven"),
            (payload.get("hardware_preflight_verified") is True and _device_evidence_proves_hardware(devices), "hardware preflight/device evidence missing"),
            (isinstance(build, Mapping) and build.get("status") == "passed" and build.get("returncode") == 0, "build did not pass"),
            (isinstance(binaries, Mapping) and all(_is_sha256(binaries.get(name)) for name in BINARY_NAMES), "host/DPU binary hashes missing"),
            (isinstance(execution, Mapping) and execution.get("status") == "passed" and execution.get("returncode") == 0, "native command did not pass"),
            (isinstance(host, Mapping) and host.get("status") == "passed" and host.get("release_status") == "released" and host.get("native_run_completed") is True and host.get("validation_performed") is True and host.get("host_exact_validation") is True, "host result did not pass validation/release"),
            (payload.get("native_execution") is True and payload.get("validation_performed") is True and payload.get("exact_validation") is True, "native exact validation missing"),
            (payload.get("fallback") is False and payload.get("simulator_kernel_executed") is False, "fallback/simulator execution is forbidden"),
            (payload.get("observed_dpu_count") == provider.requested_dpus and payload.get("configured_tasklets_per_dpu") == provider.requested_tasklets, "observed/configured hardware counts mismatch"),
            (isinstance(inputs, Mapping) and _is_sha256(inputs.get("a_u32")) and _is_sha256(inputs.get("b_u32")) and _is_sha256(payload.get("output_hash")), "input/output hashes missing"),
            (_is_sha256(post_patch) and payload.get("source_hash") == hashes.get("combined_sha256") == post_patch, "staged post-patch source hash missing"),
            (_is_sha256(patch_hash) and patch.get("applied") is True and patch.get("staged_sha256") == patch_hash and hashes.get("patch_sha256") == patch_hash, "tracked patch hash missing"),
            (payload.get("release_status") == "released" and payload.get("failure_stage") is None and payload.get("reason") is None, "release/failure status contradicts pass"),
            (isinstance(hashes, Mapping) and isinstance(preflight_hashes, Mapping) and isinstance(preflight_patch, Mapping) and all(hashes.get(key) == preflight_hashes.get(key) for key in ("owned_qualification_sha256", "upstream_library_sha256", "patch_sha256")) and all(patch.get(key) == preflight_patch.get(key) for key in ("path", "sha256")), "execution source/patch fingerprint does not match preflight"),
            (isinstance(compilers, Mapping) and isinstance(preflight_compilers, Mapping) and _identity_matches(compilers.get("host_cc"), preflight_compilers.get("host_cc")) and _identity_matches(compilers.get("dpu_cc"), preflight_compilers.get("dpu_cc")), "compiler identities do not match preflight"),
            (expected_dpu_cc is None or _identity_matches(compilers.get("dpu_cc") if isinstance(compilers, Mapping) else None, expected_dpu_cc), "DPU compiler provenance mismatch"),
        )
        errors.extend(message for valid, message in checks if not valid)
    return tuple(dict.fromkeys(errors))
def parse_runner_result(
    payload: Mapping[str, Any],
    provider: ProviderSpec,
    *,
    expected_host_cc: Mapping[str, Any] | None = None,
    expected_dpu_cc: Mapping[str, Any] | None = None,
    expected_preflight: Mapping[str, Any] | None = None,
) -> RunnerQualification:
    errors = simplepim_runner_schema_errors(
        payload,
        provider,
        mode="execute",
        expected_host_cc=expected_host_cc,
        expected_dpu_cc=expected_dpu_cc,
        expected_preflight=expected_preflight,
    )
    qualified = payload.get("status") == "passed" and not errors
    status = "qualified" if qualified else "failed"
    reason = "physical SimplePIM qualification contract passed" if qualified else (errors[0] if errors else str(payload.get("reason")))
    return RunnerQualification(
        status,
        reason,
        errors,
        payload.get("status") if isinstance(payload.get("status"), str) else None,
        payload.get("hardware_preflight_verified") if type(payload.get("hardware_preflight_verified")) is bool else None,
        payload.get("target") if isinstance(payload.get("target"), str) else None,
        payload.get("observed_dpu_count") if type(payload.get("observed_dpu_count")) is int else None,
        payload.get("configured_tasklets_per_dpu") if type(payload.get("configured_tasklets_per_dpu")) is int else None,
        payload.get("observed_tasklets_per_dpu") if type(payload.get("observed_tasklets_per_dpu")) is int else None,
        payload.get("native_execution") if type(payload.get("native_execution")) is bool else None,
        payload.get("validation_performed") if type(payload.get("validation_performed")) is bool else None,
        payload.get("exact_validation") if type(payload.get("exact_validation")) is bool else None,
        payload.get("release_status") if isinstance(payload.get("release_status"), str) else None,
        payload.get("fallback") if type(payload.get("fallback")) is bool else None,
        payload.get("simulator_kernel_executed") if type(payload.get("simulator_kernel_executed")) is bool else None,
    )
