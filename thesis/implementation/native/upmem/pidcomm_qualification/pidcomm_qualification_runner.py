"""Isolated PID-Comm physical qualification runner."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[3]
THESIS_REPO_ROOT = ROOT.parents[1]
EXTERNAL_ROOT = ROOT / "external" / "PID-Comm"
SOURCE_ROOT = Path(__file__).resolve().parent / "qualification"
PINNED_COMMIT = "cecc39e29e6576ced73b2041db6e357769a6531a"
PIDCOMM_BUNDLED_SDK_VERSION = "2021.3.0"
CANDIDATE_DPU_COUNTS = (2, 4, 64)
PAYLOAD_BYTES = 256
PAYLOAD_DTYPE = "int32"
PAYLOAD_OPERATION = "sum_all_reduce"
PHYSICAL_OPT_IN = "UPMEM_ALLOW_PHYSICAL_HARDWARE"
SIMULATOR_SELECTOR_KEYS = (
    "DPU_BACKEND",
    "DPU_PROFILE",
    "UPMEM_BACKEND",
    "UPMEM_MODE",
    "UPMEM_TARGET",
    "UPMEM_PROFILE",
    "UPMEM_PROFILE_BASE",
)
SDK_BINARIES = (
    "alltoall_22",
    "alltoall_22_int32",
    "alltoall_22_int8",
    "alltoall_x_2",
    "alltoall_x_2_int32",
    "alltoall_x_2_int8",
    "ar_24_int32",
    "ar_24_int8",
    "dpu_ar_2_int32",
    "dpu_ar_2_int8",
    "dpu_ar_2_y_int32",
    "dpu_ar_2_y_int8",
    "data_relocate_clockwise_int32",
    "data_relocate_clockwise_int8",
    "data_relocate_clockwise",
    "dpu_user",
    "rs_22_int8",
    "rs_22_int32",
    "rs_24_int8",
    "rs_24_int32",
)
SOURCE_FILES = (
    "host.c",
    "dpu_user.c",
    "pidcomm_binary_paths.h",
    "Makefile",
)
EXTERNAL_SOURCE_FILES = (
    "pidcomm_lib/support/commlib.c",
    "upmem-2021.3.0_opt/include/dpu/pidcomm.h",
    "upmem-2021.3.0_opt/include/dpu/support.h",
)
REQUIRED_SYMBOLS = (
    "dpu_alloc_comm",
    "pidcomm_all_reduce",
    "init_hypercube_manager",
    "dpu_get_nr_dpus",
)
RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z")
SCHEMA_VERSION = "pidcomm_qualification_v1"


def _capture(command: Sequence[str], *, env: Mapping[str, str] | None = None) -> dict[str, Any]:
    try:
        result = subprocess.run(
            list(command),
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
            env=dict(env) if env is not None else None,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {"returncode": None, "stdout": "", "stderr": str(exc)}
    return {
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def _run_process(
    command: Sequence[str],
    *,
    cwd: Path,
    env: Mapping[str, str],
    timeout_seconds: float,
) -> dict[str, Any]:
    started = time.monotonic()
    try:
        result = subprocess.run(
            list(command),
            cwd=cwd,
            env=dict(env),
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "returncode": None,
            "stdout": _text(exc.stdout),
            "stderr": _text(exc.stderr),
            "timed_out": True,
            "elapsed_seconds": time.monotonic() - started,
        }
    except OSError as exc:
        return {
            "returncode": None,
            "stdout": "",
            "stderr": str(exc),
            "timed_out": False,
            "elapsed_seconds": time.monotonic() - started,
        }
    return {
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "timed_out": False,
        "elapsed_seconds": time.monotonic() - started,
    }


def _text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    return value.decode(errors="replace") if isinstance(value, bytes) else value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode("utf-8")).hexdigest()


def _file_record(path: Path, *, declared_path: str, kind: str) -> dict[str, Any]:
    return {
        "kind": kind,
        "declared_path": declared_path,
        "path": str(path),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _input_records() -> dict[str, list[dict[str, Any]]]:
    thesis = [
        _file_record(SOURCE_ROOT / relative, declared_path=relative, kind="thesis_owned")
        for relative in SOURCE_FILES
    ]
    external = [
        _file_record(EXTERNAL_ROOT / relative, declared_path=relative, kind="pidcomm_source")
        for relative in EXTERNAL_SOURCE_FILES
    ]
    external.extend(
        _file_record(
            _binary_source(name),
            declared_path=str(_binary_source(name).relative_to(EXTERNAL_ROOT)),
            kind="pidcomm_prebuilt_binary",
        )
        for name in SDK_BINARIES
    )
    return {"thesis_owned": thesis, "pidcomm_external": external}


def _is_under(path: str | None, root: Path) -> bool:
    if not path:
        return False
    candidate = Path(path).expanduser().resolve()
    root = root.resolve()
    return candidate == root or root in candidate.parents


def _resolve_tool(name: str, environment: Mapping[str, str]) -> str | None:
    configured = environment.get(name.upper().replace("-", "_"))
    if configured:
        return str(Path(configured).expanduser().resolve())
    return shutil.which(name, path=environment.get("PATH"))


def _sdk_root(pkg_config: str | None, environment: Mapping[str, str]) -> str | None:
    if not pkg_config:
        return None
    result = _capture([pkg_config, "--variable=prefix", "dpu"], env=environment)
    if result["returncode"] == 0 and result["stdout"].strip():
        return str(Path(result["stdout"].strip()).resolve())
    return None


def _pkg_config_fact(pkg_config: str | None, environment: Mapping[str, str]) -> dict[str, Any]:
    fact: dict[str, Any] = {"path": pkg_config, "cflags": None, "libs": None}
    if not pkg_config:
        return fact
    for key, command in (
        ("cflags", [pkg_config, "--cflags", "dpu"]),
        ("libs", [pkg_config, "--libs", "dpu"]),
    ):
        result = _capture(command, env=environment)
        fact[key] = {
            "command": command,
            "returncode": result["returncode"],
            "stdout": result["stdout"].strip(),
            "stderr": result["stderr"].strip(),
        }
    return fact


def _version_file(sdk_root: str | None) -> dict[str, Any]:
    if not sdk_root:
        return {"path": None, "text": None}
    path = Path(sdk_root) / "share" / "upmem" / "version"
    try:
        return {"path": str(path), "text": path.read_text(encoding="utf-8").strip()}
    except OSError as exc:
        return {"path": str(path), "text": None, "error": str(exc)}


def _cpu_flags(environment: Mapping[str, str]) -> dict[str, Any]:
    result = _capture(["lscpu"], env=environment)
    flags = ""
    for line in result["stdout"].splitlines():
        if line.lower().startswith(("flags:", "flags ", "features:")):
            flags = line.split(":", 1)[-1].strip()
            break
    if not flags:
        try:
            for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
                if line.lower().startswith(("flags", "features")) and ":" in line:
                    flags = line.split(":", 1)[1].strip()
                    break
        except OSError:
            pass
    flag_set = sorted(set(flags.split()))
    return {
        "command": ["lscpu"],
        "flags": flag_set,
        "avx512f": "avx512f" in flag_set,
        "returncode": result["returncode"],
        "stderr": result["stderr"].strip(),
    }


def _tool_identity(path: str | None, version_command: Sequence[str], environment: Mapping[str, str]) -> dict[str, Any]:
    if not path:
        return {"path": None, "version": None, "returncode": None}
    result = _capture(version_command, env=environment)
    return {
        "path": path,
        "version": (result["stdout"] or result["stderr"]).strip(),
        "returncode": result["returncode"],
    }


def _git_fact(command: Sequence[str], environment: Mapping[str, str]) -> dict[str, Any]:
    result = _capture(command, env=environment)
    return {
        "command": list(command),
        "returncode": result["returncode"],
        "stdout": result["stdout"].strip(),
        "stderr": result["stderr"].strip(),
    }


def _run_id() -> str:
    now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{now}-{os.getpid()}"


def _validate_run_id(run_id: str) -> str:
    if not isinstance(run_id, str) or not RUN_ID_PATTERN.fullmatch(run_id) or run_id in {".", ".."}:
        raise ValueError("run_id must be one safe path component matching [A-Za-z0-9][A-Za-z0-9._-]{0,63}")
    return run_id


def _require_inputs() -> None:
    for relative in SOURCE_FILES:
        if not (SOURCE_ROOT / relative).is_file():
            raise ValueError(f"missing thesis-owned qualification source: {SOURCE_ROOT / relative}")
    for relative in EXTERNAL_SOURCE_FILES:
        if not (EXTERNAL_ROOT / relative).is_file():
            raise ValueError(f"missing pinned PID-Comm source: {EXTERNAL_ROOT / relative}")
    for name in SDK_BINARIES:
        if not _binary_source(name).is_file():
            raise ValueError(f"missing pinned PID-Comm DPU binary: {name}")


def _binary_source(name: str) -> Path:
    sdk_root = EXTERNAL_ROOT / "upmem-2021.3.0_opt"
    include_binary = sdk_root / "include" / "dpu" / "bin" / name
    return include_binary if include_binary.is_file() else sdk_root / "bin" / name


EXTERNAL_FILES = EXTERNAL_SOURCE_FILES + tuple(
    str(_binary_source(name).relative_to(EXTERNAL_ROOT)) for name in SDK_BINARIES
)


def qualification_plan(workdir: Path, *, run_id: str | None = None, environment: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Collect an allocation-free plan and toolchain manifest."""
    _require_inputs()
    root = workdir.expanduser().resolve()
    env = dict(os.environ if environment is None else environment)
    host_cc = _resolve_tool("gcc", env)
    dpu_cc = _resolve_tool("dpu-upmem-dpurte-clang", env)
    pkg_config = _resolve_tool("dpu-pkg-config", env)
    sdk_root = _sdk_root(pkg_config, env)
    rid = _validate_run_id(run_id if run_id is not None else _run_id())
    build_root = root / "build" / "pidcomm_qualification" / rid
    stage_root = build_root / "staged"
    run_root = root / "runs" / "pidcomm_qualification" / rid
    git_head = _git_fact(["git", "-C", str(EXTERNAL_ROOT), "rev-parse", "HEAD"], env)
    git_status = _git_fact(["git", "-C", str(EXTERNAL_ROOT), "status", "--porcelain"], env)
    thesis_head = _git_fact(["git", "-C", str(THESIS_REPO_ROOT), "rev-parse", "HEAD"], env)
    thesis_status = _git_fact(["git", "-C", str(THESIS_REPO_ROOT), "status", "--porcelain"], env)
    input_records = _input_records()
    pkg_config_fact = _pkg_config_fact(pkg_config, env)
    source_fingerprint = {
        "path": str(EXTERNAL_ROOT),
        "pinned_commit": PINNED_COMMIT,
        "head_commit": git_head["stdout"],
        "commit_matches_pin": git_head["stdout"] == PINNED_COMMIT,
        "clean": git_status["returncode"] == 0 and git_status["stdout"] == "",
        "status": git_status,
        "thesis_git": {
            "repository": str(THESIS_REPO_ROOT),
            "head_commit": thesis_head["stdout"],
            "dirty": thesis_status["returncode"] != 0 or bool(thesis_status["stdout"]),
            "status": thesis_status,
        },
        "inputs": input_records,
        "input_hash": _hash_json(input_records),
    }
    bundled_tool_paths = [
        name
        for name, path in (
            ("host_compiler", host_cc),
            ("dpu_compiler", dpu_cc),
            ("dpu_pkg_config", pkg_config),
        )
        if _is_under(path, EXTERNAL_ROOT)
    ]
    bundled_ld_library_path = any(
        _is_under(item, EXTERNAL_ROOT)
        for item in env.get("LD_LIBRARY_PATH", "").split(os.pathsep)
        if item
    )
    commands = {
        "build": [
            "make",
            "-C",
            str(stage_root),
            f"STAGE={stage_root}",
            f"HOST_CC={host_cc or 'gcc'}",
            f"DPU_CC={dpu_cc or 'dpu-upmem-dpurte-clang'}",
            f"DPU_PKG_CONFIG={pkg_config or 'dpu-pkg-config'}",
            "all",
        ],
        "candidate_runs": [
            [str(stage_root / "bin" / "pidcomm_qualifier"), str(count)]
            for count in CANDIDATE_DPU_COUNTS
        ],
        "compatibility": [
            "make",
            "-C",
            str(stage_root),
            f"STAGE={stage_root}",
            f"HOST_CC={host_cc or 'gcc'}",
            f"DPU_CC={dpu_cc or 'dpu-upmem-dpurte-clang'}",
            f"DPU_PKG_CONFIG={pkg_config or 'dpu-pkg-config'}",
            "compatibility",
        ],
    }
    plan = {
        "schema_version": SCHEMA_VERSION,
        "status": "prepared",
        "qualification": "isolated_pidcomm_all_reduce",
        "workdir": str(root),
        "run_id": rid,
        "build_root": str(build_root),
        "stage_root": str(stage_root),
        "run_root": str(run_root),
        "cpu": _cpu_flags(env),
        "toolchain": {
            "host_compiler": _tool_identity(host_cc, [host_cc, "--version"] if host_cc else [], env),
            "dpu_compiler": _tool_identity(dpu_cc, [dpu_cc, "--version"] if dpu_cc else [], env),
            "dpu_pkg_config": _tool_identity(pkg_config, [pkg_config, "--version"] if pkg_config else [], env),
            "system_sdk_root": sdk_root,
            "system_sdk_version": _version_file(sdk_root),
            "linked_sdk_path": sdk_root,
            "linked_sdk_version": _version_file(sdk_root),
            "pkg_config": pkg_config_fact,
            "pidcomm_bundled_sdk_version": PIDCOMM_BUNDLED_SDK_VERSION,
            "pidcomm_bundled_sdk_used": False,
            "provenance": "system UPMEM SDK linked with staged PID-Comm source and prebuilt binaries",
            "bundled_tool_paths": bundled_tool_paths,
        },
        "source": source_fingerprint,
        "contract": {
            "candidate_dpu_counts": list(CANDIDATE_DPU_COUNTS),
            "payload_bytes": PAYLOAD_BYTES,
            "payload_dtype": PAYLOAD_DTYPE,
            "operation": PAYLOAD_OPERATION,
        },
        "commands": commands,
        "execution_policy": {
            "physical_opt_in": f"{PHYSICAL_OPT_IN}=1",
            "simulator_fallback": False,
            "bundled_sdk_on_path": bool(bundled_tool_paths),
            "bundled_sdk_in_ld_library_path": bundled_ld_library_path,
            "dpu_allocation_attempted": False,
            "dpu_launch_attempted": False,
        },
        "command_fingerprint": _hash_json(commands),
        "required_symbols": list(REQUIRED_SYMBOLS),
        "allocation_free": True,
    }
    return plan


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_log(path: Path, result: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "COMMAND RESULT\n"
        f"returncode={result.get('returncode')}\n"
        f"timed_out={result.get('timed_out', False)}\n\n"
        "STDOUT\n"
        f"{result.get('stdout', '')}\n"
        "STDERR\n"
        f"{result.get('stderr', '')}\n",
        encoding="utf-8",
    )


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _blocked(
    plan: Mapping[str, Any],
    run_root: Path,
    *,
    stage: str,
    detail: str,
    log_path: Path | None = None,
    candidates: list[dict[str, Any]] | None = None,
    allocation_attempted: bool = False,
    launch_attempted: bool = False,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "event": "pidcomm_sdk_compatibility_blocked",
        "status": "blocked",
        "reason": "pidcomm_sdk_compatibility_blocked",
        "failure_stage": stage,
        "detail": detail,
        "log_path": _relative(log_path, run_root.parent.parent.parent) if log_path else None,
        "run_id": plan["run_id"],
        "candidate_dpu_counts": plan["contract"]["candidate_dpu_counts"],
        "payload_bytes": plan["contract"]["payload_bytes"],
        "payload_dtype": plan["contract"]["payload_dtype"],
        "operation": plan["contract"]["operation"],
        "fallback": False,
        "simulator_kernel_executed": False,
        "dpu_allocation_attempted": allocation_attempted,
        "dpu_launch_attempted": launch_attempted,
        "candidates": candidates or [],
    }


def _preflight(plan: Mapping[str, Any], environment: Mapping[str, str]) -> tuple[bool, str]:
    if environment.get(PHYSICAL_OPT_IN) != "1":
        return False, f"{PHYSICAL_OPT_IN}=1 is required"
    selectors = [key for key in SIMULATOR_SELECTOR_KEYS if environment.get(key)]
    if selectors:
        return False, f"simulator/backend selectors are forbidden: {','.join(selectors)}"
    if not plan["cpu"]["avx512f"]:
        return False, "required CPU flag avx512f is absent"
    source = plan["source"]
    if not source["commit_matches_pin"]:
        return False, f"PID-Comm checkout is not pinned to {PINNED_COMMIT}"
    if not source["clean"]:
        return False, "PID-Comm checkout is not clean"
    toolchain = plan["toolchain"]
    for name in ("host_compiler", "dpu_compiler", "dpu_pkg_config"):
        if not toolchain[name]["path"]:
            return False, f"missing installed tool: {name}"
    sdk_root = toolchain["system_sdk_root"]
    if _is_under(sdk_root, EXTERNAL_ROOT):
        return False, "installed SDK resolves inside the PID-Comm checkout"
    if toolchain["bundled_tool_paths"]:
        return False, "installed compiler or pkg-config resolves inside the PID-Comm checkout"
    if _is_under(environment.get("UPMEM_HOME"), EXTERNAL_ROOT):
        return False, "UPMEM_HOME resolves inside the PID-Comm checkout"
    if any(_is_under(item, EXTERNAL_ROOT) for item in environment.get("LD_LIBRARY_PATH", "").split(os.pathsep) if item):
        return False, "LD_LIBRARY_PATH contains the PID-Comm checkout"
    return True, "preflight_passed"


def _stage(plan: Mapping[str, Any]) -> dict[str, Any]:
    stage_root = Path(plan["stage_root"])
    if stage_root.exists():
        raise ValueError(f"staging directory already exists: {stage_root}")
    stage_root.mkdir(parents=True)
    for relative in SOURCE_FILES:
        shutil.copy2(SOURCE_ROOT / relative, stage_root / relative)
    include = stage_root / "include"
    include.mkdir()
    shutil.copy2(SOURCE_ROOT / "pidcomm_binary_paths.h", include / "pidcomm_binary_paths.h")
    for relative in ("pidcomm.h", "support.h"):
        shutil.copy2(EXTERNAL_ROOT / "upmem-2021.3.0_opt" / "include" / "dpu" / relative, include / relative)
    shutil.copy2(EXTERNAL_ROOT / "pidcomm_lib" / "support" / "commlib.c", stage_root / "pidcomm_commlib.c")
    binary_dir = stage_root / "pidcomm_bin"
    binary_dir.mkdir()
    for name in SDK_BINARIES:
        shutil.copy2(_binary_source(name), binary_dir / name)
    staged_hashes: list[dict[str, Any]] = []
    for relative in SOURCE_FILES:
        staged_hashes.append(
            _file_record(stage_root / relative, declared_path=relative, kind="thesis_owned_staged")
        )
    for relative in ("pidcomm_commlib.c", "include/pidcomm.h", "include/support.h"):
        staged_hashes.append(
            _file_record(stage_root / relative, declared_path=relative, kind="pidcomm_source_staged")
        )
    staged_hashes.append(
        _file_record(
            stage_root / "include/pidcomm_binary_paths.h",
            declared_path="include/pidcomm_binary_paths.h",
            kind="thesis_owned_staged",
        )
    )
    for name in SDK_BINARIES:
        staged_hashes.append(
            _file_record(binary_dir / name, declared_path=f"pidcomm_bin/{name}", kind="pidcomm_binary_staged")
        )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "source_commit": plan["source"]["head_commit"],
        "thesis_git": plan["source"]["thesis_git"],
        "thesis_sources": list(SOURCE_FILES),
        "pidcomm_sources": list(EXTERNAL_SOURCE_FILES),
        "staged_binaries": list(SDK_BINARIES),
        "input_hash": plan["source"]["input_hash"],
        "input_records": plan["source"]["inputs"],
        "staged_hashes": staged_hashes,
        "linked_sdk_path": plan["toolchain"]["linked_sdk_path"],
        "linked_sdk_version": plan["toolchain"]["linked_sdk_version"],
        "bundled_sdk_staged": False,
    }
    _write_json(stage_root / "stage_manifest.json", manifest)
    return manifest


def _physical_environment(environment: Mapping[str, str]) -> dict[str, str]:
    result = dict(environment)
    result["DPU_BACKEND"] = "hw"
    result["UPMEM_PROFILE_BASE"] = "backend=hw"
    return result


def _expected_topology(count: int) -> dict[str, Any]:
    topology = {
        2: ([2, 1, 1], "100"),
        4: ([2, 2, 1], "110"),
        64: ([8, 8, 1], "110"),
    }
    axis_lengths, communicator = topology[count]
    return {"dimension": 3, "axis_lengths": axis_lengths, "communicator": communicator}


def _host_manifest_errors(count: int, manifest: Mapping[str, Any] | None) -> list[str]:
    if manifest is None:
        return ["missing host manifest"]
    errors: list[str] = []
    expected = {
        "status": "passed",
        "dpu_count": count,
        "payload_bytes": PAYLOAD_BYTES,
        "payload_dtype": PAYLOAD_DTYPE,
        "operation": PAYLOAD_OPERATION,
        "topology": _expected_topology(count),
        "hardware_observed": True,
        "fallback": False,
        "pidcomm_api": "pidcomm_all_reduce",
    }
    for key, value in expected.items():
        actual = manifest.get(key)
        if isinstance(value, bool):
            valid = type(actual) is bool and actual is value
        elif isinstance(value, int):
            valid = type(actual) is int and actual == value
        else:
            valid = actual == value
        if not valid:
            errors.append(f"{key}={manifest.get(key)!r}, expected {value!r}")
    return errors


def _candidate_result(count: int, result: Mapping[str, Any], log_path: Path, root: Path) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "dpu_count": count,
        "status": "passed" if result.get("returncode") == 0 else "failed",
        "returncode": result.get("returncode"),
        "timed_out": result.get("timed_out", False),
        "log_path": _relative(log_path, root),
    }
    lines = [line for line in str(result.get("stdout", "")).splitlines() if line.strip()]
    if lines:
        try:
            payload["host_manifest"] = json.loads(lines[-1])
        except json.JSONDecodeError:
            payload["host_manifest_parse_error"] = True
    manifest = payload.get("host_manifest")
    payload["validation_errors"] = _host_manifest_errors(
        count, manifest if isinstance(manifest, Mapping) else None
    )
    payload["host_manifest_valid"] = result.get("returncode") == 0 and not payload["validation_errors"]
    payload["status"] = "passed" if payload["host_manifest_valid"] else "failed"
    return payload


def execute(plan: dict[str, Any], environment: Mapping[str, str], *, timeout_seconds: float) -> dict[str, Any]:
    run_root = Path(plan["run_root"])
    run_root.mkdir(parents=True, exist_ok=False)
    preflight_payload = {
        "schema_version": SCHEMA_VERSION,
        "allocation_free": True,
        "physical_opt_in": environment.get(PHYSICAL_OPT_IN) == "1",
        "cpu": plan["cpu"],
        "toolchain": plan["toolchain"],
        "source": plan["source"],
        "contract": plan["contract"],
    }
    ok, detail = _preflight(plan, environment)
    preflight_payload["status"] = "passed" if ok else "blocked"
    preflight_payload["detail"] = detail
    _write_json(run_root / "preflight.json", preflight_payload)
    if not ok:
        return _blocked(plan, run_root, stage="preflight", detail=detail, log_path=run_root / "preflight.json")

    try:
        _stage(plan)
    except (OSError, ValueError) as exc:
        log_path = run_root / "logs" / "staging.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(str(exc) + "\n", encoding="utf-8")
        return _blocked(plan, run_root, stage="staging", detail=str(exc), log_path=log_path)

    env = _physical_environment(environment)
    compatibility_result = _run_process(
        plan["commands"]["compatibility"],
        cwd=Path(plan["stage_root"]),
        env=env,
        timeout_seconds=timeout_seconds,
    )
    compatibility_log = run_root / "logs" / "compatibility.log"
    _write_log(compatibility_log, compatibility_result)
    if compatibility_result.get("returncode") != 0:
        return _blocked(
            plan,
            run_root,
            stage="compatibility_preflight",
            detail="installed SDK/PID-Comm symbols or ABI are incompatible; no DPU allocation attempted",
            log_path=compatibility_log,
        )

    build_result = _run_process(plan["commands"]["build"], cwd=Path(plan["stage_root"]), env=env, timeout_seconds=timeout_seconds)
    build_log = run_root / "logs" / "build.log"
    _write_log(build_log, build_result)
    if build_result.get("returncode") != 0:
        return _blocked(plan, run_root, stage="build", detail="installed SDK/PID-Comm ABI or build incompatibility", log_path=build_log)

    candidates: list[dict[str, Any]] = []
    for count, command in zip(plan["contract"]["candidate_dpu_counts"], plan["commands"]["candidate_runs"], strict=True):
        result = _run_process(command, cwd=Path(plan["stage_root"]), env=env, timeout_seconds=timeout_seconds)
        log_path = run_root / "logs" / f"candidate-{count}.log"
        _write_log(log_path, result)
        candidate = _candidate_result(count, result, log_path, run_root.parent)
        candidates.append(candidate)
        if candidate["host_manifest_valid"]:
            manifest = candidate["host_manifest"]
            payload = {
                "schema_version": SCHEMA_VERSION,
                "event": "pidcomm_qualification_passed",
                "status": "qualified",
                "selected_dpu_count": count,
                "candidate_dpu_counts": plan["contract"]["candidate_dpu_counts"],
                "payload_bytes": PAYLOAD_BYTES,
                "payload_dtype": PAYLOAD_DTYPE,
                "operation": PAYLOAD_OPERATION,
                "topology": manifest["topology"],
                "hardware_observed": manifest["hardware_observed"],
                "fallback": False,
                "pidcomm_api": manifest["pidcomm_api"],
                "simulator_kernel_executed": False,
                "dpu_allocation_attempted": True,
                "dpu_launch_attempted": True,
                "candidates": candidates,
            }
            _write_json(run_root / "result.json", payload)
            return payload

    last_log = run_root / "logs" / f"candidate-{plan['contract']['candidate_dpu_counts'][-1]}.log"
    payload = _blocked(
        plan,
        run_root,
        stage="runtime",
        detail="all physical candidates failed or returned an invalid host manifest",
        log_path=last_log,
        candidates=candidates,
        allocation_attempted=True,
        launch_attempted=True,
    )
    _write_json(run_root / "result.json", payload)
    return payload


def compatibility_probe(plan: dict[str, Any], environment: Mapping[str, str], *, timeout_seconds: float) -> dict[str, Any]:
    """Stage and compile/link the qualifier without requiring opt-in or hardware."""
    run_root = Path(plan["run_root"])
    run_root.mkdir(parents=True, exist_ok=False)
    try:
        _stage(plan)
    except (OSError, ValueError) as exc:
        log_path = run_root / "logs" / "staging.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(str(exc) + "\n", encoding="utf-8")
        payload = _blocked(plan, run_root, stage="staging", detail=str(exc), log_path=log_path)
        _write_json(run_root / "result.json", payload)
        return payload

    result = _run_process(
        plan["commands"]["compatibility"],
        cwd=Path(plan["stage_root"]),
        env=dict(environment),
        timeout_seconds=timeout_seconds,
    )
    log_path = run_root / "logs" / "compatibility.log"
    _write_log(log_path, result)
    if result.get("returncode") != 0:
        payload = _blocked(
            plan,
            run_root,
            stage="compatibility_preflight",
            detail="installed SDK/PID-Comm symbols or ABI are incompatible; no DPU allocation attempted",
            log_path=log_path,
        )
    else:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "event": "pidcomm_compatibility_probe_passed",
            "status": "compatible",
            "run_id": plan["run_id"],
            "required_symbols": list(REQUIRED_SYMBOLS),
            "fallback": False,
            "simulator_kernel_executed": False,
            "dpu_allocation_attempted": False,
            "dpu_launch_attempted": False,
            "log_path": _relative(log_path, run_root.parent),
        }
    _write_json(run_root / "result.json", payload)
    return payload


def _emit(payload: Mapping[str, Any], output: Path | None) -> int:
    encoded = json.dumps(payload, sort_keys=True)
    print(encoded)
    if output:
        _write_json(output, payload)
    return 0 if payload.get("status") in {"prepared", "compatible", "qualified"} else 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="isolated PID-Comm physical qualifier")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--compatibility-only", action="store_true")
    parser.add_argument("--workdir", type=Path, default=ROOT)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--timeout-seconds", type=float, default=600.0)
    args = parser.parse_args(argv)
    try:
        plan = qualification_plan(args.workdir, run_id=args.run_id)
        plan_path = Path(plan["build_root"]) / "plan.json"
        _write_json(plan_path, plan)
        if args.prepare_only:
            payload = {
                "schema_version": SCHEMA_VERSION,
                "status": "prepared",
                "event": "pidcomm_qualification_plan_prepared",
                "plan_path": str(plan_path),
                "allocation_free": True,
                "execution_policy": plan["execution_policy"],
                "contract": plan["contract"],
            }
            return _emit(payload, args.json_output)
        if args.compatibility_only:
            payload = compatibility_probe(plan, os.environ, timeout_seconds=args.timeout_seconds)
        else:
            payload = execute(plan, os.environ, timeout_seconds=args.timeout_seconds)
        payload["plan_path"] = str(plan_path)
        _write_json(Path(plan["run_root"]) / "result.json", payload)
        return _emit(payload, args.json_output)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "event": "pidcomm_sdk_compatibility_blocked",
            "status": "blocked",
            "reason": "pidcomm_sdk_compatibility_blocked",
            "failure_stage": "preflight",
            "detail": f"{type(exc).__name__}: {exc}",
            "fallback": False,
            "simulator_kernel_executed": False,
            "dpu_allocation_attempted": False,
            "dpu_launch_attempted": False,
        }
        return _emit(payload, args.json_output)


if __name__ == "__main__":
    raise SystemExit(main())
