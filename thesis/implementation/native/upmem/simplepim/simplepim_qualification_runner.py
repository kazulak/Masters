"""Prepare or execute the isolated physical SimplePIM qualification probe."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import stat
import struct
import subprocess
import time
from typing import Any, Mapping, Sequence
SCHEMA_VERSION = "simplepim_provider_qualification_v1"
HOST_SCHEMA_VERSION = "simplepim_qualification_host_v2"
PROVIDER_ID = "simplepim"
PROBE_ID = "simplepim_va_map_zip_v1"
REQUESTED_DPUS = 1
CONFIGURED_TASKLETS = 12
ELEMENTS = 256
UINT32_BYTES = 4
LOGICAL_INPUT_BYTES = 2 * ELEMENTS * UINT32_BYTES
LOGICAL_OUTPUT_BYTES = ELEMENTS * UINT32_BYTES
LOGICAL_TOTAL_BYTES = LOGICAL_INPUT_BYTES + LOGICAL_OUTPUT_BYTES
DEFAULT_TIMEOUT_SECONDS = 60.0
PATCH_TIMEOUT_SECONDS = 10.0
PATCH_RELATIVE_PATH = "patches/simplepim-map-unroll-rest.patch"
PATCH_TARGET = "lib/processing/map/MapProcessing.h"
BUGGY_UNROLL_LINE = "uint32_t unroll_block_rest = copy_block_size-unroll_block_rest;"
FIXED_UNROLL_LINE = "uint32_t unroll_block_rest = copy_block_size-unroll_block_size;"
BACKEND_PROFILE = "backend=hw"
PHYSICAL_TARGET = "physical_hardware"
HOST_CC = "gcc"
DPU_CC = "dpu-upmem-dpurte-clang"
BACKEND_SELECTOR_KEYS = ("DPU_BACKEND", "DPU_PROFILE", "SIMPLEPIM_BACKEND", "UPMEM_BACKEND", "UPMEM_MODE", "UPMEM_TARGET", "UPMEM_PROFILE", "UPMEM_PROFILE_BASE")
SIMULATOR_ALIASES = {"sim", "simulation", "simulator", "fsim", "casim", "functional_simulator"}
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SimplePIM physical qualification runner")
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--prepare-only", action="store_true")
    modes.add_argument("--execute", action="store_true")
    parser.add_argument("--workdir", required=True, type=Path)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--timeout-seconds", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--backend", default="physical")
    parser.add_argument("--target", default="physical_hardware")
    args = parser.parse_args(argv)
    try:
        plan = qualification_plan(args.workdir)
        if reason := _cli_selector_rejection(args.backend, args.target):
            return _emit(_failure_payload(plan, "backend_selection", reason), args.json_output)
        if args.prepare_only:
            payload = _base_payload(plan, "prepared")
            payload.update(
                commands=plan["commands"],
                reason="prepare_only_no_compiler_or_hardware_invoked",
            )
            return _emit(payload, args.json_output)
        return _execute(plan, args, args.json_output)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        return _emit(
            _failure_payload(_minimal_plan(args.workdir), "preflight", f"{type(exc).__name__}: {exc}"),
            args.json_output,
        )
def qualification_plan(workdir: Path) -> dict[str, Any]:
    """Fingerprint canonical sources and commands without creating a build."""
    workdir = workdir.expanduser().resolve()
    source_root = Path(__file__).resolve().parent / "qualification"
    external_root = Path(__file__).resolve().parents[3] / "external" / "SimplePIM"
    for relative in ("ATTRIBUTION.md", "Makefile", "Param.h", "host.c", "small_table_init.c", "va_funcs/map.h", PATCH_RELATIVE_PATH):
        _require_file(source_root / relative)
    for relative in ("lib/communication/CommHelper.c", "lib/communication/CommOps.c", "lib/management/Management.c", "lib/management/SmallTableInit_dpu.c", "lib/processing/ProcessingHelper.c", "lib/processing/map/Map.c", "lib/processing/map/MapProcessing.h", "lib/processing/map/map_dpu.c", "lib/processing/zip/Zip.c", "lib/processing/zip/ZipProcessing.c", "lib/processing/zip/zip_dpu.c"):
        _require_file(external_root / relative)
    host = (source_root / "host.c").read_text(encoding="utf-8")
    va_map = (source_root / "va_funcs/map.h").read_text(encoding="utf-8")
    upstream_map = (external_root / PATCH_TARGET).read_text(encoding="utf-8")
    patch = (source_root / PATCH_RELATIVE_PATH).read_text(encoding="utf-8")
    if "table_zip" not in host or "table_map" not in host or "map_func" not in va_map:
        raise ValueError("qualification source does not contain the VA map/zip contract")
    if upstream_map.count(BUGGY_UNROLL_LINE) != 2 or patch.count(BUGGY_UNROLL_LINE) != 2 or patch.count(FIXED_UNROLL_LINE) != 2:
        raise ValueError("pinned MapProcessing.h and patch do not match the two-defect contract")
    build_root = workdir / "build"
    staged_simplepim = build_root / "staged" / "SimplePIM"
    staged_benchmark = staged_simplepim / "benchmarks" / "va"
    host_cc = _resolve_host_cc(os.environ)
    commands = {
        "patch": [
            "git",
            "apply",
            "--no-index",
            "--verbose",
            str(staged_benchmark / PATCH_RELATIVE_PATH),
        ],
        "build": [
            "make",
            "-f",
            str(staged_benchmark / "Makefile"),
            "clean",
            "all",
            f"HOST_CC={host_cc}",
            f"DPU_CC={DPU_CC}",
        ],
        "run": [
            str(staged_benchmark / "bin/host"),
            str(build_root / "inputs/a_u32.bin"),
            str(build_root / "inputs/b_u32.bin"),
            str(build_root / "outputs/result_u32.bin"),
        ],
    }
    command_environment = {
        "patch": {
            # Git 2.25 can discover an enclosing worktree even with
            # --no-index and silently skip this patch. Stop discovery above
            # the isolated staged tree for deterministic patch application.
            "GIT_CEILING_DIRECTORIES": str(staged_simplepim.parent),
        }
    }
    owned_hash = _hash_tree(source_root)
    library_hash = _hash_tree(external_root / "lib")
    patch_hash = _hash_file(source_root / PATCH_RELATIVE_PATH)
    return {
        "workdir": workdir,
        "build_root": build_root,
        "source_root": source_root,
        "external_root": external_root,
        "staged_simplepim": staged_simplepim,
        "staged_benchmark": staged_benchmark,
        "binary_dir": staged_benchmark / "bin",
        "inputs_dir": build_root / "inputs",
        "outputs_dir": build_root / "outputs",
        "patch_path": source_root / PATCH_RELATIVE_PATH,
        "staged_patch_path": staged_benchmark / PATCH_RELATIVE_PATH,
        "host_cc": host_cc,
        "commands": commands,
        "command_environment": command_environment,
        "command_fingerprint": _hash_json({"commands": commands, "environment": command_environment}),
        "effective_compilers": {
            "host_cc": _compiler_identity(HOST_CC, Path(host_cc)),
            "dpu_cc": _compiler_identity(DPU_CC),
        },
        "source_fingerprint": {
            "owned_qualification_sha256": owned_hash,
            "upstream_library_sha256": library_hash,
            "upstream_map_processing_sha256": _hash_file(external_root / PATCH_TARGET),
            "patch_sha256": patch_hash,
            "staged_source_before_patch_sha256": None,
            "staged_source_after_patch_sha256": None,
            "staged_patch_sha256": None,
            "combined_sha256": _combine_hashes((owned_hash, library_hash, patch_hash)),
            "upstream_submodule": str(external_root),
        },
        "staged_patch": {
            "path": PATCH_RELATIVE_PATH,
            "sha256": patch_hash,
            "staged_sha256": None,
            "applied": False,
            "replacement_count": 0,
            "command_fingerprint": _hash_json(
                {"command": commands["patch"], "environment": command_environment["patch"]}
            ),
            "environment": dict(command_environment["patch"]),
            "staged_source_before_sha256": None,
            "staged_source_after_sha256": None,
            "staged_target_sha256": None,
        },
    }
def _execute(plan: Mapping[str, Any], args: argparse.Namespace, json_output: Path | None) -> int:
    if os.environ.get("UPMEM_ALLOW_PHYSICAL_HARDWARE") != "1":
        return _emit(
            _failure_payload(plan, "opt_in", "UPMEM_ALLOW_PHYSICAL_HARDWARE=1_required"),
            json_output,
        )
    if selector := _external_backend_selector(os.environ):
        return _emit(
            _failure_payload(
                plan,
                "backend_selection",
                f"external_backend_selector_rejected:{selector}",
            ),
            json_output,
        )
    preflight = _hardware_device_preflight()
    if not preflight["verified"]:
        payload = _failure_payload(plan, "hardware_preflight", str(preflight["reason"]))
        _record_hardware_preflight(payload, preflight)
        return _emit(payload, json_output)
    try:
        staged = _stage_sources(plan)
    except subprocess.TimeoutExpired:
        payload = _failure_payload(plan, "staged_patch_timeout", "tracked_patch_application_timed_out")
        _record_hardware_preflight(payload, preflight)
        return _emit(payload, json_output)
    except (OSError, ValueError) as exc:
        payload = _failure_payload(plan, "staged_patch", f"staged_patch_failed:{exc}")
        _record_hardware_preflight(payload, preflight)
        return _emit(payload, json_output)
    env = _physical_environment(os.environ, str(plan["host_cc"]))
    started = time.perf_counter()
    build = _run_command(plan["commands"]["build"], plan["staged_benchmark"], env, args.timeout_seconds)
    build_time = time.perf_counter() - started
    if build["status"] != "passed":
        payload = _failure_payload(
            plan,
            "build_timeout" if build["status"] == "timeout" else "build",
            "physical_sdk_build_timeout" if build["status"] == "timeout" else "physical_sdk_build_failed",
        )
        _record_hardware_preflight(payload, preflight)
        _record_staged_evidence(payload, staged)
        payload.update(
            timing={"build_s": build_time},
            binary_hashes=_binary_hashes(plan["binary_dir"]),
            commands=plan["commands"],
            build=_command_summary(build, plan["commands"]["build"]),
        )
        return _emit(payload, json_output)
    plan["inputs_dir"].mkdir(parents=True, exist_ok=True)
    plan["outputs_dir"].mkdir(parents=True, exist_ok=True)
    for path in (
        plan["inputs_dir"] / "a_u32.bin",
        plan["inputs_dir"] / "b_u32.bin",
        plan["outputs_dir"] / "result_u32.bin",
    ):
        path.unlink(missing_ok=True)
    started = time.perf_counter()
    run = _run_command(plan["commands"]["run"], plan["staged_benchmark"], env, args.timeout_seconds)
    payload = _execution_payload(
        plan,
        build,
        run,
        _parse_host_result(run.get("stdout", ""), run.get("stderr", "")),
        _validate_artifacts(
            plan["inputs_dir"] / "a_u32.bin",
            plan["inputs_dir"] / "b_u32.bin",
            plan["outputs_dir"] / "result_u32.bin",
        ),
        preflight,
        staged,
        build_time,
        time.perf_counter() - started,
    )
    return _emit(payload, json_output)
def _stage_sources(plan: Mapping[str, Any]) -> dict[str, Any]:
    external = plan["external_root"]
    upstream_target = external / PATCH_TARGET
    planned = plan["source_fingerprint"]
    current = (_hash_tree(plan["source_root"]), _hash_tree(external / "lib"), _hash_file(upstream_target), _hash_file(plan["patch_path"]))
    expected = (planned["owned_qualification_sha256"], planned["upstream_library_sha256"], planned["upstream_map_processing_sha256"], planned["patch_sha256"])
    if current != expected:
        raise ValueError("qualification source changed since qualification plan")
    before_upstream = (current[2], current[1])
    _reset_build_root(plan)
    shutil.copytree(plan["source_root"], plan["staged_benchmark"])
    shutil.copytree(external / "lib", plan["staged_simplepim"] / "lib")
    staged_target = plan["staged_simplepim"] / PATCH_TARGET
    if _hash_file(plan["staged_patch_path"]) != _hash_file(plan["patch_path"]):
        raise ValueError("staged patch differs from tracked patch")
    text = staged_target.read_text(encoding="utf-8")
    if text.count(BUGGY_UNROLL_LINE) != 2 or FIXED_UNROLL_LINE in text:
        raise ValueError("staged source does not match the pinned source")
    source_before = _hash_tree(plan["staged_simplepim"])
    result = subprocess.run(
        plan["commands"]["patch"],
        cwd=plan["staged_simplepim"],
        env={**os.environ, **plan["command_environment"]["patch"]},
        capture_output=True,
        text=True,
        check=False,
        timeout=PATCH_TIMEOUT_SECONDS,
    )
    if result.returncode != 0:
        raise ValueError((result.stderr or result.stdout).strip() or "tracked patch application failed")
    text = staged_target.read_text(encoding="utf-8")
    if text.count(FIXED_UNROLL_LINE) != 2 or BUGGY_UNROLL_LINE in text:
        raise ValueError("tracked patch did not apply exactly twice")
    source_after = _hash_tree(plan["staged_simplepim"])
    if before_upstream != (_hash_file(upstream_target), _hash_tree(external / "lib")):
        raise ValueError("pinned SimplePIM source changed during staging")
    return {
        "path": PATCH_RELATIVE_PATH,
        "sha256": _hash_file(plan["patch_path"]),
        "staged_sha256": _hash_file(plan["staged_patch_path"]),
        "applied": True,
        "replacement_count": 2,
        "command_fingerprint": _hash_json(
            {
                "command": plan["commands"]["patch"],
                "environment": plan["command_environment"]["patch"],
            }
        ),
        "environment": dict(plan["command_environment"]["patch"]),
        "staged_source_before_sha256": source_before,
        "staged_source_after_sha256": source_after,
        "staged_target_sha256": _hash_file(staged_target),
        "source_hashes": {
            **_source_hashes(plan),
            "upstream_map_processing_sha256": _hash_file(staged_target),
            "staged_source_before_patch_sha256": source_before,
            "staged_source_after_patch_sha256": source_after,
            "staged_patch_sha256": _hash_file(plan["staged_patch_path"]),
            "combined_sha256": source_after,
        },
    }
def _reset_build_root(plan: Mapping[str, Any]) -> None:
    build = Path(plan["build_root"])
    if build != Path(plan["workdir"]) / "build":
        raise ValueError("refusing to reset an unexpected build root")
    if build.is_symlink() or (build.exists() and not build.is_dir()):
        build.unlink()
    elif build.exists():
        shutil.rmtree(build)
    build.mkdir(parents=True)
def _execution_payload(
    plan: Mapping[str, Any],
    build: Mapping[str, Any],
    run: Mapping[str, Any],
    host: Mapping[str, Any] | None,
    validation: Mapping[str, Any],
    preflight: Mapping[str, Any],
    staged: Mapping[str, Any],
    build_time: float,
    host_time: float,
) -> dict[str, Any]:
    payload = _base_payload(plan, "failed")
    _record_hardware_preflight(payload, preflight)
    _record_staged_evidence(payload, staged)
    payload.update(
        binary_hashes=_binary_hashes(plan["binary_dir"]),
        input_hashes={
            "a_u32": _optional_hash(plan["inputs_dir"] / "a_u32.bin"),
            "b_u32": _optional_hash(plan["inputs_dir"] / "b_u32.bin"),
        },
        output_hash=_optional_hash(plan["outputs_dir"] / "result_u32.bin"),
        timing={"build_s": build_time, "host_wall_s": host_time},
        validation_performed=validation["performed"],
        exact_validation=validation["passed"],
        commands=plan["commands"],
        build=_command_summary(build, plan["commands"]["build"]),
        execution=_command_summary(run, plan["commands"]["run"]),
    )
    if run.get("status") == "timeout":
        payload.update(
            failure_stage="host_timeout",
            reason="host_timeout_cleanup_unverified",
            release_status="unknown",
            timeout_cleanup=run.get("timeout_cleanup"),
        )
        return payload
    if host is None:
        payload.update(
            failure_stage="host_process",
            reason="native_process_failed_without_host_result_release_unconfirmed",
            release_status="unknown",
        )
        return payload
    payload["host_result"] = dict(host)
    payload["observed_dpu_count"] = host.get("observed_dpu_count")
    payload["release_status"] = host.get("release_status", "unknown")
    payload["timing"].update(host.get("timing", {}) if isinstance(host.get("timing"), Mapping) else {})
    host_ok = _host_contract_failure(host) is None and run.get("status") == "passed" and host.get("status") == "passed"
    payload["native_execution"] = host_ok
    if host_ok and preflight["verified"]:
        payload["target"] = payload["target_observed"] = "physical_hardware"
    if host_ok and validation["passed"]:
        payload.update(status="passed", failure_stage=None, reason=None)
    elif not validation["passed"] and validation["performed"]:
        payload.update(failure_stage=validation["failure_stage"], reason=validation["reason"])
    elif not host_ok:
        payload.update(
            failure_stage="host_result_contract",
            reason=_host_contract_failure(host) or "native process failed",
        )
    else:
        payload.update(
            failure_stage="qualification_contract",
            reason="qualification contract not proven",
        )
    return payload
def _host_contract_failure(host: Mapping[str, Any]) -> str | None:
    expected = {
        "schema_version": HOST_SCHEMA_VERSION,
        "provider_id": PROVIDER_ID,
        "probe_id": PROBE_ID,
        "backend_profile": BACKEND_PROFILE,
        "requested_dpu_count": REQUESTED_DPUS,
        "configured_tasklets_per_dpu": CONFIGURED_TASKLETS,
        "observed_tasklets_per_dpu": None,
        "native_run_completed": True,
        "validation_performed": True,
        "host_exact_validation": True,
        "fallback": False,
        "release_status": "released",
        "logical_input_bytes": LOGICAL_INPUT_BYTES,
        "logical_output_bytes": LOGICAL_OUTPUT_BYTES,
        "physical_transfer_bytes_available": False,
        "physical_transfer_bytes": None,
        "failure_stage": None,
        "reason": None,
    }
    if not isinstance(host, Mapping) or any(host.get(key) != value for key, value in expected.items()):
        return "host result contract mismatch"
    if host.get("observed_dpu_count") != REQUESTED_DPUS or not isinstance(host.get("timing"), Mapping):
        return "host observation contract mismatch"
    return None
def _validate_artifacts(a_path: Path, b_path: Path, output_path: Path) -> dict[str, Any]:
    try:
        blobs = {
            "a": a_path.read_bytes(),
            "b": b_path.read_bytes(),
            "output": output_path.read_bytes(),
        }
    except OSError as exc:
        return {
            "performed": False,
            "passed": False,
            "failure_stage": "artifact_read",
            "reason": f"artifact_read_failed:{exc}",
        }
    size = ELEMENTS * UINT32_BYTES
    if any(len(blob) != size or len(blob) % 8 for blob in blobs.values()):
        return {
            "performed": False,
            "passed": False,
            "failure_stage": "artifact_size",
            "reason": "artifact_uint32_element_count_mismatch",
        }
    values = {key: struct.unpack(f"<{ELEMENTS}I", blob) for key, blob in blobs.items()}
    if values["a"] != _deterministic_values(0) or values["b"] != _deterministic_values(1):
        return {
            "performed": True,
            "passed": False,
            "failure_stage": "input_validation",
            "reason": "deterministic_input_validation_failed",
        }
    expected = tuple((a + b) & 0xFFFFFFFF for a, b in zip(values["a"], values["b"], strict=True))
    return {
        "performed": True,
        "passed": values["output"] == expected,
        "failure_stage": None if values["output"] == expected else "exact_validation",
        "reason": "independent_exact_uint32_validation_passed" if values["output"] == expected else "independent_exact_uint32_validation_failed",
    }
def _deterministic_values(salt: int) -> tuple[int, ...]:
    return tuple(17 + ((index * 13 + salt * 5) % 1000) for index in range(ELEMENTS))
def _pack_uint32(values: Sequence[int]) -> bytes:
    return struct.pack(f"<{len(values)}I", *values)
def _parse_host_result(stdout: str, stderr: str) -> dict[str, Any] | None:
    for line in reversed((stdout + "\n" + stderr).splitlines()):
        try:
            value = json.loads(line.strip())
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and value.get("schema_version") == HOST_SCHEMA_VERSION:
            return value
    return None
def _hardware_device_preflight(device_root: Path = Path("/dev"), sysfs_root: Path = Path("/sys/class/dpu_rank")) -> dict[str, Any]:
    evidence = []
    for path in sorted(device_root.glob("dpu_rank*")):
        item = {
            "path": str(path),
            "exists": False,
            "character_device": False,
            "readable": False,
            "writable": False,
            "sysfs_path": str(sysfs_root / path.name),
            "sysfs_exists": (sysfs_root / path.name).exists(),
        }
        try:
            mode = path.stat().st_mode
        except OSError as exc:
            item["error"] = str(exc)
        else:
            item.update(
                exists=True,
                character_device=stat.S_ISCHR(mode),
                readable=os.access(path, os.R_OK),
                writable=os.access(path, os.W_OK),
            )
        evidence.append(item)
    verified = any(item["exists"] and item["character_device"] and item["readable"] and item["writable"] for item in evidence)
    return {
        "verified": verified,
        "device_nodes": evidence,
        "required_pattern": str(device_root / "dpu_rank*"),
        "reason": "hardware_device_node_verified" if verified else "no_accessible_physical_dpu_rank_device_node",
    }
def _run_command(command: Sequence[str], cwd: Path, env: Mapping[str, str], timeout_seconds: float) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        process = subprocess.Popen(
            list(command),
            cwd=cwd,
            env=dict(env),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
    except OSError as exc:
        return {
            "status": "failed",
            "returncode": None,
            "stdout": "",
            "stderr": str(exc),
            "wall_s": time.perf_counter() - started,
        }
    try:
        stdout, stderr = process.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired as initial_timeout:
        errors: list[str] = []
        output_complete = True
        try:
            os.killpg(process.pid, signal.SIGTERM)
            term = True
        except OSError as exc:
            errors.append(str(exc))
            term = False
        try:
            stdout, stderr = process.communicate(timeout=2)
            killed = False
        except subprocess.TimeoutExpired as term_timeout:
            try:
                os.killpg(process.pid, signal.SIGKILL)
                killed = True
            except OSError as exc:
                errors.append(str(exc))
                killed = False
            try:
                stdout, stderr = process.communicate(timeout=2)
            except subprocess.TimeoutExpired as kill_timeout:
                output_complete = False
                stdout = _timeout_text(kill_timeout.output or term_timeout.output or initial_timeout.output)
                stderr = _timeout_text(kill_timeout.stderr or term_timeout.stderr or initial_timeout.stderr)
                for stream in (getattr(process, "stdout", None), getattr(process, "stderr", None)):
                    if stream is not None:
                        stream.close()
        return {
            "status": "timeout",
            "returncode": process.returncode,
            "stdout": stdout,
            "stderr": stderr,
            "wall_s": time.perf_counter() - started,
            "timeout_cleanup": {
                "attempted": True,
                "verified": False,
                "verification": "unavailable",
                "output_capture_complete": output_complete,
                "sigterm_sent": term,
                "sigkill_sent": killed,
                "process_exited": process.returncode is not None,
                "signal_errors": errors,
            },
        }
    return {
        "status": "passed" if process.returncode == 0 else "failed",
        "returncode": process.returncode,
        "stdout": stdout,
        "stderr": stderr,
        "wall_s": time.perf_counter() - started,
    }
def _base_payload(plan: Mapping[str, Any], status: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "provider_id": PROVIDER_ID,
        "probe_id": PROBE_ID,
        "status": status,
        "target": None,
        "target_observed": None,
        "requested_dpu_count": REQUESTED_DPUS,
        "observed_dpu_count": None,
        "configured_tasklets_per_dpu": CONFIGURED_TASKLETS,
        "observed_tasklets_per_dpu": None,
        "hardware_preflight_verified": False,
        "device_evidence": [],
        "native_execution": False,
        "validation_performed": False,
        "exact_validation": False,
        "fallback": False,
        "simulator_kernel_executed": False,
        "release_status": "not_attempted",
        "backend_profile": BACKEND_PROFILE,
        "source_hash": plan["source_fingerprint"]["combined_sha256"],
        "source_hashes": dict(plan["source_fingerprint"]),
        "command_fingerprint": plan["command_fingerprint"],
        "command_environment": {
            key: dict(value) for key, value in plan.get("command_environment", {}).items()
        },
        "effective_compilers": plan["effective_compilers"],
        "staged_patch": dict(plan["staged_patch"]),
        "binary_hashes": {},
        "input_hashes": {},
        "output_hash": None,
        "logical_transfer_bytes": {
            "h2d": LOGICAL_INPUT_BYTES,
            "d2h": LOGICAL_OUTPUT_BYTES,
            "total": LOGICAL_TOTAL_BYTES,
            "scope": "logical_application_payload_only",
        },
        "payload_sizes_8_byte_aligned": True,
        "physical_transfer_bytes_available": False,
        "physical_transfer_bytes": None,
        "timing": {},
        "failure_stage": None,
        "reason": "not_run",
    }
def _failure_payload(plan: Mapping[str, Any], stage: str, reason: str) -> dict[str, Any]:
    payload = _base_payload(plan, "failed")
    payload.update(failure_stage=stage, reason=reason)
    return payload
def _record_hardware_preflight(payload: dict[str, Any], preflight: Mapping[str, Any]) -> None:
    payload.update(
        hardware_preflight_verified=bool(preflight["verified"]),
        device_evidence=preflight["device_nodes"],
    )
def _record_staged_evidence(payload: dict[str, Any], staged: Mapping[str, Any]) -> None:
    payload["staged_patch"] = {key: value for key, value in staged.items() if key != "source_hashes"}
    payload["source_hashes"] = dict(staged["source_hashes"])
    payload["source_hash"] = payload["source_hashes"]["combined_sha256"]
def _command_summary(result: Mapping[str, Any], command: Sequence[str]) -> dict[str, Any]:
    return {
        "command": list(command),
        "command_fingerprint": _hash_json(command),
        "status": result.get("status"),
        "returncode": result.get("returncode"),
        "wall_s": result.get("wall_s"),
        "stdout_tail": str(result.get("stdout") or "")[-1000:],
        "stderr_tail": str(result.get("stderr") or "")[-1000:],
        **({"timeout_cleanup": result["timeout_cleanup"]} if result.get("timeout_cleanup") else {}),
    }
def _emit(payload: Mapping[str, Any], json_output: Path | None) -> int:
    _validate_output_schema(payload)
    if json_output:
        json_output.parent.mkdir(parents=True, exist_ok=True)
        json_output.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))
    return 0 if payload.get("status") in {"prepared", "passed"} else 1
def _cli_selector_rejection(backend: str, target: str) -> str | None:
    return f"simulator selector rejected: {backend}/{target}" if _is_simulator_value(backend) or _is_simulator_value(target) else None
def _external_backend_selector(env: Mapping[str, str]) -> str | None:
    return next(
        (f"{key}={env[key]}" for key in BACKEND_SELECTOR_KEYS if env.get(key) and _is_simulator_value(env[key])),
        None,
    )
def _is_simulator_value(value: str) -> bool:
    return "simulator" in value.lower() or value.strip().lower() in SIMULATOR_ALIASES or any(token in SIMULATOR_ALIASES for token in value.lower().replace("=", "_").split("_"))
def _physical_environment(env: Mapping[str, str], host_cc: str | None = None) -> dict[str, str]:
    result = dict(env)
    result.update(DPU_BACKEND="hw", UPMEM_ALLOW_PHYSICAL_HARDWARE="1")
    if host_cc:
        result["HOST_CC"] = host_cc
    return result
def _minimal_plan(workdir: Path) -> dict[str, Any]:
    return {
        "workdir": workdir,
        "commands": {},
        "command_fingerprint": None,
        "effective_compilers": {
            "host_cc": _compiler_identity(HOST_CC),
            "dpu_cc": _compiler_identity(DPU_CC),
        },
        "source_fingerprint": {"combined_sha256": "0" * 64},
        "staged_patch": {"path": PATCH_RELATIVE_PATH, "applied": False},
    }
def _require_file(path: Path) -> None:
    if not path.is_file():
        raise ValueError(f"required source file missing: {path}")
def _resolve_host_cc(env: Mapping[str, str]) -> str:
    raw = shutil.which(env.get("HOST_CC", HOST_CC), path=env.get("PATH"))
    if not raw:
        raise ValueError("HOST_CC compiler is unavailable")
    return str(Path(raw).resolve())
def _compiler_identity(command: str, resolved_path: Path | None = None) -> dict[str, Any]:
    raw = str(resolved_path) if resolved_path else shutil.which(command)
    path = Path(raw).resolve() if raw else None
    return {
        "command": command,
        "available": bool(path and path.is_file()),
        "path": str(path) if path and path.is_file() else None,
        "sha256": _hash_file(path) if path and path.is_file() else None,
    }
def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
def _hash_tree(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(path.rglob("*")):
        if item.is_file():
            digest.update(item.relative_to(path).as_posix().encode() + b"\0" + _hash_file(item).encode() + b"\n")
    return digest.hexdigest()
def _combine_hashes(values: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(values).encode()).hexdigest()
def _hash_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
def _optional_hash(path: Path) -> str | None:
    return _hash_file(path) if path.is_file() else None
def _binary_hashes(path: Path) -> dict[str, str]:
    return {item.name: _hash_file(item) for item in sorted(path.iterdir()) if item.is_file()} if path.is_dir() else {}
def _source_hashes(plan: Mapping[str, Any]) -> dict[str, Any]:
    return dict(plan["source_fingerprint"])
def _validate_output_schema(payload: Mapping[str, Any]) -> None:
    required = {
        "schema_version",
        "provider_id",
        "probe_id",
        "status",
        "source_hash",
        "source_hashes",
        "staged_patch",
        "fallback",
        "simulator_kernel_executed",
        "release_status",
    }
    missing = required - payload.keys()
    if missing:
        raise ValueError(f"missing required fields: {sorted(missing)}")
    if payload["fallback"] or payload["simulator_kernel_executed"]:
        raise ValueError("fallback/simulator execution is forbidden")
    if payload["source_hash"] != payload["source_hashes"].get("combined_sha256"):
        raise ValueError("source hash chain is inconsistent")
    if payload["observed_tasklets_per_dpu"] is not None:
        raise ValueError("configured tasklets are not independently observed")
    if payload["status"] == "prepared":
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
            and isinstance(payload.get("staged_patch"), Mapping) and payload["staged_patch"].get("applied") is False
            and payload["staged_patch"].get("staged_sha256") is None
            and all(payload["source_hashes"].get(key) is None for key in ("staged_source_before_patch_sha256", "staged_source_after_patch_sha256", "staged_patch_sha256"))
            and all(key not in payload for key in ("build", "execution", "host_result", "timeout_cleanup"))
        )
        if not inert:
            raise ValueError("prepared result must not claim staging, build, or hardware execution")
        return
    if payload["status"] != "passed":
        return
    build, execution, host = payload.get("build"), payload.get("execution"), payload.get("host_result")
    binaries, inputs = payload.get("binary_hashes"), payload.get("input_hashes")
    hashes, patch = payload.get("source_hashes"), payload.get("staged_patch")
    compilers, devices = payload.get("effective_compilers"), payload.get("device_evidence")
    valid = (
        payload.get("target") == PHYSICAL_TARGET
        and payload.get("hardware_preflight_verified") is True
        and _device_evidence_proves_hardware(devices)
        and isinstance(build, Mapping) and build.get("status") == "passed" and build.get("returncode") == 0
        and isinstance(binaries, Mapping) and all(_is_sha256(binaries.get(name)) for name in ("host", "dpu_init_binary", "dpu_zip", "dpu_map_va_funcs"))
        and isinstance(execution, Mapping) and execution.get("status") == "passed" and execution.get("returncode") == 0
        and _host_contract_failure(host) is None
        and payload.get("native_execution") is True and payload.get("validation_performed") is True and payload.get("exact_validation") is True
        and payload.get("observed_dpu_count") == REQUESTED_DPUS and payload.get("configured_tasklets_per_dpu") == CONFIGURED_TASKLETS
        and isinstance(inputs, Mapping) and _is_sha256(inputs.get("a_u32")) and _is_sha256(inputs.get("b_u32")) and _is_sha256(payload.get("output_hash"))
        and isinstance(hashes, Mapping) and _is_sha256(hashes.get("staged_source_after_patch_sha256")) and payload.get("source_hash") == hashes.get("combined_sha256") == hashes.get("staged_source_after_patch_sha256")
        and isinstance(patch, Mapping) and patch.get("applied") is True and _is_sha256(patch.get("sha256")) and patch.get("staged_sha256") == patch.get("sha256") == hashes.get("patch_sha256")
        and isinstance(compilers, Mapping) and all(isinstance(compilers.get(role), Mapping) and compilers[role].get("available") is True and _is_sha256(compilers[role].get("sha256")) for role in ("host_cc", "dpu_cc"))
        and payload.get("release_status") == "released" and payload.get("failure_stage") is None and payload.get("reason") is None
    )
    if not valid:
        raise ValueError("passed result lacks required physical qualification evidence")
def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)
def _device_evidence_proves_hardware(value: Any) -> bool:
    return isinstance(value, list) and any(isinstance(item, Mapping) and isinstance(item.get("path"), str) and bool(item["path"]) and all(item.get(key) is True for key in ("exists", "character_device", "readable", "writable")) for item in value)
def _timeout_text(value: str | bytes | None) -> str:
    return value.decode(errors="replace") if isinstance(value, bytes) else value or ""
if __name__ == "__main__":
    raise SystemExit(main())
