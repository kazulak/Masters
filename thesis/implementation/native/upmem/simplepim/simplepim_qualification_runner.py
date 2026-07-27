"""Prepare or execute the isolated physical SimplePIM qualification probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shlex
import shutil
import signal
import stat
import struct
import subprocess
import time
from pathlib import Path
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
PROCESS_GROUP_GRACE_SECONDS = 2.0
PROCESS_GROUP_KILL_WAIT_SECONDS = 2.0
PATCH_RELATIVE_PATH = "patches/simplepim-map-unroll-rest.patch"
PATCH_TARGET = "lib/processing/map/MapProcessing.h"
BUGGY_UNROLL_LINE = "uint32_t unroll_block_rest = copy_block_size-unroll_block_rest;"
FIXED_UNROLL_LINE = "uint32_t unroll_block_rest = copy_block_size-unroll_block_size;"
BACKEND_PROFILE = "backend=hw"
HOST_CC = "gcc"
DPU_CC = "dpu-upmem-dpurte-clang"
BACKEND_SELECTOR_KEYS = (
    "DPU_BACKEND",
    "DPU_PROFILE",
    "SIMPLEPIM_BACKEND",
    "UPMEM_BACKEND",
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
    "cycle_accurate_simulator",
    "backend=sim",
    "backend=simulator",
}

if LOGICAL_INPUT_BYTES % 8 != 0 or LOGICAL_OUTPUT_BYTES % 8 != 0:
    raise RuntimeError("SimplePIM logical payload sizes must be 8-byte aligned")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="SimplePIM physical qualification runner"
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--workdir", required=True, type=Path)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument(
        "--timeout-seconds", type=float, default=DEFAULT_TIMEOUT_SECONDS
    )
    parser.add_argument("--backend", default="physical")
    parser.add_argument("--target", default="physical_hardware")
    args = parser.parse_args(argv)

    try:
        plan = qualification_plan(args.workdir)
        selector_reason = _cli_selector_rejection(args.backend, args.target)
        if selector_reason is not None:
            return _emit(
                _failure_payload(plan, "backend_selection", selector_reason),
                args.json_output,
            )
        if args.prepare_only:
            payload = _base_payload(plan, "prepared")
            payload.update(
                {
                    "commands": plan["commands"],
                    "reason": "prepare_only_no_compiler_or_hardware_invoked",
                }
            )
            return _emit(payload, args.json_output)
        return _execute(plan, args, args.json_output)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        payload = _failure_payload(
            _minimal_plan(args.workdir),
            "preflight",
            f"{type(exc).__name__}: {exc}",
        )
        return _emit(payload, args.json_output)


def qualification_plan(workdir: Path) -> dict[str, Any]:
    """Verify and fingerprint source and commands without staging or executing."""
    workdir = workdir.expanduser().resolve()
    source_root = Path(__file__).resolve().parent / "qualification"
    external_root = Path(__file__).resolve().parents[3] / "external" / "SimplePIM"
    required_owned = (
        "ATTRIBUTION.md",
        "Makefile",
        "Param.h",
        "host.c",
        "small_table_init.c",
        "va_funcs/map.h",
        PATCH_RELATIVE_PATH,
    )
    required_upstream = (
        "lib/communication/CommHelper.c",
        "lib/communication/CommHelper.h",
        "lib/communication/CommOps.c",
        "lib/communication/CommOps.h",
        "lib/management/Management.c",
        "lib/management/Management.h",
        "lib/management/SmallTableInit_dpu.c",
        "lib/processing/ProcessingHelper.c",
        "lib/processing/ProcessingHelper.h",
        "lib/processing/ProcessingHelperHost.h",
        "lib/processing/map/Map.c",
        "lib/processing/map/Map.h",
        "lib/processing/map/MapArgs.h",
        "lib/processing/map/MapProcessing.h",
        "lib/processing/map/map_dpu.c",
        "lib/processing/zip/Zip.c",
        "lib/processing/zip/Zip.h",
        "lib/processing/zip/ZipArgs.h",
        "lib/processing/zip/ZipProcessing.c",
        "lib/processing/zip/zip_dpu.c",
    )
    for relative in required_owned:
        _require_file(source_root / relative)
    for relative in required_upstream:
        _require_file(external_root / relative)

    host_text = (source_root / "host.c").read_text(encoding="utf-8")
    map_text = (source_root / "va_funcs/map.h").read_text(encoding="utf-8")
    upstream_map_path = external_root / PATCH_TARGET
    upstream_map_text = upstream_map_path.read_text(encoding="utf-8")
    patch_path = source_root / PATCH_RELATIVE_PATH
    patch_text = patch_path.read_text(encoding="utf-8")
    if (
        "table_zip" not in host_text
        or "table_map" not in host_text
        or "map_func" not in map_text
    ):
        raise ValueError(
            "qualification source does not contain the VA map/zip contract"
        )
    if upstream_map_text.count(BUGGY_UNROLL_LINE) != 2:
        raise ValueError(
            "pinned MapProcessing.h no longer has the expected two defects"
        )
    if patch_text.count(BUGGY_UNROLL_LINE) != 2:
        raise ValueError("tracked patch does not remove both pinned defective lines")
    if patch_text.count(FIXED_UNROLL_LINE) != 2:
        raise ValueError("tracked patch does not fix both map paths")
    hunk_headers = [line for line in patch_text.splitlines() if line.startswith("@@ ")]
    if (
        len(hunk_headers) != 2
        or any(_patch_hunk_context_size(header) <= 1 for header in hunk_headers)
        or not _patch_hunks_have_nonblank_context(patch_text)
    ):
        raise ValueError("tracked MapProcessing.h patch must contain contextual hunks")

    build_root = workdir / "build"
    staged_simplepim = build_root / "staged" / "SimplePIM"
    staged_benchmark = staged_simplepim / "benchmarks" / "va"
    staged_patch_path = staged_benchmark / PATCH_RELATIVE_PATH
    binary_dir = staged_benchmark / "bin"
    inputs_dir = build_root / "inputs"
    outputs_dir = build_root / "outputs"
    host_cc = _resolve_host_cc(os.environ)
    commands = {
        "patch": [
            "git",
            "apply",
            "--no-index",
            "--verbose",
            str(staged_patch_path),
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
            str(binary_dir / "host"),
            str(inputs_dir / "a_u32.bin"),
            str(inputs_dir / "b_u32.bin"),
            str(outputs_dir / "result_u32.bin"),
        ],
    }
    owned_hash = _hash_tree(source_root)
    upstream_hash = _hash_tree(external_root / "lib")
    patch_hash = _hash_file(patch_path)
    effective_compilers = {
        "host_cc": _compiler_identity(HOST_CC, Path(host_cc)),
        "dpu_cc": _compiler_identity(DPU_CC),
    }
    return {
        "workdir": workdir,
        "build_root": build_root,
        "staged_simplepim": staged_simplepim,
        "staged_benchmark": staged_benchmark,
        "binary_dir": binary_dir,
        "inputs_dir": inputs_dir,
        "outputs_dir": outputs_dir,
        "source_root": source_root,
        "external_root": external_root,
        "patch_path": patch_path,
        "staged_patch_path": staged_patch_path,
        "host_cc": host_cc,
        "commands": commands,
        "command_fingerprint": _hash_json(commands),
        "effective_compilers": effective_compilers,
        "source_fingerprint": {
            "owned_qualification_sha256": owned_hash,
            "upstream_library_sha256": upstream_hash,
            "upstream_map_processing_sha256": _hash_file(upstream_map_path),
            "patch_sha256": patch_hash,
            "staged_source_before_patch_sha256": None,
            "staged_source_after_patch_sha256": None,
            "staged_patch_sha256": None,
            "combined_sha256": _combine_hashes((owned_hash, upstream_hash, patch_hash)),
            "upstream_submodule": str(external_root),
        },
        "staged_patch": {
            "path": PATCH_RELATIVE_PATH,
            "sha256": patch_hash,
            "staged_sha256": None,
            "applied": False,
            "replacement_count": 0,
            "command_fingerprint": _hash_json(commands["patch"]),
            "staged_source_before_sha256": None,
            "staged_source_after_sha256": None,
            "staged_target_sha256": None,
        },
    }


def _execute(
    plan: Mapping[str, Any],
    args: argparse.Namespace,
    json_output: Path | None,
) -> int:
    if os.environ.get("UPMEM_ALLOW_PHYSICAL_HARDWARE") != "1":
        return _emit(
            _failure_payload(
                plan,
                "opt_in",
                "UPMEM_ALLOW_PHYSICAL_HARDWARE=1_required",
            ),
            json_output,
        )
    selector = _external_backend_selector(os.environ)
    if selector is not None:
        return _emit(
            _failure_payload(
                plan,
                "backend_selection",
                f"external_backend_selector_rejected:{selector}",
            ),
            json_output,
        )

    hardware_preflight = _hardware_device_preflight()
    if not hardware_preflight["verified"]:
        payload = _failure_payload(
            plan,
            "hardware_preflight",
            str(hardware_preflight["reason"]),
        )
        _record_hardware_preflight(payload, hardware_preflight)
        return _emit(payload, json_output)

    try:
        staged_patch = _stage_sources(plan)
    except (OSError, ValueError) as exc:
        payload = _failure_payload(
            plan,
            "staged_patch",
            f"staged_patch_failed:{exc}",
        )
        _record_hardware_preflight(payload, hardware_preflight)
        return _emit(payload, json_output)

    physical_env = _physical_environment(os.environ, str(plan["host_cc"]))
    build_started = time.perf_counter()
    build = _run_command(
        plan["commands"]["build"],
        plan["staged_benchmark"],
        physical_env,
        args.timeout_seconds,
    )
    build_time = time.perf_counter() - build_started
    if build["status"] != "passed":
        payload = _failure_payload(
            plan,
            "build_timeout" if build["status"] == "timeout" else "build",
            "physical_sdk_build_timeout"
            if build["status"] == "timeout"
            else "physical_sdk_build_failed",
        )
        _record_hardware_preflight(payload, hardware_preflight)
        _record_staged_evidence(payload, staged_patch)
        payload["timing"] = {"build_s": build_time}
        payload["binary_hashes"] = _binary_hashes(plan["binary_dir"])
        payload["commands"] = {
            name: list(command) for name, command in plan["commands"].items()
        }
        payload["build"] = _command_summary(
            build,
            plan["commands"]["build"],
        )
        return _emit(payload, json_output)

    plan["inputs_dir"].mkdir(parents=True, exist_ok=True)
    plan["outputs_dir"].mkdir(parents=True, exist_ok=True)
    run_started = time.perf_counter()
    run = _run_command(
        plan["commands"]["run"],
        plan["staged_benchmark"],
        physical_env,
        args.timeout_seconds,
    )
    host_time = time.perf_counter() - run_started
    host_result = _parse_host_result(run.get("stdout", ""), run.get("stderr", ""))
    artifact_validation = _validate_artifacts(
        plan["inputs_dir"] / "a_u32.bin",
        plan["inputs_dir"] / "b_u32.bin",
        plan["outputs_dir"] / "result_u32.bin",
    )
    payload = _execution_payload(
        plan=plan,
        build=build,
        run=run,
        host_result=host_result,
        artifact_validation=artifact_validation,
        hardware_preflight=hardware_preflight,
        staged_patch=staged_patch,
        build_time=build_time,
        host_time=host_time,
    )
    return _emit(payload, json_output)


def _stage_sources(plan: Mapping[str, Any]) -> dict[str, Any]:
    upstream_target = plan["external_root"] / PATCH_TARGET
    upstream_hash_before = _hash_file(upstream_target)
    upstream_library_hash_before = _hash_tree(plan["external_root"] / "lib")
    _reset_build_root(plan)
    shutil.copytree(
        plan["source_root"],
        plan["staged_benchmark"],
    )
    shutil.copytree(
        plan["external_root"] / "lib",
        plan["staged_simplepim"] / "lib",
    )

    staged_target = plan["staged_simplepim"] / PATCH_TARGET
    staged_patch_path = plan["staged_patch_path"]
    tracked_patch_hash = _hash_file(plan["patch_path"])
    staged_patch_hash = _hash_file(staged_patch_path)
    if staged_patch_hash != tracked_patch_hash:
        raise ValueError("staged patch artifact differs from the tracked patch")

    unpatched_text = staged_target.read_text(encoding="utf-8")
    if (
        unpatched_text.count(BUGGY_UNROLL_LINE) != 2
        or FIXED_UNROLL_LINE in unpatched_text
    ):
        raise ValueError("staged MapProcessing.h does not match the pinned source")
    staged_source_before = _hash_tree(plan["staged_simplepim"])

    try:
        patch_env = dict(os.environ)
        patch_env["GIT_CEILING_DIRECTORIES"] = str(plan["build_root"])
        patch_result = subprocess.run(
            list(plan["commands"]["patch"]),
            cwd=plan["staged_simplepim"],
            env=patch_env,
            capture_output=True,
            text=True,
            check=False,
            timeout=PATCH_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        raise ValueError("tracked patch application timed out") from exc
    if patch_result.returncode != 0:
        detail = (patch_result.stderr or patch_result.stdout).strip()
        raise ValueError(f"tracked patch application failed: {detail}")

    patched_text = staged_target.read_text(encoding="utf-8")
    if patched_text.count(FIXED_UNROLL_LINE) != 2 or BUGGY_UNROLL_LINE in patched_text:
        raise ValueError("staged MapProcessing.h patch did not apply exactly twice")
    staged_source_after = _hash_tree(plan["staged_simplepim"])
    staged_owned_hash = _hash_tree(plan["staged_benchmark"])
    staged_library_hash = _hash_tree(plan["staged_simplepim"] / "lib")
    staged_target_hash = _hash_file(staged_target)
    if (
        _hash_file(upstream_target) != upstream_hash_before
        or _hash_tree(plan["external_root"] / "lib") != upstream_library_hash_before
    ):
        raise ValueError("pinned SimplePIM submodule changed during staging")
    return {
        "path": PATCH_RELATIVE_PATH,
        "sha256": tracked_patch_hash,
        "applied": True,
        "replacement_count": 2,
        "staged_target_sha256": staged_target_hash,
        "staged_patch_sha256": staged_patch_hash,
        "command_fingerprint": _hash_json(plan["commands"]["patch"]),
        "staged_source_before_sha256": staged_source_before,
        "staged_source_after_sha256": staged_source_after,
        "source_hashes": {
            "owned_qualification_sha256": staged_owned_hash,
            "upstream_library_sha256": staged_library_hash,
            "upstream_map_processing_sha256": staged_target_hash,
            "patch_sha256": staged_patch_hash,
            "staged_source_before_patch_sha256": staged_source_before,
            "staged_source_after_patch_sha256": staged_source_after,
            "staged_patch_sha256": staged_patch_hash,
            "combined_sha256": staged_source_after,
            "upstream_submodule": str(plan["external_root"]),
        },
    }


def _reset_build_root(plan: Mapping[str, Any]) -> None:
    build_root = Path(plan["build_root"])
    expected = Path(plan["workdir"]) / "build"
    if build_root != expected or build_root.name != "build":
        raise ValueError("refusing to reset a non-workdir build root")
    if build_root.is_symlink() or (build_root.exists() and not build_root.is_dir()):
        build_root.unlink()
    elif build_root.exists():
        shutil.rmtree(build_root)
    build_root.mkdir(parents=True)


def _execution_payload(
    *,
    plan: Mapping[str, Any],
    build: Mapping[str, Any],
    run: Mapping[str, Any],
    host_result: Mapping[str, Any] | None,
    artifact_validation: Mapping[str, Any],
    hardware_preflight: Mapping[str, Any],
    staged_patch: Mapping[str, Any],
    build_time: float,
    host_time: float,
) -> dict[str, Any]:
    payload = _base_payload(plan, "failed")
    _record_hardware_preflight(payload, hardware_preflight)
    payload.update(
        {
            "binary_hashes": _binary_hashes(plan["binary_dir"]),
            "input_hashes": {
                "a_u32": _optional_hash(plan["inputs_dir"] / "a_u32.bin"),
                "b_u32": _optional_hash(plan["inputs_dir"] / "b_u32.bin"),
            },
            "output_hash": _optional_hash(plan["outputs_dir"] / "result_u32.bin"),
            "timing": {"build_s": build_time, "host_wall_s": host_time},
            "validation_performed": bool(artifact_validation["performed"]),
            "exact_validation": bool(artifact_validation["passed"]),
            "commands": {
                name: list(command) for name, command in plan["commands"].items()
            },
            "build": _command_summary(build, plan["commands"]["build"]),
            "execution": _command_summary(run, plan["commands"]["run"]),
        }
    )
    _record_staged_evidence(payload, staged_patch)
    if run.get("status") == "timeout":
        payload.update(
            {
                "release_status": "unknown",
                "failure_stage": "host_timeout",
                "reason": "host_process_group_terminated_after_timeout",
                "timeout_cleanup": dict(run.get("timeout_cleanup") or {}),
            }
        )
        return payload
    if host_result is None:
        payload.update(
            {
                "release_status": "unknown",
                "failure_stage": (
                    "host_process"
                    if run.get("status") == "failed"
                    else "host_result_parse"
                ),
                "reason": (
                    "upstream_native_process_failed_without_host_result_"
                    "release_unconfirmed"
                    if run.get("status") == "failed"
                    else "host_json_result_missing_release_unconfirmed"
                ),
            }
        )
        return payload

    observed_dpus = host_result.get("observed_dpu_count")
    if isinstance(observed_dpus, int) and not isinstance(observed_dpus, bool):
        payload["observed_dpu_count"] = observed_dpus
    release_status = host_result.get("release_status")
    if release_status in {"released", "failed", "not_attempted", "unknown"}:
        payload["release_status"] = release_status
    host_timing = host_result.get("timing")
    if isinstance(host_timing, Mapping):
        payload["timing"].update(
            {
                key: value
                for key, value in host_timing.items()
                if isinstance(key, str)
                and isinstance(value, (int, float))
                and not isinstance(value, bool)
            }
        )
    payload["host_result"] = dict(host_result)

    host_contract_failure = _host_contract_failure(host_result)
    native_run_passed = (
        run.get("status") == "passed"
        and host_contract_failure is None
        and host_result.get("status") == "passed"
    )
    payload["native_execution"] = native_run_passed
    if hardware_preflight["verified"] and native_run_passed:
        payload["target"] = "physical_hardware"
        payload["target_observed"] = "physical_hardware"

    passed = native_run_passed and artifact_validation["passed"] is True
    if passed:
        payload.update(
            {
                "status": "passed",
                "failure_stage": None,
                "reason": None,
            }
        )
        return payload

    if run.get("status") != "passed":
        payload["release_status"] = "unknown"
        payload["failure_stage"] = "host_process"
        payload["reason"] = "native_process_failed_release_unconfirmed"
    elif artifact_validation["passed"] is False:
        payload["failure_stage"] = str(artifact_validation["failure_stage"])
        payload["reason"] = str(artifact_validation["reason"])
    elif host_contract_failure is not None:
        payload["failure_stage"] = "host_result_contract"
        payload["reason"] = host_contract_failure
    else:
        payload["failure_stage"] = "qualification_contract"
        payload["reason"] = "qualification_contract_not_proven"
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
        "status": "passed",
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
    if set(host) != set(expected).union({"observed_dpu_count", "timing"}):
        return "host_result_invalid:fields"
    for key, value in expected.items():
        if host.get(key) != value or (
            isinstance(value, bool) and not isinstance(host.get(key), bool)
        ):
            return f"host_result_invalid:{key}"
    observed = host.get("observed_dpu_count")
    if not isinstance(observed, int) or isinstance(observed, bool):
        return "host_result_invalid:observed_dpu_count_type"
    if observed != REQUESTED_DPUS:
        return "host_result_invalid:observed_dpu_count"
    timing = host.get("timing")
    if not isinstance(timing, Mapping) or any(
        not isinstance(key, str)
        or not isinstance(value, (int, float))
        or isinstance(value, bool)
        or value < 0
        for key, value in timing.items()
    ):
        return "host_result_invalid:timing"
    return None


def _validate_artifacts(
    a_path: Path, b_path: Path, output_path: Path
) -> dict[str, Any]:
    paths = {"a": a_path, "b": b_path, "output": output_path}
    try:
        blobs = {name: path.read_bytes() for name, path in paths.items()}
    except OSError as exc:
        return {
            "performed": False,
            "passed": False,
            "failure_stage": "artifact_read",
            "reason": f"artifact_read_failed:{exc}",
        }
    expected_size = ELEMENTS * UINT32_BYTES
    sizes = {name: len(blob) for name, blob in blobs.items()}
    if any(size % 8 != 0 for size in sizes.values()):
        return {
            "performed": False,
            "passed": False,
            "failure_stage": "artifact_size",
            "reason": "artifact_size_not_8_byte_aligned",
        }
    if any(size != expected_size for size in sizes.values()):
        return {
            "performed": False,
            "passed": False,
            "failure_stage": "artifact_size",
            "reason": "artifact_uint32_element_count_mismatch",
        }
    unpack_format = f"<{ELEMENTS}I"
    a_values = struct.unpack(unpack_format, blobs["a"])
    b_values = struct.unpack(unpack_format, blobs["b"])
    output_values = struct.unpack(unpack_format, blobs["output"])
    if a_values != _deterministic_values(0) or b_values != _deterministic_values(1):
        return {
            "performed": True,
            "passed": False,
            "failure_stage": "input_validation",
            "reason": "deterministic_input_validation_failed",
        }
    expected_output = tuple(
        (left + right) & 0xFFFFFFFF
        for left, right in zip(a_values, b_values, strict=True)
    )
    if output_values != expected_output:
        return {
            "performed": True,
            "passed": False,
            "failure_stage": "exact_validation",
            "reason": "independent_exact_uint32_validation_failed",
        }
    return {
        "performed": True,
        "passed": True,
        "failure_stage": None,
        "reason": "independent_exact_uint32_validation_passed",
    }


def _deterministic_values(salt: int) -> tuple[int, ...]:
    return tuple(17 + ((index * 13 + salt * 5) % 1000) for index in range(ELEMENTS))


def _pack_uint32(values: Sequence[int]) -> bytes:
    return struct.pack(f"<{len(values)}I", *values)


def _parse_host_result(stdout: str, stderr: str) -> dict[str, Any] | None:
    for line in reversed((stdout + "\n" + stderr).splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (
            isinstance(value, dict)
            and value.get("schema_version") == HOST_SCHEMA_VERSION
        ):
            return value
    return None


def _hardware_device_preflight(
    device_root: Path = Path("/dev"),
    sysfs_root: Path = Path("/sys/class/dpu_rank"),
) -> dict[str, Any]:
    evidence: list[dict[str, Any]] = []
    for path in sorted(device_root.glob("dpu_rank*")):
        item: dict[str, Any] = {
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
                {
                    "exists": True,
                    "character_device": stat.S_ISCHR(mode),
                    "readable": os.access(path, os.R_OK),
                    "writable": os.access(path, os.W_OK),
                }
            )
        evidence.append(item)
    verified = any(
        item["exists"]
        and item["character_device"]
        and item["readable"]
        and item["writable"]
        for item in evidence
    )
    return {
        "verified": verified,
        "device_nodes": evidence,
        "required_pattern": str(device_root / "dpu_rank*"),
        "reason": "hardware_device_node_verified"
        if verified
        else "no_accessible_physical_dpu_rank_device_node",
    }


def _run_command(
    command: Sequence[str],
    cwd: Path,
    env: Mapping[str, str],
    timeout_seconds: float,
) -> dict[str, Any]:
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
    except subprocess.TimeoutExpired:
        cleanup = _terminate_process_group(process, process.pid)
        return {
            "status": "timeout",
            "returncode": process.returncode,
            "stdout": cleanup["stdout"],
            "stderr": cleanup["stderr"],
            "wall_s": time.perf_counter() - started,
            "timeout_cleanup": {
                "process_group": process.pid,
                "sigterm_sent": cleanup["sigterm_sent"],
                "sigkill_sent": cleanup["sigkill_sent"],
                "process_exited": process.returncode is not None,
                "group_probe_performed": cleanup["group_probe_performed"],
                "leader_exited_before_group_probe": cleanup[
                    "leader_exited_before_group_probe"
                ],
                "live_members_after_sigterm": cleanup["live_members_after_sigterm"],
                "live_members_after_cleanup": cleanup["live_members_after_cleanup"],
                "process_group_terminated": cleanup["process_group_terminated"],
                "signal_errors": cleanup["signal_errors"],
            },
        }
    return {
        "status": "passed" if process.returncode == 0 else "failed",
        "returncode": process.returncode,
        "stdout": stdout,
        "stderr": stderr,
        "wall_s": time.perf_counter() - started,
    }


def _terminate_process_group(
    process: subprocess.Popen[str],
    process_group: int,
) -> dict[str, Any]:
    signal_errors: list[str] = []
    sigterm_sent = _send_process_group_signal(
        process_group,
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
    process_group_exists = _process_group_exists(process_group)
    sigkill_sent = False
    if process_group_exists:
        sigkill_sent = _send_process_group_signal(
            process_group,
            signal.SIGKILL,
            signal_errors,
        )

    if not communicated:
        try:
            stdout, stderr = process.communicate(
                timeout=PROCESS_GROUP_KILL_WAIT_SECONDS
            )
        except subprocess.TimeoutExpired as exc:
            stdout = _timeout_output(exc.output)
            stderr = _timeout_output(exc.stderr)
            if process.poll() is None:
                process.kill()
                try:
                    stdout, stderr = process.communicate(timeout=1.0)
                except subprocess.TimeoutExpired:
                    pass

    live_members_after_cleanup = _wait_for_live_process_group_exit(
        process_group,
        PROCESS_GROUP_KILL_WAIT_SECONDS,
    )
    return {
        "stdout": stdout,
        "stderr": stderr,
        "sigterm_sent": sigterm_sent,
        "sigkill_sent": sigkill_sent,
        "group_probe_performed": True,
        "leader_exited_before_group_probe": leader_exited_before_group_probe,
        "live_members_after_sigterm": list(live_members_after_sigterm),
        "live_members_after_cleanup": list(live_members_after_cleanup),
        "process_group_terminated": not live_members_after_cleanup,
        "signal_errors": signal_errors,
    }


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


def _cli_selector_rejection(backend: str, target: str) -> str | None:
    if _is_simulator_value(backend):
        return f"simulator_backend_rejected:--backend={backend}"
    if _is_simulator_value(target):
        return f"simulator_backend_rejected:--target={target}"
    if backend.strip().lower() not in {"physical", "hardware", "hw"}:
        return f"unsupported_backend_selector:--backend={backend}"
    if target.strip().lower() not in {
        "physical",
        "physical_hardware",
        "hardware",
        "hw",
    }:
        return f"unsupported_target_selector:--target={target}"
    return None


def _external_backend_selector(env: Mapping[str, str]) -> str | None:
    for key in BACKEND_SELECTOR_KEYS:
        if key in env and env[key].strip():
            return f"{key}={env[key]}"
    return None


def _is_simulator_value(value: str) -> bool:
    normalized = value.strip().lower().replace("-", "_")
    return normalized in SIMULATOR_ALIASES or "simulator" in normalized


def _physical_environment(
    env: Mapping[str, str],
    host_cc: str | None = None,
) -> dict[str, str]:
    result = dict(env)
    for key in BACKEND_SELECTOR_KEYS:
        result.pop(key, None)
    result["DPU_BACKEND"] = "hw"
    result["HOST_CC"] = host_cc or _resolve_host_cc(env)
    result["DPU_CC"] = DPU_CC
    return result


def _record_hardware_preflight(
    payload: dict[str, Any],
    preflight: Mapping[str, Any],
) -> None:
    payload["hardware_preflight_verified"] = preflight.get("verified") is True
    nodes = preflight.get("device_nodes")
    payload["device_evidence"] = list(nodes) if isinstance(nodes, list) else []


def _record_staged_evidence(
    payload: dict[str, Any],
    staged_patch: Mapping[str, Any],
) -> None:
    payload["staged_patch"] = {
        "path": staged_patch["path"],
        "sha256": staged_patch["sha256"],
        "staged_sha256": staged_patch["staged_patch_sha256"],
        "applied": staged_patch["applied"],
        "replacement_count": staged_patch["replacement_count"],
        "command_fingerprint": staged_patch["command_fingerprint"],
        "staged_source_before_sha256": staged_patch["staged_source_before_sha256"],
        "staged_source_after_sha256": staged_patch["staged_source_after_sha256"],
        "staged_target_sha256": staged_patch["staged_target_sha256"],
    }
    source_hashes = dict(staged_patch["source_hashes"])
    payload["source_hashes"] = source_hashes
    payload["source_hash"] = source_hashes["combined_sha256"]


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
        "source_hash": plan["source_fingerprint"].get("combined_sha256"),
        "source_hashes": dict(plan["source_fingerprint"]),
        "command_fingerprint": plan.get("command_fingerprint"),
        "effective_compilers": dict(plan["effective_compilers"]),
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


def _failure_payload(
    plan: Mapping[str, Any],
    stage: str,
    reason: str,
) -> dict[str, Any]:
    payload = _base_payload(plan, "failed")
    payload["failure_stage"] = stage
    payload["reason"] = reason
    return payload


def _emit(payload: Mapping[str, Any], json_output: Path | None) -> int:
    _validate_output_schema(payload)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    if json_output is not None:
        json_output.parent.mkdir(parents=True, exist_ok=True)
        json_output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0 if payload["status"] in {"prepared", "passed"} else 1


def _validate_output_schema(payload: Mapping[str, Any]) -> None:
    required = {
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
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"strict output schema missing fields: {sorted(missing)}")
    optional = (
        {"commands"}
        if payload.get("status") == "prepared"
        else {
            "commands",
            "build",
            "execution",
            "host_result",
            "timeout_cleanup",
        }
    )
    allowed = required.union(optional)
    unknown = set(payload).difference(allowed)
    if unknown:
        raise ValueError(f"strict output schema has unknown fields: {sorted(unknown)}")
    if payload["schema_version"] != SCHEMA_VERSION:
        raise ValueError("strict output schema_version mismatch")
    if payload["provider_id"] != PROVIDER_ID or payload["probe_id"] != PROBE_ID:
        raise ValueError("strict provider/probe identity mismatch")
    if payload["status"] not in {"prepared", "passed", "failed"}:
        raise ValueError("strict status is invalid")
    if payload["target"] not in {None, "physical_hardware"}:
        raise ValueError("strict target is invalid")
    if payload["target_observed"] != payload["target"]:
        raise ValueError("target and target_observed must agree")
    if payload["requested_dpu_count"] != REQUESTED_DPUS:
        raise ValueError("strict requested_dpu_count is invalid")
    observed_dpus = payload["observed_dpu_count"]
    if observed_dpus is not None and (
        not isinstance(observed_dpus, int) or isinstance(observed_dpus, bool)
    ):
        raise ValueError("strict observed_dpu_count type is invalid")
    if payload["configured_tasklets_per_dpu"] != CONFIGURED_TASKLETS:
        raise ValueError("strict configured tasklet count is invalid")
    if payload["observed_tasklets_per_dpu"] is not None:
        raise ValueError("tasklets were configured, not independently observed")
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
        if not isinstance(payload[key], bool):
            raise ValueError(f"strict boolean field has wrong type: {key}")
    if payload["fallback"] is not False:
        raise ValueError("fallback must remain false")
    if payload["simulator_kernel_executed"] is not False:
        raise ValueError("simulator execution must remain false")
    if payload["physical_transfer_bytes_available"] is not False:
        raise ValueError("physical transfer bytes must remain unavailable")
    if payload["physical_transfer_bytes"] is not None:
        raise ValueError("physical transfer bytes must be null")
    if payload["release_status"] not in {
        "not_attempted",
        "released",
        "failed",
        "unknown",
    }:
        raise ValueError("strict release_status is invalid")
    if payload["backend_profile"] != BACKEND_PROFILE:
        raise ValueError("strict backend_profile is invalid")
    for key in (
        "device_evidence",
        "source_hashes",
        "effective_compilers",
        "staged_patch",
        "binary_hashes",
        "input_hashes",
        "logical_transfer_bytes",
        "timing",
    ):
        expected_type = list if key == "device_evidence" else Mapping
        if not isinstance(payload[key], expected_type):
            raise ValueError(f"strict container field has wrong type: {key}")
    for key in ("source_hash", "command_fingerprint", "output_hash"):
        value = payload[key]
        if value is not None and not _is_sha256(value):
            raise ValueError(f"strict nullable SHA-256 field is invalid: {key}")
    source_hashes = payload["source_hashes"]
    complete_source_hash_keys = {
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
    minimal_source_hash_keys = {
        "combined_sha256",
        "staged_source_before_patch_sha256",
        "staged_source_after_patch_sha256",
        "staged_patch_sha256",
    }
    allowed_source_keys = (
        {frozenset(complete_source_hash_keys), frozenset(minimal_source_hash_keys)}
        if payload["status"] == "failed"
        else {frozenset(complete_source_hash_keys)}
    )
    if frozenset(source_hashes) not in allowed_source_keys:
        raise ValueError("strict source hash fields are invalid")
    for key in (
        "combined_sha256",
        "owned_qualification_sha256",
        "upstream_library_sha256",
        "upstream_map_processing_sha256",
        "patch_sha256",
        "staged_source_before_patch_sha256",
        "staged_source_after_patch_sha256",
        "staged_patch_sha256",
    ):
        if (
            key in source_hashes
            and source_hashes[key] is not None
            and not _is_sha256(source_hashes[key])
        ):
            raise ValueError(f"strict source hash is invalid: {key}")
    effective_compilers = payload["effective_compilers"]
    expected_compilers = {"host_cc": HOST_CC, "dpu_cc": DPU_CC}
    if set(effective_compilers) != set(expected_compilers):
        raise ValueError("strict effective compiler roles are invalid")
    for role, expected_command in expected_compilers.items():
        identity = effective_compilers[role]
        if not isinstance(identity, Mapping):
            raise ValueError(f"strict compiler identity is invalid: {role}")
        if set(identity) != {"command", "available", "path", "sha256"}:
            raise ValueError(f"strict compiler identity fields are invalid: {role}")
        if identity.get("command") != expected_command:
            raise ValueError(f"strict compiler command is invalid: {role}")
        available = identity.get("available")
        path = identity.get("path")
        compiler_hash = identity.get("sha256")
        if not isinstance(available, bool):
            raise ValueError(f"strict compiler availability is invalid: {role}")
        if path is not None and not isinstance(path, str):
            raise ValueError(f"strict compiler path is invalid: {role}")
        if compiler_hash is not None and not _is_sha256(compiler_hash):
            raise ValueError(f"strict compiler hash is invalid: {role}")
        if available != (path is not None and compiler_hash is not None):
            raise ValueError(f"strict compiler evidence is inconsistent: {role}")
        if available:
            compiler_path = Path(path)
            if (
                not compiler_path.is_file()
                or _hash_file(compiler_path) != compiler_hash
            ):
                raise ValueError(f"strict compiler file evidence is invalid: {role}")
    staged_patch = payload["staged_patch"]
    if set(staged_patch) != {
        "path",
        "sha256",
        "staged_sha256",
        "applied",
        "replacement_count",
        "command_fingerprint",
        "staged_source_before_sha256",
        "staged_source_after_sha256",
        "staged_target_sha256",
    }:
        raise ValueError("strict staged patch fields are invalid")
    if staged_patch.get("path") != PATCH_RELATIVE_PATH:
        raise ValueError("strict staged patch path is invalid")
    if staged_patch.get("sha256") is not None and not _is_sha256(
        staged_patch["sha256"]
    ):
        raise ValueError("strict staged patch hash is invalid")
    staged_patch_hash = staged_patch.get("staged_sha256")
    if staged_patch_hash is not None and not _is_sha256(staged_patch_hash):
        raise ValueError("strict copied patch hash is invalid")
    if not isinstance(staged_patch.get("applied"), bool):
        raise ValueError("strict staged patch applied flag is invalid")
    replacement_count = staged_patch.get("replacement_count")
    if not isinstance(replacement_count, int) or isinstance(replacement_count, bool):
        raise ValueError("strict staged patch replacement count is invalid")
    staged_target_hash = staged_patch.get("staged_target_sha256")
    if staged_target_hash is not None and not _is_sha256(staged_target_hash):
        raise ValueError("strict staged target hash is invalid")
    patch_command_hash = staged_patch.get("command_fingerprint")
    if patch_command_hash is not None and not _is_sha256(patch_command_hash):
        raise ValueError("strict patch command fingerprint is invalid")
    staged_source_hashes = (
        staged_patch.get("staged_source_before_sha256"),
        staged_patch.get("staged_source_after_sha256"),
    )
    if any(
        value is not None and not _is_sha256(value) for value in staged_source_hashes
    ):
        raise ValueError("strict staged source tree hash is invalid")
    if staged_patch["applied"]:
        if not (
            replacement_count == 2
            and staged_patch_hash == staged_patch.get("sha256")
            and all(staged_source_hashes)
            and staged_source_hashes[0] != staged_source_hashes[1]
            and staged_target_hash is not None
            and source_hashes.get("patch_sha256") == staged_patch.get("sha256")
            and source_hashes.get("upstream_map_processing_sha256")
            == staged_target_hash
            and source_hashes.get("staged_source_before_patch_sha256")
            == staged_source_hashes[0]
            and source_hashes.get("staged_patch_sha256") == staged_patch_hash
        ):
            raise ValueError("strict applied patch evidence is incomplete")
        source_after = staged_source_hashes[1]
        if not (
            payload["source_hash"]
            == source_hashes.get("combined_sha256")
            == source_hashes.get("staged_source_after_patch_sha256")
            == source_after
        ):
            raise ValueError("strict staged source hash chain is inconsistent")
    elif replacement_count != 0:
        raise ValueError("strict unapplied patch replacement count is invalid")
    for collection_key in ("binary_hashes", "input_hashes"):
        for name, value in payload[collection_key].items():
            if not isinstance(name, str) or (
                value is not None and not _is_sha256(value)
            ):
                raise ValueError(f"strict artifact hash is invalid: {collection_key}")
    for item in payload["device_evidence"]:
        if not isinstance(item, Mapping):
            raise ValueError("strict device evidence item must be a mapping")
        required_device_keys = {
            "path",
            "exists",
            "character_device",
            "readable",
            "writable",
            "sysfs_path",
            "sysfs_exists",
        }
        if not required_device_keys.issubset(item) or set(item).difference(
            required_device_keys | {"error"}
        ):
            raise ValueError("strict device evidence fields are invalid")
        for key in (
            "exists",
            "character_device",
            "readable",
            "writable",
            "sysfs_exists",
        ):
            if not isinstance(item[key], bool):
                raise ValueError(f"strict device evidence boolean is invalid: {key}")
    logical = payload["logical_transfer_bytes"]
    if logical.get("h2d") != LOGICAL_INPUT_BYTES:
        raise ValueError("strict logical H2D byte count is invalid")
    if logical.get("d2h") != LOGICAL_OUTPUT_BYTES:
        raise ValueError("strict logical D2H byte count is invalid")
    if logical.get("total") != LOGICAL_TOTAL_BYTES:
        raise ValueError("strict logical total byte count is invalid")
    if logical.get("scope") != "logical_application_payload_only":
        raise ValueError("strict logical byte scope is invalid")
    if payload["payload_sizes_8_byte_aligned"] is not True:
        raise ValueError("strict payload alignment assertion is invalid")
    if payload["reason"] is not None and not isinstance(payload["reason"], str):
        raise ValueError("strict reason type is invalid")
    if payload["status"] != "passed" and not payload["reason"]:
        raise ValueError("non-passed result requires a reason")
    if payload["failure_stage"] is not None and not isinstance(
        payload["failure_stage"], str
    ):
        raise ValueError("strict failure_stage type is invalid")
    if payload["target"] == "physical_hardware" and not (
        payload["hardware_preflight_verified"] and payload["native_execution"]
    ):
        raise ValueError("physical target requires preflight and native execution")
    commands: Mapping[str, Any] | None = None
    if "commands" in payload:
        commands = payload["commands"]
        if not isinstance(commands, Mapping) or set(commands) != {
            "patch",
            "build",
            "run",
        }:
            raise ValueError("strict commands evidence is invalid")
        if staged_patch.get("command_fingerprint") != _hash_json(commands.get("patch")):
            raise ValueError("patch command fingerprint does not match command")
        if payload["command_fingerprint"] != _hash_json(commands):
            raise ValueError("command fingerprint does not match commands")
        host_identity = effective_compilers["host_cc"]
        dpu_identity = effective_compilers["dpu_cc"]
        if (
            f"HOST_CC={host_identity.get('path')}" not in commands["build"]
            or f"DPU_CC={dpu_identity.get('command')}" not in commands["build"]
        ):
            raise ValueError("build command contradicts compiler identities")
    build_evidence = payload.get("build")
    execution_evidence = payload.get("execution")
    if build_evidence is not None:
        expected = commands.get("build") if commands is not None else None
        _validate_command_evidence_schema(build_evidence, expected, "build")
    if execution_evidence is not None:
        expected = commands.get("run") if commands is not None else None
        _validate_command_evidence_schema(execution_evidence, expected, "execution")
    host_result = payload.get("host_result")
    if payload["status"] == "passed":
        host_passed = (
            isinstance(host_result, Mapping)
            and _host_contract_failure(host_result) is None
            and host_result.get("configured_tasklets_per_dpu")
            == payload["configured_tasklets_per_dpu"]
            and host_result.get("observed_dpu_count") == payload["observed_dpu_count"]
            and host_result.get("release_status") == payload["release_status"]
        )
        build_passed = (
            isinstance(build_evidence, Mapping)
            and build_evidence.get("status") == "passed"
            and build_evidence.get("returncode") == 0
        )
        execution_passed = (
            isinstance(execution_evidence, Mapping)
            and execution_evidence.get("status") == "passed"
            and execution_evidence.get("returncode") == 0
        )
        if not (
            payload["target"] == "physical_hardware"
            and payload["hardware_preflight_verified"]
            and _device_evidence_proves_physical(payload["device_evidence"])
            and payload["observed_dpu_count"] == REQUESTED_DPUS
            and payload["native_execution"]
            and payload["validation_performed"]
            and payload["exact_validation"]
            and payload["release_status"] == "released"
            and payload["failure_stage"] is None
            and payload["reason"] is None
            and payload["staged_patch"].get("applied") is True
            and payload["effective_compilers"]["host_cc"].get("available") is True
            and payload["effective_compilers"]["dpu_cc"].get("available") is True
            and commands is not None
            and build_passed
            and execution_passed
            and host_passed
        ):
            raise ValueError("passed result does not satisfy strict qualification")
    if "timeout_cleanup" in payload:
        _validate_timeout_cleanup_schema(payload["timeout_cleanup"])
    build = payload.get("build")
    if isinstance(build, Mapping) and "timeout_cleanup" in build:
        _validate_timeout_cleanup_schema(build["timeout_cleanup"])


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_command_evidence_schema(
    value: Any,
    expected_command: Any,
    label: str,
) -> None:
    if not isinstance(value, Mapping):
        raise ValueError(f"strict {label} evidence must be a mapping")
    required = {
        "command",
        "command_fingerprint",
        "status",
        "returncode",
        "wall_s",
        "stdout_tail",
        "stderr_tail",
    }
    if set(value).difference(required | {"timeout_cleanup"}) or not required.issubset(
        value
    ):
        raise ValueError(f"strict {label} evidence fields are invalid")
    command = value["command"]
    if (
        not isinstance(command, list)
        or not command
        or any(not isinstance(argument, str) or not argument for argument in command)
    ):
        raise ValueError(f"strict {label} command is invalid")
    if expected_command is not None and command != list(expected_command):
        raise ValueError(f"strict {label} command contradicts planned command")
    if value["command_fingerprint"] != _hash_json(command):
        raise ValueError(f"strict {label} command fingerprint is invalid")
    status = value["status"]
    returncode = value["returncode"]
    if status not in {"passed", "failed", "timeout"}:
        raise ValueError(f"strict {label} status is invalid")
    if returncode is not None and (
        not isinstance(returncode, int) or isinstance(returncode, bool)
    ):
        raise ValueError(f"strict {label} returncode is invalid")
    if status == "passed" and returncode != 0:
        raise ValueError(f"strict {label} passed status contradicts returncode")
    if status == "failed" and returncode == 0:
        raise ValueError(f"strict {label} failed status contradicts returncode")
    wall_s = value["wall_s"]
    if (
        not isinstance(wall_s, (int, float))
        or isinstance(wall_s, bool)
        or not math.isfinite(wall_s)
        or wall_s < 0
    ):
        raise ValueError(f"strict {label} wall time is invalid")
    if not isinstance(value["stdout_tail"], str) or not isinstance(
        value["stderr_tail"], str
    ):
        raise ValueError(f"strict {label} output tails are invalid")
    if "timeout_cleanup" in value:
        _validate_timeout_cleanup_schema(value["timeout_cleanup"])


def _device_evidence_proves_physical(value: Any) -> bool:
    return bool(
        isinstance(value, list)
        and any(
            isinstance(item, Mapping)
            and str(item.get("path", "")).startswith("/dev/dpu_rank")
            and item.get("exists") is True
            and item.get("character_device") is True
            and item.get("readable") is True
            and item.get("writable") is True
            for item in value
        )
    )


def _validate_timeout_cleanup_schema(value: Any) -> None:
    if not isinstance(value, Mapping):
        raise ValueError("strict timeout cleanup must be a mapping")
    required = {
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
    if set(value) != required:
        raise ValueError("strict timeout cleanup fields are invalid")
    if (
        not isinstance(value["process_group"], int)
        or isinstance(value["process_group"], bool)
        or value["process_group"] <= 0
    ):
        raise ValueError("strict timeout process group is invalid")
    for key in (
        "sigterm_sent",
        "sigkill_sent",
        "process_exited",
        "group_probe_performed",
        "leader_exited_before_group_probe",
        "process_group_terminated",
    ):
        if not isinstance(value[key], bool):
            raise ValueError(f"strict timeout boolean is invalid: {key}")
    for key in ("live_members_after_sigterm", "live_members_after_cleanup"):
        members = value[key]
        if not isinstance(members, list) or any(
            not isinstance(member, int) or isinstance(member, bool) or member <= 0
            for member in members
        ):
            raise ValueError(f"strict timeout PID list is invalid: {key}")
    errors = value["signal_errors"]
    if not isinstance(errors, list) or any(
        not isinstance(message, str) for message in errors
    ):
        raise ValueError("strict timeout signal errors are invalid")


def _minimal_plan(workdir: Path) -> dict[str, Any]:
    return {
        "workdir": workdir,
        "commands": {},
        "command_fingerprint": None,
        "effective_compilers": {
            "host_cc": _compiler_identity(HOST_CC),
            "dpu_cc": _compiler_identity(DPU_CC),
        },
        "source_fingerprint": {
            "combined_sha256": None,
            "staged_source_before_patch_sha256": None,
            "staged_source_after_patch_sha256": None,
            "staged_patch_sha256": None,
        },
        "staged_patch": {
            "path": PATCH_RELATIVE_PATH,
            "sha256": None,
            "staged_sha256": None,
            "applied": False,
            "replacement_count": 0,
            "command_fingerprint": None,
            "staged_source_before_sha256": None,
            "staged_source_after_sha256": None,
            "staged_target_sha256": None,
        },
    }


def _command_summary(
    result: Mapping[str, Any],
    command: Sequence[str],
) -> dict[str, Any]:
    executed_command = list(command)
    summary = {
        "command": executed_command,
        "command_fingerprint": _hash_json(executed_command),
        **{key: result.get(key) for key in ("status", "returncode", "wall_s")},
    }
    for key in ("stdout", "stderr"):
        text = str(result.get(key) or "")
        summary[f"{key}_tail"] = text[-1000:] if text else ""
    if isinstance(result.get("timeout_cleanup"), Mapping):
        summary["timeout_cleanup"] = dict(result["timeout_cleanup"])
    return summary


def _require_file(path: Path) -> None:
    if not path.is_file():
        raise ValueError(f"required source file missing: {path}")


def _patch_hunk_context_size(header: str) -> int:
    fields = header.split()
    if len(fields) < 3:
        return 0

    def line_count(field: str) -> int:
        _, separator, count = field.partition(",")
        if not separator:
            return 1
        try:
            return int(count)
        except ValueError:
            return 0

    return min(line_count(fields[1]), line_count(fields[2]))


def _patch_hunks_have_nonblank_context(patch_text: str) -> bool:
    hunks: list[list[str]] = []
    current: list[str] | None = None
    for line in patch_text.splitlines():
        if line.startswith("@@ "):
            current = []
            hunks.append(current)
        elif current is not None:
            current.append(line)
    return bool(hunks) and all(
        any(line.startswith(" ") and line[1:].strip() for line in hunk)
        for hunk in hunks
    )


def _resolve_host_cc(env: Mapping[str, str]) -> str:
    configured = env.get("HOST_CC", HOST_CC).strip()
    if not configured:
        raise ValueError("HOST_CC must name one compiler executable")
    try:
        command = shlex.split(configured)
    except ValueError as exc:
        raise ValueError(f"HOST_CC is not a valid compiler selector: {exc}") from exc
    if len(command) != 1:
        raise ValueError("HOST_CC must name one compiler executable without arguments")
    raw_path = shutil.which(command[0], path=env.get("PATH"))
    if raw_path is None:
        raise ValueError(f"HOST_CC compiler is unavailable: {command[0]}")
    resolved = Path(raw_path).resolve()
    if not resolved.is_file():
        raise ValueError(f"HOST_CC compiler is not a file: {resolved}")
    return str(resolved)


def _compiler_identity(
    command: str,
    resolved_path: Path | None = None,
) -> dict[str, Any]:
    raw_path = (
        str(resolved_path) if resolved_path is not None else shutil.which(command)
    )
    path = Path(raw_path).resolve() if raw_path else None
    available = bool(path and path.is_file())
    return {
        "command": command,
        "available": available,
        "path": str(path) if available else None,
        "sha256": _hash_file(path) if available and path is not None else None,
    }


def _hash_tree(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(path.rglob("*")):
        if item.is_file():
            digest.update(item.relative_to(path).as_posix().encode("utf-8"))
            digest.update(b"\0")
            digest.update(_hash_file(item).encode("ascii"))
            digest.update(b"\n")
    return digest.hexdigest()


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _combine_hashes(values: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _hash_json(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _optional_hash(path: Path) -> str | None:
    return _hash_file(path) if path.is_file() else None


def _binary_hashes(path: Path) -> dict[str, str]:
    if not path.is_dir():
        return {}
    return {
        item.name: _hash_file(item) for item in sorted(path.iterdir()) if item.is_file()
    }


if __name__ == "__main__":
    raise SystemExit(main())
