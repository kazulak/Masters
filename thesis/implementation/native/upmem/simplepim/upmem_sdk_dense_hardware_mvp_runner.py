"""Hardware-only Phase 1A UPMEM dense MVP runner.

This module deliberately keeps the native boundary small: one isolated build and
one host invocation.  It never selects the SDK simulator and never computes a
replacement result when the physical path fails.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np


SCHEMA = "dense_bridge_v1"
BRIDGE_ID = "upmem_dense_bridge_v1"
BACKEND = "upmem_sdk_hardware_dense"
MAX_DIM = 4
PROFILE_VERSION = "hardware_mvp_l1_v1"
STAGES = {
    "hardware_opt_in_missing",
    "hardware_profile_violation",
    "sdk_discovery_failed",
    "native_build_failed",
    "hardware_allocation_failed",
    "binary_load_failed",
    "argument_transfer_failed",
    "operand_transfer_failed",
    "kernel_launch_failed",
    "kernel_timeout",
    "result_transfer_failed",
    "output_manifest_failed",
    "output_validation_failed",
    "hardware_release_failed",
}


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    return value


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def _inside(base: Path, relative: str) -> Path:
    path = (base / relative).resolve()
    if base.resolve() not in path.parents:
        raise ValueError(f"manifest path escapes bridge directory: {relative}")
    return path


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _base(
    manifest: dict[str, Any] | None,
    started: float,
    reason: str,
    error: str | None,
    *,
    returncode: int | None = None,
    command: Any = None,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA,
        "bridge_id": BRIDGE_ID,
        "manifest_kind": "dense_bridge_output",
        "backend": BACKEND,
        "status": "failed",
        "input_manifest": "input_manifest.json",
        "route_id": str((manifest or {}).get("route_id", "")),
        "task_id": str((manifest or {}).get("task_id", "")),
        "output_blob": None,
        "accumulator_blob": None,
        "validation_metrics": {},
        "compute_time_s": 0.0,
        "write_time_s": 0.0,
        "total_time_s": time.perf_counter() - started,
        "external_command_executed": command is not None,
        "execution_implemented": True,
        "error": error,
        "metadata": {
            "reason": reason,
            "error_type": reason,
            "hardware_stage": reason,
            "backend_family": "upmem_sdk",
            "target": "hardware",
            "hardware_kernel_executed": False,
            "speedup_claims": False,
            "timing_labels": "hardware_bringup_functionality_only",
            "command": command,
            "returncode": returncode,
        },
    }


def _read_status(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if isinstance(value, dict):
        return value
    return {}


def _tools(env: Mapping[str, str]) -> dict[str, str | None]:
    home = env.get("UPMEM_HOME")

    def find(name: str) -> str | None:
        if home:
            candidate = Path(home) / "bin" / name
            if candidate.exists():
                return str(candidate)
        return shutil.which(name, path=env.get("PATH"))

    return {name: find(name) for name in ("dpu-upmem-dpurte-clang", "dpu-pkg-config")}


def _run(
    command: list[str], cwd: Path, env: Mapping[str, str], timeout: float
) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            env=dict(env),
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )
        return {
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "timed_out": False,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "returncode": None,
            "stdout": str(exc.stdout or ""),
            "stderr": str(exc.stderr or ""),
            "timed_out": True,
        }


def _snippet(value: object, limit: int = 2000) -> str:
    text = str(value or "")
    return text if len(text) <= limit else text[-limit:]


def _failure_stage(
    result: dict[str, Any], default: str, status: Mapping[str, Any] | None = None
) -> str:
    text = f"{result.get('stdout', '')}\n{result.get('stderr', '')}"
    stage = str((status or {}).get("failure_stage", ""))
    if stage in STAGES:
        return stage
    lower = text.lower()
    keyword_stages = (
        (("alloc", "dpu_alloc"), "hardware_allocation_failed"),
        (("load", "dpu_load", "binary"), "binary_load_failed"),
        (("argument", "dense_args"), "argument_transfer_failed"),
        (("operand", "dense_a", "dense_b", "broadcast"), "operand_transfer_failed"),
        (("launch", "dpu_launch"), "kernel_launch_failed"),
        (("copy_from", "result", "dense_c"), "result_transfer_failed"),
        (("free", "release"), "hardware_release_failed"),
    )
    for keywords, candidate in keyword_stages:
        if any(keyword in lower for keyword in keywords):
            return candidate
    return default


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="UPMEM physical hardware dense MVP runner"
    )
    parser.add_argument("--input-manifest", required=True)
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--backend-id", default=BACKEND)
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    args = parser.parse_args(argv)
    started = time.perf_counter()
    input_path, output_path = Path(args.input_manifest), Path(args.output_manifest)
    bridge = input_path.parent.resolve()
    manifest: dict[str, Any] | None = None
    command: list[str] | None = None

    def fail(stage: str, message: str, result: dict[str, Any] | None = None) -> int:
        payload = _base(
            manifest,
            started,
            stage,
            message,
            returncode=result.get("returncode") if result else None,
            command=command,
        )
        if result:
            payload["metadata"].update(
                {
                    "stdout_snippet": result.get("stdout", "")[-2000:],
                    "stderr_snippet": result.get("stderr", "")[-2000:],
                }
            )
        _write(output_path, payload)
        return 1

    try:
        if float(args.timeout_seconds) != 30.0:
            return fail(
                "hardware_profile_violation",
                "hardware_mvp_l1_v1 requires a fixed 30-second timeout",
            )
        manifest = json.loads(input_path.read_text(encoding="utf-8"))
        if (
            manifest.get("schema_version") != SCHEMA
            or manifest.get("bridge_id") != BRIDGE_ID
        ):
            return fail(
                "output_manifest_failed", "unsupported dense bridge input manifest"
            )
        if args.backend_id != BACKEND:
            return fail(
                "hardware_profile_violation",
                "hardware backend ID must be upmem_sdk_hardware_dense",
            )
        if os.environ.get("UPMEM_ALLOW_PHYSICAL_HARDWARE") != "1":
            return fail(
                "hardware_opt_in_missing", "UPMEM_ALLOW_PHYSICAL_HARDWARE=1 is required"
            )
        if os.environ.get("DPU_BACKEND"):
            return fail(
                "hardware_profile_violation", "DPU_BACKEND must not be inherited"
            )
        if os.environ.get("UPMEM_PROFILE", "hw") != "hw":
            return fail("hardware_profile_violation", "UPMEM_PROFILE must be hw")

        dims = tuple(int(manifest[key]) for key in ("gemm_m", "gemm_k", "gemm_n"))
        if any(value <= 0 or value > MAX_DIM for value in dims):
            return fail(
                "hardware_profile_violation",
                f"hardware MVP supports dimensions <= {MAX_DIM}",
            )
        if (
            manifest.get("dequantization", {}).get("left", {}).get("route_dtype")
            != "int8"
            or manifest.get("dequantization", {}).get("right", {}).get("route_dtype")
            != "int8"
        ):
            return fail(
                "hardware_profile_violation", "hardware MVP requires int8 operands"
            )
        if not _profile_metadata_valid(manifest):
            return fail(
                "hardware_profile_violation",
                "hardware MVP profile metadata is missing or invalid",
            )
        left_path = _inside(bridge, manifest["operands"]["left"]["relative_path"])
        right_path = _inside(bridge, manifest["operands"]["right"]["relative_path"])
        expected_path = _inside(bridge, manifest["expected_output"]["relative_path"])
        left = np.load(left_path, allow_pickle=False).astype(np.int8, copy=False)
        right = np.load(right_path, allow_pickle=False).astype(np.int8, copy=False)
        expected = np.load(expected_path, allow_pickle=False)
        if left.shape != (dims[0], dims[1]) or right.shape != (dims[1], dims[2]):
            return fail(
                "hardware_profile_violation",
                "operand shapes do not match GEMM dimensions",
            )
        tools = _tools(os.environ)
        if any(value is None for value in tools.values()):
            return fail(
                "sdk_discovery_failed",
                "UPMEM SDK compiler and dpu-pkg-config are required",
            )

        source = Path(__file__).resolve().parent / "upmem_sdk_dense"
        work = bridge / "hardware_runner_work"
        source_snapshot, build = work / "src", work / "build"
        shutil.copytree(source, source_snapshot, dirs_exist_ok=True)
        shutil.copytree(source, build, dirs_exist_ok=True)
        build_env = dict(os.environ)
        build_env.pop("DPU_BACKEND", None)
        build_env["UPMEM_PROFILE"] = "hw"
        build_cmd = [
            "make",
            "clean",
            "all",
            "MAX_DIM=4",
            "L2_MAX_DIM=4",
            "L2_TILE_MAX_DIM=4",
            "NR_TASKLETS=1",
            "UPMEM_DENSE_HARDWARE_MVP=1",
        ]
        build_started = time.perf_counter()
        build_result = _run(build_cmd, build, build_env, args.timeout_seconds)
        build_time_s = time.perf_counter() - build_started
        if build_result["timed_out"] or build_result["returncode"] != 0:
            return fail(
                "native_build_failed",
                "isolated hardware native build failed",
                build_result,
            )

        inputs, outputs = work / "inputs", work / "outputs"
        inputs.mkdir(parents=True, exist_ok=True)
        outputs.mkdir(parents=True, exist_ok=True)
        left_bin, right_bin, raw_bin = (
            inputs / "left_i8.bin",
            inputs / "right_i8.bin",
            outputs / "raw_i32.bin",
        )
        _write_l1_padded_i8(left_bin, left, rows=dims[0], cols=dims[1])
        _write_l1_padded_i8(right_bin, right, rows=dims[1], cols=dims[2])
        binary = build / "bin" / "dpu_dense"
        host = build / "bin" / "host"
        command = [
            str(host),
            str(binary),
            str(dims[0]),
            str(dims[1]),
            str(dims[2]),
            str(left_bin),
            str(right_bin),
            str(raw_bin),
        ]
        status_path = work / "host_status.json"
        host_env = dict(build_env)
        host_env["UPMEM_DENSE_STATUS_JSON"] = str(status_path)
        host_started = time.perf_counter()
        result = _run(command, bridge, host_env, args.timeout_seconds)
        host_time_s = time.perf_counter() - host_started
        host_status = _read_status(status_path)
        if result["timed_out"]:
            return fail("kernel_timeout", "hardware host invocation timed out", result)
        if result["returncode"] != 0:
            return fail(
                _failure_stage(result, "kernel_launch_failed", host_status),
                "hardware host invocation failed",
                result,
            )
        if not host_status:
            return fail(
                "output_manifest_failed",
                "hardware host did not emit its required status sidecar",
                result,
            )
        if host_status.get("success") is not True:
            return fail(
                _failure_stage(result, "output_manifest_failed", host_status),
                "hardware host status sidecar reports failure",
                result,
            )
        if not raw_bin.exists():
            return fail(
                "result_transfer_failed",
                "native output buffer was not produced",
                result,
            )

        raw_full = np.fromfile(raw_bin, dtype="<i4")
        if raw_full.size != MAX_DIM * MAX_DIM:
            return fail(
                "result_transfer_failed",
                "native L1 output has an unexpected padded size",
                result,
            )
        accumulator = raw_full.reshape((MAX_DIM, MAX_DIM))[: dims[0], : dims[2]].astype(
            "<i4", copy=False
        )
        reference = np.matmul(left.astype(np.int32), right.astype(np.int32)).astype(
            "<i4", copy=False
        )
        if not np.array_equal(accumulator, reference):
            return fail(
                "output_validation_failed",
                "raw int32 accumulator differs from CPU reference",
                result,
            )
        expected_accumulator = _load_expected_accumulator(bridge, manifest, dims)
        if expected_accumulator is not None and not np.array_equal(
            accumulator, expected_accumulator
        ):
            return fail(
                "output_validation_failed",
                "raw int32 accumulator differs from retained CPU reference",
                result,
            )
        scales = float(manifest["dequantization"]["left"]["scale"]) * float(
            manifest["dequantization"]["right"]["scale"]
        )
        output = (accumulator.astype(np.float64) * scales).reshape(expected.shape)
        if not np.allclose(output, expected, atol=1e-12, rtol=1e-12):
            return fail(
                "output_validation_failed",
                "dequantized output differs from expected reference",
                result,
            )

        raw_path = outputs / "hardware_accumulator_crop_i32.npy"
        out_path = bridge / "outputs" / "upmem_sdk_hardware_output.npy"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(raw_path, accumulator, allow_pickle=False)
        write_started = time.perf_counter()
        np.save(out_path, output.astype(expected.dtype, copy=False), allow_pickle=False)
        write_time = time.perf_counter() - write_started
        h2d_components = {
            "arguments": 40,
            "left_padded_int8": MAX_DIM * MAX_DIM,
            "right_padded_int8": MAX_DIM * MAX_DIM,
        }
        d2h_components = {"output_padded_int32": MAX_DIM * MAX_DIM * 4}
        h2d_bytes = sum(h2d_components.values())
        d2h_bytes = sum(d2h_components.values())
        sdk_tools = _tool_versions(tools)
        metadata = {
            "reason": "upmem_sdk_hardware_executed",
            "error_type": None,
            "backend_family": "upmem_sdk",
            "target": "hardware",
            "hardware_status_json": host_status,
            "raw_accumulator_crop": True,
            "cpu_reference": "int8_x_int8_to_int32_exact",
            "hashes": {
                "left": _hash_file(left_path),
                "right": _hash_file(right_path),
                "accumulator": _hash_file(raw_path),
                "output": _hash_file(out_path),
                "host_binary": _hash_file(host),
                "dpu_binary": _hash_file(binary),
                "input_manifest": _hash_file(input_path),
                "native_source_snapshot": _hash_tree(source_snapshot),
            },
            "application_visible_transfer_bytes": {
                "h2d": h2d_bytes,
                "d2h": d2h_bytes,
                "total": h2d_bytes + d2h_bytes,
                "h2d_components": h2d_components,
                "d2h_components": d2h_components,
                "scope": "application_visible_sdk_buffers_not_physical_bus_counters",
            },
            "timing_labels": "hardware_bringup_functionality_only",
            "speedup_claims": False,
            "hardware_kernel_executed": True,
            "native_kernel_executed": True,
            "simulator_kernel_executed": False,
            "cpu_fallback_used": False,
            "native_build": build_cmd,
            "native_source_snapshot_path": source_snapshot.relative_to(
                bridge
            ).as_posix(),
            "host_command": command,
            "host_invocation_count": 1,
            "build_time_s": build_time_s,
            "native_process_wall_time_s": host_time_s,
            "native_build_stdout_snippet": _snippet(build_result.get("stdout")),
            "native_build_stderr_snippet": _snippet(build_result.get("stderr")),
            "host_stdout_snippet": _snippet(result.get("stdout")),
            "host_stderr_snippet": _snippet(result.get("stderr")),
            "timing_decomposition_available": False,
            "timing_decomposition_note": "SDK host program does not separately measure allocation/load/H2D/kernel/D2H in Phase 1A.",
            "sdk_tools": sdk_tools,
            "sdk_metadata": {
                "upmem_profile": "hw",
                "tools": sdk_tools,
            },
            "compiler_metadata": {
                "dpu_upmem_dpurte_clang": sdk_tools.get("dpu-upmem-dpurte-clang"),
                "host_gcc": _command_version("gcc", os.environ),
            },
        }
        payload = {
            "schema_version": SCHEMA,
            "bridge_id": BRIDGE_ID,
            "manifest_kind": "dense_bridge_output",
            "backend": BACKEND,
            "status": "upmem_sdk_hardware_executed",
            "input_manifest": input_path.name,
            "route_id": manifest.get("route_id", ""),
            "task_id": manifest.get("task_id", ""),
            "output_blob": {
                "relative_path": out_path.relative_to(bridge).as_posix(),
                "dtype": str(output.dtype),
                "shape": output.shape,
                "representation": "dequantized_output",
                "nbytes": output.nbytes,
                "labels": manifest.get("output_labels", ()),
                "role": "hardware_output",
            },
            "accumulator_blob": {
                "relative_path": raw_path.relative_to(bridge).as_posix(),
                "dtype": "<i4",
                "shape": accumulator.shape,
                "representation": "int32_accumulator_crop",
                "nbytes": accumulator.nbytes,
                "role": "hardware_accumulator_crop",
            },
            "validation_metrics": {
                "reference_kind": "exact_int8_x_int8_to_int32_cpu_reference",
                "exact_integer_passed": True,
                "passed": True,
                "max_abs_error": 0.0,
                "l2_error": 0.0,
                "relative_l2_error": 0.0,
            },
            "compute_time_s": 0.0,
            "write_time_s": write_time,
            "total_time_s": time.perf_counter() - started,
            "external_command_executed": True,
            "execution_implemented": True,
            "error": None,
            "metadata": metadata,
        }
        _write(output_path, payload)
        return 0
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        return fail(
            "output_manifest_failed"
            if manifest is None
            else "output_validation_failed",
            str(exc),
        )


def _write_l1_padded_i8(path: Path, array: np.ndarray, *, rows: int, cols: int) -> None:
    padded = np.zeros((MAX_DIM, MAX_DIM), dtype=np.int8)
    padded[:rows, :cols] = array
    padded.tofile(path)


def _profile_metadata_valid(manifest: Mapping[str, Any]) -> bool:
    metadata = manifest.get("metadata")
    fixed = manifest.get("fixed_point_spec")
    operands = manifest.get("operands")
    if (
        not isinstance(metadata, Mapping)
        or not isinstance(fixed, Mapping)
        or not isinstance(operands, Mapping)
    ):
        return False
    return (
        metadata.get("hardware_profile_version") == PROFILE_VERSION
        and metadata.get("target") == "hardware"
        and metadata.get("execution_class") == "L1_WRAM"
        and metadata.get("backend_id") == BACKEND
        and metadata.get("requested_dpu_count") == 1
        and metadata.get("tasklets_per_dpu") == 1
        and metadata.get("synchronous_execution") is True
        and metadata.get("performance_claim_applicable") is False
        and fixed.get("route_dtype") == "int8"
        and fixed.get("complex_policy") == "reject"
        and float(fixed.get("scale", 0.0)) == 1.0
        and isinstance(operands.get("left"), Mapping)
        and isinstance(operands.get("right"), Mapping)
        and operands["left"].get("dtype") == "int8"
        and operands["right"].get("dtype") == "int8"
    )


def _load_expected_accumulator(
    bridge: Path,
    manifest: Mapping[str, Any],
    dims: tuple[int, int, int],
) -> np.ndarray | None:
    metadata = manifest.get("metadata")
    expected = (
        metadata.get("expected_accumulator") if isinstance(metadata, Mapping) else None
    )
    if not isinstance(expected, Mapping):
        return None
    relative = expected.get("relative_path")
    if not isinstance(relative, str):
        return None
    value = np.load(_inside(bridge, relative), allow_pickle=False).astype(
        "<i4", copy=False
    )
    if value.shape != (dims[0], dims[2]):
        raise ValueError("retained expected accumulator has the wrong shape")
    return value


def _hash_tree(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(path.rglob("*")):
        if not item.is_file():
            continue
        digest.update(item.relative_to(path).as_posix().encode("utf-8"))
        digest.update(_hash_file(item).encode("ascii"))
    return digest.hexdigest()


def _tool_versions(tools: Mapping[str, str | None]) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name, path in tools.items():
        if path is None:
            versions[name] = None
            continue
        result = _run([path, "--version"], Path.cwd(), os.environ, 10.0)
        text = (result.get("stdout") or result.get("stderr") or "").strip()
        versions[name] = (
            text.splitlines()[0] if result.get("returncode") == 0 and text else None
        )
    return versions


def _command_version(command: str, env: Mapping[str, str]) -> str | None:
    path = shutil.which(command, path=env.get("PATH"))
    if path is None:
        return None
    result = _run([path, "--version"], Path.cwd(), env, 10.0)
    text = (result.get("stdout") or result.get("stderr") or "").strip()
    return text.splitlines()[0] if result.get("returncode") == 0 and text else None


if __name__ == "__main__":
    raise SystemExit(main())
