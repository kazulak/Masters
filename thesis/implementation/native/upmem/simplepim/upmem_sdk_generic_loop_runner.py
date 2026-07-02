from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np


GENERIC_BRIDGE_SCHEMA_VERSION = "generic_contraction_bridge_v1"
GENERIC_BRIDGE_ID = "upmem_generic_contraction_bridge_v1"
BACKEND_ID = "upmem_sdk_simulator_generic_loop"
KERNEL_FAMILY = "generic_loop_fallback"
DEFAULT_MAX_RANK = 6
DEFAULT_MAX_ELEMS = 4096
DEFAULT_TIMEOUT_SECONDS = 30.0
SNIPPET_LIMIT = 2000
MODE_INT8_SCALED = "int8_scaled"
MODE_FLOAT32_NO_QUANT = "float32_no_quant"
NATIVE_MODE_IDS = {
    MODE_INT8_SCALED: 0,
    MODE_FLOAT32_NO_QUANT: 1,
}


def main() -> int:
    parser = argparse.ArgumentParser(description="UPMEM SDK simulator generic tensor-contraction loop runner")
    parser.add_argument("--input-manifest", required=True)
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--backend-id", default=BACKEND_ID)
    parser.add_argument("--target", default="simulator", choices=("simulator", "hardware"))
    parser.add_argument("--source-dir")
    parser.add_argument("--timeout-seconds", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    args = parser.parse_args()

    started = time.perf_counter()
    input_path = Path(args.input_manifest)
    output_path = Path(args.output_manifest)
    bridge_dir = input_path.parent.resolve()
    manifest: dict[str, Any] | None = None
    try:
        manifest = _load_manifest(input_path)
        prepared = _prepare_inputs(manifest, bridge_dir, os.environ)
        max_elems = _positive_int_env(os.environ, "UPMEM_GENERIC_MAX_ELEMS", DEFAULT_MAX_ELEMS)
        if max_elems is None:
            _write_output_manifest(
                output_path,
                backend=args.backend_id,
                status="unsupported",
                manifest=manifest,
                input_manifest_path=input_path,
                output_blob=None,
                validation_metrics={"status": "not_applicable", "reason": "invalid_max_elems"},
                total_time_s=time.perf_counter() - started,
                error="Invalid UPMEM_GENERIC_MAX_ELEMS; expected a positive integer",
                reason="unsupported_shape_for_initial_backend",
                error_type="unsupported_shape_for_initial_backend",
                external_command_executed=True,
                execution_implemented=False,
                metadata_extra=_base_metadata(args.target, False, max_elems=DEFAULT_MAX_ELEMS),
            )
            return 0
        if max(prepared["left"].size, prepared["right"].size, prepared["expected"].size) > max_elems:
            _write_output_manifest(
                output_path,
                backend=args.backend_id,
                status="unsupported",
                manifest=manifest,
                input_manifest_path=input_path,
                output_blob=None,
                validation_metrics={"status": "not_applicable", "reason": "generic_loop_element_cap_exceeded"},
                total_time_s=time.perf_counter() - started,
                error=f"Generic loop tensor size exceeds UPMEM_GENERIC_MAX_ELEMS={max_elems}",
                reason="generic_loop_element_cap_exceeded",
                error_type="generic_loop_element_cap_exceeded",
                external_command_executed=True,
                execution_implemented=False,
                metadata_extra=_base_metadata(args.target, False, max_elems=max_elems),
            )
            return 0
        if args.target == "hardware":
            _write_output_manifest(
                output_path,
                backend=args.backend_id,
                status="not_implemented",
                manifest=manifest,
                input_manifest_path=input_path,
                output_blob=None,
                validation_metrics={"status": "not_applicable", "reason": "hardware_target_disabled"},
                total_time_s=time.perf_counter() - started,
                error=None,
                reason="hardware_target_disabled",
                error_type=None,
                external_command_executed=True,
                execution_implemented=False,
                metadata_extra=_base_metadata(args.target, False, max_elems=max_elems),
            )
            return 0

        missing_tools = tuple(name for name, path in _required_tools(os.environ).items() if path is None)
        if missing_tools:
            _write_output_manifest(
                output_path,
                backend=args.backend_id,
                status="skipped",
                manifest=manifest,
                input_manifest_path=input_path,
                output_blob=None,
                validation_metrics={"status": "not_applicable", "reason": "upmem_sdk_simulator_unavailable"},
                total_time_s=time.perf_counter() - started,
                error=f"Missing required UPMEM SDK simulator tools: {', '.join(missing_tools)}",
                reason="upmem_sdk_simulator_unavailable",
                error_type=None,
                external_command_executed=True,
                execution_implemented=False,
                metadata_extra={**_base_metadata(args.target, False, max_elems=max_elems), "missing_tools": missing_tools},
            )
            return 0

        source_dir = Path(args.source_dir).resolve() if args.source_dir else Path(__file__).resolve().parent / "upmem_sdk_generic_loop"
        runner_work = bridge_dir / "runner_work"
        source_snapshot = runner_work / "src"
        build_dir = runner_work / "build"
        inputs_dir = runner_work / "inputs"
        outputs_dir = runner_work / "outputs"
        _copy_source_tree(source_dir, source_snapshot)
        _copy_source_tree(source_dir, build_dir)
        inputs_dir.mkdir(parents=True, exist_ok=True)
        outputs_dir.mkdir(parents=True, exist_ok=True)

        args_path = inputs_dir / "generic_args.bin"
        left_path = inputs_dir / "left_i8.bin"
        right_path = inputs_dir / "right_i8.bin"
        raw_output_path = outputs_dir / "generic_output_i32.bin"
        args_path.write_bytes(_pack_args(prepared["native_index_metadata"]))
        prepared["left"].astype(prepared["input_dtype"], copy=False).ravel().tofile(left_path)
        prepared["right"].astype(prepared["input_dtype"], copy=False).ravel().tofile(right_path)

        build_started = time.perf_counter()
        build = _run_command(
            ("make", f"MAX_RANK={DEFAULT_MAX_RANK}", f"MAX_ELEMS={max_elems}"),
            cwd=build_dir,
            env={**os.environ, "DPU_BACKEND": "simulator"},
            timeout_seconds=args.timeout_seconds,
        )
        build_time_s = time.perf_counter() - build_started
        if build["status"] != "passed":
            _write_output_manifest(
                output_path,
                backend=args.backend_id,
                status="failed",
                manifest=manifest,
                input_manifest_path=input_path,
                output_blob=None,
                validation_metrics={},
                total_time_s=time.perf_counter() - started,
                error="UPMEM SDK generic loop build failed",
                reason="runner_build_failed",
                error_type="runner_build_failed",
                external_command_executed=True,
                execution_implemented=True,
                metadata_extra={**_base_metadata(args.target, False, max_elems=max_elems), "build_time_s": build_time_s, "build": build},
            )
            return 1

        run_started = time.perf_counter()
        run = _run_command(
            (
                str(build_dir / "bin" / "host"),
                str(build_dir / "bin" / "dpu_generic"),
                str(args_path),
                str(left_path),
                str(right_path),
                str(raw_output_path),
            ),
            cwd=build_dir,
            env={**os.environ, "DPU_BACKEND": "simulator"},
            timeout_seconds=args.timeout_seconds,
        )
        simulator_run_time_s = time.perf_counter() - run_started
        if run["status"] != "passed":
            _write_output_manifest(
                output_path,
                backend=args.backend_id,
                status="failed",
                manifest=manifest,
                input_manifest_path=input_path,
                output_blob=None,
                validation_metrics={},
                total_time_s=time.perf_counter() - started,
                error="UPMEM SDK generic loop runner execution failed",
                reason="runner_execution_failed",
                error_type="runner_execution_failed",
                external_command_executed=True,
                execution_implemented=True,
                metadata_extra={**_base_metadata(args.target, True, max_elems=max_elems), "build_time_s": build_time_s, "simulator_run_time_s": simulator_run_time_s, "build": build, "run": run},
            )
            return 1

        output_shape = tuple(int(dim) for dim in manifest["output_shape"])
        if prepared["operand_mode"] == MODE_FLOAT32_NO_QUANT:
            raw_output = np.fromfile(raw_output_path, dtype="<f4").reshape(output_shape)
            output = raw_output.astype(np.float32, copy=False)
        else:
            raw_output = np.fromfile(raw_output_path, dtype="<i4").reshape(output_shape)
            output = raw_output.astype(np.float64) * float(prepared["output_scale"])
        expected = prepared["expected"]
        validation = _validation_metrics(expected, output)
        output_blob_path = bridge_dir / "outputs" / "upmem_sdk_generic_loop_output.npy"
        output_blob_path.parent.mkdir(parents=True, exist_ok=True)
        write_started = time.perf_counter()
        np.save(output_blob_path, output.astype(expected.dtype, copy=False), allow_pickle=False)
        write_time_s = time.perf_counter() - write_started
        output_blob = _blob_payload(output_blob_path, bridge_dir, output.astype(expected.dtype, copy=False), "generic_loop_dequantized_output")
        status = "upmem_sdk_simulator_generic_loop_executed" if validation["passed"] else "failed"
        reason = "upmem_sdk_simulator_generic_loop_executed" if validation["passed"] else "validation_failed"
        _write_output_manifest(
            output_path,
            backend=args.backend_id,
            status=status,
            manifest=manifest,
            input_manifest_path=input_path,
            output_blob=output_blob,
            validation_metrics=validation,
            compute_time_s=simulator_run_time_s,
            write_time_s=write_time_s,
            total_time_s=time.perf_counter() - started,
            error=None if validation["passed"] else "UPMEM generic loop output did not pass validation",
            reason=reason,
            error_type=None if validation["passed"] else "validation_failed",
            external_command_executed=True,
            execution_implemented=True,
            metadata_extra={
                **_base_metadata(args.target, True, max_elems=max_elems),
                "build_time_s": build_time_s,
                "runner_total_time_s": time.perf_counter() - started,
                "simulator_run_time_s": simulator_run_time_s,
                "kernel_invocation_count": 1,
                "operand_mode": prepared["operand_mode"],
                "quantization_mode": prepared["quantization_mode"],
                "input_dtype_on_dpu": prepared["input_dtype_name"],
                "accumulator_dtype_on_dpu": prepared["accumulator_dtype_on_dpu"],
                "output_dtype_on_dpu": prepared["output_dtype_on_dpu"],
                "unquantized_mode_kind": prepared["unquantized_mode_kind"],
                "scaling_applied": prepared["scaling_applied"],
                "actual_h2d_bytes": prepared["actual_h2d_bytes"],
                "actual_d2h_bytes": prepared["actual_d2h_bytes"],
                "full_precision_h2d_bytes_model": prepared["full_precision_h2d_bytes_model"],
                "full_precision_d2h_bytes_model": prepared["full_precision_d2h_bytes_model"],
                "build": build,
                "run": run,
                "runner_work": {"source": "runner_work/src", "build": "runner_work/build", "inputs": "runner_work/inputs", "outputs": "runner_work/outputs"},
            },
        )
        return 0 if validation["passed"] else 1
    except Exception as exc:
        reason = "input_manifest_invalid" if manifest is None else "operand_blob_invalid"
        _write_output_manifest(
            output_path,
            backend=args.backend_id,
            status="failed",
            manifest=manifest,
            input_manifest_path=input_path,
            output_blob=None,
            validation_metrics={},
            total_time_s=time.perf_counter() - started,
            error=str(exc),
            reason=reason,
            error_type=reason,
            external_command_executed=True,
            execution_implemented=False,
            metadata_extra=_base_metadata(args.target, False),
        )
        return 1


def _load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != GENERIC_BRIDGE_SCHEMA_VERSION:
        raise ValueError("unsupported generic bridge schema_version")
    if payload.get("bridge_id") != GENERIC_BRIDGE_ID:
        raise ValueError("unsupported generic bridge_id")
    if payload.get("manifest_kind") != "generic_contraction_bridge_input":
        raise ValueError("input manifest_kind must be generic_contraction_bridge_input")
    return payload


def _prepare_inputs(manifest: dict[str, Any], bridge_dir: Path, env: Mapping[str, str]) -> dict[str, Any]:
    left_meta = dict(manifest["operands"]["left"])
    right_meta = dict(manifest["operands"]["right"])
    expected_meta = dict(manifest["expected_quantized_reference_output"])
    left = np.load(_resolve_manifest_path(bridge_dir, str(left_meta["relative_path"])), allow_pickle=False)
    right = np.load(_resolve_manifest_path(bridge_dir, str(right_meta["relative_path"])), allow_pickle=False)
    expected = np.load(_resolve_manifest_path(bridge_dir, str(expected_meta["relative_path"])), allow_pickle=False)
    _validate_blob(left, left_meta, "left")
    _validate_blob(right, right_meta, "right")
    _validate_blob(expected, expected_meta, "expected_quantized_reference_output")
    metadata = dict(manifest.get("metadata") or {})
    native = dict(manifest["native_index_metadata"])
    operand_mode = str(metadata.get("operand_mode") or native.get("operand_mode") or MODE_INT8_SCALED)
    if operand_mode not in NATIVE_MODE_IDS:
        raise ValueError(f"unsupported generic operand mode: {operand_mode}")
    if operand_mode == MODE_FLOAT32_NO_QUANT:
        if str(left.dtype) != "float32" or str(right.dtype) != "float32":
            raise ValueError("float32_no_quant generic loop backend requires float32 operands")
        input_dtype = np.dtype("<f4")
        input_dtype_name = "float32"
        accumulator_dtype = "float32"
        output_dtype = "float32"
        scaling_applied = False
        unquantized_mode_kind = MODE_FLOAT32_NO_QUANT
    else:
        if str(left.dtype) != "int8" or str(right.dtype) != "int8":
            raise ValueError("int8_scaled generic loop backend requires int8 operands")
        input_dtype = np.dtype("int8")
        input_dtype_name = "int8"
        accumulator_dtype = "int32"
        output_dtype = "int32"
        scaling_applied = True
        unquantized_mode_kind = None
    args_bytes = _pack_args(native)
    actual_h2d_bytes = _align8(left.size * input_dtype.itemsize) + _align8(right.size * input_dtype.itemsize) + len(args_bytes)
    actual_d2h_bytes = _align8(expected.size * (4 if operand_mode == MODE_FLOAT32_NO_QUANT else 4))
    return {
        "left": left,
        "right": right,
        "expected": expected,
        "native_index_metadata": native,
        "output_scale": float(manifest["dequantization"]["output_scale"]),
        "operand_mode": operand_mode,
        "quantization_mode": str(metadata.get("quantization_mode") or "per_task_input_quantize"),
        "input_dtype": input_dtype,
        "input_dtype_name": input_dtype_name,
        "accumulator_dtype_on_dpu": accumulator_dtype,
        "output_dtype_on_dpu": output_dtype,
        "unquantized_mode_kind": unquantized_mode_kind,
        "scaling_applied": scaling_applied,
        "actual_h2d_bytes": int(actual_h2d_bytes),
        "actual_d2h_bytes": int(actual_d2h_bytes),
        "full_precision_h2d_bytes_model": int(metadata.get("full_precision_h2d_bytes_model") or actual_h2d_bytes),
        "full_precision_d2h_bytes_model": int(metadata.get("full_precision_d2h_bytes_model") or actual_d2h_bytes),
    }


def _pack_args(native: dict[str, Any]) -> bytes:
    max_rank = DEFAULT_MAX_RANK

    def u32_array(name: str) -> list[int]:
        values = [int(value) for value in native.get(name, ())]
        return (values + [0] * max_rank)[:max_rank]

    def i32_array(name: str) -> list[int]:
        values = [int(value) for value in native.get(name, ())]
        return (values + [-1] * max_rank)[:max_rank]

    fields: list[int] = [
        int(native["left_rank"]),
        int(native["right_rank"]),
        int(native["output_rank"]),
        int(native["contracted_rank"]),
        _product(tuple(int(dim) for dim in native["left_shape"])),
        _product(tuple(int(dim) for dim in native["right_shape"])),
        int(native["output_element_count"]),
        int(native["contracted_combination_count"]),
        int(NATIVE_MODE_IDS.get(str(native.get("operand_mode") or MODE_INT8_SCALED), 0)),
    ]
    for key in ("left_shape", "right_shape", "output_shape", "contracted_dims", "left_strides", "right_strides", "output_strides"):
        fields.extend(u32_array(key))
    signed_fields: list[int] = []
    for key in ("output_to_left_axes", "output_to_right_axes", "contracted_to_left_axes", "contracted_to_right_axes"):
        signed_fields.extend(i32_array(key))
    return struct.pack("<" + "I" * len(fields) + "i" * len(signed_fields), *(fields + signed_fields))


def _write_output_manifest(
    path: Path,
    *,
    backend: str,
    status: str,
    manifest: dict[str, Any] | None,
    input_manifest_path: Path,
    output_blob: dict[str, Any] | None,
    validation_metrics: dict[str, Any],
    total_time_s: float,
    error: str | None,
    reason: str,
    error_type: str | None,
    external_command_executed: bool,
    execution_implemented: bool,
    metadata_extra: dict[str, Any] | None = None,
    compute_time_s: float = 0.0,
    write_time_s: float = 0.0,
) -> None:
    payload = {
        "schema_version": GENERIC_BRIDGE_SCHEMA_VERSION,
        "bridge_id": GENERIC_BRIDGE_ID,
        "manifest_kind": "generic_contraction_bridge_output",
        "backend": backend,
        "status": status,
        "input_manifest": input_manifest_path.name,
        "route_id": str((manifest or {}).get("route_id", "")),
        "task_id": str((manifest or {}).get("task_id", "")),
        "output_blob": output_blob,
        "validation_metrics": validation_metrics,
        "compute_time_s": float(compute_time_s),
        "write_time_s": float(write_time_s),
        "total_time_s": float(total_time_s),
        "external_command_executed": bool(external_command_executed),
        "execution_implemented": bool(execution_implemented),
        "error": error,
        "metadata": {
            "reason": reason,
            "error_type": error_type,
            "backend_family": "upmem_sdk",
            "kernel_family": KERNEL_FAMILY,
            "target": "simulator" if (metadata_extra or {}).get("target") != "hardware" else "hardware",
            "simplepim_api_used": False,
            "native_sdk_control_path": True,
            "upmem_dpu_program_executed": status == "upmem_sdk_simulator_generic_loop_executed",
            "simulator_kernel_executed": status == "upmem_sdk_simulator_generic_loop_executed",
            "hardware_kernel_executed": False,
            **(metadata_extra or {}),
        },
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _validation_metrics(expected: np.ndarray, actual: np.ndarray) -> dict[str, Any]:
    diff = actual - expected
    max_abs_error = float(np.max(np.abs(diff))) if diff.size else 0.0
    l2_error = float(np.linalg.norm(diff.ravel()))
    expected_norm = float(np.linalg.norm(expected.ravel()))
    relative_l2_error = 0.0 if expected_norm == 0.0 and l2_error == 0.0 else (None if expected_norm == 0.0 else l2_error / expected_norm)
    max_abs_tolerance = 1.0e-5 if str(np.asarray(expected).dtype) == "float32" else 1.0e-12
    return {
        "reference_kind": "expected_reference_output_vs_upmem_sdk_generic_loop",
        "max_abs_error": max_abs_error,
        "l2_error": l2_error,
        "relative_l2_error": relative_l2_error,
        "expected_norm": expected_norm,
        "output_norm": float(np.linalg.norm(actual.ravel())),
        "max_abs_tolerance": max_abs_tolerance,
        "tolerance_rationale": "JSON scale round-trip can introduce sub-ulp dequantization differences; native integer output is still compared to the quantized int32 accumulation reference.",
        "passed": max_abs_error <= max_abs_tolerance,
    }


def _required_tools(env: Mapping[str, str]) -> dict[str, str | None]:
    return {
        "make": shutil.which("make", path=env.get("PATH")),
        "dpu-upmem-dpurte-clang": shutil.which("dpu-upmem-dpurte-clang", path=env.get("PATH")),
        "dpu-pkg-config": shutil.which("dpu-pkg-config", path=env.get("PATH")),
    }


def _copy_source_tree(source: Path, dest: Path) -> None:
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(source, dest)


def _run_command(command: tuple[str, ...], *, cwd: Path, env: Mapping[str, str], timeout_seconds: float) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        completed = subprocess.run(command, cwd=cwd, env=dict(env), capture_output=True, text=True, check=False, timeout=timeout_seconds)
        status = "passed" if completed.returncode == 0 else "failed"
        return {
            "command": command,
            "cwd": ".",
            "status": status,
            "returncode": int(completed.returncode),
            "elapsed_s": float(time.perf_counter() - started),
            "stdout_snippet": _bounded_snippet(completed.stdout),
            "stderr_snippet": _bounded_snippet(completed.stderr),
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "command": command,
            "cwd": ".",
            "status": "timeout",
            "returncode": None,
            "elapsed_s": float(time.perf_counter() - started),
            "stdout_snippet": _bounded_snippet(exc.stdout or ""),
            "stderr_snippet": _bounded_snippet(exc.stderr or ""),
        }


def _resolve_manifest_path(bridge_dir: Path, relative_path: str) -> Path:
    path = Path(relative_path)
    if path.is_absolute():
        raise ValueError(f"manifest path must be relative: {relative_path}")
    resolved = (bridge_dir / path).resolve()
    try:
        resolved.relative_to(bridge_dir.resolve())
    except ValueError as exc:
        raise ValueError(f"manifest path escapes bridge directory: {relative_path}") from exc
    return resolved


def _validate_blob(array: np.ndarray, metadata: dict[str, Any], role: str) -> None:
    if tuple(int(dim) for dim in array.shape) != tuple(int(dim) for dim in metadata["shape"]):
        raise ValueError(f"{role} blob shape mismatch")
    if str(array.dtype) != str(metadata["dtype"]):
        raise ValueError(f"{role} blob dtype mismatch")


def _blob_payload(path: Path, bridge_dir: Path, array: np.ndarray, representation: str) -> dict[str, Any]:
    return {
        "relative_path": path.resolve().relative_to(bridge_dir.resolve()).as_posix(),
        "dtype": str(array.dtype),
        "shape": [int(dim) for dim in array.shape],
        "representation": representation,
        "nbytes": int(array.nbytes),
        "role": "generic_loop_output",
    }


def _base_metadata(target: str, kernel_executed: bool, *, max_elems: int | None = None) -> dict[str, Any]:
    return {
        "target": target,
        "backend_family": "upmem_sdk",
        "kernel_family": KERNEL_FAMILY,
        "operand_mode": None,
        "quantization_mode": None,
        "simplepim_api_used": False,
        "native_sdk_control_path": True,
        "upmem_dpu_program_executed": kernel_executed,
        "simulator_kernel_executed": kernel_executed and target == "simulator",
        "hardware_kernel_executed": False,
        "max_elems": max_elems,
    }


def _positive_int_env(env: Mapping[str, str], name: str, default: int) -> int | None:
    raw = env.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        return None
    return value if value > 0 else None


def _product(shape: tuple[int, ...]) -> int:
    value = 1
    for dim in shape:
        value *= int(dim)
    return int(value)


def _align8(bytes_count: int) -> int:
    return int((int(bytes_count) + 7) & ~7)


def _bounded_snippet(text: str, limit: int = SNIPPET_LIMIT) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "...<truncated>"


if __name__ == "__main__":
    raise SystemExit(main())
