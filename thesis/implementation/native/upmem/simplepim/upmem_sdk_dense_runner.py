from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np


DENSE_BRIDGE_SCHEMA_VERSION = "dense_bridge_v1"
DENSE_BRIDGE_ID = "upmem_dense_bridge_v1"
BACKEND_ID = "upmem_sdk_simulator_dense"
DEFAULT_MAX_DIM = 16
DEFAULT_L2_NATIVE_MAX_DIM = 512
DEFAULT_L2_MAX_HOST_BLOB_BYTES = 16 * 1024 * 1024
DEFAULT_L2_EFFECTIVE_WRAM_BYTES = 60 * 1024
DEFAULT_L2_ALIGNMENT_RESERVE_BYTES = 2048
DEFAULT_TIMEOUT_SECONDS = 30.0
SNIPPET_LIMIT = 2000
EXECUTION_CLASS_L1 = "L1_WRAM"
EXECUTION_CLASS_L2 = "L2_SINGLE_DPU_MRAM"
KERNEL_STRATEGY_L1 = "l1_padded_direct_v1"
KERNEL_STRATEGY_L2 = "l2_single_dpu_mram_wram_tiled_v1"
L2_TILE_CANDIDATES = (
    (64, 64, 64),
    (64, 32, 64),
    (32, 64, 64),
    (32, 32, 64),
    (32, 32, 32),
    (16, 32, 64),
    (32, 16, 64),
    (16, 16, 64),
    (16, 16, 32),
    (8, 16, 32),
    (16, 8, 32),
    (8, 8, 32),
)


def main() -> int:
    parser = argparse.ArgumentParser(description="UPMEM SDK simulator dense bridge runner")
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
        max_dim = _max_dim_from_env(os.environ)
        if max_dim is None:
            _write_output_manifest(
                output_path,
                backend=args.backend_id,
                status="unsupported",
                manifest=manifest,
                input_manifest_path=input_path,
                output_blob=None,
                validation_metrics={},
                total_time_s=time.perf_counter() - started,
                error="Invalid UPMEM_DENSE_SIM_MAX_DIM; expected a positive integer",
                reason="unsupported_shape_for_initial_backend",
                error_type="unsupported_shape_for_initial_backend",
                external_command_executed=True,
                execution_implemented=False,
                metadata_extra=_base_metadata(args.target, False, prepared if "prepared" in locals() else None),
            )
            return 0
        shape_reason = _shape_limit_reason(prepared, max_dim)
        if shape_reason is not None:
            _write_output_manifest(
                output_path,
                backend=args.backend_id,
                status="unsupported",
                manifest=manifest,
                input_manifest_path=input_path,
                output_blob=None,
                validation_metrics={},
                total_time_s=time.perf_counter() - started,
                error=shape_reason,
                reason="unsupported_shape_for_initial_backend",
                error_type="unsupported_shape_for_initial_backend",
                external_command_executed=True,
                execution_implemented=False,
                metadata_extra={**_base_metadata(args.target, False, prepared), "max_dim": max_dim},
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
                metadata_extra={**_base_metadata(args.target, False, prepared), "max_dim": max_dim},
            )
            return 0

        tools = _required_tools(os.environ)
        missing_tools = tuple(name for name, path in tools.items() if path is None)
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
                metadata_extra={**_base_metadata(args.target, False, prepared), "missing_tools": missing_tools, "max_dim": max_dim},
            )
            return 0

        source_dir = Path(args.source_dir).resolve() if args.source_dir else Path(__file__).resolve().parent / "upmem_sdk_dense"
        runner_work = bridge_dir / "runner_work"
        source_snapshot = runner_work / "src"
        build_dir = runner_work / "build"
        inputs_dir = runner_work / "inputs"
        raw_outputs_dir = runner_work / "outputs"
        try:
            _copy_source_tree(source_dir, source_snapshot)
            _copy_source_tree(source_dir, build_dir)
            inputs_dir.mkdir(parents=True, exist_ok=True)
            raw_outputs_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            _write_output_manifest(
                output_path,
                backend=args.backend_id,
                status="failed",
                manifest=manifest,
                input_manifest_path=input_path,
                output_blob=None,
                validation_metrics={},
                total_time_s=time.perf_counter() - started,
                error=f"Failed to prepare UPMEM SDK dense runner build workspace: {exc}",
                reason="runner_build_failed",
                error_type="runner_build_failed",
                external_command_executed=True,
                execution_implemented=True,
                metadata_extra={**_base_metadata(args.target, False, prepared), "max_dim": max_dim},
            )
            return 1

        build_started = time.perf_counter()
        l2_native_max_dim = int(prepared.get("l2_native_max_dim", DEFAULT_L2_NATIVE_MAX_DIM))
        build = _run_command(
            ("make", f"MAX_DIM={max_dim}", f"L2_MAX_DIM={l2_native_max_dim}"),
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
                error="UPMEM SDK dense runner build failed",
                reason="runner_build_failed",
                error_type="runner_build_failed",
                external_command_executed=True,
                execution_implemented=True,
                metadata_extra={
                    **_base_metadata(args.target, False, prepared),
                    "max_dim": max_dim,
                    "l2_native_max_dim": l2_native_max_dim,
                    "build_time_s": build_time_s,
                    "build": build,
                },
            )
            return 1

        run_started = time.perf_counter()
        kernel_outputs: dict[str, np.ndarray] = {}
        command_records: list[dict[str, Any]] = []
        for item in prepared["components"]:
            component_output, record = _run_component(
                item,
                build_dir=build_dir,
                inputs_dir=inputs_dir,
                outputs_dir=raw_outputs_dir,
                max_dim=max_dim,
                timeout_seconds=args.timeout_seconds,
                execution_class=str(prepared["execution_class"]),
            )
            command_records.append(record)
            if record["status"] != "passed":
                _write_output_manifest(
                    output_path,
                    backend=args.backend_id,
                    status="failed",
                    manifest=manifest,
                    input_manifest_path=input_path,
                    output_blob=None,
                    validation_metrics={},
                    total_time_s=time.perf_counter() - started,
                    error="UPMEM SDK dense runner execution failed",
                    reason="runner_execution_failed",
                    error_type="runner_execution_failed",
                    external_command_executed=True,
                    execution_implemented=True,
                    metadata_extra={
                        **_base_metadata(args.target, True, prepared),
                        "max_dim": max_dim,
                        "l2_native_max_dim": l2_native_max_dim,
                        "build_time_s": build_time_s,
                        "simulator_run_time_s": time.perf_counter() - run_started,
                        "build": build,
                        "commands": command_records,
                    },
                )
                return 1
            kernel_outputs[item["name"]] = component_output
        simulator_run_time_s = time.perf_counter() - run_started

        matrix_output = _combine_outputs(prepared, kernel_outputs)
        output = _restore_output_order(matrix_output, manifest)
        expected = prepared["expected"]
        validation = _validation_metrics(expected, output)
        output_blob_path = bridge_dir / "outputs" / "upmem_sdk_simulator_output.npy"
        output_blob_path.parent.mkdir(parents=True, exist_ok=True)
        output_to_write = output.astype(expected.dtype, copy=False)
        write_started = time.perf_counter()
        np.save(output_blob_path, output_to_write, allow_pickle=False)
        write_time_s = time.perf_counter() - write_started
        output_blob = _blob_payload(output_blob_path, bridge_dir, output_to_write, manifest, prepared["mode"])

        status = "upmem_sdk_simulator_executed" if validation["passed"] else "failed"
        reason = None if validation["passed"] else "validation_failed"
        error_type = None if validation["passed"] else "validation_failed"
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
            error=None if validation["passed"] else "UPMEM simulator output did not pass validation",
            reason=reason or "upmem_sdk_simulator_executed",
            error_type=error_type,
            external_command_executed=True,
            execution_implemented=True,
            metadata_extra={
                **_base_metadata(args.target, True, prepared),
                "max_dim": max_dim,
                "l2_native_max_dim": l2_native_max_dim,
                "kernel_invocation_count": len(prepared["components"]),
                "complex_execution_mode": prepared["mode"],
                "build_time_s": build_time_s,
                "runner_total_time_s": time.perf_counter() - started,
                "simulator_run_time_s": simulator_run_time_s,
                "build": build,
                "commands": command_records,
                "runner_work": {
                    "source": "runner_work/src",
                    "build": "runner_work/build",
                    "inputs": "runner_work/inputs",
                    "outputs": "runner_work/outputs",
                },
            },
        )
        return 0 if validation["passed"] else 1
    except UnsupportedInput as exc:
        _write_output_manifest(
            output_path,
            backend=args.backend_id,
            status="unsupported",
            manifest=manifest,
            input_manifest_path=input_path,
            output_blob=None,
            validation_metrics={},
            total_time_s=time.perf_counter() - started,
            error=str(exc),
            reason=exc.reason,
            error_type=exc.reason,
            external_command_executed=True,
            execution_implemented=False,
            metadata_extra=_base_metadata(args.target, False),
        )
        return 0
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
    if payload.get("schema_version") != DENSE_BRIDGE_SCHEMA_VERSION:
        raise ValueError("unsupported dense bridge schema_version")
    if payload.get("bridge_id") != DENSE_BRIDGE_ID:
        raise ValueError("unsupported dense bridge_id")
    if payload.get("manifest_kind") != "dense_bridge_input":
        raise ValueError("input manifest_kind must be dense_bridge_input")
    return payload


def _prepare_inputs(manifest: dict[str, Any], bridge_dir: Path, env: Mapping[str, str]) -> dict[str, Any]:
    tile_plan = manifest.get("tile_plan") or {}
    requires_tiling = bool(tile_plan.get("requires_tiling"))

    left_meta = dict(manifest["operands"]["left"])
    right_meta = dict(manifest["operands"]["right"])
    expected_meta = dict(manifest["expected_output"])
    left = np.load(_resolve_manifest_path(bridge_dir, str(left_meta["relative_path"])), allow_pickle=False)
    right = np.load(_resolve_manifest_path(bridge_dir, str(right_meta["relative_path"])), allow_pickle=False)
    expected = np.load(_resolve_manifest_path(bridge_dir, str(expected_meta["relative_path"])), allow_pickle=False)
    _validate_blob(left, left_meta, "left")
    _validate_blob(right, right_meta, "right")
    _validate_blob(expected, expected_meta, "expected_output")

    left_deq = dict(manifest["dequantization"]["left"])
    right_deq = dict(manifest["dequantization"]["right"])
    if left_deq.get("route_dtype") != "int8" or right_deq.get("route_dtype") != "int8":
        raise UnsupportedInput("unsupported_dtype", "UPMEM SDK simulator dense backend supports int8 operands only")

    m = int(manifest["gemm_m"])
    k = int(manifest["gemm_k"])
    n = int(manifest["gemm_n"])
    left_rep = str(left_deq["representation"])
    right_rep = str(right_deq["representation"])
    if left_rep == "real" and right_rep == "real":
        if tuple(left.shape) != (m, k) or tuple(right.shape) != (k, n):
            raise UnsupportedInput("unsupported_shape", "Real operand shapes do not match GEMM dimensions")
        if requires_tiling:
            l2_plan = _plan_l2_tiled_execution(m, k, n, env)
            if not l2_plan["supported"]:
                raise UnsupportedInput(str(l2_plan["reason"] or "unsupported_l2_tile_plan"), "L2 tiled simulator backend does not support this GEMM shape")
            execution_class = EXECUTION_CLASS_L2
            kernel_strategy = KERNEL_STRATEGY_L2
        else:
            l2_plan = None
            execution_class = EXECUTION_CLASS_L1
            kernel_strategy = KERNEL_STRATEGY_L1
        components = (
            {
                "name": "real",
                "left": left.astype(np.int8, copy=False),
                "right": right.astype(np.int8, copy=False),
                "left_scale": float(left_deq["scale"]),
                "right_scale": float(right_deq["scale"]),
                "l2_plan": l2_plan,
            },
        )
        return {
            "mode": "real_single_gemm",
            "m": m,
            "k": k,
            "n": n,
            "execution_class": execution_class,
            "kernel_strategy": kernel_strategy,
            "l2_plan": l2_plan,
            "l2_native_max_dim": _l2_native_max_dim_from_env(env),
            "components": components,
            "expected": expected,
        }
    if left_rep == "split_complex_real_imag" and right_rep == "split_complex_real_imag":
        if requires_tiling:
            raise UnsupportedInput("complex_l2_not_implemented", "L2 tiled simulator backend supports real-valued operands only")
        if tuple(left.shape) != (m, k, 2) or tuple(right.shape) != (k, n, 2):
            raise UnsupportedInput("unsupported_complex_layout", "Split-complex operand shapes do not match GEMM dimensions")
        left_scale = float(left_deq["scale"])
        right_scale = float(right_deq["scale"])
        components = (
            {"name": "ar_br", "left": left[..., 0], "right": right[..., 0], "left_scale": left_scale, "right_scale": right_scale},
            {"name": "ai_bi", "left": left[..., 1], "right": right[..., 1], "left_scale": left_scale, "right_scale": right_scale},
            {"name": "ar_bi", "left": left[..., 0], "right": right[..., 1], "left_scale": left_scale, "right_scale": right_scale},
            {"name": "ai_br", "left": left[..., 1], "right": right[..., 0], "left_scale": left_scale, "right_scale": right_scale},
        )
        return {
            "mode": "split_complex_four_gemm",
            "m": m,
            "k": k,
            "n": n,
            "execution_class": EXECUTION_CLASS_L1,
            "kernel_strategy": KERNEL_STRATEGY_L1,
            "l2_plan": None,
            "l2_native_max_dim": _l2_native_max_dim_from_env(env),
            "components": components,
            "expected": expected,
        }
    raise UnsupportedInput("unsupported_complex_layout", f"Unsupported operand representations: {left_rep}, {right_rep}")


def _shape_limit_reason(prepared: dict[str, Any], max_dim: int) -> str | None:
    dims = (int(prepared["m"]), int(prepared["k"]), int(prepared["n"]))
    if any(dim <= 0 for dim in dims):
        return f"Invalid GEMM dimensions: {dims}"
    if prepared.get("execution_class") == EXECUTION_CLASS_L2:
        return None
    if any(dim > max_dim for dim in dims):
        return f"GEMM dimensions {dims} exceed initial backend max dim {max_dim}"
    return None


def _max_dim_from_env(env: Mapping[str, str]) -> int | None:
    raw = env.get("UPMEM_DENSE_SIM_MAX_DIM")
    if raw is None or not raw.strip():
        return DEFAULT_MAX_DIM
    try:
        value = int(raw)
    except ValueError:
        return None
    if value <= 0:
        return None
    return value


def _l2_native_max_dim_from_env(env: Mapping[str, str]) -> int:
    return _positive_int_env(env, "UPMEM_DENSE_L2_NATIVE_MAX_DIM", DEFAULT_L2_NATIVE_MAX_DIM)


def _l2_max_host_blob_bytes_from_env(env: Mapping[str, str]) -> int:
    return _positive_int_env(env, "UPMEM_DENSE_L2_MAX_HOST_BLOB_BYTES", DEFAULT_L2_MAX_HOST_BLOB_BYTES)


def _positive_int_env(env: Mapping[str, str], key: str, default: int) -> int:
    raw = env.get(key)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def _plan_l2_tiled_execution(m: int, k: int, n: int, env: Mapping[str, str]) -> dict[str, Any]:
    m = int(m)
    k = int(k)
    n = int(n)
    native_max_dim = _l2_native_max_dim_from_env(env)
    max_host_blob_bytes = _l2_max_host_blob_bytes_from_env(env)
    a_bytes = m * k
    b_bytes = k * n
    c_bytes = m * n * 4
    c_accumulator_bytes = m * n * 4
    total_mram_bytes = a_bytes + b_bytes + c_bytes
    conservative_full_task_bytes = total_mram_bytes + c_accumulator_bytes
    host_blob_bytes = a_bytes + b_bytes + (m * n * 8)
    base = {
        "execution_class": EXECUTION_CLASS_L2,
        "kernel_strategy": KERNEL_STRATEGY_L2,
        "gemm_m": m,
        "gemm_k": k,
        "gemm_n": n,
        "effective_wram_bytes": DEFAULT_L2_EFFECTIVE_WRAM_BYTES,
        "mram_bytes_a": a_bytes,
        "mram_bytes_b": b_bytes,
        "mram_bytes_c": c_bytes,
        "total_mram_bytes": total_mram_bytes,
        "conservative_full_task_bytes": conservative_full_task_bytes,
        "max_l2_host_blob_bytes": max_host_blob_bytes,
        "host_blob_bytes": host_blob_bytes,
        "native_max_dim": native_max_dim,
        "mram_resident_operands": True,
        "wram_tiled": True,
    }
    if min(m, k, n) <= 0 or max(m, k, n) > native_max_dim or any(dim % 8 != 0 for dim in (m, k, n)):
        return {**base, "supported": False, "reason": "unsupported_l2_native_shape_limit"}
    if conservative_full_task_bytes <= DEFAULT_L2_EFFECTIVE_WRAM_BYTES:
        return {**base, "supported": False, "reason": "not_l2_wram_resident"}
    if host_blob_bytes > max_host_blob_bytes:
        return {**base, "supported": False, "reason": "unsupported_l2_blob_size"}
    for candidate_m, candidate_k, candidate_n in L2_TILE_CANDIDATES:
        tile_m = min(candidate_m, m)
        tile_k = min(candidate_k, k)
        tile_n = min(candidate_n, n)
        a_tile = tile_m * tile_k
        b_tile = tile_k * tile_n
        accumulator_tile = tile_m * tile_n * 4
        scratch = 0
        wram_bytes = a_tile + b_tile + accumulator_tile + scratch + DEFAULT_L2_ALIGNMENT_RESERVE_BYTES
        if wram_bytes > DEFAULT_L2_EFFECTIVE_WRAM_BYTES:
            continue
        output_tile_count = int(np.ceil(m / tile_m) * np.ceil(n / tile_n))
        k_tile_count = int(np.ceil(k / tile_k))
        return {
            **base,
            "supported": True,
            "reason": None,
            "tile_m": tile_m,
            "tile_k": tile_k,
            "tile_n": tile_n,
            "output_tile_count": output_tile_count,
            "k_tile_count": k_tile_count,
            "total_tile_steps": output_tile_count * k_tile_count,
            "input_a_tile_bytes": a_tile,
            "input_b_tile_bytes": b_tile,
            "accumulator_tile_bytes": accumulator_tile,
            "local_output_scratch_bytes": scratch,
            "alignment_padding_bytes": DEFAULT_L2_ALIGNMENT_RESERVE_BYTES,
            "estimated_wram_bytes_per_tile": wram_bytes,
        }
    return {**base, "supported": False, "reason": "unsupported_l2_tile_plan"}


def _required_tools(env: Mapping[str, str]) -> dict[str, str | None]:
    return {
        "make": shutil.which("make"),
        "dpu-upmem-dpurte-clang": _find_tool("dpu-upmem-dpurte-clang", env),
        "dpu-pkg-config": _find_tool("dpu-pkg-config", env),
    }


def _find_tool(name: str, env: Mapping[str, str]) -> str | None:
    upmem_home = env.get("UPMEM_HOME")
    if upmem_home:
        candidate = Path(upmem_home).expanduser() / "bin" / name
        if candidate.exists():
            return str(candidate)
    return shutil.which(name)


def _copy_source_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".git"))


def _run_component(
    item: dict[str, Any],
    *,
    build_dir: Path,
    inputs_dir: Path,
    outputs_dir: Path,
    max_dim: int,
    timeout_seconds: float,
    execution_class: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    left_path = inputs_dir / f"{item['name']}_left_i8.bin"
    right_path = inputs_dir / f"{item['name']}_right_i8.bin"
    out_path = outputs_dir / f"{item['name']}_out_i32.bin"
    m, k = item["left"].shape
    _, n = item["right"].shape
    if execution_class == EXECUTION_CLASS_L2:
        plan = dict(item.get("l2_plan") or {})
        _write_exact_i8(left_path, item["left"])
        _write_exact_i8(right_path, item["right"])
        command = (
            "./bin/host",
            "bin/dpu_dense",
            "l2",
            str(int(m)),
            str(int(k)),
            str(int(n)),
            str(int(plan["tile_m"])),
            str(int(plan["tile_k"])),
            str(int(plan["tile_n"])),
            _relative_path(left_path, build_dir),
            _relative_path(right_path, build_dir),
            _relative_path(out_path, build_dir),
        )
        output_count = int(m) * int(n)
        output_shape = (int(m), int(n))
    else:
        _write_padded_i8(left_path, item["left"], max_dim)
        _write_padded_i8(right_path, item["right"], max_dim)
        command = (
            "./bin/host",
            "bin/dpu_dense",
            str(int(m)),
            str(int(k)),
            str(int(n)),
            _relative_path(left_path, build_dir),
            _relative_path(right_path, build_dir),
            _relative_path(out_path, build_dir),
        )
        output_count = max_dim * max_dim
        output_shape = (max_dim, max_dim)
    record = _run_command(command, cwd=build_dir, env={**os.environ, "DPU_BACKEND": "simulator"}, timeout_seconds=timeout_seconds)
    if record["status"] != "passed":
        return np.zeros((int(m), int(n)), dtype=np.int32), record
    output = np.fromfile(out_path, dtype="<i4", count=output_count).reshape(output_shape)[: int(m), : int(n)]
    return output.astype(np.int32, copy=False), record


def _write_exact_i8(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.asarray(array, dtype=np.int8).astype(np.int8, copy=False).tofile(path)


def _write_padded_i8(path: Path, array: np.ndarray, max_dim: int) -> None:
    padded = np.zeros((max_dim, max_dim), dtype=np.int8)
    rows, cols = array.shape
    padded[:rows, :cols] = np.asarray(array, dtype=np.int8)
    path.parent.mkdir(parents=True, exist_ok=True)
    padded.astype(np.int8, copy=False).tofile(path)


def _combine_outputs(prepared: dict[str, Any], outputs: dict[str, np.ndarray]) -> np.ndarray:
    components = {item["name"]: item for item in prepared["components"]}
    if prepared["mode"] == "real_single_gemm":
        item = components["real"]
        return outputs["real"].astype(np.float64) * float(item["left_scale"]) * float(item["right_scale"])
    ar_br = outputs["ar_br"].astype(np.float64) * components["ar_br"]["left_scale"] * components["ar_br"]["right_scale"]
    ai_bi = outputs["ai_bi"].astype(np.float64) * components["ai_bi"]["left_scale"] * components["ai_bi"]["right_scale"]
    ar_bi = outputs["ar_bi"].astype(np.float64) * components["ar_bi"]["left_scale"] * components["ar_bi"]["right_scale"]
    ai_br = outputs["ai_br"].astype(np.float64) * components["ai_br"]["left_scale"] * components["ai_br"]["right_scale"]
    return (ar_br - ai_bi) + 1j * (ar_bi + ai_br)


def _restore_output_order(matrix_output: np.ndarray, manifest: dict[str, Any]) -> np.ndarray:
    left_free = tuple(int(label) for label in manifest["left_free_labels"])
    right_free = tuple(int(label) for label in manifest["right_free_labels"])
    output_labels = tuple(int(label) for label in manifest["output_labels"])
    output_shape = tuple(int(dim) for dim in manifest["output_shape"])
    gemm_labels = left_free + right_free
    gemm_shape = tuple(int(output_shape[output_labels.index(label)]) for label in gemm_labels)
    tensor_output = np.asarray(matrix_output).reshape(gemm_shape)
    if gemm_labels == output_labels:
        return tensor_output
    axes = tuple(gemm_labels.index(label) for label in output_labels)
    return np.transpose(tensor_output, axes)


def _validation_metrics(expected: np.ndarray, output: np.ndarray) -> dict[str, Any]:
    diff = output - expected
    max_abs_error = float(np.max(np.abs(diff))) if diff.size else 0.0
    l2_error = float(np.linalg.norm(diff.ravel())) if diff.size else 0.0
    expected_norm = float(np.linalg.norm(expected.ravel()))
    relative_l2_error = 0.0 if expected_norm == 0.0 and l2_error == 0.0 else (None if expected_norm == 0.0 else l2_error / expected_norm)
    policy = {
        "reference": "expected_dequantized_output.npy",
        "comparison": "np.allclose",
        "atol": 1.0e-9,
        "rtol": 1.0e-9,
    }
    passed = bool(np.allclose(expected, output, atol=policy["atol"], rtol=policy["rtol"]))
    return {
        "reference_kind": "expected_dequantized_output_vs_upmem_sdk_simulator_dense",
        "max_abs_error": max_abs_error,
        "l2_error": l2_error,
        "relative_l2_error": relative_l2_error,
        "expected_norm": expected_norm,
        "output_norm": float(np.linalg.norm(output.ravel())),
        "passed": passed,
        "policy": policy,
    }


def _run_command(
    command: tuple[str, ...],
    *,
    cwd: Path,
    env: Mapping[str, str],
    timeout_seconds: float,
) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            list(command),
            cwd=cwd,
            env=dict(env),
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_seconds,
        )
        return {
            "command": command,
            "cwd": _relative_path(cwd, cwd.parent.parent),
            "return_code": int(completed.returncode),
            "status": "passed" if completed.returncode == 0 else "failed",
            "timed_out": False,
            "stdout_snippet": _bounded_snippet(completed.stdout),
            "stderr_snippet": _bounded_snippet(completed.stderr),
            "elapsed_time_s": float(time.perf_counter() - started),
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "command": command,
            "cwd": _relative_path(cwd, cwd.parent.parent),
            "return_code": None,
            "status": "timed_out",
            "timed_out": True,
            "stdout_snippet": _bounded_snippet(_decode_timeout_output(exc.stdout)),
            "stderr_snippet": _bounded_snippet(_decode_timeout_output(exc.stderr)),
            "elapsed_time_s": float(time.perf_counter() - started),
            "error": f"Command timed out after {timeout_seconds} seconds",
        }


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
    metadata_extra: dict[str, Any],
    compute_time_s: float = 0.0,
    write_time_s: float = 0.0,
) -> None:
    metadata = {
        "reason": reason,
        "error_type": error_type,
        **metadata_extra,
    }
    payload = {
        "schema_version": DENSE_BRIDGE_SCHEMA_VERSION,
        "bridge_id": DENSE_BRIDGE_ID,
        "manifest_kind": "dense_bridge_output",
        "backend": backend,
        "status": status,
        "input_manifest": input_manifest_path.name,
        "route_id": str(manifest.get("route_id", "")) if manifest is not None else "",
        "task_id": str(manifest.get("task_id", "")) if manifest is not None else "",
        "output_blob": output_blob,
        "accumulator_blob": None,
        "validation_metrics": validation_metrics,
        "compute_time_s": float(compute_time_s),
        "write_time_s": float(write_time_s),
        "total_time_s": float(total_time_s),
        "external_command_executed": external_command_executed,
        "execution_implemented": execution_implemented,
        "error": error,
        "metadata": metadata,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _base_metadata(target: str, dpu_executed: bool, prepared: dict[str, Any] | None = None) -> dict[str, Any]:
    execution_class = str((prepared or {}).get("execution_class") or EXECUTION_CLASS_L1)
    kernel_strategy = str((prepared or {}).get("kernel_strategy") or KERNEL_STRATEGY_L1)
    l2_plan = dict((prepared or {}).get("l2_plan") or {})
    metadata: dict[str, Any] = {
        "backend_family": "upmem_sdk",
        "simplepim_api_used": False,
        "simplepim_bridge_lane": True,
        "target": target,
        "execution_class": execution_class,
        "kernel_strategy": kernel_strategy,
        "native_buffer_layout": "row_major" if execution_class == EXECUTION_CLASS_L2 else "row_major_padded",
        "native_buffer_stride": "dynamic_gemm_n" if execution_class == EXECUTION_CLASS_L2 else "max_dim",
        "stride_model": "row_major_dynamic_stride_v1" if execution_class == EXECUTION_CLASS_L2 else "explicit_padded_stride_v1",
        "native_int32_output_dtype": "<i4",
        "mram_resident_operands": execution_class == EXECUTION_CLASS_L2,
        "wram_tiled": execution_class == EXECUTION_CLASS_L2,
        "upmem_dpu_program_executed": dpu_executed,
        "simulator_kernel_executed": dpu_executed and target == "simulator",
        "hardware_kernel_executed": False,
        "bringup_timing_note": "Collected timings are bring-up timings, not final performance evidence.",
    }
    if l2_plan:
        metadata["l2_tile_plan"] = {
            key: l2_plan.get(key)
            for key in (
                "tile_m",
                "tile_k",
                "tile_n",
                "output_tile_count",
                "k_tile_count",
                "total_tile_steps",
                "estimated_wram_bytes_per_tile",
                "effective_wram_bytes",
                "mram_bytes_a",
                "mram_bytes_b",
                "mram_bytes_c",
                "total_mram_bytes",
                "host_blob_bytes",
                "max_l2_host_blob_bytes",
                "native_max_dim",
            )
        }
    return metadata


def _blob_payload(path: Path, bridge_dir: Path, array: np.ndarray, manifest: dict[str, Any], mode: str) -> dict[str, Any]:
    return {
        "relative_path": _relative_path(path, bridge_dir),
        "dtype": str(array.dtype),
        "shape": tuple(int(dim) for dim in array.shape),
        "representation": "dequantized_output",
        "nbytes": int(array.nbytes),
        "labels": tuple(int(label) for label in manifest.get("output_labels", ())),
        "role": f"{mode}_upmem_sdk_simulator_output",
    }


def _validate_blob(array: np.ndarray, metadata: dict[str, Any], role: str) -> None:
    expected_shape = tuple(int(dim) for dim in metadata["shape"])
    expected_dtype = np.dtype(str(metadata["dtype"]))
    if tuple(array.shape) != expected_shape:
        raise ValueError(f"{role} blob shape {array.shape} does not match manifest shape {expected_shape}")
    if array.dtype != expected_dtype:
        raise ValueError(f"{role} blob dtype {array.dtype} does not match manifest dtype {expected_dtype}")


def _resolve_manifest_path(base_dir: Path, relative_path: str) -> Path:
    rel = Path(relative_path)
    if rel.is_absolute() or ".." in rel.parts:
        raise ValueError(f"Bridge manifest path must be relative and stay inside the bridge directory: {relative_path}")
    resolved = (base_dir / rel).resolve()
    resolved.relative_to(base_dir.resolve())
    return resolved


def _relative_path(path: Path, base_dir: Path) -> str:
    return os.path.relpath(path.resolve(), base_dir.resolve())


def _bounded_snippet(value: str | None) -> str:
    if not value:
        return ""
    if len(value) <= SNIPPET_LIMIT:
        return value
    return value[:SNIPPET_LIMIT] + "...<truncated>"


def _decode_timeout_output(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


class UnsupportedInput(ValueError):
    def __init__(self, reason: str, message: str):
        super().__init__(message)
        self.reason = reason


if __name__ == "__main__":
    raise SystemExit(main())
