from __future__ import annotations

import argparse
import hashlib
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
HARDWARE_BACKEND_ID = "upmem_sdk_hardware_generic_loop"
KERNEL_FAMILY = "generic_loop_fallback"
DEFAULT_MAX_RANK = 16
DEFAULT_MAX_ELEMS = 65536
MAX_COMPILED_ELEMS = DEFAULT_MAX_ELEMS
OUTPUT_TILE_ELEMENTS = 256
GENERIC_KERNEL_STRATEGY = "mram_resident_output_tiled_v1"
DEFAULT_TIMEOUT_SECONDS = 30.0
SNIPPET_LIMIT = 2000
MODE_INT8_SCALED = "int8_scaled"
MODE_FLOAT32_NO_QUANT = "float32_no_quant"
HARDWARE_PROFILE_VERSION = "hardware_generic_loop_mvp_v1"
HARDWARE_SDK_ALLOCATION_PROFILE = "backend=hw"
HARDWARE_MAX_RANK = 4
HARDWARE_MAX_ELEMS = 16
HARDWARE_OUTPUT_TILE_ELEMENTS = 8
HARDWARE_TIMEOUT_SECONDS = 30.0
NATIVE_MODE_IDS = {
    MODE_INT8_SCALED: 0,
    MODE_FLOAT32_NO_QUANT: 1,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="UPMEM SDK simulator generic tensor-contraction loop runner")
    parser.add_argument("--input-manifest", required=True)
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--backend-id", default=BACKEND_ID)
    parser.add_argument("--target", default="simulator", choices=("simulator", "hardware"))
    parser.add_argument("--source-dir")
    parser.add_argument("--timeout-seconds", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    args = parser.parse_args(argv)

    started = time.perf_counter()
    input_path = Path(args.input_manifest)
    output_path = Path(args.output_manifest)
    bridge_dir = input_path.parent.resolve()
    manifest: dict[str, Any] | None = None
    try:
        manifest = _load_manifest(input_path)
        if args.target == "hardware":
            return _run_hardware_mvp(args, manifest, input_path, output_path, bridge_dir, started)
        prepared = _prepare_inputs(manifest, bridge_dir, os.environ)
        requested_max_elems = _positive_int_env(os.environ, "UPMEM_GENERIC_MAX_ELEMS", DEFAULT_MAX_ELEMS)
        max_elems = requested_max_elems
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
                metadata_extra=_base_metadata(args.target, False, max_elems=MAX_COMPILED_ELEMS),
            )
            return 0
        if max_elems > MAX_COMPILED_ELEMS:
            _write_output_manifest(
                output_path,
                backend=args.backend_id,
                status="unsupported",
                manifest=manifest,
                input_manifest_path=input_path,
                output_blob=None,
                validation_metrics={"status": "not_applicable", "reason": "requested_element_cap_exceeds_compiled_limit"},
                total_time_s=time.perf_counter() - started,
                error=f"UPMEM_GENERIC_MAX_ELEMS={max_elems} exceeds compiled limit {MAX_COMPILED_ELEMS}",
                reason="requested_element_cap_exceeds_compiled_limit",
                error_type="requested_element_cap_exceeds_compiled_limit",
                external_command_executed=True,
                execution_implemented=False,
                metadata_extra=_base_metadata(args.target, False, max_elems=MAX_COMPILED_ELEMS),
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
        transfer_accounting_path = outputs_dir / "transfer_accounting.json"
        args_path.write_bytes(_pack_args(prepared["native_index_metadata"]))
        prepared["left"].astype(prepared["input_dtype"], copy=False).ravel().tofile(left_path)
        prepared["right"].astype(prepared["input_dtype"], copy=False).ravel().tofile(right_path)

        build_started = time.perf_counter()
        build = _run_command(
            ("make", "clean", "all", f"MAX_RANK={DEFAULT_MAX_RANK}", f"MAX_ELEMS={max_elems}"),
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
            env={
                **os.environ,
                "DPU_BACKEND": "simulator",
                "UPMEM_GENERIC_TRANSFER_ACCOUNTING_JSON": str(transfer_accounting_path),
            },
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

        try:
            transfer_accounting = _load_transfer_accounting(transfer_accounting_path)
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
                error=f"UPMEM SDK generic loop transfer accounting invalid: {exc}",
                reason="transfer_accounting_invalid",
                error_type="transfer_accounting_invalid",
                external_command_executed=True,
                execution_implemented=True,
                metadata_extra={
                    **_base_metadata(args.target, True, max_elems=max_elems),
                    "build_time_s": build_time_s,
                    "simulator_run_time_s": simulator_run_time_s,
                    "build": build,
                    "run": run,
                },
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
                **transfer_accounting,
                "full_precision_h2d_bytes_model": prepared["full_precision_h2d_bytes_model"],
                "full_precision_d2h_bytes_model": prepared["full_precision_d2h_bytes_model"],
                "generic_kernel_strategy": prepared["generic_kernel_strategy"],
                "native_max_rank": prepared["native_max_rank"],
                "native_max_tensor_elements": prepared["native_max_tensor_elements"],
                "generic_output_tile_elements": prepared["generic_output_tile_elements"],
                "generic_output_tile_count": prepared["generic_output_tile_count"],
                "mram_resident_operands": prepared["mram_resident_operands"],
                "wram_output_tiled": prepared["wram_output_tiled"],
                "mram_tiled_task_count": prepared["mram_tiled_task_count"],
                "mram_read_bytes_model": prepared["mram_read_bytes_model"],
                "mram_write_bytes_model": prepared["mram_write_bytes_model"],
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


def _run_hardware_mvp(
    args: argparse.Namespace,
    manifest: dict[str, Any],
    input_path: Path,
    output_path: Path,
    bridge_dir: Path,
    started: float,
) -> int:
    """Execute the intentionally fixed one-DPU generic-loop hardware MVP.

    This branch owns physical SDK selection.  It must never inherit a
    simulator selector or retry through a different executor.
    """

    command: tuple[str, ...] | None = None

    def fail(
        stage: str,
        message: str,
        *,
        run: Mapping[str, Any] | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> int:
        metadata = _hardware_metadata(False, stage)
        if run:
            metadata.update(
                {
                    "stdout_snippet": run.get("stdout_snippet"),
                    "stderr_snippet": run.get("stderr_snippet"),
                    "returncode": run.get("returncode"),
                    "command": run.get("command"),
                }
            )
        if extra:
            metadata.update(dict(extra))
        _write_output_manifest(
            output_path,
            backend=args.backend_id,
            status="failed",
            manifest=manifest,
            input_manifest_path=input_path,
            output_blob=None,
            validation_metrics={"status": "failed", "reason": stage},
            total_time_s=time.perf_counter() - started,
            error=message,
            reason=stage,
            error_type=stage,
            external_command_executed=command is not None,
            execution_implemented=True,
            metadata_extra=metadata,
        )
        return 1

    try:
        if float(args.timeout_seconds) != HARDWARE_TIMEOUT_SECONDS:
            return fail(
                "hardware_profile_violation",
                f"{HARDWARE_PROFILE_VERSION} requires a fixed {HARDWARE_TIMEOUT_SECONDS:g}-second timeout",
            )
        if args.backend_id != HARDWARE_BACKEND_ID:
            return fail(
                "hardware_profile_violation",
                f"hardware backend ID must be {HARDWARE_BACKEND_ID}",
            )
        if os.environ.get("UPMEM_ALLOW_PHYSICAL_HARDWARE") != "1":
            return fail(
                "hardware_opt_in_missing",
                "UPMEM_ALLOW_PHYSICAL_HARDWARE=1 is required",
            )
        if os.environ.get("DPU_BACKEND"):
            return fail(
                "hardware_profile_violation",
                "DPU_BACKEND must not be inherited by the physical generic MVP",
            )
        if args.source_dir:
            return fail(
                "hardware_profile_violation",
                "physical generic MVP does not permit an alternate native source directory",
            )
        _validate_hardware_manifest(manifest)
        prepared = _prepare_inputs(manifest, bridge_dir, os.environ)
        if prepared["operand_mode"] != MODE_INT8_SCALED:
            return fail(
                "hardware_profile_violation",
                "physical generic MVP requires identity-scale int8 operands",
            )
        if (
            prepared["left"].size > HARDWARE_MAX_ELEMS
            or prepared["right"].size > HARDWARE_MAX_ELEMS
            or prepared["expected"].size != HARDWARE_MAX_ELEMS
        ):
            return fail(
                "hardware_profile_violation",
                "physical generic MVP requires operands <=16 elements and exactly 16 outputs",
            )
        native = dict(prepared["native_index_metadata"])
        args_blob = _pack_args(native, max_rank=HARDWARE_MAX_RANK)
        native_env = _sanitised_hardware_env(os.environ)
        tools = _hardware_tools(native_env)
        missing_tools = tuple(name for name, path in tools.items() if path is None)
        if missing_tools:
            return fail(
                "sdk_discovery_failed",
                "missing required UPMEM SDK tools: " + ", ".join(missing_tools),
                extra={"missing_tools": missing_tools},
            )

        source_dir = Path(__file__).resolve().parent / "upmem_sdk_generic_loop"
        runner_work = bridge_dir / "hardware_runner_work"
        source_snapshot = runner_work / "src"
        build_dir = runner_work / "build"
        inputs_dir = runner_work / "inputs"
        outputs_dir = runner_work / "outputs"
        _copy_source_tree(source_dir, source_snapshot)
        _copy_source_tree(source_dir, build_dir)
        inputs_dir.mkdir(parents=True, exist_ok=True)
        outputs_dir.mkdir(parents=True, exist_ok=True)

        build_command = (
            "make",
            "clean",
            "all",
            f"MAX_RANK={HARDWARE_MAX_RANK}",
            f"MAX_ELEMS={HARDWARE_MAX_ELEMS}",
            f"OUTPUT_TILE_ELEMS={HARDWARE_OUTPUT_TILE_ELEMENTS}",
            "NR_TASKLETS=1",
            "UPMEM_GENERIC_HARDWARE_MVP=1",
        )
        build_started = time.perf_counter()
        build = _run_command(
            build_command,
            cwd=build_dir,
            env=native_env,
            timeout_seconds=args.timeout_seconds,
        )
        build_time_s = time.perf_counter() - build_started
        if build["status"] != "passed":
            return fail(
                "native_build_failed",
                "isolated physical generic-loop build failed",
                run=build,
                extra={"build_time_s": build_time_s},
            )

        args_path = inputs_dir / "generic_args.bin"
        left_path = inputs_dir / "left_i8.bin"
        right_path = inputs_dir / "right_i8.bin"
        raw_output_path = outputs_dir / "generic_output_i32.bin"
        status_path = runner_work / "host_status.json"
        args_path.write_bytes(args_blob)
        prepared["left"].astype(np.int8, copy=False).ravel().tofile(left_path)
        prepared["right"].astype(np.int8, copy=False).ravel().tofile(right_path)
        command = (
            str(build_dir / "bin" / "host"),
            str(build_dir / "bin" / "dpu_generic"),
            str(args_path),
            str(left_path),
            str(right_path),
            str(raw_output_path),
        )
        host_env = dict(native_env)
        host_env["UPMEM_GENERIC_STATUS_JSON"] = str(status_path)
        run_started = time.perf_counter()
        run = _run_command(
            command,
            cwd=build_dir,
            env=host_env,
            timeout_seconds=args.timeout_seconds,
        )
        host_wall_time_s = time.perf_counter() - run_started
        host_status = _load_status(status_path)
        if run["status"] == "timeout":
            return fail("kernel_timeout", "physical generic host invocation timed out", run=run)
        if run["status"] != "passed":
            return fail(
                _hardware_failure_stage(run, host_status),
                "physical generic host invocation failed",
                run=run,
                extra={"hardware_status_json": host_status},
            )
        if not _hardware_status_valid(host_status):
            return fail(
                "output_manifest_failed",
                "physical generic host did not prove one-DPU backend=hw execution",
                run=run,
                extra={"hardware_status_json": host_status},
            )
        if not raw_output_path.exists():
            return fail("result_transfer_failed", "physical generic output buffer was not produced", run=run)

        output_shape = tuple(int(dim) for dim in manifest["output_shape"])
        accumulator = np.fromfile(raw_output_path, dtype="<i4")
        if accumulator.size != int(np.prod(output_shape)):
            return fail(
                "result_transfer_failed",
                "physical generic raw int32 output has an unexpected element count",
                run=run,
            )
        accumulator = accumulator.reshape(output_shape)
        expected_float = np.asarray(prepared["expected"])
        expected_i32 = np.rint(expected_float).astype("<i4")
        if not np.array_equal(expected_float, expected_i32.astype(expected_float.dtype)):
            return fail(
                "hardware_profile_violation",
                "identity-scale generic hardware reference is not integer-valued",
                run=run,
            )
        if not np.array_equal(accumulator, expected_i32):
            return fail(
                "output_validation_failed",
                "raw int32 generic output differs from the retained CPU reference",
                run=run,
            )
        output = accumulator.astype(np.float64) * float(prepared["output_scale"])
        if not np.allclose(output, expected_float, rtol=0.0, atol=0.0):
            return fail(
                "output_validation_failed",
                "dequantized generic output differs from expected reference",
                run=run,
            )

        raw_accumulator_path = outputs_dir / "hardware_accumulator_i32.npy"
        output_blob_path = bridge_dir / "outputs" / "upmem_sdk_hardware_generic_loop_output.npy"
        output_blob_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(raw_accumulator_path, accumulator, allow_pickle=False)
        write_started = time.perf_counter()
        stored_output = output.astype(expected_float.dtype, copy=False)
        np.save(output_blob_path, stored_output, allow_pickle=False)
        write_time_s = time.perf_counter() - write_started
        transfer = {
            "h2d": len(args_blob) + _align8(prepared["left"].nbytes) + _align8(prepared["right"].nbytes),
            "d2h": _align8(accumulator.nbytes),
        }
        transfer["total"] = int(transfer["h2d"] + transfer["d2h"])
        metadata = {
            **_hardware_metadata(True, None),
            "reason": "upmem_sdk_hardware_generic_loop_executed",
            "hardware_status_json": host_status,
            "application_visible_transfer_bytes": {
                **transfer,
                "h2d_components": {
                    "arguments": len(args_blob),
                    "left_int8": _align8(prepared["left"].nbytes),
                    "right_int8": _align8(prepared["right"].nbytes),
                },
                "d2h_components": {"output_int32": _align8(accumulator.nbytes)},
                "scope": "application_visible_sdk_buffers_not_physical_bus_counters",
            },
            "hashes": {
                "left": _hash_file(left_path),
                "right": _hash_file(right_path),
                "accumulator": _hash_file(raw_accumulator_path),
                "output": _hash_file(output_blob_path),
                "host_binary": _hash_file(build_dir / "bin" / "host"),
                "dpu_binary": _hash_file(build_dir / "bin" / "dpu_generic"),
                "input_manifest": _hash_file(input_path),
                "native_source_snapshot": _hash_tree(source_snapshot),
            },
            "native_build": build_command,
            "host_command": command,
            "build_time_s": build_time_s,
            "native_process_wall_time_s": host_wall_time_s,
            "native_build_stdout_snippet": build.get("stdout_snippet"),
            "native_build_stderr_snippet": build.get("stderr_snippet"),
            "host_stdout_snippet": run.get("stdout_snippet"),
            "host_stderr_snippet": run.get("stderr_snippet"),
            "allocation_time_s": host_status.get("allocation_time_s"),
            "binary_load_time_s": host_status.get("binary_load_time_s"),
            "h2d_time_s": host_status.get("h2d_time_s"),
            "kernel_time_s": host_status.get("kernel_time_s"),
            "d2h_time_s": host_status.get("d2h_time_s"),
            "host_output_write_time_s": host_status.get("output_write_time_s"),
            "reconstruction_time_s": write_time_s,
            "timing_decomposition_available": all(
                host_status.get(name) is not None
                for name in ("allocation_time_s", "binary_load_time_s", "h2d_time_s", "kernel_time_s", "d2h_time_s")
            ),
            "timing_decomposition_note": "Host-side SDK stage timings are bring-up diagnostics only, not isolated kernel benchmarks.",
            "sdk_tools": _tool_versions(tools, native_env),
            "runner_work": {
                "source": "hardware_runner_work/src",
                "build": "hardware_runner_work/build",
                "inputs": "hardware_runner_work/inputs",
                "outputs": "hardware_runner_work/outputs",
            },
        }
        _write_output_manifest(
            output_path,
            backend=args.backend_id,
            status="upmem_sdk_hardware_generic_loop_executed",
            manifest=manifest,
            input_manifest_path=input_path,
            output_blob=_blob_payload(output_blob_path, bridge_dir, stored_output, "generic_loop_hardware_dequantized_output"),
            validation_metrics={
                "reference_kind": "exact_int8_x_int8_to_int32_cpu_generic_loop_reference",
                "exact_integer_passed": True,
                "passed": True,
                "max_abs_error": 0.0,
                "l2_error": 0.0,
                "relative_l2_error": 0.0,
            },
            # This is the synchronous host-side SDK launch interval. It is
            # retained for bring-up diagnostics only and is not a performance
            # or speedup metric for this MVP.
            compute_time_s=float(host_status.get("kernel_time_s") or 0.0),
            write_time_s=write_time_s,
            total_time_s=time.perf_counter() - started,
            error=None,
            reason="upmem_sdk_hardware_generic_loop_executed",
            error_type=None,
            external_command_executed=True,
            execution_implemented=True,
            metadata_extra=metadata,
        )
        return 0
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        return fail("output_manifest_failed", str(exc))


def _validate_hardware_manifest(manifest: Mapping[str, Any]) -> None:
    metadata = manifest.get("metadata")
    native = manifest.get("native_index_metadata")
    fixed = manifest.get("fixed_point_spec")
    if not isinstance(metadata, Mapping) or not isinstance(native, Mapping) or not isinstance(fixed, Mapping):
        raise ValueError("hardware generic manifest is incomplete")
    expected = {
        "hardware_profile_version": HARDWARE_PROFILE_VERSION,
        "target": "hardware",
        "execution_class": "MRAM_WRAM_TILED",
        "backend_id": HARDWARE_BACKEND_ID,
        "requested_dpu_count": 1,
        "tasklets_per_dpu": 1,
        "max_rank": HARDWARE_MAX_RANK,
        "max_tensor_elements": HARDWARE_MAX_ELEMS,
        "output_tile_elements": HARDWARE_OUTPUT_TILE_ELEMENTS,
        "synchronous_execution": True,
        "performance_claim_applicable": False,
        "synthetic_real_taskgraph_mvp": True,
        "not_real_quantum_circuit": True,
        "sdk_allocation_profile": HARDWARE_SDK_ALLOCATION_PROFILE,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"hardware generic manifest metadata {key} is invalid")
    if manifest.get("backend_id") != HARDWARE_BACKEND_ID or manifest.get("execution_target") != "upmem_hardware":
        raise ValueError("hardware generic manifest backend or target is invalid")
    if fixed.get("route_dtype") != "int8" or fixed.get("complex_policy") != "reject" or float(fixed.get("scale", 0.0)) != 1.0:
        raise ValueError("hardware generic manifest must use identity int8 quantization")
    if (
        int(native.get("left_rank", -1)) != 3
        or int(native.get("right_rank", -1)) != 3
        or int(native.get("output_rank", -1)) != 4
        or int(native.get("output_element_count", -1)) != HARDWARE_MAX_ELEMS
        or int(native.get("contracted_combination_count", -1)) != 2
        or int(native.get("generic_output_tile_elements", -1)) != HARDWARE_OUTPUT_TILE_ELEMENTS
        or int(native.get("generic_output_tile_count", -1)) != 2
    ):
        raise ValueError("hardware generic manifest native task contract is invalid")


def _hardware_metadata(kernel_executed: bool, failure_stage: str | None) -> dict[str, Any]:
    return {
        **_base_metadata("hardware", kernel_executed, max_elems=HARDWARE_MAX_ELEMS),
        "hardware_profile_version": HARDWARE_PROFILE_VERSION,
        "target": "hardware",
        "execution_class": "MRAM_WRAM_TILED",
        "backend_id": HARDWARE_BACKEND_ID,
        "requested_dpu_count": 1,
        "tasklets_per_dpu": 1,
        "sdk_allocation_profile": HARDWARE_SDK_ALLOCATION_PROFILE,
        "sdk_allocation_profile_source": "compiled_native_literal",
        "hardware_kernel_executed": kernel_executed,
        "native_kernel_executed": kernel_executed,
        "simulator_kernel_executed": False,
        "cpu_fallback_used": False,
        "hardware_functionality_evidence": True,
        "hardware_speedup_applicable": False,
        "timing_labels": "hardware_bringup_functionality_only",
        "failure_stage": failure_stage,
        "generic_output_tile_elements": HARDWARE_OUTPUT_TILE_ELEMENTS,
        "generic_output_tile_count": 2,
        "mram_tiled_task_count": 1,
    }


def _sanitised_hardware_env(env: Mapping[str, str]) -> dict[str, str]:
    result = dict(env)
    for name in ("UPMEM_PROFILE", "UPMEM_PROFILE_BASE", "DPU_BACKEND"):
        result.pop(name, None)
    return result


def _hardware_tools(env: Mapping[str, str]) -> dict[str, str | None]:
    home = env.get("UPMEM_HOME")

    def find(name: str) -> str | None:
        if home:
            candidate = Path(home) / "bin" / name
            if candidate.exists():
                return str(candidate)
        return shutil.which(name, path=env.get("PATH"))

    return {name: find(name) for name in ("make", "dpu-upmem-dpurte-clang", "dpu-pkg-config")}


def _load_status(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _hardware_status_valid(status: Mapping[str, Any]) -> bool:
    return (
        status.get("success") is True
        and status.get("failure_stage") is None
        and status.get("allocation_profile") == HARDWARE_SDK_ALLOCATION_PROFILE
        and status.get("requested_dpus") == 1
        and status.get("allocated_dpus") == 1
        and status.get("tasklets") == 1
    )


def _hardware_failure_stage(run: Mapping[str, Any], status: Mapping[str, Any]) -> str:
    stage = str(status.get("failure_stage") or "")
    if stage in {
        "hardware_profile_violation", "hardware_allocation_failed", "binary_load_failed",
        "argument_transfer_failed", "operand_transfer_failed", "kernel_launch_failed",
        "result_transfer_failed", "output_manifest_failed", "hardware_release_failed",
    }:
        return stage
    text = f"{run.get('stdout_snippet', '')}\n{run.get('stderr_snippet', '')}".lower()
    for needles, candidate in (
        (("dpu_alloc", "allocation"), "hardware_allocation_failed"),
        (("dpu_load", "binary"), "binary_load_failed"),
        (("generic_args", "argument"), "argument_transfer_failed"),
        (("generic_a", "generic_b", "operand"), "operand_transfer_failed"),
        (("dpu_launch", "launch"), "kernel_launch_failed"),
        (("generic_c", "copy_from", "result"), "result_transfer_failed"),
        (("dpu_free", "release"), "hardware_release_failed"),
    ):
        if any(needle in text for needle in needles):
            return candidate
    return "kernel_launch_failed"


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _hash_tree(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(path.rglob("*")):
        if item.is_file():
            digest.update(item.relative_to(path).as_posix().encode("utf-8"))
            digest.update(_hash_file(item).encode("ascii"))
    return digest.hexdigest()


def _tool_versions(tools: Mapping[str, str | None], env: Mapping[str, str]) -> dict[str, str | None]:
    return {name: _command_version(path, env) if path else None for name, path in tools.items()}


def _command_version(command: str | None, env: Mapping[str, str]) -> str | None:
    if not command:
        return None
    try:
        completed = subprocess.run([command, "--version"], env=dict(env), capture_output=True, text=True, check=False, timeout=5.0)
    except (OSError, subprocess.TimeoutExpired):
        return None
    output = (completed.stdout or completed.stderr or "").strip().splitlines()
    return output[0] if output else None


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
    tile_metadata = _tile_metadata(native)
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
        **tile_metadata,
    }


def _load_transfer_accounting(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError("native host did not write transfer accounting sidecar")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "upmem_sdk_generic_loop_transfer_accounting_v1":
        raise ValueError("unsupported transfer accounting schema")
    required = (
        "prepared_payload_h2d_bytes",
        "prepared_payload_d2h_bytes",
        "actual_h2d_bytes",
        "actual_d2h_bytes",
        "actual_transfer_bytes",
        "control_bytes",
        "alignment_padding_bytes",
    )
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError("transfer accounting missing field(s): " + ", ".join(missing))
    actual_h2d = int(payload["actual_h2d_bytes"])
    actual_d2h = int(payload["actual_d2h_bytes"])
    total = int(payload["actual_transfer_bytes"])
    if min(actual_h2d, actual_d2h, total) < 0 or total != actual_h2d + actual_d2h:
        raise ValueError("transfer accounting byte invariant failed")
    if payload.get("physical_bus_bytes_available") is not False:
        raise ValueError("transfer accounting must not claim physical bus counters")
    payload["transfer_accounting_artifact"] = path.name
    return payload


def _pack_args(native: dict[str, Any], *, max_rank: int = DEFAULT_MAX_RANK) -> bytes:

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
    # The runner must compile its requested profile from source. Carrying a
    # developer's local native binaries into an isolated work tree can cause
    # Make to reuse a binary built with incompatible rank or element limits.
    shutil.copytree(source, dest, ignore=shutil.ignore_patterns("bin", "__pycache__", "*.pyc"))


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
        "generic_kernel_strategy": GENERIC_KERNEL_STRATEGY,
        "native_max_rank": DEFAULT_MAX_RANK,
        "native_max_tensor_elements": MAX_COMPILED_ELEMS,
        "generic_output_tile_elements": OUTPUT_TILE_ELEMENTS,
        "generic_output_tile_count": None,
        "mram_resident_operands": True,
        "wram_output_tiled": True,
        "mram_tiled_task_count": 0,
        "mram_read_bytes_model": 0,
        "mram_write_bytes_model": 0,
    }


def _tile_metadata(native: Mapping[str, Any]) -> dict[str, Any]:
    output_elements = int(native.get("output_element_count", 0) or 0)
    contracted = int(native.get("contracted_combination_count", 0) or 0)
    tile_count = (output_elements + OUTPUT_TILE_ELEMENTS - 1) // OUTPUT_TILE_ELEMENTS
    return {
        "generic_kernel_strategy": str(native.get("generic_kernel_strategy") or GENERIC_KERNEL_STRATEGY),
        "native_max_rank": DEFAULT_MAX_RANK,
        "native_max_tensor_elements": MAX_COMPILED_ELEMS,
        "generic_output_tile_elements": OUTPUT_TILE_ELEMENTS,
        "generic_output_tile_count": tile_count,
        "mram_resident_operands": True,
        "wram_output_tiled": True,
        "mram_tiled_task_count": int(tile_count > 1),
        "mram_read_bytes_model": output_elements * contracted * 2 * 8,
        "mram_write_bytes_model": _align8(output_elements * 4),
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
