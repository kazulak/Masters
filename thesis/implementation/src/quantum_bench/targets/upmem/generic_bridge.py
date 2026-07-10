from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Mapping

import numpy as np

from quantum_bench.core.jsonio import write_json
from quantum_bench.core.records import JsonDict, to_jsonable

if TYPE_CHECKING:
    from quantum_bench.routing.generic_prepare import GenericTaskPreparationResult


GENERIC_BRIDGE_SCHEMA_VERSION = "generic_contraction_bridge_v1"
GENERIC_BRIDGE_ID = "upmem_generic_contraction_bridge_v1"
GENERIC_LOOP_BACKEND_ID = "upmem_sdk_simulator_generic_loop"
GENERIC_LOOP_KERNEL_FAMILY = "generic_loop_fallback"

GenericBridgeStatus = Literal[
    "upmem_sdk_simulator_generic_loop_executed",
    "skipped",
    "not_implemented",
    "failed",
    "unsupported",
]


@dataclass(frozen=True)
class GenericBridgeBackendIdentity:
    backend_id: str
    display_name: str
    backend_kind: str
    kernel_family: str
    execution_mode: str
    external_command_capable: bool
    implemented: bool
    description: str


@dataclass(frozen=True)
class GenericBridgeBlob:
    relative_path: str
    dtype: str
    shape: tuple[int, ...]
    representation: str
    nbytes: int
    role: str


@dataclass(frozen=True)
class GenericBridgeInputManifest:
    schema_version: str
    bridge_id: str
    manifest_kind: str
    route_id: str
    backend_id: str
    kernel_family: str
    execution_target: str
    execution_scope: str
    task_id: str
    input_tensor_ids: tuple[str, str]
    output_tensor_id: str
    preparation_status: str
    external_command_executed: bool
    execution_implemented: bool
    input_shapes: tuple[tuple[int, ...], tuple[int, ...]]
    output_shape: tuple[int, ...]
    audit_labels: JsonDict
    native_index_metadata: JsonDict
    fixed_point_spec: JsonDict
    conversion_records: JsonDict
    dequantization: JsonDict
    operands: JsonDict
    expected_quantized_reference_output: GenericBridgeBlob
    full_precision_reference_output: GenericBridgeBlob
    metadata: JsonDict = field(default_factory=dict)

    def to_json_dict(self) -> JsonDict:
        return to_jsonable(self)


@dataclass(frozen=True)
class GenericBridgeOutputManifest:
    schema_version: str
    bridge_id: str
    manifest_kind: str
    backend: str
    status: GenericBridgeStatus
    input_manifest: str
    route_id: str
    task_id: str
    output_blob: GenericBridgeBlob | None
    validation_metrics: JsonDict
    compute_time_s: float
    write_time_s: float
    total_time_s: float
    external_command_executed: bool
    execution_implemented: bool
    error: str | None = None
    metadata: JsonDict = field(default_factory=dict)

    def to_json_dict(self) -> JsonDict:
        return to_jsonable(self)


@dataclass(frozen=True)
class GenericBridgeExecutionResult:
    schema_version: str
    bridge_id: str
    execution_status: GenericBridgeStatus
    backend_id: str
    backend_identity: GenericBridgeBackendIdentity | None
    reason: str | None
    error: str | None
    error_type: str | None
    input_manifest_path: str
    output_manifest_path: str | None
    output_blob_path: str | None
    output_manifest: GenericBridgeOutputManifest | None
    invocation_metadata: JsonDict
    external_command_executed: bool
    execution_implemented: bool
    metadata: JsonDict = field(default_factory=dict)

    def to_json_dict(self) -> JsonDict:
        return to_jsonable(self)


def write_generic_bridge_input_manifest(
    preparation_result: "GenericTaskPreparationResult",
    bridge_dir: Path,
) -> GenericBridgeInputManifest:
    if preparation_result.status != "prepared" or preparation_result.prepared_operands is None:
        raise ValueError(f"generic preparation is not bridgeable: {preparation_result.status}")
    operands = preparation_result.prepared_operands
    operands_dir = bridge_dir / "operands"
    references_dir = bridge_dir / "references"
    operands_dir.mkdir(parents=True, exist_ok=True)
    references_dir.mkdir(parents=True, exist_ok=True)

    operand_mode = str(getattr(operands, "operand_mode", "") or preparation_result.metadata.get("operand_mode") or "int8_scaled")
    left_operand = np.asarray(operands.left_operand if operands.left_operand is not None else operands.left_quantized)
    right_operand = np.asarray(operands.right_operand if operands.right_operand is not None else operands.right_quantized)
    expected_reference = np.asarray(
        operands.expected_reference_output
        if operands.expected_reference_output is not None
        else operands.expected_quantized_reference_output
    )

    if operand_mode == "float32_no_quant":
        left_path = operands_dir / "left_float32.npy"
        right_path = operands_dir / "right_float32.npy"
        expected_quantized_path = references_dir / "expected_float32_reference_output.npy"
        left_role = "left_float32_input"
        right_role = "right_float32_input"
        expected_role = "expected_float32_reference_output"
    else:
        left_path = operands_dir / "left_quantized.npy"
        right_path = operands_dir / "right_quantized.npy"
        expected_quantized_path = references_dir / "expected_quantized_reference_output.npy"
        left_role = "left_quantized_input"
        right_role = "right_quantized_input"
        expected_role = "expected_quantized_reference_output"
    full_precision_path = references_dir / "full_precision_reference_output.npy"
    np.save(left_path, left_operand, allow_pickle=False)
    np.save(right_path, right_operand, allow_pickle=False)
    np.save(expected_quantized_path, expected_reference, allow_pickle=False)
    np.save(full_precision_path, operands.full_precision_reference_output, allow_pickle=False)

    left_conversion = to_jsonable(preparation_result.left_conversion)
    right_conversion = to_jsonable(preparation_result.right_conversion)
    output_scale = 1.0
    if isinstance(left_conversion, dict) and isinstance(right_conversion, dict):
        output_scale = float(left_conversion["scale"]) * float(right_conversion["scale"])
    manifest = GenericBridgeInputManifest(
        schema_version=GENERIC_BRIDGE_SCHEMA_VERSION,
        bridge_id=GENERIC_BRIDGE_ID,
        manifest_kind="generic_contraction_bridge_input",
        route_id=preparation_result.route_id,
        backend_id=GENERIC_LOOP_BACKEND_ID,
        kernel_family=GENERIC_LOOP_KERNEL_FAMILY,
        execution_target="upmem_simulator",
        execution_scope="task_level",
        task_id=preparation_result.task_id,
        input_tensor_ids=preparation_result.input_tensor_ids,
        output_tensor_id=preparation_result.output_tensor_id,
        preparation_status=preparation_result.status,
        external_command_executed=False,
        execution_implemented=False,
        input_shapes=preparation_result.input_shapes,
        output_shape=preparation_result.output_shape,
        audit_labels={
            "left_labels": preparation_result.left_labels,
            "right_labels": preparation_result.right_labels,
            "contracted_labels": preparation_result.contracted_labels,
            "output_labels": preparation_result.output_labels,
        },
        native_index_metadata={
            "operand_mode": operand_mode,
            "left_rank": len(preparation_result.input_shapes[0]),
            "right_rank": len(preparation_result.input_shapes[1]),
            "output_rank": len(preparation_result.output_shape),
            "contracted_rank": len(preparation_result.contracted_dims),
            "left_shape": preparation_result.input_shapes[0],
            "right_shape": preparation_result.input_shapes[1],
            "output_shape": preparation_result.output_shape,
            "left_strides": preparation_result.left_strides,
            "right_strides": preparation_result.right_strides,
            "output_strides": preparation_result.output_strides,
            "output_to_left_axes": preparation_result.output_to_left_axes,
            "output_to_right_axes": preparation_result.output_to_right_axes,
            "contracted_to_left_axes": preparation_result.contracted_to_left_axes,
            "contracted_to_right_axes": preparation_result.contracted_to_right_axes,
            "contracted_dims": preparation_result.contracted_dims,
            "output_element_count": preparation_result.output_element_count,
            "contracted_combination_count": preparation_result.contracted_combination_count,
            "generic_kernel_strategy": preparation_result.metadata.get("generic_kernel_strategy"),
            "native_max_rank": preparation_result.metadata.get("native_max_rank"),
            "native_max_tensor_elements": preparation_result.metadata.get("native_max_tensor_elements"),
            "generic_output_tile_elements": preparation_result.metadata.get("generic_output_tile_elements"),
            "generic_output_tile_count": preparation_result.metadata.get("generic_output_tile_count"),
            "mram_resident_operands": preparation_result.metadata.get("mram_resident_operands"),
            "wram_output_tiled": preparation_result.metadata.get("wram_output_tiled"),
            "mram_tiled_task_count": preparation_result.metadata.get("mram_tiled_task_count"),
            "mram_read_bytes_model": preparation_result.metadata.get("mram_read_bytes_model"),
            "mram_write_bytes_model": preparation_result.metadata.get("mram_write_bytes_model"),
        },
        fixed_point_spec=to_jsonable(preparation_result.fixed_point_spec),
        conversion_records={"left": left_conversion, "right": right_conversion},
        dequantization={
            "left": {
                "scale": float(left_conversion["scale"]) if isinstance(left_conversion, dict) else 1.0,
                "zero_point": int(left_conversion["zero_point"]) if isinstance(left_conversion, dict) else 0,
            },
            "right": {
                "scale": float(right_conversion["scale"]) if isinstance(right_conversion, dict) else 1.0,
                "zero_point": int(right_conversion["zero_point"]) if isinstance(right_conversion, dict) else 0,
            },
            "output_scale": output_scale,
        },
        operands={
            "left": _blob_metadata(left_path, bridge_dir, left_operand, left_role),
            "right": _blob_metadata(right_path, bridge_dir, right_operand, right_role),
        },
        expected_quantized_reference_output=_blob_metadata(
            expected_quantized_path,
            bridge_dir,
            expected_reference,
            expected_role,
        ),
        full_precision_reference_output=_blob_metadata(
            full_precision_path,
            bridge_dir,
            operands.full_precision_reference_output,
            "full_precision_reference_output",
        ),
        metadata={
            "native_contract_uses_string_labels": False,
            "simplepim_api_used": False,
            "native_sdk_control_path": True,
            "quantization_mode": preparation_result.metadata.get("quantization_mode", "per_task_input_quantize"),
            "operand_mode": operand_mode,
            "input_dtype_on_dpu": preparation_result.metadata.get("input_dtype_on_dpu", str(left_operand.dtype)),
            "accumulator_dtype_on_dpu": preparation_result.metadata.get("accumulator_dtype_on_dpu"),
            "output_dtype_on_dpu": preparation_result.metadata.get("output_dtype_on_dpu"),
            "unquantized_mode_kind": preparation_result.metadata.get("unquantized_mode_kind"),
            "scaling_applied": bool(preparation_result.metadata.get("scaling_applied", operand_mode != "float32_no_quant")),
            "validation_target": preparation_result.metadata.get("validation_target", expected_role),
            "full_precision_reference_is_validation_target": False,
            "actual_h2d_bytes_model": preparation_result.metadata.get("actual_h2d_bytes_model"),
            "actual_d2h_bytes_model": preparation_result.metadata.get("actual_d2h_bytes_model"),
            "full_precision_h2d_bytes_model": preparation_result.metadata.get("full_precision_h2d_bytes_model"),
            "full_precision_d2h_bytes_model": preparation_result.metadata.get("full_precision_d2h_bytes_model"),
            "generic_kernel_strategy": preparation_result.metadata.get("generic_kernel_strategy"),
            "native_max_rank": preparation_result.metadata.get("native_max_rank"),
            "native_max_tensor_elements": preparation_result.metadata.get("native_max_tensor_elements"),
            "generic_output_tile_elements": preparation_result.metadata.get("generic_output_tile_elements"),
            "generic_output_tile_count": preparation_result.metadata.get("generic_output_tile_count"),
            "mram_resident_operands": preparation_result.metadata.get("mram_resident_operands"),
            "wram_output_tiled": preparation_result.metadata.get("wram_output_tiled"),
            "mram_tiled_task_count": preparation_result.metadata.get("mram_tiled_task_count"),
            "mram_read_bytes_model": preparation_result.metadata.get("mram_read_bytes_model"),
            "mram_write_bytes_model": preparation_result.metadata.get("mram_write_bytes_model"),
        },
    )
    write_json(bridge_dir / "input_manifest.json", manifest)
    return manifest


def generic_bridge_backend_registry() -> dict[str, GenericBridgeBackendIdentity]:
    return {
        GENERIC_LOOP_BACKEND_ID: GenericBridgeBackendIdentity(
            backend_id=GENERIC_LOOP_BACKEND_ID,
            display_name="UPMEM SDK Simulator Generic Loop Fallback",
            backend_kind="upmem_sdk_simulator_generic_loop",
            kernel_family=GENERIC_LOOP_KERNEL_FAMILY,
            execution_mode="external_process",
            external_command_capable=True,
            implemented=True,
            description="Executes small real-valued binary tensor contractions with an unoptimized UPMEM SDK simulator loop kernel.",
        )
    }


def get_generic_bridge_backend(backend_id: str) -> GenericBridgeBackendIdentity | None:
    return generic_bridge_backend_registry().get(backend_id)


def read_generic_bridge_input_manifest(path: Path) -> GenericBridgeInputManifest:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != GENERIC_BRIDGE_SCHEMA_VERSION:
        raise ValueError("unsupported generic bridge schema_version")
    if payload.get("bridge_id") != GENERIC_BRIDGE_ID:
        raise ValueError("unsupported generic bridge_id")
    if payload.get("manifest_kind") != "generic_contraction_bridge_input":
        raise ValueError("input manifest_kind must be generic_contraction_bridge_input")
    return _input_manifest_from_payload(payload)


def read_generic_bridge_output_manifest(path: Path) -> GenericBridgeOutputManifest:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != GENERIC_BRIDGE_SCHEMA_VERSION:
        raise ValueError("unsupported generic bridge output schema_version")
    if payload.get("bridge_id") != GENERIC_BRIDGE_ID:
        raise ValueError("unsupported generic bridge output bridge_id")
    return _output_manifest_from_payload(payload)


def execute_generic_bridge(
    input_manifest_path: Path,
    backend: str = GENERIC_LOOP_BACKEND_ID,
    *,
    execute_external: bool = False,
    env: Mapping[str, str] | None = None,
) -> GenericBridgeExecutionResult:
    identity = get_generic_bridge_backend(backend)
    output_manifest_path = input_manifest_path.parent / "output_manifest.json"
    if identity is None:
        output_manifest = _nonexecuted_output_manifest(
            backend=backend,
            status="unsupported",
            reason="unsupported_backend",
            error=f"Unsupported generic bridge backend: {backend}",
            input_manifest_path=input_manifest_path,
            manifest=None,
            total_time_s=0.0,
        )
        write_json(output_manifest_path, output_manifest)
        return _execution_result(input_manifest_path, output_manifest_path, None, backend, None, "unsupported", "unsupported_backend", output_manifest.error, "unsupported_backend", output_manifest, {})
    if backend == GENERIC_LOOP_BACKEND_ID:
        return _execute_upmem_sdk_generic_loop(input_manifest_path, identity, execute_external=execute_external, env=env)
    output_manifest = _nonexecuted_output_manifest(
        backend=backend,
        status="unsupported",
        reason="unsupported_backend",
        error=f"Unsupported generic bridge backend: {backend}",
        input_manifest_path=input_manifest_path,
        manifest=None,
        total_time_s=0.0,
    )
    write_json(output_manifest_path, output_manifest)
    return _execution_result(input_manifest_path, output_manifest_path, None, backend, identity, "unsupported", "unsupported_backend", output_manifest.error, "unsupported_backend", output_manifest, {})


def _execute_upmem_sdk_generic_loop(
    input_manifest_path: Path,
    identity: GenericBridgeBackendIdentity,
    *,
    execute_external: bool,
    env: Mapping[str, str] | None,
) -> GenericBridgeExecutionResult:
    started = time.perf_counter()
    bridge_dir = input_manifest_path.parent
    output_manifest_path = bridge_dir / "output_manifest.json"
    manifest: GenericBridgeInputManifest | None = None
    try:
        manifest = read_generic_bridge_input_manifest(input_manifest_path)
        _validate_bridge_input_files(manifest, bridge_dir)
    except Exception as exc:
        output_manifest = _nonexecuted_output_manifest(
            backend=identity.backend_id,
            status="failed",
            reason="input_manifest_invalid",
            error=str(exc),
            input_manifest_path=input_manifest_path,
            manifest=manifest,
            total_time_s=float(time.perf_counter() - started),
        )
        write_json(output_manifest_path, output_manifest)
        return _execution_result(input_manifest_path, output_manifest_path, None, identity.backend_id, identity, "failed", "input_manifest_invalid", str(exc), "input_manifest_invalid", output_manifest, {})
    if not execute_external:
        output_manifest = _nonexecuted_output_manifest(
            backend=identity.backend_id,
            status="not_implemented",
            reason="generic_external_execution_disabled",
            error=None,
            input_manifest_path=input_manifest_path,
            manifest=manifest,
            total_time_s=float(time.perf_counter() - started),
        )
        write_json(output_manifest_path, output_manifest)
        return _execution_result(input_manifest_path, output_manifest_path, None, identity.backend_id, identity, "not_implemented", "generic_external_execution_disabled", None, None, output_manifest, {})

    runner = Path(__file__).resolve().parents[4] / "native" / "upmem" / "simplepim" / "upmem_sdk_generic_loop_runner.py"
    invocation = {
        "backend_id": identity.backend_id,
        "command": (sys.executable, str(runner), "--input-manifest", input_manifest_path.name, "--output-manifest", output_manifest_path.name, "--backend-id", identity.backend_id, "--target", "simulator"),
        "working_directory": ".",
        "external_command_executed": True,
    }
    completed = subprocess.run(
        list(invocation["command"]),
        cwd=bridge_dir,
        env={**os.environ, **dict(env or {})},
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        output_manifest = _nonexecuted_output_manifest(
            backend=identity.backend_id,
            status="failed",
            reason="runner_execution_failed",
            error=f"Generic loop runner exited with code {completed.returncode}",
            input_manifest_path=input_manifest_path,
            manifest=manifest,
            total_time_s=float(time.perf_counter() - started),
            external_command_executed=True,
            execution_implemented=True,
            metadata_extra={
                "stdout_snippet": _bounded_snippet(completed.stdout),
                "stderr_snippet": _bounded_snippet(completed.stderr),
                "returncode": int(completed.returncode),
            },
        )
        write_json(output_manifest_path, output_manifest)
        return _execution_result(input_manifest_path, output_manifest_path, None, identity.backend_id, identity, "failed", "runner_execution_failed", output_manifest.error, "runner_execution_failed", output_manifest, invocation)
    try:
        output_manifest = read_generic_bridge_output_manifest(output_manifest_path)
    except Exception as exc:
        output_manifest = _nonexecuted_output_manifest(
            backend=identity.backend_id,
            status="failed",
            reason="output_manifest_invalid",
            error=str(exc),
            input_manifest_path=input_manifest_path,
            manifest=manifest,
            total_time_s=float(time.perf_counter() - started),
            external_command_executed=True,
            execution_implemented=True,
        )
        write_json(output_manifest_path, output_manifest)
        return _execution_result(input_manifest_path, output_manifest_path, None, identity.backend_id, identity, "failed", "output_manifest_invalid", str(exc), "output_manifest_invalid", output_manifest, invocation)
    output_blob_path = None
    if output_manifest.output_blob is not None:
        output_blob_path = _resolve_manifest_path(bridge_dir, output_manifest.output_blob.relative_path)
        if not output_blob_path.exists():
            return _execution_result(input_manifest_path, output_manifest_path, None, identity.backend_id, identity, "failed", "output_blob_missing", "Generic bridge output blob missing", "output_blob_missing", output_manifest, invocation)
    if output_manifest.status != "upmem_sdk_simulator_generic_loop_executed":
        return _execution_result(input_manifest_path, output_manifest_path, output_blob_path, identity.backend_id, identity, output_manifest.status, str(output_manifest.metadata.get("reason") or output_manifest.status), output_manifest.error, str(output_manifest.metadata.get("error_type") or output_manifest.status), output_manifest, invocation)
    return _execution_result(
        input_manifest_path,
        output_manifest_path,
        output_blob_path,
        identity.backend_id,
        identity,
        output_manifest.status,
        "upmem_sdk_simulator_generic_loop_executed",
        None,
        None,
        output_manifest,
        invocation,
    )


def _validate_bridge_input_files(manifest: GenericBridgeInputManifest, bridge_dir: Path) -> None:
    for blob in (
        manifest.operands["left"],
        manifest.operands["right"],
        to_jsonable(manifest.expected_quantized_reference_output),
        to_jsonable(manifest.full_precision_reference_output),
    ):
        path = _resolve_manifest_path(bridge_dir, str(blob["relative_path"]))
        if not path.exists():
            raise ValueError(f"missing bridge blob: {blob['relative_path']}")
        array = np.load(path, allow_pickle=False)
        if tuple(int(dim) for dim in array.shape) != tuple(int(dim) for dim in blob["shape"]):
            raise ValueError(f"blob shape mismatch for {blob['relative_path']}")
        if str(array.dtype) != str(blob["dtype"]):
            raise ValueError(f"blob dtype mismatch for {blob['relative_path']}")


def _blob_metadata(path: Path, bridge_dir: Path, array: np.ndarray, role: str) -> GenericBridgeBlob:
    return GenericBridgeBlob(
        relative_path=_relative_blob_path(path, bridge_dir),
        dtype=str(array.dtype),
        shape=tuple(int(dim) for dim in array.shape),
        representation=role,
        nbytes=int(array.nbytes),
        role=role,
    )


def _relative_blob_path(path: Path, bridge_dir: Path) -> str:
    return path.resolve().relative_to(bridge_dir.resolve()).as_posix()


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


def _input_manifest_from_payload(payload: JsonDict) -> GenericBridgeInputManifest:
    return GenericBridgeInputManifest(
        schema_version=str(payload["schema_version"]),
        bridge_id=str(payload["bridge_id"]),
        manifest_kind=str(payload["manifest_kind"]),
        route_id=str(payload["route_id"]),
        backend_id=str(payload["backend_id"]),
        kernel_family=str(payload["kernel_family"]),
        execution_target=str(payload["execution_target"]),
        execution_scope=str(payload["execution_scope"]),
        task_id=str(payload["task_id"]),
        input_tensor_ids=tuple(str(item) for item in payload["input_tensor_ids"]),
        output_tensor_id=str(payload["output_tensor_id"]),
        preparation_status=str(payload["preparation_status"]),
        external_command_executed=bool(payload["external_command_executed"]),
        execution_implemented=bool(payload["execution_implemented"]),
        input_shapes=tuple(tuple(int(dim) for dim in shape) for shape in payload["input_shapes"]),
        output_shape=tuple(int(dim) for dim in payload["output_shape"]),
        audit_labels=dict(payload["audit_labels"]),
        native_index_metadata=dict(payload["native_index_metadata"]),
        fixed_point_spec=dict(payload["fixed_point_spec"]),
        conversion_records=dict(payload["conversion_records"]),
        dequantization=dict(payload["dequantization"]),
        operands=dict(payload["operands"]),
        expected_quantized_reference_output=_blob_from_payload(payload["expected_quantized_reference_output"]),
        full_precision_reference_output=_blob_from_payload(payload["full_precision_reference_output"]),
        metadata=dict(payload.get("metadata") or {}),
    )


def _output_manifest_from_payload(payload: JsonDict) -> GenericBridgeOutputManifest:
    return GenericBridgeOutputManifest(
        schema_version=str(payload["schema_version"]),
        bridge_id=str(payload["bridge_id"]),
        manifest_kind=str(payload["manifest_kind"]),
        backend=str(payload["backend"]),
        status=payload["status"],
        input_manifest=str(payload["input_manifest"]),
        route_id=str(payload["route_id"]),
        task_id=str(payload["task_id"]),
        output_blob=_blob_from_payload(payload["output_blob"]) if payload.get("output_blob") else None,
        validation_metrics=dict(payload.get("validation_metrics") or {}),
        compute_time_s=float(payload.get("compute_time_s", 0.0) or 0.0),
        write_time_s=float(payload.get("write_time_s", 0.0) or 0.0),
        total_time_s=float(payload.get("total_time_s", 0.0) or 0.0),
        external_command_executed=bool(payload.get("external_command_executed", False)),
        execution_implemented=bool(payload.get("execution_implemented", False)),
        error=payload.get("error"),
        metadata=dict(payload.get("metadata") or {}),
    )


def _blob_from_payload(payload: JsonDict) -> GenericBridgeBlob:
    return GenericBridgeBlob(
        relative_path=str(payload["relative_path"]),
        dtype=str(payload["dtype"]),
        shape=tuple(int(dim) for dim in payload["shape"]),
        representation=str(payload["representation"]),
        nbytes=int(payload["nbytes"]),
        role=str(payload["role"]),
    )


def _nonexecuted_output_manifest(
    *,
    backend: str,
    status: GenericBridgeStatus,
    reason: str,
    error: str | None,
    input_manifest_path: Path,
    manifest: GenericBridgeInputManifest | None,
    total_time_s: float,
    external_command_executed: bool = False,
    execution_implemented: bool = False,
    metadata_extra: JsonDict | None = None,
) -> GenericBridgeOutputManifest:
    return GenericBridgeOutputManifest(
        schema_version=GENERIC_BRIDGE_SCHEMA_VERSION,
        bridge_id=GENERIC_BRIDGE_ID,
        manifest_kind="generic_contraction_bridge_output",
        backend=backend,
        status=status,
        input_manifest=input_manifest_path.name,
        route_id=manifest.route_id if manifest is not None else "",
        task_id=manifest.task_id if manifest is not None else "",
        output_blob=None,
        validation_metrics={"status": "not_applicable", "reason": reason},
        compute_time_s=0.0,
        write_time_s=0.0,
        total_time_s=float(total_time_s),
        external_command_executed=external_command_executed,
        execution_implemented=execution_implemented,
        error=error,
        metadata={
            "reason": reason,
            "error_type": reason if error else None,
            "kernel_family": GENERIC_LOOP_KERNEL_FAMILY,
            "simplepim_api_used": False,
            "native_sdk_control_path": True,
            "simulator_kernel_executed": False,
            "hardware_kernel_executed": False,
            **(metadata_extra or {}),
        },
    )


def _execution_result(
    input_manifest_path: Path,
    output_manifest_path: Path | None,
    output_blob_path: Path | None,
    backend_id: str,
    backend_identity: GenericBridgeBackendIdentity | None,
    execution_status: GenericBridgeStatus,
    reason: str | None,
    error: str | None,
    error_type: str | None,
    output_manifest: GenericBridgeOutputManifest | None,
    invocation_metadata: JsonDict,
) -> GenericBridgeExecutionResult:
    bridge_dir = input_manifest_path.parent
    metadata = dict(output_manifest.metadata) if output_manifest is not None else {}
    return GenericBridgeExecutionResult(
        schema_version=GENERIC_BRIDGE_SCHEMA_VERSION,
        bridge_id=GENERIC_BRIDGE_ID,
        execution_status=execution_status,
        backend_id=backend_id,
        backend_identity=backend_identity,
        reason=reason,
        error=error,
        error_type=error_type,
        input_manifest_path=_relative_result_path(input_manifest_path, bridge_dir),
        output_manifest_path=_relative_result_path(output_manifest_path, bridge_dir),
        output_blob_path=_relative_result_path(output_blob_path, bridge_dir),
        output_manifest=output_manifest,
        invocation_metadata=invocation_metadata,
        external_command_executed=bool(output_manifest.external_command_executed if output_manifest is not None else False),
        execution_implemented=bool(output_manifest.execution_implemented if output_manifest is not None else False),
        metadata=metadata,
    )


def _relative_result_path(path: Path | None, base_dir: Path) -> str | None:
    if path is None:
        return None
    try:
        return path.resolve().relative_to(base_dir.resolve()).as_posix()
    except ValueError:
        return path.name


def _bounded_snippet(text: str, limit: int = 2000) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "...<truncated>"
