from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Mapping

import numpy as np

from quantum_bench.core.jsonio import write_json
from quantum_bench.core.records import JsonDict, to_jsonable
from quantum_bench.formats import conversion_error_metrics
from quantum_bench.targets.upmem.simplepim import probe_simplepim

if TYPE_CHECKING:
    from quantum_bench.routing.dense_prepare import DenseTaskPreparationResult


DENSE_BRIDGE_SCHEMA_VERSION = "dense_bridge_v1"
DENSE_BRIDGE_ID = "upmem_dense_bridge_v1"

DenseBridgeStatus = Literal["mock_executed", "stub_executed", "skipped", "not_implemented", "failed", "unsupported"]
DenseBridgeBackendId = Literal["mock_numpy_dequantized", "simplepim_external", "simplepim_external_stub"]


@dataclass(frozen=True)
class DenseBridgeBackendIdentity:
    backend_id: str
    display_name: str
    backend_kind: str
    execution_mode: str
    external_command_capable: bool
    implemented: bool
    description: str


@dataclass(frozen=True)
class DenseBridgeExecutionRequest:
    input_manifest_path: Path
    backend: str = "mock_numpy_dequantized"
    execute_external: bool = False
    environment: Mapping[str, str] | None = None
    metadata: JsonDict = field(default_factory=dict)


@dataclass(frozen=True)
class DenseBridgeBlob:
    relative_path: str
    dtype: str
    shape: tuple[int, ...]
    representation: str
    nbytes: int
    labels: tuple[int, ...] = ()
    role: str = ""


@dataclass(frozen=True)
class DenseBridgeInputManifest:
    schema_version: str
    bridge_id: str
    manifest_kind: str
    route_id: str
    task_id: str
    input_tensor_ids: tuple[str, str]
    output_tensor_id: str
    preparation_status: str
    simplepim_probe: JsonDict
    external_command_executed: bool
    execution_implemented: bool
    gemm_m: int
    gemm_k: int
    gemm_n: int
    left_labels: tuple[int, ...]
    right_labels: tuple[int, ...]
    contracted_labels: tuple[int, ...]
    ordered_contracted_labels: tuple[int, ...]
    left_free_labels: tuple[int, ...]
    right_free_labels: tuple[int, ...]
    gemm_output_labels: tuple[int, ...]
    output_labels: tuple[int, ...]
    input_shapes: tuple[tuple[int, ...], tuple[int, ...]]
    left_matrix_shape: tuple[int, int]
    right_matrix_shape: tuple[int, int]
    output_shape: tuple[int, ...]
    fixed_point_spec: JsonDict
    conversion_records: JsonDict
    dequantization: JsonDict
    tile_plan: JsonDict
    upmem_task_estimate: JsonDict
    operands: JsonDict
    expected_output: DenseBridgeBlob
    metadata: JsonDict = field(default_factory=dict)

    def to_json_dict(self) -> JsonDict:
        return to_jsonable(self)


@dataclass(frozen=True)
class DenseBridgeOutputManifest:
    schema_version: str
    bridge_id: str
    manifest_kind: str
    backend: str
    status: DenseBridgeStatus
    input_manifest: str
    route_id: str
    task_id: str
    output_blob: DenseBridgeBlob | None
    accumulator_blob: JsonDict | None
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
class DenseBridgeResult:
    status: DenseBridgeStatus
    backend: str
    input_manifest_path: Path
    output_manifest_path: Path | None
    output_blob_path: Path | None
    error: str | None
    output_manifest: DenseBridgeOutputManifest | None
    external_command_executed: bool
    execution_implemented: bool
    metadata: JsonDict = field(default_factory=dict)

    def to_json_dict(self) -> JsonDict:
        base_dir = self.input_manifest_path.parent
        return to_jsonable(
            {
                "status": self.status,
                "backend": self.backend,
                "input_manifest_path": _relative_result_path(self.input_manifest_path, base_dir),
                "output_manifest_path": _relative_result_path(self.output_manifest_path, base_dir),
                "output_blob_path": _relative_result_path(self.output_blob_path, base_dir),
                "error": self.error,
                "output_manifest": self.output_manifest,
                "external_command_executed": self.external_command_executed,
                "execution_implemented": self.execution_implemented,
                "metadata": self.metadata,
            }
        )


@dataclass(frozen=True)
class DenseBridgeExecutionResult:
    schema_version: str
    bridge_id: str
    execution_status: DenseBridgeStatus
    backend_id: str
    backend_identity: DenseBridgeBackendIdentity | None
    reason: str | None
    error: str | None
    error_type: str | None
    input_manifest_path: str
    output_manifest_path: str | None
    output_blob_path: str | None
    output_manifest: DenseBridgeOutputManifest | None
    invocation_metadata: JsonDict
    external_command_executed: bool
    execution_implemented: bool
    metadata: JsonDict = field(default_factory=dict)

    def to_json_dict(self) -> JsonDict:
        return to_jsonable(self)


def write_dense_bridge_input_manifest(
    preparation_result: "DenseTaskPreparationResult",
    bridge_dir: Path,
) -> DenseBridgeInputManifest:
    _validate_preparation_result(preparation_result)
    prepared_operands = preparation_result.prepared_operands
    if prepared_operands is None:  # pragma: no cover - guarded above
        raise ValueError("prepared_operands are required for dense bridge input")

    operands_dir = bridge_dir / "operands"
    references_dir = bridge_dir / "references"
    operands_dir.mkdir(parents=True, exist_ok=True)
    references_dir.mkdir(parents=True, exist_ok=True)

    left_path = operands_dir / "left_quantized.npy"
    right_path = operands_dir / "right_quantized.npy"
    expected_path = references_dir / "expected_dequantized_output.npy"
    np.save(left_path, prepared_operands.left_quantized, allow_pickle=False)
    np.save(right_path, prepared_operands.right_quantized, allow_pickle=False)
    np.save(expected_path, prepared_operands.dequantized_output, allow_pickle=False)

    left_conversion = to_jsonable(preparation_result.left_conversion)
    right_conversion = to_jsonable(preparation_result.right_conversion)
    manifest = DenseBridgeInputManifest(
        schema_version=DENSE_BRIDGE_SCHEMA_VERSION,
        bridge_id=DENSE_BRIDGE_ID,
        manifest_kind="dense_bridge_input",
        route_id=preparation_result.route_id,
        task_id=preparation_result.task_id,
        input_tensor_ids=preparation_result.input_tensor_ids,
        output_tensor_id=preparation_result.output_tensor_id,
        preparation_status=preparation_result.status,
        simplepim_probe=preparation_result.simplepim_probe,
        external_command_executed=False,
        execution_implemented=False,
        gemm_m=preparation_result.gemm_m,
        gemm_k=preparation_result.gemm_k,
        gemm_n=preparation_result.gemm_n,
        left_labels=preparation_result.left_labels,
        right_labels=preparation_result.right_labels,
        contracted_labels=preparation_result.contracted_labels,
        ordered_contracted_labels=preparation_result.ordered_contracted_labels,
        left_free_labels=preparation_result.left_free_labels,
        right_free_labels=preparation_result.right_free_labels,
        gemm_output_labels=preparation_result.gemm_output_labels,
        output_labels=preparation_result.output_labels,
        input_shapes=preparation_result.input_shapes,
        left_matrix_shape=_require_pair_shape(preparation_result.left_matrix_shape, "left_matrix_shape"),
        right_matrix_shape=_require_pair_shape(preparation_result.right_matrix_shape, "right_matrix_shape"),
        output_shape=preparation_result.output_shape,
        fixed_point_spec=to_jsonable(preparation_result.fixed_point_spec),
        conversion_records={
            "left": left_conversion,
            "right": right_conversion,
        },
        dequantization={
            "left": _dequantization_payload(left_conversion),
            "right": _dequantization_payload(right_conversion),
            "output_scale_hint": float(left_conversion["scale"]) * float(right_conversion["scale"]),
        },
        tile_plan=dict(preparation_result.tile_plan or {}),
        upmem_task_estimate=dict(preparation_result.upmem_task_estimate or {}),
        operands={
            "left": _blob_metadata(left_path, bridge_dir, prepared_operands.left_quantized, left_conversion, "left_operand"),
            "right": _blob_metadata(right_path, bridge_dir, prepared_operands.right_quantized, right_conversion, "right_operand"),
        },
        expected_output=DenseBridgeBlob(
            relative_path=_relative_blob_path(expected_path, bridge_dir),
            dtype=str(prepared_operands.dequantized_output.dtype),
            shape=_array_shape(prepared_operands.dequantized_output),
            representation="dequantized_output_reference",
            nbytes=int(prepared_operands.dequantized_output.nbytes),
            labels=preparation_result.output_labels,
            role="expected_output",
        ),
        metadata={
            "blob_format": "npy",
            "blob_format_reason": "Wave 2C.7 uses .npy for inspectable dtype and shape metadata",
            "native_raw_int8_buffers_deferred": True,
            "complex_kernel_mapping_implemented": False,
            "mock_backend_available": True,
            "simplepim_or_native_execution_implemented": False,
        },
    )
    write_json(bridge_dir / "input_manifest.json", manifest)
    return manifest


def read_dense_bridge_input_manifest(path: Path) -> DenseBridgeInputManifest:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return _input_manifest_from_payload(payload)


def read_dense_bridge_output_manifest(path: Path) -> DenseBridgeOutputManifest:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return _output_manifest_from_payload(payload)


def dense_bridge_backend_registry() -> dict[str, DenseBridgeBackendIdentity]:
    return {
        "mock_numpy_dequantized": DenseBridgeBackendIdentity(
            backend_id="mock_numpy_dequantized",
            display_name="Mock NumPy Dequantized Dense Bridge",
            backend_kind="mock",
            execution_mode="in_process_python",
            external_command_capable=False,
            implemented=True,
            description="Reads dense bridge manifests and validates the file boundary with local NumPy GEMM.",
        ),
        "simplepim_external": DenseBridgeBackendIdentity(
            backend_id="simplepim_external",
            display_name="SimplePIM External Dense Bridge",
            backend_kind="simplepim_external",
            execution_mode="external_process",
            external_command_capable=True,
            implemented=False,
            description="Future SimplePIM bridge adapter; records invocation metadata only in this wave.",
        ),
        "simplepim_external_stub": DenseBridgeBackendIdentity(
            backend_id="simplepim_external_stub",
            display_name="SimplePIM External Dense Bridge Stub",
            backend_kind="simplepim_external_stub",
            execution_mode="external_process",
            external_command_capable=True,
            implemented=False,
            description="Non-executing external-process stub that validates the dense bridge file contract.",
        ),
    }


def get_dense_bridge_backend(backend_id: str) -> DenseBridgeBackendIdentity | None:
    return dense_bridge_backend_registry().get(backend_id)


def execute_dense_bridge(
    input_manifest_path: Path,
    backend: str = "mock_numpy_dequantized",
    *,
    execute_external: bool = False,
    env: Mapping[str, str] | None = None,
) -> DenseBridgeExecutionResult:
    request = DenseBridgeExecutionRequest(
        input_manifest_path=input_manifest_path,
        backend=backend,
        execute_external=execute_external,
        environment=env,
    )
    identity = get_dense_bridge_backend(request.backend)
    bridge_dir = input_manifest_path.parent
    output_manifest_path = bridge_dir / "output_manifest.json"

    if identity is None:
        output_manifest = _nonexecuted_output_manifest(
            backend=request.backend,
            status="unsupported",
            reason="unsupported_backend",
            error=f"Unsupported dense bridge backend: {request.backend}",
            error_type="unsupported_backend",
            input_manifest_path=input_manifest_path,
            manifest=None,
            invocation_metadata={"backend_id": request.backend},
            total_time_s=0.0,
        )
        write_json(output_manifest_path, output_manifest)
        return _execution_result(
            input_manifest_path=input_manifest_path,
            output_manifest_path=output_manifest_path,
            output_blob_path=None,
            backend_id=request.backend,
            backend_identity=None,
            execution_status="unsupported",
            reason="unsupported_backend",
            error=f"Unsupported dense bridge backend: {request.backend}",
            error_type="unsupported_backend",
            output_manifest=output_manifest,
            invocation_metadata={"backend_id": request.backend},
        )

    if request.backend == "mock_numpy_dequantized":
        result = run_mock_dense_bridge(input_manifest_path)
        return _execution_result_from_mock(result, identity)

    if request.backend == "simplepim_external":
        return _execute_simplepim_external_bridge(request, identity)

    if request.backend == "simplepim_external_stub":
        return _execute_simplepim_external_stub_bridge(request, identity)

    output_manifest = _nonexecuted_output_manifest(
        backend=request.backend,
        status="unsupported",
        reason="unsupported_backend",
        error=f"Unsupported dense bridge backend: {request.backend}",
        error_type="unsupported_backend",
        input_manifest_path=input_manifest_path,
        manifest=None,
        invocation_metadata={"backend_id": request.backend},
        total_time_s=0.0,
    )
    write_json(output_manifest_path, output_manifest)
    return _execution_result(
        input_manifest_path=input_manifest_path,
        output_manifest_path=output_manifest_path,
        output_blob_path=None,
        backend_id=request.backend,
        backend_identity=identity,
        execution_status="unsupported",
        reason="unsupported_backend",
        error=f"Unsupported dense bridge backend: {request.backend}",
        error_type="unsupported_backend",
        output_manifest=output_manifest,
        invocation_metadata={"backend_id": request.backend},
    )


def run_mock_dense_bridge(input_manifest_path: Path) -> DenseBridgeResult:
    started = time.perf_counter()
    backend = "mock_numpy_dequantized"
    output_manifest_path = input_manifest_path.parent / "output_manifest.json"
    output_blob_path = input_manifest_path.parent / "outputs" / "mock_dequantized_output.npy"
    try:
        manifest = read_dense_bridge_input_manifest(input_manifest_path)
        bridge_dir = input_manifest_path.parent
        left = np.load(_resolve_manifest_path(bridge_dir, manifest.operands["left"]["relative_path"]), allow_pickle=False)
        right = np.load(_resolve_manifest_path(bridge_dir, manifest.operands["right"]["relative_path"]), allow_pickle=False)
        _validate_loaded_blob(left, manifest.operands["left"], "left")
        _validate_loaded_blob(right, manifest.operands["right"], "right")

        compute_started = time.perf_counter()
        left_dequantized = _dequantize_blob(left, manifest.dequantization["left"])
        right_dequantized = _dequantize_blob(right, manifest.dequantization["right"])
        matrix_output = left_dequantized @ right_dequantized
        output = _restore_output_order(
            matrix_output,
            manifest.left_free_labels,
            manifest.right_free_labels,
            manifest.output_labels,
            manifest.output_shape,
        )
        compute_time_s = time.perf_counter() - compute_started

        write_started = time.perf_counter()
        output_blob_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_blob_path, output, allow_pickle=False)
        write_time_s = time.perf_counter() - write_started

        expected = np.load(_resolve_manifest_path(bridge_dir, manifest.expected_output.relative_path), allow_pickle=False)
        _validate_loaded_blob(expected, to_jsonable(manifest.expected_output), "expected_output")
        validation = conversion_error_metrics(expected, output)
        output_manifest = DenseBridgeOutputManifest(
            schema_version=DENSE_BRIDGE_SCHEMA_VERSION,
            bridge_id=DENSE_BRIDGE_ID,
            manifest_kind="dense_bridge_output",
            backend=backend,
            status="mock_executed",
            input_manifest=_relative_blob_path(input_manifest_path, bridge_dir),
            route_id=manifest.route_id,
            task_id=manifest.task_id,
            output_blob=DenseBridgeBlob(
                relative_path=_relative_blob_path(output_blob_path, bridge_dir),
                dtype=str(output.dtype),
                shape=_array_shape(output),
                representation="dequantized_output",
                nbytes=int(output.nbytes),
                labels=manifest.output_labels,
                role="mock_output",
            ),
            accumulator_blob=None,
            validation_metrics={
                "reference_kind": "expected_dequantized_output_vs_mock_numpy_dequantized",
                "max_abs_error": validation.max_abs_error,
                "l2_error": validation.l2_error,
                "relative_l2_error": validation.relative_l2_error,
                "expected_norm": float(np.linalg.norm(expected.ravel())),
                "output_norm": float(np.linalg.norm(output.ravel())),
            },
            compute_time_s=float(compute_time_s),
            write_time_s=float(write_time_s),
            total_time_s=float(time.perf_counter() - started),
            external_command_executed=False,
            execution_implemented=False,
            metadata={
                "mock_means": "file boundary validated with NumPy; no SimplePIM or native UPMEM execution happened",
                "native_output_contract": "future native bridge must produce output_manifest.json and a compatible output blob",
                "blob_format": "npy",
            },
        )
        write_json(output_manifest_path, output_manifest)
        return DenseBridgeResult(
            status="mock_executed",
            backend=backend,
            input_manifest_path=input_manifest_path,
            output_manifest_path=output_manifest_path,
            output_blob_path=output_blob_path,
            error=None,
            output_manifest=output_manifest,
            external_command_executed=False,
            execution_implemented=False,
        )
    except Exception as exc:
        output_manifest = DenseBridgeOutputManifest(
            schema_version=DENSE_BRIDGE_SCHEMA_VERSION,
            bridge_id=DENSE_BRIDGE_ID,
            manifest_kind="dense_bridge_output",
            backend=backend,
            status="failed",
            input_manifest=input_manifest_path.name,
            route_id="",
            task_id="",
            output_blob=None,
            accumulator_blob=None,
            validation_metrics={},
            compute_time_s=0.0,
            write_time_s=0.0,
            total_time_s=float(time.perf_counter() - started),
            external_command_executed=False,
            execution_implemented=False,
            error=str(exc),
            metadata={"mock_means": "file boundary validation failed before any external command"},
        )
        write_json(output_manifest_path, output_manifest)
        return DenseBridgeResult(
            status="failed",
            backend=backend,
            input_manifest_path=input_manifest_path,
            output_manifest_path=output_manifest_path,
            output_blob_path=None,
            error=str(exc),
            output_manifest=output_manifest,
            external_command_executed=False,
            execution_implemented=False,
        )


def _execute_simplepim_external_bridge(
    request: DenseBridgeExecutionRequest,
    identity: DenseBridgeBackendIdentity,
) -> DenseBridgeExecutionResult:
    started = time.perf_counter()
    input_manifest_path = request.input_manifest_path
    output_manifest_path = input_manifest_path.parent / "output_manifest.json"
    manifest: DenseBridgeInputManifest | None = None
    try:
        manifest = read_dense_bridge_input_manifest(input_manifest_path)
        _validate_bridge_input_files(manifest, input_manifest_path.parent)
    except Exception as exc:
        invocation_metadata = _simplepim_invocation_metadata(
            input_manifest_path=input_manifest_path,
            probe_payload={},
            env=request.environment,
        )
        output_manifest = _nonexecuted_output_manifest(
            backend=identity.backend_id,
            status="failed",
            reason="invalid_bridge_input_manifest",
            error=str(exc),
            error_type="invalid_bridge_input_manifest",
            input_manifest_path=input_manifest_path,
            manifest=manifest,
            invocation_metadata=invocation_metadata,
            total_time_s=float(time.perf_counter() - started),
        )
        write_json(output_manifest_path, output_manifest)
        return _execution_result(
            input_manifest_path=input_manifest_path,
            output_manifest_path=output_manifest_path,
            output_blob_path=None,
            backend_id=identity.backend_id,
            backend_identity=identity,
            execution_status="failed",
            reason="invalid_bridge_input_manifest",
            error=str(exc),
            error_type="invalid_bridge_input_manifest",
            output_manifest=output_manifest,
            invocation_metadata=invocation_metadata,
        )

    probe = probe_simplepim(env=request.environment if request.environment is not None else os.environ)
    probe_payload = probe.to_json_dict()
    invocation_metadata = _simplepim_invocation_metadata(
        input_manifest_path=input_manifest_path,
        probe_payload=probe_payload,
        env=request.environment,
    )

    if request.execute_external:
        status: DenseBridgeStatus = "not_implemented"
        reason = "simplepim_external_execution_not_implemented"
    elif probe.simplepim_probe_status == "unavailable":
        status = "skipped"
        reason = "simplepim_unavailable"
    else:
        status = "not_implemented"
        reason = "simplepim_external_execution_disabled"

    output_manifest = _nonexecuted_output_manifest(
        backend=identity.backend_id,
        status=status,
        reason=reason,
        error=None,
        error_type=None,
        input_manifest_path=input_manifest_path,
        manifest=manifest,
        invocation_metadata=invocation_metadata,
        total_time_s=float(time.perf_counter() - started),
    )
    write_json(output_manifest_path, output_manifest)
    return _execution_result(
        input_manifest_path=input_manifest_path,
        output_manifest_path=output_manifest_path,
        output_blob_path=None,
        backend_id=identity.backend_id,
        backend_identity=identity,
        execution_status=status,
        reason=reason,
        error=None,
        error_type=None,
        output_manifest=output_manifest,
        invocation_metadata=invocation_metadata,
    )


def _execute_simplepim_external_stub_bridge(
    request: DenseBridgeExecutionRequest,
    identity: DenseBridgeBackendIdentity,
) -> DenseBridgeExecutionResult:
    started = time.perf_counter()
    input_manifest_path = request.input_manifest_path
    bridge_dir = input_manifest_path.parent
    output_manifest_path = bridge_dir / "output_manifest.json"
    manifest: DenseBridgeInputManifest | None = None
    try:
        manifest = read_dense_bridge_input_manifest(input_manifest_path)
        _validate_bridge_input_files(manifest, bridge_dir)
    except Exception as exc:
        invocation_metadata = _simplepim_stub_invocation_metadata(
            input_manifest_path=input_manifest_path,
            stub_path=None,
            env=request.environment,
        )
        output_manifest = _nonexecuted_output_manifest(
            backend=identity.backend_id,
            status="failed",
            reason="invalid_bridge_input_manifest",
            error=str(exc),
            error_type="invalid_bridge_input_manifest",
            input_manifest_path=input_manifest_path,
            manifest=manifest,
            invocation_metadata=invocation_metadata,
            total_time_s=float(time.perf_counter() - started),
        )
        write_json(output_manifest_path, output_manifest)
        return _execution_result(
            input_manifest_path=input_manifest_path,
            output_manifest_path=output_manifest_path,
            output_blob_path=None,
            backend_id=identity.backend_id,
            backend_identity=identity,
            execution_status="failed",
            reason="invalid_bridge_input_manifest",
            error=str(exc),
            error_type="invalid_bridge_input_manifest",
            output_manifest=output_manifest,
            invocation_metadata=invocation_metadata,
        )

    stub_path = _configured_stub_path(request.environment)
    invocation_metadata = _simplepim_stub_invocation_metadata(
        input_manifest_path=input_manifest_path,
        stub_path=stub_path,
        env=request.environment,
    )

    if not request.execute_external:
        output_manifest = _nonexecuted_output_manifest(
            backend=identity.backend_id,
            status="not_implemented",
            reason="simplepim_external_stub_execution_disabled",
            error=None,
            error_type=None,
            input_manifest_path=input_manifest_path,
            manifest=manifest,
            invocation_metadata=invocation_metadata,
            total_time_s=float(time.perf_counter() - started),
        )
        write_json(output_manifest_path, output_manifest)
        return _execution_result(
            input_manifest_path=input_manifest_path,
            output_manifest_path=output_manifest_path,
            output_blob_path=None,
            backend_id=identity.backend_id,
            backend_identity=identity,
            execution_status="not_implemented",
            reason="simplepim_external_stub_execution_disabled",
            error=None,
            error_type=None,
            output_manifest=output_manifest,
            invocation_metadata=invocation_metadata,
        )

    if stub_path is None or not stub_path.exists():
        error = None if stub_path is None else f"Configured stub path does not exist: {stub_path}"
        output_manifest = _nonexecuted_output_manifest(
            backend=identity.backend_id,
            status="skipped",
            reason="simplepim_external_stub_unavailable",
            error=error,
            error_type=None,
            input_manifest_path=input_manifest_path,
            manifest=manifest,
            invocation_metadata=invocation_metadata,
            total_time_s=float(time.perf_counter() - started),
        )
        write_json(output_manifest_path, output_manifest)
        return _execution_result(
            input_manifest_path=input_manifest_path,
            output_manifest_path=output_manifest_path,
            output_blob_path=None,
            backend_id=identity.backend_id,
            backend_identity=identity,
            execution_status="skipped",
            reason="simplepim_external_stub_unavailable",
            error=error,
            error_type=None,
            output_manifest=output_manifest,
            invocation_metadata=invocation_metadata,
        )

    command = [
        sys.executable,
        str(stub_path),
        "--input-manifest",
        input_manifest_path.name,
        "--output-manifest",
        output_manifest_path.name,
        "--backend-id",
        identity.backend_id,
    ]
    invocation_metadata["command_args"] = tuple(command)
    invocation_metadata["external_command_executed"] = True
    completed = subprocess.run(
        command,
        cwd=bridge_dir,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        output_manifest = _nonexecuted_output_manifest(
            backend=identity.backend_id,
            status="failed",
            reason="simplepim_external_stub_failed",
            error=f"External stub exited with code {completed.returncode}",
            error_type="simplepim_external_stub_failed",
            input_manifest_path=input_manifest_path,
            manifest=manifest,
            invocation_metadata=invocation_metadata,
            total_time_s=float(time.perf_counter() - started),
            external_command_executed=True,
            metadata_extra={
                "native_kernel_executed": False,
                "stdout_snippet": _bounded_snippet(completed.stdout),
                "stderr_snippet": _bounded_snippet(completed.stderr),
                "returncode": int(completed.returncode),
            },
        )
        write_json(output_manifest_path, output_manifest)
        return _execution_result(
            input_manifest_path=input_manifest_path,
            output_manifest_path=output_manifest_path,
            output_blob_path=None,
            backend_id=identity.backend_id,
            backend_identity=identity,
            execution_status="failed",
            reason="simplepim_external_stub_failed",
            error=f"External stub exited with code {completed.returncode}",
            error_type="simplepim_external_stub_failed",
            output_manifest=output_manifest,
            invocation_metadata=invocation_metadata,
            external_command_executed=True,
            metadata={"native_kernel_executed": False},
        )

    try:
        output_manifest = read_dense_bridge_output_manifest(output_manifest_path)
    except Exception as exc:
        output_manifest = _nonexecuted_output_manifest(
            backend=identity.backend_id,
            status="failed",
            reason="simplepim_external_stub_output_manifest_invalid",
            error=str(exc),
            error_type="simplepim_external_stub_output_manifest_invalid",
            input_manifest_path=input_manifest_path,
            manifest=manifest,
            invocation_metadata=invocation_metadata,
            total_time_s=float(time.perf_counter() - started),
            external_command_executed=True,
            metadata_extra={
                "native_kernel_executed": False,
                "stdout_snippet": _bounded_snippet(completed.stdout),
                "stderr_snippet": _bounded_snippet(completed.stderr),
                "returncode": int(completed.returncode),
            },
        )
        write_json(output_manifest_path, output_manifest)
        return _execution_result(
            input_manifest_path=input_manifest_path,
            output_manifest_path=output_manifest_path,
            output_blob_path=None,
            backend_id=identity.backend_id,
            backend_identity=identity,
            execution_status="failed",
            reason="simplepim_external_stub_output_manifest_invalid",
            error=str(exc),
            error_type="simplepim_external_stub_output_manifest_invalid",
            output_manifest=output_manifest,
            invocation_metadata=invocation_metadata,
            external_command_executed=True,
            metadata={"native_kernel_executed": False},
        )

    if output_manifest.status == "failed":
        reason = str(output_manifest.metadata.get("reason") or "simplepim_external_stub_failed")
        error_type = str(output_manifest.metadata.get("error_type") or reason)
        return _execution_result(
            input_manifest_path=input_manifest_path,
            output_manifest_path=output_manifest_path,
            output_blob_path=None,
            backend_id=identity.backend_id,
            backend_identity=identity,
            execution_status="failed",
            reason=reason,
            error=output_manifest.error,
            error_type=error_type,
            output_manifest=output_manifest,
            invocation_metadata=invocation_metadata,
            external_command_executed=output_manifest.external_command_executed,
            metadata={"native_kernel_executed": False},
        )

    if output_manifest.status != "stub_executed":
        output_manifest = _nonexecuted_output_manifest(
            backend=identity.backend_id,
            status="failed",
            reason="simplepim_external_stub_output_manifest_invalid",
            error=f"Unexpected stub output status: {output_manifest.status}",
            error_type="simplepim_external_stub_output_manifest_invalid",
            input_manifest_path=input_manifest_path,
            manifest=manifest,
            invocation_metadata=invocation_metadata,
            total_time_s=float(time.perf_counter() - started),
            external_command_executed=True,
            metadata_extra={
                "native_kernel_executed": False,
                "stdout_snippet": _bounded_snippet(completed.stdout),
                "stderr_snippet": _bounded_snippet(completed.stderr),
                "returncode": int(completed.returncode),
            },
        )
        write_json(output_manifest_path, output_manifest)
        return _execution_result(
            input_manifest_path=input_manifest_path,
            output_manifest_path=output_manifest_path,
            output_blob_path=None,
            backend_id=identity.backend_id,
            backend_identity=identity,
            execution_status="failed",
            reason="simplepim_external_stub_output_manifest_invalid",
            error=output_manifest.error,
            error_type="simplepim_external_stub_output_manifest_invalid",
            output_manifest=output_manifest,
            invocation_metadata=invocation_metadata,
            external_command_executed=True,
            metadata={"native_kernel_executed": False},
        )

    return _execution_result(
        input_manifest_path=input_manifest_path,
        output_manifest_path=output_manifest_path,
        output_blob_path=None,
        backend_id=identity.backend_id,
        backend_identity=identity,
        execution_status="stub_executed",
        reason="external_stub_contract_executed",
        error=None,
        error_type=None,
        output_manifest=output_manifest,
        invocation_metadata=invocation_metadata,
        external_command_executed=output_manifest.external_command_executed,
        execution_implemented=output_manifest.execution_implemented,
        metadata={"native_kernel_executed": False},
    )


def _execution_result_from_mock(
    result: DenseBridgeResult,
    identity: DenseBridgeBackendIdentity,
) -> DenseBridgeExecutionResult:
    reason = None if result.status == "mock_executed" else "mock_bridge_failed"
    error_type = None if result.status == "mock_executed" else "mock_bridge_failed"
    return _execution_result(
        input_manifest_path=result.input_manifest_path,
        output_manifest_path=result.output_manifest_path,
        output_blob_path=result.output_blob_path,
        backend_id=identity.backend_id,
        backend_identity=identity,
        execution_status=result.status,
        reason=reason,
        error=result.error,
        error_type=error_type,
        output_manifest=result.output_manifest,
        invocation_metadata={
            "backend_id": identity.backend_id,
            "execution_mode": identity.execution_mode,
            "external_command": None,
            "external_command_executed": False,
            "blob_format": "npy",
        },
    )


def _execution_result(
    input_manifest_path: Path,
    output_manifest_path: Path | None,
    output_blob_path: Path | None,
    backend_id: str,
    backend_identity: DenseBridgeBackendIdentity | None,
    execution_status: DenseBridgeStatus,
    reason: str | None,
    error: str | None,
    error_type: str | None,
    output_manifest: DenseBridgeOutputManifest | None,
    invocation_metadata: JsonDict,
    external_command_executed: bool = False,
    execution_implemented: bool = False,
    metadata: JsonDict | None = None,
) -> DenseBridgeExecutionResult:
    base_dir = input_manifest_path.parent
    result_metadata = {
        "blob_format": "npy",
        "simplepim_or_native_execution_implemented": False,
    }
    if metadata:
        result_metadata.update(metadata)
    return DenseBridgeExecutionResult(
        schema_version=DENSE_BRIDGE_SCHEMA_VERSION,
        bridge_id=DENSE_BRIDGE_ID,
        execution_status=execution_status,
        backend_id=backend_id,
        backend_identity=backend_identity,
        reason=reason,
        error=error,
        error_type=error_type,
        input_manifest_path=_relative_result_path(input_manifest_path, base_dir),
        output_manifest_path=_relative_result_path(output_manifest_path, base_dir),
        output_blob_path=_relative_result_path(output_blob_path, base_dir),
        output_manifest=output_manifest,
        invocation_metadata=invocation_metadata,
        external_command_executed=external_command_executed,
        execution_implemented=execution_implemented,
        metadata=result_metadata,
    )


def _nonexecuted_output_manifest(
    backend: str,
    status: DenseBridgeStatus,
    reason: str,
    error: str | None,
    error_type: str | None,
    input_manifest_path: Path,
    manifest: DenseBridgeInputManifest | None,
    invocation_metadata: JsonDict,
    total_time_s: float,
    external_command_executed: bool = False,
    metadata_extra: JsonDict | None = None,
) -> DenseBridgeOutputManifest:
    bridge_dir = input_manifest_path.parent
    metadata = {
        "reason": reason,
        "error_type": error_type,
        "invocation_metadata": invocation_metadata,
        "mock_or_external_note": "No SimplePIM or native UPMEM kernel was executed",
    }
    if metadata_extra:
        metadata.update(metadata_extra)
    return DenseBridgeOutputManifest(
        schema_version=DENSE_BRIDGE_SCHEMA_VERSION,
        bridge_id=DENSE_BRIDGE_ID,
        manifest_kind="dense_bridge_output",
        backend=backend,
        status=status,
        input_manifest=_relative_result_path(input_manifest_path, bridge_dir) or input_manifest_path.name,
        route_id=manifest.route_id if manifest is not None else "",
        task_id=manifest.task_id if manifest is not None else "",
        output_blob=None,
        accumulator_blob=None,
        validation_metrics={},
        compute_time_s=0.0,
        write_time_s=0.0,
        total_time_s=total_time_s,
        external_command_executed=external_command_executed,
        execution_implemented=False,
        error=error,
        metadata=metadata,
    )


def _simplepim_invocation_metadata(
    input_manifest_path: Path,
    probe_payload: JsonDict,
    env: Mapping[str, str] | None,
) -> JsonDict:
    environment = env if env is not None else os.environ
    environment_keys = ("SIMPLEPIM_HOME", "SIMPLEPIM_BIN", "SIMPLEPIM_LIB")
    configured_environment_keys = tuple(key for key in environment_keys if environment.get(key))
    metadata: JsonDict = {
        "backend_id": "simplepim_external",
        "working_directory": ".",
        "input_manifest_path": _relative_result_path(input_manifest_path, input_manifest_path.parent)
        or input_manifest_path.name,
        "expected_output_manifest_path": "output_manifest.json",
        "expected_output_blob_path": "outputs/simplepim_output.npy",
        "environment_keys": environment_keys,
        "configured_environment_keys": configured_environment_keys,
        "blob_format": "npy",
        "external_command_executed": False,
    }
    command_path = probe_payload.get("simplepim_command_path")
    if command_path:
        metadata["command_path"] = command_path
    metadata["simplepim_available"] = bool(probe_payload.get("simplepim_available", False))
    metadata["simplepim_probe_status"] = probe_payload.get("simplepim_probe_status")
    return metadata


def _configured_stub_path(env: Mapping[str, str] | None) -> Path | None:
    environment = env if env is not None else os.environ
    raw = environment.get("SIMPLEPIM_STUB_BIN")
    if not raw:
        return None
    path = Path(raw)
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.resolve()


def _simplepim_stub_invocation_metadata(
    input_manifest_path: Path,
    stub_path: Path | None,
    env: Mapping[str, str] | None,
) -> JsonDict:
    environment = env if env is not None else os.environ
    environment_keys = ("SIMPLEPIM_STUB_BIN",)
    configured_environment_keys = tuple(key for key in environment_keys if environment.get(key))
    metadata: JsonDict = {
        "backend_id": "simplepim_external_stub",
        "working_directory": ".",
        "input_manifest_path": _relative_result_path(input_manifest_path, input_manifest_path.parent)
        or input_manifest_path.name,
        "expected_output_manifest_path": "output_manifest.json",
        "expected_output_blob_path": None,
        "environment_keys": environment_keys,
        "configured_environment_keys": configured_environment_keys,
        "blob_format": "npy",
        "external_command_executed": False,
        "native_kernel_executed": False,
    }
    if stub_path is not None:
        metadata["command_path"] = str(stub_path)
        metadata["command_args"] = (
            sys.executable,
            str(stub_path),
            "--input-manifest",
            input_manifest_path.name,
            "--output-manifest",
            "output_manifest.json",
            "--backend-id",
            "simplepim_external_stub",
        )
    return metadata


def _bounded_snippet(value: str | None, limit: int = 2000) -> str:
    if not value:
        return ""
    if len(value) <= limit:
        return value
    return value[:limit] + "...<truncated>"


def _validate_bridge_input_files(manifest: DenseBridgeInputManifest, bridge_dir: Path) -> None:
    left = np.load(_resolve_manifest_path(bridge_dir, manifest.operands["left"]["relative_path"]), allow_pickle=False)
    right = np.load(_resolve_manifest_path(bridge_dir, manifest.operands["right"]["relative_path"]), allow_pickle=False)
    expected = np.load(_resolve_manifest_path(bridge_dir, manifest.expected_output.relative_path), allow_pickle=False)
    _validate_loaded_blob(left, manifest.operands["left"], "left")
    _validate_loaded_blob(right, manifest.operands["right"], "right")
    _validate_loaded_blob(expected, to_jsonable(manifest.expected_output), "expected_output")


def _validate_preparation_result(preparation_result: "DenseTaskPreparationResult") -> None:
    if preparation_result.prepared_operands is None:
        raise ValueError("prepared_operands are required for dense bridge input")
    tile_plan = preparation_result.tile_plan or {}
    if tile_plan.get("requires_tiling"):
        raise ValueError("Dense bridge input requires a non-tiled task; executable tiling is not implemented")
    if tile_plan.get("requires_host_aggregation"):
        raise ValueError("Dense bridge input requires no host aggregation; split-K aggregation is not implemented")
    if preparation_result.status not in {"prepared", "simplepim_unavailable"}:
        raise ValueError(f"Preparation status {preparation_result.status!r} cannot be bridged")
    if preparation_result.left_conversion is None or preparation_result.right_conversion is None:
        raise ValueError("Fixed-point conversion records are required for dense bridge input")
    if preparation_result.left_matrix_shape is None or preparation_result.right_matrix_shape is None:
        raise ValueError("Matrix shapes are required for dense bridge input")


def _input_manifest_from_payload(payload: JsonDict) -> DenseBridgeInputManifest:
    return DenseBridgeInputManifest(
        schema_version=str(payload["schema_version"]),
        bridge_id=str(payload["bridge_id"]),
        manifest_kind=str(payload["manifest_kind"]),
        route_id=str(payload["route_id"]),
        task_id=str(payload["task_id"]),
        input_tensor_ids=_str_pair(payload["input_tensor_ids"]),
        output_tensor_id=str(payload["output_tensor_id"]),
        preparation_status=str(payload["preparation_status"]),
        simplepim_probe=dict(payload["simplepim_probe"]),
        external_command_executed=bool(payload["external_command_executed"]),
        execution_implemented=bool(payload["execution_implemented"]),
        gemm_m=int(payload["gemm_m"]),
        gemm_k=int(payload["gemm_k"]),
        gemm_n=int(payload["gemm_n"]),
        left_labels=_int_tuple(payload["left_labels"]),
        right_labels=_int_tuple(payload["right_labels"]),
        contracted_labels=_int_tuple(payload["contracted_labels"]),
        ordered_contracted_labels=_int_tuple(payload["ordered_contracted_labels"]),
        left_free_labels=_int_tuple(payload["left_free_labels"]),
        right_free_labels=_int_tuple(payload["right_free_labels"]),
        gemm_output_labels=_int_tuple(payload["gemm_output_labels"]),
        output_labels=_int_tuple(payload["output_labels"]),
        input_shapes=_shape_pair(payload["input_shapes"]),
        left_matrix_shape=_int_pair(payload["left_matrix_shape"]),
        right_matrix_shape=_int_pair(payload["right_matrix_shape"]),
        output_shape=_int_tuple(payload["output_shape"]),
        fixed_point_spec=dict(payload["fixed_point_spec"]),
        conversion_records=dict(payload["conversion_records"]),
        dequantization=dict(payload["dequantization"]),
        tile_plan=dict(payload["tile_plan"]),
        upmem_task_estimate=dict(payload["upmem_task_estimate"]),
        operands=dict(payload["operands"]),
        expected_output=_blob_from_payload(payload["expected_output"]),
        metadata=dict(payload.get("metadata", {})),
    )


def _output_manifest_from_payload(payload: JsonDict) -> DenseBridgeOutputManifest:
    output_blob = payload.get("output_blob")
    return DenseBridgeOutputManifest(
        schema_version=str(payload["schema_version"]),
        bridge_id=str(payload["bridge_id"]),
        manifest_kind=str(payload["manifest_kind"]),
        backend=str(payload["backend"]),
        status=payload["status"],
        input_manifest=str(payload["input_manifest"]),
        route_id=str(payload["route_id"]),
        task_id=str(payload["task_id"]),
        output_blob=_blob_from_payload(output_blob) if output_blob else None,
        accumulator_blob=payload.get("accumulator_blob"),
        validation_metrics=dict(payload["validation_metrics"]),
        compute_time_s=float(payload["compute_time_s"]),
        write_time_s=float(payload["write_time_s"]),
        total_time_s=float(payload["total_time_s"]),
        external_command_executed=bool(payload["external_command_executed"]),
        execution_implemented=bool(payload["execution_implemented"]),
        error=payload.get("error"),
        metadata=dict(payload.get("metadata", {})),
    )


def _blob_metadata(
    path: Path,
    base_dir: Path,
    array: np.ndarray,
    conversion_record: JsonDict,
    role: str,
) -> JsonDict:
    return to_jsonable(
        DenseBridgeBlob(
            relative_path=_relative_blob_path(path, base_dir),
            dtype=str(array.dtype),
            shape=_array_shape(array),
            representation=str(conversion_record["representation"]),
            nbytes=int(array.nbytes),
            labels=(),
            role=role,
        )
    )


def _blob_from_payload(payload: JsonDict) -> DenseBridgeBlob:
    return DenseBridgeBlob(
        relative_path=str(payload["relative_path"]),
        dtype=str(payload["dtype"]),
        shape=_int_tuple(payload["shape"]),
        representation=str(payload["representation"]),
        nbytes=int(payload["nbytes"]),
        labels=_int_tuple(payload.get("labels", ())),
        role=str(payload.get("role", "")),
    )


def _dequantization_payload(conversion_record: JsonDict) -> JsonDict:
    return {
        "scale": float(conversion_record["scale"]),
        "zero_point": int(conversion_record["zero_point"]),
        "representation": conversion_record["representation"],
        "source_shape": tuple(conversion_record["shape"]),
        "converted_shape": tuple(conversion_record["converted_shape"]),
        "route_dtype": conversion_record["route_dtype"],
        "source_dtype": conversion_record["source_dtype"],
    }


def _dequantize_blob(array: np.ndarray, metadata: JsonDict) -> np.ndarray:
    result = (array.astype(np.float64) - int(metadata["zero_point"])) * float(metadata["scale"])
    representation = metadata["representation"]
    source_shape = tuple(int(dim) for dim in metadata["source_shape"])
    if representation == "split_complex_real_imag":
        expected_shape = (*source_shape, 2)
        if tuple(result.shape) != expected_shape:
            raise ValueError(f"Converted complex blob shape {result.shape} does not match {expected_shape}")
        return result[..., 0] + 1j * result[..., 1]
    if representation != "real":
        raise ValueError(f"Unsupported bridge blob representation: {representation}")
    return result.reshape(source_shape)


def _restore_output_order(
    matrix_output: np.ndarray,
    left_free: tuple[int, ...],
    right_free: tuple[int, ...],
    output_labels: tuple[int, ...],
    output_shape: tuple[int, ...],
) -> np.ndarray:
    gemm_labels = left_free + right_free
    gemm_shape = tuple(int(output_shape[output_labels.index(label)]) for label in gemm_labels)
    tensor_output = np.asarray(matrix_output).reshape(gemm_shape)
    if gemm_labels == output_labels:
        return tensor_output
    axes = tuple(gemm_labels.index(label) for label in output_labels)
    return np.transpose(tensor_output, axes)


def _validate_loaded_blob(array: np.ndarray, metadata: JsonDict, role: str) -> None:
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
    return base_dir / rel


def _relative_blob_path(path: Path, base_dir: Path) -> str:
    return path.relative_to(base_dir).as_posix()


def _relative_result_path(path: Path | None, base_dir: Path) -> str | None:
    if path is None:
        return None
    try:
        return path.relative_to(base_dir).as_posix()
    except ValueError:
        return path.name


def _require_pair_shape(value: tuple[int, int] | None, field_name: str) -> tuple[int, int]:
    if value is None:
        raise ValueError(f"{field_name} is required")
    return (int(value[0]), int(value[1]))


def _array_shape(array: np.ndarray) -> tuple[int, ...]:
    return tuple(int(dim) for dim in array.shape)


def _int_tuple(value: Any) -> tuple[int, ...]:
    return tuple(int(item) for item in value)


def _int_pair(value: Any) -> tuple[int, int]:
    items = _int_tuple(value)
    if len(items) != 2:
        raise ValueError(f"Expected a pair, got {items}")
    return (items[0], items[1])


def _str_pair(value: Any) -> tuple[str, str]:
    items = tuple(str(item) for item in value)
    if len(items) != 2:
        raise ValueError(f"Expected a pair, got {items}")
    return (items[0], items[1])


def _shape_pair(value: Any) -> tuple[tuple[int, ...], tuple[int, ...]]:
    items = tuple(_int_tuple(item) for item in value)
    if len(items) != 2:
        raise ValueError(f"Expected a shape pair, got {items}")
    return (items[0], items[1])
