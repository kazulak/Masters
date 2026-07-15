from __future__ import annotations

import json
import os
import shutil
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
from quantum_bench.targets.upmem.hardware_mvp import HARDWARE_MVP_SDK_ALLOCATION_PROFILE
from quantum_bench.targets.upmem.simplepim import probe_simplepim
from quantum_bench.targets.upmem.tile_plan import (
    UPMEM_EXECUTION_CLASS_L1_WRAM,
    UPMEM_EXECUTION_CLASS_L2_SINGLE_DPU_MRAM,
    UPMEM_L1_KERNEL_STRATEGY,
    UPMEM_L2_KERNEL_STRATEGY,
    UPMEM_L2_MAX_HOST_BLOB_BYTES,
    UPMEM_L2_NATIVE_MAX_DIM,
    plan_l2_tiled_execution,
)

if TYPE_CHECKING:
    from quantum_bench.routing.dense_prepare import DenseTaskPreparationResult


DENSE_BRIDGE_SCHEMA_VERSION = "dense_bridge_v1"
DENSE_BRIDGE_ID = "upmem_dense_bridge_v1"
UPMEM_SDK_HARDWARE_DENSE_BACKEND_ID = "upmem_sdk_hardware_dense"

DenseBridgeStatus = Literal[
    "mock_executed",
    "stub_executed",
    "upmem_sdk_simulator_executed",
    "upmem_sdk_hardware_executed",
    "skipped",
    "not_implemented",
    "failed",
    "unsupported",
]
DenseBridgeBackendId = Literal[
    "mock_numpy_dequantized",
    "simplepim_external",
    "simplepim_external_stub",
    "upmem_sdk_simulator_dense",
    "upmem_sdk_hardware_dense",
]


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
                "input_manifest_path": _relative_result_path(
                    self.input_manifest_path, base_dir
                ),
                "output_manifest_path": _relative_result_path(
                    self.output_manifest_path, base_dir
                ),
                "output_blob_path": _relative_result_path(
                    self.output_blob_path, base_dir
                ),
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
    requires_tiling = bool((preparation_result.tile_plan or {}).get("requires_tiling"))
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
        left_matrix_shape=_require_pair_shape(
            preparation_result.left_matrix_shape, "left_matrix_shape"
        ),
        right_matrix_shape=_require_pair_shape(
            preparation_result.right_matrix_shape, "right_matrix_shape"
        ),
        output_shape=preparation_result.output_shape,
        fixed_point_spec=to_jsonable(preparation_result.fixed_point_spec),
        conversion_records={
            "left": left_conversion,
            "right": right_conversion,
        },
        dequantization={
            "left": _dequantization_payload(left_conversion),
            "right": _dequantization_payload(right_conversion),
            "output_scale_hint": float(left_conversion["scale"])
            * float(right_conversion["scale"]),
        },
        tile_plan=dict(preparation_result.tile_plan or {}),
        upmem_task_estimate=dict(preparation_result.upmem_task_estimate or {}),
        operands={
            "left": _blob_metadata(
                left_path,
                bridge_dir,
                prepared_operands.left_quantized,
                left_conversion,
                "left_operand",
            ),
            "right": _blob_metadata(
                right_path,
                bridge_dir,
                prepared_operands.right_quantized,
                right_conversion,
                "right_operand",
            ),
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
            "execution_class_hint": UPMEM_EXECUTION_CLASS_L2_SINGLE_DPU_MRAM
            if requires_tiling
            else UPMEM_EXECUTION_CLASS_L1_WRAM,
            "kernel_strategy_hint": UPMEM_L2_KERNEL_STRATEGY
            if requires_tiling
            else UPMEM_L1_KERNEL_STRATEGY,
        },
    )
    write_json(bridge_dir / "input_manifest.json", manifest)
    return manifest


def dense_bridge_manifest_eligibility(
    preparation_result: object | None,
) -> tuple[bool, str | None]:
    """Return whether a dense preparation result can be serialized as a bridge input."""

    if preparation_result is None:
        return False, "dense_preparation_missing"
    if getattr(preparation_result, "prepared_operands", None) is None:
        return False, "prepared_operands_missing"
    tile_plan = getattr(preparation_result, "tile_plan", None)
    if tile_plan is None:
        return False, "tile_plan_missing"
    if isinstance(tile_plan, dict):
        if tile_plan.get("requires_tiling") is True:
            return False, "requires_tiling_not_implemented"
        if tile_plan.get("requires_host_aggregation") is True:
            return False, "requires_host_aggregation_not_representable"
    status = getattr(preparation_result, "status", None)
    if status not in {"prepared", "simplepim_unavailable"}:
        return False, f"non_bridgeable_preparation_status:{status}"
    if (
        getattr(preparation_result, "left_conversion", None) is None
        or getattr(preparation_result, "right_conversion", None) is None
    ):
        return False, "conversion_records_missing"
    if (
        getattr(preparation_result, "left_matrix_shape", None) is None
        or getattr(preparation_result, "right_matrix_shape", None) is None
    ):
        return False, "matrix_shapes_missing"
    return True, None


def dense_bridge_backend_manifest_eligibility(
    preparation_result: object | None,
    backend: str,
) -> tuple[bool, str | None]:
    if backend not in {"upmem_sdk_simulator_dense", "upmem_sdk_hardware_dense"}:
        return dense_bridge_manifest_eligibility(preparation_result)
    common = _backend_manifest_common_rejection(preparation_result)
    if common is not None:
        return False, common
    status = getattr(preparation_result, "status", None)
    tile_plan = getattr(preparation_result, "tile_plan", None)
    tile_plan = tile_plan if isinstance(tile_plan, dict) else {}
    if (
        status in {"prepared", "simplepim_unavailable"}
        and tile_plan.get("requires_tiling") is not True
    ):
        return True, None
    if (
        status != "requires_executable_tiling_not_implemented"
        or tile_plan.get("requires_tiling") is not True
    ):
        return False, f"non_bridgeable_preparation_status:{status}"
    left_conversion = to_jsonable(getattr(preparation_result, "left_conversion", None))
    right_conversion = to_jsonable(
        getattr(preparation_result, "right_conversion", None)
    )
    if (
        left_conversion.get("representation") != "real"
        or right_conversion.get("representation") != "real"
    ):
        return False, "complex_l2_not_implemented"
    l2_plan = plan_l2_tiled_execution(
        int(getattr(preparation_result, "gemm_m", 0) or 0),
        int(getattr(preparation_result, "gemm_k", 0) or 0),
        int(getattr(preparation_result, "gemm_n", 0) or 0),
    )
    if not l2_plan.supported:
        return False, str(l2_plan.reason or "unsupported_l2_tile_plan")
    return True, None


def _backend_manifest_common_rejection(preparation_result: object | None) -> str | None:
    if preparation_result is None:
        return "dense_preparation_missing"
    if getattr(preparation_result, "prepared_operands", None) is None:
        return "prepared_operands_missing"
    if getattr(preparation_result, "tile_plan", None) is None:
        return "tile_plan_missing"
    if (
        getattr(preparation_result, "left_conversion", None) is None
        or getattr(preparation_result, "right_conversion", None) is None
    ):
        return "conversion_records_missing"
    if (
        getattr(preparation_result, "left_matrix_shape", None) is None
        or getattr(preparation_result, "right_matrix_shape", None) is None
    ):
        return "matrix_shapes_missing"
    return None


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
        "upmem_sdk_simulator_dense": DenseBridgeBackendIdentity(
            backend_id="upmem_sdk_simulator_dense",
            display_name="UPMEM SDK Simulator Dense Bridge",
            backend_kind="upmem_sdk_simulator_dense",
            execution_mode="external_process",
            external_command_capable=True,
            implemented=True,
            description=(
                "Executes task-level L1 direct and L2 WRAM-tiled dense bridge GEMMs "
                "through the UPMEM SDK simulator subset. This is not a SimplePIM API GEMM primitive."
            ),
        ),
        "upmem_sdk_hardware_dense": DenseBridgeBackendIdentity(
            backend_id="upmem_sdk_hardware_dense",
            display_name="UPMEM SDK Hardware Dense Bridge",
            backend_kind="upmem_sdk_hardware_dense",
            execution_mode="external_process",
            external_command_capable=True,
            implemented=True,
            description="Hardware-only UPMEM SDK dense int8 MVP; never falls back to a simulator or CPU executor.",
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

    if request.backend == "upmem_sdk_simulator_dense":
        return _execute_upmem_sdk_simulator_dense_bridge(request, identity)

    if request.backend == "upmem_sdk_hardware_dense":
        return _execute_upmem_sdk_hardware_dense_bridge(request, identity)

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
    output_blob_path = (
        input_manifest_path.parent / "outputs" / "mock_dequantized_output.npy"
    )
    try:
        manifest = read_dense_bridge_input_manifest(input_manifest_path)
        bridge_dir = input_manifest_path.parent
        left = np.load(
            _resolve_manifest_path(
                bridge_dir, manifest.operands["left"]["relative_path"]
            ),
            allow_pickle=False,
        )
        right = np.load(
            _resolve_manifest_path(
                bridge_dir, manifest.operands["right"]["relative_path"]
            ),
            allow_pickle=False,
        )
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

        expected = np.load(
            _resolve_manifest_path(bridge_dir, manifest.expected_output.relative_path),
            allow_pickle=False,
        )
        _validate_loaded_blob(
            expected, to_jsonable(manifest.expected_output), "expected_output"
        )
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
            metadata={
                "mock_means": "file boundary validation failed before any external command"
            },
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


def _execute_upmem_sdk_hardware_dense_bridge(
    request: DenseBridgeExecutionRequest,
    identity: DenseBridgeBackendIdentity,
) -> DenseBridgeExecutionResult:
    """Execute the physical-only runner and enforce its additive hardware contract."""
    started = time.perf_counter()
    input_path = request.input_manifest_path
    bridge_dir = input_path.parent
    output_path = bridge_dir / "output_manifest.json"
    environment = dict(
        os.environ if request.environment is None else request.environment
    )
    invocation: JsonDict = {
        "backend_id": identity.backend_id,
        "backend_family": "upmem_sdk",
        "target": "hardware",
        "simplepim_api_used": False,
        "hardware_kernel_executed": False,
        "external_command_executed": False,
    }
    manifest: DenseBridgeInputManifest | None = None

    try:
        manifest = read_dense_bridge_input_manifest(input_path)
        _validate_bridge_input_files(manifest, bridge_dir)
        from quantum_bench.targets.upmem.hardware_mvp import (
            validate_hardware_mvp_manifest,
        )

        validate_hardware_mvp_manifest(
            json.loads(input_path.read_text(encoding="utf-8"))
        )
    except Exception as exc:
        return _write_hardware_bridge_failure(
            input_path,
            output_path,
            identity,
            manifest,
            invocation,
            "hardware_profile_violation",
            str(exc),
            started,
        )

    if environment.get("UPMEM_ALLOW_PHYSICAL_HARDWARE") != "1":
        return _write_hardware_bridge_failure(
            input_path,
            output_path,
            identity,
            manifest,
            invocation,
            "hardware_opt_in_missing",
            "UPMEM_ALLOW_PHYSICAL_HARDWARE=1 is required",
            started,
        )
    if not request.execute_external:
        return _write_hardware_bridge_failure(
            input_path,
            output_path,
            identity,
            manifest,
            invocation,
            "hardware_profile_violation",
            "execute_external=True is required",
            started,
        )
    if environment.get("DPU_BACKEND"):
        return _write_hardware_bridge_failure(
            input_path,
            output_path,
            identity,
            manifest,
            invocation,
            "hardware_profile_violation",
            "DPU_BACKEND must not be inherited for hardware execution",
            started,
        )
    if environment.get("UPMEM_DENSE_HARDWARE_RUNNER"):
        return _write_hardware_bridge_failure(
            input_path,
            output_path,
            identity,
            manifest,
            invocation,
            "hardware_profile_violation",
            "UPMEM_DENSE_HARDWARE_RUNNER override is forbidden by hardware_mvp_l1_v2",
            started,
        )

    runner = _upmem_sdk_hardware_runner_path()
    timeout = _upmem_sdk_hardware_timeout()
    command = [
        sys.executable,
        str(runner),
        "--input-manifest",
        input_path.name,
        "--output-manifest",
        output_path.name,
        "--backend-id",
        identity.backend_id,
        "--timeout-seconds",
        str(timeout),
    ]
    invocation.update(
        {
            "command_args": tuple(command),
            "runner_path": str(runner),
            "timeout_seconds": timeout,
        }
    )
    child_env = dict(environment)
    invocation["sanitized_environment"] = {
        "UPMEM_PROFILE_present": "UPMEM_PROFILE" in child_env,
        "UPMEM_PROFILE_BASE_present": "UPMEM_PROFILE_BASE" in child_env,
        "DPU_BACKEND_present": "DPU_BACKEND" in child_env,
    }
    for name in ("UPMEM_PROFILE", "UPMEM_PROFILE_BASE", "DPU_BACKEND"):
        child_env.pop(name, None)
    try:
        completed = subprocess.run(
            command,
            cwd=bridge_dir,
            env=child_env,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout + 2.0,
        )
        invocation.update(
            {
                "external_command_executed": True,
                "returncode": int(completed.returncode),
                "stdout_snippet": _bounded_snippet(completed.stdout),
                "stderr_snippet": _bounded_snippet(completed.stderr),
            }
        )
    except subprocess.TimeoutExpired:
        invocation.update({"external_command_executed": True, "timed_out": True})
        return _write_hardware_bridge_failure(
            input_path,
            output_path,
            identity,
            manifest,
            invocation,
            "kernel_timeout",
            f"hardware runner timed out after {timeout + 2.0} seconds",
            started,
        )

    try:
        output = read_dense_bridge_output_manifest(output_path)
        payload = json.loads(output_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return _write_hardware_bridge_failure(
            input_path,
            output_path,
            identity,
            manifest,
            invocation,
            "output_manifest_failed",
            str(exc),
            started,
        )
    if completed.returncode != 0 and output.status == "upmem_sdk_hardware_executed":
        failure_reason = _hardware_failure_reason(
            output.metadata, output.error, completed.stdout, completed.stderr
        )
        return _write_hardware_bridge_failure(
            input_path,
            output_path,
            identity,
            manifest,
            invocation,
            failure_reason or "output_validation_failed",
            f"hardware runner exited with code {completed.returncode}",
            started,
            output,
        )
    if output.status == "upmem_sdk_hardware_executed":
        try:
            _validate_hardware_output_contract(payload, output, bridge_dir)
        except Exception as exc:
            failure_reason = _hardware_failure_reason(output.metadata, str(exc))
            return _write_hardware_bridge_failure(
                input_path,
                output_path,
                identity,
                manifest,
                invocation,
                failure_reason or "output_validation_failed",
                str(exc),
                started,
                output,
            )
        invocation["hardware_kernel_executed"] = True
        return _execution_result(
            input_path,
            output_path,
            _resolve_manifest_path(bridge_dir, output.output_blob.relative_path)
            if output.output_blob
            else None,
            identity.backend_id,
            identity,
            "upmem_sdk_hardware_executed",
            "upmem_sdk_hardware_executed",
            None,
            None,
            output,
            invocation,
            external_command_executed=True,
            execution_implemented=True,
            metadata=dict(output.metadata),
        )
    reason = _hardware_failure_reason(
        output.metadata, output.error, output.status
    ) or str(output.metadata.get("reason") or output.status)
    return _execution_result(
        input_path,
        output_path,
        None,
        identity.backend_id,
        identity,
        output.status,
        reason,
        output.error,
        str(output.metadata.get("error_type"))
        if output.metadata.get("error_type")
        else None,
        output,
        invocation,
        external_command_executed=True,
        execution_implemented=output.execution_implemented,
        metadata=dict(output.metadata),
    )


def _write_hardware_bridge_failure(
    input_path: Path,
    output_path: Path,
    identity: DenseBridgeBackendIdentity,
    manifest: DenseBridgeInputManifest | None,
    invocation: JsonDict,
    reason: str,
    error: str,
    started: float,
    output: DenseBridgeOutputManifest | None = None,
) -> DenseBridgeExecutionResult:
    if output is None:
        failed = _nonexecuted_output_manifest(
            identity.backend_id,
            "failed",
            reason,
            error,
            reason,
            input_path,
            manifest,
            invocation,
            float(time.perf_counter() - started),
            external_command_executed=bool(invocation.get("external_command_executed")),
            metadata_extra={
                "backend_family": "upmem_sdk",
                "target": "hardware",
                "hardware_kernel_executed": False,
            },
        )
    else:
        failed = DenseBridgeOutputManifest(
            schema_version=output.schema_version,
            bridge_id=output.bridge_id,
            manifest_kind=output.manifest_kind,
            backend=identity.backend_id,
            status="failed",
            input_manifest=output.input_manifest,
            route_id=output.route_id,
            task_id=output.task_id,
            output_blob=output.output_blob,
            accumulator_blob=output.accumulator_blob,
            validation_metrics={
                **dict(output.validation_metrics),
                "passed": False,
                "exact_integer_passed": False,
            },
            compute_time_s=output.compute_time_s,
            write_time_s=output.write_time_s,
            total_time_s=float(time.perf_counter() - started),
            external_command_executed=bool(invocation.get("external_command_executed")),
            execution_implemented=output.execution_implemented,
            error=error,
            metadata={
                **dict(output.metadata),
                "reason": reason,
                "error_type": reason,
                "hardware_stage": reason,
                "hardware_kernel_executed": False,
                "simulator_kernel_executed": False,
                "cpu_fallback_used": False,
            },
        )
    write_json(output_path, failed)
    return _execution_result(
        input_path,
        output_path,
        None,
        identity.backend_id,
        identity,
        "failed",
        reason,
        error,
        reason,
        failed,
        invocation,
        external_command_executed=bool(invocation.get("external_command_executed")),
        metadata=dict(failed.metadata),
    )


def _validate_hardware_output_contract(
    payload: JsonDict, output: DenseBridgeOutputManifest, bridge_dir: Path
) -> None:
    required = {
        "hardware_status_json",
        "raw_accumulator_crop",
        "cpu_reference",
        "hashes",
        "application_visible_transfer_bytes",
        "timing_labels",
        "speedup_claims",
    }
    missing = sorted(required.difference(output.metadata))
    if missing:
        raise ValueError(
            f"hardware output metadata missing required fields: {', '.join(missing)}"
        )
    if (
        output.backend != "upmem_sdk_hardware_dense"
        or output.status != "upmem_sdk_hardware_executed"
    ):
        raise ValueError("hardware output identity/status mismatch")
    if output.metadata.get("sdk_allocation_profile") != HARDWARE_MVP_SDK_ALLOCATION_PROFILE:
        raise ValueError("hardware output does not prove sdk allocation profile backend=hw")
    if output.output_blob is None or output.accumulator_blob is None:
        raise ValueError("hardware output and accumulator blobs are required")
    if (
        output.validation_metrics.get("exact_integer_passed") is not True
        or output.validation_metrics.get("passed") is not True
    ):
        raise ValueError("exact integer validation did not pass")
    if output.metadata["speedup_claims"] is not False:
        raise ValueError("hardware bring-up must not claim speedup")
    if output.metadata["timing_labels"] != "hardware_bringup_functionality_only":
        raise ValueError("hardware timing label is not functionality-only")
    if not isinstance(output.metadata["hardware_status_json"], dict):
        raise ValueError("hardware_status_json must be an object")
    status = output.metadata["hardware_status_json"]
    if (
        status.get("success") is not True
        or status.get("failure_stage") is not None
        or status.get("allocation_profile")
        != HARDWARE_MVP_SDK_ALLOCATION_PROFILE
        or int(status.get("requested_dpus", 0)) != 1
        or int(status.get("allocated_dpus", 0)) != 1
        or int(status.get("tasklets", 0)) != 1
    ):
        raise ValueError(
            "hardware status does not prove one-DPU/one-tasklet successful execution"
        )
    if output.metadata.get("hardware_kernel_executed") is not True:
        raise ValueError("hardware output does not prove native kernel execution")
    if output.metadata.get("simulator_kernel_executed") is not False:
        raise ValueError(
            "hardware output must state that the simulator did not execute"
        )
    if output.metadata.get("cpu_fallback_used") is not False:
        raise ValueError("hardware output must state that no CPU fallback was used")
    hashes = output.metadata["hashes"]
    if not isinstance(hashes, dict) or not {
        "left",
        "right",
        "accumulator",
        "output",
    }.issubset(hashes):
        raise ValueError(
            "hardware hashes must cover both operands, accumulator, and output"
        )
    transfers = output.metadata["application_visible_transfer_bytes"]
    if not isinstance(transfers, dict) or any(
        key not in transfers for key in ("h2d", "d2h", "total")
    ):
        raise ValueError("application-visible H2D/D2H accounting is incomplete")
    if int(transfers["h2d"]) + int(transfers["d2h"]) != int(transfers["total"]):
        raise ValueError("application-visible transfer total is inconsistent")
    raw_path = _resolve_manifest_path(
        bridge_dir, str(output.accumulator_blob["relative_path"])
    )
    raw = np.load(raw_path, allow_pickle=False)
    _validate_loaded_blob(raw, output.accumulator_blob, "hardware_accumulator")
    if np.dtype(output.accumulator_blob["dtype"]) != np.dtype("<i4"):
        raise ValueError("hardware accumulator must be little-endian int32")
    try:
        input_payload = json.loads(
            (bridge_dir / output.input_manifest).read_text(encoding="utf-8")
        )
        reference_info = (input_payload.get("metadata") or {}).get(
            "expected_accumulator"
        )
        if not isinstance(reference_info, dict):
            raise ValueError("retained exact CPU accumulator reference is missing")
        expected_path = _resolve_manifest_path(
            bridge_dir, str(reference_info["relative_path"])
        )
        expected = np.load(expected_path, allow_pickle=False).astype("<i4", copy=False)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"hardware exact CPU reference cannot be loaded: {exc}"
        ) from exc
    if not np.array_equal(raw.astype("<i4", copy=False), expected):
        raise ValueError(
            "hardware accumulator does not match the retained exact CPU reference"
        )


def _hardware_failure_reason(
    metadata: Mapping[str, Any], *values: object
) -> str | None:
    """Map SDK invalid-profile reports to the stable hardware failure stage."""
    text = " ".join(str(value) for value in values if value is not None).lower()
    if "invalid profile" in text or "invalid dpu profile" in text:
        return "hardware_profile_violation"
    if metadata.get("hardware_stage") == "hardware_profile_violation":
        return "hardware_profile_violation"
    return None


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

    probe = probe_simplepim(
        env=request.environment if request.environment is not None else os.environ
    )
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
        error = (
            None
            if stub_path is None
            else f"Configured stub path does not exist: {stub_path}"
        )
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
        reason = str(
            output_manifest.metadata.get("reason") or "simplepim_external_stub_failed"
        )
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


def _execute_upmem_sdk_simulator_dense_bridge(
    request: DenseBridgeExecutionRequest,
    identity: DenseBridgeBackendIdentity,
) -> DenseBridgeExecutionResult:
    started = time.perf_counter()
    input_manifest_path = request.input_manifest_path
    bridge_dir = input_manifest_path.parent
    output_manifest_path = bridge_dir / "output_manifest.json"
    manifest: DenseBridgeInputManifest | None = None
    invocation_metadata: JsonDict = _upmem_sdk_simulator_invocation_metadata(
        input_manifest_path=input_manifest_path,
        env=request.environment,
    )

    try:
        manifest = read_dense_bridge_input_manifest(input_manifest_path)
    except Exception as exc:
        output_manifest = _nonexecuted_output_manifest(
            backend=identity.backend_id,
            status="failed",
            reason="input_manifest_invalid",
            error=str(exc),
            error_type="input_manifest_invalid",
            input_manifest_path=input_manifest_path,
            manifest=None,
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
            reason="input_manifest_invalid",
            error=str(exc),
            error_type="input_manifest_invalid",
            output_manifest=output_manifest,
            invocation_metadata=invocation_metadata,
        )

    try:
        _validate_bridge_input_files(manifest, bridge_dir)
    except Exception as exc:
        output_manifest = _nonexecuted_output_manifest(
            backend=identity.backend_id,
            status="failed",
            reason="operand_blob_invalid",
            error=str(exc),
            error_type="operand_blob_invalid",
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
            reason="operand_blob_invalid",
            error=str(exc),
            error_type="operand_blob_invalid",
            output_manifest=output_manifest,
            invocation_metadata=invocation_metadata,
        )

    static_rejection = _upmem_sdk_simulator_static_rejection(
        manifest, request.environment
    )
    if static_rejection is not None:
        reason, error = static_rejection
        output_manifest = _nonexecuted_output_manifest(
            backend=identity.backend_id,
            status="unsupported",
            reason=reason,
            error=error,
            error_type=reason,
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
            execution_status="unsupported",
            reason=reason,
            error=error,
            error_type=reason,
            output_manifest=output_manifest,
            invocation_metadata=invocation_metadata,
        )

    if not request.execute_external:
        output_manifest = _nonexecuted_output_manifest(
            backend=identity.backend_id,
            status="not_implemented",
            reason="upmem_sdk_simulator_execution_disabled",
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
            reason="upmem_sdk_simulator_execution_disabled",
            error=None,
            error_type=None,
            output_manifest=output_manifest,
            invocation_metadata=invocation_metadata,
        )

    missing_tools = _missing_upmem_sdk_simulator_tools(request.environment)
    runner_path = _upmem_sdk_dense_runner_path()
    if missing_tools or not runner_path.exists():
        error = None
        if missing_tools:
            error = f"Missing required UPMEM SDK simulator tools: {', '.join(missing_tools)}"
        elif not runner_path.exists():
            error = f"UPMEM SDK dense runner not found: {runner_path}"
        output_manifest = _nonexecuted_output_manifest(
            backend=identity.backend_id,
            status="skipped",
            reason="upmem_sdk_simulator_unavailable",
            error=error,
            error_type=None,
            input_manifest_path=input_manifest_path,
            manifest=manifest,
            invocation_metadata=invocation_metadata,
            total_time_s=float(time.perf_counter() - started),
            metadata_extra={
                "backend_family": "upmem_sdk",
                "simplepim_api_used": False,
                "simplepim_bridge_lane": True,
                "target": "simulator",
                "upmem_dpu_program_executed": False,
                "simulator_kernel_executed": False,
                "hardware_kernel_executed": False,
                "missing_tools": missing_tools,
            },
        )
        write_json(output_manifest_path, output_manifest)
        return _execution_result(
            input_manifest_path=input_manifest_path,
            output_manifest_path=output_manifest_path,
            output_blob_path=None,
            backend_id=identity.backend_id,
            backend_identity=identity,
            execution_status="skipped",
            reason="upmem_sdk_simulator_unavailable",
            error=error,
            error_type=None,
            output_manifest=output_manifest,
            invocation_metadata=invocation_metadata,
            metadata={
                "backend_family": "upmem_sdk",
                "simplepim_api_used": False,
                "simplepim_bridge_lane": True,
            },
        )

    timeout_seconds = _upmem_sdk_simulator_timeout(request.environment)
    command = [
        sys.executable,
        str(runner_path),
        "--input-manifest",
        input_manifest_path.name,
        "--output-manifest",
        output_manifest_path.name,
        "--backend-id",
        identity.backend_id,
        "--target",
        "simulator",
        "--timeout-seconds",
        str(timeout_seconds),
    ]
    invocation_metadata["command_args"] = tuple(command)
    invocation_metadata["external_command_executed"] = True
    completed: subprocess.CompletedProcess[str] | None = None
    try:
        completed = subprocess.run(
            command,
            cwd=bridge_dir,
            env={
                **dict(
                    os.environ if request.environment is None else request.environment
                ),
                "DPU_BACKEND": "simulator",
            },
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_seconds + 5.0,
        )
    except subprocess.TimeoutExpired as exc:
        error = f"UPMEM SDK simulator dense runner timed out after {timeout_seconds + 5.0} seconds"
        output_manifest = _nonexecuted_output_manifest(
            backend=identity.backend_id,
            status="failed",
            reason="runner_execution_failed",
            error=error,
            error_type="runner_execution_failed",
            input_manifest_path=input_manifest_path,
            manifest=manifest,
            invocation_metadata=invocation_metadata,
            total_time_s=float(time.perf_counter() - started),
            external_command_executed=True,
            metadata_extra={
                "backend_family": "upmem_sdk",
                "simplepim_api_used": False,
                "simplepim_bridge_lane": True,
                "target": "simulator",
                "upmem_dpu_program_executed": False,
                "simulator_kernel_executed": False,
                "hardware_kernel_executed": False,
                "stdout_snippet": _bounded_snippet(_decode_timeout_output(exc.stdout)),
                "stderr_snippet": _bounded_snippet(_decode_timeout_output(exc.stderr)),
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
            reason="runner_execution_failed",
            error=error,
            error_type="runner_execution_failed",
            output_manifest=output_manifest,
            invocation_metadata=invocation_metadata,
            external_command_executed=True,
            metadata={"backend_family": "upmem_sdk"},
        )

    try:
        output_manifest = read_dense_bridge_output_manifest(output_manifest_path)
    except Exception as exc:
        reason = (
            "runner_execution_failed"
            if completed.returncode != 0
            else "output_manifest_invalid"
        )
        output_manifest = _nonexecuted_output_manifest(
            backend=identity.backend_id,
            status="failed",
            reason=reason,
            error=str(exc),
            error_type=reason,
            input_manifest_path=input_manifest_path,
            manifest=manifest,
            invocation_metadata=invocation_metadata,
            total_time_s=float(time.perf_counter() - started),
            external_command_executed=True,
            metadata_extra={
                "backend_family": "upmem_sdk",
                "simplepim_api_used": False,
                "simplepim_bridge_lane": True,
                "target": "simulator",
                "upmem_dpu_program_executed": False,
                "simulator_kernel_executed": False,
                "hardware_kernel_executed": False,
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
            reason=reason,
            error=str(exc),
            error_type=reason,
            output_manifest=output_manifest,
            invocation_metadata=invocation_metadata,
            external_command_executed=True,
            metadata={"backend_family": "upmem_sdk"},
        )

    if (
        completed.returncode != 0
        and output_manifest.status == "upmem_sdk_simulator_executed"
    ):
        reason = "runner_execution_failed"
        error = (
            f"UPMEM SDK simulator dense runner exited with code {completed.returncode}"
        )
        output_manifest = _nonexecuted_output_manifest(
            backend=identity.backend_id,
            status="failed",
            reason=reason,
            error=error,
            error_type=reason,
            input_manifest_path=input_manifest_path,
            manifest=manifest,
            invocation_metadata=invocation_metadata,
            total_time_s=float(time.perf_counter() - started),
            external_command_executed=True,
            metadata_extra={
                "backend_family": "upmem_sdk",
                "simplepim_api_used": False,
                "simplepim_bridge_lane": True,
                "target": "simulator",
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
            reason=reason,
            error=error,
            error_type=reason,
            output_manifest=output_manifest,
            invocation_metadata=invocation_metadata,
            external_command_executed=True,
            metadata={"backend_family": "upmem_sdk"},
        )

    if output_manifest.status in {
        "failed",
        "unsupported",
        "skipped",
        "not_implemented",
    }:
        reason = str(output_manifest.metadata.get("reason") or output_manifest.status)
        error_type = output_manifest.metadata.get("error_type")
        return _execution_result(
            input_manifest_path=input_manifest_path,
            output_manifest_path=output_manifest_path,
            output_blob_path=None,
            backend_id=identity.backend_id,
            backend_identity=identity,
            execution_status=output_manifest.status,
            reason=reason,
            error=output_manifest.error,
            error_type=str(error_type) if error_type else None,
            output_manifest=output_manifest,
            invocation_metadata=invocation_metadata,
            external_command_executed=output_manifest.external_command_executed,
            execution_implemented=output_manifest.execution_implemented,
            metadata=dict(output_manifest.metadata),
        )

    if output_manifest.status != "upmem_sdk_simulator_executed":
        reason = "output_manifest_invalid"
        error = (
            f"Unexpected UPMEM SDK simulator output status: {output_manifest.status}"
        )
        output_manifest = _nonexecuted_output_manifest(
            backend=identity.backend_id,
            status="failed",
            reason=reason,
            error=error,
            error_type=reason,
            input_manifest_path=input_manifest_path,
            manifest=manifest,
            invocation_metadata=invocation_metadata,
            total_time_s=float(time.perf_counter() - started),
            external_command_executed=True,
            metadata_extra={"backend_family": "upmem_sdk"},
        )
        write_json(output_manifest_path, output_manifest)
        return _execution_result(
            input_manifest_path=input_manifest_path,
            output_manifest_path=output_manifest_path,
            output_blob_path=None,
            backend_id=identity.backend_id,
            backend_identity=identity,
            execution_status="failed",
            reason=reason,
            error=error,
            error_type=reason,
            output_manifest=output_manifest,
            invocation_metadata=invocation_metadata,
            external_command_executed=True,
            metadata={"backend_family": "upmem_sdk"},
        )

    output_blob_path: Path | None = None
    if output_manifest.output_blob is None:
        reason = "output_blob_missing"
        error = "UPMEM SDK simulator output manifest did not include output_blob"
    else:
        output_blob_path = _resolve_manifest_path(
            bridge_dir, output_manifest.output_blob.relative_path
        )
        if not output_blob_path.exists():
            reason = "output_blob_missing"
            error = f"UPMEM SDK simulator output blob is missing: {output_manifest.output_blob.relative_path}"
        else:
            try:
                output_array = np.load(output_blob_path, allow_pickle=False)
                _validate_loaded_blob(
                    output_array,
                    to_jsonable(output_manifest.output_blob),
                    "upmem_sdk_simulator_output",
                )
                reason = None
                error = None
            except Exception as exc:
                reason = "output_manifest_invalid"
                error = str(exc)
    if reason is not None:
        failed_manifest = _nonexecuted_output_manifest(
            backend=identity.backend_id,
            status="failed",
            reason=reason,
            error=error,
            error_type=reason,
            input_manifest_path=input_manifest_path,
            manifest=manifest,
            invocation_metadata=invocation_metadata,
            total_time_s=float(time.perf_counter() - started),
            external_command_executed=True,
            metadata_extra=dict(output_manifest.metadata),
        )
        write_json(output_manifest_path, failed_manifest)
        return _execution_result(
            input_manifest_path=input_manifest_path,
            output_manifest_path=output_manifest_path,
            output_blob_path=None,
            backend_id=identity.backend_id,
            backend_identity=identity,
            execution_status="failed",
            reason=reason,
            error=error,
            error_type=reason,
            output_manifest=failed_manifest,
            invocation_metadata=invocation_metadata,
            external_command_executed=True,
            metadata=dict(failed_manifest.metadata),
        )

    if output_manifest.validation_metrics.get("passed") is False:
        reason = "validation_failed"
        return _execution_result(
            input_manifest_path=input_manifest_path,
            output_manifest_path=output_manifest_path,
            output_blob_path=output_blob_path,
            backend_id=identity.backend_id,
            backend_identity=identity,
            execution_status="failed",
            reason=reason,
            error=output_manifest.error
            or "UPMEM SDK simulator output did not pass validation",
            error_type=reason,
            output_manifest=output_manifest,
            invocation_metadata=invocation_metadata,
            external_command_executed=output_manifest.external_command_executed,
            execution_implemented=output_manifest.execution_implemented,
            metadata=dict(output_manifest.metadata),
        )

    return _execution_result(
        input_manifest_path=input_manifest_path,
        output_manifest_path=output_manifest_path,
        output_blob_path=output_blob_path,
        backend_id=identity.backend_id,
        backend_identity=identity,
        execution_status="upmem_sdk_simulator_executed",
        reason="upmem_sdk_simulator_executed",
        error=None,
        error_type=None,
        output_manifest=output_manifest,
        invocation_metadata=invocation_metadata,
        external_command_executed=output_manifest.external_command_executed,
        execution_implemented=output_manifest.execution_implemented,
        metadata={
            **dict(output_manifest.metadata),
            "simplepim_or_native_execution_implemented": True,
        },
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
        input_manifest=_relative_result_path(input_manifest_path, bridge_dir)
        or input_manifest_path.name,
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
    configured_environment_keys = tuple(
        key for key in environment_keys if environment.get(key)
    )
    metadata: JsonDict = {
        "backend_id": "simplepim_external",
        "working_directory": ".",
        "input_manifest_path": _relative_result_path(
            input_manifest_path, input_manifest_path.parent
        )
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
    metadata["simplepim_available"] = bool(
        probe_payload.get("simplepim_available", False)
    )
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
    configured_environment_keys = tuple(
        key for key in environment_keys if environment.get(key)
    )
    metadata: JsonDict = {
        "backend_id": "simplepim_external_stub",
        "working_directory": ".",
        "input_manifest_path": _relative_result_path(
            input_manifest_path, input_manifest_path.parent
        )
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


def _upmem_sdk_dense_runner_path() -> Path:
    root_dir = Path(__file__).resolve().parents[4]
    return root_dir / "native" / "upmem" / "simplepim" / "upmem_sdk_dense_runner.py"


def _upmem_sdk_hardware_runner_path() -> Path:
    root_dir = Path(__file__).resolve().parents[4]
    return (
        root_dir
        / "native"
        / "upmem"
        / "simplepim"
        / "upmem_sdk_dense_hardware_mvp_runner.py"
    )


def _upmem_sdk_hardware_timeout() -> float:
    """The hardware MVP timeout is profile-owned, never environment-owned."""

    return 30.0


def _upmem_sdk_simulator_invocation_metadata(
    input_manifest_path: Path,
    env: Mapping[str, str] | None,
) -> JsonDict:
    environment = env if env is not None else os.environ
    runner_path = _upmem_sdk_dense_runner_path()
    timeout_seconds = _upmem_sdk_simulator_timeout(environment)
    metadata: JsonDict = {
        "backend_id": "upmem_sdk_simulator_dense",
        "backend_family": "upmem_sdk",
        "simplepim_api_used": False,
        "simplepim_bridge_lane": True,
        "working_directory": ".",
        "input_manifest_path": _relative_result_path(
            input_manifest_path, input_manifest_path.parent
        )
        or input_manifest_path.name,
        "expected_output_manifest_path": "output_manifest.json",
        "expected_output_blob_path": "outputs/upmem_sdk_simulator_output.npy",
        "target": "simulator",
        "environment_keys": (
            "UPMEM_HOME",
            "UPMEM_DENSE_SIM_MAX_DIM",
            "UPMEM_DENSE_SIM_TIMEOUT_SECONDS",
            "UPMEM_DENSE_L2_MAX_HOST_BLOB_BYTES",
            "UPMEM_DENSE_L2_NATIVE_MAX_DIM",
        ),
        "configured_environment_keys": tuple(
            key
            for key in (
                "UPMEM_HOME",
                "UPMEM_DENSE_SIM_MAX_DIM",
                "UPMEM_DENSE_SIM_TIMEOUT_SECONDS",
                "UPMEM_DENSE_L2_MAX_HOST_BLOB_BYTES",
                "UPMEM_DENSE_L2_NATIVE_MAX_DIM",
            )
            if environment.get(key)
        ),
        "blob_format": "npy",
        "native_buffer_layout": "row_major_padded",
        "native_buffer_stride": "max_dim",
        "stride_model": "explicit_padded_stride_v1",
        "native_int32_output_dtype": "<i4",
        "external_command_executed": False,
        "timeout_seconds": timeout_seconds,
        "runner_path": str(runner_path),
    }
    metadata["command_args"] = (
        sys.executable,
        str(runner_path),
        "--input-manifest",
        input_manifest_path.name,
        "--output-manifest",
        "output_manifest.json",
        "--backend-id",
        "upmem_sdk_simulator_dense",
        "--target",
        "simulator",
        "--timeout-seconds",
        str(timeout_seconds),
    )
    return metadata


def _missing_upmem_sdk_simulator_tools(
    env: Mapping[str, str] | None,
) -> tuple[str, ...]:
    environment = env if env is not None else os.environ
    required = {
        "make": shutil.which("make"),
        "dpu-upmem-dpurte-clang": _find_upmem_tool(
            "dpu-upmem-dpurte-clang", environment
        ),
        "dpu-pkg-config": _find_upmem_tool("dpu-pkg-config", environment),
    }
    return tuple(name for name, path in required.items() if path is None)


def _find_upmem_tool(name: str, env: Mapping[str, str]) -> str | None:
    upmem_home = env.get("UPMEM_HOME")
    if upmem_home:
        candidate = Path(upmem_home).expanduser() / "bin" / name
        if candidate.exists():
            return str(candidate)
    return shutil.which(name)


def _upmem_sdk_simulator_timeout(env: Mapping[str, str] | None) -> float:
    environment = env if env is not None else os.environ
    raw = environment.get("UPMEM_DENSE_SIM_TIMEOUT_SECONDS")
    if not raw:
        return 30.0
    try:
        value = float(raw)
    except ValueError:
        return 30.0
    if value <= 0.0:
        return 30.0
    return value


def _upmem_sdk_simulator_static_rejection(
    manifest: DenseBridgeInputManifest,
    env: Mapping[str, str] | None,
) -> tuple[str, str] | None:
    tile_plan = manifest.tile_plan or {}
    left_deq = dict(manifest.dequantization.get("left", {}))
    right_deq = dict(manifest.dequantization.get("right", {}))
    if left_deq.get("route_dtype") != "int8" or right_deq.get("route_dtype") != "int8":
        return (
            "unsupported_dtype",
            "UPMEM SDK simulator dense backend supports int8 operands only",
        )
    dims = (manifest.gemm_m, manifest.gemm_k, manifest.gemm_n)
    if any(dim <= 0 for dim in dims):
        return ("unsupported_shape", f"Invalid GEMM dimensions: {dims}")

    left_meta = dict(manifest.operands.get("left", {}))
    right_meta = dict(manifest.operands.get("right", {}))
    left_shape = tuple(int(dim) for dim in left_meta.get("shape", ()))
    right_shape = tuple(int(dim) for dim in right_meta.get("shape", ()))
    left_rep = str(left_deq.get("representation"))
    right_rep = str(right_deq.get("representation"))
    if left_rep == "real" and right_rep == "real":
        if left_shape != (manifest.gemm_m, manifest.gemm_k) or right_shape != (
            manifest.gemm_k,
            manifest.gemm_n,
        ):
            return (
                "unsupported_shape",
                "Real operand shapes do not match GEMM dimensions",
            )
    elif (
        left_rep == "split_complex_real_imag" and right_rep == "split_complex_real_imag"
    ):
        if left_shape != (manifest.gemm_m, manifest.gemm_k, 2) or right_shape != (
            manifest.gemm_k,
            manifest.gemm_n,
            2,
        ):
            return (
                "unsupported_complex_layout",
                "Split-complex operand shapes do not match GEMM dimensions",
            )
    else:
        return (
            "unsupported_complex_layout",
            f"Unsupported operand representations: {left_rep}, {right_rep}",
        )

    if tile_plan.get("requires_tiling"):
        if left_rep != "real" or right_rep != "real":
            return (
                "complex_l2_not_implemented",
                "L2 tiled simulator backend supports real-valued operands only",
            )
        l2_plan = plan_l2_tiled_execution(
            manifest.gemm_m,
            manifest.gemm_k,
            manifest.gemm_n,
            max_l2_host_blob_bytes=_upmem_sdk_l2_max_host_blob_bytes(env),
            native_max_dim=_upmem_sdk_l2_native_max_dim(env),
        )
        if not l2_plan.supported:
            return (
                str(l2_plan.reason or "unsupported_l2_tile_plan"),
                "L2 tiled simulator backend does not support this GEMM shape",
            )
        return None

    max_dim = _upmem_sdk_simulator_max_dim(env)
    if max_dim is None:
        return (
            "unsupported_shape_for_initial_backend",
            "Invalid UPMEM_DENSE_SIM_MAX_DIM; expected a positive integer",
        )
    if any(dim > max_dim for dim in dims):
        return (
            "unsupported_shape_for_initial_backend",
            f"GEMM dimensions {dims} exceed initial backend max dim {max_dim}",
        )
    return None


def _upmem_sdk_simulator_max_dim(env: Mapping[str, str] | None) -> int | None:
    environment = env if env is not None else os.environ
    raw = environment.get("UPMEM_DENSE_SIM_MAX_DIM")
    if raw is None or not raw.strip():
        return 16
    try:
        value = int(raw)
    except ValueError:
        return None
    if value <= 0:
        return None
    return value


def _upmem_sdk_l2_max_host_blob_bytes(env: Mapping[str, str] | None) -> int:
    environment = env if env is not None else os.environ
    return _positive_int_env(
        environment, "UPMEM_DENSE_L2_MAX_HOST_BLOB_BYTES", UPMEM_L2_MAX_HOST_BLOB_BYTES
    )


def _upmem_sdk_l2_native_max_dim(env: Mapping[str, str] | None) -> int:
    environment = env if env is not None else os.environ
    return _positive_int_env(
        environment, "UPMEM_DENSE_L2_NATIVE_MAX_DIM", UPMEM_L2_NATIVE_MAX_DIM
    )


def _positive_int_env(env: Mapping[str, str], key: str, default: int) -> int:
    raw = env.get(key)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def _bounded_snippet(value: str | None, limit: int = 2000) -> str:
    if not value:
        return ""
    if len(value) <= limit:
        return value
    return value[:limit] + "...<truncated>"


def _decode_timeout_output(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _validate_bridge_input_files(
    manifest: DenseBridgeInputManifest, bridge_dir: Path
) -> None:
    left = np.load(
        _resolve_manifest_path(bridge_dir, manifest.operands["left"]["relative_path"]),
        allow_pickle=False,
    )
    right = np.load(
        _resolve_manifest_path(bridge_dir, manifest.operands["right"]["relative_path"]),
        allow_pickle=False,
    )
    expected = np.load(
        _resolve_manifest_path(bridge_dir, manifest.expected_output.relative_path),
        allow_pickle=False,
    )
    _validate_loaded_blob(left, manifest.operands["left"], "left")
    _validate_loaded_blob(right, manifest.operands["right"], "right")
    _validate_loaded_blob(
        expected, to_jsonable(manifest.expected_output), "expected_output"
    )


def _validate_preparation_result(
    preparation_result: "DenseTaskPreparationResult",
) -> None:
    if preparation_result.prepared_operands is None:
        raise ValueError("prepared_operands are required for dense bridge input")
    tile_plan = preparation_result.tile_plan or {}
    if preparation_result.status not in {
        "prepared",
        "simplepim_unavailable",
        "requires_executable_tiling_not_implemented",
    }:
        raise ValueError(
            f"Preparation status {preparation_result.status!r} cannot be bridged"
        )
    if (
        tile_plan.get("requires_tiling")
        and preparation_result.status != "requires_executable_tiling_not_implemented"
    ):
        raise ValueError(
            "Tiled dense bridge input must use requires_executable_tiling_not_implemented preparation status"
        )
    if (
        preparation_result.left_conversion is None
        or preparation_result.right_conversion is None
    ):
        raise ValueError(
            "Fixed-point conversion records are required for dense bridge input"
        )
    if (
        preparation_result.left_matrix_shape is None
        or preparation_result.right_matrix_shape is None
    ):
        raise ValueError("Matrix shapes are required for dense bridge input")
    if tile_plan.get("requires_tiling"):
        operands = preparation_result.prepared_operands
        host_blob_bytes = int(
            operands.left_quantized.nbytes
            + operands.right_quantized.nbytes
            + operands.dequantized_output.nbytes
        )
        if host_blob_bytes > UPMEM_L2_MAX_HOST_BLOB_BYTES:
            raise ValueError("unsupported_l2_blob_size")


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
    result = (array.astype(np.float64) - int(metadata["zero_point"])) * float(
        metadata["scale"]
    )
    representation = metadata["representation"]
    source_shape = tuple(int(dim) for dim in metadata["source_shape"])
    if representation == "split_complex_real_imag":
        expected_shape = (*source_shape, 2)
        if tuple(result.shape) != expected_shape:
            raise ValueError(
                f"Converted complex blob shape {result.shape} does not match {expected_shape}"
            )
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
    gemm_shape = tuple(
        int(output_shape[output_labels.index(label)]) for label in gemm_labels
    )
    tensor_output = np.asarray(matrix_output).reshape(gemm_shape)
    if gemm_labels == output_labels:
        return tensor_output
    axes = tuple(gemm_labels.index(label) for label in output_labels)
    return np.transpose(tensor_output, axes)


def _validate_loaded_blob(array: np.ndarray, metadata: JsonDict, role: str) -> None:
    expected_shape = tuple(int(dim) for dim in metadata["shape"])
    expected_dtype = np.dtype(str(metadata["dtype"]))
    if tuple(array.shape) != expected_shape:
        raise ValueError(
            f"{role} blob shape {array.shape} does not match manifest shape {expected_shape}"
        )
    if array.dtype != expected_dtype:
        raise ValueError(
            f"{role} blob dtype {array.dtype} does not match manifest dtype {expected_dtype}"
        )


def _resolve_manifest_path(base_dir: Path, relative_path: str) -> Path:
    rel = Path(relative_path)
    if rel.is_absolute() or ".." in rel.parts:
        raise ValueError(
            f"Bridge manifest path must be relative and stay inside the bridge directory: {relative_path}"
        )
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


def _require_pair_shape(
    value: tuple[int, int] | None, field_name: str
) -> tuple[int, int]:
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
