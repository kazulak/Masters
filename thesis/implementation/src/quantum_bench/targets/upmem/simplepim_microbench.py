from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from quantum_bench.core.records import ContractionTask, JsonDict, to_jsonable
from quantum_bench.formats import FixedPointSpec, conversion_error_metrics, dequantize_fixed_point, quantize_fixed_point
from quantum_bench.targets.upmem.schedule import estimate_dense_task
from quantum_bench.targets.upmem.simplepim import SimplePimProbeResult, probe_simplepim


SIMPLEPIM_DENSE_MICROBENCH_SCHEMA_VERSION = "simplepim_dense_microbench_v1"
SIMPLEPIM_DENSE_MICROBENCH_ID = "simplepim_dense_gemm_dry_run"

SimplePimDenseMicrobenchStatus = Literal[
    "skipped",
    "configured_but_unverified",
    "ready",
    "not_implemented",
    "executed",
    "failed",
]


@dataclass(frozen=True)
class SimplePimDenseMicrobenchInput:
    gemm_m: int
    gemm_k: int
    gemm_n: int
    route_id: str = "dense_gemm"
    task_id: str = "simplepim_dense_gemm"
    source_dtype: str = "float32"
    route_dtype: str = "int8"
    seed: int = 0
    distribution: str = "standard_normal"
    dry_run: bool = True


@dataclass(frozen=True)
class SimplePimDenseMicrobenchResult:
    schema_version: str
    microbench_id: str
    input: SimplePimDenseMicrobenchInput
    status: SimplePimDenseMicrobenchStatus
    skip_reason: str | None
    error: str | None
    simplepim_probe: JsonDict
    fixed_point_spec: JsonDict
    tile_plan: JsonDict
    upmem_task_estimate: JsonDict
    input_shapes: tuple[tuple[int, ...], tuple[int, ...]]
    output_shape: tuple[int, ...]
    conversion_records: JsonDict
    conversion_time_s: float
    reference_time_s: float
    dequantization_time_s: float
    kernel_time_s: float | None
    host_aggregation_time_s: float | None
    total_time_s: float
    host_to_dpu_bytes: int
    dpu_to_host_bytes: int
    mram_to_wram_bytes: int
    validation_metrics: JsonDict
    external_command_executed: bool
    execution_implemented: bool
    metadata: JsonDict = field(default_factory=dict)

    def to_json_dict(self) -> JsonDict:
        return to_jsonable(self)


def make_simplepim_dense_microbench_input(
    gemm_m: int,
    gemm_k: int,
    gemm_n: int,
    *,
    route_dtype: str = "int8",
    source_dtype: str = "float32",
    seed: int = 0,
    task_id: str = "simplepim_dense_gemm",
    dry_run: bool = True,
) -> SimplePimDenseMicrobenchInput:
    value = SimplePimDenseMicrobenchInput(
        gemm_m=gemm_m,
        gemm_k=gemm_k,
        gemm_n=gemm_n,
        route_dtype=route_dtype,
        source_dtype=source_dtype,
        seed=seed,
        task_id=task_id,
        dry_run=dry_run,
    )
    _validate_input(value)
    return value


def prepare_simplepim_dense_microbench(
    microbench_input: SimplePimDenseMicrobenchInput,
    probe: SimplePimProbeResult | None = None,
    *,
    execute: bool = False,
) -> SimplePimDenseMicrobenchResult:
    started = time.perf_counter()
    probe_result = probe or probe_simplepim()
    probe_payload = probe_result.to_json_dict()
    empty_payload = _empty_result_payload(microbench_input, probe_payload)

    try:
        _validate_input(microbench_input)
        fixed_point_spec = FixedPointSpec(
            route_dtype=microbench_input.route_dtype,  # type: ignore[arg-type]
            complex_policy="reject",
        )
        task = _synthetic_dense_task(microbench_input)
        upmem_estimate = estimate_dense_task(task)

        rng = np.random.default_rng(microbench_input.seed)
        source_dtype = np.dtype(microbench_input.source_dtype)
        left = _sample_matrix(rng, (microbench_input.gemm_m, microbench_input.gemm_k), source_dtype)
        right = _sample_matrix(rng, (microbench_input.gemm_k, microbench_input.gemm_n), source_dtype)

        left_converted = quantize_fixed_point(left, fixed_point_spec)
        right_converted = quantize_fixed_point(right, fixed_point_spec)
        conversion_time_s = left_converted.record.conversion_time_s + right_converted.record.conversion_time_s

        dequantize_started = time.perf_counter()
        left_dequantized = dequantize_fixed_point(left_converted, dtype=np.float64)
        right_dequantized = dequantize_fixed_point(right_converted, dtype=np.float64)
        dequantization_time_s = time.perf_counter() - dequantize_started

        reference_started = time.perf_counter()
        reference = left.astype(np.float64) @ right.astype(np.float64)
        quantized_reference = left_dequantized @ right_dequantized
        reference_time_s = time.perf_counter() - reference_started

        validation = conversion_error_metrics(reference, quantized_reference)
        status, reason = _resolve_status(microbench_input, probe_payload, upmem_estimate, execute)
        return SimplePimDenseMicrobenchResult(
            schema_version=SIMPLEPIM_DENSE_MICROBENCH_SCHEMA_VERSION,
            microbench_id=SIMPLEPIM_DENSE_MICROBENCH_ID,
            input=microbench_input,
            status=status,
            skip_reason=reason,
            error=None,
            simplepim_probe=probe_payload,
            fixed_point_spec=to_jsonable(fixed_point_spec),
            tile_plan=upmem_estimate.tile_plan.as_summary(),
            upmem_task_estimate=upmem_estimate.as_task_estimate(),
            input_shapes=(tuple(left.shape), tuple(right.shape)),
            output_shape=tuple(reference.shape),
            conversion_records={
                "left": to_jsonable(left_converted.record),
                "right": to_jsonable(right_converted.record),
            },
            conversion_time_s=float(conversion_time_s),
            reference_time_s=float(reference_time_s),
            dequantization_time_s=float(dequantization_time_s),
            kernel_time_s=None,
            host_aggregation_time_s=None,
            total_time_s=float(time.perf_counter() - started),
            host_to_dpu_bytes=upmem_estimate.host_to_dpu_bytes,
            dpu_to_host_bytes=upmem_estimate.dpu_to_host_bytes,
            mram_to_wram_bytes=upmem_estimate.mram_to_wram_bytes,
            validation_metrics={
                "reference_kind": "numpy_float_gemm_vs_dequantized_input_gemm",
                "max_abs_error": validation.max_abs_error,
                "l2_error": validation.l2_error,
                "relative_l2_error": validation.relative_l2_error,
                "reference_norm": float(np.linalg.norm(reference.ravel())),
                "dequantized_reference_norm": float(np.linalg.norm(quantized_reference.ravel())),
            },
            external_command_executed=False,
            execution_implemented=False,
            metadata={
                "ready_means": "ready for a future SimplePIM bridge attempt; no kernel was executed",
                "status_priority": _status_priority(),
            },
        )
    except Exception as exc:
        return SimplePimDenseMicrobenchResult(
            **empty_payload,
            status="failed",
            skip_reason=None,
            error=str(exc),
            total_time_s=float(time.perf_counter() - started),
            metadata={"status_priority": _status_priority()},
        )


def _resolve_status(
    microbench_input: SimplePimDenseMicrobenchInput,
    probe_payload: JsonDict,
    upmem_estimate: object,
    execute: bool,
) -> tuple[SimplePimDenseMicrobenchStatus, str | None]:
    if execute or not microbench_input.dry_run:
        return "not_implemented", "simplepim_execution_not_implemented"
    if bool(getattr(upmem_estimate, "requires_tiling", False)):
        return "not_implemented", "requires_executable_tiling_not_implemented"
    if bool(getattr(upmem_estimate, "tile_plan").requires_host_aggregation):
        return "not_implemented", "requires_host_aggregation_not_implemented"

    probe_status = probe_payload.get("simplepim_probe_status")
    if probe_status == "unavailable":
        return "skipped", str(probe_payload.get("skip_reason") or "SimplePIM is not configured")
    if probe_status == "configured_but_unverified":
        return "configured_but_unverified", str(
            probe_payload.get("skip_reason") or "SimplePIM configuration exists but executable availability is not verified"
        )
    if probe_payload.get("simplepim_available"):
        return "ready", None
    return "skipped", "SimplePIM is not configured"


def _empty_result_payload(microbench_input: SimplePimDenseMicrobenchInput, probe_payload: JsonDict) -> JsonDict:
    return {
        "schema_version": SIMPLEPIM_DENSE_MICROBENCH_SCHEMA_VERSION,
        "microbench_id": SIMPLEPIM_DENSE_MICROBENCH_ID,
        "input": microbench_input,
        "simplepim_probe": probe_payload,
        "fixed_point_spec": {},
        "tile_plan": {},
        "upmem_task_estimate": {},
        "input_shapes": (),
        "output_shape": (),
        "conversion_records": {},
        "conversion_time_s": 0.0,
        "reference_time_s": 0.0,
        "dequantization_time_s": 0.0,
        "kernel_time_s": None,
        "host_aggregation_time_s": None,
        "host_to_dpu_bytes": 0,
        "dpu_to_host_bytes": 0,
        "mram_to_wram_bytes": 0,
        "validation_metrics": {},
        "external_command_executed": False,
        "execution_implemented": False,
    }


def _synthetic_dense_task(microbench_input: SimplePimDenseMicrobenchInput) -> ContractionTask:
    return ContractionTask(
        id=microbench_input.task_id,
        input_tensor_ids=(f"{microbench_input.task_id}_left", f"{microbench_input.task_id}_right"),
        output_tensor_id=f"{microbench_input.task_id}_out",
        dependencies=(),
        index_expression="ab,bc->ac",
        input_shapes=(
            (microbench_input.gemm_m, microbench_input.gemm_k),
            (microbench_input.gemm_k, microbench_input.gemm_n),
        ),
        output_shape=(microbench_input.gemm_m, microbench_input.gemm_n),
        left_labels=(0, 1),
        right_labels=(1, 2),
        contracted_labels=(1,),
        output_labels=(0, 2),
        gemm_m=microbench_input.gemm_m,
        gemm_k=microbench_input.gemm_k,
        gemm_n=microbench_input.gemm_n,
        structure="dense",
        estimated_flops=2 * microbench_input.gemm_m * microbench_input.gemm_k * microbench_input.gemm_n,
        estimated_bytes=(
            microbench_input.gemm_m * microbench_input.gemm_k
            + microbench_input.gemm_k * microbench_input.gemm_n
            + microbench_input.gemm_m * microbench_input.gemm_n
        ),
    )


def _sample_matrix(rng: np.random.Generator, shape: tuple[int, int], dtype: np.dtype) -> np.ndarray:
    return rng.standard_normal(shape).astype(dtype)


def _validate_input(microbench_input: SimplePimDenseMicrobenchInput) -> None:
    if microbench_input.gemm_m <= 0 or microbench_input.gemm_k <= 0 or microbench_input.gemm_n <= 0:
        raise ValueError("GEMM dimensions must be positive")
    if microbench_input.route_id != "dense_gemm":
        raise ValueError("SimplePIM dense microbenchmark only supports route_id='dense_gemm'")
    if microbench_input.route_dtype not in {"int8", "int16"}:
        raise ValueError("route_dtype must be int8 or int16")
    if microbench_input.source_dtype not in {"float32", "float64"}:
        raise ValueError("source_dtype must be float32 or float64")
    if microbench_input.distribution != "standard_normal":
        raise ValueError("Only distribution='standard_normal' is implemented")


def _status_priority() -> tuple[str, ...]:
    return (
        "failed: invalid input or host preparation exception",
        "not_implemented: execute=True or future execution requested",
        "not_implemented: executable tiling/aggregation required",
        "skipped: no SimplePIM executable/configuration",
        "configured_but_unverified: SIMPLEPIM_HOME or SIMPLEPIM_LIB only",
        "ready: SimplePIM command/bin discoverable and dry-run preparation succeeded",
    )
