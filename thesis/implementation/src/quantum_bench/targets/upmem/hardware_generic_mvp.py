"""Fixed-profile physical generic-loop TaskGraph MVP definitions.

This profile is deliberately narrower than the simulator generic-loop route.
It exists solely to prove one real-valued TaskGraph contraction on one physical
DPU.  It is not a performance, scaling, or general quantum-TN route.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml


UPMEM_HARDWARE_GENERIC_MVP_SUITE_SCHEMA_VERSION = "upmem_hardware_generic_mvp_v1"
HARDWARE_GENERIC_MVP_BACKEND_ID = "upmem_sdk_hardware_generic_loop"
HARDWARE_GENERIC_MVP_ROUTE_ID = "upmem_tn_hardware_generic_loop_mvp"
HARDWARE_GENERIC_MVP_PROFILE_VERSION = "hardware_generic_loop_mvp_v1"
HARDWARE_GENERIC_MVP_SDK_ALLOCATION_PROFILE = "backend=hw"
HARDWARE_GENERIC_MVP_MAX_RANK = 4
HARDWARE_GENERIC_MVP_MAX_ELEMS = 16
HARDWARE_GENERIC_MVP_OUTPUT_TILE_ELEMENTS = 8
HARDWARE_GENERIC_MVP_REPETITIONS = 5
HARDWARE_GENERIC_MVP_TIMEOUT_S = 30.0


@dataclass(frozen=True)
class HardwareGenericMvpProfile:
    version: str
    target: str
    execution_class: str
    backend_id: str
    route_id: str
    kernel_strategy: str
    requested_dpu_count: int
    tasklets_per_dpu: int
    max_rank: int
    max_tensor_elements: int
    output_tile_elements: int
    repetitions: int
    timeout_s: float
    input_dtype: str
    accumulator_dtype: str
    complex_policy: str
    synchronous_execution: bool
    performance_claim_applicable: bool

    def to_json_dict(self) -> dict[str, object]:
        return {
            "hardware_profile_version": self.version,
            "target": self.target,
            "execution_class": self.execution_class,
            "backend_id": self.backend_id,
            "route_id": self.route_id,
            "kernel_strategy": self.kernel_strategy,
            "requested_dpu_count": self.requested_dpu_count,
            "tasklets_per_dpu": self.tasklets_per_dpu,
            "max_rank": self.max_rank,
            "max_tensor_elements": self.max_tensor_elements,
            "output_tile_elements": self.output_tile_elements,
            "repetitions": self.repetitions,
            "timeout_s": self.timeout_s,
            "input_dtype": self.input_dtype,
            "accumulator_dtype": self.accumulator_dtype,
            "complex_policy": self.complex_policy,
            "synchronous_execution": self.synchronous_execution,
            "performance_claim_applicable": self.performance_claim_applicable,
        }


@dataclass(frozen=True)
class HardwareGenericMvpCase:
    case_id: str
    left_int8: np.ndarray
    right_int8: np.ndarray

    @property
    def output_shape(self) -> tuple[int, int, int, int]:
        return (
            int(self.left_int8.shape[0]),
            int(self.left_int8.shape[1]),
            int(self.right_int8.shape[1]),
            int(self.right_int8.shape[2]),
        )


@dataclass(frozen=True)
class HardwareGenericMvpSuite:
    suite_path: Path
    suite_id: str
    profile: HardwareGenericMvpProfile
    cases: tuple[HardwareGenericMvpCase, ...]


def load_hardware_generic_mvp_suite(path: Path) -> HardwareGenericMvpSuite:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("hardware_profile_violation: suite must be a mapping")
    if payload.get("schema_version") != UPMEM_HARDWARE_GENERIC_MVP_SUITE_SCHEMA_VERSION:
        raise ValueError("hardware_profile_violation: unsupported generic hardware suite schema")
    suite_id = payload.get("suite_id")
    if suite_id != "upmem_hardware_generic_mvp":
        raise ValueError("hardware_profile_violation: suite_id must be upmem_hardware_generic_mvp")
    profile = _parse_profile(payload.get("profile"))
    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, list) or len(raw_cases) != 1:
        raise ValueError("hardware_profile_violation: generic hardware MVP requires exactly one case")
    case = _parse_case(raw_cases[0])
    if case.left_int8.shape != (2, 2, 2) or case.right_int8.shape != (2, 2, 2):
        raise ValueError("hardware_profile_violation: generic hardware MVP requires 2x2x2 operands")
    if int(np.prod(case.output_shape)) != HARDWARE_GENERIC_MVP_MAX_ELEMS:
        raise ValueError("hardware_profile_violation: generic hardware MVP must produce exactly 16 outputs")
    return HardwareGenericMvpSuite(path.resolve(), str(suite_id), profile, (case,))


def validate_hardware_generic_mvp_manifest(
    manifest: Mapping[str, Any], *, profile: HardwareGenericMvpProfile | None = None
) -> None:
    """Validate the immutable hardware bridge contract before execution."""

    expected = profile or _canonical_profile()
    metadata = manifest.get("metadata")
    native = manifest.get("native_index_metadata")
    fixed = manifest.get("fixed_point_spec")
    operands = manifest.get("operands")
    if not all(isinstance(value, Mapping) for value in (metadata, native, fixed, operands)):
        raise ValueError("hardware_profile_violation: generic hardware manifest is incomplete")
    for key, value in hardware_generic_mvp_profile_metadata(expected).items():
        if metadata.get(key) != value:
            raise ValueError(f"hardware_profile_violation: metadata {key} is invalid")
    input_shapes = tuple(tuple(int(dim) for dim in shape) for shape in (manifest.get("input_shapes") or ()))
    if input_shapes != ((2, 2, 2), (2, 2, 2)):
        raise ValueError("hardware_profile_violation: input shapes must be 2x2x2")
    output_shape = tuple(int(dim) for dim in (manifest.get("output_shape") or ()))
    if output_shape != (2, 2, 2, 2):
        raise ValueError("hardware_profile_violation: output shape must be 2x2x2x2")
    if manifest.get("route_id") != expected.route_id:
        raise ValueError("hardware_profile_violation: route ID is invalid")
    if manifest.get("backend_id") != expected.backend_id:
        raise ValueError("hardware_profile_violation: backend ID is invalid")
    if fixed.get("route_dtype") != "int8" or fixed.get("complex_policy") != "reject" or float(fixed.get("scale", 0.0)) != 1.0:
        raise ValueError("hardware_profile_violation: generic hardware MVP requires identity int8 quantization")
    if native.get("left_rank") != 3 or native.get("right_rank") != 3 or native.get("output_rank") != 4:
        raise ValueError("hardware_profile_violation: rank-3 x rank-3 -> rank-4 task required")
    if native.get("output_element_count") != 16 or native.get("contracted_combination_count") != 2:
        raise ValueError("hardware_profile_violation: fixed output/contracted element counts required")
    if native.get("generic_output_tile_elements") not in (None, expected.output_tile_elements):
        raise ValueError("hardware_profile_violation: generic output tile size is invalid")
    for name in ("left", "right"):
        operand = operands.get(name)
        if not isinstance(operand, Mapping) or operand.get("dtype") != "int8":
            raise ValueError(f"hardware_profile_violation: {name} operand must be int8")


def hardware_generic_mvp_profile_metadata(
    profile: HardwareGenericMvpProfile | None = None,
) -> dict[str, object]:
    current = profile or _canonical_profile()
    return {
        **current.to_json_dict(),
        "sdk_allocation_profile": HARDWARE_GENERIC_MVP_SDK_ALLOCATION_PROFILE,
        "sdk_allocation_profile_source": "compiled_native_literal",
        "hardware_functionality_evidence": True,
        "synthetic_real_taskgraph_mvp": True,
        "not_real_quantum_circuit": True,
        "hardware_speedup_applicable": False,
    }


def _canonical_profile() -> HardwareGenericMvpProfile:
    return HardwareGenericMvpProfile(
        version=HARDWARE_GENERIC_MVP_PROFILE_VERSION,
        target="hardware",
        execution_class="MRAM_WRAM_TILED",
        backend_id=HARDWARE_GENERIC_MVP_BACKEND_ID,
        route_id=HARDWARE_GENERIC_MVP_ROUTE_ID,
        kernel_strategy="generic_loop_output_tiled_int8_int32_v1",
        requested_dpu_count=1,
        tasklets_per_dpu=1,
        max_rank=HARDWARE_GENERIC_MVP_MAX_RANK,
        max_tensor_elements=HARDWARE_GENERIC_MVP_MAX_ELEMS,
        output_tile_elements=HARDWARE_GENERIC_MVP_OUTPUT_TILE_ELEMENTS,
        repetitions=HARDWARE_GENERIC_MVP_REPETITIONS,
        timeout_s=HARDWARE_GENERIC_MVP_TIMEOUT_S,
        input_dtype="int8",
        accumulator_dtype="int32",
        complex_policy="reject",
        synchronous_execution=True,
        performance_claim_applicable=False,
    )


def _parse_profile(value: object) -> HardwareGenericMvpProfile:
    if not isinstance(value, Mapping):
        raise ValueError("hardware_profile_violation: profile must be a mapping")
    canonical = _canonical_profile()
    expected = canonical.to_json_dict()
    _require_exact_keys(value, set(expected))
    for key, expected_value in expected.items():
        if value.get(key) != expected_value:
            raise ValueError(f"hardware_profile_violation: {key} must be {expected_value!r}")
    return canonical


def _parse_case(value: object) -> HardwareGenericMvpCase:
    if not isinstance(value, Mapping):
        raise ValueError("hardware_profile_violation: case must be a mapping")
    _require_exact_keys(value, {"id", "left_int8", "right_int8"})
    case_id = value.get("id")
    if case_id != "generic_real_abc_cde_2":
        raise ValueError("hardware_profile_violation: unsupported generic hardware MVP case")
    return HardwareGenericMvpCase(
        case_id=str(case_id),
        left_int8=_int8_array(value.get("left_int8"), "left_int8"),
        right_int8=_int8_array(value.get("right_int8"), "right_int8"),
    )


def _int8_array(value: object, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != (2, 2, 2):
        raise ValueError(f"hardware_profile_violation: {name} must have shape 2x2x2")
    if not np.issubdtype(array.dtype, np.integer) or np.any(array < -128) or np.any(array > 127):
        raise ValueError(f"hardware_profile_violation: {name} must contain int8 values")
    return array.astype(np.int8, copy=False)


def _require_exact_keys(value: Mapping[str, object], expected: set[str]) -> None:
    unexpected = set(value) - expected
    missing = expected - set(value)
    if unexpected or missing:
        pieces: list[str] = []
        if unexpected:
            pieces.append("unexpected=" + ",".join(sorted(unexpected)))
        if missing:
            pieces.append("missing=" + ",".join(sorted(missing)))
        raise ValueError("hardware_profile_violation: " + "; ".join(pieces))
