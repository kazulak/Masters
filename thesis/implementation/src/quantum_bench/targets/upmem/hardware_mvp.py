"""Strict configuration contract for the first physical UPMEM dense MVP.

This module intentionally models only the bring-up experiment.  It is not a
general hardware scheduler, benchmark suite loader, or device capability
layer.  Keeping the profile narrow makes the first hardware evidence easy to
audit and prevents environment variables from silently expanding its scope.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping

import numpy as np
import yaml


UPMEM_HARDWARE_MVP_SUITE_SCHEMA_VERSION = "upmem_hardware_mvp_v1"
HARDWARE_MVP_PROFILE_VERSION = "hardware_mvp_l1_v2"
HARDWARE_MVP_SDK_ALLOCATION_PROFILE = "backend=hw"
HARDWARE_MVP_BACKEND_ID = "upmem_sdk_hardware_dense"
HARDWARE_MVP_ROUTE_ID = "upmem_dense_l1_int8_hardware_mvp"
HARDWARE_MVP_EXECUTION_CLASS = "L1_WRAM"
HARDWARE_MVP_KERNEL_STRATEGY = "l1_direct_int8_int32_v1"
HARDWARE_MVP_MAX_DIM = 4
HARDWARE_MVP_REQUESTED_DPU_COUNT = 1
HARDWARE_MVP_TASKLETS_PER_DPU = 1
HARDWARE_MVP_REPETITIONS = 5
HARDWARE_MVP_TIMEOUT_S = 30.0
HARDWARE_MVP_PROFILE_FIELDS = frozenset(
    {
        "hardware_profile_version",
        "target",
        "execution_class",
        "backend_id",
        "route_id",
        "kernel_strategy",
        "requested_dpu_count",
        "tasklets_per_dpu",
        "max_dim",
        "repetitions",
        "timeout_s",
        "input_dtype",
        "accumulator_dtype",
        "complex_policy",
        "synchronous_execution",
        "performance_claim_applicable",
    }
)


@dataclass(frozen=True)
class HardwareMvpProfile:
    version: str = HARDWARE_MVP_PROFILE_VERSION
    target: Literal["hardware"] = "hardware"
    execution_class: str = HARDWARE_MVP_EXECUTION_CLASS
    backend_id: str = HARDWARE_MVP_BACKEND_ID
    route_id: str = HARDWARE_MVP_ROUTE_ID
    kernel_strategy: str = HARDWARE_MVP_KERNEL_STRATEGY
    requested_dpu_count: int = HARDWARE_MVP_REQUESTED_DPU_COUNT
    tasklets_per_dpu: int = HARDWARE_MVP_TASKLETS_PER_DPU
    max_dim: int = HARDWARE_MVP_MAX_DIM
    repetitions: int = HARDWARE_MVP_REPETITIONS
    timeout_s: float = HARDWARE_MVP_TIMEOUT_S
    input_dtype: str = "int8"
    accumulator_dtype: str = "int32"
    complex_policy: str = "reject"
    synchronous_execution: bool = True
    performance_claim_applicable: bool = False

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
            "max_dim": self.max_dim,
            "repetitions": self.repetitions,
            "timeout_s": self.timeout_s,
            "input_dtype": self.input_dtype,
            "accumulator_dtype": self.accumulator_dtype,
            "complex_policy": self.complex_policy,
            "synchronous_execution": self.synchronous_execution,
            "performance_claim_applicable": self.performance_claim_applicable,
        }


HARDWARE_MVP_PROFILE = HardwareMvpProfile()

HARDWARE_MVP_CANONICAL_OPERANDS: dict[str, tuple[np.ndarray, np.ndarray]] = {
    "dense_l1_2x2": (
        np.asarray(((2, -3), (4, 1)), dtype=np.int8),
        np.asarray(((5, 2), (-1, 3)), dtype=np.int8),
    ),
    "dense_l1_4x4": (
        np.asarray(
            ((1, -2, 3, 0), (4, 1, -1, 2), (0, 3, 2, -4), (-2, 5, 1, 3)), dtype=np.int8
        ),
        np.asarray(
            ((2, 1, -3, 4), (-1, 3, 2, 0), (4, -2, 1, 5), (3, 0, -4, 2)), dtype=np.int8
        ),
    ),
}


@dataclass(frozen=True)
class HardwareMvpCase:
    case_id: str
    left_int8: np.ndarray
    right_int8: np.ndarray

    @property
    def m(self) -> int:
        return int(self.left_int8.shape[0])

    @property
    def k(self) -> int:
        return int(self.left_int8.shape[1])

    @property
    def n(self) -> int:
        return int(self.right_int8.shape[1])

    @property
    def expected_accumulator(self) -> np.ndarray:
        return self.left_int8.astype(np.int32) @ self.right_int8.astype(np.int32)

    def to_json_dict(self) -> dict[str, object]:
        return {
            "case_id": self.case_id,
            "dimensions": {"m": self.m, "k": self.k, "n": self.n},
            "left_int8": self.left_int8.tolist(),
            "right_int8": self.right_int8.tolist(),
        }


@dataclass(frozen=True)
class HardwareMvpSuite:
    schema_version: str
    suite_id: str
    profile: HardwareMvpProfile
    cases: tuple[HardwareMvpCase, ...]
    suite_path: Path


def load_hardware_mvp_suite(path: Path) -> HardwareMvpSuite:
    """Load the deliberately small, hardware-only MVP suite.

    The regular schema-v2 suite loader remains circuit-oriented.  This
    hardware bring-up suite has a narrower contract and therefore refuses all
    fields that could change execution scope.
    """

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Hardware MVP suite {path} must contain a YAML mapping")
    _require_exact_keys(
        raw, {"schema_version", "suite_id", "profile", "cases"}, "hardware MVP suite"
    )
    if raw.get("schema_version") != UPMEM_HARDWARE_MVP_SUITE_SCHEMA_VERSION:
        raise ValueError(
            f"Hardware MVP suite must use schema_version: {UPMEM_HARDWARE_MVP_SUITE_SCHEMA_VERSION}"
        )
    suite_id = raw.get("suite_id")
    if not isinstance(suite_id, str) or not suite_id:
        raise ValueError("Hardware MVP suite requires a non-empty suite_id")
    profile = _parse_profile(raw.get("profile"))
    raw_cases = raw.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("Hardware MVP suite requires a non-empty cases list")
    cases = tuple(_parse_case(raw_case, profile) for raw_case in raw_cases)
    ids = [case.case_id for case in cases]
    if len(set(ids)) != len(ids):
        raise ValueError("Hardware MVP case IDs must be unique")
    _validate_canonical_cases(cases)
    return HardwareMvpSuite(
        schema_version=UPMEM_HARDWARE_MVP_SUITE_SCHEMA_VERSION,
        suite_id=suite_id,
        profile=profile,
        cases=cases,
        suite_path=path.resolve(),
    )


def validate_hardware_mvp_manifest(
    manifest: Mapping[str, Any],
    *,
    profile: HardwareMvpProfile = HARDWARE_MVP_PROFILE,
) -> None:
    """Reject a dense bridge manifest outside the physical MVP contract."""

    dims = tuple(
        int(manifest.get(name, 0) or 0) for name in ("gemm_m", "gemm_k", "gemm_n")
    )
    if any(dimension <= 0 or dimension > profile.max_dim for dimension in dims):
        raise ValueError(
            f"hardware_profile_violation: dimensions {dims} exceed 1..{profile.max_dim}"
        )
    fixed = manifest.get("fixed_point_spec")
    if not isinstance(fixed, Mapping):
        raise ValueError("hardware_profile_violation: fixed_point_spec missing")
    metadata = manifest.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("hardware_profile_violation: hardware profile metadata missing")
    expected_metadata = {
        "hardware_profile_version": profile.version,
        "sdk_allocation_profile": HARDWARE_MVP_SDK_ALLOCATION_PROFILE,
        "sdk_allocation_profile_source": "compiled_native_literal",
        "target": profile.target,
        "execution_class": profile.execution_class,
        "backend_id": profile.backend_id,
        "requested_dpu_count": profile.requested_dpu_count,
        "tasklets_per_dpu": profile.tasklets_per_dpu,
        "synchronous_execution": True,
        "performance_claim_applicable": False,
    }
    for field, expected in expected_metadata.items():
        if metadata.get(field) != expected:
            raise ValueError(
                "hardware_profile_violation: "
                f"metadata field {field} must be {expected!r}"
            )
    if fixed.get("route_dtype") != profile.input_dtype:
        raise ValueError("hardware_profile_violation: input dtype must be int8")
    if fixed.get("complex_policy") != profile.complex_policy:
        raise ValueError(
            "hardware_profile_violation: complex policy must reject complex inputs"
        )
    tile_plan = manifest.get("tile_plan")
    if isinstance(tile_plan, Mapping) and tile_plan.get("requires_tiling") is True:
        raise ValueError(
            "hardware_profile_violation: L1 hardware MVP does not permit tiling"
        )
    operands = manifest.get("operands")
    if not isinstance(operands, Mapping):
        raise ValueError("hardware_profile_violation: operands missing")
    for name in ("left", "right"):
        operand = operands.get(name)
        if (
            not isinstance(operand, Mapping)
            or operand.get("dtype") != profile.input_dtype
        ):
            raise ValueError(f"hardware_profile_violation: {name} operand must be int8")


def hardware_mvp_profile_metadata(
    profile: HardwareMvpProfile = HARDWARE_MVP_PROFILE,
) -> dict[str, object]:
    """Return metadata suitable for manifests and normalized evidence."""

    return {
        **profile.to_json_dict(),
        "sdk_allocation_profile": HARDWARE_MVP_SDK_ALLOCATION_PROFILE,
        "sdk_allocation_profile_source": "compiled_native_literal",
        "hardware_functionality_evidence": True,
        "hardware_speedup_applicable": False,
        "performance_claim_applicable": False,
        "timing_scope": "hardware_bringup_functionality_only",
        "timing_is_bringup_only": True,
        "simulator_kernel_executed": False,
        "cpu_fallback_used": False,
        "synchronous_execution": True,
    }


def _parse_profile(value: object) -> HardwareMvpProfile:
    if not isinstance(value, Mapping):
        raise ValueError("Hardware MVP suite profile must be a mapping")
    _require_exact_keys(
        value, HARDWARE_MVP_PROFILE_FIELDS, "hardware MVP profile", allow_missing=True
    )
    expected = HARDWARE_MVP_PROFILE.to_json_dict()
    supplied = dict(value)
    # The suite is allowed to restate the fixed values but not override them.
    for field, expected_value in expected.items():
        if field not in supplied:
            continue
        actual = supplied[field]
        if actual != expected_value:
            raise ValueError(
                f"hardware_profile_violation: profile field {field} must be {expected_value!r}, got {actual!r}"
            )
    return HARDWARE_MVP_PROFILE


def _parse_case(value: object, profile: HardwareMvpProfile) -> HardwareMvpCase:
    if not isinstance(value, Mapping):
        raise ValueError("Hardware MVP cases must be mappings")
    _require_exact_keys(value, {"id", "left_int8", "right_int8"}, "hardware MVP case")
    case_id = value.get("id")
    if not isinstance(case_id, str) or not case_id:
        raise ValueError("Hardware MVP case requires a non-empty id")
    left = _int8_matrix(value.get("left_int8"), "left_int8")
    right = _int8_matrix(value.get("right_int8"), "right_int8")
    if left.shape[1] != right.shape[0]:
        raise ValueError(
            f"Hardware MVP case {case_id} has incompatible shapes {left.shape} and {right.shape}"
        )
    dims = (int(left.shape[0]), int(left.shape[1]), int(right.shape[1]))
    if any(dimension <= 0 or dimension > profile.max_dim for dimension in dims):
        raise ValueError(
            f"Hardware MVP case {case_id} dimensions {dims} exceed 1..{profile.max_dim}"
        )
    return HardwareMvpCase(case_id=case_id, left_int8=left, right_int8=right)


def _validate_canonical_cases(cases: tuple[HardwareMvpCase, ...]) -> None:
    if tuple(case.case_id for case in cases) != tuple(HARDWARE_MVP_CANONICAL_OPERANDS):
        raise ValueError(
            "hardware_profile_violation: hardware_mvp_l1_v2 requires the fixed 2x2 then 4x4 case order"
        )
    for case in cases:
        expected_left, expected_right = HARDWARE_MVP_CANONICAL_OPERANDS[case.case_id]
        if not np.array_equal(case.left_int8, expected_left) or not np.array_equal(
            case.right_int8, expected_right
        ):
            raise ValueError(
                f"hardware_profile_violation: {case.case_id} operands are fixed by hardware_mvp_l1_v2"
            )


def _int8_matrix(value: object, name: str) -> np.ndarray:
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(row, list) and row for row in value)
    ):
        raise ValueError(f"{name} must be a non-empty nested list")
    width = len(value[0])
    if any(len(row) != width for row in value):
        raise ValueError(f"{name} must be rectangular")
    if any(
        isinstance(item, bool) or not isinstance(item, int)
        for row in value
        for item in row
    ):
        raise ValueError(f"{name} must contain integer values")
    try:
        numbers = np.asarray(value, dtype=np.int64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain integer values") from exc
    if np.any(numbers < -128) or np.any(numbers > 127):
        raise ValueError(f"{name} values must fit int8")
    return numbers.astype(np.int8)


def _require_exact_keys(
    value: Mapping[str, Any] | set[str],
    expected: set[str],
    context: str,
    *,
    allow_missing: bool = False,
) -> None:
    present = set(value) if isinstance(value, Mapping) else value
    unknown = sorted(present - expected)
    if unknown:
        raise ValueError(f"Unknown {context} field(s): {', '.join(unknown)}")
    if not allow_missing:
        missing = sorted(expected - present)
        if missing:
            raise ValueError(f"Missing {context} field(s): {', '.join(missing)}")
