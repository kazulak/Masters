"""Contracts for the additive physical UPMEM TaskGraph correctness route.

The fixed generic MVP remains intentionally unchanged.  This module defines a
separate, bounded profile for the next step: executing a real circuit-derived
TaskGraph on one physical DPU.  The initial implementation rebuilds native
source once per run but creates one native allocation/load/release session per
logical TaskGraph contraction so split-complex recombination and per-task
quantization remain explicit on the host.  It is correctness and
measurement-foundation evidence, not hardware-speedup evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from quantum_bench.bench.config import load_suite
from quantum_bench.core.records import JsonDict


UPMEM_HARDWARE_TASKGRAPH_SUITE_SCHEMA_VERSION = "upmem_hardware_taskgraph_v1"
HARDWARE_TASKGRAPH_BACKEND_ID = "upmem_sdk_hardware_taskgraph"
HARDWARE_TASKGRAPH_ROUTE_ID = "upmem_tn_hardware_taskgraph"
HARDWARE_TASKGRAPH_PROFILE_VERSION = "hardware_taskgraph_single_dpu_v1"
HARDWARE_TASKGRAPH_SESSION_PROTOCOL = "generic_loop_batch_session_v1"
HARDWARE_TASKGRAPH_MAX_RANK = 16
HARDWARE_TASKGRAPH_MAX_TENSOR_ELEMENTS = 256
HARDWARE_TASKGRAPH_MAX_CONTRACTED_COMBINATIONS = 256
HARDWARE_TASKGRAPH_OUTPUT_TILE_ELEMENTS = 64
HARDWARE_TASKGRAPH_TIMEOUT_S = 30.0
HARDWARE_TASKGRAPH_NUMERIC_MODES = ("none", "per_task_input_quantize")
HARDWARE_TASKGRAPH_COMPLEX_POLICY = "split_real_imag_float32"


@dataclass(frozen=True)
class HardwareTaskGraphProfile:
    version: str
    target: str
    backend_id: str
    route_id: str
    session_protocol: str
    requested_dpu_count: int
    tasklets_per_dpu: int
    max_rank: int
    max_tensor_elements: int
    max_contracted_combinations: int
    output_tile_elements: int
    numeric_modes: tuple[str, ...]
    complex_policy: str
    synchronous_execution: bool
    timeout_s: float
    performance_claim_applicable: bool

    def to_json_dict(self) -> JsonDict:
        return {
            "hardware_profile_version": self.version,
            "target": self.target,
            "backend_id": self.backend_id,
            "route_id": self.route_id,
            "session_protocol": self.session_protocol,
            "requested_dpu_count": self.requested_dpu_count,
            "tasklets_per_dpu": self.tasklets_per_dpu,
            "max_rank": self.max_rank,
            "max_tensor_elements": self.max_tensor_elements,
            "max_contracted_combinations": self.max_contracted_combinations,
            "output_tile_elements": self.output_tile_elements,
            "numeric_modes": list(self.numeric_modes),
            "complex_policy": self.complex_policy,
            "synchronous_execution": self.synchronous_execution,
            "timeout_s": self.timeout_s,
            "performance_claim_applicable": self.performance_claim_applicable,
        }


@dataclass(frozen=True)
class HardwareTaskGraphSuite:
    suite_path: Path
    suite: JsonDict
    profile: HardwareTaskGraphProfile


def load_hardware_taskgraph_suite(path: Path) -> HardwareTaskGraphSuite:
    """Load a normal v2 circuit suite with a constrained hardware profile."""

    suite = load_suite(path)
    metadata = suite.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("hardware_profile_violation: suite metadata must be a mapping")
    if (
        metadata.get("hardware_taskgraph_schema_version")
        != UPMEM_HARDWARE_TASKGRAPH_SUITE_SCHEMA_VERSION
    ):
        raise ValueError(
            "hardware_profile_violation: unsupported hardware TaskGraph suite schema"
        )
    profile = _parse_profile(metadata.get("hardware_profile"))
    routes = tuple(
        str(route) for route in (suite.get("route_policy") or {}).get("routes", ())
    )
    if routes != (HARDWARE_TASKGRAPH_ROUTE_ID,):
        raise ValueError(
            "hardware_profile_violation: hardware TaskGraph suite must contain only its physical route"
        )
    for case in suite.get(
        "cases", ()
    ):  # ``load_suite`` has already validated circuit shape.
        if case.get("hardware_numeric_coverage") not in {"real", "split_complex"}:
            raise ValueError(
                "hardware_profile_violation: each workload needs hardware_numeric_coverage real or split_complex"
            )
    return HardwareTaskGraphSuite(path.resolve(), suite, profile)


def hardware_taskgraph_profile_metadata(profile: HardwareTaskGraphProfile) -> JsonDict:
    return {
        **profile.to_json_dict(),
        "hardware_functionality_evidence": True,
        "hardware_speedup_applicable": bool(profile.performance_claim_applicable),
        "native_build_reuse_required": True,
        "logical_task_session_only": True,
        "simulator_kernel_executed": False,
        "cpu_fallback_used": False,
        "claim_boundary": (
            "single-DPU physical TaskGraph correctness and timing-foundation evidence; "
            "no hardware speedup, energy, multi-DPU, or scheduler claim"
        ),
    }


def validate_hardware_taskgraph_execution_request(
    *, execute: bool, environment: Mapping[str, str] | None = None
) -> None:
    env = environment or {}
    if not execute:
        raise ValueError("hardware TaskGraph execution requires --execute")
    if env.get("UPMEM_ALLOW_PHYSICAL_HARDWARE") != "1":
        raise ValueError(
            "UPMEM_ALLOW_PHYSICAL_HARDWARE=1 is required for physical UPMEM execution"
        )
    if env.get("DPU_BACKEND"):
        raise ValueError("DPU_BACKEND must be unset for physical TaskGraph execution")


def _parse_profile(value: object) -> HardwareTaskGraphProfile:
    if not isinstance(value, Mapping):
        raise ValueError(
            "hardware_profile_violation: hardware_profile must be a mapping"
        )
    expected = _canonical_profile().to_json_dict()
    if set(value) != set(expected):
        missing = sorted(set(expected) - set(value))
        unexpected = sorted(set(value) - set(expected))
        raise ValueError(
            "hardware_profile_violation: profile keys differ "
            f"missing={','.join(missing)} unexpected={','.join(unexpected)}"
        )
    for key, expected_value in expected.items():
        actual = value.get(key)
        if actual != expected_value:
            raise ValueError(
                f"hardware_profile_violation: {key} must be {expected_value!r}"
            )
    return _canonical_profile()


def _canonical_profile() -> HardwareTaskGraphProfile:
    return HardwareTaskGraphProfile(
        version=HARDWARE_TASKGRAPH_PROFILE_VERSION,
        target="hardware",
        backend_id=HARDWARE_TASKGRAPH_BACKEND_ID,
        route_id=HARDWARE_TASKGRAPH_ROUTE_ID,
        session_protocol=HARDWARE_TASKGRAPH_SESSION_PROTOCOL,
        requested_dpu_count=1,
        tasklets_per_dpu=1,
        max_rank=HARDWARE_TASKGRAPH_MAX_RANK,
        max_tensor_elements=HARDWARE_TASKGRAPH_MAX_TENSOR_ELEMENTS,
        max_contracted_combinations=HARDWARE_TASKGRAPH_MAX_CONTRACTED_COMBINATIONS,
        output_tile_elements=HARDWARE_TASKGRAPH_OUTPUT_TILE_ELEMENTS,
        numeric_modes=HARDWARE_TASKGRAPH_NUMERIC_MODES,
        complex_policy=HARDWARE_TASKGRAPH_COMPLEX_POLICY,
        synchronous_execution=True,
        timeout_s=HARDWARE_TASKGRAPH_TIMEOUT_S,
        performance_claim_applicable=False,
    )
