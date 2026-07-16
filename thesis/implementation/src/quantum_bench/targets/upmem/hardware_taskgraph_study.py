"""Contracts for the physical one-DPU path/quantization timing study.

This is deliberately separate from the one-DPU correctness route.  The study
uses one persistent physical DPU session per circuit case, compares two
explicit contraction paths and two numeric modes, and records steady-state
full-TaskGraph timing only.  It is still not a CPU/GPU speedup, energy, or
multi-DPU benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import yaml

from quantum_bench.bench.config import load_suite
from quantum_bench.core.records import JsonDict


UPMEM_HARDWARE_TASKGRAPH_STUDY_SUITE_SCHEMA_VERSION = (
    "upmem_hardware_taskgraph_study_v1"
)
HARDWARE_TASKGRAPH_STUDY_BACKEND_ID = "upmem_sdk_hardware_taskgraph_persistent"
HARDWARE_TASKGRAPH_STUDY_ROUTE_ID = "upmem_tn_hardware_taskgraph_persistent"
HARDWARE_TASKGRAPH_STUDY_PROFILE_VERSION = "hardware_taskgraph_single_dpu_persistent_v1"
HARDWARE_TASKGRAPH_STUDY_SESSION_PROTOCOL = "generic_loop_interactive_session_v1"
HARDWARE_TASKGRAPH_STUDY_MAX_RANK = 16
HARDWARE_TASKGRAPH_STUDY_MAX_TENSOR_ELEMENTS = 256
HARDWARE_TASKGRAPH_STUDY_MAX_CONTRACTED_COMBINATIONS = 256
HARDWARE_TASKGRAPH_STUDY_OUTPUT_TILE_ELEMENTS = 64
HARDWARE_TASKGRAPH_STUDY_TIMEOUT_S = 30.0
HARDWARE_TASKGRAPH_STUDY_NUMERIC_MODES = ("none", "per_task_input_quantize")
HARDWARE_TASKGRAPH_STUDY_COMPLEX_POLICY = "split_real_imag_float32"
HARDWARE_TASKGRAPH_STUDY_VARIANTS = (
    "opt_einsum_greedy",
    "custom_upmem_v2_balanced",
)
HARDWARE_TASKGRAPH_STUDY_WORKLOAD_IDS = (
    "bv_3q_one_dpu",
    "bv_4q_one_dpu",
    "bv_5q_one_dpu",
    "xor_3q_one_dpu",
    "xor_4q_one_dpu",
    "xor_5q_one_dpu",
    "edc_3q_one_dpu",
    "edc_4q_one_dpu",
    "edc_5q_one_dpu",
    "bb84_3q_one_dpu",
    "bb84_4q_one_dpu",
    "bb84_5q_one_dpu",
    "quantization_stress_2q_one_dpu",
)


@dataclass(frozen=True)
class HardwareTaskGraphStudyProfile:
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
class HardwareTaskGraphStudyVariant:
    variant_id: str
    label: str
    planner: JsonDict


@dataclass(frozen=True)
class HardwareTaskGraphStudySuite:
    suite_path: Path
    suite: JsonDict
    profile: HardwareTaskGraphStudyProfile
    variants: tuple[HardwareTaskGraphStudyVariant, ...]


def load_hardware_taskgraph_study_suite(path: Path) -> HardwareTaskGraphStudySuite:
    """Load the fixed physical path/quantization study suite."""

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("hardware_profile_violation: study suite must be a mapping")
    suite = load_suite(path)
    metadata = suite.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("hardware_profile_violation: study metadata must be a mapping")
    if metadata.get("hardware_taskgraph_study_schema_version") != (
        UPMEM_HARDWARE_TASKGRAPH_STUDY_SUITE_SCHEMA_VERSION
    ):
        raise ValueError(
            "hardware_profile_violation: unsupported hardware study suite schema"
        )
    profile = _parse_profile(metadata.get("hardware_profile"))
    routes = tuple(
        str(route) for route in (suite.get("route_policy") or {}).get("routes", ())
    )
    if routes != (HARDWARE_TASKGRAPH_STUDY_ROUTE_ID,):
        raise ValueError(
            "hardware_profile_violation: study suite must contain only its persistent route"
        )
    if int(suite.get("warmups", 0)) != 2 or int(suite.get("repeats", 0)) != 7:
        raise ValueError(
            "hardware_profile_violation: study requires two warmups and seven repeats"
        )
    case_ids = tuple(str(case.get("case_id")) for case in suite.get("cases", ()))
    if case_ids != HARDWARE_TASKGRAPH_STUDY_WORKLOAD_IDS:
        raise ValueError(
            "hardware_profile_violation: study workload set differs from the fixed one-DPU matrix"
        )
    for case in suite.get("cases", ()):
        if case.get("hardware_numeric_coverage") not in {"real", "split_complex"}:
            raise ValueError(
                "hardware_profile_violation: each study workload needs numeric coverage"
            )
    variants = _parse_variants(raw.get("path_variants"))
    return HardwareTaskGraphStudySuite(path.resolve(), suite, profile, variants)


def hardware_taskgraph_study_profile_metadata(
    profile: HardwareTaskGraphStudyProfile,
) -> JsonDict:
    return {
        **profile.to_json_dict(),
        "hardware_functionality_evidence": True,
        "hardware_timing_available": True,
        "timing_is_bringup_only": False,
        "within_route_timing_comparison_allowed": True,
        "hardware_speedup_applicable": False,
        "cross_backend_speedup_applicable": False,
        "native_build_reuse_required": True,
        "persistent_session_reuse_required": True,
        "session_scope": "case_benchmark_block",
        "upmem_parallelism_mode": "sequential",
        "upmem_parallelism_evidence_type": "hardware_executed",
        "task_assignment_strategy": "sequential_single_dpu",
        "dpu_group_count": 1,
        "multi_dpu_execution": False,
        "claim_boundary": (
            "one-DPU steady-state path and numeric-mode ablation; no CPU/GPU speedup, "
            "energy, parallel scheduling, or multi-DPU claim"
        ),
    }


def validate_hardware_taskgraph_study_execution_request(
    *, execute: bool, environment: Mapping[str, str] | None = None
) -> None:
    env = environment or {}
    if not execute:
        raise ValueError("hardware TaskGraph study execution requires --execute")
    if env.get("UPMEM_ALLOW_PHYSICAL_HARDWARE") != "1":
        raise ValueError(
            "UPMEM_ALLOW_PHYSICAL_HARDWARE=1 is required for physical UPMEM execution"
        )
    if env.get("DPU_BACKEND"):
        raise ValueError(
            "DPU_BACKEND must be unset for physical TaskGraph study execution"
        )


def _parse_profile(value: object) -> HardwareTaskGraphStudyProfile:
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
        if value.get(key) != expected_value:
            raise ValueError(
                f"hardware_profile_violation: {key} must be {expected_value!r}"
            )
    return _canonical_profile()


def _parse_variants(value: object) -> tuple[HardwareTaskGraphStudyVariant, ...]:
    if not isinstance(value, list) or len(value) != len(
        HARDWARE_TASKGRAPH_STUDY_VARIANTS
    ):
        raise ValueError(
            "hardware_profile_violation: study requires exactly two path variants"
        )
    parsed: list[HardwareTaskGraphStudyVariant] = []
    for item in value:
        if not isinstance(item, Mapping):
            raise ValueError(
                "hardware_profile_violation: path variants must be mappings"
            )
        if set(item) != {"id", "label", "planner"}:
            raise ValueError("hardware_profile_violation: invalid path variant fields")
        planner = item.get("planner")
        if not isinstance(planner, Mapping):
            raise ValueError(
                "hardware_profile_violation: path variant planner must be a mapping"
            )
        parsed.append(
            HardwareTaskGraphStudyVariant(
                variant_id=str(item.get("id") or ""),
                label=str(item.get("label") or ""),
                planner=dict(planner),
            )
        )
    if tuple(item.variant_id for item in parsed) != HARDWARE_TASKGRAPH_STUDY_VARIANTS:
        raise ValueError("hardware_profile_violation: unsupported path variant IDs")
    expected = {
        "opt_einsum_greedy": {"engine": "opt_einsum", "optimize": "greedy"},
        "custom_upmem_v2_balanced": {
            "engine": "custom_upmem",
            "algorithm": "greedy",
            "objective_version": "upmem_path_cost_v2",
            "selection_scope": "projected_prefix",
            "weight_profile": "balanced_literature_informed",
            "normalization": "fixed_log1p_generic_budgets_v2",
            "execution_policy": "generic_single_dpu_split_complex_v2",
        },
    }
    for item in parsed:
        if item.planner != expected[item.variant_id]:
            raise ValueError(
                f"hardware_profile_violation: planner for {item.variant_id} is not fixed"
            )
        if not item.label:
            raise ValueError(
                "hardware_profile_violation: path variant label is required"
            )
    return tuple(parsed)


def _canonical_profile() -> HardwareTaskGraphStudyProfile:
    return HardwareTaskGraphStudyProfile(
        version=HARDWARE_TASKGRAPH_STUDY_PROFILE_VERSION,
        target="hardware",
        backend_id=HARDWARE_TASKGRAPH_STUDY_BACKEND_ID,
        route_id=HARDWARE_TASKGRAPH_STUDY_ROUTE_ID,
        session_protocol=HARDWARE_TASKGRAPH_STUDY_SESSION_PROTOCOL,
        requested_dpu_count=1,
        tasklets_per_dpu=1,
        max_rank=HARDWARE_TASKGRAPH_STUDY_MAX_RANK,
        max_tensor_elements=HARDWARE_TASKGRAPH_STUDY_MAX_TENSOR_ELEMENTS,
        max_contracted_combinations=HARDWARE_TASKGRAPH_STUDY_MAX_CONTRACTED_COMBINATIONS,
        output_tile_elements=HARDWARE_TASKGRAPH_STUDY_OUTPUT_TILE_ELEMENTS,
        numeric_modes=HARDWARE_TASKGRAPH_STUDY_NUMERIC_MODES,
        complex_policy=HARDWARE_TASKGRAPH_STUDY_COMPLEX_POLICY,
        synchronous_execution=True,
        timeout_s=HARDWARE_TASKGRAPH_STUDY_TIMEOUT_S,
        performance_claim_applicable=False,
    )
