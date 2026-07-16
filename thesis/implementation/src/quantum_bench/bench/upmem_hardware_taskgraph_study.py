"""Preparation and execution entry points for the physical one-DPU study.

The existing ``upmem-hardware-taskgraph`` command is correctness-only: it
opens a physical session for every logical contraction.  This module is an
additive route for a deliberately narrow timing study.  It prepares two fixed
contraction paths for the same circuit-derived tensor network and later runs
them through one persistent native session per circuit case.

Nothing here treats the resulting timings as cross-backend speedup or energy
evidence.  Its only comparison surface is the selected path and numeric mode
inside the same one-DPU execution profile.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shutil
from typing import Any, Mapping

import numpy as np

from quantum_bench.bench.run_dirs import (
    EVIDENCE_ARTIFACT_KIND,
    create_run_dir,
    sanitize,
)
from quantum_bench.circuits import load_circuit
from quantum_bench.core.jsonio import write_json
from quantum_bench.core.records import JsonDict
from quantum_bench.environment import capture_environment
from quantum_bench.routing.generic_prepare import (
    GenericTaskPreparationCaps,
    generic_structural_feasibility,
)
from quantum_bench.targets.upmem.hardware_session import (
    HardwareSessionBuild,
    build_hardware_session,
)
from quantum_bench.targets.upmem.hardware_taskgraph_study import (
    HardwareTaskGraphStudySuite,
    hardware_taskgraph_study_profile_metadata,
    load_hardware_taskgraph_study_suite,
    validate_hardware_taskgraph_study_execution_request,
)
from quantum_bench.tn import (
    build_execution_bundle,
    build_tensor_network,
    contraction_path_structure_hash,
    execute_task_sequence_np_einsum,
    plan_task_graph_with_config,
    with_execution_identity,
)


UPMEM_HARDWARE_TASKGRAPH_STUDY_PLAN_SCHEMA_VERSION = (
    "upmem_hardware_taskgraph_study_plan_v1"
)
UPMEM_HARDWARE_TASKGRAPH_STUDY_BENCHMARK_SCHEMA_VERSION = (
    "upmem_hardware_taskgraph_study_benchmark_v1"
)


@dataclass(frozen=True)
class UpmemHardwareTaskGraphStudyPlanResult:
    plan_dir: Path
    summary_path: Path
    status: str


@dataclass(frozen=True)
class UpmemHardwareTaskGraphStudyResult:
    run_dir: Path
    summary_path: Path
    status: str
    row_count: int


def prepare_upmem_hardware_taskgraph_study(
    root_dir: Path,
    *,
    suite_path: Path,
    build: bool = False,
    environment: Mapping[str, str] | None = None,
) -> UpmemHardwareTaskGraphStudyPlanResult:
    """Prepare the fixed study without allocating or launching a DPU."""

    suite = load_hardware_taskgraph_study_suite(suite_path)
    env = dict(os.environ if environment is None else environment)
    plan_dir = _unique_dir(root_dir / "build" / "upmem_hardware_taskgraph_study_plan")
    plan_dir.mkdir(parents=True, exist_ok=False)
    config_dir = plan_dir / "config"
    config_dir.mkdir()
    shutil.copy2(suite.suite_path, config_dir / "resolved_suite.yml")
    write_json(
        config_dir / "hardware_profile.json",
        hardware_taskgraph_study_profile_metadata(suite.profile),
    )
    write_json(plan_dir / "environment.json", capture_environment(root_dir))

    case_rows: list[JsonDict] = []
    status = "prepared"
    failure_stage: str | None = None
    for case in suite.suite["cases"]:
        try:
            prepared = prepare_study_case(
                root_dir,
                plan_dir / "cases" / sanitize(str(case["case_id"])),
                suite,
                case,
            )
            case_rows.append(_prepared_case_row(case, prepared))
        except Exception as exc:
            status = "failed"
            failure_stage = "hardware_profile_violation"
            case_rows.append(
                {"case_id": case.get("case_id"), "status": "failed", "reason": str(exc)}
            )
            break

    native_build: JsonDict = {"attempted": False, "status": "not_requested"}
    if status == "prepared" and build:
        try:
            native = build_hardware_session(
                root_dir,
                plan_dir / "native_session",
                profile=suite.profile,
                environment=env,
            )
            native_build = _native_build_metadata(native, plan_dir)
        except Exception as exc:
            status = "failed"
            failure_stage = _failure_stage(str(exc), default="native_build_failed")
            native_build = {"attempted": True, "status": "failed", "reason": str(exc)}

    summary_path = plan_dir / "upmem_hardware_taskgraph_study_plan.json"
    write_json(
        summary_path,
        {
            "schema_version": UPMEM_HARDWARE_TASKGRAPH_STUDY_PLAN_SCHEMA_VERSION,
            "status": status,
            "failure_stage": failure_stage,
            "suite_id": suite.suite["suite_id"],
            "suite_path": str(suite.suite_path),
            "profile": hardware_taskgraph_study_profile_metadata(suite.profile),
            "prepared_cases": case_rows,
            "native_build": native_build,
            "dpu_allocation_attempted": False,
            "dpu_launch_attempted": False,
            "notes": [
                "Preparation verifies both fixed paths against the same circuit and tensor-network identity.",
                "Preparation validates the generic-loop structural caps and never allocates or launches a DPU.",
                "The custom UPMEM v2 path is modeled planner selection evidence; the study executes that selected path under both numeric modes without claiming an int8-aware planner objective.",
            ],
        },
    )
    return UpmemHardwareTaskGraphStudyPlanResult(plan_dir, summary_path, status)


def prepare_study_case(
    root_dir: Path,
    case_dir: Path,
    suite: HardwareTaskGraphStudySuite,
    case: Mapping[str, Any],
) -> JsonDict:
    """Lower both required paths and prove their pre-execution invariants."""

    case_dir.mkdir(parents=True, exist_ok=True)
    circuit = load_circuit(dict(case), root_dir)
    network = build_tensor_network(circuit)
    caps = GenericTaskPreparationCaps(
        max_rank=suite.profile.max_rank,
        max_tensor_elements=suite.profile.max_tensor_elements,
        max_contracted_combinations=suite.profile.max_contracted_combinations,
    )
    variants: dict[str, JsonDict] = {}
    reference_output: np.ndarray | None = None
    circuit_hash: str | None = None
    network_hash: str | None = None
    structures: set[str] = set()

    for variant in suite.variants:
        graph = with_execution_identity(
            plan_task_graph_with_config(network, dict(variant.planner))
        )
        output, execution_metrics = execute_task_sequence_np_einsum(graph, network)
        if reference_output is None:
            reference_output = np.asarray(output)
        elif not np.allclose(
            np.asarray(output), reference_output, rtol=1e-10, atol=1e-10
        ):
            raise ValueError(
                "hardware_profile_violation: fixed path variants do not reconstruct the same CPU tensor"
            )
        if circuit_hash is None:
            circuit_hash = graph.circuit_semantics_hash
            network_hash = graph.tensor_network_hash
        elif (
            graph.circuit_semantics_hash != circuit_hash
            or graph.tensor_network_hash != network_hash
        ):
            raise ValueError(
                "hardware_profile_violation: path variants must share circuit and tensor-network identity"
            )

        structural_rows: list[JsonDict] = []
        for task in graph.tasks:
            feasibility = generic_structural_feasibility(
                task,
                caps,
                check_int32_accumulation=True,
            )
            structural_rows.append(
                {
                    "task_id": task.id,
                    "feasible": feasibility.feasible,
                    "rejection_reasons": list(feasibility.rejection_reasons),
                    **feasibility.metadata,
                }
            )
            if not feasibility.feasible:
                raise ValueError(
                    "hardware_profile_violation: generic-loop caps reject "
                    f"{variant.variant_id}/{task.id}: {','.join(feasibility.rejection_reasons)}"
                )

        variant_dir = case_dir / "paths" / sanitize(variant.variant_id)
        variant_dir.mkdir(parents=True, exist_ok=False)
        bundle = build_execution_bundle(
            graph, case_id=str(case["case_id"]), suite_id=str(suite.suite["suite_id"])
        )
        bundle_path = variant_dir / "execution_bundle.json"
        write_json(bundle_path, bundle)
        output_path = variant_dir / "cpu_reference_final_tensor.npy"
        np.save(output_path, np.asarray(output), allow_pickle=False)
        structure_hash = contraction_path_structure_hash(graph)
        structures.add(structure_hash)
        write_json(
            variant_dir / "path_preparation.json",
            {
                "case_id": case["case_id"],
                "path_variant_id": variant.variant_id,
                "path_variant_label": variant.label,
                "planner": variant.planner,
                "task_count": len(graph.tasks),
                "execution_bundle": bundle_path.name,
                "cpu_reference_final_tensor": output_path.name,
                "reference_execution": execution_metrics,
                "circuit_semantics_hash": graph.circuit_semantics_hash,
                "tensor_network_hash": graph.tensor_network_hash,
                "contraction_plan_hash": graph.contraction_plan_hash,
                "contraction_path_structure_hash": structure_hash,
                "generic_structural_feasibility": structural_rows,
                "planner_numeric_scope": (
                    "custom_upmem_v2 models float32/split-complex path pressure; "
                    "per-task int8 is an executed numeric-mode ablation, not an int8 planner claim"
                ),
            },
        )
        variants[variant.variant_id] = {
            "variant": variant,
            "graph": graph,
            "reference_output": np.asarray(output),
            "bundle_path": bundle_path,
            "reference_path": output_path,
            "execution_metrics": execution_metrics,
            "structural_rows": structural_rows,
        }

    if len(structures) != len(suite.variants):
        raise ValueError(
            "hardware_profile_violation: fixed study variants did not produce distinct contraction paths"
        )
    return {
        "circuit": circuit,
        "network": network,
        "variants": variants,
        "circuit_semantics_hash": circuit_hash,
        "tensor_network_hash": network_hash,
        "path_structure_hashes": structures,
    }


def run_upmem_hardware_taskgraph_study(
    root_dir: Path,
    *,
    suite_path: Path,
    environment: Mapping[str, str] | None = None,
) -> UpmemHardwareTaskGraphStudyResult:
    """Run the physical study.

    The runtime implementation is imported lazily so `--prepare-only` remains
    fully usable in environments without a physical UPMEM SDK.
    """

    from quantum_bench.bench.upmem_hardware_taskgraph_study_runtime import (
        run_study_suite,
    )

    env = dict(os.environ if environment is None else environment)
    validate_hardware_taskgraph_study_execution_request(execute=True, environment=env)
    suite = load_hardware_taskgraph_study_suite(suite_path)
    run_dir = create_run_dir(
        root_dir,
        str(suite.suite["suite_id"]),
        artifact_kind=EVIDENCE_ARTIFACT_KIND,
        route_label="upmem_hw_taskgraph_study",
    )
    return run_study_suite(root_dir, run_dir, suite, environment=env)


def _prepared_case_row(
    case: Mapping[str, Any], prepared: Mapping[str, Any]
) -> JsonDict:
    variants = prepared["variants"]
    return {
        "case_id": case["case_id"],
        "status": "prepared",
        "n_qubits": prepared["circuit"].n_qubits,
        "hardware_numeric_coverage": case.get("hardware_numeric_coverage"),
        "circuit_semantics_hash": prepared["circuit_semantics_hash"],
        "tensor_network_hash": prepared["tensor_network_hash"],
        "path_variants": [
            {
                "path_variant_id": variant_id,
                "task_count": len(value["graph"].tasks),
                "contraction_plan_hash": value["graph"].contraction_plan_hash,
                "contraction_path_structure_hash": contraction_path_structure_hash(
                    value["graph"]
                ),
                "execution_bundle": str(
                    value["bundle_path"].relative_to(value["bundle_path"].parents[3])
                ),
            }
            for variant_id, value in variants.items()
        ],
    }


def _native_build_metadata(build: HardwareSessionBuild, root: Path) -> JsonDict:
    return {
        "attempted": True,
        "status": "passed",
        "source_tree_hash": build.source_tree_hash,
        "host_binary_hash": build.host_binary_hash,
        "dpu_binary_hash": build.dpu_binary_hash,
        "build_time_s": build.build_time_s,
        "build_command": list(build.build_command),
        "sdk_tools": build.sdk_tools,
        "session_root": str(build.session_root.relative_to(root))
        if build.session_root.is_relative_to(root)
        else str(build.session_root),
    }


def _failure_stage(error: str, *, default: str | None) -> str | None:
    known = (
        "hardware_opt_in_missing",
        "hardware_profile_violation",
        "sdk_discovery_failed",
        "native_build_failed",
        "hardware_allocation_failed",
        "binary_load_failed",
        "argument_transfer_failed",
        "operand_transfer_failed",
        "kernel_launch_failed",
        "kernel_timeout",
        "result_transfer_failed",
        "output_manifest_failed",
        "output_validation_failed",
        "hardware_release_failed",
    )
    for stage in known:
        if stage in error:
            return stage
    return default


def _unique_dir(parent: Path) -> Path:
    from datetime import datetime

    parent.mkdir(parents=True, exist_ok=True)
    base = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    candidate = parent / base
    suffix = 1
    while candidate.exists():
        candidate = parent / f"{base}_{suffix:02d}"
        suffix += 1
    return candidate
