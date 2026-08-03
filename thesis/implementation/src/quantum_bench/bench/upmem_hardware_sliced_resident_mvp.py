"""Internal research M2 runner for the fixed two-DPU sliced-resident MVP.

This is intentionally a narrow evidence command.  It does not provide a
general scheduler, a simulator route, retries, or performance claims.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any, Mapping

import numpy as np
import yaml

from quantum_bench.bench.config import load_suite
from quantum_bench.bench.reporting import write_normalized_records, write_run_manifest
from quantum_bench.bench.run_dirs import (
    EVIDENCE_ARTIFACT_KIND,
    create_run_dir,
    sanitize,
)
from quantum_bench.circuits import load_circuit
from quantum_bench.core.jsonio import write_json, write_jsonl
from quantum_bench.environment import capture_environment
from quantum_bench.targets.upmem.hardware_sliced_resident_session import (
    BACKEND_ID,
    ROUTE_ID,
    SlicedResidentHardwareProfile,
    build_sliced_resident_hardware_session,
    execute_sliced_resident_hardware_session,
    parse_sliced_resident_hardware_profile,
)
from quantum_bench.targets.upmem.hardware_taskgraph_sliced_resident import (
    build_two_slice_resident_graph_packages,
    build_two_slice_resident_plan,
    load_and_reconstruct_two_slice_native_outputs,
    reconstruct_host_slice_outputs,
    validate_written_two_slice_packages,
    write_two_slice_resident_graph_packages,
)
from quantum_bench.targets.upmem.hardware_taskgraph_resident import (
    build_resident_policy_reference,
    validate_resident_graph_package_file,
)
from quantum_bench.tn import (
    build_tensor_network,
    execute_task_sequence_np_einsum,
    plan_task_graph_with_config,
    executor_config_hash,
    with_execution_identity,
)
from quantum_bench.tn.contract import contract_binary_task
from quantum_bench.tn.execution import order_final_tensor
from quantum_bench.tn.slicing import (
    SliceInputRestriction,
    build_slice_aware_taskgraph_model,
)


MVP_SCHEMA_VERSION = "upmem_hardware_sliced_resident_m2_v1"
M2_1_SCHEMA_VERSION = "upmem_hardware_sliced_resident_m2_1_v1"
M2_2_SCHEMA_VERSION = "upmem_hardware_sliced_resident_m2_2_v1"
PLAN_SCHEMA_VERSION = "upmem_hardware_sliced_resident_mvp_plan_v1"
RUNTIME_SCHEMA_VERSION = "upmem_hardware_sliced_resident_mvp_runtime_v1"
ROUTE_LABEL = "upmem_hw_sliced_resident"
CLAIM_BOUNDARY = "internal/research MVP only; no speedup claim and no energy claim"
M2_2_CLAIM_BOUNDARY = (
    "internal/research M2.2 only; no speedup, scaling, or energy claim"
)
_WORKLOAD_IDS = ("one_qubit_x_m2", "one_qubit_h_m2", "one_qubit_z_m2")
_M2_1_WORKLOAD_IDS = ("one_qubit_hx_m2_1",)
_M2_2_WORKLOAD_IDS = ("one_qubit_hx_m2_2",)
IMPLEMENTATION_ROOT = Path(__file__).resolve().parents[3]
CANONICAL_M2_SUITE_PATH = (
    IMPLEMENTATION_ROOT
    / "configs"
    / "suites"
    / "upmem_hardware_sliced_resident_mvp.yml"
)
CANONICAL_M2_1_SUITE_PATH = (
    IMPLEMENTATION_ROOT
    / "configs"
    / "suites"
    / "upmem_hardware_sliced_resident_m2_1.yml"
)
CANONICAL_M2_2_SUITE_PATH = (
    IMPLEMENTATION_ROOT
    / "configs"
    / "suites"
    / "upmem_hardware_sliced_resident_m2_2.yml"
)
_CANONICAL_QASM_PATHS = {
    "one_qubit_x_m2": "configs/circuits/upmem_m2/one_qubit_x.qasm",
    "one_qubit_h_m2": "configs/circuits/upmem_m2/one_qubit_h.qasm",
    "one_qubit_z_m2": "configs/circuits/upmem_m2/one_qubit_z.qasm",
}
_M2_1_QASM_PATHS = {
    "one_qubit_hx_m2_1": "configs/circuits/upmem_m2/one_qubit_hx.qasm",
}
_M2_2_QASM_PATHS = {
    "one_qubit_hx_m2_2": "configs/circuits/upmem_m2/one_qubit_hx.qasm",
}
M2_2_NUMERIC_MODES = ("none", "per_task_resident_requantize")
SLICE_NONZERO_THRESHOLD = 1.0e-7
M2_2_NONE_VALIDATION_TOLERANCE = 1.0e-6
M2_2_REQUANTIZED_VALIDATION_TOLERANCE = 1.0e-2


@dataclass(frozen=True)
class M2Suite:
    path: Path
    suite: dict[str, Any]
    raw: dict[str, Any]
    profile: SlicedResidentHardwareProfile
    fixture_version: str
    fixture_scope: str
    require_nonzero_slice_partials: bool
    numeric_modes: tuple[str, ...] = ("none",)

    @property
    def is_m2_2(self) -> bool:
        return self.fixture_version == M2_2_SCHEMA_VERSION


@dataclass(frozen=True)
class M2PlanResult:
    plan_dir: Path
    summary_path: Path
    status: str


@dataclass(frozen=True)
class M2RunResult:
    run_dir: Path
    summary_path: Path
    status: str
    row_count: int


def _claim_boundary(m2: M2Suite) -> str:
    return M2_2_CLAIM_BOUNDARY if m2.is_m2_2 else CLAIM_BOUNDARY


def load_m2_suite(path: Path) -> M2Suite:
    """Load one of the committed, deliberately narrow M2 fixtures."""

    resolved_path = path.resolve()
    if resolved_path not in {
        CANONICAL_M2_SUITE_PATH.resolve(),
        CANONICAL_M2_1_SUITE_PATH.resolve(),
        CANONICAL_M2_2_SUITE_PATH.resolve(),
    }:
        raise ValueError(
            "hardware_profile_violation: M2 suite must be one of the committed "
            "sliced-resident fixtures"
        )
    is_m2_1 = resolved_path == CANONICAL_M2_1_SUITE_PATH.resolve()
    is_m2_2 = resolved_path == CANONICAL_M2_2_SUITE_PATH.resolve()
    expected_suite_id = (
        "upmem_hardware_sliced_resident_m2_2"
        if is_m2_2
        else "upmem_hardware_sliced_resident_m2_1"
        if is_m2_1
        else "upmem_hardware_sliced_resident_mvp"
    )
    expected_schema = (
        M2_2_SCHEMA_VERSION
        if is_m2_2
        else M2_1_SCHEMA_VERSION
        if is_m2_1
        else MVP_SCHEMA_VERSION
    )
    expected_workload_ids = (
        _M2_2_WORKLOAD_IDS
        if is_m2_2
        else _M2_1_WORKLOAD_IDS
        if is_m2_1
        else _WORKLOAD_IDS
    )
    expected_qasm_paths = (
        _M2_2_QASM_PATHS
        if is_m2_2
        else _M2_1_QASM_PATHS
        if is_m2_1
        else _CANONICAL_QASM_PATHS
    )
    expected_numeric_modes = M2_2_NUMERIC_MODES if is_m2_2 else ("none",)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("hardware_profile_violation: M2 suite must be a mapping")
    suite = load_suite(path)
    metadata = raw.get("metadata")
    routes = raw.get("routes")
    workloads = raw.get("workloads")
    if (
        raw.get("schema_version") != 2
        or raw.get("suite_id") != expected_suite_id
        or not isinstance(metadata, dict)
        or metadata.get(
            "hardware_sliced_resident_m2_2_schema_version"
            if is_m2_2
            else "hardware_sliced_resident_m2_1_schema_version"
            if is_m2_1
            else "hardware_sliced_resident_m2_schema_version"
        )
        != expected_schema
        or (
            not is_m2_2
            and metadata.get("quantization_mode") != "none"
        )
        or (
            is_m2_2
            and metadata.get("numeric_modes") != list(expected_numeric_modes)
        )
        or raw.get("defaults", {}).get("warmups") != 1
        or raw.get("defaults", {}).get("repeats")
        != (5 if is_m2_2 else 3)
        or raw.get("defaults", {}).get("planner")
        != {"engine": "opt_einsum", "optimize": "greedy"}
        or not isinstance(routes, list)
        or len(routes) != 1
        or not isinstance(workloads, list)
        or len(workloads) != len(expected_workload_ids)
        or tuple(item.get("id") for item in workloads if isinstance(item, dict))
        != expected_workload_ids
    ):
        raise ValueError(
            "hardware_profile_violation: suite is not the committed M2 MVP schema"
        )
    route = routes[0]
    options = route.get("options") if isinstance(route, dict) else None
    expected_route_options = {
        "backend_id": BACKEND_ID,
        "slices": 2,
        "requested_dpu_count": 2,
        "tasklets_per_dpu": 1,
    }
    if not is_m2_2:
        expected_route_options["quantization_mode"] = "none"
    else:
        expected_route_options["numeric_modes"] = list(expected_numeric_modes)
    if (
        not isinstance(options, dict)
        or route.get("id") != ROUTE_ID
        or options != expected_route_options
    ):
        raise ValueError(
            "hardware_profile_violation: M2 route differs from committed route"
        )
    profile = parse_sliced_resident_hardware_profile(
        metadata.get("hardware_profile", {}),
        allowed_numeric_modes=expected_numeric_modes,
    )
    for workload in workloads:
        circuit = workload.get("circuit") if isinstance(workload, dict) else None
        workload_id = workload.get("id") if isinstance(workload, dict) else None
        if (
            not isinstance(circuit, dict)
            or circuit.get("kind") != "qasm_file"
            or circuit.get("path") != expected_qasm_paths.get(workload_id)
            or circuit.get("name")
            != Path(str(expected_qasm_paths.get(workload_id))).stem
            or not isinstance(workload.get("expected_output"), list)
        ):
            raise ValueError(
                "hardware_profile_violation: M2 workloads must use their canonical QASM paths and expected output"
            )
    fixture_scope = str(
        metadata.get(
            "fixture_scope",
            "single_gate_operator_on_zero_initial_state"
            if not (is_m2_1 or is_m2_2)
            else "single_gate_operator_on_prepared_real_input_state",
        )
    )
    require_nonzero = bool(metadata.get("require_nonzero_slice_partials", False))
    if (is_m2_1 or is_m2_2) and not require_nonzero:
        raise ValueError(
            "hardware_profile_violation: M2.1 must require nonzero slice partials"
        )
    return M2Suite(
        path.resolve(),
        suite,
        raw,
        profile,
        M2_2_SCHEMA_VERSION
        if is_m2_2
        else M2_1_SCHEMA_VERSION
        if is_m2_1
        else MVP_SCHEMA_VERSION,
        fixture_scope,
        require_nonzero,
        expected_numeric_modes,
    )


def prepare_upmem_hardware_sliced_resident_mvp(
    root_dir: Path,
    *,
    suite_path: Path,
    build: bool = False,
    environment: Mapping[str, str] | None = None,
) -> M2PlanResult:
    """Materialize plans and package manifests without invoking the adapter."""

    m2 = load_m2_suite(suite_path)
    env = dict(os.environ if environment is None else environment)
    plan_dir = _unique_dir(root_dir / "build" / "upmem_hardware_sliced_resident_mvp")
    plan_dir.mkdir(parents=True)
    _write_common_artifacts(plan_dir, root_dir, m2)
    native = {"attempted": False, "status": "not_requested"}
    native_manifest_validation: dict[str, Any] = {
        "status": "not_run",
        "reason": "native host build was not requested",
        "entries": [],
        "dpu_allocation_attempted": False,
        "dpu_launch_attempted": False,
    }
    dpu_binary = plan_dir / "native_session" / "unbuilt_dpu_resident_two_dpu"
    dpu_binary.parent.mkdir(parents=True, exist_ok=True)
    dpu_binary.touch()
    if build:
        try:
            built = build_sliced_resident_hardware_session(
                root_dir,
                plan_dir / "native_session",
                profile=m2.profile,
                environment=env,
            )
            dpu_binary = built.dpu_binary
            native = _build_metadata(built, plan_dir)
        except Exception as exc:  # Build failures are retained as a plan result.
            native = {
                "attempted": True,
                "status": "failed",
                "failure_stage": _stage(str(exc), "native_build_failed"),
                "reason": str(exc),
            }
            native_manifest_validation = {
                "status": "unavailable",
                "reason": "native host build failed",
                "entries": [],
                "dpu_allocation_attempted": False,
                "dpu_launch_attempted": False,
            }
    rows: list[dict[str, Any]] = []
    if native.get("status") != "failed":
        for case in m2.suite["cases"]:
            try:
                prepared = _prepare_case(_suite_root(m2), case, m2)
                modes = m2.numeric_modes if m2.is_m2_2 else ("none",)
                for numeric_mode in modes:
                    for phase, repeat_id in _phase_ids(m2.suite):
                        artifact_dir = _artifact_dir(
                            plan_dir,
                            str(case["case_id"]),
                            phase,
                            repeat_id,
                            numeric_mode=numeric_mode if m2.is_m2_2 else None,
                        )
                        artifacts = _write_packages(
                            prepared,
                            m2,
                            dpu_binary,
                            artifact_dir,
                            # Keep semantic request IDs identical to execution.
                            # The plan root remains separate from the evidence
                            # root, but the native parser must see the same
                            # request-id shape and length during preflight.
                            prefix="execute",
                            numeric_mode=numeric_mode,
                        )
                        if native.get("status") == "passed":
                            validation = _validate_native_manifests(
                                native["host_binary_path"],
                                artifacts["manifest_paths"],
                                timeout_s=float(m2.profile.timeout_s),
                            )
                            validation["numeric_mode"] = numeric_mode
                            artifacts["native_manifest_validation"] = validation
                            write_json(
                                artifact_dir / "native_manifest_validation.json",
                                validation,
                            )
                            native_manifest_validation["entries"].append(validation)
                            if validation["status"] != "passed":
                                row = _plan_row(
                                    case,
                                    prepared,
                                    phase,
                                    repeat_id,
                                    artifacts,
                                    numeric_mode=numeric_mode,
                                )
                                row.update(
                                    {
                                        "status": "failed",
                                        "failure_stage": validation.get(
                                            "failure_stage",
                                            "native_manifest_validation_failed",
                                        ),
                                        "reason": validation.get("reason"),
                                    }
                                )
                                rows.append(row)
                                continue
                        rows.append(
                            _plan_row(
                                case,
                                prepared,
                                phase,
                                repeat_id,
                                artifacts,
                                numeric_mode=numeric_mode,
                            )
                        )
            except Exception as exc:
                rows.append(
                    {
                        "case_id": case.get("case_id"),
                        "status": "failed",
                        "failure_stage": _stage(str(exc), "hardware_profile_violation"),
                        "reason": str(exc),
                    }
                )
    if native.get("status") == "passed":
        entries = native_manifest_validation["entries"]
        native_manifest_validation["status"] = (
            "passed"
            if entries and all(item.get("status") == "passed" for item in entries)
            else "failed"
        )
        native_manifest_validation["reason"] = (
            None
            if native_manifest_validation["status"] == "passed"
            else "native manifest parser rejected one or more packages"
        )
    status = (
        "prepared"
        if native.get("status") != "failed"
        and all(row.get("status") != "failed" for row in rows)
        else "failed"
    )
    summary_path = plan_dir / "upmem_hardware_sliced_resident_mvp_plan.json"
    write_json(
        summary_path,
        {
            "schema_version": PLAN_SCHEMA_VERSION,
            "status": status,
            "suite_id": m2.suite["suite_id"],
            "route_id": ROUTE_ID,
            "backend_id": BACKEND_ID,
            "profile": _profile_metadata(
                m2.profile,
                numeric_modes=m2.numeric_modes if m2.is_m2_2 else None,
            ),
            "prepared_operations": rows,
            "native_build": native,
            "native_manifest_validation": native_manifest_validation,
            "dpu_allocation_attempted": False,
            "dpu_launch_attempted": False,
            "claim_boundary": _claim_boundary(m2),
        },
    )
    return M2PlanResult(plan_dir, summary_path, status)


def run_upmem_hardware_sliced_resident_mvp(
    root_dir: Path,
    *,
    suite_path: Path,
    environment: Mapping[str, str] | None = None,
) -> M2RunResult:
    m2 = load_m2_suite(suite_path)
    env = dict(os.environ if environment is None else environment)
    run_dir = create_run_dir(
        root_dir,
        str(m2.suite["suite_id"]),
        artifact_kind=EVIDENCE_ARTIFACT_KIND,
        route_label=ROUTE_LABEL,
    )
    _write_common_artifacts(run_dir, root_dir, m2)
    manifest = write_run_manifest(
        run_dir,
        run_kind="internal_research_upmem_hardware_sliced_resident_mvp",
        suite_id=str(m2.suite["suite_id"]),
        suite_path=str(m2.path),
        artifact_kind=EVIDENCE_ARTIFACT_KIND,
        route_label=ROUTE_LABEL,
        route_id=ROUTE_ID,
        backend_id=BACKEND_ID,
        execution_scope=(
            "physical_two_dpu_two_slice_full_replicated_prefix_taskgraph"
            if m2.fixture_version in {M2_1_SCHEMA_VERSION, M2_2_SCHEMA_VERSION}
            else "physical_two_dpu_two_slice_terminal_contraction"
        ),
        evidence_type="physical_hardware_internal_research_mvp",
        upmem_execution_mode="two_dpu_sliced_resident",
        quantization_mode=None if m2.is_m2_2 else "none",
        quantization_modes=m2.numeric_modes if m2.is_m2_2 else (),
        artifact_retention="full",
        summary="upmem_hardware_sliced_resident_mvp_summary.json",
        root_dir=root_dir,
    )
    manifest.update(
        {
            "execution_model": (
                "dependent_prefix_replicated"
                if m2.fixture_version in {M2_1_SCHEMA_VERSION, M2_2_SCHEMA_VERSION}
                else "terminal_contraction_input_restriction"
            ),
            "operation_count": None,
            "fixture_version": m2.fixture_version,
            "fixture_scope": m2.fixture_scope,
            **(
                {"numeric_modes": list(m2.numeric_modes)}
                if m2.is_m2_2
                else {}
            ),
        }
    )
    records: list[dict[str, Any]] = []
    warmups: list[dict[str, Any]] = []
    if env.get("UPMEM_ALLOW_PHYSICAL_HARDWARE") != "1":
        records.append(
            _failure_record(
                m2,
                "hardware_opt_in_missing: UPMEM_ALLOW_PHYSICAL_HARDWARE=1 is required",
                None,
                "execute",
                None,
                "hardware_opt_in_missing",
            )
        )
        return _finish_run(run_dir, manifest, m2, records, warmups, native=None)
    try:
        native = build_sliced_resident_hardware_session(
            root_dir, run_dir / "native_session", profile=m2.profile, environment=env
        )
    except Exception as exc:
        records.append(
            _failure_record(
                m2,
                str(exc),
                None,
                "execute",
                None,
                _stage(str(exc), "native_build_failed"),
            )
        )
        return _finish_run(run_dir, manifest, m2, records, warmups, native=None)
    for case in m2.suite["cases"]:
        try:
            prepared = _prepare_case(_suite_root(m2), case, m2)
        except Exception as exc:
            records.append(
                _failure_record(
                    m2,
                    str(exc),
                    case,
                    "prepare",
                    None,
                    _stage(str(exc), "hardware_profile_violation"),
                )
            )
            continue
        manifest.update(
            {
                "operation_count": len(prepared["graph"].tasks),
                "source_task_count": prepared["source_task_count"],
            }
        )
        for phase, repeat_id in _phase_ids(m2.suite):
            for numeric_mode in (m2.numeric_modes if m2.is_m2_2 else ("none",)):
                record = _run_operation(
                    run_dir,
                    native,
                    m2,
                    case,
                    prepared,
                    phase,
                    repeat_id,
                    env,
                    numeric_mode=numeric_mode,
                )
                if phase == "warmup":
                    warmups.append(record)
                else:
                    records.append(record)
    return _finish_run(run_dir, manifest, m2, records, warmups, native=native)


def _prepare_case(
    root_dir: Path, case: Mapping[str, Any], m2: M2Suite
) -> dict[str, Any]:
    circuit = load_circuit(dict(case), root_dir)
    network = build_tensor_network(circuit)
    graph = with_execution_identity(
        plan_task_graph_with_config(
            network, {"engine": "opt_einsum", "optimize": "greedy"}
        )
    )
    is_two_operation_fixture = m2.fixture_version in {
        M2_1_SCHEMA_VERSION,
        M2_2_SCHEMA_VERSION,
    }
    if is_two_operation_fixture:
        if (
            circuit.n_qubits != 1
            or len(circuit.operations) != 2
            or len(graph.tasks) != 2
            or graph.tasks[0].dependencies
            or graph.tasks[1].dependencies != (graph.tasks[0].id,)
        ):
            raise ValueError(
                "hardware_profile_violation: strict M2.1/M2.2 requires a two-operation H-X dependent TaskGraph"
            )
        task = graph.tasks[-1]
        if task.gemm_k != 2 or not task.contracted_labels:
            raise ValueError(
                "hardware_profile_violation: strict M2.1/M2.2 requires a terminal dimension-2 contraction"
            )
        model = build_slice_aware_taskgraph_model(
            graph, max_slice_count=2, sliced_task_id=task.id
        )
        restrictions_by_slice = tuple(
            tuple(_m2_1_prefix_restrictions(graph, model.sliced_indices[0], slice_id))
            for slice_id in (0, 1)
        )
        if any(len(restrictions) != 2 for restrictions in restrictions_by_slice):
            raise ValueError(
                "hardware_profile_violation: strict M2.1/M2.2 requires two initial-input slice restrictions"
            )
        model = replace(
            model,
            slice_model_kind="dependent_prefix_replicated",
            slice_tasks=tuple(
                replace(slice_task, input_restrictions=restrictions)
                for slice_task, restrictions in zip(
                    model.slice_tasks, restrictions_by_slice, strict=True
                )
            ),
        )
    else:
        if circuit.n_qubits != 1 or len(circuit.operations) != 1 or len(graph.tasks) != 1:
            raise ValueError(
                "hardware_profile_violation: M2 requires a single-gate one-qubit terminal TaskGraph"
            )
        task = graph.tasks[0]
        if task.dependencies or task.gemm_k != 2 or not task.contracted_labels:
            raise ValueError(
                "hardware_profile_violation: M2 requires one terminal dimension-2 contraction"
            )
        model = None
    reference, _ = execute_task_sequence_np_einsum(graph, network)
    expected = np.asarray(case["expected_output"], dtype=np.complex128)
    if not np.allclose(reference, expected, atol=1.0e-12, rtol=1.0e-12):
        raise ValueError(
            "hardware_profile_violation: CPU TaskGraph reference differs from expected output"
        )
    plan = build_two_slice_resident_plan(graph, network, model=model)
    reference_partials = {
        item.slice_id: _independent_cpu_slice_reference(
            graph, network, item.slice_task.input_restrictions
        )
        for item in plan.slice_plans
    }
    source_path = _qasm_path(root_dir, case)
    return {
        "case_id": str(case["case_id"]),
        "circuit": circuit,
        "network": network,
        "graph": graph,
        "plan": plan,
        "reference": np.asarray(reference),
        "expected": expected,
        "reference_partials": reference_partials,
        "fixture_version": m2.fixture_version,
        "fixture_scope": m2.fixture_scope,
        "source_task_count": len(graph.tasks),
        "tensor_count": len(network.tensors),
        "selected_task_id": plan.model.sliced_task_id,
        "qasm_source_sha256": _file_hash(source_path),
        "source_path": str(source_path),
    }


def _m2_1_prefix_restrictions(
    graph: Any, label: int, slice_id: int
) -> tuple[SliceInputRestriction, ...]:
    restrictions: list[SliceInputRestriction] = []
    task_output_ids = {task.output_tensor_id for task in graph.tasks}
    for tensor in graph.network.tensors:
        if tensor.id in task_output_ids or label not in tensor.labels:
            continue
        restrictions.append(
            SliceInputRestriction(
                tensor_id=tensor.id,
                label=label,
                axis=tensor.labels.index(label),
                value=slice_id,
            )
        )
    return tuple(restrictions)


def _independent_cpu_slice_reference(
    graph: Any,
    network: Any,
    restrictions: tuple[SliceInputRestriction, ...],
) -> np.ndarray:
    """Execute a restricted source graph without using resident package lowering.

    This reference follows the original TaskGraph and only restricts source
    operands.  In particular, it never inserts a host-computed intermediate
    tensor into the package input set used by the physical route.
    """

    source_tensors = {tensor.spec.id: tensor for tensor in network.tensors}
    source_ids = set(source_tensors)
    intermediate_ids = {task.output_tensor_id for task in graph.tasks}
    restricted: dict[str, np.ndarray] = {}
    labels: dict[str, tuple[int, ...]] = {}
    for tensor_id, tensor in source_tensors.items():
        value = np.asarray(tensor.array, dtype=np.complex128)
        tensor_restrictions = [
            restriction
            for restriction in restrictions
            if restriction.tensor_id == tensor_id
        ]
        for restriction in tensor_restrictions:
            if (
                tensor_id not in source_ids
                or tensor_id in intermediate_ids
                or restriction.axis < 0
                or restriction.axis >= value.ndim
                or tensor.spec.labels[restriction.axis] != restriction.label
                or restriction.value < 0
                or restriction.value >= value.shape[restriction.axis]
            ):
                raise ValueError("m2_1_cpu_reference_restriction_invalid")
            value = np.take(value, [restriction.value], axis=restriction.axis)
        restricted[tensor_id] = value
        labels[tensor_id] = tensor.spec.labels

    for task in graph.tasks:
        left_id, right_id = task.input_tensor_ids
        if left_id not in restricted or right_id not in restricted:
            raise ValueError("m2_1_cpu_reference_missing_task_input")
        left = restricted[left_id]
        right = restricted[right_id]
        input_shapes = (tuple(left.shape), tuple(right.shape))
        dimensions: dict[int, int] = {}
        for task_labels, shape in zip(
            (task.left_labels, task.right_labels), input_shapes, strict=True
        ):
            for label, dimension in zip(task_labels, shape, strict=True):
                previous = dimensions.setdefault(int(label), int(dimension))
                if previous != int(dimension):
                    raise ValueError("m2_1_cpu_reference_label_dimension_mismatch")
        output_shape = tuple(dimensions[int(label)] for label in task.output_labels)
        dynamic_task = replace(
            task,
            input_shapes=input_shapes,
            output_shape=output_shape,
        )
        restricted[task.output_tensor_id] = contract_binary_task(
            dynamic_task, left, right
        )
        labels[task.output_tensor_id] = task.output_labels

    final_id = graph.tasks[-1].output_tensor_id
    output = restricted[final_id]
    final_labels = labels[final_id]
    if final_labels != graph.network.output_labels:
        output, _ = order_final_tensor(
            output, final_labels, graph.network.output_labels
        )
    output = np.asarray(output, dtype=np.complex128)
    if np.any(np.abs(output.imag) > 1.0e-12):
        raise ValueError("m2_1_cpu_reference_nonzero_imaginary_output")
    return np.asarray(output.real, dtype=np.float32)


def _run_operation(
    run_dir: Path,
    native: Any,
    m2: M2Suite,
    case: Mapping[str, Any],
    prepared: Mapping[str, Any],
    phase: str,
    repeat_id: int,
    environment: Mapping[str, str],
    *,
    numeric_mode: str = "none",
) -> dict[str, Any]:
    started = time.perf_counter()
    artifact_dir = _artifact_dir(
        run_dir,
        str(case["case_id"]),
        phase,
        repeat_id,
        numeric_mode=numeric_mode if m2.is_m2_2 else None,
    )
    artifacts: dict[str, Any] | None = None
    session: Any | None = None
    try:
        artifacts = _write_packages(
            prepared,
            m2,
            native.dpu_binary,
            artifact_dir,
            prefix="execute",
            numeric_mode=numeric_mode,
        )
        response_path = _response_path(
            native,
            str(case["case_id"]),
            phase,
            repeat_id,
            m2_2=m2.is_m2_2,
            numeric_mode=numeric_mode,
        )
        session = execute_sliced_resident_hardware_session(
            native,
            manifest_paths=artifacts["manifest_paths"],
            response_path=response_path,
            profile=m2.profile,
            environment=environment,
        )
        write_json(artifact_dir / "native_response.json", session.response)
        if session.status != "completed":
            raise RuntimeError(session.failure_stage or "hardware_session_failed")
        if m2.is_m2_2 and session.response.get("quantization_mode") != numeric_mode:
            raise RuntimeError("response_numeric_mode_mismatch")
        policy_output = None
        policy_metrics = None
        policy_partials = None
        if m2.is_m2_2:
            policy_partials = {}
            for item in artifacts["packages"]:
                policy = build_resident_policy_reference(
                    item.package.graph,
                    item.network,
                    quantization_mode=numeric_mode,
                )
                policy_partials[item.slice_id] = np.asarray(policy["output"])
            policy_output = reconstruct_host_slice_outputs(
                prepared["plan"], policy_partials
            )
        reconstruction_started = time.perf_counter()
        output, reconstruction = load_and_reconstruct_two_slice_native_outputs(
            prepared["plan"],
            artifacts["packages"],
            session.response_path,
            reference_partials=(
                policy_partials
                if policy_partials is not None
                else prepared.get("reference_partials")
            ),
        )
        reconstruction_time = time.perf_counter() - reconstruction_started
        np.save(artifact_dir / "reconstructed_output.npy", output)
        validation_tolerance = (
            M2_2_REQUANTIZED_VALIDATION_TOLERANCE
            if m2.is_m2_2 and numeric_mode == "per_task_resident_requantize"
            else M2_2_NONE_VALIDATION_TOLERANCE
        )
        cpu_ok = bool(
            np.allclose(
                output,
                prepared["reference"],
                atol=M2_2_NONE_VALIDATION_TOLERANCE,
                rtol=M2_2_NONE_VALIDATION_TOLERANCE,
            )
        )
        if m2.is_m2_2:
            policy_metrics = _accuracy_metrics(
                policy_output,
                output,
                tolerance=1.0e-5,
                reference_kind="dpu_mirroring_policy_reference",
            )
            full_precision_metrics = _accuracy_metrics(
                prepared["reference"],
                output,
                tolerance=validation_tolerance,
                reference_kind=(
                    "cpu_exact_taskgraph_full_precision"
                    if numeric_mode == "none"
                    else "cpu_exact_taskgraph_full_precision_with_fixed_hx_quantization_tolerance"
                ),
            )
            expected_ok = bool(
                np.allclose(
                    output,
                    prepared["expected"],
                    atol=validation_tolerance,
                    rtol=validation_tolerance,
                )
            )
        else:
            full_precision_metrics = None
            expected_ok = bool(
                np.allclose(output, prepared["expected"], atol=1.0e-6, rtol=1.0e-6)
            )
        status = "completed" if m2.is_m2_2 or (cpu_ok and expected_ok) else "failed"
        return _record(
            m2,
            case,
            prepared,
            phase,
            repeat_id,
            run_dir=run_dir,
            status=status,
            failure_stage=None if status == "completed" else "output_validation_failed",
            reason=None if status == "completed" else "output_validation_failed",
            native=native,
            session=session,
            artifacts=artifacts,
            reconstruction=reconstruction,
            output=output,
            cpu_ok=cpu_ok,
            expected_ok=expected_ok,
            numeric_mode=numeric_mode,
            policy_reference=policy_metrics,
            full_precision_accuracy=full_precision_metrics,
            reconstruction_time_s=reconstruction_time,
            total_time_s=time.perf_counter() - started,
        )
    except Exception as exc:
        return _failure_record(
            m2,
            str(exc),
            case,
            phase,
            repeat_id,
            _stage(str(exc), "operation_failed"),
            prepared=prepared,
            total_time_s=time.perf_counter() - started,
            numeric_mode=numeric_mode,
            operation_evidence=_operation_evidence(run_dir, session, artifacts),
            observed_response=(
                session.response
                if session is not None and isinstance(session.response, Mapping)
                else None
            ),
        )


def _write_packages(
    prepared: Mapping[str, Any],
    m2: M2Suite,
    dpu_binary: Path,
    artifact_dir: Path,
    *,
    prefix: str,
    numeric_mode: str = "none",
) -> dict[str, Any]:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    packages = build_two_slice_resident_graph_packages(
        prepared["plan"],
        case_id=prepared["case_id"],
        suite_id=m2.suite["suite_id"],
        quantization_mode=numeric_mode,
    )
    request_prefix = (
        f"{prefix}-{sanitize(artifact_dir.parent.name)}-{artifact_dir.name}"
    )
    written = write_two_slice_resident_graph_packages(
        packages,
        dpu_binary.parent,
        dpu_binary=dpu_binary,
        request_id_prefix=request_prefix,
    )
    mode_hash = _executor_config_hash(numeric_mode)
    for item in written:
        if item.package.manifest_path is None:
            raise ValueError("sliced_resident_package_write_incomplete")
        manifest = json.loads(item.package.manifest_path.read_text(encoding="utf-8"))
        manifest["executor_config_hash"] = mode_hash
        manifest["numeric_mode"] = numeric_mode
        binding = manifest.get("slice_execution")
        if isinstance(binding, dict):
            binding["numeric_mode"] = numeric_mode
            binding["executor_config_hash"] = mode_hash
        write_json(item.package.manifest_path, manifest)
    validation = validate_written_two_slice_packages(prepared["plan"], written)
    validation["binary_manifest_bindings"] = _validate_manifest_bindings_against_binary(
        written
    )
    manifest_paths = tuple(item.package.manifest_path for item in written)
    if any(path is None for path in manifest_paths):
        raise ValueError("sliced_resident_package_write_incomplete")
    write_json(artifact_dir / "slice_plan.json", prepared["plan"].to_json_dict())
    for item in written:
        shutil.copy2(
            item.package.manifest_path,
            artifact_dir / f"slice_{item.slice_id}_manifest.json",
        )
    write_json(artifact_dir / "package_preflight.json", validation)
    return {
        "packages": written,
        "validation": validation,
        "manifest_paths": tuple(manifest_paths),
    }


def _validate_manifest_bindings_against_binary(
    packages: tuple[Any, ...],
) -> dict[str, Any]:
    """Check JSON file entries against the encoded resident slot descriptors."""

    validated: list[dict[str, Any]] = []
    for item in packages:
        package = item.package
        if package.manifest_path is None or package.package_path is None:
            raise ValueError("sliced_resident_package_write_incomplete")
        manifest = json.loads(package.manifest_path.read_text(encoding="utf-8"))
        binary = validate_resident_graph_package_file(package.package_path)
        descriptors = {
            int(item["slot_id"]): item for item in binary["slot_descriptors"]
        }
        initial = manifest.get("initial_slots")
        final = manifest.get("final_outputs")
        if not isinstance(initial, list) or not isinstance(final, list):
            raise ValueError("sliced_resident_manifest_file_entries_missing")
        initial_ids = {int(entry["slot_id"]) for entry in initial}
        final_ids = {int(entry["slot_id"]) for entry in final}
        if initial_ids != set(binary["initial_slot_ids"]):
            raise ValueError("sliced_resident_manifest_initial_slots_binary_mismatch")
        if final_ids != set(binary["final_slot_ids"]):
            raise ValueError("sliced_resident_manifest_final_outputs_binary_mismatch")
        if initial_ids & final_ids:
            raise ValueError("sliced_resident_manifest_initial_final_slot_alias")
        for entry in (*initial, *final):
            slot_id = int(entry["slot_id"])
            descriptor = descriptors.get(slot_id)
            elements = int(entry["elements"])
            raw_bytes = int(entry["raw_bytes"])
            transfer_bytes = int(entry["transfer_bytes"])
            if (
                descriptor is None
                or elements != int(descriptor["element_count"])
                or raw_bytes != elements * 4
                or transfer_bytes != ((raw_bytes + 7) // 8) * 8
            ):
                raise ValueError(
                    "sliced_resident_manifest_file_entry_binary_mismatch"
                )
        validated.append(
            {
                "slice_id": int(item.slice_id),
                "package_path": str(package.package_path),
                "initial_slot_ids": sorted(initial_ids),
                "final_slot_ids": sorted(final_ids),
                "binary_metadata": binary,
            }
        )
    return {"status": "passed", "packages": validated}


def _validate_native_manifests(
    host_binary: str | Path,
    manifest_paths: tuple[Path, ...],
    *,
    timeout_s: float,
) -> dict[str, Any]:
    command = [
        str(host_binary),
        "--validate-slice-packages",
        *(str(path) for path in manifest_paths),
    ]
    result: dict[str, Any] = {
        "status": "failed",
        "command": command,
        "dpu_allocation_attempted": False,
        "dpu_launch_attempted": False,
    }
    if not Path(host_binary).is_file():
        result.update(
            {
                "status": "unavailable",
                "reason": "native host binary is unavailable",
                "failure_stage": "native_manifest_validation_unavailable",
            }
        )
        return result
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as exc:
        result.update(
            {
                "reason": "native manifest parser timed out",
                "failure_stage": "native_manifest_validation_timeout",
                "stdout": str(exc.stdout or ""),
                "stderr": str(exc.stderr or ""),
            }
        )
        return result
    stdout = completed.stdout.strip()
    stderr = completed.stderr.strip()
    parsed: Any = None
    try:
        parsed = json.loads(stdout)
    except json.JSONDecodeError:
        pass
    passed = completed.returncode == 0 and isinstance(parsed, Mapping) and parsed.get("status") == "valid"
    result.update(
        {
            "status": "passed" if passed else "failed",
            "returncode": completed.returncode,
            "stdout": stdout,
            "stderr": stderr,
            "result": parsed,
        }
    )
    if not passed:
        result.update(
            {
                "reason": (
                    parsed.get("reason")
                    if isinstance(parsed, Mapping)
                    else "native manifest parser returned no valid JSON verdict"
                ),
                "failure_stage": "native_manifest_validation_failed",
            }
        )
    return result


def _executor_config_hash(numeric_mode: str) -> str:
    return executor_config_hash(
        ROUTE_ID,
        {
            "backend_id": BACKEND_ID,
            "hardware_profile_version": "hardware_sliced_resident_two_dpu_m2_v1",
            "slices": 2,
            "requested_dpu_count": 2,
            "tasklets_per_dpu": 1,
            "operation_count": 2,
            "numeric_mode": numeric_mode,
            "resident_numeric_policy": (
                "float32_accumulate"
                if numeric_mode == "none"
                else "per_task_resident_requantize"
            ),
        },
    )


def _accuracy_metrics(
    reference: Any,
    actual: Any,
    *,
    tolerance: float,
    reference_kind: str,
) -> dict[str, Any]:
    difference = np.asarray(actual, dtype=np.complex128) - np.asarray(
        reference, dtype=np.complex128
    )
    max_abs = float(np.max(np.abs(difference))) if difference.size else 0.0
    l2 = float(np.linalg.norm(difference))
    reference_norm = float(
        np.linalg.norm(np.asarray(reference, dtype=np.complex128))
    )
    relative_l2 = l2 / reference_norm if reference_norm else l2
    passed = max_abs <= tolerance
    return {
        "status": "passed" if passed else "failed",
        "passed": passed,
        "available": True,
        "reference_kind": reference_kind,
        "max_abs_error": max_abs,
        "l2_error": l2,
        "relative_l2_error": relative_l2,
        "max_abs_tolerance": tolerance,
    }


def _record(
    m2: M2Suite,
    case: Mapping[str, Any],
    prepared: Mapping[str, Any],
    phase: str,
    repeat_id: int,
    *,
    run_dir: Path,
    status: str,
    failure_stage: str | None,
    reason: str | None,
    native: Any,
    session: Any,
    artifacts: Mapping[str, Any],
    reconstruction: Mapping[str, Any],
    output: np.ndarray,
    cpu_ok: bool,
    expected_ok: bool,
    reconstruction_time_s: float,
    total_time_s: float,
    numeric_mode: str = "none",
    policy_reference: Mapping[str, Any] | None = None,
    full_precision_accuracy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    response = session.response
    allocation = response.get("allocation", {})
    launch = response.get("launch", {})
    release = response.get("release", {})
    planned_h2d, planned_d2h = _transfer_bytes(artifacts["packages"])
    planned_transfer = planned_h2d + planned_d2h
    observed_h2d = response.get("actual_h2d_bytes")
    observed_d2h = response.get("actual_d2h_bytes")
    observed_transfer = response.get("actual_transfer_bytes")
    native_transfer_available = all(
        isinstance(value, int) and not isinstance(value, bool) and value >= 0
        for value in (observed_h2d, observed_d2h, observed_transfer)
    )
    transfer_invariant = (
        native_transfer_available
        and observed_transfer == observed_h2d + observed_d2h
    )
    transfer_matches_plan = (
        transfer_invariant
        and observed_h2d == planned_h2d
        and observed_d2h == planned_d2h
    )
    transfer_status = (
        "passed"
        if transfer_matches_plan
        else "failed"
        if m2.fixture_version in {M2_1_SCHEMA_VERSION, M2_2_SCHEMA_VERSION}
        else "legacy_m2_planned_only"
        if not native_transfer_available
        else "failed"
    )
    h2d = int(observed_h2d) if transfer_invariant else planned_h2d
    d2h = int(observed_d2h) if transfer_invariant else planned_d2h
    transfer = h2d + d2h
    partial_outputs = reconstruction.get("partial_outputs", {})
    is_m2_2 = m2.is_m2_2
    require_sentinel = m2.fixture_version in {
        M2_1_SCHEMA_VERSION,
        M2_2_SCHEMA_VERSION,
    }
    observed_counts = _observed_operation_completion_counts(
        response, require_sentinel=require_sentinel
    )
    observed_slice_count = _observed_completed_slice_count(
        response, require_sentinel=require_sentinel
    )
    observed_task_count = (
        None if observed_counts is None else sum(observed_counts)
    )
    execution_contract_status = (
        "passed"
        if response.get("hardware_execution") is True
        and response.get("backend_id") == BACKEND_ID
        and response.get("backend_family") == "upmem_sdk"
        and response.get("target_requested") == "hardware"
        and response.get("target_observed") == "hardware"
        and response.get("hardware_kernel_executed") is True
        and response.get("simulator_kernel_executed") is False
        and response.get("cpu_fallback_used") is False
        and allocation.get("verified") is True
        and launch.get("completed") is True
        and release.get("confirmed") is True
        and (
            not is_m2_2
            or (
                response.get("operation_count") == 2
                and response.get("quantization_mode") == numeric_mode
            )
        )
        and (
            m2.fixture_version not in {M2_1_SCHEMA_VERSION, M2_2_SCHEMA_VERSION}
            or (observed_counts is not None and observed_slice_count == 2)
        )
        else "failed"
    )
    package_status = (
        "passed"
        if artifacts.get("validation", {}).get("validated") is True
        else "failed"
    )
    per_slice_status = (
        reconstruction.get("per_slice_output_validation_status", "not_run")
        if m2.fixture_version in {M2_1_SCHEMA_VERSION, M2_2_SCHEMA_VERSION}
        else "not_applicable_historical_m2"
    )
    reconstruction_status = (
        "passed"
        if is_m2_2 and np.all(np.isfinite(np.asarray(output)))
        else "passed"
        if cpu_ok
        else "failed"
    )
    final_status = "passed" if expected_ok else "failed"
    useful_status = reconstruction.get("slice_useful_work", {}).get(
        "status", "not_run"
    )
    if is_m2_2:
        policy_reference = policy_reference or {
            "status": "not_run",
            "passed": False,
            "available": False,
        }
        full_precision_accuracy = full_precision_accuracy or {
            "status": "not_run",
            "passed": False,
            "available": False,
        }
        policy_status = str(policy_reference.get("status", "not_run"))
        full_precision_status = str(
            full_precision_accuracy.get("status", "not_run")
        )
        scientific_status = (
            "passed"
            if all(
                value == "passed"
                for value in (
                    execution_contract_status,
                    package_status,
                    per_slice_status,
                    reconstruction_status,
                    final_status,
                    useful_status,
                    transfer_status,
                    policy_status,
                    full_precision_status,
                )
            )
            else "failed"
        )
    else:
        policy_status = "not_run"
        full_precision_status = "not_run"
        scientific_status = (
            "passed"
            if all(
                value == "passed"
                for value in (
                    execution_contract_status,
                    package_status,
                    per_slice_status
                    if m2.fixture_version == M2_1_SCHEMA_VERSION
                    else "passed",
                    reconstruction_status,
                    final_status,
                    useful_status if m2.require_nonzero_slice_partials else "passed",
                    transfer_status
                    if m2.fixture_version == M2_1_SCHEMA_VERSION
                    else "passed",
                )
            )
            else "failed"
        )
    record_status = (
        "completed"
        if status == "completed" and scientific_status == "passed"
        else "failed"
    )
    effective_failure_stage = (
        failure_stage
        if record_status == "failed" and failure_stage is not None
        else None
        if record_status == "completed"
        else "scientific_validation_failed"
    )
    effective_reason = (
        reason
        if record_status == "failed" and reason is not None
        else None
        if record_status == "completed"
        else "scientific_validation_failed"
    )
    return {
        "schema_version": "upmem_hardware_sliced_resident_mvp_record_v1",
        "status": record_status,
        "suite_id": m2.suite["suite_id"],
        "case_id": case["case_id"],
        "workload_id": case["workload_id"],
        "phase": phase,
        "repeat_id": repeat_id,
        "route_id": ROUTE_ID,
        "backend_id": BACKEND_ID,
        "execution_scope": (
            "physical_two_dpu_two_slice_full_replicated_prefix_taskgraph"
            if prepared["fixture_version"] in {M2_1_SCHEMA_VERSION, M2_2_SCHEMA_VERSION}
            else "physical_two_dpu_two_slice_terminal_contraction"
        ),
        "parallelism_mode": "slicing",
        "parallelism_evidence_type": "executed_dispatch_only",
        "slicing_enabled": True,
        "slicing_backend": "internal_taskgraph",
        "slicing_strategy": (
            "contraction_index_restriction_with_replicated_prefix"
            if prepared["fixture_version"] in {M2_1_SCHEMA_VERSION, M2_2_SCHEMA_VERSION}
            else "contraction_index_input_restriction"
        ),
        "slice_ids": [0, 1],
        "slice_parallel_execution": False,
        "slice_parallel_wave_count": 1,
        "slice_overlap_measured": False,
        "dispatch_concurrency_status": "asynchronous_set_launch_unmeasured_overlap",
        "device_launch_mode": "asynchronous_dpu_set",
        "host_completion_mode": "blocking_sync",
        "device_completion_state": response.get("device_completion_state", "unknown"),
        "device_completion_confirmed": response.get("device_completion_confirmed"),
        "native_execution_sentinel_available": response.get(
            "native_execution_sentinel_available"
        ),
        "completion_evidence": response.get("completion_evidence"),
        "completion_sentinel_read_counts": response.get(
            "completion_sentinel_read_counts"
        ),
        "fixture_version": prepared["fixture_version"],
        "fixture_scope": prepared["fixture_scope"],
        "selected_task_id": prepared["selected_task_id"],
        "quantization_mode": numeric_mode,
        **({"numeric_mode": numeric_mode} if is_m2_2 else {}),
        "n_qubits": prepared["circuit"].n_qubits,
        "gate_count": len(prepared["circuit"].operations),
        "task_count": len(prepared["graph"].tasks),
        "contracted_dimension": prepared["graph"].tasks[-1].gemm_k,
        "source_task_count": prepared["source_task_count"],
        "source_task_completion_count": observed_task_count,
        "source_task_completion_scope": "replicated_slice_operations",
        "expanded_task_count": prepared["source_task_count"] * 2,
        "executed_task_count": observed_task_count,
        "completed_task_count": observed_task_count,
        "completed_slice_count": observed_slice_count,
        "source_slice_count": 2,
        "executed_slice_count": observed_slice_count,
        "slice_model_task_count": 2,
        "slice_model_executed_task_count": observed_task_count,
        "observed_operation_completion_counts": observed_counts,
        "slice_count": 2,
        "requested_dpu_count": 2,
        "allocated_dpu_count": allocation.get("allocated_dpus"),
        "tasklets_per_dpu": 1,
        "target_requested": response.get("target_requested"),
        "target_observed": response.get("target_observed"),
        "backend_family": response.get("backend_family"),
        "operation_count": response.get("operation_count"),
        "hardware_execution": response.get("hardware_execution"),
        "native_kernel_executed": response.get("hardware_kernel_executed"),
        "hardware_kernel_executed": response.get("hardware_kernel_executed"),
        "simulator_kernel_executed": response.get("simulator_kernel_executed"),
        "cpu_fallback_used": response.get("cpu_fallback_used"),
        "allocation_evidence": allocation,
        "launch_evidence": launch,
        "sync_evidence": {"synchronize_count": launch.get("synchronize_count")},
        "release_evidence": release,
        "circuit_semantics_hash": prepared["graph"].circuit_semantics_hash,
        "tensor_network_hash": prepared["graph"].tensor_network_hash,
        "contraction_plan_hash": prepared["graph"].contraction_plan_hash,
        **(
            {"executor_config_hash": _executor_config_hash(numeric_mode)}
            if is_m2_2
            else {}
        ),
        "qasm_source_sha256": prepared["qasm_source_sha256"],
        "source_hashes": artifacts["validation"]["source_hashes"],
        "source_hashes_preserved": True,
        "derived_slice_package_hashes": {
            str(item.slice_id): {
                "descriptor_sha256": item.package.descriptor_sha256,
            }
            for item in artifacts["packages"]
        },
        "native_source_tree_hash": native.source_tree_hash,
        "binary_source_tree_hash": native.source_tree_hash,
        "host_binary_hash": native.host_binary_hash,
        "dpu_binary_hash": native.dpu_binary_hash,
        "build_time_s": native.build_time_s,
        "process_time_s": session.process_time_s,
        "timing_scope": "host_observed_sdk_process_wall_and_blocking_sync",
        "timing_is_bringup_only": True,
        "native_clock": response.get("timing", {}).get("clock", "unknown"),
        "timing_breakdown_status": response.get("timing", {}).get(
            "status", "unavailable"
        ),
        "stage_timings": response.get("timing"),
        "reconstruction_time_s": reconstruction_time_s,
        "total_time_s": total_time_s,
        "application_visible_h2d_bytes": h2d,
        "application_visible_d2h_bytes": d2h,
        "application_visible_transfer_bytes": transfer,
        "application_visible_total_bytes": transfer,
        "actual_h2d_bytes": h2d,
        "actual_d2h_bytes": d2h,
        "actual_transfer_bytes": transfer,
        "planned_h2d_bytes": planned_h2d,
        "planned_d2h_bytes": planned_d2h,
        "planned_transfer_bytes": planned_transfer,
        "observed_h2d_bytes": observed_h2d if native_transfer_available else None,
        "observed_d2h_bytes": observed_d2h if native_transfer_available else None,
        "observed_transfer_bytes": observed_transfer if native_transfer_available else None,
        "transfer_accounting_status": transfer_status,
        "transfer_accounting_invariant": transfer_invariant,
        "transfer_matches_manifest_plan": transfer_matches_plan,
        "actual_transfer_source": "native_response" if transfer_invariant else "manifest_compatibility",
        "execution_contract_status": execution_contract_status,
        **(
            {
                "response_numeric_mode": response.get("quantization_mode"),
                "policy_reference_validation": policy_reference,
                "full_precision_accuracy": full_precision_accuracy,
                "policy_reference_status": policy_status,
                "full_precision_accuracy_status": full_precision_status,
                "policy_reference_max_abs_error": policy_reference.get(
                    "max_abs_error"
                ),
                "policy_reference_l2_error": policy_reference.get("l2_error"),
                "policy_reference_relative_l2_error": policy_reference.get(
                    "relative_l2_error"
                ),
                "full_precision_max_abs_error": full_precision_accuracy.get(
                    "max_abs_error"
                ),
                "full_precision_l2_error": full_precision_accuracy.get("l2_error"),
                "full_precision_relative_l2_error": full_precision_accuracy.get(
                    "relative_l2_error"
                ),
            }
            if is_m2_2
            else {}
        ),
        "slice_package_validation_status": package_status,
        "per_slice_output_validation_status": per_slice_status,
        "reconstruction_validation_status": reconstruction_status,
        "final_output_validation_status": final_status,
        "scientific_validation_status": scientific_status,
        "validation_status": "passed" if scientific_status == "passed" else "failed",
        "cpu_reference_validation": (
            cpu_ok
            if not is_m2_2 or numeric_mode == "none"
            else policy_status == "passed"
        ),
        "strict_cpu_reference_validation": cpu_ok,
        "cpu_reference_validation_kind": (
            "strict_float32_full_precision"
            if not is_m2_2 or numeric_mode == "none"
            else "dpu_policy_reference_for_requantized_mode"
        ),
        "expected_output_validation": expected_ok,
        "validation_errors": []
        if record_status == "completed"
        else [effective_reason or "output_validation_failed"],
        "output_hash": _array_hash(output),
        "reconstruction": reconstruction,
        "per_slice_useful_work": partial_outputs,
        "hardware_functionality_evidence": record_status == "completed",
        "hardware_speedup_applicable": False,
        "energy_measurement_available": False,
        "claim_boundary": _claim_boundary(m2),
        "speedup_claim": "not_applicable",
        "energy_claim": "not_applicable",
        "performance_claim_applicable": False,
        "failure_stage": effective_failure_stage,
        "reason": effective_reason,
        **_operation_evidence(run_dir=run_dir, session=session, artifacts=artifacts),
    }


def _failure_record(
    m2: M2Suite,
    reason: str,
    case: Mapping[str, Any] | None,
    phase: str,
    repeat_id: int | None,
    failure_stage: str,
    *,
    prepared: Mapping[str, Any] | None = None,
    total_time_s: float | None = None,
    operation_evidence: Mapping[str, Any] | None = None,
    observed_response: Mapping[str, Any] | None = None,
    numeric_mode: str = "none",
) -> dict[str, Any]:
    observed = observed_response or {}
    record = {
        "schema_version": "upmem_hardware_sliced_resident_mvp_record_v1",
        "status": "failed",
        "suite_id": m2.suite["suite_id"],
        "case_id": case.get("case_id") if case else None,
        "workload_id": case.get("workload_id") if case else None,
        "phase": phase,
        "repeat_id": repeat_id,
        "route_id": ROUTE_ID,
        "backend_id": BACKEND_ID,
        "backend_family": observed.get("backend_family"),
        "target_requested": "hardware",
        "target_observed": observed.get("target_observed", "not_observed"),
        "hardware_execution": observed.get("hardware_execution", False),
        "native_kernel_executed": observed.get("hardware_kernel_executed", False),
        "hardware_kernel_executed": observed.get("hardware_kernel_executed", False),
        "simulator_kernel_executed": observed.get("simulator_kernel_executed", False),
        "cpu_fallback_used": observed.get("cpu_fallback_used", False),
        "kernel_execution_status": observed.get(
            "kernel_execution_status", "not_observed"
        ),
        "quantization_mode": numeric_mode,
        **({"numeric_mode": numeric_mode} if m2.is_m2_2 else {}),
        "parallelism_mode": "slicing",
        "parallelism_evidence_type": "executed_dispatch_only",
        "slicing_enabled": True,
        "slicing_backend": "internal_taskgraph",
        "slice_parallel_execution": False,
        "slice_parallel_wave_count": 1,
        "slice_overlap_measured": False,
        "dispatch_concurrency_status": "not_run",
        "device_launch_mode": "asynchronous_dpu_set",
        "host_completion_mode": "blocking_sync",
        "device_completion_state": "not_run",
        "device_completion_confirmed": False,
        "native_execution_sentinel_available": None,
        "completion_evidence": None,
        "completion_sentinel_read_counts": None,
        "slice_count": 2,
        "source_task_count": None,
        "source_task_completion_count": None,
        "expanded_task_count": None,
        "executed_task_count": None,
        "completed_task_count": None,
        "completed_slice_count": 0,
        "slice_model_task_count": None,
        "slice_model_executed_task_count": None,
        "requested_dpu_count": 2,
        "tasklets_per_dpu": 1,
        "failure_stage": failure_stage,
        "reason": reason,
        "validation_status": "not_run",
        "execution_contract_status": "not_run",
        "slice_package_validation_status": "not_run",
        "per_slice_output_validation_status": "not_run",
        "reconstruction_validation_status": "not_run",
        "final_output_validation_status": "not_run",
        "scientific_validation_status": "not_run",
        **(
            {
                "response_numeric_mode": observed.get("quantization_mode"),
                "policy_reference_validation": None,
                "full_precision_accuracy": None,
                "policy_reference_status": "not_run",
                "full_precision_accuracy_status": "not_run",
                "strict_cpu_reference_validation": False,
                "cpu_reference_validation_kind": (
                    "strict_float32_full_precision"
                    if numeric_mode == "none"
                    else "dpu_policy_reference_for_requantized_mode"
                ),
                "executor_config_hash": _executor_config_hash(numeric_mode),
            }
            if m2.is_m2_2
            else {}
        ),
        "validation_errors": [reason],
        "cpu_reference_validation": False,
        "expected_output_validation": False,
        "allocated_dpu_count": 0,
        "allocation_evidence": None,
        "launch_evidence": None,
        "sync_evidence": None,
        "release_evidence": None,
        "build_time_s": None,
        "process_time_s": None,
        "reconstruction_time_s": None,
        "application_visible_h2d_bytes": 0,
        "application_visible_d2h_bytes": 0,
        "application_visible_transfer_bytes": 0,
        "application_visible_total_bytes": 0,
        "actual_h2d_bytes": 0,
        "actual_d2h_bytes": 0,
        "actual_transfer_bytes": 0,
        "planned_h2d_bytes": None,
        "planned_d2h_bytes": None,
        "planned_transfer_bytes": None,
        "observed_h2d_bytes": None,
        "observed_d2h_bytes": None,
        "observed_transfer_bytes": None,
        "transfer_accounting_status": "not_run",
        "transfer_accounting_invariant": False,
        "transfer_matches_manifest_plan": False,
        "actual_transfer_source": "not_run",
        "total_time_s": total_time_s,
        "claim_boundary": _claim_boundary(m2),
        "speedup_claim": "not_applicable",
        "energy_claim": "not_applicable",
        "performance_claim_applicable": False,
        "hardware_functionality_evidence": False,
        "hardware_speedup_applicable": False,
        "timing_is_bringup_only": True,
        **(
            dict(operation_evidence)
            if operation_evidence
            else _operation_evidence(None, None, None)
        ),
    }
    if prepared:
        record.update(
            {
                "n_qubits": prepared["circuit"].n_qubits,
                "gate_count": len(prepared["circuit"].operations),
                "task_count": len(prepared["graph"].tasks),
                "contracted_dimension": prepared["graph"].tasks[-1].gemm_k,
                "source_task_count": prepared["source_task_count"],
                "tensor_count": prepared["tensor_count"],
                "selected_task_id": prepared["selected_task_id"],
                "fixture_version": prepared["fixture_version"],
                "fixture_scope": prepared["fixture_scope"],
                "circuit_semantics_hash": prepared["graph"].circuit_semantics_hash,
                "tensor_network_hash": prepared["graph"].tensor_network_hash,
                "contraction_plan_hash": prepared["graph"].contraction_plan_hash,
                "qasm_source_sha256": prepared["qasm_source_sha256"],
            }
        )
    return record


def _finish_run(
    run_dir: Path,
    manifest: dict[str, Any],
    m2: M2Suite,
    records: list[dict[str, Any]],
    warmups: list[dict[str, Any]],
    native: Any | None,
) -> M2RunResult:
    write_jsonl(run_dir / "warmups.jsonl", warmups)
    write_normalized_records(run_dir, records)
    expected_measured_rows = (
        len(m2.suite["cases"])
        * int(m2.suite["repeats"])
        * (len(m2.numeric_modes) if m2.is_m2_2 else 1)
    )
    expected_warmup_rows = (
        len(m2.suite["cases"])
        * int(m2.suite["warmups"])
        * (len(m2.numeric_modes) if m2.is_m2_2 else 1)
    )

    def admitted(row: Mapping[str, Any]) -> bool:
        return (
            row.get("status") == "completed"
            and row.get("validation_status") == "passed"
        )

    measured_passed_count = sum(admitted(row) for row in records)
    warmup_passed_count = sum(admitted(row) for row in warmups)
    measured_failed_count = len(records) - measured_passed_count
    warmup_failed_count = len(warmups) - warmup_passed_count
    measured_complete = (
        len(records) == expected_measured_rows
        and measured_passed_count == expected_measured_rows
    )
    warmup_complete = (
        len(warmups) == expected_warmup_rows
        and warmup_passed_count == expected_warmup_rows
    )
    completed = measured_complete and warmup_complete
    summary_path = run_dir / "upmem_hardware_sliced_resident_mvp_summary.json"
    write_json(
        summary_path,
        {
            "schema_version": RUNTIME_SCHEMA_VERSION,
            "status": "completed" if completed else "failed",
            "suite_id": m2.suite["suite_id"],
            "route_id": ROUTE_ID,
            "backend_id": BACKEND_ID,
            "measured_row_count": len(records),
            "warmup_count": len(warmups),
            "warmup_passed_count": warmup_passed_count,
            "warmup_failed_count": warmup_failed_count,
            "warmup_status": "passed" if warmup_complete else "failed",
            "measured_passed_count": measured_passed_count,
            "measured_failed_count": measured_failed_count,
            "measured_status": "passed" if measured_complete else "failed",
            "all_required_records_validated": completed,
            "expected_measured_row_count": expected_measured_rows,
            "expected_warmup_row_count": expected_warmup_rows,
            **({"numeric_modes": list(m2.numeric_modes)} if m2.is_m2_2 else {}),
            "normalized_records": "normalized_records.jsonl",
            "warmups": "warmups.jsonl",
            "native_build": _build_metadata(native, run_dir)
            if native
            else {"attempted": False, "status": "not_available"},
            "claim_boundary": _claim_boundary(m2),
        },
    )
    manifest.update(
        {
            "summary": summary_path.name,
            "hardware_available": "verified_by_execution"
            if completed
            else "not_verified_by_execution",
        }
    )
    write_json(run_dir / "run_manifest.json", manifest)
    return M2RunResult(
        run_dir, summary_path, "completed" if completed else "failed", len(records)
    )


def _write_common_artifacts(directory: Path, root_dir: Path, m2: M2Suite) -> None:
    (directory / "config").mkdir(parents=True, exist_ok=True)
    shutil.copy2(m2.path, directory / "config" / "resolved_suite.yml")
    shutil.copy2(m2.path, directory / "resolved_suite.yml")
    write_json(
        directory / "config" / "hardware_profile.json",
        _profile_metadata(
            m2.profile,
            numeric_modes=m2.numeric_modes if m2.is_m2_2 else None,
        ),
    )
    write_json(directory / "environment.json", capture_environment(root_dir))


def _phase_ids(suite: Mapping[str, Any]) -> tuple[tuple[str, int], ...]:
    return tuple(("warmup", item) for item in range(int(suite["warmups"]))) + tuple(
        ("measured", item) for item in range(int(suite["repeats"]))
    )


def _artifact_dir(
    root: Path,
    case_id: str,
    phase: str,
    repeat_id: int,
    *,
    numeric_mode: str | None = None,
) -> Path:
    base = root / "cases" / sanitize(case_id)
    if numeric_mode is not None:
        base = base / sanitize(numeric_mode)
    return base / f"{phase}_{repeat_id:02d}"


def _response_path(
    native: Any,
    case_id: str,
    phase: str,
    repeat_id: int,
    *,
    m2_2: bool,
    numeric_mode: str,
) -> Path:
    mode_suffix = f"-{sanitize(numeric_mode)}" if m2_2 else "-default"
    return native.session_root / (
        f"{sanitize(case_id)}-{phase}-{repeat_id:02d}{mode_suffix}-response.json"
    )


def _plan_row(
    case: Mapping[str, Any],
    prepared: Mapping[str, Any],
    phase: str,
    repeat_id: int,
    artifacts: Mapping[str, Any],
    *,
    numeric_mode: str = "none",
) -> dict[str, Any]:
    row = {
        "case_id": case["case_id"],
        "phase": phase,
        "repeat_id": repeat_id,
        "status": "prepared",
        "n_qubits": prepared["circuit"].n_qubits,
        "gate_count": len(prepared["circuit"].operations),
        "task_count": len(prepared["graph"].tasks),
        "contracted_dimension": prepared["graph"].tasks[-1].gemm_k,
        "source_task_count": prepared["source_task_count"],
        "tensor_count": prepared["tensor_count"],
        "selected_task_id": prepared["selected_task_id"],
        "slice_count": 2,
        "source_hashes": artifacts["validation"]["source_hashes"],
        "native_manifest_validation": artifacts.get(
            "native_manifest_validation",
            {"status": "not_run", "reason": "native validation not requested"},
        ),
    }
    if prepared.get("fixture_version") == M2_2_SCHEMA_VERSION:
        row.update(
            {
                "numeric_mode": numeric_mode,
                "quantization_mode": numeric_mode,
                "executor_config_hash": _executor_config_hash(numeric_mode),
                "circuit_semantics_hash": prepared["graph"].circuit_semantics_hash,
                "tensor_network_hash": prepared["graph"].tensor_network_hash,
                "contraction_plan_hash": prepared["graph"].contraction_plan_hash,
            }
        )
    return row


def _transfer_bytes(packages: Any) -> tuple[int, int]:
    h2d = d2h = 0
    for item in packages:
        payload = json.loads(item.package.manifest_path.read_text(encoding="utf-8"))
        h2d += sum(
            int(payload.get(key, 0))
            for key in (
                "initial_h2d_bytes",
                "descriptor_h2d_bytes",
                "control_h2d_bytes",
            )
        )
        d2h += int(payload.get("final_d2h_bytes", 0))
    transfer = h2d + d2h
    assert transfer == h2d + d2h
    return h2d, d2h


def _observed_operation_completion_counts(
    response: Mapping[str, Any],
    *,
    require_sentinel: bool = False,
) -> tuple[int, ...] | None:
    """Read completion counts reported by the native response.

    The one-operation M2 response predates explicit count fields.  Its
    per-slice completion marker is retained as a compatibility observation;
    M2.1 responses must provide the explicit native counts.
    """

    values = response.get("observed_operation_completion_counts")
    if require_sentinel and response.get("native_execution_sentinel_available") is not True:
        return None
    if require_sentinel and response.get("completion_evidence") != (
        "dpu_written_completion_sentinel_read_after_each_sync"
    ):
        return None
    if isinstance(values, list) and len(values) == 2:
        if all(
            isinstance(value, int) and not isinstance(value, bool) and value >= 0
            for value in values
        ):
            if require_sentinel and response.get("completion_sentinel_read_counts") != list(values):
                return None
            return tuple(int(value) for value in values)
    slices = response.get("slices")
    if not isinstance(slices, list) or len(slices) != 2:
        return None
    fallback: list[int] = []
    for entry in slices:
        if not isinstance(entry, Mapping):
            return None
        if require_sentinel and not (
            isinstance(entry.get("dpu_completion_sentinel"), Mapping)
            and entry["dpu_completion_sentinel"].get("verified") is True
        ):
            return None
        value = entry.get("observed_operation_completion_count")
        if value is None:
            value = entry.get("completed_operation_count")
        if value is None and entry.get("completion_confirmed") is True:
            value = entry.get("operation_count", 1)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            return None
        fallback.append(int(value))
    return tuple(fallback)


def _observed_completed_slice_count(
    response: Mapping[str, Any], *, require_sentinel: bool = False
) -> int | None:
    slices = response.get("slices")
    if not isinstance(slices, list) or len(slices) != 2:
        return None
    completed = 0
    for entry in slices:
        if not isinstance(entry, Mapping):
            return None
        if require_sentinel and not (
            isinstance(entry.get("dpu_completion_sentinel"), Mapping)
            and entry["dpu_completion_sentinel"].get("verified") is True
        ):
            return None
        if entry.get("completion_confirmed") is True:
            completed += 1
            continue
        value = entry.get("observed_operation_completion_count")
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            completed += 1
    return completed


def _qasm_path(root_dir: Path, case: Mapping[str, Any]) -> Path:
    path = Path(str(case["circuit"]["path"]))
    return path if path.is_absolute() else (root_dir / path).resolve()


def _suite_root(m2: M2Suite) -> Path:
    """Use the implementation root, never one inferred from a supplied YAML."""

    del m2
    return IMPLEMENTATION_ROOT


def _operation_evidence(
    run_dir: Path | None, session: Any | None, artifacts: Mapping[str, Any] | None
) -> dict[str, Any]:
    def relative(path: Path | None) -> str | None:
        if path is None:
            return None
        if run_dir is not None:
            try:
                return str(path.resolve().relative_to(run_dir.resolve()))
            except ValueError:
                pass
        return str(path)

    manifest_paths = () if not artifacts else artifacts.get("manifest_paths", ())
    return {
        "package_manifest_artifacts": [relative(path) for path in manifest_paths],
        "native_response_artifact": relative(session.response_path)
        if session
        else None,
        "native_session_command": list(session.command) if session else None,
        "native_stdout_snippet": session.stdout_snippet if session else None,
        "native_stderr_snippet": session.stderr_snippet if session else None,
        "native_failure_stage": session.failure_stage if session else None,
        "native_timed_out": session.timed_out if session else None,
        "native_cleanup_confirmed": session.cleanup_confirmed if session else None,
    }


def _profile_metadata(
    profile: SlicedResidentHardwareProfile,
    *,
    numeric_modes: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    result = {
        "hardware_profile_version": profile.version,
        "target": profile.target,
        "backend_id": profile.backend_id,
        "route_id": profile.route_id,
        "requested_dpu_count": profile.requested_dpu_count,
        "slices": profile.slices,
        "tasklets_per_dpu": profile.tasklets_per_dpu,
        "numeric_mode": None if numeric_modes is not None else profile.numeric_mode,
        "synchronous_execution": profile.synchronous_execution,
        "device_launch_mode": profile.device_launch_mode,
        "host_completion_mode": profile.host_completion_mode,
        "timeout_s": profile.timeout_s,
        "performance_claim_applicable": profile.performance_claim_applicable,
    }
    if numeric_modes is not None:
        result["numeric_modes"] = list(numeric_modes)
    return result


def _build_metadata(build: Any, root: Path) -> dict[str, Any]:
    return {
        "attempted": True,
        "status": "passed",
        "source_tree_hash": build.source_tree_hash,
        "host_binary_hash": build.host_binary_hash,
        "host_binary_path": str(build.host_binary),
        "dpu_binary_hash": build.dpu_binary_hash,
        "build_time_s": build.build_time_s,
        "build_command": list(build.build_command),
        "sdk_tools": build.sdk_tools,
        "session_root": str(build.session_root.relative_to(root))
        if build.session_root.is_relative_to(root)
        else str(build.session_root),
    }


def _stage(reason: str, default: str) -> str:
    for stage in (
        "hardware_opt_in_missing",
        "hardware_profile_violation",
        "sdk_discovery_failed",
        "native_build_timeout",
        "native_build_failed",
        "sliced_resident",
        "hardware_allocation_failed",
        "kernel_launch_failed",
        "kernel_timeout",
        "hardware_session_timeout",
        "output_validation_failed",
    ):
        if stage in reason:
            return stage
    return default


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _array_hash(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    return hashlib.sha256(
        str(array.dtype).encode("ascii")
        + repr(tuple(array.shape)).encode("ascii")
        + array.tobytes()
    ).hexdigest()


def _unique_dir(parent: Path) -> Path:
    parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    candidate = parent / stamp
    suffix = 1
    while candidate.exists():
        candidate = parent / f"{stamp}_{suffix:02d}"
        suffix += 1
    return candidate
