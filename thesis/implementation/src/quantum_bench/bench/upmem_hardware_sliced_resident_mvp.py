"""Internal research M2 runner for the fixed two-DPU sliced-resident MVP.

This is intentionally a narrow evidence command.  It does not provide a
general scheduler, a simulator route, retries, or performance claims.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
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
    validate_written_two_slice_packages,
    write_two_slice_resident_graph_packages,
)
from quantum_bench.tn import (
    build_tensor_network,
    execute_task_sequence_np_einsum,
    plan_task_graph_with_config,
    with_execution_identity,
)


MVP_SCHEMA_VERSION = "upmem_hardware_sliced_resident_m2_v1"
PLAN_SCHEMA_VERSION = "upmem_hardware_sliced_resident_mvp_plan_v1"
RUNTIME_SCHEMA_VERSION = "upmem_hardware_sliced_resident_mvp_runtime_v1"
ROUTE_LABEL = "upmem_hw_sliced_resident"
CLAIM_BOUNDARY = "internal/research MVP only; no speedup claim and no energy claim"
_WORKLOAD_IDS = ("one_qubit_x_m2", "one_qubit_h_m2", "one_qubit_z_m2")
IMPLEMENTATION_ROOT = Path(__file__).resolve().parents[3]
CANONICAL_M2_SUITE_PATH = (
    IMPLEMENTATION_ROOT
    / "configs"
    / "suites"
    / "upmem_hardware_sliced_resident_mvp.yml"
)
_CANONICAL_QASM_PATHS = {
    "one_qubit_x_m2": "configs/circuits/upmem_m2/one_qubit_x.qasm",
    "one_qubit_h_m2": "configs/circuits/upmem_m2/one_qubit_h.qasm",
    "one_qubit_z_m2": "configs/circuits/upmem_m2/one_qubit_z.qasm",
}


@dataclass(frozen=True)
class M2Suite:
    path: Path
    suite: dict[str, Any]
    raw: dict[str, Any]
    profile: SlicedResidentHardwareProfile


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


def load_m2_suite(path: Path) -> M2Suite:
    """Load only the committed M2 fixture; accepting a broader suite is unsafe."""

    if path.resolve() != CANONICAL_M2_SUITE_PATH.resolve():
        raise ValueError(
            "hardware_profile_violation: M2 suite must be the canonical "
            "configs/suites/upmem_hardware_sliced_resident_mvp.yml fixture"
        )
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("hardware_profile_violation: M2 suite must be a mapping")
    suite = load_suite(path)
    metadata = raw.get("metadata")
    routes = raw.get("routes")
    workloads = raw.get("workloads")
    if (
        raw.get("schema_version") != 2
        or raw.get("suite_id") != "upmem_hardware_sliced_resident_mvp"
        or not isinstance(metadata, dict)
        or metadata.get("hardware_sliced_resident_m2_schema_version")
        != MVP_SCHEMA_VERSION
        or metadata.get("quantization_mode") != "none"
        or raw.get("defaults", {}).get("warmups") != 1
        or raw.get("defaults", {}).get("repeats") != 3
        or raw.get("defaults", {}).get("planner")
        != {"engine": "opt_einsum", "optimize": "greedy"}
        or not isinstance(routes, list)
        or len(routes) != 1
        or not isinstance(workloads, list)
        or len(workloads) != 3
        or tuple(item.get("id") for item in workloads if isinstance(item, dict))
        != _WORKLOAD_IDS
    ):
        raise ValueError(
            "hardware_profile_violation: suite is not the committed M2 MVP schema"
        )
    route = routes[0]
    options = route.get("options") if isinstance(route, dict) else None
    if (
        not isinstance(options, dict)
        or route.get("id") != ROUTE_ID
        or options
        != {
            "backend_id": BACKEND_ID,
            "quantization_mode": "none",
            "slices": 2,
            "requested_dpu_count": 2,
            "tasklets_per_dpu": 1,
        }
    ):
        raise ValueError(
            "hardware_profile_violation: M2 route differs from committed route"
        )
    profile = parse_sliced_resident_hardware_profile(
        metadata.get("hardware_profile", {})
    )
    for workload in workloads:
        circuit = workload.get("circuit") if isinstance(workload, dict) else None
        workload_id = workload.get("id") if isinstance(workload, dict) else None
        if (
            not isinstance(circuit, dict)
            or circuit.get("kind") != "qasm_file"
            or circuit.get("path") != _CANONICAL_QASM_PATHS.get(workload_id)
            or circuit.get("name") != str(workload_id).removesuffix("_m2")
            or not isinstance(workload.get("expected_output"), list)
        ):
            raise ValueError(
                "hardware_profile_violation: M2 workloads must use their canonical QASM paths and expected output"
            )
    return M2Suite(path.resolve(), suite, raw, profile)


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
    rows: list[dict[str, Any]] = []
    if native.get("status") != "failed":
        for case in m2.suite["cases"]:
            try:
                prepared = _prepare_case(_suite_root(m2), case, m2)
                for phase, repeat_id in _phase_ids(m2.suite):
                    artifact_dir = _artifact_dir(
                        plan_dir, str(case["case_id"]), phase, repeat_id
                    )
                    artifacts = _write_packages(
                        prepared, m2, dpu_binary, artifact_dir, prefix="plan"
                    )
                    rows.append(_plan_row(case, prepared, phase, repeat_id, artifacts))
            except Exception as exc:
                rows.append(
                    {
                        "case_id": case.get("case_id"),
                        "status": "failed",
                        "failure_stage": _stage(str(exc), "hardware_profile_violation"),
                        "reason": str(exc),
                    }
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
            "profile": _profile_metadata(m2.profile),
            "prepared_operations": rows,
            "native_build": native,
            "dpu_allocation_attempted": False,
            "dpu_launch_attempted": False,
            "claim_boundary": CLAIM_BOUNDARY,
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
        execution_scope="physical_two_dpu_two_slice_terminal_contraction",
        evidence_type="physical_hardware_internal_research_mvp",
        upmem_execution_mode="two_dpu_sliced_resident",
        quantization_mode="none",
        artifact_retention="full",
        summary="upmem_hardware_sliced_resident_mvp_summary.json",
        root_dir=root_dir,
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
        for phase, repeat_id in _phase_ids(m2.suite):
            record = _run_operation(
                run_dir, native, m2, case, prepared, phase, repeat_id, env
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
    if circuit.n_qubits != 1 or len(circuit.operations) != 1 or len(graph.tasks) != 1:
        raise ValueError(
            "hardware_profile_violation: M2 requires a single-gate one-qubit terminal TaskGraph"
        )
    task = graph.tasks[0]
    if task.dependencies or task.gemm_k != 2 or not task.contracted_labels:
        raise ValueError(
            "hardware_profile_violation: M2 requires one terminal dimension-2 contraction"
        )
    reference, _ = execute_task_sequence_np_einsum(graph, network)
    expected = np.asarray(case["expected_output"], dtype=np.complex128)
    if not np.allclose(reference, expected, atol=1.0e-12, rtol=1.0e-12):
        raise ValueError(
            "hardware_profile_violation: CPU TaskGraph reference differs from expected output"
        )
    plan = build_two_slice_resident_plan(graph, network)
    source_path = _qasm_path(root_dir, case)
    return {
        "case_id": str(case["case_id"]),
        "circuit": circuit,
        "network": network,
        "graph": graph,
        "plan": plan,
        "reference": np.asarray(reference),
        "expected": expected,
        "qasm_source_sha256": _file_hash(source_path),
        "source_path": str(source_path),
    }


def _run_operation(
    run_dir: Path,
    native: Any,
    m2: M2Suite,
    case: Mapping[str, Any],
    prepared: Mapping[str, Any],
    phase: str,
    repeat_id: int,
    environment: Mapping[str, str],
) -> dict[str, Any]:
    started = time.perf_counter()
    artifact_dir = _artifact_dir(run_dir, str(case["case_id"]), phase, repeat_id)
    artifacts: dict[str, Any] | None = None
    session: Any | None = None
    try:
        artifacts = _write_packages(
            prepared, m2, native.dpu_binary, artifact_dir, prefix="execute"
        )
        response_path = (
            native.session_root
            / f"{sanitize(str(case['case_id']))}-{phase}-{repeat_id:02d}-response.json"
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
        reconstruction_started = time.perf_counter()
        output, reconstruction = load_and_reconstruct_two_slice_native_outputs(
            prepared["plan"], artifacts["packages"], session.response_path
        )
        reconstruction_time = time.perf_counter() - reconstruction_started
        np.save(artifact_dir / "reconstructed_output.npy", output)
        cpu_ok = bool(
            np.allclose(output, prepared["reference"], atol=1.0e-6, rtol=1.0e-6)
        )
        expected_ok = bool(
            np.allclose(output, prepared["expected"], atol=1.0e-6, rtol=1.0e-6)
        )
        status = "completed" if cpu_ok and expected_ok else "failed"
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
            operation_evidence=_operation_evidence(run_dir, session, artifacts),
        )


def _write_packages(
    prepared: Mapping[str, Any],
    m2: M2Suite,
    dpu_binary: Path,
    artifact_dir: Path,
    *,
    prefix: str,
) -> dict[str, Any]:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    packages = build_two_slice_resident_graph_packages(
        prepared["plan"],
        case_id=prepared["case_id"],
        suite_id=m2.suite["suite_id"],
        quantization_mode="none",
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
    validation = validate_written_two_slice_packages(prepared["plan"], written)
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
) -> dict[str, Any]:
    response = session.response
    allocation = response.get("allocation", {})
    launch = response.get("launch", {})
    release = response.get("release", {})
    h2d, d2h = _transfer_bytes(artifacts["packages"])
    transfer = h2d + d2h
    return {
        "schema_version": "upmem_hardware_sliced_resident_mvp_record_v1",
        "status": status,
        "suite_id": m2.suite["suite_id"],
        "case_id": case["case_id"],
        "workload_id": case["workload_id"],
        "phase": phase,
        "repeat_id": repeat_id,
        "route_id": ROUTE_ID,
        "backend_id": BACKEND_ID,
        "execution_scope": "physical_two_dpu_two_slice_terminal_contraction",
        "quantization_mode": "none",
        "n_qubits": prepared["circuit"].n_qubits,
        "gate_count": len(prepared["circuit"].operations),
        "task_count": len(prepared["graph"].tasks),
        "contracted_dimension": prepared["graph"].tasks[0].gemm_k,
        "slice_count": 2,
        "requested_dpu_count": 2,
        "allocated_dpu_count": allocation.get("allocated_dpus"),
        "tasklets_per_dpu": 1,
        "hardware_execution": response.get("hardware_execution"),
        "cpu_fallback_used": response.get("cpu_fallback_used"),
        "allocation_evidence": allocation,
        "launch_evidence": launch,
        "sync_evidence": {"synchronize_count": launch.get("synchronize_count")},
        "release_evidence": release,
        "circuit_semantics_hash": prepared["graph"].circuit_semantics_hash,
        "tensor_network_hash": prepared["graph"].tensor_network_hash,
        "contraction_plan_hash": prepared["graph"].contraction_plan_hash,
        "qasm_source_sha256": prepared["qasm_source_sha256"],
        "source_hashes": artifacts["validation"]["source_hashes"],
        "native_source_tree_hash": native.source_tree_hash,
        "binary_source_tree_hash": native.source_tree_hash,
        "host_binary_hash": native.host_binary_hash,
        "dpu_binary_hash": native.dpu_binary_hash,
        "build_time_s": native.build_time_s,
        "process_time_s": session.process_time_s,
        "reconstruction_time_s": reconstruction_time_s,
        "total_time_s": total_time_s,
        "application_visible_h2d_bytes": h2d,
        "application_visible_d2h_bytes": d2h,
        "application_visible_transfer_bytes": transfer,
        "application_visible_total_bytes": transfer,
        "actual_h2d_bytes": h2d,
        "actual_d2h_bytes": d2h,
        "actual_transfer_bytes": transfer,
        "validation_status": "passed" if status == "completed" else "failed",
        "cpu_reference_validation": cpu_ok,
        "expected_output_validation": expected_ok,
        "validation_errors": []
        if status == "completed"
        else ["output_validation_failed"],
        "failure_stage": failure_stage,
        "reason": reason,
        "output_hash": _array_hash(output),
        "reconstruction": reconstruction,
        "claim_boundary": CLAIM_BOUNDARY,
        "speedup_claim": "not_applicable",
        "energy_claim": "not_applicable",
        "performance_claim_applicable": False,
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
) -> dict[str, Any]:
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
        "quantization_mode": "none",
        "slice_count": 2,
        "requested_dpu_count": 2,
        "tasklets_per_dpu": 1,
        "failure_stage": failure_stage,
        "reason": reason,
        "validation_status": "not_run",
        "validation_errors": [reason],
        "cpu_reference_validation": False,
        "expected_output_validation": False,
        "cpu_fallback_used": False,
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
        "total_time_s": total_time_s,
        "claim_boundary": CLAIM_BOUNDARY,
        "speedup_claim": "not_applicable",
        "energy_claim": "not_applicable",
        "performance_claim_applicable": False,
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
                "contracted_dimension": prepared["graph"].tasks[0].gemm_k,
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
    completed = len(records) == 9 and all(
        row.get("status") == "completed" for row in records
    )
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
            "expected_measured_row_count": 9,
            "normalized_records": "normalized_records.jsonl",
            "warmups": "warmups.jsonl",
            "native_build": _build_metadata(native, run_dir)
            if native
            else {"attempted": False, "status": "not_available"},
            "claim_boundary": CLAIM_BOUNDARY,
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
        directory / "config" / "hardware_profile.json", _profile_metadata(m2.profile)
    )
    write_json(directory / "environment.json", capture_environment(root_dir))


def _phase_ids(suite: Mapping[str, Any]) -> tuple[tuple[str, int], ...]:
    return tuple(("warmup", item) for item in range(int(suite["warmups"]))) + tuple(
        ("measured", item) for item in range(int(suite["repeats"]))
    )


def _artifact_dir(root: Path, case_id: str, phase: str, repeat_id: int) -> Path:
    return root / "cases" / sanitize(case_id) / f"{phase}_{repeat_id:02d}"


def _plan_row(
    case: Mapping[str, Any],
    prepared: Mapping[str, Any],
    phase: str,
    repeat_id: int,
    artifacts: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "case_id": case["case_id"],
        "phase": phase,
        "repeat_id": repeat_id,
        "status": "prepared",
        "n_qubits": prepared["circuit"].n_qubits,
        "gate_count": len(prepared["circuit"].operations),
        "task_count": len(prepared["graph"].tasks),
        "contracted_dimension": prepared["graph"].tasks[0].gemm_k,
        "slice_count": 2,
        "source_hashes": artifacts["validation"]["source_hashes"],
    }


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


def _profile_metadata(profile: SlicedResidentHardwareProfile) -> dict[str, Any]:
    return {
        "hardware_profile_version": profile.version,
        "target": profile.target,
        "backend_id": profile.backend_id,
        "route_id": profile.route_id,
        "requested_dpu_count": profile.requested_dpu_count,
        "slices": profile.slices,
        "tasklets_per_dpu": profile.tasklets_per_dpu,
        "numeric_mode": profile.numeric_mode,
        "synchronous_execution": profile.synchronous_execution,
        "timeout_s": profile.timeout_s,
        "performance_claim_applicable": profile.performance_claim_applicable,
    }


def _build_metadata(build: Any, root: Path) -> dict[str, Any]:
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
