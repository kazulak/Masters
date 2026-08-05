"""M4.1 differential study: raw SDK versus SimplePIM management control."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime
import json
import hashlib
import os
from pathlib import Path
import subprocess
import shutil
import time
from typing import Any, Mapping

import yaml
import numpy as np

from quantum_bench.bench import upmem_hardware_frontier_m3_1 as m31
from quantum_bench.bench.reporting import artifact_ref, write_normalized_records, write_run_manifest
from quantum_bench.bench.run_dirs import EVIDENCE_ARTIFACT_KIND, create_run_dir, sanitize
from quantum_bench.core.jsonio import write_json, write_jsonl
from quantum_bench.core.records import JsonDict
from quantum_bench.environment import capture_environment
from quantum_bench.tn import executor_config_hash
from quantum_bench.targets.upmem.hardware_frontier_session import (
    execute_hardware_frontier_session,
    parse_hardware_frontier_profile,
    _require_physical_opt_in,
)
from quantum_bench.targets.upmem.simplepim_frontier_session import (
    M41_BACKEND_ID,
    M41_PROFILE_ID,
    M41_ROUTE_ID,
    SIMPLEPIM_PROVIDER_ID,
    SimplePimFrontierBuild,
    SimplePimFrontierProfile,
    build_simplepim_frontier_session,
    execute_simplepim_frontier_session,
    parse_simplepim_frontier_profile,
    simplepim_build_metadata,
)


SUITE_SCHEMA_VERSION = "upmem_hardware_taskgraph_m4_1_v1"
PLAN_SCHEMA_VERSION = "upmem_hardware_taskgraph_m4_1_plan_v1"
RUN_SCHEMA_VERSION = "upmem_hardware_taskgraph_m4_1_run_v1"
PROVIDER_ID = "upmem_taskgraph_m4_1_differential"
RAW_PROVIDER_ID = "raw_sdk"
KERNEL_PROVIDER_ID = "thesis_resident_generic_contract"
CPU_REFERENCE_ATOL = 1.0e-6
CPU_REFERENCE_VALIDATION_CONTRACT = "m3_1_same_plan_abs_atol_1e-6_rtol_0"
RAW_PROFILE_ID = str(m31.EXPECTED_PROFILE["hardware_profile_version"])
CLAIM_BOUNDARY = (
    "simplepim_management_control_plane_functionality_and_host_observed_bringup_timing_only; "
    "no speedup, scaling, concurrency, or energy claim"
)
CANONICAL_SUITE_PATH = (
    Path(__file__).resolve().parents[3]
    / "configs"
    / "suites"
    / "upmem_hardware_taskgraph_m4_1.yml"
)
ROUTE_VARIANTS = {
    RAW_PROVIDER_ID: {
        "route_id": m31.ROUTE_ID,
        "backend_id": m31.BACKEND_ID,
    },
    SIMPLEPIM_PROVIDER_ID: {
        "route_id": M41_ROUTE_ID,
        "backend_id": M41_BACKEND_ID,
    },
}


def load_upmem_hardware_taskgraph_m4_1_suite(path: Path) -> m31.M31Suite:
    resolved = path.resolve()
    if resolved != CANONICAL_SUITE_PATH.resolve():
        raise ValueError("hardware_profile_violation: M4.1 requires the committed suite")
    raw = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("suite_id") != "upmem_hardware_taskgraph_m4_1":
        raise ValueError("hardware_profile_violation: M4.1 suite identity differs")
    if raw.get("schema_version") != 1 or raw.get("fail_fast") is not True:
        raise ValueError("hardware_profile_violation: M4.1 suite schema differs")
    defaults = raw.get("defaults")
    workloads = raw.get("workloads")
    routes = raw.get("routes")
    metadata = raw.get("metadata")
    if (
        not isinstance(defaults, dict)
        or defaults.get("warmups") != 1
        or defaults.get("repeats") != 5
        or defaults.get("planner") != {"engine": "opt_einsum", "optimize": "greedy"}
        or not isinstance(workloads, list)
        or len(workloads) != 1
        or not isinstance(routes, list)
        or len(routes) != 2
        or not isinstance(metadata, dict)
    ):
        raise ValueError("hardware_profile_violation: M4.1 workload contract differs")
    profile_data = metadata.get("hardware_profile")
    profile = parse_simplepim_frontier_profile(profile_data)
    workload = workloads[0]
    if not isinstance(workload, dict):
        raise ValueError("hardware_profile_violation: M4.1 workload is invalid")
    if workload.get("expected_path") != m31.EXPECTED_PATH:
        raise ValueError("hardware_profile_violation: M4.1 path differs from M3.1")
    if workload.get("circuit") != {
        "kind": "qasm_file",
        "name": "one_qubit_ry_h_ry_a",
        "path": m31.EXPECTED_QASM,
    }:
        raise ValueError("hardware_profile_violation: M4.1 circuit differs from M3.1")
    route_ids = {route.get("id") for route in routes if isinstance(route, dict)}
    if route_ids != set(ROUTE_VARIANTS):
        raise ValueError("hardware_profile_violation: M4.1 provider variants differ")
    suite = {
        "schema_version": 2,
        "suite_id": raw["suite_id"],
        "metadata": metadata,
        "warmups": 1,
        "repeats": 5,
        "planner": dict(defaults["planner"]),
        "cases": [
            {
                **{key: value for key, value in workload.items() if key != "id"},
                "case_id": workload["id"],
                "workload_id": workload["id"],
            }
        ],
        "route_policy": {"routes": list(ROUTE_VARIANTS), "fail_fast": True},
        "_suite_path": str(resolved),
    }
    return m31.M31Suite(resolved, suite, profile)


def prepare_upmem_hardware_taskgraph_m4_1(
    root_dir: Path,
    *,
    suite_path: Path,
    build: bool = False,
    environment: Mapping[str, str] | None = None,
) -> JsonDict:
    suite = load_upmem_hardware_taskgraph_m4_1_suite(suite_path)
    env = dict(os.environ if environment is None else environment)
    plan_dir = _unique_dir(root_dir / "build" / "upmem_hardware_taskgraph_m4_1_plan")
    plan_dir.mkdir(parents=True, exist_ok=False)
    (plan_dir / "config").mkdir()
    shutil.copy2(suite.path, plan_dir / "config" / "resolved_suite.yml")
    write_json(plan_dir / "config" / "hardware_profile.json", _profile_json(suite.profile))
    write_json(plan_dir / "environment.json", capture_environment(root_dir))
    prepared = None
    built: SimplePimFrontierBuild | None = None
    build_attempted = False
    status = "prepared"
    failure_stage = None
    failure_reason = None
    try:
        case = suite.suite["cases"][0]
        prepared = m31._prepare_case(root_dir, plan_dir / "cases" / sanitize(case["case_id"]), suite, case)
        if build:
            build_attempted = True
            built = build_simplepim_frontier_session(
                root_dir, plan_dir / "native_session", profile=suite.profile, environment=env
            )
            package = m31._write_package(
                prepared, built.session_root, built.simplepim.dpu_binary, request_id="prepare"
            )
            prepared = replace(prepared, package=package)
    except Exception as exc:
        status = "failed"
        failure_stage = _failure_stage(str(exc), "hardware_profile_violation")
        failure_reason = str(exc)
    summary = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "status": status,
        "failure_stage": failure_stage,
        "failure_reason": failure_reason,
        "suite_id": suite.suite["suite_id"],
        "route_id": M41_ROUTE_ID,
        "backend_id": M41_BACKEND_ID,
        "provider_id": PROVIDER_ID,
        "profile": _profile_json(suite.profile),
        "prepared_case": m31._prepared_json(prepared) if prepared is not None else None,
        "native_build": (
            simplepim_build_metadata(built, plan_dir)
            if built
            else {"attempted": build_attempted, "status": "failed" if build_attempted else "not_requested"}
        ),
        "dpu_allocation_attempted": False,
        "dpu_launch_attempted": False,
        "raw_sdk_route_present": True,
        "simplepim_management_route_present": True,
        "claim_boundary": CLAIM_BOUNDARY,
    }
    summary_path = plan_dir / "upmem_hardware_taskgraph_m4_1_plan.json"
    write_json(summary_path, summary)
    return {
        "plan_dir": str(plan_dir),
        "summary_path": str(summary_path),
        "status": status,
        "dpu_allocation_attempted": False,
        "dpu_launch_attempted": False,
    }


def run_upmem_hardware_taskgraph_m4_1(
    root_dir: Path,
    *,
    suite_path: Path,
    environment: Mapping[str, str] | None = None,
) -> JsonDict:
    env = dict(os.environ if environment is None else environment)
    _require_execution_environment(env)
    suite = load_upmem_hardware_taskgraph_m4_1_suite(suite_path)
    run_dir = create_run_dir(
        root_dir,
        str(suite.suite["suite_id"]),
        artifact_kind=EVIDENCE_ARTIFACT_KIND,
        route_label="upmem_hw_taskgraph_m4_1",
    )
    return _run_suite(root_dir, run_dir, suite, env)


def _run_suite(root_dir: Path, run_dir: Path, suite: m31.M31Suite, env: Mapping[str, str]) -> JsonDict:
    shutil.copy2(suite.path, run_dir / "config" / "resolved_suite.yml")
    write_json(run_dir / "config" / "hardware_profile.json", _profile_json(suite.profile))
    write_json(run_dir / "environment.json", capture_environment(root_dir))
    run_manifest = write_run_manifest(
        run_dir,
        run_kind="upmem_hardware_taskgraph_m4_1",
        suite_id=suite.suite["suite_id"],
        suite_path=str(suite.path),
        artifact_kind=EVIDENCE_ARTIFACT_KIND,
        route_label="upmem_hw_taskgraph_m4_1",
        route_id=M41_ROUTE_ID,
        backend_id=M41_BACKEND_ID,
        execution_scope="physical_two_dpu_frontier_differential_control_provider",
        evidence_type="executed_dispatch_only",
        upmem_execution_mode="frontier_graph_request",
        artifact_retention="full",
        summary="upmem_hardware_taskgraph_m4_1_summary.json",
        policies=("opt_einsum_greedy",),
        quantization_modes=("none",),
        root_dir=root_dir,
    )
    run_manifest["requested_environment"] = m31._requested_environment(env)
    records: list[JsonDict] = []
    warmups: list[JsonDict] = []
    build: SimplePimFrontierBuild | None = None
    prepared = None
    failure_stage = None
    failure_reason = None
    try:
        build = build_simplepim_frontier_session(
            root_dir, run_dir / "native_session", profile=suite.profile, environment=env
        )
        case = suite.suite["cases"][0]
        prepared = m31._prepare_case(root_dir, run_dir / "cases" / sanitize(case["case_id"]), suite, case)
        for warmup_id in range(1):
            pair = _execute_pair(build, prepared, suite, env, run_dir, f"warmup-{warmup_id:02d}", True)
            warmups.extend(pair)
            if any(row.get("status") != "completed" for row in pair):
                break
        if not warmups or all(row.get("status") == "completed" for row in warmups):
            for repeat_id in range(5):
                pair = _execute_pair(build, prepared, suite, env, run_dir, f"measured-{repeat_id:02d}", False)
                records.extend(pair)
                if any(row.get("status") != "completed" for row in pair):
                    break
    except Exception as exc:
        failure_stage = _failure_stage(str(exc), "native_build_failed")
        failure_reason = str(exc)

    write_jsonl(run_dir / "warmups.jsonl", warmups)
    write_normalized_records(run_dir, records)
    passed = (
        build is not None
        and prepared is not None
        and len(warmups) == 2
        and len(records) == 10
        and all(row.get("status") == "completed" for row in [*warmups, *records])
        and all(row.get("raw_vs_simplepim_output_equal") is True for row in [*warmups, *records])
        and all(row.get("scientific_identity_equal") is True for row in [*warmups, *records])
        and all(row.get("complete_taskgraph_executed") is True for row in [*warmups, *records])
        and all(row.get("all_tasks_completed") is True for row in [*warmups, *records])
    )
    if not passed and failure_stage is None:
        failure_stage = next(
            (str(row.get("failure_stage")) for row in [*warmups, *records] if row.get("failure_stage")),
            "execution_failed",
        )
    if failure_reason is None and failure_stage is not None:
        failure_reason = next(
            (str(row.get("reason")) for row in [*warmups, *records] if row.get("failure_stage")),
            None,
        )
    summary_path = run_dir / "upmem_hardware_taskgraph_m4_1_summary.json"
    summary = {
        "schema_version": RUN_SCHEMA_VERSION,
        "status": "completed" if passed else "failed",
        "failure_stage": failure_stage,
        "failure_reason": failure_reason,
        "suite_id": suite.suite["suite_id"],
        "route_id": M41_ROUTE_ID,
        "backend_id": M41_BACKEND_ID,
        "provider_id": PROVIDER_ID,
        "row_count": len(records),
        "warmup_count": len(warmups),
        "provider_row_counts": {
            provider: sum(row.get("provider_id") == provider for row in records)
            for provider in (RAW_PROVIDER_ID, SIMPLEPIM_PROVIDER_ID)
        },
        "graph_identity": _graph_identity(prepared),
        "cross_route_output_equality": "passed" if passed else "not_run",
        "native_build": simplepim_build_metadata(build, run_dir) if build else {"attempted": True, "status": "failed"},
        "allocation_scope": "per_request",
        "persistent_allocation": False,
        "hardware": passed,
        "simulator_kernel_executed": False,
        "cpu_fallback_used": False,
        "validation_status": "passed" if passed else "failed",
        "scientific_validation_status": "passed" if passed else "failed",
        "timing_is_bringup_only": True,
        "hardware_speedup_applicable": False,
        "claim_boundary": CLAIM_BOUNDARY,
        "no_speedup_claim": True,
        "no_scaling_claim": True,
        "no_concurrency_claim": True,
        "no_energy_claim": True,
        "source_commit": _git_commit(root_dir),
        "normalized_records": "normalized_records.jsonl",
        "warmups": "warmups.jsonl",
        "requested_environment": m31._requested_environment(env),
    }
    write_json(summary_path, summary)
    run_manifest.update({
        "summary": summary_path.name,
        "hardware_available": "verified_by_execution" if passed else "not_verified_by_execution",
        "evidence_type": "executed_dispatch_only",
        "claim_boundary": CLAIM_BOUNDARY,
        "timing_is_bringup_only": True,
        "hardware_speedup_applicable": False,
        "allocation_scope": "per_request",
        "persistent_allocation": False,
    })
    write_json(run_dir / "run_manifest.json", run_manifest)
    return {
        "run_dir": str(run_dir),
        "summary_path": str(summary_path),
        "status": "completed" if passed else "failed",
        "row_count": len(records),
    }


def _execute_pair(
    build: SimplePimFrontierBuild,
    prepared: Any,
    suite: m31.M31Suite,
    env: Mapping[str, str],
    run_dir: Path,
    request_id: str,
    warmup: bool,
) -> list[JsonDict]:
    outputs: dict[str, Any] = {}
    rows: list[JsonDict] = []
    for provider in (RAW_PROVIDER_ID, SIMPLEPIM_PROVIDER_ID):
        rows.append(
            _execute_provider(
                provider, build, prepared, suite, env, run_dir, request_id, warmup, outputs
            )
        )
    equal = False
    if len(outputs) == 2:
        equal = bool((outputs[RAW_PROVIDER_ID] == outputs[SIMPLEPIM_PROVIDER_ID]).all())
    identity_fields = (
        "circuit_semantics_hash",
        "tensor_network_hash",
        "contraction_plan_hash",
        "graph_serialized_sha256",
        "input_tensor_hash",
        "numeric_mode",
        "source_task_count",
        "frontier_wave_count",
        "task_assignment_fingerprint",
        "package_file_sha256",
    )
    identity_equal = len(rows) == 2 and all(
        rows[0].get(field) == rows[1].get(field) for field in identity_fields
    )
    for row in rows:
        row["raw_vs_simplepim_output_equal"] = equal
        row["scientific_identity_equal"] = identity_equal
        row["raw_vs_simplepim_max_abs_error"] = (
            float(abs(outputs[RAW_PROVIDER_ID] - outputs[SIMPLEPIM_PROVIDER_ID]).max())
            if len(outputs) == 2
            else None
        )
    return rows


def _execute_provider(
    provider: str,
    build: SimplePimFrontierBuild,
    prepared: Any,
    suite: m31.M31Suite,
    env: Mapping[str, str],
    run_dir: Path,
    request_id: str,
    warmup: bool,
    outputs: dict[str, Any],
) -> JsonDict:
    started = time.perf_counter()
    native = None
    actual = None
    cpu_assessment = _cpu_reference_assessment(None, prepared.reference)
    package_file_sha256 = None
    provider_build = build.raw if provider == RAW_PROVIDER_ID else build.simplepim
    native_request_id = f"{provider}-{request_id}"
    try:
        package = m31._write_package(
            prepared, build.session_root, provider_build.dpu_binary, request_id=native_request_id
        )
        if package.package_path is not None:
            package_file_sha256 = hashlib.sha256(
                package.package_path.read_bytes()
            ).hexdigest()
        response_path = build.session_root / f"{sanitize(native_request_id)}_response.json"
        if provider == RAW_PROVIDER_ID:
            raw_profile = parse_hardware_frontier_profile(
                {**m31.EXPECTED_PROFILE, "timeout_s": suite.profile.timeout_s}
            )
            native = execute_hardware_frontier_session(
                build.raw,
                manifest_path=package.manifest_path,
                response_path=response_path,
                profile=raw_profile,
                environment=env,
            )
        else:
            native = execute_simplepim_frontier_session(
                build,
                manifest_path=package.manifest_path,
                response_path=response_path,
                profile=suite.profile,
                environment=env,
            )
        if native.status != "completed":
            row = m31._record(
                suite, prepared, native, None, native_request_id, warmup, started, "failed",
                native.failure_stage or "native request failed", run_dir=run_dir,
                failure_stage=native.failure_stage or "native_host_failed",
            )
        else:
            manifest = json.loads(package.manifest_path.read_text(encoding="utf-8"))
            actual = m31._load_final_output(
                build.session_root, manifest, prepared.graph.tasks[-1].output_shape
            )
            cpu_assessment = _cpu_reference_assessment(actual, prepared.reference)
            if cpu_assessment["cpu_reference_validation_status"] != "passed":
                raise ValueError(str(cpu_assessment["cpu_reference_validation_reason"]))
            outputs[provider] = actual
            row = m31._record(
                suite, prepared, native, actual, native_request_id, warmup, started, "completed",
                None, run_dir=run_dir, validated_native_response=True,
            )
    except Exception as exc:
        row = m31._record(
            suite, prepared, native, actual, native_request_id, warmup, started, "failed", str(exc),
            run_dir=run_dir, failure_stage=_failure_stage(str(exc), "response_evidence_invalid"),
        )
    row.update(_provider_evidence(provider, build, run_dir, native))
    row.update(_scientific_identity(prepared, native, package_file_sha256))
    row.update(cpu_assessment)
    row["validation_tolerance_abs"] = CPU_REFERENCE_ATOL
    row["validation_max_abs_error"] = cpu_assessment["cpu_reference_max_abs_error"]
    if actual is not None:
        row["output_shape"] = list(actual.shape)
    row["repeat_id"] = None if warmup else int(request_id.rsplit("-", 1)[-1])
    row["request_id"] = native_request_id
    return row


def _provider_evidence(
    provider: str,
    build: SimplePimFrontierBuild,
    run_dir: Path,
    native: Any,
) -> JsonDict:
    response = (
        native.response
        if native is not None and isinstance(getattr(native, "response", None), Mapping)
        else None
    )
    simple = provider == SIMPLEPIM_PROVIDER_ID
    allocation = response.get("allocation") if response is not None else None
    release = response.get("release") if response is not None else None

    def field(name: str) -> Any:
        return response.get(name) if response is not None else None

    def nested(value: Any, name: str) -> Any:
        return value.get(name) if isinstance(value, Mapping) else None

    return {
        "provider_id": provider,
        "native_provider_id": field("provider_id"),
        "control_provider": field("control_provider"),
        "kernel_provider": field("kernel_provider"),
        "route_id": M41_ROUTE_ID if simple else m31.ROUTE_ID,
        "backend_id": M41_BACKEND_ID if simple else m31.BACKEND_ID,
        "hardware_profile_version": M41_PROFILE_ID if simple else RAW_PROFILE_ID,
        "executor_config_hash": executor_config_hash(
            M41_ROUTE_ID if simple else m31.ROUTE_ID,
            {
                "profile": M41_PROFILE_ID if simple else RAW_PROFILE_ID,
                "backend_id": M41_BACKEND_ID if simple else m31.BACKEND_ID,
                "control_provider": SIMPLEPIM_PROVIDER_ID if simple else RAW_PROVIDER_ID,
                "kernel_provider": KERNEL_PROVIDER_ID,
                "allocation_scope": "per_request",
            },
        ),
        "target_requested": field("target_requested"),
        "target_observed": field("target_observed"),
        "hardware_execution": field("hardware_execution"),
        "hardware_allocation_verified": nested(allocation, "verified"),
        "native_kernel_executed": field("hardware_kernel_executed"),
        "hardware_kernel_executed": field("hardware_kernel_executed"),
        "simulator_kernel_executed": field("simulator_kernel_executed"),
        "cpu_fallback_used": field("cpu_fallback_used"),
        "fallback_used": (
            bool(field("cpu_fallback_used") or field("simulator_kernel_executed"))
            if response is not None
            and field("cpu_fallback_used") is not None
            and field("simulator_kernel_executed") is not None
            else None
        ),
        "no_cpu_fallback": field("no_cpu_fallback"),
        "no_simulator_fallback": field("no_simulator_fallback"),
        "native_failure_fallback_used": field("native_failure_fallback_used"),
        "hardware_no_fallback": field("hardware_no_fallback"),
        "simplepim_management_api_used": field("simplepim_management_api_used"),
        "simplepim_management_extension": (
            "table_management_init_with_profile" if simple else "not_used"
        ),
        "simplepim_source_commit": build.simplepim_source_commit if simple else None,
        "simplepim_source_dirty": build.simplepim_source_dirty if simple else None,
        "simplepim_staged_source_tree_sha256": (
            build.simplepim_staged_source_tree_sha256 if simple else None
        ),
        "simplepim_patch_sha256": build.simplepim_patch_sha256 if simple else None,
        "simplepim_stage_manifest_sha256": (
            build.simplepim_stage_manifest_sha256 if simple else None
        ),
        "simplepim_initialization_binary_hash": (
            build.simplepim_initialization_binary_hash if simple else None
        ),
        "simplepim_stage_manifest": (
            artifact_ref(
                run_dir,
                build.simplepim_stage_manifest.resolve().relative_to(run_dir.resolve()),
                role="simplepim_stage_manifest",
            )
            if simple
            else None
        ),
        "simplepim_management_allocation_used": field("simplepim_management_allocation_used"),
        "simplepim_management_object_created": field("simplepim_management_object_created"),
        "allocation_source": field("allocation_source"),
        "allocation_profile": field("allocation_profile"),
        "simplepim_operator_api_used": field("simplepim_operator_api_used"),
        "simplepim_operator_names": field("simplepim_operator_names"),
        "simplepim_kernel_executed": field("simplepim_kernel_executed"),
        "raw_sdk_direct_allocation_used": field("raw_sdk_direct_allocation_used"),
        "raw_sdk_load_used": field("raw_sdk_load_used"),
        "raw_sdk_transfer_used": field("raw_sdk_transfer_used"),
        "raw_sdk_launch_used": field("raw_sdk_launch_used"),
        "raw_sdk_sync_used": field("raw_sdk_sync_used"),
        "raw_sdk_custom_binary_load": field("raw_sdk_load_used"),
        "raw_sdk_custom_transfer": field("raw_sdk_transfer_used"),
        "raw_sdk_custom_launch": field("raw_sdk_launch_used"),
        "raw_sdk_control_calls_used": field("raw_sdk_control_calls_used"),
        "thesis_owned_kernel_executed": field("thesis_owned_kernel_executed"),
        "thesis_resident_kernel_executed": field("thesis_resident_kernel_executed"),
        "simplepim_heap_used": field("simplepim_heap_used"),
        "simplepim_table_transport_used": field("simplepim_table_transport_used"),
        "any_task_completed": field("any_task_completed"),
        "all_tasks_completed": field("all_tasks_completed"),
        "complete_taskgraph_executed": field("complete_taskgraph_executed"),
        "provider_release_attempted": field("provider_release_attempted"),
        "provider_release_succeeded": field("provider_release_succeeded"),
        "provider_release_error": field("provider_release_error"),
        "release_attempted": nested(release, "attempted"),
        "release_confirmed": nested(release, "confirmed"),
        "allocation_still_owned": nested(release, "allocation_still_owned"),
        "allocation_scope": "per_request",
        "persistent_allocation": False,
        "timing_is_bringup_only": True,
        "hardware_speedup_applicable": False,
        "native_provider_response": response,
        "application_visible_h2d_bytes": field("actual_h2d_bytes"),
        "application_visible_d2h_bytes": field("actual_d2h_bytes"),
        "application_visible_transfer_bytes": field("actual_transfer_bytes"),
        "package_hash": (
            response.get("hashes", {}).get("package_fnv1a64")
            if response is not None and isinstance(response.get("hashes"), Mapping)
            else None
        ),
    }


def _cpu_reference_assessment(actual: Any, reference: Any) -> JsonDict:
    if actual is None:
        return {
            "cpu_reference_validation_contract": CPU_REFERENCE_VALIDATION_CONTRACT,
            "cpu_reference_validation_status": "not_run",
            "cpu_reference_validation_reason": None,
            "cpu_reference_shape_match": None,
            "cpu_reference_max_abs_error": None,
        }
    observed = np.asarray(actual)
    expected = np.asarray(reference)
    shape_match = observed.shape == expected.shape
    max_abs_error = (
        float(np.max(np.abs(observed - expected)))
        if shape_match and observed.size > 0
        else (0.0 if shape_match else None)
    )
    passed = shape_match and np.allclose(
        observed, expected, rtol=0.0, atol=CPU_REFERENCE_ATOL
    )
    reason = None
    if not shape_match:
        reason = (
            "output_validation_failed: final output shape differs from CPU reference "
            f"({observed.shape!r} != {expected.shape!r})"
        )
    elif not passed:
        reason = (
            "output_validation_failed: final output exceeds CPU reference tolerance "
            f"(max_abs_error={max_abs_error}, atol={CPU_REFERENCE_ATOL}, rtol=0)"
        )
    return {
        "cpu_reference_validation_contract": CPU_REFERENCE_VALIDATION_CONTRACT,
        "cpu_reference_validation_status": "passed" if passed else "failed",
        "cpu_reference_validation_reason": reason,
        "cpu_reference_shape_match": shape_match,
        "cpu_reference_max_abs_error": max_abs_error,
    }


def _scientific_identity(prepared: Any, native: Any, package_file_sha256: str | None) -> JsonDict:
    response = native.response if native is not None else {}
    wave_plan = response.get("wave_plan")
    assignments = json.dumps(wave_plan, sort_keys=True, separators=(",", ":")) if wave_plan is not None else None
    graph_path = prepared.case_dir / "task_graph.json"
    return {
        "circuit_semantics_hash": prepared.graph.circuit_semantics_hash,
        "tensor_network_hash": prepared.graph.tensor_network_hash,
        "contraction_plan_hash": prepared.graph.contraction_plan_hash,
        "graph_serialized_sha256": _sha256_file(graph_path),
        "input_tensor_hash": _input_tensor_hash(prepared.network),
        "numeric_mode": response.get("numeric_mode", "none"),
        "source_task_count": response.get("launch", {}).get("task_count") if isinstance(response.get("launch"), Mapping) else None,
        "frontier_wave_count": len(wave_plan) if isinstance(wave_plan, list) else None,
        "task_assignment_fingerprint": hashlib.sha256(assignments.encode("utf-8")).hexdigest() if assignments is not None else None,
        "wave_plan": wave_plan,
        "package_file_sha256": package_file_sha256,
    }


def _sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _input_tensor_hash(network: Any) -> str | None:
    tensors = getattr(network, "tensors", None)
    if tensors is None:
        return None
    digest = hashlib.sha256()
    for tensor in sorted(tensors, key=lambda item: str(item.spec.id)):
        array = getattr(tensor, "array", None)
        if array is None:
            return None
        digest.update(str(tensor.spec.id).encode("utf-8"))
        digest.update(str(tensor.spec.dtype).encode("utf-8"))
        digest.update(json.dumps(list(tensor.spec.shape), separators=(",", ":")).encode("utf-8"))
        digest.update(np.asarray(array).tobytes(order="C"))
    return digest.hexdigest()


def _profile_json(profile: SimplePimFrontierProfile) -> JsonDict:
    return {
        "hardware_profile_version": profile.version,
        "target": profile.target,
        "backend_id": M41_BACKEND_ID,
        "route_id": M41_ROUTE_ID,
        "native_schema": m31.NATIVE_SCHEMA,
        "requested_dpu_count": profile.requested_dpu_count,
        "tasklets_per_dpu": profile.tasklets_per_dpu,
        "numeric_mode": profile.numeric_mode,
        "numeric_modes": [profile.numeric_mode],
        "synchronous_execution": True,
        "device_launch_mode": "asynchronous_dpu_set",
        "host_completion_mode": "blocking_sync",
        "timeout_s": profile.timeout_s,
        "performance_claim_applicable": False,
    }


def _graph_identity(prepared: Any | None) -> JsonDict | None:
    if prepared is None:
        return None
    return {
        "circuit_semantics_hash": prepared.graph.circuit_semantics_hash,
        "tensor_network_hash": prepared.graph.tensor_network_hash,
        "contraction_plan_hash": prepared.graph.contraction_plan_hash,
        "path": [list(step) for step in prepared.graph.path],
        "task_count": len(prepared.graph.tasks),
    }


def _require_execution_environment(environment: Mapping[str, str]) -> None:
    _require_physical_opt_in(environment)


def _failure_stage(reason: str, default: str) -> str:
    stage = reason.split(":", 1)[0].strip()
    return stage if stage and " " not in stage else default


def _git_commit(root_dir: Path) -> str | None:
    completed = subprocess.run(
        ["git", "-C", str(root_dir), "rev-parse", "--verify", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def _unique_dir(parent: Path) -> Path:
    parent.mkdir(parents=True, exist_ok=True)
    base = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    candidate = parent / base
    suffix = 1
    while candidate.exists():
        candidate = parent / f"{base}_{suffix:02d}"
        suffix += 1
    return candidate


__all__ = [
    "load_upmem_hardware_taskgraph_m4_1_suite",
    "prepare_upmem_hardware_taskgraph_m4_1",
    "run_upmem_hardware_taskgraph_m4_1",
]
