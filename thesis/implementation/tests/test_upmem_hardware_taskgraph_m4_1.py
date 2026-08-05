from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
import pytest

from quantum_bench.bench import upmem_hardware_taskgraph_m4_1 as m41
import quantum_bench.targets.upmem.simplepim_frontier_session as simplepim_session
from scripts import upmem_m4_1_report
from quantum_bench.targets.upmem.simplepim_frontier_session import (
    M41_BACKEND_ID,
    M41_PROFILE_ID,
    M41_ROUTE_ID,
    SIMPLEPIM_MANAGEMENT_API,
    SIMPLEPIM_PROVIDER_ID,
    parse_simplepim_frontier_profile,
    validate_simplepim_frontier_response,
)


ROOT = Path(__file__).parents[1]
SUITE_PATH = ROOT / "configs" / "suites" / "upmem_hardware_taskgraph_m4_1.yml"


def test_m41_suite_is_fixed_and_reuses_m31_graph() -> None:
    suite = m41.load_upmem_hardware_taskgraph_m4_1_suite(SUITE_PATH)
    case = suite.suite["cases"][0]

    assert suite.suite["warmups"] == 1
    assert suite.suite["repeats"] == 5
    assert suite.suite["planner"] == {"engine": "opt_einsum", "optimize": "greedy"}
    assert case["circuit"]["path"] == m41.m31.EXPECTED_QASM
    assert case["expected_path"] == m41.m31.EXPECTED_PATH
    assert suite.suite["route_policy"]["routes"] == ["raw_sdk", SIMPLEPIM_PROVIDER_ID]
    assert suite.profile.version == M41_PROFILE_ID
    assert suite.profile.requested_dpu_count == 2
    assert suite.profile.tasklets_per_dpu == 1
    assert suite.profile.performance_claim_applicable is False


def test_m41_rejects_noncanonical_suite() -> None:
    with pytest.raises(ValueError, match="committed suite"):
        m41.load_upmem_hardware_taskgraph_m4_1_suite(ROOT / "configs" / "suites" / "upmem_hardware_frontier_m3_1.yml")


@pytest.mark.parametrize(
    "override",
    [
        {"target": "simulator"},
        {"requested_dpu_count": 1},
        {"tasklets_per_dpu": 2},
        {"numeric_mode": "int8"},
        {"performance_claim_applicable": True},
    ],
)
def test_m41_profile_rejects_scope_expansion(override: dict[str, object]) -> None:
    value: dict[str, object] = {
        "hardware_profile_version": M41_PROFILE_ID,
        "target": "hardware",
        "requested_dpu_count": 2,
        "tasklets_per_dpu": 1,
        "numeric_mode": "none",
        "performance_claim_applicable": False,
    }
    value.update(override)
    with pytest.raises(ValueError, match="hardware_profile_violation"):
        parse_simplepim_frontier_profile(value)


def test_m41_prepare_does_not_allocate_or_build(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    plan_dir = tmp_path / "plan"
    monkeypatch.setattr(m41, "_unique_dir", lambda _parent: plan_dir)
    monkeypatch.setattr(m41, "build_simplepim_frontier_session", pytest.fail)

    result = m41.prepare_upmem_hardware_taskgraph_m4_1(ROOT, suite_path=SUITE_PATH)
    payload = json.loads(Path(result["summary_path"]).read_text(encoding="utf-8"))

    assert result["status"] == "prepared"
    assert payload["dpu_allocation_attempted"] is False
    assert payload["dpu_launch_attempted"] is False
    assert payload["prepared_case"]["task_count"] == 3
    assert payload["prepared_case"]["path"] == [[0, 1], [0, 1], [0, 1]]


@pytest.mark.skipif(
    shutil.which("dpu-pkg-config") is None
    or shutil.which("dpu-upmem-dpurte-clang") is None
    or not os.environ.get("UPMEM_HOME"),
    reason="UPMEM SDK compiler is unavailable",
)
def test_m41_prepared_simplepim_manifest_passes_native_validation_without_allocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan_dir = tmp_path / "plan"
    monkeypatch.setattr(m41, "_unique_dir", lambda _parent: plan_dir)

    result = m41.prepare_upmem_hardware_taskgraph_m4_1(
        ROOT,
        suite_path=SUITE_PATH,
        build=True,
        environment=dict(os.environ),
    )

    assert result["status"] == "prepared"
    summary = json.loads(Path(result["summary_path"]).read_text(encoding="utf-8"))
    validation = summary["simplepim_manifest_validation"]
    assert validation["status"] == "passed"
    assert validation["allocation_attempted"] is False
    assert validation["launch_attempted"] is False

    manifest_path = next(
        path
        for path in (plan_dir / "native_session").rglob("*.json")
        if json.loads(path.read_text(encoding="utf-8")).get("manifest_kind")
        == "frontier_graph_request"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["route_id"] == m41.M41_ROUTE_ID
    assert manifest["backend_id"] == m41.M41_BACKEND_ID
    assert manifest["hardware_profile_version"] == m41.M41_PROFILE_ID
    assert manifest["frontier_identity"]["route_id"] == m41.M41_ROUTE_ID
    assert manifest["frontier_identity"]["backend_id"] == m41.M41_BACKEND_ID
    assert manifest["frontier_identity"]["profile_id"] == m41.M41_PROFILE_ID

    package_path = (manifest_path.parent / manifest["package_path"]).resolve()
    assert manifest["resident_package_binding"]["package_sha256"] == hashlib.sha256(
        package_path.read_bytes()
    ).hexdigest()
    assert summary["prepared_case"]["frontier_manifest"]["resident_package_binding"] == manifest[
        "resident_package_binding"
    ]


def test_m41_execution_requires_explicit_physical_opt_in() -> None:
    with pytest.raises(ValueError, match="hardware_opt_in_missing"):
        m41._require_execution_environment({})
    with pytest.raises(ValueError, match="selects a simulator"):
        m41._require_execution_environment(
            {"UPMEM_ALLOW_PHYSICAL_HARDWARE": "1", "UPMEM_PROFILE": "simulator"}
        )


def test_simplepim_response_requires_complete_taskgraph(monkeypatch: pytest.MonkeyPatch) -> None:
    required = {
        "provider_id": SIMPLEPIM_PROVIDER_ID,
        "route_id": M41_ROUTE_ID,
        "backend_id": M41_BACKEND_ID,
        "hardware_profile_version": M41_PROFILE_ID,
        "profile_id": M41_PROFILE_ID,
        "control_provider": SIMPLEPIM_PROVIDER_ID,
        "kernel_provider": "thesis_resident_generic_contract",
        "simplepim_management_api_used": SIMPLEPIM_MANAGEMENT_API,
        "provider_init_called": True,
        "provider_init_succeeded": True,
        "simplepim_management_init_called": True,
        "simplepim_management_allocation_used": True,
        "simplepim_management_object_created": True,
        "allocation_source": SIMPLEPIM_PROVIDER_ID,
        "allocation_profile": "backend=hw",
        "simplepim_operator_api_used": False,
        "simplepim_operator_names": [],
        "simplepim_kernel_executed": False,
        "raw_sdk_direct_allocation_used": False,
        "raw_sdk_load_used": True,
        "raw_sdk_transfer_used": True,
        "raw_sdk_launch_used": True,
        "raw_sdk_sync_used": True,
        "raw_sdk_control_calls_used": True,
        "any_task_completed": True,
        "thesis_owned_kernel_executed": True,
        "thesis_resident_kernel_executed": True,
        "simplepim_heap_used": False,
        "simplepim_table_transport_used": False,
        "all_tasks_completed": True,
        "complete_taskgraph_executed": True,
        "provider_release_attempted": True,
        "provider_release_succeeded": True,
        "provider_release_error": 0,
        "timing_is_bringup_only": True,
        "hardware_speedup_applicable": False,
    }
    called = False

    def fake_base(_response: object, _manifest: object) -> None:
        nonlocal called
        called = True

    monkeypatch.setattr(
        "quantum_bench.targets.upmem.simplepim_frontier_session.validate_frontier_native_response",
        fake_base,
    )
    validate_simplepim_frontier_response(required, {})
    assert called is True

    missing = dict(required)
    missing["complete_taskgraph_executed"] = False
    with pytest.raises(ValueError, match="complete_taskgraph_executed"):
        validate_simplepim_frontier_response(missing, {})


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("simplepim_management_allocation_used", False),
        ("simplepim_management_object_created", False),
        ("allocation_source", "raw_sdk"),
        ("raw_sdk_direct_allocation_used", True),
        ("raw_sdk_transfer_used", False),
        ("any_task_completed", False),
        ("provider_release_succeeded", False),
        ("provider_release_error", 1),
    ],
)
def test_simplepim_response_rejects_incorrect_authoritative_flags(
    field: str, invalid: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    response = _simplepim_success_response()
    response[field] = invalid
    monkeypatch.setattr(
        "quantum_bench.targets.upmem.simplepim_frontier_session.validate_frontier_native_response",
        lambda *_args: None,
    )
    with pytest.raises(ValueError, match=field):
        validate_simplepim_frontier_response(response, {})


@pytest.mark.parametrize(
    ("field", "stale_value"),
    [
        ("route_id", m41.m31.ROUTE_ID),
        ("backend_id", m41.m31.BACKEND_ID),
        ("hardware_profile_version", m41.RAW_PROFILE_ID),
        ("profile_id", m41.RAW_PROFILE_ID),
    ],
)
def test_simplepim_execution_response_rejects_stale_provider_identity(
    field: str,
    stale_value: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = _simplepim_success_response()
    response[field] = stale_value
    monkeypatch.setattr(simplepim_session, "validate_frontier_native_response", lambda *_: None)

    with pytest.raises(ValueError, match=field):
        validate_simplepim_frontier_response(response, {})


@pytest.mark.parametrize(
    ("field", "stale_value"),
    [
        ("route_id", m41.m31.ROUTE_ID),
        ("backend_id", m41.m31.BACKEND_ID),
        ("hardware_profile_version", m41.RAW_PROFILE_ID),
        ("profile_id", m41.RAW_PROFILE_ID),
    ],
)
def test_simplepim_parser_response_rejects_stale_provider_identity(
    field: str,
    stale_value: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path = tmp_path / "request.json"
    manifest_path.write_text("{}\n", encoding="ascii")
    host_binary = tmp_path / "host"
    build = SimpleNamespace(
        simplepim=SimpleNamespace(
            session_root=tmp_path,
            build_dir=tmp_path,
            host_binary=host_binary,
        )
    )
    response = {
        "route_id": M41_ROUTE_ID,
        "backend_id": M41_BACKEND_ID,
        "hardware_profile_version": M41_PROFILE_ID,
        "profile_id": M41_PROFILE_ID,
    }
    response[field] = stale_value
    monkeypatch.setattr(
        simplepim_session,
        "_run_command",
        lambda *_args, **_kwargs: {
            "returncode": 0,
            "stdout_snippet": json.dumps(response),
            "stderr_snippet": "",
        },
    )
    monkeypatch.setattr(
        simplepim_session,
        "validate_frontier_package_validation_response",
        lambda *_args: None,
    )

    with pytest.raises(ValueError, match=field):
        simplepim_session.validate_simplepim_frontier_manifest(
            build,
            manifest_path=manifest_path,
            profile=m41.load_upmem_hardware_taskgraph_m4_1_suite(SUITE_PATH).profile,
            environment={},
        )


@pytest.mark.parametrize(
    ("outer_identity", "nested_identity"),
    [
        (
            (m41.m31.ROUTE_ID, m41.m31.BACKEND_ID, m41.RAW_PROFILE_ID),
            (M41_ROUTE_ID, M41_BACKEND_ID, M41_PROFILE_ID),
        ),
        (
            (M41_ROUTE_ID, M41_BACKEND_ID, M41_PROFILE_ID),
            (m41.m31.ROUTE_ID, m41.m31.BACKEND_ID, m41.RAW_PROFILE_ID),
        ),
    ],
)
def test_simplepim_manifest_rewrite_rejects_mixed_outer_and_nested_identity(
    outer_identity: tuple[str, str, str],
    nested_identity: tuple[str, str, str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path = tmp_path / "request.json"
    manifest_path.write_text(
        json.dumps(
            {
                "route_id": outer_identity[0],
                "backend_id": outer_identity[1],
                "hardware_profile_version": outer_identity[2],
                "frontier_identity": {
                    "route_id": nested_identity[0],
                    "backend_id": nested_identity[1],
                    "profile_id": nested_identity[2],
                },
            }
        ),
        encoding="ascii",
    )
    build = SimpleNamespace(simplepim=SimpleNamespace(session_root=tmp_path))
    monkeypatch.setattr(simplepim_session, "_validate_frontier_manifest", lambda *_: None)

    with pytest.raises(ValueError, match="outer and nested"):
        simplepim_session.rewrite_simplepim_frontier_manifest(
            manifest_path,
            build=build,
        )


def test_cpu_reference_contract_accepts_float32_rounding() -> None:
    reference = np.asarray([0.7986355100472927, 0.6018150231520483], dtype=np.complex128)
    actual = reference.real.astype(np.float32).astype(np.complex128)

    result = m41._cpu_reference_assessment(actual, reference)

    assert result["cpu_reference_validation_contract"] == m41.CPU_REFERENCE_VALIDATION_CONTRACT
    assert result["cpu_reference_validation_status"] == "passed"
    assert result["cpu_reference_shape_match"] is True
    assert 0.0 <= result["cpu_reference_max_abs_error"] <= m41.CPU_REFERENCE_ATOL


def test_cpu_reference_contract_rejects_error_above_tolerance() -> None:
    reference = np.asarray([0.25, 0.75], dtype=np.complex128)
    actual = reference + np.asarray([2.0e-6, 0.0])

    result = m41._cpu_reference_assessment(actual, reference)

    assert result["cpu_reference_validation_status"] == "failed"
    assert result["cpu_reference_max_abs_error"] > m41.CPU_REFERENCE_ATOL
    assert "output_validation_failed" in result["cpu_reference_validation_reason"]


def test_provider_evidence_preserves_partial_native_response(tmp_path: Path) -> None:
    stage = tmp_path / "stage.json"
    stage.write_text("{}\n", encoding="ascii")
    build = SimpleNamespace(
        simplepim_source_commit="commit",
        simplepim_source_dirty=False,
        simplepim_staged_source_tree_sha256="tree",
        simplepim_patch_sha256="patch",
        simplepim_stage_manifest_sha256="marker",
        simplepim_initialization_binary_hash="init",
        simplepim_stage_manifest=stage,
    )
    native = SimpleNamespace(
        response={
            "provider_id": SIMPLEPIM_PROVIDER_ID,
            "control_provider": SIMPLEPIM_PROVIDER_ID,
            "kernel_provider": "thesis_resident_generic_contract",
            "raw_sdk_load_used": True,
            "raw_sdk_transfer_used": False,
            "raw_sdk_launch_used": True,
            "thesis_resident_kernel_executed": False,
            "allocation": {"verified": True},
            "release": {"attempted": True, "confirmed": False},
        }
    )

    result = m41._provider_evidence(
        SIMPLEPIM_PROVIDER_ID, build, tmp_path, native
    )

    assert result["raw_sdk_custom_binary_load"] is True
    assert result["raw_sdk_custom_transfer"] is False
    assert result["raw_sdk_custom_launch"] is True
    assert result["raw_sdk_sync_used"] is None
    assert result["thesis_resident_kernel_executed"] is False
    assert result["hardware_allocation_verified"] is True
    assert result["release_confirmed"] is False

    absent = m41._provider_evidence(m41.RAW_PROVIDER_ID, build, tmp_path, None)
    assert absent["raw_sdk_custom_binary_load"] is None
    assert absent["hardware_allocation_verified"] is None
    assert absent["native_provider_response"] is None


def test_m41_report_rejects_empty_and_accepts_complete_records(tmp_path: Path) -> None:
    summary = _report_summary()
    (tmp_path / "upmem_hardware_taskgraph_m4_1_summary.json").write_text(
        json.dumps(summary), encoding="utf-8"
    )
    records_path = tmp_path / "normalized_records.jsonl"
    records_path.write_text("", encoding="utf-8")
    empty = upmem_m4_1_report.inspect_m4_1_run(tmp_path)
    assert empty["status"] == "invalid_or_incomplete"
    assert empty["checks"]["exact_measured_row_count"] is False

    records = _report_records()
    records_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    valid = upmem_m4_1_report.inspect_m4_1_run(tmp_path)
    assert valid["status"] == "valid_functionality_evidence"
    assert valid["provider_counts"] == {"raw_sdk": 5, SIMPLEPIM_PROVIDER_ID: 5}

    records[-1]["provider_release_succeeded"] = False
    records_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    invalid = upmem_m4_1_report.inspect_m4_1_run(tmp_path)
    assert invalid["status"] == "invalid_or_incomplete"
    assert invalid["checks"]["simplepim_provider_truth"] is False


def test_m41_report_rejects_duplicate_and_unpaired_repeats(tmp_path: Path) -> None:
    (tmp_path / "upmem_hardware_taskgraph_m4_1_summary.json").write_text(
        json.dumps(_report_summary()), encoding="utf-8"
    )
    records_path = tmp_path / "normalized_records.jsonl"

    duplicate = _report_records()
    next(
        row
        for row in duplicate
        if row["provider_id"] == "raw_sdk" and row["repeat_id"] == 4
    )["repeat_id"] = 0
    records_path.write_text(
        "".join(json.dumps(record) + "\n" for record in duplicate), encoding="utf-8"
    )
    duplicate_result = upmem_m4_1_report.inspect_m4_1_run(tmp_path)
    assert duplicate_result["status"] == "invalid_or_incomplete"
    assert duplicate_result["checks"]["five_distinct_measured_pairs"] is False

    unpaired = _report_records()
    next(
        row
        for row in unpaired
        if row["provider_id"] == SIMPLEPIM_PROVIDER_ID and row["repeat_id"] == 4
    )["case_id"] = "different_case"
    records_path.write_text(
        "".join(json.dumps(record) + "\n" for record in unpaired), encoding="utf-8"
    )
    unpaired_result = upmem_m4_1_report.inspect_m4_1_run(tmp_path)
    assert unpaired_result["status"] == "invalid_or_incomplete"
    assert unpaired_result["checks"]["five_distinct_measured_pairs"] is False


def test_m41_report_rejects_invalid_pair_error_metrics(tmp_path: Path) -> None:
    (tmp_path / "upmem_hardware_taskgraph_m4_1_summary.json").write_text(
        json.dumps(_report_summary()), encoding="utf-8"
    )
    records_path = tmp_path / "normalized_records.jsonl"

    cpu_error = _report_records()
    cpu_error[0]["cpu_reference_max_abs_error"] = 2.0e-6
    records_path.write_text(
        "".join(json.dumps(record) + "\n" for record in cpu_error), encoding="utf-8"
    )
    cpu_result = upmem_m4_1_report.inspect_m4_1_run(tmp_path)
    assert cpu_result["checks"]["cpu_reference_errors_within_tolerance"] is False

    provider_error = _report_records()
    provider_error[0]["raw_vs_simplepim_max_abs_error"] = 1.0e-8
    records_path.write_text(
        "".join(json.dumps(record) + "\n" for record in provider_error), encoding="utf-8"
    )
    provider_result = upmem_m4_1_report.inspect_m4_1_run(tmp_path)
    assert provider_result["checks"]["exact_same_binary_pair_outputs"] is False


def _simplepim_success_response() -> dict[str, object]:
    return {
        "provider_id": SIMPLEPIM_PROVIDER_ID,
        "route_id": M41_ROUTE_ID,
        "backend_id": M41_BACKEND_ID,
        "hardware_profile_version": M41_PROFILE_ID,
        "profile_id": M41_PROFILE_ID,
        "control_provider": SIMPLEPIM_PROVIDER_ID,
        "kernel_provider": "thesis_resident_generic_contract",
        "simplepim_management_api_used": SIMPLEPIM_MANAGEMENT_API,
        "provider_init_called": True,
        "provider_init_succeeded": True,
        "simplepim_management_init_called": True,
        "simplepim_management_allocation_used": True,
        "simplepim_management_object_created": True,
        "allocation_source": SIMPLEPIM_PROVIDER_ID,
        "allocation_profile": "backend=hw",
        "simplepim_operator_api_used": False,
        "simplepim_operator_names": [],
        "simplepim_kernel_executed": False,
        "raw_sdk_direct_allocation_used": False,
        "raw_sdk_load_used": True,
        "raw_sdk_transfer_used": True,
        "raw_sdk_launch_used": True,
        "raw_sdk_sync_used": True,
        "raw_sdk_control_calls_used": True,
        "any_task_completed": True,
        "all_tasks_completed": True,
        "complete_taskgraph_executed": True,
        "thesis_owned_kernel_executed": True,
        "thesis_resident_kernel_executed": True,
        "provider_release_attempted": True,
        "provider_release_succeeded": True,
        "provider_release_error": 0,
        "simplepim_heap_used": False,
        "simplepim_table_transport_used": False,
        "timing_is_bringup_only": True,
        "hardware_speedup_applicable": False,
    }


def _report_summary() -> dict[str, object]:
    return {
        "status": "completed",
        "row_count": 10,
        "hardware_speedup_applicable": False,
        "timing_is_bringup_only": True,
        "allocation_scope": "per_request",
        "persistent_allocation": False,
        "validation_status": "passed",
        "scientific_validation_status": "passed",
        "cross_route_output_equality": "passed",
        "claim_boundary": "functionality only",
    }


def _report_records() -> list[dict[str, object]]:
    return [
        _report_record(provider, repeat_id)
        for repeat_id in range(5)
        for provider in ("raw_sdk", SIMPLEPIM_PROVIDER_ID)
    ]


def _report_record(provider: str, repeat_id: int) -> dict[str, object]:
    simple = provider == SIMPLEPIM_PROVIDER_ID
    return {
        "case_id": "m3_1_ry_h_ry_a_opt_einsum_greedy",
        "repeat_id": repeat_id,
        "warmup": False,
        "provider_id": provider,
        "native_provider_id": provider,
        "control_provider": provider,
        "kernel_provider": "thesis_resident_generic_contract",
        "status": "completed",
        "raw_vs_simplepim_output_equal": True,
        "scientific_identity_equal": True,
        "hardware_execution": True,
        "target_observed": "hardware",
        "hardware_allocation_verified": True,
        "allocated_dpu_count": 2,
        "native_kernel_executed": True,
        "cpu_fallback_used": False,
        "simulator_kernel_executed": False,
        "no_cpu_fallback": True,
        "no_simulator_fallback": True,
        "native_failure_fallback_used": False,
        "hardware_no_fallback": True,
        "release_attempted": True,
        "release_confirmed": True,
        "allocation_still_owned": False,
        "all_tasks_completed": True,
        "complete_taskgraph_executed": True,
        "raw_sdk_direct_allocation_used": not simple,
        "raw_sdk_load_used": True,
        "raw_sdk_transfer_used": True,
        "raw_sdk_launch_used": True,
        "raw_sdk_sync_used": True,
        "raw_sdk_control_calls_used": True,
        "thesis_resident_kernel_executed": True,
        "cpu_reference_validation_status": "passed",
        "cpu_reference_max_abs_error": 3.787677971267556e-08,
        "validation_tolerance_abs": 1.0e-6,
        "raw_vs_simplepim_max_abs_error": 0.0,
        "circuit_semantics_hash": "circuit-semantics-sha256",
        "tensor_network_hash": "tensor-network-sha256",
        "contraction_plan_hash": "contraction-plan-sha256",
        "graph_serialized_sha256": "serialized-graph-sha256",
        "input_tensor_hash": "input-tensor-sha256",
        "numeric_mode": "none",
        "source_task_count": 3,
        "frontier_wave_count": 2,
        "task_assignment_fingerprint": "task-assignment-sha256",
        "package_file_sha256": "resident-package-sha256",
        "allocation_source": provider,
        "allocation_profile": "backend=hw",
        "simplepim_management_allocation_used": True if simple else False,
        "simplepim_management_object_created": True if simple else False,
        "simplepim_operator_api_used": False,
        "simplepim_operator_names": [],
        "simplepim_kernel_executed": False,
        "provider_release_attempted": True if simple else False,
        "provider_release_succeeded": True if simple else False,
        "provider_release_error": 0,
    }
