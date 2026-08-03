from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.upmem_m2_2_report import ReportError, generate_report, validate_source


MODES = ("none", "per_task_resident_requantize")


def _row(mode: str, repeat: int, *, phase: str = "measured", **updates):
    row = {
        "case_id": "one_qubit_hx_m2_2",
        "workload_id": "one_qubit_hx_m2_2",
        "repeat_id": repeat,
        "phase": phase,
        "numeric_mode": mode,
        "status": "completed",
        "backend_family": "upmem_sdk",
        "route_id": "upmem_tn_hardware_sliced_resident_two_dpu",
        "backend_id": "upmem_sdk_hardware_sliced_resident_two_dpu",
        "target_requested": "hardware",
        "target_observed": "hardware",
        "requested_dpu_count": 2,
        "allocated_dpu_count": 2,
        "tasklets_per_dpu": 1,
        "hardware_execution": True,
        "hardware_functionality_evidence": True,
        "hardware_kernel_executed": True,
        "native_kernel_executed": True,
        "simulator_kernel_executed": False,
        "cpu_fallback_used": False,
        "fixture_version": "upmem_hardware_sliced_resident_m2_2_v1",
        "fixture_scope": "two_operation_h_then_x_full_graph_replicated_prefix",
        "execution_scope": "physical_two_dpu_two_slice_full_replicated_prefix_taskgraph",
        "operation_count": 2,
        "slice_count": 2,
        "source_slice_count": 2,
        "slice_ids": [0, 1],
        "expanded_task_count": 4,
        "executed_task_count": 4,
        "completed_task_count": 4,
        "executed_slice_count": 2,
        "completed_slice_count": 2,
        "source_task_count": 2,
        "source_task_completion_count": 4,
        "source_task_completion_scope": "replicated_slice_operations",
        "slice_model_task_count": 2,
        "slice_model_executed_task_count": 4,
        "observed_operation_completion_counts": [2, 2],
        "device_completion_confirmed": True,
        "native_cleanup_confirmed": True,
        "release_evidence": {"confirmed": True},
        "execution_contract_status": "passed",
        "policy_reference_status": "passed",
        "full_precision_accuracy_status": "passed",
        "scientific_validation_status": "passed",
        "reconstruction_validation_status": "passed",
        "per_slice_output_validation_status": "passed",
        "expected_output_validation": True,
        "strict_cpu_reference_validation": True,
        "transfer_accounting_invariant": True,
        "transfer_matches_manifest_plan": True,
        "source_hashes_preserved": True,
        "slice_package_validation_status": "passed",
        "hardware_speedup_applicable": False,
        "energy_measurement_available": False,
        "actual_h2d_bytes": 3408,
        "actual_d2h_bytes": 16,
        "actual_transfer_bytes": 3424,
        "total_time_s": 0.1 + repeat / 1000,
        "process_time_s": 0.06 + repeat / 1000,
        "reconstruction_time_s": 0.02,
        "stage_timings": {
            "total_route_time_s": 0.05 + repeat / 10000,
            "sync_wait_time_s": 0.001,
            "status": "host_stage_boundaries",
            "kernel_time_s": None,
            "sync_wait_is_not_pure_kernel_time": True,
        },
        "timing_is_bringup_only": True,
        "timing_scope": "host_observed_sdk_process_wall_and_blocking_sync",
        "timing_breakdown_status": "host_stage_boundaries",
        "policy_reference_max_abs_error": 0.0,
        "policy_reference_l2_error": 0.0,
        "full_precision_max_abs_error": 1.2e-8,
        "full_precision_l2_error": 1.7e-8,
        "validation_status": "passed",
        "response_numeric_mode": mode,
        "circuit_semantics_hash": "circuit",
        "tensor_network_hash": "network",
        "contraction_plan_hash": "plan",
        "executor_config_hash": f"executor-{mode}",
        "planned_h2d_bytes": 3408,
        "planned_d2h_bytes": 16,
        "planned_transfer_bytes": 3424,
        "observed_h2d_bytes": 3408,
        "observed_d2h_bytes": 16,
        "observed_transfer_bytes": 3424,
        "application_visible_h2d_bytes": 3408,
        "application_visible_d2h_bytes": 16,
        "application_visible_transfer_bytes": 3424,
        "application_visible_total_bytes": 3424,
    }
    if phase == "measured":
        row.update(
            {
                "duplicate_contraction_check": None,
                "missing_dependency_check": None,
                "dependency_violation_detected": False,
            }
        )
    row.update(updates)
    return row


def _write_run(root: Path, *, bad: dict | None = None) -> Path:
    root.mkdir()
    warmups = [_row(mode, 0, phase="warmup") for mode in MODES]
    measured = [
        _row(mode, repeat, **(bad or {})) for mode in MODES for repeat in range(5)
    ]
    (root / "warmups.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in warmups), encoding="utf-8"
    )
    (root / "normalized_records.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in measured), encoding="utf-8"
    )
    (root / "run_manifest.json").write_text(
        json.dumps(
            {
                "benchmark_source_commit": "commit",
                "benchmark_source_worktree_dirty": False,
                "fixture_version": "upmem_hardware_sliced_resident_m2_2_v1",
                "fixture_scope": "two_operation_h_then_x_full_graph_replicated_prefix",
                "execution_scope": "physical_two_dpu_two_slice_full_replicated_prefix_taskgraph",
                "operation_count": 2,
                "source_task_count": 2,
            }
        ),
        encoding="utf-8",
    )
    (root / "upmem_hardware_sliced_resident_mvp_summary.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "measured_passed_count": 10,
                "warmup_passed_count": 2,
                "fixture_version": "upmem_hardware_sliced_resident_m2_2_v1",
                "fixture_scope": "two_operation_h_then_x_full_graph_replicated_prefix",
                "execution_scope": "physical_two_dpu_two_slice_full_replicated_prefix_taskgraph",
                "operation_count": 2,
                "source_task_count": 2,
                "native_build": {
                    "sdk_tools": {
                        "dpu-upmem-dpurte-clang": "/usr/bin/dpu-upmem-dpurte-clang"
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    (root / "environment.json").write_text(
        json.dumps(
            {
                "hostname": "test-host",
                "git_commit": "commit",
                "upmem": {"dpu_compiler": "/usr/bin/dpu-upmem-dpurte-clang"},
            }
        ),
        encoding="utf-8",
    )
    (root / "config").mkdir()
    (root / "config" / "resolved_suite.yml").write_text(
        "suite_id: upmem_hardware_sliced_resident_m2_2\n", encoding="utf-8"
    )
    (root / "config" / "hardware_profile.json").write_text(
        json.dumps(
            {
                "hardware_profile_version": "hardware_sliced_resident_two_dpu_m2_v1",
                "target": "hardware",
                "backend_id": "upmem_sdk_hardware_sliced_resident_two_dpu",
                "route_id": "upmem_tn_hardware_sliced_resident_two_dpu",
                "requested_dpu_count": 2,
                "slices": 2,
                "tasklets_per_dpu": 1,
                "numeric_modes": list(MODES),
                "performance_claim_applicable": False,
            }
        ),
        encoding="utf-8",
    )
    return root


def test_valid_source_generates_required_tables_and_plots(tmp_path: Path) -> None:
    output = generate_report(
        _write_run(tmp_path / "run"), tmp_path / "report", comparison_root=tmp_path
    )
    assert {path.name for path in output.iterdir()} == {
        "mode_statistics.csv",
        "paired_mode_ratios.csv",
        "validation_rows.csv",
        "runtime_by_mode.png",
        "accuracy_by_mode.png",
        "benchmark_summary.md",
        "report_manifest.json",
    }
    text = (output / "benchmark_summary.md").read_text(encoding="utf-8")
    assert "ratio_of_medians" in text
    assert "median_of_paired_ratios" in text
    assert "no speedup claim" in text
    validation = (output / "validation_rows.csv").read_text(encoding="utf-8")
    assert len(validation.splitlines()) == 13
    assert "record_json" not in validation
    manifest = json.loads((output / "report_manifest.json").read_text(encoding="utf-8"))
    assert manifest["statistics"]["iqr_definition"].startswith("numpy.percentile")
    assert {item["path"] for item in manifest["plots"]} == {
        "runtime_by_mode.png",
        "accuracy_by_mode.png",
    }
    assert (
        manifest["artifacts"]["runtime_by_mode.png"]["source_csv"]
        == "mode_statistics.csv"
    )


def test_source_rejects_mismatched_plan_hash_and_preserves_validation_rows(
    tmp_path: Path,
) -> None:
    run = _write_run(tmp_path / "run")
    records = [
        json.loads(line)
        for line in (run / "normalized_records.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    records[0]["contraction_plan_hash"] = "other"
    (run / "normalized_records.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in records), encoding="utf-8"
    )
    with pytest.raises(ReportError, match="source rejected"):
        generate_report(run, tmp_path / "report", comparison_root=tmp_path)
    rows = (tmp_path / "report" / "validation_rows.csv").read_text(encoding="utf-8")
    assert "passed" in rows
    assert "single_contraction_plan_hash" in (
        tmp_path / "report" / "benchmark_summary.md"
    ).read_text(encoding="utf-8")


def test_validate_source_requires_physical_execution() -> None:
    warmups = [_row(mode, 0, phase="warmup") for mode in MODES]
    measured = [_row(mode, repeat) for mode in MODES for repeat in range(5)]
    _, context = validate_source(warmups, measured)
    assert context["errors"] == []
    measured[0]["simulator_kernel_executed"] = True
    _, context = validate_source(warmups, measured)
    assert "one_or_more_rows_failed_admission" in context["errors"]


@pytest.mark.parametrize(
    ("mutate", "expected"),
    [
        (
            lambda rows: rows.append(_row("none", 4)),
            "duplicate_measured_repeat_ids_none",
        ),
        (
            lambda rows: rows.__setitem__(0, _row("none", 1)),
            "measured_repeat_ids_none=",
        ),
        (
            lambda rows: rows[0].__setitem__("validation_status", "failed"),
            "one_or_more_rows_failed_admission",
        ),
        (
            lambda rows: rows[0].__setitem__("route_id", "wrong_route"),
            "one_or_more_rows_failed_admission",
        ),
        (
            lambda rows: rows[0].__setitem__("backend_id", "wrong_backend"),
            "one_or_more_rows_failed_admission",
        ),
        (
            lambda rows: rows[0].__setitem__("release_evidence", {"confirmed": False}),
            "one_or_more_rows_failed_admission",
        ),
        (
            lambda rows: next(
                row for row in rows if row["numeric_mode"] == MODES[1]
            ).__setitem__("response_numeric_mode", MODES[0]),
            "one_or_more_rows_failed_admission",
        ),
        (
            lambda rows: rows[0].__setitem__("actual_h2d_bytes", True),
            "one_or_more_rows_failed_admission",
        ),
        (
            lambda rows: rows[0].__setitem__("actual_transfer_bytes", 1),
            "one_or_more_rows_failed_admission",
        ),
        (
            lambda rows: rows[0].__setitem__("planned_transfer_bytes", 1),
            "one_or_more_rows_failed_admission",
        ),
        (
            lambda rows: rows[0].__setitem__("observed_d2h_bytes", 1),
            "one_or_more_rows_failed_admission",
        ),
        (
            lambda rows: rows[0].__setitem__("application_visible_total_bytes", 1),
            "one_or_more_rows_failed_admission",
        ),
        (
            lambda rows: rows[0].__setitem__("phase", "warmup"),
            "one_or_more_rows_failed_admission",
        ),
    ],
)
def test_validate_source_rejects_m2_2_admission_regressions(
    mutate, expected: str
) -> None:
    warmups = [_row(mode, 0, phase="warmup") for mode in MODES]
    measured = [_row(mode, repeat) for mode in MODES for repeat in range(5)]
    mutate(measured)
    _, context = validate_source(warmups, measured)
    assert any(expected in error for error in context["errors"])


def test_generate_report_rejects_output_outside_comparison_root(tmp_path: Path) -> None:
    with pytest.raises(ReportError, match="under comparison root"):
        generate_report(
            _write_run(tmp_path / "run"),
            tmp_path / "outside",
            comparison_root=tmp_path / "comparison",
        )


def test_pairing_is_exactly_five_repeats(tmp_path: Path) -> None:
    run = _write_run(tmp_path / "run")
    records = [
        json.loads(line)
        for line in (run / "normalized_records.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    records = [
        row
        for row in records
        if not (row["numeric_mode"] == "none" and row["repeat_id"] == 4)
    ]
    (run / "normalized_records.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in records), encoding="utf-8"
    )
    with pytest.raises(ReportError, match="source rejected"):
        generate_report(run, tmp_path / "report", comparison_root=tmp_path)


@pytest.mark.parametrize(
    ("updates", "expected"),
    [
        ({"fixture_version": "wrong"}, "fixture_version_expected"),
        ({"operation_count": 3}, "operation_count_expected"),
        ({"completed_task_count": 3}, "completed_task_count_expected"),
        ({"case_id": "other_case"}, "single_case_id"),
        ({"workload_id": "other_workload"}, "single_workload_id"),
        ({"total_time_s": 0.0}, "positive_finite_total_time_s"),
        ({"stage_timings": {}}, "positive_finite_stage_timings.total_route_time_s"),
        (
            {"full_precision_max_abs_error": None},
            "finite_nonnegative_full_precision_max_abs_error",
        ),
        ({"timing_scope": "kernel_only"}, "timing_scope"),
    ],
)
def test_validate_source_rejects_m2_2_contract_regressions(
    updates: dict[str, object], expected: str
) -> None:
    warmups = [_row(mode, 0, phase="warmup") for mode in MODES]
    measured = [_row(mode, repeat) for mode in MODES for repeat in range(5)]
    measured[0].update(updates)
    _, context = validate_source(warmups, measured)
    assert any(expected in error for error in context["errors"])


def test_report_preserves_required_provenance(tmp_path: Path) -> None:
    output = generate_report(
        _write_run(tmp_path / "run"), tmp_path / "report", comparison_root=tmp_path
    )
    manifest = json.loads((output / "report_manifest.json").read_text())
    assert "environment.json" in manifest["source_hashes"]
    assert "config/resolved_suite.yml" in manifest["source_hashes"]
    assert "config/hardware_profile.json" in manifest["source_hashes"]
    assert manifest["provenance"]["hostname"] == "test-host"
    assert manifest["provenance"]["sdk_metadata"]["dpu-upmem-dpurte-clang"]
    assert manifest["provenance"]["worktree_state"] == {
        "benchmark_source_worktree_dirty": False
    }
    assert (
        manifest["provenance"]["hardware_profile"]["hardware_profile_version"]
        == "hardware_sliced_resident_two_dpu_m2_v1"
    )
    assert manifest["provenance"]["hardware_profile_sha256"]
    assert manifest["provenance"]["row_hashes"] == {}


@pytest.mark.parametrize(
    ("phase", "field"),
    [
        ("warmup", "fixture_version"),
        ("measured", "fixture_scope"),
        ("measured", "operation_count"),
        ("warmup", "completed_slice_count"),
        ("measured", "source_task_completion_count"),
        ("measured", "slice_model_executed_task_count"),
        ("measured", "duplicate_contraction_check"),
        ("measured", "dependency_violation_detected"),
    ],
)
def test_required_fixture_task_fields_cannot_be_omitted(phase: str, field: str) -> None:
    warmups = [_row(mode, 0, phase="warmup") for mode in MODES]
    measured = [_row(mode, repeat) for mode in MODES for repeat in range(5)]
    rows = warmups if phase == "warmup" else measured
    del rows[0][field]
    _, context = validate_source(warmups, measured)
    assert f"missing_{field}" in context["errors"]


@pytest.mark.parametrize(
    "relative_path",
    ["environment.json", "config/resolved_suite.yml", "config/hardware_profile.json"],
)
def test_required_provenance_artifact_cannot_be_omitted(
    tmp_path: Path, relative_path: str
) -> None:
    run = _write_run(tmp_path / "run")
    (run / relative_path).unlink()
    with pytest.raises(ReportError, match="missing M2.2 source artifacts"):
        generate_report(run, tmp_path / "report", comparison_root=tmp_path)


@pytest.mark.parametrize("mutation", ["worktree", "sdk", "hardware_profile"])
def test_required_provenance_metadata_cannot_be_omitted_or_changed(
    tmp_path: Path, mutation: str
) -> None:
    run = _write_run(tmp_path / "run")
    if mutation == "worktree":
        path = run / "run_manifest.json"
        data = json.loads(path.read_text())
        del data["benchmark_source_worktree_dirty"]
    elif mutation == "sdk":
        path = run / "upmem_hardware_sliced_resident_mvp_summary.json"
        data = json.loads(path.read_text())
        del data["native_build"]
    else:
        path = run / "config" / "hardware_profile.json"
        data = json.loads(path.read_text())
        data["requested_dpu_count"] = 4
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ReportError, match="source rejected"):
        generate_report(run, tmp_path / "report", comparison_root=tmp_path)


def test_current_m2_2_evidence_report_is_admitted(tmp_path: Path) -> None:
    source = (
        Path(__file__).parents[1]
        / "runs"
        / "inbox"
        / "eth"
        / "m2_2"
        / "2026-08-03_10-17-21"
    )
    if not source.is_dir():
        pytest.skip("copied ETH M2.2 evidence is not present")
    output = generate_report(source, tmp_path / "report", comparison_root=tmp_path)
    manifest = json.loads((output / "report_manifest.json").read_text())
    assert manifest["status"] == "valid"
    assert manifest["counts"] == {"warmups": 2, "measured": 10, "measured_per_mode": 5}
    assert "environment.json" in manifest["source_hashes"]
    assert "config/resolved_suite.yml" in manifest["source_hashes"]
    assert "config/hardware_profile.json" in manifest["source_hashes"]
    assert manifest["provenance"]["row_hashes"]["host_binary_hash"]
