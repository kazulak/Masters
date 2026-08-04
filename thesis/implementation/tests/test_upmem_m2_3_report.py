from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.image as mpimg
import pytest

from quantum_bench.bench.upmem_hardware_sliced_resident_mvp import (
    M2_3_EXECUTION_ROUTE_POLICY,
    M2_3_NATIVE_HARDWARE_PROFILE_VERSION,
    M2_3_PLANNER_CANDIDATE_EVIDENCE_TYPE,
    M2_3_PLANNER_ROUTE_RELATION,
    M2_3_SCHEMA_VERSION,
)
from scripts.upmem_m2_3_report import (
    CANONICAL_WORKLOAD_IDS,
    EXPECTED_EXECUTION_ROUTE_POLICY,
    EXPECTED_MODES,
    EXPECTED_NATIVE_PROFILE,
    EXPECTED_PATHS,
    EXPECTED_PLANNER_EVIDENCE,
    EXPECTED_PLANNER_ROUTE_RELATION,
    EXPECTED_SCHEMA,
    ReportError,
    generate_report,
    validate_source,
)


def test_report_contract_constants_match_current_runner_schema() -> None:
    assert EXPECTED_SCHEMA == M2_3_SCHEMA_VERSION
    assert EXPECTED_NATIVE_PROFILE == M2_3_NATIVE_HARDWARE_PROFILE_VERSION
    assert EXPECTED_PLANNER_EVIDENCE == M2_3_PLANNER_CANDIDATE_EVIDENCE_TYPE
    assert EXPECTED_EXECUTION_ROUTE_POLICY == M2_3_EXECUTION_ROUTE_POLICY
    assert EXPECTED_PLANNER_ROUTE_RELATION == M2_3_PLANNER_ROUTE_RELATION
    assert set(CANONICAL_WORKLOAD_IDS.values()) == {
        "m2_3_ry_h_ry_a_opt_einsum_greedy",
        "m2_3_ry_h_ry_a_custom_upmem_v2_balanced",
        "m2_3_ry_h_ry_b_opt_einsum_greedy",
        "m2_3_ry_h_ry_b_custom_upmem_v2_balanced",
    }


def _row(
    fixture: str,
    path: str,
    mode: str,
    phase: str,
    repeat: int,
    **updates,
) -> dict:
    plan = "plan-greedy" if path == "opt_einsum_greedy" else "plan-custom"
    row = {
        "schema_version": "upmem_hardware_sliced_resident_mvp_record_v1",
        "experiment_schema_version": "upmem_hardware_sliced_resident_m2_3_v1",
        "status": "completed",
        "suite_id": "upmem_hardware_sliced_resident_m2_3",
        "case_id": f"m2_3_{fixture}_{path}",
        "workload_id": f"m2_3_{fixture}_{path}",
        "fixture_id": fixture,
        "path_variant_id": path,
        "phase": phase,
        "repeat_id": repeat,
        "numeric_mode": mode,
        "quantization_mode": mode,
        "response_numeric_mode": mode,
        "route_id": "upmem_tn_hardware_sliced_resident_two_dpu",
        "backend_id": "upmem_sdk_hardware_sliced_resident_two_dpu",
        "backend_family": "upmem_sdk",
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
        "hardware_speedup_applicable": False,
        "performance_claim_applicable": False,
        "energy_measurement_available": False,
        "fixture_version": "upmem_hardware_sliced_resident_m2_3_v1",
        "fixture_scope": "three_operation_ry_h_ry_full_graph_replicated_prefix",
        "execution_scope": "physical_two_dpu_two_slice_full_replicated_prefix_taskgraph",
        "experiment_profile_version": "hardware_sliced_resident_two_dpu_m2_3_v1",
        "hardware_profile_version": "hardware_sliced_resident_two_dpu_m2_3_v1",
        "native_session_profile_version": "hardware_sliced_resident_two_dpu_m2_v1",
        "operation_count": 3,
        "source_task_count": 3,
        "source_task_completion_count": 3,
        "source_task_completion_scope": (
            "unique_source_tasks_completed_on_every_slice"
        ),
        "expanded_task_count": 6,
        "expanded_task_completion_count": 6,
        "expanded_task_count_scope": (
            "physical_source_operation_instances_across_two_slices"
        ),
        "executed_task_count": 6,
        "completed_task_count": 6,
        "executed_task_count_scope": (
            "compatibility_alias_for_expanded_task_completion_count"
        ),
        "completed_task_count_scope": (
            "compatibility_alias_for_expanded_task_completion_count"
        ),
        "slice_model_task_count": 2,
        "slice_model_operation_count": 3,
        "slice_model_executed_task_count": 2,
        "slice_descriptor_count": 2,
        "slice_descriptor_completion_count": 2,
        "slice_model_task_count_scope": "slice_descriptors",
        "slice_model_executed_task_count_scope": "completed_slice_descriptors",
        "slice_model_operation_count_scope": ("source_operations_replicated_per_slice"),
        "expanded_physical_operation_count": 6,
        "expanded_physical_operation_completion_count": 6,
        "operations_per_slice": 3,
        "slice_count": 2,
        "source_slice_count": 2,
        "executed_slice_count": 2,
        "completed_slice_count": 2,
        "slice_ids": [0, 1],
        "observed_operation_completion_counts": [3, 3],
        "completed_operation_count_per_slice": [3, 3],
        "completed_physical_task_instance_count": 6,
        "expected_physical_task_instance_count": 6,
        "physical_task_instances_per_slice": [3, 3],
        "device_completion_confirmed": True,
        "native_execution_sentinel_available": True,
        "completion_sentinel_read_counts": [3, 3],
        "native_cleanup_confirmed": True,
        "allocation_evidence": {
            "verified": True,
            "requested_dpus": 2,
            "allocated_dpus": 2,
            "profile": "backend=hw",
        },
        "launch_evidence": {
            "completed": True,
            "mode": "asynchronous",
            "operation_count": 3,
            "async_launch_count": 2,
            "synchronize_count": 2,
            "device_launch_mode": "asynchronous_dpu_set",
            "host_completion_mode": "blocking_sync",
        },
        "release_evidence": {"confirmed": True},
        "execution_contract_status": "passed",
        "slice_package_validation_status": "passed",
        "per_slice_output_validation_status": "passed",
        "reconstruction_validation_status": "passed",
        "final_output_validation_status": "passed",
        "policy_reference_status": "passed",
        "full_precision_accuracy_status": "passed",
        "scientific_validation_status": "passed",
        "validation_status": "passed",
        "expected_output_validation": True,
        "strict_cpu_reference_validation": mode == "none",
        "transfer_accounting_invariant": True,
        "transfer_matches_manifest_plan": True,
        "validation_errors": [],
        "timing_is_bringup_only": True,
        "timing_scope": "host_observed_sdk_process_wall_and_blocking_sync",
        "timing_breakdown_status": "host_stage_boundaries",
        "total_time_s": 0.10 + repeat / 1000,
        "process_time_s": 0.06 + repeat / 1000,
        "reconstruction_time_s": 0.02,
        "stage_timings": {
            "total_route_time_s": 0.05 + repeat / 10000,
            "sync_wait_time_s": 0.001,
            "status": "host_stage_boundaries",
            "kernel_time_s": None,
            "sync_wait_is_not_pure_kernel_time": True,
        },
        "actual_h2d_bytes": 3408,
        "actual_d2h_bytes": 16,
        "actual_transfer_bytes": 3424,
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
        "policy_reference_max_abs_error": 0.0 if mode == "none" else 0.001,
        "full_precision_max_abs_error": 1.0e-8 if mode == "none" else 0.002,
        "circuit_semantics_hash": f"circuit-{fixture}",
        "tensor_network_hash": f"network-{fixture}",
        "contraction_plan_hash": plan,
        "executor_config_hash": f"executor-{mode}",
        "planner_path": (
            [[0, 1], [0, 1], [0, 1]]
            if path == "opt_einsum_greedy"
            else [[0, 1], [0, 2], [0, 1]]
        ),
        "expected_path": (
            [[0, 1], [0, 1], [0, 1]]
            if path == "opt_einsum_greedy"
            else [[0, 1], [0, 2], [0, 1]]
        ),
        "planner_path_matches_expected": True,
        "planner_candidate_evidence_type": "modeled",
        "execution_route_policy": "hardware_sliced_resident_two_dpu_m2_3_v1",
        "planner_policy_matches_execution_route": False,
        "planner_route_relation": (
            "fixed_modeled_candidate_path_executed_on_different_two_dpu_"
            "sliced_resident_route"
        ),
        "planner_engine": "opt_einsum"
        if path == "opt_einsum_greedy"
        else "custom_upmem",
        "planner_config": (
            {"engine": "opt_einsum", "optimize": "greedy"}
            if path == "opt_einsum_greedy"
            else {
                "engine": "custom_upmem",
                "algorithm": "greedy",
                "objective_version": "upmem_path_cost_v2",
                "selection_scope": "projected_prefix",
                "weight_profile": "balanced_literature_informed",
                "normalization": "fixed_log1p_generic_budgets_v2",
                "execution_policy": "generic_single_dpu_split_complex_v2",
            }
        ),
        "planner_objective_version": (
            None if path == "opt_einsum_greedy" else "upmem_path_cost_v2"
        ),
        "planner_selection_scope": (
            None if path == "opt_einsum_greedy" else "projected_prefix"
        ),
        "planner_weight_profile": (
            None if path == "opt_einsum_greedy" else "balanced_literature_informed"
        ),
        "planner_profile": (
            None if path == "opt_einsum_greedy" else "balanced_literature_informed"
        ),
        "planner_normalization": (
            None if path == "opt_einsum_greedy" else "fixed_log1p_generic_budgets_v2"
        ),
        "planner_execution_policy": (
            None
            if path == "opt_einsum_greedy"
            else "generic_single_dpu_split_complex_v2"
        ),
        "host_binary_hash": "host-hash",
        "dpu_binary_hash": "dpu-hash",
        "native_source_tree_hash": "source-tree-hash",
        "binary_source_tree_hash": "source-tree-hash",
    }
    row.update(updates)
    return row


def _source_run(root: Path, *, update: dict | None = None) -> Path:
    root.mkdir(parents=True)
    warmups = [
        _row(fixture, path, mode, "warmup", 0)
        for fixture in ("ry_h_ry_a", "ry_h_ry_b")
        for path in EXPECTED_PATHS
        for mode in EXPECTED_MODES
    ]
    measured = [
        _row(fixture, path, mode, "measured", repeat)
        for fixture in ("ry_h_ry_a", "ry_h_ry_b")
        for path in EXPECTED_PATHS
        for mode in EXPECTED_MODES
        for repeat in range(5)
    ]
    if update:
        target = update.pop("target", "measured")
        index = update.pop("index", 0)
        (warmups if target == "warmup" else measured)[index].update(update)
    (root / "normalized_records.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in measured), encoding="utf-8"
    )
    (root / "warmups.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in warmups), encoding="utf-8"
    )
    (root / "run_manifest.json").write_text(
        json.dumps(
            {
                "suite_id": "upmem_hardware_sliced_resident_m2_3",
                "fixture_version": "upmem_hardware_sliced_resident_m2_3_v1",
                "fixture_scope": "three_operation_ry_h_ry_full_graph_replicated_prefix",
                "execution_scope": "physical_two_dpu_two_slice_full_replicated_prefix_taskgraph",
                "route_id": "upmem_tn_hardware_sliced_resident_two_dpu",
                "backend_id": "upmem_sdk_hardware_sliced_resident_two_dpu",
                "requested_dpu_count": 2,
                "tasklets_per_dpu": 1,
                "numeric_modes": list(EXPECTED_MODES),
                "operation_count": 3,
                "source_task_count": 3,
                "benchmark_source_commit": "a" * 40,
                "benchmark_source_worktree_dirty": False,
            }
        ),
        encoding="utf-8",
    )
    (root / "environment.json").write_text(
        json.dumps(
            {
                "git_commit": "a" * 40,
                "platform": "Linux-5.4.0-test-x86_64-with-glibc2.31",
                "machine": "x86_64",
                "upmem": {
                    "UPMEM_HOME": None,
                    "dpu_compiler": "/usr/bin/dpu-upmem-dpurte-clang",
                },
            }
        ),
        encoding="utf-8",
    )
    (root / "upmem_hardware_sliced_resident_mvp_summary.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "route_id": "upmem_tn_hardware_sliced_resident_two_dpu",
                "backend_id": "upmem_sdk_hardware_sliced_resident_two_dpu",
                "measured_row_count": 40,
                "warmup_count": 8,
                "measured_passed_count": 40,
                "warmup_passed_count": 8,
                "expected_measured_row_count": 40,
                "expected_warmup_row_count": 8,
                "all_required_records_validated": True,
                "fixture_version": "upmem_hardware_sliced_resident_m2_3_v1",
                "experiment_profile_version": "hardware_sliced_resident_two_dpu_m2_3_v1",
                "native_build": {
                    "attempted": True,
                    "status": "passed",
                    "source_tree_hash": "source-tree-hash",
                    "host_binary_hash": "host-hash",
                    "dpu_binary_hash": "dpu-hash",
                    "sdk_tools": {
                        "dpu-pkg-config": "/usr/bin/dpu-pkg-config",
                        "dpu-upmem-dpurte-clang": ("/usr/bin/dpu-upmem-dpurte-clang"),
                        "make": "/usr/bin/make",
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    config = root / "config"
    config.mkdir()
    (config / "resolved_suite.yml").write_text(
        "suite_id: upmem_hardware_sliced_resident_m2_3\n", encoding="utf-8"
    )
    (config / "hardware_profile.json").write_text(
        json.dumps(
            {
                "hardware_profile_version": "hardware_sliced_resident_two_dpu_m2_3_v1",
                "native_session_profile_version": (
                    "hardware_sliced_resident_two_dpu_m2_v1"
                ),
                "target": "hardware",
                "backend_id": "upmem_sdk_hardware_sliced_resident_two_dpu",
                "route_id": "upmem_tn_hardware_sliced_resident_two_dpu",
                "requested_dpu_count": 2,
                "slices": 2,
                "tasklets_per_dpu": 1,
                "numeric_modes": list(EXPECTED_MODES),
                "performance_claim_applicable": False,
            }
        ),
        encoding="utf-8",
    )
    return root


def test_valid_m2_3_report_has_exact_outputs_and_readable_plots(tmp_path: Path) -> None:
    source = _source_run(tmp_path / "source")
    output = generate_report(source, comparison_root=tmp_path / "comparisons")
    assert (output / "combination_statistics.csv").is_file()
    assert (output / "paired_numeric_mode_ratios.csv").is_file()
    assert (output / "paired_path_ratios.csv").is_file()
    statistics = list(csv.DictReader((output / "combination_statistics.csv").open()))
    numeric_pairs = list(
        csv.DictReader((output / "paired_numeric_mode_ratios.csv").open())
    )
    path_pairs = list(csv.DictReader((output / "paired_path_ratios.csv").open()))
    assert len(statistics) == 8
    assert len(numeric_pairs) == 20
    assert len(path_pairs) == 20
    assert "median_native_sdk_stage_time_s" in statistics[0]
    assert "median_python_end_to_end_time_s" in statistics[0]
    assert "native_sdk_stage_timing_ratio" in numeric_pairs[0]
    assert "python_end_to_end_timing_ratio" in numeric_pairs[0]
    for name in (
        "runtime_by_circuit_path_mode.png",
        "quantization_error_by_circuit_path.png",
        "timing_ratios.png",
    ):
        image = mpimg.imread(output / name)
        assert image.shape[0] >= 400
        assert image.shape[1] >= 700
    manifest = json.loads((output / "report_manifest.json").read_text())
    assert manifest["status"] == "completed"
    assert all("source_csv" in plot for plot in manifest["plots"])
    assert manifest["timing_interpretation"]["primary_metric"] == (
        "stage_timings.total_route_time_s"
    )
    assert manifest["provenance"]["hostname"] is None
    assert manifest["provenance"]["missing_provenance_caveats"] == [
        "hostname_unavailable_in_source_environment"
    ]
    summary = (output / "benchmark_summary.md").read_text()
    assert "Claims allowed" in summary
    assert "Claims not allowed" in summary
    assert "not speedup" in summary
    assert "host-observed native SDK stage" in summary


def test_current_runner_schema_fields_are_admitted_and_mapped_directly(
    tmp_path: Path,
) -> None:
    source = _source_run(tmp_path / "source")
    output = generate_report(source, comparison_root=tmp_path / "comparisons")
    validation = list(csv.DictReader((output / "validation_rows.csv").open()))
    first = next(row for row in validation if row["record_type"] == "row")
    assert first["admission_status"] == "passed"
    assert first["source_task_completion_scope"] == (
        "unique_source_tasks_completed_on_every_slice"
    )
    assert first["expanded_task_count"] == "6"
    assert first["expanded_task_completion_count"] == "6"
    assert first["completed_task_count"] == "6"
    assert first["slice_model_task_count"] == "2"
    assert first["slice_model_executed_task_count"] == "2"
    assert first["slice_model_task_count_scope"] == "slice_descriptors"
    assert first["slice_model_executed_task_count_scope"] == (
        "completed_slice_descriptors"
    )
    assert first["allocation_verified"] == "True"
    assert first["launch_completed"] == "True"
    assert first["release_confirmed"] == "True"
    assert first["planner_candidate_evidence_type"] == "modeled"
    assert first["planner_policy_matches_execution_route"] == "False"

    row = _row("ry_h_ry_a", "opt_einsum_greedy", "none", "measured", 0)
    del row["expanded_task_completion_count"]
    validation, _ = validate_source([], [row])
    assert "missing_expanded_task_completion_count" in validation[0]["admission_errors"]
    assert validation[0]["expanded_task_completion_count"] is None
    assert validation[0]["completed_task_count"] == 6


def test_matrix_duplicate_and_missing_rows_are_rejected(tmp_path: Path) -> None:
    source = _source_run(tmp_path / "source")
    rows = json.loads(
        "["
        + ",".join(
            line
            for line in (source / "normalized_records.jsonl").read_text().splitlines()
        )
        + "]"
    )
    rows.pop()
    rows.append(dict(rows[0]))
    rows[-1]["repeat_id"] = rows[0]["repeat_id"]
    validation, context = validate_source(
        json.loads(
            "[" + ",".join((source / "warmups.jsonl").read_text().splitlines()) + "]"
        ),
        rows,
    )
    assert not context["valid"]
    assert "measured_duplicate_matrix_key" in context["errors"]
    assert "measured_matrix_not_exact" in context["errors"]
    assert any(row["record_type"] == "global" for row in validation)


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("target_observed", "simulator", "target_observed_expected"),
        ("timing_scope", "kernel_only", "timing_scope_expected"),
        ("source_task_completion_count", 6, "source_task_completion_count_expected"),
        (
            "source_task_completion_scope",
            "source_graph_operations",
            "source_task_completion_scope_expected",
        ),
        (
            "expanded_task_completion_count",
            5,
            "expanded_task_completion_count_expected",
        ),
        (
            "slice_model_executed_task_count",
            6,
            "slice_model_executed_task_count_expected",
        ),
        ("actual_transfer_bytes", 1, "actual_parts_sum"),
        ("planner_execution_policy", "wrong", "planner_execution_policy_expected"),
        (
            "planner_candidate_evidence_type",
            "executed",
            "planner_candidate_evidence_type_expected",
        ),
        (
            "planner_policy_matches_execution_route",
            True,
            "planner_policy_matches_execution_route_expected",
        ),
        (
            "native_session_profile_version",
            "wrong",
            "native_session_profile_version_expected",
        ),
        ("case_id", "noncanonical", "case_id_expected"),
    ],
)
def test_contract_fields_reject_invalid_rows(
    field: str,
    value: object,
    error: str,
) -> None:
    row = _row(
        "ry_h_ry_a",
        "custom_upmem_v2_balanced",
        "none",
        "measured",
        0,
    )
    row[field] = value
    _, context = validate_source([], [row])
    assert not context["valid"]
    assert any(error in item for item in context["errors"])


@pytest.mark.parametrize(
    ("container", "field", "value", "error"),
    [
        ("allocation_evidence", "verified", False, "allocation_evidence_not_verified"),
        ("launch_evidence", "completed", False, "launch_evidence_not_completed"),
        ("release_evidence", "confirmed", False, "release_not_confirmed"),
    ],
)
def test_physical_evidence_objects_are_required(
    container: str,
    field: str,
    value: object,
    error: str,
) -> None:
    row = _row("ry_h_ry_a", "opt_einsum_greedy", "none", "measured", 0)
    row[container][field] = value
    validation, _ = validate_source([], [row])
    assert error in validation[0]["admission_errors"]


def test_required_provenance_rejects_missing_compiler_and_hash_drift(
    tmp_path: Path,
) -> None:
    source = _source_run(tmp_path / "source")
    environment_path = source / "environment.json"
    environment = json.loads(environment_path.read_text())
    environment["upmem"]["dpu_compiler"] = None
    environment_path.write_text(json.dumps(environment), encoding="utf-8")
    with pytest.raises(ReportError):
        generate_report(source, comparison_root=tmp_path / "comparisons-a")
    rejected = next((tmp_path / "comparisons-a").iterdir())
    validation = (rejected / "validation_rows.csv").read_text()
    assert "missing_environment_upmem_compiler" in validation

    source = _source_run(tmp_path / "source-b")
    rows_path = source / "normalized_records.jsonl"
    rows = [json.loads(line) for line in rows_path.read_text().splitlines()]
    rows[0]["host_binary_hash"] = "different-host-hash"
    rows_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    with pytest.raises(ReportError):
        generate_report(source, comparison_root=tmp_path / "comparisons-b")
    rejected = next((tmp_path / "comparisons-b").iterdir())
    validation = (rejected / "validation_rows.csv").read_text()
    assert "unstable_or_missing_host_binary_hash" in validation


def test_rejection_retains_validation_table_and_has_no_plots(tmp_path: Path) -> None:
    source = _source_run(tmp_path / "source", update={"field": "ignored"})
    (source / "normalized_records.jsonl").write_text("{broken\n", encoding="utf-8")
    with pytest.raises(ReportError):
        generate_report(source, comparison_root=tmp_path / "comparisons")
    output = next((tmp_path / "comparisons").iterdir())
    assert (output / "validation_rows.csv").is_file()
    assert not list(output.glob("*.png"))
    assert (
        json.loads((output / "report_manifest.json").read_text())["status"]
        == "rejected"
    )


def test_output_must_remain_under_comparison_root(tmp_path: Path) -> None:
    source = _source_run(tmp_path / "source")
    with pytest.raises(ReportError):
        generate_report(
            source,
            output_dir=tmp_path / "outside",
            comparison_root=tmp_path / "comparisons",
        )
