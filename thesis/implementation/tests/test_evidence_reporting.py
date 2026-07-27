"""Active evidence normalization, reporting, and claim-boundary contracts."""

from __future__ import annotations

import csv
import json
import warnings
from pathlib import Path

import pytest

import scripts.research_benchmark_pack as report_pack_module
from quantum_bench.bench.reporting import report_run
from quantum_bench.bench.result_artifacts import compare_results

from .support import (
    cpu_gpu_pair_records,
    hardware_evidence_records,
    planner_evidence_records,
    record_with_updates,
    tn_upmem_pair_records,
    write_evidence_run,
)


def _tn_comparison_records() -> list[dict[str, object]]:
    base = cpu_gpu_pair_records()[0]
    return [
        record_with_updates(
            base,
            suite_id="thesis_cpu_tn_quimb",
            route_id="quest_cpu_full_state_exact",
            execution_target="cpu",
            contraction_execution_target="cpu",
            benchmark_role="serious_full_state_baseline",
            simulation_compute_time_s=2.0,
            total_wall_time_s=2.2,
        ),
        record_with_updates(
            base,
            suite_id="thesis_cpu_tn_quimb",
            route_id="quimb_tn_exact",
            execution_target="cpu",
            contraction_execution_target="cpu",
            backend_family="quimb",
            benchmark_role="serious_external_tn_baseline",
            simulation_compute_time_s=1.0,
            total_wall_time_s=1.1,
        ),
    ]


def test_route_stats_and_pairing_use_normalized_records() -> None:
    records = cpu_gpu_pair_records()
    stats = report_pack_module.per_case_route_stats(records)
    pairs = report_pack_module.paired_speedups(records)

    assert len(stats) == 4
    assert len(pairs) == 4
    assert {row["gpu_route_id"] for row in pairs} == {"quest_gpu_full_state_exact"}
    assert all(row["compute_speedup_cpu_over_gpu"] > 1.0 for row in pairs)


def test_gpu_claim_requires_verified_same_case_execution() -> None:
    records = cpu_gpu_pair_records()
    unverified = record_with_updates(records[1], gpu_backend_verified=False)
    mismatched = record_with_updates(records[1], case_id="different_case")

    assert report_pack_module.paired_speedups([records[0], unverified]) == []
    assert report_pack_module.paired_speedups([records[0], mismatched]) == []


def test_quest_vs_quimb_is_reported_as_ratio() -> None:
    rows = report_pack_module.per_case_route_stats(_tn_comparison_records())
    comparison = report_pack_module.full_state_tn_comparison(rows)

    assert len(comparison) == 1
    assert comparison[0]["quimb_unsliced_time_over_quest_time"] == pytest.approx(0.5)
    assert "speedup" not in " ".join(comparison[0])


def test_same_plan_execution_requires_a_matching_plan_hash() -> None:
    records = tn_upmem_pair_records()
    same_plan = report_pack_module.same_plan_execution(records)
    mismatched = record_with_updates(records[1], contraction_plan_hash="different-plan")

    assert len(same_plan) == 2
    assert {row["contraction_plan_hash"] for row in same_plan} == {
        records[0]["contraction_plan_hash"]
    }
    assert report_pack_module.same_plan_execution([records[0], mismatched]) == []


def test_unsupported_rows_remain_visible_in_quantization_attribution() -> None:
    records = tn_upmem_pair_records()
    rows = report_pack_module.upmem_quantization_attribution(records)
    unsupported = report_pack_module.unsupported_cases(records)

    assert len(rows) == 1
    assert rows[0]["quantized_max_abs_error_vs_full_precision"] == pytest.approx(0.01)
    assert len(unsupported) == 1
    assert unsupported[0]["resource_skip_reason"] == "rank_cap_exceeded"


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("hardware_speedup_applicable", True, "hardware speedup"),
        ("cpu_fallback_used", True, "CPU fallback"),
        ("actual_h2d_bytes", 76, "transfer"),
    ],
)
def test_claim_guards_reject_unsafe_upmem_evidence(
    field: str, value: object, message: str
) -> None:
    record = record_with_updates(tn_upmem_pair_records()[1], **{field: value})

    issues = report_pack_module._claim_guard_issues([record])

    assert any(message.lower() in issue.lower() for issue in issues)


def test_planner_report_rejects_mixed_semantics() -> None:
    records = planner_evidence_records()
    mixed = record_with_updates(records[1], pim_objective_version="upmem_path_cost_v1")

    context = report_pack_module.planner_semantic_context(mixed and [records[0], mixed])

    assert context["issues"]
    assert "pim_objective_version" in " ".join(context["issues"])


def test_physical_report_does_not_promote_incomplete_rows() -> None:
    row = hardware_evidence_records()[0]
    incomplete = record_with_updates(row, hardware_release_verified=False)

    assert report_pack_module.upmem_one_dpu_runtime_summary([incomplete]) == []
    assert report_pack_module.upmem_physical_taskgraph_breakdown([incomplete]) == []


@pytest.mark.parametrize("nested_field", ["policy_reference_validation", "full_precision_accuracy"])
def test_physical_report_rejects_contradictory_nested_accuracy(nested_field: str) -> None:
    row = record_with_updates(
        hardware_evidence_records()[0],
        validation_status="passed",
        **{nested_field: {"status": "failed", "passed": False}},
    )

    assert report_pack_module._physical_taskgraph_issues(row)
    assert report_pack_module.upmem_physical_taskgraph_breakdown([row]) == []
    assert report_pack_module.upmem_one_dpu_runtime_summary([row]) == []


def test_physical_report_rejects_completely_absent_transfer_evidence() -> None:
    row = record_with_updates(
        hardware_evidence_records()[0],
        suite_id=report_pack_module.RESIDENT_SUITE_ID,
        actual_h2d_bytes=None,
        actual_d2h_bytes=None,
        actual_transfer_bytes=None,
        actual_transfer_bytes_invariant=None,
    )

    assert any("transfer" in issue for issue in report_pack_module._physical_taskgraph_issues(row))
    assert report_pack_module.upmem_physical_taskgraph_breakdown([row]) == []
    assert report_pack_module.upmem_one_dpu_runtime_summary([row]) == []


def test_legacy_snapshot_reader_preserves_rows_without_new_evidence_fields() -> None:
    row = record_with_updates(
        hardware_evidence_records()[0],
        suite_id="legacy_physical_taskgraph",
        legacy_snapshot_reader=True,
        actual_h2d_bytes=None,
        actual_d2h_bytes=None,
        actual_transfer_bytes=None,
        actual_transfer_bytes_invariant=None,
    )
    row.pop("policy_reference_validation")
    row.pop("full_precision_accuracy")

    assert report_pack_module._physical_taskgraph_issues(row) == []
    assert len(report_pack_module.upmem_physical_taskgraph_breakdown([row])) == 1
    assert len(report_pack_module.upmem_one_dpu_runtime_summary([row])) == 1


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("target_requested", "cpu"),
        ("target_observed", "cpu"),
        ("requested_dpu_count", 2),
        ("allocated_dpu_count", 2),
        ("tasklets_per_dpu", 2),
        ("allocation_count", 2),
        ("hardware_allocation_verified", False),
        ("native_execution", False),
        ("native_hardware_backend", False),
        ("hardware_execution", False),
        ("hardware_kernel_executed", False),
        ("simulator_kernel_executed", True),
        ("cpu_fallback_used", True),
        ("hardware_release_verified", False),
        ("release_confirmed", False),
        ("physical_dependency_chain_verified", False),
        ("validation_status", "failed"),
        ("task_count", 0),
        ("validated_task_count", 0),
        ("hardware_timing_available", False),
        ("timing_is_bringup_only", True),
        ("steady_state_graph_execution_s", None),
        ("actual_transfer_bytes", 95),
        ("actual_transfer_bytes_invariant", "failed"),
        ("intermediate_h2d_bytes", 8),
    ],
)
def test_physical_taskgraph_guard_rejects_each_unsafe_provenance_flag(
    field: str, value: object
) -> None:
    row = record_with_updates(hardware_evidence_records()[0], **{field: value})

    assert report_pack_module._physical_taskgraph_issues(row)
    assert report_pack_module.upmem_physical_taskgraph_breakdown([row]) == []


def test_resident_evidence_check_applies_physical_guard(tmp_path: Path) -> None:
    suite_id = report_pack_module.RESIDENT_SUITE_ID
    row = record_with_updates(
        hardware_evidence_records()[0],
        suite_id=suite_id,
        route_id=report_pack_module.RESIDENT_ROUTE_ID,
    )
    run = write_evidence_run(tmp_path, [row], suite_id=suite_id)

    assert report_pack_module.check_resident_evidence(run) == 0
    unsafe = record_with_updates(row, release_confirmed=False)
    write_evidence_run(tmp_path / "unsafe", [unsafe], suite_id=suite_id)

    with pytest.raises(ValueError, match="physical guard"):
        report_pack_module.check_resident_evidence(tmp_path / "unsafe" / "run")


def test_claim_guards_keep_physical_route_diagnostic_only() -> None:
    row = hardware_evidence_records()[0]
    issues = report_pack_module._claim_guard_issues(
        [record_with_updates(row, hardware_speedup_applicable=True)]
    )

    assert any("hardware speedup" in issue.lower() for issue in issues)


def test_compare_results_writes_normalized_schema_and_respects_output_boundary(
    tmp_path: Path,
) -> None:
    run = write_evidence_run(tmp_path, cpu_gpu_pair_records())
    output = tmp_path / "runs" / "comparisons" / "compact"

    result = compare_results([run], output, root_dir=tmp_path)

    payload = json.loads(result.artifact_path.read_text(encoding="utf-8"))
    assert payload["record_count"] == 8
    assert payload["metadata"]["simulator_timings_are_not_hardware_speedups"] is True
    with result.csv_path.open(newline="", encoding="utf-8") as handle:
        assert "route_id" in next(csv.DictReader(handle))
    with pytest.raises(ValueError, match="runs/evidence"):
        compare_results([run], tmp_path / "runs" / "evidence" / "bad", root_dir=tmp_path)


def test_report_run_keeps_input_evidence_readable(tmp_path: Path) -> None:
    run = write_evidence_run(tmp_path, cpu_gpu_pair_records())
    before = (run / "normalized_records.jsonl").read_text(encoding="utf-8")

    report_run(run, tmp_path / "report", output_plots=False, root_dir=tmp_path)

    manifest = json.loads(
        (tmp_path / "report" / "report_run.json").read_text(encoding="utf-8")
    )
    assert manifest["record_count"] == 8
    assert (run / "normalized_records.jsonl").read_text(encoding="utf-8") == before


def test_generic_validation_aggregation_uses_scientific_status_only_for_active_resident_route(
    tmp_path: Path,
) -> None:
    active = record_with_updates(
        hardware_evidence_records()[0],
        scientific_validation_status="failed",
        upmem_parallelism_evidence_type="hardware_executed",
    )
    legacy = record_with_updates(
        cpu_gpu_pair_records()[0],
        scientific_validation_status="failed",
    )
    run = write_evidence_run(tmp_path, [active, legacy])

    report_run(run, tmp_path / "report", output_plots=False, root_dir=tmp_path)
    validation = json.loads(
        (tmp_path / "report" / "validation" / "validation_summary.json").read_text(encoding="utf-8")
    )
    assert validation["passed_count"] == 1
    assert validation["failed_count"] == 1
    failures = (tmp_path / "report" / "validation" / "validation_failures.jsonl").read_text(encoding="utf-8")
    assert active["case_id"] in failures
    assert legacy["case_id"] not in failures

    comparison = compare_results([run], tmp_path / "comparison", root_dir=tmp_path)
    payload = json.loads(comparison.artifact_path.read_text(encoding="utf-8"))
    resident_summary = next(
        row for row in payload["parallelism_mode_summary"] if row["route_id"] == active["route_id"]
    )
    assert resident_summary["validation_passed_count"] == 0


def test_report_pack_agg_render_has_readable_correlated_outputs(tmp_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.image as mpimg

    records = cpu_gpu_pair_records() + tn_upmem_pair_records() + planner_evidence_records()
    run = write_evidence_run(tmp_path, records)
    output = tmp_path / "report"

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        assert report_pack_module.report_pack(
            tmp_path, output, inputs=[run], suite_filter=None
        ) == 0

    manifest = json.loads((output / "plot_manifest.json").read_text(encoding="utf-8"))
    entries = manifest["plots"]
    assert entries
    assert any(entry["status"] == "generated_valid" for entry in entries)
    for entry in entries:
        if not str(entry["status"]).startswith("generated"):
            continue
        plot = output / "plots" / str(entry["plot"])
        source_csv = output / str(entry["source_csv"])
        assert plot.is_file() and plot.stat().st_size > 1000
        assert source_csv.is_file()
        assert mpimg.imread(plot).size > 0
