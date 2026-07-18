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
    assert report_pack_module.upmem_physical_taskgraph_breakdown([incomplete])


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
    for entry in entries:
        if not str(entry["status"]).startswith("generated"):
            continue
        plot = output / "plots" / str(entry["plot"])
        source_csv = output / str(entry["source_csv"])
        assert plot.is_file() and plot.stat().st_size > 1000
        assert source_csv.is_file()
        assert mpimg.imread(plot).size > 0
