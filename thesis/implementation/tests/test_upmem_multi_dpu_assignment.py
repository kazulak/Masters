from __future__ import annotations

import csv
import json
from pathlib import Path

from quantum_bench.bench.result_artifacts import compare_results, load_result_records
from quantum_bench.bench.upmem_multi_dpu_assignment import run_upmem_multi_dpu_assignment


def _suite(path: Path) -> Path:
    path.write_text(
        """
schema_version: 2
suite_id: upmem_assignment_test
defaults:
  planner: {engine: opt_einsum, optimize: greedy}
workloads:
  - id: bell_2q
    circuit: {kind: builtin, name: bell_2q}
  - id: ghz_4q
    circuit: {kind: builtin, name: ghz_chain, n_qubits: 4}
routes:
  - id: upmem_tn_sdk_simulator_quantized
    required: false
validation: {}
""",
        encoding="utf-8",
    )
    return path


def _flatten_assignments(plan: dict) -> list[dict]:
    return [
        assignment
        for case in plan["cases"]
        for wave in case["frontier_waves"]
        for assignment in wave["assignments"]
    ]


def test_upmem_multi_dpu_assignment_writes_modeled_evidence(tmp_path: Path) -> None:
    result = run_upmem_multi_dpu_assignment(
        tmp_path,
        suite_path=_suite(tmp_path / "suite.yml"),
        dpu_group_count=2,
        strategy="frontier_round_robin_dpu_groups",
    )
    plan = json.loads(result.plan_path.read_text(encoding="utf-8"))
    records = load_result_records([result.run_dir])

    assert result.status == "completed"
    assert result.case_count == 2
    assert result.run_dir.parent == tmp_path / "runs" / "evidence" / "upmem_assignment_test" / "upmem_multi_dpu_assignment"
    assert (result.run_dir / "run_manifest.json").exists()
    assert (result.run_dir / "environment.json").exists()
    assert (result.run_dir / "upmem_multi_dpu_assignments.csv").exists()
    assert result.normalized_records_path.exists()
    assert plan["schema_version"] == "upmem_multi_dpu_assignment_v1"
    assert plan["metadata"]["modeled_only"] is True
    assert plan["metadata"]["dpu_programs_executed"] is False
    assert plan["summary"]["assigned_task_count"] == plan["summary"]["task_count"]
    assert plan["summary"]["executed_dpu_task_count"] == 0

    for case in plan["cases"]:
        task_ids = [
            assignment["task_id"]
            for wave in case["frontier_waves"]
            for assignment in wave["assignments"]
        ]
        assert len(task_ids) == case["task_count"]
        assert len(set(task_ids)) == case["task_count"]
        assert case["assigned_task_count"] == case["task_count"]
        assert case["executed_dpu_task_count"] == 0
        assert case["dpu_assignment_validation_status"] == "passed"

    assert len(records) == 2
    for record in records:
        assert record["route_id"] == "upmem_multi_dpu_assignment_model"
        assert record["benchmark_role"] == "modeled_upmem_multi_dpu_assignment"
        assert record["parallelism_mode"] == "modeled_only"
        assert record["parallelism_evidence_type"] == "modeled"
        assert record["execution_plan_executed"] is False
        assert record["upmem_parallelism_mode"] == "frontier_multi_dpu"
        assert record["upmem_parallelism_evidence_type"] == "modeled"
        assert record["dpu_group_count"] == 2
        assert record["assigned_task_count"] == record["task_count"]
        assert record["executed_dpu_task_count"] == 0
        assert record["dpu_program_invocations"] == 0
        assert record["upmem_program_executed"] is False
        assert record["hardware_execution"] is False
        assert record["hardware_speedup_applicable"] is False
        assert record["cpu_fallback_used"] is False
        assert record["validation_status"] == "not_applicable_modeled_only"
        assert record["dpu_assignment_validation_status"] == "passed"

    with (result.run_dir / "upmem_multi_dpu_assignments.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == plan["summary"]["assigned_task_count"]
    assert {int(row["dpu_group_id"]) for row in rows} <= {0, 1}


def test_upmem_multi_dpu_assignment_sequential_strategy_uses_one_group(tmp_path: Path) -> None:
    result = run_upmem_multi_dpu_assignment(
        tmp_path,
        suite_path=_suite(tmp_path / "suite.yml"),
        dpu_group_count=4,
        strategy="sequential_single_dpu",
    )
    plan = json.loads(result.plan_path.read_text(encoding="utf-8"))

    assert {assignment["dpu_group_id"] for assignment in _flatten_assignments(plan)} == {0}
    records = load_result_records([result.run_dir])
    assert {record["upmem_parallelism_mode"] for record in records} == {"sequential"}


def test_upmem_multi_dpu_assignment_compare_results_includes_assignment_fields(tmp_path: Path) -> None:
    result = run_upmem_multi_dpu_assignment(
        tmp_path,
        suite_path=_suite(tmp_path / "suite.yml"),
        dpu_group_count=2,
        strategy="frontier_size_aware_dpu_groups",
    )
    comparison = compare_results([result.run_dir], tmp_path / "comparison", comparison_type="upmem_assignment_model")

    with comparison.csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    assert rows[0]["upmem_parallelism_evidence_type"] == "modeled"
    assert rows[0]["task_assignment_strategy"] == "frontier_size_aware_dpu_groups"
    assert rows[0]["assigned_task_count"]
    assert rows[0]["executed_dpu_task_count"] == "0"
    assert rows[0]["dpu_assignment_validation_status"] == "passed"
    assert (comparison.run_dir / "parallelism_mode_summary.csv").exists()
    assert (comparison.run_dir / "parallelism_comparison_summary.md").exists()

    with (comparison.run_dir / "parallelism_mode_summary.csv").open(encoding="utf-8", newline="") as handle:
        summary_rows = list(csv.DictReader(handle))
    assert summary_rows[0]["route_id"] == "upmem_multi_dpu_assignment_model"
    assert summary_rows[0]["parallelism_mode"] == "modeled_only"
    assert summary_rows[0]["upmem_parallelism_evidence_type"] == "modeled"
    assert summary_rows[0]["same_family_timing_group"] == "upmem_sdk"
    assert summary_rows[0]["assigned_task_count"]
    assert summary_rows[0]["executed_dpu_task_count"] == "0"
    assert summary_rows[0]["hardware_speedup_applicable"] == "False"

    summary_md = (comparison.run_dir / "parallelism_comparison_summary.md").read_text(encoding="utf-8")
    assert "UPMEM modeled assignment" in summary_md
    assert "modeled_upmem_assignment_not_executed" in summary_md
    assert "| upmem_multi_dpu_assignment_model |" in summary_md
    assert f"| {summary_rows[0]['assigned_task_count']} | 0 |" in summary_md
