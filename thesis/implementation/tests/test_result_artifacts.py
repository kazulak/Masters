from __future__ import annotations

import csv
import json
from pathlib import Path

from quantum_bench.bench.generic_task_bridge import run_generic_task_bridge
from quantum_bench.bench.result_artifacts import KERNEL_FAMILIES, compare_results, load_result_records


def test_kernel_family_vocabulary_contains_generic_and_dense() -> None:
    assert "dense_gemm" in KERNEL_FAMILIES
    assert "einsum_contraction" in KERNEL_FAMILIES
    assert "full_state_vector" in KERNEL_FAMILIES
    assert "generic_loop_fallback" in KERNEL_FAMILIES
    assert "unsupported" in KERNEL_FAMILIES


def test_compare_results_loads_generic_summary_directory(tmp_path: Path) -> None:
    bridge = run_generic_task_bridge(tmp_path / "runs", case="bell_2q", task_index=0, execute_external=False)

    records = load_result_records([bridge.run_dir])
    assert len(records) == 1
    assert records[0]["kernel_family"] == "generic_loop_fallback"
    assert records[0]["execution_scope"] == "task_level"
    assert records[0]["hardware_speedup"] == "not_applicable"

    out_dir = tmp_path / "comparison"
    result = compare_results([bridge.run_dir], out_dir)
    payload = json.loads(result.artifact_path.read_text(encoding="utf-8"))
    summary_md = result.summary_path.read_text(encoding="utf-8")

    assert result.record_count == 1
    assert payload["schema_version"] == "compare_results_v1"
    assert payload["records"][0]["kernel_family"] == "generic_loop_fallback"
    assert payload["metadata"]["simulator_timings_are_not_hardware_speedups"] is True
    assert "Simulator timings are task-level development evidence" in summary_md
    assert (out_dir / "kernel_family_summary.csv").exists()

    with result.csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["kernel_family"] == "generic_loop_fallback"
    assert rows[0]["hardware_speedup"] == "not_applicable"


def test_compare_results_fails_gracefully_without_compatible_artifacts(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()

    try:
        compare_results([empty], tmp_path / "out")
    except ValueError as exc:
        assert "no compatible benchmark result artifacts found" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("compare_results should reject directories with no compatible artifacts")
