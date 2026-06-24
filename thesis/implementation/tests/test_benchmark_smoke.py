from __future__ import annotations

import json
from pathlib import Path

from quantum_bench.bench.runner import run_suite


ROOT = Path(__file__).resolve().parents[1]


def test_smoke_suite_writes_raw_summary_and_plots_contract(tmp_path: Path) -> None:
    run_dir = run_suite(ROOT / "configs" / "suites" / "smoke.yml", tmp_path)
    raw_rows = []
    for raw in sorted((run_dir / "raw").glob("*.jsonl")):
        raw_rows.extend(json.loads(line) for line in raw.read_text(encoding="utf-8").splitlines())
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert len(raw_rows) == 8
    assert summary["record_count"] == 8
    assert (run_dir / "environment.json").exists()
    assert (run_dir / "metrics" / "metrics.csv").exists()
    assert any(row["status"] == "skipped" and row["route"] == "upmem_dense_int8_placeholder" for row in raw_rows)
    assert any(row["status"] == "passed" and row["route"] == "cpu_tn_einsum_exact" for row in raw_rows)
    for row in raw_rows:
        assert "route" + "_alias" not in row
        assert row["role"]
        assert row["simulation_method"]
        assert row["kernel_family"]
        assert row["hardware_target"]
        assert row["execution_mode"]
        assert row["output_contract"]
        assert row["validation_mode"]
    assert summary["validated_routes"]
    assert summary["skipped_or_probe_routes"]


def test_smoke_v2_suite_runs_with_same_contract(tmp_path: Path) -> None:
    run_dir = run_suite(ROOT / "configs" / "suites" / "smoke_v2.yml", tmp_path)
    raw_rows = []
    for raw in sorted((run_dir / "raw").glob("*.jsonl")):
        raw_rows.extend(json.loads(line) for line in raw.read_text(encoding="utf-8").splitlines())
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert len(raw_rows) == 8
    assert summary["record_count"] == 8
    assert any(row["route"] == "cpu_tn_einsum_exact" and row["role"] == "reference" for row in raw_rows)
    assert any(row["route"] == "upmem_dense_int8_placeholder" and row["role"] == "candidate" for row in raw_rows)
    assert summary["validated_routes"]
