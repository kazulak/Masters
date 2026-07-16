from __future__ import annotations

import json
from pathlib import Path

import pytest

import quantum_bench.bench.upmem_hardware_taskgraph as benchmark


ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT / "configs/suites/upmem_hardware_taskgraph_correctness.yml"


def test_prepare_only_works_on_hardware_taskgraph_suite_without_build(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def build_must_not_run(*args: object, **kwargs: object) -> object:
        raise AssertionError("prepare-only must not build native session")

    monkeypatch.setattr(benchmark, "build_hardware_session", build_must_not_run)
    result = benchmark.prepare_upmem_hardware_taskgraph(
        tmp_path,
        suite_path=SUITE,
        build=False,
        environment={"DPU_BACKEND": "simulator"},
    )

    assert result.status == "prepared"
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert summary["native_build"] == {"attempted": False, "status": "not_requested"}
    assert summary["dpu_allocation_attempted"] is False
    assert summary["dpu_launch_attempted"] is False
    assert len(summary["prepared_cases"]) == 3
    assert all(case["status"] == "prepared" for case in summary["prepared_cases"])
