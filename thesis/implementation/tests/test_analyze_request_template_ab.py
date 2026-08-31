from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "analyze_request_template_ab.py"
SPEC = importlib.util.spec_from_file_location("analyze_request_template_ab", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
analyzer = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = analyzer
SPEC.loader.exec_module(analyzer)


def _arm(source: str, scale: float) -> dict[str, object]:
    measurements = {}
    for case_index, case_id in enumerate(analyzer.EXPECTED_CASES, 1):
        for route_index, route_id in enumerate(analyzer.EXPECTED_ROUTES, 1):
            for block in analyzer.MEASUREMENT_BLOCKS:
                values = {
                    component: scale * (case_index + route_index + block)
                    for component in analyzer.COMPONENTS
                }
                values.update(
                    {
                        "payload_record_count": case_index,
                        "payload_files_created": case_index * 2,
                        "payload_bytes_staged": case_index * 8,
                        "payload_bytes_hashed": case_index * 8,
                    }
                )
                measurements[(case_id, route_id, block)] = values
    return {
        "source_commit": source,
        "experiment_id": f"experiment-{source}",
        "run_id": f"run-{source}",
        "measurements": measurements,
    }


def test_paired_summary_is_deterministic_and_descriptive() -> None:
    first = analyzer._paired_summary([4.0, 5.0, 6.0], [2.0, 2.5, 3.0], seed=17)
    second = analyzer._paired_summary([4.0, 5.0, 6.0], [2.0, 2.5, 3.0], seed=17)

    assert first == second
    assert first["descriptive_speedup"] == pytest.approx(2.0)
    assert first["optimized_change_fraction"] == pytest.approx(-0.5)
    assert first["diagnostic_only"] is True
    assert first["bootstrap_resamples"] == 10_000


def test_number_rejects_nonfinite_values() -> None:
    for value in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError, match="finite"):
            analyzer._number(value, "duration")


def _physical_sample(session_id: str, *, hardware_kernel_executed: bool = True) -> dict[str, object]:
    return {
        "status": "success",
        "attempt_kind": "warmup",
        "case_id": analyzer.EXPECTED_CASES[0],
        "route_id": analyzer.EXPECTED_ROUTES[0],
        "block_id": 0,
        "session_instance_id": session_id,
        "measurement": {"scope_id": "steady_execution_v1"},
        "backend_facts": {
            "target_observed": "physical_hardware",
            "simulator_kernel_executed": False,
            "cpu_fallback_used": False,
            "hardware_kernel_executed": hardware_kernel_executed,
            "requested_dpus": 1,
            "allocated_dpus": 1,
            "active_dpus": 1,
            "operation_facts": [
                {
                    "target_observed": "physical_hardware",
                    "simulator_kernel_executed": False,
                    "cpu_fallback_used": False,
                    "hardware_kernel_executed": True,
                    "timing": {},
                }
            ],
        },
    }


def test_physical_validation_requires_top_level_hardware_execution() -> None:
    with pytest.raises(ValueError, match="hardware_kernel_executed"):
        analyzer._validate_physical_sample(
            _physical_sample("session", hardware_kernel_executed=False)
        )


def test_physical_validation_uses_operation_fact_when_top_level_is_absent() -> None:
    sample = _physical_sample("session")
    del sample["backend_facts"]["hardware_kernel_executed"]
    analyzer._validate_physical_sample(sample)


def test_load_arm_requires_a_bijective_sample_session_mapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    samples = [_physical_sample(f"session-{index}") for index in range(36)]
    sessions = [
        {
            "session_instance_id": f"session-{index}",
            "status": "success",
            "release_verified": True,
        }
        for index in range(36)
    ]
    monkeypatch.setattr(analyzer, "_json", lambda _path: {"status": "completed", "source_worktree_dirty": False})
    monkeypatch.setattr(
        analyzer,
        "_jsonl",
        lambda path: sessions if path.name == "sessions.jsonl" else samples,
    )
    samples[0]["session_instance_id"] = samples[1]["session_instance_id"]

    with pytest.raises(ValueError, match="unique and present"):
        analyzer._load_arm(Path("arm"))


def test_analysis_pairs_each_cell_and_component_without_pooling(monkeypatch) -> None:
    arms = {
        "baseline": _arm("baseline", 2.0),
        "optimized": _arm("optimized", 1.0),
    }
    monkeypatch.setattr(analyzer, "_load_arm", lambda path: arms[path.name])

    result = analyzer.analyze(Path("baseline"), Path("optimized"))

    assert result["cases"] == list(analyzer.EXPECTED_CASES)
    assert result["routes"] == list(analyzer.EXPECTED_ROUTES)
    assert len(result["rows"]) == 6 * (len(analyzer.COMPONENTS) + len(analyzer.COUNTERS))
    total = next(
        row
        for row in result["rows"]
        if row["case_id"] == analyzer.EXPECTED_CASES[0]
        and row["route_id"] == analyzer.EXPECTED_ROUTES[0]
        and row["component"] == "total_wall_s"
    )
    assert total["descriptive_speedup"] == pytest.approx(2.0)
    assert total["diagnostic_only"] is True
