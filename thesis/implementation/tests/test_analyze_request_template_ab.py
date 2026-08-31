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
