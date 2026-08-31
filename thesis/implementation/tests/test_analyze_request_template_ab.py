from __future__ import annotations

import importlib.util
import json
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
                        "payload_record_count": case_index * 4,
                        "payload_files_created": case_index * 2,
                        "payload_bytes_staged": case_index * 8,
                        "payload_bytes_hashed": case_index * 8,
                        "payload_record_staging_residual_s": 0.0,
                        "attempt_elapsed_s": scale * 3 * (case_index + route_index + block),
                    }
                )
                measurements[(case_id, route_id, block)] = values
    return {
        "source_commit": source,
        "experiment_id": f"experiment-{source}",
        "run_id": f"run-{source}",
        "environment_identity": {},
        "experiment_contract": {},
        "identities": {},
        "binary_identities": {},
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
        "plan_id": "greedy",
        "experiment_id": "experiment",
        "run_id": "run",
        "identities": {
            field: f"{field}-value" for field in analyzer.IDENTITY_FIELDS
        },
        "numeric_facts": {"numeric_policy": "split_complex_float32_v1"},
        "validation": {
            "accuracy_qualified": True,
            "full_precision_passed": True,
            "policy_reference_passed": True,
        },
        "measurement": {"scope_id": "steady_execution_v1"},
        "backend_facts": {
            "target_observed": "physical_hardware",
            "simulator_kernel_executed": False,
            "cpu_fallback_used": False,
            "hardware_kernel_executed": hardware_kernel_executed,
            "kernel_implementation_id": "kernel",
            "kernel_policy": "kernel-policy",
            "intermediate_policy": "host-roundtrip",
            "requested_dpus": 1,
            "allocated_dpus": 1,
            "active_dpus": 1,
            "rank_count": 1,
            "tasklets_per_dpu": 1,
            "operation_facts": [
                {
                    "target_observed": "physical_hardware",
                    "simulator_kernel_executed": False,
                    "cpu_fallback_used": False,
                    "hardware_kernel_executed": True,
                    "lane_pass_count": 4,
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
            "experiment_id": "experiment",
            "run_id": "run",
            "case_id": analyzer.EXPECTED_CASES[0],
            "route_id": analyzer.EXPECTED_ROUTES[0],
            "status": "success",
            "release_verified": True,
            "terminal_backend_facts": {
                field: "a" * 64 for field in analyzer.BINARY_HASH_FIELDS
            },
        }
        for index in range(36)
    ]
    manifest = {
        "status": "completed",
        "source_worktree_dirty": False,
        "source_commit": "a" * 40,
        "experiment_id": "experiment",
        "run_id": "run",
        "configuration": {
            "environment": {
                field: {} if field in {"blas", "thread_environment"} else []
                for field in analyzer.ENVIRONMENT_FIELDS
            },
            "experiment": {
                "cases": {},
                "matrix": [],
                "plans": [],
                "routes": {},
                "collection": {"claim_policy": "diagnostic_v1"},
            },
        },
    }
    report = {
        "status": "completed",
        "artifact_status": "completed",
        "experiment_id": "experiment",
        "run_id": "run",
        "qualification": {"claim_eligible_aggregate_count": 0},
        "speedup_count": 0,
        "speedup_rejections": {"candidate_diagnostic_claim_policy": 6},
    }

    def fake_json(path: Path) -> dict[str, object]:
        return report if path.name == "report.json" else manifest

    monkeypatch.setattr(analyzer, "_json", fake_json)
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

    assert len(result["reconciliation_rows"]) == 6
    reconciliation = result["reconciliation_rows"][0]
    assert reconciliation["baseline_total_wall_s_median"] == pytest.approx(10.0)
    assert reconciliation["optimized_total_wall_s_median"] == pytest.approx(5.0)
    assert reconciliation["median_steady_total_time_saved_s"] == pytest.approx(5.0)
    assert reconciliation["median_session_inclusive_time_saved_s"] == pytest.approx(
        15.0
    )
    assert not any(key.startswith("optimized_template") for key in reconciliation)


def test_reconciliation_document_and_csv_are_self_contained(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    arms = {
        "baseline": _arm("baseline", 2.0),
        "optimized": _arm("optimized", 1.0),
    }
    monkeypatch.setattr(analyzer, "_load_arm", lambda path: arms[path.name])

    result = analyzer.analyze(Path("baseline"), Path("optimized"))
    document = analyzer._reconciliation_document(result)
    json_path = tmp_path / "request_template_ab_reconciliation.json"
    csv_path = tmp_path / "request_template_ab_reconciliation.csv"
    json_path.write_text(json.dumps(document), encoding="utf-8")
    analyzer._write_reconciliation_csv(csv_path, document["rows"])

    loaded = json.loads(json_path.read_text(encoding="utf-8"))
    assert loaded["rows"] == result["reconciliation_rows"]
    assert "measurement.total_wall_s" in loaded["timing_semantics"]["attempt_elapsed_s"]
    assert loaded["template_semantics"]["cardinality"].startswith("not inferred")
    assert csv_path.read_text(encoding="utf-8").count("\n") == 7


def test_shared_identity_rejects_different_controlled_environment(monkeypatch) -> None:
    arms = {
        "baseline": _arm("baseline", 2.0),
        "optimized": _arm("optimized", 1.0),
    }
    arms["optimized"]["environment_identity"] = {"governor": "performance"}
    monkeypatch.setattr(analyzer, "_load_arm", lambda path: arms[path.name])

    with pytest.raises(ValueError, match="controlled environment"):
        analyzer.analyze(Path("baseline"), Path("optimized"))


def test_paired_delta_uses_same_block_not_difference_of_medians() -> None:
    result = analyzer._paired_delta_summary(
        [1.0, 2.0, 3.0, 4.0, 100.0],
        [1.0, 100.0, 100.0, 100.0, 100.0],
    )

    assert result["median"] == pytest.approx(96.0)
    assert result["median"] != pytest.approx(
        analyzer.median([1.0, 100.0, 100.0, 100.0, 100.0])
        - analyzer.median([1.0, 2.0, 3.0, 4.0, 100.0])
    )
