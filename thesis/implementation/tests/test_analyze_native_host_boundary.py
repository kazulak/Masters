from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "analyze_native_host_boundary.py"
SPEC = importlib.util.spec_from_file_location("analyze_native_host_boundary", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
analyzer = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = analyzer
SPEC.loader.exec_module(analyzer)


def _timing(scale: float = 1.0) -> dict[str, float | int]:
    return {
        "total_wall_s": 11.0 * scale,
        "request_wave_wall_sum_s": 10.0 * scale,
        "rank_response_total_route_max_sum_s": 5.0 * scale,
        "rank_response_h2d_max_sum_s": 0.5 * scale,
        "rank_response_kernel_max_sum_s": 3.0 * scale,
        "rank_response_d2h_max_sum_s": 0.5 * scale,
        "request_build_sum_s": 2.0 * scale,
        "request_work_unit_materialization_sum_s": 0.2 * scale,
        "request_artifact_build_sum_s": 1.2 * scale,
        "request_payload_record_staging_sum_s": 0.8 * scale,
        "request_manifest_sidecar_staging_sum_s": 0.2 * scale,
        "request_payload_materialization_sum_s": 0.1 * scale,
        "request_payload_file_write_sum_s": 0.2 * scale,
        "request_payload_hashing_sum_s": 0.2 * scale,
        "request_payload_record_construction_sum_s": 0.2 * scale,
        "rank_submit_parallel_wall_sum_s": 7.0 * scale,
        "rank_submit_total_max_sum_s": 6.0 * scale,
        "rank_submit_artifact_validation_max_sum_s": 0.4 * scale,
        "rank_submit_protocol_write_max_sum_s": 0.2 * scale,
        "rank_submit_response_wait_max_sum_s": 5.0 * scale,
        "rank_submit_response_validation_max_sum_s": 0.1 * scale,
        "coordinator_response_processing_sum_s": 0.5 * scale,
        "request_payload_record_count": 4,
        "request_payload_files_created": 8,
        "request_payload_bytes_staged": 1024,
        "request_payload_bytes_hashed": 1024,
    }


def _sample(scale: float = 1.0) -> dict[str, object]:
    return {
        "case_id": "stress",
        "route_id": "route",
        "plan_id": "greedy",
        "measurement": {"total_wall_s": 11.0 * scale},
        "backend_facts": {
            "total_wave_count": 2,
            "operation_facts": [
                {
                    "rank_count": 1,
                    "requested_dpu_count": 1,
                    "tasklets_per_dpu": 8,
                    "timing": _timing(scale),
                }
            ],
        },
    }


def test_operation_residuals_are_nonnegative_and_nested() -> None:
    result = analyzer._operation_attribution(_sample()["backend_facts"]["operation_facts"][0])
    costs = result["costs"]

    assert costs["payload_record_residual_s"]["value_s"] == pytest.approx(0.1)
    assert costs["request_build_residual_s"]["value_s"] == pytest.approx(0.6)
    assert costs["host_request_overhead_s"]["value_s"] == pytest.approx(5.0)
    assert result["counters"]["request_payload_record_count"] == 4


def test_sample_includes_session_attempt_and_request_counts() -> None:
    value = analyzer._sample_attribution(
        _sample(),
        {
            "status": "success",
            "case_id": "stress",
            "route_id": "route",
            "open_s": 1.0,
            "session_close_s": 2.0,
        },
    )

    assert value is not None
    assert value["costs"]["attempt_elapsed_s"] == pytest.approx(14.0)
    assert value["counters"]["request_count"] == 8


def test_summary_does_not_add_nested_timers() -> None:
    first = analyzer._sample_attribution(_sample())
    second = analyzer._sample_attribution(_sample(1.1))
    assert first is not None and second is not None
    summary = analyzer._summary([first, second])

    assert summary["costs"]["request_wave_wall_s"]["median_s"] == pytest.approx(10.5)
    assert summary["costs"]["native_kernel_s"]["median_s"] == pytest.approx(3.15)
    assert summary["counters"]["request_count"]["median"] == pytest.approx(8.0)


def test_real_canonical_artifact_has_two_stress_routes() -> None:
    evidence = Path(
        "/home/tom/repos/Masters/.agent-work-circuit-sensitivity/thesis/implementation"
        "/runs/qualification/request-template-v1/c4efb3f17e29672e91a0a844881ead53ccf9f2c7"
        "/parallel-diagnostic/run"
    )
    if not (evidence / "samples.jsonl").is_file():
        pytest.skip("canonical qualification evidence is not present")
    manifest = json.loads((evidence / "manifest.json").read_text(encoding="utf-8"))
    samples = [
        json.loads(line)
        for line in (evidence / "samples.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    sessions = [
        json.loads(line)
        for line in (evidence / "sessions.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    result = analyzer.derive_attribution(manifest, samples, sessions)
    assert {row["route_id"] for row in result["measurement_cells"]} >= {
        "upmem_float32_1dpu_t8",
        "upmem_float32_4dpu_t8",
    }
    assert all(row["median_request_count"] is not None for row in result["measurement_cells"])
