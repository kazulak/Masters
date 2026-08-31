from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest

from quantum_bench.experiment import load_experiment_config


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "analyze_request_construction.py"
SPEC = importlib.util.spec_from_file_location("analyze_request_construction", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
analyzer = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = analyzer
SPEC.loader.exec_module(analyzer)


def _timing(
    *,
    parent: float = 1.0,
    materialization: float = 0.1,
    file_write: float = 0.2,
    hashing: float = 0.3,
    record_construction: float = 0.1,
) -> dict[str, float | int]:
    return {
        "request_payload_record_staging_sum_s": parent,
        "request_payload_materialization_sum_s": materialization,
        "request_payload_file_write_sum_s": file_write,
        "request_payload_hashing_sum_s": hashing,
        "request_payload_record_construction_sum_s": record_construction,
        "request_payload_record_count": 2,
        "request_payload_files_created": 4,
        "request_payload_bytes_staged": 128,
        "request_payload_bytes_hashed": 128,
    }


def _sample(
    total: float,
    *,
    attempt_kind: str = "measurement",
    session_id: str | None = None,
    timing: dict[str, float | int] | None = None,
) -> dict[str, object]:
    return {
        "status": "success",
        "attempt_kind": attempt_kind,
        "case_id": "stress18",
        "route_id": "upmem_float32_1dpu_t8",
        "session_instance_id": session_id,
        "measurement": {"total_wall_s": total},
        "backend_facts": {
            "operation_facts": [
                {
                    "requested_dpu_count": 1,
                    "tasklets_per_dpu": 8,
                    "timing": _timing() if timing is None else timing,
                }
            ]
        },
    }


def test_child_accounting_and_residual_are_disjoint() -> None:
    result = analyzer._operation_attribution({"timing": _timing()})

    assert result["residual_s"] == pytest.approx(0.3)
    assert sum(result[name] for name in analyzer.CHILDREN) + result["residual_s"] == pytest.approx(1.0)


def test_child_accounting_rejects_negative_residual() -> None:
    with pytest.raises(ValueError, match="materially negative"):
        analyzer._operation_attribution({"timing": _timing(parent=0.6)})


def test_derive_ignores_warmup_and_reports_dominant_child() -> None:
    result = analyzer.derive_attribution(
        {
            "source_commit": "a" * 40,
            "experiment_id": "request-construction-attribution-diagnostic-v1",
        },
        (_sample(100.0, attempt_kind="warmup"), _sample(10.0), _sample(12.0)),
    )

    row = result["measurement_cells"][0]
    assert row["measurement_count"] == 2
    assert row["median_total_wall_s"] == pytest.approx(11.0)
    assert row["dominant_child"] == "payload_hashing_s"
    assert row["median_payload_residual_s"] == pytest.approx(0.3)


def test_summary_uses_sample_paired_residuals_and_share_ratios() -> None:
    samples = (
        _sample(
            10.0,
            timing=_timing(
                parent=1.0,
                materialization=0.1,
                file_write=0.1,
                hashing=0.1,
                record_construction=0.1,
            ),
        ),
        _sample(
            20.0,
            timing=_timing(
                parent=2.0,
                materialization=0.4,
                file_write=0.1,
                hashing=0.1,
                record_construction=0.1,
            ),
        ),
        _sample(
            1000.0,
            timing=_timing(
                parent=100.0,
                materialization=0.9,
                file_write=0.1,
                hashing=0.1,
                record_construction=0.1,
            ),
        ),
    )

    result = analyzer.derive_attribution(
        {"source_commit": "a" * 40, "experiment_id": "example"}, samples
    )
    summary = result["measurement_cells"][0]["summary"]

    assert summary["residual"]["median_s"] == pytest.approx(1.3)
    assert summary["children"]["payload_materialization_s"]["median_parent_share"] == pytest.approx(0.1)
    assert summary["children"]["payload_materialization_s"]["median_s"] / summary["median_parent_s"] == pytest.approx(0.2)
    assert summary["children"]["payload_materialization_s"]["median_parent_share"] != pytest.approx(
        summary["children"]["payload_materialization_s"]["median_s"] / summary["median_parent_s"]
    )


def test_attempt_time_is_joined_from_session_evidence() -> None:
    sample = _sample(10.0, session_id="session-1")
    session = {
        "session_instance_id": "session-1",
        "open_s": 2.0,
        "session_close_s": 3.0,
    }

    result = analyzer.derive_attribution(
        {"source_commit": "a" * 40, "experiment_id": "example"},
        (sample,),
        (session,),
    )
    row = result["measurement_cells"][0]

    assert row["median_session_open_s"] == pytest.approx(2.0)
    assert row["median_session_close_s"] == pytest.approx(3.0)
    assert row["median_attempt_elapsed_s"] == pytest.approx(15.0)


def test_attribution_config_has_only_two_physical_routes() -> None:
    path = ROOT / "configs" / "tn_benchmark_request_construction_attribution_diagnostic.yml"
    config = load_experiment_config(path)
    assert tuple(config["cases"]) == (
        "quantization_stress_18q_l2",
        "hs_18q_d1",
        "ghz_chain_18q",
    )
    assert tuple(config["routes"]) == (
        "upmem_float32_1dpu_t8",
        "upmem_float32_4dpu_t8",
    )
    assert all(
        tuple(item["route_ids"]) == tuple(config["routes"])
        for item in config["matrix"]
    )
    assert config["collection"]["warmup_blocks"] == 1
    assert config["collection"]["measurement_blocks"] == 5
    assert config["collection"]["claim_policy"] == "diagnostic_v1"


def test_attribution_json_is_serializable() -> None:
    result = analyzer.derive_attribution(
        {"source_commit": "b" * 40, "experiment_id": "example"},
        (_sample(10.0),),
    )
    json.dumps(result)
