from __future__ import annotations

import pytest

from quantum_bench.bench.summary import energy_status, write_summary
from quantum_bench.core.jsonio import append_jsonl


def _physical_energy_row() -> dict[str, object]:
    return {
        "case_id": "case-1",
        "route": "upmem-physical",
        "engine_id": "upmem_m5",
        "status": "passed",
        "timing_s": 1.0,
        "total_time_s": 1.0,
        "scientific_validation_status": "passed",
        "exact_once": True,
        "no_fallback_used": True,
        "target_observed": "physical_hardware",
        "hardware_allocation_verified": True,
        "native_kernel_executed": True,
        "hardware_kernel_executed": True,
        "simulator": False,
        "simulator_kernel_executed": False,
        "cpu_fallback": False,
        "cpu_fallback_used": False,
        "release_succeeded": True,
        "energy_joules": 0.5,
        "energy_source": "external_meter_measured",
        "energy_measurement_status": "measured",
        "energy_measurement_boundary": "host_and_dimm",
        "energy_sensor_id": "meter-1",
        "energy_measurement_interval_s": 1.0,
        "energy_sample_count": 2,
        "timing_schema_version": 2,
        "timing_scope": "whole_route_v1",
    }


@pytest.mark.parametrize("source", ["tdp_estimate", "model", "arbitrary_source"])
def test_summary_energy_rejects_non_measured_sources(source: str) -> None:
    row = _physical_energy_row()
    row["energy_source"] = source

    status = energy_status([row])

    assert status["status"] == "rejected"
    assert status["rejected_records"] == 1
    assert "energy source is not an approved measured source" in status["reasons"]


@pytest.mark.parametrize("missing_field", ["energy_measurement_interval_s", "energy_sample_count"])
def test_summary_energy_rejects_missing_interval_or_samples(missing_field: str) -> None:
    row = _physical_energy_row()
    row.pop(missing_field)

    status = energy_status([row])

    assert status["status"] == "rejected"
    assert status["rejected_records"] == 1


def test_summary_energy_accepts_complete_physical_measurement() -> None:
    status = energy_status([_physical_energy_row()])

    assert status == {
        "status": "measured",
        "measured_records": 1,
        "rejected_records": 0,
        "total_records": 1,
    }


def test_rejected_pseudo_energy_row_is_preserved_and_blocks_group_measurement(tmp_path) -> None:
    valid = _physical_energy_row()
    rejected = {**valid, "energy_source": "tdp_estimate"}
    append_jsonl(tmp_path / "raw" / "case.jsonl", valid)
    append_jsonl(tmp_path / "raw" / "case.jsonl", rejected)

    summary = write_summary(tmp_path)

    assert summary["record_count"] == 2
    assert len(summary["rows"]) == 1
    row = summary["rows"][0]
    assert row["energy_status"] == "rejected"
    assert row["energy_rejected_count"] == 1
    assert row["energy_median_j"] is None
    assert summary["energy_status"]["status"] == "rejected"
