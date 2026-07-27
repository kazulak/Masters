"""Focused admission tests for the research benchmark pack."""

from __future__ import annotations

import pytest

import scripts.research_benchmark_pack as report_pack_module

from .support import hardware_evidence_records, record_with_updates


def _resident_record() -> dict[str, object]:
    return record_with_updates(
        hardware_evidence_records()[0],
        suite_id=report_pack_module.RESIDENT_SUITE_ID,
        route_id=report_pack_module.RESIDENT_ROUTE_ID,
        persistent_session_reused=False,
    )


def _persistent_record() -> dict[str, object]:
    return record_with_updates(
        hardware_evidence_records()[0],
        suite_id="historical_persistent_suite",
        route_id="upmem_tn_hardware_taskgraph_persistent",
        session_scope="case_benchmark_block",
        persistent_session_reused=True,
    )


def test_valid_resident_record_is_admitted() -> None:
    record = _resident_record()

    assert report_pack_module._is_valid_one_dpu_record(record)
    assert len(report_pack_module.upmem_one_dpu_runtime_summary([record])) == 1


def test_one_dpu_admission_requires_suite_id() -> None:
    record = _resident_record()
    record.pop("suite_id")

    assert not report_pack_module._is_valid_one_dpu_record(record)
    assert report_pack_module.upmem_one_dpu_runtime_summary([record]) == []


@pytest.mark.parametrize(
    ("session_scope", "persistent_session_reused"),
    [
        ("case_benchmark_block", False),
        ("other_scope", True),
    ],
)
def test_historical_persistent_route_keeps_session_requirements(
    session_scope: str, persistent_session_reused: bool
) -> None:
    record = record_with_updates(
        _persistent_record(),
        session_scope=session_scope,
        persistent_session_reused=persistent_session_reused,
    )

    assert not report_pack_module._is_valid_one_dpu_record(record)

