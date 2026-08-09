from __future__ import annotations

from scripts import upmem_m4_2_report as report


def test_m42_report_uses_per_iteration_timing_and_conditional_verdict() -> None:
    row = {
        "case_id": "rank1_dot",
        "repeat_id": 0,
        "total_route_time_s": 0.25,
        "application_visible_h2d_bytes": 2048,
        "application_visible_d2h_bytes": 16,
        "exact_integer_match": True,
        "simplepim_operator_api_used": True,
    }

    exported = report._record_csv_row(row)

    assert report.CONDITIONAL_STATUS == "host_reported_functionality_evidence_conditional"
    assert report.REPORT_TIME_FIELD in report._record_csv_fields()
    assert "total_route_time_s" not in report._record_csv_fields()
    assert exported[report.REPORT_TIME_FIELD] == 0.25

