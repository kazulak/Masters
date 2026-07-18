"""Snapshot verification and read-only compatibility contracts."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from quantum_bench.bench.result_artifacts import load_result_records

from scripts.thesis_snapshot import verify_snapshot


ROOT = Path(__file__).resolve().parents[1]
SNAPSHOTS = (
    "current",
    "planner_v2",
    "physical_hardware_mvp_v1",
)


@pytest.mark.parametrize("snapshot_name", SNAPSHOTS)
def test_tracked_snapshot_verifies_and_legacy_reader_preserves_routes(
    snapshot_name: str,
) -> None:
    snapshot = ROOT / "thesis_results" / snapshot_name

    verify_snapshot(snapshot)
    manifest = json.loads(
        (snapshot / "snapshot_manifest.json").read_text(encoding="utf-8")
    )
    inputs = [snapshot / entry["snapshot_path"] for entry in manifest["selected_evidence"]]
    records = load_result_records(inputs)
    routes = {str(record.get("route_id")) for record in records}

    assert records
    assert None not in routes
    if snapshot_name == "current":
        assert {
            "cpu_tn_frontier_exact",
            "cpu_tn_hybrid_sliced_frontier_exact",
        } <= routes
    elif snapshot_name == "planner_v2":
        assert routes == {"planner_candidate_model"}
    else:
        assert "upmem_dense_l1_int8_hardware_mvp" in routes


def test_historical_snapshot_reader_does_not_rename_route_labels() -> None:
    snapshot = ROOT / "thesis_results" / "current"
    manifest = json.loads(
        (snapshot / "snapshot_manifest.json").read_text(encoding="utf-8")
    )
    inputs = [snapshot / entry["snapshot_path"] for entry in manifest["selected_evidence"]]
    route_ids = {
        record.get("route_id") for record in load_result_records(inputs)
    }

    assert "cpu_tn_frontier_exact" in route_ids
    assert "cpu_tn_hybrid_sliced_frontier_exact" in route_ids
    assert "upmem_tn_runtime" in route_ids


def test_snapshot_checksum_rejects_modified_historical_content(tmp_path: Path) -> None:
    source = ROOT / "thesis_results" / "planner_v2"
    copy = tmp_path / "planner_v2"
    shutil.copytree(source, copy)
    readme = copy / "README.md"
    readme.write_text(readme.read_text(encoding="utf-8") + "\nchanged\n", encoding="utf-8")

    with pytest.raises(ValueError, match="checksum"):
        verify_snapshot(copy)

