"""Snapshot verification and read-only compatibility contracts."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from quantum_bench.bench.result_artifacts import load_result_records
from scripts import thesis_snapshot

from scripts.thesis_snapshot import verify_snapshot


ROOT = Path(__file__).resolve().parents[1]
SNAPSHOTS = (
    "current",
    "planner_v2",
    "physical_hardware_mvp_v1",
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _historical_fixture(root: Path, *, dirty: bool = False) -> tuple[Path, Path]:
    evidence = root / "runs" / "evidence" / "research_cpu_gpu" / "route" / "historical"
    _write_json(
        evidence / "run_manifest.json",
        {
            "suite_id": "research_cpu_gpu",
            "run_id": "historical",
            "benchmark_source_commit": "historical-commit",
            "benchmark_source_worktree_dirty": dirty,
            "repository_worktree_dirty": False,
            "provenance_scope": "thesis/implementation",
        },
    )
    _write_json(evidence / "environment.json", {"python": "fixture"})
    (evidence / "normalized_records.jsonl").write_text(
        json.dumps(
            {
                "schema_version": "benchmark_result_artifact_v1",
                "suite_id": "research_cpu_gpu",
                "case_id": "case_4q",
                "route_id": "quest_cpu_full_state_exact",
                "backend_family": "quest",
                "benchmark_role": "serious_full_state_baseline",
                "execution_model": "full_state",
                "contraction_execution_target": "cpu",
                "status": "completed",
                "validation_status": "passed",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    pack = root / "pack"
    _write_json(
        pack / "benchmark_manifest.json",
        {
            "benchmark_source_commit": "historical-commit",
            "benchmark_source_worktree_dirty": False,
            "evidence_inputs": [str(evidence)],
        },
    )
    return pack, evidence


def _fake_snapshot_report(_root: Path, out: Path, *, inputs: list[Path], suite_filter) -> int:
    out.mkdir(parents=True)
    _write_json(out / "benchmark_manifest.json", {"evidence_inputs": [str(item) for item in inputs]})
    _write_json(out / "plot_manifest.json", {"plots": []})
    (out / "benchmark_summary.md").write_text("# fixture\n", encoding="utf-8")
    (out / "per_case_route_stats.csv").write_text("case_id,route_id\ncase_4q,quest_cpu_full_state_exact\n", encoding="utf-8")
    (out / "plots").mkdir()
    (out / "plots" / "runtime.png").write_bytes(b"fixture")
    return 0


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


def test_clean_historical_promotion_is_allowed_without_touching_tracked_snapshots(
    monkeypatch, tmp_path: Path
) -> None:
    pack, _ = _historical_fixture(tmp_path)
    monkeypatch.setattr(thesis_snapshot, "ROOT", tmp_path)
    monkeypatch.setattr(thesis_snapshot, "report_pack", _fake_snapshot_report)
    monkeypatch.setattr(
        thesis_snapshot,
        "_git",
        lambda *args: "current-commit" if args[:2] == ("rev-parse", "HEAD") else "",
    )

    output = tmp_path / "thesis_results" / "historical_clean"
    assert thesis_snapshot.promote_snapshot(pack, output, historical=True) == 0
    thesis_snapshot.verify_snapshot(output)
    manifest = json.loads((output / "snapshot_manifest.json").read_text(encoding="utf-8"))

    assert manifest["historical_evidence_promotion"] is True
    assert manifest["benchmark_source_commit"] == "historical-commit"
    assert manifest["report_generation_commit"] == "current-commit"


@pytest.mark.parametrize("failure", ["dirty", "traversal", "binary"])
def test_historical_promotion_guards_reject_dirty_traversal_and_binary(
    monkeypatch, tmp_path: Path, failure: str
) -> None:
    pack, _ = _historical_fixture(tmp_path, dirty=failure == "dirty")
    monkeypatch.setattr(thesis_snapshot, "ROOT", tmp_path)
    monkeypatch.setattr(thesis_snapshot, "report_pack", _fake_snapshot_report)
    monkeypatch.setattr(
        thesis_snapshot,
        "_git",
        lambda *args: "current-commit" if args[:2] == ("rev-parse", "HEAD") else "",
    )

    if failure == "dirty":
        with pytest.raises(ValueError, match="evidence source is dirty"):
            thesis_snapshot.promote_snapshot(
                pack, tmp_path / "thesis_results" / "historical_dirty", historical=True
            )
    elif failure == "traversal":
        with pytest.raises(ValueError, match="destination"):
            thesis_snapshot.promote_snapshot(
                pack, tmp_path / "thesis_results" / "..", historical=True
            )
    else:
        output = tmp_path / "thesis_results" / "historical_binary"
        thesis_snapshot.promote_snapshot(pack, output, historical=True)
        (output / "evidence" / "full_state_performance" / "bad.bin").write_bytes(b"binary")
        with pytest.raises(ValueError, match="forbidden artifact"):
            thesis_snapshot.verify_snapshot(output)
