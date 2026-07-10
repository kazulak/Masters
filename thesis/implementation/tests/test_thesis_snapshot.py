from __future__ import annotations

import json
from pathlib import Path

from scripts import thesis_runs, thesis_snapshot


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _evidence(root: Path, suite_id: str, role_name: str) -> Path:
    run = root / "runs" / "evidence" / suite_id / "route" / role_name
    _write_json(
        run / "run_manifest.json",
        {
            "suite_id": suite_id,
            "run_id": role_name,
            "git_commit": "test-head",
            "summary": "summary.json",
        },
    )
    _write_json(run / "environment.json", {"python": "test"})
    _write_json(run / "summary.json", {"status": "completed"})
    (run / "normalized_records.jsonl").write_text(
        json.dumps(
            {
                "schema_version": "benchmark_result_artifact_v1",
                "suite_id": suite_id,
                "case_id": "case_4q",
                "route_id": "quest_cpu_full_state_exact",
                "backend_family": "quest",
                "benchmark_role": "serious_full_state_baseline",
                "execution_model": "full_state",
                "contraction_execution_target": "cpu",
                "status": "completed",
                "validation_status": "passed",
                "repeat_id": 0,
                "n_qubits": 4,
                "simulation_compute_time_s": 1.0,
                "total_wall_time_s": 1.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (run / "config").mkdir()
    (run / "config" / "resolved_suite.yml").write_text(f"suite_id: {suite_id}\n", encoding="utf-8")
    return run


def _fake_report(_root: Path, out: Path, *, inputs: list[Path], suite_filter) -> int:
    out.mkdir(parents=True)
    _write_json(
        out / "benchmark_manifest.json",
        {"root": str(_root), "evidence_inputs": [str(item) for item in inputs], "command_line": "test"},
    )
    _write_json(out / "plot_manifest.json", {"plots": []})
    (out / "benchmark_summary.md").write_text("# Snapshot report\n", encoding="utf-8")
    (out / "per_case_route_stats.csv").write_text("case_id,route_id\ncase_4q,quest_cpu_full_state_exact\n", encoding="utf-8")
    (out / "plots").mkdir()
    (out / "plots" / "runtime.png").write_bytes(b"png")
    return 0


def test_promote_snapshot_copies_compact_evidence_and_report(monkeypatch, tmp_path: Path) -> None:
    evidence = _evidence(tmp_path, "research_cpu_gpu", "2026-07-10_12-00-00")
    pack = tmp_path / "runs" / "comparisons" / "research_pack" / "latest-pack"
    _write_json(
        pack / "benchmark_manifest.json",
        {"git_commit": "test-head", "dirty_worktree": False, "evidence_inputs": [str(evidence)]},
    )
    monkeypatch.setattr(thesis_snapshot, "report_pack", _fake_report)
    monkeypatch.setattr(thesis_snapshot, "_git", lambda *args: "test-head" if args[:2] == ("rev-parse", "HEAD") else "")
    monkeypatch.setattr(thesis_snapshot, "ROOT", tmp_path)
    out = tmp_path / "thesis_results" / "current"

    assert thesis_snapshot.promote_snapshot(pack, out) == 0
    thesis_snapshot.verify_snapshot(out)
    assert (out / "evidence" / "full_state_performance" / "normalized_records.jsonl").is_file()
    assert (out / "tables" / "per_case_route_stats.csv").is_file()
    assert (out / "plots" / "runtime.png").is_file()
    assert not list(out.rglob("*.npy"))


def test_snapshot_verification_rejects_binary_evidence(monkeypatch, tmp_path: Path) -> None:
    evidence = _evidence(tmp_path, "research_cpu_gpu", "run")
    pack = tmp_path / "pack"
    _write_json(pack / "benchmark_manifest.json", {"git_commit": "test-head", "dirty_worktree": False, "evidence_inputs": [str(evidence)]})
    monkeypatch.setattr(thesis_snapshot, "report_pack", _fake_report)
    monkeypatch.setattr(thesis_snapshot, "_git", lambda *args: "test-head" if args[:2] == ("rev-parse", "HEAD") else "")
    monkeypatch.setattr(thesis_snapshot, "ROOT", tmp_path)
    out = tmp_path / "snapshot"
    thesis_snapshot.promote_snapshot(pack, out)
    (out / "evidence" / "bad.npy").write_bytes(b"bad")

    try:
        thesis_snapshot.verify_snapshot(out)
    except ValueError as exc:
        assert "forbidden artifact" in str(exc)
    else:
        raise AssertionError("binary evidence should fail snapshot verification")


def test_prune_runs_keeps_only_snapshot_selection(monkeypatch, tmp_path: Path) -> None:
    keep = _evidence(tmp_path, "research_cpu_gpu", "keep")
    remove = _evidence(tmp_path, "research_cpu_tn", "remove")
    snapshot_manifest = tmp_path / "thesis_results" / "current" / "snapshot_manifest.json"
    _write_json(
        snapshot_manifest,
        {
            "selected_evidence": [
                {"source_run": keep.relative_to(tmp_path).as_posix()},
            ]
        },
    )
    monkeypatch.setattr(thesis_runs, "ROOT", tmp_path)
    monkeypatch.setattr(thesis_runs, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(thesis_runs, "SNAPSHOT", snapshot_manifest)

    thesis_runs.prune_runs(apply=True)

    assert keep.exists()
    assert not remove.exists()
