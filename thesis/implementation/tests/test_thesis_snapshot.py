from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import thesis_runs, thesis_snapshot


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _evidence(
    root: Path,
    suite_id: str,
    role_name: str,
    *,
    source_dirty: bool = False,
    commit: str = "test-head",
) -> Path:
    run = root / "runs" / "evidence" / suite_id / "route" / role_name
    _write_json(
        run / "run_manifest.json",
        {
            "suite_id": suite_id,
            "run_id": role_name,
            "git_commit": commit,
            "dirty_tree": source_dirty,
            "dirty_worktree": source_dirty,
            "benchmark_source_commit": commit,
            "benchmark_source_worktree_dirty": source_dirty,
            "repository_worktree_dirty": False,
            "provenance_scope": "thesis/implementation",
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
    evidence = _evidence(tmp_path, "thesis_full_state_cpu_gpu", "2026-07-10_12-00-00")
    pack = tmp_path / "runs" / "comparisons" / "research_pack" / "latest-pack"
    _write_json(
        pack / "benchmark_manifest.json",
        {
            "git_commit": "test-head",
            "dirty_worktree": False,
            "benchmark_source_commit": "test-head",
            "benchmark_source_worktree_dirty": False,
            "repository_worktree_dirty": False,
            "provenance_scope": "thesis/implementation",
            "evidence_inputs": [str(evidence)],
        },
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
    manifest = json.loads((out / "snapshot_manifest.json").read_text(encoding="utf-8"))
    assert manifest["dirty_worktree"] is False
    assert manifest["benchmark_source_commit"] == "test-head"
    assert manifest["benchmark_source_worktree_dirty"] is False
    assert manifest["repository_worktree_dirty"] is False
    assert manifest["provenance_scope"] == "thesis/implementation"
    assert manifest["report_generation_commit"] == "test-head"
    assert manifest["snapshot_promotion_commit"] == "test-head"
    assert manifest["report_generation_worktree_dirty"] is False
    assert manifest["snapshot_promotion_worktree_dirty"] is False
    assert set(manifest["provenance_stages"]) == {"benchmark_source", "report_generation", "snapshot_promotion"}
    assert all(not stage["worktree_dirty"] for stage in manifest["provenance_stages"].values())
    report_manifest = json.loads((out / "report_manifest.json").read_text(encoding="utf-8"))
    assert report_manifest["report_generation_commit"] == "test-head"
    assert report_manifest["report_generation_worktree_dirty"] is False
    assert report_manifest["report_generation_provenance"]["benchmark_source_worktree_dirty"] is False


def test_historical_promotion_accepts_clean_non_head_evidence_and_records_stages(
    monkeypatch, tmp_path: Path
) -> None:
    evidence = _evidence(
        tmp_path,
        "research_cpu_gpu",
        "historical-run",
        commit="historical-commit",
    )
    pack = tmp_path / "pack"
    _write_json(
        pack / "benchmark_manifest.json",
        {
            "benchmark_source_commit": "historical-commit",
            "benchmark_source_worktree_dirty": False,
            "evidence_inputs": [str(evidence)],
        },
    )
    monkeypatch.setattr(thesis_snapshot, "report_pack", _fake_report)

    def fake_git(*args: str) -> str:
        if args[:2] == ("rev-parse", "HEAD"):
            return "current-commit"
        return ""

    monkeypatch.setattr(thesis_snapshot, "_git", fake_git)
    monkeypatch.setattr(thesis_snapshot, "ROOT", tmp_path)
    out = tmp_path / "thesis_results" / "historical_clean"

    assert thesis_snapshot.promote_snapshot(pack, out, historical=True) == 0
    manifest = json.loads((out / "snapshot_manifest.json").read_text(encoding="utf-8"))
    assert manifest["historical_evidence_promotion"] is True
    assert manifest["snapshot_id"] == "historical_clean"
    assert manifest["benchmark_source_commit"] == "historical-commit"
    assert manifest["report_generation_commit"] == "current-commit"
    assert manifest["snapshot_promotion_commit"] == "current-commit"


def test_historical_promotion_rejects_dirty_evidence_and_current_destination(
    monkeypatch, tmp_path: Path
) -> None:
    dirty = _evidence(
        tmp_path,
        "research_cpu_gpu",
        "historical-dirty",
        source_dirty=True,
        commit="historical-commit",
    )
    pack = tmp_path / "pack"
    _write_json(
        pack / "benchmark_manifest.json",
        {
            "benchmark_source_commit": "historical-commit",
            "benchmark_source_worktree_dirty": False,
            "evidence_inputs": [str(dirty)],
        },
    )
    monkeypatch.setattr(thesis_snapshot, "report_pack", _fake_report)
    monkeypatch.setattr(
        thesis_snapshot,
        "_git",
        lambda *args: "current-commit" if args[:2] == ("rev-parse", "HEAD") else "",
    )
    monkeypatch.setattr(thesis_snapshot, "ROOT", tmp_path)

    with pytest.raises(ValueError, match="evidence source is dirty"):
        thesis_snapshot.promote_snapshot(
            pack, tmp_path / "thesis_results" / "historical_dirty", historical=True
        )
    with pytest.raises(ValueError, match="named snapshot"):
        thesis_snapshot.promote_snapshot(
            pack, tmp_path / "thesis_results" / "current", historical=True
        )


def test_historical_promotion_rejects_traversal_to_repository_root(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(thesis_snapshot, "ROOT", tmp_path)
    with pytest.raises(ValueError, match="destination"):
        thesis_snapshot.promote_snapshot(
            tmp_path / "missing-pack",
            tmp_path / "thesis_results" / "..",
            historical=True,
        )
    assert not (tmp_path / "snapshot_manifest.json").exists()


def test_normal_promotion_remains_strict_for_non_head_evidence(
    monkeypatch, tmp_path: Path
) -> None:
    evidence = _evidence(tmp_path, "research_cpu_gpu", "mismatched", commit="old-commit")
    pack = tmp_path / "pack"
    _write_json(
        pack / "benchmark_manifest.json",
        {
            "benchmark_source_commit": "old-commit",
            "benchmark_source_worktree_dirty": False,
            "evidence_inputs": [str(evidence)],
        },
    )
    monkeypatch.setattr(thesis_snapshot, "_git", lambda *args: "current-commit" if args[:2] == ("rev-parse", "HEAD") else "")
    monkeypatch.setattr(thesis_snapshot, "ROOT", tmp_path)

    with pytest.raises(ValueError, match="does not match current HEAD"):
        thesis_snapshot.promote_snapshot(pack, tmp_path / "snapshot")


def test_outside_repository_dirtiness_is_recorded_without_dirty_source(monkeypatch, tmp_path: Path) -> None:
    evidence = _evidence(tmp_path, "research_cpu_gpu", "run")
    pack = tmp_path / "pack"
    _write_json(
        pack / "benchmark_manifest.json",
        {
            "git_commit": "test-head",
            "dirty_worktree": False,
            "benchmark_source_commit": "test-head",
            "benchmark_source_worktree_dirty": False,
            "repository_worktree_dirty": True,
            "provenance_scope": "thesis/implementation",
            "evidence_inputs": [str(evidence)],
        },
    )
    monkeypatch.setattr(thesis_snapshot, "report_pack", _fake_report)

    def fake_git(*args: str) -> str:
        if args[:2] == ("rev-parse", "HEAD"):
            return "test-head"
        if args[-1] == ":(top)**":
            return "?? outside.txt"
        return ""

    monkeypatch.setattr(thesis_snapshot, "_git", fake_git)
    monkeypatch.setattr(thesis_snapshot, "ROOT", tmp_path)
    out = tmp_path / "snapshot"

    assert thesis_snapshot.promote_snapshot(pack, out) == 0
    manifest = json.loads((out / "snapshot_manifest.json").read_text(encoding="utf-8"))
    assert manifest["benchmark_source_worktree_dirty"] is False
    assert manifest["repository_worktree_dirty"] is True
    assert manifest["provenance_stages"]["snapshot_promotion"]["repository_worktree_dirty"] is True


def test_hardware_mvp_capsule_keeps_profile_and_explicit_role(tmp_path: Path) -> None:
    evidence = _evidence(tmp_path, "upmem_hardware_mvp", "run")
    _write_json(
        evidence / "config" / "hardware_profile.json",
        {"version": "hardware_mvp_l1_v2", "requested_dpu_count": 1},
    )
    staging = tmp_path / "staging"
    (staging / "evidence").mkdir(parents=True)
    (staging / "suites").mkdir()

    entry = thesis_snapshot._copy_evidence_capsule(evidence, staging, set())

    assert entry["role"] == "upmem_hardware_functionality"
    capsule = staging / entry["snapshot_path"]
    assert (capsule / "hardware_profile.json").is_file()
    assert (staging / "suites" / "upmem_hardware_functionality.yml").is_file()


def test_taskgraph_hardware_capsule_uses_a_distinct_role(tmp_path: Path) -> None:
    evidence = _evidence(tmp_path, "upmem_hardware_taskgraph_correctness", "run")
    staging = tmp_path / "staging"
    (staging / "evidence").mkdir(parents=True)
    (staging / "suites").mkdir()

    entry = thesis_snapshot._copy_evidence_capsule(evidence, staging, set())

    assert entry["role"] == "upmem_hardware_taskgraph_correctness"


def test_taskgraph_study_capsule_uses_host_rehydrated_role(tmp_path: Path) -> None:
    evidence = _evidence(
        tmp_path, "upmem_hardware_taskgraph_path_quantization", "study-run"
    )
    staging = tmp_path / "staging"
    (staging / "evidence").mkdir(parents=True)
    (staging / "suites").mkdir()

    entry = thesis_snapshot._copy_evidence_capsule(evidence, staging, set())

    assert entry["role"] == "physical_one_dpu_taskgraph_host_rehydrated"


def test_dirty_implementation_source_fails_promotion(monkeypatch, tmp_path: Path) -> None:
    evidence = _evidence(tmp_path, "research_cpu_gpu", "run")
    pack = tmp_path / "pack"
    _write_json(
        pack / "benchmark_manifest.json",
        {"git_commit": "test-head", "dirty_worktree": False, "evidence_inputs": [str(evidence)]},
    )
    monkeypatch.setattr(thesis_snapshot, "report_pack", _fake_report)

    def fake_git(*args: str) -> str:
        if args[:2] == ("rev-parse", "HEAD"):
            return "test-head"
        if args[-1] == ":(top)thesis/implementation":
            return " M thesis/implementation/src/quantum_bench/bench/reporting.py"
        return ""

    monkeypatch.setattr(thesis_snapshot, "_git", fake_git)
    monkeypatch.setattr(thesis_snapshot, "ROOT", tmp_path)

    try:
        thesis_snapshot.promote_snapshot(pack, tmp_path / "snapshot")
    except ValueError as exc:
        assert "clean thesis/implementation source" in str(exc)
    else:
        raise AssertionError("dirty implementation source should fail snapshot promotion")


def test_dirty_evidence_fails_promotion(monkeypatch, tmp_path: Path) -> None:
    evidence = _evidence(tmp_path, "research_cpu_gpu", "run", source_dirty=True)
    pack = tmp_path / "pack"
    _write_json(
        pack / "benchmark_manifest.json",
        {"git_commit": "test-head", "dirty_worktree": False, "evidence_inputs": [str(evidence)]},
    )
    monkeypatch.setattr(thesis_snapshot, "report_pack", _fake_report)
    monkeypatch.setattr(
        thesis_snapshot,
        "_git",
        lambda *args: "test-head" if args[:2] == ("rev-parse", "HEAD") else "",
    )
    monkeypatch.setattr(thesis_snapshot, "ROOT", tmp_path)

    try:
        thesis_snapshot.promote_snapshot(pack, tmp_path / "snapshot")
    except ValueError as exc:
        assert "evidence source is dirty" in str(exc)
    else:
        raise AssertionError("dirty evidence should fail snapshot promotion")


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


def test_snapshot_verification_rejects_symlinks(monkeypatch, tmp_path: Path) -> None:
    evidence = _evidence(tmp_path, "research_cpu_gpu", "run")
    pack = tmp_path / "pack"
    _write_json(pack / "benchmark_manifest.json", {"git_commit": "test-head", "dirty_worktree": False, "evidence_inputs": [str(evidence)]})
    monkeypatch.setattr(thesis_snapshot, "report_pack", _fake_report)
    monkeypatch.setattr(thesis_snapshot, "_git", lambda *args: "test-head" if args[:2] == ("rev-parse", "HEAD") else "")
    monkeypatch.setattr(thesis_snapshot, "ROOT", tmp_path)
    out = tmp_path / "snapshot"
    thesis_snapshot.promote_snapshot(pack, out)
    (out / "bad-link").symlink_to("missing")

    try:
        thesis_snapshot.verify_snapshot(out)
    except ValueError as exc:
        assert "cannot contain symlinks" in str(exc)
    else:
        raise AssertionError("snapshot symlinks should fail verification")


def test_snapshot_verification_rejects_mixed_planner_semantics(monkeypatch, tmp_path: Path) -> None:
    evidence = _evidence(tmp_path, "research_cpu_gpu", "run")
    pack = tmp_path / "pack"
    _write_json(pack / "benchmark_manifest.json", {"git_commit": "test-head", "dirty_worktree": False, "evidence_inputs": [str(evidence)]})
    monkeypatch.setattr(thesis_snapshot, "report_pack", _fake_report)
    monkeypatch.setattr(thesis_snapshot, "_git", lambda *args: "test-head" if args[:2] == ("rev-parse", "HEAD") else "")
    monkeypatch.setattr(thesis_snapshot, "ROOT", tmp_path)
    out = tmp_path / "snapshot"
    thesis_snapshot.promote_snapshot(pack, out)

    records = [
        {"route_id": "planner_candidate_model", "pim_objective_version": "upmem_path_cost_v1", "pim_weight_profile": "balanced"},
        {"route_id": "planner_candidate_model", "pim_objective_version": "upmem_path_cost_v2", "pim_weight_profile": "balanced"},
    ]
    (out / "evidence" / "full_state_performance" / "normalized_records.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="snapshot planner semantic versions are mixed"):
        thesis_snapshot.verify_snapshot(out)


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


def test_run_listing_does_not_duplicate_latest_symlinks(monkeypatch, tmp_path: Path) -> None:
    run = _evidence(tmp_path, "research_cpu_gpu", "2026-07-10_12-00-00")
    latest = run.parent / "latest"
    latest.symlink_to(run.name)
    monkeypatch.setattr(thesis_runs, "ROOT", tmp_path)
    monkeypatch.setattr(thesis_runs, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(thesis_runs, "SNAPSHOT", tmp_path / "missing_snapshot.json")

    rows = thesis_runs._run_rows()

    assert [row["run_id"] for row in rows] == [run.name]
