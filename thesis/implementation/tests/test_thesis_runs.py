from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts import thesis_runs


def _manifest(path: Path, selected: list[str], source_pack: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"selected_evidence": [{"source_run": value} for value in selected], "source_pack": source_pack}), encoding="utf-8")


def test_protection_scans_all_snapshot_manifests(monkeypatch, tmp_path: Path) -> None:
    keep_current = tmp_path / "runs" / "evidence" / "suite" / "route" / "keep-current"
    keep_release = tmp_path / "runs" / "evidence" / "suite" / "route" / "keep-release"
    remove = tmp_path / "runs" / "evidence" / "suite" / "route" / "remove"
    for path in (keep_current, keep_release, remove):
        path.mkdir(parents=True)
        (path / "run_manifest.json").write_text("{}", encoding="utf-8")
    _manifest(tmp_path / "thesis_results" / "current" / "snapshot_manifest.json", [str(keep_current.relative_to(tmp_path))])
    _manifest(tmp_path / "thesis_results" / "release" / "snapshot_manifest.json", [str(keep_release.relative_to(tmp_path))])
    monkeypatch.setattr(thesis_runs, "ROOT", tmp_path)
    monkeypatch.setattr(thesis_runs, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(thesis_runs, "SNAPSHOT", tmp_path / "thesis_results" / "current" / "snapshot_manifest.json")

    thesis_runs.prune_runs(apply=True)

    assert keep_current.exists()
    assert keep_release.exists()
    assert not remove.exists()


def test_protection_keeps_active_latest_target(monkeypatch, tmp_path: Path) -> None:
    active = tmp_path / "runs" / "evidence" / "suite" / "route" / "active"
    remove = tmp_path / "runs" / "evidence" / "suite" / "route" / "remove"
    for path in (active, remove):
        path.mkdir(parents=True)
        (path / "run_manifest.json").write_text("{}", encoding="utf-8")
    (active.parent / "latest").symlink_to(active.name)
    _manifest(tmp_path / "thesis_results" / "current" / "snapshot_manifest.json", [])
    monkeypatch.setattr(thesis_runs, "ROOT", tmp_path)
    monkeypatch.setattr(thesis_runs, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(thesis_runs, "SNAPSHOT", tmp_path / "thesis_results" / "current" / "snapshot_manifest.json")

    thesis_runs.prune_runs(apply=True)

    assert active.exists()
    assert not remove.exists()


def test_archive_requires_apply_and_verifies_checksums(tmp_path: Path) -> None:
    root = tmp_path / "checkout" / "thesis" / "implementation"
    source = root / "runs" / "evidence" / "suite" / "route" / "run-1"
    source.mkdir(parents=True)
    (source / "run_manifest.json").write_text("{}", encoding="utf-8")
    (source / "records.jsonl").write_text("record\n", encoding="utf-8")
    (source / "nested.txt").write_text("nested", encoding="utf-8")
    # The archive must live outside the simulated checkout, not merely outside
    # the implementation directory.
    archive_root = tmp_path / "archive"
    original_digest = hashlib.sha256((source / "records.jsonl").read_bytes()).hexdigest()
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(thesis_runs, "ROOT", root)
    try:
        destination = thesis_runs.archive_run(source, archive_root=archive_root)
        assert source.exists()
        assert not destination.exists()

        destination = thesis_runs.archive_run(source, archive_root=archive_root, apply=True)
        assert not source.exists()
        assert hashlib.sha256((destination / "records.jsonl").read_bytes()).hexdigest() == original_digest
        assert (destination / "nested.txt").read_text(encoding="utf-8") == "nested"
    finally:
        monkeypatch.undo()


def test_archive_rejects_in_repository_archive_root(monkeypatch, tmp_path: Path) -> None:
    root = tmp_path / "checkout" / "thesis" / "implementation"
    source = root / "runs" / "run"
    source.mkdir(parents=True)
    (source / "run_manifest.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(thesis_runs, "ROOT", root)
    with pytest.raises(ValueError, match="outside"):
        thesis_runs.archive_run(source, archive_root=root / "archive")


def test_archive_rejects_arbitrary_repository_directory(monkeypatch, tmp_path: Path) -> None:
    root = tmp_path / "checkout" / "thesis" / "implementation"
    source = root / "docs"
    source.mkdir(parents=True)
    monkeypatch.setattr(thesis_runs, "ROOT", root)

    with pytest.raises(ValueError, match="explicit evidence or comparison run"):
        thesis_runs.archive_run(source, archive_root=tmp_path / "archive")
