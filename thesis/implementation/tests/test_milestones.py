from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from quantum_bench.bench import milestones as ledger_module


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "milestones.yml"
SOURCE_COMMIT = "c7bbf957d17346e819c52fc45ca592c3bcb691ca"


def _ledger(tmp_path: Path, milestone: dict[str, object]) -> Path:
    path = tmp_path / "milestones.yml"
    path.write_text(
        yaml.safe_dump({"schema_version": 1, "milestones": [milestone]}),
        encoding="utf-8",
    )
    return path


def _observed_milestone() -> dict[str, object]:
    return {
        "id": "test",
        "title": "Test milestone",
        "status": "development_observed",
        "summary": "Observed in an ignored development run.",
        "source_commit": "b550c46b24ed1da07deb3d4a1043d751df987f5d",
        "source_commands": ["make m5-circuit-study"],
        "replay_commands": ["make m5-circuit-study"],
        "evidence_origin": "ignored_external",
        "allowed_claims": ["bounded observation"],
        "prohibited_claims": ["clean-clone verification"],
        "sources": ["docs/README.md"],
    }


def _capsule_milestone() -> dict[str, object]:
    return {
        "id": "M4.5-test",
        "source_commit": SOURCE_COMMIT,
        "evidence": {
            "checksums": "checksums.json",
            "capsule_manifest": "capsule_manifest.json",
        },
    }


def _write_valid_capsule(root: Path) -> dict[str, object]:
    root.mkdir()
    readme = root / "README.md"
    artifact = root / "artifact.json"
    capsule = root / "capsule_manifest.json"
    readme.write_text("evidence\n", encoding="utf-8")
    artifact.write_text('{"status": "passed"}\n', encoding="utf-8")
    capsule.write_text(
        json.dumps(
            {"source_commit": SOURCE_COMMIT, "files": ["artifact.json"]},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    checksums = {
        "README.md": hashlib.sha256(readme.read_bytes()).hexdigest(),
        "artifact.json": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        "capsule_manifest.json": hashlib.sha256(capsule.read_bytes()).hexdigest(),
    }
    (root / "checksums.json").write_text(
        json.dumps(checksums, indent=2) + "\n", encoding="utf-8"
    )
    return _capsule_milestone()


def test_committed_ledger_and_generated_document_verify() -> None:
    result = ledger_module.verify_ledger(CONFIG, ROOT)

    assert result["status"] == "verified"
    assert result["verified_tracked_evidence"] == 1
    assert ledger_module.render_ledger(CONFIG, ROOT) == (
        ROOT / "docs" / "MILESTONES.md"
    ).read_text(encoding="utf-8")


def test_cli_verifies_committed_ledger() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "quantum_bench.bench", "milestones", "verify"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert '"status": "verified"' in result.stdout


def test_thesis_verify_runs_ledger_before_snapshot_verification() -> None:
    result = subprocess.run(
        ["make", "-n", f"PYTHON={sys.executable}", "thesis-verify"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    ledger_command = "quantum_bench.bench milestones verify"
    snapshot_command = "thesis_snapshot.py verify"
    assert ledger_command in result.stdout
    assert snapshot_command in result.stdout
    assert result.stdout.index(ledger_command) < result.stdout.index(snapshot_command)


def test_unknown_historical_source_target_is_rejected(tmp_path: Path) -> None:
    milestone = _observed_milestone()
    milestone["source_commands"] = ["make missing-at-source"]

    with pytest.raises(ValueError, match="source_commands.*unknown Make target"):
        ledger_module.verify_ledger(_ledger(tmp_path, milestone), ROOT)


def test_unknown_current_replay_target_is_rejected(tmp_path: Path) -> None:
    milestone = _observed_milestone()
    milestone["replay_commands"] = ["make missing-current-target"]

    with pytest.raises(ValueError, match="replay_commands.*unknown Make target"):
        ledger_module.verify_ledger(_ledger(tmp_path, milestone), ROOT)


def test_commit_fields_require_full_shas(tmp_path: Path) -> None:
    milestone = _observed_milestone()
    milestone["source_commit"] = "b550c46"

    with pytest.raises(ValueError, match="full 40-character Git SHA"):
        ledger_module.verify_ledger(_ledger(tmp_path, milestone), ROOT)


def test_ignored_external_evidence_cannot_claim_tracked_verification(
    tmp_path: Path,
) -> None:
    milestone = _observed_milestone()
    milestone["status"] = "tracked_verified"

    with pytest.raises(ValueError, match="evidence_origin must be tracked"):
        ledger_module.verify_ledger(_ledger(tmp_path, milestone), ROOT)


def test_non_tracked_entries_cannot_supply_checksum_evidence(tmp_path: Path) -> None:
    milestone = _observed_milestone()
    milestone["evidence"] = {
        "root": "runs/inbox/eth",
        "checksums": "checksums.json",
        "capsule_manifest": "capsule_manifest.json",
    }

    with pytest.raises(ValueError, match="cannot be checksum-verified locally"):
        ledger_module.verify_ledger(_ledger(tmp_path, milestone), ROOT)


def test_external_validation_requires_retained_hashes(tmp_path: Path) -> None:
    milestone = _observed_milestone()
    milestone["status"] = "development_validated_external"

    with pytest.raises(ValueError, match="requires external_hashes"):
        ledger_module.verify_ledger(_ledger(tmp_path, milestone), ROOT)


def test_non_evidence_status_requires_none_origin(tmp_path: Path) -> None:
    milestone = _observed_milestone()
    milestone["status"] = "planned"

    with pytest.raises(ValueError, match="evidence_origin must be none"):
        ledger_module.verify_ledger(_ledger(tmp_path, milestone), ROOT)


def test_shallow_clone_has_actionable_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def shallow_git(cwd: Path, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            ["git", "-C", str(cwd), *args], 0, stdout="true\n", stderr=""
        )

    monkeypatch.setattr(ledger_module, "_git", shallow_git)

    with pytest.raises(ValueError, match="git fetch --unshallow"):
        ledger_module._require_full_history(ROOT)


def test_stale_generated_document_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stale = tmp_path / "MILESTONES.md"
    stale.write_text("stale\n", encoding="utf-8")
    monkeypatch.setattr(ledger_module, "GENERATED_DOCUMENT", stale)

    with pytest.raises(ValueError, match="run make milestones-render"):
        ledger_module.verify_ledger(CONFIG, ROOT)


def test_capsule_source_commit_must_match(tmp_path: Path) -> None:
    milestone = _write_valid_capsule(tmp_path / "capsule")
    milestone["source_commit"] = "b550c46b24ed1da07deb3d4a1043d751df987f5d"

    with pytest.raises(ValueError, match="Capsule source_commit does not match"):
        ledger_module._verify_capsule_contents(milestone, tmp_path / "capsule")


def test_checksum_paths_cannot_escape_capsule(tmp_path: Path) -> None:
    root = tmp_path / "capsule"
    milestone = _write_valid_capsule(root)
    capsule = root / "capsule_manifest.json"
    capsule.write_text(
        json.dumps({"source_commit": SOURCE_COMMIT, "files": ["../outside"]}),
        encoding="utf-8",
    )
    checksums = {
        "../outside": "0" * 64,
        "README.md": hashlib.sha256((root / "README.md").read_bytes()).hexdigest(),
        "capsule_manifest.json": hashlib.sha256(capsule.read_bytes()).hexdigest(),
    }
    (root / "checksums.json").write_text(json.dumps(checksums), encoding="utf-8")

    with pytest.raises(ValueError, match="safe relative POSIX path"):
        ledger_module._verify_capsule_contents(milestone, root)


def test_checksum_manifest_cannot_cover_itself(tmp_path: Path) -> None:
    root = tmp_path / "capsule"
    milestone = _write_valid_capsule(root)
    checksums_path = root / "checksums.json"
    checksums = json.loads(checksums_path.read_text(encoding="utf-8"))
    checksums["checksums.json"] = "0" * 64
    checksums_path.write_text(json.dumps(checksums), encoding="utf-8")

    with pytest.raises(ValueError, match="must exclude itself"):
        ledger_module._verify_capsule_contents(milestone, root)


def test_checksum_coverage_must_exactly_match_capsule(tmp_path: Path) -> None:
    root = tmp_path / "capsule"
    milestone = _write_valid_capsule(root)
    checksums_path = root / "checksums.json"
    checksums = json.loads(checksums_path.read_text(encoding="utf-8"))
    del checksums["artifact.json"]
    checksums_path.write_text(json.dumps(checksums), encoding="utf-8")

    with pytest.raises(ValueError, match="does not exactly match capsule files"):
        ledger_module._verify_capsule_contents(milestone, root)
